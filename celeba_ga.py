import random

import pandas as pd
import torch

DEFAULT_LATENT_CLAMP = 3.0


class Individual:
    def __init__(self, image_id, z, source, score):
        self.image_id = image_id
        self.z = z
        self.source = source
        self.score = score


def filter_exact_matches(df, selected_attrs):
    filtered = df.copy()
    for attr, value in selected_attrs.items():
        if attr in filtered.columns:
            filtered = filtered[filtered[attr] == value]
    return filtered


def compute_partial_match_scores(df, selected_attrs):
    tmp = df.copy()
    if not selected_attrs:
        tmp["match_score"] = 0
        return tmp

    score = pd.Series(0, index=tmp.index)
    for attr, value in selected_attrs.items():
        if attr in tmp.columns:
            score += (tmp[attr] == value).astype(int)
    tmp["match_score"] = score
    return tmp.sort_values("match_score", ascending=False)


def initialize_population_from_attributes(
    attr_df,
    selected_attrs,
    pop_size,
    encode_image_id_to_latent,
):
    selected_ids = []
    selected_sources = []
    selected_scores = []

    exact_df = filter_exact_matches(attr_df, selected_attrs)
    if len(exact_df) > 0:
        exact_sample = exact_df.sample(n=min(pop_size, len(exact_df)), replace=False)
        for image_id in exact_sample["image_id"].tolist():
            selected_ids.append(image_id)
            selected_sources.append("exact")
            selected_scores.append(len(selected_attrs))

    if len(selected_ids) < pop_size:
        scored_df = compute_partial_match_scores(attr_df, selected_attrs)
        for _, row in scored_df.iterrows():
            image_id = row["image_id"]
            if image_id in selected_ids:
                continue
            if row["match_score"] <= 0 and selected_attrs:
                break
            selected_ids.append(image_id)
            selected_sources.append("partial" if selected_attrs else "random")
            selected_scores.append(int(row["match_score"]))
            if len(selected_ids) == pop_size:
                break

    if len(selected_ids) < pop_size:
        remaining_df = attr_df[~attr_df["image_id"].isin(selected_ids)]
        n_missing = pop_size - len(selected_ids)
        if len(remaining_df) > 0:
            sample_n = min(n_missing, len(remaining_df))
            random_sample = remaining_df.sample(n=sample_n, replace=False)
            for image_id in random_sample["image_id"].tolist():
                selected_ids.append(image_id)
                selected_sources.append("random")
                selected_scores.append(0)

    population = []
    for image_id, source, score in zip(selected_ids, selected_sources, selected_scores):
        z = encode_image_id_to_latent(image_id)
        population.append(Individual(image_id=image_id, z=z, source=source, score=score))

    return population


def crossover(parent1, parent2):
    alpha = torch.rand(1).item()
    child = alpha * parent1 + (1.0 - alpha) * parent2
    return child


def mutate(
    z,
    mutation_std=0.15,
    clamp_value=DEFAULT_LATENT_CLAMP,
):
    out = z + mutation_std * torch.randn_like(z)
    out = torch.clamp(out, -clamp_value, clamp_value)
    return out


def sort_population_by_fitness(population, fitness_scores):
    paired = list(zip(population, fitness_scores))
    paired.sort(key=lambda x: x[1], reverse=True)
    sorted_pop = [p[0] for p in paired]
    sorted_scores = [p[1] for p in paired]
    return sorted_pop, sorted_scores


def create_next_generation(
    population,
    selected_indices,
    attr_df,
    pop_size,
    elite_size,
    mutation_std,
    encode_image_id_to_latent,
    random_injection_count=1,
    latent_clamp=DEFAULT_LATENT_CLAMP,
    project_latent=None,
):
    if not selected_indices:
        raise ValueError("Tu dois sélectionner au moins un individu.")

    fitness_scores = [0.0] * len(population)
    for rank, idx in enumerate(selected_indices):
        fitness_scores[idx] = float(len(selected_indices) - rank)

    ranked_population, _ = sort_population_by_fitness(population, fitness_scores)
    next_population = []
    elite_count = min(elite_size, len(ranked_population))
    for i in range(elite_count):
        elite = ranked_population[i]
        elite_z = elite.z.clone()
        if project_latent is not None:
            elite_z = project_latent(elite_z)
        next_population.append(
            Individual(
                image_id=elite.image_id,
                z=elite_z,
                source=f"elite:{elite.source}",
                score=elite.score,
            )
        )

    parent_pool_size = max(2, min(len(selected_indices), len(ranked_population)))
    parent_pool = ranked_population[:parent_pool_size]

    injections = min(random_injection_count, max(0, pop_size - len(next_population)))
    if injections > 0:
        sample_count = min(injections, len(attr_df))
        remaining_df = attr_df.sample(n=sample_count, replace=False)
        for image_id in remaining_df["image_id"].tolist():
            z = encode_image_id_to_latent(image_id)
            next_population.append(
                Individual(image_id=image_id, z=z, source="random_injected", score=0)
            )

    child_idx = 0
    while len(next_population) < pop_size:
        p1 = random.choice(parent_pool)
        p2 = random.choice(parent_pool)
        child_z = crossover(p1.z, p2.z)
        child_z = mutate(child_z, mutation_std=mutation_std, clamp_value=latent_clamp)
        if project_latent is not None:
            child_z = project_latent(child_z)
        next_population.append(
            Individual(
                image_id="child_gen",
                z=child_z,
                source=f"child_{child_idx}",
                score=0,
            )
        )
        child_idx += 1

    return next_population[:pop_size]
