import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
from urllib import error, request

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# =========================
# Config
# =========================
IMG_SIZE = 128
Z_DIM = 256
BASE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_ATTR_CSV = "celeba/list_attr_celeba.csv"
DEFAULT_IMAGE_DIR = "celeba/img_align_celeba/img_align_celeba"
DEFAULT_MODEL_PATH = "vae_128_epoch-10.pth"
DEFAULT_POP_SIZE = 6
LATENT_CLAMP = 3.0
DEFAULT_MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"
DEFAULT_MISTRAL_MODEL = "mistral-large-latest"
DEFAULT_MISTRAL_TIMEOUT = 30
APP_DIR = Path(__file__).resolve().parent
APP_ENV_PATH = str(APP_DIR / ".env.local")
PERSON_CLASS_INDEX = 15
DEFAULT_PERSON_MASK_THRESHOLD = 0.55
DEFAULT_BG_VALUE = 0.5

TFM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

SEGMENTATION_NORM = transforms.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)

# =========================
# Model
# =========================
class ConvVAE128(nn.Module):
    def __init__(self, in_channels=3, z_dim=256, base=64):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, base, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4, base * 8, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 8, base * 8, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.enc_out = base * 8 * 4 * 4
        self.fc_mu = nn.Linear(self.enc_out, z_dim)
        self.fc_logvar = nn.Linear(self.enc_out, z_dim)

        self.fc_dec = nn.Linear(z_dim, self.enc_out)

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base * 8, base * 8, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base * 8, base * 4, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base, in_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def encode(self, x):
        h = self.enc(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z).view(z.size(0), -1, 4, 4)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


@dataclass
class Individual:
    image_id: str
    z: torch.Tensor
    source: str
    score: int


# =========================
# Utilities
# =========================
def denorm(x: torch.Tensor) -> torch.Tensor:
    return (x * 0.5 + 0.5).clamp(0, 1)


def tensor_to_display_image(img_tensor: torch.Tensor):
    img = denorm(img_tensor.detach().cpu()).permute(1, 2, 0).numpy()
    return img


@st.cache_data(show_spinner=False)
def load_attr_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    if "image_id" not in df.columns:
        raise ValueError("Le CSV doit contenir une colonne 'image_id'.")
    return df


@st.cache_resource(show_spinner=False)
def load_model(model_path: str, z_dim: int, base: int, device_str: str):
    device = torch.device(device_str)
    model = ConvVAE128(in_channels=3, z_dim=z_dim, base=base).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def load_person_segmentation_model(device_str: str):
    device = torch.device(device_str)
    try:
        from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

        seg_model = models.segmentation.deeplabv3_resnet50(
            weights=DeepLabV3_ResNet50_Weights.DEFAULT
        ).to(device)
    except Exception:
        seg_model = models.segmentation.deeplabv3_resnet50(pretrained=True).to(device)
    seg_model.eval()
    return seg_model


def load_image_from_id(image_dir: str, image_id: str, device: torch.device) -> torch.Tensor:
    img_path = os.path.join(image_dir, image_id)
    img = Image.open(img_path).convert("RGB")
    return TFM(img).unsqueeze(0).to(device)


def encode_image_id_to_latent(model, image_dir: str, image_id: str, device: torch.device) -> torch.Tensor:
    image_tensor = load_image_from_id(image_dir, image_id, device)
    with torch.no_grad():
        mu, _ = model.encode(image_tensor)
    return mu.squeeze(0).detach().cpu()


def encode_tensor_to_latent(model, image_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        mu, _ = model.encode(image_tensor.unsqueeze(0).to(device))
    return mu.squeeze(0).detach().cpu()


def decode_latent_tensor(model, z_cpu: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        z = z_cpu.unsqueeze(0).to(device)
        recon = model.decode(z).squeeze(0).detach().cpu()
    return recon


def remove_background_from_image_tensor(
    img_tensor: torch.Tensor,
    seg_model,
    seg_device: torch.device,
    threshold: float = DEFAULT_PERSON_MASK_THRESHOLD,
    bg_value: float = DEFAULT_BG_VALUE,
) -> torch.Tensor:
    image_01 = denorm(img_tensor).to(seg_device)
    seg_input = SEGMENTATION_NORM(image_01).unsqueeze(0)
    with torch.no_grad():
        seg_out = seg_model(seg_input)["out"][0]
        probs = torch.softmax(seg_out, dim=0)
        person_mask = (probs[PERSON_CLASS_INDEX] > threshold).float().unsqueeze(0)

    background = torch.full_like(image_01, bg_value)
    fg_img = image_01 * person_mask + background * (1.0 - person_mask)
    return (fg_img * 2.0 - 1.0).detach().cpu()


def remove_background_from_latent(
    model,
    seg_model,
    z_cpu: torch.Tensor,
    vae_device: torch.device,
    seg_device: torch.device,
    threshold: float = DEFAULT_PERSON_MASK_THRESHOLD,
    bg_value: float = DEFAULT_BG_VALUE,
) -> torch.Tensor:
    decoded = decode_latent_tensor(model, z_cpu, vae_device)
    bg_removed = remove_background_from_image_tensor(
        decoded,
        seg_model=seg_model,
        seg_device=seg_device,
        threshold=threshold,
        bg_value=bg_value,
    )
    return encode_tensor_to_latent(model, bg_removed, vae_device)


def filter_exact_matches(df: pd.DataFrame, selected_attrs: Dict[str, int]) -> pd.DataFrame:
    filtered = df.copy()
    for attr, value in selected_attrs.items():
        if attr in filtered.columns:
            filtered = filtered[filtered[attr] == value]
    return filtered


def compute_partial_match_scores(df: pd.DataFrame, selected_attrs: Dict[str, int]) -> pd.DataFrame:
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


def build_selected_attrs_from_form(form_values: Dict[str, str]) -> Dict[str, int]:
    selected_attrs = {}
    for attr, mode in form_values.items():
        if mode == "Présent":
            selected_attrs[attr] = 1
        elif mode == "Absent":
            selected_attrs[attr] = -1
    return selected_attrs


def parse_env_file(env_path: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    if not os.path.exists(env_path):
        return parsed

    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            parsed[key] = value

    return parsed


def load_app_config(env_path: str = APP_ENV_PATH) -> Dict[str, str]:
    config = {
        "mistral_api_key": "",
        "mistral_model": DEFAULT_MISTRAL_MODEL,
        "mistral_endpoint": DEFAULT_MISTRAL_ENDPOINT,
        "mistral_timeout_seconds": DEFAULT_MISTRAL_TIMEOUT,
    }

    file_env = parse_env_file(env_path)
    file_api_key = file_env.get("MISTRAL_API_KEY", "").strip()
    file_model = file_env.get("MISTRAL_MODEL", "").strip()
    file_endpoint = file_env.get("MISTRAL_ENDPOINT", "").strip()
    file_timeout = file_env.get("MISTRAL_TIMEOUT_SECONDS", "").strip()

    if file_api_key:
        config["mistral_api_key"] = file_api_key
    if file_model:
        config["mistral_model"] = file_model
    if file_endpoint:
        config["mistral_endpoint"] = file_endpoint
    if file_timeout:
        try:
            config["mistral_timeout_seconds"] = max(5, int(file_timeout))
        except ValueError:
            config["mistral_timeout_seconds"] = DEFAULT_MISTRAL_TIMEOUT

    env_key = os.getenv("MISTRAL_API_KEY", "").strip()
    env_model = os.getenv("MISTRAL_MODEL", "").strip()
    env_endpoint = os.getenv("MISTRAL_ENDPOINT", "").strip()
    env_timeout = os.getenv("MISTRAL_TIMEOUT_SECONDS", "").strip()

    if env_key:
        config["mistral_api_key"] = env_key
    if env_model:
        config["mistral_model"] = env_model
    if env_endpoint:
        config["mistral_endpoint"] = env_endpoint
    if env_timeout:
        try:
            config["mistral_timeout_seconds"] = max(5, int(env_timeout))
        except ValueError:
            config["mistral_timeout_seconds"] = DEFAULT_MISTRAL_TIMEOUT

    return config


def normalize_attr_name(attr_name: str) -> str:
    return attr_name.strip().lower().replace("-", "_").replace(" ", "_")


def normalize_attr_mode(raw_value) -> str:
    if raw_value is None:
        return "Indifférent"

    value = str(raw_value).strip().lower()
    present_values = {"present", "présent", "1", "+1", "true", "yes", "oui"}
    absent_values = {"absent", "-1", "false", "no", "non"}

    if value in present_values:
        return "Présent"
    if value in absent_values:
        return "Absent"
    return "Indifférent"


def extract_first_json_object(text: str) -> Dict:
    candidate = text.strip()

    if candidate.startswith("```"):
        lines = [line for line in candidate.splitlines() if not line.strip().startswith("```")]
        candidate = "\n".join(lines).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Réponse Mistral invalide: JSON introuvable.")

    return json.loads(candidate[start: end + 1])


def infer_form_values_from_prompt(
    prompt_text: str,
    available_attrs: List[str],
    api_key: str,
    model_name: str = DEFAULT_MISTRAL_MODEL,
    endpoint: str = DEFAULT_MISTRAL_ENDPOINT,
    timeout_seconds: int = DEFAULT_MISTRAL_TIMEOUT,
) -> Dict[str, str]:
    if not prompt_text.strip():
        raise ValueError("Le prompt est vide.")
    if not api_key.strip():
        raise ValueError("La clé API Mistral est vide.")
    if not available_attrs:
        return {}

    system_prompt = (
        "Tu reçois une description de visage et une liste d'attributs CelebA autorisés. "
        "Retourne uniquement un objet JSON. "
        "Format attendu: {\"attributes\": {\"Nom_Attribut\": \"Présent|Absent|Indifférent\"}}. "
        "N'invente pas d'attributs hors liste. "
        "Mets Indifférent quand l'information n'est pas explicite."
    )
    user_prompt = (
        f"Description:\n{prompt_text}\n\n"
        f"Attributs autorisés:\n{', '.join(available_attrs)}\n\n"
        "Réponds en JSON uniquement."
    )
    payload = {
        "model": model_name,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    req = request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            raw_response = response.read().decode("utf-8")
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Erreur API Mistral ({exc.code}): {details}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Impossible de joindre Mistral: {exc.reason}") from exc

    parsed_response = json.loads(raw_response)
    choices = parsed_response.get("choices", [])
    if not choices:
        raise RuntimeError("Réponse Mistral sans 'choices'.")

    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text_part = item.get("text")
                if text_part:
                    parts.append(str(text_part))
            elif isinstance(item, str):
                parts.append(item)
        content = "\n".join(parts)

    output = extract_first_json_object(str(content))
    attributes_payload = output.get("attributes", output)
    if not isinstance(attributes_payload, dict):
        raise RuntimeError("JSON Mistral invalide: clé 'attributes' absente ou invalide.")

    mapped = {attr: "Indifférent" for attr in available_attrs}
    allowed_map = {normalize_attr_name(attr): attr for attr in available_attrs}
    for raw_attr_name, raw_mode in attributes_payload.items():
        normalized_name = normalize_attr_name(str(raw_attr_name))
        attr_name = allowed_map.get(normalized_name)
        if not attr_name:
            continue
        mapped[attr_name] = normalize_attr_mode(raw_mode)

    return mapped


def initialize_population_from_attributes(
    model,
    attr_df: pd.DataFrame,
    image_dir: str,
    selected_attrs: Dict[str, int],
    device: torch.device,
    pop_size: int = DEFAULT_POP_SIZE,
) -> List[Individual]:
    selected_ids: List[str] = []
    selected_sources: List[str] = []
    selected_scores: List[int] = []

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

    population: List[Individual] = []
    for image_id, source, score in zip(selected_ids, selected_sources, selected_scores):
        z = encode_image_id_to_latent(model, image_dir, image_id, device)
        population.append(Individual(image_id=image_id, z=z, source=source, score=score))

    return population


# =========================
# Genetic algorithm
# =========================
def crossover(parent1: torch.Tensor, parent2: torch.Tensor) -> torch.Tensor:
    alpha = torch.rand(1).item()
    child = alpha * parent1 + (1.0 - alpha) * parent2
    return child


def mutate(z: torch.Tensor, mutation_std: float = 0.15, clamp_value: float = LATENT_CLAMP) -> torch.Tensor:
    out = z + mutation_std * torch.randn_like(z)
    out = torch.clamp(out, -clamp_value, clamp_value)
    return out


def project_latent_to_manifold(
    model,
    z_cpu: torch.Tensor,
    device: torch.device,
    blend: float,
    steps: int,
) -> torch.Tensor:
    z = z_cpu.unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(max(1, steps)):
            recon = model.decode(z)
            mu, _ = model.encode(recon)
            z = (1.0 - blend) * z + blend * mu
    return z.squeeze(0).detach().cpu()


def sort_population_by_fitness(population: List[Individual], fitness_scores: List[float]):
    paired = list(zip(population, fitness_scores))
    paired.sort(key=lambda x: x[1], reverse=True)
    sorted_pop = [p[0] for p in paired]
    sorted_scores = [p[1] for p in paired]
    return sorted_pop, sorted_scores


def create_next_generation(
    population: List[Individual],
    selected_indices: List[int],
    attr_df: pd.DataFrame,
    model,
    image_dir: str,
    device: torch.device,
    pop_size: int,
    elite_size: int,
    mutation_std: float,
    random_injection_count: int = 1,
    latent_clamp: float = LATENT_CLAMP,
    projection_blend: float = 0.65,
    projection_steps: int = 1,
) -> List[Individual]:
    if not selected_indices:
        raise ValueError("Tu dois sélectionner au moins un individu.")

    fitness_scores = [0.0] * len(population)
    for rank, idx in enumerate(selected_indices):
        fitness_scores[idx] = float(len(selected_indices) - rank)

    population, fitness_scores = sort_population_by_fitness(population, fitness_scores)
    next_population: List[Individual] = []
    elite_count = min(elite_size, len(population))
    for i in range(elite_count):
        elite = population[i]
        elite_z = elite.z.clone()
        if projection_blend > 0:
            elite_z = project_latent_to_manifold(
                model=model,
                z_cpu=elite_z,
                device=device,
                blend=projection_blend,
                steps=projection_steps,
            )
        next_population.append(
            Individual(
                image_id=elite.image_id,
                z=elite_z,
                source=f"elite:{elite.source}",
                score=elite.score,
            )
        )

    parent_pool_size = max(2, min(len(selected_indices), len(population)))
    parent_pool = population[:parent_pool_size]

    injections = min(random_injection_count, max(0, pop_size - len(next_population)))
    if injections > 0:
        remaining_df = attr_df.sample(n=injections, replace=False)
        for image_id in remaining_df["image_id"].tolist():
            z = encode_image_id_to_latent(model, image_dir, image_id, device)
            next_population.append(
                Individual(image_id=image_id, z=z, source="random_injected", score=0)
            )

    child_idx = 0
    while len(next_population) < pop_size:
        p1 = random.choice(parent_pool)
        p2 = random.choice(parent_pool)
        child_z = crossover(p1.z, p2.z)
        child_z = mutate(child_z, mutation_std=mutation_std, clamp_value=latent_clamp)
        if projection_blend > 0:
            child_z = project_latent_to_manifold(
                model=model,
                z_cpu=child_z,
                device=device,
                blend=projection_blend,
                steps=projection_steps,
            )
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


# =========================
# Session state
# =========================
def init_session_state():
    defaults = {
        "population": None,
        "generation": 0,
        "selected_attrs": {},
        "last_error": None,
        "selection_order": [],
        "clicked_index": None,
        "auto_form_values": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =========================
# UI rendering
# =========================


def render_population(model, population: List[Individual], device: torch.device):
    if not population:
        return None

    st.subheader(f"Population courante — génération {st.session_state.generation}")
    decoded_images = [decode_latent_tensor(model, ind.z, device) for ind in population]

    cols_per_row = 3
    chosen_idx = None
    for row_start in range(0, len(population), cols_per_row):
        cols = st.columns(cols_per_row)
        row_inds = population[row_start: row_start + cols_per_row]
        row_imgs = decoded_images[row_start: row_start + cols_per_row]

        for col_idx, (col, individual, img) in enumerate(zip(cols, row_inds, row_imgs)):
            absolute_idx = row_start + col_idx
            is_selected = st.session_state.clicked_index == absolute_idx
            with col:
                st.image(tensor_to_display_image(img), use_container_width=True)
                if st.button(
                    "Choisir ce parent",
                    key=f"choose_parent_g{st.session_state.generation}_i{absolute_idx}",
                    use_container_width=True,
                    type="primary" if is_selected else "secondary",
                ):
                    chosen_idx = absolute_idx

    return chosen_idx


def reset_session_state():
    st.session_state.population = None
    st.session_state.generation = 0
    st.session_state.selected_attrs = {}
    st.session_state.last_error = None
    st.session_state.selection_order = []
    st.session_state.clicked_index = None
    st.session_state.auto_form_values = {}
    for key in list(st.session_state.keys()):
        if key.startswith("attr_"):
            del st.session_state[key]


def advance_generation_from_current_selection(
    population: List[Individual],
    attr_df: pd.DataFrame,
    model,
    image_dir: str,
    device: torch.device,
    pop_size: int,
    elite_size: int,
    mutation_std: float,
    random_injection_count: int,
    latent_clamp: float,
    projection_blend: float,
    projection_steps: int,
    remove_parent_background: bool,
    person_mask_threshold: float,
    seg_model=None,
):
    if st.session_state.clicked_index is None:
        raise ValueError("Tu dois cliquer sur une image avant de générer la suite.")

    selected_idx = st.session_state.clicked_index
    working_population = population
    if remove_parent_background and seg_model is not None:
        selected_parent = population[selected_idx]
        cleaned_latent = remove_background_from_latent(
            model=model,
            seg_model=seg_model,
            z_cpu=selected_parent.z,
            vae_device=device,
            seg_device=device,
            threshold=person_mask_threshold,
        )
        working_population = list(population)
        working_population[selected_idx] = Individual(
            image_id=selected_parent.image_id,
            z=cleaned_latent,
            source=f"{selected_parent.source}+bg_removed",
            score=selected_parent.score,
        )

    next_population = create_next_generation(
        population=working_population,
        selected_indices=[selected_idx],
        attr_df=attr_df,
        model=model,
        image_dir=image_dir,
        device=device,
        pop_size=pop_size,
        elite_size=elite_size,
        mutation_std=mutation_std,
        random_injection_count=random_injection_count,
        latent_clamp=latent_clamp,
        projection_blend=projection_blend,
        projection_steps=projection_steps,
    )
    st.session_state.population = next_population
    st.session_state.generation += 1
    st.session_state.selection_order = []
    st.session_state.clicked_index = None


# =========================
# Main
# =========================
def main():
    st.set_page_config(page_title="CelebA Portrait Robot GA", layout="wide")
    init_session_state()

    try:
        app_config = load_app_config(APP_ENV_PATH)
    except Exception as exc:
        st.warning(f"Impossible de lire {APP_ENV_PATH}: {exc}")
        app_config = {
            "mistral_api_key": "",
            "mistral_model": DEFAULT_MISTRAL_MODEL,
            "mistral_endpoint": DEFAULT_MISTRAL_ENDPOINT,
            "mistral_timeout_seconds": DEFAULT_MISTRAL_TIMEOUT,
        }

    for key, value in app_config.items():
        if key not in st.session_state:
            st.session_state[key] = value
    st.session_state["mistral_api_key"] = app_config.get("mistral_api_key", "")

    st.title("Portrait robot")

    with st.sidebar:
        st.header("Configuration")
        model_path = DEFAULT_MODEL_PATH
        attr_csv_path = DEFAULT_ATTR_CSV
        image_dir = DEFAULT_IMAGE_DIR

        pop_size = st.slider("Taille de population", min_value=4, max_value=12, value=6, step=1)
        elite_size = st.slider("Nombre d'élites", min_value=1, max_value=4, value=2, step=1)
        mutation_std = st.slider("Force de mutation", min_value=0.01, max_value=1.0, value=0.15, step=0.01)
        random_injection_count = st.slider("Injections aléatoires / génération", min_value=0, max_value=3, value=1, step=1)
        st.subheader("Anti-bruit")
        latent_clamp = st.slider("Clamp latent", min_value=1.5, max_value=5.0, value=2.8, step=0.1)
        projection_blend = st.slider("Projection anti-bruit", min_value=0.0, max_value=1.0, value=0.65, step=0.05)
        projection_steps = st.slider("Passes de projection", min_value=1, max_value=3, value=1, step=1)
        remove_parent_background = st.checkbox(
            "Retirer le background du parent sélectionné",
            value=True,
        )
        person_mask_threshold = st.slider(
            "Seuil segmentation personne",
            min_value=0.30,
            max_value=0.90,
            value=DEFAULT_PERSON_MASK_THRESHOLD,
            step=0.01,
        )

        mistral_api_key = str(st.session_state.get("mistral_api_key", "")).strip()
        mistral_model = str(st.session_state.get("mistral_model", DEFAULT_MISTRAL_MODEL))
        mistral_endpoint = str(st.session_state.get("mistral_endpoint", DEFAULT_MISTRAL_ENDPOINT))
        mistral_timeout_seconds = int(st.session_state.get("mistral_timeout_seconds", DEFAULT_MISTRAL_TIMEOUT))

        if st.button("Réinitialiser la session"):
            reset_session_state()
            st.rerun()

    missing = []
    if not os.path.exists(attr_csv_path):
        missing.append(f"CSV introuvable: {attr_csv_path}")
    if not os.path.exists(model_path):
        missing.append(f"Modèle introuvable: {model_path}")
    if not os.path.isdir(image_dir):
        missing.append(f"Dossier images introuvable: {image_dir}")

    if missing:
        st.error("\n".join(missing))
        st.stop()

    try:
        attr_df = load_attr_df(attr_csv_path)
        model = load_model(model_path, Z_DIM, BASE, str(DEVICE))
    except Exception as exc:
        st.exception(exc)
        st.stop()

    seg_model = None
    if remove_parent_background:
        try:
            seg_model = load_person_segmentation_model(str(DEVICE))
        except Exception as exc:
            st.warning(f"Modèle de segmentation indisponible: {exc}")

    available_form_attrs = [c for c in attr_df.columns if c != "image_id"]
    if not available_form_attrs:
        st.error("Aucun attribut exploitable trouvé dans le CSV (hors colonne image_id).")
        st.stop()

    st.subheader("1. Décrire le visage cible")
    with st.form("prompt_form"):
        face_description_prompt = st.text_input(
            "Description du visage",
            key="face_description_prompt",
            placeholder="Ex: femme jeune, cheveux blonds, lunettes, sourire léger, pas de barbe",
            help="Appuie sur Entrée pour lancer la génération.",
        )
        generate_from_prompt = st.form_submit_button("Générer les visages", type="primary")

    if generate_from_prompt:
        if not face_description_prompt.strip():
            st.warning("Ajoute d'abord une description du visage.")
        elif not mistral_api_key.strip():
            st.warning("Ajoute la clé API Mistral dans `.env.local`.")
        else:
            with st.spinner("Génération en cours..."):
                try:
                    auto_form_values = infer_form_values_from_prompt(
                        prompt_text=face_description_prompt,
                        available_attrs=available_form_attrs,
                        api_key=mistral_api_key,
                        model_name=mistral_model,
                        endpoint=mistral_endpoint,
                        timeout_seconds=int(mistral_timeout_seconds),
                    )
                    st.session_state.auto_form_values = auto_form_values
                    selected_attrs = build_selected_attrs_from_form(auto_form_values)
                    st.session_state.selected_attrs = selected_attrs

                    population = initialize_population_from_attributes(
                        model=model,
                        attr_df=attr_df,
                        image_dir=image_dir,
                        selected_attrs=selected_attrs,
                        device=DEVICE,
                        pop_size=pop_size,
                    )
                    st.session_state.population = population
                    st.session_state.generation = 0
                    st.session_state.selection_order = []
                    st.session_state.clicked_index = None
                    st.success("Population initiale générée automatiquement à partir du prompt.")
                except Exception as exc:
                    st.session_state.last_error = str(exc)
                    st.exception(exc)

    if st.session_state.auto_form_values:
        auto_selected = {
            k: v for k, v in st.session_state.auto_form_values.items() if v != "Indifférent"
        }
        if auto_selected:
            st.caption(f"Attributs détectés automatiquement: {auto_selected}")
        else:
            st.caption("Aucun attribut explicite détecté dans le prompt (tout en 'Indifférent').")

    population = st.session_state.population
    if population:
        clicked_idx = render_population(model, population, DEVICE)
        if clicked_idx is not None:
            try:
                st.session_state.clicked_index = clicked_idx
                st.session_state.selection_order = [clicked_idx]
                advance_generation_from_current_selection(
                    population=population,
                    attr_df=attr_df,
                    model=model,
                    image_dir=image_dir,
                    device=DEVICE,
                    pop_size=pop_size,
                    elite_size=elite_size,
                    mutation_std=mutation_std,
                    random_injection_count=random_injection_count,
                    latent_clamp=latent_clamp,
                    projection_blend=projection_blend,
                    projection_steps=projection_steps,
                    remove_parent_background=remove_parent_background,
                    person_mask_threshold=person_mask_threshold,
                    seg_model=seg_model,
                )
                st.rerun()
            except Exception as exc:
                st.exception(exc)

        st.subheader("2. Sélection humaine")
        st.info("Choisis un parent avec le bouton 'Choisir ce parent' sous une image pour générer automatiquement la génération suivante.")

        if st.button("Regénérer une population initiale avec les mêmes attributs"):
            try:
                population = initialize_population_from_attributes(
                    model=model,
                    attr_df=attr_df,
                    image_dir=image_dir,
                    selected_attrs=st.session_state.selected_attrs,
                    device=DEVICE,
                    pop_size=pop_size,
                )
                st.session_state.population = population
                st.session_state.generation = 0
                st.session_state.selection_order = []
                st.session_state.clicked_index = None
                st.rerun()
            except Exception as exc:
                st.exception(exc)

        with st.expander("Détails techniques"):
            st.markdown(
                f"""
- Device: `{DEVICE}`
- Population size: `{pop_size}`
- Elite size: `{elite_size}`
- Mutation std: `{mutation_std}`
- Latent clamp: `{latent_clamp}`
- Projection blend: `{projection_blend}`
- Projection steps: `{projection_steps}`
- Remove parent background: `{remove_parent_background}`
- Person mask threshold: `{person_mask_threshold}`
- Random injections: `{random_injection_count}`
- Generation: `{st.session_state.generation}`
                """
            )
    else:
        st.warning("Aucune population encore générée.")


if __name__ == "__main__":
    main()
