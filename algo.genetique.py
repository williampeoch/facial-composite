import torch


# Fonction pour determiner la force de mutation en fonction de la generation, l'effet de mutation
#  diminue au fil des generations pour permettre une convergence plus fine vers une
# solution optimale.


"""// VERSION EXPONETIEL//
def get_mutation_strength(generation): #MODIFIER
    return 0.8 * torch.exp(torch.tensor(-0.2 * generation))
"""

'//VERSION SIGMOID//'
def get_mutation_strength(generation):
    initial = 0.8
    final = 0.12
    k = 0.8
    x0 = 6

    return final + (initial - final) / (1 + torch.exp(torch.tensor(k * (generation - x0))))


# Fonction pour determiner le nombre de mutations a appliquer en fonction de la generation
def get_num_mutations(generation):
    max_mut = 16
    min_mut = 6
    k = 0.5
    x0 = 6

    value = min_mut + (max_mut - min_mut) / (1 + torch.exp(torch.tensor(k * (generation - x0))))
    return int(value.item())

# Fonction pour muter un vecteur parent en ajoutant du bruit aleatoire a certaines de ses dimensions
"""//PREMIER VERSION SANS CPUPLAGE//
def mutate_vector(parent_vector, generation):
    child = parent_vector.clone()

    strength = get_mutation_strength(generation)
    num_mutations = get_num_mutations(generation)

    indices = torch.randperm(parent_vector.shape[0])[:num_mutations]
    noise = torch.randn(num_mutations) * strength

    child[indices] += noise
    return child
  """

'//DEUXIEME VERSION, COUPLAGE '
def mutate_vector(parent_vector, generation):
    child = parent_vector.clone()

    base_strength = get_mutation_strength(generation)
    num_mutations = get_num_mutations(generation)

    # pequeña variacion entre hijos
    strength = base_strength * torch.empty(1).uniform_(0.85, 1.15).item()

    indices = torch.randperm(parent_vector.shape[0])[:num_mutations]
    noise = torch.randn(num_mutations) * strength

    child[indices] += noise
    return child
#Generer des enfants comme nouvelles versions pour la choix de l'utilisateur.
def generate_children(parent_vector, generation, num_children=5):
    children = []

    for _ in range(num_children):
        child = mutate_vector(parent_vector, generation)
        children.append(child)

    return children

#Aplication de l'algorithme genetique pour iterer a travers les generations, generer des enfants,
# et permettre a l'utilisateur de choisir le meilleur vecteur pour la prochaine generation.
def genetic_algorithm(initial_vector, max_generations=10):
    parent = initial_vector.clone()

    for generation in range(1, max_generations + 1):
        print(f"\n--- Generation {generation} ---")

        children = generate_children(parent, generation, num_children=5)

        print("Option 0 = conserver le parent")
        for i in range(5):
            print(f"Option {i+1} = fils {i+1}")

        choice = int(input("Chiossisez entre (0 a 5): "))

        if 1 <= choice <= 5:
            parent = children[choice - 1]
        else:
            print("Option invalide. Le parent reste inchangé.")

    return parent


initial_vector = torch.randn(32)
print(initial_vector)
final_vector = genetic_algorithm(initial_vector, max_generations=5)
print(final_vector)
