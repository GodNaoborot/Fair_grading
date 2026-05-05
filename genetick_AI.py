import numpy as np
import random
from deap import base, creator, tools, algorithms
from scipy.stats import beta
import scipy.stats as st

# ================== CONFIGURATION ==================
L = 5  # Vector dimension
POP_SIZE = 100  # Population size
GENERATIONS = 200  # Number of generations
P_MUT = 0.25  # Mutation probability per element (1/4 elements rerolled)
CXPB = 0.8  # Crossover probability (applies to all selected pairs)
MUTPB = 0.2  # Probability of mutating an individual (enables P_MUT)
TOURNAMENT_SIZE = 3  # Tournament selection size
BETA_A, BETA_B = 2, 2  # Beta(2,2) parameters (gives values in [0,1])

# Domain bounds: each element ∈ [0,1]
X_MIN, X_MAX = 0.0, 1.0

# Random seed for reproducibility
random.seed(42)
np.random.seed(42)


# ===================================================
def f_2(x):
    return (1 - x) ** 4

def f_3(x):
    term1 = 1 / (x * np.sqrt(3 * np.pi))
    exponent = -0.5 * ((np.log(x) + np.sqrt(1.5) - np.log(3)) ** 2)
    return term1 * np.exp(exponent)

def f_4(x):
    return f_3(1 - x)

def f_5(x):
    return f_2(1 - x)
# ================== FITNESS FUNCTION ==================
def your_fitness_function(x, y):
    """
    Replace with your actual fitness logic.
    Example: maximize ||x - target||₂² + ||y - target||₂²
    """
    target = np.full(L, 0.5)  # Example target
    error = (np.linalg.norm(x - target) + np.linalg.norm(y - target)) / 2
    return 1.0 / (1.0 + error)  # Minimization converted to maximization

def calculate_prob(ratings_matrix, st_it, student_alpha=2, student_beta=2,
                                      item_alpha=2, item_beta=2):
    methods = {2: f_2,
               3:f_3,
               4:f_4,
               5:f_5,}

    n_students, n_items = ratings_matrix.shape
    students = st_it[:n_students]
    items = st_it[n_students:]

    total_product = 1.0
    for key, func in methods.items():
        mask = (ratings_matrix == key)
        if not np.any(mask):
            continue
        rows, cols = np.where(mask)  # row indices and column indices where key appears
        # Extract the corresponding v1 and v2 values (both are 1D arrays of same length)
        v1_vals = students[rows]
        v2_vals = items[cols]
        # Compute the function on these paired values
        f_vals = func(v1_vals/(v1_vals + v2_vals))  # shape = (number_of_matches,)
        # Multiply into the total product
        total_product *= np.prod(f_vals)


    return (np.prod(beta.pdf(students,student_alpha,student_beta))*
            np.prod(beta.pdf(items,item_alpha,item_beta))*total_product)

# ================== OPERATORS ==================
def init_individual():
    """Create a single individual: x and y vectors from Beta(2,2)."""
    x = np.random.beta(BETA_A, BETA_B, L).tolist()
    return x  # Concatenate into a single list


def custom_mutation(individual):
    """
    Reroll each element (x and y parts) with probability P_MUT.
    New values drawn from Beta(2,2).
    """
    # Indices for x and y parts
    x_indices = range(L)

    # Mutate
    for i in x_indices:
        if random.random() < P_MUT:
            individual[i] = np.random.beta(BETA_A, BETA_B)

    return individual,


def custom_crossover(ind1, ind2):
    """
    Element-wise mean crossover for both x and y vectors.
    """
    # Split individuals into x and y components
    x1 = ind1
    x2 = ind2

    # Compute mean
    child1_x = [(a + b) / 2 for a, b in zip(x1, x2)]

    child2_x = child1_x[:]  # Same outcome for both offspring

    # Write back to the individuals (modified in-place)
    ind1[:] = child1_x
    ind2[:] = child2_x
    return ind1, ind2


# ================== SETUP DEAP ==================
# Define fitness (maximization)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Register individual and population
toolbox.register("individual", tools.initIterate, creator.Individual, init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register operators
toolbox.register("evaluate", lambda ind: (your_fitness_function(ind[:L], ind[L:]),))
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
toolbox.register("mate", custom_crossover)
toolbox.register("mutate", custom_mutation)

# Optional: enable elitism via hall of fame
hof = tools.HallOfFame(1)


# ================== RUN ALGORITHM ==================

if __name__ == "__main__":
    # Initial population
    pop = toolbox.population(n=POP_SIZE)

    print("Starting evolution...")
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    # Run the GA
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB,
                                   ngen=GENERATIONS, stats=stats, halloffame=hof,
                                   verbose=True)

    # Extract best individual
    best = hof[0]
    best_x = best[:L]
    best_y = best[L:]
    best_fitness = best.fitness.values[0]

    print("\n=== Optimal solution found ===")
    print(f"x* = {np.array(best_x)}")
    print(f"y* = {np.array(best_y)}")
    print(f"Fitness = {best_fitness:.6f}")