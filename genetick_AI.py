import numpy as np
import random
from scipy.stats import beta
import scipy.stats as st
from deap import base, creator, tools, algorithms

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

def generate_data(students_size, items_size, random_state=42):
    np.random.seed(random_state)
    return np.random.randint(low=2, high=6, size=students_size * items_size).reshape(students_size, items_size)

def calculate_prob(ratings_matrix, st_it, student_alpha=2, student_beta=2,
                   item_alpha=2, item_beta=2):
    methods = {2: f_2,
               3: f_3,
               4: f_4,
               5: f_5, }

    n_students, n_items = ratings_matrix.shape
    students = st_it[:n_students]
    items = st_it[n_students:]

    total_product = 1.0
    for key, func in methods.items():
        mask = (ratings_matrix == key)
        if not np.any(mask):
            continue
        rows, cols = np.where(mask)
        v1_vals = students[rows]
        v2_vals = items[cols]
        f_vals = func(v1_vals / (v1_vals + v2_vals))
        total_product *= np.prod(f_vals)

    return (np.prod(beta.pdf(students, student_alpha, student_beta)) *
            np.prod(beta.pdf(items, item_alpha, item_beta)) * total_product)

def generate_rating_matrix(n_students, m_items, seed=42):
    np.random.seed(seed)
    perf = st.beta(a=2, b=2).rvs(size=n_students)
    diff = st.beta(a=2, b=2).rvs(size=m_items)
    ratio = np.zeros((n_students, m_items))
    rating_matrix = np.zeros_like(ratio)

    for i, perf_i in enumerate(perf):
        for j, diff_j in enumerate(diff):
            ratio[i, j] = perf_i / (perf_i + diff_j)
            w2 = f_2(ratio[i, j])
            w3 = f_3(ratio[i, j])
            w4 = f_4(ratio[i, j])
            w5 = f_5(ratio[i, j])
            w_summ = w2 + w3 + w4 + w5
            p2 = w2 / w_summ
            p3 = w3 / w_summ
            p4 = w4 / w_summ
            p5 = w5 / w_summ
            marks = np.random.multinomial(n=1, pvals=[p2, p3, p4, p5], size=1).flatten()

            def eval_mark(marks):
                for i, mark in enumerate(marks):
                    if mark == 1:
                        return i + 2

            rating_matrix[i, j] = eval_mark(marks)

    return rating_matrix, perf, diff


# ----------------------------------------------------------------------
# 2. Genetic algorithm using DEAP (replaces ea_solve)
# ----------------------------------------------------------------------

def deap_optimize(ratings_matrix, student_alpha=2, student_beta=2,
                  item_alpha=2, item_beta=2,
                  pop_size=100, generations=200,
                  p_mut=0.25, cxpb=0.8, mutpb=0.2,
                  tournament_size=3, verbose=True):
    """
    Maximise calculate_prob(ratings_matrix, x) using a custom GA.

    Parameters
    ----------
    ratings_matrix : 2D array
        Observed ratings (2..5).
    student_alpha, student_beta, item_alpha, item_beta : float
        Beta prior parameters.
    pop_size : int
        Population size.
    generations : int
        Number of generations.
    p_mut : float
        Probability of mutating (re‑rolling) each element.
    cxpb : float
        Probability of crossover between two parents.
    mutpb : float
        Probability of mutating an individual (overall).
    tournament_size : int
        Size of tournament selection.
    verbose : bool
        Print progress statistics.

    Returns
    -------
    best_individual : list
        Flattened array of optimal parameters.
    best_fitness : float
        Corresponding fitness.
    """
    # Dimensions
    n_students, n_items = ratings_matrix.shape
    dim = n_students + n_items

    # Fixed parameters for the fitness function
    fit_kwargs = {
        'student_alpha': student_alpha,
        'student_beta': student_beta,
        'item_alpha': item_alpha,
        'item_beta': item_beta
    }

    # --- DEAP setup -------------------------------------------------
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Initialisation: each element from Beta(2,2)
    def init_individual():
        # Draw student part from Beta(2,2)
        student_part = np.random.beta(2, 2, n_students).tolist()
        # Draw item part from Beta(2,2)
        item_part = np.random.beta(2, 2, n_items).tolist()
        return student_part + item_part

    toolbox.register("individual", tools.initIterate, creator.Individual, init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        return calculate_prob(ratings_matrix, np.array(individual), **fit_kwargs),

    toolbox.register("evaluate", evaluate)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    # Custom crossover: elementwise mean of student parts and item parts separately
    def custom_crossover(ind1, ind2):
        n_stu = n_students
        # Split
        s1, i1 = ind1[:n_stu], ind1[n_stu:]
        s2, i2 = ind2[:n_stu], ind2[n_stu:]
        # Mean for both children (commutative → same result for both)
        child_s = [(a + b) / 2.0 for a, b in zip(s1, s2)]
        child_i = [(a + b) / 2.0 for a, b in zip(i1, i2)]
        # Write back
        ind1[:] = child_s + child_i
        ind2[:] = child_s + child_i
        return ind1, ind2

    # Custom mutation: reroll each element with probability p_mut from Beta(2,2)
    def custom_mutation(individual):
        for i in range(len(individual)):
            if random.random() < p_mut:
                individual[i] = np.random.beta(2, 2)
        return individual,

    toolbox.register("mate", custom_crossover)
    toolbox.register("mutate", custom_mutation)

    # Optional: Hall of Fame (elitism)
    hof = tools.HallOfFame(1)

    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    # --- Run the GA -------------------------------------------------
    random.seed(42)
    np.random.seed(42)

    pop = toolbox.population(n=pop_size)

    if verbose:
        print(f"Starting GA for {dim} parameters, pop_size={pop_size}, generations={generations}")

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb,
                                   ngen=generations, stats=stats, halloffame=hof,
                                   verbose=verbose)

    best_individual = hof[0]
    best_fitness = best_individual.fitness.values[0]

    if verbose:
        print("\n=== Optimal solution found ===")
        print(f"Student abilities: {np.array(best_individual[:n_students])}")
        print(f"Item difficulties: {np.array(best_individual[n_students:])}")
        print(f"Fitness = {best_fitness:.6f}")

    return best_individual, best_fitness


# ----------------------------------------------------------------------
# 3. Replace `try_find_opt` – now uses deap_optimize instead of ea_solve
# ----------------------------------------------------------------------

def try_find_opt(ratings_matrix, student_alpha=2, student_beta=2,
                 item_alpha=2, item_beta=2):
    """
    Run the custom GA to maximise the probability.
    Returns the best individual (flat list) – just like the old `ea_solve`.
    """
    best_ind, best_fitness = deap_optimize(
        ratings_matrix,
        student_alpha=student_alpha,
        student_beta=student_beta,
        item_alpha=item_alpha,
        item_beta=item_beta,
        pop_size=100,
        generations=200,
        p_mut=0.25,
        cxpb=0.8,
        mutpb=0.2,
        tournament_size=3,
        verbose=False
    )
    return best_ind  # mimic the return of ea_solve (which returned the solution vector)


# ----------------------------------------------------------------------
# 4. Test (your original test block)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    marks, perf, diff = generate_rating_matrix(3, 3)
    print(type(perf), perf, type(diff), diff)
    print(marks)
    print(np.concatenate((perf, diff)))

    res = try_find_opt(marks)
    print(res)

    print(np.concatenate((perf, diff)))
    print(np.abs(np.array(res) - np.concatenate((perf, diff))))
    print(marks)
    shapes = [(5, 5), (10, 10), (20, 20)]
    for shape in shapes:
        marks, perf, diff = generate_rating_matrix(*shape)
        res = try_find_opt(marks)
        print(np.abs(np.array(res) - np.concatenate((perf, diff))))
        print(max(np.abs(np.array(res) - np.concatenate((perf, diff)))))
        print()