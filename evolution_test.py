from leap_ec.simple import ea_solve
import numpy as np
from scipy.stats import beta

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

def f(x):
    """A real-valued function to be optimized."""
    x = np.array(x)
    return sum((x)-0.5)**2

def generate_data(students_size, items_size, random_state=42):
    np.random.seed(random_state)
    return np.random.randint(low=2, high=6, size=students_size * items_size).reshape(students_size, items_size)

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

def try_find_opt(ratings_matrix, student_alpha=2, student_beta=2,
                                      item_alpha=2, item_beta=2):
    result = ea_solve(lambda x: calculate_prob(ratings_matrix, x,
                                               student_alpha=student_alpha, student_beta=student_beta,
                                                item_alpha=item_alpha, item_beta=item_beta)
                      , bounds=[(0, 1) for _ in range(np.sum(ratings_matrix.shape))], maximize=True)
    return result

marks = generate_data(4,3)
print(marks)

res = try_find_opt(marks)
print(res)