from leap_ec.simple import ea_solve
import numpy as np
from scipy.stats import beta
import scipy.stats as st

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

# marks = generate_data(4,3)
# print(marks)


def generate_rating_matrix(n_students, m_items, seed=42):
    """
    params:
        n_studetns (int): количество студентов
        m_items (int): количество предметов
    """
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

if __name__ == '__main__':
    marks, perf, diff = generate_rating_matrix(3,3)
    print(type(perf),perf,type(diff),diff)
    print(marks)
    print(np.concat((perf,diff)))

    res = try_find_opt(marks)
    print(res)

    print(np.concat((perf,diff)))
    print(np.abs(np.array(res) - np.concat((perf,diff))))
    print(marks)

    shapes = [(5, 5), (10, 10), (20, 20)]
    for shape in shapes:
        marks, perf, diff = generate_rating_matrix(*shape)
        res = try_find_opt(marks)
        print(np.abs(np.array(res) - np.concatenate((perf, diff))))
        print(max(np.abs(np.array(res) - np.concatenate((perf, diff)))))
        print()