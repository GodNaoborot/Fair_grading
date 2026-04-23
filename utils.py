import pymc as pm
import numpy as np
import pytensor.tensor as pt

def create_bipartite_bayesian_network_cond(
    ratings_matrix,
    student_alpha=2, student_beta=2,
    item_alpha=2, item_beta=2,
    draws=1000, tune=1000, chains=2, cores=2,
):
    """
    params:
        ratings_matrix (np.array): матрица оценок
        students_alpha, students_beta (float, float): параметры бета распределения у успеваемости студентов
        items_alpha, items_beta (float, float): параметры бета распределения у сложности предметов
        draws : количество сгенерированных величин
        tune: количество итераций для разогрева цепи
        chains: число цепей
        cores: число ядер
    """
    
    def f_2(x):
            return (1 - x) ** 4

    def f_3(x):
        term1 = 1 / (x * pm.math.sqrt(3 * np.pi))
        exponent = -0.5 * ((pm.math.log(x) + pm.math.sqrt(1.5) - pm.math.log(3)) ** 2)
        return term1 * pm.math.exp(exponent)

    def f_4(x):
        return f_3(1 - x)

    def f_5(x):
        return f_2(1 - x)

    n_students, n_items = ratings_matrix.shape

    obs_rows, obs_cols = np.where(~np.isnan(ratings_matrix))
    obs_ratings = ratings_matrix[obs_rows, obs_cols].astype(int)
    n_obs = len(obs_ratings)

    if n_obs == 0:
        raise ValueError("No observed ratings (all NaN).")
    
    obs_shifted = obs_ratings - 2

    with pm.Model() as bipartite_model:
        # Latent variables (vectorized)
        student_ability = pm.Beta("student_ability",
                                  alpha=student_alpha, beta=student_beta,
                                  shape=n_students)
        item_difficulty = pm.Beta("item_difficulty",
                                  alpha=item_alpha, beta=item_beta,
                                  shape=n_items)
        
    
        a = student_ability[obs_rows]
        d = item_difficulty[obs_cols]
        ratio = (a / (a + d))

        eps = 1e-12
        ratio_clip = pt.clip(ratio, eps, 1 - eps)

        p2 = f_2(ratio_clip)
        p3 = f_3(ratio_clip)
        p4 = f_4(ratio_clip)
        p5 = f_5(ratio_clip)

        else_probs_raw = pt.stack([p2, p3, p4, p5], axis=1)
        else_probs = else_probs_raw / else_probs_raw.sum(axis=1, keepdims=True)

        const_5 = pt.as_tensor_variable([0.001, 0.001, 0.001, 0.997])
        const_2 = pt.as_tensor_variable([0.997, 0.001, 0.001, 0.001])

        temp = 30.0

        logit_5 = temp * (a - (0.8 + 0.2 * d))
        logit_2 = temp * ((0.2 * d) - a)

        p_extreme_5 = pm.math.sigmoid(logit_5)
        p_extreme_2 = pm.math.sigmoid(logit_2)

        total_extreme = p_extreme_5 + p_extreme_2
        p_5 = p_extreme_5 / (total_extreme + 1e-8)
        p_2 = p_extreme_2 / (total_extreme + 1e-8)
        p_middle = 1.0 - p_5 - p_2

        prob_vec = (p_5[:, None] * const_5[None, :]) + \
           (p_2[:, None] * const_2[None, :]) + \
           (p_middle[:, None] * else_probs)

        pm.Categorical("ratings_obs", p=prob_vec, observed=obs_shifted)

        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=0.9,
            return_inferencedata=True,
            max_tree_depth=30
        )

    return trace, bipartite_model


def create_bipartite_bayesian_network_nocond(ratings_matrix, student_alpha=2, student_beta=2,
                                    item_alpha=2, item_beta=2,
                                    draws=1000, tune=1000, chains=2, cores=2):
    """
    params:
        ratings_matrix (np.array): матрица оценок
        students_alpha, students_beta (float, float): параметры бета распределения у успеваемости студентов
        items_alpha, items_beta (float, float): параметры бета распределения у сложности предметов
        draws : количество сгенерированных величин
        tune: количество итераций для разогрева цепи
        chains: число цепей
        cores: число ядер
    """
    def f_2(x):
                return (1 - x) ** 4

    def f_3(x):
        term1 = 1 / (x * pm.math.sqrt(3 * np.pi))
        exponent = -0.5 * ((pm.math.log(x) + pm.math.sqrt(1.5) - pm.math.log(3)) ** 2)
        return term1 * pm.math.exp(exponent)

    def f_4(x):
        return f_3(1 - x)

    def f_5(x):
        return f_2(1 - x)

    n_students, n_items = ratings_matrix.shape

    obs_rows, obs_cols = np.where(~np.isnan(ratings_matrix))
    obs_ratings = ratings_matrix[obs_rows, obs_cols].astype(int)
    n_obs = len(obs_ratings)

    if n_obs == 0:
        raise ValueError("No observed ratings (all NaN).")

    obs_shifted = obs_ratings - 2

    with pm.Model() as bipartite_model:
        # Latent variables (vectorized)
        student_ability = pm.Beta("student_ability",
                                alpha=student_alpha, beta=student_beta,
                                shape=n_students)
        item_difficulty = pm.Beta("item_difficulty",
                                alpha=item_alpha, beta=item_beta,
                                shape=n_items)
    
        a = student_ability[obs_rows]
        d = item_difficulty[obs_cols]
        ratio = (a / (a + d))

        eps = 1e-12
        ratio_clip = pt.clip(ratio, eps, 1 - eps)

        p2 = f_2(ratio_clip)
        p3 = f_3(ratio_clip)
        p4 = f_4(ratio_clip)
        p5 = f_5(ratio_clip)

        else_probs_raw = pt.stack([p2, p3, p4, p5], axis=1)
        else_probs = else_probs_raw / else_probs_raw.sum(axis=1, keepdims=True)

        pm.Categorical("ratings_obs", p=else_probs, observed=obs_shifted)

        # 6. Сэмплирование
        trace = pm.sample(
        draws=draws,
        tune=tune,
        chains=chains,
        cores=cores,
        target_accept=0.9,
        return_inferencedata=True
        )

    return trace, bipartite_model