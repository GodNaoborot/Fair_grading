"""
Байесовские IRT-подобные модели оценок студентов.

Доступно два варианта модели:
  - create_bipartite_bayesian_network_cond   — с «гейтом» на крайние оценки
                                                (двойка / пятёрка определяются
                                                отдельной сигмоидой)
  - create_bipartite_bayesian_network_nocond — без гейта, P(оценка) полностью
                                                задаётся f_2..f_5

Обе модели принимают параметр `ratio_kind`, задающий формулу «совпадения»
между способностью студента `s ∈ [0, 1]` и сложностью предмета `d ∈ [0, 1]`:

  - "legacy_sd" : s / (s + d)            — симметричен, исходная формула
  - "current"   : 1 - d * (1 - s)        — асимметричен, текущий дефолт
  - "sigmoid"   : sigmoid(k * (s - d))   — гладкая S-образная, параметр k

Функции f_2..f_5 нормированы по построению (f_2 + f_3 + f_4 + f_5 == 1):

  f_2(x) = (1-x)^3 / ((1-x)^3 + 10 * x^2)        # P(оценка = 2)
  f_5(x) = f_2(1 - x)                            # P(оценка = 5)
  f_3(x) = (1-x) * (1 - f_2(x) - f_5(x))         # P(оценка = 3)
  f_4(x) = f_3(1 - x)                            # P(оценка = 4)

Приоры
------
`item_alpha` и `item_beta` могут быть скаляром или 1-D массивом длины
n_items (per-item приор, например из chi_square_priors).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt


# ---------------------------------------------------------------------------
# Семейство f-функций (унормированные P(оценка))
# ---------------------------------------------------------------------------
def f_2(x):
    """Вес для оценки 2 (логистический спад при росте x)."""
    return ((1 - x) ** 3) / ((1 - x) ** 3 + 10 * x ** 2)


def f_5(x):
    """Вес для оценки 5 — симметрия к f_2."""
    return f_2(1 - x)


def f_3(x):
    """Вес для оценки 3 — остаток, смещённый в сторону низкого x."""
    return (1 - x) * (1 - f_2(x) - f_5(x))


def f_4(x):
    """Вес для оценки 4 — отражение f_3."""
    return f_3(1 - x)


# ---------------------------------------------------------------------------
# Семейство ratio (работает и с numpy, и с pytensor)
# ---------------------------------------------------------------------------
RATIO_KINDS = ("legacy_sd", "current", "sigmoid")
SIGMOID_K_DEFAULT = 5.0


def _is_pt(x):
    """True если x — переменная pytensor (используем pt.* / pm.math.* вместо np.*)."""
    return hasattr(x, "owner")


def ratio_legacy_sd(s, d, eps=1e-9):
    """Исходная формула `s / (s + d)`. Симметрична, корректна при s+d > 0."""
    return s / (s + d + eps)


def ratio_current(s, d):
    """`1 - d * (1 - s)`. Стремится к 1 в углах (s=0, d=0) и (s=1, d=1) — асимметрично."""
    return 1.0 - d * (1.0 - s)


def ratio_sigmoid(s, d, k=SIGMOID_K_DEFAULT):
    """`sigmoid(k * (s - d))`. Симметричная гладкая S-кривая; большее k = резче переход."""
    z = k * (s - d)
    if _is_pt(z):
        return pm.math.sigmoid(z)
    return 1.0 / (1.0 + np.exp(-z))


RATIO_FUNCTIONS = {
    "legacy_sd": ratio_legacy_sd,
    "current":   ratio_current,
    "sigmoid":   ratio_sigmoid,
}


def ratio(s, d, kind="current", **kw):
    """Диспетчер: вызывает выбранную ratio-функцию."""
    if kind not in RATIO_FUNCTIONS:
        raise ValueError(f"неизвестный ratio kind {kind!r}; ожидался один из {RATIO_KINDS}")
    return RATIO_FUNCTIONS[kind](s, d, **kw)


# ---------------------------------------------------------------------------
# Внутренние хелперы
# ---------------------------------------------------------------------------
def _compute_ratio_pt(a, d, ratio_kind="current", ratio_kwargs=None):
    """Считает clipped ratio для pytensor."""
    ratio_kwargs = ratio_kwargs or {}
    r = ratio(a, d, kind=ratio_kind, **ratio_kwargs)
    eps = 1e-9
    return pt.clip(r, eps, 1 - eps)


def _likelihood_probs_from_r(r):
    """Из готового ratio собирает (n_obs, 4) тензор вероятностей оценок."""
    p2 = f_2(r)
    p3 = f_3(r)
    p4 = f_4(r)
    p5 = f_5(r)
    probs = pt.stack([p2, p3, p4, p5], axis=1)
    # Перенормировка на случай небольшого float-дрейфа; сумма ≈ 1 уже.
    return probs / probs.sum(axis=1, keepdims=True)


def _setup_observed(ratings_matrix):
    """Достаёт индексы наблюдаемых ячеек и сдвигает оценки 2..5 → 0..3."""
    n_students, n_items = ratings_matrix.shape
    obs_rows, obs_cols = np.where(~np.isnan(ratings_matrix))
    obs_ratings = ratings_matrix[obs_rows, obs_cols].astype(int)
    if len(obs_ratings) == 0:
        raise ValueError("Нет наблюдаемых оценок (вся матрица NaN).")
    return n_students, n_items, obs_rows, obs_cols, obs_ratings - 2


# ---------------------------------------------------------------------------
# Модели
# ---------------------------------------------------------------------------
def create_bipartite_bayesian_network_cond(
    ratings_matrix,
    student_alpha=2, student_beta=2,
    item_alpha=2, item_beta=2,
    ratio_kind="current", ratio_kwargs=None,
    gate_lo=0.1, gate_hi=0.9, gate_temp=30.0,
    draws=1000, tune=2000, chains=4, cores=2,
    target_accept=0.95, max_tree_depth=30,
):
    """Двудольная IRT-модель с гейтом крайних оценок (2 и 5).

    Гейт теперь живёт в координатах `ratio` (а не raw a, d): сигмоиды центрированы
    на `gate_hi` (по умолчанию 0.9 — толкает к пятёрке) и `gate_lo` (0.1 —
    толкает к двойке). Это согласовано с самой f-моделью: всё работает в одних
    координатах [0, 1].

    `item_alpha`, `item_beta` могут быть скалярами или массивами длины n_items.
    `ratio_kind` — один из utils.RATIO_KINDS, `ratio_kwargs` пробрасывается.
    """
    n_students, n_items, obs_rows, obs_cols, obs_shifted = _setup_observed(ratings_matrix)

    with pm.Model() as bipartite_model:
        # Латентные способности студентов и сложности предметов
        student_ability = pm.Beta(
            "student_ability",
            alpha=student_alpha, beta=student_beta,
            shape=n_students,
        )
        item_difficulty = pm.Beta(
            "item_difficulty",
            alpha=item_alpha, beta=item_beta,
            shape=n_items,
        )

        # Выбираем s, d для всех наблюдаемых пар (студент, предмет)
        a = student_ability[obs_rows]
        d = item_difficulty[obs_cols]

        # Ratio в выбранной системе координат + базовая f-вероятность по оценкам
        r = _compute_ratio_pt(a, d, ratio_kind=ratio_kind, ratio_kwargs=ratio_kwargs)
        else_probs = _likelihood_probs_from_r(r)

        # Жёсткие распределения для крайних оценок (на случай гейта)
        const_5 = pt.as_tensor_variable([0.001, 0.001, 0.001, 0.997])
        const_2 = pt.as_tensor_variable([0.997, 0.001, 0.001, 0.001])

        # Гейт: центр на gate_hi → пятёрка, центр на gate_lo → двойка.
        # Оба порога — в координатах ratio, не в raw (a, d).
        logit_5 = gate_temp * (r - gate_hi)
        logit_2 = gate_temp * (gate_lo - r)

        p_extreme_5 = pm.math.sigmoid(logit_5)
        p_extreme_2 = pm.math.sigmoid(logit_2)

        total_extreme = p_extreme_5 + p_extreme_2
        p_5 = p_extreme_5 / (total_extreme + 1e-8)
        p_2 = p_extreme_2 / (total_extreme + 1e-8)
        p_middle = 1.0 - p_5 - p_2

        # Финальное распределение оценок: смесь экстремальных и f-вероятностей
        prob_vec = (
            p_5[:, None] * const_5[None, :]
            + p_2[:, None] * const_2[None, :]
            + p_middle[:, None] * else_probs
        )

        pm.Categorical("ratings_obs", p=prob_vec, observed=obs_shifted)

        trace = pm.sample(
            draws=draws, tune=tune, chains=chains, cores=cores,
            target_accept=target_accept, return_inferencedata=True,
            max_tree_depth=max_tree_depth,
        )

    return trace, bipartite_model


def create_bipartite_bayesian_network_nocond(
    ratings_matrix,
    student_alpha=2, student_beta=2,
    item_alpha=2, item_beta=2,
    ratio_kind="current", ratio_kwargs=None,
    draws=1000, tune=2000, chains=4, cores=2,
    target_accept=0.95,
):
    """Двудольная IRT-модель БЕЗ гейта крайних оценок.

    Все четыре вероятности оценок берутся напрямую из f_2..f_5(ratio).
    """
    n_students, n_items, obs_rows, obs_cols, obs_shifted = _setup_observed(ratings_matrix)

    with pm.Model() as bipartite_model:
        student_ability = pm.Beta(
            "student_ability",
            alpha=student_alpha, beta=student_beta,
            shape=n_students,
        )
        item_difficulty = pm.Beta(
            "item_difficulty",
            alpha=item_alpha, beta=item_beta,
            shape=n_items,
        )

        a = student_ability[obs_rows]
        d = item_difficulty[obs_cols]
        r = _compute_ratio_pt(a, d, ratio_kind=ratio_kind, ratio_kwargs=ratio_kwargs)
        probs = _likelihood_probs_from_r(r)

        pm.Categorical("ratings_obs", p=probs, observed=obs_shifted)

        trace = pm.sample(
            draws=draws, tune=tune, chains=chains, cores=cores,
            target_accept=target_accept, return_inferencedata=True,
        )

    return trace, bipartite_model


# ---------------------------------------------------------------------------
# Chi-square приоры для item_difficulty
# ---------------------------------------------------------------------------
# Эталонное распределение оценок «лёгкого» предмета (много пятёрок)
CHI_REF_EASY = np.array([0.03, 0.10, 0.26, 0.61])
# Эталонное распределение оценок «сложного» предмета (много троек)
CHI_REF_HARD = np.array([0.15, 0.57, 0.24, 0.14])
# Концентрация α+β для производного Beta-приора (как у Beta(2, 2))
CHI_PRIOR_CONCENTRATION = 4.0


def chi_square_difficulty(empirical_dist,
                          d_easy=CHI_REF_EASY,
                          d_hard=CHI_REF_HARD):
    """Оценка сложности предмета через χ²-расстояния до эталонов.

    Возвращает число из [0, 1]: 0 — выглядит как «лёгкий», 1 — как «сложный».
    """
    e = np.asarray(empirical_dist, dtype=float)
    de = np.asarray(d_easy, dtype=float)
    dh = np.asarray(d_hard, dtype=float)
    d_ez = float(np.sum((e - de) ** 2 / de))
    d_hd = float(np.sum((e - dh) ** 2 / dh))
    return d_ez / (d_ez + d_hd)


def chi_square_priors(ratings_matrix,
                      d_easy=CHI_REF_EASY,
                      d_hard=CHI_REF_HARD,
                      concentration=CHI_PRIOR_CONCENTRATION):
    """Per-item Beta-приор из chi-square оценок.

    Концентрация α+β фиксирована (по умолчанию = 4, как у Beta(2,2)) —
    сохраняется «сила» приора, дисперсия = μ(1-μ)/(s+1).

    Returns
    -------
    alpha : np.ndarray (n_items,)
    beta  : np.ndarray (n_items,)
    mu    : np.ndarray (n_items,)  — chi-square среднее
    """
    grades = np.array([2, 3, 4, 5])
    _, n_items = ratings_matrix.shape
    mu = np.zeros(n_items)
    for j in range(n_items):
        col = ratings_matrix[:, j]
        col = col[~np.isnan(col)].astype(int)
        if col.size == 0:
            mu[j] = 0.5
            continue
        counts = np.array([(col == g).sum() for g in grades], dtype=float)
        mu[j] = chi_square_difficulty(counts / counts.sum(), d_easy, d_hard)

    eps = 1e-3
    mu_clip = np.clip(mu, eps, 1 - eps)
    alpha = mu_clip * concentration
    beta = (1 - mu_clip) * concentration
    return alpha, beta, mu
