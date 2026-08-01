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
  - "current"   : 1 - d * (1 - s)        — асимметричен
  - "sigmoid"   : sigmoid(k * (s - d))   — гладкая S-образная, параметр k;
                                           текущий дефолт

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
    """P(оценка = 2). Убывает от 1 при x=0 до 0 при x=1."""
    return ((1 - x) ** 3) / ((1 - x) ** 3 + 10 * x ** 2)


def f_5(x):
    """P(оценка = 5). Отражение f_2 относительно x = 1/2."""
    return f_2(1 - x)


def f_3(x):
    """P(оценка = 3). Остаток после f_2 и f_5, смещённый к низким x."""
    return (1 - x) * (1 - f_2(x) - f_5(x))


def f_4(x):
    """P(оценка = 4). Отражение f_3 относительно x = 1/2."""
    return f_3(1 - x)


# ---------------------------------------------------------------------------
# Семейство ratio (работает и с numpy, и с pytensor)
# ---------------------------------------------------------------------------
RATIO_KINDS = ("legacy_sd", "current", "sigmoid")
SIGMOID_K_DEFAULT = 5.0


def _is_pt(x):
    """True если x — символьная переменная pytensor, а не numpy-массив."""
    return hasattr(x, "owner")


def ratio_legacy_sd(s, d, eps=1e-9):
    """`s / (s + d)`. Симметрична по s ↔ d; при s = d даёт 1/2."""
    return s / (s + d + eps)


def ratio_current(s, d):
    """`1 - d * (1 - s)`. Асимметрична: равна 1 и при d=0, и при s=1."""
    return 1.0 - d * (1.0 - s)


def ratio_sigmoid(s, d, k=SIGMOID_K_DEFAULT):
    """`sigmoid(k * (s - d))`. Симметрична; большее k — резче переход."""
    z = k * (s - d)
    if _is_pt(z):
        return pm.math.sigmoid(z)
    return 1.0 / (1.0 + np.exp(-z))


RATIO_FUNCTIONS = {
    "legacy_sd": ratio_legacy_sd,
    "current":   ratio_current,
    "sigmoid":   ratio_sigmoid,
}


def ratio(s, d, kind="sigmoid", **kw):
    """Диспетчер: вызывает выбранную ratio-функцию."""
    if kind not in RATIO_FUNCTIONS:
        raise ValueError(f"неизвестный ratio kind {kind!r}; ожидался один из {RATIO_KINDS}")
    return RATIO_FUNCTIONS[kind](s, d, **kw)


# ---------------------------------------------------------------------------
# Внутренние хелперы
# ---------------------------------------------------------------------------
def _compute_ratio_pt(a, d, ratio_kind="sigmoid", ratio_kwargs=None):
    """Ratio для pytensor, обрезанный в (0, 1) — f_i не определены на границах."""
    ratio_kwargs = ratio_kwargs or {}
    r = ratio(a, d, kind=ratio_kind, **ratio_kwargs)
    eps = 1e-9
    return pt.clip(r, eps, 1 - eps)


def _likelihood_probs_from_r(r):
    """Из ratio собирает (n_obs, 4) тензор вероятностей оценок."""
    probs = pt.stack([f_2(r), f_3(r), f_4(r), f_5(r)], axis=1)
    # Сумма равна 1 по построению; делим только чтобы убрать float-дрейф.
    return probs / probs.sum(axis=1, keepdims=True)


GRADES = np.array([2, 3, 4, 5])

# Вырожденные распределения, к которым тянет гейт модели `cond`
GATE_CONST_2 = np.array([0.997, 0.001, 0.001, 0.001])
GATE_CONST_5 = np.array([0.001, 0.001, 0.001, 0.997])

# Пороги гейта в координатах ratio. Сдвинуты внутрь относительно краёв
# намеренно: при 0.1/0.9 гейт срабатывает только там, где f-функции и без
# него дают 0.997, то есть `cond` становится неотличима от `nocond`.
# При 0.3/0.7 гейт влияет на середину шкалы — ради чего он и нужен.
GATE_LO_DEFAULT = 0.3
GATE_HI_DEFAULT = 0.7
GATE_TEMP_DEFAULT = 15.0


def grade_probs(s, d, ratio_kind="sigmoid", ratio_kwargs=None,
                gate=False, gate_lo=GATE_LO_DEFAULT, gate_hi=GATE_HI_DEFAULT,
                gate_temp=GATE_TEMP_DEFAULT):
    """Вероятности оценок (2, 3, 4, 5) для numpy-входов.

    Повторяет правдоподобие моделей: `gate=False` соответствует `nocond`,
    `gate=True` — `cond` с теми же порогами. Нужна для апостериорных
    предсказаний, где выборки s и d уже получены из трассы.

    `s` и `d` broadcast'ятся друг с другом; результат имеет форму
    `np.broadcast(s, d).shape + (4,)`.
    """
    ratio_kwargs = ratio_kwargs or {}
    r = ratio(np.asarray(s, dtype=float), np.asarray(d, dtype=float),
              kind=ratio_kind, **ratio_kwargs)
    eps = 1e-9
    r = np.clip(r, eps, 1 - eps)

    probs = np.stack([f_2(r), f_3(r), f_4(r), f_5(r)], axis=-1)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    if not gate:
        return probs

    p_extreme_5 = 1.0 / (1.0 + np.exp(-gate_temp * (r - gate_hi)))
    p_extreme_2 = 1.0 / (1.0 + np.exp(-gate_temp * (gate_lo - r)))
    # См. комментарий в create_bipartite_bayesian_network_cond: нормировать
    # сигмоиды друг на друга нельзя, иначе p_middle зануляется.
    overflow = np.maximum(p_extreme_5 + p_extreme_2, 1.0)
    p_5 = p_extreme_5 / overflow
    p_2 = p_extreme_2 / overflow
    p_middle = 1.0 - p_5 - p_2

    return (p_5[..., None] * GATE_CONST_5
            + p_2[..., None] * GATE_CONST_2
            + p_middle[..., None] * probs)


def expected_grade(s, d, **kw):
    """Ожидаемая оценка E[оценка | s, d]. Принимает те же kwargs, что grade_probs."""
    return grade_probs(s, d, **kw) @ GRADES


def _setup_observed(ratings_matrix):
    """Достаёт индексы наблюдаемых ячеек и сдвигает оценки 2..5 → 0..3."""
    n_students, n_items = ratings_matrix.shape
    obs_rows, obs_cols = np.where(~np.isnan(ratings_matrix))
    obs_ratings = ratings_matrix[obs_rows, obs_cols].astype(int)
    if len(obs_ratings) == 0:
        raise ValueError("Нет наблюдаемых оценок (вся матрица NaN).")
    return n_students, n_items, obs_rows, obs_cols, obs_ratings - 2


# ---------------------------------------------------------------------------
# Построение моделей (без сэмплирования)
# ---------------------------------------------------------------------------
def _add_latents(ratings_matrix, student_alpha, student_beta, item_alpha, item_beta):
    """Общая часть обеих моделей: латентные s, d и индексация наблюдений."""
    n_students, n_items, obs_rows, obs_cols, obs_shifted = _setup_observed(ratings_matrix)

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
    return student_ability[obs_rows], item_difficulty[obs_cols], obs_shifted


def build_model_cond(
    ratings_matrix,
    student_alpha=2, student_beta=2,
    item_alpha=2, item_beta=2,
    ratio_kind="sigmoid", ratio_kwargs=None,
    gate_lo=GATE_LO_DEFAULT, gate_hi=GATE_HI_DEFAULT, gate_temp=GATE_TEMP_DEFAULT,
):
    """Двудольная IRT-модель с гейтом крайних оценок (2 и 5).

    Гейт задан в координатах `ratio`: сигмоиды центрированы на `gate_hi`
    (толкает к пятёрке) и `gate_lo` (толкает к двойке), крутизна — `gate_temp`.

    `item_alpha`, `item_beta` — скаляр или массив длины n_items.
    `ratio_kind` — один из RATIO_KINDS, `ratio_kwargs` пробрасывается в него.
    """
    with pm.Model() as model:
        a, d, obs_shifted = _add_latents(
            ratings_matrix, student_alpha, student_beta, item_alpha, item_beta)

        r = _compute_ratio_pt(a, d, ratio_kind=ratio_kind, ratio_kwargs=ratio_kwargs)
        else_probs = _likelihood_probs_from_r(r)

        # Вырожденные распределения, к которым тянет гейт
        const_5 = pt.as_tensor_variable(GATE_CONST_5)
        const_2 = pt.as_tensor_variable(GATE_CONST_2)

        p_extreme_5 = pm.math.sigmoid(gate_temp * (r - gate_hi))
        p_extreme_2 = pm.math.sigmoid(gate_temp * (gate_lo - r))

        # Веса берутся из сигмоид напрямую. Нормировать их друг на друга нельзя:
        # тогда p_5 + p_2 == 1 тождественно, p_middle зануляется и f_2..f_5
        # не влияют на правдоподобие вообще. Делим только если сумма превысила 1
        # (при gate_lo < gate_hi этого не происходит).
        overflow = pt.maximum(p_extreme_5 + p_extreme_2, 1.0)
        p_5 = p_extreme_5 / overflow
        p_2 = p_extreme_2 / overflow
        p_middle = 1.0 - p_5 - p_2

        prob_vec = (
            p_5[:, None] * const_5[None, :]
            + p_2[:, None] * const_2[None, :]
            + p_middle[:, None] * else_probs
        )
        pm.Categorical("ratings_obs", p=prob_vec, observed=obs_shifted)

    return model


def build_model_nocond(
    ratings_matrix,
    student_alpha=2, student_beta=2,
    item_alpha=2, item_beta=2,
    ratio_kind="sigmoid", ratio_kwargs=None,
):
    """Двудольная IRT-модель БЕЗ гейта: P(оценка) целиком задана f_2..f_5(ratio)."""
    with pm.Model() as model:
        a, d, obs_shifted = _add_latents(
            ratings_matrix, student_alpha, student_beta, item_alpha, item_beta)

        r = _compute_ratio_pt(a, d, ratio_kind=ratio_kind, ratio_kwargs=ratio_kwargs)
        pm.Categorical("ratings_obs", p=_likelihood_probs_from_r(r),
                       observed=obs_shifted)

    return model


BUILDERS = {"cond": build_model_cond, "nocond": build_model_nocond}


# ---------------------------------------------------------------------------
# Построение + сэмплирование
# ---------------------------------------------------------------------------
def _sample(model, draws, tune, chains, cores, target_accept, nuts_sampler,
            log_likelihood=False):
    with model:
        return pm.sample(
            draws=draws, tune=tune, chains=chains, cores=cores,
            target_accept=target_accept, return_inferencedata=True,
            nuts_sampler=nuts_sampler,
            idata_kwargs={"log_likelihood": log_likelihood},
        )


def create_bipartite_bayesian_network_cond(
    ratings_matrix,
    student_alpha=2, student_beta=2,
    item_alpha=2, item_beta=2,
    ratio_kind="sigmoid", ratio_kwargs=None,
    gate_lo=GATE_LO_DEFAULT, gate_hi=GATE_HI_DEFAULT, gate_temp=GATE_TEMP_DEFAULT,
    draws=1000, tune=2000, chains=4, cores=2,
    target_accept=0.95, nuts_sampler="pymc", log_likelihood=False,
):
    """`build_model_cond` + сэмплирование. Возвращает (trace, model)."""
    model = build_model_cond(
        ratings_matrix,
        student_alpha=student_alpha, student_beta=student_beta,
        item_alpha=item_alpha, item_beta=item_beta,
        ratio_kind=ratio_kind, ratio_kwargs=ratio_kwargs,
        gate_lo=gate_lo, gate_hi=gate_hi, gate_temp=gate_temp,
    )
    trace = _sample(model, draws, tune, chains, cores, target_accept,
                    nuts_sampler, log_likelihood)
    return trace, model


def create_bipartite_bayesian_network_nocond(
    ratings_matrix,
    student_alpha=2, student_beta=2,
    item_alpha=2, item_beta=2,
    ratio_kind="sigmoid", ratio_kwargs=None,
    draws=1000, tune=2000, chains=4, cores=2,
    target_accept=0.95, nuts_sampler="pymc", log_likelihood=False,
):
    """`build_model_nocond` + сэмплирование. Возвращает (trace, model)."""
    model = build_model_nocond(
        ratings_matrix,
        student_alpha=student_alpha, student_beta=student_beta,
        item_alpha=item_alpha, item_beta=item_beta,
        ratio_kind=ratio_kind, ratio_kwargs=ratio_kwargs,
    )
    trace = _sample(model, draws, tune, chains, cores, target_accept,
                    nuts_sampler, log_likelihood)
    return trace, model


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
    """Сложность предмета как относительное χ²-расстояние до двух эталонов.

    Возвращает число из [0, 1]: 0 — распределение оценок как у «лёгкого»
    предмета, 1 — как у «сложного».
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
    """Per-item Beta-приор с центром в chi-square оценке сложности.

    Концентрация α+β фиксирована (по умолчанию 4, как у Beta(2,2)), поэтому
    «сила» приора одинакова для всех предметов, дисперсия = μ(1-μ)/(α+β+1).

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
