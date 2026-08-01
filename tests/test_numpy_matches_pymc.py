"""numpy-двойники правдоподобия должны совпадать с самими PyMC-моделями.

Без этой проверки scripts/validate.py считал бы предсказательные метрики
одной формулой, а MCMC — другой, и сравнение семейств было бы недостоверным.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytensor  # noqa: E402

from utils import (  # noqa: E402
    BUILDERS, GRADES,
    grade_probs, grade_probs_tilted, grade_probs_ordered,
)

RATIO_KIND = "sigmoid"


@pytest.fixture(scope="module")
def ratings():
    """Маленькая матрица оценок без пропусков — хватает для сверки формул."""
    rng = np.random.default_rng(0)
    return rng.choice(GRADES, size=(12, 5)).astype(float)


def evaluate(model, names, point=None):
    """Считает именованные величины модели в одной точке пространства значений.

    Ключевой момент — replace_rvs_by_values: без него граф остаётся случайным
    и pytensor досэмплировал бы s и d заново вместо подстановки точки.
    """
    point = point or model.initial_point()
    graphs = model.replace_rvs_by_values([model[name] for name in names])
    fn = pytensor.function(model.value_vars, graphs, on_unused_input="ignore")
    values = fn(*[point[v.name] for v in model.value_vars])
    return dict(zip(names, values)), point


def categorical_p(model, point):
    """Вектор вероятностей, который модель подставляет в Categorical."""
    p = model["ratings_obs"].owner.inputs[-1]
    graph = model.replace_rvs_by_values([p])[0]
    fn = pytensor.function(model.value_vars, graph, on_unused_input="ignore")
    return fn(*[point[v.name] for v in model.value_vars])


@pytest.mark.parametrize("variant", ["nocond", "cond"])
def test_closed_form_matches_model(ratings, variant):
    model = BUILDERS[variant](ratings, ratio_kind=RATIO_KIND)
    vals, point = evaluate(model, ["student_ability", "item_difficulty"])
    rows, cols = np.where(~np.isnan(ratings))

    expected = grade_probs(vals["student_ability"][rows],
                           vals["item_difficulty"][cols],
                           ratio_kind=RATIO_KIND, gate=(variant == "cond"))
    actual = categorical_p(model, point)
    assert np.allclose(actual, expected, atol=1e-9), (
        f"{variant}: numpy-двойник расходится с моделью"
    )


def test_tilted_matches_model(ratings):
    model = BUILDERS["tilted"](ratings, ratio_kind=RATIO_KIND)
    vals, point = evaluate(
        model, ["student_ability", "item_difficulty", "grade_weights"])
    rows, cols = np.where(~np.isnan(ratings))

    expected = grade_probs_tilted(vals["student_ability"][rows],
                                  vals["item_difficulty"][cols],
                                  vals["grade_weights"], ratio_kind=RATIO_KIND)
    actual = categorical_p(model, point)
    assert np.allclose(actual, expected, atol=1e-9)


def test_ordered_matches_model(ratings):
    """У OrderedLogistic вероятности внутри, поэтому сверяем логправдоподобие."""
    model = BUILDERS["ordered"](ratings, ratio_kind=RATIO_KIND)
    vals, point = evaluate(
        model, ["student_ability", "item_difficulty", "cutpoints"])
    rows, cols = np.where(~np.isnan(ratings))
    observed = ratings[rows, cols].astype(int) - 2

    expected = grade_probs_ordered(vals["student_ability"][rows],
                                   vals["item_difficulty"][cols],
                                   vals["cutpoints"], ratio_kind=RATIO_KIND)
    logp_expected = float(np.log(expected[np.arange(len(observed)), observed]).sum())
    logp_actual = float(
        model.compile_logp(vars=[model["ratings_obs"]], sum=True)(point)
    )
    assert np.isclose(logp_actual, logp_expected, atol=1e-6), (
        f"ordered: логправдоподобие расходится ({logp_actual} vs {logp_expected})"
    )


def test_ordered_cutpoints_are_anchored(ratings):
    """Средний порог обязан быть закреплён в нуле.

    Регрессия на ненаблюдаемость: если освободить все три порога, прибавление
    константы к ним и сдвиг `s` на неё же дают идентичное правдоподобие.
    Постериор становится плоским хребтом — на семестре 2 это давало r_hat 1.53
    и ESS 8. Якорь убирает ровно эту свободу.
    """
    model = BUILDERS["ordered"](ratings, ratio_kind=RATIO_KIND)
    vals, _ = evaluate(model, ["cutpoints"])
    cutpoints = vals["cutpoints"]

    assert cutpoints.shape == (3,)
    assert cutpoints[1] == pytest.approx(0.0, abs=1e-12), (
        "средний порог должен быть жёстко закреплён в нуле"
    )
    assert cutpoints[0] < cutpoints[1] < cutpoints[2], "пороги должны быть возрастающими"

    free = [v.name for v in model.free_RVs]
    assert "cutpoints" not in free, "cutpoints должен быть Deterministic, а не свободным"
    assert "cutpoint_gaps" in free


def test_tilted_reduces_to_nocond_with_equal_weights(ratings):
    """Наклон с равными весами обязан совпасть с nocond — проверка на модели."""
    tilted = BUILDERS["tilted"](ratings, ratio_kind=RATIO_KIND)
    plain = BUILDERS["nocond"](ratings, ratio_kind=RATIO_KIND)

    vals, point = evaluate(
        tilted, ["student_ability", "item_difficulty", "grade_weights"])
    assert np.allclose(vals["grade_weights"], 0.25, atol=1e-9), (
        "начальная точка Dirichlet(1,1,1,1) должна быть равномерной"
    )

    rows, cols = np.where(~np.isnan(ratings))
    base = grade_probs(vals["student_ability"][rows], vals["item_difficulty"][cols],
                       ratio_kind=RATIO_KIND)
    assert np.allclose(categorical_p(tilted, point), base, atol=1e-9)
    _, plain_point = evaluate(plain, ["student_ability"])
    assert np.allclose(categorical_p(plain, plain_point), base, atol=1e-9)
