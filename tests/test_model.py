"""Свойства модели, которые должны выполняться при любых правках формул.

Запуск:  pytest
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils import (  # noqa: E402
    f_2, f_3, f_4, f_5, ratio, RATIO_KINDS, GRADES,
    grade_probs, expected_grade, grade_probs_tilted, grade_probs_ordered,
    chi_square_difficulty, chi_square_priors,
    CHI_REF_EASY, CHI_REF_HARD, CHI_PRIOR_CONCENTRATION,
    GATE_LO_DEFAULT, GATE_HI_DEFAULT,
)

GRID = np.linspace(0.01, 0.99, 99)


# ---------------------------------------------------------------------------
# Семейство f
# ---------------------------------------------------------------------------
def test_f_normalized():
    """f_2 + f_3 + f_4 + f_5 == 1 по построению, без отдельной нормировки."""
    total = f_2(GRID) + f_3(GRID) + f_4(GRID) + f_5(GRID)
    assert np.allclose(total, 1.0, atol=1e-12)


def test_f_nonnegative():
    for f in (f_2, f_3, f_4, f_5):
        assert np.all(f(GRID) >= 0), f"{f.__name__} уходит в отрицательные значения"


@pytest.mark.parametrize("fa, fb", [(f_2, f_5), (f_3, f_4)])
def test_f_mirror_symmetry(fa, fb):
    """f_5 — отражение f_2, f_4 — отражение f_3."""
    assert np.allclose(fa(GRID), fb(1 - GRID), atol=1e-12)


def test_f2_decreasing_f5_increasing():
    assert np.all(np.diff(f_2(GRID)) < 0), "f_2 должна убывать по x"
    assert np.all(np.diff(f_5(GRID)) > 0), "f_5 должна возрастать по x"


# ---------------------------------------------------------------------------
# Семейство ratio
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", RATIO_KINDS)
def test_ratio_in_unit_interval(kind):
    s, d = np.meshgrid(GRID, GRID)
    r = ratio(s, d, kind=kind)
    assert np.all(r >= 0) and np.all(r <= 1), f"{kind} выходит за [0, 1]"


@pytest.mark.parametrize("kind", RATIO_KINDS)
def test_ratio_monotone_in_ability(kind):
    """При фиксированной сложности ratio не убывает по способности студента."""
    for d in (0.2, 0.5, 0.8):
        r = ratio(GRID, np.full_like(GRID, d), kind=kind)
        assert np.all(np.diff(r) >= -1e-12), f"{kind} убывает по s при d={d}"


@pytest.mark.parametrize("kind", ["legacy_sd", "sigmoid"])
def test_symmetric_ratios_are_half_on_diagonal(kind):
    """Симметричные варианты при s == d дают ровно 1/2."""
    r = ratio(GRID, GRID, kind=kind)
    assert np.allclose(r, 0.5, atol=1e-6)


def test_unknown_ratio_kind_raises():
    with pytest.raises(ValueError, match="неизвестный ratio kind"):
        ratio(0.5, 0.5, kind="не-существует")


# ---------------------------------------------------------------------------
# Вероятности оценок
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", RATIO_KINDS)
@pytest.mark.parametrize("gate", [False, True])
def test_grade_probs_is_a_distribution(kind, gate):
    s, d = np.meshgrid(GRID, GRID)
    p = grade_probs(s, d, ratio_kind=kind, gate=gate)
    assert p.shape == s.shape + (4,)
    assert np.all(p >= 0)
    assert np.allclose(p.sum(axis=-1), 1.0, atol=1e-9)


@pytest.mark.parametrize("kind", RATIO_KINDS)
def test_expected_grade_within_scale(kind):
    s, d = np.meshgrid(GRID, GRID)
    e = expected_grade(s, d, ratio_kind=kind)
    assert np.all(e >= 2.0) and np.all(e <= 5.0)


@pytest.mark.parametrize("kind", RATIO_KINDS)
def test_expected_grade_increases_with_ability(kind):
    """Более способный студент на том же предмете не может ожидать меньше."""
    for d in (0.2, 0.5, 0.8):
        e = expected_grade(GRID, np.full_like(GRID, d), ratio_kind=kind)
        assert np.all(np.diff(e) >= -1e-9), f"{kind}: E[оценка] падает по s при d={d}"


@pytest.mark.parametrize("kind", ["legacy_sd", "sigmoid"])
def test_expected_grade_decreases_with_difficulty(kind):
    """На более сложном предмете тот же студент ожидает не больше.

    `current` сюда не входит намеренно: он асимметричен и в углу (s=1, d=1)
    даёт ratio=1 — известное свойство, разобранное в model_comparison.ipynb.
    """
    for s in (0.2, 0.5, 0.8):
        e = expected_grade(np.full_like(GRID, s), GRID, ratio_kind=kind)
        assert np.all(np.diff(e) <= 1e-9), f"{kind}: E[оценка] растёт по d при s={s}"


def test_gate_pushes_extremes():
    """В зоне срабатывания гейт поднимает вероятность крайней оценки."""
    kw = dict(ratio_kind="sigmoid")
    # ratio ≈ 0.88 — выше порога gate_hi=0.9 гейт ещё не насыщен,
    # но уже заметно тянет распределение к пятёрке
    assert grade_probs(0.7, 0.3, gate=True, **kw)[3] > grade_probs(0.7, 0.3, gate=False, **kw)[3]
    assert grade_probs(0.3, 0.7, gate=True, **kw)[0] > grade_probs(0.3, 0.7, gate=False, **kw)[0]


def test_gate_is_a_real_mixture():
    """Регрессия: вес f-вероятностей в `cond` не должен зануляться.

    Раньше обе сигмоиды гейта нормировались друг на друга, из-за чего
    p_5 + p_2 == 1 тождественно, p_middle == 0, и семейство f_2..f_5
    не влияло на правдоподобие вообще — модель вырождалась в выбор между
    2 и 5, выдавая на середине шкалы [0.499, 0.001, 0.001, 0.499].

    Проверяем не «гейт выключен» (это зависит от порогов), а само свойство
    смеси: на равных s и d середина обязана доминировать.
    """
    p = grade_probs(0.5, 0.5, ratio_kind="sigmoid", gate=True)
    assert p[1] + p[2] > p[0] + p[3], (
        f"при s=d середина должна перевешивать края, получено {p.round(4)}"
    )
    assert p[1] > 0.2 and p[2] > 0.2, (
        f"тройка и четвёрка не должны быть подавлены, получено {p.round(4)}"
    )


def test_gate_thresholds_are_inside_the_scale():
    """Пороги гейта должны реально влиять, иначе `cond` вырождается в `nocond`.

    При порогах у самых краёв (0.1 / 0.9) f-функции и без гейта дают ~0.997,
    поэтому дефолты сдвинуты внутрь.
    """
    assert 0.2 <= GATE_LO_DEFAULT < 0.5 < GATE_HI_DEFAULT <= 0.8
    wide = grade_probs(0.7, 0.5, ratio_kind="sigmoid", gate=True,
                       gate_lo=0.1, gate_hi=0.9, gate_temp=30.0)
    tight = grade_probs(0.7, 0.5, ratio_kind="sigmoid", gate=True)
    assert tight[3] > wide[3] + 0.2, (
        "текущие пороги должны заметно усиливать пятёрку по сравнению с 0.1/0.9"
    )


@pytest.mark.parametrize("kind", RATIO_KINDS)
def test_gate_never_makes_middle_grades_impossible(kind):
    """Ни при каком (s, d) модель с гейтом не должна обнулять оценки 3 и 4."""
    s, d = np.meshgrid(np.linspace(0.3, 0.7, 21), np.linspace(0.3, 0.7, 21))
    p = grade_probs(s, d, ratio_kind=kind, gate=True)
    assert np.all(p[..., 1] > 1e-3), f"{kind}: тройка стала невозможной"
    assert np.all(p[..., 2] > 1e-3), f"{kind}: четвёрка стала невозможной"


# ---------------------------------------------------------------------------
# Альтернативные семейства правдоподобия
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", RATIO_KINDS)
def test_tilted_is_a_distribution(kind):
    s, d = np.meshgrid(GRID, GRID)
    p = grade_probs_tilted(s, d, [0.05, 0.3, 0.3, 0.35], ratio_kind=kind)
    assert p.shape == s.shape + (4,)
    assert np.all(p >= 0)
    assert np.allclose(p.sum(axis=-1), 1.0, atol=1e-9)


def test_tilted_with_flat_weights_equals_base():
    """Равные веса не должны ничего менять — это тождественный наклон."""
    base = grade_probs(GRID, 1 - GRID, ratio_kind="sigmoid")
    flat = grade_probs_tilted(GRID, 1 - GRID, np.full(4, 0.25), ratio_kind="sigmoid")
    assert np.allclose(base, flat, atol=1e-12)


def test_tilted_shifts_marginal_in_the_expected_direction():
    """Уменьшение веса двойки должно уменьшать её вероятность везде."""
    kw = dict(ratio_kind="sigmoid")
    base = grade_probs_tilted(GRID, 1 - GRID, [0.25, 0.25, 0.25, 0.25], **kw)
    fewer_twos = grade_probs_tilted(GRID, 1 - GRID, [0.02, 0.33, 0.32, 0.33], **kw)
    assert np.all(fewer_twos[..., 0] <= base[..., 0] + 1e-12)


@pytest.mark.parametrize("kind", RATIO_KINDS)
def test_ordered_is_a_distribution(kind):
    s, d = np.meshgrid(GRID, GRID)
    p = grade_probs_ordered(s, d, [-2.0, 0.0, 2.0], ratio_kind=kind)
    assert p.shape == s.shape + (4,)
    assert np.all(p >= -1e-12)
    assert np.allclose(p.sum(axis=-1), 1.0, atol=1e-9)


@pytest.mark.parametrize("kind", RATIO_KINDS)
def test_ordered_is_stochastically_monotone(kind):
    """Более способный студент не может иметь больше шансов на двойку."""
    for d in (0.3, 0.5, 0.7):
        p = grade_probs_ordered(GRID, np.full_like(GRID, d), [-2.0, 0.0, 2.0],
                                ratio_kind=kind)
        assert np.all(np.diff(p[..., 0]) <= 1e-12), f"{kind}: P(2) растёт по s"
        assert np.all(np.diff(p[..., 3]) >= -1e-12), f"{kind}: P(5) падает по s"


def test_ordered_cutpoints_control_marginal_share():
    """Сдвиг всех порогов вниз должен снижать долю низких оценок."""
    kw = dict(ratio_kind="sigmoid")
    strict = grade_probs_ordered(GRID, 1 - GRID, [-1.0, 1.0, 3.0], **kw)
    lenient = grade_probs_ordered(GRID, 1 - GRID, [-3.0, -1.0, 1.0], **kw)
    assert lenient[..., 0].mean() < strict[..., 0].mean()
    assert lenient[..., 3].mean() > strict[..., 3].mean()


# ---------------------------------------------------------------------------
# Chi-square приоры
# ---------------------------------------------------------------------------
def test_chi_square_recognizes_reference_distributions():
    """Эталонное «лёгкое» распределение даёт ~0, «сложное» — ~1."""
    assert chi_square_difficulty(CHI_REF_EASY) < 0.01
    assert chi_square_difficulty(CHI_REF_HARD) > 0.99


def test_chi_square_in_unit_interval():
    rng = np.random.default_rng(0)
    for _ in range(200):
        dist = rng.dirichlet(np.ones(4))
        score = chi_square_difficulty(dist)
        assert 0.0 <= score <= 1.0


def test_chi_square_priors_shapes_and_concentration():
    rng = np.random.default_rng(1)
    matrix = rng.choice([2, 3, 4, 5], size=(40, 5)).astype(float)
    matrix[rng.random(matrix.shape) < 0.2] = np.nan

    alpha, beta, mu = chi_square_priors(matrix)
    assert alpha.shape == beta.shape == mu.shape == (5,)
    assert np.allclose(alpha + beta, CHI_PRIOR_CONCENTRATION)
    assert np.all(alpha > 0) and np.all(beta > 0)
    # Центр Beta-приора совпадает с chi-square оценкой
    assert np.allclose(alpha / (alpha + beta), mu, atol=1e-3)


def test_chi_square_priors_handles_empty_column():
    """Предмет без единой оценки получает нейтральный приор mu=0.5."""
    matrix = np.full((10, 3), np.nan)
    matrix[:, 0] = 4.0
    _, _, mu = chi_square_priors(matrix)
    assert mu[1] == 0.5 and mu[2] == 0.5


# ---------------------------------------------------------------------------
# Согласованность констант
# ---------------------------------------------------------------------------
def test_grades_constant_matches_scale():
    assert list(GRADES) == [2, 3, 4, 5]
