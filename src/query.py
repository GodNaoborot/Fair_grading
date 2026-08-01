"""Апостериорные запросы к посчитанным трассам.

Отвечает на три вопроса:
  - какая у студента способность `s` (распределение, а не точка);
  - какая у предмета сложность `d`;
  - какую оценку модель ожидает от этой пары и совпало ли это с реальностью.

Всё считается по выборкам из трассы, поэтому на выходе всегда распределение
с интервалом неопределённости, а не одно число.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import arviz as az

from utils import GRADES, grade_probs, grade_probs_ordered, grade_probs_tilted

HDI_PROB = 0.94


# ---------------------------------------------------------------------------
# Загрузка
# ---------------------------------------------------------------------------
def trace_path(traces_dir, sem, model, ratio_kind):
    return traces_dir / f"trace_sem{sem}_{model}_{ratio_kind}.nc"


def load_grades(data_dir, sem):
    """DataFrame оценок: индекс — student_id, колонки — предметы."""
    return pd.read_csv(data_dir / f"grades_sem{sem}.csv", index_col="student_id")


def load_trace(traces_dir, sem, model, ratio_kind):
    path = trace_path(traces_dir, sem, model, ratio_kind)
    if not path.exists():
        raise FileNotFoundError(
            f"нет трассы {path.name}. Сначала запустите:\n"
            f"    python scripts/run_mcmc_chi_prior.py"
        )
    return az.from_netcdf(path)


# ---------------------------------------------------------------------------
# Разрешение имён в индексы
# ---------------------------------------------------------------------------
def resolve_student(grades, student):
    """student_id → позиционный индекс строки. Принимает id как int или строку."""
    ids = grades.index.tolist()
    try:
        sid = int(student)
    except (TypeError, ValueError):
        raise ValueError(f"student_id должен быть числом, получено {student!r}")
    if sid not in ids:
        raise ValueError(f"студента {sid} нет в выборке (есть {min(ids)}..{max(ids)})")
    return ids.index(sid), sid


def resolve_subject(grades, subject):
    """Имя предмета → индекс колонки. Допускает регистронезависимое вхождение."""
    subjects = grades.columns.tolist()
    if subject in subjects:
        return subjects.index(subject), subject

    needle = str(subject).strip().casefold()
    hits = [s for s in subjects if needle in s.casefold()]
    if len(hits) == 1:
        return subjects.index(hits[0]), hits[0]
    if len(hits) > 1:
        raise ValueError(
            f"«{subject}» подходит сразу нескольким предметам: {', '.join(hits)}"
        )
    raise ValueError(
        f"предмет «{subject}» не найден. Доступны:\n  " + "\n  ".join(subjects)
    )


# ---------------------------------------------------------------------------
# Выборки из постериора
# ---------------------------------------------------------------------------
def _flat_samples(trace, var_name, idx):
    """Все draw'ы по всем цепям для одного элемента вектора."""
    da = trace.posterior[var_name]
    return da.isel({da.dims[-1]: idx}).values.reshape(-1)


def student_samples(trace, student_idx):
    return _flat_samples(trace, "student_ability", student_idx)


def item_samples(trace, item_idx):
    return _flat_samples(trace, "item_difficulty", item_idx)


@dataclass
class Summary:
    """Точечная сводка по набору выборок."""
    mean: float
    sd: float
    hdi_low: float
    hdi_high: float

    def __str__(self):
        return (f"{self.mean:.3f} ± {self.sd:.3f}   "
                f"{int(HDI_PROB * 100)}% HDI [{self.hdi_low:.3f}, {self.hdi_high:.3f}]")


def summarize(samples, hdi_prob=HDI_PROB):
    lo, hi = az.hdi(np.asarray(samples), hdi_prob=hdi_prob)
    return Summary(mean=float(np.mean(samples)), sd=float(np.std(samples)),
                   hdi_low=float(lo), hdi_high=float(hi))


def rank_of(trace, var_name, idx):
    """Позиция элемента по апостериорному среднему (1 — самый низкий) и размер группы."""
    means = trace.posterior[var_name].mean(("chain", "draw")).values
    order = np.argsort(means)
    return int(np.where(order == idx)[0][0]) + 1, len(means)


# ---------------------------------------------------------------------------
# Предсказание оценки
# ---------------------------------------------------------------------------
# Дополнительная переменная постериора для вариантов, у которых правдоподобие
# не задаётся одними (s, d). У cond/nocond её нет.
EXTRA_VAR = {"tilted": "grade_weights", "ordered": "cutpoints"}


def _extra_samples(trace, variant, n_draws):
    var = EXTRA_VAR.get(variant)
    if var is None:
        return None
    return trace.posterior[var].values.reshape(n_draws, -1)


def probs_for_draw(variant, s, d, ratio_kind, extra=None):
    """Вероятности оценок для одного draw — тем же правдоподобием, что в модели."""
    if variant == "tilted":
        return grade_probs_tilted(s, d, extra, ratio_kind=ratio_kind)
    if variant == "ordered":
        return grade_probs_ordered(s, d, extra, ratio_kind=ratio_kind)
    return grade_probs(s, d, ratio_kind=ratio_kind, gate=(variant == "cond"))


def predicted_grade_distribution(trace, student_idx, item_idx,
                                 ratio_kind, variant):
    """Апостериорное распределение оценки для пары (студент, предмет).

    Для каждого draw считается вектор вероятностей оценок, затем усредняется
    по draws — это маргинализация по неопределённости в параметрах, а не
    подстановка их средних значений.
    """
    s = student_samples(trace, student_idx)
    d = item_samples(trace, item_idx)
    extra = _extra_samples(trace, variant, len(s))
    if extra is None:
        return probs_for_draw(variant, s, d, ratio_kind).mean(axis=0)

    total = np.zeros(len(GRADES))
    for i in range(len(s)):
        total += probs_for_draw(variant, s[i], d[i], ratio_kind, extra[i])
    return total / len(s)


def observed_grade(grades, student_idx, item_idx):
    """Фактическая оценка или None, если предмет студентом не сдавался."""
    value = grades.iloc[student_idx, item_idx]
    return None if pd.isna(value) else int(value)


def grade_table(probs, observed=None):
    """DataFrame с вероятностями оценок и пометкой фактической."""
    df = pd.DataFrame({"оценка": GRADES, "P": probs})
    df["P"] = df["P"].round(4)
    if observed is not None:
        df["факт"] = ["<--" if g == observed else "" for g in GRADES]
    return df.set_index("оценка")
