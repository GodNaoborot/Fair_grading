"""Совместимость с arviz 0.x и 1.x.

В arviz 1.x часть публичного API переехала в пакеты arviz-stats/arviz-base.
Различия, которые задевают этот проект:

  - `hdi`: аргумент `hdi_prob` переименован в `prob`;
  - `compare`: убран аргумент `ic`, метод задаётся через `method`;
  - `InferenceData.groups`: было методом, стало свойством.

Имена при этом сохранились, поэтому проверка через `hasattr` различий не
выявляет. Модуль определяет фактические сигнатуры один раз при импорте и
предоставляет общий интерфейс для обеих веток arviz.
"""
import inspect

import arviz as az

_HDI_PARAMS = inspect.signature(az.hdi).parameters
HDI_PROB_KW = "prob" if "prob" in _HDI_PARAMS else "hdi_prob"

_COMPARE_PARAMS = inspect.signature(az.compare).parameters
COMPARE_HAS_IC = "ic" in _COMPARE_PARAMS

ARVIZ_MAJOR = int(az.__version__.split(".")[0])


def hdi(data, prob=0.94, **kwargs):
    """`az.hdi` с единым именем аргумента вероятности."""
    return az.hdi(data, **{HDI_PROB_KW: prob}, **kwargs)


# Имя поля с точечной оценкой elpd: в arviz 0.x — `elpd_loo`, в 1.x — `elpd`.
ELPD_ATTRS = ("elpd_loo", "elpd")


def elpd(loo_result):
    """Точечная оценка elpd из результата `az.loo`."""
    for name in ELPD_ATTRS:
        if hasattr(loo_result, name):
            return float(getattr(loo_result, name))
    raise AttributeError(f"в результате LOO нет ни одного поля из {ELPD_ATTRS}")


def compare(idatas):
    """`az.compare` по LOO с приведением названий колонок к виду arviz 0.x."""
    table = az.compare(idatas, ic="loo") if COMPARE_HAS_IC else az.compare(idatas)
    return table.rename(columns={"elpd": "elpd_loo", "elpd_diff": "elpd_diff"})


def group_names(idata):
    """Имена групп трассы без ведущего слеша и без корня.

    В arviz 0.x `groups` — метод, возвращающий имена вида `posterior`.
    В 1.x трасса представлена `DataTree`, а `groups` стал свойством и
    возвращает пути вида `/posterior`, включая корень `/`.
    """
    attr = idata.groups
    raw = attr() if callable(attr) else attr
    return tuple(name.strip("/") for name in raw if name.strip("/"))


def attr_dicts(idata):
    """Все словари атрибутов трассы: корневой и по одному на группу."""
    return [idata.attrs] + [getattr(idata, name).attrs
                            for name in group_names(idata)]
