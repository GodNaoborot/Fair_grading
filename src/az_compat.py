"""Совместимость с arviz 0.x и 1.x.

В arviz 1.x часть публичного API переехала в пакеты arviz-stats/arviz-base,
и у двух используемых здесь функций изменились сигнатуры:

  - `hdi`: аргумент `hdi_prob` переименован в `prob`;
  - `compare`: убран аргумент `ic`, метод задаётся через `method`.

Имена функций при этом сохранились, поэтому проверка через `hasattr` ничего
не выявляет — различие только в сигнатурах. Модуль определяет их один раз
при импорте и предоставляет общий интерфейс.
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


def compare(idatas):
    """`az.compare` по LOO."""
    if COMPARE_HAS_IC:
        return az.compare(idatas, ic="loo")
    return az.compare(idatas)
