"""MCMC для обоих семестров: 3 варианта ratio × 2 модели × 2 семестра = 12 трасс.

Сохраняет results/traces/trace_sem{1,2}_{cond,nocond}_{legacy_sd,current,sigmoid}.nc
Уже посчитанные трассы переиспользуются, поэтому повторный запуск дёшев.

    python scripts/run_mcmc_chi_prior.py                  # полный прогон
    python scripts/run_mcmc_chi_prior.py --quick          # черновой, для проверки пайплайна
    python scripts/run_mcmc_chi_prior.py --sem 1 --ratio sigmoid
"""
import _paths   # noqa: F401  (первым: добавляет src/ в sys.path)

import argparse
import time

import numpy as np
import pandas as pd
import arviz as az

import az_compat

import pymc as pm

from utils import BUILDERS, chi_square_priors, RATIO_KINDS
from _paths import DATA, TRACES

MODELS = tuple(BUILDERS)
# Базовый набор: только эти комбинации считаются, если не указано иное.
# tilted/ordered — альтернативные правдоподобия, они сравниваются с nocond
# в scripts/validate.py и запускаются явным --model.
DEFAULT_MODELS = ("cond", "nocond")

# log_likelihood в трассу не пишем — это (draws × n_obs) float64, сотни МБ
# на файл. validate.py досчитывает его на прорежённой выборке.


def load(sem):
    df = pd.read_csv(DATA / f"grades_sem{sem}.csv", index_col="student_id")
    return df.columns.tolist(), df.to_numpy(dtype="float32")


NETCDF_SAFE = (str, int, float, complex, bytes, list, tuple, np.ndarray, np.number)


def sanitize_attrs(trace):
    """Приводит атрибуты трассы к типам, которые понимает netCDF.

    nutpie кладёт свои настройки словарём в attrs самой InferenceData,
    а netCDF принимает только строки, числа и массивы — без этого
    `to_netcdf` падает с TypeError.
    """
    for attrs in az_compat.attr_dicts(trace):
        for key, value in list(attrs.items()):
            if not isinstance(value, NETCDF_SAFE):
                attrs[key] = str(value)
    return trace


def run_or_load(name, variant, mat, item_alpha, item_beta, ratio_kind, sampler):
    path = TRACES / f"{name}.nc"
    if path.exists():
        print(f"[кэш] {name}")
        return az.from_netcdf(path)

    print(f"[счёт] {name} ...", flush=True)
    started = time.time()
    np.random.seed(42)
    model = BUILDERS[variant](
        mat, item_alpha=item_alpha, item_beta=item_beta, ratio_kind=ratio_kind)
    with model:
        trace = pm.sample(return_inferencedata=True, **sampler)
    sanitize_attrs(trace).to_netcdf(path)
    print(f"[готово] {name}  за {time.time() - started:.0f} с")
    return trace


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sem", type=int, choices=(1, 2), action="append",
                   help="считать только этот семестр (можно указать дважды)")
    p.add_argument("--model", choices=tuple(MODELS), action="append",
                   help="считать только этот вариант модели")
    p.add_argument("--ratio", choices=RATIO_KINDS, action="append",
                   help="считать только этот вариант ratio")
    p.add_argument("--quick", action="store_true",
                   help="черновые настройки (tune=500, draws=500) для проверки пайплайна")
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--tune", type=int, default=2000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--cores", type=int, default=2)
    p.add_argument("--target-accept", type=float, default=0.95,
                   help="доля принятых шагов NUTS; поднимать при плохом r_hat")
    p.add_argument("--sampler", choices=("nutpie", "pymc"), default="nutpie",
                   help="бэкенд NUTS. nutpie (по умолчанию) компилирует через numba "
                        "и не требует C++-компилятора; pymc без g++ работает на "
                        "чистом Python и практически неприменим")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    sampler = dict(draws=args.draws, tune=args.tune,
                   chains=args.chains, cores=args.cores,
                   target_accept=args.target_accept,
                   nuts_sampler=args.sampler)
    if args.quick:
        sampler.update(draws=500, tune=500)
        print("режим --quick: черновые настройки, диагностики будут слабые\n")

    semesters = args.sem or [1, 2]
    models = args.model or list(DEFAULT_MODELS)
    ratios = args.ratio or list(RATIO_KINDS)

    total_started = time.time()
    for sem in semesters:
        subjects, mat = load(sem)
        n_stud, n_item = mat.shape
        print(f"\n=== Семестр {sem}: {n_stud} студентов x {n_item} предметов ===")

        alpha, beta, mu = chi_square_priors(mat)
        for j, name in enumerate(subjects):
            print(f"  [{j}] {name[:38]:<38} mu={mu[j]:.3f}  Beta(a={alpha[j]:.2f}, b={beta[j]:.2f})")

        for ratio_kind in ratios:
            for variant in models:
                run_or_load(
                    name=f"trace_sem{sem}_{variant}_{ratio_kind}",
                    variant=variant,
                    mat=mat,
                    item_alpha=alpha,
                    item_beta=beta,
                    ratio_kind=ratio_kind,
                    sampler=sampler,
                )

    print(f"\nВсё. Суммарно {time.time() - total_started:.0f} с.")
    print("Дальше: python scripts/diagnostics.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
