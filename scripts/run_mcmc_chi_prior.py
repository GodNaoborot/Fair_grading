"""Запуск MCMC для sem1 и sem2 с chi-square приорами и тремя вариантами ratio.

Сохраняет 12 трасс:
  results/traces/trace_sem{1,2}_{cond,nocond}_{legacy_sd,current,sigmoid}.nc

Настройки: chains=4, tune=2000, draws=1000, target_accept=0.95.
"""
import _paths   # noqa: F401  (должен идти первым: подкручивает sys.path)

import numpy as np
import pandas as pd
import arviz as az

from utils import (
    create_bipartite_bayesian_network_cond,
    create_bipartite_bayesian_network_nocond,
    chi_square_priors,
    RATIO_KINDS,
)
from _paths import DATA, TRACES


def load(sem):
    df = pd.read_csv(DATA / f"grades_sem{sem}.csv", index_col="student_id")
    return df.columns.tolist(), df.to_numpy(dtype="float32")


def run_or_load(name, fn, mat, item_alpha, item_beta, ratio_kind,
                draws=1000, tune=2000, chains=4, cores=2):
    path = TRACES / f"{name}.nc"
    if path.exists():
        print(f"[skip] {name}  (cached)")
        return az.from_netcdf(path)
    print(f"[run]  {name}")
    np.random.seed(42)
    trace, _ = fn(
        mat,
        item_alpha=item_alpha, item_beta=item_beta,
        ratio_kind=ratio_kind,
        draws=draws, tune=tune, chains=chains, cores=cores,
    )
    trace.to_netcdf(path)
    return trace


if __name__ == "__main__":
    for sem in (1, 2):
        subjects, mat = load(sem)
        n_stud, n_item = mat.shape
        print(f"\n=== Semester {sem}: {n_stud} students x {n_item} items ===")

        alpha, beta, mu = chi_square_priors(mat)
        for j, name in enumerate(subjects):
            print(f"  [{j}] {name[:38]:<38} mu={mu[j]:.3f}  Beta(a={alpha[j]:.2f}, b={beta[j]:.2f})")

        for ratio_kind in RATIO_KINDS:
            for variant, fn in [
                ("cond",   create_bipartite_bayesian_network_cond),
                ("nocond", create_bipartite_bayesian_network_nocond),
            ]:
                run_or_load(
                    name=f"trace_sem{sem}_{variant}_{ratio_kind}",
                    fn=fn,
                    mat=mat,
                    item_alpha=alpha,
                    item_beta=beta,
                    ratio_kind=ratio_kind,
                )

    print("\nDone.")
