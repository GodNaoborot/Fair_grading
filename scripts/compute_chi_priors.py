"""Считает chi-square score сложности и Beta(α, β) приор для каждого предмета.

Beta-приор сохраняет концентрацию Beta(2,2) (α + β = 4), поэтому дисперсия
равна 0.05 при μ=0.5 и естественно сужается к 0 или 1.

Результат: data/chi_priors.json
"""
import _paths   # noqa: F401  (должен идти первым: подкручивает sys.path)

import json
import pandas as pd

from utils import chi_square_priors
from _paths import DATA


if __name__ == "__main__":
    out = {}
    print("Beta(2,2): variance(at mu=0.5) = 0.05, concentration alpha+beta = 4\n")
    for sem in (1, 2):
        df = pd.read_csv(DATA / f"grades_sem{sem}.csv", index_col="student_id")
        mat = df.to_numpy(dtype="float32")
        alpha, beta, mu = chi_square_priors(mat)
        print(f"==== Семестр {sem} ({df.shape[0]} студентов × {df.shape[1]} предметов) ====")
        rows = []
        for j, subj in enumerate(df.columns):
            print(f"  {subj:45s}  chi={mu[j]:.3f}  -> Beta(a={alpha[j]:.2f}, b={beta[j]:.2f})")
            rows.append({"subject": subj, "mu": float(mu[j]),
                         "alpha": float(alpha[j]), "beta": float(beta[j])})
        out[str(sem)] = rows
        print()

    with open(DATA / "chi_priors.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"сохранено: {DATA / 'chi_priors.json'}")
