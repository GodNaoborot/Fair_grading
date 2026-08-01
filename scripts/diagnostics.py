"""Диагностика сходимости MCMC по всем посчитанным трассам.

Для каждой трассы считает r_hat, ESS и число дивергенций и складывает
в одну таблицу. Без этой проверки апостериорные числа доверия не заслуживают.

Результат: results/diagnostics.csv + печать в консоль.

    python scripts/diagnostics.py
"""
import _paths   # noqa: F401  (первым: добавляет src/ в sys.path)

import numpy as np
import pandas as pd
import arviz as az

from utils import BUILDERS, RATIO_KINDS
from _paths import TRACES, RESULTS

# Порог r_hat, выше которого цепи считаются несошедшимися
RHAT_LIMIT = 1.01
# Порог ESS, ниже которого выборка слишком автокоррелирована
ESS_LIMIT = 400


def divergence_count(trace):
    if "sample_stats" not in trace or "diverging" not in trace.sample_stats:
        return np.nan
    return int(trace.sample_stats["diverging"].values.sum())


def summarize_trace(trace):
    """r_hat / ESS по обеим группам параметров + дивергенции."""
    rhat = az.rhat(trace)
    ess = az.ess(trace)
    row = {}
    for var in ("student_ability", "item_difficulty"):
        r = rhat[var].values
        e = ess[var].values
        row[f"rhat_max_{var}"] = float(np.max(r))
        row[f"rhat_bad_{var}"] = int(np.sum(r > RHAT_LIMIT))
        row[f"ess_min_{var}"] = float(np.min(e))
        row[f"ess_bad_{var}"] = int(np.sum(e < ESS_LIMIT))
    row["divergences"] = divergence_count(trace)
    n_chains = trace.posterior.sizes["chain"]
    n_draws = trace.posterior.sizes["draw"]
    row["draws_total"] = n_chains * n_draws
    return row


def verdict(row):
    """Короткий вывод: сошлось или нет."""
    bad_rhat = row["rhat_bad_student_ability"] + row["rhat_bad_item_difficulty"]
    bad_ess = row["ess_bad_student_ability"] + row["ess_bad_item_difficulty"]
    div = row["divergences"]
    problems = []
    if bad_rhat:
        problems.append(f"r_hat>{RHAT_LIMIT}: {bad_rhat}")
    if bad_ess:
        problems.append(f"ESS<{ESS_LIMIT}: {bad_ess}")
    if div and not np.isnan(div):
        problems.append(f"дивергенций: {int(div)}")
    return "OK" if not problems else "; ".join(problems)


def main():
    rows = []
    for sem in (1, 2):
        for model in BUILDERS:
            for kind in RATIO_KINDS:
                name = f"trace_sem{sem}_{model}_{kind}"
                path = TRACES / f"{name}.nc"
                if not path.exists():
                    print(f"[skip] {name} — файла нет")
                    continue
                trace = az.from_netcdf(path)
                row = {"sem": sem, "model": model, "ratio": kind}
                row.update(summarize_trace(trace))
                row["verdict"] = verdict(row)
                rows.append(row)
                print(f"[ok]   {name}: {row['verdict']}")

    if not rows:
        print("\nТрасс не найдено. Сначала: python scripts/run_mcmc_chi_prior.py")
        return 1

    df = pd.DataFrame(rows)
    out = RESULTS / "diagnostics.csv"
    df.to_csv(out, index=False)

    print("\n" + "=" * 78)
    print("СВОДКА")
    print("=" * 78)
    compact = df[["sem", "model", "ratio",
                  "rhat_max_student_ability", "rhat_max_item_difficulty",
                  "ess_min_student_ability", "ess_min_item_difficulty",
                  "divergences", "verdict"]]
    compact.columns = ["сем", "модель", "ratio", "rhat_s", "rhat_d",
                       "ESS_s", "ESS_d", "div", "вердикт"]
    print(compact.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    n_ok = int((df["verdict"] == "OK").sum())
    print(f"\nсошлось: {n_ok} из {len(df)} трасс")
    print(f"таблица сохранена: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
