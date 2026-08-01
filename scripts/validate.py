"""Валидация моделей: LOO-сравнение и posterior predictive check.

Отвечает на два вопроса, на которые диагностика сходимости не отвечает:

  1. Какая из шести конфигураций (3 ratio × 2 модели) лучше описывает данные?
     Считается PSIS-LOO — оценка предсказательной способности с поправкой на
     сложность модели. Сравнение делается ОТДЕЛЬНО по каждому семестру:
     LOO сопоставим только между моделями, обученными на одних наблюдениях.

  2. Насколько точно модель воспроизводит уже известные оценки? Считается
     log predictive density, доля угаданных оценок и MAE — ровно на тех
     вероятностях, которые выдаёт scripts/query.py. Метрики выборочные,
     поэтому дополняют LOO, а не заменяют его.

  3. Похожи ли данные, порождённые моделью, на настоящие? Из постериора
     сэмплируются оценки и их распределение сравнивается с фактическим.

Результат: results/loo_compare_sem{1,2}.csv, results/predictive_scores.csv,
           results/ppc_summary.csv, results/plots/ppc_sem{1,2}.png

    python scripts/validate.py
    python scripts/validate.py --thin 4     # реже прорежать (точнее, дольше)
"""
import _paths   # noqa: F401  (первым: добавляет src/ в sys.path)

import argparse
import warnings

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import BUILDERS, GRADES, chi_square_priors, grade_probs, RATIO_KINDS
from _paths import DATA, TRACES, RESULTS, PLOTS

# Сколько draws оставить для LOO. Полная трасса — 16000 draws × ~1100
# наблюдений, log_likelihood на ней занимает сотни мегабайт и считается
# долго; для устойчивой оценки LOO хватает пары тысяч.
DEFAULT_THIN = 8
# Порог Парето-k, выше которого оценка LOO для наблюдения ненадёжна
PARETO_K_LIMIT = 0.7


def load(sem):
    df = pd.read_csv(DATA / f"grades_sem{sem}.csv", index_col="student_id")
    return df, df.to_numpy(dtype="float32")


def build(sem_matrix, variant, ratio_kind):
    alpha, beta, _ = chi_square_priors(sem_matrix)
    return BUILDERS[variant](sem_matrix, item_alpha=alpha, item_beta=beta,
                             ratio_kind=ratio_kind)


def trace_with_loglik(sem, variant, ratio_kind, matrix, thin):
    """Загружает трассу и досчитывает log_likelihood на прорежённой выборке."""
    path = TRACES / f"trace_sem{sem}_{variant}_{ratio_kind}.nc"
    if not path.exists():
        return None, None
    idata = az.from_netcdf(path)
    idata = idata.sel(draw=slice(None, None, thin))
    model = build(matrix, variant, ratio_kind)
    with model:
        pm.compute_log_likelihood(idata, progressbar=False)
    return idata, model


# ---------------------------------------------------------------------------
# (1) LOO-сравнение
# ---------------------------------------------------------------------------
def compare_semester(sem, thin):
    df, matrix = load(sem)
    print(f"\n=== Семестр {sem}: LOO-сравнение шести конфигураций ===")

    idatas, bad_k = {}, {}
    for variant in ("cond", "nocond"):
        for ratio_kind in RATIO_KINDS:
            name = f"{variant}_{ratio_kind}"
            idata, _ = trace_with_loglik(sem, variant, ratio_kind, matrix, thin)
            if idata is None:
                print(f"  [нет трассы] {name}")
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                loo = az.loo(idata, pointwise=True)
            n_bad = int((loo.pareto_k.values > PARETO_K_LIMIT).sum())
            bad_k[name] = n_bad
            idatas[name] = idata
            flag = "" if n_bad == 0 else f"  [Парето-k > {PARETO_K_LIMIT}: {n_bad}]"
            print(f"  {name:<20} elpd_loo = {loo.elpd_loo:9.1f} ± {loo.se:5.1f}{flag}")

    if len(idatas) < 2:
        print("  сравнивать нечего")
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = az.compare(idatas, ic="loo")
    table["bad_pareto_k"] = [bad_k[i] for i in table.index]

    out = RESULTS / f"loo_compare_sem{sem}.csv"
    table.to_csv(out)
    print(f"\n  рейтинг (rank 0 — лучшая):")
    cols = [c for c in ("rank", "elpd_loo", "elpd_diff", "dse", "weight", "bad_pareto_k")
            if c in table.columns]
    print(table[cols].to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"  сохранено: {out}")
    return table


# ---------------------------------------------------------------------------
# (2) Прямая предсказательная точность на наблюдённых оценках
# ---------------------------------------------------------------------------
def predictive_scores(sem, thin):
    """Насколько хорошо модель угадывает уже известные оценки.

    Дополняет LOO: тот оценивает предсказание вне выборки и штрафует за
    сложность, а здесь считается ровно то, что выдаёт scripts/query.py —
    вероятности с маргинализацией по постериору. Метрики выборочные (модель
    эти оценки видела), поэтому смотреть их надо ВМЕСТЕ с LOO, а не вместо.
    """
    df, matrix = load(sem)
    obs_r, obs_c = np.where(~np.isnan(matrix))
    actual = matrix[obs_r, obs_c].astype(int)
    idx = actual - 2

    rows = []
    for variant in ("cond", "nocond"):
        for ratio_kind in RATIO_KINDS:
            path = TRACES / f"trace_sem{sem}_{variant}_{ratio_kind}.nc"
            if not path.exists():
                continue
            idata = az.from_netcdf(path).sel(draw=slice(None, None, thin))
            s = idata.posterior["student_ability"].values.reshape(-1, matrix.shape[0])
            d = idata.posterior["item_difficulty"].values.reshape(-1, matrix.shape[1])

            total = np.zeros((len(actual), len(GRADES)))
            for s_draw, d_draw in zip(s, d):
                total += grade_probs(s_draw[obs_r], d_draw[obs_c],
                                     ratio_kind=ratio_kind, gate=(variant == "cond"))
            probs = total / len(s)

            p_actual = probs[np.arange(len(idx)), idx]
            predicted = GRADES[probs.argmax(axis=1)]
            rows.append({
                "sem": sem, "model": variant, "ratio": ratio_kind,
                "log_pred": round(float(np.log(p_actual).mean()), 4),
                "accuracy": round(float((predicted == actual).mean()), 4),
                "mae": round(float(np.abs(probs @ GRADES - actual).mean()), 4),
            })
    return rows


# ---------------------------------------------------------------------------
# (3) Posterior predictive check
# ---------------------------------------------------------------------------
def ppc_semester(sem, variant, ratio_kind, thin):
    """Сравнивает распределение оценок из модели с фактическим."""
    df, matrix = load(sem)
    idata, model = trace_with_loglik(sem, variant, ratio_kind, matrix, thin)
    if idata is None:
        return None

    with model:
        pm.sample_posterior_predictive(idata, extend_inferencedata=True,
                                       progressbar=False)

    # observed / predicted хранятся сдвинутыми (0..3), возвращаем к шкале 2..5
    observed = idata.observed_data["ratings_obs"].values + 2
    predicted = idata.posterior_predictive["ratings_obs"].values.reshape(-1, observed.size) + 2

    obs_freq = np.array([(observed == g).mean() for g in GRADES])
    pred_draws = np.stack([[(row == g).mean() for g in GRADES] for row in predicted])
    pred_mean = pred_draws.mean(axis=0)
    pred_lo = np.percentile(pred_draws, 3, axis=0)
    pred_hi = np.percentile(pred_draws, 97, axis=0)

    rows = []
    for i, g in enumerate(GRADES):
        inside = pred_lo[i] <= obs_freq[i] <= pred_hi[i]
        rows.append({
            "sem": sem, "model": variant, "ratio": ratio_kind, "оценка": int(g),
            "факт": round(float(obs_freq[i]), 4),
            "модель": round(float(pred_mean[i]), 4),
            "интервал_lo": round(float(pred_lo[i]), 4),
            "интервал_hi": round(float(pred_hi[i]), 4),
            "попал": bool(inside),
        })
    return rows, obs_freq, pred_mean, pred_lo, pred_hi


def plot_ppc(sem, variant, ratio_kind, obs, mean, lo, hi):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(GRADES))
    ax.bar(x - 0.2, obs, width=0.4, label="факт", color="#1f77b4",
           edgecolor="black", lw=0.5)
    ax.bar(x + 0.2, mean, width=0.4, label="модель (среднее)", color="#d62728",
           alpha=0.85, edgecolor="black", lw=0.5)
    ax.errorbar(x + 0.2, mean, yerr=[mean - lo, hi - mean], fmt="none",
                ecolor="black", capsize=5, lw=1.4, label="94% интервал модели")
    ax.set_xticks(x)
    ax.set_xticklabels(GRADES)
    ax.set_xlabel("оценка")
    ax.set_ylabel("доля")
    ax.set_title(f"Семестр {sem} · {variant} · {ratio_kind}\n"
                 f"распределение оценок: факт против модели",
                 fontsize=12, weight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = PLOTS / f"ppc_sem{sem}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  сохранено: {out}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--thin", type=int, default=DEFAULT_THIN,
                   help=f"брать каждый N-й draw (по умолчанию {DEFAULT_THIN})")
    p.add_argument("--model", default="nocond", choices=tuple(BUILDERS),
                   help="какую конфигурацию проверять через PPC")
    p.add_argument("--ratio", default="sigmoid", choices=RATIO_KINDS)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    for sem in (1, 2):
        compare_semester(sem, args.thin)

    print("\n=== Предсказательная точность на наблюдённых оценках ===")
    score_rows = []
    for sem in (1, 2):
        score_rows.extend(predictive_scores(sem, args.thin))
    if score_rows:
        scores = pd.DataFrame(score_rows).sort_values(
            ["sem", "log_pred"], ascending=[True, False])
        print(scores.to_string(index=False))
        out = RESULTS / "predictive_scores.csv"
        scores.to_csv(out, index=False)
        print(f"сохранено: {out}")

    print(f"\n=== Posterior predictive check: {args.model} · {args.ratio} ===")
    ppc_rows = []
    for sem in (1, 2):
        result = ppc_semester(sem, args.model, args.ratio, args.thin)
        if result is None:
            print(f"  [нет трассы] семестр {sem}")
            continue
        rows, obs, mean, lo, hi = result
        ppc_rows.extend(rows)
        print(f"\n  Семестр {sem}:")
        print(pd.DataFrame(rows)[["оценка", "факт", "модель",
                                  "интервал_lo", "интервал_hi", "попал"]]
              .to_string(index=False))
        plot_ppc(sem, args.model, args.ratio, obs, mean, lo, hi)

    if ppc_rows:
        out = RESULTS / "ppc_summary.csv"
        pd.DataFrame(ppc_rows).to_csv(out, index=False)
        hits = sum(r["попал"] for r in ppc_rows)
        print(f"\nPPC: фактическая доля попала в 94% интервал модели "
              f"в {hits} случаях из {len(ppc_rows)}")
        print(f"сохранено: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
