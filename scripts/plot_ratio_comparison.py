"""Сравнение трёх вариантов ratio: формы функций + сдвиги в постериорах.

Сохраняет (в results/plots/):
  ratio_surfaces.png            — 2D-карты ratio(s, d) + ожидаемой оценки
  ratio_curves.png              — E[grade | s] при разных d
  ratio_posterior_compare.png   — bar-charts постериорного item_difficulty
                                   по каждому предмету (4 панели: 2 сем × 2 модели)
  ratio_hdi_sem1_<model>.png    — forest plot HDI, 3 ratio бок о бок
"""
import _paths   # noqa: F401  (должен идти первым: подкручивает sys.path)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import arviz as az

from utils import f_2, f_3, f_4, f_5, ratio, RATIO_KINDS
from _paths import DATA, TRACES, PLOTS


# Цветовая палитра для трёх вариантов ratio.
COLORS = {
    "legacy_sd": "#1f77b4",   # blue
    "current":   "#d62728",   # red
    "sigmoid":   "#2ca02c",   # green
}
LABELS = {
    "legacy_sd": "legacy: s / (s + d)",
    "current":   "current: 1 − d (1 − s)",
    "sigmoid":   "sigmoid: σ(5(s − d))",
}


# ---------------------------------------------------------------------------
# (1) Тепловые карты ratio(s, d) и ожидаемой оценки
# ---------------------------------------------------------------------------
def plot_surfaces():
    s = np.linspace(1e-3, 1 - 1e-3, 200)
    d = np.linspace(1e-3, 1 - 1e-3, 200)
    S, D = np.meshgrid(s, d, indexing="ij")
    grades = np.array([2, 3, 4, 5])

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))

    for col, kind in enumerate(RATIO_KINDS):
        R = ratio(S, D, kind=kind)

        # Top row: ratio(s, d) heatmap
        ax = axes[0, col]
        im = ax.imshow(R, origin="lower", extent=[0, 1, 0, 1],
                       vmin=0, vmax=1, cmap="viridis", aspect="equal")
        ax.set_title(f"{LABELS[kind]}\nratio(s, d)", fontsize=11)
        ax.set_xlabel("d (item difficulty)"); ax.set_ylabel("s (student ability)")
        cs = ax.contour(d, s, R, levels=[0.2, 0.4, 0.6, 0.8], colors="white", linewidths=0.8)
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f")
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Bottom row: E[grade | s, d]
        p2 = f_2(R); p3 = f_3(R); p4 = f_4(R); p5 = f_5(R)
        E = 2 * p2 + 3 * p3 + 4 * p4 + 5 * p5
        ax2 = axes[1, col]
        im2 = ax2.imshow(E, origin="lower", extent=[0, 1, 0, 1],
                         vmin=2, vmax=5, cmap="RdYlGn", aspect="equal")
        ax2.set_title(f"E[grade | s, d]   range [{E.min():.2f}, {E.max():.2f}]",
                      fontsize=11)
        ax2.set_xlabel("d (item difficulty)"); ax2.set_ylabel("s (student ability)")
        cs2 = ax2.contour(d, s, E, levels=[2.5, 3.0, 3.5, 4.0, 4.5],
                          colors="black", linewidths=0.6)
        ax2.clabel(cs2, inline=True, fontsize=8, fmt="%.1f")
        plt.colorbar(im2, ax=ax2, shrink=0.8)

        # Mark the four corner cases on both rows
        for ax_ in (ax, ax2):
            for (ss, dd) in [(0, 1), (0.5, 0.5), (1, 0), (0, 0), (1, 1)]:
                ax_.scatter([dd], [ss], s=40, c="white",
                            edgecolor="black", lw=1.2, zorder=5)

    fig.suptitle("ratio(s, d) surfaces and resulting E[grade] under three formulations",
                 fontsize=14, weight="bold")
    fig.tight_layout()
    out = PLOTS / "ratio_surfaces.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# (2) Профили E[grade | s] вдоль диагонали s = d и при фиксированном d
# ---------------------------------------------------------------------------
def plot_curves():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    s = np.linspace(0.01, 0.99, 200)

    # Diagonal slice: s = d -> "equally matched" student & subject
    for ax, d_fixed, ttl in zip(
        axes,
        [None, 0.3, 0.8],
        ["diagonal: s = d (equally matched)", "d = 0.3 (easy subject)", "d = 0.8 (hard subject)"],
    ):
        for kind in RATIO_KINDS:
            if d_fixed is None:
                R = ratio(s, s, kind=kind)
            else:
                R = ratio(s, np.full_like(s, d_fixed), kind=kind)
            E = 2 * f_2(R) + 3 * f_3(R) + 4 * f_4(R) + 5 * f_5(R)
            ax.plot(s, E, color=COLORS[kind], lw=2.2, label=LABELS[kind])
        ax.set_title(ttl, fontsize=11)
        ax.set_xlabel("student ability s")
        ax.set_ylabel("E[grade | s, d]")
        ax.set_ylim(1.95, 5.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8.5, loc="lower right")
        ax.axhline(2, color="grey", ls=":", lw=0.7)
        ax.axhline(5, color="grey", ls=":", lw=0.7)

    fig.suptitle("E[grade | s, d] as a function of student ability under three ratios",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    out = PLOTS / "ratio_curves.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# (3) Сравнение постериоров из MCMC-трасс
# ---------------------------------------------------------------------------
def load_traces(sem, model):
    traces = {}
    for kind in RATIO_KINDS:
        p = TRACES / f"trace_sem{sem}_{model}_{kind}.nc"
        if p.exists():
            traces[kind] = az.from_netcdf(p)
    return traces


def plot_posterior_bars():
    fig, axes = plt.subplots(2, 2, figsize=(17, 14))
    combos = [(1, "cond"), (1, "nocond"), (2, "cond"), (2, "nocond")]

    for ax, (sem, model) in zip(axes.flatten(), combos):
        subjects = pd.read_csv(DATA / f"grades_sem{sem}.csv",
                               index_col="student_id").columns.tolist()
        traces = load_traces(sem, model)
        if not traces:
            ax.set_title(f"sem{sem}/{model}: no traces", color="grey")
            continue

        n_item = len(subjects)
        y = np.arange(n_item)
        w = 0.27   # bar half-width per ratio
        order = None
        # use legacy mean for ordering if available, else current
        anchor = "legacy_sd" if "legacy_sd" in traces else next(iter(traces))
        means_anchor = traces[anchor].posterior["item_difficulty"].mean(
            ("chain", "draw")).values
        order = np.argsort(means_anchor)

        for offset, kind in zip([-w, 0, w], RATIO_KINDS):
            if kind not in traces:
                continue
            m = traces[kind].posterior["item_difficulty"].mean(("chain", "draw")).values
            ax.barh(y + offset, m[order], height=w * 0.9,
                    color=COLORS[kind], label=LABELS[kind], alpha=0.9,
                    edgecolor="black", lw=0.4)

        ax.set_yticks(y)
        ax.set_yticklabels([subjects[i] for i in order], fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_xlabel("posterior mean of item_difficulty")
        ax.set_title(f"sem {sem} · {model}", fontsize=12, weight="bold")
        ax.grid(True, axis="x", alpha=0.3)
        ax.legend(loc="lower right", fontsize=8.5)

    fig.suptitle("item_difficulty under three ratio formulations (chi-square prior, current f)",
                 fontsize=14, weight="bold")
    fig.tight_layout()
    out = PLOTS / "ratio_posterior_compare.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# (4) HDI forest-plot для sem1 — по одному на каждый вариант модели
# ---------------------------------------------------------------------------
def plot_hdi_forests():
    subjects = pd.read_csv(DATA / "grades_sem1.csv",
                           index_col="student_id").columns.tolist()
    n = len(subjects)

    for model in ("cond", "nocond"):
        traces = load_traces(1, model)
        if len(traces) < 1:
            print(f"no traces for sem1/{model}")
            continue

        # Order subjects by mean of the first available ratio
        anchor = "legacy_sd" if "legacy_sd" in traces else next(iter(traces))
        m_anchor = traces[anchor].posterior["item_difficulty"].mean(
            ("chain", "draw")).values
        order = np.argsort(m_anchor)

        fig, axes = plt.subplots(1, len(traces),
                                 figsize=(5 * len(traces) + 1, 0.55 * n + 2),
                                 sharey=True,
                                 gridspec_kw={"wspace": 0.05})
        if len(traces) == 1:
            axes = [axes]

        for ax, kind in zip(axes, [k for k in RATIO_KINDS if k in traces]):
            t = traces[kind]
            means = t.posterior["item_difficulty"].mean(("chain", "draw")).values
            hdi = az.hdi(t, var_names=["item_difficulty"],
                         hdi_prob=0.94)["item_difficulty"].values
            color = COLORS[kind]
            for yi, i in enumerate(order):
                ax.plot([hdi[i, 0], hdi[i, 1]], [yi, yi],
                        color=color, lw=2.5, alpha=0.85)
                ax.plot([hdi[i, 0], hdi[i, 0]], [yi - 0.18, yi + 0.18],
                        color=color, lw=2)
                ax.plot([hdi[i, 1], hdi[i, 1]], [yi - 0.18, yi + 0.18],
                        color=color, lw=2)
                ax.scatter([means[i]], [yi], color=color, s=72, zorder=3,
                           edgecolor="black", lw=0.6)
                ax.text(hdi[i, 1] + 0.01, yi,
                        f"{means[i]:.2f}",
                        va="center", fontsize=8.5)
            ax.set_xlim(0, 1)
            ax.axvline(0.5, color="grey", ls=":", lw=0.7)
            ax.set_xlabel("item_difficulty")
            ax.set_title(LABELS[kind], fontsize=11, weight="bold")
            ax.grid(True, axis="x", alpha=0.3)

        axes[0].set_yticks(np.arange(n))
        axes[0].set_yticklabels([subjects[i] for i in order], fontsize=10)

        fig.suptitle(
            f"Semester 1 · {model} · 94% HDI for item_difficulty under 3 ratios",
            fontsize=13, weight="bold", y=1.005,
        )
        fig.tight_layout()
        out = PLOTS / f"ratio_hdi_sem1_{model}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    plot_surfaces()
    plot_curves()
    plot_posterior_bars()
    plot_hdi_forests()
    print("Done.")
