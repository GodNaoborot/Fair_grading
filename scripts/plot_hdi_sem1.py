"""HDI по предметам семестра 1: LEGACY-стек vs CURRENT-стек.

По одному графику на каждый вариант модели (cond, nocond). Общая ось Y,
чтобы визуально сравнивать левую (legacy) и правую (current) панели.
"""
import _paths   # noqa: F401  (должен идти первым: подкручивает sys.path)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import arviz as az

from _paths import DATA, TRACES, PLOTS


subjects = pd.read_csv(DATA / "grades_sem1.csv",
                       index_col="student_id").columns.tolist()
n = len(subjects)


def stats(trace):
    da = trace.posterior["item_difficulty"]
    means = da.mean(("chain", "draw")).values
    hdi = az.hdi(trace, var_names=["item_difficulty"],
                 hdi_prob=0.94)["item_difficulty"].values
    return means, hdi


def plot_side_by_side(model_name, out_path):
    t_old = az.from_netcdf(TRACES / f"trace_sem1_{model_name}_old.nc")
    t_new = az.from_netcdf(TRACES / f"trace_sem1_{model_name}_chi.nc")
    m_o, h_o = stats(t_old)
    m_n, h_n = stats(t_new)

    order = np.argsort(m_o)
    y = np.arange(n)

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(14, 0.7 * n + 2),
        sharey=True, gridspec_kw={"wspace": 0.05},
    )

    def draw(ax, means, hdi, color, title):
        for yi, i in enumerate(order):
            ax.plot([hdi[i, 0], hdi[i, 1]], [yi, yi],
                    color=color, lw=2.5, alpha=0.8)
            ax.plot([hdi[i, 0], hdi[i, 0]], [yi - 0.15, yi + 0.15], color=color, lw=2)
            ax.plot([hdi[i, 1], hdi[i, 1]], [yi - 0.15, yi + 0.15], color=color, lw=2)
            ax.scatter([means[i]], [yi], color=color, s=70, zorder=3,
                       edgecolor="black", lw=0.6)
            ax.text(hdi[i, 1] + 0.01, yi,
                    f"{means[i]:.2f} [{hdi[i,0]:.2f}, {hdi[i,1]:.2f}]",
                    va="center", fontsize=8.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("item_difficulty")
        ax.axvline(0.5, color="grey", ls=":", lw=0.7)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.grid(True, axis="x", alpha=0.3)

    draw(ax_l, m_o, h_o, "steelblue", "LEGACY (old f, ratio s/(s+d), Beta(2,2))")
    draw(ax_r, m_n, h_n, "tomato",    "CURRENT (new f, 1−d(1−s), chi-square prior)")

    ax_l.set_yticks(y)
    ax_l.set_yticklabels([subjects[i] for i in order], fontsize=10)

    fig.suptitle(
        f"Семестр 1 · модель «{model_name}» · 94% HDI для item_difficulty "
        f"(legacy → current; предметы отсортированы по legacy mean)",
        fontsize=13, weight="bold", y=1.005,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    plot_side_by_side("cond",   PLOTS / "hdi_sem1_cond_legacy_vs_current.png")
    plot_side_by_side("nocond", PLOTS / "hdi_sem1_nocond_legacy_vs_current.png")
    print("Done.")
