"""Сравнение legacy-стека (старые f + старый ratio + Beta(2,2)) и current-стека
(новые f + новый ratio + chi-square приор).

Сохраняет два сводных PNG в results/plots/. Legacy-трассы (trace_sem*_*_old.nc)
сэмплились со старыми степенями f^76^12, ratio s/(s+d), Beta(2,2) приором;
current-трассы помечены суффиксом _chi.
"""
import _paths   # noqa: F401  (должен идти первым: подкручивает sys.path)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import arviz as az

from _paths import DATA, TRACES, PLOTS


# ----- Functions for the top rows -----
def f2_legacy(x): return (1 - x) ** 4
def f5_legacy(x): return f2_legacy(1 - x)
def f3_legacy(x):
    t1 = 1 / (x * np.sqrt(3 * np.pi))
    e  = -0.5 * ((np.log(x) + np.sqrt(1.5) - np.log(3)) ** 2)
    return t1 * np.exp(e)
def f4_legacy(x): return f3_legacy(1 - x)

def f2(x): return ((1 - x) ** 3) / ((1 - x) ** 3 + 10 * x ** 2)
def f5(x): return f2(1 - x)
def f3(x): return (1 - x) * (1 - f2(x) - f5(x))
def f4(x): return f3(1 - x)


def probs(f2, f3, f4, f5, x):
    v = np.stack([f2(x), f3(x), f4(x), f5(x)])
    return v / v.sum(axis=0, keepdims=True)


def load_subjects(sem):
    return pd.read_csv(DATA / f"grades_sem{sem}.csv",
                       index_col="student_id").columns.tolist()


def load_shifts():
    shifts = {}
    for sem in (1, 2):
        for model in ("cond", "nocond"):
            t_old = az.from_netcdf(TRACES / f"trace_sem{sem}_{model}_old.nc")
            t_new = az.from_netcdf(TRACES / f"trace_sem{sem}_{model}_chi.nc")
            m_o = t_old.posterior["item_difficulty"].mean(("chain", "draw")).values
            m_n = t_new.posterior["item_difficulty"].mean(("chain", "draw")).values
            shifts[(sem, model)] = (m_o, m_n, m_n - m_o)
    return shifts


def plot_summary():
    fig = plt.figure(figsize=(16, 16))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.4], hspace=0.55, wspace=0.30)
    x = np.linspace(1e-3, 1 - 1e-3, 500)

    pairs = [
        ("f_2 (P=2)", f2_legacy, f2),
        ("f_3 (P=3)", f3_legacy, f3),
        ("f_4 (P=4)", f4_legacy, f4),
        ("f_5 (P=5)", f5_legacy, f5),
    ]
    for i, (title, fl, fc) in enumerate(pairs):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(x, fl(x), color="steelblue", lw=2.2, label="legacy")
        ax.plot(x, fc(x), color="tomato",    lw=2.2, ls="--", label="current")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    po = probs(f2_legacy, f3_legacy, f4_legacy, f5_legacy, x)
    pn = probs(f2, f3, f4, f5, x)
    labels = ["P(grade=2)", "P(grade=3)", "P(grade=4)", "P(grade=5)"]
    for i in range(4):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(x, po[i], color="steelblue", lw=2.2, label="legacy")
        ax.plot(x, pn[i], color="tomato",    lw=2.2, ls="--", label="current")
        ax.set_title(labels[i] + "  (normalized)", fontsize=11)
        ax.set_xlabel("x")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    shifts = load_shifts()
    combos = [(1, "cond"), (1, "nocond"), (2, "cond"), (2, "nocond")]
    for k, (sem, model) in enumerate(combos):
        ax = fig.add_subplot(gs[2, k])
        subjects = load_subjects(sem)
        m_o, m_n, d = shifts[(sem, model)]
        order = np.argsort(d)
        y = np.arange(len(subjects))
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in d[order]]
        ax.barh(y, d[order], color=colors, alpha=0.85, edgecolor="black", lw=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels([subjects[i] for i in order], fontsize=9)
        ax.axvline(0, color="black", lw=0.7)
        ax.set_xlabel("Δ item_difficulty  (current − legacy)")
        ax.set_title(f"Сем {sem} · {model}", fontsize=12, weight="bold")
        ax.grid(True, axis="x", alpha=0.3)
        for yi, di in zip(y, d[order]):
            ax.text(di + (0.003 if di >= 0 else -0.003), yi,
                    f"{di:+.03f}", va="center",
                    ha="left" if di >= 0 else "right", fontsize=8.5)
        lim = max(abs(d).max() * 1.4, 0.05)
        ax.set_xlim(-lim, lim)

    fig.suptitle(
        "Legacy → current stack: формы f, нормированные вероятности и сдвиг item_difficulty",
        fontsize=14, weight="bold", y=0.995,
    )
    out = PLOTS / "summary_legacy_vs_current.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    fig2, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (sem, model) in zip(axes, combos):
        subjects = load_subjects(sem)
        m_o, m_n, _ = shifts[(sem, model)]
        ax.scatter(m_o, m_n, s=120, c="#d62728", edgecolor="black", zorder=3)
        for i, name in enumerate(subjects):
            short = name[:22] + ("…" if len(name) > 22 else "")
            ax.annotate(f"  {short}", (m_o[i], m_n[i]), fontsize=8, alpha=0.9, zorder=4)
        lo = max(0, min(m_o.min(), m_n.min()) - 0.05)
        hi = min(1, max(m_o.max(), m_n.max()) + 0.05)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.6, label="y = x")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("item_difficulty — legacy")
        ax.set_ylabel("item_difficulty — current")
        ax.set_title(f"Сем {sem} · {model}", weight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)

    fig2.suptitle(
        "item_difficulty: current vs legacy (точки выше диагонали — «стал сложнее»)",
        fontsize=14, weight="bold", y=1.02,
    )
    fig2.tight_layout()
    out2 = PLOTS / "summary_scatter_legacy_vs_current.png"
    fig2.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close(fig2)
    print(f"saved {out2}")


if __name__ == "__main__":
    plot_summary()
