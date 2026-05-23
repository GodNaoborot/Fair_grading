"""График семейства f_2..f_5 для legacy и текущих формул."""
import _paths   # noqa: F401  (должен идти первым: подкручивает sys.path)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _paths import PLOTS


# Legacy variant (kept for visual reference; the codebase no longer uses it).
def f2_legacy(x): return (1 - x) ** 4
def f5_legacy(x): return f2_legacy(1 - x)
def f3_legacy(x):
    t1 = 1 / (x * np.sqrt(3 * np.pi))
    e = -0.5 * ((np.log(x) + np.sqrt(1.5) - np.log(3)) ** 2)
    return t1 * np.exp(e)
def f4_legacy(x): return f3_legacy(1 - x)


# Current formulas (also in src/utils.py).
def f2(x): return ((1 - x) ** 3) / ((1 - x) ** 3 + 10 * x ** 2)
def f5(x): return f2(1 - x)
def f3(x): return (1 - x) * (1 - f2(x) - f5(x))
def f4(x): return f3(1 - x)


if __name__ == "__main__":
    x = np.linspace(1e-6, 1 - 1e-6, 500)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    pairs = [
        (f2_legacy, f2, "f_2  (оценка 2)"),
        (f3_legacy, f3, "f_3  (оценка 3)"),
        (f4_legacy, f4, "f_4  (оценка 4)"),
        (f5_legacy, f5, "f_5  (оценка 5)"),
    ]
    for ax, (fl, fc, name) in zip(axes.flatten(), pairs):
        ax.plot(x, fl(x), label="legacy", color="steelblue", lw=2)
        ax.plot(x, fc(x), label="current", color="tomato", ls="--", lw=2)
        ax.set_title(name)
        ax.set_xlabel("x = match between student & item")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = PLOTS / "f_functions_legacy_vs_current.png"
    plt.savefig(out, dpi=120)
    print(f"Saved {out}")

    print("\nNormalization check (current functions):")
    print(f'{"x":>5}  {"sum":>6}  {"f2":>6}  {"f3":>6}  {"f4":>6}  {"f5":>6}')
    for xi in (0.1, 0.3, 0.5, 0.7, 0.9):
        a, b, c, d = f2(xi), f3(xi), f4(xi), f5(xi)
        print(f"{xi:>5.1f}  {a+b+c+d:>6.4f}  {a:>6.4f}  {b:>6.4f}  {c:>6.4f}  {d:>6.4f}")
