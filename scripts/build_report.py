"""Сборка PDF-отчёта из report/report.typ.

Таблицы в отчёте читаются напрямую из results/*.csv средствами Typst, поэтому
сборка после нового прогона обновляет цифры без правки текста.

    python scripts/build_report.py
"""
import _paths   # noqa: F401  (первым: добавляет src/ в sys.path)

import sys

from _paths import ROOT, RESULTS

REQUIRED = (
    "diagnostics.csv",
    "loo_compare_sem1.csv",
    "loo_compare_sem2.csv",
    "predictive_scores.csv",
    "ppc_summary.csv",
)


def main():
    missing = [name for name in REQUIRED if not (RESULTS / name).exists()]
    if missing:
        print("Не хватает результатов: " + ", ".join(missing), file=sys.stderr)
        print("Сначала: python scripts/diagnostics.py && python scripts/validate.py",
              file=sys.stderr)
        return 1

    try:
        import typst
    except ImportError:
        print("Нет пакета typst. Установите: pip install typst", file=sys.stderr)
        return 1

    source = ROOT / "report" / "report.typ"
    output = ROOT / "report" / "fair_grading.pdf"
    # root обязателен: без него Typst не выпускает чтение за пределы каталога
    # с исходником и не видит results/
    typst.compile(str(source), output=str(output), root=str(ROOT))
    print(f"собрано: {output}  ({output.stat().st_size // 1024} КБ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
