"""Запрос к посчитанной модели: способность студента, сложность предмета,
ожидаемая оценка для пары.

Примеры
-------
    # всё про студента: способность + прогноз по каждому его предмету
    python scripts/query.py --sem 1 --student 42

    # всё про предмет: сложность + распределение оценок
    python scripts/query.py --sem 1 --subject "Математический анализ"

    # конкретная пара
    python scripts/query.py --sem 1 --student 42 --subject "Высшая алгебра"

    # список того, что вообще есть
    python scripts/query.py --sem 1 --list
"""
import _paths   # noqa: F401  (первым: добавляет src/ в sys.path)

import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import GRADES, RATIO_KINDS
from query import (
    HDI_PROB, load_grades, load_trace, resolve_student, resolve_subject,
    student_samples, item_samples, summarize, rank_of,
    predicted_grade_distribution, observed_grade, grade_table,
)
from _paths import DATA, TRACES, PLOTS


def build_parser():
    p = argparse.ArgumentParser(
        description="Апостериорные запросы к модели справедливых оценок.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Примеры")[1] if "Примеры" in __doc__ else None,
    )
    p.add_argument("--sem", type=int, choices=(1, 2), default=1,
                   help="номер семестра (по умолчанию 1)")
    p.add_argument("--student", help="student_id из data/grades_sem*.csv")
    p.add_argument("--subject", help="название предмета (можно часть названия)")
    p.add_argument("--model", choices=("cond", "nocond"), default="nocond",
                   help="вариант модели: с гейтом крайних оценок или без (по умолчанию nocond)")
    p.add_argument("--ratio", choices=RATIO_KINDS, default="sigmoid",
                   help="формула совпадения студент/предмет (по умолчанию sigmoid)")
    p.add_argument("--list", action="store_true",
                   help="показать доступных студентов и предметы и выйти")
    p.add_argument("--plot", action="store_true",
                   help="сохранить график постериора в results/plots/")
    return p


def print_header(text):
    print(f"\n{text}")
    print("-" * len(text))


def cmd_list(grades, sem):
    ids = grades.index.tolist()
    print(f"Семестр {sem}: {len(ids)} студентов, {len(grades.columns)} предметов")
    print(f"student_id: от {min(ids)} до {max(ids)}")
    print_header("Предметы")
    for j, name in enumerate(grades.columns):
        filled = int(grades.iloc[:, j].notna().sum())
        print(f"  [{j}] {name}   ({filled} оценок)")


def report_student(trace, grades, student_idx, sid, ratio_kind, gate):
    samples = student_samples(trace, student_idx)
    stats = summarize(samples)
    rank, total = rank_of(trace, "student_ability", student_idx)

    print_header(f"Студент #{sid}")
    print(f"  способность s : {stats}")
    print(f"  место в потоке: {rank} из {total} (по возрастанию способности)")

    row = grades.iloc[student_idx]
    taken = [(j, name) for j, name in enumerate(grades.columns) if not np.isnan(row.iloc[j])]
    if not taken:
        print("  оценок нет — предметы не сдавались")
        return samples

    print_header("Прогноз по сданным предметам")
    print(f"  {'предмет':<45} {'факт':>5} {'E[оценка]':>10} {'P(факт)':>9}")
    for j, name in taken:
        probs = predicted_grade_distribution(trace, student_idx, j, ratio_kind, gate)
        actual = int(row.iloc[j])
        expected = float(probs @ GRADES)
        p_actual = float(probs[GRADES == actual][0])
        short = name if len(name) <= 45 else name[:42] + "..."
        print(f"  {short:<45} {actual:>5} {expected:>10.2f} {p_actual:>9.3f}")
    return samples


def report_subject(trace, grades, item_idx, subject):
    samples = item_samples(trace, item_idx)
    stats = summarize(samples)
    rank, total = rank_of(trace, "item_difficulty", item_idx)

    print_header(f"Предмет «{subject}»")
    print(f"  сложность d   : {stats}")
    print(f"  место по сложности: {rank} из {total} (по возрастанию)")

    col = grades.iloc[:, item_idx].dropna().astype(int)
    print(f"  сдавали: {len(col)} студентов, средняя оценка {col.mean():.2f}")
    counts = col.value_counts().reindex(GRADES, fill_value=0)
    dist = "  ".join(f"{g}: {counts[g]:>3} ({counts[g] / len(col):.0%})" for g in GRADES)
    print(f"  фактическое распределение — {dist}")
    return samples


def report_pair(trace, grades, student_idx, sid, item_idx, subject,
                ratio_kind, gate):
    probs = predicted_grade_distribution(trace, student_idx, item_idx, ratio_kind, gate)
    actual = observed_grade(grades, student_idx, item_idx)

    print_header(f"Студент #{sid} × «{subject}»")
    print(grade_table(probs, actual).to_string())
    print(f"\n  E[оценка] = {probs @ GRADES:.2f}")
    if actual is None:
        print("  предмет этим студентом не сдавался — сравнить не с чем")
    else:
        print(f"  фактическая оценка = {actual}, модель давала ей P = {probs[GRADES == actual][0]:.3f}")
    return probs


def save_plot(out_path, title, student=None, subject=None, pair_probs=None):
    panels = sum(x is not None for x in (student, subject, pair_probs))
    fig, axes = plt.subplots(1, panels, figsize=(5.5 * panels, 4.2), squeeze=False)
    axes = axes[0]
    k = 0

    for samples, label, color in ((student, "способность s", "#1f77b4"),
                                  (subject, "сложность d", "#d62728")):
        if samples is None:
            continue
        ax = axes[k]; k += 1
        stats = summarize(samples)
        ax.hist(samples, bins=50, color=color, alpha=0.8, density=True)
        ax.axvspan(stats.hdi_low, stats.hdi_high, color=color, alpha=0.15,
                   label=f"{int(HDI_PROB * 100)}% HDI")
        ax.axvline(np.mean(samples), color="black", ls="--", lw=1.2, label="среднее")
        ax.set_xlim(0, 1)
        ax.set_xlabel(label)
        ax.set_ylabel("плотность")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    if pair_probs is not None:
        ax = axes[k]
        ax.bar(GRADES, pair_probs, color="#2ca02c", alpha=0.85,
               edgecolor="black", lw=0.6)
        for g, p in zip(GRADES, pair_probs):
            ax.text(g, p + 0.01, f"{p:.2f}", ha="center", fontsize=10)
        ax.set_xticks(GRADES)
        ax.set_ylim(0, min(1.0, max(pair_probs) * 1.25))
        ax.set_xlabel("оценка")
        ax.set_ylabel("апостериорная вероятность")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nграфик сохранён: {out_path}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    grades = load_grades(DATA, args.sem)

    if args.list:
        cmd_list(grades, args.sem)
        return 0

    if not args.student and not args.subject:
        print("Укажите --student и/или --subject (или --list, чтобы посмотреть, что есть).",
              file=sys.stderr)
        return 2

    trace = load_trace(TRACES, args.sem, args.model, args.ratio)
    gate = args.model == "cond"
    print(f"трасса: сем {args.sem} · {args.model} · ratio={args.ratio}")

    student_idx = sid = item_idx = subject = None
    s_samples = d_samples = pair_probs = None

    if args.student:
        student_idx, sid = resolve_student(grades, args.student)
    if args.subject:
        item_idx, subject = resolve_subject(grades, args.subject)

    if student_idx is not None and item_idx is None:
        s_samples = report_student(trace, grades, student_idx, sid, args.ratio, gate)
    elif item_idx is not None and student_idx is None:
        d_samples = report_subject(trace, grades, item_idx, subject)
    else:
        s_samples = student_samples(trace, student_idx)
        d_samples = item_samples(trace, item_idx)
        print_header(f"Студент #{sid}")
        print(f"  способность s : {summarize(s_samples)}")
        print_header(f"Предмет «{subject}»")
        print(f"  сложность d   : {summarize(d_samples)}")
        pair_probs = report_pair(trace, grades, student_idx, sid, item_idx,
                                 subject, args.ratio, gate)

    if args.plot:
        parts = [f"sem{args.sem}", args.model, args.ratio]
        if sid is not None:
            parts.append(f"student{sid}")
        if item_idx is not None:
            parts.append(f"item{item_idx}")
        title = " · ".join(filter(None, [
            f"Семестр {args.sem}",
            f"студент #{sid}" if sid is not None else None,
            f"«{subject}»" if subject else None,
        ]))
        save_plot(PLOTS / ("query_" + "_".join(parts) + ".png"), title,
                  student=s_samples, subject=d_samples, pair_probs=pair_probs)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        sys.exit(1)
