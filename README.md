# Fair_grading

Байесовская IRT-подобная модель оценок студентов (шкала 2–5).
Для каждого студента оценивается латентная **способность** `s ∈ [0, 1]`,
для каждого предмета — латентная **сложность** `d ∈ [0, 1]`.

## Структура репозитория

```
Fair_grading/
├── data/                       # сырые данные (gitignored)
│   ├── grades_sem1.csv
│   ├── grades_sem2.csv
│   ├── Grades.xls
│   └── chi_priors.json         # генерится compute_chi_priors.py
├── src/
│   ├── utils.py                # модели (cond / nocond) + chi-square приоры
│   └── parser_lib.py           # парсер таблицы оценок
├── notebooks/
│   ├── algorithm.ipynb         # короткий пример использования модели
│   ├── test_synthetic_data.ipynb
│   ├── chi_square.ipynb        # подбор chi-square score сложности
│   ├── parser_test.ipynb
│   └── model_comparison.ipynb  # ⭐ сводный ноутбук со сравнением трёх ratio
├── scripts/                    # точки входа: запускать как python scripts/<имя>.py
│   ├── _paths.py               # общие пути проекта (импортится первым)
│   ├── compute_chi_priors.py
│   ├── run_mcmc_chi_prior.py
│   ├── plot_ratio_comparison.py
│   ├── plot_f_functions.py
│   ├── plot_changes_summary.py
│   └── plot_hdi_sem1.py
├── results/                    # все артефакты (gitignored)
│   ├── traces/                 # *.nc — arviz InferenceData
│   ├── plots/                  # *.png
│   └── logs/                   # *.log
├── presentation/               # LaTeX-выкладка
├── requirements.txt
├── README.md
└── .gitignore
```

## Модель

В `src/utils.py` определены две модели:

- `create_bipartite_bayesian_network_cond` — с «гейтом» на крайние оценки
  (две дополнительные сигмоиды толкают сильных студентов к 5, слабых — к 2).
  Гейт работает в координатах `ratio` (порог 0.9 для пятёрки, 0.1 для двойки).
- `create_bipartite_bayesian_network_nocond` — без гейта, все четыре вероятности
  определяются только функциями `f_2..f_5`.

Обе модели принимают **скаляр или per-item массив** `item_alpha` / `item_beta`,
так что приор из `chi_square_priors` подключается напрямую.

### Формула правдоподобия

```
# совпадение между студентом и предметом, выбирается параметром ratio_kind:
ratio = s / (s + d)            # "legacy_sd"  — симметричный
      | 1 − d · (1 − s)        # "current"    — текущий дефолт
      | sigmoid(k · (s − d))   # "sigmoid"    — гладкий, k=5 по умолчанию

f_2(x) = (1−x)^3 / ((1−x)^3 + 10·x²)
f_5(x) = f_2(1 − x)
f_3(x) = (1−x) · (1 − f_2(x) − f_5(x))
f_4(x) = f_3(1 − x)
```

`f_2 + f_3 + f_4 + f_5 ≡ 1` по построению (отдельная нормировка не нужна).

### Семантика угловых точек

| s | d | сценарий | ожидаемая оценка |
|---|---|---|---|
| 0 | 1 | бездарь + сложный | 2 |
| 0.5 | 0.5 | средний + средний | 3-4 |
| 1 | 0 | умный + лёгкий | 5 |
| 0 | 0 | бездарь + лёгкий | 3-4 (точно не 5) |
| 1 | 1 | умный + сложный | 3-4 (точно не 5) |

Полное сравнение трёх вариантов `ratio` по этим точкам и постериорам —
в `notebooks/model_comparison.ipynb`.

## Chi-square приоры

`scripts/compute_chi_priors.py` читает `data/grades_sem*.csv`, считает
эмпирическое распределение оценок 2/3/4/5 по каждому предмету и выводит
per-item Beta-приор, центр которого совпадает с chi-square score сложности.
Концентрация фиксирована `α + β = 4` (та же «сила», что у `Beta(2, 2)`).

## Типичный сценарий запуска

```bash
# 0. (один раз) установить зависимости
pip install -r requirements.txt

# 1. посчитать chi-square приоры → data/chi_priors.json
python scripts/compute_chi_priors.py

# 2. запустить MCMC (12 трасс: 3 ratio × 2 модели × 2 семестра)
python scripts/run_mcmc_chi_prior.py

# 3. собрать графики сравнения трёх ratio
python scripts/plot_ratio_comparison.py

# 4. (опционально) посмотреть на сами f-функции
python scripts/plot_f_functions.py
```

После шага 3 откройте `notebooks/model_comparison.ipynb` — все картинки
встроены inline.

## Настройки MCMC по умолчанию

`chains=4, tune=2000, draws=1000, cores=2, target_accept=0.95`.
Этого достаточно, чтобы получить `rhat ≤ 1.01` для большинства параметров
на текущих матрицах (138 × 6 и 138 × 10).
