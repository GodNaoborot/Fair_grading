"""Парсер выгрузки оценок из Excel в две матрицы (семестр 1 и семестр 2).

Ячейка исходной таблицы выглядит как `4(12.01)3(25.06)` — оценка и дата
сдачи; по месяцу даты предмет относится к первому или второму семестру.
"""
import re

import pandas as pd


def parse(input_path: str = 'Grades.xls', sheet_name: str = "23-24 1к 1п"):
    df = pd.read_excel(input_path, sheet_name=sheet_name)
    to_drop = []
    for key in df.keys():
        i = 1
        while not isinstance(df[key][i], str):
            i += 1
            if i == len(df[key]):
                break
        if i == len(df[key]):
            to_drop.append(key)
        elif df[key][i][0] in '+-':
            to_drop.append(key)
    df.drop(to_drop, axis=1, inplace=True)

    df_semester1 = {}
    df_semester2 = {}
    for key in df.keys():
        i = 1
        while not isinstance(df[key][i], str):
            i += 1
        dates = [int(j.split(".")[1]) for j in re.findall(r'\((.*?)\)', df[key][i])]
        grade_cell = [j[0] for j in re.findall(r'.\(', df[key][i])]
        j = 1
        while grade_cell[j - 1] not in '345' and j < len(grade_cell):
            j += 1
        if j >= len(grade_cell):
            date_2 = []
        else:
            date_2 = [dates[j]]
        dates_valid = [dates[0]] + date_2
        if any(item in dates_valid for item in (11, 12, 1, 2, 3, 4)):
            df_semester1[key] = []
            for cell in df[key]:
                if isinstance(cell, float):
                    df_semester1[key].append(cell)
                else:
                    if len(cell) < 2:
                        df_semester1[key].append(float("nan"))
                        continue
                    grade_cell = [j[0] for j in re.findall(r'.\(', cell)]
                    j = 0
                    while j < len(grade_cell) and grade_cell[j] not in '345':
                        j += 1
                    j = min(j, len(grade_cell) - 1)
                    if grade_cell[j].isdigit():
                        df_semester1[key].append(int(grade_cell[j]))
                    else:
                        df_semester1[key].append(float("nan"))
        if any(item in dates for item in (5, 6, 7, 8, 9, 10)):
            df_semester2[key] = []
            for cell in df[key]:
                if isinstance(cell, float):
                    df_semester2[key].append(cell)
                else:
                    if len(cell) < 2:
                        df_semester2[key].append(float("nan"))
                        continue
                    grade_cell = [j[0] for j in re.findall(r'.\(', cell)]
                    if grade_cell[-1].isdigit():
                        df_semester2[key].append(int(grade_cell[-1]))
                    else:
                        df_semester2[key].append(float("nan"))

    df_semester1 = pd.DataFrame(df_semester1)
    df_semester2 = pd.DataFrame(df_semester2)

    return df_semester1.to_numpy(dtype='float32'), df_semester2.to_numpy(dtype='float32')


if __name__ == "__main__":
    print(parse())