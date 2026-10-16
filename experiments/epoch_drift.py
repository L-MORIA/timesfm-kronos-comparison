"""Эпохальный перелом IC: дрейф рынка или собственное содержание моделей?

В walk-forward бэктесте IC резко меняет знак на границе 2025-03: до неё все
предикторы отрицательны (-0.09), после — положительны (+0.12…+0.21). Прежде
чем читать это как «модели заработали», нужно проверить, не является ли знак
IC следствием направления рынка в эпохе.

Механика: mean_revert_ret = (mean/last − 1)·100 — предсказание без участия
модели, чистый откат к среднему окна. В периоде падения всё, что предсказывает
рост, получает отрицательный IC просто потому, что рынок падал. Поэтому
«IC < 0» здесь может значить «рынок шёл вниз», а не «модель ошибается».

Проверяем тремя способами:
  1. дрейф эпохи — средняя реализованная доходность и доля положительных;
  2. IC внутри эпох с Newey-West-поправкой (окна по-прежнему перекрываются);
  3. остаток Kronos минус mean_revert — добавляет ли модель что-то сверх
     механического отката.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "logs", "backtest_observations.csv")

EPOCH_SPLIT = "2025-03-01"   # MOEX включил выходные сессии
STEP = 10                    # шаг тестовых дат, баров
PREDS = ["kronos_ret", "timesfm_ret", "ensemble_ret",
         "mean_revert_ret", "momentum_20_ret"]


def load_significance():
    spec = importlib.util.spec_from_file_location(
        "ics", os.path.join(ROOT, "experiments", "ic_significance.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ics"] = mod
    spec.loader.exec_module(mod)
    return mod


def ic_and_t(ics, ok, p, L):
    """IC, N_eff и t_eff по согласованному пулу рядов (как в ic_significance)."""
    ra = pd.Series(ok[p].values).rank().to_numpy(float)
    rb = pd.Series(ok["realized_ret"].values).rank().to_numpy(float)
    ic = ics.spearman(ok[p].values, ok["realized_ret"].values)
    prods = (ra - ra.mean()) * (rb - rb.mean())
    n_eff, t_eff = ics.effective_n(ic, prods, L)
    return ic, n_eff, t_eff


def cross_sectional_series(df, p):
    """Ряд кросс-секционных IC: по одной величине на дату.

    Обе величины — и предсказание, и реализация — трендуют по времени, если
    рынок идёт в одну сторону. На разрезе по дате (2025-03) такой тренд целиком
    попадает в одну эпоху и выдаёт себя за IC. Единственный способ этого
    избежать — убрать общий уровень даты: ранги считаются внутри (дата,
    горизонт) по трём тикерам, и коррелируются уже центрированные ранги.

    Ряд индексирован датой, поэтому перекрытие будущих окон между соседними
    датами здесь НАСТОЯЩЕЕ и N_eff надо считать по нему, а не по числу строк.
    Внутри даты всего 3 точки — мощность низкая, и это ограничение метода,
    а не результата.
    """
    vals = []
    for _, g in df.groupby(["date", "horizon"]):
        g = g[g[p].notna() & g["realized_ret"].notna()]
        if len(g) < 3:
            continue
        rx = g[p].rank().to_numpy(float)
        ry = g["realized_ret"].rank().to_numpy(float)
        if np.std(rx) == 0 or np.std(ry) == 0:
            continue
        c = float(np.corrcoef(rx, ry)[0, 1])
        if np.isfinite(c):
            vals.append(c)
    return np.asarray(vals, float)


def main():
    ics = load_significance()
    df = pd.read_csv(CSV)
    df["date"] = pd.to_datetime(df["date"])
    df["epoch"] = np.where(df["date"] < EPOCH_SPLIT,
                           "до 2025-03", "с 2025-03")

    print("=" * 86)
    print("ДРЕЙФ ЭПОХИ — если рынок шёл вниз, отрицательный IC предсказывает рост")
    print("=" * 86)
    print(f"{'эпоха':<14}{'h':>4}{'N':>7}{'средн.realized%':>18}"
          f"{'медиана%':>11}{'доля > 0':>11}")
    print("-" * 86)
    for epoch, sub_e in df.groupby("epoch"):
        for h in sorted(df["horizon"].unique()):
            r = sub_e[sub_e["horizon"] == h]["realized_ret"]
            print(f"{epoch:<14}{h:>4}{len(r):>7}{r.mean():>18.2f}"
                  f"{r.median():>11.2f}{(r > 0).mean() * 100:>10.1f}%")
    print()

    print("=" * 86)
    print("IC ПО ЭПОХАМ — с Newey-West-поправкой на перекрытие окон")
    print("=" * 86)
    print(f"{'предиктор':<17}{'h':>4}{'эпоха':<14}{'N':>7}{'IC':>8}"
          f"{'N_eff':>8}{'t_eff':>8}{'вердикт':>12}")
    print("-" * 86)
    for h in sorted(df["horizon"].unique()):
        L = ics.overlap_length(STEP, h)
        for epoch in ("до 2025-03", "с 2025-03"):
            sub_h = df[(df["horizon"] == h) & (df["epoch"] == epoch)]
            for p in PREDS:
                ok = sub_h[sub_h[p].notna() & sub_h["realized_ret"].notna()]
                if ok.empty:
                    continue
                ic, n_eff, t_eff = ic_and_t(ics, ok, p, L)
                sig = "значим" if abs(t_eff) >= 2 else "не значим"
                print(f"{p:<17}{h:>4}{epoch:<14}{len(ok):>7}{ic:>8.3f}"
                      f"{n_eff:>8.1f}{t_eff:>8.2f}{sig:>12}")
        print(f"{'':17}  (перекрытие L = {L})")
        print()

    print("=" * 86)
    print("СКОЛЬКО МОДЕЛЬ ДОБАВЛЯЕТ СВЕРХ МЕХАНИЧЕСКОГО ОТКАТА")
    print("=" * 86)
    print("corr(kronos_ret, mean_revert_ret) — 1.0 означает «модель = среднее окна»")
    print(f"{'h':>4}{'эпоха':<14}{'corr':>8}{'IC остатка':>13}"
          f"{'средн.|K−MR| п.п.':>21}")
    print("-" * 86)
    for h in sorted(df["horizon"].unique()):
        L = ics.overlap_length(STEP, h)
        for epoch in ("до 2025-03", "с 2025-03"):
            sub_h = df[(df["horizon"] == h) & (df["epoch"] == epoch)].copy()
            ok = sub_h[sub_h["kronos_ret"].notna()
                       & sub_h["mean_revert_ret"].notna()
                       & sub_h["realized_ret"].notna()]
            if ok.empty:
                continue
            corr = ok["kronos_ret"].corr(ok["mean_revert_ret"])
            ok["residual"] = ok["kronos_ret"] - ok["mean_revert_ret"]
            ic_res, n_eff, t_eff = ic_and_t(ics, ok, "residual", L)
            gap = (ok["kronos_ret"] - ok["mean_revert_ret"]).abs().mean()
            print(f"{h:>4}{epoch:<14}{corr:>8.3f}{ic_res:>13.3f}{gap:>21.2f}")
    print()

    print("=" * 86)
    print("IC БЕЗ ОБЩЕГО ВРЕМЕННОГО ТРЕНДА (ранги внутри даты, 3 тикера)")
    print("=" * 86)
    print("Это главная проверка: без неё тренд рынка выдаёт себя за IC.")
    print(f"{'предиктор':<17}{'h':>4}{'эпоха':<14}{'N':>7}{'IC':>8}{'N_eff':>8}"
          f"{'t_eff':>8}{'вердикт':>12}")
    print("-" * 86)
    for h in sorted(df["horizon"].unique()):
        L = ics.overlap_length(STEP, h)
        for epoch in ("до 2025-03", "с 2025-03"):
            sub_h = df[(df["horizon"] == h) & (df["epoch"] == epoch)]
            for p in PREDS:
                ser = cross_sectional_series(sub_h, p)
                if len(ser) < 10:
                    continue
                ic = float(ser.mean())
                n_eff, t_eff = ics.effective_n(ic, ser, L)
                sig = "значим" if abs(t_eff) >= 2 else "не значим"
                print(f"{p:<17}{h:>4}{epoch:<14}{len(ser):>7}{ic:>8.3f}"
                      f"{n_eff:>8.1f}{t_eff:>8.2f}{sig:>12}")
        print(f"{'':17}  (L = {L} перекрывающихся соседних дат)")
        print()

    print("=" * 86)
    print("IC ensemble ПО ТИКЕРАМ — эффект рыночный или одного бумаги")
    print("=" * 86)
    for h in sorted(df["horizon"].unique()):
        for epoch in ("до 2025-03", "с 2025-03"):
            line = []
            for tk in sorted(df["ticker"].unique()):
                sub = df[(df["horizon"] == h) & (df["epoch"] == epoch)
                         & (df["ticker"] == tk)]
                ok = sub[sub["ensemble_ret"].notna()
                         & sub["realized_ret"].notna()]
                if len(ok) < 5:
                    continue
                line.append(f"{tk}={ics.spearman(ok['ensemble_ret'].values, ok['realized_ret'].values):+.3f}")
            if line:
                print(f"  h={h:>3}  {epoch:<14} " + "  ".join(line))
    print()


if __name__ == "__main__":
    main()