"""Проверка t-статистик IC на перекрывающихся окнах.

Проблема: бэктест берёт тестовые даты с шагом 10 дней при горизонте до 90
баров. Значит, 9 из 10 наблюдений перекрываются — они делят общий кусок
будущего. Наивная формула se(IC) ≈ (1 − IC²/2)/√(N−3) это не учитывает и
завышает значимость примерно в √L раз, где L — длина перекрытия.

Здесь считается эффективная выборка через последовательность автокорреляций
попарного произведения (аналог Newey-West для ранговой корреляции):

    N_eff = N / (1 + 2·Σ ρ_k,  k = 1..L)
    t_eff = IC · √N_eff

Запуск:
    python experiments/ic_significance.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "logs", "backtest_observations.csv")

PREDS = ["kronos_ret", "timesfm_ret", "ensemble_ret",
         "mean_revert_ret", "momentum_20_ret"]


def load():
    spec = importlib.util.spec_from_file_location(
        "m", os.path.join(ROOT, "timesfm_kronos_compare.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def spearman(a, b):
    if len(a) < 5 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    ra = pd.Series(a).rank().to_numpy(float)
    rb = pd.Series(b).rank().to_numpy(float)
    c = np.corrcoef(ra, rb)[0, 1]
    return float(c) if np.isfinite(c) else np.nan


def overlap_length(step, horizon):
    """Сколько соседних наблюдений делят будущее окно."""
    return max(int(np.ceil(horizon / step)) - 1, 1)


def acf_sum(x, lags):
    """Сумма автокорреляций 1..lags с коррекцией на конечность ряда."""
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    denom = np.dot(x, x)
    if denom == 0:
        return 0.0
    total = 0.0
    for k in range(1, lags + 1):
        if k >= n:
            break
        rho = np.dot(x[:-k], x[k:]) / denom * (n / (n - k))
        total += rho
    return total


def effective_n(ic, series, lags):
    """N_eff по последовательности автокорреляций попарного произведения.

    N_eff ограничен сверху числом наблюдений: эффективная выборка не может
    превышать фактическую. Без этой крышки оценка вырождается — у дискретных
    рядов (например, кросс-секционных по 3 тикерам) сумма автокорреляций
    уходит к −1/2, factor → 0 и N_eff формально улетает в десятки тысяч,
    давая t_eff в сотни при IC ≈ 0.0005. Это численная дегенерация, а не
    результат.
    """
    n = len(series)
    s = acf_sum(series, lags)
    factor = max(1.0 + 2.0 * s, 1e-6)
    n_eff = min(n / factor, float(n))
    t_eff = ic * np.sqrt(n_eff) if np.isfinite(ic) else np.nan
    return n_eff, t_eff


def main(df=None, step=10):
    m = load()
    if df is None:
        df = pd.read_csv(CSV)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    print("=" * 84)
    print("t-статистика IC с учётом перекрытия окон (Newey-West-подобная поправка)")
    print("=" * 84)
    print(f"шаг тестовых дат = {step} дней; NaN — t наивный, t_eff — поправленный\n")

    hdr = (f"{'предиктор':<17}{'h':>4}{'N':>6}{'IC':>8}{'t_naive':>9}"
           f"{'N_eff':>8}{'t_eff':>8}{'вердикт':>12}")
    print(hdr)
    print("-" * len(hdr))

    verdicts = {}
    for h in sorted(df["horizon"].unique()):
        sub_all = df[df["horizon"] == h]
        L = overlap_length(step, h)
        for p in PREDS:
            ok = sub_all[sub_all[p].notna() & sub_all["realized_ret"].notna()]
            if ok.empty:
                continue

            # Один согласованный ряд: ранги предсказания и факта по всему
            # пулу. Раньше IC усреднялся по тикерам, а t считался по пулу —
            # величины разные, сравнивать их бессмысленно.
            ra = pd.Series(ok[p].values).rank().to_numpy(float)
            rb = pd.Series(ok["realized_ret"].values).rank().to_numpy(float)
            ic = spearman(ok[p].values, ok["realized_ret"].values)
            prods = (ra - ra.mean()) * (rb - rb.mean())
            n_eff, t_eff = effective_n(ic, prods, L)
            t_naive = ic * np.sqrt(max(len(ok) - 3, 1)) if np.isfinite(ic) else np.nan

            sig = ("значим" if abs(t_eff) >= 2 else "не значим")
            print(f"{p:<17}{h:>4}{len(ok):>6}{ic:>8.3f}{t_naive:>9.2f}"
                  f"{n_eff:>8.1f}{t_eff:>8.2f}{sig:>12}")
            verdicts[(p, h)] = (ic, t_eff)
        print(f"{'':17}  (перекрытие L = {L} соседних наблюдений)")
        print()

    print("=" * 84)
    print("ИТОГ")
    print("=" * 84)
    strong = {k: v for k, v in verdicts.items() if abs(v[1]) >= 2}
    if not strong:
        print("Ни один предиктор не пережил поправку на перекрытие окон.")
        print("Прежние «значимые» t были завышены примерно в √L раз.")
    else:
        print("Пережили поправку:")
        for (p, h), (ic, t) in sorted(strong.items(), key=lambda kv: -abs(kv[1][1])):
            print(f"  {p:<17} h={h:<3} IC={ic:+.3f}  t_eff={t:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)