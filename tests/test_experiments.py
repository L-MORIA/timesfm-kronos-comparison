"""Проверки инвариантов измерительного аппарата (`experiments/`).

Зачем отдельный файл: до 25.10.2026 в `tests/test_units.py` было НОЛЬ
ссылок на `experiments/` — все числа трёх отчётов (панель, walk-forward,
power) считались кодом без единой проверки. Сдвиг на единицу в
`overlap_length` или подмена shift() на shift(-1) в предикторах изменили бы
вердикт молча.

Запуск:
    python tests/test_experiments.py
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ics = load("ics_t", os.path.join(EXP, "ic_significance.py"))
pd_mod = load("pd_t", os.path.join(EXP, "panel_daily.py"))
bt = load("bt_t", os.path.join(EXP, "backtest.py"))
pilot = load("pilot_t", os.path.join(EXP, "intraday_pilot.py"))
pw = load("pw_t", os.path.join(EXP, "power_analysis.py"))

failures = []


def check(name, actual, expected):
    ok = actual == expected
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"         got={actual!r} expected={expected!r}")
        failures.append(name)


def close(name, actual, expected, tol=1e-9):
    ok = actual is not None and abs(float(actual) - float(expected)) <= tol
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"         got={actual!r} expected={expected!r} (tol={tol})")
        failures.append(name)


print("=== утечек нет: предикторы не читают будущее (audit 2026-10-25, MB5) ===")


def _prices(n=220, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.012, n))), index=idx)


P = _prices()
T = P.index[150]
COLS = ["mean_revert_w60", "momentum_20", "momentum_5", "volnorm_mom_20"]

base = pd_mod.predictor_table(P)
future_broken = P.copy()
future_broken.iloc[151:] *= 3.0            # цены ПОСЛЕ T — ломаем
broken = pd_mod.predictor_table(future_broken)

check("предикторы до T не меняются от правки будущего",
      base.loc[:T, COLS].equals(broken.loc[:T, COLS]), True)
check("realized[T] меняется (тест не тривиален)",
      bool(base.loc[T, "realized"] != broken.loc[T, "realized"]), True)
check("realized[T] = возврат T->T+1",
      close("realized[T] = возврат T→T+1",
            base.loc[T, "realized"],
            float(P.iloc[151] / P.iloc[150] - 1), 1e-9) is None, True)

print("=== разрыв в торгах: realized не «h = 1» через длинный разрыв ===")
gap_idx = pd.DatetimeIndex(["2022-02-23", "2022-02-24", "2022-02-25",
                             "2022-03-24", "2022-03-25", "2022-03-28"])
gap_px = pd.Series([100.0, 101.0, 102.0, 110.0, 111.0, 112.0], index=gap_idx)
g = pd_mod.predictor_table(gap_px)
check("дата перед разрывом 27 дней: realized = NaN",
      bool(pd.isna(g.loc[pd.Timestamp("2022-02-25"), "realized"])), True)
check("дата после разрыва: realized заполнен",
      bool(pd.notna(g.loc[pd.Timestamp("2022-03-24"), "realized"])), True)

print("=== ic_significance: длина перекрытия и N_eff ===")
check("overlap_length(h=30, step=10)", ics.overlap_length(10, 30), 2)
check("overlap_length(h=60, step=10)", ics.overlap_length(10, 60), 5)
check("overlap_length(h=90, step=10)", ics.overlap_length(10, 90), 8)
check("overlap_length(h=30, step=30) — окна не перекрываются",
      ics.overlap_length(30, 30), 1)

rng = np.random.default_rng(11)
white = rng.normal(size=500)
close("acf_sum на независимом ряде ≈ 0", ics.acf_sum(white, 10), 0.0, 0.35)
ar = np.zeros(500)
for i in range(1, 500):
    ar[i] = 0.8 * ar[i - 1] + rng.normal()
check("acf_sum на AR(1) с rho=0.8 > 0", ics.acf_sum(ar, 10) > 1.0, True)

n_eff, t_eff = ics.effective_n(0.05, white, 10)
check("N_eff не превышает N (крышка)", n_eff <= len(white) + 1e-9, True)
check("N_eff на независимом ряде ≈ N", abs(n_eff - len(white)) < 60, True)
close("t_eff = IC·√N_eff (корреляция по пулу)", t_eff, 0.05 * np.sqrt(n_eff), 1e-9)
close("mde = alpha_z/√N_eff", ics.mde(n_eff), 2.0 / np.sqrt(n_eff), 1e-9)

print("=== spearman_ic: ранг + Пирсон, как scipy ===")
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
y = np.array([2.0, 1.0, 4.0, 3.0, 6.0, 5.0])
manual = float(np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1])
close("spearman_ic == ручной ранг+Пирсон", bt.spearman_ic(x, y), manual, 1e-12)
close("spearman_ic на монотонной связи = 1", bt.spearman_ic(x, 2 * x + 1), 1.0, 1e-12)
check("spearman_ic при <5 точках -> NaN", bool(np.isnan(bt.spearman_ic(x[:3], y[:3]))), True)

print("=== panel_daily: кросс-спецман и спред Roll ===")
close("cross_spearman на 3 точках считается (не NaN)",
      float(pd_mod.cross_spearman(np.array([1.0, 2.0, 3.0]),
                                   np.array([1.0, 2.0, 3.0]))), 1.0, 1e-12)
# Искусственный bid-ask bounce: цена «пилотирует» вокруг уровня с шагом
# вверх-вниз — так Roll и должен увидеть спред.
rng2 = np.random.default_rng(3)
level = 100.0
path = []
for _ in range(400):
    level += rng2.normal(0, 0.3)
    path.append(level + 0.25 * np.sign(rng2.normal()))
roll = pd_mod.roll_spread_daily(pd.Series(path))
check("Roll на bid-ask bounce даёт спред > 0", roll > 0.0, True)
smooth = pd.Series(100.0 + np.arange(300) * 0.01)
roll_smooth = pd_mod.roll_spread_daily(smooth)
close("Roll на гладком ряде ≈ 0", roll_smooth, 0.0, 1e-9)

print("=== intraday_pilot: ICC между окнами одного дня ===")
import datetime as _dt
rows = []
for d in range(20):
    day = _dt.date(2026, 8, 3) + _dt.timedelta(days=d)
    base_val = 0.5 + 0.01 * d
    for k in range(6):
        ts = _dt.datetime(day.year, day.month, day.day, 10, 5 * k)
        rows.append((ts, base_val))          # внутри дня значение постоянно
icc_same, m_same, _ = pilot.icc_within_day(rows)
check("ICC при одинаковых окнах внутри дня ≈ 1", icc_same > 0.95, True)
close("окон в дне = 6", m_same, 6.0, 1e-9)

rows_noise = []
rng3 = np.random.default_rng(5)
for d in range(20):
    day = _dt.date(2026, 8, 3) + _dt.timedelta(days=d)
    for k in range(6):
        ts = _dt.datetime(day.year, day.month, day.day, 10, 5 * k)
        rows_noise.append((ts, float(rng3.normal())))
icc_noise, _, _ = pilot.icc_within_day(rows_noise)
check("ICC на независимых окнах ≈ 0", icc_noise < 0.3, True)

print("=== power_analysis: mde_for и required_n согласованы ===")
mde_at = pw.mde_for(pw.required_n(0.10, 0.05, 0.80))
close("mde_for(required_n(0.10)) ≈ 0.10", mde_at, 0.10, 0.01)

print()
if failures:
    print(f"FAILED: {len(failures)} -> {failures}")
    sys.exit(1)
print("ALL CHECKS PASSED")
