"""Этап 1 (продолжение): проверка режимов календаря на исторических окнах.

Скрипт отвечает на три вопроса, критичных для дизайна бэктеста:

1. Когда MOEX включил выходные сессии — `has_weekend_sessions()` обязан
   переключаться сам на исторических окнах.
2. Есть ли разрывы календаря (закрытия рынка) длиной > 10 дней, которые
   сломают walk-forward.
3. Совпадает ли календарь, который скрипт выдаёт Kronos, с реальностью:
   N баров должны покрывать N дней.

Запуск:
    python experiments/recon_calendar.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = _load("m", os.path.join(ROOT, "timesfm_kronos_compare.py"))
r = _load("r", os.path.join(ROOT, "experiments", "data_recon.py"))


def main():
    print("=== 1. Когда начались выходные сессии MOEX ===")
    df, _ = r.fetch_full("LKOH")
    df["begin"] = pd.to_datetime(df["begin"])
    wk = df[df["begin"].dt.weekday >= 5]
    early = wk[wk["begin"] < "2025-01-01"]
    late = wk[wk["begin"] >= "2025-01-01"]
    print(f"  выходных баров до 2025 : {len(early)}  (единичные, не сессия)")
    print(f"  выходных баров с 2025  : {len(late)}")
    if len(late):
        first = late["begin"].min()
        print(f"  первый выходной бар   : {first.date()} {first.day_name()}")
        cnt = late.groupby(late["begin"].dt.to_period("M")).size()
        head = ", ".join(f"{k}={v}" for k, v in list(cnt.items())[:6])
        print(f"  по месяцам (начало)   : {head}")

    print()
    print("=== 2. has_weekend_sessions() на исторических окнах ===")
    print("   (окно = последние 112 баров, как в проде)")
    print("   окно до           weekend  вых.баров  freq   90 баров = ?")
    for end in ["2021-06-01", "2022-03-01", "2023-06-01", "2024-09-01",
                "2025-03-01", "2025-10-01", "2026-10-14"]:
        sub = df[df["begin"] <= end].tail(112)
        if sub.empty:
            continue
        flag = m.has_weekend_sessions(sub)
        n_wk = int((sub["begin"].dt.weekday >= 5).sum())
        fb = m.future_bars(sub["begin"].iloc[-1], 90, flag)
        span = (fb.iloc[-1] - sub["begin"].iloc[-1]).days
        freq = "D" if flag else "B"
        mark = "" if 85 <= span <= 95 else "  <-- ОТКЛОНЕНИЕ"
        print(f"   {end:<17} {str(flag):<8} {n_wk:<10} {freq:<6} "
              f"{span:>3} дней{mark}")

    print()
    print("=== 3. Разрывы календаря > 10 дней ===")
    for t in ["SBERP", "GAZP", "LKOH"]:
        d, _x = r.fetch_full(t)
        d["begin"] = pd.to_datetime(d["begin"])
        d = d.sort_values("begin").reset_index(drop=True)
        gap = d["begin"].diff().dt.days
        bad = d[gap > 10]
        print(f"  {t}: {len(bad)}")
        for i in bad.index:
            prev_d = d["begin"].loc[i - 1].date()
            print(f"     {int(gap[i]):>3} дн: {prev_d} -> {d['begin'].loc[i].date()}")

    print()
    print("=== 4. Качество выборки с 2021 ===")
    print("   тикер   баров_2021+  дублей  NaN  close<=0")
    for t in ["SBERP", "GAZP", "LKOH"]:
        d, _x = r.fetch_full(t)
        d["begin"] = pd.to_datetime(d["begin"])
        sub = d[d["begin"] >= "2021-01-01"]
        dup = int(sub["begin"].duplicated().sum())
        nan = int(sub[["open", "high", "low", "close"]].isna().sum().sum())
        bad = int((pd.to_numeric(sub["close"], errors="coerce").fillna(0) <= 0).sum())
        print(f"   {t:<8} {len(sub):<12} {dup:<8} {nan:<5} {bad}")


if __name__ == "__main__":
    sys.exit(main() or 0)