"""Этап 1 Roadmap: разведка данных для walk-forward бэктеста.

Отвечает на вопросы, без которых бэктест бессмысленно строить:

1. Сколько истории реально доступно по каждому тикеру.
2. Сколько торговых дней в году — и когда появились выходные сессии
   (от этого зависит has_weekend_sessions() на исторических окнах).
3. Есть ли разрывы, которые сломают walk-forward.
4. Есть ли дубликаты дат — и не теряет ли drop_duplicates реальные данные.
5. Не портится ли история ревизиями MOEX (риск look-ahead в бэктесте).

Запуск:
    python experiments/data_recon.py
"""
from __future__ import annotations

import datetime as dt
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import requests

MOEX = ("https://iss.moex.com/iss/engines/stock/markets/shares"
        "/boards/TQBR/securities/{t}/candles.json")
PAGE = 500
MAX_PAGES = 200
TICKERS = ["SBERP", "GAZP", "LKOH"]
FROM = "2000-01-01"


def fetch_full(ticker: str, frm: str = FROM, till: str | None = None):
    """Вся история дневных баров с пагинацией. Без фильтрации."""
    till = till or dt.date.today().strftime("%Y-%m-%d")
    rows, cols, start = [], None, 0
    for _ in range(MAX_PAGES):
        r = requests.get(MOEX.format(t=ticker), timeout=25, params={
            "interval": 24, "from": frm, "till": till,
            "limit": PAGE, "start": start})
        r.raise_for_status()
        blk = r.json().get("candles") or {}
        data = blk.get("data") or []
        if cols is None:
            cols = blk.get("columns") or []
        if not data:
            break
        rows.extend(data)
        if len(data) < PAGE:
            break
        start += len(data)
    if not rows:
        return None, 0
    df = pd.DataFrame(rows, columns=cols)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df, len(rows)


def report(ticker: str):
    df, n_raw = fetch_full(ticker)
    if df is None:
        print(f"[{ticker}] НЕТ ДАННЫХ")
        return None
    df["begin"] = pd.to_datetime(df["begin"])
    for c in ("open", "high", "low", "close", "volume", "value"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    raw_n = len(df)
    dupes = int(df["begin"].duplicated().sum())
    df = (df.drop_duplicates(subset=["begin"])
            .sort_values("begin").reset_index(drop=True))
    zero_close = int((df["close"] <= 0).sum())
    nan_total = int(df[["open", "high", "low", "close"]].isna().sum().sum())

    gaps = df["begin"].diff().dt.days.dropna()
    print(f"\n{'='*72}")
    print(f"{ticker}")
    print(f"{'='*72}")
    print(f"  сырых строк           : {raw_n}")
    print(f"  после дедупликации   : {len(df)}   (дубликатов дат: {dupes})")
    print(f"  период               : {df['begin'].iloc[0].date()} .. "
          f"{df['begin'].iloc[-1].date()}")
    print(f"  календарных лет      : "
          f"{(df['begin'].iloc[-1] - df['begin'].iloc[0]).days / 365.25:.1f}")
    print(f"  баров в год (среднее): {len(df) / max(((df['begin'].iloc[-1] - df['begin'].iloc[0]).days / 365.25), 0.01):.0f}")
    print(f"  close <= 0           : {zero_close}")
    print(f"  NaN в OHLC           : {nan_total}")

    gap_counts = Counter(gaps.astype(int).tolist())
    top = sorted(gap_counts.items(), key=lambda kv: -kv[0] * kv[1])[:6]
    print(f"  разрывы (дни: кол-во): "
          + ", ".join(f"{g}д×{c}" for g, c in sorted(gap_counts.items())[:8]))

    # ── Плотность и доля выходных по годам ──────────────────────
    per_year = df.groupby(df["begin"].dt.year).size()
    wknd = df[df["begin"].dt.weekday >= 5]
    wknd_per_year = wknd.groupby(wknd["begin"].dt.year).size()

    print(f"\n  год   баров   выходных   доля_выходных   max_gap")
    prev_year = None
    for y in sorted(per_year.index):
        if prev_year is not None and y - prev_year > 1:
            print(f"  --- пропуск лет: {prev_year+1}..{y-1} ---")
        prev_year = y
        if y < 2015 and y % 5:      # ранние годы покажем через один
            continue
        sub = df[df["begin"].dt.year == y]
        mg = int(sub["begin"].diff().dt.days.max())
        w = int(wknd_per_year.get(y, 0))
        print(f"  {y}  {per_year[y]:>5}   {w:>7}   {w/per_year[y]*100:>10.1f}%   {mg:>7}")
    return df


def main():
    print("Разведка данных MOEX ISS (дневные бары)")
    print(f"Запрошено с {FROM} по сегодня. Пагинация limit={PAGE}.")
    frames = {}
    for t in TICKERS:
        try:
            frames[t] = report(t)
        except Exception as e:                      # noqa: BLE001
            print(f"[{t}] ОШИБКА: {type(e).__name__}: {e}")

    # ── Сводка для бэктеста ─────────────────────────────────────
    print(f"\n\n{'='*72}")
    print("СВОДКА ДЛЯ БЭКТЕСТА")
    print(f"{'='*72}")
    print(f"{'тикер':<8}{'первый бар':<14}{'2021+':<8}{'баров/год':<11}{'выходных%':<11}")
    for t, df in frames.items():
        if df is None or df.empty:
            continue
        recent = df[df["begin"] >= "2021-01-01"]
        yrs = max(((df['begin'].iloc[-1] - df['begin'].iloc[0]).days / 365.25), 0.01)
        wknd_pct = (df["begin"].dt.weekday >= 5).mean() * 100
        print(f"{t:<8}{str(df['begin'].iloc[0].date()):<14}{len(recent):<8}"
              f"{len(df)/yrs:<11.0f}{wknd_pct:<11.1f}")

    if all(v is not None for v in frames.values()):
        n_recent = min(len(d[d["begin"] >= "2021-01-01"]) for d in frames.values())
        print(f"\n  Минимум баров с 2021 (все тикеры): {n_recent}")
        for step in (5, 10):
            print(f"  При шаге {step} торговых дней: "
                  f"~{n_recent // step} точек на тикер, "
                  f"~{n_recent // step * len(TICKERS)} прогонов всего")
        print("\n  Требование к горизонту 90 дней: нужно минимум 112 баров")
        print("  истории ПЕРЕД каждой тестовой датой (см. HISTORY_DAYS).")


if __name__ == "__main__":
    sys.exit(main() or 0)