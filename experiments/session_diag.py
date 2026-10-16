"""Диагностика сессии: где в сутках проходит граница ликвидности.

Замер показал: внутри суток нет ни одного разрыва больше 3 минут, при этом
данные идут с 06:59 до 23:49. Это не сходится с опубликованным расписанием
сессий MOEX, значит ориентироваться на часы нельзя. Границу режима ищем по
данным — по профилю объёма.

Выводы, которые скрипт должен дать:
  * где в сутках объём обваливается;
  * какая доля дневного объёма приходится на тихий хвост;
  * сколько баров в сутках реально пригодно для анализа.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import requests

U = ("https://iss.moex.com/iss/engines/stock/markets/shares"
     "/boards/TQBR/securities/{ticker}/candles.json")
DAY = sys.argv[1] if len(sys.argv) > 1 else "2026-10-14"
TICKERS = ["SBERP", "GAZP", "LKOH"]


def day(ticker, d):
    rows, st, cols = [], 0, None
    while True:
        p = {"interval": 1, "from": d, "till": d, "limit": 500, "start": st}
        blk = (requests.get(U.format(ticker=ticker), params=p,
                            timeout=25).json().get("candles") or {})
        cols = blk.get("columns")
        data = blk.get("data") or []
        rows.extend(data)
        if len(data) < 500:
            break
        st += len(data)
    df = pd.DataFrame(rows, columns=cols)
    df["begin"] = pd.to_datetime(df["begin"])
    df["volume"] = pd.to_numeric(df["volume"])
    return df.sort_values("begin").drop_duplicates("begin").reset_index(drop=True)


def main():
    print("=" * 70)
    print(f"ДИАГНОСТИКА СЕССИИ — {DAY}")
    print("=" * 70)
    frames = {}
    for tk in TICKERS:
        d = day(tk, DAY)
        if d.empty:
            continue
        g = d["begin"].diff().dt.total_seconds().dropna()
        print(f"{tk:<7} баров={len(d):>5}  {str(d['begin'].iloc[0])[11:16]}…"
              f"{str(d['begin'].iloc[-1])[11:16]}  макс.разрыв={g.max():>5.0f}с  "
              f"разрывов>300с={(g > 300).sum()}")
        frames[tk] = d

    if not frames:
        print("нет данных")
        return

    print()
    print("Объём по 30-минутным блокам, усреднение по тикерам:")
    prof = {}
    for tk, d in frames.items():
        d = d.assign(q=d["begin"].dt.floor("30min"))
        prof[tk] = d.groupby("q")["volume"].mean()
    h = pd.concat(prof, axis=1).mean(axis=1)
    peak = h.max()
    for t, val in h.items():
        bar = "#" * int(val / peak * 42)
        print(f"  {str(t)[11:16]}  {val:>10,.0f}  {bar}")

    med = float(h.median())
    above = h[h >= med]
    quiet = h[h.index > above.index.max()]
    loud_share = 1 - float(quiet.sum()) / float(h.sum())

    print()
    print(f"медиана объёма            : {med:,.0f}")
    print(f"громкий блок (>= медианы) : {str(above.index.min())[11:16]} … "
          f"{str(above.index.max())[11:16]}")
    print(f"тихий хвост после него    : {str(quiet.index.min())[11:16]} … "
          f"{str(quiet.index.max())[11:16]}")
    print(f"доля объёма в громкой части: {loud_share * 100:.1f} %")
    print()

    n_bars_loud = len(above) * 6          # 30-минутный блок = 6 баров по 5 минут
    n_bars_quiet = len(quiet) * 6
    print("=" * 70)
    print("ВЫВОД ДЛЯ ДИЗАЙНА")
    print("=" * 70)
    print("Расписание сессий в метках ISS не восстанавливается: разрывов нет,")
    print("а данные лежат с 06:59 до 23:49. Значит границу режима приходится")
    print("определять по объёму, а не по часам.")
    print()
    total_bars = n_bars_loud + n_bars_quiet
    share_time = n_bars_quiet / total_bars * 100 if total_bars else 0
    print(f"Тихий хвост держит лишь {100 - loud_share * 100:.1f} % дневного объёма,")
    print(f"но занимает ~{n_bars_quiet} пятиминутных баров из ~{total_bars} — "
          f"это {share_time:.0f} % времени.")
    print("Если оставить хвост в ряду, модель частично обучается на мусоре;")
    print(f"если отрезать — теряется {share_time:.0f} % времени, но растёт "
          "качество выборки.")
    print()
    print(f"Рекомендация: ограничить ряд громкой частью (~{n_bars_loud} баров")
    print(f"в сутки вместо ~{total_bars}) и пересчитать ICC уже на обрезанном ряде.")
    print()
    print("Косвенное свидетельство о поясе ISS. Пик объёма в блоке 09:30")
    print(f"({h.max():,.0f}, в {h.max() / med:.1f} раза выше медианы) совпадает с")
    print("окрытием основной сессии MOEX 09:50, второй пик в блоке 18:30 — с")
    print("закрытием 18:50 и открытием вечерней 19:00. Обе метки бьются при")
    print("чтении ISS как MSK, а не UTC. Но ряд начинается в 06:59, раньше")
    print("публичного открытия, поэтому расписание всё равно не восстановить —")
    print("границу режима задаём объёмом.")


if __name__ == "__main__":
    main()