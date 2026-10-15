"""Эксперимент: три способа посчитать сигнал на одних и тех же прогнозах.

Мотивация (audits/2026-10-14-audit.md, часть 3): Kronos притягивается к
средней цене окна истории, поэтому абсолютный порог ±2% превращает его
прогноз в «продажу» для любого актива, выросшего за окно. Проверяем,
отвязывает ли нормировка сигнал от уровня цены и прошлого тренда.

Схемы (для каждой модели и горизонта h):

  A abs      chg%     = (pred − last)/last · 100,  порог ±2%
            То, что сейчас в проде.

  B z_vol    (pred − last) / (σ_d · last · √h),   порог ±1σ
            Насколько ход прогноза велик относительно шума h-шаговой
            диффузии. Отвязывает сигнал от волатильности и горизонта.

  C z_excess (pred − mean) / (σ_d · last · √h),   порог ±1σ
            Отклонение от «механического отката к среднему». Если Kronos
            просто откатывается, здесь ≈0 — то есть модель не сказала
            ничего сверх арифметики окна.

σ_d — std дневных лог-доходностей окна.
last — последнее закрытие, mean — среднее закрытие окна.

Запуск:
    python experiments/signal_normalization.py
"""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "tkm", os.path.join(ROOT, "timesfm_kronos_compare.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

import numpy as np

ABS_THRESHOLD = 2.0      # %, как в проде
Z_THRESHOLD = 1.0        # σ


def window_stats(df):
    close = df["close"].astype(float)
    last = float(close.iloc[-1])
    mean = float(close.mean())
    rets = np.log(close / close.shift(1)).dropna()
    sigma_d = float(rets.std())
    return last, mean, sigma_d


def z_denom(sigma_d, last, horizon):
    return sigma_d * last * np.sqrt(horizon)


def classify(value, threshold):
    if value >= threshold:
        return "ПОКУПКА"
    if value <= -threshold:
        return "ПРОДАЖА"
    return "ДЕРЖАТЬ"


def main():
    import torch
    torch.manual_seed(m.SEED)

    print("[INIT] Загрузка моделей...")
    kp = m.load_kronos()
    tf = m.load_timesfm()

    rows = []
    for ticker in m.TICKERS:
        df = m.fetch_moex_candles(ticker)
        if df is None or len(df) < m.MIN_BARS:
            print(f"[{ticker}] пропущен: данных нет")
            continue
        last, mean, sigma_d = window_stats(df)
        weekend = m.has_weekend_sessions(df)
        values = df["close"].values.astype(np.float32)

        for horizon in m.HORIZONS:
            kp_out = m.kronos_predict(kp, df, min(horizon, m.KRONOS_MAX_PRED_LEN),
                                      weekend)
            tf_out = m.timesfm_forecast(tf, values, horizon)
            denom = z_denom(sigma_d, last, horizon)
            for name, arr in (("Kronos", kp_out), ("TimesFM", tf_out)):
                pred = float(arr[-1])
                rows.append({
                    "ticker": ticker, "horizon": horizon, "model": name,
                    "pred": pred, "last": last, "mean": mean,
                    "chg": (pred - last) / last * 100,
                    "z_vol": (pred - last) / denom,
                    "z_excess": (pred - mean) / denom,
                })
            print(f"  [{ticker}] h={horizon}: last={last:.2f} "
                  f"mean={mean:.2f} sigma_d={sigma_d*100:.2f}% "
                  f"denom={denom:.2f}")

    print()
    hdr = (f"{'тикер':<6}{'h':>4}{'модель':>9}{'прогноз':>11}{'chg%':>9}"
           f"{'z_vol':>9}{'z_exc':>9}  {'A ±2%':>9}{'B ±1σ':>9}{'C ±1σ':>9}")
    print("=== РЕЗУЛЬТАТЫ ===")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        a = classify(r["chg"], ABS_THRESHOLD)
        b = classify(r["z_vol"], Z_THRESHOLD)
        c = classify(r["z_excess"], Z_THRESHOLD)
        r["A"], r["B"], r["C"] = a, b, c
        print(f"{r['ticker']:<6}{r['horizon']:>4}{r['model']:>9}"
              f"{r['pred']:>11.2f}{r['chg']:>9.2f}"
              f"{r['z_vol']:>9.2f}{r['z_excess']:>9.2f}  "
              f"{a:>9}{b:>9}{c:>9}")

    print()
    print("=== СОГЛАСИЕ МОДЕЛЕЙ (пары Kronos/TimesFM на тикер×горизонт) ===")
    pairs = {}
    for i in range(0, len(rows), 2):
        k, t = rows[i], rows[i + 1]
        key = (k["ticker"], k["horizon"])
        pairs[key] = (k, t)
    for label, field in (("A abs ±2%", "A"), ("B z_vol ±1σ", "B"), ("C z_excess ±1σ", "C")):
        agree = sum(1 for k, t in pairs.values() if k[field] == t[field])
        same_dir = sum(1 for k, t in pairs.values()
                       if k[field] == t[field] != "ДЕРЖАТЬ")
        dist = {v: 0 for v in ("ПОКУПКА", "ПРОДАЖА", "ДЕРЖАТЬ")}
        for k, _ in pairs.values():
            dist[k[field]] += 1
        for _, t in pairs.values():
            dist[t[field]] += 1
        print(f"  {label:<16} совпадений {agree}/{len(pairs)}  "
              f"(в т.ч. направленных {same_dir})  "
              f"распределение Kronos+TimesFM: {dist}")

    print()
    print("=== ДИАГНОСТИКА: z_excess у Kronos (если ≈0 — это механика отката) ===")
    for i in range(0, len(rows), 2):
        k = rows[i]
        flag = "откат" if abs(k["z_excess"]) < 1 else "свой сигнал"
        print(f"  {k['ticker']:<6} h={k['horizon']:>3}  "
              f"z_vol={k['z_vol']:>7.2f}  z_excess={k['z_excess']:>7.2f}   {flag}")


if __name__ == "__main__":
    main()