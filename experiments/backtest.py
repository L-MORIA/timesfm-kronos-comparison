"""Этап 2 Roadmap: walk-forward бэктест без утечки будущего.

Вопрос, который он отвечает: имеют ли вообще смысл цифры, которые скрипт
производит дважды в день.

Устройство одной точки
----------------------
Тестовая дата `T`, горизонт `h`:

1. Окно = последние WINDOW_BARS баров, **заканчивающихся на T включительно**.
   Прогноз делается из последнего закрытия окна, как в проде.
2. Считаются `last`, `mean`, `sigma_d` строго по этому окну.
3. Kronos и TimesFM дают прогноз на h баров вперёд.
4. Реализованная доходность берётся из `close[T+h]` — **после** прогона,
   отдельно, из той же выборки. Модели она не передаётся.

Все предсказатели выражаются в одном масштабе — ожидаемая доходность в %,
чтобы сравнивать их между собой и с фактом по одной метрике.

Предикторы
----------
kronos        прогноз Kronos
timesfm       прогноз TimesFM 2.5
ensemble      среднее (kronos + timesfm) / 2
random_walk   0% — цена не изменится
mean_revert   (mean_window − last) / last — «вернусь к среднему»
momentum_20   (last − close[−21]) / close[−21] — импульс за ~месяц

mean_revert — ключевая базовая линия: именно этим, судя по диагностике
z_excess, занимается Kronos. Если Kronos её не обыгрывает — он не добавляет
ничего, кроме стоимости прогона.

Запуск:
    python experiments/backtest.py                  # по умолчанию
    python experiments/backtest.py --step 5         # плотнее
    python experiments/backtest.py --quick          # короткая проверка
    python experiments/backtest.py --horizons 30
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
import zlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = _load("m", os.path.join(ROOT, "timesfm_kronos_compare.py"))
r = _load("r", os.path.join(ROOT, "experiments", "data_recon.py"))

WINDOW_BARS = 112            # столько баров видит модель (≈HISTORY_DAYS)
FROM_DATE = "2021-01-01"
# Закрытие рынка 2022-02-25 .. 2022-03-24: горизонт через него не
# интерпретируется (audit, этап 1). Исключаем ВСЁ окно до 2022-05-25 —
# не только сам разрыв, но и бары после него, пока не наберётся 60 баров
# истории (иначе ранние окна не восстанавливаются).
EXCLUDE = [("2022-02-25", "2022-05-25")]
GAP_MAX_DAYS = 7             # разрыв в торгах длиннее — горизонт «перепрыгивает»
OUT_CSV = os.path.join(ROOT, "logs", "backtest_observations.csv")


def _seed_for(ticker, date, h) -> int:
    """Детерминированный seed НА ТОЧКУ.

    Раньше seed переустанавливался на одно и то же значение перед каждой
    точкой, поэтому шум сэмплирования Kronos был ОДИНАКОВЫМ во всех 435
    наблюдениях, а для одной даты шумовые компоненты на h=30/60/90 были
    коррелированы на 1.0 (audit 2026-10-25, m9). Теперь seed зависит от
    (тикер, дата, горизонт) — розыгрыш разный, но воспроизводимый, поэтому
    --resume по-прежнему даёт те же числа.
    """
    return (m.SEED + zlib.crc32(f"{ticker}|{date}|{h}".encode("utf-8"))) % (2 ** 31)


def excluded(date) -> bool:
    d = pd.Timestamp(date)
    return any(pd.Timestamp(a) <= d <= pd.Timestamp(b) for a, b in EXCLUDE)


def spearman_ic(pred: np.ndarray, real: np.ndarray) -> float:
    """Ранговая корреляция Спирмена. Без scipy: ранг + Пирсон — то же самое.

    venv проекта намеренно минимален, scipy там нет, а тащить 60 МБ ради
    одной функции не нужно.
    """
    if len(pred) < 5:
        return np.nan
    if np.std(pred) == 0 or np.std(real) == 0:
        return np.nan
    rp = pd.Series(pred).rank().to_numpy(dtype=float)
    rr = pd.Series(real).rank().to_numpy(dtype=float)
    c = np.corrcoef(rp, rr)[0, 1]
    return float(c) if np.isfinite(c) else np.nan


def t_stat_ic(pred: np.ndarray, real: np.ndarray) -> float:
    """t-статистика для IC через перестановочную оценку (без scipy)."""
    ic = spearman_ic(pred, real)
    if not np.isfinite(ic):
        return np.nan
    n = len(pred)
    # Приближение Фишера: se(IC) ≈ (1 - IC²/2) / sqrt(n-3)
    se = (1 - ic ** 2 / 2) / np.sqrt(max(n - 3, 1))
    return ic / se if se else np.nan


def test_dates(df: pd.DataFrame, step: int, horizons: list[int]):
    """Индексы тестовых дат: нужны WINDOW_BARS слева и max(horizons) справа."""
    n = len(df)
    max_h = max(horizons)
    out = []
    i = WINDOW_BARS - 1
    while i + max_h < n:
        out.append(i)
        i += step
    return out


def save_rows(rows: list[dict], path: str) -> None:
    """Инкрементально дописать наблюдения в CSV.

    Первый прогон писал CSV только в самом конце. Фоновая задача имеет
    жёсткий лимит 3600 с, и прогон на шаге 10 (~3 ч) был убит по таймауту
    уже после того, как SBERP отработал целиком — 435 наблюдений пропали.
    Теперь состояние на диске после каждой точки.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8")


def run(args):
    import torch

    horizons = args.horizons
    print("=" * 78)
    print("WALK-FORWARD БЭКТЕСТ — есть ли прогностическая сила?")
    print("=" * 78)
    print(f"  окно модели      : {WINDOW_BARS} баров")
    print(f"  период           : с {FROM_DATE}")
    print(f"  шаг тестовых дат : {args.step} торговых дней")
    print(f"  горизонты        : {horizons} баров")
    print(f"  исключено        : {EXCLUDE}")
    print(f"  seed             : {m.SEED}")
    print()

    rows = []
    done_keys = set()
    if args.resume and os.path.isfile(OUT_CSV):
        prev = pd.read_csv(OUT_CSV)
        rows = prev.to_dict("records")
        done_keys = {(r["ticker"], str(r["date"]), int(r["horizon"])) for r in rows}
        print(f"[RESUME] загружено {len(rows)} наблюдений, "
              f"пропускаем уже посчитанные (ticker, date, horizon)")
    else:
        if os.path.isfile(OUT_CSV):
            os.remove(OUT_CSV)
    print()

    print("[INIT] Загрузка моделей (один раз на весь прогон)...")
    t0 = time.time()
    kp = m.load_kronos()
    tf = m.load_timesfm()
    print(f"[INIT] готово за {time.time() - t0:.1f} с\n")

    t_start = time.time()
    n_gap_skipped = 0

    for ticker in args.tickers:
        print(f"{'='*78}\n{ticker}")
        df, _n_raw = r.fetch_full(ticker)
        if df is None:
            print("  нет данных")
            continue
        # Схему нормализует сам основной модуль — свой экземпляр колонок
        # раньше приводил к KeyError: 'amount'.
        df = m.normalize_candles(df, ticker)
        if df is None:
            continue
        df = df[df["begin"] >= pd.Timestamp("2019-01-01")].reset_index(drop=True)

        # Разрывы в торгах: горизонт, перекрывающий длинный разрыв, не
        # интерпретируется (апрель 2022 — 27 календарных дней без торгов).
        # Такие точки исключаем целиком, а не оставляем NaN в realized.
        gaps = df["begin"].diff().dt.days.fillna(0).to_numpy()

        idxs = test_dates(df, args.step, horizons)
        idxs = [i for i in idxs if df["begin"].iloc[i] >= pd.Timestamp(FROM_DATE)]
        idxs = [i for i in idxs if not excluded(df["begin"].iloc[i])]
        n_all = len(idxs)
        idxs = [i for i in idxs
                if not all((ticker, str(df["begin"].iloc[i].date()), h) in done_keys
                           for h in horizons)]
        print(f"  баров в выборке : {len(df)}  ({df['begin'].iloc[0].date()} …)")
        print(f"  тестовых дат    : {len(idxs)}"
              + (f" (из {n_all}, часть уже посчитана)" if len(idxs) != n_all else ""))
        print()
        if not idxs:
            print("  все точки уже посчитаны — пропускаем тикер\n")
            continue

        t_t = time.time()
        for n_done, i in enumerate(idxs, 1):
            T = df["begin"].iloc[i]
            win = df.iloc[i - WINDOW_BARS + 1: i + 1].copy()
            last, mean, sigma_d = m.window_stats(win)
            weekend = m.has_weekend_sessions(win)
            vals = win["close"].values.astype(np.float32)

            for h in horizons:
                seg = gaps[i + 1: i + h + 1]
                if len(seg) and float(seg.max()) > GAP_MAX_DAYS:
                    n_gap_skipped += 1
                    continue
                torch.manual_seed(_seed_for(ticker, T.date(), h))
                try:
                    # Реальные даты будущих баров: в бэктесте календарь известен,
                    # и календарная аппроксимация future_bars() сдвигает метки
                    # времени Kronos на 2–6 дней (audit 2026-10-25, M1).
                    kout = m.kronos_predict(
                        kp, win, min(h, m.KRONOS_MAX_PRED_LEN), weekend,
                        y_ts_override=df["begin"].iloc[i + 1: i + 1 + h].values,
                    )
                except Exception as e:                       # noqa: BLE001
                    print(f"  [{T.date()} h={h}] Kronos сбой: {e}")
                    kout = None
                try:
                    tout = m.timesfm_forecast(tf, vals, h)
                except Exception as e:                       # noqa: BLE001
                    print(f"  [{T.date()} h={h}] TimesFM сбой: {e}")
                    tout = None

                k_pred = float(kout[-1]) if kout is not None else np.nan
                t_pred = float(tout[-1]) if tout is not None else np.nan
                k_ret = (k_pred / last - 1) * 100 if np.isfinite(k_pred) else np.nan
                t_ret = (t_pred / last - 1) * 100 if np.isfinite(t_pred) else np.nan
                # isfinite проверяем по КАЖДОМУ значению, а не по их сумме:
                # k=+inf и t=-inf дают NaN в сумме, и оба ответа терялись бы
                # (audit 2026-10-25, m5).
                if np.isfinite(k_ret) and np.isfinite(t_ret):
                    ens = float(np.nanmean([k_ret, t_ret]))
                else:
                    ens = np.nan

                realized = (df["close"].iloc[i + h] / last - 1) * 100
                span = (df["begin"].iloc[i + h] - T).days
                mom_ref = float(df["close"].iloc[max(i - 20, 0)])

                rows.append({
                    "ticker": ticker, "date": T.date(), "horizon": h,
                    "weekend": weekend, "last": last, "mean": mean,
                    "sigma_d": sigma_d, "cal_span_days": span,
                    "kronos_ret": k_ret, "timesfm_ret": t_ret,
                    "ensemble_ret": ens,
                    "random_walk_ret": 0.0,
                    "mean_revert_ret": (mean / last - 1) * 100,
                    "momentum_20_ret": (last / mom_ref - 1) * 100,
                    "realized_ret": realized,
                })

            # Пишем на диск после каждой точки: фоновый лимит 3600 с может
            # оборвать прогон в любой момент.
            save_rows(rows, OUT_CSV)

            if n_done % args.progress_every == 0 or n_done == len(idxs):
                el = time.time() - t_t
                rate = el / n_done
                eta = rate * (len(idxs) - n_done)
                print(f"  {n_done:>4}/{len(idxs)}  {rate:.2f} с/точка  "
                      f"осталось ~{eta/60:.1f} мин")

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    total = time.time() - t_start
    print(f"\n{'='*78}")
    print(f"Наблюдений: {len(out)}   Время: {total/60:.1f} мин")
    if n_gap_skipped:
        print(f"Пропущено точек с горизонтом через длинный разрыв торгов: "
              f"{n_gap_skipped}")
    print(f"Сохранено: {OUT_CSV}")
    return out


def summarize(df: pd.DataFrame):
    preds = ["kronos_ret", "timesfm_ret", "ensemble_ret",
             "random_walk_ret", "mean_revert_ret", "momentum_20_ret"]
    print(f"\n{'='*78}")
    print("IC (ранговая корреляция предсказания с фактом) и t-статистика")
    print("=" * 78)
    print(f"{'предиктор':<18}{'h':>4}{'N':>7}{'IC':>9}{'t':>8}"
          f"{'hit%':>8}{'MAE':>9}")
    for h in sorted(df["horizon"].unique()):
        sub = df[df["horizon"] == h]
        for p in preds:
            ok = sub[sub[p].notna() & sub["realized_ret"].notna()]
            if ok.empty:
                continue
            ic = spearman_ic(ok[p].values, ok["realized_ret"].values)
            t = t_stat_ic(ok[p].values, ok["realized_ret"].values)
            sign_ok = ok["realized_ret"].abs() > 0.01      # игнор плоских движений
            if sign_ok.any():
                hit = (np.sign(ok.loc[sign_ok, p])
                       == np.sign(ok.loc[sign_ok, "realized_ret"])).mean() * 100
            else:
                hit = np.nan
            mae = (ok[p] - ok["realized_ret"]).abs().mean()
            print(f"{p:<18}{h:>4}{len(ok):>7}{ic:>9.3f}{t:>8.2f}"
                  f"{hit:>8.1f}{mae:>9.2f}")
        print()

    print(f"{'='*78}")
    print("По эпохам (режим календаря) — средний IC на тикер*горизонт")
    print("=" * 78)
    for era, label in ((False, "до 2025-03 (только будни)"),
                       (True, "с 2025-03 (7 дней/неделю)")):
        sub = df[df["weekend"] == era]
        if sub.empty:
            continue
        print(f"\n  {label}   N={len(sub)}")
        for p in preds:
            ics = []
            for (tk, h), grp in sub.groupby(["ticker", "horizon"]):
                g = grp[grp[p].notna() & grp["realized_ret"].notna()]
                if len(g) >= 5:
                    ics.append(spearman_ic(g[p].values, g["realized_ret"].values))
            if ics:
                print(f"    {p:<18}IC = {np.nanmean(ics):+.3f}  "
                      f"(по {len(ics)} сериям)")

    print(f"\n{'='*78}")
    print("Kronos против базовой линии mean_revert — прямой тест")
    print("=" * 78)
    d = df[df["kronos_ret"].notna() & df["mean_revert_ret"].notna()].copy()
    d["delta"] = d["kronos_ret"] - d["mean_revert_ret"]
    for h in sorted(d["horizon"].unique()):
        s = d[d["horizon"] == h]
        if s.empty:
            continue
        corr = (spearman_ic(s["kronos_ret"].values, s["mean_revert_ret"].values)
                if np.isfinite(s["kronos_ret"].std() or np.nan)
                and s["mean_revert_ret"].std() else np.nan)
        mad = s["delta"].abs().mean()
        print(f"  h={h:<4} corr(Kronos, mean_revert) = {corr:+.3f}   "
              f"среднее|разница прогнозов| = {mad:.2f} п.п.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=10,
                    help="шаг между тестовыми датами, торговых дней")
    ap.add_argument("--horizons", type=int, nargs="+", default=[30, 60, 90])
    ap.add_argument("--tickers", type=str, nargs="+", default=["SBERP", "GAZP", "LKOH"])
    ap.add_argument("--quick", action="store_true",
                    help="быстрая проверка: крупный шаг, один горизонт")
    ap.add_argument("--resume", action="store_true",
                    help="дописать к существующему CSV, пропуская посчитанное")
    ap.add_argument("--progress-every", type=int, default=25)
    ap.add_argument("--summary-only", action="store_true",
                    help="не прогонять, только пересчитать по сохранённому CSV")
    args = ap.parse_args()
    if args.quick:
        args.step = 40
        args.horizons = [30]
        args.resume = False

    if args.summary_only:
        summarize(pd.read_csv(OUT_CSV))
        return 0

    df = run(args)
    summarize(df)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)