"""TimesFM 2.5 + Kronos Mini — сравнение сигналов MOEX.

Загружает данные из MOEX ISS API (как Kronos), запускает обе модели,
сравнивает прогнозы на горизонтах 30/60/90 дней.

Ничего не меняет в существующих проектах — отдельный скрипт.
"""

import sys
import os
import datetime
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── MOEX ISS API (как в Kronos) ────────────────────────────────
MOEX_BASE = "https://iss.moex.com/iss/engines/stock/markets/shares"
BOARDS = {
    "SBER": "TQBR",
    "GAZP": "TQBR",
    "LKOH": "TQBR",
    "SBERP": "TQBR",
    "VTBR": "TQBR",
}

def moex_url(ticker):
    board = BOARDS.get(ticker, "TQBR")
    return f"{MOEX_BASE}/boards/{board}/securities/{ticker}/candles.json"


def fetch_moex_candles(ticker, days=60):
    """Fetch OHLCV candles from MOEX ISS — как в Kronos."""
    dfs = []
    today = datetime.date.today()

    for offset in range(days):
        day = today - datetime.timedelta(days=offset)
        day_str = day.strftime("%Y-%m-%d")
        next_day = (day + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

        url = moex_url(ticker)
        params = {
            "interval": 60,  # 1-hour candles
            "limit": 500,
            "from": day_str,
            "till": next_day,
        }

        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code != 200:
                continue
            data = r.json()["candles"]["data"]
            if not data:
                continue
            cols = r.json()["candles"]["columns"]

            df = pd.DataFrame(data, columns=cols)
            df = df.rename(columns={
                "OPEN": "open", "CLOSE": "close", "HIGH": "high",
                "LOW": "low", "VOLUME": "volume", "BEGIN": "begin"
            })
            df["begin"] = pd.to_datetime(df["begin"])
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["begin"]).sort_values("begin").reset_index(drop=True)
    return combined


# ── Kronos Mini (из проекта kronos-signal) ─────────────────────

def load_kronos():
    """Load Kronos-mini from local project (same as kronos-signal/run.py)."""
    KRONOS_PROJECT = os.path.expanduser("~/kronos-signal")
    KRONOS_SRC = os.path.join(KRONOS_PROJECT, "kronos-source")

    if KRONOS_SRC not in sys.path:
        sys.path.insert(0, KRONOS_SRC)

    from model import Kronos, KronosTokenizer, KronosPredictor

    MODEL_PATH = os.path.join(KRONOS_PROJECT, "models", "Kronos-mini")
    TOKENIZER_PATH = os.path.join(KRONOS_PROJECT, "models", "Kronos-Tokenizer-2k")

    print("[Kronos] Loading model...")
    tok = KronosTokenizer.from_pretrained(TOKENIZER_PATH)
    model = Kronos.from_pretrained(MODEL_PATH)

    # Force CPU mode due to RTX 5060 Ti CUDA incompatibility (sm_120 not supported)
    device = "cpu"
    predictor = KronosPredictor(model, tok, device=device, max_context=2048)
    print(f"[Kronos] Ready (device={device})")
    return predictor


def kronos_predict(predictor, df_full, pred_len):
    """Run Kronos prediction. Returns forecast close prices."""
    # Prepare timestamps from 'begin' column (datetime) — MUST be Series not DatetimeIndex
    x_ts = pd.Series(df_full["begin"])
    last_ts = df_full["begin"].iloc[-1]
    y_ts_future = pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=pred_len, freq="h")

    # Kronos calc_time_stamps expects Series with .dt accessor
    if isinstance(y_ts_future, pd.DatetimeIndex):
        y_ts_future = pd.Series(y_ts_future)

    try:
        pred_df = predictor.predict(
            df=df_full[["open", "high", "low", "close", "volume"]],
            x_timestamp=x_ts,
            y_timestamp=y_ts_future,
            pred_len=pred_len,
            T=1.0,
            top_k=0,
            top_p=0.9,
            sample_count=1,
            verbose=False,
        )
        return pred_df["close"].values
    except Exception as e:
        print(f"  [Kronos] ERROR: {e}")
        import traceback; traceback.print_exc()
        return None


# ── TimesFM 2.5 ───────────────────────────────────────────────

def load_timesfm():
    """Load and compile TimesFM 2.5."""
    from timesfm.timesfm_2p5 import timesfm_2p5_torch as tftorch
    from timesfm.configs import ForecastConfig

    print("[TimesFM] Loading model...")
    model = tftorch.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")

    config = ForecastConfig(max_context=512, max_horizon=128)
    print("[TimesFM] Compiling...")
    model.compile(config)
    # Sanity check: HORIZONS in main() must stay within TimesFM's compiled max_horizon.
    # If you bump HORIZONS above 128, recompile with a larger max_horizon here.
    # Force CPU mode due to RTX 5060 Ti CUDA incompatibility (sm_120 not supported)
    model._device = "cpu"
    print("[TimesFM] Ready (device=cpu)!")
    return model


def timesfm_forecast(model, values, horizon):
    """Run TimesFM forecast. Returns array of close prices."""
    point_forecast, _ = model.forecast(horizon=horizon, inputs=[values])
    return point_forecast[0].astype(np.float64)


# ── Signal comparison ─────────────────────────────────────────

def compute_signal(pred_price, last_price):
    """Compute BUY/SELL/HOLD signal."""
    chg = (pred_price - last_price) / last_price * 100
    if chg >= 2.0:
        return "BUY", chg
    elif chg <= -2.0:
        return "SELL", chg
    else:
        return "HOLD", chg


def compare_signals(kronos_signal, kronos_chg, timesfm_signal, timesfm_chg):
    """Compare two model signals."""
    if kronos_signal == timesfm_signal:
        agreement = "CONCORDANT"
        confidence = "HIGH"
    elif abs(kronos_chg) < 0.5 and abs(timesfm_chg) < 0.5:
        agreement = "NEUTRAL-MATCH"
        confidence = "MEDIUM"
    else:
        agreement = "CONFLICT"
        confidence = "LOW"

    return agreement, confidence


# ── Main pipeline ─────────────────────────────────────────────

def main():
    TICKERS = ["SBERP", "GAZP", "LKOH"]  # Сбер преф, Газпром, Лукойл
    HORIZONS = [30, 60, 90]  # дни прогнозирования
    # Sanity: 30-day horizon is required for the 30d chart section below.
    if 30 not in HORIZONS:
        print("[WARN] HORIZONS does not contain 30 — 30d chart will be skipped (no cache).")
    # Sanity: TimesFM was compiled with max_horizon=128; do not exceed.
    assert max(HORIZONS) <= 128, (
        f"TimesFM max_horizon=128, but HORIZONS={HORIZONS}. "
        "Recompile TimesFM with a larger max_horizon or shrink HORIZONS."
    )

    print("=" * 80)
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M MSK")
    print(f"  TIMESFM 2.5 + KRONOS MINI — COMPARISON")
    print(f"  {now_str}")
    print("=" * 80)

    # Load both models
    print("\n[INIT] Loading Kronos-mini...")
    kronos_predictor = load_kronos()

    print("\n[INIT] Loading TimesFM 2.5...")
    timesfm_model = load_timesfm()

    all_results = []
    chart_data = {}  # кэш истории+прогноза (30d) на тикер, чтобы не дублировать запрос/инференс ниже

    for ticker in TICKERS:
        print(f"\n{'='*80}")
        print(f"[{ticker}] Fetching MOEX ISS data (60 days)...")

        # 1. Fetch data from MOEX
        df = fetch_moex_candles(ticker, days=60)
        if df is None or len(df) < 100:
            print(f"  ERROR: not enough data for {ticker}")
            continue

        print(f"  Got {len(df)} hourly candles")
        last_price = float(df["close"].iloc[-1])
        last_date = df["begin"].iloc[-1]
        print(f"  Period: {df['begin'].iloc[0].date()} — {last_date.date()}")
        print(f"  Last close: {last_price:.2f} RUB")

        # Prepare data for both models
        values_np = df["close"].values.astype(np.float32)

        results_for_ticker = []

        for horizon in HORIZONS:
            print(f"\n  --- Horizon: {horizon} days ---")

            # Kronos prediction (convert days to hours)
            # NOTE: Kronos is autoregressive — accuracy drops after ~20 days (500h)
            kronos_hours = min(horizon * 24, 500)
            kronos_limit_warning = horizon > 20
            print(f"  [Kronos] Predicting {kronos_hours}h ahead...")
            if kronos_limit_warning:
                print(f"    ⚠ Kronos autoregressive limit: accuracy drops after ~20 days")
            kronos_preds = kronos_predict(kronos_predictor, df, kronos_hours)

            if kronos_preds is not None:
                # Take last prediction as horizon endpoint
                kronos_horizon_pred = kronos_preds[-1]
                kronos_signal, kronos_chg = compute_signal(kronos_horizon_pred, last_price)
                print(f"  [Kronos] {horizon}d forecast: {kronos_horizon_pred:.2f} ({kronos_chg:+.2f}%) -> [{kronos_signal}]")
            else:
                kronos_signal = "ERROR"
                kronos_chg = 0
                kronos_horizon_pred = last_price

            # TimesFM prediction — model is compiled with max_horizon=128, and
            # HORIZONS never exceeds that, so no extra clamping is needed here.
            timesfm_horizon = horizon
            print(f"  [TimesFM] Predicting {timesfm_horizon}d ahead...")
            try:
                timesfm_preds = timesfm_forecast(timesfm_model, values_np, timesfm_horizon)
                timesfm_horizon_pred = timesfm_preds[-1]
                timesfm_signal, timesfm_chg = compute_signal(timesfm_horizon_pred, last_price)
                print(f"  [TimesFM] {horizon}d forecast: {timesfm_horizon_pred:.2f} ({timesfm_chg:+.2f}%) -> [{timesfm_signal}]")
            except Exception as e:
                timesfm_signal = "ERROR"
                timesfm_chg = 0
                timesfm_horizon_pred = last_price
                timesfm_preds = None
                print(f"  [TimesFM] ERROR: {e}")

            # Cache the 30d history + TimesFM forecast now so the chart section
            # below doesn't need to re-fetch MOEX data and re-run the model.
            if horizon == 30 and timesfm_preds is not None:
                chart_data[ticker] = {
                    "df": df,
                    "last_price": last_price,
                    "last_date": last_date,
                    "tf_preds": timesfm_preds,
                }

            # Compare signals
            agreement, confidence = compare_signals(
                kronos_signal, kronos_chg,
                timesfm_signal, timesfm_chg
            )

            results_for_ticker.append({
                "ticker": ticker,
                "horizon": horizon,
                "last_price": last_price,
                # Kronos
                "kronos_pred": kronos_horizon_pred,
                "kronos_signal": kronos_signal,
                "kronos_chg": kronos_chg,
                # TimesFM
                "timesfm_pred": timesfm_horizon_pred,
                "timesfm_signal": timesfm_signal,
                "timesfm_chg": timesfm_chg,
                # Comparison
                "agreement": agreement,
                "confidence": confidence,
            })

        all_results.extend(results_for_ticker)

    # ── Output comparison table ────────────────────────────────
    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print("=" * 80)

    for horizon in HORIZONS:
        print(f"\n--- {horizon} days forecast ---")
        header = (f"{'Ticker':<7} {'Last':>9} | "
                 f"{'Kronos':>12} {'Sig':>5} {'Chg%':>8} | "
                 f"{'TimesFM':>12} {'Sig':>5} {'Chg%':>8} | "
                 f"{'Agree':>10}")
        sep = "-" * len(header)
        print(sep)
        print(header)
        print(sep)

        for r in all_results:
            if r["horizon"] != horizon:
                continue
            print(
                f"{r['ticker']:<7} {r['last_price']:>9.2f} | "
                f"{r['kronos_pred']:>12.2f} {r['kronos_signal']:>5} {r['kronos_chg']:>+7.2f}% | "
                f"{r['timesfm_pred']:>12.2f} {r['timesfm_signal']:>5} {r['timesfm_chg']:>+7.2f}% | "
                f"{r['agreement']:>10}"
            )
        print(sep)

    # ── Interpretation ─────────────────────────────────────────
    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print("=" * 80)

    for r in all_results:
        if r["horizon"] != 30:
            continue  # Focus on 30d for interpretation

        ticker = r["ticker"]
        k_sig = r["kronos_signal"]
        t_sig = r["timesfm_signal"]
        agree = r["agreement"]

        print(f"\n[{ticker}] 30-day forecast:")
        print(f"  Kronos:    {r['kronos_pred']:.2f} RUB ({r['kronos_chg']:+.2f}%) -> [{k_sig}]")
        print(f"  TimesFM:   {r['timesfm_pred']:.2f} RUB ({r['timesfm_chg']:+.2f}%) -> [{t_sig}]")
        print(f"  Agreement: {agree} (confidence: {r['confidence']})")

        # Combined recommendation
        if agree == "CONCORDANT":
            if k_sig == "BUY":
                print(f"  => COMBINED: BUY — обе модели видят рост >2%")
            elif k_sig == "SELL":
                print(f"  => COMBINED: SELL — обе модели видят падение >2%")
            else:
                print(f"  => COMBINED: HOLD — обе модели видят боковик")
        elif agree == "NEUTRAL-MATCH":
            print(f"  => COMBINED: HOLD — обе модели близки к текущей цене, нет сигнала")
        else:
            print(f"  => COMBINED: CONFLICT — модели расходятся, ждать подтверждения")

    # ── Save chart (30d only) ──────────────────────────────────
    print(f"\n{'='*80}")
    print("Saving charts...")

    fig, axes = plt.subplots(len(TICKERS), 1, figsize=(14, 5*len(TICKERS)))
    if len(TICKERS) == 1:
        axes = [axes]

    for idx, ticker in enumerate(TICKERS):
        cached = chart_data.get(ticker)
        if cached is None:
            continue

        df = cached["df"]
        last_price = cached["last_price"]
        last_date = cached["last_date"]
        tf_preds = cached["tf_preds"]

        ax = axes[idx]
        history_dates = df["begin"]
        history_values = df["close"]

        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=30, freq="D"
        )

        ax.plot(history_dates, history_values, label=f"{ticker} History", color="blue", linewidth=1.5)
        ax.plot(forecast_dates, tf_preds, label="TimesFM 2.5 (30d)", color="red", linewidth=2, linestyle="--")

        # Mark last price
        ax.axhline(y=last_price, color="gray", linestyle=":", alpha=0.5, label=f"Last: {last_price:.2f}")

        ax.set_xlabel("Date")
        ax.set_ylabel("Price (RUB)")
        ax.set_title(f"{ticker}: History + TimesFM 30d Forecast", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

    out_path = "timesfm_kronos_comparison.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Chart saved: {out_path}")

    # ── Save results to log file ───────────────────────────────
    import os
    LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"comparison_{datetime.date.today().strftime('%Y%m%d')}.log")

    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(f"\n{'='*80}\n")
        lf.write(f"TIMESFM 2.5 + KRONOS MINI — COMPARISON\n")
        lf.write(f"{now_str}\n")
        lf.write(f"{'='*80}\n")
        # Write the table and interpretation to log too
        for horizon in HORIZONS:
            lf.write(f"\n--- {horizon} days forecast ---\n")
            header = (f"{'Ticker':<7} {'Last':>9} | "
                     f"{'Kronos':>12} {'Sig':>5} {'Chg%':>8} | "
                     f"{'TimesFM':>12} {'Sig':>5} {'Chg%':>8} | "
                     f"{'Agree':>10}")
            sep = "-" * len(header)
            lf.write(sep + "\n")
            lf.write(header + "\n")
            lf.write(sep + "\n")

            for r in all_results:
                if r["horizon"] != horizon:
                    continue
                lf.write(
                    f"{r['ticker']:<7} {r['last_price']:>9.2f} | "
                    f"{r['kronos_pred']:>12.2f} {r['kronos_signal']:>5} {r['kronos_chg']:>+7.2f}% | "
                    f"{r['timesfm_pred']:>12.2f} {r['timesfm_signal']:>5} {r['timesfm_chg']:>+7.2f}% | "
                    f"{r['agreement']:>10}\n"
                )
            lf.write(sep + "\n")

        lf.write(f"\n{'='*80}\n")
        lf.write("INTERPRETATION\n")
        lf.write("=" * 80 + "\n")
        for r in all_results:
            if r["horizon"] != 30:
                continue
            ticker = r["ticker"]
            k_sig = r["kronos_signal"]
            t_sig = r["timesfm_signal"]
            agree = r["agreement"]

            lf.write(f"\n[{ticker}] 30-day forecast:\n")
            lf.write(f"  Kronos:    {r['kronos_pred']:.2f} RUB ({r['kronos_chg']:+.2f}%) -> [{k_sig}]\n")
            lf.write(f"  TimesFM:   {r['timesfm_pred']:.2f} RUB ({r['timesfm_chg']:+.2f}%) -> [{t_sig}]\n")
            lf.write(f"  Agreement: {agree} (confidence: {r['confidence']})\n")

            if agree == "CONCORDANT":
                if k_sig == "BUY":
                    lf.write(f"  => COMBINED: BUY — обе модели видят рост >2%\n")
                elif k_sig == "SELL":
                    lf.write(f"  => COMBINED: SELL — обе модели видят падение >2%\n")
                else:
                    lf.write(f"  => COMBINED: HOLD — обе модели видят боковик\n")
            elif agree == "NEUTRAL-MATCH":
                lf.write(f"  => COMBINED: HOLD — обе модели близки к текущей цене, нет сигнала\n")
            else:
                lf.write(f"  => COMBINED: CONFLICT — модели расходятся, ждать подтверждения\n")

        lf.write(f"\n{'='*80}\n")
        lf.write("DONE\n")
        lf.write("=" * 80 + "\n")

    print(f"Log saved: {log_path}")


if __name__ == "__main__":
    main()
