"""TimesFM 2.5 + Kronos Mini — сравнение сигналов MOEX.

Загружает данные из MOEX ISS API (как Kronos), запускает обе модели и
сравнивает прогнозы на горизонтах 30/60/90 дней.

Единицы времени (важно)
-----------------------
История — ДНЕВНЫЕ бары (INTERVAL=24): один бар = один торговый день MOEX.
Поэтому HORIZONS в днях совпадают и с числом баров на входе Kronos, и с
числом шагов на входе TimesFM. Горизонт «30 дней» — это буквально 30 баров.

До 2026-10-14 скрипт работал на часовых барах (INTERVAL=60, ~15 свечей на
календарный день). Тогда horizon=30 означало 30 ШАГОВ ≈ 2 дня, а Kronos с
kronos_hours=min(horizon*24, 500) выдавал одну и ту же 500-шаговую развёртку
на все три горизонта. Числа в logs/comparison_*.log до этой даты несопоставимы
с текущими. См. audits/2026-10-14-audit.md (C1, C2, C3).

Ничего не меняет в существующих проектах — отдельный скрипт.
"""

import datetime
import os
import sys
import time
from zoneinfo import ZoneInfo  # stdlib, Python 3.9+; MSK — Europe/Moscow

import requests

# Абсолютный путь к директории скрипта: артефакты (png, logs) пишем рядом
# со скриптом, а не в cwd — cron может запускать из любой директории.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Force CPU mode before importing torch (RTX 5060 Ti sm_120 not supported)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Fix SIGSEGV in Kronos decode_s1() on CPU PyTorch builds (sm_120 not supported).
# PyTorch >2.x with OMP/MKL threads > 1 races during the long autoregressive
# loop. Single-threaded mode is stable. Setting these env vars BEFORE importing
# numpy/torch ensures the OMP runtime sees them on first pool allocation;
# _pin_threads() below is the runtime cap.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Ensure local venv takes precedence over Hermes global
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# Reproducibility: Kronos samples autoregressively (sample_logits=True is
# hardcoded in kronos.auto_regressive_inference), so an unseeded run gives a
# different rollout every time. Seeded once per run in main().
SEED = 20261014


def _pin_threads():
    """Ограничить intra/inter-op пул потоков до 1.

    set_num_interop_threads() можно вызвать только один раз за процесс —
    повторный вызов бросает RuntimeError. Это ломало run_debug.py, который
    уже пинит потоки до импорта основного скрипта (audit H1).
    """
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # уже установлено ранее в этом процессе (run_debug.py)


_pin_threads()


# ── Конфигурация ──────────────────────────────────────────────
INTERVAL = 24                 # MOEX ISS: 24 = дневные бары (audit C1)
FREQ = "B"                    # business days для будущих таймстемпов
HISTORY_DAYS = 120            # ~82 торговых бара; хватает для горизонта 90
MIN_BARS = 30                 # меньше — прогноз недостоверен
BOARD = "TQBR"                # MOEX base board for all tickers below
TICKERS = ["SBERP", "GAZP", "LKOH"]  # Сбер преф, Газпром, Лукойл
HORIZONS = [30, 60, 90]       # дни == бары == шаги TimesFM
TIMESFM_MAX_HORIZON = 128     # соответствует ForecastConfig(max_horizon=)
KRONOS_MAX_PRED_LEN = 500     # авторегрессивный предел Kronos (баров)
KRONOS_SAMPLE_COUNT = 8       # Kronos усредняет сэмплы внутри (kronos.py:394)
MOEX_PAGE_LIMIT = 500         # максимум MOEX ISS на страницу
MOEX_MAX_PAGES = 20           # страховка от бесконечной пагинации
MOEX_ATTEMPTS = 3             # ретраи при сетевых ошибках (audit H5)

# Как считаем сигнал. Порог ±2% был привязан к уровню цены и не учитывал
# ни волатильность, ни горизонт: на LKOH, выросшем на +26.5% за окно,
# Kronos давал −25.7% и автоматически попадал в ПРОДАЖА (audit, часть 3).
#
#   "z_vol" — ход прогноза в единицах шума h-шаговой диффузии:
#             z = (pred − last) / (σ_дневн · last · √h),  порог ±Z_THRESHOLD.
#             Сохраняет смысл «прогноз доходности», но отвязан от цены,
#             волатильности и горизонта.
#   "abs"   — старое поведение: (pred − last)/last·100, порог ±2%.
#
# Эксперимент 2026-10-15 (experiments/signal_normalization.py):
# согласие моделей выросло с 1/9 до 6/9, число ПРОДАЖА упало с 7 до 3.
SIGNAL_MODE = "z_vol"
Z_THRESHOLD = 1.0             # ±1σ h-шаговой диффузии
ABS_THRESHOLD = 2.0           # используется только при SIGNAL_MODE == "abs"

if max(HORIZONS) > TIMESFM_MAX_HORIZON:
    raise ValueError(
        f"HORIZONS={HORIZONS} превышает скомпилированный max_horizon="
        f"{TIMESFM_MAX_HORIZON}. Увеличь max_horizon в load_timesfm() "
        f"или уменьши HORIZONS."
    )

# Метка «прогноза нет». Раньше здесь стояло "ERROR", из-за чего два отказа
# моделей сравнивались как равные строки и давали «СОВПАДЕНИЕ / ВЫСОКАЯ»
# (audit C4), а ветка «НЕЙТРАЛЬНО» оказывалась достижима только при отказе
# (audit C5).
NO_DATA = "НЕТ ДАННЫХ"


# ── MOEX ISS API (как в Kronos) ────────────────────────────────
MOEX_BASE = "https://iss.moex.com/iss/engines/stock/markets/shares"


def moex_url(ticker):
    return f"{MOEX_BASE}/boards/{BOARD}/securities/{ticker}/candles.json"


def _moex_json(url, params, ticker):
    """GET с ретраями. Возвращает разобранный JSON или None после MOEX_ATTEMPTS."""
    last_err = None
    for attempt in range(MOEX_ATTEMPTS):
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001 — MOEX отдаёт и HTTP, и JSON-ошибки
            last_err = e
            if attempt + 1 < MOEX_ATTEMPTS:
                time.sleep(2 ** attempt)
    print(f"  [{ticker}] MOEX ISS недоступен после {MOEX_ATTEMPTS} попыток: {last_err}")
    return None


def fetch_moex_candles(ticker, days=HISTORY_DAYS):
    """Fetch OHLCV candles from MOEX ISS — одним периодом с пагинацией.

    Один HTTP-запрос на страницу (≤500 баров) с offset'ом в `start`.
    `VALUE` (оборот в рублях) сохраняется как `amount` — Kronos умеет
    брать реальный оборот, иначе подставляет аппроксимацию volume*price
    (kronos.py:531) — audit H3.
    """
    end = datetime.date.today()
    start = end - datetime.timedelta(days=days)

    url = moex_url(ticker)
    base_params = {
        "interval": INTERVAL,
        "from": start.strftime("%Y-%m-%d"),
        "till": end.strftime("%Y-%m-%d"),
    }

    rows, cols, start_row = [], None, 0
    for _ in range(MOEX_MAX_PAGES):
        payload = _moex_json(url, dict(base_params, limit=MOEX_PAGE_LIMIT,
                                       start=start_row), ticker)
        if payload is None:
            return None
        block = payload.get("candles") or {}
        data = block.get("data") or []
        if cols is None:
            cols = block.get("columns")
        if not data:
            break
        rows.extend(data)
        if len(data) < MOEX_PAGE_LIMIT:  # последняя страница
            break
        start_row += len(data)
    else:
        # audit M4: без этой границы цикл не завершился бы, если бы MOEX
        # проигнорировал offset `start` и отдавал одну и ту же страницу.
        print(f"  [{ticker}] пагинация остановлена на MOEX_MAX_PAGES={MOEX_MAX_PAGES}")

    if not rows or not cols:
        return None

    df = pd.DataFrame(rows, columns=cols)
    # MOEX ISS отдаёт колонки уже в нижнем регистре
    # (['open','close','high','low','value','volume','begin','end']) и может
    # вернуть другой регистр в зависимости от endpoint — приводим к нижнему
    # регистру до маппинга. Старая карта {"OPEN": "open", ...} была no-op,
    # из-за чего схема молча не проверялась.
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in ("open", "high", "low", "close", "volume", "value", "begin")
               if c not in df.columns]
    if missing:
        print(f"  [{ticker}] неожиданная схема MOEX, нет колонок: {missing}")
        return None
    # Оборот в рублях: Kronos ждёт колонку с именем 'amount'
    # (kronos.py:491). Без неё он подставляет аппроксимацию
    # volume*mean(price) (kronos.py:531) — audit H3.
    df = df.rename(columns={"value": "amount"})
    df["begin"] = pd.to_datetime(df["begin"])

    numeric = ["open", "high", "low", "close", "volume", "amount"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    df = (df.drop_duplicates(subset=["begin"])
            .sort_values("begin")
            .reset_index(drop=True))
    df = df[df["close"] > 0].reset_index(drop=True)
    return df


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


def has_weekend_sessions(df, lookback=20):
    """Торгует ли инструмент по выходным.

    MOEX запустил единую выходную сессию: в баровне SBERP/GAZP/LKOH реально
    попадает ~13 суббот и ~13 воскресений из 112 дневных баров. Если брать
    будущие метки как freq='B' (пн-пт), календарь меток начнёт систематически
    отставать от реального: 90 баров растянутся на ~126 календарных дней —
    ровно тот класс ошибки, что описан в audit C3. Поэтому режим выбирается
    по данным, а не предполагается.
    """
    tail = df["begin"].tail(lookback)
    return bool((tail.dt.weekday >= 5).any())


def future_bars(last_ts, n, trades_weekends):
    """n будущих торговых дней после last_ts.

    7 дней/неделю -> freq='D'; только будни -> freq='B' (выходние пропускаются).
    """
    if trades_weekends:
        start, freq = last_ts + pd.offsets.Day(1), "D"
    else:
        start, freq = last_ts + pd.offsets.BDay(1), "B"
    return pd.Series(pd.date_range(start=start, periods=n, freq=freq))


def kronos_predict(predictor, df_full, pred_len, trades_weekends):
    """Run Kronos prediction. Returns array of predicted close prices."""
    x_ts = pd.Series(df_full["begin"].values)
    y_ts_future = future_bars(df_full["begin"].iloc[-1], pred_len, trades_weekends)

    try:
        pred_df = predictor.predict(
            df=df_full[["open", "high", "low", "close", "volume", "amount"]],
            x_timestamp=x_ts,
            y_timestamp=y_ts_future,
            pred_len=pred_len,
            T=1.0,
            top_k=0,
            top_p=0.9,
            # Kronos жёстко сэмплирует (sample_logits=True в kronos.py:438),
            # поэтому sample_count=1 даёт один шумный розыгрыш. При >1 Kronos
            # усредняет сэмплы внутри себя (kronos.py:394,465) — это и есть
            # устранение дисперсии (audit C2).
            sample_count=KRONOS_SAMPLE_COUNT,
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
    # Фикс 2026-09-08: строго офлайн — HF геоблок 403, онлайн fallback запрещён
    try:
        model = tftorch.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch", local_files_only=True)
    except Exception as e:
        print(f"[TimesFM] local cache missing — пропуск прогноза (offline only): {e}")
        raise RuntimeError(
            f"TimesFM кэш отсутствует offline: {e} — пропустите прогон, "
            f"не пробуйте онлайн (HF 403)") from e

    config = ForecastConfig(max_context=512, max_horizon=TIMESFM_MAX_HORIZON)
    print("[TimesFM] Compiling...")
    model.compile(config)
    print("[TimesFM] Ready (device=cpu)!")
    return model


def timesfm_forecast(model, values, horizon):
    """Run TimesFM forecast. Returns array of close prices, len == horizon."""
    point_forecast, _ = model.forecast(horizon=horizon, inputs=[values])
    return point_forecast[0].astype(np.float64)


# ── Signal comparison ─────────────────────────────────────────

def window_stats(df):
    """Статистики окна: последняя цена, среднее, дневная волатильность.

    sigma_d — std дневных лог-доходностей. Используется как масштаб шума:
    h-шаговый шум ≈ sigma_d · last · √h.
    """
    close = df["close"].astype(float)
    last = float(close.iloc[-1])
    mean = float(close.mean())
    rets = np.log(close / close.shift(1)).dropna()
    sigma_d = float(rets.std()) if len(rets) > 1 else 0.0
    return last, mean, sigma_d


def compute_signal(pred_price, last_price, sigma_d=None, horizon=None):
    """Сигнал модели. Возвращает (метка, chg%, z).

    z = (pred − last) / (sigma_d · last · √horizon) — ход прогноза в единицах
    шума h-шаговой диффузии. Отвязывает вердикт от уровня цены, волатильности
    и горизонта (audit, часть 3). При sigma_d=None или SIGNAL_MODE=="abs"
    используется старый порог ±ABS_THRESHOLD%.
    """
    chg = (pred_price - last_price) / last_price * 100
    if SIGNAL_MODE == "z_vol" and sigma_d and horizon and sigma_d > 0:
        denom = sigma_d * last_price * np.sqrt(horizon)
        z = (pred_price - last_price) / denom
        if z >= Z_THRESHOLD:
            return "ПОКУПКА", chg, z
        if z <= -Z_THRESHOLD:
            return "ПРОДАЖА", chg, z
        return "ДЕРЖАТЬ", chg, z
    if chg >= ABS_THRESHOLD:
        return "ПОКУПКА", chg, None
    if chg <= -ABS_THRESHOLD:
        return "ПРОДАЖА", chg, None
    return "ДЕРЖАТЬ", chg, None


def reversion_excess(pred_price, mean, sigma_d, last_price, horizon):
    """Насколько прогноз отличается от механического отката к среднему.

    z_excess = (pred − mean) / (sigma_d · last · √horizon).
    Нулевое значение означает, что модель сказала ровно то, что уже
    содержится в арифметике окна (среднее), то есть ничего не предсказала.
    Это ДИАГНОСТИКА модели, а не сигнал: значение нельзя читать как
    «покупка/продажа», поскольку нулевая отметка соответствует прогнозу
    = среднее, а не = текущая цена.
    """
    if not sigma_d or sigma_d <= 0:
        return None
    return (pred_price - mean) / (sigma_d * last_price * np.sqrt(horizon))


def compare_signals(kronos_signal, timesfm_signal):
    """Compare two model signals (returns Russian labels)."""
    # Отказ любой из моделей — не согласие, а отсутствие результата.
    # Проверка идёт первым, иначе два NO_DATA сравнились бы как равные строки.
    if NO_DATA in (kronos_signal, timesfm_signal):
        return NO_DATA, NO_DATA

    if kronos_signal == timesfm_signal:
        if kronos_signal == "ДЕРЖАТЬ":
            # Обе модели согласились ничего не делать — это не высокая уверенность.
            return "НЕЙТРАЛЬНО", "СРЕДНЯЯ"
        return "СОВПАДЕНИЕ", "ВЫСОКАЯ"

    return "РАСХОЖДЕНИЕ", "НИЗКАЯ"


# ── Рендеринг (единый источник для stdout и лога) ─────────────
# Раньше таблица и интерпретация печатались и писались в лог двумя
# копиями, которые разъехались (разная ширина колонок) — audit M-дрейф.

_TABLE_HEADER = (f"{'Тикер':<6} {'Цена':>9} {'σ%':>6} | "
                 f"{'Kronos':>10} {'z':>6} {'Сигн':>10} {'Изм%':>8} | "
                 f"{'TimesFM':>10} {'z':>6} {'Сигн':>10} {'Изм%':>8} | "
                 f"{'Согл.':>12}")


def _num(value, spec):
    """Форматировать число; None → «—» (чтобы не выдавать last_price за прогноз)."""
    if value is None:
        return "—"
    return format(value, spec)


def render_table(results, horizon):
    rows = [r for r in results if r["horizon"] == horizon]
    sep = "-" * len(_TABLE_HEADER)
    lines = [sep, _TABLE_HEADER, sep]
    for r in rows:
        sig = r.get("sigma_d")
        lines.append(
            f"{r['ticker']:<6} {_num(r['last_price'], '>9.2f')} "
            f"{_num(None if sig is None else sig * 100, '>6.2f')} | "
            f"{_num(r['kronos_pred'], '>10.2f')} {_num(r.get('kronos_z'), '>6.2f')} "
            f"{r['kronos_signal']:>10} {_num(r['kronos_chg'], '>+7.2f')}% | "
            f"{_num(r['timesfm_pred'], '>10.2f')} {_num(r.get('timesfm_z'), '>6.2f')} "
            f"{r['timesfm_signal']:>10} {_num(r['timesfm_chg'], '>+7.2f')}% | "
            f"{r['agreement']:>12}"
        )
    lines.append(sep)
    if not rows:
        lines = [f"(нет данных за {horizon} дней)"]
    return "\n".join(lines)


def render_interpretation(results, horizon=30):
    rows = [r for r in results if r["horizon"] == horizon]
    if not rows:
        return f"(нет данных за {horizon} дней)"

    lines = []
    for r in rows:
        agree, k_sig = r["agreement"], r["kronos_signal"]
        lines.append(f"\n[{r['ticker']}] {horizon}-day forecast:")
        lines.append(
            f"  Kronos:    {_num(r['kronos_pred'], '.2f')} RUB "
            f"({_num(r['kronos_chg'], '+.2f')}%, z={_num(r.get('kronos_z'), '+.2f')}) -> [{k_sig}]"
        )
        lines.append(
            f"  TimesFM:   {_num(r['timesfm_pred'], '.2f')} RUB "
            f"({_num(r['timesfm_chg'], '+.2f')}%, z={_num(r.get('timesfm_z'), '+.2f')}) -> [{r['timesfm_signal']}]"
        )
        lines.append(f"  Agreement: {agree} (confidence: {r['confidence']})")
        lines.append("  => " + _recommendation(agree, k_sig))
    return "\n".join(lines)


def render_diagnostics(results):
    """Диагностика моделей: отличается ли прогноз от механического отката.

    z_excess ≈ 0 означает, что модель повторила арифметику окна (среднее) и
    не сказала ничего сверх неё.
    """
    lines = ["Модель              тикер   h    z_excess   вердикт",
             "-" * 58]
    for r in results:
        for model in ("kronos", "timesfm"):
            ze = r.get(f"{model}_excess")
            if ze is None:
                continue
            verdict = ("откат к среднему" if abs(ze) < Z_THRESHOLD
                       else "собственный сигнал")
            lines.append(f"{model:<18}{r['ticker']:>7}{r['horizon']:>5}"
                         f"{ze:>11.2f}   {verdict}")
    return "\n".join(lines)


def _recommendation(agreement, kronos_signal):
    if agreement == NO_DATA:
        return "СОВОКУПНЫЙ: НЕТ ДАННЫХ — прогноз одной из моделей недоступен"
    if agreement == "СОВПАДЕНИЕ":
        word = "рост" if kronos_signal == "ПОКУПКА" else "падение"
        unit = (f"z > {Z_THRESHOLD:.0f}σ" if SIGNAL_MODE == "z_vol"
                else f"{ABS_THRESHOLD:.0f}%")
        return f"СОВОКУПНЫЙ: {kronos_signal} — обе модели видят {word} ({unit})"
    if agreement == "НЕЙТРАЛЬНО":
        return "СОВОКУПНЫЙ: ДЕРЖАТЬ — обе модели видят боковик"
    return "СОВОКУПНЫЙ: РАСХОЖДЕНИЕ — модели расходятся, ждать подтверждения"


# ── Main pipeline ─────────────────────────────────────────────

def main():
    now_str = datetime.datetime.now(tz=ZoneInfo("Europe/Moscow")).strftime("%Y-%m-%d %H:%M MSK")
    print("=" * 80)
    print("  TIMESFM 2.5 + KRONOS MINI — COMPARISON")
    print(f"  {now_str}")
    print(f"  units: daily bars (interval={INTERVAL}), "
          f"horizons={HORIZONS} days, seeds={SEED}")
    print(f"  signal mode: {SIGNAL_MODE} (порог ±{Z_THRESHOLD:.0f}σ "
          f"h-шаговой диффузии)")
    print("=" * 80)

    torch.manual_seed(SEED)  # см. SEED

    print("\n[INIT] Loading Kronos-mini...")
    kronos_predictor = load_kronos()

    print("\n[INIT] Loading TimesFM 2.5...")
    timesfm_model = load_timesfm()

    all_results = []
    # кэш истории + прогноза (первый горизонт) на тикер, чтобы не дублировать
    # запрос/инференс в секции графиков
    chart_data = {}

    for ticker in TICKERS:
        print(f"\n{'='*80}")
        print(f"[{ticker}] Fetching MOEX ISS data ({HISTORY_DAYS} days)...")

        # 1. Fetch data from MOEX
        df = fetch_moex_candles(ticker)
        if df is None or len(df) < MIN_BARS:
            got = 0 if df is None else len(df)
            print(f"  ERROR: not enough data for {ticker} ({got} bars, need {MIN_BARS})")
            continue

        print(f"  Got {len(df)} daily bars")
        last_price = float(df["close"].iloc[-1])
        last_date = df["begin"].iloc[-1]
        print(f"  Period: {df['begin'].iloc[0].date()} — {last_date.date()}")
        print(f"  Last close: {last_price:.2f} RUB")

        # Календарь торгов определяется по данным, а не предполагается (audit C3).
        trades_weekends = has_weekend_sessions(df)
        print(f"  Trading calendar: "
              f"{'7 days/week (weekend sessions)' if trades_weekends else 'Mon-Fri only'}")

        # Prepare data for both models
        values_np = df["close"].values.astype(np.float32)
        last_price, mean_price, sigma_d = window_stats(df)
        print(f"  Window: mean={mean_price:.2f} sigma_d={sigma_d*100:.2f}%/day")

        results_for_ticker = []

        for horizon in HORIZONS:
            print(f"\n  --- Horizon: {horizon} days ---")

            # Kronos prediction. Горизонт в днях == число баров, потому что
            # INTERVAL=24. Предупреждение выставляем по фактическому усечению,
            # а не по магическому порогу (audit M2).
            kronos_bars = min(horizon, KRONOS_MAX_PRED_LEN)
            capped = kronos_bars < horizon
            print(f"  [Kronos] Predicting {kronos_bars} bars ahead"
                  + (f" (запрошено {horizon}d, cap={KRONOS_MAX_PRED_LEN})" if capped else "..."))
            kronos_preds = kronos_predict(kronos_predictor, df, kronos_bars,
                                          trades_weekends)

            kronos_z = kronos_excess = None
            if kronos_preds is not None:
                kronos_horizon_pred = float(kronos_preds[-1])
                kronos_signal, kronos_chg, kronos_z = compute_signal(
                    kronos_horizon_pred, last_price, sigma_d, horizon)
                kronos_excess = reversion_excess(
                    kronos_horizon_pred, mean_price, sigma_d, last_price, horizon)
                print(f"  [Kronos] {horizon}d forecast: "
                      f"{kronos_horizon_pred:.2f} ({kronos_chg:+.2f}%, "
                      f"z={kronos_z:+.2f}) -> [{kronos_signal}]")
            else:
                kronos_signal = NO_DATA
                kronos_chg = None
                kronos_horizon_pred = None

            # TimesFM prediction. max_context=512 покрывает всю историю
            # (~112 дневных баров), усечения больше нет (audit H4).
            print(f"  [TimesFM] Predicting {horizon}d ahead...")
            timesfm_preds = None
            timesfm_z = timesfm_excess = None
            try:
                timesfm_preds = timesfm_forecast(timesfm_model, values_np, horizon)
                timesfm_horizon_pred = float(timesfm_preds[-1])
                timesfm_signal, timesfm_chg, timesfm_z = compute_signal(
                    timesfm_horizon_pred, last_price, sigma_d, horizon)
                timesfm_excess = reversion_excess(
                    timesfm_horizon_pred, mean_price, sigma_d, last_price, horizon)
                print(f"  [TimesFM] {horizon}d forecast: {timesfm_horizon_pred:.2f} "
                      f"({timesfm_chg:+.2f}%, z={timesfm_z:+.2f}) -> [{timesfm_signal}]")
            except Exception as e:
                timesfm_signal = NO_DATA
                timesfm_chg = None
                timesfm_horizon_pred = None
                print(f"  [TimesFM] ERROR: {e}")

            # Cache the first-horizon history + TimesFM forecast now so the chart
            # section below doesn't re-fetch MOEX data or re-run the model.
            # Даты прогноза считаются ТОЙ ЖЕ функцией, что и метки для Kronos,
            # поэтому ось графика совпадает с длиной прогноза (audit H2).
            if horizon == HORIZONS[0] and timesfm_preds is not None:
                chart_data[ticker] = {
                    "df": df,
                    "last_price": last_price,
                    "last_date": last_date,
                    "tf_preds": timesfm_preds,
                    "forecast_dates": future_bars(
                        last_date, len(timesfm_preds), trades_weekends),
                }

            agreement, confidence = compare_signals(kronos_signal, timesfm_signal)

            results_for_ticker.append({
                "ticker": ticker,
                "horizon": horizon,
                "last_price": last_price,
                "mean_price": mean_price,
                "sigma_d": sigma_d,
                "kronos_pred": kronos_horizon_pred,
                "kronos_signal": kronos_signal,
                "kronos_chg": kronos_chg,
                "kronos_z": kronos_z,
                "kronos_excess": kronos_excess,
                "timesfm_pred": timesfm_horizon_pred,
                "timesfm_signal": timesfm_signal,
                "timesfm_chg": timesfm_chg,
                "timesfm_z": timesfm_z,
                "timesfm_excess": timesfm_excess,
                "agreement": agreement,
                "confidence": confidence,
            })

        all_results.extend(results_for_ticker)

    # ── Output comparison table ────────────────────────────────
    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print("=" * 80)

    sections = []
    for horizon in HORIZONS:
        table = render_table(all_results, horizon)
        sections.append(table)
        print(f"\n--- Прогноз на {horizon} дней ---")
        print(table)

    # ── Interpretation ─────────────────────────────────────────
    interpretation = render_interpretation(all_results, HORIZONS[0])
    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print("=" * 80)
    print(interpretation)

    # ── Diagnostics: is the forecast more than mechanical reversion? ──
    diagnostics = render_diagnostics(all_results)
    print(f"\n{'='*80}")
    print("MODEL DIAGNOSTICS (z_excess ~ 0 => прогноз = арифметика окна)")
    print("=" * 80)
    print(diagnostics)

    # ── Save chart (первый горизонт) ───────────────────────────
    print(f"\n{'='*80}")
    print("Saving charts...")
    save_chart(chart_data, HORIZONS[0])

    # ── Save results to log file ───────────────────────────────
    LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"comparison_{datetime.date.today().strftime('%Y%m%d')}.log")

    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(f"\n{'='*80}\n")
        lf.write("TIMESFM 2.5 + KRONOS MINI — COMPARISON\n")
        lf.write(f"{now_str}\n")
        lf.write(f"units: daily bars (interval={INTERVAL}), "
                 f"horizons={HORIZONS} days, seeds={SEED}\n")
        lf.write(f"signal mode: {SIGNAL_MODE} (порог ±{Z_THRESHOLD:.0f}σ)\n")
        lf.write(f"{'='*80}\n")
        for horizon, table in zip(HORIZONS, sections):
            lf.write(f"\n--- Прогноз на {horizon} дней ---\n{table}\n")
        lf.write(f"\n{'='*80}\n")
        lf.write("INTERPRETATION\n")
        lf.write("=" * 80 + "\n")
        lf.write(interpretation + "\n")
        lf.write(f"\n{'='*80}\n")
        lf.write("MODEL DIAGNOSTICS (z_excess ~ 0 => прогноз = арифметика окна)\n")
        lf.write("=" * 80 + "\n")
        lf.write(diagnostics + "\n")
        lf.write(f"\n{'='*80}\nDONE\n{'='*80}\n")

    print(f"Log saved: {log_path}")


def save_chart(chart_data, horizon, out_path=None):
    """Построить график истории + прогноза. Ось X = реальные торговые дни."""
    tickers = [t for t in TICKERS if t in chart_data]
    if not tickers:
        print("  График не построен: нет ни одного успешного прогноза TimesFM.")
        return

    fig, axes = plt.subplots(len(tickers), 1, figsize=(14, 5 * len(tickers)),
                             squeeze=False)

    for ax, ticker in zip(axes[:, 0], tickers):
        cached = chart_data[ticker]
        df = cached["df"]
        last_price = cached["last_price"]
        tf_preds = cached["tf_preds"]

        # Ось X обязана совпадать с длиной прогноза, иначе прогноз на 2 дня
        # растягивается на 30-дневную шкалу (audit H2).
        forecast_dates = cached["forecast_dates"]
        if len(forecast_dates) != len(tf_preds):
            raise ValueError(
                f"[{ticker}] ось графика ({len(forecast_dates)} дат) "
                f"не совпадает с прогнозом ({len(tf_preds)} точек)")

        ax.plot(df["begin"], df["close"], label=f"{ticker} History",
                color="blue", linewidth=1.5)
        ax.plot(forecast_dates, tf_preds,
                label=f"TimesFM 2.5 ({len(tf_preds)}d)", color="red",
                linewidth=2, linestyle="--")
        ax.axhline(y=last_price, color="gray", linestyle=":", alpha=0.5,
                   label=f"Last: {last_price:.2f}")

        ax.set_xlabel("Date")
        ax.set_ylabel("Price (RUB)")
        ax.set_title(f"{ticker}: History + TimesFM {horizon}d Forecast", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

    out_path = out_path or os.path.join(SCRIPT_DIR, "timesfm_kronos_comparison.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Chart saved: {out_path}")


if __name__ == "__main__":
    main()