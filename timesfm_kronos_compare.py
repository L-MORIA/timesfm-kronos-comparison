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
import logging
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

LOG = logging.getLogger("timesfm_kronos_compare")

# Причина, по которой torch отказался пинтить inter-op пул. Заполняется на
# этапе импорта (до конфигурации логгера), поэтому само сообщение уходит в
# run-лог позже, из _setup_logging().
_THREAD_PIN_NOTE = None


def _setup_logging():
    """Подключить обработчики логирования.

    stdout — Обязательно именно stdout, а не stderr по умолчанию: cron-обёртка
    (profiles/old-laptop/scripts/timesfm-kronos-compare.py:62) читает
    proc.stdout и доставляет его в чат дословно, а от stderr берёт только
    последние 1500 символов (там же стр. 76). Уход сообщений в stderr тихо
    обрезал бы отчёт в доставке. Формат stdout — «%(message)s», без
    timestamps и уровней: этот поток читает человек в чате.

    Формат файла полный: timestamp, уровень, имя логгера — там нужен разбор
    после падения.
    """
    logs_dir = os.path.join(SCRIPT_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = os.path.join(logs_dir, f"run_{stamp}.log")

    LOG.setLevel(logging.DEBUG)
    LOG.handlers.clear()  # иначе повторный вызов в одном процессе дублирует вывод
    LOG.propagate = False  # записи не должны утекать в root-логгер

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(logging.Formatter("%(message)s"))
    LOG.addHandler(stdout_handler)

    file_handler = logging.FileHandler(run_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    LOG.addHandler(file_handler)

    if _THREAD_PIN_NOTE:
        # Сообщение уровня DEBUG, произошедшее до конфигурации, иначе потерялось бы.
        LOG.debug("torch отклонил set_num_interop_threads(1) при импорте: %s",
                  _THREAD_PIN_NOTE)
    LOG.info("run-log: %s", run_path)
    return run_path


def _pin_threads():
    """Ограничить intra/inter-op пул потоков до 1.

    set_num_interop_threads() можно вызвать только один раз за процесс —
    повторный вызов бросает RuntimeError. Это ломало run_debug.py, который
    уже пинит потоки до импорта основного скрипта (audit H1).

    RuntimeError здесь означает два противоположных случая, и молчать обоих
    нельзя:
      1) пул уже создан в этом процессе (run_debug.py) — ожидаемо и безобидно;
      2) «parallel work has started» — гарантия single-thread НЕ сработала,
         то есть ровно тот класс SIGSEGV, ради которого файл и пиннит потоки.
    Текст исключения сохраняется и уходит в run-лог на уровне DEBUG.
    """
    global _THREAD_PIN_NOTE
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as e:
        _THREAD_PIN_NOTE = str(e)


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

# MOEX ISS отдаёт свечу текущего торгового дня, даже когда день ещё не
# закрыт. В 11:00 это ~2 часа торгов: close = 286.38 вместо итогового.
# Такой бар искажает всё — last_price, sigma_дневн (частичный возврат),
# окно Kronos и ось графика — плюс делает прогоны одного дня несравнимыми
# между собой. Единая торговая сессия MOEX заканчивается в 23:50 MSK,
# поэтому дневной бар считаем закрытым только после этого времени.
MOEX_SESSION_END_MIN = 23 * 60 + 50  # 23:50 MSK

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
    """URL дневных свечей MOEX ISS для бумаги на BOARD.

    Контракт жёсткий: BOARD («TQBR») зашит в путь, и при смене площадки MOEX
    этот endpoint начинает отдавать 404 без внятного сообщения — _moex_json()
    съест это в ретраях и вернёт None, и тикер молча выпадет из отчёта.
    Проверять здесь надо и BOARD, и то, что бумага вообще на нём торгуется.
    """
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
    LOG.warning("  [%s] MOEX ISS недоступен после %d попыток: %s",
                ticker, MOEX_ATTEMPTS, last_err)
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
        LOG.warning("  [%s] пагинация остановлена на MOEX_MAX_PAGES=%d",
                    ticker, MOEX_MAX_PAGES)

    if not rows or not cols:
        return None

    return normalize_candles(pd.DataFrame(rows, columns=cols), ticker)


def normalize_candles(df, ticker=None):
    """Привести сырой ответ MOEX к внутренней схеме.

    Единственный источник правды о преобразовании колонок: используется и
    продом, и бэктестом. Раньше схема дублировалась в разных местах, и
    бэктест падал с `KeyError: 'amount'`, потому что в нём был свой
    экземпляр сырых колонок.

    - регистр приводится к нижнему: MOEX отдаёт `open/close/...`, но может
      вернуть другой регистр в зависимости от endpoint;
    - оборот `value` -> `amount`, как ждёт Kronos (kronos.py:491). Иначе он
      подставляет аппроксимацию volume*mean(price) (kronos.py:531) — audit H3;
    - незакрытый бар текущего дня отбрасывается — audit, часть 5.
    """
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in ("open", "high", "low", "close", "volume", "value", "begin")
               if c not in df.columns]
    if missing:
        LOG.error("  [%s] неожиданная схема MOEX, нет колонок: %s", ticker, missing)
        return None

    df = df.rename(columns={"value": "amount"})
    df["begin"] = pd.to_datetime(df["begin"])

    numeric = ["open", "high", "low", "close", "volume", "amount"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    df = (df.drop_duplicates(subset=["begin"])
            .sort_values("begin")
            .reset_index(drop=True))
    df = df[df["close"] > 0].reset_index(drop=True)

    df, dropped = drop_incomplete_daily(df)
    if dropped and ticker:
        LOG.info("  [%s] отброшен незакрытый бар текущего дня — "
                 "последнее закрытие = предыдущая сессия", ticker)
    return df


def drop_incomplete_daily(df, now=None):
    """Отбросить бар текущего дня, если торги ещё не завершились.

    MOEX отдаёт дневную свечу сегодняшнего дня в любой момент времени. До
    закрытия сессии её close — это не итоговая цена, а текущая рыночная.
    Использовать её как последнее закрытие нельзя: от этого зависят
    last_price, sigma_дневн, окно Kronos, база z и ось графика.

    Возвращает (df, dropped_bool).
    """
    if df is None or df.empty:
        return df, False
    now = now or datetime.datetime.now(tz=ZoneInfo("Europe/Moscow"))
    minutes_now = now.hour * 60 + now.minute
    if minutes_now >= MOEX_SESSION_END_MIN:
        return df, False  # сессия завершена — сегодняшний бар финальный
    if df["begin"].iloc[-1].date() != now.date():
        return df, False  # последний бар — уже закрытый предыдущий день
    return df.iloc[:-1].reset_index(drop=True), True


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

    LOG.info("[Kronos] Loading model...")
    tok = KronosTokenizer.from_pretrained(TOKENIZER_PATH)
    model = Kronos.from_pretrained(MODEL_PATH)

    # Force CPU mode due to RTX 5060 Ti CUDA incompatibility (sm_120 not supported)
    device = "cpu"
    predictor = KronosPredictor(model, tok, device=device, max_context=2048)
    LOG.info("[Kronos] Ready (device=%s)", device)
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
        LOG.error("  [Kronos] ERROR: %s", e)
        # Полный трейс — только в run-лог: stdout уходит в чат дословно,
        # и стек-трейс там занимал бы больше места, чем сам отчёт.
        LOG.debug("  [Kronos] traceback:", exc_info=True)
        return None


# ── TimesFM 2.5 ───────────────────────────────────────────────

def load_timesfm():
    """Load and compile TimesFM 2.5."""
    from timesfm.timesfm_2p5 import timesfm_2p5_torch as tftorch
    from timesfm.configs import ForecastConfig

    LOG.info("[TimesFM] Loading model...")
    # Фикс 2026-09-08: строго офлайн — HF геоблок 403, онлайн fallback запрещён
    try:
        model = tftorch.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch", local_files_only=True)
    except Exception as e:
        LOG.error("[TimesFM] local cache missing — пропуск прогноза "
                  "(offline only): %s", e)
        raise RuntimeError(
            f"TimesFM кэш отсутствует offline: {e} — пропустите прогон, "
            f"не пробуйте онлайн (HF 403)") from e

    config = ForecastConfig(max_context=512, max_horizon=TIMESFM_MAX_HORIZON)
    LOG.info("[TimesFM] Compiling...")
    model.compile(config)
    LOG.info("[TimesFM] Ready (device=cpu)!")
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
    """Таблица сравнения по одному горизонту. Возвращает готовый текст.

    Единственный источник строк таблицы: их используют и вывод в чат, и
    отчёт comparison_*.log. Копия таблицы в двух местах — это и был audit
    M-дрейф (расъехавшаяся ширина колонок), поэтому дублировать разметку
    здесь нельзя.
    """
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
    """Текстовые вердикты по одному горизонту. Возвращает готовый текст.

    Источник строк интерпретации для обоих каналов вывода — как и
    render_table(), копия здесь означала бы расхождение отчёта с чатом.
    Рендерится только HORIZONS[0]: на 60 и 90 днях вердикт повторяет тот же
    текст, различаясь лишь числом в заголовке.
    """
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
    """Сводный вердикт по строке таблицы.

    Здесь живёт бизнес-правило, которое нельзя потерять при правках: две
    модели, обе сказавшие «ДЕРЖАТЬ», дают согласие низкой ставки —
    «НЕЙТРАЛЬНО/СРЕДНЯЯ», а не «СОВПАДЕНИЕ/ВЫСОКАЯ». Иначе две модели,
    которые ничего не предсказали, читались бы как уверенный сигнал
    (это тот же класс ошибки, что C4/C5). Добавляя ветку, проверь, что
    она не относится к случаю «обе модели бездействуют».
    """
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
# Фазы разделены по стоимости и по способу отказа, а не по числу строк:
# загрузка моделей (самая долгая, своя точка отказа), сбор данных, прогноз
# по горизонтам, вывод. Отчёт вынесен отдельно — у него свой контракт формата.


def _banner(now_str):
    """Шапка прогона. Те же строки дублируются в отчёт (см. _write_report)."""
    LOG.info("=" * 80)
    LOG.info("  TIMESFM 2.5 + KRONOS MINI — COMPARISON")
    LOG.info("  %s", now_str)
    LOG.info("  units: daily bars (interval=%d), horizons=%s days, seeds=%d",
             INTERVAL, HORIZONS, SEED)
    LOG.info("  signal mode: %s (порог ±%.0fσ h-шаговой диффузии)",
             SIGNAL_MODE, Z_THRESHOLD)
    LOG.info("=" * 80)


def _fetch_ticker(ticker):
    """Скачать бары MOEX. None — тикер пропущен, у него не будет строк."""
    LOG.info("\n%s", "=" * 80)
    LOG.info("[%s] Fetching MOEX ISS data (%d days)...", ticker, HISTORY_DAYS)
    df = fetch_moex_candles(ticker)
    if df is None or len(df) < MIN_BARS:
        got = 0 if df is None else len(df)
        LOG.error("  ERROR: not enough data for %s (%d bars, need %d)",
                  ticker, got, MIN_BARS)
        return None
    return df


def _ticker_context(df):
    """Контекст окна, разобранный один раз на тикер.

    Календарь торгов выводится из данных, а не предполагается (audit C3);
    last_price/mean/sigma берутся одним вызовом window_stats(), чтобы цена
    последнего бара не вычислялась в коде двумя разными способами.
    """
    last_price, mean_price, sigma_d = window_stats(df)
    trades_weekends = has_weekend_sessions(df)
    last_date = df["begin"].iloc[-1]

    LOG.info("  Got %d daily bars", len(df))
    LOG.info("  Period: %s — %s", df["begin"].iloc[0].date(), last_date.date())
    LOG.info("  Last close: %.2f RUB", last_price)
    LOG.info("  Trading calendar: %s",
             "7 days/week (weekend sessions)" if trades_weekends else "Mon-Fri only")
    LOG.info("  Window: mean=%.2f sigma_d=%.2f%%/day", mean_price, sigma_d * 100)

    return {
        "df": df,
        "values_np": df["close"].values.astype(np.float32),
        "last_price": last_price,
        "mean_price": mean_price,
        "sigma_d": sigma_d,
        "last_date": last_date,
        "trades_weekends": trades_weekends,
    }


def _forecast_horizon(ticker, horizon, ctx, kronos_predictor, timesfm_model,
                      chart_data):
    """Прогноз обеих моделей на один горизонт. Возвращает строку результата.

    Отказ любой из моделей не прерывает прогон: строка получает NO_DATA, и
    compare_signals() превращает такую пару в «нет данных», а не в согласие.

    z печатается через _num(): при SIGNAL_MODE == "abs" compute_signal()
    возвращает z = None, и форматирование None как числа падало бы с TypeError.
    """
    LOG.info("\n  --- Horizon: %s days ---", horizon)

    # Kronos. Горизонт в днях == число баров, потому что INTERVAL=24.
    # Предупреждение выставляем по фактическому усечению, а не по
    # магическому порогу (audit M2).
    kronos_bars = min(horizon, KRONOS_MAX_PRED_LEN)
    capped = kronos_bars < horizon
    LOG.info("  [Kronos] Predicting %d bars ahead%s", kronos_bars,
             (f" (запрошено {horizon}d, cap={KRONOS_MAX_PRED_LEN})"
              if capped else "..."))
    kronos_preds = kronos_predict(kronos_predictor, ctx["df"], kronos_bars,
                                  ctx["trades_weekends"])

    kronos_z = kronos_excess = None
    if kronos_preds is not None:
        kronos_horizon_pred = float(kronos_preds[-1])
        kronos_signal, kronos_chg, kronos_z = compute_signal(
            kronos_horizon_pred, ctx["last_price"], ctx["sigma_d"], horizon)
        kronos_excess = reversion_excess(
            kronos_horizon_pred, ctx["mean_price"], ctx["sigma_d"],
            ctx["last_price"], horizon)
        LOG.info("  [Kronos] %dd forecast: %.2f (%+.2f%%, z=%s) -> [%s]",
                 horizon, kronos_horizon_pred, kronos_chg,
                 _num(kronos_z, "+.2f"), kronos_signal)
    else:
        kronos_signal = NO_DATA
        kronos_chg = None
        kronos_horizon_pred = None

    # TimesFM. max_context=512 покрывает всю историю (~112 дневных баров),
    # усечения больше нет (audit H4).
    LOG.info("  [TimesFM] Predicting %dd ahead...", horizon)
    timesfm_preds = None
    timesfm_z = timesfm_excess = None
    try:
        timesfm_preds = timesfm_forecast(timesfm_model, ctx["values_np"], horizon)
        timesfm_horizon_pred = float(timesfm_preds[-1])
        timesfm_signal, timesfm_chg, timesfm_z = compute_signal(
            timesfm_horizon_pred, ctx["last_price"], ctx["sigma_d"], horizon)
        timesfm_excess = reversion_excess(
            timesfm_horizon_pred, ctx["mean_price"], ctx["sigma_d"],
            ctx["last_price"], horizon)
        LOG.info("  [TimesFM] %dd forecast: %.2f (%+.2f%%, z=%s) -> [%s]",
                 horizon, timesfm_horizon_pred, timesfm_chg,
                 _num(timesfm_z, "+.2f"), timesfm_signal)
    except Exception as e:
        # timesfm_preds обязателен к сбросу: проверка `is not None` ниже
        # кэширует прогноз в график, и упавший прогноз попал бы в PNG.
        timesfm_preds = None
        timesfm_signal = NO_DATA
        timesfm_chg = None
        timesfm_horizon_pred = None
        LOG.error("  [TimesFM] ERROR: %s", e)

    # Кэш истории + прогноза первого горизонта, чтобы секция графиков не
    # перезапрашивала MOEX и не гоняла модель второй раз. Даты считаются
    # ТОЙ ЖЕ функцией, что и метки для Kronos, поэтому ось графика совпадает
    # с длиной прогноза (audit H2).
    if horizon == HORIZONS[0] and timesfm_preds is not None:
        chart_data[ticker] = {
            "df": ctx["df"],
            "last_price": ctx["last_price"],
            "last_date": ctx["last_date"],
            "tf_preds": timesfm_preds,
            "forecast_dates": future_bars(
                ctx["last_date"], len(timesfm_preds), ctx["trades_weekends"]),
        }

    agreement, confidence = compare_signals(kronos_signal, timesfm_signal)

    return {
        "ticker": ticker,
        "horizon": horizon,
        "last_price": ctx["last_price"],
        "mean_price": ctx["mean_price"],
        "sigma_d": ctx["sigma_d"],
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
    }


def _write_report(now_str, tables, interpretation, diagnostics):
    """Дописать секцию прогона в logs/comparison_YYYYMMDD.log. Возвращает путь.

    Формат закреплён logs/README.md: строки `units:` и `signal mode:` —
    маркер, по которому записи до и после 2026-10-14 различаются. Менять
    разметку здесь нельзя, иначе старые прогоны станут неотличимы от новых,
    а именно по ним считалась вся сверка юнитов.
    """
    log_dir = os.path.join(SCRIPT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir, f"comparison_{datetime.date.today().strftime('%Y%m%d')}.log")

    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(f"\n{'='*80}\n")
        lf.write("TIMESFM 2.5 + KRONOS MINI — COMPARISON\n")
        lf.write(f"{now_str}\n")
        lf.write(f"units: daily bars (interval={INTERVAL}), "
                 f"horizons={HORIZONS} days, seeds={SEED}\n")
        lf.write(f"signal mode: {SIGNAL_MODE} (порог ±{Z_THRESHOLD:.0f}σ)\n")
        lf.write(f"{'='*80}\n")
        for horizon, table in zip(HORIZONS, tables):
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
    return log_path


def main():
    """Полный прогон: баннер → модели → тикеры × горизонты → отчёт → график."""
    _setup_logging()
    now_str = datetime.datetime.now(tz=ZoneInfo("Europe/Moscow")).strftime(
        "%Y-%m-%d %H:%M MSK")
    _banner(now_str)

    torch.manual_seed(SEED)  # см. SEED

    LOG.info("\n[INIT] Loading Kronos-mini...")
    kronos_predictor = load_kronos()

    LOG.info("\n[INIT] Loading TimesFM 2.5...")
    timesfm_model = load_timesfm()

    all_results = []
    # кэш истории + прогноза (первый горизонт) на тикер, чтобы не дублировать
    # запрос/инференс в секции графиков
    chart_data = {}

    for ticker in TICKERS:
        df = _fetch_ticker(ticker)
        if df is None:
            continue
        ctx = _ticker_context(df)
        for horizon in HORIZONS:
            all_results.append(_forecast_horizon(
                ticker, horizon, ctx, kronos_predictor, timesfm_model, chart_data))

    # ── Output comparison table ────────────────────────────────
    LOG.info("\n%s", "=" * 80)
    LOG.info("COMPARISON TABLE")
    LOG.info("=" * 80)

    tables = []
    for horizon in HORIZONS:
        table = render_table(all_results, horizon)
        tables.append(table)
        LOG.info("\n--- Прогноз на %d дней ---", horizon)
        LOG.info("%s", table)

    # ── Interpretation ─────────────────────────────────────────
    interpretation = render_interpretation(all_results, HORIZONS[0])
    LOG.info("\n%s", "=" * 80)
    LOG.info("INTERPRETATION")
    LOG.info("=" * 80)
    LOG.info("%s", interpretation)

    # ── Diagnostics: is the forecast more than mechanical reversion? ──
    diagnostics = render_diagnostics(all_results)
    LOG.info("\n%s", "=" * 80)
    LOG.info("MODEL DIAGNOSTICS (z_excess ~ 0 => прогноз = арифметика окна)")
    LOG.info("=" * 80)
    LOG.info("%s", diagnostics)

    # ── Save chart (первый горизонт) ───────────────────────────
    LOG.info("\n%s", "=" * 80)
    LOG.info("Saving charts...")
    save_chart(chart_data, HORIZONS[0])

    # ── Save results to log file ───────────────────────────────
    log_path = _write_report(now_str, tables, interpretation, diagnostics)
    LOG.info("Log saved: %s", log_path)


def save_chart(chart_data, horizon, out_path=None):
    """Построить график истории + прогноза. Ось X = реальные торговые дни."""
    tickers = [t for t in TICKERS if t in chart_data]
    if not tickers:
        LOG.info("  График не построен: нет ни одного успешного прогноза TimesFM.")
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
    LOG.info("Chart saved: %s", out_path)


if __name__ == "__main__":
    main()