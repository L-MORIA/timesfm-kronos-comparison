"""Проверки инвариантов timesfm_kronos_compare.py (audit 2026-10-14).

Запуск без зависимостей:
    python tests/test_units.py          # офлайн-проверки
    python tests/test_units.py --live   # плюс реальный запрос к MOEX ISS
"""
import importlib.util
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "tkm", os.path.join(ROOT, "timesfm_kronos_compare.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

failures = []


def check(name, actual, expected):
    ok = actual == expected
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"         got={actual!r} expected={expected!r}")
        failures.append(name)


print("=== compare_signals: отказ ≠ согласие (audit C4) ===")
check("оба ERROR -> НЕТ ДАННЫХ",
      m.compare_signals(m.NO_DATA, m.NO_DATA), ("НЕТ ДАННЫХ", "НЕТ ДАННЫХ"))
check("Kronos упал -> НЕТ ДАННЫХ",
      m.compare_signals(m.NO_DATA, "ДЕРЖАТЬ"), ("НЕТ ДАННЫХ", "НЕТ ДАННЫХ"))
check("TimesFM упал -> НЕТ ДАННЫХ",
      m.compare_signals("ПОКУПКА", m.NO_DATA), ("НЕТ ДАННЫХ", "НЕТ ДАННЫХ"))

print("=== compare_signals: таксономия (audit C5) ===")
check("оба ПОКУПКА -> СОВПАДЕНИЕ/ВЫСОКАЯ",
      m.compare_signals("ПОКУПКА", "ПОКУПКА"), ("СОВПАДЕНИЕ", "ВЫСОКАЯ"))
check("оба ПРОДАЖА -> СОВПАДЕНИЕ/ВЫСОКАЯ",
      m.compare_signals("ПРОДАЖА", "ПРОДАЖА"), ("СОВПАДЕНИЕ", "ВЫСОКАЯ"))
check("оба ДЕРЖАТЬ -> НЕЙТРАЛЬНО/СРЕДНЯЯ",
      m.compare_signals("ДЕРЖАТЬ", "ДЕРЖАТЬ"), ("НЕЙТРАЛЬНО", "СРЕДНЯЯ"))
check("разные -> РАСХОЖДЕНИЕ/НИЗКАЯ",
      m.compare_signals("ПОКУПКА", "ПРОДАЖА"), ("РАСХОЖДЕНИЕ", "НИЗКАЯ"))

# Регрессия C5: ветка НЕЙТРАЛЬНО должна быть достижима БЕЗ отказа модели,
# иначе она маскирует ошибку под «среднюю уверенность».
reached = {m.compare_signals(a, b)[0]
           for a, b in [("ДЕРЖАТЬ", "ДЕРЖАТЬ"), ("ПОКУПКА", "ПРОДАЖА")]}
check("НЕЙТРАЛЬНО достижимо без NO_DATA", "НЕЙТРАЛЬНО" in reached, True)

print("=== compute_signal: порог по волатильности ===")
check("SIGNAL_MODE по умолчанию", m.SIGNAL_MODE, "z_vol")
SD, LAST, H = 0.02, 100.0, 30
DENOM = SD * LAST * (H ** 0.5)
check("ход ровно +1σ -> ПОКУПКА",
      m.compute_signal(LAST + DENOM, LAST, SD, H)[0], "ПОКУПКА")
check("ход ровно -1σ -> ПРОДАЖА",
      m.compute_signal(LAST - DENOM, LAST, SD, H)[0], "ПРОДАЖА")
check("ход +0.5σ -> ДЕРЖАТЬ",
      m.compute_signal(LAST + DENOM * 0.5, LAST, SD, H)[0], "ДЕРЖАТЬ")
check("возвращает 3 значения", len(m.compute_signal(105.0, 100.0, SD, H)), 3)
check("z = ход / denom", round(m.compute_signal(105.0, 100.0, SD, H)[2], 6),
      round(5.0 / DENOM, 6))

print("=== compute_signal: отвязка от цены и горизонта (audit часть 3) ===")
# Тот же относительный ход (+3%) на дешёвом и дорогом активе -> одинаковый z
z_cheap = m.compute_signal(10.0 * 1.03, 10.0, SD, H)[2]
z_rich = m.compute_signal(1000.0 * 1.03, 1000.0, SD, H)[2]
check("z одинаков при цене 10 и 1000", round(z_cheap, 9), round(z_rich, 9))
check("вердикт одинаков при цене 10 и 1000",
      m.compute_signal(10.0 * 1.03, 10.0, SD, H)[0],
      m.compute_signal(1000.0 * 1.03, 1000.0, SD, H)[0])
# Тот же абсолютный ход на 90 днях слабее, чем на 30 (шум ~ sqrt(h))
check("+11 руб на 30d -> ПОКУПКА",
      m.compute_signal(111.0, 100.0, SD, 30)[0], "ПОКУПКА")
check("+11 руб на 90d -> ДЕРЖАТЬ",
      m.compute_signal(111.0, 100.0, SD, 90)[0], "ДЕРЖАТЬ")
# Регрессия, ради которой всё затевалось: волатильный актив не должен
# выглядеть «нейтральным» только из-за высокой цены.
check("высокая волатильность съедает тот же % хода",
      m.compute_signal(111.0, 100.0, 0.05, 30)[0], "ДЕРЖАТЬ")

print("=== reversion_excess: диагностика отката ===")
LM, MEAN, SD3 = 100.0, 80.0, 0.02
check("прогноз = среднее -> z_excess 0",
      round(m.reversion_excess(MEAN, MEAN, SD3, LM, 30), 9), 0.0)
check("прогноз = последняя -> z_excess > 1",
      m.reversion_excess(LM, MEAN, SD3, LM, 30) > 1, True)
check("sigma=0 -> None", m.reversion_excess(90.0, 80.0, 0.0, LM, 30), None)

print("=== render_diagnostics ===")
diag = m.render_diagnostics([{"ticker": "T", "horizon": 30,
                              "kronos_excess": 0.21, "timesfm_excess": 1.52}])
check("Kronos помечен как откат", "откат к среднему" in diag, True)
check("TimesFM помечен как свой сигнал", "собственный сигнал" in diag, True)

print("=== future_bars: календарь из данных (audit C3) ===")
fri = pd.Timestamp("2026-10-16")
wd_only = m.future_bars(fri, 10, trades_weekends=False)
check("Mon-Fri: стартует со следующего пн", wd_only[0].date().isoformat(), "2026-10-19")
check("Mon-Fri: 10 баров", len(wd_only), 10)
check("Mon-Fri: нет сб/вс", any(d.weekday() >= 5 for d in wd_only), False)

seven = m.future_bars(fri, 10, trades_weekends=True)
check("7 дней: стартует со следующего дня", seven[0].date().isoformat(), "2026-10-17")
check("7 дней: 10 баров", len(seven), 10)
check("7 дней: включает сб", any(d.weekday() == 5 for d in seven), True)
# Дрейф календаря: 90 будних баров растянулись бы на ~126 дней -> было бы ложно
check("7 дней: 90 баров = 90 дней",
      (m.future_bars(fri, 90, True).iloc[-1] - fri).days, 90)

print("=== формат: None не выдаётся за прогноз ===")
check("_num(None)", m._num(None, ">9.2f"), "—")
check("_num(float)", m._num(287.04, ">9.2f").strip(), "287.04")

print("=== рендер: NO_DATA не печатается как согласие ===")
res = [{"ticker": "SBERP", "horizon": 30, "last_price": 287.04, "sigma_d": 0.017,
        "kronos_pred": None, "kronos_signal": m.NO_DATA, "kronos_chg": None,
        "kronos_z": None, "kronos_excess": None,
        "timesfm_pred": None, "timesfm_signal": m.NO_DATA, "timesfm_chg": None,
        "timesfm_z": None, "timesfm_excess": None,
        "agreement": m.NO_DATA, "confidence": m.NO_DATA}]
table = m.render_table(res, 30)
interp = m.render_interpretation(res, 30)
check("в таблице НЕТ ДАННЫХ", m.NO_DATA in table, True)
check("нет выдуманной цены", "287.04 RUB (" not in interp, True)
check("рекомендация НЕТ ДАННЫХ",
      "НЕТ ДАННЫХ" in m._recommendation(m.NO_DATA, m.NO_DATA), True)
check("СОВПАДЕНИЕ не упоминается", "СОВПАДЕНИЕ" not in interp, True)
check("в таблице есть колонка z", "z" in table, True)
check("в таблице есть колонка σ", "σ" in table, True)

print("=== конфигурация ===")
check("HORIZONS <= max_horizon", max(m.HORIZONS) <= m.TIMESFM_MAX_HORIZON, True)
check("INTERVAL = 24 (дневные бары)", m.INTERVAL, 24)
check("Kronos cap не режет горизонт",
      min(max(m.HORIZONS), m.KRONOS_MAX_PRED_LEN), max(m.HORIZONS))
check("Z_THRESHOLD = 1.0", m.Z_THRESHOLD, 1.0)

print("=== график: ось X совпадает с прогнозом (audit H2) ===")
import matplotlib
matplotlib.use("Agg")
import tempfile

n_bars, n_fc = 112, 30
idx = pd.bdate_range("2026-01-01", periods=n_bars, freq="B")
synth_df = pd.DataFrame({"begin": idx, "close": [100 + i * 0.1 for i in range(n_bars)]})
fc_dates = m.future_bars(idx[-1], n_fc, trades_weekends=True)
chart = {"SBERP": {"df": synth_df, "last_price": 111.1,
                   "last_date": idx[-1],
                   "tf_preds": [111.0 + i * 0.05 for i in range(n_fc)],
                   "forecast_dates": fc_dates}}
with tempfile.TemporaryDirectory() as td:
    png = os.path.join(td, "chart.png")
    m.save_chart(chart, 30, out_path=png)
    check("PNG создан", os.path.exists(png), True)
    check("PNG непустой", os.path.getsize(png) > 5000, True)
check("дат прогноза == точек прогноза", len(fc_dates), n_fc)
check("последняя дата прогноза = последний бар + 30 дней",
      (fc_dates.iloc[-1] - idx[-1]).days, 30)

bad = {"SBERP": {"df": synth_df, "last_price": 111.1, "last_date": idx[-1],
                 "tf_preds": [1.0] * 30, "forecast_dates": fc_dates[:10]}}
with tempfile.TemporaryDirectory() as td:
    raised = False
    try:
        m.save_chart(bad, 30, out_path=os.path.join(td, "bad.png"))
    except ValueError:
        raised = True
    check("рассинхрон дат/точек -> ValueError", raised, True)

print("=== пустой chart_data: не рисуем пустышку ===")
with tempfile.TemporaryDirectory() as td:
    png = os.path.join(td, "none.png")
    m.save_chart({}, 30, out_path=png)
    check("файл не создан", os.path.exists(png), False)

if "--live" in sys.argv:
    print("=== MOEX ISS: дневные бары, оборот, статистики окна (audit H3) ===")
    for ticker in m.TICKERS:
        df = m.fetch_moex_candles(ticker)
        if df is None:
            check(f"{ticker} загружен", False, True)
            continue
        check(f"{ticker}: >= MIN_BARS ({len(df)} >= {m.MIN_BARS})",
              len(df) >= m.MIN_BARS, True)
        check(f"{ticker}: есть amount", "amount" in df.columns, True)
        check(f"{ticker}: нет дублей дат", int(df["begin"].duplicated().sum()), 0)
        check(f"{ticker}: нет NaN", int(df.isna().sum().sum()), 0)
        check(f"{ticker}: close > 0", bool((df["close"] > 0).all()), True)

        gaps = int((df["begin"].diff().dt.days > 4).sum())
        wknd = m.has_weekend_sessions(df)
        n_wknd = int((df["begin"].dt.weekday >= 5).sum())
        last, mean, sd = m.window_stats(df)
        print(f"         баров={len(df)} период={df['begin'].iloc[0].date()}"
              f"..{df['begin'].iloc[-1].date()} пропусков>4д={gaps}")
        print(f"         last={last:.2f} mean={mean:.2f} sigma_d={sd*100:.2f}%/day")
        print(f"         выходных баров={n_wknd} -> future_bars(freq="
              f"{'D' if wknd else 'B'})")
        check(f"{ticker}: sigma_d > 0", sd > 0, True)
        check(f"{ticker}: 0 < mean <= max", 0 < mean <= df["close"].max(), True)
        # Дрейф календаря недопустим: N баров == N дней назад
        n = 90
        fb = m.future_bars(df["begin"].iloc[-1], n, wknd)
        drift = (fb.iloc[-1] - df["begin"].iloc[-1]).days
        check(f"{ticker}: {n} баров ≈ {n} дней (дрейф={drift})", abs(drift - n) <= 2, True)

print()
if failures:
    print(f"FAILED: {len(failures)} -> {failures}")
    sys.exit(1)
print("ALL CHECKS PASSED")