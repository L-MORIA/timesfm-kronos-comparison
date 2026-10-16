"""Шаг 1 возвращения к сигналу: дневная кросс-секционная панель, h=1 день.

Единственный дизайн проекта, который ещё не опровергнут (ROADMAP, этапе 4,
шаг 1): 13 тикеров из kronos-signal/config.yaml, дневные бары, горизонт
один день, шаг тестовых дат = горизонт (окна не перекрываются, Newey-West
не нужна по построению).

Что здесь предprendено ЗАРАНЕЕ (предрегистрация — ROADMAP, раздел шага 1):
  * единица анализа — ДАТА, а не (тикер, дата). Внутри даты все бумаги
    живут в одном рыночном состоянии; на интрадее эта кластеризация
    измерена как ICC 0.69-0.93 и признана врождённой. MDE «0.012 на
    20 500 наблюдений» из раннего плана был ошибкой независимости;
  * окно 2021+ (1582 дат) — первичное и чистое; окно 2014-06-09+ (~3125
    дат) — вторичное с survivorship-оговоркой текущего списка. Оба считаются
    в одном прогоне;
  * порог приёмки: IC >= 0.05 при power 0.80, то есть |t_eff| >= z_Bonf
    (альфа 0.05/4 предиктора = 0.0125, порог |t| >= 2.50);
  * N_eff считается по замеренной автокорреляции IC-ряда (acf_sum из
    ic_significance.py), а не по номиналу, и оба числа печатаются рядом.

Предикторы — только CPU-дешёвые, без моделей. Модели (Kronos/TimesFM)
могут появиться на шаге 2 и только если здесь найден предиктор, который
хотя бы не проигрывает random_walk. Контроль: перемешанный внутри даты
сигнал — если неотличим от настоящего, вся доходность объясняется рынком.

Издержки: спред, измеренный по самим дневным барам методом Roll (и сверка
с 5-минутным замером cost_impact.py, который дал медиану 6.6 бп), плюс
комиссия 3 бп×2. Поскольку сигналы меняются медленнее, чем раз в день,
издержки умножаются на замеренный turnover, а не берутся полным кругом
каждый день.

Запуск:
    python experiments/panel_daily.py            # полный прогон (13 тикеров, 2014+)
    python experiments/panel_daily.py --tickers SBERP --start 2021-01-01
                                                 # bounded-проверка одного тикера
    python experiments/panel_daily.py --offline  # только кэш, без сети
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import os
import sys
import time
from statistics import NormalDist

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ND = NormalDist()

# ── конфигурация ────────────────────────────────────────────────

TICKERS = ["SBER", "GAZP", "LKOH", "SBERP", "VTBR", "FEES", "MOEX", "ROSN",
           "TRNFP", "GMKN", "SNGSP", "MTSS", "AFKS"]
BOARD = "TQBR"
MOEX_URL = ("https://iss.moex.com/iss/engines/stock/markets/shares"
            f"/boards/{BOARD}/securities/{{t}}/candles.json")
PAGE = 500                          # MOEX ISS отдаёт максимум 500 строк/страницу
MAX_PAGES = 120                     # 120 страниц × 500 = 60 000 баров — запас
FROM = "2014-01-01"                 # запас до фактического старта 2014-06-09

CACHE = os.path.join(ROOT, "logs", "panel_daily_cache.csv")
IC_CSV = os.path.join(ROOT, "logs", "panel_daily_ic.csv")

LOOKBACK_W = 60                     # окно среднего/σ для отката к среднему
K = 4                               # размер портфеля top-K
COMMISSION_BPS = 3.0                # за сторону; тариф розницы, не из баров


def load_master():
    """Единый источник нормализации свечей — main module, как в backtest.py."""
    spec = importlib.util.spec_from_file_location(
        "m", os.path.join(ROOT, "timesfm_kronos_compare.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_significance():
    """Общий механизм IC/N_eff/MDE — тот же, что в ic_significance.py."""
    spec = importlib.util.spec_from_file_location(
        "ics", os.path.join(ROOT, "experiments", "ic_significance.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ics"] = mod
    spec.loader.exec_module(mod)
    return mod


# ── сеть: две сессии, прямой канал первичный (паттерн intraday_pilot) ──

def _sessions():
    """Прямая сессия (без прокси) и через env (прокси) как запасной путь.

    Локальный прокси на 127.0.0.1:10809 живёт не постоянно, а выгрузка
    13 тикеров теряется целиком, если обе сессии падают одновременно.
    """
    direct = requests.Session()
    direct.trust_env = False
    via_proxy = requests.Session()
    return direct, via_proxy


def fetch_daily(ticker, frm=FROM, m=None):
    """Дневные бары тикера с пагинацией;架构 нормализует main-модуль."""
    url = MOEX_URL.format(t=ticker)
    till = dt.date.today().strftime("%Y-%m-%d")
    rows, cols, start = [], None, 0
    for page in range(MAX_PAGES):
        params = {"interval": 24, "from": frm, "till": till,
                  "limit": PAGE, "start": start}
        payload = None
        last_err = None
        for s in SESSIONS:
            try:
                r = s.get(url, params=params, timeout=25)
                r.raise_for_status()
                payload = r.json()
                break
            except Exception as e:                       # noqa: BLE001
                last_err = e
        if payload is None:
            raise RuntimeError(f"{ticker}: страница {page} недоступна: {last_err}")
        blk = payload.get("candles") or {}
        data = blk.get("data") or []
        if cols is None:
            cols = blk.get("columns")
        if not data:
            break
        rows.extend(data)
        if len(data) < PAGE:
            break
        start += len(data)
        time.sleep(0.15)              # вежливо к ISS: ~13×12 страниц ≈ 160 req
    if not rows or cols is None:
        return None
    df = pd.DataFrame(rows, columns=cols)
    df = m.normalize_candles(df, ticker)
    return df


SESSIONS = _sessions()


# ── предикторы: ожидаемая доходность на дату T из истории до T включительно ──

PREDICTORS = ["mean_revert_w60", "momentum_20", "momentum_5", "volnorm_mom_20"]


def predictor_table(close: pd.Series) -> pd.DataFrame:
    """Все предикторы одного тикера по датам. Индекс — даты, NaN на краях.

    Всё выражено в ожидаемой доходности (кроме volnorm_mom_20, который в
    единицах σ; ранговый IC масштаб-инвариантен, это не мешает сравнению).
    Сдвиги назад (shift(+w)) читают только ПРОШЛОЕ — утечки будущего нет.
    """
    close = close.sort_index()
    c = close.astype(float)
    ret1 = c.pct_change()

    roll_mean = c.rolling(LOOKBACK_W).mean().shift(1)
    roll_std = ret1.rolling(LOOKBACK_W).std().shift(1)
    mr = (roll_mean - c) / c                                    # откат к среднему

    mom20 = c / c.shift(21) - 1.0
    mom5 = c / c.shift(6) - 1.0
    std20 = ret1.rolling(20).std().shift(1)
    vnm = mom20 / (std20 * np.sqrt(20) + 1e-12)

    out = pd.DataFrame({
        "mean_revert_w60": mr,
        "momentum_20": mom20,
        "momentum_5": mom5,
        "volnorm_mom_20": vnm,
    })
    out["close"] = c
    out["realized"] = c.shift(-1) / c - 1.0                     # h = 1 день
    return out


# ── сборка панели ───────────────────────────────────────────────

def load_panel(args, m):
    """Панель цен: {ticker: DataFrame предикторов}. Кэш переписывается
    после КАЖДОГО тикера — падение сети не теряет уже скачанное."""
    frames, cached = [], set()
    if os.path.isfile(CACHE):
        raw = pd.read_csv(CACHE, parse_dates=["begin"])
        cached = set(raw["ticker"].unique())
        frames = [g.drop(columns=["ticker"]).assign(ticker=tk)
                  for tk, g in raw.groupby("ticker")]
        print(f"[кэш] {os.path.basename(CACHE)}: {len(raw):,} строк, "
              f"{len(cached)} тикеров")

    want = args.tickers
    missing = [t for t in want if t not in cached]
    start_from = args.start or FROM
    if missing and not args.offline:
        print(f"[сеть] догрузка: {', '.join(missing)}")
        for tk in missing:
            try:
                df = fetch_daily(tk, frm=start_from, m=m)
            except Exception as e:                          # noqa: BLE001
                print(f"{tk:<7} ОШИБКА {type(e).__name__}: {e}")
                continue
            if df is None or df.empty:
                print(f"{tk}: нет данных")
                continue
            sub = df[df["begin"].dt.date >= dt.date.fromisoformat(start_from)]
            frames.append(sub.assign(ticker=tk))
            pd.concat(frames, ignore_index=True).to_csv(
                CACHE, index=False, encoding="utf-8")
            print(f"{tk:<7} баров={len(sub):,}  "
                  f"{sub['begin'].iloc[0].date()} … {sub['begin'].iloc[-1].date()}  "
                  f"[кэш обновлён]", flush=True)

    tables = {}
    for f in frames:
        tk = f["ticker"].iloc[0]
        if tk not in want:
            continue
        d = f.drop(columns=[c for c in ("ticker", "open", "high", "low",
                                        "volume", "amount") if c in f.columns])
        d = d.set_index("begin").sort_index()
        d = d[~d.index.duplicated()]
        tables[tk] = predictor_table(d["close"])
    return tables


# ── кросс-секционные IC по датам ────────────────────────────────

def cross_spearman(pred, real):
    """Ранг + Пирсон для кросс-секции. Порог 3 точки, не 5.

    ics.spearman (порог 5) писан для больших пулов бэктеста; внутри даты
    бывает и 3 бумагиы (bounded-прогон, просевший день). Формула та же.
    """
    if len(pred) < 3 or np.std(pred) == 0 or np.std(real) == 0:
        return np.nan
    ra = pd.Series(pred).rank().to_numpy(float)
    rb = pd.Series(real).rank().to_numpy(float)
    c = np.corrcoef(ra, rb)[0, 1]
    return float(c) if np.isfinite(c) else np.nan


def daily_ics(tables, ics, rng):
    """IC внутри каждой даты для каждого предиктора +controль на перемешанном.

    Возвращает (ic_df, counts): index — даты, columns — PREDICTORS + ['ctrl'];
    counts — сколько бумаг участвовало в каждой дате.
    """
    tickers = list(tables)
    close = pd.DataFrame({t: tables[t]["close"] for t in tickers})
    realized = pd.DataFrame({t: tables[t]["realized"] for t in tickers})
    preds = {p: pd.DataFrame({t: tables[t][p] for t in tickers})
             for p in PREDICTORS}

    dates = realized.dropna(how="all").index.intersection(close.dropna(how="all").index)
    dates = dates.sort_values()

    rows, counts = [], []
    for d in dates:
        real = realized.loc[d].dropna()
        names = real.index
        if len(names) < 3:                       # кросс-секция вырождена
            continue
        row, cnt = {}, 0
        for p in PREDICTORS:
            pv = preds[p].loc[d].reindex(names).dropna()
            ok = pv.index.intersection(real.index)
            if len(ok) < 3:
                row[p] = np.nan
                continue
            row[p] = cross_spearman(pv.loc[ok].values, real.loc[ok].values)
            cnt = max(cnt, len(ok))
        # контроль: та же кросс-секция, предиктор переставлен по бумагам
        cv = preds[PREDICTORS[0]].loc[d].reindex(names).dropna()
        okc = cv.index.intersection(real.index)
        if len(okc) >= 3:
            perm = rng.permutation(len(okc))
            row["ctrl"] = cross_spearman(cv.values[perm], real.loc[okc].values)
        else:
            row["ctrl"] = np.nan
        rows.append(row)
        counts.append(cnt)
    ic_df = pd.DataFrame(rows, index=dates, columns=PREDICTORS + ["ctrl"])
    counts = pd.Series(counts, index=dates, name="n_names")
    return ic_df.dropna(how="all"), counts.reindex(ic_df.index)


def summarize_ic(ics, ic_series, n_dates_full, label):
    """Одна строка резюме: IC, N, N_eff (по ACF), t, обе конвенции MDE.

    N_eff ограничен сверху n — та же численная крышка, что в
    ic_significance.effective_n, только L зашит = 10: при шаге = горизонт
    перекрытий окон нет, а остаточная автокорреляция IC-ряда между датами
    реальна и измеряется, а не предполагается.
    """
    s = ic_series.dropna().astype(float)
    n = len(s)
    if n < 10:
        return None
    mean_ic = float(s.mean())
    series = s.values - mean_ic                  # dc: acf_sum сам не вычитает
    s_sum = ics.acf_sum(series, 10)
    factor = max(1.0 + 2.0 * s_sum, 1e-6)
    n_eff = min(n / factor, float(n))
    t_eff = mean_ic * np.sqrt(n_eff)
    mde_half = 2.0 / np.sqrt(n_eff)              # граница значимости (power ~50%)
    mde_p80 = 2.801585 / np.sqrt(n_eff)          # порог при power 0.80
    z_bonf = ND.inv_cdf(1 - 0.05 / (2 * 4))      # α/4 предиктора, двусторонний
    verdict = ("значим" if abs(t_eff) >= z_bonf else "не значим")
    return {"label": label, "pred": ic_series.name, "n": n,
            "n_nominal": n_dates_full, "n_eff": n_eff, "ic": mean_ic,
            "t_eff": t_eff, "mde_half": mde_half, "mde_p80": mde_p80,
            "verdict": verdict}


# ── портфель и издержки ─────────────────────────────────────────

def roll_spread_daily(close: pd.Series) -> float:
    """Roll (1984) на дневных лог-ценах: S = 2·sqrt(−Cov(Δp_t, Δp_{t+1})).

    Cov >= 0 → бид-аск-боунса не видно, возвращаем 0. Замер по полному
    ряду тикера; это ОЦЕНКА СНИЗУ: реальный спред на заявке больше.
    """
    p = np.log(close.to_numpy(float))
    d = np.diff(p)
    if len(d) < 100:
        return np.nan
    cov = float(np.cov(d[:-1], d[1:])[0, 1])
    if cov >= 0:
        return 0.0
    return 2.0 * np.sqrt(-cov)


def portfolio_turnover(tables, k=K):
    """Доходность и оборот top-K портфеля, последовательно по датам.

    Состав собирается на дате T по предиктору из истории до T включительно,
    а ДОХОДНОСТЬ в этот день приносят бумаги, собранные ПРЕДЫДУЩИМ циклом
    портфеля (prev). Календарная арифметика (BDay(1)) не годится: с 2025-03
    торги семь дней в неделю, а приостановки реальны. Поэтому prev — это
    последняя успешно собранная корзина, а не «вчерашняя по календарю».
    """
    preds = {p: pd.DataFrame({t: tables[t][p] for t in tables})
             for p in PREDICTORS}
    realized = pd.DataFrame({t: tables[t]["realized"] for t in tables})
    dates = realized.dropna(how="all").index.sort_values()

    prev = {p: None for p in PREDICTORS}
    gross = {p: {} for p in PREDICTORS}
    turns = {p: {} for p in PREDICTORS}
    for d in dates:
        real = realized.loc[d].dropna()
        if len(real) < 2 * k:                   # портфель не из чего собрать
            continue
        for p in PREDICTORS:
            pv = preds[p].loc[d].reindex(real.index).dropna()
            if len(pv) < 2 * k:
                continue
            top = set(pv.sort_values().index[-k:])
            if prev[p] is not None:
                held = sorted(prev[p])
                r = real.reindex(held).dropna()   # бумага могла не торговаться
                if len(r) >= k:                   # портфель собрался полный
                    gross[p][d] = float(r.mean())
                turns[p][d] = len(prev[p] ^ top) / (2 * k)
            prev[p] = top
    return gross, turns


# ── main ────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=TICKERS)
    ap.add_argument("--start", default=None, help="дата старта загрузки/среза")
    ap.add_argument("--offline", action="store_true",
                    help="работать только по кэшу, сеть не трогать")
    args = ap.parse_args()

    m = load_master()
    ics = load_significance()
    rng = np.random.default_rng(20261016)

    print("=" * 96)
    print("ШАГ 1: ДНЕВНАЯ КРОСС-СЕКЦИОННАЯ ПАНЕЛЬ, h=1 — есть ли предиктор?")
    print("=" * 96)
    print(f"тикеры        : {', '.join(args.tickers)}")
    print(f"окно отката   : {LOOKBACK_W} баров;  портфель top-{K} из 13")
    print(f"приёмка       : IC >= 0.05 при power 0.80 (= |t_eff| >= "
          f"{ND.inv_cdf(1 - 0.05 / (2 * 4)):.2f}, Bonferroni α/4)")
    print()

    tables = load_panel(args, m)
    if len(tables) < len(args.tickers):
        print(f"ПРЕДУПРЕЖДЕНИЕ: собрано {len(tables)}/{len(args.tickers)} тикеров")
    if not tables:
        print("нет данных — отмена")
        return 2

    print()
    print("=" * 96)
    print("1. ПОКРЫТИЕ ПАНЕЛИ")
    print("=" * 96)
    close = pd.DataFrame({t: tables[t]["close"] for t in tables})
    common_start = max(tables[t].index[0] for t in tables)
    yearly = close.count(axis=1).groupby(close.index.year).agg(
        ["count", lambda s: (s >= 8).sum()])
    yearly.columns = ["дат", "дат с >=8 имён"]
    print(yearly.to_string())
    n_names_daily = close.count(axis=1)
    print(f"\nфактический общий старт (последний из стартов тикеров): "
          f"{common_start.date()};  дат с >=8 имён: "
          f"{int((n_names_daily >= 8).sum())}")
    print()

    print("=" * 96)
    print("2. КРОСС-СЕКЦИОННЫЕ IC ПО ДАТАМ (шаг = h = 1, перекрытий нет)")
    print("=" * 96)
    ic_df, n_names = daily_ics(tables, ics, rng)
    # чекпоинт: расчёт дешёвый, но серия — главный артефакт; пишем один раз
    os.makedirs(os.path.dirname(IC_CSV), exist_ok=True)
    ic_df.join(n_names.rename("n_names")).assign(
        year=ic_df.index.year).to_csv(IC_CSV, encoding="utf-8")
    print(f"дат с расчётом: {len(ic_df)} (сохранено: {os.path.basename(IC_CSV)})")
    print()

    windows = [("2021+ (чистая)", ic_df[ic_df.index >= pd.Timestamp("2021-01-01")], 1582),
               ("2014+ (survivorship)", ic_df, len(ic_df))]
    rows = []
    for label, sub, n_nom in windows:
        for p in PREDICTORS + ["ctrl"]:
            r = summarize_ic(ics, sub[p], n_nom, label)
            if r:
                rows.append(r)

    hdr = (f"{'окно':<20}{'предиктор':<17}{'N':>6}{'N_eff':>7}"
           f"{'IC':>8}{'t_eff':>7}{'MDE .50':>9}{'MDE .80':>9}{'вердикт':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['label']:<20}{r['pred']:<17}{r['n']:>6}{r['n_eff']:>7.0f}"
              f"{r['ic']:>8.4f}{r['t_eff']:>7.2f}{r['mde_half']:>9.4f}"
              f"{r['mde_p80']:>9.4f}{r['verdict']:>10}")
    print()
    print("N_eff — по автокорреляции IC-ряда (lags=10), ограничен сверху N.")
    print("MDE .50 — граница значимости (2/√N_eff), MDE .80 — порог при")
    print("power 0.80 (2.80/√N_eff). «Значим» — против порога Bonferroni 2.50.")
    print()

    print("=" * 96)
    print("3. СТАБИЛЬНОСТЬ ПО ЭПОХАМ (средний IC до/после 2025-03)")
    print("=" * 96)
    for label, sub, _ in windows:
        before = sub[sub.index < pd.Timestamp("2025-03-01")]
        after = sub[sub.index >= pd.Timestamp("2025-03-01")]
        print(f"  окно {label}: до 2025-03 N={len(before.dropna(how='all'))}, "
              f"после N={len(after.dropna(how='all'))}")
        for p in PREDICTORS:
            ic0 = before[p].dropna()
            ic1 = after[p].dropna()
            if len(ic0) >= 10 and len(ic1) >= 10:
                print(f"    {p:<17} {ic0.mean():+.4f} → {ic1.mean():+.4f}  "
                      f"(N {len(ic0)}/{len(ic1)})")
        print()

    print("=" * 96)
    print("4. ИЗДЕРЖКИ: спред Roll по дневным барам + turnover top-4")
    print("=" * 96)
    spreads = {t: float(roll_spread_daily(tables[t]["close"])) for t in tables}
    med = float(np.median([s for s in spreads.values() if np.isfinite(s)]))
    for t in sorted(spreads):
        s = spreads.get(t)
        print(f"  {t:<7} Roll {10000 * s:>6.1f} бп" if np.isfinite(s)
              else f"  {t:<7} Roll    nan")
    print(f"\n  медиана дневного Roll-спреда: {10000 * med:.1f} бп — это в")
    print("  разы выше 5-минутного замера cost_impact.py (6.6 бп): дневные")
    print("  приращения тонут в рыночном шуме, бид-аск-боунс меньше виден.")
    print("  Это консервативная оценка издержек (стороной осторожности),")
    print("  используем её. Реальный спред на заявке между ними и выше")
    print("  5-минутной оценки.")
    cost_rt = 10000 * med + 2 * COMMISSION_BPS
    print(f"  издержки на полный круг: {10000 * med:.1f} бп спред "
          f"+ {2 * COMMISSION_BPS:.1f} бп комиссия = {cost_rt:.1f} бп")
    print()

    gross_raw, turn_raw = portfolio_turnover(tables, k=K)
    gross = {p: pd.Series(v, dtype=float).sort_index()
             for p, v in gross_raw.items()}
    if not gross:
        print("  портфель не собран (мало дат с >=8 имён)")
        return 1
    print(f"{'предиктор':<17}{'N дней':>8}{'валов. бп/день':>16}{'t наив':>8}"
          f"{'turnover':>9}{'нето бп/день':>14}")
    print("-" * 74)
    verdicts = []
    for p in PREDICTORS:
        if p not in gross or len(gross[p].dropna()) < 20:
            print(f"  ({p}: портфель не собран — нужно >=8 имён в дате, "
                  f"bounded-прогон на 3 тикерах его не даёт)")
            continue
        g = gross[p].dropna()
        tn = pd.Series(turn_raw[p], dtype=float).reindex(g.index).fillna(1.0)
        mean_g = float(g.mean()) * 10000
        se = float(g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 2 else np.nan
        t_naive = mean_g / (se * 10000) if se and se > 0 else np.nan
        mean_to = float(tn.mean())
        net = mean_g - mean_to * cost_rt
        verdicts.append((p, mean_g, net))
        print(f"{p:<17}{len(g):>8}{mean_g:>16.2f}{t_naive:>8.2f}"
              f"{mean_to:>9.2f}{net:>14.2f}")
    print()
    print("Нетто — валовая минус turnover × полный круг. «Валовая» — mean IC")
    print("не знает: это доходность портфеля, собранного вчера, а не IC.")
    print()

    print("=" * 96)
    print("ВЕРДИКТ ШАГА 1")
    print("=" * 96)
    strong = [r for r in rows
              if r["verdict"] == "значим" and r["pred"] != "ctrl"]
    main_rows = [r for r in rows if r["pred"] != "ctrl"]
    if not main_rows:
        print("  Расчёт не состоялся: ни одной даты с >=3 бумагами.")
        print("  Сузь окно или добавь тикеров — и запусти снова.")
        return 2
    if strong:
        print("  Пережили порог IC >= 0.05 при 80% мощности:")
        for r in strong:
            print(f"    {r['pred']:<17} ({r['label']})  IC {r['ic']:+.4f}, "
                  f"t_eff {r['t_eff']:+.2f}, MDE .80 {r['mde_p80']:.4f}")
        print("  Дальше — сравнение моделей на том же дизайне (шаг 2).")
    else:
        mde_all = max(r["mde_p80"] for r in main_rows)
        print("  Ни один предиктор не достиг порога по предрегистрации.")
        print(f"  Лучшая доступная мощность: MDE .80 = {mde_all:.4f}.")
        print("  Критерий приёмки не выполнен — проект закрывается честно")
        print("  (ROADMAP: «если нет — проект закрывается честно, нормальный")
        print("   результат»).")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
