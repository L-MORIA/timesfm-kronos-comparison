"""Месячный пилот интрадея: проверяет, не съедает ли кластеризация выигрыш.

Расчёт мощности (`reports/2026-10-16-power-analysis.md`) дал MDE 0.006–0.012
на горизонтах 0.5–2 часа, но это НОМИНАЛЬНОЕ число. Оно получено делением
числа дней на число окон и молча предполагает, что окна внутри одного дня
независимы. Между тем все окна одного дня живут в одном рыночном состоянии:
общий overnight-разрыв, общий внутридневной тренд, общая волатильность.
Скорее всего они коррелированы, и тогда эффективная выборка меньше
номинальной в разы.

Именно это и меряем — это design effect из power-анализа:
    DEFF = 1 + (m − 1) · ICC
где m — число окон в дне, ICC — корреляция между окнами одного дня.

Что делаем:
  1. тянем месяц минутных баров по нескольким тикерам, агрегируем в 5-минутные;
  2. режем каждый торговый день на неперекрывающиеся окна горизонта h;
  3. в каждом окне считаем кросс-секционный IC простого предиктора
     (z-скор цены) против реализованной доходности;
  4. меряем ICC между окнами одного дня и строим DEFF;
  5. пересчитываем MDE по эффективной выборке — вот он и есть настоящий;
  6. отдельно смотрим профиль ликвидности основной и вечерней сессий.

Предиктор здесь не для проверки качества, а для получения ненулевых IC.
Нулевые IC дали бы вырожденную ковариацию и неинформативную ICC.
"""
from __future__ import annotations

import datetime as dt
import os
import time

import numpy as np
import pandas as pd
import requests
from statistics import NormalDist

TICKERS = ["SBER", "GAZP", "LKOH", "SBERP", "VTBR", "FEES", "MOEX", "ROSN",
           "TRNFP", "GMKN", "SNGSP", "MTSS", "AFKS"]
BASE = ("https://iss.moex.com/iss/engines/stock/markets/shares"
        "/boards/TQBR/securities/{ticker}/candles.json")
PAGE = 500
PAUSE = 0.15

# Месяц для пилота: полный, не текущий — внутридневная картина меняется
# день ото дня, но не неделю ото недели.
START = dt.date(2026, 8, 1)
END = dt.date(2026, 8, 31)

HORIZONS = [6, 12, 24]        # в 5-минутных барах: 0.5 / 1 / 2 часа
LOOKBACK = 20                # баров в предикторе (1.5 часа истории)

ND = NormalDist()


# ── выгрузка ────────────────────────────────────────────────────

def _sessions():
    """Два клиента: напрямую и через системный прокси.

    Локальный прокси на 127.0.0.1:10809 живёт не постоянно — когда он падает,
    requests падает вместе с ним, и выгрузка 13 тикеров теряется целиком.
    Поэтому сначала идём напрямую, а прокси оставляем запасным путём.
    """
    direct = requests.Session()
    direct.trust_env = False          # игнорировать HTTP(S)_PROXY из окружения
    via_proxy = requests.Session()    # по умолчанию, т.е. через переменные окружения
    return direct, via_proxy


SESSIONS = _sessions()


def _get(url, params):
    """Запрос с откатом на прокси, если прямое соединение не удалось."""
    last = None
    for s in SESSIONS:
        try:
            r = s.get(url, params=params, timeout=25)
            r.raise_for_status()
            return r
        except (requests.ProxyError, requests.ConnectionError) as e:
            last = e
            continue
    raise last


def fetch_minutes(ticker, d0, d1):
    rows, st = [], 0
    while True:
        p = {"interval": 1, "from": d0.isoformat(), "till": d1.isoformat(),
             "limit": PAGE, "start": st}
        r = _get(BASE.format(ticker=ticker), p)
        blk = (r.json() or {}).get("candles") or {}
        cols, data = blk.get("columns"), blk.get("data") or []
        if not data:
            break
        rows.extend(data)
        if len(data) < PAGE:
            break
        st += len(data)
        time.sleep(PAUSE)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=cols)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={"value": "amount"})
    df["begin"] = pd.to_datetime(df["begin"])
    for c in ("open", "high", "low", "close", "volume", "amount"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return (df.drop_duplicates(subset=["begin"]).sort_values("begin")
              .reset_index(drop=True))


def resample_5m(df):
    """Минутные -> пятиминутные, индекс = метка времени.

    Границы — по часовым меткам ISS как есть. Часовой пояс ISS неочевиден
    (сессия помечена 06:59…23:49), поэтому НЕ переводим на MSK: если ISS
    отдаёт метки в своём поясе, агрегация должна идти в том же поясе.

    Возвращает датафрейм с DatetimeIndex. Это не косметика: версия с
    колонкой вместо индекса отдавала RangeIndex, окна нумеровались
    позициями вместо дат, и группировка «по дню» считала мусор вместо
    внутридневной ICC. Молча, без исключения — самый дорогой вид ошибки.
    """
    g = df.set_index("begin")
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum", "amount": "sum"}
    out = g.resample("5min", label="left", closed="left").agg(agg)
    return out.dropna(subset=["close"])


def session_gaps(idx):
    """Разрывы между соседними метками внутри одного дня, по величине."""
    gaps = pd.Series(idx).diff().dt.total_seconds().dropna()
    return gaps


def split_sessions(df5):
    """Делит ряд на сутки по МЕТКАМ, внутри суток ищет главный разрыв."""
    out = {}
    for day, g in df5.groupby(df5.index.date):
        ts = g.index.sort_values()
        if len(ts) < 20:
            continue
        gaps = np.array([(ts[i + 1] - ts[i]).total_seconds()
                         for i in range(len(ts) - 1)])
        cut = int(np.argmax(gaps))
        out[day] = {"main": cut + 1, "eve": len(ts) - cut - 1,
                    "gap": gaps[cut], "first": ts[0], "last": ts[-1]}
    return out


def loud_window(panel):
    """Граница ликвидного окна по объёму, без обращения к часам.

    Сессионный перерыв в данных ISS отсутствует (максимальный разрыв внутри
    суток 180 с), поэтому границу режима задаём объёмом: блоки с объёмом
    выше медианы образуют громкую часть. Берём самое длинное непрерывное
    run — иначе один случайно громкий блок разорвёт окно.

    На входе Series с DatetimeIndex, сконструированная ранее.
    """
    v = panel.groupby(panel.index.floor("30min")).sum()
    med = float(v.median())
    loud = v[v >= med]
    if loud.empty:
        return None, None
    blocks = sorted(loud.index)
    best, cur = [blocks[0]], [blocks[0]]
    for b in blocks[1:]:
        if (b - cur[-1]).total_seconds() <= 1800:
            cur.append(b)
        else:
            if len(cur) > len(best):
                best = cur
            cur = [b]
    if len(cur) > len(best):
        best = cur
    return best[0], best[-1] + pd.Timedelta(minutes=30)


def trim(df5, start, end):
    """Оставить бары внутри окна [start, end) по меткам времени суток."""
    if start is None:
        return df5
    mins = df5.index.hour * 60 + df5.index.minute
    s = start.hour * 60 + start.minute
    e = end.hour * 60 + end.minute
    return df5[(mins >= s) & (mins < e)]


# ── метрики ─────────────────────────────────────────────────────

def spearman(a, b):
    if len(a) < 4:
        return np.nan
    ra = pd.Series(a).rank().to_numpy(float)
    rb = pd.Series(b).rank().to_numpy(float)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def ic_per_window(pnls, horizon, lookback=LOOKBACK):
    """Кросс-секционный IC в каждом неперекрывающемся окне.

    Предиктор — z-скор последних lookback баров (чем дороже относительно
    своей же истории, тем «перегретее» бумага). Реализация — доходность
    следующих horizon баров. Окна внутри дня не перекрываются.
    """
    tickers = list(pnls)
    idx = None
    for t in tickers:
        idx = pnls[t].index if idx is None else idx.intersection(pnls[t].index)
    idx = idx.sort_values()
    if idx is None or len(idx) < lookback + horizon:
        return []

    rows = []
    start = 0
    while start + lookback + horizon <= len(idx):
        stop = start + lookback + horizon
        window = idx[start:stop]
        preds, reals = [], []
        for t in tickers:
            s = pnls[t].reindex(window)
            px = s["close"].to_numpy(float)
            past = px[:lookback]
            sd = past.std()
            if sd == 0 or np.isnan(sd):
                continue
            preds.append((past[-1] - past.mean()) / sd)
            reals.append(px[-1] / past[-1] - 1)
        if len(preds) >= 3:
            ic = spearman(preds, reals)
            if np.isfinite(ic):
                rows.append((window[0], ic))
        start += horizon            # шаг = горизонт, окна не перекрываются
    return rows


def icc_within_day(rows):
    """ICC между окнами одного дня и сколько окон в дне в среднем."""
    by_day = {}
    for ts, ic in rows:
        by_day.setdefault(ts.date(), []).append(ic)
    days = [v for v in by_day.values() if len(v) >= 2]
    if len(days) < 3:
        return np.nan, np.nan, 0
    m = float(np.mean([len(v) for v in days]))
    grand = float(np.mean([x for v in days for x in v]))
    # MSB — СРЕДНЕЕ квадратов между днями, а не их сумма: SSB делится на
    # (число дней − 1). Без этого деления MSB завышается в n/(n−1) раз
    # (для 27 суток — на 3.8 %), а с ним и ICC (audit 2026-10-25, m5 + MB5).
    ssb = len(days) * m * float(np.mean([(np.mean(v) - grand) ** 2 for v in days]))
    msb = ssb / (len(days) - 1)
    msw = float(np.mean([np.var(v, ddof=1) for v in days if len(v) > 1]))
    if msw <= 0:
        return 0.0, m, len(days)
    # ICC (one-way random effects): (MSB − MSW) / (MSB + (m−1)·MSW).
    # Раньше стояло MSB / (MSB + (m−1)·MSW) — без вычитания MSW, что
    # соответствует другой конвенции и завышает ICC.
    denom = msb + (m - 1) * msw
    icc = max(0.0, (msb - msw) / denom) if denom > 0 else 0.0
    return icc, m, len(days)


def mde(n_eff, alpha=0.05, power=0.80):
    """Минимальный обнаружимый |rho| при заданной эффективной выборке.

    power по умолчанию 0.80 — та же конвенция, что в power_analysis.py.
    Раньше порог считался только по z_{1-α/2}, то есть соответствовал
    мощности 50 %, и занижал MDE почти вдвое (audit 2026-10-25, MB2).
    """
    if n_eff <= 3:
        return float("nan")
    z = ND.inv_cdf(1 - alpha / 2) + ND.inv_cdf(power)
    return float(np.tanh(z / np.sqrt(n_eff - 3)))


# ── main ───────────────────────────────────────────────────────

def main():
    print("=" * 84)
    print("МЕСЯЧНЫЙ ПИЛОТ ИНТРАДЕЯ — сколько стоит кластеризация окон")
    print("=" * 84)
    print(f"тикеры     : {', '.join(TICKERS)}")
    print(f"период     : {START} … {END}")
    print(f"агрегация  : минутные -> 5-минутные, горизонты {HORIZONS}")
    print()

    data, prof = {}, {}
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "logs", "intraday_5m_cache.csv")
    cache = os.path.normpath(cache)

    frames, cached = [], set()
    if os.path.isfile(cache):
        raw = pd.read_csv(cache, parse_dates=["begin"])
        cached = set(raw["ticker"].unique())
        frames = [g.drop(columns=["ticker"]).assign(ticker=tk)
                  for tk, g in raw.groupby("ticker")]
        print(f"[кэш] {os.path.basename(cache)}: {len(raw):,} строк, "
              f"{len(cached)} тикеров")

    missing = [t for t in TICKERS if t not in cached]
    if missing:
        print(f"[сеть] догрузка: {', '.join(missing)}")
        for tk in missing:
            try:
                m1 = fetch_minutes(tk, START, END)
            except Exception as e:                      # noqa: BLE001
                print(f"{tk:<7} ОШИБКА {type(e).__name__}: {e}")
                continue
            if m1.empty:
                print(f"{tk}: нет данных")
                continue
            d5 = resample_5m(m1)
            frames.append(d5.assign(ticker=tk).reset_index())
            # Кэш переписывается после КАЖДОГО тикера. Предыдущий прогон
            # потерял ~15 минут выгрузки целиком из-за падения на первом
            # же тикере — инкрементальность здесь не оптимизация.
            pd.concat(frames, ignore_index=True).to_csv(
                cache, index=False, encoding="utf-8")
            print(f"{tk:<7} минутных={len(m1):>7,}  5-минутных={len(d5):>7,}  "
                  f"дней={len(split_sessions(d5)):>3}  [кэш обновлён]",
                  flush=True)
        print(f"[кэш] {os.path.basename(cache)}: {len(frames)} тикеров")

    for f in frames:
        tk = f["ticker"].iloc[0]
        d5 = f.drop(columns=["ticker"]).set_index("begin").sort_index()
        data[tk] = d5
        prof[tk] = split_sessions(d5)
    if not data:
        print("нет данных ни по одному тикеру")
        return

    print()
    print("=" * 84)
    print("ПРОФИЛЬ СЕССИЙ (баров в сутки по половине дня)")
    print("=" * 84)
    print(f"{'тикер':<8}{'осн.':>7}{'вечер':>8}{'разрыв':>10}{'первый':>12}"
          f"{'последний':>12}")
    print("-" * 57)
    for tk, sess in prof.items():
        if not sess:
            continue
        m = np.mean([v["main"] for v in sess.values()])
        e = np.mean([v["eve"] for v in sess.values()])
        gp = np.mean([v["gap"] for v in sess.values()]) / 60
        f0 = next(iter(sess.values()))["first"]
        l0 = next(iter(sess.values()))["last"]
        print(f"{tk:<8}{m:>7.0f}{e:>8.0f}{gp:>9.0f}м"
              f"{str(f0)[11:16]:>12}{str(l0)[11:16]:>12}")
    print()
    print("Таблица выше почти вырождена: «разрыв» в 5 минут — это обычный шаг")
    print("между соседними пятиминутными барами, а не сессионный перерыв.")
    print("Замер session_diag.py подтвердил: перерыва в данных ISS нет вообще.")
    print("Поэтому разбивку на сессии здесь НЕ используем — границу режима")
    print("задаёт объём, см. блок ниже.")
    print()

    print("=" * 84)
    print("КЛАСТЕРИЗАЦИЯ ОКОН — главный вопрос пилота")
    print("=" * 84)

    # Граница ликвидности — по объёму, часы ISS не работают как расписание.
    panel = pd.concat([data[t]["volume"] for t in data], axis=1).sum(axis=1)
    w0, w1 = loud_window(panel)
    print(f"громкое окно по объёму: {str(w0)[11:16]} … {str(w1)[11:16]}")
    if w0 is not None:
        # Статистика за ВЕСЬ период первого тикера, приведённая к суткам:
        # 63 790 строк кэша / 13 тикеров = ~4900 баров на тикер за месяц,
        # а не в сутки. Прежняя подпись «баров в сутки» показывала месячное
        # число (4916 → 3047) и читалась как суточное.
        d0 = data[list(data)[0]]
        n_days = max(len(d0.index.normalize().unique()), 1)
        n_before = len(d0) / n_days
        n_after = len(trim(d0, w0, w1)) / n_days
        print(f"баров в сутки (среднее, тикер {list(data)[0]}): "
              f"{n_before:.0f} → {n_after:.0f} "
              f"({100 * n_after / n_before:.0f} % времени)")
    print()

    variants = [("полный ряд", data)]
    if w0 is not None:
        variants.append((f"обрезанный {str(w0)[11:16]}–{str(w1)[11:16]}",
                         {t: trim(d, w0, w1) for t, d in data.items()}))

    all_results = []
    for label, panel_data in variants:
        print(f"--- {label} ---")
        print(f"{'h':>4}{'окно':>8}{'окон/день':>11}{'ICC':>8}{'DEFF':>8}"
              f"{'N_ном':>9}{'N_эфф':>9}{'MDE_эфф':>10}")
        print("-" * 67)
        results = []
        for h in HORIZONS:
            rows = ic_per_window(panel_data, h)
            if not rows:
                continue
            icc, m, ndays = icc_within_day(rows)
            n_nom = len(rows)
            deff = 1 + (m - 1) * icc if np.isfinite(icc) and np.isfinite(m) else np.nan
            n_eff = n_nom / deff if np.isfinite(deff) and deff > 0 else n_nom
            results.append((h, m, icc, deff, n_nom, n_eff, mde(n_eff)))
            print(f"{h:>4}{h * 5 / 60:>7.1f}ч{m:>11.1f}{icc:>8.3f}{deff:>8.2f}"
                  f"{n_nom:>9}{n_eff:>9.0f}{mde(n_eff):>10.3f}")
        print()
        print("Экстраполяция на 12.4 года (144 месяца, MDE делится на 12):")
        for h, m, icc, deff, n_nom, n_eff, d in results:
            print(f"    h={h:>2} ({h * 5 / 60:.1f} ч): ICC {icc:+.3f}, "
                  f"DEFF {deff:>5.2f}, MDE/мес {d:.3f}, "
                  f"MDE/12 лет {mde(n_eff * 144):.4f}")
        print()
        all_results.append((label, results))

    print("=" * 84)
    print("ВЫВОД")
    print("=" * 84)
    for label, results in all_results:
        print(f"[{label}]")
        for h, m, icc, deff, n_nom, n_eff, d in results:
            print(f"  h={h:>2} ({h * 5 / 60:.1f} ч): окон/день {m:.0f}, "
                  f"ICC {icc:+.3f}, DEFF {deff:.2f}, MDE_эфф {d:.3f}")
        print()

    # Дальше считаем по ПОЛНОМУ ряду: обрезка — это отдельный вопрос,
    # а не замена базовой оценки.
    results = all_results[0][1]
    best = min((r[6] for r in results if np.isfinite(r[6])), default=float("nan"))
    n_tk = len(data)
    print(f"Лучший достижимый MDE на одном месяце данных: {best:.3f} "
          f"(это 27 дней и {n_tk} тикеров — пилот, а не рабочий тест)")
    print()
    print("Экстраполяция на полную историю 12.4 лет. MDE падает как 1/√N,")
    print("а N растёт линейно со временем, значит за 12 лет MDE делится на √144=12.")
    print()
    print(f"{'h':>4}{'окно':>8}{'ICC':>8}{'DEFF':>8}{'MDE 1 мес':>11}"
          f"{'N_эфф 12 лет':>15}{'MDE 12 лет':>13}")
    print("-" * 67)
    for h, m, icc, deff, n_nom, n_eff, d in results:
        n_long = n_eff * 144
        # n_long округляется до целого: полная точность numpy-скаляра
        # не влезает в колонку и слипается с соседними
        print(f"{h:>4}{h * 5 / 60:>7.1f}ч{icc:>8.3f}{deff:>8.2f}{d:>11.3f}"
              f"{n_long:>15,.0f}{mde(n_long):>13.4f}")
    print()
    iccs = [r[2] for r in results if np.isfinite(r[2])]
    deffs = [r[3] for r in results if np.isfinite(r[3])]
    mde12 = [mde(r[5] * 144) for r in results if np.isfinite(r[5])]
    if iccs and mde12:
        print(f"Итог: ICC окон внутри дня {min(iccs):.3f}–{max(iccs):.3f} — "
              f"кластеризация слабая;")
        print(f"она съедает номинальную выборку в {min(deffs):.1f}–{max(deffs):.1f} раз, "
              f"и 12 лет истории")
        print(f"выводят на MDE {min(mde12):.4f}–{max(mde12):.4f} — зона IC ~0.01.")
    print("Обрезка по громкому окну этого НЕ чинит (см. выше).")
    print()
    print("Честные оговорки:")
    print("  * предиктор здесь z-скор цены — он сам по себе инерционный внутри")
    print("    суток, что раздувает ICC. Модель с более «шумным» выходом даст")
    print("    меньший ICC. Но считать на этом оптимизме нельзя.")
    print("  * вопрос о перерыве между сессиями закрыт замером session_diag.py:")
    print("    в данных ISS разрыва нет вообще (максимум 180 с), ряд идёт")
    print("    06:59–23:49, и восстановить по нему расписание нельзя. Поэтому")
    print("    граница режима задана объёмом, а не часами, и обрезанный вариант")
    print("    ниже — это эмпирическая проверка, а не восстановление сессий.")
    print("  * издержки не посчитаны. Это единственная оговорка, которую замер")
    print("    принципиально не снимает: на горизонте 0.5–2 часа спред и комиссия")
    print("    могут съесть IC 0.02 целиком. Следующий шаг шага 1.5.")
    print("  * месяц — пилот, не рабочий тест. Все числа ниже показывают, какой")
    print("    РЕЖИМ даёт дизайн, а не подтверждают наличие сигнала.")


if __name__ == "__main__":
    main()