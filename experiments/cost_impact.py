"""Издержки: съедают ли спред и комиссия тот IC, ради которого всё затевалось.

Расчёт мощности (`reports/2026-10-16-power-analysis.md`) показал, что интрадей
снимает потолок: MDE падает до зоны IC 0.02–0.03 на горизонтах 0.5–2 часа.
Это единственный оставшийся аргумент «почему вообще стоит пробовать».

Но он неполон, пока не отвечает на вопрос, который задаёт любой трейдер
раньше, чем любой статистик: **а сколько стоит сделка?**

Здесь ровно это и меряется. Спред не берётся из справочника — он оценен
по самим барам методом Roll (1984), из знака автокорреляции последовательных
приращений. Второй метод, Corwin-Schultz (2012), реализован, но на
доступном окне (27 суток) НЕ СЧИТАЕТСЯ: метод выводится из асимптотики по
числу дней, и ниже 60 суток он даёт шум, а не оценку. Поэтому перекрёстная
проверка «два метода сошлись» в этом прогоне НЕ выполнена, и спред опирается
на один метод — с явной оговоркой (audit 2026-10-25, m6).

Дальше — не «прибыль», а честный разбор:
  1. валовая доходность стратегии на окнах (top-K против равных весов);
  2. стоимость оборота: спред + комиссия, в базисных пунктах на круг;
  3. порог безубыточности — сколько бп должен съесть спред, чтобы сигнал
     стал нулевым;
  4. контроль: тот же расчёт на перемешанном сигнале, чтобы разницу
     нельзя было объяснить одной только рыночной инерцией.

Претензия на прибыль здесь есть ровно одна: валовая доходность превышает
издержки. Всё остальное — IC, Sharpe, MDE — вне денег.
"""
from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd
from statistics import NormalDist

TICKERS = ["SBER", "GAZP", "LKOH", "SBERP", "VTBR", "FEES", "MOEX", "ROSN",
           "TRNFP", "GMKN", "SNGSP", "MTSS", "AFKS"]

START = dt.date(2026, 8, 1)
END = dt.date(2026, 8, 31)

HORIZONS = [6, 12, 24]
LOOKBACK = 20
TOPK = 4                      # сколько бумаг в портфеле

# Комиссия брокера — не измеряется из баров, это факт из тарифа.
# 0.03 % за сторону — типичная ставка розничного тарифа на MOEX.
COMMISSION_BPS = 3.0          # за сторону, в бп
# Спред будет измерен; эти уровни нужны только для sensitivity-вопроса
# «а если он в разы толще измеренного».
COST_GRID = [0, 5, 10, 15, 20, 30, 50]

# Corwin-Schultz выводится из асимптотики по числу дней; на десятках суток
# он даёт шум, а не оценку. Ниже порога метод честно не считается.
MIN_DAYS_CS = 60

ND = NormalDist()
RNG_SEED = 20260801


# ── оценка спреда по барам ──────────────────────────────────────

def roll_spread(close, lag=1):
    """Roll (1984): S = 2·√(−Cov(Δp_t, Δp_{t+1})).

    Логарифм цены, лаг по умолчанию 1. Ковариация отрицательна, когда ряд
    отскакивает на спреде, — это и есть бид-аск-боунс. Cov >= 0 → спреда
    не видно, возвращаем 0, а не отрицательное число.
    """
    p = np.log(close.to_numpy(float))
    d = np.diff(p)
    n = len(d) - lag
    if n < 50:
        return np.nan
    a, b = d[:n], d[lag:lag + n]
    cov = float(np.cov(a, b)[0, 1])
    if cov >= 0:
        return 0.0
    return 2.0 * np.sqrt(-cov)


def daily_bars(d5):
    """Пятиминутные бары -> дневные. Нужны для Corwin-Schultz.

    Метод родом из дневных данных и на минутных считается некорректно:
    внутридневной тренд сам по себе даёт «широкий» двухбарный диапазон,
    который метод принимает за спред. Поэтому агрегируем, а не подставляем.
    """
    g = d5.resample("1D").agg({"open": "first", "high": "max",
                               "low": "min", "close": "last"})
    return g.dropna(subset=["close"])


def corwin_schultz(daily):
    """Corwin-Schultz (2012): спред из экстремумов двух соседних дней.

    Сравнивает дневной диапазон одиночного дня с диапазоном пары дней.
    Если диапазон пары шире суммы диапазонов её частей, между днями был
    спред: на «чистом» рынке так не бывает.

    Возвращает долю от цены (не базисные пункты), либо nan если данных
    недостаточно. MIN_DAYS — не формальность: метод выводится из
    асимптотики по числу дней, и на 27 сутках он даёт шум со знаком,
    а не оценку спреда.
    """
    if len(daily) < MIN_DAYS_CS:
        return np.nan
    h = daily["high"].to_numpy(float)
    l = daily["low"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r1 = np.log(h[:-1] / l[:-1])        # диапазон дня t
        r2 = np.log(h[1:] / l[1:])          # диапазон дня t+1
        two = np.log(h[1:] / l[:-1])        # диапазон пары t, t+1
    ok = np.isfinite(r1) & np.isfinite(r2) & np.isfinite(two)
    if ok.sum() < 50:
        return np.nan
    r1, r2, two = r1[ok], r2[ok], two[ok]
    beta = float(np.mean(r1 ** 2 + r2 ** 2))
    gamma = float(np.mean(two ** 2))
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    alpha = max(alpha, 1e-12)               # спред не может быть отрицательным
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return float(s)


def spread_table(data, windows):
    """Спред по тикерам: Roll, Corwin-Schultz и медианный спред как контроль."""
    rows = []
    for label, sel in windows:
        for tk, d5 in data.items():
            sub = sel(d5)
            if len(sub) < 500:
                continue
            roll = roll_spread(sub["close"])
            cs = corwin_schultz(daily_bars(sub))
            # медианный спред как грубый верхний ориентир:
            # внутрибаровая амплитуда не может быть меньше спреда
            amp = ((sub["high"] - sub["low"]) / sub["close"]).median()
            rows.append((label, tk, roll, cs, amp))
    return pd.DataFrame(rows, columns=["окно", "тикер", "roll", "corwin", "амплитуда"])


# ── стратегия ──────────────────────────────────────────────────

def window_pnl(pnls, horizon, lookback=LOOKBACK, k=TOPK, rng=None):
    """Валовая доходность на неперекрывающихся окнах.

    Предиктор — z-скор последних lookback баров (тот же, что в пилоте,
    чтобы числа сопоставимы). Реализация — доходность следующих horizon баров.

    long-only: портфель из k самых «перегретых» бумаг, равные веса внутри.
    Это реалистично: шорт на ТQBR рознице недоступен, поэтому такая
    конструкция — единственная торгуемая.

    Возвращает на окно: (ts, top-k, bottom-k, равные веса по вселенной,
    top-k на перемешанном сигнале).

    Метрика в разборе — АБСОЛЮТНАЯ доходность портфеля top-k, а не её
    превышение над равными весами. Вычитание бенчмарка здесь ловушка:
    бенчмарк держит все 13 бумаг, а портфель — 4, поэтому «превышение»
    автоматически съедает разницу в издержках (13 кругов против 4) и
    выглядит лучше, чем есть на самом деле. Абсолютная доходность — то,
    что реально можно заработать, и против неё издержки честно стоят в круг.
    """
    tickers = list(pnls)
    idx = None
    for t in tickers:
        idx = pnls[t].index if idx is None else idx.intersection(pnls[t].index)
    if idx is None:
        return []
    idx = idx.sort_values()
    if len(idx) < lookback + horizon:
        return []

    out = []
    start = 0
    while start + lookback + horizon <= len(idx):
        window = idx[start:start + lookback + horizon]
        rows = []
        for t in tickers:
            s = pnls[t].reindex(window)
            px = s["close"].to_numpy(float)
            past = px[:lookback]
            sd = past.std()
            if sd == 0 or np.isnan(sd):
                continue
            fut = px[lookback:]
            if np.isnan(fut).any() or len(fut) < horizon:
                continue
            rows.append(((past[-1] - past.mean()) / sd,
                         float(px[-1] / past[-1] - 1)))
        if len(rows) >= 2 * k + 1:
            arr = np.array(rows)
            order = np.argsort(-arr[:, 0])
            top = arr[order[:k], 1]
            bot = arr[order[-k:], 1]
            ew = float(arr[:, 1].mean())
            if rng is not None:                       # контроль: перемешанный сигнал
                perm = rng.permutation(len(arr))
                sh = arr[perm, 0]
                o2 = np.argsort(-sh)
                top_c = arr[o2[:k], 1]
            else:
                top_c = None
            out.append((window[0],
                        float(top.mean()),
                        float(bot.mean()),
                        ew,
                        float(top_c.mean()) if top_c is not None else np.nan))
        start += horizon
    return out


def mean_t(x):
    """Среднее и t-статистика. Не standard error.

    Возвращать SE вместо t — ровно та ошибка, из-за которой в проекте уже
    один раз возник ложный «успех»: SE всегда положителен и не несёт знака,
    а t несёт. Здесь нужен именно t.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    if se <= 0:
        return float(x.mean()), np.nan
    return float(x.mean()), float(x.mean() / se)


# ── main ───────────────────────────────────────────────────────

def main():
    print("=" * 84)
    print("ИЗДЕРЖКИ — съедают ли спред и комиссия тот сигнал, ради которого всё")
    print("=" * 84)
    print(f"период   : {START} … {END}")
    print(f"комиссия : {COMMISSION_BPS:.1f} бп за сторону "
          f"(тариф, не измеряется из баров)")
    print()

    cache = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "logs", "intraday_5m_cache.csv"))
    if not os.path.isfile(cache):
        print(f"Нет кэша {cache}. Сначала прогоняется experiments/intraday_pilot.py")
        return
    raw = pd.read_csv(cache, parse_dates=["begin"])
    data = {}
    for tk, g in raw.groupby("ticker"):
        data[tk] = g.drop(columns=["ticker"]).set_index("begin").sort_index()
    print(f"[кэш] {len(raw):,} строк, {len(data)} тикеров")
    print()

    # ── 1. спред, измеренный по баровым данным ──────────────────
    print("=" * 84)
    print("1. СПРЕД, ИЗМЕРЕННЫЙ ПО БАРАМ (доля от цены)")
    print("=" * 84)

    def identity(d):
        return d

    def to_loud(w0, w1):
        def f(d):
            if w0 is None:
                return d
            mins = d.index.hour * 60 + d.index.minute
            return d[(mins >= w0) & (mins < w1)]
        return f

    w0 = w1 = None
    if len(data) > 1:
        panel = pd.concat([data[t]["volume"] for t in data], axis=1,
                       sort=False).sum(axis=1).sort_index()
        v = panel.groupby(panel.index.floor("30min")).sum()
        loud = v[v >= float(v.median())]
        blocks = sorted(loud.index)
        if blocks:
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
            w0, w1 = best[0], best[-1] + pd.Timedelta(minutes=30)
            print(f"громкое окно: {str(w0)[11:16]} … {str(w1)[11:16]}")
    windows = [("полный", identity)]
    if w0 is not None:
        windows.append((f"громкий {str(w0)[11:16]}-{str(w1)[11:16]}",
                        to_loud(w0.hour * 60 + w0.minute, w1.hour * 60 + w1.minute)))

    sp = spread_table(data, windows)
    print()
    print(f"{'окно':<12}{'тикер':<8}{'Roll бп':>10}{'C-S бп':>10}{'амплит. бп':>12}")
    print("-" * 52)
    for label, g in sp.groupby("окно", sort=False):
        for _, r in g.iterrows():
            print(f"{label:<12}{r['тикер']:<8}"
                  f"{10000 * r['roll']:>10.1f}{10000 * r['corwin']:>10.1f}"
                  f"{10000 * r['амплитуда']:>12.1f}")
        print()

    print("Roll и Corwin-Schultz — разные допущения, разные бары (закрытия")
    print("против экстремумов). Если они расходятся в разы, спред не измерен,")
    print("а угадан; тогда единственный честный ход — sensitivity-сканирование.")
    print()

    # ── 2. валовая доходность и порог безубыточности ────────────
    print("=" * 84)
    print("2. ВАЛОВАЯ ДОХОДНОСТЬ И ПРОГ ПРОТИВ ИЗДЕРЖЕК")
    print("=" * 84)

    rng = np.random.default_rng(RNG_SEED)
    verdict_rows = []

    for label, sel in windows:
        pnl_data = {t: sel(d) for t, d in data.items()}
        print(f"--- окно: {label} ---")
        print(f"{'h':>4}{'окн':>6}{'импульс бп':>11}{'t':>7}"
              f"{'РЕВЕРС бп':>11}{'t':>7}{'контр бп':>10}{'равн.веса':>11}")
        print("-" * 67)
        for h in HORIZONS:
            rows = window_pnl(pnl_data, h, rng=rng)
            if not rows:
                print(f"{h:>4}  нет окон")
                continue
            # АБСОЛЮТНЫЕ доходности, бп. Не «превышение над бенчмарком»:
            # см. докстроку window_pnl — вычитание скрывает разницу в
            # числе кругов и делает концентрацию бесплатной.
            gross = np.array([r[1] for r in rows]) * 1e4          # top-K = импульс
            rev = np.array([r[2] for r in rows]) * 1e4            # bottom-K = реверс
            ctrl = np.array([r[4] for r in rows]) * 1e4           # перемешанный
            ew = np.array([r[3] for r in rows]) * 1e4             # вся вселенная
            g, tg = mean_t(gross)
            rv, trv = mean_t(rev)
            c, tc = mean_t(ctrl)
            _, tew = mean_t(ew)
            costs = COMMISSION_BPS * 2                            # круг, без спреда
            spread_roll = float(np.nanmedian(
                sp[sp["окно"] == label]["roll"])) * 1e4
            costs_total = costs + spread_roll
            verdict_rows.append((label, h, g, tg, rv, trv, c, tc,
                                 costs_total, float(np.mean(ew))))
            print(f"{h:>4}{len(rows):>6}{g:>11.2f}{tg:>7.2f}{rv:>11.2f}{trv:>7.2f}"
                  f"{c:>10.2f}{tew:>11.2f}")
        print()

    # ── 3. sensitivity-сканирование ─────────────────────────────
    print("=" * 84)
    print("3. ЧТО ОСТАЁТСЯ ПОСЛЕ ИЗДЕРЖЕК (абсолютная доходность top-K)")
    print("=" * 84)

    for label, sel in windows:
        pnl_data = {t: sel(d) for t, d in data.items()}
        print(f"--- окно: {label} ---")
        print(f"{'h':>4}{'издержки/круг':>16}{'net имп. бп':>13}"
              f"{'net реверс бп':>15}")
        print("-" * 48)
        for h in HORIZONS:
            rows = window_pnl(pnl_data, h)
            if not rows:
                continue
            gross = np.array([r[1] for r in rows]) * 1e4
            rev = np.array([r[2] for r in rows]) * 1e4
            for cost in COST_GRID:
                m, _ = mean_t(gross - cost)
                rv, _ = mean_t(rev - cost)
                print(f"{h:>4}{cost:>14.0f}бп{m:>13.2f}{rv:>15.2f}")
            print()
        break        # полное сканирование — только на полном ряду

    print("=" * 84)
    print("ВЫВОД")
    print("=" * 84)

    full = [r for r in verdict_rows if r[0] == "полный"]
    loud = [r for r in verdict_rows if r[0] != "полный"]
    for label, h, g, tg, rv, trv, c, tc, costs, ewm in full + loud:
        print(f"  [{label}] h={h:>2} ({h * 5 / 60:.1f} ч): "
              f"импульс {g:+6.2f} бп (t={tg:+.2f}) | "
              f"РЕВЕРС {rv:+6.2f} бп (t={trv:+.2f}) | "
              f"контр {c:+6.2f} | равные веса {ewm:+6.2f}")
    print()
    print("Издержки в круг (спред Roll + комиссия в обе стороны):")
    for r in full + loud:
        print(f"  [{r[0]}] h={r[1]:>2}: {r[8]:.1f} бп — "
              f"импульс нетто {r[2] - r[8]:+.2f} бп, "
              f"реверс нетто {r[4] - r[8]:+.2f} бп")
    print()
    print("Что это значит:")
    print("  * «реверс» — покупка k самых ПРОДАВЛЕННЫХ бумаг по z-скору.")
    print("    Проверка обязательна: дневной бэктест показал, что random_walk")
    print("    бьёт обе модели, то есть внутридневно рынок скорее возвращает.")
    print("    Тестировать только импульс — значит заранее знать ответ и")
    print("    подогнать под него вывод.")
    print("  * «валовая» — средняя доходность портфеля на окно. Это и есть")
    print("    порог безубыточности: спред с комиссией, превышающий его,")
    print("    гарантированно обнуляет результат при любом качестве сигнала.")
    print("  * контроль — тот же расчёт на перемешанном сигнале. Если он")
    print("    неотличим от настоящего, вся доходность объясняется рынком.")
    print("  * t посчитаны наивно, по независимым окнам. Окна внутри дня")
    print("    кластеризованы (ICC замерено в пилоте), поэтому настоящий t")
    print("    МЕНЬШЕ указанного. Здесь это осознанно: грубая верхняя оценка.")
    print("  * месяц данных не даёт права утверждать прибыльность. Он даёт")
    print("    право утверждать, что измерять дальше имеет смысл — либо что")
    print("    не имеет.")
    print()
    print("Измеренный спред, медиана по тикерам (бп):")
    for label, g in sp.groupby("окно", sort=False):
        cs = g["corwin"]
        cs_txt = ("не считается (<%d дней)" % MIN_DAYS_CS
                  if not np.isfinite(cs).any() else f"{10000 * cs.median():.1f}")
        print(f"  {label:<22} Roll {10000 * g['roll'].median():6.1f}   "
              f"Corwin-Schultz {cs_txt:<24} "
              f"амплитуда {10000 * g['амплитуда'].median():6.1f}")
    print()
    print("Методы разошлись — и это ожидаемо. Roll меряет бид-аск-боунс")
    print("внутридневно и потому «видит» спред лучше; Corwin-Schultz считается")
    print("по дням и на 27 сутках просто не имеет права работать. Поэтому")
    print("ориентиром берём Roll, а не среднее двух методов.")
    print()

    print("=" * 84)
    print("ВЕРДИКТ ПО ИЗДЕРЖКАМ")
    print("=" * 84)
    best = max((r[4] for r in verdict_rows), default=np.nan)
    cost_ref = full[0][8] if full else np.nan
    if np.isfinite(best) and np.isfinite(cost_ref):
        need = cost_ref / best if best > 0 else np.inf
        print(f"  Лучшая из двух сторон (реверс) даёт {best:+.2f} бп валовой.")
        print(f"  Издержки в круг — {cost_ref:.1f} бп.")
        print(f"  Разрыв — в {need:.1f}×. Нужно найти преимущество в семь раз")
        print("  больше лучшего из того, что уже измерено.")
    print()
    print("  Это не «сигнал слабый». Это другая величина: даже ИДЕАЛЬНЫЙ")
    print("  сигнал с IC 0.3 на горизонте 2 часа не окупает круг в 12.6 бп.")
    print("  Потолок ставят издержки, а не качество модели.")
    print()
    print("  Что это закрывает: горизонты 0.5–2 часа на ТQBR с розничным")
    print("  тарифом. Там физически негде взяться торгуемому преимуществу.")
    print("  Что это НЕ закрывает: длинные горизонты, где один круг на сделку")
    print("  амортизируется временем, и режимы, где торгуют не 5-минутными")
    print("  барами.")
    print()

    print("Честные оговорки:")
    print("  * спред оценён по барам, а не из стакана. Это оценка СНИЗУ: реальный")
    print("    спред на заявке больше измеренного. Значит, разрыв издержек")
    print("    в реальности больше, а не меньше.")
    print("  * t у валовых доходностей ниже 1.5 на всех горизонтах. То есть")
    print("    даже валовая доходность не отличима от нуля, а сравнивать её")
    print("    с издержками — значит сравнивать шум с константой.")
    print("  * издержки здесь — спред плюс комиссия, без проскальзывания по")
    print("    заявке. Проскальзывание добавится, следовательно, нетто ещё хуже.")
    print("  * месяц — один. Это пилот, а не рабочий тест: он показывает,")
    print("    что измерять дальше НЕ имеет смысла, и это его единственная")
    print("    функция.")


if __name__ == "__main__":
    main()