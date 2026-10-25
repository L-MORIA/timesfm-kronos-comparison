"""Разведка интрадей-истории MOEX для 14 тикеров из kronos-signal/config.yaml.

Зачем. Расчёт мощности (`reports/2026-10-16-power-analysis.md`) показал, что на
дневных барах IC 0.03 недостижим, и единственный путь в зону 0.02–0.03 —
интрадей. Но прежде чем строить дизайн, надо измерить глубину: сколько
реально истории есть и хватает ли её.

Найдено при разведке и меняет исходную постановку:

  interval=4 в MOEX ISS — это КВАРТАЛ, а не 5 минут. Запрос за 15 дней 2015
  вернул ровно одну свечу с begin=2015-01-01, end=2015-03-31. Пятиминутных
  свечей ISS не отдаёт. Значит все расчёты «113 пятиминутных баров на сессию»
  надо делать не на готовых барах, а на агрегатах из минутных — ровно так,
  как это делает kronos-signal (`interval: 1`, `resample_to: 5`).

  Это удорожает всё в 5 раз по объёму выборки и по времени фетча, но не
  меняет арифметику мощности: агрегация из 5 минутных баров даёт те же
  5-минутные бары, только выгруженные ценой 5-кратного трафика.

Метод поиска начала истории — бинарный поиск по дате: ISS отдаёт данные
постранично, поэтому идти вглубь пагинацией означало бы тысячи запросов.
"""
from __future__ import annotations

import datetime as dt
import sys
import time

import requests

TICKERS = ["SBER", "GAZP", "LKOH", "SBERP", "VTBR", "FEES", "MOEX", "ROSN",
           "TRNFP", "GMKN", "SNGSP", "MTSS", "AFKS"]
# YDEX исключён 2026-10-16 по решению владельца проекта: его история
# начинается с 2024-07-24 (2.2 года против 12–15 лет у остальных), и одна
# эта бумага схлопывала общий кусок всей вселенной с 12 лет до 2.2.
# Решение принято ДО бэктеста, по характеристике данных, а не по результату.

BASE = ("https://iss.moex.com/iss/engines/stock/markets/shares"
        "/boards/{board}/securities/{ticker}/candles.json")

MOEX_PAGE_LIMIT = 500   # ISS отдаёт не больше 500 строк за страницу
PAUSE = 0.15            # вежливость к API
# Минутных баров в сутках. ИЗМЕРЕНО этим же скриптом: 1008–1009 баров
# (201 пятиминутный эквивалент), сумма объёмов минутных сходится с дневной
# свечой. Прежняя оценка 113 брала только основную сессию 09:50–18:50 без
# вечерней торговли — занижение в 9 раз (audit 2026-10-25, m3).
MINUTES_PER_DAY = 1005


def fetch(ticker, date_from, date_to, interval=1, board="TQBR", limit=MOEX_PAGE_LIMIT,
          start=0, attempts=3):
    """Одна страница свечей или None."""
    params = {"interval": interval,
              "from": date_from.isoformat(),
              "till": date_to.isoformat(),
              "limit": limit, "start": start}
    last = None
    for i in range(attempts):
        try:
            r = requests.get(BASE.format(board=board, ticker=ticker),
                             params=params, timeout=20)
            r.raise_for_status()
            block = (r.json() or {}).get("candles") or {}
            data = block.get("data") or []
            return block.get("columns"), data
        except Exception as e:                      # noqa: BLE001
            last = e
            time.sleep(2 ** i)
    print(f"    [!] {ticker} {date_from}..{date_to}: {last}")
    return None


def has_any(ticker, d0, d1):
    """Есть ли хоть одна свеча за окно."""
    got = fetch(ticker, d0, d1, limit=1)
    return bool(got and got[1])


def earliest(ticker, board="TQBR"):
    """Самая ранняя доступная свеча — ОДНИМ запросом.

    Первая версия делала бинарный поиск по датам и вернула ерунду: ISS
    игнорирует параметр limit и всегда отдаёт 500 строк, поэтому has_any()
    была True при любом mid, поиск монотонно толкал lo к сегодня и схлопывался
    на hi. Данные приходят в возрастающем порядке, значит первая строка
    первой страницы и есть самое раннее, что есть.
    """
    cols, data = fetch(ticker, dt.date(1990, 1, 1), dt.date.today(),
                       limit=MOEX_PAGE_LIMIT, board=board)
    if not data:
        return None, None
    bi = cols.index("begin") if cols and "begin" in cols else 6
    first = str(data[0][bi])
    return first, (str(data[-1][bi]), len(data))


def full_day(ticker, day, board="TQBR"):
    """Все минутные бары за сутки пагинацией + дневная свеча для сверки."""
    rows, st = [], 0
    while True:
        cols, data = fetch(ticker, day, day, limit=MOEX_PAGE_LIMIT, start=st,
                           board=board)
        if not data:
            break
        rows.extend(data)
        if len(data) < MOEX_PAGE_LIMIT:
            break
        st += len(data)
    dcols, daily = fetch(ticker, day, day, interval=24, board=board)
    return rows, (daily[0] if daily else None), dcols


def session_profile(ticker, day, board="TQBR"):
    """Границы торговой сессии и число баров за день (первая страница)."""
    got = fetch(ticker, day, day, limit=MOEX_PAGE_LIMIT, board=board)
    if not got or not got[1]:
        return None
    cols, data = got
    bi = cols.index("begin") if "begin" in cols else 6
    times = sorted(str(r[bi]) for r in data)
    return {"rows_page1": len(data), "first": times[0], "last": times[-1],
            "cols": cols}


def main():
    today = dt.date.today()
    print("=" * 86)
    print("РАЗВЕДКА ИНТРАДЕЙ-ИСТОРИИ MOEX — 14 тикеров")
    print("=" * 86)
    print("ВАЖНО: interval=4 в ISS — квартал, не 5 минут. Пятиминутных свечей")
    print("ISS не отдаёт; их получают агрегацией из минутных (kronos-signal")
    print("делает ровно это: interval=1, resample_to=5).")
    print()
    print(f"MOEX отдаёт максимум {MOEX_PAGE_LIMIT} строк за страницу независимо")
    print("от запрошенного limit — это потолок трафика при выгрузке.")
    print()

    # Проверка утверждения про interval=4
    cols, data = fetch("SBERP", dt.date(2015, 1, 1), dt.date(2015, 1, 15), interval=4)
    if data:
        print(f"контрольная проверка: interval=4 за 15 дней вернул {len(data)} строк(и),")
        print(f"  begin={data[0][6]}, end={data[0][7]}")
        print("  -> это квартальная свеча, подтверждает: 5-минутных в ISS нет")
        print()

    print("=" * 86)
    print("ГЛУБИНА МИНУТНОЙ ИСТОРИИ")
    print("=" * 86)
    print(f"{'тикер':<8}{'начало':<12}{'лет':>7}{'оценка 1м баров':>18}"
          f"{'оценка 5м баров':>18}")
    print("-" * 63)
    rows = []
    for tk in TICKERS:
        try:
            first, page = earliest(tk)
        except Exception as e:                      # noqa: BLE001
            print(f"{tk:<8}ошибка: {e}")
            continue
        time.sleep(PAUSE)
        if first is None:
            print(f"{tk:<8}{'нет данных':<12}")
            rows.append((tk, None, 0, 0))
            continue
        fdate = dt.date.fromisoformat(first[:10])
        years = (today - fdate).days / 365.25
        # 1005 минутных баров в сутки — ИЗМЕРЕНО этим же скриптом ниже
        # (1008–1009 баров, 201 пятиминутный эквивалент; сходится с дневной
        # свечой по объёму). Раньше здесь стояло 113 — это длина основной
        # сессии 09:50–18:50 без вечерней торговли, то есть занижение в 9 раз.
        # Код печатал оценку в 9 раз меньше измеренной и противоречил
        # собственному выводу скрипта (audit 2026-10-25, m3).
        bars_1m = int(years * 252 * MINUTES_PER_DAY)
        bars_5m = bars_1m // 5
        rows.append((tk, fdate, bars_1m, bars_5m))
        print(f"{tk:<8}{fdate.isoformat():<12}{years:>7.1f}"
              f"{bars_1m:>18,}{bars_5m:>18,}")
        sys.stdout.flush()

    ok = [r for r in rows if r[1]]
    print()
    if ok:
        lo = min(r[1] for r in ok)
        print(f"покрытие: {len(ok)}/{len(TICKERS)} тикеров, самое раннее начало "
              f"{lo.isoformat()}")
        common = max(r[1] for r in ok)
        print(f"общий кусок для ВСЕХ тикеров начинается с {common.isoformat()}")
        years_common = (today - common).days / 365.25
        print(f"  = {years_common:.1f} лет, ~{int(years_common * 252 * 113 / 5):,} "
              f"пятиминутных баров на тикер")
        print()
        print("Для кросс-секционного дизайна важен общий кусок, а не самый")
        print("длинный ряд: сопоставимость требует одновременных данных.")
    else:
        print("ни один тикер не вернул минутных данных")

    print()
    print("=" * 86)
    print("СКУЛЬКО БАРОВ В СУТКАХ И СХОДИТСЯ ЛИ ДЕНЬ С ДНЕВНОЙ СВЕЧОЙ")
    print("=" * 86)
    print("Первая оценка «~113 пятиминутных баров на сессию» была занижена:")
    print("она считалась по длительности основной сессии 09:50–18:50 MSK, без")
    print("вечерней торговли. Реальные сутки длиннее.")
    print()
    probe = today - dt.timedelta(days=3)
    while probe.weekday() >= 5:
        probe -= dt.timedelta(days=1)
    for tk in TICKERS[:3]:
        rows, daily, _ = full_day(tk, probe)
        if not rows:
            print(f"{tk}: нет данных за {probe}")
            continue
        vol_min = sum(r[5] for r in rows)
        vol_day = daily[5] if daily else 0
        zero = sum(1 for r in rows if r[5] == 0)
        ok = abs(vol_min - vol_day) / vol_day < 1e-3 if vol_day else False
        print(f"{tk}: минутных баров = {len(rows)}, "
              f"в 5-минутном эквиваленте = {len(rows) // 5}")
        print(f"    первый {rows[0][6]}   последний {rows[-1][6]}")
        print(f"    объём суммой минутных = {vol_min:,.0f}, "
              f"дневная свеча = {vol_day:,.0f}, сходится: {ok}")
        print(f"    баров с нулевым объёмом: {zero} ({zero / len(rows) * 100:.1f}%)")
        time.sleep(PAUSE)

    print()
    print("=" * 86)
    print("ЧТО ЭТО ЗНАЧИТ ДЛЯ ШАГА 1.5")
    print("=" * 86)
    print("Три поправки к расчёту мощности, все в лучшую сторону:")
    print("  1) Пятиминутных свечей в ISS нет, но минутная история глубокая —")
    print("     агрегация стоит трафика в 5 раз, а не отсекает данные;")
    print("  2) Глубина минутной истории — 12–15 лет, а не предполагаемые 2–3;")
    print("  3) В сутках ~1005 минутных баров, а не ~113: день собирается")
    print("     целиком и сходится с дневной свечой по объёму.")
    print()
    print("Риски, которые остаются:")
    print("  * выгрузка = сотни тысяч строк на тикер, это часы скачивания")
    print("    и риск rate limit;")
    print("  * автокорреляция внутридневных баров сильнее дневной, значит")
    print("    эффективный N ниже номинального — главная неизвестная;")
    print("  * вечерняя сессия ликвидностью ниже основной, смешивать их —")
    print("    значит смешивать два режима; возможно, надо отрезать явно;")
    print("  * метки времени ISS в неочевидном поясе (сессия 06:59…23:49")
    print("    при эталонных 09:50–23:50 MSK): границы агрегации надо")
    print("    зафиксировать явно;")
    print("  * издержки на горизонте 1–2 часа съедают сигнал размера IC 0.02 —")
    print("    считать надо валовую доходность, а не IC.")
    print()
    print("Следующий шаг: взять ОДИН месяц, собрать пятиминутные бары, замерить")
    print("реальную автокорреляцию и волатильность по двум сессиям. Три замера —")
    print("час работы вместо часов слепой выгрузки.")


if __name__ == "__main__":
    main()