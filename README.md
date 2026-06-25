# TimesFM 2.5 + Kronos Mini — Сравнение сигналов MOEX

AI-ансамбль из двух моделей для прогнозирования цен акций MOEX (Сбербанк преф, Газпром, Лукойл).

## Что делает

Загружает данные с MOEX ISS API → запускает **TimesFM 2.5** (Google Foundation Model) и **Kronos Mini** (4M params) → сравнивает сигналы BUY/SELL/HOLD на горизонтах 30/60/90 дней.

## Быстрый старт

```bash
# Создать venv
uv venv .venv --python 3.12

# Установить зависимости
uv pip install timesfm einops torch yfinance pandas matplotlib requests pyyaml

# Запустить сравнение
python timesfm_kronos_compare.py
```

## Результаты

- **stdout**: таблица сравнения + интерпретация сигналов
- **График**: `timesfm_kronos_comparison.png`
- **Лог**: `logs/comparison_YYYYMMDD.log` (автоматически)

## Cronjob

Запускается автоматически в 11:00 и 18:00 каждый день через Hermes Agent cron.

```bash
# Проверить статус cronjob
hermes cron list

# Запустить вручную
hermes cron run <job_id>
```

## Архитектура

```
timesfm-kronos-comparison/
├── timesfm_kronos_compare.py   # основной скрипт
├── logs/                       # результаты (автоматически)
│   └── comparison_YYYYMMDD.log
├── .gitignore
└── README.md
```

## Модели

| Модель | Размер | Тип | Горизонт |
|--------|--------|-----|----------|
| **TimesFM 2.5** | 200M params | Foundation Model (Google) | до 128 дней |
| **Kronos Mini** | 4M params | Autoregressive (MOEX) | до ~20 дней |

## Сигналы

- **CONCORDANT**: обе модели согласны → HIGH confidence
- **NEUTRAL-MATCH**: обе близки к текущей цене → MEDIUM confidence  
- **CONFLICT**: модели расходятся → LOW confidence, ждать подтверждения

## Зависимости

- Python 3.12+
- timesfm (Google)
- Kronos Mini (локально из `~/kronos-signal/models/`)
- MOEX ISS API (бесплатно, без ключа)
