# TimesFM 2.5 + Kronos Mini — Сравнение сигналов MOEX

AI-ансамбль из двух моделей для прогнозирования цен акций MOEX (Сбербанк преф, Газпром, Лукойл).

## Что делает

Загружает данные с MOEX ISS API → запускает **TimesFM 2.5** (Google Foundation Model) и **Kronos Mini** (4M params) → сравнивает сигналы BUY/SELL/HOLD на горизонтах 30/60/90 дней.

## Быстрый старт — выбор варианта

### Вариант А: CUDA (GPU с sm_80+ — Ampere RTX 30xx / A100 / H100 / Hopper)
```bash
# Создать venv (Python 3.12; НЕ использовать Hermes global venv)
uv venv .venv --python 3.12

# Установить зависимости с CUDA (PyTorch cu121)
uv pip install -r requirements-cuda.txt

# Запустить сравнение (автоматически использует GPU)
python timesfm_kronos_compare.py
```

### Вариант Б: CPU-only (любой CPU, GPU без CUDA-поддержки, Blackwell RTX 50xx sm_120)
```bash
# Создать venv (Python 3.12; НЕ использовать Hermes global venv)
uv venv .venv --python 3.12

# Установить зависимости CPU-only (PyTorch CPU)
uv pip install -r requirements-cpu.txt

# Запустить сравнение (принудительно CPU)
python timesfm_kronos_compare.py
```

> **Важно:** RTX 5060 Ti (Blackwell, sm_120) **не поддерживается** стабильным PyTorch. Используй CPU-вариант. Если у тебя Ampere (RTX 30xx) или новее с поддержкой — CUDA-вариант даст ускорение в 5-10x.

> **Принудительный CPU:** В `load_kronos()` / `load_timesfm()` стоит `device="cpu"`. Для CUDA запуска поменяй на `device="cuda"`.

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
├── requirements-cpu.txt        # CPU-only стек (torch CPU)
├── requirements-cuda.txt       # CUDA стек (torch cu121)
├── logs/                       # результаты (автоматически)
│   └── comparison_YYYYMMDD.log
├── .gitignore
└── README.md
```

> Kronos Mini загружается из соседнего проекта `~/kronos-signal/`
> (модели лежат в `~/kronos-signal/models/Kronos-mini/` и
> `~/kronos-signal/models/Kronos-Tokenizer-2k/`). См. `load_kronos()`.

## Модели

| Модель | Размер | Тип | Горизонт |
|--------|--------|-----|----------|
| **TimesFM 2.5** | 200M params | Foundation Model (Google) | до 128 дней |
| **Kronos Mini** | 4M params | Autoregressive (MOEX) | до ~20 дней |

> **Ограничение горизонта:** `HORIZONS = [30, 60, 90]` защищён `assert max(HORIZONS) <= 128`
> в `main()`. Чтобы выйти за 128 дней, пересоберите TimesFM с большим
> `max_horizon` в `load_timesfm()`.

## Сигналы

- **CONCORDANT**: обе модели согласны → HIGH confidence
- **NEUTRAL-MATCH**: обе близки к текущей цене → MEDIUM confidence
- **CONFLICT**: модели расходятся → LOW confidence, ждать подтверждения

## Производительность

Внутри `main()` результаты 30-дневного прогноза (история + TimesFM-forecast)
кэшируются в `chart_data[ticker]`, поэтому секция графиков **не** повторяет
запрос к MOEX ISS (~30 HTTP-вызовов на тикер) и **не** гоняет TimesFM
повторно. Это сокращает время работы примерно вдвое.

## Зависимости

- Python 3.12+
- `timesfm` (Google)
- `torch` — выбери вариант: CPU-only (`requirements-cpu.txt`) или CUDA (`requirements-cuda.txt`)
- `einops`, `requests`, `numpy`, `pandas`, `matplotlib`
- Kronos Mini — локально из `~/kronos-signal/models/`
- MOEX ISS API (бесплатно, без ключа)