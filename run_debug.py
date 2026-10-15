"""Diagnostics wrapper: faulthandler + single-threaded mode (SIGSEGV trace).

Порядок важен и правильный именно такой:

1. `faulthandler.enable()` — включается до всего, чтобы нативный краш
   (SIGSEGV — это не Python-исключение) печатал C-трейс.
2. env-переменные OMP/MKL/CUDA выставляются ДО `import torch`. Это
   единственный момент, когда OMP-runtime ещё не создал пул потоков,
   поэтому только здесь переменные и действуют. Основной скрипт делает
   то же самое на своём уровне модуля — но он импортируется ПОЗЖЕ, так
   что реальный фикс single-thread обеспечивает именно этот файл.
3. Основной скрипт запускается через runpy; он повторно вызовет
   `torch.set_num_interop_threads(1)`, что запрещено PyTorch (RuntimeError
   при втором вызове) — поэтому вызов обёрнут в try/except в обоих файлах.
"""

import faulthandler
import logging
import os
import runpy

faulthandler.enable()

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import torch

# Импорт logging не поднимает ни одного потока, поэтому single-thread
# гарантия выше от него не страдает. Уровень DEBUG — чтобы причина отказа
# set_num_interop_threads была видна в stderr, а не терялась в `pass`.
# basicConfig() здесь НЕ используется: он поднял бы уровень root-логгера
# до DEBUG, и в stderr хлынул бы поток debug-сообщений torch/transformers,
# а обёртка cron читает из stderr только последние 1500 символов.
_DEBUG_LOG = logging.getLogger("run_debug")
_DEBUG_LOG.setLevel(logging.DEBUG)
_DEBUG_LOG.addHandler(logging.StreamHandler())
_DEBUG_LOG.propagate = False

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError as e:
    # Два разных случая, см. timesfm_kronos_compare._pin_threads(): пул уже
    # создан (ожидаемо) — либо параллельная работа уже началась, и тогда
    # single-thread не гарантирован, а это и есть класс SIGSEGV.
    _DEBUG_LOG.debug("set_num_interop_threads(1) отклонён PyTorch: %s", e)
print("faulthandler + single-threaded mode enabled", flush=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
runpy.run_path(os.path.join(SCRIPT_DIR, "timesfm_kronos_compare.py"),
               run_name="__main__")