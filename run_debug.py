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
import os
import runpy

faulthandler.enable()

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import torch

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
print("faulthandler + single-threaded mode enabled", flush=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
runpy.run_path(os.path.join(SCRIPT_DIR, "timesfm_kronos_compare.py"),
               run_name="__main__")