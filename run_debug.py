import faulthandler, sys
faulthandler.enable()
import torch
torch.set_num_threads(1)
print("faulthandler + single-threaded mode enabled", flush=True)
import runpy
runpy.run_path("timesfm_kronos_compare.py", run_name="__main__")
