#!/usr/bin/env python3
import sys,platform,shutil
print("Python:",sys.version.split()[0]);print("OS:",platform.platform())
for exe in ("ffmpeg","ffprobe"):print(exe,shutil.which(exe) or "MISSING")
try:
 import torch
 print("Torch:",torch.__version__); print("CUDA available:",torch.cuda.is_available()); print("MPS available:",bool(getattr(torch.backends,"mps",None) and torch.backends.mps.is_available()))
except Exception as e:print("Torch: not installed (baseline remains usable)",e)
try:
 import transformers; print("Transformers:",transformers.__version__)
except Exception:print("Transformers: not installed (model branches disabled)")
