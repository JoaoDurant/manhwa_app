import torch
import sys
import os

print(f"PYTHON_EXE: {sys.executable}")
print(f"TORCH_VERSION: {torch.__version__}")
try:
    print(f"TORCH_FILE: {torch.__file__}")
except:
    print("TORCH_FILE: unknown")

print(f"HAS_CUDA_ATTR: {hasattr(torch, 'cuda')}")
if hasattr(torch, 'cuda'):
    print(f"CUDA_AVAILABLE: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"DEVICE_NAME: {torch.cuda.get_device_name(0)}")
else:
    print("CRITICAL: torch.cuda attribute MISSING")

import indextts
print(f"INDEXTTS_FILE: {indextts.__file__}")
