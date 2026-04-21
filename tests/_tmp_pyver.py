import sys, subprocess
print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")
# Check torch version
try:
    import torch
    print(f"Torch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Arch List: {torch.cuda.get_arch_list()}")
    sm120 = "sm_120" in torch.cuda.get_arch_list()
    print(f"sm_120 nativo: {'YES' if sm120 else 'NO'}")
except ImportError as e:
    print(f"torch not found: {e}")
