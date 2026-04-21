import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

print("=== GPU DIAGNOSTIC ===")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
props = torch.cuda.get_device_properties(0)
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"VRAM Total: {props.total_memory / 1e9:.1f} GB")
arch_list = torch.cuda.get_arch_list()
print(f"Arch List: {arch_list}")
sm120_ok = "sm_120" in arch_list
print(f"sm_120 nativo: {'[OK] SIM' if sm120_ok else '[FAIL] NAO - PTX fallback ativo (instale cu128)'}")
print(f"BF16 suportado: {torch.cuda.is_bf16_supported()}")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print(f"TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}")
print(f"cuDNN deterministic: {torch.backends.cudnn.deterministic}")
alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "NAO CONFIGURADO")
print(f"ALLOC_CONF: {alloc}")
vram_free = torch.cuda.mem_get_info()[0] / 1e9
vram_total = props.total_memory / 1e9
print(f"VRAM Livre: {vram_free:.1f} GB / {vram_total:.1f} GB")
compute_cap = f"sm_{props.major}{props.minor}"
print(f"Compute Capability: {compute_cap}")

print("\n=== ANALISE ===")
all_ok = True
if not sm120_ok:
    print("[CRITICO] sm_120 NOT in arch_list!")
    print("   PTX JIT fallback ativo => performance 3-5x inferior")
    all_ok = False
else:
    print("[OK] sm_120 nativo confirmado - CUDA kernels otimizados")

if not torch.backends.cuda.matmul.allow_tf32:
    print("[WARN] TF32 matmul desativado - ativando agora")
else:
    print("[OK] TF32 matmul ativado")

if not torch.cuda.is_bf16_supported():
    print("[WARN] BF16 nao suportado")
    all_ok = False
else:
    print("[OK] BF16 suportado - precisao mista disponivel")

if all_ok:
    print("\n[TESTE A] PASS - GPU totalmente compativel com RTX 5070 Ti")
else:
    print("\n[TESTE A] FAIL - Ver problemas acima")
print("=== GPU DIAGNOSTIC COMPLETE ===")
