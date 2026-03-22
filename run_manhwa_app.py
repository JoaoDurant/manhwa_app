#!/usr/bin/env python3
"""
run_manhwa_app.py - Launcher for the Manhwa Video Creator desktop application.

Run from the Chatterbox TTS Server root directory:
    python run_manhwa_app.py

Requirements:
    - The existing Chatterbox TTS Server virtual environment must be activated.
    - Extra dependencies installed via:
        pip install -r requirements_manhwa.txt
    - FFmpeg must be installed and available in PATH.
"""

import os
import sys
import subprocess
from pathlib import Path

def restart_with_venv():
    """Tenta reiniciar o app usando o venv_main se o ambiente global estiver quebrado."""
    root = Path(__file__).resolve().parent
    venv_python = root / "venv_main" / "Scripts" / "python.exe"
    if venv_python.exists() and sys.executable != str(venv_python):
        print(f"\n[AUTO-FIX] Ambiente global inconsistente. Reiniciando via venv_main...")
        cmd = [str(venv_python)] + sys.argv
        sys.exit(subprocess.call(cmd))

# Verificacao de integridade de dependencias criticas
try:
    import torch
    # Se torch estiver vazio ou quebrado (erro comum em upgrade falho)
    if not hasattr(torch, "LongTensor"):
        restart_with_venv()
except (ImportError, AttributeError):
    restart_with_venv()

# Se chegou aqui e ainda nao tem torch, avisa o usuario
try:
    import torch
except ImportError:
    print("\n" + "="*60)
    print(" ERRO CRITICO: Ambiente Python corrompido.")
    print(" Acao: Por favor, execute o arquivo: SUPER_FIX_RTX5000.bat")
    print("="*60)
    import time; time.sleep(5)
    sys.exit(1)

# -----------------------------------------------------------------------
# NOTA: A variável CUDA_LAUNCH_BLOCKING=1 foi removida pois alterava o
# scheduling interno do Chatterbox TTS causando device-side asserts.
# Para resolver o cudaErrorLaunchTimeout (Windows WDDM TDR timeout),
# a solução definitiva é aumentar o TdrDelay no registro do Windows:
#   reg add "HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" /v TdrDelay /t REG_DWORD /d 60 /f
#   (Reiniciar o PC após aplicar)

# Ensure this file's directory (repo root) is on sys.path so that
# engine.py, config.py, utils.py etc. are importable.
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# --- POLYFILL FOR TRANSFORMERS VERSION CONFLICT ---
try:
    import transformers.generation.logits_process as lp
    if not hasattr(lp, "UnnormalizedLogitsProcessor"):
        class UnnormalizedLogitsProcessor:
            def __init__(self, *args, **kwargs): pass
            def __call__(self, input_ids, scores, **kwargs): return scores
        lp.UnnormalizedLogitsProcessor = UnnormalizedLogitsProcessor
        print("[DEBUG] Polyfilled UnnormalizedLogitsProcessor for Transformers compatibility.")
    
    if not hasattr(lp, "MinPLogitsWarper"):
        class MinPLogitsWarper:
            def __init__(self, *args, **kwargs): pass
            def __call__(self, input_ids, scores, **kwargs): return scores
        lp.MinPLogitsWarper = MinPLogitsWarper
        print("[DEBUG] Polyfilled MinPLogitsWarper for Transformers compatibility.")
except ImportError:
    pass

# Change working directory to repo root so relative paths (config.yaml, output/, etc.) work.
os.chdir(str(ROOT_DIR))


def check_dependencies():
    """Check that required packages are installed and print helpful errors."""
    missing = []

    try:
        import PySide6
    except ImportError:
        missing.append("PySide6")

    try:
        import PIL
    except ImportError:
        missing.append("Pillow")

    try:
        import whisper
    except ImportError:
        missing.append("openai-whisper")

    try:
        import pydub
    except ImportError:
        missing.append("pydub")

    if missing:
        print("=" * 60)
        print("ERROR: Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print()
        print("Install them with:")
        print("  pip install -r requirements_manhwa.txt")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    check_dependencies()

    # --- VERIFICAÇÃO DE AMBIENTE (WORKERS) ---
    root = Path(__file__).resolve().parent
    missing_venvs = []
    if not (root / "venv_qwen").exists(): missing_venvs.append("venv_qwen")
    if not (root / "venv_indextts").exists(): missing_venvs.append("venv_indextts")
    
    if missing_venvs:
        print("\n" + "!"*60)
        print(" AVISO: WORKERS NÃO INSTALADOS")
        print(" O Qwen e o IndexTTS não vão funcionar até você resolver isso.")
        print(" Ação: Feche este terminal e execute: setup_workers.bat")
        print("!"*60 + "\n")

    # --- VERIFICAÇÃO DE COMPATIBILIDADE GPU (RTX 50-SERIES) ---
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            torch_ver = torch.__version__
            # Alerta apenas se for placa Blackwell (sm_100+) E pytorch antigo (+cu11)
            if major >= 10 and "+cu11" in torch_ver:
                print("\n" + "#"*60)
                print(" ALERTA DE COMPATIBILIDADE GPU (RTX 5070 Ti detectada)")
                print(f" Seu PyTorch ({torch_ver}) é incompatível com esta placa.")
                print(" Isso vai causar CRASH durante a geração.")
                print(" Ação: Feche o app e execute o arquivo: SUPER_FIX_RTX5000.bat")
                print("#"*60 + "\n")
    except Exception:
        pass

    try:
        from manhwa_app.app import main
        main()
    except Exception as e:
        print("\n" + "="*60)
        print(f" CRASH FATAL NO APP: {e}")
        print(" Possível causa: Incompatibilidade de Driver/PyTorch com RTX 5000.")
        print(" Tente atualizar o PyTorch conforme o alerta acima.")
        print("="*60)
        input("\nPressione Enter para sair...")
