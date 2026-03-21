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

import sys
import os
from pathlib import Path

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

    from manhwa_app.app import main
    main()
