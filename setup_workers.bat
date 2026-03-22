@echo off
setlocal
title Manhwa App — Setup Workers

echo ============================================
echo  SETUP WORKERS TTS (primeira vez apenas)
echo ============================================
echo.

:: ---- VENV QWEN ----
echo [1/4] Criando venv_qwen...
if not exist "venv_qwen" (
    python -m venv venv_qwen
)

echo [2/4] Instalando dependencias do Qwen...
call venv_qwen\Scripts\activate

:: PyTorch com CUDA 12.8 primeiro (RTX 5070 Ti)
pip install torch torchvision torchaudio ^
    --index-url https://download.pytorch.org/whl/cu128 ^
    --quiet

:: Dependencias do Qwen
pip install --upgrade qwen-tts transformers>=4.44.0 ^
    accelerate>=0.30.0 numpy>=2.0.0 ^
    flask>=3.0.0 soundfile>=0.12.1 ^
    --quiet

echo [OK] venv_qwen configurado.
call deactivate

:: ---- VENV INDEXTTS ----
echo [3/4] Criando venv_indextts...
if not exist "venv_indextts" (
    python -m venv venv_indextts
)

echo [4/4] Instalando dependencias do IndexTTS...
call venv_indextts\Scripts\activate

:: PyTorch com CUDA 12.8 primeiro
pip install torch torchvision torchaudio ^
    --index-url https://download.pytorch.org/whl/cu128 ^
    --quiet

:: IndexTTS do repo oficial
pip install git+https://github.com/index-tts/index-tts.git --quiet

:: Dependencias adicionais
pip install transformers>=4.44.0 accelerate>=0.30.0 ^
    numpy>=2.0.0 flask>=3.0.0 soundfile>=0.12.1 ^
    huggingface_hub pynini==2.1.6 --find-links https://github.com/kylebgorman/pynini/releases --quiet

echo [OK] venv_indextts configurado.
call deactivate

echo.
echo ============================================
echo  SETUP CONCLUIDO!
echo  Proximos passos:
echo  1. Baixar modelo IndexTTS-2:
echo     venv_indextts\Scripts\python -c "from huggingface_hub import snapshot_download; snapshot_download('IndexTeam/IndexTTS-2', local_dir='checkpoints')"
echo  2. Executar: start_workers.bat
echo  3. Executar: start.bat (app principal)
echo ============================================
pause
