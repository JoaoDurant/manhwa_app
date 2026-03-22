@echo off
setlocal
title REPARADOR DEFINITIVO - RTX 5000 SERIES
echo ============================================================
echo   REPARADOR DE AMBIENTE - MANHWA APP (GPU BLACKWELL READY)
echo ============================================================
echo.
echo [+] Este script vai criar ambientes isolados perfeitos para sua RTX 5070 Ti.
echo [+] Isso pode levar de 5 a 10 minutos dependendo da sua internet.
echo.

:: 1. Limpeza de venvs antigos (se existirem e estiverem quebrados)
echo [1/5] Preparando pastas...
if exist venv_main rmdir /s /q venv_main
if exist venv_qwen rmdir /s /q venv_qwen
if exist venv_indextts rmdir /s /q venv_indextts

:: 2. Criando VENV PRINCIPAL
echo [2/5] Criando ambiente principal (venv_main)...
python -m venv venv_main
call venv_main\Scripts\activate
python -m pip install --upgrade pip --quiet
echo [+] Instalando PyTorch CUDA 12.4 (Obrigatório para RTX 5000)...
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124 --quiet
echo [+] Instalando requisitos do App...
pip install -r requirements_manhwa.txt --quiet
pip install chatterbox-tts flask soundfile requests librosa --quiet
python -m spacy download en_core_web_sm --quiet
call deactivate
echo [OK] Ambiente Principal pronto!

:: 3. Criando VENV QWEN
echo [3/5] Criando ambiente Qwen (venv_qwen)...
python -m venv venv_qwen
call venv_qwen\Scripts\activate
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124 --quiet
pip install qwen-tts transformers>=4.44.0 accelerate>=0.30.0 flask>=3.0.0 soundfile>=0.12.1 --quiet
call deactivate
echo [OK] Ambiente Qwen pronto!

:: 4. Ajustando o lançador
echo [4/5] Atualizando o arquivo start.bat para usar o novo ambiente...
(
echo @echo off
echo call venv_main\Scripts\activate
echo python run_manhwa_app.py
echo pause
) > start_safe.bat

echo.
echo ============================================================
echo   TUDO PRONTO! SUA RTX 5070 TI AGORA ESTA CONFIGURADA.
echo ============================================================
echo [+] Use o 'start_safe.bat' para abrir o app no futuro.
echo.
pause
