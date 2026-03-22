@echo off
setlocal
title Manhwa Video Creator - Setup
 
echo ========================================================
echo  MANHWA VIDEO CREATOR — SETUP COMPLETO (PADRAO UV)
echo ========================================================
echo.
 
:: Verificar Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERRO] Python 3.10+ nao encontrado.
    pause & exit /b 1
)
 
:: Instalar uv
uv --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [+] Instalando uv (gerenciador de pacotes moderno)...
    powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    echo [AVISO] Feche e reabra o terminal, depois execute setup.bat novamente.
    pause & exit /b 0
)
echo [OK] uv disponivel.
 
:: Venv principal (Chatterbox + Kokoro + app + Qwen)
if not exist "venv\Scripts\activate" (
    echo [+] Criando venv principal...
    python -m venv venv
)
call venv\Scripts\activate
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
python -m spacy download pt_core_news_lg --quiet >nul 2>&1
python -m spacy download en_core_web_sm  --quiet >nul 2>&1
 
:: Qwen3-TTS no venv principal
echo [+] Instalando Qwen3-TTS no ambiente principal...
pip install "transformers>=4.44.0" "accelerate>=0.30.0" "numpy>=2.0.0" --quiet
pip install --upgrade qwen-tts --quiet
if %errorlevel% neq 0 (
    echo [AVISO] Falha ao instalar qwen-tts via pip. Tentando do repositorio...
    pip install git+https://github.com/QwenLM/Qwen3-TTS.git --quiet
)
echo [+] Tentando instalar flash-attn (opcional)...
pip install flash-attn --no-build-isolation --quiet 2>nul
call deactivate
echo [OK] Venv principal configurado.
 
:: Criar pasta engines se nao existir
if not exist "engines\" mkdir engines
 
:: IndexTTS (uv venv isolado)
echo.
echo [+] Configurando IndexTTS com uv...
if not exist "engines\index-tts\" (
    echo [+] Clonando IndexTTS...
    git clone https://github.com/index-tts/index-tts.git engines\index-tts
    if %errorlevel% neq 0 (
        echo [ERRO] Falha ao clonar. Verifique git e conexao.
        pause & exit /b 1
    )
)
 
cd engines\index-tts
echo [+] Instalando dependencias via uv sync (pode demorar)...
uv sync --no-dev --link-mode=copy 2>nul
if %errorlevel% neq 0 (
    echo [+] Tentando sem extras opcionais (deepspeed)...
    uv sync --no-dev --no-extra deepspeed --link-mode=copy
)
 
:: Copiar worker para dentro do repo (usando os caminhos relativos corretos)
if exist "..\..\engine_workers\tts_worker.py" (
    copy /Y "..\..\engine_workers\tts_worker.py" "tts_worker.py" >nul
) else (
    echo [AVISO] engine_workers/tts_worker.py nao encontrado.
)
 
:: Baixar modelo se necessario
if not exist "checkpoints\gpt.pth" (
    echo [+] Baixando IndexTTS-2 (pode demorar varios minutos)...
    uv tool run huggingface-cli download IndexTeam/IndexTTS-2 --local-dir=checkpoints
)
cd ..\..
echo [OK] IndexTTS configurado.
 
echo.
echo ========================================================
echo  SETUP CONCLUIDO!
echo  Execute: start.bat
echo ========================================================
pause
