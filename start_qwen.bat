@echo off
title Qwen3-TTS Worker

if not exist "venv_qwen\Scripts\python.exe" (
    echo [ERRO] venv_qwen nao encontrado. Execute setup_workers.bat primeiro.
    pause & exit /b 1
)

:: Liberar VRAM — encerrar qualquer outro worker ativo
call stop_workers.bat

echo [+] Iniciando Qwen3-TTS na porta 5001 (VRAM inteira disponivel)...
echo     Aguarde o modelo carregar (~30s).
echo.
venv_qwen\Scripts\python.exe workers\qwen_server.py
