@echo off
title IndexTTS Worker

if not exist "venv_indextts\Scripts\python.exe" (
    echo [ERRO] venv_indextts nao encontrado. Execute setup_workers.bat primeiro.
    pause & exit /b 1
)

:: Liberar VRAM — encerrar qualquer outro worker ativo
call stop_workers.bat

echo [+] Iniciando IndexTTS na porta 5002 (VRAM inteira disponivel)...
echo     Aguarde o modelo carregar (~45s).
echo.
venv_indextts\Scripts\python.exe workers\indextts_server.py
