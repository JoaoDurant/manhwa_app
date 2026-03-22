@echo off
setlocal
title Manhwa App — Workers TTS

echo ============================================
echo  MANHWA APP — INICIANDO WORKERS TTS
echo ============================================
echo.

:: Verificar se os venvs existem
if not exist "venv_qwen\Scripts\python.exe" (
    echo [ERRO] venv_qwen nao encontrado.
    echo Execute setup_workers.bat primeiro.
    pause
    exit /b 1
)

if not exist "venv_indextts\Scripts\python.exe" (
    echo [ERRO] venv_indextts nao encontrado.
    echo Execute setup_workers.bat primeiro.
    pause
    exit /b 1
)

echo [+] Iniciando worker Qwen3-TTS na porta 5001...
start "Qwen Worker (porta 5001)" /min ^
    venv_qwen\Scripts\python.exe workers\qwen_server.py

echo [+] Iniciando worker IndexTTS na porta 5002...
start "IndexTTS Worker (porta 5002)" /min ^
    venv_indextts\Scripts\python.exe workers\indextts_server.py

echo.
echo [+] Aguardando workers inicializarem (modelos carregam em ~30-60s)...
echo     Voce pode acompanhar nos terminais abertos em background.
echo.
echo [OK] Workers iniciados. Pode fechar esta janela.
echo      Para encerrar os workers: feche as janelas "Qwen Worker" e "IndexTTS Worker"
