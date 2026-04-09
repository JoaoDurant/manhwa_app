@echo off
setlocal
cd /d "%~dp0"
title Manhwa Video Creator - Launcher

echo ==========================================
echo    Iniciando Manhwa Video Creator...
echo ==========================================

:: Verifica se o ambiente virtual existe
if not exist "venv_main\Scripts\python.exe" (
    echo [ERRO] Ambiente virtual 'venv_main' nao encontrado na pasta atual.
    echo Certifique-se de que a instalacao foi concluida corretamente.
    echo.
    pause
    exit /b 1
)

:: Executa o launcher principal usando o python do venv
echo [INFO] Utilizando interpretador: venv_main
"venv_main\Scripts\python.exe" run_manhwa_app.py

:: Se o script fechar por erro ou conclusao
if %ERRORLEVEL% neq 0 (
    echo.
    echo [AVISO] O programa encerrou com codigo de erro %ERRORLEVEL%.
    pause
) else (
    echo.
    echo [INFO] Programa finalizado com sucesso.
)

endlocal
