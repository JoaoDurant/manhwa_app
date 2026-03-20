@echo off
setlocal
title Manhwa Video Creator - Setup & Start

echo ========================================================
echo        MANHWA VIDEO CREATOR - AUTO INSTALLER
echo ========================================================
echo.

:: Verificar se o Python está instalado
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERRO] Python nao encontrado! Instale o Python 3.10+ para continuar.
    pause
    exit /b
)

:: Criar ambiente virtual se não existir
if not exist "venv" (
    echo [+] Criando ambiente virtual (venv)...
    python -m venv venv
)

:: Ativar ambiente virtual
echo [+] Ativando ambiente virtual...
call venv\Scripts\activate

:: Atualizar PIP
echo [+] Atualizando PIP...
python -m pip install --upgrade pip

:: Instalar dependencias
echo [+] Instalando dependencias (isso pode demorar varios minutos)...
pip install -r requirements.txt

:: Baixar modelos SpaCy necessários
echo [+] Baixando modelos de linguagem (SpaCy)...
python -m spacy download pt_core_news_lg
python -m spacy download en_core_web_sm

echo.
echo ========================================================
echo        SETUP CONCLUIDO! INICIANDO O PROGRAMA...
echo ========================================================
echo.

:: Iniciar o programa
python run_manhwa_app.py

pause
