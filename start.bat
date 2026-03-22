@echo off
setlocal
title Manhwa Video Creator
 
:: Ativar venv principal e rodar
if exist "venv\Scripts\activate" (
    call venv\Scripts\activate
)
python run_manhwa_app.py
pause
