@echo off
:: stop_workers.bat — Encerra todos os workers TTS ativos
:: Chamado automaticamente antes de iniciar um novo worker (para liberar VRAM)

echo [+] Encerrando workers TTS ativos...

:: Matar processos Python rodando os servers nas portas 5001/5002
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5001 "') do (
    taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5002 "') do (
    taskkill /F /PID %%a >nul 2>&1
)

echo [OK] Workers encerrados (VRAM liberada).
