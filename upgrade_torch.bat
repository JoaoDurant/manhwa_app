@echo off
echo Matando processos Python presos...
taskkill /F /IM python.exe /T >nul 2>&1
echo.
echo Desinstalando dependencias antigas do PyTorch (%VIRTUAL_ENV%)...
call .\venv_main\Scripts\activate.bat
pip uninstall -y torch torchvision torchaudio xformers flash-attn
echo.
echo Limpando cache do pip...
pip cache purge
echo.
echo Instalando PyTorch mais recente com CUDA 12.6/12.8 (Sm_120 Blackwell Support)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --upgrade
echo.
echo Finalizado.
