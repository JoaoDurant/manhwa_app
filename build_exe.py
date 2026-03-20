import os
import subprocess
import sys
import shutil

def main():
    print("Iniciando processo de build do Manhwa Video Creator...")
    
    # 1. Instalar o PyInstaller se necessário
    try:
        import PyInstaller
    except ImportError:
        print("PyInstaller não encontrado. Instalando...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        
    # 2. Executar o PyInstaller
    print("\nExecutando o PyInstaller. Isso pode demorar vários minutos devido às bibliotecas pesadas (PyTorch, Diffusers, etc.)...")
    
    # Parâmetros:
    # --noconfirm: sobrescreve a pasta dist anterior
    # --onedir: cria uma pasta contendo o .exe (melhor para apps que usam PyTorch, pois --onefile extrai 3GB de temp files a cada vez)
    # --windowed: oculta a janela do prompt de comando (já que é uma GUI)
    # --name: Nome do app
    # --icon: Ícone do app
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--windowed",
        "--name", "ManhwaCreator",
        "--icon", "icon.ico",
        # Inclui a pasta manhwa_app explicitly
        "--add-data", f"manhwa_app{os.pathsep}manhwa_app",
        "--add-data", f"config.json{os.pathsep}.",
        "--add-data", f"config.yaml{os.pathsep}.",
        # Ignora arquivos desnecessários de teste para poupar um pouco de espaço, se possível
        "--exclude-module", "matplotlib",
        "--exclude-module", "tensorboard",
        "run_manhwa_app.py"
    ]
    
    try:
        subprocess.check_call(cmd)
        print("\n\n✓ Build concluído com sucesso!")
        print("O arquivo executável está localizado na pasta: dist/ManhwaCreator/")
        print("NOTA: Para que o programa funcione corretamente, certifique-se de executar o ManhwaCreator.exe a partir da pasta raiz do projeto, ou copie a pasta 'voices' e 'model_cache' para dentro da pasta dist/ManhwaCreator.")
    except Exception as e:
        print(f"\nErro durante o build: {e}")

if __name__ == "__main__":
    main()
