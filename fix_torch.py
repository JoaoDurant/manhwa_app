
import subprocess
import sys

def run_pip(args):
    cmd = [sys.executable, "-m", "pip"] + args
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error (code {result.returncode}):")
        print(result.stdout)
        print(result.stderr)
    else:
        print("Success!")
    return result.returncode == 0

# Instalar versao especifica e compativel com RTX 5000 (2.5.1 + CU124)
success = run_pip([
    "install", 
    "torch==2.5.1+cu124", 
    "torchvision==0.20.1+cu124", 
    "torchaudio==2.5.1+cu124", 
    "--index-url", "https://download.pytorch.org/whl/cu124",
    "--no-cache-dir"
])

if success:
    print("\n[OK] PyTorch atualizado com sucesso!")
else:
    print("\n[FAIL] Nao foi possivel atualizar automaticamente.")
