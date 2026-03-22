import sys
from pathlib import Path

# Adiciona o diretório atual ao sys.path para simular a rodada pela raiz do projeto
sys.path.insert(0, str(Path(__file__).parent.resolve()))

import engine

def test_engine(name, loader):
    print(f"\n--- Testando {name} ---")
    try:
        success = loader()
        if success:
            print(f"✅ {name} carregado com sucesso.")
        else:
            print(f"❌ Falha ao carregar {name}.")
    except Exception as e:
        print(f"💥 Erro inesperado no {name}: {e}")

if __name__ == "__main__":
    print("Testando fallbacks do Engine.py")
    test_engine("Chatterbox", engine.load_model)
    test_engine("Kokoro", engine.load_kokoro_engine)
    test_engine("Qwen", engine.load_qwen_model)
