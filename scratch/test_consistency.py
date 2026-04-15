# scratch/test_consistency.py
import sys
import os
import torch
import numpy as np
from pathlib import Path

# Adiciona o diretório raiz ao sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import engine

def test():
    print("--- Testando Determinismo Industrial - RTX 5070 Ti ---")
    
    if not engine.load_model("turbo"):
        print("FAIL: Falha no load.")
        return

    text = "Consistency is the foundation of quality."
    seed = 42
    
    print(f"\n[RUN 1] Gerando com semente {seed}...")
    w1, _ = engine.synthesize(text, seed=seed)
    
    print(f"[RUN 2] Gerando com semente {seed} (deve ser identico)...")
    w2, _ = engine.synthesize(text, seed=seed)
    
    if w1 is None or w2 is None:
        print("FAIL: Geracao falhou.")
        return

    # Compara tensores
    diff = torch.abs(w1 - w2).max().item()
    print(f"\nINFO: Diferenca Maxima entre Tensores: {diff:.8f}")
    
    if diff < 1e-4:
        print("OK: As geracoes sao DETERMINISTICAS!")
    else:
        print("WARN: Variacao detectada. Mas verifique a entonacao.")

if __name__ == "__main__":
    test()
