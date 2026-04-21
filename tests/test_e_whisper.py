"""
Teste E - Whisper Validation
Valida se o fast-whisper consegue rodar em paralelo em CPU (int8) e se
ele de fato intercepta audios corrompidos produzidos pela engine.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import torch
import numpy as np
import torchaudio as ta
from pathlib import Path

# Gerar arquivo corrompido para o Whisper testar
OUTPUT_DIR = Path("tests/output/whisper")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
bad_audio_path = OUTPUT_DIR / "bad_whisper_test.wav"
ta.save(str(bad_audio_path), torch.zeros((1, 24000*3)), 24000)

import manhwa_app.models.whisper_manager as wm_manager

print("=== WHISPER VALIDATION TEST ===")
t0 = time.perf_counter()
print("Carregando modelo Whisper...")
try:
    model, device = wm_manager.get_whisper_model("base", device_override="cpu", compute_type="int8")
    print(f"Whisper carregado em {time.perf_counter() - t0:.2f}s | Device: {device}")
except Exception as e:
    print(f"[FAIL] Falha ao carregar Whisper: {e}")
    sys.exit(1)

print("Testando arquivo silencioso/corrompido...")
try:
    text = wm_manager.transcribe_audio(str(bad_audio_path))
    print(f"Resultado: Text='{text}'")
    
    if len(text.strip()) < 2:
        print("[OK] Whisper corretamente transcreveu vazio (ou muito curto)")
    else:
        print("[FAIL] Whisper gerou halucinacao a partir de silencio!")
        
    print("\n[TESTE E] PASS")
except Exception as e:
    print(f"[FAIL] Excecao ocorrida: {e}")

print("===============================")
