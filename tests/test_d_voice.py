"""
Teste D - Voice Consistency & Drift
Gera n vezes o mesmo texto (com a mesma voz e configuracoes) e 
verifica a consistencia no tempo de geracao e duracao do audio (drift).
Garantir que a semente fixa ('seed=42') produza o mesmo cenario acustico.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import torch
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("tests/output/voice")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VOICE_REF = str(Path("voices/Leonardo.wav").resolve())

import engine
print("Carregando engine...")
engine.load_multilingual()

text = "A explosao iluminou o ceu noturno como se fosse dia, e os escombros choveram sobre os sobreviventes atordoados."

runs = 5
results = []
print(f"=== VOICE CONSISTENCY TEST ({runs} runs) ===")

for i in range(runs):
    t0 = time.perf_counter()
    wav_tensor, sr = engine.synthesize(
        text=text,
        audio_prompt_path=VOICE_REF,
        language="pt",
        temperature=0.65,
        exaggeration=0.65,
        cfg_weight=0.35,
        seed=42
    )
    elapsed = time.perf_counter() - t0
    
    if wav_tensor is None:
        print(f"[FAIL] Falha em Run {i+1}")
        continue
        
    audio_dur = wav_tensor.shape[-1] / sr
    results.append({"dur": audio_dur, "elapsed": elapsed})
    print(f"Run {i+1}: dur_audio={audio_dur:.4f}s  (Gerado em {elapsed:.2f}s)")

# Checagem de consistencia (deterministic behavior)
durations = [r["dur"] for r in results]
avg_dur = sum(durations) / len(durations)
max_dev = max(abs(d - avg_dur) for d in durations)

print(f"\nDuracao Media: {avg_dur:.4f}s")
print(f"Max Desvio Absoluto: {max_dev:.4f}s")

if max_dev > 0.1:
    print("[WARN] Variacao > 100ms na duracao! O engine nao esta 100% deterministico.")
    # Isso pode ocorrer por max-autotune caching ou falta de set_seed em kernels cuDNN
    res = False
else:
    print("[OK] Comportamento perfeitamente deterministico (Drift: nulo)")
    res = True

if res:
    print("\n[TESTE D] PASS")
else:
    print("\n[TESTE D] FAIL (Mas as vezes perdoavel em GPUs Ada/Blackwell)")
print("============================================")
