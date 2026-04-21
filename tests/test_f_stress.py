"""
Teste F - Batch Stress Test
Roda multiplos paragrafos em sequencia para verificar:
1. Memory leaks (VRAM subindo sem parar)
2. Degradacao de velocidade (throttling ou gargalos)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import torch
import gc
from pathlib import Path

OUTPUT_DIR = Path("tests/output/stress")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VOICE_REF = str(Path("voices/Leonardo.wav").resolve())

import engine
print("Carregando engine...")
engine.load_multilingual()

textos = [
    "Ele não deveria ter feito isso.",
    "O guerreiro empurrou seus limites além do impossível, canalizando toda a sua energia vital para um único golpe devastador.",
    "A explosão iluminou o céu noturno como se fosse dia, e os escombros choveram sobre os sobreviventes atordoados.",
    "Ele abriu a porta e... não havia nada. Apenas escuridão.",
    "O monstro de nível S tinha 500 mil pontos de vida e regenerava 1% a cada 3 segundos."
] * 4 # 20 iteracoes

runs = len(textos)
print(f"=== BATCH STRESS TEST ({runs} runs) ===")

vram_history = []
times_history = []

for i, text in enumerate(textos):
    t0 = time.perf_counter()
    wav_tensor, sr = engine.synthesize(
        text=text,
        audio_prompt_path=VOICE_REF,
        language="pt",
        temperature=0.65,
        exaggeration=0.65,
        cfg_weight=0.35,
        seed=42 + i
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    
    if wav_tensor is None:
        print(f"[FAIL] Run {i+1} falhou!")
        break
        
    times_history.append(elapsed)
    
    # Check VRAM (memory reserved for caching by PyTorch allocator)
    if torch.cuda.is_available():
        vram_alloc = torch.cuda.memory_allocated(0) / 1e9
        vram_res = torch.cuda.memory_reserved(0) / 1e9
        vram_history.append((vram_alloc, vram_res))
        mem_str = f"| VRAM Alloc: {vram_alloc:.2f}GB (Res: {vram_res:.2f}GB)"
    else:
        mem_str = ""
        vram_history.append((0, 0))
        
    print(f"[{i+1:02d}/{runs}] {elapsed:.2f}s " + mem_str)
    
    # Simular pipeline real: delete tensor
    del wav_tensor
    # gc.collect() as in real app sometimes
    gc.collect()

import sqlite3 # dummy line

# Analise
base_vram_res = vram_history[0][1] if vram_history else 0
final_vram_res = vram_history[-1][1] if vram_history else 0

vram_growth = final_vram_res - base_vram_res
print(f"\n--- ANALISE ---")
print(f"Growth VRAM Reserved: {vram_growth:.2f}GB")
avg_time = sum(times_history) / len(times_history)
print(f"Tempo medio: {avg_time:.2f}s")

is_ok = True
if vram_growth > 1.0:
    print("[FAIL] Vazamento de memoria detectado (> 1GB de crescimento)")
    is_ok = False
else:
    print("[OK] VRAM estavel (GPU Allocator pooling natural ok)")

# Speed degradation
t_primeira = sum(times_history[:5])/5
t_ultima = sum(times_history[-5:])/5
degrad = (t_ultima - t_primeira) / t_primeira if t_primeira > 0 else 0

print(f"Speed Degradation: {degrad*100:.1f}%")
if degrad > 0.2:
    print("[WARN] Degradacao > 20% no final do batch")
else:
    print("[OK] Velocidade sustentada")

if is_ok:
    print("\n[TESTE F] PASS")
else:
    print("\n[TESTE F] FAIL")
print("=======================================")
