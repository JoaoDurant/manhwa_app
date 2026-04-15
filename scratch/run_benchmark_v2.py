# scratch/run_benchmark_v2.py
import sys
import os
import time
import torch
from pathlib import Path

# Adiciona o diretório raiz ao sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import engine
from manhwa_app.models.whisper_manager import get_whisper_model

def run_benchmark():
    # Evitando emojis para compatibilidade com console Windows CP1252
    print("--- Iniciando Benchmark de Otimizacao (v2) - RTX 5070 Ti ---")
    
    if not torch.cuda.is_available():
        print("WAINING: CUDA nao detectado. Rodando em CPU.")
    else:
        print(f"Device: {torch.cuda.get_device_name(0)}")
        
    print(f"DType Padrao (Engine): {getattr(engine, 'DEFAULT_DTYPE', 'float32')}")
    
    # 2. Carregar Modelo
    print("\n[STEP] Carregando Chatterbox Turbo...")
    t0 = time.time()
    if not engine.load_model("turbo"):
        print("FAIL: Falha ao carregar modelo.")
        return
    print(f"OK: Carregado em {time.time() - t0:.2f}s")
    
    # 3. Teste de Síntese e Cache de Embedding
    text = "Isso e um teste de performance com cache de embedding e precisao adaptativa."
    
    # Tenta achar um audio de voz real para o teste de embedding
    voice = None
    voices_dir = _ROOT / "voices"
    if voices_dir.exists():
        wavs = list(voices_dir.glob("*.wav"))
        if wavs:
            voice = str(wavs[0])
            print(f"Using voice reference: {wavs[0].name}")

    print("\n[TEST] Teste 1: Geracao Inicial (Cold Start + Embedding Computation)")
    t1 = time.time()
    wav, sr = engine.synthesize(text, audio_prompt_path=voice)
    d1 = time.time() - t1
    print(f"Tempo 1: {d1:.2f}s")
    
    if wav is None:
        print("FAIL: Geracao falhou.")
        return

    print("\n[TEST] Teste 2: Geracao com Cache de Embedding (Warm Start)")
    t2 = time.time()
    wav, sr = engine.synthesize(text, audio_prompt_path=voice)
    d2 = time.time() - t2
    print(f"Tempo 2: {d2:.2f}s")
    
    improvement = ((d1 - d2) / d1) * 100 if d1 > 0 else 0
    print(f"\nINFO: Ganho com Cache de Embedding: {improvement:.1f}%")
    
    # 4. Verificar Whisper em CPU
    print("\n[STEP] Verificando Whisper (Offload)")
    t3 = time.time()
    model, device = get_whisper_model("base", device_override="cpu")
    print(f"OK: Whisper carregado em: {device} (Tempo: {time.time() - t3:.2f}s)")
    
    print("\n--- Benchmark concluido com sucesso! ---")

if __name__ == "__main__":
    run_benchmark()
