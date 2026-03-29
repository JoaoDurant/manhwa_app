import torch
import time
import os
import sys

# Add project root to path
sys.path.insert(0, os.getcwd())

try:
    from kokoro import KPipeline
    print("Kokoro package found.")
except ImportError:
    print("Kokoro package NOT found.")
    sys.exit(1)

def benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Loading
    t0 = time.time()
    pipeline = KPipeline(lang_code='a', device=device)
    print(f"Load time: {time.time() - t0:.2f}s")
    
    text = "The quick brown fox jumps over the lazy dog. This is a performance benchmark for Kokoro TTS."
    voice = "af_heart"
    
    # 2. First run (Warmup + G2P + Inference)
    print("\n--- First Run (Warmup) ---")
    t0 = time.time()
    # Pipeline call is a generator
    generator = pipeline(text, voice=voice, speed=1.0)
    t_init = time.time() - t0
    
    t_start_gen = time.time()
    audio_parts = []
    for gs, ps, audio in generator:
        audio_parts.append(audio)
    t_total_gen = time.time() - t_start_gen
    
    print(f"Init time (G2P/Phonemizer): {t_init:.4f}s")
    print(f"Generation time (Inference): {t_total_gen:.4f}s")
    print(f"Total time: {time.time() - t0:.4f}s")
    
    # 3. Second run (Cached?)
    print("\n--- Second Run ---")
    t0 = time.time()
    generator = pipeline(text, voice=voice, speed=1.0)
    t_init = time.time() - t0
    
    t_start_gen = time.time()
    for gs, ps, audio in generator:
        pass
    t_total_gen = time.time() - t_start_gen
    
    print(f"Init time: {t_init:.4f}s")
    print(f"Generation time: {t_total_gen:.4f}s")
    print(f"Total time: {time.time() - t0:.4f}s")

    # 4. Check model dtype
    if hasattr(pipeline, 'model'):
        p = next(pipeline.model.parameters())
        print(f"\nModel Dtype: {p.dtype}")
        print(f"Model Device: {p.device}")

if __name__ == "__main__":
    benchmark()
