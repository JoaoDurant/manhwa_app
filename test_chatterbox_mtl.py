"""
Test Chatterbox Multilingual with eager attn fix.
Bypasses the UI config layer completely.
"""
import sys
import os
import logging
import time
from pathlib import Path
import soundfile as sf
import torch

# Force the venv root
sys.path.insert(0, str(Path(__file__).parent.resolve()))

logging.basicConfig(level=logging.WARNING)

# 1. Patch LLAMA_CONFIGS BEFORE engine imports T3
print("[TEST] Patching LLAMA_CONFIGS to use attn_implementation=eager...")
from chatterbox.models.t3.llama_configs import LLAMA_CONFIGS
for k in LLAMA_CONFIGS:
    LLAMA_CONFIGS[k]["attn_implementation"] = "eager"
print(f"[TEST] Configs patched: {list(LLAMA_CONFIGS.keys())}")

# 2. Import engine AFTER the patch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n[TEST] Dispositivo: {device}")

# 3. Load model
print("[TEST] Carregando ChatterboxMultilingualTTS...")
t0 = time.time()
model = ChatterboxMultilingualTTS.from_pretrained(device=device)
print(f"[TEST] Modelo carregado em {time.time()-t0:.2f}s")

# 4. Generate audio
print("[TEST] Gerando audio com eager attn...")
t1 = time.time()
with torch.inference_mode():
    wav = model.generate(
        text="Testing this generation right now. It should bypass SDPA natively.",
        language_id="en",
    )
elapsed = time.time() - t1
print(f"[TEST] Audio gerado! Shape: {wav.shape} | Tempo: {elapsed:.2f}s")

# 5. Save
out_path = "test_mtl_output.wav"
wav_np = wav.squeeze().cpu().numpy()
sf.write(out_path, wav_np, model.sr)
print(f"[TEST] Salvo em: {out_path}")
print("\n[TEST] SUCESSO! Chatterbox Multilingual funciona com eager attn + CUDA.")
