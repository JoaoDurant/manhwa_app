"""
Teste C - Acoustic Quality
Verifica se o audio gerado sofre de degradacao, estalos ou low-volume.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import torch
import numpy as np
from pathlib import Path
import torchaudio as ta

OUTPUT_DIR = Path("tests/output/acoustic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VOICE_REF = str(Path("voices/Leonardo.wav").resolve())

import engine
print("Carregando engine...")
engine.load_multilingual()

text = "O guerreiro empurrou seus limites alem do impossivel."
wav_tensor, sr = engine.synthesize(
    text=text,
    audio_prompt_path=VOICE_REF,
    language="pt",
    temperature=0.65,
    exaggeration=0.65,
    cfg_weight=0.35,
    seed=42
)

# Move Tensor para numpy
if getattr(wav_tensor, "device", "cpu").type != "cpu":
    wav_tensor = wav_tensor.cpu()
audio_np = wav_tensor.numpy()

# Calcular métricas
max_amp = np.max(np.abs(audio_np))
rms = np.sqrt(np.mean(audio_np**2))
has_clipping = max_amp >= 0.99

# Presenca: % do tempo com energia acima do silencio
frame_size = sr // 10  # 100ms
energy = [np.sum(audio_np[i:i+frame_size]**2) for i in range(0, len(audio_np), frame_size)]
threshold = 0.001 * np.max(energy)
presence_ratio = sum(1 for e in energy if e > threshold) / len(energy)

print("=== ACOUSTIC QUALITY ===")
print(f"Max Amplitude: {max_amp:.4f}")
print(f"RMS LeveL: {rms:.4f}")
print(f"Presence Ratio: {presence_ratio*100:.1f}%")
print(f"Clipping: {'SIM' if has_clipping else 'NAO'}")

all_ok = True
if max_amp < 0.1:
    print("[FAIL] Audio muito baixo (ghost audio)")
    all_ok = False
elif max_amp > 0.99:
    print("[FAIL] Audio clipado")
    all_ok = False

if presence_ratio < 0.3:
    print("[FAIL] Excesso de silencio profundo no audio")
    all_ok = False

if all_ok:
    print("\n[TESTE C] PASS")
else:
    print("\n[TESTE C] FAIL")
print("========================")
