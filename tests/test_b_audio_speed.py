"""
Teste B — Benchmark de Velocidade de Geracao TTS
Mede RTF (Real-Time Factor) para textos short/medium/long.
Detecta warm-up penalty e valida que torchaudio funciona corretamente.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import torch
import gc
from pathlib import Path

OUTPUT_DIR = Path("tests/output/speed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Encontra arquivo de voz de referencia disponivel
VOICE_DIR = Path("voices")
VOICE_REF = None
for candidate in ["Leonardo.wav", "Gabriel.wav", "Adrian.wav"]:
    p = VOICE_DIR / candidate
    if p.exists():
        VOICE_REF = str(p)
        break
if VOICE_REF is None:
    for wav in VOICE_DIR.glob("*.wav"):
        VOICE_REF = str(wav)
        break

print(f"Usando voz de referencia: {VOICE_REF}")

# Verifica torchaudio
try:
    import torchaudio as ta
    print(f"torchaudio: {ta.__version__}")
except ImportError:
    print("[FAIL] torchaudio nao instalado")
    sys.exit(1)

# Carrega engine
print("\nCarregando modelo Chatterbox Multilingual...")
import engine
t_load = time.perf_counter()
ok = engine.load_multilingual()
t_load_end = time.perf_counter()
if not ok:
    print("[FAIL] Falha ao carregar modelo multilingual")
    sys.exit(1)
print(f"Modelo carregado em {t_load_end - t_load:.1f}s")
print(f"Device: {engine.model_device} | Tipo: {engine.loaded_model_type}")
if torch.cuda.is_available():
    vram = torch.cuda.memory_reserved(0) / 1e9
    print(f"VRAM apos carga: {vram:.2f} GB")

texts = [
    ("short",  "Ele nao deveria ter feito isso."),
    ("medium", "O guerreiro empurrou seus limites alem do impossivel, canalizando toda a sua energia vital para um unico golpe devastador."),
    ("long",   "Ha mil anos, quando os deuses ainda caminhavam entre os mortais, um pacto foi selado com o sangue dos tres reis originais. Esse pacto garantia que nenhuma forca do alem poderia cruzar o veu que separava os mundos."),
]

results = {}
TARGETS_RTF = {"short": 1.0, "medium": 1.0, "long": 1.0}
TARGETS_S   = {"short": 3.0, "medium": 6.0, "long": 12.0}

print("\n=== SPEED BENCHMARK ===")

for label, text in texts:
    times_list = []
    wav_tensor = None

    for run in range(3):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        wav_tensor, sr = engine.synthesize(
            text=text,
            audio_prompt_path=VOICE_REF,
            language="pt",
            temperature=0.65,
            exaggeration=0.65,
            cfg_weight=0.35,
            seed=3000 + run,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times_list.append(elapsed)

        if wav_tensor is not None:
            out_path = OUTPUT_DIR / f"{label}_run{run+1}.wav"
            wav_save = wav_tensor.cpu()
            if wav_save.dim() == 1:
                wav_save = wav_save.unsqueeze(0)
            ta.save(str(out_path), wav_save, sr or 24000)

        # GC entre runs
        gc.collect()

    if wav_tensor is None:
        print(f"[{label.upper()}] FAIL - nao foi possivel gerar audio")
        continue

    avg = sum(times_list) / len(times_list)
    audio_s = wav_tensor.shape[-1] / (sr or 24000)
    rtf = avg / audio_s if audio_s > 0 else 999
    chars_per_s = len(text) / avg

    results[label] = {
        "avg_s": avg, "times": times_list,
        "chars_per_s": chars_per_s,
        "audio_duration_s": audio_s, "rtf": rtf,
    }

    rtf_ok = rtf <= TARGETS_RTF[label]
    spd_ok = avg <= TARGETS_S[label]
    warmup = times_list[0] - min(times_list[1:]) if len(times_list) > 1 else 0

    status = "[OK]" if (rtf_ok and spd_ok) else "[WARN]"
    print(f"\n[{label.upper()}] len={len(text)} chars | audio={audio_s:.2f}s")
    print(f"  {status} Runs: {[f'{t:.2f}s' for t in times_list]}")
    print(f"  Media: {avg:.2f}s | RTF: {rtf:.3f} ({'faster-than-realtime' if rtf < 1 else 'SLOWER-than-realtime'})")
    print(f"  chars/s: {chars_per_s:.1f}")
    if warmup > 2.0:
        print(f"  [WARN] Warm-up penalty: +{warmup:.1f}s no run1")

# Sumario
print("\n=== SPEED BENCHMARK SUMMARY ===")
all_pass = True
for label, r in results.items():
    rtf_pass = r["rtf"] <= TARGETS_RTF.get(label, 1.0)
    spd_pass = r["avg_s"] <= TARGETS_S.get(label, 10.0)
    st = "[PASS]" if (rtf_pass and spd_pass) else "[FAIL]"
    if not (rtf_pass and spd_pass):
        all_pass = False
    print(f"  {st} {label}: avg={r['avg_s']:.2f}s RTF={r['rtf']:.3f}")

if all_pass:
    print("\n[TESTE B] PASS")
else:
    print("\n[TESTE B] FAIL - ver detalhes acima")
print("=== SPEED BENCHMARK COMPLETE ===")
