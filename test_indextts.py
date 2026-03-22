# test_indextts.py — teste de instalacao e sintese do IndexTTS
# Executar APOS baixar o modelo em checkpoints/ e ter um WAV de referencia

import gc
import sys
import os
import numpy as np
import torch
from pathlib import Path

# Add current dir and index-tts to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "index-tts"))

# Verificar instalacao
try:
    from indextts.infer_v2 import IndexTTS2
    VERSION = "v2"
    print("[OK] IndexTTS2 (v2) importado com sucesso.")
except ImportError:
    try:
        from indextts.infer import IndexTTS as IndexTTS2
        VERSION = "v1"
        print("[OK] IndexTTS (v1/v1.5) importado com sucesso.")
    except ImportError:
        print("[ERRO] IndexTTS nao instalado!")
        print("Execute: git clone https://github.com/index-tts/index-tts.git && pip install -e .")
        sys.exit(1)

# Verificar modelo baixado
MODEL_DIR = "checkpoints"
CFG_PATH  = "checkpoints/config.yaml"
if not Path(MODEL_DIR).exists():
    print(f"[ERRO] Pasta de modelos nao encontrada: {MODEL_DIR}")
    print("Baixe com: python -c \"from huggingface_hub import snapshot_download; snapshot_download('IndexTeam/IndexTTS-2', local_dir='checkpoints')\"")
    # sys.exit(1) # Don't exit yet, maybe user hasn't downloaded but we want to see the error later

# Verificar prompt de referencia
PROMPT_WAV = None
for candidate in ["presets/", "test_data/", "./"]:
    wav_files = list(Path(candidate).glob("*.wav")) if Path(candidate).exists() else []
    if wav_files:
        PROMPT_WAV = str(wav_files[0])
        break

if PROMPT_WAV is None:
    print("[AVISO] Nenhum WAV de referencia encontrado em presets/, test_data/ ou ./")
    print("Coloque um WAV de 3-15 segundos em presets/ e execute novamente.")
    # sys.exit(1)

from utils import chunk_text_for_indextts, save_audio_to_file

TEXT_TEST = (
    "The system detects the presence of an intruder. Alarms activate throughout "
    "the facility. The protagonist moves quickly through the corridor, avoiding "
    "the security cameras with precision and speed."
)


def test_indextts():
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16   = (device == "cuda")
    use_kernel = (device == "cuda") and sys.platform != "win32"

    print(f"\nCarregando IndexTTS | version={VERSION} | device={device} | fp16={use_fp16}")
    try:
        if VERSION == "v2":
            tts = IndexTTS2(
                cfg_path=CFG_PATH,
                model_dir=MODEL_DIR,
                use_fp16=use_fp16,
                use_cuda_kernel=use_kernel,
                use_deepspeed=False,   # False no Windows
            )
        else:
            tts = IndexTTS2(
                model_dir=MODEL_DIR,
                cfg_path=CFG_PATH,
                is_fp16=use_fp16,
                use_cuda_kernel=use_kernel,
            )
        print("[OK] Modelo carregado!")
    except Exception as e:
        print(f"[FALHA] Erro ao carregar IndexTTS: {e}")
        return

    # Testar chunking
    chunks = chunk_text_for_indextts(TEXT_TEST)
    print(f"\nTexto dividido em {len(chunks)} chunk(s): {[len(c) for c in chunks]} chars")

    import tempfile, soundfile as sf
    parts = []
    sr_final = None

    if PROMPT_WAV:
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i+1}/{len(chunks)}: '{chunk[:70]}'")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp = f.name
            try:
                if VERSION == "v2":
                    tts.infer(spk_audio_prompt=PROMPT_WAV, text=chunk, output_path=tmp)
                else:
                    tts.infer(voice=PROMPT_WAV, text=chunk, output_path=tmp)
                audio, sr = sf.read(tmp, dtype="float32")
                if audio.ndim == 2:
                    audio = audio.mean(axis=1)
                parts.append(audio)
                sr_final = sr
                print(f"  [OK] {len(audio)} samples @ {sr} Hz ({len(audio)/sr:.2f}s)")
            except Exception as e:
                print(f"  [FALHA] Erro na sintese: {e}")
            finally:
                Path(tmp).unlink(missing_ok=True)

        if parts:
            final = np.concatenate(parts) if len(parts) > 1 else parts[0]
            save_audio_to_file(final, sr_final, "test_indextts_output.wav")
            print(f"\n[OK] Salvo: test_indextts_output.wav ({len(final)/sr_final:.2f}s total)")
    else:
        print("\n[PULADO] Sintese pulada por falta de audio de referencia.")

    del tts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[OK] VRAM liberada.")


if __name__ == "__main__":
    test_indextts()
