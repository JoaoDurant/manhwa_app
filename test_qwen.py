# test_qwen.py — versao corrigida
# Testa os 7 problemas corrigidos: import, attn, chunking, warmup, singleton

import gc
import contextlib
import numpy as np
import torch
import sys
import os

# Add current dir to path
sys.path.append(os.getcwd())

# Problema 1: import defensivo
try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    try:
        from qwen_tts.core.models.modeling_qwen3_tts import (
            Qwen3TTSForConditionalGeneration as Qwen3TTSModel,
        )
    except ImportError:
        raise RuntimeError(
            "Execute: pip install --upgrade qwen-tts\n"
            "Ou: pip install git+https://github.com/QwenLM/Qwen3-TTS.git"
        )

from utils import chunk_text_for_qwen, save_audio_to_file

# Texto longo para testar TDR (mesmo do test_qwen.py original)
TEXT_LONG = (
    "Testando a geracao do Qwen com texto longo para verificar se o "
    "chunking previne o TDR do Windows. Se o kernel da GPU demorar mais de "
    "2 segundos, o WDDM mata o processo silenciosamente. O chunking divide "
    "o texto em pedacos menores para que cada chamada seja rapida. "
    "Esta e a ultima frase do texto de teste para garantir cobertura total."
)


def test_qwen_corrigido():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype  = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )

    # Problema 2: attn correto
    try:
        import flash_attn  # noqa
        attn = "flash_attention_2"
        print("[OK] flash_attention_2 disponivel")
    except ImportError:
        attn = "eager"
        print("[AVISO] flash-attn ausente, usando 'eager' (OK, so mais lento)")

    print(f"\nCarregando | device={device} | dtype={dtype} | attn={attn}")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map=device,
        dtype=dtype,
        attn_implementation=attn,
    )
    print("[OK] Modelo carregado!")

    # Problema 6: warmup
    print("Warmup...")
    ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if torch.cuda.is_available() else contextlib.nullcontext()
    )
    with torch.inference_mode():
        with ctx:
            model.generate_custom_voice(text="Hello.", language="Auto", speaker="Ryan")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("[OK] Warmup concluido!")

    # Problema 4: chunking para texto longo
    chunks = chunk_text_for_qwen(TEXT_LONG)
    print(f"\nTexto ({len(TEXT_LONG)} chars) dividido em {len(chunks)} chunk(s)")
    print(f"Tamanhos: {[len(c) for c in chunks]}")

    parts = []
    sr_final = None
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}/{len(chunks)} ({len(chunk)} chars):")
        print(f"  '{chunk[:70]}...'")
        with torch.inference_mode():
            with ctx:
                res_wavs, sr = model.generate_custom_voice(
                    text=chunk, language="Auto", speaker="Ryan",
                )
        wav = res_wavs[0]
        wav_np = wav.cpu().float().numpy() if isinstance(wav, torch.Tensor) else np.array(wav, dtype=np.float32)
        parts.append(wav_np)
        sr_final = sr
        print(f"  [OK] {len(wav_np)} samples @ {sr} Hz ({len(wav_np)/sr:.2f}s)")

    final = np.concatenate(parts) if len(parts) > 1 else parts[0]
    save_audio_to_file(final, sr_final, "test_qwen_output.wav")
    print(f"\n[OK] Salvo: test_qwen_output.wav ({len(final)/sr_final:.2f}s total)")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[OK] VRAM liberada.")


if __name__ == "__main__":
    test_qwen_corrigido()
