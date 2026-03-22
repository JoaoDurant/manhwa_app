import os
import sys
import torch
import numpy as np
import soundfile as sf
from pathlib import Path

# Add current dir to path
sys.path.append(os.getcwd())

import engine
import utils

def test_integrated_qwen():
    print("=== Teste Integrado Qwen3-TTS (Singleton + Chunking) ===")
    
    text = (
        "Este é um parágrafo longo para testar o sistema de chunking do Qwen. "
        "O objetivo é garantir que frases longas não causem TDR no Windows. "
        "Estamos dividindo o texto em pedaços de 500 caracteres e concatenando o áudio final. "
        "Isso mantém a fluidez e evita que o kernel da GPU bloqueie o sistema operacional. "
        "Vamos adicionar mais texto para ultrapassar o limite de 500 caracteres propositalmente. "
        "A raposa rápida dá um pulo sobre o cachorro preguiçoso. " * 3
    )
    
    print(f"Texto de teste ({len(text)} chars): {text[:100]}...")
    
    # 1. Carregar Modelo
    print("\n1. Carregando modelo via singleton...")
    success = engine.load_qwen_model()
    if not success:
        print("FALHA: Não foi possível carregar o modelo Qwen.")
        return

    # 2. Testar Geração por Parágrafo (com chunking interno)
    print("\n2. Gerando áudio por parágrafo (utils.generate_paragraph_audio_qwen)...")
    audio, sr = utils.generate_paragraph_audio_qwen(
        text=text,
        speaker="Ryan",
        language="Auto"
    )
    
    if audio is not None:
        print(f"SUCESSO! Áudio gerado: {len(audio)} samples, {sr} Hz")
        out_path = "test_qwen_integrated_output.wav"
        sf.write(out_path, audio, sr)
        print(f"Arquivo salvo em: {out_path}")
    else:
        print("FALHA: Áudio não foi gerado.")

    # 3. Testar Unload/Reload (VRAM Management)
    print("\n3. Testando descarregamento de VRAM...")
    engine.unload_qwen_model()
    print("Modelo descarregado.")
    
    print("\n=== Teste Concluído! ===")

if __name__ == "__main__":
    test_integrated_qwen()
