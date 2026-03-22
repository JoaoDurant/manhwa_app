import os
import sys
import torch
import soundfile as sf
from pathlib import Path

# Add current dir to path
sys.path.append(os.getcwd())

import engine

def test_multilingual():
    print("=== Teste Chatterbox Multilingual (Stability Check) ===")
    
    # Repositório oficial do multilingual
    repo_id = "JoaoDurant/chatterbox-multilingual" 
    
    # 1. Forçar carregamento do Multilingual
    print("\n1. Carregando Multilingual via load_model em engine.py...")
    # Mockando config para usar o multilingual
    engine.config_manager.update_and_save({"model": {"repo_id": repo_id}})
    
    # Reset model load state to force reload
    engine.chatterbox_model = None
    engine.MODEL_LOADED = False
    
    success = engine.load_model()
    if not success:
        print("FALHA: Não foi possível carregar o modelo Multilingual.")
        return

    print(f"Modelo carregado: {engine.loaded_model_class_name} ({engine.loaded_model_type})")
    print(f"Idiomas Suportados: {engine.SUPPORTED_LANGUAGES}")

    # 2. Testar Síntese em Português
    text = "Olá, testando o motor multilíngue do Chatterbox em alta precisão float trinta e dois."
    print(f"\n2. Sintetizando texto: '{text}'")
    
    wav_tensor, sr = engine.synthesize(
        text=text,
        language="pt",
        temperature=0.7,
        exaggeration=0.0
    )
    
    if wav_tensor is not None:
        print(f"SUCESSO! sr = {sr}, shape = {wav_tensor.shape}")
        out_path = "test_multilingual_output.wav"
        data = wav_tensor.squeeze(0).cpu().numpy()
        sf.write(out_path, data, sr)
        print(f"Arquivo salvo em: {out_path}")
    else:
        print("FALHA: Síntese retornou None.")

if __name__ == "__main__":
    test_multilingual()
