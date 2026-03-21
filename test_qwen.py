import os
import sys
import gc
import torch

from qwen_tts import Qwen3TTSModel

def test_qwen():
    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    
    print(f"Carregando {model_id} em {device} com {dtype}...")
    model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map=device,
        dtype=dtype,
        attn_implementation="sdpa",
    )
    print("Modelo carregado!")
    
    with torch.inference_mode():
        try:
            print("Gerando áudio...")
            res_wavs, sr = model.generate_custom_voice(
                text="Testando a geração silenciosa do Qwen com um texto muito mais longo para tentar causar um timeout de TDR no Windows. Se o kernel da GPU demorar mais de 2 segundos para executar, o Windows WDDM irá matar o processo CUDA silenciosamente, causando exatamente o erro que estamos observando no aplicativo principal. Por isso, este texto tem que ser bem longo, com várias e várias palavras, para atrasar a geração o máximo possível. Vamos adicionar mais uma frase só para garantir. E outra frase também.",
                language="Auto",
                speaker="Ryan",
            )
            print("Sucesso! sr =", sr)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("Erro durante a geração:", e)

if __name__ == "__main__":
    test_qwen()
