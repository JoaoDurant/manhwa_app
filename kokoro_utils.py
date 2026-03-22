# File: kokoro_utils.py
import os
import sys
import torch
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Ativar Modo Offline para evitar esperas de rede/download desnecessarias
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_kokoro_model = None
_kokoro_device = None

def generate_audio_kokoro(text: str, voice_path: str, device: str = 'auto') -> tuple:
    """
    Interface simplificada para o motor Kokoro-TTS-Local.
    Retorna (wav_np, sample_rate).
    """
    global _kokoro_model, _kokoro_device
    
    try:
        kokoro_root = Path(__file__).parent / "Kokoro-TTS-Local-master"
        if str(kokoro_root) not in sys.path:
            sys.path.append(str(kokoro_root))
            
        from models import build_model, generate_speech
        
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        if _kokoro_model is None or _kokoro_device != device:
            model_path = kokoro_root / "kokoro-v1_0.pth"
            logger.info(f"Carregando modelo Kokoro em {device}...")
            try:
                _kokoro_model = build_model(str(model_path), device)
                _kokoro_device = device
            except Exception as e:
                err_msg = str(e).lower()
                if "no kernel image" in err_msg or "sm_120" in err_msg or "sm_100" in err_msg:
                    logger.warning(f"INCOMPATIBILIDADE GPU DETECTADA NO KOKORO (RTX 50): {e}. Usando fallback para CPU...")
                    _kokoro_model = build_model(str(model_path), 'cpu')
                    _kokoro_device = 'cpu'
                else:
                    raise e
            
        # O Kokoro espera o path da voz como string
        # Se voice_path for apenas o nome da voz (ex: 'af_bella'), expandir para o path real
        if not os.path.exists(voice_path):
             alt_path = kokoro_root / "voices" / f"{voice_path}.pt"
             if alt_path.exists():
                 voice_path = str(alt_path)
             else:
                 logger.error(f"Voz Kokoro nao encontrada: {voice_path}")
                 return None, None

        # Gerar
        # generator = model(text, voice=str(voice_path), speed=1.0, split_pattern=r'\n+')
        generator = _kokoro_model(text, voice=str(voice_path), speed=1.0)
        
        all_audio = []
        for _, _, audio in generator:
            if audio is not None:
                if isinstance(audio, torch.Tensor):
                    audio = audio.cpu().numpy()
                all_audio.append(audio)
        
        if not all_audio:
            return None, None
            
        final_audio = np.concatenate(all_audio)
        return final_audio, 24000
        
    except Exception as e:
        logger.error(f"Erro no kokoro_utils: {e}", exc_info=True)
        return None, None
