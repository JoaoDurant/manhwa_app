# manhwa_app/models/whisper_manager.py
import gc
import logging
import threading
from typing import Optional

import torch

logger = logging.getLogger(__name__)

try:
    from faster_whisper import WhisperModel
    _FASTER_WHISPER_AVAILABLE = True
except ImportError:
    _FASTER_WHISPER_AVAILABLE = False
    logger.warning("faster-whisper não instalado. Tentaremos fallback.")

try:
    import whisper
    _OPENAI_WHISPER_AVAILABLE = True
except ImportError:
    _OPENAI_WHISPER_AVAILABLE = False

# ---------------------------------------------------------------------------
# Singleton do Whisper — carregamento lazy, CUDA + float16 + torch.compile opcional
# ---------------------------------------------------------------------------
_whisper_model = None
_whisper_model_name_loaded: Optional[str] = None
_whisper_device: Optional[str] = None
_WHISPER_LOCK = threading.Lock()

def get_whisper_model(model_name: str = "base"):
    """
    Carrega o modelo faster-whisper de forma lazy e thread-safe.
    Isso evita dupla inicialização caso haja multiprocessamento ou múltiplas
    threads requisitando o validador de Whisper.
    """
    global _whisper_model, _whisper_device, _whisper_model_name_loaded
    with _WHISPER_LOCK:
        # Verifica novamente dentro do lock (double-checked locking)
        if _whisper_model is not None and _whisper_model_name_loaded == model_name:
            return _whisper_model, _whisper_device

        # Descarregar modelo anterior se trocar de tamanho
        if _whisper_model is not None:
            del _whisper_model
            _whisper_model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Carregando faster-whisper '{model_name}' em {device}…")

        try:
            if _FASTER_WHISPER_AVAILABLE:
                compute_type = "int8_float16" if device == "cuda" else "int8"
                model = WhisperModel(model_name, device=device, compute_type=compute_type)
                _whisper_model = model
                _whisper_device = device
                _whisper_model_name_loaded = model_name
                logger.info(f"faster-whisper '{model_name}' pronto em {device} (compute: {compute_type}).")
                return _whisper_model, _whisper_device
            elif _OPENAI_WHISPER_AVAILABLE:
                logger.warning("Usando fallback para openai-whisper.")
                model = whisper.load_model(model_name, device=device)
                _whisper_model = model
                _whisper_device = device
                _whisper_model_name_loaded = model_name
                return _whisper_model, _whisper_device
            else:
                logger.error("Nenhuma biblioteca Whisper disponível.")
                return None, None
        except Exception as e:
            logger.error(f"Falha ao carregar Whisper: {e}", exc_info=True)
            return None, None


def unload_whisper():
    """
    Descarrega o Whisper e libera a memória GPU/CPU alocada.
    """
    global _whisper_model, _whisper_device, _whisper_model_name_loaded
    with _WHISPER_LOCK:
        if _whisper_model is not None:
            del _whisper_model
            _whisper_model = None
            _whisper_device = None
            _whisper_model_name_loaded = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Modelo Whisper descarregado.")


def transcribe_audio(wav_path: str, whisper_model_name: str = "base") -> str:
    """
    Transcreve um WAV preexistente utilizando o motor Whisper instanciado.
    Emprega uma amostragem nos primeiros 10s para verificação ultrarrápida (útil para TTS check).
    """
    model, device = get_whisper_model(whisper_model_name)
    if model is None:
        return ""
    try:
        # Tentar API abstrata do faster-whisper primeiro
        if hasattr(model, 'transcribe') and hasattr(model, 'supported_languages'):
            # faster-whisper: transcrevemos apenas os primeiros 10 segundos
            segments, _ = model.transcribe(
                wav_path,
                beam_size=1,
                language=None,
                clip_timestamps="0,10",  # Amostragem: primeiros 10s
                vad_filter=True,
            )
            return " ".join(s.text for s in segments).strip()
        else:
            # openai-whisper fallback
            result = model.transcribe(wav_path, fp16=(device == "cuda"))
            return result.get("text", "").strip()
    except Exception as e:
        logger.error(f"Transcrição Whisper falhou para {wav_path}: {e}", exc_info=True)
        return ""
