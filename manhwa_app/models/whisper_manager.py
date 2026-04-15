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

def get_whisper_model(model_name: str = "base", device_override: str = None):
    """
    Carrega o modelo faster-whisper de forma lazy e thread-safe.
    """
    global _whisper_model, _whisper_device, _whisper_model_name_loaded
    with _WHISPER_LOCK:
        if _whisper_model is not None and _whisper_model_name_loaded == model_name:
            if device_override is None or _whisper_device == device_override:
                return _whisper_model, _whisper_device

        # Descarregar se trocar de tamanho ou se dispositivo mudou drasticamente
        if _whisper_model is not None:
            del _whisper_model
            _whisper_model = None
            gc.collect()

        device = device_override if device_override else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Carregando faster-whisper '{model_name}' em {device}…")

        try:
            if _FASTER_WHISPER_AVAILABLE:
                compute_type = "int8_float16" if device == "cuda" else "int8"
                model = WhisperModel(model_name, device=device, compute_type=compute_type)
                _whisper_model = model
                _whisper_device = device
                _whisper_model_name_loaded = model_name
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
            error_msg = str(e)
            if "no kernel image" in error_msg or "sm_120" in error_msg or "sm_100" in error_msg:
                logger.warning(
                    f"INCOMPATIBILIDADE GPU DETECTADA NO WHISPER (RTX 50): {error_msg}. "
                    "Tentando carregamento em CPU..."
                )
                try:
                    if _FASTER_WHISPER_AVAILABLE:
                        model = WhisperModel(model_name, device="cpu", compute_type="int8")
                    elif _OPENAI_WHISPER_AVAILABLE:
                        model = whisper.load_model(model_name, device="cpu")
                    _whisper_model = model
                    _whisper_device = "cpu"
                    _whisper_model_name_loaded = model_name
                    logger.info(f"Whisper carregado em CPU com sucesso.")
                    return _whisper_model, _whisper_device
                except Exception as e_cpu:
                    logger.error(f"Falha total no Whisper (GPU e CPU): {e_cpu}")
                    return None, None
            else:
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


def transcribe_audio(wav_path: str, whisper_model_name: str = "base", device_override: str = None) -> str:
    """
    Transcreve um WAV preexistente utilizando o motor Whisper instanciado.
    Emprega uma amostragem nos primeiros 10s para verificação ultrarrápida (útil para TTS check).
    """
    model, device = get_whisper_model(whisper_model_name, device_override=device_override)
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
