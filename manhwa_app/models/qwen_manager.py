# manhwa_app/models/qwen_manager.py
import gc
import logging
import threading
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Otimização de suporte bfloat16 sem crashar o CUDA context dentro de thread
try:
    _BF16_SUPPORTED = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
except Exception:
    _BF16_SUPPORTED = False

try:
    from qwen_tts import Qwen3TTSModel
    _QWEN_AVAILABLE = True
except ImportError:
    _QWEN_AVAILABLE = False
    logger.warning("qwen_tts não instalado no sistema.")

# ---------------------------------------------------------------------------
# Singleton do Qwen3-TTS — carregamento lazy, bfloat16 + flash_attention 2
# ---------------------------------------------------------------------------
_qwen_model = None
_qwen_task_loaded: Optional[str] = None
_QWEN_LOCK = threading.Lock()

def get_qwen_model(task: str = "CustomVoice"):
    """
    Carrega o modelo Qwen3-TTS (CustomVoice, VoiceDesign, Base/VoiceClone) de forma
    lazy e thread-safe. Assegura que dois modelos de tasks diferentes não sejam carregados ao mesmo tempo,
    estourando a memória VRAM. Utiliza bfloat16 em arquiteturas compatíveis (Amper/Ada+).
    """
    global _qwen_model, _qwen_task_loaded
    with _QWEN_LOCK:
        # Se o modelo já está carregado para essa task exata, apenas retorna a instância pronta
        if _qwen_model is not None and _qwen_task_loaded == task:
            return _qwen_model

        # Se mudou a task ou se a task era None, descarrega qualquer modelo preso na memória VRAM
        if _qwen_model is not None:
            del _qwen_model
            _qwen_model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(f"Carregando Qwen3-TTS ({task}) em CUDA...")
        try:
            if not _QWEN_AVAILABLE:
                logger.error("qwen_tts package is not available!")
                return None
            
            model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
            if task == "VoiceDesign":
                model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
            elif task == "VoiceClone":
                model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

            # Aproveita as instruções bfloat16 do BF16 do núcleo tensor onde suportado
            # FP16 pode engasgar ou recorrer a FP32 para operações SDPA
            _qwen_model = Qwen3TTSModel.from_pretrained(
                model_id,
                device_map="cuda:0" if torch.cuda.is_available() else "cpu",
                dtype=torch.bfloat16 if _BF16_SUPPORTED else torch.float16,
                attn_implementation="sdpa", # SDPA otimizado, muito mais rápido em torch 2.1+
            )
            _qwen_task_loaded = task
            logger.info(f"Qwen3-TTS '{model_id}' pronto.")
            return _qwen_model
        except Exception as e:
            logger.error(f"Falha ao carregar Qwen3-TTS: {e}", exc_info=True)
            return None
