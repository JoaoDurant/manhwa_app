# manhwa_app/models/__init__.py
"""
Model Managers para gerenciar carregamento e memória da VRAM independentemente.
Evita carregar o mesmo modelo em paralelo ou trocar de modelo sem necessidade.
"""
import torch

# Ativar otimizações extremas para GPU Ampere/Ada (RTX 3000/4000/5000)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

from manhwa_app.models.qwen_manager import get_qwen_model
from manhwa_app.models.whisper_manager import get_whisper_model, unload_whisper, transcribe_audio

__all__ = [
    "get_qwen_model",
    "get_whisper_model",
    "unload_whisper",
    "transcribe_audio"
]
