# manhwa_app/models/qwen_manager.py
import logging
logger = logging.getLogger(__name__)

def get_qwen_model(task: str = "CustomVoice"):
    """
    Stub: O modelo Qwen agora roda em um worker HTTP isolado.
    O app principal nao deve importar 'qwen_tts' diretamente para evitar conflitos de NumPy.
    """
    logger.debug(f"get_qwen_model: Stub chamado para task {task}. O carregamento ocorre no worker.")
    return None

