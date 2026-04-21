import re
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Union

logger = logging.getLogger(__name__)

def natural_sort_key(s: Union[str, Path]) -> List[Union[int, str]]:
    """
    Chave de ordenação natural para strings e objetos Path.
    Garante que 'file10.txt' venha depois de 'file2.txt'.
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]

def _append_log(widget, message: str):
    """
    Helper para adicionar logs com timestamp a um QTextEdit (ou qualquer widget com .append()).
    """
    try:
        time_str = datetime.now().strftime("%H:%M:%S")
        widget.append(f"[{time_str}] {message}")
        if hasattr(widget, "verticalScrollBar"):
            sb = widget.verticalScrollBar()
            sb.setValue(sb.maximum())
    except Exception as e:
        logger.debug(f"Erro ao adicionar log no widget: {e}")

def get_safe_path(path: Union[str, Path], max_len: int = 240) -> Path:
    """
    Tenta encurtar o caminho se ele exceder o limite do Windows (MAX_PATH).
    """
    p = Path(path).resolve()
    if os.name == 'nt' and len(str(p)) > max_len:
        # Tenta usar o nome curto ou caminhos relativos se possível
        # Por enquanto, apenas avisamos ou sugerimos caminhos mais curtos
        logger.warning(f"Caminho potencialmente muito longo para Windows: {p}")
    return p
