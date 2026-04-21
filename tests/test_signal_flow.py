import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unittest.mock import MagicMock, patch
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

app = QApplication.instance() or QApplication(sys.argv)

received = {}

# Mock MacroTab e conecta sinais
with patch("engine.load_model", return_value=MagicMock()):
    from manhwa_app.audio_pipeline import AudioPipeline

    pipeline = AudioPipeline.__new__(AudioPipeline)
    # Inicializa apenas os sinais, sem carregar modelo
    super(AudioPipeline, pipeline).__init__()

    def on_para_started(idx, total, preview):
        received["para_started"] = (idx, total, preview)

    def on_para_done(idx, total, elapsed, sim, rms, attempts):
        received["para_done"] = (idx, total, elapsed, sim, rms, attempts)

    def on_para_retry(idx, attempt, reason):
        received["para_retry"] = (idx, attempt, reason)

    pipeline.paragraph_started.connect(on_para_started)
    pipeline.paragraph_done_stats.connect(on_para_done)
    pipeline.paragraph_retry.connect(on_para_retry)

    # Emite sinais manualmente
    pipeline.paragraph_started.emit(1, 47, "Mas então o rei declarou...")
    pipeline.paragraph_done_stats.emit(1, 47, 4.3, 0.94, 0.28, 1)
    pipeline.paragraph_retry.emit(2, 2, "similarity 0.71 < 0.82")

    # Processa eventos pendentes
    app.processEvents()

assert received.get("para_started") == (1, 47, "Mas então o rei declarou..."), \
    f"[FAIL] paragraph_started não recebido corretamente: {received.get('para_started')}"

assert received.get("para_done") == (1, 47, 4.3, 0.94, 0.28, 1), \
    f"[FAIL] paragraph_done_stats não recebido: {received.get('para_done')}"

assert received.get("para_retry") == (2, 2, "similarity 0.71 < 0.82"), \
    f"[FAIL] paragraph_retry não recebido: {received.get('para_retry')}"

print("[ALL SIGNAL FLOWS OK]")
