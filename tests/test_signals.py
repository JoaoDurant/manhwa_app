import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

app = QApplication.instance() or QApplication(sys.argv)

# --- AudioPipeline signals ---
from manhwa_app.audio_pipeline import AudioPipeline

required_audio_signals = {
    "paragraph_started":    (int, int, str),
    "paragraph_done_stats": (int, int, float, float, float, int),
    "paragraph_retry":      (int, int, str),
    "stage_complete":       (int, float),
}

# --- VideoPipeline signals ---
from manhwa_app.video_pipeline import VideoPipeline

required_video_signals = {
    "video_progress": (int, int),
    "video_complete": (str, float, float),
}

# --- MacroCoordinator signals ---
from manhwa_app.macro_core import MacroCoordinator

required_macro_signals = {
    "job_started":     (str, int, int),
    "job_complete":    (str, float, str),
    "job_failed":      (str, str),
    "queue_complete":  (float,),
}

# Verificação genérica
def check_signals(cls, required: dict, label: str):
    obj = cls.__new__(cls)
    for name, types in required.items():
        assert hasattr(cls, name), f"[FAIL] {label} missing signal: {name}"
        sig = getattr(obj, name)
        assert callable(sig.emit), f"[FAIL] {label}.{name} is not emittable"
        print(f"  [PASS] {label}.{name} ✓")

check_signals(AudioPipeline,    required_audio_signals, "AudioPipeline")
check_signals(VideoPipeline,    required_video_signals, "VideoPipeline")
check_signals(MacroCoordinator, required_macro_signals, "MacroCoordinator")

print("\n[ALL SIGNALS OK]")
