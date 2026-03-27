# manhwa_app/app.py
# Aplicação PySide6 principal do Manhwa Video Creator.
# UI dark-themed com 3 abas: Geração de Áudio, Imagens, Geração de Vídeo.

import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

# --- OTIMIZAÇÃO CUDA RTX (5070 Ti / Blackwell / Ada) ---
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

from PySide6.QtCore import (
    QObject, Qt, QThread, QUrl, Signal, Slot,
)
from PySide6.QtGui import (
    QColor, QDragEnterEvent, QDropEvent, QFont, QPixmap,
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPushButton, QProgressBar,
    QScrollArea, QSizePolicy, QStatusBar, QTabWidget, QTextEdit,
    QVBoxLayout, QWidget, QSlider, QComboBox, QDoubleSpinBox, QSpinBox,
    QCheckBox, QInputDialog, QDialog, QTableWidget, QHeaderView,
    QDialogButtonBox, QTableWidgetItem,
)

from manhwa_app.audio_pipeline import split_into_paragraphs
from manhwa_app.video_pipeline import EFFECTS

# Using config natively instead of heavy audio pipeline
from config import config_manager as _config_manager
from utils import resolve_voice_path

# Import protegido — o painel Gemini fica desativado se a lib não estiver instalada
try:
    from google import genai as _genai_check  # noqa: F401
    from gemini_processor import GeminiProcessor
    _GEMINI_AVAILABLE = True
except ImportError:
    GeminiProcessor = None
    _GEMINI_AVAILABLE = False

logger = logging.getLogger(__name__)

def _append_log(widget: QTextEdit, message: str):
    """Helper to append timestamped logs to a QTextEdit."""
    from datetime import datetime
    time_str = datetime.now().strftime("%H:%M:%S")
    widget.append(f"[{time_str}] {message}")
    widget.verticalScrollBar().setValue(widget.verticalScrollBar().maximum())


# ---------------------------------------------------------------------------
# Worker thread do Gemini (padrão QThread do projeto)
# ---------------------------------------------------------------------------
import threading as _threading


class GeminiWorker(QThread):
    """Runs Gemini tasks (process content or list models) in a background thread."""

    progress = Signal(int, int, str)
    finished = Signal(object)          # Can be dict for process or list for model listing
    error = Signal(str)

    def __init__(self, *args, **kwargs):
        """
        Flexible init:
        Task Process: (txt_path, api_key, languages, delay_seconds, ...)
        Task List: (api_key, task="list_models")
        """
        super().__init__()
        # Se o segundo argumento for "list_models", é a tarefa de listagem
        if len(args) >= 2 and args[1] == "list_models":
            self.task = "list_models"
            self.api_key = args[0]
        else:
            self.task = "process"
            # Mapear argumentos posicionais se fornecidos, senão usar kwargs
            if len(args) >= 4:
                self.txt_path = args[0]
                self.api_key = args[1]
                self.languages = args[2]
                self.delay_seconds = args[3]
            else:
                self.txt_path = kwargs.get("txt_path")
                self.api_key = kwargs.get("api_key")
                self.languages = kwargs.get("languages")
                self.delay_seconds = kwargs.get("delay_seconds")
            
            self.model_name = kwargs.get("model_name", "gemini-1.5-flash")
            self.revision_prompt = kwargs.get("revision_prompt")
            self.translation_prompt = kwargs.get("translation_prompt")
            self.chunk_size = kwargs.get("chunk_size", 12)
            self.overlap = kwargs.get("overlap", 2)
            self.thinking_level = kwargs.get("thinking_level", "high")
            self.media_resolution = kwargs.get("media_resolution", "media_resolution_high")
            self._stop_event = _threading.Event()

    def cancel(self):
        if hasattr(self, "_stop_event"):
            self._stop_event.set()

    def run(self):
        try:
            from google import genai as _genai
            client = _genai.Client(api_key=self.api_key)

            if self.task == "list_models":
                models = []
                for m in client.models.list():
                    models.append(m.name)
                self.finished.emit(models)
                return

            if GeminiProcessor is None:
                self.error.emit("GeminiProcessor não disponível.")
                return

            processor = GeminiProcessor(model_name=self.model_name)
            result = processor.process(
                txt_path=self.txt_path,
                api_key=self.api_key,
                languages=self.languages,
                delay_seconds=self.delay_seconds,
                revision_prompt=self.revision_prompt,
                translation_prompt=self.translation_prompt,
                chunk_size=self.chunk_size,
                overlap=self.overlap,
                thinking_level=self.thinking_level,
                media_resolution=self.media_resolution,
                progress_callback=lambda a, t, m: self.progress.emit(a, t, m),
                stop_event=self._stop_event,
            )
            self.finished.emit(result)
        except Exception as exc:
            logger.error(f"GeminiWorker erro ({self.task}): {exc}", exc_info=True)
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Thread para carregamento de modelos em background
# ---------------------------------------------------------------------------
class ModelLoaderThread(QThread):
    finished_loading = Signal(bool, str)

    def __init__(self, tts_engine: str, model_type: str):
        super().__init__()
        self.tts_engine = tts_engine
        self.model_type = model_type

    def run(self):
        try:
            from manhwa_app.audio_pipeline import _engine, _ENGINE_AVAILABLE
            if not _ENGINE_AVAILABLE or _engine is None:
                self.finished_loading.emit(False, "Engine não disponível")
                return

            # Para Chatterbox, o engine real é o subtipo (turbo, multilingual, original)
            target = self.tts_engine
            if self.tts_engine == "chatterbox":
                target = self.model_type

            logger.info(f"ModelLoaderThread: trocando para {target}")
            
            if _engine.switch_to_engine(target):
                name_map = {
                    "turbo": "Chatterbox Turbo",
                    "multilingual": "Chatterbox Multilingual",
                    "original": "Chatterbox Original",
                    "qwen": "Qwen3-TTS",
                    "indextts": "IndexTTS",
                    "kokoro": "Kokoro TTS"
                }
                display_name = name_map.get(target, target.capitalize())
                
                # Detect the actual device used (could be CPU fallback)
                actual_device = "GPU"
                try:
                    if hasattr(_engine, "engine") and _engine.engine and hasattr(_engine.engine, "device"):
                         if str(_engine.engine.device) == "cpu":
                             actual_device = "CPU (Fallback)"
                except: pass
                
                self.finished_loading.emit(True, f"{display_name} [{actual_device}]")
            else:
                self.finished_loading.emit(False, f"Falha ao carregar {target}")
                
        except Exception as e:
            logger.error(f"Erro no ModelLoaderThread: {e}", exc_info=True)
            self.finished_loading.emit(False, str(e))

# ---------------------------------------------------------------------------
# Configurações de sessão (persistência entre execuções)
# ---------------------------------------------------------------------------
SESSION_FILE = Path(__file__).resolve().parent.parent / "session_config.json"

def _load_session() -> dict:
    try:
        if SESSION_FILE.exists():
            return json.loads(SESSION_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def _save_session(data: dict):
    try:
        SESSION_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Não foi possível salvar sessão: {e}")

# ---------------------------------------------------------------------------
# Sistema de Temas
# ---------------------------------------------------------------------------

THEMES = {
    "\U0001f311 Dark (Padr\u00e3o)": {
        "bg":"#0d0d12","surface":"#13131e","surface2":"#1e1e2e",
        "border":"rgba(255,255,255,0.07)","border2":"rgba(255,255,255,0.13)",
        "accent":"#3b82f6","accent2":"#1d4ed8","accent3":"#60a5fa",
        "text":"#e2e8f0","subtext":"#94a3b8","dim":"#64748b",
        "tab_bg":"#181826","tab_sel":"#1e2a4a","header_bg":"#0a0a10",
        "danger":"#7f1d1d","danger_txt":"#fca5a5",
        "success":"#14532d","success_txt":"#86efac","type":"dark",
    },
    "\U0001f9db Dracula": {
        "bg":"#1e1f29","surface":"#282a36","surface2":"#44475a",
        "border":"rgba(98,114,164,0.30)","border2":"rgba(139,233,253,0.20)",
        "accent":"#bd93f9","accent2":"#6272a4","accent3":"#ff79c6",
        "text":"#f8f8f2","subtext":"#8be9fd","dim":"#6272a4",
        "tab_bg":"#282a36","tab_sel":"#44475a","header_bg":"#191a23",
        "danger":"#ff5555","danger_txt":"#ffb8b8",
        "success":"#50fa7b","success_txt":"#e6ffe6","type":"dark",
    },
    "\U0001f338 Sakura (Japon\u00eas)": {
        "bg":"#160b10","surface":"#1f0f16","surface2":"#2e1520",
        "border":"rgba(232,120,154,0.18)","border2":"rgba(232,120,154,0.35)",
        "accent":"#e8789a","accent2":"#9b2d50","accent3":"#f4aac0",
        "text":"#f5e8ec","subtext":"#c89aaa","dim":"#7a4a58",
        "tab_bg":"#1c0e14","tab_sel":"#2e1520","header_bg":"#0d0608",
        "danger":"#8b1a32","danger_txt":"#ffb8c8",
        "success":"#2d6040","success_txt":"#a8f0c0","type":"dark",
    },
    "\u2744 Nord": {
        "bg":"#2e3440","surface":"#3b4252","surface2":"#434c5e",
        "border":"rgba(216,222,233,0.10)","border2":"rgba(136,192,208,0.25)",
        "accent":"#88c0d0","accent2":"#5e81ac","accent3":"#81a1c1",
        "text":"#eceff4","subtext":"#d8dee9","dim":"#4c566a",
        "tab_bg":"#3b4252","tab_sel":"#434c5e","header_bg":"#242932",
        "danger":"#bf616a","danger_txt":"#ffd0d4",
        "success":"#a3be8c","success_txt":"#e0ffe0","type":"dark",
    },
    "\U0001f30a Deep Ocean": {
        "bg":"#060d18","surface":"#0b1526","surface2":"#0f2040",
        "border":"rgba(0,180,255,0.12)","border2":"rgba(0,200,255,0.25)",
        "accent":"#00b4d8","accent2":"#0077b6","accent3":"#90e0ef",
        "text":"#caf0f8","subtext":"#90e0ef","dim":"#0a4060",
        "tab_bg":"#0b1526","tab_sel":"#0f2040","header_bg":"#030912",
        "danger":"#7f1d1d","danger_txt":"#fca5a5",
        "success":"#064e3b","success_txt":"#6ee7b7","type":"dark",
    },
    "🌿 Matcha": {
        "bg":"#1a2421","surface":"#23302b","surface2":"#2d3d37",
        "border":"rgba(144,238,144,0.1)","border2":"rgba(144,238,144,0.2)",
        "accent":"#6b9e78","accent2":"#4d7a58","accent3":"#8cbf99",
        "text":"#e8f0e8","subtext":"#a3b8ad","dim":"#687d72",
        "tab_bg":"#1f2b27","tab_sel":"#2d3d37","header_bg":"#141c19",
        "danger":"#9b2d30","danger_txt":"#ffb8b8",
        "success":"#2d6b45","success_txt":"#8ce8b5","type":"dark",
    },
    "🌟 Cyberpunk": {
        "bg":"#0b0c10","surface":"#1f2833","surface2":"#45a29e",
        "border":"rgba(102,252,241,0.2)","border2":"rgba(102,252,241,0.4)",
        "accent":"#66fcf1","accent2":"#45a29e","accent3":"#c5c6c7",
        "text":"#c5c6c7","subtext":"#66fcf1","dim":"#45a29e",
        "tab_bg":"#12181f","tab_sel":"#1f2833","header_bg":"#050608",
        "danger":"#ff003c","danger_txt":"#ff8a8a",
        "success":"#00ff66","success_txt":"#aaffcc","type":"dark",
    }
}

def natural_sort_key(s):
    """Ordenação alfanumérica natural (ex: 2.png precede 10.png)."""
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

def _build_stylesheet(t: dict) -> str:
    light = t.get("type") == "light"
    inp   = "#f8faf5" if light else "#13131e"
    inp2  = "#eef5ea" if light else "#161626"
    bb1   = t["surface2"] if light else "#252535"
    bb2   = t["surface"]  if light else "#1e1e2e"
    bh1   = "#d0daca"    if light else "#2e2e42"
    bh2   = "#c8dcc0"    if light else "#262636"
    sc    = t["surface"]  if light else "#0d0d12"
    sh    = t["surface2"] if light else "#2d2d3d"
    btxt  = t["text"]    if light else "#c4cad4"
    return f"""
QMainWindow,QWidget{{background:{t['bg']};color:{t['text']};font-family:"Segoe UI","Inter",sans-serif;font-size:13px;}}
QTabWidget::pane{{border:1px solid {t['border']};background:{t['surface']};border-radius:10px;top:-1px;}}
QTabBar::tab{{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 {t['surface2']},stop:1 {t['tab_bg']});color:{t['dim']};padding:10px 20px;border:1px solid {t['border']};border-bottom:none;border-top-left-radius:8px;border-top-right-radius:8px;min-width:90px;margin-right:3px;font-weight:500;font-size:12px;}}
QTabBar::tab:selected{{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 {t['tab_sel']},stop:1 {t['surface']});color:{t['accent3']};border-bottom:2px solid {t['accent']};font-weight:700;}}
QTabBar::tab:hover:!selected{{background:{t['surface2']};color:{t['subtext']};}}
QPushButton{{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 {bb1},stop:1 {bb2});color:{btxt};border:1px solid {t['border']};border-radius:7px;padding:6px 10px;font-weight:500;font-size:12px;}}
QPushButton:hover{{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 {bh1},stop:1 {bh2});border-color:{t['border2']};color:{t['text']};}}
QPushButton:pressed{{background:{t['surface']};border-color:{t['accent']};}}
QPushButton:disabled{{color:{t['dim']};border-color:{t['border']};}}
QPushButton#primary{{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 {t['accent']},stop:1 {t['accent2']});color:#fff;border-color:{t['accent2']};font-weight:600;padding:8px 16px;}}
QPushButton#primary:hover{{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 {t['accent3']},stop:1 {t['accent']});}}
QPushButton#primary:disabled{{opacity:0.45;}}
QPushButton#danger{{background:{t['danger']};color:{t['danger_txt']};border-color:{t['danger']};}}
QPushButton#play{{background:{t['success']};color:{t['success_txt']};border-color:{t['success']};padding:5px 10px;font-size:12px;}}
QLineEdit,QTextEdit,QListWidget{{background:{inp};color:{t['text']};border:1px solid {t['border']};border-radius:7px;padding:6px 10px;selection-background-color:{t['accent']};}}
QLineEdit:focus,QTextEdit:focus{{border-color:{t['accent']};background:{inp2};}}
QListWidget::item{{padding:5px;border-radius:4px;}}
QListWidget::item:selected{{background:{t['accent']};color:#fff;}}
QListWidget::item:hover:!selected{{background:{t['surface2']};}}
QProgressBar{{background:{t['surface']};border:1px solid {t['border']};border-radius:5px;height:18px;text-align:center;color:{t['text']};font-size:11px;}}
QProgressBar::chunk{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 {t['accent2']},stop:1 {t['accent3']});border-radius:4px;}}
QGroupBox{{color:{t['subtext']};border:1px solid {t['border']};border-radius:10px;margin-top:16px;padding:14px;font-weight:700;font-size:11px;letter-spacing:0.5px;background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 {t['surface2']},stop:1 {t['surface']});}}
QGroupBox::title{{subcontrol-origin:margin;left:12px;padding:0 8px;color:{t['subtext']};text-transform:uppercase;}}
QLabel{{color:{t['subtext']};}}
QLabel#heading{{color:{t['text']};font-size:15px;font-weight:700;}}
QLabel#subheading{{color:{t['dim']};font-size:12px;}}
QScrollArea{{border:none;background:transparent;}}
QScrollArea>QWidget>QWidget{{background:transparent;}}
QComboBox{{background:{inp};color:{t['text']};border:1px solid {t['border']};border-radius:7px;padding:6px 10px;min-height:30px;}}
QComboBox:focus{{border-color:{t['accent']};}}
QComboBox::drop-down{{border:none;width:22px;}}
QComboBox QAbstractItemView{{background:{t['surface']};color:{t['text']};selection-background-color:{t['accent']};border:1px solid {t['border2']};border-radius:6px;padding:4px;}}
QDoubleSpinBox,QSpinBox{{background:{inp};color:{t['text']};border:1px solid {t['border']};border-radius:7px;padding:5px 8px;}}
QDoubleSpinBox:focus,QSpinBox:focus{{border-color:{t['accent']};}}
QDoubleSpinBox::up-button,QSpinBox::up-button,QDoubleSpinBox::down-button,QSpinBox::down-button{{background:transparent;border:none;width:16px;}}
QSlider::groove:horizontal{{height:5px;background:{t['surface2']};border-radius:3px;}}
QSlider::handle:horizontal{{background:{t['accent']};border:2px solid {t['accent2']};width:14px;height:14px;margin:-5px 0;border-radius:7px;}}
QSlider::sub-page:horizontal{{background:{t['accent']};border-radius:3px;}}
QStatusBar{{background:{t['header_bg']};color:{t['dim']};border-top:1px solid {t['border']};font-size:11px;}}
QScrollBar:vertical{{background:{sc};width:8px;border-radius:4px;}}
QScrollBar::handle:vertical{{background:{sh};border-radius:4px;min-height:28px;margin:2px;}}
QScrollBar::handle:vertical:hover{{background:{t['accent']};}}
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{{height:0;}}
QScrollBar:horizontal{{background:{sc};height:8px;border-radius:4px;}}
QScrollBar::handle:horizontal{{background:{sh};border-radius:4px;min-width:28px;margin:2px;}}
QScrollBar::handle:horizontal:hover{{background:{t['accent']};}}
QScrollBar::add-line:horizontal,QScrollBar::sub-line:horizontal{{width:0;}}
QToolTip{{background:{t['surface2']};color:{t['text']};border:1px solid {t['border2']};border-radius:5px;padding:5px 8px;font-size:12px;}}
QCheckBox{{spacing:7px;color:{t['subtext']};}}
QCheckBox::indicator{{width:15px;height:15px;border-radius:4px;border:1px solid {t['border2']};background:{inp};}}
QCheckBox::indicator:checked{{background:{t['accent']};border-color:{t['accent2']};}}
QMenuBar{{background:{t['header_bg']};color:{t['text']};border-bottom:1px solid {t['border']};}}
QMenuBar::item:selected{{background:{t['surface2']};border-radius:4px;}}
QMenu{{background:{t['surface']};color:{t['text']};border:1px solid {t['border2']};border-radius:6px;padding:4px;}}
QMenu::item{{padding:6px 24px 6px 12px;border-radius:4px;}}
QMenu::item:selected{{background:{t['accent']};color:#fff;}}
QMenu::separator{{background:{t['border']};height:1px;margin:4px 8px;}}
"""

DARK_STYLESHEET = _build_stylesheet(THEMES["\U0001f311 Dark (Padr\u00e3o)"])

# ---------------------------------------------------------------------------
# Helper de log colorido (HTML)
# ---------------------------------------------------------------------------
def _colored_log(msg: str) -> str:
    """Retorna linha HTML colorida com base no prefixo."""
    stripped = msg.strip()
    if stripped.startswith("✓") or stripped.startswith("✅"):
        color = "#6ddd6d"
    elif stripped.startswith("⚠") or stripped.startswith("⚡"):
        color = "#f0c040"
    elif stripped.startswith("✗") or stripped.startswith("❌"):
        color = "#e05555"
    else:
        color = "#c8c8c8"
    safe = msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<span style="color:{color};">{safe}</span>'

def _append_log(text_edit: QTextEdit, msg: str):
    """Adiciona linha colorida ao QTextEdit e rola para o final."""
    text_edit.append(_colored_log(msg))
    sb = text_edit.verticalScrollBar()
    sb.setValue(sb.maximum())

# ---------------------------------------------------------------------------
# Nova Aba de Configurações TTS
# ---------------------------------------------------------------------------
class TtsConfigTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(18)

        def _dspin(lo, hi, val, step=0.05, decimals=2):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setSingleStep(step)
            s.setValue(val)
            s.setDecimals(decimals)
            s.setMinimumHeight(32)
            return s

        def _ispin(lo, hi, val):
            s = QSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setMinimumHeight(32)
            return s

        # --- Modelo ---
        model_group = QGroupBox("🤖  Motores e Modelos")
        mg = QGridLayout(model_group)
        mg.setSpacing(14)
        
        mg.addWidget(QLabel("Motor TTS:"), 0, 0)
        self.engine_combo = QComboBox()
        self.engine_combo.addItem("Chatterbox TTS", "chatterbox")
        self.engine_combo.addItem("Kokoro TTS", "kokoro")
        self.engine_combo.addItem("Qwen3-TTS (All)", "qwen")
        self.engine_combo.addItem("IndexTTS (Zero-Shot)", "indextts")
        self.engine_combo.setToolTip("Qual motor gerador base usar.")
        self.engine_combo.setMinimumHeight(32)
        mg.addWidget(self.engine_combo, 0, 1)

        self.lbl_model_chatterbox = QLabel("Modelo (Chatterbox):")
        mg.addWidget(self.lbl_model_chatterbox, 1, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItem("⚡ Turbo  (Inglês, rápido)", "turbo")
        self.model_combo.addItem("🌐 Multilingual  (Vários idiomas)", "multilingual")
        self.model_combo.addItem("🔊 Original  (Expressivo)", "original")
        self.model_combo.setToolTip("Turbo = Inglês apenas; Multilingual = Vários idiomas; Original = Expressivo padrão")
        self.model_combo.setMinimumHeight(32)
        mg.addWidget(self.model_combo, 1, 1)
        root.addWidget(model_group)
        
        self.engine_combo.currentIndexChanged.connect(self._on_engine_change)
        self.model_combo.currentIndexChanged.connect(self._trigger_preload)

        # --- Geração ---
        gen_group = QGroupBox("⚙️  Parâmetros de Geração")
        gg = QGridLayout(gen_group)
        gg.setSpacing(14)
        gg.setColumnStretch(1, 1)
        gg.setColumnStretch(3, 1)

        # Row 0: Temperature | Speed Factor
        gg.addWidget(QLabel("Temperature:"), 0, 0)
        self.temp_spin = _dspin(0.05, 5.0, 0.8, step=0.05)  # DEFAULT: 0.8
        self.temp_spin.setToolTip("Aleatoriedade da geração (padrão: 0.8)")
        gg.addWidget(self.temp_spin, 0, 1)
        gg.addWidget(QLabel("Speed Factor:"), 0, 2)
        self.speed_spin = _dspin(0.5, 3.0, 1.0, step=0.05)  # DEFAULT: 1.0
        self.speed_spin.setToolTip("Velocidade da fala (1.0 = normal)")
        gg.addWidget(self.speed_spin, 0, 3)

        # Row 1: Exaggeration | CFG Weight
        gg.addWidget(QLabel("Exaggeration:"), 1, 0)
        self.exag_spin = _dspin(0.25, 2.0, 0.5)
        self.exag_spin.setToolTip("Expressividade emocional (0.5 = padrão)")
        gg.addWidget(self.exag_spin, 1, 1)
        gg.addWidget(QLabel("CFG/Pace:"), 1, 2)
        self.cfg_spin = _dspin(0.2, 1.0, 0.5)
        self.cfg_spin.setToolTip("Peso CFG / Ritmo (0.5 = padrão)")
        gg.addWidget(self.cfg_spin, 1, 3)

        # Row 2: Seed | formato
        gg.addWidget(QLabel("Generation Seed:"), 2, 0)
        self.seed_spin = _ispin(0, 2**31 - 1, 3000)  # DEFAULT: 3000
        self.seed_spin.setToolTip("Semente de geração. 0 = aleatório. Valor fixo reproduz resultados.")
        gg.addWidget(self.seed_spin, 2, 1)
        gg.addWidget(QLabel("Output Format:"), 2, 2)
        self.format_combo = QComboBox()
        self.format_combo.addItem("WAV", "wav")   # DEFAULT: WAV
        self.format_combo.addItem("MP3", "mp3")
        self.format_combo.addItem("OGG", "ogg")
        self.format_combo.setToolTip("Formato do arquivo de áudio gerado")
        self.format_combo.setMinimumHeight(32)
        gg.addWidget(self.format_combo, 2, 3)

        # Row 3: Sample Rate
        gg.addWidget(QLabel("Sample Rate (Hz):"), 3, 0)
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.addItem("24000 Hz  (padrão Chatterbox)", 24000)
        self.sample_rate_combo.addItem("44100 Hz  (CD Quality)", 44100)
        self.sample_rate_combo.addItem("22050 Hz", 22050)
        self.sample_rate_combo.setToolTip("Taxa de amostragem do áudio de saída (24000 = padrão Chatterbox)")
        self.sample_rate_combo.setMinimumHeight(32)
        gg.addWidget(self.sample_rate_combo, 3, 1)

        root.addWidget(gen_group)

        # --- Avançado ---
        adv_group = QGroupBox("🛠️  Avançado")
        ag = QGridLayout(adv_group)
        ag.setSpacing(14)
        ag.setColumnStretch(1, 1)
        ag.setColumnStretch(3, 1)

        # Row 0: min_p | top_p
        ag.addWidget(QLabel("Min P:"), 0, 0)
        self.minp_spin = _dspin(0.0, 1.0, 0.05, step=0.01)
        self.minp_spin.setToolTip("Descarta provabilidades minúsculas. Recomendado 0.02 - 0.1. (0 = desativar)")
        ag.addWidget(self.minp_spin, 0, 1)

        ag.addWidget(QLabel("Top P:"), 0, 2)
        self.topp_spin = _dspin(0.0, 1.0, 1.0, step=0.01)
        self.topp_spin.setToolTip("Limita amostragem ao Top percentil. (1.0 = desativar)")
        ag.addWidget(self.topp_spin, 0, 3)

        # Row 1: top_k | repetition_penalty
        ag.addWidget(QLabel("Top K:"), 1, 0)
        self.topk_spin = _ispin(0, 5000, 1000)
        self.topk_spin.setToolTip("Amostragem entre os N tokens mais prováveis")
        ag.addWidget(self.topk_spin, 1, 1)

        ag.addWidget(QLabel("Repet. Penalty:"), 1, 2)
        self.rep_spin = _dspin(1.0, 2.0, 1.2, step=0.05)
        self.rep_spin.setToolTip("Penaliza tokens recém gerados (> 1.0 = desvia de loops de fala)")
        ag.addWidget(self.rep_spin, 1, 3)

        # Row 2: norm_loudness | vad_chk
        self.norm_loudness_chk = QCheckBox("Normalize Loudness (-27 LUFS)")
        self.norm_loudness_chk.setChecked(True)
        self.norm_loudness_chk.setToolTip("Apenas Turbo: normaliza o volume final")
        ag.addWidget(self.norm_loudness_chk, 2, 0, 1, 2)
        
        self.vad_chk = QCheckBox("Ref VAD Trimming")
        self.vad_chk.setChecked(False)
        self.vad_chk.setToolTip("Voice Activity Detection: Remove silêncio do arquivo de referência (clonagem)")
        ag.addWidget(self.vad_chk, 2, 2, 1, 2)

        root.addWidget(adv_group)

        # --- Qualidade Whisper ---
        whisper_group = QGroupBox("🎙️  Verificação de Qualidade  (Whisper)")
        wg = QGridLayout(whisper_group)
        wg.setSpacing(14)
        wg.setColumnStretch(1, 1)
        wg.setColumnStretch(3, 1)

        wg.addWidget(QLabel("Modelo Whisper:"), 0, 0)
        self.whisper_combo = QComboBox()
        self.whisper_combo.addItems(["base", "small", "medium", "large"])
        self.whisper_combo.setToolTip("Modelo de reconhecimento de fala para verificar a qualidade")
        self.whisper_combo.setMinimumHeight(32)
        wg.addWidget(self.whisper_combo, 0, 1)

        wg.addWidget(QLabel("Threshold:"), 0, 2)
        self.sim_spin = _dspin(0.0, 1.0, 0.0)  # DEFAULT: 0 = desabilitado para máxima velocidade
        self.sim_spin.setToolTip("Similaridade mínima para aceitar o áudio. 0 = desabilitado (mais rápido). 0.75 = verificação via Whisper")
        wg.addWidget(self.sim_spin, 0, 3)

        wg.addWidget(QLabel("Máx. tentativas:"), 1, 0)
        self.retries_spin = _ispin(1, 10, 3)
        self.retries_spin.setToolTip("Número máximo de re-gerações por parágrafo")
        wg.addWidget(self.retries_spin, 1, 1)

        root.addWidget(whisper_group)
        
        # --- Efeitos de Áudio e Texto ---
        fx_group = QGroupBox("✨ Processamento Profissional FFmpeg")
        fxg = QGridLayout(fx_group)
        fxg.setSpacing(10)
        
        self.spacy_chk = QCheckBox("Pré-processar Texto (Adicionar Pausas / Dividir frases com spaCy)")
        self.fx_highpass_chk = QCheckBox("Highpass (Corta Graves/Rumble)")
        self.fx_deesser_chk = QCheckBox("De-esser (Suaviza sibilância 'S')")
        self.fx_comp_chk = QCheckBox("Compressor (Nivela volume da voz)")
        self.fx_silence_chk = QCheckBox("Remove Silêncio (> 0.5s)")
        self.fx_reverb_chk = QCheckBox("Reverb Leve (Sala)")
        self.fx_loudnorm_chk = QCheckBox("Loudnorm (-16 LUFS, Padrão Youtube)")
        
        self.fx_highpass_chk.setChecked(True)
        self.fx_deesser_chk.setChecked(True)
        self.fx_comp_chk.setChecked(True)
        self.fx_silence_chk.setChecked(True)
        self.fx_loudnorm_chk.setChecked(True)

        fxg.addWidget(self.spacy_chk, 0, 0, 1, 2)
        fxg.addWidget(self.fx_highpass_chk, 1, 0)
        fxg.addWidget(self.fx_deesser_chk, 1, 1)
        fxg.addWidget(self.fx_comp_chk, 2, 0)
        fxg.addWidget(self.fx_silence_chk, 2, 1)
        fxg.addWidget(self.fx_reverb_chk, 3, 0)
        fxg.addWidget(self.fx_loudnorm_chk, 3, 1)
        
        root.addWidget(fx_group)

        # --- Presets ---
        preset_group = QGroupBox("Presets de Configuração")
        preset_layout = QHBoxLayout(preset_group)
        
        self.preset_config_combo = QComboBox()
        self.preset_config_combo.activated.connect(self._on_preset_selected)
        
        self.btn_save_config_preset = QPushButton("💾 Salvar Preset")
        self.btn_save_config_preset.clicked.connect(self._save_config_preset)
        
        self.btn_delete_config_preset = QPushButton("🗑️ Deletar")
        self.btn_delete_config_preset.clicked.connect(self._delete_config_preset)
        
        self.btn_load_defaults = QPushButton("↺ Padrão")
        self.btn_load_defaults.clicked.connect(self._restore_explicit)
        
        preset_layout.addWidget(self.preset_config_combo, 1)
        preset_layout.addWidget(self.btn_save_config_preset)
        preset_layout.addWidget(self.btn_delete_config_preset)
        preset_layout.addWidget(self.btn_load_defaults)
        
        root.addWidget(preset_group)
        
        self._load_presets()
        root.addStretch()

    def _load_presets(self):
        self.preset_config_combo.clear()
        self.preset_config_combo.addItem("--- Selecione um Preset ---", None)
        presets_dir = Path("presets")
        if presets_dir.exists():
            for p in presets_dir.glob("*.json"):
                self.preset_config_combo.addItem(p.stem, str(p))
                
    def _on_preset_selected(self, index):
        path = self.preset_config_combo.itemData(index)
        if path and Path(path).exists():
            import json
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.load_session(data)
                QMessageBox.information(self, "Preset Carregado", f"Preset '{Path(path).stem}' carregado com sucesso.")
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Falha ao carregar preset: {str(e)}")
                
    def _save_config_preset(self):
        name, ok = QInputDialog.getText(self, "Salvar Preset", "Nome do Preset:")
        if ok and name.strip():
            presets_dir = Path("presets")
            presets_dir.mkdir(exist_ok=True)
            path = presets_dir / f"{name.strip()}.json"
            import json
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.get_session(), f, indent=4)
            self._load_presets()
            idx = self.preset_config_combo.findText(name.strip())
            if idx >= 0: self.preset_config_combo.setCurrentIndex(idx)
            QMessageBox.information(self, "Preset Salvo", f"Preset '{name.strip()}' salvo com sucesso.")
            
    def _delete_config_preset(self):
        path = self.preset_config_combo.currentData()
        if path and Path(path).exists():
            name = self.preset_config_combo.currentText()
            ans = QMessageBox.question(self, "Deletar", f"Tem certeza que deseja deletar o preset '{name}'?")
            # CORRIGIDO: QMessageBox.Yes está depreciado — usar StandardButton.Yes
            if ans == QMessageBox.StandardButton.Yes:
                Path(path).unlink()
                self._load_presets()
                self.preset_config_combo.setCurrentIndex(0)

    def _restore_explicit(self):
        # Valores padrao
        self.model_combo.setCurrentIndex(0)
        self.temp_spin.setValue(0.65)
        self.speed_spin.setValue(1.0)
        self.exag_spin.setValue(0.55)
        self.cfg_spin.setValue(0.60)
        self.seed_spin.setValue(3000)
        self.format_combo.setCurrentIndex(0)
        self.sample_rate_combo.setCurrentIndex(0)
        self.minp_spin.setValue(0.05)
        self.topp_spin.setValue(1.0)
        self.topk_spin.setValue(1000)
        self.rep_spin.setValue(1.2)
        self.norm_loudness_chk.setChecked(True)
        self.whisper_combo.setCurrentIndex(0)
        self.sim_spin.setValue(0.0)   # 0 = desabilitado por padrão (máxima velocidade)
        self.retries_spin.setValue(1)   # 1 tentativa (sem re-geração automática)
        
        # Desliga todos os FX por padrão
        self.spacy_chk.setChecked(False)
        self.fx_highpass_chk.setChecked(True)
        self.fx_deesser_chk.setChecked(True)
        self.fx_comp_chk.setChecked(True)
        self.fx_silence_chk.setChecked(True)
        self.fx_reverb_chk.setChecked(False)
        self.fx_loudnorm_chk.setChecked(True)

    def _on_engine_change(self):
        eng = self.engine_combo.currentData()
        
        # Oculta/Desabilita opções do Chatterbox se não for chatterbox
        self.lbl_model_chatterbox.setVisible(eng == "chatterbox")
        self.model_combo.setVisible(eng == "chatterbox")
        self.exag_spin.setEnabled(eng == "chatterbox")
        self.cfg_spin.setEnabled(eng == "chatterbox")
        self.rep_spin.setEnabled(eng == "chatterbox")
        self.norm_loudness_chk.setEnabled(eng == "chatterbox")
        
        # Atualiza a lista de vozes e a UI de Engine do AudioTab
        main_win = self.window()
        if hasattr(main_win, "audio_tab"):
            main_win.audio_tab._populate_voices()
            main_win.audio_tab._update_engine_ui(eng)
        
        self._trigger_preload()

    def _trigger_preload(self):
        main_win = self.window()
        if hasattr(main_win, "trigger_model_preload"):
            main_win.trigger_model_preload()

    def get_session(self):
        data = {
            "tts_engine": self.engine_combo.currentData(),
            "model_type": self.model_combo.currentData(),
            "temperature": self.temp_spin.value(),
            "speed": self.speed_spin.value(),
            "exaggeration": self.exag_spin.value(),
            "cfg_weight": self.cfg_spin.value(),
            "seed": self.seed_spin.value(),
            "output_format": self.format_combo.currentData(),
            "sample_rate": self.sample_rate_combo.currentData(),
            "min_p": self.minp_spin.value(),
            "top_p": self.topp_spin.value(),
            "top_k": self.topk_spin.value(),
            "repetition_penalty": self.rep_spin.value(),
            "norm_loudness": self.norm_loudness_chk.isChecked(),
            "max_retries": self.retries_spin.value(),
            "whisper_model": self.whisper_combo.currentText(),
            "similarity_threshold": self.sim_spin.value(),
            "use_spacy": self.spacy_chk.isChecked(),
            "ref_vad_trimming": self.vad_chk.isChecked(),
            "fx_highpass":   self.fx_highpass_chk.isChecked(),
            "fx_deesser":    self.fx_deesser_chk.isChecked(),
            "fx_compressor": self.fx_comp_chk.isChecked(),
            "fx_silence":    self.fx_silence_chk.isChecked(),
            "fx_reverb":     self.fx_reverb_chk.isChecked(),
            "fx_loudnorm":   self.fx_loudnorm_chk.isChecked(),
        }
        
        # Get Qwen params explicitly (since they belong to AudioTab but are vital for pipeline)
        main_win = self.window()
        if hasattr(main_win, "audio_tab"):
            data["qwen_task"] = main_win.audio_tab.qwen_task_combo.currentText()
            data["qwen_instruct"] = main_win.audio_tab.qwen_instruct.text().strip()
            data["qwen_ref_text"] = main_win.audio_tab.qwen_ref_text.text().strip()
        else:
            data["qwen_task"] = "CustomVoice"
            data["qwen_instruct"] = ""
            data["qwen_ref_text"] = ""
        return data

    def load_session(self, data):
        if "tts_engine" in data:
            idx = self.engine_combo.findData(data["tts_engine"])
            if idx >= 0: self.engine_combo.setCurrentIndex(idx)
        if "temperature" in data: self.temp_spin.setValue(data["temperature"])
        if "speed" in data: self.speed_spin.setValue(data["speed"])
        if "exaggeration" in data: self.exag_spin.setValue(data["exaggeration"])
        if "cfg_weight" in data: self.cfg_spin.setValue(data["cfg_weight"])
        if "seed" in data: self.seed_spin.setValue(data["seed"])
        if "output_format" in data:
            idx = self.format_combo.findData(data["output_format"])
            if idx >= 0: self.format_combo.setCurrentIndex(idx)
        if "sample_rate" in data:
            idx = self.sample_rate_combo.findData(data["sample_rate"])
            if idx >= 0: self.sample_rate_combo.setCurrentIndex(idx)
        if "min_p" in data: self.minp_spin.setValue(data["min_p"])
        if "top_p" in data: self.topp_spin.setValue(data["top_p"])
        if "top_k" in data: self.topk_spin.setValue(data["top_k"])
        if "repetition_penalty" in data: self.rep_spin.setValue(data["repetition_penalty"])
        if "norm_loudness" in data: self.norm_loudness_chk.setChecked(data["norm_loudness"])
        if "max_retries" in data: self.retries_spin.setValue(data["max_retries"])
        if "whisper_model" in data:
            idx = self.whisper_combo.findText(data["whisper_model"])
            if idx >= 0: self.whisper_combo.setCurrentIndex(idx)
        if "similarity_threshold" in data: self.sim_spin.setValue(data["similarity_threshold"])
        if "use_spacy" in data: self.spacy_chk.setChecked(data["use_spacy"])
        if "ref_vad_trimming" in data: self.vad_chk.setChecked(data["ref_vad_trimming"])
        # CORRIGIDO: usar nomes corretos dos widgets; mapeamento de session keys antigas/novas
        if "fx_highpass" in data: self.fx_highpass_chk.setChecked(data["fx_highpass"])
        if "fx_deesser" in data: self.fx_deesser_chk.setChecked(data["fx_deesser"])
        if "fx_compressor" in data: self.fx_comp_chk.setChecked(data["fx_compressor"])
        if "fx_silence" in data: self.fx_silence_chk.setChecked(data["fx_silence"])
        if "fx_reverb" in data: self.fx_reverb_chk.setChecked(data["fx_reverb"])
        if "fx_loudnorm" in data: self.fx_loudnorm_chk.setChecked(data["fx_loudnorm"])
        # Mapeamento retrocompatível com sessões salvas antes da correção
        if "fx_compressor" not in data and "fx_noise_reduction" in data:
            self.fx_comp_chk.setChecked(data.get("fx_compressor", False))

    def reset_defaults(self):
        self.engine_combo.setCurrentIndex(0)
        self.temp_spin.setValue(0.65)
        self.speed_spin.setValue(1.0)
        self.exag_spin.setValue(0.55)
        self.cfg_spin.setValue(0.60)
        self.seed_spin.setValue(3000)
        self.format_combo.setCurrentIndex(0)
        idx = self.sample_rate_combo.findData(24000)
        if idx >= 0: self.sample_rate_combo.setCurrentIndex(idx)
        self.minp_spin.setValue(0.05)
        self.topp_spin.setValue(1.0)
        self.topk_spin.setValue(1000)
        self.rep_spin.setValue(1.2)
        self.norm_loudness_chk.setChecked(True)
        self.retries_spin.setValue(1)       # 1 tentativa — sem overhead
        self.whisper_combo.setCurrentIndex(0)
        self.sim_spin.setValue(0.0)         # 0 = Whisper desabilitado (mais rápido)
        self.spacy_chk.setChecked(False)    # spaCy desabilitado por padrão
        self.vad_chk.setChecked(False)
        # CORRIGIDO: usar os nomes reais dos widgets de FX
        self.fx_highpass_chk.setChecked(True)
        self.fx_deesser_chk.setChecked(True)
        self.fx_comp_chk.setChecked(True)
        self.fx_silence_chk.setChecked(True)
        self.fx_reverb_chk.setChecked(False)
        self.fx_loudnorm_chk.setChecked(True)



class LanguageConfigTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mapping = {}
        self._cards = []
        self._preview_thread = None
        self._preview_pipeline = None
        self._setup_ui()

    def _setup_ui(self):
        root_vbox = QVBoxLayout(self)

        lbl_info = QLabel(
            "Configure a Engine, Voz, Velocidade e Temperatura para traduções de cada idioma.\n"
            "Arquivos convertidos com o modelo (terminados com '_' + idioma. Ex: roteiro_es.txt)\n"
            "usarão as opções mapeadas aqui automaticamente no painel principal."
        )
        lbl_info.setWordWrap(True)
        lbl_info.setStyleSheet("color:#a0a0a0; font-size:12px; margin-bottom:10px;")
        root_vbox.addWidget(lbl_info)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        self.scroll_widget = QWidget()
        self.cards_layout = QVBoxLayout(self.scroll_widget)
        self.cards_layout.setSpacing(15)
        self.cards_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll.setWidget(self.scroll_widget)
        
        root_vbox.addWidget(self.scroll)

    def _setup_ui(self):
        root_vbox = QVBoxLayout(self)

        lbl_info = QLabel(
            "Configure opções extremas do TTS para traduções de cada idioma.\n"
            "Arquivos convertidos usarão estes cards como base (ex: roteiro_es.txt)."
        )
        lbl_info.setWordWrap(True)
        lbl_info.setStyleSheet("color:#a0a0a0; font-size:12px; margin-bottom:10px;")
        root_vbox.addWidget(lbl_info)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        self.scroll_widget = QWidget()
        self.cards_layout = QVBoxLayout(self.scroll_widget)
        self.cards_layout.setSpacing(15)
        self.cards_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll.setWidget(self.scroll_widget)
        
        root_vbox.addWidget(self.scroll)

        btn_row = QHBoxLayout()
        self.btn_add = QPushButton("➕ Adicionar Idioma")
        self.btn_add.setObjectName("primary")
        self.btn_add.clicked.connect(lambda: self._add_card())
        btn_row.addWidget(self.btn_add)
        
        self.btn_save = QPushButton("💾 Salvar Configurações Manualmente")
        self.btn_save.setObjectName("primary")
        self.btn_save.clicked.connect(lambda: _save_session(self.window().get_session()) if hasattr(self.window(), "get_session") else QMessageBox.information(self,"Salvo","A janela principal fará o save."))
        btn_row.addWidget(self.btn_save)

        btn_row.addStretch()
        root_vbox.addLayout(btn_row)

    def _add_card(self, lang="pt", engine="chatterbox", voice="", config=None):
        from PySide6.QtWidgets import QFrame, QGroupBox, QInputDialog
        import copy

        if config is None: config = {}
        
        speed = config.get("speed", 1.0)
        temp = config.get("temperature", 0.65)
        exag = config.get("exaggeration", 0.5)
        cfg_w = config.get("cfg_weight", 0.5)
        seed = config.get("seed", 3000)
        min_p = config.get("min_p", 0.05)
        top_p = config.get("top_p", 1.0)
        top_k = config.get("top_k", 1000)
        rep_p = config.get("repetition_penalty", 1.2)
        fx_hp = config.get("fx_highpass", True)
        fx_comp = config.get("fx_compressor", True)
        fx_de = config.get("fx_deesser", True)
        fx_rev = config.get("fx_reverb", False)
        fx_sil = config.get("fx_silence", True)
        fx_loud = config.get("fx_loudnorm", True)

        card = QFrame()
        card.setStyleSheet("QFrame { background:#222225; border-radius:8px; border:1px solid #3a3a40; }")
        cv = QVBoxLayout(card)
        cv.setContentsMargins(15, 15, 15, 15)
        cv.setSpacing(10)

        # L1: Idioma, Engine, Rm
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Idioma (Sufixo):"))
        lang_combo = QComboBox()
        lang_combo.addItems(["en", "pt", "es", "fr", "ja", "ko", "zh", "Outro..."])
        r1.addWidget(lang_combo)

        def handle_lang_combo(t):
            if t == "Outro...":
                text, ok = QInputDialog.getText(self, "Novo Sufixo", "Digite o sufixo (ex: it, ru):")
                if ok and text.strip():
                    lang_combo.insertItem(lang_combo.count()-1, text.strip())
                    lang_combo.setCurrentIndex(lang_combo.count()-2)
                else:
                    lang_combo.setCurrentIndex(0)
        lang_combo.currentTextChanged.connect(handle_lang_combo)

        idx = lang_combo.findText(lang)
        if idx >= 0:
            lang_combo.setCurrentIndex(idx)
        elif lang:
            lang_combo.insertItem(lang_combo.count()-1, lang)
            lang_combo.setCurrentIndex(lang_combo.count()-2)

        r1.addSpacing(15)
        r1.addWidget(QLabel("Engine TTS:"))
        engine_combo = QComboBox()
        engine_combo.addItems(["chatterbox", "kokoro", "qwen", "indextts"])
        engine_combo.setCurrentText(engine)
        r1.addWidget(engine_combo)
        
        r1.addStretch()
        btn_rm = QPushButton("✖ Remover")
        btn_rm.setObjectName("danger")
        btn_rm.clicked.connect(lambda: self._remove_card(card))
        r1.addWidget(btn_rm)
        cv.addLayout(r1)

        # L2: Voz
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Voz:"))
        voice_combo = QComboBox()
        r2.addWidget(voice_combo, 1)
        btn_clone = QPushButton("📁 Clonar (Custom)")
        btn_clone.clicked.connect(lambda: self._browse_custom_voice(voice_combo))
        r2.addWidget(btn_clone)
        cv.addLayout(r2)

        def update_voices(eng, combo, current_v):
            combo.clear()
            if eng == "chatterbox":
                combo.addItem("Sem clonagem (Voz do Modelo)", "")
                base = Path(__file__).parent.parent / "voices"
                if base.exists():
                    for p in base.rglob("*.wav"): combo.addItem(p.stem, str(p))
            elif eng == "kokoro":
                base = Path(__file__).parent.parent / "Kokoro-TTS-Local-master" / "voices"
                if base.exists():
                    for p in base.glob("*.pt"): combo.addItem(p.stem, str(p))
            elif eng == "qwen":
                combo.addItems(["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"])
                qdir = Path(__file__).parent / "qwen_voices"
                if qdir.exists():
                    for p in qdir.glob("*.pt"): combo.addItem(p.stem, str(p))
            elif eng == "indextts":
                pdir = Path("presets")
                if pdir.exists():
                    for p in pdir.glob("*.wav"): combo.addItem(p.stem, str(p))
            
            idx = combo.findData(current_v)
            if idx >= 0: combo.setCurrentIndex(idx)
            else:
                idx = combo.findText(current_v)
                if idx >= 0: combo.setCurrentIndex(idx)
                elif current_v:
                    combo.addItem(current_v, current_v)
                    combo.setCurrentIndex(combo.count()-1)

        update_voices(engine, voice_combo, voice)
        engine_combo.currentTextChanged.connect(lambda txt, c=voice_combo: update_voices(txt, c, ""))
        
        # L3: Opções Avançadas (Toggle)
        adv_group = QGroupBox("⚙️ Opções TTS e Efeitos")
        adv_group.setCheckable(True)
        adv_group.setChecked(False) # Recolhido por padrão
        
        gl = QGridLayout(adv_group)
        w_spd = QDoubleSpinBox(); w_spd.setRange(0.5,2.0); w_spd.setSingleStep(0.1); w_spd.setValue(speed)
        w_tmp = QDoubleSpinBox(); w_tmp.setRange(0.1,1.0); w_tmp.setSingleStep(0.05); w_tmp.setValue(temp)
        w_exa = QDoubleSpinBox(); w_exa.setRange(0.0,1.0); w_exa.setSingleStep(0.05); w_exa.setValue(exag)
        w_cfg = QDoubleSpinBox(); w_cfg.setRange(0.0,1.0); w_cfg.setSingleStep(0.05); w_cfg.setValue(cfg_w)
        w_seed= QSpinBox(); w_seed.setRange(0,999999); w_seed.setValue(seed)
        w_minp= QDoubleSpinBox(); w_minp.setRange(0.0,1.0); w_minp.setSingleStep(0.05); w_minp.setValue(min_p)
        w_topp= QDoubleSpinBox(); w_topp.setRange(0.0,1.0); w_topp.setSingleStep(0.05); w_topp.setValue(top_p)
        w_topk= QSpinBox(); w_topk.setRange(1,9999); w_topk.setValue(top_k)
        w_rep = QDoubleSpinBox(); w_rep.setRange(1.0,2.0); w_rep.setSingleStep(0.1); w_rep.setValue(rep_p)
        
        chk_hp = QCheckBox("Ruído"); chk_hp.setChecked(fx_hp)
        chk_cp = QCheckBox("Compr."); chk_cp.setChecked(fx_comp)
        chk_de = QCheckBox("Eq."); chk_de.setChecked(fx_de)
        chk_rv = QCheckBox("Reverb"); chk_rv.setChecked(fx_rev)
        chk_sl = QCheckBox("Crt.Sil"); chk_sl.setChecked(fx_sil)
        chk_nm = QCheckBox("Norm."); chk_nm.setChecked(fx_loud)
        
        w_spd.setToolTip("Velocidade global. 1.0 é normal.")
        w_tmp.setToolTip("Temperatura (criatividade/emoção). ~0.65 é bom.")
        w_exa.setToolTip("Exagero de emoção (Chatterbox).")
        w_cfg.setToolTip("CFG Weight (fidelidade ao input de voz).")
        w_seed.setToolTip("Controle de semente (3000 padrão).")

        gl.addWidget(QLabel("Vel:"), 0,0); gl.addWidget(w_spd, 0,1)
        gl.addWidget(QLabel("Temp:"), 0,2); gl.addWidget(w_tmp, 0,3)
        gl.addWidget(QLabel("Exag:"), 0,4); gl.addWidget(w_exa, 0,5)
        gl.addWidget(QLabel("CFG:"), 0,6); gl.addWidget(w_cfg, 0,7)
        
        gl.addWidget(QLabel("Seed:"), 1,0); gl.addWidget(w_seed, 1,1)
        gl.addWidget(QLabel("MinP:"), 1,2); gl.addWidget(w_minp, 1,3)
        gl.addWidget(QLabel("TopP:"), 1,4); gl.addWidget(w_topp, 1,5)
        gl.addWidget(QLabel("TopK:"), 1,6); gl.addWidget(w_topk, 1,7)
        
        gl.addWidget(QLabel("RepPen:"), 2,0); gl.addWidget(w_rep, 2,1)
        
        fx_box = QHBoxLayout()
        fx_box.addWidget(chk_hp); fx_box.addWidget(chk_cp); fx_box.addWidget(chk_de)
        fx_box.addWidget(chk_rv); fx_box.addWidget(chk_sl); fx_box.addWidget(chk_nm)
        gl.addLayout(fx_box, 3, 0, 1, 8)
        
        cv.addWidget(adv_group)

        # L4: Test row
        r4 = QHBoxLayout()
        r4.addWidget(QLabel("Teste de Voz:"))
        test_edit = QLineEdit("Um teste da voz configurada para este idioma.")
        r4.addWidget(test_edit, 1)
        
        btn_test = QPushButton("▶ Testar Voz")
        btn_test.clicked.connect(lambda: self._test_card_voice(
            engine_combo.currentText(), 
            resolve_voice_path(voice_combo.currentData() or voice_combo.currentText()), 
            lang_combo.currentText() or "en", 
            test_edit.text(), 
            btn_test,
            w_spd.value(),
            w_tmp.value()
        ))
        r4.addWidget(btn_test)
        cv.addLayout(r4)

        self.cards_layout.addWidget(card)
        self._cards.append((
            card, lang_combo, engine_combo, voice_combo,
            w_spd, w_tmp, w_exa, w_cfg, w_seed, w_minp,
            w_topp, w_topk, w_rep, chk_hp, chk_cp, chk_de, chk_rv, chk_sl, chk_nm
        ))

    def _browse_custom_voice(self, combo):
        path, _ = QFileDialog.getOpenFileName(self, "Selecionar Voz para Clonagem", "", "Audio Files (*.wav *.mp3 *.flac *.m4a)")
        if path:
            combo.addItem(f"[Custom] {Path(path).stem}", path)
            combo.setCurrentIndex(combo.count()-1)

    def _remove_card(self, card_widget):
        for rec in self._cards:
            if rec[0] == card_widget:
                self._cards.remove(rec)
                card_widget.deleteLater()
                break

    def get_session(self):
        new_map = {}
        for cdict in self._cards:
            l = cdict[1].currentText().strip()
            if not l: continue
            new_map[l] = {
                "engine": cdict[2].currentText(),
                "voice": resolve_voice_path(cdict[3].currentData() or cdict[3].currentText()),
                "speed": cdict[4].value(),
                "temperature": cdict[5].value(),
                "exaggeration": cdict[6].value(),
                "cfg_weight": cdict[7].value(),
                "seed": cdict[8].value(),
                "min_p": cdict[9].value(),
                "top_p": cdict[10].value(),
                "top_k": cdict[11].value(),
                "repetition_penalty": cdict[12].value(),
                "fx_highpass": cdict[13].isChecked(),
                "fx_compressor": cdict[14].isChecked(),
                "fx_deesser": cdict[15].isChecked(),
                "fx_reverb": cdict[16].isChecked(),
                "fx_silence": cdict[17].isChecked(),
                "fx_loudnorm": cdict[18].isChecked()
            }
        return {"language_mapping": new_map}

    def load_session(self, sess: dict):
        self.mapping = sess.get("language_mapping", {})
        # clear cards
        for rec in list(self._cards):
            self._remove_card(rec[0])
        # rebuild
        for lang, dat in self.mapping.items():
            self._add_card(lang, dat.get("engine", "chatterbox"), dat.get("voice", ""), dat)

    def _test_voice_custom(self, text, engine, voice_path, lang_code, btn_widget=None, speed=None, temperature=None):
        # Wrapper para facilitar chamadas externas (ex: do Modo Expresso)
        # Se btn_widget for None, usamos um botão genérico
        
        main_win = self.window()
        if speed is None or temperature is None:
            s_val = 1.0; t_val = 0.65
            if hasattr(main_win, "tts_tab"):
                tts_cfg = main_win.tts_tab.get_session()
                s_val = tts_cfg.get("speed", 1.0)
                t_val = tts_cfg.get("temperature", 0.65)
            if speed is None: speed = s_val
            if temperature is None: temperature = t_val
            
        # Mock de botão se necessário
        class MockBtn:
            def setEnabled(self, b): pass
            def setText(self, t): pass
        
        target_btn = btn_widget if btn_widget else MockBtn()
        # [NEW] Get model_type from source if available (card or tts_tab)
        model_type = "turbo"
        if hasattr(btn_widget, "parent") and btn_widget.parent():
            # Tentar achar o card que contém este botão
            p = btn_widget.parent()
            while p and not hasattr(p, "get_config"): p = p.parent()
            if p: 
                card_cfg = p.get_config()
                model_type = card_cfg.get("model_type", "turbo")

        self._test_card_voice(engine, voice_path, lang_code, text, target_btn, speed, temperature, model_type)

    def _test_card_voice(self, eng, voice, lang, txt, btn_widget, speed, temperature, model_type="turbo"):
        if not txt.strip(): return
        
        # [THREAD SAFETY] Cleanup any previous thread and pipeline safely BEFORE creating new ones
        if hasattr(self, "_preview_thread") and self._preview_thread and self._preview_thread.isRunning():
            # AVOID WAITING ON SELF
            import PySide6.QtCore as QtCore
            if QtCore.QThread.currentThread() == self._preview_thread:
                print("[WARNING] Tentativa de wait() no próprio thread. Abortando cleanup direto.")
            else:
                if hasattr(self, "_preview_pipeline") and self._preview_pipeline:
                    self._preview_pipeline.cancel()
                self._preview_thread.quit()
                if not self._preview_thread.wait(2000):
                    self._preview_thread.terminate()
                    self._preview_thread.wait()

        from manhwa_app.audio_pipeline import AudioPipeline
        import tempfile
        from PySide6.QtCore import QThread
        
        cfg = {}
        w = self.window()
        if hasattr(w, "audio_tab"):
            cfg = w.audio_tab._get_tts_config()

        self._preview_dir = tempfile.mkdtemp()
        tmp_txt_path = Path(self._preview_dir) / "preview.txt"
        tmp_txt_path.write_text(txt.strip(), encoding="utf-8")

        btn_widget.setEnabled(False)
        btn_widget.setText("Gerando...")

        preview_configs = [{"path": str(tmp_txt_path), "voice": voice, "lang": lang, "speed": speed, "temperature": temperature}]
        
        self._preview_pipeline = AudioPipeline(
            file_configs=preview_configs,
            project_name="preview",
            output_root=self._preview_dir,
            tts_engine=eng,
            model_type=model_type,
            whisper_model=cfg.get("whisper_model", "base"),
            similarity_threshold=cfg.get("similarity_threshold", 0.75),
            max_retries=1,
            temperature=temperature,
            speed=speed,
            output_format="wav",
            fx_noise_reduction=cfg.get("fx_highpass", False),
            fx_compressor=cfg.get("fx_compressor", False),
            fx_eq=cfg.get("fx_deesser", False),
            fx_reverb=cfg.get("fx_reverb", False),
            fx_enhancer=cfg.get("fx_silence", False),
            fx_normalize=cfg.get("fx_loudnorm", False),
        )
        
        self._preview_thread = QThread()
        self._preview_thread.setObjectName("PreviewThread")
        self._preview_pipeline.moveToThread(self._preview_thread)
        self._preview_thread.started.connect(self._preview_pipeline.run)

        # --- Qt canonical thread teardown ---
        # 1) When pipeline finishes, tell thread's event loop to stop
        self._preview_pipeline.finished.connect(self._preview_thread.quit)
        # 2) When thread actually stops, schedule it for deletion
        self._preview_thread.finished.connect(self._preview_thread.deleteLater)
        # 3) Notify UI in main thread (QueuedConnection fires after thread.quit is processed)
        from PySide6.QtCore import Qt
        self._preview_pipeline.finished.connect(
            lambda s, m: self._on_preview_done(s, m, btn_widget),
            Qt.ConnectionType.QueuedConnection
        )
        self._preview_thread.start()

    def _on_preview_done(self, success, msg, btn_widget):
        """Called in MAIN thread via QueuedConnection when pipeline finishes.
        Thread teardown is handled by Qt signals — we only update the UI here.
        """
        # Clear references (thread/pipeline self-delete via finished→deleteLater signals)
        self._preview_thread = None
        self._preview_pipeline = None

        try:
            import torch, gc
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
        except: pass

        btn_widget.setEnabled(True)
        btn_widget.setText("📢 Testar")

        if success:
            audios_dir = Path(self._preview_dir) / "preview" / "audios"
            files = sorted(audios_dir.glob("*.wav"))
            if files:
                from PySide6.QtMultimedia import QSoundEffect
                from PySide6.QtCore import QUrl
                if not hasattr(self, "_preview_effect"):
                    self._preview_effect = QSoundEffect()
                    self._preview_effect.setVolume(1.0)
                self._preview_effect.setSource(QUrl.fromLocalFile(str(files[-1])))
                self._preview_effect.play()
        else:
            QMessageBox.warning(self, "Erro", f"Falha no preview:\n{msg}")

# ---------------------------------------------------------------------------
# Aba 1 — Geração de Áudio
# ---------------------------------------------------------------------------

class AudioTab(QWidget):
    audio_generated = Signal(list)
    run_all_audio_done = Signal(int, str, list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._files = []
        self._worker_thread = None
        self._pipeline = None
        self._generated = []
        self._player = QMediaPlayer(self)
        self._audio_out = QAudioOutput(self)
        self._player.setAudioOutput(self._audio_out)
        self._audio_out.setVolume(1.0)
        self._run_all_mode = False
        self._run_all_index = 0
        self._current_run_all_project = ""
        self._playing_row = None
        self._chain_active = False
        self._chain_index = 0   # CORRIGIDO: inicializar aqui para evitar AttributeError em _stop_audio
        self._setup_ui()
        self._player.playbackStateChanged.connect(self._on_playback_state)

    def _setup_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(14)
        left = QWidget()
        left.setMaximumWidth(420)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(10)

        # ----------------------------------------------------------------
        # Painel Gemini — Revisar e Traduzir (pré-processamento opcional)
        # ----------------------------------------------------------------
        self._gemini_worker: Optional[GeminiWorker] = None
        self._setup_gemini_panel(lv)

        proj_group = QGroupBox("Projeto & Saída")

        pg = QGridLayout(proj_group)
        pg.addWidget(QLabel("Nome do Projeto:"), 0, 0)
        self.project_edit = QLineEdit("meu_projeto")
        pg.addWidget(self.project_edit, 0, 1)
        pg.addWidget(QLabel("Pasta Raiz:"), 1, 0)
        out_row = QHBoxLayout()
        out_row.setContentsMargins(0, 0, 0, 0)
        self.output_root_edit = QLineEdit("output")
        btn_out = QPushButton("…")
        btn_out.setFixedWidth(30)
        btn_out.clicked.connect(self._browse_out)
        out_row.addWidget(self.output_root_edit)
        out_row.addWidget(btn_out)
        pg.addLayout(out_row, 1, 1)
        lv.addWidget(proj_group)
        files_group = QGroupBox("Arquivos de Texto (.txt)")
        fv = QVBoxLayout(files_group)
        tb = QHBoxLayout()
        btn_add = QPushButton("＋ Add .txt")
        btn_add.clicked.connect(self._add_files)
        btn_folder = QPushButton("📁 Add Pasta")
        btn_folder.clicked.connect(self._add_folder)
        btn_clear = QPushButton("✖ Limpar")
        btn_clear.clicked.connect(self._clear_files)
        tb.addWidget(btn_add)
        tb.addWidget(btn_folder)
        tb.addWidget(btn_clear)
        fv.addLayout(tb)
        self.files_list = QListWidget()
        self.files_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        fv.addWidget(self.files_list)
        # Voz e Idioma Global (Preset)
        vox_group = QGroupBox("Preset Rápido (Aplicar a todos)")
        vg = QGridLayout(vox_group)
        vg.setSpacing(10)
        
        vg.addWidget(QLabel("Voz:"), 0, 0)
        self.preset_voice_combo = QComboBox()
        self.preset_voice_combo.addItem("Sem clonagem (Voz do Modelo)", "")
        self._populate_voices()
        vg.addWidget(self.preset_voice_combo, 0, 1)

        btn_custom = QPushButton("📁 Arquivo...")
        btn_custom.setFixedWidth(90)
        btn_custom.clicked.connect(self._browse_custom_voice)
        vg.addWidget(btn_custom, 0, 2)
        
        vg.addWidget(QLabel("Idioma:"), 1, 0)
        self.preset_lang_combo = QComboBox()
        self.preset_lang_combo.addItems(["en", "pt", "es", "fr", "ja", "ko", "zh"])
        vg.addWidget(self.preset_lang_combo, 1, 1)
        
        btn_lang_cfg = QPushButton("⚙ Mapear Vozes (Tradução)")
        btn_lang_cfg.setToolTip("Adicionar regras de vozes e engines diferentes para cada idioma das traduções")
        btn_lang_cfg.setStyleSheet("background-color: #2b4566; color: #a4cefe; font-weight: bold; border: 1px solid #3c5ea4; padding: 4px;")
        btn_lang_cfg.clicked.connect(self._open_language_mapping)
        vg.addWidget(btn_lang_cfg, 1, 2)
        
        lv.addWidget(vox_group)
        self._custom_voice_path = None
        
        # --- Configurações Específicas do Qwen3 ---
        self.qwen_group = QGroupBox("🧠 Instruções Qwen3")
        qg = QGridLayout(self.qwen_group)
        
        qg.addWidget(QLabel("Módulo:"), 0, 0)
        self.qwen_task_combo = QComboBox()
        self.qwen_task_combo.addItems(["CustomVoice", "VoiceDesign", "VoiceClone"])
        self.qwen_task_combo.setToolTip("CustomVoice=Presets; VoiceDesign=Criar do Zero; VoiceClone=Usar Referência")
        qg.addWidget(self.qwen_task_combo, 0, 1)

        qg.addWidget(QLabel("Prompt de Interpretação:"), 1, 0)
        self.qwen_instruct = QLineEdit()
        self.qwen_instruct.setPlaceholderText("Ex: Fale de forma assustada e ofegante...")
        qg.addWidget(self.qwen_instruct, 1, 1)

        self.lbl_qwen_ref_text = QLabel("Texto Referência:")
        qg.addWidget(self.lbl_qwen_ref_text, 2, 0)
        self.qwen_ref_text = QLineEdit()
        self.qwen_ref_text.setPlaceholderText("Texto exato do áudio clonado (Para VoiceClone)")
        qg.addWidget(self.qwen_ref_text, 2, 1)

        self.btn_save_qwen = QPushButton("💾 Salvar Voz Atual (Prompt)")
        self.btn_save_qwen.clicked.connect(self._save_qwen_voice)
        qg.addWidget(self.btn_save_qwen, 3, 0, 1, 2)

        lv.addWidget(self.qwen_group)
        self.qwen_group.setVisible(False)
        self.qwen_task_combo.currentIndexChanged.connect(self._on_qwen_task_change)

        # --- Preview de Voz ---
        preview_group = QGroupBox("Teste Rápido de Voz")
        p_layout = QHBoxLayout(preview_group)
        self.preview_text = QLineEdit("Um teste rápido para escutar e validar a voz.")
        self.preview_text.setPlaceholderText("Escreva algo para testar a voz...")
        self.btn_preview = QPushButton("▶ Testar Voz")
        self.btn_preview.setFixedWidth(120)
        self.btn_preview.clicked.connect(self._start_preview)
        p_layout.addWidget(self.preview_text)
        p_layout.addWidget(self.btn_preview)
        lv.addWidget(preview_group)
        
        lv.addWidget(files_group)
        actions = QHBoxLayout()
        self.btn_generate = QPushButton("⚡ Gerar Áudios")
        self.btn_generate.setObjectName("primary")
        self.btn_generate.setMinimumHeight(42)
        self.btn_generate.clicked.connect(self._start_normal)
        actions.addWidget(self.btn_generate)
        self.btn_run_all = QPushButton("🚀 Run All (Áudio+Vídeo)")
        self.btn_run_all.setObjectName("primary")
        self.btn_run_all.setStyleSheet("background-color: #059669; border-color: #047857;")
        self.btn_run_all.setMinimumHeight(42)
        self.btn_run_all.clicked.connect(self._start_run_all)
        actions.addWidget(self.btn_run_all)
        self.btn_cancel = QPushButton("✖ Cancelar")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setMinimumHeight(42)
        self.btn_cancel.clicked.connect(self._cancel_generation)
        actions.addWidget(self.btn_cancel)
        lv.addLayout(actions)
        root.addWidget(left)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(10)
        prog_group = QGroupBox("Progresso")
        pgv = QVBoxLayout(prog_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        pgv.addWidget(self.progress_bar)
        rv.addWidget(prog_group)
        log_group = QGroupBox("Log")
        logv = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 10))
        self.log_text.setStyleSheet("background:#171717; color:#d0d0d0; border-radius: 4px;")
        logv.addWidget(self.log_text)
        rv.addWidget(log_group)
        out_group = QGroupBox("Áudios Gerados")
        ov = QVBoxLayout(out_group)
        self.audio_list = QListWidget()
        self.audio_list.setMaximumHeight(150)
        ov.addWidget(self.audio_list)
        player_row = QHBoxLayout()
        self.btn_play = QPushButton("▶ Reproduzir Atual")
        self.btn_play_chain = QPushButton("⏭ Reproduzir Todos")
        self.btn_stop = QPushButton("⏹ Parar")
        self.btn_regen = QPushButton("↻ Regravar Atual")
        self.btn_open_folder = QPushButton("📁 Abrir Saída")
        self.btn_play.clicked.connect(self._play_selected)
        self.btn_play_chain.clicked.connect(self._play_chain)
        self.btn_stop.clicked.connect(self._stop_audio)
        self.btn_regen.clicked.connect(self._regen_selected)
        self.btn_open_folder.clicked.connect(self._open_output_folder)
        player_row.addWidget(self.btn_play)
        player_row.addWidget(self.btn_play_chain)
        player_row.addWidget(self.btn_stop)
        player_row.addWidget(self.btn_regen)
        player_row.addStretch()
        player_row.addWidget(self.btn_open_folder)
        ov.addLayout(player_row)
        rv.addWidget(out_group)
        root.addWidget(right)

        # Conexões Gemini (dependem de files_list e _gemini_run_btn ser criado antes)
        if hasattr(self, "files_list") and hasattr(self, "_gemini_run_btn"):
            self.files_list.model().rowsInserted.connect(self._update_gemini_btn_state)
            self.files_list.model().rowsRemoved.connect(self._update_gemini_btn_state)
            self._update_gemini_btn_state()


    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Selecionar diretório de saída")
        if d: self.output_root_edit.setText(d)

    def _browse_custom_voice(self):
        path, _ = QFileDialog.getOpenFileName(self, "Selecionar Áudio p/ Clone", "", "Áudio (*.wav *.mp3 *.ogg *.flac)")
        if path:
            import shutil
            repo_root = Path(__file__).parent.parent
            cfg = self._get_tts_config()
            engine_mode = cfg.get("tts_engine", "chatterbox")
            
            if engine_mode == "chatterbox":
                target_dir = repo_root / "voices" / "cloned"
            else:
                target_dir = repo_root / "Kokoro-TTS-Local-master" / "voices" / "cloned"
                
            target_dir.mkdir(parents=True, exist_ok=True)
            dest_path = target_dir / Path(path).name
            
            if dest_path.resolve() != Path(path).resolve():
                try:
                    shutil.copy2(path, dest_path)
                    path = str(dest_path)
                except Exception as e:
                    print(f"Não foi possível copiar arquivo de voz para cloned/: {e}")
            
            self._custom_voice_path = path
            name = Path(path).name
            idx = self.preset_voice_combo.findText(f"👤 {name.replace('_', ' ').title()} (Clonada)")
            if idx >= 0:
                self.preset_voice_combo.setCurrentIndex(idx)
            else:
                self.preset_voice_combo.addItem(f"👤 {name.replace('_', ' ').title()} (Clonada)", path)
                self.preset_voice_combo.setCurrentIndex(self.preset_voice_combo.count() - 1)

    def _start_preview(self):
        txt = self.preview_text.text().strip()
        if not txt:
            QMessageBox.warning(self, "Sem Texto", "Digite um texto para testar a voz.")
            return

        import tempfile
        self._preview_dir = tempfile.mkdtemp()
        tmp_txt_path = Path(self._preview_dir) / "preview.txt"
        tmp_txt_path.write_text(txt, encoding="utf-8")

        self.btn_preview.setEnabled(False)
        self.btn_preview.setText("Gerando...")

        cfg = self._get_tts_config()
        voice_val = self.preset_voice_combo.currentData()
        lang_val = self.preset_lang_combo.currentText()
        
        print(f"[UI] Início do Preview | Engine: {cfg.get('tts_engine', 'chatterbox')} | Texto: {len(txt)} chars | Lang: {lang_val} | Voice: {voice_val}")
        
        preview_configs = [{"path": str(tmp_txt_path), "voice": voice_val, "lang": lang_val}]

        from manhwa_app.audio_pipeline import AudioPipeline
        self._preview_pipeline = AudioPipeline(
            file_configs=preview_configs,
            project_name="preview",
            output_root=self._preview_dir,
            tts_engine=cfg.get("tts_engine", "chatterbox"),
            model_type=cfg.get("model_type", "turbo"),
            whisper_model=cfg.get("whisper_model", "base"),
            similarity_threshold=cfg.get("similarity_threshold", 0.75),
            max_retries=1,
            temperature=cfg.get("temperature", 0.8),
            exaggeration=cfg.get("exaggeration", 0.5),
            cfg_weight=cfg.get("cfg_weight", 0.5),
            seed=cfg.get("seed", 3000),
            speed=cfg.get("speed", 1.0),
            output_format="wav",
            min_p=cfg.get("min_p", 0.05),
            top_p=cfg.get("top_p", 1.0),
            top_k=cfg.get("top_k", 1000),
            repetition_penalty=cfg.get("repetition_penalty", 1.2),
            norm_loudness=cfg.get("norm_loudness", True),
            ref_vad_trimming=cfg.get("ref_vad_trimming", False),
            # CORRIGIDO: usar novas chaves canonicas de FX (get_session retorna fx_highpass, fx_deesser, etc.)
            fx_noise_reduction=cfg.get("fx_highpass", False),
            fx_compressor=cfg.get("fx_compressor", False),
            fx_eq=cfg.get("fx_deesser", False),
            fx_reverb=cfg.get("fx_reverb", False),
            fx_enhancer=cfg.get("fx_silence", False),
            fx_normalize=cfg.get("fx_loudnorm", False),
            use_spacy=cfg.get("use_spacy", False),
        )
        self._preview_thread = QThread()
        self._preview_thread.setStackSize(16 * 1024 * 1024)  # Evita stack overflow silencioso com modelos grandes (Qwen)
        self._preview_pipeline.moveToThread(self._preview_thread)
        self._preview_thread.started.connect(self._preview_pipeline.run)
        
        self._preview_pipeline.log_message.connect(lambda msg: _append_log(self.log_text, "[PREVIEW] " + msg))
        self._preview_pipeline.finished.connect(self._on_preview_finished)
        
        print("[UI] Criando thread de background para o AudioPipeline...")
        self._preview_thread.start()

    def _on_preview_finished(self, success, msg):
        print(f"[UI] Fim do Preview | Sucesso: {success} | Msg: {msg}")
        if self._preview_thread:
            self._preview_thread.quit()
            self._preview_thread.wait()
            self._preview_thread.deleteLater()
            self._preview_thread = None
            
        if self._preview_pipeline:
            self._preview_pipeline.deleteLater()
            self._preview_pipeline = None

        
        self.btn_preview.setEnabled(True)
        self.btn_preview.setText("▶ Testar Voz")

        # CORRIGIDO: limpar CUDA após preview para evitar estado corrompido
        # O Chatterbox deixa tensores internos na stream CUDA após cada síntese.
        # Sem este flush, após 2+ previews o próximo pipeline falha com
        # cudaErrorDeviceSideAssert em embed_ref() → ref_wav.to(device).
        try:
            import torch, gc as _gc
            if torch.cuda.is_available():
                _gc.collect()
                torch.cuda.empty_cache()
        except Exception:
            pass

        # Null out pipeline/thread refs já limpos


        if success:
            audios_dir = Path(self._preview_dir) / "preview" / "audios"
            audio_files = list(audios_dir.glob("*.wav"))
            if audio_files:
                from PySide6.QtMultimedia import QSoundEffect
                from PySide6.QtCore import QUrl
                
                if not hasattr(self, "_preview_effect"):
                    self._preview_effect = QSoundEffect()
                    self._preview_effect.setVolume(1.0)
                
                audio_file = audio_files[0]
                self._preview_effect.setSource(QUrl.fromLocalFile(str(audio_file)))
                self._preview_effect.play()
        else:
            QMessageBox.warning(self, "Erro no Preview", f"Falha ao gerar o preview:\n{msg}")


    def _populate_voices(self):
        # Backup the current path so we can restore it if it's still available
        old_val = self.preset_voice_combo.currentData()
        self.preset_voice_combo.clear()
        
        cfg = self._get_tts_config()
        engine = cfg.get("tts_engine", "chatterbox")
        
        if engine == "chatterbox":
            self.preset_voice_combo.addItem("Sem clonagem (Voz do Modelo)", "")
            base_dir = Path(__file__).parent.parent / "voices"
            if base_dir.exists():
                cloned, standard = [], []
                for p in base_dir.rglob("*.wav"):
                    rel = p.relative_to(base_dir)
                    if "cloned" in rel.parts:
                        cloned.append(p)
                    else:
                        standard.append(p)

                def format_name_chatterbox(p):
                    return p.stem.replace("_", " ").title()
                    
                for path in sorted(standard, key=natural_sort_key):
                    self.preset_voice_combo.addItem(f"🔊 {format_name_chatterbox(path)}", str(path))
                for path in sorted(cloned, key=natural_sort_key):
                    self.preset_voice_combo.addItem(f"👤 {format_name_chatterbox(path)} (Clonada)", str(path))
                    
        elif engine == "kokoro":
            base_dir = Path(__file__).parent.parent / "Kokoro-TTS-Local-master" / "voices"
            if base_dir.exists():
                def get_kokoro_lang_str(voice_name):
                    prefix = voice_name[:2]
                    mapping = {
                        'af': '[EN-US]', 'am': '[EN-US]',
                        'bf': '[EN-GB]', 'bm': '[EN-GB]',
                        'jf': '[JA-JP]', 'jm': '[JA-JP]',
                        'zf': '[ZH-CN]', 'zm': '[ZH-CN]',
                        'ef': '[ES-ES]', 'em': '[ES-ES]',
                        'ff': '[FR-FR]', 'fm': '[FR-FR]',
                        'hf': '[HI-IN]', 'hm': '[HI-IN]',
                        'if': '[IT-IT]', 'im': '[IT-IT]',
                        'pf': '[PT-BR]', 'pm': '[PT-BR]',
                    }
                    return mapping.get(prefix, '[EN-US]')

                voices = list(base_dir.glob("*.pt"))
                for path in sorted(voices, key=natural_sort_key):
                    name = path.stem
                    lang = get_kokoro_lang_str(name)
                    self.preset_voice_combo.addItem(f"✨ {lang} {name}", str(path))
                    
        elif engine == "qwen":
            self.preset_voice_combo.addItem("Vivian (Female, Chinese, Lively)", "Vivian")
            self.preset_voice_combo.addItem("Serena (Female, Chinese, Gentle)", "Serena")
            self.preset_voice_combo.addItem("Uncle_Fu (Male, Chinese, Mellow)", "Uncle_Fu")
            self.preset_voice_combo.addItem("Dylan (Male, Beijing Dialect, Clear)", "Dylan")
            self.preset_voice_combo.addItem("Eric (Male, Sichuan Dialect, Bright)", "Eric")
            self.preset_voice_combo.addItem("Ryan (Male, English, Dynamic)", "Ryan")
            self.preset_voice_combo.addItem("Aiden (Male, English, Clear)", "Aiden")
            self.preset_voice_combo.addItem("Ono_Anna (Female, Japanese, Playful)", "Ono_Anna")
            self.preset_voice_combo.addItem("Sohee (Female, Korean, Warm)", "Sohee")
            
            qwen_dir = Path(__file__).parent / "qwen_voices"
            if qwen_dir.exists():
                for pt_file in sorted(qwen_dir.glob("*.pt"), key=natural_sort_key):
                    self.preset_voice_combo.addItem(f"💾 {pt_file.stem} (Voz Salva)", str(pt_file))
        
        elif engine == "indextts":
            # IndexTTS usa .wav como prompt para zero-shot
            presets_dir = Path("presets")
            if presets_dir.exists():
                for wav_file in sorted(presets_dir.glob("*.wav"), key=natural_sort_key):
                    self.preset_voice_combo.addItem(f"👤 {wav_file.stem} (Clone WAV)", str(wav_file))
            else:
                self.preset_voice_combo.addItem("Nenhuma voz (.wav) em presets/", "")
                    
        if old_val:
            idx = self.preset_voice_combo.findData(old_val)
            if idx >= 0:
                self.preset_voice_combo.setCurrentIndex(idx)

    def _update_engine_ui(self, engine_mode: str):
        # Oculta botões inuteis no Qwen
        self.qwen_group.setVisible(engine_mode == "qwen")
        self._on_qwen_task_change()
        # Se for IndexTTS, garantir que o grupo de clonagem (se houver) esteja visível ou o modo correto
        # (Neste app, IndexTTS usa o seletor comum de vozes que já populamos acima)

    def _on_qwen_task_change(self):
        task = self.qwen_task_combo.currentText()
        is_clone = task == "VoiceClone"
        self.lbl_qwen_ref_text.setVisible(is_clone)
        self.qwen_ref_text.setVisible(is_clone)

    def _save_qwen_voice(self):
        task = self.qwen_task_combo.currentText()
        if task == "CustomVoice":
            QMessageBox.information(self, "Aviso", "CustomVoice usa presets nativos. Selecione VoiceDesign ou VoiceClone para salvar uma nova voz.")
            return
            
        name, ok = QInputDialog.getText(self, "Salvar Voz", "Nome da voz (sem espaços curtos, ex: meu_narrador):")
        if not ok or not name.strip():
            return
            
        import re
        safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', name.strip())
        save_path = Path(__file__).parent / "qwen_voices" / f"{safe_name}.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        ref_t = self.qwen_ref_text.text().strip()
        ref_a = self._custom_voice_path
        inst = self.qwen_instruct.text().strip()
        
        if task == "VoiceClone" and (not ref_a or not ref_t):
            QMessageBox.warning(self, "Aviso", "Para salvar um VoiceClone, você precisa carregar um arquivo de áudio de referência e digitar o texto exato falado nele.")
            return
            
        QMessageBox.information(self, "Salvando...", "A extração da assinatura vocal começará no fundo. Aguarde a mensagem de sucesso!")
        
        class SaveThread(QThread):
            finished_ok = Signal(bool, str)
            def __init__(self, s_path, tsk, r_t, r_a, ins):
                super().__init__()
                self.s_path = s_path
                self.tsk = tsk
                self.r_t = r_t
                self.r_a = r_a
                self.ins = ins
            def run(self):
                try:
                    import torch
                    from manhwa_app.audio_pipeline import _get_qwen_model
                    if self.tsk == "VoiceClone":
                        q_model = _get_qwen_model("VoiceClone")
                        if not q_model:
                            self.finished_ok.emit(False, "Falha ao carregar Base Model")
                            return
                        prompt = q_model.create_voice_clone_prompt(ref_audio=str(self.r_a), ref_text=self.r_t)
                    else:
                        design_model = _get_qwen_model("VoiceDesign")
                        q_model = _get_qwen_model("VoiceClone")
                        if not design_model or not q_model:
                            self.finished_ok.emit(False, "Falha ao carregar Modelos Qwen")
                            return
                        with torch.inference_mode():
                            ref_wavs, sr = design_model.generate_voice_design(
                                text="Esta é a assinatura vocal primária estabelecida.",
                                language="Auto",
                                instruct=self.ins or "Uma voz padrão."
                            )
                        prompt = q_model.create_voice_clone_prompt(ref_audio=(ref_wavs[0], sr), ref_text="Esta é a assinatura vocal primária estabelecida.")
                    
                    torch.save(prompt, str(self.s_path))
                    self.finished_ok.emit(True, f"Voz salva em: {self.s_path.name}")
                except Exception as e:
                    self.finished_ok.emit(False, str(e))
                    
        self._qwen_save_thread = SaveThread(save_path, task, ref_t, ref_a, inst)
        self._qwen_save_thread.setStackSize(16 * 1024 * 1024)
        def _on_save_done(ok, msg):
            if ok:
                QMessageBox.information(self, "Sucesso", msg)
                self._populate_voices()
            else:
                QMessageBox.critical(self, "Erro", msg)
        self._qwen_save_thread.finished_ok.connect(_on_save_done)
        self._qwen_save_thread.start()

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Selecionar Textos", "", "Textos (*.txt)")
        if paths:
            paths = sorted(paths, key=natural_sort_key)
            for p in paths:
                self._files.append({"path": p})
            self._refresh_list()

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Selecionar Pasta de Textos")
        if folder:
            paths = sorted([str(p) for p in Path(folder).glob("*.txt")], key=natural_sort_key)
            for p in paths:
                self._files.append({"path": p})
            self._refresh_list()

    def _clear_files(self):
        self._files.clear()
        self._refresh_list()

    def _refresh_list(self):
        self.files_list.clear()
        for i, d in enumerate(self._files):
            name = Path(d['path']).name
            self.files_list.addItem(f"#{i+1} {name}")

    def _get_tts_config(self):
        main_win = self.window()
        if hasattr(main_win, "tts_tab"):
            return main_win.tts_tab.get_session()
        return {}

    def _start_normal(self):
        self._run_all_mode = False
        self._start_pipeline(self._files)

    def _start_run_all(self):
        if not self._files:
            QMessageBox.warning(self, "Sem Arquivos", "Adicione pelo menos um .txt")
            return
        main_win = self.window()
        if hasattr(main_win, "images_tab") and not main_win.images_tab.get_images():
            QMessageBox.warning(self, "Sem Imagens", "Vá na aba Imagens e adicione imagens antes do Run All.")
            return
        reply = QMessageBox.question(self, "Run All", "Isso irá gerar áudio e vídeo para TODOS os arquivos .txt sequencialmente. Deseja continuar?")
        if reply != QMessageBox.StandardButton.Yes:
           return
        self._run_all_mode = True
        self._run_all_index = 0
        self._run_all_next()

    def _run_all_next(self):
        if self._run_all_index >= len(self._files):
            self._run_all_mode = False
            _append_log(self.log_text, "\n🎉 Run All concluído com sucesso para todos os arquivos!")
            QMessageBox.information(self, "Run All Concluído", "Todos os projetos foram gerados.")
            return
        file_cfg = self._files[self._run_all_index]
        _append_log(self.log_text, f"\n=== RUN ALL: Iniciando {Path(file_cfg['path']).name} ===")
        proj_base = self.project_edit.text().strip() or "projeto"
        if len(self._files) > 1:
            proj_name = f"{proj_base}_{self._run_all_index + 1}"
        else:
            proj_name = proj_base
        self._current_run_all_project = proj_name
        self._start_pipeline([file_cfg], project_override=proj_name)

    def _open_language_mapping(self):
        w = self.window()
        if hasattr(w, "tabs"):
            lct = getattr(w, "language_config_tab", None)
            if lct:
                w.tabs.setCurrentWidget(lct)

    def _start_pipeline(self, configs, project_override=None):
        if not configs: return
        # CORRIGIDO: evitar double-start se pipeline já estiver rodando
        if self._worker_thread and self._worker_thread.isRunning():
            return
        self.btn_generate.setEnabled(False)
        self.btn_run_all.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self._generated.clear()
        self.audio_list.clear()
        cfg = self._get_tts_config()
        
        voice_val = self.preset_voice_combo.currentData()
        lang_val = self.preset_lang_combo.currentText()
        
        w = self.window()
        lct = getattr(w, "language_config_tab", None)
        mapping = lct.mapping if lct else {}

        base_speed = cfg.get("speed", 1.0)
        base_temp = cfg.get("temperature", 0.65)

        new_configs = []
        for c in configs:
            c_path = c["path"]
            c_name = Path(c_path).stem
            c_lang = lang_val
            c_voice = voice_val
            c_engine = cfg.get("tts_engine", "chatterbox")
            
            c_cfg = {
                "path": c_path, "voice": c_voice, "lang": c_lang, "engine": c_engine,
                "speed": base_speed, "temperature": base_temp,
                "exaggeration": cfg.get("exaggeration", 0.5),
                "cfg_weight": cfg.get("cfg_weight", 0.5),
                "seed": cfg.get("seed", 3000),
                "min_p": cfg.get("min_p", 0.05),
                "top_p": cfg.get("top_p", 1.0),
                "top_k": cfg.get("top_k", 1000),
                "repetition_penalty": cfg.get("repetition_penalty", 1.2),
                "fx_highpass": cfg.get("fx_highpass", True),
                "fx_compressor": cfg.get("fx_compressor", True),
                "fx_deesser": cfg.get("fx_deesser", True),
                "fx_reverb": cfg.get("fx_reverb", False),
                "fx_silence": cfg.get("fx_silence", True),
                "fx_loudnorm": cfg.get("fx_loudnorm", True)
            }
            
            parts = c_name.split("_")
            if len(parts) > 1 and parts[-1] in mapping:
                l_code = parts[-1]
                m = mapping[l_code]
                c_cfg["lang"] = l_code
                c_cfg["voice"] = m.get("voice", c_voice)
                c_cfg["engine"] = m.get("engine", c_engine)
                for k in ["speed", "temperature", "exaggeration", "cfg_weight", "seed", "min_p", "top_p", "top_k", "repetition_penalty", "fx_highpass", "fx_compressor", "fx_deesser", "fx_reverb", "fx_silence", "fx_loudnorm"]:
                    if k in m: c_cfg[k] = m[k]
                
                print(f"[UI] Trocando configuração dinâmica. {c_name} => Lang:{c_cfg['lang']}, Eng:{c_cfg['engine']}, SP:{c_cfg['speed']}, SEED:{c_cfg['seed']}")
                
            new_configs.append(c_cfg)
        
        from manhwa_app.audio_pipeline import AudioPipeline
        self._pipeline = AudioPipeline(
            file_configs=new_configs,
            project_name=project_override or (self.project_edit.text().strip() or "projeto"),
            output_root=self.output_root_edit.text().strip() or "output",
            tts_engine=cfg.get("tts_engine", "chatterbox"),
            model_type=cfg.get("model_type", "turbo"),
            whisper_model=cfg.get("whisper_model", "base"),
            similarity_threshold=cfg.get("similarity_threshold", 0.75),
            max_retries=cfg.get("max_retries", 3),
            temperature=cfg.get("temperature", 0.65),
            exaggeration=cfg.get("exaggeration", 0.5),
            cfg_weight=cfg.get("cfg_weight", 0.5),
            seed=cfg.get("seed", 3000),
            speed=cfg.get("speed", 1.0),
            output_format=cfg.get("output_format", "wav"),
            min_p=cfg.get("min_p", 0.05),
            top_p=cfg.get("top_p", 1.0),
            top_k=cfg.get("top_k", 1000),
            repetition_penalty=cfg.get("repetition_penalty", 1.2),
            norm_loudness=cfg.get("norm_loudness", True),
            ref_vad_trimming=cfg.get("ref_vad_trimming", False),
            # CORRIGIDO: usar novas chaves de FX emitidas por get_session()
            # AudioPipeline espera: fx_noise_reduction, fx_compressor, fx_reverb, fx_normalize, fx_enhancer
            # Mapeamento: get_session() novo → AudioPipeline param
            fx_noise_reduction=cfg.get("fx_highpass", False),   # highpass → noise_reduction (ativador geral de limpeza)
            fx_compressor=cfg.get("fx_compressor", False),
            fx_eq=cfg.get("fx_deesser", False),                  # deesser → eq slot
            fx_reverb=cfg.get("fx_reverb", False),
            fx_enhancer=cfg.get("fx_silence", False),            # silence removal → enhancer slot
            fx_normalize=cfg.get("fx_loudnorm", False),          # fx_loudnorm → normalize
            use_spacy=cfg.get("use_spacy", False),
        )
        self._worker_thread = QThread(self)
        self._worker_thread.setStackSize(16 * 1024 * 1024)
        self._pipeline.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._pipeline.run)
        self._pipeline.log_message.connect(self._on_log)
        self._pipeline.progress.connect(self._on_progress)
        self._pipeline.paragraph_done.connect(self._on_pdone)
        self._pipeline.finished.connect(self._on_done)
        self._worker_thread.start()

    @Slot(str)
    def _on_log(self, msg: str):
        _append_log(self.log_text, msg)

    @Slot(int, int)
    def _on_progress(self, current: int, total: int):
        if total > 0:
            self.progress_bar.setValue(int(current * 100 / total))
            self.progress_bar.setFormat(f"Parágrafo {current}/{total}")

    @Slot(int, str, str)
    def _on_pdone(self, index: int, wav_path: str, text: str):
        self._generated.append((index, wav_path, text))
        short = text[:40] + ("…" if len(text) > 40 else "")
        item = QListWidgetItem(f"#{index}  {Path(wav_path).name}  |  {short}")
        item.setData(Qt.ItemDataRole.UserRole, wav_path)
        self.audio_list.addItem(item)

    @Slot(bool, str)
    def _on_done(self, success: bool, message: str):
        # Parar e aguardar a thread com timeout seguro
        if self._worker_thread:
            if self._worker_thread.isRunning():
                self._worker_thread.quit()
                if not self._worker_thread.wait(5000):  # 5s timeout
                    self._worker_thread.terminate()  # force if stuck
                    self._worker_thread.wait(2000)
            self._worker_thread.deleteLater()
            self._worker_thread = None
            
        if self._pipeline:
            self._pipeline.deleteLater()
            self._pipeline = None
            
        if success:
            self.progress_bar.setValue(100)
            paths = [g[1] for g in self._generated]
            if self._run_all_mode:
                _append_log(self.log_text, f"✓ Áudio pronto. Passando para Geração de Vídeo...")
                self.run_all_audio_done.emit(self._run_all_index, self._current_run_all_project, paths)
            else:
                self.btn_generate.setEnabled(True)
                self.btn_run_all.setEnabled(True)
                self.btn_cancel.setEnabled(False)
                self.audio_generated.emit(paths)
                _append_log(self.log_text, f"✓ {message}")
        else:
            self.btn_generate.setEnabled(True)
            self.btn_run_all.setEnabled(True)
            self.btn_cancel.setEnabled(False)
            self._run_all_mode = False
            _append_log(self.log_text, f"✗ {message}")

    def continue_run_all(self):
        if self._run_all_mode:
            self._run_all_index += 1
            self._run_all_next()

    def _cancel_generation(self):
        if self._pipeline:
            self._pipeline.cancel()
        self._run_all_mode = False
        self.btn_cancel.setEnabled(False)
        _append_log(self.log_text, "⚠ Cancelamento solicitado…")

    def _play_selected(self):
        row = self.audio_list.currentRow()
        if row < 0: return
        wav = self.audio_list.item(row).data(Qt.ItemDataRole.UserRole)
        self._play_wav(wav, row)

    def _play_wav(self, wav: str, row: Optional[int] = None):
        self._player.stop()
        self._player.setSource(QUrl.fromLocalFile(wav))
        self._player.play()
        self._playing_row = row

    def _stop_audio(self):
        self._chain_active = False
        self._player.stop()

    def _play_chain(self):
        if self.audio_list.count() == 0: return
        self._chain_active = True
        self._chain_index = 0
        self._play_chain_next()

    def _play_chain_next(self):
        if not self._chain_active or self._chain_index >= self.audio_list.count():
            self._chain_active = False
            return
        item = self.audio_list.item(self._chain_index)
        self.audio_list.setCurrentRow(self._chain_index)
        wav = item.data(Qt.ItemDataRole.UserRole)
        self._play_wav(wav, self._chain_index)

    @Slot()
    def _on_playback_state(self):
        from PySide6.QtMultimedia import QMediaPlayer as _QMP
        if self._player.playbackState() == _QMP.PlaybackState.StoppedState:
            if self._chain_active:
                self._chain_index += 1
                self._play_chain_next()

    def _regen_selected(self):
        row = self.audio_list.currentRow()
        if row < 0 or row >= len(self._generated):
            QMessageBox.information(self, "Selecione", "Selecione um áudio na lista.")
            return
        idx, wav_path, text = self._generated[row]
        project = self.project_edit.text().strip() or "projeto"
        import tempfile, textwrap
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(text)
            tmp_txt = f.name
        row_voice = self.preset_voice_combo.currentData()
        row_lang = self.preset_lang_combo.currentText()
        cfg = self._get_tts_config()
        from manhwa_app.audio_pipeline import AudioPipeline
        pipeline = AudioPipeline(
            file_configs=[{"path": tmp_txt, "voice": row_voice, "lang": row_lang}],
            project_name=project,
            output_root=self.output_root_edit.text().strip() or "output",
            model_type=cfg.get("model_type", "turbo"),
            whisper_model=cfg.get("whisper_model", "base"),
            similarity_threshold=cfg.get("similarity_threshold", 0.75),
            max_retries=cfg.get("max_retries", 3),
            temperature=cfg.get("temperature", 0.8),
            exaggeration=cfg.get("exaggeration", 0.5),
            cfg_weight=cfg.get("cfg_weight", 0.5),
            seed=cfg.get("seed", 0)
        )
        self._regen_thread = QThread(self)
        self._regen_pipeline = pipeline
        self._regen_pipeline.moveToThread(self._regen_thread)
        def _on_regen_done(success, msg):
            Path(tmp_txt).unlink(missing_ok=True)
            self._regen_thread.quit()
            # CORRIGIDO: aguardar a thread terminar para evitar 'QThread destroyed while running'
            thread.wait(3000)   # timeout de 3s
            new_wav = (Path(self.output_root_edit.text().strip() or "output") / project / "audios" / "audio_1.wav")
            if success and new_wav.exists():
                import shutil
                shutil.copy2(str(new_wav), str(wav_path))
                _append_log(self.log_text, f"✓ Áudio #{idx} regravado.")
            else:
                _append_log(self.log_text, f"✗ Falha ao regravar #{idx}.")
        self._regen_thread.started.connect(self._regen_pipeline.run)
        self._regen_pipeline.log_message.connect(self._on_log)
        self._regen_pipeline.finished.connect(_on_regen_done)
        _append_log(self.log_text, f"↻ Regravando parágrafo #{idx}…")
        self._regen_thread.start()

    def _open_output_folder(self):
        project = self.project_edit.text().strip() or "projeto"
        folder = Path(self.output_root_edit.text().strip() or "output") / project
        import subprocess as _sp
        if folder.exists():
            _sp.Popen(["explorer", str(folder.resolve())])

    def get_generated_paths(self) -> List[str]:
        return [g[1] for g in self._generated]

    def load_session(self, data: dict):
        if "project" in data: self.project_edit.setText(data["project"])
        if "output_root" in data: self.output_root_edit.setText(data["output_root"])
        if "preset_lang" in data:
            idx = self.preset_lang_combo.findText(data["preset_lang"])
            if idx >= 0: self.preset_lang_combo.setCurrentIndex(idx)
        if "preset_voice" in data:
            idx = self.preset_voice_combo.findData(data["preset_voice"])
            if idx >= 0:
                self.preset_voice_combo.setCurrentIndex(idx)
            else:
                idx = self.preset_voice_combo.findText(data["preset_voice"])
                if idx >= 0: self.preset_voice_combo.setCurrentIndex(idx)

    def get_session(self) -> dict:
        return {
            "project": self.project_edit.text().strip(),
            "output_root": self.output_root_edit.text().strip(),
            "preset_lang": self.preset_lang_combo.currentText(),
            "preset_voice": self.preset_voice_combo.currentData() or self.preset_voice_combo.currentText()
        }

    def reset_defaults(self):
        self.project_edit.setText("")
        self.output_root_edit.setText("output")

    # ------------------------------------------------------------------
    # Painel Gemini — Revisar e Traduzir (pré-processamento opcional)
    # ------------------------------------------------------------------

    def _setup_gemini_panel(self, parent_layout: QVBoxLayout):
        """Build the collapsible Gemini pre-processing panel."""
        self._gemini_collapsed = True

        # Outer container widget (shown/hidden together)
        self._gemini_container = QWidget()
        cv = QVBoxLayout(self._gemini_container)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(6)

        # Header row: toggle button
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        self._gemini_toggle_btn = QPushButton("▶  🌐 Revisar e Traduzir com Gemini")
        self._gemini_toggle_btn.setCheckable(True)
        self._gemini_toggle_btn.setChecked(False)
        self._gemini_toggle_btn.setStyleSheet(
            "text-align:left; padding:6px 10px; font-weight:600;"
        )
        self._gemini_toggle_btn.clicked.connect(self._toggle_gemini_panel)
        header_row.addWidget(self._gemini_toggle_btn)
        cv.addLayout(header_row)

        # Collapsible body
        self._gemini_body = QWidget()
        bv = QVBoxLayout(self._gemini_body)
        bv.setContentsMargins(8, 4, 8, 8)
        bv.setSpacing(8)

        if not _GEMINI_AVAILABLE:
            warn_lbl = QLabel(
                "⚠ Biblioteca google-genai não instalada.\n"
                "  pip install google-genai"
            )
            warn_lbl.setStyleSheet("color:#f0c040; font-size:11px;")
            bv.addWidget(warn_lbl)
            self._gemini_body.setVisible(False)
            cv.addWidget(self._gemini_body)
            parent_layout.addWidget(self._gemini_container)
            self._gemini_toggle_btn.setEnabled(False)
            return

        # Model selection (synced with SettingsTab)
        mod_row = QHBoxLayout()
        mod_row.setContentsMargins(0, 0, 0, 0)
        mod_row.addWidget(QLabel("Modelo:"))
        self._gemini_model_combo = QComboBox()
        # Copiar itens do SettingsTab (se disponível) ou popular
        self._gemini_model_combo.setMinimumHeight(30)
        self._gemini_model_combo.currentIndexChanged.connect(self._update_gemini_btn_state)
        bv.addLayout(mod_row)
        mod_row.addWidget(self._gemini_model_combo, 1)

        # Syncing logic will be added via MainWindow later or here if possible
        # But AudioTab doesn't have easy access to settings_tab until runtime
        
        info_lbl = QLabel(
            "💡 Outras configurações na aba <b>🔧 Configurações</b>."
        )
        info_lbl.setStyleSheet("color:#6dbdff; font-size:11px;")
        info_lbl.setWordWrap(True)
        bv.addWidget(info_lbl)


        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 4, 0, 0)
        self._gemini_run_btn = QPushButton("▶  Processar Roteiro")
        self._gemini_run_btn.setObjectName("primary")
        self._gemini_run_btn.setEnabled(False)
        self._gemini_run_btn.clicked.connect(self._on_gemini_start)
        self._gemini_cancel_btn = QPushButton("⏹  Cancelar")
        self._gemini_cancel_btn.setObjectName("danger")
        self._gemini_cancel_btn.setEnabled(False)
        self._gemini_cancel_btn.clicked.connect(self._on_gemini_cancel)
        btn_row.addWidget(self._gemini_run_btn, 1)
        btn_row.addWidget(self._gemini_cancel_btn)
        bv.addLayout(btn_row)

        # Progress bar (hidden until running)
        self._gemini_progress = QProgressBar()
        self._gemini_progress.setRange(0, 100)
        self._gemini_progress.setValue(0)
        self._gemini_progress.setVisible(False)
        bv.addWidget(self._gemini_progress)

        # Status label
        self._gemini_status_lbl = QLabel("")
        self._gemini_status_lbl.setStyleSheet("font-size:11px;")
        self._gemini_status_lbl.setWordWrap(True)
        bv.addWidget(self._gemini_status_lbl)

        self._gemini_body.setVisible(False)
        cv.addWidget(self._gemini_body)
        parent_layout.addWidget(self._gemini_container)




    def _toggle_gemini_panel(self, checked: bool):
        self._gemini_body.setVisible(checked)
        arrow = "▼" if checked else "▶"
        self._gemini_toggle_btn.setText(f"{arrow}  🌐 Revisar e Traduzir com Gemini")

    def _update_gemini_btn_state(self):
        if not _GEMINI_AVAILABLE:
            return
        has_files = self.files_list.count() > 0
        has_model = bool(getattr(self, "_gemini_model_combo", None) and 
                         self._gemini_model_combo.currentText())
        self._gemini_run_btn.setEnabled(has_files and has_model)


    def _on_gemini_start(self):
        """Launch GeminiWorker for the currently selected (or first) .txt file."""
        if not _GEMINI_AVAILABLE or GeminiProcessor is None:
            return

        # Resolve which file to process
        selected = self.files_list.selectedItems()
        if selected:
            row = self.files_list.row(selected[0])
        else:
            row = 0
        if row >= len(self._files):
            return
        txt_path = self._files[row]["path"]

        settings = self.window().settings_tab
        api_key = settings.get_api_key()
        if not api_key:
            self._gemini_status_lbl.setText("❌ Insira sua API Key na aba <b>Configurações</b> antes de continuar.")
            self.window().tabs.setCurrentWidget(settings)
            return

        languages = settings.get_languages()
        delay = settings.get_delay()
        
        # Use local combo if it has data, otherwise fallback to settings
        model_name = self._gemini_model_combo.currentData() or settings.get_model_name()
        
        revision_prompt = settings.get_revision_prompt()

        translation_prompt = settings.get_translation_prompt()
        chunk_size = settings.get_chunk_size()
        overlap = settings.get_overlap()
        thinking_level = settings.get_thinking_level()
        media_resolution = settings.get_media_resolution()
        
        self._gemini_worker = GeminiWorker(
            txt_path=txt_path,
            api_key=api_key,
            languages=languages,
            delay_seconds=delay,
            model_name=model_name,
            revision_prompt=revision_prompt,
            translation_prompt=translation_prompt,
            chunk_size=chunk_size,
            overlap=overlap,
            thinking_level=thinking_level,
            media_resolution=media_resolution
        )
        self._gemini_worker.progress.connect(self._on_gemini_progress)
        self._gemini_worker.finished.connect(self._on_gemini_done)
        self._gemini_worker.error.connect(self._on_gemini_error)

        self._gemini_run_btn.setEnabled(False)
        self._gemini_cancel_btn.setEnabled(True)
        self._gemini_progress.setValue(0)
        self._gemini_progress.setVisible(True)
        self._gemini_status_lbl.setText("⏳ Iniciando…")
        self._gemini_worker.start()

    def _on_gemini_progress(self, current: int, total: int, message: str):
        if total > 0:
            pct = int(current * 100 / total)
            self._gemini_progress.setValue(pct)
        self._gemini_status_lbl.setText(message)

    def _on_gemini_done(self, result: dict):
        self._gemini_run_btn.setEnabled(True)
        self._gemini_cancel_btn.setEnabled(False)
        self._gemini_progress.setValue(100)
        if result:
            paths = "\n".join(f"  ✅ {Path(p).name}" for p in result.values())
            self._gemini_status_lbl.setText(f"Concluído!\n{paths}")
            
            # --- Auto-load mechanism ---
            selected = self.files_list.selectedItems()
            row = self.files_list.row(selected[0]) if selected else 0
            if 0 <= row < len(self._files):
                # Remove original
                self._files.pop(row)
                self.files_list.takeItem(row)
                
                # Insert generated files at the same position
                insert_idx = row
                for lang_code, new_path in result.items():
                    self._files.insert(insert_idx, {"path": str(new_path), "voice": None, "lang": None})
                    self.files_list.insertItem(insert_idx, Path(new_path).name)
                    insert_idx += 1
                
                # Select the first newly added file
                self.files_list.setCurrentRow(row)

        else:
            self._gemini_status_lbl.setText("⚠ Processamento concluído sem saída.")
        if self._gemini_worker:
            self._gemini_worker.deleteLater()
            self._gemini_worker = None

    def _on_gemini_error(self, message: str):
        self._gemini_run_btn.setEnabled(True)
        self._gemini_cancel_btn.setEnabled(False)
        self._gemini_progress.setVisible(False)
        self._gemini_status_lbl.setText(f"❌ {message}")
        if self._gemini_worker:
            self._gemini_worker.deleteLater()
            self._gemini_worker = None

    def _on_gemini_cancel(self):
        if self._gemini_worker and self._gemini_worker.isRunning():
            self._gemini_worker.cancel()
            self._gemini_status_lbl.setText("⏹ Cancelamento solicitado…")
            self._gemini_cancel_btn.setEnabled(False)

# ---------------------------------------------------------------------------
# Aba 2 — Imagens
# ---------------------------------------------------------------------------


THUMB_SIZE = 130

class ImageThumbWidget(QWidget):
    """Cartão de miniatura para a grade de imagens."""

    def __init__(self, index: int, image_path: str, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self._index = index
        self.setFixedSize(THUMB_SIZE + 20, THUMB_SIZE + 46)
        lv = QVBoxLayout(self)
        lv.setContentsMargins(4, 4, 4, 4)
        lv.setSpacing(3)

        self.thumb_label = QLabel()
        self.thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb_label.setFixedSize(THUMB_SIZE, THUMB_SIZE)
        self.thumb_label.setStyleSheet(
            "background:#2a2a2a; border:1px solid #3a3a3a; border-radius:4px;"
        )
        self._load_thumbnail()
        lv.addWidget(self.thumb_label)

        self.num_label = QLabel(f"#{index}")
        self.num_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.num_label.setStyleSheet("color:#5865f2; font-weight:600; font-size:11px;")
        lv.addWidget(self.num_label)

        name = Path(image_path).name
        if len(name) > 15:
            name = name[:12] + "…"
        lbl = QLabel(name)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color:#707070; font-size:9px;")
        lv.addWidget(lbl)

    def _load_thumbnail(self):
        try:
            px = QPixmap(self.image_path)
            if not px.isNull():
                sc = px.scaled(THUMB_SIZE, THUMB_SIZE,
                               Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
                self.thumb_label.setPixmap(sc)
            else:
                self.thumb_label.setText("?")
        except Exception:
            self.thumb_label.setText("✗")

    def update_index(self, idx: int):
        self._index = idx
        self.num_label.setText(f"#{idx}")


class ImagesTab(QWidget):
    images_changed = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._images: List[str] = []
        self._thumb_widgets: List[ImageThumbWidget] = []
        self._selected_index: Optional[int] = None
        self._setup_ui()

    def _setup_ui(self):
        mv = QVBoxLayout(self)
        mv.setContentsMargins(14, 14, 14, 14)
        mv.setSpacing(10)

        # Barra de ferramentas
        tb = QHBoxLayout()
        self.btn_add  = QPushButton("＋ Adicionar Imagens")
        self.btn_folder = QPushButton("📁 Usar Pasta")
        self.btn_up   = QPushButton("▲")
        self.btn_down = QPushButton("▼")
        self.btn_repl = QPushButton("⇄ Substituir")
        self.btn_rem  = QPushButton("✖ Remover")
        self.btn_rem.setObjectName("danger")
        self.btn_clear = QPushButton("Limpar Tudo")
        self.btn_clear.setObjectName("danger")
        self.img_count_label = QLabel("0 imagens")
        self.img_count_label.setObjectName("subheading")

        for w in [self.btn_add, self.btn_folder, self.btn_up, self.btn_down,
                  self.btn_repl, self.btn_rem, self.btn_clear]:
            tb.addWidget(w)
        tb.addStretch()
        tb.addWidget(self.img_count_label)
        mv.addLayout(tb)

        # Info de paridade
        self.parity_label = QLabel("")
        self.parity_label.setObjectName("subheading")
        mv.addWidget(self.parity_label)

        # Grade de miniaturas (scroll)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAcceptDrops(True)
        self.scroll_area.dragEnterEvent = self._drag_enter
        self.scroll_area.dropEvent = self._drop

        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(10)
        self.grid_layout.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self.scroll_area.setWidget(self.grid_widget)
        mv.addWidget(self.scroll_area)

        self.sel_label = QLabel("Nenhuma imagem selecionada")
        self.sel_label.setObjectName("subheading")
        mv.addWidget(self.sel_label)

        # Conectar
        self.btn_add.clicked.connect(self._add_images)
        self.btn_folder.clicked.connect(self._load_from_folder)
        self.btn_up.clicked.connect(self._move_up)
        self.btn_down.clicked.connect(self._move_down)
        self.btn_repl.clicked.connect(self._replace_image)
        self.btn_rem.clicked.connect(self._remove_image)
        self.btn_clear.clicked.connect(self._clear_all)

    def _drag_enter(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def _drop(self, event: QDropEvent):
        paths = []
        for url in event.mimeData().urls():
            p = url.toLocalFile()
            if Path(p).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                paths.append(p)
        if paths:
            paths = sorted(paths, key=natural_sort_key)
            self._add_paths(paths)
            event.acceptProposedAction()

    def _add_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Selecionar imagens", "",
            "Imagens (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if paths:
            paths = sorted(paths, key=natural_sort_key)
            self._add_paths(paths)

    def _load_from_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Selecionar pasta de imagens")
        if not folder:
            return
        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        paths = sorted(
            [str(p) for p in Path(folder).iterdir()
             if p.is_file() and p.suffix.lower() in exts],
            key=natural_sort_key
        )
        self._add_paths(paths)

    def _add_paths(self, paths: List[str]):
        for p in paths:
            self._images.append(p)
        self._rebuild_grid()
        self.images_changed.emit(self._images)

    def set_images(self, paths: List[str]):
        """Seta as imagens carregadas de fora (Ex: aba de vídeo)."""
        self._images = list(paths)
        self._rebuild_grid()
        self.images_changed.emit(self._images)

    def _rebuild_grid(self):
        for w in self._thumb_widgets:
            w.setParent(None)
            w.deleteLater()
        self._thumb_widgets = []
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        cols = max(1, self.scroll_area.width() // (THUMB_SIZE + 32))
        for i, path in enumerate(self._images):
            thumb = ImageThumbWidget(i + 1, path)
            thumb.mousePressEvent = lambda e, idx=i: self._select_thumb(idx)
            thumb.setStyleSheet("QWidget{background:#2a2a2a;border-radius:6px;}")
            self.grid_layout.addWidget(thumb, i // cols, i % cols)
            self._thumb_widgets.append(thumb)
            if i % 5 == 0:
                QApplication.processEvents()

        n = len(self._images)
        self.img_count_label.setText(f"{n} imagem{'ns' if n != 1 else ''}")
        self._selected_index = None
        self.sel_label.setText("Nenhuma imagem selecionada")

    def _select_thumb(self, idx: int):
        if self._selected_index is not None and self._selected_index < len(self._thumb_widgets):
            self._thumb_widgets[self._selected_index].setStyleSheet(
                "QWidget{background:#2a2a2a;border-radius:6px;}"
            )
        self._selected_index = idx
        if idx < len(self._thumb_widgets):
            self._thumb_widgets[idx].setStyleSheet(
                "QWidget{background:#1a3060;border-radius:6px;border:1px solid #5865f2;}"
            )
            self.sel_label.setText(f"Selecionada: #{idx+1} — {Path(self._images[idx]).name}")

    def _move_up(self):
        i = self._selected_index
        if i is None or i == 0:
            return
        self._images[i], self._images[i-1] = self._images[i-1], self._images[i]
        self._rebuild_grid()
        self._select_thumb(i - 1)
        self.images_changed.emit(self._images)

    def _move_down(self):
        i = self._selected_index
        if i is None or i >= len(self._images) - 1:
            return
        self._images[i], self._images[i+1] = self._images[i+1], self._images[i]
        self._rebuild_grid()
        self._select_thumb(i + 1)
        self.images_changed.emit(self._images)

    def _replace_image(self):
        i = self._selected_index
        if i is None:
            QMessageBox.information(self, "Selecione", "Clique em uma imagem para selecioná-la.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Substituir imagem", "",
            "Imagens (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if path:
            self._images[i] = path
            self._rebuild_grid()
            self._select_thumb(i)
            self.images_changed.emit(self._images)

    def _remove_image(self):
        i = self._selected_index
        if i is None:
            return
        self._images.pop(i)
        self._rebuild_grid()
        self.images_changed.emit(self._images)

    def _clear_all(self):
        if self._images:
            if QMessageBox.question(self, "Limpar", "Remover todas as imagens?") == QMessageBox.StandardButton.Yes:
                self._images = []
                self._rebuild_grid()
                self.images_changed.emit(self._images)

    def get_images(self) -> List[str]:
        return list(self._images)

    def set_images(self, paths: List[str]):
        self._images = list(paths)
        self._rebuild_grid()
        self.images_changed.emit(self._images)

    def update_parity(self, n_audios: int, imgs_per_audio: int = 1) -> Tuple[int, bool]:
        n_imgs = len(self._images)
        if n_audios == 0 and n_imgs == 0:
            self.parity_label.setText("")
            return 0, False
        
        matched = min(n_audios, n_imgs // imgs_per_audio)
        ok = (n_imgs == n_audios * imgs_per_audio) and n_audios > 0
        icon = "✓" if ok else "✗"
        color = "#6ddd6d" if ok else "#e05555"
        self.parity_label.setText(
            f'<span style="color:{color};">{icon} {n_audios} áudios / {n_imgs} imagens (modo {imgs_per_audio}x)</span>'
        )
        return matched, ok

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._images:
            self._rebuild_grid()


# ---------------------------------------------------------------------------
# Aba 3 — Geração de Vídeo
# ---------------------------------------------------------------------------

class VideoTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._audio_paths: List[str] = []
        self._image_paths: List[str] = []
        self._worker_thread: Optional[QThread] = None
        self._pipeline: Optional[VideoPipeline] = None
        self._last_preview_path: Optional[str] = None
        self._setup_ui()

    def _setup_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(14)

        left = QWidget()
        left.setMaximumWidth(330)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(10)

        # Resumo dos pares + botões de pasta
        pairs_group = QGroupBox("Pares Áudio ↔ Imagem")
        pv = QVBoxLayout(pairs_group)

        # Botões de carregamento por pasta
        folder_row = QHBoxLayout()
        btn_folder_both = QPushButton("📁 Áud+Img")
        btn_folder_both.setToolTip("Carrega áudios e imagens de uma mesma pasta, emparelha em ordem alfabética.")
        btn_folder_both.clicked.connect(self._load_folder_audio_image)
        btn_folder_audio = QPushButton("🔊 Áudios")
        btn_folder_audio.setToolTip("Carrega apenas arquivos de áudio de uma pasta.")
        btn_folder_audio.clicked.connect(self._load_folder_audio_only)
        btn_folder_imgs = QPushButton("🖼 Imagens")
        btn_folder_imgs.setToolTip("Carrega apenas imagens de uma pasta.")
        btn_folder_imgs.clicked.connect(self._load_folder_images_only)
        folder_row.addWidget(btn_folder_both)
        folder_row.addWidget(btn_folder_audio)
        folder_row.addWidget(btn_folder_imgs)
        pv.addLayout(folder_row)

        self.pairs_list = QListWidget()
        self.pairs_list.setMinimumHeight(160)
        pv.addWidget(self.pairs_list)
        self.pairs_info = QLabel("Carregue áudios na Aba 1 e imagens na Aba 2, ou use os botões de pasta acima.")
        self.pairs_info.setObjectName("subheading")
        self.pairs_info.setWordWrap(True)
        pv.addWidget(self.pairs_info)
        lv.addWidget(pairs_group)

        # Layout, Efeito e Transição
        fx_group = QGroupBox("Layout, Movimento e Transição")
        fxv = QVBoxLayout(fx_group)

        fxv.addWidget(QLabel("Layout de Imagens:"))
        self.layout_combo = QComboBox()
        self.layout_combo.addItem("Uma Imagem por Cena", "single")
        self.layout_combo.addItem("Duas Imagens por Cena (Split)", "split")
        self.layout_combo.addItem("Alternar: 1 / 2 Imagens", "mixed")
        self.layout_combo.currentIndexChanged.connect(self._on_layout_changed)
        fxv.addWidget(self.layout_combo)

        fxv.addWidget(QLabel("Efeito de Câmera:"))
        self.effect_combo = QComboBox()
        self.effect_combo.addItem("Automático (alternado)", "auto")
        self.effect_combo.addItem("Zoom In", "zoom_in")
        self.effect_combo.addItem("Zoom Out", "zoom_out")
        self.effect_combo.addItem("Pan Cima", "pan_up")
        self.effect_combo.addItem("Pan Baixo", "pan_down")
        fxv.addWidget(self.effect_combo)

        fxv.addWidget(QLabel("Transição entre Cenas:"))
        trans_row = QHBoxLayout()
        self.transition_combo = QComboBox()
        self.transition_combo.addItem("🌑 Fade (Escurecer)", "fade")
        self.transition_combo.addItem("💫 Blur Suave (Fluido)", "blur")
        self.transition_combo.addItem("⚡ Sem Transição (Corte)", "none")
        self.transition_combo.setToolTip(
            "Fade: escurece para preto.\n"
            "Blur Suave: desfoca levemente no início/fim da cena (fluido).\n"
            "Sem Transição: corte direto entre cenas."
        )
        trans_row.addWidget(self.transition_combo)
        
        self.transition_time_spin = QDoubleSpinBox()
        self.transition_time_spin.setRange(0.0, 5.0)
        self.transition_time_spin.setSingleStep(0.1)
        self.transition_time_spin.setValue(0.2)
        self.transition_time_spin.setSuffix("s")
        self.transition_time_spin.setToolTip("Duração do Fade/Blur em segundos")
        self.transition_time_spin.setFixedWidth(70)
        trans_row.addWidget(self.transition_time_spin)
        
        fxv.addLayout(trans_row)
        lv.addWidget(fx_group)

        # Filtros de Vídeo e Movimentação (Production)
        prod_group = QGroupBox("✨ Qualidade Visual & Movimento")
        prod_v = QVBoxLayout(prod_group)
        prod_grid = QGridLayout()
        
        self.chk_better_easing = QCheckBox("Easing Cinemático (Parabólico)")
        self.chk_better_easing.setChecked(True)
        self.chk_better_easing.setToolTip("Habilita movimento de câmera mais suave e natural nas extremidades")
        prod_grid.addWidget(self.chk_better_easing, 0, 0, 1, 2)
        
        self.chk_color_grading = QCheckBox("Color Grading")
        self.chk_color_grading.setChecked(True)
        self.chk_color_grading.setToolTip("Contraste suave e saturação viva (evita imagem lavada)")
        prod_grid.addWidget(self.chk_color_grading, 1, 0)
        
        self.chk_sharpen = QCheckBox("Sharpen")
        self.chk_sharpen.setChecked(True)
        self.chk_sharpen.setToolTip("Realça contornos levemente e nitidez geral")
        prod_grid.addWidget(self.chk_sharpen, 1, 1)

        self.chk_grain = QCheckBox("Film Grain Sutil")
        self.chk_grain.setChecked(False)
        self.chk_grain.setToolTip("Adiciona ruído cinematográfico de alta frequência à imagem final")
        prod_grid.addWidget(self.chk_grain, 2, 0)
        
        prod_v.addLayout(prod_grid)
        lv.addWidget(prod_group)

        # Música de Fundo
        bgm_group = QGroupBox("Música de Fundo (Opcional)")
        bgmv = QVBoxLayout(bgm_group)
        bgm_row = QHBoxLayout()
        self.bgm_path_edit = QLineEdit()
        self.bgm_path_edit.setPlaceholderText("Caminho do áudio (.mp3, .wav)...")
        btn_bgm_browse = QPushButton("…")
        btn_bgm_browse.setFixedWidth(32)
        btn_bgm_browse.clicked.connect(self._browse_bgm)
        bgm_row.addWidget(self.bgm_path_edit)
        bgm_row.addWidget(btn_bgm_browse)
        bgmv.addLayout(bgm_row)

        vol_row = QHBoxLayout()
        vol_row.addWidget(QLabel("Volume (%):"))
        self.bgm_vol_spin = QSpinBox()
        self.bgm_vol_spin.setRange(1, 100)
        self.bgm_vol_spin.setValue(10)
        self.bgm_vol_spin.setToolTip("Volume da música de fundo (padrão: 10%)")
        vol_row.addWidget(self.bgm_vol_spin)
        
        self.chk_auto_ducking = QCheckBox("Auto-Ducking")
        self.chk_auto_ducking.setChecked(True)
        self.chk_auto_ducking.setToolTip("Abaixa o volume da música automaticamente quando há narração")
        vol_row.addWidget(self.chk_auto_ducking)
        
        vol_row.addStretch()
        bgmv.addLayout(vol_row)
        lv.addWidget(bgm_group)

        # Saída
        out_group = QGroupBox("Arquivo de Saída")
        ov = QVBoxLayout(out_group)
        out_row = QHBoxLayout()
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("output/projeto/projeto_final.mp4")
        btn_browse = QPushButton("…")
        btn_browse.setFixedWidth(32)
        btn_browse.clicked.connect(self._browse_output)
        out_row.addWidget(self.output_path_edit)
        out_row.addWidget(btn_browse)
        ov.addLayout(out_row)
        lv.addWidget(out_group)

        # Botoes de Controle
        buttons_layout = QHBoxLayout()
        
        self.btn_preview = QPushButton("👀 Preview (3 Cenas)")
        self.btn_preview.setMinimumHeight(38)
        self.btn_preview.clicked.connect(self._start_preview)
        buttons_layout.addWidget(self.btn_preview)
        
        self.btn_generate = QPushButton("🎬 Gerar Vídeo")
        self.btn_generate.setObjectName("primary")
        self.btn_generate.setMinimumHeight(38)
        self.btn_generate.clicked.connect(self._start_generation)
        buttons_layout.addWidget(self.btn_generate)
        
        lv.addLayout(buttons_layout)

        self.btn_cancel = QPushButton("✖ Cancelar")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel)
        lv.addWidget(self.btn_cancel)
        lv.addStretch()
        root.addWidget(left)

        # Painel direito: progresso + log + preview
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(10)

        prog_group = QGroupBox("Progresso")
        pgv = QVBoxLayout(prog_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        pgv.addWidget(self.progress_bar)
        rv.addWidget(prog_group)

        # QMediaPlayer embutido
        self.video_group = QGroupBox("Player de Preview")
        vpv = QVBoxLayout(self.video_group)
        self.video_widget = QVideoWidget()
        # Fixando explicitamente o tamanho em 16:9 (480x270) conforme pedido
        self.video_widget.setFixedSize(480, 270)
        # Player Backend
        self.vp_player = QMediaPlayer()
        self.vp_audio = QAudioOutput()
        self.vp_player.setAudioOutput(self.vp_audio)
        self.vp_player.setVideoOutput(self.video_widget)
        self.vp_audio.setVolume(0.8)
        
        # Botões do Player
        vp_btns = QHBoxLayout()
        self.btn_vp_play = QPushButton("▶ Play")
        self.btn_vp_pause = QPushButton("⏸ Pause")
        self.btn_vp_stop = QPushButton("⏹ Stop")
        self.btn_vp_play.clicked.connect(self.vp_player.play)
        self.btn_vp_pause.clicked.connect(self.vp_player.pause)
        self.btn_vp_stop.clicked.connect(self.vp_player.stop)
        
        for b in [self.btn_vp_play, self.btn_vp_pause, self.btn_vp_stop]:
            b.setFixedWidth(80)
            vp_btns.addWidget(b)
        vp_btns.addStretch()

        vpv.addWidget(self.video_widget)
        vpv.addLayout(vp_btns)
        rv.addWidget(self.video_group)

        log_group = QGroupBox("Log")
        logv = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 10))
        self.log_text.setStyleSheet("background:#171717;")
        logv.addWidget(self.log_text)
        rv.addWidget(log_group)

        root.addWidget(right)

    def update_audio_paths(self, paths: List[str]):
        self._audio_paths = sorted(paths, key=natural_sort_key)
        self._refresh_pairs()

    def update_image_paths(self, paths: List[str]):
        self._image_paths = list(paths)
        self._refresh_pairs()

    def get_images_per_audio(self) -> int:
        return 1

    def _on_layout_changed(self):
        self._refresh_pairs()
        main_win = self.window()
        if hasattr(main_win, "images_tab"):
            main_win.images_tab.update_parity(len(self._audio_paths), self.get_images_per_audio())

    def _can_pair(self, img_path):
        try:
            from PIL import Image
            with Image.open(img_path) as im:
                ratio = im.width / im.height
                if ratio < 0.4: return False  # Muito longo verticalmente
                if ratio > 1.2: return False  # Paisagem
                return True
        except:
            return False

    def _build_mixed_pairs(self):
        pairs = []
        a_idx = i_idx = 0
        counter = 0
        import random
        random.seed(len(self._audio_paths) + 123)
        while a_idx < len(self._audio_paths):
            can_split = (a_idx + 1 < len(self._audio_paths)) and (i_idx + 1 < len(self._image_paths))
            make_split = False
            
            if can_split:
                if counter >= random.randint(3, 5):
                    if self._can_pair(self._image_paths[i_idx]) and self._can_pair(self._image_paths[i_idx+1]):
                        make_split = True
                        
            if make_split:
                pairs.append((
                    (self._audio_paths[a_idx], self._audio_paths[a_idx+1]),
                    (self._image_paths[i_idx], self._image_paths[i_idx+1])
                ))
                a_idx += 2
                i_idx += 2
                counter = 0
            else:
                img = self._image_paths[i_idx] if i_idx < len(self._image_paths) else None
                if img: pairs.append((self._audio_paths[a_idx], img)); i_idx += 1
                a_idx += 1
                counter += 1
        return pairs

    def _refresh_pairs(self):
        self.pairs_list.clear()
        mode = self.layout_combo.currentData()
        
        if mode == "split":
            n_scenes = max((len(self._audio_paths) + 1) // 2, (len(self._image_paths) + 1) // 2)
            n_audios = len(self._audio_paths)
            n_imgs = len(self._image_paths)
            ok = (n_audios // 2 == n_scenes and n_imgs // 2 == n_scenes)

            for i in range(n_scenes):
                a1 = Path(self._audio_paths[2*i]).name if 2*i < n_audios else "[?]"
                a2 = Path(self._audio_paths[2*i+1]).name if 2*i+1 < n_audios else "[?]"
                i1 = Path(self._image_paths[2*i]).name if 2*i < n_imgs else "[?]"
                i2 = Path(self._image_paths[2*i+1]).name if 2*i+1 < n_imgs else "[?]"
                self.pairs_list.addItem(f"Cena {i+1}: ({a1}, {a2}) ↔ ({i1}, {i2})")
                
            icon = "✓" if ok else "⚠"
            color = "#6ddd6d" if ok else "#e05555"
            self.pairs_info.setText(
                f'<span style="color:{color};">{icon} {n_audios} áudios / {n_imgs} imagens (modo 2x)</span>'
            )
            return
            
        elif mode == "mixed":
            pairs = self._build_mixed_pairs()
            ok = len(pairs) > 0 and len(self._audio_paths) > 0
            
            for i, p in enumerate(pairs):
                if isinstance(p[0], tuple):
                    a1 = Path(p[0][0]).name; a2 = Path(p[0][1]).name
                    i1 = Path(p[1][0]).name; i2 = Path(p[1][1]).name
                    self.pairs_list.addItem(f"Cena {i+1} [Split]: ({a1}, {a2}) ↔ ({i1}, {i2})")
                else:
                    a1 = Path(p[0]).name
                    i1 = Path(p[1]).name if p[1] else "[?]"
                    self.pairs_list.addItem(f"Cena {i+1} [Single]: {a1} ↔ {i1}")
            
            icon = "✓" if ok else "⚠"
            color = "#6ddd6d" if ok else "#e05555"
            self.pairs_info.setText(f'<span style="color:{color};">{icon} Modo Misto</span>')
            return

        imgs_per_audio = self.get_images_per_audio()
        n_audios = len(self._audio_paths)
        n_imgs = len(self._image_paths)
        
        n = max(n_audios, (n_imgs + imgs_per_audio - 1) // imgs_per_audio)
        for i in range(n):
            a_idx = i
            a_str = Path(self._audio_paths[a_idx]).name if a_idx < n_audios else "[?]"
            i_strs = []
            for j in range(imgs_per_audio):
                i_idx = i * imgs_per_audio + j
                i_strs.append(Path(self._image_paths[i_idx]).name if i_idx < n_imgs else "[?]")
            self.pairs_list.addItem(f"{a_str}  ↔  {', '.join(i_strs)}")
            
        main_win = self.window()
        if hasattr(main_win, "images_tab"):
            matched, ready = main_win.images_tab.update_parity(n_audios, imgs_per_audio)
            if ok := ready:
                icon = "✓"
                self.pairs_info.setStyleSheet("color:#6ddd6d;")
            else:
                icon = "⚠"
                self.pairs_info.setStyleSheet("color:#e05555;")
            if n > 0:
                self.pairs_info.setText(
                    f"{icon} {matched}/{n} pares prontos. "
                    f"{'Pronto para gerar!' if ready else 'Alguns pares estão faltando.'}"
                )

    # ------------------------------------------------------------------
    # Carregamento por pasta — métodos auxiliares
    # ------------------------------------------------------------------
    _AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".flac", ".aac", ".m4a"}
    _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    def _load_folder_audio_image(self):
        """Carrega áudios E imagens de uma pasta (incluindo subpastas), emparelha em ordem."""
        folder = QFileDialog.getExistingDirectory(
            self, "Selecionar pasta com Áudios e Imagens"
        )
        if not folder:
            return
        p = Path(folder)
        audio_files = sorted(
            [str(f) for f in p.rglob("*") if f.is_file() and f.suffix.lower() in self._AUDIO_EXTS],
            key=natural_sort_key
        )
        image_files = sorted(
            [str(f) for f in p.rglob("*") if f.is_file() and f.suffix.lower() in self._IMAGE_EXTS],
            key=natural_sort_key
        )
        if not audio_files and not image_files:
            QMessageBox.information(self, "Pasta Vazia", "Nenhum arquivo de áudio ou imagem encontrado.")
            return
        self._audio_paths = audio_files
        self._image_paths = image_files
        # Sync ImagesTab se disponível
        main_win = self.window()
        if hasattr(main_win, "images_tab"):
            if hasattr(main_win.images_tab, "set_images"):
                main_win.images_tab.set_images(image_files)
        self._refresh_pairs()
        QMessageBox.information(
            self, "Pasta Carregada",
            f"✓  {len(audio_files)} áudio(s) e {len(image_files)} imagem(ns) carregados de:\n{folder}"
        )

    def _load_folder_audio_only(self):
        """Carrega apenas áudios de uma pasta (e subpastas)."""
        folder = QFileDialog.getExistingDirectory(
            self, "Selecionar pasta de Áudios"
        )
        if not folder:
            return
        p = Path(folder)
        audio_files = sorted(
            [str(f) for f in p.rglob("*") if f.is_file() and f.suffix.lower() in self._AUDIO_EXTS],
            key=natural_sort_key
        )
        if not audio_files:
            QMessageBox.information(self, "Sem Áudios", "Nenhum arquivo de áudio encontrado nessa pasta.")
            return
        self._audio_paths = audio_files
        self._refresh_pairs()
        QMessageBox.information(
            self, "Áudios Carregados",
            f"✓  {len(audio_files)} áudio(s) carregado(s) de:\n{folder}"
        )

    def _load_folder_images_only(self):
        """Carrega apenas imagens de uma pasta (e subpastas) e sincroniza."""
        folder = QFileDialog.getExistingDirectory(
            self, "Selecionar pasta de Imagens"
        )
        if not folder:
            return
        p = Path(folder)
        image_files = sorted(
            [str(f) for f in p.rglob("*") if f.is_file() and f.suffix.lower() in self._IMAGE_EXTS],
            key=natural_sort_key
        )
        if not image_files:
            QMessageBox.information(self, "Sem Imagens", "Nenhuma imagem encontrada nessa pasta.")
            return
        self._image_paths = image_files
        # Sync ImagesTab se disponível
        main_win = self.window()
        if hasattr(main_win, "images_tab"):
            if hasattr(main_win.images_tab, "set_images"):
                main_win.images_tab.set_images(image_files)
        self._refresh_pairs()
        QMessageBox.information(
            self, "Imagens Carregadas",
            f"✓  {len(image_files)} imagem(ns) carregada(s) de:\n{folder}"
        )

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Salvar vídeo como", "output/video.mp4", "MP4 (*.mp4)"
        )
        if path:
            self.output_path_edit.setText(path)

    def _browse_bgm(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Selecionar Música de Fundo", "", "Áudio (*.mp3 *.wav *.ogg *.flac *.m4a *.aac)"
        )
        if path:
            self.bgm_path_edit.setText(path)

    def _start_preview(self):
        self._start_generation(is_preview=True)

    def _start_generation(self, is_preview=False):
        mode = self.layout_combo.currentData()
        if mode == "split":
            n = min(len(self._audio_paths) // 2, len(self._image_paths) // 2)
            if n == 0:
                QMessageBox.warning(self, "Sem pares completos",
                    "Carregue áudios (Aba 1) e imagens (Aba 2) em pares (2 áudios e 2 imagens por cena).")
                return
            pairs = [((self._audio_paths[2*i], self._audio_paths[2*i+1]), (self._image_paths[2*i], self._image_paths[2*i+1])) for i in range(n)]
        elif mode == "mixed":
            pairs = self._build_mixed_pairs()
            if not pairs:
                QMessageBox.warning(self, "Sem pares completos", "Pares insuficientes para modo misto.")
                return
        else:
            n = min(len(self._audio_paths), len(self._image_paths))
            if n == 0:
                QMessageBox.warning(self, "Sem pares",
                    "Carregue áudios (Aba 1) e imagens (Aba 2) primeiro.")
                return
            pairs = [(self._audio_paths[i], self._image_paths[i]) for i in range(n)]
        # Derivar pasta de saída do mesmo local que os áudios
        output_path = self.output_path_edit.text().strip()
        if not output_path:
            if self._audio_paths:
                audio_dir = Path(self._audio_paths[0]).parent
                # audios ficam em .../proj/audios/ → subir um nível para .../proj/
                proj_dir = audio_dir.parent if audio_dir.name == "audios" else audio_dir
                proj_name = proj_dir.name or "video"
                output_path = str(proj_dir / f"{proj_name}_final.mp4")
            else:
                output_path = "output/video.mp4"
        # Preview Mode: Limita severamente o lote as 3 primeiras cenas e renomeia
        if is_preview:
            pairs = pairs[:3]
            if not pairs:
                QMessageBox.warning(self, "Sem pares", "Não há cenas suficientes para preview.")
                return
            p_out = Path(output_path)
            output_path = str(p_out.with_name(f"{p_out.stem}_preview{p_out.suffix}"))
            self.log_text.append("👀 MODO PREVIEW ATIVADO: Gerando apenas as primeiras 3 cenas!")
            self._last_preview_path = output_path
        else:
            self._last_preview_path = None

        effect_mode = self.effect_combo.currentData() or "auto"
        transition_mode = self.transition_combo.currentData() or "fade"
        transition_time = self.transition_time_spin.value()

        self.btn_generate.setEnabled(False)
        self.btn_preview.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        if not is_preview:
            self.log_text.clear()
        self.progress_bar.setValue(0)

        bgm_path = self.bgm_path_edit.text().strip()
        bgm_vol = self.bgm_vol_spin.value()

        from manhwa_app.video_pipeline import VideoPipeline
        self._pipeline = VideoPipeline(pairs=pairs, output_path=output_path,
                                       effect_mode=effect_mode,
                                       transition_mode=transition_mode,
                                       transition_time=transition_time,
                                       bg_music_path=bgm_path,
                                       bg_music_volume=bgm_vol,
                                       config=self.get_session())
        self._worker_thread = QThread(self)
        self._pipeline.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._pipeline.run)
        self._pipeline.log_message.connect(self._on_log)
        self._pipeline.progress.connect(self._on_progress)
        self._pipeline.finished.connect(self._on_done)
        self._worker_thread.start()

    def start_run_all_video(self, project_name: str, audio_paths: List[str]):
        """Inicia a geração de vídeo automaticamente (usado pelo Run All)."""
        self.update_audio_paths(audio_paths)
        if not self._image_paths:
            self._on_log("⚠ Nenhuma imagem encontrada. Abortando vídeo.")
            self._on_done(False, "Nenhuma imagem encontrada na Aba 2.")
            return

        mode = self.layout_combo.currentData()
        if mode == "split":
            n = min(len(self._audio_paths) // 2, len(self._image_paths) // 2)
            pairs = [((self._audio_paths[2*i], self._audio_paths[2*i+1]), (self._image_paths[2*i], self._image_paths[2*i+1])) for i in range(n)]
        elif mode == "mixed":
            pairs = self._build_mixed_pairs()
        else:
            n = min(len(self._audio_paths), len(self._image_paths))
            pairs = [(self._audio_paths[i], self._image_paths[i]) for i in range(n)]
        
        # Pega a pasta de saída do áudio. O MainWindow já configurou o `output_root` no audio_tab.
        audio_root = self.window().audio_tab.output_root_edit.text().strip() or "output"
        out_mp4 = Path(audio_root) / project_name / f"{project_name}_final.mp4"
        self.output_path_edit.setText(str(out_mp4))
        
        effect_mode = self.effect_combo.currentData() or "auto"
        
        self.btn_generate.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.log_text.clear()
        self.progress_bar.setValue(0)
        self._on_log(f"=== RUN ALL: Iniciando Vídeo para {project_name} ===")

        bgm_path = self.bgm_path_edit.text().strip()
        bgm_vol = self.bgm_vol_spin.value()

        from manhwa_app.video_pipeline import VideoPipeline
        self._pipeline = VideoPipeline(pairs=pairs, output_path=str(out_mp4),
                                       effect_mode=effect_mode,
                                       bg_music_path=bgm_path,
                                       bg_music_volume=bgm_vol,
                                       config=self.get_session())
        self._worker_thread = QThread(self)
        self._pipeline.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._pipeline.run)
        self._pipeline.log_message.connect(self._on_log)
        self._pipeline.progress.connect(self._on_progress)
        self._pipeline.finished.connect(self._on_done_run_all)
        self._worker_thread.start()

    @Slot(bool, str)
    def _on_done_run_all(self, success: bool, message: str):
        if self._worker_thread:
            self._worker_thread.quit()
            self._worker_thread.wait()
            self._worker_thread.deleteLater()
            self._worker_thread = None
            
        if self._pipeline:
            self._pipeline.deleteLater()
            self._pipeline = None
            
        self.btn_generate.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        if success:
            self.progress_bar.setValue(100)
            self._on_log(f"✓ Vídeo gerado: {message}")
        else:
            self._on_log(f"✗ Falha no vídeo: {message}")
        
        # Avisa a AudioTab para continuar o loop
        self.window().audio_tab.continue_run_all()

    @Slot(str)
    def _on_log(self, msg: str):
        _append_log(self.log_text, msg)

    @Slot(int, int)
    def _on_progress(self, current: int, total: int):
        if total > 0:
            pct = int(current * 100 / total)
            self.progress_bar.setValue(pct)
            self.progress_bar.setFormat(f"Clip {current}/{total} — {pct}%")

    @Slot(bool, str)
    def _on_done(self, success: bool, message: str):
        if self._worker_thread:
            self._worker_thread.quit()
            self._worker_thread.wait()
            self._worker_thread.deleteLater()
            self._worker_thread = None
            
        if self._pipeline:
            self._pipeline.deleteLater()
            self._pipeline = None
            
        self.btn_generate.setEnabled(True)
        self.btn_preview.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        
        if success:
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("Concluído!")
            
            # Toca o Preview automaticamente se foi ativado
            if self._last_preview_path and Path(self._last_preview_path).exists():
                from PySide6.QtCore import QUrl
                self.vp_player.setSource(QUrl.fromLocalFile(self._last_preview_path))
                self.vp_player.play()
                self._on_log(f"▶ Reproduzindo Preview na janela: {Path(self._last_preview_path).name}")
            else:
                QMessageBox.information(self, "Vídeo Pronto", f"Vídeo salvo em:\n{message}")
        else:
            QMessageBox.critical(self, "Falha na geração", message)

    def _cancel(self):
        if self._pipeline:
            self._pipeline.cancel()
        self.btn_cancel.setEnabled(False)

    def suggest_output_path(self, project: str, output_root: str):
        """Sugerir caminho de saída com base no nome do projeto."""
        if project and not self.output_path_edit.text().strip():
            p = Path(output_root) / project / f"{project}_final.mp4"
            self.output_path_edit.setText(str(p))

    def get_session(self) -> dict:
        return {
            "layout": self.layout_combo.currentData(),
            "effect": self.effect_combo.currentData(),
            "transition": self.transition_combo.currentData(),
            "transition_time": self.transition_time_spin.value(),
            "bg_music_path": self.bgm_path_edit.text().strip() if hasattr(self, "bgm_path_edit") else "",
            "bg_music_volume": self.bgm_vol_spin.value() if hasattr(self, "bgm_vol_spin") else 10,
            "production": {
                "video": {
                    # CORRIGIDO: getattr(self, "chk_x", QCheckBox()).isChecked() criava widget descartável
                    # que sempre retornava False. Usar hasattr + fallback True para valores de qualidade
                    "better_easing":  self.chk_better_easing.isChecked() if hasattr(self, "chk_better_easing") else True,
                    "color_grading":  self.chk_color_grading.isChecked() if hasattr(self, "chk_color_grading") else True,
                    "sharpen":        self.chk_sharpen.isChecked() if hasattr(self, "chk_sharpen") else True,
                    "film_grain":     self.chk_grain.isChecked() if hasattr(self, "chk_grain") else False,
                },
                "sound_design": {
                    "auto_ducking":   self.chk_auto_ducking.isChecked() if hasattr(self, "chk_auto_ducking") else False,
                }
            }
        }

    def load_session(self, data: dict):
        if "layout" in data:
            idx = self.layout_combo.findData(data["layout"])
            if idx >= 0: self.layout_combo.setCurrentIndex(idx)
        if "effect" in data:
            idx = self.effect_combo.findData(data["effect"])
            if idx >= 0: self.effect_combo.setCurrentIndex(idx)
        if "transition" in data:
            idx = self.transition_combo.findData(data["transition"])
            if idx >= 0: self.transition_combo.setCurrentIndex(idx)
        if "transition_time" in data:
            self.transition_time_spin.setValue(data["transition_time"])
        if hasattr(self, "bgm_path_edit") and "bg_music_path" in data:
            self.bgm_path_edit.setText(data["bg_music_path"])
        if hasattr(self, "bgm_vol_spin") and "bg_music_volume" in data:
            self.bgm_vol_spin.setValue(data["bg_music_volume"])
            
        prod = data.get("production", {})
        vid = prod.get("video", {})
        snd = prod.get("sound_design", {})
        
        if hasattr(self, "chk_better_easing"):
            self.chk_better_easing.setChecked(vid.get("better_easing", True))
        if hasattr(self, "chk_color_grading"):
            self.chk_color_grading.setChecked(vid.get("color_grading", True))
        if hasattr(self, "chk_sharpen"):
            self.chk_sharpen.setChecked(vid.get("sharpen", True))
        if hasattr(self, "chk_grain"):
            self.chk_grain.setChecked(vid.get("film_grain", False))
        if hasattr(self, "chk_auto_ducking"):
            self.chk_auto_ducking.setChecked(snd.get("auto_ducking", True))

    def reset_defaults(self):
        self.layout_combo.setCurrentIndex(0)
        self.effect_combo.setCurrentIndex(0)
        self.transition_combo.setCurrentIndex(0)
        self.transition_time_spin.setValue(0.2)
        self.output_path_edit.setText("")
        if hasattr(self, "bgm_path_edit"):
            self.bgm_path_edit.setText("")
        if hasattr(self, "bgm_vol_spin"):
            self.bgm_vol_spin.setValue(10)
        
        if hasattr(self, "chk_better_easing"):
            self.chk_better_easing.setChecked(True)
        if hasattr(self, "chk_color_grading"):
            self.chk_color_grading.setChecked(True)
        if hasattr(self, "chk_sharpen"):
            self.chk_sharpen.setChecked(True)
        if hasattr(self, "chk_grain"):
            self.chk_grain.setChecked(False)
        if hasattr(self, "chk_auto_ducking"):
            self.chk_auto_ducking.setChecked(True)


# ---------------------------------------------------------------------------
# Aba de Configurações
# ---------------------------------------------------------------------------

_DEFAULT_REVISION_PROMPT = """Você é um editor profissional especializado em roteiros de manhwa e webtoon narrados em áudio.

Sua tarefa: revisar APENAS os parágrafos do BLOCO ATUAL.

REGRAS OBRIGATÓRIAS:
1. Mantenha o sentido e a narrativa original — não reescreva, apenas corrija e melhore fluidez
2. Cada parágrafo deve terminar de forma que o próximo continue naturalmente (coesão narrativa)
3. Remova ou substitua: símbolos especiais (*, #, →, —), reticências excessivas (... ... ...), onomatopeias escritas (BOOM!, POW!), parênteses com indicações de cena
4. Escreva por extenso: números isolados vire palavras ("3" → "três"), siglas ambíguas explique ("km" → "quilômetros")
5. Frases muito longas (mais de 40 palavras): quebre em duas frases naturais
6. Não adicione frases novas — apenas refine o que existe
7. Mantenha o mesmo número de parágrafos que recebeu

CONTEXTO (não edite, apenas leia para entender a continuidade):
[ANTES]: {context_before}
[DEPOIS]: {context_after}

BLOCO ATUAL para revisar:
{current_block_formatted}

Responda SOMENTE com JSON válido, sem markdown, sem explicações:
{{"paragrafos": ["texto revisado 1", "texto revisado 2", ...]}}

O array deve ter exatamente {n} elementos, na mesma ordem dos parágrafos recebidos."""

_DEFAULT_TRANSLATION_PROMPT = """Você é um tradutor profissional especializado em manhwa e webtoon para narração em áudio.

Idioma de destino: {language_name}
Idioma de origem: português brasileiro

REGRAS OBRIGATÓRIAS:
1. Traduza com naturalidade no idioma alvo — não traduza literalmente palavra por palavra
2. Mantenha o tom narrativo, a emoção e o ritmo do texto original
3. Nomes próprios de personagens: mantenha como estão (não traduza nomes)
4. A tradução deve soar como narração de áudio, não como texto escrito
5. Cada parágrafo deve fluir para o próximo (mesma coesão do original)
6. Mantenha o mesmo número de parágrafos

CONTEXTO em português (não traduza, só leia para entender a continuidade):
[ANTES]: {context_before}
[DEPOIS]: {context_after}

BLOCO ATUAL em português para traduzir:
{current_block_formatted}

Responda SOMENTE com JSON válido, sem markdown, sem explicações:
{{"paragrafos": ["texto traduzido 1", "texto traduzido 2", ...]}}

O array deve ter exatamente {n} elementos, na mesma ordem dos parágrafos recebidos."""


class SettingsTab(QWidget):
    """Aba de configurações gerais: Gemini, aparência, paths e sessão."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Wrap everything in a scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        vl = QVBoxLayout(inner)
        vl.setContentsMargins(24, 20, 24, 24)
        vl.setSpacing(18)
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        # ── 1. Gemini API ─────────────────────────────────────────────
        gem_box = QGroupBox("🔑  Gemini API")
        gg = QGridLayout(gem_box)
        gg.setSpacing(10)

        gg.addWidget(QLabel("Modelo:"), 0, 0)
        self.model_combo = QComboBox()
        # Modelos Solicitados
        self.model_combo.addItem("Gemini 2.5 Flash (🔥 Recomendado)", "gemini-2.5-flash")
        self.model_combo.addItem("Gemini 3 Flash", "gemini-3-flash-preview")
        self.model_combo.addItem("Gemini 3.1 Flash", "gemini-3.1-flash-lite-preview")

        self.model_combo.setEditable(False)
        self.model_combo.setMinimumHeight(40)
        gg.addWidget(self.model_combo, 0, 1, 1, 3) # Ocupar mais espaço



        self.btn_fetch_models = QPushButton("🔄 Atualizar Lista")
        self.btn_fetch_models.setToolTip("Busca modelos disponíveis para sua chave API no Google")
        self.btn_fetch_models.clicked.connect(self._fetch_models)
        gg.addWidget(self.btn_fetch_models, 0, 3)


        gg.addWidget(QLabel("API Key:"), 1, 0)
        self.api_key_combo = QComboBox()
        self.api_key_combo.setEditable(True)
        self.api_key_combo.lineEdit().setPlaceholderText("Cole ou digite sua API Key...")
        self.api_key_combo.lineEdit().setEchoMode(QLineEdit.EchoMode.Password)
        
        kw = QWidget()
        kh = QHBoxLayout(kw)
        kh.setContentsMargins(0, 0, 0, 0)
        kh.setSpacing(5)
        kh.addWidget(self.api_key_combo, 1)

        self.btn_save_key = QPushButton("💾")
        self.btn_save_key.setToolTip("Salvar chave atual na lista")
        self.btn_save_key.setFixedWidth(34)
        self.btn_save_key.clicked.connect(self._save_api_key)
        kh.addWidget(self.btn_save_key)

        self.btn_rem_key = QPushButton("🗑")
        self.btn_rem_key.setToolTip("Remover chave atual da lista")
        self.btn_rem_key.setFixedWidth(34)
        self.btn_rem_key.clicked.connect(self._remove_api_key)
        kh.addWidget(self.btn_rem_key)

        gg.addWidget(kw, 1, 1, 1, 2)

        self._btn_eye = QPushButton("👁")
        self._btn_eye.setFixedWidth(34)
        self._btn_eye.setCheckable(True)
        self._btn_eye.setToolTip("Mostrar/ocultar chave")
        self._btn_eye.toggled.connect(
            lambda on: self.api_key_combo.lineEdit().setEchoMode(
                QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password)
        )
        gg.addWidget(self._btn_eye, 1, 3)

        self.btn_test = QPushButton("🔗  Testar Conexão")
        self.btn_test.setObjectName("primary")
        self.btn_test.clicked.connect(self._test_connection)
        gg.addWidget(self.btn_test, 2, 1)

        self.test_result_lbl = QLabel("")
        self.test_result_lbl.setWordWrap(True)
        gg.addWidget(self.test_result_lbl, 2, 2, 1, 2)
        vl.addWidget(gem_box)

        # ── 2. Comportamento Gemini ───────────────────────────────────
        behav_box = QGroupBox("⚙  Comportamento Gemini")
        bg = QGridLayout(behav_box)
        bg.setSpacing(10)

        bg.addWidget(QLabel("Delay entre chamadas:"), 0, 0)
        self.delay_slider = QSlider(Qt.Orientation.Horizontal)
        self.delay_slider.setRange(2, 30)
        self.delay_slider.setValue(4)
        self.delay_lbl = QLabel("4 s")
        self.delay_slider.valueChanged.connect(lambda v: self.delay_lbl.setText(f"{v} s"))
        bg.addWidget(self.delay_slider, 0, 1)
        bg.addWidget(self.delay_lbl, 0, 2)

        bg.addWidget(QLabel("Idiomas padrão:"), 1, 0)
        lang_w = QWidget()
        lh = QHBoxLayout(lang_w)
        lh.setContentsMargins(0, 0, 0, 0)
        self.chk_en = QCheckBox("Inglês (en)")
        self.chk_es = QCheckBox("Espanhol (es)")
        self.chk_fr = QCheckBox("Francês (fr)")
        self.chk_de = QCheckBox("Alemão (de)")
        self.chk_ja = QCheckBox("Japonês (ja)")
        self.chk_ko = QCheckBox("Coreano (ko)")
        for chk in [self.chk_en, self.chk_es, self.chk_fr, self.chk_de, self.chk_ja, self.chk_ko]:
            lh.addWidget(chk)
        lh.addStretch()
        bg.addWidget(lang_w, 1, 1, 1, 2)

        bg.addWidget(QLabel("Tamanho de chunk:"), 2, 0)
        self.chunk_spin = QSpinBox()
        self.chunk_spin.setRange(4, 40)
        self.chunk_spin.setValue(12)
        self.chunk_spin.setToolTip("Parágrafos por chamada à API (padrão: 12)")
        bg.addWidget(self.chunk_spin, 2, 1)

        bg.addWidget(QLabel("Overlap (contexto):"), 3, 0)
        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 6)
        self.overlap_spin.setValue(2)
        self.overlap_spin.setToolTip("Parágrafos de contexto antes/depois de cada chunk (padrão: 2)")
        bg.addWidget(self.overlap_spin, 3, 1)

        # Thinking Level (Gemini 3+)
        bg.addWidget(QLabel("Nível de Raciocínio (Gemini 3):"), 4, 0)
        self.thinking_combo = QComboBox()
        self.thinking_combo.addItem("high (Padrão, Máximo Raciocínio)", "high")
        self.thinking_combo.addItem("medium (Equilibrado)", "medium")
        self.thinking_combo.addItem("low (Menor Latência)", "low")
        self.thinking_combo.addItem("minimal (Quase sem pensamento)", "minimal")
        self.thinking_combo.setToolTip("Controla a profundidade do raciocínio interno do modelo.")
        bg.addWidget(self.thinking_combo, 4, 1, 1, 2)

        # Media Resolution (Gemini 3+ Vision)
        bg.addWidget(QLabel("Resolução de Mídia (Visão):"), 5, 0)
        self.media_res_combo = QComboBox()
        self.media_res_combo.addItem("media_resolution_high (Recomendado Imagens)", "media_resolution_high")
        self.media_res_combo.addItem("media_resolution_medium (Recomendado PDFs)", "media_resolution_medium")
        self.media_res_combo.addItem("media_resolution_low (Recomendado Vídeo)", "media_resolution_low")
        self.media_res_combo.addItem("media_resolution_ultra_high (Detalhes Extremos)", "media_resolution_ultra_high")
        self.media_res_combo.setToolTip("Determina o detalhamento no processamento de imagens/vídeos.")
        bg.addWidget(self.media_res_combo, 5, 1, 1, 2)

        vl.addWidget(behav_box)

        # ── 3. Prompt de Revisão ──────────────────────────────────────
        rev_box = QGroupBox("📝  Prompt de Revisão  (variáveis: {context_before} {context_after} {current_block_formatted} {n})")
        rv = QVBoxLayout(rev_box)
        self.revision_prompt_edit = QTextEdit()
        self.revision_prompt_edit.setPlainText(_DEFAULT_REVISION_PROMPT)
        self.revision_prompt_edit.setMinimumHeight(200)
        self.revision_prompt_edit.setFont(QFont("Consolas", 10))
        rv.addWidget(self.revision_prompt_edit)
        btn_reset_rev = QPushButton("↺  Restaurar Prompt Padrão")
        btn_reset_rev.clicked.connect(
            lambda: self.revision_prompt_edit.setPlainText(_DEFAULT_REVISION_PROMPT))
        rv.addWidget(btn_reset_rev)
        vl.addWidget(rev_box)

        # ── 4. Prompt de Tradução ─────────────────────────────────────
        tr_box = QGroupBox("🌐  Prompt de Tradução  (variáveis: {language_name} {context_before} {context_after} {current_block_formatted} {n})")
        tv = QVBoxLayout(tr_box)
        self.translation_prompt_edit = QTextEdit()
        self.translation_prompt_edit.setPlainText(_DEFAULT_TRANSLATION_PROMPT)
        self.translation_prompt_edit.setMinimumHeight(200)
        self.translation_prompt_edit.setFont(QFont("Consolas", 10))
        tv.addWidget(self.translation_prompt_edit)
        btn_reset_tr = QPushButton("↺  Restaurar Prompt Padrão")
        btn_reset_tr.clicked.connect(
            lambda: self.translation_prompt_edit.setPlainText(_DEFAULT_TRANSLATION_PROMPT))
        tv.addWidget(btn_reset_tr)
        vl.addWidget(tr_box)

        # ── 5. Cache Gemini ───────────────────────────────────────────
        cache_box = QGroupBox("🗑  Cache Gemini")
        ch = QHBoxLayout(cache_box)
        self.cache_size_lbl = QLabel("Calculando...")
        ch.addWidget(self.cache_size_lbl, 1)
        btn_refresh_cache = QPushButton("🔄 Atualizar")
        btn_refresh_cache.clicked.connect(self._refresh_cache_info)
        ch.addWidget(btn_refresh_cache)
        btn_clear_cache = QPushButton("🗑  Limpar Cache")
        btn_clear_cache.setObjectName("danger")
        btn_clear_cache.clicked.connect(self._clear_cache)
        ch.addWidget(btn_clear_cache)
        vl.addWidget(cache_box)
        self._refresh_cache_info()

        # ── 6. Aparência ──────────────────────────────────────────────
        appear_box = QGroupBox("🎨  Aparência")
        ag = QGridLayout(appear_box)
        ag.setSpacing(10)

        ag.addWidget(QLabel("Tema:"), 0, 0)
        self.theme_combo = QComboBox()
        for name in THEMES:
            self.theme_combo.addItem(name)
        self.theme_combo.setMinimumHeight(32)
        self.theme_combo.currentTextChanged.connect(self._on_theme_changed)
        ag.addWidget(self.theme_combo, 0, 1, 1, 3)

        ag.addWidget(QLabel("Imagem de Fundo:"), 1, 0)
        self.bg_path_edit = QLineEdit()
        self.bg_path_edit.setPlaceholderText("(nenhuma)")
        self.bg_path_edit.setReadOnly(True)
        ag.addWidget(self.bg_path_edit, 1, 1)
        btn_bg = QPushButton("📁 Selecionar")
        btn_bg.clicked.connect(self._browse_bg)
        ag.addWidget(btn_bg, 1, 2)
        btn_clear_bg = QPushButton("✖")
        btn_clear_bg.setObjectName("danger")
        btn_clear_bg.setFixedWidth(32)
        btn_clear_bg.clicked.connect(self._clear_bg)
        ag.addWidget(btn_clear_bg, 1, 3)

        ag.addWidget(QLabel("Opacidade do Fundo:"), 2, 0)
        self.overlay_slider = QSlider(Qt.Orientation.Horizontal)
        self.overlay_slider.setRange(0, 240)
        self.overlay_slider.setValue(180)
        self.overlay_lbl = QLabel("180")
        self.overlay_slider.valueChanged.connect(
            lambda v: (self.overlay_lbl.setText(str(v)), self._sync_bg_to_main()))
        ag.addWidget(self.overlay_slider, 2, 1)
        ag.addWidget(self.overlay_lbl, 2, 2)
        vl.addWidget(appear_box)

        # ── 7. Paths Padrão ───────────────────────────────────────────
        paths_box = QGroupBox("📁  Paths Padrão")
        pg = QGridLayout(paths_box)
        pg.setSpacing(10)

        pg.addWidget(QLabel("Pasta de Saída Padrão:"), 0, 0)
        self.default_output_edit = QLineEdit("output")
        pg.addWidget(self.default_output_edit, 0, 1)
        btn_out = QPushButton("📁")
        btn_out.setFixedWidth(34)
        btn_out.clicked.connect(lambda: self._browse_dir(self.default_output_edit))
        pg.addWidget(btn_out, 0, 2)

        pg.addWidget(QLabel("Pasta de Vozes Padrão:"), 1, 0)
        self.default_voices_edit = QLineEdit("voices")
        pg.addWidget(self.default_voices_edit, 1, 1)
        btn_voi = QPushButton("📁")
        btn_voi.setFixedWidth(34)
        btn_voi.clicked.connect(lambda: self._browse_dir(self.default_voices_edit))
        pg.addWidget(btn_voi, 1, 2)
        vl.addWidget(paths_box)

        # ── 8. Sessão ─────────────────────────────────────────────────
        sess_box = QGroupBox("💾  Sessão")
        sh = QHBoxLayout(sess_box)
        btn_export = QPushButton("📤  Exportar Configurações")
        btn_export.clicked.connect(self._export_session)
        sh.addWidget(btn_export)
        btn_import = QPushButton("📥  Importar Configurações")
        btn_import.clicked.connect(self._import_session)
        sh.addWidget(btn_import)
        sh.addStretch()
        vl.addWidget(sess_box)

        vl.addStretch()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def _save_api_key(self):
        text = self.api_key_combo.currentText().strip()
        if text and self.api_key_combo.findText(text) == -1:
            self.api_key_combo.addItem(text)
            self.api_key_combo.setCurrentText(text)

    def _remove_api_key(self):
        idx = self.api_key_combo.currentIndex()
        if idx >= 0:
            self.api_key_combo.removeItem(idx)

    def get_all_api_keys(self) -> list:
        return [self.api_key_combo.itemText(i) for i in range(self.api_key_combo.count())]

    def get_api_key(self) -> str:
        return self.api_key_combo.currentText().strip()

    def get_model_name(self) -> str:
        return self.model_combo.currentData() or self.model_combo.currentText().strip()

    def get_delay(self) -> float:
        return float(self.delay_slider.value())

    def get_languages(self) -> list:
        langs = []
        mapping = [
            (self.chk_en, "en"), (self.chk_es, "es"), (self.chk_fr, "fr"),
            (self.chk_de, "de"), (self.chk_ja, "ja"), (self.chk_ko, "ko"),
        ]
        for chk, code in mapping:
            if chk.isChecked():
                langs.append(code)
        return langs

    def get_revision_prompt(self) -> str:
        return self.revision_prompt_edit.toPlainText()

    def get_translation_prompt(self) -> str:
        return self.translation_prompt_edit.toPlainText()

    def get_chunk_size(self) -> int:
        return self.chunk_spin.value()

    def get_overlap(self) -> int:
        return self.overlap_spin.value()

    def get_thinking_level(self) -> str:
        return self.thinking_combo.currentData()

    def get_media_resolution(self) -> str:
        return self.media_res_combo.currentData()


    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------

    def get_session(self) -> dict:
        return {
            "gemini": {
                "api_key": self.get_api_key(),
                "api_keys": self.get_all_api_keys(),
                "model_name": self.get_model_name(),
                "delay": self.delay_slider.value(),
                "languages": self.get_languages(),
                "chunk_size": self.chunk_spin.value(),
                "overlap": self.overlap_spin.value(),
                "thinking_level": self.get_thinking_level(),
                "media_resolution": self.get_media_resolution(),
                "revision_prompt": self.get_revision_prompt(),
                "translation_prompt": self.get_translation_prompt(),
                "language_mapping": getattr(self, "language_mapping", {})
            },
            "appearance": {
                "theme": self.theme_combo.currentText(),
                "bg_overlay_alpha": self.overlay_slider.value(),
            },
            "paths": {
                "default_output": self.default_output_edit.text().strip(),
                "default_voices": self.default_voices_edit.text().strip(),
            }
        }

    def load_session(self, data: dict):
        gem = data.get("gemini", {})
        self.language_mapping = gem.get("language_mapping", {})
        if "api_keys" in gem:
            self.api_key_combo.clear()
            self.api_key_combo.addItems(gem["api_keys"])
        if "api_key" in gem:
            idx = self.api_key_combo.findText(gem["api_key"])
            if idx >= 0:
                self.api_key_combo.setCurrentIndex(idx)
            else:
                self.api_key_combo.setCurrentText(gem["api_key"])
        if "model_name" in gem:
            idx = self.model_combo.findData(gem["model_name"])
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
            else:
                self.model_combo.setCurrentText(gem["model_name"])
        if "delay" in gem:
            self.delay_slider.setValue(int(gem["delay"]))
        if "chunk_size" in gem:
            self.chunk_spin.setValue(int(gem["chunk_size"]))
        if "overlap" in gem:
            self.overlap_spin.setValue(int(gem["overlap"]))
        if "thinking_level" in gem:
            idx = self.thinking_combo.findData(gem["thinking_level"])
            if idx >= 0: self.thinking_combo.setCurrentIndex(idx)
        if "media_resolution" in gem:
            idx = self.media_res_combo.findData(gem["media_resolution"])
            if idx >= 0: self.media_res_combo.setCurrentIndex(idx)
        if "revision_prompt" in gem:
            self.revision_prompt_edit.setPlainText(gem["revision_prompt"])
        if "translation_prompt" in gem:
            self.translation_prompt_edit.setPlainText(gem["translation_prompt"])

        if "languages" in gem:
            mapping = {"en": self.chk_en, "es": self.chk_es, "fr": self.chk_fr,
                       "de": self.chk_de, "ja": self.chk_ja, "ko": self.chk_ko}
            for code, chk in mapping.items():

                chk.setChecked(code in gem["languages"])
        if "chunk_size" in gem:
            self.chunk_spin.setValue(int(gem["chunk_size"]))
        if "overlap" in gem:
            self.overlap_spin.setValue(int(gem["overlap"]))
        if "revision_prompt" in gem:
            self.revision_prompt_edit.setPlainText(gem["revision_prompt"])
        if "translation_prompt" in gem:
            self.translation_prompt_edit.setPlainText(gem["translation_prompt"])

        app_ = data.get("appearance", {})
        if "theme" in app_:
            self.theme_combo.setCurrentText(app_["theme"])
        if "bg_overlay_alpha" in app_:
            self.overlay_slider.setValue(int(app_["bg_overlay_alpha"]))

        paths_ = data.get("paths", {})
        if "default_output" in paths_:
            self.default_output_edit.setText(paths_["default_output"])
        if "default_voices" in paths_:
            self.default_voices_edit.setText(paths_["default_voices"])

    def reset_defaults(self):
        self.api_key_edit.clear()
        self.model_combo.setCurrentIndex(0)
        self.delay_slider.setValue(4)
        for chk in [self.chk_en, self.chk_es, self.chk_fr, self.chk_de, self.chk_ja, self.chk_ko]:
            chk.setChecked(False)
        self.chunk_spin.setValue(12)
        self.overlap_spin.setValue(2)
        self.thinking_combo.setCurrentIndex(0)
        self.media_res_combo.setCurrentIndex(0)
        self.revision_prompt_edit.setPlainText(_DEFAULT_REVISION_PROMPT)

        self.translation_prompt_edit.setPlainText(_DEFAULT_TRANSLATION_PROMPT)
        self.theme_combo.setCurrentText("🌑 Dark (Padrão)")
        self.overlay_slider.setValue(180)
        self.default_output_edit.setText("output")
        self.default_voices_edit.setText("voices")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _test_connection(self):
        key = self.get_api_key()
        model = self.get_model_name()
        if not key:
            self.test_result_lbl.setText("❌ Insira uma API Key antes de testar.")
            self.test_result_lbl.setStyleSheet("color:#e05555;")
            return
        if not _GEMINI_AVAILABLE:
            self.test_result_lbl.setText("❌ google-genai não instalado: pip install google-genai")
            self.test_result_lbl.setStyleSheet("color:#e05555;")
            return
        self.btn_test.setEnabled(False)
        self.test_result_lbl.setText("⏳ Testando conexão...")
        self.test_result_lbl.setStyleSheet("color:#f0c040;")

        class _TestWorker(QThread):
            done = Signal(bool, str)
            def __init__(self, k, m, t, r):
                super().__init__()
                self._k = k
                self._m = m
                self._t = t
                self._r = r
            def run(self):
                try:
                    from google import genai as _genai
                    from google.genai import types as _types
                    client = _genai.Client(api_key=self._k)
                    
                    config = None
                    if self._t and "gemini-3" in self._m:
                        config = _types.GenerateContentConfig(
                            thinking_config=_types.ThinkingConfig(thinking_level=self._t)
                        )
                    
                    client.models.generate_content(model=self._m, contents="ok", config=config)
                    self.done.emit(True, f"✅ Conectado ao modelo {self._m} com sucesso!")

                except Exception as e:
                    self.done.emit(False, f"❌ {e}")

        think = self.get_thinking_level()
        res = self.get_media_resolution()
        self._test_worker = _TestWorker(key, model, think, res)

        self._test_worker.done.connect(self._on_test_done)
        self._test_worker.start()

    def _fetch_models(self):
        key = self.api_key_edit.text().strip()
        if not key:
            QMessageBox.warning(self, "Sem Chave", "Insira uma API Key para buscar os modelos.")
            return
        
        self.btn_fetch_models.setEnabled(False)
        self.btn_fetch_models.setText("Buscando...")
        self.test_result_lbl.setText("⏳ Buscando lista de modelos...")

        self._fetch_worker = GeminiWorker(key, "list_models")
        self._fetch_worker.finished.connect(self._on_fetch_done)
        self._fetch_worker.error.connect(self._on_fetch_error)
        self._fetch_worker.start()

    def _on_fetch_done(self, data):
        self.btn_fetch_models.setEnabled(True)
        self.btn_fetch_models.setText("🔄 Atualizar Lista")
        
        if not isinstance(data, list):
             self.test_result_lbl.setText("❌ Resposta inesperada ao buscar modelos.")
             return

        
        current_model = self.get_model_name()
        self.model_combo.clear()
        
        # Priorizar modelos úteis
        important = ["flash", "pro", "thinking"]
        
        for m_id in data:
            name = m_id.replace("models/", "")
            desc = name
            if "flash" in name: desc += " (Equilibrado)"
            if "pro" in name: desc += " (Avançado)"
            if "lite" in name: desc += " (Poupador)"
            
            self.model_combo.addItem(desc, name)
            
        # Tentar restaurar o modelo que estava selecionado
        idx = self.model_combo.findData(current_model)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        else:
            self.model_combo.setCurrentText(current_model)
            
        self.test_result_lbl.setText(f"✅ total de {len(data)} modelos carregados.")
        
        # Notificar MainWindow para atualizar o combo da aba Áudio
        main_win = self.window()
        if hasattr(main_win, "audio_tab") and hasattr(main_win.audio_tab, "_gemini_model_combo"):
             main_win.audio_tab._gemini_model_combo.clear()
             for i in range(self.model_combo.count()):
                 main_win.audio_tab._gemini_model_combo.addItem(
                     self.model_combo.itemText(i),
                     self.model_combo.itemData(i)
                 )
             main_win.audio_tab._gemini_model_combo.setCurrentIndex(self.model_combo.currentIndex())

    def _on_fetch_error(self, msg: str):
        self.btn_fetch_models.setEnabled(True)
        self.btn_fetch_models.setText("🔄 Atualizar Lista")
        self.test_result_lbl.setText(f"❌ Erro ao buscar modelos: {msg}")

    def _on_test_done(self, success: bool, msg: str):


        self.btn_test.setEnabled(True)
        if not success:
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                msg = "❌ Quota Excedida (429): Você atingiu o limite de requisições do Gemini (Tier Gratuito). Aguarde 1 minuto ou aumente o 'Delay' e tente novamente."
            self.test_result_lbl.setStyleSheet("color:#e05555;")
        else:
            self.test_result_lbl.setStyleSheet("color:#6ddd6d;")
        
        self.test_result_lbl.setText(msg)


    def _refresh_cache_info(self):
        cache_dir = Path("gemini_cache")
        if cache_dir.exists():
            files = list(cache_dir.glob("*.json"))
            total = sum(f.stat().st_size for f in files)
            kb = total / 1024
            self.cache_size_lbl.setText(
                f"{len(files)} arquivo(s) de cache — {kb:.1f} KB"
            )
        else:
            self.cache_size_lbl.setText("Pasta gemini_cache não encontrada.")

    def _clear_cache(self):
        cache_dir = Path("gemini_cache")
        if not cache_dir.exists():
            QMessageBox.information(self, "Cache", "Nenhum cache encontrado.")
            return
        ans = QMessageBox.question(
            self, "Limpar Cache",
            "Tem certeza? Todo o progresso de revisão/tradução em cache será perdido.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if ans == QMessageBox.StandardButton.Yes:
            import shutil
            shutil.rmtree(str(cache_dir), ignore_errors=True)
            cache_dir.mkdir(exist_ok=True)
            self._refresh_cache_info()
            QMessageBox.information(self, "Cache Limpo", "Cache Gemini removido com sucesso.")

    def _on_theme_changed(self, name: str):
        # Sync with MainWindow if available
        mw = self.window()
        if hasattr(mw, "theme_combo"):
            mw.theme_combo.setCurrentText(name)

    def _browse_bg(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Selecionar imagem de fundo", "",
            "Imagens (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if path:
            self.bg_path_edit.setText(path)
            # Apply to MainWindow
            mw = self.window()
            if hasattr(mw, "_browse_background"):
                from PySide6.QtGui import QPixmap
                px = QPixmap(path)
                if not px.isNull():
                    mw._bg_pixmap = px
                    mw.btn_clear_bg.setVisible(True)
                    mw.overlay_lbl.setVisible(True)
                    mw.overlay_slider.setVisible(True)
                    mw._apply_header_style()
                    mw.update()

    def _clear_bg(self):
        self.bg_path_edit.clear()
        mw = self.window()
        if hasattr(mw, "_clear_background"):
            mw._clear_background()

    def _sync_bg_to_main(self):
        mw = self.window()
        if hasattr(mw, "overlay_slider"):
            mw.overlay_slider.setValue(self.overlay_slider.value())
            mw._bg_overlay_alpha = self.overlay_slider.value()
            mw.update()

    def _browse_dir(self, edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "Selecionar pasta", edit.text())
        if path:
            edit.setText(path)

    def _export_session(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar Configurações", "manhwa_config.json", "JSON (*.json)"
        )
        if path:
            mw = self.window()
            data = mw.get_session() if hasattr(mw, "get_session") else self.get_session()
            Path(path).write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            QMessageBox.information(self, "Exportado", f"Configurações salvas em:\n{path}")

    def _import_session(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Importar Configurações", "", "JSON (*.json)"
        )
        if path:
            try:
                data = json.loads(Path(path).read_text(encoding="utf-8"))
                mw = self.window()
                if hasattr(mw, "_restore_session"):
                    mw._session = data
                    mw._restore_session()
                else:
                    self.load_session(data.get("settings_tab", data))
                QMessageBox.information(self, "Importado", "Configurações carregadas com sucesso.")
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Falha ao importar: {e}")


# ---------------------------------------------------------------------------
# Janela Principal
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Utilitários de Teste Localizado
# ---------------------------------------------------------------------------
SAMPLE_TEXTS = {
    "pt": "Olá! Esta é uma amostra da minha voz em português. Espero que a qualidade esteja excelente para o seu vídeo.",
    "en": "Hello there! This is a sample of my voice in English. I hope the quality is perfect for your project.",
    "es": "Hola, esta es una muestra de mi voz para su proyecto.",
    "fr": "Bonjour, ceci est un échantillon de ma voix pour votre projet.",
    "de": "Hallo, dies ist eine Probe meiner Stimme für Ihr Projekt.",
    "it": "Ciao! Questo è un campione della mia voce in italiano. Spero que la qualità sia ottima per o teu vídeo.",
    "ru": "Привет, это образец моего голоса для вашего проекта.",
    "ja": "こんにちは、これはあなたのプロジェクトのための私の声のサンプルです。",
    "ko": "안녕하세요, 이것은 당신의 프로젝트를 위한 제 목소리 샘플입니다.",
    "zh": "你好，这是我为你的项目提供的声音样本。"
}

class LanguageConfigCard(QFrame):
    """
    Card dinâmico para configuração de voz por idioma no Modo Expresso.
    """
    def __init__(self, lang_code, lang_name, parent=None):
        super().__init__(parent)
        self.lang_code = lang_code
        self.lang_name = lang_name
        self._setup_ui()

    def _setup_ui(self):
        self.setObjectName("configCard")
        self.setStyleSheet("""
            QFrame#configCard {
                background: #1e1e2e;
                border: 1px solid #3b82f6;
                border-radius: 10px;
                padding: 2px;
            }
        """)
        
        main_v = QVBoxLayout(self)
        main_v.setContentsMargins(10, 8, 10, 8)
        main_v.setSpacing(6)

        # Top Row: Info + Engine + Voice + Controls
        top_row = QHBoxLayout()
        top_row.setSpacing(10)

        lbl = QLabel(f"<b>{self.lang_name}</b>")
        lbl.setFixedWidth(90)
        lbl.setStyleSheet("font-size: 13px; color: #60a5fa;")
        top_row.addWidget(lbl)
        
        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["chatterbox", "kokoro", "qwen", "indextts"])
        self.engine_combo.setFixedWidth(100)
        self.engine_combo.currentTextChanged.connect(self._update_voice_combo)
        self.engine_combo.currentTextChanged.connect(self._on_engine_changed)
        top_row.addWidget(self.engine_combo)
        
        # Submodelo (Turbo vs Multilingual vs Original)
        self.submodel_combo = QComboBox()
        self.submodel_combo.addItems(["turbo", "multilingual", "original"])
        self.submodel_combo.setFixedWidth(100)
        top_row.addWidget(self.submodel_combo)
        
        self.voice_combo = QComboBox()
        self.voice_combo.setMinimumWidth(180)
        top_row.addWidget(self.voice_combo)
        
        # Clearer buttons with text
        self.voice_browse = QPushButton("📁 Voz")
        self.voice_browse.setFixedWidth(75)
        self.voice_browse.setToolTip("Carregar arquivo de voz customizado")
        self.voice_browse.clicked.connect(self._browse_voice)
        top_row.addWidget(self.voice_browse)

        self.btn_adv = QPushButton("⚙️ Config")
        self.btn_adv.setCheckable(True)
        self.btn_adv.setFixedWidth(85)
        self.btn_adv.setToolTip("Parâmetros Avançados (Speed, Temp, etc)")
        top_row.addWidget(self.btn_adv)
        
        self.btn_test = QPushButton("📢 Testar")
        self.btn_test.setFixedWidth(85)
        self.btn_test.setObjectName("primary")
        self.btn_test.clicked.connect(self._test_voice)
        top_row.addWidget(self.btn_test)
        
        top_row.addStretch()
        main_v.addLayout(top_row)

        # Advanced Panel
        self.adv_frame = QFrame()
        self.adv_frame.setVisible(False)
        self.btn_adv.toggled.connect(self.adv_frame.setVisible)
        self.adv_frame.setStyleSheet("background: #161625; border-radius: 6px; border: 1px solid #333344;")
        
        avg = QGridLayout(self.adv_frame)
        avg.setContentsMargins(10, 8, 10, 8)
        avg.setSpacing(8)

        self.sp_speed = QDoubleSpinBox(); self.sp_speed.setRange(0.5, 2.5); self.sp_speed.setValue(1.0); self.sp_speed.setSingleStep(0.1); self.sp_speed.setFixedWidth(60)
        self.sp_temp = QDoubleSpinBox(); self.sp_temp.setRange(0.1, 1.0); self.sp_temp.setValue(0.65); self.sp_temp.setSingleStep(0.05); self.sp_temp.setFixedWidth(60)
        self.sp_exag = QDoubleSpinBox(); self.sp_exag.setRange(0.0, 1.0); self.sp_exag.setValue(0.0); self.sp_exag.setSingleStep(0.1); self.sp_exag.setFixedWidth(60)
        self.sp_cfg = QDoubleSpinBox(); self.sp_cfg.setRange(0.0, 1.0); self.sp_cfg.setValue(0.5); self.sp_cfg.setSingleStep(0.1); self.sp_cfg.setFixedWidth(60)
        self.sp_seed = QSpinBox(); self.sp_seed.setRange(-1, 999999); self.sp_seed.setValue(-1); self.sp_seed.setFixedWidth(80)

        def add_param(txt, spin, r, c):
            lbl = QLabel(txt)
            lbl.setStyleSheet("font-size: 10px; font-weight: bold; color: #94a3b8;")
            avg.addWidget(lbl, r, c)
            avg.addWidget(spin, r, c+1)

        add_param("Veloc:", self.sp_speed, 0, 0)
        add_param("Temp:", self.sp_temp, 0, 2)
        add_param("Exag:", self.sp_exag, 0, 4)
        add_param("CFG:", self.sp_cfg, 1, 0)
        add_param("Seed:", self.sp_seed, 1, 2)

        main_v.addWidget(self.adv_frame)
        self._on_engine_changed(self.engine_combo.currentText())
        self._update_voice_combo()

    def _on_engine_changed(self, eng):
        self.submodel_combo.setVisible(eng == "chatterbox")

    def _update_voice_combo(self):
        eng = self.engine_combo.currentText()
        self.voice_combo.clear()
        
        if eng == "chatterbox":
            self.voice_combo.addItem("Sem clonagem (Voz do Modelo)", "")
            vdir = Path("voices")
            if vdir.exists():
                for p in vdir.glob("*.wav"): self.voice_combo.addItem(p.stem, str(p))
            # Tambem checar em manhwa_app/voices se existir
            vdir2 = Path(__file__).parent / "voices"
            if vdir2.exists():
                for p in vdir2.glob("*.wav"): self.voice_combo.addItem(p.stem, str(p))
        elif eng == "kokoro":
            v_list = ["af_heart", "af_bella", "af_nicole", "af_aoi", "af_sarah", "am_adam", "am_michael", "bf_isabella", "bf_emma", "bm_george", "bm_lewis"]
            for v in v_list: self.voice_combo.addItem(v, v)
        elif eng == "qwen":
            self.voice_combo.addItems(["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"])
            qdir = Path(__file__).parent / "qwen_voices"
            if qdir.exists():
                for p in qdir.glob("*.pt"): self.voice_combo.addItem(p.stem, str(p))
        elif eng == "indextts":
            pdir = Path("presets")
            if pdir.exists():
                for p in pdir.glob("*.wav"): self.voice_combo.addItem(p.stem, str(p))

    def _browse_voice(self):
        path, _ = QFileDialog.getOpenFileName(self, f"Voz para {self.lang_name}", "", "Áudio (*.wav *.mp3)")
        if path:
            self.voice_combo.addItem(f"[Custom] {Path(path).stem}", path)
            self.voice_combo.setCurrentIndex(self.voice_combo.count()-1)

    def _test_voice(self):
        text = SAMPLE_TEXTS.get(self.lang_code, SAMPLE_TEXTS["en"])
        engine_str = self.engine_combo.currentText()
        voice_val = resolve_voice_path(self.voice_combo.currentData() or self.voice_combo.currentText())
        
        main_win = self.window()
        if hasattr(main_win, "language_config_tab"):
            main_win.language_config_tab._test_voice_custom(
                text=text, 
                engine=engine_str, 
                voice_path=voice_val,
                lang_code=self.lang_code,
                btn_widget=self.btn_test,
                speed=self.sp_speed.value(),
                temperature=self.sp_temp.value()
            )

    def get_config(self):
        return {
            "voice": resolve_voice_path(self.voice_combo.currentData() or self.voice_combo.currentText()),
            "engine": self.engine_combo.currentText(),
            "model_type": self.submodel_combo.currentText() if self.engine_combo.currentText() == "chatterbox" else "default",
            "speed": self.sp_speed.value(),
            "temperature": self.sp_temp.value(),
            "exaggeration": self.sp_exag.value(),
            "cfg_weight": self.sp_cfg.value(),
            "seed": self.sp_seed.value()
        }

    def load_config(self, cfg):
        if not cfg: return
        if "engine" in cfg:
            self.engine_combo.setCurrentText(cfg["engine"])
        if "model_type" in cfg and cfg.get("engine") == "chatterbox":
            self.submodel_combo.setCurrentText(cfg["model_type"])
        
        # Need to ensure voice combo is updated before setting voice
        self._update_voice_combo()
        
        if "voice" in cfg:
            idx = self.voice_combo.findData(cfg["voice"])
            if idx >= 0:
                self.voice_combo.setCurrentIndex(idx)
            else:
                self.voice_combo.setCurrentText(cfg["voice"])
        
        if "speed" in cfg: self.sp_speed.setValue(cfg["speed"])
        if "temperature" in cfg: self.sp_temp.setValue(cfg["temperature"])
        if "exaggeration" in cfg: self.sp_exag.setValue(cfg["exaggeration"])
        if "cfg_weight" in cfg: self.sp_cfg.setValue(cfg["cfg_weight"])
        if "seed" in cfg: self.sp_seed.setValue(cfg["seed"])

# ---------------------------------------------------------------------------
# Aba Express - One Click Pipeline
# ---------------------------------------------------------------------------

class ExpressTab(QWidget):
    """
    Modo Expresso: Um único clique para Processamento Gemini -> TTS -> Vídeo.
    """
    log_message = Signal(str)
    pipeline_finished = Signal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._files = [] # list of {"path": str}
        self._images = []
        self._worker = None
        self._setup_ui()

    def _setup_ui(self):
        # Base Layout (Scroll Area)
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setObjectName("expressScroll")
        outer_layout.addWidget(self.scroll)

        self.content = QWidget()
        self.content.setObjectName("expressContent")
        self.scroll.setWidget(self.content)
        
        layout = QVBoxLayout(self.content)
        layout.setContentsMargins(30, 25, 30, 25)
        layout.setSpacing(20)

        # Header Title
        title_box = QHBoxLayout()
        title = QLabel("🚀 Modo Expresso")
        title.setStyleSheet("font-size: 24px; font-weight: 800; color: #60a5fa; letter-spacing: 1px;")
        title_box.addWidget(title)
        title_box.addStretch()
        layout.addLayout(title_box)

        desc = QLabel("Processamento inteligente: Revisão Gemini → TTS Multilíngue → Montagem de Vídeo.")
        desc.setStyleSheet("color: #94a3b8; font-size: 13px; margin-bottom: 5px;")
        layout.addWidget(desc)

        # Centralized Config Area
        cfg_row = QHBoxLayout()
        cfg_row.setSpacing(15)

        v1 = QVBoxLayout(); v1.addWidget(QLabel("NOME DO PROJETO")); self.proj_edit = QLineEdit("projeto_expresso_1"); self.proj_edit.setMinimumHeight(35); v1.addWidget(self.proj_edit)
        cfg_row.addLayout(v1, 2)

        v2 = QVBoxLayout(); v2.addWidget(QLabel("IDIOMA ORIGINAL (TXT)")); self.source_lang_combo = QComboBox(); self.source_lang_combo.setMinimumHeight(35)
        for code, name in [("pt", "Português"), ("en", "Inglês"), ("es", "Espanhol"), ("ja", "Japonês"), ("ko", "Coreano"), ("zh", "Chinês")]:
            self.source_lang_combo.addItem(f"{name} ({code})", code)
        v2.addWidget(self.source_lang_combo)
        cfg_row.addLayout(v2, 1)

        layout.addLayout(cfg_row)

        # File Drop Areas
        file_row = QHBoxLayout()
        file_row.setSpacing(20)
        
        # Style for drop zone
        drop_style = """
            QFrame { 
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #1e1e2e, stop:1 #13131e); 
                border: 2px solid #334155; 
                border-radius: 12px; 
            }
            QFrame:hover { border-color: #3b82f6; background: #1a1a2a; }
        """
        
        self.drop_txt = QFrame()
        self.drop_txt.setAcceptDrops(True)
        self.drop_txt.setMinimumHeight(120)
        self.drop_txt.setStyleSheet(drop_style)
        dt_v = QVBoxLayout(self.drop_txt)
        dt_v.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.txt_lbl = QLabel("📄\n<b>CLIQUE OU ARRASTE</b>\nRoteiro (.txt)")
        self.txt_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.txt_lbl.setStyleSheet("color: #cbd5e1; font-size: 13px; border: none; background: transparent;")
        dt_v.addWidget(self.txt_lbl)
        file_row.addWidget(self.drop_txt)
        
        self.drop_img = QFrame()
        self.drop_img.setAcceptDrops(True)
        self.drop_img.setMinimumHeight(120)
        self.drop_img.setStyleSheet(drop_style.replace("#3b82f6", "#e8789a"))
        di_v = QVBoxLayout(self.drop_img)
        di_v.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_lbl = QLabel("🖼️\n<b>CLIQUE OU ARRASTE</b>\nPasta de Imagens")
        self.img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_lbl.setStyleSheet("color: #cbd5e1; font-size: 13px; border: none; background: transparent;")
        di_v.addWidget(self.img_lbl)
        file_row.addWidget(self.drop_img)
        
        layout.addLayout(file_row)

        # Mouse clicks
        self.drop_txt.mousePressEvent = lambda e: self._pick_txt()
        self.drop_img.mousePressEvent = lambda e: self._pick_images()

        # Target Languages Selection (Grid)
        langs_group = QGroupBox("🌍 IDIOMAS DE DESTINO (TRADUÇÃO GEMINI)")
        self.lang_grid = QGridLayout(langs_group)
        self.lang_grid.setContentsMargins(15, 15, 15, 15)
        self.lang_grid.setSpacing(10)
        
        self.lang_checkboxes = {}
        self.target_langs = [
            ("pt", "Português"), ("en", "Inglês"), ("es", "Espanhol"), 
            ("fr", "Francês"), ("de", "Alemão"), ("it", "Italiano"),
            ("ru", "Russo"), ("ja", "Japonês"), ("ko", "Coreano"), ("zh", "Chinês")
        ]
        
        cols = 5
        for i, (code, name) in enumerate(self.target_langs):
            chk = QCheckBox(name)
            chk.setProperty("lang_code", code)
            chk.setProperty("lang_name", name)
            if code == "pt":
                chk.setChecked(True)
            chk.toggled.connect(self._on_lang_toggled)
            self.lang_grid.addWidget(chk, i // cols, i % cols)
            self.lang_checkboxes[code] = chk
            
        layout.addWidget(langs_group)

        # Language Config Cards
        voices_header = QHBoxLayout()
        v_title = QLabel("🎙️ CONFIGURAÇÕES DE VOZ POR IDIOMA")
        v_title.setStyleSheet("font-weight: bold; color: #94a3b8; font-size: 11px;")
        voices_header.addWidget(v_title)
        voices_header.addStretch()
        
        self.chk_use_saved = QCheckBox("Sincronizar Global")
        self.chk_use_saved.setChecked(True)
        self.chk_use_saved.setStyleSheet("color: #3b82f6; font-size: 10px; font-weight: bold;")
        voices_header.addWidget(self.chk_use_saved)
        layout.addLayout(voices_header)

        self.cards_container = QWidget()
        self.cards_v = QVBoxLayout(self.cards_container)
        self.cards_v.setContentsMargins(0, 5, 0, 5)
        self.cards_v.setSpacing(8)
        layout.addWidget(self.cards_container)

        self.language_cards = {}
        self._add_lang_card("pt", "Português")


        # Control Buttons
        self.btn_run = QPushButton("⚡ INICIAR PIPELINE COMPLETO")
        self.btn_run.setObjectName("primary")
        self.btn_run.setFixedHeight(55)
        self.btn_run.setStyleSheet("font-size: 16px; font-weight: 800; border-radius: 12px;")
        self.btn_run.clicked.connect(self._run_all)
        layout.addWidget(self.btn_run)

        # Logs & Progress
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(120)
        self.log_text.setStyleSheet("background:#0a0a0f; border: 1px solid #1e293b; border-radius: 8px; color: #94a3b8; font-family: 'Consolas', monospace;")
        layout.addWidget(self.log_text)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(6)
        layout.addWidget(self.progress_bar)

        # Apply specific stylesheet for scroll area
        self.setStyleSheet("""
            QScrollArea#expressScroll { background: transparent; }
            QWidget#expressContent { background: transparent; }
            QGroupBox { font-size: 10px; color: #64748b; margin-top: 15px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
        """)

    def _pick_txt(self):
        path, _ = QFileDialog.getOpenFileName(self, "Selecionar Roteiro", "", "Texto (*.txt)")
        if path: self._set_txt(path)

    def _pick_images(self):
        folder = QFileDialog.getExistingDirectory(self, "Selecionar Pasta de Imagens")
        if folder: self._set_images(folder)

    def _on_lang_toggled(self, checked):
        chk = self.sender()
        code = chk.property("lang_code")
        name = chk.property("lang_name")
        if checked:
            self._add_lang_card(code, name)
        else:
            self._remove_lang_card(code)

    def _add_lang_card(self, code, name):
        if code in self.language_cards: return
        card = LanguageConfigCard(code, name, self)
        self.cards_v.addWidget(card)
        self.language_cards[code] = card
        
        # Apply pending overrides from session load
        if hasattr(self, "_pending_overrides") and code in self._pending_overrides:
            card.load_config(self._pending_overrides[code])

    def _remove_lang_card(self, code):
        if code not in self.language_cards: return
        card = self.language_cards.pop(code)
        card.setParent(None)
        card.deleteLater()

    def _get_overrides(self):
        ov = {}
        for code, card in self.language_cards.items():
            cfg = card.get_config()
            if cfg["voice"] or cfg["engine"] != "chatterbox":
                ov[code] = cfg
        return ov

    def _set_txt(self, path):
        self._files = [{"path": path}]
        self.txt_lbl.setText(f"✅ Roteiro: {Path(path).name}")

    def _set_images(self, folder):
        from pathlib import Path
        imgs = sorted([str(p) for p in Path(folder).glob("*") if p.suffix.lower() in [".jpg", ".png", ".webp", ".jpeg"]], key=natural_sort_key)
        self._images = imgs
        self.img_lbl.setText(f"✅ {len(imgs)} imagens carregadas")

    def get_session(self):
        return {
            "project_name": self.proj_edit.text(),
            "source_lang": self.source_lang_combo.currentData(),
            "selected_langs": [code for code, chk in self.lang_checkboxes.items() if chk.isChecked()],
            "overrides": self._get_overrides(),
            "use_saved_tts": self.chk_use_saved.isChecked()
        }

    def load_session(self, data):
        if not data: return
        if "project_name" in data: 
            self.proj_edit.setText(data["project_name"])
        if "source_lang" in data:
            idx = self.source_lang_combo.findData(data["source_lang"])
            if idx >= 0: self.source_lang_combo.setCurrentIndex(idx)
        
        if "selected_langs" in data:
            for code, chk in self.lang_checkboxes.items():
                if code == "pt": continue 
                chk.setChecked(code in data["selected_langs"])
        
        if "use_saved_tts" in data:
            self.chk_use_saved.setChecked(data["use_saved_tts"])
            
        overrides = data.get("overrides", {})
        for code, card in self.language_cards.items():
            if code in overrides:
                card.load_config(overrides[code])
        self._pending_overrides = overrides

    def _run_all(self):
        if not self._files:
            QMessageBox.warning(self, "Erro", "Selecione um arquivo .txt primeiro.")
            return
        if not self._images:
            QMessageBox.warning(self, "Erro", "Selecione as imagens primeiro.")
            return

        selected_langs = [code for code, chk in self.lang_checkboxes.items() if chk.isChecked()]
        if not selected_langs:
            QMessageBox.warning(self, "Erro", "Selecione ao menos um idioma.")
            return

        self.btn_run.setEnabled(False)
        self.log_text.clear()
        self.progress_bar.setValue(0)
        
        main_win = self.window()
        cfg = main_win.settings_tab.get_session()
        api_key = main_win.settings_tab.get_api_key()
        
        if not api_key:
            QMessageBox.warning(self, "Erro", "API Key do Gemini não configurada na aba Configurações.")
            self.btn_run.setEnabled(True)
            return

        overrides = self._get_overrides()

        self._orchestrator = ExpressOrchestrator(
            parent=self,
            txt_path=self._files[0]["path"],
            image_paths=self._images,
            project_name=self.proj_edit.text().strip(),
            api_key=api_key,
            source_lang=self.source_lang_combo.currentData(),
            languages=selected_langs,
            use_saved_tts=self.chk_use_saved.isChecked(),
            main_cfg=cfg,
            overrides=overrides
        )
        self._orchestrator.log_message.connect(lambda m: _append_log(self.log_text, m))
        self._orchestrator.progress.connect(self.progress_bar.setValue)
        self._orchestrator.finished.connect(self._on_done)
        self._orchestrator.start()

    def _on_done(self, success, msg):
        self.btn_run.setEnabled(True)
        if success:
            QMessageBox.information(self, "Concluído", f"Pipeline concluído com sucesso!\n{msg}")
        else:
            QMessageBox.critical(self, "Erro", f"Pipeline falhou:\n{msg}")

class ExpressOrchestrator(QThread):
    log_message = Signal(str)
    progress = Signal(int)
    finished = Signal(bool, str)

    def __init__(self, parent, txt_path, image_paths, project_name, api_key, source_lang, languages, use_saved_tts, main_cfg, overrides=None):
        super().__init__(parent)
        self.txt_path = txt_path
        self.image_paths = image_paths
        self.project_name = project_name
        self.api_key = api_key
        self.source_lang = source_lang
        self.languages = languages
        self.use_saved_tts = use_saved_tts
        self.main_cfg = main_cfg
        self.overrides = overrides or {}

    def run(self):
        try:
            # 1. Gemini Process
            self.log_message.emit("--- FASE 1: Revisão e Tradução Gemini ---")
            self.progress.emit(10)
            
            # Use original revised (pt) + requested langs
            target_langs = list(set(self.languages))
            
            from gemini_processor import GeminiProcessor
            proc = GeminiProcessor(model_name=self.main_cfg.get("gemini", {}).get("model_name", "gemini-1.5-flash"))
            
            # Map paths
            lang_paths = proc.process(
                txt_path=self.txt_path,
                api_key=self.api_key,
                languages=target_langs,
                source_lang=self.source_lang,
                delay_seconds=float(self.main_cfg.get("gemini", {}).get("delay", 4)),
                revision_prompt=self.main_cfg.get("gemini", {}).get("revision_prompt"),
                translation_prompt=self.main_cfg.get("gemini", {}).get("translation_prompt"),
                chunk_size=int(self.main_cfg.get("gemini", {}).get("chunk_size", 12)),
                overlap=int(self.main_cfg.get("gemini", {}).get("overlap", 2)),
                progress_callback=lambda c, t, m: self.log_message.emit(f"Gemini: {m}")
            )
            
            if not lang_paths:
                self.finished.emit(False, "Gemini não gerou arquivos.")
                return

            self.progress.emit(40)
            self.log_message.emit("--- FASE 2: Geração de Áudio e Vídeo por Idioma ---")

            main_win = self.parent().window()
            mapping = main_win.language_config_tab.mapping if self.use_saved_tts else {}
            tts_global = main_win.tts_tab.get_session()
            video_tab = main_win.video_tab
            
            from manhwa_app.audio_pipeline import AudioPipeline
            from manhwa_app.video_pipeline import VideoPipeline

            results_summary = []

            for i, (l_code, l_path) in enumerate(lang_paths.items()):
                # Filtrar para processar apenas o que o usuário escolheu
                if l_code not in self.languages:
                    continue
                    
                self.log_message.emit(f"\n[IDIOMA: {l_code.upper()}]")
                sub_project = f"{self.project_name}_{l_code}"
                
                # Config baseada no tts_global + mapeamento + overrides
                c_cfg = {
                    "path": l_path, 
                    "lang": l_code,
                    "engine": tts_global.get("tts_engine", "chatterbox"),
                    "voice": tts_global.get("voice", ""),
                    "speed": tts_global.get("speed", 1.0),
                    "temperature": tts_global.get("temperature", 0.65),
                    "exaggeration": tts_global.get("exaggeration", 0.0),
                    "cfg_weight": tts_global.get("cfg_weight", 0.5),
                    "seed": tts_global.get("seed", -1)
                }
                # Se usar o mapeamento salvo (Aba Idiomas)
                if self.use_saved_tts and l_code in mapping:
                    c_cfg.update(mapping[l_code])
                
                # Se tiver overrides específicos desta sessão (Express Tab)
                if l_code in self.overrides:
                    c_cfg.update(self.overrides[l_code])

                # 2. Audio Pipeline para este idioma
                audio_pipe = AudioPipeline(
                    file_configs=[c_cfg],
                    project_name=sub_project,
                    output_root="output",
                    **tts_global
                )
                audio_pipe.log_message.connect(self.log_message.emit)
                audio_pipe.run()

                # 3. Video Pipeline para este idioma
                audios_dir = Path(f"output/{sub_project}/audios")
                audio_files = sorted([str(p) for p in audios_dir.glob("*.wav")], key=natural_sort_key)
                
                if not audio_files:
                    self.log_message.emit(f"⚠ Falha ao gerar áudios para {l_code}. Pulando vídeo.")
                    continue

                output_mp4 = f"output/{sub_project}/{sub_project}_final.mp4"
                
                # Montar pares Áudio/Imagem
                n = min(len(audio_files), len(self.image_paths))
                pairs = [(audio_files[j], self.image_paths[j]) for j in range(n)]

                self.log_message.emit(f"🎥 Montando vídeo: {output_mp4}")
                video_pipe = VideoPipeline(
                    pairs=pairs,
                    output_path=output_mp4,
                    effect_mode=video_tab.effect_combo.currentData() or "auto",
                    transition_mode=video_tab.transition_combo.currentData() or "fade",
                    transition_time=video_tab.transition_time_spin.value(),
                    bg_music_path=video_tab.bgm_path_edit.text().strip() or None,
                    bg_music_volume=video_tab.bgm_vol_spin.value() / 100.0 if hasattr(video_tab, 'bgm_vol_spin') else 0.5
                )
                video_pipe.log_message.connect(self.log_message.emit)
                video_pipe.run()
                
                results_summary.append(f"{l_code.upper()}: {output_mp4}")
                self.progress.emit(40 + int((i + 1) / len(lang_paths) * 60))

            summary_str = "\n".join(results_summary)
            self.finished.emit(True, f"Processamento concluído para {len(results_summary)} idiomas!\n\nProjetos:\n{summary_str}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log_message.emit(f"❌ ERRO NO ORQUESTRADOR: {str(e)}")
            self.finished.emit(False, str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Manhwa Video Creator")
        self.setMinimumSize(850, 600)
        self.resize(1100, 760)
        self._session = _load_session()
        # Background image state
        self._bg_pixmap = None
        self._bg_overlay_alpha = 180  # 0-255 (dark themes ~180, light ~100)
        self._current_theme_key = "\U0001f311 Dark (Padr\u00e3o)"
        self._setup_ui()
        self._loader_thread = None
        self._check_model_in_background()
        self._restore_session()
        
        # Inicia primeiro preload (a checagem interna bloqueia duplicatas)
        self.trigger_model_preload()

    def trigger_model_preload(self):
        """Dispara o carregamento do modelo em background."""
        if not hasattr(self, "tts_tab"): return
        
        cfg = self.tts_tab.get_session()
        engine_name = cfg.get("tts_engine", "chatterbox")
        model_type = cfg.get("model_type", "turbo")

        if hasattr(self, "_loader_thread") and self._loader_thread and self._loader_thread.isRunning():
            return # Já carregando

        self.model_status_label.setText("TTS: carregando…")
        self.model_status_label.setStyleSheet("color:#f0b040;font-size:11px;")
        
        if hasattr(self, "audio_tab") and hasattr(self.audio_tab, "btn_generate"):
            self.audio_tab.btn_generate.setEnabled(False)
            self.audio_tab.btn_generate.setText("Carregando modelo...")
        
        self._loader_thread = ModelLoaderThread(engine_name, model_type)
        self._loader_thread.setStackSize(16 * 1024 * 1024)
        self._loader_thread.finished_loading.connect(self._on_model_preloaded)
        self._loader_thread.start()

    def _on_model_preloaded(self, success, info):
        if hasattr(self, "audio_tab") and hasattr(self.audio_tab, "btn_generate"):
            self.audio_tab.btn_generate.setEnabled(True)
            self.audio_tab.btn_generate.setText("🎙️ Gerar Áudio")
            
        if success:
            self.model_status_label.setText(f"TTS: {info} Pronto ✅")
            self.model_status_label.setStyleSheet("color:#6ddd6d;font-size:11px;font-weight:bold;")
        else:
            self.model_status_label.setText(f"TTS: Erro ❌ ({info[:15]}...)")
            self.model_status_label.setStyleSheet("color:#e05555;font-size:11px;")

    # ------------------------------------------------------------------
    # Background image rendering
    # ------------------------------------------------------------------
    def paintEvent(self, event):
        if self._bg_pixmap and not self._bg_pixmap.isNull():
            from PySide6.QtGui import QPainter, QColor
            painter = QPainter(self)
            scaled = self._bg_pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation
            )
            x = (scaled.width() - self.width()) // 2
            y = (scaled.height() - self.height()) // 2
            painter.drawPixmap(-x, -y, scaled)
            painter.fillRect(self.rect(), QColor(0, 0, 0, self._bg_overlay_alpha))
        else:
            super().paintEvent(event)

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------
    def _setup_ui(self):
        # Create outer scroll area for responsiveness
        self.main_scroll = QScrollArea()
        self.main_scroll.setWidgetResizable(True)
        self.main_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; } QScrollArea > QWidget > QWidget { background: transparent; }")
        
        central = QWidget()
        central.setObjectName("mainCentralWidget")
        # Ensure the central widget is transparent so the drawn paintEvent background is visible
        central.setStyleSheet("QWidget#mainCentralWidget { background: transparent; }")
        
        self.main_scroll.setWidget(central)
        self.setCentralWidget(self.main_scroll)
        
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Header ──────────────────────────────────────────────────
        header = QWidget()
        header.setObjectName("appHeader")
        header.setFixedHeight(56)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(20, 0, 16, 0)
        hl.setSpacing(10)

        title = QLabel("🎬  Manhwa Video Creator")
        title.setObjectName("heading")
        title.setStyleSheet("font-size:17px;font-weight:800;letter-spacing:0.5px;")
        hl.addWidget(title)
        hl.addStretch()

        # Theme selector
        theme_lbl = QLabel("🎨")
        theme_lbl.setStyleSheet("font-size:14px;")
        theme_lbl.setToolTip("Tema")
        hl.addWidget(theme_lbl)
        self.theme_combo = QComboBox()
        self.theme_combo.setFixedWidth(180)
        self.theme_combo.setToolTip("Selecionar tema de cores")
        for name in THEMES:
            self.theme_combo.addItem(name)
        self.theme_combo.setCurrentText(self._current_theme_key)
        self.theme_combo.currentTextChanged.connect(self._on_theme_changed)
        hl.addWidget(self.theme_combo)

        # Botão Reset
        self.btn_reset = QPushButton("↺ Reset")
        self.btn_reset.setToolTip("Restaurar para as configurações padrão")
        self.btn_reset.clicked.connect(self._reset_to_defaults)
        hl.addWidget(self.btn_reset)

        # Background image button
        self.btn_bg = QPushButton("🖼")
        self.btn_bg.setFixedSize(32, 32)
        self.btn_bg.setToolTip("Selecionar imagem de fundo")
        self.btn_bg.clicked.connect(self._browse_background)
        hl.addWidget(self.btn_bg)

        # Clear background button
        self.btn_clear_bg = QPushButton("✖")
        self.btn_clear_bg.setFixedSize(32, 32)
        self.btn_clear_bg.setObjectName("danger")
        self.btn_clear_bg.setToolTip("Remover imagem de fundo")
        self.btn_clear_bg.clicked.connect(self._clear_background)
        self.btn_clear_bg.setVisible(False)
        hl.addWidget(self.btn_clear_bg)

        # Overlay opacity slider (only visible when bg image is set)
        self.overlay_lbl = QLabel("Opacidade:")
        self.overlay_lbl.setStyleSheet("font-size:11px;")
        self.overlay_lbl.setVisible(False)
        hl.addWidget(self.overlay_lbl)
        self.overlay_slider = QSlider(Qt.Horizontal)
        self.overlay_slider.setRange(0, 240)
        self.overlay_slider.setValue(self._bg_overlay_alpha)
        self.overlay_slider.setFixedWidth(90)
        self.overlay_slider.setToolTip("Escurecimento do fundo (0 = transparente, 240 = escuro)")
        self.overlay_slider.valueChanged.connect(self._on_overlay_changed)
        self.overlay_slider.setVisible(False)
        hl.addWidget(self.overlay_slider)

        sep = QLabel(" │ ")
        sep.setStyleSheet("color:rgba(255,255,255,0.15);")
        hl.addWidget(sep)

        self.device_label = QLabel("GPU: detectando…")
        self.device_label.setStyleSheet("font-size:11px;opacity:0.6;")
        hl.addWidget(self.device_label)

        sep2 = QLabel(" │ ")
        sep2.setStyleSheet("color:rgba(255,255,255,0.15);")
        hl.addWidget(sep2)

        self.model_status_label = QLabel("TTS: carregando…")
        self.model_status_label.setStyleSheet("color:#f0b040;font-size:11px;")
        hl.addWidget(self.model_status_label)

        layout.addWidget(header)

        # ── Tabs ─────────────────────────────────────────────────────
        self.tabs = QTabWidget()
        self.express_tab   = ExpressTab()
        self.audio_tab     = AudioTab()

        self.language_config_tab = LanguageConfigTab(self)
        self.tts_tab       = TtsConfigTab()
        self.images_tab    = ImagesTab()
        self.video_tab     = VideoTab()
        self.settings_tab  = SettingsTab()
        self.tabs.addTab(self.express_tab,  "🚀 Express")
        self.tabs.addTab(self.audio_tab,    "📝 Áudio")

        self.tabs.addTab(self.language_config_tab, "🗣️ Traduções Multilíngue")
        self.tabs.addTab(self.tts_tab,      "⚙️ TTS")
        self.tabs.addTab(self.images_tab,   "🖼 Imagens")
        self.tabs.addTab(self.video_tab,    "🎬 Vídeo")
        self.tabs.addTab(self.settings_tab, "🔧 Configurações")
        # Ensure QTabWidget is transparent at its base, so bg image falls through
        self.tabs.setStyleSheet("QTabWidget { background: transparent; }")
        layout.addWidget(self.tabs)

        # ── Sincronização Gemini ─────────────────────────────────────
        # Sincronizar Combos do Gemini entre Áudio e Configurações
        if hasattr(self.audio_tab, "_gemini_model_combo"):
            for i in range(self.settings_tab.model_combo.count()):
                self.audio_tab._gemini_model_combo.addItem(
                    self.settings_tab.model_combo.itemText(i),
                    self.settings_tab.model_combo.itemData(i)
                )
            self.audio_tab._gemini_model_combo.setCurrentIndex(self.settings_tab.model_combo.currentIndex())

            def sync_to_audio(idx):
                self.audio_tab._gemini_model_combo.blockSignals(True)
                self.audio_tab._gemini_model_combo.setCurrentIndex(idx)
                self.audio_tab._gemini_model_combo.blockSignals(False)

            def sync_to_settings(idx):
                self.settings_tab.model_combo.blockSignals(True)
                self.settings_tab.model_combo.setCurrentIndex(idx)
                self.settings_tab.model_combo.blockSignals(False)

            self.settings_tab.model_combo.currentIndexChanged.connect(sync_to_audio)
            self.audio_tab._gemini_model_combo.currentIndexChanged.connect(sync_to_settings)


        # ── Status bar ───────────────────────────────────────────────
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(
            "Pronto — carregue um .txt na aba 📝 Áudio para começar."
        )

        # ── Cross-tab connections ────────────────────────────────────
        self.audio_tab.audio_generated.connect(self._on_audio_generated)
        self.audio_tab.run_all_audio_done.connect(self._on_run_all_audio_done)
        self.images_tab.images_changed.connect(self.video_tab.update_image_paths)
        self.images_tab.images_changed.connect(self._on_images_changed)
        self.audio_tab.project_edit.textChanged.connect(self._on_project_name_changed)

        # Apply header style for current theme
        self._apply_header_style()

    def _apply_header_style(self):
        t = THEMES.get(self._current_theme_key, THEMES["\U0001f311 Dark (Padr\u00e3o)"])
        is_light = t.get("type") == "light"
        if self._bg_pixmap:
            # Transparent header when bg image is active
            header_style = (
                "QWidget#appHeader{background:rgba(0,0,0,0.42);"
                "border-bottom:1px solid rgba(255,255,255,0.08);}"
            )
            self.tabs.setStyleSheet(
                "QTabWidget::pane { background: transparent; border: none; } "
                "QTabWidget { background: transparent; } "
                "QTabWidget > QWidget { background: transparent; } "
                "QGroupBox { background: rgba(0, 0, 0, 0.4); border: 1px solid rgba(255, 255, 255, 0.15); }"
            )
        elif is_light:
            header_style = (
                f"QWidget#appHeader{{background:{t['header_bg']};"
                f"border-bottom:1px solid {t['border']};}}"
            )
            self.tabs.setStyleSheet("")
        else:
            header_style = (
                f"QWidget#appHeader{{background:{t['header_bg']};"
                f"border-bottom:1px solid {t['border']};}}"
            )
            self.tabs.setStyleSheet("")
        # Apply only to the header widget to avoid overriding global sheet
        header = self.main_scroll.widget().layout().itemAt(0).widget()
        header.setStyleSheet(header_style)

    # ------------------------------------------------------------------
    # Theme & background slots
    # ------------------------------------------------------------------
    @Slot(str)
    def _on_theme_changed(self, name: str):
        self._current_theme_key = name
        t = THEMES.get(name, THEMES["\U0001f311 Dark (Padr\u00e3o)"])
        QApplication.instance().setStyleSheet(_build_stylesheet(t))
        # Adjust default overlay alpha based on theme type
        if not self.overlay_slider.isVisible():
            is_light = t.get("type") == "light"
            self._bg_overlay_alpha = 100 if is_light else 180
            self.overlay_slider.setValue(self._bg_overlay_alpha)
        self._apply_header_style()
        self.update()

    def _browse_background(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Selecionar imagem de fundo", "",
            "Imagens (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if path:
            from PySide6.QtGui import QPixmap
            px = QPixmap(path)
            if not px.isNull():
                self._bg_pixmap = px
                self.btn_clear_bg.setVisible(True)
                self.overlay_lbl.setVisible(True)
                self.overlay_slider.setVisible(True)
                self._apply_header_style()
                self.update()

    def _clear_background(self):
        self._bg_pixmap = None
        self.btn_clear_bg.setVisible(False)
        self.overlay_lbl.setVisible(False)
        self.overlay_slider.setVisible(False)
        self._apply_header_style()
        self.update()

    @Slot(int)
    def _on_overlay_changed(self, val: int):
        self._bg_overlay_alpha = val
        self.update()

    # ------------------------------------------------------------------
    # Cross-tab callbacks
    # ------------------------------------------------------------------
    def _on_run_all_audio_done(self, txt_index: int, proj_name: str, audio_paths: List[str]):
        self.tabs.setCurrentWidget(self.video_tab)
        self.video_tab.start_run_all_video(proj_name, audio_paths)

    def _check_model_in_background(self):
        class _Loader(QThread):
            result = Signal(bool, str, str)
            def run(self_t):
                try:
                    from manhwa_app.audio_pipeline import _engine, _ENGINE_AVAILABLE
                    engine = _engine
                    if engine is None or not _ENGINE_AVAILABLE:
                        self_t.result.emit(False, "cpu", "engine não disponível")
                        return
                    
                    ok = True
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    vram_info = ""
                    if torch.cuda.is_available():
                        props = torch.cuda.get_device_properties(0)
                        total_gb = props.total_memory / (1024**3)
                        free_b, _ = torch.cuda.mem_get_info()
                        free_gb = free_b / (1024**3)
                        vram_info = f"{props.name}  {free_gb:.1f}/{total_gb:.1f}GB"
                    else:
                        vram_info = "CPU (sem CUDA)"
                    self_t.result.emit(ok, device, vram_info)
                except Exception as e:
                    self_t.result.emit(False, "cpu", str(e))

        self._loader = _Loader(self)
        self._loader.result.connect(self._on_model_loaded)
        self._loader.start()

    @Slot(bool, str, str)
    def _on_model_loaded(self, success: bool, device: str, vram_info: str):
        self.device_label.setText(vram_info)
        if success:
            self.model_status_label.setText(f"TTS: ✓ {device.upper()}")
            self.model_status_label.setStyleSheet("color:#60c060;font-size:11px;")
            self.status_bar.showMessage(f"Modelo TTS carregado em {device.upper()}. Pronto.")
        else:
            self.model_status_label.setText("TTS: ✗ Falha")
            self.model_status_label.setStyleSheet("color:#e05050;font-size:11px;")
            self.status_bar.showMessage("Falha ao carregar modelo TTS. Verifique o console.")

    @Slot(list)
    def _on_audio_generated(self, paths: List[str]):
        self.video_tab.update_audio_paths(paths)
        self.images_tab.update_parity(len(paths), self.video_tab.get_images_per_audio())
        self.status_bar.showMessage(
            f"✓ {len(paths)} áudio(s) gerado(s). Vá para Imagens para carregar imagens."
        )
        if not self.images_tab.get_images():
            self.tabs.setCurrentWidget(self.images_tab)

    @Slot(list)
    def _on_images_changed(self, paths: List[str]):
        self.video_tab.update_image_paths(paths)
        self.images_tab.update_parity(
            len(self.audio_tab.get_generated_paths()),
            self.video_tab.get_images_per_audio()
        )

    @Slot(str)
    def _on_project_name_changed(self, name: str):
        self.video_tab.suggest_output_path(
            name,
            self.audio_tab.output_root_edit.text().strip() or "output"
        )

    def _restore_session(self):
        if self._session:
            # Migration from old format:
            if "project" in self._session and "audio_tab" not in self._session:
                self.audio_tab.load_session(self._session)
                self.tts_tab.load_session(self._session)
            else:
                if "theme" in self._session:
                    self.theme_combo.setCurrentText(self._session["theme"])
                if "bg_overlay_alpha" in self._session:
                    self.overlay_slider.setValue(self._session["bg_overlay_alpha"])
                    self._bg_overlay_alpha = self._session["bg_overlay_alpha"]
                if "audio_tab" in self._session:
                    self.audio_tab.load_session(self._session["audio_tab"])
                if "language_config_tab" in self._session:
                    self.language_config_tab.load_session(self._session["language_config_tab"])
                if "tts_tab" in self._session:
                    self.tts_tab.load_session(self._session["tts_tab"])
                if "video_tab" in self._session:
                    self.video_tab.load_session(self._session["video_tab"])
                if "settings_tab" in self._session:
                    self.settings_tab.load_session(self._session["settings_tab"])
                if "express_tab" in self._session:
                    self.express_tab.load_session(self._session["express_tab"])

    def get_session(self) -> dict:
        return {
            "theme": self.theme_combo.currentText(),
            "bg_overlay_alpha": self.overlay_slider.value(),
            "audio_tab": self.audio_tab.get_session(),
            "language_config_tab": self.language_config_tab.get_session(),
            "tts_tab": self.tts_tab.get_session(),
            "video_tab": self.video_tab.get_session(),
            "settings_tab": self.settings_tab.get_session(),
            "express_tab": self.express_tab.get_session(),
        }

    def _reset_to_defaults(self):
        reply = QMessageBox.question(self, "Resetar", "Deseja restaurar as configurações padrão em todas as abas?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.audio_tab.reset_defaults()
            self.tts_tab.reset_defaults()
            self.video_tab.reset_defaults()
            self.settings_tab.reset_defaults()
            self.theme_combo.setCurrentText("\U0001f311 Dark (Padr\u00e3o)")
            self._clear_background()
            QMessageBox.information(self, "Resetado", "Configurações restauradas com sucesso.")

    def closeEvent(self, event):
        _save_session(self.get_session())
        
        # Força o encerramento de qualquer geração em andamento
        for tab in [self.audio_tab, self.video_tab]:
            if hasattr(tab, "_pipeline") and tab._pipeline:
                tab._pipeline.cancel()
            if hasattr(tab, "_worker_thread") and tab._worker_thread:
                tab._worker_thread.quit()
                tab._worker_thread.wait(2000)
                

        try:
            import engine
            if engine.MODEL_LOADED and engine.chatterbox_model is not None:
                del engine.chatterbox_model
                engine.chatterbox_model = None
                engine.MODEL_LOADED = False
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Erro ao liberar modelo: {e}")
        super().closeEvent(event)




# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Manhwa Video Creator")
    app.setOrganizationName("Chatterbox Tools")
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLESHEET)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

