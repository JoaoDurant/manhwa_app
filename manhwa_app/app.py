# manhwa_app/app.py
# Aplicação PySide6 principal do Manhwa Video Creator.
# UI dark-themed com 3 abas: Geração de Áudio, Imagens, Geração de Vídeo.

import gc
import json
import logging
import os
import sys
import time
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
    QVBoxLayout, QWidget, QSlider as _QSlider, QComboBox as _QComboBox, QDoubleSpinBox as _QDoubleSpinBox, QSpinBox as _QSpinBox,
    QCheckBox, QInputDialog, QDialog, QTableWidget, QHeaderView,
    QDialogButtonBox, QTableWidgetItem,
)

class QComboBox(_QComboBox):
    def wheelEvent(self, e): e.ignore()

class QSpinBox(_QSpinBox):
    def wheelEvent(self, e): e.ignore()

class QDoubleSpinBox(_QDoubleSpinBox):
    def wheelEvent(self, e): e.ignore()

class QSlider(_QSlider):
    def wheelEvent(self, e): e.ignore()

from manhwa_app.audio_pipeline import split_into_paragraphs
from manhwa_app.video_pipeline import EFFECTS

from config import config_manager as _config_manager
from utils import resolve_voice_path

logger = logging.getLogger(__name__)

# Mapeamento de prefixos de voz Kokoro por idioma
KOKORO_LANG_MAP = {
    "en": ["af_", "am_", "bf_", "bm_"],
    "en-us": ["af_", "am_"],
    "en-gb": ["bf_", "bm_"],
    "pt": ["pf_", "pm_"],
    "pt-br": ["pf_", "pm_"],
    "es": ["ef_", "em_"],
    "fr": ["ff_"],
    "ja": ["jf_", "jm_"],
    "zh": ["zf_", "zm_"],
    "it": ["if_", "im_"],
    "hi": ["hf_", "hm_"]
}

def _append_log(widget: QTextEdit, message: str):
    """Helper to append timestamped logs to a QTextEdit."""
    from datetime import datetime
    time_str = datetime.now().strftime("%H:%M:%S")
    widget.append(f"[{time_str}] {message}")
    widget.verticalScrollBar().setValue(widget.verticalScrollBar().maximum())

# ---------------------------------------------------------------------------
# Thread para carregamento de modelos em background
# ---------------------------------------------------------------------------
class ModelLoaderThread(QThread):
    finished_loading = Signal(bool, str)

    def __init__(self, tts_engine: str, model_type: str, parent=None):
        super().__init__(parent)
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
        self.retries_spin = _ispin(1, 10, 5)
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
        self.phonetic_chk = QCheckBox("Alfabeto Fonético Int. (Tradução para Kokoro TTS)")
        
        self.fx_highpass_chk.setChecked(True)
        self.fx_deesser_chk.setChecked(True)
        self.fx_comp_chk.setChecked(True)
        self.fx_silence_chk.setChecked(True)
        self.fx_loudnorm_chk.setChecked(True)
        self.phonetic_chk.setChecked(False)

        fxg.addWidget(self.spacy_chk, 0, 0, 1, 2)
        fxg.addWidget(self.fx_highpass_chk, 1, 0)
        fxg.addWidget(self.fx_deesser_chk, 1, 1)
        fxg.addWidget(self.fx_comp_chk, 2, 0)
        fxg.addWidget(self.fx_silence_chk, 2, 1)
        fxg.addWidget(self.fx_reverb_chk, 3, 0)
        fxg.addWidget(self.fx_loudnorm_chk, 3, 1)
        fxg.addWidget(self.phonetic_chk, 4, 0, 1, 2)
        
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
        self.sim_spin.setValue(0.75)   # Default 0.75 para garantir que a verificação ocorra
        self.retries_spin.setValue(5)   # 5 tentativas padrão
        
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
            "use_phonetic":  self.phonetic_chk.isChecked(),
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
        if "use_phonetic" in data: self.phonetic_chk.setChecked(data["use_phonetic"])
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
        self.retries_spin.setValue(5)       # 5 tentativas — padrão de alta qualidade
        self.whisper_combo.setCurrentIndex(0)
        self.sim_spin.setValue(0.0)         # 0 = Whisper desabilitado (mais rápido)
        self.spacy_chk.setChecked(False)    # spaCy desabilitado por padrão
        self.phonetic_chk.setChecked(False)
        self.vad_chk.setChecked(False)
        # CORRIGIDO: usar os nomes reais dos widgets de FX
        self.fx_highpass_chk.setChecked(True)
        self.fx_deesser_chk.setChecked(True)
        self.fx_comp_chk.setChecked(True)
        self.fx_silence_chk.setChecked(True)
        self.fx_reverb_chk.setChecked(False)
        self.fx_loudnorm_chk.setChecked(True)



# ---------------------------------------------------------------------------
# Aba 1 — Geração de Áudio
# ---------------------------------------------------------------------------

class AudioTab(QWidget):
    audio_generated = Signal(list)
    run_all_audio_done = Signal(int, str, list)
    # [BUG FIX] Pass-through sinal para MainWindow iniciar o preload durante a geração do AudioPipeline
    engine_switch_requested = Signal(str, str) # engine_str, model_type
    # Sinais passthrough para o Dashboard monitorar geração manual
    pipeline_progress = Signal(int, int)   # current, total
    pipeline_log      = Signal(str)        # mensagem de log
    pipeline_finished = Signal(bool, str)  # success, message

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
        self.preset_lang_combo.currentIndexChanged.connect(self._populate_voices)
        vg.addWidget(self.preset_lang_combo, 1, 1)
        
        
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

        # Overrides da fila (engine / submodelo)
        if hasattr(self, "_queued_overrides") and self._queued_overrides:
            _ot = self._queued_overrides
            if _ot.engine_override:      cfg["tts_engine"] = _ot.engine_override
            if _ot.model_type_override:  cfg["model_type"] = _ot.model_type_override
            self._queued_overrides = None

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
            # CORRIGIDO: Forçar similarity_threshold=0.0 no Preview para evitar o carregamento
            # desnecessário do Faster-Whisper, que causa um atraso de ~8 segundos antes de gerar.
            similarity_threshold=0.0,
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
            # CORRIGIDO: Nomes de FX para AudioPipeline (fx_highpass, fx_compressor, etc.)
            fx_highpass=cfg.get("fx_highpass", False),
            fx_compressor=cfg.get("fx_compressor", False),
            fx_deesser=cfg.get("fx_deesser", False),
            fx_reverb=cfg.get("fx_reverb", False),
            fx_silence=cfg.get("fx_silence", False),
            fx_loudnorm=cfg.get("fx_loudnorm", False),
            use_spacy=cfg.get("use_spacy", False),
            use_phonetic=cfg.get("use_phonetic", False),
        )
        self._preview_thread = QThread(self)
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
                from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
                from PySide6.QtCore import QUrl
                
                if not hasattr(self, "_preview_audio_output"):
                    self._preview_audio_output = QAudioOutput(self)
                    self._preview_audio_output.setVolume(1.0)
                    self._preview_player = QMediaPlayer(self)
                    self._preview_player.setAudioOutput(self._preview_audio_output)
                
                audio_file = audio_files[0]
                self._preview_player.setSource(QUrl.fromLocalFile(str(audio_file)))
                self._preview_player.play()
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
            # ── Vozes built-in do hexgrad/Kokoro-82M ──────────────────────────
            KOKORO_BUILTIN = [
                # American English (Female)
                ("af_heart",   "[EN-US] Heart (F)"),
                ("af_alloy",   "[EN-US] Alloy (F)"),
                ("af_aoede",   "[EN-US] Aoede (F)"),
                ("af_bella",   "[EN-US] Bella (F)"),
                ("af_jessica", "[EN-US] Jessica (F)"),
                ("af_kore",    "[EN-US] Kore (F)"),
                ("af_nicole",  "[EN-US] Nicole (F)"),
                ("af_nova",    "[EN-US] Nova (F)"),
                ("af_river",   "[EN-US] River (F)"),
                ("af_sarah",   "[EN-US] Sarah (F)"),
                ("af_sky",     "[EN-US] Sky (F)"),
                # American English (Male)
                ("am_adam",    "[EN-US] Adam (M)"),
                ("am_echo",    "[EN-US] Echo (M)"),
                ("am_eric",    "[EN-US] Eric (M)"),
                ("am_fenrir",  "[EN-US] Fenrir (M)"),
                ("am_liam",    "[EN-US] Liam (M)"),
                ("am_michael", "[EN-US] Michael (M)"),
                ("am_onyx",    "[EN-US] Onyx (M)"),
                ("am_orion",   "[EN-US] Orion (M)"),
                ("am_puck",    "[EN-US] Puck (M)"),
                ("am_santa",   "[EN-US] Santa (M)"),
                # British English (Female)
                ("bf_alice",   "[EN-GB] Alice (F)"),
                ("bf_emma",    "[EN-GB] Emma (F)"),
                ("bf_isabella","[EN-GB] Isabella (F)"),
                ("bf_lily",    "[EN-GB] Lily (F)"),
                # British English (Male)
                ("bm_daniel",  "[EN-GB] Daniel (M)"),
                ("bm_fable",   "[EN-GB] Fable (M)"),
                ("bm_george",  "[EN-GB] George (M)"),
                ("bm_lewis",   "[EN-GB] Lewis (M)"),
                # Japanese (Female / Male)
                ("jf_alpha",   "[JA-JP] Alpha (F)"),
                ("jf_gongitsune","[JA-JP] Gongitsune (F)"),
                ("jf_nezuko",  "[JA-JP] Nezuko (F)"),
                ("jf_tebukuro","[JA-JP] Tebukuro (F)"),
                ("jm_kumo",    "[JA-JP] Kumo (M)"),
                # Chinese (Female / Male)
                ("zf_xiaobei", "[ZH-CN] Xiaobei (F)"),
                ("zf_xiaoni",  "[ZH-CN] Xiaoni (F)"),
                ("zf_xiaoxiao","[ZH-CN] Xiaoxiao (F)"),
                ("zf_xiaoyi",  "[ZH-CN] Xiaoyi (F)"),
                ("zm_yunjian", "[ZH-CN] Yunjian (M)"),
                ("zm_yunxi",   "[ZH-CN] Yunxi (M)"),
                ("zm_yunxia",  "[ZH-CN] Yunxia (M)"),
                ("zm_yunyang", "[ZH-CN] Yunyang (M)"),
                # Spanish
                ("ef_dora",    "[ES-ES] Dora (F)"),
                ("em_alex",    "[ES-ES] Alex (M)"),
                ("em_santa",   "[ES-ES] Santa (M)"),
                # French
                ("ff_siwis",   "[FR-FR] Siwis (F)"),
                # Hindi
                ("hf_alpha",   "[HI-IN] Alpha (F)"),
                ("hm_omega",   "[HI-IN] Omega (M)"),
                # Italian
                ("if_sara",    "[IT-IT] Sara (F)"),
                ("im_nicola",  "[IT-IT] Nicola (M)"),
                # Portuguese (BR)
                ("pf_dora",    "[PT-BR] Dora (F)"),
                ("pm_alex",    "[PT-BR] Alex (M)"),
                ("pm_santa",   "[PT-BR] Santa (M)"),
            ]
            
            # Filtro por idioma
            lang = self.preset_lang_combo.currentText().lower()
            prefixes = KOKORO_LANG_MAP.get(lang, [])
            
            for voice_id, label in KOKORO_BUILTIN:
                # Se houver filtro para o idioma, aplica; senão mostra tudo (ou nada se preferir)
                if not prefixes or any(voice_id.startswith(p) for p in prefixes):
                    self.preset_voice_combo.addItem(f"🎤 {label}", voice_id)

            # ── Vozes .pt customizadas (pasta local) ──────────────────────────
            base_dir = Path(__file__).parent.parent / "Kokoro-TTS-Local-master" / "voices"
            if base_dir.exists():
                for path in sorted(base_dir.glob("*.pt"), key=natural_sort_key):
                    # Filtra vozes customizadas pelos mesmos prefixos das built-in
                    if not prefixes or any(path.stem.startswith(p) for p in prefixes):
                        self.preset_voice_combo.addItem(f"💾 {path.stem} (Custom)", str(path))
                    
        elif engine == "qwen":
            # Vozes Qwen filtradas por idioma
            lang = self.preset_lang_combo.currentText().lower()
            qwen_presets = {
                "zh": ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric"],
                "en": ["Ryan", "Aiden"],
                "ja": ["Ono_Anna"],
                "ko": ["Sohee"]
            }
            
            relevant = qwen_presets.get(lang, [])
            if not relevant: # Fallback: mostra tudo se o idioma for desconhecido
                 self.preset_voice_combo.addItem("Vivian (Female, Chinese, Lively)", "Vivian")
                 self.preset_voice_combo.addItem("Serena (Female, Chinese, Gentle)", "Serena")
                 self.preset_voice_combo.addItem("Uncle_Fu (Male, Chinese, Mellow)", "Uncle_Fu")
                 self.preset_voice_combo.addItem("Dylan (Male, Beijing Dialect, Clear)", "Dylan")
                 self.preset_voice_combo.addItem("Eric (Male, Sichuan Dialect, Bright)", "Eric")
                 self.preset_voice_combo.addItem("Ryan (Male, English, Dynamic)", "Ryan")
                 self.preset_voice_combo.addItem("Aiden (Male, English, Clear)", "Aiden")
                 self.preset_voice_combo.addItem("Ono_Anna (Female, Japanese, Playful)", "Ono_Anna")
                 self.preset_voice_combo.addItem("Sohee (Female, Korean, Warm)", "Sohee")
            else:
                for v in relevant:
                    # Labels simplificados para o filtro
                    self.preset_voice_combo.addItem(v, v)
                    
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

        if hasattr(self, "_queued_overrides") and self._queued_overrides:
            _ot = self._queued_overrides
            if _ot.engine_override:     cfg["tts_engine"] = _ot.engine_override
            if _ot.model_type_override: cfg["model_type"] = _ot.model_type_override
            self._queued_overrides = None

        voice_val = self.preset_voice_combo.currentData()
        lang_val = self.preset_lang_combo.currentText()
        

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
            # CORRIGIDO: Nomes de FX agora coincidem com o construtor do AudioPipeline
            fx_highpass=cfg.get("fx_highpass", False),
            fx_compressor=cfg.get("fx_compressor", False),
            fx_deesser=cfg.get("fx_deesser", False),
            fx_reverb=cfg.get("fx_reverb", False),
            fx_silence=cfg.get("fx_silence", False),
            fx_loudnorm=cfg.get("fx_loudnorm", False),
            use_spacy=cfg.get("use_spacy", False),
            use_phonetic=cfg.get("use_phonetic", False),
        )
        self._worker_thread = QThread(self)
        self._worker_thread.setStackSize(16 * 1024 * 1024)
        self._pipeline.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._pipeline.run)
        self._pipeline.log_message.connect(self._on_log)
        self._pipeline.progress.connect(self._on_progress)
        self._pipeline.paragraph_done.connect(self._on_pdone)
        self._pipeline.finished.connect(self._on_done)
        # Passa o sinal do pipeline (thread) pra cima para a MainWindow (UI)
        self._pipeline.engine_switch_needed.connect(self.engine_switch_requested.emit)
        # Sinais passthrough para o Dashboard
        self._pipeline.progress.connect(self.pipeline_progress.emit)
        self._pipeline.log_message.connect(self.pipeline_log.emit)
        self._pipeline.finished.connect(self.pipeline_finished.emit)
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
            seed=cfg.get("seed", 0),
            use_phonetic=cfg.get("use_phonetic", False)
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

    def configure_and_run_queued(self, task: "QueueTask"):
        """Configura a aba Áudio para uma tarefa da fila e inicia geração.

        A fila usa as configurações do sistema (TTS, Áudio, Imagens, etc.) como base.
        Apenas os campos com override definido na tarefa são substituídos.
        """
        self.project_edit.setText(task.project_name)
        lang  = task.lang_override  or self.preset_lang_combo.currentText()
        voice = task.voice_override or ""
        self._files = [{"path": task.txt_path, "voice": voice, "lang": lang}]
        self._refresh_list()

        if task.lang_override:
            idx = self.preset_lang_combo.findText(task.lang_override)
            if idx >= 0: self.preset_lang_combo.setCurrentIndex(idx)

        if task.voice_override:
            idx = self.preset_voice_combo.findData(task.voice_override)
            if idx >= 0:
                self.preset_voice_combo.setCurrentIndex(idx)
            else:
                # Path de arquivo de clonagem (.wav/.mp3/.pt) — adiciona temporariamente
                from pathlib import Path as _P
                if _P(task.voice_override).exists():
                    self.preset_voice_combo.addItem(
                        f"📎 {_P(task.voice_override).name} (fila)", task.voice_override
                    )
                    idx = self.preset_voice_combo.findData(task.voice_override)
                    if idx >= 0: self.preset_voice_combo.setCurrentIndex(idx)

        # Propaga engine e submodelo override para a TtsConfigTab (mantém resto do sistema)
        if task.engine_override or task.model_type_override:
            main_win = self.window()
            if hasattr(main_win, "tts_tab"):
                tts = main_win.tts_tab
                if task.engine_override:
                    idx = tts.engine_combo.findData(task.engine_override)
                    if idx >= 0: tts.engine_combo.setCurrentIndex(idx)
                if task.model_type_override:
                    idx = tts.model_combo.findData(task.model_type_override)
                    if idx >= 0: tts.model_combo.setCurrentIndex(idx)

        self._queued_overrides = task
        self._start_normal()


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
        self.layout_combo.addItem("Alternar: Sequencial (3-5)", "mixed_seq")
        self.layout_combo.addItem("Alternar: Aleatório (70/30)", "mixed_prob")
        self.layout_combo.currentIndexChanged.connect(self._on_layout_changed)
        fxv.addWidget(self.layout_combo)

        fxv.addWidget(QLabel("Efeito de Câmera:"))
        self.effect_combo = QComboBox()
        self.effect_combo.addItem("Automático", "auto")
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
        
        self.chk_denoise = QCheckBox("Denoise (HQDN3D)")
        self.chk_denoise.setChecked(False)
        self.chk_denoise.setToolTip("Reduz granulação da imagem/compressão antes da renderização")
        prod_grid.addWidget(self.chk_denoise, 2, 1)

        self.chk_vibrance = QCheckBox("Vibrance (Cores HDR)")
        self.chk_vibrance.setChecked(False)
        self.chk_vibrance.setToolTip("Força o reequilíbrio de Gamma e Saturação extrema para imagens muito escuras")
        prod_grid.addWidget(self.chk_vibrance, 3, 0)
        
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
                # Regra estrita: Se for muito vertical (height > width * 1.4), NÃO usar split
                if ratio < 0.71: return False  
                # Paisagem (horizontal), também costuma amassar duas na mesma tela, bloqueamos acima de 1.2
                if ratio > 1.2: return False  
                return True
        except:
            return False

    def _build_mixed_pairs(self, mode="mixed_seq"):
        pairs = []
        a_idx = i_idx = 0
        import random
        random.seed(time.time())
        
        # Parâmetros conforme o modo
        next_split_target = 2 if mode == "mixed_seq" else random.randint(3, 5)
        current_counter = 0

        while a_idx < len(self._audio_paths):
            can_split = (a_idx + 1 < len(self._audio_paths)) and (i_idx + 1 < len(self._image_paths))
            make_split = False
            
            if can_split:
                if mode == "mixed_seq":
                    # Forçar um split exatamente a cada 2 cenas simples
                    if current_counter >= next_split_target:
                        make_split = True
                else:
                    # Regra 70/30: Probabilidade pura (30% de chance de split)
                    if random.random() < 0.30:
                        make_split = True
                
                # Independente do modo, fotos verticais e proporções devem ser validadas - Regra 5
                if make_split:
                    if not self._can_pair(self._image_paths[i_idx]) or not self._can_pair(self._image_paths[i_idx+1]):
                        make_split = False
            
            if make_split:
                pairs.append((
                    (self._audio_paths[a_idx], self._audio_paths[a_idx+1]),
                    (self._image_paths[i_idx], self._image_paths[i_idx+1])
                ))
                a_idx += 2
                i_idx += 2
                current_counter = 0
                if mode != "mixed_seq":
                    next_split_target = random.randint(3, 5)
            else:
                img = self._image_paths[i_idx] if i_idx < len(self._image_paths) else None
                if img:
                    pairs.append((self._audio_paths[a_idx], img))
                    i_idx += 1
                a_idx += 1
                current_counter += 1
                
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
            [str(f) for f in p.rglob("*") if f.is_file() and f.suffix.lower() in self._AUDIO_EXTS and not f.name.endswith(("_best.wav", "_tmp.wav", "_silence.wav"))],
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
            [str(f) for f in p.rglob("*") if f.is_file() and f.suffix.lower() in self._AUDIO_EXTS and not f.name.endswith(("_best.wav", "_tmp.wav", "_silence.wav"))],
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
        elif mode in ["mixed_seq", "mixed_prob"]:
            pairs = self._build_mixed_pairs(mode=mode)
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
                                       layout_mode=mode, # Passando o modo Explicitamente (mixed, split, single)
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
                from PySide6.QtCore import QUrl, QTimer
                self._on_log(f"⌛ Preparando visualização de: {Path(self._last_preview_path).name}...")
                
                # Para qualquer reprodução anterior e limpa o cache do player
                self.vp_player.stop()
                self.vp_player.setSource(QUrl())
                
                # Pequeno delay (500ms) para garantir que o SO liberou o arquivo após o FFmpeg fechar
                def delayed_play():
                    if Path(self._last_preview_path).exists():
                        self.vp_player.setSource(QUrl.fromLocalFile(self._last_preview_path))
                        self.vp_player.play()
                        self._on_log(f"▶ Reproduzindo Preview na janela.")
                
                QTimer.singleShot(600, delayed_play)
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
                    "denoise":        self.chk_denoise.isChecked() if hasattr(self, "chk_denoise") else False,
                    "vibrance":       self.chk_vibrance.isChecked() if hasattr(self, "chk_vibrance") else False,
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
        if hasattr(self, "chk_denoise"):
            self.chk_denoise.setChecked(vid.get("denoise", False))
        if hasattr(self, "chk_vibrance"):
            self.chk_vibrance.setChecked(vid.get("vibrance", False))
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
        if hasattr(self, "chk_denoise"):
            self.chk_denoise.setChecked(False)
        if hasattr(self, "chk_vibrance"):
            self.chk_vibrance.setChecked(False)
        if hasattr(self, "chk_auto_ducking"):
            self.chk_auto_ducking.setChecked(True)


# ---------------------------------------------------------------------------
# Aba de Configurações
# ---------------------------------------------------------------------------

class SettingsTab(QWidget):
    """Aba de configurações gerais: aparência, paths e sessão."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        vl = QVBoxLayout(inner)
        vl.setContentsMargins(24, 20, 24, 24)
        vl.setSpacing(18)
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        # ── 1. Aparência ──────────────────────────────────────────────
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

        # ── 2. Paths Padrão ───────────────────────────────────────────
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

        # ── 3. Sessão ─────────────────────────────────────────────────
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
    # Session persistence
    # ------------------------------------------------------------------

    def get_session(self) -> dict:
        return {
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
        # Compatibilidade com sessões antigas que tinham chave "gemini"
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
        self.theme_combo.setCurrentText("🌑 Dark (Padrão)")
        self.overlay_slider.setValue(180)
        self.default_output_edit.setText("output")
        self.default_voices_edit.setText("voices")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_theme_changed(self, name: str):
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

    def _update_prompt_lang_combo(self):
        self.combo_prompt_lang.blockSignals(True)
        current = self.combo_prompt_lang.currentData()
        self.combo_prompt_lang.clear()
        self.combo_prompt_lang.addItem("Genérico (Padrão)", "Generic")
        
        mapping = [
            ("Inglês (en)", "en"), ("Espanhol (es)", "es"), ("Francês (fr)", "fr"),
            ("Alemão (de)", "de"), ("Japonês (ja)", "ja"), ("Coreano (ko)", "ko"),
        ]
        for name, code in mapping:
            self.combo_prompt_lang.addItem(name, code)
            
        idx = self.combo_prompt_lang.findData(current)
        if idx >= 0:
            self.combo_prompt_lang.setCurrentIndex(idx)
        self.combo_prompt_lang.blockSignals(False)

    def _on_prompt_lang_changed(self, index):
        lang_code = self.combo_prompt_lang.currentData()
        self._current_prompt_lang = lang_code
        
        self.translation_prompt_edit.blockSignals(True)
        if lang_code == "Generic":
            self.translation_prompt_edit.setPlainText(self.get_session()["gemini"].get("translation_prompt", _DEFAULT_TRANSLATION_PROMPT))
        else:
            prompt = self._per_lang_prompts.get(lang_code, _DEFAULT_TRANSLATION_PROMPT)
            self.translation_prompt_edit.setPlainText(prompt)
        self.translation_prompt_edit.blockSignals(False)

    def _on_translation_prompt_edited(self):
        lang_code = self._current_prompt_lang
        text = self.translation_prompt_edit.toPlainText()
        if lang_code and lang_code != "Generic":
            self._per_lang_prompts[lang_code] = text
        # If generic, it's handled by get_session directly from the widget

    def _reset_current_translation_prompt(self):
        self.translation_prompt_edit.setPlainText(_DEFAULT_TRANSLATION_PROMPT)
        self._on_translation_prompt_edited()


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

# ---------------------------------------------------------------------------
# Queue System — Task + Orchestrator + Tabs
# ---------------------------------------------------------------------------

class QueueTask:
    """Representa uma tarefa na fila de geração."""
    MODES = {
        "audio":           "🔊 Só Áudio",
        "audio+video":     "🎬 Áudio + Vídeo",
        "audio+video+alt": "🖼 Áudio + Vídeo (outras imagens)",
    }

    def __init__(self, project_name: str, txt_path: str, mode: str,
                 engine_override: str = None, model_type_override: str = None,
                 voice_override: str = None, lang_override: str = None,
                 img_path: str = None):
        self.project_name        = project_name
        self.txt_path            = txt_path
        self.mode                = mode
        self.engine_override     = engine_override
        self.model_type_override = model_type_override
        self.voice_override      = voice_override
        self.lang_override       = lang_override
        self.img_path            = img_path
        self.status              = "pending"   # pending/running/done/error
        self.progress_current    = 0
        self.progress_total      = 0

    def mode_label(self):
        return self.MODES.get(self.mode, self.mode)

    def engine_label(self):
        return self.engine_override or "— atual —"


class QueueCoordinator(QObject):
    """Coordenador de fila event-driven. Usa o pipeline da aba Audio."""
    task_started  = Signal(int)
    task_progress = Signal(int, int, int)
    task_log      = Signal(int, str)
    task_finished = Signal(int, bool)
    queue_log     = Signal(str)
    all_done      = Signal()

    def __init__(self, tasks: list, main_window, parent=None):
        super().__init__(parent)
        self.tasks       = list(tasks)
        self.main_window = main_window
        self._idx        = 0
        self._cancelled  = False
        self._audio_paths: list = []

    def cancel(self):
        self._cancelled = True
        at = self.main_window.audio_tab
        if at._pipeline:
            at._pipeline.cancel()

    def start(self):
        if not self.tasks:
            self.all_done.emit(); return
        self._idx = 0
        self._cancelled = False
        self._connect_audio()
        self._run_current()

    def _connect_audio(self):
        at = self.main_window.audio_tab
        at.pipeline_progress.connect(self._on_progress)
        at.pipeline_log.connect(self._on_log)
        at.pipeline_finished.connect(self._on_audio_done)
        at.audio_generated.connect(self._on_generated)

    def _disconnect_audio(self):
        at = self.main_window.audio_tab
        for sig, slot in [
            (at.pipeline_progress, self._on_progress),
            (at.pipeline_log,      self._on_log),
            (at.pipeline_finished, self._on_audio_done),
            (at.audio_generated,   self._on_generated),
        ]:
            try: sig.disconnect(slot)
            except: pass

    def _run_current(self):
        if self._cancelled or self._idx >= len(self.tasks):
            self._finish(); return
        task = self.tasks[self._idx]
        task.status = "running"
        self.task_started.emit(self._idx)
        n = self._idx + 1; total = len(self.tasks)
        self.queue_log.emit(f"\nProcessando [{n}/{total}] {task.project_name} ({task.mode_label()})")
        self.main_window.audio_tab.configure_and_run_queued(task)

    @Slot(int, int)
    def _on_progress(self, current: int, total: int):
        self.task_progress.emit(self._idx, current, total)

    @Slot(str)
    def _on_log(self, msg: str):
        self.task_log.emit(self._idx, msg)

    @Slot(list)
    def _on_generated(self, paths: list):
        self._audio_paths = paths

    @Slot(bool, str)
    def _on_audio_done(self, success: bool, msg: str):
        task = self.tasks[self._idx]
        if not success:
            task.status = "error"
            self.queue_log.emit("Erro: " + msg)
            self.task_finished.emit(self._idx, False)
            self._idx += 1; self._run_current(); return
        if task.mode in ("audio+video", "audio+video+alt"):
            self._start_video(task)
        else:
            task.status = "done"
            self.task_finished.emit(self._idx, True)
            self._idx += 1; self._run_current()

    def _start_video(self, task):
        from manhwa_app.video_pipeline import VideoPipeline
        w = self.main_window
        out_root = w.audio_tab.output_root_edit.text().strip() or "output"
        if task.mode == "audio+video+alt" and task.img_path:
            images = sorted(
                [str(p) for p in Path(task.img_path).glob("*")
                 if p.suffix.lower() in (".jpg", ".png", ".webp", ".jpeg")],
                key=natural_sort_key
            )
        else:
            images = w.images_tab.get_images()
        audios_dir = Path(out_root) / task.project_name / "audios"
        audios = sorted(
            [str(p) for p in audios_dir.glob("audio_*.wav")],
            key=lambda x: int(Path(x).stem.split("_")[1])
                          if Path(x).stem.split("_")[1].isdigit() else 0
        )
        if not (audios and images):
            self.task_log.emit(self._idx, "Sem audios/imagens para video.")
            task.status = "done"
            self.task_finished.emit(self._idx, True)
            self._idx += 1; self._run_current(); return
        vcfg = w.video_tab.get_session()
        pairs = list(zip(audios, images[:len(audios)]))
        out_vid = str(Path(out_root) / task.project_name / f"{task.project_name}_final.mp4")
        self._v_pipe = VideoPipeline(
            pairs=pairs, output_path=out_vid,
            effect_mode=vcfg.get("effect", "auto"),
            transition_mode=vcfg.get("transition", "fade"),
            transition_time=vcfg.get("transition_time", 0.5),
        )
        self._v_pipe.log_message.connect(
            lambda m, i=self._idx: self.task_log.emit(i, m)
        )
        self._v_thread = QThread(self)
        self._v_pipe.moveToThread(self._v_thread)
        self._v_thread.started.connect(self._v_pipe.run)
        self._v_pipe.finished.connect(self._on_video_done)
        self._v_thread.start()

    @Slot(bool, str)
    def _on_video_done(self, success: bool, msg: str):
        self._v_thread.quit()
        task = self.tasks[self._idx]
        task.status = "done" if success else "error"
        self.task_finished.emit(self._idx, success)
        self._idx += 1; self._run_current()

    def _finish(self):
        self._disconnect_audio()
        self.queue_log.emit("\nFila concluida!")
        self.all_done.emit()


# Alias
QueueOrchestrator = QueueCoordinator

# ---------------------------------------------------------------------------
# Dashboard Tab
# ---------------------------------------------------------------------------
class DashboardTab(QWidget):
    """Central de monitoramento em tempo real."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._task_rows: list = []   # list of (label_status, label_name, bar, label_pct)
        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(14)

        # ── Status cards ────────────────────────────────────────────
        cards_row = QHBoxLayout()
        cards_row.setSpacing(12)

        def _card(title, value_attr):
            card = QFrame()
            card.setObjectName("dashCard")
            card.setStyleSheet(
                "QFrame#dashCard { background: #1a1a2e; border: 1px solid #2d2d4a;"
                " border-radius: 10px; padding: 10px; }"
            )
            cv = QVBoxLayout(card)
            cv.setSpacing(4)
            lbl_t = QLabel(title)
            lbl_t.setStyleSheet("color:#8888aa; font-size:11px;")
            lbl_v = QLabel("—")
            lbl_v.setStyleSheet("font-size:15px; font-weight:700; color:#e0e0ff;")
            lbl_v.setObjectName(value_attr)
            cv.addWidget(lbl_t)
            cv.addWidget(lbl_v)
            setattr(self, value_attr, lbl_v)
            return card

        cards_row.addWidget(_card("🖥 GPU / VRAM",   "lbl_gpu"))
        cards_row.addWidget(_card("🤖 Modelo TTS",   "lbl_model"))
        cards_row.addWidget(_card("⛓ Fila",          "lbl_queue_status"))
        cards_row.addWidget(_card("✅ Concluídos",    "lbl_done"))
        root.addLayout(cards_row)

        # ── Geração atual (aba Áudio) ────────────────────────────────
        manual_group = QGroupBox("⚡ Geração Manual (Aba Áudio)")
        mgv = QVBoxLayout(manual_group)
        self.manual_progress = QProgressBar()
        self.manual_progress.setRange(0, 100)
        self.manual_progress.setValue(0)
        self.manual_progress.setFormat("Aguardando…")
        self.manual_progress.setFixedHeight(22)
        mgv.addWidget(self.manual_progress)
        root.addWidget(manual_group)

        # ── Fila de tarefas ─────────────────────────────────────────
        queue_group = QGroupBox("⛓ Tarefas na Fila")
        qv = QVBoxLayout(queue_group)
        self.tasks_scroll = QScrollArea()
        self.tasks_scroll.setWidgetResizable(True)
        self.tasks_scroll.setMaximumHeight(240)
        self.tasks_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.tasks_container = QWidget()
        self.tasks_layout = QVBoxLayout(self.tasks_container)
        self.tasks_layout.setSpacing(4)
        self.tasks_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.tasks_scroll.setWidget(self.tasks_container)
        qv.addWidget(self.tasks_scroll)
        self.lbl_empty = QLabel("Nenhuma tarefa na fila.")
        self.lbl_empty.setStyleSheet("color:#666; font-style:italic;")
        self.lbl_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        qv.addWidget(self.lbl_empty)
        root.addWidget(queue_group)

        # ── Log unificado ────────────────────────────────────────────
        log_group = QGroupBox("📋 Log Unificado")
        lgv = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setStyleSheet("background:#0f0f1a; color:#b0b0d0; border-radius:4px;")
        lgv.addWidget(self.log_text)
        btn_clear = QPushButton("🗑 Limpar Log")
        btn_clear.setFixedWidth(120)
        btn_clear.clicked.connect(self.log_text.clear)
        lgv.addWidget(btn_clear)
        root.addWidget(log_group, 1)

    # --- Public API used by MainWindow / QueueOrchestrator connections ---

    def update_gpu_info(self, info: str):
        self.lbl_gpu.setText(info)

    def update_model_info(self, info: str):
        self.lbl_model.setText(info)

    @Slot(int, int)
    def on_manual_progress(self, current: int, total: int):
        if total > 0:
            pct = int(current * 100 / total)
            self.manual_progress.setValue(pct)
            self.manual_progress.setFormat(f"Parágrafo {current}/{total}  ({pct}%)")

    @Slot(str)
    def on_manual_log(self, msg: str):
        _append_log(self.log_text, msg)

    @Slot(bool, str)
    def on_manual_finished(self, success: bool, msg: str):
        self.manual_progress.setValue(100 if success else 0)
        self.manual_progress.setFormat("✅ Concluído" if success else "❌ Erro")
        _append_log(self.log_text, f"{'✅' if success else '❌'} [Áudio Manual] {msg}")

    def rebuild_task_rows(self, tasks: list):
        """Recria a lista de linhas de tarefas no dashboard."""
        # Clear
        while self.tasks_layout.count():
            item = self.tasks_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._task_rows.clear()

        self.lbl_empty.setVisible(len(tasks) == 0)
        self.tasks_scroll.setVisible(len(tasks) > 0)

        for t in tasks:
            row_w = QFrame()
            row_w.setStyleSheet(
                "QFrame { background:#161628; border:1px solid #2a2a40;"
                " border-radius:6px; padding:4px 8px; }"
            )
            rl = QHBoxLayout(row_w)
            rl.setContentsMargins(4, 2, 4, 2)
            rl.setSpacing(8)

            lbl_s = QLabel("⏳")
            lbl_s.setFixedWidth(22)
            lbl_n = QLabel(f"<b>{t.project_name}</b>  <span style='color:#888'>{t.mode_label()} | {t.engine_label()}</span>")
            lbl_n.setTextFormat(Qt.RichText)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFixedWidth(140)
            bar.setFixedHeight(14)
            lbl_pct = QLabel("0%")
            lbl_pct.setFixedWidth(34)
            lbl_pct.setStyleSheet("font-size:10px; color:#888;")

            rl.addWidget(lbl_s)
            rl.addWidget(lbl_n, 1)
            rl.addWidget(bar)
            rl.addWidget(lbl_pct)
            self.tasks_layout.addWidget(row_w)
            self._task_rows.append((lbl_s, bar, lbl_pct))

        self._update_queue_card(tasks)

    def _update_queue_card(self, tasks):
        pending = sum(1 for t in tasks if t.status == "pending")
        running = sum(1 for t in tasks if t.status == "running")
        done    = sum(1 for t in tasks if t.status in ("done", "error"))
        self.lbl_queue_status.setText(f"⏳{pending}  ⚡{running}")
        self.lbl_done.setText(str(done))

    @Slot(int)
    def on_task_started(self, idx: int):
        if idx < len(self._task_rows):
            lbl_s, bar, lbl_pct = self._task_rows[idx]
            lbl_s.setText("⚡")
            bar.setValue(0)
            lbl_pct.setText("0%")

    @Slot(int, int, int)
    def on_task_progress(self, idx: int, current: int, total: int):
        if idx < len(self._task_rows):
            _, bar, lbl_pct = self._task_rows[idx]
            pct = int(current * 100 / total) if total > 0 else 0
            bar.setValue(pct)
            lbl_pct.setText(f"{pct}%")

    @Slot(int, str)
    def on_task_log(self, idx: int, msg: str):
        _append_log(self.log_text, msg)

    @Slot(int, bool)
    def on_task_finished(self, idx: int, success: bool):
        if idx < len(self._task_rows):
            lbl_s, bar, lbl_pct = self._task_rows[idx]
            lbl_s.setText("✅" if success else "❌")
            bar.setValue(100 if success else bar.value())

    @Slot(str)
    def on_queue_log(self, msg: str):
        _append_log(self.log_text, msg)


# ---------------------------------------------------------------------------
# Queue Tab
# ---------------------------------------------------------------------------
class QueueTab(QWidget):
    """Aba de controle da fila de geração."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tasks: list = []
        self._orchestrator = None
        self._current_txt: str = ""
        self._alt_img_path: str = ""
        self._setup_ui()

    def _setup_ui(self):
        lv = QVBoxLayout(self)
        lv.setContentsMargins(15, 15, 15, 15)
        lv.setSpacing(12)

        add_group = QGroupBox("➕ Adicionar Tarefa")
        fl = QGridLayout(add_group)
        fl.setSpacing(10)

        row = 0
        fl.addWidget(QLabel("Nome do Projeto:"), row, 0)
        self.proj_edit = QLineEdit("projeto_fila")
        fl.addWidget(self.proj_edit, row, 1, 1, 2); row += 1

        fl.addWidget(QLabel("Arquivo .txt:"), row, 0)
        self.btn_txt = QPushButton("📄 Selecionar…")
        self.btn_txt.clicked.connect(self._pick_txt)
        fl.addWidget(self.btn_txt, row, 1, 1, 2); row += 1
        self.lbl_txt = QLabel("(nenhum)")
        self.lbl_txt.setStyleSheet("color:#888; font-size:10px;")
        fl.addWidget(self.lbl_txt, row, 0, 1, 3); row += 1

        fl.addWidget(QLabel("Modo:"), row, 0)
        self.mode_combo = QComboBox()
        for k, v in QueueTask.MODES.items():
            self.mode_combo.addItem(v, k)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        fl.addWidget(self.mode_combo, row, 1, 1, 2); row += 1

        fl.addWidget(QLabel("Engine Override:"), row, 0)
        self.q_engine_combo = QComboBox()
        self.q_engine_combo.addItem("— Usar Atual —", None)
        self.q_engine_combo.addItem("Chatterbox TTS", "chatterbox")
        self.q_engine_combo.addItem("Kokoro TTS", "kokoro")
        self.q_engine_combo.addItem("Qwen3-TTS", "qwen")
        self.q_engine_combo.addItem("IndexTTS (Zero-Shot)", "indextts")
        self.q_engine_combo.currentIndexChanged.connect(self._on_q_engine_changed)
        fl.addWidget(self.q_engine_combo, row, 1)

        # Submodelo — apenas visível quando Chatterbox está selecionado
        self.q_model_combo = QComboBox()
        self.q_model_combo.addItem("— Submodelo Atual —", None)
        self.q_model_combo.addItem("⚡ Turbo  (Inglês, rápido)", "turbo")
        self.q_model_combo.addItem("🌐 Multilingual  (Vários idiomas)", "multilingual")
        self.q_model_combo.addItem("🔊 Original  (Expressivo)", "original")
        self.q_model_combo.setToolTip("Apenas para Chatterbox TTS")
        fl.addWidget(self.q_model_combo, row, 2); row += 1

        fl.addWidget(QLabel("Voz (ID ou Caminho):"), row, 0)
        self.q_voice_edit = QLineEdit()
        self.q_voice_edit.setPlaceholderText("ex: pf_dora / voice.pt")
        fl.addWidget(self.q_voice_edit, row, 1)
        btn_v_browse = QPushButton("📁")
        btn_v_browse.setFixedWidth(35)
        btn_v_browse.clicked.connect(self._pick_voice_file)
        fl.addWidget(btn_v_browse, row, 2); row += 1

        fl.addWidget(QLabel("Idioma:"), row, 0)
        self.q_lang_combo = QComboBox()
        self.q_lang_combo.addItem("— Usar Atual —", None)
        for lg in ["pt", "en", "es", "fr", "ja", "ko", "zh"]:
            self.q_lang_combo.addItem(lg, lg)
        fl.addWidget(self.q_lang_combo, row, 1, 1, 2); row += 1

        self.lbl_img = QLabel("Pasta de Imagens:")
        fl.addWidget(self.lbl_img, row, 0)
        self.img_row = QWidget()
        ir = QHBoxLayout(self.img_row)
        ir.setContentsMargins(0, 0, 0, 0)
        self.btn_imgs = QPushButton("🖼 Selecionar…")
        self.btn_imgs.clicked.connect(self._pick_imgs)
        ir.addWidget(self.btn_imgs)
        fl.addWidget(self.img_row, row, 1, 1, 2); row += 1
        self.lbl_imgs = QLabel("(aba Imagens)")
        self.lbl_imgs.setStyleSheet("color:#888; font-size:10px;")
        fl.addWidget(self.lbl_imgs, row, 0, 1, 3); row += 1

        btn_add = QPushButton("➕ Adicionar à Fila")
        btn_add.setObjectName("primary")
        btn_add.setMinimumHeight(40)
        btn_add.clicked.connect(self._add_task)
        fl.addWidget(btn_add, row, 0, 1, 3); row += 1
        lv.addWidget(add_group)

        lv.addWidget(QLabel("📋 Tarefas Pendentes:"))
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.task_list_container = QWidget()
        self._list_layout = QVBoxLayout(self.task_list_container)
        self._list_layout.setContentsMargins(5, 5, 5, 5)
        self._list_layout.setSpacing(8)
        self._list_layout.addStretch()
        self.scroll.setWidget(self.task_list_container)
        lv.addWidget(self.scroll)

        ctrl_group = QWidget()
        cl = QHBoxLayout(ctrl_group)
        cl.setContentsMargins(0, 5, 0, 0)
        self.btn_start = QPushButton("🚀 Iniciar Fila")
        self.btn_start.setObjectName("primary")
        self.btn_start.setMinimumHeight(40)
        self.btn_start.clicked.connect(self._start_queue)
        self.btn_stop = QPushButton("🛑 Parar")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_queue)
        cl.addWidget(self.btn_start)
        cl.addWidget(self.btn_stop)
        lv.addWidget(ctrl_group)

    def _pick_txt(self):
        f, _ = QFileDialog.getOpenFileName(self, "Arquivo de Texto", "", "Text Files (*.txt)")
        if f:
            self._current_txt = f
            self.lbl_txt.setText(Path(f).name)
            p_name = Path(f).stem.replace(" ", "_").lower()
            self.proj_edit.setText(p_name)

    def _pick_imgs(self):
        f = QFileDialog.getExistingDirectory(self, "Pasta de Imagens")
        if f:
            self._alt_img_path = f
            self.lbl_imgs.setText(Path(f).name)

    def _pick_voice_file(self):
        f, _ = QFileDialog.getOpenFileName(self, "Arquivo de Voz", "", "Voz (*.pt *.wav *.mp3 *.pth)")
        if f:
            self.q_voice_edit.setText(f)

    def _on_q_engine_changed(self):
        """Mostra o submodelo apenas quando Chatterbox está selecionado (igual ao TtsConfigTab)."""
        eng = self.q_engine_combo.currentData()
        is_chatterbox = (eng == "chatterbox" or eng is None)
        self.q_model_combo.setVisible(is_chatterbox)

    def _on_mode_changed(self):
        m = self.mode_combo.currentData()
        is_alt = (m == "audio+video+alt")
        self.lbl_img.setVisible(is_alt)
        self.img_row.setVisible(is_alt)
        self.lbl_imgs.setVisible(is_alt)

    def _add_task(self):
        if not self._current_txt:
            QMessageBox.warning(self, "Erro", "Selecione um arquivo .txt")
            return
        mode = self.mode_combo.currentData()
        voice_val = self.q_voice_edit.text().strip() or None
        task = QueueTask(
            project_name=self.proj_edit.text().strip() or "projeto",
            txt_path=self._current_txt,
            mode=mode,
            engine_override=self.q_engine_combo.currentData(),
            model_type_override=self.q_model_combo.currentData(),
            voice_override=voice_val,
            lang_override=self.q_lang_combo.currentData(),
            img_path=self._alt_img_path if mode == "audio+video+alt" else None
        )
        self._tasks.append(task)
        self._refresh_list()
    def _notify_dashboard(self):
        w = self.window()
        if hasattr(w, "dashboard_tab"):
            w.dashboard_tab.rebuild_task_rows(self._tasks)

    def _start_queue(self):
        if not self._tasks:
            QMessageBox.warning(self, "Fila Vazia", "Adicione tarefas primeiro.")
            return
        if self._orchestrator and self._orchestrator.isRunning():
            return

        w = self.window()
        self._orchestrator = QueueOrchestrator(self._tasks, w, parent=self)

        if hasattr(w, "dashboard_tab"):
            db = w.dashboard_tab
            db.rebuild_task_rows(self._tasks)
            self._orchestrator.task_started.connect(db.on_task_started)
            self._orchestrator.task_progress.connect(db.on_task_progress)
            self._orchestrator.task_log.connect(db.on_task_log)
            self._orchestrator.task_finished.connect(db.on_task_finished)
            self._orchestrator.queue_log.connect(db.on_queue_log)

        # engine auto-switch
        self._orchestrator.engine_switch_needed.connect(w._handle_audio_tab_engine_switch)

        self._orchestrator.task_started.connect(self._on_task_started)
        self._orchestrator.task_finished.connect(self._on_task_finished)
        self._orchestrator.all_done.connect(self._on_all_done)

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        w.dashboard_tab and w.tabs.setCurrentWidget(w.dashboard_tab)
        self._orchestrator.start()

    def _on_task_started(self, idx: int):
        if idx < len(self._tasks):
            self._tasks[idx].status = "running"
        self._refresh_list()

    def _on_task_finished(self, idx: int, success: bool):
        if idx < len(self._tasks):
            self._tasks[idx].status = "done" if success else "error"
        self._refresh_list()

    def _stop_queue(self):
        if self._orchestrator:
            self._orchestrator.cancel()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _on_all_done(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        QMessageBox.information(self, "Fila Concluída", "Todas as tarefas foram processadas!")

    def get_session(self) -> dict:
        return {}   # tasks are transient — not persisted across sessions

    def load_session(self, data: dict):
        pass


# ---------------------------------------------------------------------------
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

    def closeEvent(self, event):
        """[BUG 1 FIX] Encerra threads de preview antes de fechar."""
        if hasattr(self, "language_config_tab"):
            tab = self.language_config_tab
            if hasattr(tab, "_preview_thread") and tab._preview_thread is not None:
                if tab._preview_thread.isRunning():
                    logger.info("[FIX] closeEvent: encerrando PreviewThread antes de sair")
                    if hasattr(tab, "_preview_pipeline") and tab._preview_pipeline:
                        tab._preview_pipeline.cancel()
                    tab._preview_thread.requestInterruption()
                    tab._preview_thread.quit()
                    tab._preview_thread.wait(5000)
        super().closeEvent(event)

    def trigger_model_preload(self, explicit_engine=None, explicit_model_type=None):
        """Dispara o carregamento do modelo em background."""
        cfg = self.tts_tab.get_session() if hasattr(self, "tts_tab") else {}
        engine_name = explicit_engine if explicit_engine else cfg.get("tts_engine", "chatterbox")
        model_type = explicit_model_type if explicit_model_type else cfg.get("model_type", "turbo")

        if hasattr(self, "_loader_thread") and self._loader_thread and self._loader_thread.isRunning():
            return # Já carregando

        self.model_status_label.setText("TTS: carregando…")
        self.model_status_label.setStyleSheet("color:#f0b040;font-size:11px;")
        
        if hasattr(self, "audio_tab") and hasattr(self.audio_tab, "btn_generate"):
            self.audio_tab.btn_generate.setEnabled(False)
            self.audio_tab.btn_generate.setText("Carregando modelo...")
        
        self._loader_thread = ModelLoaderThread(engine_name, model_type, parent=self)
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

        # [AUDIO PIPELINE FIX] Liberar a QThread se estivermos rodando um batch de múltiplos idiomas
        if hasattr(self, "audio_tab") and hasattr(self.audio_tab, "_pipeline") and self.audio_tab._pipeline is not None:
             self.audio_tab._pipeline.confirm_switch_done()

    @Slot(str, str)
    def _handle_audio_tab_engine_switch(self, engine_str: str, model_type: str):
        """[BUG FIX] Disparado durante QThread da AudioPipeline quando múltiplos idiomas exigem troca."""
        logger.info(f"[MAIN_WINDOW] AudioPipeline solicitou troca de engine para {engine_str}/{model_type}")
        self.trigger_model_preload(explicit_engine=engine_str, explicit_model_type=model_type)

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

        # ── Tabs ──────────────────────────────────────────────────────────────────
        self.tabs = QTabWidget()

        self.dashboard_tab = DashboardTab()

        self.audio_tab     = AudioTab()

        self.tts_tab       = TtsConfigTab()

        self.images_tab    = ImagesTab()

        self.video_tab     = VideoTab()

        self.settings_tab  = SettingsTab()

        self.queue_tab     = QueueTab()

        self.tabs.addTab(self.dashboard_tab, "📊 Dashboard")

        self.tabs.addTab(self.audio_tab,    "📝 Áudio")

        self.tabs.addTab(self.tts_tab,      "⚙️ TTS")

        self.tabs.addTab(self.images_tab,   "🖼 Imagens")

        self.tabs.addTab(self.video_tab,    "🎬 Vídeo")

        self.tabs.addTab(self.settings_tab, "🔧 Configurações")

        self.tabs.addTab(self.queue_tab,    "⛓️ Fila")

        self.tabs.setStyleSheet("QTabWidget { background: transparent; }")
        layout.addWidget(self.tabs)


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
        
        # Conexões para troca de modelos em background
        if hasattr(self.audio_tab, "engine_switch_requested"):
            self.audio_tab.engine_switch_requested.connect(self._handle_audio_tab_engine_switch)


        # Conexões Dashboard ← geração manual da aba Áudio

        self.audio_tab.pipeline_progress.connect(self.dashboard_tab.on_manual_progress)

        self.audio_tab.pipeline_log.connect(self.dashboard_tab.on_manual_log)

        self.audio_tab.pipeline_finished.connect(self.dashboard_tab.on_manual_finished)


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
                if "tts_tab" in self._session:
                    self.tts_tab.load_session(self._session["tts_tab"])
                if "video_tab" in self._session:
                    self.video_tab.load_session(self._session["video_tab"])
                if "settings_tab" in self._session:
                    self.settings_tab.load_session(self._session["settings_tab"])

    def get_session(self) -> dict:
        return {
            "theme": self.theme_combo.currentText(),
            "bg_overlay_alpha": self.overlay_slider.value(),
            "audio_tab": self.audio_tab.get_session(),
            "tts_tab": self.tts_tab.get_session(),
            "video_tab": self.video_tab.get_session(),
            "settings_tab": self.settings_tab.get_session(),
        }

    def _reset_to_defaults(self):
        reply = QMessageBox.question(self, "Resetar", "Deseja restaurar as configurações padrão em todas as abas?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.audio_tab.reset_defaults()
            self.tts_tab.reset_defaults()
            self.video_tab.reset_defaults()
            self.settings_tab.reset_defaults()
            # self.queue_tab.reset_defaults() # Not implemented but could be added
            self.status_bar.showMessage("Configurações resetadas.")
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

