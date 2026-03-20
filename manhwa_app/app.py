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
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPushButton, QProgressBar,
    QScrollArea, QSizePolicy, QStatusBar, QTabWidget, QTextEdit,
    QVBoxLayout, QWidget, QSlider, QComboBox, QDoubleSpinBox, QSpinBox,
    QCheckBox, QInputDialog,
)

from manhwa_app.audio_pipeline import AudioPipeline, split_into_paragraphs
from manhwa_app.video_pipeline import VideoPipeline, EFFECTS

# Reutilizar a referência a engine já importada pelo audio_pipeline no thread principal.
# Isso evita re-importações dentro de QThreads que causavam 'No module named chatterbox'.
from manhwa_app.audio_pipeline import _engine, _ENGINE_AVAILABLE, _config_manager

logger = logging.getLogger(__name__)

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
            if not _ENGINE_AVAILABLE or _engine is None:
                self.finished_loading.emit(False, "Engine não disponível")
                return

            # Atualiza config se necessário
            if self.tts_engine == "chatterbox":
                current = _config_manager.get_string("model.repo_id", "")
                if current != self.model_type:
                    _config_manager.update_and_save({"model": {"repo_id": self.model_type}})
                
                # Se já carregado mas tipo diferente, limpa
                if _engine.MODEL_LOADED and _engine.loaded_model_type != self.model_type:
                    _engine.chatterbox_model = None
                    _engine.MODEL_LOADED = False
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()

                # Carrega
                if not _engine.MODEL_LOADED:
                    if _engine.load_model():
                        self.finished_loading.emit(True, f"Chatterbox ({self.model_type})")
                    else:
                        self.finished_loading.emit(False, "Falha Chatterbox")
                else:
                    self.finished_loading.emit(True, f"Reutilizando {_engine.loaded_model_type}")

            elif self.tts_engine == "kokoro":
                # Kokoro local load check
                kokoro_path = Path(__file__).resolve().parent.parent / "Kokoro-TTS-Local-master"
                if str(kokoro_path) not in sys.path:
                    sys.path.append(str(kokoro_path))
                import models as k_models
                # Apenas valida importação e disponibilidade
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.finished_loading.emit(True, f"Kokoro ({device.upper()})")
        except Exception as e:
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
}

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
        fx_group = QGroupBox("✨ Efeitos e Processamento")
        fxg = QGridLayout(fx_group)
        fxg.setSpacing(10)
        
        self.spacy_chk = QCheckBox("Pré-processar Texto (Adicionar Pausas / Dividir frases com spaCy)")
        self.fx_nr_chk = QCheckBox("Noise Reduction (Leve)")
        self.fx_enhancer_chk = QCheckBox("Enhancer (Isolamento de Voz Krisp-like)")
        self.fx_comp_chk = QCheckBox("Compressor (Nivelar Dinâmica)")
        self.fx_eq_chk = QCheckBox("EQ Leve (Grave + Presença)")
        self.fx_reverb_chk = QCheckBox("Reverb Leve (Estúdio)")
        self.fx_norm_chk = QCheckBox("Normalize (-1 dB)")
        
        fxg.addWidget(self.spacy_chk, 0, 0, 1, 2)
        fxg.addWidget(self.fx_nr_chk, 1, 0)
        fxg.addWidget(self.fx_enhancer_chk, 1, 1)
        fxg.addWidget(self.fx_comp_chk, 2, 0)
        fxg.addWidget(self.fx_eq_chk, 2, 1)
        fxg.addWidget(self.fx_reverb_chk, 3, 0)
        fxg.addWidget(self.fx_norm_chk, 3, 1)
        
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
            if ans == QMessageBox.Yes:
                Path(path).unlink()
                self._load_presets()
                self.preset_config_combo.setCurrentIndex(0)

    def _restore_explicit(self):
        # Valores padrao
        self.model_combo.setCurrentIndex(0)
        self.temp_spin.setValue(0.65)
        self.speed_spin.setValue(1.0)
        self.exag_spin.setValue(0.5)
        self.cfg_spin.setValue(0.5)
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
        self.fx_nr_chk.setChecked(False)
        self.fx_enhancer_chk.setChecked(False)
        self.fx_comp_chk.setChecked(False)
        self.fx_eq_chk.setChecked(False)
        self.fx_reverb_chk.setChecked(False)
        self.fx_norm_chk.setChecked(False)

    def _on_engine_change(self):
        is_kokoro = self.engine_combo.currentData() == "kokoro"
        
        # Oculta opções do Chatterbox
        self.lbl_model_chatterbox.setVisible(not is_kokoro)
        self.model_combo.setVisible(not is_kokoro)
        self.exag_spin.setEnabled(not is_kokoro)
        self.cfg_spin.setEnabled(not is_kokoro)
        self.rep_spin.setEnabled(not is_kokoro)
        self.norm_loudness_chk.setEnabled(not is_kokoro)
        
        # Atualiza a lista de vozes do AudioTab
        main_win = self.window()
        if hasattr(main_win, "audio_tab"):
            main_win.audio_tab._populate_voices()
        
        self._trigger_preload()

    def _trigger_preload(self):
        main_win = self.window()
        if hasattr(main_win, "trigger_model_preload"):
            main_win.trigger_model_preload()

    def get_session(self):
        return {
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
            "fx_noise_reduction": self.fx_nr_chk.isChecked(),
            "fx_compressor": self.fx_comp_chk.isChecked(),
            "fx_eq": self.fx_eq_chk.isChecked(),
            "fx_reverb": self.fx_reverb_chk.isChecked(),
            "fx_enhancer": self.fx_enhancer_chk.isChecked(),
            "fx_normalize": self.fx_norm_chk.isChecked(),
        }

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
        if "fx_noise_reduction" in data: self.fx_nr_chk.setChecked(data["fx_noise_reduction"])
        if "fx_compressor" in data: self.fx_comp_chk.setChecked(data["fx_compressor"])
        if "fx_eq" in data: self.fx_eq_chk.setChecked(data["fx_eq"])
        if "fx_reverb" in data: self.fx_reverb_chk.setChecked(data["fx_reverb"])
        if "fx_enhancer" in data: self.fx_enhancer_chk.setChecked(data["fx_enhancer"])
        if "fx_normalize" in data: self.fx_norm_chk.setChecked(data["fx_normalize"])

    def reset_defaults(self):
        self.engine_combo.setCurrentIndex(0)
        self.temp_spin.setValue(0.65)
        self.speed_spin.setValue(1.0)
        self.exag_spin.setValue(0.5)
        self.cfg_spin.setValue(0.5)
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
        self.fx_nr_chk.setChecked(False)
        self.fx_comp_chk.setChecked(False)
        self.fx_eq_chk.setChecked(False)
        self.fx_reverb_chk.setChecked(False)
        self.fx_enhancer_chk.setChecked(False)
        self.fx_norm_chk.setChecked(False)


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
        vg = QHBoxLayout(vox_group)
        
        vg.addWidget(QLabel("Voz:"))
        self.preset_voice_combo = QComboBox()
        self.preset_voice_combo.addItem("Sem clonagem (Voz do Modelo)", "")
        self._populate_voices()
        vg.addWidget(self.preset_voice_combo, 1)

        btn_custom = QPushButton("📁 Arquivo...")
        btn_custom.setFixedWidth(90)
        btn_custom.clicked.connect(self._browse_custom_voice)
        vg.addWidget(btn_custom)
        
        vg.addWidget(QLabel("Idioma:"))
        self.preset_lang_combo = QComboBox()
        self.preset_lang_combo.addItems(["en", "pt", "es", "fr", "ja", "ko", "zh"])
        vg.addWidget(self.preset_lang_combo)
        
        lv.addWidget(vox_group)
        self._custom_voice_path = None
        
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
        voice_val = self.preset_voice_combo.currentData()
        lang_val = self.preset_lang_combo.currentText()
        
        preview_configs = [{"path": str(tmp_txt_path), "voice": voice_val, "lang": lang_val}]

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
            fx_noise_reduction=cfg.get("fx_noise_reduction", False),
            fx_compressor=cfg.get("fx_compressor", False),
            fx_eq=cfg.get("fx_eq", False),
            fx_reverb=cfg.get("fx_reverb", False),
            fx_enhancer=cfg.get("fx_enhancer", False),
            fx_normalize=cfg.get("fx_normalize", False),
            use_spacy=cfg.get("use_spacy", False),
        )
        self._preview_thread = QThread()
        self._preview_pipeline.moveToThread(self._preview_thread)
        self._preview_thread.started.connect(self._preview_pipeline.run)
        
        self._preview_pipeline.log_message.connect(lambda msg: _append_log(self.log_text, "[PREVIEW] " + msg))
        self._preview_pipeline.finished.connect(self._on_preview_finished)
        self._preview_thread.start()

    def _on_preview_finished(self, success, msg):
        self._preview_thread.quit()
        self._preview_thread.wait()
        
        self.btn_preview.setEnabled(True)
        self.btn_preview.setText("▶ Testar Voz")

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
                    
                for path in sorted(standard):
                    self.preset_voice_combo.addItem(f"🔊 {format_name_chatterbox(path)}", str(path))
                for path in sorted(cloned):
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
                for path in sorted(voices):
                    name = path.stem
                    lang = get_kokoro_lang_str(name)
                    self.preset_voice_combo.addItem(f"✨ {lang} {name}", str(path))
                    
        if old_val:
            idx = self.preset_voice_combo.findData(old_val)
            if idx >= 0:
                self.preset_voice_combo.setCurrentIndex(idx)

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Selecionar Textos", "", "Textos (*.txt)")
        for p in paths:
            self._files.append({"path": p})
        self._refresh_list()

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Selecionar Pasta de Textos")
        if folder:
            paths = sorted([str(p) for p in Path(folder).glob("*.txt")])
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
        self.btn_generate.setEnabled(False)
        self.btn_run_all.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self._generated.clear()
        self.audio_list.clear()
        cfg = self._get_tts_config()
        
        voice_val = self.preset_voice_combo.currentData()
        lang_val = self.preset_lang_combo.currentText()
        new_configs = [{"path": c["path"], "voice": voice_val, "lang": lang_val} for c in configs]
        
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
            fx_noise_reduction=cfg.get("fx_noise_reduction", False),
            fx_compressor=cfg.get("fx_compressor", False),
            fx_eq=cfg.get("fx_eq", False),
            fx_reverb=cfg.get("fx_reverb", False),
            fx_enhancer=cfg.get("fx_enhancer", False),
            fx_normalize=cfg.get("fx_normalize", False),
            use_spacy=cfg.get("use_spacy", False),
        )
        self._worker_thread = QThread(self)
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
        if self._worker_thread and self._worker_thread.isRunning():
            self._worker_thread.quit()
            if not self._worker_thread.wait(5000):  # 5s timeout
                self._worker_thread.terminate()  # force if stuck
                self._worker_thread.wait(2000)
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
        thread = QThread(self)
        pipeline.moveToThread(thread)
        def _on_regen_done(success, msg):
            Path(tmp_txt).unlink(missing_ok=True)
            thread.quit()
            new_wav = (Path(self.output_root_edit.text().strip() or "output") / project / "audios" / "audio_1.wav")
            if success and new_wav.exists():
                import shutil
                shutil.copy2(str(new_wav), str(wav_path))
                _append_log(self.log_text, f"✓ Áudio #{idx} regravado.")
            else:
                _append_log(self.log_text, f"✗ Falha ao regravar #{idx}.")
        thread.started.connect(pipeline.run)
        pipeline.log_message.connect(self._on_log)
        pipeline.finished.connect(_on_regen_done)
        _append_log(self.log_text, f"↻ Regravando parágrafo #{idx}…")
        thread.start()

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

    def get_session(self) -> dict:
        return {
            "project": self.project_edit.text().strip(),
            "output_root": self.output_root_edit.text().strip()
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
            self._add_paths(paths)
            event.acceptProposedAction()

    def _add_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Selecionar imagens", "",
            "Imagens (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        self._add_paths(paths)

    def _load_from_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Selecionar pasta de imagens")
        if not folder:
            return
        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        paths = sorted(
            [str(p) for p in Path(folder).iterdir()
             if p.is_file() and p.suffix.lower() in exts],
            key=lambda x: x.lower()
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

        self.btn_generate = QPushButton("🎬 Gerar Vídeo")
        self.btn_generate.setObjectName("primary")
        self.btn_generate.setMinimumHeight(38)
        self.btn_generate.clicked.connect(self._start_generation)
        lv.addWidget(self.btn_generate)

        self.btn_cancel = QPushButton("✖ Cancelar")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel)
        lv.addWidget(self.btn_cancel)
        lv.addStretch()
        root.addWidget(left)

        # Painel direito: progresso + log
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
        self.log_text.setStyleSheet("background:#171717;")
        logv.addWidget(self.log_text)
        rv.addWidget(log_group)

        root.addWidget(right)

    def update_audio_paths(self, paths: List[str]):
        import re
        def _num(p):
            m = re.search(r'audio_(\d+)', Path(p).name)
            return int(m.group(1)) if m else 0
        self._audio_paths = sorted(paths, key=_num)
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
            key=lambda x: x.lower()
        )
        image_files = sorted(
            [str(f) for f in p.rglob("*") if f.is_file() and f.suffix.lower() in self._IMAGE_EXTS],
            key=lambda x: x.lower()
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
            key=lambda x: x.lower()
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
            key=lambda x: x.lower()
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

    def _start_generation(self):
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
        effect_mode = self.effect_combo.currentData() or "auto"
        transition_mode = self.transition_combo.currentData() or "fade"
        transition_time = self.transition_time_spin.value()

        self.btn_generate.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.log_text.clear()
        self.progress_bar.setValue(0)

        bgm_path = self.bgm_path_edit.text().strip()
        bgm_vol = self.bgm_vol_spin.value()

        self._pipeline = VideoPipeline(pairs=pairs, output_path=output_path,
                                       effect_mode=effect_mode,
                                       transition_mode=transition_mode,
                                       transition_time=transition_time,
                                       bg_music_path=bgm_path,
                                       bg_music_volume=bgm_vol)
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

        self._pipeline = VideoPipeline(pairs=pairs, output_path=str(out_mp4),
                                       effect_mode=effect_mode,
                                       bg_music_path=bgm_path,
                                       bg_music_volume=bgm_vol)
        self._worker_thread = QThread(self)
        self._pipeline.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._pipeline.run)
        self._pipeline.log_message.connect(self._on_log)
        self._pipeline.progress.connect(self._on_progress)
        self._pipeline.finished.connect(self._on_done_run_all)
        self._worker_thread.start()

    @Slot(bool, str)
    def _on_done_run_all(self, success: bool, message: str):
        self._worker_thread.quit()
        self._worker_thread.wait()
        self._worker_thread = None
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
        self._worker_thread.quit()
        self._worker_thread.wait()
        self.btn_generate.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        if success:
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("Concluído!")
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
            "bg_music_path": getattr(self, "bgm_path_edit", QLineEdit()).text().strip(),
            "bg_music_volume": getattr(self, "bgm_vol_spin", QSpinBox()).value() if hasattr(self, "bgm_vol_spin") else 10
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


# ---------------------------------------------------------------------------
# Janela Principal
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
        self._check_model_in_background()
        self._restore_session()
        
        self._loader_thread = None
        # Inicia primeiro preload
        self.trigger_model_preload()

    def trigger_model_preload(self):
        """Dispara o carregamento do modelo em background."""
        if not hasattr(self, "tts_tab"): return
        
        cfg = self.tts_tab.get_session()
        engine_name = cfg.get("tts_engine", "chatterbox")
        model_type = cfg.get("model_type", "turbo")

        if self._loader_thread and self._loader_thread.isRunning():
            return # Já carregando

        self.model_status_label.setText("TTS: carregando…")
        self.model_status_label.setStyleSheet("color:#f0b040;font-size:11px;")
        
        self._loader_thread = ModelLoaderThread(engine_name, model_type)
        self._loader_thread.finished_loading.connect(self._on_model_preloaded)
        self._loader_thread.start()

    def _on_model_preloaded(self, success, info):
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
        self.audio_tab   = AudioTab()
        self.tts_tab     = TtsConfigTab()
        self.images_tab  = ImagesTab()
        self.video_tab   = VideoTab()
        self.tabs.addTab(self.audio_tab,  "📝 Áudio")
        self.tabs.addTab(self.tts_tab,    "⚙️ TTS")
        self.tabs.addTab(self.images_tab, "🖼 Imagens")
        self.tabs.addTab(self.video_tab,  "🎬 Vídeo")
        # Ensure QTabWidget is transparent at its base, so bg image falls through
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
                    engine = _engine
                    if engine is None or not _ENGINE_AVAILABLE:
                        self_t.result.emit(False, "cpu", "engine não disponível")
                        return
                    if not engine.MODEL_LOADED:
                        ok = engine.load_model()
                    else:
                        ok = True
                    device = engine.model_device or "cpu"
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

    def get_session(self) -> dict:
        return {
            "theme": self.theme_combo.currentText(),
            "bg_overlay_alpha": self.overlay_slider.value(),
            "audio_tab": self.audio_tab.get_session(),
            "tts_tab": self.tts_tab.get_session(),
            "video_tab": self.video_tab.get_session(),
        }

    def _reset_to_defaults(self):
        reply = QMessageBox.question(self, "Resetar", "Deseja restaurar as configurações padrão em todas as abas?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.audio_tab.reset_defaults()
            self.tts_tab.reset_defaults()
            self.video_tab.reset_defaults()
            self.theme_combo.setCurrentText("\U0001f311 Dark (Padr\u00e3o)")
            self._clear_background()
            QMessageBox.information(self, "Resetado", "Configurações restauradas com sucesso.")

    def closeEvent(self, event):
        _save_session(self.get_session())
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

