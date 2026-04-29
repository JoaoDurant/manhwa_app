# manhwa_app/macro_core.py
#
# ============================================================
# MACRO ENGINE — Coordenador Central de Jobs
# ============================================================
#
# PARIDADE TOTAL COM video_pipeline.py e VideoTab (app.py):
#
#   [FIX 1] PAIRING LOGIC corrigida:
#     - "mixed_seq" e "mixed_prob" (nomes corretos do VideoTab)
#     - _can_pair() adicionada para validar proporção antes de split
#     - Modo "split" usa zip simples (igual VideoTab)
#     - Modo "single" usa zip direto sem loop complexo
#
#   [FIX 2] TRANSITION MODE/TIME propagados:
#     - Macro não mais hardcoda "none"/0.0
#     - Lê job.video_params["transition"] e ["transition_time"]
#
#   [FIX 3] EFFECT MODE propagado:
#     - Macro não mais hardcoda "auto"
#     - Lê job.video_params["effect"]
#
#   [FIX 4] CONFIG production/video com fallback seguro:
#     - Se video_params vazio, usa defaults que espelham VideoTab.reset_defaults()
#     - color_grading=True, sharpen=True, better_easing=True por padrão
#
#   [FIX 5] SINAIS video_scene_done e video_complete conectados:
#     - MacroTab recebe progresso de cenas individualmente
#     - Log inclui métricas de velocidade/tempo do vídeo
#
#   [FIX 6] AUDIO extensions completas:
#     - Macro agora aceita .wav .mp3 .ogg .flac .aac .m4a (igual VideoTab)
#
#   [FIX 7] THREAD cleanup robusto:
#     - deleteLater() adicionado no pipeline após quit/wait
#     - _video_pipeline=None após cleanup
#
#   [FIX 8] SMART SKIP movido para ANTES do cálculo de pares.
#
#   AUTO-DETECT DE IDIOMA, AUTO-SELEÇÃO DE MODELO:
#     Quando job.lang for "auto", detecta via fasttext.
#     Turbo é EN-only; PT/ES/FR/etc. mudam para multilingual.

import os
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any, Tuple, Union
from PySide6.QtCore import QObject, Signal, Slot, Qt, QTimer

from manhwa_app.utils import _append_log, natural_sort_key, get_safe_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extensões aceitas (paridade com VideoTab._AUDIO_EXTS / _IMAGE_EXTS)
# ---------------------------------------------------------------------------
_AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".flac", ".aac", ".m4a"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Sufixos de arquivos temporários gerados pelo pipeline (não usar como áudio)
_AUDIO_TMP_SUFFIXES = ("_best.wav", "_tmp.wav", "_silence.wav", "_sil.wav")


@dataclass
class MacroJob:
    """Representa uma tarefa atômica no Macro Engine."""
    id: str
    project_name: str
    workflow: str      # 'audio', 'audio_video', 'video_edit'
    txt_path: str      # Caminho do roteiro
    img_dir: str       # Pasta de imagens (opcional)
    engine: str        # 'chatterbox', 'kokoro'
    model_type: str    # 'turbo', 'fast', 'multilingual'
    voice: str         # ID da voz ou Caminho .pt/.wav
    lang: str          # 'pt', 'en', 'es', 'auto' (auto = detecta do arquivo)
    output_root: str   # Pasta base de saída
    audio_dir: str = ""  # Pasta de áudios explícita (Modo Edit)

    # Parâmetros de Áudio
    speed: float = 1.0
    temperature: float = 0.65
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    top_p: float = 1.0
    top_k: int = 1000
    repetition_penalty: float = 1.2

    # Status
    status: str = "pending"  # pending, running, stage_audio, stage_video, done, error
    progress: int = 0
    message: str = ""

    # Parâmetros complexos (captura total da UI)
    audio_params: Dict[str, Any] = field(default_factory=dict)
    video_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        valid_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def mode_label(self):
        map_ = {
            "audio":       "🔊 Voz",
            "audio_video": "🎬 Full",
            "video_edit":  "🛠️ Edit"
        }
        return map_.get(self.workflow, self.workflow)

    def engine_label(self):
        return f"{self.engine} ({self.model_type})"


# ---------------------------------------------------------------------------
# Helpers de paridade com VideoTab
# ---------------------------------------------------------------------------

def _can_pair_images(img_path: str) -> bool:
    """
    Verifica se uma imagem tem proporção adequada para layout split.
    Espelha exatamente VideoTab._can_pair().
    Retorna False para imagens muito verticais (ratio < 0.71) ou muito
    horizontais (ratio > 1.2), pois ficam distorcidas em split 50/50.
    """
    try:
        from PIL import Image as _PILImage
        with _PILImage.open(img_path) as im:
            ratio = im.width / im.height
            if ratio < 0.71:
                return False
            if ratio > 1.2:
                return False
            return True
    except Exception:
        return False


def _build_video_pairs(
    audio_paths: List[str],
    image_paths: List[str],
    layout_mode: str,
    seed: int = 42,
) -> List[Union[Tuple, Tuple[Tuple, Tuple]]]:
    """
    Constrói a lista de pares (audio, imagem) exatamente como VideoTab faz.

    Retorna lista de:
      - (audio_str, image_str)              → cena single
      - ((audio1, audio2), (img1, img2))    → cena split

    Paridade garantida com:
      - VideoTab._build_mixed_pairs()
      - VideoTab._start_generation() para modos single e split
    """
    import random as _random
    _random.seed(seed)

    pairs: List = []

    if layout_mode == "split":
        # Exatamente como VideoTab._start_generation() modo split:
        #   n = min(len(audio)//2, len(images)//2)
        n = min(len(audio_paths) // 2, len(image_paths) // 2)
        for i in range(n):
            pairs.append((
                (audio_paths[2 * i], audio_paths[2 * i + 1]),
                (image_paths[2 * i], image_paths[2 * i + 1])
            ))
        return pairs

    if layout_mode in ("mixed_seq", "mixed_prob"):
        # Espelha VideoTab._build_mixed_pairs(mode)
        a_idx = i_idx = 0
        current_counter = 0
        next_split_target = 2 if layout_mode == "mixed_seq" else _random.randint(3, 5)

        while a_idx < len(audio_paths):
            can_split = (
                (a_idx + 1 < len(audio_paths)) and
                (i_idx + 1 < len(image_paths))
            )
            make_split = False

            if can_split:
                if layout_mode == "mixed_seq":
                    if current_counter >= next_split_target:
                        make_split = True
                else:  # mixed_prob — regra 70/30
                    if _random.random() < 0.30:
                        make_split = True

                # Validação de proporção das imagens (VideoTab._can_pair)
                if make_split:
                    if (not _can_pair_images(image_paths[i_idx]) or
                            not _can_pair_images(image_paths[i_idx + 1])):
                        make_split = False

            if make_split:
                pairs.append((
                    (audio_paths[a_idx], audio_paths[a_idx + 1]),
                    (image_paths[i_idx], image_paths[i_idx + 1])
                ))
                a_idx += 2
                i_idx += 2
                current_counter = 0
                if layout_mode != "mixed_seq":
                    next_split_target = _random.randint(3, 5)
            else:
                img = image_paths[i_idx] if i_idx < len(image_paths) else image_paths[-1]
                pairs.append((audio_paths[a_idx], img))
                a_idx += 1
                i_idx += 1
                current_counter += 1

        return pairs

    # "single" ou qualquer fallback — mesmo que VideoTab modo single:
    #   n = min(len(audio), len(images))
    n = min(len(audio_paths), len(image_paths))
    return [(audio_paths[i], image_paths[i]) for i in range(n)]


def _resolve_video_config(video_params: dict) -> dict:
    """
    Garante que video_params tenha todos os campos que VideoPipeline/VideoTab
    espera, com os mesmos defaults de VideoTab.reset_defaults().

    Isso resolve o problema de jobs criados sem snapshot completo.
    """
    # Defaults que espelham VideoTab.reset_defaults() + get_session()
    defaults = {
        "layout":          "single",
        "effect":          "auto",
        "transition":      "none",
        "transition_time": 0.0,
        "bg_music_path":   "",
        "bg_music_volume": 10,
        "production": {
            "video": {
                "better_easing": True,
                "color_grading": True,
                "sharpen":       True,
                "film_grain":    False,
                "denoise":       False,
                "vibrance":      False,
            },
            "sound_design": {
                "auto_ducking": True,
            }
        }
    }

    # Merge superficial das chaves de topo
    merged = {**defaults, **video_params}

    # Merge profundo de "production" para não perder sub-chaves
    if "production" in video_params:
        vp = video_params["production"]
        # video sub-dict
        merged["production"]["video"] = {
            **defaults["production"]["video"],
            **vp.get("video", {})
        }
        # sound_design sub-dict
        merged["production"]["sound_design"] = {
            **defaults["production"]["sound_design"],
            **vp.get("sound_design", {})
        }

    return merged


# ---------------------------------------------------------------------------
# MacroCoordinator
# ---------------------------------------------------------------------------

class MacroCoordinator(QObject):
    """
    Coordenador central do Macro Engine.
    Executa tarefas em sequência, gerencia trocas de GPU e monitora pipelines.
    """
    job_started   = Signal(str, int, int)   # job_id, job_idx, total_jobs
    job_progress  = Signal(str, int, int)   # job_id, current, total
    job_log       = Signal(str, str)        # job_id, message
    job_finished  = Signal(str, bool, str)  # job_id, success, final_message
    job_complete  = Signal(str, float, str)
    job_failed    = Signal(str, str)

    queue_log      = Signal(str)
    queue_complete = Signal(float)          # elapsed_s
    all_done       = Signal()

    # Sinal para solicitar troca de modelo à MainWindow
    request_model_switch = Signal(str, str)  # engine, model_type

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.jobs: List[MacroJob] = []
        self._current_idx = -1
        self._is_running = False
        self._is_switching_model = False

        self._audio_pipeline = None
        self._video_pipeline = None
        self._audio_thread = None
        self._video_thread = None

    # ── Queue Management ──────────────────────────────────────────────────

    def add_job(self, job: MacroJob):
        self.jobs.append(job)

    def clear_jobs(self):
        if self._is_running:
            return False
        self.jobs.clear()
        self._current_idx = -1
        return True

    def update_job(self, job_id: str, new_data: dict) -> bool:
        for job in self.jobs:
            if job.id == job_id:
                for k, v in new_data.items():
                    if hasattr(job, k):
                        setattr(job, k, v)
                return True
        return False

    def remove_job(self, jid: str) -> bool:
        for i, job in enumerate(self.jobs):
            if job.id == jid:
                if i == self._current_idx and self._is_running:
                    if self._audio_pipeline:
                        self._audio_pipeline.cancel()
                    if self._video_pipeline:
                        self._video_pipeline.cancel()
                    self.queue_log.emit(
                        f"⚠️ Job atual ({job.project_name}) removido e cancelado."
                    )
                self.jobs.pop(i)
                if i <= self._current_idx:
                    self._current_idx -= 1
                return True
        return False

    def start(self, restart=True):
        if not self.jobs or self._is_running:
            return
        self._is_running = True
        if restart:
            self._current_idx = -1
            self._queue_start_time = time.time()
            for job in self.jobs:
                job.status = "pending"
                job.progress = 0
                job.message = ""
            self.queue_log.emit("🚀 Iniciando Macro Engine Geral...")
        else:
            self.queue_log.emit("▶️ Retomando Macro Engine...")
        self._run_next()

    def stop(self):
        self._is_running = False
        if self._audio_pipeline:
            self._audio_pipeline.cancel()
        if self._video_pipeline:
            self._video_pipeline.cancel()
        self.queue_log.emit("🛑 Macro parado pelo usuário.")

    # ── Internal Navigation ───────────────────────────────────────────────

    def _run_next(self):
        if not self._is_running:
            return

        self.clear_vram()

        self._current_idx += 1
        if self._current_idx >= len(self.jobs):
            self._is_running = False
            total_elapsed = time.time() - getattr(self, "_queue_start_time", time.time())
            self.queue_log.emit("🏁 Todas as tarefas do Macro concluídas!")
            self.queue_complete.emit(total_elapsed)
            self.all_done.emit()
            return

        job = self.jobs[self._current_idx]
        if job.status in ("done", "error"):
            self._run_next()
            return

        job.status = "running"
        self.job_started.emit(job.id, self._current_idx, len(self.jobs))
        self.queue_log.emit(
            f"➡ Processando [{self._current_idx + 1}/{len(self.jobs)}] "
            f"{job.project_name} ({job.workflow})"
        )

        # ── AUTO-DETECT DE IDIOMA ─────────────────────────────────────────
        if job.lang and job.lang.lower() == "auto":
            detected = self._auto_detect_lang(job)
            job.lang = detected
            self.queue_log.emit(
                f"🔍 [AUTO-LANG] Idioma definido como: '{detected}' para {job.project_name}"
            )

        # ── AUTO-CORREÇÃO DE MODELO ───────────────────────────────────────
        lang_prefix = job.lang.lower().split("-")[0] if job.lang else "en"
        if lang_prefix in ("pt", "es", "fr", "de", "ja", "zh", "ko") and job.model_type == "turbo":
            self.queue_log.emit(
                f"⚙️ [AUTO-MODEL] Idioma '{job.lang}' requer Multilingual. "
                f"Ajustando model_type: turbo → multilingual."
            )
            job.model_type = "multilingual"

        self._check_model_switch(job)

    # ── Language Detection ────────────────────────────────────────────────

    def _auto_detect_lang(self, job: "MacroJob") -> str:
        try:
            import fasttext
            model_path = "lid.176.bin"
            if os.path.exists(model_path):
                fasttext.FastText.eprint = lambda x: None
                model = fasttext.load_model(model_path)
                if job.txt_path and os.path.exists(job.txt_path):
                    with open(job.txt_path, "r", encoding="utf-8") as f:
                        text = f.read(2000).replace("\n", " ").strip()
                    if len(text) >= 20:
                        pred = model.predict(text)
                        lang = pred[0][0].replace("__label__", "")
                        conf = pred[1][0]
                        self.queue_log.emit(f"🔍 [LANG DETECT] {lang} ({conf:.2f})")
                        if conf < 0.75:
                            self.queue_log.emit("⚠️ Baixa confiança — fallback inteligente")
                            return self._fallback_lang(job, text)
                        return lang
            else:
                self.queue_log.emit(
                    f"⚠️ [LANG DETECT] Modelo '{model_path}' não encontrado. Usando fallback."
                )
        except Exception as e:
            self.queue_log.emit(f"⚠️ Erro detecção: {e}")

        text_for_fallback = ""
        if job.txt_path and os.path.exists(job.txt_path):
            try:
                with open(job.txt_path, "r", encoding="utf-8") as f:
                    text_for_fallback = f.read(2000).replace("\n", " ").strip()
            except Exception:
                pass
        return self._fallback_lang(job, text_for_fallback)

    def _fallback_lang(self, job: "MacroJob", text: str) -> str:
        text = text.lower()
        if any(w in text for w in ["não", "você", "isso", "cara", "então", "agora", "mas"]):
            return "pt"
        if any(w in text for w in ["hola", "esto", "pero", "ahora", "entonces", "bien"]):
            return "es"
        if not text and job.txt_path:
            stem = Path(job.txt_path).stem.lower()
            if "pt" in stem or "br" in stem:
                return "pt"
            if "es" in stem:
                return "es"
        return "en"

    # ── Model Switch ──────────────────────────────────────────────────────

    def _check_model_switch(self, job: "MacroJob"):
        if job.workflow == "video_edit":
            self._start_job_workflow(job)
            return

        current_engine = getattr(self.main_window, "_current_engine", "")
        current_model  = getattr(self.main_window, "_current_model_type", "")

        if job.engine != current_engine or job.model_type != current_model:
            self.job_log.emit(
                job.id, f"⚙ Trocando engine para {job.engine} ({job.model_type})..."
            )
            self._is_switching_model = True
            self.main_window.model_loaded.connect(self._on_model_loaded_callback)
            self.request_model_switch.emit(job.engine, job.model_type)
        else:
            self._start_job_workflow(job)

    @Slot(bool, str)
    def _on_model_loaded_callback(self, success, info):
        try:
            self.main_window.model_loaded.disconnect(self._on_model_loaded_callback)
        except Exception:
            pass
        self._is_switching_model = False
        job = self.jobs[self._current_idx]
        if success:
            self.job_log.emit(job.id, "✓ Engine carregada com sucesso.")
            self._start_job_workflow(job)
        else:
            self.job_log.emit(job.id, f"✗ Erro ao carregar engine: {info}")
            self._on_job_failed(job, f"Falha no carregamento do modelo: {info}")

    def _start_job_workflow(self, job: "MacroJob"):
        if not self._is_running:
            return
        if job.workflow in ("audio", "audio_video"):
            self._run_audio_stage(job)
        elif job.workflow == "video_edit":
            self._run_video_stage(job)
        else:
            self._on_job_failed(job, f"Workflow desconhecido: {job.workflow}")

    # ── PRESET MAP ────────────────────────────────────────────────────────

    _LANG_PRESET_MAP = {
        "pt":    "🇧🇷 Manhwa PT-BR - Chatterbox Narração",
        "pt-br": "🇧🇷 Manhwa PT-BR - Chatterbox Narração",
        "es":    "🇪🇸 Manhwa ES - Chatterbox Narração",
        "en":    "🇺🇸 Manhwa EN - Chatterbox Narração",
        "en-us": "🇺🇸 Manhwa EN - Chatterbox Narração",
        "en-gb": "🇺🇸 Manhwa EN - Chatterbox Narração",
    }

    def _load_lang_preset(self, lang: str) -> dict:
        lang_key = lang.lower().strip()
        preset_name = self._LANG_PRESET_MAP.get(lang_key)
        if not preset_name:
            prefix = lang_key.split("-")[0]
            preset_name = self._LANG_PRESET_MAP.get(prefix)
        if not preset_name:
            self.queue_log.emit(
                f"ℹ️ [AUTO-PRESET] Nenhum preset mapeado para idioma '{lang}'. "
                f"Usando parâmetros do job."
            )
            return {}
        preset_path = Path("presets") / f"{preset_name}.json"
        if not preset_path.exists():
            self.queue_log.emit(
                f"⚠️ [AUTO-PRESET] Preset '{preset_name}.json' não encontrado. "
                f"Usando parâmetros do job."
            )
            return {}
        try:
            raw = json.loads(preset_path.read_text(encoding="utf-8"))
            params = {k: v for k, v in raw.items() if not k.startswith("_")}
            self.queue_log.emit(
                f"✅ [AUTO-PRESET] Preset '{preset_name}' carregado para '{lang}'."
            )
            return params
        except Exception as e:
            self.queue_log.emit(f"⚠️ [AUTO-PRESET] Falha ao ler preset '{preset_name}': {e}")
            return {}

    # ── AUDIO STAGE ───────────────────────────────────────────────────────

    def _run_audio_stage(self, job: "MacroJob"):
        job.status = "stage_audio"

        # Desconecta pipeline anterior completamente
        if self._audio_pipeline is not None:
            try:
                self._audio_pipeline.log_message.disconnect()
                self._audio_pipeline.progress.disconnect()
                self._audio_pipeline.finished.disconnect()
                self._audio_pipeline.engine_switch_needed.disconnect()
                if hasattr(self._audio_pipeline, "paragraph_retry"):
                    self._audio_pipeline.paragraph_retry.disconnect()
                if hasattr(self._audio_pipeline, "paragraph_started"):
                    self._audio_pipeline.paragraph_started.disconnect()
                if hasattr(self._audio_pipeline, "paragraph_done_stats"):
                    self._audio_pipeline.paragraph_done_stats.disconnect()
            except Exception:
                pass
            self._audio_pipeline = None

        # [SMART SKIP] Verifica se todos os áudios já existem
        try:
            from manhwa_app.audio_pipeline import split_into_paragraphs
            if os.path.exists(job.txt_path):
                txt_content = Path(job.txt_path).read_text(encoding="utf-8")
                pars = split_into_paragraphs(txt_content)
                total_needed = len(pars)
                out_dir = Path(job.output_root) / job.project_name / "audios"
                existing_count = 0
                if out_dir.exists():
                    for i in range(1, total_needed + 1):
                        p = out_dir / f"audio_{i}.wav"
                        if p.exists() and p.stat().st_size > 1024:
                            existing_count += 1
                if existing_count >= total_needed and total_needed > 0:
                    self.job_log.emit(
                        job.id,
                        f"⏭️ [SKIP] Todos os {total_needed} áudios finais válidos existem. "
                        f"Pulando TTS..."
                    )
                    if job.workflow == "audio_video":
                        self._run_video_stage(job)
                    else:
                        self._on_job_success(job, "Áudio já existente (pulado).")
                    return
        except Exception as e:
            logger.debug(f"Falha no Smart Skip Audio: {e}")

        self.job_log.emit(job.id, "🎤 Iniciando geração de áudio (TTS)...")

        try:
            # Parâmetros Chatterbox por idioma
            lang_chatterbox_params = {}
            try:
                from manhwa_app.audio_fx import get_recommended_chatterbox_params
                lang_chatterbox_params = get_recommended_chatterbox_params(job.lang)
                self.queue_log.emit(
                    f"🎛️ [AUTO-PARAMS] Parâmetros Chatterbox para '{job.lang}': "
                    f"exaggeration={lang_chatterbox_params.get('exaggeration', 'n/a')}, "
                    f"cfg_weight={lang_chatterbox_params.get('cfg_weight', 'n/a')}"
                )
            except Exception as e:
                logger.warning(f"[AUTO-PARAMS] Falha ao obter params Chatterbox: {e}")

            # Resolve voz
            resolved_voice = job.voice
            if not resolved_voice and job.audio_params:
                resolved_voice = (
                    job.audio_params.get("voice") or
                    job.audio_params.get("preset_voice") or
                    ""
                )
            if resolved_voice:
                self.queue_log.emit(f"🎤 [VOZ] Usando: {resolved_voice}")
                self.job_log.emit(
                    job.id,
                    f"🎤 Voz de referência: "
                    f"'{Path(resolved_voice).name if resolved_voice else 'NENHUMA'}'"
                )
            else:
                self.queue_log.emit("⚠️ [VOZ] Nenhuma voz de referência — modelo usará voz padrão.")

            # Preset do idioma
            lang_preset = self._load_lang_preset(job.lang)

            config = {
                "project_name":        job.project_name,
                "output_root":         job.output_root,
                "tts_engine":          job.engine,
                "model_type":          job.model_type,
                "txt_path":            job.txt_path,
                "voice":               resolved_voice,
                "lang":                job.lang,
                "temperature":         job.temperature,
                "speed":               job.speed,
                "exaggeration":        job.exaggeration,
                "cfg_weight":          job.cfg_weight,
                "top_p":               job.top_p,
                "top_k":               job.top_k,
                "repetition_penalty":  job.repetition_penalty,
                # Defaults de segurança
                "seed":                   3000,
                "min_p":                  0.05,
                "norm_loudness":          True,
                "similarity_threshold":   0.0,
                "whisper_model":          "base",
                "max_retries":            5,
                "fx_natural_mode":        False,
            }

            # Prioridade 1: params Chatterbox por idioma
            if lang_chatterbox_params:
                for k, v in lang_chatterbox_params.items():
                    if k not in ("project_name", "output_root", "txt_path", "voice", "lang"):
                        config[k] = v

            # Prioridade 2: preset do idioma
            if lang_preset:
                _skip = {"project_name", "output_root", "txt_path", "voice", "lang"}
                for k, v in lang_preset.items():
                    if k not in _skip:
                        config[k] = v

            # Prioridade 3: audio_params do job (escolha manual — sobrescreve tudo)
            if job.audio_params:
                _structural = {
                    "audio_params", "video_params", "tts_engine", "model_type",
                    "lang", "voice", "project_name", "txt_path", "output_root"
                }
                safe_audio_params = {
                    k: v for k, v in job.audio_params.items()
                    if k not in _structural
                }
                config.update(safe_audio_params)
                if "fx_natural_mode" in job.audio_params:
                    config["fx_natural_mode"] = job.audio_params["fx_natural_mode"]

            # Integração com FX oficial
            config["production"] = {
                "audio": {
                    "highpass":       config.get("fx_highpass", False),
                    "lowpass":        config.get("fx_noise_reduction", False),
                    "deesser":        config.get("fx_deesser", False),
                    "compressor":     config.get("fx_compressor", False),
                    "reverb":         config.get("fx_reverb", False),
                    "remove_silence": config.get("fx_silence", False),
                    "normalize":      config.get("fx_loudnorm", config.get("fx_normalize", False)),
                    "natural_mode":   config.get("fx_natural_mode", False)
                }
            }

            at = self.main_window.audio_tab
            self._audio_pipeline = at.create_macro_pipeline(config)

            self._audio_pipeline.log_message.connect(
                lambda m: self.job_log.emit(job.id, f"[TTS] {m}"),
                Qt.ConnectionType.UniqueConnection
            )
            self._audio_pipeline.progress.connect(
                lambda c, t: self._on_pipeline_progress(job, c, t),
                Qt.ConnectionType.UniqueConnection
            )
            self._audio_pipeline.finished.connect(
                self._on_audio_stage_done,
                Qt.ConnectionType.UniqueConnection
            )
            self._audio_pipeline.engine_switch_needed.connect(
                self._on_pipeline_engine_switch_needed,
                Qt.ConnectionType.UniqueConnection
            )

            from PySide6.QtCore import QThread
            self._audio_thread = QThread()
            self._audio_thread.setStackSize(16 * 1024 * 1024)
            self._audio_pipeline.moveToThread(self._audio_thread)
            self._audio_thread.started.connect(self._audio_pipeline.run)
            self._audio_thread.start()

        except Exception as e:
            logger.error(f"Erro ao iniciar Audio Stage: {e}", exc_info=True)
            self._on_job_failed(job, f"Erro interno: {e}")

    def _on_pipeline_progress(self, job, current, total):
        self.job_progress.emit(job.id, current, total)

    @Slot(str, str)
    def _on_pipeline_engine_switch_needed(self, engine_str: str, model_type: str):
        self.queue_log.emit(
            f"⚙️ [MACRO] Pipeline solicitou engine switch: {engine_str}/{model_type}"
        )
        if self._audio_pipeline and hasattr(self._audio_pipeline, "confirm_switch_done"):
            self._audio_pipeline.confirm_switch_done()
        if hasattr(self.main_window, "trigger_model_preload"):
            self.main_window.trigger_model_preload(
                explicit_engine=engine_str,
                explicit_model_type=model_type
            )

    @Slot(bool, str)
    def _on_audio_stage_done(self, success, message):
        if not self._is_running:
            return
        job = self.jobs[self._current_idx]

        # Desconecta primeiro — antes de qualquer deleteLater
        if self._audio_pipeline is not None:
            try:
                self._audio_pipeline.log_message.disconnect()
                self._audio_pipeline.progress.disconnect()
                self._audio_pipeline.finished.disconnect()
                self._audio_pipeline.engine_switch_needed.disconnect()
            except Exception:
                pass

        if hasattr(self, "_audio_thread") and self._audio_thread is not None:
            self._audio_thread.quit()
            self._audio_thread.wait(5000)
            self._audio_thread.deleteLater()
            self._audio_thread = None

        if self._audio_pipeline is not None:
            self._audio_pipeline.deleteLater()
            self._audio_pipeline = None

        if success:
            self.job_log.emit(job.id, "✓ Áudio finalizado.")
            if job.workflow == "audio_video":
                # [FIX] Nunca usar time.sleep() no event loop Qt.
                # Limpa VRAM e agenda o estágio de vídeo via QTimer (non-blocking).
                self.clear_vram()
                QTimer.singleShot(600, lambda: self._run_video_stage(job))
            else:
                self._on_job_success(job, "Áudio gerado com sucesso.")
        else:
            self._on_job_failed(job, f"Erro no TTS: {message}")

    # ── VIDEO STAGE ───────────────────────────────────────────────────────

    def _run_video_stage(self, job: "MacroJob"):
        job.status = "stage_video"

        # Desconecta pipeline de vídeo anterior
        if self._video_pipeline is not None:
            try:
                self._video_pipeline.log_message.disconnect()
                self._video_pipeline.progress.disconnect()
                self._video_pipeline.finished.disconnect()
                if hasattr(self._video_pipeline, "video_scene_done"):
                    self._video_pipeline.video_scene_done.disconnect()
                if hasattr(self._video_pipeline, "video_complete"):
                    self._video_pipeline.video_complete.disconnect()
            except Exception:
                pass
            self._video_pipeline.deleteLater()
            self._video_pipeline = None

        self.job_log.emit(job.id, "🎬 Iniciando composição de vídeo...")

        try:
            out_root = job.output_root
            proj_dir = Path(out_root) / job.project_name
            proj_dir.mkdir(parents=True, exist_ok=True)

            # ── [FIX 8] SMART SKIP antes de qualquer cálculo pesado ──────
            output_mp4 = str(proj_dir / f"{job.project_name}_final.mp4")
            if os.path.exists(output_mp4):
                self.job_log.emit(
                    job.id,
                    f"⏭️ [SKIP] Vídeo final já existe: {Path(output_mp4).name}"
                )
                self._on_job_success(job, "Vídeo já existente (pulado).")
                return

            # ── Resolve pasta de áudios ───────────────────────────────────
            if job.audio_dir and os.path.exists(job.audio_dir):
                audios_dir = Path(job.audio_dir)
            else:
                audios_dir = proj_dir / "audios"

            if not audios_dir.exists():
                self._on_job_failed(
                    job, f"Pasta de áudios não encontrada em: {audios_dir}"
                )
                return

            # [FIX 6] Aceita todas as extensões de áudio, igual ao VideoTab
            audio_paths = sorted(
                [
                    str(p) for p in audios_dir.iterdir()
                    if p.is_file()
                    and p.suffix.lower() in _AUDIO_EXTS
                    and not any(p.name.endswith(s) for s in _AUDIO_TMP_SUFFIXES)
                ],
                key=natural_sort_key
            )

            if not audio_paths:
                self._on_job_failed(job, "Nenhum áudio encontrado para compor o vídeo.")
                return

            # ── Resolve pasta de imagens ──────────────────────────────────
            if job.img_dir and os.path.exists(job.img_dir):
                img_dir = Path(job.img_dir)
            else:
                img_dir = proj_dir / "images"

            image_paths = []
            if img_dir.exists():
                image_paths = sorted(
                    [
                        str(p) for p in img_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
                    ],
                    key=natural_sort_key
                )

            # Fallback para ImagesTab
            if not image_paths:
                image_paths = self.main_window.images_tab.get_images()

            if not image_paths:
                self._on_job_failed(job, "Nenhuma imagem encontrada para o vídeo.")
                return

            # ── [FIX 4] Resolve config de vídeo com defaults completos ────
            video_cfg = _resolve_video_config(job.video_params)

            # ── [FIX 1] PAIRING LOGIC — paridade total com VideoTab ───────
            layout_mode = video_cfg.get("layout", "single")
            self.queue_log.emit(
                f"🎬 [LAYOUT] Preparando pares para modo: '{layout_mode}' "
                f"({len(audio_paths)} áudios / {len(image_paths)} imagens)"
            )

            pairs = _build_video_pairs(
                audio_paths=audio_paths,
                image_paths=image_paths,
                layout_mode=layout_mode,
                seed=42,  # Fixo para reprodutibilidade no Macro
            )

            if not pairs:
                self._on_job_failed(job, "Nenhum par áudio/imagem válido gerado.")
                return

            # Conta e loga split vs single para diagnóstico
            n_splits = sum(1 for p in pairs if isinstance(p[0], tuple))
            n_single = len(pairs) - n_splits
            self.queue_log.emit(
                f"🎬 [PARES] {len(pairs)} cenas: {n_single} single + {n_splits} split"
            )

            # ── [FIX 2] Transition mode/time do snapshot da UI ───────────
            transition_mode = video_cfg.get("transition", "none")
            transition_time = float(video_cfg.get("transition_time", 0.0))

            # ── [FIX 3] Effect mode do snapshot da UI ────────────────────
            effect_mode = video_cfg.get("effect", "auto")

            # ── BGM com validação de path aninhado ────────────────────────
            bg_music_path   = video_cfg.get("bg_music_path", "")
            bg_music_volume = int(video_cfg.get("bg_music_volume", 10))

            # Valida path do BGM se fornecido
            if bg_music_path and not os.path.exists(bg_music_path):
                self.job_log.emit(
                    job.id,
                    f"⚠️ [BGM] Arquivo de música não encontrado: {bg_music_path} — ignorando."
                )
                bg_music_path = ""

            self.job_log.emit(
                job.id,
                f"🎬 Configuração: layout={layout_mode} | effect={effect_mode} | "
                f"transition={transition_mode} ({transition_time}s) | "
                f"bgm={'sim' if bg_music_path else 'não'}"
            )

            # ── Cria o VideoPipeline ──────────────────────────────────────
            from manhwa_app.video_pipeline import VideoPipeline
            self._video_pipeline = VideoPipeline(
                pairs=pairs,
                output_path=output_mp4,
                effect_mode=effect_mode,
                layout_mode=layout_mode,
                transition_mode=transition_mode,
                transition_time=transition_time,
                bg_music_path=bg_music_path,
                bg_music_volume=bg_music_volume,
                config=video_cfg,           # [FIX 4] config completo com defaults
                parent=self
            )

            # ── [FIX 5] Conecta TODOS os sinais do VideoPipeline ─────────
            self._video_pipeline.log_message.connect(
                lambda m: self.job_log.emit(job.id, f"[VIDEO] {m}"),
                Qt.ConnectionType.UniqueConnection
            )
            self._video_pipeline.progress.connect(
                lambda c, t: self._on_pipeline_progress(job, c, t),
                Qt.ConnectionType.UniqueConnection
            )
            self._video_pipeline.finished.connect(
                self._on_video_stage_done,
                Qt.ConnectionType.UniqueConnection
            )
            # video_scene_done → atualiza barra de cenas no MacroTab
            self._video_pipeline.video_scene_done.connect(
                lambda c, t: self.job_progress.emit(job.id, c, t),
                Qt.ConnectionType.UniqueConnection
            )
            # video_complete → loga métricas de velocidade/tempo
            self._video_pipeline.video_complete.connect(
                lambda path, total_t, speed: self.job_log.emit(
                    job.id,
                    f"[VIDEO] ✓ Concluído: {Path(path).name} | "
                    f"duração {total_t:.1f}s | velocidade {speed:.2f}x realtime"
                ),
                Qt.ConnectionType.UniqueConnection
            )

            # ── Inicia thread de vídeo ────────────────────────────────────
            from PySide6.QtCore import QThread
            self._video_thread = QThread()
            # Vídeo não precisa de stack extra (sem recursão profunda),
            # mas mantemos consistência com o resto do app
            self._video_pipeline.moveToThread(self._video_thread)
            self._video_thread.started.connect(self._video_pipeline.run)
            self._video_thread.start()

        except Exception as e:
            logger.error(f"Erro ao iniciar Video Stage: {e}", exc_info=True)
            self._on_job_failed(job, f"Erro Vídeo: {e}")

    @Slot(bool, str)
    def _on_video_stage_done(self, success, message):
        if not self._is_running:
            return
        job = self.jobs[self._current_idx]

        # [FIX 7] Cleanup robusto de thread E pipeline
        if hasattr(self, "_video_thread") and self._video_thread is not None:
            self._video_thread.quit()
            self._video_thread.wait(10000)
            self._video_thread.deleteLater()
            self._video_thread = None

        if self._video_pipeline is not None:
            self._video_pipeline.deleteLater()
            self._video_pipeline = None

        # Limpa thread de áudio remanescente se existir
        if hasattr(self, "_audio_thread") and self._audio_thread is not None:
            self._audio_thread.quit()
            self._audio_thread.wait(5000)
            self._audio_thread.deleteLater()
            self._audio_thread = None

        if success:
            self.job_log.emit(job.id, "✓ Vídeo finalizado com sucesso.")
            self._on_job_success(job, "Geração completa finalizada.")
        else:
            self._on_job_failed(job, f"Erro no Vídeo: {message}")

    # ── VRAM ──────────────────────────────────────────────────────────────

    def clear_vram(self):
        """Limpa cache do CUDA, cache de tensores e força coleta de lixo."""
        try:
            from manhwa_app.video_pipeline import clear_pipeline_cache
            clear_pipeline_cache()
            import torch
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            self.queue_log.emit("🧹 VRAM e Cache de Tensores limpos para o próximo estágio.")
        except Exception as e:
            logger.debug(f"Falha ao limpar VRAM: {e}")

    # ── AUX ───────────────────────────────────────────────────────────────

    def _on_job_success(self, job: "MacroJob", msg: str):
        job.status = "done"
        job.progress = 100
        job.message = msg
        self.job_finished.emit(job.id, True, msg)
        self.queue_log.emit(f"✅ Tarefa concluída: {job.project_name}")
        # [FIX] QTimer para não bloquear o event loop do Qt entre jobs
        self.clear_vram()
        QTimer.singleShot(600, self._run_next)

    def _on_job_failed(self, job: "MacroJob", msg: str):
        job.status = "error"
        job.message = msg
        self.job_finished.emit(job.id, False, msg)
        self.queue_log.emit(f"❌ Falha na tarefa {job.project_name}: {msg}")
        self._run_next()