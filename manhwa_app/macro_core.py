# manhwa_app/macro_core.py
#
# ============================================================
# MACRO ENGINE — Coordenador Central de Jobs
# ============================================================
#
# MELHORIAS vs versão anterior:
#
#   AUTO-DETECT DE IDIOMA NO MACRO:
#     Quando job.lang está vazio ou é "auto", o MacroCoordinator agora
#     chama detect_language_from_file(job.txt_path) para detectar o idioma
#     automaticamente antes de carregar o preset.
#     Isso garante que PT-BR, ES e EN usem os parâmetros corretos de FX
#     e síntese sem intervenção manual.
#
#   PARÂMETROS CHATTERBOX POR IDIOMA:
#     _run_audio_stage() agora incorpora get_recommended_chatterbox_params()
#     do audio_fx.py para aplicar exaggeration/cfg_weight/temperature corretos
#     por idioma automaticamente.
#
#   AUTO-SELEÇÃO DE MODELO:
#     Se job.lang detectado for "pt" ou "es" e model_type for "turbo",
#     o MacroCoordinator sugere automaticamente "multilingual" (com log).
#     (O Turbo é EN-only; PT/ES precisam do Multilingual.)

import os
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any
from PySide6.QtCore import QObject, Signal, Slot

from manhwa_app.utils import _append_log, natural_sort_key, get_safe_path

logger = logging.getLogger(__name__)


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
    top_p: float = 0.85

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


class MacroCoordinator(QObject):
    """
    Coordenador central do Macro Engine.
    Executa tarefas em sequência, gerencia trocas de GPU e monitora pipelines.
    """
    job_started  = Signal(str, int, int) # job_id, job_idx, total_jobs
    job_progress = Signal(str, int, int) # job_id, current, total
    job_log      = Signal(str, str)      # job_id, message
    job_finished = Signal(str, bool, str) # job_id, success, final_message
    job_complete = Signal(str, float, str)
    job_failed   = Signal(str, str)

    queue_log = Signal(str)
    queue_complete = Signal(float)       # elapsed_s
    all_done  = Signal()

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

    def add_job(self, job: MacroJob):
        self.jobs.append(job)

    def clear_jobs(self):
        if self._is_running:
            return False
        self.jobs.clear()
        return True

    def remove_job(self, jid: str) -> bool:
        """Remove um job da fila. Se for o atual, tenta cancelar pipelines."""
        for i, job in enumerate(self.jobs):
            if job.id == jid:
                if i == self._current_idx and self._is_running:
                    if self._audio_pipeline:
                        self._audio_pipeline.cancel()
                    if self._video_pipeline:
                        self._video_pipeline.cancel()
                    self.queue_log.emit(f"⚠️ Job atual ({job.project_name}) removido e cancelado.")
                self.jobs.pop(i)
                if i < self._current_idx:
                    self._current_idx -= 1
                return True
        return False

    def start(self):
        if not self.jobs or self._is_running:
            return
        self._is_running = True
        self._current_idx = -1
        self._queue_start_time = time.time()
        self.queue_log.emit("🚀 Iniciando Macro Engine Geral...")
        self._run_next()

    def stop(self):
        self._is_running = False
        if self._audio_pipeline:
            self._audio_pipeline.cancel()
        if self._video_pipeline:
            self._video_pipeline.cancel()
        self.queue_log.emit("🛑 Macro parado pelo usuário.")

    def _run_next(self):
        if not self._is_running:
            return

        self._current_idx += 1
        if self._current_idx >= len(self.jobs):
            self._is_running = False
            total_elapsed = time.time() - getattr(self, "_queue_start_time", time.time())
            self.queue_log.emit("🏁 Todas as tarefas do Macro concluídas!")
            self.queue_complete.emit(total_elapsed)
            self.all_done.emit()
            return

        # Limpeza de VRAM
        try:
            import torch
            import gc
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
            self.queue_log.emit("🧹 VRAM limpa para o próximo Job.")
        except Exception:
            pass

        job = self.jobs[self._current_idx]
        if job.status in ("done", "error"):
            self._run_next()
            return

        job.status = "running"
        self.job_started.emit(job.id, self._current_idx, len(self.jobs))
        self.queue_log.emit(
            f"➡ Processando [{self._current_idx+1}/{len(self.jobs)}] "
            f"{job.project_name} ({job.workflow})"
        )

        # ── AUTO-DETECT DE IDIOMA ─────────────────────────────────────────
        # Se lang estiver vazio ou for 'auto', detecta do arquivo .txt
        if not job.lang or job.lang.lower() == "auto":
            detected = self._auto_detect_lang(job)
            job.lang = detected
            self.queue_log.emit(f"🔍 [AUTO-LANG] Idioma detectado: '{detected}' para {job.project_name}")

        # ── AUTO-CORREÇÃO DE MODELO ───────────────────────────────────────
        # Turbo é EN-only: se PT ou ES foram detectados, muda para multilingual
        lang_prefix = job.lang.lower().split("-")[0]
        if lang_prefix in ("pt", "es", "fr", "de", "ja", "zh", "ko") and job.model_type == "turbo":
            self.queue_log.emit(
                f"⚙️ [AUTO-MODEL] Idioma '{job.lang}' requer Multilingual. "
                f"Ajustando model_type: turbo → multilingual."
            )
            job.model_type = "multilingual"

        self._check_model_switch(job)

    def _auto_detect_lang(self, job: MacroJob) -> str:
        """
        Detecta idioma do job a partir do arquivo .txt.
        Fallback: 'en' se não conseguir.
        """
        try:
            from manhwa_app.text_processor import detect_language_from_file
            if job.txt_path and os.path.exists(job.txt_path):
                return detect_language_from_file(job.txt_path)
        except Exception as e:
            logger.warning(f"[AUTO-LANG] Falha na detecção de idioma: {e}")
        return "en"

    def _check_model_switch(self, job: MacroJob):
        if job.workflow == "video_edit":
            self._start_job_workflow(job)
            return

        current_engine = getattr(self.main_window, "_current_engine", "")
        current_model  = getattr(self.main_window, "_current_model_type", "")

        if job.engine != current_engine or job.model_type != current_model:
            self.job_log.emit(job.id, f"⚙ Trocando engine para {job.engine} ({job.model_type})...")
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

    def _start_job_workflow(self, job: MacroJob):
        if not self._is_running:
            return

        if job.workflow in ("audio", "audio_video"):
            self._run_audio_stage(job)
        elif job.workflow == "video_edit":
            self._run_video_stage(job)
        else:
            self._on_job_failed(job, f"Workflow desconhecido: {job.workflow}")

    # ── PRESET MAP ────────────────────────────────────────────────────────

    # Mapeamento: prefixo de idioma → nome do preset (sem .json)
    _LANG_PRESET_MAP = {
        "pt":    "🇧🇷 Manhwa PT-BR - Chatterbox Narração",
        "pt-br": "🇧🇷 Manhwa PT-BR - Chatterbox Narração",
        "es":    "🇪🇸 Manhwa ES - Chatterbox Narração",
        "en":    "🇺🇸 Manhwa EN - Chatterbox Narração",
        "en-us": "🇺🇸 Manhwa EN - Chatterbox Narração",
        "en-gb": "🇺🇸 Manhwa EN - Chatterbox Narração",
    }

    def _load_lang_preset(self, lang: str) -> dict:
        """
        Carrega preset JSON correspondente ao idioma do job.
        Retorna dict com parâmetros do preset, ou {} se não encontrado.
        Campos ignorados: chaves que começam com '_' (comentários internos).
        """
        lang_key = lang.lower().strip()
        preset_name = self._LANG_PRESET_MAP.get(lang_key)

        # Fallback: tenta prefixo (ex: "pt-br" → "pt")
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
            self.queue_log.emit(f"✅ [AUTO-PRESET] Preset '{preset_name}' carregado para '{lang}'.")
            return params
        except Exception as e:
            self.queue_log.emit(f"⚠️ [AUTO-PRESET] Falha ao ler preset '{preset_name}': {e}")
            return {}

    # ── AUDIO STAGE ───────────────────────────────────────────────────────

    def _run_audio_stage(self, job: MacroJob):
        job.status = "stage_audio"
        
        # [SMART SKIP] Verifica se todos os áudios já existem para este projeto
        # Precisamos ler o arquivo para saber o total de parágrafos
        try:
            from manhwa_app.audio_pipeline import split_into_paragraphs
            if os.path.exists(job.txt_path):
                txt_content = Path(job.txt_path).read_text(encoding="utf-8")
                pars = split_into_paragraphs(txt_content)
                total_needed = len(pars)
                out_dir = Path(job.output_root) / job.project_name / "audios"

                # [P5 FIX] Valida tamanho do arquivo (>1KB) além da existência.
                # Arquivo corrompido/vazio (zero-byte após crash de VRAM) seria
                # detectado como "existente" e causaria pulo indevido.
                existing_count = 0
                if out_dir.exists():
                    for i in range(1, total_needed + 1):
                        p = out_dir / f"audio_{i}.wav"
                        if p.exists() and p.stat().st_size > 1024:  # >1KB = arquivo válido
                            existing_count += 1

                if existing_count >= total_needed and total_needed > 0:
                    self.job_log.emit(job.id, f"⏭️ [SKIP] Todos os {total_needed} áudios finais válidos existem. Pulando TTS...")
                    if job.workflow == "audio_video":
                        self._run_video_stage(job)
                    else:
                        self._on_job_success(job, "Áudio já existente (pulado).")
                    return
        except Exception as e:
            logger.debug(f"Falha no Smart Skip Audio: {e}")

        self.job_log.emit(job.id, "🎤 Iniciando geração de áudio (TTS)...")

        try:
            # ── Parâmetros recomendados por idioma (do audio_fx.py) ────────
            # Incorpora exaggeration/cfg_weight/temperature otimizados por idioma
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

            # ── Preset do idioma ───────────────────────────────────────────
            lang_preset = self._load_lang_preset(job.lang)

            # Configuração base do job
            config = {
                "project_name": job.project_name,
                "output_root":  job.output_root,
                "tts_engine":   job.engine,
                "model_type":   job.model_type,
                "txt_path":     job.txt_path,
                "voice":        job.voice,
                "lang":         job.lang,
                "temperature":  job.temperature,
                "speed":        job.speed,
                "top_p":        job.top_p,
                # Defaults de segurança (sobrescritos por preset e audio_params)
                "exaggeration":           0.65,
                "cfg_weight":             0.35,
                "seed":                   3000,
                "min_p":                  0.05,
                "top_k":                  1000,
                "repetition_penalty":     1.15,
                "norm_loudness":          True,
                "similarity_threshold":   0.75,
                "whisper_model":          "base",
            }

            # Prioridade 1: parâmetros Chatterbox por idioma (exaggeration, cfg_weight, etc.)
            # Aplica PRIMEIRO para que preset e audio_params possam sobrescrever se necessário
            if lang_chatterbox_params:
                for k, v in lang_chatterbox_params.items():
                    if k not in ("project_name", "output_root", "txt_path", "voice", "lang"):
                        config[k] = v

            # Prioridade 2: preset do idioma (FX, temperature, etc.)
            if lang_preset:
                _skip = {"project_name", "output_root", "txt_path", "voice", "lang"}
                for k, v in lang_preset.items():
                    if k not in _skip:
                        config[k] = v

            # Prioridade 3: audio_params do job (escolha manual do usuário — sobrescreve tudo)
            if job.audio_params:
                config.update(job.audio_params)

            # Delega criação do pipeline para a AudioTab
            at = self.main_window.audio_tab
            self._audio_pipeline = at.create_macro_pipeline(config)

            self._audio_pipeline.log_message.connect(
                lambda m: self.job_log.emit(job.id, f"[TTS] {m}")
            )
            self._audio_pipeline.progress.connect(
                lambda c, t: self._on_pipeline_progress(job, c, t)
            )
            self._audio_pipeline.finished.connect(self._on_audio_stage_done)
            self._audio_pipeline.engine_switch_needed.connect(
                self._on_pipeline_engine_switch_needed
            )

            from PySide6.QtCore import QThread
            self._audio_thread = QThread()
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
        """
        Disparado quando o AudioPipeline precisa trocar de modelo durante a geração.
        No modo Macro, o switch já foi feito de forma síncrona — aqui apenas
        liberamos o _switch_event e atualizamos a UI.
        """
        self.queue_log.emit(f"⚙️ [MACRO] Pipeline solicitou engine switch: {engine_str}/{model_type}")
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

        if hasattr(self, "_audio_thread"):
            self._audio_thread.quit()
            self._audio_thread.wait()

        if success:
            self.job_log.emit(job.id, "✓ Áudio finalizado.")
            if job.workflow == "audio_video":
                self._run_video_stage(job)
            else:
                self._on_job_success(job, "Áudio gerado com sucesso.")
        else:
            self._on_job_failed(job, f"Erro no TTS: {message}")

    # ── VIDEO STAGE ───────────────────────────────────────────────────────

    def _run_video_stage(self, job: MacroJob):
        job.status = "stage_video"
        self.job_log.emit(job.id, "🎬 Iniciando composição de vídeo...")

        try:
            out_root = job.output_root
            proj_dir = Path(out_root) / job.project_name

            if not proj_dir.exists():
                self._on_job_failed(job, f"Diretório do projeto não encontrado: {proj_dir}")
                return

            # Áudios
            if job.audio_dir and os.path.exists(job.audio_dir):
                audios_dir = Path(job.audio_dir)
            else:
                audios_dir = proj_dir / "audios"

            # Imagens
            if job.img_dir and os.path.exists(job.img_dir):
                img_dir = Path(job.img_dir)
            else:
                img_dir = proj_dir / "images"

            image_paths = []
            if img_dir.exists():
                image_paths = sorted(
                    [str(p) for p in img_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")],
                    key=natural_sort_key
                )

            if not image_paths:
                image_paths = self.main_window.images_tab.get_images()

            if not image_paths:
                self._on_job_failed(job, "Nenhuma imagem encontrada para o vídeo.")
                return

            if not audios_dir.exists():
                self._on_job_failed(job, f"Pasta de áudios não encontrada em: {audios_dir}")
                return

            audio_paths = sorted(
                [str(p) for p in audios_dir.glob("*.wav")],
                key=natural_sort_key
            )

            if not audio_paths:
                self._on_job_failed(job, "Nenhum áudio encontrado para compor o vídeo.")
                return

            pairs = list(zip(audio_paths, image_paths[:len(audio_paths)]))
            output_mp4 = str(proj_dir / f"{job.project_name}_final.mp4")

            # [SMART SKIP] Verifica se o vídeo final já existe
            if os.path.exists(output_mp4):
                self.job_log.emit(job.id, f"⏭️ [SKIP] Vídeo final já existe em: {Path(output_mp4).name}")
                self._on_job_success(job, "Vídeo já existente (pulado).")
                return

            from manhwa_app.video_pipeline import VideoPipeline
            self._video_pipeline = VideoPipeline(
                pairs=pairs,
                output_path=output_mp4,
                effect_mode="auto",
                transition_mode="fade",
                transition_time=job.video_params.get("transition_time", 0.2),
                bg_music_path=job.video_params.get("bg_music_path", ""),
                bg_music_volume=job.video_params.get("bg_music_volume", 10),
                config=job.video_params,
                parent=self
            )

            self._video_pipeline.log_message.connect(
                lambda m: self.job_log.emit(job.id, f"[VIDEO] {m}")
            )
            self._video_pipeline.progress.connect(
                lambda c, t: self._on_pipeline_progress(job, c, t)
            )
            self._video_pipeline.finished.connect(self._on_video_stage_done)

            from PySide6.QtCore import QThread
            self._video_thread = QThread()
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

        if hasattr(self, "_video_thread"):
            self._video_thread.quit()
            self._video_thread.wait()

        if success:
            self.job_log.emit(job.id, "✓ Vídeo finalizado com sucesso.")
            self._on_job_success(job, "Geração completa finalizada.")
        else:
            self._on_job_failed(job, f"Erro no Vídeo: {message}")

    # ── AUX ───────────────────────────────────────────────────────────────

    def _on_job_success(self, job, msg):
        job.status = "done"
        job.progress = 100
        job.message = msg
        self.job_finished.emit(job.id, True, msg)
        self.queue_log.emit(f"✅ Tarefa concluída: {job.project_name}")
        self._run_next()

    def _on_job_failed(self, job, msg):
        job.status = "error"
        job.message = msg
        self.job_finished.emit(job.id, False, msg)
        self.queue_log.emit(f"❌ Falha na tarefa {job.project_name}: {msg}")
        self._run_next()