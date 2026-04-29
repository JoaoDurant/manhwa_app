# manhwa_app/audio_pipeline.py
"""
Pipeline de síntese de áudio para o Manhwa Video Creator.

Arquitetura GPU-first para RTX 5070 Ti (Blackwell sm_120):
  - Síntese TTS (Chatterbox Turbo) → GPU, sequencial
  - Transcrição Whisper            → GPU, inline no loop de síntese (sem thread separada)
  - Pós-processamento (silêncio)   → CPU, thread pool paralela

Fluxo por parágrafo:
  texto → [pré-proc CPU] → [síntese GPU] → audio_N_tmp.wav
       → [Whisper GPU inline] → retry se necessário
       → [pós-proc CPU async] → audio_N.wav
"""
import gc
import re
import os
import sys
import json
import time
import shutil
import tempfile
import threading
import logging
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
from PySide6.QtCore import QObject, Signal
import numpy as np
import torch
from manhwa_app.audio_fx import apply_audio_post_processing

# --- REPO SETUP ---
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import engine as _engine
    import utils as _utils
    from config import config_manager as _config_manager
    from manhwa_app.advanced_text_processor import process_text
    from manhwa_app.models import get_whisper_model, unload_whisper, transcribe_audio
    from manhwa_app.utils import get_safe_path
    _ENGINE_AVAILABLE = True
except ImportError as _ie:
    _engine = _utils = _config_manager = None
    _ENGINE_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CUDA backend flags — nao toca em set_num_threads (ja definido em engine.py)
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    except Exception:
        pass


# ===========================================================================
# HELPERS DE TEXTO
# ===========================================================================

def split_into_paragraphs(text: str) -> List[str]:
    text = text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    raw = re.split(r"\n\n+", text) if "\n\n" in text else re.split(r"\n+", text)
    result = []
    for block in raw:
        block = re.sub(r"^\s*\d+\s*\n+", "", block)
        clean = re.sub(r"\s+", " ", block.replace("\n", " ")).strip()
        if len(clean) >= 3:
            result.append(clean)
    return result


def _normalize_text_for_tts(text: str, lang: str = "en", engine: str = "chatterbox") -> str:
    text = text.replace("\ufeff", "")
    if engine == "kokoro":
        return re.sub(r" {2,}", " ", text).strip()
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r" +([.!?,;:])", r"\1", text)
    return text.strip()


def _text_similarity(expected: str, transcribed: str) -> float:
    from difflib import SequenceMatcher
    if not transcribed:
        return 0.0
    return SequenceMatcher(None, expected.lower().strip(), transcribed.lower().strip()).ratio()


def _check_absurd(expected: str, transcribed: str, sim: float) -> Tuple[bool, str]:
    if not transcribed or len(transcribed) < 3:
        return True, "Mudo/Vazio"
    if sim < 0.65:
        return True, f"Erro de conteudo ({sim:.2f})"
    ew, tw = len(expected.split()), len(transcribed.split())
    if tw > (ew * 1.6) + 3:
        return True, "Alucinacao/Loop"
    if tw < (ew * 0.5) and ew > 4:
        return True, "Corte massivo"
    return False, ""


# ===========================================================================
# HELPERS DE AUDIO
# ===========================================================================

def _evaluate_quality(path: str) -> float:
    try:
        import soundfile as sf
        arr, sr = sf.read(path, dtype="float32")
        if arr.ndim > 1:
            arr = arr[:, 0]
        if len(arr) == 0:
            return 0.0
        frame_len = int(sr * 0.05)
        if len(arr) < frame_len:
            return 0.0
        frames = np.array_split(arr, max(1, len(arr) // frame_len))
        rms = np.array([np.sqrt(np.mean(f ** 2)) for f in frames])
        rms = rms[rms > 0.001]
        if len(rms) < 2:
            return 0.0
        return float(np.std(rms) / (np.mean(rms) + 1e-6))
    except Exception:
        return 0.0


def _remove_silence(inp: str, out: str, sr: int) -> str:
    """Remove silencio de inicio/fim. Retorna inp se falhar."""
    try:
        import soundfile as sf
        import utils as u
        arr, actual_sr = sf.read(inp, dtype="float32")
        if arr.ndim > 1:
            arr = arr[:, 0]
            
        use_sr = actual_sr if actual_sr else sr
        if actual_sr and actual_sr != sr:
            logger.warning(f"[SILENCE] Usando SR real do arquivo ({actual_sr}) ao invés do solicitado ({sr}) para preservar o pitch.")
            
        arr = u.trim_lead_trail_silence(arr, use_sr, -35.0, 400)
        arr = u.fix_internal_silence(arr, use_sr, -40.0, 250, 280)
        if u.save_audio_to_file(arr, use_sr, out):
            return out
    except Exception as e:
        logger.warning(f"[SILENCE] {inp}: {e} — mantendo original.")
    return inp


def _add_padding(audio_path: str, sr: int, pad_secs: float = 0.2) -> bool:
    """Adiciona um pequeno silêncio no final do áudio para evitar corte de palavras."""
    try:
        import soundfile as sf
        arr, sr_file = sf.read(audio_path, dtype="float32")
        if sr_file != sr:
            logger.warning(f"[PAD] Sample rate mismatch {sr_file}!={sr}")
        pad_len = int(sr_file * pad_secs)
        if pad_len <= 0:
            return True
        pad = np.zeros(pad_len, dtype=arr.dtype)
        if arr.ndim > 1:
            pad = np.zeros((pad_len, arr.shape[1]), dtype=arr.dtype)
        arr_padded = np.concatenate([arr, pad])
        sf.write(audio_path, arr_padded, sr_file, subtype="PCM_16")
        return True
    except Exception as e:
        logger.error(f"[PAD] Falha ao adicionar padding: {e}")
        return False


def _save_tensor_to_wav(wav_tensor, sr: int, out_path: str) -> bool:
    """
    Persiste tensor em disco como WAV PCM-16 via escrita atomica.

    Usa tempfile.mkstemp(suffix=".wav") no mesmo diretorio de out_path para que:
      1. soundfile detecte o formato WAV pelo sufixo (evita audio_N_tmp.wav.wav.tmp)
      2. os.replace() seja atomico (mesmo filesystem, Windows e Linux)

    Retorna True se out_path existe e nao esta vazio.
    """
    import soundfile as sf

    tmp_fd = None
    tmp_path = None
    try:
        if isinstance(wav_tensor, torch.Tensor):
            arr = wav_tensor.detach().cpu().numpy()
        else:
            arr = np.asarray(wav_tensor, dtype=np.float32)

        if arr.ndim == 2:
            arr = arr[0] if arr.shape[0] == 1 else arr[:, 0]
        arr = arr.astype(np.float32)

        peak = float(np.abs(arr).max())
        if peak == 0.0:
            logger.warning(f"[SAVE] Tensor silencioso: {out_path}")
            return False
        if peak > 1.0:
            arr = arr / peak

        parent_dir = os.path.dirname(os.path.abspath(out_path)) or "."
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav", dir=parent_dir)
        os.close(tmp_fd)
        tmp_fd = None

        sf.write(tmp_path, arr, sr, subtype="PCM_16")
        os.replace(tmp_path, out_path)
        tmp_path = None  # rename OK — nao deletar no finally

        p = Path(out_path)
        ok = p.exists() and p.stat().st_size > 0
        if not ok:
            logger.error(f"[SAVE] Arquivo criado mas invalido: {out_path}")
        return ok

    except ImportError:
        logger.error("[SAVE] soundfile nao disponivel: pip install soundfile")
        return False
    except Exception as e:
        logger.error(f"[SAVE] Falha em {out_path}: {e}")
        return False
    finally:
        if tmp_fd is not None:
            try:
                os.close(tmp_fd)
            except Exception:
                pass
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _prepare_voice_reference(voice_path: str, target_sr: int = 24000) -> str:
    """
    Garante que o arquivo de referência está em WAV mono 24000 Hz.
    Retorna o caminho do arquivo normalizado (temporário se convertido).
    """
    if not voice_path or not os.path.exists(voice_path):
        return voice_path

    try:
        # Usa ffprobe para checar o SR real sem carregar o arquivo inteiro
        import subprocess, json
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", voice_path],
            capture_output=True, text=True, timeout=10
        )
        info = json.loads(probe.stdout)
        streams = info.get("streams", [])
        audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
        
        if not audio_streams:
            return voice_path
            
        src_sr = int(audio_streams[0].get("sample_rate", target_sr))
        src_ext = Path(voice_path).suffix.lower()

        # Se já está em WAV 24000 Hz, usa direto
        if src_sr == target_sr and src_ext == ".wav":
            return voice_path

        # Precisa converter
        logger.info(f"[VOICE REF] Convertendo referência: {src_sr}Hz {src_ext} → {target_sr}Hz WAV")
        
        tmp = tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False,
            dir=tempfile.gettempdir()
        )
        tmp.close()

        result = subprocess.run([
            "ffmpeg", "-y", "-i", voice_path,
            "-ar", str(target_sr),
            "-ac", "1",          # mono
            "-sample_fmt", "s16",
            tmp.name
        ], capture_output=True, timeout=30)

        if result.returncode == 0 and os.path.exists(tmp.name):
            logger.info(f"[VOICE REF] Conversão OK → {tmp.name}")
            return tmp.name
        else:
            logger.warning(f"[VOICE REF] Falha na conversão, usando original. stderr: {result.stderr.decode()[:200]}")
            return voice_path

    except Exception as e:
        logger.warning(f"[VOICE REF] Erro ao preparar referência: {e} — usando original")
        return voice_path


def _synthesize_and_save(
    engine_name: str,
    text: str,
    out_path: str,
    voice: Optional[str],
    temperature: float,
    seed: int,
    sample_rate: int,
    language: str = "en",
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    min_p: float = 0.05,
    top_p: float = 0.85,
    top_k: int = 1000,
    repetition_penalty: float = 1.15,
    norm_loudness: bool = True,
) -> bool:
    """
    Sintetiza e salva audio em out_path.
    Trata contratos A (tensor,sr), B (ndarray,sr) e C (bool legado).
    """
    if not _ENGINE_AVAILABLE:
        logger.error("[SYNTH] Engine nao disponivel.")
        return False

    active = _engine.get_active_engine()

    # ---- KOKORO ----
    if active == "kokoro" or engine_name == "kokoro":
        try:
            res = _engine.synthesize_kokoro(
                text=text,
                voice=voice if isinstance(voice, str) else "af_heart",
                speed=1.0,
            )
        except Exception as e:
            logger.error(f"[SYNTH/KOKORO] {e}")
            return False
        if isinstance(res, tuple) and len(res) == 2:
            arr, sr = res
            if arr is not None:
                return _save_tensor_to_wav(arr, sr or sample_rate, out_path)
        logger.error(f"[SYNTH/KOKORO] Retorno inesperado: {type(res)!r}")
        return False

    # ---- CHATTERBOX ----
    _tmp_voice_to_cleanup = None
    if voice and os.path.exists(str(voice)):
        normalized_voice = _prepare_voice_reference(voice, target_sr=24000)
        if normalized_voice != voice:
            _tmp_voice_to_cleanup = normalized_voice
        voice = normalized_voice

    try:
        res = _engine.synthesize(
            text=text,
            audio_prompt_path=voice,
            temperature=temperature,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            seed=seed,
            language=language,
            min_p=min_p,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            norm_loudness=norm_loudness,
        )
    except Exception as e:
        logger.error(f"[SYNTH] engine.synthesize excecao: {e}")
        return False
    finally:
        # Limpa arquivo temporário de conversão
        if _tmp_voice_to_cleanup and os.path.exists(_tmp_voice_to_cleanup):
            try:
                os.remove(_tmp_voice_to_cleanup)
            except Exception:
                pass

    if isinstance(res, tuple) and len(res) == 2:
        wav_tensor, sr = res
        if wav_tensor is None:
            logger.error(f"[SYNTH] Tensor None: {out_path}")
            return False
        saved = _save_tensor_to_wav(wav_tensor, sr or sample_rate, out_path)
        if not saved:
            logger.error(f"[SYNTH] _save_tensor_to_wav falhou: {out_path}")
        return saved

    if isinstance(res, bool):
        p = Path(out_path)
        exists = p.exists() and p.stat().st_size > 0
        if res and not exists:
            logger.error(f"[SYNTH] Engine True mas arquivo ausente: {out_path}")
        return res and exists

    logger.error(f"[SYNTH] Retorno desconhecido: {type(res)!r}")
    return False


# ===========================================================================
# PIPELINE PRINCIPAL
# ===========================================================================

class AudioPipeline(QObject):
    progress             = Signal(int, int)
    log_message          = Signal(str)
    paragraph_done       = Signal(int, str, str)
    finished             = Signal(bool, str)
    paragraph_ready      = Signal(int, str, dict)
    engine_switch_needed = Signal(str, str)
    paragraph_started    = Signal(int, int, str)
    paragraph_done_stats = Signal(int, int, float, float, float, int)
    paragraph_retry      = Signal(int, int, str)

    def __init__(
        self,
        file_configs,
        project_name: str,
        output_root: str = "output",
        tts_engine: str = "chatterbox",
        model_type: str = "turbo",
        whisper_model: str = "base",
        similarity_threshold: float = 0.75,
        max_retries: int = 3,
        temperature: float = 0.65,
        **kwargs,
    ):
        super().__init__(kwargs.get("parent"))
        self.file_configs         = file_configs
        self.project_name         = project_name
        self.output_root          = output_root
        self.tts_engine           = tts_engine
        self.model_type           = model_type
        self.whisper_model        = whisper_model
        self.similarity_threshold = similarity_threshold
        self.max_retries          = max_retries
        self.temperature          = temperature
        self.exaggeration         = kwargs.get("exaggeration", 0.5)
        self.cfg_weight           = kwargs.get("cfg_weight", 0.5)
        self.speed                = kwargs.get("speed", 1.0)
        self.min_p                = kwargs.get("min_p", 0.05)
        self.top_p                = kwargs.get("top_p", 1.0)
        self.top_k                = kwargs.get("top_k", 1000)
        self.repetition_penalty   = kwargs.get("repetition_penalty", 1.2)
        self.norm_loudness        = kwargs.get("norm_loudness", True)
        self.lang                 = kwargs.get("lang", "pt")
        self.sample_rate          = kwargs.get("sample_rate", 24000)
        self._config              = kwargs  # Captura as configs de FX passadas por kwargs
        self._cancelled           = False
        self._state_lock          = threading.Lock()

        # Pos-proc (I/O + CPU) roda em paralelo — nao usa GPU, nao disputa VRAM
        self._post_exec = ThreadPoolExecutor(max_workers=2, thread_name_prefix="postproc")

        self.paragrafos_map:    Dict[int, dict] = {}
        self.completed_indices: Set[int]        = set()
        self._start_time = 0.0
        self._all_paras: List[tuple]            = []
        self._out_dir:   Optional[Path]         = None

    def cancel(self):
        self._cancelled = True

    # -----------------------------------------------------------------------
    def _preprocess(self, para: str, lang: str) -> Tuple[str, str]:
        try:
            opts = {
                "normalize_text": True, "clean_symbols": True,
                "improve_punctuation": True, "add_natural_pauses": True,
                "convert_numbers": True, "use_phonetic": False, "use_spacy": True,
            }
            verified = process_text(para, opts, lang=lang)
            tts_text = _normalize_text_for_tts(verified, lang=lang, engine=self.tts_engine)
            return tts_text, verified
        except Exception:
            return para, para

    # -----------------------------------------------------------------------
    def _post_process(
        self,
        idx: int,
        tmp_path: Path,
        para: str,
        src: str,
        sr: int,
        sim: float,
        elapsed: float,
        attempts: int,
        rms: float,
        lang: str = "pt",
    ):
        """
        tmp_path (audio_N_tmp.wav)
          -> _remove_silence -> sil_path (audio_N_sil.wav)
          -> _add_padding -> (200ms extra para evitar corte)
          -> audio_fx -> final_out (audio_N.wav)
          -> remove tmps
          -> emite sinais Qt
        """
        try:
            out_dir = self._out_dir

            if not tmp_path.exists():
                logger.error(f"[POST #{idx}] Ausente: {tmp_path}")
                self.log_message.emit(f"  ✗ [POST #{idx}] Arquivo temporario nao encontrado.")
                return

            sil_path = out_dir / f"audio_{idx}_sil.wav"
            after_silence = _remove_silence(str(tmp_path), str(sil_path), sr)

            # Adiciona 250ms de silence no fim para proteger o final da frase e dar respiro sem ficar exagerado
            _add_padding(after_silence, sr, pad_secs=0.25)

            final_out = out_dir / f"audio_{idx}.wav"
            
            # Aplica FX (Highpass, Comp, Equalizador, etc.)
            config_dict = getattr(self, '_config', {})
            fx_ok = apply_audio_post_processing(
                input_wav=after_silence,
                output_wav=str(final_out),
                config=config_dict,
                lang=lang,
            )
            
            if not fx_ok:
                logger.warning(f"[POST #{idx}] FX falhou, copiando sem efeitos.")
                self.log_message.emit(f"  ⚠️ [POST #{idx}] Falha no FX. Arquivo salvo sem efeitos FFmpeg.")
                shutil.copy2(after_silence, str(final_out))
            else:
                active_fx = [k for k, v in config_dict.items() if k.startswith('fx_') and v]
                fx_names = ", ".join([k.replace("fx_", "") for k in active_fx]) if active_fx else "Padrão Spectral"
                self.log_message.emit(f"  🎵 FX aplicado [{fx_names}] no áudio #{idx}")

            if not final_out.exists() or final_out.stat().st_size == 0:
                logger.error(f"[POST #{idx}] Final nao criado: {final_out}")
                self.log_message.emit(f"  ✗ [POST #{idx}] Falha ao criar audio_{idx}.wav")
                return

            for tmp in [tmp_path, Path(sil_path)]:
                if tmp.exists() and tmp.resolve() != final_out.resolve():
                    try:
                        tmp.unlink()
                    except Exception:
                        pass

            entry = {
                "index": idx, "audio": f"audio_{idx}.wav",
                "texto": para, "arquivo_origem": src,
                "similaridade": round(sim, 3),
            }
            with self._state_lock:
                self.paragrafos_map[idx] = entry
                self.completed_indices.add(idx)
                done = len(self.completed_indices)

            total = len(self._all_paras)
            avg   = (time.time() - self._start_time) / done if done > 0 else 0
            eta   = int(avg * (total - done))
            eta_s = f"{eta // 60}m {eta % 60}s" if eta >= 60 else f"{eta}s"

            self.paragraph_done.emit(idx, str(final_out), para)
            self.paragraph_ready.emit(idx, str(final_out), entry)
            self.paragraph_done_stats.emit(idx, total, elapsed, sim, rms, attempts)
            self.progress.emit(done, total)
            self.log_message.emit(
                f"  ✓ [#{idx}/{total}] Concluido | Sim: {sim:.2f} | ETA: {eta_s}"
            )

        except Exception as e:
            logger.exception(f"[POST #{idx}] Excecao")
            self.log_message.emit(f"  ✗ [POST #{idx}] Erro: {e}")

    # -----------------------------------------------------------------------
    def _synthesize_with_retry(
        self,
        idx: int,
        tts_text: str,
        verified_text: str,
        tmp_path: Path,
        p_cfg: dict,
        lang: str,
    ) -> Tuple[bool, float, int]:
        """
        Sintetiza com retry. Whisper roda inline na GPU (mesma thread).
        Retorna (sucesso, similaridade, tentativas).
        """
        voice       = p_cfg.get("voice")
        engine_name = p_cfg.get("engine", self.tts_engine)

        for attempt in range(1, self.max_retries + 1):
            if self._cancelled:
                return False, 0.0, attempt

            temp = min(self.temperature + (attempt - 1) * 0.05, 0.95)
            seed = idx * 100 + attempt

            # [PARITY FIX] Garante que parâmetros de prosódia sejam passados para a síntese.
            # Se não estiverem no p_cfg (nível parágrafo), usa o global (self).
            ok = _synthesize_and_save(
                engine_name=engine_name,
                text=tts_text,
                out_path=str(tmp_path),
                voice=voice,
                temperature=temp,
                seed=seed,
                sample_rate=self.sample_rate,
                language=lang,
                exaggeration=p_cfg.get("exaggeration", self.exaggeration),
                cfg_weight=p_cfg.get("cfg_weight", self.cfg_weight),
                min_p=p_cfg.get("min_p", self.min_p),
                top_p=p_cfg.get("top_p", self.top_p),
                top_k=p_cfg.get("top_k", self.top_k),
                repetition_penalty=p_cfg.get("repetition_penalty", self.repetition_penalty),
                norm_loudness=p_cfg.get("norm_loudness", self.norm_loudness),
            )

            if not ok or not tmp_path.exists():
                logger.warning(f"[SYNTH #{idx}] Tentativa {attempt} falhou (sem arquivo).")
                if attempt == self.max_retries:
                    return False, 0.0, attempt
                continue

            # Sem validacao Whisper — aceita imediatamente
            if self.similarity_threshold <= 0:
                return True, 1.0, attempt

            # Validacao Whisper (GPU inline)
            try:
                trans = transcribe_audio(
                    str(tmp_path), self.whisper_model,
                    language=lang.split("-")[0],
                )
                sim = _text_similarity(verified_text, trans)
                is_bad, reason = _check_absurd(verified_text, trans, sim)

                if not is_bad and sim >= self.similarity_threshold:
                    return True, sim, attempt

                if not reason:
                    reason = f"Sim={sim:.2f} vs req={self.similarity_threshold:.2f}"

                logger.warning(f"[SYNTH #{idx}] Tentativa {attempt}: Sim={sim:.2f} ({reason})")

                if attempt < self.max_retries:
                    self.paragraph_retry.emit(idx, attempt, reason)
                    self.log_message.emit(
                        f"  ↩️ [#{idx}] Sim={sim:.2f} ({reason}) — retry {attempt + 1}/{self.max_retries}"
                    )
                else:
                    # Ultima tentativa: aceita com o que tiver
                    return True, sim, attempt

            except Exception as e:
                logger.warning(f"[SYNTH #{idx}] Whisper falhou na tentativa {attempt}: {e}")
                return True, 0.0, attempt  # aceita audio mesmo sem validacao

        return False, 0.0, self.max_retries

    # -----------------------------------------------------------------------
    def run(self):
        self._start_time = time.time()

        if not _ENGINE_AVAILABLE:
            self.finished.emit(False, "Engine nao disponivel.")
            return

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            free_gb = torch.cuda.mem_get_info()[0] / 1e9
            self.log_message.emit(
                f"🖥️  GPU: {torch.cuda.get_device_name(0)} | VRAM livre: {free_gb:.1f} GB"
            )

        self._all_paras = []
        for fcfg in self.file_configs:
            try:
                text = Path(fcfg["path"]).read_text(encoding="utf-8-sig", errors="replace")
                for p in split_into_paragraphs(text):
                    self._all_paras.append((p, Path(fcfg["path"]).name, fcfg))
            except Exception as e:
                logger.warning(f"[RUN] Erro ao ler {fcfg.get('path')}: {e}")

        if not self._all_paras:
            self.finished.emit(False, "Nenhum paragrafo encontrado.")
            return

        self._out_dir = get_safe_path(Path(self.output_root) / self.project_name / "audios")
        self._out_dir.mkdir(parents=True, exist_ok=True)

        # Pre-carrega Whisper na GPU antes de comecar
        if self.similarity_threshold > 0:
            self.log_message.emit(f"🔌 Carregando Whisper [{self.whisper_model}] GPU/fp16...")
            try:
                get_whisper_model(self.whisper_model, device_override="cuda", compute_type="float16")
            except Exception as e:
                logger.warning(f"[RUN] Falha ao pre-carregar Whisper: {e}")

        if self.tts_engine == "chatterbox":
            _engine.switch_to_engine(self.model_type)

        total = len(self._all_paras)
        self.log_message.emit(f"🚀 Iniciando sintese de {total} paragrafos...")

        post_futures: List[concurrent.futures.Future] = []

        for i, (para, src, p_cfg) in enumerate(self._all_paras, 1):
            if self._cancelled:
                self.log_message.emit("⚠️ Cancelamento solicitado.")
                break

            lang = p_cfg.get("lang", self.lang)
            self.paragraph_started.emit(i, total, f"{para[:60]}...")

            t0 = time.time()
            tts_text, verified_text = self._preprocess(para, lang)
            tmp_path = self._out_dir / f"audio_{i}_tmp.wav"

            # Sintese + Whisper inline na GPU (sequencial — sem disputa de VRAM)
            ok, sim, attempts = self._synthesize_with_retry(
                idx=i,
                tts_text=tts_text,
                verified_text=verified_text,
                tmp_path=tmp_path,
                p_cfg=p_cfg,
                lang=lang,
            )

            if ok and tmp_path.exists():
                elapsed = time.time() - t0
                rms = _evaluate_quality(str(tmp_path))
                fut = self._post_exec.submit(
                    self._post_process,
                    i, tmp_path, para, src,
                    self.sample_rate, sim, elapsed, attempts, rms, lang
                )
                post_futures.append(fut)
            else:
                self.log_message.emit(
                    f"  ✗ [#{i}/{total}] Sintese falhou apos {attempts} tentativa(s)."
                )
                logger.error(f"[RUN #{i}] Falha definitiva: {para[:80]!r}")

            # Limpa fragmentacao de VRAM a cada 10 paragrafos
            if i % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Aguarda pos-processamentos
        self.log_message.emit("⌛ Finalizando pos-processamento...")
        for fut in concurrent.futures.as_completed(post_futures):
            try:
                fut.result()
            except Exception as e:
                logger.exception(f"[RUN] Excecao em future de pos-proc: {e}")

        self._post_exec.shutdown(wait=False)
        unload_whisper()

        # Gera paragrafos.json apenas com arquivos existentes
        valid_entries = []
        for k in sorted(self.paragrafos_map):
            entry = self.paragrafos_map[k]
            audio_path = self._out_dir / entry["audio"]
            if audio_path.exists() and audio_path.stat().st_size > 0:
                valid_entries.append(entry)
            else:
                logger.warning(f"[JSON] #{k} excluido: '{entry['audio']}' ausente.")

        meta_path = self._out_dir.parent / "paragrafos.json"
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(valid_entries, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[JSON] Falha ao salvar paragrafos.json: {e}")

        done_count = len(self.completed_indices)
        skipped    = total - done_count
        summary    = f"✓ Concluido! {done_count}/{total} audios gerados."
        if skipped > 0:
            summary += f" ⚠️ {skipped} falharam — verifique os logs."

        self.log_message.emit(summary)
        self.finished.emit(done_count > 0, "Sucesso" if done_count > 0 else "Nenhum audio gerado.")