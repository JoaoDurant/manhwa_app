# manhwa_app/audio_pipeline.py
# Pipeline de geração de áudio para o Manhwa Video Creator.
#
# Integra com o código Chatterbox existente:
#   - engine.load_model() / engine.synthesize()  → TTS
#   - utils.save_audio_tensor_to_file()           → salvar tensor
#   - utils.trim_lead_trail_silence()             → remover silêncio inicial/final
#   - utils.fix_internal_silence()               → reduzir pausas internas longas
#   - utils.save_audio_to_file()                 → salvar array numpy
#
# Recursos extras:
#   - Verificação via Whisper com re-tentativas
#   - Separação por parágrafos de arquivos .txt
#   - Salvamento de paragrafos.json
#   - Otimizações CUDA (empty_cache, verificação de VRAM)

import gc
import json
import logging
import re
import sys
import time
import contextlib
from difflib import SequenceMatcher
import os
import threading
import concurrent.futures
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import shutil
from PySide6.QtCore import QObject, Signal

# --- OTIMIZAÇÃO CUDA RTX (5070 Ti) ---
if torch.cuda.is_available():
    # torch.backends.cudnn.benchmark = True  # DESATIVADO: causa lentidão em TTS com tamanhos de frases variados
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

# Garantir que o diretório raiz do repositório (engine.py, config.py, utils.py)
# esteja no sys.path mesmo quando rodando dentro de uma QThread.
# O run_manhwa_app.py já faz o chdir + sys.path, mas repetimos aqui por segurança
# para o caso do módulo ser importado diretamente (ex: testes unitários).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Importar engine e utils no nível do módulo — no thread principal, onde sys.path já
# está configurado. Isso garante que as QThreads filhas re-usem sys.modules sem precisar
# re-executar as importações (evita 'No module named chatterbox' dentro de threads).
try:
    import engine as _engine
    import utils as _utils
    # Importar config do diretório raiz
    from config import config_manager as _config_manager
    from manhwa_app.text_processor import process_text_fluency, init_spacy
    from manhwa_app.audio_fx import apply_audio_post_processing
    _ENGINE_AVAILABLE = True
except ImportError as _import_err:
    _engine = None  # type: ignore
    _utils = None   # type: ignore
    _config_manager = None # type: ignore
    _ENGINE_AVAILABLE = False
    logging.getLogger(__name__).error(
        f"Falha ao importar engine/utils/config: {_import_err}. "
        "Certifique-se de rodar pelo diretório do Chatterbox TTS Server."
    )

logger = logging.getLogger(__name__)
_CPU_COUNT = os.cpu_count() or 4

# ---------------------------------------------------------------------------
# Singleton do Whisper — carregamento lazy, CUDA + float16 + torch.compile opcional
# ---------------------------------------------------------------------------
_whisper_model = None
_whisper_model_name_loaded: Optional[str] = None
_whisper_device: Optional[str] = None
_WHISPER_LOCK = threading.Lock()


def _get_whisper_model(model_name: str = "base"):
    """
    Carrega o modelo faster-whisper de forma lazy.
    faster-whisper é 3-5x mais rápido que openai-whisper com compute_type=int8_float16.
    """
    global _whisper_model, _whisper_device, _whisper_model_name_loaded
    with _WHISPER_LOCK:
        if _whisper_model is not None and _whisper_model_name_loaded == model_name:
            return _whisper_model, _whisper_device

    # Descarregar modelo anterior se trocar de tamanho
    if _whisper_model is not None:
        del _whisper_model
        _whisper_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Carregando faster-whisper '{model_name}' em {device}…")

    try:
        from faster_whisper import WhisperModel
        compute_type = "int8_float16" if device == "cuda" else "int8"
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        _whisper_model = model
        _whisper_device = device
        _whisper_model_name_loaded = model_name
        logger.info(f"faster-whisper '{model_name}' pronto em {device} (compute: {compute_type}).")
        return _whisper_model, _whisper_device
    except ImportError:
        # Fallback para openai-whisper se faster-whisper não estiver instalado
        logger.warning("faster-whisper não instalado, usando openai-whisper (mais lento). Execute: pip install faster-whisper")
        try:
            import whisper
            model = whisper.load_model(model_name, device=device)
            _whisper_model = model
            _whisper_device = device
            _whisper_model_name_loaded = model_name
            return _whisper_model, _whisper_device
        except Exception as e:
            logger.error(f"Falha ao carregar Whisper: {e}", exc_info=True)
            return None, None
    except Exception as e:
        logger.error(f"Falha ao carregar faster-whisper: {e}", exc_info=True)
        return None, None


def unload_whisper():
    """Descarrega o Whisper e libera memória GPU/CPU."""
    global _whisper_model, _whisper_device, _whisper_model_name_loaded
    if _whisper_model is not None:
        del _whisper_model
        _whisper_model = None
        _whisper_device = None
        _whisper_model_name_loaded = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Modelo Whisper descarregado.")


# ---------------------------------------------------------------------------
# Helpers de texto
# ---------------------------------------------------------------------------

def split_into_paragraphs(text: str) -> List[str]:
    """
    Divide o texto em parágrafos separados por duas ou mais linhas em branco.
    • Normaliza CR/LF → LF.
    • Colapsa newlines simples dentro de um parágrafo em espaços.
    • Ignora parágrafos vazios ou com menos de 3 caracteres.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    raw = re.split(r"\n{2,}", text)
    result = []
    for p in raw:
        clean = " ".join(line.strip() for line in p.split("\n") if line.strip())
        if clean and len(clean) >= 3:
            result.append(clean)
    return result


def _normalize_text_for_tts(text: str, lang: str = "en") -> str:
    """
    Normaliza o texto antes de enviar para TTS.
    Corrige o problema do align_stream_analyzer do Chatterbox multilingual
    que detecta tokens de vírgula repetidos e força EOS prematuramente.
    """
    # 1) Normalizar espaços duplos
    text = re.sub(r' {2,}', ' ', text)

    # 2) Para ling. latinas (es, pt, it, fr): substituir vírgulas por pausa mais longa.
    #    O alinhador trata vírgulas consecutivas como repetição de token e força EOS.
    if lang in ("es", "pt", "it", "fr", "de"):
        # Vírgula seguida de espaço e palavra -> ponto e vírgula (pausa mais longa e token distinto)
        text = re.sub(r',\s+', '.  ', text)
        # Vírgula seguida de aspas ou parênteses
        text = re.sub(r',(["\'«\(\)])', r'.\1', text)

    # 3) Remover pontuação dupla/tripla
    text = re.sub(r'[.!?]{2,}', '...', text)
    # 4) Remover espaço antes de pontuação
    text = re.sub(r' +([.!?,;:])', r'\1', text)

    return text.strip()


def _text_similarity(a: str, b: str) -> float:
    """Similaridade normalizada SequenceMatcher em [0, 1]."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


# ---------------------------------------------------------------------------
# Transcrição Whisper
# ---------------------------------------------------------------------------

def transcribe_audio(wav_path: str, whisper_model_name: str = "base") -> str:
    """
    Transcreve um WAV com faster-whisper (ou openai-whisper como fallback).
    Usa amostragem dos primeiros 10s para validarção rápida — 80% mais rápido em áudios longos.
    """
    model, device = _get_whisper_model(whisper_model_name)
    if model is None:
        return ""
    try:
        # Tentar API do faster-whisper primeiro
        if hasattr(model, 'transcribe') and hasattr(model, 'supported_languages'):
            # faster-whisper: transcrevemos apenas os primeiros 10 segundos
            segments, _ = model.transcribe(
                wav_path,
                beam_size=1,
                language=None,
                clip_timestamps="0,10",  # Amostragem: primeiros 10s
                vad_filter=True,
            )
            return " ".join(s.text for s in segments).strip()
        else:
            # openai-whisper fallback
            result = model.transcribe(wav_path, fp16=(device == "cuda"))
            return result.get("text", "").strip()
    except Exception as e:
        logger.error(f"Transcrição Whisper falhou para {wav_path}: {e}", exc_info=True)
        return ""


# ---------------------------------------------------------------------------
# Remoção de silêncio via utils.py do Chatterbox
# ---------------------------------------------------------------------------

def _remove_silence_from_file(wav_path: str, out_path: str, sample_rate: int) -> str:
    """
    Carrega um WAV, aplica trim_lead_trail_silence + fix_internal_silence do utils.py,
    depois salva com utils.save_audio_to_file.
    Retorna out_path em sucesso, wav_path em falha.
    
    CORREÇÃO: utils.save_audio_to_file(array, sr, path) — sem parâmetro output_format.
    """
    try:
        import soundfile as sf
        import utils as _utils

        audio_arr, sr = sf.read(wav_path, dtype="float32")
        if audio_arr.ndim > 1:
            audio_arr = audio_arr[:, 0]  # usar apenas canal 1

        # Passo 1: remover silêncio inicial e final
        audio_arr = _utils.trim_lead_trail_silence(
            audio_arr,
            sr,
            silence_threshold_db=-40.0,
            padding_ms=200,  # manter 200ms de padding natural
        )

        # Passo 2: encurtar silêncios internos longos
        audio_arr = _utils.fix_internal_silence(
            audio_arr,
            sr,
            silence_threshold_db=-40.0,
            min_silence_to_fix_ms=500,   # silêncios > 500ms serão encurtados
            max_allowed_silence_ms=300,  # manter máximo 300ms de pausa
        )

        # Passo 3: salvar arquivo final
        if _utils.save_audio_to_file(audio_arr, sr, out_path):
            logger.info(f"Silêncio removido → {out_path}")
            return out_path
        else:
            logger.warning(f"save_audio_to_file falhou para {out_path}. Mantendo original.")
            return wav_path

    except Exception as e:
        logger.error(f"Remoção de silêncio falhou para {wav_path}: {e}", exc_info=True)
        return wav_path


# ---------------------------------------------------------------------------
# Verificação de VRAM disponível
# ---------------------------------------------------------------------------

def _get_free_vram_gb() -> float:
    """
    Retorna VRAM livre em GB. Retorna 0.0 se CUDA não disponível.
    """
    if not torch.cuda.is_available():
        return 0.0
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return free_bytes / (1024 ** 3)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Pipeline de áudio principal (QObject com sinais PySide6)
# ---------------------------------------------------------------------------

class AudioPipeline(QObject):
    """
    Roda dentro de uma QThread. Emite sinais PySide6 para atualizar a UI.

    Integração:
        • engine.load_model()  — carrega o TTS Chatterbox (CUDA/CPU via config)
        • engine.synthesize()  — gera tensor WAV por parágrafo
        • utils.save_audio_tensor_to_file() — salva tensor em disco
        • utils.trim_lead_trail_silence()   — remove silêncio inicial/final
        • utils.fix_internal_silence()      — encurta pausas internas longas
        • Whisper (openai-whisper)          — verifica similaridade da transcrição
    """

    progress       = Signal(int, int)            # (índice_atual, total)
    log_message    = Signal(str)                  # linha de status
    paragraph_done = Signal(int, str, str)        # (índice 1-based, wav_path, texto)
    finished       = Signal(bool, str)            # (sucesso, mensagem)

    def __init__(
        self,
        file_configs: List[dict],       # [{"path":str, "voice":str, "lang":str}]
        project_name: str,
        output_root: str = "output",
        tts_engine: str = "chatterbox",
        model_type: str = "turbo",      # "turbo" ou "default"
        whisper_model: str = "base",
        similarity_threshold: float = 0.75,
        max_retries: int = 3,
        temperature: float = 0.65,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        seed: int = 3000,
        speed: float = 1.0,
        output_format: str = "wav",
        min_p: float = 0.05,
        top_p: float = 1.0,
        top_k: int = 1000,
        repetition_penalty: float = 1.2,
        norm_loudness: bool = True,
        ref_vad_trimming: bool = False,
        fx_noise_reduction: bool = False,
        fx_compressor: bool = False,
        fx_eq: bool = False,
        fx_reverb: bool = False,
        fx_enhancer: bool = False,
        fx_normalize: bool = False,
        use_spacy: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._lock = threading.Lock()
        self.file_configs          = file_configs
        self.project_name          = project_name
        self.output_root           = output_root
        self.tts_engine            = tts_engine
        self.model_type            = model_type
        self.whisper_model         = whisper_model
        self.similarity_threshold  = similarity_threshold
        self.max_retries           = max_retries
        self.temperature           = temperature
        self.exaggeration          = exaggeration
        self.cfg_weight            = cfg_weight
        self.seed                  = seed
        self.speed                 = speed
        self.output_format         = output_format
        self.min_p                 = min_p
        self.top_p                 = top_p
        self.top_k                 = top_k
        self.repetition_penalty    = repetition_penalty
        self.norm_loudness         = norm_loudness
        self.ref_vad_trimming      = ref_vad_trimming
        self.fx_noise_reduction    = fx_noise_reduction
        self.fx_compressor         = fx_compressor
        self.fx_eq                 = fx_eq
        self.fx_reverb             = fx_reverb
        self.fx_enhancer           = fx_enhancer
        self.fx_normalize          = fx_normalize
        self.use_spacy             = use_spacy
        self._cancelled            = False
        
        # Otimização CPU: usar todas as threads disponíveis para processamento paralelo (FX, SpaCy)
        if hasattr(torch, 'set_num_threads'):
            torch.set_num_threads(os.cpu_count() or 4)

        # Buffer de logs para evitar overhead de UI (emit agrupado a cada 0.5s)
        self._log_buffer: list = []
        self._log_last_flush: float = 0.0
        # Métrica de performance para detectar degradação progressiva
        self._baseline_avg: float = 0.0

    def cancel(self):
        self._cancelled = True

    def _emit_log(self, msg: str):
        """Emite log imediato (sem buffer) para mensagens críticas."""
        self.log_message.emit(msg)

    def _buffered_log(self, msg: str):
        """Acumula logs e emite em lote a cada 0.5s para reduzir overhead de UI."""
        self._log_buffer.append(msg)
        now = time.time()
        if now - self._log_last_flush >= 0.5:
            combined = "\n".join(self._log_buffer)
            self.log_message.emit(combined)
            self._log_buffer.clear()
            self._log_last_flush = now

    def _flush_log_buffer(self):
        """Força emissão de todos os logs pendentes."""
        if self._log_buffer:
            self.log_message.emit("\n".join(self._log_buffer))
            self._log_buffer.clear()

    def _log_vram(self, stage: str):
        """Monitor de VRAM em tempo real."""
        if torch.cuda.is_available():
            free_gb = _get_free_vram_gb()
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            total_gb = total_bytes / (1024**3)
            used_gb = total_gb - free_gb
            self.log_message.emit(f"📊 [VRAM] {stage} | Uso: {used_gb:.1f}GB / {total_gb:.1f}GB (Livre: {free_gb:.1f}GB)")
        else:
            self.log_message.emit(f"💻 [CPU] {stage} | Threads: {torch.get_num_threads()}")

    # ------------------------------------------------------------------
    # Ponto de entrada principal (chamado por QThread.started ou diretamente)
    # ------------------------------------------------------------------

    def run(self):
        start_time = time.time()
        try:
            self._run_internal(start_time)
        except Exception as e:
            logger.error(f"Falha não capturada no AudioPipeline: {e}", exc_info=True)
            self.log_message.emit(f"❌ Erro interno no pipeline: {e}")
            self.finished.emit(False, f"Erro interno: {e}")

    def _run_internal(self, start_time: float):
        # Verificar se engine/utils/config foram importados com sucesso
        if not _ENGINE_AVAILABLE or _engine is None or _utils is None or _config_manager is None:
            self.log_message.emit("❌ Erro fatal: Dependências (engine, utils, config) não carregáveis.")
            self.finished.emit(
                False,
                "Falha crítica nos módulos internos. Verifique os logs."
            )
            return

        engine = _engine
        utils  = _utils

        # -------- Atualiza configuração de modelo via config_manager (apenas se mudou) --------
        current_model_in_config = _config_manager.get_string("model.repo_id", "")
        if current_model_in_config != self.model_type:
            _config_manager.update_and_save({"model": {"repo_id": self.model_type}})

        # Identifica se precisamos forçar o recarregamento do engine
        if engine.MODEL_LOADED and engine.loaded_model_type != self.model_type:
            self.log_message.emit("Trocando modelo TTS. Limpando VRAM…")
            engine.chatterbox_model = None
            engine.MODEL_LOADED = False
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Garante que GPU finalizou antes de liberar
                torch.cuda.empty_cache()
                gc.collect()
        
        self._log_vram("Início do Pipeline")

        # -------- Lê e separa os parágrafos de todos os .txt --------
        # all_paragraphs contém: (texto, source_file, voice_path, lang)
        all_paragraphs: List[Tuple[str, str, Optional[str], str]] = []
        for fcfg in self.file_configs:
            txt_path = fcfg["path"]
            voice_path = fcfg.get("voice") or None
            lang = fcfg.get("lang", "en")  # fallback
            
            try:
                text = Path(txt_path).read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                self.log_message.emit(f"⚠ Não foi possível ler '{txt_path}': {e}")
                continue

            paragraphs = split_into_paragraphs(text)
            self.log_message.emit(
                f"✓ {len(paragraphs)} parágrafo(s) em '{Path(txt_path).name}' "
                f"(Idioma: {lang}, Voz: {'Padrão' if not voice_path else Path(voice_path).stem})."
            )
            for p in paragraphs:
                all_paragraphs.append((p, Path(txt_path).name, voice_path, lang))

        if not all_paragraphs:
            self.finished.emit(False, "Nenhum parágrafo encontrado nos arquivos (separe com linhas em branco).")
            return

        # -------- Prepara diretório de saída --------
        out_dir = Path(self.output_root) / self.project_name
        audios_dir = out_dir / "audios"
        audios_dir.mkdir(parents=True, exist_ok=True)
        self.log_message.emit(f"Pasta de saída: {out_dir.resolve()}")

        # -------- Garante que o modelo TTS está carregado --------
        if self.tts_engine == "chatterbox":
            if not engine.MODEL_LOADED:
                self.log_message.emit("Carregando modelo Chatterbox TTS… (primeira execução pode demorar)")
                if not engine.load_model():
                    self.finished.emit(False, "Falha ao carregar modelo TTS. Verifique os logs.")
                    return
            device_str = (engine.model_device or "cpu").upper()
            model_type_log = engine.loaded_model_type or "original"
            self.log_message.emit(f"✓ Modelo Chatterbox pronto em {device_str} (tipo: {model_type_log}).")
        elif self.tts_engine == "kokoro":
            self.log_message.emit("Carregando motor Kokoro-TTS (Local)…")
            kokoro_path = str(_REPO_ROOT / "Kokoro-TTS-Local-master")
            if kokoro_path not in sys.path:
                sys.path.append(kokoro_path)
            try:
                import models as kokoro_models
                k_device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.log_message.emit(f"✓ Módulo Kokoro pronto p/ inicialização. (Device: {k_device.upper()})")
            except ImportError as e:
                self.finished.emit(False, f"Falha ao importar dependências do Kokoro: {e}")
                return

        # -------- Processa parágrafos SEQUENCIALMENTE (o engine.synthesize usa Lock global) --------
        # NOTA IMPORTANTE: o engine.synthesize() usa _synthesis_lock (threading.Lock global),
        # o que torna qualquer paralelismo ineficaz. Um loop sequencial é mais limpo,
        # elimina overhead de ThreadPoolExecutor, e evita corridas de VRAM.
        total = len(all_paragraphs)
        completed = 0
        generated_map: Dict[int, str] = {}
        paragrafos_map: Dict[int, dict] = {}

        # Pre-loading de modelos pesados ANTES do loop principal
        if self.use_spacy:
            self.log_message.emit("Pre-carregando SpaCy (NLP)…")
            # Inicializa SpaCy para cada idioma único presente
            langs_in_use = set(p[3] for p in all_paragraphs)
            for lng in langs_in_use:
                init_spacy(lng)

        # Whisper: carregar apenas se threshold > 0 (verificação ativa)
        use_whisper_verification = self.similarity_threshold > 0
        if use_whisper_verification:
            self.log_message.emit(f"Pre-carregando faster-whisper ({self.whisper_model}) para verificação de qualidade…")
            # Carrega Whisper em thread paralela enquanto TTS já está pronto
            import threading as _wt
            _wt.Thread(target=_get_whisper_model, args=(self.whisper_model,), daemon=True).start()

        # Definir seed globalmente UMA VEZ antes do loop (não a cada parágrafo)
        if self.seed != 0 and self.tts_engine == "chatterbox":
            import engine as _eng_ref
            _eng_ref.set_seed(self.seed)
            self.log_message.emit(f"Seed global definida: {self.seed}")

        self.log_message.emit(f"🚀 Iniciando Geração de {total} parágrafo(s) (sequencial, GPU otimizada)...")

        kokoro_models_cache = {}

        for idx, (paragraph, source_file, voice_path, lang) in enumerate(all_paragraphs, start=1):
            if self._cancelled:
                break

            wav_final = audios_dir / f"audio_{idx}.wav"
            wav_tmp   = audios_dir / f"audio_{idx}_tmp.wav"
            
            # Limpeza estratégica no início de cada parágrafo
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            success  = False
            best_sim = 0.0

            for attempt in range(1, self.max_retries + 1):
                if self._cancelled:
                    break

                # ---- Pre-processamento de texto ----
                tts_text_input = paragraph
                if self.use_spacy:
                    tts_text_input = process_text_fluency(tts_text_input, lang=lang)

                # Pular parágrafos muito curtos (<10 chars) que disparam EOS imediato
                if len(tts_text_input.strip()) < 10:
                    self.log_message.emit(f"  ⚠ [#{idx}] Parágrafo muito curto ignorado: '{tts_text_input[:40]}'")
                    continue

                # ---- Síntese TTS ----
                wav_tensor = None
                sample_rate = 24000

                try:
                    if self.tts_engine == "chatterbox":
                        # seed=0 aqui: já definimos globalmente antes do loop
                        wav_tensor, sample_rate = engine.synthesize(
                            text=tts_text_input,
                            audio_prompt_path=voice_path,
                            temperature=self.temperature,
                            exaggeration=self.exaggeration,
                            cfg_weight=self.cfg_weight,
                            seed=0,  # seed já foi setada globalmente
                            language=lang,
                            min_p=self.min_p,
                            top_p=self.top_p,
                            top_k=self.top_k,
                            repetition_penalty=self.repetition_penalty,
                            norm_loudness=self.norm_loudness,
                        )
                    elif self.tts_engine == "kokoro":
                        import models as kokoro_models
                        voice_name = Path(voice_path).stem if voice_path else 'af_heart'
                        k_lang_code = kokoro_models.get_language_code_from_voice(voice_name)
                        k_device = 'cuda' if torch.cuda.is_available() else 'cpu'

                        if k_lang_code not in kokoro_models_cache:
                            kokoro_models_cache[k_lang_code] = kokoro_models.build_model(None, k_device, lang_code=k_lang_code)
                        k_pipeline = kokoro_models_cache[k_lang_code]

                        rtx_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
                        with torch.inference_mode():
                            with torch.autocast(device_type="cuda" if k_device == "cuda" else "cpu", dtype=rtx_dtype) if k_device == "cuda" else contextlib.nullcontext():
                                audio_res, _ = kokoro_models.generate_speech(
                                    model=k_pipeline, text=tts_text_input, voice=voice_name,
                                    lang=k_lang_code, device=k_device, speed=self.speed
                                )
                        if audio_res is not None:
                            wav_tensor = audio_res.cpu().to(torch.float32).unsqueeze(0)

                    if wav_tensor is not None:
                        utils.save_audio_tensor_to_file(wav_tensor, sample_rate, str(wav_tmp), output_format=self.output_format)
                        
                        # Limpeza Imediata do Tensor de Áudio
                        del wav_tensor
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()  # Garante que GPU finalizou
                            torch.cuda.empty_cache()

                        # ---- Verificação via Whisper: APENAS a partir da 2ª tentativa ----
                        # Na 1ª tentativa, confiar na síntese e pular STT (economiza ~15-30s/par.)
                        if use_whisper_verification and attempt > 1:
                            transcription = transcribe_audio(str(wav_tmp), self.whisper_model)
                            if transcription:
                                sim = _text_similarity(paragraph, transcription)
                                best_sim = max(best_sim, sim)
                                if sim < self.similarity_threshold and attempt < self.max_retries:
                                    self.log_message.emit(f"  ↻ [#{idx}] Sim={sim:.2f} < {self.similarity_threshold} → retentando...")
                                    _cleanup(wav_tmp)
                                    continue  # re-tenta
                            
                            # Limpeza Pós Whisper
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()  # Garante que Whisper encerrou
                                torch.cuda.empty_cache()
                                gc.collect()
                                self._log_vram(f"Pós-Whisper #{idx}")

                        # ---- Remoção de silêncio ----
                        silence_out = str(wav_tmp).replace("_tmp", "_silence")
                        final_path_no_fx = _remove_silence_from_file(str(wav_tmp), silence_out, sample_rate)

                        # ---- FX de áudio (opcionais) ----
                        if self.fx_normalize or self.fx_reverb or self.fx_compressor or self.fx_noise_reduction:
                            fx_config = {
                                "fx_noise_reduction": self.fx_noise_reduction,
                                "fx_compressor": self.fx_compressor,
                                "fx_eq": self.fx_eq,
                                "fx_reverb": self.fx_reverb,
                                "fx_enhancer": self.fx_enhancer,
                                "fx_normalize": self.fx_normalize,
                            }
                            apply_audio_post_processing(final_path_no_fx, str(wav_final), fx_config)
                        else:
                            shutil.copy2(final_path_no_fx, str(wav_final))

                        _cleanup(wav_tmp)
                        _cleanup(silence_out)
                        success = True
                        break

                except Exception as e:
                    self.log_message.emit(f"  ✗ [#{idx}] Erro na síntese (tentativa {attempt}): {e}")
                    _cleanup(wav_tmp)

            if success:
                generated_map[idx] = str(wav_final)
                paragrafos_map[idx] = {
                    "index": idx,
                    "audio": f"audio_{idx}.wav",
                    "texto": paragraph,
                    "arquivo_origem": source_file,
                    "similaridade": round(best_sim, 3),
                }
                completed += 1
                elapsed = time.time() - start_time
                avg = elapsed / completed if completed > 0 else 0
                eta = int(avg * (total - completed))
                eta_s = f"{eta//60}m {eta%60}s" if eta >= 60 else f"{eta}s"
                self.progress.emit(completed, total)
                
                # Auto-detecção de degradação de performance
                if self._baseline_avg == 0.0 and completed >= 2:
                    self._baseline_avg = avg
                elif self._baseline_avg > 0 and avg > self._baseline_avg * 1.8:
                    self.log_message.emit(f"⚠️ Performance degradada ({avg:.1f}s/par vs baseline {self._baseline_avg:.1f}s). Defragmentando VRAM…")
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        gc.collect()
                
                self.log_message.emit(f"  ✓ [#{idx}/{total}] Concluído em {elapsed:.1f}s total. ETA: {eta_s}")
                self.paragraph_done.emit(idx, str(wav_final), paragraph)
            else:
                self.log_message.emit(f"  ✗ [#{idx}] Falha após {self.max_retries} tentativa(s).")

        self._flush_log_buffer()  # Garante que logs pendentes são enviados

        # Monta resultados finais ordenados
        generated = [generated_map[k] for k in sorted(generated_map.keys())]
        paragrafos_json = [paragrafos_map[k] for k in sorted(paragrafos_map.keys())]

        # Liberar cache CUDA final
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # -------- Salvar paragrafos.json --------
        json_path = out_dir / "paragrafos.json"
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(paragrafos_json, f, ensure_ascii=False, indent=2)
            self.log_message.emit(f"✓ paragrafos.json salvo em: {json_path}")
        except Exception as e:
            self.log_message.emit(f"⚠ Não foi possível salvar paragrafos.json: {e}")

        total_time = time.time() - start_time
        mins, secs = divmod(int(total_time), 60)
        time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"

        msg = f"Geração concluída! {len(generated)}/{total} áudios em '{audios_dir}'."
        self.log_message.emit(f"\n✓ {msg} (Tempo demorado: {time_str})")
        self.finished.emit(len(generated) > 0, f"{msg} (Tempo: {time_str})")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _cleanup(path):
    """Deleta arquivo temporário se existir."""
    try:
        p = Path(path)
        if p.exists():
            p.unlink()
    except Exception:
        pass
