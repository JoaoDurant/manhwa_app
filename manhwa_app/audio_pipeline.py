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
import re
import os
import sys
import json
import uuid
import time
import shutil
import threading
import traceback
import importlib
import logging
import contextlib
import concurrent.futures
from pathlib import Path
from difflib import SequenceMatcher
from typing import Any, List, Optional, Tuple, Dict, Generator

# Suppress Numba and HTTP chatty debug logs
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

import numpy as np
import torch
import shutil
from PySide6.QtCore import QObject, Signal

# --- OTIMIZAÇÃO CUDA RTX (5070 Ti) ---
if torch.cuda.is_available():
    # torch.backends.cudnn.benchmark = True  # DESATIVADO no Kokoro, mas útil para Chatterbox
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision('high')
        # [MAX PERFORMANCE] Forçar Flash Attention / SDPA (Hardware max util)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
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
    # =============================================================================
    # Imports de suporte
    # =============================================================================
    from config import config_manager as _config_manager
    from queue import Queue
    from concurrent.futures import ThreadPoolExecutor
    from manhwa_app.text_processor import process_text_fluency, init_spacy
    from manhwa_app.advanced_text_processor import process_text, improve_pronunciation_for_tts
    from manhwa_app.audio_fx import apply_audio_post_processing
    from manhwa_app.models import get_qwen_model, get_whisper_model, unload_whisper, transcribe_audio
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

# Modelos carregados via manhwa_app.models
# ---------------------------------------------------------------------------
# Helpers de texto
# ---------------------------------------------------------------------------

def split_into_paragraphs(text: str) -> List[str]:
    """
    Divide o texto em parágrafos.
    O usuário utiliza Enter para separar parágrafos correspondentes a 1 arquivo de áudio.
    """
    # Remove BOM (Byte Order Mark) e normaliza quebras
    text = text.replace('\ufeff', '').replace("\r\n", "\n").replace("\r", "\n")
    
    # Se o texto "fora de formato" usar quebras duplas para parágrafos, usamos isso.
    # Caso contrário, separamos por cada quebra.
    if "\n\n" in text:
        raw = re.split(r"\n\n+", text)
    else:
        raw = re.split(r"\n+", text)
    
    result = []
    
    for block in raw:
        # Se o bloco começa com um número sozinho numa linha (ex: "1\nTexto"), é um número de painel de script. Removemos ele.
        block = re.sub(r'^\s*\d+\s*\n+', '', block)
        
        # Transforma possíveis quebras de bloco "sujas" (ex: PDF ou colunas) em espaços
        clean_block = block.replace("\n", " ")
        clean_block = re.sub(r"\s+", " ", clean_block).strip()
        if clean_block and len(clean_block) >= 3:
            result.append(clean_block)
            
    return result


def _normalize_text_for_tts(text: str, lang: str = "en", engine: str = "chatterbox") -> str:
    """
    Normaliza o texto antes de enviar para TTS.
    """
    if engine == "kokoro":
        # BYPASS PARA KOKORO: O modelo v1.0 é extremamente sensível à pontuação original.
        # Qualquer alteração (ex: mudar vírgula por ponto) mata a prosódia natural.
        # Apenas limpamos espaços duplos e caracteres invisíveis.
        text = text.replace('\ufeff', '')
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()

    # 1) Normalizar espaços duplos
    text = re.sub(r' {2,}', ' ', text)
    
    # ... Restante da lógica Chatterbox (limpeza agressiva) ...
    text = re.sub(r'\.{4,}', '...', text)
    text = re.sub(r' +([.!?,;:])', r'\1', text)

    return text.strip()


def _text_similarity(a: str, b: str) -> float:
    """Similaridade normalizada SequenceMatcher em [0, 1]."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()



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
            silence_threshold_db=-35.0,  # Menos agressivo para não cortar o final das palavras
            padding_ms=400,              # Aumentado para manter respiração/eco natural no final
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
    # [EVENTO] Sinaliza que um parágrafo terminou o pós-processamento e está pronto
    paragraph_ready = Signal(int, str, dict) # index, wav_path, metadata
    # [BUG FIX] Sinal para a thread principal sinalizando necessidade de troca de engine
    engine_switch_needed = Signal(str, str) # (engine_str, model_type)

    def __init__(
        self,
        file_configs: List[dict],       # [{"path":str, "voice":str, "lang":str}]
        project_name: str,
        output_root: str = "output",
        tts_engine: str = "chatterbox",
        model_type: str = "turbo",      # "turbo" ou "default"
        whisper_model: str = "base",
        similarity_threshold: float = 0.75,
        max_retries: int = 5,
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
        fx_highpass: bool = False,
        fx_deesser: bool = False,
        fx_compressor: bool = False,
        fx_silence: bool = False,
        fx_reverb: bool = False,
        fx_loudnorm: bool = False,
        use_spacy: bool = False,
        use_phonetic: bool = False,
        qwen_task: str = "CustomVoice",
        qwen_instruct: str = "",
        qwen_ref_text: str = "",
        parent=None,
        sample_rate: int = 24000,
        **kwargs,
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
        self.sample_rate           = sample_rate
        self.min_p                 = min_p
        self.top_p                 = top_p
        self.top_k                 = top_k
        self.repetition_penalty    = repetition_penalty
        self.norm_loudness         = norm_loudness
        self.ref_vad_trimming      = ref_vad_trimming
        self.fx_highpass           = fx_highpass
        self.fx_deesser            = fx_deesser
        self.fx_compressor         = fx_compressor
        self.fx_silence            = fx_silence
        self.fx_reverb             = fx_reverb
        self.fx_loudnorm           = fx_loudnorm
        self.use_spacy             = use_spacy
        self.use_phonetic          = use_phonetic
        self.qwen_task             = qwen_task
        self.qwen_instruct         = qwen_instruct
        self.qwen_ref_text         = qwen_ref_text
        self._cancelled            = False
        
        # [PIPELINE PARALLELISM] - 3-stage worker system (Pre-proc | Synthesis | Post-proc)
        from concurrent.futures import ThreadPoolExecutor
        self._executor = ThreadPoolExecutor(max_workers=3)
        
        # [BUG FIX] Event lock for engine switching
        self._switch_event = threading.Event()
        self._switch_event.set() # Inicialmente liberado
        # Otimização CPU: usar todas as threads disponíveis para processamento paralelo (FX, SpaCy)
        if hasattr(torch, 'set_num_threads'):
            torch.set_num_threads(os.cpu_count() or 4)

        # Buffer de logs para evitar overhead de UI (emit agrupado a cada 0.5s)
        self._log_buffer: list = []
        self._log_last_flush: float = 0.0
        # [ESTADO COMPARTILHADO - 3-STAGE PIPELINE]
        self._state_lock = threading.Lock()
        self.generated_map: Dict[int, str] = {}
        self.paragrafos_map: Dict[int, dict] = {}
        self.completed_indices: Set[int] = set()
        
        # Referências para Stage 3 (UI/ETA)
        self._start_time_ref: float = 0.0
        self._all_paragraphs_ref: List[Tuple[str, str, dict]] = []
        
        # [ESTABILIDADE] Pool de threads persistente durante a vida do worker
        self._poster: Optional[ThreadPoolExecutor] = None
        
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

    def confirm_switch_done(self):
        """[BUG FIX] Main thread chama isso para liberar a geração do pipeline."""
        if hasattr(self, "_switch_event"):
            self._switch_event.set()

    def _get_kokoro_lang(self, lang):
        _m = {"pt": "p", "pt-br": "p", "en": "a", "en-us": "a", "en-gb": "b", "es": "e", "fr": "f", "ja": "j", "zh": "z"}
        return _m.get(lang.lower(), "a")

    def _preprocess_paragraph_task(self, paragraph, lang):
        """Tarefa executada em thread do pool para paralelizar o Stage 1 (CPU)."""
        try:
            from manhwa_app.text_processor import process_text_fluency
            from manhwa_app.advanced_text_processor import process_text
            
            kokoro_lang = self._get_kokoro_lang(lang)
            tts_text = paragraph
            
            if self.use_spacy:
                tts_text = process_text_fluency(tts_text, lang=lang)
            
            adv_cfg = _config_manager.get("production.text_processing", {
                "normalize_text": True, "clean_symbols": True, "improve_punctuation": True,
                "add_natural_pauses": True, "convert_numbers": True
            })
            verification_text = process_text(tts_text, adv_cfg, lang=kokoro_lang)
            
            # _normalize_text_for_tts é uma função local deste módulo (audio_pipeline.py)
            tts_text = _normalize_text_for_tts(verification_text, lang=lang, engine=self.tts_engine)
            
            return (tts_text, verification_text, kokoro_lang)
        except Exception as e:
            logger.error(f"Erro no Stage 1 (Pre-proc): {e}")
            return (paragraph, paragraph, lang)

    def _post_process_task(self, idx, wav_tmp, silence_out, wav_final, p_cfg, paragraph, source_file, sample_rate, fixed_sim=0.0):
        """
        [STAGE 3 - INDUSTRIAL] Pós-processamento e I/O Assíncrono.
        A validação Whisper ocorreu sincronamente no Stage 2 para permitir retentativas.
        """
        best_sim = fixed_sim
        try:
            # 1. Remoção de silêncio (CPU em background)
            final_path_no_fx = _remove_silence_from_file(str(wav_tmp), str(silence_out), sample_rate)
            
            # 3. Aplicação de FX
            # [QUALIDADE KOKORO] Por padrão, desativamos filtros agressivos para Kokoro 
            # para preservar o brilho original dos 24kHz.
            is_kokoro = (self.tts_engine == "kokoro")
            
            f_hp = p_cfg.get("fx_highpass", self.fx_highpass if not is_kokoro else False)
            f_de = p_cfg.get("fx_deesser", self.fx_deesser if not is_kokoro else False)
            f_cp = p_cfg.get("fx_compressor", self.fx_compressor)
            f_sl = p_cfg.get("fx_silence", self.fx_silence)
            f_rv = p_cfg.get("fx_reverb", self.fx_reverb)
            f_ln = p_cfg.get("fx_loudnorm", self.fx_loudnorm)

            _any_fx = f_hp or f_de or f_cp or f_sl or f_rv or f_ln
            if _any_fx:
                fx_config = {"production": {"audio": {"highpass": f_hp, "deesser": f_de, "compressor": f_cp, "reverb": f_rv, "remove_silence": f_sl, "normalize": f_ln}}}
                try:
                    from manhwa_app.audio_fx import apply_audio_post_processing
                    apply_audio_post_processing(final_path_no_fx, str(wav_final), fx_config)
                except Exception as fx_err:
                    logger.warning(f"FX falhou parágrafo {idx}: {fx_err}. Usando áudio sem FX.")
                    shutil.copy2(final_path_no_fx, str(wav_final))
            else:
                shutil.copy2(final_path_no_fx, str(wav_final))

            # 4. Notificação e Mapas (Thread-Safe)
            with self._state_lock:
                self.generated_map[idx] = str(wav_final)
                self.paragrafos_map[idx] = {
                    "index": idx,
                    "audio": f"audio_{idx}.wav",
                    "texto": paragraph,
                    "arquivo_origem": source_file,
                    "similaridade": round(best_sim, 3),
                }
                self.completed_indices.add(idx)
                current_completed = len(self.completed_indices)
            
            # 5. UI Updates
            self.paragraph_done.emit(idx, str(wav_final), paragraph)
            self.paragraph_ready.emit(idx, str(wav_final), self.paragrafos_map[idx])
            
            # Calcular progresso/ETA
            elapsed = time.time() - self._start_time_ref
            total = len(self._all_paragraphs_ref)
            avg = elapsed / current_completed if current_completed > 0 else 0
            eta = int(avg * (total - current_completed))
            eta_s = f"{eta//60}m {eta%60}s" if eta >= 60 else f"{eta}s"
            
            self.progress.emit(current_completed, total)
            self.log_message.emit(f"  ✓ [#{idx}/{total}] Áudio finalizado (Sim: {best_sim:.2f}). ETA: {eta_s}")
            
        except Exception as e:
            logger.error(f"Erro Crítico no Stage 3 [#{idx}]: {e}", exc_info=True)
            self.log_message.emit(f"  ✗ Erro ao processar áudio [#{idx}]: {str(e)[:150]}")
        finally:
            # LIMPEZA ATÔMICA
            _cleanup(wav_tmp)
            _cleanup(silence_out)
            if best_wav_path: _cleanup(best_wav_path)
            

    def run(self):
        start_time = time.time()
        try:
            self._run_internal(start_time)
        except Exception as e:
            logger.error(f"Falha não capturada no AudioPipeline: {e}", exc_info=True)
            self.log_message.emit(f"❌ Erro interno no pipeline: {e}")
            self.finished.emit(False, f"Erro interno: {e}")

    def _find_resume_index(self, audios_dir: Path, all_paragraphs: list) -> int:
        """
        Transcreve os últimos 3 áudios gerados e faz fuzzy match iterativo para descobrir o ponto exato de parada.
        Retorna o índice (0-based) de onde a lista all_paragraphs deve retomar.
        """
        import difflib, unicodedata, re

        wav_files = []
        for f in audios_dir.glob("audio_*.wav"):
            try:
                num = int(f.stem.split('_')[1])
                wav_files.append((num, f))
            except (IndexError, ValueError):
                pass

        if not wav_files:
            return 0

        wav_files.sort(key=lambda x: x[0])
        check_files = wav_files[-3:]
        
        if self.tts_engine == "qwen":
            self.log_message.emit("⚡ Qwen ativo: Bypass da verificação Whisper no Resume para economizar VRAM. Assumindo índice sequecial ativo.")
            return wav_files[-1][0]

        model, _ = get_whisper_model(self.whisper_model)
        if model is None:
            self.log_message.emit("⚠ Whisper não disponível. Assumindo índice de forma passiva.")
            return wav_files[-1][0]

        def _normalize(t):
            t = unicodedata.normalize('NFKD', t.lower()).encode('ASCII', 'ignore').decode('utf-8')
            t = re.sub(r'[^a-z0-9\s]', '', t)
            return re.sub(r'\s+', ' ', t).strip()

        best_overall_index = 0

        for num, wav_path in check_files:
            try:
                # faster-whisper returns segments, openai-whisper returns dict
                if hasattr(model, 'transcribe') and hasattr(model, 'supported_languages'):
                    segments, _ = model.transcribe(str(wav_path), beam_size=5)
                    transcribed = " ".join(seg.text for seg in segments)
                else:
                    result = model.transcribe(str(wav_path), fp16=torch.cuda.is_available())
                    transcribed = result.get("text", "")

                norm_transcribed = _normalize(transcribed)
                
                # Procura correspondência na janela de tolerância de +/- 10
                start_search = max(0, num - 10)
                end_search = min(len(all_paragraphs), num + 10)
                
                best_match_idx = -1
                best_ratio = 0.0
                
                for i in range(start_search, end_search):
                    norm_orig = _normalize(all_paragraphs[i][0])
                    sm = difflib.SequenceMatcher(None, norm_transcribed, norm_orig)
                    ratio = sm.ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_match_idx = i
                        
                if best_ratio > 0.70 and best_match_idx >= 0:
                    self.log_message.emit(f"  ✓ Validou áudio #{num} correspondente ao índice {best_match_idx} do texto original.")
                    best_overall_index = max(best_overall_index, best_match_idx + 1)
            except Exception as e:
                self.log_message.emit(f"  ⚠ Resumo: Falha ao analisar {wav_path.name}: {e}")
                
        if best_overall_index == 0:
            self.log_message.emit("  ⚠ Não encontrou match seguro. Recomeçando do zero.")
            
        return best_overall_index

    def _run_internal(self, start_time: float):
        import time
        print(f"[PIPELINE] Início do _run_internal | Engine alvo: {self.tts_engine} | StartTime: {start_time}")
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

        # Flush agressivo de VRAM antes de começar o pipeline
        # CORRIGIDO: garante que erros de runs anteriores não deixem VRAM presa
        try:
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
        except Exception:
            pass

        # -------- Lê e separa os parágrafos de todos os .txt --------
        # all_paragraphs contém: (texto, source_file, paragraph_config)
        all_paragraphs: List[Tuple[str, str, dict]] = []
        for fcfg in self.file_configs:
            txt_path = fcfg["path"]
            try:
                text = Path(txt_path).read_text(encoding="utf-8-sig", errors="replace")
                paragraphs = split_into_paragraphs(text)
            except Exception as e:
                self.log_message.emit(f"⚠ Não foi possível ler '{txt_path}': {e}")
                continue

            self.log_message.emit(
                f"✓ {len(paragraphs)} parágrafo(s) em '{Path(txt_path).name}' "
                f"(Idioma: {fcfg.get('lang','en')}, Engine: {fcfg.get('engine', self.tts_engine)})."
            )
            for p in paragraphs:
                all_paragraphs.append((p, Path(txt_path).name, fcfg))

        self._all_paragraphs_ref = all_paragraphs
        self._start_time_ref = start_time

        if not all_paragraphs:
            self.finished.emit(False, "Nenhum parágrafo encontrado para processar.")
            return

        # [ESTABILIDADE] Pool persistente gerenciado via self._poster (Stage 3 + Whisper GPU)
        from concurrent.futures import ThreadPoolExecutor
        self._poster = ThreadPoolExecutor(max_workers=12)

        try:
            out_dir = Path(self.output_root) / self.project_name
            audios_dir = out_dir / "audios"
            audios_dir.mkdir(parents=True, exist_ok=True)
            
            total = len(all_paragraphs)
            use_whisper_verification = (self.similarity_threshold > 0.0)
            if self.tts_engine == "qwen":
                use_whisper_verification = False
            elif use_whisper_verification:
                self.log_message.emit(f"🔌 Carregando Whisper [{self.whisper_model}] em CUDA para validação...")
                from manhwa_app.models import get_whisper_model
                get_whisper_model(self.whisper_model, device_override=None) # Força CUDA

            if self.tts_engine == "kokoro":
                if engine.get_active_engine() != "kokoro":
                    self.log_message.emit("Carregando motor Kokoro-TTS...")
                    engine.switch_to_engine("kokoro")
            elif self.tts_engine == "chatterbox":
                if engine.get_active_engine() != "chatterbox":
                    self.log_message.emit("Carregando motor Chatterbox...")
                    engine.switch_to_engine("chatterbox")

            self.log_message.emit(f"🚀 Iniciando Geração de {total} parágrafo(s)...")


            # -------- Detecção de Lacunas (Smart Fill) --------
            missing_indices = self._get_missing_indices(audios_dir, all_paragraphs)
            total = len(all_paragraphs)
            existing_count = total - len(missing_indices)
            
            # Popula completados com o que já existe para o progresso da UI
            for idx_ext in range(1, total + 1):
                if idx_ext not in missing_indices:
                    with self._state_lock:
                        self.completed_indices.add(idx_ext)

            if existing_count > 0:
                if len(missing_indices) == 0:
                    self.log_message.emit("✅ Todos os áudios já existem. Geração concluída!")
                    self.progress.emit(total, total)
                    return
                self.log_message.emit(f"🔄 [SMART-FILL] {existing_count} áudios encontrados. Gerando apenas os {len(missing_indices)} faltantes.")
                self.progress.emit(existing_count, total)

            # -------- Loop Principal (Stage 2 - GPU) --------
            # [LOOKAHEAD] Inicializa o pre-processamento para o primeiro item da lista de faltantes
            next_preprocess_future = None
            if missing_indices:
                first_missing_idx = missing_indices[0]
                p_next_p0, _, p_next_cfg0 = all_paragraphs[first_missing_idx - 1]
                next_preprocess_future = self._executor.submit(self._preprocess_paragraph_task, p_next_p0, p_next_cfg0.get("lang", "en"))

            for i, idx in enumerate(missing_indices): # idx é 1-based
                # paragraph indexing is idx - 1
                paragraph, source_file, p_cfg = all_paragraphs[idx - 1]
                
                if self._cancelled or self.thread().isInterruptionRequested():
                    break

                wav_final = audios_dir / f"audio_{idx}.wav"
                wav_tmp   = audios_dir / f"audio_{idx}_tmp.wav"
                
                lang = p_cfg.get("lang", "en")
                kokoro_lang = self._get_kokoro_lang(lang)
                chatter_lang = lang.lower().split("-")[0]
                voice_path = p_cfg.get("voice")
                engine_str = p_cfg.get("engine", self.tts_engine)
                p_temp = p_cfg.get("temperature", self.temperature)
                
                try:
                    tts_text_base, verification_text, kokoro_lang = next_preprocess_future.result(timeout=60)
                except Exception as e:
                    logger.warning(f"Erro no lookahead #{idx}: {e}")
                    tts_text_base, verification_text = (paragraph, paragraph)

                # Dispara lookahead para o próximo item da lista de faltantes
                if i + 1 < len(missing_indices):
                    next_idx = missing_indices[i + 1]
                    p_next, _, cfg_next = all_paragraphs[next_idx - 1]
                    next_preprocess_future = self._executor.submit(self._preprocess_paragraph_task, p_next, cfg_next.get("lang", "en"))

                tts_text_input = _normalize_text_for_tts(tts_text_base, lang=lang, engine=engine_str)

                # Geração com Loop de Retentativa (Aceleração CUDA + Whisper Sync + Best Effort)
                success = False
                best_sim_so_far = -1.0
                best_wav_path = audios_dir / f"audio_{idx}_best_effort.wav"
                current_temp = p_temp
                
                from manhwa_app.models import transcribe_audio
                
                for attempt in range(1, 4): # Max 3 tentativas
                    if self._cancelled: break
                    
                    # Log de tentativa se for retry
                    if attempt > 1:
                        self.log_message.emit(f"  ⚠️ [RETRY] #{idx} (Tentativa {attempt}/3) | Temp: {current_temp:.2f}...")
                    
                    success = utils.generate_paragraph_audio(
                        text=tts_text_input, output_path=str(wav_tmp),
                        engine_name=engine_str, 
                        audio_prompt_path=voice_path, # Para Chatterbox/Turbo
                        kokoro_voice=voice_path,     # Para Kokoro
                        kokoro_lang=kokoro_lang, language=chatter_lang,
                        temperature=current_temp, sample_rate=self.sample_rate
                    )

                    if success:
                        # Validação Imediata (GPU Whisper é ultra-rápido)
                        similarity = 1.0
                        if use_whisper_verification and verification_text:
                            transcription = transcribe_audio(str(wav_tmp), self.whisper_model, device_override=None)
                            similarity = _text_similarity(verification_text, transcription)
                            
                            msg_val = f"  [VALIDAÇÃO] #{idx} Sim: {similarity:.2f}"
                            logger.info(msg_val)
                            print(msg_val)
                            
                            # Rastrear a melhor tentativa para o modo de segurança (best effort)
                            if similarity > best_sim_so_far:
                                best_sim_so_far = similarity
                                # Salva esta tentativa caso todas as outras falhem
                                try:
                                    if best_wav_path.exists(): best_wav_path.unlink()
                                    shutil.copy2(str(wav_tmp), str(best_wav_path))
                                except: pass

                            if similarity < self.similarity_threshold:
                                if attempt < 3:
                                    self.log_message.emit(f"  ⚠️ [REGEN] #{idx}: Similaridade baixa ({similarity:.2f} < {self.similarity_threshold}). Regerando...")
                                    current_temp += 0.05
                                    _cleanup(wav_tmp)
                                    continue # Tenta de novo
                                else:
                                    # Falhou após 3 tentativas - Modo Best Effort
                                    self.log_message.emit(f"  ⚖️ [BEST-EFFORT] #{idx}: Nenhuma tentativa atingiu o limite. Usando melhor versão (Sim: {best_sim_so_far:.2f}).")
                                    shutil.copy2(str(best_wav_path), str(wav_tmp))
                                    similarity = best_sim_so_far
                        
                        # Se chegou aqui, passou na validação ou estamos no modo best effort
                        best_sim = similarity
                        _wav_tmp_path = Path(str(wav_tmp))
                        silence_out = _wav_tmp_path.with_name(_wav_tmp_path.stem.replace("_tmp", "_silence") + _wav_tmp_path.suffix)
                        
                        self._poster.submit(
                            self._post_process_task, 
                            idx, Path(str(wav_tmp)), silence_out, wav_final, 
                            p_cfg, paragraph, source_file, self.sample_rate,
                            best_sim
                        )
                        break
                    else:
                        _cleanup(wav_tmp)
                
                # Limpeza final do arquivo de best effort
                _cleanup(best_wav_path)
                
                # --- ANTI-FRAGMENTAÇÃO VRAM (Solução para degradação do Chatterbox) ---
                # Quando rodamos Transformers generate() 1000+ vezes, a VRAM se fragmenta
                # porque o alocador do PyTorch não devolve a memória solta ao SO.
                # Forçamos a limpeza das referências da iteração anterior para manter a velocidade 100%.
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    # Vazio leve: executa rápido e garante blocos grandes livres
                    torch.cuda.empty_cache()

            self.log_message.emit("⌛ Aguardando finalização dos áudios em segundo plano...")
            
        except Exception as e:
            logger.error(f"Erro no loop _run_internal: {e}", exc_info=True)
            self.log_message.emit(f"❌ Erro fatal: {e}")
        finally:
            if self._poster:
                self._poster.shutdown(wait=True)
                self._poster = None

        with self._state_lock:
            paragrafos_json = [self.paragrafos_map[k] for k in sorted(self.paragrafos_map.keys())]
            generated_count = len(self.completed_indices)

        json_path = out_dir / "paragrafos.json"
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(paragrafos_json, f, ensure_ascii=False, indent=2)
            self.log_message.emit(f"✓ Metadados salvos em: {json_path}")
        except: pass

        total_time = time.time() - start_time
        mins, secs = divmod(int(total_time), 60)
        time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"

        msg = f"Geração concluída! {generated_count}/{total} áudios gerados."
        self.log_message.emit(f"\n✓ {msg} (Tempo Total: {time_str})")
        self.finished.emit(generated_count > 0, msg)

    def _get_missing_indices(self, audios_dir: Path, all_paragraphs: list) -> List[int]:
        """
        Retorna a lista de índices (1-based) que não possuem o arquivo audio_N.wav correspondente.
        Isto permite o modo 'Smart Fill' (preencher buracos na sequência).
        """
        missing = []
        for i in range(1, len(all_paragraphs) + 1):
            wav_path = audios_dir / f"audio_{i}.wav"
            if not wav_path.exists():
                missing.append(i)
        return missing

    def _find_resume_index(self, audios_dir: Path, all_paragraphs: list) -> int:
        """Encontra o ponto de retomada baseado nos arquivos existentes."""
        wav_files = list(audios_dir.glob("audio_*.wav"))
        if not wav_files: return 0
        
        valid_indices = []
        for f in wav_files:
            try:
                num = int(f.stem.split('_')[1])
                valid_indices.append(num)
            except: pass
        if not valid_indices: return 0
        
        max_idx = max(valid_indices)
        return min(max_idx, len(all_paragraphs))

    def _get_kokoro_lang(self, lang: str) -> str:
        _k_map = {"pt": "p", "pt-br": "p", "en": "a", "en-us": "a", "en-gb": "b", "es": "e", "fr": "f", "ja": "j", "zh": "z"}
        return _k_map.get(lang.lower(), "a")

    def run(self):
        start_time = time.time()
        try:
            self._run_internal(start_time)
        except Exception as e:
            logger.error(f"Crash no AudioWorker: {e}", exc_info=True)
            self.finished.emit(False, str(e))

def _cleanup(path):
    """Deleta arquivo temporário se existir."""
    try:
        p = Path(path)
        if p.exists():
            p.unlink()
    except Exception:
        pass
