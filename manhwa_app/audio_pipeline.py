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
    from manhwa_app.advanced_text_processor import process_text
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
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Separa quebras explícitas (1 ou mais newlines representam parágrafos distintos)
    raw = re.split(r"\n+", text)
    
    result = []
    
    for block in raw:
        clean_block = block.strip()
        if clean_block and len(clean_block) >= 3:
            result.append(clean_block)
            
    return result


def _normalize_text_for_tts(text: str, lang: str = "en") -> str:
    """
    Normaliza o texto antes de enviar para TTS.
    Corrige o problema do align_stream_analyzer do Chatterbox multilingual
    que detecta tokens de vírgula repetidos e força EOS prematuramente.
    """
    # 1) Normalizar espaços duplos
    text = re.sub(r' {2,}', ' ', text)

    # 2) Para ling. latinas (es, pt, it, fr, de): substituir vírgulas por pausa mais longa.
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
        qwen_task: str = "CustomVoice",
        qwen_instruct: str = "",
        qwen_ref_text: str = "",
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
        self.qwen_task             = qwen_task
        self.qwen_instruct         = qwen_instruct
        self.qwen_ref_text         = qwen_ref_text
        self._cancelled            = False
        
        # [BUG FIX] Event lock for engine switching
        self._switch_event = threading.Event()
        self._switch_event.set() # Inicialmente liberado
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

    def confirm_switch_done(self):
        """[BUG FIX] Main thread chama isso para liberar a geração do pipeline."""
        if hasattr(self, "_switch_event"):
            self._switch_event.set()

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
                text = Path(txt_path).read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                self.log_message.emit(f"⚠ Não foi possível ler '{txt_path}': {e}")
                continue

            paragraphs = split_into_paragraphs(text)
            self.log_message.emit(
                f"✓ {len(paragraphs)} parágrafo(s) em '{Path(txt_path).name}' "
                f"(Idioma: {fcfg.get('lang','en')}, Engine: {fcfg.get('engine','chatterbox')})."
            )
            for p in paragraphs:
                all_paragraphs.append((p, Path(txt_path).name, fcfg))

        if not all_paragraphs:
            self.finished.emit(False, "Nenhum parágrafo encontrado nos arquivos (separe com linhas em branco).")
            return

        # -------- Prepara diretório de saída --------
        out_dir = Path(self.output_root) / self.project_name
        audios_dir = out_dir / "audios"
        audios_dir.mkdir(parents=True, exist_ok=True)
        self.log_message.emit(f"Pasta de saída: {out_dir.resolve()}")

        # -------- Garante que o modelo TTS está carregado --------
        # O manager do dispatcher vai carregar sob demanda, mas logamos aqui para claridade
        if self.tts_engine == "chatterbox":
            self.log_message.emit("Verificando/Carregando modelo Chatterbox TTS...")
            if not engine.load_model(model_type=self.model_type):
                self.finished.emit(False, "Falha ao carregar modelo Chatterbox. Verifique os logs.")
                return
            device_str = (engine.model_device or "cpu").upper()
            model_type_log = engine.loaded_model_type or "original"
            self.log_message.emit(f"✓ Modelo Chatterbox pronto em {device_str} (tipo: {model_type_log}).")
            
        elif self.tts_engine == "kokoro":
            self.log_message.emit("Verificando/Carregando motor Kokoro TTS...")
            if not engine.load_kokoro_engine():
                self.finished.emit(False, "Falha ao carregar modelo Kokoro. Verifique os logs.")
                return
            self.log_message.emit("✓ Módulo Kokoro pronto.")
            
        elif self.tts_engine == "qwen":
            self.log_message.emit("Verificando/Carregando motor Qwen TTS...")
            if not engine.load_qwen_model():
                self.finished.emit(False, "Falha ao carregar modelo Qwen. Verifique os logs.")
                return
            self.log_message.emit("✓ Módulo Qwen pronto.")

        # -------- Processa parágrafos SEQUENCIALMENTE (o engine.synthesize usa Lock global) --------
        # NOTA IMPORTANTE: o engine.synthesize() usa _synthesis_lock (threading.Lock global),
        # o que torna qualquer paralelismo ineficaz. Um loop sequencial é mais limpo,
        # elimina overhead de ThreadPoolExecutor, e evita corridas de VRAM.
        total = len(all_paragraphs)
        completed = 0
        generated_map: Dict[int, str] = {}
        paragrafos_map: Dict[int, dict] = {}

        # Pre-loading de modelos pesados ANTES do loop principal
        use_whisper_verification = (self.similarity_threshold > 0.0)
        
        if self.use_spacy:
            self.log_message.emit("Pre-carregando SpaCy (NLP)…")
            # Inicializa SpaCy para cada idioma único presente
            langs_in_use = set(p[3] for p in all_paragraphs)
            for lng in langs_in_use:
                init_spacy(lng)

        # ---- Desativa verificações e recargas desnecessárias se for Qwen ----
        if self.tts_engine == "qwen":
            use_whisper_verification = False
            self.log_message.emit("⚡ Otimização: Whisper desativado para o Qwen.")
        else:
            if use_whisper_verification:
                self.log_message.emit(f"Carregando faster-whisper ({self.whisper_model}) para verificação de qualidade…")
                get_whisper_model(self.whisper_model)  # síncrono — garante disponibilidade no loop

        # Definir seed globalmente UMA VEZ antes do loop (não a cada parágrafo)
        # CORRIGIDO: envolvido em try/except — se GPU estiver em estado de erro
        # de um pipeline anterior (ex: cudaErrorLaunchTimeout), manual_seed_all()
        # lança exceção FORA do try/except de síntese, quebrando _run_internal desde
        # o início sem mensagem de erro útil.
        if self.seed != 0 and self.tts_engine == "chatterbox":
            try:
                import engine as _eng_ref
                _eng_ref.set_seed(self.seed)
                self.log_message.emit(f"Seed global definida: {self.seed}")
            except Exception as _seed_err:
                self.log_message.emit(f"  ⚠ set_seed falhou (GPU pode estar em estado de erro): {str(_seed_err)[:80]}")
                # Não abortar — a síntese pode ainda funcionar sem seed global

        self.log_message.emit(f"🚀 Iniciando Geração de {total} parágrafo(s) (sequencial, GPU otimizada)...")

        kokoro_models_cache = {}


        # --- Engine Switch (Worker / Kokoro) if needed ---
        if self.tts_engine in ("qwen", "indextts"):
            self.log_message.emit(f"⚡ [Worker] Geração delegada ao worker isolado ({self.tts_engine}).")
        elif self.tts_engine == "kokoro":
            import engine as _eng
            if _eng.get_active_engine() != "kokoro":
                self.log_message.emit("Carregando motor Kokoro-TTS (Local)...")
                _eng.switch_to_engine("kokoro")

        # -------- Sistema de Recovery / Continuação Automática --------
        start_index = 0
        existing_audios = list(audios_dir.glob("audio_*.wav"))
        if existing_audios:
            start_index = self._find_resume_index(audios_dir, all_paragraphs)
            if start_index > 0:
                self.log_message.emit(f"🔄 Retomando geração a partir do parágrafo {start_index + 1}...")
                
                # Reconstrói os mapas prévios para não perder o log do que já foi feito na UI
                for num, _ in [(int(f.stem.split('_')[1]), f) for f in existing_audios if f.stem.split('_')[1].isdigit()]:
                    if num <= start_index:
                        generated_map[num] = str(audios_dir / f"audio_{num}.wav")
                        paragrafos_map[num] = {"index": num, "audio": f"audio_{num}.wav"}
                        
                completed = start_index
                self.progress.emit(completed, total)

        pending_paragraphs = all_paragraphs[start_index:]

        for idx, (paragraph, source_file, p_cfg) in enumerate(pending_paragraphs, start=start_index + 1):
            lang = p_cfg.get("lang", "en")
            voice_path = p_cfg.get("voice")
            engine_str = p_cfg.get("engine", self.tts_engine)
            p_speed = p_cfg.get("speed", self.speed)
            p_temp = p_cfg.get("temperature", self.temperature)
            if self._cancelled or self.thread().isInterruptionRequested():
                break

            wav_final = audios_dir / f"audio_{idx}.wav"
            wav_tmp   = audios_dir / f"audio_{idx}_tmp.wav"
            
            # CORRIGIDO: Restaurada a limpeza do cache para evitar WDDM swap (lentidão severa no Windows)
            # que foi a causa relatada do aumento para ~2 mins por parágrafo
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                except Exception as _sync_err:
                    self.log_message.emit(f"  ⚠ GPU sync/cache falhou (ignorado): {str(_sync_err)[:100]}")

            success  = False
            best_sim = 0.0

            # Medição de tempo exato por parágrafo para detectar memory leak (ex: acúmulo de KV cache)
            paragraph_start_time = time.time()

            for attempt in range(1, self.max_retries + 1):
                if self._cancelled:
                    break

                # ---- Pre-processamento de texto ----
                tts_text_input = paragraph
                if self.use_spacy:
                    tts_text_input = process_text_fluency(tts_text_input, lang=lang)
                
                # Integrando o advanced_text_processor guiado por config
                if _config_manager:
                    advanced_text_config = _config_manager.get("production.text_processing", {
                        "normalize_text": True,
                        "remove_accents": False,
                        "clean_symbols": True,
                        "improve_punctuation": True,
                        "add_natural_pauses": True,
                        "lowercase": False,
                        "use_phonetic": False
                    })
                    tts_text_input = process_text(tts_text_input, advanced_text_config)

                # CORRIGIDO: normalização TTS específica para idiomas latinos
                # Ex: espanhol com vírgulas causa EOS prematuro no alinhador Chatterbox.
                # _normalize_text_for_tts já existia no módulo, mas não era chamada!
                tts_text_input = _normalize_text_for_tts(tts_text_input, lang=lang)

                # LOG DE DIAGNÓSTICO: mostra o texto que será enviado ao TTS (primeiros 120 chars)
                if attempt == 1:
                    preview = tts_text_input[:120].replace('\n', ' ')
                    self.log_message.emit(f"  ℹ [#{idx}] Texto TTS (tentativa 1): '{preview}{'...' if len(tts_text_input) > 120 else ''}'")

                # Pular parágrafos muito curtos (<10 chars) que disparam EOS imediato
                if len(tts_text_input.strip()) < 10:
                    self.log_message.emit(f"  ⚠ [#{idx}] Parágrafo muito curto ignorado: '{tts_text_input[:40]}'")
                    break  # CORRIGIDO: break para sair das tentativas, não 'continue' que reiniciava o loop

                # ---- Síntese TTS ----
                wav_tensor = None
                sample_rate = 24000

                try:
                    # 1. Estratégia de Síntese
                    wav_tensor = None
                    sample_rate = 24000
                    
                    import time
                    t0_gen = time.time()
                    logger.info(f"[PIPELINE] Iniciando geração de áudio (Parágrafo {idx}) | Texto len: {len(tts_text_input)} chars | Engine: {engine_str}")
                    logger.info(f"[PIPELINE-DEBUG] Final voice for Kokoro='{voice_path}', engine='{engine_str}'")
                    
                    # --- NOVO DISPATCHER UNIFICADO (engine.py + utils.py) ---
                    # Substitui os blocos If/Else fragmentados por um ponto único de entrada
                    
                    active_eng = getattr(_engine, "get_active_engine")() if hasattr(_engine, "get_active_engine") else "none"
                    current_model = getattr(_engine, "loaded_model_type", "none")
                    
                    print(f"[DEBUG] AudioPipeline: engine_str='{engine_str}', model_type='{self.model_type}'")
                    print(f"[DEBUG] Engine State: active='{active_eng}', loaded_type='{current_model}'")

                    # [BUG 2 FIX] _needs_engine_switch: mapeia nomes de engine para sistemas
                    # 'kokoro' é um sistema separado de 'chatterbox' (turbo/original/multilingual).
                    # A comparação antiga (active_eng vs engine_str) misturava namespaces.
                    KOKORO_ENGINES = {'kokoro'}
                    CHATTERBOX_SUBTYPES = {'turbo', 'original', 'multilingual'}

                    def _needs_engine_switch(active, requested, loaded_subtype, requested_model_type):
                        active_is_kokoro = active in KOKORO_ENGINES
                        req_is_kokoro = requested in KOKORO_ENGINES
                        # Sistemas diferentes → sempre trocar
                        if active_is_kokoro != req_is_kokoro:
                            return True
                        # Chatterbox: checar submodelo
                        if not req_is_kokoro:
                            if loaded_subtype != requested_model_type:
                                return True
                        return False

                    needs_switch = _needs_engine_switch(
                        active_eng, engine_str, current_model,
                        self.model_type if self.model_type else "turbo"
                    )

                    if needs_switch:
                        target = engine_str
                        if engine_str == "chatterbox":
                            target = self.model_type if self.model_type else "turbo"

                        logger.info(f"[PIPELINE] Pausando geração para troca de engine: {active_eng} → {target}")
                        self.log_message.emit(f"🔄 Solicitando motor TTS: {target.upper()}...")
                        
                        # Bloqueia a execução desta thread e sinaliza a thread principal
                        self._switch_event.clear()
                        self.engine_switch_needed.emit(engine_str, self.model_type)
                        
                        # Aguarda a main thread baixar/carregar e setar self._switch_event.set()
                        swapped = self._switch_event.wait(timeout=120)
                        if not swapped:
                            raise Exception("Tempo esgotado aguardando troca de engine.")
                        
                        # Re-valida para garantir que a memória foi preenchida
                        import engine as eng_local
                        active_eng = getattr(eng_local, "get_active_engine")() if hasattr(eng_local, "get_active_engine") else "none"
                        logger.info(f"[PIPELINE] Troca de engine confirmada: {active_eng}")
                    else:
                        print(f"[DEBUG] Switch não necessário: {active_eng}/{current_model} já é o engine correto")


                    success = _utils.generate_paragraph_audio(
                        text=tts_text_input,
                        output_path=str(wav_tmp),
                        engine_name=engine_str,
                        audio_prompt_path=voice_path,
                        kokoro_voice=voice_path,  # [FIX] Kokoro expects 'kokoro_voice' in kwargs
                        qwen_speaker=self.qwen_speaker if hasattr(self, 'qwen_speaker') else "Ryan",
                        qwen_language="Auto",
                        indextts_speed=p_speed,
                        temperature=p_temp,
                        exaggeration=p_cfg.get("exaggeration", self.exaggeration),
                        cfg_weight=p_cfg.get("cfg_weight", self.cfg_weight),
                        seed=p_cfg.get("seed", self.seed)
                    )

                    if not success:
                        print(f"[PIPELINE] Fim da geração (Parágrafo {idx}) | Sucesso: False | Tempo: {time.time() - t0_gen:.2f}s")
                        self.log_message.emit(f"  ✗ Falha na síntese do parágrafo via {self.tts_engine}")
                        _cleanup(wav_tmp)
                        continue
                        
                    print(f"[PIPELINE] Fim da geração (Parágrafo {idx}) | Sucesso: True | Tempo: {time.time() - t0_gen:.2f}s")
                    
                    # Carregar para verificação (se necessário) ou apenas para marcar sucesso
                    # Como generate_paragraph_audio já salvou o arquivo, não precisamos re-salvar.
                    # Mas o código abaixo espera wav_tensor != None para continuar.
                    wav_tensor = True # Flag para o loop continuar

                    if wav_tensor is not None:
                        # Geração bem-sucedida! (O arquivo já foi salvo pelo dispatcher em wav_tmp)
                        paragraph_end_time = time.time()
                        self.log_message.emit(f"  ⏱ ↳ Geração do parágrafo durou {paragraph_end_time - paragraph_start_time:.1f}s")

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
                                # Sem synchronize() extra aqui — já foi feito após a síntese
                                torch.cuda.empty_cache()
                                gc.collect()
                                self._log_vram(f"Pós-Whisper #{idx}")

                        # ---- Remoção de silêncio ----
                        # CORRIGIDO: str.replace("_tmp", "_silence") era frágil: se o CAMINHO
                        # da pasta de saída contivesse "_tmp" (ex: pasta "tmp_test"),
                        # o path de saída ficava corrompido. Usando stem/suffix agora.
                        _wav_tmp_path = Path(str(wav_tmp))
                        silence_out = str(_wav_tmp_path.with_name(_wav_tmp_path.stem.replace("_tmp", "_silence") + _wav_tmp_path.suffix))
                        final_path_no_fx = _remove_silence_from_file(str(wav_tmp), silence_out, sample_rate)

                        # ---- FX de áudio (opcionais) ----
                        # CORRIGIDO: usa parâmetros de p_cfg se existirem, senão fallback global
                        f_hp = p_cfg.get("fx_highpass", self.fx_noise_reduction)
                        f_cp = p_cfg.get("fx_compressor", self.fx_compressor)
                        f_eq = p_cfg.get("fx_deesser", self.fx_eq)
                        f_rv = p_cfg.get("fx_reverb", self.fx_reverb)
                        f_sl = p_cfg.get("fx_silence", self.fx_enhancer)
                        f_nm = p_cfg.get("fx_loudnorm", self.fx_normalize)

                        _any_fx = f_hp or f_cp or f_eq or f_rv or f_sl or f_nm
                        if _any_fx:
                            fx_config = {
                                "production": {
                                    "audio": {
                                        "highpass":       f_hp,
                                        "compressor":     f_cp,
                                        "reverb":         f_rv,
                                        "normalize":      f_nm,
                                        "remove_silence": f_sl,
                                    }
                                },
                                "fx_noise_reduction": f_hp,
                                "fx_compressor":      f_cp,
                                "fx_reverb":          f_rv,
                                "fx_normalize":       f_nm,
                            }
                            apply_audio_post_processing(final_path_no_fx, str(wav_final), fx_config)
                        else:
                            shutil.copy2(final_path_no_fx, str(wav_final))

                        _cleanup(wav_tmp)
                        _cleanup(silence_out)
                        success = True
                        break

                except Exception as e:
                    err_msg = str(e)
                    self.log_message.emit(f"  ✗ [#{idx}] Erro na síntese (tentativa {attempt}): {err_msg[:200]}")
                    # CORRIGIDO: erros CUDA são assíncronos no Windows — a exceção aparece
                    # na próxima operação de sync DEPOIS do kernel falhar. Reseta o estado
                    # da GPU aqui para evitar que o erro se propague para o cleanup.
                    if "cuda" in err_msg.lower() or "cudaer" in err_msg.lower() or "accelerator" in err_msg.lower():
                        self.log_message.emit(f"  ⚠ Erro CUDA detectado. Resetando modelo e VRAM...")
                        # CORRIGIDO: força descarga do modelo Chatterbox da VRAM
                        # Sem isso, os ~11GB do modelo ficam "presos" após falha,
                        # causando 'VRAM 0.0GB livre' em execuções subsequentes.
                        try:
                            import engine as _eng_reset
                            _eng_reset.chatterbox_model = None
                            _eng_reset.MODEL_LOADED = False
                            logger.info("Engine Chatterbox resetado após erro CUDA.")
                        except Exception:
                            pass
                        try:
                            if torch.cuda.is_available():
                                gc.collect()
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
                    _cleanup(wav_tmp)

            if success:
                # CORRIGIDO: verificar que o arquivo realmente existe antes de emitir
                # Evita preencher a lista de áudios com paths inválidos
                if wav_final.exists():
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
                    
                    # Usa o tempo exato do gerador deste parágrafo (sem a lentidão acumulada da média geral)
                    paragraph_time = time.time() - paragraph_start_time
                    
                    avg = elapsed / completed if completed > 0 else 0
                    eta = int(avg * (total - completed))
                    eta_s = f"{eta//60}m {eta%60}s" if eta >= 60 else f"{eta}s"
                    self.progress.emit(completed, total)

                    # Auto-detecção de degradação de performance (Cache Leak / Fragmentação)
                    is_degraded = (self._baseline_avg > 0 and paragraph_time > self._baseline_avg * 2.0)
                    is_scheduled_reset = (completed > 0 and completed % 50 == 0)

                    if self._baseline_avg == 0.0 and completed >= 2:
                        self._baseline_avg = paragraph_time

                    if is_degraded or is_scheduled_reset:
                        reason = f"Parágrafo levou {paragraph_time:.1f}s vs Original {self._baseline_avg:.1f}s" if is_degraded else "Reset programado a cada 50 iterações"
                        self.log_message.emit(f"♻️ Manutenção Preventiva VRAM ({reason}). Recarregando modelo...")
                        
                        # Limpa memória alocada do PyTorch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gc.collect()
                            
                        # Reseta a engine TTS para matar memory leaks do Transformer (KV cache preso)
                        if self.tts_engine == "chatterbox":
                            try:
                                import engine as _eng_reset
                                _eng_reset.reload_model()
                            except Exception as e:
                                self.log_message.emit(f"  ⚠ Falha ao recarregar engine: {str(e)[:100]}")
                        
                        # Reseta o baseline para forçar uma nova aferição na próxima rodada limpa
                        self._baseline_avg = 0.0

                    self.log_message.emit(f"  ✓ [#{idx}/{total}] Concluído em {elapsed:.1f}s total. ETA: {eta_s}")
                    self.paragraph_done.emit(idx, str(wav_final), paragraph)
                else:
                    self.log_message.emit(f"  ✗ [#{idx}] Síntese reportou sucesso mas arquivo não encontrado: {wav_final}")
            else:
                self.log_message.emit(f"  ✗ [#{idx}] Falha após {self.max_retries} tentativa(s).")

        self._flush_log_buffer()  # Garante que logs pendentes são enviados

        # CORRIGIDO: liberar cache Kokoro após pipeline (modelos ficam em VRAM entre re-execuções)
        # CORRIGIDO: envolvido em try/except global — se GPU estiver em estado de erro (ex:
        # após cudaErrorLaunchTimeout), operacoes CUDA podem relançar a exceção aqui
        try:
            for _k_model in kokoro_models_cache.values():
                try:
                    del _k_model
                except Exception:
                    pass
            kokoro_models_cache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as _gpu_cleanup_err:
            logger.warning(f"GPU cleanup após pipeline falhou (ignorado): {_gpu_cleanup_err}")

        # Monta resultados finais ordenados
        generated = [generated_map[k] for k in sorted(generated_map.keys())]
        paragrafos_json = [paragrafos_map[k] for k in sorted(paragrafos_map.keys())]

        # Liberar cache CUDA final
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        except Exception:
            pass

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
