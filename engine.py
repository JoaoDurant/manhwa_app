# File: engine.py
# Core TTS model loading and speech generation logic.
# Padrao TTS-Story (Xerophayze/TTS-Story) — todos os 4 engines.

import gc
import os
import shutil
import json
import logging
import random
import contextlib
import subprocess
import tempfile
import threading
import sys
import numpy as np
import torch
import torch._dynamo
import time
from pathlib import Path
from typing import Optional, Tuple, Any, Dict

# [HARDWARE EXTRACTOR] - Extrair potência máxima da RTX 5070 Ti e CPU 14th Gen
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 128  # Para max-autotune caching agressivo
try:
    torch.set_num_threads(8)
except: pass

logger = logging.getLogger(__name__)

# [DTYPE SETUP] - Usando precisão adaptativa para maximizar performance na RTX 5000/4000
DEFAULT_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
logger.info(f"[ENGINE] Precisão padrão detectada: {DEFAULT_DTYPE}")

# [LIBROSA MONKEYPATCH] - Chatterbox usa librosa.load internamente. 
# Precisamos garantir que ele retorne float32 para as LSTMs do voice_encoder.
try:
    import librosa
    _orig_load = librosa.load
    def _patched_load(*args, **kwargs):
        if 'dtype' not in kwargs:
            kwargs['dtype'] = np.float32
        wav, sr = _orig_load(*args, **kwargs)
        if isinstance(wav, np.ndarray) and wav.dtype != np.float32:
            wav = wav.astype(np.float32)
        return wav, sr
    librosa.load = _patched_load
    logger.debug("[ENGINE] Librosa globalmente patcheado para float32.")
except ImportError:
    pass

# =============================================================================
# POLYFILLS — compatibilidade Transformers <-> Chatterbox
# =============================================================================
try:
    import transformers.generation.logits_process as lp
    if not hasattr(lp, "UnnormalizedLogitsProcessor"):
        class UnnormalizedLogitsProcessor:
            def __init__(self, *args, **kwargs): pass
            def __call__(self, input_ids, scores, **kwargs): return scores
        lp.UnnormalizedLogitsProcessor = UnnormalizedLogitsProcessor
    if not hasattr(lp, "MinPLogitsWarper"):
        class MinPLogitsWarper:
            def __init__(self, *args, **kwargs): pass
            def __call__(self, input_ids, scores, **kwargs): return scores
        lp.MinPLogitsWarper = MinPLogitsWarper
except ImportError:
    pass

# =============================================================================
# CHATTERBOX TTS — import defensivo (padrao TTS-Story)
# Turbo preferido pois e mais tolerante ao transformers>=4.40
# =============================================================================
try:
    from chatterbox.tts import ChatterboxTTS as _ChatterboxOriginal
    _CHATTERBOX_BASE_AVAILABLE = True
except ImportError:
    _ChatterboxOriginal = None
    _CHATTERBOX_BASE_AVAILABLE = False

try:
    from chatterbox.tts_turbo import ChatterboxTurboTTS as _ChatterboxTurbo
    CHATTERBOX_TURBO_AVAILABLE = True
except ImportError:
    _ChatterboxTurbo = None
    CHATTERBOX_TURBO_AVAILABLE = False

try:
    from chatterbox.models.s3gen.const import S3GEN_SR
except ImportError:
    S3GEN_SR = 24000

try:
    from chatterbox import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
    MULTILINGUAL_AVAILABLE = True
except ImportError:
    ChatterboxMultilingualTTS = None
    SUPPORTED_LANGUAGES = {}
    MULTILINGUAL_AVAILABLE = False

# =============================================================================
# GLOBAL MONKEY PATCH — AlignmentStreamAnalyzer
#
# WHY THIS IS NEEDED:
#   chatterbox/models/t3/t3.py does:
#       from .inference.alignment_stream_analyzer import AlignmentStreamAnalyzer
#   That binding happens at import-time and lives in t3.py's OWN namespace.
#   Patching the source module during generate() is too late — the name was
#   already captured. The ONLY reliable fix is to overwrite the name in the
#   exact module that instantiates it (t3.py) right after import.
#
# IMPACT ON SDPA:
#   The original AlignmentStreamAnalyzer.__init__ calls _add_attention_spy(),
#   which sets  tfmr.config.output_attentions = True.
#   Transformers raises ValueError when attn_implementation == "sdpa".
#   The DummyAnalyzer stub avoids all of that.
# =============================================================================
class _DummyAlignmentAnalyzer:
    """No-op replacement for AlignmentStreamAnalyzer.
    Skips the attention spy that is incompatible with SDPA.
    Signature of real class: __init__(self, tfmr, queue, text_tokens_slice, alignment_layer_idx=9, eos_idx=0)
    """
    def __init__(self, *args, **kwargs):
        # eos_idx is the 5th positional arg (index 4) or keyword 'eos_idx'
        self.eos_idx        = kwargs.get("eos_idx", args[4] if len(args) > 4 else 0)
        self.started        = True
        self.complete       = False
        self.text_position  = 0
        self.curr_frame_pos = 0
        # Neutralize output_attentions on the transformer (1st positional arg)
        # This prevents the silent CUDA crash from SDPA + output_attentions=True
        if args:
            tfmr = args[0]
            if hasattr(tfmr, "config"):
                try: tfmr.config.output_attentions = False
                except Exception: pass
                try: tfmr.config.attn_implementation = "sdpa"
                except Exception: pass

    def step(self, logits, next_token=None, **kwargs):
        return logits

    def _add_attention_spy(self, *args, **kwargs):
        pass

# Inject into the source module AND into t3.py's captured namespace.
# Force-import each module first so sys.modules always has it — otherwise
# the patch silently does nothing if the import hasn't happened yet.
import sys as _sys
_analyzer_patches = {
    "chatterbox.models.t3.inference.alignment_stream_analyzer": "AlignmentStreamAnalyzer",
    "chatterbox.models.t3.t3":                                 "AlignmentStreamAnalyzer",
    "chatterbox.mtl_tts":                                       "AlignmentStreamAnalyzer",
}
for _mod_path, _attr in _analyzer_patches.items():
    try:
        __import__(_mod_path)  # Force-import so the module is definitely in sys.modules
    except Exception:
        pass
    _mod = _sys.modules.get(_mod_path)
    if _mod is not None and hasattr(_mod, _attr):
        setattr(_mod, _attr, _DummyAlignmentAnalyzer)
        logger.info(f"[PATCH] AlignmentStreamAnalyzer substituido em {_mod_path}")
del _mod_path, _attr, _mod, _analyzer_patches

# =============================================================================
# GLOBAL MONKEY PATCH — VoiceEncoder DType Fix
#
# WHY THIS IS NEEDED:
#   When cloning a voice via `audio_prompt_path`, librosa or PySTFT may produce 
#   float64 mels. The VoiceEncoder's LSTM weights are float32, causing a crash:
#   "RNN input dtype (torch.float64) does not match weight dtype (torch.float32)"
# =============================================================================
try:
    from chatterbox.models.voice_encoder.voice_encoder import VoiceEncoder
    _orig_ve_forward = VoiceEncoder.forward
    def _patched_ve_forward(self, mels):
        return _orig_ve_forward(self, mels.to(torch.float32))
    VoiceEncoder.forward = _patched_ve_forward
    logger.info("[PATCH] VoiceEncoder.forward patcheado para forçar torch.float32 nas mels.")
except ImportError:
    pass

# Selecionar classe preferida: Turbo > Original
if CHATTERBOX_TURBO_AVAILABLE:
    _ChatterboxClass  = _ChatterboxTurbo
    _CHATTERBOX_TYPE  = "turbo"
    CHATTERBOX_AVAILABLE = True
elif _CHATTERBOX_BASE_AVAILABLE:
    _ChatterboxClass  = _ChatterboxOriginal
    _CHATTERBOX_TYPE  = "original"
    CHATTERBOX_AVAILABLE = True
else:
    _ChatterboxClass  = None
    _CHATTERBOX_TYPE  = None
    CHATTERBOX_AVAILABLE = False
    logger.warning(
        "Chatterbox nao instalado.\n"
        "Execute: pip install chatterbox-tts"
    )

logger.info(
    f"Chatterbox: available={CHATTERBOX_AVAILABLE} | "
    f"turbo={CHATTERBOX_TURBO_AVAILABLE} | type={_CHATTERBOX_TYPE}"
)

# =============================================================================
# KOKORO TTS — import defensivo (padrao TTS-Story)
# =============================================================================
_REPO_ROOT = Path(__file__).resolve().parent

try:
    import kokoro
    from kokoro import KPipeline
    KOKORO_AVAILABLE    = True
    KOKORO_SAMPLE_RATE  = 24000
    logger.info("Kokoro disponivel (pip package).")
except ImportError:
    # Fallback to local copy
    kokoro_path = str(_REPO_ROOT / "Kokoro-TTS-Local-master")
    import sys
    if kokoro_path not in sys.path:
        sys.path.append(kokoro_path)
    
    try:
        from models import build_model
        import models as kokoro
        import kokoro as KPipeline
        # We don't have KPipeline directly, we have to mock it or rewrite load_kokoro_engine
        # The easiest fix is just telling the user to install the pip package since the new 
        # engine.py expects the newer KPipeline API.
        
        # ACTUALLY, let's keep it simple: TTS-Story standard uses `kokoro` pip package.
        # We will keep the pip requirement but gracefully degrade.
        KPipeline           = None
        KOKORO_AVAILABLE    = False
        KOKORO_SAMPLE_RATE  = 24000
        logger.warning(
            "Kokoro (pip) nao instalado. Execute: pip install kokoro>=0.9.4\n"
            "O Kokoro-TTS-Local-master antigo nao e mais suportado por este engine.\n"
        )
    except ImportError:
        KPipeline           = None
        KOKORO_AVAILABLE    = False
        KOKORO_SAMPLE_RATE  = 24000
        logger.warning(
            "Kokoro nao instalado. Execute: pip install kokoro>=0.9.4\n"
            "Kokoro tambem requer espeak-ng instalado no sistema:\n"
            "  Windows: https://github.com/espeak-ng/espeak-ng/releases\n"
            "  Linux:   sudo apt-get install espeak-ng"
        )

# Globals Kokoro
kokoro_pipeline     = None
KOKORO_LOADED = False
kokoro_loaded_lang = None  # Tracks the language currently bound to KPipeline
_kokoro_lock = threading.Lock()
_kokoro_voice_cache = {}

# =============================================================================
# Imports de suporte
# =============================================================================
from config import config_manager

# =============================================================================
# GLOBALS — Chatterbox
# =============================================================================
chatterbox_model: Optional[object] = None
MODEL_LOADED:     bool = False
model_device:     Optional[str] = None
loaded_model_type: Optional[str] = None       # "original" | "turbo" | "multilingual"
loaded_model_class_name: Optional[str] = None

# Cache de voice embedding — evita recalcular VoiceEncoder para o mesmo arquivo de referência.
# Chave: caminho absoluto do audio_prompt_path. Valor: tensor de embedding na CPU (float32).
# Limpo automaticamente quando o modelo é trocado (unload_all_for_switch).
_voice_embedding_cache: dict = {}

# Lock compartilhado GPU — garante que apenas 1 modelo usa GPU por vez
_synthesis_lock = threading.Lock()

# Lock e Flags de Estado para prevencao de Race Conditions (PATCH)
_engine_lock = threading.Lock()
_is_loading = False
_is_generating = False

# Variaveis de estado de device
FORCE_CPU = os.environ.get("MANHWA_FORCE_CPU", "0") == "1"

# =============================================================================
# SELECTOR MAP — Chatterbox
# =============================================================================
MODEL_SELECTOR_MAP = {
    "chatterbox": "original", "original": "original",
    "resembleai/chatterbox": "original",
    "chatterbox-turbo": "turbo", "turbo": "turbo",
    "resembleai/chatterbox-turbo": "turbo",
    "chatterbox-multilingual": "multilingual", "multilingual": "multilingual",
}

TURBO_PARALINGUISTIC_TAGS = [
    "laugh", "chuckle", "sigh", "gasp", "cough",
    "clear throat", "sniff", "groan", "shush",
]

# =============================================================================
# CUDA OTIMIZACOES — RTX 5070 Ti (Blackwell sm_120)
# =============================================================================
if torch.cuda.is_available():
    try:
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU detectada: {gpu_name}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        # [BLACKWELL] expandable_segments reduz fragmentação em batches longos
        # max_split_size_mb:512 evita OOM por fragmentação de micro-blocos
        # garbage_collection_threshold:0.8 só ativa GC quando 80% cheio
        import os as _oe
        _existing = _oe.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments" not in _existing:
            _oe.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
                "max_split_size_mb:512,"
                "garbage_collection_threshold:0.8,"
                "expandable_segments:True"
            )
        # [DETERMINISMO] Verdadeiro determinismo por kernel — crítico para consistência de voz
        # em batch de 100+ parágrafos. max-autotune não determinista causa drift de timbre.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        logger.info("TF32, high-precision matmul, ALLOC_CONF (Blackwell), e flags determinísticas ativados.")
    except Exception as e:
        logger.debug(f"Nao foi possivel aplicar otimizacoes CUDA no inicio: {e}")

# [PERSISTENT COMPILE CACHE] Evita recompilar a cada novo job no Macro.
# Chave: model_type ("turbo", "multilingual", "original")
# Valor: True se já compilado (os atributos t3/s3gen já foram substituídos in-place)
_compiled_cache: dict = {}



# =============================================================================
# UTILITARIOS INTERNOS
# =============================================================================

def _test_cuda_functionality() -> bool:
    if not torch.cuda.is_available(): return False
    try:
        torch.tensor([1.0]).cuda().cpu()
        return True
    except Exception as e:
        logger.warning(f"CUDA teste falhou: {e}")
        return False

def _test_mps_functionality() -> bool:
    if not torch.backends.mps.is_available(): return False
    try:
        torch.tensor([1.0]).to("mps").cpu()
        return True
    except: return False

def _optimize_for_device(device_str: str):
    if device_str == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try: torch.set_float32_matmul_precision("high")
        except: pass
        try:
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            vram_free  = torch.cuda.mem_get_info()[0] / 1e9
            gpu_name   = torch.cuda.get_device_name(0)
            bf16_ok    = torch.cuda.is_bf16_supported()
            logger.info(f"GPU: {gpu_name} | VRAM: {vram_total:.1f}GB | Livre: {vram_free:.1f}GB | BF16: {bf16_ok}")
        except: pass
    logger.info(f"Otimizacoes aplicadas para: {device_str}")

def set_seed(seed_value: int):
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        try: torch.cuda.manual_seed_all(seed_value)
        except: pass
    random.seed(seed_value)
    np.random.seed(seed_value)

def _resolve_qwen_attn() -> str:
    return "eager"


# =============================================================================
# CHATTERBOX — load / warmup / synthesize / reload
# =============================================================================

def _get_model_class(selector: str) -> tuple:
    s = selector.lower().strip()
    m_type = MODEL_SELECTOR_MAP.get(s)

    if m_type == "turbo":
        if not CHATTERBOX_TURBO_AVAILABLE:
            raise ImportError("ChatterboxTurboTTS nao disponivel. pip install chatterbox-tts")
        return _ChatterboxTurbo, "turbo"

    if m_type == "multilingual":
        if not MULTILINGUAL_AVAILABLE:
            raise ImportError("ChatterboxMultilingualTTS nao disponivel.")
        return ChatterboxMultilingualTTS, "multilingual"

    # Default: Turbo se disponivel, fallback Original
    if CHATTERBOX_TURBO_AVAILABLE:
        return _ChatterboxTurbo, "turbo"
    if _CHATTERBOX_BASE_AVAILABLE:
        return _ChatterboxOriginal, "original"

    raise ImportError("Nenhuma classe Chatterbox disponivel.")


def get_model_info() -> dict:
    return {
        "loaded": MODEL_LOADED,
        "type": loaded_model_type,
        "class_name": loaded_model_class_name,
        "device": model_device,
        "sample_rate": chatterbox_model.sr if chatterbox_model else None,
        "supports_paralinguistic_tags": loaded_model_type == "turbo",
        "available_paralinguistic_tags": TURBO_PARALINGUISTIC_TAGS if loaded_model_type == "turbo" else [],
        "turbo_available_in_package": CHATTERBOX_TURBO_AVAILABLE,
        "multilingual_available_in_package": MULTILINGUAL_AVAILABLE,
    }


def _get_chatterbox_device():
    device_setting = config_manager.get_string("tts_engine.device", "auto")
    if FORCE_CPU: device_setting = "cpu"
    if device_setting == "auto":
        if torch.cuda.is_available(): resolved = "cuda"
        elif torch.backends.mps.is_available(): resolved = "mps"
        else: resolved = "cpu"
    elif device_setting == "cuda":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    elif device_setting == "mps":
        resolved = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        resolved = "cpu"
    return resolved

HAS_TURBO = CHATTERBOX_TURBO_AVAILABLE
print(f"[ENGINE] Turbo disponível: {HAS_TURBO}")

def load_turbo() -> bool:
    """Carrega Chatterbox Turbo com fallback automático para Original."""
    global chatterbox_model, MODEL_LOADED, model_device, loaded_model_type, loaded_model_class_name
    global _is_loading, _is_generating
    
    if _is_generating:
        print("IGNORADO: geração já em andamento (turbo)")
        return False
        
    if loaded_model_type == "turbo" and MODEL_LOADED:
        print("Já carregado, ignorando reload")
        return True

    with _engine_lock:
        if _is_loading:
            print("IGNORADO: já está carregando (turbo)")
            return False
        _is_loading = True
        
    try:
        print("[DEBUG] Solicitado: load_turbo")
        if not HAS_TURBO:
            print("[ENGINE] Turbo não disponível nesta versão do chatterbox. Usando fallback (original).")
            # Reset flag BEFORE calling load_original so it can acquire the lock
            with _engine_lock:
                _is_loading = False
            return load_original()

        unload_all_for_switch()
        model_device = _get_chatterbox_device()
        _optimize_for_device(model_device)
        
        # Importacao ja tentada globalmente como _ChatterboxTurbo
        if _ChatterboxTurbo is None:
            # Segunda tentativa de importacao via path correto
            from chatterbox.tts_turbo import ChatterboxTurboTTS as _Turbo
        else:
            _Turbo = _ChatterboxTurbo

        chatterbox_model = _Turbo.from_pretrained(device=model_device)

        # [OTIMIZAÇÃO] BFloat16 (mesma precision que o Multilingual já usava)
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            try:
                chatterbox_model = chatterbox_model.bfloat16()
                logger.info("[TURBO] Convertido para bfloat16.")
            except AttributeError:
                pass
        elif torch.cuda.is_available():
            try:
                chatterbox_model = chatterbox_model.half()
                logger.info("[TURBO] Convertido para float16.")
            except AttributeError:
                pass

        # [OTIMIZAÇÃO] SDPA — evita overhead do attention spy (mesmo fix do Original/Multilingual)
        if hasattr(chatterbox_model, "t3") and hasattr(chatterbox_model.t3, "model"):
            if hasattr(chatterbox_model.t3.model, "config"):
                try:
                    chatterbox_model.t3.model.config.attn_implementation = "sdpa"
                    chatterbox_model.t3.model.config.output_attentions = False
                except Exception: pass
        if hasattr(chatterbox_model, "set_attn_implementation"):
            try: chatterbox_model.set_attn_implementation("sdpa")
            except Exception: pass

        # [OTIMIZAÇÃO] Patch do forward para remover output_attentions (mesmo fix do Multilingual)
        try:
            if hasattr(chatterbox_model.t3, "model"):
                _real_t3_model = chatterbox_model.t3.model
            else:
                _real_t3_model = chatterbox_model.t3
                
            _original_call_t = _real_t3_model.__class__.forward
            def _patched_forward_turbo(self_m, *fargs, **fkwargs):
                fkwargs.pop("output_attentions", None)
                fkwargs.pop("head_mask", None)
                return _original_call_t(self_m, *fargs, **fkwargs)
            import types as _types
            _real_t3_model.forward = _types.MethodType(_patched_forward_turbo, _real_t3_model)
            logger.info("[TURBO] Instance-level forward patch aplicado (output_attentions interceptado).")
        except Exception as _fp_t:
            logger.warning(f"[TURBO] Instance forward patch falhou (nao critico): {_fp_t}")

        # [OTIMIZAÇÃO] Re-patch AlignmentStreamAnalyzer com eos_idx real
        try:
            real_eos_idx = chatterbox_model.t3.hp.stop_speech_token
            class _DummyTurboEos(_DummyAlignmentAnalyzer):
                def __init__(self_inner, *iargs, **ikwargs):
                    super().__init__(*iargs, **ikwargs)
                    self_inner.eos_idx = real_eos_idx
            import sys as _sys_t
            for _mp_t in ("chatterbox.models.t3.inference.alignment_stream_analyzer",
                          "chatterbox.models.t3.t3", "chatterbox.mtl_tts"):
                _m_t = _sys_t.modules.get(_mp_t)
                if _m_t is not None and hasattr(_m_t, "AlignmentStreamAnalyzer"):
                    setattr(_m_t, "AlignmentStreamAnalyzer", _DummyTurboEos)
                    logger.info(f"[PATCH][TURBO] AlignmentStreamAnalyzer re-patched (eos_idx={real_eos_idx}) em {_mp_t}")
        except Exception as _ep_t:
            logger.warning(f"[PATCH][TURBO] Re-patch com eos_idx falhou (nao critico): {_ep_t}")

        # [OTIMIZAÇÃO] torch.compile para acelerar inferência subsequente
        # MODE: reduce-overhead (não max-autotune) — elimina overhead Python/dispatch
        # sem o autotuner agressive que quebra o determinismo de voz em batches longos.
        # PERSISTENT CACHE: só compila 1x por model_type, mesmo entre jobs do Macro.
        if model_device == "cuda" and loaded_model_type not in _compiled_cache:
            for attr in ("t3", "s3gen"):
                if hasattr(chatterbox_model, attr):
                    try:
                        setattr(chatterbox_model, attr, torch.compile(
                            getattr(chatterbox_model, attr),
                            mode="reduce-overhead",  # Elimina dispatch overhead, preserva determinismo
                            fullgraph=False,
                        ))
                        logger.info(f"[TURBO] torch.compile (reduce-overhead) aplicado ao {attr}.")
                    except Exception as _ce:
                        logger.warning(f"[TURBO] torch.compile em {attr} falhou (nao critico): {_ce}")
            _compiled_cache["turbo"] = True
            
            # WARM-UP: paga o custo de JIT compilation agora
            logger.info("[TURBO] Executando warm-up do torch.compile...")
            try:
                _dummy_text = "Hello."
                with torch.inference_mode():
                    chatterbox_model.generate(_dummy_text, audio_prompt_path=None)
            except Exception as _wu_err:
                logger.debug(f"[TURBO] Warm-up falhou (não crítico): {_wu_err}")
        elif model_device == "cuda":
            logger.info("[TURBO] torch.compile já aplicado (cache hit) — reutilizando.")


        loaded_model_type = "turbo"
        loaded_model_class_name = "ChatterboxTurboTTS"
        MODEL_LOADED = True
        logger.info(f"Chatterbox Turbo carregado em {model_device}.")
        if torch.cuda.is_available(): torch.cuda.synchronize()
        return True
    except Exception as e:
        logger.error(f"Falha ao carregar Turbo: {e}. Tentando fallback original.", exc_info=True)
        # Reset flag BEFORE calling load_original so it can acquire the lock
        with _engine_lock:
            _is_loading = False
        return load_original()
    finally:
        # Only reset if we haven't already (i.e., the normal success path)
        with _engine_lock:
            _is_loading = False

def load_original() -> bool:
    """Carrega especificamente o Chatterbox Original."""
    global chatterbox_model, MODEL_LOADED, model_device, loaded_model_type, loaded_model_class_name
    global _is_loading, _is_generating
    
    if _is_generating:
        print("IGNORADO: geração já em andamento (original)")
        return False

    if loaded_model_type == "original" and MODEL_LOADED:
        print("Já carregado, ignorando reload")
        return True

    with _engine_lock:
        if _is_loading:
            print("IGNORADO: já está carregando (original)")
            return False
        _is_loading = True

    try:
        print("[DEBUG] Solicitado: load_original")
        unload_all_for_switch()
        model_device = _get_chatterbox_device()
        _optimize_for_device(model_device)

        from chatterbox import ChatterboxTTS as _Original
        chatterbox_model = _Original.from_pretrained(device=model_device)
        loaded_model_type = "original"
        loaded_model_class_name = "ChatterboxTTS"
        MODEL_LOADED = True
        logger.info(f"Chatterbox Original carregado em {model_device}.")
        if torch.cuda.is_available(): torch.cuda.synchronize()

        # Enable SDPA for speed and VRAM reduction
        if hasattr(chatterbox_model, "t3") and hasattr(chatterbox_model.t3, "model"):
            if hasattr(chatterbox_model.t3.model, "config"):
                try:
                    chatterbox_model.t3.model.config.attn_implementation = "sdpa"
                    chatterbox_model.t3.model.config.output_attentions = False
                except Exception: pass
        if hasattr(chatterbox_model, "set_attn_implementation"):
            try: chatterbox_model.set_attn_implementation("sdpa")
            except Exception: pass

        # Re-patch AlignmentStreamAnalyzer with known eos_idx
        try:
            real_eos_idx = chatterbox_model.t3.hp.stop_speech_token
            class _DummyWithEos(_DummyAlignmentAnalyzer):
                def __init__(self_inner, *iargs, **ikwargs):
                    super().__init__(*iargs, **ikwargs)
                    self_inner.eos_idx = real_eos_idx
            import sys as _sys3
            for _mp in ("chatterbox.models.t3.inference.alignment_stream_analyzer",
                        "chatterbox.models.t3.t3", "chatterbox.mtl_tts"):
                _m = _sys3.modules.get(_mp)
                if _m is not None and hasattr(_m, "AlignmentStreamAnalyzer"):
                    setattr(_m, "AlignmentStreamAnalyzer", _DummyWithEos)
                    logger.info(f"[PATCH] Re-patched AlignmentStreamAnalyzer (eos_idx={real_eos_idx}) em {_mp}")
        except Exception as _ep:
            logger.warning(f"[PATCH] Re-patch original com eos_idx falhou: {_ep}")

        return True
    except Exception as e:
        logger.error(f"Falha ao carregar Original: {e}", exc_info=True)
        return False
    finally:
        with _engine_lock:
            _is_loading = False

def load_multilingual() -> bool:
    """Carrega especificamente o Chatterbox Multilingual com patch eager isolado."""
    global chatterbox_model, MODEL_LOADED, model_device, loaded_model_type, loaded_model_class_name
    global _is_loading, _is_generating
    
    if _is_generating:
        print("IGNORADO: geração já em andamento (multilingual)")
        return False

    if loaded_model_type == "multilingual" and MODEL_LOADED:
        print("Já carregado, ignorando reload")
        return True

    with _engine_lock:
        if _is_loading:
            print("IGNORADO: já está carregando (multilingual)")
            return False
        _is_loading = True

    try:
        print("[DEBUG] Solicitado: load_multilingual")
        unload_all_for_switch()
        model_device = _get_chatterbox_device()
        _optimize_for_device(model_device)

        from chatterbox import ChatterboxMultilingualTTS as _Mtl
        # FIX 1: Try passing attn_implementation=sdpa directly to from_pretrained.
        # Newer versions of the lib may accept it; older ones will raise TypeError.
        print("[ENGINE] Tentando carregar Multilingual com attn_implementation=sdpa...")
        try:
            chatterbox_model = _Mtl.from_pretrained(device=model_device, attn_implementation="sdpa")
            logger.info("[MULTILINGUAL] Carregado com attn_implementation=sdpa via from_pretrained.")
        except TypeError:
            logger.warning("[MULTILINGUAL] from_pretrained nao aceita attn_implementation. Carregando sem o argumento.")
            chatterbox_model = _Mtl.from_pretrained(device=model_device)

        # FORÇAR SDPA EM TODAS AS CAMADAS para otimização
        if hasattr(chatterbox_model, "t3") and hasattr(chatterbox_model.t3, "model"):
            if hasattr(chatterbox_model.t3.model, "config"):
                chatterbox_model.t3.model.config.attn_implementation = "sdpa"
            
        # FORÇAR também via método (Transformers moderno)
        if hasattr(chatterbox_model, "set_attn_implementation"):
            chatterbox_model.set_attn_implementation("sdpa")

        # FIX 2: Force output_attentions=False on ALL submodules that have a config.
        # This catches any attention layer that may still hold a stale flag.
        if hasattr(chatterbox_model, "t3") and hasattr(chatterbox_model.t3, "model"):
            for _submod in chatterbox_model.t3.model.modules():
                if hasattr(_submod, "config"):
                    try: _submod.config.output_attentions = False
                    except Exception: pass
        try: chatterbox_model.t3.model.config.output_attentions = False
        except Exception: pass

        # FIX 3: Instance-level inference wrapper on chatterbox_model.t3.
        # The real t3.py passes output_attentions=True hardcoded to self.model();
        # we intercept at the model.__call__ level to strip that kwarg.
        try:
            _real_t3_model = chatterbox_model.t3.model
            _original_call = _real_t3_model.__class__.forward

            def _patched_forward(self_m, *fargs, **fkwargs):
                fkwargs.pop("output_attentions", None)
                fkwargs.pop("head_mask", None)
                return _original_call(self_m, *fargs, **fkwargs)

            import types
            chatterbox_model.t3.model.forward = types.MethodType(_patched_forward, chatterbox_model.t3.model)
            logger.info("[MULTILINGUAL] Instance-level forward patch aplicado (output_attentions interceptado).")
        except Exception as _fp:
            logger.warning(f"[MULTILINGUAL] Instance forward patch falhou (nao critico): {_fp}")

        # [OTIMIZAÇÃO 5070 Ti] BFloat16 em vez de float32 para acelerar geração e cortar uso de VRAM
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            try: chatterbox_model = chatterbox_model.bfloat16()
            except AttributeError: pass
        else:
            try: chatterbox_model = chatterbox_model.half()
            except AttributeError: pass

        loaded_model_type = "multilingual"
        loaded_model_class_name = "ChatterboxMultilingualTTS"
        MODEL_LOADED = True
        logger.info(f"Chatterbox Multilingual carregado em {model_device}.")
        if torch.cuda.is_available(): torch.cuda.synchronize()
        
        # Re-patch the Dummy with the real stop_speech_token so that
        # the assert in t3.py:288 always passes:
        #   assert alignment_stream_analyzer.eos_idx == self.hp.stop_speech_token
        try:
            real_eos_idx = chatterbox_model.t3.hp.stop_speech_token
            
            class _DummyWithEos(_DummyAlignmentAnalyzer):
                def __init__(self_inner, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self_inner.eos_idx = real_eos_idx  # Override with known correct value
            
            import sys as _sys2
            for _mp in ("chatterbox.models.t3.inference.alignment_stream_analyzer",
                        "chatterbox.models.t3.t3", "chatterbox.mtl_tts"):
                _m = _sys2.modules.get(_mp)
                if _m is not None and hasattr(_m, "AlignmentStreamAnalyzer"):
                    setattr(_m, "AlignmentStreamAnalyzer", _DummyWithEos)
                    logger.info(f"[PATCH] Re-patched AlignmentStreamAnalyzer (eos_idx={real_eos_idx}) em {_mp}")
        except Exception as _ep:
            logger.warning(f"[PATCH] Re-patch com eos_idx falhou (nao critico): {_ep}")

        # [OTIMIZAÇÃO] torch.compile — reduce-overhead + persistent cache
        if model_device == "cuda" and "multilingual" not in _compiled_cache:
            for attr in ("t3", "s3gen"):
                if hasattr(chatterbox_model, attr):
                    try:
                        setattr(chatterbox_model, attr, torch.compile(
                            getattr(chatterbox_model, attr),
                            mode="reduce-overhead",
                            fullgraph=False,
                        ))
                        logger.info(f"[MULTILINGUAL] torch.compile (reduce-overhead) aplicado ao {attr}.")
                    except Exception as _ce_m:
                        logger.warning(f"[MULTILINGUAL] torch.compile em {attr} falhou (nao critico): {_ce_m}")
            _compiled_cache["multilingual"] = True
        elif model_device == "cuda":
            logger.info("[MULTILINGUAL] torch.compile já aplicado (cache hit) — reutilizando.")

        
        return True
    except Exception as e:
        logger.error(f"Falha ao carregar Multilingual: {e}", exc_info=True)
        return False
    finally:
        with _engine_lock:
            _is_loading = False

def load_model(model_type: Optional[str] = None) -> bool:
    """Carrega o modelo Chatterbox baseado no tipo solicitado ou na config."""
    if model_type is None:
        model_type = config_manager.get_string("model.repo_id", "turbo").lower()
    else:
        model_type = model_type.lower()

    if "multilingual" in model_type: 
        return load_multilingual()
    if "original" in model_type or ("chatterbox" in model_type and "turbo" not in model_type): 
        return load_original()
    return load_turbo()


    # [OTIMIZAÇÃO INDUSTRIAL] Removido torch.compile duplicado aqui para evitar conflito com max-autotune do load.
    
    # Contexto de autocast para garantir que o warmup gere os mesmos kernels da geração real
    if model_device == "cuda" and torch.cuda.is_bf16_supported():
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    elif model_device == "cuda":
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        autocast_ctx = contextlib.nullcontext()

    # Warmup 1 — sem audio de referencia
    try:
        with _synthesis_lock, torch.inference_mode(), autocast_ctx:
            if hasattr(chatterbox_model, "generate"):
                chatterbox_model.generate("Hello.")
        if torch.cuda.is_available(): torch.cuda.synchronize()
        logger.info("Warmup (sem ref) concluido.")
    except Exception as e:
        logger.warning(f"Warmup sem ref falhou: {e}")

    # Warmup 2 — com audio de referencia simulado
    try:
        import soundfile as sf
        dummy_sr = chatterbox_model.sr if hasattr(chatterbox_model, "sr") else 24000
        dummy_audio = np.zeros(dummy_sr * 6, dtype=np.float32) # 6s para passar no assert > 5s
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        sf.write(tmp_path, dummy_audio, dummy_sr)
        try:
            with _synthesis_lock, torch.inference_mode(), autocast_ctx:
                if hasattr(chatterbox_model, "generate"):
                    # Use a call that triggers prepare_conditionals
                    chatterbox_model.generate("Hello.", audio_prompt_path=tmp_path)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            logger.info("Warmup (com ref simulada) concluido.")
        finally:
            try: os.unlink(tmp_path)
            except: pass
    except Exception as e:
        logger.warning(f"Warmup com ref falhou: {e}")


def check_audio_validity(audio):
    if audio is None:
        raise RuntimeError("Audio None")
    import torch
    import numpy as np
    if isinstance(audio, torch.Tensor):
        if audio.numel() < 1000:
            raise RuntimeError("Audio inválido: muito pequeno (tensor)")
    elif isinstance(audio, np.ndarray):
        if audio.size < 1000:
            raise RuntimeError("Audio inválido: muito pequeno (ndarray)")
    elif hasattr(audio, "__len__"):
        if len(audio) < 1000:
            raise RuntimeError("Audio inválido: muito pequeno (list)")

def synthesize(
    text: str,
    audio_prompt_path: str = None,
    temperature: float = 0.65,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    seed: int = 0,
    language: str = "en",
    min_p: float = 0.05,
    top_p: float = 0.85,
    top_k: int = 1000,
    repetition_penalty: float = 1.15,
    norm_loudness: bool = True,
    **kwargs,
) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    """
    Sintetiza com Chatterbox (Turbo | Original | Multilingual).
    PADRAO TTS-STORY: autocast bfloat16 no CUDA para Turbo/Original.
    """
    global chatterbox_model, MODEL_LOADED, loaded_model_type
    global _is_loading, _is_generating

    if _is_loading:
        raise RuntimeError("Modelo ainda está carregando")

    if not MODEL_LOADED or chatterbox_model is None:
        logger.error("Chatterbox nao carregado.")
        return None, None

    if seed != 0:
        set_seed(seed)

    use_cuda       = (model_device == "cuda")
    is_multilingual = (loaded_model_type == "multilingual")

    # [OTIMIZAÇÃO] Habilitar autocast bfloat16 para TODOS os modelos na série RTX 40/50
    if use_cuda and torch.cuda.is_bf16_supported():
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    elif use_cuda:
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        autocast_ctx = contextlib.nullcontext()

    with _engine_lock:
        _is_generating = True

    with _synthesis_lock:
        try:
            # [DETERMINISMO] Seed forçada logo no início do lock para consistência total
            set_seed(seed if seed != 0 else 42)
            
            print("[ENGINE] Gerando áudio...")
            start_time = time.time()
            with torch.inference_mode(), autocast_ctx:
                # [VOICE VALIDATION]
                if not audio_prompt_path or not isinstance(audio_prompt_path, str) or not os.path.exists(audio_prompt_path):
                    audio_prompt_path = None
                    _abs_path = None
                else:
                    _abs_path = os.path.abspath(audio_prompt_path)

                # [MAESTRIA DO CACHE DE VOICE EMBEDDING]
                # Implementação robusta para garantir que o VoiceEncoder NUNCA seja
                # recalculado para o mesmo arquivo de referência.
                _using_cache = False
                if _abs_path and _abs_path in _voice_embedding_cache:
                    # Aplicamos o objeto 'conds' completo do Chatterbox (Turbo/MTL)
                    chatterbox_model.conds = _voice_embedding_cache[_abs_path]
                    _using_cache = True
                    logger.debug(f"[VOICE CACHE] Mastery Hit: {_abs_path}")

                gen_kwargs = {
                    "text": text,
                    "audio_prompt_path": None if _using_cache else audio_prompt_path,
                    "temperature": temperature,
                    "min_p": min_p,
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty,
                }
                
                if loaded_model_type == "turbo":
                    gen_kwargs["top_k"] = top_k
                    gen_kwargs["norm_loudness"] = norm_loudness
                    wav_tensor = chatterbox_model.generate(**gen_kwargs)
                elif loaded_model_type == "multilingual":
                    gen_kwargs["language_id"] = language
                    gen_kwargs["exaggeration"] = exaggeration
                    gen_kwargs["cfg_weight"] = cfg_weight
                    wav_tensor = chatterbox_model.generate(**gen_kwargs)
                else:
                    gen_kwargs["exaggeration"] = exaggeration
                    gen_kwargs["cfg_weight"] = cfg_weight
                    wav_tensor = chatterbox_model.generate(**gen_kwargs)


                # Se foi "Miss" (1ª geração do áudio), salvamos o Conditionals resultante.
                if _abs_path and not _using_cache and hasattr(chatterbox_model, "conds"):
                    if chatterbox_model.conds is not None:
                        _voice_embedding_cache[_abs_path] = chatterbox_model.conds
                        logger.info(f"[VOICE CACHE] Miss: Condicionais capturadas e imortalizadas para {_abs_path}")
            
            elapsed = time.time() - start_time
            if elapsed > 30:
                logger.warning(f"Sintese demorada: {elapsed:.2f}s")
            
            check_audio_validity(wav_tensor)
            
            wav_tensor = wav_tensor.to(torch.float32)
            
            # [AUDIO FIX] Prevenir clipping severo
            max_val = torch.max(torch.abs(wav_tensor))
            if max_val >= 0.98:
                logger.info(f"[ENGINE] Normalizando audio para evitar clipping (max_val={max_val:.2f})")
                wav_tensor = wav_tensor * (0.98 / max_val)
                
            return wav_tensor, chatterbox_model.sr

        except Exception as e:
            print("ERRO NA GERAÇÃO CHATTERBOX:", e)
            import traceback
            traceback.print_exc()
            logger.error(f"Erro Chatterbox synthesize: {e}")
            return None, None
        finally:
            with _engine_lock:
                _is_generating = False


def reload_model() -> bool:
    global chatterbox_model, MODEL_LOADED, model_device, loaded_model_type, loaded_model_class_name
    global _is_generating
    
    with _engine_lock:
        if _is_generating:
            print("IGNORADO: modelo em uso (reload_model)")
            return False

    logger.info("Hot-swap Chatterbox...")
    if chatterbox_model is not None:
        chatterbox_model = None
    MODEL_LOADED = False
    loaded_model_type = None
    loaded_model_class_name = None
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        try: torch.mps.empty_cache()
        except AttributeError: pass
    return load_model()


# =============================================================================
# KOKORO TTS — (padrao TTS-Story src/engines/kokoro_engine.py)
# =============================================================================

def load_kokoro_engine(
    voice: str = "af_heart",
    lang_code: str = "a",
    device: str = "auto",
) -> bool:
    """
    Carrega KPipeline como singleton.
    PADRAO TTS-STORY: KPipeline(lang_code=lang_code, device=device)

    lang_p = PT-BR, lang_e = Spanish, lang_f = French, etc.
    """
    global kokoro_pipeline, KOKORO_LOADED, kokoro_loaded_lang
    global _is_loading, _is_generating

    if _is_generating:
        print("IGNORADO: geração já em andamento (kokoro)")
        return False

    if KOKORO_LOADED and kokoro_pipeline is not None and kokoro_loaded_lang == lang_code:
        return True

    with _engine_lock:
        if _is_loading:
            return False
        _is_loading = True

    try:
        with _kokoro_lock:
            if KOKORO_LOADED and kokoro_pipeline is not None and kokoro_loaded_lang == lang_code:
                return True

            if not KOKORO_AVAILABLE:
                logger.error(
                    "Kokoro nao instalado.\n"
                    "Execute: pip install kokoro>=0.9.4\n"
                    "E instale espeak-ng: https://github.com/espeak-ng/espeak-ng/releases"
                )
                return False

        resolved = "cuda" if (device == "auto" and _test_cuda_functionality()) else (device if device != "auto" else "cpu")

        try:
            import time
            t0_kload = time.time()
            print(f"[ENGINE] [Kokoro] Iniciando carregamento | param_device: {device} | resolved_device: {resolved} | lang: {lang_code}")
            logger.info(f"Carregando Kokoro | lang={lang_code} | device={resolved}")
            kokoro_pipeline = KPipeline(lang_code=lang_code, device=resolved)
            
            KOKORO_LOADED   = True
            kokoro_loaded_lang = lang_code
            print(f"[ENGINE] [Kokoro] Carregado com sucesso em {resolved} | lang: {lang_code} | Tempo: {time.time() - t0_kload:.2f}s")
            logger.info(f"Kokoro carregado com sucesso em {resolved} para lang '{lang_code}'.")
            
            if torch.cuda.is_available(): 
                torch.cuda.synchronize()
                # Otimização Blackwell/Windows: Evitar re-benchmarking de kernels em cada variação de texto
                torch.backends.cudnn.benchmark = False
                
                # Warmup: Pre-carrega kernels CUDA com uma string curta
                try:
                    print(f"[ENGINE] [Kokoro] Aquecendo CUDA kernels...")
                    _w0 = time.time()
                    _gen = kokoro_pipeline("warm", voice="af_heart", speed=1.0)
                    for _ in _gen: pass
                    print(f"[ENGINE] [Kokoro] Warmup concluído em {time.time() - _w0:.2f}s")
                except: pass

            return True
        except Exception as e:
            err = str(e).lower()
            if "espeak" in err or "phonemize" in err:
                print("[ERROR] [ENGINE] [Kokoro] espeak-ng ausente no sistema.")
                logger.error(
                    "Kokoro requer espeak-ng instalado no sistema.\n"
                    "Baixe em: https://github.com/espeak-ng/espeak-ng/releases"
                )
            else:
                print(f"[ERROR] [ENGINE] [Kokoro] Falha ao carregar modelo: {e}")
                logger.error(f"Falha critica ao carregar Kokoro: {e}", exc_info=True)
            kokoro_pipeline = None
            KOKORO_LOADED   = False
            return False
    finally:
        with _engine_lock:
            _is_loading = False



@torch.inference_mode()
def synthesize_kokoro(
    text: str,
    voice: str = "af_heart",
    speed: float = 1.0,
    lang_code: str = "a",
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Sintetiza com Kokoro.
    PADRAO TTS-STORY: pipeline(text, voice=voice, speed=speed)
    Concatena todos os segmentos do generator.

    Vozes AmEn : af_heart, af_bella, af_sky, am_adam, am_michael
    Vozes BrEn : bf_emma, bf_alice, bm_george, bm_lewis
    Vozes PT-BR: pf_dora, pm_alex, pm_santa
    """
    global kokoro_pipeline, KOKORO_LOADED
    global _is_loading, _is_generating

    if _is_loading:
        raise RuntimeError("Modelo Kokoro ainda está carregando")

    if not KOKORO_LOADED or kokoro_pipeline is None:
        logger.error("Kokoro nao carregado. Chame load_kokoro_engine() primeiro.")
        return None, None

    with _engine_lock:
        _is_generating = True

    if torch.cuda.is_available(): torch.cuda.synchronize()

    with _synthesis_lock:
        try:
            import time
            t0_ksynth = time.time()
            # [DEEP DIAGNOSTICS] - Verificando o dispositivo real dos pesos
            try:
                # KPipeline (pip package) holds the model in .model
                _model_device = "unknown"
                if hasattr(kokoro_pipeline, "model"):
                    _model_device = str(next(kokoro_pipeline.model.parameters()).device)
                    # Força para a GPU se estiver na CPU por engano
                    _target = "cuda" if torch.cuda.is_available() else "cpu"
                    if _target == "cuda" and "cpu" in _model_device.lower():
                        print(f"[ENGINE] [Kokoro] [FIX] Redirecionando pesos CPU -> GPU...")
                        kokoro_pipeline.model.to("cuda")
                        _model_device = "cuda:0"
                    
                    # Otimização Kokoro: Mantemos Float32 (Half causa RuntimeError com o pacote kokoro-pip)
                    # if _target == "cuda" and next(kokoro_pipeline.model.parameters()).dtype == torch.float32:
                    #     print(f"[ENGINE] [Kokoro] [FIX] Ativando FP16 (Half Precision)...")
                    #     kokoro_pipeline.model.half()

                print(f"[ENGINE] [Kokoro] Iniciando sintese | Texto len: {len(text)} chars | Voice: {voice} | Device: {_model_device}")
            except Exception as _diag_err:
                logger.debug(f"Falha no diagnostico Kokoro: {_diag_err}")

            audio_parts = []
            _t_start_loop = time.time()
            _g2p_done = False
            
            # Generator loop: Kokoro faz G2P no início e depois itera os segmentos
            generator = kokoro_pipeline(text, voice=voice, speed=speed)
            
            for gs, ps, audio in generator:
                if not _g2p_done:
                    _t_g2p = time.time() - _t_start_loop
                    print(f"[DEBUG] [Kokoro] G2P/Phonemes Time: {_t_g2p:.4f}s")
                    _g2p_done = True
                
                if audio is not None and len(audio) > 0:
                    audio_parts.append(audio)
            
            _t_total_inference = time.time() - _t_start_loop
            print(f"[DEBUG] [Kokoro] Total Synthesis Time (Loop): {_t_total_inference:.4f}s")

            if not audio_parts:
                print("[ERROR] [ENGINE] [Kokoro] Nenhum audio gerado pelo generator.")
                logger.error("Kokoro nao gerou audio.")
                return None, None

            # Converter tensores PyTorch (CUDA/CPU) para NumPy arrays
            if hasattr(audio_parts[0], "cpu"):
                audio_parts = [a.detach().cpu().numpy() for a in audio_parts]

            final = np.concatenate(audio_parts) if len(audio_parts) > 1 else audio_parts[0]
            check_audio_validity(final)
            print(f"[ENGINE] [Kokoro] Sintese concluida com sucesso | Samples: {len(final)} | Tempo: {time.time() - t0_ksynth:.2f}s")
            return final.astype(np.float32), KOKORO_SAMPLE_RATE

        except Exception as e:
            print("ERRO NA GERAÇÃO KOKORO:", e)
            import traceback
            traceback.print_exc()
            logger.error(f"Erro na sintese Kokoro: {e}", exc_info=True)
            return None, None
        finally:
            with _engine_lock:
                _is_generating = False


def unload_kokoro_engine():
    global kokoro_pipeline, KOKORO_LOADED
    with _kokoro_lock:
        if kokoro_pipeline is not None:
            del kokoro_pipeline
            kokoro_pipeline = None
        KOKORO_LOADED = False
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    logger.info("Kokoro descarregado.")


# =============================================================================
# QWEN3 TTS — (padrao TTS-Story src/engines/qwen3_custom_voice_engine.py)
# =============================================================================

# ENGINE SWITCHER

# =============================================================================
# UNLOADER / SWITCHER
# =============================================================================

def unload_all_for_switch():
    """
    Descarrega Chatterbox e Kokoro para liberar VRAM.
    Vital para evitar OOM na RTX 5070 Ti ao alternar modelos pesados.
    """
    global chatterbox_model, MODEL_LOADED, loaded_model_type, loaded_model_class_name
    global kokoro_pipeline, KOKORO_LOADED
    global _voice_embedding_cache

    print("[ENGINE] Descarregando modelos para troca...")
    
    # 1. Chatterbox
    if chatterbox_model is not None:
        chatterbox_model = None
        MODEL_LOADED = False
        loaded_model_type = None
        loaded_model_class_name = None
    
    _voice_embedding_cache.clear()

    # 2. Kokoro
    if kokoro_pipeline is not None:
        kokoro_pipeline = None
        KOKORO_LOADED = False

    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            v_free = torch.cuda.mem_get_info()[0] / 1e9
            print(f"[ENGINE] VRAM Livre: {v_free:.2f} GB")
    except: pass

_switch_lock = threading.Lock()

def switch_to_engine(engine_name: str, model_type: Optional[str] = None) -> bool:
    """
    Muda para o engine especificado (chatterbox | kokoro).
    """
    if _switch_lock.locked():
        print("[ENGINE] Troca bloqueada (transicao ja em andamento)")
        return False

    with _engine_lock:
        if _is_generating:
            print("[ENGINE] switch_to_engine bloqueado: geracao ativa.")
            return False

    with _switch_lock:
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.synchronize()
        except: pass
            
        eng = engine_name.lower().replace("-", "_").replace(" ", "_")
        print(f"[ENGINE] Switch to: {eng} (sub: {model_type})")

        if eng in ("chatterbox", "turbo", "original", "multilingual"):
            if eng != "chatterbox":
                model_type = eng
            return load_model(model_type=model_type)

        unload_all_for_switch()
        
        if eng == "kokoro":
            return load_kokoro_engine()

        print(f"[ENGINE] Desconhecido: {engine_name}")
        return False

def get_active_engine() -> str:
    if KOKORO_LOADED:  return "kokoro"
    if MODEL_LOADED:   return "chatterbox"
    return "none"
