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
import numpy as np
import torch
import time
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

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
    if kokoro_path not in sys.path:
        import sys
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
KOKORO_LOADED: bool = False
_kokoro_lock        = threading.Lock()

# =============================================================================
# QWEN3 TTS — import defensivo (padrao TTS-Story)
# NUNCA usar sdpa — causa lentidao severa
# =============================================================================
try:
    from qwen_tts import Qwen3TTSModel
    QWEN3_AVAILABLE = True
except ImportError:
    try:
        from qwen_tts.core.models.modeling_qwen3_tts import (
            Qwen3TTSForConditionalGeneration as Qwen3TTSModel,
        )
        QWEN3_AVAILABLE = True
    except ImportError:
        Qwen3TTSModel   = None
        QWEN3_AVAILABLE = False
        logger.warning("qwen_tts nao instalado. Execute: pip install --upgrade qwen-tts")

# Globals Qwen
qwen_model        = None
QWEN_LOADED: bool = False
qwen_device       = None
_qwen_lock        = threading.Lock()

_QWEN_SPEAKER_MAP = {
    "aiden":    "Aiden",  "dylan":    "Dylan",
    "eric":     "Eric",   "ono_anna": "Ono_Anna",
    "ryan":     "Ryan",   "serena":   "Serena",
    "sohee":    "Sohee",  "uncle_fu": "Uncle_Fu",
    "vivian":   "Vivian",
}

# =============================================================================
# INDEXTTS — subprocess via uv (padrao TTS-Story)
# Requer: engines/index-tts/ clonado + uv sync + checkpoints/
# =============================================================================
INDEX_TTS_ENGINE_DIR = (_REPO_ROOT / "engines" / "index-tts").resolve()
INDEX_TTS_WORKER     = (INDEX_TTS_ENGINE_DIR / "tts_worker.py").resolve()
INDEX_TTS_VENV_DIR   = (INDEX_TTS_ENGINE_DIR / ".venv").resolve()
INDEX_TTS_AVAILABLE  = (
    INDEX_TTS_ENGINE_DIR.exists()
    and INDEX_TTS_WORKER.exists()
    and INDEX_TTS_VENV_DIR.exists()
)

if not INDEX_TTS_AVAILABLE:
    reason_parts = []
    if not INDEX_TTS_ENGINE_DIR.exists():
        reason_parts.append("engines/index-tts/ nao encontrado")
    elif not INDEX_TTS_VENV_DIR.exists():
        reason_parts.append(".venv nao encontrado — execute setup.bat")
    elif not INDEX_TTS_WORKER.exists():
        reason_parts.append("tts_worker.py nao encontrado em engines/index-tts/")
    if reason_parts:
        logger.warning(
            f"IndexTTS indisponivel: {', '.join(reason_parts)}\n"
            "Para configurar: execute setup.bat"
        )

_indextts_proc       = None
_indextts_proc_lock  = threading.Lock()
INDEX_TTS_LOADED: bool = False

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

# Lock compartilhado GPU — garante que apenas 1 modelo usa GPU por vez
_synthesis_lock = threading.Lock()

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
# CUDA OTIMIZACOES — RTX 5070 Ti (Blackwell)
# =============================================================================
if torch.cuda.is_available():
    try:
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU detectada: {gpu_name}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        logger.info("TF32 e high-precision matmul ativados.")
    except Exception as e:
        logger.debug(f"Nao foi possivel aplicar otimizacoes CUDA no inicio: {e}")


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
        torch.backends.cudnn.benchmark = False  # TTS: comprimentos variáveis
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
    """PADRAO TTS-STORY: flash_attention_2 > eager. Nunca sdpa."""
    try:
        import flash_attn  # noqa
        logger.info("Qwen: flash_attention_2 disponivel.")
        return "flash_attention_2"
    except ImportError:
        logger.warning(
            "Qwen: flash-attn ausente, usando 'eager'.\n"
            "Para melhor performance: pip install flash-attn --no-build-isolation"
        )
        return "eager"

def _get_uv_path() -> str:
    """Retorna o caminho do uv, tentando locais comuns no Windows."""
    # 1. Tentar no PATH
    if shutil.which("uv"):
        return "uv"
    
    # 2. Tentar caminhos comuns no Windows
    user_home = Path.home()
    common_paths = [
        user_home / ".local" / "bin" / "uv.exe",
        user_home / ".cargo" / "bin" / "uv.exe",
        Path(os.environ.get("APPDATA", "")) / "uv" / "bin" / "uv.exe"
    ]
    for p in common_paths:
        if p.exists():
            return str(p)
            
    return "uv" # Fallback para o PATH

def _check_uv_installed() -> bool:
    uv_path = _get_uv_path()
    try:
        return subprocess.run([uv_path, "--version"], capture_output=True, timeout=5).returncode == 0
    except: return False


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
        if _test_cuda_functionality():   resolved = "cuda"
        elif _test_mps_functionality():  resolved = "mps"
        else:                            resolved = "cpu"
    elif device_setting == "cuda":
        resolved = "cuda" if _test_cuda_functionality() else "cpu"
    elif device_setting == "mps":
        resolved = "mps" if _test_mps_functionality() else "cpu"
    else:
        resolved = "cpu"
    return resolved

HAS_TURBO = CHATTERBOX_TURBO_AVAILABLE
print(f"[ENGINE] Turbo disponível: {HAS_TURBO}")

def load_turbo() -> bool:
    """Carrega Chatterbox Turbo com fallback automático para Original."""
    global chatterbox_model, MODEL_LOADED, model_device, loaded_model_type, loaded_model_class_name
    print("[DEBUG] Solicitado: load_turbo")
    
    if not HAS_TURBO:
        print("[ENGINE] Turbo não disponível nesta versão do chatterbox. Usando fallback (original).")
        return load_original()

    if loaded_model_type == "turbo" and MODEL_LOADED:
        return True
    
    unload_all_for_switch()
    model_device = _get_chatterbox_device()
    _optimize_for_device(model_device)
    
    try:
        # Importacao ja tentada globalmente como _ChatterboxTurbo
        if _ChatterboxTurbo is None:
            # Segunda tentativa de importacao via path correto
            from chatterbox.tts_turbo import ChatterboxTurboTTS as _Turbo
        else:
            _Turbo = _ChatterboxTurbo

        chatterbox_model = _Turbo.from_pretrained(device=model_device)
        loaded_model_type = "turbo"
        loaded_model_class_name = "ChatterboxTurboTTS"
        MODEL_LOADED = True
        logger.info(f"Chatterbox Turbo carregado em {model_device}.")
        return True
    except Exception as e:
        logger.error(f"Falha ao carregar Turbo: {e}. Tentando fallback original.", exc_info=True)
        return load_original()

def load_original() -> bool:
    """Carrega especificamente o Chatterbox Original."""
    global chatterbox_model, MODEL_LOADED, model_device, loaded_model_type, loaded_model_class_name
    print("[DEBUG] Solicitado: load_original")
    if loaded_model_type == "original" and MODEL_LOADED:
        return True

    unload_all_for_switch()
    model_device = _get_chatterbox_device()
    _optimize_for_device(model_device)

    try:
        from chatterbox import ChatterboxTTS as _Original
        chatterbox_model = _Original.from_pretrained(device=model_device)
        loaded_model_type = "original"
        loaded_model_class_name = "ChatterboxTTS"
        MODEL_LOADED = True
        logger.info(f"Chatterbox Original carregado em {model_device}.")
        return True
    except Exception as e:
        logger.error(f"Falha ao carregar Original: {e}", exc_info=True)
        return False

def load_multilingual() -> bool:
    """Carrega especificamente o Chatterbox Multilingual com patch eager isolado."""
    global chatterbox_model, MODEL_LOADED, model_device, loaded_model_type, loaded_model_class_name
    print("[DEBUG] Solicitado: load_multilingual")
    if loaded_model_type == "multilingual" and MODEL_LOADED:
        return True

    unload_all_for_switch()
    model_device = _get_chatterbox_device()
    _optimize_for_device(model_device)

    from copy import deepcopy
    from chatterbox.models.t3.llama_configs import LLAMA_CONFIGS
    original_configs = deepcopy(LLAMA_CONFIGS)
    
    try:
        print("[PATCH] Aplicando eager apenas para multilingual")
        for k in LLAMA_CONFIGS:
            LLAMA_CONFIGS[k]["attn_implementation"] = "eager"
        
        from chatterbox import ChatterboxMultilingualTTS as _Mtl
        chatterbox_model = _Mtl.from_pretrained(device=model_device)
        try: chatterbox_model = chatterbox_model.float()
        except AttributeError: pass
        
        loaded_model_type = "multilingual"
        loaded_model_class_name = "ChatterboxMultilingualTTS"
        MODEL_LOADED = True
        logger.info(f"Chatterbox Multilingual carregado em {model_device}.")
        return True
    except Exception as e:
        logger.error(f"Falha ao carregar Multilingual: {e}", exc_info=True)
        return False
    finally:
        print("[PATCH] Restaurando configs originais")
        LLAMA_CONFIGS.clear()
        LLAMA_CONFIGS.update(original_configs)

def load_model() -> bool:
    """Funcao legada para compatibilidade - usa a config atual."""
    sel = config_manager.get_string("model.repo_id", "turbo").lower()
    if "multilingual" in sel: return load_multilingual()
    if "original" in sel or "chatterbox" in sel and "turbo" not in sel: return load_original()
    return load_turbo()


def warmup_model():
    """Warmup completo — aplicar torch.compile ANTES do forward dummy."""
    global chatterbox_model, MODEL_LOADED, model_device
    if not MODEL_LOADED or chatterbox_model is None: return

    if model_device == "cuda" and hasattr(torch, "compile"):
        for attr in ("t3", "s3gen"):
            if hasattr(chatterbox_model, attr):
                try:
                    setattr(chatterbox_model, attr, torch.compile(
                        getattr(chatterbox_model, attr),
                        mode="reduce-overhead", fullgraph=False,
                    ))
                    logger.info(f"torch.compile aplicado ao {attr}.")
                except Exception as e:
                    logger.warning(f"torch.compile em {attr} falhou (nao critico): {e}")

    # Warmup 1 — sem audio de referencia
    try:
        with _synthesis_lock, torch.inference_mode():
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
        dummy_audio = np.zeros(dummy_sr * 3, dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        sf.write(tmp_path, dummy_audio, dummy_sr)
        try:
            with _synthesis_lock, torch.inference_mode():
                if hasattr(chatterbox_model, "generate"):
                    chatterbox_model.generate("Hello.", audio_prompt_path=tmp_path)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            logger.info("Warmup (com ref simulada) concluido.")
        finally:
            try: os.unlink(tmp_path)
            except: pass
    except Exception as e:
        logger.warning(f"Warmup com ref falhou: {e}")


def synthesize(
    text: str,
    audio_prompt_path: str = None,
    temperature: float = 0.8,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    seed: int = 0,
    language: str = "en",
    min_p: float = 0.05,
    top_p: float = 0.95,
    top_k: int = 1000,
    repetition_penalty: float = 1.2,
    norm_loudness: bool = True,
    **kwargs,
) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    """
    Sintetiza com Chatterbox (Turbo | Original | Multilingual).
    PADRAO TTS-STORY: autocast bfloat16 no CUDA para Turbo/Original.
    """
    global chatterbox_model, MODEL_LOADED, loaded_model_type

    if not MODEL_LOADED or chatterbox_model is None:
        logger.error("Chatterbox nao carregado.")
        return None, None

    if seed != 0:
        set_seed(seed)

    use_cuda       = (model_device == "cuda")
    is_multilingual = (loaded_model_type == "multilingual")

    if use_cuda and not is_multilingual and torch.cuda.is_bf16_supported():
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = contextlib.nullcontext()

    with _synthesis_lock:
        print("[ENGINE] Gerando áudio...")
        start_time = time.time()
        try:
            with torch.inference_mode(), autocast_ctx:
                if loaded_model_type == "turbo":
                    wav_tensor = chatterbox_model.generate(
                        text=text,
                        audio_prompt_path=audio_prompt_path,
                        temperature=temperature,
                        min_p=min_p,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        norm_loudness=norm_loudness,
                    )
                elif loaded_model_type == "multilingual":
                    wav_tensor = chatterbox_model.generate(
                        text=text,
                        language_id=language,
                        audio_prompt_path=audio_prompt_path,
                        temperature=temperature,
                        exaggeration=exaggeration,
                        cfg_weight=cfg_weight,
                        min_p=min_p,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                    )
                else:
                    # ORIGINAL
                    wav_tensor = chatterbox_model.generate(
                        text=text,
                        audio_prompt_path=audio_prompt_path,
                        temperature=temperature,
                        exaggeration=exaggeration,
                        cfg_weight=cfg_weight,
                        min_p=min_p,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                    )
            
            elapsed = time.time() - start_time
            if elapsed > 30:
                logger.warning(f"Sintese demorada: {elapsed:.2f}s")
            
            print(f"[DEBUG] Tipo retorno: {type(wav_tensor)}")
            if wav_tensor is None:
                raise ValueError("Modelo retornou None na geracao")

            if hasattr(wav_tensor, "shape"):
                print(f"[DEBUG] Shape retorno: {wav_tensor.shape}")

            wav_tensor = wav_tensor.to(torch.float32)
            return wav_tensor, chatterbox_model.sr

        except Exception as e:
            logger.error(f"Erro Chatterbox synthesize: {e}", exc_info=True)
            return None, None


def reload_model() -> bool:
    global chatterbox_model, MODEL_LOADED, model_device, loaded_model_type, loaded_model_class_name
    logger.info("Hot-swap Chatterbox...")
    if chatterbox_model is not None:
        del chatterbox_model
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

    lang_codes: "a"=AmEn, "b"=BrEn, "j"=Japanese, "z"=Chinese,
                "p"=PT-BR, "e"=Spanish, "f"=French
    """
    global kokoro_pipeline, KOKORO_LOADED

    with _kokoro_lock:
        if KOKORO_LOADED: return True

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
            print(f"[ENGINE] [Kokoro] Carregado com sucesso em {resolved} | Tempo: {time.time() - t0_kload:.2f}s")
            logger.info(f"Kokoro carregado com sucesso em {resolved}.")
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

    if not KOKORO_LOADED or kokoro_pipeline is None:
        logger.error("Kokoro nao carregado. Chame load_kokoro_engine() primeiro.")
        return None, None

    with _synthesis_lock:
        try:
            import time
            t0_ksynth = time.time()
            print(f"[ENGINE] [Kokoro] Iniciando sintese | Texto len: {len(text)} chars | Voice: {voice}")
            
            audio_parts = []
            for gs, ps, audio in kokoro_pipeline(text, voice=voice, speed=speed):
                if audio is not None and len(audio) > 0:
                    audio_parts.append(audio)

            if not audio_parts:
                print("[ERROR] [ENGINE] [Kokoro] Nenhum audio gerado pelo generator.")
                logger.error("Kokoro nao gerou audio.")
                return None, None

            # Converter tensores PyTorch (CUDA/CPU) para NumPy arrays
            if hasattr(audio_parts[0], "cpu"):
                audio_parts = [a.detach().cpu().numpy() for a in audio_parts]

            final = np.concatenate(audio_parts) if len(audio_parts) > 1 else audio_parts[0]
            print(f"[ENGINE] [Kokoro] Sintese concluida com sucesso | Samples: {len(final)} | Tempo: {time.time() - t0_ksynth:.2f}s")
            return final.astype(np.float32), KOKORO_SAMPLE_RATE

        except Exception as e:
            print(f"[ERROR] [ENGINE] [Kokoro] Erro critico na sintese: {e}")
            logger.error(f"Erro na sintese Kokoro: {e}", exc_info=True)
            return None, None


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

def load_qwen_model(device: str = "auto") -> bool:
    global qwen_model, QWEN_LOADED, qwen_device

    with _qwen_lock:
        if QWEN_LOADED: return True
        if not QWEN3_AVAILABLE:
            logger.error("qwen_tts nao instalado. Execute: pip install --upgrade qwen-tts")
            return False

        if device == "auto":
            if _test_cuda_functionality():   resolved = "cuda"
            elif _test_mps_functionality():  resolved = "mps"
            else:                            resolved = "cpu"
        else:
            resolved = device

        qwen_device = resolved
        _optimize_for_device(str(resolved))

        if resolved == "cuda" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        attn = _resolve_qwen_attn()

        try:
            logger.info(
                f"Carregando Qwen3-TTS | device={resolved} | dtype={dtype} | attn={attn}"
            )
            qwen_model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                device_map=resolved,
                dtype=dtype,
                attn_implementation=attn,
            )
            QWEN_LOADED = True
            _warmup_qwen()
            return True
        except Exception as e:
            logger.error(f"Falha critica ao carregar Qwen: {e}", exc_info=True)
            qwen_model  = None
            QWEN_LOADED = False
            return False


def _warmup_qwen():
    global qwen_model, QWEN_LOADED
    if not QWEN_LOADED or qwen_model is None: return
    try:
        logger.info("Warmup Qwen...")
        ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if qwen_device == "cuda" and torch.cuda.is_bf16_supported()
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), ctx:
            qwen_model.generate_custom_voice(
                text="Hello.", language="Auto",
                speaker=_QWEN_SPEAKER_MAP["ryan"],
            )
        if torch.cuda.is_available(): torch.cuda.synchronize()
        logger.info("Warmup Qwen concluido.")
    except Exception as e:
        logger.warning(f"Warmup Qwen falhou (nao critico): {e}")


def synthesize_qwen(
    text: str,
    speaker: str = "Ryan",
    language: str = "Auto",
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    PADRAO TTS-STORY src/engines/qwen3_custom_voice_engine.py:
      - generate_custom_voice() com speaker normalizado
      - autocast bfloat16 no CUDA
      - retorna numpy float32
    """
    global qwen_model, QWEN_LOADED

    if not QWEN_LOADED or qwen_model is None:
        logger.error("Qwen nao carregado. Chame load_qwen_model() primeiro.")
        return None, None

    speaker_final = _QWEN_SPEAKER_MAP.get(
        speaker.strip().lower().replace(" ", "_"), "Ryan"
    )

    with _synthesis_lock:
        try:
            ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if qwen_device == "cuda" and torch.cuda.is_bf16_supported()
                else contextlib.nullcontext()
            )
            with torch.inference_mode(), ctx:
                res_wavs, sr = qwen_model.generate_custom_voice(
                    text=text, speaker=speaker_final, language=language,
                )

            if not res_wavs:
                logger.error("Qwen retornou lista vazia.")
                return None, None

            wav = res_wavs[0]
            wav_np = (
                wav.cpu().float().numpy()
                if isinstance(wav, torch.Tensor)
                else np.array(wav, dtype=np.float32)
            )
            return wav_np, int(sr)

        except Exception as e:
            logger.error(f"Erro na sintese Qwen: {e}", exc_info=True)
            return None, None


def unload_qwen_model():
    global qwen_model, QWEN_LOADED
    with _qwen_lock:
        if qwen_model is not None:
            del qwen_model
            qwen_model = None
        QWEN_LOADED = False
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    logger.info("Qwen descarregado.")


# =============================================================================
# INDEXTTS — subprocess via uv (padrao TTS-Story)
# =============================================================================

def _start_indextts_worker() -> bool:
    global _indextts_proc
    if _indextts_proc and _indextts_proc.poll() is None:
        return True  # ja rodando

    if not INDEX_TTS_AVAILABLE:
        logger.error("IndexTTS nao configurado. Execute setup.bat.")
        return False

    # Garantir que o uv esta no PATH de forma robusta
    uv_path = _get_uv_path()
    if uv_path != "uv":
        uv_dir = str(Path(uv_path).parent)
        env_path = os.environ.get("PATH", "")
        if uv_dir not in env_path:
            os.environ["PATH"] = uv_dir + os.pathsep + env_path

    if not _check_uv_installed():
        logger.error(f"uv nao instalado ou nao encontrado em {uv_path}. Instale via: powershell -c 'irm https://astral.sh/uv/install.ps1 | iex'")
        return False

    try:
        env = dict(os.environ)
        # Removido PYTHONPATH para evitar shadowing — venv cuida disso
        if "PYTHONPATH" in env: del env["PYTHONPATH"]
        
        # O IndexTTS usa o python do proprio venv para garantir isolamento
        venv_python = str(INDEX_TTS_ENGINE_DIR / ".venv" / "Scripts" / "python.exe")
        
        # O script deve ser passado apenas pelo nome se o cwd for a pasta do engine
        # para evitar caminhos duplicados como engines/index-tts/engines/index-tts/tts_worker.py
        worker_script = "tts_worker.py"
        
        # Fallback para uv se o venv nao existir (improvavel se uv sync rodou)
        if not os.path.exists(venv_python):
            args = [uv_path, "run", "python", worker_script]
        else:
            args = [venv_python, worker_script]
            
        print(f"DEBUG IndexTTS CMD: {args}")
        logger.info(f"Iniciando IndexTTS via: {' '.join(args)}")

        _indextts_proc = subprocess.Popen(
            args,
            cwd=str(INDEX_TTS_ENGINE_DIR),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True, bufsize=1, env=env,
        )

        def _fwd():
            for line in _indextts_proc.stderr:
                if line.strip():
                    logger.debug(f"[IndexTTS] {line.rstrip()}")
        threading.Thread(target=_fwd, daemon=True).start()

        import time
        deadline = time.monotonic() + 120
        logger.info("Aguardando IndexTTS carregar (~30-60s)...")
        while time.monotonic() < deadline:
            if _indextts_proc.poll() is not None:
                logger.error("Worker IndexTTS encerrou inesperadamente.")
                return False
            line = _indextts_proc.stdout.readline()
            if not line:
                time.sleep(0.1)
                continue
            try:
                msg = json.loads(line.strip())
                if msg.get("status") == "ready":
                    logger.info("[ENGINE] IndexTTS carregado com sucesso.")
                    INDEX_TTS_LOADED = True
                    return True
            except json.JSONDecodeError:
                continue

        logger.error("Timeout aguardando IndexTTS.")
        _stop_indextts_worker()
        return False

    except Exception as e:
        logger.error(f"Falha ao iniciar IndexTTS: {e}", exc_info=True)
        return False


def _stop_indextts_worker():
    global _indextts_proc
    if _indextts_proc and _indextts_proc.poll() is None:
        try:
            _indextts_proc.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
            _indextts_proc.stdin.flush()
            _indextts_proc.wait(timeout=10)
        except Exception:
            try: _indextts_proc.kill()
            except: pass
    _indextts_proc = None
    logger.info("Worker IndexTTS encerrado.")


def synthesize_indextts(
    text: str,
    audio_prompt_path: str,
    output_path: str = None,
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Sintetiza com IndexTTS via subprocess uv (padrao TTS-Story).
    O worker e iniciado sob demanda e mantido vivo durante o batch.
    """
    global _indextts_proc

    with _indextts_proc_lock:
        if not _indextts_proc or _indextts_proc.poll() is not None:
            if not _start_indextts_worker():
                return None, None

    own_tmp = False
    if not output_path:
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        own_tmp = True

    try:
        req = json.dumps({
            "text":   text,
            "prompt": str(Path(audio_prompt_path).resolve()),
            "output": str(Path(output_path).resolve()),
        })

        with _indextts_proc_lock:
            _indextts_proc.stdin.write(req + "\n")
            _indextts_proc.stdin.flush()

            import time
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if _indextts_proc.poll() is not None:
                    logger.error("Worker IndexTTS encerrou durante sintese.")
                    return None, None
                line = _indextts_proc.stdout.readline()
                if not line:
                    time.sleep(0.05)
                    continue
                try:
                    resp = json.loads(line.strip())
                    if resp.get("status") == "ok":
                        import soundfile as sf
                        audio_np, sr = sf.read(output_path, dtype="float32")
                        if audio_np.ndim == 2:
                            audio_np = audio_np.mean(axis=1)
                        return audio_np, int(sr)
                    elif resp.get("status") == "error":
                        logger.error(f"IndexTTS erro: {resp.get('message')}")
                        return None, None
                except json.JSONDecodeError:
                    continue

        logger.error("Timeout aguardando resposta IndexTTS.")
        return None, None

    except Exception as e:
        logger.error(f"Erro na comunicacao com IndexTTS: {e}", exc_info=True)
        return None, None
    finally:
        if own_tmp and Path(output_path).exists():
            try: Path(output_path).unlink()
            except: pass


def unload_indextts():
    global INDEX_TTS_LOADED
    _stop_indextts_worker()
    INDEX_TTS_LOADED = False
    logger.info("IndexTTS descarregado.")


# =============================================================================
# VRAM SWITCH — descarregar todos os engines
# =============================================================================

def unload_all_for_switch():
    """
    PADRAO TTS-STORY: limpa VRAM antes de carregar novo engine.
    """
    global chatterbox_model, MODEL_LOADED, loaded_model_type, loaded_model_class_name
    global qwen_model, QWEN_LOADED
    global kokoro_pipeline, KOKORO_LOADED
    global INDEX_TTS_LOADED

    if chatterbox_model is not None:
        logger.info("Descarregando Chatterbox...")
        del chatterbox_model
        chatterbox_model        = None
        MODEL_LOADED            = False
        loaded_model_type       = None
        loaded_model_class_name = None

    if qwen_model is not None:
        logger.info("Descarregando Qwen...")
        del qwen_model
        qwen_model  = None
        QWEN_LOADED = False

    if kokoro_pipeline is not None:
        logger.info("Descarregando Kokoro...")
        del kokoro_pipeline
        kokoro_pipeline = None
        KOKORO_LOADED   = False

    _stop_indextts_worker()
    INDEX_TTS_LOADED = False

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            vram_free = torch.cuda.mem_get_info()[0] / 1e9
            logger.info(f"VRAM livre apos descarga: {vram_free:.2f} GB")
        except: pass
    if torch.backends.mps.is_available():
        try: torch.mps.empty_cache()
        except: pass


# =============================================================================
# ENGINE SWITCHER
# =============================================================================

def switch_to_engine(engine: str) -> bool:
    """
    Muda para o engine especificado, descarregando o anterior primeiro.
    Engines: original | turbo | multilingual | kokoro | qwen | indextts
    """
    eng = engine.lower().replace("-", "_").replace(" ", "_")
    print(f"[DEBUG] Engine solicitado: {eng}")
    logger.info(f"switch_to_engine: {eng}")

    if eng == "turbo":
        return load_turbo()
    
    if eng == "original":
        return load_original()
    
    if eng == "multilingual":
        return load_multilingual()

    unload_all_for_switch()
    torch.cuda.empty_cache()

    if eng == "kokoro":
        return load_kokoro_engine()

    if eng in ("qwen", "qwen3"):
        return load_qwen_model()

    if eng == "indextts":
        global INDEX_TTS_LOADED
        INDEX_TTS_LOADED = _start_indextts_worker()
        return INDEX_TTS_LOADED

    logger.error(f"Engine desconhecido: '{engine}'.")
    return False


def get_active_engine() -> str:
    if QWEN_LOADED:    return "qwen"
    if KOKORO_LOADED:  return "kokoro"
    if INDEX_TTS_LOADED or (_indextts_proc and _indextts_proc.poll() is None): return "indextts"
    if MODEL_LOADED:   return "chatterbox"
    return "none"
