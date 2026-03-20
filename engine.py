# File: engine.py
# Core TTS model loading and speech generation logic.

import gc
import logging
import random
import contextlib
import threading
import numpy as np
import torch
from typing import Optional, Tuple
from pathlib import Path

from chatterbox.tts import ChatterboxTTS  # Main TTS engine class
from chatterbox.models.s3gen.const import (
    S3GEN_SR,
)  # Default sample rate from the engine

# Defensive Turbo import - Turbo may not be available in older package versions
try:
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    TURBO_AVAILABLE = True
except ImportError:
    ChatterboxTurboTTS = None
    TURBO_AVAILABLE = False

# Defensive Multilingual import
try:
    from chatterbox import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

    MULTILINGUAL_AVAILABLE = True
except ImportError:
    ChatterboxMultilingualTTS = None
    SUPPORTED_LANGUAGES = {}
    MULTILINGUAL_AVAILABLE = False

# Import the singleton config_manager
from config import config_manager

logger = logging.getLogger(__name__)

# --- RTX 5070 Ti (Blackwell) Optimizations ---
if torch.cuda.is_available():
    try:
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU detected: {gpu_name}")
        # Blackwell (RTX 50xx) and Ada Lovelace (RTX 40xx) support bfloat16 natively
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        logger.info("TF32 and high-precision matmul enabled for CUDA.")
    except Exception as e:
        logger.debug(f"Could not apply CUDA optimizations at module load: {e}")

# Model selector whitelist - maps config values to model types
MODEL_SELECTOR_MAP = {
    # Original model selectors
    "chatterbox": "original",
    "original": "original",
    "resembleai/chatterbox": "original",
    # Turbo model selectors
    "chatterbox-turbo": "turbo",
    "turbo": "turbo",
    "resembleai/chatterbox-turbo": "turbo",
    # Multilingual model selectors
    "chatterbox-multilingual": "multilingual",
    "multilingual": "multilingual",
}

# Paralinguistic tags supported by Turbo model
TURBO_PARALINGUISTIC_TAGS = [
    "laugh",
    "chuckle",
    "sigh",
    "gasp",
    "cough",
    "clear throat",
    "sniff",
    "groan",
    "shush",
]

# --- Global Module Variables ---
chatterbox_model: Optional[ChatterboxTTS] = None
MODEL_LOADED: bool = False
model_device: Optional[str] = (
    None  # Stores the resolved device string ('cuda' or 'cpu')
)

# Track which model type is loaded
loaded_model_type: Optional[str] = None  # "original" or "turbo"
loaded_model_class_name: Optional[str] = None  # "ChatterboxTTS" or "ChatterboxTurboTTS"
_synthesis_lock = threading.Lock()  # Lock para garantir que a GPU processe um por vez sem conflitos internos


def set_seed(seed_value: int):
    """
    Sets the seed for torch, random, and numpy for reproducibility.
    This is called if a non-zero seed is provided for generation.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    logger.info(f"Global seed set to: {seed_value}")


def _test_cuda_functionality() -> bool:
    """
    Tests if CUDA is actually functional, not just available.

    Returns:
        bool: True if CUDA works, False otherwise.
    """
    if not torch.cuda.is_available():
        return False

    try:
        test_tensor = torch.tensor([1.0])
        test_tensor = test_tensor.cuda()
        test_tensor = test_tensor.cpu()
        return True
    except Exception as e:
        logger.warning(f"CUDA functionality test failed: {e}")
        return False


def _test_mps_functionality() -> bool:
    """
    Tests if MPS is actually functional, not just available.

    Returns:
        bool: True if MPS works, False otherwise.
    """
    if not torch.backends.mps.is_available():
        return False

    try:
        test_tensor = torch.tensor([1.0])
        test_tensor = test_tensor.to("mps")
        test_tensor = test_tensor.cpu()
        return True
    except Exception as e:
        logger.warning(f"MPS functionality test failed: {e}")
        return False


def _test_dml_functionality() -> bool:
    """
    Tests if DirectML is available for AMD/Intel/Any GPU on Windows.
    """
    try:
        import torch_directml
        return torch_directml.is_available()
    except ImportError:
        return False
        
def _optimize_for_device(device_str: str):
    """
    Applies general optimizations for the selected device.
    Specific optimizations for RTX 50xx (Blackwell) series.
    """
    if device_str == "cuda":
        # Para RTX 5070 Ti (Blackwell): bfloat16 nativo, matmul TF32 acelerado
        torch.backends.cudnn.benchmark = False  # False para TTS: comprimentos variáveis
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass
        # Log GPU info para diagnóstico
        try:
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            vram_free, _ = torch.cuda.mem_get_info()
            gpu_name = torch.cuda.get_device_name(0)
            bf16_ok = torch.cuda.is_bf16_supported()
            logger.info(f"GPU: {gpu_name} | VRAM: {vram_total:.1f} GB total, {vram_free/1e9:.1f} GB livre")
            logger.info(f"BFloat16 suportado: {bf16_ok} | TF32 ativado")
        except Exception:
            pass
    elif device_str == "mps":
        pass

    logger.info(f"Applied PyTorch optimizations for device: {device_str}")


def _get_model_class(selector: str) -> tuple:
    """
    Determines which model class to use based on the config selector value.

    Args:
        selector: The value from config model.repo_id

    Returns:
        Tuple of (model_class, model_type_string)

    Raises:
        ImportError: If Turbo or Multilingual is selected but not available in the package
    """
    selector_normalized = selector.lower().strip()
    model_type = MODEL_SELECTOR_MAP.get(selector_normalized)

    if model_type == "turbo":
        if not TURBO_AVAILABLE:
            raise ImportError(
                f"Model selector '{selector}' requires ChatterboxTurboTTS, "
                f"but it is not available in the installed chatterbox package. "
                f"Please update the chatterbox-tts package to the latest version, "
                f"or use 'chatterbox' to select the original model."
            )
        logger.info(
            f"Model selector '{selector}' resolved to Turbo model (ChatterboxTurboTTS)"
        )
        return ChatterboxTurboTTS, "turbo"

    if model_type == "multilingual":
        if not MULTILINGUAL_AVAILABLE:
            raise ImportError(
                f"Model selector '{selector}' requires ChatterboxMultilingualTTS, "
                f"but it is not available in the installed chatterbox package. "
                f"Please update the chatterbox-tts package to the latest version, "
                f"or use 'chatterbox' to select the original model."
            )
        logger.info(
            f"Model selector '{selector}' resolved to Multilingual model (ChatterboxMultilingualTTS)"
        )
        return ChatterboxMultilingualTTS, "multilingual"

    if model_type == "original":
        logger.info(
            f"Model selector '{selector}' resolved to Original model (ChatterboxTTS)"
        )
        return ChatterboxTTS, "original"

    # Unknown selector - default to original with warning
    logger.warning(
        f"Unknown model selector '{selector}'. "
        f"Valid values: chatterbox, chatterbox-turbo, chatterbox-multilingual, original, turbo, multilingual, "
        f"ResembleAI/chatterbox, ResembleAI/chatterbox-turbo. "
        f"Defaulting to original ChatterboxTTS model."
    )
    return ChatterboxTTS, "original"


def get_model_info() -> dict:
    """
    Returns information about the currently loaded model.
    Used by the API to expose model details to the UI.

    Returns:
        Dictionary containing model information
    """
    return {
        "loaded": MODEL_LOADED,
        "type": loaded_model_type,  # "original", "turbo", or "multilingual"
        "class_name": loaded_model_class_name,
        "device": model_device,
        "sample_rate": chatterbox_model.sr if chatterbox_model else None,
        "supports_paralinguistic_tags": loaded_model_type == "turbo",
        "available_paralinguistic_tags": (
            TURBO_PARALINGUISTIC_TAGS if loaded_model_type == "turbo" else []
        ),
        "turbo_available_in_package": TURBO_AVAILABLE,
        "multilingual_available_in_package": MULTILINGUAL_AVAILABLE,
        "supports_multilingual": loaded_model_type == "multilingual",
        "supported_languages": (
            SUPPORTED_LANGUAGES if loaded_model_type == "multilingual" else {"en": "English"}
        ),
    }


def load_model() -> bool:
    """
    Loads the TTS model.
    This version directly attempts to load from the Hugging Face repository (or its cache)
    using `from_pretrained`, bypassing the local `paths.model_cache` directory.
    Updates global variables `chatterbox_model`, `MODEL_LOADED`, and `model_device`.

    Returns:
        bool: True if the model was loaded successfully, False otherwise.
    """
    global chatterbox_model, MODEL_LOADED, model_device
    global loaded_model_type, loaded_model_class_name

    if MODEL_LOADED:
        logger.info("TTS model is already loaded.")
        return True

    try:
        # Determine processing device with robust CUDA detection and intelligent fallback
        device_setting = config_manager.get_string("tts_engine.device", "auto")

        if device_setting == "auto":
            if _test_cuda_functionality():
                resolved_device_str = "cuda"
                logger.info("CUDA functionality test passed. Using CUDA.")
            elif _test_mps_functionality():
                resolved_device_str = "mps"
                logger.info("MPS functionality test passed. Using MPS.")
            elif _test_dml_functionality():
                import torch_directml
                resolved_device_str = torch_directml.device()
                logger.info("DirectML available. Using generic GPU via DML.")
            else:
                resolved_device_str = "cpu"
                logger.info("CUDA, MPS and DML not available. Using CPU.")

        elif device_setting == "cuda":
            if _test_cuda_functionality():
                resolved_device_str = "cuda"
                logger.info("CUDA requested and functional. Using CUDA.")
            else:
                resolved_device_str = "cpu"
                logger.warning(
                    "CUDA was requested in config but functionality test failed. "
                    "Automatically falling back to CPU."
                )

        elif device_setting == "mps":
            if _test_mps_functionality():
                resolved_device_str = "mps"
                logger.info("MPS requested and functional. Using MPS.")
            else:
                resolved_device_str = "cpu"
                logger.warning(
                    "MPS was requested in config but functionality test failed. "
                    "Automatically falling back to CPU."
                )
        
        elif device_setting == "dml":
            if _test_dml_functionality():
                import torch_directml
                resolved_device_str = torch_directml.device()
                logger.info("DirectML requested and functional. Using DML.")
            else:
                resolved_device_str = "cpu"
                logger.warning(
                    "DirectML was requested in config but functionality test failed. "
                    "Automatically falling back to CPU."
                )

        elif device_setting == "cpu":
            resolved_device_str = "cpu"
            logger.info("CPU device explicitly requested in config. Using CPU.")

        else:
            logger.warning(
                f"Invalid device setting '{device_setting}' in config. "
                f"Defaulting to auto-detection."
            )
            if _test_cuda_functionality():
                resolved_device_str = "cuda"
            elif _test_mps_functionality():
                resolved_device_str = "mps"
            elif _test_dml_functionality():
                import torch_directml
                resolved_device_str = torch_directml.device()
            else:
                resolved_device_str = "cpu"
            logger.info(f"Auto-detection resolved to: {resolved_device_str}")

        model_device = resolved_device_str
        logger.info(f"Final device selection: {model_device}")
        _optimize_for_device(str(model_device))

        # Get the model selector from config
        model_selector = config_manager.get_string("model.repo_id", "chatterbox-turbo")

        logger.info(f"Model selector from config: '{model_selector}'")

        try:
            # Determine which model class to use
            model_class, model_type = _get_model_class(model_selector)

            logger.info(
                f"Initializing {model_class.__name__} on device '{model_device}'..."
            )
            logger.info(f"Model type: {model_type}")
            if model_type == "turbo":
                logger.info(
                    f"Turbo model supports paralinguistic tags: {TURBO_PARALINGUISTIC_TAGS}"
                )

            # Load the model using from_pretrained - handles HuggingFace downloads automatically
            chatterbox_model = model_class.from_pretrained(device=model_device)

            # Store model metadata
            loaded_model_type = model_type
            loaded_model_class_name = model_class.__name__

            logger.info(f"Successfully loaded {model_class.__name__} on {model_device}")
            logger.info(f"Model sample rate: {chatterbox_model.sr} Hz")
        except ImportError as e_import:
            logger.error(
                f"Failed to load model due to import error: {e_import}",
                exc_info=True,
            )
            chatterbox_model = None
            MODEL_LOADED = False
            return False
        except Exception as e_hf:
            logger.error(
                f"Failed to load model using from_pretrained: {e_hf}",
                exc_info=True,
            )
            chatterbox_model = None
            MODEL_LOADED = False
            return False

        MODEL_LOADED = True
        if chatterbox_model:
            logger.info(
                f"TTS Model loaded successfully on {model_device}. Engine sample rate: {chatterbox_model.sr} Hz."
            )
        else:
            logger.error(
                "Model loading sequence completed, but chatterbox_model is None. This indicates an unexpected issue."
            )
            MODEL_LOADED = False
            return False

        return True

    except Exception as e:
        logger.error(
            f"An unexpected error occurred during model loading: {e}", exc_info=True
        )
        chatterbox_model = None
        MODEL_LOADED = False
        return False


def synthesize(
    text: str,
    audio_prompt_path: Optional[str] = None,
    temperature: float = 0.8,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    seed: int = 0,
    language: str = "en",
    min_p: float = 0.05,
    top_p: float = 1.0,
    top_k: int = 1000,
    repetition_penalty: float = 1.2,
    norm_loudness: bool = True,
) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    """
    Synthesizes audio from text using the loaded TTS model.

    Args:
        text: The text to synthesize.
        audio_prompt_path: Path to an audio file for voice cloning or predefined voice.
        temperature: Controls randomness in generation.
        exaggeration: Controls expressiveness.
        cfg_weight: Classifier-Free Guidance weight.
        seed: Random seed for generation. If 0, default randomness is used.
              If non-zero, a global seed is set for reproducibility.
        language: Language code for multilingual model (e.g., 'en', 'it', 'de').

    Returns:
        A tuple containing the audio waveform (torch.Tensor) and the sample rate (int),
        or (None, None) if synthesis fails.
    """
    global chatterbox_model

    if not MODEL_LOADED or chatterbox_model is None:
        logger.error("TTS model is not loaded. Cannot synthesize audio.")
        return None, None

    try:
        # Set seed globally if a specific seed value is provided and is non-zero.
        if seed != 0:
            logger.debug(f"Applying user-provided seed for generation: {seed}")
            set_seed(seed)
        # (seed==0: sem log — reduzir ruído no console)

        logger.debug(
            f"Synthesizing with params: audio_prompt='{audio_prompt_path}', temp={temperature}, "
            f"exag={exaggeration}, cfg_weight={cfg_weight}, seed_applied_globally_if_nonzero={seed}, "
            f"language={language}"
        )

        # Determina a precisão correta para cada modelo:
        # - turbo/original: bfloat16 (Blackwell suporta nativamente, mais rápido)
        # - multilingual:   float32  (bfloat16 causa repetição de tokens e instabilidade)
        use_cuda = model_device == "cuda"
        is_multilingual = (loaded_model_type == "multilingual")
        if use_cuda and not is_multilingual and torch.cuda.is_bf16_supported():
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            autocast_ctx = contextlib.nullcontext()

        with torch.inference_mode():
            with autocast_ctx:
                if loaded_model_type == "multilingual":
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
                elif loaded_model_type == "turbo":
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
                else:  # original
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

        # Ensure tensor is cast to float32 upon return so PySoundFile accepts it
        if wav_tensor is not None:
            wav_tensor = wav_tensor.to(torch.float32)

        # The ChatterboxTTS.generate method already returns a CPU tensor.
        return wav_tensor, chatterbox_model.sr

    except Exception as e:
        logger.error(f"Error during TTS synthesis: {e}", exc_info=True)
        return None, None


def reload_model() -> bool:
    """
    Unloads the current model, clears GPU memory, and reloads the model
    based on the current configuration. Used for hot-swapping models
    without restarting the server process.

    Returns:
        bool: True if the new model loaded successfully, False otherwise.
    """
    global chatterbox_model, MODEL_LOADED, model_device, loaded_model_type, loaded_model_class_name

    logger.info("Initiating model hot-swap/reload sequence...")

    # 1. Unload existing model
    if chatterbox_model is not None:
        logger.info("Unloading existing TTS model from memory...")
        del chatterbox_model
        chatterbox_model = None

    # 2. Reset state flags
    MODEL_LOADED = False
    loaded_model_type = None
    loaded_model_class_name = None

    # 3. Force Python Garbage Collection
    gc.collect()
    logger.info("Python garbage collection completed.")

    # 4. Clear GPU Cache (CUDA)
    if torch.cuda.is_available():
        logger.info("Clearing CUDA cache...")
        torch.cuda.empty_cache()

    # 5. Clear GPU Cache (MPS - Apple Silicon)
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
            logger.info("Cleared MPS cache.")
        except AttributeError:
            # Older PyTorch versions may not have mps.empty_cache()
            logger.debug(
                "torch.mps.empty_cache() not available in this PyTorch version."
            )

    # 6. Reload model from the (now updated) configuration
    logger.info("Memory cleared. Reloading model from updated config...")
    return load_model()


# --- End File: engine.py ---
