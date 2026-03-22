# workers/qwen_server.py
# Servidor Flask persistente para Qwen3-TTS
# Roda no venv_qwen — completamente isolado do Chatterbox
#
# INICIAR: venv_qwen\Scripts\python workers\qwen_server.py
# PORTA:   5001 (configuravel via variavel de ambiente QWEN_PORT)
#
# O modelo e carregado UMA UNICA VEZ quando o servidor sobe.
# Chamadas subsequentes usam o modelo ja na VRAM — sem reload.

import gc
import io
import os
import sys
import logging
import contextlib
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from flask import Flask, request, send_file, jsonify

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [QWEN-WORKER] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Import defensivo do Qwen ---
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
        logger.error(
            "qwen_tts nao instalado neste venv.\n"
            "Execute: venv_qwen\\Scripts\\pip install --upgrade qwen-tts"
        )
        sys.exit(1)

# --- Configuracao ---
PORT         = int(os.environ.get("QWEN_PORT", 5001))
MODEL_ID     = os.environ.get("QWEN_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
SAMPLE_RATE  = 24000

# Mapeamento de speakers (normalizacao de nomes)
SPEAKER_MAP = {
    "aiden":    "Aiden",  "dylan":    "Dylan",
    "eric":     "Eric",   "ono_anna": "Ono_Anna",
    "ryan":     "Ryan",   "serena":   "Serena",
    "sohee":    "Sohee",  "uncle_fu": "Uncle_Fu",
    "vivian":   "Vivian",
}
DEFAULT_SPEAKER = "Ryan"

# --- Estado global do modelo ---
model     = None
device    = None
_lock     = threading.Lock()   # serializa inferencias — 1 por vez na GPU


def _resolve_device() -> str:
    if torch.cuda.is_available():
        try:
            t = torch.tensor([1.0]).cuda()
            t.cpu()
            return "cuda"
        except Exception:
            pass
    return "cpu"


def _resolve_dtype(dev: str):
    if dev == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def _resolve_attn() -> str:
    """
    flash_attention_2 > eager (nunca sdpa — lento para autoregressivo).
    Padrao TTS-Story: _resolve_qwen_attn()
    """
    try:
        import flash_attn  # noqa: F401
        logger.info("Atencao: flash_attention_2 disponivel.")
        return "flash_attention_2"
    except ImportError:
        logger.warning(
            "flash-attn nao instalado — usando 'eager'.\n"
            "Para melhor performance: pip install flash-attn --no-build-isolation"
        )
        return "eager"


def load_model():
    """
    Carrega o Qwen3-TTS na VRAM uma unica vez.
    Aplicando todas as otimizacoes para RTX 5070 Ti (Blackwell):
      - bfloat16 nativo (economiza ~3.5 GB vs float32)
      - flash_attention_2 ou eager (nunca sdpa)
      - TF32 habilitado
      - torch.backends.cudnn.benchmark = False (TTS tem comprimentos variaveis)
    """
    global model, device

    device = _resolve_device()
    dtype  = _resolve_dtype(device)
    attn   = _resolve_attn()

    # Otimizacoes CUDA (mesmo padrao de _optimize_for_device() do engine.py)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        torch.backends.cudnn.benchmark        = False  # importante para TTS
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        try:
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            vram_free  = torch.cuda.mem_get_info()[0] / 1e9
            gpu_name   = torch.cuda.get_device_name(0)
            logger.info(f"GPU: {gpu_name} | VRAM: {vram_total:.1f}GB total, {vram_free:.1f}GB livre")
        except Exception:
            pass

    logger.info(f"Carregando {MODEL_ID} | device={device} | dtype={dtype} | attn={attn}")

    try:
        model = Qwen3TTSModel.from_pretrained(
            MODEL_ID,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn,
        )
    except Exception as e:
        error_msg = str(e).lower()
        if "no kernel image" in error_msg or "sm_120" in error_msg or "sm_100" in error_msg:
            logger.warning(f"DETECCAO BLACKWELL: Falha na GPU ({error_msg}). Caindo para CPU...")
            device = "cpu"
            dtype  = torch.float32
            model = Qwen3TTSModel.from_pretrained(
                MODEL_ID,
                device_map="cpu",
                dtype=torch.float32,
                attn_implementation="eager",
            )
        else:
            raise e

    logger.info("Modelo carregado. Executando warmup...")
    _warmup()
    logger.info(f"Qwen3-TTS pronto na porta {PORT}.")


def _warmup():
    """Sintese dummy para pre-alocar buffers CUDA — elimina cold-start."""
    try:
        ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device == "cuda" else contextlib.nullcontext()
        )
        with torch.inference_mode():
            with ctx:
                model.generate_custom_voice(
                    text="Hello.",
                    language="Auto",
                    speaker=DEFAULT_SPEAKER,
                )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.info("Warmup concluido.")
    except Exception as e:
        logger.warning(f"Warmup falhou (nao critico): {e}")


# --- Flask App ---
app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    vram_info = {}
    if torch.cuda.is_available():
        try:
            vram_info = {
                "allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
                "free_gb":      round(torch.cuda.mem_get_info()[0] / 1e9, 2),
            }
        except Exception:
            pass
    return jsonify({
        "status":  "ok",
        "model":   MODEL_ID,
        "device":  device,
        "vram":    vram_info,
    })


@app.route("/synthesize", methods=["POST"])
def synthesize():
    """
    POST /synthesize
    Body JSON: {
        "text":     "texto a sintetizar",
        "speaker":  "Ryan",          (opcional, default: Ryan)
        "language": "Auto"           (opcional, default: Auto)
    }
    Retorna: audio/wav binario
    """
    data     = request.get_json(silent=True) or {}
    text     = (data.get("text") or "").strip()
    speaker  = (data.get("speaker") or DEFAULT_SPEAKER).strip().lower().replace(" ", "_")
    language = (data.get("language") or "Auto").strip()

    if not text:
        return jsonify({"error": "text is required"}), 400

    # Normalizar speaker
    speaker_final = SPEAKER_MAP.get(speaker, DEFAULT_SPEAKER)

    # Serializar acesso a GPU — 1 inferencia por vez
    with _lock:
        try:
            ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device == "cuda" and torch.cuda.is_bf16_supported()
                else contextlib.nullcontext()
            )
            with torch.inference_mode():
                with ctx:
                    res_wavs, sr = model.generate_custom_voice(
                        text=text,
                        speaker=speaker_final,
                        language=language,
                    )

            if not res_wavs:
                return jsonify({"error": "model returned empty audio"}), 500

            wav = res_wavs[0]
            wav_np = (
                wav.cpu().float().numpy()
                if isinstance(wav, torch.Tensor)
                else np.array(wav, dtype=np.float32)
            )

            # Serializar para WAV em memoria
            buf = io.BytesIO()
            sf.write(buf, wav_np, int(sr), format="WAV")
            buf.seek(0)

            return send_file(buf, mimetype="audio/wav")

        except Exception as e:
            logger.error(f"Erro na sintese: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500


@app.route("/speakers", methods=["GET"])
def list_speakers():
    return jsonify({"speakers": list(SPEAKER_MAP.keys())})


if __name__ == "__main__":
    load_model()
    # threaded=False — garante que apenas 1 request processa por vez na GPU
    # Isso evita conflitos de contexto CUDA em inferencias paralelas
    app.run(
        host="127.0.0.1",
        port=PORT,
        threaded=False,
        debug=False,
    )
