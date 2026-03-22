# workers/indextts_server.py
# Servidor Flask persistente para IndexTTS
# Roda no venv_indextts — completamente isolado do Chatterbox
#
# INICIAR: venv_indextts\Scripts\python workers\indextts_server.py
# PORTA:   5002 (configuravel via variavel de ambiente INDEXTTS_PORT)

import gc
import io
import os
import sys
import logging
import tempfile
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from flask import Flask, request, send_file, jsonify

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [INDEXTTS-WORKER] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Import defensivo do IndexTTS ---
try:
    from indextts.infer_v2 import IndexTTS2 as _IndexTTSClass
    INDEX_VERSION = "v2"
    logger.info("IndexTTS versao 2 disponivel.")
except ImportError:
    try:
        from indextts.infer import IndexTTS as _IndexTTSClass
        INDEX_VERSION = "v1"
        logger.info("IndexTTS versao 1.x disponivel.")
    except ImportError:
        logger.error(
            "IndexTTS nao instalado neste venv.\n"
            "Execute: pip install git+https://github.com/index-tts/index-tts.git"
        )
        sys.exit(1)

# --- Configuracao ---
PORT      = int(os.environ.get("INDEXTTS_PORT", 5002))
MODEL_DIR = os.environ.get("INDEXTTS_MODEL_DIR", "checkpoints")
CFG_PATH  = os.environ.get("INDEXTTS_CFG", "checkpoints/config.yaml")

# --- Estado global ---
model     = None
device    = None
_lock     = threading.Lock()


def _resolve_device() -> str:
    if torch.cuda.is_available():
        try:
            t = torch.tensor([1.0]).cuda()
            t.cpu()
            return "cuda"
        except Exception:
            pass
    return "cpu"


def load_model():
    global model, device

    device   = _resolve_device()
    use_fp16 = (device == "cuda")

    # Otimizacoes CUDA
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        torch.backends.cudnn.benchmark        = False
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # use_cuda_kernel: True = kernels compilados (mais rapido, ~2min 1a vez)
    # use_deepspeed: False no Windows (nao suportado)
    use_kernel = (device == "cuda") and sys.platform != "win32"

    logger.info(
        f"Carregando IndexTTS ({INDEX_VERSION}) | "
        f"device={device} | fp16={use_fp16} | cuda_kernel={use_kernel}"
    )

    if not Path(MODEL_DIR).exists():
        logger.error(
            f"Pasta de modelos nao encontrada: {MODEL_DIR}\n"
            "Baixe com:\n"
            "  python -c \"from huggingface_hub import snapshot_download; "
            "snapshot_download('IndexTeam/IndexTTS-2', local_dir='checkpoints')\""
        )
        sys.exit(1)

    try:
        if INDEX_VERSION == "v2":
            model = _IndexTTSClass(
                cfg_path=CFG_PATH,
                model_dir=MODEL_DIR,
                use_fp16=use_fp16,
                use_cuda_kernel=use_kernel,
                use_deepspeed=False,
            )
        else:
            model = _IndexTTSClass(
                model_dir=MODEL_DIR,
                cfg_path=CFG_PATH,
                is_fp16=use_fp16,
                use_cuda_kernel=use_kernel,
            )
    except Exception as e:
        error_msg = str(e)
        if "no kernel image" in error_msg or "sm_120" in error_msg or "sm_100" in error_msg:
            logger.warning(
                f"INCOMPATIBILIDADE GPU DETECTADA NO INDEX (RTX 50): {error_msg}. "
                "Caindo para CPU..."
            )
            device   = "cpu"
            use_fp16 = False
            if INDEX_VERSION == "v2":
                model = _IndexTTSClass(
                    cfg_path=CFG_PATH,
                    model_dir=MODEL_DIR,
                    use_fp16=False,
                    use_cuda_kernel=False,
                    use_deepspeed=False,
                )
            else:
                model = _IndexTTSClass(
                    model_dir=MODEL_DIR,
                    cfg_path=CFG_PATH,
                    is_fp16=False,
                    use_cuda_kernel=False,
                )
        else:
            raise e

    logger.info("Modelo carregado. Executando warmup...")
    _warmup()
    logger.info(f"IndexTTS pronto na porta {PORT}.")


def _warmup():
    """
    IndexTTS precisa de um prompt WAV real para warmup.
    Procura o primeiro .wav em presets/ ou pula se nao encontrar.
    """
    warmup_prompt = None
    for folder in ["presets", "test_data", "."]:
        wavs = list(Path(folder).glob("*.wav")) if Path(folder).exists() else []
        if wavs:
            warmup_prompt = str(wavs[0])
            break

    if not warmup_prompt:
        logger.warning("Warmup pulado — nenhum WAV encontrado em presets/")
        return

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        if INDEX_VERSION == "v2":
            model.infer(spk_audio_prompt=warmup_prompt, text="Hello.", output_path=tmp)
        else:
            model.infer(voice=warmup_prompt, text="Hello.", output_path=tmp)
        Path(tmp).unlink(missing_ok=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.info(f"Warmup concluido usando: {warmup_prompt}")
    except Exception as e:
        logger.warning(f"Warmup falhou (nao critico): {e}")
        try:
            Path(tmp).unlink(missing_ok=True)
        except Exception:
            pass


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
        "version": INDEX_VERSION,
        "device":  device,
        "vram":    vram_info,
    })


@app.route("/synthesize", methods=["POST"])
def synthesize():
    """
    POST /synthesize
    Body JSON: {
        "text":             "texto a sintetizar",
        "audio_prompt_path": "presets/minha_voz.wav"   (obrigatorio)
    }
    Retorna: audio/wav binario

    IndexTTS salva em arquivo .wav internamente — este endpoint le esse
    arquivo, retorna os bytes e deleta o temporario automaticamente.
    """
    data              = request.get_json(silent=True) or {}
    text              = (data.get("text") or "").strip()
    audio_prompt_path = (data.get("audio_prompt_path") or "").strip()

    if not text:
        return jsonify({"error": "text is required"}), 400
    if not audio_prompt_path:
        return jsonify({"error": "audio_prompt_path is required"}), 400
    if not Path(audio_prompt_path).exists():
        return jsonify({"error": f"prompt file not found: {audio_prompt_path}"}), 400

    with _lock:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name

            if INDEX_VERSION == "v2":
                model.infer(
                    spk_audio_prompt=audio_prompt_path,
                    text=text,
                    output_path=tmp_path,
                )
            else:
                model.infer(
                    voice=audio_prompt_path,
                    text=text,
                    output_path=tmp_path,
                )

            audio_np, sr = sf.read(tmp_path, dtype="float32")
            if audio_np.ndim == 2:
                audio_np = audio_np.mean(axis=1)

            buf = io.BytesIO()
            sf.write(buf, audio_np, sr, format="WAV")
            buf.seek(0)

            return send_file(buf, mimetype="audio/wav")

        except Exception as e:
            logger.error(f"Erro na sintese: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    load_model()
    app.run(
        host="127.0.0.1",
        port=PORT,
        threaded=False,
        debug=False,
    )
