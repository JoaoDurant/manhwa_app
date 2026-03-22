# engines/index-tts/tts_worker.py
# Worker IndexTTS — roda via: uv run python tts_worker.py
# Protocolo: JSON por linha no stdin/stdout
#
# Formato de request (stdin, uma linha JSON por chunk):
#   {"text": "...", "prompt": "path/to/voice.wav", "output": "path/to/out.wav"}
#   {"cmd": "exit"}    <- para encerrar o worker
#
# Formato de response (stdout, uma linha JSON por chunk):
#   {"status": "ok",    "output": "path/to/out.wav", "duration": 2.34}
#   {"status": "error", "message": "erro aqui"}
#   {"status": "ready"} <- enviado uma vez quando o modelo esta carregado

import json
import sys
import os
import time
import tempfile
from pathlib import Path

# IndexTTS requer PYTHONPATH com o raiz do repo
sys.path.insert(0, str(Path(__file__).parent))

def _log(msg: str):
    """Log para stderr — nao interfere com o protocolo JSON no stdout."""
    print(f"[INDEXTTS-WORKER] {msg}", file=sys.stderr, flush=True)


def _load_model():
    """
    Carrega IndexTTS com as otimizacoes para RTX 5070 Ti.
    uv garante que todas as dependencias corretas estao no .venv.
    """
    import torch

    # Otimizacoes CUDA (mesmo padrao do engine.py existente)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        torch.backends.cudnn.benchmark        = False
        try:
            # FIX: RTX 50 Blackwell detect check
            if torch.cuda.get_device_capability(0) >= (12, 0):
                 _log("RTX 50 (Blackwell) detectada. Verificando suporte kernel...")
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        try:
            gpu   = torch.cuda.get_device_name(0)
            vram  = torch.cuda.get_device_properties(0).total_memory / 1e9
            free  = torch.cuda.mem_get_info()[0] / 1e9
            _log(f"GPU: {gpu} | VRAM: {vram:.1f}GB total, {free:.1f}GB livre")
        except Exception:
            pass

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = (device == "cuda")
    # use_cuda_kernel=True compila kernels na primeira execucao (~2min)
    # Em Windows pode falhar — usar False como fallback seguro
    use_kernel = (device == "cuda") and (sys.platform != "win32")

    # Detectar versao disponivel (v2 > v1)
    try:
        from indextts.infer_v2 import IndexTTS2
        version = "v2"
        _log("IndexTTS versao 2 detectada.")
    except ImportError:
        try:
            from indextts.infer import IndexTTS as IndexTTS2
            version = "v1"
            _log("IndexTTS versao 1.x detectada.")
        except ImportError:
            _log("ERRO: IndexTTS nao instalado corretamente no venv.")
            sys.exit(1)

    model_dir = str(Path(__file__).parent / "checkpoints")
    cfg_path  = str(Path(__file__).parent / "checkpoints" / "config.yaml")

    if not Path(model_dir).exists():
        _log(f"ERRO: checkpoints/ nao encontrado em {model_dir}")
        _log("Baixe com: uv tool run huggingface-cli download IndexTeam/IndexTTS-2 --local-dir=checkpoints")
        sys.exit(1)

    _log(f"Carregando IndexTTS ({version}) | device={device} | fp16={use_fp16}")

    try:
        if version == "v2":
            model = IndexTTS2(
                cfg_path=cfg_path,
                model_dir=model_dir,
                use_fp16=use_fp16,
                use_cuda_kernel=use_kernel,
                use_deepspeed=False,   # False no Windows
            )
        else:
            model = IndexTTS2(
                model_dir=model_dir,
                cfg_path=cfg_path,
                is_fp16=use_fp16,
                use_cuda_kernel=use_kernel,
            )
    except Exception as e:
        err_msg = str(e).lower()
        if "no kernel image" in err_msg or "sm_120" in err_msg:
            _log(f"INCOMPATIBILIDADE GPU DETECTADA (sm_120). Fallback para CPU...")
            device = "cpu"
            use_fp16 = False
            if version == "v2":
                model = IndexTTS2(cfg_path=cfg_path, model_dir=model_dir, use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)
            else:
                model = IndexTTS2(model_dir=model_dir, cfg_path=cfg_path, is_fp16=False, use_cuda_kernel=False)
        else:
            raise e

    _log(f"Modelo carregado em {device}. Executando warmup...")
    _warmup(model, version)
    return model, version


def _warmup(model, version: str):
    """Warmup com primeiro WAV de referencia encontrado em presets/ ou checkpoints/."""
    warmup_prompt = None
    # Procurar na pasta presets/ do projeto raiz (dois niveis acima)
    # O worker roda em engines/index-tts/
    root = Path(__file__).parent.parent.parent
    for folder in [root / "presets", root / "test_data", Path(".")]:
        wavs = list(Path(folder).glob("*.wav")) if Path(folder).exists() else []
        if wavs:
            warmup_prompt = str(wavs[0])
            break

    if not warmup_prompt:
        _log("Warmup pulado — nenhum WAV encontrado em presets/")
        return

    try:
        import torch
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        if version == "v2":
            model.infer(spk_audio_prompt=warmup_prompt, text="Hello.", output_path=tmp)
        else:
            model.infer(voice=warmup_prompt, text="Hello.", output_path=tmp)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        Path(tmp).unlink(missing_ok=True)
        _log(f"Warmup concluido usando: {warmup_prompt}")
    except Exception as e:
        _log(f"Warmup falhou (nao critico): {e}")
        try:
            Path(tmp).unlink(missing_ok=True)
        except Exception:
            pass


def _run_worker():
    """
    Loop principal do worker.
    Le requests JSON do stdin, sintetiza, responde JSON no stdout.
    """
    model, version = _load_model()

    # Sinalizar que o modelo esta pronto — o engine.py aguarda esta linha
    print(json.dumps({"status": "ready"}), flush=True)
    _log("Pronto para receber requests.")

    import torch

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            print(json.dumps({"status": "error", "message": f"JSON invalido: {e}"}), flush=True)
            continue

        # Comando de saida
        if req.get("cmd") == "exit":
            _log("Recebido cmd=exit. Encerrando.")
            break

        text   = (req.get("text") or "").strip()
        prompt = (req.get("prompt") or "").strip()
        output = (req.get("output") or "").strip()

        if not text:
            print(json.dumps({"status": "error", "message": "text is required"}), flush=True)
            continue
        if not prompt or not Path(prompt).exists():
            print(json.dumps({"status": "error", "message": f"prompt nao encontrado: {prompt}"}), flush=True)
            continue
        if not output:
            print(json.dumps({"status": "error", "message": "output path is required"}), flush=True)
            continue

        t_start = time.monotonic()
        try:
            # IndexTTS salva em arquivo — usar output direto
            Path(output).parent.mkdir(parents=True, exist_ok=True)

            if version == "v2":
                model.infer(
                    spk_audio_prompt=prompt,
                    text=text,
                    output_path=output,
                )
            else:
                model.infer(
                    voice=prompt,
                    text=text,
                    output_path=output,
                )

            elapsed = time.monotonic() - t_start

            # Verificar se o arquivo foi gerado
            if not Path(output).exists():
                raise RuntimeError(f"Arquivo de saida nao foi criado: {output}")

            # Medir duracao do audio gerado
            duration = None
            try:
                import soundfile as sf
                info = sf.info(output)
                duration = round(info.frames / info.samplerate, 2)
            except Exception:
                pass

            print(json.dumps({
                "status":   "ok",
                "output":   output,
                "duration": duration,
                "elapsed":  round(elapsed, 2),
            }), flush=True)

        except Exception as e:
            elapsed = time.monotonic() - t_start
            _log(f"Erro na sintese: {e}")
            print(json.dumps({
                "status":  "error",
                "message": str(e),
                "elapsed": round(elapsed, 2),
            }), flush=True)
            # Limpar arquivo parcial se existir
            if output and Path(output).exists():
                try:
                    Path(output).unlink()
                except Exception:
                    pass

    _log("Worker encerrado.")


if __name__ == "__main__":
    _run_worker()
