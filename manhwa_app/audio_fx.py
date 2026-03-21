# manhwa_app/audio_fx.py
import logging
import os
import subprocess
import shutil
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

def apply_audio_post_processing(input_wav: str, output_wav: str, config: dict) -> bool:
    """
    Aplica uma cadeia profissional de efeitos no áudio gerado pelo TTS usando FFmpeg.
    Baseado na estrutura "production.audio" do config.json.
    """
    if not os.path.exists(input_wav):
        return False

    # Extrai configs do bloco "production" -> "audio" (ou usa fallbacks da interface antiga)
    prod_conf = config.get("production", {}).get("audio", {})
    
    # Mapeamento dinâmico: se a config nova não existir, olha pra config antiga pra retrocompatibilidade
    do_highpass = prod_conf.get("highpass", config.get("fx_enhancer", False))
    do_lowpass  = prod_conf.get("lowpass", config.get("fx_noise_reduction", False))
    do_deesser  = prod_conf.get("deesser", False)
    do_compand  = prod_conf.get("compressor", config.get("fx_compressor", False))
    do_reverb   = prod_conf.get("reverb", config.get("fx_reverb", False))
    do_silence  = prod_conf.get("remove_silence", False)
    do_loudnorm = prod_conf.get("normalize", config.get("fx_normalize", False))

    filters = []

    # 1. Limpeza de frequências (Highpass/Lowpass)
    if do_highpass:
        filters.append("highpass=f=80")  # Remove rumble/graves inúteis
    if do_lowpass:
        filters.append("lowpass=f=12000") # Limpa chiados de alta frequência

    # 2. De-esser (suaviza sibilância, os "S" fortes)
    if do_deesser:
        # Usa um compander focado em frequências altas para atenuar o S
        filters.append("deesser=i=0.01:f=0.8")

    # 3. Compressão Dinâmica (mantém o volume da narração constante)
    if do_compand:
        # acompressor padrão: ratio 4:1, threshold -15dB
        filters.append("acompressor=threshold=-15dB:ratio=4:attack=5:release=50:makeup=2")

    # 4. Reverb sutil (Espacialização)
    if do_reverb:
        filters.append("aecho=0.8:0.88:40:0.1")

    # 5. Remoção de Silêncio
    if do_silence:
        # Remove silêncios maiores que 0.5s (ajusta o ritmo da fala)
        filters.append("silenceremove=stop_periods=-1:stop_duration=0.5:stop_threshold=-40dB")

    # 6. Normalização Broadcast (Loudnorm)
    # Recomendação YouTube: LUFS -14. Usamos -16 para margem segura, True Peak -1.5
    if do_loudnorm:
        filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")

    # Se nenhum filtro estiver ativo, apenas copia
    if not filters:
        shutil.copy2(input_wav, output_wav)
        return True

    # CORRIGIDO: usar UUID para garantir nome único e evitar colissão com input_wav
    # (se input_wav já terminar em .fx.wav, o nome antigo era idêntico ao input)
    unique_id = uuid.uuid4().hex[:8]
    # CORRIGIDO: inicializar temp_wav antes do try para que o finally sempre possa
    # fazer o cleanup sem NameError caso a exceção ocorra antes da atribuição
    temp_wav = str(Path(input_wav).parent / f"_fx_tmp_{unique_id}.wav")
    filter_string = ",".join(filters)
    
    cmd = [
        "ffmpeg", "-y",
        "-i", input_wav,
        "-af", filter_string,
        "-ac", "1",  # Output mono
        "-ar", "24000", # Padroniza sample rate
        temp_wav
    ]
    
    try:
        # Determine creation flags for subprocess, defaulting to 0
        create_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

        result = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            timeout=300,  # CORRIGIDO: timeout de 5min evita bloqueio infinito
            creationflags=create_flags
        )
        if result.returncode == 0 and os.path.exists(temp_wav):
            shutil.move(temp_wav, output_wav)
            return True
        else:
            raw_err = result.stderr
            err = ""
            if raw_err:
                err = raw_err.decode("utf-8", errors="replace")
                if len(err) > 500:
                    err = err[-500:]
            logger.error(f"Erro no FFmpeg (audio_fx): {err}")
            return False
    except subprocess.TimeoutExpired as exc:
        # CORRIGIDO: subprocess.run com timeout não mata automaticamente o processo
        # O processo FFmpeg fica vivo em background consumindo CPU indefinidamente
        logger.error(f"FFmpeg excedeu timeout de 300s para {input_wav}")
        if exc.output is not None or hasattr(exc, 'proc'):
            pass  # subprocess.run já lida, mas o processo pode ainda estar vivo
        # Matar processo filho explícitamente (Python 3.9+ subprocess.run não mata em Windows)
        try:
            import psutil
            for proc in psutil.process_iter(['name', 'cmdline']):
                if 'ffmpeg' in (proc.info.get('name') or '').lower():
                    cmdline = proc.info.get('cmdline') or []
                    if temp_wav in ' '.join(cmdline):
                        proc.kill()
        except Exception:
            pass
        return False
    except Exception as e:
        logger.error(f"Exceção no processamento FFmpeg: {e}")
        return False
    finally:
        # CORRIGIDO: limpa apenas se o arquivo temporario ainda existir
        # (shutil.move já pode tê-lo removido no sucesso)
        if os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except Exception as e:
                logger.warning(f"Não foi possível remover arquivo temporário {temp_wav}: {e}")

