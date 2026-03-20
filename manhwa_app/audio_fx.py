# manhwa_app/audio_fx.py
import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

def apply_audio_post_processing(input_wav: str, output_wav: str, config: dict) -> bool:
    """
    Aplica uma cadeia de efeitos no áudio gerado pelo TTS usando FFmpeg,
    e faz a normalização de pico via PyDub (se solicitado).
    """
    if not os.path.exists(input_wav):
        return False

    filters = []

    # Noise Reduction (leve)
    if config.get("fx_noise_reduction", False):
        filters.append("afftdn=nf=-20")

    # Enhancer (Krisp-like) - Highpass + forte gate/denoise + EQ no vocal (presença)
    if config.get("fx_enhancer", False):
        filters.append("highpass=f=80")
        filters.append("afftdn=nf=-25")
        filters.append("equalizer=f=3000:t=q:w=1:g=3") # Boost de presença

    # Compressor
    if config.get("fx_compressor", False):
        # Aumentar rms volume sutilmente, threshold -15dB
        filters.append("acompressor=threshold=-15dB:ratio=4:attack=5:release=50:makeup=2")

    # EQ (Grave + Presença)
    if config.get("fx_eq", False):
        # Boost grave (120Hz) e Boost presença (4000Hz)
        filters.append("equalizer=f=120:t=q:w=1:g=2")
        filters.append("equalizer=f=4000:t=q:w=1:g=2")

    # Reverb (leve)
    if config.get("fx_reverb", False):
        # Echo curto/slapback para simular sala
        filters.append("aecho=0.8:0.88:40:0.1")

    # Se não houver filtros FFmpeg E não houver Normalize, retorna original
    need_normalize = config.get("fx_normalize", False)

    if not filters and not need_normalize:
        import shutil
        shutil.copy2(input_wav, output_wav)
        return True

    temp_wav = input_wav
    
    # 1. Executa FFmpeg chain (se houver)
    if filters:
        temp_wav = str(Path(input_wav).with_suffix('.fx.wav'))
        filter_str = ",".join(filters)
        cmd = [
            "ffmpeg", "-y",
            "-i", input_wav,
            "-af", filter_str,
            "-ac", "1", # Output mono
            temp_wav
        ]
        try:
            subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            )
        except Exception as e:
            logger.error(f"Erro no FFmpeg: {e}")
            return False

    # 2. Normalização com PyDub (-1 dB)
    if need_normalize:
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(temp_wav)
            
            # Pydub target dBFS (peak) is normally 0. We want -1.0 dBFS
            # Difference from current peak
            diff = audio.max_dBFS - (-1.0)
            target_gain = -diff
            
            # Apply gain if not purely digital zero
            if audio.max_dBFS != float('-inf'):
                audio = audio.apply_gain(target_gain)
            
            audio.export(output_wav, format="wav")
            
        except ImportError:
            logger.warning("Pydub não instalado. Aplicando cópia simples em invés de normalize.")
            import shutil
            shutil.copy2(temp_wav, output_wav)
        except Exception as e:
            logger.error(f"Erro no PyDub normalize: {e}")
            return False
            
    elif filters:
        # Renomeia tmp processado pelo FFmpeg para output_wav
        import shutil
        shutil.move(temp_wav, output_wav)

    # Limpeza
    if temp_wav != input_wav and temp_wav != output_wav and os.path.exists(temp_wav):
        try:
            os.remove(temp_wav)
        except:
            pass

    return True
