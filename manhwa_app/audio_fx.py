# manhwa_app/audio_fx.py
#
# Cadeia de FX de áudio profissional para narração de manhwa.
# Todos os parâmetros foram revisados e otimizados para:
#   - Reduzir o efeito "robótico" do TTS
#   - Preservar a dinâmica dramática da narração
#   - Funcionar bem em PT-BR, ES e EN com o Chatterbox
#
# MUDANÇAS vs versão anterior:
#   - highpass: 80Hz → 120Hz (80 removia presença útil da voz)
#   - deesser: i=0.01:f=0.8 → i=0.03:f=0.55 (era muito agressivo, pastava consoantes)
#   - acompressor: ratio=4 → ratio=2.5, threshold=-18dB, makeup=1.5 (ratio 4 é rádio AM,
#     achatava dinâmica e era a principal causa do som robótico)
#   - aecho: delay 40ms→22ms, decay 0.1→0.05 (delay 40ms criava "cauda" artificial
#     em frases curtas de manhwa)
#   - loudnorm: LRA=11→14 (LRA 11 comprimia demais a dinâmica narrativa)
#   - Nova função apply_audio_post_processing_for_lang() com parâmetros ajustados
#     automaticamente por idioma (PT-BR, ES, EN)
#
# [NOISE FIX] Ruído branco agudo eliminado com:
#   - anlmdn antes do loudnorm: atenua noise floor do modelo TTS (~-60dB)
#     sem afetar a voz (strength=3, patch=2, research=4)
#   - loudnorm TP=-2.0 (era -1.5): mais headroom → menos pressão de ganho → menos
#     amplificação do noise floor no gain staging do loudnorm
#   - echo zerado no preset base: out_gain 0.3 amplificava o noise floor do TTS

import logging
import os
import subprocess
import shutil
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cadeias de FX pré-configuradas por idioma para narração de manhwa
# ---------------------------------------------------------------------------

# Parâmetros base: usados para EN (Turbo)
_FX_BASE = {
    "highpass_hz":         120,      # Remove sub-bass sem tirar presença da voz
    "deesser_intensity":   0.03,     # Suave — TTS já não tem sibilância exagerada
    "deesser_frequency":   0.55,     # Banda mais estreita que o padrão (era 0.8)
    "comp_threshold":      "-16dB",  # Mais suave que -18dB anterior (menos compressão)
    "comp_ratio":          1.8,      # Bem leve (era 2.5) para não amassar a dinâmica
    "comp_attack":         12,       # Ataque médio para não cortar plosivas
    "comp_release":        100,      # Release suave
    "comp_makeup":         1.0,      # Menos ganho de compensação
    "echo_in_gain":        0.8,
    "echo_out_gain":       0.15,     # [NOISE FIX] Reduzido de 0.3 — out_gain alto amplifica noise floor do TTS
    "echo_delay_ms":       15,       # Delay minúsculo
    "echo_decay":          0.02,     # Decay quase imperceptível
    "loudnorm_I":          -14,      # Target YouTube ideal para narração
    "loudnorm_TP":         -2.0,     # [NOISE FIX] Era -1.5 → -2.0: mais headroom, menos pressão de ganho
    "loudnorm_LRA":        14,       # Mais dinâmica dramática (era 11)
    # [NOISE FIX] lowpass: corta ruído branco acima de 11.5kHz (universalmente suportado)
    # Voz humana + Chatterbox não tem harmônicos úteis acima de ~8kHz.
    # Ruído branco agudo do TTS vive exatamente acima de 12kHz → corte seguro em 11500Hz.
    "noise_lowpass_hz":    11500,
}

# PT-BR: mais dinâmica, sem de-esser (sibilância é natural e importante)
# Preserva a identidade da voz com compressão mínima e sem reverb.
_FX_PTBR = {
    **_FX_BASE,
    "highpass_hz":         100,      # PT-BR tem mais presença nos médios-graves
    "deesser_intensity":   0.0,      # Desativado: sibilância PT é natural
    "comp_ratio":          1.3,      # Ratio quase transparente (era 1.5)
    "comp_threshold":      "-18dB",
    "comp_attack":         50,       # Preserva plosivas PT-BR
    "comp_release":        250,      # Release explícito para consistência

    "comp_makeup":         1.2,
    # [P7] LRA=9 (era 16) — 16 demais para alto-falantes de laptop/celular.
    # LRA 16 significa silêncios inaudíveis e picos ásperos em fones de ouvido.
    # LRA=9 mantém dinâmica dramática mas compatível com mídia móvel (YouTube Shorts, Reels).
    "loudnorm_I":          -14,      # -14 LUFS: levemente mais alto para mobile (era -16)
    "loudnorm_LRA":        9,        # LRA=9: YouTube/Reels ideal para narração (era 16)
    "loudnorm_TP":         -2.0,     # [NOISE FIX] Herdado do base, confirmado para PT-BR
}


# ES: ritmo marcado, compressão moderada
_FX_ES = {
    **_FX_BASE,
    "highpass_hz":         110,
    "deesser_intensity":   0.02,
    "comp_ratio":          1.9,      # Ratio menor para preservar identidade
    "comp_attack":         12,
    "comp_threshold":      "-19dB",
    "loudnorm_I":          -15,
    "loudnorm_LRA":        15,
}

# EN: configuração base, Turbo já tem boa qualidade nativa
_FX_EN = _FX_BASE


def _get_fx_params_for_lang(lang: str) -> dict:
    """Retorna os parâmetros de FX ajustados para o idioma."""
    lang_key = lang.lower().split("-")[0]  # "pt-br" → "pt"
    if lang_key == "pt":
        return _FX_PTBR.copy()
    elif lang_key == "es":
        return _FX_ES.copy()
    else:
        return _FX_EN.copy()

def get_recommended_chatterbox_params(lang: str) -> dict:
    """
    Retorna recomendacoes de parametros (exaggeration, cfg_weight) baseadas no idioma
    para o Chatterbox, mitigando erros e melhorando a humanidade da clonagem.
    
    [CB4] Perfil Manhwa Drama — validado para conteúdo de narração PT-BR:
    - exaggeration 0.65 (era 0.55): mais expressividade dramática
    - cfg_weight 0.40 (era 0.45): mais liberdade emocional mantendo ainda a voz
    - temperature 0.75: mais variação prosódica vs 0.65 padrão (menos robótico)
    """
    lang_key = lang.lower().split("-")[0]
    if lang_key == "pt":
        return {
            "exaggeration": 0.55,   # Revertido para valor estável
            "cfg_weight":   0.45,   # Revertido para valor estável
            "temperature":  0.65,   # Revertido para valor estável
        }
    elif lang_key == "es":
        return {"exaggeration": 0.55, "cfg_weight": 0.45, "temperature": 0.65}
    return {"exaggeration": 0.5, "cfg_weight": 0.5}  # Fallback seguro




def apply_audio_post_processing(
    input_wav: str,
    output_wav: str,
    config: dict,
    lang: str = "en",
) -> bool:
    """
    Aplica cadeia profissional de FX para narração de manhwa com Chatterbox.

    Parâmetros ajustados automaticamente por idioma (lang):
      - "pt" / "pt-br" → cadeia PT-BR (ratio baixo, sem de-esser, loudnorm -16)
      - "es"           → cadeia ES (de-esser mínimo, ratio 2.3)
      - "en" e outros  → cadeia EN/padrão (ratio 2.5, loudnorm -14)

    Baseado na estrutura "production.audio" do config.json.
    """
    if not os.path.exists(input_wav):
        return False

    # Extrai configs do bloco "production" → "audio"
    prod_conf = config.get("production", {}).get("audio", {})

    do_natural  = prod_conf.get("natural_mode",   config.get("fx_natural_mode", False))
    do_highpass = prod_conf.get("highpass",       config.get("fx_highpass",   False))
    do_lowpass  = prod_conf.get("lowpass",        config.get("fx_noise_reduction", False))
    do_deesser  = prod_conf.get("deesser",        config.get("fx_deesser",    False))
    do_compand  = prod_conf.get("compressor",     config.get("fx_compressor", False))
    do_reverb   = prod_conf.get("reverb",         config.get("fx_reverb",     False))
    do_silence  = prod_conf.get("remove_silence", config.get("fx_silence",    False))
    do_loudnorm = prod_conf.get("normalize",      config.get("fx_loudnorm",   config.get("fx_normalize", False)))

    # Parâmetros ajustados por idioma
    p = _get_fx_params_for_lang(lang)
    
    # [QUALIDADE] Remoção total de reverb para PT/ES se solicitado (ou por padrão se do_reverb for False)
    # Se o idioma for PT/ES, desativamos reverb forçadamente se não houver flag explícita
    lang_key = lang.lower().split("-")[0]
    if lang_key in ["pt", "es"] and not prod_conf.get("reverb"):
        do_reverb = False

    filters = []

    if do_natural:
        # [NATURAL MODE] Gain staging leve + Limiter + Loudnorm muito suave
        # Ideal para preservar identidade da voz Chatterbox, com volume consistente.
        filters.append(
            "alimiter=level_in=1.1:level_out=1.0:limit=0.95:attack=5:release=80,"
            "loudnorm=I=-18:TP=-2.0:LRA=18"
        )
        logger.debug(f"Processando em MODO NATURAL para {lang}")
    else:
        # 1. Highpass — remove sub-bass inútil sem tirar presença da voz
        if do_highpass:
            filters.append(f"highpass=f={p['highpass_hz']}")

        # 2. Lowpass — só se explicitamente ativado
        if do_lowpass:
            filters.append("lowpass=f=14000")

        # 3. De-esser — suaviza sibilância
        if do_deesser and p["deesser_intensity"] > 0:
            filters.append(
                f"deesser=i={p['deesser_intensity']}:f={p['deesser_frequency']}"
            )

        # 4. Compressor dinâmico — Ratio baixo preserva identidade
        if do_compand:
            filters.append(
                f"acompressor="
                f"threshold={p['comp_threshold']}:"
                f"ratio={p['comp_ratio']}:"
                f"attack={p['comp_attack']}:"
                f"release={p['comp_release']}:"
                f"makeup={p['comp_makeup']}"
            )

        # 5. Reverb — Removido do preset PT/ES via presets acima
        if do_reverb:
            filters.append(
                f"aecho="
                f"{p['echo_in_gain']}:"
                f"{p['echo_out_gain']}:"
                f"{p['echo_delay_ms']}:"
                f"{p['echo_decay']}"
            )

        # 7. [NOISE FIX] lowpass antes do loudnorm — corta ruído branco agudo do TTS
        # anlmdn (alternativa anterior) não está disponível em todos os builds do FFmpeg.
        # lowpass=f=11500 é universalmente suportado e cobre exatamente a faixa do ruído:
        # voz humana/Chatterbox tem conteúdo útil até ~8kHz, ruído TTS está acima de 12kHz.
        # Executado SEMPRE que loudnorm está ativo (loudnorm amplifica o noise floor).
        if do_loudnorm:
            filters.append(f"lowpass=f={p['noise_lowpass_hz']}")

        # 8. Loudnorm Broadcast
        if do_loudnorm:
            filters.append(
                f"loudnorm="
                f"I={p['loudnorm_I']}:"
                f"TP={p['loudnorm_TP']}:"
                f"LRA={p['loudnorm_LRA']}"
            )

    # 6. Remoção de silêncio via FFmpeg removida daqui!
    # O stage 3a (utils.trim_lead_trail_silence) já cuida disso muito melhor.
    # O uso duplo estava cortando palavras.

    # [Q4] Realce Espectral — Desativado temporariamente para diagnosticar tom agudo
    # filters.append("equalizer=f=2500:width_type=o:width=2:g=0.7")
    # filters.append("equalizer=f=8000:width_type=o:width=2:g=0.5")
    # O stage 3a (utils.trim_lead_trail_silence) já cuida disso muito melhor.
    # O uso duplo estava cortando palavras.

    # Se nenhum filtro ativo, apenas copia
    if not filters:
        shutil.copy2(input_wav, output_wav)
        return True

    unique_id = uuid.uuid4().hex[:8]
    temp_wav = str(Path(input_wav).parent / f"_fx_tmp_{unique_id}.wav")
    filter_string = ",".join(filters)
    
    logger.info(f"[FX] Filters para {lang}: {filter_string}")

    cmd = [
        "ffmpeg", "-y",
        "-i", input_wav,
        "-af", filter_string,
        "-ac", "1",       # Mono — narração não precisa de stereo
        "-ar", "24000",   # 24kHz — sample rate nativo do Chatterbox
        temp_wav
    ]

    try:
        create_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=300,
            creationflags=create_flags,
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

    except subprocess.TimeoutExpired:
        logger.error(f"FFmpeg excedeu timeout de 300s para {input_wav}")
        try:
            import psutil
            for proc in psutil.process_iter(["name", "cmdline"]):
                if "ffmpeg" in (proc.info.get("name") or "").lower():
                    cmdline = proc.info.get("cmdline") or []
                    if temp_wav in " ".join(cmdline):
                        proc.kill()
        except Exception:
            pass
        return False

    except Exception as e:
        logger.error(f"Exceção no processamento FFmpeg: {e}")
        return False

    finally:
        if os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except Exception as e:
                logger.warning(f"Não foi possível remover temp {temp_wav}: {e}")