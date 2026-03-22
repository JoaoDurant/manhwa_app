import sys
import time
import traceback
import logging
from pathlib import Path
import soundfile as sf
import gc
import torch

# Configura logs detalhados
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("test_kokoro")

# Adiciona o diretório atual ao sys.path para importar engine.py corretamente
sys.path.insert(0, str(Path(__file__).parent.resolve()))

try:
    import engine
    logger.info("Módulo engine importado com sucesso.")
except Exception as e:
    logger.critical(f"Falha ao importar engine.py:\n{traceback.format_exc()}")
    sys.exit(1)

def test_device(device_name: str, text: str = "Hello world, this is a test of the Kokoro TTS engine isolated from the main app."):
    logger.info("=" * 60)
    logger.info(f" INICIANDO TESTE ISOLADO: KOKORO EM {device_name.upper()}")
    logger.info("=" * 60)

    # 1. Liberar memória antes
    engine.unload_kokoro_engine()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Carregar o Modelo
    t0_load = time.time()
    logger.info(f"Tentando carregar Kokoro no device: {device_name}...")
    try:
        # Passar "cuda" ou "cpu" explicitly
        success = engine.load_kokoro_engine(lang_code="a", device=device_name)
        
        if not success:
            logger.error(f"load_kokoro_engine retornou False no passo de carregamento ({device_name}).")
            return False
            
        load_time = time.time() - t0_load
        logger.info(f"✅ Modelo carregado com sucesso em {load_time:.2f}s.")
        
    except Exception as e:
        logger.error(f"❌ Exceção Crítica durante o CARREGAMENTO em {device_name}:\n{traceback.format_exc()}")
        return False

    # 3. Gerar o Áudio
    t0_gen = time.time()
    logger.info(f"Iniciando síntese de áudio. Texto: '{text}'")
    try:
        # Utiliza uma voz AmEn suportada pelo Kokoro (e.g. af_heart)
        audio_array, sample_rate = engine.synthesize_kokoro(
            text=text,
            voice="af_heart",
            speed=1.0,
            lang_code="a"
        )
        
        if audio_array is None:
            logger.error(f"synthesize_kokoro retornou None (falha interna na geração em {device_name}).")
            return False
            
        gen_time = time.time() - t0_gen
        logger.info(f"✅ Áudio gerado com sucesso em {gen_time:.2f}s! ({len(audio_array)} samples, {sample_rate}Hz)")
        
        # 4. Salvar o Áudio
        out_path = f"debug_kokoro_{device_name}.wav"
        sf.write(out_path, audio_array, sample_rate)
        logger.info(f"💾 Áudio salvo em: {out_path}")
        
        return True

    except Exception as e:
        logger.error(f"❌ Exceção Crítica durante a GERAÇÃO em {device_name}:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    logger.info("Iniciando script de isolamento do Kokoro TTS.")
    
    # Testa CUDA primeiro (que sabidamente deve dar erro sm_120 no seu caso, mas o log vai mostrar se o fallback da engine agir)
    logger.info(">>> TESTANDO CHAMADA EXPLÍCITA: CUDA")
    test_device("cuda")
    
    # Testa CPU explicitamente
    logger.info(">>> TESTANDO CHAMADA EXPLÍCITA: CPU")
    test_device("cpu")
    
    logger.info("=" * 60)
    logger.info("TODOS OS TESTES FINALIZADOS.")
