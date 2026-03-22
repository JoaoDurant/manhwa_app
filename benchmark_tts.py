import sys
import time
import logging
import gc
from pathlib import Path
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent.resolve()))
import engine
from manhwa_app.text_processor import process_text_fluency
from manhwa_app.advanced_text_processor import process_text

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("benchmark")

def run_benchmark():
    logger.info("="*60)
    logger.info(" DIAGNÓSTICO DE GARGALOS TTS (100% PRÁTICO) ")
    logger.info("="*60)
    
    texto_cru = "Hello world, this is a benchmarking test to correctly identify processing delays in text, model loading, audio synthesis, and file saving."
    
    # 1. TEMPO DE PROCESSAMENTO DE TEXTO (NLP / Regex)
    logger.info("\n--- 1. PROCESSAMENTO DE TEXTO ---")
    t0_text = time.time()
    # Simula o fluxo do audio_pipeline.py
    texto_fluente = process_text_fluency(texto_cru, lang="en")
    texto_final = process_text(texto_fluente, {"normalize_text": True, "clean_symbols": True})
    t_text = time.time() - t0_text
    logger.info(f"➜ Tempo de Processamento de Texto: {t_text:.4f}s")
    
    # Limpa VRAM antes das IAs
    engine.unload_kokoro_engine()
    gc.collect()
    
    # 2. TEMPO DE CARREGAMENTO DO MODELO (Disk -> RAM -> GPU/CPU)
    logger.info("\n--- 2. CARREGAMENTO DO MODELO (KOKORO) ---")
    t0_load = time.time()
    # Chama o Kokoro para testar o peso do modelo
    success = engine.load_kokoro_engine(lang_code="a", device="cuda")
    t_load = time.time() - t0_load
    if not success:
        logger.error("Falha ao carregar o modelo.")
        sys.exit(1)
    logger.info(f"➜ Tempo de Carregamento (Pesos + CUDA/Fallback): {t_load:.2f}s")
    
    # 3. TEMPO DE GERAÇÃO DE ÁUDIO (Inferência / Forward Pass)
    logger.info("\n--- 3. GERAÇÃO DE ÁUDIO (SÍNTESE) ---")
    t0_gen = time.time()
    audio_arr, sr = engine.synthesize_kokoro(texto_final, voice="af_heart")
    t_gen = time.time() - t0_gen
    if audio_arr is None:
        logger.error("Falha na geração do áudio.")
        sys.exit(1)
    duracao_audio = len(audio_arr) / sr
    rtf = t_gen / duracao_audio if duracao_audio > 0 else 0
    logger.info(f"➜ Tempo de Geração (Inferência ML): {t_gen:.2f}s (gerou {duracao_audio:.2f}s de áudio)")
    logger.info(f"➜ Fator Tempo-Real (RTF): {rtf:.2fx} (Menor que 1.0 é mais rápido que a fala real)")
    
    # 4. TEMPO DE ESCRITA I/O (RAM -> DISCO)
    logger.info("\n--- 4. ESCRITA DE ARQUIVO (I/O) ---")
    t0_io = time.time()
    out_path = Path("benchmark_output.wav")
    sf.write(str(out_path), audio_arr, sr)
    t_io = time.time() - t0_io
    logger.info(f"➜ Tempo de Salvamento WAV: {t_io:.4f}s")
    if out_path.exists():
        out_path.unlink() # Cleanup
        
    logger.info("\n" + "="*60)
    logger.info(" RESULTADO E ANÁLISE DO GARGALO ")
    logger.info("="*60)
    logger.info(f"1. Texto:       {t_text:.4f}s")
    logger.info(f"2. Modelo Load: {t_load:.2f}s")
    logger.info(f"3. INFERÊNCIA:  {t_gen:.2f}s  <<< GARGALO REAL")
    logger.info(f"4. Salvar I/O:  {t_io:.4f}s")

if __name__ == "__main__":
    run_benchmark()
