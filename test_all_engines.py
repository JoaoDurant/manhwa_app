# test_all_engines.py
# Benchmark de todos os modelos e submodelos TTS
# Verifica performance, carregamento e VRAM.

import sys
import os
import time
import torch
import gc
from pathlib import Path

# Add current dir and index-tts to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "index-tts"))

try:
    import engine as _engine
    import utils as _utils
except Exception as e:
    print(f"[!] Erro ao importar engine/utils: {e}")
    _engine = None
    _utils = None

# Texto padrão para o teste (mesmo tamanho para comparação justa)
TEST_TEXT = (
    "In the heart of the ancient forest, a mysterious portal began to glow. "
    "A lone warrior approached, feeling the power radiating from the stone archway."
)

# Configuração de vozes
VOICE_REF = None
candidates = [
    "presets/Ryan.wav", 
    "presets/default.wav", 
    "test_data/ref.wav", 
    "test_qwen_output.wav"
]
for candidate in candidates:
    if Path(candidate).exists():
        VOICE_REF = str(Path(candidate).resolve())
        break

if not VOICE_REF:
    # Criar um dummy WAV se necessário ou usar o primeiro encontrado
    wav_files = list(Path("presets/").glob("*.wav"))
    if wav_files:
        VOICE_REF = str(wav_files[0].resolve())

def benchmark_engine(engine_name, **kwargs):
    print(f"\n{'='*60}")
    print(f" TESTING ENGINE: {engine_name.upper()}")
    print(f"{'='*60}")
    
    if not _engine or not _utils:
        print("[!] Erro: engine ou utils nao foram importados corretamente.")
        return {"engine": engine_name, "status": "IMPORT ERROR"}

    # 1. Garantir VRAM limpa antes de carregar
    _engine.unload_all_for_switch()
    
    output_file = f"output_{engine_name}.wav"
    
    # 2. Primeira síntese (inclui CARREGAMENTO + WARMUP)
    start_total = time.time()
    success = _utils.generate_paragraph_audio(
        text=TEST_TEXT,
        output_path=output_file,
        engine_name=engine_name,
        audio_prompt_path=VOICE_REF,
        **kwargs
    )
    end_total = time.time()
    
    if success:
        load_and_gen_time = end_total - start_total
        print(f" [1st Run] (Load + Gen): {load_and_gen_time:.2f}s")
        
        # 3. Segunda síntese (APENAS GERAÇÃO - Singleton check)
        start_gen = time.time()
        _utils.generate_paragraph_audio(
            text=TEST_TEXT,
            output_path=output_file,
            engine_name=engine_name,
            audio_prompt_path=VOICE_REF,
            **kwargs
        )
        end_gen = time.time()
        pure_gen_time = end_gen - start_gen
        print(f" [2nd Run] (Pure Gen): {pure_gen_time:.2f}s")
        
        return {
            "engine": engine_name,
            "total_1st": load_and_gen_time,
            "pure_gen": pure_gen_time,
            "status": "OK"
        }
    else:
        print(f" [!] FAILED to generate with {engine_name}")
        return {
            "engine": engine_name,
            "status": "FAILED"
        }

def run_all_benchmarks():
    results = []
    
    if not VOICE_REF:
        print("[ERRO] Nenhum áudio de referência encontrado em presets/. Não é possível testar Chatterbox/IndexTTS.")
        return

    print(f"Usando áudio de referência: {VOICE_REF}")

    # Lista de motores para testar
    engines_to_test = [
        ("chatterbox", {}),
        ("chatterbox_turbo", {}),
        ("multilingual", {}),
        ("kokoro", {}),
        ("qwen", {"qwen_speaker": "Ryan"}),
        ("indextts", {"indextts_speed": 1.0})
    ]

    for name, args in engines_to_test:
        try:
            res = benchmark_engine(name, **args)
            results.append(res)
        except Exception as e:
            print(f" [CRASH] Engine {name} falhou catastroficamente: {e}")
            results.append({"engine": name, "status": f"CRASH: {e}"})

    print(f"\n\n{'#'*60}")
    print(" FINAL RESULTS SUMMARY")
    print(f"{'#'*60}")
    print(f"{'Engine':<20} | {'Status':<10} | {'1st (s)':<10} | {'2nd (s)':<10}")
    print("-" * 60)
    for r in results:
        if r["status"] == "OK":
            print(f"{r['engine']:<20} | {r['status']:<10} | {r['total_1st']:<10.2f} | {r['pure_gen']:<10.2f}")
        else:
            print(f"{r['engine']:<20} | {r['status']:<10} | {'-':<10} | {'-':<10}")

if __name__ == "__main__":
    run_all_benchmarks()
