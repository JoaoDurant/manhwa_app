import sys
import os
import time
import logging

# --- POLYFILLS FOR COMPATIBILITY ---
try:
    import transformers.generation.logits_process as lp
    if not hasattr(lp, "UnnormalizedLogitsProcessor"):
        class UnnormalizedLogitsProcessor:
            def __init__(self, *args, **kwargs): pass
            def __call__(self, input_ids, scores, **kwargs): return scores
        lp.UnnormalizedLogitsProcessor = UnnormalizedLogitsProcessor
    if not hasattr(lp, "MinPLogitsWarper"):
        class MinPLogitsWarper:
            def __init__(self, *args, **kwargs): pass
            def __call__(self, input_ids, scores, **kwargs): return scores
        lp.MinPLogitsWarper = MinPLogitsWarper
except ImportError:
    pass

# -----------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark")

# Change directory to the root to ensure everything loads properly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import (
    load_model, synthesize,
    load_qwen_model, synthesize_qwen, unload_qwen_model, QWEN3_AVAILABLE,
    load_indextts_model, synthesize_indextts, unload_indextts_model, INDEX_TTS_AVAILABLE,
    MULTILINGUAL_AVAILABLE,
    unload_all_for_switch, config_manager
)

def format_time(seconds):
    return f"{seconds:.2f}s"

def run_benchmark():
    test_text = "Hello world. This is a benchmark test for the Manhwa Video Creator TTS engines. We are testing how fast the audio generation is."
    
    # Patch config_manager to allow dynamic model switching
    original_get_string = config_manager.get_string
    CURRENT_MODEL = "original"
    
    def patched_get_string(key, default=None, **kwargs):
        if key == "model.repo_id":
            return CURRENT_MODEL
        return original_get_string(key, default, **kwargs)
        
    config_manager.get_string = patched_get_string

    print("\n" + "="*50)
    print(" BENCKMARKING TTS MODELS")
    print("="*50)

    # 1. Chatterbox (Original/Multilingual)
    if True:  # Chatterbox is always available as the core engine
        print("\n--- Testing Chatterbox (Original) ---")
        unload_all_for_switch()
        CURRENT_MODEL = "original"
        
        t0 = time.time()
        load_model()
        t_load_cb = time.time() - t0
        print(f" Load time: {format_time(t_load_cb)}")
        
        try:
            t0 = time.time()
            audio_data, sr = synthesize(test_text)
            t_gen_cb = time.time() - t0
            if audio_data is not None:
                print(f" Generation time: {format_time(t_gen_cb)}, length: {len(audio_data)/sr:.2f}s audio.")
            else:
                print(f" Chatterbox Original synthesis returned None.")
        except Exception as e:
            print(f" Chatterbox Original generation failed: {e}")

        if MULTILINGUAL_AVAILABLE:
            print("\n--- Testing Chatterbox (Multilingual) ---")
            unload_all_for_switch()
            CURRENT_MODEL = "multilingual"
            
            t0 = time.time()
            load_model()
            t_load_multi = time.time() - t0
            print(f" Load time: {format_time(t_load_multi)}")
            
            try:
                t0 = time.time()
                audio_data, sr = synthesize(test_text)
                t_gen_multi = time.time() - t0
                if audio_data is not None:
                    print(f" Generation time: {format_time(t_gen_multi)}, length: {len(audio_data)/sr:.2f}s audio.")
                else:
                    print(f" Chatterbox Multilingual synthesis returned None.")
            except Exception as e:
                print(f" Chatterbox Multilingual generation failed: {e}")
    else:
        print("\n--- Chatterbox NOT AVAILABLE ---")

    # 2. Qwen TTS
    if QWEN3_AVAILABLE:
        print("\n--- Testing Qwen3-TTS ---")
        unload_all_for_switch()
        
        t0 = time.time()
        load_qwen_model()
        t_load_qwen = time.time() - t0
        print(f" Load time: {format_time(t_load_qwen)}")
        
        try:
            t0 = time.time()
            audio_data, sr = synthesize_qwen(test_text)
            t_gen_qwen = time.time() - t0
            if audio_data is not None:
                print(f" Generation time: {format_time(t_gen_qwen)}, length: {len(audio_data)/sr:.2f}s audio.")
            else:
                print(f" Qwen3-TTS synthesis returned None.")
        except Exception as e:
            print(f" Qwen3-TTS generation failed: {e}")
    else:
        print("\n--- Qwen3-TTS NOT AVAILABLE ---")

    # 3. IndexTTS
    if INDEX_TTS_AVAILABLE:
        print("\n--- Testing IndexTTS ---")
        unload_all_for_switch()
        
        t0 = time.time()
        load_indextts_model()
        t_load_index = time.time() - t0
        print(f" Load time: {format_time(t_load_index)}")
        
        try:
            t0 = time.time()
            audio_data, sr = synthesize_indextts(test_text)
            t_gen_index = time.time() - t0
            if audio_data is not None:
                print(f" Generation time: {format_time(t_gen_index)}, length: {len(audio_data)/sr:.2f}s audio.")
            else:
                print(f" IndexTTS synthesis returned None.")
        except Exception as e:
            print(f" IndexTTS generation failed: {e}")
    else:
        print("\n--- IndexTTS NOT AVAILABLE ---")

    print("\n" + "="*50)
    print(" BENCHMARK COMPLETE")
    print("="*50)


if __name__ == "__main__":
    run_benchmark()
