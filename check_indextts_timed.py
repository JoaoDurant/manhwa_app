import sys
import time
from pathlib import Path

# Simulate tts_worker.py behavior
sys.path.insert(0, str(Path("e:/backup/v5/engines/index-tts")))

def test_import(m):
    print(f"Testing {m}...", end="", flush=True)
    t0 = time.time()
    try:
        __import__(m)
        print(f" SUCCESS ({time.time()-t0:.2f}s)")
    except Exception as e:
        print(f" FAILED ({time.time()-t0:.2f}s) -> {e}")

modules_to_test = [
    "torch",
    "torchaudio",
    "librosa",
    "omegaconf",
    "safetensors",
    "transformers",
    "modelscope",
    "indextts",
    "indextts.gpt",
    "indextts.utils",
    "indextts.s2mel",
    "indextts.infer_v2"
]

for m in modules_to_test:
    test_import(m)
