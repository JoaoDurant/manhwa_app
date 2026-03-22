import sys
from pathlib import Path

# Simulate tts_worker.py behavior
sys.path.insert(0, str(Path("e:/backup/v5/engines/index-tts")))

print("Attempting to import indextts...")
try:
    import indextts
    print(f"indextts imported from {indextts.__file__}")
except Exception as e:
    print(f"FAILED to import indextts: {e}")
    sys.exit(1)

modules_to_test = [
    "indextts.infer_v2",
    "indextts.infer",
    "indextts.gpt.model_v2",
    "indextts.utils.maskgct_utils",
    "modelscope",
    "omegaconf",
    "safetensors",
    "transformers",
    "torchaudio",
    "librosa"
]

for m in modules_to_test:
    print(f"Testing {m}...")
    try:
        __import__(m)
        print(f"  SUCCESS: {m}")
    except Exception as e:
        print(f"  FAILED: {m} -> {e}")
        import traceback
        traceback.print_exc()
