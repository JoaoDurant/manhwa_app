import sys
import time
from pathlib import Path

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
    "indextts.gpt.model_v2",
    "indextts.utils.maskgct_utils",
    "indextts.utils.checkpoint",
    "indextts.utils.front",
    "indextts.s2mel.modules.commons",
    "indextts.s2mel.modules.bigvgan.bigvgan",
    "indextts.s2mel.modules.campplus.DTDNN",
    "indextts.s2mel.modules.audio"
]

for m in modules_to_test:
    test_import(m)
