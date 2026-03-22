import sys
from pathlib import Path

sys.path.insert(0, str(Path("e:/backup/v5/engines/index-tts")))

print("Importing front...")
import indextts.utils.front as front
print("SUCCESS front")

print("Importing bigvgan...")
from indextts.s2mel.modules.bigvgan import bigvgan
print("SUCCESS bigvgan")
