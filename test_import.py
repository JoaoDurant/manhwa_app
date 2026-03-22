
import traceback
import sys
import os

print(f"Python version: {sys.version}")
print(f"CWD: {os.getcwd()}")
print(f"sys.path: {sys.path}")

try:
    print("\nAttempting to import engine...")
    import engine
    print("Engine imported successfully.")
except Exception:
    print("\nFAILED to import engine. Traceback:")
    traceback.print_exc()

try:
    print("\nAttempting to import transformers...")
    import transformers
    print(f"Transformers version: {transformers.__version__}")
    from transformers.generation.logits_process import UnnormalizedLogitsProcessor
    print("UnnormalizedLogitsProcessor found.")
except Exception:
    print("\nTransformers check FAILED. Traceback:")
    traceback.print_exc()
