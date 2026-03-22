import sys
from pathlib import Path

sys.path.insert(0, str(Path("e:/backup/v5/engines/index-tts")))

def t(m):
    print(f"Importing {m}...", end="", flush=True)
    import importlib
    importlib.import_module(m)
    print(" DONE")

t("os")
t("json")
t("re")
t("time")
t("librosa")
t("torch")
t("torchaudio")
t("omegaconf")
t("indextts.gpt.model_v2")
t("indextts.utils.maskgct_utils")
t("indextts.utils.checkpoint")
t("indextts.utils.front")
t("indextts.s2mel.modules.commons")
t("indextts.s2mel.modules.bigvgan.bigvgan")
t("indextts.s2mel.modules.campplus.DTDNN")
t("indextts.s2mel.modules.audio")
t("transformers")
t("modelscope")
t("huggingface_hub")
t("safetensors")
t("indextts.infer_v2")
print("ALL DONE")
