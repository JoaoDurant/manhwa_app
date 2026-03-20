import torch
import torchvision
from torchvision.ops import nms  # isso deve importar sem erro

print("Torch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("NMS disponível:", nms)  # só pra confirmar