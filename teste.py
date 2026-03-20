import torch
import time

print("Antes da geração:")
print("GPU mem:", torch.cuda.memory_allocated() / 1024**2, "MB")

start = time.time()

# SUA geração aqui (chatterbox ou kokoro)

print("Tempo:", time.time() - start)

print("Depois da geração:")
print("GPU mem:", torch.cuda.memory_allocated() / 1024**2, "MB")