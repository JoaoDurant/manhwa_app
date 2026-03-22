import torch
import sys

print("========================================")
print("     GPU BLACKWELL VALIDATION SCRIPT    ")
print("========================================")
print(f"Python Version: {sys.version.split(' ')[0]}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"Device Name: {device_name}")
    print(f"Compute Capability: {cap[0]}.{cap[1]}")
    
    try:
        print("\nTestando alocacao de Tensor na VRAM...")
        tensor_a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
        tensor_b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
        print("Testando multiplicacao de Matriz (MatMul CUDA)...")
        tensor_c = torch.matmul(tensor_a, tensor_b)
        
        # Sincroniza para capturar erros assincronos
        torch.cuda.synchronize()
        
        print("SUCESSO ABSOLUTO! O kernel de GPU rodou perfeitamente.")
        sys.exit(0)
    except Exception as e:
        print(f"FALHA CRITICA NA EXECUCAO CUDA: {e}")
        sys.exit(1)
else:
    print("CUDA nao detectado ou drivers corrompidos.")
    sys.exit(1)
