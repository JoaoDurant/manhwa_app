
import torch
import time
import numpy as np
import os
from PIL import Image

def test_gpu_speed():
    if not torch.cuda.is_available():
        print("CUDA NOT AVAILABLE")
        return
    
    device = 'cuda'
    print(f"Testing on {torch.cuda.get_device_name(0)}")
    
    # Simulate a 1080p frame
    bg = torch.zeros((1, 3, 1080, 1920), device=device)
    fg = torch.randn((1, 4, 900, 800), device=device)
    
    iters = 100
    start = time.time()
    
    for i in range(iters):
        # Clipping/Alpha blending simulation
        alpha = fg[:, 3:4, :, :] / 255.0
        rgb = fg[:, 0:3, :, :]
        
        # This is where the bottleneck might be
        bg[:, :, 0:900, 0:800] = bg[:, :, 0:900, 0:800] * (1.0 - alpha) + rgb * alpha
        
        # Transfer to CPU (the real bottleneck)
        frame = bg.to(torch.uint8).squeeze(0).permute(1, 2, 0).cpu().numpy()
        
    end = time.time()
    fps = iters / (end - start)
    print(f"Estimated FPS (with CPU transfer): {fps:.2f}")
    
    # Test without CPU transfer
    start = time.time()
    for i in range(iters):
        alpha = fg[:, 3:4, :, :] / 255.0
        rgb = fg[:, 0:3, :, :]
        bg[:, :, 0:900, 0:800] = bg[:, :, 0:900, 0:800] * (1.0 - alpha) + rgb * alpha
    
    end = time.time()
    fps_pure = iters / (end - start)
    print(f"Estimated FPS (Pure GPU): {fps_pure:.2f}")

if __name__ == "__main__":
    test_gpu_speed()
