#!/usr/bin/env python3
"""
Quick inference speed test for DiffusionPolicy
"""
import time
import torch
import numpy as np
from model.model_dp import DiffusionPolicy

def quick_test():
    """
    Quick test with dummy model (no checkpoint needed)
    """
    print("Quick DiffusionPolicy Inference Speed Test")
    print("=" * 50)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Model configuration
    image_shape = (3, 180, 320)  # RGB pointcloud
    embed_dim = 256
    window_size = 1
    qpos_dim = 9
    output_type = 'ee_pose'
    action_chunk = 50
    batch_size = 1
    
    print(f"\nModel Configuration:")
    print(f"  Image shape: {image_shape}")
    print(f"  Action chunk: {action_chunk}")
    print(f"  Output type: {output_type}")
    
    # Create model
    print(f"\nCreating model...")
    model = DiffusionPolicy(
        image_shape=image_shape,
        embed_dim=embed_dim,
        window_size=window_size,
        qpos_dim=qpos_dim,
        output_type=output_type,
        action_chunk=action_chunk,
        num_diffusion_steps=100,
        encoder_type='cnn',
        use_diffusers_scheduler=True,
        scheduler_type='ddim',
        beta_schedule='squaredcos_cap_v2'
    ).to(device)
    
    model.eval()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create sample data
    frames = torch.randn(batch_size, window_size, *image_shape, device=device)
    qpos = torch.randn(batch_size, window_size, qpos_dim, device=device)
    
    print(f"\nInput shapes:")
    print(f"  Frames: {frames.shape}")
    print(f"  QPos: {qpos.shape}")
    
    # Test different denoising steps
    denoising_steps_list = [5, 10, 20, 50, 100]
    num_warmup = 3
    num_trials = 10
    
    print(f"\n{'Steps':<8} {'Mean (ms)':<12} {'Std (ms)':<10} {'Output Shape':<15}")
    print(f"{'-'*8} {'-'*12} {'-'*10} {'-'*15}")
    
    for num_steps in denoising_steps_list:
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model.forward(frames, qpos, num_denoising_steps=num_steps)
        
        # Synchronize
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_trials):
                start_time = time.time()
                output = model.forward(frames, qpos, num_denoising_steps=num_steps)
                if device.startswith('cuda'):
                    torch.cuda.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)
        
        # Statistics
        times = np.array(times)
        mean_time = np.mean(times) * 1000  # Convert to ms
        std_time = np.std(times) * 1000
        
        print(f"{num_steps:<8} {mean_time:<12.2f} {std_time:<10.2f} {str(output.shape):<15}")
    
    # Real-time analysis
    print(f"\nReal-time Analysis:")
    print(f"  Control frequency: 10 Hz (100ms budget)")
    print(f"  Fastest config: {min(denoising_steps_list)} steps")
    
    # Test batch processing
    print(f"\nBatch Processing Test:")
    batch_sizes = [1, 2, 4, 8]
    num_steps = 20  # Fixed steps
    
    print(f"{'Batch':<8} {'Time/Sample (ms)':<18} {'Throughput (samples/s)':<22}")
    print(f"{'-'*8} {'-'*18} {'-'*22}")
    
    for bs in batch_sizes:
        frames_batch = torch.randn(bs, window_size, *image_shape, device=device)
        qpos_batch = torch.randn(bs, window_size, qpos_dim, device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model.forward(frames_batch, qpos_batch, num_denoising_steps=num_steps)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(5):
                start_time = time.time()
                _ = model.forward(frames_batch, qpos_batch, num_denoising_steps=num_steps)
                if device.startswith('cuda'):
                    torch.cuda.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)
        
        mean_time = np.mean(times)
        time_per_sample = mean_time / bs * 1000  # ms per sample
        throughput = bs / mean_time  # samples per second
        
        print(f"{bs:<8} {time_per_sample:<18.2f} {throughput:<22.2f}")
    
    print(f"\n" + "="*50)
    print("Test completed!")

if __name__ == "__main__":
    quick_test()
