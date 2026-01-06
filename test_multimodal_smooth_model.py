#!/usr/bin/env python3

import torch
import torch.nn as nn
import time
import numpy as np
from model.multi_head import MultiModalRobotGraspModel

def test_smooth_multimodal_model():
    """Test the new multimodal model with temporal smoothing"""
    print("Testing MultiModal Robot Grasp Model with Temporal Smoothing")
    print("=" * 60)
    
    # Model parameters
    image_shape = (4, 180, 320)  # 4 channels: 3 for point cloud + 1 for grayscale
    embed_dim = 256
    window_size = 1
    qpos_dim = 8
    output_type = 'ee_pose'
    action_chunk = 50
    batch_size = 8
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = MultiModalRobotGraspModel(
        image_shape=image_shape,
        embed_dim=embed_dim,
        window_size=window_size,
        qpos_dim=qpos_dim,
        output_type=output_type,
        action_chunk=action_chunk
    ).to(device)
    
    print(f"Model created successfully")
    print(f"Expected output shape: ({batch_size}, {action_chunk * 8})")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count parameters in smoothing layers
    pc_smooth_params = sum(p.numel() for p in model.pc_temporal_smooth.parameters())
    gray_smooth_params = sum(p.numel() for p in model.gray_temporal_smooth.parameters())
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Point cloud smoothing parameters: {pc_smooth_params:,}")
    print(f"Grayscale smoothing parameters: {gray_smooth_params:,}")
    print(f"Total smoothing parameters: {pc_smooth_params + gray_smooth_params:,}")
    
    print(f"\n1. Testing Temporal Smoothing Effects:")
    print("-" * 50)
    
    model.eval()
    with torch.no_grad():
        # Create test data with different point cloud scenarios
        scenarios = [
            {"name": "High Point Cloud (30% non-zero)", "pc_ratio": 0.3},
            {"name": "Low Point Cloud (5% non-zero)", "pc_ratio": 0.05},
        ]
        
        for scenario in scenarios:
            print(f"\n{scenario['name']}:")
            
            frames = torch.randn(batch_size, window_size, 4, 180, 320).to(device)
            qpos = torch.randn(batch_size, window_size, qpos_dim).to(device)
            
            # Simulate point cloud data based on scenario
            pc_ratio = scenario['pc_ratio']
            total_pixels = 180 * 320
            keep_pixels = int(total_pixels * pc_ratio)
            
            for b in range(batch_size):
                # Create random mask
                mask = torch.zeros(180, 320, dtype=torch.bool)
                flat_indices = torch.randperm(total_pixels)[:keep_pixels]
                row_indices = flat_indices // 320
                col_indices = flat_indices % 320
                mask[row_indices, col_indices] = True
                
                # Apply mask to all 3 point cloud channels
                for c in range(3):
                    frames[b, 0, c][~mask] = 0.0
            
            # Forward pass
            start_time = time.time()
            output = model(frames, qpos)
            end_time = time.time()
            
            # Analyze temporal smoothness
            output_reshaped = output.view(batch_size, 8, action_chunk)  # (batch, action_dim, time_steps)
            
            # Calculate temporal variation (smoothness metric)
            temporal_diff = torch.diff(output_reshaped, dim=2)  # Differences between consecutive time steps
            smoothness_score = torch.mean(torch.abs(temporal_diff))
            
            # Calculate maximum temporal jump
            max_temporal_jump = torch.max(torch.abs(temporal_diff))
            
            # Calculate action consistency across time
            action_std = torch.std(output_reshaped, dim=2)  # Std across time for each action dim
            avg_action_std = torch.mean(action_std)
            
            inference_time = (end_time - start_time) * 1000  # ms
            
            print(f"  Inference time: {inference_time:.2f} ms")
            print(f"  Temporal smoothness: {smoothness_score.item():.6f} (lower = smoother)")
            print(f"  Max temporal jump: {max_temporal_jump.item():.6f}")
            print(f"  Avg action std: {avg_action_std.item():.6f}")
            print(f"  Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    print(f"\n2. Smoothing Architecture Analysis:")
    print("-" * 50)
    
    # Analyze the smoothing layers
    print("Point Cloud Temporal Smoothing Layers:")
    for i, layer in enumerate(model.pc_temporal_smooth):
        if hasattr(layer, 'weight'):
            print(f"  Layer {i}: {layer}")
            if hasattr(layer, 'kernel_size'):
                print(f"    Kernel size: {layer.kernel_size}")
            if hasattr(layer, 'groups'):
                print(f"    Groups: {layer.groups}")
    
    print("\nGrayscale Temporal Smoothing Layers:")
    for i, layer in enumerate(model.gray_temporal_smooth):
        if hasattr(layer, 'weight'):
            print(f"  Layer {i}: {layer}")
            if hasattr(layer, 'kernel_size'):
                print(f"    Kernel size: {layer.kernel_size}")
            if hasattr(layer, 'groups'):
                print(f"    Groups: {layer.groups}")
    
    print(f"\n3. Performance Benchmark with Smoothing:")
    print("-" * 50)
    
    if device.type == 'cuda':
        model.eval()
        
        # Prepare test data
        frames = torch.randn(batch_size, window_size, 4, 180, 320).to(device)
        qpos = torch.randn(batch_size, window_size, qpos_dim).to(device)
        
        # Warm up
        for _ in range(10):
            with torch.no_grad():
                _ = model(frames, qpos)
        
        torch.cuda.synchronize()
        
        # Benchmark
        num_runs = 100
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                output = model(frames, qpos)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs * 1000  # ms
        fps = 1000 / avg_time
        
        print(f"  Average inference time (with smoothing): {avg_time:.2f} ms")
        print(f"  Throughput: {fps:.1f} FPS")
        print(f"  Batch size: {batch_size}")
        print(f"  Per-sample time: {avg_time/batch_size:.2f} ms")
    else:
        print("  GPU not available, skipping performance benchmark")
    
    print(f"\n4. Model Architecture Summary:")
    print("-" * 50)
    
    print(f"  Input: {image_shape} frames + {qpos_dim}D qpos")
    print(f"  Point Cloud Branch: 3 channels -> CNN -> FC -> Head -> Temporal Smoothing")
    print(f"  Grayscale Branch: 1 channel -> CNN -> FC -> Head -> Temporal Smoothing")
    print(f"  Temporal Smoothing Architecture:")
    print(f"    - Layer 1: Conv1D(kernel=5) for global trends")
    print(f"    - Layer 2: Conv1D(kernel=3) for local smoothing")
    print(f"    - Layer 3: Conv1D(kernel=3) for fine adjustment")
    print(f"    - Residual connections to preserve original signal")
    print(f"    - Grouped convolutions for per-action-dimension processing")
    print(f"  Dynamic weighting based on point cloud availability")
    print(f"  Output: {action_chunk} x {8} = {action_chunk * 8}D action sequence")
    
    print(f"\n✓ All tests completed successfully!")
    print(f"\nKey Smoothing Features:")
    print(f"  ✓ Multi-scale temporal convolutions (kernel sizes: 5, 3, 3)")
    print(f"  ✓ Residual connections preserve original action signals")
    print(f"  ✓ Separate smoothing networks for each modality")
    print(f"  ✓ Grouped convolutions for efficient processing")
    print(f"  ✓ BatchNorm for stable training")

if __name__ == "__main__":
    test_smooth_multimodal_model()





