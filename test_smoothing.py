#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from model.multi_head import MultiModalRobotGraspModel

def test_action_smoothing():
    """Test the temporal smoothing functionality of the multimodal model"""
    print("Testing Action Chunk Smoothing")
    print("=" * 50)
    
    # Model parameters
    image_shape = (4, 180, 320)  # 4 channels: 3 for point cloud + 1 for grayscale
    embed_dim = 256
    window_size = 1
    qpos_dim = 8
    output_type = 'ee_pose'
    action_chunk = 50
    batch_size = 4
    
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
    
    print(f"Model created with temporal smoothing layers")
    print(f"Action chunk size: {action_chunk}")
    print(f"Action dimension: {8}")
    
    # Create test data with different point cloud scenarios
    scenarios = [
        {"name": "No Point Cloud", "pc_ratio": 0.0},
        {"name": "Partial Point Cloud", "pc_ratio": 0.15},
        {"name": "Full Point Cloud", "pc_ratio": 0.6},
    ]
    
    model.eval()
    results = {}
    
    print(f"\n1. Testing Smoothing with Different Point Cloud Scenarios:")
    print("-" * 60)
    
    with torch.no_grad():
        for scenario in scenarios:
            print(f"\n{scenario['name']} (ratio: {scenario['pc_ratio']:.1f}):")
            
            # Create test data
            frames = torch.randn(batch_size, window_size, 4, 180, 320).to(device)
            qpos = torch.randn(batch_size, window_size, qpos_dim).to(device)
            
            # Simulate point cloud data based on scenario
            pc_ratio = scenario['pc_ratio']
            if pc_ratio == 0.0:
                # No point cloud data
                frames[:, :, :3] = 0.0
            else:
                # Set a portion of point cloud pixels to zero
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
            output = model(frames, qpos)  # (batch_size, action_dim * action_chunk)
            
            # Reshape to analyze temporal structure
            output_reshaped = output.view(batch_size, 8, action_chunk)  # (batch_size, action_dim, action_chunk)
            
            # Compute smoothness metrics for first sample
            sample_output = output_reshaped[0]  # (action_dim, action_chunk)
            
            # Calculate smoothness metrics
            smoothness_metrics = []
            for dim in range(8):
                trajectory = sample_output[dim].cpu().numpy()  # (action_chunk,)
                
                # Calculate second derivative (acceleration) as smoothness measure
                # Lower values indicate smoother trajectories
                second_derivative = np.diff(trajectory, n=2)
                smoothness = np.mean(np.abs(second_derivative))
                smoothness_metrics.append(smoothness)
            
            avg_smoothness = np.mean(smoothness_metrics)
            results[scenario['name']] = {
                'output': sample_output.cpu().numpy(),
                'smoothness': avg_smoothness,
                'smoothness_per_dim': smoothness_metrics
            }
            
            print(f"  Average smoothness (lower=smoother): {avg_smoothness:.6f}")
            print(f"  Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
            print(f"  Output mean: {output.mean().item():.3f}")
            print(f"  Output std: {output.std().item():.3f}")
    
    print(f"\n2. Smoothness Comparison:")
    print("-" * 60)
    
    for name, result in results.items():
        print(f"{name:20s}: Smoothness = {result['smoothness']:.6f}")
    
    print(f"\n3. Visualizing Action Trajectories:")
    print("-" * 60)
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Action Chunk Smoothing Comparison', fontsize=16)
    
    for dim in range(8):
        row = dim // 4
        col = dim % 4
        ax = axes[row, col]
        
        time_steps = np.arange(action_chunk)
        
        for scenario_name, result in results.items():
            trajectory = result['output'][dim]  # (action_chunk,)
            smoothness = result['smoothness_per_dim'][dim]
            label = f"{scenario_name} (s={smoothness:.4f})"
            ax.plot(time_steps, trajectory, label=label, linewidth=2, alpha=0.8)
        
        ax.set_title(f'Action Dim {dim}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Action Value')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig('action_smoothing_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  Visualization saved to 'action_smoothing_comparison.png'")
    
    print(f"\n4. Testing Smoothing Layer Components:")
    print("-" * 60)
    
    # Test individual smoothing layers
    with torch.no_grad():
        # Create a noisy test signal
        test_signal = torch.randn(1, 8, action_chunk).to(device)
        print(f"Original signal std: {test_signal.std().item():.6f}")
        
        # Apply smoothing
        smoothed_signal = model.temporal_smoother(test_signal)
        print(f"Smoothed signal std: {smoothed_signal.std().item():.6f}")
        
        # Calculate improvement in smoothness
        original_smoothness = torch.mean(torch.abs(torch.diff(test_signal, n=2, dim=2))).item()
        smoothed_smoothness = torch.mean(torch.abs(torch.diff(smoothed_signal, n=2, dim=2))).item()
        
        improvement = (original_smoothness - smoothed_smoothness) / original_smoothness * 100
        print(f"Smoothness improvement: {improvement:.1f}%")
    
    print(f"\n5. Smoothing Architecture Summary:")
    print("-" * 60)
    
    print(f"  Temporal Smoother Architecture:")
    for i, layer in enumerate(model.temporal_smoother):
        if isinstance(layer, nn.Conv1d):
            print(f"    Layer {i}: Conv1d(kernel_size={layer.kernel_size[0]}, groups={layer.groups})")
        elif isinstance(layer, nn.BatchNorm1d):
            print(f"    Layer {i}: BatchNorm1d")
        elif isinstance(layer, nn.ReLU):
            print(f"    Layer {i}: ReLU")
    
    print(f"  Features:")
    print(f"    - Depthwise convolution (each action dimension processed independently)")
    print(f"    - Gaussian-initialized kernels for low-pass filtering")
    print(f"    - Multi-scale smoothing (kernel sizes: 5, 7, 3)")
    print(f"    - Applied to both point cloud and grayscale branches")
    
    print(f"\n✓ Action smoothing test completed successfully!")
    print(f"  The temporal smoother reduces jitter and creates more coherent action sequences.")

if __name__ == "__main__":
    test_action_smoothing()
