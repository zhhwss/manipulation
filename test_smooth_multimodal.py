#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from model.multi_head import MultiModalRobotGraspModel

def test_smoothing_effectiveness():
    """Test the smoothing effectiveness of the new multimodal model"""
    print("Testing Temporal Smoothing in MultiModal Model")
    print("=" * 60)
    
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
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    temporal_params = sum(p.numel() for p in model.temporal_smoother.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Temporal smoother parameters: {temporal_params:,} ({temporal_params/total_params*100:.1f}%)")
    
    print(f"\n1. Analyzing Temporal Smoother Architecture:")
    print("-" * 50)
    
    for i, module in enumerate(model.temporal_smoother):
        if isinstance(module, nn.Conv1d):
            kernel_size = module.kernel_size[0]
            in_channels = module.in_channels
            out_channels = module.out_channels
            groups = module.groups
            print(f"  Conv1d Layer {i//3 + 1}: kernel={kernel_size}, channels={in_channels}→{out_channels}, groups={groups}")
    
    print(f"\n2. Testing Smoothing on Synthetic Noisy Data:")
    print("-" * 50)
    
    model.eval()
    with torch.no_grad():
        # Create test data with medium point cloud coverage
        frames = torch.randn(batch_size, window_size, 4, 180, 320).to(device)
        qpos = torch.randn(batch_size, window_size, qpos_dim).to(device)
        
        # Set 20% of point cloud pixels to non-zero (medium coverage)
        total_pixels = 180 * 320
        keep_pixels = int(total_pixels * 0.2)
        
        for b in range(batch_size):
            # Zero out point cloud first
            frames[b, :, :3] = 0.0
            
            # Set random pixels to non-zero values
            flat_indices = torch.randperm(total_pixels)[:keep_pixels]
            row_indices = flat_indices // 320
            col_indices = flat_indices % 320
            
            for c in range(3):
                frames[b, 0, c, row_indices, col_indices] = torch.randn(keep_pixels).to(device)
        
        # Get model output
        output = model(frames, qpos)
        
        print(f"  Output shape: {output.shape}")
        print(f"  Expected shape: ({batch_size}, {action_chunk * 8})")
        
        # Analyze output smoothness
        output_np = output.cpu().numpy()
        
        for b in range(min(2, batch_size)):  # Analyze first 2 samples
            sample_output = output_np[b].reshape(action_chunk, 8)  # (50, 8)
            
            # Calculate smoothness metrics for each action dimension
            smoothness_scores = []
            for dim in range(8):
                trajectory = sample_output[:, dim]
                
                # Calculate total variation (measure of roughness)
                total_variation = np.sum(np.abs(np.diff(trajectory)))
                
                # Calculate standard deviation of differences (smoothness)
                diff_std = np.std(np.diff(trajectory))
                
                smoothness_scores.append({'tv': total_variation, 'diff_std': diff_std})
            
            avg_tv = np.mean([s['tv'] for s in smoothness_scores])
            avg_diff_std = np.mean([s['diff_std'] for s in smoothness_scores])
            
            print(f"  Sample {b+1}:")
            print(f"    Average Total Variation: {avg_tv:.6f} (lower = smoother)")
            print(f"    Average Diff Std: {avg_diff_std:.6f} (lower = smoother)")
    
    print(f"\n3. Comparing With and Without Smoothing:")
    print("-" * 50)
    
    # Create a version without smoothing for comparison
    class MultiModalWithoutSmoothing(MultiModalRobotGraspModel):
        def forward(self, frames, qpos):
            batch_size, window_size, channels, height, width = frames.shape
            
            # Reshape for CNN processing
            frames_flat = frames.view(batch_size, window_size * channels, height, width)
            qpos_flat = qpos.view(batch_size, window_size * self.qpos_dim)
            
            # Split into point cloud and grayscale
            pointcloud_data = frames_flat[:, :3*window_size]
            grayscale_data = frames_flat[:, 3*window_size:]
            
            # Compute point cloud mask ratio
            pc_ratios = self._compute_pointcloud_mask_ratio(pointcloud_data)
            
            # Process qpos (shared)
            qpos_features = self.qpos_fc(qpos_flat)
            
            # Process branches WITHOUT smoothing
            pc_cnn_features = self.pointcloud_cnn(pointcloud_data)
            pc_cnn_features = pc_cnn_features.view(batch_size, -1)
            pc_features = self.pc_fc(pc_cnn_features)
            pc_combined = torch.cat([pc_features, qpos_features], dim=-1)
            pc_output = self.pc_head(pc_combined)
            
            gray_cnn_features = self.grayscale_cnn(grayscale_data)
            gray_cnn_features = gray_cnn_features.view(batch_size, -1)
            gray_features = self.gray_fc(gray_cnn_features)
            gray_combined = torch.cat([gray_features, qpos_features], dim=-1)
            gray_output = self.gray_head(gray_combined)
            
            # Compute weights and combine WITHOUT smoothing
            use_average = pc_ratios > 0.1
            pc_weights = torch.where(use_average, 
                                    torch.tensor(0.5, device=frames.device), 
                                    pc_ratios / 0.1)
            gray_weights = 1.0 - pc_weights
            
            pc_weights = pc_weights.unsqueeze(1)
            gray_weights = gray_weights.unsqueeze(1)
            
            combined_output = pc_weights * pc_output + gray_weights * gray_output
            combined_output = (combined_output + 1) / 2 * (self.out_max - self.out_min) + self.out_min
            
            return combined_output
    
    model_no_smooth = MultiModalWithoutSmoothing(
        image_shape=image_shape,
        embed_dim=embed_dim,
        window_size=window_size,
        qpos_dim=qpos_dim,
        output_type=output_type,
        action_chunk=action_chunk
    ).to(device)
    
    # Copy weights from original model (except smoothing layers)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'temporal_smoother' not in name:
                if hasattr(model_no_smooth, name.replace('.', '_')):
                    getattr(model_no_smooth, name.replace('.', '_')).copy_(param)
                else:
                    # Handle nested parameters
                    try:
                        parts = name.split('.')
                        target = model_no_smooth
                        for part in parts[:-1]:
                            target = getattr(target, part)
                        getattr(target, parts[-1]).copy_(param)
                    except:
                        pass  # Skip if parameter doesn't exist in no-smooth model
    
    model_no_smooth.eval()
    
    with torch.no_grad():
        output_smooth = model(frames, qpos)
        output_no_smooth = model_no_smooth(frames, qpos)
        
        # Compare smoothness
        smooth_np = output_smooth.cpu().numpy()
        no_smooth_np = output_no_smooth.cpu().numpy()
        
        for b in range(min(1, batch_size)):  # Analyze first sample
            smooth_traj = smooth_np[b].reshape(action_chunk, 8)
            no_smooth_traj = no_smooth_np[b].reshape(action_chunk, 8)
            
            smooth_tv = np.mean([np.sum(np.abs(np.diff(smooth_traj[:, d]))) for d in range(8)])
            no_smooth_tv = np.mean([np.sum(np.abs(np.diff(no_smooth_traj[:, d]))) for d in range(8)])
            
            improvement = (no_smooth_tv - smooth_tv) / no_smooth_tv * 100
            
            print(f"  Sample {b+1} Smoothness Comparison:")
            print(f"    Without smoothing TV: {no_smooth_tv:.6f}")
            print(f"    With smoothing TV: {smooth_tv:.6f}")
            print(f"    Smoothness improvement: {improvement:.1f}%")
    
    print(f"\n4. Performance Impact:")
    print("-" * 50)
    
    if device.type == 'cuda':
        import time
        
        # Warm up
        for _ in range(10):
            with torch.no_grad():
                _ = model(frames, qpos)
                _ = model_no_smooth(frames, qpos)
        
        torch.cuda.synchronize()
        
        # Benchmark with smoothing
        num_runs = 50
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(frames, qpos)
        torch.cuda.synchronize()
        smooth_time = (time.time() - start_time) / num_runs * 1000
        
        # Benchmark without smoothing
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model_no_smooth(frames, qpos)
        torch.cuda.synchronize()
        no_smooth_time = (time.time() - start_time) / num_runs * 1000
        
        overhead = (smooth_time - no_smooth_time) / no_smooth_time * 100
        
        print(f"  Without smoothing: {no_smooth_time:.2f} ms")
        print(f"  With smoothing: {smooth_time:.2f} ms")
        print(f"  Overhead: {overhead:.1f}%")
    else:
        print("  GPU not available, skipping performance benchmark")
    
    print(f"\n✓ Temporal Smoothing Analysis Complete!")
    print(f"\nSummary:")
    print(f"- The model now includes temporal smoothing layers")
    print(f"- Smoothing is applied to both point cloud and grayscale branches")
    print(f"- The temporal smoother uses depthwise convolutions for efficiency")
    print(f"- Gaussian kernels are used for low-pass filtering")
    print(f"- Action trajectories should be significantly smoother")

if __name__ == "__main__":
    test_smoothing_effectiveness()





