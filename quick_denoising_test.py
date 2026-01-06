#!/usr/bin/env python3
"""
Quick denoising accuracy test with dummy model
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from model.model_dp import DiffusionPolicy

def create_dummy_data(batch_size=4, action_chunk=50, action_output_dim=8, device='cuda:0'):
    """
    Create dummy data for testing
    """
    # Create realistic action sequences (sinusoidal + noise)
    timesteps = np.linspace(0, 2*np.pi, action_chunk)
    
    targets = []
    for b in range(batch_size):
        target_seq = []
        for dim in range(action_output_dim):
            # Create different frequency patterns for each dimension
            freq = 0.5 + dim * 0.3
            amplitude = 0.5 + dim * 0.1
            phase = b * np.pi / 4  # Different phase for each batch
            
            if dim < 3:  # Position (x, y, z)
                signal = amplitude * np.sin(freq * timesteps + phase)
            elif dim < 7:  # Orientation (qx, qy, qz, qw)  
                signal = amplitude * 0.5 * np.cos(freq * timesteps + phase)
            else:  # Gripper
                signal = 0.5 * (np.sin(freq * timesteps + phase) > 0).astype(float)
            
            target_seq.append(signal)
        
        # Stack dimensions and flatten
        target_matrix = np.array(target_seq).T  # (action_chunk, action_output_dim)
        target_flat = target_matrix.flatten()   # (action_chunk * action_output_dim,)
        targets.append(target_flat)
    
    targets = torch.tensor(np.array(targets), dtype=torch.float32, device=device)
    
    # Create dummy frames and qpos
    frames = torch.randn(batch_size, 1, 3, 180, 320, device=device)  # RGB pointcloud
    qpos = torch.randn(batch_size, 1, 9, device=device)
    
    return frames, qpos, targets

def test_denoising_with_dummy_model(device='cuda:0', num_denoising_steps=20):
    """
    Test denoising with a dummy model
    """
    print("Quick Denoising Accuracy Test")
    print("=" * 50)
    print(f"Using device: {device}")
    
    # Model configuration
    image_shape = (3, 180, 320)
    embed_dim = 256
    window_size = 1
    qpos_dim = 9
    output_type = 'ee_pose'
    action_chunk = 50
    action_output_dim = 8
    batch_size = 4
    
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
    
    # Create dummy data
    print(f"\nCreating dummy data...")
    frames, qpos, targets = create_dummy_data(batch_size, action_chunk, action_output_dim, device)
    
    print(f"Data shapes:")
    print(f"  Frames: {frames.shape}")
    print(f"  QPos: {qpos.shape}")
    print(f"  Targets: {targets.shape}")
    
    # Test denoising process
    print(f"\nTesting denoising process...")
    
    with torch.no_grad():
        # 1. Add maximum noise to targets
        if hasattr(model, 'noise_scheduler') and model.use_diffusers_scheduler:
            max_timesteps = torch.full((targets.shape[0],), model.num_diffusion_steps - 1, 
                                     device=device, dtype=torch.long)
            noise = torch.randn_like(targets)
            noisy_targets, _ = model.add_noise(targets, max_timesteps, noise)
        else:
            # Fallback: add pure noise
            noisy_targets = torch.randn_like(targets)
        
        print(f"Added noise to targets")
        print(f"  Original targets range: [{targets.min():.3f}, {targets.max():.3f}]")
        print(f"  Noisy targets range: [{noisy_targets.min():.3f}, {noisy_targets.max():.3f}]")
        
        # 2. Denoise back to clean actions
        print(f"Denoising with {num_denoising_steps} steps...")
        denoised_actions = model.conditional_sample(frames, qpos, num_denoising_steps, noisy_init=noisy_targets)
        
        print(f"Denoising completed")
        print(f"  Denoised actions range: [{denoised_actions.min():.3f}, {denoised_actions.max():.3f}]")
        
        # 3. Calculate metrics
        denoising_mse = nn.MSELoss()(denoised_actions, targets)
        noise_mse = nn.MSELoss()(noisy_targets, targets)
        
        print(f"\nMetrics:")
        print(f"  Noise MSE: {noise_mse.item():.6f}")
        print(f"  Denoising MSE: {denoising_mse.item():.6f}")
        print(f"  Improvement: {((noise_mse - denoising_mse) / noise_mse * 100).item():.1f}%")
    
    return targets, noisy_targets, denoised_actions

def visualize_results(targets, noisy_targets, denoised_actions, 
                     action_chunk=50, action_output_dim=8, num_samples=2):
    """
    Visualize denoising results
    """
    print(f"\nCreating visualizations...")
    
    # Action dimension names
    action_names = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'gripper']
    
    # Convert to numpy and reshape
    targets_np = targets.cpu().numpy()
    noisy_np = noisy_targets.cpu().numpy()
    denoised_np = denoised_actions.cpu().numpy()
    
    # Create comparison plot
    fig, axes = plt.subplots(num_samples, action_output_dim, figsize=(20, 6*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for sample_idx in range(num_samples):
        # Reshape from flat to (action_chunk, action_output_dim)
        original = targets_np[sample_idx].reshape(action_chunk, action_output_dim)
        noisy = noisy_np[sample_idx].reshape(action_chunk, action_output_dim)
        denoised = denoised_np[sample_idx].reshape(action_chunk, action_output_dim)
        
        for dim_idx in range(action_output_dim):
            ax = axes[sample_idx, dim_idx]
            
            timesteps = np.arange(action_chunk)
            ax.plot(timesteps, original[:, dim_idx], 'g-', label='Original', linewidth=2.5)
            ax.plot(timesteps, noisy[:, dim_idx], 'r:', label='Noisy', alpha=0.8, linewidth=1.5)
            ax.plot(timesteps, denoised[:, dim_idx], 'b--', label='Denoised', linewidth=2)
            
            ax.set_title(f'Sample {sample_idx+1}: {action_names[dim_idx]}', fontsize=12)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Calculate and display MSE for this dimension
            dim_mse = np.mean((denoised[:, dim_idx] - original[:, dim_idx])**2)
            ax.text(0.02, 0.98, f'MSE: {dim_mse:.4f}', transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Denoising Results Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig('quick_denoising_test.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: quick_denoising_test.png")
    
    # Create error analysis
    plt.figure(figsize=(12, 8))
    
    errors_by_dim = []
    noise_errors_by_dim = []
    
    for dim_idx in range(action_output_dim):
        dim_errors = []
        dim_noise_errors = []
        
        for sample_idx in range(min(num_samples, targets_np.shape[0])):
            original = targets_np[sample_idx].reshape(action_chunk, action_output_dim)
            noisy = noisy_np[sample_idx].reshape(action_chunk, action_output_dim)
            denoised = denoised_np[sample_idx].reshape(action_chunk, action_output_dim)
            
            mse = np.mean((denoised[:, dim_idx] - original[:, dim_idx])**2)
            noise_mse = np.mean((noisy[:, dim_idx] - original[:, dim_idx])**2)
            
            dim_errors.append(mse)
            dim_noise_errors.append(noise_mse)
        
        errors_by_dim.append(np.mean(dim_errors))
        noise_errors_by_dim.append(np.mean(dim_noise_errors))
    
    x_pos = np.arange(action_output_dim)
    width = 0.35
    
    plt.bar(x_pos - width/2, noise_errors_by_dim, width, label='Noise Error', alpha=0.7, color='red')
    plt.bar(x_pos + width/2, errors_by_dim, width, label='Denoising Error', alpha=0.7, color='blue')
    
    plt.xlabel('Action Dimension')
    plt.ylabel('Mean Squared Error')
    plt.title('Denoising Error Analysis by Dimension')
    plt.xticks(x_pos, action_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('quick_error_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Saved error analysis to: quick_error_analysis.png")
    
    # Print summary
    print(f"\nDimension-wise Performance:")
    print(f"{'Dimension':<12} {'Noise MSE':<12} {'Denoise MSE':<12} {'Improvement':<12}")
    print(f"{'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    
    for dim_idx in range(action_output_dim):
        noise_mse = noise_errors_by_dim[dim_idx]
        denoise_mse = errors_by_dim[dim_idx]
        improvement = (noise_mse - denoise_mse) / noise_mse * 100
        
        print(f"{action_names[dim_idx]:<12} {noise_mse:<12.4f} {denoise_mse:<12.4f} {improvement:<12.1f}%")

def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Test with different denoising steps
    for num_steps in [10, 20, 50]:
        print(f"\n{'='*60}")
        print(f"Testing with {num_steps} denoising steps")
        print(f"{'='*60}")
        
        targets, noisy_targets, denoised_actions = test_denoising_with_dummy_model(
            device=device, num_denoising_steps=num_steps
        )
        
        # Only visualize for the first test
        if num_steps == 20:
            visualize_results(targets, noisy_targets, denoised_actions)
    
    print(f"\n" + "="*60)
    print("QUICK DENOISING TEST COMPLETED")
    print(f"="*60)
    print("Check the generated images:")
    print("  - quick_denoising_test.png")
    print("  - quick_error_analysis.png")

if __name__ == "__main__":
    main()
