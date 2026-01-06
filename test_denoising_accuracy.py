#!/usr/bin/env python3
"""
Test denoising accuracy and visualize the results
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from model.model_dp import DiffusionPolicy
from data_loader import create_data_loaders

def load_trained_model(checkpoint_path, device='cuda:0'):
    """
    Load trained model from checkpoint
    """
    print(f"Loading model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get model arguments from checkpoint
    if 'args' in checkpoint:
        args = checkpoint['args']
        print(f"Model configuration:")
        print(f"  Image type: {args.image_type}")
        print(f"  Output type: {args.output_type}")
        print(f"  Action chunk: {args.action_chunk}")
        print(f"  Encoder type: {args.encoder_type}")
        
        # Determine image shape (simplified)
        if args.image_type == 'pointcloud':
            image_shape = (3, 180, 320)
        elif args.image_type == 'depth_mask':
            image_shape = (1, 180, 320)
        else:
            image_shape = (3, 180, 320)  # default
            
        if hasattr(args, 'third_view') and args.third_view:
            image_shape = (image_shape[0] * 2, *image_shape[1:])
        if hasattr(args, 'color_view') and args.color_view:
            image_shape = (image_shape[0] + 1, *image_shape[1:])
        
        # Parse down_dims
        down_dims = tuple(int(x) for x in args.down_dims.split(','))
        
        # Create model
        model = DiffusionPolicy(
            image_shape=image_shape,
            embed_dim=args.embed_dim,
            window_size=args.window_size,
            qpos_dim=9,  # Assuming standard qpos dimension
            output_type=args.output_type,
            action_chunk=args.action_chunk,
            num_diffusion_steps=args.num_diffusion_steps,
            encoder_type=args.encoder_type,
            use_diffusers_scheduler=args.use_diffusers_scheduler,
            scheduler_type=getattr(args, 'scheduler_type', 'ddim'),
            beta_schedule=getattr(args, 'beta_schedule', 'squaredcos_cap_v2'),
            beta_start=getattr(args, 'beta_start', 0.0001),
            beta_end=getattr(args, 'beta_end', 0.02),
            prediction_type=getattr(args, 'prediction_type', 'sample'),
            down_dims=down_dims,
            kernel_size=getattr(args, 'kernel_size', 5),
            n_groups=getattr(args, 'n_groups', 8)
        )
        
        return_args = args
        
    else:
        # Fallback: create model with default parameters
        print("Warning: No args found in checkpoint, using default parameters")
        image_shape = (3, 180, 320)
        model = DiffusionPolicy(
            image_shape=image_shape,
            embed_dim=256,
            window_size=1,
            qpos_dim=9,
            output_type='ee_pose',
            action_chunk=50,
            num_diffusion_steps=100,
            encoder_type='cnn'
        )
        
        # Create dummy args
        class DummyArgs:
            def __init__(self):
                self.data_dir = 'data/20250730_shui'
                self.image_type = 'pointcloud'
                self.output_type = 'ee_pose'
                self.action_chunk = 50
                self.third_view = False
                self.color_view = False
                self.image_scale = 1.0
                self.window_size = 1
                self.batch_size = 8
                self.train_split = 0.8
                self.num_workers = 4
                self.data_aug = 1
        
        return_args = DummyArgs()
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, return_args

def test_denoising_accuracy(model, val_loader, device, num_denoising_steps=20, num_samples=5):
    """
    Test denoising accuracy and collect samples for visualization
    """
    print(f"\nTesting denoising accuracy...")
    print(f"Denoising steps: {num_denoising_steps}")
    
    model.eval()
    
    samples_data = []
    total_loss = 0
    total_denoising_mse = 0
    num_batches = 0
    
    with torch.no_grad():
        for indices, frames, qpos, targets in val_loader:
            frames = frames.to(device)
            qpos = qpos.to(device)
            targets = targets.to(device)
            
            batch_size = targets.shape[0]
            
            # 1. Compute noise prediction loss
            batch = {
                'frames': frames,
                'qpos': qpos,
                'actions': targets
            }
            
            loss, loss_dict = model.compute_loss(batch)
            total_loss += loss.item()
            
            # 2. Full denoising evaluation
            # Start from ground truth targets and add MAXIMUM noise (simulate full diffusion)
            if hasattr(model, 'noise_scheduler') and hasattr(model, 'use_diffusers_scheduler') and model.use_diffusers_scheduler:
                # Use diffusers scheduler to add maximum noise
                max_timesteps = torch.full((targets.shape[0],), model.num_diffusion_steps - 1, device=device, dtype=torch.long)
                noise = torch.randn_like(targets)
                noisy_targets, _ = model.add_noise(targets, max_timesteps, noise)
            else:
                # Use custom scheduler to add maximum noise (t = num_diffusion_steps - 1)
                if hasattr(model, 'alphas_cumprod'):
                    t = model.num_diffusion_steps - 1  # Maximum timestep
                    alpha_t = model.alphas_cumprod[t]  # Should be very small, close to 0
                    noise = torch.randn_like(targets)
                    noisy_targets = torch.sqrt(alpha_t) * targets + torch.sqrt(1 - alpha_t) * noise
                else:
                    # Fallback: use pure noise (complete corruption)
                    noisy_targets = torch.randn_like(targets)
            
            # Use model's conditional_sample method to denoise from pure noise back to clean actions
            denoised_actions = model.conditional_sample(frames, qpos, num_denoising_steps, noisy_init=noisy_targets)
            
            # Calculate denoising MSE (how well we recovered the original targets from pure noise)
            denoising_mse = nn.MSELoss()(denoised_actions, targets)
            total_denoising_mse += denoising_mse.item()
            
            # Collect samples for visualization
            if len(samples_data) < num_samples:
                for i in range(min(batch_size, num_samples - len(samples_data))):
                    sample = {
                        'original': targets[i].cpu().numpy(),
                        'noisy': noisy_targets[i].cpu().numpy(),
                        'denoised': denoised_actions[i].cpu().numpy(),
                        'frames': frames[i].cpu().numpy(),
                        'qpos': qpos[i].cpu().numpy()
                    }
                    samples_data.append(sample)
            
            num_batches += 1
            
            if len(samples_data) >= num_samples:
                break
    
    avg_loss = total_loss / num_batches
    avg_denoising_mse = total_denoising_mse / num_batches
    
    print(f"Results:")
    print(f"  Noise prediction loss: {avg_loss:.6f}")
    print(f"  Denoising MSE: {avg_denoising_mse:.6f}")
    print(f"  Collected {len(samples_data)} samples for visualization")
    
    return samples_data, avg_loss, avg_denoising_mse

def visualize_denoising_results(samples_data, output_type='ee_pose', action_chunk=50):
    """
    Visualize the denoising results
    """
    print(f"\nCreating visualizations...")
    
    # Action dimension names
    if output_type == 'ee_pose':
        action_names = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'gripper']
        action_output_dim = 8
    else:  # position
        action_names = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
        action_output_dim = 7
    
    num_samples = len(samples_data)
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_samples, action_output_dim, figsize=(20, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for sample_idx, sample in enumerate(samples_data):
        original = sample['original'].reshape(action_chunk, action_output_dim)
        noisy = sample['noisy'].reshape(action_chunk, action_output_dim)
        denoised = sample['denoised'].reshape(action_chunk, action_output_dim)
        
        for dim_idx in range(action_output_dim):
            ax = axes[sample_idx, dim_idx]
            
            # Plot time series
            timesteps = np.arange(action_chunk)
            ax.plot(timesteps, original[:, dim_idx], 'g-', label='Original', linewidth=2)
            ax.plot(timesteps, noisy[:, dim_idx], 'r--', label='Noisy', alpha=0.7)
            ax.plot(timesteps, denoised[:, dim_idx], 'b-', label='Denoised', linewidth=2)
            
            ax.set_title(f'Sample {sample_idx+1}: {action_names[dim_idx]}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('denoising_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: denoising_comparison.png")
    
    # Create error analysis plot
    plt.figure(figsize=(15, 10))
    
    # Calculate errors for each sample
    for sample_idx, sample in enumerate(samples_data):
        original = sample['original'].reshape(action_chunk, action_output_dim)
        noisy = sample['noisy'].reshape(action_chunk, action_output_dim)
        denoised = sample['denoised'].reshape(action_chunk, action_output_dim)
        
        # Calculate MSE for each dimension
        denoising_error = np.mean((denoised - original)**2, axis=0)
        noise_error = np.mean((noisy - original)**2, axis=0)
        
        x_pos = np.arange(action_output_dim) + sample_idx * 0.15
        
        if sample_idx == 0:
            plt.bar(x_pos - 0.075, noise_error, 0.15, label='Noise Error', alpha=0.7, color='red')
            plt.bar(x_pos + 0.075, denoising_error, 0.15, label='Denoising Error', alpha=0.7, color='blue')
        else:
            plt.bar(x_pos - 0.075, noise_error, 0.15, alpha=0.7, color='red')
            plt.bar(x_pos + 0.075, denoising_error, 0.15, alpha=0.7, color='blue')
    
    plt.xlabel('Action Dimension')
    plt.ylabel('Mean Squared Error')
    plt.title('Denoising Error Analysis')
    plt.xticks(np.arange(action_output_dim), action_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('denoising_error_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Saved error analysis to: denoising_error_analysis.png")
    
    # Create summary statistics
    print(f"\nDenoising Performance Summary:")
    print(f"{'Dimension':<12} {'Noise MSE':<12} {'Denoise MSE':<12} {'Improvement':<12}")
    print(f"{'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    
    total_noise_mse = 0
    total_denoise_mse = 0
    
    for dim_idx in range(action_output_dim):
        dim_noise_errors = []
        dim_denoise_errors = []
        
        for sample in samples_data:
            original = sample['original'].reshape(action_chunk, action_output_dim)
            noisy = sample['noisy'].reshape(action_chunk, action_output_dim)
            denoised = sample['denoised'].reshape(action_chunk, action_output_dim)
            
            noise_mse = np.mean((noisy[:, dim_idx] - original[:, dim_idx])**2)
            denoise_mse = np.mean((denoised[:, dim_idx] - original[:, dim_idx])**2)
            
            dim_noise_errors.append(noise_mse)
            dim_denoise_errors.append(denoise_mse)
        
        avg_noise_mse = np.mean(dim_noise_errors)
        avg_denoise_mse = np.mean(dim_denoise_errors)
        improvement = (avg_noise_mse - avg_denoise_mse) / avg_noise_mse * 100
        
        total_noise_mse += avg_noise_mse
        total_denoise_mse += avg_denoise_mse
        
        print(f"{action_names[dim_idx]:<12} {avg_noise_mse:<12.4f} {avg_denoise_mse:<12.4f} {improvement:<12.1f}%")
    
    overall_improvement = (total_noise_mse - total_denoise_mse) / total_noise_mse * 100
    print(f"{'-'*48}")
    print(f"{'Overall':<12} {total_noise_mse:<12.4f} {total_denoise_mse:<12.4f} {overall_improvement:<12.1f}%")

def main():
    parser = argparse.ArgumentParser(description='Test DiffusionPolicy denoising accuracy')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pth file)')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Data directory (if different from training)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use for inference')
    parser.add_argument('--denoising_steps', type=int, default=20,
                       help='Number of denoising steps')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for validation')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return
    
    # Check device
    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Load model
    try:
        model, model_args = load_trained_model(args.checkpoint, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Use data_dir from args or model_args
    data_dir = args.data_dir if args.data_dir else model_args.data_dir
    
    # Create validation data loader
    print(f"\nCreating validation data loader...")
    print(f"Data directory: {data_dir}")
    
    try:
        _, val_loader, _ = create_data_loaders(
            data_dir=data_dir,
            image_type=model_args.image_type,
            third_view=getattr(model_args, 'third_view', False),
            image_scale=getattr(model_args, 'image_scale', 1.0),
            output_type=model_args.output_type,
            action_chunk=model_args.action_chunk,
            action_stride=1,
            batch_size=args.batch_size,
            train_split=0.8,
            window_size=model_args.window_size,
            num_workers=4,
            data_aug=1,
            cache=None,
            color_view=getattr(model_args, 'color_view', False)
        )
        
        print(f"Validation samples: {len(val_loader.dataset)}")
        
    except Exception as e:
        print(f"Error creating data loader: {e}")
        return
    
    # Test denoising accuracy
    try:
        samples_data, noise_loss, denoising_mse = test_denoising_accuracy(
            model, val_loader, device, args.denoising_steps, args.num_samples
        )
        
        # Visualize results
        visualize_denoising_results(
            samples_data, 
            model_args.output_type, 
            model_args.action_chunk
        )
        
        print(f"\n" + "="*60)
        print("DENOISING ACCURACY TEST COMPLETED")
        print(f"="*60)
        print(f"Noise prediction loss: {noise_loss:.6f}")
        print(f"Denoising MSE: {denoising_mse:.6f}")
        print(f"Visualizations saved:")
        print(f"  - denoising_comparison.png")
        print(f"  - denoising_error_analysis.png")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
