#!/usr/bin/env python3
"""
Test denoising accuracy with real data from data loader
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from model.model_dp import DiffusionPolicy
from data_loader import create_data_loaders
import cv2

def load_trained_model(checkpoint_path, device='cuda:0'):
    """
    Load trained model from checkpoint with all original training parameters
    """
    print(f"Loading model from: {checkpoint_path}")
    
    # Load checkpoint (set weights_only=False for compatibility with argparse.Namespace)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get model arguments from checkpoint
    if 'args' not in checkpoint:
        raise ValueError("Checkpoint does not contain training arguments. Cannot reconstruct model.")
    
    args = checkpoint['args']
    print(f"Model configuration from checkpoint:")
    print(f"  Image type: {args.image_type}")
    print(f"  Third view: {getattr(args, 'third_view', False)}")
    print(f"  Color view: {getattr(args, 'color_view', False)}")
    print(f"  Output type: {args.output_type}")
    print(f"  Action chunk: {args.action_chunk}")
    print(f"  Window size: {args.window_size}")
    print(f"  Encoder type: {args.encoder_type}")
    print(f"  Embed dim: {args.embed_dim}")
    print(f"  Diffusion steps: {args.num_diffusion_steps}")
    print(f"  Use diffusers scheduler: {args.use_diffusers_scheduler}")
    
    return args, checkpoint

def create_model_from_args(args, image_shape, qpos_dim, device='cuda:0'):
    """
    Create model using the exact same parameters as training
    """
    # Parse down_dims if it exists
    if hasattr(args, 'down_dims'):
        down_dims = tuple(int(x) for x in args.down_dims.split(','))
    else:
        down_dims = (256, 512, 1024)  # default
    
    # Create model with exact same parameters as training
    model = DiffusionPolicy(
        image_shape=image_shape,
        embed_dim=args.embed_dim,
        window_size=args.window_size,
        qpos_dim=qpos_dim,
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
    ).to(device)
    
    model.eval()
    return model

def save_sample_images(frames, sample_idx, save_dir='images'):
    """
    Save sample images from frames
    
    Args:
        frames: (window_size, channels, height, width)
        sample_idx: sample index for naming
        save_dir: directory to save images
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Process each frame in the window
    for frame_idx in range(frames.shape[0]):
        frame = frames[frame_idx]  # (channels, height, width)
        
        if frame.shape[0] == 1:  # Grayscale/depth
            # Convert to numpy and scale to 0-255
            img = frame[0].cpu().numpy()
            img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # Apply colormap for better visualization
            
        elif frame.shape[0] == 3:  # RGB or pointcloud
            # Convert to numpy (channels, height, width) -> (height, width, channels)
            img = frame.cpu().numpy().transpose(1, 2, 0)
            
            # Normalize to 0-1 range if needed
            if img.max() > 1.0:
                img = img / 255.0
            
            # Clip and scale to 0-255
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        elif frame.shape[0] == 4:  # Pointcloud + grayscale
            # Split into pointcloud (first 3) and grayscale (last 1)
            pc_frame = frame[:3].cpu().numpy().transpose(1, 2, 0)
            gray_frame = frame[3].cpu().numpy()
            
            # Process pointcloud
            pc_img = np.clip(pc_frame, 0, 1)
            pc_img = (pc_img * 255).astype(np.uint8)
            pc_img = cv2.cvtColor(pc_img, cv2.COLOR_RGB2BGR)
            
            # Process grayscale
            gray_img = ((gray_frame - gray_frame.min()) / (gray_frame.max() - gray_frame.min()) * 255).astype(np.uint8)
            gray_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)
            
            # Save both
            cv2.imwrite(os.path.join(save_dir, f'sample_{sample_idx}_frame_{frame_idx}_pointcloud.png'), pc_img)
            cv2.imwrite(os.path.join(save_dir, f'sample_{sample_idx}_frame_{frame_idx}_grayscale.png'), gray_img)
            continue
        
        else:
            print(f"Warning: Unsupported channel count: {frame.shape[0]}")
            continue
        
        # Save image
        filename = os.path.join(save_dir, f'sample_{sample_idx}_frame_{frame_idx}.png')
        cv2.imwrite(filename, img)

def test_denoising_with_trained_model(checkpoint_path, device='cuda:0', 
                                     num_samples=3, num_denoising_steps=20, 
                                     data_dir_override=None):
    """
    Test denoising with trained model and real data
    """
    print("Testing Denoising with Trained Model and Real Data")
    print("=" * 50)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Using device: {device}")
    
    # Load trained model and get training arguments
    print(f"\nLoading trained model...")
    try:
        args, checkpoint = load_trained_model(checkpoint_path, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Use data directory from checkpoint or override
    data_dir = data_dir_override if data_dir_override else args.data_dir
    print(f"Data directory: {data_dir}")
    
    # Create data loaders using EXACT same parameters as training
    print(f"\nCreating data loaders with training configuration...")
    try:
        train_loader, val_loader, full_dataset = create_data_loaders(
            data_dir=data_dir,
            image_type=args.image_type,
            third_view=getattr(args, 'third_view', False),
            image_scale=getattr(args, 'image_scale', 1.0),
            output_type=args.output_type,
            action_chunk=args.action_chunk,
            action_stride=getattr(args, 'action_stride', 1),
            batch_size=8,  # Can be different for testing
            train_split=0.8,  # Same split as training
            window_size=args.window_size,
            num_workers=4,
            data_aug=1,  # No augmentation for testing
            cache=None,
            color_view=getattr(args, 'color_view', False)
        )
        
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        return None
    
    # Get a sample batch to determine image shape and qpos_dim
    print(f"\nDetermining data shapes...")
    sample_batch = next(iter(val_loader))
    sample_indices, sample_frames, sample_qpos, sample_targets = sample_batch
    
    # Get dimensions from the sample
    image_shape = sample_frames.shape[2:]  # (channels, height, width)
    qpos_dim = sample_qpos.shape[-1]
    
    print(f"Detected shapes:")
    print(f"  Image shape: {image_shape}")
    print(f"  QPos dimension: {qpos_dim}")
    print(f"  Target shape: {sample_targets.shape}")
    
    # Create model with exact same architecture as training
    print(f"\nCreating model with training configuration...")
    model = create_model_from_args(args, image_shape, qpos_dim, device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load trained weights
    print(f"Loading trained weights...")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Print training info
    if 'epoch' in checkpoint:
        print(f"Loaded model from epoch {checkpoint['epoch']}")
    if 'val_metrics' in checkpoint:
        val_metrics = checkpoint['val_metrics']
        print(f"Training metrics:")
        for key, value in val_metrics.items():
            print(f"  {key}: {value:.6f}")
    if 'improvement_score' in checkpoint:
        print(f"Improvement score: {checkpoint['improvement_score']:.4f}")
    
    # Get sample data
    print(f"\nLoading sample data...")
    sample_data = []
    
    with torch.no_grad():
        for batch_idx, (indices, frames, qpos, targets) in enumerate(val_loader):
            frames = frames.to(device)
            qpos = qpos.to(device) 
            targets = targets.to(device)
            
            batch_size = frames.shape[0]
            
            # Process each sample in the batch
            for i in range(min(batch_size, num_samples - len(sample_data))):
                sample_idx = len(sample_data)
                
                print(f"Processing sample {sample_idx + 1}...")
                
                # Extract single sample
                sample_frames = frames[i:i+1]  # (1, window_size, channels, height, width)
                sample_qpos = qpos[i:i+1]      # (1, window_size, qpos_dim)
                sample_targets = targets[i:i+1] # (1, action_output_dim * action_chunk)
                print(f"  Sample shapes:")
                print(f"    Frames: {sample_frames.shape}")
                print(f"    QPos: {sample_qpos.shape}")
                print(f"    Targets: {sample_targets.shape}")
                
                # Save sample images
                save_sample_images(sample_frames[0], sample_idx)  # Remove batch dimension
                
                # Add maximum noise to targets (pure random noise for testing)
                noisy_targets = torch.randn_like(sample_targets)
                
                print(f"  Noisy action values (first 10 elements):")
                print(f"    Original: {sample_targets[0][:10].cpu().numpy()}")
                print(f"    Noisy:    {noisy_targets[0][:10].cpu().numpy()}")
                print(f"    Noise stats - Mean: {noisy_targets.mean().item():.4f}, Std: {noisy_targets.std().item():.4f}")
                
                # Denoise back to clean actions
                print(f"  Denoising with {num_denoising_steps} steps...")
                denoised_actions = model.conditional_sample(
                    sample_frames, sample_qpos, 5, noisy_init=noisy_targets
                )
                
                print(f"  Denoised action values (first 10 elements):")
                print(f"    Original: {sample_targets[0][:10].cpu().numpy()}")
                print(f"    Denoised: {denoised_actions[0][:10].cpu().numpy()}")
                print(f"    Denoised stats - Mean: {denoised_actions.mean().item():.4f}, Std: {denoised_actions.std().item():.4f}")
                
                # Calculate metrics
                denoising_mse = nn.MSELoss()(denoised_actions, sample_targets)
                noise_mse = nn.MSELoss()(noisy_targets, sample_targets)
                improvement = ((noise_mse - denoising_mse) / noise_mse * 100).item()
                
                # Store sample data
                sample_info = {
                    'index': indices[i].item(),
                    'frames': sample_frames[0].cpu().numpy(),  # (window_size, channels, height, width)
                    'qpos': sample_qpos[0].cpu().numpy(),      # (window_size, qpos_dim)
                    'original': sample_targets[0].cpu().numpy(),  # (action_output_dim * action_chunk,)
                    'noisy': noisy_targets[0].cpu().numpy(),
                    'denoised': denoised_actions[0].cpu().numpy(),
                    'noise_mse': noise_mse.item(),
                    'denoising_mse': denoising_mse.item(),
                    'improvement': improvement,
                    'args': args  # Store training arguments for later use
                }
                
                sample_data.append(sample_info)
                
                print(f"  Sample {sample_idx + 1} metrics:")
                print(f"    Noise MSE: {noise_mse.item():.6f}")
                print(f"    Denoising MSE: {denoising_mse.item():.6f}")
                print(f"    Improvement: {improvement:.1f}%")
                print(f"    Images saved to: images/sample_{sample_idx}_*.png")
                
                if len(sample_data) >= num_samples:
                    break
            
            if len(sample_data) >= num_samples:
                break
    
    return sample_data

def visualize_denoising_results(sample_data, output_type='ee_pose', action_chunk=50):
    """
    Visualize denoising results for sampled data
    """
    print(f"\nCreating denoising visualizations...")
    
    # Action dimension names
    if output_type == 'ee_pose':
        action_names = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'gripper']
        action_output_dim = 8
    else:  # position
        action_names = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
        action_output_dim = 7
    
    num_samples = len(sample_data)
    
    # Create images directory if not exists
    os.makedirs('images', exist_ok=True)
    
    # Create comparison plot for each sample
    for sample_idx, sample in enumerate(sample_data):
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        # Reshape data
        original = sample['original'].reshape(action_chunk, action_output_dim)
        noisy = sample['noisy'].reshape(action_chunk, action_output_dim)
        denoised = sample['denoised'].reshape(action_chunk, action_output_dim)
        
        for dim_idx in range(action_output_dim):
            ax = axes[dim_idx]
            
            timesteps = np.arange(action_chunk)
            ax.plot(timesteps, original[:, dim_idx], 'g-', label='Original', linewidth=2.5)
            ax.plot(timesteps, noisy[:, dim_idx], 'r:', label='Noisy', alpha=0.8, linewidth=1.5)
            ax.plot(timesteps, denoised[:, dim_idx], 'b--', label='Denoised', linewidth=2)
            
            ax.set_title(f'{action_names[dim_idx]}', fontsize=12)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Calculate MSE for this dimension
            dim_mse = np.mean((denoised[:, dim_idx] - original[:, dim_idx])**2)
            ax.text(0.02, 0.98, f'MSE: {dim_mse:.4f}', transform=ax.transAxes, 
                   verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle(f'Sample {sample_idx + 1} - Denoising Results\n'
                    f'Data Index: {sample["index"]}, Overall Improvement: {sample["improvement"]:.1f}%', 
                    fontsize=14)
        plt.tight_layout()
        
        # Save individual sample plot
        plt.savefig(f'images/sample_{sample_idx}_denoising_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved denoising comparison for sample {sample_idx + 1}")
    
    # Create summary plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: MSE comparison by sample
    sample_indices = range(len(sample_data))
    noise_mses = [sample['noise_mse'] for sample in sample_data]
    denoising_mses = [sample['denoising_mse'] for sample in sample_data]
    
    x_pos = np.arange(len(sample_data))
    width = 0.35
    
    ax1.bar(x_pos - width/2, noise_mses, width, label='Noise MSE', alpha=0.7, color='red')
    ax1.bar(x_pos + width/2, denoising_mses, width, label='Denoising MSE', alpha=0.7, color='blue')
    
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_title('MSE Comparison by Sample')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'Sample {i+1}' for i in range(len(sample_data))])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Improvement percentage
    improvements = [sample['improvement'] for sample in sample_data]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    
    ax2.bar(x_pos, improvements, color=colors, alpha=0.7)
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Denoising Improvement by Sample')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'Sample {i+1}' for i in range(len(sample_data))])
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('images/denoising_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved summary plot: images/denoising_summary.png")
    
    # Print detailed statistics
    print(f"\nDetailed Sample Statistics:")
    print(f"{'Sample':<8} {'Data Index':<12} {'Noise MSE':<12} {'Denoise MSE':<12} {'Improvement':<12}")
    print(f"{'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    
    for i, sample in enumerate(sample_data):
        print(f"{i+1:<8} {sample['index']:<12} {sample['noise_mse']:<12.6f} "
              f"{sample['denoising_mse']:<12.6f} {sample['improvement']:<12.1f}%")
    
    # Overall statistics
    avg_noise_mse = np.mean([sample['noise_mse'] for sample in sample_data])
    avg_denoising_mse = np.mean([sample['denoising_mse'] for sample in sample_data])
    avg_improvement = np.mean([sample['improvement'] for sample in sample_data])
    
    print(f"{'-'*56}")
    print(f"{'Average':<8} {'':<12} {avg_noise_mse:<12.6f} {avg_denoising_mse:<12.6f} {avg_improvement:<12.1f}%")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test denoising with trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Data directory (override the one from training)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of samples to analyze')
    parser.add_argument('--denoising_steps', type=int, default=20,
                       help='Number of denoising steps')
    
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
    
    # Test denoising with trained model
    sample_data = test_denoising_with_trained_model(
        checkpoint_path=args.checkpoint,
        device=device,
        num_samples=args.num_samples,
        num_denoising_steps=args.denoising_steps,
        data_dir_override=args.data_dir
    )
    
    if sample_data is None:
        print("Failed to load data")
        return
    
    # Get training args from sample_data for visualization
    if sample_data and 'args' in sample_data[0]:
        training_args = sample_data[0]['args']
        output_type = training_args.output_type
        action_chunk = training_args.action_chunk
    else:
        # Fallback defaults
        output_type = 'ee_pose'
        action_chunk = 50
    
    # Visualize results using training parameters
    visualize_denoising_results(sample_data, output_type, action_chunk)
    
    print(f"\n" + "="*60)
    print("TRAINED MODEL DENOISING TEST COMPLETED")
    print(f"="*60)
    print(f"Model: {args.checkpoint}")
    print(f"Analyzed {len(sample_data)} samples")
    print(f"Results saved to 'images/' directory:")
    print(f"  - Sample images: sample_*_frame_*.png")
    print(f"  - Denoising comparisons: sample_*_denoising_comparison.png") 
    print(f"  - Summary plot: denoising_summary.png")

if __name__ == "__main__":
    main()
