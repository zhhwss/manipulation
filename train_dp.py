import argparse
import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from data_loader import create_data_loaders, create_data_loaders_by_trajectory
from model.model_dp import DiffusionPolicy, diffusion_loss
import matplotlib.pyplot as plt
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Robot Grasp Imitation Learning')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/20250730_shui',
                       help='Directory containing json and mp4 files')
    parser.add_argument('--image_type', type=str, default='pointcloud', 
                       choices=['depth_mask', 'color', 'pointcloud', 'depth_mask3', 'color3', 'pointcloud3'],
                       help='Type of image to use')
    parser.add_argument('--image_scale', type=float, default=1.0,
                       help='Scale factor for image size')
    parser.add_argument('--output_type', type=str, default='ee_pose',
                       choices=['position', 'ee_pose', 'delta_ee_pose'],
                       help='Type of output to use')
    parser.add_argument('--action_chunk', type=int, default=50,
                       help='Number of actions to use for prediction')
    parser.add_argument('--action_stride', type=int, default=1,
                       help='Stride of actions to use for prediction')
    parser.add_argument('--window_size', type=int, default=1,
                       help='Number of frames to use for prediction (10)')
    parser.add_argument('--data-aug', type=int, default=1, help='Number of data augmentation (5)')
    
    # Model parameters
    parser.add_argument('--embed_dim', type=int, default=256,
                       help='Output dimension of encoder')
    parser.add_argument('--num_diffusion_steps', type=int, default=100,
                       help='Number of diffusion timesteps')
    parser.add_argument('--unet_layers', type=int, default=1,
                       help='Number of U-Net layers (deprecated, now using ConditionalUnet1D)')
    parser.add_argument('--encoder_type', type=str, default='cnn', choices=['cnn', 'pointnet'],
                       help='Type of encoder: cnn or pointnet')
    parser.add_argument('--use_diffusers_scheduler', action='store_true', default=True,
                       help='Use diffusers scheduler instead of custom scheduler')
    parser.add_argument('--scheduler_type', type=str, default='ddim', 
                       choices=['ddpm', 'ddim'],
                       help='Type of diffusers scheduler (following DP3 uses DDIM)')
    parser.add_argument('--beta_schedule', type=str, default='squaredcos_cap_v2', 
                       choices=['linear', 'scaled_linear', 'squaredcos_cap_v2'],
                       help='Beta schedule for diffusers scheduler (DP3 uses squaredcos_cap_v2)')
    parser.add_argument('--beta_start', type=float, default=0.0001,
                       help='Beta start value for scheduler')
    parser.add_argument('--beta_end', type=float, default=0.02,
                       help='Beta end value for scheduler')
    parser.add_argument('--prediction_type', type=str, default='sample',
                       choices=['epsilon', 'sample', 'v_prediction'],
                       help='Prediction type for scheduler (DP3 uses sample)')
    parser.add_argument('--down_dims', type=str, default='256,512,1024',
                       help='Comma-separated list of down dimensions for ConditionalUnet1D')
    parser.add_argument('--kernel_size', type=int, default=5,
                       help='Kernel size for ConditionalUnet1D')
    parser.add_argument('--n_groups', type=int, default=8,
                       help='Number of groups for GroupNorm in ConditionalUnet1D')

    parser.add_argument('--third_view', action='store_true', default=False,
                       help='Whether to use third view images (default: False)')
    parser.add_argument('--color_view', action='store_true', default=False,
                       help='Whether to use third view images (default: False)')
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=500,
                       help='Number of training epochs')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Fraction of data for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    
    # Device and logging
    parser.add_argument('--device', type=str, default='cuda:3',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--save_interval', type=int, default=100,
                       help='Save model every N epochs')
    
    return parser.parse_args()

def determine_image_size(data_dir, image_type, scale=1.0, third_view=False, color_view=False):
    """Determine image size"""
    # Load a sample image to determine input shape
    import cv2
    import glob
    
    # Find first mp4 file
    mp4_files = glob.glob(os.path.join(data_dir, f'*_{image_type}.mp4'))
    if not mp4_files:
        raise ValueError(f"No {image_type} mp4 files found in {data_dir}")
    
    # Load first frame
    cap = cv2.VideoCapture(mp4_files[0])
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError("Could not read sample frame")
    
    # Process frame
    if image_type == 'depth_mask' or image_type == 'depth_mask3':
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif image_type == "color" or image_type == "color3":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))

    frame = frame.astype(np.float32) / 255.0
    # Add batch and channel dimensions for testing
    if len(frame.shape) == 2:
        frame = frame[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)
    else:
        frame = frame[np.newaxis, :, :, :]  # (1, H, W, 3)
        frame = np.transpose(frame, (0, 3, 1, 2))  # (1, 3, H, W)
        
    image_shape = frame.shape[1:]
    if third_view:
        image_shape = (2*image_shape[0], *image_shape[1:])
    if color_view:
        image_shape = (image_shape[0] + 1, *image_shape[1:])

    print(f"Image size: {image_shape}")
    print(f"CNN input shape: {frame.shape}")
    
    return image_shape

def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch with Enhanced Diffusion Policy"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for indices, frames, qpos, targets in tqdm(train_loader, desc="Training"):
        frames = frames.to(device)
        qpos = qpos.to(device)
        targets = targets.to(device)
        
        # Create batch dict for model.compute_loss
        batch = {
            'frames': frames,
            'qpos': qpos, 
            'actions': targets
        }
        
        # Compute loss using model's built-in method
        loss, loss_dict = model.compute_loss(batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    return avg_loss

def validate(model, val_loader, device, num_denoising_steps=20):
    """
    Validate the Enhanced Diffusion Policy model
    
    For Diffusion Policy validation, we:
    1. Calculate noise prediction loss using model.compute_loss
    2. Perform full denoising process and calculate MSE
    
    Returns validation metrics
    """
    model.eval()
    total_loss = 0
    total_denoising_mse = 0
    num_batches = 0
    
    with torch.no_grad():
        for indices, frames, qpos, targets in tqdm(val_loader, desc="Validating"):
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
            
            # 2. Full denoising evaluation (every few batches to save time)
            if num_batches % 5 == 0:  # Sample every 5 batches
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
            
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_denoising_mse = total_denoising_mse / (num_batches // 5 + 1)  # Approximate
    
    return {
        'noise_loss': avg_loss,              # Noise prediction loss (lower is better)
        'denoising_mse': avg_denoising_mse,  # Full denoising accuracy (lower is better)
    }

def evaluate_diffusion_improvement(current_metrics, best_metrics, improvement_weights=None):
    """
    评估Diffusion Policy模型是否有改进
    
    主要考虑：
    1. Noise Loss - 噪声预测损失，越低越好
    2. Denoising MSE - 完整去噪后的MSE，越低越好
    
    Args:
        current_metrics: 当前验证指标
        best_metrics: 历史最佳指标
        improvement_weights: 各指标的权重 
    
    Returns:
        tuple: (is_improved, improvement_score, reason)
    """
    if best_metrics is None:
        return True, float('inf'), "First validation"
    
    # 默认权重：去噪MSE更重要（最终效果），噪声损失次之
    if improvement_weights is None:
        improvement_weights = {
            'denoising_mse': 0.7,  # 最终去噪效果，权重最高
            'noise_loss': 0.3,     # 噪声预测损失，权重较低
        }
    
    # 计算各指标的相对改进程度
    improvements = {}
    improvement_details = []
    
    # 对于越低越好的指标 (noise_loss, denoising_mse)
    for metric in ['noise_loss', 'denoising_mse']:
        if metric in current_metrics and metric in best_metrics:
            if best_metrics[metric] > 0:
                # 相对改进 = (旧值 - 新值) / 旧值，正值表示改进
                improvement = (best_metrics[metric] - current_metrics[metric]) / best_metrics[metric]
                improvements[metric] = improvement
                improvement_details.append(f"{metric}: {improvement*100:.2f}%")
            else:
                improvements[metric] = 0
    
    # 计算加权改进分数
    weighted_improvement = 0
    total_weight = 0
    
    for metric, weight in improvement_weights.items():
        if metric in improvements:
            weighted_improvement += improvements[metric] * weight
            total_weight += weight
    
    if total_weight > 0:
        improvement_score = weighted_improvement / total_weight
    else:
        improvement_score = 0
    
    # 判断是否改进
    is_improved = improvement_score > 0
    
    # 生成解释
    if is_improved:
        reason = f"Overall improvement: {improvement_score*100:.2f}%. Details: {', '.join(improvement_details)}"
    else:
        reason = f"No improvement: {improvement_score*100:.2f}%. Details: {', '.join(improvement_details)}"
    
    # 额外检查：如果去噪MSE显著改进，优先考虑
    denoising_improvement = improvements.get('denoising_mse', 0)
    if denoising_improvement > 0.02:  # 去噪MSE改进超过2%
        is_improved = True
        reason += f" (Denoising significantly improved: {denoising_improvement*100:.2f}%)"
    
    # 额外检查：如果所有指标都在恶化，肯定不是改进
    if all(imp < -0.01 for imp in improvements.values()):  # 所有指标都恶化超过1%
        is_improved = False
        reason += " (All metrics degraded)"
    
    return is_improved, improvement_score, reason

def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    encoder_suffix = f"_{args.encoder_type}" if args.encoder_type != 'cnn' else ""
    if args.use_diffusers_scheduler:
        scheduler_suffix = f"_{args.scheduler_type}_{args.beta_schedule}"
    else:
        scheduler_suffix = "_custom"
    output_dir = os.path.join('result', f"diffusion_policy_dp3{encoder_suffix}{scheduler_suffix}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir=output_dir)
    
    # Determine image size and CNN output dimension
    image_shape = determine_image_size(
        args.data_dir, args.image_type, args.image_scale, args.third_view, args.color_view
    )
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, full_dataset = create_data_loaders(
        args.data_dir, args.image_type, args.third_view, args.image_scale, args.output_type, args.action_chunk, args.action_stride,
        args.batch_size, args.train_split, args.window_size, args.num_workers, args.data_aug, cache=None, color_view=args.color_view
    )
    # train_loader, val_loader, full_dataset = create_data_loaders(
        # args.data_dir, args.image_type, args.third_view, args.image_scale, args.output_type, args.action_chunk, args.action_stride,
        # args.batch_size, args.train_split, args.window_size, args.num_workers, args.data_aug
    # )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Get dimensions from first batch
    sample_indices, sample_frames, sample_qpos, sample_outputs = next(iter(train_loader))
    qpos_dim = sample_qpos.shape[-1]
    output_dim = sample_outputs.shape[-1]
    
    print(f"QPOS dimension: {qpos_dim}")
    print(f"Output type: {args.output_type}")
    print(f"Output dimension: {output_dim}")
    
    # Parse down_dims
    down_dims = tuple(int(x) for x in args.down_dims.split(','))
    
    # Create Enhanced Diffusion Policy model
    model = DiffusionPolicy(
        image_shape=image_shape,
        embed_dim=args.embed_dim,
        window_size=args.window_size,
        qpos_dim=qpos_dim,
        output_type=args.output_type,
        action_chunk=args.action_chunk,
        unet_layers=args.unet_layers,  # Kept for compatibility but not used
        num_diffusion_steps=args.num_diffusion_steps,
        encoder_type=args.encoder_type,
        use_diffusers_scheduler=args.use_diffusers_scheduler,
        scheduler_type=args.scheduler_type,
        beta_schedule=args.beta_schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        prediction_type=args.prediction_type,
        down_dims=down_dims,
        kernel_size=args.kernel_size,
        n_groups=args.n_groups
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Number of diffusion steps: {args.num_diffusion_steps}")
    print(f"Encoder type: {args.encoder_type}")
    print(f"Using diffusers scheduler: {args.use_diffusers_scheduler}")
    print(f"Scheduler type: {args.scheduler_type}")
    print(f"Beta schedule: {args.beta_schedule}")
    print(f"Beta start/end: {args.beta_start}/{args.beta_end}")
    print(f"Prediction type: {args.prediction_type}")
    print(f"ConditionalUnet1D down_dims: {down_dims}")
    print(f"ConditionalUnet1D kernel_size: {args.kernel_size}")
    print(f"ConditionalUnet1D n_groups: {args.n_groups}")
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    print("Starting Diffusion Policy training...")
    best_metrics = None
    best_improvement_score = float('-inf')
    
    for epoch in range(args.num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        
        # Evaluate improvement
        is_improved, improvement_score, reason = evaluate_diffusion_improvement(val_metrics, best_metrics)
        
        # Log to tensorboard
        writer.add_scalar('Loss/Train_Noise', train_loss, epoch)
        writer.add_scalar('Loss/Val_Noise', val_metrics['noise_loss'], epoch)
        writer.add_scalar('Loss/Val_Denoising_MSE', val_metrics['denoising_mse'], epoch)
        writer.add_scalar('Metrics/Improvement_Score', improvement_score, epoch)
        
        # Print comprehensive metrics
        print(f"\nEpoch {epoch+1}/{args.num_epochs}:")
        print(f"  Train Noise Loss: {train_loss:.6f}")
        print(f"  Val Noise Loss: {val_metrics['noise_loss']:.6f}")
        print(f"  Val Denoising MSE: {val_metrics['denoising_mse']:.6f}")
        print(f"  Improvement: {'✓' if is_improved else '✗'} ({improvement_score:.3f}) - {reason}")
        
        # Save best model based on improvement
        if is_improved:
            best_metrics = val_metrics.copy()
            best_improvement_score = improvement_score
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_metrics': val_metrics,
                'improvement_score': improvement_score,
                'args': args
            }, os.path.join(output_dir, 'best_model.pth'))
            print(f"  ★ Best model saved! (Score: {improvement_score:.4f})")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_metrics': val_metrics,
                'improvement_score': improvement_score,
                'args': args
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    writer.close()
    
    # Final summary
    print(f"\n{'='*60}")
    print("Diffusion Policy Training Summary")
    print(f"{'='*60}")
    if best_metrics is not None:
        print(f"Best validation metrics achieved:")
        print(f"  Noise Loss: {best_metrics['noise_loss']:.6f}")
        print(f"  Denoising MSE: {best_metrics['denoising_mse']:.6f}")
        print(f"  Improvement Score: {best_improvement_score:.4f}")
    else:
        print("No validation metrics recorded")
    
    print(f"\nModel configuration:")
    print(f"  Encoder type: {args.encoder_type}")
    print(f"  Diffusion steps: {args.num_diffusion_steps}")
    print(f"  Use diffusers scheduler: {args.use_diffusers_scheduler}")
    print(f"  ConditionalUnet1D down_dims: {down_dims}")
    print(f"  Kernel size: {args.kernel_size}")
    print(f"  N groups: {args.n_groups}")
    print(f"  Embed dim: {args.embed_dim}")
    print(f"  Action chunk: {args.action_chunk}")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Best model: {os.path.join(output_dir, 'best_model.pth')}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
