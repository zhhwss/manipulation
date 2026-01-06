import argparse
import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from data_loader import create_data_loaders, create_data_loaders_by_trajectory
from model.model_mdn import RobotGraspModel_v2, mdn_negative_log_likelihood, sample_from_mdn
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
                       help='Output dimension of CNN')
    parser.add_argument('--num_mixtures', type=int, default=5,
                       help='Number of mixture components for MDN')
    parser.add_argument('--unet_layers', type=int, default=1,
                       help='Number of U-Net layers')

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
    """Train for one epoch with MDN"""
    model.train()
    total_loss = 0
    total_mixture_entropy = 0
    num_batches = 0
    
    for indices, frames, qpos, targets in tqdm(train_loader, desc="Training"):
        frames = frames.to(device)
        qpos = qpos.to(device)
        targets = targets.to(device)
        
        # Forward pass (MDN mode)
        mdn_params = model(frames, qpos, mode='mdn_params')
        
        # Calculate MDN loss (NLL)
        loss = mdn_negative_log_likelihood(mdn_params, targets)
        
        # Calculate mixture entropy for monitoring
        log_pi = mdn_params['log_pi']  # (batch_size, num_mixtures)
        pi = torch.exp(log_pi)
        mixture_entropy = -(pi * log_pi).sum(dim=1).mean()  # Average entropy across batch
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_mixture_entropy += mixture_entropy.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_entropy = total_mixture_entropy / num_batches
    
    return avg_loss, avg_entropy

def validate(model, val_loader, device):
    """
    Validate the MDN model using multiple metrics
    
    For MDN validation, we need to consider:
    1. Negative Log Likelihood (NLL) - primary loss
    2. Sample quality - how well samples match targets
    3. Mixture diversity - are all components being used?
    4. Prediction accuracy - deterministic evaluation
    
    Returns dict with comprehensive validation metrics
    """
    model.eval()
    total_nll = 0
    total_sample_mse = 0
    total_max_prob_mse = 0
    total_mixture_entropy = 0
    total_mixture_usage = 0
    num_batches = 0
    
    with torch.no_grad():
        for indices, frames, qpos, targets in tqdm(val_loader, desc="Validating"):
            frames = frames.to(device)
            qpos = qpos.to(device)
            targets = targets.to(device)
            
            # 1. Get MDN parameters and calculate NLL
            mdn_params = model(frames, qpos, mode='mdn_params')
            nll_loss = mdn_negative_log_likelihood(mdn_params, targets)
            
            # 2. Sample from MDN and calculate sample quality
            sampled_predictions = model(frames, qpos, mode='sample')
            sample_mse = nn.MSELoss()(sampled_predictions, targets)
            
            # 3. Get max probability predictions (deterministic)
            max_prob_predictions = model(frames, qpos, mode='max_prob')
            max_prob_mse = nn.MSELoss()(max_prob_predictions, targets)
            
            # 4. Analyze mixture components
            log_pi = mdn_params['log_pi']  # (batch_size, num_mixtures)
            pi = torch.exp(log_pi)
            
            # Mixture entropy (higher = more diverse usage)
            mixture_entropy = -(pi * log_pi).sum(dim=1).mean()
            
            # Mixture usage (how many components are actively used)
            # Component is "used" if its average weight > threshold
            avg_weights = pi.mean(dim=0)  # (num_mixtures,)
            usage_threshold = 0.05  # 5% minimum usage
            active_components = (avg_weights > usage_threshold).sum().float()
            mixture_usage = active_components / pi.shape[1]  # Fraction of components used
            
            # Accumulate metrics
            total_nll += nll_loss.item()
            total_sample_mse += sample_mse.item()
            total_max_prob_mse += max_prob_mse.item()
            total_mixture_entropy += mixture_entropy.item()
            total_mixture_usage += mixture_usage.item()
            num_batches += 1
    
    # Calculate averages
    metrics = {
        'nll': total_nll / num_batches,                    # Primary loss (lower is better)
        'sample_mse': total_sample_mse / num_batches,      # Sample quality (lower is better)
        'max_prob_mse': total_max_prob_mse / num_batches,  # Deterministic accuracy (lower is better)
        'mixture_entropy': total_mixture_entropy / num_batches,  # Diversity (higher is better)
        'mixture_usage': total_mixture_usage / num_batches,      # Component usage (higher is better)
    }
    
    return metrics

def evaluate_mdn_improvement(current_metrics, best_metrics, improvement_weights=None):
    """
    评估MDN模型是否有改进
    
    MDN模型的改进不能仅看单一指标，需要综合考虑：
    1. NLL (负对数似然) - 主要损失，越低越好
    2. 预测准确性 - 样本质量和确定性预测，越低越好  
    3. 模型多样性 - 混合组分的使用情况，适度即可
    
    Args:
        current_metrics: 当前验证指标
        best_metrics: 历史最佳指标
        improvement_weights: 各指标的权重 
    
    Returns:
        tuple: (is_improved, improvement_score, reason)
    """
    if best_metrics is None:
        return True, float('inf'), "First validation"
    
    # 默认权重：NLL最重要，预测准确性次之，多样性作为参考
    if improvement_weights is None:
        improvement_weights = {
            'nll': 0.5,           # 主要损失，权重最高
            'max_prob_mse': 0.3,  # 确定性预测准确性
            'sample_mse': 0.15,   # 采样质量
            'mixture_entropy': 0.05  # 混合多样性（权重较小）
        }
    
    # 计算各指标的相对改进程度
    improvements = {}
    
    # 对于越低越好的指标 (NLL, MSE)
    for metric in ['nll', 'max_prob_mse', 'sample_mse']:
        if metric in current_metrics and metric in best_metrics:
            if best_metrics[metric] > 0:
                # 相对改进 = (旧值 - 新值) / 旧值，正值表示改进
                improvements[metric] = (best_metrics[metric] - current_metrics[metric]) / best_metrics[metric]
            else:
                improvements[metric] = 0
    
    # 对于越高越好的指标 (entropy)
    for metric in ['mixture_entropy']:
        if metric in current_metrics and metric in best_metrics:
            if best_metrics[metric] > 0:
                # 相对改进 = (新值 - 旧值) / 旧值，正值表示改进
                improvements[metric] = (current_metrics[metric] - best_metrics[metric]) / best_metrics[metric]
            else:
                improvements[metric] = current_metrics[metric] if current_metrics[metric] > 0 else 0
    
    # 计算加权改进分数
    weighted_improvement = 0
    total_weight = 0
    improvement_details = []
    
    for metric, weight in improvement_weights.items():
        if metric in improvements:
            weighted_improvement += improvements[metric] * weight
            total_weight += weight
            improvement_details.append(f"{metric}: {improvements[metric]*100:.2f}%")
    
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
    
    # 额外检查：如果NLL显著改进，即使其他指标略差也认为是改进
    nll_improvement = improvements.get('nll', 0)
    if nll_improvement > 0.02:  # NLL改进超过2%
        is_improved = True
        reason += f" (NLL significantly improved: {nll_improvement*100:.2f}%)"
    
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
    output_dir = os.path.join('result', f"mdn_training_{timestamp}")
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
    
    # Create MDN model
    model = RobotGraspModel_v2(
        image_shape=image_shape,
        embed_dim=args.embed_dim,
        window_size=args.window_size,
        qpos_dim=qpos_dim,
        output_type=args.output_type,
        action_chunk=args.action_chunk,
        unet_layers=args.unet_layers,
        num_mixtures=args.num_mixtures
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Number of mixture components: {args.num_mixtures}")
    
    # Setup training (no criterion needed for MDN - using NLL loss)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    print("Starting MDN training...")
    best_metrics = None
    best_improvement_score = float('-inf')
    
    for epoch in range(args.num_epochs):
        # Train
        train_loss, train_entropy = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        
        # Evaluate improvement
        is_improved, improvement_score, reason = evaluate_mdn_improvement(val_metrics, best_metrics)
        
        # Log to tensorboard
        writer.add_scalar('Loss/Train_NLL', train_loss, epoch)
        writer.add_scalar('Loss/Val_NLL', val_metrics['nll'], epoch)
        writer.add_scalar('Loss/Val_Sample_MSE', val_metrics['sample_mse'], epoch)
        writer.add_scalar('Loss/Val_MaxProb_MSE', val_metrics['max_prob_mse'], epoch)
        writer.add_scalar('Metrics/Train_Entropy', train_entropy, epoch)
        writer.add_scalar('Metrics/Val_Mixture_Entropy', val_metrics['mixture_entropy'], epoch)
        writer.add_scalar('Metrics/Val_Mixture_Usage', val_metrics['mixture_usage'], epoch)
        writer.add_scalar('Metrics/Improvement_Score', improvement_score, epoch)
        
        # Print comprehensive metrics
        print(f"\nEpoch {epoch+1}/{args.num_epochs}:")
        print(f"  Train NLL: {train_loss:.6f}, Train Entropy: {train_entropy:.6f}")
        print(f"  Val NLL: {val_metrics['nll']:.6f}")
        print(f"  Val Sample MSE: {val_metrics['sample_mse']:.6f}")
        print(f"  Val MaxProb MSE: {val_metrics['max_prob_mse']:.6f}")
        print(f"  Mixture Entropy: {val_metrics['mixture_entropy']:.6f}")
        print(f"  Mixture Usage: {val_metrics['mixture_usage']:.3f} ({val_metrics['mixture_usage']*args.num_mixtures:.1f}/{args.num_mixtures} components)")
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
                'train_entropy': train_entropy,
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
                'train_entropy': train_entropy,
                'val_metrics': val_metrics,
                'improvement_score': improvement_score,
                'args': args
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    writer.close()
    
    # Final summary
    print(f"\n{'='*60}")
    print("MDN Training Summary")
    print(f"{'='*60}")
    if best_metrics is not None:
        print(f"Best validation metrics achieved:")
        print(f"  NLL Loss: {best_metrics['nll']:.6f}")
        print(f"  Sample MSE: {best_metrics['sample_mse']:.6f}")
        print(f"  MaxProb MSE: {best_metrics['max_prob_mse']:.6f}")
        print(f"  Mixture Entropy: {best_metrics['mixture_entropy']:.6f}")
        print(f"  Mixture Usage: {best_metrics['mixture_usage']:.3f} ({best_metrics['mixture_usage']*args.num_mixtures:.1f}/{args.num_mixtures} components)")
        print(f"  Improvement Score: {best_improvement_score:.4f}")
    else:
        print("No validation metrics recorded")
    
    print(f"\nModel configuration:")
    print(f"  Mixtures: {args.num_mixtures}")
    print(f"  U-Net layers: {args.unet_layers}")
    print(f"  Embed dim: {args.embed_dim}")
    print(f"  Action chunk: {args.action_chunk}")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Best model: {os.path.join(output_dir, 'best_model.pth')}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
