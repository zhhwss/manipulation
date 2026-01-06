import argparse
import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from data_loader import create_data_loaders, create_data_loaders_by_trajectory
from model.model_bn1 import RobotGraspModel
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Robot Grasp Imitation Learning with Best of N')
    
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
    parser.add_argument('--num_heads', type=int, default=3,
                       help='Number of output heads for Best of N')

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

def train_epoch(model, train_loader, criterion, optimizer, device, epoch=0):
    """Train for one epoch with Best of N strategy (per sample) - optimized"""
    model.train()
    total_loss = 0
    num_batches = 0
    head_selection_stats = {i: 0 for i in range(model.num_heads)}
    
    # Use all heads training for first 10 epochs
    use_all_heads = epoch < 1
    
    for indices, frames, qpos, targets in tqdm(train_loader, desc=f"Training ({'All Heads' if use_all_heads else 'Best of N'})"):
        frames = frames.to(device)
        qpos = qpos.to(device)
        targets = targets.to(device)
        
        batch_size = targets.shape[0]
        
        # Forward pass - get all head outputs
        outputs = model(frames, qpos, training=True)  # List of outputs from all heads
        
        if use_all_heads:
            # First 10 epochs: train all heads equally
            total_batch_loss = 0
            for i, output in enumerate(outputs):
                head_loss = criterion(output, targets)
                total_batch_loss += head_loss
                # Update statistics (equal selection for all heads)
                head_selection_stats[i] += batch_size // model.num_heads
                if i < batch_size % model.num_heads:  # Handle remainder
                    head_selection_stats[i] += 1
            
            batch_loss = total_batch_loss / model.num_heads
            
            # Update model statistics (equal for all heads)
            for head_idx in range(model.num_heads):
                model.head_selection_counts[head_idx] += batch_size // model.num_heads
                if head_idx < batch_size % model.num_heads:
                    model.head_selection_counts[head_idx] += 1
            model.total_selections += batch_size
            
        else:
            # After epoch 10: use Best of N strategy
        # Compute per-sample losses for all heads efficiently
        # Shape: [num_heads, batch_size]
        per_sample_losses = []
        for output in outputs:
            # Compute MSE loss per sample (no reduction)
            sample_wise_loss = torch.mean((output - targets) ** 2, dim=1)  # [batch_size]
            per_sample_losses.append(sample_wise_loss)
        
        # Stack to get [num_heads, batch_size]
        losses_tensor = torch.stack(per_sample_losses, dim=0)  # [num_heads, batch_size]
        
        # Find the best head for each sample (vectorized)
        min_loss_indices = torch.argmin(losses_tensor, dim=0)  # [batch_size]
        
        # Get the minimum losses for each sample (vectorized)
        batch_indices = torch.arange(batch_size, device=device)
        selected_losses = losses_tensor[min_loss_indices, batch_indices]  # [batch_size]
        
        # Average loss for the batch
        batch_loss = torch.mean(selected_losses)
        
        # Update head selection statistics (vectorized)
        for head_idx in range(model.num_heads):
            count = torch.sum(min_loss_indices == head_idx).item()
            head_selection_stats[head_idx] += count
            if count > 0:
                model.head_selection_counts[head_idx] += count
        model.total_selections += batch_size
        
        # Backward pass
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        
        total_loss += batch_loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    # Get head statistics
    head_stats = model.get_head_statistics()
    
    return avg_loss, head_selection_stats, head_stats

def validate(model, val_loader, criterion, device):
    """Validate the model with Best of N strategy (per sample) - optimized"""
    model.eval()
    total_loss = 0
    head_selection_stats = {i: 0 for i in range(model.num_heads)}
    all_losses = {i: 0.0 for i in range(model.num_heads)}
    total_samples = 0
    
    with torch.no_grad():
        for indices, frames, qpos, targets in tqdm(val_loader, desc="Validating"):
            frames = frames.to(device)
            qpos = qpos.to(device)
            targets = targets.to(device)
            
            batch_size = targets.shape[0]
            total_samples += batch_size
            
            # Forward pass - get all head outputs
            outputs = model(frames, qpos, training=True)  # List of outputs from all heads
            
            # Compute per-sample losses for all heads efficiently
            # Shape: [num_heads, batch_size]
            per_sample_losses = []
            for i, output in enumerate(outputs):
                # Compute MSE loss per sample (no reduction)
                sample_wise_loss = torch.mean((output - targets) ** 2, dim=1)  # [batch_size]
                per_sample_losses.append(sample_wise_loss)
                
                # Add to total loss for this head (vectorized)
                all_losses[i] += torch.sum(sample_wise_loss).item()
            
            # Stack to get [num_heads, batch_size]
            losses_tensor = torch.stack(per_sample_losses, dim=0)  # [num_heads, batch_size]
            
            # Find the best head for each sample (vectorized)
            min_loss_indices = torch.argmin(losses_tensor, dim=0)  # [batch_size]
            
            # Get the minimum losses for each sample (vectorized)
            batch_indices = torch.arange(batch_size, device=device)
            selected_losses = losses_tensor[min_loss_indices, batch_indices]  # [batch_size]
            
            # Accumulate total loss (vectorized)
            total_loss += torch.sum(selected_losses).item()
            
            # Update head selection statistics (vectorized) - only for this validation
            for head_idx in range(model.num_heads):
                count = torch.sum(min_loss_indices == head_idx).item()
                head_selection_stats[head_idx] += count
    
    avg_loss = total_loss / total_samples
    
    # Average losses for each head
    for i in range(model.num_heads):
        all_losses[i] /= total_samples
    
    # Calculate validation-based head statistics (fresh calculation, no accumulation)
    val_head_stats = {
        'selection_counts': list(head_selection_stats.values()),
        'selection_rates': [count / total_samples for count in head_selection_stats.values()],
        'total_selections': total_samples,
        'valid_heads': [i for i, rate in enumerate([count / total_samples for count in head_selection_stats.values()]) 
                       if rate >= (1.0 / 6.0)]
    }
    
    # Ensure at least one head is valid
    if len(val_head_stats['valid_heads']) == 0:
        val_head_stats['valid_heads'] = [0]
    
    return avg_loss, head_selection_stats, all_losses, val_head_stats

def update_model_head_stats_from_validation(model, val_head_stats):
    """Update model's head selection statistics based on validation results"""
    # Reset model statistics to validation-based statistics
    model.head_selection_counts.zero_()
    model.total_selections.zero_()
    
    # Set new statistics based on validation
    for i, count in enumerate(val_head_stats['selection_counts']):
        model.head_selection_counts[i] = count
    model.total_selections.fill_(val_head_stats['total_selections'])

def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join('result', f"best_of_n1_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir=output_dir)
    
    # Determine image size and CNN output dimension
    image_shape = determine_image_size(
        args.data_dir, args.image_type, args.image_scale, args.third_view, args.color_view
    )
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, full_dataset = create_data_loaders_by_trajectory(
        args.data_dir, args.image_type, args.third_view, args.image_scale, args.output_type, args.action_chunk, args.action_stride,
        args.batch_size, args.train_split, args.window_size, args.num_workers, args.data_aug, cache=None, color_view=args.color_view
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Get dimensions from first batch
    sample_indices, sample_frames, sample_qpos, sample_outputs = next(iter(train_loader))
    qpos_dim = sample_qpos.shape[-1]
    output_dim = sample_outputs.shape[-1]
    
    print(f"QPOS dimension: {qpos_dim}")
    print(f"Output type: {args.output_type}")
    print(f"Output dimension: {output_dim}")
    print(f"Number of heads: {args.num_heads}")
    
    # Create model
    model = RobotGraspModel(
        image_shape=image_shape,
        embed_dim=args.embed_dim,
        window_size=args.window_size,
        qpos_dim=qpos_dim,
        output_type=args.output_type,
        action_chunk=args.action_chunk,
        unet_layers=2,
        num_heads=args.num_heads
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    print("Starting Best of N training...")
    print("  - Epochs 1-10: All heads will be trained equally")
    print("  - Epochs 11+: Best of N strategy (per-sample head selection)")
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        # Train
        train_loss, train_head_stats, train_head_info = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss, val_head_stats, val_head_losses, val_head_info = validate(model, val_loader, criterion, device)
        
        # Log to tensorboard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        
        # Log individual head losses during validation
        for i in range(model.num_heads):
            writer.add_scalar(f'Loss/Val_Head_{i}', val_head_losses[i], epoch)
        
        # Log head selection rates (training and validation)
        for i in range(model.num_heads):
            if train_head_info['total_selections'] > 0:
                train_rate = train_head_info['selection_rates'][i]
                writer.add_scalar(f'Head_Selection/Train_Head_{i}_Rate', train_rate, epoch)
            
            val_rate = val_head_info['selection_rates'][i]
            writer.add_scalar(f'Head_Selection/Val_Head_{i}_Rate', val_rate, epoch)
        
        # Print comprehensive metrics
        training_mode = "All Heads" if epoch < 10 else "Best of N"
        print(f"\nEpoch {epoch+1}/{args.num_epochs} ({training_mode}):")
        print(f"  Train Loss (min): {train_loss:.6f}")
        print(f"  Val Loss (min): {val_loss:.6f}")
        
        # Print individual head validation losses
        print("  Val Head Losses:", end="")
        for i in range(model.num_heads):
            print(f" Head_{i}: {val_head_losses[i]:.6f}", end="")
        print()
        
        # Print head selection statistics
        print(f"  Head Selection (Train): {train_head_stats}")
        print(f"  Head Selection (Val): {val_head_stats}")
        
        # Print head statistics from both training and validation
        if train_head_info['total_selections'] > 0:
            print(f"  Train Head Rates: {[f'{rate:.3f}' for rate in train_head_info['selection_rates']]}")
            print(f"  Train Valid Heads: {train_head_info['valid_heads']}")
        
        print(f"  Val Head Rates: {[f'{rate:.3f}' for rate in val_head_info['selection_rates']]}")
        print(f"  Val Valid Heads: {val_head_info['valid_heads']}")
        
        # Save best model based on minimum validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Update model's head statistics based on validation results (only for best checkpoint)
            update_model_head_stats_from_validation(model, val_head_info)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),  # Contains val-based head statistics
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_head_losses': val_head_losses,
                'train_head_statistics': train_head_info,  # Training-based statistics
                'val_head_statistics': val_head_info,      # Validation-based statistics (used in model)
                'args': args
            }, os.path.join(output_dir, 'best_model.pth'))
            print(f"  ★ Best model saved! (Val Loss: {val_loss:.6f}) - Using validation head statistics")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),  # Contains val-based head statistics
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_head_losses': val_head_losses,
                'train_head_statistics': train_head_info,  # Training-based statistics
                'val_head_statistics': val_head_info,      # Validation-based statistics (used in model)
                'args': args
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    writer.close()
    
    # Final summary
    print(f"\n{'='*80}")
    print("Best of N Training Summary")
    print(f"{'='*80}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    # Final head statistics (based on last validation)
    final_stats = model.get_head_statistics()
    print(f"\nFinal Head Statistics (based on validation data):")
    print(f"  Total selections: {final_stats['total_selections']}")
    print(f"  Selection counts: {final_stats['selection_counts']}")
    print(f"  Selection rates: {[f'{rate:.3f}' for rate in final_stats['selection_rates']]}")
    print(f"  Valid heads (>1/6): {final_stats['valid_heads']}")
    print(f"\nNote: Model's head statistics are updated from validation data for each best checkpoint.")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Best model: {os.path.join(output_dir, 'best_model.pth')}")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
