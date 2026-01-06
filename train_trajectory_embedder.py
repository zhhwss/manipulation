import argparse
import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_loader import create_data_loaders
from model.trajectory_embedder import (
    TrajectoryEmbedder, 
    compute_trajectory_similarity_loss, 
    compute_regularization_loss
)

def parse_args():
    parser = argparse.ArgumentParser(description='Trajectory Embedding Learning')
    
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
                       help='Number of frames to use for prediction')
    parser.add_argument('--data-aug', type=int, default=1, help='Number of data augmentation')
    
    # Model parameters
    parser.add_argument('--embed_dim', type=int, default=256,
                       help='Feature embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension for trajectory embedder')
    parser.add_argument('--embedding_dim', type=int, default=2,
                       help='Final trajectory embedding dimension')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--train_split', type=float, default=0.6,
                       help='Fraction of data for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    
    # Loss parameters
    parser.add_argument('--margin', type=float, default=1.0,
                       help='Margin for contrastive loss')
    parser.add_argument('--same_weight', type=float, default=1.0,
                       help='Weight for same trajectory loss')
    parser.add_argument('--diff_weight', type=float, default=1.0,
                       help='Weight for different trajectory loss')
    parser.add_argument('--reg_weight', type=float, default=0.01,
                       help='Weight for regularization loss')
    parser.add_argument('--target_std', type=float, default=0.1,
                       help='Target standard deviation for regularization')
    
    # Device and logging
    parser.add_argument('--device', type=str, default='cuda:3',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--save_interval', type=int, default=50,
                       help='Save model every N epochs')
    parser.add_argument('--plot_interval', type=int, default=20,
                       help='Plot embeddings every N epochs')
    
    return parser.parse_args()

def determine_image_size(data_dir, image_type, scale=1.0):
    """Determine image size from sample data"""
    import cv2
    import glob
    
    mp4_files = glob.glob(os.path.join(data_dir, f'*_{image_type}.mp4'))
    if not mp4_files:
        raise ValueError(f"No {image_type} mp4 files found in {data_dir}")
    
    cap = cv2.VideoCapture(mp4_files[0])
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError("Could not read sample frame")
    
    if image_type == 'depth_mask' or image_type == 'depth_mask3':
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif image_type == "color" or image_type == "color3":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
    frame = frame.astype(np.float32) / 255.0
    
    if len(frame.shape) == 2:
        frame = frame[np.newaxis, np.newaxis, :, :]
    else:
        frame = frame[np.newaxis, :, :, :]
        frame = np.transpose(frame, (0, 3, 1, 2))
        
    image_shape = frame.shape[1:]
    return image_shape

def train_epoch(model, train_loader, optimizer, device, args):
    """Train for one epoch"""
    model.train()
    total_sim_loss = 0
    total_reg_loss = 0
    total_loss = 0
    num_batches = 0
    
    for indices, frames, qpos, targets in tqdm(train_loader, desc="Training"):
        indices = indices.to(device)
        frames = frames.to(device)
        qpos = qpos.to(device)
        targets = targets.to(device)
        
        # Forward pass
        means, stds = model(frames, qpos, targets)
        
        # Compute losses
        sim_loss = compute_trajectory_similarity_loss(
            means, stds, indices, 
            margin=args.margin,
            same_weight=args.same_weight,
            diff_weight=args.diff_weight
        )
        
        reg_loss = compute_regularization_loss(stds, target_std=args.target_std)
        
        # Total loss
        loss = sim_loss + args.reg_weight * reg_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_sim_loss += sim_loss.item()
        total_reg_loss += reg_loss.item()
        total_loss += loss.item()
        num_batches += 1
    
    return {
        'sim_loss': total_sim_loss / num_batches,
        'reg_loss': total_reg_loss / num_batches,
        'total_loss': total_loss / num_batches
    }

def validate(model, val_loader, device, args):
    """Validate the model"""
    model.eval()
    total_sim_loss = 0
    total_reg_loss = 0
    total_loss = 0
    num_batches = 0
    
    all_means = []
    all_stds = []
    all_indices = []
    
    with torch.no_grad():
        for indices, frames, qpos, targets in tqdm(val_loader, desc="Validating"):
            indices = indices.to(device)
            frames = frames.to(device)
            qpos = qpos.to(device)
            targets = targets.to(device)
            
            means, stds = model(frames, qpos, targets)
            
            # Compute losses
            sim_loss = compute_trajectory_similarity_loss(
                means, stds, indices,
                margin=args.margin,
                same_weight=args.same_weight,
                diff_weight=args.diff_weight
            )
            
            reg_loss = compute_regularization_loss(stds, target_std=args.target_std)
            loss = sim_loss + args.reg_weight * reg_loss
            
            total_sim_loss += sim_loss.item()
            total_reg_loss += reg_loss.item()
            total_loss += loss.item()
            num_batches += 1
            
            # Collect embeddings for visualization
            all_means.append(means.cpu())
            all_stds.append(stds.cpu())
            all_indices.append(indices.cpu())
    
    # Concatenate all embeddings
    all_means = torch.cat(all_means, dim=0)
    all_stds = torch.cat(all_stds, dim=0)
    all_indices = torch.cat(all_indices, dim=0)
    
    return {
        'sim_loss': total_sim_loss / num_batches,
        'reg_loss': total_reg_loss / num_batches,
        'total_loss': total_loss / num_batches,
        'embeddings': (all_means, all_stds, all_indices)
    }

def plot_embeddings(means, stds, indices, epoch, output_dir):
    """Plot 2D embeddings with uncertainty ellipses"""
    plt.figure(figsize=(12, 10))
    
    # Get unique trajectory indices
    unique_indices = torch.unique(indices)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_indices)))
    
    for i, traj_idx in enumerate(unique_indices):
        mask = indices == traj_idx
        traj_means = means[mask]
        traj_stds = stds[mask]
        
        color = colors[i % len(colors)]
        
        # Plot mean points
        plt.scatter(traj_means[:, 0], traj_means[:, 1], 
                   c=[color], label=f'Trajectory {traj_idx.item()}', alpha=0.7, s=30)
        
        # Plot uncertainty ellipses (sample a few for clarity)
        for j in range(min(5, len(traj_means))):
            mean = traj_means[j]
            std = traj_stds[j]
            
            ellipse = plt.matplotlib.patches.Ellipse(
                (mean[0], mean[1]), 
                width=2*std[0], 
                height=2*std[1],
                alpha=0.2, 
                color=color
            )
            plt.gca().add_patch(ellipse)
    
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.xlabel('Embedding Dim 1')
    plt.ylabel('Embedding Dim 2')
    plt.title(f'Trajectory Embeddings - Epoch {epoch}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, f'embeddings_epoch_{epoch}.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()

def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join('result', f"trajectory_embedder_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir=output_dir)
    
    # Determine image size
    image_shape = determine_image_size(args.data_dir, args.image_type, args.image_scale)
    print(f"Image shape: {image_shape}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, full_dataset = create_data_loaders(
        args.data_dir, args.image_type, args.image_scale, args.output_type, 
        args.action_chunk, args.action_stride, args.batch_size, args.train_split, 
        args.window_size, args.num_workers, args.data_aug
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Get dimensions from first batch
    sample_indices, sample_frames, sample_qpos, sample_targets = next(iter(train_loader))
    qpos_dim = sample_qpos.shape[-1]
    action_chunk = sample_targets.shape[-1]
    
    print(f"QPOS dimension: {qpos_dim}")
    print(f"Action chunk: {action_chunk}")
    print(f"Unique trajectories in sample: {len(torch.unique(sample_indices))}")
    
    # Create model
    model = TrajectoryEmbedder(
        image_shape=image_shape,
        embed_dim=args.embed_dim,
        window_size=args.window_size,
        qpos_dim=qpos_dim,
        action_chunk=action_chunk,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, args)
        
        # Validate
        val_metrics = validate(model, val_loader, device, args)
        
        # Update learning rate
        scheduler.step(val_metrics['total_loss'])
        
        # Log metrics
        writer.add_scalar('Loss/Train_Total', train_metrics['total_loss'], epoch)
        writer.add_scalar('Loss/Train_Similarity', train_metrics['sim_loss'], epoch)
        writer.add_scalar('Loss/Train_Regularization', train_metrics['reg_loss'], epoch)
        writer.add_scalar('Loss/Val_Total', val_metrics['total_loss'], epoch)
        writer.add_scalar('Loss/Val_Similarity', val_metrics['sim_loss'], epoch)
        writer.add_scalar('Loss/Val_Regularization', val_metrics['reg_loss'], epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Epoch {epoch+1}/{args.num_epochs}:")
        print(f"  Train - Total: {train_metrics['total_loss']:.6f}, "
              f"Sim: {train_metrics['sim_loss']:.6f}, Reg: {train_metrics['reg_loss']:.6f}")
        print(f"  Val   - Total: {val_metrics['total_loss']:.6f}, "
              f"Sim: {val_metrics['sim_loss']:.6f}, Reg: {val_metrics['reg_loss']:.6f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Plot embeddings
        if (epoch + 1) % args.plot_interval == 0:
            means, stds, indices = val_metrics['embeddings']
            plot_embeddings(means, stds, indices, epoch + 1, output_dir)
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'args': args
            }, os.path.join(output_dir, 'best_model.pth'))
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'args': args
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Final embedding plot
    val_metrics = validate(model, val_loader, device, args)
    means, stds, indices = val_metrics['embeddings']
    plot_embeddings(means, stds, indices, args.num_epochs, output_dir)
    
    writer.close()
    print(f"Training completed. Results saved to {output_dir}")

if __name__ == '__main__':
    main() 