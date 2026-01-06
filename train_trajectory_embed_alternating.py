import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import json

from data_loader import create_data_loaders_by_trajectory
from model.model_trajectory_embed import RobotGraspModelWithTrajectoryEmbed

def get_unique_trajectories(data_loader):
    """Get all unique trajectory indices from the data loader"""
    unique_indices = set()
    for batch in data_loader:
        indices, _, _, _ = batch
        unique_indices.update(indices.cpu().numpy())
    return sorted(list(unique_indices))

def train_network_epoch(model, data_loader, criterion, optimizer, device):
    """Train the network parameters (freeze trajectory embeddings)"""
    model.train()
    
    # Freeze trajectory embeddings
    model.trajectory_embeddings.requires_grad = False
    
    total_loss = 0.0
    num_batches = 0
    
    for batch in data_loader:
        indices, frames, qpos, target = batch
        indices = indices.to(device)
        frames = frames.to(device)
        qpos = qpos.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(frames, qpos, indices)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check for NaN gradients
        has_nan = False
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN gradient in {name}")
                has_nan = True
        
        if not has_nan:
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    # Unfreeze trajectory embeddings
    model.trajectory_embeddings.requires_grad = True
    
    return total_loss / num_batches

def update_trajectory_embeddings(model, data_loader, optimizer, device, args):
    """Update trajectory embeddings using validation data (freeze network)"""
    model.train()
    
    # Freeze all parameters except trajectory embeddings
    for name, param in model.named_parameters():
        if 'trajectory_embeddings' not in name:
            param.requires_grad = False
    
    total_loss = 0.0
    num_batches = 0
    trajectory_losses = {}
    
    for batch in data_loader:
        indices, frames, qpos, target = batch
        indices = indices.to(device)
        frames = frames.to(device)
        qpos = qpos.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(frames, qpos, indices)
        loss = nn.functional.mse_loss(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check for NaN gradients
        if not torch.isnan(model.trajectory_embeddings.grad).any():
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Track per-trajectory losses
        batch_losses = nn.functional.mse_loss(output, target, reduction='none').mean(dim=1)
        for i, idx in enumerate(indices.cpu().numpy()):
            if idx not in trajectory_losses:
                trajectory_losses[idx] = []
            trajectory_losses[idx].append(batch_losses[i].item())
    
    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    return total_loss / num_batches, trajectory_losses

def validate(model, data_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    all_means = []
    all_stds = []
    all_indices = []
    
    with torch.no_grad():
        for batch in data_loader:
            indices, frames, qpos, target = batch
            indices = indices.to(device)
            frames = frames.to(device)
            qpos = qpos.to(device)
            target = target.to(device)
            
            # Forward pass
            output = model(frames, qpos, indices)
            loss = nn.functional.mse_loss(output, target)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Collect embedding info for plotting
            means, stds = model.get_trajectory_embeddings_info()
            for idx in indices.cpu().numpy():
                all_indices.append(idx)
                all_means.append(means[idx].cpu().numpy())
                all_stds.append(stds[idx].cpu().numpy())
    
    return total_loss / num_batches, np.array(all_means), np.array(all_stds), np.array(all_indices)

def plot_trajectory_embeddings(means, stds, indices, epoch, output_dir):
    """Plot trajectory embeddings with uncertainty ellipses"""
    plt.figure(figsize=(12, 8))
    
    # Plot means as points
    unique_indices = np.unique(indices)
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(unique_indices), 10)))
    
    for i, idx in enumerate(unique_indices[:50]):  # Plot first 50 trajectories
        mask = indices == idx
        if np.sum(mask) == 0:
            continue
            
        mean = means[mask][0]  # Take first occurrence
        std = stds[mask][0]
        
        color = colors[i % len(colors)]
        
        # Plot mean
        plt.scatter(mean[0], mean[1], c=[color], s=50, alpha=0.7, label=f'Traj {idx}' if i < 10 else "")
        
        # Plot uncertainty ellipse
        from matplotlib.patches import Ellipse
        ellipse = Ellipse(mean, 2*std[0], 2*std[1], alpha=0.3, color=color)
        plt.gca().add_patch(ellipse)
    
    plt.xlabel('Embedding X')
    plt.ylabel('Embedding Y')
    plt.title(f'Trajectory Embeddings (Epoch {epoch})')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    if len(unique_indices) <= 10:
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'trajectory_embeddings_epoch_{epoch}.png'), dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train Robot Grasp Model with Trajectory Embedding')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to data directory')
    parser.add_argument('--image_type', type=str, default='depth_mask',
                       choices=['depth_mask', 'color', 'pointcloud'],
                       help='Type of image data')
    parser.add_argument('--image_scale', type=float, default=1.0,
                       help='Scale factor for images')
    parser.add_argument('--output_type', type=str, default='ee_pose',
                       choices=['position', 'ee_pose'],
                       help='Type of output prediction')
    parser.add_argument('--action_chunk', type=int, default=50,
                       help='Number of future actions to predict')
    parser.add_argument('--action_stride', type=int, default=1,
                       help='Stride for action sampling')
    parser.add_argument('--window_size', type=int, default=1,
                       help='Number of frames to use for prediction')
    parser.add_argument('--data-aug', type=int, default=1,
                       help='Data augmentation factor')
    
    # Model parameters
    parser.add_argument('--embed_dim', type=int, default=256,
                       help='Feature embedding dimension')
    parser.add_argument('--num_trajectories', type=int, default=1000,
                       help='Maximum number of trajectories')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate for network parameters')
    parser.add_argument('--embed_learning_rate', type=float, default=1e-3,
                       help='Learning rate for trajectory embeddings')
    parser.add_argument('--num_epochs', type=int, default=300,
                       help='Number of training epochs')
    parser.add_argument('--train_split', type=float, default=0.6,
                       help='Fraction of data for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--alternating_freq', type=int, default=5,
                       help='Frequency of embedding updates (every N epochs)')
    
    # Device and logging
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training')
    parser.add_argument('--save_interval', type=int, default=50,
                       help='Save model every N epochs')
    parser.add_argument('--plot_interval', type=int, default=20,
                       help='Plot embeddings every N epochs')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"result/trajectory_embed_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Create data loaders (split by trajectory)
    train_loader, val_loader, full_dataset = create_data_loaders_by_trajectory(
        data_dir=args.data_dir,
        image_type=args.image_type,
        image_scale=args.image_scale,
        output_type=args.output_type,
        action_chunk=args.action_chunk,
        action_stride=args.action_stride,
        batch_size=args.batch_size,
        train_split=args.train_split,
        window_size=args.window_size,
        num_workers=args.num_workers,
        data_aug=getattr(args, 'data_aug', 1)
    )
    
    # Get sample batch to determine dimensions
    sample_batch = next(iter(train_loader))
    sample_indices, sample_frames, sample_qpos, sample_target = sample_batch
    
    image_shape = sample_frames.shape[2:]  # (C, H, W)
    qpos_dim = sample_qpos.shape[-1]
    
    print(f"Image shape: {image_shape}")
    print(f"QPos dimension: {qpos_dim}")
    print(f"Target shape: {sample_target.shape}")
    
    # Determine actual number of trajectories
    all_train_indices = get_unique_trajectories(train_loader)
    all_val_indices = get_unique_trajectories(val_loader)
    all_indices = sorted(list(set(all_train_indices + all_val_indices)))
    
    actual_num_trajectories = max(all_indices) + 1
    print(f"Actual number of trajectories needed: {actual_num_trajectories}")
    print(f"Using num_trajectories: {max(args.num_trajectories, actual_num_trajectories)}")
    
    # Create model
    model = RobotGraspModelWithTrajectoryEmbed(
        image_shape=image_shape,
        embed_dim=args.embed_dim,
        window_size=args.window_size,
        qpos_dim=qpos_dim,
        output_type=args.output_type,
        action_chunk=args.action_chunk,
        num_trajectories=max(args.num_trajectories, actual_num_trajectories)
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizers
    network_params = [p for name, p in model.named_parameters() if 'trajectory_embeddings' not in name]
    embed_params = [model.trajectory_embeddings]
    
    network_optimizer = optim.Adam(network_params, lr=args.learning_rate)
    embed_optimizer = optim.Adam(embed_params, lr=args.embed_learning_rate)
    
    # Loss criterion
    criterion = nn.MSELoss()
    
    # Tensorboard writer
    writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))
    
    # Training loop
    best_val_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(args.num_epochs):
        # Phase 1: Train network (freeze embeddings)
        train_loss = train_network_epoch(model, train_loader, criterion, network_optimizer, device)
        
        # Phase 2: Update embeddings (every alternating_freq epochs)
        embed_loss = 0.0
        if (epoch + 1) % args.alternating_freq == 0:
            embed_loss, trajectory_losses = update_trajectory_embeddings(
                model, val_loader, embed_optimizer, device, args
            )
        
        # Validation
        val_loss, means, stds, indices = validate(model, val_loader, criterion, device)
        
        # Logging
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print(f"  Train Loss: {train_loss:.6f}")
        if embed_loss > 0:
            print(f"  Embed Loss: {embed_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        
        # Tensorboard logging
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        if embed_loss > 0:
            writer.add_scalar('Loss/Embedding', embed_loss, epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'network_optimizer_state_dict': network_optimizer.state_dict(),
                'embed_optimizer_state_dict': embed_optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': vars(args)
            }, os.path.join(output_dir, 'best_model.pth'))
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'network_optimizer_state_dict': network_optimizer.state_dict(),
                'embed_optimizer_state_dict': embed_optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': vars(args)
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Plot embeddings periodically
        if (epoch + 1) % args.plot_interval == 0:
            plot_trajectory_embeddings(means, stds, indices, epoch+1, output_dir)
    
    writer.close()
    print(f"Training completed! Results saved to {output_dir}")

if __name__ == "__main__":
    main() 