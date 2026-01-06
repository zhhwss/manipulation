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
from collections import defaultdict

from data_loader import create_data_loaders
from model.model_trajectory_embed import RobotGraspModelWithTrajectoryEmbed

def get_unique_trajectories(data_loader):
    """Get all unique trajectory indices from the data loader"""
    unique_indices = set()
    for indices, _, _, _ in data_loader:
        unique_indices.update(indices.tolist())
    return sorted(list(unique_indices))

def train_network_epoch(model, train_loader, criterion, optimizer, device):
    """Train network parameters (freeze trajectory embeddings)"""
    model.train()
    
    # Freeze trajectory embeddings
    model.trajectory_embeddings.requires_grad = False
    
    total_loss = 0
    num_batches = 0
    
    for indices, frames, qpos, targets in tqdm(train_loader, desc="Training Network"):
        indices = indices.to(device)
        frames = frames.to(device)
        qpos = qpos.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        predictions = model(frames, qpos, indices)
        loss = criterion(predictions, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    # Unfreeze trajectory embeddings
    model.trajectory_embeddings.requires_grad = True
    
    return total_loss / num_batches

def update_trajectory_embeddings(model, val_loader, embed_optimizer, device, args):
    """Update trajectory embeddings using validation data"""
    model.eval()
    
    # Freeze network parameters, only train trajectory embeddings
    for param in model.parameters():
        param.requires_grad = False
    model.trajectory_embeddings.requires_grad = True
    
    total_loss = 0
    num_batches = 0
    
    # Collect trajectory-specific losses
    trajectory_losses = defaultdict(list)
    trajectory_predictions = defaultdict(list)
    trajectory_targets = defaultdict(list)
    
    for indices, frames, qpos, targets in tqdm(val_loader, desc="Updating Embeddings"):
        indices = indices.to(device)
        frames = frames.to(device)
        qpos = qpos.to(device)
        targets = targets.to(device)
        
        embed_optimizer.zero_grad()
        
        predictions = model(frames, qpos, indices)
        
        # Compute individual losses for each sample
        individual_losses = nn.functional.mse_loss(predictions, targets, reduction='none').mean(dim=1)
        
        # Group by trajectory index
        for i, idx in enumerate(indices):
            traj_idx = idx.item()
            trajectory_losses[traj_idx].append(individual_losses[i].item())
            trajectory_predictions[traj_idx].append(predictions[i].detach().cpu())
            trajectory_targets[traj_idx].append(targets[i].detach().cpu())
        
        # Compute total loss for backprop
        loss = individual_losses.mean()
        loss.backward()
        embed_optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    return total_loss / num_batches, trajectory_losses

def validate(model, val_loader, criterion, device):
    """Standard validation"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for indices, frames, qpos, targets in val_loader:
            indices = indices.to(device)
            frames = frames.to(device)
            qpos = qpos.to(device)
            targets = targets.to(device)
            
            predictions = model(frames, qpos, indices)
            loss = criterion(predictions, targets)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def plot_trajectory_embeddings(model, unique_trajectories, epoch, output_dir):
    """Plot trajectory embeddings"""
    model.eval()
    
    with torch.no_grad():
        # Get all trajectory embeddings
        traj_indices = torch.tensor(unique_trajectories, device=next(model.parameters()).device)
        means, stds = model.get_trajectory_embeddings_info(traj_indices)
        
        means = means.cpu().numpy()
        stds = stds.cpu().numpy()
        
        plt.figure(figsize=(12, 10))
        
        # Plot means
        plt.scatter(means[:, 0], means[:, 1], alpha=0.7, s=30, c=unique_trajectories, cmap='tab10')
        
        # Plot uncertainty ellipses for a subset
        for i in range(min(20, len(means))):
            ellipse = plt.matplotlib.patches.Ellipse(
                (means[i, 0], means[i, 1]), 
                width=2*stds[i, 0], 
                height=2*stds[i, 1],
                alpha=0.2
            )
            plt.gca().add_patch(ellipse)
        
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.xlabel('Embedding Dim 1')
        plt.ylabel('Embedding Dim 2')
        plt.title(f'Trajectory Embeddings - Epoch {epoch}')
        plt.colorbar(label='Trajectory Index')
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(output_dir, f'trajectory_embeddings_epoch_{epoch}.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Trajectory Embedding with Alternating Training')
    
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
    parser.add_argument('--num_trajectories', type=int, default=1000,
                       help='Maximum number of trajectories')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate for network')
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

def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join('result', f"trajectory_embed_alternating_val_{timestamp}")
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
    
    # Get dimensions and unique trajectories
    sample_indices, sample_frames, sample_qpos, sample_targets = next(iter(train_loader))
    qpos_dim = sample_qpos.shape[-1]
    action_chunk = sample_targets.shape[-1]
    
    # Get all unique trajectory indices
    unique_trajectories = get_unique_trajectories(train_loader) + get_unique_trajectories(val_loader)
    unique_trajectories = sorted(list(set(unique_trajectories)))
    max_traj_idx = max(unique_trajectories)
    
    print(f"QPOS dimension: {qpos_dim}")
    print(f"Action chunk: {action_chunk}")
    print(f"Number of unique trajectories: {len(unique_trajectories)}")
    print(f"Max trajectory index: {max_traj_idx}")
    
    # Adjust num_trajectories to accommodate all trajectories
    args.num_trajectories = max(args.num_trajectories, max_traj_idx + 1)
    print(f"Using num_trajectories: {args.num_trajectories}")
    
    # Create model
    model = RobotGraspModelWithTrajectoryEmbed(
        image_shape=image_shape,
        embed_dim=args.embed_dim,
        window_size=args.window_size,
        qpos_dim=qpos_dim,
        output_type=args.output_type,
        action_chunk=args.action_chunk,
        num_trajectories=args.num_trajectories
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trajectory embedding parameters: {model.trajectory_embeddings.numel():,}")
    
    # Setup optimizers
    # Network optimizer (excludes trajectory embeddings)
    network_params = [p for name, p in model.named_parameters() if 'trajectory_embeddings' not in name]
    network_optimizer = optim.Adam(network_params, lr=args.learning_rate)
    
    # Embedding optimizer (only trajectory embeddings)
    embed_optimizer = optim.Adam([model.trajectory_embeddings], lr=args.embed_learning_rate)
    
    criterion = nn.MSELoss()
    
    # Training loop
    print("Starting alternating training...")
    best_val_loss = float('inf')
    
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
        val_loss = validate(model, val_loader, criterion, device)
        
        # Log metrics
        writer.add_scalar('Loss/Train_Network', train_loss, epoch)
        writer.add_scalar('Loss/Embedding_Update', embed_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Learning_Rate/Network', network_optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Learning_Rate/Embedding', embed_optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Epoch {epoch+1}/{args.num_epochs}:")
        print(f"  Train Loss: {train_loss:.6f}")
        if embed_loss > 0:
            print(f"  Embed Loss: {embed_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        
        # Plot trajectory embeddings
        if (epoch + 1) % args.plot_interval == 0:
            plot_trajectory_embeddings(model, unique_trajectories, epoch + 1, output_dir)
        
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
                'unique_trajectories': unique_trajectories,
                'args': args
            }, os.path.join(output_dir, 'best_model.pth'))
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'network_optimizer_state_dict': network_optimizer.state_dict(),
                'embed_optimizer_state_dict': embed_optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'unique_trajectories': unique_trajectories,
                'args': args
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Final embedding plot
    plot_trajectory_embeddings(model, unique_trajectories, args.num_epochs, output_dir)
    
    writer.close()
    print(f"Training completed. Results saved to {output_dir}")

if __name__ == '__main__':
    main() 