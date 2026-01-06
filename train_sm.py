import argparse
import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from data_loader import create_data_loaders, create_data_loaders_by_trajectory
from model.model import RobotGraspModel_v2, RobotGraspModel_v3
from model.model3 import RobotGraspModelLowRank, RobotGraspModelSmooth, RobotGraspModelRNN, RobotGraspModelTransformer, RobotGraspModelGaussian


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
    
    parser.add_argument('--rank', type=int, default=80,
                       help='rank of network')
    parser.add_argument('--model_type', type=str, default='low_rank',
                       choices=['low_rank', 'smooth', 'rnn', 'transformer', 'gaussian'],
                       help='Type of model to use')
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

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for indices, frames, qpos, targets in tqdm(train_loader):
        frames = frames.to(device)
        qpos = qpos.to(device)
        targets = targets.to(device)
        
        # Forward pass
        predictions = model(frames, qpos)
        loss = criterion(predictions, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for indices, frames, qpos, targets in val_loader:
            frames = frames.to(device)
            qpos = qpos.to(device)
            targets = targets.to(device)
            
            predictions = model(frames, qpos)
            loss = criterion(predictions, targets)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join('result', f"{args.model_type}_{timestamp}")
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
    
    # Create model
    if args.model_type == 'low_rank':
        model = RobotGraspModelLowRank(
            image_shape=image_shape,
            embed_dim=args.embed_dim,
            window_size=args.window_size,
            qpos_dim=qpos_dim,
            output_type=args.output_type,
            action_chunk=args.action_chunk,
            rank=args.rank,
        ).to(device)
    elif args.model_type == 'smooth':
        model = RobotGraspModelSmooth(
            image_shape=image_shape,
            embed_dim=args.embed_dim,
            window_size=args.window_size,
            qpos_dim=qpos_dim,
            output_type=args.output_type,
            action_chunk=args.action_chunk,
        ).to(device)
    elif args.model_type == 'rnn':
        model = RobotGraspModelRNN(
            image_shape=image_shape,
            embed_dim=args.embed_dim,
            window_size=args.window_size,
            qpos_dim=qpos_dim,
            output_type=args.output_type,
            action_chunk=args.action_chunk,
        ).to(device)
    elif args.model_type == 'transformer':
        model = RobotGraspModelTransformer(
            image_shape=image_shape,
            embed_dim=args.embed_dim,
            window_size=args.window_size,
            qpos_dim=qpos_dim,
            output_type=args.output_type,
            action_chunk=args.action_chunk,
        ).to(device)
    elif args.model_type == 'gaussian':
        model = RobotGraspModelGaussian(
            image_shape=image_shape,
            embed_dim=args.embed_dim,
            window_size=args.window_size,
            qpos_dim=qpos_dim,
            output_type=args.output_type,
            action_chunk=args.action_chunk,
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Log
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        
        print(f"Epoch {epoch+1}/{args.num_epochs}: "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': args
            }, os.path.join(output_dir, 'best_model.pth'))
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': args
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    writer.close()
    print(f"Training completed. Results saved to {output_dir}")

if __name__ == '__main__':
    main()
