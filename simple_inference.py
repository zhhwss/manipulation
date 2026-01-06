#!/usr/bin/env python3
"""
Simple inference script for Best of N model
Usage: python simple_inference.py --checkpoint path/to/best_model.pth --data_dir path/to/data
"""

import torch
import numpy as np
import argparse
import os
from model.model_bn import RobotGraspModel

def load_model(checkpoint_path, device='cuda'):
    """Load trained Best of N model"""
    print(f"Loading model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint['args']
    
    # Determine image shape
    if args.image_type in ['color', 'color3']:
        image_shape = (3, 180, 320)
    elif args.image_type in ['pointcloud', 'pointcloud3']:
        image_shape = (3, 180, 320)
    else:  # depth_mask
        image_shape = (1, 180, 320)
    
    if args.third_view:
        image_shape = (2 * image_shape[0], *image_shape[1:])
    if args.color_view:
        image_shape = (image_shape[0] + 1, *image_shape[1:])
    
    # Create and load model
    model = RobotGraspModel(
        image_shape=image_shape,
        embed_dim=args.embed_dim,
        window_size=args.window_size,
        qpos_dim=8,
        output_type=args.output_type,
        action_chunk=args.action_chunk,
        unet_layers=2,
        num_heads=args.num_heads
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Show head statistics
    head_stats = model.get_head_statistics()
    print(f"Model loaded! Valid heads: {head_stats['valid_heads']}")
    print(f"Head selection rates: {[f'{r:.3f}' for r in head_stats['selection_rates']]}")
    
    return model, args

def predict_actions(model, frames, qpos, device='cuda'):
    """
    Predict actions using the Best of N model
    
    Args:
        model: Loaded Best of N model
        frames: Input frames, shape (batch_size, window_size, channels, height, width)
        qpos: Robot states, shape (batch_size, window_size, qpos_dim)
        device: Device to run inference on
    
    Returns:
        predicted_actions: Shape (batch_size, action_chunk * action_dim)
    """
    with torch.no_grad():
        frames_tensor = torch.from_numpy(frames).float().to(device)
        qpos_tensor = torch.from_numpy(qpos).float().to(device)
        
        # Use automatic head selection (recommended)
        predictions = model(frames_tensor, qpos_tensor, head_idx=None, training=False)
        
        return predictions.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description='Best of N Model Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to run inference on')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, model_args = load_model(args.checkpoint, device)
    
    # Example inference with dummy data
    print(f"\nRunning example inference...")
    
    # Create example input (replace with your real data)
    batch_size = 2
    window_size = model_args.window_size
    action_chunk = model_args.action_chunk
    
    # Example image data (replace with your preprocessed frames)
    if model_args.image_type in ['color', 'color3']:
        channels = 3
    elif model_args.image_type in ['pointcloud', 'pointcloud3']:
        channels = 3
    else:
        channels = 1
    
    if model_args.third_view:
        channels *= 2
    if model_args.color_view:
        channels += 1
    
    # Example data
    frames = np.random.randn(batch_size, window_size, channels, 180, 320)  # Normalized [0,1]
    qpos = np.random.randn(batch_size, window_size, 8)  # Robot state
    
    print(f"Input shapes: frames {frames.shape}, qpos {qpos.shape}")
    
    # Predict actions
    predicted_actions = predict_actions(model, frames, qpos, device)
    
    print(f"Predicted actions shape: {predicted_actions.shape}")
    print(f"Action chunk size: {action_chunk}")
    print(f"Action dimension: {predicted_actions.shape[1] // action_chunk}")
    
    # Reshape to (batch_size, action_chunk, action_dim)
    action_dim = predicted_actions.shape[1] // action_chunk
    actions_reshaped = predicted_actions.reshape(batch_size, action_chunk, action_dim)
    
    print(f"\nPredicted actions (first sample, first 5 timesteps):")
    for t in range(min(5, action_chunk)):
        print(f"  t={t}: {actions_reshaped[0, t]}")
    
    print(f"\nInference completed successfully!")
    
    # Usage example
    print(f"\n" + "="*60)
    print("Usage in your code:")
    print("="*60)
    print("""
# Load model once
model, model_args = load_model('path/to/best_model.pth')

# For each inference
frames = your_preprocessed_frames  # Shape: (batch, window, channels, H, W)
qpos = your_robot_states          # Shape: (batch, window, qpos_dim)

# Predict actions
actions = predict_actions(model, frames, qpos)

# actions shape: (batch_size, action_chunk * action_dim)
# Reshape if needed: actions.reshape(batch, action_chunk, action_dim)
""")

if __name__ == "__main__":
    main() 
 
 