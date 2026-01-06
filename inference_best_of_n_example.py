#!/usr/bin/env python3

import torch
import numpy as np
import os
import cv2
import json
from model.model_bn import RobotGraspModel
from data_loader import RobotGraspDataset

def load_best_of_n_model(checkpoint_path, device='cuda'):
    """
    Load a trained Best of N model from checkpoint
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        device: Device to load the model on
    
    Returns:
        model: Loaded model
        checkpoint: Full checkpoint data (contains args, statistics, etc.)
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint['args']
    
    print(f"Checkpoint info:")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Train Loss: {checkpoint['train_loss']:.6f}")
    print(f"  Val Loss: {checkpoint['val_loss']:.6f}")
    
    # Determine image shape from args or use default
    if hasattr(args, 'image_shape'):
        image_shape = args.image_shape
    else:
        # Default image shape based on image type
        if args.image_type in ['color', 'color3']:
            image_shape = (3, 180, 320)  # RGB
        elif args.image_type in ['pointcloud', 'pointcloud3']:
            image_shape = (3, 180, 320)  # XYZ pointcloud
        else:  # depth_mask
            image_shape = (1, 180, 320)  # Grayscale
        
        # Adjust for third_view and color_view
        if args.third_view:
            image_shape = (2 * image_shape[0], *image_shape[1:])
        if args.color_view:
            image_shape = (image_shape[0] + 1, *image_shape[1:])
    
    # Determine qpos_dim from data
    qpos_dim = 8  # Default for ee_pose
    
    # Create model with same architecture
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
    
    # Load model state dict - this automatically loads head selection statistics!
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Number of heads: {model.num_heads}")
    
    # Display head selection statistics that were automatically loaded
    head_stats = model.get_head_statistics()
    print(f"\nHead Selection Statistics (loaded from checkpoint - based on validation data):")
    print(f"  Total validation samples: {head_stats['total_selections']}")
    print(f"  Selection counts: {head_stats['selection_counts']}")
    print(f"  Selection rates: {[f'{rate:.3f}' for rate in head_stats['selection_rates']]}")
    print(f"  Valid heads (>1/6 selection rate): {head_stats['valid_heads']}")
    
    # Also show training vs validation statistics if available
    if 'train_head_statistics' in checkpoint and 'val_head_statistics' in checkpoint:
        train_stats = checkpoint['train_head_statistics']
        val_stats = checkpoint['val_head_statistics']
        print(f"\nComparison (Training vs Validation):")
        print(f"  Train rates: {[f'{rate:.3f}' for rate in train_stats['selection_rates']]}")
        print(f"  Val rates:   {[f'{rate:.3f}' for rate in val_stats['selection_rates']]}")
        print(f"  Train valid heads: {train_stats['valid_heads']}")
        print(f"  Val valid heads:   {val_stats['valid_heads']}")
        print(f"  Note: Model uses validation-based statistics for inference")
    
    return model, checkpoint

def preprocess_frame(frame, image_type, image_scale=1.0):
    """
    Preprocess a single frame according to image type
    """
    # Process frame based on type
    if image_type == 'depth_mask' or image_type == 'depth_mask3':
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif image_type == "color" or image_type == "color3":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize if needed
    if image_scale != 1.0:
        new_width = int(frame.shape[1] * image_scale)
        new_height = int(frame.shape[0] * image_scale)
        frame = cv2.resize(frame, (new_width, new_height))
    
    # Normalize to [0, 1]
    frame = frame.astype(np.float32) / 255.0
    
    # Add channel dimension if grayscale
    if len(frame.shape) == 2:
        frame = frame[np.newaxis, :, :]  # (1, H, W)
    else:
        frame = np.transpose(frame, (2, 0, 1))  # (3, H, W)
    
    return frame

def run_inference_on_real_data(model, checkpoint, data_dir, device='cuda'):
    """
    Run inference on real data from the dataset
    """
    print(f"\n{'='*60}")
    print("Running Inference on Real Data")
    print(f"{'='*60}")
    
    args = checkpoint['args']
    
    # Find some real data files
    import glob
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return
    
    # Load first few samples
    for i, json_file in enumerate(json_files[:3]):  # Test on first 3 files
        print(f"\nProcessing file {i+1}: {os.path.basename(json_file)}")
        
        try:
            # Load JSON data
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Load corresponding video
            video_file = json_file.replace('.json', f'_{args.image_type}.mp4')
            if not os.path.exists(video_file):
                print(f"  Video file not found: {video_file}")
                continue
            
            cap = cv2.VideoCapture(video_file)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                print(f"  Could not read frame from {video_file}")
                continue
            
            # Preprocess frame
            processed_frame = preprocess_frame(frame, args.image_type, args.image_scale)
            
            # Get qpos (robot state)
            qpos = data['qpos'][:8]  # Take first 8 dimensions for ee_pose
            
            # Convert to tensors and add batch dimension
            frames_tensor = torch.from_numpy(processed_frame).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, C, H, W)
            qpos_tensor = torch.from_numpy(np.array(qpos)).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 8)
            
            print(f"  Input shapes: frames {frames_tensor.shape}, qpos {qpos_tensor.shape}")
            
            # Run inference with different methods
            with torch.no_grad():
                # Method 1: Automatic head selection (recommended)
                output_auto = model(frames_tensor, qpos_tensor, head_idx=None, training=False)
                print(f"  Auto head selection - Output shape: {output_auto.shape}")
                print(f"    First 5 values: {output_auto[0, :5].cpu().numpy()}")
                
                # Method 2: Try all valid heads
                valid_heads = model.get_valid_heads()
                print(f"  Valid heads: {valid_heads}")
                
                for head_idx in valid_heads:
                    output_specific = model(frames_tensor, qpos_tensor, head_idx=head_idx, training=False)
                    print(f"  Head {head_idx} - Output mean: {output_specific.mean().item():.4f}, std: {output_specific.std().item():.4f}")
                
                # Method 3: Get all head outputs for comparison
                all_outputs = model(frames_tensor, qpos_tensor, training=True)
                print(f"  All heads comparison:")
                for j, output in enumerate(all_outputs):
                    print(f"    Head {j}: mean={output.mean().item():.4f}, std={output.std().item():.4f}")
        
        except Exception as e:
            print(f"  Error processing {json_file}: {e}")

def run_batch_inference_example(model, device='cuda'):
    """
    Example of batch inference for multiple samples
    """
    print(f"\n{'='*60}")
    print("Batch Inference Example")
    print(f"{'='*60}")
    
    batch_size = 4
    # Create dummy batch data (replace with your real data)
    frames = torch.randn(batch_size, 1, 3, 180, 320).to(device)  # Example shape
    qpos = torch.randn(batch_size, 1, 8).to(device)
    
    print(f"Batch size: {batch_size}")
    print(f"Input shapes: frames {frames.shape}, qpos {qpos.shape}")
    
    with torch.no_grad():
        # Batch inference
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        outputs = model(frames, qpos, head_idx=None, training=False)
        end_time.record()
        
        torch.cuda.synchronize()
        inference_time = start_time.elapsed_time(end_time)
        
        print(f"Batch inference completed:")
        print(f"  Output shape: {outputs.shape}")
        print(f"  Inference time: {inference_time:.2f} ms")
        print(f"  Time per sample: {inference_time/batch_size:.2f} ms")
        
        # Show some statistics
        for i in range(batch_size):
            sample_output = outputs[i]
            print(f"  Sample {i}: mean={sample_output.mean().item():.4f}, std={sample_output.std().item():.4f}")

def main():
    print("Best of N Model Inference Example")
    print("=" * 60)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Path to your trained model (update this path)
    checkpoint_path = "result/best_of_n_20250101_120000/best_model.pth"  # Update this path
    data_dir = "data/20250730_shui"  # Update this path
    
    if os.path.exists(checkpoint_path):
        print(f"\n{'='*60}")
        print("Loading Trained Model")
        print(f"{'='*60}")
        
        # Load model
        model, checkpoint = load_best_of_n_model(checkpoint_path, device)
        
        # Run different inference examples
        run_batch_inference_example(model, device)
        
        # Run inference on real data if data directory exists
        if os.path.exists(data_dir):
            run_inference_on_real_data(model, checkpoint, data_dir, device)
        else:
            print(f"\nData directory not found: {data_dir}")
            print("Update data_dir in main() to run inference on real data")
        
    else:
        print(f"\n{'='*60}")
        print("Model Training Required")
        print(f"{'='*60}")
        print(f"Checkpoint not found: {checkpoint_path}")
        print("\nTo use this inference example:")
        print("1. Train a Best of N model using train_bn.py:")
        print("   python train_bn.py --data_dir your_data_dir --num_heads 3")
        print("2. Update checkpoint_path in this script to point to your best_model.pth")
        print("3. Update data_dir to point to your data directory")
        print("4. Run this script again")
        
        print(f"\nExample training command:")
        print(f"python train_bn.py \\")
        print(f"  --data_dir data/20250730_shui \\")
        print(f"  --image_type pointcloud \\")
        print(f"  --output_type ee_pose \\")
        print(f"  --action_chunk 50 \\")
        print(f"  --num_heads 3 \\")
        print(f"  --batch_size 8 \\")
        print(f"  --num_epochs 100 \\")
        print(f"  --device cuda:0")

if __name__ == "__main__":
    main() 
 
 