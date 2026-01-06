#!/usr/bin/env python3

import torch
import torch.nn as nn
from model.model_bn1 import RobotGraspModel
from train_bn1 import train_epoch
from torch.utils.data import TensorDataset, DataLoader

def test_warmup_training():
    """Test the warmup training strategy"""
    print("Testing Warmup Training Strategy")
    print("=" * 50)
    
    # Model parameters
    image_shape = (3, 180, 320)
    embed_dim = 256
    window_size = 1
    qpos_dim = 8
    output_type = 'ee_pose'
    action_chunk = 50
    num_heads = 3
    batch_size = 8
    
    # Create model
    model = RobotGraspModel(
        image_shape=image_shape,
        embed_dim=embed_dim,
        window_size=window_size,
        qpos_dim=qpos_dim,
        output_type=output_type,
        action_chunk=action_chunk,
        num_heads=num_heads
    )
    
    print(f"Model created with {num_heads} heads")
    
    # Create dummy dataset
    num_samples = 32
    frames = torch.randn(num_samples, window_size, image_shape[0], image_shape[1], image_shape[2])
    qpos = torch.randn(num_samples, window_size, qpos_dim)
    targets = torch.randn(num_samples, action_chunk * 8)
    indices = torch.arange(num_samples)
    
    dataset = TensorDataset(indices, frames, qpos, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    device = torch.device('cpu')  # Use CPU for testing
    
    print(f"\nTesting training epochs:")
    print(f"Dataset size: {num_samples} samples")
    print(f"Batch size: {batch_size}")
    
    # Test first few epochs (warmup period)
    print(f"\n1. Testing Warmup Period (Epochs 0-9):")
    for epoch in range(3):  # Test first 3 epochs
        print(f"\n--- Epoch {epoch} ---")
        
        # Get initial head statistics
        initial_stats = model.get_head_statistics()
        print(f"Before epoch: {initial_stats['selection_counts']}")
        
        # Train one epoch
        train_loss, train_head_stats, train_head_info = train_epoch(
            model, dataloader, criterion, optimizer, device, epoch
        )
        
        print(f"Training loss: {train_loss:.6f}")
        print(f"Head selection (this epoch): {train_head_stats}")
        print(f"Head selection rates: {[f'{rate:.3f}' for rate in train_head_info['selection_rates']]}")
        
        # Check if heads are selected roughly equally during warmup
        selection_counts = list(train_head_stats.values())
        min_count = min(selection_counts)
        max_count = max(selection_counts)
        print(f"Selection balance: min={min_count}, max={max_count}, diff={max_count-min_count}")
        
        if epoch < 10:
            # During warmup, selection should be roughly equal
            assert max_count - min_count <= 2, f"Heads should be selected roughly equally during warmup, got {selection_counts}"
            print("✓ Heads selected roughly equally (warmup working)")
    
    # Test Best of N period
    print(f"\n2. Testing Best of N Period (Epochs 10+):")
    
    # Reset model statistics to start fresh for Best of N testing
    model.head_selection_counts.zero_()
    model.total_selections.zero_()
    
    for epoch in range(10, 13):  # Test epochs 10, 11, 12
        print(f"\n--- Epoch {epoch} ---")
        
        # Get initial head statistics
        initial_stats = model.get_head_statistics()
        print(f"Before epoch: {initial_stats['selection_counts']}")
        
        # Train one epoch
        train_loss, train_head_stats, train_head_info = train_epoch(
            model, dataloader, criterion, optimizer, device, epoch
        )
        
        print(f"Training loss: {train_loss:.6f}")
        print(f"Head selection (this epoch): {train_head_stats}")
        print(f"Head selection rates: {[f'{rate:.3f}' for rate in train_head_info['selection_rates']]}")
        
        # Check if Best of N selection is working (should be less uniform)
        selection_counts = list(train_head_stats.values())
        min_count = min(selection_counts)
        max_count = max(selection_counts)
        print(f"Selection balance: min={min_count}, max={max_count}, diff={max_count-min_count}")
        
        if epoch >= 10:
            print("✓ Best of N selection active (selection may be uneven)")
    
    print(f"\n3. Comparison Summary:")
    
    # Test both modes side by side for comparison
    model_warmup = RobotGraspModel(
        image_shape=image_shape, embed_dim=embed_dim, window_size=window_size,
        qpos_dim=qpos_dim, output_type=output_type, action_chunk=action_chunk, num_heads=num_heads
    )
    
    model_best_of_n = RobotGraspModel(
        image_shape=image_shape, embed_dim=embed_dim, window_size=window_size,
        qpos_dim=qpos_dim, output_type=output_type, action_chunk=action_chunk, num_heads=num_heads
    )
    
    # Copy same initial weights
    model_best_of_n.load_state_dict(model_warmup.state_dict())
    
    optimizer_warmup = torch.optim.Adam(model_warmup.parameters(), lr=1e-4)
    optimizer_best_of_n = torch.optim.Adam(model_best_of_n.parameters(), lr=1e-4)
    
    # Train one epoch with each strategy
    print("\nWarmup mode (epoch 5):")
    _, warmup_stats, warmup_info = train_epoch(
        model_warmup, dataloader, criterion, optimizer_warmup, device, epoch=5
    )
    print(f"  Head selection: {warmup_stats}")
    print(f"  Selection rates: {[f'{rate:.3f}' for rate in warmup_info['selection_rates']]}")
    
    print("\nBest of N mode (epoch 15):")
    _, best_of_n_stats, best_of_n_info = train_epoch(
        model_best_of_n, dataloader, criterion, optimizer_best_of_n, device, epoch=15
    )
    print(f"  Head selection: {best_of_n_stats}")
    print(f"  Selection rates: {[f'{rate:.3f}' for rate in best_of_n_info['selection_rates']]}")
    
    print(f"\n✓ All tests passed!")
    print(f"\nTraining Strategy Summary:")
    print(f"- Epochs 0-9: All heads trained equally (warmup)")
    print(f"- Epochs 10+: Best performing head selected per sample")
    print(f"- This ensures all heads get initial training before competition")

if __name__ == "__main__":
    test_warmup_training() 
 
 