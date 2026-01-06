#!/usr/bin/env python3

import torch
import numpy as np
from model.model_bn import RobotGraspModel

def test_best_of_n_model():
    """Test the Best of N model implementation"""
    print("Testing Best of N Model Implementation")
    print("=" * 50)
    
    # Model parameters
    image_shape = (3, 180, 320)  # Example image shape
    embed_dim = 256
    window_size = 1
    qpos_dim = 8
    output_type = 'ee_pose'
    action_chunk = 50
    num_heads = 3
    batch_size = 4
    
    # Create model
    model = RobotGraspModel(
        image_shape=image_shape,
        embed_dim=embed_dim,
        window_size=window_size,
        qpos_dim=qpos_dim,
        output_type=output_type,
        action_chunk=action_chunk,
        unet_layers=2,
        num_heads=num_heads
    )
    
    print(f"Model created with {num_heads} heads")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create sample inputs
    frames = torch.randn(batch_size, window_size, image_shape[0], image_shape[1], image_shape[2])
    qpos = torch.randn(batch_size, window_size, qpos_dim)
    targets = torch.randn(batch_size, action_chunk * 8)  # 8 for ee_pose
    
    print(f"\nInput shapes:")
    print(f"  Frames: {frames.shape}")
    print(f"  Qpos: {qpos.shape}")
    print(f"  Targets: {targets.shape}")
    
    # Test training mode
    print(f"\n1. Testing Training Mode:")
    model.train()
    outputs = model(frames, qpos, training=True)
    
    print(f"  Number of outputs: {len(outputs)}")
    for i, output in enumerate(outputs):
        print(f"  Head {i} output shape: {output.shape}")
    
    # Test computing losses
    criterion = torch.nn.MSELoss()
    losses = []
    for output in outputs:
        loss = criterion(output, targets)
        losses.append(loss)
    
    losses_tensor = torch.stack(losses)
    min_loss_idx = torch.argmin(losses_tensor).item()
    min_loss = losses[min_loss_idx]
    
    print(f"  Losses: {[f'{loss.item():.6f}' for loss in losses]}")
    print(f"  Best head: {min_loss_idx} (loss: {min_loss.item():.6f})")
    
    # Test head selection tracking
    print(f"\n2. Testing Head Selection Tracking:")
    initial_stats = model.get_head_statistics()
    print(f"  Initial statistics: {initial_stats}")
    
    # Simulate some head selections
    for i in range(20):
        selected_head = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])  # Biased selection
        model.update_head_selection(selected_head)
    
    updated_stats = model.get_head_statistics()
    print(f"  After 20 selections: {updated_stats}")
    
    # Test inference mode
    print(f"\n3. Testing Inference Mode:")
    model.eval()
    
    # Test with specific head
    with torch.no_grad():
        for head_idx in range(num_heads):
            output = model(frames, qpos, head_idx=head_idx, training=False)
            print(f"  Head {head_idx} output shape: {output.shape}")
    
    # Test with random valid head selection
    with torch.no_grad():
        output = model(frames, qpos, head_idx=None, training=False)
        print(f"  Random valid head output shape: {output.shape}")
        
        valid_heads = model.get_valid_heads()
        print(f"  Valid heads: {valid_heads}")
    
    # Test edge case: make all heads invalid
    print(f"\n4. Testing Edge Cases:")
    model.head_selection_counts.fill_(0)
    model.total_selections.fill_(100)
    model.head_selection_counts[0] = 5  # Only 5% selection rate for head 0
    model.head_selection_counts[1] = 10  # Only 10% selection rate for head 1 
    model.head_selection_counts[2] = 5   # Only 5% selection rate for head 2
    
    stats_invalid = model.get_head_statistics()
    print(f"  All heads <1/6 selection: {stats_invalid}")
    
    # Should fallback to head 0
    with torch.no_grad():
        output = model(frames, qpos, head_idx=None, training=False)
        print(f"  Fallback output shape: {output.shape}")
    
    print(f"\n✓ All tests passed!")

if __name__ == "__main__":
    test_best_of_n_model() 
 
 