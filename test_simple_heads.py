#!/usr/bin/env python3

import torch
import numpy as np
from model.model_bn1 import RobotGraspModel

def test_model_structure():
    """Test the simplified Best of N model structure"""
    print("Testing Simplified Best of N Model")
    print("=" * 50)
    
    # Model parameters
    image_shape = (3, 180, 320)
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
        num_heads=num_heads
    )
    
    print(f"Model created with {num_heads} heads")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Count parameters for shared vs head-specific parts
    shared_params = 0
    head_params = 0
    
    # Shared parameters (CNN + FC layers)
    for name, param in model.named_parameters():
        if 'output_projections' not in name and 'head_selection' not in name:
            shared_params += param.numel()
    
    # Head-specific parameters
    for head_proj in model.output_projections:
        for param in head_proj.parameters():
            head_params += param.numel()
    
    print(f"Shared parameters: {shared_params:,}")
    print(f"Head-specific parameters: {head_params:,} ({head_params//num_heads:,} per head)")
    print(f"Head parameters ratio: {head_params/shared_params:.2f}")
    
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
    
    # Test computing losses and head selection
    criterion = torch.nn.MSELoss()
    per_sample_losses = []
    for output in outputs:
        sample_wise_loss = torch.mean((output - targets) ** 2, dim=1)  # [batch_size]
        per_sample_losses.append(sample_wise_loss)
    
    losses_tensor = torch.stack(per_sample_losses, dim=0)  # [num_heads, batch_size]
    min_loss_indices = torch.argmin(losses_tensor, dim=0)  # [batch_size]
    
    print(f"  Sample-wise best heads: {min_loss_indices.tolist()}")
    
    # Test head selection tracking
    print(f"\n2. Testing Head Selection Tracking:")
    initial_stats = model.get_head_statistics()
    print(f"  Initial statistics: {initial_stats}")
    
    # Simulate head selections
    for i in range(batch_size):
        best_head = min_loss_indices[i].item()
        model.update_head_selection(best_head)
    
    updated_stats = model.get_head_statistics()
    print(f"  After {batch_size} selections: {updated_stats}")
    
    # Test inference mode
    print(f"\n3. Testing Inference Mode:")
    model.eval()
    
    # Test with specific head
    with torch.no_grad():
        for head_idx in range(num_heads):
            output = model(frames, qpos, head_idx=head_idx, training=False)
            print(f"  Head {head_idx} output shape: {output.shape}")
    
    # Test with automatic head selection
    with torch.no_grad():
        output = model(frames, qpos, head_idx=None, training=False)
        print(f"  Auto head selection output shape: {output.shape}")
        
        valid_heads = model.get_valid_heads()
        print(f"  Valid heads: {valid_heads}")
    
    # Test output differences between heads
    print(f"\n4. Testing Head Output Differences:")
    model.eval()
    with torch.no_grad():
        all_outputs = model(frames, qpos, training=True)
        
        # Compare outputs between heads
        for i in range(num_heads):
            for j in range(i+1, num_heads):
                diff = torch.mean(torch.abs(all_outputs[i] - all_outputs[j])).item()
                print(f"  Mean absolute difference between Head {i} and Head {j}: {diff:.6f}")
    
    print(f"\n✓ All tests passed!")
    
    # Show model structure summary
    print(f"\n" + "="*50)
    print("Model Structure Summary")
    print("="*50)
    print("Shared Components:")
    print("  - CNN backbone (Conv2d layers)")
    print("  - FC1 (image features)")
    print("  - FC2 (qpos features)")
    print(f"  - Total shared params: {shared_params:,}")
    print()
    print("Head-Specific Components:")
    print("  - Each head has its own output_proj:")
    print("    * Linear(embed_dim*2 -> embed_dim)")
    print("    * ReLU + 2x ResBlock")
    print("    * Linear(embed_dim -> output_dim)")
    print("    * Tanh activation")
    print(f"  - Params per head: {head_params//num_heads:,}")
    print(f"  - Total head params: {head_params:,}")
    print()
    print(f"Head/Shared ratio: {head_params/shared_params:.2f} (should be > 0.1 for good diversity)")

if __name__ == "__main__":
    test_model_structure() 
 
 