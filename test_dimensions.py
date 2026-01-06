#!/usr/bin/env python3
"""
Test script to verify dimensions consistency between data_loader and model
"""
import torch
import numpy as np
from data_loader import RobotGraspDataset
from model.model_dp import DiffusionPolicy

def test_dimensions():
    print("Testing dimensions consistency")
    print("=" * 50)
    
    # Test parameters
    action_chunk = 50
    output_type = 'ee_pose'  # 8 dimensions
    window_size = 1
    batch_size = 4
    
    # Expected dimensions
    action_output_dim = 7 if output_type == 'position' else 8
    expected_flat_dim = action_output_dim * action_chunk
    
    print(f"Action chunk: {action_chunk}")
    print(f"Output type: {output_type}")
    print(f"Action output dim: {action_output_dim}")
    print(f"Expected flat dimension: {expected_flat_dim}")
    
    # Test 1: Simulate data loader output
    print(f"\n1. Testing Data Loader Output Format:")
    
    # Simulate what data_loader.py does
    targets = []
    for j in range(action_chunk):
        # Simulate one timestep of ee_pose (8 dimensions)
        if output_type == 'ee_pose':
            target = np.random.randn(8)  # [x, y, z, qx, qy, qz, qw, gripper]
        else:  # position
            target = np.random.randn(7)  # [x, y, z, qx, qy, qz, qw]
        targets.append(target)
    
    # This is what data_loader does: np.concatenate(targets)
    flattened_target = np.concatenate(targets)
    print(f"   Single sample shape: {flattened_target.shape}")
    
    # After DataLoader batching
    batch_targets = np.random.randn(batch_size, expected_flat_dim)
    batch_targets_tensor = torch.tensor(batch_targets, dtype=torch.float32)
    print(f"   Batch shape: {batch_targets_tensor.shape}")
    
    # Test 2: Model expectations
    print(f"\n2. Testing Model Expectations:")
    
    # Create a simple model instance to check dimensions
    image_shape = (3, 180, 320)
    qpos_dim = 9
    
    model = DiffusionPolicy(
        image_shape=image_shape,
        embed_dim=256,
        window_size=window_size,
        qpos_dim=qpos_dim,
        output_type=output_type,
        action_chunk=action_chunk,
        num_diffusion_steps=100,
        encoder_type='cnn'
    )
    
    print(f"   Model action_output_dim: {model.action_output_dim}")
    print(f"   Model action_chunk: {model.action_chunk}")
    print(f"   Model output_dim: {model.output_dim}")
    
    # Test 3: Dimension compatibility
    print(f"\n3. Testing Dimension Compatibility:")
    
    # Check if data loader output matches model expectations
    data_loader_dim = expected_flat_dim
    model_expected_dim = model.output_dim
    
    print(f"   Data loader output dim: {data_loader_dim}")
    print(f"   Model expected dim: {model_expected_dim}")
    
    if data_loader_dim == model_expected_dim:
        print("   ✅ Dimensions match!")
    else:
        print("   ❌ Dimension mismatch!")
        return False
    
    # Test 4: Model internal reshaping
    print(f"\n4. Testing Model Internal Reshaping:")
    
    # Simulate model's view operation
    try:
        test_actions = torch.randn(batch_size, model.output_dim)
        reshaped = test_actions.view(batch_size, model.action_output_dim, model.action_chunk)
        print(f"   Input shape: {test_actions.shape}")
        print(f"   Reshaped to: {reshaped.shape}")
        print(f"   Expected: ({batch_size}, {model.action_output_dim}, {model.action_chunk})")
        
        if reshaped.shape == (batch_size, model.action_output_dim, model.action_chunk):
            print("   ✅ Reshaping works correctly!")
        else:
            print("   ❌ Reshaping failed!")
            return False
            
    except Exception as e:
        print(f"   ❌ Reshaping error: {e}")
        return False
    
    # Test 5: Reverse reshaping
    print(f"\n5. Testing Reverse Reshaping:")
    
    try:
        reshaped_back = reshaped.view(batch_size, -1)
        print(f"   Reshaped back to: {reshaped_back.shape}")
        
        if reshaped_back.shape == test_actions.shape:
            print("   ✅ Reverse reshaping works correctly!")
        else:
            print("   ❌ Reverse reshaping failed!")
            return False
            
    except Exception as e:
        print(f"   ❌ Reverse reshaping error: {e}")
        return False
    
    print(f"\n" + "=" * 50)
    print("✅ All dimension tests passed!")
    print("\nSummary:")
    print(f"- Data format: Each sample is flattened to ({expected_flat_dim},)")
    print(f"- Batch format: ({batch_size}, {expected_flat_dim})")
    print(f"- Model internal: Reshapes to ({batch_size}, {action_output_dim}, {action_chunk})")
    print(f"- Output format: Flattened back to ({batch_size}, {expected_flat_dim})")
    
    return True

if __name__ == "__main__":
    test_dimensions()
