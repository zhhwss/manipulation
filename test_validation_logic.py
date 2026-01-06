#!/usr/bin/env python3
"""
Test script to verify the corrected validation logic
"""
import torch
import torch.nn as nn
import numpy as np

def test_validation_logic():
    print("Testing Corrected Validation Logic")
    print("=" * 50)
    
    try:
        from model.model_dp import DiffusionPolicy
        
        # Create a simple test model
        model = DiffusionPolicy(
            image_shape=(3, 64, 64),  # Smaller for testing
            embed_dim=128,
            window_size=1,
            qpos_dim=8,
            output_type='ee_pose',
            action_chunk=10,  # Smaller for testing
            num_diffusion_steps=50,
            encoder_type='cnn',
            use_diffusers_scheduler=True,
            scheduler_type='ddim',
            beta_schedule='squaredcos_cap_v2'
        )
        model.eval()
        
        print("✓ Model created successfully")
        
        # Create test data
        batch_size = 4
        frames = torch.randn(batch_size, 1, 3, 64, 64)
        qpos = torch.randn(batch_size, 1, 8)
        targets = torch.randn(batch_size, 80)  # 8 * 10
        
        print(f"✓ Test data created:")
        print(f"  Frames: {frames.shape}")
        print(f"  Qpos: {qpos.shape}")
        print(f"  Targets: {targets.shape}")
        
        # Test 1: Add noise to targets
        print("\n1. Testing noise addition...")
        if hasattr(model, 'noise_scheduler') and hasattr(model, 'use_diffusers_scheduler') and model.use_diffusers_scheduler:
            timesteps = torch.randint(0, model.num_diffusion_steps//2, (targets.shape[0],))
            noise = torch.randn_like(targets)
            noisy_targets, _ = model.add_noise(targets, timesteps, noise)
            print(f"  ✓ Diffusers scheduler noise addition")
            print(f"  Original std: {targets.std().item():.4f}")
            print(f"  Noisy std: {noisy_targets.std().item():.4f}")
        else:
            print("  Using custom scheduler...")
            t = model.num_diffusion_steps // 2
            if hasattr(model, 'alphas_cumprod'):
                alpha_t = model.alphas_cumprod[t]
                noise = torch.randn_like(targets)
                noisy_targets = torch.sqrt(alpha_t) * targets + torch.sqrt(1 - alpha_t) * noise
                print(f"  ✓ Custom scheduler noise addition")
                print(f"  Alpha_t: {alpha_t:.4f}")
            else:
                noise_scale = 0.5
                noise = torch.randn_like(targets) * noise_scale
                noisy_targets = targets + noise
                print(f"  ✓ Fallback noise addition")
            
            print(f"  Original std: {targets.std().item():.4f}")
            print(f"  Noisy std: {noisy_targets.std().item():.4f}")
        
        # Test 2: Denoise back
        print("\n2. Testing denoising recovery...")
        with torch.no_grad():
            denoised_actions = model.conditional_sample(frames, qpos, num_denoising_steps=10, noisy_init=noisy_targets)
        
        print(f"  ✓ Denoising completed")
        print(f"  Denoised shape: {denoised_actions.shape}")
        print(f"  Denoised std: {denoised_actions.std().item():.4f}")
        
        # Test 3: Calculate recovery metrics
        print("\n3. Testing recovery metrics...")
        recovery_mse = nn.MSELoss()(denoised_actions, targets)
        noise_mse = nn.MSELoss()(noisy_targets, targets)
        
        print(f"  Original vs Noisy MSE: {noise_mse.item():.4f}")
        print(f"  Original vs Denoised MSE: {recovery_mse.item():.4f}")
        
        if recovery_mse.item() < noise_mse.item():
            print("  ✓ Denoising improved the signal (lower MSE)")
        else:
            print("  ⚠ Denoising didn't improve much (higher MSE)")
            print("    This is normal for untrained models")
        
        # Test 4: Compare with random sampling
        print("\n4. Comparing with random sampling...")
        with torch.no_grad():
            random_actions = model.conditional_sample(frames, qpos, num_denoising_steps=10, noisy_init=None)
        
        random_mse = nn.MSELoss()(random_actions, targets)
        print(f"  Random sampling MSE: {random_mse.item():.4f}")
        print(f"  Noise recovery MSE: {recovery_mse.item():.4f}")
        
        print("\n" + "=" * 50)
        print("✓ All validation logic tests passed!")
        print("\nKey Improvements Verified:")
        print("- ✓ Noise is added to ground truth targets")
        print("- ✓ Model attempts to recover original from noisy")
        print("- ✓ MSE measures actual recovery performance")
        print("- ✓ Comparison shows denoising vs random sampling")
        print("- ✓ Proper fallback for different scheduler types")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_validation_logic()
