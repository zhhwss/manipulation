#!/usr/bin/env python3
"""
Test inference speed of trained DiffusionPolicy model
"""
import time
import torch
import numpy as np
import argparse
import os
from model.model_dp import DiffusionPolicy

def load_trained_model(checkpoint_path, device='cuda:0'):
    """
    Load trained model from checkpoint
    """
    print(f"Loading model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get model arguments from checkpoint
    if 'args' in checkpoint:
        args = checkpoint['args']
        print(f"Model configuration:")
        print(f"  Image type: {args.image_type}")
        print(f"  Output type: {args.output_type}")
        print(f"  Action chunk: {args.action_chunk}")
        print(f"  Encoder type: {args.encoder_type}")
        print(f"  Embed dim: {args.embed_dim}")
        print(f"  Diffusion steps: {args.num_diffusion_steps}")
        
        # Determine image shape (simplified)
        if args.image_type == 'pointcloud':
            image_shape = (3, 180, 320)
        elif args.image_type == 'depth_mask':
            image_shape = (1, 180, 320)
        else:
            image_shape = (3, 180, 320)  # default
            
        if hasattr(args, 'third_view') and args.third_view:
            image_shape = (image_shape[0] * 2, *image_shape[1:])
        if hasattr(args, 'color_view') and args.color_view:
            image_shape = (image_shape[0] + 1, *image_shape[1:])
        
        # Parse down_dims
        down_dims = tuple(int(x) for x in args.down_dims.split(','))
        
        # Create model
        model = DiffusionPolicy(
            image_shape=image_shape,
            embed_dim=args.embed_dim,
            window_size=args.window_size,
            qpos_dim=9,  # Assuming standard qpos dimension
            output_type=args.output_type,
            action_chunk=args.action_chunk,
            num_diffusion_steps=args.num_diffusion_steps,
            encoder_type=args.encoder_type,
            use_diffusers_scheduler=args.use_diffusers_scheduler,
            scheduler_type=getattr(args, 'scheduler_type', 'ddim'),
            beta_schedule=getattr(args, 'beta_schedule', 'squaredcos_cap_v2'),
            beta_start=getattr(args, 'beta_start', 0.0001),
            beta_end=getattr(args, 'beta_end', 0.02),
            prediction_type=getattr(args, 'prediction_type', 'sample'),
            down_dims=down_dims,
            kernel_size=getattr(args, 'kernel_size', 5),
            n_groups=getattr(args, 'n_groups', 8)
        )
        
    else:
        # Fallback: create model with default parameters
        print("Warning: No args found in checkpoint, using default parameters")
        image_shape = (3, 180, 320)
        model = DiffusionPolicy(
            image_shape=image_shape,
            embed_dim=256,
            window_size=1,
            qpos_dim=9,
            output_type='ee_pose',
            action_chunk=50,
            num_diffusion_steps=100,
            encoder_type='cnn'
        )
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, image_shape

def test_inference_speed(model, image_shape, device='cuda:0', batch_sizes=[1, 2, 4, 8], 
                        denoising_steps_list=[10, 20, 50, 100], num_warmup=5, num_trials=20):
    """
    Test inference speed with different configurations
    """
    print(f"\n{'='*60}")
    print("INFERENCE SPEED TEST")
    print(f"{'='*60}")
    
    qpos_dim = 9  # Standard qpos dimension
    window_size = 1
    
    results = []
    
    for batch_size in batch_sizes:
        for num_denoising_steps in denoising_steps_list:
            print(f"\nTesting: Batch={batch_size}, Denoising Steps={num_denoising_steps}")
            
            # Create sample data
            frames = torch.randn(batch_size, window_size, *image_shape, device=device)
            qpos = torch.randn(batch_size, window_size, qpos_dim, device=device)
            
            # Warmup
            print(f"  Warming up ({num_warmup} runs)...")
            with torch.no_grad():
                for _ in range(num_warmup):
                    _ = model.forward(frames, qpos, num_denoising_steps=num_denoising_steps)
                    
            # Synchronize GPU
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            
            # Benchmark
            print(f"  Benchmarking ({num_trials} runs)...")
            times = []
            
            with torch.no_grad():
                for i in range(num_trials):
                    start_time = time.time()
                    
                    # Forward pass
                    output = model.forward(frames, qpos, num_denoising_steps=num_denoising_steps)
                    
                    # Synchronize GPU
                    if device.startswith('cuda'):
                        torch.cuda.synchronize()
                    
                    end_time = time.time()
                    elapsed = end_time - start_time
                    times.append(elapsed)
                    
                    if (i + 1) % 5 == 0:
                        print(f"    Progress: {i+1}/{num_trials}")
            
            # Calculate statistics
            times = np.array(times)
            mean_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            
            # Calculate throughput
            samples_per_second = batch_size / mean_time
            
            result = {
                'batch_size': batch_size,
                'denoising_steps': num_denoising_steps,
                'mean_time': mean_time,
                'std_time': std_time,
                'min_time': min_time,
                'max_time': max_time,
                'samples_per_second': samples_per_second,
                'output_shape': output.shape
            }
            results.append(result)
            
            print(f"  Results:")
            print(f"    Mean time: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms")
            print(f"    Min time: {min_time*1000:.2f} ms")
            print(f"    Max time: {max_time*1000:.2f} ms")
            print(f"    Throughput: {samples_per_second:.2f} samples/sec")
            print(f"    Output shape: {output.shape}")
    
    return results

def print_summary(results):
    """
    Print summary of results
    """
    print(f"\n{'='*80}")
    print("INFERENCE SPEED SUMMARY")
    print(f"{'='*80}")
    
    # Table header
    print(f"{'Batch':<8} {'Steps':<8} {'Mean (ms)':<12} {'Std (ms)':<10} {'Min (ms)':<10} {'Max (ms)':<10} {'Samples/sec':<12}")
    print(f"{'-'*8} {'-'*8} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
    
    # Table rows
    for result in results:
        print(f"{result['batch_size']:<8} "
              f"{result['denoising_steps']:<8} "
              f"{result['mean_time']*1000:<12.2f} "
              f"{result['std_time']*1000:<10.2f} "
              f"{result['min_time']*1000:<10.2f} "
              f"{result['max_time']*1000:<10.2f} "
              f"{result['samples_per_second']:<12.2f}")
    
    # Find best configurations
    print(f"\n{'='*60}")
    print("PERFORMANCE ANALYSIS")
    print(f"{'='*60}")
    
    # Fastest single sample
    single_sample_results = [r for r in results if r['batch_size'] == 1]
    if single_sample_results:
        fastest_single = min(single_sample_results, key=lambda x: x['mean_time'])
        print(f"Fastest single sample: {fastest_single['mean_time']*1000:.2f} ms "
              f"(Steps: {fastest_single['denoising_steps']})")
    
    # Highest throughput
    highest_throughput = max(results, key=lambda x: x['samples_per_second'])
    print(f"Highest throughput: {highest_throughput['samples_per_second']:.2f} samples/sec "
          f"(Batch: {highest_throughput['batch_size']}, Steps: {highest_throughput['denoising_steps']})")
    
    # Real-time analysis (assuming 10 Hz control frequency)
    print(f"\nReal-time Analysis (10 Hz control frequency):")
    realtime_budget = 100  # ms
    realtime_configs = [r for r in single_sample_results if r['mean_time']*1000 <= realtime_budget]
    
    if realtime_configs:
        print(f"Configurations suitable for real-time control (<{realtime_budget}ms):")
        for config in sorted(realtime_configs, key=lambda x: x['mean_time']):
            print(f"  Steps: {config['denoising_steps']:2d}, Time: {config['mean_time']*1000:6.2f} ms")
    else:
        print(f"No configurations meet real-time budget of {realtime_budget}ms")

def main():
    parser = argparse.ArgumentParser(description='Test DiffusionPolicy inference speed')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pth file)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use for inference')
    parser.add_argument('--batch_sizes', type=str, default='1,2,4,8',
                       help='Comma-separated list of batch sizes to test')
    parser.add_argument('--denoising_steps', type=str, default='10,20,50,100',
                       help='Comma-separated list of denoising steps to test')
    parser.add_argument('--num_warmup', type=int, default=5,
                       help='Number of warmup runs')
    parser.add_argument('--num_trials', type=int, default=20,
                       help='Number of benchmark trials')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return
    
    # Parse lists
    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    denoising_steps_list = [int(x) for x in args.denoising_steps.split(',')]
    
    # Check device
    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Load model
    try:
        model, image_shape = load_trained_model(args.checkpoint, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Test inference speed
    try:
        results = test_inference_speed(
            model, image_shape, device, 
            batch_sizes, denoising_steps_list,
            args.num_warmup, args.num_trials
        )
        
        # Print summary
        print_summary(results)
        
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
