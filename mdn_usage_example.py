#!/usr/bin/env python3
"""
MDN (Mixture Density Network) 使用示例

演示如何使用修改后的 RobotGraspModel_v2 进行训练和推理
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from model.model_mdn import RobotGraspModel_v2, mdn_negative_log_likelihood, visualize_trajectory_predictions

def create_sample_data(batch_size=8, num_mixtures=3):
    """创建示例数据用于测试"""
    # 模拟图像和qpos数据
    image_shape = (4, 120, 160)  # (channels, height, width)
    window_size = 10
    qpos_dim = 8
    action_chunk = 50
    
    frames = torch.randn(batch_size, window_size, image_shape[0], image_shape[1], image_shape[2])
    qpos = torch.randn(batch_size, window_size, qpos_dim)
    
    # 模拟多模态目标数据（混合了不同的模式）
    action_output_dim = 8  # ee_pose
    output_dim = action_output_dim * action_chunk
    
    # 创建多模态的目标数据
    targets = []
    for i in range(batch_size):
        # 随机选择一个模式
        mode = np.random.randint(0, num_mixtures)
        if mode == 0:
            # 模式1: 高动作值
            target = torch.randn(output_dim) * 0.5 + 2.0
        elif mode == 1:
            # 模式2: 低动作值
            target = torch.randn(output_dim) * 0.3 - 1.0
        else:
            # 模式3: 中等动作值
            target = torch.randn(output_dim) * 0.2
        targets.append(target)
    
    targets = torch.stack(targets)
    
    return frames, qpos, targets

def demonstrate_mdn_training():
    """演示 MDN 训练过程"""
    print("=== MDN Training Demonstration ===")
    
    # 模型参数
    image_shape = (4, 120, 160)
    embed_dim = 256
    window_size = 10
    qpos_dim = 8
    output_type = 'ee_pose'
    action_chunk = 50
    num_mixtures = 5
    
    # 创建模型
    model = RobotGraspModel_v2(
        image_shape=image_shape,
        embed_dim=embed_dim,
        window_size=window_size,
        qpos_dim=qpos_dim,
        output_type=output_type,
        action_chunk=action_chunk,
        num_mixtures=num_mixtures
    )
    
    print(f"Created MDN model with {num_mixtures} mixture components")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 训练循环
    num_epochs = 5
    batch_size = 8
    
    for epoch in range(num_epochs):
        model.train()
        
        # 获取训练数据
        frames, qpos, targets = create_sample_data(batch_size, num_mixtures)
        
        # 前向传播（训练模式）
        mdn_params = model(frames, qpos, mode='mdn_params')
        
        # 计算MDN损失
        loss = mdn_negative_log_likelihood(mdn_params, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
        
        # 显示混合组件权重的分布
        with torch.no_grad():
            pi = torch.exp(mdn_params['log_pi']).mean(dim=0)  # 平均权重
            weights_str = ", ".join([f"{w:.3f}" for w in pi])
            print(f"  Average mixture weights: [{weights_str}]")
    
    return model

def demonstrate_mdn_inference():
    """演示 MDN 推理过程"""
    print("\n=== MDN Inference Demonstration ===")
    
    # 使用训练好的模型
    model = demonstrate_mdn_training()
    model.eval()
    
    # 创建测试数据
    frames, qpos, true_targets = create_sample_data(batch_size=4)
    
    with torch.no_grad():
        print("\n1. Getting MDN parameters:")
        mdn_params = model(frames, qpos, mode='mdn_params')
        
        print(f"   - log_pi shape: {mdn_params['log_pi'].shape}")
        print(f"   - mu shape: {mdn_params['mu'].shape}")
        print(f"   - sigma shape: {mdn_params['sigma'].shape}")
        
        print("\n2. Sampling from MDN:")
        sampled_output = model(frames, qpos, mode='sample')
        print(f"   - Sampled output shape: {sampled_output.shape}")
        
        print("\n3. Getting maximum probability component:")
        max_prob_output = model(frames, qpos, mode='max_prob')
        print(f"   - Max prob output shape: {max_prob_output.shape}")
        
        # 比较不同推理方式的结果
        print("\n4. Comparing inference methods:")
        for i in range(min(2, frames.shape[0])):  # 只显示前2个样本
            print(f"\n   Sample {i+1}:")
            print(f"   - True target (first 5 dims): {true_targets[i][:5].numpy()}")
            print(f"   - Sampled (first 5 dims): {sampled_output[i][:5].numpy()}")
            print(f"   - Max prob (first 5 dims): {max_prob_output[i][:5].numpy()}")
            
            # 显示混合组件权重
            pi = torch.exp(mdn_params['log_pi'][i])
            weights_str = ", ".join([f"{w:.3f}" for w in pi])
            print(f"   - Mixture weights: [{weights_str}]")

def test_trajectory_visualization():
    """测试轨迹可视化功能"""
    print("\n=== Testing Trajectory Visualization ===")
    
    # 创建示例轨迹嵌入数据 (从notebook中的例子)
    num_trajectories = 20
    
    # 模拟轨迹参数 (mean 和 std)
    np.random.seed(42)
    
    # 创建几个聚类中心
    centers = np.array([
        [0.0, 0.0],   # 中心聚类
        [2.0, 1.0],   # 右上聚类
        [-1.5, -1.0], # 左下聚类
        [1.0, -2.0]   # 右下聚类
    ])
    
    means = []
    stds = []
    
    for i in range(num_trajectories):
        # 随机选择一个聚类中心
        center_idx = i % len(centers)
        center = centers[center_idx]
        
        # 在聚类中心周围添加噪声
        mean = center + np.random.normal(0, 0.3, 2)
        
        # 随机生成标准差（半径）
        std = np.random.uniform(0.1, 0.5, 2)
        
        means.append(mean)
        stds.append(std)
    
    means = np.array(means)
    stds = np.array(stds)
    
    print(f"Generated {num_trajectories} trajectory embeddings")
    print(f"Mean range: [{means.min():.2f}, {means.max():.2f}]")
    print(f"Std range: [{stds.min():.2f}, {stds.max():.2f}]")
    
    # 可视化
    visualize_trajectory_predictions(means, stds, "Example Trajectory Embeddings")

def compare_mdn_vs_deterministic():
    """比较 MDN 与确定性模型的输出"""
    print("\n=== MDN vs Deterministic Comparison ===")
    
    # 创建相同的输入数据
    frames, qpos, targets = create_sample_data(batch_size=4)
    
    # MDN 模型
    mdn_model = RobotGraspModel_v2(
        image_shape=(4, 120, 160),
        embed_dim=256,
        window_size=10,
        qpos_dim=8,
        output_type='ee_pose',
        action_chunk=50,
        num_mixtures=3
    )
    
    # 使用相同输入获取不同的输出
    with torch.no_grad():
        # MDN 输出多个样本
        samples_1 = mdn_model(frames, qpos, mode='sample')
        samples_2 = mdn_model(frames, qpos, mode='sample')
        samples_3 = mdn_model(frames, qpos, mode='sample')
        
        # 最大概率组件输出（确定性）
        max_prob = mdn_model(frames, qpos, mode='max_prob')
        
        print("Demonstrating MDN's stochastic nature:")
        print("Different samples from the same input should be different,")
        print("while max_prob output should be deterministic.\n")
        
        for i in range(min(2, frames.shape[0])):
            print(f"Sample {i+1} (first 3 dimensions):")
            print(f"  Sample 1: {samples_1[i][:3].numpy()}")
            print(f"  Sample 2: {samples_2[i][:3].numpy()}")
            print(f"  Sample 3: {samples_3[i][:3].numpy()}")
            print(f"  Max prob: {max_prob[i][:3].numpy()}")
            
            # 计算样本之间的差异
            diff_12 = torch.norm(samples_1[i] - samples_2[i]).item()
            diff_13 = torch.norm(samples_1[i] - samples_3[i]).item()
            print(f"  L2 distance (sample1 vs sample2): {diff_12:.4f}")
            print(f"  L2 distance (sample1 vs sample3): {diff_13:.4f}")
            print()

if __name__ == "__main__":
    print("MDN (Mixture Density Network) Usage Examples")
    print("=" * 50)
    
    # 设置随机种子以便复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # 1. 演示训练过程
        demonstrate_mdn_training()
        
        # 2. 演示推理过程
        demonstrate_mdn_inference()
        
        # 3. 测试轨迹可视化
        test_trajectory_visualization()
        
        # 4. 比较MDN与确定性输出
        compare_mdn_vs_deterministic()
        
        print("\n" + "=" * 50)
        print("All demonstrations completed successfully!")
        print("\nKey points about using MDN:")
        print("- Use mode='mdn_params' during training to get parameters for NLL loss")
        print("- Use mode='sample' during inference for stochastic predictions")
        print("- Use mode='max_prob' for deterministic predictions (highest probability component)")
        print("- Never use mean of mixture components - always sample or take max component!")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc() 
 
 