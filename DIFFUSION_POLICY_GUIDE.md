# Diffusion Policy 训练指南

## 概述

这个实现将机器人抓取任务改造为 Diffusion Policy，使用 DDPM (Denoising Diffusion Probabilistic Models) 来生成机器人动作轨迹。

## 🎯 主要特点

### 1. **Diffusion Policy 架构**
- **条件输入**: 图像 (frames) + 机器人姿态 (qpos) 作为条件，在整个去噪过程中保持不变
- **高效特征复用**: 条件特征 (frames+qpos) 只计算一次，在所有去噪步骤中重复使用
- **U-Net 骨干**: 在 action_chunk 维度上进行真实的下采样/上采样
- **时间步嵌入**: 使用正弦位置编码表示当前去噪时间步

### 2. **训练过程**
```python
# 每个训练步骤:
timesteps = torch.randint(0, num_diffusion_steps, (batch_size,))  # 随机时间步
noise = torch.randn_like(targets)                                 # 随机噪声
noisy_actions = model.add_noise(targets, timesteps, noise)        # 加噪声
predicted_noise = model.predict_noise(noisy_actions, frames, qpos, timesteps)  # 预测噪声
loss = mse_loss(predicted_noise, noise)                           # MSE损失
```

### 3. **验证过程**
- **噪声预测损失**: 评估模型预测噪声的能力
- **完整去噪评估**: 从纯噪声开始，执行完整的DDPM去噪过程，计算最终结果与目标的MSE

## 🚀 使用方法

### 基本训练
```bash
python train_dp.py \
    --data_dir ../lhx/piper_real_cnn/data/20250808 \
    --image_type depth_mask \
    --num_diffusion_steps 100 \
    --batch_size 8 \
    --learning_rate 1e-4
```

### 主要参数

- `--num_diffusion_steps`: 扩散步数 (默认: 100)
- `--unet_layers`: U-Net 层数 (默认: 1)  
- `--embed_dim`: 嵌入维度 (默认: 256)
- `--action_chunk`: 动作序列长度 (默认: 50)

## 📊 训练输出解释

### 训练过程输出示例：
```
Epoch 15/500:
  Train Noise Loss: 0.045678
  Val Noise Loss: 0.043210
  Val Denoising MSE: 0.012345
  Improvement: ✓ (0.025) - Overall improvement: 2.5%. Details: denoising_mse: 3.2%, noise_loss: 1.8%
  ★ Best model saved! (Score: 0.0250)
```

### 指标含义：

1. **Train Noise Loss**: 训练时的噪声预测损失 (MSE)
2. **Val Noise Loss**: 验证时的噪声预测损失 (MSE) 
3. **Val Denoising MSE**: 完整去噪过程后的动作MSE
4. **Improvement Score**: 综合改进评分

### 改进判断逻辑：

#### ✓ 改进情况：
- 综合改进分数 > 0
- 去噪MSE显著改进 (>2%)
- 大部分指标都在改善

#### ✗ 无改进情况：
- 综合改进分数 ≤ 0
- 所有指标都在恶化

## 🔧 核心技术细节

### 1. **噪声调度 (Noise Schedule)**
```python
def _cosine_beta_schedule(self, timesteps):
    s = 0.008
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)
```

### 2. **高效条件特征计算**
```python
# 条件特征只计算一次
frames_flat = frames.view(batch_size, window_size * in_channels, *image_size)
qpos_flat = qpos.view(batch_size, window_size * qpos_dim)
h = self.cnn(frames_flat)
h1 = self.fc1(h)
h2 = self.fc2(qpos_flat)
cond_features = torch.cat([h1, h2], dim=-1)  # 在所有时间步重复使用
```

### 3. **时间步嵌入**
```python
def _timestep_embedding(self, timesteps, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half) / half)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding
```

### 4. **去噪步骤**
```python
def denoise_step(self, noisy_actions, frames, qpos, timestep, cond_features=None):
    # 预测噪声
    predicted_noise = self.predict_noise(noisy_actions, frames, qpos, timestep, cond_features)
    
    # DDPM去噪公式
    alpha_t = self.alphas[timestep]
    alpha_cumprod_t = self.alphas_cumprod[timestep]
    beta_t = self.betas[timestep]
    
    pred_original_sample = (noisy_actions - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / sqrt(alpha_cumprod_t)
    # ... DDPM标准公式
    
    return pred_prev_sample
```

## 🎯 推理/采样

### 从模型采样动作
```python
# 从模型采样
sampled_actions = model.sample(frames, qpos, num_samples=5)  # (batch, num_samples, action_dim)

# 单次采样
with torch.no_grad():
    actions = model.sample(frames, qpos, num_samples=1)[:, 0]  # (batch, action_dim)
```

### DDPM采样过程
```python
# 从纯噪声开始
actions = torch.randn(batch_size, action_dim)

# 执行去噪步骤
for t in reversed(range(num_diffusion_steps)):
    timestep = torch.full((batch_size,), t)
    if t > 0:
        noise = torch.randn_like(actions)
        actions = model.denoise_step(actions, frames, qpos, timestep) + sqrt(betas[t]) * noise
    else:
        actions = model.denoise_step(actions, frames, qpos, timestep)
```

## 📈 性能优化

### 1. **验证时的快速去噪**
- 只使用20步去噪（而非训练时的100步）
- 每5个batch才执行一次完整去噪评估
- 条件特征预计算并重复使用

### 2. **内存优化**
- 条件特征计算与去噪过程分离
- 避免在去噪循环中重复计算CNN特征
- 使用 `torch.no_grad()` 进行推理

### 3. **训练效率**
- 每个epoch随机采样时间步
- 并行处理整个batch的噪声添加
- 重用计算图避免不必要的梯度计算

## 🆚 与传统方法对比

| 方面 | 传统回归 | Diffusion Policy |
|------|----------|------------------|
| 输出方式 | 确定性单一输出 | 概率分布采样 |
| 损失函数 | MSE | 噪声预测MSE |
| 训练复杂度 | 简单 | 中等（需要时间步） |
| 推理速度 | 快 (单次前向) | 慢 (多步去噪) |
| 表达能力 | 有限 | 强 (多模态) |
| 泛化能力 | 一般 | 好 (随机性) |

## 🔧 故障排除

### 问题1: 去噪MSE不收敛
```
Val Denoising MSE: 5.678901 (not improving)
```
**原因**: 学习率过高，或去噪步数不足
**解决**: 降低学习率，增加 `--num_diffusion_steps`

### 问题2: 噪声损失震荡
```
Train Noise Loss fluctuating wildly
```
**原因**: 批量大小过小，时间步采样不稳定  
**解决**: 增加 `--batch_size`，使用学习率调度

### 问题3: 推理速度太慢
```
Sampling takes too long during validation
```
**原因**: 去噪步数过多
**解决**: 推理时使用DDIM采样，或减少去噪步数

## 🎉 主要优势

1. **多模态输出**: 能够生成多样化的动作轨迹
2. **鲁棒性强**: 通过随机性提高泛化能力  
3. **高质量采样**: DDPM提供高质量的动作序列
4. **条件控制精确**: 图像和机器人状态精确控制生成过程
5. **可扩展性好**: 易于扩展到更复杂的动作空间

Diffusion Policy为机器人学习提供了一个强大而灵活的新范式！ 
 
 