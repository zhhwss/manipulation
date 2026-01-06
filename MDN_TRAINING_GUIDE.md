# MDN 训练指南

## 概述

这个训练脚本专门为 Mixture Density Network (MDN) 设计，提供了完整的训练、验证和模型改进评估功能。

## 主要改进

### 1. MDN 特有的损失函数
- **训练损失**: 使用负对数似然 (NLL) 替代 MSE
- **验证指标**: 多维度评估模型性能

### 2. 综合验证指标

#### 主要指标：
- **NLL (Negative Log Likelihood)**: 主要损失函数，越低越好
- **Sample MSE**: 采样预测的均方误差，越低越好  
- **MaxProb MSE**: 最大概率组分预测的均方误差，越低越好

#### 辅助指标：
- **Mixture Entropy**: 混合组分使用的多样性，适度即可
- **Mixture Usage**: 活跃组分的比例，反映模型复杂度

### 3. 智能改进评估

模型改进不再仅看单一指标，而是综合考虑：

```python
improvement_weights = {
    'nll': 0.5,           # 主要损失，权重最高
    'max_prob_mse': 0.3,  # 确定性预测准确性
    'sample_mse': 0.15,   # 采样质量
    'mixture_entropy': 0.05  # 混合多样性（权重较小）
}
```

## 使用方法

### 基本训练
```bash
python train_mdn.py \
    --data_dir ../lhx/piper_real_cnn/data/20250808 \
    --image_type depth_mask \
    --num_mixtures 5 \
    --batch_size 8 \
    --learning_rate 1e-4
```

### 主要参数

- `--num_mixtures`: 混合组分数量 (默认: 5)
- `--unet_layers`: U-Net 层数 (默认: 1)  
- `--embed_dim`: 嵌入维度 (默认: 256)
- `--action_chunk`: 动作序列长度 (默认: 50)

## 输出解释

### 训练过程输出示例：
```
Epoch 15/500:
  Train NLL: 2.456789, Train Entropy: 1.234567
  Val NLL: 2.445678
  Val Sample MSE: 0.123456
  Val MaxProb MSE: 0.098765
  Mixture Entropy: 1.456789
  Mixture Usage: 0.800 (4.0/5 components)
  Improvement: ✓ (0.025) - Overall improvement: 2.5%. Details: nll: 1.2%, max_prob_mse: 3.4%
  ★ Best model saved! (Score: 0.0250)
```

### 指标含义：

1. **Train NLL**: 训练时的负对数似然
2. **Val NLL**: 验证时的负对数似然（主要指标）
3. **Sample/MaxProb MSE**: 不同推理方式的预测准确性
4. **Mixture Usage**: 活跃组分比例
   - 0.8 表示80%的组分在使用
   - 4.0/5 表示5个组分中有4个活跃

### 改进判断逻辑：

#### ✓ 改进情况：
- 综合改进分数 > 0
- NLL 显著改进 (>2%) 
- 大部分指标都在改善

#### ✗ 无改进情况：
- 综合改进分数 ≤ 0
- 所有指标都在恶化
- 关键指标（NLL）大幅恶化

## 最佳实践

### 1. 混合组分数量选择
- **3-5个**: 适合简单任务
- **5-10个**: 适合复杂多模态任务  
- **>10个**: 可能过拟合，需要更多数据

### 2. 验证指标监控
- **主要关注**: NLL 和 MaxProb MSE
- **辅助观察**: Mixture Usage（应在 0.6-0.9 之间）
- **警惕信号**: Usage < 0.4（组分未充分利用）或 > 0.95（可能退化为单一模式）

### 3. 超参数调优
- 学习率可以比普通模型略高（MDN训练相对稳定）
- 批量大小建议 ≥ 8（保证混合组分统计稳定）
- 早停基于综合改进分数，而非单一指标

## 文件输出

训练完成后会在 `result/mdn_training_YYYYMMDD_HHMMSS/` 目录下生成：

- `best_model.pth`: 最佳模型（基于改进评估）
- `checkpoint_epoch_*.pth`: 定期检查点
- TensorBoard 日志文件

## 故障排除

### 问题1: Mixture Usage 过低
```
Mixture Usage: 0.200 (1.0/5 components)
```
**原因**: 模型退化为单一模式
**解决**: 降低学习率，增加正则化，检查数据多样性

### 问题2: NLL 不收敛
```
Val NLL: 5.678901 (not improving)
```
**原因**: 学习率过高，或数据预处理问题
**解决**: 降低学习率，检查数据归一化

### 问题3: 所有指标都在恶化
```
Improvement: ✗ (-0.050) - All metrics degraded
```
**原因**: 过拟合或学习率问题
**解决**: 使用早停，调整学习率调度

## 与传统训练的区别

| 方面 | 传统训练 | MDN训练 |
|------|----------|---------|
| 损失函数 | MSE | NLL |
| 验证指标 | 单一MSE | 多维度指标 |
| 改进判断 | val_loss < best_loss | 综合评估分数 |
| 推理方式 | 确定性输出 | 采样/最大概率 |
| 模型复杂度 | 固定 | 动态（混合组分） |

MDN训练更加复杂但也更加灵活，能够处理多模态预测任务。 
 
 