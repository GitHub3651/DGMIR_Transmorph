# TransMorph 显存优化指南

## 显存需求估算

### 原始配置 (192×160×192)
- **模型参数**: ~178 MB
- **前向传播**: ~2-3 GB
- **反向传播**: ~2-3 GB  
- **损失计算**: ~300 MB
- **总计**: **~5-7 GB**

## 解决方案

### 方案 1: 降低图像分辨率（最有效）⭐

#### 选项 1A: 分辨率降到 128×128×128
```bash
python train_TransMorph_custom.py --img_size 128 128 128
```
**显存需求**: ~2-3 GB  
**速度提升**: 3-4倍  
**精度影响**: 中等（仍能学到主要特征）

#### 选项 1B: 分辨率降到 96×96×96
```bash
python train_TransMorph_custom.py --img_size 96 96 96
```
**显存需求**: ~1-2 GB  
**速度提升**: 8-10倍  
**精度影响**: 较大（适合快速实验）

### 方案 2: 使用混合精度训练（FP16）

```bash
python train_TransMorph_custom.py --half_precision
```
**显存节省**: ~30-40%  
**速度提升**: 1.2-1.5倍  
**精度影响**: 几乎无（需要 GPU 支持 FP16）

### 方案 3: 使用梯度检查点

```bash
python train_TransMorph_custom.py --use_checkpoint
```
**显存节省**: ~40-50%  
**速度降低**: 20-30%  
**精度影响**: 无

### 方案 4: 组合方案（推荐）⭐⭐⭐

```bash
# 小显存卡 (4-6 GB): 降低分辨率
python train_TransMorph_custom.py --img_size 128 128 128

# 中等显存卡 (8-12 GB): 混合精度
python train_TransMorph_custom.py --half_precision

# 大显存卡 (16+ GB): 原始配置
python train_TransMorph_custom.py
```

## 不同显卡的推荐配置

| GPU 显存 | 推荐配置 | 命令 |
|---------|---------|------|
| 4 GB | 96×96×96 + FP16 | `--img_size 96 96 96 --half_precision` |
| 6 GB | 128×128×128 | `--img_size 128 128 128` |
| 8 GB | 128×128×128 + FP16 | `--img_size 128 128 128 --half_precision` |
| 12 GB | 160×128×160 | `--img_size 160 128 160` |
| 16+ GB | 192×160×192 (原始) | 默认配置 |

## DGMIR 的显存使用情况

DGMIR 模型更小，同样配置下显存需求约 **2-3 GB**，这也是为什么 TransMorph 显存消耗更大的原因。

## 其他优化技巧

1. **减少 DataLoader workers**: `num_workers=0` (节省 ~500 MB)
2. **关闭 pin_memory**: `pin_memory=False` (节省 ~200 MB)
3. **禁用梯度累积的张量**: 在验证时使用 `torch.no_grad()`
4. **清理缓存**: 每个 epoch 后调用 `torch.cuda.empty_cache()`

## 监控显存使用

```python
import torch
print(f'已分配显存: {torch.cuda.memory_allocated()/1024**3:.2f} GB')
print(f'缓存显存: {torch.cuda.memory_reserved()/1024**3:.2f} GB')
```

## 注意事项

⚠️ **降低分辨率会影响最终配准精度**  
⚠️ **混合精度需要 Volta 架构及以上的 GPU (GTX 16xx, RTX 系列)**  
⚠️ **如果只是测试代码，建议先用小分辨率快速验证**
