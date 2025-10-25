# TransMorph Training Pipeline for Custom Dataset

本文档说明如何使用 TransMorph 训练和测试脚本，与 DGMIR 进行公平对比实验。

## 📁 文件结构

```
DGMIR/
├── dataset/                              # 数据集目录
│   ├── train/                           # 训练集 (5样本)
│   ├── val/                             # 验证集 (2样本)
│   └── test/                            # 测试集 (1样本)
├── TransMorph_Core/                     # TransMorph核心代码
│   ├── models/                          # 模型定义
│   ├── data/
│   │   └── custom_dataset.py           # ✨ 自定义Dataset类
│   └── configs_TransMorph_custom.py    # ✨ 自定义配置
├── train_TransMorph_custom.py          # ✨ 训练脚本
├── test_TransMorph_custom.py           # ✨ 测试脚本
└── TransMorph_output/                   # 输出目录 (将自动创建)
    └── models/                          # 模型保存目录
```

## ⚙️ 配置对齐 (与 DGMIR 一致)

| 配置项 | DGMIR | TransMorph (本实现) | 说明 |
|--------|-------|---------------------|------|
| **数据加载** | GetLoader_Brats2018 | CustomDataset | 相同逻辑 |
| **数据格式** | .nii.gz + txt列表 | .nii.gz (目录结构) | 功能等价 |
| **图像尺寸** | (192, 160, 192) | (192, 160, 192) | ✓ |
| **类别数** | 5 (0-4) | 5 (0-4) | ✓ |
| **归一化** | Min-Max | Min-Max | ✓ |
| **中心裁剪** | center_crop | center_crop | ✓ |
| **Batch Size** | 1 | 1 | ✓ |
| **Epochs** | 300 | 300 | ✓ |
| **Learning Rate** | 1e-4 | 1e-4 | ✓ |
| **优化器** | Adam | Adam | ✓ |
| **损失函数** | MIND-SSC + Grad + Dice | MIND-SSC + Grad + Dice | ✓ |
| **损失权重** | [1.0, 0.5, 0.5] | [1.0, 0.5, 0.5] | ✓ |
| **验证指标** | Dice (dicegup) | Dice (dicegup) | ✓ |
| **测试指标** | Dice, HD95, NJD | Dice, HD95, NJD | ✓ |
| **数据增强** | 无 | 无 | ✓ |
| **TensorBoard** | 无 | 无 | ✓ |

## 🚀 使用方法

### 1. 训练 TransMorph

```bash
# 基本训练命令
python train_TransMorph_custom.py --data_root ./dataset --output_dir ./TransMorph_output --gpu 0

# 完整参数
python train_TransMorph_custom.py \
    --data_root ./dataset \
    --output_dir ./TransMorph_output \
    --batch_size 1 \
    --n_epoch 300 \
    --lr 1e-4 \
    --img_size 192 160 192 \
    --cls_num 5 \
    --gpu 0
```

**训练参数说明：**
- `--data_root`: 数据集根目录 (包含train/val/test子目录)
- `--output_dir`: 输出目录 (保存模型和日志)
- `--batch_size`: 批大小 (默认1, 与DGMIR一致)
- `--n_epoch`: 训练轮数 (默认300)
- `--lr`: 学习率 (默认1e-4)
- `--img_size`: 图像尺寸 (默认192 160 192)
- `--cls_num`: 分割类别数 (默认5, 包括背景)
- `--gpu`: GPU编号 (默认0)
- `--resume`: 从检查点恢复训练 (可选)

**输出文件：**
- `TransMorph_output/models/TransMorph_best.pth`: 最佳Dice模型
- `TransMorph_output/models/TransMorph_final.pth`: 最终epoch模型
- `TransMorph_output/models/TransMorph_epoch_*.pth`: 每50轮保存

### 2. 测试 TransMorph

```bash
# 测试最佳模型 (test集)
python test_TransMorph_custom.py \
    --data_root ./dataset \
    --model_path ./TransMorph_output/models/TransMorph_best.pth \
    --split test \
    --gpu 0

# 测试验证集
python test_TransMorph_custom.py \
    --data_root ./dataset \
    --model_path ./TransMorph_output/models/TransMorph_best.pth \
    --split val \
    --gpu 0
```

**测试参数说明：**
- `--model_path`: 模型检查点路径 (必需)
- `--split`: 测试数据集 (val 或 test)
- 其他参数同训练脚本

**评估指标 (与DGMIR完全一致)：**
1. **Dice**: 平均Dice系数 (使用dicegup, 排除背景)
2. **HD95**: Hausdorff距离95百分位数
3. **NJD_mean**: 负Jacobian行列式平均值 (衡量形变合理性)
4. **NJD_std**: 负Jacobian行列式标准差

输出格式:
```
NJD_mean: 0.001234, NJD_std: 0.123456, HD95: 2.345678±0.456789, DICE: 0.823456±0.045678
```

### 3. 从检查点恢复训练

```bash
python train_TransMorph_custom.py \
    --data_root ./dataset \
    --output_dir ./TransMorph_output \
    --resume ./TransMorph_output/models/TransMorph_epoch_50.pth \
    --gpu 0
```

## 📊 训练过程监控

训练时会显示：
```
Epoch 1/300: 100%|██████████| 5/5 [00:15<00:00, 3.12s/it, Loss=1.2345, MIND=0.1234, Grad=0.5678, Dice=0.5433]
[Train] Epoch 1 - Loss: 1.2345, MIND: 0.1234, Grad: 0.5678, Dice: 0.5433
[Val] Epoch 1 - Dice: 0.4567
*** New best Dice: 0.4567 ***
```

## 🔍 与 DGMIR 对比实验

完整的对比流程：

1. **训练 DGMIR** (已有):
```bash
python train2.py
```

2. **训练 TransMorph** (新):
```bash
python train_TransMorph_custom.py --data_root ./dataset --output_dir ./TransMorph_output --gpu 0
```

3. **测试 DGMIR**:
```bash
python test2.py
```

4. **测试 TransMorph**:
```bash
python test_TransMorph_custom.py \
    --data_root ./dataset \
    --model_path ./TransMorph_output/models/TransMorph_best.pth \
    --split test \
    --gpu 0
```

5. **对比结果**:
   - 使用相同的4个指标: Dice, HD95, NJD_mean, NJD_std
   - 相同的损失函数和权重
   - 相同的数据集划分和预处理
   - 确保公平对比

## 📝 代码核心组件

### CustomDataset (data/custom_dataset.py)
- 从 `dataset/` 目录结构读取数据
- 与 DGMIR 的 `GetLoader_Brats2018` 相同的预处理流程
- 返回 `(fixed, moving, fixed_seg, moving_seg)` 四元组

### 损失函数 (train_TransMorph_custom.py)
```python
# 与 DGMIR 完全一致
mind_loss = MINDSSCLoss()(x_def, fixed)           # 图像相似度
grad_loss = Grad(penalty='l2')(flow)               # 形变场平滑度
dice_loss = 1.0 - mean(compute_per_channel_dice()) # 分割监督

total_loss = 1.0*mind_loss + 0.5*grad_loss + 0.5*dice_loss
```

### 评估指标 (test_TransMorph_custom.py)
```python
# 复用 DGMIR 的评估函数
dice_score = dicegup(def_seg, fixed_seg, num_classes=5)
hd95 = compute_robust_hausdorff(surface_distances, 95.0)
njd = jacobian_determinant_gpu(flow)
```

## ⚠️ 注意事项

1. **窗口大小调整**: TransMorph 的 `window_size` 必须能整除 `img_size`
   - 原始: img_size=(160,192,224), window_size=(5,6,7)
   - 自定义: img_size=(192,160,192), window_size=(6,5,6)
   - 验证: 192/6=32✓, 160/5=32✓, 192/6=32✓

2. **GPU显存**: TransMorph参数量~15M, batch_size=1时约需8GB显存

3. **数据格式**: 确保数据集目录结构正确
   ```
   dataset/train/fixed/image/*.nii.gz
   dataset/train/fixed/seg/*.nii.gz
   dataset/train/moving/image/*.nii.gz
   dataset/train/moving/seg/*.nii.gz
   ```

4. **模型保存**: 
   - 每个epoch后验证，保存最佳Dice模型
   - 每50轮保存检查点
   - 训练结束保存最终模型

## 🐛 问题排查

**问题1: 导入错误 `cannot import TransMorph`**
- 确保在项目根目录运行脚本
- 脚本中已添加 `sys.path.append('./TransMorph_Core')`

**问题2: CUDA out of memory**
- 确认 batch_size=1
- 检查GPU显存 (建议≥8GB)
- 尝试减小 embed_dim (96→48)

**问题3: 窗口大小不匹配**
- 使用 `configs_TransMorph_custom.py` 中的配置
- window_size=(6,5,6) 适配 img_size=(192,160,192)

**问题4: 数据集找不到**
- 检查 `--data_root` 路径
- 确认 dataset/train, dataset/val, dataset/test 存在
- 查看 CustomDataset 的文件列表输出

## 📚 参考

- **TransMorph 论文**: Chen et al., "TransMorph: Transformer for unsupervised medical image registration", Medical Image Analysis, 2022
- **DGMIR 论文**: 您的MICCAI 2025投稿
- **数据集**: TCIA MR-CT Cross-modal Registration

---

**创建时间**: 2025-10-25  
**作者**: GitHub Copilot  
**版本**: 1.0
