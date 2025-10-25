# 工作空间结构说明

## 📁 项目组织结构

本工作空间包含两个独立的医学图像配准项目，已经过整理以便于管理和使用。

```
DGMIR/  (根目录)
├── DGMIR_Project/           # DGMIR 项目文件夹
│   ├── train2.py            # DGMIR 训练脚本
│   ├── test2.py             # DGMIR 测试脚本
│   ├── model2.py            # DGMIR 模型定义
│   ├── dataloader.py        # DGMIR 数据加载器
│   ├── losses.py            # 损失函数实现
│   ├── metric.py            # 评估指标实现
│   ├── functions.py         # 辅助函数
│   ├── lookup_tables.py     # 查找表
│   ├── data/                # 数据列表文件 (.txt)
│   └── PROJECT_DOCUMENTATION.md  # DGMIR 详细文档
│
├── TransMorph_Project/      # TransMorph 项目文件夹
│   ├── train_TransMorph_custom.py   # TransMorph 训练脚本
│   ├── test_TransMorph_custom.py    # TransMorph 测试脚本
│   ├── TransMorph_README.md         # TransMorph 使用说明
│   └── TransMorph_Core/             # TransMorph 核心文件
│       ├── models/                  # 模型定义
│       │   └── TransMorph.py
│       ├── data/                    # 数据加载
│       │   ├── datasets.py
│       │   └── custom_dataset.py
│       ├── losses.py                # TransMorph 损失函数
│       ├── utils.py                 # 工具函数
│       └── configs_TransMorph.py    # 配置文件
│
├── dataset/                 # 共享数据集（两个项目共用）
│   ├── train/               # 训练集 (5个样本)
│   ├── val/                 # 验证集 (2个样本)
│   └── test/                # 测试集 (1个样本)
│
├── data/                    # 原始数据列表文件
│   ├── BraTS2018_train.txt
│   ├── BraTS2018_valid.txt
│   └── BraTS2018_test.txt
│
├── generate_datalist.py     # 生成数据列表的工具脚本
├── README.md                # 项目总体说明
└── WORKSPACE_STRUCTURE.md   # 本文件 - 工作空间结构说明
```

---

## 🎯 项目说明

### **DGMIR_Project**
- **全称**: Dual-Guided Multimodal Image Registration
- **论文**: MICCAI 2025
- **任务**: 多模态医学图像配准（MR-CT 跨模态）
- **特点**: 双引导机制、多尺度特征细化、多尺度算子细化

**主要文件**:
- `train2.py`: 训练入口，配置 batch=1, epoch=300, lr=1e-4
- `test2.py`: 测试脚本，计算 Dice, HD95, NJD 等指标
- `model2.py`: DGMIR 网络架构（DualPath_encoder, MFRG, MORG）
- `losses.py`: MIND-SSC, Grad(L2), Dice 损失函数
- `PROJECT_DOCUMENTATION.md`: 完整的项目文档

---

### **TransMorph_Project**
- **全称**: Transformer for Medical Image Registration
- **论文**: Medical Image Analysis 2022
- **任务**: 基于 Transformer 的医学图像配准
- **特点**: Swin Transformer、混合 CNN-Transformer 架构

**主要文件**:
- `train_TransMorph_custom.py`: 自定义训练脚本（配置对齐 DGMIR）
- `test_TransMorph_custom.py`: 自定义测试脚本（使用 DGMIR 评估指标）
- `TransMorph_Core/`: TransMorph 核心代码
  - `models/TransMorph.py`: TransMorph 模型定义
  - `data/custom_dataset.py`: 自定义数据集类
  - `configs_TransMorph.py`: 配置文件（已调整为 192×160×192）
- `TransMorph_README.md`: 详细使用说明

---

## 🔧 配置对齐

为了公平比较两个模型，**TransMorph_Project 已配置为与 DGMIR 完全一致**：

| 配置项 | DGMIR | TransMorph (自定义) | 状态 |
|--------|-------|---------------------|------|
| **数据输入尺寸** | (192, 160, 192) | (192, 160, 192) | ✅ 一致 |
| **Batch Size** | 1 | 1 | ✅ 一致 |
| **训练轮数** | 300 | 300 | ✅ 一致 |
| **学习率** | 1e-4 | 1e-4 | ✅ 一致 |
| **优化器** | Adam | Adam | ✅ 一致 |
| **损失函数** | MIND-SSC + Grad(L2) + Dice | MIND-SSC + Grad(L2) + Dice | ✅ 一致 |
| **损失权重** | [1.0, 0.5, 0.5] | [1.0, 0.5, 0.5] | ✅ 一致 |
| **验证指标** | Dice (dicegup) | Dice (dicegup) | ✅ 一致 |
| **测试指标** | Dice, HD95, NJD | Dice, HD95, NJD | ✅ 一致 |
| **数据加载** | .txt 列表 + .nii.gz | .txt 列表 + .nii.gz | ✅ 一致 |
| **数据增强** | 无 | 无 | ✅ 一致 |

---

## 🚀 快速开始

### **训练 DGMIR**
```bash
cd DGMIR_Project
python train2.py
```

### **训练 TransMorph**
```bash
cd TransMorph_Project
python train_TransMorph_custom.py
```

### **测试 DGMIR**
```bash
cd DGMIR_Project
python test2.py
```

### **测试 TransMorph**
```bash
cd TransMorph_Project
python test_TransMorph_custom.py
```

---

## 📊 数据集信息

- **来源**: TCIA (The Cancer Imaging Archive)
- **任务**: MR → CT 跨模态配准
- **样本数**: 8 个配对样本
- **图像尺寸**: (192, 160, 192)
- **体素间距**: 2mm × 2mm × 2mm
- **分割类别**: 5 类（0: 背景, 1-4: 前景类别）

**数据划分**:
- 训练集: 5 个样本 (0006, 0008, 0010, 0014, 0016)
- 验证集: 2 个样本 (0002, 0012)
- 测试集: 1 个样本 (0004)

---

## 📝 注意事项

1. **共享数据集**: `dataset/` 文件夹被两个项目共用，请勿重复移动或删除
2. **路径引用**: 训练和测试脚本中的路径已更新为相对于各自项目文件夹
3. **损失函数复用**: TransMorph 项目复用了 DGMIR 的损失函数实现（从 `../DGMIR_Project/losses.py` 导入）
4. **评估指标复用**: TransMorph 测试脚本使用 DGMIR 的评估指标（从 `../DGMIR_Project/metric.py` 导入）
5. **模型保存**: 两个项目的模型分别保存在各自的 `experiments/` 文件夹中

---

## 🔄 工作流程建议

### **对比实验流程**:
1. **数据准备**: 确认 `dataset/` 中的数据已正确划分
2. **训练 DGMIR**: 在 `DGMIR_Project/` 中训练，保存最佳模型
3. **训练 TransMorph**: 在 `TransMorph_Project/` 中训练，使用相同配置
4. **评估对比**: 使用各自的测试脚本在测试集上评估
5. **结果分析**: 比较两个模型在 Dice, HD95, NJD 指标上的表现

---

## 📚 更多信息

- DGMIR 详细文档: `DGMIR_Project/PROJECT_DOCUMENTATION.md`
- TransMorph 使用说明: `TransMorph_Project/TransMorph_README.md`
- 原始 README: `README.md`

---

**最后更新**: 2025年10月25日
