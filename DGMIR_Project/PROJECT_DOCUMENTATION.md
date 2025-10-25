# DGMIR 项目详细说明文档

## 项目概述

**DGMIR (Dual-Guided Multimodal Medical Image Registration)** 是一个基于深度学习的医学图像配准框架，发表于 MICCAI 2025 会议。该项目专注于多模态医学图像配准，采用双引导机制结合多视图增强和现场模态移除技术。

### 论文信息
- **标题**: DGMIR: Dual-Guided Multimodal Medical Image Registration Based on Multi-view Augmentation and On-Site Modality Removal
- **会议**: MICCAI 2025
- **论文链接**: 
  - https://papers.miccai.org/miccai-2025/paper/1691_paper.pdf
  - https://link.springer.com/chapter/10.1007/978-3-032-04927-8_15

### 作者
Le, Gao; Shu, Yucheng; Qiao, Lihong; Yang, Lijian; Xiao, Bin; Li, Weisheng; Gao, Xinbo

---

## 项目结构

```
DGMIR/
├── model2.py              # 核心网络模型定义
├── train2.py              # 训练脚本
├── test2.py               # 测试和评估脚本
├── dataloader.py          # 数据加载器
├── losses.py              # 损失函数定义
├── functions.py           # 实用函数（空间变换等）
├── metric.py              # 评估指标（Dice、Hausdorff距离等）
├── lookup_tables.py       # 表面距离计算的查找表
├── README.md              # 项目简介
└── data/                  # 数据集配置
    ├── BraTS2018_train.txt    # 训练集路径列表
    ├── BraTS2018_valid.txt    # 验证集路径列表
    ├── BraTS2018_test.txt     # 测试集路径列表
    └── README.md              # 数据格式说明
```

---

## 核心模块详解

### 1. 模型架构 (model2.py)

#### 1.1 主要组件

**DGMIR 主模型**
- **输入**: 固定图像(fixed)和移动图像(moving)，维度为 [B, 1, 160, 192, 128]
- **输出**: 形变场(flow)和配准后的图像(warped)
- **模式**: 支持 'train' 和 'test' 两种模式

#### 1.2 网络结构

**DualPath_encoder (双路径编码器)**
```
特点：
- 共享权重的双路径设计，分别处理固定图像和移动图像
- 4层卷积下采样结构
- 通道数递增：1 → 16 → 32 → 32 → 64
- 使用 LeakyReLU(0.2) 激活函数
```

**MFRG (Multi-scale Feature Refinement and Guidance)**
```
功能：多尺度特征精炼和引导模块
- 自适应平均池化和最大池化
- 通道注意力机制
- 可选的预特征融合
- Mix 模块实现自适应特征融合
```

**MORG (Multi-scale Operator Refinement and Guidance)**
```
功能：多尺度算子精炼和引导模块
- 分组卷积提取局部特征
- 空洞卷积增大感受野
- 全局平均卷积（可学习温度参数）
- 空间相关性建模
```

**Flow_decoder (形变场解码器)**
```
结构：
- 4个尺度的渐进式解码
- 每层包含 MFRG + MORG + RegHead
- 使用空间变换器进行形变累积
- 双线性插值进行上采样（factor=2）
```

**RegHead_block (配准头模块)**
```
组成：
- 两个卷积块（可配置层数）
- InstanceNorm + LeakyReLU
- 最终输出3通道形变场
- 权重初始化：Normal(0, 1e-5)
```

#### 1.3 关键设计

1. **Mix 模块**: 可学习的特征融合权重
2. **ResizeTransformer**: 支持图像和形变场的缩放
3. **渐进式形变累积**: flow = STN(flow, flow_i) + flow_i

### 2. 训练流程 (train2.py)

#### 2.1 训练配置
```python
配置参数:
- 批次大小 (batch): 1
- 训练轮数 (n_epoch): 300
- 学习率 (lr): 1e-4（采用多项式衰减，power=0.9）
- 图像尺寸 (vol_size): [160, 192, 128]
- 损失权重 (weight): [1.0, 0.5, 0.5]  # [相似度, 平滑度, Dice]
- 分割类别数 (cls_num): 29
```

#### 2.2 损失函数

**组合损失 = 相似度损失 + 平滑度损失 + Dice损失**

1. **相似度损失**: MIND-SSC (Modality Independent Neighbourhood Descriptor)
   - 模态无关的邻域描述子
   - 适合多模态图像配准

2. **平滑度损失**: L2范数梯度约束
   - 确保形变场的平滑性
   - 避免非物理变形

3. **Dice损失**: 基于分割标签的监督
   - 使用 one-hot 编码
   - 计算29个类别的平均Dice

#### 2.3 训练特点

- **优化器**: Adam
- **学习率策略**: 多项式衰减
- **数据加载**: 每次只使用数据集的1/4（通过迭代器控制）
- **最佳模型保存**: 基于验证集Dice分数
- **检查点保存**: 包含模型权重、优化器状态和当前epoch

### 3. 测试流程 (test2.py)

#### 3.1 评估指标

1. **NJD (Negative Jacobian Determinant)**
   - 评估形变场的可逆性
   - 计算雅可比行列式 < 0 的比例
   - 理想值：接近0

2. **HD95 (95th Percentile Hausdorff Distance)**
   - 鲁棒的表面距离度量
   - 考虑表面元素面积
   - 单位：mm

3. **Dice 系数**
   - 分割重叠度评估
   - 范围：[0, 1]，越大越好

#### 3.2 测试特点

- 使用 `torch.no_grad()` 加速推理
- 支持最近邻插值进行分割图像变换
- 逐类别计算HD95（排除背景和空类别）

### 4. 数据加载 (dataloader.py)

#### 4.1 数据格式

**文本文件格式** (每4行为一组训练样本):
```
Line 1: 固定图像分割标签路径
Line 2: 移动图像分割标签路径
Line 3: 固定图像路径
Line 4: 移动图像路径
```

#### 4.2 数据处理流程

1. **读取**: 使用 nibabel 读取 NIfTI 格式文件
2. **裁剪**: 中心裁剪到目标尺寸 [160, 192, 128]
3. **归一化**: 如果像素均值 > 1，则进行最小-最大归一化
4. **转换**: 转为 CUDA tensor，dtype=float32

#### 4.3 关键函数

- `imgnorm()`: 最小-最大归一化
- `center_crop()`: 3D中心裁剪
- `GetLoader_Brats2018`: BraTS2018数据集加载器

### 5. 损失函数库 (losses.py)

#### 5.1 相似度度量

1. **NCC Loss (Normalized Cross Correlation)**
   - 局部归一化互相关
   - 窗口大小：9×9×9
   - 适合单模态配准

2. **MIND-SSC Loss**
   ```
   参数:
   - radius: 2 (自相似上下文半径)
   - dilation: 2 (邻域扩张率)
   - penalty: 'l2' (惩罚模式)
   
   特点:
   - 12个自相似上下文元素
   - 6-邻域结构（上下左右前后）
   - 模态无关性
   ```

3. **Mutual Information**
   - 互信息损失
   - 基于高斯近似的直方图
   - 支持局部互信息变体

#### 5.2 形变正则化

**Grad Loss**
- L1 或 L2 范数
- 计算形变场的空间梯度
- 多尺度平均

#### 5.3 分割监督

1. **compute_per_channel_dice**
   - 每通道Dice计算
   - 支持 one-hot 编码
   - 可选的通道权重

2. **dicegup**
   - 基于唯一类别的Dice计算
   - 自动排除背景
   - 用于测试阶段

### 6. 评估指标 (metric.py)

#### 6.1 表面距离计算

**compute_surface_distances()**
```
功能：计算两个分割掩码之间的表面距离

步骤：
1. 计算掩码的边界框
2. 裁剪到最小处理子体积
3. 使用邻域编码（2×2×2）识别表面体素
4. 距离变换计算最近距离
5. 根据表面元素面积排序

输出：
- distances_gt_to_pred: GT到预测的距离
- distances_pred_to_gt: 预测到GT的距离
- surfel_areas_gt: GT表面元素面积
- surfel_areas_pred: 预测表面元素面积
```

**compute_robust_hausdorff()**
- 计算指定百分位数的Hausdorff距离
- 考虑表面元素面积权重
- 通常使用95%分位数（HD95）

#### 6.2 形变质量

**jacobian_determinant_gpu()**
```
功能：GPU加速的雅可比行列式计算

实现：
- 使用3D卷积计算梯度
- 分别计算x, y, z方向梯度
- 计算3×3雅可比矩阵的行列式
- 排除边界区域（各边2个体素）
```

### 7. 实用函数 (functions.py)

#### 7.1 空间变换器

**SpatialTransformer2**
```python
特点：
- 支持任意尺寸输入
- 动态创建采样网格
- 支持 'bilinear' 和 'nearest' 插值
- 用于图像和分割的形变

使用：
warped = stn(moving, flow, mode='bilinear')
warped_seg = stn(moving_seg, flow, mode='nearest')
```

#### 7.2 其他工具

- `generate_grid_unit()`: 生成归一化网格
- `model_structure()`: 打印模型结构和参数量
- `compute_module_memory_usage()`: 计算显存占用
- `count_labels()`: 统计NIfTI标签分布

### 8. 查找表 (lookup_tables.py)

#### 8.1 用途
- 支持表面距离计算
- 基于Marching Cubes算法
- 预定义256种2×2×2邻域配置的表面法向量

#### 8.2 核心表

1. **ENCODE_NEIGHBOURHOOD_3D_KERNEL**: 3D邻域编码核
2. **_NEIGHBOUR_CODE_TO_NORMALS**: 邻域代码到表面法向量映射
3. **create_table_neighbour_code_to_surface_area()**: 创建表面积查找表

---

## 使用指南

### 环境要求

```bash
# 主要依赖
torch >= 1.8.0
nibabel
numpy
scipy
```

### 数据准备

1. **准备BraTS2018数据集**（或其他脑部MRI数据集）

2. **创建路径列表文件**
   
   在 `data/` 目录下创建三个文本文件：
   - `BraTS2018_train.txt`
   - `BraTS2018_valid.txt`
   - `BraTS2018_test.txt`

   格式示例：
   ```
   /path/to/fixed_seg.nii.gz
   /path/to/moving_seg.nii.gz
   /path/to/fixed_img.nii.gz
   /path/to/moving_img.nii.gz
   /path/to/fixed_seg2.nii.gz
   /path/to/moving_seg2.nii.gz
   /path/to/fixed_img2.nii.gz
   /path/to/moving_img2.nii.gz
   ...
   ```

### 训练模型

```python
python train2.py
```

**可调整参数** (在 train2.py 中修改):
- `batch`: 批次大小（默认1）
- `n_epoch`: 训练轮数（默认300）
- `lr`: 初始学习率（默认1e-4）
- `vol_size`: 图像尺寸（默认[160, 192, 128]）
- `weight`: 损失权重（默认[1, 0.5, 0.5]）
- `model_path`: 模型保存路径

### 测试模型

```python
python test2.py
```

**需要修改的参数**:
- `model_path`: 训练好的模型路径
- `ckpt_name`: 检查点文件名
- `test_path`: 测试数据路径列表

### 继续训练

在 `train2.py` 中取消注释以下代码：
```python
checkpoint = torch.load(os.path.join(model_path, 'ckpt.pth'))
net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch_last = checkpoint['epoch']
```

---

## 技术亮点

### 1. 双路径架构
- 固定图像和移动图像独立编码，共享权重
- 保留各自的特征表示

### 2. MFRG模块
- 多尺度特征融合
- 通道注意力机制
- 自适应权重学习

### 3. MORG模块
- 多尺度空间上下文
- 全局-局部特征对比
- 空间相关性建模

### 4. 渐进式配准
- 由粗到精的4层金字塔
- 形变场逐层累积和细化
- 提高配准精度和收敛速度

### 5. MIND-SSC损失
- 模态无关性
- 对噪声和强度变化鲁棒
- 适合多模态配准任务

### 6. 多重监督
- 图像相似度 + 形变平滑度 + 分割Dice
- 充分利用标注信息
- 提高配准质量

---

## 性能评估

### 评估指标

| 指标 | 说明 | 理想值 |
|------|------|--------|
| Dice | 分割重叠度 | 越大越好 (0-1) |
| HD95 | 95%分位Hausdorff距离 | 越小越好 (mm) |
| NJD Mean | 负雅可比比例 | 越小越好 (接近0) |
| NJD Std | 雅可比标准差 | 越小越好 |

### 测试输出示例
```
NJD_mean: 0.001234, NJD_std: 0.056789, 
HD95: 2.345±0.678, 
DICE: 0.8567±0.0234
```

---

## 常见问题

### Q1: 显存不足怎么办？
**解决方案**:
1. 减小 `vol_size`
2. 确保 `batch=1`
3. 使用梯度累积
4. 减少模型通道数

### Q2: 如何适配其他数据集？
**步骤**:
1. 修改 `dataloader.py` 中的数据加载逻辑
2. 调整 `vol_size` 适配图像尺寸
3. 修改 `cls_num` 适配分割类别数
4. 根据需要调整损失权重

### Q3: 训练不收敛？
**检查项**:
1. 学习率是否合适
2. 损失权重是否平衡
3. 数据归一化是否正确
4. 初始化是否合理

### Q4: 如何加速训练？
**建议**:
1. 使用混合精度训练（AMP）
2. 增大批次大小（如果显存允许）
3. 使用多GPU训练（需修改代码）
4. 使用更高效的数据加载器

---

## 引用

如果您在研究中使用此代码，请引用：

```bibtex
@inproceedings{le2025dgmir,
  title={DGMIR: Dual-Guided Multimodal Medical Image Registration Based on Multi-view Augmentation and On-Site Modality Removal},
  author={Le, Gao and Shu, Yucheng and Qiao, Lihong and Yang, Lijian and Xiao, Bin and Li, Weisheng and Gao, Xinbo},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={152--162},
  year={2025},
  organization={Springer}
}
```

---

## 许可证

请参考原始论文和代码库的许可证条款。

---

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 GitHub Issue
- 参考原始论文获取作者联系方式

---

## 更新日志

- **2025-10**: 初始版本发布
- 基于 MICCAI 2025 论文实现

---

**文档版本**: 1.0  
**最后更新**: 2025年10月23日  
**作者**: 基于原始代码库整理
