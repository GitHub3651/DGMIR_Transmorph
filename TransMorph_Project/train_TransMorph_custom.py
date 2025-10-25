"""
TransMorph 训练脚本 - 配置对齐 DGMIR
作者: GitHub Copilot
描述: 使用与 DGMIR 相同的损失函数、评估指标和超参数训练 TransMorph
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# 获取脚本所在目录并添加 TransMorph_Core 到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'TransMorph_Core'))

# 导入 TransMorph 组件
from TransMorph_Core.models.TransMorph import CONFIGS as CONFIGS_TM
from TransMorph_Core.models.TransMorph import TransMorph
from TransMorph_Core.models.configs_TransMorph import get_3DTransMorph_config

# 导入自定义数据集
from TransMorph_Core.data.custom_dataset import CustomDataset

# 导入 DGMIR 损失函数和评估指标（现在在 TransMorph_Core 中）
from TransMorph_Core.losses import MINDSSCLoss, Grad, compute_per_channel_dice, dicegup
from TransMorph_Core.functions import SpatialTransformer2


def count_parameters(model):
    """统计模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_dirs(output_dir):
    """创建输出目录"""
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def save_checkpoint(model, optimizer, epoch, best_dice, save_path, history=None):
    """保存模型检查点"""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_dice': best_dice,
    }
    
    # 保存训练历史（用于后续画图）
    if history is not None:
        state['history'] = history
    
    torch.save(state, save_path)
    print(f'模型已保存至 {save_path}')


def train_epoch(model, train_loader, optimizer, device, epoch, args):
    """训练一个 epoch"""
    model.train()
    
    # 损失函数（与 DGMIR 相同）
    mind_loss_fn = MINDSSCLoss()
    grad_loss_fn = Grad(penalty='l2')
    
    # 用于分割的空间变换器
    stn = SpatialTransformer2().to(device)
    
    # 损失权重 [MIND, Grad, Dice]
    weights = [1.0, 0.5, 0.5]
    
    total_loss = 0.0
    total_mind_loss = 0.0
    total_grad_loss = 0.0
    total_dice_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.n_epoch}')
    
    for batch_idx, (fixed, moving, fixed_seg, moving_seg) in enumerate(pbar):
        # 移动到设备（与 DGMIR 一致）
        fixed = fixed.to(device)  # (B, D, H, W)
        moving = moving.to(device)  # (B, D, H, W)
        fixed_seg = fixed_seg.to(device)  # (B, D, H, W)
        moving_seg = moving_seg.to(device)  # (B, D, H, W)
        
        # 添加 channel 维度（与 DGMIR 一致）
        fixed = torch.unsqueeze(fixed, dim=1)  # (B, 1, D, H, W)
        moving = torch.unsqueeze(moving, dim=1)  # (B, 1, D, H, W)
        fixed_seg = torch.unsqueeze(fixed_seg, dim=1)  # (B, 1, D, H, W)
        moving_seg = torch.unsqueeze(moving_seg, dim=1)  # (B, 1, D, H, W)
        
        # 将 fixed 和 moving 拼接作为输入 (B, 2, D, H, W)
        x_in = torch.cat([moving, fixed], dim=1)
        
        # 前向传播
        x_def, flow = model(x_in)
        
        # 使用变形场变换分割（使用 SpatialTransformer2，与 DGMIR 相同）
        def_seg = stn(moving_seg, flow)
        
        # 计算损失
        # 1. MIND-SSC 损失（图像相似度）
        mind_loss = mind_loss_fn(x_def, fixed)
        
        # 2. 梯度损失(平滑性)
        grad_loss = grad_loss_fn(flow)
        
        # 3. Dice 损失(分割监督)
        # compute_per_channel_dice 已经返回负的 Dice 均值作为损失
        dice_loss = compute_per_channel_dice(def_seg, fixed_seg, classes=args.cls_num)
        
        # 带权重的总损失
        loss = weights[0] * mind_loss + weights[1] * grad_loss + weights[2] * dice_loss
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 累积损失
        total_loss += loss.item()
        total_mind_loss += mind_loss.item()
        total_grad_loss += grad_loss.item()
        total_dice_loss += dice_loss.item()
        
        # 更新进度条（注：Dice损失是负值，-1到0之间，越接近0越好）
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'MIND': f'{mind_loss.item():.4f}',
            'Grad': f'{grad_loss.item():.4f}',
            'DiceLoss': f'{dice_loss.item():.4f}'
        })
    
    # 平均损失
    n_batches = len(train_loader)
    avg_loss = total_loss / n_batches
    avg_mind = total_mind_loss / n_batches
    avg_grad = total_grad_loss / n_batches
    avg_dice = total_dice_loss / n_batches
    
    print(f'[训练] Epoch {epoch} - Loss: {avg_loss:.4f}, MIND: {avg_mind:.4f}, Grad: {avg_grad:.4f}, Dice: {avg_dice:.4f}')
    
    # 返回所有损失信息（用于记录历史）
    return {
        'total_loss': avg_loss,
        'mind_loss': avg_mind,
        'grad_loss': avg_grad,
        'dice_loss': avg_dice
    }


def validate(model, val_loader, device, epoch, args):
    """使用 Dice 指标验证模型（与 DGMIR 相同）"""
    model.eval()
    
    # 用于分割的空间变换器
    stn = SpatialTransformer2().to(device)
    
    dice_list = []
    
    with torch.no_grad():
        for fixed, moving, fixed_seg, moving_seg in val_loader:
            # 移动到设备（与 DGMIR 一致）
            fixed = fixed.to(device)  # (B, D, H, W)
            moving = moving.to(device)  # (B, D, H, W)
            fixed_seg = fixed_seg.to(device)  # (B, D, H, W)
            moving_seg = moving_seg.to(device)  # (B, D, H, W)
            
            # 添加 channel 维度（与 DGMIR 一致）
            fixed = torch.unsqueeze(fixed, dim=1)  # (B, 1, D, H, W)
            moving = torch.unsqueeze(moving, dim=1)  # (B, 1, D, H, W)
            fixed_seg = torch.unsqueeze(fixed_seg, dim=1)  # (B, 1, D, H, W)
            moving_seg = torch.unsqueeze(moving_seg, dim=1)  # (B, 1, D, H, W)
            
            # 拼接输入
            x_in = torch.cat([moving, fixed], dim=1)
            
            # 前向传播
            x_def, flow = model(x_in)
            
            # 使用 SpatialTransformer2 变换分割
            def_seg = stn(moving_seg, flow)
            
            # 使用 dicegup 计算 Dice（与 DGMIR 验证相同）
            dice_score = dicegup(def_seg, fixed_seg, num_classes=args.cls_num)
            dice_list.append(dice_score.item())
    
    # 平均 Dice
    avg_dice = np.mean(dice_list)
    print(f'[验证] Epoch {epoch} - Dice: {avg_dice:.4f}')
    
    return avg_dice


def main():
    # 参数设置
    parser = argparse.ArgumentParser(description='TransMorph 训练 - DGMIR 对齐')
    parser.add_argument('--data_root', type=str, default='../dataset', help='数据集根目录')
    parser.add_argument('--output_dir', type=str, default='./TransMorph_output', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小（与 DGMIR 相同）')
    parser.add_argument('--n_epoch', type=int, default=300, help='训练轮数（与 DGMIR 相同）')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率（与 DGMIR 相同）')
    parser.add_argument('--img_size', type=int, nargs=3, default=[192, 160, 192], help='图像尺寸 (D, H, W)')
    parser.add_argument('--cls_num', type=int, default=5, help='类别数量 (0-4)')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--resume', type=str, default=None, help='从检查点恢复')
    
    args = parser.parse_args()
    
    # 转换为绝对路径（相对于脚本所在目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    args.data_root = os.path.abspath(os.path.join(script_dir, args.data_root))
    args.output_dir = os.path.abspath(os.path.join(script_dir, args.output_dir))
    
    # 设置 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    print(f'数据集路径: {args.data_root}')
    print(f'输出路径: {args.output_dir}')
    
    # 创建输出目录
    model_dir = make_dirs(args.output_dir)
    
    # 获取 TransMorph 配置并修改为自定义数据
    config = get_3DTransMorph_config()
    config.img_size = tuple(args.img_size)
    
    # 调整 window_size 使其能被 img_size 整除
    # 原始: (5, 6, 7) 适用于 (160, 192, 224)
    # 自定义: (6, 5, 6) 适用于 (192, 160, 192)
    # 192/6=32, 160/5=32, 192/6=32 ✓
    config.window_size = (6, 5, 6)
    
    print(f'配置: img_size={config.img_size}, window_size={config.window_size}')
    
    # 创建模型
    model = TransMorph(config)
    model = model.to(device)
    
    print(f'模型已创建，可训练参数: {count_parameters(model):,}')
    
    # 创建数据集（无数据增强 - 与 DGMIR 对齐）
    train_dataset = CustomDataset(
        data_root=args.data_root,
        split='train',
        vol_size=tuple(args.img_size)
    )
    
    val_dataset = CustomDataset(
        data_root=args.data_root,
        split='val',
        vol_size=tuple(args.img_size)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f'训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}')
    
    # 优化器（与 DGMIR 相同: Adam with lr=1e-4）
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 初始化训练历史记录（用于后续画图）
    history = {
        'epochs': [],           # epoch 编号
        'train_total_loss': [], # 训练总损失
        'train_mind_loss': [],  # MIND 损失
        'train_grad_loss': [],  # 梯度损失
        'train_dice_loss': [],  # Dice 损失
        'val_dice': []          # 验证 Dice
    }
    
    # 如果指定则从检查点恢复
    start_epoch = 1
    best_dice = 0.0
    
    if args.resume:
        if os.path.exists(args.resume):
            print(f'从检查点恢复: {args.resume}')
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_dice = checkpoint.get('best_dice', 0.0)
            # 恢复训练历史
            if 'history' in checkpoint:
                history = checkpoint['history']
                print(f'已恢复 {len(history["epochs"])} 个 epoch 的训练历史')
            print(f'从 epoch {start_epoch-1} 恢复, 最佳 Dice: {best_dice:.4f}')
        else:
            print(f'检查点未找到: {args.resume}')
    
    # 训练循环
    print(f'\n=== 开始训练 ===')
    print(f'Epoch 范围: {start_epoch} 到 {args.n_epoch}')
    print(f'批次大小: {args.batch_size}')
    print(f'学习率: {args.lr}')
    print(f'损失权重: [MIND=1.0, Grad=0.5, Dice=0.5]')
    print(f'验证指标: Dice (dicegup)\n')
    
    for epoch in range(start_epoch, args.n_epoch + 1):
        # 训练
        train_losses = train_epoch(model, train_loader, optimizer, device, epoch, args)
        
        # 验证
        val_dice = validate(model, val_loader, device, epoch, args)
        
        # 记录训练历史
        history['epochs'].append(epoch)
        history['train_total_loss'].append(train_losses['total_loss'])
        history['train_mind_loss'].append(train_losses['mind_loss'])
        history['train_grad_loss'].append(train_losses['grad_loss'])
        history['train_dice_loss'].append(train_losses['dice_loss'])
        history['val_dice'].append(val_dice)
        
        # 保存最佳模型
        if val_dice > best_dice:
            best_dice = val_dice
            save_path = os.path.join(model_dir, 'TransMorph_best.pth')
            save_checkpoint(model, optimizer, epoch, best_dice, save_path, history)
            print(f'*** 新的最佳 Dice: {best_dice:.4f} ***')
        
        # 每 50 个 epoch 保存检查点
        if epoch % 50 == 0:
            save_path = os.path.join(model_dir, f'TransMorph_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, best_dice, save_path, history)
        
        print('-' * 80)
    
    # 保存最终模型
    final_path = os.path.join(model_dir, 'TransMorph_final.pth')
    save_checkpoint(model, optimizer, args.n_epoch, best_dice, final_path, history)
    
    print(f'\n=== 训练完成 ===')
    print(f'最佳 Dice: {best_dice:.4f}')
    print(f'训练历史已保存 {len(history["epochs"])} 个 epoch 的数据')
    print(f'最终模型已保存至: {final_path}')


if __name__ == '__main__':
    main()
