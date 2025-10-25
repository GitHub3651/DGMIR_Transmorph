"""
Testing Script for TransMorph - Aligned with DGMIR Evaluation
Author: GitHub Copilot
Description: Test TransMorph with same metrics as DGMIR (Dice, HD95, NJD)
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# 获取脚本所在目录并添加 TransMorph_Core 到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'TransMorph_Core'))

# Import TransMorph components
from TransMorph_Core.models.TransMorph import TransMorph
from TransMorph_Core.models.configs_TransMorph import get_3DTransMorph_config

# Import custom dataset
from TransMorph_Core.data.custom_dataset import CustomDataset

# Import DGMIR metrics and functions (now in TransMorph_Core)
from TransMorph_Core.losses import dicegup
from TransMorph_Core.functions import SpatialTransformer2
from TransMorph_Core.metric import jacobian_determinant_gpu, compute_surface_distances, compute_robust_hausdorff


def test_model(model, test_loader, device, args):
    """
    Test model with 4 metrics (same as DGMIR):
    1. Dice (dicegup)
    2. HD95 (Hausdorff Distance 95th percentile)
    3. NJD_mean (negative Jacobian determinant mean)
    4. NJD_std (negative Jacobian determinant std)
    """
    model.eval()
    
    # Spatial transformer for segmentation
    stn = SpatialTransformer2().to(device)
    
    dice_list = []
    hd95_list = []
    njd_mean_list = []
    njd_std_list = []
    
    print('\n=== Testing ===')
    
    with torch.no_grad():
        for idx, (fixed, moving, fixed_seg, moving_seg) in enumerate(tqdm(test_loader, desc='Testing')):
            # 移动到设备（与 DGMIR 一致）
            fixed = fixed.to(device)  # (1, D, H, W)
            moving = moving.to(device)  # (1, D, H, W)
            fixed_seg = fixed_seg.to(device)  # (1, D, H, W)
            moving_seg = moving_seg.to(device)  # (1, D, H, W)
            
            # 添加 channel 维度（与 DGMIR 一致）
            fixed = torch.unsqueeze(fixed, dim=1)  # (1, 1, D, H, W)
            moving = torch.unsqueeze(moving, dim=1)  # (1, 1, D, H, W)
            fixed_seg = torch.unsqueeze(fixed_seg, dim=1)  # (1, 1, D, H, W)
            moving_seg = torch.unsqueeze(moving_seg, dim=1)  # (1, 1, D, H, W)
            
            # 拼接输入
            x_in = torch.cat([moving, fixed], dim=1)  # (1, 2, D, H, W)
            
            # 前向传播
            x_def, flow = model(x_in)
            
            # 使用 SpatialTransformer2 变换分割（与 DGMIR 相同）
            def_seg = stn(moving_seg, flow)
            
            # 1. Dice Score (using dicegup - same as DGMIR)
            dice_score = dicegup(def_seg[0, 0], fixed_seg[0, 0], num_classes=args.cls_num)
            dice_list.append(dice_score.item())
            
            # 2. Negative Jacobian Determinant (NJD)
            # Permute flow to (B, 1, W, H, D) for jacobian computation
            njd = jacobian_determinant_gpu(flow.permute(0, 1, 4, 3, 2)).cpu().numpy()
            njd_mean = np.mean(njd < 0)  # Percentage of negative Jacobian
            njd_std = np.std(njd)
            njd_mean_list.append(njd_mean)
            njd_std_list.append(njd_std)
            
            # 3. Hausdorff Distance 95th percentile (HD95)
            warped_seg = def_seg[0, 0].cpu().numpy()
            fixed_seg_np = fixed_seg[0, 0].cpu().numpy()
            
            # Compute HD95 for each class (excluding background)
            hd95_values = []
            for class_idx in range(1, args.cls_num):
                fixed_mask = (fixed_seg_np == class_idx)
                warped_mask = (warped_seg == class_idx)
                
                # Skip if either mask is empty
                if fixed_mask.sum() == 0 or warped_mask.sum() == 0:
                    continue
                
                # Compute surface distances
                surface_distances = compute_surface_distances(
                    fixed_mask, 
                    warped_mask, 
                    np.ones(3)  # spacing in mm
                )
                
                # Compute robust Hausdorff (95th percentile)
                hd95 = compute_robust_hausdorff(surface_distances, 95.0)
                hd95_values.append(hd95)
            
            # Average HD95 across classes
            if len(hd95_values) > 0:
                avg_hd95 = np.mean(hd95_values)
                hd95_list.append(avg_hd95)
            else:
                hd95_list.append(0.0)
            
            # Print per-sample results
            print(f'Sample {idx+1}: Dice={dice_score.item():.4f}, HD95={hd95_list[-1]:.4f}, '
                  f'NJD_mean={njd_mean:.6f}, NJD_std={njd_std:.6f}')
    
    # Compute statistics
    dice_mean = np.mean(dice_list)
    dice_std = np.std(dice_list)
    hd95_mean = np.mean(hd95_list)
    hd95_std = np.std(hd95_list)
    njd_mean_avg = np.mean(njd_mean_list)
    njd_std_avg = np.mean(njd_std_list)
    
    # Print final results (same format as DGMIR)
    print('\n=== Test Results ===')
    print(f'NJD_mean: {njd_mean_avg:.6f}, NJD_std: {njd_std_avg:.6f}, '
          f'HD95: {hd95_mean:.6f}±{hd95_std:.6f}, '
          f'DICE: {dice_mean:.6f}±{dice_std:.6f}')
    
    return dice_mean, hd95_mean, njd_mean_avg, njd_std_avg


def main():
    # Arguments
    parser = argparse.ArgumentParser(description='TransMorph Testing - DGMIR Aligned')
    parser.add_argument('--data_root', type=str, default='../dataset', help='Root directory of dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--img_size', type=int, nargs=3, default=[192, 160, 192], help='Image size (D, H, W)')
    parser.add_argument('--cls_num', type=int, default=5, help='Number of classes (0-4)')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'], help='Test split')
    
    args = parser.parse_args()
    
    # 转换为绝对路径（相对于脚本所在目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    args.data_root = os.path.abspath(os.path.join(script_dir, args.data_root))
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'数据集路径: {args.data_root}')
    
    # Get TransMorph config and modify for custom data
    config = get_3DTransMorph_config()
    config.img_size = tuple(args.img_size)
    config.window_size = (6, 5, 6)  # Adjusted for (192, 160, 192)
    
    print(f'Config: img_size={config.img_size}, window_size={config.window_size}')
    
    # Create model
    model = TransMorph(config)
    model = model.to(device)
    
    # Load checkpoint
    if not os.path.exists(args.model_path):
        print(f'Error: Model checkpoint not found at {args.model_path}')
        return
    
    print(f'Loading model from: {args.model_path}')
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Model loaded (epoch {checkpoint["epoch"]}, best Dice: {checkpoint.get("best_dice", 0.0):.4f})')
    
    # Create test dataset
    test_dataset = CustomDataset(
        data_root=args.data_root,
        split=args.split,
        vol_size=tuple(args.img_size)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f'Test samples: {len(test_dataset)}')
    
    # Test model
    test_model(model, test_loader, device, args)


if __name__ == '__main__':
    main()
