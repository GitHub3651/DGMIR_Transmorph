"""
实际测量 TransMorph 训练显存使用
通过真实运行一次前向+反向传播来测量
"""

import os
import sys
import torch
import torch.nn as nn

# 添加路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'TransMorph_Core'))

from TransMorph_Core.models.TransMorph import TransMorph
from TransMorph_Core.models.configs_TransMorph import get_3DTransMorph_config
from TransMorph_Core.losses import MINDSSCLoss, Grad, compute_per_channel_dice
from TransMorph_Core.functions import SpatialTransformer2


def format_bytes(bytes_val):
    """格式化字节数为 GB"""
    return bytes_val / (1024**3)


def test_memory_usage(img_size=(192, 160, 192), batch_size=1):
    """
    测试实际显存使用
    """
    print("=" * 80)
    print(f"实际显存测试")
    print(f"图像尺寸: {img_size}, Batch size: {batch_size}")
    print("=" * 80)
    print()
    
    if not torch.cuda.is_available():
        print("错误: 没有可用的 CUDA 设备")
        return
    
    device = torch.device('cuda')
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # 记录初始状态
    print("【0. 初始状态】")
    mem_0 = torch.cuda.memory_allocated()
    print(f"  已分配: {format_bytes(mem_0):.3f} GB")
    print()
    
    # 1. 创建模型
    print("【1. 加载模型】")
    config = get_3DTransMorph_config()
    config.img_size = img_size
    config.window_size = (
        img_size[0] // 32,
        img_size[1] // 32, 
        img_size[2] // 32
    )
    
    model = TransMorph(config).to(device)
    stn = SpatialTransformer2().to(device)
    
    mem_1 = torch.cuda.memory_allocated()
    print(f"  模型显存: {format_bytes(mem_1 - mem_0):.3f} GB")
    print(f"  累计显存: {format_bytes(mem_1):.3f} GB")
    print()
    
    # 2. 创建损失函数
    print("【2. 创建损失函数】")
    mind_loss_fn = MINDSSCLoss(radius=2, dilation=2, penalty='l2').to(device)
    grad_loss_fn = Grad(penalty='l2').to(device)
    
    mem_2 = torch.cuda.memory_allocated()
    print(f"  损失函数显存: {format_bytes(mem_2 - mem_1):.3f} GB")
    print(f"  累计显存: {format_bytes(mem_2):.3f} GB")
    print()
    
    # 3. 创建输入数据
    print("【3. 创建输入数据】")
    D, H, W = img_size
    cls_num = 5
    
    # 图像数据
    fixed = torch.randn(batch_size, 1, D, H, W, device=device)
    moving = torch.randn(batch_size, 1, D, H, W, device=device)
    
    # 分割数据 (one-hot)
    fixed_seg = torch.zeros(batch_size, cls_num, D, H, W, device=device)
    moving_seg = torch.zeros(batch_size, cls_num, D, H, W, device=device)
    
    # 随机填充分割
    for b in range(batch_size):
        fixed_seg_idx = torch.randint(0, cls_num, (D, H, W), device=device)
        moving_seg_idx = torch.randint(0, cls_num, (D, H, W), device=device)
        fixed_seg[b] = torch.nn.functional.one_hot(fixed_seg_idx, cls_num).permute(3, 0, 1, 2).float()
        moving_seg[b] = torch.nn.functional.one_hot(moving_seg_idx, cls_num).permute(3, 0, 1, 2).float()
    
    mem_3 = torch.cuda.memory_allocated()
    print(f"  输入数据显存: {format_bytes(mem_3 - mem_2):.3f} GB")
    print(f"  累计显存: {format_bytes(mem_3):.3f} GB")
    print()
    
    # 4. 前向传播
    print("【4. 前向传播】")
    model.train()
    
    # 拼接输入
    x = torch.cat([fixed, moving], dim=1)  # (B, 2, D, H, W)
    
    # 模型前向 (返回 out, flow)
    _, flow = model(x)
    
    mem_4 = torch.cuda.memory_allocated()
    print(f"  前向传播显存增加: {format_bytes(mem_4 - mem_3):.3f} GB")
    print(f"  累计显存: {format_bytes(mem_4):.3f} GB")
    print()
    
    # 5. 计算输出（变形）
    print("【5. 应用变形场】")
    x_def = stn(moving, flow)
    def_seg = stn(moving_seg, flow)
    
    mem_5 = torch.cuda.memory_allocated()
    print(f"  变形显存增加: {format_bytes(mem_5 - mem_4):.3f} GB")
    print(f"  累计显存: {format_bytes(mem_5):.3f} GB")
    print()
    
    # 6. 计算损失
    print("【6. 计算损失】")
    weights = [1.0, 0.5, 0.5]
    
    mind_loss = mind_loss_fn(x_def, fixed)
    grad_loss = grad_loss_fn(flow)
    dice_loss = compute_per_channel_dice(def_seg, fixed_seg, classes=cls_num)
    
    loss = weights[0] * mind_loss + weights[1] * grad_loss + weights[2] * dice_loss
    
    mem_6 = torch.cuda.memory_allocated()
    print(f"  损失计算显存增加: {format_bytes(mem_6 - mem_5):.3f} GB")
    print(f"  累计显存: {format_bytes(mem_6):.3f} GB")
    print()
    
    # 7. 反向传播
    print("【7. 反向传播】")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss.backward()
    
    mem_7 = torch.cuda.memory_allocated()
    peak_mem = torch.cuda.max_memory_allocated()
    
    print(f"  反向传播显存增加: {format_bytes(mem_7 - mem_6):.3f} GB")
    print(f"  当前显存: {format_bytes(mem_7):.3f} GB")
    print(f"  峰值显存: {format_bytes(peak_mem):.3f} GB")
    print()
    
    # 8. 优化器步进
    print("【8. 优化器更新】")
    optimizer.step()
    
    mem_8 = torch.cuda.memory_allocated()
    final_peak = torch.cuda.max_memory_allocated()
    
    print(f"  优化后显存: {format_bytes(mem_8):.3f} GB")
    print(f"  最终峰值: {format_bytes(final_peak):.3f} GB")
    print()
    
    # 总结
    print("=" * 80)
    print("【总结】")
    print(f"  配置: {img_size}, batch_size={batch_size}")
    print(f"  实际峰值显存: {format_bytes(final_peak):.3f} GB")
    print(f"  推荐显存（含余量）: {format_bytes(final_peak * 1.15):.3f} GB")
    print("=" * 80)
    print()
    
    # 清理
    del model, stn, optimizer, loss, flow, x_def, def_seg
    del fixed, moving, fixed_seg, moving_seg, x
    del mind_loss, grad_loss, dice_loss
    torch.cuda.empty_cache()
    
    return final_peak


if __name__ == "__main__":
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 25 + "实际显存测量" + " " * 37 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    # 测试不同配置
    configs = [
        ((96, 96, 96), "96×96×96"),
        ((128, 128, 128), "128×128×128"),
        ((160, 160, 160), "160×160×160"),
        ((192, 160, 192), "192×160×192 (原始)"),
    ]
    
    results = []
    
    for img_size, name in configs:
        try:
            print(f"\n{'=' * 80}")
            print(f"测试配置: {name}")
            print(f"{'=' * 80}\n")
            
            peak = test_memory_usage(img_size=img_size, batch_size=1)
            results.append((name, peak))
            
            # 清理显存
            torch.cuda.empty_cache()
            
            print(f"✅ {name} 测试完成\n")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ {name}: 显存不足 (OOM)\n")
                results.append((name, None))
                torch.cuda.empty_cache()
            else:
                raise e
    
    # 汇总结果
    print("\n" + "=" * 80)
    print("【测试结果汇总】")
    print("=" * 80)
    print(f"{'配置':<25} {'峰值显存':<15} {'推荐显存 (+15%)':<20}")
    print("-" * 80)
    
    for name, peak in results:
        if peak is not None:
            peak_gb = format_bytes(peak)
            recommend_gb = format_bytes(peak * 1.15)
            print(f"{name:<25} {peak_gb:.2f} GB{'':<8} {recommend_gb:.2f} GB")
        else:
            print(f"{name:<25} {'OOM':<15} {'N/A':<20}")
    
    print("=" * 80)
    print()
