"""
从训练好的模型文件中读取历史数据并画出损失曲线
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_training_curves(model_path, save_path=None):
    """
    从模型文件中读取训练历史并画图
    
    Args:
        model_path: 模型文件路径 (如 'TransMorph_output/models/TransMorph_best.pth')
        save_path: 图片保存路径 (可选，如果不指定则显示图片)
    """
    
    # 加载模型检查点
    print(f'加载模型: {model_path}')
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 检查是否包含训练历史
    if 'history' not in checkpoint:
        print('错误: 模型文件中没有训练历史数据')
        print('提示: 请使用包含训练历史的新模型文件')
        return
    
    history = checkpoint['history']
    
    # 打印基本信息
    print(f'训练信息:')
    print(f'  - 总 Epoch 数: {checkpoint["epoch"]}')
    print(f'  - 最佳 Dice: {checkpoint["best_dice"]:.4f}')
    print(f'  - 历史记录数: {len(history["epochs"])} 个 epoch')
    
    # 提取数据
    epochs = history['epochs']
    train_total_loss = history['train_total_loss']
    train_mind_loss = history['train_mind_loss']
    train_grad_loss = history['train_grad_loss']
    train_dice_loss = history['train_dice_loss']
    val_dice = history['val_dice']
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('TransMorph Training Curves', fontsize=16, fontweight='bold')
    
    # 1. 总损失曲线
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_total_loss, 'b-', linewidth=2, label='Total Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Total Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # 2. 各损失分量
    ax2 = axes[0, 1]
    ax2.plot(epochs, train_mind_loss, 'r-', linewidth=1.5, label='MIND Loss', alpha=0.8)
    ax2.plot(epochs, train_grad_loss, 'g-', linewidth=1.5, label='Grad Loss', alpha=0.8)
    ax2.plot(epochs, train_dice_loss, 'orange', linewidth=1.5, label='Dice Loss', alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Loss Components', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # 3. 验证 Dice (转换为正值显示)
    ax3 = axes[1, 0]
    val_dice_positive = [d for d in val_dice]  # Dice 本身就是正值
    ax3.plot(epochs, val_dice_positive, 'purple', linewidth=2, marker='o', 
             markersize=3, label='Val Dice', alpha=0.8)
    ax3.axhline(y=checkpoint['best_dice'], color='red', linestyle='--', 
                linewidth=1.5, label=f'Best: {checkpoint["best_dice"]:.4f}', alpha=0.7)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Dice Coefficient', fontsize=12)
    ax3.set_title('Validation Dice Score', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    ax3.set_ylim([0, 1])
    
    # 4. 损失统计信息
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # 计算统计信息
    final_total_loss = train_total_loss[-1]
    final_mind_loss = train_mind_loss[-1]
    final_grad_loss = train_grad_loss[-1]
    final_dice_loss = train_dice_loss[-1]
    final_val_dice = val_dice[-1]
    
    min_total_loss = min(train_total_loss)
    max_val_dice = max(val_dice)
    
    stats_text = f"""
    Training Statistics:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Final Epoch: {epochs[-1]}
    
    Final Losses:
      • Total Loss:  {final_total_loss:.6f}
      • MIND Loss:   {final_mind_loss:.6f}
      • Grad Loss:   {final_grad_loss:.6f}
      • Dice Loss:   {final_dice_loss:.6f}
    
    Final Val Dice:  {final_val_dice:.4f}
    Best Val Dice:   {max_val_dice:.4f}
    
    Training Summary:
      • Min Total Loss:  {min_total_loss:.6f}
      • Max Val Dice:    {max_val_dice:.4f}
      • Total Epochs:    {len(epochs)}
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Model: {os.path.basename(model_path)}
    """
    
    ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'图片已保存至: {save_path}')
    else:
        plt.show()
    
    plt.close()


def compare_models(model_paths, labels=None, save_path=None):
    """
    比较多个模型的训练曲线
    
    Args:
        model_paths: 模型文件路径列表
        labels: 每个模型的标签 (可选)
        save_path: 图片保存路径 (可选)
    """
    
    if labels is None:
        labels = [f'Model {i+1}' for i in range(len(model_paths))]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_paths)))
    
    for i, (path, label) in enumerate(zip(model_paths, labels)):
        checkpoint = torch.load(path, map_location='cpu')
        if 'history' not in checkpoint:
            print(f'警告: {path} 没有训练历史，跳过')
            continue
        
        history = checkpoint['history']
        epochs = history['epochs']
        
        # 总损失
        axes[0].plot(epochs, history['train_total_loss'], 
                    color=colors[i], linewidth=2, label=label, alpha=0.8)
        
        # 验证 Dice
        axes[1].plot(epochs, history['val_dice'], 
                    color=colors[i], linewidth=2, label=label, alpha=0.8)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Total Loss', fontsize=12)
    axes[0].set_title('Training Loss Comparison', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Dice Coefficient', fontsize=12)
    axes[1].set_title('Validation Dice Comparison', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'对比图已保存至: {save_path}')
    else:
        plt.show()
    
    plt.close()


if __name__ == '__main__':
    # 示例用法
    
    # 1. 画单个模型的曲线
    model_path = 'TransMorph_Project/TransMorph_output/models/TransMorph_best.pth'
    
    if os.path.exists(model_path):
        # 显示图片
        plot_training_curves(model_path)
        
        # 或保存图片
        # plot_training_curves(model_path, save_path='training_curves.png')
    else:
        print(f'模型文件不存在: {model_path}')
        print('请先训练模型，或修改 model_path 为正确路径')
    
    # 2. 比较多个模型（如果有的话）
    # model_paths = [
    #     'TransMorph_output/models/TransMorph_best.pth',
    #     'TransMorph_output/models/TransMorph_epoch_100.pth',
    #     'TransMorph_output/models/TransMorph_final.pth'
    # ]
    # labels = ['Best Model', 'Epoch 100', 'Final Model']
    # compare_models(model_paths, labels, save_path='model_comparison.png')
