"""
可视化脚本
绘制训练曲线、模型结构图、混淆矩阵和错误样本
"""

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
import torch

from model import get_model
from utils import load_training_history


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MNIST CNN 可视化脚本')
    
    parser.add_argument('--result_dir', type=str, required=True,
                       help='结果目录路径（包含训练历史和评估结果）')
    parser.add_argument('--model', type=str, default='with_bn',
                       choices=['with_bn', 'without_bn'],
                       help='模型类型: with_bn 或 without_bn')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='可视化结果保存目录 (默认: 与 result_dir 相同)')
    
    return parser.parse_args()


def plot_training_curves(history: Dict, save_path: str):
    """
    绘制训练曲线（损失和准确率）
    
    Args:
        history: 训练历史字典
        save_path: 保存路径
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 损失曲线
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train Accuracy', linewidth=2, markersize=4)
    ax2.plot(epochs, history['val_acc'], 'r-s', label='Val Accuracy', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存到: {save_path}")
    plt.close()


def plot_confusion_matrix(cm_path: str, save_path: str):
    """
    可视化混淆矩阵
    
    Args:
        cm_path: 混淆矩阵文件路径
        save_path: 保存路径
    """
    cm = np.load(cm_path)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10),
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵可视化已保存到: {save_path}")
    plt.close()
    
    # 绘制归一化的混淆矩阵
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10),
                cbar_kws={'label': 'Proportion'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    normalized_path = save_path.replace('.png', '_normalized.png')
    plt.savefig(normalized_path, dpi=300, bbox_inches='tight')
    print(f"归一化混淆矩阵已保存到: {normalized_path}")
    plt.close()


def plot_error_samples(error_analysis_path: str, save_path: str, num_samples: int = 20):
    """
    可视化错误分类样本
    
    Args:
        error_analysis_path: 错误分析文件路径
        save_path: 保存路径
        num_samples: 显示的样本数量
    """
    # 加载错误信息
    with open(error_analysis_path, 'r', encoding='utf-8') as f:
        error_info = json.load(f)
    
    # 加载图像数据
    image_path = error_analysis_path.replace('.json', '_images.npy')
    if not os.path.exists(image_path):
        print(f"错误样本图像文件不存在: {image_path}")
        return
    
    images = np.load(image_path)
    
    # 限制显示数量
    num_samples = min(num_samples, len(error_info))
    
    # 计算网格大小
    cols = 5
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten() if rows * cols > 1 else [axes]
    
    for idx in range(num_samples):
        ax = axes[idx]
        
        # 获取图像（去除通道维度并反归一化）
        img = images[idx].squeeze()
        # 反归一化: img = img * std + mean
        img = img * 0.3081 + 0.1307
        img = np.clip(img, 0, 1)
        
        # 显示图像
        ax.imshow(img, cmap='gray')
        
        # 设置标题
        true_label = error_info[idx]['true_label']
        pred_label = error_info[idx]['pred_label']
        confidence = error_info[idx]['confidence']
        ax.set_title(f'True: {true_label}, Pred: {pred_label}\nConf: {confidence:.2f}',
                    fontsize=10)
        ax.axis('off')
    
    # 隐藏多余的子图
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"错误样本可视化已保存到: {save_path}")
    plt.close()


def plot_model_structure(model_type: str, save_path: str):
    """
    可视化模型结构
    
    Args:
        model_type: 模型类型
        save_path: 保存路径
    """
    try:
        from torchinfo import summary
        
        model = get_model(model_type=model_type)
        
        # 获取模型摘要
        model_stats = summary(model, input_size=(1, 1, 28, 28), 
                            verbose=0,
                            col_names=["input_size", "output_size", "num_params", "kernel_size"])
        
        # 保存为文本文件
        txt_path = save_path.replace('.png', '.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(str(model_stats))
        
        print(f"模型结构信息已保存到: {txt_path}")
        
        # 创建简单的架构图
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        
        # 模型层信息
        layer_info = []
        layer_info.append("Input: 1x28x28")
        layer_info.append("↓")
        layer_info.append("Conv2d (1→32, 3x3)")
        if model_type == 'with_bn':
            layer_info.append("BatchNorm2d (32)")
        layer_info.append("ReLU")
        layer_info.append("MaxPool2d (2x2)")
        layer_info.append("↓ 32x14x14")
        layer_info.append("")
        layer_info.append("Conv2d (32→64, 3x3)")
        if model_type == 'with_bn':
            layer_info.append("BatchNorm2d (64)")
        layer_info.append("ReLU")
        layer_info.append("MaxPool2d (2x2)")
        layer_info.append("↓ 64x7x7")
        layer_info.append("")
        layer_info.append("Flatten → 3136")
        layer_info.append("↓")
        layer_info.append("Linear (3136→128)")
        if model_type == 'with_bn':
            layer_info.append("BatchNorm1d (128)")
        layer_info.append("ReLU")
        layer_info.append("Dropout (0.5)")
        layer_info.append("↓")
        layer_info.append("Linear (128→10)")
        layer_info.append("↓")
        layer_info.append("Output: 10 classes")
        
        # 显示文本
        y_position = 0.95
        for line in layer_info:
            fontsize = 12
            fontweight = 'bold' if line.startswith(('Input', 'Output', 'Conv', 'Linear')) else 'normal'
            color = 'blue' if line.startswith(('Conv', 'Linear')) else 'black'
            color = 'green' if 'BatchNorm' in line else color
            
            ax.text(0.5, y_position, line, 
                   ha='center', va='top', fontsize=fontsize,
                   fontweight=fontweight, color=color,
                   transform=ax.transAxes)
            y_position -= 0.035
        
        title = f"CNN Architecture ({'With' if model_type == 'with_bn' else 'Without'} Batch Normalization)"
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"模型架构图已保存到: {save_path}")
        plt.close()
        
    except ImportError:
        print("警告: torchinfo 未安装，跳过模型结构可视化")
        print("可以使用 'pip install torchinfo' 安装")


def plot_performance_comparison(result_dirs: list, labels: list, save_path: str):
    """
    比较多个实验的性能
    
    Args:
        result_dirs: 结果目录列表
        labels: 实验标签列表
        save_path: 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for result_dir, label in zip(result_dirs, labels):
        history_path = os.path.join(result_dir, 'training_history.json')
        if not os.path.exists(history_path):
            print(f"警告: {history_path} 不存在，跳过")
            continue
        
        history = load_training_history(history_path)
        epochs = range(1, len(history['train_loss']) + 1)
        
        # 绘制损失
        ax1.plot(epochs, history['val_loss'], '-o', label=label, linewidth=2, markersize=4)
        # 绘制准确率
        ax2.plot(epochs, history['val_acc'], '-o', label=label, linewidth=2, markersize=4)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Validation Loss', fontsize=12)
    ax1.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Accuracy', fontsize=12)
    ax2.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"性能比较图已保存到: {save_path}")
    plt.close()


def main():
    """主可视化函数"""
    args = parse_args()
    
    # 确定保存目录
    if args.save_dir is None:
        args.save_dir = args.result_dir
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 80)
    print("可视化配置:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("=" * 80)
    
    # 绘制训练曲线
    history_path = os.path.join(args.result_dir, 'training_history.json')
    if os.path.exists(history_path):
        history = load_training_history(history_path)
        curves_path = os.path.join(args.save_dir, 'training_curves.png')
        plot_training_curves(history, curves_path)
    else:
        print(f"警告: 训练历史文件不存在: {history_path}")
    
    # 绘制混淆矩阵
    cm_path = os.path.join(args.result_dir, 'confusion_matrix.npy')
    if os.path.exists(cm_path):
        cm_vis_path = os.path.join(args.save_dir, 'confusion_matrix.png')
        plot_confusion_matrix(cm_path, cm_vis_path)
    else:
        print(f"警告: 混淆矩阵文件不存在: {cm_path}")
    
    # 绘制错误样本
    error_path = os.path.join(args.result_dir, 'error_analysis.json')
    if os.path.exists(error_path):
        error_vis_path = os.path.join(args.save_dir, 'error_samples.png')
        plot_error_samples(error_path, error_vis_path)
    else:
        print(f"警告: 错误分析文件不存在: {error_path}")
    
    # 绘制模型结构
    model_path = os.path.join(args.save_dir, 'model_architecture.png')
    plot_model_structure(args.model, model_path)
    
    print("\n可视化完成!")


if __name__ == '__main__':
    main()
