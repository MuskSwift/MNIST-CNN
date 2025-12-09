"""
评估脚本
计算模型在测试集上的准确率、混淆矩阵和错误样本分析
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from typing import Tuple, List
import json

from model import get_model
from dataset import get_test_loader
from utils import load_checkpoint, get_device


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MNIST CNN 评估脚本')
    
    # 模型参数
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--model', type=str, default='with_bn',
                       choices=['with_bn', 'without_bn'],
                       help='模型类型: with_bn 或 without_bn')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='数据存储目录 (默认: ./data)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小 (默认: 64)')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='数据加载工作进程数 (默认: 2)')
    
    # 保存参数
    parser.add_argument('--save_dir', type=str, default=None,
                       help='结果保存目录 (默认: 与检查点相同目录)')
    
    return parser.parse_args()


def evaluate_model(model: nn.Module, test_loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    评估模型性能
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
    
    Returns:
        真实标签、预测标签和错误样本列表
    """
    model.eval()
    
    all_targets = []
    all_predictions = []
    error_samples = []  # 存储错误样本 (image, true_label, pred_label, confidence)
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output = model(data)
            pred = output.argmax(dim=1)
            
            # 获取预测概率
            probabilities = torch.softmax(output, dim=1)
            
            # 记录所有预测和真实标签
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(pred.cpu().numpy())
            
            # 找出错误样本
            mask = pred != target
            if mask.any():
                error_indices = torch.where(mask)[0]
                for idx in error_indices:
                    error_samples.append({
                        'image': data[idx].cpu().numpy(),
                        'true_label': target[idx].item(),
                        'pred_label': pred[idx].item(),
                        'confidence': probabilities[idx][pred[idx]].item()
                    })
    
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    
    return all_targets, all_predictions, error_samples


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
    """
    计算并保存混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        save_path: 保存路径
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # 保存为 numpy 数组
    np.save(save_path, cm)
    print(f"混淆矩阵已保存到: {save_path}")
    
    # 打印混淆矩阵
    print("\n混淆矩阵:")
    print(cm)
    
    return cm


def save_classification_report(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
    """
    生成并保存分类报告
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        save_path: 保存路径
    """
    report = classification_report(y_true, y_pred, 
                                   target_names=[str(i) for i in range(10)],
                                   digits=4)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("分类报告\n")
        f.write("=" * 80 + "\n")
        f.write(report)
    
    print(f"分类报告已保存到: {save_path}")
    print("\n" + report)


def save_error_analysis(error_samples: List, save_path: str, max_samples: int = 100):
    """
    保存错误样本分析
    
    Args:
        error_samples: 错误样本列表
        save_path: 保存路径
        max_samples: 最多保存的样本数
    """
    # 限制保存的样本数量
    if len(error_samples) > max_samples:
        error_samples = error_samples[:max_samples]
    
    # 准备保存数据（不包括图像数据，图像单独保存）
    error_info = []
    for i, sample in enumerate(error_samples):
        error_info.append({
            'index': i,
            'true_label': int(sample['true_label']),
            'pred_label': int(sample['pred_label']),
            'confidence': float(sample['confidence'])
        })
    
    # 保存错误信息
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(error_info, f, indent=4, ensure_ascii=False)
    
    print(f"错误样本分析已保存到: {save_path}")
    print(f"总共 {len(error_samples)} 个错误样本")
    
    # 保存图像数据
    images = np.array([sample['image'] for sample in error_samples])
    image_path = save_path.replace('.json', '_images.npy')
    np.save(image_path, images)
    print(f"错误样本图像已保存到: {image_path}")
    
    # 统计错误类型
    error_types = {}
    for sample in error_samples:
        key = (sample['true_label'], sample['pred_label'])
        error_types[key] = error_types.get(key, 0) + 1
    
    print("\n最常见的错误类型 (真实标签 -> 预测标签):")
    sorted_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)
    for (true_label, pred_label), count in sorted_errors[:10]:
        print(f"  {true_label} -> {pred_label}: {count} 次")


def main():
    """主评估函数"""
    args = parse_args()
    
    # 确定保存目录
    if args.save_dir is None:
        args.save_dir = os.path.dirname(args.checkpoint)
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 80)
    print("评估配置:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("=" * 80)
    
    # 获取设备
    device = get_device()
    
    # 加载测试数据
    test_loader = get_test_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # 创建模型
    model = get_model(model_type=args.model, num_classes=10).to(device)
    
    # 加载检查点
    checkpoint = load_checkpoint(args.checkpoint, model)
    
    if 'val_acc' in checkpoint:
        print(f"检查点验证准确率: {checkpoint['val_acc']:.4f}")
    if 'epoch' in checkpoint:
        print(f"检查点训练轮数: {checkpoint['epoch']}")
    
    # 评估模型
    print("\n开始评估...")
    y_true, y_pred, error_samples = evaluate_model(model, test_loader, device)
    
    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n测试集准确率: {accuracy:.4f}")
    
    # 保存混淆矩阵
    cm_path = os.path.join(args.save_dir, 'confusion_matrix.npy')
    save_confusion_matrix(y_true, y_pred, cm_path)
    
    # 保存分类报告
    report_path = os.path.join(args.save_dir, 'classification_report.txt')
    save_classification_report(y_true, y_pred, report_path)
    
    # 保存错误样本分析
    error_path = os.path.join(args.save_dir, 'error_analysis.json')
    save_error_analysis(error_samples, error_path)
    
    # 保存评估结果摘要
    summary = {
        'accuracy': float(accuracy),
        'total_samples': len(y_true),
        'correct_predictions': int((y_true == y_pred).sum()),
        'error_samples': len(error_samples),
        'checkpoint': args.checkpoint
    }
    
    summary_path = os.path.join(args.save_dir, 'evaluation_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    print(f"\n评估摘要已保存到: {summary_path}")
    print("\n评估完成!")


if __name__ == '__main__':
    main()
