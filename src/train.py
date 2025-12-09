"""
训练脚本
支持不同的模型类型、优化器和学习率配置
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
import time

from model import get_model
from dataset import get_mnist_loaders
from utils import (
    save_checkpoint, 
    save_training_history, 
    get_device, 
    count_parameters,
    AverageMeter,
    set_seed
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MNIST CNN 训练脚本')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='with_bn',
                       choices=['with_bn', 'without_bn'],
                       help='模型类型: with_bn 或 without_bn')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout 比率 (默认: 0.5)')
    
    # 优化器参数
    parser.add_argument('--optimizer', type=str, default='Adam',
                       choices=['SGD', 'Adam', 'RMSprop'],
                       help='优化器类型 (默认: Adam)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率 (默认: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD 动量 (默认: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                       help='权重衰减 (默认: 0.0001)')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=10,
                       help='训练轮数 (默认: 10)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小 (默认: 64)')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='验证集划分比例 (默认: 0.1)')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='数据存储目录 (默认: ./data)')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='数据加载工作进程数 (默认: 2)')
    
    # 保存参数
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='结果保存目录 (默认: ./results)')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='实验名称 (默认: 根据参数自动生成)')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42)')
    parser.add_argument('--print_freq', type=int, default=100,
                       help='打印频率 (默认: 100)')
    
    return parser.parse_args()


def get_optimizer(model: nn.Module, optimizer_name: str, lr: float, 
                 momentum: float = 0.9, weight_decay: float = 0.0001) -> optim.Optimizer:
    """
    获取优化器
    
    Args:
        model: 模型
        optimizer_name: 优化器名称
        lr: 学习率
        momentum: 动量（仅用于 SGD）
        weight_decay: 权重衰减
    
    Returns:
        优化器实例
    """
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, 
                            momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"未知的优化器: {optimizer_name}")
    
    print(f"优化器: {optimizer_name}, 学习率: {lr}")
    return optimizer


def train_epoch(model: nn.Module, train_loader, criterion: nn.Module,
               optimizer: optim.Optimizer, device: torch.device,
               epoch: int, print_freq: int = 100) -> Tuple[float, float]:
    """
    训练一个 epoch
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        epoch: 当前 epoch
        print_freq: 打印频率
    
    Returns:
        平均损失和准确率
    """
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 计算准确率
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        accuracy = correct / data.size(0)
        
        # 更新统计
        losses.update(loss.item(), data.size(0))
        accuracies.update(accuracy, data.size(0))
        
        # 打印进度
        if batch_idx % print_freq == 0:
            print(f'Epoch: [{epoch}][{batch_idx}/{len(train_loader)}]\t'
                  f'Loss: {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Acc: {accuracies.val:.4f} ({accuracies.avg:.4f})')
    
    return losses.avg, accuracies.avg


def validate(model: nn.Module, val_loader, criterion: nn.Module,
            device: torch.device) -> Tuple[float, float]:
    """
    在验证集上评估模型
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备
    
    Returns:
        平均损失和准确率
    """
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output = model(data)
            loss = criterion(output, target)
            
            # 计算准确率
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy = correct / data.size(0)
            
            # 更新统计
            losses.update(loss.item(), data.size(0))
            accuracies.update(accuracy, data.size(0))
    
    return losses.avg, accuracies.avg


def main():
    """主训练函数"""
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建实验名称
    if args.experiment_name is None:
        args.experiment_name = f"{args.model}_{args.optimizer}_lr{args.lr}"
    
    # 创建保存目录
    experiment_dir = os.path.join(args.save_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 打印配置
    print("=" * 80)
    print("训练配置:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("=" * 80)
    
    # 获取设备
    device = get_device()
    
    # 加载数据
    train_loader, val_loader, test_loader = get_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    # 创建模型
    model = get_model(
        model_type=args.model,
        num_classes=10,
        dropout_rate=args.dropout
    ).to(device)
    
    # 打印模型信息
    num_params = count_parameters(model)
    print(f"模型参数数量: {num_params:,}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(
        model, 
        args.optimizer, 
        args.lr, 
        args.momentum, 
        args.weight_decay
    )
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_time': []
    }
    
    best_val_acc = 0.0
    
    # 训练循环
    print("\n开始训练...")
    print("=" * 80)
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, 
            device, epoch, args.print_freq
        )
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_time'].append(epoch_time)
        
        # 打印摘要
        print(f'\nEpoch {epoch} 摘要:')
        print(f'  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}')
        print(f'  验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}')
        print(f'  用时: {epoch_time:.2f}s')
        print("=" * 80)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'args': vars(args)
            }
            save_checkpoint(checkpoint, experiment_dir, 'best_model.pth')
    
    # 保存最终模型
    final_checkpoint = {
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': history['train_loss'][-1],
        'train_acc': history['train_acc'][-1],
        'val_loss': history['val_loss'][-1],
        'val_acc': history['val_acc'][-1],
        'args': vars(args)
    }
    save_checkpoint(final_checkpoint, experiment_dir, 'final_model.pth')
    
    # 保存训练历史
    save_training_history(history, experiment_dir)
    
    print(f"\n训练完成!")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    print(f"结果保存在: {experiment_dir}")


if __name__ == '__main__':
    main()
