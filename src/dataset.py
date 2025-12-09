"""
数据集加载和预处理模块
负责 MNIST 数据集的加载、预处理和划分
"""

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from typing import Tuple, Optional


def get_data_transforms(train: bool = True) -> transforms.Compose:
    """
    获取数据预处理转换
    
    Args:
        train: 是否为训练集（训练集可以添加数据增强）
    
    Returns:
        transforms.Compose: 组合的数据转换
    """
    if train:
        # 训练集数据增强
        transform = transforms.Compose([
            transforms.RandomRotation(10),  # 随机旋转 ±10 度
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST 数据集的均值和标准差
        ])
    else:
        # 测试集只做标准化
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    return transform


def get_mnist_loaders(data_dir: str = './data',
                     batch_size: int = 64,
                     val_split: float = 0.1,
                     num_workers: int = 2,
                     seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    获取 MNIST 数据集的 DataLoader
    
    Args:
        data_dir: 数据存储目录
        batch_size: 批次大小
        val_split: 验证集划分比例
        num_workers: 数据加载的工作进程数
        seed: 随机种子
    
    Returns:
        train_loader, val_loader, test_loader: 训练、验证和测试数据加载器
    """
    # 训练集转换（包含数据增强）
    train_transform = get_data_transforms(train=True)
    # 测试集转换（不含数据增强）
    test_transform = get_data_transforms(train=False)
    
    # 下载并加载训练数据
    full_train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # 加载测试数据
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # 将训练数据划分为训练集和验证集
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    # 使用固定的随机种子确保可重复性
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=generator
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"数据集加载完成:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")
    print(f"  批次大小: {batch_size}")
    
    return train_loader, val_loader, test_loader


def get_test_loader(data_dir: str = './data',
                   batch_size: int = 64,
                   num_workers: int = 2) -> DataLoader:
    """
    仅获取测试集的 DataLoader（用于评估）
    
    Args:
        data_dir: 数据存储目录
        batch_size: 批次大小
        num_workers: 数据加载的工作进程数
    
    Returns:
        test_loader: 测试数据加载器
    """
    test_transform = get_data_transforms(train=False)
    
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"测试集加载完成: {len(test_dataset)} 样本")
    
    return test_loader
