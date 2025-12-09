"""
工具函数模块
提供训练过程中需要的辅助函数
"""

import os
import json
import random
import numpy as np
import torch
from typing import Dict, Any
import matplotlib.pyplot as plt


def save_checkpoint(state: Dict[str, Any], save_dir: str, filename: str = 'checkpoint.pth') -> None:
    """
    保存模型检查点
    
    Args:
        state: 包含模型状态的字典
        save_dir: 保存目录
        filename: 文件名
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    print(f"检查点已保存到: {filepath}")


def load_checkpoint(filepath: str, model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer = None) -> Dict[str, Any]:
    """
    加载模型检查点
    
    Args:
        filepath: 检查点文件路径
        model: 模型实例
        optimizer: 优化器实例（可选）
    
    Returns:
        包含检查点信息的字典
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"检查点已从 {filepath} 加载")
    return checkpoint


def save_training_history(history: Dict[str, list], save_dir: str, 
                         filename: str = 'training_history.json') -> None:
    """
    保存训练历史记录
    
    Args:
        history: 训练历史字典
        save_dir: 保存目录
        filename: 文件名
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4, ensure_ascii=False)
    print(f"训练历史已保存到: {filepath}")


def load_training_history(filepath: str) -> Dict[str, list]:
    """
    加载训练历史记录
    
    Args:
        filepath: 训练历史文件路径
    
    Returns:
        训练历史字典
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        history = json.load(f)
    print(f"训练历史已从 {filepath} 加载")
    return history


def get_device() -> torch.device:
    """
    获取可用的设备（CUDA 或 CPU）
    
    Returns:
        torch.device: 可用的设备
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """
    计算模型的参数数量
    
    Args:
        model: PyTorch 模型
    
    Returns:
        参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    """计算并存储平均值和当前值"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置所有统计数据"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        更新统计数据
        
        Args:
            val: 当前值
            n: 样本数量
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed: int = 42) -> None:
    """
    设置随机种子以确保可重复性
    
    Args:
        seed: 随机种子值
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 确保卷积算法的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
