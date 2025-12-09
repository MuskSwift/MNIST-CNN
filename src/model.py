"""
CNN 模型定义模块
包含带 Batch Normalization 和不带 Batch Normalization 的两种 CNN 结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CNN_WithoutBN(nn.Module):
    """
    不带 Batch Normalization 的 CNN 模型
    
    结构: Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> FC
    """
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        """
        Args:
            num_classes: 分类类别数
            dropout_rate: Dropout 比率
        """
        super(CNN_WithoutBN, self).__init__()
        
        # 第一个卷积块
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二个卷积块
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, 1, 28, 28)
        
        Returns:
            输出张量 (batch_size, num_classes)
        """
        # 第一个卷积块: 28x28 -> 14x14
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # 第二个卷积块: 14x14 -> 7x7
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class CNN_WithBN(nn.Module):
    """
    带 Batch Normalization 的 CNN 模型
    
    结构: Conv -> BN -> ReLU -> Pool -> Conv -> BN -> ReLU -> Pool -> FC
    """
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        """
        Args:
            num_classes: 分类类别数
            dropout_rate: Dropout 比率
        """
        super(CNN_WithBN, self).__init__()
        
        # 第一个卷积块
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二个卷积块
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, 1, 28, 28)
        
        Returns:
            输出张量 (batch_size, num_classes)
        """
        # 第一个卷积块: 28x28 -> 14x14
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # 第二个卷积块: 14x14 -> 7x7
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def get_model(model_type: str = 'with_bn', num_classes: int = 10, 
              dropout_rate: float = 0.5) -> nn.Module:
    """
    获取指定类型的模型
    
    Args:
        model_type: 模型类型 ('with_bn' 或 'without_bn')
        num_classes: 分类类别数
        dropout_rate: Dropout 比率
    
    Returns:
        模型实例
    """
    if model_type == 'with_bn':
        model = CNN_WithBN(num_classes=num_classes, dropout_rate=dropout_rate)
        print("创建带 Batch Normalization 的 CNN 模型")
    elif model_type == 'without_bn':
        model = CNN_WithoutBN(num_classes=num_classes, dropout_rate=dropout_rate)
        print("创建不带 Batch Normalization 的 CNN 模型")
    else:
        raise ValueError(f"未知的模型类型: {model_type}. 请使用 'with_bn' 或 'without_bn'")
    
    return model


if __name__ == "__main__":
    # 测试模型
    print("=" * 50)
    print("测试不带 BN 的模型:")
    model1 = CNN_WithoutBN()
    x = torch.randn(2, 1, 28, 28)
    output1 = model1(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output1.shape}")
    
    print("\n" + "=" * 50)
    print("测试带 BN 的模型:")
    model2 = CNN_WithBN()
    output2 = model2(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output2.shape}")
    
    # 计算参数数量
    params1 = sum(p.numel() for p in model1.parameters())
    params2 = sum(p.numel() for p in model2.parameters())
    print("\n" + "=" * 50)
    print(f"不带 BN 模型参数数量: {params1:,}")
    print(f"带 BN 模型参数数量: {params2:,}")
