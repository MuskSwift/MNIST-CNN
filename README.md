# MNIST-CNN 手写数字分类项目

使用卷积神经网络（CNN）完成 MNIST 手写数字分类任务，支持多种实验对比和评测。

## 项目简介

本项目实现了两种 CNN 架构来完成 MNIST 手写数字分类任务：
- **不带 Batch Normalization 的 CNN**
- **带 Batch Normalization 的 CNN**

项目支持多种优化器（SGD、Adam、RMSprop）和不同学习率的实验对比，并提供完整的训练、评估和可视化功能。

## 项目结构

```
├── README.md          # 项目说明文档
├── environment.yml    # Conda 环境配置文件
├── src/
│   ├── __init__.py    # 模块初始化
│   ├── model.py       # CNN 模型定义（带 BN 和不带 BN）
│   ├── dataset.py     # 数据加载和预处理
│   ├── train.py       # 训练脚本
│   ├── evaluate.py    # 评估脚本
│   ├── visualize.py   # 可视化脚本
│   └── utils.py       # 工具函数
└── results/           # 存放实验结果（由 .gitignore 排除）
```

## 环境配置

### 使用 Conda（推荐）

1. 创建并激活环境：
```bash
conda env create -f environment.yml
conda activate mnist-cnn
```

2. 验证安装：
```bash
python -c "import torch; print(torch.__version__)"
```

### 手动安装

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn torchinfo
```

### 依赖包版本

- Python 3.8+
- PyTorch >= 1.10
- torchvision >= 0.11
- numpy >= 1.21
- matplotlib >= 3.5
- seaborn >= 0.11
- scikit-learn >= 1.0
- torchinfo >= 1.7

## 数据说明

项目使用 MNIST 数据集，包含 70,000 张 28x28 像素的手写数字图像：
- **训练集**: 60,000 张图像（自动划分为训练集 54,000 和验证集 6,000）
- **测试集**: 10,000 张图像

数据集将自动下载到 `./data` 目录（首次运行时）。

数据预处理包括：
- 归一化：均值 0.1307，标准差 0.3081
- 训练集数据增强：随机旋转 ±10 度

## 使用方法

### 1. 训练模型

#### 基本训练命令

```bash
# 训练带 BN 的模型，使用 Adam 优化器
python src/train.py --model with_bn --optimizer Adam --lr 0.001 --epochs 10

# 训练不带 BN 的模型，使用 SGD 优化器
python src/train.py --model without_bn --optimizer SGD --lr 0.01 --epochs 10
```

#### 完整参数说明

```bash
python src/train.py \
    --model with_bn              # 模型类型: with_bn 或 without_bn
    --optimizer Adam             # 优化器: SGD, Adam, RMSprop
    --lr 0.001                   # 学习率
    --epochs 10                  # 训练轮数
    --batch_size 64              # 批次大小
    --dropout 0.5                # Dropout 比率
    --momentum 0.9               # SGD 动量
    --weight_decay 0.0001        # 权重衰减
    --val_split 0.1              # 验证集划分比例
    --data_dir ./data            # 数据目录
    --save_dir ./results         # 结果保存目录
    --experiment_name my_exp     # 实验名称（可选）
    --seed 42                    # 随机种子
```

训练结果将保存在 `./results/<experiment_name>/` 目录下，包括：
- `best_model.pth`: 验证集上最佳模型
- `final_model.pth`: 最终模型
- `training_history.json`: 训练历史记录

### 2. 评估模型

```bash
# 评估训练好的模型
python src/evaluate.py \
    --checkpoint ./results/with_bn_Adam_lr0.001/best_model.pth \
    --model with_bn \
    --data_dir ./data
```

评估结果包括：
- `confusion_matrix.npy`: 混淆矩阵
- `classification_report.txt`: 分类报告（精确率、召回率、F1 分数）
- `error_analysis.json`: 错误样本分析
- `error_analysis_images.npy`: 错误样本图像
- `evaluation_summary.json`: 评估摘要

### 3. 可视化结果

```bash
# 可视化训练和评估结果
python src/visualize.py \
    --result_dir ./results/with_bn_Adam_lr0.001 \
    --model with_bn
```

生成的可视化包括：
- `training_curves.png`: 训练和验证的损失/准确率曲线
- `confusion_matrix.png`: 混淆矩阵热力图
- `confusion_matrix_normalized.png`: 归一化混淆矩阵
- `error_samples.png`: 错误分类样本展示
- `model_architecture.png`: 模型结构图
- `model_architecture.txt`: 模型结构详细信息

## 实验对比

### 1. Batch Normalization 对比

**目的**: 比较带 BN 和不带 BN 的模型性能

```bash
# 不带 BN
python src/train.py --model without_bn --optimizer Adam --lr 0.001 --epochs 10 \
    --experiment_name without_bn_adam_lr0.001

# 带 BN
python src/train.py --model with_bn --optimizer Adam --lr 0.001 --epochs 10 \
    --experiment_name with_bn_adam_lr0.001
```

**预期结果**:
- 带 BN 的模型收敛更快
- 带 BN 的模型训练更稳定
- 带 BN 的模型可能达到更高的准确率

### 2. 学习率对比

**目的**: 比较不同学习率的影响

```bash
# 学习率 0.0001
python src/train.py --model with_bn --optimizer Adam --lr 0.0001 --epochs 10 \
    --experiment_name with_bn_adam_lr0.0001

# 学习率 0.001
python src/train.py --model with_bn --optimizer Adam --lr 0.001 --epochs 10 \
    --experiment_name with_bn_adam_lr0.001

# 学习率 0.01
python src/train.py --model with_bn --optimizer Adam --lr 0.01 --epochs 10 \
    --experiment_name with_bn_adam_lr0.01
```

**预期结果**:
- 学习率过低：收敛慢
- 学习率适中：收敛快且稳定
- 学习率过高：可能不收敛或震荡

### 3. 优化器对比

**目的**: 比较不同优化器的性能

```bash
# SGD
python src/train.py --model with_bn --optimizer SGD --lr 0.01 --momentum 0.9 --epochs 10 \
    --experiment_name with_bn_sgd_lr0.01

# Adam
python src/train.py --model with_bn --optimizer Adam --lr 0.001 --epochs 10 \
    --experiment_name with_bn_adam_lr0.001

# RMSprop
python src/train.py --model with_bn --optimizer RMSprop --lr 0.001 --epochs 10 \
    --experiment_name with_bn_rmsprop_lr0.001
```

**预期结果**:
- Adam: 通常收敛最快，性能稳定
- SGD with momentum: 需要更多调参，但可能达到更好的泛化
- RMSprop: 性能介于 Adam 和 SGD 之间

## 模型架构

### 不带 Batch Normalization 的 CNN

```
Input (1, 28, 28)
    ↓
Conv2d (1→32, 3x3, padding=1)
    ↓
ReLU
    ↓
MaxPool2d (2x2)
    ↓ (32, 14, 14)
Conv2d (32→64, 3x3, padding=1)
    ↓
ReLU
    ↓
MaxPool2d (2x2)
    ↓ (64, 7, 7)
Flatten → 3136
    ↓
Linear (3136→128)
    ↓
ReLU
    ↓
Dropout (0.5)
    ↓
Linear (128→10)
    ↓
Output (10 classes)
```

### 带 Batch Normalization 的 CNN

```
Input (1, 28, 28)
    ↓
Conv2d (1→32, 3x3, padding=1)
    ↓
BatchNorm2d (32)
    ↓
ReLU
    ↓
MaxPool2d (2x2)
    ↓ (32, 14, 14)
Conv2d (32→64, 3x3, padding=1)
    ↓
BatchNorm2d (64)
    ↓
ReLU
    ↓
MaxPool2d (2x2)
    ↓ (64, 7, 7)
Flatten → 3136
    ↓
Linear (3136→128)
    ↓
BatchNorm1d (128)
    ↓
ReLU
    ↓
Dropout (0.5)
    ↓
Linear (128→10)
    ↓
Output (10 classes)
```

## 代码规范

- 使用 Python 类型提示 (Type Hints)
- 包含中文注释说明关键功能
- 模块化设计，职责分离
- 使用 argparse 处理命令行参数
- 遵循 PEP 8 代码风格

## 常见问题

### Q: 如何调整批次大小？

A: 使用 `--batch_size` 参数：
```bash
python src/train.py --batch_size 128
```

### Q: 训练速度慢怎么办？

A: 
1. 使用 GPU：确保 CUDA 可用
2. 增加 `--num_workers` 参数（建议不超过 CPU 核心数）
3. 使用更大的批次大小

### Q: 如何恢复训练？

A: 修改 `train.py` 中的 `load_checkpoint` 调用即可从检查点恢复。

### Q: 模型准确率低怎么办？

A:
1. 增加训练轮数 `--epochs`
2. 尝试不同的学习率 `--lr`
3. 尝试不同的优化器 `--optimizer`
4. 使用带 BN 的模型 `--model with_bn`

## 预期性能

在默认配置下（Adam 优化器，学习率 0.001，10 个 epoch）：
- **不带 BN**: 测试准确率约 98.5%
- **带 BN**: 测试准确率约 99.0%

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 作者

MuskSwift

## 参考资料

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Batch Normalization Paper](https://arxiv.org/abs/1502.03167)