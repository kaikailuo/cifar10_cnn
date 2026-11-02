# CIFAR-10 CNN 图像分类

基于卷积神经网络（CNN）在 CIFAR-10 数据集上完成图像分类的课程作业。完整题目要求见 [TASK.md](TASK.md)。

## 交付物

1. **源码**：CNN 模型定义、训练与评估流程、实验脚本
2. **报告**：使用 LaTeX 撰写，介绍方法、实验设置、结果与分析

## 计划覆盖的内容

| 要求 | 内容 |
| --- | --- |
| 基础（20 pts） | CNN 结构设计、代码实现与训练过程 |
| 基础 | Dropout 与多种 Normalization（如 BatchNorm / LayerNorm / GroupNorm 等）对泛化能力的影响对比 |
| 基础 | 通过交叉验证搜索最优超参数 |
| 附加（+5 pts） | 其它提升性能的方法：加深网络、其它正则化、模型集成等 |
| 附加（+5 pts） | 分析现有 CNN 模型的问题，提出解决方案并实验验证 |

## 目录结构

```
.
├── TASK.md        # 作业原始要求
├── README.md      # 项目总描述
├── data/
│   ├── raw/cifar10/cifar-10-batches-py/  # CIFAR-10 原始数据（不进 Git）
│   └── splits/                           # 固定训练/验证划分索引（进 Git）
├── src/           # 源码（待建）
├── experiments/   # 实验配置与结果（待建）
└── report/        # LaTeX 报告（模板沿用 A1）
```

## 数据

- `data/raw/cifar10/cifar-10-batches-py/`：CIFAR-10 官方 Python 版本，从 A1 项目复制，目录结构与 `torchvision.datasets.CIFAR10(root="data/raw/cifar10")` 兼容，无需再下载。该目录不进 Git，新环境需自行下载。
- `data/splits/cifar10_split_v1.{npz,json}`：A1 使用的分层划分（训练 45000 / 验证 5000，每类 4500 / 500，seed 20261011），`npz` 内含 `train_indices`、`val_indices`。沿用它可以与 A1 的 MLP 结果在同一验证集上直接对比。

## 与 A1 的关系

A1（NumPy MLP）仓库：<https://github.com/kaikailuo/numpy-mlp-classification>。A1 在 CIFAR-10 上的 MLP 基准测试准确率为 50.39%，可作为本次 CNN 的对照起点。
