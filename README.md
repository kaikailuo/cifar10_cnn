# CIFAR-10 CNN 图像分类

基于卷积神经网络（CNN）在 CIFAR-10 数据集上完成图像分类的课程作业（A2）。完整题目要求见 [TASK.md](TASK.md)。

## 交付物

1. **源码**：`src/`，PyTorch 实现的模型、训练、交叉验证、分析与报告图表生成
2. **报告**：`report/`，LaTeX 撰写

## 实验设计

所有实验共用一个 VGG 风格基准网络（3 段 × 2 个 3×3 卷积，通道 64/128/256，FC 256 → 10），SGD + Nesterov 动量，2 轮预热后余弦退火。基础部分不做数据增强，以便让正则化与归一化对泛化的影响清楚可见。**训练阶段从不使用测试集**，所有选择冻结后由 `src/analysis.py` 统一评估一次。

| 组 | 对应要求 | 内容 | 次数 |
| --- | --- | --- | ---: |
| `e0_baseline` | 基础：CNN 结构与训练 | 无归一化、无 Dropout 的基准网络，3 个种子 | 3 |
| `e1_dropout` | 基础：Dropout | FC Dropout 0.3/0.5/0.7，卷积通道 Dropout 0.1/0.2 + FC 0.5，各 3 种子 | 15 |
| `e2_norm` | 基础：Normalization | BN / LN / IN / GN，各 3 种子 | 12 |
| `e2_batch` | 基础：Normalization | Batch 16 下 BN / GN / LN（学习率线性缩放） | 3 |
| `e2_lr` | 基础：Normalization | 无归一化 / BN / GN × 学习率 0.005/0.1/0.4 | 9 |
| `e2_dropout_norm` | 基础：Dropout × BN | BN + Dropout 组合，各 3 种子 | 6 |
| `e3_search` | 基础：交叉验证选超参 | 24 组随机搜索（lr、weight decay、Dropout、batch、BN/GN），15 轮 | 24 |
| `e3_cv` | 基础：交叉验证选超参 | 搜索前 4 名做分层 5 折交叉验证，30 轮 | 20 |
| `e3_final` | 基础：交叉验证选超参 | 交叉验证最优配置在全部 50000 张上重训 | 1 |
| `b1_ablation` | 附加 1：提升性能 | 逐项累加：裁剪翻转 → Cutout → 标签平滑 → 加宽 → ResNet-56；另做集成 | 6 |
| `b2_depth` | 附加 2：退化问题 | Plain vs ResNet，20/32/56 层 | 6 |
| `b2_shift` | 附加 2：平移敏感 | MaxBlurPool（抗混叠下采样）vs MaxPool | 2 |
| 分析 | 附加 2：过度自信 | ECE / 可靠性图，温度缩放、标签平滑（复用 b1 模型） | 0 |

阶段依赖：阶段 1（e0–e2、e3_search、b2_depth）→ 阶段 2（e3_cv）→ 阶段 3（e3_final、b1、b2_shift）→ 分析与出图。

## 运行

### 本机（Intel Mac，CPU，仅用于调试）

```bash
uv sync --group dev
uv run --group dev pytest -q
uv run python -m src.run --smoke --jobs 2   # 每个实验只用 512 张图训练 2 轮，结果在 results_smoke/
```

### GPU 服务器（如 AutoDL，PyTorch 镜像）

```bash
pip install -r requirements-gpu.txt
python -m pytest -q
python -m src.run --smoke --jobs 4          # 先在 GPU 上冒烟一遍
python -m src.run --jobs 4                  # 正式全部实验 + 分析 + 出图，建议放在 tmux 里
```

- `--jobs N`：同时训练的实验数。模型很小，一张 4090 可同时跑 3–4 个；先看 `nvidia-smi` 利用率再调整。
- 中断后重新执行同一条命令即可：已完成的实验跳过，未完成的从最近一轮的断点续跑。
- 只跑部分组：`python -m src.run --groups e1_dropout,e2_norm --jobs 4`；之后用 `python -m src.run --analyze-only` 做分析与出图。
- 需要上传：整个仓库 + `data/raw/cifar10/cifar-10-batches-py/`（178 MB）。

## 输出

- `results/<组>/<实验>/`：`config.json`、`history.csv`（逐轮训练/验证指标）、`summary.json`、`train.log`，以及不进 Git 的 `model.pt`、`val_logits.npz`、`test_logits.npz`。
- `results/analysis/analysis.json`：测试集结果、集成、校准、平移一致性。
- `report/tables/*.tex`、`report/figures/*.pdf`：报告直接 `\input` / `\includegraphics` 的三线表与图。

## 目录结构

```
.
├── TASK.md                # 作业原始要求
├── src/
│   ├── config.py          # 单次实验的全部配置项及默认值
│   ├── data.py            # CIFAR-10 读取、固定划分、5 折划分、GPU 端批处理与增强
│   ├── models.py          # VGG 风格 CNN、CIFAR ResNet / Plain 网络、各类归一化、BlurPool
│   ├── train.py           # 单次训练（可断点续跑）
│   ├── experiments.py     # 全部实验组定义
│   ├── run.py             # 分阶段并行调度入口
│   ├── analysis.py        # 测试评估、集成、校准、平移一致性
│   └── report.py          # 生成 LaTeX 表格与 PDF 图
├── tests/
├── data/
│   ├── raw/cifar10/cifar-10-batches-py/  # CIFAR-10 原始数据（不进 Git）
│   └── splits/            # A1 固定划分 + 5 折划分索引
├── results/               # 实验结果
└── report/                # LaTeX 报告
```

## 数据划分

- `data/splits/cifar10_split_v1.{npz,json}`：沿用 A1 的分层划分（训练 45000 / 验证 5000，seed 20261011），可与 A1 的 MLP 在同一验证集上对比。
- `data/splits/cifar10_kfold5_v1.{npz,json}`：50000 张训练图的分层 5 折划分（seed 20261102），每折每类 1000 张。

## 与 A1 的关系

A1（NumPy MLP）仓库：<https://github.com/kaikailuo/numpy-mlp-classification>。A1 在 CIFAR-10 上的 MLP 测试准确率为 50.39%，报告中作为对照。
