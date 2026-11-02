# 交接文档：A2 CIFAR-10 CNN（GPU 服务器阶段）

写于 2026-11-02。接手方：在有 GPU 的服务器上完成剩余工作的 Agent（Codex）。**截止时间：2026-11-03 晚**，请按“先跑实验、再写报告”的顺序推进。

## 1. 任务与交付物

课程作业 A2：构建 CNN 完成 CIFAR-10 图像分类，题目见 [TASK.md](TASK.md)。最终交付：

1. **源码**：本仓库 `src/`（已完成）
2. **LaTeX 报告**：`report/`（框架已有，正文待写）

评分点：基础 20 分（CNN 结构与训练、Dropout 与多种 Normalization 对泛化的影响、交叉验证选超参）+ 附加 1（提升性能，+5）+ 附加 2（分析 CNN 的问题并提出、验证解决方案，+5）。

## 2. 当前状态

| 项目 | 状态 |
| --- | --- |
| 实验代码（`src/`） | ✅ 完成，提交 `71e708e` |
| 单元测试（`tests/`，20 个） | ✅ 本机通过 |
| 端到端冒烟（512 张图 × 2 轮，53 次训练 + 分析 + 出图） | ✅ 本机 CPU 通过；断点续跑、发散处理已单独验证 |
| 正式实验 | ❌ 未跑（本机无 GPU） |
| 报告正文 | ❌ 只有章节骨架和 TODO |

实验设计、每组实验的内容与次数见 [README.md](README.md) 的“实验设计”表，此处不再重复。

## 3. 服务器准备

推荐单张 RTX 4090 / 3090（24G）、PyTorch 2.x + CUDA 12 镜像；仓库放在数据盘（AutoDL 为 `/root/autodl-tmp`）。

1. **代码与数据**：代码用 `git clone https://github.com/kaikailuo/cifar10_cnn.git` 获取（访问 GitHub 慢时，AutoDL 上可先 `source /etc/network_turbo`）。数据集**不在 Git 里**，需要单独上传 `data/raw/cifar10/cifar-10-batches-py/`（178 MB，8 个文件）到仓库内同一路径。校验：

   ```bash
   md5sum data/raw/cifar10/cifar-10-batches-py/data_batch_1 data/raw/cifar10/cifar-10-batches-py/test_batch
   # c99cafc152244af753f735de768cd75f  data_batch_1
   # 40351d587109b95175f43aff81a1287e  test_batch
   ```

2. **依赖**：torch、numpy 用镜像自带的，不要用 uv，也不要按 `pyproject.toml` 重装 torch（那里的 2.2.2 只是给 Intel Mac 用的）。

   ```bash
   pip install -r requirements-gpu.txt
   python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.is_bf16_supported())"
   ```

3. **自检**：

   ```bash
   python -m pytest -q                       # 应为 20 passed
   python -m src.run --smoke --jobs 4        # GPU 上几分钟，结果在 results_smoke/（已被 git 忽略）
   ```

4. **吞吐测试，决定 `--jobs`**：冒烟时另开终端看 `nvidia-smi`。模型很小，单个进程吃不满 GPU；默认 `--jobs 4`，GPU 利用率仍低且显存充足可调到 6。`results_smoke/*/*/history.csv` 的 `epoch_time` 列可以粗略看出速度。

## 4. 正式运行

```bash
tmux new -s a2
python -m src.run --jobs 4 2>&1 | tee run.log
```

- 分三个阶段自动依次执行：阶段 1（e0、e1、e2_*、e3_search、b2_depth，共 78 次）→ 阶段 2（e3_cv，20 次，依赖 e3_search 的前 4 名）→ 阶段 3（e3_final、b1_ablation、b2_shift，9 次，依赖交叉验证的最优配置）→ `analysis` → `report`。
- **估计耗时**：单张 4090 约 2 小时（按计算量估算，未实测）。阶段 1 前几次训练完成后，用 `summary.json` 里的 `train_time_s` 校准一下预估。
- **中断恢复**：重新执行同一条命令即可。已有 `summary.json` 的跳过；未完成的从 `checkpoint.pt` 续跑（按轮保存）。
- **单次失败**：`run.py` 打印 `FAILED`，并在该阶段结束后停止。到 `results/<组>/<实验>/train.log` 看原因，修复后重跑同一命令。
- 只做分析和出图：`python -m src.run --analyze-only`。

### 产出

- `results/<组>/<实验>/`：`config.json`、`history.csv`（逐轮训练/验证指标）、`summary.json`、`train.log`，以及不进 Git 的 `model.pt`、`*_logits.npz`
- `results/analysis/analysis.json`：测试集结果、集成、校准、平移一致性
- `report/tables/*.tex`（12 个）与 `report/figures/*.pdf`（8 个）：直接被报告引用

## 5. 必须遵守的约束

1. **测试集只在 `src/analysis.py` 里使用一次**，所有选择都在验证集或交叉验证上完成。不要依据测试结果回头改配置、改选择。报告里要写明这一点。
2. **不要在跑完之后修改实验定义**（`src/experiments.py`、`src/config.py` 的默认值）。`run.py` 发现 `config.json` 与当前定义不一致会直接报错，这是有意为之，用来防止结果和配置对不上。确需修改时，删除受影响的结果目录后重跑，并在报告中说明。
3. **数字只能来自实际运行**，不要编造或“合理估计”任何结果。某个实验没跑完，报告里就写没跑完。
4. 改代码后先跑 `python -m pytest -q`，再用 `--smoke` 验证一遍。

## 6. 已知的预期现象与风险

| 现象 | 说明 |
| --- | --- |
| `e2_lr/none-lr0.4`（可能还有 `none-lr0.1`）发散 | 预期内，这正是“归一化允许更大学习率”的证据。`summary.json` 里 `diverged: true`，表格中显示“发散”，图中画在 10%（随机猜测水平）。报告里要说明。 |
| `e2_batch` 的 Batch 16 训练明显更慢 | 每轮步数是 Batch 128 的 8 倍，主要受 kernel 启动开销限制，属正常。 |
| 温度缩放的搜索范围是 [0.05, 20] | 若某模型拟合出的 T 落在边界上（冒烟时 ResNet-56 出现过，原因是只训练了 2 轮），说明该模型的校准结果不可信，报告中要注明。 |
| 分析阶段要求 e3/b1/b2_shift 的实验不发散 | 这些实验用的是交叉验证选出的超参，正常不会发散。万一发散，`analysis.py` 会因缺少 `model.pt` 报错，需要处理后再分析。 |
| `report.build` 要求所有组都已完成 | 时间不够、只跑了部分组时，需要改 `src/report.py`，跳过缺失的组。 |
| GPU 不支持 bf16（如 V100、T4） | 自动退回 fp32，速度会变慢，但结果正确。 |

**时间不够时的优先级**（逐组用 `--groups` 运行）：
1. `e0_baseline,e1_dropout,e2_norm,e3_search`
2. `e3_cv`
3. `e3_final,b1_ablation`
4. `b2_depth,b2_shift`
5. `e2_batch,e2_lr,e2_dropout_norm`

前三步覆盖全部基础分和附加 1。

## 7. 写报告

### 工程与编译

- 入口 `report/main.tex`（ctexart + 自定义样式 `BUPTReport.sty`，XeLaTeX，BibTeX 用 `gbt7714-numeric`）。封面字段在 `metadata.tex`，已填好。
- 样式需要 Liberation Serif / Liberation Sans / DejaVu Sans Mono 字体和 Fandol 中文字体。服务器上若没有 TeX 环境，**不要花时间装**，作者会在 Overleaf 上编译。只需保证源码正确，表格、图片路径与实际文件一致。
- 表格用 `\input{tables/xxx}`，外层自己写 `table` 环境、`\caption`、`\label`；图用 `\includegraphics{xxx.pdf}`（`graphicspath` 已设为 `figures/`）。图中文字是英文（避免依赖中文字体），图注用中文写在 LaTeX 里。

### 章节与素材对应

| 章节文件 | 内容 | 对应的表与图 |
| --- | --- | --- |
| `01_introduction` | 任务、目标，与 A1 MLP（CIFAR-10 测试 50.39%）的衔接 | — |
| `02_theory` | 卷积/池化、Dropout、BN/LN/IN/GN 的公式与区别 | — |
| `03_implementation` | 代码结构（参考 README 目录）、GPU 端数据管线、训练流程、可复现性措施 | — |
| `04_experiments` | 数据划分（A1 的 45000/5000 + 分层 5 折）、基准网络、优化设置、各组实验设计、测试集使用纪律 | — |
| `05_results` | E0 基准、E1 Dropout、E2 归一化（批大小、学习率、与 Dropout 的组合）、E3 搜索与交叉验证、最终测试结果 | `dropout`、`norm`、`norm_batch`、`norm_lr`、`dropout_norm`、`search`、`cv`、`final`；`baseline_curves`、`dropout_gap`、`norm_curves`、`norm_lr` |
| `06_extensions` | 附加 1：逐项累加消融与集成；附加 2：退化问题、校准、平移一致性 | `b1_ablation`、`depth`、`calibration`、`shift`；`b1_ablation`、`depth_train_error`、`reliability`、`shift_accuracy` |
| `07_conclusion` | 主要结论与局限 | — |
| `08_appendix` | 复现命令、仓库地址 <https://github.com/kaikailuo/cifar10_cnn>、AI 辅助声明 | — |

摘要（`main.tex` 中的 TODO）最后写，150–250 字，包含最终测试准确率等关键数字。

### 写作要求

- 中文学术风格，参考 A1 报告的语气：结论要克制，“在本实验设置下”，不夸大。
- 每个结论都对应到具体的表或图的数字。若结果与预期不符（例如 Dropout 没有提升、BN 与 Dropout 组合反而更好），如实报告并分析可能原因，**不要为了符合预期调整叙述**。
- 方差：多种子实验报告“均值 ± 标准差”；单种子实验在文中注明“单次运行”。差异小于标准差时，不要说一种方法优于另一种。
- 引用 `references.bib` 里已有的条目，例如 `krizhevskyCifar`、`lecun1998gradient`、`srivastava2014dropout`、`ioffe2015batchnorm`、`ba2016layernorm`、`ulyanov2016instancenorm`、`wu2018groupnorm`、`he2016resnet`、`krizhevsky2012imagenet`、`he2015rectifiers`。A2 新增的条目（文件中已标注）尚未对照官方页面核对，有网络时请核对作者、年份、页码。
- 正文提到但 bib 中没有的方法需要补充真实文献，至少包括：Cutout（DeVries & Taylor, 2017, arXiv:1708.04552）、标签平滑（Szegedy et al., 2016, CVPR, “Rethinking the Inception Architecture”）、温度缩放与 ECE（Guo et al., 2017, ICML, “On Calibration of Modern Neural Networks”）、BlurPool（Zhang, 2019, ICML, “Making Convolutional Networks Shift-Invariant Again”）、Dropout 与 BN 的方差偏移（Li et al., 2019, CVPR, “Understanding the Disharmony between Dropout and Batch Normalization”）。**不确定的书目信息宁可省略，也不要编造。**
- 附录写明：本项目使用 AI 工具辅助代码与报告撰写，实验数据均来自实际运行，由作者核验。

## 8. 收尾与回传

- 提交：`results/` 下的配置、历史、摘要、`analysis.json`，`report/tables`、`report/figures`，报告源码。`.gitignore` 已排除 `*.pt`、`results/**/*.npz`、`results_smoke/`。
- **提交前请先征得作者同意**（作者习惯先检查再提交）。
- 远程仓库：<https://github.com/kaikailuo/cifar10_cnn>（分支 `main`）。服务器上 `git clone` 后再单独上传数据集；推送前同样要征得作者同意。模型权重如需保留，单独打包，不进 Git。
- 结束前写一段简短的结果摘要：各组关键数字、发散或异常的实验、与第 6 节预期不同之处，方便作者快速审阅。
