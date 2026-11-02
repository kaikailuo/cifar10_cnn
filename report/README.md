# A2 课程实验报告（LaTeX）

报告标题：《基于卷积神经网络的 CIFAR-10 图像分类实验》。入口为 `main.tex`，样式沿用 A1 报告工程的 `BUPTReport.sty`（12pt、Fandol 中文字体、GB/T 7714 数字引用），模板来源与许可证见 `TEMPLATE_CREDITS.md`。

## 目录

| 路径 | 内容 |
| --- | --- |
| `main.tex` | 主文件：封面、摘要、目录、正文、参考文献、附录 |
| `metadata.tex` | 封面字段（姓名、学号、题目、日期等） |
| `BUPTReport.sty` | 排版样式，自 A1 原样迁移，仅改 `graphicspath` 与默认 `pdfsubject` |
| `sections/` | 各章节源码（当前为骨架，带 TODO） |
| `figures/`、`tables/` | 实验图与三线表 |
| `references.bib` | 参考文献：A1 中仍适用的条目 + A2 新增的 CNN / 归一化条目 |

## 编译

Overleaf：上传 `report/` 目录，主文件 `main.tex`，编译器 XeLaTeX，BibTeX 使用 `gbt7714-numeric`。

本地（需完整 TeX Live，且安装 Liberation Serif / Liberation Sans / DejaVu Sans Mono 字体）：

```sh
xelatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```
