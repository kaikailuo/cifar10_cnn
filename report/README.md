# 最终报告源码

本目录是 `A2_CIFAR10_report_v3_logo.pdf` 的唯一 LaTeX 工程。作者已验收的 18 页 PDF 位于 `../output/pdf/`，文件原样保留。工程来自对应的最终源码包，整理时仅调整绘图脚本的默认结果路径与使用说明。

## 编译

在本目录运行（需要 TeX Live 的 XeLaTeX、BibTeX、ctex、Fandol、gbt7714 与 latexmk）：

```sh
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

编译输出在忽略的 `build/` 中，不自动覆盖已验收 PDF。Overleaf 上传本目录源码，入口选 `main.tex`，编译器选 XeLaTeX。随附西文字体、许可证及来源记录；中文采用 Fandol。

## 重绘

在仓库根目录运行（NumPy、Matplotlib）：

```sh
python report/build_figures.py
python report/draw_architecture.py
```

默认读取 `results/`；也可用 `--results <目录>` 指向冻结记录。只重绘，不训练或执行测试推理。`src/report.py` 是实验流水线的原始出图器，会写入同名图；最终报告使用本目录 `build_figures.py` 的样式，重新运行正式流水线后应再次运行此脚本恢复最终样式。

## 内容

- `main.tex`、`sections/`、`metadata.tex`、`BUPTReport.sty`：正文与排版。
- `tables/`：原始结果表和正文展示表。
- `figures/`：9 张矢量 PDF、结构图 SVG、封面校徽；PNG 绘图预览不入 Git。
- `references.bib`、`bibliography_style.bib`：书目与样式控制。
- `fonts/`、`TEMPLATE_CREDITS.md`：字体和模板来源及许可证。
- `VERIFICATION.json`：报告交付时的历史验收记录，非本轮新跑的检查。

实验原始记录统一保留在仓库根目录 `results/`，无需复制 `records/` 或额外保存源码 zip。
