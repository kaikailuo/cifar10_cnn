"""Turn results into LaTeX tables (booktabs) and PDF figures for the report.

Real runs write to report/tables and report/figures; smoke runs to results_smoke/report/.
Figure text is English so the PDFs render without CJK fonts; captions live in the LaTeX sources.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .data import ROOT  # noqa: E402
from .experiments import Context, load_group, variant_of  # noqa: E402

A1_MLP_CIFAR_TEST = 0.5039  # A1 NumPy MLP baseline, for reference

# Categorical slots in fixed order (validated reference palette, light mode); text stays in ink colours.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ---- loading and aggregation -----------------------------------------------------------------

def variants(ctx: Context, group: str) -> dict[str, list[dict]]:
    """{variant: [summary + history per seed]} for a finished group."""
    out: dict[str, list[dict]] = {}
    for name, (cfg, summary) in load_group(ctx, group).items():
        run = dict(summary, config=cfg, history=history(ctx.results / group / name))
        out.setdefault(variant_of(name), []).append(run)
    return out


def history(run_dir: Path) -> dict[str, np.ndarray]:
    import csv
    with (run_dir / "history.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    return {k: np.array([float(r[k]) if r.get(k) not in (None, "") else np.nan for r in rows]) for k in rows[0]}


def pct(values, std: bool = True) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return "发散"
    mean = 100 * np.mean(vals)
    if std and len(vals) > 1:
        return f"{mean:.2f} $\\pm$ {100 * np.std(vals, ddof=1):.2f}"
    return f"{mean:.2f}"


def field(runs: list[dict], key: str) -> list:
    return [None if r["diverged"] else r.get(key) for r in runs]


def write_table(path: Path, header: list[str], rows: list[list[str]], align: str | None = None) -> None:
    align = align or "l" + "c" * (len(header) - 1)
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


# ---- figures ---------------------------------------------------------------------------------

def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9, "axes.titlesize": 9,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
        "legend.frameon": False, "legend.fontsize": 8, "text.color": INK,
    })


def save(fig, path: Path) -> None:
    fig.savefig(path)
    plt.close(fig)


def fig_curves(path: Path, curves: list[tuple[str, dict]]) -> None:
    """Train (solid) vs validation (dashed) accuracy per epoch, one colour per configuration."""
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for i, (label, h) in enumerate(curves):
        ax.plot(h["epoch"], 100 * h["train_eval_acc"], color=SERIES[i], label=f"{label} (train)")
        ax.plot(h["epoch"], 100 * h["val_acc"], color=SERIES[i], linestyle="--", label=f"{label} (val)")
    ax.set(xlabel="Epoch", ylabel="Accuracy (%)")
    ax.legend(ncol=2)
    save(fig, path)


def fig_train_val_points(path: Path, labels: list[str], train: list[list], val: list[list]) -> None:
    """Mean ± std of train and validation accuracy per configuration; the vertical distance is the gap."""
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    x = np.arange(len(labels))
    for j, (name, data) in enumerate((("train", train), ("val", val))):
        m = [100 * np.mean([v for v in d if v is not None]) if any(v is not None for v in d) else np.nan for d in data]
        s = [100 * np.std([v for v in d if v is not None], ddof=1) if sum(v is not None for v in d) > 1 else 0 for d in data]
        ax.errorbar(x, m, yerr=s, fmt="o", markersize=6, capsize=3, color=SERIES[j], label=name)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    save(fig, path)


def fig_lines(path: Path, series: list[tuple[str, np.ndarray, np.ndarray]], xlabel: str, ylabel: str,
              logx: bool = False, markers: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for i, (label, x, y) in enumerate(series):
        ax.plot(x, y, color=SERIES[i], label=label, marker="o" if markers else None, markersize=6)
    if logx:
        ax.set_xscale("log")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.legend()
    save(fig, path)


# ---- build -----------------------------------------------------------------------------------

def build(ctx: Context) -> None:
    out = ROOT / "report" if not ctx.smoke else ctx.results / "report"
    tables, figures = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    style()

    e0 = variants(ctx, "e0_baseline")
    e1 = variants(ctx, "e1_dropout")
    e2 = variants(ctx, "e2_norm")
    e2b = variants(ctx, "e2_batch")
    e2lr = variants(ctx, "e2_lr")
    e2dn = variants(ctx, "e2_dropout_norm")
    base = e0["vgg-none"]

    # E1 dropout
    drop = [("无 Dropout", base), ("FC 0.3", e1["fc0.3"]), ("FC 0.5", e1["fc0.5"]), ("FC 0.7", e1["fc0.7"]),
            ("Conv 0.1 + FC 0.5", e1["conv0.1-fc0.5"]), ("Conv 0.2 + FC 0.5", e1["conv0.2-fc0.5"])]
    write_table(tables / "dropout.tex", ["配置", "训练准确率 (\\%)", "验证准确率 (\\%)", "泛化差距 (pp)", "验证损失"],
                [[n, pct(field(r, "final_train_eval_acc")), pct(field(r, "final_val_acc")), pct(field(r, "gap")),
                  f"{np.mean([x for x in field(r, 'final_val_loss') if x is not None]):.3f}"] for n, r in drop])
    fig_train_val_points(figures / "dropout_gap.pdf", ["none", "fc0.3", "fc0.5", "fc0.7", "c0.1+fc0.5", "c0.2+fc0.5"],
                         [field(r, "final_train_eval_acc") for _, r in drop], [field(r, "final_val_acc") for _, r in drop])

    # E2 normalization
    norms = [("无", "none", base), ("BatchNorm", "bn", e2["bn"]), ("LayerNorm", "ln", e2["ln"]),
             ("InstanceNorm", "in", e2["in"]), ("GroupNorm", "gn", e2["gn"])]
    write_table(tables / "norm.tex", ["归一化", "训练准确率 (\\%)", "验证准确率 (\\%)", "泛化差距 (pp)", "每轮耗时 (s)"],
                [[n, pct(field(r, "final_train_eval_acc")), pct(field(r, "final_val_acc")), pct(field(r, "gap")),
                  f"{np.mean([np.nanmean(x['history']['epoch_time']) for x in r]):.1f}"] for n, _, r in norms])
    fig_lines(figures / "norm_curves.pdf", [(k, r[0]["history"]["epoch"], 100 * r[0]["history"]["val_acc"]) for _, k, r in norms],
              "Epoch", "Validation accuracy (%)")
    fig_curves(figures / "baseline_curves.pdf", [("no norm", base[0]["history"]), ("fc dropout 0.5", e1["fc0.5"][0]["history"]),
                                                 ("batchnorm", e2["bn"][0]["history"])])

    bs_rows = []
    for label, key in (("BatchNorm", "bn"), ("GroupNorm", "gn"), ("LayerNorm", "ln")):
        big, small = field(e2[key][:1], "final_val_acc"), field(e2b[f"{key}-bs16"], "final_val_acc")
        drop_pp = "--" if None in big + small else f"{100 * (small[0] - big[0]):+.2f}"
        bs_rows.append([label, pct(big), pct(small), drop_pp])
    write_table(tables / "norm_batch.tex", ["归一化", "Batch 128 (\\%)", "Batch 16 (\\%)", "变化 (pp)"], bs_rows)

    lrs = (0.005, 0.02, 0.1, 0.4)
    lr_series, lr_rows = [], []
    for label, key in (("无", "none"), ("BatchNorm", "bn"), ("GroupNorm", "gn")):
        ref = base[:1] if key == "none" else e2[key][:1]
        accs = [field(ref if lr == 0.02 else e2lr[f"{key}-lr{lr}"], "final_val_acc")[0] for lr in lrs]
        lr_rows.append([label] + [pct([a]) for a in accs])
        lr_series.append((key, np.array(lrs), np.array([100 * a if a is not None else 10.0 for a in accs])))
    write_table(tables / "norm_lr.tex", ["归一化"] + [f"lr={lr}" for lr in lrs], lr_rows)
    fig_lines(figures / "norm_lr.pdf", lr_series, "Peak learning rate", "Validation accuracy (%)", logx=True, markers=True)

    dn = [("无", base, e1["fc0.5"], e1["conv0.1-fc0.5"]), ("BatchNorm", e2["bn"], e2dn["bn-fc0.5"], e2dn["bn-conv0.1-fc0.5"])]
    write_table(tables / "dropout_norm.tex", ["归一化", "无 Dropout", "FC 0.5", "Conv 0.1 + FC 0.5"],
                [[n] + [pct(field(r, "final_val_acc")) for r in rs] for n, *rs in dn])

    # E3 search and cross-validation
    search = sorted(variants(ctx, "e3_search").items(), key=lambda kv: -(kv[1][0]["final_val_acc"] or 0))
    write_table(tables / "search.tex", ["配置", "lr", "weight decay", "FC Dropout", "Batch", "归一化", "验证准确率 (\\%)"],
                [[v, f"{r[0]['config']['lr']:.4g}", f"{r[0]['config']['weight_decay']:.2e}", f"{r[0]['config']['fc_dropout']:.1f}",
                  str(r[0]["config"]["batch_size"]), r[0]["config"]["norm"].upper(), pct(field(r, "final_val_acc"))]
                 for v, r in search])
    analysis = json.loads((ctx.results / "analysis" / "analysis.json").read_text())
    fin = analysis["final"]
    write_table(tables / "cv.tex", ["配置", "lr", "wd", "Dropout", "Batch", "归一化"] + [f"折 {k + 1}" for k in range(len(fin["cv"][0]["fold_val_acc"]))] + ["均值 $\\pm$ 标准差"],
                [[c["variant"], f"{c['hp']['lr']:.4g}", f"{c['hp']['weight_decay']:.1e}", f"{c['hp']['fc_dropout']:.1f}",
                  str(c["hp"]["batch_size"]), c["hp"]["norm"].upper()] + [f"{100 * a:.2f}" for a in c["fold_val_acc"]]
                 + [f"{100 * c['mean']:.2f} $\\pm$ {100 * c['std']:.2f}"] for c in fin["cv"]])
    fold_test = [f["acc"] for f in fin["fold_test"]]
    write_table(tables / "final.tex", ["模型", "测试准确率 (\\%)", "测试 NLL"], [
        ["A1 NumPy MLP（参考）", f"{100 * A1_MLP_CIFAR_TEST:.2f}", "--"],
        [f"最优配置 {fin['winner']}，全量 50000 张重训", f"{100 * fin['final_test']['acc']:.2f}", f"{fin['final_test']['nll']:.3f}"],
        ["5 个交叉验证折模型（单模型均值）", pct(fold_test), f"{np.mean([f['nll'] for f in fin['fold_test']]):.3f}"],
        ["5 个折模型集成（概率平均）", f"{100 * fin['cv_ensemble_test']['acc']:.2f}", f"{fin['cv_ensemble_test']['nll']:.3f}"],
    ])

    # Bonus 1 ablation
    b1 = analysis["b1"]
    b1_labels = {"b1-0-base": "CV 最优配置（无增强）", "b1-1-flipcrop": "+ 随机裁剪/翻转", "b1-2-cutout": "+ Cutout",
                 "b1-3-ls": "+ 标签平滑 0.1", "b1-4-wide": "+ 通道加宽 ×2", "b1-5-resnet56": "换用 ResNet-56"}
    rows = [[b1_labels[variant_of(n)], f"{r['params'] / 1e6:.2f}M", f"{100 * r['val_acc']:.2f}", f"{100 * r['test']['acc']:.2f}"]
            for n, r in b1["runs"].items()]
    rows.append([f"集成（验证最优 3 个）", "--", "--", f"{100 * b1['ensemble_test']['acc']:.2f}"])
    write_table(tables / "b1_ablation.tex", ["方法（逐项累加）", "参数量", "验证准确率 (\\%)", "测试准确率 (\\%)"], rows)
    names = list(b1["runs"])
    # Dot plot rather than bars: the accuracy axis is deliberately not anchored at zero.
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    vals = [100 * b1["runs"][n]["val_acc"] for n in names]
    ax.plot(vals, range(len(names)), "o", color=SERIES[0], markersize=7)
    for i, v in enumerate(vals):
        ax.text(v + 0.25, i, f"{v:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(names)), [variant_of(n) for n in names])
    ax.invert_yaxis()
    ax.set_xlim(min(vals) - 1.5, max(vals) + 2)
    ax.set_xlabel("Validation accuracy (%)")
    ax.grid(axis="y", visible=False)
    save(fig, figures / "b1_ablation.pdf")

    # Bonus 2: degradation
    depth = variants(ctx, "b2_depth")
    depth_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), sharey=True)
    for ax, kind in zip(axes, ("plain", "res")):
        for i, d in enumerate((20, 32, 56)):
            h = depth[f"{kind}{d}"][0]["history"]
            ax.plot(h["epoch"], 100 * (1 - h["train_eval_acc"]), color=SERIES[i], label=f"{d} layers")
        ax.set(title="Plain" if kind == "plain" else "ResNet", xlabel="Epoch")
        ax.legend()
    axes[0].set_ylabel("Training error (%)")
    save(fig, figures / "depth_train_error.pdf")
    for d in (20, 32, 56):
        row = [f"{d}"]
        for kind in ("plain", "res"):
            r = depth[f"{kind}{d}"]
            tr, va = field(r, "final_train_eval_acc")[0], field(r, "final_val_acc")[0]
            row += ["发散", "发散"] if tr is None else [f"{100 * (1 - tr):.2f}", f"{100 * (1 - va):.2f}"]
        depth_rows.append(row)
    write_table(tables / "depth.tex", ["层数", "Plain 训练误差", "Plain 验证误差", "ResNet 训练误差", "ResNet 验证误差"], depth_rows)

    # Bonus 2: calibration
    cal = analysis["calibration"]
    write_table(tables / "calibration.tex", ["模型", "准确率 (\\%)", "平均置信度 (\\%)", "NLL", "ECE (\\%)", "温度 $T$", "TS 后 NLL", "TS 后 ECE (\\%)"],
                [[variant_of(n), f"{100 * c['acc']:.2f}", f"{100 * c['mean_conf']:.2f}", f"{c['nll']:.3f}", f"{100 * c['ece']:.2f}",
                  f"{c['temperature']:.2f}", f"{c['nll_ts']:.3f}", f"{100 * c['ece_ts']:.2f}"] for n, c in cal.items()])
    panels = [("b1-2-cutout__s0", "bins", "Cutout model, raw"), ("b1-2-cutout__s0", "bins_ts", "Cutout model, temperature-scaled"),
              ("b1-3-ls__s0", "bins", "Label smoothing, raw")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, (name, key, title) in zip(axes, panels):
        bins = cal[name][key]
        centers = [(b["lo"] + b["hi"]) / 2 for b in bins]
        ax.bar(centers, [b["acc"] for b in bins], width=1 / 15 - 0.01, color=SERIES[0], label="accuracy")
        ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle="--", label="perfect calibration")
        ax.set(title=title, xlabel="Confidence", xlim=(0, 1), ylim=(0, 1))
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="upper left")
    save(fig, figures / "reliability.pdf")

    # Bonus 2: shift consistency
    shift = analysis["shift"]
    write_table(tables / "shift.tex", ["下采样", "训练增强", "原图准确率 (\\%)", "平移后平均准确率 (\\%)", "平移一致性 (\\%)"],
                [[k.split("-")[0].replace("max", "MaxPool").replace("blur", "MaxBlurPool"), k.split("-")[1],
                  f"{100 * s['clean_acc']:.2f}", f"{100 * s['mean_shifted_acc']:.2f}", f"{100 * s['consistency']:.2f}"]
                 for k, s in shift.items()])
    fig_lines(figures / "shift_accuracy.pdf",
              [(k, np.array([int(m) for m in s["acc_by_magnitude"]]), 100 * np.array(list(s["acc_by_magnitude"].values())))
               for k, s in shift.items()], "Shift magnitude (pixels, Chebyshev)", "Test accuracy (%)", markers=True)

    print(f"tables and figures written to {out}", flush=True)
