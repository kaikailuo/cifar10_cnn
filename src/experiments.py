"""Experiment groups. Each group maps run names ("<variant>__<tag>") to configs.

Phase 1 groups are independent; e3_cv needs e3_search; phase 3 groups build on the CV winner.
The test set is never touched during training: src/analysis.py scores it once, after all selections.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import make_config

SEEDS = (0, 1, 2)
TUNED = ("lr", "weight_decay", "fc_dropout", "batch_size", "norm")
SEARCH_SIZE = 24
SEARCH_EPOCHS = 15
CV_TOP_K = 4
CV_EPOCHS = 30
BONUS_EPOCHS = 40


class MissingResults(RuntimeError):
    pass


@dataclass(frozen=True)
class Context:
    results: Path
    smoke: bool = False

    @property
    def seeds(self) -> tuple[int, ...]:
        return SEEDS[:1] if self.smoke else SEEDS


def variant_of(name: str) -> str:
    return name.split("__")[0]


def seeded(ctx: Context, variants: dict[str, dict]) -> dict[str, dict]:
    return {f"{v}__s{s}": make_config(**ov, seed=s) for v, ov in variants.items() for s in ctx.seeds}


def load_group(ctx: Context, group: str) -> dict[str, tuple[dict, dict]]:
    """{run name: (config, summary)} for a finished group; raises if any run is missing."""
    expected = GROUPS[group][1](ctx)
    out = {}
    for name in expected:
        run_dir = ctx.results / group / name
        if not (run_dir / "summary.json").exists():
            raise MissingResults(f"{group}/{name} has not finished")
        out[name] = (json.loads((run_dir / "config.json").read_text()), json.loads((run_dir / "summary.json").read_text()))
    return out


# ---- Basic part ------------------------------------------------------------------------------

def e0_baseline(ctx):
    # Plain VGG: no normalization, no dropout, no augmentation. Doubles as the "none"/"p=0" reference below.
    return seeded(ctx, {"vgg-none": {}})


def e1_dropout(ctx):
    return seeded(ctx, {
        "fc0.3": {"fc_dropout": 0.3},
        "fc0.5": {"fc_dropout": 0.5},
        "fc0.7": {"fc_dropout": 0.7},
        "conv0.1-fc0.5": {"conv_dropout": 0.1, "fc_dropout": 0.5},
        "conv0.2-fc0.5": {"conv_dropout": 0.2, "fc_dropout": 0.5},
    })


def e2_norm(ctx):
    return seeded(ctx, {n: {"norm": n} for n in ("bn", "ln", "in", "gn")})


def e2_batch(ctx):
    # Small batches with the learning rate scaled linearly; batch 128 counterparts come from e0/e2_norm.
    return {f"{n}-bs16__s0": make_config(norm=n, batch_size=16, lr=0.02 * 16 / 128) for n in ("bn", "gn", "ln")}


def e2_lr(ctx):
    # lr = 0.02 counterparts come from e0/e2_norm.
    return {f"{n}-lr{lr}__s0": make_config(norm=n, lr=lr) for n in ("none", "bn", "gn") for lr in (0.005, 0.1, 0.4)}


def e2_dropout_norm(ctx):
    return seeded(ctx, {
        "bn-fc0.5": {"norm": "bn", "fc_dropout": 0.5},
        "bn-conv0.1-fc0.5": {"norm": "bn", "conv_dropout": 0.1, "fc_dropout": 0.5},
    })


def search_space(n: int = SEARCH_SIZE) -> list[dict]:
    rng = np.random.default_rng(20261103)
    out = []
    for _ in range(n):
        out.append({
            "lr": round(float(10 ** rng.uniform(np.log10(0.005), np.log10(0.2))), 5),
            "weight_decay": round(float(10 ** rng.uniform(-5, np.log10(2e-3))), 7),
            "fc_dropout": float(rng.choice([0.0, 0.3, 0.5])),
            "batch_size": int(rng.choice([64, 128, 256])),
            "norm": str(rng.choice(["bn", "gn"])),
        })
    return out


def e3_search(ctx):
    # Random search on the fixed holdout split with a shortened schedule, to shortlist candidates for CV.
    space = search_space()[: 4 if ctx.smoke else SEARCH_SIZE]
    return {f"cfg{i:02d}__s0": make_config(**hp, epochs=SEARCH_EPOCHS) for i, hp in enumerate(space)}


def cv_candidates(ctx) -> list[tuple[str, dict]]:
    search = load_group(ctx, "e3_search")
    ranked = sorted((s["final_val_acc"], n) for n, (_, s) in search.items() if not s["diverged"])[::-1]
    top = [n for _, n in ranked[: 2 if ctx.smoke else CV_TOP_K]]
    return [(variant_of(n), {k: search[n][0][k] for k in TUNED}) for n in top]


def e3_cv(ctx):
    from .data import KFOLD_K
    return {f"{v}__f{k}": make_config(**hp, epochs=CV_EPOCHS, split="fold", fold=k)
            for v, hp in cv_candidates(ctx) for k in range(KFOLD_K)}


def cv_ranking(ctx) -> list[tuple[str, dict, list[float]]]:
    """CV candidates sorted by mean validation accuracy over folds: (variant, hyperparameters, fold accuracies)."""
    cv = load_group(ctx, "e3_cv")
    rows = {}
    for name, (cfg, summary) in cv.items():
        v = variant_of(name)
        rows.setdefault(v, ({k: cfg[k] for k in TUNED}, []))[1].append(summary["final_val_acc"] or 0.0)
    return sorted(((v, hp, accs) for v, (hp, accs) in rows.items()), key=lambda r: -np.mean(r[2]))


def cv_winner(ctx) -> tuple[str, dict]:
    v, hp, _ = cv_ranking(ctx)[0]
    return v, hp


def e3_final(ctx):
    v, hp = cv_winner(ctx)
    return {f"{v}__full": make_config(**hp, epochs=CV_EPOCHS, split="full")}


# ---- Bonus part ------------------------------------------------------------------------------

def b1_ablation(ctx):
    # Cumulative additions on top of the CV-selected hyperparameters, all on the holdout split.
    _, hp = cv_winner(ctx)
    aug = {"augment": "flipcrop"}
    cut = {**aug, "cutout": 16}
    ls = {**cut, "label_smoothing": 0.1}
    steps = {
        "b1-0-base": {},
        "b1-1-flipcrop": aug,
        "b1-2-cutout": cut,
        "b1-3-ls": ls,
        "b1-4-wide": {**ls, "widths": [128, 256, 512]},
        "b1-5-resnet56": {**ls, "model": "resnet", "depth": 56},
    }
    return {f"{v}__s0": make_config(**{**hp, "epochs": BONUS_EPOCHS, **ov}) for v, ov in steps.items()}


def b2_depth(ctx):
    # Degradation problem: plain vs residual networks of growing depth, standard CIFAR ResNet recipe.
    return {f"{'res' if r else 'plain'}{d}__s0": make_config(model="resnet", depth=d, residual=r, norm="bn", lr=0.1,
                                                           augment="flipcrop")
            for d in (20, 32, 56) for r in (True, False)}


def b2_shift(ctx):
    # Anti-aliased pooling; the max-pooling counterparts are b1-0-base and b1-1-flipcrop.
    _, hp = cv_winner(ctx)
    return {f"blur-{a}__s0": make_config(**{**hp, "epochs": BONUS_EPOCHS, "pool": "blur", "augment": a})
            for a in ("none", "flipcrop")}


GROUPS = {
    "e0_baseline": (1, e0_baseline),
    "e1_dropout": (1, e1_dropout),
    "e2_norm": (1, e2_norm),
    "e2_batch": (1, e2_batch),
    "e2_lr": (1, e2_lr),
    "e2_dropout_norm": (1, e2_dropout_norm),
    "e3_search": (1, e3_search),
    "b2_depth": (1, b2_depth),
    "e3_cv": (2, e3_cv),
    "e3_final": (3, e3_final),
    "b1_ablation": (3, b1_ablation),
    "b2_shift": (3, b2_shift),
}
