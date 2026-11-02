from __future__ import annotations

import copy

# Every run is fully described by one flat dict; experiment groups only override keys.
DEFAULTS: dict = {
    # architecture
    "model": "vgg",  # vgg | resnet
    "widths": [64, 128, 256],  # vgg stage widths
    "convs_per_stage": 2,
    "fc_hidden": 256,
    "depth": 20,  # resnet depth, 6n + 2
    "residual": True,  # False gives the plain (shortcut-free) counterpart
    "norm": "none",  # none | bn | ln | in | gn
    "gn_groups": 8,
    "pool": "max",  # max | blur (anti-aliased max pooling)
    # regularization
    "conv_dropout": 0.0,  # channel-wise Dropout2d after each vgg stage
    "fc_dropout": 0.0,
    "weight_decay": 5e-4,
    "label_smoothing": 0.0,
    "augment": "none",  # none | flipcrop
    "cutout": 0,  # square side in pixels, 0 disables
    # optimization
    "epochs": 30,
    "batch_size": 128,
    "lr": 0.02,
    "momentum": 0.9,
    "warmup_epochs": 2,
    # data
    "split": "holdout",  # holdout (A1 45000/5000) | fold (5-fold CV) | full (all 50000, no val)
    "fold": None,
    "train_limit": None,  # cap on training samples (smoke runs)
    "val_limit": None,
    "seed": 0,
}

SMOKE_TRAIN_LIMIT = 512
SMOKE_VAL_LIMIT = 256
SMOKE_MAX_EPOCHS = 2


def make_config(**overrides) -> dict:
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"unknown config keys: {sorted(unknown)}")
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update(overrides)
    return cfg


def smoke_version(cfg: dict) -> dict:
    """Shrink a run to a few hundred samples and two epochs to check the pipeline end to end."""
    cfg = copy.deepcopy(cfg)
    cfg["epochs"] = min(cfg["epochs"], SMOKE_MAX_EPOCHS)
    cfg["warmup_epochs"] = min(cfg["warmup_epochs"], cfg["epochs"] - 1)
    cfg["train_limit"] = SMOKE_TRAIN_LIMIT
    cfg["val_limit"] = SMOKE_VAL_LIMIT
    return cfg
