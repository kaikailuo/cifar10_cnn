from __future__ import annotations

import hashlib
import json
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "cifar10" / "cifar-10-batches-py"
SPLIT_DIR = ROOT / "data" / "splits"
HOLDOUT_SPLIT = SPLIT_DIR / "cifar10_split_v1.npz"  # inherited from A1
KFOLD_SPLIT = SPLIT_DIR / "cifar10_kfold5_v1.npz"
KFOLD_K = 5
KFOLD_SEED = 20261102

CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
# Per-channel statistics of the 50000 CIFAR-10 training images.
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def _load_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        record = pickle.load(handle, encoding="bytes")
    x = np.asarray(record[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    y = np.asarray(record[b"labels"], dtype=np.int64)
    return x, y


@lru_cache(maxsize=1)
def load_cifar10(raw_dir: Path = RAW_DIR) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_x, train_y, test_x, test_y); images are uint8 NCHW."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"missing CIFAR-10 batches at {raw_dir}")
    batches = [_load_batch(raw_dir / f"data_batch_{i}") for i in range(1, 6)]
    train_x = np.concatenate([b[0] for b in batches])
    train_y = np.concatenate([b[1] for b in batches])
    test_x, test_y = _load_batch(raw_dir / "test_batch")
    if len(train_y) != 50000 or len(test_y) != 10000:
        raise ValueError("unexpected CIFAR-10 sample counts")
    return train_x, train_y, test_x, test_y


def _label_hash(labels: np.ndarray) -> str:
    # A1 hashed uint8 labels; keep the same convention so its split metadata can be verified.
    return hashlib.sha256(labels.astype(np.uint8).tobytes()).hexdigest()


def holdout_split(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    meta = json.loads(HOLDOUT_SPLIT.with_suffix(".json").read_text())
    if meta["label_sha256"] != _label_hash(labels):
        raise ValueError("holdout split was created for different labels")
    with np.load(HOLDOUT_SPLIT) as archive:
        return archive["train_indices"].astype(np.int64), archive["val_indices"].astype(np.int64)


def stratified_folds(labels: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Assign each sample a fold id so every fold holds the same number of each class."""
    rng = np.random.default_rng(seed)
    folds = np.empty(len(labels), dtype=np.int64)
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % k
    return folds


def kfold_ids(labels: np.ndarray) -> np.ndarray:
    meta_path = KFOLD_SPLIT.with_suffix(".json")
    if KFOLD_SPLIT.exists():
        meta = json.loads(meta_path.read_text())
        if meta["label_sha256"] != _label_hash(labels) or meta["k"] != KFOLD_K or meta["seed"] != KFOLD_SEED:
            raise ValueError("k-fold split metadata does not match")
        with np.load(KFOLD_SPLIT) as archive:
            return archive["fold_ids"].astype(np.int64)
    folds = stratified_folds(labels, KFOLD_K, KFOLD_SEED)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(KFOLD_SPLIT, fold_ids=folds.astype(np.int8))
    meta = {
        "dataset": "cifar10",
        "algorithm": "stratified_round_robin",
        "k": KFOLD_K,
        "seed": KFOLD_SEED,
        "label_sha256": _label_hash(labels),
        "fold_sizes": np.bincount(folds, minlength=KFOLD_K).tolist(),
        "fold_class_counts": [np.bincount(labels[folds == f], minlength=10).tolist() for f in range(KFOLD_K)],
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return folds


def resolve_split(cfg: dict, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Training and validation indices into the 50000 training images (validation is None for 'full')."""
    if cfg["split"] == "holdout":
        train_idx, val_idx = holdout_split(labels)
    elif cfg["split"] == "fold":
        folds = kfold_ids(labels)
        train_idx, val_idx = np.flatnonzero(folds != cfg["fold"]), np.flatnonzero(folds == cfg["fold"])
    elif cfg["split"] == "full":
        train_idx, val_idx = np.arange(len(labels)), None
    else:
        raise ValueError(f"unknown split {cfg['split']!r}")
    if cfg["train_limit"]:
        train_idx = train_idx[: cfg["train_limit"]]
    if val_idx is not None and cfg["val_limit"]:
        val_idx = val_idx[: cfg["val_limit"]]
    return train_idx, val_idx


def normalize(x_uint8: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN, device=x_uint8.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x_uint8.device).view(1, 3, 1, 1)
    return (x_uint8.float() / 255.0 - mean) / std


def random_crop_flip(x: torch.Tensor, gen: torch.Generator, pad: int = 4) -> torch.Tensor:
    """Per-sample random crop from a zero-padded image (zero = mean colour after normalization) and horizontal flip."""
    b, c, h, w = x.shape
    dev = x.device
    padded = F.pad(x, (pad, pad, pad, pad))
    oy = torch.randint(0, 2 * pad + 1, (b,), device=dev, generator=gen)
    ox = torch.randint(0, 2 * pad + 1, (b,), device=dev, generator=gen)
    rows = (oy[:, None] + torch.arange(h, device=dev))[:, None, :, None]
    cols = (ox[:, None] + torch.arange(w, device=dev))[:, None, None, :]
    out = padded[torch.arange(b, device=dev)[:, None, None, None], torch.arange(c, device=dev)[None, :, None, None], rows, cols]
    flip = torch.rand(b, device=dev, generator=gen) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(3), out)


def cutout(x: torch.Tensor, size: int, gen: torch.Generator) -> torch.Tensor:
    """Zero a size x size square (clipped at borders) at a random centre in every image (DeVries & Taylor, 2017)."""
    b, _, h, w = x.shape
    dev = x.device
    cy = torch.randint(0, h, (b, 1), device=dev, generator=gen)
    cx = torch.randint(0, w, (b, 1), device=dev, generator=gen)
    half = size // 2
    ys = torch.arange(h, device=dev)[None, :]
    xs = torch.arange(w, device=dev)[None, :]
    in_y = (ys >= cy - half) & (ys < cy - half + size)
    in_x = (xs >= cx - half) & (xs < cx - half + size)
    mask = in_y[:, :, None] & in_x[:, None, :]
    return x * (~mask)[:, None].to(x.dtype)


class DeviceLoader:
    """Keeps the uint8 images on the target device and batches, normalizes and augments them there.

    This avoids a CPU data-loading bottleneck when many small models share one GPU.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, batch_size: int, device: torch.device, *,
                 shuffle: bool = False, augment: str = "none", cutout: int = 0, seed: int = 0,
                 drop_last: bool = False):
        self.x = torch.from_numpy(np.ascontiguousarray(x)).to(device)
        self.y = torch.from_numpy(np.ascontiguousarray(y)).to(device)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.cutout = cutout
        self.drop_last = drop_last
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)

    def __len__(self) -> int:
        n = len(self.y)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def __iter__(self):
        n = len(self.y)
        order = (torch.randperm(n, device=self.x.device, generator=self.gen) if self.shuffle
                 else torch.arange(n, device=self.x.device))
        for i in range(len(self)):
            idx = order[i * self.batch_size:(i + 1) * self.batch_size]
            xb = normalize(self.x[idx])
            if self.augment == "flipcrop":
                xb = random_crop_flip(xb, self.gen)
            elif self.augment != "none":
                raise ValueError(f"unknown augment {self.augment!r}")
            if self.cutout:
                xb = cutout(xb, self.cutout, self.gen)
            yield xb, self.y[idx]
