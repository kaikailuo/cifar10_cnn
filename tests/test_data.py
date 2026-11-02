import numpy as np
import torch

from src.data import cutout, kfold_ids, load_cifar10, random_crop_flip, resolve_split, stratified_folds
from src.config import make_config


def test_stratified_folds_balanced():
    labels = np.repeat(np.arange(10), 50)
    folds = stratified_folds(labels, 5, seed=0)
    for f in range(5):
        assert np.all(np.bincount(labels[folds == f], minlength=10) == 10)


def test_splits_are_disjoint_and_complete():
    _, y, _, _ = load_cifar10()
    tr, va = resolve_split(make_config(split="holdout"), y)
    assert len(tr) == 45000 and len(va) == 5000 and not np.intersect1d(tr, va).size
    folds = kfold_ids(y)
    assert np.all(np.bincount(folds) == 10000)
    tr, va = resolve_split(make_config(split="fold", fold=2), y)
    assert len(np.union1d(tr, va)) == 50000 and not np.intersect1d(tr, va).size


def test_crop_flip_preserves_shape_and_content_range():
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(8, 3, 32, 32)
    out = random_crop_flip(x, gen)
    assert out.shape == x.shape
    # each output pixel is either an input pixel or padding zero
    assert set(out.flatten().tolist()) <= set(x.flatten().tolist()) | {0.0}


def test_cutout_zeroes_at_most_one_square():
    gen = torch.Generator().manual_seed(0)
    x = torch.ones(16, 3, 32, 32)
    out = cutout(x, 16, gen)
    zeroed = (out[:, 0] == 0).sum(dim=(1, 2))
    assert torch.all(zeroed > 0) and torch.all(zeroed <= 256)
