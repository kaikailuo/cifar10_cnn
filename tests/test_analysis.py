import numpy as np
import torch

from src.analysis import fit_temperature, reliability, shift_consistency, softmax, translate


def test_reliability_perfect_and_overconfident():
    labels = np.array([0, 1, 0, 1])
    probs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    assert reliability(probs, labels)[0] == 0.0
    wrong = probs[:, ::-1]
    assert np.isclose(reliability(wrong, labels)[0], 1.0)


def test_temperature_softens_overconfident_logits():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 10, 2000)
    logits = rng.normal(size=(2000, 10))
    logits[np.arange(2000), labels] += 1.0
    assert fit_temperature(logits * 5, labels) > 2.0


def test_shift_consistency_bounds():
    same = np.zeros((9, 5), dtype=int)
    assert shift_consistency(same) == 1.0
    alternating = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8]]).T.repeat(5, axis=1)
    assert shift_consistency(alternating) == 0.0


def test_translate_moves_pixels_and_fills_zero():
    x = torch.arange(16.0).view(1, 1, 4, 4)
    out = translate(x, 1, -1)
    assert out[0, 0, 1, 0] == x[0, 0, 0, 1]
    assert torch.all(out[0, 0, 0] == 0) and torch.all(out[0, 0, :, 3] == 0)
    assert softmax(np.zeros((1, 4))).sum() == 1.0
