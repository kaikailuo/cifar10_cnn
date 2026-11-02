"""Post-training analysis: the only place the test set is used.

Runs after every selection is frozen: final test scores, ensembles, calibration (bonus 2)
and shift consistency (bonus 2). Writes <results>/analysis/*.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .data import DeviceLoader, load_cifar10
from .experiments import Context, cv_ranking, load_group
from .models import build_model
from .train import autocast_for, pick_device

SMOKE_TEST_LIMIT = 500
ECE_BINS = 15
MAX_SHIFT = 4


# ---- metrics on numpy logits -----------------------------------------------------------------

def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None))))


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(probs.argmax(1) == labels))


def reliability(probs: np.ndarray, labels: np.ndarray, bins: int = ECE_BINS) -> tuple[float, list[dict]]:
    """Expected calibration error with equal-width confidence bins, plus per-bin statistics."""
    conf = probs.max(1)
    correct = probs.argmax(1) == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if not mask.any():
            continue
        acc_b, conf_b = float(correct[mask].mean()), float(conf[mask].mean())
        ece += mask.mean() * abs(acc_b - conf_b)
        rows.append({"lo": float(lo), "hi": float(hi), "count": int(mask.sum()), "acc": acc_b, "conf": conf_b})
    return float(ece), rows


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature minimizing validation NLL (Guo et al., 2017), by a dense log-spaced grid search."""
    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 600))
    losses = [nll(softmax(logits, t), labels) for t in grid]
    return float(grid[int(np.argmin(losses))])


def shift_consistency(preds: np.ndarray) -> float:
    """Probability that two independently drawn shifts give the same prediction.

    preds has shape (num_shifts, num_images); computed exactly over all ordered pairs of distinct shifts.
    """
    s = preds.shape[0]
    counts = np.stack([(preds == c).sum(0) for c in range(10)])  # (10, N)
    return float(np.mean((counts * (counts - 1)).sum(0) / (s * (s - 1))))


# ---- model evaluation ------------------------------------------------------------------------

def translate(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Shift images by integer pixels, filling with zeros (the mean colour after normalization)."""
    out = torch.zeros_like(x)
    h, w = x.shape[-2:]
    src_y, dst_y = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    src_x, dst_x = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[..., dst_y, dst_x] = x[..., src_y, src_x]
    return out


class Evaluator:
    def __init__(self, ctx: Context, device: torch.device):
        self.ctx = ctx
        self.device = device
        _, _, test_x, test_y = load_cifar10()
        if ctx.smoke:
            test_x, test_y = test_x[:SMOKE_TEST_LIMIT], test_y[:SMOKE_TEST_LIMIT]
        self.labels = test_y
        self.loader = DeviceLoader(test_x, test_y, 1000, device)

    def model(self, run_dir: Path):
        cfg = json.loads((run_dir / "config.json").read_text())
        model = build_model(cfg).to(self.device, memory_format=torch.channels_last)
        model.load_state_dict(torch.load(run_dir / "model.pt", map_location=self.device))
        return model.eval()

    @torch.no_grad()
    def logits(self, run_dir: Path, dy: int = 0, dx: int = 0) -> np.ndarray:
        cache = run_dir / "test_logits.npz"
        if dy == 0 and dx == 0 and cache.exists():
            return np.load(cache)["logits"]
        model = self.model(run_dir)
        out = []
        for xb, _ in self.loader:
            if dy or dx:
                xb = translate(xb, dy, dx)
            with autocast_for(self.device):
                out.append(model(xb.contiguous(memory_format=torch.channels_last)).float().cpu())
        logits = torch.cat(out).numpy()
        if dy == 0 and dx == 0:
            np.savez_compressed(cache, logits=logits)
        return logits

    def scores(self, probs: np.ndarray) -> dict:
        return {"acc": accuracy(probs, self.labels), "nll": nll(probs, self.labels)}


def run(ctx: Context, device_name: str = "auto") -> dict:
    device = pick_device(device_name)
    ev = Evaluator(ctx, device)
    out_dir = ctx.results / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    rd = lambda group, name: ctx.results / group / name  # noqa: E731
    results: dict = {}

    # Final model selected by cross-validation, and the ensemble of its five fold models.
    ranking = cv_ranking(ctx)
    winner = ranking[0][0]
    (final_name,) = load_group(ctx, "e3_final")
    fold_runs = [rd("e3_cv", n) for n in load_group(ctx, "e3_cv") if n.startswith(f"{winner}__")]
    fold_probs = [softmax(ev.logits(d)) for d in fold_runs]
    results["final"] = {
        "winner": winner,
        "cv": [{"variant": v, "hp": hp, "fold_val_acc": accs, "mean": float(np.mean(accs)), "std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0}
               for v, hp, accs in ranking],
        "final_test": ev.scores(softmax(ev.logits(rd("e3_final", final_name)))),
        "fold_test": [ev.scores(p) for p in fold_probs],
        "cv_ensemble_test": ev.scores(np.mean(fold_probs, axis=0)),
    }

    # Bonus 1: cumulative ablation and an ensemble of the three best models by validation accuracy.
    b1 = load_group(ctx, "b1_ablation")
    b1_rows, b1_probs = {}, {}
    for name, (_, summary) in b1.items():
        b1_probs[name] = softmax(ev.logits(rd("b1_ablation", name)))
        b1_rows[name] = {"val_acc": summary["final_val_acc"], "params": summary["params"], "test": ev.scores(b1_probs[name])}
    top3 = sorted(b1, key=lambda n: -(b1[n][1]["final_val_acc"] or 0))[:3]
    results["b1"] = {"runs": b1_rows, "ensemble_members": top3,
                     "ensemble_test": ev.scores(np.mean([b1_probs[n] for n in top3], axis=0))}

    # Bonus 2: calibration before and after temperature scaling (temperature fitted on validation logits).
    calib = {}
    for name in b1:
        val = np.load(rd("b1_ablation", name) / "val_logits.npz")
        t = fit_temperature(val["logits"], val["labels"])
        logits = ev.logits(rd("b1_ablation", name))
        before, after = softmax(logits), softmax(logits, t)
        ece_b, bins_b = reliability(before, ev.labels)
        ece_a, bins_a = reliability(after, ev.labels)
        calib[name] = {"temperature": t, "acc": accuracy(before, ev.labels),
                       "nll": nll(before, ev.labels), "ece": ece_b, "bins": bins_b,
                       "nll_ts": nll(after, ev.labels), "ece_ts": ece_a, "bins_ts": bins_a,
                       "mean_conf": float(before.max(1).mean())}
    results["calibration"] = calib

    # Bonus 2: prediction stability under small translations, max pooling vs anti-aliased pooling.
    shifts = [(dy, dx) for dy in range(-MAX_SHIFT, MAX_SHIFT + 1) for dx in range(-MAX_SHIFT, MAX_SHIFT + 1)]
    shift_models = {"max-none": rd("b1_ablation", "b1-0-base__s0"), "max-flipcrop": rd("b1_ablation", "b1-1-flipcrop__s0"),
                    "blur-none": rd("b2_shift", "blur-none__s0"), "blur-flipcrop": rd("b2_shift", "blur-flipcrop__s0")}
    shift = {}
    for key, run_dir in shift_models.items():
        preds = np.stack([ev.logits(run_dir, dy, dx).argmax(1) for dy, dx in shifts])
        correct = preds == ev.labels[None, :]
        by_mag = {}
        for (dy, dx), c in zip(shifts, correct):
            by_mag.setdefault(max(abs(dy), abs(dx)), []).append(c.mean())
        shift[key] = {"clean_acc": float(correct[shifts.index((0, 0))].mean()),
                      "mean_shifted_acc": float(correct.mean()),
                      "acc_by_magnitude": {str(m): float(np.mean(v)) for m, v in sorted(by_mag.items())},
                      "consistency": shift_consistency(preds)}
    results["shift"] = shift

    (out_dir / "analysis.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"analysis written to {out_dir / 'analysis.json'}", flush=True)
    return results
