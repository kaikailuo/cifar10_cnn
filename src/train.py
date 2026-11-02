"""Train one run described by <run_dir>/config.json; resumable, writes history.csv and summary.json."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import DeviceLoader, load_cifar10, resolve_split
from .models import build_model, count_params

TRAIN_EVAL_SIZE = 5000  # fixed training subset scored in eval mode each epoch, for the generalization gap
EVAL_BATCH = 1000


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def autocast_for(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def predict(model, loader: DeviceLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return float32 logits and labels for every sample of an unshuffled loader."""
    model.eval()
    logits, labels = [], []
    for xb, yb in loader:
        with autocast_for(device):
            out = model(xb.contiguous(memory_format=torch.channels_last))
        logits.append(out.float())
        labels.append(yb)
    return torch.cat(logits), torch.cat(labels)


def score(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    loss = F.cross_entropy(logits, labels).item()
    acc = (logits.argmax(1) == labels).float().mean().item()
    return loss, acc


def lr_lambda(warmup_steps: int, total_steps: int):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return fn


def make_optimizer(model, cfg: dict) -> torch.optim.Optimizer:
    # No weight decay on normalization parameters and biases.
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    groups = [{"params": decay, "weight_decay": cfg["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.SGD(groups, lr=cfg["lr"], momentum=cfg["momentum"], nesterov=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_run(run_dir: Path, device: torch.device) -> dict:
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    cfg = json.loads((run_dir / "config.json").read_text())
    seed_everything(cfg["seed"])
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_x, train_y, _, _ = load_cifar10()
    train_idx, val_idx = resolve_split(cfg, train_y)
    loader = DeviceLoader(train_x[train_idx], train_y[train_idx], cfg["batch_size"], device, shuffle=True,
                          augment=cfg["augment"], cutout=cfg["cutout"], seed=cfg["seed"], drop_last=True)
    eval_idx = np.random.default_rng(0).choice(train_idx, size=min(TRAIN_EVAL_SIZE, len(train_idx)), replace=False)
    train_eval = DeviceLoader(train_x[eval_idx], train_y[eval_idx], EVAL_BATCH, device)
    val = DeviceLoader(train_x[val_idx], train_y[val_idx], EVAL_BATCH, device) if val_idx is not None else None

    model = build_model(cfg).to(device, memory_format=torch.channels_last)
    optimizer = make_optimizer(model, cfg)
    steps = len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda(cfg["warmup_epochs"] * steps, cfg["epochs"] * steps))

    history: list[dict] = []
    start_epoch = 0
    ckpt_path = run_dir / "checkpoint.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        loader.gen.set_state(ckpt["loader_gen"].to(loader.gen.get_state().device))
        torch.set_rng_state(ckpt["torch_rng"])
        if device.type == "cuda" and ckpt.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ckpt["cuda_rng"])
        history = ckpt["history"]
        start_epoch = len(history)
        print(f"resumed at epoch {start_epoch}", flush=True)

    diverged = False
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        t0 = time.perf_counter()
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        seen = 0
        for xb, yb in loader:
            with autocast_for(device):
                logits = model(xb.contiguous(memory_format=torch.channels_last))
            loss = F.cross_entropy(logits.float(), yb, label_smoothing=cfg["label_smoothing"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.detach() * len(yb)
            correct += (logits.detach().argmax(1) == yb).sum()
            seen += len(yb)
        train_loss = (loss_sum / seen).item()
        row = {"epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"], "train_loss": train_loss,
               "train_acc": (correct / seen).item()}
        if not math.isfinite(train_loss):
            diverged = True
        else:
            row["train_eval_loss"], row["train_eval_acc"] = score(*predict(model, train_eval, device))
            if val is not None:
                row["val_loss"], row["val_acc"] = score(*predict(model, val, device))
        if device.type == "cuda":
            torch.cuda.synchronize()
        row["epoch_time"] = time.perf_counter() - t0
        history.append(row)
        print(json.dumps(row), flush=True)
        if diverged:
            break
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "loader_gen": loader.gen.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None,
                    "history": history}, ckpt_path)

    with (run_dir / "history.csv").open("w", newline="") as handle:
        fields = list(dict.fromkeys(k for row in history for k in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)

    last = history[-1]
    summary = {
        "diverged": diverged,
        "epochs_run": len(history),
        "params": count_params(model),
        "train_time_s": sum(r["epoch_time"] for r in history),
        "final_train_eval_acc": last.get("train_eval_acc"),
        "final_train_eval_loss": last.get("train_eval_loss"),
        "final_val_acc": last.get("val_acc"),
        "final_val_loss": last.get("val_loss"),
        "device": str(device),
        "torch": torch.__version__,
    }
    if val is not None and not diverged:
        best = max(history, key=lambda r: r["val_acc"])
        summary["best_val_acc"], summary["best_epoch"] = best["val_acc"], best["epoch"]
        summary["gap"] = last["train_eval_acc"] - last["val_acc"]
        logits, labels = predict(model, val, device)
        np.savez_compressed(run_dir / "val_logits.npz", logits=logits.cpu().numpy(), labels=labels.cpu().numpy(),
                            indices=val_idx)
    if not diverged:
        torch.save(model.state_dict(), run_dir / "model.pt")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    ckpt_path.unlink(missing_ok=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads for torch (0 = torch default)")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    train_run(args.run_dir, pick_device(args.device))


if __name__ == "__main__":
    main()
