"""Run experiment groups phase by phase, several runs in parallel, then analysis and report assets.

    python -m src.run --smoke --jobs 2          # tiny end-to-end check on CPU
    python -m src.run --jobs 4                  # everything (GPU)
    python -m src.run --groups e1_dropout,e2_norm --jobs 4
    python -m src.run --analyze-only

Finished runs (summary.json present) are skipped and interrupted runs resume from their checkpoint,
so the same command can simply be re-issued after an interruption.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .config import smoke_version
from .data import ROOT
from .experiments import GROUPS, Context, MissingResults


def results_dir(smoke: bool) -> Path:
    return ROOT / ("results_smoke" if smoke else "results")


def prepare(ctx: Context, groups: list[str]) -> list[Path]:
    """Write config.json for every run of the groups; return the run dirs that still need training."""
    pending = []
    for group in groups:
        for name, cfg in GROUPS[group][1](ctx).items():
            if ctx.smoke:
                cfg = smoke_version(cfg)
            run_dir = ctx.results / group / name
            config_path = run_dir / "config.json"
            if config_path.exists():
                if json.loads(config_path.read_text()) != cfg:
                    raise RuntimeError(f"{config_path} differs from the current definition; delete the run to redo it")
            else:
                run_dir.mkdir(parents=True, exist_ok=True)
                config_path.write_text(json.dumps(cfg, indent=2) + "\n")
            if not (run_dir / "summary.json").exists():
                pending.append(run_dir)
    return pending


def execute(run_dirs: list[Path], jobs: int, device: str) -> list[Path]:
    threads = max(1, (os.cpu_count() or 1) // jobs)

    def launch(run_dir: Path) -> tuple[Path, int, float]:
        t0 = time.perf_counter()
        cmd = [sys.executable, "-m", "src.train", str(run_dir), "--device", device, "--threads", str(threads)]
        with (run_dir / "train.log").open("a") as log:
            code = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        return run_dir, code, time.perf_counter() - t0

    failed = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(launch, d) for d in run_dirs]
        for i, future in enumerate(as_completed(futures), 1):
            run_dir, code, seconds = future.result()
            status = "ok" if code == 0 else f"FAILED (exit {code}, see train.log)"
            print(f"[{i}/{len(run_dirs)}] {run_dir.parent.name}/{run_dir.name} {status} {seconds / 60:.1f} min", flush=True)
            if code != 0:
                failed.append(run_dir)
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups", help="comma-separated group names (default: all, in phase order)")
    parser.add_argument("--jobs", type=int, default=1, help="runs trained concurrently")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true", help="tiny data and 2 epochs, results in results_smoke/")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--no-analyze", action="store_true")
    args = parser.parse_args()

    ctx = Context(results_dir(args.smoke), args.smoke)
    selected = args.groups.split(",") if args.groups else list(GROUPS)
    unknown = set(selected) - set(GROUPS)
    if unknown:
        parser.error(f"unknown groups: {sorted(unknown)}")

    if not args.analyze_only:
        for phase in sorted({GROUPS[g][0] for g in selected}):
            groups = [g for g in selected if GROUPS[g][0] == phase]
            try:
                pending = prepare(ctx, groups)
            except MissingResults as err:
                sys.exit(f"phase {phase} ({', '.join(groups)}) needs earlier results: {err}")
            print(f"== phase {phase}: {', '.join(groups)} — {len(pending)} runs to train", flush=True)
            failed = execute(pending, args.jobs, args.device)
            if failed:
                sys.exit(f"{len(failed)} runs failed in phase {phase}; fix and re-run the same command")

    if not args.no_analyze and (args.analyze_only or not args.groups):
        from . import analysis, report
        analysis.run(ctx, args.device)
        report.build(ctx)


if __name__ == "__main__":
    main()
