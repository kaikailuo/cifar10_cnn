#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/cifar10_cnn
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
mkdir -p preparation/formal_logs
log="preparation/formal_logs/run_$(date +%Y%m%d_%H%M%S).log"
printf '%s\n' "$log" > preparation/formal_logs/latest_log_path.txt
/root/miniconda3/bin/python -u -m src.run --jobs 6 --device cuda 2>&1 | tee "$log"
