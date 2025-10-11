#!/bin/bash -l

#PBS -N AGRIVLA_TO_LEROBOT
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=24:ngpus=1:gpu_id=A100:mem=200gb
#PBS -j oe
#PBS -o /home/n10813934/logs/agrivla_to_lerobot.log

set -euo pipefail

echo "== Job start: $(date) on $(hostname) =="

# ─── Paths ──────────────────────────────────────────────────────────────────
REPO_DIR="/home/n10813934/gitRepos/openpi"
TFDS_DIR="/home/n10813934/data/tfds_datasets"

# ─── Activate environment ───────────────────────────────────────────────────
cd "$REPO_DIR"
source .venv/bin/activate

# ─── Define log path dynamically ───────────────────────────────
LOG_DIR="/home/n10813934/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/agrivla_to_lerobot.${PBS_JOBID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

# Allow lots of file handles for parallel image writers
ulimit -n 4096 || true

# ─── Run conversion (builds all specs by default) ───────────────────────────
uv run examples/libero/convert_libero.py \
  --data_dir "$TFDS_DIR" \
  --repo_prefix "dpad25/agrivla_pi0" \
  --seed 42
  # --push_to_hub  # uncomment only if you've logged in and want to push

echo "✅ LeRobot dataset conversion complete at $(date)"
