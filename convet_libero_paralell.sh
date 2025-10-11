#!/bin/bash -l
#PBS -N AGRIVLA_TO_LEROBOT
#PBS -l walltime=6:00:00
#PBS -l select=1:ncpus=12:ngpus=1:gpu_id=A100:mem=100gb
#PBS -J 1-8
#PBS -j oe
#PBS -o /home/n10813934/logs/agrivla_to_lerobot_${PBS_ARRAY_INDEX}.log

set -euo pipefail
echo "== Job start: $(date) on $(hostname) =="

# ─── Paths ──────────────────────────────────────────────────────────────────
REPO_DIR="/home/n10813934/gitRepos/openpi"
TFDS_DIR="/home/n10813934/data/tfds_datasets"
LOG_DIR="/home/n10813934/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/agrivla_to_lerobot_${PBS_ARRAY_INDEX}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

cd "$REPO_DIR"
source .venv/bin/activate
ulimit -n 4096 || true

# ─── Specs mapping ─────────────────────────────────────────────────────────
SPECS=(
  "all"
  "tomatoes_only"
  "tomatoes_plus:10"
  "tomatoes_plus:20"
  "tomatoes_plus:50"
  "tomatoes_plus:100"
  "tomatoes_plus:200"
  "chillis_only"
)
SPEC=${SPECS[$((PBS_ARRAY_INDEX-1))]}

echo "▶️  Converting spec: $SPEC"

uv run examples/libero/convert_libero.py \
  --data_dir "$TFDS_DIR" \
  --repo_prefix "dpad25/agrivla_pi0" \
  --specs "$SPEC" \
  --seed 42 \
  --clobber

echo "✅ Conversion complete for $SPEC at $(date)"