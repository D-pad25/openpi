#!/bin/bash -l

#PBS -N AGRIVLA_TO_LEROBOT
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=24:ngpus=1:gpu_id=A100:mem=32gb
#PBS -j oe
#PBS -o /home/n10813934/logs/agrivla_to_lerobot.$PBS_JOBID.log

set -euo pipefail

echo "== Job start: $(date) on $(hostname) =="

# ─── Paths ──────────────────────────────────────────────────────────────────
REPO_DIR="/home/n10813934/gitRepos/openpi"
TFDS_DIR="/home/n10813934/data/tfds_datasets"

# ─── Activate environment ───────────────────────────────────────────────────
cd "$REPO_DIR"
source .venv/bin/activate

# Allow lots of file handles for parallel image writers
ulimit -n 4096 || true

# ─── Run conversion (builds all specs by default) ───────────────────────────
# Adjust the script path below to where you saved the proposed Python script.
# (e.g., examples/agrivla/convert_agrivla_to_lerobot.py)
uv run examples/agrivla/convert_agrivla_to_lerobot.py \
  --data_dir "$TFDS_DIR" \
  --repo_prefix "dpad25/agrivla_pi0" \
  --seed 42
  # --push_to_hub  # uncomment only if you've logged in and want to push

echo "✅ LeRobot dataset conversion complete at $(date)"
