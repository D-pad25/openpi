#!/bin/bash -l

#PBS -N LEROBOT_CONVERSION
#PBS -l walltime=6:00:00
#PBS -l select=1:ncpus=12:ngpus=1:gpu_id=H100:mem=32gb

set -euo pipefail

# ─── Setup working directory ──────────────────────────────────────────────
cd /home/n10813934/gitRepos/openpi  # Adjust if repo path differs

# ─── Activate your virtual environment ────────────────────────────────────
source .venv/bin/activate

# ─── Run LeRobot conversion ───────────────────────────────────────────────
# Use uv to run the conversion script
uv run examples/libero/convert_libero_data_to_lerobot.py \
    --data_dir "/home/n10813934/data/tfds_datasets"

# ─── Completion message ──────────────────────────────────────────────────
echo "✅ LeRobot dataset conversion complete."

exit
