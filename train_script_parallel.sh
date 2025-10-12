#!/bin/bash -l
#PBS -N AGRIVLA_PI0_TRAIN
#PBS -l walltime=30:00:00
#PBS -l select=1:ncpus=12:ngpus=1:gpu_id=A100:mem=150gb
#PBS -J 1-8
#PBS -j oe
#PBS -o /home/n10813934/logs/agrivla_pi0_train_${PBS_ARRAY_INDEX}.log

set -euo pipefail
echo "== Job start: $(date) on $(hostname) =="

# ─── Paths ────────────────────────────────────────────────────────────────
REPO_DIR="/home/n10813934/gitRepos/openpi"
LOG_DIR="/home/n10813934/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/agrivla_pi0_train_${PBS_ARRAY_INDEX}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

cd "$REPO_DIR"
source .venv/bin/activate
ulimit -n 4096 || true

# ─── Environment setup ────────────────────────────────────────────────────
export OPENPI_DATA_HOME=$HOME/.cache/openpi
source ~/.wandb_secrets  # contains WANDB_API_KEY, etc.
# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# ─── Dataset / config mapping ─────────────────────────────────────────────
CONFIGS=(
  "pi0_lora_xarm6_agrivla_pi0_all"
  "pi0_lora_xarm6_agrivla_pi0_tomatoes_only"
  "pi0_lora_xarm6_agrivla_pi0_tomatoes_plus_10"
  "pi0_lora_xarm6_agrivla_pi0_tomatoes_plus_20"
  "pi0_lora_xarm6_agrivla_pi0_tomatoes_plus_50"
  "pi0_lora_xarm6_agrivla_pi0_tomatoes_plus_100"
  "pi0_lora_xarm6_agrivla_pi0_tomatoes_plus_200"
  "pi0_lora_xarm6_agrivla_pi0_chillis_only"
)
CONFIG=${CONFIGS[$((PBS_ARRAY_INDEX-1))]}

EXP_NAME="${CONFIG}_$(date +%Y%m%d_%H%M)"

echo "▶️  Computing normalization stats for config: $CONFIG"
uv run scripts/compute_norm_stats.py --config-name "$CONFIG"

echo "✅ Norm stats computed for $CONFIG"
echo "▶️  Starting training for config: $CONFIG"
echo "Experiment name: $EXP_NAME"

# ─── Run training ─────────────────────────────────────────────────────────
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py "$CONFIG" --exp-name="$EXP_NAME" --overwrite

echo "✅ Training complete for $CONFIG at $(date)"
