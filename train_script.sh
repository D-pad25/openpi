#!/bin/bash -l

 #PBS -N EVAL

 #PBS -l walltime=48:00:00
 #PBS -l select=1:ncpus=12:ngpus=1:gpu_id=A100:mem=200gb
 #PBS -o /home/n10813934/logs/agrivla_pi0FAST_train_.log

 set -euo pipefail
 echo "== Job start: $(date) on $(hostname) =="
# ─── Setup working directory ──────────────────────────────────────────────
 REPO_DIR="/home/n10813934/gitRepos/openpi"
 LOG_DIR="/home/n10813934/logs"
 mkdir -p "$LOG_DIR"
 LOG_FILE="$LOG_DIR/agrivla_pi0FAST_train_$(date +%Y%m%d_%H%M).log"
 exec > >(tee -a "$LOG_FILE") 2>&1
 echo "Logging to $LOG_FILE"

 cd "$REPO_DIR"

# ─── Activate your virtual environment ────────────────────────────────────
 source .venv/bin/activate
 ulimit -n 4096 || true
# ─── Update enviroment path to cache direcotry  ───────────────────────────
 export OPENPI_DATA_HOME=$HOME/.cache/openpi

# ─── Add W&B environment variables ────────────────────────────────────────
source ~/.wandb_secrets

# ─── Run your model using uv and config name ──────────────────────────────
# Round 2 -
 CONFIG="pi0_fast_fullfinetune_xarm6_agrivla_pi0_all"
 EXP_NAME="${CONFIG}_$(date +%Y%m%d_%H%M)"

 # Run normalization stats computation
#  echo "▶️  Computing normalization stats for config: $CONFIG"
#  uv run scripts/compute_norm_stats.py --config-name "$CONFIG"
#  echo "✅ Norm stats computed for $CONFIG"

 # Train
 echo "▶️  Starting training for config: $CONFIG"
 echo "Experiment name: $EXP_NAME"
 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py "$CONFIG" --exp-name="$EXP_NAME" --overwrite

 exit
