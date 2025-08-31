#!/bin/bash -l

 #PBS -N EVAL

 #PBS -l walltime=18:00:00
 #PBS -l select=1:ncpus=4:ngpus=1:gpu_id=H100:mem=80gb

set -euo pipefail

# ─── Setup working directory ──────────────────────────────────────────────
 cd /home/n10813934/gitRepos/openpi  # Adjust path to your actual repo

# ─── Activate your virtual environment ────────────────────────────────────
 source .venv/bin/activate

# ─── Update enviroment path to cache direcotry  ───────────────────────────
 export OPENPI_DATA_HOME=$HOME/.cache/openpi

# ─── Add W&B environment variables ────────────────────────────────────────
source ~/.wandb_secrets

# ─── Run your model using uv and config name ──────────────────────────────
# Round 1 - 
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_xarm6_low_mem_finetune --exp-name=pi0_xarm6_lora_pickTomatoes_noFrozenFrames --overwrite

# Round 2 -
 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0base_lora_xarm6_round2_fulldataset --exp-name=pi0base_lora_xarm6_round2_fulldataset --overwrite

 exit
