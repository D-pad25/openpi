#!/bin/bash -l

 #PBS -N EVAL

 #PBS -l walltime=12:00:00
 #PBS -l select=1:ncpus=4:ngpus=1:gpu_id=H100:mem=64gb

# ─── Setup working directory ──────────────────────────────────────────────
 cd /home/n10813934/gitRepos/openpi  # Adjust path to your actual repo

# ─── Activate your virtual environment ────────────────────────────────────
 source .venv/bin/activate

# ─── Update enviroment path to cache direcotry  ───────────────────────────
 export OPENPI_DATA_HOME=$HOME/.cache/openpi

# ─── Log in to weights and biases to monitor progrss  ─────────────────────
 wandb login
# ─── Run your model using uv and config name ──────────────────────────────
 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_xarm6_low_mem_finetune --exp-name=pi0_xarm6_lora_pickTomatoes_noFrozenFrames --overwrite

 exit
