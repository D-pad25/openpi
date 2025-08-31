#!/bin/bash -l

# ─── Setup working directory ──────────────────────────────────────────────
 cd /home/n10813934/gitRepos/openpi  # Adjust path to your actual repo

# ─── Activate your virtual environment ────────────────────────────────────
 source .venv/bin/activate

# ─── Update enviroment path to cache direcotry  ───────────────────────────
 export OPENPI_DATA_HOME=$HOME/.cache/openpi