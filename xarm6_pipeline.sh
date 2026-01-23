#!/usr/bin/env bash
set -euo pipefail

###############################################
# User configuration – EDIT THESE TO SUIT YOU
###############################################

# HPC details
HPC_USER="n10813934"
HPC_PREFIX="aqua"
HPC_HOST="aqua.qut.edu.au"
HPC_REPO_DIR="/home/n10813934/gitRepos/openpi"   # path to repo on HPC

# Local machine repo path (where xarm6_control lives)
LOCAL_REPO_DIR="$HOME/openpi"           # adjust as needed

# Dataset paths (on HPC)
DATA_DIR="/home/n10813934/data/tfds_datasets"
LEROBOT_ROOT="/home/n10813934/.cache/huggingface/lerobot/dpad25/agrivla_pick_tomatoes_v1"
REPO_ID="dpad25/agrivla_pick_tomatoes_v1"

# Training config
TRAIN_CONFIG="pi0base_lora_xarm6_round2_fulldataset"

# Policy server / tunnel settings
POLICY_PORT="8000"
REMOTE_POLICY_HOST="10.13.22.1"         # internal node for tunnel

# Virtual environment name (both HPC + local)
VENV_DIR=".venv"

###############################################
# Helpers
###############################################

usage() {
  cat <<EOF
Usage: $0 <command>

Data & training (HPC via ssh):
  convert-data     Convert RLDS -> LeRobot format
  inspect-data     Run inspect_lerobot_dataset.py
  train            Compute norm stats (and use config.py)

Policy server (HPC via ssh):
  serve-policy     Run scripts/serve_policy.py on HPC

Networking:
  tunnel           Create SSH tunnel: localhost:${POLICY_PORT}  ->${REMOTE_POLICY_HOST}:${POLICY_PORT}

Local xArm client side:
  cameras          Launch camera nodes (requires local venv)
  gripper          Run gripper server (outside venv)
  client-default   Run client with default prompt
  client-tomato    Run client with tomato prompt
  client-chilli    Run client with chilli prompt

General:
  help             Show this message

EOF
}

run_hpc_cmd() {
  # Start an interactive qsub session and run the given command inside it
  ssh "${HPC_PREFIX}" << EOF
qsub -I -S /bin/bash \
    -l select=1:ncpus=12:ngpus=1:mem=64gb:gpu_id=H100 \
    -l walltime=4:00:00 << 'INNER_EOF'
cd '${HPC_REPO_DIR}'
source '${VENV_DIR}/bin/activate'
$1
INNER_EOF
EOF
}

###############################################
# Commands
###############################################

cmd="${1:-help}"
shift || true

case "$cmd" in
  ################################################
  # DATA & TRAINING (HPC)
  ################################################
  convert-data)
    echo ">>> Converting RLDS data -> LeRobot on HPC..."
    run_hpc_cmd "uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir '${DATA_DIR}'"
    echo ">>> Done. Output should be in: ${LEROBOT_ROOT}"
    ;;

  inspect-data)
    echo ">>> Inspecting LeRobot dataset on HPC..."
    run_hpc_cmd "uv run examples/libero/inspect_lerobot_dataset.py \
      --repo_id '${REPO_ID}' \
      --root '${LEROBOT_ROOT}'"
    ;;

  train)
    echo ">>> Running compute_norm_stats on HPC using config: ${TRAIN_CONFIG}"
    # If you use W&B secrets, they should live in ~/.wandb_secrets on the HPC
    run_hpc_cmd "source ~/.wandb_secrets 2>/dev/null || true; \
      uv run scripts/compute_norm_stats.py --config-name '${TRAIN_CONFIG}'"
    ;;

  ################################################
  # POLICY SERVER (HPC)
  ################################################
  serve-policy)
    echo ">>> Starting policy server on HPC (port ${POLICY_PORT})..."
    run_hpc_cmd "export OPENPI_DATA_HOME=\$HOME/.cache/openpi; \
      uv run scripts/serve_policy.py --env HUGGING_FACE --port ${POLICY_PORT}"
    # This will keep running in the SSH session until you stop it.
    ;;

  ################################################
  # TUNNEL (LOCAL)
  ################################################
  tunnel)
    echo ">>> Creating SSH tunnel localhost:${POLICY_PORT} -> ${REMOTE_POLICY_HOST}:${POLICY_PORT} via ${HPC_HOST}..."
    echo ">>> Leave this command running while you use the client."
    ssh -L "${POLICY_PORT}:${REMOTE_POLICY_HOST}:${POLICY_PORT}" "${HPC_USER}@${HPC_HOST}"
    ;;

  ################################################
  # LOCAL CLIENT SIDE
  ################################################
  cameras)
    echo ">>> Launching camera nodes (local machine)..."
    cd "${LOCAL_REPO_DIR}"
    source "${VENV_DIR}/bin/activate"
    uv run src/xarm6_control/zmq_core/launch_camera_nodes.py
    ;;

  gripper)
    echo ">>> Running gripper server (local, outside venv recommended)..."
    cd "${LOCAL_REPO_DIR}"
    # Intentionally NOT activating venv here, per your note
    python src/xarm6_control/hardware/gripper/server_async.py
    ;;

  client-default)
    echo ">>> Running xArm client (default prompt)..."
    cd "${LOCAL_REPO_DIR}"
    source "${VENV_DIR}/bin/activate"
    uv run src/xarm6_control/main2.py \
      --remote_host localhost \
      --remote_port "${POLICY_PORT}"
    ;;

  client-tomato)
    echo ">>> Running xArm client (tomato prompt)..."
    cd "${LOCAL_REPO_DIR}"
    source "${VENV_DIR}/bin/activate"
    uv run src/xarm6_control/main2.py \
      --remote_host localhost \
      --remote_port "${POLICY_PORT}" \
      --prompt "Pick a ripe, red tomato and drop it in the blue bucket. [crop=tomato]"
    ;;

  client-chilli)
    echo ">>> Running xArm client (chilli prompt)..."
    cd "${LOCAL_REPO_DIR}"
    source "${VENV_DIR}/bin/activate"
    uv run src/xarm6_control/main2.py \
      --remote_host localhost \
      --remote_port "${POLICY_PORT}" \
      --prompt "Pick a red chilli and drop it in the blue bucket. [crop=chilli]"
    ;;

  help|*)
    usage
    ;;
esac
