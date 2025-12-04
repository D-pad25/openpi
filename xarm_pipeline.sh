#!/usr/bin/env bash
set -euo pipefail

###############################################
# User configuration – EDIT THESE TO SUIT YOU
###############################################

# HPC details
HPC_USER="n10813934"
# This should match your SSH alias or explicit user@host
# e.g. "aqua" if you have an SSH config entry, or "n10813934@aqua.qut.edu.au"
HPC_PREFIX="aqua"
HPC_HOST="aqua.qut.edu.au"
HPC_REPO_DIR="/home/n10813934/gitRepos/openpi"   # path to repo on HPC

# Local machine repo path (where xarm6_control lives)
LOCAL_REPO_DIR="$HOME/openpi"           # adjust to e.g. "$HOME/Thesis/openpi" if needed

# Dataset paths (on HPC)
DATA_DIR="/home/n10813934/data/tfds_datasets"
LEROBOT_ROOT="/home/n10813934/.cache/huggingface/lerobot/dpad25/agrivla_pick_tomatoes_v1"
REPO_ID="dpad25/agrivla_pick_tomatoes_v1"

# Training config
TRAIN_CONFIG="pi0base_lora_xarm6_round2_fulldataset"

# Policy server / tunnel settings
POLICY_PORT="8000"

# Virtual environment name (both HPC + local)
VENV_DIR=".venv"

###############################################
# Helpers
###############################################

usage() {
  cat <<EOF
Usage: $0 <command>

Data & training (HPC via qsub on GPU node):
  convert-data     Convert RLDS -> LeRobot format (GPU job)
  inspect-data     Run inspect_lerobot_dataset.py (GPU job)
  train            Compute norm stats (and use config.py) (GPU job)

Policy server (HPC via qsub on GPU node):
  serve-policy     Run scripts/serve_policy.py on GPU node (job; prints PBS job ID)

Networking:
  tunnel           Create SSH tunnel to the current policy server job (uses ~/.openpi_policy_endpoint)

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

# Submit a GPU job on the HPC via qsub and run the given command inside the job.
# The command is passed as a single string (job_cmd).
run_hpc_cmd() {
  local job_cmd="$1"

  ssh "${HPC_PREFIX}" "export JOB_CMD=\"$job_cmd\" HPC_REPO_DIR='${HPC_REPO_DIR}' VENV_DIR='${VENV_DIR}'; qsub -V << 'EOF'
#!/bin/bash
#PBS -N openpi_cmd
#PBS -l select=1:ncpus=12:ngpus=1:mem=64gb:gpu_id=H100
#PBS -l walltime=04:00:00
#PBS -j oe

set -euo pipefail

cd \"\$HPC_REPO_DIR\"
source \"\$VENV_DIR/bin/activate\"

NODE_HOST=\$(hostname)
NODE_IP=$(hostname -I | tr ' ' '\n' | grep '^10\.' | head -n 1)

echo \"[openpi_cmd] Running on node: \$NODE_HOST with IP: \$NODE_IP\"
echo \"[openpi_cmd] Using Python: \$(which python)\"
echo \"[openpi_cmd] Starting command: \$JOB_CMD\"

# If this is the policy server job, record endpoint for the tunnel
if [[ \"\$JOB_CMD\" == *\"serve_policy.py\"* ]]; then
  echo \"HOST=\$NODE_HOST\" >  \"\$HOME/.openpi_policy_endpoint\"
  echo \"IP=\$NODE_IP\"     >> \"\$HOME/.openpi_policy_endpoint\"
  echo \"PORT=8000\"        >> \"\$HOME/.openpi_policy_endpoint\"
fi

eval \"\$JOB_CMD\"

echo \"[openpi_cmd] Command finished.\"
EOF"
}

###############################################
# Commands
###############################################

cmd="${1:-help}"
shift || true

case "$cmd" in
  ################################################
  # DATA & TRAINING (HPC via GPU job)
  ################################################
  convert-data)
    echo ">>> Submitting RLDS -> LeRobot conversion as GPU job on HPC..."
    run_hpc_cmd "uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir '${DATA_DIR}'"
    echo ">>> Job submitted. Output dataset should end up in: ${LEROBOT_ROOT}"
    ;;

  inspect-data)
    echo ">>> Submitting LeRobot dataset inspection as GPU job on HPC..."
    run_hpc_cmd "uv run examples/libero/inspect_lerobot_dataset.py --repo_id '${REPO_ID}' --root '${LEROBOT_ROOT}'"
    ;;

  train)
    echo ">>> Submitting compute_norm_stats as GPU job on HPC using config: ${TRAIN_CONFIG}"
    run_hpc_cmd "source ~/.wandb_secrets 2>/dev/null || true; uv run scripts/compute_norm_stats.py --config-name '${TRAIN_CONFIG}'"
    ;;

  ################################################
  # POLICY SERVER (HPC via GPU job)
  ################################################
  serve-policy)
    echo ">>> Submitting policy server as GPU job on HPC (port ${POLICY_PORT})..."
    run_hpc_cmd "export OPENPI_DATA_HOME=\$HOME/.cache/openpi; uv run scripts/serve_policy.py --env XARM --port ${POLICY_PORT}"
    echo ">>> Policy server job submitted. Once it's running, you can start the tunnel."
    ;;

  ################################################
  # TUNNEL (LOCAL)
  ################################################
  tunnel)
    echo ">>> Discovering policy server endpoint on HPC..."
    endpoint=$(ssh "${HPC_PREFIX}" "cat ~/.openpi_policy_endpoint 2>/dev/null || echo ''")

    if [[ -z "$endpoint" ]]; then
      echo "!!! No policy endpoint file found on HPC (~/.openpi_policy_endpoint)."
      echo "    Make sure a serve-policy job is running first."
      exit 1
    fi

    # parse IP and PORT from the file
    remote_ip=$(echo "$endpoint"  | awk -F= '/^IP=/{print $2}')
    remote_port=$(echo "$endpoint" | awk -F= '/^PORT=/{print $2}')

    if [[ -z "$remote_ip" || -z "$remote_port" ]]; then
      echo "!!! Could not parse IP/PORT from ~/.openpi_policy_endpoint"
      echo "Content was:"
      echo "$endpoint"
      exit 1
    fi

    echo ">>> Creating SSH tunnel localhost:${remote_port} -> ${remote_ip}:${remote_port} via ${HPC_HOST}..."
    echo ">>> Leave this command running while you use the client."
    ssh -L "${remote_port}:${remote_ip}:${remote_port}" "${HPC_PREFIX}"
    ;;

  ################################################
  # LOCAL CLIENT SIDE (xArm machine)
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
    python src/xarm6_control/gripper_server_async_v2.py
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
