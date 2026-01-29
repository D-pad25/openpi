#!/usr/bin/env bash
set -euo pipefail

###############################################
# User configuration – EDIT THESE TO SUIT YOU
###############################################

# Local machine repo path (where xarm6_control lives)
LOCAL_REPO_DIR="$HOME/openpi" # If on local

# Virtual environment name (local)
VENV_DIR=".venv"

###############################################
# Helpers
###############################################

usage() {
  cat <<EOF
Usage: $0 <command>

Local xArm client side:
  cameras          Launch camera nodes (requires local venv)

General:
  help             Show this message

EOF
}

###############################################
# Commands
###############################################

cmd="${1:-help}"
shift || true

case "$cmd" in
  ################################################
  # LOCAL CLIENT SIDE (xArm machine)
  ################################################
  cameras)
    echo ">>> Launching camera nodes (local machine)..."
    cd "${LOCAL_REPO_DIR}"
    source "${VENV_DIR}/bin/activate"
    uv run src/xarm6_control/comms/zmq/launch_camera_nodes.py
    ;;

  help|*)
    usage
    ;;
esac
