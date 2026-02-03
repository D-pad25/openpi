# xarm6_control

Control stack for running OpenPI policies on a physical **xArm6** robot. This repository provides the tooling required to:

- Serve an OpenPI policy (locally or on an HPC GPU node)
- Stream **base** and **wrist** RealSense camera feeds to the client runtime
- Execute policy inference and robot actions on the real robot
- Control the gripper via either a **ROS TCP bridge** or **direct USB (Dynamixel)**

> **Safety notice:** This repository enables autonomous motion on a real robot. Use appropriate guarding, enforce workcell limits, validate behavior at low speed, and ensure a working E-stop and supervised bring-up.

---

## Table of Contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Repository Setup](#repository-setup)
- [Data Transformation (RLDS → LeRobot)](#data-transformation-rlds--lerobot)
- [Training](#training)
- [Dashboard Application](#dashboard-application)
- [Running Policies on xArm6 (CLI)](#running-policies-on-xarm6-cli)
- [Gripper Modes](#gripper-modes)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Notes for Client Delivery](#notes-for-client-delivery)

---

## Architecture

High-level data/control flow:

```text
                  ┌──────────────────────────┐
                  │        Dashboard         │
                  │  start/stop server, UI   │
                  └───────────┬──────────────┘
                      │───────|───────|
        Local mode    ▼               ▼  HPC mode (job + tunnel)
  ┌────────────────────────┐      ┌───────────────────────────┐
  │   Policy Server (WS)   │      │   Policy Server on HPC    │
  │ scripts/serve_policy.py│      │ scripts/serve_policy.py   │
  └───────────┬────────────┘      └─────────────┬─────────────┘
              │                                 |
              |<───────────SSH tunnel───────────│
              │                                  
              ▼                                 
      ┌─────────────────┐               ┌─────────────────┐
      │   xArm Client   │──────────────>│  Real Robot     │
      │  cli/main.py    │ control cmds  │ xArm6 + gripper│
      └───────┬─────────┘               └─────────────────┘
              │
              ▼
    ┌──────────────────────┐
    │ Camera pipeline (ZMQ)│
    │ base + wrist streams │
    └──────────────────────┘
```

---

## Prerequisites

### General

- Linux is recommended for real-robot runtime and camera streaming.
- Python environment managed via `uv`.
- Network access to the xArm controller (robot IP) and required ports.
- Intel RealSense SDK installed and permission to access camera devices.

### Optional components

- **ROS gripper mode**
  - ROS 1 environment
  - `roscore` available
  - `rosserial` (or equivalent microcontroller bridge) available and configured

- **HPC mode**
  - SSH access and ability to submit GPU jobs
  - The `openpi` repository and environment must exist on the HPC at:
    ```text
    /home/<username>/openpi
    /home/<username>/openpi/.venv
    ```

---

## Repository Setup

From the repository root:

```bash
uv sync
```

Unless otherwise stated, execute Python entrypoints using `uv run ...`.

For details on setup (i.e. setting up the repository on the HPC), please refer to the main [`README.md` Installation section](../../README.md#installation).

---

## Data Transformation (RLDS → LeRobot)

This repository includes scripts to convert RLDS-formatted datasets (e.g., TFDS) into **LeRobot** format compatible with OpenPI/pi0 workflows.

### Convert RLDS dataset to LeRobot

```bash
uv run examples/libero/convert_libero_data_to_lerobot.py   --data_dir /path/to/your/tfds_dataset
```

Notes:
- The conversion script writes to the standard Hugging Face/LeRobot cache location unless overridden.
- The script prints the final dataset destination path and identifier.

### Inspect the converted dataset

```bash
uv run examples/libero/inspect_lerobot_dataset.py   --repo_id <org_or_user>/<dataset_name>   --root /path/to/lerobot_dataset_root
```

Example placeholders:
- `--repo_id`: `myorg/agrivla_pick_tomatoes_v1`
- `--root`: path printed by the conversion script (commonly under `~/.cache/huggingface/lerobot/...`)

---

## Training

Training is configured via named config entries (e.g., Hydra-style `--config-name`).

1) Update your training configuration (dataset path, run name, output directories) in the relevant config module used by your setup (e.g., `config.py`).

2) Compute normalization statistics:

```bash
uv run scripts/compute_norm_stats.py --config-name <your_config_name>
```

Example:

```bash
uv run scripts/compute_norm_stats.py --config-name pi0base_lora_xarm6_round2_fulldataset
```

### Weights & Biases (optional)

If using W&B, load credentials via environment variables. A common pattern is to source a local secrets file:

```bash
source ~/.wandb_secrets
```

Example `~/.wandb_secrets`:

```bash
# W&B authentication (keep private; do not commit)
export WANDB_API_KEY="xxxxxxxx"
export WANDB_MODE="online"      # or "offline"
export WANDB_PROJECT="agrivla"
```

---

## Dashboard Application

The dashboard provides a single operator interface to:

- Start/stop the policy server (local or HPC)
- View server/orchestrator status
- Stream camera feeds
- Launch the xArm client with a specified prompt

### Dashboard prerequisites

- Local virtual environment created and dependencies installed (`uv sync`).
- Required ports available on the local machine:
  - Dashboard UI: **9000**
  - Policy server (local or forwarded): **8000**
  - Camera streams: wrist **5000**, base **5001**
- If using **ROS gripper mode**, `roscore` and your ROS bridge (e.g., `rosserial`) must be running.
- If using **HPC mode**, ensure the HPC repository layout matches the paths listed in [Prerequisites](#prerequisites).

### Launch

#### Step 1: Activate the virtual environment

```bash
source .venv/bin/activate
```

#### Step 2: Launch the dashboard application

```bash
uv run src/xarm6_control/dashboard/dashboard_app.py
```

#### Step 3: (ROS gripper mode only) Run the gripper server

```bash
python src/xarm6_control/hardware/gripper/server_async.py
```

#### Step 4: Open the dashboard in a web browser (or equivalent)

```text
http://localhost:9000
```

### Local mode

- Runs `scripts/serve_policy.py` on the same machine as the dashboard.
- Intended for workstation deployments where GPU and UI are co-located.
- By default, server health is checked on the configured endpoint (commonly `ws://localhost:8000`).

### HPC mode

- Submits a GPU job on the HPC environment.
- Waits for endpoint publication, then establishes an SSH tunnel back to the local machine.
- Requires valid HPC credentials and network access for tunneling.

### Model weights / Hugging Face cache

A typical server launch for Hugging Face–hosted artifacts:

```bash
uv run scripts/serve_policy.py --env HUGGING_FACE --port 8000
```

Cache location is controlled via:

```bash
export OPENPI_DATA_HOME="$HOME/.cache/openpi"
```

In managed HPC jobs, `OPENPI_DATA_HOME` may be set automatically by the job script/orchestrator.

### Known issues

#### Camera server not starting / only one camera visible

**Description**  
In some cases, the camera server may fail to start correctly or only a single camera stream (base *or* wrist) is visible in the dashboard.

**Cause**  
Camera processes from a previous run did not terminate cleanly, leaving ports bound. This prevents new camera server instances from starting.

**Affected ports**
- Wrist camera: **5000**
- Base camera: **5001**

**Resolution**  
Manually terminate the processes currently bound to the camera ports, then relaunch the camera nodes and refresh the dashboard.

```bash
lsof -i :5000
lsof -i :5001
kill -9 <PID>
```

---

## Running Policies on xArm6 (CLI)

This section provides an alternative method to run policies via the command line (instead of the dashboard).

### 1) Start the policy server (HPC)

```bash
ssh <username>@<hpc_host>
cd /home/<username>/openpi
source .venv/bin/activate
export OPENPI_DATA_HOME="$HOME/.cache/openpi"
uv run scripts/serve_policy.py --env HUGGING_FACE --port 8000
```

### 2) Create an SSH tunnel to the server node

From the client machine:

```bash
ssh -L 8000:<server_node_ip_or_hostname>:8000 <username>@<hpc_host>
```

This forwards the remote server endpoint to `localhost:8000` on the client machine.

### 3) Run the client (near the robot)

Activate the local environment:

```bash
source .venv/bin/activate
```

#### 3.1 Launch camera nodes

```bash
uv run src/xarm6_control/comms/zmq/launch_camera_nodes.py
```

#### 3.2 (ROS gripper mode only) Start the gripper server

```bash
python src/xarm6_control/hardware/gripper/server_async.py
```

#### 3.3 Run the client

Default prompt:

```bash
uv run src/xarm6_control/cli/main.py --remote_host localhost --remote_port 8000
```

Tomato example:

```bash
uv run src/xarm6_control/cli/main.py   --remote_host localhost --remote_port 8000   --prompt "Pick a ripe, red tomato and drop it in the blue bucket. [crop=tomato]"
```

Chilli example:

```bash
uv run src/xarm6_control/cli/main.py   --remote_host localhost --remote_port 8000   --prompt "Pick a red chilli and drop it in the blue bucket. [crop=chilli]"
```

---

## Gripper Modes

Two primary gripper control paths are supported.

### ROS mode (TCP ↔ ROS topics)

- Intended for systems where the gripper is controlled through ROS topics (often via a microcontroller bridge).
- Requires the TCP ↔ ROS bridge server:

```bash
python src/xarm6_control/hardware/gripper/server_async.py
```

### USB Dynamixel mode (direct control)

- Intended for setups where the gripper is directly connected over USB (Dynamixel).
- Does **not** require ROS or the gripper bridge server.
- Ensure the user has permission to access the USB device (e.g., `/dev/ttyUSB*`) and that baudrate/ID/protocol match the hardware configuration.

---

## Configuration

Common configuration surfaces include:

- **Ports**
  - Dashboard UI: `9000`
  - Policy server: `8000`
  - Camera streams: wrist `5000`, base `5001`

- **Caching**
  - `OPENPI_DATA_HOME` controls local/HPC cache storage for checkpoints and artifacts:
    ```bash
    export OPENPI_DATA_HOME="$HOME/.cache/openpi"
    ```

---

## Troubleshooting

### Dashboard starts but policy server is unreachable

- Confirm the policy server is running and reachable on the expected interface.
- In HPC mode, confirm the SSH tunnel is active and forwarding the correct server node address/port.
- Verify local port `8000` is not already in use.

### Cameras not streaming

- Confirm RealSense devices are visible (`rs-enumerate-devices`).
- Check udev permissions (Linux) and confirm the process has access to `/dev/bus/usb/...`.
- If running headless, confirm your pipeline does not depend on GUI-only backends.

### ROS gripper server errors

- Confirm `roscore` is running.
- Confirm your ROS device bridge is active and expected topics exist.
- Verify the TCP server port is available and not blocked by local firewall rules.
