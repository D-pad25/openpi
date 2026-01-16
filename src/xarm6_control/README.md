# xarm6_control

This module provides real-time control and observation utilities for the xArm6 robot, enabling integration with OpenPI Vision-Language-Action (VLA) policies. It allows the xArm6 to run inference-based actions from pre-trained models using real sensor inputs and live camera feeds.

---

## 🔧 Features

- Retrieve joint states and camera observations from the xArm6 robot
- Format observations for OpenPI policies (e.g. pi0, xarm6_policy)
- Send actions to the xArm6 controller via `position`, `velocity`, or `servo` mode
- Support for real-world deployment and debug-friendly logging

---

## Data Transformation

### How to transform data from RLDS into Lerobot
To transform the data, run:
```bash
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /home/n10813934/data/tfds_datasets
```

This will convert the data stored in RLDS into pi0's lerobot format. The resuling dataset will be stored in:
`
/home/n10813934/.cache/huggingface/lerobot/dpad25/agrivla_pick_tomatoes_v1
`

### How to confirm data transform
A custom script has been developed (`inspect_lerobot_dataset.py`). To run this script, you will need to pass in the repo_id and the root directory for the dataset you have just created. For this example, I ran:
```bash
uv run examples/libero/inspect_lerobot_dataset.py \
  --repo_id dpad25/agrivla_pick_tomatoes_v1 \
  --root /home/n10813934/.cache/huggingface/lerobot/dpad25/agrivla_pick_tomatoes_v1
```

## Training
When you want to train, make sure to first update the traininig config located in `config.py` to include you updated name and dataset path. 

Then run:
```bash
uv run scripts/compute_norm_stats.py --config-name pi0base_lora_xarm6_round2_fulldataset
```

Replacing `pi0base_lora_xarm6_round2_fulldataset` with the name of your config

### Weights and Biases
The following line exists in the train basch script:
```bash
source ~/.wandb_secrets
```

This line gets the following enviroment variables:

```bash
cat <<'EOF' > ~/.wandb_secrets
# W&B authentication (private)
export WANDB_API_KEY=xxxxxxxx (can be found at [text](https://wandb.ai/authorize))
export WANDB_MODE=online
export WANDB_PROJECT=agrivla
EOF
```

## 🚀 Running a Policy on xArm6

### 1. Run the Policy Server on the HPC

#### 1.1 ssh into the HPC

```bash
ssh n10813934@aqua.qut.edu.au
```

#### 1.2 Activate the virtual environment

```bash
source .venv/bin/activate
```

#### 1.3 Export the cache location to access checkpoints etc.

```bash
export OPENPI_DATA_HOME=$HOME/.cache/openpi 
```
 
#### 1.4 Run the policy. Note that the configuration (checkpoints, norm stats etc.) are governed by the --env key

```bash
uv run scripts/serve_policy.py --env XARM --port 8000
```

### 2. Set up SSH tunnel from client to the HPC server node

```bash
ssh -L 8000:10.13.22.1:8000 n10813934@aqua.qut.edu.au
```

### 3. Run the client (make relative path)

#### 3.1 Activate virtual enviroment

```bash
source .venv/bin/activate
```

#### 3.2 Launch camera nodes
> - **NOTE**: You need to be in the virtual enviroment to run this
```bash
uv run src/xarm6_control/comms/zmq/launch_camera_nodes.py
```

#### 3.3 Run the gripper server
> - **NOTE**: You need to be outside virtual enviroment to run this

```bash
python src/xarm6_control/hardware/gripper/server_async.py
```

#### 3.4 Run the Client script with default prompt
##### To run with default propmt
```bash
uv run src/xarm6_control/cli/main.py --remote_host localhost --remote_port 8000
```

##### To run with tomato propmt
```bash
uv run src/xarm6_control/cli/main.py --remote_host localhost --remote_port 8000 --prompt "Pick a ripe, red tomato and drop it in the blue bucket. [crop=tomato]"
```

##### To run with tomato propmt
```bash
uv run src/xarm6_control/cli/main.py --remote_host localhost --remote_port 8000 --prompt "Pick a red chilli and drop it in the blue bucket. [crop=chilli]"
```