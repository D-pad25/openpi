# xarm6_control

This module provides real-time control and observation utilities for the xArm6 robot, enabling integration with OpenPI Vision-Language-Action (VLA) policies. It allows the xArm6 to run inference-based actions from pre-trained models using real sensor inputs and live camera feeds.

## 🔧 Features

- Retrieve joint states and camera observations from the xArm6 robot
- Format observations for OpenPI policies (e.g. pi0, xarm6_policy)
- Send actions to the xArm6 controller via `position`, `velocity`, or `servo` mode
- Support for real-world deployment and debug-friendly logging

## 📁 Folder Structure
xarm6_control/
│
├── controller.py # Action execution utilities (e.g. apply_joint_action)
├── observation.py # Camera and joint state capture
├── main.py # Entry-point to run a policy on real robot
└── README.md # This file

## Get the policy running
1. Get the server up on running on the HPC
1.1 Activate the virtual enviroment
```bash
source .venv/bin/activate
```

1.2 Export the cache location to access checkpoints etc.
```bash
export OPENPI_DATA_HOME=$HOME/.cache/openpi 
```
 
1.3 Run the policy. Note that the configuration (checkpoints, norm stats etc.) are governed by the --env key
```bash
uv run scripts/serve_policy.py --env XARM --port 8000
```

2. Set up SSH tunnel from client to the HPC server node
```bash
ssh -L 8080:10.13.22.1:8080 n10813934@aqua.qut.edu.au
```

3. Run the client 
```bash
uv run xarm6_control/main2.py --remote_host localhost --remote_port 8080
```