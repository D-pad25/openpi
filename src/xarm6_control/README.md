# xarm6_control

This module provides real-time control and observation utilities for the xArm6 robot, enabling integration with OpenPI Vision-Language-Action (VLA) policies. It allows the xArm6 to run inference-based actions from pre-trained models using real sensor inputs and live camera feeds.

---

## 🔧 Features

- Retrieve joint states and camera observations from the xArm6 robot
- Format observations for OpenPI policies (e.g. pi0, xarm6_policy)
- Send actions to the xArm6 controller via `position`, `velocity`, or `servo` mode
- Support for real-world deployment and debug-friendly logging

---

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

#### 3.1 Set up enviroment

- This step activates the local Python environment and configures ROS Noetic support.  
- It ensures that ROS-related Python modules (e.g., `rospy`) are accessible, and that the workspace is correctly set up for ROS-integrated inference and control.

```bash
source .venv/bin/activate
source /opt/ros/noetic/setup.bash
export PYTHONPATH=$PYTHONPATH:/opt/ros/noetic/lib/python3/dist-packages
```

#### 3.2 Run the gripper server

> **Note:** I need to test the `gripper_server_async`.  
> When testing this, make sure to also change which class is referenced in `xarm_env.py`.


```bash
uv run src/xarm6_control/gripper_server
```

#### 3.2 (a) Run the script (refferencing local dir and rospy)

```bash
PYTHONPATH=.:/opt/ros/noetic/lib/python3/dist-packages uv run src/xarm6_control/main2.py --remote_host localhost --remote_port 8000
```

#### 3.2 (b) You should be able to just run this now

```bash
uv run src/xarm6_control/main2.py --remote_host localhost --remote_port 8000
```