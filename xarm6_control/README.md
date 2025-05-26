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
