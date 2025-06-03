#!/bin/bash
# run_main2.sh

# Activate your virtual environment
source .venv/bin/activate

# Source the ROS Noetic setup
source /opt/ros/noetic/setup.bash

# Extend PYTHONPATH so the script can find both local code and rospy
export PYTHONPATH=.:$PYTHONPATH:/opt/ros/noetic/lib/python3/dist-packages

# Run your ROS-based script
python xarm6_control/main2.py --remote_host localhost --remote_port 8000