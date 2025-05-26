# xarm_env.py

import numpy as np
import os
import time
import pickle
from xarm.wrapper import XArmAPI

# JOINT_LIMITS = {
#     "lower": np.radians([-360, -118, -225, -360,  97, -360]),
#     "upper": np.radians([ 360,  120,   11,  360, 180,  360])
# }


class XArmRealEnv:
    def __init__(self, ip="192.168.1.203", camera_dict=None):
        self.arm = XArmAPI(ip, is_radian=True)
        self._initialize_arm()
        self.camera_dict = camera_dict or {}

    def _initialize_arm(self):
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(True)
        self.arm.set_mode(1)
        self.arm.set_collision_sensitivity(0)
        self.arm.set_state(state=0)

    def _get_joint_position(self):
        code, joint_position = self.arm.get_servo_angle(is_radian=True)
        while code != 0 or joint_position is None:
            print(f"[WARN] get_servo_angle() failed with code {code}. Retrying...")
            self._initialize_arm()
            code, joint_position = self.arm.get_servo_angle(is_radian=True)
        return joint_position

    def _get_gripper_position(self):
        code, gripper_pos = self.arm.get_gripper_position()
        while code != 0 or gripper_pos is None:
            print(f"[WARN] get_gripper_position() failed with code {code}. Retrying...")
            self._initialize_arm()
            code, gripper_pos = self.arm.get_gripper_position()
        return gripper_pos

    def get_observation(self):
        joint_position = self._get_joint_position()
        # gripper_pos = self._get_gripper_position()
        # gripper_pos = (gripper_pos - 800) / (0 - 800)  # Normalize to [0, 1]

        obs = {
            "joint_position": np.array(joint_position[:6]),  # Keep only 6 joints
            # "gripper_position": np.array([gripper_pos]),
            "gripper_position": np.array([0]),  # Use a placeholder for gripper position
        }

         # Build state with 6 joints + 1 gripper
        obs["state"] = np.concatenate([obs["joint_position"], obs["gripper_position"]])

        # Include camera observations if available
        for name, camera in self.camera_dict.items():
            image, depth = camera.read()
            obs[f"{name}_rgb"] = image
            obs[f"{name}_depth"] = depth

        return obs
    
    def get_frames(self):
        frames = {}
        for name, camera in self.camera_dict.items():
            image, depth = camera.read()
            frames[f"{name}_rgb"] = image
            frames[f"{name}_depth"] = depth
        return frames
    
    def generate_joint_trajectory(self, current_angles, target_angles, max_delta):
        current_angles = np.array(current_angles)
        target_angles = np.array(target_angles)
        
        # Calculate the maximum number of steps needed for any joint
        deltas = np.abs(target_angles - current_angles)
        num_steps = int(np.max(deltas / max_delta))
        
        # If no step is needed, just return the target
        if num_steps == 0:
            return [target_angles.tolist()]

        # Linearly interpolate between current and target for each joint
        trajectory = [
            (current_angles + (target_angles - current_angles) * step / num_steps).tolist()
            for step in range(1, num_steps + 1)
        ]
        return trajectory

    def step(self, action: np.ndarray):
        # Clip joint angles to physical joint limits
        # joint_action = np.clip(action[:6], JOINT_LIMITS["lower"], JOINT_LIMITS["upper"])
        joint_action = action[:6]
        gripper_action = np.clip(action[-1], 0, 1)


        # Convert to degrees for printing
        joint_action_deg = np.round(np.degrees(joint_action), 2)
        print(f"[STEP] Joint action: {joint_action}, Gripper action: {gripper_action}")
        print(f"[STEP] Joint action (deg): {joint_action_deg}, Gripper action: {gripper_action:.3f}")
        self.arm.set_servo_angle_j(joint_action, is_radian=True, wait=False)
        # gripper_mm = 800 + gripper_action * (0 - 800)
        # self.arm.set_gripper_position(gripper_mm, wait=False)

    def save_step_data(self, log_dir, step_idx, obs, action):
        os.makedirs(log_dir, exist_ok=True)

        data = {
            "step_idx": step_idx,
            "timestamp": time.time(),
            "base_rgb": obs["base_rgb"],
            "wrist_rgb": obs["wrist_rgb"],
            "joint_position": obs["joint_position"],
            "gripper_position": obs["gripper_position"],
            "action": action,
        }

        file_path = os.path.join(log_dir, f"{step_idx:05d}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(data, f)

# mock_xarm_env.py
class MockXArmEnv:
    def __init__(self, camera_dict=None):
        self.camera_dict = camera_dict or {}

    def get_observation(self):
        obs = {
            "joint_position": np.random.uniform(low=-1.0, high=1.0, size=(6,)),
            "gripper_position": np.array([np.random.uniform(0.0, 1.0)]),
        }

        # Fake camera images if cameras exist
        for name in self.camera_dict:
            obs[f"{name}_rgb"] = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)

        return obs

    def step(self, action):
        print(f"[STEP] Action received: {action}")

