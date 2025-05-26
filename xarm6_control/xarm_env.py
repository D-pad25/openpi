# xarm_env.py

import numpy as np
from xarm.wrapper import XArmAPI


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
            "gripper_position": 0,  # Use a placeholder for gripper position
        }

         # Build state with 6 joints + 1 gripper
        obs["state"] = np.concatenate([obs["joint_position"], obs["gripper_position"]])

        # Include camera observations if available
        for name, camera in self.camera_dict.items():
            image, depth = camera.read()
            obs[f"{name}_rgb"] = image
            obs[f"{name}_depth"] = depth

        return obs

    def step(self, action: np.ndarray):
        # xArm6 has 6 joints, so slice first 6
        joint_action = np.clip(action[:6], -1, 1)
        gripper_action = np.clip(action[-1], 0, 1)

        # self.arm.set_servo_angle_j(joint_action, is_radian=True, wait=False)
        # gripper_mm = 800 + gripper_action * (0 - 800)
        # self.arm.set_gripper_position(gripper_mm, wait=False)

# mock_xarm_env.py

import numpy as np

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

