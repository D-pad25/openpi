# xarm_env.py

import copy
import numpy as np
import os
import time
import pickle
from xarm.wrapper import XArmAPI
import datetime
import socket
import asyncio
import json
from gripper_client_async_v2 import GripperClientAsync

class GripperClient:
    def __init__(self, host='127.0.0.1', port=22345):
        self.host = host
        self.send_gripper_port = port
        self.sock = None

    def connect(self):
        if self.sock is None:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.send_gripper_port))
            # print("[Client] Connected to gripper server")

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None
            # print("[Client] Disconnected")

    def send_command(self, message: str) -> str:
        """Send raw command and get response."""
        try:
            self.connect()
            self.sock.sendall(f"{message.strip()}\n".encode())
            response = self.sock.recv(1024).decode().strip()
            return response
        except Exception as e:
            print(f"[Client Error] {e}")
            return "ERROR"

    def send_gripper_command(self, value: float):
        response = self.send_command(f"SET:{value:.3f}")
        # print(f"[Client] Sent gripper command: {value:.3f} | Server: {response}")

    def receive_gripper_position(self):
        response = self.send_command("GET")
        # print(f"[Client] Gripper state: {response}")
        return response

class GripperClientAsync_old:
    def __init__(self, host='127.0.0.1', port=22345):
        self.host = host
        self.port = port
        self.reader = None
        self.writer = None

    async def connect(self):
        if self.reader is None or self.writer is None:
            self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
            print("[Client] Connected to gripper server")

    async def disconnect(self):
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
            print("[Client] Disconnected")
            self.reader = self.writer = None

    async def send_gripper_command(self, cmd_type: str, payload=None):
        await self.connect()
        message = json.dumps({"cmd": cmd_type, "value": payload})
        self.writer.write((message + "\n").encode())
        await self.writer.drain()

        response = await self.reader.readline()
        return json.loads(response.decode())

    async def set_gripper(self, value: float):
        assert 0.0 <= value <= 1.0, "Gripper value must be between 0.0 and 1.0"
        return await self.send_gripper_command("SET", round(value, 3))

    async def receive_gripper_position(self):
        return await self.send_gripper_command("GET")
    
class XArmRealEnv:
    def __init__(self, ip="192.168.1.203", camera_dict=None):
        self.arm = XArmAPI(ip, is_radian=True)
        print(f"Connecting to xArm at {ip}...")
        self._initialize_arm()
        print("xArm connected and initialized.")
        print("Setting up gripper client...")
        self.gripper = GripperClientAsync()
        print("Gripper client initialized.")
        self.camera_dict = camera_dict or {}

    def _initialize_arm(self):
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(True)
        self.arm.set_mode(1)
        self.arm.set_collision_sensitivity(0)
        self.arm.set_state(state=0)

    def _get_normalized_gripper_position(self) -> float:
        """Returns the gripper position as a normalized float in [0.0, 1.0]."""
        raw_response = self.gripper.receive_gripper_position()
        # print(f"[DEBUG] Raw gripper response: '{raw_response}'")
        try:
            return max(0.0, min(1.0, float(raw_response) / 255.0))  # Normalize to [0, 1]
        except Exception:
            print(f"[WARN] Failed to parse gripper state: '{raw_response}'")
            return 0.0
        

    def _get_joint_position(self):
        code, joint_position = self.arm.get_servo_angle(is_radian=True)
        while code != 0 or joint_position is None:
            print(f"[WARN] get_servo_angle() failed with code {code}. Retrying...")
            self._initialize_arm()
            code, joint_position = self.arm.get_servo_angle(is_radian=True)
        return joint_position

    def get_observation(self):
        joint_position = self._get_joint_position()
        gripper_position = self._get_normalized_gripper_position()
        # print(f"[DEBUG] Joint position: {joint_position}")
        # print(f"[DEBUG] Gripper position: {gripper_position:.3f}")
        obs = {
            "joint_position": np.array(joint_position[:6]),  # Keep only 6 joints
            "gripper_position": np.array([gripper_position]),
        }

        # Build state with 6 joints + 1 gripper
        obs["state"] = np.concatenate([obs["joint_position"], obs["gripper_position"]])

        # Include camera observations if available
        # print("[DEBUG] Reading camera images...")
        for name, camera in self.camera_dict.items():
            image, depth = camera.read()
            obs[f"{name}_rgb"] = image
            obs[f"{name}_depth"] = depth
        # print("[DEBUG] Camera images read successfully.")
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
        deltas = np.abs(target_angles[:6] - current_angles[:6])
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
        # print(f"[STEP] Joint action: {joint_action}, Gripper action: {gripper_action}")
        # print(f"[STEP] Joint action (deg): {joint_action_deg}, Gripper action: {gripper_action:.3f}")
        self.arm.set_servo_angle_j(joint_action, is_radian=True, wait=False)
        self.gripper.send_gripper_command(gripper_action)
        # self.arm.set_gripper_position(gripper_mm, wait=False)
    
    def step_through_interpolated_trajectory(self,
        trajectory, obs, step_idx, log_dir, control_hz, step_through_instructions, save
    ):
        # print(f"[INFO] Interpolation created {len(trajectory)} steps.")

        for i, interpolated_action in enumerate(trajectory):
            start_time = time.time()

            # Print proposed interpolated action
            if step_through_instructions:
                current_deg = np.degrees(obs["joint_position"][:6])
                proposed_deg = np.degrees(interpolated_action[:6])
                delta_deg = proposed_deg - current_deg
                gripper = interpolated_action[-1]

                print(f"\n→ INTERPOLATION STEP {i+1}:")
                print("  Current (deg): ", np.round(current_deg, 2))
                print("  Proposed (deg):", np.round(proposed_deg, 2))
                print("  Δ Delta (deg): ", np.round(delta_deg, 2))
                print(f"  Gripper pose: {obs['gripper_position']}, Gripper action: {gripper:.3f}")

                cmd = input("Press [Enter] to execute, 's' to skip, or 'q' to quit: ").strip().lower()
                if cmd == "q":
                    print("Exiting policy execution.")
                    exit()
                elif cmd == "s":
                    print("Skipping this step.")
                    continue
                print("✅ Executing safe action...")

            if save:
                obs_to_save = copy.deepcopy(obs)
                self.save_step_data(log_dir, step_idx, obs_to_save, interpolated_action)
            self.step(np.array(interpolated_action))
            # print(f"interpolated_action: {np.round(np.degrees(interpolated_action[:6]), 2)} deg | Gripper: {interpolated_action[-1]:.3f}")

            elapsed = time.time() - start_time
            time.sleep(max(0.0, (1.0 / control_hz) - elapsed))

            obs = self.get_observation()
        return obs

    def save_step_data(self, log_dir, step_idx, obs, action):
        data = {
            "timestamp": time.time(),
            "base_rgb": obs["base_rgb"],
            "wrist_rgb": obs["wrist_rgb"],
            "joint_position": obs["joint_position"],
            "gripper_position": obs["gripper_position"],
            "action": action,
        }
        file_name = f"{data['timestamp']:.6f}.pkl"
        file_path = os.path.join(log_dir, file_name)
        with open(file_path, "wb") as f:
            pickle.dump(data, f)

# mock_xarm_env.py
'''
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
'''
# mock_xarm_env.py
class MockXArmEnv:
    def __init__(self, camera_dict=None):
        self.camera_dict = camera_dict or {}
        self.current_joint_position = np.random.uniform(low=-1.0, high=1.0, size=(6,))
        self.current_gripper_position = np.random.uniform(0.0, 1.0)

    def get_observation(self):
        obs = {
            "joint_position": self.current_joint_position,
            "gripper_position": np.array([self.current_gripper_position]),
        }

        # Build state with 6 joints + 1 gripper
        obs["state"] = np.concatenate([obs["joint_position"], obs["gripper_position"]])

        # Fake camera images
        for name in self.camera_dict:
            obs[f"{name}_rgb"] = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
            obs[f"{name}_depth"] = np.random.rand(480, 640).astype(np.float32)

        return obs

    def get_frames(self):
        frames = {}
        for name in self.camera_dict:
            frames[f"{name}_rgb"] = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
            frames[f"{name}_depth"] = np.random.rand(480, 640).astype(np.float32)
        return frames

    def generate_joint_trajectory(self, current_angles, target_angles, max_delta):
        current_angles = np.array(current_angles)
        target_angles = np.array(target_angles)
        deltas = np.abs(target_angles[:6] - current_angles[:6])
        num_steps = int(np.max(deltas / max_delta))

        if num_steps == 0:
            return [target_angles.tolist()]

        trajectory = [
            (current_angles + (target_angles - current_angles) * step / num_steps).tolist()
            for step in range(1, num_steps + 1)
        ]
        return trajectory

    def step(self, action):
        print(f"[MOCK STEP] Action received: {action}")
        self.current_joint_position = np.array(action[:6])
        self.current_gripper_position = float(np.clip(action[-1], 0.0, 1.0))

    def step_through_interpolated_trajectory(
        self, trajectory, obs, step_idx, log_dir, control_hz, step_through_instructions, save
    ):
        for i, interpolated_action in enumerate(trajectory):
            start_time = time.time()

            if step_through_instructions:
                current_deg = np.degrees(obs["joint_position"][:6])
                proposed_deg = np.degrees(interpolated_action[:6])
                delta_deg = proposed_deg - current_deg
                gripper = interpolated_action[-1]

                print(f"\n→ INTERPOLATION STEP {i+1}:")
                print("  Current (deg): ", np.round(current_deg, 2))
                print("  Proposed (deg):", np.round(proposed_deg, 2))
                print("  Δ Delta (deg): ", np.round(delta_deg, 2))
                print(f"  Gripper pose: {obs['gripper_position']}, Gripper action: {gripper:.3f}")

                cmd = input("Press [Enter] to execute, 's' to skip, or 'q' to quit: ").strip().lower()
                if cmd == "q":
                    print("Exiting mock policy execution.")
                    exit()
                elif cmd == "s":
                    print("Skipping this step.")
                    continue
                print("✅ Executing mock safe action...")

            if save:
                obs_to_save = copy.deepcopy(obs)
                self.save_step_data(log_dir, step_idx, obs_to_save, interpolated_action)

            self.step(np.array(interpolated_action))
            elapsed = time.time() - start_time
            time.sleep(max(0.0, (1.0 / control_hz) - elapsed))
            obs = self.get_observation()
        return obs

    def save_step_data(self, log_dir, step_idx, obs, action):
        data = {
            "timestamp": time.time(),
            "base_rgb": obs.get("base_rgb"),
            "wrist_rgb": obs.get("wrist_rgb"),
            "joint_position": obs["joint_position"],
            "gripper_position": obs["gripper_position"],
            "action": action,
        }
        os.makedirs(log_dir, exist_ok=True)
        file_name = f"{data['timestamp']:.6f}.pkl"
        file_path = os.path.join(log_dir, file_name)
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
