# main.py

import time
import tyro
import copy
import numpy as np
from openpi_client import websocket_client_policy, image_tools
from xarm_env import XArmRealEnv, MockXArmEnv
from zmq_core.camera_node import ZMQClientCamera
import datetime
import os
from plot_attention import plot_attention_map

from PIL import Image

class MockCamera:
    def read(self, img_size=None):
        # Return fake RGB and depth images
        rgb = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
        depth = np.random.randint(0, 65536, size=(480, 640), dtype=np.uint16)
        return rgb, depth
    
def main(
    remote_host: str = "localhost",
    remote_port: int = 8000,
    wrist_camera_port: int = 5000,
    base_camera_port: int = 5001,
    max_steps: int = 5000,
    prompt: str = "Pick a ripe, red tomato and drop it in the blue bucket.",
    mock: bool = False,
    control_hz: float = 30.0,  # ← New parameter: control frequency in Hz
    step_through_instructions: bool = False,  # New argument
    delta_threshold: float = 0.25,  # New argument for delta threshold
    # log_dir: str = "/media/acrv/DanielsSSD/VLA_data",
    log_dir: str = os.path.expanduser("~/Thesis/attention_maps"),
    # log_dir: str = os.path.expanduser("~/test_logs"),
    save: bool = False,  # New argument to control saving behavior
    plot_attention: bool = False,  # New argument to control attention map plotting
):
    # Create a log directory if it doesn't exist
    if save:
        log_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(log_dir, exist_ok=True)
        print(f"Log directory created at: {log_dir}")

    # Create camera clients
    camera_clients = {}

    # Initialize camera clients
    if mock:
        camera_clients = {
            "wrist": MockCamera(),
            "base": MockCamera(),
        }
        env = MockXArmEnv(camera_dict=camera_clients)
    else:
        camera_clients = {
            "wrist": ZMQClientCamera(port=wrist_camera_port, host=remote_host),
            "base": ZMQClientCamera(port=base_camera_port, host=remote_host),
        }
        # print
        env = XArmRealEnv(camera_dict=camera_clients)
    print("Attempting to connect to server...")
    # Connect to the policy server
    policy_client = websocket_client_policy.WebsocketClientPolicy(
        host=remote_host,
        port=remote_port,
    )


    
    print(f"Connecting to policy server at ws://{remote_host}:{remote_port}...")
    actions_from_chunk_completed = 0
    action_chunk = []
    
    for step_idx in range(max_steps):
        start_time = time.time()
        # print(f"[INFO] Step {step_idx+1}/{max_steps} - Getting observation...")
        obs = env.get_observation()
        # print(f"[INFO] Step {step_idx+1}/{max_steps} - Observations received.")
        # Get new action_chunk if empty or 25 steps have passed
        if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= 25:
            # pad images as per policy requirements
            if mock:
                # Load saved images
                base_img_path = os.path.expanduser("~/Thesis/Data/base_rgb.png")
                wrist_img_path = os.path.expanduser("~/Thesis/Data/wrist_rgb.png")
                # Open and convert to numpy arrays
                obs["base_rgb"] = np.array(Image.open(base_img_path).convert("RGB"))
                obs["wrist_rgb"] = np.array(Image.open(wrist_img_path).convert("RGB"))
                # Resize with pad to match policy input size
                base_rgb = image_tools.resize_with_pad(obs["base_rgb"], 224, 224)
                wrist_rgb = image_tools.resize_with_pad(obs["wrist_rgb"], 224, 224)
            else:
                base_rgb = image_tools.resize_with_pad(obs["base_rgb"], 224, 224)
                wrist_rgb = image_tools.resize_with_pad(obs["wrist_rgb"], 224, 224)

            observation = {
                "state": np.concatenate([obs["joint_position"], obs["gripper_position"]]),
                "image": base_rgb,
                "wrist_image": wrist_rgb,
                "prompt": prompt,
            }
            # Request new action chunk from policy
            result = policy_client.infer(observation)

            actions = result["actions"]
            print("Returning keys from infer:", result.keys())
            attn_weights = result["attn_weights"]

            print("Actions shape:", actions.shape)
            # Access and print attention weights
            if "attn_weights" in result:
                attn_weights = result["attn_weights"]
                print("Client-side Attention Weights Shapes:")
                for source_name, blocks in attn_weights.items():
                    print(f"  Source: {source_name}")
                    for block_name, array in blocks.items():
                        print(f"    {block_name}: shape = {np.asarray(array).shape}")
            else:
                print("No attention weights returned.")

            if plot_attention:
                plot_attention_map(
                    image=obs["wrist_rgb"],      # Input image
                    attn_weights=result["attn_weights"], # From your client
                    source_name="right_wrist_0_rgb",
                    block="block12",
                    head=0,
                    token_idx=0,  # usually the [CLS] token
                    log_dir=log_dir
                )
                        
            # action_chunk = policy_client.infer(observation)["actions"]
            action_chunk = actions
            actions_from_chunk_completed = 0

        action = action_chunk[actions_from_chunk_completed]
        actions_from_chunk_completed += 1

        # Convert joint positions and compute delta (in degrees)
        current_joints_deg = np.degrees(obs["joint_position"][:6])
        action_joints_deg = np.degrees(action[:6])
        delta_deg = action_joints_deg - current_joints_deg

        # In step-through mode, pause for input
        if step_through_instructions:
            start_time = time.time()
            print(f"\n[STEP {step_idx+1}, ACTION {actions_from_chunk_completed}]")
            print("Current Joint State (deg):", np.round(current_joints_deg[:6], 2))
            print("Proposed Action (deg):     ", np.round(action_joints_deg, 2))
            print("Delta (deg):               ", np.round(delta_deg, 2))
            print(f"Gripper pose: {obs['gripper_position']}, Gripper action: {action[-1]:.3f}")

            if np.any(np.abs(delta_deg) > delta_threshold):
                print("⚠️ Warning: large joint delta detected!")

            cmd = input("Press [Enter] to execute, 's' to skip, or 'q' to quit: ").strip().lower()
            if cmd == "q":
                print("Exiting policy execution.")
                break
            elif cmd == "s":
                print("Skipping this action.")
                actions_from_chunk_completed = 0
                continue

            if np.any(np.abs(delta_deg) > delta_threshold):
                print(f"[INFO] Large delta detected (>{delta_threshold} deg). Generating interpolated steps...")
                # Convert current and action joints to radians for interpolation
                state = np.concatenate([obs["joint_position"], obs["gripper_position"]])
                # Interpolate trajectory if delta exceeds threshold
                interpolated_trajectory = env.generate_joint_trajectory(state, action, delta_threshold * np.pi / 180.0)
                obs = env.step_through_interpolated_trajectory(interpolated_trajectory, obs, step_idx, log_dir, control_hz, step_through_instructions, save)
                continue


            # This line runs ONLY if user pressed [Enter]
            print("✅ Executing action...")
            if save:
                obs_to_save = copy.deepcopy(obs)
                env.save_step_data(log_dir, step_idx, obs_to_save, action)
            env.step(np.array(action))
            elapsed = time.time() - start_time
            time.sleep(max(0.0, (1.0 / control_hz) - elapsed))

        # Execute action
        if not step_through_instructions and np.any(np.abs(delta_deg) < delta_threshold):
            if save:
                obs_to_save = copy.deepcopy(obs)
                env.save_step_data(log_dir, step_idx, obs_to_save, action)
            env.step(np.array(action))


        if not step_through_instructions and np.any(np.abs(delta_deg) > delta_threshold):
            # print(f"[INFO] Large delta detected (>{delta_threshold} deg). Requesting new action chunk.")
            # Convert current and action joints to radians for interpolation
            state = np.concatenate([obs["joint_position"], obs["gripper_position"]])
            # Interpolate trajectory if delta exceeds threshold
            interpolated_trajectory = env.generate_joint_trajectory(state, action, delta_threshold * np.pi / 180.0)
            obs = env.step_through_interpolated_trajectory(interpolated_trajectory, obs, step_idx, log_dir, control_hz, step_through_instructions, save)
            continue

        if not step_through_instructions:
            elapsed = time.time() - start_time
            time.sleep(max(0.0, (1.0 / control_hz) - elapsed))


if __name__ == "__main__":
    tyro.cli(main)
