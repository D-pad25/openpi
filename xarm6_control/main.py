# main.py

import time
import tyro
import numpy as np
from openpi_client import websocket_client_policy, image_tools
from xarm_env import XArmRealEnv, MockXArmEnv
from zmq_core.camera_node import ZMQClientCamera

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
    max_steps: int = 100,
    prompt: str = "Pick a ripe, red tomato and drop it in the blue bucket.",
    mock: bool = True,
    control_hz: float = 25.0,  # ← New parameter: control frequency in Hz
    step_through_instructions: bool = True,  # New argument
    delta_threshold: float = 20.0,  # New argument for delta threshold
):
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
        env = XArmRealEnv(camera_dict=camera_clients)

    # Connect to the policy server
    policy_client = websocket_client_policy.WebsocketClientPolicy(
        host=remote_host,
        port=remote_port,
    )

    for _ in range(max_steps):
        start_time = time.time()

        obs = env.get_observation()

        base_rgb = image_tools.resize_with_pad(obs["base_rgb"], 224, 224)
        wrist_rgb = image_tools.resize_with_pad(obs["wrist_rgb"], 224, 224)

        observation = {
            "state": np.concatenate([obs["joint_position"], obs["gripper_position"]]),
            "image": base_rgb,
            "wrist_image": wrist_rgb,
            "prompt": prompt,
        }

        action_chunk = policy_client.infer(observation)["actions"]

        for i, action in enumerate(action_chunk):
            if step_through_instructions:
                current_joints_rad = obs["joint_position"]
                current_joints_deg = np.degrees(current_joints_rad)
                action_joints_rad = np.array(action[:6])
                action_joints_deg = np.degrees(action_joints_rad)
                delta_deg = action_joints_deg - current_joints_deg[:6]

                print(f"\n[STEP {i+1}]")
                print("Current Joint State (deg):", np.round(current_joints_deg[:6], 2))
                print("Proposed Action (deg):     ", np.round(action_joints_deg, 2))
                print("Delta (deg):               ", np.round(delta_deg, 2))
                print(f"Gripper pose: {obs['gripper_position']}, Gripper action: {action[-1]:.3f}")


                if np.any(np.abs(delta_deg) > delta_threshold):
                    print("⚠️ Warning: large joint delta detected!")

                cmd = input("Press [Enter] to execute, 's' to skip, or 'q' to quit: ").strip().lower()
                if cmd == "q":
                    print("Exiting policy execution.")
                    return
                elif cmd == "s":
                    print("Skipping this action.")
                    continue
                else:
                    print("Executing action...")
                    env.step(np.array(action))

                obs["joint_position"] = action_joints_rad  # Update observation after step
                obs["gripper_position"] = np.array([action[-1]])
            else:
                env.step(np.array(action))
                # Maintain control rate
                elapsed = time.time() - start_time
                sleep_time = max(0.0, (1.0 / control_hz) - elapsed)
                time.sleep(sleep_time)


if __name__ == "__main__":
    tyro.cli(main)
