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
    max_steps: int = 2000,
    prompt: str = "Pick a ripe, red tomato and drop it in the blue bucket.",
    mock: bool = False,
    control_hz: float = 5.0,  # ← New parameter: control frequency in Hz
    step_through_instructions: bool = True,  # New argument
    delta_threshold: float = 0.25,  # New argument for delta threshold
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

    actions_from_chunk_completed = 0
    action_chunk = []

    for step_idx in range(max_steps):
        start_time = time.time()

        # Get new action_chunk if empty or 25 steps have passed
        if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= 25:
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
            actions_from_chunk_completed = 0

        action = action_chunk[actions_from_chunk_completed]
        actions_from_chunk_completed += 1

        # Convert joint positions from radians to degrees for display
        current_joints_rad = obs["joint_position"]
        current_joints_deg = np.degrees(current_joints_rad)
        action_joints_rad = np.array(action[:6])
        action_joints_deg = np.degrees(action_joints_rad)
        delta_deg = action_joints_deg - current_joints_deg[:6]

        # In step-through mode, pause for input
        if step_through_instructions:
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

            # if np.any(np.abs(delta_deg) > delta_threshold):
            #     print("❌ Delta too large — action not executed. Skipping.")

            #     print(f"[INFO] Large delta detected (>{delta_threshold} deg). Requesting new action chunk.")
            #     interpolated_trajectory = env.generate_joint_trajectory(
            #         current_joints_rad, action_joints_rad, max_delta=(delta_threshold * np.pi / 180.0)
            #     )

            #     if interpolated_trajectory:
            #         for interpolated_action in interpolated_trajectory:
            #             if not step_through_instructions:
            #                 start_time = time.time()
            #                 env.step(np.array(interpolated_action))
            #                 obs["joint_position"] = interpolated_action[:6]
            #                 obs["gripper_position"] = np.array([interpolated_action[-1]])
            #                 time.sleep(max(0.0, (1.0 / control_hz) - (time.time() - start_time)))
            #     else:
            #         print("[WARN] No interpolated steps produced.")
            #     actions_from_chunk_completed = 0
            #     continue

            if np.any(np.abs(delta_deg) > delta_threshold):
                print("❌ Delta too large — interpolating instead of skipping.")
                print(f"[INFO] Large delta detected (>{delta_threshold} deg). Generating interpolated steps...")

                interpolated_trajectory = env.generate_joint_trajectory(
                    current_joints_rad, action_joints_rad, delta_threshold * np.pi / 180.0
                )

                if interpolated_trajectory:
                    print(f"[INFO] Interpolation created {len(interpolated_trajectory)} steps.")

                    for i, interpolated_action in enumerate(interpolated_trajectory):
                        # Print proposed interpolated action
                        print(f"\n→ INTERPOLATOIN STEP {i+1}: {np.round(np.degrees(interpolated_action[:6]), 2)} deg | Gripper: {interpolated_action[-1]:.3f}")

                        # Show current joint state and delta
                        current_joints_deg = np.degrees(obs["joint_position"][:6])
                        proposed_action_deg = np.degrees(interpolated_action[:6])
                        delta_deg = proposed_action_deg - current_joints_deg

                        print("Current Joint State (deg):", np.round(current_joints_deg, 2))
                        print("Proposed Action (deg):     ", np.round(proposed_action_deg, 2))
                        print("Delta (deg):               ", np.round(delta_deg, 2))
                        print(f"Gripper pose: {obs['gripper_position']}, Gripper action: {interpolated_action[-1]:.3f}")

                        # Prompt user
                        cmd = input("Press [Enter] to execute, 's' to skip, or 'q' to quit: ").strip().lower()
                        if cmd == "q":
                            print("Exiting policy execution.")
                            exit()  # Ensures entire script exits cleanly
                        elif cmd == "s":
                            print("Skipping this step.")
                            continue

                        # Execute step
                        print("✅ Executing safe action...")
                        env.step(np.array(interpolated_action))
                        obs["joint_position"] = interpolated_action[:6]
                        obs["gripper_position"] = np.array([interpolated_action[-1]])
                    continue

                else:
                    print("[WARN] No interpolated steps produced.")
                continue

            # This line runs ONLY if user pressed [Enter]
            
            env.step(np.array(action))

        if not step_through_instructions and np.any(np.abs(delta_deg) > delta_threshold):
            print(f"[INFO] Large delta detected (>{delta_threshold} deg). Requesting new action chunk.")
            interpolated_trajectory = env.generate_joint_trajectory(
                current_joints_rad, action_joints_rad, max_delta=(delta_threshold * np.pi / 180.0)
            )

            if interpolated_trajectory:
                for interpolated_action in interpolated_trajectory:
                    if not step_through_instructions:
                        start_time = time.time()
                        env.step(np.array(interpolated_action))
                        obs["joint_position"] = interpolated_action[:6]
                        obs["gripper_position"] = np.array([interpolated_action[-1]])
                        time.sleep(max(0.0, (1.0 / control_hz) - (time.time() - start_time)))
            else:
                print("[WARN] No interpolated steps produced.")
            # actions_from_chunk_completed = 0
            continue  # Skip executing this action

        # Execute action
        if not step_through_instructions:
            env.step(np.array(action))

        # Update state after step
        obs["joint_position"] = action_joints_rad
        obs["gripper_position"] = np.array([action[-1]])

        if not step_through_instructions:
            elapsed = time.time() - start_time
            time.sleep(max(0.0, (1.0 / control_hz) - elapsed))


if __name__ == "__main__":
    tyro.cli(main)
