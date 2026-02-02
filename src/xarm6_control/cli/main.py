# main.py

import sys
from pathlib import Path

# Ensure src/ is in sys.path so we can import xarm6_control
script_file = Path(__file__).resolve()
src_dir = script_file.parent.parent.parent  # Go up: cli/ -> xarm6_control/ -> src/
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import time
import tyro
import copy
import numpy as np
from openpi_client import websocket_client_policy, image_tools
from xarm6_control.robot_env.xarm_env import XArmRealEnv, MockXArmEnv
from xarm6_control.comms.zmq.camera_node import ZMQClientCamera
from xarm6_control.sensors.transforms.resize_pkl import resize_with_pad_custom
import datetime
import os
import csv

class MockCamera:
    def read(self, img_size=None):
        # Return fake RGB and depth images
        rgb = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
        depth = np.random.randint(0, 65536, size=(480, 640), dtype=np.uint16)
        return rgb, depth

def ask_validation_questions(log_dir: str, total_time: float):
    """Ask validation questions and append results to a shared CSV log."""
    print("\n🧠 Validation Questionnaire")
    print("Please answer the following questions to validate this trial.\n")

    # Mapping for shorthand normalization
    normalize_map = {
        "y": "yes", "n": "no",
        "l": "left", "r": "right",
        "t": "tomato", "c": "chilli",
        "b": "both",
    }

    def normalize_response(resp: str) -> str:
        """Normalizes short or mixed-case answers."""
        r = resp.strip().lower()
        if r in normalize_map:
            return normalize_map[r]
        # handle combined shorthand like "t+c"
        if "+" in r or "," in r:
            parts = [normalize_map.get(p.strip(), p.strip()) for p in r.replace(",", "+").split("+")]
            if "tomato" in parts and "chilli" in parts:
                return "both"
        return r

    questions = [
        ("initial_position", "Starting position (left/center/right): "),
        ("bucket_side", "Bucket location (left/right): "),
        ("fruit_variant", "Fruit type (tomato/chilli): "),
        ("fruits_present", "Which fruits were present in the scene? (tomato/chilli/both/none): "),
        ("location", "Crop location (e.g., row2_col3): "),
        ("in_training_set", "Was this location in training data? (yes/no/unsure): "),
        ("reached_fruit", "Robot reached target fruit? (yes/no/partial): "),
        ("grasp_success", "Successfully grasped fruit? (yes/no/dropped early): "),
        ("correct_fruit", "Was it the requested fruit? (yes/no/unclear): "),
        ("bucket_drop", "Dropped fruit in bucket? (yes/no/missed): "),
        ("attempts", "Number of grasp attempts: "),
        ("collision", "Collision severity (none/minor/major): "),
        ("recovery_needed", "Manual intervention required? (yes/no): "),
        ("perceived_difficulty", "How difficult was this case (1–5): "),
        ("good_for_presentation", "Was this run good for presentation/video? (yes/no): "),
        ("notes", "Additional notes/observations: "),
    ]

    answers = {}
    for key, prompt in questions:
        raw = input(prompt).strip()
        answers[key] = normalize_response(raw)

    # Add automatic metadata
    answers["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    answers["run_folder"] = os.path.basename(log_dir)
    answers["total_time_sec"] = total_time

    # Numeric coercion
    try:
        answers["attempts"] = int(answers["attempts"])
    except (ValueError, KeyError):
        answers["attempts"] = None

    try:
        answers["perceived_difficulty"] = int(answers["perceived_difficulty"])
    except (ValueError, KeyError):
        answers["perceived_difficulty"] = None

    # CSV path
    csv_path = os.path.join(os.path.dirname(log_dir), "validation_log.csv")
    write_header = not os.path.exists(csv_path)

    fieldnames = [
        "timestamp", "run_folder", "total_time_sec",
        "initial_position", "bucket_side", "fruit_variant", "fruits_present", "location",
        "in_training_set", "reached_fruit", "grasp_success", "correct_fruit",
        "bucket_drop", "attempts", "collision", "recovery_needed",
        "perceived_difficulty", "good_for_presentation", "notes"
    ]

    # Fill missing fields
    for field in fieldnames:
        if field not in answers:
            answers[field] = ""

    # Append to CSV
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            print(f"\n🆕 Created new validation log at: {csv_path}")
        else:
            print(f"\n📄 Appending to existing validation log: {csv_path}")
        writer.writerow(answers)

    # Summary
    print(f"\n⏱️  Total run time: {total_time:.2f} s")
    print(f"✅ Validation results appended successfully.\n")




def main(
    remote_host: str = "localhost",
    remote_port: int = 8000,
    wrist_camera_port: int = 5000,
    base_camera_port: int = 5001,
    max_steps: int = 5000,
    prompt: str = "Pick a ripe, red tomato and drop it in the blue bucket.",
    mock: bool = False,
    validate: bool = True,
    control_hz: float = 30.0,  # ← New parameter: control frequency in Hz
    step_through_instructions: bool = False,  # New argument
    delta_threshold: float = 0.25,  # New argument for delta threshold
    log_dir: str = "/media/acrv/DanielsSSD/final_test",
    # log_dir: str = os.path.expanduser("~/test_logs"),
    save: bool = False,  # New argument to control saving behavior
    gripper_mode: str = "ros",  # "ros" for ROS/Teensy, "usb" for direct USB control
    gripper_usb_port: str = "/dev/ttyUSB0",  # Serial port for USB mode (Linux) or "COM3" (Windows)
    gripper_host: str = "127.0.0.1",  # Host for ROS mode TCP server
    gripper_port: int = 22345,  # Port for ROS mode TCP server
    action_dim: int = 25, #25 Hz normal
    use_rerun: bool = False,
):
    # Create a log directory if it doesn't exist
    if save:
        log_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(log_dir, exist_ok=True)
        print(f"Log directory created at: {log_dir}")
    if use_rerun:
        import rerun as rr
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
        env = XArmRealEnv(
            camera_dict=camera_clients,
            gripper_mode=gripper_mode,
            gripper_usb_port=gripper_usb_port,
            gripper_host=gripper_host,
            gripper_port=gripper_port,
            use_rerun=use_rerun,
        )
    print("Attempting to connect to server...")
    sys.stdout.flush()  # Ensure output is visible immediately
    
    # Connect to the policy server
    try:
        policy_client = websocket_client_policy.WebsocketClientPolicy(
            host=remote_host,
            port=remote_port,
        )
        print(f"✅ Policy client created. Connecting to ws://{remote_host}:{remote_port}...")
        sys.stdout.flush()
        
        # Test connection by trying to get metadata or ping
        # The client should connect on first infer() call, but let's verify it's working
        print("✅ Policy client initialized successfully.")
        sys.stdout.flush()
    except Exception as e:
        print(f"❌ Failed to create policy client: {e}")
        sys.stdout.flush()
        raise

    actions_from_chunk_completed = 0
    action_chunk = []
    start_time_log = time.time()
    print(f"🚀 Starting policy execution loop (max_steps={max_steps})...")
    sys.stdout.flush()
    
    try:
        for step_idx in range(max_steps):
            start_time = time.time()
            # Get observation
            try:
                obs = env.get_observation()
                if step_idx == 0:
                    print(f"[Step {step_idx+1}] ✅ Got initial observation")
                    sys.stdout.flush()
            except Exception as e:
                print(f"❌ Error getting observation at step {step_idx+1}: {e}")
                sys.stdout.flush()
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
                raise
            # Get new action_chunk if empty or 25 steps have passed
            if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= action_dim:
                try:
                    # pad images as per policy requirements
                    # base_rgb = image_tools.resize_with_pad(obs["base_rgb"], 224, 224)
                    # wrist_rgb = image_tools.resize_with_pad(obs["wrist_rgb"], 224, 224)
                    base_rgb = resize_with_pad_custom(obs["base_rgb"], 224, 224)
                    wrist_rgb = resize_with_pad_custom(obs["wrist_rgb"], 224, 224)

                    observation = {
                        "state": np.concatenate([obs["joint_position"], obs["gripper_position"]]),
                        "image": base_rgb,
                        "wrist_image": wrist_rgb,
                        "prompt": prompt,
                    }
                    # Request new action chunk from policy
                    if step_idx % action_dim == 0:  # Log every action_dim steps to avoid spam
                        print(f"[Step {step_idx+1}] Requesting new action chunk from policy...")
                        sys.stdout.flush()
                    action_chunk = policy_client.infer(observation)["actions"]
                    actions_from_chunk_completed = 0
                    if step_idx % action_dim == 0:
                        print(f"[Step {step_idx+1}] ✅ Received action chunk with {len(action_chunk)} actions")
                        sys.stdout.flush()
                except Exception as e:
                    print(f"❌ Error getting action chunk at step {step_idx+1}: {e}")
                    sys.stdout.flush()
                    import traceback
                    traceback.print_exc()
                    sys.stdout.flush()
                    raise

            action = action_chunk[actions_from_chunk_completed]
            rr.log(f"/xarm6/joint_positions", action[:6])
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
                
    except KeyboardInterrupt:
        print("\n🛑 Execution manually stopped by user.")
        sys.stdout.flush()
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        sys.stdout.flush()
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        raise
    finally:
        end_time = time.time()
        total_time = round(end_time - start_time_log, 2)
        if save and validate:
            ask_validation_questions(log_dir, total_time)
        else:
            print("\n(No log_dir provided, skipping validation recording.)")

if __name__ == "__main__":
    tyro.cli(main)