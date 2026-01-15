# main.py

import sys
from pathlib import Path
import os

# Ensure we use the correct xarm6_control package from src/
# This handles cases where there might be a duplicate package elsewhere
script_file = Path(__file__).resolve()
src_dir = script_file.parent.parent.parent  # Go up: cli/ -> xarm6_control/ -> src/

# Remove any existing xarm6_control from sys.modules to force re-import from correct location
if 'xarm6_control' in sys.modules:
    # Remove the package and all its submodules
    to_remove = [name for name in sys.modules.keys() if name.startswith('xarm6_control')]
    for name in to_remove:
        del sys.modules[name]

# Ensure src/ is first in sys.path (but don't add duplicates)
if str(src_dir) in sys.path:
    sys.path.remove(str(src_dir))
sys.path.insert(0, str(src_dir))

# Diagnostic: Check which xarm6_control we're about to import
print(f"[DEBUG] Looking for xarm6_control...")
print(f"[DEBUG] Script location: {script_file}")
print(f"[DEBUG] src_dir: {src_dir}")
print(f"[DEBUG] Expected xarm6_control at: {src_dir / 'xarm6_control'}")
print(f"[DEBUG] Does it exist? {(src_dir / 'xarm6_control').exists()}")
print(f"[DEBUG] Does robot_env/ exist? {(src_dir / 'xarm6_control' / 'robot_env').exists()}")
print(f"[DEBUG] sys.path[0] = {sys.path[0]}")

# Verify the correct package location exists
correct_xarm6_path = src_dir / 'xarm6_control'
correct_env_path = correct_xarm6_path / 'robot_env'

if not correct_xarm6_path.exists():
    raise RuntimeError(f"Expected xarm6_control at {correct_xarm6_path} but it doesn't exist!")

# Check what directories actually exist in xarm6_control
if correct_xarm6_path.exists():
    existing_dirs = [d.name for d in correct_xarm6_path.iterdir() if d.is_dir()]
    expected_dirs = ['env', 'cli', 'comms', 'hardware', 'sensors', 'dashboard', 'analysis', 'tests', 'scripts']
    expected_dirs = ['robot_env', 'cli', 'comms', 'hardware', 'sensors', 'dashboard', 'analysis', 'tests', 'scripts']
    missing_dirs = [d for d in expected_dirs if d not in existing_dirs]
    if missing_dirs:
        print(f"\n⚠️  WARNING: Missing directories in xarm6_control: {missing_dirs}")
        print(f"   Existing directories: {existing_dirs}")
        print(f"   This suggests a file sync issue between Windows/Linux.")
        print(f"   Please ensure all directories are synced to Linux.\n")

if not correct_env_path.exists():
    raise RuntimeError(
        f"Expected robot_env/ at {correct_env_path} but it doesn't exist!\n"
        f"This is likely a file sync issue. Check if the robot_env/ directory exists on Linux.\n"
        f"Try: ls -la {correct_env_path.parent}/"
        f"\nNote: Directory was renamed from 'env' to 'robot_env' to avoid .gitignore conflict."
    )

# Force import from correct location by manipulating the package path
import importlib.util
import types

# Load xarm6_control from the correct location
xarm6_spec = importlib.util.spec_from_file_location(
    "xarm6_control",
    correct_xarm6_path / "__init__.py"
)
if xarm6_spec is None or xarm6_spec.loader is None:
    raise RuntimeError(f"Could not create spec for xarm6_control at {correct_xarm6_path}")

# Create the module
xarm6_control = importlib.util.module_from_spec(xarm6_spec)
xarm6_control.__path__ = [str(correct_xarm6_path)]

# Add to sys.modules BEFORE importing so Python uses this one
sys.modules['xarm6_control'] = xarm6_control

# Execute the __init__.py to initialize the package
xarm6_spec.loader.exec_module(xarm6_control)

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
    start_time_log = time.time()
    try:
        for step_idx in range(max_steps):
            start_time = time.time()
            # print(f"[INFO] Step {step_idx+1}/{max_steps} - Getting observation...")
            obs = env.get_observation()
            # print(f"[INFO] Step {step_idx+1}/{max_steps} - Observations received.")
            # Get new action_chunk if empty or 25 steps have passed
            if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= 25:
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
                action_chunk = policy_client.infer(observation)["actions"]
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
                
    except KeyboardInterrupt:
        print("\n🛑 Execution manually stopped by user.")
    finally:
        end_time = time.time()
        total_time = round(end_time - start_time_log, 2)
        if save and validate:
            ask_validation_questions(log_dir, total_time)
        else:
            print("\n(No log_dir provided, skipping validation recording.)")

if __name__ == "__main__":
    tyro.cli(main)
