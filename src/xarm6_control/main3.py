#!/usr/bin/env python3
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
    # Rates
    control_hz: float = 100.0,         # low-level loop (servo) rate
    policy_hz: float = 30.0,           # policy/native model rate
    save: bool = False,                # enable saving
    save_hz: float = 30.0,             # decimated save rate
    # UX / safety
    step_through_instructions: bool = False,
    delta_threshold: float = 0.25,     # degrees; threshold to trigger interpolation
    log_dir: str = "/media/acrv/DanielsSSD/final_test",
):
    # Create a log directory if it doesn't exist
    if save:
        log_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(log_dir, exist_ok=True)
        print(f"Log directory created at: {log_dir}")

    # Initialize camera clients & environment
    if mock:
        camera_clients = {"wrist": MockCamera(), "base": MockCamera()}
        env = MockXArmEnv(camera_dict=camera_clients)
    else:
        camera_clients = {
            "wrist": ZMQClientCamera(port=wrist_camera_port, host=remote_host),
            "base": ZMQClientCamera(port=base_camera_port, host=remote_host),
        }
        env = XArmRealEnv(camera_dict=camera_clients)

    print("Attempting to connect to server...")
    policy_client = websocket_client_policy.WebsocketClientPolicy(
        host=remote_host,
        port=remote_port,
    )
    print(f"Connecting to policy server at ws://{remote_host}:{remote_port}...")

    # --- timing helpers (drift-free) ---
    control_dt = 1.0 / float(control_hz)
    ctp = max(1, int(round(control_hz / policy_hz)))  # control ticks per policy tick

    save_enabled = bool(save and save_hz and save_hz > 0)
    save_dt = (1.0 / float(save_hz)) if save_enabled else None

    next_control_time = time.perf_counter()       # next scheduled control tick
    next_save_time = time.perf_counter()          # next time we are allowed to save
    save_step_count = 0                           # contiguous index for saved samples

    def should_save(now: float | None = None) -> bool:
        """Return True at ~save_hz; advances internal gate without drift."""
        nonlocal next_save_time
        if not save_enabled:
            return False
        t = time.perf_counter() if now is None else now
        if t < next_save_time:
            return False
        # catch up in fixed quanta if we're behind
        while t >= next_save_time:
            next_save_time += save_dt
        return True

    # --- main loop state ---
    actions_from_chunk_completed = 0   # index within the 25-step action chunk
    action_chunk = []
    tick_in_policy = 0                 # 0..(ctp-1); advance policy when 0

    start_time_log = time.perf_counter()

    try:
        for step_idx in range(max_steps):
            # Grab observation (your cameras are ~30 FPS, so this repeats frames at 100 Hz)
            obs = env.get_observation()

            # If it’s a policy tick, (re)request chunk when needed and compute delta
            do_policy_tick = (tick_in_policy == 0)

            if do_policy_tick and (actions_from_chunk_completed == 0 or actions_from_chunk_completed >= 25 or not action_chunk):
                # pad images as per policy requirements
                base_rgb = image_tools.resize_with_pad(obs["base_rgb"], 224, 224)
                wrist_rgb = image_tools.resize_with_pad(obs["wrist_rgb"], 224, 224)

                observation = {
                    "state": np.concatenate([obs["joint_position"], obs["gripper_position"]]),
                    "image": base_rgb,
                    "wrist_image": wrist_rgb,
                    "prompt": prompt,
                }
                # Request new action chunk from policy (expected 25 actions)
                result = policy_client.infer(observation)
                action_chunk = result["actions"]
                actions_from_chunk_completed = 0

            # Hold the same policy action across ctp control ticks
            if not action_chunk:
                # If we somehow failed to get a chunk, just idle this tick
                action = np.zeros(7, dtype=np.float32)
            else:
                action = action_chunk[actions_from_chunk_completed]

            # Policy-step logic (delta check + optional step-through)
            if do_policy_tick:
                current_joints_deg = np.degrees(obs["joint_position"][:6])
                action_joints_deg = np.degrees(action[:6])
                delta_deg = action_joints_deg - current_joints_deg
            else:
                # For held ticks we don’t recompute delta
                delta_deg = None

            # --- STEP-THROUGH MODE ---
            if step_through_instructions and do_policy_tick:
                print(f"\n[STEP {step_idx+1}, POLICY ACTION {actions_from_chunk_completed+1}]")
                print("Current Joint State (deg):", np.round(current_joints_deg[:6], 2))
                print("Proposed Action (deg):     ", np.round(action_joints_deg, 2))
                print("Delta (deg):               ", np.round(delta_deg, 2))
                print(f"Gripper pose: {obs['gripper_position']}, Gripper action: {action[-1]:.3f}")
                if np.any(np.abs(delta_deg) > delta_threshold):
                    print("⚠️  Warning: large joint delta detected!")

                cmd = input("Press [Enter] to execute, 's' to skip, or 'q' to quit: ").strip().lower()
                if cmd == "q":
                    print("Exiting policy execution.")
                    break
                elif cmd == "s":
                    print("Skipping this action.")
                    actions_from_chunk_completed = 0
                    # advance tick schedule and continue
                    tick_in_policy = (tick_in_policy + 1) % ctp
                    # drift-free control sleep
                    now = time.perf_counter()
                    sleep_s = next_control_time - now
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                    next_control_time += control_dt
                    continue

                # If large delta: generate intermediate trajectory.
                if np.any(np.abs(delta_deg) > delta_threshold):
                    print(f"[INFO] Large delta (>{delta_threshold} deg). Generating interpolated steps...")
                    state = np.concatenate([obs["joint_position"], obs["gripper_position"]])
                    interpolated_trajectory = env.generate_joint_trajectory(
                        state, action, delta_threshold * np.pi / 180.0
                    )
                    # We disable internal saving here to avoid over-logging inside env.
                    # Your gated saver will still capture at ~save_hz in the main loop.
                    obs = env.step_through_interpolated_trajectory(
                        interpolated_trajectory, obs, step_idx, log_dir,
                        control_hz, step_through_instructions, False  # save=False
                    )

            # --- EXECUTION (applies to both modes) ---
            # Time-gated save (about save_hz)
            now = time.perf_counter()
            if save and should_save(now):
                obs_to_save = copy.deepcopy(obs)
                env.save_step_data(log_dir, save_step_count, obs_to_save, action)
                save_step_count += 1

            # Send command this control tick
            env.step(np.array(action))

            # If it was a policy tick and we didn't early-continue, advance action index
            if do_policy_tick:
                actions_from_chunk_completed += 1
                # recycle to start if chunk is exhausted (should be refilled next policy tick)
                if actions_from_chunk_completed >= 25:
                    actions_from_chunk_completed = 0

            # Advance control and (occasionally) policy clocks
            tick_in_policy = (tick_in_policy + 1) % ctp

            # Drift-free control scheduler
            now = time.perf_counter()
            sleep_s = next_control_time - now
            if sleep_s > 0:
                time.sleep(sleep_s)
            next_control_time += control_dt

    except KeyboardInterrupt:
        print("\n🛑 Execution manually stopped by user.")
    finally:
        end_time = time.perf_counter()
        total_time = round(end_time - start_time_log, 2)
        if save and validate:
            ask_validation_questions(log_dir, total_time)
        else:
            print("\n(No log_dir provided, skipping validation recording.)")


if __name__ == "__main__":
    tyro.cli(main)
