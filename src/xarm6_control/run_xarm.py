# main.py

import time
import tyro
import copy
import math
import threading
import queue
import numpy as np
from collections import deque
from openpi_client import websocket_client_policy, image_tools
from xarm_env import XArmRealEnv, MockXArmEnv
from zmq_core.camera_node import ZMQClientCamera
import datetime
import os
from typing import Optional, Tuple, Deque, List


# ---------------------------
# Camera utilities
# ---------------------------
class MockCamera:
    def read(self, img_size=None):
        rgb = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
        depth = np.random.randint(0, 65536, size=(480, 640), dtype=np.uint16)
        return rgb, depth


class NonBlockingCamera:
    """
    Wraps a blocking camera with a background thread so .read() returns
    the latest frame immediately (no waiting).
    """
    def __init__(self, cam, name: str = "", read_size: Optional[Tuple[int, int]] = None):
        self._cam = cam
        self._name = name
        self._read_size = read_size
        self._lock = threading.Lock()
        self._latest_rgb = None
        self._latest_depth = None
        self._running = True
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def _loop(self):
        while self._running:
            try:
                # Read at a fixed size (if provided) to avoid per-call negotiation overhead
                rgb, depth = self._cam.read(self._read_size)
                with self._lock:
                    self._latest_rgb = rgb
                    self._latest_depth = depth
            except Exception:
                # Avoid crashing the reader; brief backoff
                time.sleep(0.01)

    def read(self, img_size=None):
        # Return the most recent frame available without blocking
        with self._lock:
            if self._latest_rgb is None:
                # If nothing yet, return a minimal dummy frame to keep the loop moving
                rgb = np.zeros((224, 224, 3), dtype=np.uint8)
                depth = np.zeros((224, 224), dtype=np.uint16)
                return rgb, depth
            return self._latest_rgb, self._latest_depth

    def stop(self):
        self._running = False
        # thread is daemon; no join needed


# ---------------------------
# Interpolation utilities
# ---------------------------
def interpolate_action(
    current_joints_rad: np.ndarray,  # shape (6,)
    current_gripper: float,          # scalar
    target_action_rad: np.ndarray,   # shape (7,) -> [6 joints, 1 gripper]
    max_delta_per_step_rad: float
) -> List[np.ndarray]:
    """
    Create a list of intermediate actions (each shape (7,)) that step from
    current to target with per-joint step <= max_delta_per_step_rad.
    Includes the final target action.
    """
    target_joints = target_action_rad[:6]
    target_grip = float(target_action_rad[-1])

    delta = target_joints - current_joints_rad
    max_delta = float(np.max(np.abs(delta))) if delta.size else 0.0
    n_steps = int(math.ceil(max_delta / max_delta_per_step_rad)) if max_delta_per_step_rad > 0 else 1
    n_steps = max(1, n_steps)

    actions = []
    for i in range(1, n_steps + 1):
        alpha = i / n_steps
        joints_i = current_joints_rad + alpha * delta
        grip_i = (1.0 - alpha) * current_gripper + alpha * target_grip
        a = np.zeros(7, dtype=np.float32)
        a[:6] = joints_i
        a[6] = grip_i
        actions.append(a)
    return actions


# ---------------------------
# Async logger
# ---------------------------
class AsyncLogger:
    def __init__(self, env, log_dir: str, max_queue: int = 512):
        self._env = env
        self._log_dir = log_dir
        self._q = queue.Queue(maxsize=max_queue)
        self._running = True
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def log(self, step_idx: int, obs: dict, action: np.ndarray):
        try:
            self._q.put_nowait((step_idx, copy.deepcopy(obs), np.array(action)))
        except queue.Full:
            # Drop logs if the queue is backed up (protect control loop)
            pass

    def _loop(self):
        while self._running:
            try:
                item = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            step_idx, obs, action = item
            try:
                self._env.save_step_data(self._log_dir, step_idx, obs, action)
            except Exception:
                # swallow logging errors
                pass

    def stop(self):
        self._running = False
        try:
            self._q.put_nowait(None)
        except Exception:
            pass
        # thread is daemon; no join required


# ---------------------------
# Policy prefetch
# ---------------------------
class PolicyPrefetcher:
    def __init__(self, policy_client):
        self._client = policy_client
        self._lock = threading.Lock()
        self._ready = False
        self._chunk = None
        self._busy = False

    def prefetch(self, observation: dict):
        with self._lock:
            if self._busy:  # already fetching
                return
            self._busy = True
            self._ready = False
            self._chunk = None
        t = threading.Thread(target=self._worker, args=(observation,), daemon=True)
        t.start()

    def _worker(self, observation: dict):
        try:
            chunk = self._client.infer(observation)["actions"]
            with self._lock:
                self._chunk = chunk
                self._ready = True
        except Exception:
            with self._lock:
                self._chunk = None
                self._ready = False
        finally:
            with self._lock:
                self._busy = False

    def take_if_ready(self) -> Optional[list]:
        with self._lock:
            if self._ready and self._chunk is not None:
                chunk = self._chunk
                self._chunk = None
                self._ready = False
                return chunk
            return None


# ---------------------------
# Main
# ---------------------------
def main(
    remote_host: str = "localhost",
    remote_port: int = 8000,
    wrist_camera_port: int = 5000,
    base_camera_port: int = 5001,
    max_steps: int = 5000,
    prompt: str = "Pick a ripe, red tomato and drop it in the blue bucket.",
    mock: bool = False,
    control_hz: float = 30.0,
    step_through_instructions: bool = False,
    delta_threshold: float = 0.25,  # degrees per tick, for interpolation threshold
    log_dir: str = "/media/acrv/DanielsSSD/Test_sem2",
    save: bool = False,
    # New tuning knobs:
    prefetch_margin: int = 5,          # prefetch next chunk when this many steps remain
    camera_read_size: Optional[Tuple[int, int]] = None,  # e.g., (640, 480); None uses camera default
):
    # Prepare logging directory
    if save:
        log_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(log_dir, exist_ok=True)
        print(f"[LOG] Writing to: {log_dir}")

    # Build cameras (wrapped non-blocking)
    if mock:
        raw_cams = {
            "wrist": MockCamera(),
            "base": MockCamera(),
        }
    else:
        raw_cams = {
            "wrist": ZMQClientCamera(port=wrist_camera_port, host=remote_host),
            "base": ZMQClientCamera(port=base_camera_port, host=remote_host),
        }

    cameras = {
        name: NonBlockingCamera(cam, name=name, read_size=camera_read_size)
        for name, cam in raw_cams.items()
    }

    # Environments
    env = MockXArmEnv(camera_dict=cameras) if mock else XArmRealEnv(camera_dict=cameras)

    # Policy client
    print(f"[NET] Connecting to policy server at ws://{remote_host}:{remote_port} ...")
    policy_client = websocket_client_policy.WebsocketClientPolicy(
        host=remote_host,
        port=remote_port,
    )

    # Background logger
    logger = AsyncLogger(env, log_dir) if save else None

    # Control loop setup
    period = 1.0 / float(control_hz)
    next_t = time.perf_counter()

    action_chunk: List[np.ndarray] = []
    actions_from_chunk_completed = 0
    chunk_len = 0

    # Interpolation buffer (sub-steps for a single high-level action)
    interp_buffer: Deque[np.ndarray] = deque()
    pending_action_increment = 0  # when >0, increment chunk index after buffer drains

    # Utility for building policy observation (only when requesting a new chunk)
    def build_policy_obs(obs_dict: dict) -> dict:
        base_rgb = image_tools.resize_with_pad(obs_dict["base_rgb"], 224, 224)
        wrist_rgb = image_tools.resize_with_pad(obs_dict["wrist_rgb"], 224, 224)
        state = np.concatenate([obs_dict["joint_position"], obs_dict["gripper_position"]])
        return {
            "state": state,
            "image": base_rgb,
            "wrist_image": wrist_rgb,
            "prompt": prompt,
        }

    # Policy prefetcher
    prefetcher = PolicyPrefetcher(policy_client)

    print(f"[CTRL] Starting control at {control_hz:.1f} Hz (period {period*1000:.2f} ms)")
    max_delta_per_step_rad = float(delta_threshold) * math.pi / 180.0

    try:
        for step_idx in range(max_steps):
            # ---- Drift-free tick start
            now = time.perf_counter()

            # ---- Fast observation (non-blocking cameras)
            obs = env.get_observation()  # should be fast; cameras are cached

            # ---- If we still have interpolated micro-steps, send one and continue
            if interp_buffer:
                sub_action = interp_buffer.popleft()
                if save:
                    logger.log(step_idx, obs, sub_action)
                env.step(sub_action)

                # If we've just drained the buffer, mark the parent action as completed
                if not interp_buffer and pending_action_increment > 0:
                    actions_from_chunk_completed += pending_action_increment
                    pending_action_increment = 0

            else:
                # Need a new chunk?
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= chunk_len:
                    # try to take a prefetched one first
                    ready = prefetcher.take_if_ready()
                    if ready is None:
                        # sync fetch
                        observation = build_policy_obs(obs)
                        ready = policy_client.infer(observation)["actions"]
                    action_chunk = ready
                    chunk_len = len(action_chunk) if action_chunk is not None else 0
                    actions_from_chunk_completed = 0
                    if chunk_len == 0:
                        # nothing to do; skip this tick
                        pass

                # With a valid chunk, maybe prefetch the next
                if chunk_len > 0 and (chunk_len - actions_from_chunk_completed) <= prefetch_margin:
                    # Build obs for prefetch (includes resized images)
                    prefetcher.prefetch(build_policy_obs(obs))

                # Execute next high-level action (or begin interpolation)
                if chunk_len > 0 and actions_from_chunk_completed < chunk_len:
                    action = np.array(action_chunk[actions_from_chunk_completed], dtype=np.float32)

                    # Compute delta in radians
                    current_joints = np.array(obs["joint_position"][:6], dtype=np.float32)
                    current_grip = float(obs["gripper_position"])
                    target_joints = action[:6]
                    delta = target_joints - current_joints

                    if np.any(np.abs(delta) > max_delta_per_step_rad):
                        # Build sub-steps and start executing them over upcoming ticks
                        sub_actions = interpolate_action(
                            current_joints, current_grip, action, max_delta_per_step_rad
                        )
                        interp_buffer.extend(sub_actions)
                        pending_action_increment = 1
                        # Execute the first sub-step immediately this tick
                        sub_action = interp_buffer.popleft()
                        if save:
                            logger.log(step_idx, obs, sub_action)
                        env.step(sub_action)
                        if not interp_buffer and pending_action_increment > 0:
                            actions_from_chunk_completed += pending_action_increment
                            pending_action_increment = 0
                    else:
                        # Small delta: send action directly
                        if save:
                            logger.log(step_idx, obs, action)
                        env.step(action)
                        actions_from_chunk_completed += 1

            # ---- Drift-free wait
            next_t += period
            remain = next_t - time.perf_counter()
            if remain > 0:
                time.sleep(remain)
            else:
                # We're late; reset schedule to avoid drift buildup
                next_t = time.perf_counter()

            # ---- Optional step-through (debug mode)
            if step_through_instructions:
                # In debug mode we pause AFTER executing the tick to inspect
                current_joints_deg = np.degrees(obs["joint_position"][:6])
                print(f"\n[STEP {step_idx+1}] Current joints (deg): {np.round(current_joints_deg, 2)}")
                cmd = input("Press [Enter] to continue, 'q' to quit: ").strip().lower()
                if cmd == "q":
                    print("Exiting (debug).")
                    break

    finally:
        # Cleanup background helpers
        if logger is not None:
            logger.stop()
        for cam in cameras.values():
            if isinstance(cam, NonBlockingCamera):
                cam.stop()
