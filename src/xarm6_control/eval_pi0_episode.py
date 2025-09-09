#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline evaluation for a pi0 policy server using a recorded episode of step-wise .pkl files.
This mirrors your live pi0 main.py loop as closely as possible:
- Resizes/pads base & wrist images to 224x224 (HWC, uint8)
- Builds observation dict: {state (7,), image, wrist_image, prompt}
- Requests an action *chunk* from the websocket policy
- Consumes actions one-by-one to emulate control_hz stepping (no sleeps offline)
- Computes MAE/RMSE vs recorded 'control' when available
- Also computes a proxy metric vs observed joint delta q[t+1]-q[t]

Outputs:
- <out_dir>/metrics.csv
- <out_dir>/summary.json (episode aggregates)
- Optional per-step JSON dumps in <out_dir>/steps (enable --save-steps)

Usage example:
  python eval_pi0_episode.py \
    --episode-dir /home/n10813934/data/0828_173511 \
    --out-dir ~/diff_eval/pi0_epoch20_$(date +%F_%H-%M) \
    --remote-host localhost --remote-port 8000 \
    --control-hz 30 --prompt "Pick the ripe tomato"
"""
from __future__ import annotations

import os
import re
import io
import cv2
import json
import math
import glob
import dill
import tyro
import torch
import pickle
import warnings
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import deque

# --- Optional import of your image_tools, else fallback ---
try:
    from openpi_client import image_tools as _imgtools
except Exception:
    _imgtools = None
    from PIL import Image
    def _resize_with_pad(img: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
        ih, iw = img.shape[:2]
        oh, ow = height, width
        if iw/ih >= ow/oh:
            rw = ow
            rh = math.ceil(rw / iw * ih)
        else:
            rh = oh
            rw = math.ceil(rh / ih * iw)
        pil = Image.fromarray(img)
        pil = pil.resize((rw, rh), resample=method)
        resized = np.asarray(pil)
        y0 = max((rh - oh)//2, 0)
        x0 = max((rw - ow)//2, 0)
        crop = resized[y0:y0+oh, x0:x0+ow]
        out = np.zeros((oh, ow, img.shape[2]), dtype=img.dtype)
        y_off = max((oh - crop.shape[0])//2, 0)
        x_off = max((ow - crop.shape[1])//2, 0)
        out[y_off:y_off+crop.shape[0], x_off:x_off+crop.shape[1]] = crop
        return out
else:
    def _resize_with_pad(img: np.ndarray, height: int, width: int) -> np.ndarray:
        return _imgtools.resize_with_pad(img, height, width)

# --- Websocket client ---
from openpi_client import websocket_client_policy

# ---------- Episode utilities ----------

def _read_pkl(path: str) -> Dict:
    with open(path, 'rb') as f:
        try:
            return pickle.load(f)
        except Exception:
            f.seek(0)
            return dill.load(f)


def _nat_sort(paths: List[str]) -> List[str]:
    def key(p: str):
        nums = re.findall(r"\d+", os.path.basename(p))
        return [int(n) for n in nums] if nums else [0]
    return sorted(paths, key=key)


def _coerce_obs_keys(d: Dict) -> Dict:
    out = {}
    for cam in ("base_rgb", "wrist_rgb"):
        if cam not in d:
            raise KeyError(f"Missing key '{cam}' in step")
        out[cam] = d[cam]
    # joints
    if "joint_position" in d:
        jp = np.asarray(d["joint_position"], dtype=np.float32).reshape(-1)
    elif "joint_positions" in d:
        jp = np.asarray(d["joint_positions"], dtype=np.float32).reshape(-1)
    else:
        raise KeyError("Missing joint_position(s) in step")
    out["joint_position"] = jp[:6]
    # gripper
    gp = np.float32(d.get("gripper_position", 0.0)).reshape(()).item()
    out["gripper_position"] = np.float32(gp)
    # control (7,) optional
    ctrl = None
    for k in ("control", "action", "command"):
        if k in d:
            ctrl = np.asarray(d[k], dtype=np.float32).reshape(-1)[:7]
            break
    out["control"] = ctrl
    return out


@dataclass
class Args:
    episode_dir: str
    out_dir: str = "./eval_out_pi0"
    remote_host: str = "localhost"
    remote_port: int = 8000
    control_hz: float = 30.0
    H: int = 224
    W: int = 224
    prompt: str = "Pick a ripe, red tomato and drop it in the blue bucket."
    save_steps: bool = False


# ---------- Metrics helpers ----------

def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))

def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ---------- Main ----------

def main(args: Args):
    os.makedirs(args.out_dir, exist_ok=True)

    # Load episode
    paths = _nat_sort(glob.glob(os.path.join(args.episode_dir, "*.pkl")))
    if not paths:
        raise FileNotFoundError(f"No .pkl found in {args.episode_dir}")
    steps = [_coerce_obs_keys(_read_pkl(p)) for p in paths]
    print(f"[EPISODE] Loaded {len(steps)} steps from {args.episode_dir}")

    # Connect websocket policy
    client = websocket_client_policy.WebsocketClientPolicy(
        host=args.remote_host,
        port=args.remote_port,
    )
    print(f"[PI0] Connected to ws://{args.remote_host}:{args.remote_port}")

    # Evaluation loop
    dt = 1.0 / float(args.control_hz)
    metrics = []

    action_chunk = None
    chunk_len = 0
    chunk_used = 0

    # Iterate through episode
    for i in range(len(steps)):
        obs = steps[i]
        base = _resize_with_pad(obs["base_rgb"], args.H, args.W)
        wrist = _resize_with_pad(obs["wrist_rgb"], args.H, args.W)
        state7 = np.concatenate([obs["joint_position"], np.atleast_1d(obs["gripper_position"]).astype(np.float32)])

        # Get new chunk if needed
        if action_chunk is None or chunk_used >= chunk_len:
            req = {
                "state": state7,             # (7,) float32
                "image": base,               # HWC uint8
                "wrist_image": wrist,        # HWC uint8
                "prompt": args.prompt,
            }
            resp = client.infer(req)
            action_chunk = np.asarray(resp.get("actions", []), dtype=np.float32)
            if action_chunk.ndim == 1:
                action_chunk = action_chunk[None, :]
            if action_chunk.size == 0 or action_chunk.shape[-1] != 7:
                raise RuntimeError(f"pi0 server returned invalid actions shape: {action_chunk.shape}")
            chunk_len = action_chunk.shape[0]
            chunk_used = 0
            print(f"[PI0] New action chunk at step {i}: {chunk_len} actions")

        a_pred = action_chunk[chunk_used]
        chunk_used += 1

        # Compute errors vs recorded control
        a_true = obs.get("control", None)
        if a_true is not None:
            mae_all7 = _mae(a_pred, a_true)
            rmse_all7 = _rmse(a_pred, a_true)
            mae_j = _mae(a_pred[:6], a_true[:6])
            rmse_j = _rmse(a_pred[:6], a_true[:6])
            mae_g = _mae(np.array([a_pred[-1]]), np.array([a_true[-1]]))
            rmse_g = _rmse(np.array([a_pred[-1]]), np.array([a_true[-1]]))
        else:
            mae_all7 = rmse_all7 = mae_j = rmse_j = mae_g = rmse_g = float('nan')

        # Proxy vs qdelta if i+1 exists
        mae_vs_qdelta = rmse_vs_qdelta = None
        if i + 1 < len(steps):
            q_now = np.concatenate([steps[i]["joint_position"], np.atleast_1d(steps[i]["gripper_position"])])
            q_nxt = np.concatenate([steps[i+1]["joint_position"], np.atleast_1d(steps[i+1]["gripper_position"])])
            q_delta = q_nxt - q_now
            mae_vs_qdelta = _mae(a_pred, q_delta)
            rmse_vs_qdelta = _rmse(a_pred, q_delta)

        metrics.append({
            "step": i,
            "mae_all7": mae_all7,
            "rmse_all7": rmse_all7,
            "mae_joints6": mae_j,
            "rmse_joints6": rmse_j,
            "mae_grip": mae_g,
            "rmse_grip": rmse_g,
            "mae_vs_qdelta_all7": mae_vs_qdelta,
            "rmse_vs_qdelta_all7": rmse_vs_qdelta,
        })

        if args.save_steps:
            step_dir = os.path.join(args.out_dir, "steps")
            os.makedirs(step_dir, exist_ok=True)
            with open(os.path.join(step_dir, f"step_{i:06d}.json"), "w") as f:
                json.dump({
                    "step": i,
                    "a_pred": a_pred.tolist(),
                    "a_true": (a_true.tolist() if a_true is not None else None),
                }, f, indent=2)

    # Write CSV
    import csv
    csv_path = os.path.join(args.out_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        w.writeheader()
        for m in metrics:
            w.writerow(m)

    # Aggregate summary
    def _nanmean(vals):
        arr = np.array(vals, dtype=np.float64)
        return float(np.nanmean(arr))

    summary = {
        "mae_all7": _nanmean([m["mae_all7"] for m in metrics]),
        "rmse_all7": _nanmean([m["rmse_all7"] for m in metrics]),
        "mae_joints6": _nanmean([m["mae_joints6"] for m in metrics]),
        "rmse_joints6": _nanmean([m["rmse_joints6"] for m in metrics]),
        "mae_grip": _nanmean([m["mae_grip"] for m in metrics]),
        "rmse_grip": _nanmean([m["rmse_grip"] for m in metrics]),
        "mae_vs_qdelta_all7": _nanmean([m["mae_vs_qdelta_all7"] for m in metrics]),
        "rmse_vs_qdelta_all7": _nanmean([m["rmse_vs_qdelta_all7"] for m in metrics]),
    }

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== Episode Summary =====")
    for k, v in summary.items():
        print(f"{k:>22s}: {v:.6f}")
    print(f"\nWritten: {csv_path}")


if __name__ == "__main__":
    tyro.cli(Args).call(main)
