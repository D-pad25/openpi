import os
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tyro

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME, LeRobotDataset

# --------------------
# Defaults / constants
# --------------------
# v2 = tomatoes only, no crop_type; v3 = tomatoes + chillis, has crop_type
RAW_DATASET_NAMES = ["agrivla_dataset_v2", "agrivla_dataset_v3"]

@dataclass
class Args:
    data_dir: str
    repo_prefix: str = "dpad25/agrivla_pi0"
    specs: List[str] = field(default_factory=lambda: [
        "all",
        "tomatoes_only",
        "tomatoes_plus:10",
        "tomatoes_plus:20",
        "tomatoes_plus:50",
        "tomatoes_plus:100",
        "tomatoes_plus:200",
        "chillis_only",
    ])
    seed: int = 42
    push_to_hub: bool = False
    clobber: bool = True  # If True, delete existing output folder before creating

# --------------------
# Helpers
# --------------------
def _to_text(x) -> str:
    try:
        if isinstance(x, (bytes, np.bytes_)):
            return x.decode("utf-8")
        if isinstance(x, np.ndarray) and x.dtype.type is np.bytes_:
            return x.tobytes().decode("utf-8")
        if tf.is_tensor(x):
            return _to_text(x.numpy())
    except Exception:
        pass
    return str(x)

def _peek_step_np(ep) -> Optional[dict]:
    try:
        return next(ep["steps"].take(1).as_numpy_iterator())
    except StopIteration:
        return None

def _episode_crop(ep) -> str:
    """Return 'tomato' or 'chilli'. v2 has no crop_type ⇒ default tomato."""
    # 1) episode-level
    if "crop_type" in ep:
        c = _to_text(ep["crop_type"]).lower().strip()
        if c:
            return c
    # 2) step-level (peek step 0)
    s0 = _peek_step_np(ep)
    if s0 is not None and "crop_type" in s0:
        c = _to_text(s0["crop_type"]).lower().strip()
        if c:
            return c
    # 3) default (v2 path)
    return "tomato"

def _episode_instruction(ep) -> str:
    # episode-level first
    for key in ("language_instruction", "instruction", "prompt"):
        if key in ep:
            v = _to_text(ep[key]).strip()
            if v:
                return v
    # then step 0
    s0 = _peek_step_np(ep)
    if s0 is not None:
        for key in ("language_instruction", "instruction", "prompt"):
            if key in s0:
                v = _to_text(s0[key]).strip()
                if v:
                    return v
    return "perform the picking task"

def _should_include(
    crop: str,
    chilli_selected: int,
    chilli_limit: Optional[int],
    include_tomatoes: bool,
    include_chillis: bool,
) -> Tuple[bool, int]:
    """Return (include, updated_chilli_selected)."""
    if crop == "tomato" and include_tomatoes:
        return True, chilli_selected
    if crop == "chilli" and include_chillis:
        if (chilli_limit is None) or (chilli_selected < chilli_limit):
            return True, chilli_selected + 1
    return False, chilli_selected

def _parse_spec(spec: str) -> Tuple[str, bool, bool, Optional[int]]:
    """
    Returns (spec_name, include_tomatoes, include_chillis, chilli_limit)
    """
    s = spec.strip().lower()
    if s == "all":
        return "all", True, True, None
    if s == "tomatoes_only":
        return "tomatoes_only", True, False, None
    if s == "chillis_only":
        return "chillis_only", False, True, None  # all chillis
    if s.startswith("tomatoes_plus:"):
        try:
            n = int(s.split(":", 1)[1])
        except Exception:
            raise ValueError(f"Bad spec '{spec}'. Use tomatoes_plus:N.")
        return f"tomatoes_plus_{n}", True, True, n
    raise ValueError(f"Unknown spec '{spec}'")

def _create_dataset(repo_id: str, clobber: bool) -> LeRobotDataset:
    out_path = LEROBOT_HOME / repo_id
    if out_path.exists():
        if clobber:
            shutil.rmtree(out_path)
            print(f"🧹 Removed existing dataset at {out_path}")
        else:
            raise FileExistsError(f"{out_path} exists (use --clobber to overwrite)")

    print(f"🚀 Creating LeRobot dataset: {out_path}")
    ds = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="xarm6",
        fps=30,
        features={
            "image": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {"dtype": "float32", "shape": (7,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    return ds

def _write_episode_to_lerobot(ds: LeRobotDataset, ep, crop: str):
    # Stream frames
    for step_np in ep["steps"].as_numpy_iterator():
        ds.add_frame(
            {
                "image": step_np["observation"]["image"],         # <<< maybe rename to base_rgb
                "wrist_image": step_np["observation"]["wrist_image"],  # <<< maybe rename to wrist_rgb
                "state": step_np["observation"]["state"].astype("float32"),
                "actions": step_np["action"].astype("float32"),
            }
        )
    task_text = _episode_instruction(ep)
    ds.save_episode(task=f"{task_text} [crop={crop}]")

def build_one_spec(args: Args, spec: str):
    spec_name, include_tomatoes, include_chillis, chilli_limit = _parse_spec(spec)
    repo_id = f"{args.repo_prefix}_{spec_name}"
    ds = _create_dataset(repo_id, args.clobber)

    print(f"\n==== Building split: {spec_name} ====")
    chilli_selected = 0
    counts = {"tomato_eps": 0, "chilli_eps": 0, "frames": 0}

    for raw_name in RAW_DATASET_NAMES:
        print(f"📦 Loading TFDS dataset: {raw_name}")
        raw_ds = tfds.load(raw_name, data_dir=args.data_dir, split="train", shuffle_files=True)
        # raw_ds = raw_ds.shuffle(2048, seed=args.seed, reshuffle_each_iteration=False)
        # convert to a list of examples (shuffled via Python, not TFDS)
        all_eps = list(raw_ds.as_numpy_iterator())
        random.shuffle(all_eps)  # ✅ lightweight randomization

        for ep in all_eps:
            crop = _episode_crop(ep)
            include, chilli_selected = _should_include(
                crop, chilli_selected, chilli_limit, include_tomatoes, include_chillis
            )
            if not include:
                continue

            # quick frame count by iterating once (cheap; we stream anyway)
            n_frames = 0
            for _ in ep["steps"].as_numpy_iterator():
                n_frames += 1
            counts["frames"] += n_frames

            # write episode (iterate again to stream frames)
            _write_episode_to_lerobot(ds, ep, crop)

            if crop == "tomato":
                counts["tomato_eps"] += 1
            else:
                counts["chilli_eps"] += 1

    ds.consolidate(run_compute_stats=False)
    print(f"✅ Saved: {LEROBOT_HOME / repo_id}")
    print(
        f"   Summary → tomatoes: {counts['tomato_eps']} eps | "
        f"chillis: {counts['chilli_eps']} eps | frames: {counts['frames']}"
    )

    if args.push_to_hub:
        ds.push_to_hub(
            tags=["agrivla", "xarm6", "rlds", "pi0"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )

def main(args: Args):
    random.seed(args.seed)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    for spec in args.specs:
        build_one_spec(args, spec)

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)