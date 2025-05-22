import argparse
import cv2
import numpy as np
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_COLOR = (0, 255, 0)
LINE_TYPE = 1


def overlay_text(frame, state, action):
    """Overlay state and action as text onto an image."""
    y = 20
    spacing = 20
    for i, val in enumerate(state):
        cv2.putText(frame, f"state[{i}]: {val:.3f}", (10, y), FONT, FONT_SCALE, FONT_COLOR, LINE_TYPE)
        y += spacing
    for i, val in enumerate(action):
        cv2.putText(frame, f"action[{i}]: {val:.3f}", (200, (i + 1) * spacing), FONT, FONT_SCALE, FONT_COLOR, LINE_TYPE)
    return frame


def render_episode(repo_id: str, root: Path, episode_index: int = 0, out_path: Path = Path("episode_000000.mp4")):
    dataset = LeRobotDataset(repo_id, root=root, local_files_only=True)

    from_idx = dataset.episode_data_index["from"][episode_index].item()
    to_idx = dataset.episode_data_index["to"][episode_index].item()
    print(f"🎬 Rendering Episode {episode_index} with {to_idx - from_idx} frames")

    # Get the first frame to determine shape
    first_frame = dataset[from_idx]
    left = first_frame["image"].permute(1, 2, 0).numpy()
    right = first_frame["wrist_image"].permute(1, 2, 0).numpy()
    height, width, _ = left.shape
    combined_width = width * 2

    video_writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (combined_width, height),
    )

    for idx in range(from_idx, to_idx):
        frame = dataset[idx]
        left = np.ascontiguousarray(frame["image"].permute(1, 2, 0).cpu().numpy(), dtype=np.uint8)
        right = np.ascontiguousarray(frame["wrist_image"].permute(1, 2, 0).cpu().numpy(), dtype=np.uint8)


        # Combine horizontally
        combined = np.concatenate((left, right), axis=1)

        # Overlay text
        state = frame["state"].numpy()
        action = frame["actions"].numpy()
        combined = overlay_text(combined, state, action)

        video_writer.write(combined)

    video_writer.release()
    print(f"✅ Saved video to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", required=True, help="e.g. dpad25/agrivla_prune_tomatoes_v1")
    parser.add_argument("--root", required=True, help="e.g. ~/data/lerobot")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--out", type=str, default="episode_000000.mp4")
    args = parser.parse_args()

    render_episode(
        repo_id=args.repo_id,
        root=Path(args.root).expanduser().resolve(),
        episode_index=args.episode,
        out_path=Path(args.out),
    )
