import argparse
from pathlib import Path
from pprint import pprint
import numpy as np

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


def inspect_lerobot_dataset(repo_id: str, root: Path = None):
    print(f"🔍 Inspecting dataset: {repo_id}")
    root = root.expanduser().resolve() if root else None

    print("\n📖 Loading metadata...")
    meta = LeRobotDatasetMetadata(repo_id, root=root, local_files_only=True)
    print(f"📁 Located dataset in: {meta.root}")
    print(f"📦 Episodes: {meta.total_episodes}")
    print(f"🎥 FPS: {meta.fps}")
    print(f"🤖 Robot type: {meta.robot_type}")
    print("🧠 Tasks:")
    pprint(meta.tasks)
    print("🔑 Feature keys:")
    pprint(meta.features)

    print("\n📂 Loading full dataset...")
    dataset = LeRobotDataset(repo_id, root=root, local_files_only=True)
    print(f"✅ Loaded {dataset.num_episodes} episodes with {dataset.num_frames} total frames")

    for episode_index in range(dataset.num_episodes):
        try:
            from_idx = dataset.episode_data_index["from"][episode_index].item()
            to_idx = dataset.episode_data_index["to"][episode_index].item()
            frame = dataset[from_idx]
            task_index = int(frame.get("task_index", -1))
            prompt = meta.tasks.get(task_index, "⚠️ Unknown task")
            print(f"\n🧪 Episode {episode_index + 1}/{dataset.num_episodes}")
            print(f"   📝 Task index: {task_index}")
            print(f"   🧾 Prompt: {prompt}")
            print(f"   🔢 Steps: {to_idx - from_idx}")

            for key in ["image", "wrist_image", "state", "actions"]:
                value = frame.get(key)
                shape = np.array(value).shape if value is not None else "missing"
                print(f"   📦 {key}: shape = {shape}")

        except Exception as e:
            print(f"❌ Failed to inspect episode {episode_index}: {e}")

    print("\n✅ Inspection complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True, help="e.g. dpad25/agrivla_prune_tomatoes_v1")
    parser.add_argument("--root", type=str, default=None, help="Optional root path (e.g. ~/data/lerobot)")
    args = parser.parse_args()

    root_path = Path(args.root) if args.root else None
    inspect_lerobot_dataset(repo_id=args.repo_id, root=root_path)
