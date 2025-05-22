import argparse
from pathlib import Path
import numpy as np

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def inspect_lerobot_dataset(repo_id: str, root: Path = None):
    print(f"🔍 Loading dataset: {repo_id}")
    dataset = LeRobotDataset(repo_id=repo_id, root=root, local_files_only=True)
    print(f"📁 Found {dataset.num_episodes} episodes in '{dataset.root}'")
    for i in range(dataset.num_episodes):
        try:
            episode = dataset[i]
            # task = dataset.task[i]
            print(f"\n🧪 Episode {i + 1}/{dataset.num_episodes}")
            # print(f"   📝 Task: {task}")
            print(f"   📝 Task index: {episode[0]['task_index']}")
            print(f"   📝 Prompt: {dataset.tasks[int(episode[0]['task_index'])]}")
            print(f"   🔢 Steps: {len(episode)}")
            
            # Sample frame inspection
            frame = episode[0]
            for key in ["image", "wrist_image", "state", "actions"]:
                value = frame.get(key)
                shape = np.array(value).shape if value is not None else "missing"
                print(f"   📦 {key}: shape = {shape}")

        except Exception as e:
            print(f"❌ Failed to load episode {i}: {e}")

    print("\n✅ Inspection complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True, help="e.g. dpad25/agrivla_prune_tomatoes_v1")
    parser.add_argument("--root", type=str, default=None, help="Optional path override (e.g. ~/data/lerobot)")
    args = parser.parse_args()

    root_path = Path(args.root).expanduser() if args.root else None
    inspect_lerobot_dataset(repo_id=args.repo_id, root=root_path)
