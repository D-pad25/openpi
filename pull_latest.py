import subprocess
import os
import argparse

# Set the path to your Git repo
repo_path = "/home/d_pad25/Thesis/openpi"

def pull_latest(branch):
    try:
        print(f"📂 Navigating to: {repo_path}")
        os.chdir(repo_path)

        print(f"🔄 Pulling latest changes from branch '{branch}'...")
        result = subprocess.run(["git", "pull", "origin", branch], check=True, text=True, capture_output=True)
        print("✅ Pull complete.\n")
        print(result.stdout)

    except subprocess.CalledProcessError as e:
        print("❌ Git pull failed:")
        print(e.stderr)
    except FileNotFoundError:
        print(f"❌ Directory not found: {repo_path}")
    except Exception as ex:
        print(f"⚠️ Unexpected error: {ex}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull the latest changes from a Git branch.")
    parser.add_argument("--branch", type=str, default="main", help="Name of the Git branch to pull (default: 'main')")
    args = parser.parse_args()

    pull_latest(args.branch)
# Test the script