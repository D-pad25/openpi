from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download
import numpy as np

# Load model config and download checkpoint
model_name = "pi0_fast_droid"
cfg = config.get_config(model_name)
checkpoint_path = download.maybe_download(f"s3://openpi-assets/checkpoints/{model_name}")

# Create trained policy
policy = policy_config.create_trained_policy(cfg, checkpoint_path)

# Create dummy observation (shape/keys may vary across models)
example = {
    "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation/wrist_image_left": np.zeros((224, 224, 3), dtype=np.uint8),
    "prompt": "pick up the fork"
}

# Run inference
output = policy.infer(example)
print("Action output:", output["actions"])
