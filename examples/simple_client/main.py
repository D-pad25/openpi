import dataclasses
import enum
import logging
import os
import time
from tkinter import Image

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client import image_tools
import tyro


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    XARM = "xarm"


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    mock: bool = False
    env: EnvMode = EnvMode.ALOHA_SIM
    num_steps: int = 10
    plot_attention: bool = False


def main(args: Args) -> None:
    obs_fn = {
        EnvMode.ALOHA: _random_observation_aloha,
        EnvMode.ALOHA_SIM: _random_observation_aloha,
        EnvMode.DROID: _random_observation_droid,
        EnvMode.LIBERO: _random_observation_libero,
        EnvMode.XARM: _random_observation_xarm,
    }[args.env]
    
    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info(f"Server metadata: {policy.get_server_metadata()}")

    # If using mock data
    if args.mock:
        # Load saved images
        obs = obs_fn()
        base_img_path = os.path.expanduser("~/Thesis/Data/base_rgb_3.png")
        wrist_img_path = os.path.expanduser("~/Thesis/Data/wrist_rgb_3.png")
        # Open and convert to numpy arrays
        obs["base_rgb"] = np.array(Image.open(base_img_path).convert("RGB"))
        obs["wrist_rgb"] = np.array(Image.open(wrist_img_path).convert("RGB"))
        # Resize with pad to match policy input size
        base_rgb = image_tools.resize_with_pad(obs["base_rgb"], 224, 224)
        wrist_rgb = image_tools.resize_with_pad(obs["wrist_rgb"], 224, 224)
    
    if args.env == EnvMode.DROID:
        obs["observation/exterior_image_1_left"] = base_rgb
        obs["observation/wrist_image_left"] = wrist_rgb

    # Send 1 observation to make sure the model is loaded.
    result = policy.infer(obs_fn())
    if args.plot_attention:
        attn_dir = "/home/d_pad25/Thesis/attention_maps/attn_export2"
        os.makedirs(attn_dir, exist_ok=True)
        frame_img = obs["wrist_image_left"]          # HxWx3 uint8 (or your chosen source)
        aw = result["attn_weights"]          # your dict: {source_name: {blockXX: (H,T,T)}}
        # save attention + the image, compressed
        np.savez_compressed(
            os.path.join(attn_dir, "pi0_fast_base.npz"),
            image=frame_img,
            **{f"{src}__{blk}": arr for src, blocks in aw.items() for blk, arr in blocks.items()}
        )

    start = time.time()
    for step in range(args.num_steps):
        output = policy.infer(obs_fn())
        actions_rad = output.get("actions")
        actions_deg = np.degrees(actions_rad)
        print(f"Step {step+1}: Actions (degrees) = {actions_deg}")
    end = time.time()

    print(f"Total time taken: {end - start:.2f} s")
    print(f"Average inference time: {1000 * (end - start) / args.num_steps:.2f} ms")


def _random_observation_aloha() -> dict:
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


def _random_observation_droid() -> dict:
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "Pick a ripe, red tomato and drop it in the blue bucket.",
    }


def _random_observation_libero() -> dict:
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }

def _random_observation_xarm() -> dict:
    return {
        "state": np.random.rand(7),
        "image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
