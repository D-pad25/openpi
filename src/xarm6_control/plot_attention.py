import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

def plot_attention_map(image: np.ndarray,
                       attn_weights: dict[str, dict[str, np.ndarray]],
                       source_name: str = "right_wrist_0_rgb",
                       block: str = "block12",
                       head: int = 0,
                       token_idx: int = 0,
                       log_dir: str = "."):
    """
    Plots an attention map overlayed on the input image.
    """
    try:
        attn = attn_weights[source_name][block]
    except KeyError as e:
        print(f"Missing key: {e}")
        return

    if head >= attn.shape[0] or token_idx >= attn.shape[1]:
        print(f"Invalid head ({head}) or token index ({token_idx})")
        return

    attn_map = attn[head, token_idx]  # shape (num_tokens,)
    num_tokens = attn_map.shape[0]
    grid_size = int(np.sqrt(num_tokens))

    if grid_size * grid_size != num_tokens:
        print("Token count is not a perfect square. Cannot reshape.")
        return

    # Reshape to 2D
    attn_2d = attn_map.reshape(grid_size, grid_size)
    attn_2d /= attn_2d.max() + 1e-8  # Normalize

    # Resize to image resolution
    attn_up = cv2.resize(attn_2d, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(image / 255.0)  # normalize if uint8
    plt.imshow(attn_up, cmap="jet", alpha=0.5)
    plt.title(f"Attention Map\n{source_name}, {block}, head={head}, token={token_idx}")
    plt.axis("off")
    plt.tight_layout()

    filename = f"attnmap_{source_name}_{block}_head{head}_token{token_idx}.png"
    save_path = os.path.join(log_dir, filename)
    plt.savefig(save_path)
    print(f"✅ Attention map saved to: {save_path}")
    plt.close()
