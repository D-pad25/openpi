import matplotlib.pyplot as plt
import numpy as np
import einops


def plot_attention_map(image: np.ndarray,
                       attn_weights: dict[str, dict[str, np.ndarray]],
                       source_name: str = "right_wrist_0_rgb",
                       block: str = "block12",
                       head: int = 0,
                       token_idx: int = 0,
                       upscale: int = 14):
    """
    Plots an attention map overlayed on the input image.

    Parameters:
    - image: np.ndarray of shape (H, W, 3)
    - attn_weights: Nested dict from model output
    - source_name: The name of the attention source (e.g., 'right_wrist_0_rgb')
    - block: The transformer block to visualize (e.g., 'block12')
    - head: The attention head index to visualize (0-based)
    - token_idx: The query token to visualize (usually 0 is [CLS])
    - upscale: How much to upscale each token (assumes square grid)
    """

    # Extract attention map
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

    # Reshape and upscale
    attn_2d = attn_map.reshape(grid_size, grid_size)
    attn_2d = attn_2d.copy()  # Make the array writeable
    attn_2d /= attn_2d.max()

    attn_up = einops.repeat(attn_2d, 'h w -> (h ph) (w pw)', ph=upscale, pw=upscale)

    # Crop to match image size (if needed)
    attn_up = attn_up[:image.shape[0], :image.shape[1]]

    # Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(image / 255.0)  # normalize if image is uint8
    plt.imshow(attn_up, cmap="jet", alpha=0.5)
    plt.title(f"Attention Map\n{source_name}, {block}, head={head}, token={token_idx}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
