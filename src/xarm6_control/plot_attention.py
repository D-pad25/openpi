import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

'''
Example Function Call:
plot_attention_map(
    image=obs["base_rgb"],      # Input image
    attn_weights=result["attn_weights"], # From your client
    source_name="base_0_rgb",
    block="block26",
    head=0,
    token_idx=0,  # usually the [CLS] token
    log_dir=log_dir
)
'''
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
    # attn_2d = attn_map.reshape(grid_size, grid_size)
    attn_2d = attn_map.reshape(grid_size, grid_size).copy()
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


def plot_attention_map_all_blocks(image: np.ndarray,
                                   attn_weights: dict[str, dict[str, np.ndarray]],
                                   source_name: str = "right_wrist_0_rgb",
                                   token_idx: int = 0,
                                   log_dir: str = "."):
    """
    Plots averaged attention maps (across all heads) for each block,
    overlayed on the original image.
    Saves one image per block.
    """
    os.makedirs(log_dir, exist_ok=True)

    # Ensure the source exists
    if source_name not in attn_weights:
        print(f"❌ Source '{source_name}' not found.")
        return

    for block, attn in attn_weights[source_name].items():
        if attn.ndim != 3:
            print(f"Skipping {block} due to unexpected shape {attn.shape}")
            continue

        # Combine all heads: shape (256, 256)
        attn_avg = attn.mean(axis=0)  # shape: (tokens, tokens)

        if token_idx >= attn_avg.shape[0]:
            print(f"❌ Invalid token index {token_idx} for block {block}")
            continue

        attn_map = attn_avg[token_idx]  # shape: (num_tokens,)
        num_tokens = attn_map.shape[0]
        grid_size = int(np.sqrt(num_tokens))

        if grid_size * grid_size != num_tokens:
            print(f"❌ Token count {num_tokens} is not a perfect square in {block}")
            continue

        # Reshape and normalize
        attn_2d = attn_map.reshape(grid_size, grid_size).copy()
        attn_2d /= attn_2d.max() + 1e-8

        # Upsample to image size
        attn_up = cv2.resize(attn_2d, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Plot
        plt.figure(figsize=(6, 6))
        plt.imshow(image / 255.0)
        plt.imshow(attn_up, cmap="jet", alpha=0.5)
        plt.title(f"Averaged Attention\n{source_name}, {block}, token={token_idx}")
        plt.axis("off")
        plt.tight_layout()

        # Save
        filename = f"attnmap_{source_name}_{block}_avg_token{token_idx}.png"
        save_path = os.path.join(log_dir, filename)
        plt.savefig(save_path)
        print(f"✅ Saved: {save_path}")
        plt.close()

def plot_combined_attention_map(image: np.ndarray,
                                attn_weights: dict[str, dict[str, np.ndarray]],
                                source_name: str = "right_wrist_0_rgb",
                                log_dir: str = "."):
    """
    Combines attention across all blocks, heads, and token indices into a single attention map,
    resizes it to the input image, and overlays the result.
    """
    combined_attn = None
    num_blocks = 0

    try:
        blocks = attn_weights[source_name]
    except KeyError as e:
        print(f"Missing source name: {e}")
        return

    for block_name, block_attn in blocks.items():
        # Shape: (num_heads, num_tokens, num_tokens)
        num_heads, num_tokens_q, num_tokens_k = block_attn.shape
        if num_tokens_q != num_tokens_k:
            print(f"Skipping block {block_name} due to asymmetric token shapes")
            continue

        # Combine over all heads and query tokens
        avg_attn = block_attn.mean(axis=0).mean(axis=0)  # shape: (num_tokens,)
        
        grid_size = int(np.sqrt(avg_attn.shape[0]))
        if grid_size * grid_size != avg_attn.shape[0]:
            print(f"Skipping block {block_name}: token count {avg_attn.shape[0]} is not a perfect square")
            continue

        attn_2d = avg_attn.reshape(grid_size, grid_size).copy()
        attn_2d /= attn_2d.max() + 1e-8  # Normalize
        if combined_attn is None:
            combined_attn = attn_2d
        else:
            combined_attn += attn_2d
        num_blocks += 1

    if combined_attn is None or num_blocks == 0:
        print("No valid attention maps found.")
        return

    # Normalize combined attention map
    combined_attn /= num_blocks
    attn_up = cv2.resize(combined_attn, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(image / 255.0 if image.dtype == np.uint8 else image)
    plt.imshow(attn_up, cmap="jet", alpha=0.5)
    plt.title(f"Combined Attention Map\n{source_name} (All Blocks & Tokens)")
    plt.axis("off")
    plt.tight_layout()

    filename = f"combined_attnmap_{source_name}.png"
    save_path = os.path.join(log_dir, filename)
    plt.savefig(save_path)
    print(f"✅ Combined attention map saved to: {save_path}")
    plt.close()