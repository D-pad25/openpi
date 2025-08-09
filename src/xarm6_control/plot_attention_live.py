import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from typing import Dict

def _infer_grid(n):
    g = int(np.sqrt(n))
    if g*g != n:
        raise ValueError(f"Token count {n} not a perfect square")
    return g

def _row_to_heatmap(row, img_hw, mask_first_token=True):
    row = row.astype(np.float32)
    if mask_first_token and row.shape[0] > 1:
        row[0] = 0.0
    g = _infer_grid(row.shape[0])
    m = row.reshape(g, g).copy()
    m /= (m.max() + 1e-8)
    return cv2.resize(m, (img_hw[1], img_hw[0]), interpolation=cv2.INTER_LINEAR)

def interactive_attn_viewer_token_block(
    image: np.ndarray,
    attn_weights: Dict[str, Dict[str, np.ndarray]],
    source_name: str = "base_rgb_0",
    start_block: str = "block_26",
    token0: int = 0,
    combine_heads: str = "mean",   # "mean", "max", or "head"
    head_index: int = 0,           # used if combine_heads == "head"
    mask_first_token: bool = True,
    log_dir: str = ".",
):
    """
    Horizontal slider: block index
    Vertical slider: token index
    """
    os.makedirs(log_dir, exist_ok=True)
    blocks_dict = attn_weights[source_name]
    # Sort block keys like block_00, block_01, ...
    blocks = sorted(blocks_dict.keys(), key=lambda k: int(k.split("_")[1]))
    if start_block not in blocks:
        start_block = blocks[-1]

    def get_attn(block_key):
        A = blocks_dict[block_key]  # (heads, T, T)
        if A.ndim != 3 or A.shape[1] != A.shape[2]:
            raise ValueError(f"Bad attn shape in {block_key}: {A.shape}")
        return A

    A0 = get_attn(start_block)
    num_heads, T, _ = A0.shape
    grid = _infer_grid(T)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    plt.subplots_adjust(left=0.15, bottom=0.25)

    # initial indices
    b0 = blocks.index(start_block)
    t0 = np.clip(token0, 0, T-1)

    def attn_row(block_idx, token_idx):
        A = get_attn(blocks[block_idx])  # (H,T,T)
        if combine_heads == "mean":
            row = A.mean(axis=0)[token_idx]          # (T,)
        elif combine_heads == "max":
            row = A.max(axis=0)[token_idx]           # (T,)
        else:  # "head"
            h = np.clip(head_index, 0, num_heads-1)
            row = A[h, token_idx]                    # (T,)
        return row

    heat = _row_to_heatmap(attn_row(b0, t0), image.shape[:2], mask_first_token)
    im0 = ax.imshow(image / 255.0 if image.dtype == np.uint8 else image)
    im1 = ax.imshow(heat, alpha=0.5)
    ax.set_axis_off()
    title = ax.set_title(f"{source_name} | {blocks[b0]} | token={t0} | "
                         f"{combine_heads}{'' if combine_heads!='head' else f' (h={head_index})'} "
                         f"(grid={grid}×{grid})")

    # sliders
    ax_block = plt.axes([0.15, 0.15, 0.7, 0.03])        # horizontal: block
    ax_tok   = plt.axes([0.05, 0.25, 0.03, 0.6])        # vertical: token
    s_block  = Slider(ax=ax_block, label="Block", valmin=0, valmax=len(blocks)-1, valinit=b0, valstep=1)
    s_tok    = Slider(ax=ax_tok,   label="Token", valmin=0, valmax=T-1,            valinit=t0, valstep=1, orientation='vertical')

    # buttons
    ax_save = plt.axes([0.15, 0.08, 0.2, 0.05])
    ax_quit = plt.axes([0.70, 0.08, 0.15, 0.05])
    b_save  = Button(ax_save, "Save view")
    b_quit  = Button(ax_quit, "Close")

    def update(_):
        bi = int(s_block.val); ti = int(s_tok.val)
        heat = _row_to_heatmap(attn_row(bi, ti), image.shape[:2], mask_first_token)
        im1.set_data(heat)
        title.set_text(f"{source_name} | {blocks[bi]} | token={ti} | "
                       f"{combine_heads}{'' if combine_heads!='head' else f' (h={head_index})'} "
                       f"(grid={grid}×{grid})")
        fig.canvas.draw_idle()

    s_block.on_changed(update)
    s_tok.on_changed(update)

    def do_save(event):
        bi = int(s_block.val); ti = int(s_tok.val)
        blk = blocks[bi]
        tag = combine_heads if combine_heads != "head" else f"head{head_index}"
        fname = f"attnmap_{source_name}_{blk}_{tag}_token{ti}.png"
        fpath = os.path.join(log_dir, fname)
        plt.savefig(fpath, dpi=150, bbox_inches="tight")
        print(f"✅ Saved: {fpath}")

    def do_quit(event):
        plt.close(fig)

    b_save.on_clicked(do_save)
    b_quit.on_clicked(do_quit)

    plt.show()
