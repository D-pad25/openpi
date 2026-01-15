#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_supervisor_plots.py  (thesis-ready)

Create a suite of publication-ready plots comparing multiple evaluation runs
(e.g., Diffusion vs pi0) from their metrics.csv files.

Outputs (saved to --out-dir):
  1) aggregate_bars.(png|pdf)
  2) per_step_mae_all7.(png|pdf)
  3) per_step_mae_joints6.(png|pdf)
  4) per_step_mae_grip.(png|pdf)
  5) rolling_mae_all7.(png|pdf)
  6) hist_mae_all7.(png|pdf)
  7) cdf_mae_all7.(png|pdf)
  8) violin_mae_all7.(png|pdf)
  9) improvement_vs_<label>.(png|pdf)
 10) summary_table.(png|pdf)

If *_vs_qdelta columns exist, analogous plots are produced with suffix `_vs_qdelta`.

Usage:
  python make_supervisor_plots.py \
    --out-dir /path/to/out \
    /path/to/diffusion/metrics.csv Diffusion \
    /path/to/pi0/metrics.csv pi0
"""
import os
import json
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cycler

# ---------- Helpers / Style ----------

OKABE_ITO = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#000000",  # black
]

def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def _swap_ext(path: str, new_ext: str) -> str:
    root, _ = os.path.splitext(path)
    return f"{root}.{new_ext.lstrip('.')}"

def _save_png_pdf(fig: plt.Figure, out_png_path: str):
    fig.savefig(out_png_path, dpi=300, bbox_inches="tight")
    fig.savefig(_swap_ext(out_png_path, "pdf"), bbox_inches="tight")

def _read_csv_labeled(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df["run"] = label
    required = [
        "step",
        "mae_all7", "rmse_all7",
        "mae_joints6", "rmse_joints6",
        "mae_grip", "rmse_grip",
    ]
    for k in required:
        if k not in df.columns:
            raise KeyError(f"Missing column '{k}' in {path}")
    return df

def _aggregate(df: pd.DataFrame) -> dict:
    nanmean = lambda s: float(np.nanmean(s.values))
    return {
        "mae_all7": nanmean(df["mae_all7"]),
        "rmse_all7": nanmean(df["rmse_all7"]),
        "mae_joints6": nanmean(df["mae_joints6"]),
        "rmse_joints6": nanmean(df["rmse_joints6"]),
        "mae_grip": nanmean(df["mae_grip"]),
        "rmse_grip": nanmean(df["rmse_grip"]),
        "N_steps": int(df.shape[0]),
    }

def _trim_to_common_length(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    L = min(int(df["step"].max()) + 1 for df in dfs)
    return [df[df["step"] < L].reset_index(drop=True) for df in dfs]

def _nice_style():
    plt.rcParams.update({
        # Fonts (PDF-safe)
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "CMU Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # Sizes
        "figure.dpi": 200,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,

        # Axes & grid
        "axes.prop_cycle": cycler(color=OKABE_ITO),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.35,
        "grid.color": "#cccccc",

        # Legend
        "legend.frameon": False,
        "legend.fontsize": 12,
    })

# ---------- Plotters ----------

def plot_aggregate_bars(aggregates: List[Tuple[str, dict]], out_path_png: str):
    """Clean grouped bars: no per-bar text; legend outside; y-grid only."""
    labels = [lbl for lbl, _ in aggregates]
    metrics = [
        "mae_all7", "rmse_all7",
        "mae_joints6", "rmse_joints6",
        "mae_grip", "rmse_grip",
    ]
    values = np.array([[agg[m] for m in metrics] for _, agg in aggregates])  # (R, M)

    fig, ax = plt.subplots(figsize=(11, 5.6))
    x = np.arange(len(metrics))
    width = 0.8 / max(1, len(labels))

    # y-grid only, no x-grid
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    for i, lbl in enumerate(labels):
        ax.bar(
            x + i * width - (len(labels) - 1) * width / 2,
            values[i],
            width,
            label=lbl,
            edgecolor="none",
        )

    ax.set_xticks(x, metrics, rotation=12, ha="right")
    ax.set_ylabel("Error (rad)")
    ax.set_title("Aggregate MAE / RMSE", weight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=min(3, len(labels)))
    ax.margins(y=0.15)

    fig.tight_layout()
    _save_png_pdf(fig, out_path_png)
    plt.close(fig)

def plot_per_step_overlay(dfs: List[pd.DataFrame], labels: List[str], col: str, out_path_png: str):
    dfs = _trim_to_common_length(dfs)
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    for df, lbl in zip(dfs, labels):
        ax.plot(df["step"], df[col], label=lbl, linewidth=2.0)
    ax.set_xlabel("Step")
    ax.set_ylabel("Error (rad)")
    ax.set_title(f"{col} per step", weight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=min(3, len(labels)))
    fig.tight_layout()
    _save_png_pdf(fig, out_path_png)
    plt.close(fig)

def plot_rolling(dfs: List[pd.DataFrame], labels: List[str], col: str, out_path_png: str, window: int = 25):
    dfs = _trim_to_common_length(dfs)
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    for df, lbl in zip(dfs, labels):
        roll = df[col].rolling(window=window, min_periods=1).mean()
        ax.plot(df["step"], roll, label=lbl, linewidth=2.2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Error (rad)")
    ax.set_title(f"{col} rolling mean (window={window})", weight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=min(3, len(labels)))
    fig.tight_layout()
    _save_png_pdf(fig, out_path_png)
    plt.close(fig)

def plot_hist(dfs: List[pd.DataFrame], labels: List[str], col: str, out_path_png: str, bins: int = 40):
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.grid(False)
    for df, lbl in zip(dfs, labels):
        ax.hist(
            df[col].dropna().values,
            bins=bins,
            alpha=0.45,
            label=lbl,
            density=True,
            edgecolor="none",
        )
    ax.set_xlabel("Error (rad)")
    ax.set_ylabel("Density")
    ax.set_title(f"{col} distribution", weight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=min(3, len(labels)))
    fig.tight_layout()
    _save_png_pdf(fig, out_path_png)
    plt.close(fig)

def plot_cdf(dfs: List[pd.DataFrame], labels: List[str], col: str, out_path_png: str):
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for df, lbl in zip(dfs, labels):
        x = np.sort(df[col].dropna().values)
        y = np.linspace(0, 1, len(x), endpoint=True)
        ax.plot(x, y, label=lbl, linewidth=2.2)
    ax.set_xlabel("Error (rad)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title(f"{col} CDF", weight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save_png_pdf(fig, out_path_png)
    plt.close(fig)

def plot_violin(dfs: List[pd.DataFrame], labels: List[str], col: str, out_path_png: str):
    data = [df[col].dropna().values for df in dfs]
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    parts = ax.violinplot(data, showmeans=True, showmedians=True)

    # Style violins
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(OKABE_ITO[i % len(OKABE_ITO)])
        b.set_edgecolor("none")
        b.set_alpha(0.7)
    for k in ("cbars", "cmins", "cmaxes"):
        parts[k].set_visible(False)
    parts["cmeans"].set_linewidth(1.8)
    parts["cmedians"].set_linewidth(1.8)

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right")  # angled x labels (requested)
    ax.set_ylabel("Error (rad)")
    ax.set_title(f"Distribution of {col}", weight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    fig.tight_layout()
    _save_png_pdf(fig, out_path_png)
    plt.close(fig)

def plot_improvement_by_step(baseline: pd.DataFrame, candidate: pd.DataFrame, col: str, out_path_png: str):
    b, c = _trim_to_common_length([baseline, candidate])
    diff = b[col].values - c[col].values
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ax.plot(b["step"], diff, linewidth=2.0)
    ax.axhline(0.0, color="#555555", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Step")
    ax.set_ylabel("Δ Error (rad)")
    ax.set_title(f"Improvement over baseline ({col})", weight="bold")
    # No legend here – keeps the chart clean
    fig.tight_layout()
    _save_png_pdf(fig, out_path_png)
    plt.close(fig)

def render_summary_table(aggregates: List[Tuple[str, dict]], baseline_idx: int, out_path_png: str):
    labels = [lbl for lbl, _ in aggregates]
    metrics = ["mae_all7","rmse_all7","mae_joints6","rmse_joints6","mae_grip","rmse_grip"]
    rows = []
    base = aggregates[baseline_idx][1]
    for i, (lbl, agg) in enumerate(aggregates):
        row_vals = [agg[m] for m in metrics]
        impr = [0.0 for _ in metrics] if i == baseline_idx else [100.0 * (base[m] - agg[m]) / base[m] for m in metrics]
        rows.append([lbl] + row_vals + impr)

    cols = ["Run"] + metrics + [m + "_impr%" for m in metrics]
    fmt_rows = []
    for r in rows:
        fr = []
        for v in r:
            fr.append(f"{v:.3f}" if isinstance(v, float) else v)
        fmt_rows.append(fr)

    fig, ax = plt.subplots(figsize=(min(16, 4 + 1.6 * len(labels)), 1.0 + 0.45 * len(labels)))
    ax.axis("off")
    table = ax.table(cellText=fmt_rows, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.25)
    fig.tight_layout()
    _save_png_pdf(fig, out_path_png)
    plt.close(fig)

# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("items", nargs="+", help="Pairs: CSV LABEL [CSV LABEL ...]")
    args = parser.parse_args()

    if len(args.items) % 2 != 0:
        raise SystemExit("Provide pairs: CSV_PATH LABEL [CSV PATH LABEL ...]")

    _nice_style()
    _ensure_dir(args.out_dir)

    pairs: List[Tuple[str, str]] = [(args.items[i], args.items[i + 1]) for i in range(0, len(args.items), 2)]
    dfs: List[pd.DataFrame] = []
    labels: List[str] = []
    for path, lbl in pairs:
        df = _read_csv_labeled(path, lbl)
        dfs.append(df)
        labels.append(lbl)

    # Aggregates and save JSON
    aggs = [(lbl, _aggregate(df)) for lbl, df in zip(labels, dfs)]
    with open(os.path.join(args.out_dir, "aggregates.json"), "w") as f:
        json.dump({lbl: agg for lbl, agg in aggs}, f, indent=2)

    # 1) Aggregate bars
    plot_aggregate_bars(aggs, os.path.join(args.out_dir, "aggregate_bars.png"))

    # 2-4) Per-step overlays
    plot_per_step_overlay(dfs, labels, "mae_all7",    os.path.join(args.out_dir, "per_step_mae_all7.png"))
    plot_per_step_overlay(dfs, labels, "mae_joints6", os.path.join(args.out_dir, "per_step_mae_joints6.png"))
    plot_per_step_overlay(dfs, labels, "mae_grip",    os.path.join(args.out_dir, "per_step_mae_grip.png"))

    # 5) Rolling average (25-step)
    plot_rolling(dfs, labels, "mae_all7", os.path.join(args.out_dir, "rolling_mae_all7.png"), window=25)

    # 6) Histogram & 7) CDF
    plot_hist(dfs, labels, "mae_all7", os.path.join(args.out_dir, "hist_mae_all7.png"))
    plot_cdf(dfs, labels,  "mae_all7", os.path.join(args.out_dir, "cdf_mae_all7.png"))

    # 8) Violin
    plot_violin(dfs, labels, "mae_all7", os.path.join(args.out_dir, "violin_mae_all7.png"))

    # 9) Improvement curve (only if >=2 runs) — first run is baseline vs each other
    if len(dfs) >= 2:
        for i in range(1, len(dfs)):
            outp = os.path.join(args.out_dir, f"improvement_vs_{labels[i]}.png")
            plot_improvement_by_step(dfs[0], dfs[i], "mae_all7", outp)

    # 10) Summary table (first run as baseline)
    render_summary_table(aggs, baseline_idx=0, out_path_png=os.path.join(args.out_dir, "summary_table.png"))

    # Optional: vs qdelta plots if present
    if "mae_vs_qdelta_all7" in dfs[0].columns:
        plot_per_step_overlay(dfs, labels, "mae_vs_qdelta_all7", os.path.join(args.out_dir, "per_step_mae_vs_qdelta_all7.png"))
        plot_hist(dfs, labels,            "mae_vs_qdelta_all7", os.path.join(args.out_dir, "hist_mae_vs_qdelta_all7.png"))
        plot_cdf(dfs, labels,             "mae_vs_qdelta_all7", os.path.join(args.out_dir, "cdf_mae_vs_qdelta_all7.png"))

    print(f"\n[OK] Saved all plots (PNG & PDF) to {args.out_dir}\n")

if __name__ == "__main__":
    main()
