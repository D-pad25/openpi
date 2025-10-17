#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_supervisor_plots.py

Create a suite of publication-ready plots comparing multiple evaluation runs
(e.g., Diffusion vs pi0) from their metrics.csv files.

Outputs (saved to --out-dir):
  1) aggregate_bars.png           – side-by-side bars for MAE/RMSE (all7, joints6, grip)
  2) per_step_mae_all7.png        – step-wise overlay of MAE(all7) (trimmed to common length)
  3) per_step_mae_joints6.png     – step-wise overlay of MAE(joints6)
  4) per_step_mae_grip.png        – step-wise overlay of MAE(grip)
  5) rolling_mae_all7.png         – 25-step rolling mean overlay (smoother trends)
  6) hist_mae_all7.png            – distribution histograms (overlayed)
  7) cdf_mae_all7.png             – cumulative distribution (lower is better)
  8) violin_mae_all7.png          – violin plot across runs (spread & medians)
  9) improvement_by_step.png      – (baseline − candidate) MAE(all7) vs step
 10) summary_table.png            – rendered table of key aggregates & % improvements

Also produces presentation variants:
  - aggregate_bars_presentation.png  (MAE only)
  - violin_mae_all7_presentation.png

If vs-qdelta columns exist, analogous plots are produced with suffix `_vs_qdelta`.

Usage:
  python make_supervisor_plots.py \
    --out-dir /path/to/out \
    /path/to/diffusion/metrics.csv Diffusion \
    /path/to/pi0/metrics.csv pi0

Notes:
- Provide pairs of arguments: CSV_PATH LABEL [CSV_PATH LABEL ...]
- The script trims step-wise overlays to the shortest run length to align steps.
"""
import os
import json
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Patch

# ---------- Helpers ----------

def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def _read_csv_labeled(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    df['run'] = label
    # sanity
    required = ['step','mae_all7','rmse_all7','mae_joints6','rmse_joints6','mae_grip','rmse_grip']
    for k in required:
        if k not in df.columns:
            raise KeyError(f"Missing column '{k}' in {path}")
    return df


def _aggregate(df: pd.DataFrame) -> dict:
    def nanmean(s):
        return float(np.nanmean(s.values))
    return {
        'mae_all7': nanmean(df['mae_all7']),
        'rmse_all7': nanmean(df['rmse_all7']),
        'mae_joints6': nanmean(df['mae_joints6']),
        'rmse_joints6': nanmean(df['rmse_joints6']),
        'mae_grip': nanmean(df['mae_grip']),
        'rmse_grip': nanmean(df['rmse_grip']),
        'N_steps': int(df.shape[0])
    }


def _trim_to_common_length(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    L = min(int(df['step'].max())+1 for df in dfs)
    out = []
    for df in dfs:
        out.append(df[df['step'] < L].reset_index(drop=True))
    return out


def _presentation_style():
    """Unified style for presentation & report plots."""
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "axes.grid": False,
        "axes.edgecolor": "0.2",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _legend_below(ax, labels=None, colors=None):
    """Move legend below the plot with centered alignment.
       If labels/colors are provided (e.g., violin), build custom handles."""
    if labels is not None and colors is not None:
        handles = [Patch(facecolor=colors[i % len(colors)], edgecolor="black", label=lbl) for i, lbl in enumerate(labels)]
        leg = ax.legend(
        handles, _labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),   # reduce from -0.18 to -0.08
        ncol=min(3, len(_labels)),
        frameon=False)
    else:
        handles, _labels = ax.get_legend_handles_labels()
        leg = ax.legend(
        handles, _labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),   # reduce from -0.18 to -0.08
        ncol=min(3, len(_labels)),
        frameon=False)
    return leg


def _two_dp_ticks(ax, x=False, y=True):
    """Format axis ticks to 2 decimal places."""
    if x:
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    if y:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# ---------- Plotters (All legends below, 2-dp labels) ----------

def plot_aggregate_bars(aggregates: List[Tuple[str, dict]], out_path: str):
    _presentation_style()
    labels = [lbl for lbl,_ in aggregates]
    metrics = ['mae_all7','rmse_all7','mae_joints6','rmse_joints6','mae_grip','rmse_grip']
    values = np.array([[agg[m] for m in metrics] for _,agg in aggregates])  # shape (R, M)

    fig, ax = plt.subplots(figsize=(12,6))
    x = np.arange(len(metrics))
    width = 0.8 / len(labels)
    colors = plt.cm.tab10.colors[:len(labels)]

    for i, lbl in enumerate(labels):
        ax.bar(x + i*width - (len(labels)-1)*width/2, values[i], width, label=lbl, color=colors[i], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(['MAE all7','RMSE all7','MAE joints6','RMSE joints6','MAE grip','RMSE grip'], rotation=20)
    ax.set_ylabel('Error (rad)')
    ax.set_title('Aggregate MAE / RMSE across runs')

    # value labels (2 dp)
    for i in range(len(labels)):
        for j in range(len(metrics)):
            v = values[i, j]
            ax.text(x[j] + i*width - (len(labels)-1)*width/2, v, f"{v:.2f}",
                    ha='center', va='bottom', fontsize=8)

    _two_dp_ticks(ax, y=True)
    _legend_below(ax)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_per_step_overlay(dfs: List[pd.DataFrame], labels: List[str], col: str, out_path: str):
    _presentation_style()
    dfs = _trim_to_common_length(dfs)
    fig, ax = plt.subplots(figsize=(12,5))
    colors = plt.cm.tab10.colors[:len(labels)]
    for (df, lbl, c) in zip(dfs, labels, colors):
        ax.plot(df['step'], df[col], label=lbl, linewidth=2.0, color=c)
    ax.set_xlabel('Step')
    ax.set_ylabel('Error (rad)')
    ax.set_title(f'Per-step {col}')
    _two_dp_ticks(ax, y=True)
    _legend_below(ax)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_rolling(dfs: List[pd.DataFrame], labels: List[str], col: str, out_path: str, window: int = 25):
    _presentation_style()
    dfs = _trim_to_common_length(dfs)
    fig, ax = plt.subplots(figsize=(12,5))
    colors = plt.cm.tab10.colors[:len(labels)]
    for (df, lbl, c) in zip(dfs, labels, colors):
        roll = df[col].rolling(window=window, min_periods=1).mean()
        ax.plot(df['step'], roll, label=lbl, linewidth=2.0, color=c)
    ax.set_xlabel('Step')
    ax.set_ylabel('Error (rad)')
    ax.set_title(f'Rolling Mean ({col}, win={window})')
    _two_dp_ticks(ax, y=True)
    _legend_below(ax)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_hist(dfs: List[pd.DataFrame], labels: List[str], col: str, out_path: str, bins: int = 40):
    _presentation_style()
    fig, ax = plt.subplots(figsize=(10,5))
    colors = plt.cm.tab10.colors[:len(labels)]
    for (df, lbl, c) in zip(dfs, labels, colors):
        ax.hist(df[col].dropna().values, bins=bins, alpha=0.5, label=lbl, density=True, color=c)
    ax.set_xlabel('Error (rad)')
    ax.set_ylabel('Density')
    ax.set_title(f'Histogram of {col}')
    _two_dp_ticks(ax, x=True, y=True)
    _legend_below(ax)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_cdf(dfs: List[pd.DataFrame], labels: List[str], col: str, out_path: str):
    _presentation_style()
    fig, ax = plt.subplots(figsize=(10,5))
    colors = plt.cm.tab10.colors[:len(labels)]
    for (df, lbl, c) in zip(dfs, labels, colors):
        x = np.sort(df[col].dropna().values)
        y = np.linspace(0, 1, len(x), endpoint=True)
        ax.plot(x, y, label=lbl, linewidth=2.0, color=c)
    ax.set_xlabel('Error (rad)')
    ax.set_ylabel('Cumulative fraction')
    ax.set_title(f'CDF of {col}')
    _two_dp_ticks(ax, x=True, y=True)
    _legend_below(ax)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_violin(dfs: List[pd.DataFrame], labels: List[str], col: str, out_path: str):
    _presentation_style()
    data = [df[col].dropna().values for df in dfs]
    colors = plt.cm.tab10.colors[:len(labels)]

    fig, ax = plt.subplots(figsize=(8,5))
    parts = ax.violinplot(data, showmeans=True, showmedians=True, widths=0.7)

    # color each violin body distinctly
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_edgecolor("black")
        pc.set_alpha(0.85)

    for k in ('cbars', 'cmins', 'cmaxes'):
        parts[k].set_edgecolor('black')
        parts[k].set_linewidth(1.2)

    ax.set_xticks(np.arange(1, len(labels)+1))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel('Error (rad)')
    ax.set_title(f'Violin plot of {col}')

    _two_dp_ticks(ax, y=True)
    _legend_below(ax, labels=labels, colors=colors)   # custom legend for violins
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_improvement_by_step(baseline: pd.DataFrame, candidate: pd.DataFrame, col: str, out_path: str):
    _presentation_style()
    dfs = _trim_to_common_length([baseline, candidate])
    b, c = dfs
    diff = b[col].values - c[col].values
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(b['step'], diff, label=f"Improvement (baseline − candidate) {col}", linewidth=2.0, color='C0')
    ax.axhline(0.0, color='k', linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Δ Error (rad)')
    ax.set_title(f'Per-step improvement over baseline ({col})')
    _two_dp_ticks(ax, y=True)
    _legend_below(ax)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def render_summary_table(aggregates: List[Tuple[str, dict]], baseline_idx: int, out_path: str):
    labels = [lbl for lbl,_ in aggregates]
    metrics = ['mae_all7','rmse_all7','mae_joints6','rmse_joints6','mae_grip','rmse_grip']
    rows = []
    base = aggregates[baseline_idx][1]
    for i,(lbl,agg) in enumerate(aggregates):
        row = [lbl]
        for m in metrics:
            row.append(agg[m])
        # % improvement vs baseline
        if i != baseline_idx:
            row += [100.0 * (base[m]-agg[m]) / base[m] for m in metrics]
        else:
            row += [0.0 for _ in metrics]
        rows.append(row)

    # build figure table (2 dp formatting)
    cols = ["Run"] + metrics + [m+"_impr%" for m in metrics]
    fig, ax = plt.subplots(figsize=(min(18, 5 + 2*len(labels)), 1 + 0.6*len(labels)))
    ax.axis('off')
    def fmt(v):
        return f"{v:.2f}" if isinstance(v, (float, np.floating)) else v

    cell_text = [[fmt(v) for v in r] for r in rows]
    table = ax.table(cellText=cell_text, colLabels=cols, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.05, 1.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

# ---------- Presentation extras (MAE-only bars) ----------

def plot_aggregate_bars_pres(aggregates: List[Tuple[str, dict]], out_path: str):
    _presentation_style()
    labels = [lbl for lbl, _ in aggregates]
    metrics = ['mae_all7', 'mae_joints6', 'mae_grip']
    values = np.array([[agg[m] for m in metrics] for _, agg in aggregates])

    colors = plt.cm.tab10.colors[:len(labels)]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.8 / len(labels)

    for i, lbl in enumerate(labels):
        ax.bar(x + i * width - (len(labels) - 1) * width / 2,
               values[i],
               width, label=lbl,
               color=colors[i % len(colors)], alpha=0.9)
        for j, v in enumerate(values[i]):
            ax.text(x[j] + i * width - (len(labels) - 1) * width / 2,
                    v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(["MAE (All 7)", "MAE (Joints 6)", "MAE (Gripper)"])
    ax.set_ylabel("Mean Absolute Error (rad)")
    ax.set_title("Model Comparison – Mean Absolute Error")

    _two_dp_ticks(ax, y=True)
    _legend_below(ax)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_violin_pres(dfs: List[pd.DataFrame], labels: List[str], col: str, out_path: str):
    _presentation_style()
    data = [df[col].dropna().values for df in dfs]
    colors = plt.cm.tab10.colors[:len(labels)]

    fig, ax = plt.subplots(figsize=(8, 6))
    parts = ax.violinplot(data, showmeans=True, showmedians=True, widths=0.7)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_edgecolor("black")
        pc.set_alpha(0.8)

    for partname in ('cbars', 'cmins', 'cmaxes'):
        vp = parts[partname]
        vp.set_edgecolor("black")
        vp.set_linewidth(1.2)

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("MAE (rad)")
    ax.set_title("Distribution of Mean Absolute Error Across Models")

    _two_dp_ticks(ax, y=True)
    # _legend_below(ax, labels=labels, colors=colors)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('items', nargs='+', help='Pairs: CSV LABEL [CSV LABEL ...]')
    args = parser.parse_args()

    if len(args.items) % 2 != 0:
        raise SystemExit("Provide pairs: CSV_PATH LABEL [CSV PATH LABEL ...]")

    _ensure_dir(args.out_dir)

    pairs: List[Tuple[str,str]] = [(args.items[i], args.items[i+1]) for i in range(0, len(args.items), 2)]

    dfs: List[pd.DataFrame] = []
    labels: List[str] = []
    for path, lbl in pairs:
        df = _read_csv_labeled(path, lbl)
        dfs.append(df)
        labels.append(lbl)

    # Aggregates and save JSON
    aggs = [(lbl, _aggregate(df)) for lbl, df in zip(labels, dfs)]
    with open(os.path.join(args.out_dir, 'aggregates.json'), 'w') as f:
        json.dump({lbl: agg for lbl,agg in aggs}, f, indent=2)

    # 1) Aggregate bars (full: MAE+RMSE)
    plot_aggregate_bars(aggs, os.path.join(args.out_dir, 'aggregate_bars.png'))

    # 2-4) Per-step overlays
    plot_per_step_overlay(dfs, labels, 'mae_all7', os.path.join(args.out_dir, 'per_step_mae_all7.png'))
    plot_per_step_overlay(dfs, labels, 'mae_joints6', os.path.join(args.out_dir, 'per_step_mae_joints6.png'))
    plot_per_step_overlay(dfs, labels, 'mae_grip', os.path.join(args.out_dir, 'per_step_mae_grip.png'))

    # 5) Rolling average (25-step)
    plot_rolling(dfs, labels, 'mae_all7', os.path.join(args.out_dir, 'rolling_mae_all7.png'), window=25)

    # 6) Histogram & 7) CDF
    plot_hist(dfs, labels, 'mae_all7', os.path.join(args.out_dir, 'hist_mae_all7.png'))
    plot_cdf(dfs, labels, 'mae_all7', os.path.join(args.out_dir, 'cdf_mae_all7.png'))

    # 8) Violin
    plot_violin(dfs, labels, 'mae_all7', os.path.join(args.out_dir, 'violin_mae_all7.png'))

    # 9) Improvement curve (only if >=2 runs) – uses first as baseline
    if len(dfs) >= 2:
        for i in range(1, len(dfs)):
            outp = os.path.join(args.out_dir, f'improvement_vs_{labels[i]}.png')
            plot_improvement_by_step(dfs[0], dfs[i], 'mae_all7', outp)

    # 10) Summary table (first run as baseline)
    render_summary_table(aggs, baseline_idx=0, out_path=os.path.join(args.out_dir, 'summary_table.png'))

    # Presentation variants
    plot_aggregate_bars_pres(aggs, os.path.join(args.out_dir, 'aggregate_bars_presentation.png'))
    plot_violin_pres(dfs, labels, 'mae_all7', os.path.join(args.out_dir, 'violin_mae_all7_presentation.png'))

    # Optional: vs qdelta plots if present
    if 'mae_vs_qdelta_all7' in dfs[0].columns:
        plot_per_step_overlay(dfs, labels, 'mae_vs_qdelta_all7',
                              os.path.join(args.out_dir, 'per_step_mae_vs_qdelta_all7.png'))
        plot_hist(dfs, labels, 'mae_vs_qdelta_all7',
                  os.path.join(args.out_dir, 'hist_mae_vs_qdelta_all7.png'))
        plot_cdf(dfs, labels, 'mae_vs_qdelta_all7',
                 os.path.join(args.out_dir, 'cdf_mae_vs_qdelta_all7.png'))

    print(f"\n[OK] Saved all plots to {args.out_dir}\n")


if __name__ == '__main__':
    main()
