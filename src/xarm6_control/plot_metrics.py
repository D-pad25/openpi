#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_supervisor_plots.py (thesis-ready)

Generates publication-quality plots comparing multiple evaluation runs
from their metrics.csv files.

Outputs (saved to --out-dir; each as PNG, PDF, SVG):
  - aggregate_bars.(png|pdf|svg)          (MAE/RMSE with SEM error bars)
  - per_step_mae_all7.(...)               (trimmed to common length)
  - per_step_mae_joints6.(...)
  - per_step_mae_grip.(...)
  - rolling_mae_all7.(...)
  - hist_mae_all7.(...)
  - cdf_mae_all7.(...)
  - violin_mae_all7.(...)
  - improvement_vs_<label>.png            (baseline − candidate per-step)
  - summary_table.(png|pdf|svg)
  - aggregates.json

If *_vs_qdelta columns exist, analogous plots are produced.
Usage example:
  uv run src/xarm6_control/make_supervisor_plots.py \
    --out-dir /path/to/out \
    /path/runA/metrics.csv "pi0_base" \
    /path/runB/metrics.csv "pi0_fast_base"
"""
import os
import json
import math
import argparse
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# -------------------- Config --------------------

# Okabe–Ito color-blind-safe palette
PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # red
    "#CC79A7",  # purple
    "#56B4E9",  # sky
    "#F0E442",  # yellow
    "#000000",  # black
]

LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
HATCHES = ["", "//", "\\\\", "xx", "..", "++"]

PRETTY_METRIC = {
    "mae_all7":        "MAE (all 7)",
    "rmse_all7":       "RMSE (all 7)",
    "mae_joints6":     "MAE (joints ×6)",
    "rmse_joints6":    "RMSE (joints ×6)",
    "mae_grip":        "MAE (gripper)",
    "rmse_grip":       "RMSE (gripper)",
    "mae_vs_qdelta_all7":  "MAE (Δq, all 7)",
    "rmse_vs_qdelta_all7": "RMSE (Δq, all 7)",
}

AGG_METRICS = ["mae_all7","rmse_all7","mae_joints6","rmse_joints6","mae_grip","rmse_grip"]

# -------------------- IO helpers --------------------

def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def _read_csv_labeled(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    df["run"] = label
    required = ["step","mae_all7","rmse_all7","mae_joints6","rmse_joints6","mae_grip","rmse_grip"]
    for k in required:
        if k not in df.columns:
            raise KeyError(f"Missing column '{k}' in {path}")
    return df

def _trim_to_common_length(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    L = min(int(df["step"].max()) + 1 for df in dfs)
    return [df[df["step"] < L].reset_index(drop=True) for df in dfs]

def _savefig(fig: mpl.figure.Figure, out_path_no_ext: str):
    fig.savefig(out_path_no_ext + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path_no_ext + ".pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out_path_no_ext + ".svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)

# -------------------- Style --------------------

def _nice_style(paper: bool = False):
    # Default to a print-friendly serif; fall back gracefully.
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.size": 11.5,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "axes.grid": True,
        "grid.color": "#aaaaaa",
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.titlepad": 8,
    })
    if paper:
        # Try LaTeX text for theses; ignore if not present.
        try:
            mpl.rcParams["text.usetex"] = True
            mpl.rcParams["font.serif"] = ["Times New Roman", "Times", "CMU Serif", "Computer Modern Roman"]
        except Exception:
            pass

# -------------------- Stats --------------------

def _agg_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Return mean and SEM for each aggregate metric."""
    out = {}
    for m in AGG_METRICS:
        vals = df[m].astype(float).to_numpy()
        vals = vals[np.isfinite(vals)]
        mean = float(np.nanmean(vals)) if vals.size else float("nan")
        sem = float(np.nanstd(vals, ddof=1) / math.sqrt(max(len(vals), 1))) if len(vals) > 1 else 0.0
        out[m] = {"mean": mean, "sem": sem}
    out["N_steps"] = {"mean": int(df.shape[0]), "sem": 0.0}
    return out

def _freedman_diaconis_bins(x: np.ndarray) -> int:
    x = x[np.isfinite(x)]
    if x.size < 2:
        return 10
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    bw = 2 * iqr / (x.size ** (1/3))
    if bw <= 0:
        return 40
    return max(10, int(np.ceil((x.max() - x.min()) / bw)))

# -------------------- Plotters --------------------

def plot_aggregate_bars(aggs: List[Tuple[str, Dict]], out_noext: str):
    """Minimal, colour-only aggregate bars (no hatches, no outlines, no annotations)."""
    labels = [lbl for lbl, _ in aggs]

    # Means only (we intentionally ignore SEM for a cleaner plot)
    means = np.array([[aggs[i][1][m]["mean"] for m in AGG_METRICS]
                      for i in range(len(aggs))])

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    x = np.arange(len(AGG_METRICS))
    width = min(0.80 / max(len(labels), 1), 0.22)

    for i, lbl in enumerate(labels):
        off = x + (i - (len(labels) - 1) / 2.0) * width
        ax.bar(
            off,
            means[i],
            width,
            label=lbl,
            color=colors[i]
            # edgecolor=None,   # no outlines
            # linewidth=0       # ensure no stroke
        )

    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY_METRIC[m] for m in AGG_METRICS], rotation=8)
    ax.set_ylabel("Error (rad)")
    ax.set_title("Aggregate error (mean) — lower is better", pad=8)
    ax.set_ylim(bottom=0)

    # Simple, compact legend above the axes; no frame for less clutter
    ax.legend(
        ncol=min(3, len(labels)),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        handlelength=1.4,
        columnspacing=1.2
    )

    # Slightly tighter layout since legend is outside
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _savefig(fig, out_noext)


def plot_per_step_overlay(dfs: List[pd.DataFrame], labels: List[str], col: str, out_noext: str):
    dfs = _trim_to_common_length(dfs)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    for i, (df, lbl) in enumerate(zip(dfs, labels)):
        ax.plot(df["step"], df[col], label=lbl, linewidth=1.8,
                color=colors[i], linestyle=LINESTYLES[i % len(LINESTYLES)])
    ax.set_xlabel("Step")
    ax.set_ylabel("Error (rad)")
    ax.set_title(f"{PRETTY_METRIC.get(col, col)} per step (trimmed to common length)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout(rect=[0, 0, 0.8, 1])
    _savefig(fig, out_noext)

def plot_rolling(dfs: List[pd.DataFrame], labels: List[str], col: str, out_noext: str, window: int = 25):
    dfs = _trim_to_common_length(dfs)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    for i, (df, lbl) in enumerate(zip(dfs, labels)):
        roll = df[col].rolling(window=window, min_periods=1).mean()
        ax.plot(df["step"], roll, label=f"{lbl} (win={window})", linewidth=2.2,
                color=colors[i], linestyle=LINESTYLES[i % len(LINESTYLES)])
    ax.set_xlabel("Step")
    ax.set_ylabel("Error (rad)")
    ax.set_title(f"Rolling mean — {PRETTY_METRIC.get(col, col)}")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout(rect=[0, 0, 0.8, 1])
    _savefig(fig, out_noext)

def plot_hist(dfs: List[pd.DataFrame], labels: List[str], col: str, out_noext: str):
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
    all_vals = np.concatenate([df[col].dropna().values for df in dfs])
    bins = _freedman_diaconis_bins(all_vals)
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    for i, (df, lbl) in enumerate(zip(dfs, labels)):
        ax.hist(df[col].dropna().values, bins=bins, histtype="stepfilled",
                alpha=0.45, density=True, label=lbl, color=colors[i], edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Error (rad)")
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution — {PRETTY_METRIC.get(col, col)}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    _savefig(fig, out_noext)

def plot_cdf(dfs: List[pd.DataFrame], labels: List[str], col: str, out_noext: str):
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    for i, (df, lbl) in enumerate(zip(dfs, labels)):
        x = np.sort(df[col].dropna().values)
        if x.size == 0:
            continue
        y = np.linspace(0, 1, len(x), endpoint=True)
        ax.plot(x, y, label=lbl, linewidth=2.0,
                color=colors[i], linestyle=LINESTYLES[i % len(LINESTYLES)])
        # 90th percentile marker
        p90 = np.percentile(x, 90)
        ax.axvline(p90, color=colors[i], linestyle=":", alpha=0.5)
    ax.set_xlabel("Error (rad)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title(f"CDF — {PRETTY_METRIC.get(col, col)} (vertical: 90th pct)")
    ax.set_xlim(left=0)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _savefig(fig, out_noext)

def plot_violin(dfs: List[pd.DataFrame], labels: List[str], col: str, out_noext: str):
    data = [df[col].dropna().values for df in dfs]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
    # color each violin
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(colors[i])
        b.set_edgecolor("black")
        b.set_alpha(0.6)
        b.set_linewidth(0.6)
    # medians
    med = parts["cmedians"]
    med.set_color("#1a1a1a")
    med.set_linewidth(1.8)
    ax.set_xticks(np.arange(1, len(labels)+1))
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("Error (rad)")
    ax.set_title(f"Violin — {PRETTY_METRIC.get(col, col)} (median line)")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    _savefig(fig, out_noext)

def plot_improvement_by_step(baseline: pd.DataFrame, candidate: pd.DataFrame, col: str, label: str, out_noext: str):
    b, c = _trim_to_common_length([baseline, candidate])
    diff = b[col].values - c[col].values
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    ax.plot(b["step"], diff, linewidth=1.8, color="#4C72B0")
    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.set_xlabel("Step")
    ax.set_ylabel("Δ Error (rad)")
    ax.set_title(f"Per-step improvement over baseline — {label} ({PRETTY_METRIC.get(col, col)})")
    fig.tight_layout()
    _savefig(fig, out_noext)

def render_summary_table(aggs: List[Tuple[str, Dict]], baseline_idx: int, out_noext: str):
    labels = [lbl for lbl,_ in aggs]
    metrics = AGG_METRICS
    rows = []
    base = aggs[baseline_idx][1]
    for i,(lbl,stat) in enumerate(aggs):
        row = [lbl]
        for m in metrics:
            row.append(stat[m]["mean"])
        # improvements vs baseline
        base_means = [base[m]["mean"] for m in metrics]
        if i != baseline_idx:
                        row += [
                100.0 * (bm - stat[m]["mean"]) / bm if (isinstance(bm, (int, float)) and bm > 1e-9) else float("nan")
                for m, bm in zip(metrics, base_means)
            ]
        else:
            row += [0.0 for _ in metrics]
        rows.append(row)

    col_headers = (
        ["Run"]
        + [PRETTY_METRIC[m] for m in metrics]
        + [f"{PRETTY_METRIC[m]} impr. (%)" for m in metrics]
    )

    fig, ax = plt.subplots(
        figsize=(min(18, 6 + 1.6 * len(labels)), 1.2 + 0.55 * len(labels))
    )
    ax.axis("off")
    table = ax.table(
        cellText=[
            [f"{v:.3f}" if isinstance(v, (int, float)) and np.isfinite(v) else v for v in r]
            for r in rows
        ],
        colLabels=col_headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.25)
    _savefig(fig, out_noext)


# -------------------- Main --------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Use LaTeX-compatible serif fonts if available.",
    )
    parser.add_argument(
        "items", nargs="+", help='Pairs: CSV_PATH LABEL [CSV_PATH LABEL ...]'
    )
    args = parser.parse_args()

    if len(args.items) % 2 != 0:
        raise SystemExit("Provide pairs: CSV_PATH LABEL [CSV_PATH LABEL ...]")

    _nice_style(paper=args.paper)
    _ensure_dir(args.out_dir)

    # Parse pairs
    pairs: List[Tuple[str, str]] = [
        (args.items[i], args.items[i + 1]) for i in range(0, len(args.items), 2)
    ]

    # Load data
    dfs: List[pd.DataFrame] = []
    labels: List[str] = []
    for path, lbl in pairs:
        df = _read_csv_labeled(path, lbl)
        dfs.append(df)
        labels.append(lbl)

    # Aggregates (mean & SEM) and JSON dump
    aggs_stats: List[Tuple[str, Dict]] = [
        (lbl, _agg_stats(df)) for lbl, df in zip(labels, dfs)
    ]
    with open(os.path.join(args.out_dir, "aggregates.json"), "w") as f:
        json.dump({lbl: stats for lbl, stats in aggs_stats}, f, indent=2)

    # 1) Aggregate bars
    plot_aggregate_bars(aggs_stats, os.path.join(args.out_dir, "aggregate_bars"))

    # 2–4) Per-step overlays
    plot_per_step_overlay(dfs, labels, "mae_all7", os.path.join(args.out_dir, "per_step_mae_all7"))
    plot_per_step_overlay(dfs, labels, "mae_joints6", os.path.join(args.out_dir, "per_step_mae_joints6"))
    plot_per_step_overlay(dfs, labels, "mae_grip", os.path.join(args.out_dir, "per_step_mae_grip"))

    # 5) Rolling average (25-step)
    plot_rolling(dfs, labels, "mae_all7", os.path.join(args.out_dir, "rolling_mae_all7"), window=25)

    # 6) Histogram & 7) CDF
    plot_hist(dfs, labels, "mae_all7", os.path.join(args.out_dir, "hist_mae_all7"))
    plot_cdf(dfs, labels, "mae_all7", os.path.join(args.out_dir, "cdf_mae_all7"))

    # 8) Violin
    plot_violin(dfs, labels, "mae_all7", os.path.join(args.out_dir, "violin_mae_all7"))

    # 9) Improvement curves vs baseline (first run)
    if len(dfs) >= 2:
        for i in range(1, len(dfs)):
            plot_improvement_by_step(
                dfs[0],
                dfs[i],
                "mae_all7",
                labels[i],
                os.path.join(args.out_dir, f"improvement_vs_{labels[i]}"),
            )

    # 10) Summary table (first run as baseline)
    render_summary_table(aggs_stats, baseline_idx=0, out_noext=os.path.join(args.out_dir, "summary_table"))

    # Optional suite for Δq columns (only for runs that have them)
    q_cols = ["mae_vs_qdelta_all7"]
    if any(all(c in df.columns for c in q_cols) for df in dfs):
        dfs_q: List[pd.DataFrame] = []
        labels_q: List[str] = []
        for df, lbl in zip(dfs, labels):
            if all(c in df.columns for c in q_cols):
                dfs_q.append(df)
                labels_q.append(lbl)
        if len(dfs_q) >= 1:
            plot_per_step_overlay(
                dfs_q, labels_q, "mae_vs_qdelta_all7",
                os.path.join(args.out_dir, "per_step_mae_vs_qdelta_all7"),
            )
            plot_hist(
                dfs_q, labels_q, "mae_vs_qdelta_all7",
                os.path.join(args.out_dir, "hist_mae_vs_qdelta_all7"),
            )
            plot_cdf(
                dfs_q, labels_q, "mae_vs_qdelta_all7",
                os.path.join(args.out_dir, "cdf_mae_vs_qdelta_all7"),
            )
            plot_violin(
                dfs_q, labels_q, "mae_vs_qdelta_all7",
                os.path.join(args.out_dir, "violin_mae_vs_qdelta_all7"),
            )

    print(f"\n[OK] Saved all plots to {args.out_dir}\n")


if __name__ == "__main__":
    main()

