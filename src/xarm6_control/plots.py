#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_supervisor_plots.py  —  Thesis-ready plots for supervisor metrics

Generates publication-quality figures comparing one or more evaluation runs
(e.g., Diffusion/ImageNet-FT vs AgriVLA) from their metrics.csv files.

Outputs (saved to --out-dir):
  1) aggregate_bars.png / .pdf         – side-by-side bars for MAE/RMSE
  2) per_step_mae_all7.png / .pdf      – step-wise overlay (raw + rolling mean ±1σ)
  3) per_step_mae_joints6.png / .pdf
  4) per_step_mae_grip.png / .pdf
  5) rolling_mae_all7.png / .pdf       – pure rolling mean overlay
  6) hist_mae_all7.png / .pdf          – error distribution (density)
  7) cdf_mae_all7.png / .pdf           – cumulative distribution
  8) violin_mae_all7.png / .pdf        – spread & medians
  9) improvement_vs_<label>.png/.pdf   – (baseline − candidate) per-step improvement
 10) summary_table.png / .pdf          – key aggregates & % improvements (table render)
 11) aggregates.json                   – numeric aggregates for reproducibility

Notes:
- Provide pairs of arguments: CSV_PATH LABEL [CSV_PATH LABEL ...]
- The script trims step-wise overlays to the shortest run length to align steps.

Example (only ImageNet-FT vs AgriVLA):
  uv run src/xarm6_control/make_supervisor_plots.py \\
    --out-dir /path/to/out \\
    /path/to/diffusion/metrics.csv "ImageNet-FT" \\
    /path/to/pi0_fineTuned/metrics.csv "AgriVLA" \\
    --pres
"""
import os
import json
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, AutoMinorLocator
from matplotlib.patches import Patch

# ---------------------- Utilities ----------------------

def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def _read_csv_labeled(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    df["run"] = label
    required = [
        "step", "mae_all7", "rmse_all7",
        "mae_joints6", "rmse_joints6",
        "mae_grip", "rmse_grip"
    ]
    missing = [k for k in required if k not in df.columns]
    if missing:
        raise KeyError(f"Missing columns {missing} in {path}")
    return df

def _aggregate(df: pd.DataFrame) -> dict:
    nm = lambda s: float(np.nanmean(s.values))
    return {
        "mae_all7":   nm(df["mae_all7"]),
        "rmse_all7":  nm(df["rmse_all7"]),
        "mae_joints6":nm(df["mae_joints6"]),
        "rmse_joints6":nm(df["rmse_joints6"]),
        "mae_grip":   nm(df["mae_grip"]),
        "rmse_grip":  nm(df["rmse_grip"]),
        "N_steps":    int(df.shape[0]),
    }

def _trim_to_common_length(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    L = min(int(df["step"].max()) + 1 for df in dfs)
    return [df[df["step"] < L].reset_index(drop=True) for df in dfs]

# ---------------------- Theme ----------------------

def _thesis_style():
    """Unified thesis style: high DPI, serif, subtle grid, tidy spines."""
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.edgecolor": "0.3",
        "axes.linewidth": 0.9,
        "pdf.fonttype": 42,   # editable text in Illustrator
        "ps.fonttype": 42,
        "font.family": "serif",
    })

def _two_dp_ticks(ax, x=False, y=True):
    if x: ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    if y: ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

def _decorate_axes(ax):
    ax.grid(True, which="major", axis="both", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.4, alpha=0.4)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.margins(x=0)

def _legend_below(ax, labels=None, colors=None):
    if labels is not None and colors is not None:
        handles = [Patch(facecolor=colors[i % len(colors)], edgecolor="black", label=lbl)
                   for i, lbl in enumerate(labels)]
        leg = ax.legend(handles=handles, loc="upper center",
                        bbox_to_anchor=(0.5, -0.12), ncol=min(4, len(labels)),
                        frameon=False)
    else:
        handles, _labels = ax.get_legend_handles_labels()
        leg = ax.legend(handles, _labels, loc="upper center",
                        bbox_to_anchor=(0.5, -0.12), ncol=min(4, len(_labels)),
                        frameon=False)
    return leg

def _save(fig, out_path: str, save_pdf: bool = True):
    fig.savefig(out_path, bbox_inches="tight")
    if save_pdf:
        root, _ = os.path.splitext(out_path)
        fig.savefig(root + ".pdf", bbox_inches="tight")
    plt.close(fig)

# ---------------------- Human labels ----------------------

_NICE = {
    "mae_all7":      ("Per-step MAE (all 7 DOF)", "Error (rad)"),
    "mae_joints6":   ("Per-step MAE (joints ×6)", "Error (rad)"),
    "mae_grip":      ("Per-step MAE (gripper)",   "Error (rad)"),
    "rmse_all7":     ("RMSE (all 7 DOF)", "Error (rad)"),
    "rmse_joints6":  ("RMSE (joints ×6)", "Error (rad)"),
    "rmse_grip":     ("RMSE (gripper)",    "Error (rad)"),
}

def _nice(col: str):
    return _NICE.get(col, (col, "Value"))

# ---------------------- Plotters ----------------------

def plot_aggregate_bars(aggregates: List[Tuple[str, dict]], out_path: str, save_pdf=True):
    _thesis_style()
    labels = [lbl for lbl, _ in aggregates]
    metrics = ["mae_all7","rmse_all7","mae_joints6","rmse_joints6","mae_grip","rmse_grip"]
    vals = np.array([[agg[m] for m in metrics] for _, agg in aggregates])

    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(len(metrics))
    width = 0.8 / len(labels)
    colors = plt.cm.tab10.colors[:len(labels)]

    for i, lbl in enumerate(labels):
        ax.bar(x + i*width - (len(labels)-1)*width/2, vals[i], width,
               label=lbl, color=colors[i], alpha=0.9, edgecolor="black", linewidth=0.7)

        # value labels on bars (2 dp)
        for j, v in enumerate(vals[i]):
            ax.text(x[j] + i*width - (len(labels)-1)*width/2, v,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(["MAE all7","RMSE all7","MAE joints6","RMSE joints6","MAE grip","RMSE grip"], rotation=0)
    ax.set_ylabel("Error (rad)")
    ax.set_title("Aggregate MAE / RMSE across runs")
    _two_dp_ticks(ax, y=True)
    _decorate_axes(ax)
    _legend_below(ax)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    _save(fig, out_path, save_pdf)

def _ymax_sensible(series_list: List[np.ndarray]) -> float:
    """Use 98th percentile (plus 5%) to keep extreme spikes from crushing scale."""
    q = [np.nanpercentile(s, 98) for s in series_list]
    m = max(q) if q else 1.0
    return float(1.05 * m)

def plot_per_step_overlay(dfs: List[pd.DataFrame], labels: List[str], col: str,
                          out_path: str, window: int = 25, save_pdf=True):
    _thesis_style()
    dfs = _trim_to_common_length(dfs)
    fig, ax = plt.subplots(figsize=(12, 3.6))
    colors = plt.cm.tab10.colors[:len(labels)]

    # compute sensible ylim on raw values
    raw_series = [df[col].values for df in dfs]
    ax.set_ylim(0, _ymax_sensible(raw_series))

    # plot each run: faint raw + strong rolling mean with ±1σ ribbon
    for (df, lbl, c) in zip(dfs, labels, colors):
        raw = df[col].astype(float).to_numpy()
        roll  = pd.Series(raw).rolling(window=window, min_periods=1)
        mean  = roll.mean().to_numpy()
        std   = roll.std().fillna(0.0).to_numpy()

        # legend label with μ ± σ over the aligned sequence
        mu, sigma = float(np.nanmean(raw)), float(np.nanstd(raw))
        leg_lbl = f"{lbl} (μ={mu:.3f}, σ={sigma:.3f})"

        ax.plot(df["step"], raw, color=c, alpha=0.25, linewidth=1.0, label=None)
        ax.fill_between(df["step"], mean - std, mean + std, color=c, alpha=0.15, linewidth=0)
        ax.plot(df["step"], mean, color=c, linewidth=2.4, label=leg_lbl)

    title, ylab = _nice(col)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    _two_dp_ticks(ax, y=True)
    _decorate_axes(ax)
    _legend_below(ax)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    _save(fig, out_path, save_pdf)

def plot_rolling(dfs: List[pd.DataFrame], labels: List[str], col: str, out_path: str,
                 window: int = 25, save_pdf=True):
    _thesis_style()
    dfs = _trim_to_common_length(dfs)
    fig, ax = plt.subplots(figsize=(12, 3.6))
    colors = plt.cm.tab10.colors[:len(labels)]

    for (df, lbl, c) in zip(dfs, labels, colors):
        roll = df[col].rolling(window=window, min_periods=1).mean()
        ax.plot(df["step"], roll, label=lbl, linewidth=2.4, color=c)

    ax.set_xlabel("Step"); ax.set_ylabel("Error (rad)")
    ax.set_title(f"Rolling Mean (win={window}) — {col}")
    _two_dp_ticks(ax, y=True)
    _decorate_axes(ax)
    _legend_below(ax)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    _save(fig, out_path, save_pdf)

def plot_hist(dfs: List[pd.DataFrame], labels: List[str], col: str, out_path: str,
              bins: int = 40, save_pdf=True):
    _thesis_style()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    colors = plt.cm.tab10.colors[:len(labels)]
    for (df, lbl, c) in zip(dfs, labels, colors):
        ax.hist(df[col].dropna().values, bins=bins, alpha=0.45, density=True,
                label=lbl, color=c, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Error (rad)"); ax.set_ylabel("Density")
    ax.set_title(f"Histogram of {col}")
    _two_dp_ticks(ax, x=True, y=True)
    _decorate_axes(ax)
    _legend_below(ax)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    _save(fig, out_path, save_pdf)

def plot_cdf(dfs: List[pd.DataFrame], labels: List[str], col: str, out_path: str, save_pdf=True):
    _thesis_style()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    colors = plt.cm.tab10.colors[:len(labels)]
    for (df, lbl, c) in zip(dfs, labels, colors):
        x = np.sort(df[col].dropna().values)
        y = np.linspace(0, 1, len(x), endpoint=True)
        ax.plot(x, y, label=lbl, linewidth=2.2, color=c)
    ax.set_xlabel("Error (rad)"); ax.set_ylabel("Cumulative fraction")
    ax.set_title(f"CDF of {col}")
    _two_dp_ticks(ax, x=True, y=True)
    _decorate_axes(ax)
    _legend_below(ax)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    _save(fig, out_path, save_pdf)

def plot_improvement_by_step(baseline: pd.DataFrame, candidate: pd.DataFrame,
                             col: str, out_path: str, save_pdf=True):
    _thesis_style()
    baseline, candidate = _trim_to_common_length([baseline, candidate])
    diff = baseline[col].values - candidate[col].values
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.plot(baseline["step"], diff, label=f"Δ (baseline − candidate) {col}",
            linewidth=2.2, color="C0")
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Step"); ax.set_ylabel("Δ Error (rad)")
    ax.set_title(f"Per-step improvement over baseline ({col})")
    _two_dp_ticks(ax, y=True)
    _decorate_axes(ax)
    _legend_below(ax)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    _save(fig, out_path, save_pdf)

def render_summary_table(aggregates: List[Tuple[str, dict]], baseline_idx: int, out_path: str, save_pdf=True):
    labels = [lbl for lbl,_ in aggregates]
    metrics = ["mae_all7","rmse_all7","mae_joints6","rmse_joints6","mae_grip","rmse_grip"]
    base = aggregates[baseline_idx][1]
    rows = []
    for i,(lbl,agg) in enumerate(aggregates):
        row = [lbl] + [agg[m] for m in metrics]
        if i != baseline_idx:
            row += [100.0*(base[m]-agg[m])/base[m] for m in metrics]
        else:
            row += [0.0 for _ in metrics]
        rows.append(row)

    cols = ["Run"] + metrics + [m+"_impr%" for m in metrics]
    fig, ax = plt.subplots(figsize=(min(18, 5 + 2*len(labels)), 1 + 0.6*len(labels)))
    ax.axis("off")
    def fmt(v):
        return f"{v:.3f}" if isinstance(v, (float, np.floating)) else v
    cell_text = [[fmt(v) for v in r] for r in rows]
    table = ax.table(cellText=cell_text, colLabels=cols, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.05, 1.25)
    fig.tight_layout()
    _save(fig, out_path, save_pdf)

# ---------------------- Presentation (MAE-only) ----------------------

def plot_aggregate_bars_pres(aggregates: List[Tuple[str, dict]], out_path: str, save_pdf=True):
    _thesis_style()
    labels = [lbl for lbl,_ in aggregates]
    metrics = ["mae_all7", "mae_joints6", "mae_grip"]
    vals = np.array([[agg[m] for m in metrics] for _,agg in aggregates])
    colors = plt.cm.tab10.colors[:len(labels)]

    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    x = np.arange(len(metrics))
    width = 0.8 / len(labels)
    for i, lbl in enumerate(labels):
        ax.bar(x + i*width - (len(labels)-1)*width/2, vals[i], width,
               color=colors[i], alpha=0.9, edgecolor="black", linewidth=0.7, label=lbl)
    ax.set_xticks(x)
    ax.set_xticklabels(["MAE (All 7)", "MAE (Joints 6)", "MAE (Gripper)"])
    ax.set_ylabel("Mean Absolute Error (rad)")
    ax.set_title("Model Comparison — Mean Absolute Error")
    _two_dp_ticks(ax, y=True)
    _decorate_axes(ax)
    _legend_below(ax)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    _save(fig, out_path, save_pdf)

def plot_violin_pres(dfs: List[pd.DataFrame], labels: List[str], col: str, out_path: str, save_pdf=True):
    _thesis_style()
    data = [df[col].dropna().values for df in dfs]
    n = len(labels)
    colors = plt.cm.tab10.colors[:n]
    fig, ax = plt.subplots(figsize=(max(7.5, 1.4*n + 2), 4.2))
    parts = ax.violinplot(data, showmeans=True, showmedians=True, widths=0.7)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_edgecolor("black")
        pc.set_alpha(0.85)
    for k in ("cbars","cmins","cmaxes","cmeans","cmedians"):
        vp = parts[k]; vp.set_edgecolor("black"); vp.set_linewidth(1.1)
    ax.set_xticks(np.arange(1, n+1))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Mean Absolute Error (rad)")
    ax.set_title("Distribution of Mean Absolute Error Across Models")
    _two_dp_ticks(ax, y=True)
    _decorate_axes(ax)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    _save(fig, out_path, save_pdf)

def plot_bar_violin_combined_pres(aggregates: List[Tuple[str, dict]],
                                  dfs: List[pd.DataFrame],
                                  labels: List[str],
                                  out_path: str,
                                  save_pdf=True):
    _thesis_style()
    metrics = ["mae_all7"]
    vals = np.array([[agg[m] for m in metrics] for _,agg in aggregates])
    colors = plt.cm.tab10.colors[:len(labels)]
    data = [df["mae_all7"].dropna().values for df in dfs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 2]})
    fig.subplots_adjust(wspace=0.25, bottom=0.28)

    x = np.arange(len(metrics)); width = 0.8 / len(labels)
    for i, lbl in enumerate(labels):
        ax1.bar(x + i*width - (len(labels)-1)*width/2, vals[i], width,
                color=colors[i], alpha=0.9, edgecolor="black", linewidth=0.8)
    ax1.set_ylabel("Mean Absolute Error (rad)")
    ax1.set_title("Model Comparison", fontweight="bold", fontsize=12)
    _two_dp_ticks(ax1, y=True)
    ax1.set_xticks([]); ax1.set_xticklabels([]); ax1.set_xlabel("")
    for s in ["top","right"]: ax1.spines[s].set_visible(False)

    parts = ax2.violinplot(data, showmeans=True, showmedians=True, widths=0.7)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_edgecolor("black"); pc.set_alpha(0.9)
    for k in ("cbars","cmins","cmaxes","cmeans","cmedians"):
        vp = parts[k]; vp.set_edgecolor("black"); vp.set_linewidth(1.1)
    ax2.set_ylabel("Mean Absolute Error (rad)")
    ax2.set_title("Distribution of Mean Absolute Error Across Models", fontweight="bold", fontsize=12)
    _two_dp_ticks(ax2, y=True)
    ax2.set_xticks([]); ax2.set_xticklabels([]); ax2.set_xlabel("")
    for s in ["top","right"]: ax2.spines[s].set_visible(False)

    handles = [Patch(facecolor=colors[i % len(colors)], edgecolor="black", label=lbl) for i, lbl in enumerate(labels)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.04),
               ncol=min(6, len(labels)), frameon=False, fontsize=10)

    fig.tight_layout(rect=[0, 0.14, 1, 1])
    _save(fig, out_path, save_pdf)

# ---------------------- Main ----------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("items", nargs="+", help="Pairs: CSV LABEL [CSV LABEL ...]")
    p.add_argument("--pres", action="store_true", help="Generate presentation variants")
    p.add_argument("--no-pdf", action="store_true", help="Do not save PDF alongside PNG")
    p.add_argument("--window", type=int, default=25, help="Rolling window for smoothing")
    args = p.parse_args()

    if len(args.items) % 2 != 0:
        raise SystemExit("Provide pairs: CSV_PATH LABEL [CSV_PATH LABEL ...]")

    _ensure_dir(args.out_dir)
    pairs: List[Tuple[str, str]] = [(args.items[i], args.items[i+1]) for i in range(0, len(args.items), 2)]

    dfs: List[pd.DataFrame] = []
    labels: List[str] = []
    for path, lbl in pairs:
        df = _read_csv_labeled(path, lbl)
        dfs.append(df); labels.append(lbl)

    # Aggregates + JSON
    aggs = [(lbl, _aggregate(df)) for lbl, df in zip(labels, dfs)]
    with open(os.path.join(args.out_dir, "aggregates.json"), "w") as f:
        json.dump({lbl: agg for lbl, agg in aggs}, f, indent=2)

    save_pdf = not args.no_pdf

    # Presentation focus (MAE) or full set
    if args.pres:
        plot_aggregate_bars_pres(aggs, os.path.join(args.out_dir, "aggregate_bars_presentation.png"), save_pdf)
        plot_violin_pres(dfs, labels, "mae_all7", os.path.join(args.out_dir, "violin_mae_all7_presentation.png"), save_pdf)
        plot_bar_violin_combined_pres(aggs, dfs, labels, os.path.join(args.out_dir, "mae_bar_violin_combined.png"), save_pdf)
    else:
        plot_aggregate_bars(aggs, os.path.join(args.out_dir, "aggregate_bars.png"), save_pdf)

        # Per-step overlays (thesis style: raw + rolling mean ±1σ)
        plot_per_step_overlay(dfs, labels, "mae_all7",
                              os.path.join(args.out_dir, "per_step_mae_all7.png"),
                              window=args.window, save_pdf=save_pdf)
        plot_per_step_overlay(dfs, labels, "mae_joints6",
                              os.path.join(args.out_dir, "per_step_mae_joints6.png"),
                              window=args.window, save_pdf=save_pdf)
        plot_per_step_overlay(dfs, labels, "mae_grip",
                              os.path.join(args.out_dir, "per_step_mae_grip.png"),
                              window=args.window, save_pdf=save_pdf)

        # Rolling-only overview
        plot_rolling(dfs, labels, "mae_all7",
                     os.path.join(args.out_dir, "rolling_mae_all7.png"),
                     window=args.window, save_pdf=save_pdf)

        # Distribution views
        plot_hist(dfs, labels, "mae_all7",
                  os.path.join(args.out_dir, "hist_mae_all7.png"), save_pdf)
        plot_cdf(dfs, labels, "mae_all7",
                  os.path.join(args.out_dir, "cdf_mae_all7.png"), save_pdf)

        # Improvement curves vs first run as baseline
        if len(dfs) >= 2:
            base_df, base_label = dfs[0], labels[0]
            for i in range(1, len(dfs)):
                outp = os.path.join(args.out_dir, f"improvement_vs_{labels[i]}.png")
                plot_improvement_by_step(base_df, dfs[i], "mae_all7", outp, save_pdf)

        # Summary table (first run as baseline)
        render_summary_table(aggs, baseline_idx=0,
                             out_path=os.path.join(args.out_dir, "summary_table.png"),
                             save_pdf=save_pdf)

        # Optional: vs qdelta if present
        if "mae_vs_qdelta_all7" in dfs[0].columns:
            plot_per_step_overlay(dfs, labels, "mae_vs_qdelta_all7",
                                  os.path.join(args.out_dir, "per_step_mae_vs_qdelta_all7.png"),
                                  window=args.window, save_pdf=save_pdf)
            plot_hist(dfs, labels, "mae_vs_qdelta_all7",
                      os.path.join(args.out_dir, "hist_mae_vs_qdelta_all7.png"), save_pdf)
            plot_cdf(dfs, labels, "mae_vs_qdelta_all7",
                      os.path.join(args.out_dir, "cdf_mae_vs_qdelta_all7.png"), save_pdf)

    print(f"\n[OK] Saved all plots to {args.out_dir}\n")

if __name__ == "__main__":
    main()
