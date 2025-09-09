#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot evaluation metrics from one or more eval_* runs.
- Supports single metrics.csv (per-step plots).
- If multiple CSVs are given, overlays aggregate metrics (epoch/diffusion vs pi0, etc.).

Usage examples:
  # Single run (per-step traces)
  python plot_eval_metrics.py metrics.csv

  # Compare multiple runs (aggregate bar chart)
  python plot_eval_metrics.py run1/metrics.csv run2/metrics.csv run3/metrics.csv
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


def plot_single(csv_path: str, out_dir: str = None):
    df = pd.read_csv(csv_path)

    if out_dir is None:
        out_dir = os.path.dirname(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    # --- Per-step errors ---
    plt.figure(figsize=(12, 6))
    plt.plot(df["step"], df["mae_all7"], label="MAE all7")
    plt.plot(df["step"], df["rmse_all7"], label="RMSE all7")
    plt.xlabel("Step")
    plt.ylabel("Error (rad)")
    plt.title("Per-step error (all 7 DoF)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "error_all7.png"))
    plt.close()

    # --- Joints vs Gripper ---
    plt.figure(figsize=(12, 6))
    plt.plot(df["step"], df["mae_joints6"], label="MAE joints6")
    plt.plot(df["step"], df["mae_grip"], label="MAE gripper")
    plt.xlabel("Step")
    plt.ylabel("Error (rad)")
    plt.title("Per-step error (joints vs gripper)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "error_joints_vs_gripper.png"))
    plt.close()

    # --- Proxy vs qdelta ---
    if "mae_vs_qdelta_all7" in df.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(df["step"], df["mae_vs_qdelta_all7"], label="MAE vs qdelta")
        plt.plot(df["step"], df["rmse_vs_qdelta_all7"], label="RMSE vs qdelta")
        plt.xlabel("Step")
        plt.ylabel("Error (rad)")
        plt.title("Action vs observed joint delta")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "error_vs_qdelta.png"))
        plt.close()

    print(f"[PLOT] Saved per-step plots to {out_dir}")


def plot_compare(csv_paths: list, out_dir: str = "./comparison_plots"):
    os.makedirs(out_dir, exist_ok=True)

    summary = []
    for path in csv_paths:
        df = pd.read_csv(path)
        label = os.path.basename(os.path.dirname(path)) or os.path.basename(path)
        row = {
            "run": label,
            "mae_all7": df["mae_all7"].mean(),
            "rmse_all7": df["rmse_all7"].mean(),
            "mae_joints6": df["mae_joints6"].mean(),
            "rmse_joints6": df["rmse_joints6"].mean(),
            "mae_grip": df["mae_grip"].mean(),
            "rmse_grip": df["rmse_grip"].mean(),
        }
        summary.append(row)

    summary_df = pd.DataFrame(summary)
    summary_df.set_index("run", inplace=True)
    print("\n[SUMMARY]\n", summary_df)

    # Plot comparison bar charts
    metrics_to_plot = ["mae_all7", "rmse_all7", "mae_joints6", "rmse_joints6", "mae_grip", "rmse_grip"]
    for metric in metrics_to_plot:
        summary_df[metric].plot(kind="bar", figsize=(10, 6), title=f"Comparison: {metric}")
        plt.ylabel("Error (rad)")
        plt.grid(True, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"compare_{metric}.png"))
        plt.close()

    print(f"[PLOT] Saved comparison plots to {out_dir}")


if __name__ == "__main__":
    csvs = sys.argv[1:]
    if not csvs:
        print("Usage: python plot_eval_metrics.py metrics.csv [metrics2.csv ...]")
        sys.exit(1)

    if len(csvs) == 1:
        plot_single(csvs[0])
    else:
        plot_compare(csvs)
