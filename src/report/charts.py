"""
Generates matplotlib charts for the report.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.style as mstyle
import numpy as np
from src.connectors.wandb_connector import RunData

mstyle.use("seaborn-v0_8-whitegrid")
COLORS = {"train": "#3498db", "val": "#e74c3c", "lr": "#2ecc71"}


def plot_loss_curves(run: RunData, output_dir: str) -> str:
    """Plot train and val loss curves. Returns path to saved PNG."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    ax = axes[0]
    train_loss_key = next(
        (m for m in run.train_metrics if "loss" in m.lower()), None
    )
    if train_loss_key:
        series = run.get_metric_series(train_loss_key)
        ax.plot(series.values, color=COLORS["train"],
                label="Train Loss", linewidth=2)

    if "val_loss" in run.val_metrics:
        series = run.get_metric_series("val_loss")
        ax.plot(series.values, color=COLORS["val"],
                label="Val Loss", linewidth=2, linestyle="--")

    ax.set_title("Loss Curves", fontsize=13, fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()

    # Accuracy curve
    ax2 = axes[1]
    val_acc_key = next(
        (m for m in run.val_metrics if "acc" in m.lower()), None
    )
    if val_acc_key:
        series = run.get_metric_series(val_acc_key)
        ax2.plot(series.values, color=COLORS["val"],
                 label="Val Accuracy", linewidth=2)
        ax2.set_title("Validation Accuracy", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
    else:
        ax2.set_visible(False)

    plt.suptitle(f"Training Curves — {run.run_name}", fontsize=14, y=1.02)
    plt.tight_layout()

    path = os.path.join(output_dir, "loss_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path
