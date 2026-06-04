"""
Local connector — reads training logs from CSV or JSON.
Useful in the abscence of WandB.

Expected CSV format:
  step, epoch, train_loss, val_loss, val_accuracy, learning_rate, ...

Expected JSON format:
  {"config": {...}, "history": [...], "summary": {...}}
"""

import json
import os
import pandas as pd
from .wandb_connector import RunData


def load_from_csv(
    csv_path: str,
    run_name: str = "local_run",
    config: dict = None,
) -> RunData:
    """Load a run from a CSV training log."""
    df = pd.read_csv(csv_path)

    metric_names = [c for c in df.columns if not c.startswith("_")]
    n_epochs = 0
    if "epoch" in df.columns:
        n_epochs = int(df["epoch"].max()) + 1

    summary = {}
    for col in metric_names:
        if df[col].notna().any():
            summary[col] = float(df[col].dropna().iloc[-1])

    return RunData(
        run_id=os.path.basename(csv_path),
        run_name=run_name,
        project="local",
        entity="local",
        url="",
        config=config or {},
        summary=summary,
        history=df,
        metric_names=metric_names,
        n_steps=len(df),
        n_epochs=n_epochs,
        state="finished",
    )


def load_from_json(json_path: str) -> RunData:
    """Load a run from a JSON training log."""
    with open(json_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data.get("history", []))
    config = data.get("config", {})
    summary = data.get("summary", {})
    metric_names = [c for c in df.columns if not c.startswith("_")]

    n_epochs = 0
    if "epoch" in df.columns:
        n_epochs = int(df["epoch"].max()) + 1

    return RunData(
        run_id=data.get("run_id", "local"),
        run_name=data.get("run_name", "local_run"),
        project=data.get("project", "local"),
        entity=data.get("entity", "local"),
        url=data.get("url", ""),
        config=config,
        summary=summary,
        history=df,
        metric_names=metric_names,
        n_steps=len(df),
        n_epochs=n_epochs,
        state=data.get("state", "finished"),
    )