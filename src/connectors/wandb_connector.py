"""
WandB connector — pulls run data into a clean RunData object.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import wandb


@dataclass
class RunData:
    """
    Normalised representation of a single training run.
    """
    run_id: str
    run_name: str
    project: str
    entity: str
    url: str

    # Config — values set before training
    config: dict

    # Summary — final values at end of training
    summary: dict

    # History — metrics at each step/epoch
    # DataFrame with columns: step, epoch (if logged), + all metrics
    history: pd.DataFrame

    # Computed fields filled by from_wandb()
    metric_names: list[str] = field(default_factory=list)
    n_steps: int = 0
    n_epochs: int = 0
    duration_seconds: Optional[float] = None
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    state: str = ""  # "finished" | "crashed" | "running"

    @property
    def has_validation(self) -> bool:
        return any("val" in m.lower() for m in self.metric_names)

    @property
    def train_metrics(self) -> list[str]:
        return [m for m in self.metric_names
                if "val" not in m.lower() and not m.startswith("_")]

    @property
    def val_metrics(self) -> list[str]:
        return [m for m in self.metric_names if "val" in m.lower()]

    def get_metric_series(self, metric: str) -> pd.Series:
        """Return the time series for a specific metric."""
        if metric not in self.history.columns:
            raise KeyError(f"Metric '{metric}' not found. Available: {self.metric_names}")
        return self.history[metric].dropna()

    def best_epoch(self, metric: str, lower_is_better: bool = True) -> tuple[int, float]:
        """Return (epoch_index, value) for the best value of a metric."""
        series = self.get_metric_series(metric)
        if lower_is_better:
            idx = series.idxmin()
        else:
            idx = series.idxmax()
        return int(idx), float(series[idx])

    def to_summary_dict(self) -> dict:
        """Compact dict for prompt injection."""
        return {
            "run_name": self.run_name,
            "project": self.project,
            "config": self.config,
            "n_epochs": self.n_epochs,
            "n_steps": self.n_steps,
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "final_values": {
                m: float(self.summary.get(m, float("nan")))
                for m in self.metric_names
                if not m.startswith("_")
            },
            "state": self.state,
        }


def load_run(run_path: str) -> RunData:
    """
    Load a single WandB run by path.

    run_path format: "entity/project/run_id"
    or just "project/run_id" if entity is set in WANDB_ENTITY env var.
    """
    api = wandb.Api(api_key=os.environ.get("WANDB_API_KEY"))
    run = api.run(run_path)

    # Pull full history of all logged metrics at every step
    history_df = run.history(pandas=True)

    # Clean up column names
    metric_names = [
        col for col in history_df.columns
        if not col.startswith("_")
    ]

    # Detect epoch count by looking for "epoch" column or infering from steps
    n_epochs = 0
    if "epoch" in history_df.columns:
        n_epochs = int(history_df["epoch"].max()) + 1
    elif "val_loss" in history_df.columns:
        # Each val_loss entry typically is one epoch
        n_epochs = history_df["val_loss"].dropna().shape[0]

    return RunData(
        run_id=run.id,
        run_name=run.name,
        project=run.project,
        entity=run.entity,
        url=run.url,
        config=dict(run.config),
        summary=dict(run.summary),
        history=history_df,
        metric_names=metric_names,
        n_steps=len(history_df),
        n_epochs=n_epochs,
        duration_seconds=run.summary.get("_runtime"),
        tags=list(run.tags),
        notes=run.notes or "",
        state=run.state,
    )


def load_project_runs(project_path: str, limit: int = 20) -> list[RunData]:
    """
    Load multiple runs from a project for comparison.

    project_path format: "entity/project"
    """
    api = wandb.Api(api_key=os.environ.get("WANDB_API_KEY"))
    runs = api.runs(project_path, order="-created_at")

    result = []
    for run in list(runs)[:limit]:
        try:
            result.append(load_run(f"{run.entity}/{run.project}/{run.id}"))
        except Exception as e:
            print(f"Warning: could not load run {run.id}: {e}")

    return result