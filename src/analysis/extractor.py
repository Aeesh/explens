"""
Extracts structured facts from RunData.
These facts are what the LLM narrator receives AND what the
consistency checker uses to verify LLM claims.

Every fact has:
- value: the actual number
- description: human-readable string
- verifiable: True if the checker can verify LLM claims about this
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from src.connectors.wandb_connector import RunData


@dataclass
class Fact:
    key: str
    value: float | str | bool | int
    description: str
    verifiable: bool = True
    unit: str = ""


@dataclass
class ExperimentFacts:
    """
    All structured facts extracted from a run.
    """
    run: RunData

    # Training dynamics
    converged: bool = False
    best_epoch: int = 0
    best_val_loss: Optional[float] = None
    best_val_accuracy: Optional[float] = None
    final_train_loss: Optional[float] = None
    final_val_loss: Optional[float] = None

    # Overfitting
    overfit: bool = False
    overfit_severity: str = "none"  # "none" | "mild" | "moderate" | "severe"
    train_val_gap: Optional[float] = None

    # Convergence quality
    loss_trend: str = ""           # "decreasing" | "plateau" | "oscillating" | "diverging"
    val_loss_trend: str = ""
    converged_early: bool = False  # best epoch much earlier than total epochs

    # Learning rate behaviour
    lr_schedule_detected: bool = False
    peak_lr: Optional[float] = None
    final_lr: Optional[float] = None

    # Stability
    loss_spikes: int = 0
    oscillation_score: float = 0.0

    # All facts as a flat list for prompt injection and checking
    facts: list[Fact] = None

    def __post_init__(self):
        self.facts = self._extract_all_facts()

    def _extract_all_facts(self) -> list[Fact]:
        facts = []
        run = self.run

        # ── Best epoch ────────────────────────────────────────────────────
        if run.has_validation and "val_loss" in run.val_metrics:
            epoch_idx, best_val = run.best_epoch("val_loss", lower_is_better=True)
            self.best_epoch = epoch_idx
            self.best_val_loss = best_val
            facts.append(Fact(
                key="best_epoch",
                value=epoch_idx,
                description=f"Best validation loss achieved at step/epoch index {epoch_idx}",
            ))
            facts.append(Fact(
                key="best_val_loss",
                value=round(best_val, 4),
                description=f"Best validation loss: {best_val:.4f}",
                unit="loss"
            ))

            # Early convergence, check if it peak in the first half
            if run.n_epochs > 0:
                self.converged_early = epoch_idx < run.n_epochs * 0.6
                facts.append(Fact(
                    key="converged_early",
                    value=self.converged_early,
                    description=f"Model converged {'early' if self.converged_early else 'late'} "
                                f"(best at epoch {epoch_idx} of {run.n_epochs})",
                ))

        # ── Val accuracy ──────────────────────────────────────────────────
        val_acc_key = next(
            (m for m in run.val_metrics if "acc" in m.lower()), None
        )
        if val_acc_key:
            _, best_acc = run.best_epoch(val_acc_key, lower_is_better=False)
            self.best_val_accuracy = best_acc
            facts.append(Fact(
                key="best_val_accuracy",
                value=round(best_acc, 4),
                description=f"Best validation accuracy: {best_acc:.4f} ({best_acc*100:.1f}%)",
                unit="accuracy"
            ))

        # ── Final losses ──────────────────────────────────────────────────
        train_loss_key = next(
            (m for m in run.train_metrics if "loss" in m.lower()), None
        )
        if train_loss_key:
            series = run.get_metric_series(train_loss_key)
            self.final_train_loss = float(series.iloc[-1])
            facts.append(Fact(
                key="final_train_loss",
                value=round(self.final_train_loss, 4),
                description=f"Final training loss: {self.final_train_loss:.4f}",
                unit="loss"
            ))

        if "val_loss" in run.val_metrics:
            series = run.get_metric_series("val_loss")
            self.final_val_loss = float(series.iloc[-1])
            facts.append(Fact(
                key="final_val_loss",
                value=round(self.final_val_loss, 4),
                description=f"Final validation loss: {self.final_val_loss:.4f}",
                unit="loss"
            ))

        # ── Overfitting ───────────────────────────────────────────────────
        if self.final_train_loss and self.final_val_loss:
            gap = self.final_val_loss - self.final_train_loss
            self.train_val_gap = gap

            if gap < 0.05:
                self.overfit_severity = "none"
            elif gap < 0.15:
                self.overfit_severity = "mild"
                self.overfit = True
            elif gap < 0.30:
                self.overfit_severity = "moderate"
                self.overfit = True
            else:
                self.overfit_severity = "severe"
                self.overfit = True

            facts.append(Fact(
                key="train_val_gap",
                value=round(gap, 4),
                description=f"Train/val loss gap: {gap:.4f} ({self.overfit_severity} overfitting)",
            ))
            facts.append(Fact(
                key="overfit",
                value=self.overfit,
                description=f"Overfitting detected: {self.overfit} (severity: {self.overfit_severity})",
            ))

        # ── Loss trend ────────────────────────────────────────────────────
        if train_loss_key:
            series = run.get_metric_series(train_loss_key).values
            self.loss_trend = _classify_trend(series)
            facts.append(Fact(
                key="loss_trend",
                value=self.loss_trend,
                description=f"Training loss trend: {self.loss_trend}",
            ))

        if "val_loss" in run.val_metrics:
            series = run.get_metric_series("val_loss").values
            self.val_loss_trend = _classify_trend(series)
            facts.append(Fact(
                key="val_loss_trend",
                value=self.val_loss_trend,
                description=f"Validation loss trend: {self.val_loss_trend}",
            ))

        # ── Learning rate ─────────────────────────────────────────────────
        lr_key = next(
            (m for m in run.metric_names if "lr" in m.lower()
             or "learning_rate" in m.lower()), None
        )
        if lr_key:
            lr_series = run.get_metric_series(lr_key)
            self.lr_schedule_detected = lr_series.std() > 0
            self.peak_lr = float(lr_series.max())
            self.final_lr = float(lr_series.iloc[-1])
            facts.append(Fact(
                key="lr_schedule",
                value=self.lr_schedule_detected,
                description=f"LR schedule: {'yes' if self.lr_schedule_detected else 'no'}, "
                            f"peak={self.peak_lr:.2e}, final={self.final_lr:.2e}",
            ))

        # ── Loss spikes ───────────────────────────────────────────────────
        if train_loss_key:
            series = run.get_metric_series(train_loss_key).values
            diffs = np.diff(series)
            threshold = np.std(diffs) * 3
            self.loss_spikes = int(np.sum(diffs > threshold))
            if self.loss_spikes > 0:
                facts.append(Fact(
                    key="loss_spikes",
                    value=self.loss_spikes,
                    description=f"Training loss spikes detected: {self.loss_spikes}",
                ))

        return facts

    def to_prompt_context(self) -> str:
        """Structured context block for LLM prompt injection."""
        lines = [
            f"VERIFIED EXPERIMENT FACTS:\n",
            f"Run: {self.run.run_name}",
            f"Project: {self.run.project}",
            f"Total steps logged: {self.run.n_steps}",
            f"Epochs detected: {self.run.n_epochs}",
            f"Config: {self.run.config}\n",
            "Measured facts:"
        ]
        for fact in self.facts:
            lines.append(f"  - {fact.description}")
        return "\n".join(lines)


def _classify_trend(values: np.ndarray) -> str:
    """Classify a metric series as decreasing/plateau/oscillating/diverging."""
    if len(values) < 3:
        return "insufficient data"

    # Fit a linear trend
    x = np.arange(len(values))
    slope, _ = np.polyfit(x, values, 1)
    r2 = np.corrcoef(x, values)[0, 1] ** 2

    # Measure oscillation, std of differences
    diffs = np.diff(values)
    oscillation = np.std(diffs) / (np.mean(np.abs(values)) + 1e-8)

    if slope > 0.001 and r2 > 0.5:
        return "diverging"
    elif oscillation > 0.1:
        return "oscillating"
    elif slope < -0.001 and r2 > 0.3:
        return "decreasing"
    else:
        return "plateau"


def extract_facts(run: RunData) -> ExperimentFacts:
    """Main entry point to extract all facts from a run."""
    return ExperimentFacts(run=run)