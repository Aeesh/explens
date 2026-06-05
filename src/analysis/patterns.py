"""
Detects higher-level training patterns from ExperimentFacts.
These are diagnostic patterns that inform the narrative and recommendations.
"""

from dataclasses import dataclass
from src.analysis.extractor import ExperimentFacts


@dataclass
class PatternMatch:
    pattern_id: str
    title: str
    description: str
    recommendation: str
    severity: str  # "info" | "warning" | "critical"


def detect_patterns(facts: ExperimentFacts) -> list[PatternMatch]:
    """
    Detect all matching patterns from experiment facts.
    Returns a list of PatternMatch objects, sorted by severity.
    """
    patterns = []

    # ── Healthy convergence ───────────────────────────────────────────────
    if (facts.loss_trend == "decreasing"
            and facts.overfit_severity in ("none", "mild")
            and not facts.converged_early):
        patterns.append(PatternMatch(
            pattern_id="healthy_convergence",
            title="Healthy Convergence",
            description="Training loss decreased steadily and validation "
                        "performance tracked training closely.",
            recommendation="Model is well-trained. Consider training for "
                           "additional epochs to see if further improvement "
                           "is possible.",
            severity="info"
        ))

    # ── Early stopping opportunity ────────────────────────────────────────
    if facts.converged_early and facts.run.n_epochs > 0:
        patterns.append(PatternMatch(
            pattern_id="early_stopping",
            title="Early Stopping Opportunity",
            description=f"Best validation performance was reached at epoch "
                        f"{facts.best_epoch} of {facts.run.n_epochs} — "
                        f"the last {facts.run.n_epochs - facts.best_epoch} "
                        f"epochs did not improve generalisation.",
            recommendation="Add early stopping with patience=2 to save "
                           "compute and avoid overfitting on future runs.",
            severity="warning"
        ))

    # ── Moderate overfitting ──────────────────────────────────────────────
    if facts.overfit_severity == "moderate":
        patterns.append(PatternMatch(
            pattern_id="moderate_overfit",
            title="Moderate Overfitting",
            description=f"Train/val loss gap is {facts.train_val_gap:.4f}. "
                        f"The model is learning training-specific patterns "
                        f"that don't generalise.",
            recommendation="Try increasing dropout, adding weight decay, "
                           "or reducing model complexity. Data augmentation "
                           "is another option if your dataset is small.",
            severity="warning"
        ))

    # ── Severe overfitting ────────────────────────────────────────────────
    if facts.overfit_severity == "severe":
        patterns.append(PatternMatch(
            pattern_id="severe_overfit",
            title="Severe Overfitting",
            description=f"Train/val loss gap is {facts.train_val_gap:.4f}. "
                        f"The model is memorising training data.",
            recommendation="Significantly increase regularisation (dropout "
                           "0.3+, weight decay 0.1+). Consider reducing "
                           "model size or collecting more training data.",
            severity="critical"
        ))

    # ── Loss oscillation ──────────────────────────────────────────────────
    if facts.loss_trend == "oscillating":
        patterns.append(PatternMatch(
            pattern_id="oscillating_loss",
            title="Unstable Training",
            description="Training loss oscillates rather than decreasing "
                        "smoothly, indicating instability.",
            recommendation="Reduce the learning rate by 2-5x. Check that "
                           "gradient clipping is enabled. Consider a "
                           "warmup schedule if not already using one.",
            severity="warning"
        ))

    # ── Loss divergence ───────────────────────────────────────────────────
    if facts.loss_trend == "diverging":
        patterns.append(PatternMatch(
            pattern_id="diverging_loss",
            title="Training Divergence",
            description="Training loss increased over time — training failed.",
            recommendation="Learning rate is almost certainly too high. "
                           "Reduce by 10x and retry. Check for data issues "
                           "(NaN values, incorrect labels).",
            severity="critical"
        ))

    # ── Loss spikes ───────────────────────────────────────────────────────
    if facts.loss_spikes >= 3:
        patterns.append(PatternMatch(
            pattern_id="loss_spikes",
            title="Gradient Instability",
            description=f"{facts.loss_spikes} sharp loss spikes detected "
                        f"during training.",
            recommendation="Enable or tighten gradient clipping "
                           "(max_norm=1.0). Consider a lower learning rate "
                           "or a more aggressive warmup.",
            severity="warning"
        ))

    # ── No learning rate schedule ─────────────────────────────────────────
    if not facts.lr_schedule_detected and facts.overfit:
        patterns.append(PatternMatch(
            pattern_id="no_lr_schedule",
            title="No Learning Rate Schedule",
            description="No learning rate decay detected. A fixed LR "
                        "combined with overfitting suggests the model "
                        "could benefit from LR annealing.",
            recommendation="Add a cosine annealing or linear decay schedule. "
                           "This often recovers 1-3% performance without "
                           "any other changes.",
            severity="info"
        ))

    # Sort: critical first, then warning, then info
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    return sorted(patterns, key=lambda p: severity_order.get(p.severity, 3))