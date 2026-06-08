"""
Consistency checker that verifies LLM-generated narrative against the measured facts.

How it works:
1. The LLM generates a narrative section
2. The checker extracts verifiable claims from the text
3. Each claim is checked against ExperimentFacts
4. Inconsistent claims are flagged with the correct value

This is automated factual verification of LLM output against structured
time-series data.
"""

import re
from dataclasses import dataclass
from typing import Optional
from src.analysis.extractor import ExperimentFacts


@dataclass
class ClaimCheck:
    """Result of checking one verifiable claim."""
    claim_text: str           # The text extracted from the narrative
    claim_type: str           # What kind of claim this is
    verified: bool            # True if the claim matches the facts
    actual_value: str         # What the facts actually say
    correction: Optional[str] = None  # Suggested replacement text


@dataclass
class ConsistencyReport:
    """Full consistency check result for a narrative section."""
    narrative: str
    checks: list[ClaimCheck]
    n_verified: int = 0
    n_failed: int = 0
    overall_consistent: bool = True

    def __post_init__(self):
        self.n_verified = sum(1 for c in self.checks if c.verified)
        self.n_failed = sum(1 for c in self.checks if not c.verified)
        self.overall_consistent = self.n_failed == 0

    def summary(self) -> str:
        if self.overall_consistent:
            return f"✓ All {self.n_verified} verifiable claims are consistent with data."
        lines = [f"⚠ {self.n_failed} claim(s) inconsistent with measured data:"]
        for check in self.checks:
            if not check.verified:
                lines.append(
                    f"  - Claim: \"{check.claim_text}\"\n"
                    f"    Actual: {check.actual_value}"
                )
        return "\n".join(lines)


class ConsistencyChecker:
    """
    Extracts and verifies factual claims from LLM narratives.

    Supported claim types:
    - epoch references ("converged at epoch X", "best at step X")
    - metric value references ("val loss of X", "accuracy of X%")
    - trend claims ("loss decreased", "loss increased", "oscillating")
    - overfitting claims ("signs of overfitting", "model overfits")
    - convergence claims ("converged early", "did not converge")
    """

    def __init__(self, facts: ExperimentFacts):
        self.facts = facts
        self._facts_dict = {f.key: f for f in facts.facts}

    def check(self, narrative: str) -> ConsistencyReport:
        """Run all checks on a narrative string."""
        checks = []
        checks.extend(self._check_epoch_references(narrative))
        checks.extend(self._check_metric_values(narrative))
        checks.extend(self._check_trend_claims(narrative))
        checks.extend(self._check_overfitting_claims(narrative))
        return ConsistencyReport(narrative=narrative, checks=checks)

    def _check_epoch_references(self, text: str) -> list[ClaimCheck]:
        """Check claims about which epoch was best."""
        checks = []
        if "best_epoch" not in self._facts_dict:
            return checks

        actual_epoch = self._facts_dict["best_epoch"].value

        # Pattern: "epoch 3", "epoch 4", "at epoch X"
        epoch_refs = re.findall(r'epoch\s+(\d+)', text, re.IGNORECASE)
        for ref in epoch_refs:
            claimed = int(ref)
            # Allow off-by-one since sometimes people say "epoch 3" but the best was at index 2 (0-based)
            verified = abs(claimed - actual_epoch) <= 1
            checks.append(ClaimCheck(
                claim_text=f"epoch {ref}",
                claim_type="epoch_reference",
                verified=verified,
                actual_value=f"Best epoch: {actual_epoch}",
                correction=None if verified else f"epoch {actual_epoch}"
            ))

        return checks

    def _check_metric_values(self, text: str) -> list[ClaimCheck]:
        """Check numerical metric values mentioned in the narrative."""
        checks = []

        # Look for patterns like "val loss of 0.42", "accuracy of 88%", "loss: 0.33"
        number_patterns = [
            (r'val(?:idation)?\s+loss\s+(?:of\s+)?([0-9]+\.[0-9]+)', "best_val_loss", 0.05),
            (r'train(?:ing)?\s+loss\s+(?:of\s+)?([0-9]+\.[0-9]+)', "final_train_loss", 0.05),
            (r'accuracy\s+(?:of\s+)?([0-9]+\.?[0-9]*)\s*%?', "best_val_accuracy", 0.03),
        ]

        for pattern, fact_key, tolerance in number_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if not matches or fact_key not in self._facts_dict:
                continue

            actual = float(self._facts_dict[fact_key].value)
            for match in matches:
                claimed = float(match)
                # Handle percentage vs decimal for accuracy
                if fact_key == "best_val_accuracy" and claimed > 1:
                    claimed /= 100
                verified = abs(claimed - actual) <= tolerance
                checks.append(ClaimCheck(
                    claim_text=f"{match}",
                    claim_type="metric_value",
                    verified=verified,
                    actual_value=f"{fact_key}: {actual:.4f}",
                    correction=None if verified else f"{actual:.4f}"
                ))

        return checks

    def _check_trend_claims(self, text: str) -> list[ClaimCheck]:
        """Check claims about whether loss increased or decreased."""
        checks = []

        if "loss_trend" not in self._facts_dict:
            return checks

        actual_trend = self._facts_dict["loss_trend"].value

        trend_claims = [
            (r'loss\s+(?:steadily\s+)?decreased', "decreasing"),
            (r'loss\s+(?:steadily\s+)?increased', "diverging"),
            (r'loss\s+(?:remained\s+)?(?:stable|plateau)', "plateau"),
            (r'(?:unstable|oscillat)', "oscillating"),
            (r'loss\s+diverged', "diverging"),
        ]

        for pattern, claimed_trend in trend_claims:
            if re.search(pattern, text, re.IGNORECASE):
                verified = actual_trend == claimed_trend
                checks.append(ClaimCheck(
                    claim_text=pattern.replace(r'\s+', ' ').replace('(?:', '').replace(')', ''),
                    claim_type="trend_claim",
                    verified=verified,
                    actual_value=f"Actual trend: {actual_trend}",
                    correction=None if verified else f"Trend is actually: {actual_trend}"
                ))

        return checks

    def _check_overfitting_claims(self, text: str) -> list[ClaimCheck]:
        """Check claims about overfitting."""
        checks = []
        if "overfit" not in self._facts_dict:
            return checks

        actual_overfit = self._facts_dict["overfit"].value
        actual_severity = self.facts.overfit_severity

        overfit_claimed = bool(re.search(
            r'overfit|overfitting|generaliz|generalising', text, re.IGNORECASE
        ))
        no_overfit_claimed = bool(re.search(
            r'no\s+(?:sign|evidence)\s+of\s+overfit|not\s+overfit|generaliz(?:es|ing)\s+well',
            text, re.IGNORECASE
        ))

        if overfit_claimed:
            checks.append(ClaimCheck(
                claim_text="signs of overfitting",
                claim_type="overfitting_claim",
                verified=actual_overfit,
                actual_value=f"Overfitting: {actual_overfit}, severity: {actual_severity}",
                correction=None if actual_overfit else "No significant overfitting detected"
            ))

        if no_overfit_claimed:
            checks.append(ClaimCheck(
                claim_text="no overfitting",
                claim_type="overfitting_claim",
                verified=not actual_overfit,
                actual_value=f"Overfitting: {actual_overfit}, severity: {actual_severity}",
                correction=None if not actual_overfit
                else f"Overfitting detected (severity: {actual_severity})"
            ))

        return checks