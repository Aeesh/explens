"""
Generates narrative sections from facts, then checks them for consistency.
"""

import json
import logging
from src.analysis.extractor import ExperimentFacts
from src.analysis.patterns import PatternMatch, detect_patterns
from src.narrator.prompts import (
    NARRATOR_SYSTEM, OVERVIEW_PROMPT, TRAINING_DYNAMICS_PROMPT,
    GENERALISATION_PROMPT, RECOMMENDATIONS_PROMPT, WHAT_NEXT_PROMPT
)
from src.narrator.checker import ConsistencyChecker, ConsistencyReport
from src.llm import get_llm

logger = logging.getLogger(__name__)


class NarratorResult:
    """Full generated report with consistency checks."""
    def __init__(self):
        self.overview: str = ""
        self.training_dynamics: str = ""
        self.generalisation: str = ""
        self.recommendations: str = ""
        self.what_next: str = ""
        self.consistency_checks: dict[str, ConsistencyReport] = {}
        self.patterns: list[PatternMatch] = []
        self.facts: ExperimentFacts = None

    def overall_consistent(self) -> bool:
        return all(r.overall_consistent for r in self.consistency_checks.values())

    def failed_checks(self) -> list:
        failed = []
        for section, report in self.consistency_checks.items():
            for check in report.checks:
                if not check.verified:
                    failed.append((section, check))
        return failed
