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


class ExperimentNarrator:

    def __init__(self):
        self._llm = get_llm()

    def generate(self, facts: ExperimentFacts) -> NarratorResult:
        """Generate full report for a single run."""
        result = NarratorResult()
        result.facts = facts
        result.patterns = detect_patterns(facts)
        checker = ConsistencyChecker(facts)

        facts_ctx = facts.to_prompt_context()
        patterns_str = "\n".join(
            f"[{p.severity.upper()}] {p.title}: {p.description} — {p.recommendation}"
            for p in result.patterns
        ) or "No significant patterns detected."

        # Generate each section and check it
        sections = [
            ("overview", OVERVIEW_PROMPT.format(facts_context=facts_ctx)),
            ("training_dynamics", TRAINING_DYNAMICS_PROMPT.format(
                facts_context=facts_ctx)),
        ]

        if facts.run.has_validation:
            sections.append(("generalisation", GENERALISATION_PROMPT.format(
                facts_context=facts_ctx)))

        sections.extend([
            ("recommendations", RECOMMENDATIONS_PROMPT.format(
                facts_context=facts_ctx, patterns=patterns_str)),
            ("what_next", WHAT_NEXT_PROMPT.format(
                facts_context=facts_ctx, patterns=patterns_str)),
        ])

        for section_name, prompt in sections:
            text = self._generate_section(prompt)
            setattr(result, section_name, text)
            check = checker.check(text)
            result.consistency_checks[section_name] = check

            if not check.overall_consistent:
                logger.warning(
                    f"Section '{section_name}' has {check.n_failed} "
                    f"inconsistent claim(s):\n{check.summary()}"
                )

        return result
