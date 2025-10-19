"""
Prompt Quality Validator - Phase 3 Optimization

Validate prompt and output quality using research-validated metrics.

Based on academic research on LLM evaluation and quality benchmarks.

Part of Research-Grounded Improvement Plan Phase 3.
"""

import re
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class PromptQualityValidator:
    """
    Validate prompt and output quality using research-validated metrics.

    Based on academic research on LLM evaluation and prompt engineering best practices.
    """

    # Generic boilerplate phrases to detect (from research)
    GENERIC_PHRASES = [
        "comprehensive analysis",
        "review the report",
        "phase 1: foundation",
        "monitor success metrics",
        "implement best practices",
        "further investigation",
        "additional research needed",
        "consult with experts",
        "consider all factors",
        "evaluate the situation",
    ]

    def __init__(self):
        self._validation_count = 0
        self._pass_count = 0
        self._fail_count = 0

    def validate_consultant_output(
        self, output: str, assigned_dimensions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Check if consultant output meets research-based quality standards.

        Quality checks:
        1. Length check: Minimum depth requirement (800 words)
        2. Quantitative check: Specific metrics (5+ numbers with units)
        3. Dimension focus: Mentions assigned dimensions
        4. No generic boilerplate: Avoids generic phrases
        5. Structured format: Has clear markdown structure

        Args:
            output: The consultant analysis output
            assigned_dimensions: List of assigned N-way dimensions

        Returns:
            Dict with validation results, score, and identified issues
        """
        self._validation_count += 1

        quality_checks = {
            "length_check": len(output) >= 800,  # Minimum depth requirement
            "quantitative_check": self._count_numbers(output) >= 5,  # Specific metrics
            "dimension_focus": self._check_dimension_focus(
                output, assigned_dimensions or []
            )
            if assigned_dimensions
            else True,
            "no_generic_boilerplate": not self._contains_generic_phrases(output),
            "structured_format": self._has_clear_structure(output),
        }

        overall_score = sum(quality_checks.values()) / len(quality_checks)
        passed = overall_score >= 0.8

        if passed:
            self._pass_count += 1
        else:
            self._fail_count += 1

        issues = [k for k, v in quality_checks.items() if not v]

        result = {
            "passed": passed,
            "score": round(overall_score, 2),
            "checks": quality_checks,
            "issues": issues,
        }

        # Log validation results
        if not passed:
            logger.warning(
                f"⚠️ Quality Validation FAILED: score={overall_score:.2f}, issues={issues}"
            )
        else:
            logger.info(f"✅ Quality Validation PASSED: score={overall_score:.2f}")

        return result

    def _count_numbers(self, text: str) -> int:
        """
        Count quantitative metrics in text.

        Matches patterns like: "€5M", "20%", "3 months", "$100K", etc.
        """
        # Match numbers with units or currency symbols
        patterns = [
            r"\d+[%€$£¥]",  # Numbers with percent or currency symbols
            r"[€$£¥]\d+",  # Currency symbols before numbers
            r"\d+\s+(?:months?|years?|percent|million|billion|days?|weeks?)",  # Numbers with time/scale units
            r"\d+\.\d+\s*(?:million|billion|thousand|%)",  # Decimal numbers with units
        ]

        count = 0
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            count += len(matches)

        return count

    def _check_dimension_focus(self, text: str, dimensions: List[str]) -> bool:
        """
        Check if output focuses on assigned dimensions.

        Returns True if at least 50% of dimensions are mentioned.
        """
        if not dimensions:
            return True

        text_lower = text.lower()
        dimension_mentions = sum(
            1 for dim in dimensions if dim.lower() in text_lower
        )
        required_mentions = len(dimensions) * 0.5

        return dimension_mentions >= required_mentions

    def _contains_generic_phrases(self, text: str) -> bool:
        """
        Detect generic boilerplate phrases.

        Returns True if generic phrases are found (bad).
        """
        text_lower = text.lower()
        return any(phrase.lower() in text_lower for phrase in self.GENERIC_PHRASES)

    def _has_clear_structure(self, text: str) -> bool:
        """
        Check if output has clear markdown structure.

        Looks for:
        - Headings (# or ##)
        - Lists (- or 1.)
        - Multiple paragraphs

        Returns True if structured formatting is present.
        """
        # Check for markdown headings
        has_headings = bool(re.search(r"^#+\s+\w+", text, re.MULTILINE))

        # Check for lists
        has_lists = bool(re.search(r"^[-*]\s+\w+|^\d+\.\s+\w+", text, re.MULTILINE))

        # Check for multiple paragraphs (at least 3)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        has_paragraphs = len(paragraphs) >= 3

        # Must have at least 2 of 3 structure indicators
        structure_score = sum([has_headings, has_lists, has_paragraphs])
        return structure_score >= 2

    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        total = self._validation_count
        pass_rate = (self._pass_count / total * 100) if total > 0 else 0

        return {
            "total_validations": total,
            "passed": self._pass_count,
            "failed": self._fail_count,
            "pass_rate_pct": round(pass_rate, 1),
        }


# Singleton instance for global access
_prompt_quality_validator = None


def get_prompt_quality_validator() -> PromptQualityValidator:
    """Get the global prompt quality validator instance"""
    global _prompt_quality_validator
    if _prompt_quality_validator is None:
        _prompt_quality_validator = PromptQualityValidator()
    return _prompt_quality_validator
