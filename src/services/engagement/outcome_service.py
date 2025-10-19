"""
Outcome Service - Business Logic Layer
=======================================

Extracts outcome reporting logic from engagements.py route handlers.

This service handles:
- Parsing and normalizing outcome values
- Recording outcomes for calibration closure

Target Complexity: CC ≤ 8 per method
"""

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class OutcomeService:
    """
    Service for handling engagement outcome reporting.

    Responsibilities:
    - Normalize outcome values (yes/no/numeric)
    - Validate outcome ranges
    - Parse special cases (too_early, unknown)
    """

    def parse_outcome(self, outcome_str: str) -> Tuple[Optional[float], Optional[str]]:
        """
        Parse and normalize outcome value.

        Complexity: CC ≤ 6

        Args:
            outcome_str: The outcome string from user

        Returns:
            Tuple of (normalized_outcome, error_or_special_status)
            - If successful: (float_value, None)
            - If too_early: (None, "too_early")
            - If invalid: (None, "invalid")
        """
        v = (outcome_str or "").strip().lower()

        # Handle positive outcomes
        if v in ("yes", "y", "true", "success", "pass"):
            return 1.0, None

        # Handle negative outcomes
        if v in ("no", "n", "false", "fail", "failure"):
            return 0.0, None

        # Handle too_early / unknown
        if v in ("too early", "early", "unknown", "na"):
            return None, "too_early"

        # Handle numeric outcomes
        try:
            outcome = float(v)
            if not (0.0 <= outcome <= 1.0):
                return None, "invalid"
            return outcome, None
        except Exception:
            return None, "invalid"
