"""
Stage Key Enumeration - V6 Canonical Keys
==========================================

Single source of truth for all pipeline stage context keys.

DESIGN PRINCIPLE: V6 executors use enum-based keys for type safety and consistency.
No legacy aliases (progressive_questions, socratic_results, etc.) are supported.

This eliminates the dual-key system that caused checkpoint resume bugs.
"""

from enum import Enum


class StageKey(str, Enum):
    """Canonical context keys for pipeline stages.

    These match the PipelineStage enum values for consistency:
    - PipelineStage.SOCRATIC_QUESTIONS.value == StageKey.SOCRATIC.value

    Usage:
        context[StageKey.SOCRATIC.value] = socratic_output.model_dump()
        data = context.get(StageKey.SOCRATIC.value)
    """

    # Core pipeline stages
    SOCRATIC = "socratic_questions"
    STRUCTURING = "problem_structuring"
    INTERACTION_SWEEP = "interaction_sweep"
    ORACLE = "hybrid_data_research"
    SELECTION = "consultant_selection"
    SYNERGY = "synergy_prompting"
    ANALYSIS = "parallel_analysis"
    DEVILS_ADVOCATE = "devils_advocate"
    SENIOR_ADVISOR = "senior_advisor"

    # Special context keys
    INITIAL_QUERY = "initial_query"
    TRACE_ID = "trace_id"
    STAGE_HISTORY = "stage_history"


# Legacy keys that are NO LONGER SUPPORTED
# These will raise validation errors if found in context
LEGACY_KEYS = {
    "progressive_questions",  # Old name for socratic_questions
    "socratic_results",       # Old alias for socratic_questions
    "problem_structure",      # Old name for problem_structuring
    "structuring_results",    # Old alias for problem_structuring
    "oracle_results",         # Old name for hybrid_data_research
    "selected_consultants",   # Nested field, not top-level key
}


def validate_no_legacy_keys(context: dict) -> None:
    """Fail-fast validation to prevent legacy keys from sneaking in.

    Args:
        context: Context dictionary to validate

    Raises:
        ValueError: If any legacy keys are present

    Note: Ignores metadata keys (starting with underscore) as they're not legacy data
    """
    # Filter out metadata keys (those starting with _) before validation
    non_metadata_keys = {k for k in context.keys() if not k.startswith('_')}
    present = LEGACY_KEYS & non_metadata_keys
    if present:
        raise ValueError(
            f"Legacy keys are no longer supported: {sorted(present)}. "
            f"Use V6 StageKey enum instead. "
            f"See src/core/stage_keys.py for canonical keys."
        )
