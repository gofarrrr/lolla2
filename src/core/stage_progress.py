"""
Stage Progress Utilities
========================

Single source of truth for pipeline stage ordering and UI stage numbering.

Rationale:
- Avoid hard-coded stage numbers scattered across the codebase
- Keep DB constraints, orchestrator status, and frontend UI in sync
"""

from __future__ import annotations

from typing import List, Tuple
from .checkpoint_models import PipelineStage


def ordered_stages(include_arbitration: bool = True) -> List[PipelineStage]:
    """Return the canonical ordered list of pipeline stages.

    By default includes Arbitration/Capture as the final stage.
    """
    stages = [
        PipelineStage.SOCRATIC_QUESTIONS,
        PipelineStage.PROBLEM_STRUCTURING,
        PipelineStage.INTERACTION_SWEEP,
        PipelineStage.HYBRID_DATA_RESEARCH,
        PipelineStage.CONSULTANT_SELECTION,
        PipelineStage.SYNERGY_PROMPTING,
        PipelineStage.PARALLEL_ANALYSIS,
        PipelineStage.DEVILS_ADVOCATE,
        PipelineStage.SENIOR_ADVISOR,
    ]
    if include_arbitration:
        stages.append(PipelineStage.ARBITRATION_CAPTURE)
    return stages


def total_stages_for_ui() -> int:
    """Total stages for UI progression (excludes COMPLETED).

    We include Arbitration/Capture so that Senior Advisor is not treated as the
    terminal stage in the number line; this keeps numbering stable at 10 when
    Arbitration is enabled.
    """
    return len(ordered_stages(include_arbitration=True))


def stage_number(stage: PipelineStage) -> int:
    """Return 1-based stage number for a given PipelineStage.

    Falls back to 1 if the stage is not in the ordered list (should not happen).
    """
    stages = ordered_stages(include_arbitration=True)
    try:
        return stages.index(stage) + 1
    except ValueError:
        return 1


def stage_progress_percent(stage: PipelineStage) -> float:
    """Compute a coarse progress percentage for display.

    Maps linearly from stage number to [0, 100], capped at 100.
    """
    n = stage_number(stage)
    total = total_stages_for_ui()
    pct = (n / max(1, total)) * 100.0
    return min(100.0, round(pct, 1))

