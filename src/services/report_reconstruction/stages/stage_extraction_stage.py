"""
Stage Extraction Stage
======================

Extracts and normalizes cognitive stage outputs from raw cognitive states.

Responsibility:
- Build stages dictionary from cognitive_states
- Extract main stages (senior_advisor, devils_advocate, consultant_selection)
- Handle V6 nested structure variations
- Extract problem_structuring and socratic_questions
- Normalize field names between V5/V6

Complexity: CC<8 (Moderate extraction logic with some branching)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.services.report_reconstruction.reconstruction_state import ReconstructionState
from src.services.report_reconstruction.reconstruction_stage import (
    ReconstructionStage,
    ReconstructionError,
)

logger = logging.getLogger(__name__)


class StageExtractionStage(ReconstructionStage):
    """
    Stage 2: Stage Extraction

    Extracts and normalizes cognitive stage outputs from raw cognitive states.
    Handles both V5 and V6 data format variations.
    """

    @property
    def name(self) -> str:
        return "stage_extraction"

    @property
    def description(self) -> str:
        return "Extract and normalize cognitive stage outputs"

    def process(self, state: ReconstructionState) -> ReconstructionState:
        """
        Extract cognitive stages from raw cognitive states.

        Args:
            state: Current reconstruction state with cognitive_states populated

        Returns:
            Updated state with extracted stages
        """
        try:
            # Build stages dictionary from cognitive_states
            stages = self._build_stages_dict(state.cognitive_states)

            # Extract main stages with V6 nested structure handling
            senior_advisor = self._extract_stage_with_unwrap(stages, "senior_advisor")
            devils_advocate = self._extract_stage_with_unwrap(stages, "devils_advocate")
            consultant_selection = self._extract_stage_with_unwrap(
                stages, "consultant_selection"
            )

            # Extract Phase 1 stages (problem_structuring, socratic_questions)
            problem_structuring = self._extract_problem_structuring(stages)
            socratic_questions = self._extract_socratic_questions(stages)

            # Return updated state
            return state.with_stages(
                senior_advisor=senior_advisor,
                devils_advocate=devils_advocate,
                consultant_selection=consultant_selection,
                problem_structuring=problem_structuring,
                socratic_questions=socratic_questions,
            )

        except Exception as e:
            raise ReconstructionError(
                self.name,
                f"Failed to extract stages for trace_id={state.trace_id}",
                cause=e,
            )

    def _build_stages_dict(
        self, cognitive_states: list[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build stages dictionary from cognitive_states list.

        Args:
            cognitive_states: List of cognitive state outputs

        Returns:
            Dictionary mapping stage_name -> cognitive_output
        """
        stages: Dict[str, Dict[str, Any]] = {}
        for row in cognitive_states:
            stage_name = row.get("stage_name")
            cognitive_output = row.get("cognitive_output", {})
            if stage_name:
                stages[stage_name] = cognitive_output
        return stages

    def _extract_stage_with_unwrap(
        self, stages: Dict[str, Any], stage_name: str
    ) -> Dict[str, Any]:
        """
        Extract stage and handle V6 nested structure.

        V6 stores data as: stage_output.{stage_name}.{field}
        This unwraps that nested structure if present.

        Args:
            stages: Stages dictionary
            stage_name: Name of stage to extract

        Returns:
            Unwrapped stage data
        """
        stage_data = stages.get(stage_name, {}) or {}

        # V6: Check if data is nested under same key name
        if isinstance(stage_data.get(stage_name), dict):
            return stage_data.get(stage_name, {})

        return stage_data

    def _extract_problem_structuring(
        self, stages: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract MECE framework from problem_structuring stage.

        Tries multiple locations:
        1. Dedicated problem_structuring stage
        2. Legacy: any stage with mece_framework field

        Args:
            stages: Stages dictionary

        Returns:
            Problem structuring output or None
        """
        try:
            # Try dedicated stage first
            problem_stage = stages.get("problem_structuring", {}) or {}

            # Handle nested structure
            if isinstance(problem_stage.get("problem_structuring"), dict):
                problem_stage = problem_stage.get("problem_structuring", {})

            # Check for MECE framework
            mece = problem_stage.get("mece_framework")
            if isinstance(mece, list):
                return {"mece_framework": mece}

            # Legacy: check other stages for MECE framework
            for stage_data in stages.values():
                if isinstance(stage_data, dict):
                    mece = stage_data.get("mece_framework")
                    if isinstance(mece, list):
                        return {"mece_framework": mece}

        except Exception as e:
            logger.debug(f"Problem structuring extraction skipped: {e}")

        return None

    def _extract_socratic_questions(
        self, stages: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract Socratic questions and strategic insights.

        Args:
            stages: Stages dictionary

        Returns:
            Socratic output or None
        """
        try:
            soc_stage = stages.get("socratic_questions", {}) or {}

            # Handle nested structure
            if isinstance(soc_stage.get("socratic_questions"), dict):
                soc_stage = soc_stage.get("socratic_questions", {})

            # Extract known fields
            fields = [
                "key_strategic_questions",
                "clarified_problem_statement",
                "key_business_insights",
                "tosca_coverage",
                "missing_tosca_elements",
                "quality_score",
                "processing_time_ms",
            ]

            soc = {k: soc_stage.get(k) for k in fields if k in soc_stage}

            if soc:
                return soc

        except Exception as e:
            logger.debug(f"Socratic extraction skipped: {e}")

        return None
