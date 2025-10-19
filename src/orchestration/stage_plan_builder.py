"""
Stage Plan Builder - Factory for Pipeline Execution Plans
=========================================================

Builds ordered stage execution plans for the orchestration engine.

This module encapsulates the logic for defining the standard 8-stage
pipeline execution order and converting PipelineStage enums to StageSpec
flow contracts.
"""
from typing import List, Optional

from src.orchestration.flow_contracts import StageSpec, StageId
from src.core.checkpoint_models import PipelineStage


class StagePlanBuilder:
    """Factory for building stage execution plans."""

    # Default 8-stage pipeline order
    DEFAULT_STAGES = [
        PipelineStage.SOCRATIC_QUESTIONS,
        PipelineStage.PROBLEM_STRUCTURING,
        PipelineStage.INTERACTION_SWEEP,
        PipelineStage.CONSULTANT_SELECTION,
        PipelineStage.SYNERGY_PROMPTING,
        PipelineStage.PARALLEL_ANALYSIS,
        PipelineStage.DEVILS_ADVOCATE,
        PipelineStage.SENIOR_ADVISOR,
    ]

    @staticmethod
    def build_default_plan() -> List[StageSpec]:
        """
        Build the default 8-stage pipeline execution plan.

        Returns:
            List of StageSpec objects in execution order with dependencies.
            Each stage depends on the previous stage completing.
        """
        plan: List[StageSpec] = []
        prev_id: Optional[StageId] = None

        for stage_enum in StagePlanBuilder.DEFAULT_STAGES:
            stage_id = StageId(stage_enum.value)

            spec = StageSpec(
                id=stage_id,
                name=getattr(stage_enum, "display_name", stage_enum.value),
                executor_key=f"exec::{stage_enum.value}",
                inputs={},
                requires=[] if prev_id is None else [prev_id],
                retry_policy={"max_retries": 0},  # No retries by default
            )

            plan.append(spec)
            prev_id = stage_id

        return plan

    @staticmethod
    def build_custom_plan(stages: List[PipelineStage]) -> List[StageSpec]:
        """
        Build a custom pipeline plan with specified stages.

        Args:
            stages: List of PipelineStage enums in desired execution order

        Returns:
            List of StageSpec objects with linear dependencies
        """
        plan: List[StageSpec] = []
        prev_id: Optional[StageId] = None

        for stage_enum in stages:
            stage_id = StageId(stage_enum.value)

            spec = StageSpec(
                id=stage_id,
                name=getattr(stage_enum, "display_name", stage_enum.value),
                executor_key=f"exec::{stage_enum.value}",
                inputs={},
                requires=[] if prev_id is None else [prev_id],
                retry_policy={"max_retries": 0},
            )

            plan.append(spec)
            prev_id = stage_id

        return plan
