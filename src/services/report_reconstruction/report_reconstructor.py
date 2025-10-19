"""
Report Reconstructor - Stage Orchestrator
=========================================

Orchestrates the execution of reconstruction stages in sequence.

Design:
- Sequential stage execution
- Immutable state flow
- Clean error handling
- Low cyclomatic complexity (CC<10)

Usage:
    reconstructor = ReportReconstructor(db_service, sanitizer)
    bundle = reconstructor.reconstruct(trace_id)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.services.report_reconstruction.reconstruction_state import ReconstructionState
from src.services.report_reconstruction.reconstruction_stage import (
    ReconstructionStage,
    ReconstructionError,
)

logger = logging.getLogger(__name__)


class ReportReconstructor:
    """
    Orchestrator for report reconstruction stages.

    Executes stages sequentially, passing immutable state through the pipeline.
    Each stage transforms the state and returns an updated version.
    """

    def __init__(
        self,
        stages: List[ReconstructionStage],
        enable_logging: bool = True,
    ):
        """
        Initialize report reconstructor.

        Args:
            stages: List of reconstruction stages to execute
            enable_logging: Whether to log stage execution
        """
        self.stages = stages
        self.enable_logging = enable_logging

    def reconstruct(self, trace_id: str) -> Dict[str, Any]:
        """
        Reconstruct report bundle from trace ID.

        Args:
            trace_id: Unique trace identifier

        Returns:
            Complete report bundle

        Raises:
            ReconstructionError: If any stage fails
        """
        # Initialize state
        state = ReconstructionState(trace_id=trace_id)

        if self.enable_logging:
            logger.info(f"🔄 Starting reconstruction for trace_id={trace_id}")

        # Execute stages sequentially
        for stage in self.stages:
            try:
                if self.enable_logging:
                    logger.debug(f"  ▶️  Executing stage: {stage.name}")

                state = stage.process(state)

                if self.enable_logging:
                    logger.debug(f"  ✅ Completed stage: {stage.name}")

            except Exception as e:
                error_msg = f"Stage '{stage.name}' failed: {str(e)}"
                logger.error(f"❌ {error_msg}")
                raise ReconstructionError(stage.name, error_msg, cause=e)

        if self.enable_logging:
            logger.info(f"✅ Reconstruction complete for trace_id={trace_id}")

        # Extract final bundle
        return state.to_bundle()

    def add_stage(self, stage: ReconstructionStage) -> None:
        """
        Add a stage to the reconstruction pipeline.

        Args:
            stage: Stage to add
        """
        self.stages.append(stage)

    def __repr__(self) -> str:
        """String representation for debugging."""
        stage_names = [s.name for s in self.stages]
        return f"ReportReconstructor(stages={stage_names})"
