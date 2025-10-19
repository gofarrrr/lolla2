"""Checkpoint management utilities extracted from the stateful orchestrator."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Optional, List
from uuid import UUID

from src.core.checkpoint_models import PipelineStage, StateCheckpoint
from src.core.exceptions import CheckpointError, RevisionError

logger = logging.getLogger(__name__)


class CheckpointService:
    """Encapsulates checkpoint and revision operations for pipeline executions."""

    def __init__(
        self,
        checkpoint_repo,
        revision_service,
    ) -> None:
        self.checkpoint_repo = checkpoint_repo
        self.revision_service = revision_service

    async def save_checkpoint(
        self,
        *,
        trace_id: UUID,
        stage_completed: PipelineStage,
        stage_output: Dict[str, any],
        user_id: Optional[UUID] = None,
        session_id: Optional[UUID] = None,
    ) -> StateCheckpoint:
        """Persist a lightweight checkpoint after a stage completes."""
        try:
            logger.info(f"🔍 DEBUG: CheckpointService.save_checkpoint called, trace_id={trace_id}, stage={stage_completed}")

            if isinstance(stage_completed, str):
                stage_completed = PipelineStage(stage_completed)

            logger.info(f"🔍 DEBUG: About to call checkpoint_repo.create_and_save_checkpoint")
            result = await self.checkpoint_repo.create_and_save_checkpoint(
                trace_id=trace_id,
                stage_completed=stage_completed,
                stage_output=stage_output,
                user_id=user_id,
                session_id=session_id,
            )
            logger.info(f"🔍 DEBUG: checkpoint_repo.create_and_save_checkpoint returned: {result.checkpoint_id if result else 'None'}")
            return result
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to save checkpoint: %s", exc)
            stage_value = (
                stage_completed.value
                if hasattr(stage_completed, "value")
                else str(stage_completed)
            )
            raise CheckpointError(
                f"Failed to save checkpoint for stage {stage_value}: {exc}"
            ) from exc

    async def resume_from_checkpoint(
        self,
        checkpoint_id: UUID,
    ) -> Dict[str, any]:
        """Restore pipeline state from a checkpoint identifier."""
        try:
            return await self.checkpoint_repo.resume_from_checkpoint(checkpoint_id)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to resume from checkpoint %s: %s", checkpoint_id, exc)
            raise CheckpointError(
                f"Failed to resume from checkpoint {checkpoint_id}: {exc}"
            ) from exc

    async def count_checkpoints(self, trace_id: UUID) -> int:
        """Return the number of checkpoints stored for a trace."""
        return await self.checkpoint_repo.count_checkpoints_for_trace(trace_id)

    # Convenience accessors used by APIs (centralize access via service)
    async def load_checkpoint(self, checkpoint_id: UUID) -> Optional[StateCheckpoint]:
        try:
            return await self.checkpoint_repo.load_checkpoint(checkpoint_id)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to load checkpoint %s: %s", checkpoint_id, exc)
            raise CheckpointError(f"Failed to load checkpoint {checkpoint_id}: {exc}") from exc

    async def load_checkpoints_for_trace(self, trace_id: UUID) -> List[StateCheckpoint]:
        try:
            return await self.checkpoint_repo.load_checkpoints_for_trace(trace_id)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to list checkpoints for %s: %s", trace_id, exc)
            raise CheckpointError(f"Failed to list checkpoints for {trace_id}: {exc}") from exc

    async def create_revision_branch(
        self,
        *,
        parent_trace_id: UUID,
        checkpoint_id: UUID,
        revision_data: Dict[str, any],
        revision_rationale: Optional[str] = None,
        user_id: Optional[UUID] = None,
        session_id: Optional[UUID] = None,
    ) -> UUID:
        """Fork a new analysis branch from an existing checkpoint."""
        try:
            child_trace_id = await self.revision_service.create_revision_branch(
                parent_trace_id=parent_trace_id,
                checkpoint_id=checkpoint_id,
                revision_data=revision_data,
                revision_rationale=revision_rationale,
                user_id=user_id,
                session_id=session_id,
            )
            logger.info(
                "Revision branch created: %s -> %s",
                parent_trace_id,
                child_trace_id,
            )
            return child_trace_id
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to create revision branch: %s", exc)
            raise RevisionError(f"Failed to create revision branch: {exc}") from exc

    async def check_revision_requested(self, trace_id: UUID) -> bool:
        """Placeholder hook for future revision-request checks."""
        # TODO(ar-04): integrate with revision queue / signal store
        return False


__all__ = ["CheckpointService"]
