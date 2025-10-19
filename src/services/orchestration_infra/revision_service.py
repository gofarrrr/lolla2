# src/services/orchestration_infra/revision_service.py
from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from src.core.pipeline_contracts import IRevisionService, ICheckpointRepository
from src.core.checkpoint_models import AnalysisRevision
from src.core.exceptions import RevisionError
from src.core.unified_context_stream import ContextEventType

logger = logging.getLogger(__name__)


class V1RevisionService(IRevisionService):
    """Service responsible for creating immutable analysis branches (revisions).

    Note: Background execution of the revised pipeline is intentionally not handled
    here to keep responsibilities narrow. The orchestrator (or a scheduler) can
    trigger execution after branch creation if needed.
    """

    def __init__(self, checkpoint_repo: ICheckpointRepository, context_stream: Any) -> None:
        self._repo = checkpoint_repo
        self._context_stream = context_stream

    async def create_revision_branch(
        self,
        parent_trace_id: UUID,
        checkpoint_id: UUID,
        revision_data: Dict[str, Any],
        revision_rationale: Optional[str] = None,
        user_id: Optional[UUID] = None,
        session_id: Optional[UUID] = None,
    ) -> UUID:
        try:
            # Create new trace for the child analysis
            child_trace_id = uuid4()
            logger.info(
                f"🌿 Creating analysis branch (service): {parent_trace_id} -> {child_trace_id}"
            )

            # Load source checkpoint
            source_checkpoint = await self._repo.load_checkpoint(checkpoint_id)
            if not source_checkpoint:
                raise RevisionError(f"Source checkpoint not found: {checkpoint_id}")

            # Create and persist revision record
            revision = AnalysisRevision(
                parent_trace_id=parent_trace_id,
                child_trace_id=child_trace_id,
                source_checkpoint_id=checkpoint_id,
                restart_from_stage=source_checkpoint.stage_completed,
                revision_data=revision_data,
                revision_rationale=revision_rationale,
                user_id=user_id,
                session_id=session_id,
            )
            await self._repo.save_revision(revision)

            # Mark processing started (or 'queued') and persist
            revision.mark_processing_started()
            await self._repo.update_revision(revision)

            # Log creation event for parity
            self._context_stream.add_event(
                ContextEventType.ENGAGEMENT_STARTED,
                {
                    "revision_branch_created": True,
                    "parent_trace_id": str(parent_trace_id),
                    "child_trace_id": str(child_trace_id),
                    "source_checkpoint_id": str(checkpoint_id),
                    "restart_from_stage": source_checkpoint.stage_completed.value,
                },
            )

            return child_trace_id
        except Exception as e:
            logger.error(f"❌ Revision service failed: {e}")
            raise RevisionError(f"Failed to create revision branch: {e}")
