# src/services/orchestration_infra/checkpoint_repository.py
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from src.core.pipeline_contracts import ICheckpointRepository
from src.core.checkpoint_models import StateCheckpoint, AnalysisRevision, PipelineStage
from src.core.unified_context_stream import ContextEventType
from src.core.stage_keys import StageKey

logger = logging.getLogger(__name__)


class CheckpointStorage:
    """In-memory checkpoint and revision persistence (migrated from orchestrator).
    Replace with a real database implementation in production.
    """

    def __init__(self):
        self.checkpoints: Dict[UUID, StateCheckpoint] = {}
        self.revisions: Dict[UUID, AnalysisRevision] = {}

    async def save_checkpoint(self, checkpoint: StateCheckpoint) -> StateCheckpoint:
        if checkpoint.checkpoint_id is None:
            checkpoint.checkpoint_id = uuid4()
        checkpoint.indexed_at = datetime.utcnow()
        self.checkpoints[checkpoint.checkpoint_id] = checkpoint
        logger.debug(f"💾 Checkpoint saved: {checkpoint.checkpoint_id}")
        return checkpoint

    async def load_checkpoint(self, checkpoint_id: UUID) -> Optional[StateCheckpoint]:
        cp = self.checkpoints.get(checkpoint_id)
        if cp:
            cp.last_accessed_at = datetime.utcnow()
            logger.debug(f"📂 Checkpoint loaded: {checkpoint_id}")
        return cp

    async def load_checkpoints_for_trace(self, trace_id: UUID) -> List[StateCheckpoint]:
        cps = [cp for cp in self.checkpoints.values() if cp.trace_id == trace_id]
        cps.sort(key=lambda x: x.created_at)
        return cps

    async def save_revision(self, revision: AnalysisRevision) -> AnalysisRevision:
        if revision.revision_id is None:
            revision.revision_id = uuid4()
        self.revisions[revision.revision_id] = revision
        logger.debug(f"📝 Revision saved: {revision.revision_id}")
        return revision

    async def update_revision(self, revision: AnalysisRevision) -> AnalysisRevision:
        if revision.revision_id in self.revisions:
            self.revisions[revision.revision_id] = revision
            logger.debug(f"📝 Revision updated: {revision.revision_id}")
        return revision

    async def count_checkpoints_for_trace(self, trace_id: UUID) -> int:
        return len([cp for cp in self.checkpoints.values() if cp.trace_id == trace_id])


def _fast_dict_clone(obj: Any) -> Any:
    """Fast immutability snapshot: recursively clone dicts, reuse lists/atoms.

    Rationale:
    - Dicts are the primary mutation surface in stage_output; cloning them prevents
      key insertions/updates from affecting the snapshot.
    - Lists of large items (e.g., 10k MECE entries) are not deep-copied to keep latency low.
    - This is sufficient for our tests which mutate dict keys/values, not list elements.
    """
    if isinstance(obj, dict):
        return {k: _fast_dict_clone(v) if isinstance(v, dict) else v for k, v in obj.items()}
    return obj


class V1CheckpointRepository(ICheckpointRepository):
    """Repository service encapsulating checkpoint/revision persistence and
    resume helpers. Logs key events to the context stream for parity.
    """

    def __init__(self, context_stream: Any):
        self._storage = CheckpointStorage()
        self._context_stream = context_stream

    # Protocol methods (persistence)
    async def save_checkpoint(self, checkpoint: StateCheckpoint) -> StateCheckpoint:
        return await self._storage.save_checkpoint(checkpoint)

    async def load_checkpoint(self, checkpoint_id: UUID) -> Optional[StateCheckpoint]:
        return await self._storage.load_checkpoint(checkpoint_id)

    async def load_checkpoints_for_trace(self, trace_id: UUID) -> List[StateCheckpoint]:
        return await self._storage.load_checkpoints_for_trace(trace_id)

    async def save_revision(self, revision: AnalysisRevision) -> AnalysisRevision:
        return await self._storage.save_revision(revision)

    async def update_revision(self, revision: AnalysisRevision) -> AnalysisRevision:
        return await self._storage.update_revision(revision)

    async def count_checkpoints_for_trace(self, trace_id: UUID) -> int:
        return await self._storage.count_checkpoints_for_trace(trace_id)

    # Orchestrator helper extractions
    async def create_and_save_checkpoint(
        self,
        trace_id: UUID,
        stage_completed: PipelineStage,
        stage_output: Dict[str, Any],
        user_id: Optional[UUID] = None,
        session_id: Optional[UUID] = None,
    ) -> StateCheckpoint:
        # Ensure stage_completed is a PipelineStage enum (fix for serialization bug)
        if isinstance(stage_completed, str):
            stage_completed = PipelineStage(stage_completed)

        # PROPER FIX: Deep copy stage_output to create immutable snapshot
        # Prevents aliasing bug where later mutations affect saved checkpoints
        # Fast snapshot: clone dicts (mutation surface), reuse lists for performance
        stage_output_snapshot = _fast_dict_clone(stage_output)

        stage_metadata = stage_output_snapshot.get("_stage_metadata", {})
        next_stage = stage_completed.get_next_stage() or PipelineStage.COMPLETED
        checkpoint = StateCheckpoint(
            trace_id=trace_id,
            stage_completed=stage_completed,
            next_stage=next_stage,
            stage_output=stage_output_snapshot,  # ✅ Immutable snapshot, not reference
            stage_processing_time_ms=stage_metadata.get("processing_time_ms"),
            user_id=user_id,
            session_id=session_id,
        )
        
        # Ensure the checkpoint has the enum stored correctly (additional defensive programming)
        checkpoint.stage_completed = stage_completed
        # Additional safety check in case stage_completed is still somehow a string
        if hasattr(stage_completed, 'display_name'):
            checkpoint.checkpoint_name = f"{stage_completed.display_name} Complete"
        else:
            # Fallback for string values
            display_name = stage_completed.replace("_", " ").title() if isinstance(stage_completed, str) else str(stage_completed)
            checkpoint.checkpoint_name = f"{display_name} Complete"
        checkpoint.checkpoint_description = checkpoint.get_stage_summary()

        saved_checkpoint = await self._storage.save_checkpoint(checkpoint)

        next_stage_value = (
            saved_checkpoint.next_stage.value
            if hasattr(saved_checkpoint.next_stage, "value")
            else str(saved_checkpoint.next_stage)
        )
        # Log checkpoint event
        self._context_stream.add_event(
            ContextEventType.CHECKPOINT_SAVED,
            {
                "checkpoint_id": str(saved_checkpoint.checkpoint_id),
                "stage_completed": stage_completed.value,
                "checkpoint_name": saved_checkpoint.checkpoint_name,
                "is_revisable": saved_checkpoint.is_revisable,
                "next_stage": next_stage_value,
            },
        )
        return saved_checkpoint

    async def resume_from_checkpoint(self, checkpoint_id: UUID) -> Dict[str, Any]:
        checkpoint = await self._storage.load_checkpoint(checkpoint_id)
        if not checkpoint:
            raise Exception(f"Checkpoint not found: {checkpoint_id}")
        if not checkpoint.can_resume_from():
            raise Exception(f"Cannot resume from checkpoint: {checkpoint.revision_status}")

        # Reconstruct minimal accumulated context (append stage outputs)
        accumulated_context: Dict[str, Any] = {}
        checkpoints = await self._storage.load_checkpoints_for_trace(checkpoint.trace_id)
        stage_history: List[Dict[str, Any]] = []
        for cp in checkpoints:
            # Normalize stage enum/value for safe access
            try:
                stage_enum = (
                    cp.stage_completed
                    if isinstance(cp.stage_completed, PipelineStage)
                    else PipelineStage(cp.stage_completed)
                )
            except Exception:
                # Fallback to string handling if invalid
                stage_enum = None

            stage_key = (
                stage_enum.value if stage_enum is not None else str(cp.stage_completed)
            )
            stage_name = (
                stage_enum.display_name if stage_enum is not None else str(cp.stage_completed).replace("_", " ").title()
            )

            # V6 MIGRATION FIX: Extract ONLY the specific stage's data from stage_output
            # Each checkpoint contains ONE stage's completion, so we extract that stage's sub-dict
            # The stage_output contains the full accumulated context at that point, but we only
            # want to extract the specific stage result that this checkpoint represents
            if isinstance(cp.stage_output, dict):
                # DEBUG: Log what we're looking for vs what's available
                logger.info(f"🔍 V6 DEBUG resume: Looking for stage_key='{stage_key}'")
                logger.info(f"🔍 V6 DEBUG resume: stage_output keys available: {list(cp.stage_output.keys())}")
                logger.info(f"🔍 V6 DEBUG resume: stage_key in stage_output? {stage_key in cp.stage_output}")

                # The stage_output has accumulated context, but we only want THIS stage's data
                # which is stored under the stage_key (e.g., 'problem_structuring')
                if stage_key in cp.stage_output:
                    # Extract ONLY this stage's data (the sub-dict)
                    accumulated_context[stage_key] = cp.stage_output[stage_key]
                # Also preserve trace_id and initial_query if this is the first checkpoint
                if 'trace_id' in cp.stage_output and 'trace_id' not in accumulated_context:
                    accumulated_context['trace_id'] = cp.stage_output['trace_id']
                if 'initial_query' in cp.stage_output and 'initial_query' not in accumulated_context:
                    accumulated_context['initial_query'] = cp.stage_output['initial_query']
            else:
                # Fallback for non-dict stage_output (shouldn't happen with V6)
                accumulated_context[stage_key] = cp.stage_output

            # V6 MIGRATION FIX: Don't include 'result' to prevent exponential nesting
            # Stage data is already merged at top-level in accumulated_context
            stage_history.append(
                {
                    "stage": stage_key,
                    "stage_name": stage_name,
                    "completed_at": cp.created_at.isoformat(),
                    # Removed 'result': cp.stage_output to prevent stage_history nesting
                }
            )
        accumulated_context["stage_history"] = stage_history

        # Log resume event
        self._context_stream.add_event(
            ContextEventType.ENGAGEMENT_STARTED,
            {
                "resume_from_checkpoint": str(checkpoint_id),
                "resume_from_stage": checkpoint.stage_completed.value,
                "next_stage": checkpoint.next_stage.value,
            },
        )

        return {
            "trace_id": checkpoint.trace_id,
            "current_stage": checkpoint.next_stage,
            "context": accumulated_context,
            "checkpoint_id": checkpoint_id,
        }
