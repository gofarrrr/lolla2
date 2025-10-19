"""
Supabase-backed checkpoint repository for persistent storage.
Part of Operation Polish - Phase 1: Database-Backed Checkpoint Persistence
"""

from typing import Any, Dict, List, Optional
from uuid import UUID
import logging

from src.core.checkpoint_models import (
    StateCheckpoint,
    AnalysisRevision,
    PipelineStage,
)
from src.services.orchestration_infra.checkpoint_repository import ICheckpointRepository
from src.services.persistence.database_service import DatabaseService
from src.core.unified_context_stream import ContextEventType
from src.core.stage_keys import StageKey

logger = logging.getLogger(__name__)


class SupabaseCheckpointRepository(ICheckpointRepository):
    """
    Database-backed checkpoint repository using Supabase PostgreSQL.
    Replaces in-memory CheckpointStorage with persistent storage.
    """

    def __init__(self, db_service: DatabaseService, context_stream: Any):
        self._db = db_service
        self._context_stream = context_stream

    # ============================================================================
    # Protocol methods (persistence) - Database-backed
    # ============================================================================

    async def save_checkpoint(self, checkpoint: StateCheckpoint) -> StateCheckpoint:
        """Save checkpoint to database as JSONB."""
        try:
            # Generate checkpoint_id if not present
            if checkpoint.checkpoint_id is None:
                from uuid import uuid4
                checkpoint.checkpoint_id = uuid4()

            logger.info(f"🔍 DEBUG: save_checkpoint called for {checkpoint.checkpoint_id}")

            # Serialize checkpoint to dict with JSON-compatible types (converts UUIDs to strings)
            checkpoint_dict = checkpoint.model_dump(mode='json')
            logger.info(f"🔍 DEBUG: checkpoint serialized, keys: {list(checkpoint_dict.keys())}")

            # Strip nested stage_history to prevent exponential data growth
            # Each checkpoint should only contain its own stage data, not the full history tree
            if 'stage_output' in checkpoint_dict and isinstance(checkpoint_dict['stage_output'], dict):
                if 'stage_history' in checkpoint_dict['stage_output']:
                    # Keep only stage names and timestamps, not full nested data
                    stage_history = checkpoint_dict['stage_output']['stage_history']
                    if isinstance(stage_history, list):
                        checkpoint_dict['stage_output']['stage_history'] = [
                            {
                                'stage': item.get('stage', 'unknown'),
                                'stage_name': item.get('stage_name', ''),
                                'completed_at': item.get('completed_at', ''),
                                # Remove nested 'result' to prevent exponential growth
                            }
                            for item in stage_history
                        ]
                    logger.info(f"🔍 DEBUG: Stripped nested stage_history data to reduce checkpoint size")

            # Prepare data for database insertion
            db_data = {
                "checkpoint_id": str(checkpoint.checkpoint_id),
                "trace_id": str(checkpoint.trace_id),
                "stage_name": checkpoint.stage_completed.value if hasattr(checkpoint.stage_completed, 'value') else str(checkpoint.stage_completed),
                "state_data": checkpoint_dict,  # Store full checkpoint as JSONB
                "user_id": str(checkpoint.user_id) if checkpoint.user_id else None,
                "session_id": str(checkpoint.session_id) if checkpoint.session_id else None,
            }
            logger.info(f"🔍 DEBUG: db_data prepared, checkpoint_id={db_data['checkpoint_id']}, trace_id={db_data['trace_id']}")

            logger.info(f"🔍 DEBUG: About to call _db.save_checkpoint_async()")
            # Transient-resilient save with exponential backoff
            import asyncio as _asyncio
            delays = [0.25, 0.75, 2.0]
            last_exc = None
            for attempt in range(len(delays) + 1):
                try:
                    result = await self._db.save_checkpoint_async(db_data)
                    logger.info(f"🔍 DEBUG: _db.save_checkpoint_async() returned: {result}")
                    break
                except Exception as e:
                    last_exc = e
                    if attempt < len(delays):
                        d = delays[attempt]
                        logger.warning(
                            f"⚠️ Checkpoint save attempt {attempt+1} failed, retrying in {d}s: {e}"
                        )
                        try:
                            await _asyncio.sleep(d)
                        except Exception:
                            pass
                    else:
                        logger.error(f"❌ Checkpoint save failed after retries: {e}")
                        raise

            logger.info(f"Saved checkpoint {checkpoint.checkpoint_id} to database")
            return checkpoint
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint {checkpoint.checkpoint_id}: {e}")
            logger.exception("Full traceback:")
            raise

    async def load_checkpoint(self, checkpoint_id: UUID) -> Optional[StateCheckpoint]:
        """Load checkpoint from database by ID."""
        try:
            row = await self._db.load_checkpoint_async(str(checkpoint_id))
            if not row:
                logger.warning(f"Checkpoint not found: {checkpoint_id}")
                return None

            # Deserialize from JSONB
            state_data = row.get("state_data", {})
            checkpoint = StateCheckpoint.model_validate(state_data)

            # Ensure stage fields are PipelineStage enums (defensive conversion)
            if isinstance(checkpoint.stage_completed, str):
                checkpoint.stage_completed = PipelineStage(checkpoint.stage_completed)
            if isinstance(checkpoint.next_stage, str):
                checkpoint.next_stage = PipelineStage(checkpoint.next_stage)

            logger.info(f"Loaded checkpoint {checkpoint_id} from database")
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            raise

    async def load_checkpoints_for_trace(self, trace_id: UUID) -> List[StateCheckpoint]:
        """Load all checkpoints for a trace, ordered by creation time (newest first)."""
        try:
            rows = await self._db.load_checkpoints_for_trace_async(str(trace_id))
            checkpoints = []
            for row in rows:
                state_data = row.get("state_data", {})
                checkpoint = StateCheckpoint.model_validate(state_data)

                # Ensure stage fields are PipelineStage enums (defensive conversion)
                if isinstance(checkpoint.stage_completed, str):
                    checkpoint.stage_completed = PipelineStage(checkpoint.stage_completed)
                if isinstance(checkpoint.next_stage, str):
                    checkpoint.next_stage = PipelineStage(checkpoint.next_stage)

                checkpoints.append(checkpoint)

            logger.info(f"Loaded {len(checkpoints)} checkpoints for trace {trace_id}")
            return checkpoints
        except Exception as e:
            logger.error(f"Failed to load checkpoints for trace {trace_id}: {e}")
            raise

    async def save_revision(self, revision: AnalysisRevision) -> AnalysisRevision:
        """Save revision (placeholder - not database-backed yet)."""
        logger.warning("save_revision called but revisions are not yet database-backed")
        return revision

    async def update_revision(self, revision: AnalysisRevision) -> AnalysisRevision:
        """Update revision (placeholder - not database-backed yet)."""
        logger.warning("update_revision called but revisions are not yet database-backed")
        return revision

    async def count_checkpoints_for_trace(self, trace_id: UUID) -> int:
        """Count checkpoints for a trace."""
        try:
            checkpoints = await self.load_checkpoints_for_trace(trace_id)
            return len(checkpoints)
        except Exception as e:
            logger.error(f"Failed to count checkpoints for trace {trace_id}: {e}")
            return 0

    # ============================================================================
    # Orchestrator helper methods (same as V1CheckpointRepository)
    # ============================================================================

    async def create_and_save_checkpoint(
        self,
        trace_id: UUID,
        stage_completed: PipelineStage,
        stage_output: Dict[str, Any],
        user_id: Optional[UUID] = None,
        session_id: Optional[UUID] = None,
    ) -> StateCheckpoint:
        """Create checkpoint from stage output and save to database."""
        logger.info(f"🔍 DEBUG: create_and_save_checkpoint called, trace_id={trace_id}, stage={stage_completed}")

        # Ensure stage_completed is a PipelineStage enum (fix for serialization bug)
        if isinstance(stage_completed, str):
            stage_completed = PipelineStage(stage_completed)

        # PROPER FIX: Deep copy stage_output to create immutable snapshot
        # Prevents aliasing bug where later mutations affect saved checkpoints
        import copy
        stage_output_snapshot = copy.deepcopy(stage_output)

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

        saved_checkpoint = await self.save_checkpoint(checkpoint)

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
        """Resume pipeline from a saved checkpoint."""
        checkpoint = await self.load_checkpoint(checkpoint_id)
        if not checkpoint:
            raise Exception(f"Checkpoint not found: {checkpoint_id}")
        if not checkpoint.can_resume_from():
            raise Exception(f"Cannot resume from checkpoint: {checkpoint.revision_status}")

        # Reconstruct minimal accumulated context (append stage outputs)
        accumulated_context: Dict[str, Any] = {}
        checkpoints = await self.load_checkpoints_for_trace(checkpoint.trace_id)

        # OPERATION FOUNDATION BUG FIX: Sort checkpoints oldest→newest for predictable merge
        # Database returns newest-first, but we need oldest-first so later stages override earlier
        checkpoints_oldest_first = sorted(checkpoints, key=lambda cp: cp.created_at)

        stage_history: List[Dict[str, Any]] = []

        # Track which canonical keys we successfully recovered for integrity telemetry
        recovered_canonical_keys: List[str] = []

        for cp in checkpoints_oldest_first:
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

            # OPERATION FOUNDATION BUG FIX: Merge ALL V6 canonical stage keys
            # The lightweight checkpoint architecture stores full accumulated_context in stage_output.
            # We must preserve ALL canonical keys (selection_results, etc), not just the current stage.
            # This fixes the bug where selection_results was lost between synergy_prompting→parallel_analysis.
            if isinstance(cp.stage_output, dict):
                # Get all valid StageKey enum values for filtering
                valid_stage_keys = {sk.value for sk in StageKey}

                # Merge ALL canonical V6 stage keys from this checkpoint
                # Process oldest→newest so later values override earlier (predictable merge)
                for key, value in cp.stage_output.items():
                    # Only merge canonical keys (in StageKey enum) and non-metadata keys
                    if key in valid_stage_keys and not key.startswith('_'):
                        accumulated_context[key] = value
                        # Track recovered keys for telemetry
                        if key not in recovered_canonical_keys:
                            recovered_canonical_keys.append(key)

                # ALSO preserve the nested stage-specific sub-dict for backward compatibility
                # Some code may still read context[stage_key] instead of context contracts
                if stage_key in cp.stage_output:
                    accumulated_context[stage_key] = cp.stage_output[stage_key]
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

        # OPERATION FOUNDATION BUG FIX: Resume integrity telemetry
        # Log which canonical keys were successfully recovered for early detection of issues
        logger.info(f"✅ Resume integrity check: recovered {len(recovered_canonical_keys)} canonical keys: {sorted(recovered_canonical_keys)}")
        self._context_stream.add_event(
            "resume_integrity_check",
            {
                "recovered_keys": sorted(recovered_canonical_keys),
                "recovered_count": len(recovered_canonical_keys),
                "has_consultant_selection": "consultant_selection" in recovered_canonical_keys,
                "checkpoint_count": len(checkpoints_oldest_first),
            },
        )

        # Log resume event
        resume_stage_value = (
            checkpoint.stage_completed.value
            if hasattr(checkpoint.stage_completed, "value")
            else str(checkpoint.stage_completed)
        )
        next_stage_value = (
            checkpoint.next_stage.value
            if hasattr(checkpoint.next_stage, "value")
            else str(checkpoint.next_stage)
        )
        self._context_stream.add_event(
            ContextEventType.ENGAGEMENT_STARTED,
            {
                "resume_from_checkpoint": str(checkpoint_id),
                "resume_from_stage": resume_stage_value,
                "next_stage": next_stage_value,
            },
        )

        return {
            "trace_id": checkpoint.trace_id,
            "current_stage": checkpoint.next_stage,
            "context": accumulated_context,
            "checkpoint_id": checkpoint_id,
        }
