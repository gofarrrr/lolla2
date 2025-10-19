"""Stage execution loop extracted from the stateful orchestrator."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from src.core.checkpoint_models import PipelineStage
from src.core.pipeline_contracts import PipelineState
from src.core.unified_context_stream import ContextEventType
from src.core.exceptions import PipelineError

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Orchestrates stage execution, checkpointing, and completion logic."""

    def __init__(
        self,
        *,
        context_stream,
        execute_stage,
        validate_stage_output,
        update_pipeline_state_with_contracts,
        merge_stage_result,
        handle_critical_stage_failure,
        log_non_critical_failure,
        save_checkpoint,
        check_revision_requested,
        count_checkpoints,
        create_final_report_contract,
        persist_final_report_to_database,
        cleanup_llm_manager,
    ) -> None:
        self.context_stream = context_stream
        self.execute_stage = execute_stage
        self.validate_stage_output = validate_stage_output
        self.update_pipeline_state_with_contracts = update_pipeline_state_with_contracts
        self.merge_stage_result = merge_stage_result
        self.handle_critical_stage_failure = handle_critical_stage_failure
        self.log_non_critical_failure = log_non_critical_failure
        self.save_checkpoint = save_checkpoint
        self.check_revision_requested = check_revision_requested
        self.count_checkpoints = count_checkpoints
        self.create_final_report_contract = create_final_report_contract
        self.persist_final_report_to_database = persist_final_report_to_database
        self.cleanup_llm_manager = cleanup_llm_manager

    async def run(
        self,
        *,
        trace_id: UUID,
        current_stage: PipelineStage,
        accumulated_context: Dict[str, Any],
        pipeline_state: PipelineState,
        start_time: float,
        initial_query: Optional[str],
        user_id: Optional[UUID],
        session_id: Optional[UUID],
    ) -> Dict[str, Any]:
        """Execute the stage loop copied from the stateful orchestrator."""
        try:
            while current_stage != PipelineStage.COMPLETED:
                logger.info("🔄 Executing stage: %s", current_stage.display_name)

                try:
                    stage_result = await self.execute_stage(
                        current_stage, accumulated_context, pipeline_state
                    )

                    validation_passed = await self.validate_stage_output(
                        current_stage, stage_result
                    )

                    if not validation_passed:
                        validation_error = Exception(
                            f"Stage output validation failed for {current_stage.display_name}"
                        )

                        if current_stage in {
                            PipelineStage.PARALLEL_ANALYSIS,
                            PipelineStage.DEVILS_ADVOCATE,
                            PipelineStage.SENIOR_ADVISOR,
                        }:
                            await self.handle_critical_stage_failure(
                                current_stage, validation_error, str(trace_id)
                            )
                        else:
                            self.log_non_critical_failure(
                                current_stage, validation_error, str(trace_id)
                            )

                    await self.update_pipeline_state_with_contracts(
                        pipeline_state, current_stage, stage_result
                    )

                    accumulated_context = self.merge_stage_result(
                        accumulated_context, current_stage, stage_result
                    )

                except Exception as stage_error:
                    logger.error(
                        "❌ Stage %s execution failed: %s",
                        current_stage.display_name,
                        stage_error,
                    )

                    if current_stage in {
                        PipelineStage.PARALLEL_ANALYSIS,
                        PipelineStage.DEVILS_ADVOCATE,
                        PipelineStage.SENIOR_ADVISOR,
                    }:
                        await self.handle_critical_stage_failure(
                            current_stage, stage_error, str(pipeline_state.trace_id)
                        )
                    else:
                        self.log_non_critical_failure(
                            current_stage, stage_error, str(pipeline_state.trace_id)
                        )
                        stage_result = {
                            "error": str(stage_error),
                            "fallback_used": True,
                            "stage_failed": True,
                            "stage_name": current_stage.display_name,
                        }
                        accumulated_context = self.merge_stage_result(
                            accumulated_context, current_stage, stage_result
                        )

                checkpoint = await self.save_checkpoint(
                    trace_id=trace_id,
                    stage_completed=current_stage,
                    stage_output=stage_result,
                    user_id=user_id,
                    session_id=session_id,
                )

                logger.info("💾 Checkpoint saved: %s", checkpoint.checkpoint_id)

                if await self.check_revision_requested(trace_id):
                    logger.info("⏸️ Revision requested - pausing pipeline execution")
                    return {
                        "status": "paused_for_revision",
                        "trace_id": trace_id,
                        "current_checkpoint": checkpoint.checkpoint_id,
                        "message": "Pipeline paused for user revision",
                    }

                current_stage = current_stage.get_next_stage() or PipelineStage.COMPLETED

            processing_time = time.time() - start_time
            self.context_stream.complete_engagement("completed")
            await self.context_stream.persist_to_database()

            try:
                if self.cleanup_llm_manager:
                    await self.cleanup_llm_manager()
            except Exception as cleanup_error:  # pragma: no cover - defensive log
                logger.warning("⚠️ LLM cleanup failed: %s", cleanup_error)

            logger.info("✅ Pipeline completed in %.2fs: %s", processing_time, trace_id)

            final_report = await self.create_final_report_contract(
                pipeline_state, processing_time
            )

            await self.persist_final_report_to_database(
                trace_id=trace_id,
                user_id=user_id,
                initial_query=initial_query or accumulated_context.get("initial_query", ""),
                final_report=final_report,
                processing_time=processing_time,
                accumulated_context=accumulated_context,
            )

            return {
                "status": "completed",
                "trace_id": trace_id,
                "final_result": accumulated_context,
                "final_report_contract": final_report.dict(),
                "pipeline_state": pipeline_state.dict(),
                "processing_time_seconds": processing_time,
                "total_checkpoints": await self.count_checkpoints(trace_id),
                "contract_efficiency_enabled": True,
            }

        except Exception as exc:
            logger.error("❌ Pipeline execution failed: %s", exc)
            self.context_stream.add_event(
                ContextEventType.ERROR_OCCURRED,
                {
                    "error": str(exc),
                    "stage": (
                        current_stage.value
                        if isinstance(current_stage, PipelineStage)
                        else "unknown"
                    ),
                },
            )

            try:
                if self.cleanup_llm_manager:
                    await self.cleanup_llm_manager()
            except Exception as cleanup_error:  # pragma: no cover - defensive log
                logger.warning("⚠️ LLM cleanup on error failed: %s", cleanup_error)

            raise PipelineError(f"Pipeline execution failed: {exc}") from exc


__all__ = ["PipelineRunner"]
