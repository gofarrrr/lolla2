"""
V6 Executor Adapter - Temporary Bridge
======================================

Adapts V6 IStageExecutor (PipelineState-based) to new flow contracts (dict-based).

**LIFECYCLE**: This adapter is TEMPORARY. Once all stage executors are migrated
to the new flow-based contracts, DELETE THIS FILE.

**PURPOSE**: Isolate the impedance mismatch between V6 executors and new architecture,
enabling incremental migration without disrupting existing tests or functionality.
"""
import logging
from typing import Any, Dict, Mapping

from src.orchestration.flow_contracts import (
    StageExecutor,
    StageInput,
    StageResult,
    StageStatus,
)
from src.core.pipeline_contracts import IStageExecutor, PipelineState

logger = logging.getLogger(__name__)


class V6ExecutorAdapter:
    """
    Wraps a V6 IStageExecutor to work with new flow contracts.

    Conversion Strategy:
    1. dict context → PipelineState (via _context_to_pipeline_state)
    2. Execute V6 executor (receives/returns PipelineState)
    3. PipelineState → dict (via _pipeline_state_to_context)

    This is a BRIDGE pattern implementation enabling V6 executors to work
    with the new DefaultPipelineManager without modification.
    """

    def __init__(self, v6_executor: IStageExecutor):
        """
        Initialize adapter with a V6 executor.

        Args:
            v6_executor: Any V6 stage executor implementing IStageExecutor protocol
        """
        self.v6_executor = v6_executor

    @property
    def idempotent(self) -> bool:
        """V6 executors are designed to be idempotent."""
        return True

    async def execute(self, sinput: StageInput) -> StageResult:
        """
        Execute V6 executor using flow contract interface.

        Args:
            sinput: StageInput containing context dict and metadata

        Returns:
            StageResult with updated context or error
        """
        try:
            # Step 1: Convert dict context → PipelineState
            logger.debug(f"🔄 V6 Adapter: Converting context dict to PipelineState for stage {sinput.stage.id}")
            state = self._context_to_pipeline_state(dict(sinput.context))

            # Step 2: Execute V6 executor
            logger.debug(f"⚙️ V6 Adapter: Executing V6 executor: {self.v6_executor.__class__.__name__}")
            updated_state = await self.v6_executor.execute(state)

            # Step 3: Convert PipelineState → dict context
            logger.debug(f"🔄 V6 Adapter: Converting PipelineState back to context dict")
            output = self._pipeline_state_to_context(updated_state)

            logger.info(f"✅ V6 Adapter: Stage completed successfully")
            return StageResult(
                status=StageStatus.SUCCEEDED,
                output=output,
                metrics={}
            )

        except Exception as e:
            logger.error(f"❌ V6 Adapter: Execution failed - {type(e).__name__}: {e}")
            return StageResult(
                status=StageStatus.FAILED,
                output={},
                metrics={},
                error=str(e)
            )

    def _context_to_pipeline_state(self, context: Dict[str, Any]) -> PipelineState:
        """
        Convert context dict to PipelineState using V6 keys with validation.

        EXTRACTED FROM: stateful_pipeline_orchestrator.py:525-718

        DESIGN PRINCIPLE: Fail-fast validation prevents legacy keys from sneaking in.
        Uses StageKey enum exclusively for type-safe, consistent key access.

        Raises:
            ValueError: If legacy keys are present
            TypeError: If values are not the expected type
        """
        from src.core.stage_keys import StageKey, validate_no_legacy_keys
        from src.core.pipeline_contracts import (
            SocraticOutput,
            ProblemStructuringOutput,
            ConsultantSelectionOutput,
            ParallelAnalysisOutput,
            DevilsAdvocateOutput,
            SeniorAdvisorOutput,
            BriefingMemo,
            InteractionSweepOutput,
        )
        from datetime import datetime, timezone

        # GUARDRAIL: Fail loud if legacy keys detected
        validate_no_legacy_keys(context)

        # Helper for safe type conversion
        def safe_convert(data, model_class, key_name):
            """Convert dict to Pydantic model with clear error messages."""
            if data is None:
                return None
            if isinstance(data, model_class):
                return data
            if isinstance(data, dict):
                try:
                    return model_class.model_validate(data)
                except Exception as e:
                    logger.error(f"❌ V6 CONVERSION FAILED: {key_name} → {model_class.__name__}")
                    logger.error(f"   Error: {e}")
                    logger.error(f"   Data type: {type(data)}")
                    logger.error(f"   Data keys: {list(data.keys())}")
                    raise TypeError(f"Invalid {key_name} data: {e}") from e
            raise TypeError(f"{key_name} must be dict or {model_class.__name__}, got {type(data)}")

        # Extract core fields
        initial_query = context.get(StageKey.INITIAL_QUERY.value, "")
        trace_id = str(context.get(StageKey.TRACE_ID.value, ""))

        # Defensive fallback for initial_query
        if not initial_query:
            try:
                socratic_block = context.get(StageKey.SOCRATIC.value)
                if isinstance(socratic_block, dict):
                    cand = socratic_block.get("clarified_problem_statement")
                    if isinstance(cand, str) and cand.strip():
                        initial_query = cand.strip()
                        logger.info("🔍 V6 FALLBACK: initial_query from Socratic")
            except Exception:
                pass

        # Extract stage results using safe_convert
        import copy

        socratic_source = context.get(StageKey.SOCRATIC.value)
        socratic_results = safe_convert(socratic_source, SocraticOutput, "socratic_questions")

        structuring_data_raw = context.get(StageKey.STRUCTURING.value)
        structuring_data = copy.deepcopy(structuring_data_raw) if structuring_data_raw else None
        structuring_results = safe_convert(structuring_data, ProblemStructuringOutput, "problem_structuring")

        interaction_sweep_data = context.get(StageKey.INTERACTION_SWEEP.value)
        interaction_sweep_results = safe_convert(interaction_sweep_data, InteractionSweepOutput, "interaction_sweep")

        oracle_data = context.get(StageKey.ORACLE.value)
        briefing_memo = safe_convert(oracle_data, BriefingMemo, "oracle_research")

        selection_data = context.get(StageKey.SELECTION.value)
        selection_results = safe_convert(selection_data, ConsultantSelectionOutput, "consultant_selection")

        analysis_data = context.get(StageKey.ANALYSIS.value)
        analysis_results = safe_convert(analysis_data, ParallelAnalysisOutput, "parallel_analysis")

        critique_data = context.get(StageKey.DEVILS_ADVOCATE.value)
        critique_results = safe_convert(critique_data, DevilsAdvocateOutput, "devils_advocate")

        advisor_data = context.get(StageKey.SENIOR_ADVISOR.value)
        final_results = safe_convert(advisor_data, SeniorAdvisorOutput, "senior_advisor")

        # Build PipelineState
        return PipelineState(
            initial_query=initial_query,
            trace_id=trace_id,
            started_at=datetime.now(timezone.utc),
            socratic_results=socratic_results,
            structuring_results=structuring_results,
            interaction_sweep_results=interaction_sweep_results,
            briefing_memo=briefing_memo,
            selection_results=selection_results,
            analysis_results=analysis_results,
            critique_results=critique_results,
            final_results=final_results,
            enhancement_metadata=context.get("enhancement_metadata"),
            complexity_level=context.get("pipeline_complexity_level"),
            processing_mode=context.get("pipeline_processing_mode"),
        )

    def _pipeline_state_to_context(self, state: PipelineState) -> Dict[str, Any]:
        """
        Convert PipelineState contract to context dict using V6 keys only.

        EXTRACTED FROM: stateful_pipeline_orchestrator.py:719-781

        DESIGN PRINCIPLE: Single source of truth - uses StageKey enum values exclusively.
        No legacy key aliases (progressive_questions, socratic_results, etc.) are generated.
        """
        from src.core.stage_keys import StageKey

        result: Dict[str, Any] = {}

        # Helper to safely dump Pydantic models
        def safe_dump(obj):
            return obj.model_dump() if hasattr(obj, "model_dump") else obj

        # V6 canonical keys only
        if state.socratic_results:
            result[StageKey.SOCRATIC.value] = safe_dump(state.socratic_results)

        if state.structuring_results:
            result[StageKey.STRUCTURING.value] = safe_dump(state.structuring_results)

        if state.interaction_sweep_results:
            result[StageKey.INTERACTION_SWEEP.value] = safe_dump(state.interaction_sweep_results)

        if state.briefing_memo:
            result[StageKey.ORACLE.value] = safe_dump(state.briefing_memo)
            result["oracle_enabled"] = True
            result["citation_count"] = len(state.briefing_memo.citations)
            result["key_findings_count"] = len(state.briefing_memo.key_findings)

        if state.selection_results:
            result[StageKey.SELECTION.value] = safe_dump(state.selection_results)

        if state.analysis_results:
            result[StageKey.ANALYSIS.value] = safe_dump(state.analysis_results)

        if state.critique_results:
            result[StageKey.DEVILS_ADVOCATE.value] = safe_dump(state.critique_results)

        if state.final_results:
            result[StageKey.SENIOR_ADVISOR.value] = safe_dump(state.final_results)

        # Preserve core fields
        if state.initial_query:
            result[StageKey.INITIAL_QUERY.value] = state.initial_query

        if state.trace_id:
            result[StageKey.TRACE_ID.value] = state.trace_id

        if state.enhancement_metadata:
            result["enhancement_metadata"] = state.enhancement_metadata
        if state.complexity_level:
            result["pipeline_complexity_level"] = state.complexity_level
        if state.processing_mode:
            result["pipeline_processing_mode"] = state.processing_mode

        return result
