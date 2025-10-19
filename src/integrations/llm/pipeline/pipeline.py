"""
LLM Call Pipeline Orchestrator

Executes pipeline stages sequentially, handling errors and telemetry.

Design Principles:
- Sequential Execution: Stages execute in registration order
- Fail-Fast: Fatal errors halt pipeline immediately
- Telemetry: Stage timing and metadata tracked
- Glass-Box: All operations logged for transparency
- Immutability: Context flows immutably through stages
"""

from typing import List, Optional, Dict, Any
import logging
import time
from datetime import datetime

from .context import LLMCallContext
from .stage import PipelineStage, PipelineStageError


class LLMCallPipeline:
    """
    Pipeline orchestrator that executes stages sequentially.

    The pipeline:
    1. Validates input context
    2. Executes stages in order
    3. Tracks timing and metadata per stage
    4. Handles errors (fatal vs non-fatal)
    5. Returns final context with response

    Attributes:
        stages: Ordered list of pipeline stages
        logger: Logger for pipeline operations

    Example:
        ```python
        # Create pipeline with stages
        pipeline = LLMCallPipeline(stages=[
            InjectionFirewallStage(),
            PIIRedactionStage(),
            ProviderAdapterStage(),
        ])

        # Execute
        context = LLMCallContext(
            messages=[{"role": "user", "content": "Hello"}],
            model="deepseek-chat",
            provider="deepseek",
            kwargs={}
        )
        result = await pipeline.execute(context)
        ```

    Error Handling:
        - Fatal errors: Stage raises exception → pipeline halts → exception propagated
        - Non-fatal errors: Stage adds to context.errors → pipeline continues
        - Validation errors: Invalid context → raises PipelineValidationError

    Telemetry:
        - Per-stage timing tracked in context.telemetry
        - Stage metadata preserved in context.stage_metadata
        - Total pipeline duration tracked
    """

    def __init__(self, stages: Optional[List[PipelineStage]] = None):
        """
        Initialize pipeline with stages.

        Args:
            stages: List of pipeline stages (executed in order).
                   If None, creates empty pipeline (stages can be added later).

        Example:
            >>> pipeline = LLMCallPipeline(stages=[
            ...     InjectionFirewallStage(),
            ...     PIIRedactionStage(),
            ... ])
        """
        self.stages: List[PipelineStage] = stages or []
        self.logger = logging.getLogger(__name__)

    def add_stage(self, stage: PipelineStage) -> None:
        """
        Add stage to pipeline (appended to end).

        Args:
            stage: Pipeline stage to add

        Example:
            >>> pipeline.add_stage(InjectionFirewallStage())
        """
        if not isinstance(stage, PipelineStage):
            raise TypeError(f"Stage must be PipelineStage instance, got {type(stage)}")

        self.stages.append(stage)
        self.logger.info(f"Added stage: {stage.name}")

    def add_stages(self, stages: List[PipelineStage]) -> None:
        """
        Add multiple stages to pipeline.

        Args:
            stages: List of stages to add

        Example:
            >>> pipeline.add_stages([
            ...     InjectionFirewallStage(),
            ...     PIIRedactionStage(),
            ... ])
        """
        for stage in stages:
            self.add_stage(stage)

    def remove_stage(self, stage_name: str) -> bool:
        """
        Remove stage from pipeline by name.

        Args:
            stage_name: Name of stage to remove

        Returns:
            True if stage was removed, False if not found

        Example:
            >>> pipeline.remove_stage("PIIRedaction")
            True
        """
        original_length = len(self.stages)
        self.stages = [s for s in self.stages if s.name != stage_name]
        removed = len(self.stages) < original_length

        if removed:
            self.logger.info(f"Removed stage: {stage_name}")
        else:
            self.logger.warning(f"Stage not found: {stage_name}")

        return removed

    def get_stage(self, stage_name: str) -> Optional[PipelineStage]:
        """
        Get stage by name.

        Args:
            stage_name: Name of stage to find

        Returns:
            Stage instance or None if not found
        """
        for stage in self.stages:
            if stage.name == stage_name:
                return stage
        return None

    def get_stage_names(self) -> List[str]:
        """
        Get names of all stages in pipeline.

        Returns:
            List of stage names in execution order
        """
        return [stage.name for stage in self.stages]

    def get_enabled_stages(self) -> List[PipelineStage]:
        """
        Get list of enabled stages.

        Returns:
            List of stages where enabled=True
        """
        return [stage for stage in self.stages if stage.is_enabled()]

    async def execute(self, context: LLMCallContext) -> LLMCallContext:
        """
        Execute pipeline stages sequentially.

        This is the main entry point for pipeline execution.

        Args:
            context: Input context with original request

        Returns:
            Final context after all stages executed

        Raises:
            PipelineValidationError: If input context is invalid
            PipelineStageError: If a stage raises a fatal error
            Exception: Any other unhandled exception from stages

        Workflow:
            1. Validate input context
            2. Log pipeline start
            3. For each stage:
               a. Check if enabled
               b. Record start time
               c. Execute stage
               d. Record duration
               e. Update context telemetry
            4. Log pipeline completion
            5. Return final context

        Example:
            >>> context = LLMCallContext(...)
            >>> final_context = await pipeline.execute(context)
            >>> if final_context.has_errors():
            ...     print("Non-fatal errors:", final_context.errors)
        """
        # Validate input
        self._validate_context(context)

        # Log start
        pipeline_start = time.time()
        self.logger.info(
            f"🚀 Pipeline execution started: {len(self.stages)} stages "
            f"({len(self.get_enabled_stages())} enabled)"
        )
        self.logger.debug(f"Input context: {context}")

        # Track stage timings
        stage_timings: Dict[str, float] = {}

        # Execute stages sequentially
        current_context = context
        for i, stage in enumerate(self.stages):
            stage_num = i + 1

            if not stage.is_enabled():
                self.logger.debug(
                    f"⏭️  Stage {stage_num}/{len(self.stages)}: {stage.name} (SKIPPED - disabled)"
                )
                continue

            self.logger.info(
                f"▶️  Stage {stage_num}/{len(self.stages)}: {stage.name}"
            )

            try:
                # Execute stage with timing
                stage_start = time.time()
                current_context = await stage.run(current_context)
                stage_duration_ms = (time.time() - stage_start) * 1000

                # Track timing
                stage_timings[stage.name] = stage_duration_ms

                self.logger.info(
                    f"✅ Stage {stage_num}/{len(self.stages)}: {stage.name} "
                    f"completed in {stage_duration_ms:.1f}ms"
                )

            except PipelineStageError as e:
                # Stage raised fatal error
                self.logger.error(
                    f"❌ Stage {stage_num}/{len(self.stages)}: {stage.name} FAILED: {e}"
                )
                # Add error to context before propagating
                current_context = current_context.with_error(
                    f"Stage {stage.name} failed: {str(e)}"
                )
                # Halt pipeline
                raise

            except Exception as e:
                # Unexpected error
                self.logger.error(
                    f"❌ Stage {stage_num}/{len(self.stages)}: {stage.name} "
                    f"raised unexpected error: {e}",
                    exc_info=True
                )
                # Add error to context
                current_context = current_context.with_error(
                    f"Stage {stage.name} unexpected error: {str(e)}"
                )
                # Halt pipeline (unexpected errors are fatal)
                raise

        # Pipeline complete
        pipeline_duration_ms = (time.time() - pipeline_start) * 1000

        # Add pipeline telemetry
        current_context = current_context.with_telemetry("pipeline_duration_ms", pipeline_duration_ms)
        current_context = current_context.with_telemetry("stage_timings", stage_timings)
        current_context = current_context.with_telemetry("stages_executed", len(stage_timings))

        # Log completion
        self.logger.info(
            f"🎉 Pipeline execution completed in {pipeline_duration_ms:.1f}ms "
            f"({len(stage_timings)} stages executed)"
        )

        # Log non-fatal errors if any
        if current_context.has_errors():
            self.logger.warning(
                f"⚠️  Pipeline completed with {len(current_context.errors)} non-fatal error(s):"
            )
            for err in current_context.errors:
                self.logger.warning(f"   - {err}")

        # Emit glass-box event for transparency
        self._emit_glass_box_event(current_context, stage_timings, pipeline_duration_ms)

        return current_context

    def _validate_context(self, context: LLMCallContext) -> None:
        """
        Validate input context before pipeline execution.

        Args:
            context: Context to validate

        Raises:
            PipelineValidationError: If validation fails

        Validates:
            - Context is LLMCallContext instance
            - Messages is non-empty
            - Provider is non-empty
            - Model is non-empty
        """
        if not isinstance(context, LLMCallContext):
            raise PipelineValidationError(
                f"Context must be LLMCallContext, got {type(context)}"
            )

        # Context __post_init__ already validates these, but double-check
        if not context.messages:
            raise PipelineValidationError("Context messages cannot be empty")

        if not context.provider or not context.provider.strip():
            raise PipelineValidationError("Context provider cannot be empty")

        if not context.model or not context.model.strip():
            raise PipelineValidationError("Context model cannot be empty")

    def _emit_glass_box_event(
        self,
        context: LLMCallContext,
        stage_timings: Dict[str, float],
        total_duration_ms: float
    ) -> None:
        """
        Emit glass-box transparency event for pipeline execution.

        Args:
            context: Final context
            stage_timings: Per-stage timing in milliseconds
            total_duration_ms: Total pipeline duration in milliseconds

        Note:
            This is a best-effort operation. Failures are logged but not propagated.
        """
        try:
            from src.core.unified_context_stream import get_unified_context_stream, ContextEventType

            cs = get_unified_context_stream()
            cs.add_event(ContextEventType.LLM_CALL_COMPLETE, {
                "event_type": "pipeline_execution_complete",
                "provider": context.get_effective_provider(),
                "model": context.get_effective_model(),
                "stages_executed": len(stage_timings),
                "stage_timings_ms": stage_timings,
                "total_duration_ms": total_duration_ms,
                "errors": context.errors,
                "has_response": context.has_response(),
                "timestamp": datetime.now().isoformat(),
            })

        except Exception as e:
            # Glass-box logging is best-effort, don't fail pipeline
            self.logger.warning(f"Failed to emit glass-box event: {e}")

    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get pipeline configuration info.

        Returns:
            Dict with pipeline metadata

        Example:
            >>> info = pipeline.get_pipeline_info()
            >>> info["total_stages"]
            5
        """
        return {
            "total_stages": len(self.stages),
            "enabled_stages": len(self.get_enabled_stages()),
            "stage_names": self.get_stage_names(),
            "stage_details": [
                {
                    "name": stage.name,
                    "enabled": stage.is_enabled(),
                    "class": stage.__class__.__name__,
                }
                for stage in self.stages
            ],
        }

    def __repr__(self) -> str:
        """
        String representation for debugging.

        Returns:
            Human-readable representation
        """
        return (
            f"LLMCallPipeline(stages={len(self.stages)}, "
            f"enabled={len(self.get_enabled_stages())})"
        )


class PipelineValidationError(Exception):
    """Raised when pipeline input validation fails."""

    def __init__(self, message: str):
        """
        Initialize validation error.

        Args:
            message: Error description
        """
        self.message = message
        super().__init__(f"Pipeline validation failed: {message}")


class PipelineExecutionError(Exception):
    """Raised when pipeline execution fails."""

    def __init__(self, message: str, stage_name: Optional[str] = None):
        """
        Initialize execution error.

        Args:
            message: Error description
            stage_name: Name of stage that caused error (if applicable)
        """
        self.message = message
        self.stage_name = stage_name
        error_msg = f"Pipeline execution failed"
        if stage_name:
            error_msg += f" at stage '{stage_name}'"
        error_msg += f": {message}"
        super().__init__(error_msg)
