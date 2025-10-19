"""
Pipeline Stage Abstract Base Class

Defines the interface for all LLM call pipeline stages.
Stages are composable units that transform context immutably.

Design Principles:
- Single Responsibility: Each stage handles ONE concern
- Immutability: Stages return new context, never mutate input
- Composability: Stages can be chained in any order
- Testability: Stages testable in isolation
- Error Handling: Clear contract for error propagation
"""

from abc import ABC, abstractmethod
from typing import Optional
import logging

from .context import LLMCallContext


class PipelineStage(ABC):
    """
    Abstract base class for all pipeline stages.

    Each stage implements the execute() method which:
    1. Receives an immutable LLMCallContext
    2. Performs stage-specific processing
    3. Returns a NEW LLMCallContext with updates

    Stages must NOT mutate the input context. Use context.with_*()
    methods to create new contexts with updates.

    Attributes:
        logger: Logger instance for stage-specific logging
        enabled: Whether this stage is enabled (can be toggled)

    Example Subclass:
        ```python
        class PIIRedactionStage(PipelineStage):
            @property
            def name(self) -> str:
                return "PIIRedaction"

            async def execute(self, context: LLMCallContext) -> LLMCallContext:
                # Redact PII from messages
                redacted_messages = self._redact_pii(context.get_effective_messages())
                return context.with_modified_messages(redacted_messages)
        ```

    Error Handling:
        - Fatal errors: Raise exceptions (e.g., InvalidInputError)
        - Non-fatal errors: Add to context.errors via context.with_error()
        - Stages should be defensive and validate inputs
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize pipeline stage.

        Args:
            enabled: Whether this stage is enabled. Disabled stages
                    skip execution and return context unchanged.
        """
        self.enabled = enabled
        self.logger = logging.getLogger(f"{__name__}.{self.name}")

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Stage name for logging and metadata.

        Returns:
            Human-readable stage name (e.g., "PIIRedaction", "InjectionFirewall")

        Note:
            Must be unique across all stages in a pipeline.
        """
        pass

    @abstractmethod
    async def execute(self, context: LLMCallContext) -> LLMCallContext:
        """
        Execute stage logic and return updated context.

        This is the main method that subclasses must implement.

        Args:
            context: Immutable input context

        Returns:
            New LLMCallContext with stage updates applied

        Raises:
            Exception: Fatal errors that should halt pipeline execution

        Implementation Requirements:
            1. MUST NOT mutate input context
            2. MUST return new context via context.with_*() methods
            3. SHOULD add stage metadata via context.with_stage_metadata()
            4. SHOULD log important actions
            5. MAY raise exceptions for fatal errors
            6. MAY add non-fatal errors via context.with_error()

        Example:
            ```python
            async def execute(self, context: LLMCallContext) -> LLMCallContext:
                self.logger.info(f"Executing {self.name}")

                # Process
                result = await self._do_work(context)

                # Return new context
                return context.with_stage_metadata(self.name, {
                    "action": "processed",
                    "result": result
                })
            ```
        """
        pass

    async def run(self, context: LLMCallContext) -> LLMCallContext:
        """
        Main entry point that wraps execute() with enable/disable logic.

        This method is called by the pipeline orchestrator. It handles:
        - Checking if stage is enabled
        - Pre-execution hooks
        - Actual execution
        - Post-execution hooks
        - Error handling and logging

        Args:
            context: Input context

        Returns:
            Updated context (or unchanged if stage disabled)

        Raises:
            Exception: Fatal errors from execute()
        """
        if not self.enabled:
            self.logger.debug(f"Stage {self.name} disabled, skipping")
            return context

        try:
            # Pre-execution hook
            context = await self.pre_execute(context)

            # Execute stage logic
            self.logger.debug(f"Executing stage: {self.name}")
            updated_context = await self.execute(context)

            # Post-execution hook
            updated_context = await self.post_execute(updated_context)

            self.logger.debug(f"Stage {self.name} completed successfully")
            return updated_context

        except Exception as e:
            self.logger.error(f"Stage {self.name} failed: {e}", exc_info=True)
            # Re-raise to halt pipeline (fatal error)
            raise

    async def pre_execute(self, context: LLMCallContext) -> LLMCallContext:
        """
        Hook called before execute().

        Subclasses can override to perform pre-execution setup,
        validation, or logging.

        Args:
            context: Input context

        Returns:
            Context (possibly modified)

        Default:
            Returns context unchanged
        """
        return context

    async def post_execute(self, context: LLMCallContext) -> LLMCallContext:
        """
        Hook called after execute().

        Subclasses can override to perform post-execution cleanup,
        validation, or logging.

        Args:
            context: Context returned from execute()

        Returns:
            Context (possibly modified)

        Default:
            Returns context unchanged
        """
        return context

    def enable(self) -> None:
        """Enable this stage."""
        self.enabled = True
        self.logger.info(f"Stage {self.name} enabled")

    def disable(self) -> None:
        """Disable this stage."""
        self.enabled = False
        self.logger.info(f"Stage {self.name} disabled")

    def is_enabled(self) -> bool:
        """
        Check if stage is enabled.

        Returns:
            True if enabled, False otherwise
        """
        return self.enabled

    def __repr__(self) -> str:
        """
        String representation for debugging.

        Returns:
            Human-readable representation
        """
        return f"{self.__class__.__name__}(name='{self.name}', enabled={self.enabled})"


class PipelineStageError(Exception):
    """
    Base exception for pipeline stage errors.

    All stage-specific exceptions should inherit from this.
    """

    def __init__(self, stage_name: str, message: str):
        """
        Initialize stage error.

        Args:
            stage_name: Name of the stage that raised the error
            message: Error description
        """
        self.stage_name = stage_name
        self.message = message
        super().__init__(f"[{stage_name}] {message}")


class StageValidationError(PipelineStageError):
    """Raised when stage input validation fails."""
    pass


class StageExecutionError(PipelineStageError):
    """Raised when stage execution fails fatally."""
    pass


class StageConfigurationError(PipelineStageError):
    """Raised when stage configuration is invalid."""
    pass
