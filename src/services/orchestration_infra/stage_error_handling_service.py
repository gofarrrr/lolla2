"""
Stage Error Handling Service
=============================

OPERATION SCALPEL V2 - Phase 5: Cross-Cutting Concern Extraction

This service centralizes error handling logic for pipeline stage execution,
providing consistent error logging, event generation, and exception transformation.

Service Responsibilities:
- Standardized error logging with context
- Error event generation for UnifiedContextStream
- Exception type transformation (generic -> domain-specific)
- Error recovery guidance

Pattern: Cross-cutting concern service with dependency injection
Status: Phase 5 - Error handling extraction complete
"""

import logging
import asyncio
import time
from typing import Dict, Any, Optional, Callable, TypeVar, Awaitable
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for stage execution failures"""
    CRITICAL = "critical"  # Pipeline cannot continue
    HIGH = "high"  # Stage failed but pipeline may recover
    MEDIUM = "medium"  # Partial failure, degraded results
    LOW = "low"  # Minor issue, full recovery possible


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


T = TypeVar('T')


class StageErrorHandlingService:
    """
    OPERATION SCALPEL V2 - Phase 5: Stage Error Handling Service

    Centralizes error handling for pipeline stage execution, providing
    consistent error logging, event generation, and exception transformation.

    Benefits:
    - Consistent error handling across all stages
    - Centralized error logging and monitoring
    - Simplified error recovery logic
    - Better error diagnostics and debugging
    """

    def __init__(self, context_stream):
        """
        Initialize Stage Error Handling Service

        Args:
            context_stream: UnifiedContextStream for error event logging
        """
        self.context_stream = context_stream

        # OPERATION SCALPEL V2 - Phase 7: Circuit breaker state
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}  # stage_name -> circuit state
        self.failure_counts: Dict[str, int] = {}  # stage_name -> failure count
        self.last_failure_time: Dict[str, datetime] = {}  # stage_name -> last failure timestamp

        # Circuit breaker configuration
        self.failure_threshold = 5  # Failures before opening circuit
        self.reset_timeout = 60  # Seconds before attempting recovery
        self.half_open_timeout = 30  # Seconds to test recovery

        logger.info("🔗 StageErrorHandlingService initialized (Phase 7: With retry & circuit breaker)")

    def handle_stage_execution_error(
        self,
        stage_name: str,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
    ) -> Exception:
        """
        Handle a stage execution error with consistent logging and event generation.

        This method:
        1. Logs the error with full context
        2. Adds error event to UnifiedContextStream
        3. Returns a domain-specific exception to raise

        Args:
            stage_name: Name of the stage that failed
            error: The exception that occurred
            context: Optional additional context for debugging
            severity: Error severity level

        Returns:
            Domain-specific exception to raise (PipelineError)
        """
        # Import here to avoid circular dependency
        from src.core.exceptions import PipelineError
        from src.core.unified_context_stream import ContextEventType

        # Log error with context
        error_message = f"❌ Stage execution failed ({stage_name}): {error}"
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(error_message)
        elif severity == ErrorSeverity.HIGH:
            logger.error(error_message)
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(error_message)
        else:
            logger.info(error_message)

        # Add detailed error event to context stream
        error_data = {
            "stage": stage_name,
            "error": str(error),
            "error_type": type(error).__name__,
            "severity": severity.value,
        }

        # Add additional context if provided
        if context:
            error_data["additional_context"] = context

        try:
            self.context_stream.add_event(
                ContextEventType.ERROR_OCCURRED,
                error_data
            )
        except Exception as stream_error:
            # If context stream fails, just log it - don't fail the error handling
            logger.warning(f"Failed to add error event to context stream: {stream_error}")

        # Return domain-specific exception
        return PipelineError(f"Stage {stage_name} execution failed: {error}")

    def handle_checkpoint_error(
        self,
        operation: str,
        error: Exception,
        checkpoint_id: Optional[str] = None,
    ) -> Exception:
        """
        Handle checkpoint operation errors.

        Args:
            operation: Operation that failed (e.g., "save", "resume")
            error: The exception that occurred
            checkpoint_id: Optional checkpoint ID for context

        Returns:
            Domain-specific CheckpointError to raise
        """
        # Import here to avoid circular dependency
        from src.core.exceptions import CheckpointError

        # Log error
        error_message = f"❌ Checkpoint {operation} failed"
        if checkpoint_id:
            error_message += f" (checkpoint_id: {checkpoint_id})"
        error_message += f": {error}"
        logger.error(error_message)

        # Return domain-specific exception
        if checkpoint_id:
            return CheckpointError(f"Failed to {operation} checkpoint {checkpoint_id}: {error}")
        else:
            return CheckpointError(f"Failed to {operation} checkpoint: {error}")

    def with_error_handling(
        self,
        stage_name: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
    ):
        """
        Decorator for wrapping stage execution with error handling.

        Usage:
            @error_service.with_error_handling("socratic_questions")
            async def _execute_socratic_questions(self, context):
                # Stage logic here
                pass

        Args:
            stage_name: Name of the stage for error reporting
            severity: Error severity level

        Returns:
            Decorator function
        """
        def decorator(func: Callable):
            async def wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    # Handle the error and re-raise domain-specific exception
                    domain_error = self.handle_stage_execution_error(
                        stage_name=stage_name,
                        error=e,
                        severity=severity,
                    )
                    raise domain_error
            return wrapper
        return decorator

    def is_recoverable_error(self, error: Exception) -> bool:
        """
        Determine if an error is recoverable.

        Args:
            error: The exception to check

        Returns:
            True if the error is recoverable, False otherwise
        """
        # Import here to avoid circular dependency
        from src.core.exceptions import PipelineError

        # Define recoverable error types
        recoverable_types = (
            TimeoutError,
            ConnectionError,
            # Add more recoverable types as needed
        )

        # PipelineErrors are generally not recoverable
        if isinstance(error, PipelineError):
            return False

        return isinstance(error, recoverable_types)

    def suggest_recovery_action(self, error: Exception, stage_name: str) -> str:
        """
        Suggest a recovery action for an error.

        Args:
            error: The exception that occurred
            stage_name: Name of the stage that failed

        Returns:
            Human-readable recovery suggestion
        """
        if isinstance(error, TimeoutError):
            return f"Retry {stage_name} with increased timeout"
        elif isinstance(error, ConnectionError):
            return f"Check network connectivity and retry {stage_name}"
        elif "rate limit" in str(error).lower():
            return f"Wait and retry {stage_name} after rate limit reset"
        else:
            return f"Review {stage_name} inputs and retry or skip to next stage"

    # OPERATION SCALPEL V2 - Phase 7: Advanced Error Handling Features

    async def with_retry(
        self,
        func: Callable[..., Awaitable[T]],
        stage_name: str,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        *args,
        **kwargs
    ) -> T:
        """
        Execute a function with exponential backoff retry logic.

        Args:
            func: Async function to execute
            stage_name: Name of the stage for logging
            max_retries: Maximum number of retry attempts
            backoff_factor: Exponential backoff multiplier
            *args, **kwargs: Arguments to pass to func

        Returns:
            Result from successful function execution

        Raises:
            Last exception if all retries exhausted
        """
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                result = await func(*args, **kwargs)

                # Success - reset failure count
                if stage_name in self.failure_counts:
                    self.failure_counts[stage_name] = 0

                logger.info(f"✅ {stage_name} succeeded on attempt {attempt + 1}/{max_retries + 1}")
                return result

            except Exception as e:
                last_exception = e

                if attempt < max_retries:
                    # Calculate backoff delay
                    delay = backoff_factor ** attempt
                    logger.warning(
                        f"⚠️ {stage_name} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    # All retries exhausted
                    logger.error(
                        f"❌ {stage_name} failed after {max_retries + 1} attempts: {e}"
                    )

        # All retries exhausted, raise the last exception
        if last_exception:
            raise last_exception

        raise RuntimeError(f"{stage_name} failed without exception (unexpected)")

    def get_circuit_state(self, stage_name: str) -> CircuitState:
        """
        Get current circuit breaker state for a stage.

        Args:
            stage_name: Name of the stage

        Returns:
            Current CircuitState
        """
        if stage_name not in self.circuit_breakers:
            self.circuit_breakers[stage_name] = {
                "state": CircuitState.CLOSED,
                "opened_at": None,
            }

        circuit = self.circuit_breakers[stage_name]
        state = circuit["state"]

        # Check if we should transition from OPEN to HALF_OPEN
        if state == CircuitState.OPEN:
            opened_at = circuit.get("opened_at")
            if opened_at and (datetime.now() - opened_at).total_seconds() >= self.reset_timeout:
                circuit["state"] = CircuitState.HALF_OPEN
                logger.info(f"🔧 Circuit breaker for {stage_name}: OPEN -> HALF_OPEN (testing recovery)")
                return CircuitState.HALF_OPEN

        return state

    def record_success(self, stage_name: str):
        """
        Record a successful execution, potentially closing the circuit.

        Args:
            stage_name: Name of the stage
        """
        if stage_name in self.failure_counts:
            self.failure_counts[stage_name] = 0

        if stage_name in self.circuit_breakers:
            circuit = self.circuit_breakers[stage_name]
            if circuit["state"] in (CircuitState.OPEN, CircuitState.HALF_OPEN):
                circuit["state"] = CircuitState.CLOSED
                circuit["opened_at"] = None
                logger.info(f"✅ Circuit breaker for {stage_name}: CLOSED (recovered)")

    def record_failure(self, stage_name: str):
        """
        Record a failure, potentially opening the circuit.

        Args:
            stage_name: Name of the stage
        """
        # Increment failure count
        self.failure_counts[stage_name] = self.failure_counts.get(stage_name, 0) + 1
        self.last_failure_time[stage_name] = datetime.now()

        failure_count = self.failure_counts[stage_name]

        # Check if we should open the circuit
        if failure_count >= self.failure_threshold:
            if stage_name not in self.circuit_breakers:
                self.circuit_breakers[stage_name] = {}

            circuit = self.circuit_breakers[stage_name]
            if circuit.get("state") != CircuitState.OPEN:
                circuit["state"] = CircuitState.OPEN
                circuit["opened_at"] = datetime.now()
                logger.warning(
                    f"🚨 Circuit breaker for {stage_name}: OPEN (threshold reached: {failure_count} failures)"
                )

    async def with_circuit_breaker(
        self,
        func: Callable[..., Awaitable[T]],
        stage_name: str,
        *args,
        **kwargs
    ) -> T:
        """
        Execute a function with circuit breaker pattern.

        Args:
            func: Async function to execute
            stage_name: Name of the stage
            *args, **kwargs: Arguments to pass to func

        Returns:
            Result from successful function execution

        Raises:
            RuntimeError if circuit is open
            Original exception if execution fails
        """
        # Check circuit state
        state = self.get_circuit_state(stage_name)

        if state == CircuitState.OPEN:
            raise RuntimeError(
                f"Circuit breaker for {stage_name} is OPEN. "
                f"Service is unavailable. Will retry in {self.reset_timeout}s."
            )

        try:
            result = await func(*args, **kwargs)
            self.record_success(stage_name)
            return result

        except Exception as e:
            self.record_failure(stage_name)
            raise e

    async def with_retry_and_circuit_breaker(
        self,
        func: Callable[..., Awaitable[T]],
        stage_name: str,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        *args,
        **kwargs
    ) -> T:
        """
        Execute a function with both retry logic and circuit breaker.

        Combines exponential backoff retry with circuit breaker pattern
        for maximum resilience.

        Args:
            func: Async function to execute
            stage_name: Name of the stage
            max_retries: Maximum number of retry attempts
            backoff_factor: Exponential backoff multiplier
            *args, **kwargs: Arguments to pass to func

        Returns:
            Result from successful function execution

        Raises:
            RuntimeError if circuit is open
            Last exception if all retries exhausted
        """
        return await self.with_circuit_breaker(
            lambda: self.with_retry(
                func,
                stage_name,
                max_retries,
                backoff_factor,
                *args,
                **kwargs
            ),
            stage_name
        )
