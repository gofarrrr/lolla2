"""
Project Service Error Handler
============================

Centralized error handling for project services implementing the Error Taxonomy
pattern from Red Team Amendments. Provides consistent error mapping, circuit
breaker protection, and comprehensive error recovery strategies.

Features:
- Error taxonomy mapping and categorization
- Circuit breaker pattern implementation
- Retry logic with exponential backoff
- Error context enrichment and logging
- Observability integration with UnifiedContextStream
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Callable, Type
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

from .specialized_contracts import (
    ProjectServiceError,
    ProjectNotFoundError,
    ProjectValidationError,
    ProjectOrchestrationError,
    ProjectAnalyticsError,
    ProjectRepositoryError,
    ICircuitBreaker,
    CircuitBreakerState,
)
from src.core.unified_context_stream import get_unified_context_stream, UnifiedContextStream


class ErrorCategory(Enum):
    """Error categorization for taxonomy mapping"""
    VALIDATION = "validation"
    REPOSITORY = "repository" 
    ORCHESTRATION = "orchestration"
    ANALYTICS = "analytics"
    NETWORK = "network"
    AUTHORIZATION = "authorization"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Rich error context for enhanced debugging"""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    operation: str
    timestamp: datetime
    user_id: Optional[str] = None
    project_id: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    circuit_breaker_triggered: bool = False


@dataclass
class RetryConfig:
    """Configuration for retry logic"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


class CircuitBreakerService(ICircuitBreaker):
    """Circuit breaker implementation for service resilience"""
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.logger = logging.getLogger(__name__)
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.breakers: Dict[str, CircuitBreakerState] = {}
    
    async def execute_with_breaker(
        self, 
        operation_name: str,
        operation_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with circuit breaker protection"""
        breaker_state = await self.get_breaker_state(operation_name)
        
        # Check if circuit is open
        if breaker_state.state == "open":
            if datetime.now(timezone.utc) < breaker_state.next_attempt_time:
                raise ProjectServiceError(
                    f"Circuit breaker open for operation: {operation_name}",
                    "CIRCUIT_BREAKER_OPEN",
                    {
                        "operation": operation_name,
                        "next_attempt_time": breaker_state.next_attempt_time.isoformat(),
                        "failure_count": breaker_state.failure_count,
                    }
                )
            else:
                # Move to half-open state
                self.breakers[operation_name].state = "half_open"
        
        try:
            # Execute the operation
            result = await operation_func(*args, **kwargs)
            
            # Success - reset or close circuit
            if breaker_state.state in ["half_open", "open"]:
                await self._reset_breaker(operation_name)
            
            return result
            
        except Exception as e:
            # Failure - record and potentially open circuit
            await self._record_failure(operation_name)
            raise
    
    async def get_breaker_state(self, service_name: str) -> CircuitBreakerState:
        """Get current circuit breaker state"""
        if service_name not in self.breakers:
            self.breakers[service_name] = CircuitBreakerState(
                service_name=service_name,
                state="closed",
                failure_count=0,
                last_failure_time=None,
                next_attempt_time=None,
            )
        
        return self.breakers[service_name]
    
    async def reset_breaker(self, service_name: str) -> bool:
        """Manually reset circuit breaker"""
        if service_name in self.breakers:
            await self._reset_breaker(service_name)
            return True
        return False
    
    async def _record_failure(self, operation_name: str) -> None:
        """Record operation failure and update circuit state"""
        if operation_name not in self.breakers:
            await self.get_breaker_state(operation_name)  # Initialize
        
        breaker = self.breakers[operation_name]
        breaker.failure_count += 1
        breaker.last_failure_time = datetime.now(timezone.utc)
        
        if breaker.failure_count >= self.failure_threshold:
            breaker.state = "open"
            breaker.next_attempt_time = datetime.now(timezone.utc) + timedelta(seconds=self.timeout_seconds)
            
            self.logger.warning(
                f"Circuit breaker opened for {operation_name} after {breaker.failure_count} failures"
            )
    
    async def _reset_breaker(self, operation_name: str) -> None:
        """Reset circuit breaker to closed state"""
        if operation_name in self.breakers:
            breaker = self.breakers[operation_name]
            breaker.state = "closed"
            breaker.failure_count = 0
            breaker.last_failure_time = None
            breaker.next_attempt_time = None
            
            self.logger.info(f"Circuit breaker reset for {operation_name}")


class ProjectErrorHandler:
    """
    Centralized error handling for project services
    
    Provides error taxonomy mapping, circuit breaker integration,
    retry logic, and comprehensive error context enrichment.
    """
    
    def __init__(
        self,
        context_stream: Optional[UnifiedContextStream] = None,
        circuit_breaker: Optional[ICircuitBreaker] = None,
        retry_config: Optional[RetryConfig] = None
    ):
        self.logger = logging.getLogger(__name__)
        self.context_stream = context_stream or get_unified_context_stream()
        self.circuit_breaker = circuit_breaker or CircuitBreakerService()
        self.retry_config = retry_config or RetryConfig()
        
        # Error taxonomy mapping
        self.error_taxonomy = self._build_error_taxonomy()
        
        self.logger.debug("🏗️ ProjectErrorHandler initialized")
    
    async def handle_error(
        self,
        error: Exception,
        operation: str,
        context: Dict[str, Any]
    ) -> ErrorContext:
        """Handle and categorize error with full context enrichment"""
        error_id = f"error_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{hash(str(error)) % 10000:04d}"
        
        # Categorize error
        category = self._categorize_error(error)
        severity = self._assess_severity(error, category)
        
        # Create error context
        error_context = ErrorContext(
            error_id=error_id,
            category=category,
            severity=severity,
            operation=operation,
            timestamp=datetime.now(timezone.utc),
            user_id=context.get("user_id"),
            project_id=context.get("project_id"),
            additional_data={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "operation_context": context,
            }
        )
        
        # Log error with context
        await self._log_error(error, error_context)
        
        # Emit observability event
        await self.context_stream.log_event(
            "PROJECT_SERVICE_ERROR",
            {
                "error_id": error_id,
                "category": category.value,
                "severity": severity.value,
                "operation": operation,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context,
            },
            metadata={
                "error_handling": True,
                "error_taxonomy": True,
            }
        )
        
        return error_context
    
    async def execute_with_protection(
        self,
        operation_name: str,
        operation_func: Callable,
        context: Dict[str, Any],
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with full error protection (circuit breaker + retry)"""
        retry_count = 0
        last_error = None
        
        while retry_count <= self.retry_config.max_attempts:
            try:
                if retry_count > 0:
                    # Calculate delay for retry
                    delay = min(
                        self.retry_config.base_delay * (self.retry_config.exponential_base ** (retry_count - 1)),
                        self.retry_config.max_delay
                    )
                    
                    if self.retry_config.jitter:
                        import random
                        delay *= (0.5 + random.random() * 0.5)  # Add jitter
                    
                    self.logger.info(f"Retrying {operation_name} (attempt {retry_count}) after {delay:.2f}s delay")
                    await asyncio.sleep(delay)
                
                # Execute with circuit breaker protection
                result = await self.circuit_breaker.execute_with_breaker(
                    operation_name,
                    operation_func,
                    *args,
                    **kwargs
                )
                
                # Success - log recovery if this was a retry
                if retry_count > 0:
                    await self.context_stream.log_event(
                        "OPERATION_RECOVERED",
                        {
                            "operation": operation_name,
                            "retry_count": retry_count,
                            "context": context,
                        },
                        metadata={"error_recovery": True}
                    )
                
                return result
                
            except Exception as e:
                last_error = e
                retry_count += 1
                
                # Handle and categorize error
                error_context = await self.handle_error(e, operation_name, context)
                error_context.retry_count = retry_count - 1
                
                # Check if we should retry
                if not self._should_retry(e, retry_count):
                    break
                
                # Check if circuit breaker triggered
                if "CIRCUIT_BREAKER_OPEN" in str(e):
                    error_context.circuit_breaker_triggered = True
                    break
        
        # All retries exhausted - handle final failure
        if last_error:
            await self.context_stream.log_event(
                "OPERATION_FAILED_PERMANENTLY",
                {
                    "operation": operation_name,
                    "total_attempts": retry_count,
                    "final_error": str(last_error),
                    "context": context,
                },
                metadata={"error_handling": True, "permanent_failure": True}
            )
            
            # Re-raise the last error
            raise last_error
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize error using taxonomy mapping"""
        error_type = type(error).__name__
        
        if error_type in self.error_taxonomy:
            return self.error_taxonomy[error_type]
        
        # Check inheritance hierarchy
        for error_class, category in self.error_taxonomy.items():
            try:
                if isinstance(error, eval(error_class)):
                    return category
            except:
                continue
        
        # Check error message patterns
        error_message = str(error).lower()
        
        if any(keyword in error_message for keyword in ["not found", "does not exist"]):
            return ErrorCategory.REPOSITORY
        elif any(keyword in error_message for keyword in ["validation", "invalid", "required"]):
            return ErrorCategory.VALIDATION
        elif any(keyword in error_message for keyword in ["network", "connection", "timeout"]):
            return ErrorCategory.NETWORK
        elif any(keyword in error_message for keyword in ["unauthorized", "forbidden", "access denied"]):
            return ErrorCategory.AUTHORIZATION
        
        return ErrorCategory.UNKNOWN
    
    def _assess_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Assess error severity based on type and category"""
        # Critical errors
        if isinstance(error, (SystemError, MemoryError, KeyboardInterrupt)):
            return ErrorSeverity.CRITICAL
        
        # High severity
        if category in [ErrorCategory.SYSTEM, ErrorCategory.AUTHORIZATION]:
            return ErrorSeverity.HIGH
        
        if isinstance(error, (ProjectRepositoryError, ProjectOrchestrationError)):
            return ErrorSeverity.HIGH
        
        # Medium severity
        if category in [ErrorCategory.NETWORK, ErrorCategory.ANALYTICS]:
            return ErrorSeverity.MEDIUM
        
        # Low severity
        if isinstance(error, ProjectValidationError):
            return ErrorSeverity.LOW
        
        # Default to medium
        return ErrorSeverity.MEDIUM
    
    def _should_retry(self, error: Exception, retry_count: int) -> bool:
        """Determine if operation should be retried"""
        if retry_count >= self.retry_config.max_attempts:
            return False
        
        # Don't retry validation errors
        if isinstance(error, ProjectValidationError):
            return False
        
        # Don't retry authorization errors
        if "unauthorized" in str(error).lower() or "forbidden" in str(error).lower():
            return False
        
        # Don't retry if circuit breaker is open
        if "CIRCUIT_BREAKER_OPEN" in str(error):
            return False
        
        # Retry for network, repository, and system errors
        retryable_categories = [
            ErrorCategory.NETWORK,
            ErrorCategory.REPOSITORY,
            ErrorCategory.SYSTEM,
        ]
        
        error_category = self._categorize_error(error)
        return error_category in retryable_categories
    
    async def _log_error(self, error: Exception, error_context: ErrorContext) -> None:
        """Log error with appropriate level based on severity"""
        log_message = (
            f"[{error_context.error_id}] {error_context.category.value.upper()} error in {error_context.operation}: "
            f"{type(error).__name__}: {str(error)}"
        )
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message, exc_info=True)
        elif error_context.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def _build_error_taxonomy(self) -> Dict[str, ErrorCategory]:
        """Build error taxonomy mapping"""
        return {
            # Validation errors
            "ProjectValidationError": ErrorCategory.VALIDATION,
            "ValidationError": ErrorCategory.VALIDATION,
            "ValueError": ErrorCategory.VALIDATION,
            
            # Repository errors
            "ProjectRepositoryError": ErrorCategory.REPOSITORY,
            "ProjectNotFoundError": ErrorCategory.REPOSITORY,
            "DatabaseError": ErrorCategory.REPOSITORY,
            "IntegrityError": ErrorCategory.REPOSITORY,
            
            # Orchestration errors
            "ProjectOrchestrationError": ErrorCategory.ORCHESTRATION,
            "WorkflowError": ErrorCategory.ORCHESTRATION,
            
            # Analytics errors
            "ProjectAnalyticsError": ErrorCategory.ANALYTICS,
            "CalculationError": ErrorCategory.ANALYTICS,
            
            # Network errors
            "ConnectionError": ErrorCategory.NETWORK,
            "TimeoutError": ErrorCategory.NETWORK,
            "HTTPError": ErrorCategory.NETWORK,
            
            # Authorization errors
            "PermissionError": ErrorCategory.AUTHORIZATION,
            "AuthenticationError": ErrorCategory.AUTHORIZATION,
            
            # System errors
            "SystemError": ErrorCategory.SYSTEM,
            "MemoryError": ErrorCategory.SYSTEM,
            "OSError": ErrorCategory.SYSTEM,
        }


def with_error_handling(operation_name: str):
    """Decorator for automatic error handling on service methods"""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            if not hasattr(self, 'error_handler'):
                # Fallback to basic error handling if no error_handler available
                return await func(self, *args, **kwargs)
            
            # Extract context from method arguments
            context = {
                "method": func.__name__,
                "service": self.__class__.__name__,
            }
            
            # Try to extract user_id and project_id from arguments
            for arg in args:
                if hasattr(arg, 'user_id'):
                    context["user_id"] = getattr(arg, 'user_id', None)
                if hasattr(arg, 'project_id'):
                    context["project_id"] = getattr(arg, 'project_id', None)
            
            return await self.error_handler.execute_with_protection(
                operation_name,
                func,
                context,
                self,
                *args,
                **kwargs
            )
        
        return wrapper
    return decorator


# ============================================================
# Factory Functions
# ============================================================

def get_circuit_breaker(
    failure_threshold: int = 5,
    timeout_seconds: int = 60
) -> ICircuitBreaker:
    """Factory function for circuit breaker"""
    return CircuitBreakerService(failure_threshold, timeout_seconds)


def get_error_handler(
    context_stream: Optional[UnifiedContextStream] = None,
    circuit_breaker: Optional[ICircuitBreaker] = None,
    retry_config: Optional[RetryConfig] = None
) -> ProjectErrorHandler:
    """Factory function for error handler"""
    return ProjectErrorHandler(context_stream, circuit_breaker, retry_config)
