"""
Context Intelligence Error Taxonomy
A1 - Discovery & Taxonomy (Red Team Amendment Applied)
"""
import logging
from typing import Dict, Any, Optional
from src.core.unified_context_stream import UnifiedContextStream, ContextEventType
from .contracts import (
    ContextIntelligenceError, ProviderError, ParseError, 
    ValidationError, TimeoutError, CancellationError
)

logger = logging.getLogger(__name__)

# Back-compat aliases for seam tests expecting Context* error names
class ContextAnalysisError(Exception):
    pass

class ContextProviderError(ProviderError):
    """Alias for provider failures (back-compat)"""
    pass

class ContextParseError(ParseError):
    """Alias for parse failures (back-compat)"""
    pass

class ContextTimeoutError(TimeoutError):
    """Alias for timeout failures (back-compat)"""
    pass

class CircuitBreakerOpen(ProviderError):
    """Alias for circuit breaker open condition (maps to provider-level failure)"""
    pass

def map_exception_to_http_status(exc: Exception) -> int:
    """Map taxonomy errors to HTTP status codes (used by seam tests)"""
    from http import HTTPStatus
    if isinstance(exc, ContextTimeoutError):
        return 408
    if isinstance(exc, (ContextProviderError, CircuitBreakerOpen)):
        return 503
    if isinstance(exc, ContextParseError):
        return 400
    if isinstance(exc, ContextAnalysisError):
        return 500
    # Default fallback
    return 500

class ContextIntelligenceErrorMapper:
    """Central error mapping with pure functions (Red Team Amendment #3)"""
    
    ERROR_TAXONOMY = {
        # Provider calls (I/O failures)
        "redis.ConnectionError": ProviderError,
        "redis.TimeoutError": TimeoutError,
        "redis.ResponseError": ProviderError,
        "supabase.AuthApiError": ProviderError,
        "supabase.StorageApiError": ProviderError,
        "httpx.ConnectError": ProviderError,
        "httpx.TimeoutException": TimeoutError,
        
        # Parsing/serialization (data format issues)
        "json.JSONDecodeError": ParseError,
        "ValueError": ParseError,
        "KeyError": ParseError,
        "TypeError": ParseError,
        
        # Business-rule validation (logic failures)
        "AssertionError": ValidationError,
        "AttributeError": ValidationError,
        
        # Cancellation/timeout (async issues)
        "asyncio.TimeoutError": TimeoutError,
        "asyncio.CancelledError": CancellationError,
        "concurrent.futures.TimeoutError": TimeoutError,
    }
    
    @staticmethod
    def map_exception(exc: Exception) -> ContextIntelligenceError:
        """Map raw exception to taxonomy exception"""
        exc_type = f"{exc.__class__.__module__}.{exc.__class__.__name__}"
        
        # Check direct mapping
        if exc_type in ContextIntelligenceErrorMapper.ERROR_TAXONOMY:
            mapped_class = ContextIntelligenceErrorMapper.ERROR_TAXONOMY[exc_type]
            return mapped_class(f"Mapped from {exc_type}: {str(exc)}")
        
        # Check class name only
        exc_name = exc.__class__.__name__
        if exc_name in ContextIntelligenceErrorMapper.ERROR_TAXONOMY:
            mapped_class = ContextIntelligenceErrorMapper.ERROR_TAXONOMY[exc_name]
            return mapped_class(f"Mapped from {exc_name}: {str(exc)}")
        
        # Default to generic provider error
        return ProviderError(f"Unmapped exception {exc_type}: {str(exc)}")
    
    @staticmethod
    async def emit_error_event(
        context_stream: UnifiedContextStream,
        error: ContextIntelligenceError,
        operation: str,
        correlation_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Emit error event to context stream with PII scrubbing"""
        
        # Build event details with PII scrubbing
        details = {
            "error_type": error.__class__.__name__,
            "operation": operation,
            "error_message": str(error)[:200],  # Truncate for safety
        }
        
        if additional_context:
            # Scrub PII from additional context
            scrubbed_context = ContextIntelligenceErrorMapper._scrub_pii(additional_context)
            details.update(scrubbed_context)
        
        await context_stream.emit_event(
            event_type=ContextEventType.ERROR_OCCURRED,
            details=details,
            correlation_id=correlation_id,
            trace_id=trace_id
        )
        
        # Log without secrets
        logger.error(f"Context Intelligence Error in {operation}: {error.__class__.__name__}: {str(error)[:200]}")
    
    @staticmethod
    def _scrub_pii(data: Dict[str, Any]) -> Dict[str, Any]:
        """Scrub PII from context data (Red Team Amendment #11)"""
        scrubbed = {}
        
        # Safe keys that don't contain PII
        safe_keys = {
            'operation_type', 'cache_level', 'provider_name', 'query_hash', 
            'context_count', 'processing_time', 'cache_hit', 'error_code'
        }
        
        for key, value in data.items():
            if key in safe_keys:
                scrubbed[key] = value
            elif isinstance(value, (int, float, bool)):
                scrubbed[key] = value
            elif isinstance(value, str) and len(value) < 50:
                # Only include short strings that are likely safe
                scrubbed[key] = value[:50]
            else:
                scrubbed[f"{key}_redacted"] = f"<redacted {type(value).__name__}>"
        
        return scrubbed

# Circuit breaker for provider resilience (Red Team Amendment #8)
class CircuitBreaker:
    """Circuit breaker for provider resilience"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def can_execute(self) -> bool:
        """Check if operation can execute based on circuit state"""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if self.last_failure_time and \
               (datetime.now() - self.last_failure_time).seconds >= self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """Record successful operation"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def record_failure(self):
        """Record failed operation"""
        from datetime import datetime
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"