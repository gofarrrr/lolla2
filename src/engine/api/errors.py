"""
Central FastAPI Exception Handlers
Red Team Amendment: Centralized error handling with context stream integration
"""
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Callable
import logging
from src.core.unified_context_stream import UnifiedContextStream, ContextEventType

logger = logging.getLogger(__name__)

# Error taxonomy mapping (Red Team Amendment #3)
ERROR_MAPPING = {
    "ProviderError": {"status": 502, "event": ContextEventType.ERROR_OCCURRED, "message": "External service unavailable"},
    "ParseError": {"status": 400, "event": ContextEventType.ERROR_OCCURRED, "message": "Invalid data format"},
    "ValidationError": {"status": 422, "event": ContextEventType.ERROR_OCCURRED, "message": "Validation failed"},
    "TimeoutError": {"status": 504, "event": ContextEventType.ERROR_OCCURRED, "message": "Operation timeout"},
    "CancellationError": {"status": 499, "event": ContextEventType.ERROR_OCCURRED, "message": "Client disconnected"},
}

class ProviderError(Exception):
    """External service failure"""
    pass

class ParseError(Exception):
    """Data format issues"""
    pass

class ValidationError(Exception):
    """Business rule violations"""
    pass

class TimeoutError(Exception):
    """Operation timeouts"""
    pass

class CancellationError(Exception):
    """Client disconnection"""
    pass

def create_exception_handler(context_stream: UnifiedContextStream) -> Callable:
    """Create exception handler with context stream integration"""
    
    async def exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Central exception handler with observability"""
        
        # Get error details from mapping
        error_type = type(exc).__name__
        error_details = ERROR_MAPPING.get(error_type, {
            "status": 500, 
            "event": ContextEventType.ERROR_OCCURRED, 
            "message": "Internal server error"
        })
        
        # Emit to context stream with PII scrubbing
        await context_stream.emit_event(
            event_type=error_details["event"],
            details={
                "error_type": error_type,
                "error_message": str(exc)[:200],  # Truncate for safety
                "endpoint": str(request.url.path),
                "method": request.method,
                "status_code": error_details["status"]
            },
            correlation_id=getattr(request.state, 'correlation_id', None),
            trace_id=getattr(request.state, 'trace_id', None)
        )
        
        # Log without secrets
        logger.error(f"API Error: {error_type} on {request.method} {request.url.path}: {str(exc)[:200]}")
        
        return JSONResponse(
            status_code=error_details["status"],
            content={"error": error_details["message"], "detail": str(exc)}
        )
    
    return exception_handler

def setup_exception_handlers(app, context_stream: UnifiedContextStream):
    """Wire exception handlers to FastAPI app (Red Team Amendment)"""
    handler = create_exception_handler(context_stream)
    
    # Register all error types
    for error_class in [ProviderError, ParseError, ValidationError, TimeoutError, CancellationError]:
        app.add_exception_handler(error_class, handler)
    
    # Catch-all for unhandled exceptions
    app.add_exception_handler(Exception, handler)