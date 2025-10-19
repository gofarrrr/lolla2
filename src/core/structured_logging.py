"""
Simple structured logging compatibility layer for F-03c validation
"""

import logging
from contextvars import ContextVar
from typing import Any, Dict, Union
from uuid import UUID

# Context variables for correlation
engagement_id_var: ContextVar[str] = ContextVar("engagement_id", default=None)
component_var: ContextVar[str] = ContextVar("component", default=None)

def get_logger(name: str, component: str = None):
    """Get logger with optional component"""
    logger = logging.getLogger(name)
    return LoggerWrapper(logger, component)

class LoggerWrapper:
    """Wrapper that provides structured logging methods"""
    
    def __init__(self, logger, component=None):
        self.logger = logger
        self.component = component
    
    def with_component(self, component):
        return LoggerWrapper(self.logger, component)
    
    def info(self, message, **kwargs):
        if kwargs:
            self.logger.info(f"{message} - {kwargs}")
        else:
            self.logger.info(message)
    
    def error(self, message, **kwargs):
        if kwargs:
            self.logger.error(f"{message} - {kwargs}")
        else:
            self.logger.error(message)
    
    def warning(self, message, **kwargs):
        if kwargs:
            self.logger.warning(f"{message} - {kwargs}")
        else:
            self.logger.warning(message)
    
    def debug(self, message, **kwargs):
        if kwargs:
            self.logger.debug(f"{message} - {kwargs}")
        else:
            self.logger.debug(message)


class StructuredLogger(LoggerWrapper):
    """Backwards-compatible structured logger facade."""

    def __init__(self, name: str, component: str = None):
        super().__init__(logging.getLogger(name), component)

    def bind(self, **kwargs):  # for compatibility with structlog-like APIs
        component = kwargs.get("component", self.component)
        return StructuredLogger(self.logger.name, component)


class LoggingContext:
    """
    Simple context manager for scoped logging context.
    Simplified version compatible with the existing logging system.
    """

    def __init__(
        self,
        engagement_id: Union[str, UUID] = None,
        component: str = None,
        span_name: str = None,
        span_attributes: Dict[str, Any] = None,
    ):
        """
        Initialize logging context.

        Args:
            engagement_id: Engagement ID for correlation
            component: Component name  
            span_name: Span name (not used in simplified version)
            span_attributes: Span attributes (not used in simplified version)
        """
        self.engagement_id = str(engagement_id) if engagement_id else None
        self.component = component
        self.span_name = span_name
        self.span_attributes = span_attributes or {}
        self.previous_engagement = None
        self.previous_component = None

    def __enter__(self):
        """Enter the context"""
        # Save previous values
        self.previous_engagement = engagement_id_var.get()
        self.previous_component = component_var.get()

        # Set new values
        if self.engagement_id:
            engagement_id_var.set(self.engagement_id)
        if self.component:
            component_var.set(self.component)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context"""
        # Restore previous values
        if self.previous_engagement is not None:
            engagement_id_var.set(self.previous_engagement)
        else:
            engagement_id_var.set(None)

        if self.previous_component is not None:
            component_var.set(self.previous_component)
        else:
            component_var.set(None)


def set_engagement_context(engagement_id: Union[str, UUID]):
    """
    Set the global engagement context for correlation.

    Args:
        engagement_id: The engagement ID to set
    """
    engagement_id_var.set(str(engagement_id))


def clear_engagement_context():
    """Clear the global engagement context"""
    engagement_id_var.set(None)


__all__ = [
    "get_logger",
    "StructuredLogger",
    "LoggerWrapper",
    "LoggingContext",
    "set_engagement_context",
    "clear_engagement_context",
]
