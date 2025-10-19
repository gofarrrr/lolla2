"""
METIS Structured Logging System
Sprint: Clarity & Consolidation (Week 1, Day 3-5)
Purpose: Implement structured, correlated logging with OpenTelemetry

This module provides structured JSON logging with automatic correlation,
distributed tracing, and cognitive exhaust capture.
"""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, Optional, Union
from uuid import UUID

import structlog

# OpenTelemetry imports (optional)
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    trace = None
    OTLPSpanExporter = None
    LoggingInstrumentor = None
    Resource = None
    TracerProvider = None
    BatchSpanProcessor = None
    ConsoleSpanExporter = None
    OPENTELEMETRY_AVAILABLE = False
    Status = None
    StatusCode = None

try:
    from opentelemetry.trace import Status, StatusCode
except ImportError:
    pass  # Already handled above

from structlog.processors import JSONRenderer, TimeStamper, add_log_level

from src.config import get_settings

settings = get_settings()

# Context variables for correlation
engagement_id_var: ContextVar[Optional[str]] = ContextVar("engagement_id", default=None)
trace_id_var: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
span_id_var: ContextVar[Optional[str]] = ContextVar("span_id", default=None)
component_var: ContextVar[Optional[str]] = ContextVar("component", default=None)


class EngagementContextProcessor:
    """
    Processor to automatically add engagement context to all log messages.
    Ensures every log can be correlated to a specific engagement.
    """

    def __call__(self, logger, method_name, event_dict):
        """Add engagement context to log event"""
        # Add engagement_id if available
        engagement_id = engagement_id_var.get()
        if engagement_id:
            event_dict["engagement_id"] = str(engagement_id)

        # Add trace context from OpenTelemetry (if available)
        if OPENTELEMETRY_AVAILABLE and trace:
            span = trace.get_current_span()
            if span and span.is_recording():
                span_context = span.get_span_context()
                event_dict["trace_id"] = format(span_context.trace_id, "032x")
                event_dict["span_id"] = format(span_context.span_id, "016x")

                # Store in context vars for access
                trace_id_var.set(event_dict["trace_id"])
                span_id_var.set(event_dict["span_id"])
        else:
            # Use stored values if available
            trace_id = trace_id_var.get()
            span_id = span_id_var.get()
            if trace_id:
                event_dict["trace_id"] = trace_id
            if span_id:
                event_dict["span_id"] = span_id

        # Add component context
        component = component_var.get()
        if component:
            event_dict["component"] = component

        return event_dict


class CognitiveExhaustProcessor:
    """
    Processor to structure cognitive exhaust (thinking processes, devil's advocate, etc.)
    as JSON rather than plain text blocks.
    """

    def __call__(self, logger, method_name, event_dict):
        """Structure cognitive exhaust fields"""
        # Check for cognitive exhaust indicators
        cognitive_fields = [
            "thinking_process",
            "devil_advocate_critique",
            "confidence_evolution",
            "mental_models_applied",
            "reasoning_steps",
            "assumptions_challenged",
        ]

        for field in cognitive_fields:
            if field in event_dict:
                value = event_dict[field]
                # If it's a string that looks like it contains structured data, try to parse it
                if isinstance(value, str):
                    # Check for <thinking> tags or similar patterns
                    if "<thinking>" in value:
                        # Extract thinking content
                        import re

                        thinking_match = re.search(
                            r"<thinking>(.*?)</thinking>", value, re.DOTALL
                        )
                        if thinking_match:
                            event_dict[field] = {
                                "type": "thinking_block",
                                "content": thinking_match.group(1).strip(),
                                "extracted_at": datetime.utcnow().isoformat(),
                            }
                    elif value.startswith("{") or value.startswith("["):
                        # Try to parse as JSON
                        try:
                            event_dict[field] = json.loads(value)
                        except json.JSONDecodeError:
                            # Keep as string if not valid JSON
                            pass

        return event_dict


class PerformanceMetricsProcessor:
    """
    Processor to add performance metrics to relevant log messages.
    """

    def __call__(self, logger, method_name, event_dict):
        """Add performance metrics"""
        # Add timestamp if not present
        if "timestamp" not in event_dict:
            event_dict["timestamp"] = datetime.utcnow().isoformat()

        # Add log level as severity
        if "level" in event_dict:
            event_dict["severity"] = event_dict["level"].upper()

        # Add performance context if available
        if OPENTELEMETRY_AVAILABLE and trace:
            span = trace.get_current_span()
            if span and span.is_recording():
                # Add span attributes as metrics
                attributes = span.attributes or {}
                if attributes:
                    event_dict["span_attributes"] = dict(attributes)

        return event_dict


class MetisLogger:
    """
    Main logger class for METIS platform.
    Provides structured logging with automatic correlation and tracing.
    """

    def __init__(self, name: str = None, component: str = None):
        """
        Initialize a MetisLogger.

        Args:
            name: Logger name (typically __name__)
            component: Component identifier for correlation
        """
        self.name = name or "metis"
        self.component = component
        if component:
            component_var.set(component)

        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                add_log_level,
                TimeStamper(fmt="iso"),
                EngagementContextProcessor(),
                CognitiveExhaustProcessor(),
                PerformanceMetricsProcessor(),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                JSONRenderer(sort_keys=True),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        # Get the logger
        self.logger = structlog.get_logger(name)

        # Initialize tracer (if OpenTelemetry is available)
        self.tracer = (
            trace.get_tracer(name or __name__)
            if OPENTELEMETRY_AVAILABLE and trace
            else None
        )

    def with_engagement(self, engagement_id: Union[str, UUID]) -> "MetisLogger":
        """
        Create a logger bound to a specific engagement.

        Args:
            engagement_id: The engagement ID to bind to

        Returns:
            A new logger instance bound to the engagement
        """
        engagement_id_var.set(str(engagement_id))
        return self

    def with_component(self, component: str) -> "MetisLogger":
        """
        Create a logger bound to a specific component.

        Args:
            component: The component name to bind to

        Returns:
            A new logger instance bound to the component
        """
        component_var.set(component)
        bound_logger = MetisLogger(self.name, component)
        return bound_logger

    def info(self, event: str, **kwargs):
        """Log an info message with structured data"""
        self.logger.info(event, **kwargs)

    def debug(self, event: str, **kwargs):
        """Log a debug message with structured data"""
        self.logger.debug(event, **kwargs)

    def warning(self, event: str, **kwargs):
        """Log a warning message with structured data"""
        self.logger.warning(event, **kwargs)

    def error(self, event: str, **kwargs):
        """Log an error message with structured data"""
        self.logger.error(event, **kwargs)

    def critical(self, event: str, **kwargs):
        """Log a critical message with structured data"""
        self.logger.critical(event, **kwargs)

    def log_cognitive_exhaust(
        self,
        thinking_process: Optional[str] = None,
        devil_advocate: Optional[str] = None,
        confidence_scores: Optional[Dict[str, float]] = None,
        mental_models: Optional[list] = None,
        reasoning_steps: Optional[list] = None,
        **kwargs,
    ):
        """
        Log cognitive exhaust in structured format.

        Args:
            thinking_process: The LLM's thinking process
            devil_advocate: Devil's advocate critique
            confidence_scores: Confidence score evolution
            mental_models: Mental models applied
            reasoning_steps: Step-by-step reasoning
            **kwargs: Additional structured data
        """
        event_data = {"event": "cognitive_exhaust", "cognitive_data": {}}

        if thinking_process:
            event_data["cognitive_data"]["thinking_process"] = thinking_process
        if devil_advocate:
            event_data["cognitive_data"]["devil_advocate_critique"] = devil_advocate
        if confidence_scores:
            event_data["cognitive_data"]["confidence_evolution"] = confidence_scores
        if mental_models:
            event_data["cognitive_data"]["mental_models_applied"] = mental_models
        if reasoning_steps:
            event_data["cognitive_data"]["reasoning_steps"] = reasoning_steps

        # Add any additional data
        event_data.update(kwargs)

        self.logger.info(**event_data)

    def span(self, name: str, attributes: Dict[str, Any] = None):
        """
        Create a traced span for distributed tracing.

        Args:
            name: Span name
            attributes: Span attributes

        Returns:
            OpenTelemetry span context manager
        """
        return self.tracer.start_as_current_span(name, attributes=attributes)

    def log_phase_transition(
        self,
        from_phase: str,
        to_phase: str,
        reason: str = None,
        metrics: Dict[str, Any] = None,
    ):
        """
        Log a phase transition in structured format.

        Args:
            from_phase: Source phase
            to_phase: Target phase
            reason: Reason for transition
            metrics: Performance metrics
        """
        self.info(
            "phase_transition",
            from_phase=from_phase,
            to_phase=to_phase,
            reason=reason,
            metrics=metrics or {},
            transition_time=datetime.utcnow().isoformat(),
        )

    def log_api_call(
        self,
        service: str,
        endpoint: str,
        method: str,
        status_code: int = None,
        duration_ms: float = None,
        error: str = None,
        **kwargs,
    ):
        """
        Log an API call in structured format.

        Args:
            service: Service name (e.g., "perplexity", "claude")
            endpoint: API endpoint
            method: HTTP method
            status_code: Response status code
            duration_ms: Call duration in milliseconds
            error: Error message if failed
            **kwargs: Additional metadata
        """
        event_data = {
            "event": "api_call",
            "service": service,
            "endpoint": endpoint,
            "method": method,
        }

        if status_code:
            event_data["status_code"] = status_code
        if duration_ms:
            event_data["duration_ms"] = duration_ms
        if error:
            event_data["error"] = error

        event_data.update(kwargs)

        if error:
            self.error(**event_data)
        else:
            self.info(**event_data)


def initialize_telemetry(
    service_name: str = "metis-cognitive-platform",
    otlp_endpoint: str = None,
    enable_console_export: bool = False,
):
    """
    Initialize OpenTelemetry with OTLP export (if available).

    Args:
        service_name: Name of the service
        otlp_endpoint: OTLP collector endpoint (e.g., "localhost:4317")
        enable_console_export: Whether to also export to console

    Returns:
        Configured TracerProvider or None if OpenTelemetry not available
    """
    if not OPENTELEMETRY_AVAILABLE:
        logging.getLogger(__name__).debug("OpenTelemetry not available - skipping telemetry initialization")
        return None

    # Create resource
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": "2.0.0",
            "deployment.environment": settings.ENVIRONMENT,
        }
    )

    # Create tracer provider
    tracer_provider = TracerProvider(resource=resource)

    # Add OTLP exporter if endpoint provided
    if otlp_endpoint:
        otlp_exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint, insecure=True  # Use insecure for local development
        )
        tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    # Add console exporter if requested
    if enable_console_export or settings.DEBUG_MODE:
        console_exporter = ConsoleSpanExporter()
        tracer_provider.add_span_processor(BatchSpanProcessor(console_exporter))

    # Set as global tracer provider
    trace.set_tracer_provider(tracer_provider)

    # Instrument logging to add trace context
    if LoggingInstrumentor:
        LoggingInstrumentor().instrument()

    return tracer_provider


def get_logger(name: str = None, component: str = None) -> MetisLogger:
    """
    Get a configured MetisLogger instance.

    Args:
        name: Logger name (typically __name__)
        component: Component identifier

    Returns:
        Configured MetisLogger
    """
    return MetisLogger(name, component)


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


class LoggingContext:
    """
    Context manager for scoped logging context.
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
            span_name: OpenTelemetry span name
            span_attributes: Span attributes
        """
        self.engagement_id = str(engagement_id) if engagement_id else None
        self.component = component
        self.span_name = span_name
        self.span_attributes = span_attributes or {}
        self.span = None
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

        # Start span if requested (and OpenTelemetry is available)
        if self.span_name and OPENTELEMETRY_AVAILABLE and trace:
            tracer = trace.get_tracer(__name__)
            self.span = tracer.start_as_current_span(
                self.span_name, attributes=self.span_attributes
            ).__enter__()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context"""
        # End span if created
        if self.span:
            if exc_type:
                self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
            self.span.__exit__(exc_type, exc_val, exc_tb)

        # Restore previous values
        if self.previous_engagement is not None:
            engagement_id_var.set(self.previous_engagement)
        else:
            engagement_id_var.set(None)

        if self.previous_component is not None:
            component_var.set(self.previous_component)
        else:
            component_var.set(None)


# Configure Python's standard logging to work with structlog
logging.basicConfig(
    format="%(message)s",
    stream=sys.stdout,
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
)

# Initialize telemetry on module import
if settings.ENVIRONMENT != "test":
    initialize_telemetry(enable_console_export=settings.DEBUG_MODE)
