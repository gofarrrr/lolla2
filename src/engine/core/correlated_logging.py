"""
Week 4.1b: Correlated Logging with Request Tracing
Advanced logging system with complete request correlation and distributed tracing

Features:
- Request-level correlation across all components
- Distributed tracing with OpenTelemetry integration
- Cognitive journey tracking (problem → synthesis)
- LLM interaction tracing with token counting
- Performance bottleneck identification
- Error correlation and debugging context
- Log aggregation and search capabilities
"""

import asyncio
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# OpenTelemetry imports
try:
    from opentelemetry import trace, baggage
    from opentelemetry.trace import Status, StatusCode, Tracer
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.logging import LoggingInstrumentor

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

from structlog.stdlib import BoundLogger

from src.core.structured_logging import get_logger
from src.core.enhanced_telemetry import get_telemetry_collector

# Context variables for request correlation
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
engagement_id_var: ContextVar[Optional[str]] = ContextVar("engagement_id", default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
session_id_var: ContextVar[Optional[str]] = ContextVar("session_id", default=None)
trace_id_var: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
span_id_var: ContextVar[Optional[str]] = ContextVar("span_id", default=None)
component_var: ContextVar[Optional[str]] = ContextVar("component", default=None)
operation_var: ContextVar[Optional[str]] = ContextVar("operation", default=None)

logger = get_logger(__name__, component="correlated_logging")


@dataclass
class RequestContext:
    """Complete request context for correlation"""

    request_id: str
    engagement_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogEvent:
    """Structured log event with full correlation"""

    timestamp: datetime
    level: str
    message: str
    component: str
    operation: Optional[str]
    request_id: Optional[str]
    engagement_id: Optional[str]
    user_id: Optional[str]
    session_id: Optional[str]
    trace_id: Optional[str]
    span_id: Optional[str]
    duration_ms: Optional[float] = None
    status: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "message": self.message,
            "component": self.component,
            "operation": self.operation,
            "request_id": self.request_id,
            "engagement_id": self.engagement_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error_type": self.error_type,
            "error_message": self.error_message,
            **self.metadata,
        }


class RequestCorrelationProcessor:
    """
    Structlog processor that automatically adds request correlation data
    to all log messages within a request context
    """

    def __call__(
        self, logger: BoundLogger, method_name: str, event_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add request correlation to log event"""

        # Add request correlation IDs
        if request_id := request_id_var.get():
            event_dict["request_id"] = request_id

        if engagement_id := engagement_id_var.get():
            event_dict["engagement_id"] = engagement_id

        if user_id := user_id_var.get():
            event_dict["user_id"] = user_id

        if session_id := session_id_var.get():
            event_dict["session_id"] = session_id

        # Add OpenTelemetry trace context
        if OTEL_AVAILABLE:
            span = trace.get_current_span()
            if span and span.is_recording():
                span_context = span.get_span_context()
                event_dict["trace_id"] = format(span_context.trace_id, "032x")
                event_dict["span_id"] = format(span_context.span_id, "016x")

                # Store in context vars for non-OTEL components
                trace_id_var.set(event_dict["trace_id"])
                span_id_var.set(event_dict["span_id"])

        # Add component and operation context
        if component := component_var.get():
            event_dict["component"] = component

        if operation := operation_var.get():
            event_dict["operation"] = operation

        # Add timestamp if not present
        if "timestamp" not in event_dict:
            event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()

        return event_dict


class CognitiveJourneyTracker:
    """
    Tracks the complete cognitive journey from problem statement to final synthesis
    Provides detailed visibility into the 6-phase processing pipeline
    """

    def __init__(self):
        """Initialize journey tracker"""
        self.active_journeys: Dict[str, Dict[str, Any]] = {}
        self.telemetry = get_telemetry_collector()

    def start_journey(
        self,
        engagement_id: str,
        problem_statement: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start tracking a cognitive journey"""

        journey_id = str(uuid.uuid4())

        journey_data = {
            "journey_id": journey_id,
            "engagement_id": engagement_id,
            "user_id": user_id,
            "problem_statement": problem_statement,
            "started_at": datetime.now(timezone.utc),
            "current_phase": "problem_structuring",
            "phases_completed": [],
            "phases_data": {},
            "llm_interactions": [],
            "research_queries": [],
            "devil_advocate_critiques": [],
            "model_selections": [],
            "performance_metrics": {
                "total_duration_ms": 0,
                "llm_tokens_used": 0,
                "llm_cost_usd": 0.0,
                "research_queries_made": 0,
                "models_applied": 0,
            },
            "metadata": metadata or {},
        }

        self.active_journeys[engagement_id] = journey_data

        # Log journey start
        logger.info(
            "🚀 Cognitive journey started",
            engagement_id=engagement_id,
            journey_id=journey_id,
            problem_statement=(
                problem_statement[:100] + "..."
                if len(problem_statement) > 100
                else problem_statement
            ),
        )

        # Record telemetry event
        self.telemetry.record_event(
            event_type="cognitive_journey.started",
            engagement_id=engagement_id,
            component="cognitive_engine",
            operation="start_journey",
            metadata={
                "journey_id": journey_id,
                "problem_length": len(problem_statement),
            },
        )

        return journey_id

    def update_phase(
        self,
        engagement_id: str,
        phase: str,
        status: str = "in_progress",
        duration_ms: Optional[float] = None,
        results: Optional[Dict[str, Any]] = None,
    ):
        """Update current phase status"""

        if engagement_id not in self.active_journeys:
            logger.warning(f"⚠️ No active journey found for engagement {engagement_id}")
            return

        journey = self.active_journeys[engagement_id]

        if status == "completed" and phase not in journey["phases_completed"]:
            journey["phases_completed"].append(phase)
            journey["phases_data"][phase] = {
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "duration_ms": duration_ms,
                "results": results or {},
            }

            logger.info(
                f"✅ Phase completed: {phase}",
                engagement_id=engagement_id,
                phase=phase,
                duration_ms=duration_ms,
                phases_completed=len(journey["phases_completed"]),
            )
        else:
            journey["current_phase"] = phase

            logger.debug(
                f"📍 Phase status update: {phase} → {status}",
                engagement_id=engagement_id,
                phase=phase,
                status=status,
            )

        # Record telemetry
        self.telemetry.record_event(
            event_type=f"cognitive_journey.phase.{status}",
            engagement_id=engagement_id,
            component="cognitive_engine",
            operation=f"phase_{phase}",
            duration_ms=duration_ms,
            status=status,
            metadata={"phase": phase},
        )

    def track_llm_interaction(
        self,
        engagement_id: str,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        duration_ms: float,
        operation: str,
        success: bool = True,
    ):
        """Track LLM API interaction"""

        if engagement_id not in self.active_journeys:
            return

        journey = self.active_journeys[engagement_id]

        interaction = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": cost_usd,
            "duration_ms": duration_ms,
            "operation": operation,
            "success": success,
        }

        journey["llm_interactions"].append(interaction)

        # Update performance metrics
        metrics = journey["performance_metrics"]
        metrics["llm_tokens_used"] += prompt_tokens + completion_tokens
        metrics["llm_cost_usd"] += cost_usd

        logger.info(
            f"🤖 LLM interaction: {provider}/{model}",
            engagement_id=engagement_id,
            provider=provider,
            model=model,
            tokens_used=prompt_tokens + completion_tokens,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
            operation=operation,
            success=success,
        )

        # Record telemetry
        self.telemetry.record_event(
            event_type="llm_interaction",
            engagement_id=engagement_id,
            component="llm_client",
            operation=operation,
            duration_ms=duration_ms,
            status="success" if success else "error",
            metadata={
                "provider": provider,
                "model": model,
                "tokens_used": prompt_tokens + completion_tokens,
                "cost_usd": cost_usd,
            },
        )

    def track_research_query(
        self,
        engagement_id: str,
        query: str,
        source: str,
        results_count: int,
        duration_ms: float,
        success: bool = True,
    ):
        """Track research query execution"""

        if engagement_id not in self.active_journeys:
            return

        journey = self.active_journeys[engagement_id]

        query_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "source": source,
            "results_count": results_count,
            "duration_ms": duration_ms,
            "success": success,
        }

        journey["research_queries"].append(query_data)
        journey["performance_metrics"]["research_queries_made"] += 1

        logger.info(
            f"🔍 Research query: {source}",
            engagement_id=engagement_id,
            query=query[:50] + "..." if len(query) > 50 else query,
            source=source,
            results_count=results_count,
            duration_ms=duration_ms,
            success=success,
        )

    def complete_journey(
        self,
        engagement_id: str,
        status: str = "completed",
        final_results: Optional[Dict[str, Any]] = None,
    ):
        """Complete and finalize a cognitive journey"""

        if engagement_id not in self.active_journeys:
            logger.warning(f"⚠️ No active journey found for engagement {engagement_id}")
            return

        journey = self.active_journeys[engagement_id]

        # Calculate total duration
        started_at = journey["started_at"]
        completed_at = datetime.now(timezone.utc)
        total_duration = (completed_at - started_at).total_seconds() * 1000

        journey["completed_at"] = completed_at.isoformat()
        journey["status"] = status
        journey["final_results"] = final_results or {}
        journey["performance_metrics"]["total_duration_ms"] = total_duration

        # Log journey completion with comprehensive summary
        metrics = journey["performance_metrics"]

        logger.info(
            f"🎉 Cognitive journey completed: {status}",
            engagement_id=engagement_id,
            journey_id=journey["journey_id"],
            status=status,
            total_duration_ms=total_duration,
            phases_completed=len(journey["phases_completed"]),
            llm_tokens_used=metrics["llm_tokens_used"],
            llm_cost_usd=metrics["llm_cost_usd"],
            research_queries_made=metrics["research_queries_made"],
            llm_interactions=len(journey["llm_interactions"]),
        )

        # Record comprehensive telemetry
        self.telemetry.record_event(
            event_type="cognitive_journey.completed",
            engagement_id=engagement_id,
            component="cognitive_engine",
            operation="complete_journey",
            duration_ms=total_duration,
            status=status,
            metadata={
                "journey_id": journey["journey_id"],
                "phases_completed": len(journey["phases_completed"]),
                "performance_metrics": metrics,
            },
        )

        # Archive completed journey (remove from active tracking)
        archived_journey = self.active_journeys.pop(engagement_id)

        # TODO: Store archived journey in database for historical analysis
        return archived_journey

    def get_journey_status(self, engagement_id: str) -> Optional[Dict[str, Any]]:
        """Get current journey status"""
        return self.active_journeys.get(engagement_id)


class CorrelatedLogger:
    """
    Enhanced logger with automatic correlation and tracing capabilities
    Integrates with telemetry system for comprehensive observability
    """

    def __init__(self, name: str, component: str):
        """Initialize correlated logger"""
        self.name = name
        self.component = component
        self.base_logger = get_logger(name, component=component)
        self.journey_tracker = CognitiveJourneyTracker()
        self.telemetry = get_telemetry_collector()

    def _get_correlation_data(self) -> Dict[str, Any]:
        """Get current correlation context"""
        return {
            "request_id": request_id_var.get(),
            "engagement_id": engagement_id_var.get(),
            "user_id": user_id_var.get(),
            "session_id": session_id_var.get(),
            "trace_id": trace_id_var.get(),
            "span_id": span_id_var.get(),
            "component": self.component,
            "operation": operation_var.get(),
        }

    def info(self, message: str, **kwargs):
        """Log info message with correlation"""
        correlation = self._get_correlation_data()
        self.base_logger.info(message, **correlation, **kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug message with correlation"""
        correlation = self._get_correlation_data()
        self.base_logger.debug(message, **correlation, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message with correlation"""
        correlation = self._get_correlation_data()
        self.base_logger.warning(message, **correlation, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message with correlation"""
        correlation = self._get_correlation_data()
        self.base_logger.error(message, **correlation, **kwargs)

        # Record error in telemetry
        self.telemetry.record_event(
            event_type="error",
            engagement_id=correlation.get("engagement_id"),
            component=self.component,
            operation=correlation.get("operation"),
            status="error",
            metadata={"error_message": message, **kwargs},
        )

    def critical(self, message: str, **kwargs):
        """Log critical message with correlation"""
        correlation = self._get_correlation_data()
        self.base_logger.critical(message, **correlation, **kwargs)

        # Record critical error in telemetry
        self.telemetry.record_event(
            event_type="critical_error",
            engagement_id=correlation.get("engagement_id"),
            component=self.component,
            operation=correlation.get("operation"),
            status="critical",
            metadata={"error_message": message, **kwargs},
        )


# Context managers for request correlation


class RequestCorrelationContext:
    """Context manager for request-level correlation"""

    def __init__(
        self,
        request_id: Optional[str] = None,
        engagement_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        component: Optional[str] = None,
        operation: Optional[str] = None,
    ):
        self.request_id = request_id or str(uuid.uuid4())
        self.engagement_id = engagement_id
        self.user_id = user_id
        self.session_id = session_id
        self.component = component
        self.operation = operation

        # Store previous context values for restoration
        self.previous_values = {}

    def __enter__(self):
        """Set correlation context"""
        # Store previous values
        self.previous_values = {
            "request_id": request_id_var.get(),
            "engagement_id": engagement_id_var.get(),
            "user_id": user_id_var.get(),
            "session_id": session_id_var.get(),
            "component": component_var.get(),
            "operation": operation_var.get(),
        }

        # Set new values
        request_id_var.set(self.request_id)
        if self.engagement_id:
            engagement_id_var.set(self.engagement_id)
        if self.user_id:
            user_id_var.set(self.user_id)
        if self.session_id:
            session_id_var.set(self.session_id)
        if self.component:
            component_var.set(self.component)
        if self.operation:
            operation_var.set(self.operation)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore previous correlation context"""
        for var_name, value in self.previous_values.items():
            if var_name == "request_id":
                request_id_var.set(value)
            elif var_name == "engagement_id":
                engagement_id_var.set(value)
            elif var_name == "user_id":
                user_id_var.set(value)
            elif var_name == "session_id":
                session_id_var.set(value)
            elif var_name == "component":
                component_var.set(value)
            elif var_name == "operation":
                operation_var.set(value)


# Global instances
_journey_tracker: Optional[CognitiveJourneyTracker] = None


def get_journey_tracker() -> CognitiveJourneyTracker:
    """Get global cognitive journey tracker"""
    global _journey_tracker
    if _journey_tracker is None:
        _journey_tracker = CognitiveJourneyTracker()
    return _journey_tracker


def get_correlated_logger(name: str, component: str) -> CorrelatedLogger:
    """Get a correlated logger instance"""
    return CorrelatedLogger(name, component)


# Convenience functions for common operations


def start_request_context(
    request_id: Optional[str] = None,
    engagement_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    component: Optional[str] = None,
    operation: Optional[str] = None,
) -> RequestCorrelationContext:
    """Start a new request correlation context"""
    return RequestCorrelationContext(
        request_id=request_id,
        engagement_id=engagement_id,
        user_id=user_id,
        session_id=session_id,
        component=component,
        operation=operation,
    )


def with_correlation(
    engagement_id: Optional[str] = None,
    component: Optional[str] = None,
    operation: Optional[str] = None,
):
    """Decorator for automatic correlation context"""

    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            with start_request_context(
                engagement_id=engagement_id,
                component=component,
                operation=operation or func.__name__,
            ):
                return await func(*args, **kwargs)

        def sync_wrapper(*args, **kwargs):
            with start_request_context(
                engagement_id=engagement_id,
                component=component,
                operation=operation or func.__name__,
            ):
                return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator
