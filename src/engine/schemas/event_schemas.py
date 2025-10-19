"""
METIS Event Schemas - CloudEvents v1.0 Compliant
P4.3: Standardized event schemas for all METIS operations

Defines CloudEvents v1.0 compliant schemas for all METIS system operations
including workflow, cognitive processing, performance monitoring, and system events.
"""

from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass, field
from src.core.enhanced_event_bus import create_metis_cloud_event, CloudEvent


# Event Type Categories following METIS domain patterns
class MetisEventCategory(str, Enum):
    """METIS event categories for systematic organization"""

    WORKFLOW = "workflow"
    ENGAGEMENT = "engagement"
    COGNITIVE = "cognitive"
    PERFORMANCE = "performance"
    MONITORING = "monitoring"
    AGENT = "agent"
    SYSTEM = "system"
    AUDIT = "audit"
    ANALYTICS = "analytics"


class MetisEventPriority(str, Enum):
    """Event priority levels for routing and processing"""

    CRITICAL = "critical"  # System failures, critical alerts
    HIGH = "high"  # Important workflow/engagement events
    MEDIUM = "medium"  # Standard operational events
    LOW = "low"  # Background/audit events


# Workflow Event Types
class WorkflowEventType(str, Enum):
    """Workflow orchestration event types"""

    NODE_STARTED = "workflow.node.started"
    NODE_COMPLETED = "workflow.node.completed"
    NODE_FAILED = "workflow.node.failed"
    PHASE_TRANSITION = "workflow.phase.transition"
    DEPENDENCY_RESOLVED = "workflow.dependency.resolved"
    EXECUTION_TIMEOUT = "workflow.execution.timeout"


# Engagement Event Types
class EngagementEventType(str, Enum):
    """Engagement lifecycle event types"""

    WORKFLOW_STARTED = "engagement.workflow.started"
    WORKFLOW_COMPLETED = "engagement.workflow.completed"
    WORKFLOW_FAILED = "engagement.workflow.failed"
    PROBLEM_STRUCTURED = "engagement.problem.structured"
    HYPOTHESES_GENERATED = "engagement.hypotheses.generated"
    ANALYSIS_EXECUTED = "engagement.analysis.executed"
    SYNTHESIS_DELIVERED = "engagement.synthesis.delivered"
    DELIVERABLE_READY = "engagement.deliverable.ready"
    CLIENT_INTERACTION = "engagement.client.interaction"


# Cognitive Engine Event Types
class CognitiveEventType(str, Enum):
    """Cognitive processing event types"""

    MODEL_SELECTED = "cognitive.model.selected"
    MODEL_APPLIED = "cognitive.model.applied"
    MODEL_VALIDATED = "cognitive.model.validated"
    MODELS_SELECTED = "cognitive.models.selected"
    PIPELINE_COMPLETED = "cognitive.pipeline.completed"
    MENTAL_MODEL_EXECUTION = "cognitive.mental_model.execution"
    REASONING_STEP = "cognitive.reasoning.step"
    INSIGHT_GENERATED = "cognitive.insight.generated"
    REVALIDATION_NEEDED = "cognitive.model.revalidation_needed"
    ANALYSIS_COMPLETED = "cognitive.analysis.completed"


# Performance Event Types
class PerformanceEventType(str, Enum):
    """Performance monitoring event types"""

    MEASUREMENT_RECORDED = "performance.measurement.recorded"
    ALERT_TRIGGERED = "performance.alert.triggered"
    ALERT_RESOLVED = "performance.alert.resolved"
    HEALTH_REPORT = "performance.health.report"
    THRESHOLD_EXCEEDED = "performance.threshold.exceeded"
    SLA_VIOLATION = "performance.sla.violation"
    OPTIMIZATION_APPLIED = "performance.optimization.applied"


# Monitoring Event Types
class MonitoringEventType(str, Enum):
    """System monitoring event types"""

    RULE_TRIGGERED = "monitoring.rule.triggered"
    METRIC_COLLECTED = "monitoring.metric.collected"
    HEALTH_CHECK = "monitoring.health.check"
    RESOURCE_USAGE = "monitoring.resource.usage"
    ERROR_DETECTED = "monitoring.error.detected"
    ANOMALY_DETECTED = "monitoring.anomaly.detected"
    ALERT_ESCALATION = "monitoring.alert.escalation"


# Agent Event Types
class AgentEventType(str, Enum):
    """Agent and worker event types"""

    TASK_STARTED = "agent.task.started"
    TASK_COMPLETED = "agent.task.completed"
    TASK_FAILED = "agent.task.failed"
    AGENT_INITIALIZED = "agent.initialized"
    AGENT_STATUS_CHANGED = "agent.status.changed"
    COLLABORATION_STARTED = "agent.collaboration.started"
    COORDINATION_EVENT = "agent.coordination.event"


# System Event Types
class SystemEventType(str, Enum):
    """System-level event types"""

    STARTUP = "system.startup"
    SHUTDOWN = "system.shutdown"
    CONFIGURATION_CHANGED = "system.configuration.changed"
    COMPONENT_READY = "system.component.ready"
    INTEGRATION_SUCCESS = "system.integration.success"
    INTEGRATION_FAILURE = "system.integration.failure"
    BACKUP_COMPLETED = "system.backup.completed"
    DEPLOYMENT_COMPLETED = "system.deployment.completed"


@dataclass
class EventSchema:
    """Base event schema with CloudEvents v1.0 compliance"""

    event_type: str
    category: MetisEventCategory
    priority: MetisEventPriority
    source: str
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    data_schema: Optional[Dict[str, Any]] = None
    description: str = ""

    def create_event(
        self,
        data: Dict[str, Any],
        engagement_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        subject: Optional[str] = None,
    ) -> CloudEvent:
        """Create CloudEvents v1.0 compliant event with schema validation"""
        # Validate required fields
        missing_fields = [field for field in self.required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        return create_metis_cloud_event(
            event_type=self.event_type,
            source=self.source,
            data=data,
            category=self.category.value,
            priority=self.priority.value,
            engagement_id=engagement_id,
            tenant_id=tenant_id,
            correlation_id=correlation_id,
            subject=subject,
        )


# Define all METIS event schemas
METIS_EVENT_SCHEMAS: Dict[str, EventSchema] = {
    # Workflow Events
    WorkflowEventType.NODE_STARTED: EventSchema(
        event_type=WorkflowEventType.NODE_STARTED,
        category=MetisEventCategory.WORKFLOW,
        priority=MetisEventPriority.MEDIUM,
        source="metis/workflow/engine",
        required_fields=["node_id", "phase", "engagement_id", "timestamp"],
        optional_fields=["dependencies", "estimated_duration"],
        description="Workflow node execution started",
    ),
    WorkflowEventType.NODE_COMPLETED: EventSchema(
        event_type=WorkflowEventType.NODE_COMPLETED,
        category=MetisEventCategory.WORKFLOW,
        priority=MetisEventPriority.MEDIUM,
        source="metis/workflow/engine",
        required_fields=["node_id", "phase", "engagement_id", "execution_time"],
        optional_fields=["result_summary", "artifacts_created"],
        description="Workflow node execution completed successfully",
    ),
    WorkflowEventType.NODE_FAILED: EventSchema(
        event_type=WorkflowEventType.NODE_FAILED,
        category=MetisEventCategory.WORKFLOW,
        priority=MetisEventPriority.HIGH,
        source="metis/workflow/engine",
        required_fields=["node_id", "phase", "engagement_id", "error", "timestamp"],
        optional_fields=["retry_count", "stack_trace"],
        description="Workflow node execution failed",
    ),
    # Engagement Events
    EngagementEventType.WORKFLOW_STARTED: EventSchema(
        event_type=EngagementEventType.WORKFLOW_STARTED,
        category=MetisEventCategory.ENGAGEMENT,
        priority=MetisEventPriority.HIGH,
        source="metis/engagement/orchestrator",
        required_fields=["engagement_id", "problem_statement", "timestamp"],
        optional_fields=["client_name", "estimated_duration", "workflow_config"],
        description="Engagement workflow initiated",
    ),
    EngagementEventType.WORKFLOW_COMPLETED: EventSchema(
        event_type=EngagementEventType.WORKFLOW_COMPLETED,
        category=MetisEventCategory.ENGAGEMENT,
        priority=MetisEventPriority.HIGH,
        source="metis/engagement/orchestrator",
        required_fields=["engagement_id", "execution_time", "phases_completed"],
        optional_fields=[
            "deliverable_artifacts_count",
            "quality_metrics",
            "client_satisfaction",
        ],
        description="Engagement workflow completed successfully",
    ),
    EngagementEventType.WORKFLOW_FAILED: EventSchema(
        event_type=EngagementEventType.WORKFLOW_FAILED,
        category=MetisEventCategory.ENGAGEMENT,
        priority=MetisEventPriority.CRITICAL,
        source="metis/engagement/orchestrator",
        required_fields=["engagement_id", "error", "timestamp"],
        optional_fields=["failed_phase", "partial_results", "recovery_options"],
        description="Engagement workflow failed",
    ),
    EngagementEventType.DELIVERABLE_READY: EventSchema(
        event_type=EngagementEventType.DELIVERABLE_READY,
        category=MetisEventCategory.ENGAGEMENT,
        priority=MetisEventPriority.HIGH,
        source="metis/engagement/synthesis",
        required_fields=["engagement_id", "deliverable_type", "confidence_level"],
        optional_fields=[
            "quality_score",
            "recommendations_count",
            "supporting_evidence",
        ],
        description="Final deliverable ready for client",
    ),
    # Cognitive Events
    CognitiveEventType.MODEL_SELECTED: EventSchema(
        event_type=CognitiveEventType.MODEL_SELECTED,
        category=MetisEventCategory.COGNITIVE,
        priority=MetisEventPriority.MEDIUM,
        source="metis/cognitive/engine",
        required_fields=["model_id", "selection_criteria", "confidence_score"],
        optional_fields=["alternatives_considered", "selection_rationale"],
        description="Mental model selected for analysis",
    ),
    CognitiveEventType.MODEL_APPLIED: EventSchema(
        event_type=CognitiveEventType.MODEL_APPLIED,
        category=MetisEventCategory.COGNITIVE,
        priority=MetisEventPriority.MEDIUM,
        source="metis/cognitive/engine",
        required_fields=["model_id", "context", "result"],
        optional_fields=["execution_time", "confidence_level", "validation_status"],
        description="Mental model applied to context",
    ),
    CognitiveEventType.PIPELINE_COMPLETED: EventSchema(
        event_type=CognitiveEventType.PIPELINE_COMPLETED,
        category=MetisEventCategory.COGNITIVE,
        priority=MetisEventPriority.HIGH,
        source="metis/cognitive/pipeline",
        required_fields=["pipeline_id", "models_applied", "result"],
        optional_fields=[
            "total_execution_time",
            "quality_metrics",
            "insights_generated",
        ],
        description="Cognitive processing pipeline completed",
    ),
    CognitiveEventType.ANALYSIS_COMPLETED: EventSchema(
        event_type=CognitiveEventType.ANALYSIS_COMPLETED,
        category=MetisEventCategory.COGNITIVE,
        priority=MetisEventPriority.HIGH,
        source="metis/cognitive/analysis",
        required_fields=["analysis_id", "analysis_type", "result"],
        optional_fields=["insights", "recommendations", "confidence_metrics"],
        description="Cognitive analysis completed",
    ),
    # Performance Events
    PerformanceEventType.MEASUREMENT_RECORDED: EventSchema(
        event_type=PerformanceEventType.MEASUREMENT_RECORDED,
        category=MetisEventCategory.PERFORMANCE,
        priority=MetisEventPriority.LOW,
        source="metis/performance/validator",
        required_fields=["metric_type", "value", "timestamp"],
        optional_fields=["baseline_value", "trend", "context"],
        description="Performance measurement recorded",
    ),
    PerformanceEventType.ALERT_TRIGGERED: EventSchema(
        event_type=PerformanceEventType.ALERT_TRIGGERED,
        category=MetisEventCategory.PERFORMANCE,
        priority=MetisEventPriority.HIGH,
        source="metis/performance/monitor",
        required_fields=["alert_type", "metric", "current_value", "threshold"],
        optional_fields=["severity", "recommended_actions", "impact_assessment"],
        description="Performance alert triggered",
    ),
    PerformanceEventType.HEALTH_REPORT: EventSchema(
        event_type=PerformanceEventType.HEALTH_REPORT,
        category=MetisEventCategory.PERFORMANCE,
        priority=MetisEventPriority.MEDIUM,
        source="metis/performance/health",
        required_fields=["overall_health", "component_statuses", "timestamp"],
        optional_fields=["recommendations", "trends", "risk_assessment"],
        description="System health report generated",
    ),
    # Monitoring Events
    MonitoringEventType.RULE_TRIGGERED: EventSchema(
        event_type=MonitoringEventType.RULE_TRIGGERED,
        category=MetisEventCategory.MONITORING,
        priority=MetisEventPriority.MEDIUM,
        source="metis/monitoring/rules",
        required_fields=["rule_id", "condition", "trigger_value"],
        optional_fields=["rule_description", "actions_taken", "escalation_level"],
        description="Monitoring rule triggered",
    ),
    MonitoringEventType.ANOMALY_DETECTED: EventSchema(
        event_type=MonitoringEventType.ANOMALY_DETECTED,
        category=MetisEventCategory.MONITORING,
        priority=MetisEventPriority.HIGH,
        source="metis/monitoring/anomaly",
        required_fields=["anomaly_type", "detected_value", "expected_range"],
        optional_fields=["confidence", "impact_assessment", "root_cause_hints"],
        description="System anomaly detected",
    ),
    # Agent Events
    AgentEventType.TASK_COMPLETED: EventSchema(
        event_type=AgentEventType.TASK_COMPLETED,
        category=MetisEventCategory.AGENT,
        priority=MetisEventPriority.MEDIUM,
        source="metis/agent/worker",
        required_fields=["agent_id", "task_id", "result", "execution_time"],
        optional_fields=["confidence", "resources_used", "next_actions"],
        description="Agent task completed successfully",
    ),
    AgentEventType.COLLABORATION_STARTED: EventSchema(
        event_type=AgentEventType.COLLABORATION_STARTED,
        category=MetisEventCategory.AGENT,
        priority=MetisEventPriority.MEDIUM,
        source="metis/agent/coordination",
        required_fields=["collaboration_id", "participating_agents", "objective"],
        optional_fields=["coordination_strategy", "expected_duration"],
        description="Multi-agent collaboration initiated",
    ),
    # System Events
    SystemEventType.COMPONENT_READY: EventSchema(
        event_type=SystemEventType.COMPONENT_READY,
        category=MetisEventCategory.SYSTEM,
        priority=MetisEventPriority.MEDIUM,
        source="metis/system/lifecycle",
        required_fields=["component_name", "status", "timestamp"],
        optional_fields=["dependencies_satisfied", "configuration", "health_check"],
        description="System component ready for operation",
    ),
    SystemEventType.INTEGRATION_SUCCESS: EventSchema(
        event_type=SystemEventType.INTEGRATION_SUCCESS,
        category=MetisEventCategory.SYSTEM,
        priority=MetisEventPriority.MEDIUM,
        source="metis/system/integration",
        required_fields=["integration_type", "external_system", "timestamp"],
        optional_fields=["data_exchanged", "performance_metrics", "validation_results"],
        description="External system integration successful",
    ),
    SystemEventType.INTEGRATION_FAILURE: EventSchema(
        event_type=SystemEventType.INTEGRATION_FAILURE,
        category=MetisEventCategory.SYSTEM,
        priority=MetisEventPriority.HIGH,
        source="metis/system/integration",
        required_fields=["integration_type", "external_system", "error", "timestamp"],
        optional_fields=["retry_attempts", "fallback_used", "impact_assessment"],
        description="External system integration failed",
    ),
}


def get_event_schema(event_type: str) -> Optional[EventSchema]:
    """Get event schema by event type"""
    return METIS_EVENT_SCHEMAS.get(event_type)


def create_standard_event(
    event_type: str,
    data: Dict[str, Any],
    engagement_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    subject: Optional[str] = None,
) -> CloudEvent:
    """Create a standardized METIS event using predefined schemas"""
    schema = get_event_schema(event_type)
    if not schema:
        raise ValueError(f"Unknown event type: {event_type}")

    return schema.create_event(
        data=data,
        engagement_id=engagement_id,
        tenant_id=tenant_id,
        correlation_id=correlation_id,
        subject=subject,
    )


def validate_event_data(event_type: str, data: Dict[str, Any]) -> List[str]:
    """Validate event data against schema requirements"""
    schema = get_event_schema(event_type)
    if not schema:
        return [f"Unknown event type: {event_type}"]

    errors = []

    # Check required fields
    missing_fields = [field for field in schema.required_fields if field not in data]
    if missing_fields:
        errors.append(f"Missing required fields: {missing_fields}")

    # Additional validation could be added here based on data_schema

    return errors


def get_all_event_types() -> Dict[str, List[str]]:
    """Get all available event types organized by category"""
    categories = {}
    for event_type, schema in METIS_EVENT_SCHEMAS.items():
        category = schema.category.value
        if category not in categories:
            categories[category] = []
        categories[category].append(event_type)

    return categories


def generate_event_documentation() -> str:
    """Generate documentation for all METIS event schemas"""
    doc = "# METIS Event Schemas Documentation\n\n"
    doc += "CloudEvents v1.0 compliant event schemas for all METIS operations.\n\n"

    categories = get_all_event_types()

    for category, event_types in categories.items():
        doc += f"## {category.title()} Events\n\n"

        for event_type in event_types:
            schema = METIS_EVENT_SCHEMAS[event_type]
            doc += f"### {event_type}\n\n"
            doc += f"**Description:** {schema.description}\n\n"
            doc += f"**Source:** `{schema.source}`\n\n"
            doc += f"**Priority:** {schema.priority.value}\n\n"

            if schema.required_fields:
                doc += "**Required Fields:**\n"
                for field in schema.required_fields:
                    doc += f"- `{field}`\n"
                doc += "\n"

            if schema.optional_fields:
                doc += "**Optional Fields:**\n"
                for field in schema.optional_fields:
                    doc += f"- `{field}`\n"
                doc += "\n"

            doc += "---\n\n"

    return doc


# Export the main schemas and utilities
__all__ = [
    "METIS_EVENT_SCHEMAS",
    "EventSchema",
    "MetisEventCategory",
    "MetisEventPriority",
    "WorkflowEventType",
    "EngagementEventType",
    "CognitiveEventType",
    "PerformanceEventType",
    "MonitoringEventType",
    "AgentEventType",
    "SystemEventType",
    "get_event_schema",
    "create_standard_event",
    "validate_event_data",
    "get_all_event_types",
    "generate_event_documentation",
]
