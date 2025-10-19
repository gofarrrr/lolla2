"""
METIS Event Factory Functions - P4.3
Convenience functions for creating standardized METIS CloudEvents v1.0 events

Provides easy-to-use factory functions for the most common METIS event types
with proper schema validation and CloudEvents v1.0 compliance.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from src.core.enhanced_event_bus import CloudEvent
from src.schemas.event_schemas import (
    create_standard_event,
    WorkflowEventType,
    EngagementEventType,
    CognitiveEventType,
    PerformanceEventType,
    AgentEventType,
    SystemEventType,
)


# Workflow Event Factories
def create_workflow_node_started_event(
    node_id: str,
    phase: str,
    engagement_id: str,
    dependencies: Optional[List[str]] = None,
    estimated_duration: Optional[float] = None,
    tenant_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> CloudEvent:
    """Create workflow node started event"""
    data = {
        "node_id": node_id,
        "phase": phase,
        "engagement_id": engagement_id,
        "timestamp": datetime.utcnow().isoformat(),
    }

    if dependencies:
        data["dependencies"] = dependencies
    if estimated_duration:
        data["estimated_duration"] = estimated_duration

    return create_standard_event(
        event_type=WorkflowEventType.NODE_STARTED,
        data=data,
        engagement_id=engagement_id,
        tenant_id=tenant_id,
        correlation_id=correlation_id,
    )


def create_workflow_node_completed_event(
    node_id: str,
    phase: str,
    engagement_id: str,
    execution_time: float,
    result_summary: Optional[str] = None,
    artifacts_created: Optional[List[str]] = None,
    tenant_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> CloudEvent:
    """Create workflow node completed event"""
    data = {
        "node_id": node_id,
        "phase": phase,
        "engagement_id": engagement_id,
        "execution_time": execution_time,
    }

    if result_summary:
        data["result_summary"] = result_summary
    if artifacts_created:
        data["artifacts_created"] = artifacts_created

    return create_standard_event(
        event_type=WorkflowEventType.NODE_COMPLETED,
        data=data,
        engagement_id=engagement_id,
        tenant_id=tenant_id,
        correlation_id=correlation_id,
    )


def create_workflow_node_failed_event(
    node_id: str,
    phase: str,
    engagement_id: str,
    error: str,
    retry_count: Optional[int] = None,
    stack_trace: Optional[str] = None,
    tenant_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> CloudEvent:
    """Create workflow node failed event"""
    data = {
        "node_id": node_id,
        "phase": phase,
        "engagement_id": engagement_id,
        "error": error,
        "timestamp": datetime.utcnow().isoformat(),
    }

    if retry_count is not None:
        data["retry_count"] = retry_count
    if stack_trace:
        data["stack_trace"] = stack_trace

    return create_standard_event(
        event_type=WorkflowEventType.NODE_FAILED,
        data=data,
        engagement_id=engagement_id,
        tenant_id=tenant_id,
        correlation_id=correlation_id,
    )


# Engagement Event Factories
def create_engagement_workflow_started_event(
    engagement_id: str,
    problem_statement: str,
    client_name: Optional[str] = None,
    estimated_duration: Optional[float] = None,
    workflow_config: Optional[Dict[str, Any]] = None,
    tenant_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> CloudEvent:
    """Create engagement workflow started event"""
    data = {
        "engagement_id": engagement_id,
        "problem_statement": problem_statement,
        "timestamp": datetime.utcnow().isoformat(),
    }

    if client_name:
        data["client_name"] = client_name
    if estimated_duration:
        data["estimated_duration"] = estimated_duration
    if workflow_config:
        data["workflow_config"] = workflow_config

    return create_standard_event(
        event_type=EngagementEventType.WORKFLOW_STARTED,
        data=data,
        engagement_id=engagement_id,
        tenant_id=tenant_id,
        correlation_id=correlation_id,
    )


def create_engagement_workflow_completed_event(
    engagement_id: str,
    execution_time: float,
    phases_completed: int,
    deliverable_artifacts_count: Optional[int] = None,
    quality_metrics: Optional[Dict[str, Any]] = None,
    client_satisfaction: Optional[float] = None,
    tenant_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> CloudEvent:
    """Create engagement workflow completed event"""
    data = {
        "engagement_id": engagement_id,
        "execution_time": execution_time,
        "phases_completed": phases_completed,
    }

    if deliverable_artifacts_count is not None:
        data["deliverable_artifacts_count"] = deliverable_artifacts_count
    if quality_metrics:
        data["quality_metrics"] = quality_metrics
    if client_satisfaction is not None:
        data["client_satisfaction"] = client_satisfaction

    return create_standard_event(
        event_type=EngagementEventType.WORKFLOW_COMPLETED,
        data=data,
        engagement_id=engagement_id,
        tenant_id=tenant_id,
        correlation_id=correlation_id,
    )


def create_engagement_workflow_failed_event(
    engagement_id: str,
    error: str,
    failed_phase: Optional[str] = None,
    partial_results: Optional[Dict[str, Any]] = None,
    recovery_options: Optional[List[str]] = None,
    tenant_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> CloudEvent:
    """Create engagement workflow failed event"""
    data = {
        "engagement_id": engagement_id,
        "error": error,
        "timestamp": datetime.utcnow().isoformat(),
    }

    if failed_phase:
        data["failed_phase"] = failed_phase
    if partial_results:
        data["partial_results"] = partial_results
    if recovery_options:
        data["recovery_options"] = recovery_options

    return create_standard_event(
        event_type=EngagementEventType.WORKFLOW_FAILED,
        data=data,
        engagement_id=engagement_id,
        tenant_id=tenant_id,
        correlation_id=correlation_id,
    )


# Cognitive Event Factories
def create_cognitive_model_selected_event(
    model_id: str,
    selection_criteria: Dict[str, Any],
    confidence_score: float,
    alternatives_considered: Optional[List[str]] = None,
    selection_rationale: Optional[str] = None,
    engagement_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> CloudEvent:
    """Create cognitive model selected event"""
    data = {
        "model_id": model_id,
        "selection_criteria": selection_criteria,
        "confidence_score": confidence_score,
    }

    if alternatives_considered:
        data["alternatives_considered"] = alternatives_considered
    if selection_rationale:
        data["selection_rationale"] = selection_rationale

    return create_standard_event(
        event_type=CognitiveEventType.MODEL_SELECTED,
        data=data,
        engagement_id=engagement_id,
        tenant_id=tenant_id,
        correlation_id=correlation_id,
    )


def create_cognitive_analysis_completed_event(
    analysis_id: str,
    analysis_type: str,
    result: Dict[str, Any],
    insights: Optional[List[str]] = None,
    recommendations: Optional[List[str]] = None,
    confidence_metrics: Optional[Dict[str, float]] = None,
    engagement_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> CloudEvent:
    """Create cognitive analysis completed event"""
    data = {
        "analysis_id": analysis_id,
        "analysis_type": analysis_type,
        "result": result,
    }

    if insights:
        data["insights"] = insights
    if recommendations:
        data["recommendations"] = recommendations
    if confidence_metrics:
        data["confidence_metrics"] = confidence_metrics

    return create_standard_event(
        event_type=CognitiveEventType.ANALYSIS_COMPLETED,
        data=data,
        engagement_id=engagement_id,
        tenant_id=tenant_id,
        correlation_id=correlation_id,
    )


# Performance Event Factories
def create_performance_measurement_event(
    metric_type: str,
    value: float,
    baseline_value: Optional[float] = None,
    trend: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    engagement_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> CloudEvent:
    """Create performance measurement recorded event"""
    data = {
        "metric_type": metric_type,
        "value": value,
        "timestamp": datetime.utcnow().isoformat(),
    }

    if baseline_value is not None:
        data["baseline_value"] = baseline_value
    if trend:
        data["trend"] = trend
    if context:
        data["context"] = context

    return create_standard_event(
        event_type=PerformanceEventType.MEASUREMENT_RECORDED,
        data=data,
        engagement_id=engagement_id,
        tenant_id=tenant_id,
        correlation_id=correlation_id,
    )


def create_performance_alert_event(
    alert_type: str,
    metric: str,
    current_value: float,
    threshold: float,
    severity: Optional[str] = None,
    recommended_actions: Optional[List[str]] = None,
    impact_assessment: Optional[str] = None,
    engagement_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> CloudEvent:
    """Create performance alert triggered event"""
    data = {
        "alert_type": alert_type,
        "metric": metric,
        "current_value": current_value,
        "threshold": threshold,
    }

    if severity:
        data["severity"] = severity
    if recommended_actions:
        data["recommended_actions"] = recommended_actions
    if impact_assessment:
        data["impact_assessment"] = impact_assessment

    return create_standard_event(
        event_type=PerformanceEventType.ALERT_TRIGGERED,
        data=data,
        engagement_id=engagement_id,
        tenant_id=tenant_id,
        correlation_id=correlation_id,
    )


# Agent Event Factories
def create_agent_task_completed_event(
    agent_id: str,
    task_id: str,
    result: Dict[str, Any],
    execution_time: float,
    confidence: Optional[float] = None,
    resources_used: Optional[Dict[str, Any]] = None,
    next_actions: Optional[List[str]] = None,
    engagement_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> CloudEvent:
    """Create agent task completed event"""
    data = {
        "agent_id": agent_id,
        "task_id": task_id,
        "result": result,
        "execution_time": execution_time,
    }

    if confidence is not None:
        data["confidence"] = confidence
    if resources_used:
        data["resources_used"] = resources_used
    if next_actions:
        data["next_actions"] = next_actions

    return create_standard_event(
        event_type=AgentEventType.TASK_COMPLETED,
        data=data,
        engagement_id=engagement_id,
        tenant_id=tenant_id,
        correlation_id=correlation_id,
    )


# System Event Factories
def create_system_component_ready_event(
    component_name: str,
    status: str,
    dependencies_satisfied: Optional[List[str]] = None,
    configuration: Optional[Dict[str, Any]] = None,
    health_check: Optional[Dict[str, Any]] = None,
    tenant_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> CloudEvent:
    """Create system component ready event"""
    data = {
        "component_name": component_name,
        "status": status,
        "timestamp": datetime.utcnow().isoformat(),
    }

    if dependencies_satisfied:
        data["dependencies_satisfied"] = dependencies_satisfied
    if configuration:
        data["configuration"] = configuration
    if health_check:
        data["health_check"] = health_check

    return create_standard_event(
        event_type=SystemEventType.COMPONENT_READY,
        data=data,
        tenant_id=tenant_id,
        correlation_id=correlation_id,
    )


# Export all factory functions
__all__ = [
    "create_workflow_node_started_event",
    "create_workflow_node_completed_event",
    "create_workflow_node_failed_event",
    "create_engagement_workflow_started_event",
    "create_engagement_workflow_completed_event",
    "create_engagement_workflow_failed_event",
    "create_cognitive_model_selected_event",
    "create_cognitive_analysis_completed_event",
    "create_performance_measurement_event",
    "create_performance_alert_event",
    "create_agent_task_completed_event",
    "create_system_component_ready_event",
]
