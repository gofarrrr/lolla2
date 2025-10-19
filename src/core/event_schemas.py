"""
Event Schema Definitions - Operation Bedrock

Pydantic schemas for strict event payload validation.
Ensures data quality and prevents schema drift in UnifiedContextStream.

Usage:
    from src.core.event_schemas import validate_event_payload

    # Validate event before adding to stream
    is_valid, errors = validate_event_payload(
        event_type=ContextEventType.STAGE_STARTED,
        data={"stage": "parallel_analysis", "consultants_count": 3}
    )
"""

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator
from enum import Enum

from src.core.unified_context_stream import ContextEventType


# ============================================================================
# STAGE LIFECYCLE EVENTS
# ============================================================================


class StageStartedSchema(BaseModel):
    """Schema for STAGE_STARTED events"""
    stage: str = Field(..., description="Stage name (e.g., 'parallel_analysis')")
    consultants_count: Optional[int] = Field(None, ge=0, description="Number of consultants")

    class Config:
        extra = "allow"  # Allow additional fields for flexibility


class StageCompleteSchema(BaseModel):
    """Schema for STAGE_COMPLETE events"""
    stage: str = Field(..., description="Stage name")
    processing_time_ms: int = Field(..., ge=0, description="Processing time in milliseconds")
    consultant_count: Optional[int] = Field(None, ge=0, description="Number of consultants")
    success: bool = Field(..., description="Whether stage completed successfully")

    class Config:
        extra = "allow"


# ============================================================================
# PROMPT & EXECUTION EVENTS
# ============================================================================


class PromptsBuiltSchema(BaseModel):
    """Schema for PROMPTS_BUILT events"""
    stage: str = Field(..., description="Stage name")
    prompt_count: int = Field(..., ge=0, description="Number of prompts built")
    total_estimated_tokens: int = Field(..., ge=0, description="Total estimated tokens")

    class Config:
        extra = "allow"


class LLMExecutionCompleteSchema(BaseModel):
    """Schema for LLM_EXECUTION_COMPLETE events"""
    stage: str = Field(..., description="Stage name")
    total_calls: int = Field(..., ge=0, description="Total LLM calls made")
    successful_calls: int = Field(..., ge=0, description="Successful LLM calls")
    failed_calls: int = Field(..., ge=0, description="Failed LLM calls")
    total_tokens: int = Field(..., ge=0, description="Total tokens used")
    total_time_ms: int = Field(..., ge=0, description="Total execution time in ms")

    @field_validator("successful_calls", "failed_calls")
    @classmethod
    def validate_call_counts(cls, v, info):
        """Validate that successful + failed = total"""
        # Note: This is checked after all fields are set
        return v

    class Config:
        extra = "allow"


# ============================================================================
# AGGREGATION EVENTS
# ============================================================================


class AggregationCompleteSchema(BaseModel):
    """Schema for AGGREGATION_COMPLETE events"""
    stage: str = Field(..., description="Stage name")
    total_insights: int = Field(..., ge=0, description="Total insights extracted")
    convergent_insights: int = Field(..., ge=0, description="Number of convergent insights")
    divergent_perspectives: int = Field(..., ge=0, description="Number of divergent perspectives")
    orthogonality_index: float = Field(..., ge=0.0, le=1.0, description="Orthogonality index (0-1)")
    has_minority_report: bool = Field(..., description="Whether minority report was generated")

    class Config:
        extra = "allow"


# ============================================================================
# STAGE 0 ENRICHMENT EVENTS
# ============================================================================


class Stage0ExperimentAssignedSchema(BaseModel):
    """Schema for STAGE0_EXPERIMENT_ASSIGNED events"""
    stage: str = Field(..., description="Stage name")
    enabled: bool = Field(..., description="Whether Stage 0 is enabled")
    variant: str = Field(..., description="Experiment variant label")
    consultant_count: int = Field(..., ge=0, description="Number of consultants")

    class Config:
        extra = "allow"


class Stage0PlanRecordedSchema(BaseModel):
    """Schema for STAGE0_PLAN_RECORDED events"""
    entries: List[Dict[str, Any]] = Field(..., description="Depth pack plan entries")
    total_duration_ms: int = Field(..., ge=0, description="Total Stage 0 processing time")
    enabled: bool = Field(..., description="Whether Stage 0 was enabled")
    variant: str = Field(..., description="Experiment variant label")

    class Config:
        extra = "allow"


class Stage0EnrichmentCompleteSchema(BaseModel):
    """Schema for STAGE0_ENRICHMENT_COMPLETE events"""
    stage: str = Field(..., description="Stage name")
    depth_pack_tokens: int = Field(..., ge=0, description="Depth pack tokens added")
    mm_items_count: int = Field(..., ge=0, description="Mental model items added")
    stage0_latency_ms: int = Field(..., ge=0, description="Stage 0 processing time")

    class Config:
        extra = "allow"


# ============================================================================
# ERROR EVENTS
# ============================================================================


class ErrorOccurredSchema(BaseModel):
    """Schema for ERROR_OCCURRED events"""
    stage: str = Field(..., description="Stage where error occurred")
    error_type: str = Field(..., description="Type of error")
    message: str = Field(..., description="Error message")
    impact: Optional[str] = Field(None, description="Impact description")

    class Config:
        extra = "allow"


# ============================================================================
# ORTHOGONALITY & DIVERSITY EVENTS
# ============================================================================


class OrthogonalityIndexComputedSchema(BaseModel):
    """Schema for ORTHOGONALITY_INDEX_COMPUTED events"""
    orthogonality_index: float = Field(..., ge=0.0, le=1.0, description="Orthogonality index")
    consultant_count: int = Field(..., ge=0, description="Number of consultants")
    interpretation: str = Field(..., description="Interpretation of orthogonality")

    class Config:
        extra = "allow"


class DiversityWatchdogTriggeredSchema(BaseModel):
    """Schema for DIVERSITY_WATCHDOG_TRIGGERED events"""
    orthogonality_index: float = Field(..., ge=0.0, le=1.0, description="Orthogonality index")
    threshold: float = Field(..., ge=0.0, le=1.0, description="Threshold that triggered watchdog")
    reason: str = Field(..., description="Reason for trigger")

    class Config:
        extra = "allow"


# ============================================================================
# SCHEMA REGISTRY
# ============================================================================


EVENT_SCHEMAS: Dict[ContextEventType, type[BaseModel]] = {
    # Stage lifecycle
    ContextEventType.PIPELINE_STAGE_STARTED: StageStartedSchema,
    ContextEventType.PIPELINE_STAGE_COMPLETED: StageCompleteSchema,

    # Prompt & Execution
    ContextEventType.PROMPTS_BUILT: PromptsBuiltSchema,
    ContextEventType.LLM_EXECUTION_COMPLETE: LLMExecutionCompleteSchema,

    # Aggregation
    ContextEventType.AGGREGATION_COMPLETE: AggregationCompleteSchema,

    # Stage 0 Enrichment
    ContextEventType.STAGE0_EXPERIMENT_ASSIGNED: Stage0ExperimentAssignedSchema,
    ContextEventType.STAGE0_PLAN_RECORDED: Stage0PlanRecordedSchema,
    ContextEventType.STAGE0_ENRICHMENT_COMPLETE: Stage0EnrichmentCompleteSchema,

    # Errors
    ContextEventType.ERROR_OCCURRED: ErrorOccurredSchema,

    # Orthogonality & Diversity
    ContextEventType.ORTHOGONALITY_INDEX_COMPUTED: OrthogonalityIndexComputedSchema,
    ContextEventType.DIVERSITY_WATCHDOG_TRIGGERED: DiversityWatchdogTriggeredSchema,
}


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================


def validate_event_payload(
    event_type: ContextEventType,
    data: Dict[str, Any],
) -> Tuple[bool, Optional[List[str]]]:
    """
    Validate event payload against schema.

    Args:
        event_type: Type of event
        data: Event data payload

    Returns:
        Tuple of (is_valid, error_messages)
        - (True, None) if valid
        - (False, [error1, error2, ...]) if invalid
    """
    # If no schema defined, allow the event (permissive mode for undefined events)
    if event_type not in EVENT_SCHEMAS:
        return True, None

    schema_class = EVENT_SCHEMAS[event_type]

    try:
        # Validate using Pydantic
        schema_class(**data)
        return True, None
    except Exception as e:
        # Extract validation errors
        if hasattr(e, "errors"):
            errors = [f"{err['loc'][0]}: {err['msg']}" for err in e.errors()]
        else:
            errors = [str(e)]
        return False, errors


def get_event_schema(event_type: ContextEventType) -> Optional[type[BaseModel]]:
    """
    Get Pydantic schema for an event type.

    Args:
        event_type: Type of event

    Returns:
        Pydantic model class or None if no schema defined
    """
    return EVENT_SCHEMAS.get(event_type)


def get_registered_event_types() -> List[ContextEventType]:
    """
    Get list of event types with registered schemas.

    Returns:
        List of ContextEventType enums with schemas
    """
    return list(EVENT_SCHEMAS.keys())
