"""
Engagement Domain Models

Models related to engagement lifecycle, clarification, exploration,
and workflow management.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from .enums import *

class EngagementContext(BaseModel):
    """Core engagement context for all cognitive operations"""

    engagement_id: UUID = Field(default_factory=uuid4)
    problem_statement: str = Field(..., min_length=10, max_length=5000)
    client_name: str = Field(default="Unknown Client")
    business_context: Dict[str, Any] = Field(default_factory=dict)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    compliance_requirements: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Week 1 Day 2: Context-Specific Learning - Add fields for contextual effectiveness tracking
    problem_classification: str = Field(
        default="general_analysis", description="Type of problem being analyzed"
    )
    industry: str = Field(
        default="general", description="Industry context for the engagement"
    )

    model_config = ConfigDict()




class WorkflowState(BaseModel):
    """Workflow execution state tracking"""

    current_phase: EngagementPhase
    completed_phases: List[EngagementPhase] = Field(default_factory=list)
    phase_results: Dict[str, Any] = Field(default_factory=dict)
    next_actions: List[str] = Field(default_factory=list)
    estimated_completion: Optional[datetime] = None
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    execution_metadata: Dict[str, Any] = Field(default_factory=dict)
    status: str = Field(default="created")
    error_details: Optional[Dict[str, Any]] = None




class DeliverableArtifact(BaseModel):
    """Structured deliverable artifact"""

    artifact_id: UUID = Field(default_factory=uuid4)
    artifact_type: str = Field(..., pattern=r"^[a-z_]+$")
    content: Dict[str, Any]
    confidence_level: ConfidenceLevel
    supporting_evidence: List[str] = Field(default_factory=list)
    methodology_used: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    audit_trail: List[str] = Field(default_factory=list)




class ClarificationQuestion(BaseModel):
    """Individual clarification question for HITL interaction"""

    question_id: str = Field(..., description="Unique identifier for the question")
    question_text: str = Field(
        ..., min_length=10, description="The actual question text"
    )
    question_type: ClarificationQuestionType = Field(
        ..., description="Type of question"
    )
    dimension: str = Field(
        ..., description="Business dimension this question addresses"
    )
    complexity: ClarificationComplexity = Field(default=ClarificationComplexity.MEDIUM)
    required: bool = Field(
        default=False, description="Whether this question is required"
    )
    context_hint: Optional[str] = Field(
        None, description="Helpful context for the user"
    )
    placeholder_text: Optional[str] = Field(
        None, description="Placeholder text for input"
    )
    validation_rules: Dict[str, Any] = Field(default_factory=dict)
    impact_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Impact on analysis quality"
    )
    business_relevance: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Business importance"
    )




class ClarificationResponse(BaseModel):
    """User's response to a clarification question"""

    question_id: str = Field(..., description="ID of the question being answered")
    response_text: str = Field(..., description="User's response")
    confidence_level: float = Field(
        default=1.0, ge=0.0, le=1.0, description="User confidence in response"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))




class ClarificationSession(BaseModel):
    """Complete HITL clarification session data"""

    session_id: str = Field(..., description="Unique session identifier")
    original_query: str = Field(..., description="Original user query")
    enhanced_query: Optional[str] = Field(
        None, description="Enhanced query after clarification"
    )
    clarity_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Original query clarity score"
    )
    questions_presented: List[ClarificationQuestion] = Field(default_factory=list)
    responses_received: List[ClarificationResponse] = Field(default_factory=list)
    session_status: str = Field(default="pending", description="Session status")
    interaction_pattern: str = Field(
        default="standard", description="Interaction pattern used"
    )
    estimated_time_minutes: int = Field(
        default=5, ge=1, description="Estimated completion time"
    )
    actual_completion_time_seconds: Optional[int] = Field(
        None, description="Actual completion time"
    )
    dimensions_clarified: List[str] = Field(
        default_factory=list, description="Business dimensions clarified"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = Field(None)

    # Quality metrics
    enhancement_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    user_satisfaction_score: Optional[float] = Field(None, ge=0.0, le=5.0)




class ExplorationContext(BaseModel):
    """Exploration vs exploitation decision context"""

    decision_type: ExplorationDecision = Field(
        ..., description="Strategy decision made"
    )
    exploration_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Current exploration rate"
    )
    diversity_score: float = Field(
        ..., ge=0.0, le=1.0, description="Model diversity maintained"
    )
    strategic_mutation: bool = Field(
        default=False, description="Whether strategic mutation was applied"
    )
    rationale: str = Field(..., description="Reasoning for exploration decision")
    expected_value: float = Field(
        ..., ge=0.0, le=1.0, description="Expected value of decision"
    )
    risk_tolerance: float = Field(
        default=0.15, ge=0.0, le=1.0, description="Risk tolerance for exploration"
    )




class FailureModeResponse(BaseModel):
    """Failure mode handling response"""

    component_name: str = Field(..., description="Component that failed")
    failure_type: str = Field(..., description="Type of failure encountered")
    severity_level: VulnerabilityDetectionLevel = Field(
        ..., description="Failure severity"
    )
    fallback_used: bool = Field(
        ..., description="Whether fallback mechanism was activated"
    )
    user_communication: str = Field(
        ..., description="User-friendly explanation of failure"
    )
    recovery_actions: List[str] = Field(
        default_factory=list, description="Actions taken for recovery"
    )
    impact_assessment: Dict[str, Any] = Field(
        default_factory=dict, description="Assessment of failure impact"
    )
    transparency_level: str = Field(
        default="high", description="Level of transparency provided to user"
    )




