"""
API Models for METIS Engagement Orchestration
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from uuid import UUID
from enum import Enum

from pydantic import BaseModel, Field


class EngagementPhase(str, Enum):
    PROBLEM_STRUCTURING = "problem_structuring"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    ANALYSIS_EXECUTION = "analysis_execution"
    SYNTHESIS_DELIVERY = "synthesis_delivery"


class EngagementStatus(str, Enum):
    CREATED = "created"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ERROR = "error"
    PAUSED = "paused"


class ProblemStatement(BaseModel):
    problem_description: str = Field(..., description="Detailed problem statement")
    business_context: Dict[str, Any] = Field(
        default_factory=dict, description="Industry, urgency, constraints"
    )
    stakeholders: List[str] = Field(
        default_factory=list, description="Key stakeholders"
    )
    success_criteria: List[str] = Field(
        default_factory=list, description="Engagement success metrics"
    )


class EngagementRequest(BaseModel):
    problem_statement: ProblemStatement
    client_name: str
    engagement_type: str = "strategy_consulting"
    priority: str = "medium"


class PhaseResult(BaseModel):
    phase: EngagementPhase
    status: str
    confidence: float
    insights: List[str] = Field(default_factory=list)
    data: Dict[str, Any] = Field(default_factory=dict)
    execution_time: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class EngagementResponse(BaseModel):
    engagement_id: UUID
    client_name: str
    problem_statement: ProblemStatement
    status: EngagementStatus
    current_phase: EngagementPhase
    progress_percentage: int
    phases: Dict[str, PhaseResult]
    overall_confidence: float
    estimated_cost: float
    created_at: datetime
    updated_at: datetime
    deliverable_ready: bool = False


class DeliverableRequest(BaseModel):
    format: str = Field("pdf", description="Export format: pdf, pptx, docx, json")
    template: str = Field(
        "mckinsey", description="Template style: mckinsey, bcg, bain, custom"
    )
    include_appendix: bool = True


class ReevaluationRequest(BaseModel):
    assumption_id: str = Field(
        ...,
        description="Identifier of the assumption to change (e.g., 'runway_months', 'market_growth_rate')",
    )
    new_value: Union[str, int, float, bool] = Field(
        ..., description="New value for the assumption"
    )
    assumption_context: Optional[str] = Field(
        None, description="Context about what this assumption represents"
    )


# HITL Clarification API Models
class ClarificationRequest(BaseModel):
    query: str = Field(..., description="User's original query")
    business_context: Dict[str, Any] = Field(
        default_factory=dict, description="Optional business context"
    )
    interaction_pattern: str = Field(
        "standard",
        description="Interaction pattern: standard, business_critical, quick_analysis",
    )
    user_id: Optional[str] = Field(None, description="Optional user identifier")


class ClarificationQuestionResponse(BaseModel):
    question_id: str
    response: str
    confidence: float = Field(
        1.0, ge=0.0, le=1.0, description="User confidence in response"
    )


class ClarificationResponseRequest(BaseModel):
    session_id: str
    responses: List[ClarificationQuestionResponse]


class ClarificationSkipRequest(BaseModel):
    session_id: str
    reason: str = Field("user_skip", description="Reason for skipping clarification")


class ClarificationResult(BaseModel):
    session_id: str
    needs_clarification: bool
    clarity_score: Optional[float] = None
    questions: Optional[List[Dict[str, Any]]] = None
    estimated_time_minutes: Optional[int] = None
    skip_option_available: bool = True
    title: Optional[str] = None
    description: Optional[str] = None


class EnhancedQueryResult(BaseModel):
    session_id: str
    success: bool
    enhanced_query: Optional[str] = None
    original_query: str
    error_message: Optional[str] = None
    clarification_count: int = 0


# NEW: V2 Tiered Clarification API Models
class QuestionTier(str, Enum):
    """Question complexity tiers"""

    ESSENTIAL = "essential"
    EXPERT = "expert"


class EngagementBriefModel(BaseModel):
    """Structured summary of user's request"""

    objective: str = Field(..., description="Core objective the user wants to achieve")
    platform: str = Field(
        ..., description="Platform, product, or system being discussed"
    )
    key_features: List[str] = Field(
        default_factory=list, description="Key features or differentiators mentioned"
    )
    confidence: float = Field(
        0.8, ge=0.0, le=1.0, description="Confidence in brief accuracy"
    )


class TieredClarificationQuestion(BaseModel):
    """Enhanced clarification question with tier information"""

    id: str = Field(..., description="Unique question identifier")
    question: str = Field(..., description="The clarification question")
    tier: QuestionTier = Field(..., description="Question complexity tier")
    dimension: str = Field(
        ..., description="Analytical dimension this question addresses"
    )
    question_type: str = Field("open_ended", description="Type of expected answer")
    complexity_level: str = Field("simple", description="Question complexity level")
    context_hint: Optional[str] = Field(None, description="Helper text for user")
    rationale: Optional[str] = Field(None, description="Why this question is needed")
    grounded_context: Optional[str] = Field(None, description="Context from research")


class TieredClarificationStartRequest(BaseModel):
    """Request to start tiered clarification process"""

    raw_query: str = Field(..., description="User's original unstructured query")
    business_context: Dict[str, Any] = Field(
        default_factory=dict, description="Optional business context"
    )
    user_expertise: Optional[str] = Field(None, description="User expertise level")


class TieredClarificationStartResponse(BaseModel):
    """Response from starting clarification with engagement brief and essential questions"""

    clarification_session_id: str = Field(
        ..., description="Session ID for tracking progress"
    )
    engagement_brief: EngagementBriefModel = Field(
        ..., description="Structured summary of request"
    )
    essential_questions: List[TieredClarificationQuestion] = Field(
        default_factory=list, description="Essential business questions"
    )
    estimated_time_minutes: int = Field(
        2, description="Estimated time to answer essential questions"
    )
    skip_option_available: bool = Field(
        True, description="Whether user can skip to analysis"
    )


class ClarificationAnswerV2(BaseModel):
    """User's answer to a clarification question"""

    question_id: str = Field(..., description="ID of the question being answered")
    answer: str = Field(..., description="User's response")
    confidence: float = Field(
        1.0, ge=0.0, le=1.0, description="User confidence in answer"
    )


class TieredClarificationContinueRequest(BaseModel):
    """Request to continue with expert questions after essential answers"""

    clarification_session_id: str = Field(..., description="Session ID")
    essential_answers: List[ClarificationAnswerV2] = Field(
        ..., description="Answers to essential questions"
    )


class TieredClarificationContinueResponse(BaseModel):
    """Response offering expert questions for deep dive"""

    session_id: str = Field(..., description="Session ID")
    status: str = Field(..., description="Status message")
    expert_questions: List[TieredClarificationQuestion] = Field(
        default_factory=list, description="Expert-level questions"
    )
    estimated_additional_minutes: int = Field(
        3, description="Estimated time for expert questions"
    )
    skip_option_available: bool = Field(
        True, description="Whether user can skip expert questions"
    )


class CreateEngagementFromClarificationRequest(BaseModel):
    """Final request to create engagement from clarification session"""

    clarification_session_id: str = Field(..., description="Session ID")
    expert_answers: Optional[List[ClarificationAnswerV2]] = Field(
        None, description="Optional expert question answers"
    )
    skip_expert: bool = Field(
        False, description="Whether expert questions were skipped"
    )


class CreateEngagementFromClarificationResponse(BaseModel):
    """Response with created engagement"""

    engagement_id: UUID = Field(..., description="Created engagement ID")
    success: bool = Field(
        True, description="Whether engagement was created successfully"
    )
    enhanced_query: str = Field(..., description="Final enhanced query for analysis")
    total_clarifications: int = Field(0, description="Total questions answered")
    processing_started: bool = Field(
        True, description="Whether analysis processing has started"
    )
