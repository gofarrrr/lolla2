"""
Engagements API - Pydantic Models
==================================

Request/response models for engagement endpoints.

Operation Bedrock: Task 10.0 - API Decomposition
"""

from typing import Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


# ============================================================================
# Request Models
# ============================================================================

class AnsweredQuestion(BaseModel):
    """Answered question from enhancement flow"""
    question_id: str
    question_text: str
    answer: str


class ResearchQuestion(BaseModel):
    """Question flagged for research during analysis"""
    question_id: str
    question_text: str


class StartEngagementRequest(BaseModel):
    """Request to start a new strategic analysis engagement"""
    user_query: str = Field(..., description="The strategic question to analyze")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    enhanced_context: Optional[Dict[str, Any]] = Field(None, description="Enhanced context from query enhancement flow")
    quality_requested: Optional[int] = Field(None, description="Quality level requested (60-95%)")

    # New fields for progressive questions integration
    enhancement_questions_session_id: Optional[str] = Field(None, description="Session ID from progressive questions")
    answered_questions: Optional[list[AnsweredQuestion]] = Field(None, description="Questions answered by user")
    research_questions: Optional[list[ResearchQuestion]] = Field(None, description="Questions to be researched by system")
    quality_target: Optional[float] = Field(None, description="Target quality score 0.5-0.95")
    interactive_mode: Optional[bool] = Field(False, description="If true, pause after generating questions to get user answers")


class AnswerSubmission(BaseModel):
    """User's answer to a question"""
    question_id: str
    answer: str


class SubmitAnswersRequest(BaseModel):
    """Request to submit answers and resume pipeline"""
    answers: list[AnswerSubmission]


class OutcomeReportRequest(BaseModel):
    """User-submitted outcome for calibration closure"""
    outcome: str = Field(
        ..., description="yes/no/early or a numeric 0..1 float as string"
    )
    notes: Optional[str] = None


# ============================================================================
# Response Models
# ============================================================================

class EngagementStatusResponse(BaseModel):
    """Response containing current engagement status"""
    trace_id: str
    status: str
    current_stage: str
    stage_number: int
    total_stages: int
    progress_percentage: float
    stage_description: str
    is_completed: bool
    error: Optional[str] = None


class QuestionsResponse(BaseModel):
    """Response containing generated questions for interactive mode"""
    trace_id: str
    questions: list[Dict[str, Any]]
    status: str
    checkpoint_id: Optional[str] = None
    ux_metadata: Optional[Dict[str, Any]] = None  # Progressive disclosure UX metadata


class SelectedConsultant(BaseModel):
    """Selected consultant with YAML-driven selection rationale"""
    consultant_id: str
    consultant_type: str = Field(default="unknown")
    specialization: str = Field(default="")
    predicted_effectiveness: float = Field(default=0.0)
    selection_rationale: str = Field(default="")
    assigned_dimensions: Optional[list] = Field(default_factory=list)


class EngagementReportResponse(BaseModel):
    """Response containing the final strategic report
    
    Fields are provided at both root level AND inside 'report' for frontend compatibility.
    The frontend expects key fields at the top level, not nested inside 'report'.
    """
    trace_id: str
    report: Dict[str, Any]
    generated_at: datetime
    markdown_content: str
    selected_consultants: Optional[list[SelectedConsultant]] = Field(default_factory=list)
    consultant_selection_methodology: Optional[str] = None
    human_interactions: Optional[Dict[str, Any]] = None
    research_provider_events: Optional[list] = None
    mece_framework: Optional[Dict[str, Any]] = None  # OPERATION CRYSTAL PALACE: Stage 2
    user_query: Optional[str] = None
    
    # Frontend expects these at root level (duplicated from report for compatibility)
    consultant_analyses: Optional[list[Dict[str, Any]]] = Field(default_factory=list)
    strategic_recommendations: Optional[list[Dict[str, Any]]] = Field(default_factory=list)
    executive_summary: Optional[str] = None
    devils_advocate_transcript: Optional[str] = None
    quality_metrics: Optional[Dict[str, Any]] = None
    key_decisions: Optional[list[Dict[str, Any]]] = Field(default_factory=list)
    da_transcript: Optional[str] = None
