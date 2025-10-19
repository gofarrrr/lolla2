"""
Projects API - Pydantic Models
===============================

Shared request/response models for V2 projects API endpoints.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ============================================================
# PROJECT CRUD MODELS
# ============================================================


class ProjectCreateRequest(BaseModel):
    """Request model for creating a new project"""

    name: str = Field(..., min_length=1, max_length=255, description="Project name")
    description: Optional[str] = Field(
        None, max_length=2000, description="Project description"
    )
    organization_id: str = Field(..., description="Organization UUID")
    settings: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "context_merging_enabled": True,
            "auto_rag_indexing": True,
            "retention_policy": "standard",
            "privacy_settings": {
                "allow_cross_project_learning": False,
                "data_classification": "internal",
            },
        },
        description="Project configuration settings",
    )


class ProjectUpdateRequest(BaseModel):
    """Request model for updating a project"""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)
    settings: Optional[Dict[str, Any]] = None
    status: Optional[str] = Field(None, pattern="^(active|archived|deleted)$")


class ProjectResponse(BaseModel):
    """Response model for project data"""

    project_id: str
    organization_id: str
    name: str
    description: Optional[str]
    settings: Dict[str, Any]
    total_analyses: int
    total_tokens_used: int
    total_cost: float
    total_rag_documents: int
    total_text_chunks: int
    last_rag_update: Optional[datetime]
    status: str
    created_at: datetime
    updated_at: datetime
    last_accessed_at: Optional[datetime]

    # Computed fields from dashboard view
    recent_analyses_count: Optional[int] = 0
    last_analysis_date: Optional[datetime] = None
    rag_health_status: Optional[str] = "healthy"


class ProjectStatusResponse(BaseModel):
    """Response model for project status check"""

    project_id: str
    has_content: bool
    chat_ready: bool
    document_count: int
    text_chunks_count: int
    last_activity_date: Optional[datetime]
    rag_health: str
    chat_features: Dict[str, Any]


# ============================================================
# ANALYSIS & KNOWLEDGE BASE MODELS
# ============================================================


class AnalysisCreateRequest(BaseModel):
    """Enhanced analysis creation request with project context"""

    project_id: str = Field(..., description="Project UUID")
    user_query: str = Field(..., min_length=1, description="User's business question")
    engagement_type: str = Field(default="deep_dive", description="Analysis type")
    case_id: Optional[str] = Field(None, description="Business case identifier")
    merge_project_context: bool = Field(
        default=False, description="Whether to merge existing project knowledge"
    )
    organization_id: str = Field(..., description="Organization UUID")

    # Optional metadata
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional analysis metadata"
    )


class ContextMergePreview(BaseModel):
    """Preview of context that would be merged"""

    project_name: str
    total_available_documents: int
    total_available_chunks: int
    preview_chunks: List[Dict[str, Any]] = Field(
        description="Sample of most relevant context chunks"
    )
    estimated_context_tokens: int
    estimated_additional_cost: float


class ProjectKnowledgeSearchRequest(BaseModel):
    """Request for searching project knowledge base"""

    project_id: str
    query: str = Field(..., min_length=1, description="Search query")
    max_results: int = Field(default=10, ge=1, le=50)
    content_types: Optional[List[str]] = Field(
        default=None,
        description="Filter by content types: 'insight', 'recommendation', 'analysis', etc.",
    )
    document_types: Optional[List[str]] = Field(
        default=None,
        description="Filter by document types: 'analysis_summary', 'consultant_insights', etc.",
    )


# ============================================================
# MENTAL MODEL MODELS
# ============================================================


class MentalModelIngestionRequest(BaseModel):
    """Request for ingesting mental model documents"""

    project_id: str = Field(..., description="Target project ID")
    directory_path: Optional[str] = Field(
        None, description="Directory containing mental model files"
    )
    file_pattern: str = Field(default="*.txt", description="File pattern to match")
    organization_id: str = Field(..., description="Organization UUID")
    overwrite_existing: bool = Field(
        default=False, description="Overwrite existing mental models"
    )


class MentalModelIngestionResponse(BaseModel):
    """Response for mental model ingestion"""

    ingestion_id: str
    project_id: str
    project_name: str
    status: str
    total_files_found: int
    successfully_parsed: int
    successfully_ingested: int
    failed_files: List[str]
    estimated_chunks_created: int
    started_at: datetime
    completed_at: Optional[datetime]
    next_steps: List[str]


# ============================================================
# MENTAL MODEL CQA MODELS
# ============================================================


class MentalModelCQARequest(BaseModel):
    """Request for mental model CQA evaluation"""

    mental_model_data: Dict[str, Any] = Field(
        ..., description="Mental model data to evaluate"
    )
    model_type_override: Optional[str] = Field(
        None, description="Override auto-detected model type"
    )
    rubric_id: Optional[str] = Field(
        None, description="Specific rubric to use for evaluation"
    )
    organization_id: str = Field(..., description="Organization UUID")


class QualityScoreResponse(BaseModel):
    """Response model for individual quality dimension score"""

    dimension: str
    score: float = Field(..., ge=0.0, le=10.0)
    rationale: str
    evidence: List[str]
    confidence: float = Field(..., ge=0.0, le=1.0)


class MentalModelCQAResponse(BaseModel):
    """Response for mental model CQA evaluation"""

    mental_model_id: str
    mental_model_name: str
    model_type: str
    rubric_used: str
    evaluation_timestamp: datetime

    # Individual dimension scores
    dimension_scores: Dict[str, QualityScoreResponse]

    # Aggregate metrics
    overall_score: float = Field(..., ge=0.0, le=10.0)
    weighted_score: float = Field(..., ge=0.0, le=10.0)
    confidence_level: float = Field(..., ge=0.0, le=1.0)

    # Quality assessment
    quality_tier: str = Field(..., pattern="^(excellent|good|average|poor)$")
    validation_status: str = Field(..., pattern="^(passed|failed|review_needed)$")

    # Audit information
    evaluator_version: str
    execution_time_ms: int
    context_stream_id: Optional[str] = None


class BatchCQARequest(BaseModel):
    """Request for batch CQA evaluation"""

    project_id: str = Field(..., description="Project ID containing mental models")
    filter_criteria: Optional[Dict[str, Any]] = Field(
        default=None, description="Filtering criteria"
    )
    organization_id: str = Field(..., description="Organization UUID")


class BatchCQAResponse(BaseModel):
    """Response for batch CQA evaluation"""

    batch_id: str
    project_id: str
    project_name: str
    total_models: int
    evaluated_models: int
    failed_evaluations: int
    average_quality_score: float
    quality_distribution: Dict[str, int]
    validation_summary: Dict[str, int]  # passed, failed, review_needed counts
    started_at: datetime
    completed_at: Optional[datetime]
    results: List[MentalModelCQAResponse]


class CQABenchmarkRequest(BaseModel):
    """Request for CQA benchmark analysis"""

    project_id: Optional[str] = Field(
        None, description="Specific project (or all projects if None)"
    )
    time_range_days: int = Field(
        default=30, ge=1, le=365, description="Time range for analysis"
    )
    model_types: Optional[List[str]] = Field(None, description="Filter by model types")
    organization_id: str = Field(..., description="Organization UUID")


class CQABenchmarkResponse(BaseModel):
    """Response for CQA benchmark analysis"""

    organization_id: str
    project_id: Optional[str]
    analysis_period: Dict[str, str]  # start_date, end_date

    # Aggregate statistics
    total_evaluations: int
    average_quality_score: float
    quality_score_trend: List[Dict[str, Any]]  # time series data

    # Quality distribution
    quality_tier_distribution: Dict[str, int]
    validation_status_distribution: Dict[str, int]
    model_type_performance: Dict[str, Dict[str, Any]]

    # Top and bottom performers
    highest_quality_models: List[Dict[str, Any]]
    lowest_quality_models: List[Dict[str, Any]]

    # Recommendations
    improvement_recommendations: List[str]
