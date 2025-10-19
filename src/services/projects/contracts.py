"""
Project Service Contracts
=========================

Operation Chimera - Head of Interface: Phase 1 (The Seam)

This module defines the clean contracts and Data Transfer Objects (DTOs)
for the Project Service layer, establishing the architectural seam that
will allow safe decoupling of API concerns from business logic.

Key Components:
- DTOs for all project operations
- IProjectService protocol defining business logic interface
- Clean separation between HTTP/API layer and business logic
"""

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any, Protocol
from pydantic import BaseModel, Field
from uuid import UUID


# ============================================================
# Data Transfer Objects (DTOs)
# ============================================================

class ProjectCreateRequest(BaseModel):
    """DTO for creating a new project"""
    
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
    """DTO for updating a project"""
    
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)
    settings: Optional[Dict[str, Any]] = None
    status: Optional[str] = Field(None, pattern="^(active|archived|deleted)$")


class ProjectDTO(BaseModel):
    """DTO representing a project entity"""
    
    project_id: str = Field(..., description="Project UUID")
    organization_id: str = Field(..., description="Organization UUID")
    name: str = Field(..., description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    settings: Dict[str, Any] = Field(default_factory=dict, description="Project settings")
    status: str = Field(default="active", description="Project status")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    last_accessed_at: Optional[datetime] = Field(None, description="Last access timestamp")
    
    # Statistics (populated from dashboard view)
    total_analyses: Optional[int] = Field(None, description="Total analysis count")
    active_analyses: Optional[int] = Field(None, description="Active analysis count")
    knowledge_base_items: Optional[int] = Field(None, description="Knowledge base item count")
    last_analysis_date: Optional[datetime] = Field(None, description="Last analysis date")


class AnalysisCreateRequest(BaseModel):
    """DTO for creating a new analysis within a project"""
    
    project_id: str = Field(..., description="Target project UUID")
    user_query: str = Field(..., min_length=1, description="Analysis query")
    engagement_type: str = Field(default="deep_dive", description="Type of analysis engagement")
    merge_project_context: bool = Field(default=False, description="Whether to merge with existing project context")
    organization_id: str = Field(..., description="Organization UUID")
    case_id: Optional[str] = Field(None, description="Optional case ID")
    file: Optional[Any] = Field(None, description="Optional uploaded file")
    analysis_type: str = Field(
        default="standard", description="Type of analysis to perform"
    )
    consultants: Optional[List[str]] = Field(
        default_factory=list, description="Specific consultants to include"
    )
    priority: str = Field(
        default="normal", pattern="^(low|normal|high|urgent)$", description="Analysis priority"
    )


class ContextMergePreview(BaseModel):
    """DTO for context merge preview results"""
    
    merged_context: Dict[str, Any] = Field(..., description="Preview of merged context")
    context_sources: List[str] = Field(..., description="Sources of context data")
    potential_conflicts: List[Dict[str, Any]] = Field(
        default_factory=list, description="Potential context conflicts"
    )
    merge_strategy: str = Field(..., description="Strategy used for merging")
    confidence_score: float = Field(..., description="Confidence in merge quality")


class ProjectKnowledgeSearchRequest(BaseModel):
    """DTO for project knowledge base search"""
    
    query: str = Field(..., min_length=1, description="Search query")
    search_type: str = Field(
        default="semantic", pattern="^(semantic|keyword|hybrid)$"
    )
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results")
    include_metadata: bool = Field(default=True, description="Include result metadata")
    knowledge_types: Optional[List[str]] = Field(
        default_factory=list, description="Filter by knowledge types"
    )


class MentalModelIngestionRequest(BaseModel):
    """DTO for mental model ingestion"""
    
    file_content: str = Field(..., description="Mental model file content")
    filename: str = Field(..., description="Original filename")
    project_id: Optional[str] = Field(None, description="Target project UUID")
    processing_options: Dict[str, Any] = Field(
        default_factory=dict, description="Processing configuration"
    )


class MentalModelIngestionResponse(BaseModel):
    """DTO for mental model ingestion results"""
    
    ingestion_id: str = Field(..., description="Ingestion operation UUID")
    status: str = Field(..., description="Ingestion status")
    models_extracted: int = Field(..., description="Number of models extracted")
    extraction_details: Dict[str, Any] = Field(..., description="Detailed extraction results")
    validation_results: Dict[str, Any] = Field(..., description="Validation results")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    created_at: datetime = Field(..., description="Ingestion timestamp")


class MentalModelCQARequest(BaseModel):
    """DTO for mental model CQA evaluation"""
    
    mental_model_name: str = Field(..., description="Mental model to evaluate")
    question: str = Field(..., description="Question to evaluate")
    context: Optional[str] = Field(None, description="Additional context")
    rubric_name: str = Field(default="default", description="Evaluation rubric")
    evaluation_mode: str = Field(
        default="comprehensive", pattern="^(quick|standard|comprehensive)$"
    )


class QualityScoreResponse(BaseModel):
    """DTO for quality assessment scores"""
    
    score: float = Field(..., ge=0.0, le=1.0, description="Quality score")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in score")
    criteria_scores: Dict[str, float] = Field(..., description="Individual criteria scores")
    feedback: str = Field(..., description="Qualitative feedback")


class MentalModelCQAResponse(BaseModel):
    """DTO for mental model CQA evaluation results"""
    
    evaluation_id: str = Field(..., description="Evaluation UUID")
    mental_model_name: str = Field(..., description="Evaluated mental model")
    question: str = Field(..., description="Evaluation question")
    answer: str = Field(..., description="Generated answer")
    quality_assessment: QualityScoreResponse = Field(..., description="Quality scores")
    processing_details: Dict[str, Any] = Field(..., description="Processing metadata")
    rubric_applied: str = Field(..., description="Applied evaluation rubric")
    evaluation_timestamp: datetime = Field(..., description="Evaluation timestamp")


class BatchCQARequest(BaseModel):
    """DTO for batch CQA evaluation"""
    
    mental_model_names: List[str] = Field(..., description="Mental models to evaluate")
    questions: List[str] = Field(..., description="Questions for evaluation")
    rubric_name: str = Field(default="default", description="Evaluation rubric")
    evaluation_mode: str = Field(default="standard", description="Evaluation mode")
    parallel_processing: bool = Field(default=True, description="Enable parallel processing")


class BatchCQAResponse(BaseModel):
    """DTO for batch CQA evaluation results"""
    
    batch_id: str = Field(..., description="Batch evaluation UUID")
    total_evaluations: int = Field(..., description="Total evaluations performed")
    completed_evaluations: int = Field(..., description="Successfully completed evaluations")
    failed_evaluations: int = Field(..., description="Failed evaluations")
    evaluation_results: List[MentalModelCQAResponse] = Field(..., description="Individual results")
    batch_summary: Dict[str, Any] = Field(..., description="Batch-level summary")
    processing_time_ms: int = Field(..., description="Total processing time")
    created_at: datetime = Field(..., description="Batch creation timestamp")


class CQABenchmarkRequest(BaseModel):
    """DTO for CQA benchmark execution"""
    
    benchmark_name: str = Field(..., description="Benchmark to execute")
    mental_model_filter: Optional[List[str]] = Field(
        None, description="Filter to specific mental models"
    )
    evaluation_parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Benchmark parameters"
    )


class CQABenchmarkResponse(BaseModel):
    """DTO for CQA benchmark results"""
    
    benchmark_id: str = Field(..., description="Benchmark execution UUID")
    benchmark_name: str = Field(..., description="Executed benchmark")
    overall_score: float = Field(..., description="Overall benchmark score")
    model_scores: Dict[str, float] = Field(..., description="Per-model scores")
    benchmark_details: Dict[str, Any] = Field(..., description="Detailed benchmark results")
    execution_time_ms: int = Field(..., description="Execution time")
    executed_at: datetime = Field(..., description="Execution timestamp")


class ProjectStatusResponse(BaseModel):
    """DTO for project status information"""
    
    project_id: str = Field(..., description="Project UUID")
    status: str = Field(..., description="Current project status")
    health_score: float = Field(..., description="Project health score")
    activity_summary: Dict[str, Any] = Field(..., description="Recent activity summary")
    resource_usage: Dict[str, Any] = Field(..., description="Resource usage metrics")
    last_updated: datetime = Field(..., description="Status last updated")


# ============================================================
# Service Protocol
# ============================================================

class IProjectService(Protocol):
    """
    Project Service Protocol
    
    Defines the interface for all project-related business logic operations.
    This protocol will be implemented by the concrete service classes and
    enables clean dependency injection in the API layer.
    """
    
    # Core CRUD Operations
    async def create_project(
        self, 
        request: ProjectCreateRequest, 
        user_id: Optional[str] = None
    ) -> ProjectDTO:
        """Create a new project with default settings"""
        ...
    
    async def get_project_by_id(
        self, 
        project_id: str, 
        user_id: Optional[str] = None
    ) -> ProjectDTO:
        """Get project by ID with statistics"""
        ...
    
    async def list_projects(
        self, 
        organization_id: str, 
        user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[ProjectDTO]:
        """List projects for an organization"""
        ...
    
    async def update_project(
        self, 
        project_id: str, 
        request: ProjectUpdateRequest, 
        user_id: Optional[str] = None
    ) -> ProjectDTO:
        """Update project details"""
        ...
    
    async def delete_project(
        self, 
        project_id: str, 
        user_id: Optional[str] = None
    ) -> bool:
        """Delete a project"""
        ...
    
    # Analysis Operations
    async def create_analysis(
        self, 
        request: AnalysisCreateRequest, 
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new analysis within a project"""
        ...
    
    async def preview_context_merge(
        self, 
        project_id: str, 
        new_context: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> ContextMergePreview:
        """Preview context merge for a project"""
        ...
    
    # Knowledge Base Operations
    async def get_knowledge_base(
        self, 
        project_id: str, 
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get project knowledge base contents"""
        ...
    
    async def search_project_knowledge(
        self, 
        project_id: str, 
        request: ProjectKnowledgeSearchRequest,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Search project knowledge base"""
        ...
    
    # Statistics and Status
    async def get_project_statistics(
        self, 
        project_id: str, 
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get project statistics and metrics"""
        ...
    
    async def get_project_status(
        self, 
        project_id: str, 
        user_id: Optional[str] = None
    ) -> ProjectStatusResponse:
        """Get detailed project status information"""
        ...
    
    # Mental Model Operations
    async def ingest_mental_models(
        self, 
        request: MentalModelIngestionRequest,
        user_id: Optional[str] = None
    ) -> MentalModelIngestionResponse:
        """Ingest mental models from uploaded content"""
        ...
    
    async def evaluate_mental_model_cqa(
        self, 
        request: MentalModelCQARequest,
        user_id: Optional[str] = None
    ) -> MentalModelCQAResponse:
        """Evaluate mental model using CQA"""
        ...
    
    async def batch_evaluate_mental_models(
        self, 
        request: BatchCQARequest,
        user_id: Optional[str] = None
    ) -> BatchCQAResponse:
        """Perform batch CQA evaluation of mental models"""
        ...
    
    async def run_cqa_benchmark(
        self, 
        request: CQABenchmarkRequest,
        user_id: Optional[str] = None
    ) -> CQABenchmarkResponse:
        """Execute CQA benchmark suite"""
        ...
    
    async def get_cqa_rubrics(
        self,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get available CQA evaluation rubrics"""
        ...
    
    # Health and Monitoring
    async def get_service_health(self) -> Dict[str, Any]:
        """Get service health status"""
        ...