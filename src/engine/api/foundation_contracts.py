"""
Foundation API Service Contracts
===============================

Operation Chimera Phase 3 - Foundation Service Extraction

This module defines the clean contracts and Data Transfer Objects (DTOs)
for the Foundation API services, establishing the architectural seam that
will allow safe extraction of business logic from the monolithic API.

Key Components:
- DTOs for all engagement and analysis operations
- Service protocols for specialized Foundation services
- Clean separation between HTTP/API layer and business logic
"""

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any, Protocol
from pydantic import BaseModel, Field
from uuid import UUID


# ============================================================
# Data Transfer Objects (DTOs) - Reused from existing models
# ============================================================

class EngagementCreateRequest(BaseModel):
    """DTO for creating new engagement"""
    
    problem_statement: str = Field(..., min_length=10, max_length=5000)
    business_context: Dict[str, Any] = Field(default_factory=dict)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    compliance_requirements: Dict[str, Any] = Field(default_factory=dict)


class EngagementResponse(BaseModel):
    """Enhanced response model with database integration"""
    
    engagement_id: str
    status: str
    created_at: str
    updated_at: Optional[str] = None
    problem_statement: str
    business_context: Dict[str, Any]
    analysis_context: Dict[str, Any] = Field(default_factory=dict)
    created_by: Optional[str] = None
    session_id: Optional[str] = None
    transparency_layers_count: int = 0
    decisions_count: int = 0


class CognitiveAnalysisRequest(BaseModel):
    """Request model for cognitive analysis"""
    
    engagement_id: str
    force_model_selection: Optional[List[str]] = None
    analysis_preferences: Dict[str, Any] = Field(default_factory=dict)
    create_transparency_layers: bool = True
    rigor_level: str = Field(default="L1", pattern="^L[0-3]$")


class CognitiveAnalysisResponse(BaseModel):
    """Enhanced response model with database integration"""
    
    engagement_id: str
    analysis_id: str
    status: str
    cognitive_state: Dict[str, Any]
    reasoning_steps: List[Dict[str, Any]]
    confidence_scores: Dict[str, float]
    selected_models: List[str]
    nway_patterns_detected: List[Dict[str, Any]]
    transparency_layers_created: int
    munger_overlay_id: Optional[str] = None
    processing_time_ms: float
    created_at: str


class ModelListResponse(BaseModel):
    """Enhanced response model with relevance scoring"""
    
    models: List[Dict[str, Any]]
    total_count: int
    categories: List[str]
    enhanced_models_count: int
    avg_effectiveness_score: float


class EngagementListResponse(BaseModel):
    """Response model for engagement listing"""
    
    engagements: List[EngagementResponse]
    total_count: int
    page: int
    page_size: int
    has_next: bool


class TransparencyLayersResponse(BaseModel):
    """Response model for transparency layers"""
    
    engagement_id: str
    layers: List[Dict[str, Any]]
    total_layers: int
    navigation_path: List[str]


class DatabaseHealthResponse(BaseModel):
    """Response model for database health"""
    
    status: str
    connection_health: Dict[str, Any]
    metrics: Dict[str, Any]
    tables_accessible: int
    performance_ms: float


# ============================================================
# Foundation Service Protocols
# ============================================================

class IFoundationRepositoryService(Protocol):
    """
    Repository service for Foundation API data persistence operations.
    
    Responsibilities:
    - CRUD operations on engagements and analysis records
    - Mental model data access and relevance scoring
    - Database health monitoring and performance tracking
    - Legacy ID handling and UUID conversion
    """
    
    async def create_engagement(
        self,
        problem_statement: str,
        business_context: Dict[str, Any],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> UUID:
        """Create engagement record in database"""
        ...
    
    async def get_engagement(
        self,
        engagement_id: UUID
    ) -> Optional[Dict[str, Any]]:
        """Get engagement record by ID"""
        ...
    
    async def list_engagements(
        self,
        limit: int = 50,
        offset: int = 0,
        status_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List engagements with pagination"""
        ...
    
    async def update_engagement(
        self,
        engagement_id: UUID,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update engagement record"""
        ...
    
    async def get_mental_models_by_relevance(
        self,
        problem_context: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get mental models ranked by relevance to problem context"""
        ...
    
    async def create_analysis_record(
        self,
        engagement_id: UUID,
        analysis_data: Dict[str, Any]
    ) -> str:
        """Create analysis record and return analysis ID"""
        ...
    
    async def get_transparency_layers(
        self,
        engagement_id: UUID
    ) -> List[Dict[str, Any]]:
        """Get transparency layers for engagement"""
        ...
    
    async def check_database_health(self) -> Dict[str, Any]:
        """Check database connection and performance health"""
        ...


class IFoundationValidationService(Protocol):
    """
    Validation service for Foundation API business rules and constraints.
    
    Responsibilities:
    - Engagement creation and update validation
    - Cognitive analysis request validation
    - Mental model selection validation
    - Access control and security validation
    """
    
    async def validate_engagement_create(
        self,
        request: EngagementCreateRequest,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate engagement creation request"""
        ...
    
    async def validate_cognitive_analysis_request(
        self,
        request: CognitiveAnalysisRequest,
        engagement_id: str
    ) -> Dict[str, Any]:
        """Validate cognitive analysis request"""
        ...
    
    async def validate_engagement_access(
        self,
        engagement_id: str,
        user_id: Optional[str] = None,
        action: str = "read"
    ) -> Dict[str, Any]:
        """Validate user access to engagement"""
        ...
    
    async def validate_mental_model_selection(
        self,
        selected_models: List[str],
        problem_context: str
    ) -> Dict[str, Any]:
        """Validate mental model selection for given context"""
        ...
    
    async def sanitize_engagement_id(
        self,
        engagement_id: str
    ) -> UUID:
        """Sanitize and convert engagement ID to UUID"""
        ...


class IFoundationAnalyticsService(Protocol):
    """
    Analytics service for Foundation API metrics and performance tracking.
    
    Responsibilities:
    - Processing time and performance metrics
    - Model effectiveness scoring and analysis
    - Health status determination and monitoring
    - Engagement analytics and reporting
    """
    
    async def calculate_processing_metrics(
        self,
        start_time: float,
        end_time: float,
        operation_type: str
    ) -> Dict[str, Any]:
        """Calculate processing time and performance metrics"""
        ...
    
    async def calculate_model_effectiveness_scores(
        self,
        models: List[Dict[str, Any]],
        problem_context: str
    ) -> Dict[str, Any]:
        """Calculate effectiveness scores for mental models"""
        ...
    
    async def determine_engagement_health(
        self,
        engagement_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine health status of engagement"""
        ...
    
    async def generate_engagement_analytics(
        self,
        engagement_id: UUID,
        analysis_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive analytics for engagement"""
        ...
    
    async def calculate_system_health_metrics(
        self,
        database_health: Dict[str, Any],
        performance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate overall system health metrics"""
        ...


class IFoundationOrchestrationService(Protocol):
    """
    Orchestration service for Foundation API complex workflows.
    
    Responsibilities:
    - Cognitive analysis workflow coordination
    - Multi-step engagement processing
    - External service integration coordination
    - Transparency layer generation orchestration
    """
    
    async def orchestrate_engagement_creation(
        self,
        request: EngagementCreateRequest,
        user_id: Optional[str] = None
    ) -> EngagementResponse:
        """Orchestrate complete engagement creation workflow"""
        ...
    
    async def orchestrate_cognitive_analysis(
        self,
        engagement_id: str,
        request: CognitiveAnalysisRequest
    ) -> CognitiveAnalysisResponse:
        """Orchestrate complete cognitive analysis workflow"""
        ...
    
    async def orchestrate_transparency_layers_generation(
        self,
        engagement_id: str,
        analysis_data: Dict[str, Any]
    ) -> TransparencyLayersResponse:
        """Orchestrate transparency layers generation workflow"""
        ...
    
    async def orchestrate_mental_model_selection(
        self,
        problem_statement: str,
        force_selection: Optional[List[str]] = None,
        preferences: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Orchestrate mental model selection workflow"""
        ...
    
    async def orchestrate_engagement_listing(
        self,
        limit: int = 50,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None
    ) -> EngagementListResponse:
        """Orchestrate engagement listing with analytics"""
        ...


# ============================================================
# Service Factory Protocol
# ============================================================

class IFoundationServiceFactory(Protocol):
    """Factory for creating Foundation API service instances with dependency injection"""
    
    def create_repository_service(self) -> IFoundationRepositoryService:
        """Create repository service instance"""
        ...
    
    def create_validation_service(self) -> IFoundationValidationService:
        """Create validation service instance"""
        ...
    
    def create_analytics_service(self) -> IFoundationAnalyticsService:
        """Create analytics service instance"""
        ...
    
    def create_orchestration_service(
        self,
        repository: IFoundationRepositoryService,
        validation: IFoundationValidationService,
        analytics: IFoundationAnalyticsService
    ) -> IFoundationOrchestrationService:
        """Create orchestration service with injected dependencies"""
        ...


# ============================================================
# Foundation API Service Protocol (Main Interface)
# ============================================================

class IFoundationAPIService(Protocol):
    """
    Foundation API Service Protocol
    
    Defines the interface for all Foundation API business logic operations.
    This protocol will be implemented by the concrete service classes and
    enables clean dependency injection in the API layer.
    """
    
    # Core Engagement Operations
    async def create_engagement(
        self,
        request: EngagementCreateRequest,
        user_id: Optional[str] = None
    ) -> EngagementResponse:
        """Create a new engagement"""
        ...
    
    async def get_engagement(
        self,
        engagement_id: str,
        user_id: Optional[str] = None
    ) -> EngagementResponse:
        """Get engagement by ID"""
        ...
    
    async def list_engagements(
        self,
        limit: int = 50,
        offset: int = 0,
        status_filter: Optional[str] = None
    ) -> EngagementListResponse:
        """List engagements with pagination"""
        ...
    
    # Cognitive Analysis Operations
    async def execute_cognitive_analysis(
        self,
        engagement_id: str,
        request: CognitiveAnalysisRequest
    ) -> CognitiveAnalysisResponse:
        """Execute cognitive analysis for engagement"""
        ...
    
    async def get_transparency_layers(
        self,
        engagement_id: str
    ) -> TransparencyLayersResponse:
        """Get transparency layers for engagement"""
        ...
    
    # Mental Model Operations
    async def list_mental_models(
        self,
        problem_context: Optional[str] = None,
        limit: int = 50
    ) -> ModelListResponse:
        """List mental models with relevance scoring"""
        ...
    
    # System Operations
    async def get_health_status(self) -> DatabaseHealthResponse:
        """Get system health status"""
        ...
    
    # Report Generation
    async def generate_engagement_report(
        self,
        engagement_id: str
    ) -> Dict[str, Any]:
        """Generate comprehensive engagement report"""
        ...


# ============================================================
# Workflow Context and Results (for Orchestration)
# ============================================================

from dataclasses import dataclass

@dataclass
class FoundationWorkflowContext:
    """Context for Foundation API workflow operations"""
    workflow_id: str
    trace_id: str
    started_at: datetime
    user_id: Optional[str] = None
    parameters: Dict[str, Any] = None


@dataclass
class FoundationWorkflowResult:
    """Result of Foundation API workflow operation"""
    workflow_id: str
    status: str
    result: Dict[str, Any]
    execution_time_ms: int
    errors: List[str]


# ============================================================
# Exception Hierarchy for Foundation Services
# ============================================================

class FoundationServiceError(Exception):
    """Base exception for Foundation service errors"""
    def __init__(self, message: str, code: str = "FOUNDATION_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


class EngagementNotFoundError(FoundationServiceError):
    """Engagement not found error"""
    pass


class EngagementValidationError(FoundationServiceError):
    """Engagement validation error"""
    pass


class CognitiveAnalysisError(FoundationServiceError):
    """Cognitive analysis processing error"""
    pass


class MentalModelError(FoundationServiceError):
    """Mental model processing error"""
    pass


class DatabaseConnectionError(FoundationServiceError):
    """Database connection error"""
    pass


class ExternalServiceError(FoundationServiceError):
    """External service integration error"""
    pass