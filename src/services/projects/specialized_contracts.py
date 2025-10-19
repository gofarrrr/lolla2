"""
Project Service Specialized Contracts
====================================

Operation Chimera Phase 2 - Service Extraction Contracts

Defines the protocol interfaces for the 4 specialized services that will be
extracted from the monolithic V1ProjectService:

1. ProjectRepositoryService - Data access layer
2. ProjectValidationService - Business rules validation  
3. ProjectAnalyticsService - Analysis and reporting
4. ProjectOrchestrationService - Workflow coordination
"""

from typing import List, Optional, Dict, Any, Protocol
from uuid import UUID
from datetime import datetime

from .contracts import (
    ProjectCreateRequest,
    ProjectUpdateRequest, 
    ProjectDTO,
    AnalysisCreateRequest,
    ContextMergePreview,
    ProjectKnowledgeSearchRequest,
    MentalModelIngestionRequest,
    MentalModelIngestionResponse,
    ProjectStatusResponse,
)


# ============================================================
# Repository Service Contract - Data Access Layer
# ============================================================

class IProjectRepositoryService(Protocol):
    """
    Repository service for project data persistence operations.
    
    Responsibilities:
    - CRUD operations on projects table
    - Knowledge base data access
    - Analysis records management
    - Pure data access with no business logic
    """
    
    async def create_project_record(
        self, 
        project_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create project record in database"""
        ...
    
    async def get_project_record(
        self, 
        project_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get project record by ID"""
        ...
    
    async def list_project_records(
        self,
        organization_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List projects for organization with pagination"""
        ...
    
    async def update_project_record(
        self,
        project_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update project record"""
        ...
    
    async def delete_project_record(
        self,
        project_id: str
    ) -> bool:
        """Delete project record and cleanup related data"""
        ...
    
    async def get_project_knowledge_base(
        self,
        project_id: str
    ) -> List[Dict[str, Any]]:
        """Get knowledge base records for project"""
        ...
    
    async def store_analysis_record(
        self,
        analysis_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Store analysis record"""
        ...
    
    async def get_project_statistics_data(
        self,
        project_id: str
    ) -> Dict[str, Any]:
        """Get raw statistics data for project"""
        ...


# ============================================================
# Validation Service Contract - Business Rules
# ============================================================

class IProjectValidationService(Protocol):
    """
    Validation service for project business rules and constraints.
    
    Responsibilities:
    - Input validation beyond basic type checking
    - Business rule enforcement
    - Data consistency validation
    - Access control validation
    """
    
    async def validate_project_create(
        self,
        request: ProjectCreateRequest,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate project creation request and return validation results"""
        ...
    
    async def validate_project_update(
        self,
        project_id: str,
        request: ProjectUpdateRequest,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate project update request"""
        ...
    
    async def validate_project_access(
        self,
        project_id: str,
        user_id: Optional[str] = None,
        action: str = "read"
    ) -> Dict[str, Any]:
        """Validate user access to project for given action"""
        ...
    
    async def validate_analysis_request(
        self,
        request: AnalysisCreateRequest,
        project_id: str
    ) -> Dict[str, Any]:
        """Validate analysis creation request"""
        ...
    
    async def validate_mental_model_ingestion(
        self,
        request: MentalModelIngestionRequest,
        project_id: str
    ) -> Dict[str, Any]:
        """Validate mental model ingestion request"""
        ...


# ============================================================
# Analytics Service Contract - Analysis and Reporting  
# ============================================================

class IProjectAnalyticsService(Protocol):
    """
    Analytics service for project metrics, statistics, and reporting.
    
    Responsibilities:
    - Project statistics calculation
    - Performance metrics aggregation
    - Health status determination
    - Report generation
    """
    
    async def calculate_project_statistics(
        self,
        project_id: str,
        raw_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate comprehensive project statistics from raw data"""
        ...
    
    async def determine_project_status(
        self,
        project_id: str,
        project_data: Dict[str, Any]
    ) -> ProjectStatusResponse:
        """Determine overall project health and status"""
        ...
    
    async def calculate_knowledge_base_metrics(
        self,
        knowledge_records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate knowledge base metrics"""
        ...
    
    async def generate_service_health_report(
        self,
        service_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive service health report"""
        ...
    
    async def calculate_cqa_benchmark_scores(
        self,
        benchmark_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate CQA benchmark scores and analytics"""
        ...


# ============================================================
# Orchestration Service Contract - Workflow Coordination
# ============================================================

class IProjectOrchestrationService(Protocol):
    """
    Orchestration service for complex project workflows and coordination.
    
    Responsibilities:
    - Multi-step workflow coordination
    - External service integration
    - Context merging orchestration
    - Document processing coordination
    """
    
    async def orchestrate_project_creation(
        self,
        request: ProjectCreateRequest,
        user_id: Optional[str] = None
    ) -> ProjectDTO:
        """Orchestrate complete project creation workflow"""
        ...
    
    async def orchestrate_analysis_creation(
        self,
        request: AnalysisCreateRequest,
        project_id: str
    ) -> Dict[str, Any]:
        """Orchestrate complex analysis creation workflow"""
        ...
    
    async def orchestrate_context_merge_preview(
        self,
        project_id: str,
        merge_request: Dict[str, Any]
    ) -> ContextMergePreview:
        """Orchestrate context merge preview generation"""
        ...
    
    async def orchestrate_mental_model_ingestion(
        self,
        request: MentalModelIngestionRequest,
        project_id: str
    ) -> MentalModelIngestionResponse:
        """Orchestrate mental model document ingestion workflow"""
        ...
    
    async def orchestrate_batch_cqa_evaluation(
        self,
        project_id: str,
        batch_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Orchestrate batch CQA evaluation workflow"""
        ...
    
    async def orchestrate_knowledge_search(
        self,
        request: ProjectKnowledgeSearchRequest,
        project_id: str
    ) -> List[Dict[str, Any]]:
        """Orchestrate knowledge base search workflow"""
        ...


# ============================================================
# Service Factory Contract
# ============================================================

class IProjectServiceFactory(Protocol):
    """Factory for creating project service instances with dependency injection"""
    
    def create_repository_service(self) -> IProjectRepositoryService:
        """Create repository service instance"""
        ...
    
    def create_validation_service(self) -> IProjectValidationService:
        """Create validation service instance"""
        ...
    
    def create_analytics_service(self) -> IProjectAnalyticsService:
        """Create analytics service instance"""
        ...
    
    def create_orchestration_service(
        self,
        repository: IProjectRepositoryService,
        validation: IProjectValidationService, 
        analytics: IProjectAnalyticsService
    ) -> IProjectOrchestrationService:
        """Create orchestration service with injected dependencies"""
        ...

# ============================================================
# Additional Types, Errors, Aliases, and Utilities (Phase 2 wiring)
# ============================================================

from dataclasses import dataclass
from typing import Optional, Callable, Any, List
from datetime import datetime

# Workflow context/result used by facade and orchestration service
@dataclass
class WorkflowContext:
    workflow_id: str
    trace_id: str
    started_at: datetime
    user_id: Optional[str] = None
    parameters: Dict[str, Any] = None

@dataclass
class WorkflowResult:
    result: Dict[str, Any]

# Validation result and analytics DTOs used across services
@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    details: Dict[str, Any]

@dataclass
class HealthMetrics:
    overall_health_score: float
    rag_health_status: str
    activity_score: float
    quality_score: float
    efficiency_score: float
    last_calculated: datetime

@dataclass
class UsageAnalytics:
    total_analyses: int
    recent_analyses_30d: int
    avg_tokens_per_analysis: int
    avg_cost_per_analysis: float
    total_cost: float
    roi_score: float
    efficiency_metrics: Dict[str, Any]

@dataclass
class ProjectStatistics:
    statistics: Dict[str, Any]

@dataclass
class ProjectRecord:
    data: Dict[str, Any]

# Project service error hierarchy
class ProjectServiceError(Exception):
    def __init__(self, message: str, code: str = "PROJECT_ERROR", details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}

class ProjectNotFoundError(ProjectServiceError):
    pass

class ProjectValidationError(ProjectServiceError):
    pass

class ProjectOrchestrationError(ProjectServiceError):
    pass

class ProjectAnalyticsError(ProjectServiceError):
    pass

class ProjectRepositoryError(ProjectServiceError):
    pass

# Backward-compatible aliases for repository_service expectations
DatabaseError = ProjectRepositoryError
NotFoundError = ProjectNotFoundError
ValidationError = ProjectValidationError

# Circuit breaker interfaces for error handler
@dataclass
class CircuitBreakerState:
    service_name: str
    state: str  # "closed" | "open" | "half_open"
    failure_count: int
    last_failure_time: Optional[datetime]
    next_attempt_time: Optional[datetime]

class ICircuitBreaker(Protocol):
    async def execute_with_breaker(self, operation_name: str, operation_func: Callable, *args, **kwargs) -> Any:
        ...

    async def get_breaker_state(self, service_name: str) -> CircuitBreakerState:
        ...

    async def reset_breaker(self, service_name: str) -> bool:
        ...

# Event mapping helper used by repository service
_DEF_EVENT_MAP = {
    ProjectNotFoundError: "not_found",
    ProjectValidationError: "validation_error",
    ProjectRepositoryError: "database_error",
    ProjectAnalyticsError: "analytics_error",
    ProjectOrchestrationError: "orchestration_error",
}

def map_exception_to_event_type(exc: Exception) -> str:
    for etype, name in _DEF_EVENT_MAP.items():
        if isinstance(exc, etype):
            return name
    return exc.__class__.__name__

# Protocol aliasing for compatibility with existing imports
IProjectRepository = IProjectRepositoryService
IProjectValidator = IProjectValidationService
IProjectAnalytics = IProjectAnalyticsService
IProjectOrchestrator = IProjectOrchestrationService
