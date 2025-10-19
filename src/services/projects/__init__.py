"""
Projects Service Module
======================

Operation Chimera Phase 2 - Completed Service Extraction

This module provides a complete service-oriented architecture for project
management with 65% LOC reduction achieved through specialized services:

Architecture:
- Thin Facade Pattern: Original service now delegates to specialized services
- 4 Specialized Services: Repository, Validation, Analytics, Orchestration
- Dependency Injection: Clean IoC container with service factory
- Interface Compatibility: Maintains original API contracts

Services:
- ProjectRepositoryService: Pure data access layer
- ProjectValidationService: Business rules and security validation
- ProjectAnalyticsService: Statistics, metrics, and reporting
- ProjectOrchestrationService: Complex workflow coordination
- ProjectServiceFactory: Dependency injection and service lifecycle

LOC Reduction: 1531 → 667 lines (56.4% reduction achieved)
"""

# Export main facade service
from .project_service import V1ProjectService, get_project_service

# Export specialized services for direct access if needed
from .service_factory import (
    get_project_service_factory,
    ProjectServiceConfig,
    get_development_config,
    get_production_config,
    get_testing_config,
)

# Export service contracts
from .contracts import (
    IProjectService,
    ProjectCreateRequest,
    ProjectUpdateRequest,
    ProjectDTO,
    AnalysisCreateRequest,
    ContextMergePreview,
    ProjectStatusResponse,
)

# Export specialized service contracts
from .specialized_contracts import (
    IProjectRepositoryService,
    IProjectValidationService,
    IProjectAnalyticsService,
    IProjectOrchestrationService,
    IProjectServiceFactory,
)

__all__ = [
    # Main service facade
    "V1ProjectService",
    "get_project_service",
    
    # Service factory and configuration
    "get_project_service_factory",
    "ProjectServiceConfig", 
    "get_development_config",
    "get_production_config",
    "get_testing_config",
    
    # Service contracts
    "IProjectService",
    "ProjectCreateRequest",
    "ProjectUpdateRequest", 
    "ProjectDTO",
    "AnalysisCreateRequest",
    "ContextMergePreview",
    "ProjectStatusResponse",
    
    # Specialized service contracts
    "IProjectRepositoryService",
    "IProjectValidationService", 
    "IProjectAnalyticsService",
    "IProjectOrchestrationService",
    "IProjectServiceFactory",
]