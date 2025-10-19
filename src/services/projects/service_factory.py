"""
Project Service Factory
=======================

Operation Chimera Phase 2 - Service Wiring

This factory implements dependency injection for the specialized project services.
It provides centralized service creation with proper dependency wiring following
the Inversion of Control (IoC) pattern.

Key Responsibilities:
- Service instantiation with dependency injection
- Lifecycle management for service instances
- Configuration management and validation
- Health monitoring and service discovery
"""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .specialized_contracts import (
    IProjectRepositoryService,
    IProjectValidationService, 
    IProjectAnalyticsService,
    IProjectOrchestrationService,
    IProjectServiceFactory,
)
from .repository_service import ProjectRepositoryService
from .validation_service import ProjectValidationService
from .analytics_service import ProjectAnalyticsService
from .orchestration_service import ProjectOrchestrationService
from src.core.unified_context_stream import get_unified_context_stream, UnifiedContextStream
from src.core.supabase_platform import get_supabase_client


@dataclass
class ProjectServiceConfig:
    """Configuration for project services"""
    enable_observability: bool = True
    enable_analytics: bool = True
    enable_validation: bool = True
    database_timeout_seconds: int = 30
    analytics_cache_ttl_seconds: int = 300
    validation_strict_mode: bool = True
    orchestration_max_retries: int = 3


class ProjectServiceFactory(IProjectServiceFactory):
    """
    Project Service Factory Implementation
    
    Provides dependency injection and service lifecycle management for
    all project-related services. Implements the Factory pattern with
    IoC container capabilities.
    """
    
    def __init__(
        self,
        config: Optional[ProjectServiceConfig] = None,
        context_stream: Optional[UnifiedContextStream] = None
    ):
        """Initialize service factory with configuration"""
        self.logger = logging.getLogger(__name__)
        self.config = config or ProjectServiceConfig()
        self.context_stream = context_stream or get_unified_context_stream()
        
        # Service instance cache
        self._repository_service: Optional[IProjectRepositoryService] = None
        self._validation_service: Optional[IProjectValidationService] = None
        self._analytics_service: Optional[IProjectAnalyticsService] = None
        self._orchestration_service: Optional[IProjectOrchestrationService] = None
        
        # Health status tracking
        self._service_health = {}
        
        self.logger.info("🏭 ProjectServiceFactory initialized with dependency injection")
    
    def create_repository_service(self) -> IProjectRepositoryService:
        """Create repository service instance with database dependencies"""
        if self._repository_service is None:
            self.logger.debug("Creating new ProjectRepositoryService instance")
            
            # Get database client
            db_client = get_supabase_client()
            
            # Create service with dependencies
            self._repository_service = ProjectRepositoryService(
                db_client=db_client,
                context_stream=self.context_stream,
                timeout_seconds=self.config.database_timeout_seconds
            )
            
            # Track service health
            self._service_health["repository"] = {"status": "healthy", "created_at": "now"}
            
            self.logger.info("✅ ProjectRepositoryService created and cached")
        
        return self._repository_service
    
    def create_validation_service(self) -> IProjectValidationService:
        """Create validation service instance"""
        if self._validation_service is None:
            self.logger.debug("Creating new ProjectValidationService instance")
            
            # Create service with configuration
            self._validation_service = ProjectValidationService(
                context_stream=self.context_stream,
                strict_mode=self.config.validation_strict_mode,
                enable_security_checks=True
            )
            
            # Track service health
            self._service_health["validation"] = {"status": "healthy", "created_at": "now"}
            
            self.logger.info("✅ ProjectValidationService created and cached")
        
        return self._validation_service
    
    def create_analytics_service(self) -> IProjectAnalyticsService:
        """Create analytics service instance"""
        if self._analytics_service is None:
            self.logger.debug("Creating new ProjectAnalyticsService instance")
            
            # Create service with configuration
            self._analytics_service = ProjectAnalyticsService(
                context_stream=self.context_stream,
                cache_ttl_seconds=self.config.analytics_cache_ttl_seconds,
                enable_benchmarking=True
            )
            
            # Track service health
            self._service_health["analytics"] = {"status": "healthy", "created_at": "now"}
            
            self.logger.info("✅ ProjectAnalyticsService created and cached")
        
        return self._analytics_service
    
    def create_orchestration_service(
        self,
        repository: Optional[IProjectRepositoryService] = None,
        validation: Optional[IProjectValidationService] = None,
        analytics: Optional[IProjectAnalyticsService] = None
    ) -> IProjectOrchestrationService:
        """Create orchestration service with injected dependencies"""
        if self._orchestration_service is None:
            self.logger.debug("Creating new ProjectOrchestrationService instance")
            
            # Use provided dependencies or create new ones
            repository_service = repository or self.create_repository_service()
            validation_service = validation or self.create_validation_service()
            analytics_service = analytics or self.create_analytics_service()
            
            # Create service with all dependencies
            self._orchestration_service = ProjectOrchestrationService(
                repository=repository_service,
                validator=validation_service,
                analytics=analytics_service,
                context_stream=self.context_stream
            )
            
            # Track service health
            self._service_health["orchestration"] = {"status": "healthy", "created_at": "now"}
            
            self.logger.info("✅ ProjectOrchestrationService created with dependency injection")
        
        return self._orchestration_service
    
    def create_complete_service_stack(self) -> Dict[str, Any]:
        """Create and wire complete service stack with all dependencies"""
        self.logger.info("🔧 Creating complete project service stack")
        
        # Create services in dependency order
        repository_service = self.create_repository_service()
        validation_service = self.create_validation_service()
        analytics_service = self.create_analytics_service()
        orchestration_service = self.create_orchestration_service(
            repository=repository_service,
            validation=validation_service,
            analytics=analytics_service
        )
        
        service_stack = {
            "repository": repository_service,
            "validation": validation_service,
            "analytics": analytics_service,
            "orchestration": orchestration_service,
        }
        
        # Validate service stack
        self._validate_service_stack(service_stack)
        
        self.logger.info("✅ Complete project service stack created and validated")
        
        return service_stack
    
    def get_service_health(self) -> Dict[str, Any]:
        """Get health status of all services"""
        health_report = {
            "factory_status": "healthy",
            "services": self._service_health.copy(),
            "total_services": len(self._service_health),
            "healthy_services": len([
                s for s in self._service_health.values() 
                if s.get("status") == "healthy"
            ]),
        }
        
        # Add configuration health
        health_report["configuration"] = {
            "observability_enabled": self.config.enable_observability,
            "analytics_enabled": self.config.enable_analytics,
            "validation_enabled": self.config.enable_validation,
            "strict_mode": self.config.validation_strict_mode,
        }
        
        return health_report
    
    def reset_services(self) -> None:
        """Reset all cached service instances (for testing/debugging)"""
        self.logger.warning("🔄 Resetting all cached service instances")
        
        self._repository_service = None
        self._validation_service = None
        self._analytics_service = None
        self._orchestration_service = None
        self._service_health.clear()
        
        self.logger.info("✅ All service instances reset")
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate factory configuration"""
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
        }
        
        # Validate timeouts
        if self.config.database_timeout_seconds <= 0:
            validation_result["errors"].append("Database timeout must be positive")
            validation_result["is_valid"] = False
        
        if self.config.analytics_cache_ttl_seconds < 0:
            validation_result["errors"].append("Analytics cache TTL cannot be negative")
            validation_result["is_valid"] = False
        
        if self.config.orchestration_max_retries < 0:
            validation_result["errors"].append("Orchestration max retries cannot be negative")
            validation_result["is_valid"] = False
        
        # Validate feature flags
        if not self.config.enable_validation:
            validation_result["warnings"].append("Validation is disabled - security risks may increase")
        
        if not self.config.enable_analytics:
            validation_result["warnings"].append("Analytics is disabled - monitoring capabilities reduced")
        
        return validation_result
    
    # ============================================================
    # Private Helper Methods
    # ============================================================
    
    def _validate_service_stack(self, service_stack: Dict[str, Any]) -> None:
        """Validate that service stack is properly configured"""
        required_services = ["repository", "validation", "analytics", "orchestration"]
        
        for service_name in required_services:
            if service_name not in service_stack:
                raise ValueError(f"Missing required service: {service_name}")
            
            service = service_stack[service_name]
            if service is None:
                raise ValueError(f"Service {service_name} is None")
        
        # Validate dependency relationships
        orchestration = service_stack["orchestration"]
        if not hasattr(orchestration, 'repository'):
            raise ValueError("Orchestration service missing repository dependency")
        
        if not hasattr(orchestration, 'validation'):
            raise ValueError("Orchestration service missing validation dependency")
        
        if not hasattr(orchestration, 'analytics'):
            raise ValueError("Orchestration service missing analytics dependency")
        
        self.logger.debug("✅ Service stack validation passed")


# ============================================================
# Global Factory Instance and Convenience Functions
# ============================================================

# Global factory instance (singleton pattern)
_global_factory: Optional[ProjectServiceFactory] = None


def get_project_service_factory(
    config: Optional[ProjectServiceConfig] = None,
    context_stream: Optional[UnifiedContextStream] = None,
    force_new: bool = False
) -> ProjectServiceFactory:
    """Get global project service factory instance"""
    global _global_factory
    
    if _global_factory is None or force_new:
        _global_factory = ProjectServiceFactory(
            config=config,
            context_stream=context_stream
        )
    
    return _global_factory


def create_project_services() -> Dict[str, Any]:
    """Convenience function to create complete service stack"""
    factory = get_project_service_factory()
    return factory.create_complete_service_stack()


def get_repository_service() -> IProjectRepositoryService:
    """Convenience function to get repository service"""
    factory = get_project_service_factory()
    return factory.create_repository_service()


def get_validation_service() -> IProjectValidationService:
    """Convenience function to get validation service"""
    factory = get_project_service_factory()
    return factory.create_validation_service()


def get_analytics_service() -> IProjectAnalyticsService:
    """Convenience function to get analytics service"""
    factory = get_project_service_factory()
    return factory.create_analytics_service()


def get_orchestration_service() -> IProjectOrchestrationService:
    """Convenience function to get orchestration service"""
    factory = get_project_service_factory()
    return factory.create_orchestration_service()


# ============================================================
# Service Health Monitoring
# ============================================================

async def check_service_health() -> Dict[str, Any]:
    """Check health of all project services"""
    factory = get_project_service_factory()
    health_report = factory.get_service_health()
    
    # Add runtime health checks
    services = factory.create_complete_service_stack()
    
    for service_name, service in services.items():
        try:
            # Basic health check - ensure service can be called
            if hasattr(service, 'get_health'):
                service_health = await service.get_health()
                health_report["services"][service_name]["runtime_health"] = service_health
            else:
                health_report["services"][service_name]["runtime_health"] = "no_health_endpoint"
        except Exception as e:
            health_report["services"][service_name]["runtime_health"] = f"error: {str(e)}"
            health_report["services"][service_name]["status"] = "unhealthy"
    
    return health_report


# ============================================================
# Configuration Templates
# ============================================================

def get_development_config() -> ProjectServiceConfig:
    """Get configuration optimized for development"""
    return ProjectServiceConfig(
        enable_observability=True,
        enable_analytics=True,
        enable_validation=True,
        database_timeout_seconds=10,
        analytics_cache_ttl_seconds=60,
        validation_strict_mode=False,
        orchestration_max_retries=1
    )


def get_production_config() -> ProjectServiceConfig:
    """Get configuration optimized for production"""
    return ProjectServiceConfig(
        enable_observability=True,
        enable_analytics=True,
        enable_validation=True,
        database_timeout_seconds=30,
        analytics_cache_ttl_seconds=300,
        validation_strict_mode=True,
        orchestration_max_retries=3
    )


def get_testing_config() -> ProjectServiceConfig:
    """Get configuration optimized for testing"""
    return ProjectServiceConfig(
        enable_observability=False,
        enable_analytics=False,
        enable_validation=True,
        database_timeout_seconds=5,
        analytics_cache_ttl_seconds=0,
        validation_strict_mode=True,
        orchestration_max_retries=0
    )
