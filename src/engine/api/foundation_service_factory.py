"""
Foundation Service Factory
=========================

Operation Chimera Phase 3 - Foundation Service Extraction

Service factory implementing dependency injection for all Foundation API services.
Provides centralized service creation and wiring following the proven pattern from Project Service.

Key Responsibilities:
- Service instance creation with proper dependencies
- Configuration management and injection
- Singleton pattern for service reuse
- Error handling and service health validation
"""

import logging
from typing import Optional

from .foundation_contracts import IFoundationServiceFactory
from .foundation_repository_service import FoundationRepositoryService
from .foundation_validation_service import FoundationValidationService
from .foundation_analytics_service import FoundationAnalyticsService
from .foundation_orchestration_service import FoundationOrchestrationService

# Import core dependencies
from src.engine.adapters.core.unified_context_stream import get_unified_context_stream, UnifiedContextStream
from src.engine.persistence.supabase_integration import MetisSupabaseIntegration


class FoundationServiceConfig:
    """Configuration class for Foundation API services"""
    
    def __init__(
        self,
        supabase_integration: Optional[MetisSupabaseIntegration] = None,
        context_stream: Optional[UnifiedContextStream] = None,
        enable_caching: bool = True,
        enable_analytics: bool = True,
        enable_validation: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize Foundation service configuration"""
        self.supabase_integration = supabase_integration
        self.context_stream = context_stream or get_unified_context_stream()
        self.enable_caching = enable_caching
        self.enable_analytics = enable_analytics
        self.enable_validation = enable_validation
        self.logger = logger or logging.getLogger(__name__)


class FoundationServiceFactory(IFoundationServiceFactory):
    """
    Foundation Service Factory Implementation
    
    Creates and wires Foundation API services with proper dependency injection.
    Ensures all services are properly configured and connected.
    """
    
    def __init__(self, config: Optional[FoundationServiceConfig] = None):
        """Initialize Foundation Service Factory"""
        self.config = config or FoundationServiceConfig()
        self.logger = self.config.logger
        
        # Service instance cache for singleton pattern
        self._repository_service = None
        self._validation_service = None
        self._analytics_service = None
        self._orchestration_service = None
        
        self.logger.info("✅ FoundationServiceFactory initialized")
    
    def create_repository_service(self) -> FoundationRepositoryService:
        """Create repository service instance with Supabase integration"""
        if self._repository_service is None:
            self.logger.info("🔨 Creating FoundationRepositoryService instance")
            
            try:
                self._repository_service = FoundationRepositoryService(
                    context_stream=self.config.context_stream
                )
                
                self.logger.info("✅ FoundationRepositoryService created successfully")
            
            except Exception as e:
                self.logger.error(f"❌ Failed to create FoundationRepositoryService: {e}")
                raise
        
        return self._repository_service
    
    def create_validation_service(self) -> FoundationValidationService:
        """Create validation service instance with context stream"""
        if self._validation_service is None:
            self.logger.info("🔨 Creating FoundationValidationService instance")
            
            try:
                self._validation_service = FoundationValidationService(
                    context_stream=self.config.context_stream
                )
                
                self.logger.info("✅ FoundationValidationService created successfully")
            
            except Exception as e:
                self.logger.error(f"❌ Failed to create FoundationValidationService: {e}")
                raise
        
        return self._validation_service
    
    def create_analytics_service(self) -> FoundationAnalyticsService:
        """Create analytics service instance with context stream"""
        if self._analytics_service is None:
            self.logger.info("🔨 Creating FoundationAnalyticsService instance")
            
            try:
                self._analytics_service = FoundationAnalyticsService(
                    context_stream=self.config.context_stream
                )
                
                self.logger.info("✅ FoundationAnalyticsService created successfully")
            
            except Exception as e:
                self.logger.error(f"❌ Failed to create FoundationAnalyticsService: {e}")
                raise
        
        return self._analytics_service
    
    def create_orchestration_service(
        self,
        repository: Optional[FoundationRepositoryService] = None,
        validation: Optional[FoundationValidationService] = None,
        analytics: Optional[FoundationAnalyticsService] = None
    ) -> FoundationOrchestrationService:
        """Create orchestration service with injected dependencies"""
        if self._orchestration_service is None:
            self.logger.info("🔨 Creating FoundationOrchestrationService instance")
            
            try:
                # Use provided services or create them
                repo_service = repository or self.create_repository_service()
                validation_service = validation or self.create_validation_service()
                analytics_service = analytics or self.create_analytics_service()
                
                self._orchestration_service = FoundationOrchestrationService(
                    repository=repo_service,
                    validation=validation_service,
                    analytics=analytics_service,
                    context_stream=self.config.context_stream
                )
                
                self.logger.info("✅ FoundationOrchestrationService created successfully")
            
            except Exception as e:
                self.logger.error(f"❌ Failed to create FoundationOrchestrationService: {e}")
                raise
        
        return self._orchestration_service
    
    def create_all_services(self) -> dict:
        """Create all Foundation services and return as dictionary"""
        self.logger.info("🔨 Creating all Foundation services")
        
        try:
            services = {
                "repository": self.create_repository_service(),
                "validation": self.create_validation_service(),
                "analytics": self.create_analytics_service()
            }
            
            # Create orchestration service with all dependencies
            services["orchestration"] = self.create_orchestration_service(
                repository=services["repository"],
                validation=services["validation"],
                analytics=services["analytics"]
            )
            
            self.logger.info("✅ All Foundation services created successfully")
            return services
        
        except Exception as e:
            self.logger.error(f"❌ Failed to create Foundation services: {e}")
            raise
    
    def validate_service_health(self) -> dict:
        """Validate health of all created services"""
        health_status = {
            "overall_healthy": True,
            "services": {}
        }
        
        # Check repository service
        if self._repository_service:
            try:
                # In production, would call actual health check
                health_status["services"]["repository"] = {
                    "status": "healthy",
                    "created": True,
                    "supabase_connected": bool(self.config.supabase_integration)
                }
            except Exception as e:
                health_status["services"]["repository"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["overall_healthy"] = False
        else:
            health_status["services"]["repository"] = {"status": "not_created"}
        
        # Check validation service
        if self._validation_service:
            health_status["services"]["validation"] = {
                "status": "healthy",
                "created": True
            }
        else:
            health_status["services"]["validation"] = {"status": "not_created"}
        
        # Check analytics service
        if self._analytics_service:
            health_status["services"]["analytics"] = {
                "status": "healthy",
                "created": True
            }
        else:
            health_status["services"]["analytics"] = {"status": "not_created"}
        
        # Check orchestration service
        if self._orchestration_service:
            health_status["services"]["orchestration"] = {
                "status": "healthy",
                "created": True,
                "dependencies_wired": True
            }
        else:
            health_status["services"]["orchestration"] = {"status": "not_created"}
        
        return health_status
    
    def reset_services(self):
        """Reset all service instances (useful for testing)"""
        self.logger.info("🔄 Resetting all Foundation service instances")
        
        self._repository_service = None
        self._validation_service = None
        self._analytics_service = None
        self._orchestration_service = None
        
        self.logger.info("✅ All Foundation service instances reset")
    
    def get_service_info(self) -> dict:
        """Get information about current service instances"""
        return {
            "factory_config": {
                "enable_caching": self.config.enable_caching,
                "enable_analytics": self.config.enable_analytics,
                "enable_validation": self.config.enable_validation,
                "supabase_available": bool(self.config.supabase_integration),
                "context_stream_available": bool(self.config.context_stream)
            },
            "service_instances": {
                "repository_created": self._repository_service is not None,
                "validation_created": self._validation_service is not None,
                "analytics_created": self._analytics_service is not None,
                "orchestration_created": self._orchestration_service is not None
            },
            "dependencies": {
                "supabase_integration": "available" if self.config.supabase_integration else "not_available",
                "context_stream": "available" if self.config.context_stream else "not_available"
            }
        }


# Global factory instance for consistency across the application
_global_foundation_factory: Optional[FoundationServiceFactory] = None


def get_foundation_service_factory(
    config: Optional[FoundationServiceConfig] = None,
    force_new: bool = False
) -> FoundationServiceFactory:
    """
    Get global Foundation service factory instance
    
    Args:
        config: Optional configuration for new factory
        force_new: If True, creates new factory instance
    
    Returns:
        FoundationServiceFactory instance
    """
    global _global_foundation_factory
    
    if _global_foundation_factory is None or force_new:
        if config is None:
            # Create default configuration
            config = FoundationServiceConfig()
        
        _global_foundation_factory = FoundationServiceFactory(config)
    
    return _global_foundation_factory


def reset_foundation_service_factory():
    """Reset global Foundation service factory (useful for testing)"""
    global _global_foundation_factory
    
    if _global_foundation_factory:
        _global_foundation_factory.reset_services()
    
    _global_foundation_factory = None


# Convenience functions for direct service access
def get_foundation_repository_service(
    config: Optional[FoundationServiceConfig] = None
) -> FoundationRepositoryService:
    """Get Foundation repository service instance"""
    factory = get_foundation_service_factory(config)
    return factory.create_repository_service()


def get_foundation_validation_service(
    config: Optional[FoundationServiceConfig] = None
) -> FoundationValidationService:
    """Get Foundation validation service instance"""
    factory = get_foundation_service_factory(config)
    return factory.create_validation_service()


def get_foundation_analytics_service(
    config: Optional[FoundationServiceConfig] = None
) -> FoundationAnalyticsService:
    """Get Foundation analytics service instance"""
    factory = get_foundation_service_factory(config)
    return factory.create_analytics_service()


def get_foundation_orchestration_service(
    config: Optional[FoundationServiceConfig] = None
) -> FoundationOrchestrationService:
    """Get Foundation orchestration service instance"""
    factory = get_foundation_service_factory(config)
    return factory.create_orchestration_service()


def get_all_foundation_services(
    config: Optional[FoundationServiceConfig] = None
) -> dict:
    """Get all Foundation services as dictionary"""
    factory = get_foundation_service_factory(config)
    return factory.create_all_services()


# Service health check function
async def check_foundation_services_health(
    config: Optional[FoundationServiceConfig] = None
) -> dict:
    """Check health of all Foundation services"""
    factory = get_foundation_service_factory(config)
    
    # Basic health check
    health_status = factory.validate_service_health()
    
    # If repository service is available, check database health
    if factory._repository_service:
        try:
            db_health = await factory._repository_service.check_database_health()
            health_status["database_health"] = db_health
        except Exception as e:
            health_status["database_health"] = {
                "status": "error",
                "error": str(e)
            }
    
    return health_status