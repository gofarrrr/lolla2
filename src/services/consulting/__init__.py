"""
Consulting Services Package
C2 - Service Extraction & Batching Complete
"""
from .contracts import (
    IFrameworkRepository, IFrameworkEvaluator, ICacheProvider,
    PerformanceConfig, FrameworkMetadata, EvaluationResult, CacheConfig,
    ConsultingFrameworkError, PerformanceThresholdExceeded, 
    FrameworkNotFoundError, EvaluationTimeoutError
)
from .framework_repository_service import (
    ProductionFrameworkRepositoryService, InMemoryCacheProvider,
    FrameworkRepositoryServiceFactory, FrameworkCategory, FrameworkComplexity
)
from .framework_evaluator_service import (
    ProductionFrameworkEvaluatorService, FrameworkEvaluatorServiceFactory,
    EngagementPhase
)

__all__ = [
    # Contracts
    'IFrameworkRepository', 'IFrameworkEvaluator', 'ICacheProvider',
    'PerformanceConfig', 'FrameworkMetadata', 'EvaluationResult', 'CacheConfig',
    # Errors
    'ConsultingFrameworkError', 'PerformanceThresholdExceeded',
    'FrameworkNotFoundError', 'EvaluationTimeoutError',
    # Enums
    'FrameworkCategory', 'FrameworkComplexity', 'EngagementPhase',
    # Services
    'ProductionFrameworkRepositoryService', 'InMemoryCacheProvider',
    'FrameworkRepositoryServiceFactory',
    'ProductionFrameworkEvaluatorService', 'FrameworkEvaluatorServiceFactory'
]