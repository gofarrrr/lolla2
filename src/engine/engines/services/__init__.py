"""
METIS V5 Engine Services
=======================

Business logic services extracted from the monolithic optimal_consultant_engine.py.
Part of the Great Refactoring: Clean, focused, stateless services.
"""

from .semantic_cluster_matcher import (
    SemanticClusterMatchingService,
    get_semantic_cluster_matching_service,
)
from .database_adapter import DatabaseAdapterService, get_database_adapter_service
from .state_manager import (
    StateManagementService,
    get_state_management_service,
    EngagementPhase,
)

__all__ = [
    "SemanticClusterMatchingService",
    "get_semantic_cluster_matching_service",
    "DatabaseAdapterService",
    "get_database_adapter_service",
    "StateManagementService",
    "get_state_management_service",
    "EngagementPhase",
]
