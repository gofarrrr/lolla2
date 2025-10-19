"""
METIS Flywheel System - Phoenix Integration
Self-improving AI platform with multi-layer learning and continuous optimization

This module consolidates the sophisticated Flywheel system discovered across the codebase.
Components have been restored and organized for V5 integration.

Architecture:
- cache/: Multi-layer intelligent caching with learning integration
- learning/: Continuous learning from user interactions and feedback
- detection/: Phantom workflow detection and prevention
- monitoring/: Real-time metrics and health monitoring
- integration/: UltraThink and V5 system bridges
- orchestration/: Learning cycle coordination and management
- core/: System resilience and graceful degradation management
"""

from .cache.flywheel_cache_system import FlywheelCacheSystem, get_flywheel_cache
from .learning.learning_loop import LearningLoop, get_learning_loop
from .detection.phantom_workflow_detector import (
    PhantomWorkflowDetector,
    get_phantom_workflow_detector,
)

# UnifiedIntelligenceDashboard moved to src.engine.monitoring
# from .monitoring.unified_intelligence_dashboard import UnifiedIntelligenceDashboard, get_unified_intelligence_dashboard
from .core.graceful_degradation import (
    GracefulDegradationManager,
    get_degradation_manager,
    with_graceful_degradation,
)

__all__ = [
    "FlywheelCacheSystem",
    "get_flywheel_cache",
    "LearningLoop",
    "get_learning_loop",
    "PhantomWorkflowDetector",
    "get_phantom_workflow_detector",
    # UnifiedIntelligenceDashboard moved to src.engine.monitoring
    "GracefulDegradationManager",
    "get_degradation_manager",
    "with_graceful_degradation",
]
