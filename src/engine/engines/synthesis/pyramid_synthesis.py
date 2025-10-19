"""
DEPRECATED: Use src.engines.pyramid package instead
This file maintained for backward compatibility

Sprint 2.1 Update: Added Context Intelligence integration
"""

# Re-export everything from the refactored modules
try:
    from .pyramid.engine import PyramidEngine as PyramidPrincipleEngine  # Legacy alias
    from .pyramid.models import PyramidNode, ExecutiveDeliverable
    from .pyramid.enums import PyramidLevel, ArgumentType, DeliverableType

    # Sprint 2.1: Context Intelligence exports
    from .pyramid.context_intelligent_engine import (
        ContextIntelligentPyramidEngine,
        create_context_intelligent_pyramid_engine,
    )
    from .pyramid.context_aware_builder import (
        ContextIntelligentPyramidBuilder,
        create_context_aware_pyramid_builder,
    )

    CONTEXT_INTELLIGENCE_AVAILABLE = True

except ImportError:
    # Fallback for environments without Context Intelligence
    CONTEXT_INTELLIGENCE_AVAILABLE = False
    ContextIntelligentPyramidEngine = None
    create_context_intelligent_pyramid_engine = None

# Re-export for backward compatibility
__all__ = [
    # Traditional components
    "PyramidPrincipleEngine",  # Legacy alias
    "PyramidNode",
    "ExecutiveDeliverable",
    "PyramidLevel",
    "ArgumentType",
    "DeliverableType",
    # Sprint 2.1: Context Intelligence (if available)
    "ContextIntelligentPyramidEngine",
    "ContextIntelligentPyramidBuilder",
    "create_context_intelligent_pyramid_engine",
    "create_context_aware_pyramid_builder",
    "CONTEXT_INTELLIGENCE_AVAILABLE",
]
