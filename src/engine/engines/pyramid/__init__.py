"""
METIS Pyramid Principle Engine Package - Sprint 2.1 Enhanced
Implementing Barbara Minto's Pyramid Principle with Context Intelligence Revolution

Features:
- Traditional Pyramid Principle implementation
- Context-Intelligent Pyramid Builder using cognitive exhaust
- AI's thinking process for maximum persuasive impact
- Historical pattern recognition for argument optimization
"""

from .engine import PyramidEngine
from .models import PyramidNode, ExecutiveDeliverable
from .enums import PyramidLevel, ArgumentType, DeliverableType

# Sprint 2.1: Context Intelligence Integration
from .context_aware_builder import (
    ContextIntelligentPyramidBuilder,
    create_context_aware_pyramid_builder,
)
from .context_intelligent_engine import (
    ContextIntelligentPyramidEngine,
    create_context_intelligent_pyramid_engine,
)

# Sprint 2.2: Context-Aware Quality Assessment
from .context_aware_quality import (
    ContextAwareQualityAssessor,
    create_context_aware_quality_assessor,
)

# Sprint 2.3: Multi-Persona Deliverable Adaptation
from .multi_persona_formatter import (
    MultiPersonaDeliverableFormatter,
    create_multi_persona_formatter,
    ExecutivePersona,
)

__all__ = [
    # Traditional components
    "PyramidEngine",
    "PyramidNode",
    "ExecutiveDeliverable",
    "PyramidLevel",
    "ArgumentType",
    "DeliverableType",
    # Sprint 2.1: Context Intelligence Enhancement
    "ContextIntelligentPyramidBuilder",
    "ContextIntelligentPyramidEngine",
    "create_context_aware_pyramid_builder",
    "create_context_intelligent_pyramid_engine",
    # Sprint 2.2: Context-Aware Quality Assessment
    "ContextAwareQualityAssessor",
    "create_context_aware_quality_assessor",
    # Sprint 2.3: Multi-Persona Deliverable Adaptation
    "MultiPersonaDeliverableFormatter",
    "create_multi_persona_formatter",
    "ExecutivePersona",
]
