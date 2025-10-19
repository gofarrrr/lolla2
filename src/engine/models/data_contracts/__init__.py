"""
METIS Unified Data Contract Architecture
F001: CloudEvents-compliant data schema for all system components

This package provides comprehensive data contracts for the METIS V5.3 platform.
All exports maintain backward compatibility with the original data_contracts.py file.

Based on PRD v7 architectural specifications and N-WAY framework analysis.
"""

# ============================================================================
# ENUMS (14 total)
# ============================================================================

from .models.enums import (
    EngagementPhase,
    MentalModelCategory,
    ConfidenceLevel,
    VulnerabilityDetectionLevel,
    ExplorationDecision,
    ClarificationQuestionType,
    ClarificationComplexity,
    ContextType,
    ContextRelevanceLevel,
    CognitiveCacheLevel,
    StrategicLayer,
    CognitiveFunction,
    ExtendedConsultantRole,
    FeedbackTier,
)

# ============================================================================
# ENGAGEMENT MODELS (8 total)
# ============================================================================

from .models.engagement_models import (
    EngagementContext,
    WorkflowState,
    DeliverableArtifact,
    ClarificationQuestion,
    ClarificationResponse,
    ClarificationSession,
    ExplorationContext,
    FailureModeResponse,
)

# ============================================================================
# CONSULTANT MODELS (5 total)
# ============================================================================

from .models.consultant_models import (
    ConsultantSpecialization,
    ScoringWeights,
    ConsultantMatrixConfig,
    ConsultantSelectionInput,
    ConsultantSelectionResult,
)

# ============================================================================
# ANALYSIS MODELS (13 total)
# ============================================================================

from .models.analysis_models import (
    MentalModelDefinition,
    ContextElement,
    ContextRelevanceScore,
    ReasoningStep,
    ResearchIntelligence,
    CognitiveState,
    HallucinationCheck,
    ContextIntelligenceResult,
    FeedbackContext,
    VulnerabilityContext,
    FallbackBehavior,
    PatternGovernanceResult,
    MetisDataContract,
    get_schema_version,
)

# ============================================================================
# FACTORY FUNCTIONS (5 total)
# ============================================================================

from .factories.event_factory import (
    create_engagement_initiated_event,
    create_model_selection_event,
    create_vulnerability_assessment_event,
    create_exploration_strategy_event,
    create_hallucination_detection_event,
)

# ============================================================================
# VALIDATORS (1 total)
# ============================================================================

from .validators.contract_validators import (
    validate_data_contract_compliance,
)

# ============================================================================
# EXPLICIT EXPORTS (__all__)
# ============================================================================

__all__ = [
    # Enums
    "EngagementPhase",
    "MentalModelCategory",
    "ConfidenceLevel",
    "VulnerabilityDetectionLevel",
    "ExplorationDecision",
    "ClarificationQuestionType",
    "ClarificationComplexity",
    "ContextType",
    "ContextRelevanceLevel",
    "CognitiveCacheLevel",
    "StrategicLayer",
    "CognitiveFunction",
    "ExtendedConsultantRole",
    "FeedbackTier",
    # Engagement Models
    "EngagementContext",
    "WorkflowState",
    "DeliverableArtifact",
    "ClarificationQuestion",
    "ClarificationResponse",
    "ClarificationSession",
    "ExplorationContext",
    "FailureModeResponse",
    # Consultant Models
    "ConsultantSpecialization",
    "ScoringWeights",
    "ConsultantMatrixConfig",
    "ConsultantSelectionInput",
    "ConsultantSelectionResult",
    # Analysis Models
    "MentalModelDefinition",
    "ContextElement",
    "ContextRelevanceScore",
    "ReasoningStep",
    "ResearchIntelligence",
    "CognitiveState",
    "HallucinationCheck",
    "ContextIntelligenceResult",
    "FeedbackContext",
    "VulnerabilityContext",
    "FallbackBehavior",
    "PatternGovernanceResult",
    "MetisDataContract",
    "get_schema_version",
    # Factory Functions
    "create_engagement_initiated_event",
    "create_model_selection_event",
    "create_vulnerability_assessment_event",
    "create_exploration_strategy_event",
    "create_hallucination_detection_event",
    # Validators
    "validate_data_contract_compliance",
]
