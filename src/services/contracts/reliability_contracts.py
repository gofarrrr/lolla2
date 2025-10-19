"""
METIS Reliability Services Contracts
Standardized data contracts and interfaces for all reliability services

Part of Phase 5 modular architecture - clean service boundaries and contracts.
"""

import abc
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from dataclasses import dataclass


# ============================================================
# ENUMS AND CLASSIFICATIONS
# ============================================================


class ConfidenceClassification(str, Enum):
    HIGH_CONFIDENCE = "HIGH_CONFIDENCE"
    MEDIUM_CONFIDENCE = "MEDIUM_CONFIDENCE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    INSUFFICIENT_CONFIDENCE = "INSUFFICIENT_CONFIDENCE"


class FailureMode(str, Enum):
    LOW_CONFIDENCE_ANALYSIS = "low_confidence_analysis"
    CONTRADICTORY_EVIDENCE = "contradictory_evidence"
    DEVILS_ADVOCATE_REJECTION = "devils_advocate_rejection"
    RESEARCH_GAPS = "research_gaps"
    LLM_VALIDATION_FAILURE = "llm_validation_failure"
    DATA_QUALITY_ISSUES = "data_quality_issues"


class RecommendationStatus(str, Enum):
    RECOMMENDED = "RECOMMENDED"
    CONDITIONAL_WITH_CAVEATS = "CONDITIONAL_WITH_CAVEATS"
    NOT_RECOMMENDED = "NOT_RECOMMENDED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class ExplorationStrategy(str, Enum):
    NOVEL_MODEL_INJECTION = "novel_model_injection"
    HYBRID_SYNTHESIS = "hybrid_synthesis"
    CROSS_INDUSTRY_TRANSFER = "cross_industry_transfer"
    MUTATION_TESTING = "mutation_testing"


class FeedbackTier(str, Enum):
    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"


class PartnershipTier(str, Enum):
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"


class ValidationLayer(str, Enum):
    FACTUAL_CONSISTENCY = "factual_consistency"
    LOGICAL_COHERENCE = "logical_coherence"
    RESEARCH_TRIANGULATION = "research_triangulation"
    CONFIDENCE_CALIBRATION = "confidence_calibration"
    CROSS_PROVIDER_VALIDATION = "cross_provider_validation"


class PatternStatus(str, Enum):
    DISCOVERED = "DISCOVERED"
    SANDBOX_TESTING = "SANDBOX_TESTING"
    PEER_REVIEW = "PEER_REVIEW"
    PRODUCTION = "PRODUCTION"
    DEPRECATED = "DEPRECATED"


# ============================================================
# DATA CONTRACTS
# ============================================================


@dataclass
class FailureAnalysisContract:
    """Contract for failure detection analysis results"""

    engagement_id: str
    failure_modes: List[str]
    confidence_classification: str
    recommendation_status: str
    limitations: List[Dict[str, Any]]
    partial_insights: List[Dict[str, Any]]
    recovery_steps: List[str]
    alternative_approaches: List[str]
    user_value_provided: str
    analysis_timestamp: datetime
    service_version: str


@dataclass
class ExplorationDecisionContract:
    """Contract for exploration vs exploitation decisions"""

    engagement_id: str
    should_explore: bool
    exploration_strategy: Optional[str]
    exploration_rationale: str
    exploration_models: List[str]
    confidence_adjustment: float
    decision_timestamp: datetime
    service_version: str


@dataclass
class FeedbackRequestContract:
    """Contract for multi-tier feedback requests"""

    engagement_id: str
    feedback_tier: str
    questions: List[str]
    incentive_offered: Dict[str, Any]
    deadline: datetime
    estimated_completion_minutes: int
    request_timestamp: datetime
    service_version: str


@dataclass
class ValidationResultContract:
    """Contract for LLM validation results"""

    engagement_id: str
    validation_layers: List[str]
    overall_passed: bool
    layer_results: List[Dict[str, Any]]
    issues_detected: List[str]
    evidence_collected: List[str]
    validation_timestamp: datetime
    service_version: str


@dataclass
class EmergentPatternContract:
    """Contract for emergent pattern discovery and governance"""

    pattern_id: str
    pattern_name: str
    discovery_engagement_id: str
    pattern_status: str
    confidence_score: float
    supporting_cases: List[str]
    validation_metrics: Dict[str, float]
    governance_stage: str
    discovery_timestamp: datetime
    service_version: str


@dataclass
class ReliabilityAssessmentContract:
    """Master contract for coordinated reliability assessment"""

    engagement_id: str
    failure_analysis: FailureAnalysisContract
    exploration_decision: ExplorationDecisionContract
    feedback_requests: List[FeedbackRequestContract]
    validation_results: List[ValidationResultContract]
    pattern_discovery: Optional[EmergentPatternContract]
    overall_reliability_score: float
    recommendations: List[str]
    assessment_timestamp: datetime
    service_version: str


@dataclass
class ServiceHealthContract:
    """Standard service health contract"""

    service_name: str
    status: str
    version: str
    capabilities: List[str]
    last_health_check: str
    error_count_24h: int = 0
    response_time_ms: float = 0.0


@dataclass
class ServiceResponseContract:
    """Standard service response wrapper"""

    success: bool
    data: Any
    error_message: Optional[str]
    processing_time_ms: float
    service_version: str


# ============================================================
# SERVICE INTERFACES
# ============================================================


class IFailureDetectionService(abc.ABC):
    """Interface for failure detection service"""

    @abc.abstractmethod
    async def analyze_engagement_health(
        self, cognitive_state: Dict[str, Any]
    ) -> FailureAnalysisContract:
        """Analyze engagement for failure modes and generate recovery strategies"""
        pass

    @abc.abstractmethod
    async def handle_component_failure(
        self, component_name: str, error: str, context: Any
    ) -> Dict[str, Any]:
        """Handle component failure with graceful degradation"""
        pass

    @abc.abstractmethod
    async def get_service_health(self) -> Dict[str, Any]:
        """Get service health status"""
        pass


class IExplorationStrategyService(abc.ABC):
    """Interface for exploration strategy service"""

    @abc.abstractmethod
    async def determine_exploration_strategy(
        self, problem_analysis: Dict[str, Any], business_context: Dict[str, Any]
    ) -> ExplorationDecisionContract:
        """Determine exploration vs exploitation strategy"""
        pass

    @abc.abstractmethod
    async def record_exploration_outcome(
        self, exploration_decision: ExplorationDecisionContract, outcome: Dict[str, Any]
    ) -> None:
        """Record exploration outcome for learning"""
        pass


class IFeedbackOrchestrationService(abc.ABC):
    """Interface for feedback orchestration service"""

    @abc.abstractmethod
    async def generate_feedback_requests(
        self, engagement_id: str, user_context: Dict[str, Any]
    ) -> List[FeedbackRequestContract]:
        """Generate multi-tier feedback requests with appropriate incentives"""
        pass

    @abc.abstractmethod
    async def determine_partnership_tier(
        self, user_feedback_history: Dict[str, Any]
    ) -> str:
        """Determine user's partnership tier based on feedback history"""
        pass


class IValidationEngineService(abc.ABC):
    """Interface for validation engine service"""

    @abc.abstractmethod
    async def validate_llm_output(
        self,
        llm_response: str,
        context: Dict[str, Any],
        research_base: List[Dict[str, Any]],
    ) -> ValidationResultContract:
        """Comprehensive multi-layer validation of LLM output"""
        pass

    @abc.abstractmethod
    async def validate_research_findings(
        self, research_intelligence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate research findings for accuracy and reliability"""
        pass


class IPatternGovernanceService(abc.ABC):
    """Interface for pattern governance service"""

    @abc.abstractmethod
    async def evaluate_for_pattern_discovery(
        self, engagement_data: Dict[str, Any], similar_engagements: List[Dict[str, Any]]
    ) -> Optional[EmergentPatternContract]:
        """Evaluate engagement for emergent pattern discovery"""
        pass

    @abc.abstractmethod
    async def advance_pattern_through_governance(self, pattern_id: str) -> bool:
        """Advance pattern through governance workflow"""
        pass


class IReliabilityCoordinatorService(abc.ABC):
    """Interface for reliability coordinator service"""

    @abc.abstractmethod
    async def assess_engagement_reliability(
        self, engagement_data: Dict[str, Any]
    ) -> ReliabilityAssessmentContract:
        """Coordinate all reliability services for comprehensive assessment"""
        pass

    @abc.abstractmethod
    async def get_cluster_health(self) -> Dict[str, Any]:
        """Get health status of entire reliability services cluster"""
        pass


# ============================================================
# UTILITY CONTRACTS
# ============================================================


@dataclass
class FeedbackIncentive:
    """Incentive structure for feedback provision"""

    tier: PartnershipTier
    credits_earned: int
    discount_percentage: float
    premium_access: bool
    co_innovation_access: bool
    revenue_sharing: bool = False


@dataclass
class ValidationResult:
    """Result of individual validation layer"""

    layer: ValidationLayer
    passed: bool
    confidence_score: float
    issues_detected: List[str]
    evidence: List[str]


# Alias for backwards compatibility
ValidationResultsContract = ValidationResult


@dataclass
class FeedbackOrchestrationContract:
    """Contract for feedback orchestration service"""

    engagement_id: str
    feedback_requests: List[Dict[str, Any]]
    partnership_tier: str
    incentives: List[Dict[str, Any]]
    orchestration_timestamp: datetime


@dataclass
class EmergentPattern:
    """Represents a discovered emergent pattern"""

    pattern_id: str
    name: str
    description: str
    discovery_engagement_id: str
    discovery_context: Dict[str, Any]
    supporting_cases: List[str]
    confidence_score: float
    validation_metrics: Dict[str, float]
    status: PatternStatus
    created_at: datetime
    version: str = "1.0"
