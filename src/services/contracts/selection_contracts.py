"""
METIS Selection Services Contracts
Standardized data contracts and interfaces for all selection services

Part of Phase 5.2 modular architecture - clean service boundaries for model selection cluster.
"""

import abc
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from dataclasses import dataclass


# ============================================================
# ENUMS AND CLASSIFICATIONS
# ============================================================


class SelectionStrategy(str, Enum):
    PERFORMANCE_OPTIMIZED = "performance_optimized"
    COGNITIVE_BALANCED = "cognitive_balanced"
    DIVERSITY_FOCUSED = "diversity_focused"
    RISK_CONSERVATIVE = "risk_conservative"
    SPEED_OPTIMIZED = "speed_optimized"


class MergeStrategy(str, Enum):
    WEIGHTED_CONFIDENCE = "weighted_confidence"
    CONSENSUS_BOOSTING = "consensus_boosting"
    HYBRID_RANKING = "hybrid_ranking"


class NWayInteractionType(str, Enum):
    SYNERGISTIC = "synergistic"
    CONFLICTING = "conflicting"
    DEPENDENT = "dependent"
    INDEPENDENT = "independent"


class ModelComplexity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SelectionSource(str, Enum):
    DATABASE_ONLY = "database_only"
    ZERO_SHOT_ONLY = "zero_shot_only"
    HYBRID = "hybrid"


# ============================================================
# DATA CONTRACTS
# ============================================================


@dataclass
class ModelScoreContract:
    """Contract for model scoring results"""

    model_id: str
    total_score: float
    component_scores: Dict[str, float]
    rationale: str
    confidence: float
    risk_factors: List[str]
    scoring_timestamp: datetime
    service_version: str


@dataclass
class SelectionContextContract:
    """Contract for selection context information"""

    problem_statement: str
    business_context: Dict[str, Any]
    problem_type: str
    complexity_level: str
    accuracy_requirement: float
    max_models: int
    selection_strategy: str
    time_constraint: Optional[str] = None
    cognitive_load_limit: str = "medium"


@dataclass
class SelectionResultContract:
    """Contract for model selection results"""

    engagement_id: str
    selected_models: List[str]  # Model IDs
    model_scores: List[ModelScoreContract]
    selection_source: str
    strategy_used: str
    models_evaluated: int
    selection_metadata: Dict[str, Any]
    total_selection_time_ms: float
    cognitive_load_assessment: str
    selection_timestamp: datetime
    service_version: str


@dataclass
class NWayInteractionContract:
    """Contract for N-Way interaction patterns"""

    interaction_id: str
    source_model: str
    target_model: str
    relationship_type: str
    strength_score: float
    confidence_score: float
    explanation: str
    context_conditions: Dict[str, Any]
    instructional_cue: str
    enhancement_type: str
    validation_timestamp: datetime
    service_version: str


@dataclass
class BayesianUpdateContract:
    """Contract for Bayesian effectiveness updates"""

    model_id: str
    context_key: str
    effectiveness_score: float
    observations_count: int
    posterior_alpha: float
    posterior_beta: float
    context_specificity: Dict[str, Any]
    update_timestamp: datetime
    service_version: str


@dataclass
class ZeroShotSelectionContract:
    """Contract for zero-shot selection results"""

    engagement_id: str
    selected_models: List[str]
    confidence_score: float
    reasoning_process: List[str]
    context_analysis: Dict[str, Any]
    novelty_factors: List[str]
    selection_timestamp: datetime
    service_version: str


@dataclass
class SelectionCoordinationContract:
    """Master contract for coordinated selection results"""

    engagement_id: str
    selection_result: SelectionResultContract
    nway_interactions: List[NWayInteractionContract]
    bayesian_updates: List[BayesianUpdateContract]
    zero_shot_result: Optional[ZeroShotSelectionContract]
    coordination_metadata: Dict[str, Any]
    overall_confidence: float
    coordination_timestamp: datetime
    service_version: str


# ============================================================
# SERVICE INTERFACES
# ============================================================


class ISelectionStrategyService(abc.ABC):
    """Interface for selection strategy service"""

    @abc.abstractmethod
    async def execute_selection_strategy(
        self,
        strategy: SelectionStrategy,
        models: List[Any],
        scores: List[ModelScoreContract],
        context: SelectionContextContract,
    ) -> SelectionResultContract:
        """Execute specific selection strategy"""
        pass

    @abc.abstractmethod
    async def recommend_strategy(
        self, context: SelectionContextContract
    ) -> SelectionStrategy:
        """Recommend optimal selection strategy for context"""
        pass


class IScoringEngineService(abc.ABC):
    """Interface for scoring engine service"""

    @abc.abstractmethod
    async def score_models(
        self, models: List[Any], context: SelectionContextContract
    ) -> List[ModelScoreContract]:
        """Score models for applicability"""
        pass

    @abc.abstractmethod
    async def calculate_component_scores(
        self, model: Any, context: SelectionContextContract
    ) -> Dict[str, float]:
        """Calculate detailed component scores"""
        pass


class INWayPatternService(abc.ABC):
    """Interface for N-Way pattern service"""

    @abc.abstractmethod
    async def detect_nway_interactions(
        self, selected_models: List[str], context: SelectionContextContract
    ) -> List[NWayInteractionContract]:
        """Detect N-Way interaction patterns"""
        pass

    @abc.abstractmethod
    async def enhance_with_nway_patterns(
        self,
        selection_result: SelectionResultContract,
        context: SelectionContextContract,
    ) -> SelectionResultContract:
        """Enhance selection with N-Way patterns"""
        pass


class IBayesianLearningService(abc.ABC):
    """Interface for Bayesian learning service"""

    @abc.abstractmethod
    async def get_learned_effectiveness(
        self, model_id: str, context: SelectionContextContract
    ) -> Optional[BayesianUpdateContract]:
        """Get learned model effectiveness for context"""
        pass

    @abc.abstractmethod
    async def update_model_effectiveness(
        self,
        model_id: str,
        effectiveness_score: float,
        context: SelectionContextContract,
    ) -> BayesianUpdateContract:
        """Update Bayesian effectiveness"""
        pass


class IZeroShotSelectionService(abc.ABC):
    """Interface for zero-shot selection service"""

    @abc.abstractmethod
    async def perform_zero_shot_selection(
        self, context: SelectionContextContract
    ) -> ZeroShotSelectionContract:
        """Perform zero-shot model selection"""
        pass

    @abc.abstractmethod
    async def merge_with_database_selection(
        self,
        zero_shot_result: ZeroShotSelectionContract,
        database_result: SelectionResultContract,
        merge_strategy: MergeStrategy,
    ) -> SelectionResultContract:
        """Merge zero-shot with database selection"""
        pass


class ISelectionCoordinatorService(abc.ABC):
    """Interface for selection coordinator service"""

    @abc.abstractmethod
    async def coordinate_model_selection(
        self, context: SelectionContextContract
    ) -> SelectionCoordinationContract:
        """Coordinate all selection services"""
        pass

    @abc.abstractmethod
    async def get_cluster_health(self) -> Dict[str, Any]:
        """Get selection cluster health status"""
        pass


# ============================================================
# UTILITY CONTRACTS
# ============================================================


@dataclass
class ModelCandidate:
    """Represents a model candidate for selection"""

    model_id: str
    model_name: str
    category: str
    application_criteria: List[str]
    performance_metrics: Dict[str, float]
    validation_status: str
    expected_improvement: float


@dataclass
class ScoringWeights:
    """Weights for scoring components"""

    criteria_matching: float = 0.30
    performance_history: float = 0.20
    validation_status: float = 0.15
    cognitive_load: float = 0.10
    diversity_bonus: float = 0.10
    nway_synergy: float = 0.15


@dataclass
class SelectionMetrics:
    """Metrics for selection performance"""

    total_candidates: int
    evaluated_candidates: int
    selection_time_ms: float
    confidence_scores: List[float]
    strategy_effectiveness: float
    exploration_triggered: bool


@dataclass
class NWayPattern:
    """N-Way interaction pattern definition"""

    pattern_id: str
    interaction_type: NWayInteractionType
    model_combinations: List[List[str]]
    success_rate: float
    context_requirements: Dict[str, Any]
    enhancement_instructions: str
