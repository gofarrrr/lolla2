"""
METIS Application Services Contracts
Standardized data contracts and interfaces for all application services

Part of Phase 5.3 modular architecture - clean service boundaries for model application cluster.
"""

import abc
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from dataclasses import dataclass


# ============================================================
# ENUMS AND CLASSIFICATIONS
# ============================================================


class ApplicationStrategy(str, Enum):
    SYSTEMS_THINKING = "systems_thinking"
    CRITICAL_THINKING = "critical_thinking"
    MECE_FRAMEWORK = "mece_framework"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    DECISION_FRAMEWORK = "decision_framework"
    GENERIC_APPLICATION = "generic_application"


class ModelApplicationStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class LLMProvider(str, Enum):
    DEEPSEEK = "deepseek"
    CLAUDE = "claude"
    GPT4 = "gpt4"
    FALLBACK = "fallback"


class PerformanceMetricType(str, Enum):
    RESPONSE_TIME = "response_time"
    ACCURACY_SCORE = "accuracy_score"
    CONFIDENCE_LEVEL = "confidence_level"
    COHERENCE_SCORE = "coherence_score"
    RELEVANCE_SCORE = "relevance_score"
    COMPLETENESS_SCORE = "completeness_score"


class OrchestrationMode(str, Enum):
    STANDARD = "standard"
    ENHANCED_NWAY = "enhanced_nway"
    RESEARCH_INTELLIGENCE = "research_intelligence"
    SIMILAR_PATTERN = "similar_pattern"


class FeatureFlag(str, Enum):
    BAYESIAN_LEARNING = "bayesian_learning"
    CONFIDENCE_CALIBRATION = "confidence_calibration"
    DECISION_CAPTURE = "decision_capture"
    VECTOR_SIMILARITY = "vector_similarity"
    NWAY_ENHANCEMENT = "nway_enhancement"


# ============================================================
# DATA CONTRACTS
# ============================================================


@dataclass
class ModelApplicationContract:
    """Contract for model application results"""

    application_id: str
    model_id: str
    strategy_used: ApplicationStrategy
    application_status: ModelApplicationStatus
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    processing_metadata: Dict[str, Any]
    confidence_score: float
    quality_metrics: Dict[str, float]
    application_timestamp: datetime
    processing_time_ms: float
    service_version: str


@dataclass
class LLMResponseContract:
    """Contract for LLM provider responses"""

    response_id: str
    provider_used: LLMProvider
    prompt_hash: str
    response_text: str
    token_count: int
    processing_time_ms: float
    confidence_score: float
    error_message: Optional[str]
    fallback_triggered: bool
    provider_metadata: Dict[str, Any]
    response_timestamp: datetime
    service_version: str


@dataclass
class PerformanceMetricsContract:
    """Contract for performance tracking data"""

    metric_id: str
    model_id: str
    engagement_id: str
    metric_type: PerformanceMetricType
    metric_value: float
    baseline_comparison: float
    trend_direction: str
    context_metadata: Dict[str, Any]
    measurement_timestamp: datetime
    service_version: str


@dataclass
class ModelOrchestrationContract:
    """Contract for model orchestration coordination"""

    orchestration_id: str
    selected_models: List[str]
    orchestration_mode: OrchestrationMode
    coordination_metadata: Dict[str, Any]
    nway_interactions: List[Dict[str, Any]]
    research_integration_data: Optional[Dict[str, Any]]
    similar_patterns_detected: List[Dict[str, Any]]
    orchestration_timestamp: datetime
    total_orchestration_time_ms: float
    service_version: str


@dataclass
class ApplicationResultContract:
    """Master contract for comprehensive application results"""

    result_id: str
    engagement_id: str
    orchestration_result: ModelOrchestrationContract
    application_results: List[ModelApplicationContract]
    llm_responses: List[LLMResponseContract]
    performance_metrics: List[PerformanceMetricsContract]
    feature_flags_used: List[FeatureFlag]
    overall_confidence: float
    processing_summary: Dict[str, Any]
    result_timestamp: datetime
    total_processing_time_ms: float
    service_version: str


@dataclass
class ModelRegistryEntry:
    """Contract for model registry information"""

    model_id: str
    model_name: str
    model_version: str
    application_strategy: ApplicationStrategy
    capabilities: List[str]
    performance_baseline: Dict[str, float]
    resource_requirements: Dict[str, Any]
    validation_status: str
    registration_timestamp: datetime
    last_updated: datetime
    service_version: str


@dataclass
class LifecycleEventContract:
    """Contract for model lifecycle events"""

    event_id: str
    model_id: str
    event_type: str
    event_status: str
    event_data: Dict[str, Any]
    triggered_by: str
    event_timestamp: datetime
    processing_time_ms: float
    service_version: str


# ============================================================
# SERVICE INTERFACES
# ============================================================


class IModelRegistryService(abc.ABC):
    """Interface for model registry service"""

    @abc.abstractmethod
    async def register_model(self, model_entry: ModelRegistryEntry) -> Dict[str, Any]:
        """Register a new model in the registry"""
        pass

    @abc.abstractmethod
    async def get_model(self, model_id: str) -> Optional[ModelRegistryEntry]:
        """Retrieve model information from registry"""
        pass

    @abc.abstractmethod
    async def list_models_by_strategy(
        self, strategy: ApplicationStrategy
    ) -> List[ModelRegistryEntry]:
        """List all models supporting a specific application strategy"""
        pass


class ILifecycleManagementService(abc.ABC):
    """Interface for model lifecycle management service"""

    @abc.abstractmethod
    async def initialize_model(self, model_id: str) -> LifecycleEventContract:
        """Initialize a model for application"""
        pass

    @abc.abstractmethod
    async def update_model_status(
        self, model_id: str, status: str, metadata: Dict[str, Any]
    ) -> LifecycleEventContract:
        """Update model lifecycle status"""
        pass

    @abc.abstractmethod
    async def retire_model(self, model_id: str) -> LifecycleEventContract:
        """Retire a model from active use"""
        pass


class IModelApplicationService(abc.ABC):
    """Interface for model application service"""

    @abc.abstractmethod
    async def apply_model_strategy(
        self,
        model_id: str,
        strategy: ApplicationStrategy,
        input_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> ModelApplicationContract:
        """Apply specific strategy using selected model"""
        pass

    @abc.abstractmethod
    async def get_supported_strategies(
        self, model_id: str
    ) -> List[ApplicationStrategy]:
        """Get list of strategies supported by model"""
        pass


class ILLMIntegrationService(abc.ABC):
    """Interface for LLM integration service"""

    @abc.abstractmethod
    async def generate_response(
        self,
        prompt: str,
        model_preferences: List[LLMProvider],
        generation_config: Dict[str, Any],
    ) -> LLMResponseContract:
        """Generate response using preferred LLM providers with fallback"""
        pass

    @abc.abstractmethod
    async def batch_generate_responses(
        self, prompts: List[str], model_preferences: List[LLMProvider]
    ) -> List[LLMResponseContract]:
        """Generate multiple responses in batch"""
        pass


class IPerformanceMonitoringService(abc.ABC):
    """Interface for performance monitoring service"""

    @abc.abstractmethod
    async def record_performance_metric(
        self,
        model_id: str,
        metric_type: PerformanceMetricType,
        metric_value: float,
        context: Dict[str, Any],
    ) -> PerformanceMetricsContract:
        """Record a performance metric"""
        pass

    @abc.abstractmethod
    async def get_performance_summary(
        self, model_id: str, time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Get performance summary for model"""
        pass

    @abc.abstractmethod
    async def compare_model_performance(
        self, model_ids: List[str], metric_type: PerformanceMetricType
    ) -> Dict[str, Any]:
        """Compare performance across multiple models"""
        pass


class IModelOrchestrationService(abc.ABC):
    """Interface for model orchestration service"""

    @abc.abstractmethod
    async def orchestrate_model_workflow(
        self,
        models: List[str],
        orchestration_mode: OrchestrationMode,
        context: Dict[str, Any],
    ) -> ModelOrchestrationContract:
        """Orchestrate complex model workflows"""
        pass

    @abc.abstractmethod
    async def detect_nway_patterns(
        self, models: List[str], context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect N-Way interaction patterns"""
        pass


class IApplicationCoordinatorService(abc.ABC):
    """Interface for application coordinator service"""

    @abc.abstractmethod
    async def coordinate_application_workflow(
        self, request_context: Dict[str, Any], feature_flags: List[FeatureFlag]
    ) -> ApplicationResultContract:
        """Coordinate complete application workflow"""
        pass

    @abc.abstractmethod
    async def get_cluster_health(self) -> Dict[str, Any]:
        """Get application cluster health status"""
        pass


# ============================================================
# UTILITY CONTRACTS
# ============================================================


@dataclass
class ApplicationWeights:
    """Weights for application quality scoring"""

    accuracy: float = 0.30
    coherence: float = 0.25
    completeness: float = 0.20
    relevance: float = 0.15
    confidence: float = 0.10


@dataclass
class OrchestrationConfig:
    """Configuration for model orchestration"""

    max_concurrent_models: int = 3
    timeout_seconds: int = 120
    retry_attempts: int = 2
    fallback_strategy: ApplicationStrategy = ApplicationStrategy.GENERIC_APPLICATION
    enable_nway_enhancement: bool = True
    enable_research_integration: bool = True


@dataclass
class FeatureFlagConfig:
    """Configuration for feature flags"""

    bayesian_learning_enabled: bool = True
    confidence_calibration_enabled: bool = True
    decision_capture_enabled: bool = False
    vector_similarity_enabled: bool = False
    nway_enhancement_enabled: bool = True


@dataclass
class ApplicationMetrics:
    """Metrics for application performance"""

    total_applications: int
    successful_applications: int
    average_processing_time_ms: float
    average_confidence_score: float
    feature_flag_usage: Dict[str, int]
    error_rate_percentage: float
