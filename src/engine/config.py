"""
TreeGlav Centralized Configuration System
Pydantic-based settings management for all components
"""

import os
from typing import Dict, Optional
from pydantic import Field

# SettingsConfigDict is part of pydantic-settings; provide a safe fallback if unavailable
try:
    from pydantic_settings import SettingsConfigDict
except Exception:  # pragma: no cover
    try:
        from pydantic import ConfigDict as SettingsConfigDict  # type: ignore
    except Exception:
        SettingsConfigDict = dict  # type: ignore

# Handle BaseSettings import with fallback
try:
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback for older pydantic versions or missing pydantic-settings
    from pydantic import BaseModel as BaseSettings


def float_env(name: str, default: float) -> float:
    """Get float value from environment with fallback to default"""
    try:
        return float(os.getenv(name, default))
    except (ValueError, TypeError):
        return default


# LLM Provider Configuration for DeepSeek-First Architecture
LLM_PROVIDER_HIERARCHY = ["deepseek", "claude"]
PRIMARY_PROVIDER = LLM_PROVIDER_HIERARCHY[0]
FALLBACK_PROVIDER = LLM_PROVIDER_HIERARCHY[1]

# Provider-specific timeout configurations
PROVIDER_TIMEOUTS = {
    "deepseek": {
        "reasoner": 180.0,  # 3 minutes for reasoning mode (legacy baseline)
        "chat": 45.0,  # 45 seconds for fast mode
        "availability": 10.0,
    },
    "claude": {"default": 60.0, "availability": 10.0},  # 1 minute for Claude
}

# Adaptive timeout matrix based on research findings
# Research shows complex tasks can benefit from 5-10 minute timeouts
ADAPTIVE_TIMEOUT_MATRIX = {
    "ultra_complex": {
        "base_timeout": 600,  # 10 minutes for ultra-complex tasks
        "min_timeout": 300,  # 5 minutes minimum
        "max_timeout": 900,  # 15 minutes maximum
        "complexity_multiplier_range": (0.8, 1.5),
        "context_multiplier_range": (1.0, 2.0),
    },
    "standard_complex": {
        "base_timeout": 300,  # 5 minutes for standard complex
        "min_timeout": 120,  # 2 minutes minimum
        "max_timeout": 480,  # 8 minutes maximum
        "complexity_multiplier_range": (0.7, 1.3),
        "context_multiplier_range": (1.0, 1.6),
    },
    "fast_response": {
        "base_timeout": 60,  # 1 minute for fast response
        "min_timeout": 30,  # 30 seconds minimum
        "max_timeout": 180,  # 3 minutes maximum
        "complexity_multiplier_range": (0.6, 1.2),
        "context_multiplier_range": (1.0, 1.3),
    },
}

# Task complexity timeout multipliers (research-backed)
TASK_COMPLEXITY_MULTIPLIERS = {
    # Ultra-complex tasks requiring extended reasoning
    "multi_model_synthesis": 1.8,
    "strategic_inversion_analysis": 1.6,
    "assumption_network_analysis": 1.4,
    "competitive_dynamics_modeling": 1.5,
    "risk_cascade_analysis": 1.7,
    # Standard complex tasks
    "challenge_generation": 1.2,
    "assumption_challenge": 1.1,
    "strategic_synthesis": 1.3,
    "complex_reasoning": 1.2,
    "evidence_synthesis": 1.1,
    # Fast response tasks
    "summary_generation": 0.7,
    "problem_classification": 0.6,
    "quick_insights": 0.5,
    "pattern_recognition": 0.6,
}


# ============================================================================
# METIS UNIFIED CONFIGURATION SYSTEM
# Sprint: Clarity & Consolidation (Week 1, Day 1-2)
# Purpose: Single Source of Truth for ALL configuration
# ============================================================================


class OrchestrationConfig(BaseSettings):
    """Orchestration and workflow configuration"""

    max_retries: int = Field(
        default=3,
        env="ORCHESTRATION_MAX_RETRIES",
        description="Maximum retry attempts for failed operations",
    )
    phase_timeout_seconds: Dict[str, int] = Field(
        default_factory=lambda: {
            "problem_structuring": 120,
            "hypothesis_generation": 180,
            "analysis_execution": 300,
            "synthesis_delivery": 120,
            "clarification": 90,
        },
        description="Timeout for each engagement phase in seconds",
    )
    contradiction_detection_enabled: bool = Field(
        default=True, env="ENABLE_CONTRADICTION_DETECTION"
    )
    contradiction_blocking_threshold: float = Field(
        default=0.8, env="CONTRADICTION_BLOCKING_THRESHOLD", ge=0.0, le=1.0
    )
    state_transition_timeout: int = Field(
        default=10,
        env="STATE_TRANSITION_TIMEOUT",
        description="Timeout for state transitions",
    )
    enable_phantom_detection: bool = Field(default=True, env="ENABLE_PHANTOM_DETECTION")


class CircuitBreakerConfig(BaseSettings):
    """Circuit breaker configuration for resilience"""

    failure_threshold: int = Field(default=5, env="CIRCUIT_BREAKER_FAILURE_THRESHOLD")
    timeout_seconds: int = Field(default=30, env="CIRCUIT_BREAKER_TIMEOUT")
    reset_timeout_seconds: int = Field(default=60, env="CIRCUIT_BREAKER_RESET_TIMEOUT")
    half_open_max_calls: int = Field(
        default=3, env="CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS"
    )
    success_threshold: int = Field(default=2, env="CIRCUIT_BREAKER_SUCCESS_THRESHOLD")


class MonitoringConfig(BaseSettings):
    """Performance monitoring and quality thresholds"""

    latency_threshold_ms: int = Field(default=100, env="MONITORING_LATENCY_THRESHOLD")
    success_rate_threshold: float = Field(
        default=0.95, env="MONITORING_SUCCESS_RATE", ge=0.0, le=1.0
    )
    alert_threshold_error_rate: float = Field(
        default=0.05, env="MONITORING_ALERT_ERROR_RATE", ge=0.0, le=1.0
    )
    metrics_collection_interval: int = Field(
        default=60, env="METRICS_COLLECTION_INTERVAL"
    )
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")


class ResearchConfig(BaseSettings):
    """Research engine configuration"""

    max_concurrent_queries: int = Field(default=5, env="RESEARCH_MAX_CONCURRENT")
    query_timeout_seconds: int = Field(default=30, env="RESEARCH_QUERY_TIMEOUT")
    max_retries: int = Field(default=3, env="RESEARCH_MAX_RETRIES")
    cache_ttl_seconds: int = Field(default=3600, env="RESEARCH_CACHE_TTL")
    min_confidence_threshold: float = Field(
        default=0.7, env="RESEARCH_MIN_CONFIDENCE", ge=0.0, le=1.0
    )
    contradiction_detection_threshold: float = Field(
        default=0.6, env="RESEARCH_CONTRADICTION_THRESHOLD", ge=0.0, le=1.0
    )


class LLMProviderConfig(BaseSettings):
    """LLM provider configuration"""

    primary_provider: str = Field(default="deepseek", env="LLM_PRIMARY_PROVIDER")
    fallback_provider: str = Field(default="claude", env="LLM_FALLBACK_PROVIDER")
    max_tokens_default: int = Field(default=4096, env="LLM_MAX_TOKENS")
    temperature_default: float = Field(
        default=0.7, env="LLM_TEMPERATURE", ge=0.0, le=2.0
    )
    retry_attempts: int = Field(default=3, env="LLM_RETRY_ATTEMPTS")
    retry_delay_seconds: int = Field(default=2, env="LLM_RETRY_DELAY")
    request_timeout_seconds: int = Field(default=60, env="LLM_REQUEST_TIMEOUT")


class CacheConfig(BaseSettings):
    """Caching configuration across all layers"""

    l1_memory_size: int = Field(default=1000, env="CACHE_L1_SIZE")
    l1_ttl_seconds: int = Field(default=3600, env="CACHE_L1_TTL")
    l2_redis_enabled: bool = Field(default=False, env="CACHE_L2_REDIS_ENABLED")
    l2_ttl_seconds: int = Field(default=86400, env="CACHE_L2_TTL")
    l3_persistent_enabled: bool = Field(default=True, env="CACHE_L3_PERSISTENT_ENABLED")
    l3_retention_days: int = Field(default=30, env="CACHE_L3_RETENTION_DAYS")
    kv_cache_optimization: bool = Field(default=True, env="KV_CACHE_OPTIMIZATION")
    kv_cache_hit_target: float = Field(
        default=0.8, env="KV_CACHE_HIT_TARGET", ge=0.0, le=1.0
    )


class ValidationConfig(BaseSettings):
    """Validation and quality control thresholds"""

    min_confidence_score: float = Field(
        default=0.6, env="VALIDATION_MIN_CONFIDENCE", ge=0.0, le=1.0
    )
    hallucination_detection_threshold: float = Field(
        default=0.7, env="HALLUCINATION_THRESHOLD", ge=0.0, le=1.0
    )
    bias_detection_threshold: float = Field(
        default=0.7, env="BIAS_DETECTION_THRESHOLD", ge=0.0, le=1.0
    )
    pattern_min_confidence: float = Field(
        default=0.8, env="PATTERN_MIN_CONFIDENCE", ge=0.0, le=1.0
    )
    max_bias_combinations: int = Field(default=10, env="MAX_BIAS_COMBINATIONS")
    evidence_min_sources: int = Field(default=3, env="EVIDENCE_MIN_SOURCES")


class CognitiveEngineSettings(BaseSettings):
    """Configuration for the Cognitive Engine"""

    # Model Selection Settings
    DEFAULT_ACCURACY_REQUIREMENT: float = Field(
        default=0.8,
        env="COGNITIVE_ACCURACY_REQUIREMENT",
        description="Default accuracy threshold for model selection",
    )
    DEFAULT_MODEL_SELECTION_LIMIT: int = Field(
        default=3,
        env="COGNITIVE_MODEL_LIMIT",
        description="Maximum number of models to select for cognitive load management",
    )

    # Processing Settings
    COGNITIVE_LOAD_HIGH_THRESHOLD: int = Field(
        default=2000, description="Character count threshold for high cognitive load"
    )
    COGNITIVE_LOAD_MEDIUM_THRESHOLD: int = Field(
        default=1000, description="Character count threshold for medium cognitive load"
    )

    # Performance Settings
    MAX_MODEL_PERFORMANCE_HISTORY: int = Field(
        default=50, description="Maximum performance history entries per model"
    )
    SYNTHESIS_CONFIDENCE_TRIANGULATION_BONUS: float = Field(
        default=0.03, description="Confidence bonus per additional model"
    )
    MAX_SYNTHESIS_CONFIDENCE: float = Field(
        default=0.95, description="Maximum synthesis confidence score"
    )

    # Validation Settings
    LOW_CONFIDENCE_THRESHOLD: float = Field(
        default=0.6, description="Threshold for low confidence warning"
    )
    MIN_MODEL_DIVERSITY_THRESHOLD: int = Field(
        default=2, description="Minimum model diversity requirement"
    )
    MIN_EVIDENCE_COMPREHENSIVENESS: int = Field(
        default=3, description="Minimum evidence sources required"
    )

    # Phantom Workflow Prevention (configurable timing thresholds)
    PHANTOM_MIN_TIMES: Dict[str, float] = Field(
        default_factory=lambda: {
            "problem_analysis": float_env("PHANTOM_MIN_PROBLEM_ANALYSIS", 2.0),
            "model_selection": float_env("PHANTOM_MIN_MODEL_SELECTION", 1.0),
            "model_application": float_env("PHANTOM_MIN_MODEL_APPLICATION", 5.0),
            "reasoning_synthesis": float_env("PHANTOM_MIN_REASONING_SYNTHESIS", 3.0),
            "validation_check": float_env("PHANTOM_MIN_VALIDATION_CHECK", 2.0),
        }
    )

    # LLM Gate Caching (TTL in minutes)
    LLM_GATE_TTL_MINUTES: int = Field(
        default=int(os.getenv("LLM_GATE_TTL_MINUTES", 15)),
        description="LLM validation gate cache TTL in minutes",
    )

    # Top-K Confidence Calibration
    CALIBRATION_TOP_K: int = Field(
        default=int(os.getenv("CALIBRATION_TOP_K", 3)),
        description="Number of top models to calibrate confidence for",
    )

    # Heavy Subsystem Guards
    DISABLE_HEAVY_WHEN_OFFLINE: bool = Field(
        default=os.getenv("DISABLE_HEAVY_WHEN_OFFLINE", "true").lower() == "true",
        description="Disable heavy subsystems when database is offline",
    )

    # Single Source N-Way Synergy
    ENGINE_NWAY_ENABLED: bool = Field(
        default=os.getenv("ENGINE_NWAY_ENABLED", "false").lower() == "true",
        description="Enable engine-level N-Way pattern matching (default: false, use ModelSelector only)",
    )

    # Operation Hyperspeed - Parallel Model Execution
    MAX_CONCURRENT_MODELS: int = Field(
        default=int(os.getenv("MAX_CONCURRENT_MODELS", 3)),
        description="Maximum concurrent mental model applications",
    )
    ENABLE_PARALLEL_MODEL_EXECUTION: bool = Field(
        default=os.getenv("ENABLE_PARALLEL_MODEL_EXECUTION", "true").lower() == "true",
        description="Enable parallel execution of mental models",
    )
    MODEL_EXECUTION_TIMEOUT: int = Field(
        default=int(os.getenv("MODEL_EXECUTION_TIMEOUT", 30)),
        description="Timeout per model execution in seconds",
    )

    # NEW: Enhanced Capabilities Configuration (Ideaflow + Ackoff Integration)
    research_enabled: bool = Field(
        default=os.getenv("RESEARCH_ENABLED", "true").lower() == "true",
        description="Enable research integration for cognitive processing",
    )
    bias_detection_enabled: bool = Field(
        default=os.getenv("BIAS_DETECTION_ENABLED", "true").lower() == "true",
        description="Enable bias detection and AI augmentation",
    )
    ENABLE_HMW_GENERATION: bool = Field(
        default=os.getenv("ENABLE_HMW_GENERATION", "true").lower() == "true",
        description="Enable HMW question generation for creative problems",
    )
    ENABLE_ASSUMPTION_CHALLENGING: bool = Field(
        default=os.getenv("ENABLE_ASSUMPTION_CHALLENGING", "true").lower() == "true",
        description="Enable assumption challenging for dissolution problems",
    )

    # HMW Generation Settings
    HMW_MAX_QUESTIONS_PER_SESSION: int = Field(
        default=int(os.getenv("HMW_MAX_QUESTIONS_PER_SESSION", 5)),
        description="Maximum HMW questions to generate per session",
    )
    HMW_MIN_QUALITY_THRESHOLD: float = Field(
        default=float_env("HMW_MIN_QUALITY_THRESHOLD", 0.6),
        description="Minimum quality score for HMW questions",
    )
    HMW_GENERATION_TIMEOUT_SECONDS: int = Field(
        default=int(os.getenv("HMW_GENERATION_TIMEOUT_SECONDS", 30)),
        description="Timeout for HMW generation in seconds",
    )

    # Assumption Challenging Settings
    ASSUMPTION_MAX_CHALLENGES_PER_SESSION: int = Field(
        default=int(os.getenv("ASSUMPTION_MAX_CHALLENGES_PER_SESSION", 8)),
        description="Maximum assumptions to challenge per session",
    )
    ASSUMPTION_MIN_DISSOLUTION_POTENTIAL: float = Field(
        default=float_env("ASSUMPTION_MIN_DISSOLUTION_POTENTIAL", 0.5),
        description="Minimum dissolution potential to include assumption",
    )
    ASSUMPTION_ANALYSIS_TIMEOUT_SECONDS: int = Field(
        default=int(os.getenv("ASSUMPTION_ANALYSIS_TIMEOUT_SECONDS", 45)),
        description="Timeout for assumption analysis in seconds",
    )

    # Enhanced Classification Settings
    CREATIVE_IDEATION_MIN_SIGNALS: int = Field(
        default=int(os.getenv("CREATIVE_IDEATION_MIN_SIGNALS", 2)),
        description="Minimum signals to classify as creative_ideation",
    )
    DISSOLUTION_CANDIDATE_MIN_SIGNALS: int = Field(
        default=int(os.getenv("DISSOLUTION_CANDIDATE_MIN_SIGNALS", 2)),
        description="Minimum signals to classify as dissolution_candidate",
    )

    # Ideaflow Metrics Settings
    ENABLE_IDEAFLOW_METRICS: bool = Field(
        default=os.getenv("ENABLE_IDEAFLOW_METRICS", "true").lower() == "true",
        description="Enable ideaflow performance metrics tracking",
    )

    # Operation Synapse: Context Intelligence Settings
    CONTEXT_CACHE_SIZE: int = Field(
        default=int(os.getenv("CONTEXT_CACHE_SIZE", 1000)),
        description="L1 context cache maximum size",
    )
    CONTEXT_CACHE_TTL: int = Field(
        default=int(os.getenv("CONTEXT_CACHE_TTL", 3600)),
        description="L1 context cache TTL in seconds",
    )
    ENABLE_CONTEXT_INTELLIGENCE: bool = Field(
        default=os.getenv("ENABLE_CONTEXT_INTELLIGENCE", "true").lower() == "true",
        description="Enable Context Intelligence Engine",
    )

    # Operation Synapse Sprint 1.2: L2 Redis Distributed Cache Settings
    REDIS_HOST: str = Field(
        default=os.getenv("REDIS_HOST", "localhost"),
        description="Redis host for L2 distributed cache",
    )
    REDIS_PORT: int = Field(
        default=int(os.getenv("REDIS_PORT", 6379)), description="Redis port"
    )
    REDIS_DB: int = Field(
        default=int(os.getenv("REDIS_DB", 0)), description="Redis database index"
    )
    REDIS_PASSWORD: Optional[str] = Field(
        default=os.getenv("REDIS_PASSWORD"), description="Redis password (optional)"
    )
    ENABLE_L2_REDIS_CACHE: bool = Field(
        default=os.getenv("ENABLE_L2_REDIS_CACHE", "false").lower() == "true",
        description="Enable L2 Redis distributed cache",
    )
    L2_CACHE_TTL: int = Field(
        default=int(os.getenv("L2_CACHE_TTL", 86400)),
        description="L2 Redis cache TTL in seconds (24h default)",
    )
    L2_CACHE_PREFIX: str = Field(
        default="metis:context:", description="Redis key prefix for L2 cache"
    )
    MAX_RELEVANT_CONTEXTS: int = Field(
        default=int(os.getenv("MAX_RELEVANT_CONTEXTS", 5)),
        description="Maximum relevant contexts to retrieve",
    )

    # Operation Synapse Sprint 1.3: L3 Supabase Persistent Cache Settings
    ENABLE_L3_SUPABASE_CACHE: bool = Field(
        default=os.getenv("ENABLE_L3_SUPABASE_CACHE", "true").lower() == "true",
        description="Enable L3 Supabase persistent cache",
    )
    L3_CACHE_TABLE: str = Field(
        default="cognitive_exhaust_cache",
        description="Supabase table for L3 persistent cache",
    )
    L3_BATCH_SIZE: int = Field(
        default=int(os.getenv("L3_BATCH_SIZE", 100)),
        description="Batch size for L3 cache operations",
    )
    L3_RETENTION_DAYS: int = Field(
        default=int(os.getenv("L3_RETENTION_DAYS", 30)),
        description="L3 cache retention period in days",
    )

    # Operation Synapse Sprint 1.5: KV-Cache Optimization (Manus.im Insights)
    ENABLE_KV_CACHE_OPTIMIZATION: bool = Field(
        default=os.getenv("ENABLE_KV_CACHE_OPTIMIZATION", "true").lower() == "true",
        description="Enable KV-cache optimization for 10x cost reduction",
    )
    KV_CACHE_HIT_TARGET: float = Field(
        default=float_env("KV_CACHE_HIT_TARGET", 0.8),
        description="Target KV-cache hit rate (80%)",
    )
    STABLE_PROMPT_PREFIX: bool = Field(
        default=os.getenv("STABLE_PROMPT_PREFIX", "true").lower() == "true",
        description="Use stable prompt prefixes for cache optimization",
    )
    CONTEXT_COMPRESSION_REVERSIBLE: bool = Field(
        default=os.getenv("CONTEXT_COMPRESSION_REVERSIBLE", "true").lower() == "true",
        description="Enable reversible context compression",
    )

    # Single-Agent Depth Mode (Cognition.ai Philosophy)
    SINGLE_AGENT_DEPTH_MODE: bool = Field(
        default=os.getenv("SINGLE_AGENT_DEPTH_MODE", "true").lower() == "true",
        description="Enable single-agent depth over multi-agent complexity",
    )
    UNIFIED_CONTEXT_MANAGEMENT: bool = Field(
        default=os.getenv("UNIFIED_CONTEXT_MANAGEMENT", "true").lower() == "true",
        description="Centralized context management",
    )
    LINEAR_PROCESSING_PIPELINE: bool = Field(
        default=os.getenv("LINEAR_PROCESSING_PIPELINE", "true").lower() == "true",
        description="Single-threaded context processing",
    )

    # Context Relevance Scoring Weights (Manus Labs Pattern Enhanced)
    CONTEXT_SEMANTIC_WEIGHT: float = Field(
        default=float_env("CONTEXT_SEMANTIC_WEIGHT", 0.4),
        description="Weight for semantic similarity scoring",
    )
    CONTEXT_TEMPORAL_WEIGHT: float = Field(
        default=float_env("CONTEXT_TEMPORAL_WEIGHT", 0.2),
        description="Weight for temporal recency scoring",
    )
    CONTEXT_FREQUENCY_WEIGHT: float = Field(
        default=float_env("CONTEXT_FREQUENCY_WEIGHT", 0.1),
        description="Weight for usage frequency scoring",
    )
    CONTEXT_COGNITIVE_WEIGHT: float = Field(
        default=float_env("CONTEXT_COGNITIVE_WEIGHT", 0.3),
        description="Weight for cognitive coherence scoring (Revolutionary Feature)",
    )

    # Cognitive Exhaust Integration Settings
    STORE_COGNITIVE_EXHAUST: bool = Field(
        default=os.getenv("STORE_COGNITIVE_EXHAUST", "true").lower() == "true",
        description="Store cognitive thinking processes for context intelligence",
    )
    MAX_COGNITIVE_EXHAUST_LENGTH: int = Field(
        default=int(os.getenv("MAX_COGNITIVE_EXHAUST_LENGTH", 10000)),
        description="Maximum length of thinking process to store",
    )
    COGNITIVE_EXHAUST_COMPRESSION: bool = Field(
        default=os.getenv("COGNITIVE_EXHAUST_COMPRESSION", "false").lower() == "true",
        description="Enable compression of cognitive exhaust data",
    )
    IDEAFLOW_TARGET_IDEAS_PER_MINUTE: float = Field(
        default=float_env("IDEAFLOW_TARGET_IDEAS_PER_MINUTE", 3.0),
        description="Target ideas per minute for ideaflow metrics",
    )
    IDEAFLOW_TARGET_DIVERSITY_SCORE: float = Field(
        default=float_env("IDEAFLOW_TARGET_DIVERSITY_SCORE", 0.7),
        description="Target diversity score for ideaflow metrics",
    )
    IDEAFLOW_TARGET_NOVELTY_SCORE: float = Field(
        default=float_env("IDEAFLOW_TARGET_NOVELTY_SCORE", 0.6),
        description="Target novelty score for ideaflow metrics",
    )

    # Cognitive Diversity Calibration Settings (Flywheel Anti-Convergence)
    ENABLE_DIVERSITY_CALIBRATION: bool = Field(
        default=os.getenv("ENABLE_DIVERSITY_CALIBRATION", "true").lower() == "true",
        description="Enable cognitive diversity calibration to prevent model convergence",
    )
    DIVERSITY_ANALYSIS_WINDOW_DAYS: int = Field(
        default=int(os.getenv("DIVERSITY_ANALYSIS_WINDOW_DAYS", 30)),
        description="Analysis window for convergence pattern detection (days)",
    )
    DIVERSITY_MIN_PATTERN_ENTROPY: float = Field(
        default=float_env("DIVERSITY_MIN_PATTERN_ENTROPY", 2.5),
        description="Minimum Shannon entropy for healthy model diversity",
    )
    DIVERSITY_MAX_GINI_COEFFICIENT: float = Field(
        default=float_env("DIVERSITY_MAX_GINI_COEFFICIENT", 0.6),
        description="Maximum Gini coefficient for model usage inequality",
    )
    DIVERSITY_MIN_UNIQUE_MODELS_RATIO: float = Field(
        default=float_env("DIVERSITY_MIN_UNIQUE_MODELS_RATIO", 0.4),
        description="Minimum ratio of unique models to available models",
    )
    DIVERSITY_TARGET_EXPLORATION_RATE: float = Field(
        default=float_env("DIVERSITY_TARGET_EXPLORATION_RATE", 0.15),
        description="Target rate of exploratory (non-optimal) model selections",
    )

    # Zero-Shot Mental Model Selection Settings (MeMo Paper Implementation)
    ENABLE_ZERO_SHOT_MODEL_SELECTION: bool = Field(
        default=os.getenv("ENABLE_ZERO_SHOT_MODEL_SELECTION", "false").lower()
        == "true",
        description="Enable zero-shot mental model selection alongside database queries",
    )
    ZERO_SHOT_CONFIDENCE_WEIGHT: float = Field(
        default=float_env("ZERO_SHOT_CONFIDENCE_WEIGHT", 0.3),
        description="Weight for zero-shot selections in hybrid approach (0.0-1.0)",
    )
    ZERO_SHOT_MAX_MODELS: int = Field(
        default=int(os.getenv("ZERO_SHOT_MAX_MODELS", 5)),
        description="Maximum mental models to select via zero-shot prompting",
    )
    ZERO_SHOT_MIN_CONFIDENCE: float = Field(
        default=float_env("ZERO_SHOT_MIN_CONFIDENCE", 0.6),
        description="Minimum confidence threshold for zero-shot selections",
    )
    ZERO_SHOT_PROMPT_COMPLEXITY: str = Field(
        default=os.getenv("ZERO_SHOT_PROMPT_COMPLEXITY", "standard"),
        description="Zero-shot prompt complexity level: minimal, standard, comprehensive",
    )
    ZERO_SHOT_TIMEOUT_SECONDS: int = Field(
        default=int(os.getenv("ZERO_SHOT_TIMEOUT_SECONDS", 60)),
        description="Timeout for zero-shot selection in seconds",
    )

    # Intervention Settings
    DIVERSITY_CRITICAL_INTERVENTION_STRENGTH: float = Field(
        default=float_env("DIVERSITY_CRITICAL_INTERVENTION_STRENGTH", 0.3),
        description="Intervention strength for critical convergence risk",
    )
    DIVERSITY_HIGH_INTERVENTION_STRENGTH: float = Field(
        default=float_env("DIVERSITY_HIGH_INTERVENTION_STRENGTH", 0.2),
        description="Intervention strength for high convergence risk",
    )
    DIVERSITY_INTERVENTION_DURATION: int = Field(
        default=int(os.getenv("DIVERSITY_INTERVENTION_DURATION", 5)),
        description="Default duration for diversity interventions (engagements)",
    )

    model_config = SettingsConfigDict(
        env_prefix="METIS_COGNITIVE_", case_sensitive=True
    )


class WorkflowEngineSettings(BaseSettings):
    """Configuration for the Workflow Engine"""

    # Performance Settings
    CONTEXT_COMPRESSION_TOKEN_LIMIT: int = Field(
        default=3000, description="Token limit before context compression"
    )
    CONTEXT_COMPRESSION_TARGET_TOKENS: int = Field(
        default=2500, description="Target tokens after compression"
    )

    # Cache Settings
    ENGAGEMENT_CACHE_TTL_SECONDS: int = Field(
        default=1800, description="TTL for complete engagement cache (30 minutes)"
    )
    PHASE_CACHE_TTL_SECONDS: int = Field(
        default=3600, description="TTL for phase result cache (1 hour)"
    )

    # Performance Targets
    SUB_2S_RESPONSE_TARGET: float = Field(
        default=2.0, description="Target response time in seconds"
    )

    # Validation Settings
    LLM_VALIDATION_ENHANCEMENT_RATIO_THRESHOLD: float = Field(
        default=0.9, description="Minimum enhancement ratio for LLM validation"
    )

    model_config = SettingsConfigDict(env_prefix="METIS_WORKFLOW_", case_sensitive=True)


class DatabaseSettings(BaseSettings):
    """Configuration for Database components"""

    # Database URLs
    POSTGRES_URL: str = Field(
        default="postgresql://localhost:5432/metis",
        description="PostgreSQL connection URL",
    )
    REDIS_URL: str = Field(
        default="redis://localhost:6379", description="Redis connection URL"
    )

    # N-way Manager Settings
    NWAY_RELEVANCE_SCORE_THRESHOLD: float = Field(
        default=0.1, description="Minimum relevance score for N-way pattern selection"
    )
    NWAY_DOMAIN_MATCH_WEIGHT: float = Field(
        default=0.5, description="Weight for domain matching in relevance scoring"
    )
    NWAY_KEYWORD_MATCH_WEIGHT: float = Field(
        default=0.1, description="Weight per keyword match in relevance scoring"
    )

    # Strength Bonuses
    NWAY_HIGH_STRENGTH_BONUS: float = Field(
        default=0.3, description="Bonus for high strength interactions"
    )
    NWAY_MEDIUM_STRENGTH_BONUS: float = Field(
        default=0.2, description="Bonus for medium strength interactions"
    )
    NWAY_LOW_STRENGTH_BONUS: float = Field(
        default=0.1, description="Bonus for low strength interactions"
    )

    # Query Limits
    NWAY_HIGH_STRENGTH_QUERY_LIMIT: int = Field(
        default=10, description="Maximum high strength interactions to query"
    )
    NWAY_MEDIUM_STRENGTH_QUERY_LIMIT: int = Field(
        default=5, description="Maximum medium strength interactions to query"
    )

    model_config = SettingsConfigDict(env_prefix="METIS_DATABASE_", case_sensitive=True)


class LLMSettings(BaseSettings):
    """Configuration for LLM integrations"""

    # API Settings
    ANTHROPIC_API_KEY: Optional[str] = Field(
        default=None, description="Anthropic API key"
    )
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key")
    PERPLEXITY_API_KEY: Optional[str] = Field(
        default=None, description="Perplexity API key"
    )
    DEEPSEEK_API_KEY: Optional[str] = Field(
        default=None, description="DeepSeek API key"
    )
    TAVILY_API_KEY: Optional[str] = Field(default=None, description="Tavily API key")
    RAGIE_API_KEY: Optional[str] = Field(default=None, description="Ragie API key")

    # Supabase Settings
    SUPABASE_URL: Optional[str] = Field(
        default=None, description="Supabase project URL"
    )
    SUPABASE_ANON_KEY: Optional[str] = Field(
        default=None, description="Supabase anonymous key"
    )
    SUPABASE_SERVICE_ROLE_KEY: Optional[str] = Field(
        default=None, description="Supabase service role key"
    )
    NEXT_PUBLIC_SUPABASE_URL: Optional[str] = Field(
        default=None, description="Next.js public Supabase URL"
    )
    NEXT_PUBLIC_SUPABASE_ANON_KEY: Optional[str] = Field(
        default=None, description="Next.js public Supabase anon key"
    )

    # Token Limits
    DEFAULT_MAX_TOKENS: int = Field(
        default=2000, description="Default maximum tokens for LLM calls"
    )
    SYNTHESIS_MAX_TOKENS: int = Field(
        default=2500, description="Maximum tokens for synthesis calls"
    )
    MECE_MAX_TOKENS: int = Field(
        default=2500, description="Maximum tokens for MECE structuring"
    )

    # Temperature Settings
    SYSTEMS_THINKING_TEMPERATURE: float = Field(
        default=0.3, description="Temperature for systems thinking analysis"
    )
    CRITICAL_THINKING_TEMPERATURE: float = Field(
        default=0.2, description="Temperature for critical thinking analysis"
    )
    MECE_STRUCTURING_TEMPERATURE: float = Field(
        default=0.1, description="Temperature for MECE structuring"
    )
    HYPOTHESIS_TESTING_TEMPERATURE: float = Field(
        default=0.2, description="Temperature for hypothesis testing"
    )
    DECISION_FRAMEWORK_TEMPERATURE: float = Field(
        default=0.3, description="Temperature for decision framework analysis"
    )

    model_config = SettingsConfigDict(env_prefix="", case_sensitive=True)


class EventBusSettings(BaseSettings):
    """Configuration for Event Bus"""

    # Kafka Settings
    KAFKA_BROKERS: str = Field(
        default="localhost:9092", description="Kafka broker addresses"
    )
    KAFKA_TOPIC_PREFIX: str = Field(
        default="metis", description="Prefix for Kafka topics"
    )

    # Event Settings
    EVENT_PUBLISHING_TIMEOUT: int = Field(
        default=5, description="Timeout for event publishing in seconds"
    )
    MAX_EVENT_HISTORY: int = Field(
        default=1000, description="Maximum events to keep in history"
    )

    model_config = SettingsConfigDict(env_prefix="METIS_EVENT_", case_sensitive=True)


class SecuritySettings(BaseSettings):
    """Security and compliance settings"""

    # Authentication
    JWT_SECRET_KEY: Optional[str] = Field(
        default=None, description="JWT secret key for authentication"
    )
    JWT_EXPIRY_HOURS: int = Field(default=24, description="JWT token expiry in hours")

    # Rate Limiting
    API_RATE_LIMIT_PER_MINUTE: int = Field(
        default=60, description="API calls per minute limit"
    )
    LLM_RATE_LIMIT_PER_MINUTE: int = Field(
        default=30, description="LLM calls per minute limit"
    )

    # Audit
    AUDIT_TRAIL_RETENTION_DAYS: int = Field(
        default=2555, description="Audit trail retention (7 years)"
    )

    model_config = SettingsConfigDict(env_prefix="METIS_SECURITY_", case_sensitive=True)


class MetisSettings(BaseSettings):
    """Master settings class combining all configuration"""

    # Core Component Settings (NEW - Clarity & Consolidation Sprint)
    orchestration: OrchestrationConfig = Field(default_factory=OrchestrationConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    llm_providers: LLMProviderConfig = Field(default_factory=LLMProviderConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    # Component Settings (Legacy)
    cognitive_engine: CognitiveEngineSettings = Field(
        default_factory=CognitiveEngineSettings
    )
    workflow_engine: WorkflowEngineSettings = Field(
        default_factory=WorkflowEngineSettings
    )
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    event_bus: EventBusSettings = Field(default_factory=EventBusSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)

    # Global Settings
    ENVIRONMENT: str = Field(
        default="development",
        description="Environment: development, staging, production",
    )
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    DEBUG_MODE: bool = Field(default=False, description="Enable debug mode")

    # Performance
    ENABLE_PERFORMANCE_MONITORING: bool = Field(
        default=True, description="Enable performance monitoring"
    )
    ENABLE_CACHING: bool = Field(default=True, description="Enable caching layers")

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=True,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )


# Global settings instance
_settings: Optional[MetisSettings] = None


def get_settings() -> MetisSettings:
    """Get or create global settings instance"""
    global _settings
    if _settings is None:
        _settings = MetisSettings()
    return _settings


def reload_settings() -> MetisSettings:
    """Reload settings from environment"""
    global _settings
    _settings = MetisSettings()
    return _settings


# Convenience functions for specific component settings
def get_cognitive_settings() -> CognitiveEngineSettings:
    """Get cognitive engine settings"""
    return get_settings().cognitive_engine


def get_workflow_settings() -> WorkflowEngineSettings:
    """Get workflow engine settings"""
    return get_settings().workflow_engine


def get_database_settings() -> DatabaseSettings:
    """Get database settings"""
    return get_settings().database


def get_llm_settings() -> LLMSettings:
    """Get LLM settings"""
    return get_settings().llm


def get_event_bus_settings() -> EventBusSettings:
    """Get event bus settings"""
    return get_settings().event_bus


def get_security_settings() -> SecuritySettings:
    """Get security settings"""
    return get_settings().security


# Legacy compatibility
POSTGRES_URL = get_database_settings().POSTGRES_URL
REDIS_URL = get_database_settings().REDIS_URL
DEVELOPMENT_MODE = get_settings().ENVIRONMENT == "development"

if DEVELOPMENT_MODE:
    print("🔧 Running METIS in development mode with centralized configuration")
