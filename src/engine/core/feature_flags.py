"""
Feature Flag System for METIS Platform
Purpose: Control feature rollout and A/B testing for Red Team Council

This module provides:
1. Feature flag evaluation with user/org targeting
2. Percentage-based rollout
3. A/B test group assignment
4. Flag override capabilities for testing
"""

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Set, Any
from uuid import UUID

from src.engine.core.structured_logging import get_logger
from src.config import get_settings

logger = get_logger(__name__, component="feature_flags")
settings = get_settings()


class FeatureFlag(Enum):
    """All feature flags in the system"""

    ENABLE_PARALLEL_VALIDATION = "enable_parallel_validation"
    ENABLE_RESEARCH_GROUNDING = "enable_research_grounding"
    ENABLE_RESEARCH_BRIEF = "enable_research_brief"
    ENABLE_ENHANCED_ARBITRATION = "enable_enhanced_arbitration"
    ENABLE_USER_GENERATED_CRITIQUES = "enable_user_generated_critiques"
    ENABLE_FLYWHEEL_METRICS = "enable_flywheel_metrics"
    ENABLE_STAGE0_ENRICHMENT = "enable_stage0_enrichment"
    # V5.4 Advanced Pipeline Features
    ENABLE_CONTEXT_ENGINEERING = "enable_context_engineering"
    ENABLE_BREADTH_MODE = "enable_breadth_mode"
    ENABLE_ADVANCED_PIPELINE = "enable_advanced_pipeline"
    ENABLE_ENHANCED_ROUTING = "enable_enhanced_routing"
    ENABLE_DEPENDENCY_AWARE_FORGES = "enable_dependency_aware_forges"
    # Additional flags unified from legacy config module
    ENABLE_USER_ARBITRATION = "enable_user_arbitration"
    ENABLE_ARBITRATED_SYNTHESIS = "enable_arbitrated_synthesis"
    ENABLE_CONTEXT_INTELLIGENCE = "enable_context_intelligence"
    ENABLE_WHAT_IF_SANDBOX = "enable_what_if_sandbox"
    ENABLE_ARBITRATION_ANALYTICS = "enable_arbitration_analytics"
    ENABLE_TWELVE_FACTOR_COMPLIANCE = "enable_twelve_factor_compliance"
    # Prompt policy canary rollout (analysis prompts)
    ENABLE_PROMPT_UPGRADE_CANARY = "enable_prompt_upgrade_canary"
    # Senior Advisor Synthesis Refinement
    ENABLE_SA_POLYGON_ENHANCEMENTS = "enable_sa_polygon_enhancements"
    ENABLE_SA_DECISION_RIBBON = "enable_sa_decision_ribbon"
    ENABLE_SA_EXEC_TABLE = "enable_sa_exec_table"
    ENABLE_SA_DISSENT_LEDGER = "enable_sa_dissent_ledger"
    ENABLE_SA_GATE_LOGIC = "enable_sa_gate_logic"


class RolloutStrategy(Enum):
    """Rollout strategies for feature flags"""

    OFF = "off"
    ON = "on"
    PERCENTAGE = "percentage"
    USER_LIST = "user_list"
    ORG_LIST = "org_list"
    BETA_USERS = "beta_users"
    A_B_TEST = "a_b_test"


@dataclass
class FeatureFlagConfig:
    """Configuration for a feature flag"""

    name: FeatureFlag
    enabled: bool
    strategy: RolloutStrategy
    percentage: float = 0.0  # For percentage rollout
    user_ids: Set[str] = None  # For user list targeting
    org_ids: Set[str] = None  # For org list targeting
    beta_group: bool = False  # For beta users
    ab_test_config: Dict[str, Any] = None  # For A/B testing
    override_env_var: str = None  # Environment variable for override
    description: str = ""
    created_at: datetime = None
    updated_at: datetime = None


class FeatureFlagService:
    """Service for managing feature flags"""

    def __init__(self):
        self.logger = logger.with_component("feature_flag_service")
        self._flags = self._initialize_flags()
        self._overrides = self._load_overrides()

    def _initialize_flags(self) -> Dict[FeatureFlag, FeatureFlagConfig]:
        """Initialize feature flag configurations"""

        return {
            FeatureFlag.ENABLE_PARALLEL_VALIDATION: FeatureFlagConfig(
                name=FeatureFlag.ENABLE_PARALLEL_VALIDATION,
                enabled=True,
                strategy=RolloutStrategy.PERCENTAGE,
                percentage=10.0,  # Start with 10% rollout
                override_env_var="FF_PARALLEL_VALIDATION",
                description="Enables Red Team Council parallel validation with three challenger agents",
                ab_test_config={
                    "experiment_id": "red_team_council_v1",
                    "control_group_size": 0.5,
                    "treatment_group_size": 0.4,
                    "holdout_group_size": 0.1,
                },
            ),
            FeatureFlag.ENABLE_RESEARCH_GROUNDING: FeatureFlagConfig(
                name=FeatureFlag.ENABLE_RESEARCH_GROUNDING,
                enabled=True,
                strategy=RolloutStrategy.ON,  # Always on after initial testing
                override_env_var="FF_RESEARCH_GROUNDING",
                description="Enables external research grounding via Perplexity API",
            ),
            FeatureFlag.ENABLE_RESEARCH_BRIEF: FeatureFlagConfig(
                name=FeatureFlag.ENABLE_RESEARCH_BRIEF,
                enabled=False,
                strategy=RolloutStrategy.PERCENTAGE,
                percentage=0.0,
                override_env_var="FF_RESEARCH_BRIEF",
                description="Enables legacy research consultant to generate a neutral Research Brief shared across consultants",
            ),
            FeatureFlag.ENABLE_ENHANCED_ARBITRATION: FeatureFlagConfig(
                name=FeatureFlag.ENABLE_ENHANCED_ARBITRATION,
                enabled=True,
                strategy=RolloutStrategy.PERCENTAGE,
                percentage=10.0,  # Start with 10% rollout
                override_env_var="FF_ENHANCED_ARBITRATION",
                description="Enables three-state disposition and user-generated critiques",
            ),
            FeatureFlag.ENABLE_USER_GENERATED_CRITIQUES: FeatureFlagConfig(
                name=FeatureFlag.ENABLE_USER_GENERATED_CRITIQUES,
                enabled=True,
                strategy=RolloutStrategy.BETA_USERS,
                beta_group=True,
                override_env_var="FF_USER_CRITIQUES",
                description="Allows users to write their own critiques",
            ),
            FeatureFlag.ENABLE_FLYWHEEL_METRICS: FeatureFlagConfig(
                name=FeatureFlag.ENABLE_FLYWHEEL_METRICS,
                enabled=True,
                strategy=RolloutStrategy.ON,  # Always collect metrics
                override_env_var="FF_FLYWHEEL_METRICS",
                description="Collects detailed metrics for Flywheel learning",
            ),
            FeatureFlag.ENABLE_STAGE0_ENRICHMENT: FeatureFlagConfig(
                name=FeatureFlag.ENABLE_STAGE0_ENRICHMENT,
                enabled=True,
                strategy=RolloutStrategy.A_B_TEST,
                override_env_var="FF_STAGE0_ENRICHMENT",
                description="Controls Stage 0 consultant depth enrichment injections",
                ab_test_config={
                    "experiment_id": "stage0_enrichment_v1",
                    "control_group_size": 0.5,
                    "treatment_group_size": 0.5,
                },
            ),
            # V5.4 Advanced Pipeline Features
            FeatureFlag.ENABLE_CONTEXT_ENGINEERING: FeatureFlagConfig(
                name=FeatureFlag.ENABLE_CONTEXT_ENGINEERING,
                enabled=False,  # Start disabled for safe rollout
                strategy=RolloutStrategy.PERCENTAGE,
                percentage=0.0,
                override_env_var="FF_CONTEXT_ENGINEERING",
                description="Enables context engineering with stage-specific compilation for token optimization",
                ab_test_config={
                    "experiment_id": "context_engineering_v1",
                    "control_group_size": 0.5,
                    "treatment_group_size": 0.5,
                },
            ),
            FeatureFlag.ENABLE_BREADTH_MODE: FeatureFlagConfig(
                name=FeatureFlag.ENABLE_BREADTH_MODE,
                enabled=False,  # Start disabled for safe rollout
                strategy=RolloutStrategy.PERCENTAGE,
                percentage=0.0,
                override_env_var="FF_BREADTH_MODE",
                description="Enables breadth mode for multi-agent parallel execution with strict constraints",
            ),
            FeatureFlag.ENABLE_ADVANCED_PIPELINE: FeatureFlagConfig(
                name=FeatureFlag.ENABLE_ADVANCED_PIPELINE,
                enabled=False,  # Start disabled for safe rollout
                strategy=RolloutStrategy.PERCENTAGE,
                percentage=0.0,
                override_env_var="FF_ADVANCED_PIPELINE",
                description="Enables CognitivePipelineChain with reflection loops and iterative refinement",
            ),
            FeatureFlag.ENABLE_ENHANCED_ROUTING: FeatureFlagConfig(
                name=FeatureFlag.ENABLE_ENHANCED_ROUTING,
                enabled=False,  # Start disabled for safe rollout
                strategy=RolloutStrategy.PERCENTAGE,
                percentage=0.0,
                override_env_var="FF_ENHANCED_ROUTING",
                description="Enables CognitiveConsultantRouter for optimized consultant selection with diversity scoring",
            ),
            FeatureFlag.ENABLE_DEPENDENCY_AWARE_FORGES: FeatureFlagConfig(
                name=FeatureFlag.ENABLE_DEPENDENCY_AWARE_FORGES,
                enabled=False,  # Start disabled for safe rollout
                strategy=RolloutStrategy.PERCENTAGE,
                percentage=0.0,
                override_env_var="FF_DEPENDENCY_AWARE_FORGES",
                description="Enables EnhancedParallelCognitiveForges with dependency management and topological sorting",
            ),
            # Unified from legacy config module (defaults preserved)
            FeatureFlag.ENABLE_USER_ARBITRATION: FeatureFlagConfig(
                name=FeatureFlag.ENABLE_USER_ARBITRATION,
                enabled=True,
                strategy=RolloutStrategy.ON,
                override_env_var="FF_ENABLE_USER_ARBITRATION",
                description="Enable user arbitration of Red Team Council critiques",
            ),
            FeatureFlag.ENABLE_ARBITRATED_SYNTHESIS: FeatureFlagConfig(
                name=FeatureFlag.ENABLE_ARBITRATED_SYNTHESIS,
                enabled=True,
                strategy=RolloutStrategy.ON,
                override_env_var="FF_ENABLE_ARBITRATED_SYNTHESIS",
                description="Enable synthesis that prioritizes user-selected critiques",
            ),
            FeatureFlag.ENABLE_CONTEXT_INTELLIGENCE: FeatureFlagConfig(
                name=FeatureFlag.ENABLE_CONTEXT_INTELLIGENCE,
                enabled=True,
                strategy=RolloutStrategy.ON,
                override_env_var="FF_ENABLE_CONTEXT_INTELLIGENCE",
                description="Enable Context Intelligence Pipeline",
            ),
            FeatureFlag.ENABLE_WHAT_IF_SANDBOX: FeatureFlagConfig(
                name=FeatureFlag.ENABLE_WHAT_IF_SANDBOX,
                enabled=True,
                strategy=RolloutStrategy.ON,
                override_env_var="FF_ENABLE_WHAT_IF_SANDBOX",
                description="Enable What-If analysis sandbox",
            ),
            FeatureFlag.ENABLE_ARBITRATION_ANALYTICS: FeatureFlagConfig(
                name=FeatureFlag.ENABLE_ARBITRATION_ANALYTICS,
                enabled=True,
                strategy=RolloutStrategy.ON,
                override_env_var="FF_ENABLE_ARBITRATION_ANALYTICS",
                description="Track arbitration metrics and critique selection patterns",
                ab_test_config=None,
            ),
            FeatureFlag.ENABLE_TWELVE_FACTOR_COMPLIANCE: FeatureFlagConfig(
                name=FeatureFlag.ENABLE_TWELVE_FACTOR_COMPLIANCE,
                enabled=False,
                strategy=RolloutStrategy.PERCENTAGE,
                percentage=0.0,
                override_env_var="FF_ENABLE_TWELVE_FACTOR_COMPLIANCE",
                description="Enforce 12-Factor compliance for all services",
            ),
            # Prompt policy canary rollout
            FeatureFlag.ENABLE_PROMPT_UPGRADE_CANARY: FeatureFlagConfig(
                name=FeatureFlag.ENABLE_PROMPT_UPGRADE_CANARY,
                enabled=True,
                strategy=RolloutStrategy.PERCENTAGE,
                percentage=10.0,  # Start with 10% canary
                override_env_var="FF_PROMPT_UPGRADE_CANARY",
                description="Routes a percentage of runs to upgraded analysis prompts (MeMo + inquiry complex + ask-back)",
            ),
            # Senior Advisor Synthesis Refinement flags
            FeatureFlag.ENABLE_SA_POLYGON_ENHANCEMENTS: FeatureFlagConfig(
                name=FeatureFlag.ENABLE_SA_POLYGON_ENHANCEMENTS,
                enabled=False,  # Start disabled for safe rollout
                strategy=RolloutStrategy.PERCENTAGE,
                percentage=0.0,
                override_env_var="FF_SA_POLYGON_ENHANCEMENTS",
                description="Enable perspective preservation and structured recommendations",
            ),
            FeatureFlag.ENABLE_SA_DECISION_RIBBON: FeatureFlagConfig(
                name=FeatureFlag.ENABLE_SA_DECISION_RIBBON,
                enabled=False,  # Start disabled for safe rollout
                strategy=RolloutStrategy.PERCENTAGE,
                percentage=0.0,
                override_env_var="FF_SA_DECISION_RIBBON",
                description="Enable decision quality ribbon with transparency metrics",
            ),
            FeatureFlag.ENABLE_SA_EXEC_TABLE: FeatureFlagConfig(
                name=FeatureFlag.ENABLE_SA_EXEC_TABLE,
                enabled=False,  # Start disabled for safe rollout
                strategy=RolloutStrategy.PERCENTAGE,
                percentage=0.0,
                override_env_var="FF_SA_EXEC_TABLE",
                description="Enable executive decision table in reports",
            ),
            FeatureFlag.ENABLE_SA_DISSENT_LEDGER: FeatureFlagConfig(
                name=FeatureFlag.ENABLE_SA_DISSENT_LEDGER,
                enabled=False,  # Start disabled for safe rollout
                strategy=RolloutStrategy.PERCENTAGE,
                percentage=0.0,
                override_env_var="FF_SA_DISSENT_LEDGER",
                description="Enable dissent ledger to prevent perspective collapse",
            ),
            FeatureFlag.ENABLE_SA_GATE_LOGIC: FeatureFlagConfig(
                name=FeatureFlag.ENABLE_SA_GATE_LOGIC,
                enabled=False,  # Start disabled for safe rollout
                strategy=RolloutStrategy.PERCENTAGE,
                percentage=0.0,
                override_env_var="FF_SA_GATE_LOGIC",
                description="Enable synthesis mode gate (converge vs preserve dissent)",
            ),
        }

    def _load_overrides(self) -> Dict[FeatureFlag, bool]:
        """Load environment variable overrides"""
        overrides = {}

        for flag, config in self._flags.items():
            if config.override_env_var:
                env_value = os.getenv(config.override_env_var)
                if env_value is not None:
                    overrides[flag] = env_value.lower() in ("true", "1", "yes", "on")
                    self.logger.info(
                        "feature_flag_override_loaded",
                        flag=flag.value,
                        override_value=overrides[flag],
                    )

        return overrides

    def is_enabled(
        self,
        flag: FeatureFlag,
        user_id: Optional[UUID] = None,
        org_id: Optional[UUID] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Check if a feature flag is enabled for the given context

        Args:
            flag: The feature flag to check
            user_id: Optional user ID for targeting
            org_id: Optional organization ID for targeting
            attributes: Optional attributes for complex targeting

        Returns:
            True if the feature is enabled, False otherwise
        """

        # Check for environment override first
        if flag in self._overrides:
            return self._overrides[flag]

        config = self._flags.get(flag)
        if not config:
            self.logger.warning("unknown_feature_flag", flag=flag.value)
            return False

        if not config.enabled:
            return False

        # Evaluate based on strategy
        if config.strategy == RolloutStrategy.OFF:
            return False

        elif config.strategy == RolloutStrategy.ON:
            return True

        elif config.strategy == RolloutStrategy.PERCENTAGE:
            return self._evaluate_percentage(flag, config.percentage, user_id or org_id)

        elif config.strategy == RolloutStrategy.USER_LIST:
            return user_id and str(user_id) in (config.user_ids or set())

        elif config.strategy == RolloutStrategy.ORG_LIST:
            return org_id and str(org_id) in (config.org_ids or set())

        elif config.strategy == RolloutStrategy.BETA_USERS:
            return self._is_beta_user(user_id, attributes)

        elif config.strategy == RolloutStrategy.A_B_TEST:
            return self._evaluate_ab_test(
                flag, config.ab_test_config, user_id or org_id
            )

        return False

    def get_experiment_group(
        self,
        flag: FeatureFlag,
        user_id: Optional[UUID] = None,
        org_id: Optional[UUID] = None,
    ) -> Optional[str]:
        """
        Get the A/B test group assignment for a user

        Returns:
            'control', 'treatment', 'holdout', or None
        """

        config = self._flags.get(flag)
        if not config or config.strategy != RolloutStrategy.A_B_TEST:
            return None

        if not config.ab_test_config:
            return None

        identifier = user_id or org_id
        if not identifier:
            return None

        # Generate consistent hash for user
        hash_input = f"{flag.value}:{identifier}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = (hash_value % 100) / 100.0

        control_size = config.ab_test_config.get("control_group_size", 0.5)
        treatment_size = config.ab_test_config.get("treatment_group_size", 0.4)

        if bucket < control_size:
            return "control"
        elif bucket < control_size + treatment_size:
            return "treatment"
        else:
            return "holdout"

    def _evaluate_percentage(
        self, flag: FeatureFlag, percentage: float, identifier: Optional[UUID]
    ) -> bool:
        """Evaluate percentage-based rollout"""

        if not identifier:
            # No identifier, use random sampling
            import random

            return random.random() * 100 < percentage

        # Consistent hashing for deterministic assignment
        hash_input = f"{flag.value}:{identifier}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = hash_value % 100

        return bucket < percentage

    def _evaluate_ab_test(
        self, flag: FeatureFlag, ab_config: Dict[str, Any], identifier: Optional[UUID]
    ) -> bool:
        """Evaluate A/B test assignment"""

        group = self.get_experiment_group(flag, identifier)
        return group == "treatment"

    def _is_beta_user(
        self, user_id: Optional[UUID], attributes: Optional[Dict[str, Any]]
    ) -> bool:
        """Check if user is in beta program"""

        if not attributes:
            return False

        return attributes.get("beta_user", False) or attributes.get(
            "early_access", False
        )

    def get_all_flags(
        self,
        user_id: Optional[UUID] = None,
        org_id: Optional[UUID] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, bool]:
        """Get all feature flag states for a given context"""

        return {
            flag.value: self.is_enabled(flag, user_id, org_id, attributes)
            for flag in FeatureFlag
        }

    def update_flag(
        self,
        flag: FeatureFlag,
        enabled: Optional[bool] = None,
        strategy: Optional[RolloutStrategy] = None,
        percentage: Optional[float] = None,
        user_ids: Optional[Set[str]] = None,
        org_ids: Optional[Set[str]] = None,
    ):
        """Update a feature flag configuration (for admin use)"""

        config = self._flags.get(flag)
        if not config:
            return

        if enabled is not None:
            config.enabled = enabled
        if strategy is not None:
            config.strategy = strategy
        if percentage is not None:
            config.percentage = percentage
        if user_ids is not None:
            config.user_ids = user_ids
        if org_ids is not None:
            config.org_ids = org_ids

        config.updated_at = datetime.utcnow()

        self.logger.info(
            "feature_flag_updated",
            flag=flag.value,
            enabled=config.enabled,
            strategy=config.strategy.value,
            percentage=config.percentage,
        )


# Singleton instance
_feature_flag_service = None


def get_feature_flag_service() -> FeatureFlagService:
    """Get the singleton feature flag service"""
    global _feature_flag_service
    if _feature_flag_service is None:
        _feature_flag_service = FeatureFlagService()
    return _feature_flag_service


def is_feature_enabled(
    flag: FeatureFlag,
    user_id: Optional[UUID] = None,
    org_id: Optional[UUID] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> bool:
    """Convenience function to check if a feature is enabled"""
    service = get_feature_flag_service()
    return service.is_enabled(flag, user_id, org_id, attributes)


def get_experiment_group(
    flag: FeatureFlag, user_id: Optional[UUID] = None, org_id: Optional[UUID] = None
) -> Optional[str]:
    """Convenience function to get experiment group"""
    service = get_feature_flag_service()
    return service.get_experiment_group(flag, user_id, org_id)


# Decorator for feature-flagged functions
def feature_flag(flag: FeatureFlag, fallback=None):
    """
    Decorator to conditionally execute functions based on feature flags

    Usage:
        @feature_flag(FeatureFlag.ENABLE_PARALLEL_VALIDATION)
        async def execute_red_team_council():
            ...
    """

    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            # Extract user_id from kwargs if present
            user_id = kwargs.get("user_id")
            org_id = kwargs.get("org_id")

            if is_feature_enabled(flag, user_id=user_id, org_id=org_id):
                return await func(*args, **kwargs)
            elif fallback:
                return await fallback(*args, **kwargs)
            else:
                logger.info(
                    "feature_flag_disabled_skipping",
                    flag=flag.value,
                    function=func.__name__,
                )
                return None

        def sync_wrapper(*args, **kwargs):
            # Extract user_id from kwargs if present
            user_id = kwargs.get("user_id")
            org_id = kwargs.get("org_id")

            if is_feature_enabled(flag, user_id=user_id, org_id=org_id):
                return func(*args, **kwargs)
            elif fallback:
                return fallback(*args, **kwargs)
            else:
                logger.info(
                    "feature_flag_disabled_skipping",
                    flag=flag.value,
                    function=func.__name__,
                )
                return None

        # Return appropriate wrapper based on function type
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
