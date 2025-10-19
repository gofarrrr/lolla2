"""
Model Configuration Manager - Extracted from model_manager.py
Handles model settings, feature flags, and catalog management
"""

import logging
from typing import Dict, Optional, Any, Set
from dataclasses import dataclass

from src.engine.models.data_contracts import MentalModelDefinition
from src.intelligence.model_catalog import MentalModelCatalog
from src.intelligence.model_selector import SelectionStrategy


@dataclass
class ModelSettings:
    """Model configuration settings"""

    enable_nway_patterns: bool = True
    enable_bayesian_optimization: bool = True
    enable_pattern_matching: bool = True
    enable_confidence_calibration: bool = True
    max_models_per_selection: int = 3
    performance_history_size: int = 100
    similarity_threshold: float = 0.75
    success_threshold: float = 0.7


@dataclass
class FeatureFlags:
    """Feature flags for model functionality"""

    hmw_generation: bool = False
    assumption_challenging: bool = False
    bias_detection: bool = True
    transparency_mode: bool = True
    research_augmentation: bool = False
    tree_search: bool = False
    advanced_analytics: bool = False
    experimental_models: bool = False


class ModelConfigurator:
    """
    Manages model configuration, settings, and feature flags
    """

    def __init__(
        self,
        model_catalog: MentalModelCatalog,
        settings: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.model_catalog = model_catalog
        self.settings = settings
        self.logger = logger or logging.getLogger(__name__)

        # Initialize configuration
        self.model_settings = self._load_model_settings()
        self.feature_flags = self._load_feature_flags()
        self.selection_strategy = SelectionStrategy.ACCURACY_FOCUSED

        # Model filtering and availability
        self._available_models_cache: Optional[Dict[str, MentalModelDefinition]] = None
        self._disabled_models: Set[str] = set()

    def _load_model_settings(self) -> ModelSettings:
        """Load model settings from configuration"""
        try:
            if not self.settings:
                return ModelSettings()

            return ModelSettings(
                enable_nway_patterns=getattr(
                    self.settings, "ENABLE_NWAY_PATTERNS", True
                ),
                enable_bayesian_optimization=getattr(
                    self.settings, "ENABLE_BAYESIAN_OPTIMIZATION", True
                ),
                enable_pattern_matching=getattr(
                    self.settings, "ENABLE_PATTERN_MATCHING", True
                ),
                enable_confidence_calibration=getattr(
                    self.settings, "ENABLE_CONFIDENCE_CALIBRATION", True
                ),
                max_models_per_selection=getattr(
                    self.settings, "MAX_MODELS_PER_SELECTION", 3
                ),
                performance_history_size=getattr(
                    self.settings, "MAX_MODEL_PERFORMANCE_HISTORY", 100
                ),
                similarity_threshold=getattr(
                    self.settings, "SIMILARITY_THRESHOLD", 0.75
                ),
                success_threshold=getattr(self.settings, "SUCCESS_THRESHOLD", 0.7),
            )

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load model settings, using defaults: {e}")
            return ModelSettings()

    def _load_feature_flags(self) -> FeatureFlags:
        """Load feature flags from configuration"""
        try:
            if not self.settings:
                return FeatureFlags()

            return FeatureFlags(
                hmw_generation=getattr(self.settings, "ENABLE_HMW_GENERATION", False),
                assumption_challenging=getattr(
                    self.settings, "ENABLE_ASSUMPTION_CHALLENGING", False
                ),
                bias_detection=getattr(self.settings, "ENABLE_BIAS_DETECTION", True),
                transparency_mode=getattr(
                    self.settings, "ENABLE_TRANSPARENCY_MODE", True
                ),
                research_augmentation=getattr(
                    self.settings, "ENABLE_RESEARCH_AUGMENTATION", False
                ),
                tree_search=getattr(self.settings, "ENABLE_TREE_SEARCH", False),
                advanced_analytics=getattr(
                    self.settings, "ENABLE_ADVANCED_ANALYTICS", False
                ),
                experimental_models=getattr(
                    self.settings, "ENABLE_EXPERIMENTAL_MODELS", False
                ),
            )

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load feature flags, using defaults: {e}")
            return FeatureFlags()

    def get_available_models(
        self, force_refresh: bool = False
    ) -> Dict[str, MentalModelDefinition]:
        """Get available models with feature flag and settings filtering"""
        try:
            # Use cache unless forced refresh
            if self._available_models_cache and not force_refresh:
                return self._available_models_cache

            # Start with all models from catalog
            available_models = {}

            for model_id, model_def in self.model_catalog.models.items():
                # Skip disabled models
                if model_id in self._disabled_models:
                    continue

                # Apply feature flag filtering
                if not self._is_model_enabled_by_features(model_def):
                    continue

                available_models[model_id] = model_def

            # Cache the result
            self._available_models_cache = available_models

            self.logger.debug(
                f"📋 Available models: {len(available_models)}/{len(self.model_catalog.models)}"
            )
            return available_models

        except Exception as e:
            self.logger.error(f"❌ Failed to get available models: {e}")
            # Return all catalog models as fallback
            return self.model_catalog.models

    def _is_model_enabled_by_features(self, model_def: MentalModelDefinition) -> bool:
        """Check if model is enabled based on feature flags"""
        try:
            # Basic models always available
            basic_models = {
                "systems_thinking",
                "critical_thinking",
                "problem_solving",
                "decision_making",
                "strategic_thinking",
                "analytical_thinking",
            }

            if model_def.model_id in basic_models:
                return True

            # Experimental models require flag
            experimental_models = {
                "nway_systems_thinking",
                "nway_critical_thinking",
                "nway_mece_structuring",
                "nway_hypothesis_testing",
            }

            if model_def.model_id in experimental_models:
                return self.feature_flags.experimental_models

            # HMW generation models
            if (
                "hmw" in model_def.model_id.lower()
                or "how_might_we" in model_def.model_id.lower()
            ):
                return self.feature_flags.hmw_generation

            # Assumption challenging models
            if (
                "assumption" in model_def.model_id.lower()
                or "challenge" in model_def.model_id.lower()
            ):
                return self.feature_flags.assumption_challenging

            # Bias detection models
            if "bias" in model_def.model_id.lower():
                return self.feature_flags.bias_detection

            # Tree search models
            if "tree_search" in model_def.model_id.lower():
                return self.feature_flags.tree_search

            # Research augmentation models
            if "research" in model_def.model_id.lower():
                return self.feature_flags.research_augmentation

            # Advanced analytics models
            if (
                hasattr(model_def, "category")
                and "advanced" in model_def.category.lower()
            ):
                return self.feature_flags.advanced_analytics

            # Default: allow model
            return True

        except Exception as e:
            self.logger.warning(
                f"⚠️ Feature flag check failed for {model_def.model_id}: {e}"
            )
            return True

    def configure_selection_strategy(
        self,
        strategy: Optional[SelectionStrategy] = None,
        custom_weights: Optional[Dict[str, float]] = None,
    ) -> SelectionStrategy:
        """Configure model selection strategy"""
        try:
            if strategy:
                self.selection_strategy = strategy

            # Apply custom weights if provided
            if custom_weights:
                # Store custom weights for selection algorithm
                self._custom_selection_weights = custom_weights
                self.logger.debug(
                    f"🎯 Applied custom selection weights: {custom_weights}"
                )

            self.logger.info(
                f"🔧 Selection strategy configured: {self.selection_strategy}"
            )
            return self.selection_strategy

        except Exception as e:
            self.logger.error(f"❌ Failed to configure selection strategy: {e}")
            return SelectionStrategy.ACCURACY_FOCUSED

    def enable_feature(self, feature_name: str) -> bool:
        """Enable a specific feature flag"""
        try:
            if hasattr(self.feature_flags, feature_name):
                setattr(self.feature_flags, feature_name, True)

                # Clear available models cache to apply changes
                self._available_models_cache = None

                self.logger.info(f"✅ Enabled feature: {feature_name}")
                return True
            else:
                self.logger.warning(f"⚠️ Unknown feature flag: {feature_name}")
                return False

        except Exception as e:
            self.logger.error(f"❌ Failed to enable feature {feature_name}: {e}")
            return False

    def disable_feature(self, feature_name: str) -> bool:
        """Disable a specific feature flag"""
        try:
            if hasattr(self.feature_flags, feature_name):
                setattr(self.feature_flags, feature_name, False)

                # Clear available models cache to apply changes
                self._available_models_cache = None

                self.logger.info(f"❌ Disabled feature: {feature_name}")
                return True
            else:
                self.logger.warning(f"⚠️ Unknown feature flag: {feature_name}")
                return False

        except Exception as e:
            self.logger.error(f"❌ Failed to disable feature {feature_name}: {e}")
            return False

    def disable_model(self, model_id: str) -> bool:
        """Temporarily disable a specific model"""
        try:
            if model_id in self.model_catalog.models:
                self._disabled_models.add(model_id)

                # Clear available models cache
                self._available_models_cache = None

                self.logger.info(f"🚫 Disabled model: {model_id}")
                return True
            else:
                self.logger.warning(f"⚠️ Model not found in catalog: {model_id}")
                return False

        except Exception as e:
            self.logger.error(f"❌ Failed to disable model {model_id}: {e}")
            return False

    def enable_model(self, model_id: str) -> bool:
        """Re-enable a disabled model"""
        try:
            if model_id in self._disabled_models:
                self._disabled_models.remove(model_id)

                # Clear available models cache
                self._available_models_cache = None

                self.logger.info(f"✅ Re-enabled model: {model_id}")
                return True
            else:
                self.logger.info(f"ℹ️ Model {model_id} was not disabled")
                return True

        except Exception as e:
            self.logger.error(f"❌ Failed to enable model {model_id}: {e}")
            return False

    def update_model_settings(self, **kwargs) -> bool:
        """Update model settings with provided values"""
        try:
            updated = False

            for key, value in kwargs.items():
                if hasattr(self.model_settings, key):
                    setattr(self.model_settings, key, value)
                    updated = True
                    self.logger.debug(f"🔧 Updated {key}: {value}")
                else:
                    self.logger.warning(f"⚠️ Unknown setting: {key}")

            if updated:
                # Clear available models cache to apply changes
                self._available_models_cache = None
                self.logger.info("🔧 Model settings updated")

            return updated

        except Exception as e:
            self.logger.error(f"❌ Failed to update model settings: {e}")
            return False

    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get complete configuration summary"""
        try:
            return {
                "model_settings": {
                    "enable_nway_patterns": self.model_settings.enable_nway_patterns,
                    "enable_bayesian_optimization": self.model_settings.enable_bayesian_optimization,
                    "enable_pattern_matching": self.model_settings.enable_pattern_matching,
                    "enable_confidence_calibration": self.model_settings.enable_confidence_calibration,
                    "max_models_per_selection": self.model_settings.max_models_per_selection,
                    "performance_history_size": self.model_settings.performance_history_size,
                    "similarity_threshold": self.model_settings.similarity_threshold,
                    "success_threshold": self.model_settings.success_threshold,
                },
                "feature_flags": {
                    "hmw_generation": self.feature_flags.hmw_generation,
                    "assumption_challenging": self.feature_flags.assumption_challenging,
                    "bias_detection": self.feature_flags.bias_detection,
                    "transparency_mode": self.feature_flags.transparency_mode,
                    "research_augmentation": self.feature_flags.research_augmentation,
                    "tree_search": self.feature_flags.tree_search,
                    "advanced_analytics": self.feature_flags.advanced_analytics,
                    "experimental_models": self.feature_flags.experimental_models,
                },
                "selection_strategy": (
                    self.selection_strategy.value
                    if hasattr(self.selection_strategy, "value")
                    else str(self.selection_strategy)
                ),
                "available_models_count": len(self.get_available_models()),
                "disabled_models": list(self._disabled_models),
                "total_catalog_models": len(self.model_catalog.models),
            }

        except Exception as e:
            self.logger.error(f"❌ Failed to generate configuration summary: {e}")
            return {"error": str(e)}

    def validate_configuration(self) -> Dict[str, Any]:
        """Validate current configuration and return any issues"""
        try:
            issues = []
            warnings = []

            # Validate model settings
            if self.model_settings.max_models_per_selection < 1:
                issues.append("max_models_per_selection must be at least 1")

            if self.model_settings.max_models_per_selection > 10:
                warnings.append(
                    "max_models_per_selection is very high, may impact performance"
                )

            if (
                self.model_settings.similarity_threshold < 0.0
                or self.model_settings.similarity_threshold > 1.0
            ):
                issues.append("similarity_threshold must be between 0.0 and 1.0")

            if (
                self.model_settings.success_threshold < 0.0
                or self.model_settings.success_threshold > 1.0
            ):
                issues.append("success_threshold must be between 0.0 and 1.0")

            # Check available models
            available_models = self.get_available_models()
            if len(available_models) == 0:
                issues.append(
                    "No models are available - check feature flags and disabled models"
                )
            elif len(available_models) < 3:
                warnings.append(
                    f"Only {len(available_models)} models available, may limit selection quality"
                )

            # Check for conflicting feature flags
            if (
                self.feature_flags.experimental_models
                and not self.feature_flags.advanced_analytics
            ):
                warnings.append(
                    "Experimental models enabled without advanced analytics"
                )

            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "warnings": warnings,
                "summary": f"{len(available_models)} models available, {len(issues)} issues, {len(warnings)} warnings",
            }

        except Exception as e:
            self.logger.error(f"❌ Configuration validation failed: {e}")
            return {
                "valid": False,
                "issues": [f"Validation error: {e}"],
                "warnings": [],
                "summary": "Configuration validation failed",
            }
