"""Feature flags adapter - bridges src.core.feature_flags to src.engine"""

from src.core.feature_flags import FeatureFlag, get_experiment_group

__all__ = ["FeatureFlag", "get_experiment_group"]
