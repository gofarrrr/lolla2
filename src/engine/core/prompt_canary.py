#!/usr/bin/env python3
"""
Prompt Canary Service
- Assigns control vs upgrade variant per engagement/trace using FeatureFlagService
- Emits PROMPT_POLICY_VARIANT_ASSIGNED event into UnifiedContextStream
"""
from dataclasses import dataclass
import hashlib
from typing import Optional

from src.engine.core.feature_flags import FeatureFlagService, FeatureFlag, RolloutStrategy
from src.core.unified_context_stream import UnifiedContextStream, ContextEventType


def _stable_bucket(s: str) -> float:
    """Map a string to a stable bucket in [0, 100)."""
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    # Use first 8 hex chars -> int -> modulo 100
    val = int(h[:8], 16)
    return float(val % 100)


@dataclass
class PromptVariantAssignment:
    trace_id: str
    variant: str  # "control" or "upgrade"
    percentage: float


class PromptCanaryService:
    def __init__(self, context_stream: UnifiedContextStream, ff_service: Optional[FeatureFlagService] = None):
        self.context_stream = context_stream
        self.ff_service = ff_service or FeatureFlagService()

    def assign_variant(self, trace_id: Optional[str] = None, engagement_id: Optional[str] = None) -> PromptVariantAssignment:
        """Assign control vs upgrade variant based on feature flag percentage.

        Emits a PROMPT_POLICY_VARIANT_ASSIGNED event.
        """
        key = (trace_id or engagement_id or self.context_stream.trace_id)
        cfg = self.ff_service._flags[FeatureFlag.ENABLE_PROMPT_UPGRADE_CANARY]

        # Default to control if disabled
        variant = "control"
        percentage = 0.0

        if cfg.enabled and cfg.strategy in {RolloutStrategy.PERCENTAGE}:
            percentage = float(cfg.percentage or 0.0)
            bucket = _stable_bucket(key)
            if bucket < percentage:
                variant = "upgrade"
        elif cfg.enabled and cfg.strategy == RolloutStrategy.ON:
            variant = "upgrade"
            percentage = 100.0

        assignment = PromptVariantAssignment(trace_id=self.context_stream.trace_id, variant=variant, percentage=percentage)

        # Emit event
        self.context_stream.add_event(
            ContextEventType.PROMPT_POLICY_VARIANT_ASSIGNED,
            data={
                "trace_id": self.context_stream.trace_id,
                "variant": assignment.variant,
                "percentage": assignment.percentage,
                "bucketing_key": key,
            },
        )
        return assignment

    @staticmethod
    def is_upgrade_variant(assignment: PromptVariantAssignment) -> bool:
        return assignment.variant == "upgrade"
