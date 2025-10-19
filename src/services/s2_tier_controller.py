"""
S2 Tier Controller - Dynamic System-2 Kernel Tier Management
===========================================================

Provides dynamic escalation and downgrade capabilities for System-2 reasoning tiers
during runtime based on budget constraints, complexity detection, and confidence levels.

This implements the dynamic tier management requirements from the System-2 Kernel spec.
"""

import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

from .s2_trigger_classifier import S2Tier, S2TriggerDecision


class TierChangeReason(Enum):
    BUDGET_CONSTRAINT = "budget_constraint"
    LOW_CONFIDENCE = "low_confidence"
    DETECTED_INCONSISTENCY = "detected_inconsistency"
    HIGH_STAKES_DETECTED = "high_stakes_detected"
    COMPLEXITY_ESCALATION = "complexity_escalation"
    SIMPLE_TASK_DOWNGRADE = "simple_task_downgrade"
    ERROR_RECOVERY = "error_recovery"
    TIME_PRESSURE = "time_pressure"
    USER_OVERRIDE = "user_override"


@dataclass
class TierChangeEvent:
    """Records a tier change event for auditability"""

    timestamp: float
    from_tier: S2Tier
    to_tier: S2Tier
    reason: TierChangeReason
    context: Dict[str, Any]
    rationale: str


@dataclass
class S2RuntimeMetrics:
    """Runtime metrics for tier decision making"""

    token_budget_used: int = 0
    token_budget_limit: int = 10000
    processing_time_seconds: float = 0.0
    confidence_scores: List[float] = field(default_factory=list)
    error_count: int = 0
    inconsistency_detected: bool = False
    complexity_indicators: Dict[str, Any] = field(default_factory=dict)


class S2TierController:
    """
    Dynamic System-2 Kernel tier controller for runtime escalation/downgrade.

    Monitors runtime conditions and adjusts reasoning tiers to optimize
    for quality vs efficiency trade-offs while respecting budget constraints.
    """

    def __init__(
        self,
        budget_escalation_threshold: float = 0.8,
        budget_downgrade_threshold: float = 0.95,
        confidence_escalation_threshold: float = 0.6,
        time_pressure_threshold: float = 30.0,
    ):
        """
        Initialize the S2 Tier Controller.

        Args:
            budget_escalation_threshold: Budget usage % that triggers tier downgrade consideration
            budget_downgrade_threshold: Budget usage % that forces tier downgrade
            confidence_escalation_threshold: Confidence below which to consider tier escalation
            time_pressure_threshold: Seconds after which to consider tier downgrade
        """
        self.budget_escalation_threshold = budget_escalation_threshold
        self.budget_downgrade_threshold = budget_downgrade_threshold
        self.confidence_escalation_threshold = confidence_escalation_threshold
        self.time_pressure_threshold = time_pressure_threshold

        # State tracking
        self.tier_changes: List[TierChangeEvent] = []
        self.current_tier: S2Tier = S2Tier.S2_DISABLED
        self.initial_tier: S2Tier = S2Tier.S2_DISABLED
        self.start_time: float = time.time()

    def initialize_tier(self, initial_decision: S2TriggerDecision) -> S2Tier:
        """Initialize the controller with the initial tier decision"""
        self.current_tier = initial_decision.tier
        self.initial_tier = initial_decision.tier
        self.start_time = time.time()
        return self.current_tier

    def evaluate_tier_adjustment(
        self,
        runtime_metrics: S2RuntimeMetrics,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[S2TriggerDecision]:
        """
        Evaluate if tier adjustment is needed based on runtime metrics.

        Args:
            runtime_metrics: Current runtime metrics
            context: Additional context for decision making

        Returns:
            S2TriggerDecision if tier change is recommended, None otherwise
        """
        context = context or {}

        # Calculate current usage metrics
        budget_usage = runtime_metrics.token_budget_used / max(
            runtime_metrics.token_budget_limit, 1
        )
        avg_confidence = sum(runtime_metrics.confidence_scores) / max(
            len(runtime_metrics.confidence_scores), 1
        )
        elapsed_time = time.time() - self.start_time

        # Priority 1: Budget constraints (always enforced)
        if budget_usage >= self.budget_downgrade_threshold:
            new_tier = self._downgrade_tier(self.current_tier)
            if new_tier != self.current_tier:
                return self._create_tier_decision(
                    new_tier,
                    TierChangeReason.BUDGET_CONSTRAINT,
                    f"Budget usage {budget_usage:.1%} exceeds threshold {self.budget_downgrade_threshold:.1%}",
                    runtime_metrics,
                    context,
                )

        # Priority 2: Error recovery and inconsistency detection
        if runtime_metrics.inconsistency_detected:
            new_tier = self._escalate_tier(self.current_tier)
            if new_tier != self.current_tier:
                return self._create_tier_decision(
                    new_tier,
                    TierChangeReason.DETECTED_INCONSISTENCY,
                    "Inconsistency detected in reasoning chain, escalating for verification",
                    runtime_metrics,
                    context,
                )

        if runtime_metrics.error_count >= 3:
            new_tier = self._escalate_tier(self.current_tier)
            if new_tier != self.current_tier:
                return self._create_tier_decision(
                    new_tier,
                    TierChangeReason.ERROR_RECOVERY,
                    f"High error count ({runtime_metrics.error_count}) requires deeper reasoning",
                    runtime_metrics,
                    context,
                )

        # Priority 3: Confidence-based escalation (if budget allows)
        if (
            budget_usage < self.budget_escalation_threshold
            and avg_confidence < self.confidence_escalation_threshold
            and len(runtime_metrics.confidence_scores) >= 2
        ):

            new_tier = self._escalate_tier(self.current_tier)
            if new_tier != self.current_tier:
                return self._create_tier_decision(
                    new_tier,
                    TierChangeReason.LOW_CONFIDENCE,
                    f"Low confidence ({avg_confidence:.2f}) below threshold {self.confidence_escalation_threshold}",
                    runtime_metrics,
                    context,
                )

        # Priority 4: Time pressure downgrade
        if elapsed_time > self.time_pressure_threshold:
            new_tier = self._downgrade_tier(self.current_tier)
            if new_tier != self.current_tier:
                return self._create_tier_decision(
                    new_tier,
                    TierChangeReason.TIME_PRESSURE,
                    f"Time pressure ({elapsed_time:.1f}s) exceeds threshold {self.time_pressure_threshold}s",
                    runtime_metrics,
                    context,
                )

        # Priority 5: Complexity-based adjustments
        complexity_score = self._calculate_complexity_score(
            runtime_metrics.complexity_indicators
        )
        if complexity_score > 0.8 and budget_usage < self.budget_escalation_threshold:
            new_tier = self._escalate_tier(self.current_tier)
            if new_tier != self.current_tier:
                return self._create_tier_decision(
                    new_tier,
                    TierChangeReason.COMPLEXITY_ESCALATION,
                    f"High complexity score ({complexity_score:.2f}) requires deeper reasoning",
                    runtime_metrics,
                    context,
                )
        elif complexity_score < 0.3 and self.current_tier != S2Tier.S2_DISABLED:
            new_tier = self._downgrade_tier(self.current_tier)
            if new_tier != self.current_tier:
                return self._create_tier_decision(
                    new_tier,
                    TierChangeReason.SIMPLE_TASK_DOWNGRADE,
                    f"Low complexity score ({complexity_score:.2f}) allows simpler reasoning",
                    runtime_metrics,
                    context,
                )

        return None

    def apply_tier_change(self, decision: S2TriggerDecision) -> None:
        """Apply a tier change and record the event"""
        if decision.tier != self.current_tier:
            # Extract change reason from rationale (set by _create_tier_decision)
            reason = getattr(decision, "_change_reason", TierChangeReason.USER_OVERRIDE)
            context = getattr(decision, "_change_context", {})

            event = TierChangeEvent(
                timestamp=time.time(),
                from_tier=self.current_tier,
                to_tier=decision.tier,
                reason=reason,
                context=context,
                rationale=decision.rationale,
            )

            self.tier_changes.append(event)
            self.current_tier = decision.tier

    def get_tier_change_history(self) -> List[TierChangeEvent]:
        """Get the history of tier changes for audit purposes"""
        return self.tier_changes.copy()

    def _escalate_tier(self, current_tier: S2Tier) -> S2Tier:
        """Escalate to next higher tier"""
        escalation_map = {
            S2Tier.S2_DISABLED: S2Tier.S2_TIER_1,
            S2Tier.S2_TIER_1: S2Tier.S2_TIER_2,
            S2Tier.S2_TIER_2: S2Tier.S2_TIER_3,
            S2Tier.S2_TIER_3: S2Tier.S2_TIER_3,  # Already at max
        }
        return escalation_map.get(current_tier, current_tier)

    def _downgrade_tier(self, current_tier: S2Tier) -> S2Tier:
        """Downgrade to next lower tier"""
        downgrade_map = {
            S2Tier.S2_TIER_3: S2Tier.S2_TIER_2,
            S2Tier.S2_TIER_2: S2Tier.S2_TIER_1,
            S2Tier.S2_TIER_1: S2Tier.S2_DISABLED,
            S2Tier.S2_DISABLED: S2Tier.S2_DISABLED,  # Already at min
        }
        return downgrade_map.get(current_tier, current_tier)

    def _calculate_complexity_score(
        self, complexity_indicators: Dict[str, Any]
    ) -> float:
        """Calculate complexity score from 0.0 to 1.0"""
        if not complexity_indicators:
            return 0.0

        score = 0.0

        # Reasoning chain length
        chain_length = complexity_indicators.get("reasoning_chain_length", 0)
        score += min(chain_length / 10.0, 0.3)  # Max 0.3 for chain length

        # Number of variables/entities
        entity_count = complexity_indicators.get("entity_count", 0)
        score += min(entity_count / 20.0, 0.2)  # Max 0.2 for entities

        # Cross-domain references
        cross_domain = complexity_indicators.get("cross_domain_references", 0)
        score += min(cross_domain / 5.0, 0.2)  # Max 0.2 for cross-domain

        # Uncertainty indicators
        uncertainty_score = complexity_indicators.get("uncertainty_score", 0.0)
        score += min(uncertainty_score, 0.3)  # Max 0.3 for uncertainty

        return min(score, 1.0)

    def _create_tier_decision(
        self,
        new_tier: S2Tier,
        reason: TierChangeReason,
        rationale: str,
        runtime_metrics: S2RuntimeMetrics,
        context: Dict[str, Any],
    ) -> S2TriggerDecision:
        """Create a tier decision with change tracking metadata"""
        decision = S2TriggerDecision(tier=new_tier, rationale=rationale)

        # Attach metadata for apply_tier_change
        decision._change_reason = reason
        decision._change_context = {
            "budget_usage": runtime_metrics.token_budget_used
            / max(runtime_metrics.token_budget_limit, 1),
            "confidence_scores": runtime_metrics.confidence_scores.copy(),
            "error_count": runtime_metrics.error_count,
            "processing_time": time.time() - self.start_time,
            **context,
        }

        return decision
