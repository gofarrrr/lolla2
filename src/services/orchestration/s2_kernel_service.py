# src/services/orchestration/s2_kernel_service.py
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.services.s2_tier_controller import S2RuntimeMetrics

logger = logging.getLogger(__name__)


@dataclass
class S2TierResult:
    s2_tier: str
    rationale: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    tier_history: List[Dict[str, Any]] = field(default_factory=list)


class S2KernelOrchestrationService:
    """
    Encapsulates S2 Kernel classification and dynamic tier adjustments.
    """

    def __init__(
        self,
        s2_classifier: Any,
        s2_tier_controller: Any,
        kernel_enabled: bool,
        tier_override: str = "auto",
    ) -> None:
        self.s2_classifier = s2_classifier
        self.s2_tier_controller = s2_tier_controller
        self.kernel_enabled = kernel_enabled
        self.tier_override = tier_override
        self._metrics: Optional[S2RuntimeMetrics] = None

    async def determine_tier(
        self,
        framework: Any,
        task_classification: Dict[str, Any],
        enhanced_query: Optional[str] = None,
    ) -> S2TierResult:
        if not self.kernel_enabled:
            logger.info("⚡ System-2 Kernel disabled - using fast path")
            return S2TierResult(
                s2_tier="S2_DISABLED", rationale="kernel disabled", metrics={}
            )

        if self.tier_override and self.tier_override != "auto":
            logger.info("🧠 S2 Kernel: Manual tier override = %s", self.tier_override)
            return S2TierResult(
                s2_tier=self.tier_override, rationale="manual override", metrics={}
            )

        # Auto classification
        query_text = enhanced_query or (
            f"{framework.framework_type.value} analysis with {len(framework.primary_dimensions)} dimensions"
        )
        metadata = {
            "framework_type": framework.framework_type.value,
            "complexity": framework.complexity_assessment,
            "domain": task_classification.get("primary_domain", "unknown"),
            "task_type": task_classification.get("task_type", "unknown"),
            "processing_time_seconds": getattr(
                framework, "processing_time_seconds", 30
            ),
        }
        s2_res = self.s2_classifier.classify(query_text, metadata)
        s2_tier = s2_res.tier.value
        rationale = s2_res.rationale
        # Initialize controller and runtime metrics
        self.s2_tier_controller.initialize_tier(s2_res)
        self._metrics = S2RuntimeMetrics(
            token_budget_limit=10000,
            complexity_indicators={
                "reasoning_chain_length": len(framework.primary_dimensions),
                "entity_count": len(framework.primary_dimensions)
                + len(getattr(framework, "secondary_considerations", []) or []),
                "uncertainty_score": (
                    0.5
                    if framework.complexity_assessment == "Medium complexity"
                    else 0.3
                ),
            },
        )
        logger.info("🧠 S2 Kernel: Initial Tier = %s (%s)", s2_tier, rationale)
        return S2TierResult(s2_tier=s2_tier, rationale=rationale, metrics={})

    def update_after_team_selection(
        self,
        current: S2TierResult,
        team_synergy_data: Dict[str, Any],
        consultant_count: int,
    ) -> S2TierResult:
        if not self.kernel_enabled or self._metrics is None:
            return current
        # Update metrics like orchestrator did
        confidence_score = team_synergy_data.get("synergy_bonus", 0.0) + 0.8
        self._metrics.confidence_scores.append(confidence_score)
        self._metrics.token_budget_used += 500
        tier_adjustment = self.s2_tier_controller.evaluate_tier_adjustment(
            self._metrics,
            {
                "stage": "post_team_selection",
                "team_synergy": team_synergy_data.get("synergy_bonus", 0.0),
                "consultant_count": consultant_count,
            },
        )
        if tier_adjustment:
            self.s2_tier_controller.apply_tier_change(tier_adjustment)
            new_tier = tier_adjustment.tier.value
            current.s2_tier = new_tier
            current.rationale = tier_adjustment.rationale
            logger.info(
                "🎚️ S2 TIER ADJUSTED: %s - %s", new_tier, tier_adjustment.rationale
            )
        return current

    def finalize_evaluation(
        self, current: S2TierResult, processing_time: float
    ) -> S2TierResult:
        if not self.kernel_enabled or self._metrics is None:
            return current
        self._metrics.processing_time_seconds = processing_time
        self._metrics.token_budget_used += 200
        final_tier_adjustment = self.s2_tier_controller.evaluate_tier_adjustment(
            self._metrics,
            {
                "stage": "final_evaluation",
                "total_processing_time": processing_time,
                "dispatch_complete": True,
            },
        )
        if final_tier_adjustment:
            self.s2_tier_controller.apply_tier_change(final_tier_adjustment)
            current.s2_tier = final_tier_adjustment.tier.value
            current.rationale = final_tier_adjustment.rationale
            logger.info(
                "🎚️ FINAL S2 TIER ADJUSTMENT: %s - %s",
                current.s2_tier,
                current.rationale,
            )
        # Tier change history for audit
        try:
            history = self.s2_tier_controller.get_tier_change_history()
            if history:
                current.tier_history = [
                    {
                        "from": h.from_tier.value,
                        "to": h.to_tier.value,
                        "reason": h.reason.value,
                    }
                    for h in history
                ]
        except Exception:
            pass
        return current
