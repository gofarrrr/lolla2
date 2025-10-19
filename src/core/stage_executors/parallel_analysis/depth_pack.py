"""
DepthPack - Stage 0 enrichment integration.

Responsibilities:
- Build depth packs for consultants (mental models, context)
- Integrate Q&A precision retrieval
- Apply breadth/depth variant treatment
- Track enrichment metrics
- Wrap aggregated output with enrichment metadata
"""

import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .interfaces import DepthPack
from .types import AggregatedOutput, EnrichedOutput, DepthContext, EnrichmentLevel
from src.services.depth_enrichment.consultant_depth_pack_builder import (
    ConsultantDepthPackBuilder,
)

logger = logging.getLogger(__name__)


@dataclass
class _MinimalConsultantBlueprint:
    """Minimal consultant blueprint for depth pack builder compatibility"""
    consultant_id: str
    consultant_type: str
    specialization: str = ""
    predicted_effectiveness: float = 0.7
    assigned_dimensions: list = None

    def __post_init__(self):
        if self.assigned_dimensions is None:
            self.assigned_dimensions = []


class StandardDepthPack(DepthPack):
    """
    Standard implementation of DepthPack interface.

    Wraps ConsultantDepthPackBuilder and provides:
    - Depth pack construction for individual consultants
    - Enrichment metadata tracking
    - Variant-aware enrichment (breadth/depth/full)
    - Q&A precision retrieval integration
    """

    def __init__(
        self,
        depth_pack_builder: Optional[ConsultantDepthPackBuilder] = None,
    ):
        """
        Initialize StandardDepthPack.

        Args:
            depth_pack_builder: Optional existing builder (or creates new one)
        """
        self.depth_pack_builder = depth_pack_builder or ConsultantDepthPackBuilder()

        # Track enrichment metrics across calls
        self._metrics: Dict[str, Any] = {
            "total_depth_tokens": 0,
            "mm_items_count": 0,
            "stage0_latency_ms": 0,
            "variant_label": "none",
        }

    def reset_metrics(self, variant_label: str = "none") -> None:
        """Reset enrichment metrics prior to building a new depth pack batch."""
        self._metrics = {
            "total_depth_tokens": 0,
            "mm_items_count": 0,
            "stage0_latency_ms": 0,
            "variant_label": variant_label,
        }

    def enrich(
        self,
        aggregated: AggregatedOutput,
        context: DepthContext,
    ) -> EnrichedOutput:
        """
        Apply depth enrichment to aggregated output.

        This wraps the aggregated output with enrichment metadata.
        The actual depth pack building happens in build_consultant_depth_pack().

        Args:
            aggregated: Base aggregated output
            context: Depth enrichment context/config

        Returns:
            EnrichedOutput with depth pack metadata

        Note:
            - If context.enable_stage0=False, returns aggregated as-is
            - Enrichment is idempotent (can be called multiple times safely)
        """
        # If Stage 0 is disabled, return aggregated output without enrichment
        if not context.enable_stage0:
            return EnrichedOutput(
                base_output=aggregated,
                enrichment_applied=False,
                enrichment_level=EnrichmentLevel.NONE,
                depth_pack_tokens=0,
                mm_items_count=0,
                stage0_latency_ms=0,
            )

        # Apply enrichment metadata from tracked metrics
        self._metrics["variant_label"] = context.variant_label
        return EnrichedOutput(
            base_output=aggregated,
            enrichment_applied=True,
            enrichment_level=context.enrichment_level,
            depth_pack_tokens=self._metrics["total_depth_tokens"],
            mm_items_count=self._metrics["mm_items_count"],
            stage0_latency_ms=self._metrics["stage0_latency_ms"],
        )

    def build_consultant_depth_pack(
        self,
        consultant_id: str,
        consultant_type: str,
        problem_context: str,
        enrichment_level: str,
    ) -> str:
        """
        Build depth pack for a single consultant.

        Args:
            consultant_id: Consultant identifier
            consultant_type: Consultant type (strategic, tactical, etc.)
            problem_context: Problem description
            enrichment_level: Enrichment depth (breadth/depth/full/none)

        Returns:
            Depth pack text to inject into consultant prompt
        """
        start_time = time.time()

        # If enrichment is disabled, return empty string
        if enrichment_level == "none" or enrichment_level == EnrichmentLevel.NONE:
            return ""

        # Create minimal consultant blueprint for builder
        blueprint = _MinimalConsultantBlueprint(
            consultant_id=consultant_id,
            consultant_type=consultant_type,
        )

        try:
            # Build depth pack using existing builder
            depth_pack_result = self.depth_pack_builder.build_depth_pack(
                blueprint,
                problem_context=problem_context,
                candidate_models=None,
            )

            depth_pack_text = depth_pack_result.text if hasattr(depth_pack_result, "text") else str(depth_pack_result)
            depth_metadata = depth_pack_result.metadata if hasattr(depth_pack_result, "metadata") else {}

            # Update metrics
            mm_items = depth_metadata.get("mm_items", [])
            token_estimate = self._estimate_tokens(depth_pack_text)

            self._metrics["total_depth_tokens"] += token_estimate
            self._metrics["mm_items_count"] += len(mm_items)

            elapsed_ms = int((time.time() - start_time) * 1000)
            self._metrics["stage0_latency_ms"] += elapsed_ms

            logger.debug(
                f"Built depth pack for {consultant_id}: {len(mm_items)} MM items, {token_estimate} tokens, {elapsed_ms}ms"
            )

            return depth_pack_text

        except Exception as exc:
            logger.warning(f"Failed to build depth pack for {consultant_id}: {exc}")
            return ""

    def get_enrichment_metrics(self) -> Dict[str, Any]:
        """
        Get depth enrichment metrics.

        Returns:
            Dict with metrics:
            - total_depth_tokens: int
            - mm_items_count: int (mental model items)
            - stage0_latency_ms: int
            - variant_label: str
        """
        return self._metrics.copy()

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Simple heuristic: 1 token ≈ 4 characters

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        if not text:
            return 0
        return len(text) // 4
