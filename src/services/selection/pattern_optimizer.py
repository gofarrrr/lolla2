# src/services/selection/pattern_optimizer.py
from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.services.selection.pattern_contracts import (
    IPatternOptimizer,
    ScoreReport,
    OptimizationResult,
)

logger = logging.getLogger(__name__)


class V1PatternOptimizer(IPatternOptimizer):
    """
    V1 implementation of IPatternOptimizer extracted from the legacy
    _select_optimal_patterns method (Keystone K-02).

    It consumes a ScoreReport and contextual information and reproduces the
    exact selection behavior while remaining decoupled from the legacy service.
    """

    def optimize(
        self,
        report: ScoreReport,
        context: Dict[str, Any],
        max_patterns: int = 3,
    ) -> OptimizationResult:
        # Short-circuit when there are no candidates
        candidate_ids: List[str] = list(report.top_patterns or [])
        if not candidate_ids:
            return OptimizationResult(
                selected_patterns=[],
                primary_pattern=None,
                confidence_score=0.0,
                rationale="No candidates provided",
                fallback_patterns=[],
            )

        scores = report.scores or {}
        # Preserve original ordering from top_patterns; ensure IDs exist in scores
        candidate_ids = [pid for pid in candidate_ids if pid in scores]

        # Context unpacking
        task_classification: Dict[str, Any] = context.get("task_classification", {}) or {}
        cluster_map: Dict[str, str] = context.get("cluster_map", {}) or {}

        # Determine selection parameters based on complexity
        complexity = str(task_classification.get("complexity", "medium")).lower()
        domain_complexity = str(task_classification.get("domain_complexity", "medium")).lower()

        if complexity == "high" or domain_complexity == "high":
            target_max = 3
            min_relevance_threshold = 0.35
            force_multi_pattern = True
            complexity_reason = "high complexity query"
        elif complexity == "medium" and len(candidate_ids) >= 3:
            target_max = 2
            min_relevance_threshold = 0.4
            force_multi_pattern = False
            complexity_reason = "medium complexity with good pattern diversity"
        else:
            target_max = 3
            min_relevance_threshold = 0.4
            force_multi_pattern = False
            complexity_reason = "standard pattern selection"

        # Respect external cap if provided (without changing legacy defaults)
        max_pick = min(target_max, max_patterns or target_max)

        # Primary pattern (highest scoring)
        primary_pattern = candidate_ids[0]
        selected_patterns: List[str] = [primary_pattern]
        used_clusters = set()
        if cluster_map.get(primary_pattern) is not None:
            used_clusters.add(cluster_map.get(primary_pattern))

        logger.info(
            f"🧩 MULTI-PATTERN SELECTION: {complexity_reason} - targeting {max_pick} patterns (threshold: {min_relevance_threshold})"
        )

        # Add complementary patterns based on complexity requirements
        for pid in candidate_ids[1:]:
            if len(selected_patterns) >= max_pick:
                break
            cluster = cluster_map.get(pid)
            if cluster not in used_clusters and scores.get(pid, 0.0) > min_relevance_threshold:
                selected_patterns.append(pid)
                if cluster is not None:
                    used_clusters.add(cluster)
                logger.debug(
                    f"✅ Added pattern {pid} from {cluster} cluster (score: {scores.get(pid, 0.0):.3f})"
                )

        # Force multi-pattern for high complexity if still single
        if force_multi_pattern and len(selected_patterns) == 1 and len(candidate_ids) > 1:
            for pid in candidate_ids[1:]:
                if pid not in selected_patterns:
                    selected_patterns.append(pid)
                    logger.info(
                        f"🎯 COMPLEXITY OVERRIDE: Added {pid} for high complexity (score: {scores.get(pid, 0.0):.3f})"
                    )
                    break

        # Fallback patterns (next best options)
        fallback_patterns: List[str] = []
        for pid in candidate_ids:
            if pid not in selected_patterns and len(fallback_patterns) < 3:
                fallback_patterns.append(pid)

        # Confidence calculation based on top score and coverage
        top_score = scores.get(primary_pattern, 0.0)
        confidence = top_score * 0.7 + min(len(selected_patterns) / 3.0, 1.0) * 0.3

        # Rationale mirroring legacy wording (minus strategy enum coupling)
        rationale = f"Selected {len(selected_patterns)} patterns: {', '.join(selected_patterns)}. "
        rationale += f"Multi-pattern strategy: {complexity_reason}. "
        if len(selected_patterns) > 1:
            # Count only non-None clusters
            cluster_count = len({c for c in used_clusters if c is not None}) or 1
            rationale += (
                f"Complex query benefits from {len(selected_patterns)} complementary patterns "
            )
            rationale += f"across {cluster_count} clusters. "
        else:
            rationale += "Single pattern sufficient for query complexity. "

        primary_rationale = report.rationale.get(primary_pattern, "primary pattern rationale unavailable")
        rationale += f"Primary: {primary_rationale}."

        return OptimizationResult(
            selected_patterns=selected_patterns,
            primary_pattern=primary_pattern,
            confidence_score=min(confidence, 1.0),
            rationale=rationale,
            fallback_patterns=fallback_patterns,
        )
