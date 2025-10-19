# src/services/selection/pattern_scorer.py
from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.services.selection.pattern_contracts import (
    IPatternScorer,
    Pattern,
    ScoreReport,
)

logger = logging.getLogger(__name__)


class V1PatternScorer(IPatternScorer):
    """
    V1 implementation of IPatternScorer.

    Mirrors the legacy scoring behavior by computing a weighted relevance score per pattern
    based on domain, consultant alignment, framework fit, S2 alignment, and complexity fit.
    Also produces per-pattern rationales and a primary selection strategy tag.
    """

    def score(self, patterns: List[Pattern], context: Dict[str, Any]) -> ScoreReport:
        framework_type: str = context.get("framework_type", "")
        task_classification: Dict[str, Any] = context.get("task_classification", {})
        consultant_types: List[str] = context.get("consultant_types", [])
        complexity: str = context.get("complexity", "medium")
        s2_tier: str = context.get("s2_tier", "S2_DISABLED")

        scores: Dict[str, float] = {}
        rationale: Dict[str, str] = {}
        strategy_by_id: Dict[str, str] = {}

        for p in patterns:
            score, rationale_text, strategy = self._score_individual_pattern(
                p, framework_type, task_classification, consultant_types, complexity, s2_tier
            )
            if score > 0.0:
                scores[p.id] = min(score, 1.0)
                rationale[p.id] = rationale_text
                strategy_by_id[p.id] = strategy

        # Sort by score desc
        top_patterns = sorted(scores.keys(), key=lambda pid: scores[pid], reverse=True)

        return ScoreReport(
            top_patterns=top_patterns,
            scores=scores,
            rationale=rationale,
            context_factors={"strategy": strategy_by_id},
        )

    # ---------- Legacy-equivalent scoring helpers ----------
    def _score_individual_pattern(
        self,
        pattern: Pattern,
        framework_type: str,
        task_classification: Dict[str, Any],
        consultant_types: List[str],
        complexity: str,
        s2_tier: str,
    ) -> (float, str, str):
        score = 0.0
        strategy = "default_fallback"
        rationale_parts: List[str] = []

        domain_score = self._calculate_domain_match(pattern, framework_type, task_classification)
        score += domain_score * 0.3
        if domain_score > 0.7:
            strategy = "domain_matched"
            rationale_parts.append(f"domain match ({domain_score:.2f})")

        consultant_score = self._calculate_consultant_alignment(pattern, consultant_types)
        score += consultant_score * 0.25
        if consultant_score > 0.8:
            strategy = "consultant_aligned"
            rationale_parts.append(f"consultant aligned ({consultant_score:.2f})")

        framework_score = self._calculate_framework_fit(pattern, framework_type, complexity)
        score += framework_score * 0.2
        if framework_score > 0.7:
            strategy = "framework_optimized"
            rationale_parts.append(f"framework optimized ({framework_score:.2f})")

        s2_score = self._calculate_s2_alignment(pattern, s2_tier)
        score += s2_score * 0.15
        if s2_score > 0.6:
            rationale_parts.append(f"S2 aligned ({s2_score:.2f})")

        complexity_score = self._calculate_complexity_fit(pattern, complexity)
        score += complexity_score * 0.1
        if complexity_score > 0.8:
            strategy = "complexity_adaptive"
            rationale_parts.append(f"complexity adaptive ({complexity_score:.2f})")

        rationale = f"{(pattern.cluster or 'unknown')} cluster: " + ", ".join(rationale_parts)
        return score, rationale, strategy

    def _calculate_domain_match(
        self,
        pattern: Pattern,
        framework_type: str,
        task_classification: Dict[str, Any],
    ) -> float:
        cluster = (pattern.cluster or "").lower()
        framework_lower = (framework_type or "").lower()
        domain = ((task_classification or {}).get("primary_domain", "") or "").lower()

        domain_clusters = {
            "strategic": ["perception", "decision", "synthesis", "reasoning"],
            "operational": ["execution", "decomposition", "compliance"],
            "financial": ["financial", "complexity", "decision"],
            "technical": ["reasoning", "decomposition", "complexity"],
            "innovation": ["synthesis", "reasoning", "metacognition"],
        }
        framework_clusters = {
            "strategic_analysis": ["perception", "decision", "synthesis"],
            "operational_optimization": ["execution", "decomposition"],
            "innovation_discovery": ["synthesis", "metacognition", "reasoning"],
            "crisis_management": ["decision", "execution", "behavioral"],
        }

        score = 0.0
        for domain_key, clusters in domain_clusters.items():
            if domain_key in domain and cluster in clusters:
                score += 0.4
                break
        for framework_key, clusters in framework_clusters.items():
            if framework_key in framework_lower and cluster in clusters:
                score += 0.6
                break
        return min(score, 1.0)

    def _calculate_consultant_alignment(
        self, pattern: Pattern, consultant_types: List[str]
    ) -> float:
        pattern_priority = pattern.consultant_priority or []
        if not pattern_priority or not consultant_types:
            return 0.3
        matches = 0
        for consultant_type in consultant_types:
            consultant_base = consultant_type.replace("_", " ").lower()
            for priority in pattern_priority:
                if priority.lower() in consultant_base or consultant_base in priority.lower():
                    matches += 1
                    break
        if not consultant_types:
            return 0.3
        return min(matches / len(consultant_types), 1.0)

    def _calculate_framework_fit(
        self, pattern: Pattern, framework_type: str, complexity: str
    ) -> float:
        title = (pattern.title or "").lower()
        framework_lower = (framework_type or "").lower()
        framework_keywords = {
            "strategic": ["strategy", "strategic", "planning", "vision", "analysis"],
            "operational": ["operation", "process", "execution", "efficiency"],
            "innovation": ["innovation", "creative", "discovery", "breakthrough"],
            "crisis": ["crisis", "emergency", "rapid", "urgent", "critical"],
        }
        score = 0.0
        for framework_key, keywords in framework_keywords.items():
            if framework_key in framework_lower:
                for keyword in keywords:
                    if keyword in title:
                        score += 0.2
        if "high" in (complexity or "").lower() and "complex" in title:
            score += 0.3
        elif "medium" in (complexity or "").lower():
            score += 0.1
        return min(score, 1.0)

    def _calculate_s2_alignment(self, pattern: Pattern, s2_tier: str) -> float:
        s2_triggers = pattern.system2_triggers or []
        if not s2_triggers:
            return 0.3
        if s2_tier == "S2_DISABLED":
            return 0.2
        elif s2_tier == "S2_TIER_1":
            return 0.6
        elif s2_tier == "S2_TIER_2":
            return 0.8
        elif s2_tier == "S2_TIER_3":
            return 1.0
        return 0.3

    def _calculate_complexity_fit(self, pattern: Pattern, complexity: str) -> float:
        models = pattern.models or []
        title = (pattern.title or "").lower()
        model_count = len(models)
        complexity_lower = (complexity or "").lower()
        if "high" in complexity_lower:
            if model_count >= 5:
                return 0.8
            elif model_count >= 3:
                return 0.6
        elif "medium" in complexity_lower:
            if 3 <= model_count <= 6:
                return 0.8
            elif model_count >= 2:
                return 0.6
        else:
            if model_count <= 3:
                return 0.8
            elif model_count <= 5:
                return 0.5
        complexity_indicators = ["complex", "sophisticated", "advanced", "deep"]
        for indicator in complexity_indicators:
            if indicator in title:
                if "high" in complexity_lower:
                    return 0.9
                else:
                    return 0.4
        return 0.5
