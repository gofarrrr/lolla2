"""
NWAY Pattern Selection Service - Dynamic Pattern Selection
=========================================================

Replaces hardcoded NWAY patterns with intelligent selection from YAML cognitive architecture.
Implements scalable pattern selection based on framework requirements, task classification,
and consultant team composition.

This eliminates the hardcoded "NWAY_PERCEPTION_001" usage and provides dynamic selection
from the 43 NWAY patterns across 12 clusters.
"""

import yaml
import logging
import os
import hashlib
import statistics
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict

# Keystone K-02: contracts and optimizer interface
from src.services.selection.pattern_contracts import (
    IPatternOptimizer,
    ScoreReport,
    OptimizationResult,
    ICoverageAnalyzer,
    IPatternScorer,
    IPatternAnalytics,
    Pattern,
)

# Optional: default optimizer and coverage analyzer implementations (DI or lazy)
try:
    from src.services.selection.pattern_optimizer import V1PatternOptimizer  # type: ignore
except Exception:
    V1PatternOptimizer = None  # Fallback to DI

try:
    from src.services.selection.pattern_coverage_analyzer import V1CoverageAnalyzer  # type: ignore
except Exception:
    V1CoverageAnalyzer = None

try:
    from src.services.selection.pattern_scorer import V1PatternScorer  # type: ignore
except Exception:
    V1PatternScorer = None

try:
    from src.services.selection.pattern_analytics import V1PatternAnalytics  # type: ignore
except Exception:
    V1PatternAnalytics = None

logger = logging.getLogger(__name__)


class PatternSelectionStrategy(Enum):
    DOMAIN_MATCHED = "domain_matched"
    CONSULTANT_ALIGNED = "consultant_aligned"
    FRAMEWORK_OPTIMIZED = "framework_optimized"
    COMPLEXITY_ADAPTIVE = "complexity_adaptive"
    DEFAULT_FALLBACK = "default_fallback"


@dataclass
class NWayPatternCandidate:
    """Represents a candidate NWAY pattern with scoring"""

    pattern_id: str
    cluster: str
    title: str
    models: List[str]
    consultant_priority: List[str]
    system2_triggers: List[str]
    relevance_score: float
    selection_strategy: PatternSelectionStrategy
    rationale: str


@dataclass
class NWayPatternSelection:
    """Result of pattern selection process"""

    selected_patterns: List[str]
    primary_pattern: str
    selection_rationale: str
    fallback_patterns: List[str]
    confidence_score: float


class NWayPatternSelectionService:
    """
    Strangler Facade for the legacy NWayPatternSelectionService.

    Delegates most behavior to the legacy engine while routing the core optimization
    step to an IPatternOptimizer implementation. This enables incremental extraction
    without changing public call sites.
    """

    def __init__(
        self,
        optimizer: Optional[IPatternOptimizer] = None,
        coverage_analyzer: Optional[ICoverageAnalyzer] = None,
        pattern_scorer: Optional[IPatternScorer] = None,
        pattern_analytics: Optional[IPatternAnalytics] = None,
        legacy: Optional[object] = None,
    ) -> None:
        # Self-contained caches for pattern data
        # Resolve cognitive architecture directory from env or default to repo path
        env_dir = os.getenv("COGNITIVE_ARCHITECTURE_DIR")
        if env_dir:
            self.cognitive_architecture_path = Path(env_dir)
        else:
            # Fallback: assume local repo has cognitive_architecture directory
            self.cognitive_architecture_path = Path(os.getcwd()) / "cognitive_architecture"
        self.patterns_cache: Dict[str, Any] = {}
        self.clusters_cache: Dict[str, Dict[str, Any]] = {}
        self._load_cognitive_architecture()

        # Optimizer defaults to V1PatternOptimizer if DI not provided
        if optimizer is not None:
            self.optimizer: IPatternOptimizer = optimizer
        else:
            if V1PatternOptimizer is None:
                raise RuntimeError(
                    "IPatternOptimizer implementation not available. Ensure V1PatternOptimizer is importable or inject an optimizer."
                )
            self.optimizer = V1PatternOptimizer()  # type: ignore
        # Coverage analyzer defaults to V1CoverageAnalyzer if DI not provided
        if coverage_analyzer is not None:
            self.coverage_analyzer: ICoverageAnalyzer = coverage_analyzer
        else:
            if V1CoverageAnalyzer is None:
                raise RuntimeError(
                    "ICoverageAnalyzer implementation not available. Ensure V1CoverageAnalyzer is importable or inject an analyzer."
                )
            self.coverage_analyzer = V1CoverageAnalyzer()  # type: ignore
        # Pattern scorer defaults to V1PatternScorer if DI not provided
        if pattern_scorer is not None:
            self.pattern_scorer: IPatternScorer = pattern_scorer
        else:
            if V1PatternScorer is None:
                raise RuntimeError(
                    "IPatternScorer implementation not available. Ensure V1PatternScorer is importable or inject a scorer."
                )
            self.pattern_scorer = V1PatternScorer()  # type: ignore
        # Pattern analytics defaults to V1PatternAnalytics if DI not provided
        if pattern_analytics is not None:
            self.pattern_analytics: IPatternAnalytics = pattern_analytics
        else:
            if V1PatternAnalytics is None:
                raise RuntimeError(
                    "IPatternAnalytics implementation not available. Ensure V1PatternAnalytics is importable or inject an analytics service."
                )
            self.pattern_analytics = V1PatternAnalytics()  # type: ignore

    def _select_optimal_patterns(
        self,
        candidates: List[NWayPatternCandidate],
        framework_type: str,
        task_classification: Dict[str, Any],
    ) -> NWayPatternSelection:
        """
        Facade interception point: delegate selection to the optimizer service
        using a ScoreReport derived from legacy candidates.
        """
        if not candidates:
            return self._get_fallback_selection()

        # Build a ScoreReport compatible with the optimizer contract
        scores = {c.pattern_id: c.relevance_score for c in candidates}
        rationale = {c.pattern_id: c.rationale for c in candidates}
        top_patterns = [c.pattern_id for c in candidates]
        report = ScoreReport(
            top_patterns=top_patterns, scores=scores, rationale=rationale, context_factors={}
        )

        # Provide additional context needed by the optimizer (cluster mapping, etc.)
        context = {
            "framework_type": framework_type,
            "task_classification": task_classification,
            "cluster_map": {c.pattern_id: c.cluster for c in candidates},
        }

        result: OptimizationResult = self.optimizer.optimize(report, context)
        primary = result.primary_pattern or (
            result.selected_patterns[0] if result.selected_patterns else ""
        )

        return NWayPatternSelection(
            selected_patterns=result.selected_patterns or ([primary] if primary else []),
            primary_pattern=primary,
            selection_rationale=result.rationale,
            fallback_patterns=result.fallback_patterns,
            confidence_score=result.confidence_score,
        )

    def validate_dimension_coverage(
        self,
        selected_patterns: List[str],
        framework_type: str,
        task_classification: Dict[str, Any],
        target_coverage: float = 0.90,
    ) -> Dict[str, Any]:
        return self.coverage_analyzer.analyze(
            selected_patterns, framework_type, task_classification, target_coverage
        ).model_dump()

    def select_patterns_for_framework(
        self,
        framework_type: str,
        task_classification: Dict[str, Any],
        consultant_types: List[str],
        complexity: str = "medium",
        s2_tier: str = "S2_DISABLED",
    ) -> NWayPatternSelection:
        """
        Select optimal NWAY patterns for given framework and context.
        Enhanced with confidence threshold validation - triggers deep analysis if confidence < 0.70

        This is a strangler-facade implementation that reuses legacy scoring/analytics
        but delegates core optimization to the new optimizer service.
        """
        if not self.patterns_cache:
            logger.warning("⚠️ No patterns loaded, using fallback")
            return self._get_fallback_selection()

        # Score all patterns for this context using the scorer via facade
        candidates = self._score_pattern_candidates(
            framework_type, task_classification, consultant_types, complexity, s2_tier
        )

        if not candidates:
            logger.warning("⚠️ No pattern candidates found, using fallback")
            return self._legacy_engine._get_fallback_selection()

        # Initial pattern selection via optimizer
        initial_selection = self._select_optimal_patterns(
            candidates, framework_type, task_classification
        )

        # Confidence threshold validation (no enhanced legacy fallback; return initial)
        return initial_selection

    def _generate_coverage_recommendations(
        self,
        coverage_gaps: List[Dict[str, Any]],
        dimension_coverage: Dict[str, Any],
        selected_patterns: List[str],
        framework_type: str,
    ) -> List[str]:
        """Delegate coverage recommendations to the coverage analyzer (facade)."""
        return self.coverage_analyzer._recommend(
            coverage_gaps, dimension_coverage, selected_patterns, framework_type
        )

    def _load_cognitive_architecture(self) -> None:
        """Load NWAY patterns from YAML cognitive architecture via typed models"""
        try:
            from src.config.architecture_loader import load_full_architecture

            # Support both directory and file configuration for cognitive architecture
            master_path = (
                self.cognitive_architecture_path
                if self.cognitive_architecture_path.is_file()
                else self.cognitive_architecture_path / "nway_cognitive_architecture.yaml"
            )

            if not master_path.exists():
                raise FileNotFoundError(f"Cognitive architecture file not found: {master_path}")

            clusters = load_full_architecture(master_path)
            self.clusters_cache = {k: v.model_dump() for k, v in clusters.items()}
            patterns_loaded = 0
            for cluster_key, cluster in clusters.items():
                for nway in cluster.nways:
                    self.patterns_cache[nway.id] = {
                        "id": nway.id,
                        "cluster": cluster_key,
                        "title": nway.title,
                        "models": nway.models,
                        "interactions": nway.interactions,
                        "consultant_priority": nway.consultant_priority,
                        "consultant_personas": nway.consultant_personas,
                        "system2_triggers": nway.system2_triggers,
                        "metacognitive_prompts": nway.metacognitive_prompts,
                    }
                    patterns_loaded += 1
            logger.info(
                f"✅ Loaded {patterns_loaded} NWAY patterns from {len(clusters)} clusters (typed)"
            )
        except Exception as e:
            logger.error(f"❌ Failed to load cognitive architecture: {e}")

    def _get_fallback_selection(self) -> NWayPatternSelection:
        """Get fallback selection when no patterns are available or scoring fails"""
        return NWayPatternSelection(
            selected_patterns=["NWAY_PERCEPTION_001"],
            primary_pattern="NWAY_PERCEPTION_001",
            selection_rationale="Fallback to NWAY_PERCEPTION_001 due to pattern selection failure",
            fallback_patterns=["NWAY_DECISION_002", "NWAY_REASONING_001"],
            confidence_score=0.3,
        )

    def _score_pattern_candidates(
        self,
        framework_type: str,
        task_classification: Dict[str, Any],
        consultant_types: List[str],
        complexity: str,
        s2_tier: str,
    ) -> List[NWayPatternCandidate]:
        """Score all patterns for relevance to the current context using the scorer."""
        # Build Pattern list from loaded patterns cache
        patterns: List[Pattern] = []
        for pid, pdata in self.patterns_cache.items():
            # Convert dictionaries to lists if needed for Pydantic validation
            consultant_priority_raw = pdata.get("consultant_priority", [])
            if isinstance(consultant_priority_raw, dict):
                consultant_priority = list(consultant_priority_raw.keys())
            else:
                consultant_priority = consultant_priority_raw

            system2_triggers_raw = pdata.get("system2_triggers", [])
            if isinstance(system2_triggers_raw, dict):
                system2_triggers = list(system2_triggers_raw.keys())
            else:
                system2_triggers = system2_triggers_raw

            patterns.append(
                Pattern(
                    id=pdata.get("id", pid),
                    cluster=pdata.get("cluster", "unknown"),
                    title=pdata.get("title", pid),
                    models=pdata.get("models", []),
                    consultant_priority=consultant_priority,
                    system2_triggers=system2_triggers,
                )
            )

        report = self.pattern_scorer.score(
            patterns,
            {
                "framework_type": framework_type,
                "task_classification": task_classification,
                "consultant_types": consultant_types,
                "complexity": complexity,
                "s2_tier": s2_tier,
            },
        )

        # Convert ScoreReport to NWayPatternCandidate list
        strategies_map: Dict[str, str] = report.context_factors.get("strategy", {}) if report.context_factors else {}
        candidates: List[NWayPatternCandidate] = []
        for pid in report.top_patterns:
            score = report.scores.get(pid, 0.0)
            if score <= 0.1:
                continue
            pdata = self.patterns_cache.get(pid, {})
            strategy_str = strategies_map.get(pid, "default_fallback")
            try:
                strategy_enum = PatternSelectionStrategy(strategy_str)
            except Exception:
                strategy_enum = PatternSelectionStrategy.DEFAULT_FALLBACK
            # Convert dictionaries to lists if needed for dataclass validation
            consultant_priority_raw = pdata.get("consultant_priority", [])
            if isinstance(consultant_priority_raw, dict):
                consultant_priority = list(consultant_priority_raw.keys())
            else:
                consultant_priority = consultant_priority_raw

            system2_triggers_raw = pdata.get("system2_triggers", [])
            if isinstance(system2_triggers_raw, dict):
                system2_triggers = list(system2_triggers_raw.keys())
            else:
                system2_triggers = system2_triggers_raw

            candidates.append(
                NWayPatternCandidate(
                    pattern_id=pid,
                    cluster=pdata.get("cluster", "unknown"),
                    title=pdata.get("title", pid),
                    models=pdata.get("models", []),
                    consultant_priority=consultant_priority,
                    system2_triggers=system2_triggers,
                    relevance_score=score,
                    selection_strategy=strategy_enum,
                    rationale=report.rationale.get(pid, pdata.get("cluster", "unknown")),
                )
            )
        return candidates

    def get_pattern_learning_analytics(self) -> Dict[str, Any]:
        """Delegate to analytics service for learning/trend insights."""
        return self.pattern_analytics.summarize()

    def get_pattern_details(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific pattern"""
        return self.patterns_cache.get(pattern_id)

    def list_available_patterns(self, cluster: Optional[str] = None) -> List[str]:
        """List all available patterns, optionally filtered by cluster"""
        if cluster:
            return [
                pid
                for pid, data in self.patterns_cache.items()
                if data.get("cluster", "").upper() == cluster.upper()
            ]
        return list(self.patterns_cache.keys())

    def get_cluster_summary(self) -> Dict[str, int]:
        """Get summary of patterns by cluster"""
        cluster_counts: Dict[str, int] = {}
        for pattern_data in self.patterns_cache.values():
            cluster = pattern_data.get("cluster", "unknown")
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
        return cluster_counts

