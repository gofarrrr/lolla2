"""
Cognitive Diversity Calibrator
Prevents mental model path convergence through systematic diversity injection

This system addresses the critical problem of cognitive ruts where the METIS system
might converge on locally optimal but globally suboptimal solution patterns by:

1. Tracking model selection patterns over time
2. Detecting convergence on preferred mental models
3. Injecting controlled diversity to prevent cognitive stagnation
4. Calibrating exploration vs exploitation balance
5. Learning from diverse approach outcomes
"""

import logging
import time
import statistics
from typing import Dict, List, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import random
import numpy as np

logger = logging.getLogger(__name__)


class ConvergenceRisk(Enum):
    """Risk levels for cognitive convergence"""

    LOW = "low"  # Healthy diversity in model selection
    MODERATE = "moderate"  # Some patterns emerging but manageable
    HIGH = "high"  # Dangerous convergence detected
    CRITICAL = "critical"  # Severe cognitive rut - immediate intervention needed


class DiversityStrategy(Enum):
    """Strategies for injecting cognitive diversity"""

    RANDOM_INJECTION = "random_injection"  # Inject random model selections
    EXPLORATION_BOOST = "exploration_boost"  # Boost underused models
    NOVELTY_SEEKING = "novelty_seeking"  # Actively seek novel combinations
    ANTI_PATTERN = "anti_pattern"  # Deliberately avoid recent patterns
    CONTRARIAN_SELECTION = (
        "contrarian_selection"  # Choose opposite of typical selection
    )


@dataclass
class ModelUsagePattern:
    """Pattern analysis for mental model usage"""

    model_id: str
    usage_count: int
    usage_frequency: float  # Usage rate over time window
    last_used: datetime
    avg_confidence: float  # Average confidence when this model is used
    avg_outcome_quality: float  # Average outcome quality
    contexts_used: Set[str]  # Problem contexts where used
    synergy_partners: List[str]  # Models commonly used together
    diversity_score: float  # How diverse this model's usage contexts are


@dataclass
class ConvergenceAnalysis:
    """Analysis of cognitive convergence patterns"""

    analysis_timestamp: datetime
    time_window_days: int
    total_engagements: int
    unique_models_used: int
    model_usage_patterns: List[ModelUsagePattern]
    convergence_risk: ConvergenceRisk
    dominant_patterns: List[str]  # Most frequently used patterns
    underutilized_models: List[str]  # Models that are rarely selected
    pattern_entropy: float  # Shannon entropy of model selection
    gini_coefficient: float  # Inequality in model usage distribution
    recommended_interventions: List[str]


@dataclass
class DiversityIntervention:
    """Specific intervention to increase cognitive diversity"""

    intervention_id: str
    strategy: DiversityStrategy
    target_models: List[str]
    intervention_strength: float  # 0.0-1.0, how strongly to apply
    duration_engagements: int  # How long to apply intervention
    expected_diversity_gain: float  # Expected improvement in diversity
    success_criteria: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class CognitiveDiversityCalibrator:
    """
    Calibrates the flywheel system to prevent mental model convergence.

    Key functions:
    1. Monitors model selection patterns across engagements
    2. Detects when system is converging on preferred models/patterns
    3. Injects controlled diversity to prevent cognitive ruts
    4. Learns which diversity strategies are most effective
    5. Maintains exploration vs exploitation balance
    """

    def __init__(self, analysis_window_days: int = 30):
        self.logger = logging.getLogger(__name__)
        self.analysis_window_days = analysis_window_days

        # Usage tracking
        self.model_usage_history: deque = deque(maxlen=10000)
        self.engagement_history: deque = deque(maxlen=5000)
        self.intervention_history: List[DiversityIntervention] = []

        # Calibration parameters
        self.diversity_targets = {
            "min_pattern_entropy": 2.5,  # Minimum Shannon entropy for healthy diversity
            "max_gini_coefficient": 0.6,  # Maximum inequality in model usage
            "min_unique_models_ratio": 0.4,  # Min ratio of unique models to available models
            "exploration_rate": 0.15,  # Target rate of exploratory (non-optimal) selections
        }

        # Current state
        self.current_convergence_risk = ConvergenceRisk.LOW
        self.active_interventions: List[DiversityIntervention] = []
        self.calibration_enabled = True

        # Learning system
        self.intervention_effectiveness: Dict[DiversityStrategy, float] = defaultdict(
            float
        )
        self.strategy_success_rates: Dict[DiversityStrategy, List[float]] = defaultdict(
            list
        )

    async def analyze_convergence_patterns(
        self, recent_engagements: List[Dict[str, Any]]
    ) -> ConvergenceAnalysis:
        """
        Analyze recent engagements for cognitive convergence patterns.

        Args:
            recent_engagements: List of engagement data with model selections

        Returns:
            ConvergenceAnalysis with risk assessment and recommendations
        """
        analysis_start = time.time()

        try:
            # Filter to analysis window
            cutoff_date = datetime.now() - timedelta(days=self.analysis_window_days)
            recent_engagements = [
                eng
                for eng in recent_engagements
                if datetime.fromisoformat(eng.get("timestamp", "")) > cutoff_date
            ]

            if len(recent_engagements) < 10:
                self.logger.warning(
                    f"Insufficient data for convergence analysis: {len(recent_engagements)} engagements"
                )
                return self._create_minimal_analysis(recent_engagements)

            # Extract model usage patterns
            model_usage_patterns = await self._analyze_model_usage_patterns(
                recent_engagements
            )

            # Calculate diversity metrics
            pattern_entropy = self._calculate_pattern_entropy(model_usage_patterns)
            gini_coefficient = self._calculate_gini_coefficient(model_usage_patterns)

            # Identify dominant patterns and underutilized models
            dominant_patterns = self._identify_dominant_patterns(model_usage_patterns)
            underutilized_models = self._identify_underutilized_models(
                model_usage_patterns
            )

            # Assess convergence risk
            convergence_risk = self._assess_convergence_risk(
                pattern_entropy,
                gini_coefficient,
                len(underutilized_models),
                len(recent_engagements),
            )

            # Generate intervention recommendations
            interventions = await self._recommend_interventions(
                convergence_risk,
                dominant_patterns,
                underutilized_models,
                model_usage_patterns,
            )

            analysis = ConvergenceAnalysis(
                analysis_timestamp=datetime.now(),
                time_window_days=self.analysis_window_days,
                total_engagements=len(recent_engagements),
                unique_models_used=len(model_usage_patterns),
                model_usage_patterns=model_usage_patterns,
                convergence_risk=convergence_risk,
                dominant_patterns=dominant_patterns,
                underutilized_models=underutilized_models,
                pattern_entropy=pattern_entropy,
                gini_coefficient=gini_coefficient,
                recommended_interventions=interventions,
            )

            analysis_time = time.time() - analysis_start
            self.logger.info(f"Convergence analysis completed in {analysis_time:.2f}s")
            self.logger.info(
                f"Risk: {convergence_risk.value}, Entropy: {pattern_entropy:.2f}, Gini: {gini_coefficient:.2f}"
            )

            return analysis

        except Exception as e:
            self.logger.error(f"Convergence analysis failed: {e}")
            return self._create_minimal_analysis(recent_engagements)

    async def _analyze_model_usage_patterns(
        self, engagements: List[Dict[str, Any]]
    ) -> List[ModelUsagePattern]:
        """Analyze usage patterns for each mental model"""

        model_stats = defaultdict(
            lambda: {
                "count": 0,
                "confidences": [],
                "outcomes": [],
                "contexts": set(),
                "timestamps": [],
                "synergy_partners": [],
            }
        )

        # Collect usage statistics
        for engagement in engagements:
            models_used = engagement.get("selected_models", [])
            confidence = engagement.get("confidence_score", 0.5)
            outcome_quality = engagement.get("outcome_quality", 0.5)
            context = engagement.get("problem_type", "unknown")
            timestamp = datetime.fromisoformat(
                engagement.get("timestamp", datetime.now().isoformat())
            )

            for model in models_used:
                model_stats[model]["count"] += 1
                model_stats[model]["confidences"].append(confidence)
                model_stats[model]["outcomes"].append(outcome_quality)
                model_stats[model]["contexts"].add(context)
                model_stats[model]["timestamps"].append(timestamp)

                # Track synergy partners (other models used in same engagement)
                partners = [m for m in models_used if m != model]
                model_stats[model]["synergy_partners"].extend(partners)

        # Calculate patterns
        total_engagements = len(engagements)
        patterns = []

        for model_id, stats in model_stats.items():
            if stats["count"] == 0:
                continue

            # Calculate diversity score based on context variety
            context_diversity = len(stats["contexts"]) / max(len(stats["contexts"]), 1)

            pattern = ModelUsagePattern(
                model_id=model_id,
                usage_count=stats["count"],
                usage_frequency=stats["count"] / total_engagements,
                last_used=(
                    max(stats["timestamps"]) if stats["timestamps"] else datetime.now()
                ),
                avg_confidence=(
                    statistics.mean(stats["confidences"])
                    if stats["confidences"]
                    else 0.5
                ),
                avg_outcome_quality=(
                    statistics.mean(stats["outcomes"]) if stats["outcomes"] else 0.5
                ),
                contexts_used=stats["contexts"],
                synergy_partners=list(set(stats["synergy_partners"])),
                diversity_score=context_diversity,
            )

            patterns.append(pattern)

        # Sort by usage frequency (most used first)
        patterns.sort(key=lambda p: p.usage_frequency, reverse=True)

        return patterns

    def _calculate_pattern_entropy(self, patterns: List[ModelUsagePattern]) -> float:
        """Calculate Shannon entropy of model usage patterns"""

        if not patterns:
            return 0.0

        total_usage = sum(p.usage_count for p in patterns)
        if total_usage == 0:
            return 0.0

        entropy = 0.0
        for pattern in patterns:
            if pattern.usage_count > 0:
                p = pattern.usage_count / total_usage
                entropy -= p * np.log2(p)

        return entropy

    def _calculate_gini_coefficient(self, patterns: List[ModelUsagePattern]) -> float:
        """Calculate Gini coefficient for model usage inequality"""

        if not patterns:
            return 0.0

        usage_counts = [p.usage_count for p in patterns]
        usage_counts.sort()

        n = len(usage_counts)
        cumsum = np.cumsum(usage_counts)

        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0.0

    def _identify_dominant_patterns(
        self, patterns: List[ModelUsagePattern]
    ) -> List[str]:
        """Identify dominant usage patterns that might indicate convergence"""

        if not patterns:
            return []

        total_usage = sum(p.usage_count for p in patterns)
        dominant_threshold = 0.3  # Models used in >30% of engagements

        dominant = []
        for pattern in patterns:
            if pattern.usage_frequency > dominant_threshold:
                dominant.append(f"{pattern.model_id} ({pattern.usage_frequency:.1%})")

        return dominant

    def _identify_underutilized_models(
        self, patterns: List[ModelUsagePattern]
    ) -> List[str]:
        """Identify models that are rarely or never used"""

        if not patterns:
            return []

        underutilized_threshold = 0.05  # Models used in <5% of engagements

        underutilized = []
        for pattern in patterns:
            if pattern.usage_frequency < underutilized_threshold:
                underutilized.append(pattern.model_id)

        # Also include models that haven't been used recently
        cutoff_date = datetime.now() - timedelta(days=7)
        for pattern in patterns:
            if (
                pattern.last_used < cutoff_date
                and pattern.model_id not in underutilized
            ):
                underutilized.append(pattern.model_id)

        return underutilized

    def _assess_convergence_risk(
        self,
        entropy: float,
        gini: float,
        underutilized_count: int,
        total_engagements: int,
    ) -> ConvergenceRisk:
        """Assess the risk level of cognitive convergence"""

        risk_factors = []

        # Entropy too low (lack of diversity)
        if entropy < self.diversity_targets["min_pattern_entropy"]:
            risk_factors.append("low_entropy")

        # Gini coefficient too high (high inequality)
        if gini > self.diversity_targets["max_gini_coefficient"]:
            risk_factors.append("high_inequality")

        # Too many underutilized models
        underutilized_ratio = underutilized_count / max(total_engagements / 10, 1)
        if underutilized_ratio > 0.5:
            risk_factors.append("many_underutilized")

        # Assess overall risk
        if len(risk_factors) >= 3:
            return ConvergenceRisk.CRITICAL
        elif len(risk_factors) >= 2:
            return ConvergenceRisk.HIGH
        elif len(risk_factors) >= 1:
            return ConvergenceRisk.MODERATE
        else:
            return ConvergenceRisk.LOW

    async def _recommend_interventions(
        self,
        risk: ConvergenceRisk,
        dominant_patterns: List[str],
        underutilized_models: List[str],
        usage_patterns: List[ModelUsagePattern],
    ) -> List[str]:
        """Recommend specific interventions based on convergence analysis"""

        interventions = []

        if risk == ConvergenceRisk.CRITICAL:
            interventions.extend(
                [
                    "Implement immediate random injection (20% rate)",
                    "Boost underutilized models with artificial confidence bonus",
                    "Apply anti-pattern selection for next 10 engagements",
                    "Temporarily disable top 2 most-used models",
                ]
            )

        elif risk == ConvergenceRisk.HIGH:
            interventions.extend(
                [
                    "Increase exploration rate to 25%",
                    "Inject novelty-seeking behavior",
                    "Boost underutilized models",
                    "Monitor for pattern disruption",
                ]
            )

        elif risk == ConvergenceRisk.MODERATE:
            interventions.extend(
                [
                    "Gentle diversity nudging",
                    "Occasional random selection",
                    "Promote model rotation",
                ]
            )

        else:  # LOW risk
            interventions.append("Continue monitoring - diversity levels healthy")

        # Add specific recommendations based on patterns
        if len(dominant_patterns) > 2:
            interventions.append(
                f"Address dominant patterns: {', '.join(dominant_patterns[:2])}"
            )

        if len(underutilized_models) > 5:
            interventions.append(
                f"Revive underutilized models: {', '.join(underutilized_models[:3])}"
            )

        return interventions

    async def apply_diversity_intervention(
        self,
        intervention: DiversityIntervention,
        current_selection: List[str],
        available_models: List[str],
        problem_context: Dict[str, Any],
    ) -> Tuple[List[str], float]:
        """
        Apply a diversity intervention to modify model selection.

        Args:
            intervention: The intervention to apply
            current_selection: Currently selected models
            available_models: All available models
            problem_context: Context about current problem

        Returns:
            Tuple of (modified_selection, diversity_boost_applied)
        """

        if not self.calibration_enabled:
            return current_selection, 0.0

        try:
            original_selection = current_selection.copy()
            diversity_boost = 0.0

            if intervention.strategy == DiversityStrategy.RANDOM_INJECTION:
                # Replace some selections with random choices
                num_to_replace = max(
                    1, int(len(current_selection) * intervention.intervention_strength)
                )

                for _ in range(num_to_replace):
                    if current_selection:
                        # Remove most confident choice and add random alternative
                        current_selection.pop(0)
                        random_model = random.choice(available_models)
                        if random_model not in current_selection:
                            current_selection.append(random_model)

                diversity_boost = num_to_replace / len(original_selection)

            elif intervention.strategy == DiversityStrategy.EXPLORATION_BOOST:
                # Boost underutilized models in selection
                for target_model in intervention.target_models:
                    if (
                        target_model in available_models
                        and target_model not in current_selection
                    ):
                        # Add underutilized model with artificial boost
                        if len(current_selection) < 5:  # Limit selection size
                            current_selection.append(target_model)
                            diversity_boost += 0.2

            elif intervention.strategy == DiversityStrategy.NOVELTY_SEEKING:
                # Seek novel model combinations
                used_combinations = self._get_recent_combinations()

                for model in available_models:
                    if model not in current_selection:
                        test_combo = sorted(current_selection + [model])
                        combo_key = "|".join(test_combo)

                        if combo_key not in used_combinations:
                            current_selection.append(model)
                            diversity_boost += 0.3
                            break

            elif intervention.strategy == DiversityStrategy.ANTI_PATTERN:
                # Avoid recently used patterns
                recent_patterns = self._get_recent_patterns()
                current_pattern = "|".join(sorted(current_selection))

                if current_pattern in recent_patterns:
                    # Modify selection to avoid pattern repetition
                    alternatives = [
                        m for m in available_models if m not in current_selection
                    ]
                    if alternatives:
                        current_selection[-1] = random.choice(alternatives)
                        diversity_boost = 0.4

            elif intervention.strategy == DiversityStrategy.CONTRARIAN_SELECTION:
                # Choose opposite of typical high-confidence selections
                contrarian_models = self._identify_contrarian_models(available_models)

                if contrarian_models:
                    # Replace highest confidence choice with contrarian choice
                    if current_selection:
                        current_selection[0] = random.choice(contrarian_models)
                        diversity_boost = 0.5

            # Log intervention application
            self.logger.info(
                f"Applied {intervention.strategy.value}: diversity boost {diversity_boost:.2f}"
            )
            if original_selection != current_selection:
                self.logger.info(
                    f"Selection changed: {original_selection} -> {current_selection}"
                )

            return current_selection, diversity_boost

        except Exception as e:
            self.logger.error(
                f"Failed to apply intervention {intervention.strategy.value}: {e}"
            )
            return current_selection, 0.0

    def _get_recent_combinations(self) -> Set[str]:
        """Get recently used model combinations"""
        recent_combos = set()
        cutoff_date = datetime.now() - timedelta(days=3)

        for engagement in self.engagement_history:
            if engagement.get("timestamp", datetime.min) > cutoff_date:
                models = engagement.get("selected_models", [])
                if models:
                    combo_key = "|".join(sorted(models))
                    recent_combos.add(combo_key)

        return recent_combos

    def _get_recent_patterns(self) -> Set[str]:
        """Get recently used selection patterns"""
        return self._get_recent_combinations()  # Same implementation for now

    def _identify_contrarian_models(self, available_models: List[str]) -> List[str]:
        """Identify models that would be contrarian (opposite) choices"""

        # Simple heuristic: models that are least frequently used recently
        model_frequencies = defaultdict(int)
        cutoff_date = datetime.now() - timedelta(days=7)

        for engagement in self.engagement_history:
            if engagement.get("timestamp", datetime.min) > cutoff_date:
                for model in engagement.get("selected_models", []):
                    model_frequencies[model] += 1

        # Sort by frequency (ascending) to get least used
        sorted_models = sorted(available_models, key=lambda m: model_frequencies[m])

        # Return bottom quartile as contrarian choices
        return sorted_models[: max(1, len(sorted_models) // 4)]

    def _create_minimal_analysis(
        self, engagements: List[Dict[str, Any]]
    ) -> ConvergenceAnalysis:
        """Create minimal analysis when insufficient data"""

        return ConvergenceAnalysis(
            analysis_timestamp=datetime.now(),
            time_window_days=self.analysis_window_days,
            total_engagements=len(engagements),
            unique_models_used=0,
            model_usage_patterns=[],
            convergence_risk=ConvergenceRisk.LOW,
            dominant_patterns=[],
            underutilized_models=[],
            pattern_entropy=0.0,
            gini_coefficient=0.0,
            recommended_interventions=["Collect more data for analysis"],
        )

    async def update_engagement_history(self, engagement_data: Dict[str, Any]):
        """Update engagement history for convergence tracking"""

        engagement_data["timestamp"] = engagement_data.get("timestamp", datetime.now())
        self.engagement_history.append(engagement_data)

        # Update model usage history
        for model in engagement_data.get("selected_models", []):
            usage_record = {
                "model_id": model,
                "timestamp": engagement_data["timestamp"],
                "confidence": engagement_data.get("confidence_score", 0.5),
                "outcome_quality": engagement_data.get("outcome_quality", 0.5),
                "problem_type": engagement_data.get("problem_type", "unknown"),
            }
            self.model_usage_history.append(usage_record)

    def get_calibration_status(self) -> Dict[str, Any]:
        """Get current calibration system status"""

        return {
            "calibration_enabled": self.calibration_enabled,
            "current_convergence_risk": self.current_convergence_risk.value,
            "active_interventions": len(self.active_interventions),
            "engagement_history_size": len(self.engagement_history),
            "model_usage_history_size": len(self.model_usage_history),
            "diversity_targets": self.diversity_targets,
            "intervention_effectiveness": dict(self.intervention_effectiveness),
            "analysis_window_days": self.analysis_window_days,
        }


# Singleton pattern for global access
_diversity_calibrator = None


def get_cognitive_diversity_calibrator() -> CognitiveDiversityCalibrator:
    """Get singleton cognitive diversity calibrator instance"""
    global _diversity_calibrator
    if _diversity_calibrator is None:
        _diversity_calibrator = CognitiveDiversityCalibrator()
    return _diversity_calibrator
