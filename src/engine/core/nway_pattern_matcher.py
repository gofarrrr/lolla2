#!/usr/bin/env python3
"""
N-Way Pattern Matcher for METIS Cognitive Platform
Advanced pattern matching system for breakthrough insights through mental model combinations

Leverages the loaded N-Way interaction patterns (223 interactions from cognitive engine)
to identify complex multi-model synergies and breakthrough insight opportunities.

Key Capabilities:
1. Multi-model pattern recognition across 128+ mental models
2. Synergistic and conflicting pattern detection
3. Breakthrough opportunity identification
4. Pattern strength scoring and validation
5. Context-aware pattern recommendation
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import statistics


class PatternMatchType(str, Enum):
    """Types of N-Way pattern matches"""

    SYNERGISTIC = "synergistic"  # Models that amplify each other
    CONFLICTING = "conflicting"  # Models that create tension/analysis
    REINFORCING = "reinforcing"  # Models that validate each other
    COMPLEMENTARY = "complementary"  # Models that fill gaps
    BREAKTHROUGH = "breakthrough"  # Rare high-value combinations


class PatternStrength(str, Enum):
    """Pattern matching strength levels"""

    WEAK = "weak"  # 0.3-0.5: Limited evidence
    MODERATE = "moderate"  # 0.5-0.7: Good evidence
    STRONG = "strong"  # 0.7-0.85: Strong evidence
    EXCEPTIONAL = "exceptional"  # 0.85+: Overwhelming evidence


class BreakthroughPotential(str, Enum):
    """Breakthrough insight potential levels"""

    LOW = "low"  # Standard analysis expected
    MEDIUM = "medium"  # Above-average insights possible
    HIGH = "high"  # Breakthrough insights likely
    EXCEPTIONAL = "exceptional"  # Revolutionary insights possible


@dataclass
class NWayPattern:
    """Individual N-Way pattern with metadata"""

    pattern_id: str
    models: List[str]
    match_type: PatternMatchType
    strength: float
    description: str
    synergy_multiplier: float = 1.0
    breakthrough_potential: BreakthroughPotential = BreakthroughPotential.LOW
    context_keywords: List[str] = field(default_factory=list)
    success_rate: float = 0.0
    usage_count: int = 0
    last_used: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatternMatch:
    """Result of pattern matching analysis"""

    matched_patterns: List[NWayPattern]
    overall_strength: float
    breakthrough_score: float
    recommended_combinations: List[Tuple[List[str], float]]
    synergy_opportunities: List[str]
    conflict_warnings: List[str]
    pattern_insights: List[str]
    confidence_level: float
    matching_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatternStats:
    """Statistics about pattern matching performance"""

    total_patterns_loaded: int = 0
    total_matches_performed: int = 0
    successful_matches: int = 0
    breakthrough_patterns_found: int = 0
    average_match_strength: float = 0.0
    most_successful_patterns: List[str] = field(default_factory=list)
    pattern_usage_frequency: Dict[str, int] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.utcnow)


class NWayPatternMatcher:
    """
    Advanced N-Way Pattern Matcher for breakthrough cognitive insights

    Analyzes combinations of mental models to identify high-value synergistic
    patterns that can produce breakthrough insights beyond individual model
    capabilities. Uses the existing 223 interaction patterns loaded in
    cognitive engine for sophisticated pattern recognition.

    Key Features:
    1. Multi-dimensional pattern matching (2-5 models per pattern)
    2. Context-aware pattern selection based on problem characteristics
    3. Breakthrough potential scoring using historical success data
    4. Real-time pattern strength validation
    5. Synergy amplification detection for lollapalooza effects
    """

    def __init__(
        self,
        min_pattern_strength: float = 0.5,
        breakthrough_threshold: float = 0.8,
        max_patterns_per_match: int = 10,
        pattern_cache_hours: int = 24,
    ):

        self.min_pattern_strength = min_pattern_strength
        self.breakthrough_threshold = breakthrough_threshold
        self.max_patterns_per_match = max_patterns_per_match
        self.pattern_cache_hours = pattern_cache_hours

        # Pattern storage
        self.loaded_patterns: Dict[str, NWayPattern] = {}
        self.pattern_combinations: Dict[str, List[NWayPattern]] = defaultdict(list)
        self.breakthrough_patterns: List[NWayPattern] = []

        # Performance tracking
        self.stats = PatternStats()
        self.pattern_history: List[PatternMatch] = []
        self.success_metrics: Dict[str, float] = defaultdict(float)

        # Pattern matching engines
        self.synergy_detector = self._init_synergy_detector()
        self.conflict_analyzer = self._init_conflict_analyzer()
        self.breakthrough_predictor = self._init_breakthrough_predictor()

        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            "✅ N-Way Pattern Matcher initialized for breakthrough insights"
        )

    def _init_synergy_detector(self) -> Dict[str, Any]:
        """Initialize synergy detection algorithms"""
        return {
            "model_alignment_weights": {
                "conceptual_overlap": 0.3,
                "methodological_synergy": 0.25,
                "evidence_reinforcement": 0.2,
                "insight_amplification": 0.15,
                "analytical_depth": 0.1,
            },
            "synergy_thresholds": {
                "weak": 0.4,
                "moderate": 0.6,
                "strong": 0.8,
                "exceptional": 0.9,
            },
        }

    def _init_conflict_analyzer(self) -> Dict[str, Any]:
        """Initialize conflict analysis algorithms"""
        return {
            "conflict_indicators": [
                "contradictory_assumptions",
                "incompatible_methodologies",
                "opposing_perspectives",
                "resource_competition",
                "timeline_conflicts",
            ],
            "resolution_strategies": {
                "synthesis": "Combine opposing views into higher-order insight",
                "temporal_sequencing": "Apply models in logical sequence",
                "contextual_selection": "Choose model based on context",
                "dialectical_analysis": "Use tension to generate new insights",
            },
        }

    def _init_breakthrough_predictor(self) -> Dict[str, Any]:
        """Initialize breakthrough prediction algorithms"""
        return {
            "breakthrough_indicators": [
                "novel_model_combinations",
                "high_synergy_multipliers",
                "cross_domain_connections",
                "paradigm_bridging_potential",
                "complexity_emergence",
            ],
            "success_factors": {
                "model_diversity": 0.25,
                "synergy_strength": 0.25,
                "novelty_score": 0.2,
                "context_relevance": 0.15,
                "historical_performance": 0.15,
            },
        }

    async def load_patterns_from_cognitive_engine(self, cognitive_engine: Any) -> int:
        """Load N-Way patterns from the cognitive engine's database"""
        try:
            # Check if cognitive engine has database connection
            if (
                not hasattr(cognitive_engine, "db_client")
                or not cognitive_engine.db_client
            ):
                self.logger.warning(
                    "⚠️ Cognitive engine doesn't have database connection"
                )
                return 0

            # Query N-Way patterns from database
            response = (
                cognitive_engine.db_client.table("nway_interactions")
                .select("*")
                .execute()
            )

            if not response.data:
                self.logger.warning("⚠️ No N-Way patterns found in database")
                return 0

            patterns_loaded = 0
            for pattern_data in response.data:
                try:
                    # Determine pattern type
                    interaction_type = pattern_data.get(
                        "interaction_type", "synergistic"
                    )
                    match_type_map = {
                        "synergistic": PatternMatchType.SYNERGISTIC,
                        "conflicting": PatternMatchType.CONFLICTING,
                        "amplifying": PatternMatchType.REINFORCING,
                        "reinforcing": PatternMatchType.REINFORCING,
                        "dampening": PatternMatchType.CONFLICTING,
                        "lollapalooza": PatternMatchType.BREAKTHROUGH,
                        "complementary": PatternMatchType.COMPLEMENTARY,
                    }
                    match_type = match_type_map.get(
                        interaction_type, PatternMatchType.SYNERGISTIC
                    )

                    # Determine breakthrough potential
                    lollapalooza_potential = float(
                        pattern_data.get("lollapalooza_potential", 0.0)
                    )
                    if lollapalooza_potential >= 0.8:
                        breakthrough_potential = BreakthroughPotential.EXCEPTIONAL
                    elif lollapalooza_potential >= 0.6:
                        breakthrough_potential = BreakthroughPotential.HIGH
                    elif lollapalooza_potential >= 0.4:
                        breakthrough_potential = BreakthroughPotential.MEDIUM
                    else:
                        breakthrough_potential = BreakthroughPotential.LOW

                    # Create NWayPattern object
                    pattern = NWayPattern(
                        pattern_id=pattern_data.get(
                            "interaction_id", f"pattern_{patterns_loaded}"
                        ),
                        models=pattern_data.get("models_involved", []),
                        match_type=match_type,
                        strength=float(pattern_data.get("strength_score", 0.5)),
                        description=pattern_data.get("mechanism_description", ""),
                        synergy_multiplier=1.0
                        + lollapalooza_potential,  # Convert to multiplier
                        breakthrough_potential=breakthrough_potential,
                        context_keywords=pattern_data.get("relevant_contexts", []),
                        success_rate=float(pattern_data.get("success_rate", 0.0)),
                        usage_count=int(pattern_data.get("usage_count", 0)),
                        metadata={
                            "emergent_effect": pattern_data.get(
                                "emergent_effect_summary", ""
                            ),
                            "synergy_description": pattern_data.get(
                                "synergy_description", ""
                            ),
                            "instructional_cue": pattern_data.get(
                                "instructional_cue_apce", ""
                            ),
                            "enhancement_type": pattern_data.get(
                                "enhancement_type", ""
                            ),
                        },
                    )

                    # Store pattern
                    self.loaded_patterns[pattern.pattern_id] = pattern

                    # Index by model combinations for fast lookup
                    for model in pattern.models:
                        self.pattern_combinations[model].append(pattern)

                    # Track breakthrough patterns
                    if pattern.breakthrough_potential in [
                        BreakthroughPotential.HIGH,
                        BreakthroughPotential.EXCEPTIONAL,
                    ]:
                        self.breakthrough_patterns.append(pattern)

                    patterns_loaded += 1

                except Exception as e:
                    self.logger.warning(
                        f"⚠️ Failed to load pattern {pattern_data.get('interaction_id', 'unknown')}: {e}"
                    )
                    continue

            # Update statistics
            self.stats.total_patterns_loaded = patterns_loaded
            self.stats.last_updated = datetime.utcnow()

            self.logger.info(
                f"✅ Loaded {patterns_loaded} N-Way patterns from database"
            )
            self.logger.info(
                f"   Breakthrough patterns: {len(self.breakthrough_patterns)}"
            )
            self.logger.info(
                f"   Pattern combinations indexed: {len(self.pattern_combinations)} models"
            )

            return patterns_loaded

        except Exception as e:
            self.logger.error(f"❌ Failed to load patterns from cognitive engine: {e}")
            return 0

    def _assess_breakthrough_potential(self, strength: float) -> BreakthroughPotential:
        """Assess breakthrough potential based on pattern strength"""
        if strength >= 0.9:
            return BreakthroughPotential.EXCEPTIONAL
        elif strength >= 0.8:
            return BreakthroughPotential.HIGH
        elif strength >= 0.6:
            return BreakthroughPotential.MEDIUM
        else:
            return BreakthroughPotential.LOW

    def _extract_keywords(self, description: str) -> List[str]:
        """Extract keywords from pattern description for context matching"""
        # Simple keyword extraction - could be enhanced with NLP
        keywords = []
        key_terms = [
            "strategic",
            "analytical",
            "creative",
            "systematic",
            "holistic",
            "quantitative",
            "qualitative",
            "framework",
            "model",
            "approach",
            "thinking",
            "analysis",
            "decision",
            "problem",
            "solution",
            "insight",
            "perspective",
            "evaluation",
            "assessment",
            "validation",
        ]

        description_lower = description.lower()
        for term in key_terms:
            if term in description_lower:
                keywords.append(term)

        return keywords

    async def find_breakthrough_patterns(
        self,
        applied_models: List[str],
        problem_context: Dict[str, Any],
        current_analysis: Dict[str, Any],
    ) -> PatternMatch:
        """
        Find breakthrough N-Way patterns for the given models and context

        Args:
            applied_models: List of mental models being applied
            problem_context: Context about the problem being analyzed
            current_analysis: Current analysis results for pattern matching

        Returns:
            PatternMatch with recommended patterns and breakthrough opportunities
        """
        start_time = datetime.utcnow()

        try:
            # Find relevant patterns for applied models
            candidate_patterns = await self._find_candidate_patterns(
                applied_models, problem_context
            )

            # Score patterns based on context and current analysis
            scored_patterns = await self._score_patterns(
                candidate_patterns, problem_context, current_analysis
            )

            # Select best patterns
            selected_patterns = await self._select_optimal_patterns(scored_patterns)

            # Generate breakthrough insights
            breakthrough_score = await self._calculate_breakthrough_score(
                selected_patterns, current_analysis
            )

            # Create recommendations
            recommendations = await self._generate_recommendations(
                selected_patterns, problem_context
            )

            # Create pattern match result
            pattern_match = PatternMatch(
                matched_patterns=selected_patterns,
                overall_strength=(
                    statistics.mean([p.strength for p in selected_patterns])
                    if selected_patterns
                    else 0.0
                ),
                breakthrough_score=breakthrough_score,
                recommended_combinations=recommendations["combinations"],
                synergy_opportunities=recommendations["synergies"],
                conflict_warnings=recommendations["conflicts"],
                pattern_insights=recommendations["insights"],
                confidence_level=self._calculate_confidence_level(
                    selected_patterns, problem_context
                ),
                matching_metadata={
                    "processing_time_ms": (
                        datetime.utcnow() - start_time
                    ).total_seconds()
                    * 1000,
                    "patterns_evaluated": len(candidate_patterns),
                    "models_analyzed": len(applied_models),
                    "context_factors": len(problem_context),
                },
            )

            # Update statistics
            await self._update_statistics(pattern_match)

            self.logger.info("🎯 N-Way pattern matching completed:")
            self.logger.info(f"   Patterns matched: {len(selected_patterns)}")
            self.logger.info(f"   Breakthrough score: {breakthrough_score:.3f}")
            self.logger.info(
                f"   Overall strength: {pattern_match.overall_strength:.3f}"
            )
            self.logger.info(
                f"   Confidence level: {pattern_match.confidence_level:.3f}"
            )

            return pattern_match

        except Exception as e:
            self.logger.error(f"❌ N-Way pattern matching failed: {e}")
            import traceback

            traceback.print_exc()

            # Return empty pattern match
            return PatternMatch(
                matched_patterns=[],
                overall_strength=0.0,
                breakthrough_score=0.0,
                recommended_combinations=[],
                synergy_opportunities=[],
                conflict_warnings=[],
                pattern_insights=[],
                confidence_level=0.0,
            )

    async def _find_candidate_patterns(
        self, applied_models: List[str], context: Dict[str, Any]
    ) -> List[NWayPattern]:
        """Find patterns that match the applied models"""
        candidates = []

        # Direct model matches
        for model in applied_models:
            if model in self.pattern_combinations:
                candidates.extend(self.pattern_combinations[model])

        # Remove duplicates
        candidates = list({p.pattern_id: p for p in candidates}.values())

        # Filter by minimum strength
        candidates = [p for p in candidates if p.strength >= self.min_pattern_strength]

        return candidates

    async def _score_patterns(
        self,
        patterns: List[NWayPattern],
        context: Dict[str, Any],
        analysis: Dict[str, Any],
    ) -> List[Tuple[NWayPattern, float]]:
        """Score patterns based on context relevance and analysis fit"""
        scored_patterns = []

        for pattern in patterns:
            score = pattern.strength

            # Context relevance bonus
            context_relevance = self._calculate_context_relevance(pattern, context)
            score += context_relevance * 0.2

            # Analysis fit bonus
            analysis_fit = self._calculate_analysis_fit(pattern, analysis)
            score += analysis_fit * 0.15

            # Historical performance bonus
            if pattern.usage_count > 0:
                performance_bonus = pattern.success_rate * 0.1
                score += performance_bonus

            # Breakthrough potential bonus
            breakthrough_bonus = {
                BreakthroughPotential.LOW: 0.0,
                BreakthroughPotential.MEDIUM: 0.05,
                BreakthroughPotential.HIGH: 0.1,
                BreakthroughPotential.EXCEPTIONAL: 0.15,
            }
            score += breakthrough_bonus[pattern.breakthrough_potential]

            scored_patterns.append((pattern, min(1.0, score)))

        # Sort by score
        scored_patterns.sort(key=lambda x: x[1], reverse=True)

        return scored_patterns

    def _calculate_context_relevance(
        self, pattern: NWayPattern, context: Dict[str, Any]
    ) -> float:
        """Calculate how relevant a pattern is to the current context"""
        relevance = 0.0

        # Problem type matching
        problem_type = context.get("problem_type", "").lower()
        if any(keyword in problem_type for keyword in pattern.context_keywords):
            relevance += 0.3

        # Complexity matching
        problem_complexity = context.get("complexity", "medium")
        if (
            pattern.breakthrough_potential == BreakthroughPotential.HIGH
            and problem_complexity in ["high", "very_high"]
        ):
            relevance += 0.2

        # Domain matching
        domain = context.get("domain", "").lower()
        if any(keyword in domain for keyword in pattern.context_keywords):
            relevance += 0.2

        return min(1.0, relevance)

    def _calculate_analysis_fit(
        self, pattern: NWayPattern, analysis: Dict[str, Any]
    ) -> float:
        """Calculate how well a pattern fits with current analysis"""
        fit = 0.0

        # Confidence alignment
        confidence_scores = analysis.get("confidence_scores", [])
        if confidence_scores:
            avg_confidence = statistics.mean(confidence_scores)
            if (
                avg_confidence > 0.8
                and pattern.match_type == PatternMatchType.SYNERGISTIC
            ):
                fit += 0.3
            elif (
                avg_confidence < 0.6
                and pattern.match_type == PatternMatchType.CONFLICTING
            ):
                fit += 0.2  # Conflicts can help when confidence is low

        # Evidence strength
        evidence_strength = analysis.get("evidence_strength", 0.5)
        if evidence_strength > 0.8 and pattern.synergy_multiplier > 1.1:
            fit += 0.2

        return min(1.0, fit)

    async def _select_optimal_patterns(
        self, scored_patterns: List[Tuple[NWayPattern, float]]
    ) -> List[NWayPattern]:
        """Select the optimal set of patterns avoiding conflicts"""
        if not scored_patterns:
            return []

        selected = []
        used_models = set()

        for pattern, score in scored_patterns[
            : self.max_patterns_per_match * 2
        ]:  # Consider more than max
            # Avoid model overlap for synergistic patterns
            pattern_models = set(pattern.models)
            if pattern.match_type == PatternMatchType.SYNERGISTIC:
                if pattern_models.intersection(used_models):
                    continue  # Skip overlapping synergistic patterns

            selected.append(pattern)
            used_models.update(pattern_models)

            if len(selected) >= self.max_patterns_per_match:
                break

        return selected

    async def _calculate_breakthrough_score(
        self, patterns: List[NWayPattern], analysis: Dict[str, Any]
    ) -> float:
        """Calculate overall breakthrough potential score"""
        if not patterns:
            return 0.0

        # Base score from pattern breakthrough potentials
        potential_scores = {
            BreakthroughPotential.LOW: 0.2,
            BreakthroughPotential.MEDIUM: 0.5,
            BreakthroughPotential.HIGH: 0.8,
            BreakthroughPotential.EXCEPTIONAL: 1.0,
        }

        base_score = statistics.mean(
            [potential_scores[p.breakthrough_potential] for p in patterns]
        )

        # Synergy multiplier bonus
        synergy_multipliers = [
            p.synergy_multiplier
            for p in patterns
            if p.match_type == PatternMatchType.SYNERGISTIC
        ]
        if synergy_multipliers:
            synergy_bonus = (statistics.mean(synergy_multipliers) - 1.0) * 0.5
            base_score += synergy_bonus

        # Pattern diversity bonus
        model_count = len(set().union(*[p.models for p in patterns]))
        diversity_bonus = min(0.2, (model_count - 2) * 0.05)
        base_score += diversity_bonus

        return min(1.0, base_score)

    async def _generate_recommendations(
        self, patterns: List[NWayPattern], context: Dict[str, Any]
    ) -> Dict[str, List]:
        """Generate actionable recommendations from selected patterns"""
        recommendations = {
            "combinations": [],
            "synergies": [],
            "conflicts": [],
            "insights": [],
        }

        # Model combinations
        for pattern in patterns:
            if pattern.match_type == PatternMatchType.SYNERGISTIC:
                combo = (pattern.models, pattern.synergy_multiplier)
                recommendations["combinations"].append(combo)

                synergy_desc = f"Combine {' + '.join(pattern.models)} for {pattern.synergy_multiplier:.1f}x amplification"
                recommendations["synergies"].append(synergy_desc)

            elif pattern.match_type == PatternMatchType.CONFLICTING:
                conflict_desc = f"Tension between {' vs '.join(pattern.models)}: {pattern.description}"
                recommendations["conflicts"].append(conflict_desc)

        # Pattern insights
        breakthrough_patterns = [
            p
            for p in patterns
            if p.breakthrough_potential
            in [BreakthroughPotential.HIGH, BreakthroughPotential.EXCEPTIONAL]
        ]
        for pattern in breakthrough_patterns:
            insight = f"🚀 Breakthrough opportunity: {pattern.description}"
            recommendations["insights"].append(insight)

        return recommendations

    def _calculate_confidence_level(
        self, patterns: List[NWayPattern], context: Dict[str, Any]
    ) -> float:
        """Calculate confidence in pattern matching results"""
        if not patterns:
            return 0.0

        # Base confidence from pattern strengths
        base_confidence = statistics.mean([p.strength for p in patterns])

        # Historical performance boost
        patterns_with_history = [p for p in patterns if p.usage_count > 0]
        if patterns_with_history:
            avg_success_rate = statistics.mean(
                [p.success_rate for p in patterns_with_history]
            )
            base_confidence = (base_confidence * 0.7) + (avg_success_rate * 0.3)

        # Context clarity boost
        context_factors = len(
            [v for v in context.values() if v is not None and v != ""]
        )
        context_clarity = min(
            1.0, context_factors / 5
        )  # Assume 5 ideal context factors
        base_confidence += context_clarity * 0.1

        return min(1.0, base_confidence)

    async def _update_statistics(self, pattern_match: PatternMatch):
        """Update pattern matching statistics"""
        self.stats.total_matches_performed += 1

        if pattern_match.overall_strength > self.min_pattern_strength:
            self.stats.successful_matches += 1

        if pattern_match.breakthrough_score > self.breakthrough_threshold:
            self.stats.breakthrough_patterns_found += 1

        # Update average
        if self.stats.total_matches_performed > 0:
            all_strengths = [
                pm.overall_strength for pm in self.pattern_history[-50:]
            ]  # Last 50
            all_strengths.append(pattern_match.overall_strength)
            self.stats.average_match_strength = statistics.mean(all_strengths)

        # Update usage counts
        for pattern in pattern_match.matched_patterns:
            pattern.usage_count += 1
            pattern.last_used = datetime.utcnow()
            self.stats.pattern_usage_frequency[pattern.pattern_id] += 1

        # Store history
        self.pattern_history.append(pattern_match)

        # Keep recent history only
        if len(self.pattern_history) > 100:
            self.pattern_history = self.pattern_history[-100:]

    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pattern matching statistics"""
        return {
            "basic_stats": {
                "total_patterns_loaded": self.stats.total_patterns_loaded,
                "total_matches_performed": self.stats.total_matches_performed,
                "successful_matches": self.stats.successful_matches,
                "success_rate": self.stats.successful_matches
                / max(1, self.stats.total_matches_performed),
                "breakthrough_patterns_found": self.stats.breakthrough_patterns_found,
                "average_match_strength": self.stats.average_match_strength,
            },
            "breakthrough_stats": {
                "breakthrough_patterns_available": len(self.breakthrough_patterns),
                "breakthrough_success_rate": self.stats.breakthrough_patterns_found
                / max(1, self.stats.total_matches_performed),
                "top_breakthrough_patterns": [
                    p.pattern_id for p in self.breakthrough_patterns[:5]
                ],
            },
            "usage_patterns": {
                "most_used_patterns": dict(
                    Counter(self.stats.pattern_usage_frequency).most_common(10)
                ),
                "pattern_types": Counter(
                    [p.match_type.value for p in self.loaded_patterns.values()]
                ),
                "strength_distribution": self._get_strength_distribution(),
            },
            "metadata": {
                "last_updated": self.stats.last_updated.isoformat(),
                "cache_hours": self.pattern_cache_hours,
                "min_strength": self.min_pattern_strength,
                "breakthrough_threshold": self.breakthrough_threshold,
            },
        }

    def _get_strength_distribution(self) -> Dict[str, int]:
        """Get distribution of pattern strengths"""
        distribution = {"weak": 0, "moderate": 0, "strong": 0, "exceptional": 0}

        for pattern in self.loaded_patterns.values():
            if pattern.strength < 0.5:
                distribution["weak"] += 1
            elif pattern.strength < 0.7:
                distribution["moderate"] += 1
            elif pattern.strength < 0.85:
                distribution["strong"] += 1
            else:
                distribution["exceptional"] += 1

        return distribution


# Singleton pattern for system-wide access
_nway_pattern_matcher_instance = None


def get_nway_pattern_matcher(
    min_pattern_strength: float = 0.5,
    breakthrough_threshold: float = 0.8,
    max_patterns_per_match: int = 10,
    pattern_cache_hours: int = 24,
) -> NWayPatternMatcher:
    """Get singleton N-Way Pattern Matcher instance"""
    global _nway_pattern_matcher_instance

    if _nway_pattern_matcher_instance is None:
        _nway_pattern_matcher_instance = NWayPatternMatcher(
            min_pattern_strength=min_pattern_strength,
            breakthrough_threshold=breakthrough_threshold,
            max_patterns_per_match=max_patterns_per_match,
            pattern_cache_hours=pattern_cache_hours,
        )

        logging.getLogger(__name__).info(
            "✅ N-Way Pattern Matcher initialized (singleton)"
        )

    return _nway_pattern_matcher_instance
