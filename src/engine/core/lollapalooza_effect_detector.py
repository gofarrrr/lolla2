#!/usr/bin/env python3
"""
Lollapalooza Effect Detection Engine for METIS Cognitive Platform
Detects breakthrough moments when multiple mental models combine for exceptional insights

Based on Charlie Munger's concept: "The really big effects come from the
intersection of multiple causes" - identifies when cognitive systems achieve
exceptional performance through synergistic model combinations.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
import statistics
import numpy as np


class LollapaloozaTrigger(str, Enum):
    """Types of triggers that can create lollapalooza effects"""

    MODEL_SYNERGY = "model_synergy"  # Multiple models working together
    CONFIDENCE_CONVERGENCE = "confidence_convergence"  # High confidence across models
    PATTERN_BREAKTHROUGH = "pattern_breakthrough"  # Novel pattern recognition
    RESEARCH_AMPLIFICATION = "research_amplification"  # Research enhancing reasoning
    CALIBRATION_PRECISION = "calibration_precision"  # Exceptional calibration accuracy
    BAYESIAN_CONVERGENCE = "bayesian_convergence"  # Bayesian learning breakthrough
    NWAY_INTERACTION = "nway_interaction"  # N-WAY pattern activation
    VALUE_ALIGNMENT = "value_alignment"  # Strong value proposition match


class LollapaloozaIntensity(str, Enum):
    """Intensity levels of lollapalooza effects"""

    MODERATE = "moderate"  # 0.7-0.79 - Notable synergy
    STRONG = "strong"  # 0.8-0.89 - Strong breakthrough
    EXCEPTIONAL = "exceptional"  # 0.9+ - Extraordinary results


@dataclass
class LollapaloozaComponent:
    """Individual component contributing to lollapalooza effect"""

    component_type: str
    contribution_score: float
    confidence_level: float
    evidence: Dict[str, Any]
    timestamp: datetime


@dataclass
class LollapaloozaEffect:
    """Detected lollapalooza effect with full context"""

    effect_id: str
    overall_score: float
    intensity: LollapaloozaIntensity
    active_triggers: List[str]
    trigger_details: Dict[str, Dict[str, Any]]
    contributing_components: Dict[str, float]

    # Context information
    engagement_id: Optional[str] = None
    problem_statement: str = ""
    business_context: Dict[str, Any] = field(default_factory=dict)

    # Effect characteristics
    synergy_multiplier: float = 1.0
    breakthrough_insights: List[str] = field(default_factory=list)
    confidence_boost: float = 0.0

    # Metadata
    detected_at: datetime = field(default_factory=datetime.utcnow)
    detection_method: str = ""
    validation_score: float = 0.0
    context_factors: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LollapaloozaStats:
    """Statistics about lollapalooza effects over time"""

    total_effects_detected: int = 0
    effects_by_intensity: Dict[LollapaloozaIntensity, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    effects_by_trigger: Dict[LollapaloozaTrigger, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    average_score: float = 0.0
    max_score: float = 0.0
    detection_rate: float = 0.0  # Effects per engagement
    last_effect_time: Optional[datetime] = None


class LollapaloozaEffectDetector:
    """
    Advanced detector for identifying lollapalooza effects in cognitive processing

    Monitors multiple dimensions of cognitive performance to identify breakthrough
    moments when the combination of effects creates exceptional results beyond
    the sum of individual components.

    Key Detection Mechanisms:
    1. Multi-model synergy analysis
    2. Confidence convergence detection
    3. Pattern breakthrough identification
    4. Research amplification effects
    5. Calibration precision spikes
    """

    def __init__(
        self,
        detection_threshold: float = 0.7,
        strong_threshold: float = 0.8,
        exceptional_threshold: float = 0.9,
        retention_hours: int = 48,
    ):

        self.logger = logging.getLogger(__name__)
        self.detection_threshold = detection_threshold
        self.strong_threshold = strong_threshold
        self.exceptional_threshold = exceptional_threshold
        self.retention_hours = retention_hours

        # Effect storage and tracking
        self.detected_effects: List[LollapaloozaEffect] = []
        self.component_contributions: Dict[str, List[LollapaloozaComponent]] = (
            defaultdict(list)
        )
        self.stats = LollapaloozaStats()

        # Detection weights for different triggers
        self.trigger_weights = {
            LollapaloozaTrigger.MODEL_SYNERGY: 0.25,
            LollapaloozaTrigger.CONFIDENCE_CONVERGENCE: 0.20,
            LollapaloozaTrigger.PATTERN_BREAKTHROUGH: 0.15,
            LollapaloozaTrigger.RESEARCH_AMPLIFICATION: 0.15,
            LollapaloozaTrigger.CALIBRATION_PRECISION: 0.10,
            LollapaloozaTrigger.BAYESIAN_CONVERGENCE: 0.05,
            LollapaloozaTrigger.NWAY_INTERACTION: 0.05,
            LollapaloozaTrigger.VALUE_ALIGNMENT: 0.05,
        }

        self.logger.info(
            f"✅ Lollapalooza Effect Detector initialized (threshold: {detection_threshold:.2f})"
        )

    async def analyze_cognitive_engagement(
        self,
        engagement_data: Dict[str, Any],
        reasoning_results: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Optional[LollapaloozaEffect]:
        """
        Analyze a cognitive engagement for lollapalooza effects

        Args:
            engagement_data: Core engagement information
            reasoning_results: Results from model applications
            context: Additional context (research, calibration, etc.)

        Returns:
            LollapaloozaEffect if detected, None otherwise
        """
        if not reasoning_results or len(reasoning_results) < 2:
            return None  # Need multiple components for lollapalooza

        # Step 1: Analyze individual components
        components = await self._analyze_components(reasoning_results, context)

        # Step 2: Detect potential triggers
        triggers = await self._detect_triggers(components, context)

        if not triggers:
            return None

        # Step 3: Calculate overall lollapalooza score
        overall_score = await self._calculate_lollapalooza_score(
            components, triggers, context
        )

        if overall_score < self.detection_threshold:
            return None

        # Step 4: Create lollapalooza effect record
        effect = await self._create_lollapalooza_effect(
            overall_score, triggers, components, engagement_data, context
        )

        # Step 5: Store and track effect
        await self._record_effect(effect)

        self.logger.info(
            f"🚀 Lollapalooza effect detected! Score: {overall_score:.3f}, Intensity: {effect.intensity.value}"
        )

        return effect

    async def _analyze_components(
        self, reasoning_results: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Analyze cognitive components for their contribution potential"""
        components = {}

        # Extract mental model synergy
        models_applied = reasoning_results.get("mental_models_applied", [])
        confidence_scores = reasoning_results.get("confidence_scores", [])

        if models_applied and confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            model_synergy = min(
                1.0, avg_confidence * (len(models_applied) / 5)
            )  # Normalize to 5 models
            components["model_synergy"] = model_synergy
        else:
            components["model_synergy"] = 0.5

        # Extract confidence alignment
        if confidence_scores:
            confidence_std = (
                np.std(confidence_scores) if len(confidence_scores) > 1 else 0
            )
            confidence_alignment = max(
                0.0, 1.0 - confidence_std
            )  # Lower std = higher alignment
            components["confidence_alignment"] = confidence_alignment
        else:
            components["confidence_alignment"] = 0.5

        # Extract pattern novelty
        pattern_data = reasoning_results.get("pattern_recognition", {})
        if isinstance(pattern_data, dict):
            novel_patterns = pattern_data.get("novel_patterns", 0)
            total_patterns = pattern_data.get("patterns_identified", 1)
            pattern_novelty = min(1.0, novel_patterns / max(1, total_patterns))
            components["pattern_novelty"] = pattern_novelty
        else:
            components["pattern_novelty"] = 0.3

        # Extract research amplification
        research_data = reasoning_results.get("research_integration", {})
        if isinstance(research_data, dict):
            research_quality = research_data.get("research_quality", 0.5)
            synthesis_quality = research_data.get("synthesis_quality", 0.5)
            research_amplification = (research_quality + synthesis_quality) / 2
            components["research_amplification"] = research_amplification
        else:
            components["research_amplification"] = 0.4

        # Extract reasoning coherence
        reasoning_coherence = reasoning_results.get("reasoning_coherence", 0.7)
        components["reasoning_coherence"] = reasoning_coherence

        # Extract evidence convergence
        evidence_strength = reasoning_results.get("evidence_strength", 0.6)
        components["evidence_convergence"] = evidence_strength

        # Extract hypothesis validation quality
        hypothesis_data = reasoning_results.get("hypothesis_validation", {})
        if isinstance(hypothesis_data, dict):
            validation_score = hypothesis_data.get("validation_score", 0.5)
            hypotheses_tested = hypothesis_data.get("hypotheses_tested", 1)
            # Boost score for more hypotheses tested
            hypothesis_bonus = min(0.2, hypotheses_tested * 0.05)
            components["hypothesis_validation"] = min(
                1.0, validation_score + hypothesis_bonus
            )
        else:
            components["hypothesis_validation"] = 0.5

        # Calculate cognitive complexity based on context
        complexity_indicators = 0
        problem_complexity = context.get("problem_complexity", "medium")
        if problem_complexity in ["high", "very_high"]:
            complexity_indicators += 0.3

        time_spent = context.get("time_spent", 30000)  # milliseconds
        if time_spent > 40000:  # Over 40 seconds suggests thorough analysis
            complexity_indicators += 0.2

        mental_models_count = len(models_applied) if models_applied else 3
        if mental_models_count >= 4:
            complexity_indicators += 0.2

        components["cognitive_complexity"] = min(1.0, 0.5 + complexity_indicators)

        return components

    def _assess_reasoning_quality(self, result: Dict[str, Any]) -> float:
        """Assess the quality of reasoning in a result"""
        quality_score = 0.5  # Base score

        reasoning_text = result.get("reasoning_text", "")
        evidence_sources = result.get("evidence_sources", [])
        assumptions = result.get("assumptions_made", [])

        # Quality indicators
        if len(reasoning_text) > 100:  # Substantial reasoning
            quality_score += 0.1

        if len(evidence_sources) > 0:  # Has evidence
            quality_score += 0.1 + min(0.2, len(evidence_sources) * 0.05)

        if len(assumptions) > 0:  # Explicit assumptions
            quality_score += 0.1

        # Look for quality keywords in reasoning
        quality_keywords = [
            "therefore",
            "because",
            "evidence suggests",
            "analysis indicates",
            "systematic",
            "comprehensive",
            "validated",
            "verified",
        ]
        keyword_count = sum(
            1
            for keyword in quality_keywords
            if keyword.lower() in reasoning_text.lower()
        )
        quality_score += min(0.2, keyword_count * 0.05)

        return min(1.0, quality_score)

    async def _detect_triggers(
        self, components: Dict[str, float], context: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Detect lollapalooza triggers from component analysis"""
        triggers = {}

        # 1. Model Synergy Trigger - Multiple models working exceptionally well together
        model_synergy_score = components.get("model_synergy", 0.5)
        triggers["model_synergy"] = {
            "score": model_synergy_score,
            "active": model_synergy_score >= self.detection_threshold,
            "description": f"Mental model synergy at {model_synergy_score:.3f}",
        }

        # 2. Confidence Convergence - High confidence across all models
        confidence_alignment = components.get("confidence_alignment", 0.5)
        triggers["confidence_convergence"] = {
            "score": confidence_alignment,
            "active": confidence_alignment >= self.detection_threshold,
            "description": f"Confidence alignment at {confidence_alignment:.3f}",
        }

        # 3. Pattern Breakthrough - Novel patterns with strong evidence
        pattern_novelty = components.get("pattern_novelty", 0.3)
        breakthrough_threshold = (
            0.6  # Lower threshold as pattern novelty is naturally lower
        )
        triggers["pattern_breakthrough"] = {
            "score": pattern_novelty,
            "active": pattern_novelty >= breakthrough_threshold,
            "description": f"Pattern novelty at {pattern_novelty:.3f}",
        }

        # 4. Research Amplification - Research significantly enhancing reasoning
        research_amplification = components.get("research_amplification", 0.4)
        triggers["research_amplification"] = {
            "score": research_amplification,
            "active": research_amplification >= self.detection_threshold,
            "description": f"Research amplification at {research_amplification:.3f}",
        }

        # 5. Calibration Precision - Extremely well-calibrated reasoning
        reasoning_coherence = components.get("reasoning_coherence", 0.7)
        precision_threshold = 0.85  # High threshold for precision
        triggers["calibration_precision"] = {
            "score": reasoning_coherence,
            "active": reasoning_coherence >= precision_threshold,
            "description": f"Reasoning coherence at {reasoning_coherence:.3f}",
        }

        # 6. Bayesian Convergence - Strong evidence convergence
        evidence_convergence = components.get("evidence_convergence", 0.6)
        triggers["bayesian_convergence"] = {
            "score": evidence_convergence,
            "active": evidence_convergence >= self.detection_threshold,
            "description": f"Evidence convergence at {evidence_convergence:.3f}",
        }

        # 7. N-Way Interaction - Multiple cognitive systems creating synergistic effects
        cognitive_complexity = components.get("cognitive_complexity", 0.5)
        hypothesis_validation = components.get("hypothesis_validation", 0.5)
        nway_score = (
            cognitive_complexity + hypothesis_validation + model_synergy_score
        ) / 3
        nway_threshold = 0.75  # High threshold for n-way interactions
        triggers["nway_interaction"] = {
            "score": nway_score,
            "active": nway_score >= nway_threshold,
            "description": f"N-way interaction score at {nway_score:.3f}",
        }

        # 8. Value Alignment - Strong alignment with analytical rigor
        hypothesis_validation = components.get("hypothesis_validation", 0.5)
        triggers["value_alignment"] = {
            "score": hypothesis_validation,
            "active": hypothesis_validation >= self.detection_threshold,
            "description": f"Hypothesis validation at {hypothesis_validation:.3f}",
        }

        return triggers

    async def _calculate_lollapalooza_score(
        self,
        components: Dict[str, float],
        triggers: Dict[str, Dict[str, Any]],
        context: Dict[str, Any],
    ) -> float:
        """Calculate overall lollapalooza effect score"""
        if not components or not triggers:
            return 0.0

        # Base score from component synergy
        component_scores = list(components.values())
        base_score = sum(component_scores) / len(component_scores)

        # Active trigger contributions with weights
        active_triggers = [
            trigger_name
            for trigger_name, trigger_data in triggers.items()
            if trigger_data.get("active", False)
        ]

        if not active_triggers:
            return min(
                base_score, 0.6
            )  # Can't have lollapalooza without active triggers

        # Calculate trigger weight contribution
        total_trigger_weight = 0.0
        for trigger_name in active_triggers:
            # Map trigger names to enum values for weight lookup
            trigger_enum_map = {
                "model_synergy": LollapaloozaTrigger.MODEL_SYNERGY,
                "confidence_convergence": LollapaloozaTrigger.CONFIDENCE_CONVERGENCE,
                "pattern_breakthrough": LollapaloozaTrigger.PATTERN_BREAKTHROUGH,
                "research_amplification": LollapaloozaTrigger.RESEARCH_AMPLIFICATION,
                "calibration_precision": LollapaloozaTrigger.CALIBRATION_PRECISION,
                "bayesian_convergence": LollapaloozaTrigger.BAYESIAN_CONVERGENCE,
                "nway_interaction": LollapaloozaTrigger.NWAY_INTERACTION,
                "value_alignment": LollapaloozaTrigger.VALUE_ALIGNMENT,
            }

            trigger_enum = trigger_enum_map.get(trigger_name)
            if trigger_enum:
                weight = self.trigger_weights.get(trigger_enum, 0.0)
                trigger_score = triggers[trigger_name].get("score", 0.0)
                total_trigger_weight += weight * trigger_score

        # Weighted combination of base score and trigger contributions
        weighted_score = (base_score * 0.6) + (total_trigger_weight * 0.4)

        # Synergy multiplier for multiple active triggers
        if len(active_triggers) >= 2:
            synergy_multiplier = (
                1.0 + (len(active_triggers) - 1) * 0.1
            )  # 10% per additional trigger
            weighted_score *= synergy_multiplier

        # Context amplifiers
        amplification_factor = 1.0

        # High-complexity problems get bonus
        problem_complexity = context.get("problem_complexity", "medium")
        if problem_complexity in ["high", "very_high"]:
            amplification_factor *= 1.05

        # Time investment indicates thorough analysis
        time_spent = context.get("time_spent", 30000)
        if time_spent > 45000:  # Over 45 seconds
            amplification_factor *= 1.05

        final_score = min(1.0, weighted_score * amplification_factor)

        return final_score

    async def _create_lollapalooza_effect(
        self,
        overall_score: float,
        triggers: Dict[str, Dict[str, Any]],
        components: Dict[str, float],
        engagement_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> LollapaloozaEffect:
        """Create comprehensive lollapalooza effect record"""

        # Determine intensity
        if overall_score >= self.exceptional_threshold:
            intensity = LollapaloozaIntensity.EXCEPTIONAL
        elif overall_score >= self.strong_threshold:
            intensity = LollapaloozaIntensity.STRONG
        else:
            intensity = LollapaloozaIntensity.MODERATE

        # Calculate synergy multiplier
        component_scores = list(components.values())
        individual_avg = (
            sum(component_scores) / len(component_scores) if component_scores else 0.5
        )
        synergy_multiplier = (
            overall_score / individual_avg if individual_avg > 0 else 1.0
        )

        # Extract active triggers
        active_triggers = [
            trigger_name
            for trigger_name, trigger_data in triggers.items()
            if trigger_data.get("active", False)
        ]

        # Extract breakthrough insights from high-scoring components
        breakthrough_insights = []
        for component_name, score in components.items():
            if score > 0.8:
                insight = f"{component_name.replace('_', ' ').title()}: {score:.3f} - Exceptional performance"
                breakthrough_insights.append(insight)

        # Add trigger-specific insights
        for trigger_name, trigger_data in triggers.items():
            if trigger_data.get("active", False):
                insight = f"🔥 {trigger_data.get('description', trigger_name)}"
                breakthrough_insights.append(insight)

        # Calculate confidence boost (estimate from model synergy and alignment)
        model_synergy = components.get("model_synergy", 0.5)
        confidence_alignment = components.get("confidence_alignment", 0.5)
        confidence_boost = max(0, (model_synergy + confidence_alignment) / 2 - 0.7)

        effect = LollapaloozaEffect(
            effect_id=f"lolla_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{len(self.detected_effects)}",
            overall_score=overall_score,
            intensity=intensity,
            active_triggers=active_triggers,
            trigger_details=triggers,
            contributing_components=components,
            engagement_id=engagement_data.get("engagement_id"),
            problem_statement=engagement_data.get("problem_statement", "")[:500],
            business_context=engagement_data.get("business_context", {}),
            synergy_multiplier=synergy_multiplier,
            breakthrough_insights=breakthrough_insights,
            confidence_boost=confidence_boost,
            detected_at=datetime.utcnow(),
            detection_method="multi_trigger_analysis",
            validation_score=overall_score,  # Simple validation for now
            context_factors=context,
        )

        return effect

    async def _record_effect(self, effect: LollapaloozaEffect):
        """Record detected lollapalooza effect and update statistics"""
        self.detected_effects.append(effect)

        # Update statistics
        self.stats.total_effects_detected += 1
        self.stats.effects_by_intensity[effect.intensity] += 1

        for trigger_name in effect.active_triggers:
            # Map trigger names back to enums for statistics
            trigger_enum_map = {
                "model_synergy": LollapaloozaTrigger.MODEL_SYNERGY,
                "confidence_convergence": LollapaloozaTrigger.CONFIDENCE_CONVERGENCE,
                "pattern_breakthrough": LollapaloozaTrigger.PATTERN_BREAKTHROUGH,
                "research_amplification": LollapaloozaTrigger.RESEARCH_AMPLIFICATION,
                "calibration_precision": LollapaloozaTrigger.CALIBRATION_PRECISION,
                "bayesian_convergence": LollapaloozaTrigger.BAYESIAN_CONVERGENCE,
                "nway_interaction": LollapaloozaTrigger.NWAY_INTERACTION,
                "value_alignment": LollapaloozaTrigger.VALUE_ALIGNMENT,
            }
            trigger_enum = trigger_enum_map.get(trigger_name)
            if trigger_enum:
                self.stats.effects_by_trigger[trigger_enum] += 1

        # Update running averages
        all_scores = [e.overall_score for e in self.detected_effects]
        self.stats.average_score = statistics.mean(all_scores)
        self.stats.max_score = max(all_scores)
        self.stats.last_effect_time = effect.detected_at

        # Store components for analysis
        for component_name, score in effect.contributing_components.items():
            self.component_contributions[component_name].append(score)

        # Cleanup old effects
        await self._cleanup_old_effects()

    async def _cleanup_old_effects(self):
        """Remove effects older than retention period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.retention_hours)

        old_count = len(self.detected_effects)
        self.detected_effects = [
            e for e in self.detected_effects if e.detected_at > cutoff_time
        ]
        new_count = len(self.detected_effects)

        if old_count > new_count:
            self.logger.debug(
                f"🧹 Cleaned up {old_count - new_count} old lollapalooza effects"
            )

    def get_recent_effects(self, hours_back: int = 24) -> List[LollapaloozaEffect]:
        """Get lollapalooza effects from recent time period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        return [e for e in self.detected_effects if e.detected_at > cutoff_time]

    def get_statistics(self) -> LollapaloozaStats:
        """Get current lollapalooza detection statistics"""
        return self.stats

    def get_top_performers(self, limit: int = 10) -> List[LollapaloozaEffect]:
        """Get top-performing lollapalooza effects by score"""
        return sorted(
            self.detected_effects, key=lambda e: e.overall_score, reverse=True
        )[:limit]

    def analyze_trigger_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in lollapalooza triggers"""
        if not self.detected_effects:
            return {"message": "No lollapalooza effects detected yet"}

        # Trigger co-occurrence analysis
        trigger_combinations = defaultdict(int)
        for effect in self.detected_effects:
            trigger_set = tuple(sorted([t.value for t in effect.primary_triggers]))
            trigger_combinations[trigger_set] += 1

        # Most effective trigger combinations
        top_combinations = sorted(
            trigger_combinations.items(), key=lambda x: x[1], reverse=True
        )

        # Average scores by trigger
        trigger_scores = defaultdict(list)
        for effect in self.detected_effects:
            for trigger in effect.primary_triggers:
                trigger_scores[trigger].append(effect.overall_score)

        avg_scores_by_trigger = {
            trigger.value: statistics.mean(scores)
            for trigger, scores in trigger_scores.items()
        }

        return {
            "total_effects": len(self.detected_effects),
            "top_trigger_combinations": [
                {"triggers": list(combo), "count": count}
                for combo, count in top_combinations[:5]
            ],
            "average_scores_by_trigger": avg_scores_by_trigger,
            "most_effective_trigger": (
                max(avg_scores_by_trigger.items(), key=lambda x: x[1])[0]
                if avg_scores_by_trigger
                else None
            ),
        }

    def export_effects_data(self) -> Dict[str, Any]:
        """Export comprehensive lollapalooza effects data"""
        return {
            "statistics": {
                "total_effects": self.stats.total_effects_detected,
                "average_score": self.stats.average_score,
                "max_score": self.stats.max_score,
                "effects_by_intensity": dict(self.stats.effects_by_intensity),
                "effects_by_trigger": {
                    k.value: v for k, v in self.stats.effects_by_trigger.items()
                },
                "detection_rate": self.stats.detection_rate,
                "last_effect_time": (
                    self.stats.last_effect_time.isoformat()
                    if self.stats.last_effect_time
                    else None
                ),
            },
            "recent_effects": [
                {
                    "effect_id": effect.effect_id,
                    "score": effect.overall_score,
                    "intensity": effect.intensity.value,
                    "triggers": [t.value for t in effect.primary_triggers],
                    "component_count": len(effect.components),
                    "synergy_multiplier": effect.synergy_multiplier,
                    "confidence_boost": effect.confidence_boost,
                    "detected_at": effect.detected_at.isoformat(),
                    "problem_statement": (
                        effect.problem_statement[:100] + "..."
                        if len(effect.problem_statement) > 100
                        else effect.problem_statement
                    ),
                }
                for effect in self.get_recent_effects(hours_back=24)
            ],
            "trigger_analysis": self.analyze_trigger_patterns(),
            "export_timestamp": datetime.utcnow().isoformat(),
        }


# Global lollapalooza detector instance
_lollapalooza_detector_instance: Optional[LollapaloozaEffectDetector] = None


def get_lollapalooza_detector(
    detection_threshold: float = 0.7,
) -> LollapaloozaEffectDetector:
    """Get or create global lollapalooza effect detector instance"""
    global _lollapalooza_detector_instance

    if _lollapalooza_detector_instance is None:
        _lollapalooza_detector_instance = LollapaloozaEffectDetector(
            detection_threshold=detection_threshold
        )

    return _lollapalooza_detector_instance


async def detect_lollapalooza_effect(
    engagement_data: Dict[str, Any],
    reasoning_results: List[Dict[str, Any]],
    context: Dict[str, Any],
) -> Optional[LollapaloozaEffect]:
    """Convenience function to detect lollapalooza effects"""
    detector = get_lollapalooza_detector()
    return await detector.analyze_cognitive_engagement(
        engagement_data, reasoning_results, context
    )
