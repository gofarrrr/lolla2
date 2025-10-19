#!/usr/bin/env python3
"""
METIS Self-Doubt Calibration Engine
Sprint 3.3: Dynamic confidence calibration based on challenge outcomes and learning

This engine implements a sophisticated self-doubt mechanism that:
1. Learns from challenge outcomes and their accuracy
2. Dynamically calibrates confidence adjustments based on historical performance
3. Adapts to different contexts and domains
4. Provides meta-cognitive awareness of reasoning quality
5. Implements Bayesian updating of confidence based on evidence

Key Features:
- Historical challenge outcome tracking and learning
- Context-aware confidence calibration
- Bayesian confidence updating
- Meta-cognitive confidence assessment
- Overconfidence detection and correction
- Domain-specific calibration learning
- Challenge effectiveness measurement
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# Core METIS components
from src.engine.models.data_contracts import ReasoningStep
from src.interfaces.context_intelligence_interface import IContextIntelligence
from src.engine.engines.internal_challenger_system import (
    InternalChallengerResult,
    AssumptionChallenge,
    ChallengeFramework,
)

logger = logging.getLogger(__name__)


class ConfidenceCalibrationLevel(Enum):
    """Levels of confidence calibration sophistication"""

    BASIC = "basic"  # Simple linear adjustment
    ADAPTIVE = "adaptive"  # Learning-based adjustment
    BAYESIAN = "bayesian"  # Bayesian updating with priors
    METACOGNITIVE = "metacognitive"  # Self-aware confidence assessment


class CalibrationDomain(Enum):
    """Different domains for specialized calibration"""

    BUSINESS_STRATEGY = "business_strategy"
    MARKET_ANALYSIS = "market_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    TECHNICAL_PLANNING = "technical_planning"
    STAKEHOLDER_DYNAMICS = "stakeholder_dynamics"
    FINANCIAL_MODELING = "financial_modeling"
    GENERAL = "general"


class OverconfidencePattern(Enum):
    """Patterns of overconfidence to detect and correct"""

    SYSTEMATIC_OPTIMISM = "systematic_optimism"  # Consistently too optimistic
    AVAILABILITY_BIAS = "availability_bias"  # Overweighting recent examples
    CONFIRMATION_SEEKING = "confirmation_seeking"  # Seeking confirming evidence
    PLANNING_FALLACY = "planning_fallacy"  # Underestimating complexity
    ANCHORING_BIAS = "anchoring_bias"  # Stuck on initial estimates
    EXPERTISE_OVERCONFIDENCE = (
        "expertise_overconfidence"  # Domain expert overconfidence
    )


@dataclass
class ChallengeOutcome:
    """Records the outcome of a specific challenge"""

    challenge_id: str
    original_challenge: AssumptionChallenge
    outcome_accuracy: float  # 0.0-1.0, how accurate the challenge was
    confidence_adjustment_made: float
    actual_outcome: Optional[str] = None  # What actually happened
    outcome_validation_date: Optional[datetime] = None
    domain: CalibrationDomain = CalibrationDomain.GENERAL
    context_factors: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfidenceCalibrationMetrics:
    """Metrics for confidence calibration performance"""

    domain: CalibrationDomain
    total_challenges: int = 0
    accurate_challenges: int = 0
    overconfident_instances: int = 0
    underconfident_instances: int = 0
    average_confidence_error: float = 0.0
    calibration_score: float = 0.0  # How well-calibrated confidence is
    brier_score: float = 0.0  # Probabilistic accuracy score
    last_updated: datetime = field(default_factory=datetime.now)

    # Pattern detection
    detected_overconfidence_patterns: List[OverconfidencePattern] = field(
        default_factory=list
    )
    pattern_confidence_adjustments: Dict[str, float] = field(default_factory=dict)


@dataclass
class MetacognitiveAssessment:
    """Meta-cognitive assessment of reasoning quality"""

    reasoning_quality_score: float  # 0.0-1.0
    confidence_appropriateness: float  # 0.0-1.0
    challenge_resistance_score: float  # How well reasoning stands up to challenges
    domain_expertise_indicator: float  # Estimated domain expertise
    uncertainty_acknowledgment: float  # How well uncertainty is acknowledged
    bias_susceptibility_score: float  # Susceptibility to various biases
    recommended_confidence_adjustment: float
    meta_reasoning: str  # Explanation of meta-cognitive assessment


class SelfDoubtCalibrationEngine:
    """
    Self-Doubt Calibration Engine for dynamic confidence adjustment

    This engine learns from challenge outcomes to improve confidence calibration
    and implements sophisticated self-doubt mechanisms.
    """

    def __init__(
        self,
        context_intelligence: Optional[IContextIntelligence] = None,
        calibration_level: ConfidenceCalibrationLevel = ConfidenceCalibrationLevel.ADAPTIVE,
        learning_rate: float = 0.1,
        overconfidence_penalty: float = 0.2,
        min_samples_for_learning: int = 5,
        logger: Optional[logging.Logger] = None,
    ):
        self.context_intelligence = context_intelligence
        self.calibration_level = calibration_level
        self.learning_rate = learning_rate
        self.overconfidence_penalty = overconfidence_penalty
        self.min_samples_for_learning = min_samples_for_learning
        self.logger = logger or logging.getLogger(__name__)

        # Learning and tracking systems
        self.challenge_outcomes: List[ChallengeOutcome] = []
        self.domain_metrics: Dict[CalibrationDomain, ConfidenceCalibrationMetrics] = {}
        self.confidence_history: List[Tuple[float, float, str]] = (
            []
        )  # (original, adjusted, domain)
        self.overconfidence_patterns: Dict[OverconfidencePattern, float] = {}

        # Bayesian priors for confidence calibration
        self.bayesian_priors = {
            CalibrationDomain.BUSINESS_STRATEGY: {"mean": 0.7, "variance": 0.2},
            CalibrationDomain.MARKET_ANALYSIS: {"mean": 0.6, "variance": 0.3},
            CalibrationDomain.RISK_ASSESSMENT: {"mean": 0.5, "variance": 0.3},
            CalibrationDomain.TECHNICAL_PLANNING: {"mean": 0.8, "variance": 0.15},
            CalibrationDomain.STAKEHOLDER_DYNAMICS: {"mean": 0.5, "variance": 0.4},
            CalibrationDomain.FINANCIAL_MODELING: {"mean": 0.7, "variance": 0.25},
            CalibrationDomain.GENERAL: {"mean": 0.65, "variance": 0.25},
        }

        # Initialize domain metrics
        for domain in CalibrationDomain:
            self.domain_metrics[domain] = ConfidenceCalibrationMetrics(domain=domain)

        self.logger.info("🎯 Self-Doubt Calibration Engine initialized")
        self.logger.info(f"   Calibration level: {calibration_level.value}")
        self.logger.info(f"   Learning rate: {learning_rate}")
        self.logger.info(f"   Overconfidence penalty: {overconfidence_penalty}")

    async def calibrate_confidence(
        self,
        original_confidence: float,
        reasoning_steps: List[ReasoningStep],
        challenger_result: InternalChallengerResult,
        domain: CalibrationDomain = CalibrationDomain.GENERAL,
        context_factors: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, MetacognitiveAssessment]:
        """
        Calibrate confidence based on challenger results and historical learning

        Args:
            original_confidence: Original confidence score
            reasoning_steps: Reasoning steps being evaluated
            challenger_result: Results from internal challenger
            domain: Domain for specialized calibration
            context_factors: Additional context factors

        Returns:
            Tuple of (calibrated_confidence, metacognitive_assessment)
        """
        self.logger.info(f"🎯 Calibrating confidence for domain {domain.value}")
        self.logger.info(f"   Original confidence: {original_confidence:.3f}")
        self.logger.info(
            f"   Challenges generated: {len(challenger_result.challenges)}"
        )

        # Step 1: Apply basic challenge-based adjustment
        challenge_adjustment = self._calculate_challenge_adjustment(challenger_result)

        # Step 2: Apply domain-specific learning adjustments
        domain_adjustment = await self._calculate_domain_adjustment(
            original_confidence, domain, reasoning_steps
        )

        # Step 3: Apply overconfidence pattern corrections
        pattern_adjustment = self._calculate_pattern_adjustment(
            original_confidence, reasoning_steps, challenger_result
        )

        # Step 4: Apply Bayesian updating (if enabled)
        bayesian_adjustment = 0.0
        if self.calibration_level in [
            ConfidenceCalibrationLevel.BAYESIAN,
            ConfidenceCalibrationLevel.METACOGNITIVE,
        ]:
            bayesian_adjustment = self._calculate_bayesian_adjustment(
                original_confidence, domain, challenger_result
            )

        # Step 5: Generate metacognitive assessment
        metacognitive_assessment = await self._generate_metacognitive_assessment(
            original_confidence, reasoning_steps, challenger_result, domain
        )

        # Combine adjustments
        total_adjustment = (
            challenge_adjustment
            + domain_adjustment
            + pattern_adjustment
            + bayesian_adjustment
            + metacognitive_assessment.recommended_confidence_adjustment
        )

        # Apply adjustment with bounds checking
        calibrated_confidence = max(
            0.0, min(1.0, original_confidence + total_adjustment)
        )

        # Record for learning
        self._record_calibration(
            original_confidence,
            calibrated_confidence,
            domain,
            challenger_result,
            context_factors,
        )

        self.logger.info("✅ Confidence calibration complete")
        self.logger.info(f"   Challenge adjustment: {challenge_adjustment:.3f}")
        self.logger.info(f"   Domain adjustment: {domain_adjustment:.3f}")
        self.logger.info(f"   Pattern adjustment: {pattern_adjustment:.3f}")
        self.logger.info(f"   Bayesian adjustment: {bayesian_adjustment:.3f}")
        self.logger.info(
            f"   Metacognitive adjustment: {metacognitive_assessment.recommended_confidence_adjustment:.3f}"
        )
        self.logger.info(f"   Final confidence: {calibrated_confidence:.3f}")

        return calibrated_confidence, metacognitive_assessment

    def _calculate_challenge_adjustment(
        self, challenger_result: InternalChallengerResult
    ) -> float:
        """Calculate confidence adjustment based on challenge results"""
        if not challenger_result.challenges:
            return 0.0

        # Base adjustment from challenger result
        base_adjustment = challenger_result.net_confidence_adjustment

        # Adjust based on challenge quality and diversity
        framework_diversity = len(
            set(c.challenge_framework for c in challenger_result.challenges)
        )
        diversity_bonus = min(0.1, framework_diversity * 0.02)

        # Red flags increase doubt
        red_flag_penalty = len(challenger_result.red_flags) * -0.05

        # High-impact challenges get more weight
        high_impact_challenges = len(
            [c for c in challenger_result.challenges if c.confidence_impact <= -0.4]
        )
        high_impact_penalty = high_impact_challenges * -0.03

        total_adjustment = (
            base_adjustment - diversity_bonus + red_flag_penalty + high_impact_penalty
        )

        # Cap adjustment to prevent extreme swings
        return max(-0.5, min(0.1, total_adjustment))

    async def _calculate_domain_adjustment(
        self,
        original_confidence: float,
        domain: CalibrationDomain,
        reasoning_steps: List[ReasoningStep],
    ) -> float:
        """Calculate domain-specific confidence adjustment based on historical learning"""
        domain_metrics = self.domain_metrics[domain]

        if domain_metrics.total_challenges < self.min_samples_for_learning:
            return 0.0  # Not enough data for learning

        # Calculate historical accuracy
        accuracy_rate = (
            domain_metrics.accurate_challenges / domain_metrics.total_challenges
        )

        # Adjust based on historical overconfidence patterns
        overconfidence_rate = (
            domain_metrics.overconfident_instances / domain_metrics.total_challenges
        )

        # Higher confidence gets more scrutiny
        confidence_scrutiny = (original_confidence - 0.5) * overconfidence_rate * -0.2

        # Poor historical calibration increases doubt
        calibration_adjustment = (0.5 - domain_metrics.calibration_score) * -0.1

        total_domain_adjustment = confidence_scrutiny + calibration_adjustment

        return max(-0.3, min(0.1, total_domain_adjustment))

    def _calculate_pattern_adjustment(
        self,
        original_confidence: float,
        reasoning_steps: List[ReasoningStep],
        challenger_result: InternalChallengerResult,
    ) -> float:
        """Calculate adjustment based on detected overconfidence patterns"""
        pattern_adjustment = 0.0

        # Detect patterns in current reasoning
        detected_patterns = self._detect_overconfidence_patterns(
            reasoning_steps, challenger_result
        )

        for pattern in detected_patterns:
            pattern_penalty = self.overconfidence_patterns.get(pattern, 0.0)
            pattern_adjustment += pattern_penalty * -0.1

            # Update pattern tracking
            self.overconfidence_patterns[pattern] = min(1.0, pattern_penalty + 0.1)

        # Systematic optimism detection
        if (
            original_confidence > 0.8
            and len(
                [
                    c
                    for c in challenger_result.challenges
                    if c.challenge_framework == ChallengeFramework.MUNGER_INVERSION
                ]
            )
            > 3
        ):
            pattern_adjustment -= (
                0.05  # Extra doubt for high confidence with many inversion challenges
            )

        return max(-0.2, pattern_adjustment)

    def _calculate_bayesian_adjustment(
        self,
        original_confidence: float,
        domain: CalibrationDomain,
        challenger_result: InternalChallengerResult,
    ) -> float:
        """Calculate Bayesian confidence adjustment using domain priors"""
        if self.calibration_level not in [
            ConfidenceCalibrationLevel.BAYESIAN,
            ConfidenceCalibrationLevel.METACOGNITIVE,
        ]:
            return 0.0

        prior = self.bayesian_priors[domain]
        domain_metrics = self.domain_metrics[domain]

        # Bayesian updating based on challenge evidence
        evidence_strength = len(challenger_result.challenges) / 10.0  # Normalize to 0-1
        evidence_direction = (
            -1 if challenger_result.net_confidence_adjustment < 0 else 1
        )

        # Update belief based on evidence
        posterior_mean = (
            prior["mean"] * (1 - evidence_strength)
            + original_confidence * evidence_strength
        )

        # Adjust confidence towards posterior
        bayesian_adjustment = (posterior_mean - original_confidence) * 0.3

        # Factor in historical calibration performance
        if domain_metrics.total_challenges > 0:
            calibration_factor = domain_metrics.calibration_score
            bayesian_adjustment *= calibration_factor

        return max(-0.3, min(0.2, bayesian_adjustment))

    async def _generate_metacognitive_assessment(
        self,
        original_confidence: float,
        reasoning_steps: List[ReasoningStep],
        challenger_result: InternalChallengerResult,
        domain: CalibrationDomain,
    ) -> MetacognitiveAssessment:
        """Generate metacognitive assessment of reasoning quality"""

        # Assess reasoning quality
        reasoning_quality = self._assess_reasoning_quality(reasoning_steps)

        # Assess confidence appropriateness
        confidence_appropriateness = self._assess_confidence_appropriateness(
            original_confidence, reasoning_steps, challenger_result
        )

        # Assess challenge resistance
        challenge_resistance = self._assess_challenge_resistance(challenger_result)

        # Estimate domain expertise
        domain_expertise = self._estimate_domain_expertise(reasoning_steps, domain)

        # Assess uncertainty acknowledgment
        uncertainty_acknowledgment = self._assess_uncertainty_acknowledgment(
            reasoning_steps
        )

        # Assess bias susceptibility
        bias_susceptibility = self._assess_bias_susceptibility(challenger_result)

        # Generate meta-reasoning explanation
        meta_reasoning = self._generate_meta_reasoning(
            reasoning_quality,
            confidence_appropriateness,
            challenge_resistance,
            domain_expertise,
            uncertainty_acknowledgment,
            bias_susceptibility,
        )

        # Calculate recommended confidence adjustment
        if self.calibration_level == ConfidenceCalibrationLevel.METACOGNITIVE:
            metacognitive_adjustment = self._calculate_metacognitive_adjustment(
                reasoning_quality, confidence_appropriateness, challenge_resistance
            )
        else:
            metacognitive_adjustment = 0.0

        return MetacognitiveAssessment(
            reasoning_quality_score=reasoning_quality,
            confidence_appropriateness=confidence_appropriateness,
            challenge_resistance_score=challenge_resistance,
            domain_expertise_indicator=domain_expertise,
            uncertainty_acknowledgment=uncertainty_acknowledgment,
            bias_susceptibility_score=bias_susceptibility,
            recommended_confidence_adjustment=metacognitive_adjustment,
            meta_reasoning=meta_reasoning,
        )

    def _assess_reasoning_quality(self, reasoning_steps: List[ReasoningStep]) -> float:
        """Assess the quality of reasoning steps"""
        if not reasoning_steps:
            return 0.0

        quality_factors = []

        for step in reasoning_steps:
            # Check for evidence backing
            evidence_quality = 0.5
            if hasattr(step, "evidence_sources") and step.evidence_sources:
                evidence_quality = min(1.0, len(step.evidence_sources) / 3.0)

            # Check for assumption awareness
            assumption_awareness = 0.5
            if hasattr(step, "assumptions_made") and step.assumptions_made:
                assumption_awareness = min(1.0, len(step.assumptions_made) / 5.0)

            # Check for confidence calibration
            confidence_calibration = 1.0
            if hasattr(step, "confidence_score") and step.confidence_score:
                if step.confidence_score > 0.9:  # Very high confidence is suspect
                    confidence_calibration = 0.6
                elif (
                    step.confidence_score < 0.3
                ):  # Very low confidence may indicate poor reasoning
                    confidence_calibration = 0.7

            step_quality = (
                evidence_quality + assumption_awareness + confidence_calibration
            ) / 3.0
            quality_factors.append(step_quality)

        return sum(quality_factors) / len(quality_factors)

    def _assess_confidence_appropriateness(
        self,
        original_confidence: float,
        reasoning_steps: List[ReasoningStep],
        challenger_result: InternalChallengerResult,
    ) -> float:
        """Assess whether confidence level is appropriate given the reasoning"""

        # High confidence with many challenges suggests overconfidence
        if original_confidence > 0.8 and len(challenger_result.challenges) > 20:
            return 0.3

        # Low confidence with few challenges might be underconfidence
        if original_confidence < 0.5 and len(challenger_result.challenges) < 5:
            return 0.7

        # Check for evidence-confidence alignment
        evidence_strength = 0.0
        for step in reasoning_steps:
            if hasattr(step, "evidence_sources") and step.evidence_sources:
                evidence_strength += len(step.evidence_sources)

        evidence_strength = min(1.0, evidence_strength / 10.0)  # Normalize

        # Confidence should roughly align with evidence strength
        confidence_evidence_alignment = 1.0 - abs(
            original_confidence - evidence_strength
        )

        return max(0.0, min(1.0, confidence_evidence_alignment))

    def _assess_challenge_resistance(
        self, challenger_result: InternalChallengerResult
    ) -> float:
        """Assess how well the reasoning stands up to challenges"""
        if not challenger_result.challenges:
            return 0.8  # No challenges found, reasoning might be solid

        # High-impact challenges suggest weak reasoning
        high_impact_challenges = len(
            [c for c in challenger_result.challenges if c.confidence_impact <= -0.4]
        )
        high_impact_ratio = high_impact_challenges / len(challenger_result.challenges)

        # Many red flags suggest poor resistance
        red_flag_penalty = min(0.5, len(challenger_result.red_flags) * 0.15)

        resistance_score = 1.0 - (high_impact_ratio * 0.6) - red_flag_penalty

        return max(0.0, resistance_score)

    def _estimate_domain_expertise(
        self, reasoning_steps: List[ReasoningStep], domain: CalibrationDomain
    ) -> float:
        """Estimate domain expertise based on reasoning sophistication"""
        if not reasoning_steps:
            return 0.5

        expertise_indicators = []

        for step in reasoning_steps:
            # Check for sophisticated mental models
            sophistication = 0.5
            if hasattr(step, "mental_model_applied") and step.mental_model_applied:
                advanced_models = [
                    "systems_thinking",
                    "first_principles",
                    "scenario_analysis",
                    "game_theory",
                ]
                if step.mental_model_applied in advanced_models:
                    sophistication = 0.8

            # Check for detailed insights
            insight_depth = 0.5
            if hasattr(step, "key_insights") and step.key_insights:
                insight_depth = min(1.0, len(step.key_insights) / 5.0)

            expertise_indicators.append((sophistication + insight_depth) / 2.0)

        return sum(expertise_indicators) / len(expertise_indicators)

    def _assess_uncertainty_acknowledgment(
        self, reasoning_steps: List[ReasoningStep]
    ) -> float:
        """Assess how well uncertainty is acknowledged in reasoning"""
        uncertainty_indicators = []

        for step in reasoning_steps:
            step_text = ""
            if hasattr(step, "reasoning_text") and step.reasoning_text:
                step_text = step.reasoning_text.lower()
            elif hasattr(step, "description"):
                step_text = step.description.lower()

            # Look for uncertainty language
            uncertainty_words = [
                "uncertain",
                "unclear",
                "might",
                "could",
                "possibly",
                "perhaps",
                "may",
                "assume",
            ]
            certainty_words = [
                "definitely",
                "certainly",
                "obviously",
                "clearly",
                "guaranteed",
                "always",
                "never",
            ]

            uncertainty_count = sum(
                1 for word in uncertainty_words if word in step_text
            )
            certainty_count = sum(1 for word in certainty_words if word in step_text)

            if uncertainty_count > certainty_count:
                uncertainty_indicators.append(0.8)
            elif certainty_count > uncertainty_count * 2:
                uncertainty_indicators.append(0.2)
            else:
                uncertainty_indicators.append(0.5)

        return (
            sum(uncertainty_indicators) / len(uncertainty_indicators)
            if uncertainty_indicators
            else 0.5
        )

    def _assess_bias_susceptibility(
        self, challenger_result: InternalChallengerResult
    ) -> float:
        """Assess susceptibility to various biases based on challenges"""
        if not challenger_result.challenges:
            return 0.5

        bias_challenges = [
            c
            for c in challenger_result.challenges
            if c.challenge_framework == ChallengeFramework.CONSTITUTIONAL_BIAS
        ]

        if not bias_challenges:
            return 0.5

        # More bias challenges found = higher susceptibility
        bias_density = len(bias_challenges) / len(challenger_result.challenges)

        return min(1.0, bias_density * 2.0)  # Scale to 0-1

    def _generate_meta_reasoning(
        self,
        reasoning_quality: float,
        confidence_appropriateness: float,
        challenge_resistance: float,
        domain_expertise: float,
        uncertainty_acknowledgment: float,
        bias_susceptibility: float,
    ) -> str:
        """Generate explanation of metacognitive assessment"""

        components = []

        if reasoning_quality >= 0.8:
            components.append("High-quality reasoning with good evidence backing")
        elif reasoning_quality <= 0.4:
            components.append(
                "Reasoning quality concerns - limited evidence or assumptions"
            )

        if confidence_appropriateness <= 0.4:
            components.append(
                "Confidence level appears inappropriate for evidence strength"
            )

        if challenge_resistance <= 0.4:
            components.append("Reasoning shows significant vulnerability to challenges")

        if domain_expertise >= 0.8:
            components.append("Strong domain expertise indicators present")
        elif domain_expertise <= 0.4:
            components.append("Limited domain expertise evident")

        if uncertainty_acknowledgment <= 0.3:
            components.append("Insufficient acknowledgment of uncertainty")

        if bias_susceptibility >= 0.7:
            components.append("High susceptibility to cognitive biases detected")

        if not components:
            return "Balanced metacognitive assessment with no major concerns identified"

        return "; ".join(components)

    def _calculate_metacognitive_adjustment(
        self,
        reasoning_quality: float,
        confidence_appropriateness: float,
        challenge_resistance: float,
    ) -> float:
        """Calculate confidence adjustment based on metacognitive assessment"""

        # Poor reasoning quality reduces confidence
        quality_adjustment = (reasoning_quality - 0.7) * 0.2

        # Inappropriate confidence gets corrected
        appropriateness_adjustment = (confidence_appropriateness - 0.5) * 0.3

        # Poor challenge resistance reduces confidence
        resistance_adjustment = (challenge_resistance - 0.6) * 0.15

        total_adjustment = (
            quality_adjustment + appropriateness_adjustment + resistance_adjustment
        )

        return max(-0.4, min(0.2, total_adjustment))

    def _detect_overconfidence_patterns(
        self,
        reasoning_steps: List[ReasoningStep],
        challenger_result: InternalChallengerResult,
    ) -> List[OverconfidencePattern]:
        """Detect overconfidence patterns in reasoning"""
        patterns = []

        # Systematic optimism - high confidence with many challenges
        avg_confidence = 0.0
        confidence_count = 0
        for step in reasoning_steps:
            if hasattr(step, "confidence_score") and step.confidence_score:
                avg_confidence += step.confidence_score
                confidence_count += 1

        if confidence_count > 0:
            avg_confidence /= confidence_count
            if avg_confidence > 0.8 and len(challenger_result.challenges) > 15:
                patterns.append(OverconfidencePattern.SYSTEMATIC_OPTIMISM)

        # Planning fallacy - underestimating complexity
        complexity_challenges = [
            c
            for c in challenger_result.challenges
            if "complex" in c.challenge_statement.lower()
            or "difficult" in c.challenge_statement.lower()
        ]
        if len(complexity_challenges) > 3:
            patterns.append(OverconfidencePattern.PLANNING_FALLACY)

        # Confirmation seeking - limited challenge resistance (inferred from few challenges for high confidence)
        avg_confidence = sum(
            getattr(step, "confidence_score", 0.5) for step in reasoning_steps
        ) / len(reasoning_steps)
        if avg_confidence > 0.8 and len(challenger_result.challenges) < 5:
            patterns.append(OverconfidencePattern.CONFIRMATION_SEEKING)

        return patterns

    def _record_calibration(
        self,
        original_confidence: float,
        calibrated_confidence: float,
        domain: CalibrationDomain,
        challenger_result: InternalChallengerResult,
        context_factors: Optional[Dict[str, Any]],
    ):
        """Record calibration for learning purposes"""
        self.confidence_history.append(
            (original_confidence, calibrated_confidence, domain.value)
        )

        # Update domain metrics
        domain_metrics = self.domain_metrics[domain]
        domain_metrics.total_challenges += 1
        domain_metrics.last_updated = datetime.now()

        # Calculate running averages (simplified for now)
        confidence_error = abs(original_confidence - calibrated_confidence)
        domain_metrics.average_confidence_error = (
            domain_metrics.average_confidence_error * 0.9 + confidence_error * 0.1
        )

    def update_challenge_outcome(self, outcome: ChallengeOutcome):
        """Update learning based on actual challenge outcome"""
        self.challenge_outcomes.append(outcome)

        # Update domain metrics based on outcome
        domain_metrics = self.domain_metrics[outcome.domain]
        domain_metrics.total_challenges += 1

        if outcome.outcome_accuracy >= 0.7:
            domain_metrics.accurate_challenges += 1

        # Update calibration score (simplified Brier score)
        domain_metrics.brier_score = self._calculate_brier_score(outcome.domain)
        domain_metrics.calibration_score = max(0.0, 1.0 - domain_metrics.brier_score)

        self.logger.info(f"📊 Challenge outcome updated for {outcome.domain.value}")
        self.logger.info(f"   Outcome accuracy: {outcome.outcome_accuracy:.3f}")
        self.logger.info(
            f"   Domain calibration score: {domain_metrics.calibration_score:.3f}"
        )

    def _calculate_brier_score(self, domain: CalibrationDomain) -> float:
        """Calculate Brier score for domain calibration"""
        domain_outcomes = [o for o in self.challenge_outcomes if o.domain == domain]

        if len(domain_outcomes) < 3:
            return 0.5  # Default moderate score

        brier_sum = 0.0
        for outcome in domain_outcomes[-10:]:  # Use last 10 outcomes
            predicted_prob = abs(outcome.confidence_adjustment_made)
            actual_outcome = 1.0 if outcome.outcome_accuracy >= 0.7 else 0.0
            brier_sum += (predicted_prob - actual_outcome) ** 2

        return brier_sum / min(len(domain_outcomes), 10)

    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get summary of calibration performance across all domains"""
        summary = {
            "total_calibrations": len(self.confidence_history),
            "domains": {},
            "overconfidence_patterns": dict(self.overconfidence_patterns),
            "recent_performance": {},
        }

        for domain, metrics in self.domain_metrics.items():
            if metrics.total_challenges > 0:
                summary["domains"][domain.value] = {
                    "total_challenges": metrics.total_challenges,
                    "accuracy_rate": metrics.accurate_challenges
                    / metrics.total_challenges,
                    "calibration_score": metrics.calibration_score,
                    "average_confidence_error": metrics.average_confidence_error,
                }

        # Recent performance (last 20 calibrations)
        if len(self.confidence_history) >= 10:
            recent_calibrations = self.confidence_history[-20:]
            recent_adjustments = [
                abs(orig - cal) for orig, cal, _ in recent_calibrations
            ]
            summary["recent_performance"] = {
                "average_adjustment": sum(recent_adjustments) / len(recent_adjustments),
                "calibration_frequency": len(recent_calibrations),
                "learning_trend": (
                    "improving"
                    if recent_adjustments[-5:] < recent_adjustments[:5]
                    else "stable"
                ),
            }

        return summary


class SelfDoubtCalibrationEngineFactory:
    """Factory for creating Self-Doubt Calibration Engine instances"""

    @staticmethod
    def create_calibration_engine(
        context_intelligence: Optional[IContextIntelligence] = None,
        calibration_level: ConfidenceCalibrationLevel = ConfidenceCalibrationLevel.ADAPTIVE,
        learning_rate: float = 0.1,
        logger: Optional[logging.Logger] = None,
    ) -> SelfDoubtCalibrationEngine:
        """Create a Self-Doubt Calibration Engine instance"""

        return SelfDoubtCalibrationEngine(
            context_intelligence=context_intelligence,
            calibration_level=calibration_level,
            learning_rate=learning_rate,
            logger=logger,
        )

    @staticmethod
    def create_bayesian_calibration_engine(
        context_intelligence: Optional[IContextIntelligence] = None,
        logger: Optional[logging.Logger] = None,
    ) -> SelfDoubtCalibrationEngine:
        """Create Bayesian calibration engine for sophisticated confidence updating"""

        return SelfDoubtCalibrationEngine(
            context_intelligence=context_intelligence,
            calibration_level=ConfidenceCalibrationLevel.BAYESIAN,
            learning_rate=0.05,  # Slower learning for Bayesian approach
            overconfidence_penalty=0.3,
            logger=logger,
        )

    @staticmethod
    def create_metacognitive_calibration_engine(
        context_intelligence: Optional[IContextIntelligence] = None,
        logger: Optional[logging.Logger] = None,
    ) -> SelfDoubtCalibrationEngine:
        """Create metacognitive calibration engine with self-awareness"""

        return SelfDoubtCalibrationEngine(
            context_intelligence=context_intelligence,
            calibration_level=ConfidenceCalibrationLevel.METACOGNITIVE,
            learning_rate=0.08,
            overconfidence_penalty=0.25,
            min_samples_for_learning=3,  # Faster learning
            logger=logger,
        )
