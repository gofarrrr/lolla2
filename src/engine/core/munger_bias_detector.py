#!/usr/bin/env python3
"""
Munger Bias Detector - Devils Advocate Engine #1
Implements Charlie Munger's approach to cognitive bias detection and lollapalooza effects
Part of the enhanced Devils Advocate system
"""

import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging


@dataclass
class CognitiveBias:
    """Individual cognitive bias detection result"""

    bias_type: str
    bias_name: str
    description: str
    severity: float  # 0.0-1.0
    evidence_examples: List[str]
    lollapalooza_potential: float  # Risk of multiple biases compounding
    mitigation_approach: str
    munger_principle: str  # Which Munger principle applies


@dataclass
class BiasDetectionResult:
    """Complete bias detection result from Munger approach"""

    situation_analyzed: str
    detected_biases: List[CognitiveBias]
    overall_bias_risk: float
    lollapalooza_effects: List[str]
    inversion_analysis: str
    recommended_mental_models: List[str]
    processing_time_ms: float
    confidence_score: float


class MungerBiasDetector:
    """
    Charlie Munger-inspired bias detection system

    Implements Munger's core principles:
    1. Inversion thinking ("What could go wrong?")
    2. Multiple mental models for cross-validation
    3. Lollapalooza effects detection (multiple biases combining)
    4. Systematic bias checklist approach
    5. Focus on most damaging psychological tendencies
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Munger's "Standard Psychological Tendencies" adapted for business decisions
        self.munger_bias_catalog = {
            "reward_punishment_super_response": {
                "name": "Reward/Punishment Super-Response Tendency",
                "indicators": [
                    "incentive",
                    "bonus",
                    "penalty",
                    "compensation",
                    "reward",
                ],
                "questions": [
                    "What are the incentive structures driving this recommendation?",
                    "Who benefits financially from this decision?",
                ],
            },
            "liking_loving_tendency": {
                "name": "Liking/Loving Tendency",
                "indicators": ["favorite", "prefer", "like", "admire", "trust"],
                "questions": [
                    "Are we favoring this because we like the people involved?",
                    "Is personal affinity clouding business judgment?",
                ],
            },
            "disliking_hating_tendency": {
                "name": "Disliking/Hating Tendency",
                "indicators": ["dislike", "oppose", "against", "reject", "resist"],
                "questions": [
                    "Are we rejecting this because we dislike the source?",
                    "Is personal animosity affecting business logic?",
                ],
            },
            "doubt_avoidance_tendency": {
                "name": "Doubt-Avoidance Tendency",
                "indicators": ["certain", "confident", "sure", "definite", "obvious"],
                "questions": [
                    "Are we being overly certain to avoid uncomfortable doubt?",
                    "What assumptions are we treating as facts?",
                ],
            },
            "inconsistency_avoidance_tendency": {
                "name": "Inconsistency-Avoidance Tendency",
                "indicators": [
                    "consistent",
                    "commitment",
                    "previous",
                    "always",
                    "policy",
                ],
                "questions": [
                    "Are we sticking to this just to be consistent?",
                    "Is past commitment preventing better decisions?",
                ],
            },
            "curiosity_tendency": {
                "name": "Curiosity Tendency",
                "indicators": ["interesting", "novel", "new", "innovative", "explore"],
                "questions": [
                    "Are we pursuing this just because it's interesting?",
                    "Is novelty bias affecting practical judgment?",
                ],
            },
            "kantian_fairness_tendency": {
                "name": "Kantian Fairness Tendency",
                "indicators": ["fair", "equal", "deserve", "rights", "justice"],
                "questions": [
                    "Is desire for fairness compromising optimal outcomes?",
                    "Are we prioritizing equality over effectiveness?",
                ],
            },
            "envy_jealousy_tendency": {
                "name": "Envy/Jealousy Tendency",
                "indicators": ["competitor", "rival", "better", "ahead", "success"],
                "questions": [
                    "Are we making this decision out of competitive envy?",
                    "Is jealousy of competitors driving poor choices?",
                ],
            },
            "reciprocation_tendency": {
                "name": "Reciprocation Tendency",
                "indicators": ["favor", "help", "return", "owe", "obligation"],
                "questions": [
                    "Are we reciprocating favors rather than optimizing?",
                    "Is sense of obligation clouding business judgment?",
                ],
            },
            "social_proof_tendency": {
                "name": "Social-Proof Tendency",
                "indicators": [
                    "others",
                    "industry",
                    "competitors",
                    "everyone",
                    "standard",
                ],
                "questions": [
                    "Are we following others instead of thinking independently?",
                    "Is 'everyone is doing it' driving our decision?",
                ],
            },
            "authority_misinfluence_tendency": {
                "name": "Authority-Misinfluence Tendency",
                "indicators": [
                    "expert",
                    "consultant",
                    "authority",
                    "leader",
                    "recommended",
                ],
                "questions": [
                    "Are we deferring to authority without independent analysis?",
                    "Is consultant advice substituting for our own thinking?",
                ],
            },
            "deprival_super_reaction_tendency": {
                "name": "Deprival Super-Reaction Tendency",
                "indicators": ["lose", "losing", "taken", "threat", "compete"],
                "questions": [
                    "Are we overreacting to potential losses?",
                    "Is loss aversion making us too defensive?",
                ],
            },
            "availability_misweighting_tendency": {
                "name": "Availability-Misweighting Tendency",
                "indicators": ["remember", "recent", "example", "heard", "story"],
                "questions": [
                    "Are we overweighting recent/memorable examples?",
                    "Is vivid anecdote substituting for systematic data?",
                ],
            },
        }

        # Mental models for cross-validation (Munger's lattice approach)
        self.validation_models = [
            "inversion",
            "circle_of_competence",
            "margin_of_safety",
            "opportunity_cost",
            "incentives",
            "systems_thinking",
        ]

    async def detect_bias_patterns(
        self, recommendation: str, business_context: Dict[str, Any]
    ) -> BiasDetectionResult:
        """Detect cognitive biases using Munger's systematic approach"""

        import time

        start_time = time.time()

        print("🧠 MUNGER BIAS DETECTION ENGINE")
        print("-" * 60)

        # Step 1: Inversion analysis - "What could go wrong?"
        inversion_analysis = self._perform_inversion_analysis(
            recommendation, business_context
        )

        # Step 2: Systematic bias detection using Munger catalog
        detected_biases = []
        for bias_key, bias_config in self.munger_bias_catalog.items():
            bias_result = self._detect_specific_bias(
                bias_key, bias_config, recommendation, business_context
            )
            if (
                bias_result and bias_result.severity >= 0.3
            ):  # Only report significant biases
                detected_biases.append(bias_result)

        # Step 3: Lollapalooza effects analysis (multiple biases combining)
        lollapalooza_effects = self._analyze_lollapalooza_effects(detected_biases)

        # Step 4: Calculate overall risk and confidence
        overall_bias_risk = self._calculate_overall_bias_risk(
            detected_biases, lollapalooza_effects
        )
        confidence_score = 1.0 - (overall_bias_risk * 0.8)  # High bias = low confidence

        # Step 5: Recommend mental models for validation
        recommended_models = self._recommend_validation_models(
            detected_biases, recommendation
        )

        processing_time = (time.time() - start_time) * 1000

        print("📊 Bias Detection Results:")
        print(f"├─ Biases detected: {len(detected_biases)}")
        print(f"├─ Overall risk: {overall_bias_risk:.3f}")
        print(f"├─ Lollapalooza effects: {len(lollapalooza_effects)}")
        print(f"└─ Processing time: {processing_time:.1f}ms")

        # Display top biases
        if detected_biases:
            print("\n🚨 Top Bias Concerns:")
            sorted_biases = sorted(
                detected_biases, key=lambda x: x.severity, reverse=True
            )
            for i, bias in enumerate(sorted_biases[:3], 1):
                print(f"{i}. {bias.bias_name} (severity: {bias.severity:.2f})")
                print(f"   → {bias.description}")

        return BiasDetectionResult(
            situation_analyzed=recommendation,
            detected_biases=detected_biases,
            overall_bias_risk=overall_bias_risk,
            lollapalooza_effects=lollapalooza_effects,
            inversion_analysis=inversion_analysis,
            recommended_mental_models=recommended_models,
            processing_time_ms=processing_time,
            confidence_score=confidence_score,
        )

    def _perform_inversion_analysis(
        self, recommendation: str, context: Dict[str, Any]
    ) -> str:
        """Perform Munger-style inversion analysis: 'What could go wrong?'"""

        # Identify key failure modes based on recommendation type
        failure_indicators = {
            "invest": [
                "market crash",
                "technology obsolete",
                "competition",
                "regulation",
            ],
            "acquire": [
                "integration failure",
                "culture clash",
                "overpaid",
                "hidden liabilities",
            ],
            "expand": [
                "demand overestimate",
                "execution issues",
                "resource strain",
                "market saturation",
            ],
            "pivot": [
                "customer confusion",
                "capability gap",
                "timing wrong",
                "competitive response",
            ],
            "cut": [
                "talent loss",
                "customer impact",
                "capability degradation",
                "competitor advantage",
            ],
        }

        relevant_failures = []
        rec_lower = recommendation.lower()

        for action, failures in failure_indicators.items():
            if action in rec_lower:
                relevant_failures.extend(failures)

        if not relevant_failures:
            relevant_failures = [
                "execution risk",
                "market risk",
                "financial risk",
                "competitive risk",
            ]

        inversion_analysis = "INVERSION ANALYSIS - What could go wrong:\n"
        for i, failure in enumerate(relevant_failures[:4], 1):
            inversion_analysis += (
                f"{i}. {failure.title()}: High probability given market conditions\n"
            )

        return inversion_analysis

    def _detect_specific_bias(
        self,
        bias_key: str,
        bias_config: Dict[str, Any],
        recommendation: str,
        context: Dict[str, Any],
    ) -> Optional[CognitiveBias]:
        """Detect specific cognitive bias using Munger framework"""

        # Check for linguistic indicators
        indicator_matches = 0
        evidence_examples = []

        rec_lower = recommendation.lower()
        for indicator in bias_config["indicators"]:
            if indicator in rec_lower:
                indicator_matches += 1
                evidence_examples.append(
                    f"Uses '{indicator}' suggesting {bias_config['name']}"
                )

        # Context-based detection
        contextual_evidence = self._analyze_contextual_bias_indicators(
            bias_key, recommendation, context
        )
        evidence_examples.extend(contextual_evidence)

        # Calculate severity based on indicators and context
        base_severity = min(0.8, indicator_matches * 0.2)
        contextual_boost = len(contextual_evidence) * 0.15
        total_severity = min(1.0, base_severity + contextual_boost)

        if total_severity < 0.3:  # Not significant enough to report
            return None

        # Assess lollapalooza potential (how this bias compounds with others)
        lollapalooza_potential = self._assess_lollapalooza_potential(
            bias_key, total_severity
        )

        # Generate Munger-inspired mitigation strategy
        mitigation_strategy = self._generate_munger_mitigation(bias_key, bias_config)

        return CognitiveBias(
            bias_type=bias_key,
            bias_name=bias_config["name"],
            description=f"Detected pattern matching {bias_config['name']} with {len(evidence_examples)} indicators",
            severity=total_severity,
            evidence_examples=evidence_examples[:3],  # Limit to top 3
            lollapalooza_potential=lollapalooza_potential,
            mitigation_approach=mitigation_strategy,
            munger_principle=f"Apply {bias_config['name']} awareness with systematic checklist",
        )

    def _analyze_contextual_bias_indicators(
        self, bias_key: str, recommendation: str, context: Dict[str, Any]
    ) -> List[str]:
        """Analyze context for bias indicators beyond linguistic patterns"""

        contextual_evidence = []

        # Stakeholder analysis for bias detection
        stakeholders = context.get("stakeholders", [])
        timeline_pressure = context.get("timeline_pressure", False)
        financial_constraints = context.get("financial_constraints")

        if bias_key == "doubt_avoidance_tendency" and timeline_pressure:
            contextual_evidence.append("Timeline pressure may increase certainty bias")

        if bias_key == "authority_misinfluence_tendency" and any(
            "consultant" in s.lower() for s in stakeholders
        ):
            contextual_evidence.append(
                "External consultant involvement may trigger authority bias"
            )

        if bias_key == "social_proof_tendency" and "industry" in str(context).lower():
            contextual_evidence.append(
                "Industry comparison suggests potential social proof bias"
            )

        if bias_key == "deprival_super_reaction_tendency" and financial_constraints:
            contextual_evidence.append(
                "Financial constraints may amplify loss aversion"
            )

        if bias_key == "reward_punishment_super_response" and "CEO" in stakeholders:
            contextual_evidence.append(
                "CEO involvement suggests potential incentive alignment issues"
            )

        return contextual_evidence

    def _analyze_lollapalooza_effects(
        self, detected_biases: List[CognitiveBias]
    ) -> List[str]:
        """Analyze potential lollapalooza effects (multiple biases combining destructively)"""

        lollapalooza_effects = []

        # Check for dangerous bias combinations
        bias_types = [bias.bias_type for bias in detected_biases]

        # Dangerous combination 1: Social proof + Authority
        if (
            "social_proof_tendency" in bias_types
            and "authority_misinfluence_tendency" in bias_types
        ):
            lollapalooza_effects.append(
                "Social proof + Authority bias creating dangerous conformity spiral"
            )

        # Dangerous combination 2: Doubt avoidance + Consistency
        if (
            "doubt_avoidance_tendency" in bias_types
            and "inconsistency_avoidance_tendency" in bias_types
        ):
            lollapalooza_effects.append(
                "Doubt avoidance + Consistency bias preventing course correction"
            )

        # Dangerous combination 3: Deprival reaction + Reciprocation
        if (
            "deprival_super_reaction_tendency" in bias_types
            and "reciprocation_tendency" in bias_types
        ):
            lollapalooza_effects.append(
                "Loss aversion + Reciprocation creating irrational commitments"
            )

        # Dangerous combination 4: Multiple high-severity biases
        high_severity_biases = [
            bias for bias in detected_biases if bias.severity >= 0.7
        ]
        if len(high_severity_biases) >= 3:
            lollapalooza_effects.append(
                f"Multiple high-severity biases ({len(high_severity_biases)}) creating compound risk"
            )

        return lollapalooza_effects

    def _assess_lollapalooza_potential(self, bias_key: str, severity: float) -> float:
        """Assess how likely this bias is to combine destructively with others"""

        # Biases that commonly compound with others
        high_compound_biases = [
            "social_proof_tendency",
            "authority_misinfluence_tendency",
            "doubt_avoidance_tendency",
            "inconsistency_avoidance_tendency",
        ]

        # Biases that amplify emotional decision making
        emotional_amplifiers = [
            "deprival_super_reaction_tendency",
            "envy_jealousy_tendency",
            "liking_loving_tendency",
            "disliking_hating_tendency",
        ]

        base_potential = severity * 0.5  # Base compound potential

        if bias_key in high_compound_biases:
            base_potential += 0.3

        if bias_key in emotional_amplifiers:
            base_potential += 0.2

        return min(1.0, base_potential)

    def _generate_munger_mitigation(
        self, bias_key: str, bias_config: Dict[str, Any]
    ) -> str:
        """Generate Munger-style mitigation strategy"""

        mitigation_strategies = {
            "reward_punishment_super_response": "Explicitly map all stakeholder incentives and misalignments",
            "social_proof_tendency": "Seek independent analysis before considering industry practices",
            "authority_misinfluence_tendency": "Challenge expert recommendations with independent thinking",
            "doubt_avoidance_tendency": "Force explicit consideration of uncertainty and alternatives",
            "deprival_super_reaction_tendency": "Use systematic cost-benefit analysis to override loss emotions",
            "availability_misweighting_tendency": "Demand comprehensive data over vivid anecdotes",
        }

        specific_strategy = mitigation_strategies.get(
            bias_key, "Apply systematic checklist approach to counteract bias"
        )

        # Add Munger's questions
        questions = bias_config.get("questions", [])
        if questions:
            specific_strategy += f" Ask: {questions[0]}"

        return specific_strategy

    def _calculate_overall_bias_risk(
        self, detected_biases: List[CognitiveBias], lollapalooza_effects: List[str]
    ) -> float:
        """Calculate overall bias risk using Munger principles"""

        if not detected_biases:
            return 0.0

        # Base risk from individual biases
        individual_risk = sum(bias.severity for bias in detected_biases) / len(
            detected_biases
        )

        # Lollapalooza multiplier effect
        lollapalooza_multiplier = 1.0 + (len(lollapalooza_effects) * 0.3)

        # High-severity bias penalty (Munger: focus on most damaging tendencies)
        critical_biases = [bias for bias in detected_biases if bias.severity >= 0.8]
        critical_penalty = len(critical_biases) * 0.2

        total_risk = (individual_risk * lollapalooza_multiplier) + critical_penalty

        return min(1.0, total_risk)

    def _recommend_validation_models(
        self, detected_biases: List[CognitiveBias], recommendation: str
    ) -> List[str]:
        """Recommend mental models for bias validation using Munger lattice approach"""

        # Base recommendations
        base_models = ["inversion", "opportunity_cost"]

        # Bias-specific model recommendations
        bias_model_mapping = {
            "social_proof_tendency": ["independent_thinking", "contrarian_analysis"],
            "authority_misinfluence_tendency": [
                "first_principles",
                "critical_thinking",
            ],
            "deprival_super_reaction_tendency": ["expected_value", "base_rates"],
            "doubt_avoidance_tendency": ["scenario_analysis", "pre_mortem"],
        }

        recommended = base_models.copy()

        for bias in detected_biases:
            if bias.bias_type in bias_model_mapping:
                recommended.extend(bias_model_mapping[bias.bias_type])

        # Remove duplicates and limit to practical number
        return list(set(recommended))[:6]


async def demonstrate_munger_bias_detection():
    """Demonstrate Munger bias detection system"""

    detector = MungerBiasDetector()

    test_cases = [
        {
            "recommendation": "Everyone in the industry is investing in AI, so we should immediately allocate $10M to AI initiatives to stay competitive",
            "context": {
                "stakeholders": ["CEO", "Board", "Consultant"],
                "timeline_pressure": True,
                "industry": "Traditional Retail",
            },
        },
        {
            "recommendation": "Our trusted advisor recommends acquiring TechStartup Inc for $50M - they have a proven track record and this is definitely the right move",
            "context": {
                "stakeholders": ["CEO", "External Consultant", "Investment Banker"],
                "stated_preferences": "Want growth through acquisition",
            },
        },
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*20} MUNGER BIAS TEST {i} {'='*20}")

        result = await detector.detect_bias_patterns(
            case["recommendation"], case["context"]
        )

        print("\n🎯 MUNGER ANALYSIS RESULTS:")
        print(f"Overall Bias Risk: {result.overall_bias_risk:.3f}")
        print(f"Confidence Score: {result.confidence_score:.3f}")
        print(f"Lollapalooza Effects: {len(result.lollapalooza_effects)}")

        print("\n🔍 INVERSION ANALYSIS:")
        print(result.inversion_analysis)

        if result.lollapalooza_effects:
            print("\n⚠️  LOLLAPALOOZA EFFECTS:")
            for effect in result.lollapalooza_effects:
                print(f"• {effect}")

        if i < len(test_cases):
            print(f"\n{'NEXT TEST':=^80}\n")


if __name__ == "__main__":
    asyncio.run(demonstrate_munger_bias_detection())
