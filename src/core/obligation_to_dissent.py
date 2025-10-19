"""
Obligation to Dissent (O2D) Challenge System
============================================

Implements McKinsey's "obligation to dissent" framework for systematic challenge generation.
Based on elite consulting practices from McKinsey, BCG, and Bain for rigorous analysis validation.

Core Philosophy:
- Junior members OBLIGATED to speak up when disagreeing
- "What would you have to believe?" as primary challenge tool
- Constructive confrontation without being disagreeable
- Pre-mortems and red-team challenges to surface risks
- Psychological safety through dialectic standards

Author: METIS V5.3 Platform
Integration: NWAY_ELITE_CONSULTING_FRAMEWORKS_001
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ChallengeType(Enum):
    """Types of systematic challenges in O2D framework"""

    WHAT_WOULD_YOU_BELIEVE = "what_would_you_believe"
    PRE_MORTEM = "pre_mortem"
    RED_TEAM_BLUE_TEAM = "red_team_blue_team"
    DIALECTIC_SYNTHESIS = "dialectic_synthesis"
    ASSUMPTION_SURFACING = "assumption_surfacing"
    STAKEHOLDER_PERSPECTIVE = "stakeholder_perspective"


class DissentTier(Enum):
    """Tier levels for dissent depth based on stakes"""

    ROUTINE = "routine"  # Quick assumption check
    SIGNIFICANT = "significant"  # Systematic challenge
    CRITICAL = "critical"  # Full red-team analysis


@dataclass
class AssumptionAnalysis:
    """Analysis of underlying assumptions in reasoning"""

    assumption_statement: str
    implicit_level: float  # 0.0 = explicit, 1.0 = deeply implicit
    evidence_strength: float  # 0.0 = no evidence, 1.0 = strong evidence
    failure_impact: float  # 0.0 = low impact, 1.0 = critical failure
    testability: float  # 0.0 = untestable, 1.0 = easily testable
    alternative_scenarios: List[str] = field(default_factory=list)
    testing_methods: List[str] = field(default_factory=list)


@dataclass
class WhatWouldYouBelieveChallenge:
    """Core McKinsey challenge: 'What would you have to believe for X to be true?'"""

    target_conclusion: str
    required_beliefs: List[str]
    belief_strength_required: Dict[str, float]  # Belief → confidence needed
    evidence_gaps: List[str]
    logical_leaps: List[str]
    unstated_assumptions: List[AssumptionAnalysis]
    confidence_threshold: float = 0.7


@dataclass
class PreMortemScenario:
    """Pre-mortem analysis: imagine failure and work backwards"""

    failure_scenario: str
    failure_probability: float
    warning_signals: List[str]
    prevention_mechanisms: List[str]
    contingency_plans: List[str]
    early_detection_metrics: List[str]


@dataclass
class RedTeamChallenge:
    """Red team vs blue team competitive challenge generation"""

    position_under_attack: str
    red_team_arguments: List[str]
    blue_team_counters: List[str]
    synthesis_position: Optional[str] = None
    winning_arguments: List[str] = field(default_factory=list)
    knowledge_gaps_exposed: List[str] = field(default_factory=list)


@dataclass
class StakeholderPerspectiveChallenge:
    """Challenge from different stakeholder viewpoints"""

    stakeholder_type: str
    stakeholder_objectives: List[str]
    perspective_on_analysis: str
    concerns_raised: List[str]
    alternative_solutions: List[str]
    trust_factors: Dict[str, float]  # Factor → trust level


@dataclass
class O2DChallenge:
    """Complete Obligation to Dissent challenge result"""

    challenge_type: ChallengeType
    challenge_tier: DissentTier
    challenge_summary: str

    # Core challenge components
    what_would_you_believe: Optional[WhatWouldYouBelieveChallenge] = None
    pre_mortem: Optional[PreMortemScenario] = None
    red_team: Optional[RedTeamChallenge] = None
    stakeholder_perspective: Optional[StakeholderPerspectiveChallenge] = None

    # Meta-analysis
    psychological_safety_score: float = 0.9  # Ensure constructive vs destructive
    solution_orientation: float = 0.8  # Focus on improvement not criticism
    forward_momentum: float = 0.8  # Generate action not paralysis

    # Integration
    implementation_difficulty: float = 0.5
    integration_points: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)


@dataclass
class O2DSystemConfig:
    """Configuration for Obligation to Dissent system"""

    default_tier: DissentTier = DissentTier.SIGNIFICANT
    psychological_safety_threshold: float = 0.8
    solution_orientation_threshold: float = 0.7
    enable_stakeholder_rotation: bool = True
    max_challenges_per_session: int = 5
    dialectic_synthesis_enabled: bool = True

    # Challenge weightings
    challenge_type_weights: Dict[ChallengeType, float] = field(
        default_factory=lambda: {
            ChallengeType.WHAT_WOULD_YOU_BELIEVE: 1.0,
            ChallengeType.PRE_MORTEM: 0.8,
            ChallengeType.RED_TEAM_BLUE_TEAM: 0.7,
            ChallengeType.STAKEHOLDER_PERSPECTIVE: 0.6,
            ChallengeType.ASSUMPTION_SURFACING: 0.9,
            ChallengeType.DIALECTIC_SYNTHESIS: 0.5,
        }
    )


class ObligationToDissentSystem:
    """
    McKinsey Obligation to Dissent (O2D) Challenge System

    Implements systematic dissent framework with psychological safety safeguards.
    Based on elite consulting practices for rigorous analysis validation.
    """

    def __init__(self, config: Optional[O2DSystemConfig] = None):
        self.config = config or O2DSystemConfig()
        self.challenge_history: List[O2DChallenge] = []

        # Stakeholder personas for perspective challenges
        self.stakeholder_personas = {
            "ceo": {
                "objectives": [
                    "shareholder value",
                    "strategic positioning",
                    "risk management",
                ],
                "concerns": [
                    "execution risk",
                    "resource allocation",
                    "competitive response",
                ],
                "trust_factors": {
                    "financial_impact": 1.0,
                    "strategic_fit": 0.9,
                    "execution_risk": 0.8,
                },
            },
            "cfo": {
                "objectives": [
                    "financial performance",
                    "cost management",
                    "compliance",
                ],
                "concerns": ["budget overruns", "ROI uncertainty", "cash flow impact"],
                "trust_factors": {
                    "financial_modeling": 1.0,
                    "cost_accuracy": 0.9,
                    "risk_quantification": 0.8,
                },
            },
            "customer": {
                "objectives": ["value delivery", "user experience", "problem solving"],
                "concerns": ["disruption", "learning curve", "value perception"],
                "trust_factors": {
                    "user_benefit": 1.0,
                    "ease_of_use": 0.9,
                    "reliability": 0.8,
                },
            },
            "competitor": {
                "objectives": [
                    "market share",
                    "competitive advantage",
                    "customer acquisition",
                ],
                "concerns": [
                    "losing ground",
                    "resource allocation",
                    "strategic response",
                ],
                "trust_factors": {
                    "competitive_intelligence": 0.8,
                    "market_dynamics": 0.7,
                    "response_speed": 0.9,
                },
            },
            "regulator": {
                "objectives": ["compliance", "public interest", "risk mitigation"],
                "concerns": [
                    "unintended consequences",
                    "precedent setting",
                    "enforcement",
                ],
                "trust_factors": {
                    "regulatory_compliance": 1.0,
                    "public_benefit": 0.9,
                    "risk_assessment": 0.8,
                },
            },
        }

    async def generate_systematic_dissent(
        self,
        analysis: str,
        context: Dict[str, Any],
        tier: Optional[DissentTier] = None,
        challenge_types: Optional[List[ChallengeType]] = None,
    ) -> List[O2DChallenge]:
        """
        Generate systematic dissent challenges for analysis validation

        Args:
            analysis: Analysis content to challenge
            context: Analysis context and background
            tier: Dissent tier level (determines depth)
            challenge_types: Specific challenge types to generate

        Returns:
            List of O2D challenges with systematic dissent
        """
        tier = tier or self.config.default_tier
        challenge_types = challenge_types or self._select_challenge_types(tier, context)

        challenges = []

        for challenge_type in challenge_types:
            if len(challenges) >= self.config.max_challenges_per_session:
                break

            challenge = await self._generate_challenge_by_type(
                challenge_type, analysis, context, tier
            )

            if challenge and self._validate_psychological_safety(challenge):
                challenges.append(challenge)

        # Apply dialectic synthesis if enabled and multiple challenges exist
        if (
            self.config.dialectic_synthesis_enabled
            and len(challenges) >= 2
            and ChallengeType.DIALECTIC_SYNTHESIS in challenge_types
        ):

            synthesis_challenge = await self._generate_dialectic_synthesis(
                challenges, analysis, context
            )
            if synthesis_challenge:
                challenges.append(synthesis_challenge)

        self.challenge_history.extend(challenges)
        return challenges

    async def _generate_challenge_by_type(
        self,
        challenge_type: ChallengeType,
        analysis: str,
        context: Dict[str, Any],
        tier: DissentTier,
    ) -> Optional[O2DChallenge]:
        """Generate specific type of challenge"""

        if challenge_type == ChallengeType.WHAT_WOULD_YOU_BELIEVE:
            return await self._generate_what_would_you_believe(analysis, context, tier)
        elif challenge_type == ChallengeType.PRE_MORTEM:
            return await self._generate_pre_mortem(analysis, context, tier)
        elif challenge_type == ChallengeType.RED_TEAM_BLUE_TEAM:
            return await self._generate_red_team_challenge(analysis, context, tier)
        elif challenge_type == ChallengeType.STAKEHOLDER_PERSPECTIVE:
            return await self._generate_stakeholder_perspective(analysis, context, tier)
        elif challenge_type == ChallengeType.ASSUMPTION_SURFACING:
            return await self._generate_assumption_surfacing(analysis, context, tier)
        else:
            return None

    async def _generate_what_would_you_believe(
        self, analysis: str, context: Dict[str, Any], tier: DissentTier
    ) -> O2DChallenge:
        """Generate 'What would you have to believe?' challenge"""

        # Extract key conclusions from analysis
        key_conclusions = self._extract_key_conclusions(analysis)

        # For each conclusion, determine required beliefs
        required_beliefs = []
        belief_strengths = {}
        evidence_gaps = []
        logical_leaps = []
        unstated_assumptions = []

        for conclusion in key_conclusions[:3]:  # Focus on top 3 conclusions
            # What beliefs are required for this conclusion?
            beliefs = self._analyze_required_beliefs(conclusion, analysis)
            required_beliefs.extend(beliefs)

            # How strong must these beliefs be?
            for belief in beliefs:
                strength = self._assess_belief_strength_requirement(belief, conclusion)
                belief_strengths[belief] = strength

                # Are there evidence gaps?
                gaps = self._identify_evidence_gaps(belief, analysis)
                evidence_gaps.extend(gaps)

                # Are there logical leaps?
                leaps = self._identify_logical_leaps(belief, conclusion)
                logical_leaps.extend(leaps)

            # Surface unstated assumptions
            assumptions = self._surface_unstated_assumptions(conclusion, analysis)
            unstated_assumptions.extend(assumptions)

        what_would_you_believe = WhatWouldYouBelieveChallenge(
            target_conclusion="; ".join(key_conclusions),
            required_beliefs=list(set(required_beliefs)),
            belief_strength_required=belief_strengths,
            evidence_gaps=list(set(evidence_gaps)),
            logical_leaps=list(set(logical_leaps)),
            unstated_assumptions=unstated_assumptions,
            confidence_threshold=0.8 if tier == DissentTier.CRITICAL else 0.7,
        )

        challenge_summary = (
            f"To accept this analysis, you would need to believe: "
            f"{', '.join(required_beliefs[:3])}{'...' if len(required_beliefs) > 3 else ''}. "
            f"Key evidence gaps: {', '.join(evidence_gaps[:2])}. "
            f"Are these beliefs sufficiently supported?"
        )

        return O2DChallenge(
            challenge_type=ChallengeType.WHAT_WOULD_YOU_BELIEVE,
            challenge_tier=tier,
            challenge_summary=challenge_summary,
            what_would_you_believe=what_would_you_believe,
            psychological_safety_score=0.9,
            solution_orientation=0.8,
            forward_momentum=0.8,
            next_steps=[
                "Gather additional evidence for key beliefs",
                "Test unstated assumptions",
                "Validate logical connections",
            ],
        )

    async def _generate_pre_mortem(
        self, analysis: str, context: Dict[str, Any], tier: DissentTier
    ) -> O2DChallenge:
        """Generate pre-mortem scenario challenge"""

        # Imagine the analysis/recommendation fails
        failure_scenarios = self._generate_failure_scenarios(analysis, context)

        # Select most likely/impactful failure
        primary_failure = max(
            failure_scenarios,
            key=lambda x: x.get("probability", 0) * x.get("impact", 0),
        )

        # Work backwards from failure
        warning_signals = self._identify_warning_signals(primary_failure, context)
        prevention_mechanisms = self._design_prevention_mechanisms(
            primary_failure, analysis
        )
        contingency_plans = self._develop_contingency_plans(primary_failure, context)
        early_detection = self._design_early_detection_metrics(primary_failure, context)

        pre_mortem = PreMortemScenario(
            failure_scenario=primary_failure.get(
                "description", "Analysis assumptions prove incorrect"
            ),
            failure_probability=primary_failure.get("probability", 0.3),
            warning_signals=warning_signals,
            prevention_mechanisms=prevention_mechanisms,
            contingency_plans=contingency_plans,
            early_detection_metrics=early_detection,
        )

        challenge_summary = (
            f"Pre-mortem: If this analysis fails, it's likely because {primary_failure.get('description')}. "
            f"Warning signals would include: {', '.join(warning_signals[:2])}. "
            f"How can we prevent this failure mode?"
        )

        return O2DChallenge(
            challenge_type=ChallengeType.PRE_MORTEM,
            challenge_tier=tier,
            challenge_summary=challenge_summary,
            pre_mortem=pre_mortem,
            psychological_safety_score=0.9,
            solution_orientation=0.9,  # Pre-mortems are inherently solution-oriented
            forward_momentum=0.8,
            next_steps=[
                "Implement early warning systems",
                "Develop contingency protocols",
                "Monitor prevention mechanisms",
            ],
        )

    async def _generate_red_team_challenge(
        self, analysis: str, context: Dict[str, Any], tier: DissentTier
    ) -> O2DChallenge:
        """Generate red team vs blue team challenge"""

        # Identify the core position to attack
        core_position = self._extract_core_position(analysis)

        # Generate red team arguments (attacking the position)
        red_team_args = self._generate_attack_arguments(
            core_position, analysis, context
        )

        # Generate blue team counters (defending the position)
        blue_team_counters = self._generate_defense_arguments(
            core_position, red_team_args, analysis
        )

        # Identify knowledge gaps exposed by the debate
        knowledge_gaps = self._identify_knowledge_gaps_from_debate(
            red_team_args, blue_team_counters
        )

        # Attempt synthesis if possible
        synthesis_position = self._attempt_synthesis(
            red_team_args, blue_team_counters, core_position
        )

        red_team = RedTeamChallenge(
            position_under_attack=core_position,
            red_team_arguments=red_team_args,
            blue_team_counters=blue_team_counters,
            synthesis_position=synthesis_position,
            knowledge_gaps_exposed=knowledge_gaps,
        )

        challenge_summary = (
            f"Red team challenge: {red_team_args[0] if red_team_args else 'Core position questioned'}. "
            f"Blue team response: {blue_team_counters[0] if blue_team_counters else 'Position defended'}. "
            f"Synthesis needed on: {synthesis_position or 'fundamental approach'}."
        )

        return O2DChallenge(
            challenge_type=ChallengeType.RED_TEAM_BLUE_TEAM,
            challenge_tier=tier,
            challenge_summary=challenge_summary,
            red_team=red_team,
            psychological_safety_score=0.8,  # Competitive but safe
            solution_orientation=0.7,
            forward_momentum=0.8,
            next_steps=[
                "Address knowledge gaps identified",
                "Refine position based on strongest arguments",
                "Test synthesis position",
            ],
        )

    async def _generate_stakeholder_perspective(
        self, analysis: str, context: Dict[str, Any], tier: DissentTier
    ) -> O2DChallenge:
        """Generate stakeholder perspective challenge"""

        # Select stakeholder type based on context
        stakeholder_type = self._select_relevant_stakeholder(context)
        stakeholder_config = self.stakeholder_personas.get(
            stakeholder_type, self.stakeholder_personas["ceo"]
        )

        # Generate perspective from this stakeholder's viewpoint
        perspective = self._generate_stakeholder_perspective_analysis(
            analysis, stakeholder_config, stakeholder_type
        )
        concerns = self._generate_stakeholder_concerns(
            analysis, stakeholder_config, stakeholder_type
        )
        alternatives = self._generate_stakeholder_alternatives(
            analysis, stakeholder_config, stakeholder_type
        )

        stakeholder_perspective = StakeholderPerspectiveChallenge(
            stakeholder_type=stakeholder_type,
            stakeholder_objectives=stakeholder_config["objectives"],
            perspective_on_analysis=perspective,
            concerns_raised=concerns,
            alternative_solutions=alternatives,
            trust_factors=stakeholder_config["trust_factors"],
        )

        challenge_summary = (
            f"From {stakeholder_type} perspective: {perspective}. "
            f"Key concerns: {', '.join(concerns[:2])}. "
            f"How does this analysis serve their objectives?"
        )

        return O2DChallenge(
            challenge_type=ChallengeType.STAKEHOLDER_PERSPECTIVE,
            challenge_tier=tier,
            challenge_summary=challenge_summary,
            stakeholder_perspective=stakeholder_perspective,
            psychological_safety_score=0.9,  # Perspective-taking is safe
            solution_orientation=0.8,
            forward_momentum=0.8,
            next_steps=[
                f"Address {stakeholder_type} concerns",
                "Refine analysis for stakeholder value",
                "Consider alternative solutions",
            ],
        )

    async def _generate_assumption_surfacing(
        self, analysis: str, context: Dict[str, Any], tier: DissentTier
    ) -> O2DChallenge:
        """Generate assumption surfacing challenge"""

        # Surface implicit assumptions
        assumptions = self._surface_all_assumptions(analysis, context)

        # Analyze each assumption
        assumption_analyses = []
        for assumption in assumptions[:5]:  # Focus on top 5
            analysis_obj = AssumptionAnalysis(
                assumption_statement=assumption,
                implicit_level=self._assess_implicit_level(assumption, analysis),
                evidence_strength=self._assess_evidence_strength(assumption, analysis),
                failure_impact=self._assess_failure_impact(assumption, analysis),
                testability=self._assess_testability(assumption, context),
                alternative_scenarios=self._generate_alternative_scenarios(assumption),
                testing_methods=self._suggest_testing_methods(assumption, context),
            )
            assumption_analyses.append(analysis_obj)

        # Sort by risk (implicit_level * failure_impact / evidence_strength)
        risky_assumptions = sorted(
            assumption_analyses,
            key=lambda a: (a.implicit_level * a.failure_impact)
            / max(a.evidence_strength, 0.1),
            reverse=True,
        )

        challenge_summary = (
            f"Critical unstated assumptions: {risky_assumptions[0].assumption_statement if risky_assumptions else 'Multiple identified'}. "
            f"Highest risk assumption has {risky_assumptions[0].evidence_strength:.1f} evidence strength "
            f"but {risky_assumptions[0].failure_impact:.1f} failure impact. "
            f"How can we test these assumptions?"
        )

        return O2DChallenge(
            challenge_type=ChallengeType.ASSUMPTION_SURFACING,
            challenge_tier=tier,
            challenge_summary=challenge_summary,
            psychological_safety_score=0.9,
            solution_orientation=0.9,  # Focus on testing, not criticism
            forward_momentum=0.9,
            next_steps=[
                "Test highest-risk assumptions",
                "Gather evidence for critical beliefs",
                "Develop assumption validation protocols",
            ],
            integration_points=[
                f"Assumption testing: {len(assumption_analyses)} identified",
                f"High-risk assumptions: {len([a for a in risky_assumptions if a.failure_impact > 0.7])}",
                f"Testable assumptions: {len([a for a in assumption_analyses if a.testability > 0.6])}",
            ],
        )

    async def _generate_dialectic_synthesis(
        self,
        existing_challenges: List[O2DChallenge],
        analysis: str,
        context: Dict[str, Any],
    ) -> Optional[O2DChallenge]:
        """Generate dialectic synthesis from multiple challenges"""

        if len(existing_challenges) < 2:
            return None

        # Extract thesis (original analysis position)
        thesis = self._extract_thesis_position(analysis)

        # Extract antithesis (challenge positions)
        antitheses = []
        for challenge in existing_challenges:
            antithesis = self._extract_antithesis_from_challenge(challenge)
            if antithesis:
                antitheses.append(antithesis)

        # Generate synthesis (higher-level resolution)
        synthesis = self._generate_synthesis_position(
            thesis, antitheses, analysis, context
        )

        challenge_summary = (
            f"Dialectic synthesis: Thesis '{thesis}' + Antitheses '{'; '.join(antitheses[:2])}' "
            f"→ Synthesis '{synthesis}'. How can we integrate opposing perspectives?"
        )

        return O2DChallenge(
            challenge_type=ChallengeType.DIALECTIC_SYNTHESIS,
            challenge_tier=DissentTier.SIGNIFICANT,
            challenge_summary=challenge_summary,
            psychological_safety_score=0.95,  # Synthesis is maximally safe
            solution_orientation=0.95,
            forward_momentum=0.9,
            next_steps=[
                "Implement synthesis approach",
                "Test integrated solution",
                "Monitor for thesis-antithesis tensions",
            ],
        )

    def _select_challenge_types(
        self, tier: DissentTier, context: Dict[str, Any]
    ) -> List[ChallengeType]:
        """Select appropriate challenge types based on tier and context"""

        base_challenges = [ChallengeType.WHAT_WOULD_YOU_BELIEVE]

        if tier == DissentTier.ROUTINE:
            return base_challenges + [ChallengeType.ASSUMPTION_SURFACING]
        elif tier == DissentTier.SIGNIFICANT:
            return base_challenges + [
                ChallengeType.PRE_MORTEM,
                ChallengeType.STAKEHOLDER_PERSPECTIVE,
                ChallengeType.ASSUMPTION_SURFACING,
            ]
        else:  # CRITICAL
            return [
                ChallengeType.WHAT_WOULD_YOU_BELIEVE,
                ChallengeType.PRE_MORTEM,
                ChallengeType.RED_TEAM_BLUE_TEAM,
                ChallengeType.STAKEHOLDER_PERSPECTIVE,
                ChallengeType.ASSUMPTION_SURFACING,
                ChallengeType.DIALECTIC_SYNTHESIS,
            ]

    def _validate_psychological_safety(self, challenge: O2DChallenge) -> bool:
        """Ensure challenge maintains psychological safety standards"""
        return (
            challenge.psychological_safety_score
            >= self.config.psychological_safety_threshold
            and challenge.solution_orientation
            >= self.config.solution_orientation_threshold
        )

    # Utility methods for analysis (simplified for brevity)
    def _extract_key_conclusions(self, analysis: str) -> List[str]:
        """Extract key conclusions from analysis text"""
        # Simplified implementation - would use NLP in production
        sentences = analysis.split(".")
        conclusions = [
            s.strip()
            for s in sentences
            if any(
                word in s.lower()
                for word in [
                    "therefore",
                    "thus",
                    "conclude",
                    "recommend",
                    "suggest",
                    "should",
                ]
            )
        ]
        return conclusions[:5]

    def _analyze_required_beliefs(self, conclusion: str, analysis: str) -> List[str]:
        """Analyze what beliefs are required for conclusion to be true"""
        # Simplified - would use logical reasoning in production
        return [
            "The data accurately represents the underlying reality",
            "The assumptions made in the analysis are valid",
            "The reasoning chain from evidence to conclusion is sound",
            "External factors will remain relatively stable",
        ]

    def _assess_belief_strength_requirement(
        self, belief: str, conclusion: str
    ) -> float:
        """Assess how strongly the belief must be held for conclusion"""
        # Simplified scoring
        if "accurately represents" in belief or "sound" in belief:
            return 0.8
        return 0.6

    def _identify_evidence_gaps(self, belief: str, analysis: str) -> List[str]:
        """Identify gaps in evidence supporting belief"""
        return [
            "Limited sample size in data collection",
            "Potential selection bias in evidence",
            "Missing validation from independent sources",
        ]

    def _identify_logical_leaps(self, belief: str, conclusion: str) -> List[str]:
        """Identify logical leaps between belief and conclusion"""
        return [
            "Correlation assumed to imply causation",
            "Historical patterns assumed to continue",
            "Single data point generalized broadly",
        ]

    def _surface_unstated_assumptions(
        self, conclusion: str, analysis: str
    ) -> List[AssumptionAnalysis]:
        """Surface unstated assumptions in reasoning"""
        return [
            AssumptionAnalysis(
                assumption_statement="Market conditions will remain stable",
                implicit_level=0.8,
                evidence_strength=0.3,
                failure_impact=0.7,
                testability=0.6,
                alternative_scenarios=[
                    "Market volatility increases",
                    "Regulatory changes occur",
                ],
                testing_methods=["Scenario modeling", "Sensitivity analysis"],
            )
        ]

    def _generate_failure_scenarios(
        self, analysis: str, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate potential failure scenarios"""
        return [
            {
                "description": "Key assumptions prove incorrect under stress",
                "probability": 0.4,
                "impact": 0.8,
            },
            {
                "description": "External conditions change faster than anticipated",
                "probability": 0.3,
                "impact": 0.7,
            },
            {
                "description": "Implementation challenges exceed expectations",
                "probability": 0.5,
                "impact": 0.6,
            },
        ]

    def _identify_warning_signals(
        self, failure_scenario: Dict[str, Any], context: Dict[str, Any]
    ) -> List[str]:
        """Identify early warning signals for failure scenario"""
        return [
            "Key metrics trending away from predictions",
            "Stakeholder resistance higher than expected",
            "Resource constraints emerging",
            "External environment showing volatility",
        ]

    def _design_prevention_mechanisms(
        self, failure_scenario: Dict[str, Any], analysis: str
    ) -> List[str]:
        """Design mechanisms to prevent failure scenario"""
        return [
            "Regular assumption validation checkpoints",
            "Continuous environmental monitoring",
            "Flexible implementation approach",
            "Early pivot mechanisms",
        ]

    def _develop_contingency_plans(
        self, failure_scenario: Dict[str, Any], context: Dict[str, Any]
    ) -> List[str]:
        """Develop contingency plans for failure scenario"""
        return [
            "Alternative approach ready for deployment",
            "Resource reallocation protocols",
            "Stakeholder communication plan",
            "Graceful degradation strategy",
        ]

    def _design_early_detection_metrics(
        self, failure_scenario: Dict[str, Any], context: Dict[str, Any]
    ) -> List[str]:
        """Design metrics for early detection of failure"""
        return [
            "Leading indicator tracking dashboard",
            "Assumption validity scores",
            "Environmental stability index",
            "Implementation progress metrics",
        ]

    def _extract_core_position(self, analysis: str) -> str:
        """Extract the core position from analysis"""
        # Simplified - would use NLP to find main thesis
        return "The analysis recommends proceeding with the proposed approach based on current evidence"

    def _generate_attack_arguments(
        self, position: str, analysis: str, context: Dict[str, Any]
    ) -> List[str]:
        """Generate red team attack arguments"""
        return [
            "Evidence base is insufficient for confident decision",
            "Alternative explanations not adequately considered",
            "Implementation risks are underestimated",
            "Cost-benefit analysis may be overly optimistic",
        ]

    def _generate_defense_arguments(
        self, position: str, attacks: List[str], analysis: str
    ) -> List[str]:
        """Generate blue team defense arguments"""
        return [
            "Evidence meets standard threshold for business decisions",
            "Alternative explanations were evaluated and found less likely",
            "Risk mitigation strategies are included in the plan",
            "Conservative assumptions used in cost-benefit analysis",
        ]

    def _identify_knowledge_gaps_from_debate(
        self, attacks: List[str], defenses: List[str]
    ) -> List[str]:
        """Identify knowledge gaps exposed by red/blue team debate"""
        return [
            "Need for additional validation data",
            "Requirement for risk quantification",
            "Gap in alternative solution analysis",
            "Missing competitive response modeling",
        ]

    def _attempt_synthesis(
        self, attacks: List[str], defenses: List[str], position: str
    ) -> Optional[str]:
        """Attempt to synthesize red and blue team positions"""
        return "Proceed with enhanced validation and risk monitoring protocols"

    def _select_relevant_stakeholder(self, context: Dict[str, Any]) -> str:
        """Select most relevant stakeholder for perspective challenge"""
        # Simplified selection logic
        if "financial" in str(context).lower():
            return "cfo"
        elif "customer" in str(context).lower():
            return "customer"
        elif "regulatory" in str(context).lower():
            return "regulator"
        else:
            return "ceo"

    def _generate_stakeholder_perspective_analysis(
        self, analysis: str, config: Dict[str, Any], stakeholder_type: str
    ) -> str:
        """Generate analysis from stakeholder perspective"""
        objectives = config.get("objectives", [])
        return f"From {stakeholder_type} perspective focusing on {', '.join(objectives)}, this analysis..."

    def _generate_stakeholder_concerns(
        self, analysis: str, config: Dict[str, Any], stakeholder_type: str
    ) -> List[str]:
        """Generate stakeholder concerns"""
        return config.get("concerns", [])[:3]

    def _generate_stakeholder_alternatives(
        self, analysis: str, config: Dict[str, Any], stakeholder_type: str
    ) -> List[str]:
        """Generate alternative solutions from stakeholder perspective"""
        return [
            f"Alternative approach optimized for {config['objectives'][0]}",
            f"Phased implementation to reduce {config['concerns'][0]}",
            f"Modified solution addressing {stakeholder_type} priorities",
        ]

    def _surface_all_assumptions(
        self, analysis: str, context: Dict[str, Any]
    ) -> List[str]:
        """Surface all implicit assumptions in analysis"""
        return [
            "Historical patterns will continue",
            "Stakeholders will act rationally",
            "External environment will remain stable",
            "Resources will be available as needed",
            "Implementation will go as planned",
        ]

    def _assess_implicit_level(self, assumption: str, analysis: str) -> float:
        """Assess how implicit/unstated the assumption is"""
        if assumption.lower() in analysis.lower():
            return 0.2  # Explicitly stated
        return 0.8  # Highly implicit

    def _assess_evidence_strength(self, assumption: str, analysis: str) -> float:
        """Assess strength of evidence supporting assumption"""
        # Simplified scoring
        return 0.5

    def _assess_failure_impact(self, assumption: str, analysis: str) -> float:
        """Assess impact if assumption fails"""
        if "continue" in assumption or "stable" in assumption:
            return 0.8  # High impact
        return 0.5

    def _assess_testability(self, assumption: str, context: Dict[str, Any]) -> float:
        """Assess how testable the assumption is"""
        if "historical" in assumption or "patterns" in assumption:
            return 0.8  # Highly testable
        return 0.4

    def _generate_alternative_scenarios(self, assumption: str) -> List[str]:
        """Generate alternative scenarios if assumption fails"""
        return [
            f"Opposite of assumption: {assumption} does not hold",
            f"Partial failure: {assumption} holds only partially",
            f"Delayed effect: {assumption} becomes invalid over time",
        ]

    def _suggest_testing_methods(
        self, assumption: str, context: Dict[str, Any]
    ) -> List[str]:
        """Suggest methods for testing assumption"""
        return [
            "Historical data analysis",
            "Scenario modeling",
            "Small-scale pilot testing",
            "Expert consultation",
        ]

    def _extract_thesis_position(self, analysis: str) -> str:
        """Extract thesis position from analysis"""
        return "Original analysis position and recommendations"

    def _extract_antithesis_from_challenge(
        self, challenge: O2DChallenge
    ) -> Optional[str]:
        """Extract antithesis position from challenge"""
        return (
            challenge.challenge_summary.split(".")[0]
            if challenge.challenge_summary
            else None
        )

    def _generate_synthesis_position(
        self, thesis: str, antitheses: List[str], analysis: str, context: Dict[str, Any]
    ) -> str:
        """Generate synthesis position integrating thesis and antitheses"""
        return f"Enhanced approach incorporating {thesis} with safeguards addressing {', '.join(antitheses[:2])}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert O2D system state to dictionary for serialization"""
        return {
            "config": {
                "default_tier": self.config.default_tier.value,
                "psychological_safety_threshold": self.config.psychological_safety_threshold,
                "solution_orientation_threshold": self.config.solution_orientation_threshold,
                "max_challenges_per_session": self.config.max_challenges_per_session,
            },
            "challenge_history_count": len(self.challenge_history),
            "stakeholder_personas": list(self.stakeholder_personas.keys()),
        }


# Example usage and testing
if __name__ == "__main__":
    import asyncio

    async def test_o2d_system():
        """Test the Obligation to Dissent system"""

        # Initialize system
        o2d = ObligationToDissentSystem()

        # Test analysis
        test_analysis = """
        Based on our analysis of market data from the past 3 years, we recommend 
        expanding into the European market. The data shows consistent growth in demand 
        for our product category, and our competitive analysis suggests we can capture 
        15% market share within 24 months. Therefore, we should proceed with the 
        expansion plan immediately to maximize first-mover advantage.
        """

        test_context = {
            "decision_type": "market_expansion",
            "stakes": "high",
            "timeline": "24_months",
            "budget": 10000000,
        }

        # Generate challenges
        challenges = await o2d.generate_systematic_dissent(
            test_analysis, test_context, tier=DissentTier.SIGNIFICANT
        )

        print(f"Generated {len(challenges)} O2D challenges:")
        for i, challenge in enumerate(challenges, 1):
            print(f"\n{i}. {challenge.challenge_type.value.upper()}")
            print(f"   Summary: {challenge.challenge_summary}")
            print(f"   Safety Score: {challenge.psychological_safety_score:.2f}")
            print(f"   Solution Orientation: {challenge.solution_orientation:.2f}")
            if challenge.next_steps:
                print(f"   Next Steps: {', '.join(challenge.next_steps[:2])}")

    # Run test
    asyncio.run(test_o2d_system())
