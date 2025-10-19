#!/usr/bin/env python3
"""
METIS Internal Challenger System
Sprint 3.2: Enhanced assumption testing with Munger-style skeptical analysis

This system creates an internal "devil's advocate" that systematically challenges
assumptions, reasoning, and conclusions using multiple skeptical frameworks:

1. Ackoff's Problem Dissolution - Challenge fundamental assumptions
2. Munger's Inversion Analysis - What could go wrong?
3. Constitutional Bias Audit - Check for systematic thinking errors
4. Evidence-Armed Validation - Research-backed challenges
5. Context-Intelligent Skepticism - Use cognitive exhaust for targeted challenges

Key Features:
- Progressive skepticism levels (L1-L4)
- Multi-dimensional assumption analysis
- Self-doubt calibration mechanisms
- Integration with Context Intelligence for targeted challenges
- Confidence adjustment based on challenge outcomes
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

# Core METIS components
from src.engine.models.data_contracts import ReasoningStep, EngagementContext
from src.interfaces.context_intelligence_interface import IContextIntelligence
from src.engine.engines.validation.assumption_challenger import AssumptionChallenger
from src.intelligence.research_armed_challenger import ResearchArmedChallenger
from src.engine.adapters.cognitive_models import  # Migrated MungerOverlay, RigorLevel

# LLM integration for internal dialogue
try:
    from src.engine.adapters.llm_integration import  # Migrated get_unified_llm_adapter

    LLM_ADAPTER_AVAILABLE = True
except ImportError:
    LLM_ADAPTER_AVAILABLE = False

# Legacy imports for backward compatibility
try:
    from src.integrations.claude_client import get_claude_client, LLMCallType

    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False

logger = logging.getLogger(__name__)


class SkepticalRigorLevel(Enum):
    """Progressive levels of skeptical analysis"""

    BASIC = "basic"  # Surface-level assumption checking
    SYSTEMATIC = "systematic"  # Multi-framework analysis
    CONSTITUTIONAL = "constitutional"  # Deep bias audit + constitutional review
    MUNGER_COMPLETE = "munger_complete"  # Full Munger + Ackoff + research-armed


class AssumptionType(Enum):
    """Categories of assumptions to challenge"""

    CAUSAL = "causal"  # A causes B relationships
    TEMPORAL = "temporal"  # Timing and sequence assumptions
    STAKEHOLDER = "stakeholder"  # Actor behavior assumptions
    RESOURCE = "resource"  # Availability and capability assumptions
    MARKET = "market"  # Business environment assumptions
    SYSTEMIC = "systemic"  # System behavior assumptions
    CONSTRAINT = "constraint"  # Limitation assumptions
    SUCCESS_CRITERIA = "success_criteria"  # Definition of success assumptions


class ChallengeFramework(Enum):
    """Skeptical analysis frameworks"""

    ACKOFF_DISSOLUTION = "ackoff_dissolution"
    MUNGER_INVERSION = "munger_inversion"
    CONSTITUTIONAL_BIAS = "constitutional_bias"
    EVIDENCE_VALIDATION = "evidence_validation"
    CONTEXT_INTELLIGENCE = "context_intelligence"
    SYSTEMS_BOUNDARIES = "systems_boundaries"
    STAKEHOLDER_OBJECTIVES = "stakeholder_objectives"


@dataclass
class AssumptionChallenge:
    """A specific challenge to an assumption"""

    assumption_id: str
    original_assumption: str
    assumption_type: AssumptionType
    challenge_framework: ChallengeFramework
    challenge_statement: str
    evidence_against: List[str] = field(default_factory=list)
    alternative_hypotheses: List[str] = field(default_factory=list)
    confidence_impact: float = 0.0  # -1.0 to 1.0
    research_backing: Optional[str] = None
    munger_principle: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class InternalChallengerResult:
    """Complete result from internal challenger analysis"""

    engagement_id: str
    original_reasoning_steps: List[ReasoningStep]
    challenges: List[AssumptionChallenge]
    rigor_level: SkepticalRigorLevel
    total_assumptions_identified: int
    assumptions_challenged: int
    net_confidence_adjustment: float
    skeptical_summary: str
    recommended_actions: List[str]
    red_flags: List[str]
    execution_time_ms: float
    frameworks_used: List[ChallengeFramework]


class InternalChallengerSystem:
    """
    Internal Challenger System for rigorous assumption testing

    This system acts as an internal skeptic that systematically challenges
    reasoning through multiple frameworks to improve decision quality.
    """

    def __init__(
        self,
        context_intelligence: Optional[IContextIntelligence] = None,
        default_rigor_level: SkepticalRigorLevel = SkepticalRigorLevel.SYSTEMATIC,
        confidence_threshold_for_rigor_increase: float = 0.85,
        logger: Optional[logging.Logger] = None,
    ):
        self.context_intelligence = context_intelligence
        self.default_rigor_level = default_rigor_level
        self.confidence_threshold = confidence_threshold_for_rigor_increase
        self.logger = logger or logging.getLogger(__name__)

        # Initialize skeptical framework components
        self.assumption_challenger = AssumptionChallenger()
        self.research_challenger = ResearchArmedChallenger()
        self.munger_overlay = MungerOverlay()

        # Internal challenger state
        self.challenge_history: List[InternalChallengerResult] = []
        self.assumption_patterns: Dict[str, int] = (
            {}
        )  # Track recurring assumption types
        self.effectiveness_metrics: Dict[ChallengeFramework, float] = {}

        self.logger.info("🎯 Internal Challenger System initialized")
        self.logger.info(f"   Default rigor level: {default_rigor_level.value}")
        self.logger.info(
            f"   Confidence threshold for rigor escalation: {confidence_threshold_for_rigor_increase}"
        )

    async def challenge_reasoning(
        self,
        reasoning_steps: List[ReasoningStep],
        engagement_id: str,
        rigor_level: Optional[SkepticalRigorLevel] = None,
        focus_areas: Optional[List[AssumptionType]] = None,
    ) -> InternalChallengerResult:
        """
        Apply internal challenger analysis to reasoning steps

        Args:
            reasoning_steps: Reasoning to challenge
            engagement_id: Engagement context
            rigor_level: Level of skeptical analysis (defaults to system default)
            focus_areas: Specific assumption types to focus on

        Returns:
            Complete challenger analysis results
        """
        start_time = datetime.now()

        # Determine rigor level (escalate for high confidence)
        effective_rigor = self._determine_rigor_level(reasoning_steps, rigor_level)

        self.logger.info(
            f"🎯 Starting internal challenger analysis for engagement {engagement_id}"
        )
        self.logger.info(f"   Rigor level: {effective_rigor.value}")
        self.logger.info(f"   Reasoning steps: {len(reasoning_steps)}")
        self.logger.info(
            f"   Focus areas: {[fa.value for fa in focus_areas] if focus_areas else 'All'}"
        )

        # Step 1: Extract assumptions from reasoning
        assumptions = await self._extract_assumptions(reasoning_steps)
        self.logger.info(f"📊 Extracted {len(assumptions)} assumptions for challenge")

        # Step 2: Apply skeptical frameworks based on rigor level
        challenges = await self._apply_skeptical_frameworks(
            assumptions, effective_rigor, focus_areas, engagement_id
        )

        # Step 3: Context Intelligence enhancement (if available)
        if self.context_intelligence:
            challenges = await self._enhance_with_context_intelligence(
                challenges, engagement_id
            )

        # Step 4: Generate skeptical summary and recommendations
        skeptical_summary = await self._generate_skeptical_summary(
            challenges, effective_rigor
        )
        recommendations = self._generate_recommendations(challenges)
        red_flags = self._identify_red_flags(challenges)

        # Step 5: Calculate confidence adjustments
        net_confidence_adjustment = self._calculate_confidence_adjustment(challenges)

        # Create result
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        frameworks_used = list(set(c.challenge_framework for c in challenges))

        result = InternalChallengerResult(
            engagement_id=engagement_id,
            original_reasoning_steps=reasoning_steps,
            challenges=challenges,
            rigor_level=effective_rigor,
            total_assumptions_identified=len(assumptions),
            assumptions_challenged=len(challenges),
            net_confidence_adjustment=net_confidence_adjustment,
            skeptical_summary=skeptical_summary,
            recommended_actions=recommendations,
            red_flags=red_flags,
            execution_time_ms=execution_time,
            frameworks_used=frameworks_used,
        )

        # Store results for learning
        self.challenge_history.append(result)
        self._update_learning_metrics(result)

        self.logger.info(f"✅ Internal challenger analysis complete")
        self.logger.info(f"   Challenges generated: {len(challenges)}")
        self.logger.info(
            f"   Net confidence adjustment: {net_confidence_adjustment:.3f}"
        )
        self.logger.info(f"   Red flags identified: {len(red_flags)}")
        self.logger.info(f"   Execution time: {execution_time:.1f}ms")

        return result

    def _determine_rigor_level(
        self,
        reasoning_steps: List[ReasoningStep],
        requested_rigor: Optional[SkepticalRigorLevel],
    ) -> SkepticalRigorLevel:
        """Determine appropriate rigor level based on reasoning confidence"""
        if requested_rigor:
            return requested_rigor

        # Calculate average confidence
        confidences = []
        for step in reasoning_steps:
            if hasattr(step, "confidence_score") and step.confidence_score:
                confidences.append(step.confidence_score)

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

        # Escalate rigor for high confidence (may indicate overconfidence)
        if avg_confidence >= 0.9:
            return SkepticalRigorLevel.MUNGER_COMPLETE
        elif avg_confidence >= self.confidence_threshold:
            return SkepticalRigorLevel.CONSTITUTIONAL
        elif avg_confidence >= 0.7:
            return SkepticalRigorLevel.SYSTEMATIC
        else:
            return SkepticalRigorLevel.BASIC

    async def _extract_assumptions(
        self, reasoning_steps: List[ReasoningStep]
    ) -> List[Dict[str, Any]]:
        """Extract assumptions from reasoning steps using LLM analysis"""
        assumptions = []

        for step in reasoning_steps:
            # Extract explicit assumptions if available
            if hasattr(step, "assumptions_made") and step.assumptions_made:
                for assumption in step.assumptions_made:
                    assumptions.append(
                        {
                            "text": assumption,
                            "source_step": (
                                step.step_id if hasattr(step, "step_id") else "unknown"
                            ),
                            "explicit": True,
                            "reasoning_context": (
                                step.reasoning_text
                                if hasattr(step, "reasoning_text")
                                else ""
                            ),
                        }
                    )

            # Use LLM to extract implicit assumptions
            if CLAUDE_AVAILABLE:
                implicit_assumptions = await self._extract_implicit_assumptions(step)
                assumptions.extend(implicit_assumptions)

        return assumptions

    async def _extract_implicit_assumptions(
        self, reasoning_step: ReasoningStep
    ) -> List[Dict[str, Any]]:
        """Use LLM to extract implicit assumptions from reasoning"""
        if not CLAUDE_AVAILABLE:
            return []

        try:
            claude_client = await get_claude_client()

            prompt = f"""
            Analyze this reasoning step and identify implicit assumptions:
            
            Reasoning: {reasoning_step.reasoning_text if hasattr(reasoning_step, 'reasoning_text') else 'N/A'}
            Model Applied: {reasoning_step.mental_model_applied if hasattr(reasoning_step, 'mental_model_applied') else 'N/A'}
            
            Identify implicit assumptions in these categories:
            1. Causal assumptions (X causes Y)
            2. Temporal assumptions (timing, sequence)
            3. Stakeholder behavior assumptions
            4. Resource/capability assumptions
            5. Market/environment assumptions
            6. System behavior assumptions
            7. Constraint assumptions
            8. Success criteria assumptions
            
            For each assumption identified, provide:
            - The assumption statement
            - The category (from above)
            - Why this assumption might be problematic
            
            Format as JSON list of assumptions.
            """

            response = await claude_client.call_claude(
                prompt=prompt, call_type=LLMCallType.REASONING, max_tokens=1500
            )

            # Parse JSON response
            try:
                implicit_assumptions = json.loads(response.content)
                return [
                    {
                        "text": assumption.get("statement", ""),
                        "source_step": (
                            reasoning_step.step_id
                            if hasattr(reasoning_step, "step_id")
                            else "unknown"
                        ),
                        "explicit": False,
                        "category": assumption.get("category", "unknown"),
                        "risk_assessment": assumption.get("risk", ""),
                        "reasoning_context": (
                            reasoning_step.reasoning_text
                            if hasattr(reasoning_step, "reasoning_text")
                            else ""
                        ),
                    }
                    for assumption in implicit_assumptions
                ]
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse assumption extraction JSON")
                return []

        except Exception as e:
            self.logger.warning(f"Implicit assumption extraction failed: {e}")
            return []

    async def _apply_skeptical_frameworks(
        self,
        assumptions: List[Dict[str, Any]],
        rigor_level: SkepticalRigorLevel,
        focus_areas: Optional[List[AssumptionType]],
        engagement_id: str,
    ) -> List[AssumptionChallenge]:
        """Apply multiple skeptical frameworks based on rigor level"""
        challenges = []

        # Determine which frameworks to apply
        frameworks = self._select_frameworks(rigor_level)

        for i, assumption in enumerate(assumptions):
            assumption_id = f"{engagement_id}_assumption_{i}"
            assumption_type = self._classify_assumption_type(assumption)

            # Skip if not in focus areas
            if focus_areas and assumption_type not in focus_areas:
                continue

            # Apply each framework
            for framework in frameworks:
                try:
                    challenge = await self._apply_framework(
                        assumption_id, assumption, assumption_type, framework
                    )
                    if challenge:
                        challenges.append(challenge)
                except Exception as e:
                    self.logger.warning(
                        f"Framework {framework.value} failed for assumption {assumption_id}: {e}"
                    )

        return challenges

    def _select_frameworks(
        self, rigor_level: SkepticalRigorLevel
    ) -> List[ChallengeFramework]:
        """Select frameworks based on rigor level"""
        base_frameworks = [ChallengeFramework.ACKOFF_DISSOLUTION]

        if rigor_level == SkepticalRigorLevel.BASIC:
            return base_frameworks

        if rigor_level in [
            SkepticalRigorLevel.SYSTEMATIC,
            SkepticalRigorLevel.CONSTITUTIONAL,
            SkepticalRigorLevel.MUNGER_COMPLETE,
        ]:
            base_frameworks.extend(
                [
                    ChallengeFramework.MUNGER_INVERSION,
                    ChallengeFramework.SYSTEMS_BOUNDARIES,
                    ChallengeFramework.STAKEHOLDER_OBJECTIVES,
                ]
            )

        if rigor_level in [
            SkepticalRigorLevel.CONSTITUTIONAL,
            SkepticalRigorLevel.MUNGER_COMPLETE,
        ]:
            base_frameworks.append(ChallengeFramework.CONSTITUTIONAL_BIAS)

        if rigor_level == SkepticalRigorLevel.MUNGER_COMPLETE:
            base_frameworks.extend(
                [
                    ChallengeFramework.EVIDENCE_VALIDATION,
                    ChallengeFramework.CONTEXT_INTELLIGENCE,
                ]
            )

        return base_frameworks

    def _classify_assumption_type(self, assumption: Dict[str, Any]) -> AssumptionType:
        """Classify assumption into type categories"""
        text = assumption.get("text", "").lower()
        category = assumption.get("category", "").lower()

        # Use category if available
        if "causal" in category:
            return AssumptionType.CAUSAL
        elif "temporal" in category or "timing" in category:
            return AssumptionType.TEMPORAL
        elif "stakeholder" in category or "behavior" in category:
            return AssumptionType.STAKEHOLDER
        elif "resource" in category or "capability" in category:
            return AssumptionType.RESOURCE
        elif "market" in category or "environment" in category:
            return AssumptionType.MARKET
        elif "system" in category:
            return AssumptionType.SYSTEMIC
        elif "constraint" in category:
            return AssumptionType.CONSTRAINT
        elif "success" in category:
            return AssumptionType.SUCCESS_CRITERIA

        # Fallback to text analysis
        if "cause" in text or "because" in text or "leads to" in text:
            return AssumptionType.CAUSAL
        elif "when" in text or "after" in text or "before" in text:
            return AssumptionType.TEMPORAL
        elif "customer" in text or "user" in text or "people will" in text:
            return AssumptionType.STAKEHOLDER
        elif "market" in text or "competition" in text or "economic" in text:
            return AssumptionType.MARKET

        return AssumptionType.SYSTEMIC  # Default

    async def _apply_framework(
        self,
        assumption_id: str,
        assumption: Dict[str, Any],
        assumption_type: AssumptionType,
        framework: ChallengeFramework,
    ) -> Optional[AssumptionChallenge]:
        """Apply a specific skeptical framework to an assumption"""

        if framework == ChallengeFramework.ACKOFF_DISSOLUTION:
            return await self._apply_ackoff_dissolution(
                assumption_id, assumption, assumption_type
            )
        elif framework == ChallengeFramework.MUNGER_INVERSION:
            return await self._apply_munger_inversion(
                assumption_id, assumption, assumption_type
            )
        elif framework == ChallengeFramework.CONSTITUTIONAL_BIAS:
            return await self._apply_constitutional_bias(
                assumption_id, assumption, assumption_type
            )
        elif framework == ChallengeFramework.EVIDENCE_VALIDATION:
            return await self._apply_evidence_validation(
                assumption_id, assumption, assumption_type
            )
        elif framework == ChallengeFramework.SYSTEMS_BOUNDARIES:
            return await self._apply_systems_boundaries(
                assumption_id, assumption, assumption_type
            )
        elif framework == ChallengeFramework.STAKEHOLDER_OBJECTIVES:
            return await self._apply_stakeholder_objectives(
                assumption_id, assumption, assumption_type
            )

        return None

    async def _apply_ackoff_dissolution(
        self,
        assumption_id: str,
        assumption: Dict[str, Any],
        assumption_type: AssumptionType,
    ) -> Optional[AssumptionChallenge]:
        """Apply Ackoff's problem dissolution approach"""
        assumption_text = assumption.get("text", "")

        # Generate Ackoff-style challenge
        challenge_statement = (
            f"What if we dissolved this assumption entirely? {assumption_text}"
        )

        # Use assumption challenger if available
        try:
            # Generate alternative hypotheses
            alternatives = [
                f"The opposite of '{assumption_text}' might be true",
                f"This assumption might be solving the wrong problem",
                f"The system might work differently than '{assumption_text}' assumes",
            ]

            return AssumptionChallenge(
                assumption_id=assumption_id,
                original_assumption=assumption_text,
                assumption_type=assumption_type,
                challenge_framework=ChallengeFramework.ACKOFF_DISSOLUTION,
                challenge_statement=challenge_statement,
                alternative_hypotheses=alternatives,
                confidence_impact=-0.3,  # Moderate doubt
                evidence_against=[],
            )
        except Exception as e:
            self.logger.warning(f"Ackoff dissolution failed: {e}")
            return None

    async def _apply_munger_inversion(
        self,
        assumption_id: str,
        assumption: Dict[str, Any],
        assumption_type: AssumptionType,
    ) -> Optional[AssumptionChallenge]:
        """Apply Munger's inversion analysis"""
        assumption_text = assumption.get("text", "")

        # Inversion challenge
        challenge_statement = f"Inversion analysis: What if '{assumption_text}' is completely wrong? What would fail?"

        # Generate failure scenarios
        failure_scenarios = [
            f"If '{assumption_text}' fails, what cascading effects occur?",
            f"What evidence would prove '{assumption_text}' is false?",
            f"Who benefits if '{assumption_text}' is wrong?",
        ]

        # Use Munger principles
        munger_principle = (
            "Invert, always invert - show me the trouble, show me the problems"
        )

        return AssumptionChallenge(
            assumption_id=assumption_id,
            original_assumption=assumption_text,
            assumption_type=assumption_type,
            challenge_framework=ChallengeFramework.MUNGER_INVERSION,
            challenge_statement=challenge_statement,
            alternative_hypotheses=failure_scenarios,
            confidence_impact=-0.4,  # Strong doubt from inversion
            munger_principle=munger_principle,
        )

    async def _apply_constitutional_bias(
        self,
        assumption_id: str,
        assumption: Dict[str, Any],
        assumption_type: AssumptionType,
    ) -> Optional[AssumptionChallenge]:
        """Apply constitutional bias audit"""
        assumption_text = assumption.get("text", "")

        # Check for common biases
        bias_patterns = {
            "confirmation": "seeking information that confirms existing beliefs",
            "availability": "overweighting easily recalled information",
            "anchoring": "being influenced by the first information received",
            "overconfidence": "being more certain than accuracy warrants",
            "planning_fallacy": "underestimating time, costs, and risks",
        }

        identified_biases = []
        for bias, description in bias_patterns.items():
            if self._check_for_bias(assumption_text, bias):
                identified_biases.append(f"{bias}: {description}")

        if identified_biases:
            challenge_statement = (
                f"Bias audit of '{assumption_text}': {', '.join(identified_biases)}"
            )

            return AssumptionChallenge(
                assumption_id=assumption_id,
                original_assumption=assumption_text,
                assumption_type=assumption_type,
                challenge_framework=ChallengeFramework.CONSTITUTIONAL_BIAS,
                challenge_statement=challenge_statement,
                evidence_against=identified_biases,
                confidence_impact=-0.5,  # Strong doubt from bias detection
            )

        return None

    def _check_for_bias(self, assumption_text: str, bias_type: str) -> bool:
        """Simple heuristic bias detection"""
        text = assumption_text.lower()

        if bias_type == "confirmation":
            return any(
                word in text
                for word in ["obviously", "clearly", "certainly", "definitely"]
            )
        elif bias_type == "availability":
            return any(
                word in text for word in ["recent", "latest", "just saw", "remember"]
            )
        elif bias_type == "anchoring":
            return any(word in text for word in ["first", "initial", "starting with"])
        elif bias_type == "overconfidence":
            return any(
                word in text for word in ["sure", "100%", "guaranteed", "always"]
            )
        elif bias_type == "planning_fallacy":
            return any(
                word in text
                for word in ["quickly", "easily", "simple", "straightforward"]
            )

        return False

    async def _apply_evidence_validation(
        self,
        assumption_id: str,
        assumption: Dict[str, Any],
        assumption_type: AssumptionType,
    ) -> Optional[AssumptionChallenge]:
        """Apply evidence-based validation through research"""
        assumption_text = assumption.get("text", "")

        # Generate evidence-based challenge
        challenge_statement = f"Evidence validation required: What data supports or contradicts '{assumption_text}'?"

        # Use research challenger if available
        try:
            evidence_questions = [
                f"What studies or data support '{assumption_text}'?",
                f"What counterexamples exist for '{assumption_text}'?",
                f"What recent developments might invalidate '{assumption_text}'?",
            ]

            return AssumptionChallenge(
                assumption_id=assumption_id,
                original_assumption=assumption_text,
                assumption_type=assumption_type,
                challenge_framework=ChallengeFramework.EVIDENCE_VALIDATION,
                challenge_statement=challenge_statement,
                alternative_hypotheses=evidence_questions,
                confidence_impact=-0.2,  # Moderate doubt pending evidence
                research_backing="Evidence validation required",
            )
        except Exception as e:
            self.logger.warning(f"Evidence validation failed: {e}")
            return None

    async def _apply_systems_boundaries(
        self,
        assumption_id: str,
        assumption: Dict[str, Any],
        assumption_type: AssumptionType,
    ) -> Optional[AssumptionChallenge]:
        """Challenge system boundary assumptions"""
        assumption_text = assumption.get("text", "")

        challenge_statement = f"Systems boundary check: Does '{assumption_text}' artificially constrain the system?"

        boundary_questions = [
            f"What if we expanded the system boundary beyond '{assumption_text}'?",
            f"What external factors does '{assumption_text}' ignore?",
            f"What stakeholders does '{assumption_text}' exclude?",
        ]

        return AssumptionChallenge(
            assumption_id=assumption_id,
            original_assumption=assumption_text,
            assumption_type=assumption_type,
            challenge_framework=ChallengeFramework.SYSTEMS_BOUNDARIES,
            challenge_statement=challenge_statement,
            alternative_hypotheses=boundary_questions,
            confidence_impact=-0.25,
        )

    async def _apply_stakeholder_objectives(
        self,
        assumption_id: str,
        assumption: Dict[str, Any],
        assumption_type: AssumptionType,
    ) -> Optional[AssumptionChallenge]:
        """Challenge stakeholder objective assumptions"""
        assumption_text = assumption.get("text", "")

        challenge_statement = f"Stakeholder objective analysis: Whose objectives does '{assumption_text}' serve?"

        objective_questions = [
            f"Which stakeholders benefit if '{assumption_text}' is true?",
            f"Which stakeholders are harmed if '{assumption_text}' is true?",
            f"What hidden agendas might '{assumption_text}' serve?",
        ]

        return AssumptionChallenge(
            assumption_id=assumption_id,
            original_assumption=assumption_text,
            assumption_type=assumption_type,
            challenge_framework=ChallengeFramework.STAKEHOLDER_OBJECTIVES,
            challenge_statement=challenge_statement,
            alternative_hypotheses=objective_questions,
            confidence_impact=-0.3,
        )

    async def _enhance_with_context_intelligence(
        self, challenges: List[AssumptionChallenge], engagement_id: str
    ) -> List[AssumptionChallenge]:
        """Enhance challenges using Context Intelligence cognitive exhaust"""
        if not self.context_intelligence:
            return challenges

        try:
            # Get relevant cognitive exhaust for challenge enhancement
            contexts = await self.context_intelligence.get_relevant_contexts(
                query="assumption challenges skeptical analysis",
                engagement_id=engagement_id,
                context_types=[
                    "challenge_validation",
                    "assumption_test",
                    "inversion_analysis",
                ],
            )

            # Enhance challenges with context insights
            for challenge in challenges:
                for context in contexts:
                    if (
                        hasattr(context, "content")
                        and challenge.original_assumption.lower()
                        in context.content.lower()
                    ):
                        challenge.evidence_against.append(
                            f"Context insight: {context.content[:100]}..."
                        )
                        challenge.confidence_impact -= (
                            0.1  # Additional doubt from context
                        )

        except Exception as e:
            self.logger.warning(f"Context Intelligence enhancement failed: {e}")

        return challenges

    async def _generate_skeptical_summary(
        self, challenges: List[AssumptionChallenge], rigor_level: SkepticalRigorLevel
    ) -> str:
        """Generate a comprehensive skeptical summary"""
        if not challenges:
            return "No significant challenges identified in the reasoning."

        # Categorize challenges by framework
        framework_groups = {}
        for challenge in challenges:
            framework = challenge.challenge_framework
            if framework not in framework_groups:
                framework_groups[framework] = []
            framework_groups[framework].append(challenge)

        # Build summary
        summary_parts = [
            f"Internal Challenger Analysis (Rigor Level: {rigor_level.value})",
            f"Total challenges identified: {len(challenges)}",
            "",
        ]

        for framework, framework_challenges in framework_groups.items():
            summary_parts.append(
                f"{framework.value.title().replace('_', ' ')} ({len(framework_challenges)} challenges):"
            )
            for challenge in framework_challenges[:2]:  # Top 2 per framework
                summary_parts.append(f"  • {challenge.challenge_statement[:100]}...")
            summary_parts.append("")

        # Overall assessment
        avg_confidence_impact = sum(c.confidence_impact for c in challenges) / len(
            challenges
        )
        if avg_confidence_impact <= -0.4:
            assessment = "SIGNIFICANT SKEPTICAL CONCERNS identified"
        elif avg_confidence_impact <= -0.2:
            assessment = "MODERATE skeptical questions raised"
        else:
            assessment = "Minor skeptical considerations noted"

        summary_parts.append(f"Overall Assessment: {assessment}")
        summary_parts.append(f"Net confidence impact: {avg_confidence_impact:.3f}")

        return "\n".join(summary_parts)

    def _generate_recommendations(
        self, challenges: List[AssumptionChallenge]
    ) -> List[str]:
        """Generate actionable recommendations from challenges"""
        recommendations = []

        # Group by assumption type
        type_groups = {}
        for challenge in challenges:
            assumption_type = challenge.assumption_type
            if assumption_type not in type_groups:
                type_groups[assumption_type] = []
            type_groups[assumption_type].append(challenge)

        # Generate recommendations per type
        for assumption_type, type_challenges in type_groups.items():
            if len(type_challenges) >= 2:  # Multiple challenges for this type
                recommendations.append(
                    f"Validate {assumption_type.value} assumptions - {len(type_challenges)} concerns identified"
                )

        # Framework-specific recommendations
        framework_counts = {}
        for challenge in challenges:
            framework = challenge.challenge_framework
            framework_counts[framework] = framework_counts.get(framework, 0) + 1

        if ChallengeFramework.EVIDENCE_VALIDATION in framework_counts:
            recommendations.append(
                "Conduct additional research to validate evidence-based assumptions"
            )

        if ChallengeFramework.CONSTITUTIONAL_BIAS in framework_counts:
            recommendations.append(
                "Review reasoning for systematic biases before final decisions"
            )

        if ChallengeFramework.MUNGER_INVERSION in framework_counts:
            recommendations.append(
                "Develop contingency plans for identified failure scenarios"
            )

        return recommendations[:5]  # Top 5 recommendations

    def _identify_red_flags(self, challenges: List[AssumptionChallenge]) -> List[str]:
        """Identify critical red flags from challenges"""
        red_flags = []

        # High-impact challenges
        high_impact_challenges = [c for c in challenges if c.confidence_impact <= -0.4]
        if high_impact_challenges:
            red_flags.append(
                f"{len(high_impact_challenges)} high-impact assumption challenges identified"
            )

        # Multiple bias detections
        bias_challenges = [
            c
            for c in challenges
            if c.challenge_framework == ChallengeFramework.CONSTITUTIONAL_BIAS
        ]
        if len(bias_challenges) >= 2:
            red_flags.append(f"Multiple systematic biases detected in reasoning")

        # Evidence gaps
        evidence_challenges = [
            c
            for c in challenges
            if c.challenge_framework == ChallengeFramework.EVIDENCE_VALIDATION
        ]
        if evidence_challenges:
            red_flags.append(f"Critical evidence gaps identified for key assumptions")

        # Stakeholder conflicts
        stakeholder_challenges = [
            c
            for c in challenges
            if c.challenge_framework == ChallengeFramework.STAKEHOLDER_OBJECTIVES
        ]
        if len(stakeholder_challenges) >= 2:
            red_flags.append(f"Potential stakeholder objective conflicts detected")

        return red_flags

    def _calculate_confidence_adjustment(
        self, challenges: List[AssumptionChallenge]
    ) -> float:
        """Calculate net confidence adjustment from challenges"""
        if not challenges:
            return 0.0

        # Weight challenges by framework importance
        framework_weights = {
            ChallengeFramework.CONSTITUTIONAL_BIAS: 1.0,
            ChallengeFramework.EVIDENCE_VALIDATION: 0.9,
            ChallengeFramework.MUNGER_INVERSION: 0.8,
            ChallengeFramework.ACKOFF_DISSOLUTION: 0.7,
            ChallengeFramework.SYSTEMS_BOUNDARIES: 0.6,
            ChallengeFramework.STAKEHOLDER_OBJECTIVES: 0.6,
            ChallengeFramework.CONTEXT_INTELLIGENCE: 0.5,
        }

        weighted_impact = 0.0
        total_weight = 0.0

        for challenge in challenges:
            weight = framework_weights.get(challenge.challenge_framework, 0.5)
            weighted_impact += challenge.confidence_impact * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        # Cap adjustment at -0.8 to prevent complete confidence destruction
        net_adjustment = max(-0.8, weighted_impact / total_weight)

        return net_adjustment

    def _update_learning_metrics(self, result: InternalChallengerResult):
        """Update learning metrics from challenger results"""
        # Track assumption pattern frequency
        for challenge in result.challenges:
            assumption_type = challenge.assumption_type.value
            self.assumption_patterns[assumption_type] = (
                self.assumption_patterns.get(assumption_type, 0) + 1
            )

        # Update framework effectiveness (placeholder for future ML)
        for framework in result.frameworks_used:
            framework_challenges = [
                c for c in result.challenges if c.challenge_framework == framework
            ]
            if framework_challenges:
                avg_impact = sum(
                    c.confidence_impact for c in framework_challenges
                ) / len(framework_challenges)
                current_effectiveness = self.effectiveness_metrics.get(framework, -0.2)
                # Simple exponential moving average
                self.effectiveness_metrics[framework] = (
                    0.8 * current_effectiveness + 0.2 * abs(avg_impact)
                )


class InternalChallengerSystemFactory:
    """Factory for creating Internal Challenger System instances"""

    @staticmethod
    def create_internal_challenger(
        context_intelligence: Optional[IContextIntelligence] = None,
        rigor_level: SkepticalRigorLevel = SkepticalRigorLevel.SYSTEMATIC,
        confidence_threshold: float = 0.85,
        logger: Optional[logging.Logger] = None,
    ) -> InternalChallengerSystem:
        """Create an Internal Challenger System instance"""

        return InternalChallengerSystem(
            context_intelligence=context_intelligence,
            default_rigor_level=rigor_level,
            confidence_threshold_for_rigor_increase=confidence_threshold,
            logger=logger,
        )

    @staticmethod
    def create_for_high_stakes_decisions(
        context_intelligence: Optional[IContextIntelligence] = None,
        logger: Optional[logging.Logger] = None,
    ) -> InternalChallengerSystem:
        """Create high-rigor internal challenger for critical decisions"""

        return InternalChallengerSystem(
            context_intelligence=context_intelligence,
            default_rigor_level=SkepticalRigorLevel.MUNGER_COMPLETE,
            confidence_threshold_for_rigor_increase=0.7,  # Lower threshold for escalation
            logger=logger,
        )
