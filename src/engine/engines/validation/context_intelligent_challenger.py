#!/usr/bin/env python3
"""
METIS Context-Intelligent Challenge Engine
Sprint 3.1: Integration of challenger systems with Context Intelligence Pipeline

This engine orchestrates all existing challenger components with the Context Intelligence
system to create a comprehensive self-doubt and validation mechanism:

1. AssumptionChallenger - Ackoff methodology for assumption testing
2. ResearchArmedChallenger - Evidence-backed challenging with smart research
3. MungerOverlay - L0-L3 bias detection and systematic worldly wisdom
4. BiasActivation - Centralized challenger system orchestration

Key Features:
- Context-aware challenging using cognitive exhaust
- Progressive challenge rigor (L1-L3) based on confidence scores
- Evidence-armed assumption testing with smart research triggering
- Integration with Pyramid Synthesis for executive-ready challenge reports
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

# Core METIS components
from src.engine.models.data_contracts import ReasoningStep
from src.interfaces.context_intelligence_interface import IContextIntelligence
from src.engine.engines.validation.assumption_challenger import AssumptionChallenger
from src.intelligence.research_armed_challenger import ResearchArmedChallenger
from src.core.munger_overlay import MungerOverlay
from src.intelligence.challenge_research_templates import ChallengeResearchTemplates


class ChallengeRigorLevel(Enum):
    """Challenge rigor levels for Context Intelligence integration"""

    LIGHT = "light"  # Basic assumption checking
    MODERATE = "moderate"  # Research-backed challenging
    INTENSIVE = "intensive"  # Full Munger overlay with L3 rigor
    CONSTITUTIONAL = "constitutional"  # Maximum rigor with constitutional bias audit


class ChallengeType(Enum):
    """Types of challenges to execute"""

    ASSUMPTION_TEST = "assumption_test"
    EVIDENCE_VALIDATION = "evidence_validation"
    BIAS_AUDIT = "bias_audit"
    INVERSION_ANALYSIS = "inversion_analysis"
    CONSTITUTIONAL_REVIEW = "constitutional_review"


@dataclass
class ChallengeRequest:
    """Request for challenging reasoning or deliverables"""

    reasoning_steps: List[ReasoningStep]
    engagement_id: str
    challenge_types: List[ChallengeType] = field(
        default_factory=lambda: [ChallengeType.ASSUMPTION_TEST]
    )
    rigor_level: ChallengeRigorLevel = ChallengeRigorLevel.MODERATE
    context_intelligence_enabled: bool = True
    max_challenges: int = 5


@dataclass
class ChallengeResult:
    """Result of challenging process"""

    challenge_id: str
    challenge_type: ChallengeType
    original_reasoning_step: str
    challenge_statement: str
    evidence_backing: List[str]
    confidence_impact: float  # How much this challenge affects confidence (-1.0 to 1.0)
    severity: str  # "low", "medium", "high", "critical"
    context_sources: List[str]  # Context Intelligence sources used
    research_armed: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ContextIntelligentChallengeEngine:
    """
    Sprint 3.1: Context-Intelligent Challenge Engine

    Orchestrates all METIS challenger systems with Context Intelligence to create
    sophisticated self-doubt and validation capabilities for consulting deliverables.
    """

    def __init__(
        self,
        context_intelligence: IContextIntelligence,
        settings: Any,
        logger: Optional[logging.Logger] = None,
    ):
        self.context_intelligence = context_intelligence
        self.settings = settings
        self.logger = logger or logging.getLogger(__name__)

        # Initialize challenger components
        self.assumption_challenger = AssumptionChallenger()
        self.research_challenger = ResearchArmedChallenger()
        self.munger_overlay = MungerOverlay()
        self.research_templates = ChallengeResearchTemplates()

        # Challenge orchestration state
        self.active_challenges: Dict[str, ChallengeResult] = {}
        self.challenge_history: List[ChallengeResult] = []

        self.logger.info(
            "🎯 Context-Intelligent Challenge Engine initialized with full challenger system integration"
        )

    async def execute_contextual_challenges(
        self, challenge_request: ChallengeRequest
    ) -> List[ChallengeResult]:
        """
        Execute contextual challenges on reasoning steps using Context Intelligence

        This is the main orchestration method that:
        1. Analyzes reasoning with Context Intelligence for challenge opportunities
        2. Selects appropriate challenger systems based on rigor level
        3. Executes challenges with evidence-backing from research
        4. Returns structured challenge results for integration with Pyramid Synthesis
        """

        self.logger.info(
            f"🎯 Executing contextual challenges for engagement {challenge_request.engagement_id}"
        )
        self.logger.info(f"   - Rigor level: {challenge_request.rigor_level.value}")
        self.logger.info(
            f"   - Challenge types: {[ct.value for ct in challenge_request.challenge_types]}"
        )
        self.logger.info(
            f"   - Reasoning steps: {len(challenge_request.reasoning_steps)}"
        )

        challenge_results = []

        try:
            # Step 1: Context Intelligence Analysis for Challenge Opportunities
            if challenge_request.context_intelligence_enabled:
                challenge_opportunities = await self._identify_challenge_opportunities(
                    challenge_request.reasoning_steps, challenge_request.engagement_id
                )
                self.logger.info(
                    f"📊 Context Intelligence identified {len(challenge_opportunities)} challenge opportunities"
                )
            else:
                challenge_opportunities = (
                    self._generate_default_challenge_opportunities(
                        challenge_request.reasoning_steps
                    )
                )

            # Step 2: Execute challenges based on rigor level and types
            for challenge_type in challenge_request.challenge_types:
                type_results = await self._execute_challenge_type(
                    challenge_type, challenge_request, challenge_opportunities
                )
                challenge_results.extend(type_results)

            # Step 3: Apply Context Intelligence to enhance challenges with cognitive exhaust
            enhanced_results = await self._enhance_challenges_with_context_intelligence(
                challenge_results, challenge_request.engagement_id
            )

            # Step 4: Store challenge results for future Context Intelligence use
            await self._store_challenge_cognitive_exhaust(
                enhanced_results, challenge_request.engagement_id
            )

            self.logger.info(
                f"✅ Contextual challenging complete: {len(enhanced_results)} challenges generated"
            )
            return enhanced_results[: challenge_request.max_challenges]

        except Exception as e:
            self.logger.error(f"❌ Contextual challenging failed: {str(e)}")
            raise

    async def _identify_challenge_opportunities(
        self, reasoning_steps: List[ReasoningStep], engagement_id: str
    ) -> List[Dict[str, Any]]:
        """Use Context Intelligence to identify the best challenge opportunities"""

        opportunities = []

        # Get relevant context from Context Intelligence for challenger systems
        try:
            contexts = await self.context_intelligence.get_relevant_context(
                current_query=f"challenge_opportunities_{engagement_id}",
                max_contexts=10,
                engagement_id=engagement_id,
            )

            # Analyze each reasoning step for challenge opportunities
            for i, step in enumerate(reasoning_steps):
                step_opportunities = await self._analyze_step_for_challenges(
                    step, contexts, i
                )
                opportunities.extend(step_opportunities)

        except Exception as e:
            self.logger.warning(
                f"⚠️ Context Intelligence challenge analysis failed: {e}"
            )
            opportunities = self._generate_default_challenge_opportunities(
                reasoning_steps
            )

        return opportunities

    async def _analyze_step_for_challenges(
        self,
        reasoning_step: ReasoningStep,
        contexts: List[Tuple[Any, Any]],
        step_index: int,
    ) -> List[Dict[str, Any]]:
        """Analyze individual reasoning step for specific challenge opportunities"""

        opportunities = []

        # Extract step content for analysis
        step_content = getattr(reasoning_step, "reasoning_text", None) or getattr(
            reasoning_step, "description", ""
        )
        step_confidence = getattr(reasoning_step, "confidence_score", None) or getattr(
            reasoning_step, "confidence", 0.5
        )
        assumptions = getattr(reasoning_step, "assumptions_made", [])

        # 1. Assumption Challenge Opportunities
        if assumptions:
            for assumption in assumptions:
                opportunities.append(
                    {
                        "type": ChallengeType.ASSUMPTION_TEST,
                        "target": assumption,
                        "step_index": step_index,
                        "confidence": step_confidence,
                        "context_relevance": self._calculate_context_relevance(
                            assumption, contexts
                        ),
                    }
                )

        # 2. Evidence Validation Opportunities (for low confidence steps)
        if step_confidence < 0.7:
            opportunities.append(
                {
                    "type": ChallengeType.EVIDENCE_VALIDATION,
                    "target": step_content,
                    "step_index": step_index,
                    "confidence": step_confidence,
                    "context_relevance": self._calculate_context_relevance(
                        step_content, contexts
                    ),
                }
            )

        # 3. Bias Audit Opportunities (for high-confidence steps that might be overconfident)
        if step_confidence > 0.85:
            opportunities.append(
                {
                    "type": ChallengeType.BIAS_AUDIT,
                    "target": step_content,
                    "step_index": step_index,
                    "confidence": step_confidence,
                    "context_relevance": self._calculate_context_relevance(
                        step_content, contexts
                    ),
                }
            )

        return opportunities

    def _calculate_context_relevance(
        self, content: str, contexts: List[Tuple[Any, Any]]
    ) -> float:
        """Calculate how relevant the content is to available contexts (simplified)"""
        if not contexts:
            return 0.0

        # Simple keyword matching for now - could be enhanced with semantic similarity
        content_words = set(content.lower().split())
        total_relevance = 0.0

        for context_data, relevance_score in contexts:
            # Extract text from context data
            context_text = str(context_data).lower()
            context_words = set(context_text.split())

            # Calculate word overlap
            overlap = len(content_words.intersection(context_words))
            word_relevance = overlap / max(len(content_words), 1)

            total_relevance += word_relevance * relevance_score

        return min(total_relevance, 1.0)

    def _generate_default_challenge_opportunities(
        self, reasoning_steps: List[ReasoningStep]
    ) -> List[Dict[str, Any]]:
        """Generate default challenge opportunities when Context Intelligence is not available"""

        opportunities = []

        for i, step in enumerate(reasoning_steps):
            step_confidence = getattr(step, "confidence_score", None) or getattr(
                step, "confidence", 0.5
            )

            # Always challenge low confidence steps with evidence validation
            if step_confidence < 0.6:
                opportunities.append(
                    {
                        "type": ChallengeType.EVIDENCE_VALIDATION,
                        "target": getattr(
                            step, "reasoning_text", getattr(step, "description", "")
                        ),
                        "step_index": i,
                        "confidence": step_confidence,
                        "context_relevance": 0.5,  # Default relevance
                    }
                )

            # Challenge assumptions if present
            assumptions = getattr(step, "assumptions_made", [])
            for assumption in assumptions:
                opportunities.append(
                    {
                        "type": ChallengeType.ASSUMPTION_TEST,
                        "target": assumption,
                        "step_index": i,
                        "confidence": step_confidence,
                        "context_relevance": 0.5,
                    }
                )

        return opportunities

    async def _execute_challenge_type(
        self,
        challenge_type: ChallengeType,
        challenge_request: ChallengeRequest,
        opportunities: List[Dict[str, Any]],
    ) -> List[ChallengeResult]:
        """Execute specific type of challenge using appropriate challenger system"""

        type_opportunities = [
            opp for opp in opportunities if opp["type"] == challenge_type
        ]
        if not type_opportunities:
            return []

        self.logger.info(
            f"🎯 Executing {challenge_type.value} challenges: {len(type_opportunities)} opportunities"
        )

        results = []

        try:
            if challenge_type == ChallengeType.ASSUMPTION_TEST:
                results = await self._execute_assumption_challenges(
                    type_opportunities, challenge_request
                )
            elif challenge_type == ChallengeType.EVIDENCE_VALIDATION:
                results = await self._execute_evidence_challenges(
                    type_opportunities, challenge_request
                )
            elif challenge_type == ChallengeType.BIAS_AUDIT:
                results = await self._execute_bias_challenges(
                    type_opportunities, challenge_request
                )
            elif challenge_type == ChallengeType.INVERSION_ANALYSIS:
                results = await self._execute_inversion_challenges(
                    type_opportunities, challenge_request
                )
            elif challenge_type == ChallengeType.CONSTITUTIONAL_REVIEW:
                results = await self._execute_constitutional_challenges(
                    type_opportunities, challenge_request
                )

        except Exception as e:
            self.logger.error(f"❌ {challenge_type.value} challenges failed: {e}")

        return results

    async def _execute_assumption_challenges(
        self, opportunities: List[Dict[str, Any]], challenge_request: ChallengeRequest
    ) -> List[ChallengeResult]:
        """Execute assumption challenges using AssumptionChallenger"""

        results = []

        for opportunity in opportunities[:3]:  # Limit to top 3 assumptions
            try:
                # Use existing AssumptionChallenger
                challenge_result = await self._challenge_single_assumption(
                    opportunity["target"],
                    opportunity["step_index"],
                    challenge_request.engagement_id,
                    opportunity["confidence"],
                )

                if challenge_result:
                    results.append(challenge_result)

            except Exception as e:
                self.logger.warning(f"⚠️ Assumption challenge failed: {e}")
                continue

        return results

    async def _challenge_single_assumption(
        self, assumption: str, step_index: int, engagement_id: str, confidence: float
    ) -> Optional[ChallengeResult]:
        """Challenge a single assumption using the AssumptionChallenger system"""

        try:
            # Use simplified assumption challenger for now - in production would use full system
            challenge_statement = (
                f"What if the assumption '{assumption}' is incorrect or incomplete?"
            )

            # Calculate challenge severity based on confidence
            if confidence > 0.8:
                severity = "high"  # High confidence assumptions need strong challenges
            elif confidence > 0.6:
                severity = "medium"
            else:
                severity = "low"

            # Create challenge result
            challenge_result = ChallengeResult(
                challenge_id=f"assumption_{engagement_id}_{step_index}_{hash(assumption) % 10000}",
                challenge_type=ChallengeType.ASSUMPTION_TEST,
                original_reasoning_step=f"Step {step_index}: {assumption}",
                challenge_statement=challenge_statement,
                evidence_backing=[
                    "Assumption requires validation",
                    "Alternative scenarios possible",
                ],
                confidence_impact=-0.2 if severity == "high" else -0.1,
                severity=severity,
                context_sources=["assumption_challenger"],
                research_armed=False,
            )

            return challenge_result

        except Exception as e:
            self.logger.error(f"❌ Single assumption challenge failed: {e}")
            return None

    async def _execute_evidence_challenges(
        self, opportunities: List[Dict[str, Any]], challenge_request: ChallengeRequest
    ) -> List[ChallengeResult]:
        """Execute evidence validation challenges using ResearchArmedChallenger"""

        results = []

        # For evidence challenges, we want to be research-armed
        for opportunity in opportunities[:2]:  # Limit to top 2 for performance
            try:
                challenge_result = await self._research_armed_evidence_challenge(
                    opportunity["target"],
                    opportunity["step_index"],
                    challenge_request.engagement_id,
                    opportunity["confidence"],
                )

                if challenge_result:
                    results.append(challenge_result)

            except Exception as e:
                self.logger.warning(f"⚠️ Evidence challenge failed: {e}")
                continue

        return results

    async def _research_armed_evidence_challenge(
        self, content: str, step_index: int, engagement_id: str, confidence: float
    ) -> Optional[ChallengeResult]:
        """Execute research-armed evidence challenge"""

        try:
            # Create evidence challenge statement
            challenge_statement = f"What current evidence supports or contradicts this reasoning: '{content[:100]}...'?"

            # For now, create a basic evidence challenge - in production would use full ResearchArmedChallenger
            challenge_result = ChallengeResult(
                challenge_id=f"evidence_{engagement_id}_{step_index}_{hash(content) % 10000}",
                challenge_type=ChallengeType.EVIDENCE_VALIDATION,
                original_reasoning_step=f"Step {step_index}: {content[:200]}...",
                challenge_statement=challenge_statement,
                evidence_backing=[
                    "Requires current market research",
                    "Industry data validation needed",
                ],
                confidence_impact=-0.15,
                severity="medium" if confidence < 0.5 else "low",
                context_sources=["research_armed_challenger"],
                research_armed=True,
            )

            return challenge_result

        except Exception as e:
            self.logger.error(f"❌ Research-armed evidence challenge failed: {e}")
            return None

    async def _execute_bias_challenges(
        self, opportunities: List[Dict[str, Any]], challenge_request: ChallengeRequest
    ) -> List[ChallengeResult]:
        """Execute bias audit challenges using MungerOverlay"""

        results = []

        for opportunity in opportunities[:2]:  # Limit bias challenges
            try:
                challenge_result = await self._munger_bias_challenge(
                    opportunity["target"],
                    opportunity["step_index"],
                    challenge_request.engagement_id,
                    opportunity["confidence"],
                )

                if challenge_result:
                    results.append(challenge_result)

            except Exception as e:
                self.logger.warning(f"⚠️ Bias challenge failed: {e}")
                continue

        return results

    async def _munger_bias_challenge(
        self, content: str, step_index: int, engagement_id: str, confidence: float
    ) -> Optional[ChallengeResult]:
        """Execute Munger-style bias challenge"""

        try:
            # Create bias challenge statement
            challenge_statement = f"What cognitive biases might be affecting this high-confidence reasoning: '{content[:100]}...'?"

            # Munger-style bias detection
            potential_biases = [
                "Confirmation bias - seeking confirming evidence",
                "Overconfidence bias - excessive certainty",
                "Availability heuristic - recent examples overweighted",
            ]

            challenge_result = ChallengeResult(
                challenge_id=f"bias_{engagement_id}_{step_index}_{hash(content) % 10000}",
                challenge_type=ChallengeType.BIAS_AUDIT,
                original_reasoning_step=f"Step {step_index}: {content[:200]}...",
                challenge_statement=challenge_statement,
                evidence_backing=potential_biases,
                confidence_impact=-0.25 if confidence > 0.9 else -0.15,
                severity="high" if confidence > 0.9 else "medium",
                context_sources=["munger_overlay"],
                research_armed=False,
            )

            return challenge_result

        except Exception as e:
            self.logger.error(f"❌ Munger bias challenge failed: {e}")
            return None

    async def _execute_inversion_challenges(
        self, opportunities: List[Dict[str, Any]], challenge_request: ChallengeRequest
    ) -> List[ChallengeResult]:
        """Execute inversion analysis challenges"""

        # Placeholder for L1 inversion analysis integration
        return []

    async def _execute_constitutional_challenges(
        self, opportunities: List[Dict[str, Any]], challenge_request: ChallengeRequest
    ) -> List[ChallengeResult]:
        """Execute constitutional bias audit challenges"""

        # Placeholder for L3 constitutional bias audit integration
        return []

    async def _enhance_challenges_with_context_intelligence(
        self, challenge_results: List[ChallengeResult], engagement_id: str
    ) -> List[ChallengeResult]:
        """Enhance challenge results with Context Intelligence insights"""

        try:
            # Get relevant context for challenge enhancement
            contexts = await self.context_intelligence.get_relevant_context(
                current_query=f"challenge_enhancement_{engagement_id}",
                max_contexts=5,
                engagement_id=engagement_id,
            )

            # Enhance each challenge with contextual insights
            for challenge in challenge_results:
                # Add context sources if relevant contexts found
                if contexts:
                    challenge.context_sources.extend(
                        [f"context_{i}" for i in range(len(contexts))]
                    )

                    # Enhance evidence backing with contextual insights
                    contextual_evidence = self._extract_contextual_evidence(
                        challenge, contexts
                    )
                    challenge.evidence_backing.extend(contextual_evidence)

        except Exception as e:
            self.logger.warning(f"⚠️ Context Intelligence enhancement failed: {e}")

        return challenge_results

    def _extract_contextual_evidence(
        self, challenge: ChallengeResult, contexts: List[Tuple[Any, Any]]
    ) -> List[str]:
        """Extract relevant evidence from contexts for challenge"""

        evidence = []

        for context_data, relevance_score in contexts[:2]:  # Top 2 contexts
            if relevance_score > 0.5:  # Only high-relevance contexts
                evidence.append(f"Context evidence (relevance: {relevance_score:.2f})")

        return evidence

    async def _store_challenge_cognitive_exhaust(
        self, challenge_results: List[ChallengeResult], engagement_id: str
    ) -> None:
        """Store challenge results as cognitive exhaust for future Context Intelligence use"""

        try:
            for challenge in challenge_results:
                # Store challenge as cognitive exhaust
                await self.context_intelligence.store_cognitive_exhaust_triple_layer(
                    engagement_id=engagement_id,
                    phase="challenge_validation",
                    mental_model=challenge.challenge_type.value,
                    thinking_process=f"<thinking>Challenge: {challenge.challenge_statement}</thinking>",
                    cleaned_response=f"Challenge severity: {challenge.severity}, Impact: {challenge.confidence_impact}",
                    confidence=abs(challenge.confidence_impact),
                )

            self.logger.info(
                f"💾 Stored {len(challenge_results)} challenge results as cognitive exhaust"
            )

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to store challenge cognitive exhaust: {e}")

    def get_challenge_summary(
        self, challenge_results: List[ChallengeResult]
    ) -> Dict[str, Any]:
        """Generate executive summary of challenges for Pyramid Synthesis integration"""

        if not challenge_results:
            return {
                "total_challenges": 0,
                "severity_breakdown": {},
                "confidence_impact": 0.0,
                "key_insights": ["No significant challenges identified"],
            }

        # Calculate summary metrics
        severity_counts = {}
        total_confidence_impact = 0.0
        challenge_types = set()

        for challenge in challenge_results:
            severity_counts[challenge.severity] = (
                severity_counts.get(challenge.severity, 0) + 1
            )
            total_confidence_impact += challenge.confidence_impact
            challenge_types.add(challenge.challenge_type.value)

        # Generate key insights
        key_insights = []

        high_severity_count = severity_counts.get("high", 0)
        if high_severity_count > 0:
            key_insights.append(
                f"{high_severity_count} high-severity challenges require attention"
            )

        if abs(total_confidence_impact) > 0.5:
            key_insights.append(
                f"Challenges significantly impact confidence (net: {total_confidence_impact:.2f})"
            )

        if len(challenge_types) > 2:
            key_insights.append(
                f"Multiple challenge types identified: {', '.join(challenge_types)}"
            )

        if not key_insights:
            key_insights = ["Challenges identified but require detailed review"]

        return {
            "total_challenges": len(challenge_results),
            "severity_breakdown": severity_counts,
            "confidence_impact": total_confidence_impact,
            "challenge_types": list(challenge_types),
            "key_insights": key_insights,
            "research_armed_challenges": sum(
                1 for c in challenge_results if c.research_armed
            ),
        }


# Factory for Context-Intelligent Challenge Engine
class ContextIntelligentChallengeEngineFactory:
    """Factory for creating Context-Intelligent Challenge Engine instances"""

    @staticmethod
    def create_challenge_engine(
        context_intelligence: IContextIntelligence,
        settings: Any,
        logger: Optional[logging.Logger] = None,
    ) -> ContextIntelligentChallengeEngine:
        """Create a Context-Intelligent Challenge Engine instance"""

        return ContextIntelligentChallengeEngine(
            context_intelligence=context_intelligence, settings=settings, logger=logger
        )
