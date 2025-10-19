"""
Assumption Challenge Engine
Targeted implementation from Ackoff's Problem Dissolution methodology

Focuses on identifying and challenging hidden assumptions that:
1. May be preventing real problem solving
2. Are causing problem recurrence
3. Are based on outdated or incorrect beliefs
4. Are limiting solution possibilities

Key Ackoff principles implemented:
- Challenge fundamental assumptions
- Question system boundaries
- Examine stakeholder objectives
- Look for constraint dissolution opportunities
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# LLM integration with Afterburner optimization
try:
    from ..core.llm_integration_adapter import get_unified_llm_adapter
    from ..config.afterburner_migration import (
        should_use_afterburner_for_component,
        report_afterburner_result,
    )

    AFTERBURNER_AVAILABLE = True
except ImportError:
    AFTERBURNER_AVAILABLE = False
    # Fallback to legacy Claude if Afterburner not available
    try:
        from ..integrations.claude_client import get_claude_client, LLMCallType

        CLAUDE_AVAILABLE = True
    except ImportError:
        CLAUDE_AVAILABLE = False

try:
    from ..config import get_cognitive_settings

    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

from ..models.data_contracts import MetisDataContract

# Research-armed challenger integration
from ..intelligence.research_armed_challenger import (
    ChallengeResearchMode,
    get_research_armed_challenger,
)

logger = logging.getLogger(__name__)


class AssumptionType(Enum):
    """Types of assumptions to challenge"""

    CONSTRAINT = "constraint"  # "We can't do X because..."
    REQUIREMENT = "requirement"  # "We must do Y because..."
    RELATIONSHIP = "relationship"  # "X always leads to Y"
    BOUNDARY = "boundary"  # "This is outside our scope"
    CAPABILITY = "capability"  # "We don't have the ability to..."
    TIMELINE = "timeline"  # "This always takes X time"
    RESOURCE = "resource"  # "We need X resources"
    STAKEHOLDER = "stakeholder"  # "Stakeholder Z wants..."


@dataclass
class ChallengedAssumption:
    """An assumption that has been identified and challenged"""

    assumption: str
    assumption_type: AssumptionType
    evidence_for: List[str]  # Evidence supporting the assumption
    evidence_against: List[str]  # Evidence challenging the assumption
    alternative_possibilities: List[str]  # What becomes possible if assumption is false
    dissolution_potential: float  # 0.0-1.0 score for problem dissolution potential
    validation_questions: List[str]  # Questions to validate/invalidate assumption


@dataclass
class AssumptionAnalysis:
    """Complete assumption analysis for a problem"""

    original_problem: str
    identified_assumptions: List[ChallengedAssumption]
    high_impact_assumptions: List[ChallengedAssumption]  # Top 3-5 for action
    dissolution_opportunities: List[str]
    reframed_problem: Optional[str]
    analysis_confidence: float
    analysis_time: float


class AssumptionChallenger:
    """
    Systematically identifies and challenges assumptions using Ackoff's approach.

    Focused implementation that:
    1. Identifies 5-10 key assumptions (not exhaustive list)
    2. Prioritizes assumptions with high dissolution potential
    3. Provides concrete validation questions
    4. Suggests problem reframing when appropriate
    """

    def __init__(
        self, research_mode: ChallengeResearchMode = ChallengeResearchMode.AUTO
    ):
        self.logger = logging.getLogger(__name__)

        # Initialize LLM adapter with Afterburner optimization
        if AFTERBURNER_AVAILABLE:
            self.llm_adapter = get_unified_llm_adapter()
            self.use_afterburner = True
            self.logger.info("🚀 Assumption Challenger using Afterburner optimization")
        else:
            self.llm_adapter = None
            self.use_afterburner = False
            self.claude_client = get_claude_client() if CLAUDE_AVAILABLE else None

        # Research-armed challenger integration
        self.research_challenger = get_research_armed_challenger(
            research_mode=research_mode,
            confidence_threshold=0.6,  # Moderate threshold for assumption challenging
        )
        self.research_enabled = research_mode != ChallengeResearchMode.OFF

        # Load configuration
        if CONFIG_AVAILABLE:
            self.settings = get_cognitive_settings()
            self.max_challenges = self.settings.ASSUMPTION_MAX_CHALLENGES_PER_SESSION
            self.min_dissolution_potential = (
                self.settings.ASSUMPTION_MIN_DISSOLUTION_POTENTIAL
            )
            self.analysis_timeout = self.settings.ASSUMPTION_ANALYSIS_TIMEOUT_SECONDS
        else:
            # Fallback defaults
            self.max_challenges = 8
            self.min_dissolution_potential = 0.5
            self.analysis_timeout = 45

        if self.research_enabled:
            self.logger.info(
                f"🔍 Assumption Challenger initialized with research mode: {research_mode.value}"
            )

        # Common assumption patterns to look for
        self.assumption_patterns = {
            AssumptionType.CONSTRAINT: [
                "can't",
                "cannot",
                "impossible",
                "unable to",
                "restricted",
                "limited by",
            ],
            AssumptionType.REQUIREMENT: [
                "must",
                "have to",
                "required",
                "necessary",
                "need to",
                "should",
            ],
            AssumptionType.RELATIONSHIP: [
                "always leads to",
                "causes",
                "results in",
                "means",
                "implies",
            ],
            AssumptionType.BOUNDARY: [
                "outside scope",
                "not our responsibility",
                "beyond our control",
            ],
            AssumptionType.CAPABILITY: [
                "don't have",
                "lack the",
                "missing",
                "not equipped",
                "inexperienced",
            ],
            AssumptionType.TIMELINE: [
                "takes",
                "requires",
                "needs time",
                "timeline",
                "schedule",
            ],
            AssumptionType.RESOURCE: [
                "need resources",
                "budget",
                "funding",
                "cost",
                "expensive",
            ],
            AssumptionType.STAKEHOLDER: [
                "expects",
                "wants",
                "demands",
                "requires from us",
            ],
        }

    async def challenge_assumptions(
        self, contract: MetisDataContract
    ) -> AssumptionAnalysis:
        """
        Identify and challenge key assumptions in the problem.

        Args:
            contract: The problem contract

        Returns:
            AssumptionAnalysis with challenged assumptions and opportunities
        """
        start_time = asyncio.get_event_loop().time()

        try:
            # Step 1: Identify potential assumptions in problem statement
            potential_assumptions = await self._identify_assumptions(contract)

            # Step 2: Challenge each assumption systematically
            challenged_assumptions = []
            for assumption in potential_assumptions:
                challenged = await self._challenge_single_assumption(
                    assumption, contract
                )
                if challenged:
                    challenged_assumptions.append(challenged)

            # Step 3: Prioritize by dissolution potential
            high_impact = self._prioritize_assumptions(challenged_assumptions)

            # Step 4: Generate dissolution opportunities
            dissolution_opportunities = await self._identify_dissolution_opportunities(
                high_impact, contract
            )

            # Step 5: Attempt problem reframing
            reframed_problem = await self._attempt_reframing(high_impact, contract)

            analysis_time = asyncio.get_event_loop().time() - start_time

            analysis = AssumptionAnalysis(
                original_problem=contract.problem_statement,
                identified_assumptions=challenged_assumptions,
                high_impact_assumptions=high_impact[:5],  # Top 5 only
                dissolution_opportunities=dissolution_opportunities,
                reframed_problem=reframed_problem,
                analysis_confidence=await self._calculate_confidence(
                    challenged_assumptions
                ),
                analysis_time=analysis_time,
            )

            self.logger.info(
                f"Challenged {len(challenged_assumptions)} assumptions in {analysis_time:.2f}s"
            )
            return analysis

        except Exception as e:
            self.logger.error(f"Assumption challenging failed: {e}")
            return AssumptionAnalysis(
                original_problem=contract.problem_statement,
                identified_assumptions=[],
                high_impact_assumptions=[],
                dissolution_opportunities=[],
                reframed_problem=None,
                analysis_confidence=0.0,
                analysis_time=0,
            )

    async def _identify_assumptions(
        self, contract: MetisDataContract
    ) -> List[Tuple[str, AssumptionType]]:
        """Identify potential assumptions in the problem statement."""

        problem_text = contract.problem_statement
        business_context = getattr(contract, "business_context", {})

        identified = []

        # Pattern-based assumption detection
        for assumption_type, patterns in self.assumption_patterns.items():
            for pattern in patterns:
                if pattern.lower() in problem_text.lower():
                    # Extract context around the pattern
                    context = self._extract_assumption_context(problem_text, pattern)
                    if context:
                        identified.append((context, assumption_type))

        # LLM-based deeper assumption identification
        if (self.use_afterburner or self.claude_client) and len(identified) < 5:
            llm_assumptions = await self._identify_hidden_assumptions_llm(contract)
            identified.extend(llm_assumptions)

        # Remove duplicates and limit to reasonable number
        seen = set()
        unique_assumptions = []
        for assumption, type_ in identified:
            if assumption not in seen:
                seen.add(assumption)
                unique_assumptions.append((assumption, type_))

        return unique_assumptions[:8]  # Max 8 assumptions to keep focused

    def _extract_assumption_context(self, text: str, pattern: str) -> Optional[str]:
        """Extract the assumption context around a pattern match."""

        text_lower = text.lower()
        pattern_index = text_lower.find(pattern.lower())

        if pattern_index == -1:
            return None

        # Extract sentence containing the pattern
        start = max(0, text.rfind(".", 0, pattern_index) + 1)
        end = text.find(".", pattern_index + len(pattern))
        if end == -1:
            end = len(text)

        assumption_sentence = text[start:end].strip()

        # Only return if it's substantial enough
        if len(assumption_sentence) > 20:
            return assumption_sentence

        return None

    async def _identify_hidden_assumptions_llm(
        self, contract: MetisDataContract
    ) -> List[Tuple[str, AssumptionType]]:
        """Use LLM to identify hidden assumptions not caught by patterns."""

        if not self.use_afterburner and not self.claude_client:
            return []

        try:
            prompt = f"""
You are an expert in Ackoff's problem dissolution methodology. Your job is to identify hidden assumptions that might be preventing real problem solving.

PROBLEM: {contract.problem_statement}

BUSINESS CONTEXT: {getattr(contract, 'business_context', {})}

Identify 3-5 key hidden assumptions that might be embedded in this problem statement. Look for:
1. Unstated constraints or boundaries
2. Implicit requirements or "must haves" 
3. Assumed relationships (X always leads to Y)
4. Capability limitations taken for granted
5. Stakeholder expectations that aren't verified

For each assumption, classify it as: constraint, requirement, relationship, boundary, capability, timeline, resource, or stakeholder.

Format response as:
ASSUMPTION: [assumption statement]
TYPE: [one of the types above]
WHY_HIDDEN: [why this assumption is not obvious]

Limit to 5 assumptions maximum.
"""

            # Use Afterburner optimization if available
            if self.use_afterburner:
                response = await self.llm_adapter.call_llm_unified(
                    prompt=prompt,
                    task_name="assumption_challenge",
                    business_context={"problem": contract.problem_statement},
                    engagement_id=getattr(contract, "engagement_id", None),
                    phase="assumption_identification",
                    max_tokens=400,
                    temperature=0.3,
                )
            else:
                # Legacy Claude fallback
                response = await self.claude_client.achat(
                    prompt,
                    call_type=LLMCallType.ASSUMPTION_IDENTIFICATION,
                    max_tokens=400,
                )

            return self._parse_llm_assumptions(response)

        except Exception as e:
            self.logger.warning(f"LLM assumption identification failed: {e}")
            return []

    def _parse_llm_assumptions(self, response: str) -> List[Tuple[str, AssumptionType]]:
        """Parse LLM response into assumption tuples."""

        assumptions = []
        lines = response.strip().split("\n")

        current_assumption = None
        current_type = None

        for line in lines:
            line = line.strip()
            if line.startswith("ASSUMPTION:"):
                current_assumption = line[11:].strip()
            elif line.startswith("TYPE:"):
                type_str = line[5:].strip().lower()
                try:
                    current_type = AssumptionType(type_str)
                except ValueError:
                    current_type = AssumptionType.CONSTRAINT  # default

                # Store completed assumption
                if current_assumption and current_type:
                    assumptions.append((current_assumption, current_type))
                    current_assumption = None
                    current_type = None

        return assumptions

    async def _challenge_single_assumption(
        self, assumption_info: Tuple[str, AssumptionType], contract: MetisDataContract
    ) -> Optional[ChallengedAssumption]:
        """Challenge a single assumption systematically."""

        assumption, assumption_type = assumption_info

        if self.use_afterburner or self.claude_client:
            return await self._challenge_assumption_llm(
                assumption, assumption_type, contract
            )
        else:
            return self._challenge_assumption_fallback(assumption, assumption_type)

    async def _challenge_assumption_llm(
        self,
        assumption: str,
        assumption_type: AssumptionType,
        contract: MetisDataContract,
    ) -> Optional[ChallengedAssumption]:
        """Use LLM to systematically challenge an assumption."""

        try:
            prompt = f"""
You are systematically challenging an assumption using Ackoff's dissolution methodology.

ORIGINAL PROBLEM: {contract.problem_statement}

ASSUMPTION TO CHALLENGE: {assumption}
ASSUMPTION TYPE: {assumption_type.value}

Systematically challenge this assumption by:

1. EVIDENCE FOR: What evidence supports this assumption being true?
2. EVIDENCE AGAINST: What evidence or cases challenge this assumption?
3. ALTERNATIVES: What becomes possible if this assumption is false?
4. DISSOLUTION: How could removing this assumption dissolve the original problem? (0.0-1.0 score)
5. VALIDATION: What specific questions could test if this assumption is actually valid?

Be concrete and specific. Focus on business reality, not philosophical speculation.

Format response as:
EVIDENCE_FOR: [2-3 points]
EVIDENCE_AGAINST: [2-3 points]  
ALTERNATIVES: [2-3 new possibilities]
DISSOLUTION_SCORE: [0.0-1.0]
VALIDATION_QUESTIONS: [2-3 specific testable questions]
"""

            # Use Afterburner optimization if available
            if self.use_afterburner:
                response = await self.llm_adapter.call_llm_unified(
                    prompt=prompt,
                    task_name="assumption_challenge",
                    business_context={
                        "problem": contract.problem_statement,
                        "assumption_type": assumption_type.value,
                    },
                    engagement_id=getattr(contract, "engagement_id", None),
                    phase="assumption_challenging",
                    max_tokens=400,
                    temperature=0.3,
                )
            else:
                # Legacy Claude fallback
                response = await self.claude_client.achat(
                    prompt, call_type=LLMCallType.ASSUMPTION_CHALLENGING, max_tokens=400
                )

            return self._parse_challenged_assumption(
                assumption, assumption_type, response
            )

        except Exception as e:
            self.logger.warning(
                f"LLM assumption challenging failed for '{assumption}': {e}"
            )
            return self._challenge_assumption_fallback(assumption, assumption_type)

    def _parse_challenged_assumption(
        self, assumption: str, assumption_type: AssumptionType, response: str
    ) -> ChallengedAssumption:
        """Parse LLM response into ChallengedAssumption structure."""

        evidence_for = []
        evidence_against = []
        alternatives = []
        dissolution_score = 0.5
        validation_questions = []

        lines = response.strip().split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith("EVIDENCE_FOR:"):
                current_section = "for"
                content = line[13:].strip()
                if content:
                    evidence_for.append(content)
            elif line.startswith("EVIDENCE_AGAINST:"):
                current_section = "against"
                content = line[17:].strip()
                if content:
                    evidence_against.append(content)
            elif line.startswith("ALTERNATIVES:"):
                current_section = "alternatives"
                content = line[13:].strip()
                if content:
                    alternatives.append(content)
            elif line.startswith("DISSOLUTION_SCORE:"):
                try:
                    dissolution_score = float(line[18:].strip())
                except ValueError:
                    dissolution_score = 0.5
            elif line.startswith("VALIDATION_QUESTIONS:"):
                current_section = "validation"
                content = line[21:].strip()
                if content:
                    validation_questions.append(content)
            elif line and current_section:
                # Continuation line
                if current_section == "for":
                    evidence_for.append(line)
                elif current_section == "against":
                    evidence_against.append(line)
                elif current_section == "alternatives":
                    alternatives.append(line)
                elif current_section == "validation":
                    validation_questions.append(line)

        return ChallengedAssumption(
            assumption=assumption,
            assumption_type=assumption_type,
            evidence_for=evidence_for[:3],  # Limit to 3
            evidence_against=evidence_against[:3],
            alternative_possibilities=alternatives[:3],
            dissolution_potential=max(0.0, min(1.0, dissolution_score)),
            validation_questions=validation_questions[:3],
        )

    def _challenge_assumption_fallback(
        self, assumption: str, assumption_type: AssumptionType
    ) -> ChallengedAssumption:
        """Fallback assumption challenging when LLM unavailable."""

        # Simple rule-based challenging
        evidence_against = []
        alternatives = []
        validation_questions = []

        if assumption_type == AssumptionType.CONSTRAINT:
            evidence_against = [
                "Constraints may be self-imposed",
                "Technology or methods may have changed",
            ]
            alternatives = ["Remove the constraint", "Work around the constraint"]
            validation_questions = [
                "Is this constraint still valid?",
                "Who imposed this constraint?",
            ]

        elif assumption_type == AssumptionType.REQUIREMENT:
            evidence_against = [
                "Requirements may be wants not needs",
                "Original reason may no longer apply",
            ]
            alternatives = [
                "Challenge the requirement",
                "Find alternative ways to meet the need",
            ]
            validation_questions = [
                "Why is this required?",
                "What happens if we don't do this?",
            ]

        else:
            evidence_against = [
                "Assumption may be outdated",
                "May not apply in all contexts",
            ]
            alternatives = ["Challenge the assumption", "Test alternative approaches"]
            validation_questions = [
                "Is this assumption still valid?",
                "When was this last verified?",
            ]

        return ChallengedAssumption(
            assumption=assumption,
            assumption_type=assumption_type,
            evidence_for=["Assumption appears in problem statement"],
            evidence_against=evidence_against,
            alternative_possibilities=alternatives,
            dissolution_potential=0.6,
            validation_questions=validation_questions,
        )

    def _prioritize_assumptions(
        self, assumptions: List[ChallengedAssumption]
    ) -> List[ChallengedAssumption]:
        """Prioritize assumptions by dissolution potential and impact."""

        return sorted(
            assumptions,
            key=lambda a: (
                a.dissolution_potential,
                len(a.alternative_possibilities),
                len(a.evidence_against),
            ),
            reverse=True,
        )

    async def _identify_dissolution_opportunities(
        self, high_impact: List[ChallengedAssumption], contract: MetisDataContract
    ) -> List[str]:
        """Identify specific opportunities for problem dissolution."""

        opportunities = []

        for assumption in high_impact[:3]:  # Top 3 only
            if assumption.dissolution_potential > 0.7:
                for alternative in assumption.alternative_possibilities:
                    opportunity = f"If we {alternative.lower()}, we could potentially dissolve the constraint: '{assumption.assumption}'"
                    opportunities.append(opportunity)

        return opportunities[:5]  # Limit to 5 opportunities

    async def _attempt_reframing(
        self, high_impact: List[ChallengedAssumption], contract: MetisDataContract
    ) -> Optional[str]:
        """Attempt to reframe the problem by removing key assumptions."""

        if not high_impact or (not self.use_afterburner and not self.claude_client):
            return None

        # Only attempt reframing if we have high-dissolution-potential assumptions
        top_assumption = high_impact[0]
        if top_assumption.dissolution_potential < 0.8:
            return None

        try:
            prompt = f"""
Given this problem and the challenged assumption, suggest a reframed problem statement.

ORIGINAL PROBLEM: {contract.problem_statement}

TOP CHALLENGED ASSUMPTION: {top_assumption.assumption}
ALTERNATIVES IF FALSE: {', '.join(top_assumption.alternative_possibilities)}

Reframe the original problem by removing or modifying this assumption. The reframed problem should:
1. Be concrete and actionable
2. Open up new solution possibilities  
3. Address the underlying need without the constraining assumption

REFRAMED PROBLEM: [your reframing]
"""

            # Use Afterburner optimization if available
            if self.use_afterburner:
                response = await self.llm_adapter.call_llm_unified(
                    prompt=prompt,
                    task_name="assumption_challenge",
                    business_context={"problem": contract.problem_statement},
                    engagement_id=getattr(contract, "engagement_id", None),
                    phase="problem_reframing",
                    max_tokens=150,
                    temperature=0.3,
                )
            else:
                # Legacy Claude fallback
                response = await self.claude_client.achat(
                    prompt, call_type=LLMCallType.PROBLEM_REFRAMING, max_tokens=150
                )

            reframed = response.strip()
            if reframed.startswith("REFRAMED PROBLEM:"):
                reframed = reframed[17:].strip()

            return reframed if len(reframed) > 20 else None

        except Exception as e:
            self.logger.warning(f"Problem reframing failed: {e}")
            return None

    async def _calculate_confidence(
        self, assumptions: List[ChallengedAssumption]
    ) -> float:
        """Calculate confidence in the assumption analysis."""

        if not assumptions:
            return 0.0

        # Confidence based on:
        # - Number of assumptions identified
        # - Quality of evidence (length of lists)
        # - Dissolution potential scores

        quantity_score = min(len(assumptions) / 5.0, 1.0)  # Optimal is 5 assumptions

        quality_score = 0.0
        if assumptions:
            avg_evidence_quality = (
                sum(
                    len(a.evidence_for)
                    + len(a.evidence_against)
                    + len(a.alternative_possibilities)
                    for a in assumptions
                )
                / len(assumptions)
                / 9.0
            )  # Max 9 items per assumption

            avg_dissolution_potential = sum(
                a.dissolution_potential for a in assumptions
            ) / len(assumptions)

            quality_score = (avg_evidence_quality + avg_dissolution_potential) / 2.0

        return quantity_score * 0.3 + quality_score * 0.7

    def integrate_with_mece(
        self, assumption_analysis: AssumptionAnalysis, mece_structure: Dict
    ) -> Dict:
        """
        Integrate assumption analysis with existing MECE structure.

        This adds assumption challenging without disrupting the original structure.
        """
        enhanced_structure = mece_structure.copy()

        enhanced_structure["assumption_challenges"] = {
            "total_assumptions": len(assumption_analysis.identified_assumptions),
            "high_impact_assumptions": [
                {
                    "assumption": a.assumption,
                    "type": a.assumption_type.value,
                    "dissolution_potential": a.dissolution_potential,
                    "key_questions": a.validation_questions[:2],  # Top 2 questions
                }
                for a in assumption_analysis.high_impact_assumptions
            ],
            "dissolution_opportunities": assumption_analysis.dissolution_opportunities,
            "reframed_problem": assumption_analysis.reframed_problem,
            "analysis_confidence": assumption_analysis.analysis_confidence,
        }

        return enhanced_structure


def get_assumption_challenger() -> AssumptionChallenger:
    """Get singleton assumption challenger instance."""
    if not hasattr(get_assumption_challenger, "_instance"):
        get_assumption_challenger._instance = AssumptionChallenger()
    return get_assumption_challenger._instance
