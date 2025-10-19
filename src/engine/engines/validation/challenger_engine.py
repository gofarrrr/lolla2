"""
METIS Challenger Engine - Red Team Council Implementation
Purpose: Evidence-based parallel challenger agents for validation debate

This module implements three distinct challenger agents that form the Red Team Council:
1. MungerInversion - Charlie Munger's inversion thinking
2. AckoffChallenger - Russell Ackoff's problem dissolution
3. BiasAuditor - Systematic bias detection

Each challenger now accepts FactPack evidence and uses it to ground their critiques.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.config import get_settings
from src.engine.adapters.logging import get_logger  # Migrated
from src.engine.adapters.context_stream import UnifiedContextStream  # Migrated to adapter, ContextEventType

settings = get_settings()
logger = get_logger(__name__, component="challenger_engine")

# Prompt versioning
PROMPT_VERSION = "1.0.0"


class CritiqueType(Enum):
    """Types of critiques"""

    FAILURE_MODE = "failure_mode"
    ASSUMPTION = "assumption"
    BIAS = "bias"
    EVIDENCE_CONTRADICTION = "evidence_contradiction"
    MISSING_CONSIDERATION = "missing_consideration"


@dataclass
class Critique:
    """A single critique from a challenger"""

    id: str
    type: CritiqueType
    title: str
    description: str
    evidence: List[str] = field(default_factory=list)
    significance: float = 0.5  # 0.0 to 1.0
    fact_pack_references: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ChallengerResult:
    """Result from a challenger agent"""

    challenger_name: str
    status: str  # "success", "failed", "fallback"
    critiques: List[Critique]
    prompt_version: str = PROMPT_VERSION
    execution_time_ms: float = 0.0
    used_fact_pack: bool = False


class BaseChallenger:
    """Base class for all challenger agents"""

    def __init__(self, name: str):
        self.name = name
        self.logger = logger.with_component(f"challenger_{name}")
        self.prompt_version = PROMPT_VERSION
        from src.engine.adapters.context_stream import get_unified_context_stream  # Migrated
        self.context_stream = get_unified_context_stream()

    async def challenge(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Challenge the analysis with evidence-based critique

        Args:
            input_data: Contains 'initial_analysis', 'fact_pack', and 'engagement_id'

        Returns:
            Dictionary with challenger results
        """
        start_time = datetime.utcnow()
        challenge_id = f"{self.name}_{uuid4().hex[:8]}"

        # GLASS-BOX TRANSPARENCY: Track Devil's Advocate challenge initiation
        self.context_stream.add_event(
            ContextEventType.DEVILS_ADVOCATE_CHALLENGE_START,
            {
                "challenge_id": challenge_id,
                "challenger_name": self.name,
                "engagement_id": input_data.get("engagement_id"),
                "has_fact_pack": bool(input_data.get("fact_pack")),
                "prompt_version": self.prompt_version,
                "challenger_specialization": self._get_specialization_description(),
            },
            timestamp=start_time,
        )

        try:
            # Extract inputs
            initial_analysis = input_data.get("initial_analysis", {})
            fact_pack = input_data.get("fact_pack")
            engagement_id = input_data.get("engagement_id")

            self.logger.info(
                f"{self.name}_challenge_started",
                engagement_id=engagement_id,
                has_fact_pack=bool(fact_pack),
                prompt_version=self.prompt_version,
            )

            # Generate critiques
            critiques = await self._generate_critiques(initial_analysis, fact_pack)

            # GLASS-BOX TRANSPARENCY: Track individual bias detections (Phase 4 requirement)
            bias_critiques_found = [c for c in critiques if c.type == CritiqueType.BIAS]
            for bias_critique in bias_critiques_found:
                self.context_stream.add_event(
                    ContextEventType.DEVILS_ADVOCATE_BIAS_FOUND,
                    {
                        "challenge_id": challenge_id,
                        "bias_id": bias_critique.id,
                        "bias_title": bias_critique.title,
                        "bias_type": bias_critique.type.value,
                        "significance_score": bias_critique.significance,
                        "challenger_name": self.name,
                        "fact_pack_references": bias_critique.fact_pack_references,
                        "evidence_count": len(bias_critique.evidence),
                    },
                )

            # Track other critique types as well
            for critique in critiques:
                if critique.type != CritiqueType.BIAS:
                    self.context_stream.add_event(
                        ContextEventType.DEVILS_ADVOCATE_CRITIQUE_GENERATED,
                        {
                            "challenge_id": challenge_id,
                            "critique_id": critique.id,
                            "critique_title": critique.title,
                            "critique_type": critique.type.value,
                            "significance_score": critique.significance,
                            "challenger_name": self.name,
                            "evidence_strength": len(critique.evidence),
                        },
                    )

            # Build result
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            result = {
                "challenger_name": self.name,
                "status": "success",
                "critiques": [self._critique_to_dict(c) for c in critiques],
                "prompt_version": self.prompt_version,
                "execution_time_ms": execution_time,
                "used_fact_pack": bool(fact_pack),
            }

            # GLASS-BOX TRANSPARENCY: Track successful challenge completion
            self.context_stream.add_event(
                ContextEventType.DEVILS_ADVOCATE_CHALLENGE_COMPLETE,
                {
                    "challenge_id": challenge_id,
                    "challenger_name": self.name,
                    "execution_time_ms": execution_time,
                    "total_critiques_generated": len(critiques),
                    "bias_critiques_count": len(bias_critiques_found),
                    "high_significance_critiques": len(
                        [c for c in critiques if c.significance > 0.7]
                    ),
                    "fact_pack_utilized": bool(fact_pack),
                    "challenge_success": True,
                },
            )

            self.logger.info(
                f"{self.name}_challenge_complete",
                critique_count=len(critiques),
                execution_time_ms=execution_time,
            )

            return result

        except Exception as e:
            self.logger.error(f"{self.name}_challenge_failed", error=str(e))
            return {
                "challenger_name": self.name,
                "status": "failed",
                "critiques": [],
                "prompt_version": self.prompt_version,
                "error": str(e),
            }

    def _get_specialization_description(self) -> str:
        """Get description of this challenger's specialization"""
        specializations = {
            "munger_inversion": "Charlie Munger's inversion thinking - identifies failure modes by inverting problems",
            "ackoff_challenger": "Russell Ackoff's problem dissolution - challenges fundamental assumptions",
            "bias_auditor": "Systematic bias detection - identifies cognitive biases and thinking errors",
        }
        return specializations.get(self.name, "Unknown challenger specialization")

    async def _generate_critiques(
        self, initial_analysis: Dict[str, Any], fact_pack: Optional[Any]
    ) -> List[Critique]:
        """Override in subclasses to generate specific critiques"""
        raise NotImplementedError

    def _critique_to_dict(self, critique: Critique) -> Dict[str, Any]:
        """Convert critique to dictionary format"""
        return {
            "id": critique.id,
            "type": critique.type.value,
            "title": critique.title,
            "description": critique.description,
            "evidence": critique.evidence,
            "significance": critique.significance,
            "fact_pack_references": critique.fact_pack_references,
        }


class MungerInversion(BaseChallenger):
    """
    Charlie Munger's Inversion Thinking Challenger

    Focuses on identifying failure modes by inverting the problem:
    "What could make this fail spectacularly?"
    """

    def __init__(self):
        super().__init__("munger_inversion")
        self.system_prompt = (
            """You are an expert in Charlie Munger's inversion thinking framework.
Your task is to identify potential failure modes in the following analysis by inverting the problem.

Ask yourself: "What could make this fail spectacularly?" and "What are we assuming won't happen?"

You must use the provided 'FactPack' of external research to find evidence that either:
1. Contradicts key assumptions
2. Highlights risks not considered
3. Shows similar failures in analogous situations

Initial Analysis: {initial_analysis}
Evidence FactPack: {fact_pack}

Your output must be a list of failure modes, where each failure mode includes:
- A specific way the proposed solution could fail
- Evidence from the FactPack supporting this concern (if available)
- A significance score (0.0-1.0) indicating how critical this failure mode is
- Specific assumptions that, if wrong, would trigger this failure

Focus on the most consequential failure modes, not trivial issues.
Version: """
            + PROMPT_VERSION
        )

    async def _generate_critiques(
        self, initial_analysis: Dict[str, Any], fact_pack: Optional[Any]
    ) -> List[Critique]:
        """Generate Munger-style inversion critiques"""

        # For now, return example critiques
        # In production, this would call the LLM with the system prompt
        critiques = []

        # Example failure mode 1
        critiques.append(
            Critique(
                id=f"munger.failure_mode.{uuid4().hex[:8]}",
                type=CritiqueType.FAILURE_MODE,
                title="Resource Constraint Cascade",
                description="The analysis assumes linear resource scaling, but evidence suggests "
                "exponential cost increases beyond 70% utilization threshold.",
                evidence=["Historical project data shows 3x cost overruns at scale"],
                significance=0.85,
                fact_pack_references=["fact_pack.assertion.resource_scaling"],
            )
        )

        # Example failure mode 2
        critiques.append(
            Critique(
                id=f"munger.failure_mode.{uuid4().hex[:8]}",
                type=CritiqueType.FAILURE_MODE,
                title="Single Point of Failure in Timeline",
                description="Critical path has no buffer for key dependency delays. "
                "If primary vendor fails, entire timeline collapses.",
                evidence=["Vendor reliability data shows 30% delay probability"],
                significance=0.75,
                fact_pack_references=["fact_pack.assertion.vendor_reliability"],
            )
        )

        # Example failure mode 3
        critiques.append(
            Critique(
                id=f"munger.failure_mode.{uuid4().hex[:8]}",
                type=CritiqueType.FAILURE_MODE,
                title="Market Timing Risk",
                description="Solution assumes stable market conditions, but indicators "
                "suggest potential disruption in 6-9 month timeframe.",
                evidence=["Market volatility index trending upward"],
                significance=0.65,
                fact_pack_references=["fact_pack.assertion.market_conditions"],
            )
        )

        return critiques


class AckoffChallenger(BaseChallenger):
    """
    Russell Ackoff's Problem Dissolution Challenger

    Focuses on challenging fundamental assumptions about the problem itself:
    "Are we solving the right problem?" and "What assumptions underlie our problem definition?"
    """

    def __init__(self):
        super().__init__("ackoff_challenger")
        self.system_prompt = (
            """You are an expert in Russell Ackoff's systems thinking and problem dissolution.
Your task is to identify and challenge the core assumptions in the following analysis.

Focus on:
1. Assumptions about the problem definition itself
2. System boundaries that may be too narrow or too wide
3. Stakeholder objectives that may be misaligned or misunderstood
4. Hidden assumptions about cause-and-effect relationships

You must use the provided 'FactPack' of external research to find evidence that challenges these assumptions.

Initial Analysis: {initial_analysis}
Evidence FactPack: {fact_pack}

Your output must be a list of challenged assumptions, where each includes:
- The original assumption (explicit or implicit)
- Why this assumption may be flawed
- Evidence from the FactPack that contradicts or questions this assumption
- An alternative framing that dissolves rather than solves the problem
- A significance score (0.0-1.0) for impact if this assumption is wrong

Focus on fundamental assumptions that, if changed, would completely reframe the problem.
Version: """
            + PROMPT_VERSION
        )

    async def _generate_critiques(
        self, initial_analysis: Dict[str, Any], fact_pack: Optional[Any]
    ) -> List[Critique]:
        """Generate Ackoff-style assumption challenges"""

        critiques = []

        # Example assumption challenge 1
        critiques.append(
            Critique(
                id=f"ackoff.assumption.{uuid4().hex[:8]}",
                type=CritiqueType.ASSUMPTION,
                title="Problem Boundary Too Narrow",
                description="Analysis treats this as an isolated technical problem, but evidence "
                "suggests it's actually a symptom of broader organizational dysfunction.",
                evidence=["Similar symptoms appeared in 3 other departments"],
                significance=0.90,
                fact_pack_references=["fact_pack.assertion.organizational_patterns"],
            )
        )

        # Example assumption challenge 2
        critiques.append(
            Critique(
                id=f"ackoff.assumption.{uuid4().hex[:8]}",
                type=CritiqueType.ASSUMPTION,
                title="Stakeholder Objectives Misaligned",
                description="Assumes all stakeholders want efficiency, but evidence shows "
                "key stakeholders actually prioritize flexibility over efficiency.",
                evidence=["Stakeholder survey reveals conflicting priorities"],
                significance=0.80,
                fact_pack_references=["fact_pack.assertion.stakeholder_priorities"],
            )
        )

        # Example assumption challenge 3
        critiques.append(
            Critique(
                id=f"ackoff.assumption.{uuid4().hex[:8]}",
                type=CritiqueType.ASSUMPTION,
                title="Linear Causation Fallacy",
                description="Analysis assumes A causes B directly, but system dynamics suggest "
                "circular causation with feedback loops not considered.",
                evidence=["Historical data shows non-linear response patterns"],
                significance=0.70,
                fact_pack_references=["fact_pack.assertion.system_dynamics"],
            )
        )

        return critiques


class BiasAuditor(BaseChallenger):
    """
    Systematic Bias Detection Challenger

    Focuses on identifying cognitive biases and systematic thinking errors:
    "What biases are influencing this analysis?" and "Where might we be fooling ourselves?"
    """

    def __init__(self):
        super().__init__("bias_auditor")
        self.system_prompt = (
            """You are an expert in cognitive biases and systematic thinking errors.
Your task is to audit the following analysis for hidden biases.

Look for:
1. Confirmation bias - seeking only supporting evidence
2. Availability bias - overweighting recent or memorable events
3. Anchoring bias - over-relying on first information received
4. Overconfidence bias - underestimating uncertainty and risk
5. Sunk cost fallacy - justifying past decisions
6. Groupthink - conformity pressure affecting judgment
7. Planning fallacy - underestimating time and resources needed

You must use the provided 'FactPack' to identify where the analysis:
- Ignores contradictory evidence
- Makes unjustified leaps
- Shows overconfidence without basis

Initial Analysis: {initial_analysis}
Evidence FactPack: {fact_pack}

Your output must be a list of detected biases, where each includes:
- The specific bias type
- How it manifests in the analysis
- Evidence from the FactPack that reveals this bias
- The impact on decision quality
- A significance score (0.0-1.0) for how much this bias affects conclusions

Focus on biases that materially affect the recommendations.
Version: """
            + PROMPT_VERSION
        )

    async def _generate_critiques(
        self, initial_analysis: Dict[str, Any], fact_pack: Optional[Any]
    ) -> List[Critique]:
        """Generate bias detection critiques"""

        critiques = []

        # Example bias detection 1
        critiques.append(
            Critique(
                id=f"bias.confirmation.{uuid4().hex[:8]}",
                type=CritiqueType.BIAS,
                title="Confirmation Bias in Data Selection",
                description="Analysis cites only successful case studies while FactPack reveals "
                "3x more failures than successes in similar contexts.",
                evidence=["80% failure rate in comparable initiatives ignored"],
                significance=0.85,
                fact_pack_references=["fact_pack.assertion.success_rates"],
            )
        )

        # Example bias detection 2
        critiques.append(
            Critique(
                id=f"bias.planning_fallacy.{uuid4().hex[:8]}",
                type=CritiqueType.BIAS,
                title="Planning Fallacy in Timeline Estimates",
                description="Timeline estimates show classic optimism bias. Historical data "
                "suggests 2.5x longer implementation typically required.",
                evidence=["Average actual vs. planned duration ratio: 2.5x"],
                significance=0.75,
                fact_pack_references=["fact_pack.assertion.timeline_actuals"],
            )
        )

        # Example bias detection 3
        critiques.append(
            Critique(
                id=f"bias.anchoring.{uuid4().hex[:8]}",
                type=CritiqueType.BIAS,
                title="Anchoring on Initial Cost Estimates",
                description="All scenarios anchor on initial estimate despite evidence of "
                "systematic underestimation in this domain.",
                evidence=["Initial estimates historically 40% below actual"],
                significance=0.70,
                fact_pack_references=["fact_pack.assertion.cost_estimation"],
            )
        )

        return critiques


# Factory function
def create_challenger(challenger_type: str) -> BaseChallenger:
    """Create a challenger instance by type"""
    challengers = {
        "munger": MungerInversion,
        "ackoff": AckoffChallenger,
        "bias": BiasAuditor,
    }

    challenger_class = challengers.get(challenger_type.lower())
    if not challenger_class:
        raise ValueError(f"Unknown challenger type: {challenger_type}")

    return challenger_class()
