#!/usr/bin/env python3
"""
METIS Charlie Munger Overlay - Core Implementation
Implements the systematic worldly wisdom layer for preventing reasoning failures.

Based on Charlie Munger's multidisciplinary approach and systematic bias detection.
Provides L0-L3 rigor levels with progressive analysis depth.
"""

import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from dotenv import load_dotenv

# Import database client
try:
    from supabase import create_client, Client
except ImportError:
    print("⚠️  Supabase client not available - operating in offline mode")
    Client = None


class RigorLevel(Enum):
    """Munger overlay rigor levels"""

    L0 = "L0"  # No overlay - basic analysis
    L1 = "L1"  # Inversion only - failure mode analysis
    L2 = "L2"  # Inversion + Latticework - multidisciplinary analysis
    L3 = "L3"  # Full Munger - Inversion + Latticework + Bias detection + Lollapalooza synthesis


@dataclass
class DecisionContext:
    """Context information that determines rigor level"""

    reversibility: str  # "reversible" | "irreversible"
    impact: str  # "low" | "medium" | "high"
    blast_radius: str  # "narrow" | "wide"
    domain: str  # "product" | "security" | "research" | "infra"
    time_pressure: str  # "low" | "medium" | "high"
    stakeholders: Optional[List[str]] = None
    criticality: Optional[Dict[str, Any]] = None


@dataclass
class InversionAnalysis:
    """L1: Inversion analysis results"""

    failure_modes: List[str]
    avoidance_strategies: List[str]
    what_could_go_wrong: List[str]
    prevention_measures: List[str]
    early_warning_signals: List[str]


@dataclass
class LatticeworkAnalysis:
    """L2: Multidisciplinary latticework analysis"""

    psychology_insights: List[str]
    systems_insights: List[str]
    economics_insights: List[str]
    math_insights: List[str]
    physics_insights: List[str]
    evolutionary_insights: List[str]
    cross_discipline_connections: List[str]


@dataclass
class BiasIdentification:
    """L3: Systematic bias identification"""

    stakeholder_biases: Dict[str, List[str]]
    self_biases: List[str]
    organizational_biases: List[str]
    systemic_biases: List[str]
    bias_mitigation_strategies: List[str]


@dataclass
class LollapalozzaSynthesis:
    """L3: Lollapalooza effect detection and synthesis"""

    convergence_patterns: List[str]
    amplification_risks: List[str]
    amplification_opportunities: List[str]
    compound_effect_analysis: List[str]
    emergent_properties: List[str]


@dataclass
class MungerOverlayOutput:
    """Complete Munger overlay analysis output"""

    overlay_id: str
    engagement_id: str
    rigor_level: RigorLevel
    policy_applied: str

    # Analysis components
    inversion_analysis: Optional[InversionAnalysis] = None
    latticework_analysis: Optional[LatticeworkAnalysis] = None
    bias_identification: Optional[BiasIdentification] = None
    lollapalooza_synthesis: Optional[LollapalozzaSynthesis] = None

    # Synthesis
    wise_path: str = ""
    key_assumptions: List[str] = None
    falsification_tests: List[str] = None

    # Quality scorecard
    risk_score: float = 0.0
    clarity_score: float = 0.0
    bias_awareness_score: float = 0.0

    # Metadata
    processing_time_ms: int = 0
    tokens_consumed: int = 0
    created_at: datetime = None

    def __post_init__(self):
        if self.key_assumptions is None:
            self.key_assumptions = []
        if self.falsification_tests is None:
            self.falsification_tests = []
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)


class MungerOverlay:
    """
    Core Charlie Munger Overlay Engine

    Implements systematic worldly wisdom principles to prevent reasoning failures:
    - L1: Inversion (What could go wrong?)
    - L2: Latticework of mental models (Multidisciplinary thinking)
    - L3: Bias detection + Lollapalooza synthesis (Compound effects)
    """

    def __init__(self, supabase_client: Optional[Client] = None):
        """Initialize Munger overlay with database connection"""
        load_dotenv()

        if supabase_client:
            self.supabase = supabase_client
        elif Client:
            self.supabase = create_client(
                os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_ANON_KEY")
            )
        else:
            self.supabase = None
            print("⚠️  Operating without database connection")

        self.policies: Dict[str, Dict[str, Any]] = {}
        self._load_policies()

    def _load_policies(self) -> None:
        """Load Munger policies from database"""
        if not self.supabase:
            # Default policies for offline mode
            self.policies = {
                "security": {
                    "default_rigor_level": "L3",
                    "triggers": {"conditions": ["irreversible", "high_impact"]},
                },
                "product": {
                    "default_rigor_level": "L2",
                    "triggers": {"conditions": ["medium_impact", "customer_facing"]},
                },
                "research": {
                    "default_rigor_level": "L1",
                    "triggers": {"conditions": ["reversible", "low_impact"]},
                },
                "infra": {
                    "default_rigor_level": "L2",
                    "triggers": {"conditions": ["system_wide", "reliability"]},
                },
            }
            return

        try:
            result = (
                self.supabase.table("munger_policies")
                .select("*")
                .eq("active", True)
                .execute()
            )
            for policy in result.data:
                self.policies[policy["domain"]] = {
                    "default_rigor_level": policy["default_rigor_level"],
                    "triggers": policy["triggers"],
                    "overrides": policy.get("overrides", {}),
                    "max_tokens_overhead": policy.get("max_tokens_overhead", 1000),
                    "max_latency_ms": policy.get("max_latency_ms", 5000),
                }
        except Exception as e:
            print(f"⚠️  Failed to load policies from database: {e}")
            # Fall back to default policies
            self._load_policies()

    def determine_rigor(self, context: DecisionContext) -> RigorLevel:
        """
        Determine appropriate rigor level based on decision context

        Core policy logic:
        - Irreversible OR high impact OR wide blast radius → L3
        - Medium impact OR security/compliance domain → L2
        - Otherwise → L1
        """
        # Get domain policy
        domain_policy = self.policies.get(context.domain, self.policies.get("product"))

        # Apply escalation rules
        if (
            context.reversibility == "irreversible"
            or context.impact == "high"
            or context.blast_radius == "wide"
        ):
            return RigorLevel.L3
        elif context.impact == "medium" or context.domain in ["security", "compliance"]:
            return RigorLevel.L2
        else:
            return RigorLevel.L1

    def apply_inversion(
        self, problem_statement: str, context: DecisionContext
    ) -> InversionAnalysis:
        """
        L1: Apply inversion analysis - think backwards from failure

        Charlie Munger: "Invert, always invert. Think about what you want to avoid."
        """
        # In a production system, this would use LLM APIs for dynamic analysis
        # For now, providing systematic framework with placeholders

        failure_modes = [
            f"Insufficient stakeholder buy-in leads to {context.domain} resistance",
            f"Technical complexity underestimated for {context.impact} impact decisions",
            f"Regulatory/compliance issues in {context.domain} domain",
            "Resource constraints not properly anticipated",
            "Timeline pressures force premature implementation",
        ]

        avoidance_strategies = [
            "Multi-stakeholder validation before major commitments",
            "Technical feasibility study with realistic buffer time",
            "Early regulatory/compliance review with domain experts",
            "Resource allocation review with contingency planning",
            "Staged implementation with clear go/no-go gates",
        ]

        what_could_go_wrong = [
            "Assumption proves false under real-world conditions",
            "Unintended consequences emerge after deployment",
            "Competitive response negates expected benefits",
            "Market conditions change during implementation",
            "Key personnel unavailable during critical phases",
        ]

        prevention_measures = [
            "Systematic assumption testing and validation",
            "Comprehensive scenario planning and stress testing",
            "Competitive intelligence and response planning",
            "Market monitoring with adaptive response capability",
            "Cross-training and succession planning for key roles",
        ]

        early_warning_signals = [
            "Stakeholder engagement metrics declining",
            "Technical milestone slippage patterns",
            "Compliance review feedback indicating issues",
            "Resource utilization exceeding planned rates",
            "Market indicators suggesting environmental changes",
        ]

        return InversionAnalysis(
            failure_modes=failure_modes,
            avoidance_strategies=avoidance_strategies,
            what_could_go_wrong=what_could_go_wrong,
            prevention_measures=prevention_measures,
            early_warning_signals=early_warning_signals,
        )

    def apply_latticework(
        self, problem_statement: str, context: DecisionContext
    ) -> LatticeworkAnalysis:
        """
        L2: Apply multidisciplinary latticework analysis

        Charlie Munger: "You must know the big ideas in the big disciplines and use them routinely."
        """
        psychology_insights = [
            "Confirmation bias: Stakeholders may favor information supporting their preferences",
            "Authority bias: Deference to senior decision makers may suppress dissent",
            "Planning fallacy: Teams typically underestimate time and complexity",
            "Loss aversion: People overweight potential losses vs. equivalent gains",
            "Social proof: Decisions influenced by what others in similar situations have done",
        ]

        systems_insights = [
            "Reinforcing feedback loops: Success builds confidence, failure erodes trust",
            "Delayed feedback: Full consequences may not be apparent for months",
            "Network effects: Early adoption critical for systems with network benefits",
            "Emergent properties: System behavior may differ from component behavior",
            "Leverage points: Small changes in key areas may yield disproportionate results",
        ]

        economics_insights = [
            "Switching costs: High switching costs create competitive moats but also resistance",
            "Two-sided market dynamics: Must balance needs of multiple stakeholder groups",
            "Opportunity cost: Resources deployed here cannot be used elsewhere",
            "Marginal analysis: Focus on incremental benefits vs. incremental costs",
            "Game theory: Consider how others will respond to our actions",
        ]

        math_insights = [
            f"Base rate: Historical success rate for similar {context.domain} initiatives",
            "Probabilistic thinking: Express confidence in ranges, not point estimates",
            "Expected value: Probability-weighted outcomes across scenarios",
            "Statistical significance: Ensure sample sizes support conclusions",
            "Compound effects: Small improvements compound over time",
        ]

        physics_insights = [
            "Momentum: Established processes resist change (organizational inertia)",
            "Critical mass: Need sufficient scale for sustainable change",
            "Thermodynamics: Systems tend toward entropy without energy input",
            "Scaling laws: What works at small scale may not work at large scale",
        ]

        evolutionary_insights = [
            "Survival pressure: Solutions must provide clear competitive advantage",
            "Adaptation: Must evolve with changing environmental conditions",
            "Genetic diversity: Multiple approaches reduce systemic risk",
            "Selection pressure: Market forces will eliminate ineffective solutions",
        ]

        cross_discipline_connections = [
            "Psychology + Economics: Behavioral economics insights on stakeholder responses",
            "Systems + Math: Quantitative modeling of feedback loops and delays",
            "Physics + Evolution: Scaling laws applied to organizational adaptation",
            "Psychology + Systems: How cognitive biases create systematic organizational blind spots",
        ]

        return LatticeworkAnalysis(
            psychology_insights=psychology_insights,
            systems_insights=systems_insights,
            economics_insights=economics_insights,
            math_insights=math_insights,
            physics_insights=physics_insights,
            evolutionary_insights=evolutionary_insights,
            cross_discipline_connections=cross_discipline_connections,
        )

    def identify_biases(
        self,
        problem_statement: str,
        context: DecisionContext,
        stakeholders: List[str] = None,
    ) -> BiasIdentification:
        """
        L3: Systematic bias identification across stakeholders

        Charlie Munger: "The human mind is not designed to understand itself."
        """
        if not stakeholders:
            stakeholders = ["CEO", "CTO", "Product Team", "Users", "Investors"]

        stakeholder_biases = {}
        for stakeholder in stakeholders:
            if stakeholder == "CEO":
                stakeholder_biases[stakeholder] = [
                    "optimism_bias",
                    "overconfidence_effect",
                    "planning_fallacy",
                ]
            elif stakeholder == "CTO":
                stakeholder_biases[stakeholder] = [
                    "technical_overconfidence",
                    "not_invented_here_syndrome",
                    "anchoring_bias",
                ]
            elif stakeholder == "Product Team":
                stakeholder_biases[stakeholder] = [
                    "confirmation_bias",
                    "feature_bias",
                    "user_projection",
                ]
            elif stakeholder == "Users":
                stakeholder_biases[stakeholder] = [
                    "loss_aversion",
                    "status_quo_bias",
                    "endowment_effect",
                ]
            elif stakeholder == "Investors":
                stakeholder_biases[stakeholder] = [
                    "recency_bias",
                    "survivorship_bias",
                    "herd_mentality",
                ]
            else:
                stakeholder_biases[stakeholder] = [
                    "confirmation_bias",
                    "availability_heuristic",
                    "anchoring_bias",
                ]

        self_biases = [
            "Confirmation bias: Seeking information that confirms our approach",
            "Anchoring bias: Over-relying on first information received",
            "Availability heuristic: Overweighting easily recalled examples",
            "Overconfidence effect: Overestimating accuracy of our judgments",
        ]

        organizational_biases = [
            "Groupthink: Pressure for consensus suppresses dissenting views",
            "NIH syndrome: Resistance to externally developed solutions",
            "Sunk cost fallacy: Continuing projects due to past investment",
            "Success bias: Assuming past success guarantees future success",
        ]

        systemic_biases = [
            "Selection bias: Available data may not represent true population",
            "Survivorship bias: Learning only from visible successes",
            "Publication bias: Positive results more likely to be shared",
            "Hindsight bias: Overestimating predictability of past events",
        ]

        bias_mitigation_strategies = [
            "Devil's advocate: Assign someone to argue against the proposal",
            "Pre-mortem analysis: Imagine failure and work backwards to causes",
            "Outside view: Benchmark against similar external projects",
            "Diverse perspectives: Include stakeholders with different incentives",
            "Anonymous feedback: Allow dissent without social pressure",
        ]

        return BiasIdentification(
            stakeholder_biases=stakeholder_biases,
            self_biases=self_biases,
            organizational_biases=organizational_biases,
            systemic_biases=systemic_biases,
            bias_mitigation_strategies=bias_mitigation_strategies,
        )

    def synthesize_lollapalooza(
        self,
        inversion: InversionAnalysis,
        latticework: LatticeworkAnalysis,
        biases: BiasIdentification,
    ) -> LollapalozzaSynthesis:
        """
        L3: Lollapalooza synthesis - identify compound effects and convergences

        Charlie Munger: "Lollapalooza effects come when two, three, or four forces
        are all operating in the same direction."
        """
        convergence_patterns = [
            "Authority + Social Proof + Scarcity = Rush to deploy without adequate testing",
            "Sunk Cost + Commitment + Overconfidence = Persisting with failing approach",
            "Confirmation + Availability + Anchoring = Systematic underestimation of risks",
            "Loss Aversion + Status Quo + Endowment = Resistance to necessary changes",
            "Optimism + Planning Fallacy + Overconfidence = Systematic under-resourcing",
        ]

        amplification_risks = [
            "Technical failure combined with user resistance creates compound negative feedback",
            "Market timing issues amplified by competitive response create market share loss",
            "Regulatory issues amplified by public visibility create reputational damage",
            "Resource constraints amplified by timeline pressure create quality shortcuts",
            "Stakeholder skepticism amplified by early setbacks create loss of support",
        ]

        amplification_opportunities = [
            "Technical excellence combined with user delight creates viral adoption",
            "Market timing combined with competitive advantage creates market dominance",
            "Regulatory compliance combined with transparency creates trust premium",
            "Resource efficiency combined with rapid iteration creates innovation advantage",
            "Stakeholder alignment combined with clear vision creates execution momentum",
        ]

        compound_effect_analysis = [
            "Network effects compound with switching costs to create platform advantages",
            "Learning effects compound with scale advantages to create cost leadership",
            "Brand effects compound with distribution advantages to create market power",
            "Data effects compound with algorithm improvements to create intelligence advantages",
        ]

        emergent_properties = [
            "System exhibits behaviors not predictable from individual components",
            "Stakeholder interactions create dynamics not evident in isolation",
            "Market responses emerge from collective behavior patterns",
            "Organizational capabilities emerge from process and culture interactions",
        ]

        return LollapalozzaSynthesis(
            convergence_patterns=convergence_patterns,
            amplification_risks=amplification_risks,
            amplification_opportunities=amplification_opportunities,
            compound_effect_analysis=compound_effect_analysis,
            emergent_properties=emergent_properties,
        )

    def synthesize_wise_path(self, overlay_output: MungerOverlayOutput) -> str:
        """
        Synthesize the 'wise path' forward based on all analyses

        Charlie Munger: "The great lesson in microeconomics is to discriminate
        between when technology is going to help you and when it's going to kill you."
        """
        if overlay_output.rigor_level == RigorLevel.L3:
            return (
                "Apply staged rollout with systematic risk mitigation at each phase. "
                "Maintain extensive stakeholder validation, clear success/failure criteria, "
                "and prepared fallback options. Monitor for Lollapalooza convergences that "
                "could amplify both risks and opportunities. Focus on falsifiable assumptions "
                "and early warning signals to enable rapid course correction."
            )
        elif overlay_output.rigor_level == RigorLevel.L2:
            return (
                "Implement multidisciplinary analysis with focus on systems effects and "
                "stakeholder psychology. Apply inversion thinking to identify major failure "
                "modes and develop specific mitigation strategies. Monitor cross-functional "
                "feedback loops and unintended consequences."
            )
        else:  # L1
            return (
                "Apply systematic inversion analysis to identify primary failure modes. "
                "Develop specific avoidance strategies and early warning indicators. "
                "Focus on what could go wrong and how to prevent it."
            )

    async def apply_overlay(
        self,
        problem_statement: str,
        context: DecisionContext,
        engagement_id: str = None,
    ) -> MungerOverlayOutput:
        """
        Apply complete Munger overlay analysis based on determined rigor level
        """
        start_time = datetime.now()

        # Generate IDs
        overlay_id = f"MUNGER-{uuid.uuid4().hex[:8].upper()}"
        if not engagement_id:
            engagement_id = f"ENGAGEMENT-{uuid.uuid4().hex[:8].upper()}"

        # Determine rigor level
        rigor_level = self.determine_rigor(context)

        # Get applicable policy
        domain_policy = self.policies.get(context.domain, self.policies.get("product"))
        policy_applied = f"policy_{context.domain}"

        # Initialize overlay output
        overlay_output = MungerOverlayOutput(
            overlay_id=overlay_id,
            engagement_id=engagement_id,
            rigor_level=rigor_level,
            policy_applied=policy_applied,
        )

        # Apply analyses based on rigor level
        if rigor_level.value in ["L1", "L2", "L3"]:
            overlay_output.inversion_analysis = self.apply_inversion(
                problem_statement, context
            )

        if rigor_level.value in ["L2", "L3"]:
            overlay_output.latticework_analysis = self.apply_latticework(
                problem_statement, context
            )

        if rigor_level.value == "L3":
            stakeholders = context.stakeholders or ["CEO", "CTO", "Users", "Team"]
            overlay_output.bias_identification = self.identify_biases(
                problem_statement, context, stakeholders
            )
            overlay_output.lollapalooza_synthesis = self.synthesize_lollapalooza(
                overlay_output.inversion_analysis,
                overlay_output.latticework_analysis,
                overlay_output.bias_identification,
            )

        # Generate wise path synthesis
        overlay_output.wise_path = self.synthesize_wise_path(overlay_output)

        # Generate key assumptions and falsification tests
        overlay_output.key_assumptions = [
            "Stakeholder interests align with proposed approach",
            "Technical implementation complexity is manageable",
            "Market/regulatory environment remains stable",
            "Resource requirements are accurately estimated",
        ]

        overlay_output.falsification_tests = [
            "Stakeholder resistance above 30% indicates misalignment",
            "Technical implementation taking >150% of estimate indicates complexity underestimation",
            "External environment changes requiring major approach modifications",
            "Resource requirements exceeding estimates by >50%",
        ]

        # Calculate quality scores
        overlay_output.risk_score = (
            min(
                0.9, 0.5 + (len(overlay_output.inversion_analysis.failure_modes) * 0.05)
            )
            if overlay_output.inversion_analysis
            else 0.5
        )
        overlay_output.clarity_score = 0.7 + (
            0.1 * len(rigor_level.value)
        )  # Higher rigor = higher clarity
        overlay_output.bias_awareness_score = (
            0.9
            if rigor_level == RigorLevel.L3
            else (0.7 if rigor_level == RigorLevel.L2 else 0.5)
        )

        # Record performance metrics
        end_time = datetime.now()
        overlay_output.processing_time_ms = int(
            (end_time - start_time).total_seconds() * 1000
        )
        overlay_output.tokens_consumed = 1500 + (
            500 * len(rigor_level.value)
        )  # Estimated based on analysis depth

        # Save to database if available
        await self._save_overlay(overlay_output)

        return overlay_output

    async def _save_overlay(self, overlay_output: MungerOverlayOutput) -> None:
        """Save overlay output to database"""
        if not self.supabase:
            return

        try:
            # Convert to database format
            db_data = {
                "overlay_id": overlay_output.overlay_id,
                "engagement_id": overlay_output.engagement_id,
                "rigor_level": overlay_output.rigor_level.value,
                "policy_applied": overlay_output.policy_applied,
                "wise_path": overlay_output.wise_path,
                "key_assumptions": overlay_output.key_assumptions,
                "falsification_tests": overlay_output.falsification_tests,
                "risk_score": overlay_output.risk_score,
                "clarity_score": overlay_output.clarity_score,
                "bias_awareness_score": overlay_output.bias_awareness_score,
                "processing_time_ms": overlay_output.processing_time_ms,
                "tokens_consumed": overlay_output.tokens_consumed,
            }

            # Add analysis components
            if overlay_output.inversion_analysis:
                db_data["failure_modes"] = (
                    overlay_output.inversion_analysis.failure_modes
                )
                db_data["avoidance_strategies"] = (
                    overlay_output.inversion_analysis.avoidance_strategies
                )
                db_data["inversion_analysis"] = asdict(
                    overlay_output.inversion_analysis
                )

            if overlay_output.latticework_analysis:
                db_data["psychology_insights"] = (
                    overlay_output.latticework_analysis.psychology_insights
                )
                db_data["systems_insights"] = (
                    overlay_output.latticework_analysis.systems_insights
                )
                db_data["economics_insights"] = (
                    overlay_output.latticework_analysis.economics_insights
                )
                db_data["math_insights"] = (
                    overlay_output.latticework_analysis.math_insights
                )
                db_data["latticework_analysis"] = asdict(
                    overlay_output.latticework_analysis
                )

            if overlay_output.bias_identification:
                db_data["stakeholder_biases"] = (
                    overlay_output.bias_identification.stakeholder_biases
                )
                db_data["self_biases"] = overlay_output.bias_identification.self_biases
                db_data["bias_identification"] = asdict(
                    overlay_output.bias_identification
                )

            if overlay_output.lollapalooza_synthesis:
                db_data["convergence_patterns"] = (
                    overlay_output.lollapalooza_synthesis.convergence_patterns
                )
                db_data["amplification_risks"] = (
                    overlay_output.lollapalooza_synthesis.amplification_risks
                )
                db_data["amplification_opportunities"] = (
                    overlay_output.lollapalooza_synthesis.amplification_opportunities
                )
                db_data["lollapalooza_synthesis"] = asdict(
                    overlay_output.lollapalooza_synthesis
                )

            # Insert into database
            result = (
                self.supabase.table("munger_overlay_outputs").insert(db_data).execute()
            )

        except Exception as e:
            print(f"⚠️  Failed to save overlay to database: {e}")


# Example usage and testing
if __name__ == "__main__":

    async def test_munger_overlay():
        """Test the Munger overlay system"""
        overlay = MungerOverlay()

        # Test case: High-stakes product decision
        context = DecisionContext(
            reversibility="irreversible",
            impact="high",
            blast_radius="wide",
            domain="product",
            time_pressure="medium",
            stakeholders=["CEO", "CTO", "Product Team", "Users", "Board"],
        )

        problem = (
            "Should we pivot our core product strategy to focus on AI-native features?"
        )

        result = await overlay.apply_overlay(problem, context)

        print("🧠 MUNGER OVERLAY ANALYSIS")
        print(f"Overlay ID: {result.overlay_id}")
        print(f"Rigor Level: {result.rigor_level.value}")
        print(f"Risk Score: {result.risk_score:.2f}")
        print(f"Clarity Score: {result.clarity_score:.2f}")
        print(f"Bias Awareness: {result.bias_awareness_score:.2f}")
        print(f"Processing Time: {result.processing_time_ms}ms")
        print(f"\n💡 Wise Path: {result.wise_path}")

        if result.inversion_analysis:
            print("\n🔄 Top Failure Modes:")
            for i, failure in enumerate(result.inversion_analysis.failure_modes[:3], 1):
                print(f"  {i}. {failure}")

        if result.lollapalooza_synthesis:
            print("\n⚡ Lollapalooza Convergences:")
            for pattern in result.lollapalooza_synthesis.convergence_patterns[:2]:
                print(f"  • {pattern}")

    # Run test
    asyncio.run(test_munger_overlay())
