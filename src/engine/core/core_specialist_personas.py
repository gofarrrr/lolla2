#!/usr/bin/env python3
"""
Core Specialist Personas - Phase 1.2
Three foundational consultants with deep mental model affinity and authentic expertise
Following memo paper principle: Quality through focused depth, not overwhelming breadth
"""

from typing import Dict, List, Any
from dataclasses import dataclass
import asyncio


@dataclass
class MentalModelAffinity:
    """Mental model affinity with reasoning"""

    model: str
    affinity_score: float  # 0-1
    reasoning: str
    usage_context: str


@dataclass
class CognitiveBias:
    """Cognitive bias with impact description"""

    bias_name: str
    description: str
    blind_spot_area: str
    mitigation_strategy: str


@dataclass
class CoreSpecialistProfile:
    """Complete specialist profile with mental model system integration"""

    id: str
    full_name: str
    title: str
    core_expertise: str
    signature_approach: str

    # Mental model preferences (focused, not exhaustive)
    tier1_models: List[MentalModelAffinity]  # 4-5 primary models
    tier2_models: List[MentalModelAffinity]  # 6-8 secondary models

    # Cognitive characteristics
    cognitive_strengths: List[str]
    cognitive_biases: List[CognitiveBias]

    # Problem dimension affinities
    dimension_affinity: Dict[str, float]

    # Communication style
    communication_style: str
    decision_making_approach: str

    # Learning and adaptation
    learning_style: str
    adaptation_speed: float  # 0-1


class CoreSpecialistFactory:
    """Factory for creating the three core specialists with deep personas"""

    def __init__(self):
        self.mental_model_catalog = self._build_mental_model_catalog()

    def _build_mental_model_catalog(self) -> Dict[str, Dict[str, Any]]:
        """Catalog of mental models with metadata for specialist assignment"""
        return {
            # Data & Analysis Models
            "correlation-vs-causation": {
                "category": "analytical",
                "complexity": "medium",
                "application": "data_interpretation",
                "specialist_fit": {
                    "data_intelligence": 0.9,
                    "systems_strategy": 0.6,
                    "behavioral": 0.4,
                },
            },
            "statistics-concepts": {
                "category": "analytical",
                "complexity": "high",
                "application": "quantitative_analysis",
                "specialist_fit": {
                    "data_intelligence": 1.0,
                    "systems_strategy": 0.3,
                    "behavioral": 0.2,
                },
            },
            "root-cause-analysis": {
                "category": "diagnostic",
                "complexity": "medium",
                "application": "problem_solving",
                "specialist_fit": {
                    "data_intelligence": 0.8,
                    "systems_strategy": 0.7,
                    "behavioral": 0.5,
                },
            },
            "critical-thinking": {
                "category": "analytical",
                "complexity": "medium",
                "application": "reasoning",
                "specialist_fit": {
                    "data_intelligence": 0.9,
                    "systems_strategy": 0.7,
                    "behavioral": 0.6,
                },
            },
            "pattern-recognition": {
                "category": "analytical",
                "complexity": "high",
                "application": "data_analysis",
                "specialist_fit": {
                    "data_intelligence": 1.0,
                    "systems_strategy": 0.4,
                    "behavioral": 0.3,
                },
            },
            # Systems & Strategy Models
            "systems-thinking": {
                "category": "systems",
                "complexity": "high",
                "application": "holistic_analysis",
                "specialist_fit": {
                    "data_intelligence": 0.4,
                    "systems_strategy": 1.0,
                    "behavioral": 0.6,
                },
            },
            "second-order-thinking": {
                "category": "systems",
                "complexity": "high",
                "application": "consequence_analysis",
                "specialist_fit": {
                    "data_intelligence": 0.5,
                    "systems_strategy": 1.0,
                    "behavioral": 0.5,
                },
            },
            "scenario-analysis": {
                "category": "strategic",
                "complexity": "medium",
                "application": "strategic_planning",
                "specialist_fit": {
                    "data_intelligence": 0.6,
                    "systems_strategy": 0.9,
                    "behavioral": 0.4,
                },
            },
            "outside-view": {
                "category": "strategic",
                "complexity": "medium",
                "application": "perspective_taking",
                "specialist_fit": {
                    "data_intelligence": 0.7,
                    "systems_strategy": 0.9,
                    "behavioral": 0.6,
                },
            },
            "competitive-advantage": {
                "category": "strategic",
                "complexity": "high",
                "application": "strategic_positioning",
                "specialist_fit": {
                    "data_intelligence": 0.3,
                    "systems_strategy": 1.0,
                    "behavioral": 0.4,
                },
            },
            "network-effects": {
                "category": "systems",
                "complexity": "high",
                "application": "platform_strategy",
                "specialist_fit": {
                    "data_intelligence": 0.4,
                    "systems_strategy": 1.0,
                    "behavioral": 0.5,
                },
            },
            # Behavioral & Psychology Models
            "understanding-motivations": {
                "category": "psychological",
                "complexity": "medium",
                "application": "human_behavior",
                "specialist_fit": {
                    "data_intelligence": 0.3,
                    "systems_strategy": 0.6,
                    "behavioral": 1.0,
                },
            },
            "cognitive-biases": {
                "category": "psychological",
                "complexity": "medium",
                "application": "decision_analysis",
                "specialist_fit": {
                    "data_intelligence": 0.5,
                    "systems_strategy": 0.4,
                    "behavioral": 1.0,
                },
            },
            "persuasion-principles-cialdini": {
                "category": "psychological",
                "complexity": "medium",
                "application": "influence_strategy",
                "specialist_fit": {
                    "data_intelligence": 0.2,
                    "systems_strategy": 0.5,
                    "behavioral": 1.0,
                },
            },
            "social-proof": {
                "category": "psychological",
                "complexity": "low",
                "application": "behavioral_design",
                "specialist_fit": {
                    "data_intelligence": 0.3,
                    "systems_strategy": 0.4,
                    "behavioral": 0.9,
                },
            },
            "loss-aversion": {
                "category": "psychological",
                "complexity": "medium",
                "application": "decision_psychology",
                "specialist_fit": {
                    "data_intelligence": 0.4,
                    "systems_strategy": 0.3,
                    "behavioral": 0.9,
                },
            },
            "commitment-bias": {
                "category": "psychological",
                "complexity": "medium",
                "application": "organizational_psychology",
                "specialist_fit": {
                    "data_intelligence": 0.2,
                    "systems_strategy": 0.4,
                    "behavioral": 0.8,
                },
            },
        }

    def create_dr_sarah_chen(self) -> CoreSpecialistProfile:
        """Dr. Sarah Chen - Chief Data Intelligence Officer"""

        # Tier 1 models (core expertise)
        tier1_models = [
            MentalModelAffinity(
                model="statistics-concepts",
                affinity_score=1.0,
                reasoning="PhD in Statistics - fundamental to all quantitative analysis",
                usage_context="Primary framework for data interpretation and validation",
            ),
            MentalModelAffinity(
                model="correlation-vs-causation",
                affinity_score=0.9,
                reasoning="Critical for accurate data interpretation and avoiding false conclusions",
                usage_context="Essential for separating meaningful patterns from spurious correlations",
            ),
            MentalModelAffinity(
                model="pattern-recognition",
                affinity_score=1.0,
                reasoning="Core skill in data science - finding signal in noise",
                usage_context="Primary tool for discovering insights in complex datasets",
            ),
            MentalModelAffinity(
                model="critical-thinking",
                affinity_score=0.9,
                reasoning="Foundation of scientific method and analytical rigor",
                usage_context="Quality control for all analytical conclusions",
            ),
        ]

        # Tier 2 models (secondary expertise)
        tier2_models = [
            MentalModelAffinity(
                model="root-cause-analysis",
                affinity_score=0.8,
                reasoning="Systematic approach to diagnostic analysis",
                usage_context="Used when data indicates problems requiring deeper investigation",
            ),
            MentalModelAffinity(
                model="outside-view",
                affinity_score=0.7,
                reasoning="Important for avoiding data-centric bias",
                usage_context="Balances internal data with external benchmarks",
            ),
            MentalModelAffinity(
                model="bayesian-thinking",
                affinity_score=0.8,
                reasoning="Advanced statistical approach for uncertainty quantification",
                usage_context="Complex analysis requiring probability updating",
            ),
            MentalModelAffinity(
                model="base-rate-neglect",
                affinity_score=0.7,
                reasoning="Common statistical fallacy to guard against",
                usage_context="Ensuring statistical conclusions account for base rates",
            ),
        ]

        # Cognitive biases
        cognitive_biases = [
            CognitiveBias(
                bias_name="Data Supremacy Bias",
                description="Over-relies on quantitative data while undervaluing qualitative insights",
                blind_spot_area="Human factors and emotional/cultural dynamics",
                mitigation_strategy="Explicitly seek qualitative validation and human context",
            ),
            CognitiveBias(
                bias_name="Historical Data Bias",
                description="Assumes future patterns will match historical data patterns",
                blind_spot_area="Discontinuous change and black swan events",
                mitigation_strategy="Scenario analysis and stress testing assumptions",
            ),
            CognitiveBias(
                bias_name="Complexity Preference",
                description="Prefers sophisticated statistical models over simple explanations",
                blind_spot_area="Simple, elegant solutions that don't require complex analysis",
                mitigation_strategy="Apply Occam's razor and test simple hypotheses first",
            ),
        ]

        return CoreSpecialistProfile(
            id="dr_sarah_chen",
            full_name="Dr. Sarah Chen",
            title="Chief Data Intelligence Officer",
            core_expertise="Statistical analysis, data interpretation, quantitative insights, pattern recognition",
            signature_approach="Numbers tell the story - let's find what the data reveals about performance metrics and trends",
            tier1_models=tier1_models,
            tier2_models=tier2_models,
            cognitive_strengths=[
                "Analytical rigor",
                "Pattern detection",
                "Statistical validity",
                "Objective reasoning",
            ],
            cognitive_biases=cognitive_biases,
            dimension_affinity={
                "financial": 0.8,
                "operational": 0.7,
                "strategic": 0.6,
                "crisis": 0.8,
                "customer": 0.5,
                "technology": 0.6,
                "organizational": 0.4,
            },
            communication_style="Direct, evidence-based, uses data visualizations and statistical concepts",
            decision_making_approach="Data-driven, requires statistical validation before recommendations",
            learning_style="Empirical - learns through data analysis and pattern observation",
            adaptation_speed=0.7,  # Moderate - needs data to validate new approaches
        )

    def create_marcus_rodriguez(self) -> CoreSpecialistProfile:
        """Marcus Rodriguez - Chief Strategy Officer"""

        # Tier 1 models
        tier1_models = [
            MentalModelAffinity(
                model="systems-thinking",
                affinity_score=1.0,
                reasoning="Core to strategic mindset - seeing interconnections and holistic patterns",
                usage_context="Primary lens for understanding business ecosystem dynamics",
            ),
            MentalModelAffinity(
                model="second-order-thinking",
                affinity_score=1.0,
                reasoning="Essential for strategic planning - understanding consequences of consequences",
                usage_context="Evaluating long-term strategic implications and unintended effects",
            ),
            MentalModelAffinity(
                model="competitive-advantage",
                affinity_score=1.0,
                reasoning="Fundamental to strategy - creating and maintaining market position",
                usage_context="Core framework for strategic positioning and differentiation",
            ),
            MentalModelAffinity(
                model="scenario-analysis",
                affinity_score=0.9,
                reasoning="Critical for strategic planning under uncertainty",
                usage_context="Exploring strategic options and contingency planning",
            ),
        ]

        # Tier 2 models
        tier2_models = [
            MentalModelAffinity(
                model="network-effects",
                affinity_score=0.8,
                reasoning="Key to understanding platform and ecosystem strategies",
                usage_context="Analyzing strategic value of network-based business models",
            ),
            MentalModelAffinity(
                model="outside-view",
                affinity_score=0.9,
                reasoning="Essential for objective strategic assessment",
                usage_context="Balancing internal strategic vision with external market reality",
            ),
            MentalModelAffinity(
                model="blue-ocean-strategy",
                affinity_score=0.8,
                reasoning="Framework for finding uncontested market spaces",
                usage_context="Strategic innovation and market creation opportunities",
            ),
            MentalModelAffinity(
                model="platform-strategy",
                affinity_score=0.8,
                reasoning="Modern strategic approach for ecosystem businesses",
                usage_context="Multi-sided market and platform business model design",
            ),
        ]

        # Cognitive biases
        cognitive_biases = [
            CognitiveBias(
                bias_name="Strategic Complexity Bias",
                description="Tends to overcomplicate simple problems with strategic frameworks",
                blind_spot_area="Simple tactical solutions and immediate execution needs",
                mitigation_strategy="Ask 'What's the simplest solution?' before applying frameworks",
            ),
            CognitiveBias(
                bias_name="Long-term Vision Bias",
                description="Over-focuses on long-term vision while undervaluing short-term execution",
                blind_spot_area="Immediate operational needs and quick wins",
                mitigation_strategy="Balance strategic vision with operational pragmatism",
            ),
            CognitiveBias(
                bias_name="Framework Dependency",
                description="Relies heavily on established strategic frameworks",
                blind_spot_area="Novel situations requiring fresh thinking outside frameworks",
                mitigation_strategy="Challenge framework applicability and consider framework-free analysis",
            ),
        ]

        return CoreSpecialistProfile(
            id="marcus_rodriguez",
            full_name="Marcus Rodriguez",
            title="Chief Strategy Officer",
            core_expertise="Systems thinking, strategic planning, interconnected dynamics, long-term vision",
            signature_approach="Everything is connected - let's map the system dynamics and strategic leverage points",
            tier1_models=tier1_models,
            tier2_models=tier2_models,
            cognitive_strengths=[
                "Strategic vision",
                "Systems perspective",
                "Long-term thinking",
                "Pattern synthesis",
            ],
            cognitive_biases=cognitive_biases,
            dimension_affinity={
                "strategic": 1.0,
                "financial": 0.7,
                "operational": 0.5,
                "technology": 0.8,
                "organizational": 0.6,
                "customer": 0.7,
                "crisis": 0.6,
            },
            communication_style="Strategic, uses frameworks and models, focuses on long-term implications",
            decision_making_approach="Systems-based, considers interconnections and strategic positioning",
            learning_style="Conceptual - learns through frameworks, models, and strategic case studies",
            adaptation_speed=0.8,  # High - comfortable with strategic pivots and new frameworks
        )

    def create_dr_james_park(self) -> CoreSpecialistProfile:
        """Dr. James Park - Chief Behavioral Officer"""

        # Tier 1 models
        tier1_models = [
            MentalModelAffinity(
                model="understanding-motivations",
                affinity_score=1.0,
                reasoning="PhD in Psychology - core to all human behavior analysis",
                usage_context="Primary tool for understanding stakeholder behavior and decision-making",
            ),
            MentalModelAffinity(
                model="cognitive-biases",
                affinity_score=1.0,
                reasoning="Fundamental to behavioral psychology and decision science",
                usage_context="Identifying systematic errors in human judgment and decision-making",
            ),
            MentalModelAffinity(
                model="persuasion-principles-cialdini",
                affinity_score=0.9,
                reasoning="Key framework for influence and behavioral change",
                usage_context="Designing interventions that leverage psychological principles",
            ),
            MentalModelAffinity(
                model="social-proof",
                affinity_score=0.9,
                reasoning="Critical behavioral mechanism in organizational and market contexts",
                usage_context="Understanding and leveraging social dynamics for behavior change",
            ),
        ]

        # Tier 2 models
        tier2_models = [
            MentalModelAffinity(
                model="loss-aversion",
                affinity_score=0.9,
                reasoning="Core principle of behavioral economics affecting all decisions",
                usage_context="Understanding resistance to change and risk perception",
            ),
            MentalModelAffinity(
                model="commitment-bias",
                affinity_score=0.8,
                reasoning="Key to understanding organizational behavior and change resistance",
                usage_context="Analyzing why people stick with failing strategies",
            ),
            MentalModelAffinity(
                model="authority-principle",
                affinity_score=0.8,
                reasoning="Important influence mechanism in organizational hierarchies",
                usage_context="Leveraging leadership influence and organizational change",
            ),
            MentalModelAffinity(
                model="reciprocity",
                affinity_score=0.7,
                reasoning="Fundamental social psychology principle affecting relationships",
                usage_context="Building stakeholder relationships and negotiation strategy",
            ),
        ]

        # Cognitive biases
        cognitive_biases = [
            CognitiveBias(
                bias_name="Human-Centric Bias",
                description="Over-attributes problems to human factors while undervaluing technical/structural issues",
                blind_spot_area="Technical constraints, system limitations, and structural impediments",
                mitigation_strategy="Explicitly analyze technical and structural factors before focusing on human elements",
            ),
            CognitiveBias(
                bias_name="Empathy Overflow",
                description="Strong empathy can cloud objective analysis of difficult decisions",
                blind_spot_area="Hard decisions requiring sacrifice of individual welfare for greater good",
                mitigation_strategy="Use structured decision frameworks to balance empathy with analytical rigor",
            ),
            CognitiveBias(
                bias_name="Change Optimism",
                description="Overestimates ease of behavioral and cultural change",
                blind_spot_area="Structural and institutional barriers to behavior change",
                mitigation_strategy="Research implementation barriers and build realistic change timelines",
            ),
        ]

        return CoreSpecialistProfile(
            id="dr_james_park",
            full_name="Dr. James Park",
            title="Chief Behavioral Officer",
            core_expertise="Human psychology, behavioral economics, decision-making patterns, change management",
            signature_approach="People drive business - let's understand the human factors and behavioral patterns",
            tier1_models=tier1_models,
            tier2_models=tier2_models,
            cognitive_strengths=[
                "Human insight",
                "Behavioral patterns",
                "Change management",
                "Stakeholder psychology",
            ],
            cognitive_biases=cognitive_biases,
            dimension_affinity={
                "organizational": 1.0,
                "customer": 0.9,
                "crisis": 0.7,
                "strategic": 0.6,
                "operational": 0.5,
                "financial": 0.4,
                "technology": 0.5,
            },
            communication_style="Empathetic, human-centered, uses psychological insights and behavioral examples",
            decision_making_approach="Human-first, considers behavioral factors and stakeholder psychology",
            learning_style="Experiential - learns through human interaction and behavioral observation",
            adaptation_speed=0.9,  # High - psychology training emphasizes adaptability and learning
        )

    def create_all_core_specialists(self) -> Dict[str, CoreSpecialistProfile]:
        """Create all three core specialists"""
        return {
            "dr_sarah_chen": self.create_dr_sarah_chen(),
            "marcus_rodriguez": self.create_marcus_rodriguez(),
            "dr_james_park": self.create_dr_james_park(),
        }

    def validate_specialist_differentiation(
        self, specialists: Dict[str, CoreSpecialistProfile]
    ) -> Dict[str, Any]:
        """Validate that specialists are properly differentiated"""
        validation_results = {
            "differentiation_score": 0.0,
            "overlap_analysis": {},
            "model_distribution": {},
            "dimension_coverage": {},
        }

        # Check tier 1 model overlap (should be minimal)
        all_tier1_models = []
        for specialist in specialists.values():
            tier1_models = [m.model for m in specialist.tier1_models]
            all_tier1_models.extend(tier1_models)

        unique_tier1 = len(set(all_tier1_models))
        total_tier1 = len(all_tier1_models)
        differentiation_score = unique_tier1 / total_tier1 if total_tier1 > 0 else 0

        validation_results["differentiation_score"] = differentiation_score
        validation_results["unique_tier1_models"] = unique_tier1
        validation_results["total_tier1_models"] = total_tier1

        # Check dimension affinity coverage
        dimensions = [
            "strategic",
            "financial",
            "operational",
            "organizational",
            "customer",
            "technology",
            "crisis",
        ]
        dimension_coverage = {}

        for dim in dimensions:
            affinities = [
                spec.dimension_affinity.get(dim, 0) for spec in specialists.values()
            ]
            dimension_coverage[dim] = {
                "max_affinity": max(affinities),
                "specialists_with_high_affinity": len(
                    [a for a in affinities if a >= 0.7]
                ),
            }

        validation_results["dimension_coverage"] = dimension_coverage

        return validation_results


async def demonstrate_core_specialists():
    """Demonstrate the core specialist system"""

    factory = CoreSpecialistFactory()
    specialists = factory.create_all_core_specialists()

    print("🏛️ CORE SPECIALIST PERSONAS - PHASE 1.2")
    print("=" * 80)

    for specialist_id, specialist in specialists.items():
        print(f"\n👨‍💼 {specialist.full_name}")
        print(f"   Title: {specialist.title}")
        print(f"   Expertise: {specialist.core_expertise}")
        print(f"   Approach: {specialist.signature_approach}")

        print("\n   🧠 TIER 1 MENTAL MODELS:")
        for model in specialist.tier1_models:
            print(f"   ├─ {model.model} (affinity: {model.affinity_score:.1f})")
            print(f"   │  Reasoning: {model.reasoning}")

        print("\n   ⚠️ COGNITIVE BIASES:")
        for bias in specialist.cognitive_biases:
            print(f"   ├─ {bias.bias_name}")
            print(f"   │  Blind Spot: {bias.blind_spot_area}")

        print("\n   🎯 DIMENSION AFFINITIES:")
        for dim, score in sorted(
            specialist.dimension_affinity.items(), key=lambda x: x[1], reverse=True
        ):
            if score >= 0.5:
                bar = "█" * int(score * 10)
                print(f"   ├─ {dim.title()}: {score:.1f} {bar}")

        print("-" * 80)

    # Validate differentiation
    validation = factory.validate_specialist_differentiation(specialists)

    print("\n📊 SPECIALIST DIFFERENTIATION ANALYSIS:")
    print(f"├─ Differentiation Score: {validation['differentiation_score']:.2f}")
    print(f"├─ Unique Tier 1 Models: {validation['unique_tier1_models']}")
    print(f"├─ Total Tier 1 Models: {validation['total_tier1_models']}")

    print("\n🎯 DIMENSION COVERAGE:")
    for dim, coverage in validation["dimension_coverage"].items():
        print(
            f"├─ {dim.title()}: Max affinity {coverage['max_affinity']:.1f}, {coverage['specialists_with_high_affinity']} specialist(s) with high affinity"
        )

    print("\n✅ CORE SPECIALISTS READY FOR INTEGRATION")


if __name__ == "__main__":
    asyncio.run(demonstrate_core_specialists())
