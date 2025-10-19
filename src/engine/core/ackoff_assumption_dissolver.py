#!/usr/bin/env python3
"""
Ackoff Assumption Dissolver - Devils Advocate Engine #2
Implements Russell Ackoff's assumption dissolution methodology
Part of the enhanced Devils Advocate system focusing on dissolving fundamental assumptions
"""

import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging


@dataclass
class DissolvedAssumption:
    """Individual assumption that has been dissolved through Ackoff methodology"""

    assumption_id: str
    assumption_text: str
    assumption_type: str  # fundamental, operational, contextual
    dissolution_strength: float  # 0.0-1.0 how completely dissolved
    alternative_framings: List[str]
    idealized_design_approach: str
    systems_thinking_insights: List[str]
    practical_implications: str


@dataclass
class AssumptionDissolveResult:
    """Complete result from Ackoff assumption dissolution process"""

    original_problem: str
    dissolved_assumptions: List[DissolvedAssumption]
    dissolution_impact_score: float
    idealized_design_vision: str
    systems_redesign_opportunities: List[str]
    fundamental_reframes: List[str]
    processing_time_ms: float
    methodology_confidence: float


class AckoffAssumptionDissolver:
    """
    Russell Ackoff-inspired assumption dissolution system

    Implements Ackoff's core methodologies:
    1. Assumption identification and dissolution
    2. Idealized design thinking (unconstrained optimal system)
    3. Systems thinking and holistic analysis
    4. Problem dissolution vs. problem solving
    5. Interactive planning and purposeful systems
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Ackoff assumption categories for systematic analysis
        self.assumption_categories = {
            "fundamental": {
                "description": "Basic beliefs about how the world works",
                "indicators": [
                    "must",
                    "always",
                    "never",
                    "impossible",
                    "everyone",
                    "obvious",
                ],
                "questions": [
                    "Why do we believe this is necessarily true?",
                    "What if the opposite were possible?",
                ],
            },
            "operational": {
                "description": "Beliefs about how business operations work",
                "indicators": [
                    "process",
                    "procedure",
                    "workflow",
                    "standard",
                    "practice",
                ],
                "questions": [
                    "Why do we do it this way?",
                    "What would happen if we eliminated this step entirely?",
                ],
            },
            "contextual": {
                "description": "Beliefs about market, industry, and environmental constraints",
                "indicators": [
                    "market",
                    "industry",
                    "customer",
                    "regulation",
                    "competition",
                ],
                "questions": [
                    "Is this context permanent?",
                    "How might this constraint disappear?",
                ],
            },
            "resource": {
                "description": "Beliefs about resource limitations and capabilities",
                "indicators": ["budget", "time", "people", "capability", "resource"],
                "questions": [
                    "Is this limitation real or perceived?",
                    "How could we transcend this constraint?",
                ],
            },
            "success": {
                "description": "Beliefs about what constitutes success and value",
                "indicators": [
                    "success",
                    "goal",
                    "objective",
                    "value",
                    "important",
                    "priority",
                ],
                "questions": [
                    "Who defined this success metric?",
                    "What would success look like from other perspectives?",
                ],
            },
        }

        # Ackoff idealized design principles
        self.idealized_design_principles = [
            "technological_omniscience",  # Assume all current tech available
            "technological_omnipotence",  # Assume perfect technological capability
            "no_resource_constraints",  # Remove artificial scarcity assumptions
            "perfect_information",  # Assume complete information availability
            "stakeholder_alignment",  # Assume perfect stakeholder cooperation
            "regulatory_optimization",  # Assume regulations support optimal outcomes
        ]

    async def dissolve_assumptions(
        self, problem_statement: str, business_context: Dict[str, Any]
    ) -> AssumptionDissolveResult:
        """Dissolve assumptions using Ackoff's systematic methodology"""

        import time

        start_time = time.time()

        print("🔬 ACKOFF ASSUMPTION DISSOLUTION ENGINE")
        print("-" * 60)
        print(f"Analyzing: {problem_statement[:80]}...")

        # Step 1: Identify embedded assumptions systematically
        identified_assumptions = self._identify_embedded_assumptions(
            problem_statement, business_context
        )

        # Step 2: Dissolve each assumption through alternative framing
        dissolved_assumptions = []
        for assumption_data in identified_assumptions:
            dissolved = self._dissolve_single_assumption(
                assumption_data, business_context
            )
            if dissolved and dissolved.dissolution_strength >= 0.3:
                dissolved_assumptions.append(dissolved)

        # Step 3: Create idealized design vision
        idealized_design = self._create_idealized_design(
            problem_statement, dissolved_assumptions
        )

        # Step 4: Identify systems redesign opportunities
        systems_opportunities = self._identify_systems_redesign_opportunities(
            dissolved_assumptions, business_context
        )

        # Step 5: Generate fundamental reframes
        fundamental_reframes = self._generate_fundamental_reframes(
            problem_statement, dissolved_assumptions
        )

        # Step 6: Calculate impact and confidence scores
        impact_score = self._calculate_dissolution_impact(dissolved_assumptions)
        confidence = self._calculate_methodology_confidence(dissolved_assumptions)

        processing_time = (time.time() - start_time) * 1000

        print("📊 Dissolution Results:")
        print(f"├─ Assumptions dissolved: {len(dissolved_assumptions)}")
        print(f"├─ Impact score: {impact_score:.3f}")
        print(f"├─ Systems opportunities: {len(systems_opportunities)}")
        print(f"└─ Processing time: {processing_time:.1f}ms")

        # Display key dissolutions
        if dissolved_assumptions:
            print("\n💡 Key Assumption Dissolutions:")
            sorted_dissolutions = sorted(
                dissolved_assumptions,
                key=lambda x: x.dissolution_strength,
                reverse=True,
            )
            for i, dissolution in enumerate(sorted_dissolutions[:3], 1):
                print(
                    f"{i}. {dissolution.assumption_text} (strength: {dissolution.dissolution_strength:.2f})"
                )
                print(
                    f"   → Alternative: {dissolution.alternative_framings[0] if dissolution.alternative_framings else 'N/A'}"
                )

        return AssumptionDissolveResult(
            original_problem=problem_statement,
            dissolved_assumptions=dissolved_assumptions,
            dissolution_impact_score=impact_score,
            idealized_design_vision=idealized_design,
            systems_redesign_opportunities=systems_opportunities,
            fundamental_reframes=fundamental_reframes,
            processing_time_ms=processing_time,
            methodology_confidence=confidence,
        )

    def _identify_embedded_assumptions(
        self, problem_statement: str, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify assumptions embedded in the problem statement and context"""

        identified_assumptions = []
        statement_lower = problem_statement.lower()

        # Scan for assumption category indicators
        for category, config in self.assumption_categories.items():
            for indicator in config["indicators"]:
                if indicator in statement_lower:
                    # Extract assumption context
                    sentences = problem_statement.split(".")
                    relevant_sentences = [
                        s for s in sentences if indicator in s.lower()
                    ]

                    for sentence in relevant_sentences:
                        assumption_data = {
                            "category": category,
                            "text": sentence.strip(),
                            "indicator": indicator,
                            "context_source": "problem_statement",
                            "questions": config["questions"],
                        }
                        identified_assumptions.append(assumption_data)

        # Identify context-based assumptions
        context_assumptions = self._extract_context_assumptions(context)
        identified_assumptions.extend(context_assumptions)

        return identified_assumptions

    def _extract_context_assumptions(
        self, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract assumptions from business context"""

        context_assumptions = []

        # Budget constraint assumptions
        if context.get("financial_constraints"):
            context_assumptions.append(
                {
                    "category": "resource",
                    "text": f"Budget is constrained: {context['financial_constraints']}",
                    "indicator": "budget",
                    "context_source": "business_context",
                    "questions": [
                        "Is this budget constraint real or arbitrary?",
                        "How could we transcend this limitation?",
                    ],
                }
            )

        # Timeline pressure assumptions
        if context.get("timeline_pressure"):
            context_assumptions.append(
                {
                    "category": "operational",
                    "text": "Decision must be made quickly due to timeline pressure",
                    "indicator": "time",
                    "context_source": "business_context",
                    "questions": [
                        "Is this deadline real or artificial?",
                        "What would happen if we took more time?",
                    ],
                }
            )

        # Stakeholder assumptions
        stakeholders = context.get("stakeholders", [])
        if stakeholders:
            context_assumptions.append(
                {
                    "category": "contextual",
                    "text": f"Stakeholders are: {', '.join(stakeholders)}",
                    "indicator": "stakeholders",
                    "context_source": "business_context",
                    "questions": [
                        "Are these the only relevant stakeholders?",
                        "Who are we not considering?",
                    ],
                }
            )

        # Industry assumptions
        if context.get("industry"):
            context_assumptions.append(
                {
                    "category": "contextual",
                    "text": f"Operating in {context['industry']} industry",
                    "indicator": "industry",
                    "context_source": "business_context",
                    "questions": [
                        "Are industry constraints permanent?",
                        "How might industry boundaries dissolve?",
                    ],
                }
            )

        return context_assumptions

    def _dissolve_single_assumption(
        self, assumption_data: Dict[str, Any], context: Dict[str, Any]
    ) -> Optional[DissolvedAssumption]:
        """Dissolve a single assumption using Ackoff methodology"""

        assumption_text = assumption_data["text"]
        category = assumption_data["category"]

        # Generate alternative framings using Ackoff principles
        alternative_framings = self._generate_alternative_framings(
            assumption_text, category
        )

        # Apply idealized design thinking
        idealized_approach = self._apply_idealized_design(assumption_text, context)

        # Generate systems thinking insights
        systems_insights = self._generate_systems_insights(assumption_text, category)

        # Calculate dissolution strength
        dissolution_strength = self._calculate_dissolution_strength(
            alternative_framings, idealized_approach, systems_insights
        )

        # Generate practical implications
        practical_implications = self._determine_practical_implications(
            assumption_text, alternative_framings, context
        )

        if dissolution_strength < 0.2:  # Not meaningful dissolution
            return None

        return DissolvedAssumption(
            assumption_id=f"ackoff_{len(assumption_text)}_{category}",
            assumption_text=assumption_text,
            assumption_type=category,
            dissolution_strength=dissolution_strength,
            alternative_framings=alternative_framings[:3],  # Top 3 alternatives
            idealized_design_approach=idealized_approach,
            systems_thinking_insights=systems_insights,
            practical_implications=practical_implications,
        )

    def _generate_alternative_framings(
        self, assumption_text: str, category: str
    ) -> List[str]:
        """Generate alternative framings that dissolve the assumption"""

        alternatives = []

        # Category-specific dissolution approaches
        if category == "fundamental":
            alternatives.extend(
                [
                    f"What if the opposite of '{assumption_text}' were true?",
                    f"Under what conditions would '{assumption_text}' not apply?",
                    f"How might '{assumption_text}' be culturally or temporally specific?",
                ]
            )

        elif category == "operational":
            alternatives.extend(
                [
                    f"What if we eliminated the need for the process in '{assumption_text}'?",
                    f"How might technology make '{assumption_text}' obsolete?",
                    f"What if we inverted the workflow described in '{assumption_text}'?",
                ]
            )

        elif category == "resource":
            alternatives.extend(
                [
                    f"How could we make the constraint in '{assumption_text}' irrelevant?",
                    f"What if we had unlimited resources for '{assumption_text}'?",
                    f"How might we transcend the scarcity implied in '{assumption_text}'?",
                ]
            )

        elif category == "success":
            alternatives.extend(
                [
                    f"What if success were measured differently than '{assumption_text}' suggests?",
                    f"Whose interests are served by the success definition in '{assumption_text}'?",
                    f"How might other stakeholders define success differently from '{assumption_text}'?",
                ]
            )

        else:  # contextual
            alternatives.extend(
                [
                    f"What if the context in '{assumption_text}' changed fundamentally?",
                    f"How might the environment evolve to make '{assumption_text}' obsolete?",
                    f"What forces could dissolve the constraints in '{assumption_text}'?",
                ]
            )

        return alternatives

    def _apply_idealized_design(
        self, assumption_text: str, context: Dict[str, Any]
    ) -> str:
        """Apply Ackoff's idealized design methodology"""

        # Imagine the ideally designed system without current constraints
        idealized_elements = []

        # Apply each idealized design principle
        if (
            "technology" in assumption_text.lower()
            or "system" in assumption_text.lower()
        ):
            idealized_elements.append(
                "Perfect technology available to solve any technical challenge"
            )

        if "budget" in assumption_text.lower() or "cost" in assumption_text.lower():
            idealized_elements.append(
                "Unlimited resources for optimal solution implementation"
            )

        if "time" in assumption_text.lower() or "deadline" in assumption_text.lower():
            idealized_elements.append(
                "Sufficient time for thorough analysis and implementation"
            )

        if (
            "stakeholder" in assumption_text.lower()
            or "people" in assumption_text.lower()
        ):
            idealized_elements.append("Perfect stakeholder alignment and cooperation")

        if (
            "regulation" in assumption_text.lower()
            or "legal" in assumption_text.lower()
        ):
            idealized_elements.append(
                "Regulatory framework optimized for best outcomes"
            )

        if (
            "information" in assumption_text.lower()
            or "data" in assumption_text.lower()
        ):
            idealized_elements.append("Complete and perfect information availability")

        if not idealized_elements:
            idealized_elements.append(
                "System designed from scratch with no legacy constraints"
            )

        idealized_design = (
            "IDEALIZED DESIGN: If we could design this perfectly from scratch: "
        )
        idealized_design += "; ".join(idealized_elements[:3])  # Top 3 elements

        return idealized_design

    def _generate_systems_insights(
        self, assumption_text: str, category: str
    ) -> List[str]:
        """Generate systems thinking insights about the assumption"""

        systems_insights = []

        # Systems thinking questions from Ackoff methodology
        systems_insights.extend(
            [
                "How does this assumption serve the larger system purpose?",
                "What system interactions does this assumption enable or constrain?",
                "How might changing this assumption affect other system components?",
            ]
        )

        # Category-specific systems insights
        if category == "operational":
            systems_insights.append(
                "This operational assumption may optimize locally while suboptimizing globally"
            )

        elif category == "resource":
            systems_insights.append(
                "Resource assumptions often create artificial scarcity in abundant systems"
            )

        elif category == "contextual":
            systems_insights.append(
                "Contextual assumptions may be legacy artifacts of outdated system states"
            )

        return systems_insights[:4]  # Limit to 4 insights

    def _calculate_dissolution_strength(
        self,
        alternative_framings: List[str],
        idealized_approach: str,
        systems_insights: List[str],
    ) -> float:
        """Calculate how completely the assumption has been dissolved"""

        # Base strength from number of quality alternatives
        alternatives_strength = min(0.6, len(alternative_framings) * 0.15)

        # Idealized design contribution
        idealized_strength = (
            0.25 if idealized_approach and len(idealized_approach) > 50 else 0.1
        )

        # Systems thinking contribution
        systems_strength = min(0.25, len(systems_insights) * 0.06)

        total_strength = alternatives_strength + idealized_strength + systems_strength

        return min(1.0, total_strength)

    def _determine_practical_implications(
        self,
        assumption_text: str,
        alternative_framings: List[str],
        context: Dict[str, Any],
    ) -> str:
        """Determine practical implications of dissolving this assumption"""

        implications = []

        # Risk assessment
        if "must" in assumption_text.lower() or "critical" in assumption_text.lower():
            implications.append(
                "HIGH IMPACT: Dissolving this assumption could fundamentally change approach"
            )

        # Opportunity assessment
        if (
            "constraint" in assumption_text.lower()
            or "limited" in assumption_text.lower()
        ):
            implications.append(
                "OPPORTUNITY: Removing this constraint could unlock new possibilities"
            )

        # Implementation considerations
        if (
            "process" in assumption_text.lower()
            or "procedure" in assumption_text.lower()
        ):
            implications.append(
                "OPERATIONAL: Would require process redesign and change management"
            )

        # Stakeholder impact
        stakeholders = context.get("stakeholders", [])
        if stakeholders:
            implications.append(
                f"STAKEHOLDER: Would affect {', '.join(stakeholders[:2])} significantly"
            )

        if not implications:
            implications.append(
                "MODERATE: Dissolution would require careful analysis and gradual implementation"
            )

        return " | ".join(implications[:2])  # Top 2 implications

    def _create_idealized_design(
        self, problem_statement: str, dissolved_assumptions: List[DissolvedAssumption]
    ) -> str:
        """Create comprehensive idealized design vision"""

        if not dissolved_assumptions:
            return "Idealized design requires assumption dissolution analysis first"

        # Extract key dissolved themes
        dissolved_themes = set()
        for assumption in dissolved_assumptions:
            dissolved_themes.add(assumption.assumption_type)

        idealized_vision = "IDEALIZED DESIGN VISION: "

        vision_elements = []
        if "resource" in dissolved_themes:
            vision_elements.append(
                "Unlimited resources enable optimal solution selection"
            )

        if "operational" in dissolved_themes:
            vision_elements.append(
                "Processes designed for maximum effectiveness without legacy constraints"
            )

        if "contextual" in dissolved_themes:
            vision_elements.append(
                "Market and regulatory environment perfectly aligned with objectives"
            )

        if "fundamental" in dissolved_themes:
            vision_elements.append(
                "System designed from first principles without inherited assumptions"
            )

        if not vision_elements:
            vision_elements.append(
                "System optimized for stakeholder value without current limitations"
            )

        idealized_vision += ". ".join(vision_elements) + "."

        return idealized_vision

    def _identify_systems_redesign_opportunities(
        self, dissolved_assumptions: List[DissolvedAssumption], context: Dict[str, Any]
    ) -> List[str]:
        """Identify systems redesign opportunities from dissolved assumptions"""

        opportunities = []

        # Group dissolutions by type for pattern recognition
        dissolution_types = {}
        for assumption in dissolved_assumptions:
            assumption_type = assumption.assumption_type
            if assumption_type not in dissolution_types:
                dissolution_types[assumption_type] = []
            dissolution_types[assumption_type].append(assumption)

        # Generate type-specific opportunities
        for dissolution_type, assumptions in dissolution_types.items():
            if dissolution_type == "operational" and len(assumptions) >= 2:
                opportunities.append(
                    "PROCESS REDESIGN: Multiple operational assumptions suggest need for end-to-end workflow redesign"
                )

            elif dissolution_type == "resource" and len(assumptions) >= 2:
                opportunities.append(
                    "RESOURCE MODEL REDESIGN: Multiple resource assumptions suggest need for new resource acquisition/allocation model"
                )

            elif dissolution_type == "contextual":
                opportunities.append(
                    "MARKET POSITION REDESIGN: Contextual assumptions suggest opportunity to reshape market dynamics"
                )

            elif dissolution_type == "fundamental":
                opportunities.append(
                    "BUSINESS MODEL REDESIGN: Fundamental assumptions suggest need to reconsider core business model"
                )

        # Cross-cutting opportunities
        if len(dissolved_assumptions) >= 4:
            opportunities.append(
                "SYSTEM ARCHITECTURE REDESIGN: Multiple dissolved assumptions suggest comprehensive system redesign opportunity"
            )

        high_impact_dissolutions = [
            a for a in dissolved_assumptions if a.dissolution_strength >= 0.7
        ]
        if high_impact_dissolutions:
            opportunities.append(
                "STRATEGIC REDESIGN: High-impact dissolutions suggest need for strategic approach reconsideration"
            )

        return opportunities[:5]  # Top 5 opportunities

    def _generate_fundamental_reframes(
        self, problem_statement: str, dissolved_assumptions: List[DissolvedAssumption]
    ) -> List[str]:
        """Generate fundamental reframes based on dissolved assumptions"""

        reframes = []

        if not dissolved_assumptions:
            return ["Original framing maintained due to lack of dissolved assumptions"]

        # Generate reframes based on dissolution patterns
        high_impact_dissolutions = [
            a for a in dissolved_assumptions if a.dissolution_strength >= 0.6
        ]

        if high_impact_dissolutions:
            reframes.append(
                "Instead of solving the stated problem, consider designing the ideal system from scratch"
            )
            reframes.append(
                "Transform from problem-solving to opportunity-creation mindset"
            )

        # Category-specific reframes
        dissolution_categories = set(a.assumption_type for a in dissolved_assumptions)

        if "resource" in dissolution_categories:
            reframes.append(
                "Reframe from resource-constrained optimization to value creation maximization"
            )

        if "operational" in dissolution_categories:
            reframes.append(
                "Reframe from process improvement to process elimination and reinvention"
            )

        if "contextual" in dissolution_categories:
            reframes.append(
                "Reframe from adapting to environment to reshaping environmental conditions"
            )

        if "fundamental" in dissolution_categories:
            reframes.append(
                "Reframe from accepting current reality to questioning the foundations of that reality"
            )

        return reframes[:4]  # Top 4 reframes

    def _calculate_dissolution_impact(
        self, dissolved_assumptions: List[DissolvedAssumption]
    ) -> float:
        """Calculate overall impact of assumption dissolution"""

        if not dissolved_assumptions:
            return 0.0

        # Weight by dissolution strength
        total_weighted_impact = sum(
            assumption.dissolution_strength for assumption in dissolved_assumptions
        )
        average_impact = total_weighted_impact / len(dissolved_assumptions)

        # Bonus for high-impact dissolutions
        high_impact_count = len(
            [a for a in dissolved_assumptions if a.dissolution_strength >= 0.7]
        )
        high_impact_bonus = high_impact_count * 0.1

        # Bonus for assumption diversity
        unique_types = len(set(a.assumption_type for a in dissolved_assumptions))
        diversity_bonus = (unique_types / 5) * 0.2  # Max 5 assumption types

        total_impact = average_impact + high_impact_bonus + diversity_bonus

        return min(1.0, total_impact)

    def _calculate_methodology_confidence(
        self, dissolved_assumptions: List[DissolvedAssumption]
    ) -> float:
        """Calculate confidence in the Ackoff methodology application"""

        if not dissolved_assumptions:
            return 0.3  # Low confidence without dissolution

        # Base confidence from successful dissolutions
        base_confidence = min(0.7, len(dissolved_assumptions) * 0.15)

        # Quality bonus from high-strength dissolutions
        high_quality_dissolutions = [
            a for a in dissolved_assumptions if a.dissolution_strength >= 0.6
        ]
        quality_bonus = len(high_quality_dissolutions) * 0.1

        # Methodology completeness bonus
        completeness_bonus = 0.2 if len(dissolved_assumptions) >= 3 else 0.1

        total_confidence = base_confidence + quality_bonus + completeness_bonus

        return min(1.0, total_confidence)


async def demonstrate_ackoff_dissolution():
    """Demonstrate Ackoff assumption dissolution system"""

    dissolver = AckoffAssumptionDissolver()

    test_cases = [
        {
            "problem": "We must cut costs by 20% to remain competitive because the market is commoditized and customers only care about price",
            "context": {
                "industry": "Manufacturing",
                "financial_constraints": "Revenue declining 15% annually",
                "stakeholders": ["CFO", "CEO", "Operations Team"],
                "timeline_pressure": True,
            },
        },
        {
            "problem": "We need to hire 50 more engineers immediately because our current team cannot deliver the product roadmap on time",
            "context": {
                "industry": "Software Technology",
                "stakeholders": ["CTO", "Engineering Managers", "HR"],
                "stated_preferences": "Want to scale engineering rapidly",
            },
        },
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*20} ACKOFF DISSOLUTION TEST {i} {'='*20}")

        result = await dissolver.dissolve_assumptions(case["problem"], case["context"])

        print("\n🎯 ACKOFF DISSOLUTION RESULTS:")
        print(f"Impact Score: {result.dissolution_impact_score:.3f}")
        print(f"Methodology Confidence: {result.methodology_confidence:.3f}")
        print(f"Systems Opportunities: {len(result.systems_redesign_opportunities)}")

        print("\n🌟 IDEALIZED DESIGN VISION:")
        print(result.idealized_design_vision)

        if result.fundamental_reframes:
            print("\n🔄 FUNDAMENTAL REFRAMES:")
            for reframe in result.fundamental_reframes[:2]:
                print(f"• {reframe}")

        if result.systems_redesign_opportunities:
            print("\n🏗️ SYSTEMS REDESIGN OPPORTUNITIES:")
            for opportunity in result.systems_redesign_opportunities[:2]:
                print(f"• {opportunity}")

        if i < len(test_cases):
            print(f"\n{'NEXT TEST':=^80}\n")


if __name__ == "__main__":
    asyncio.run(demonstrate_ackoff_dissolution())
