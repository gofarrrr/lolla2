#!/usr/bin/env python3
"""
N-Way Consultant Selection System
Select 3-5 specialized consultants for any business query
"""

import asyncio
import os
from supabase import create_client
from dotenv import load_dotenv
from typing import Dict, List, Any
from dataclasses import dataclass

load_dotenv()


@dataclass
class NWayConsultant:
    """Represents a specialized N-way consultant"""

    pattern_id: str
    consultant_name: str
    specialization: str
    expertise_areas: List[str]
    typical_problems: List[str]
    business_context: str
    model_count: int
    models_involved: List[str]
    relevance_score: float = 0.0


class NWayConsultantSelector:
    """Selects optimal N-way consultants for business queries"""

    def __init__(self):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_ANON_KEY")
        self.supabase = create_client(url, key)
        self._consultants = None

    async def _load_consultants(self):
        """Load and initialize all N-way consultants"""
        if self._consultants is not None:
            return

        result = self.supabase.table("nway_interactions").select("*").execute()
        patterns = result.data

        self._consultants = []
        for pattern in patterns:
            consultant = self._create_consultant_profile(pattern)
            self._consultants.append(consultant)

    def _create_consultant_profile(self, pattern: Dict[str, Any]) -> NWayConsultant:
        """Create a consultant profile from N-way pattern"""

        pattern_id = pattern.get("interaction_id", "")
        models = pattern.get("models_involved", [])

        # Map to proper consultant names and specializations
        consultant_profiles = {
            "NWAY_MOTIVATION_TRADEOFF_008": {
                "name": "Dr. Elena Vasquez - Change Catalyst",
                "specialization": "Change Management & Motivation Psychology",
                "expertise": [
                    "Employee engagement",
                    "Motivation systems",
                    "Change resistance",
                    "Performance psychology",
                ],
                "context": "Organizational Psychology & HR Strategy",
            },
            "NWAY_LEARNING_TEACHING_012": {
                "name": "Prof. Marcus Chen - Learning Architect",
                "specialization": "Corporate Learning & Development",
                "expertise": [
                    "Training design",
                    "Skill development",
                    "Knowledge transfer",
                    "Adult learning",
                ],
                "context": "Learning & Development Strategy",
            },
            "NWAY_ENTREPRENEUR_AGENCY_015": {
                "name": "Sarah Martinez - Venture Strategist",
                "specialization": "Entrepreneurship & Growth Strategy",
                "expertise": [
                    "Business model innovation",
                    "Growth strategy",
                    "Startup scaling",
                    "Strategic planning",
                ],
                "context": "Venture Strategy & Growth",
            },
            "NWAY_OUTLIER_ANALYSIS_017": {
                "name": "Dr. James Park - Data Intelligence Expert",
                "specialization": "Advanced Analytics & Pattern Recognition",
                "expertise": [
                    "Anomaly detection",
                    "Statistical analysis",
                    "Pattern recognition",
                    "Business intelligence",
                ],
                "context": "Data Science & Analytics",
            },
            "NWAY_NEGATIVE_SIMPLE_ACTION_021": {
                "name": "Rebecca Liu - Systems Risk Analyst",
                "specialization": "Risk Management & Systems Thinking",
                "expertise": [
                    "Unintended consequences",
                    "System failures",
                    "Complexity management",
                    "Risk assessment",
                ],
                "context": "Enterprise Risk & Process Design",
            },
            "NWAY_PERSUASION_010": {
                "name": "David Rodriguez - Influence Specialist",
                "specialization": "Persuasion & Stakeholder Alignment",
                "expertise": [
                    "Sales effectiveness",
                    "Negotiation strategy",
                    "Stakeholder buy-in",
                    "Communication",
                ],
                "context": "Sales Strategy & Stakeholder Management",
            },
            "NWAY_AUCTION_001": {
                "name": "Dr. Priya Patel - Market Designer",
                "specialization": "Market Design & Economic Strategy",
                "expertise": [
                    "Pricing strategy",
                    "Market mechanisms",
                    "Economic modeling",
                    "Auction theory",
                ],
                "context": "Strategic Economics & Pricing",
            },
            "NWAY_CREATIVITY_003": {
                "name": "Alex Thompson - Innovation Catalyst",
                "specialization": "Innovation & Creative Problem Solving",
                "expertise": [
                    "Design thinking",
                    "Product innovation",
                    "Creative processes",
                    "Innovation culture",
                ],
                "context": "Innovation Strategy & R&D",
            },
            "NWAY_DECISION_TRILEMMA_004": {
                "name": "Dr. Maria Santos - Decision Architect",
                "specialization": "Decision Science & Trade-off Analysis",
                "expertise": [
                    "Complex decisions",
                    "Trade-off analysis",
                    "Decision frameworks",
                    "Choice architecture",
                ],
                "context": "Strategic Decision Making",
            },
            "NWAY_ANALYST_CLUSTER_007": {
                "name": "Michael Kim - Business Intelligence Lead",
                "specialization": "Business Analysis & Market Intelligence",
                "expertise": [
                    "Business case development",
                    "Market analysis",
                    "Competitive intelligence",
                    "Strategic analysis",
                ],
                "context": "Strategy & Market Analysis",
            },
            "NWAY_ATYPICAL_SYNERGY_EXPLORATORY_011": {
                "name": "Dr. Lisa Zhang - Cross-Disciplinary Strategist",
                "specialization": "Integrative Strategy & Complex Problem Solving",
                "expertise": [
                    "Cross-functional integration",
                    "System synthesis",
                    "Complex problem solving",
                    "Strategic integration",
                ],
                "context": "Strategic Integration & Complexity",
            },
            "NWAY_DIAGNOSTIC_SOLVING_014": {
                "name": "Robert Johnson - Problem Solving Expert",
                "specialization": "Root Cause Analysis & Problem Diagnosis",
                "expertise": [
                    "Root cause analysis",
                    "System debugging",
                    "Performance improvement",
                    "Process optimization",
                ],
                "context": "Operational Excellence & Problem Solving",
            },
            "NWAY_TUPPERWARE_002": {
                "name": "Jennifer Wong - Distribution Strategist",
                "specialization": "Sales Channels & Distribution Strategy",
                "expertise": [
                    "Channel strategy",
                    "Network effects",
                    "Distribution optimization",
                    "Sales systems",
                ],
                "context": "Go-to-Market & Distribution",
            },
            "NWAY_UNCERTAINTY_DECISION_005": {
                "name": "Dr. Ahmed Hassan - Risk & Uncertainty Expert",
                "specialization": "Decision Making Under Uncertainty",
                "expertise": [
                    "Scenario planning",
                    "Risk assessment",
                    "Uncertainty analysis",
                    "Strategic foresight",
                ],
                "context": "Strategic Planning & Risk",
            },
            "NWAY_COCACOLA_006": {
                "name": "Emma Williams - Brand Strategist",
                "specialization": "Brand Strategy & Marketing Excellence",
                "expertise": [
                    "Brand positioning",
                    "Marketing strategy",
                    "Consumer insights",
                    "Brand architecture",
                ],
                "context": "Brand & Marketing Strategy",
            },
            "NWAY_STRATEGIST_CLUSTER_009": {
                "name": "Thomas Anderson - Strategy Director",
                "specialization": "Corporate Strategy & Competitive Intelligence",
                "expertise": [
                    "Corporate strategy",
                    "Competitive analysis",
                    "Market positioning",
                    "Strategic planning",
                ],
                "context": "Corporate Strategy",
            },
            "NWAY_PM_EXECUTION_013": {
                "name": "Monica Garcia - Execution Excellence Leader",
                "specialization": "Program Management & Delivery Excellence",
                "expertise": [
                    "Program management",
                    "Implementation strategy",
                    "Execution excellence",
                    "Change delivery",
                ],
                "context": "Operations & Program Delivery",
            },
            "NWAY_STORYTELLER_MARKETER_018": {
                "name": "Kevin O'Brien - Communications Strategist",
                "specialization": "Strategic Communications & Brand Storytelling",
                "expertise": [
                    "Brand storytelling",
                    "Content strategy",
                    "Communications planning",
                    "Message architecture",
                ],
                "context": "Marketing & Communications",
            },
            "NWAY_RESEARCHER_CLUSTER_016": {
                "name": "Dr. Samantha Lee - Research Director",
                "specialization": "Market Research & Consumer Insights",
                "expertise": [
                    "Market research",
                    "Consumer behavior",
                    "Insights generation",
                    "Research methodology",
                ],
                "context": "Market Research & Insights",
            },
            "NWAY_BIAS_MITIGATION_019": {
                "name": "Dr. Carlos Mendez - Decision Quality Expert",
                "specialization": "Decision Governance & Bias Mitigation",
                "expertise": [
                    "Decision governance",
                    "Cognitive bias detection",
                    "Process improvement",
                    "Quality assurance",
                ],
                "context": "Decision Excellence & Governance",
            },
            "NWAY_TRUST_COLLABORATION_020": {
                "name": "Rachel Cooper - Team Dynamics Specialist",
                "specialization": "Team Effectiveness & Collaboration Design",
                "expertise": [
                    "Team dynamics",
                    "Trust building",
                    "Collaboration systems",
                    "Organizational culture",
                ],
                "context": "Organizational Development",
            },
        }

        profile = consultant_profiles.get(
            pattern_id,
            {
                "name": f"Consultant - {pattern_id}",
                "specialization": "General Business Consulting",
                "expertise": ["Business strategy", "Problem solving"],
                "context": "General Consulting",
            },
        )

        return NWayConsultant(
            pattern_id=pattern_id,
            consultant_name=profile["name"],
            specialization=profile["specialization"],
            expertise_areas=profile["expertise"],
            typical_problems=profile[
                "expertise"
            ],  # Using expertise as typical problems
            business_context=profile["context"],
            model_count=len(models),
            models_involved=models,
        )

    async def select_consultants_for_query(
        self, query: str, max_consultants: int = 5
    ) -> List[NWayConsultant]:
        """Select top consultants for a business query"""

        await self._load_consultants()

        # Score consultants based on query relevance
        scored_consultants = []

        query_words = query.lower().split()

        for consultant in self._consultants:
            score = self._calculate_relevance_score(consultant, query_words)
            consultant.relevance_score = score

            if score > 0:  # Only include relevant consultants
                scored_consultants.append(consultant)

        # Sort by relevance score
        scored_consultants.sort(key=lambda c: c.relevance_score, reverse=True)

        # Return top consultants
        return scored_consultants[:max_consultants]

    def _calculate_relevance_score(
        self, consultant: NWayConsultant, query_words: List[str]
    ) -> float:
        """Calculate relevance score for consultant"""

        score = 0.0

        # Check specialization keywords
        spec_words = consultant.specialization.lower().split()
        for query_word in query_words:
            for spec_word in spec_words:
                if query_word in spec_word or spec_word in query_word:
                    score += 3.0

        # Check expertise areas
        for expertise in consultant.expertise_areas:
            expertise_words = expertise.lower().split()
            for query_word in query_words:
                for exp_word in expertise_words:
                    if query_word in exp_word or exp_word in query_word:
                        score += 2.0

        # Check business context
        context_words = consultant.business_context.lower().split()
        for query_word in query_words:
            for ctx_word in context_words:
                if query_word in ctx_word or ctx_word in query_word:
                    score += 1.0

        # Check pattern ID for direct matches
        pattern_words = consultant.pattern_id.lower().replace("_", " ").split()
        for query_word in query_words:
            for pattern_word in pattern_words:
                if query_word in pattern_word or pattern_word in query_word:
                    score += 1.5

        # Bonus for more models (more comprehensive analysis)
        score += consultant.model_count * 0.1

        return score

    async def demonstrate_consultant_selection(self):
        """Demonstrate consultant selection for various business problems"""

        print("🏢 N-WAY CONSULTANT SELECTION SYSTEM")
        print("=" * 80)

        test_queries = [
            "B2B SaaS startup with 15% monthly customer churn rate needs retention strategy",
            "Large corporation struggling with innovation culture and creative problem solving",
            "Sales team performance declining despite extensive training programs",
            "Startup needs strategic planning for European market entry",
            "Project delivery failing due to poor team collaboration and trust issues",
            "Complex pricing strategy needed for new marketplace platform",
            "Marketing campaign underperforming, need better brand storytelling",
            "Decision making process slow, too many biases affecting outcomes",
        ]

        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. QUERY: {query}")
            print("-" * 60)

            consultants = await self.select_consultants_for_query(
                query, max_consultants=3
            )

            if consultants:
                print("🎯 TOP 3 SELECTED CONSULTANTS:")
                for j, consultant in enumerate(consultants, 1):
                    print(f"\n   {j}. {consultant.consultant_name}")
                    print(f"      Specialization: {consultant.specialization}")
                    print(f"      Relevance Score: {consultant.relevance_score:.1f}")
                    print(
                        f"      Pattern: {consultant.pattern_id} ({consultant.model_count} models)"
                    )
                    print(
                        f"      Expertise: {', '.join(consultant.expertise_areas[:3])}..."
                    )
            else:
                print("   ❌ No relevant consultants found")


async def test_consultant_selector():
    """Test the consultant selection system"""

    selector = NWayConsultantSelector()
    await selector.demonstrate_consultant_selection()

    print("\n" + "=" * 80)
    print("🏁 CONSULTANT SELECTION DEMO COMPLETE")
    print("=" * 80)

    print("\n💡 KEY FEATURES:")
    print("✅ 21 Named consultants with specialized expertise")
    print("✅ Intelligent relevance scoring and matching")
    print("✅ Selects 3-5 most relevant consultants per query")
    print("✅ Ready for LLM integration with consultant-specific prompts")


if __name__ == "__main__":
    asyncio.run(test_consultant_selector())
