#!/usr/bin/env python3
"""
Domain Specialist Personas - Phase 2.1
Five specialized consultants for complete business domain coverage
Context-dependent selection based on problem classification
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio

from core_specialist_personas import (
    CoreSpecialistProfile,
    MentalModelAffinity,
    CognitiveBias,
)


class DomainSpecialization(Enum):
    """Domain specialization areas"""

    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    MARKETING = "marketing"
    TECHNOLOGY = "technology"
    ORGANIZATIONAL = "organizational"


@dataclass
class SelectionTrigger:
    """Criteria for when to select domain specialist"""

    keywords: List[str]
    financial_threshold: Optional[float] = None  # Dollar amounts
    problem_dimensions: List[str] = None
    complexity_threshold: Optional[float] = None
    urgency_level: Optional[str] = None


class DomainSpecialistFactory:
    """Factory for creating domain specialists with deep expertise"""

    def __init__(self):
        self.financial_model_catalog = self._build_financial_models()
        self.operational_model_catalog = self._build_operational_models()
        self.marketing_model_catalog = self._build_marketing_models()
        self.technology_model_catalog = self._build_technology_models()
        self.organizational_model_catalog = self._build_organizational_models()

    def _build_financial_models(self) -> Dict[str, Dict[str, Any]]:
        """Financial analysis mental models"""
        return {
            "cost-benefit-analysis": {
                "category": "financial_analysis",
                "complexity": "medium",
                "application": "investment_evaluation",
                "cfo_affinity": 1.0,
            },
            "opportunity-cost": {
                "category": "financial_analysis",
                "complexity": "low",
                "application": "resource_allocation",
                "cfo_affinity": 0.9,
            },
            "sunk-cost-fallacy": {
                "category": "financial_psychology",
                "complexity": "medium",
                "application": "decision_analysis",
                "cfo_affinity": 0.8,
            },
            "discounted-cash-flow": {
                "category": "valuation",
                "complexity": "high",
                "application": "investment_valuation",
                "cfo_affinity": 1.0,
            },
            "net-present-value": {
                "category": "valuation",
                "complexity": "high",
                "application": "project_evaluation",
                "cfo_affinity": 0.9,
            },
            "risk-assessment": {
                "category": "risk_management",
                "complexity": "medium",
                "application": "strategic_planning",
                "cfo_affinity": 0.8,
            },
        }

    def _build_operational_models(self) -> Dict[str, Dict[str, Any]]:
        """Operational excellence mental models"""
        return {
            "lean-thinking": {
                "category": "process_optimization",
                "complexity": "medium",
                "application": "waste_elimination",
                "coo_affinity": 1.0,
            },
            "six-sigma": {
                "category": "quality_management",
                "complexity": "high",
                "application": "quality_improvement",
                "coo_affinity": 1.0,
            },
            "theory-of-constraints": {
                "category": "process_optimization",
                "complexity": "medium",
                "application": "bottleneck_management",
                "coo_affinity": 0.9,
            },
            "kaizen": {
                "category": "continuous_improvement",
                "complexity": "low",
                "application": "incremental_improvement",
                "coo_affinity": 0.8,
            },
            "5s-methodology": {
                "category": "workplace_organization",
                "complexity": "low",
                "application": "operational_efficiency",
                "coo_affinity": 0.7,
            },
            "value-stream-mapping": {
                "category": "process_analysis",
                "complexity": "medium",
                "application": "process_improvement",
                "coo_affinity": 0.8,
            },
        }

    def _build_marketing_models(self) -> Dict[str, Dict[str, Any]]:
        """Marketing and brand strategy models"""
        return {
            "jobs-to-be-done": {
                "category": "customer_insight",
                "complexity": "medium",
                "application": "product_positioning",
                "cmo_affinity": 1.0,
            },
            "customer-development": {
                "category": "customer_insight",
                "complexity": "medium",
                "application": "market_validation",
                "cmo_affinity": 0.9,
            },
            "positioning-strategy": {
                "category": "brand_strategy",
                "complexity": "medium",
                "application": "market_positioning",
                "cmo_affinity": 1.0,
            },
            "brand-equity": {
                "category": "brand_strategy",
                "complexity": "high",
                "application": "brand_building",
                "cmo_affinity": 0.9,
            },
            "customer-journey-mapping": {
                "category": "customer_experience",
                "complexity": "medium",
                "application": "experience_optimization",
                "cmo_affinity": 0.8,
            },
            "viral-coefficient": {
                "category": "growth_strategy",
                "complexity": "low",
                "application": "viral_growth",
                "cmo_affinity": 0.7,
            },
        }

    def _build_technology_models(self) -> Dict[str, Dict[str, Any]]:
        """Technology and innovation models"""
        return {
            "disruptive-innovation": {
                "category": "innovation_strategy",
                "complexity": "high",
                "application": "strategic_innovation",
                "cto_affinity": 1.0,
            },
            "technology-adoption-curve": {
                "category": "technology_strategy",
                "complexity": "medium",
                "application": "adoption_planning",
                "cto_affinity": 0.9,
            },
            "platform-strategy": {
                "category": "technology_strategy",
                "complexity": "high",
                "application": "ecosystem_building",
                "cto_affinity": 0.9,
            },
            "minimum-viable-product": {
                "category": "product_development",
                "complexity": "low",
                "application": "product_validation",
                "cto_affinity": 0.8,
            },
            "technical-debt": {
                "category": "system_architecture",
                "complexity": "medium",
                "application": "system_maintenance",
                "cto_affinity": 0.8,
            },
            "scalability-patterns": {
                "category": "system_architecture",
                "complexity": "high",
                "application": "system_scaling",
                "cto_affinity": 0.7,
            },
        }

    def _build_organizational_models(self) -> Dict[str, Dict[str, Any]]:
        """Organizational development models"""
        return {
            "organizational-psychology": {
                "category": "org_development",
                "complexity": "high",
                "application": "culture_change",
                "chro_affinity": 1.0,
            },
            "change-management": {
                "category": "org_development",
                "complexity": "medium",
                "application": "transformation",
                "chro_affinity": 1.0,
            },
            "talent-development": {
                "category": "human_capital",
                "complexity": "medium",
                "application": "capability_building",
                "chro_affinity": 0.9,
            },
            "performance-management": {
                "category": "human_capital",
                "complexity": "medium",
                "application": "individual_performance",
                "chro_affinity": 0.8,
            },
            "organizational-design": {
                "category": "org_structure",
                "complexity": "high",
                "application": "structure_optimization",
                "chro_affinity": 0.9,
            },
            "leadership-development": {
                "category": "leadership",
                "complexity": "medium",
                "application": "executive_development",
                "chro_affinity": 0.8,
            },
        }

    def create_rebecca_kim_cfo(self) -> CoreSpecialistProfile:
        """Rebecca Kim - Chief Financial Officer"""

        # Tier 1 models (financial expertise core)
        tier1_models = [
            MentalModelAffinity(
                model="cost-benefit-analysis",
                affinity_score=1.0,
                reasoning="Fundamental CFO tool for evaluating all business decisions",
                usage_context="Primary framework for investment and resource allocation decisions",
            ),
            MentalModelAffinity(
                model="discounted-cash-flow",
                affinity_score=1.0,
                reasoning="Core valuation methodology for strategic financial analysis",
                usage_context="Valuing projects, acquisitions, and strategic initiatives",
            ),
            MentalModelAffinity(
                model="opportunity-cost",
                affinity_score=0.9,
                reasoning="Critical for resource allocation and strategic prioritization",
                usage_context="Ensuring optimal capital allocation across competing priorities",
            ),
            MentalModelAffinity(
                model="risk-assessment",
                affinity_score=0.8,
                reasoning="Essential for financial risk management and strategic planning",
                usage_context="Evaluating and mitigating financial and strategic risks",
            ),
        ]

        # Tier 2 models (secondary financial expertise)
        tier2_models = [
            MentalModelAffinity(
                model="net-present-value",
                affinity_score=0.9,
                reasoning="Standard financial evaluation tool",
                usage_context="Project and investment evaluation",
            ),
            MentalModelAffinity(
                model="sunk-cost-fallacy",
                affinity_score=0.8,
                reasoning="Important behavioral finance concept",
                usage_context="Avoiding poor continuation decisions",
            ),
        ]

        # Cognitive biases
        cognitive_biases = [
            CognitiveBias(
                bias_name="Financial Reductionism",
                description="Reduces complex strategic issues to financial metrics",
                blind_spot_area="Qualitative factors like culture, innovation, brand value",
                mitigation_strategy="Explicitly consider non-financial strategic factors",
            ),
            CognitiveBias(
                bias_name="Risk Aversion Bias",
                description="Over-emphasizes financial risks while undervaluing opportunities",
                blind_spot_area="High-potential but uncertain growth opportunities",
                mitigation_strategy="Balance risk assessment with opportunity evaluation",
            ),
            CognitiveBias(
                bias_name="Short-term Financial Focus",
                description="Prioritizes short-term financial performance over long-term value",
                blind_spot_area="Long-term strategic investments with delayed returns",
                mitigation_strategy="Use longer time horizons in financial analysis",
            ),
        ]

        return CoreSpecialistProfile(
            id="rebecca_kim_cfo",
            full_name="Rebecca Kim",
            title="Chief Financial Officer",
            core_expertise="Financial analysis, investment evaluation, risk assessment, capital allocation",
            signature_approach="Show me the numbers and the ROI - every decision has a financial impact that must be quantified",
            tier1_models=tier1_models,
            tier2_models=tier2_models,
            cognitive_strengths=[
                "Financial rigor",
                "Risk assessment",
                "Quantitative analysis",
                "Resource optimization",
            ],
            cognitive_biases=cognitive_biases,
            dimension_affinity={
                "financial": 1.0,
                "strategic": 0.7,
                "operational": 0.6,
                "crisis": 0.8,
                "technology": 0.5,
                "customer": 0.4,
                "organizational": 0.3,
            },
            communication_style="Numbers-focused, ROI-driven, uses financial metrics and business cases",
            decision_making_approach="Financially-driven, requires quantified business case and risk assessment",
            learning_style="Analytical - learns through financial data and business case studies",
            adaptation_speed=0.6,  # Moderate - finance requires careful validation
        )

    def create_david_kumar_coo(self) -> CoreSpecialistProfile:
        """David Kumar - Chief Operations Officer"""

        # Tier 1 models (operational excellence core)
        tier1_models = [
            MentalModelAffinity(
                model="lean-thinking",
                affinity_score=1.0,
                reasoning="Core operational philosophy for waste elimination and efficiency",
                usage_context="Primary framework for operational improvement and process optimization",
            ),
            MentalModelAffinity(
                model="six-sigma",
                affinity_score=1.0,
                reasoning="Statistical approach to quality improvement and defect reduction",
                usage_context="Quality management and process control initiatives",
            ),
            MentalModelAffinity(
                model="theory-of-constraints",
                affinity_score=0.9,
                reasoning="Critical for identifying and managing system bottlenecks",
                usage_context="Optimizing overall system performance and throughput",
            ),
            MentalModelAffinity(
                model="value-stream-mapping",
                affinity_score=0.8,
                reasoning="Essential tool for process analysis and improvement",
                usage_context="Visualizing and optimizing end-to-end processes",
            ),
        ]

        # Tier 2 models
        tier2_models = [
            MentalModelAffinity(
                model="kaizen",
                affinity_score=0.8,
                reasoning="Continuous improvement methodology",
                usage_context="Cultural change and incremental improvements",
            ),
            MentalModelAffinity(
                model="5s-methodology",
                affinity_score=0.7,
                reasoning="Workplace organization and efficiency",
                usage_context="Operational foundation and workplace optimization",
            ),
        ]

        # Cognitive biases
        cognitive_biases = [
            CognitiveBias(
                bias_name="Efficiency Obsession",
                description="Over-optimizes for efficiency at expense of effectiveness or innovation",
                blind_spot_area="Strategic flexibility and innovative experimentation",
                mitigation_strategy="Balance efficiency with adaptability and innovation capacity",
            ),
            CognitiveBias(
                bias_name="Process Rigidity",
                description="Prefers standardized processes over situational adaptation",
                blind_spot_area="Unique situations requiring process flexibility",
                mitigation_strategy="Build controlled variation and adaptation into process design",
            ),
            CognitiveBias(
                bias_name="Tactical Over Strategic",
                description="Focuses on operational tactics while missing strategic implications",
                blind_spot_area="Strategic impact of operational decisions",
                mitigation_strategy="Connect operational improvements to strategic outcomes",
            ),
        ]

        return CoreSpecialistProfile(
            id="david_kumar_coo",
            full_name="David Kumar",
            title="Chief Operations Officer",
            core_expertise="Process optimization, operational efficiency, quality management, supply chain excellence",
            signature_approach="Efficiency and quality through systematic process improvement and waste elimination",
            tier1_models=tier1_models,
            tier2_models=tier2_models,
            cognitive_strengths=[
                "Process optimization",
                "Quality management",
                "Systematic thinking",
                "Efficiency focus",
            ],
            cognitive_biases=cognitive_biases,
            dimension_affinity={
                "operational": 1.0,
                "financial": 0.7,
                "technology": 0.6,
                "strategic": 0.5,
                "organizational": 0.6,
                "customer": 0.5,
                "crisis": 0.7,
            },
            communication_style="Process-focused, metrics-driven, uses operational KPIs and efficiency measures",
            decision_making_approach="Operations-first, focuses on scalability and process excellence",
            learning_style="Systematic - learns through process analysis and operational metrics",
            adaptation_speed=0.5,  # Lower - operations requires stability and proven methods
        )

    def create_lisa_zhang_cmo(self) -> CoreSpecialistProfile:
        """Lisa Zhang - Chief Marketing Officer"""

        # Tier 1 models (marketing expertise core)
        tier1_models = [
            MentalModelAffinity(
                model="jobs-to-be-done",
                affinity_score=1.0,
                reasoning="Fundamental framework for understanding customer needs and market positioning",
                usage_context="Primary tool for product positioning and market strategy",
            ),
            MentalModelAffinity(
                model="positioning-strategy",
                affinity_score=1.0,
                reasoning="Core marketing competency for brand and competitive positioning",
                usage_context="Defining brand position and competitive differentiation",
            ),
            MentalModelAffinity(
                model="customer-development",
                affinity_score=0.9,
                reasoning="Essential for market validation and customer insight",
                usage_context="Understanding market needs and validating product-market fit",
            ),
            MentalModelAffinity(
                model="brand-equity",
                affinity_score=0.9,
                reasoning="Critical for long-term brand building and value creation",
                usage_context="Building sustainable brand value and market position",
            ),
        ]

        # Tier 2 models
        tier2_models = [
            MentalModelAffinity(
                model="customer-journey-mapping",
                affinity_score=0.8,
                reasoning="Important for customer experience optimization",
                usage_context="Improving customer experience and touchpoint optimization",
            ),
            MentalModelAffinity(
                model="viral-coefficient",
                affinity_score=0.7,
                reasoning="Key metric for growth strategy",
                usage_context="Optimizing viral and referral growth mechanisms",
            ),
        ]

        # Cognitive biases
        cognitive_biases = [
            CognitiveBias(
                bias_name="Customer Advocacy Bias",
                description="Over-emphasizes customer perspective while undervaluing business constraints",
                blind_spot_area="Operational feasibility and financial constraints",
                mitigation_strategy="Balance customer needs with business capability and profitability",
            ),
            CognitiveBias(
                bias_name="Brand Perfectionism",
                description="Pursues brand perfection at expense of speed and iteration",
                blind_spot_area="Fast iteration and testing opportunities",
                mitigation_strategy="Embrace iterative brand building and rapid testing",
            ),
            CognitiveBias(
                bias_name="Creative Over Analytical",
                description="Prefers creative solutions over data-driven optimization",
                blind_spot_area="Quantitative optimization and performance measurement",
                mitigation_strategy="Balance creativity with rigorous measurement and testing",
            ),
        ]

        return CoreSpecialistProfile(
            id="lisa_zhang_cmo",
            full_name="Lisa Zhang",
            title="Chief Marketing Officer",
            core_expertise="Brand strategy, customer insight, market positioning, growth marketing",
            signature_approach="Understanding customers and building compelling brand narratives that drive sustainable growth",
            tier1_models=tier1_models,
            tier2_models=tier2_models,
            cognitive_strengths=[
                "Customer insight",
                "Brand building",
                "Market positioning",
                "Growth strategy",
            ],
            cognitive_biases=cognitive_biases,
            dimension_affinity={
                "customer": 1.0,
                "strategic": 0.8,
                "organizational": 0.5,
                "financial": 0.6,
                "technology": 0.5,
                "operational": 0.4,
                "crisis": 0.6,
            },
            communication_style="Customer-centric, brand-focused, uses market insights and customer stories",
            decision_making_approach="Customer-first, considers brand impact and market dynamics",
            learning_style="Experiential - learns through market testing and customer feedback",
            adaptation_speed=0.8,  # High - marketing requires rapid adaptation to market changes
        )

    def create_alex_thompson_cto(self) -> CoreSpecialistProfile:
        """Alex Thompson - Chief Technology Officer"""

        # Tier 1 models (technology expertise core)
        tier1_models = [
            MentalModelAffinity(
                model="disruptive-innovation",
                affinity_score=1.0,
                reasoning="Core framework for understanding technology disruption and innovation strategy",
                usage_context="Primary lens for technology strategy and competitive positioning",
            ),
            MentalModelAffinity(
                model="platform-strategy",
                affinity_score=0.9,
                reasoning="Essential for modern technology architecture and ecosystem thinking",
                usage_context="Building scalable technology platforms and developer ecosystems",
            ),
            MentalModelAffinity(
                model="technology-adoption-curve",
                affinity_score=0.9,
                reasoning="Critical for timing technology investments and market entry",
                usage_context="Strategic timing of technology adoption and rollout",
            ),
            MentalModelAffinity(
                model="scalability-patterns",
                affinity_score=0.7,
                reasoning="Essential for building systems that can grow with business needs",
                usage_context="Architecture decisions for long-term system scalability",
            ),
        ]

        # Tier 2 models
        tier2_models = [
            MentalModelAffinity(
                model="minimum-viable-product",
                affinity_score=0.8,
                reasoning="Key approach for rapid product development and validation",
                usage_context="Product development strategy and rapid iteration",
            ),
            MentalModelAffinity(
                model="technical-debt",
                affinity_score=0.8,
                reasoning="Important concept for long-term system health",
                usage_context="Balancing speed with system maintainability",
            ),
        ]

        # Cognitive biases
        cognitive_biases = [
            CognitiveBias(
                bias_name="Technology Optimism",
                description="Over-estimates technology's ability to solve complex problems",
                blind_spot_area="Human, organizational, and market factors affecting technology success",
                mitigation_strategy="Consider human and organizational factors in technology solutions",
            ),
            CognitiveBias(
                bias_name="Technical Perfectionism",
                description="Pursues technical excellence at expense of business timeline and constraints",
                blind_spot_area="Business urgency and 'good enough' solutions",
                mitigation_strategy="Balance technical quality with business needs and timelines",
            ),
            CognitiveBias(
                bias_name="Innovation Bias",
                description="Prefers new, innovative solutions over proven, reliable approaches",
                blind_spot_area="Stable, proven technology solutions that meet business needs",
                mitigation_strategy="Consider boring, reliable technology when it fits business requirements",
            ),
        ]

        return CoreSpecialistProfile(
            id="alex_thompson_cto",
            full_name="Alex Thompson",
            title="Chief Technology Officer",
            core_expertise="Technology strategy, digital transformation, innovation management, system architecture",
            signature_approach="Technology as strategic enabler and competitive advantage through smart innovation",
            tier1_models=tier1_models,
            tier2_models=tier2_models,
            cognitive_strengths=[
                "Technical vision",
                "Innovation strategy",
                "System thinking",
                "Scalability focus",
            ],
            cognitive_biases=cognitive_biases,
            dimension_affinity={
                "technology": 1.0,
                "strategic": 0.8,
                "operational": 0.6,
                "financial": 0.5,
                "customer": 0.6,
                "organizational": 0.5,
                "crisis": 0.7,
            },
            communication_style="Innovation-focused, technically precise, uses technology trends and possibilities",
            decision_making_approach="Technology-enabled, considers scalability and competitive advantage",
            learning_style="Experimental - learns through prototyping and technical exploration",
            adaptation_speed=0.9,  # Very high - technology moves fast
        )

    def create_jennifer_walsh_chro(self) -> CoreSpecialistProfile:
        """Jennifer Walsh - Chief Human Resources Officer"""

        # Tier 1 models (organizational expertise core)
        tier1_models = [
            MentalModelAffinity(
                model="organizational-psychology",
                affinity_score=1.0,
                reasoning="Core competency for understanding and designing organizational behavior",
                usage_context="Primary framework for culture change and organizational development",
            ),
            MentalModelAffinity(
                model="change-management",
                affinity_score=1.0,
                reasoning="Essential for leading organizational transformation and adaptation",
                usage_context="Managing large-scale organizational change initiatives",
            ),
            MentalModelAffinity(
                model="talent-development",
                affinity_score=0.9,
                reasoning="Critical for building organizational capability and individual growth",
                usage_context="Developing human capital and organizational capabilities",
            ),
            MentalModelAffinity(
                model="organizational-design",
                affinity_score=0.9,
                reasoning="Important for optimizing organizational structure and effectiveness",
                usage_context="Designing organizational structures for optimal performance",
            ),
        ]

        # Tier 2 models
        tier2_models = [
            MentalModelAffinity(
                model="performance-management",
                affinity_score=0.8,
                reasoning="Key tool for individual and team performance optimization",
                usage_context="Managing individual and team performance systems",
            ),
            MentalModelAffinity(
                model="leadership-development",
                affinity_score=0.8,
                reasoning="Essential for building leadership pipeline and capability",
                usage_context="Developing leadership talent across the organization",
            ),
        ]

        # Cognitive biases
        cognitive_biases = [
            CognitiveBias(
                bias_name="People-First Bias",
                description="Prioritizes employee welfare over business performance requirements",
                blind_spot_area="Business performance requirements and competitive pressures",
                mitigation_strategy="Balance employee welfare with business performance needs",
            ),
            CognitiveBias(
                bias_name="Change Complexity Underestimation",
                description="Underestimates difficulty and timeline of organizational change",
                blind_spot_area="Organizational inertia and resistance to change",
                mitigation_strategy="Build realistic change timelines with adequate support systems",
            ),
            CognitiveBias(
                bias_name="Consensus Preference",
                description="Over-values consensus and harmony in decision-making",
                blind_spot_area="Situations requiring decisive action despite disagreement",
                mitigation_strategy="Develop comfort with necessary conflict and decisive leadership",
            ),
        ]

        return CoreSpecialistProfile(
            id="jennifer_walsh_chro",
            full_name="Jennifer Walsh",
            title="Chief Human Resources Officer",
            core_expertise="Organizational development, change management, talent strategy, culture transformation",
            signature_approach="Organizations succeed through their people - let's build the capability for sustainable performance",
            tier1_models=tier1_models,
            tier2_models=tier2_models,
            cognitive_strengths=[
                "Organizational insight",
                "Change leadership",
                "Talent development",
                "Culture building",
            ],
            cognitive_biases=cognitive_biases,
            dimension_affinity={
                "organizational": 1.0,
                "strategic": 0.6,
                "customer": 0.5,
                "crisis": 0.7,
                "technology": 0.4,
                "operational": 0.5,
                "financial": 0.4,
            },
            communication_style="People-focused, development-oriented, uses organizational insights and change examples",
            decision_making_approach="Human capital focus, considers organizational capability and change readiness",
            learning_style="Collaborative - learns through organizational interaction and change experience",
            adaptation_speed=0.7,  # Good - HR must balance stability with change
        )

    def create_selection_triggers(self) -> Dict[str, SelectionTrigger]:
        """Define when each domain specialist should be selected"""
        return {
            "rebecca_kim_cfo": SelectionTrigger(
                keywords=[
                    "revenue",
                    "cost",
                    "profit",
                    "investment",
                    "budget",
                    "financial",
                    "ROI",
                    "valuation",
                    "funding",
                ],
                financial_threshold=1000000,  # $1M+ triggers CFO
                problem_dimensions=["financial", "strategic"],
                complexity_threshold=0.6,
            ),
            "david_kumar_coo": SelectionTrigger(
                keywords=[
                    "operations",
                    "process",
                    "efficiency",
                    "quality",
                    "production",
                    "supply",
                    "workflow",
                ],
                problem_dimensions=["operational", "crisis"],
                complexity_threshold=0.4,
            ),
            "lisa_zhang_cmo": SelectionTrigger(
                keywords=[
                    "customer",
                    "brand",
                    "marketing",
                    "acquisition",
                    "retention",
                    "market",
                    "positioning",
                ],
                problem_dimensions=["customer", "strategic"],
                complexity_threshold=0.3,
            ),
            "alex_thompson_cto": SelectionTrigger(
                keywords=[
                    "technology",
                    "digital",
                    "innovation",
                    "platform",
                    "system",
                    "automation",
                    "tech",
                ],
                problem_dimensions=["technology", "strategic"],
                complexity_threshold=0.5,
            ),
            "jennifer_walsh_chro": SelectionTrigger(
                keywords=[
                    "culture",
                    "organization",
                    "team",
                    "leadership",
                    "people",
                    "change",
                    "talent",
                ],
                problem_dimensions=["organizational", "crisis"],
                complexity_threshold=0.7,
                urgency_level="high",
            ),
        }

    def create_all_domain_specialists(self) -> Dict[str, CoreSpecialistProfile]:
        """Create all five domain specialists"""
        return {
            "rebecca_kim_cfo": self.create_rebecca_kim_cfo(),
            "david_kumar_coo": self.create_david_kumar_coo(),
            "lisa_zhang_cmo": self.create_lisa_zhang_cmo(),
            "alex_thompson_cto": self.create_alex_thompson_cto(),
            "jennifer_walsh_chro": self.create_jennifer_walsh_chro(),
        }

    def validate_domain_coverage(
        self, specialists: Dict[str, CoreSpecialistProfile]
    ) -> Dict[str, Any]:
        """Validate complete business domain coverage"""

        required_domains = [
            "financial",
            "operational",
            "customer",
            "technology",
            "organizational",
        ]
        domain_coverage = {}

        for domain in required_domains:
            covering_specialists = []
            for specialist in specialists.values():
                if specialist.dimension_affinity.get(domain, 0) >= 0.7:
                    covering_specialists.append(specialist.full_name)

            domain_coverage[domain] = {
                "covered": len(covering_specialists) > 0,
                "specialists": covering_specialists,
                "coverage_strength": max(
                    [s.dimension_affinity.get(domain, 0) for s in specialists.values()]
                ),
            }

        overall_coverage = len(
            [d for d in domain_coverage.values() if d["covered"]]
        ) / len(required_domains)

        return {
            "overall_coverage": overall_coverage,
            "domain_coverage": domain_coverage,
            "total_specialists": len(specialists),
            "coverage_gaps": [
                domain
                for domain, info in domain_coverage.items()
                if not info["covered"]
            ],
        }


async def demonstrate_domain_specialists():
    """Demonstrate domain specialist system"""

    factory = DomainSpecialistFactory()
    specialists = factory.create_all_domain_specialists()
    selection_triggers = factory.create_selection_triggers()

    print("🏢 DOMAIN SPECIALIST PERSONAS - PHASE 2.1")
    print("=" * 80)

    for specialist_id, specialist in specialists.items():
        print(f"\n💼 {specialist.full_name}")
        print(f"   Title: {specialist.title}")
        print(f"   Expertise: {specialist.core_expertise}")
        print(f"   Approach: {specialist.signature_approach}")

        print("\n   🧠 TIER 1 MENTAL MODELS:")
        for model in specialist.tier1_models:
            print(
                f"   ├─ {model.model.replace('-', ' ').title()} (affinity: {model.affinity_score:.1f})"
            )

        print("\n   🎯 TOP DIMENSION AFFINITIES:")
        top_affinities = sorted(
            specialist.dimension_affinity.items(), key=lambda x: x[1], reverse=True
        )[:3]
        for dim, score in top_affinities:
            bar = "█" * int(score * 10)
            print(f"   ├─ {dim.title()}: {score:.1f} {bar}")

        # Show selection trigger
        trigger = selection_triggers.get(specialist_id, None)
        if trigger:
            print("\n   🔍 SELECTION TRIGGERS:")
            print(f"   ├─ Keywords: {', '.join(trigger.keywords[:5])}...")
            if trigger.financial_threshold:
                print(f"   ├─ Financial Threshold: ${trigger.financial_threshold:,}")
            if trigger.problem_dimensions:
                print(
                    f"   └─ Problem Dimensions: {', '.join(trigger.problem_dimensions)}"
                )

        print("-" * 80)

    # Validate domain coverage
    coverage = factory.validate_domain_coverage(specialists)

    print("\n📊 DOMAIN COVERAGE ANALYSIS:")
    print(f"├─ Overall Coverage: {coverage['overall_coverage']:.1%}")
    print(f"├─ Total Domain Specialists: {coverage['total_specialists']}")
    print(
        f"└─ Coverage Gaps: {len(coverage['coverage_gaps'])} ({'None' if len(coverage['coverage_gaps']) == 0 else ', '.join(coverage['coverage_gaps'])})"
    )

    print("\n🎯 DOMAIN-SPECIFIC COVERAGE:")
    for domain, info in coverage["domain_coverage"].items():
        status = "✅" if info["covered"] else "❌"
        print(
            f"{status} {domain.title()}: {info['coverage_strength']:.1f} max affinity"
        )
        if info["specialists"]:
            print(f"    Covered by: {', '.join(info['specialists'])}")

    print("\n✅ DOMAIN SPECIALISTS READY FOR INTEGRATION")


if __name__ == "__main__":
    asyncio.run(demonstrate_domain_specialists())
