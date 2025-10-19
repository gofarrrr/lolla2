"""
Domain-Specific Prompt Service - Research Domain Template Library
===============================================================

REFACTORING TARGET: Extract DomainSpecificPromptLibrary from enhanced_research_orchestrator.py
PATTERN: Service Extraction with Template Strategy
GOAL: Create focused, extensible domain prompt service

Responsibility:
- Domain pattern detection and classification
- Domain-specific prompt template generation
- Context primer creation for research queries
- Output specification and evidence requirements

Benefits:
- Single Responsibility Principle for prompt generation
- Easily extensible with new domain templates
- Clear prompt template interfaces
- Testable domain detection logic
"""

import logging
from typing import Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DomainPromptResult:
    """Result of domain-specific prompt generation"""

    domain_pattern: str
    prompt_template: str
    context_primer: str
    output_specification: str
    evidence_requirements: str


class DomainPromptService:
    """
    Domain-specific research prompt generation service

    Responsibility: Generate domain-tailored research prompts
    Complexity Target: Grade B (≤10 per method)
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Domain pattern keywords
        self.domain_patterns = {
            "market_analysis": [
                "market size",
                "market share",
                "industry analysis",
                "market research",
                "competitive landscape",
                "market trends",
                "industry outlook",
                "market dynamics",
            ],
            "competitor_analysis": [
                "competitors",
                "competitive analysis",
                "rival companies",
                "market competition",
                "competitor landscape",
                "competitive positioning",
                "competitor research",
            ],
            "investment_evaluation": [
                "investment",
                "valuation",
                "financial analysis",
                "roi analysis",
                "investment opportunity",
                "due diligence",
                "financial evaluation",
            ],
            "technology_trends": [
                "technology trends",
                "tech innovation",
                "emerging technologies",
                "technology analysis",
                "digital transformation",
                "tech developments",
                "innovation analysis",
            ],
            "feasibility_study": [
                "feasibility",
                "viability analysis",
                "feasibility study",
                "project evaluation",
                "implementation analysis",
                "viability assessment",
            ],
            "marketing_strategy": [
                "marketing strategy",
                "marketing plan",
                "promotional strategy",
                "marketing analysis",
                "brand strategy",
                "marketing campaigns",
                "customer acquisition",
            ],
            "health_research": [
                "health analysis",
                "medical research",
                "healthcare",
                "clinical analysis",
                "health outcomes",
                "medical evaluation",
                "health studies",
            ],
            "sustainability_analysis": [
                "sustainability",
                "environmental impact",
                "green analysis",
                "environmental assessment",
                "sustainable practices",
                "environmental research",
                "eco analysis",
            ],
            "policy_impact": [
                "policy analysis",
                "regulatory impact",
                "policy evaluation",
                "government policy",
                "regulatory analysis",
                "policy research",
                "policy assessment",
            ],
        }

    def generate_domain_prompt(
        self, research_query, strategy_decision: Dict[str, Any]
    ) -> DomainPromptResult:
        """
        Generate domain-specific research prompt

        Complexity: Target B (≤10)
        """
        query_text = research_query.query
        sophistication = research_query.sophistication_level

        # Detect domain pattern
        domain_pattern = self._detect_query_pattern(query_text)

        # Create domain-specific prompt
        prompt_template = self._create_domain_specific_prompt(
            research_query, domain_pattern, strategy_decision
        )

        # Generate supporting components
        context_primer = self._create_context_primer(research_query)
        output_spec = self._get_output_specification(sophistication)
        evidence_req = self._get_evidence_requirements(sophistication)

        return DomainPromptResult(
            domain_pattern=domain_pattern,
            prompt_template=prompt_template,
            context_primer=context_primer,
            output_specification=output_spec,
            evidence_requirements=evidence_req,
        )

    def _detect_query_pattern(self, query: str) -> str:
        """
        Detect domain pattern from query text

        Complexity: Target B (≤10)
        """
        query_lower = query.lower()
        pattern_scores = {}

        # Score each domain pattern
        for pattern, keywords in self.domain_patterns.items():
            score = sum(2 if keyword in query_lower else 0 for keyword in keywords)
            if score > 0:
                pattern_scores[pattern] = score

        # Return highest scoring pattern, or 'comprehensive' as default
        if pattern_scores:
            return max(pattern_scores.keys(), key=lambda k: pattern_scores[k])

        return "comprehensive_analysis"

    def _create_domain_specific_prompt(
        self, research_query, domain_pattern: str, strategy_decision: Dict[str, Any]
    ) -> str:
        """
        Create prompt based on detected domain pattern

        Complexity: Target B (≤10)
        """
        templates = {
            "market_analysis": self._get_market_analysis_template(),
            "competitor_analysis": self._get_competitor_analysis_template(),
            "investment_evaluation": self._get_investment_evaluation_template(),
            "technology_trends": self._get_technology_trends_template(),
            "feasibility_study": self._get_feasibility_study_template(),
            "marketing_strategy": self._get_marketing_strategy_template(),
            "health_research": self._get_health_research_template(),
            "sustainability_analysis": self._get_sustainability_analysis_template(),
            "policy_impact": self._get_policy_impact_template(),
            "comprehensive_analysis": self._get_comprehensive_analysis_template(),
        }

        base_template = templates.get(
            domain_pattern, templates["comprehensive_analysis"]
        )

        # Customize template with query specifics
        context_primer = self._create_context_primer(research_query)

        return f"{context_primer}\n\n{base_template}"

    def _create_context_primer(self, research_query) -> str:
        """
        Create context primer for research query

        Complexity: Target B (≤10)
        """
        sophistication = research_query.sophistication_level
        context = research_query.context or {}

        primer = f"Research Query: {research_query.query}\n"
        primer += f"Analysis Level: {sophistication.title()}\n"

        if context:
            primer += "Additional Context:\n"
            for key, value in context.items():
                primer += f"- {key.replace('_', ' ').title()}: {value}\n"

        return primer.strip()

    # Template Methods (Grade A complexity - simple string returns)

    def _get_market_analysis_template(self) -> str:
        """Market analysis research template"""
        return """Conduct comprehensive market analysis covering:

1. Market Size & Growth
   - Current market size and historical growth trends
   - Growth projections and market forecasts
   - Key market drivers and constraints

2. Market Structure & Dynamics
   - Market segmentation and key segments
   - Value chain analysis and profit distribution
   - Competitive intensity and market concentration

3. Trend Analysis & Future Outlook
   - Emerging trends and disruptions
   - Technology impact and digital transformation
   - Regulatory changes and market implications

Provide quantitative data, credible sources, and strategic insights."""

    def _get_competitor_analysis_template(self) -> str:
        """Competitor analysis research template"""
        return """Conduct comprehensive competitor analysis including:

1. Competitive Landscape
   - Key players and market positions
   - Market share analysis and competitive intensity
   - Competitive advantages and differentiators

2. Competitor Profiles
   - Business models and strategies
   - Financial performance and resources
   - Product/service offerings and positioning

3. Strategic Intelligence
   - Recent moves and strategic initiatives
   - Strengths, weaknesses, and vulnerabilities
   - Competitive threats and opportunities

Focus on actionable competitive intelligence with verified data."""

    def _get_investment_evaluation_template(self) -> str:
        """Investment evaluation research template"""
        return """Conduct thorough investment evaluation covering:

1. Investment Opportunity Assessment
   - Business model validation and scalability
   - Market opportunity and competitive position
   - Revenue potential and growth prospects

2. Risk Analysis
   - Market risks and competitive threats
   - Operational and execution risks
   - Regulatory and environmental risks

3. Financial Analysis
   - Valuation methodologies and benchmarks
   - Financial projections and assumptions
   - Return scenarios and exit strategies

Provide data-driven investment recommendation with supporting analysis."""

    def _get_technology_trends_template(self) -> str:
        """Technology trends research template"""
        return """Analyze technology trends and implications:

1. Technology Landscape
   - Current technology state and adoption
   - Emerging technologies and innovations
   - Technology maturity and development cycles

2. Impact Assessment
   - Industry transformation potential
   - Disruption risks and opportunities
   - Implementation challenges and requirements

3. Strategic Implications
   - Competitive implications and advantages
   - Investment requirements and timelines
   - Strategic recommendations and next steps

Focus on actionable technology intelligence with implementation guidance."""

    def _get_comprehensive_analysis_template(self) -> str:
        """Comprehensive analysis research template"""
        return """Conduct comprehensive research analysis including:

1. Situational Analysis
   - Current state assessment and key factors
   - Historical context and trend analysis
   - Stakeholder landscape and influences

2. Deep Investigation
   - Root cause analysis and contributing factors
   - Comparative analysis and benchmarking
   - Expert perspectives and industry insights

3. Strategic Synthesis
   - Key findings and critical insights
   - Implications and recommendations
   - Action items and next steps

Provide thorough, evidence-based analysis with actionable conclusions."""

    def _get_feasibility_study_template(self) -> str:
        """Feasibility study research template"""
        return """Conduct comprehensive feasibility study:

1. Technical Feasibility
   - Technical requirements and capabilities
   - Implementation complexity and challenges
   - Resource requirements and availability

2. Market Feasibility
   - Market demand and acceptance
   - Competitive landscape and positioning
   - Revenue potential and business model

3. Financial Feasibility
   - Investment requirements and funding
   - Cost structure and profitability
   - Risk assessment and mitigation

Provide clear feasibility recommendation with supporting rationale."""

    def _get_marketing_strategy_template(self) -> str:
        """Marketing strategy research template"""
        return """Develop comprehensive marketing strategy:

1. Market Understanding
   - Target audience analysis and segmentation
   - Customer needs and pain points
   - Market size and opportunity assessment

2. Competitive Analysis
   - Competitive landscape and positioning
   - Marketing strategies and tactics
   - Differentiation opportunities

3. Strategy Development
   - Marketing objectives and goals
   - Channel strategy and mix optimization
   - Messaging and positioning strategy

Focus on actionable marketing recommendations with implementation guidance."""

    def _get_health_research_template(self) -> str:
        """Health research template"""
        return """Conduct comprehensive health research analysis:

1. Evidence Review
   - Clinical evidence and research studies
   - Efficacy and safety data analysis  
   - Regulatory status and approvals

2. Health Impact Assessment
   - Population health implications
   - Risk-benefit analysis
   - Healthcare system impact

3. Implementation Considerations
   - Access and adoption barriers
   - Cost-effectiveness analysis
   - Policy and regulatory implications

Provide evidence-based health analysis with clinical validity."""

    def _get_sustainability_analysis_template(self) -> str:
        """Sustainability analysis template"""
        return """Conduct comprehensive sustainability analysis:

1. Environmental Impact
   - Carbon footprint and emissions analysis
   - Resource consumption and efficiency
   - Environmental risks and mitigation

2. Sustainability Framework
   - ESG criteria and compliance
   - Sustainability standards and certifications
   - Stakeholder expectations and requirements

3. Strategic Recommendations
   - Sustainability opportunities and initiatives
   - Implementation roadmap and priorities
   - Performance measurement and reporting

Focus on actionable sustainability strategies with measurable outcomes."""

    def _get_policy_impact_template(self) -> str:
        """Policy impact research template"""
        return """Analyze policy impact and implications:

1. Policy Analysis
   - Policy objectives and mechanisms
   - Regulatory framework and requirements
   - Implementation timeline and phases

2. Impact Assessment
   - Stakeholder impact and implications
   - Economic and social consequences
   - Industry and market effects

3. Strategic Response
   - Compliance requirements and strategies
   - Risk mitigation and opportunity capture
   - Advocacy and engagement strategies

Provide actionable policy intelligence with strategic recommendations."""

    def _get_output_specification(self, sophistication_level: str) -> str:
        """
        Get output specification based on sophistication level

        Complexity: Grade A (1)
        """
        specs = {
            "basic": "Provide clear, concise findings with key insights and recommendations.",
            "intermediate": "Provide detailed analysis with supporting data, insights, and actionable recommendations.",
            "advanced": "Provide comprehensive analysis with quantitative data, expert insights, strategic implications, and detailed recommendations.",
            "expert": "Provide expert-level analysis with rigorous data validation, multiple perspectives, strategic frameworks, and nuanced recommendations with implementation guidance.",
        }
        return specs.get(sophistication_level, specs["advanced"])

    def _get_evidence_requirements(self, sophistication_level: str) -> str:
        """
        Get evidence requirements based on sophistication level

        Complexity: Grade A (1)
        """
        requirements = {
            "basic": "Include credible sources and basic data validation.",
            "intermediate": "Include multiple credible sources, data validation, and source quality assessment.",
            "advanced": "Include diverse high-quality sources, rigorous data validation, source credibility analysis, and cross-reference verification.",
            "expert": "Include authoritative sources, comprehensive data validation, source reliability assessment, methodological review, and contradiction analysis.",
        }
        return requirements.get(sophistication_level, requirements["advanced"])


# Singleton instance for injection
_domain_prompt_service_instance = None


def get_domain_prompt_service() -> DomainPromptService:
    """Factory function for dependency injection"""
    global _domain_prompt_service_instance
    if _domain_prompt_service_instance is None:
        _domain_prompt_service_instance = DomainPromptService()
    return _domain_prompt_service_instance
