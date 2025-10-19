#!/usr/bin/env python3
"""
Cognitive Analyzer Module
Provider-agnostic cognitive analysis methods using MeMo mental models framework
"""

import json
import logging
from typing import Dict, List, Optional, Any

from .provider_interface import CognitiveAnalysisResult


class CognitiveAnalyzer:
    """Provider-agnostic cognitive analysis using mental models framework"""

    def __init__(self, llm_client):
        """Initialize with a unified LLM client"""
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)

    async def analyze_problem_structure_with_research(
        self,
        problem_statement: str,
        business_context: Dict[str, Any],
        research_data: Optional[Dict] = None,
        engagement_id: Optional[str] = None,
    ) -> CognitiveAnalysisResult:
        """Analyze problem structure using MECE framework with research enhancement"""

        research_section = ""
        research_context_note = ""
        if research_data:
            # Simplified research integration that won't break JSON formatting
            research_summary = research_data.get(
                "research_summary", "External research data available"
            )
            sources_count = research_data.get("sources_count", 0)
            research_type = research_data.get("research_type", "general")

            research_section = f"""
RESEARCH CONTEXT:
- Research Type: {research_type}
- External Sources: {sources_count} sources accessed
- Research Summary: {research_summary}
"""
            research_context_note = f"\nIMPORTANT: This analysis is enhanced with external research data ({sources_count} sources). Integrate these insights into your MECE breakdown while maintaining strict JSON format."

        # Analyze problem context to create dynamic prompts
        industry = business_context.get("industry", "business")
        company_size = business_context.get("company_size", "organization")
        decision_type = business_context.get("decision_type", "strategic decision")
        complexity = business_context.get("complexity", "medium")

        # Create context-aware prompt based on problem characteristics
        context_specific_guidance = self._get_context_specific_guidance(
            industry, company_size, decision_type
        )

        prompt = f"""
You are a McKinsey-grade strategic consultant applying the MECE (Mutually Exclusive, Collectively Exhaustive) framework to a specific {industry} {decision_type}.

PROBLEM STATEMENT:
{problem_statement}

BUSINESS CONTEXT:
{json.dumps(business_context, indent=2)}

{research_section}

CONTEXT-SPECIFIC ANALYSIS FOCUS:
{context_specific_guidance}

{research_context_note}

TASK: Apply MECE framework with deep specificity to THIS exact problem context. 

**CRITICAL JSON FORMATTING REQUIREMENTS:**
- Response MUST be valid JSON only
- NO markdown formatting (no ```)
- NO explanatory text outside JSON
- START response with {{ and END with }}

JSON Response Format:

{{
  "mental_models_selected": ["mece_structuring", "relevant mental models based on this specific context"],
  "reasoning_description": "Detailed explanation of how MECE was applied specifically to this {industry} {decision_type}, referencing actual problem elements",
  "key_insights": ["specific insights about this exact problem", "context-relevant breakthrough insights", "actionable implications for this situation"],
  "confidence_score": 0.85,
  "research_requirements": ["specific data gaps for this exact problem", "industry-specific research needs"],
  "problem_breakdown": {{
    "main_components": ["SPECIFIC components relevant to this {industry} context"],
    "decision_factors": ["SPECIFIC factors for this {decision_type}"],
    "stakeholder_impacts": ["SPECIFIC stakeholders affected by this exact problem"]
  }}
}}

REQUIREMENTS:
- Every element must be specific to this exact problem context
- Reference actual details from the problem statement and business context
- Return ONLY the JSON object above, no additional text
"""

        messages = [
            {
                "role": "system",
                "content": "You are a world-class strategic consultant specializing in MECE problem structuring with research integration.",
            },
            {"role": "user", "content": prompt},
        ]

        # Make the call through unified client
        response = await self.llm_client.call_best_available_provider(
            messages,
            phase="analyze_problem_structure_with_research",
            engagement_id=engagement_id,
        )

        # Parse JSON response with enhanced error handling
        return self._parse_structured_response(
            response, "mece_structuring", research_data is not None
        )

    async def analyze_problem_structure(
        self,
        problem_statement: str,
        business_context: Dict[str, Any],
        engagement_id: Optional[str] = None,
    ) -> CognitiveAnalysisResult:
        """Analyze problem structure using MECE framework"""

        prompt = f"""
You are a McKinsey-grade strategic consultant using the MECE (Mutually Exclusive, Collectively Exhaustive) framework.

PROBLEM STATEMENT:
{problem_statement}

BUSINESS CONTEXT:
{json.dumps(business_context, indent=2)}

TASK: Analyze this problem using MECE framework and respond in JSON format:

{{
  "mental_models_selected": ["mece_structuring", "systems_thinking", "..."],
  "reasoning_description": "Clear description of how MECE was applied to this specific problem",
  "key_insights": ["insight 1", "insight 2", "insight 3"],
  "confidence_score": 0.85,
  "research_requirements": ["specific data needed", "market research needed", "..."],
  "problem_breakdown": {{
    "main_components": ["component 1", "component 2", "..."],
    "decision_factors": ["factor 1", "factor 2", "..."],
    "stakeholder_impacts": ["impact 1", "impact 2", "..."]
  }}
}}

Focus on the SPECIFIC problem context. Do not use generic templates.
"""

        messages = [
            {
                "role": "system",
                "content": "You are a world-class strategic consultant specializing in MECE problem structuring.",
            },
            {"role": "user", "content": prompt},
        ]

        # Make the call through unified client
        response = await self.llm_client.call_best_available_provider(
            messages,
            phase="analyze_problem_structure",
            engagement_id=engagement_id,
            context_data={
                "method": "analyze_problem_structure",
                "problem_statement": (
                    problem_statement[:200] + "..."
                    if len(problem_statement) > 200
                    else problem_statement
                ),
                "business_context_keys": list(business_context.keys()),
            },
        )

        return self._parse_structured_response(response, "mece_structuring", False)

    async def generate_hypotheses(
        self,
        problem_statement: str,
        research_data: Optional[Dict] = None,
        previous_analysis: Optional[str] = None,
        **kwargs,
    ) -> CognitiveAnalysisResult:
        """Generate testable hypotheses with research integration and dynamic context awareness"""

        # Extract problem context for dynamic prompt generation
        focus_areas = self._analyze_problem_focus(problem_statement)

        research_section = ""
        research_integration_guidance = ""
        if research_data:
            research_section = f"""
EXTERNAL RESEARCH DATA:
{json.dumps(research_data, indent=2)}
"""
            research_integration_guidance = """
RESEARCH INTEGRATION REQUIREMENT:
- Generate hypotheses that leverage specific insights from the external research
- Reference actual data points and trends identified in the research
- Create testable predictions that can be validated against market data
- Ensure hypotheses reflect real-world constraints and opportunities identified in research
"""

        prompt = f"""
You are a McKinsey-grade strategic consultant generating testable hypotheses for a complex strategic challenge.

PROBLEM STATEMENT:
{problem_statement}

{research_section}

PROBLEM ANALYSIS:
This problem requires hypothesis generation focused on: {focus_areas}

{research_integration_guidance}

TASK: Generate testable, specific hypotheses that directly address this problem context. Respond in JSON format:

{{
  "mental_models_selected": ["hypothesis_testing", "scenario_analysis", "mental models relevant to {focus_areas}"],
  "reasoning_description": "Detailed explanation of why these specific hypotheses were generated for this {focus_areas} challenge, incorporating research insights",
  "key_insights": ["hypothesis 1 specific to this problem", "hypothesis 2 addressing core uncertainties", "hypothesis 3 testing critical assumptions"],
  "confidence_score": 0.85,
  "research_requirements": ["specific data needed to validate these exact hypotheses"],
  "hypotheses": [
    {{
      "hypothesis": "SPECIFIC testable statement about this exact problem",
      "testable_prediction": "SPECIFIC measurable outcome if this hypothesis is true",
      "validation_method": "SPECIFIC approach to test this hypothesis in this context",
      "confidence": 0.8,
      "research_support": "How external research supports or challenges this hypothesis"
    }}
  ]
}}

CRITICAL REQUIREMENTS:
1. Every hypothesis must be testable and specific to THIS problem
2. Address the core uncertainties and assumptions in the problem statement
3. Create hypotheses that differentiate between alternative strategic paths
4. Ensure validation methods are practical and executable
5. Reference specific research insights if available
"""

        messages = [
            {
                "role": "system",
                "content": "You are a strategic consultant specializing in hypothesis generation and testing.",
            },
            {"role": "user", "content": prompt},
        ]

        response = await self.llm_client.call_best_available_provider(messages)

        return self._parse_structured_response(
            response, "hypothesis_testing", research_data is not None
        )

    async def execute_analysis(
        self,
        problem_statement: str,
        hypotheses: List[str] = None,
        research_data: Optional[Dict] = None,
        accumulated_context: Optional[Dict] = None,
        **kwargs,
    ) -> CognitiveAnalysisResult:
        """Execute multi-criteria analysis with dynamic framework selection"""

        # Determine analysis frameworks based on problem type
        analysis_frameworks = self._select_analysis_frameworks(problem_statement)
        criteria_guidance = self._get_analysis_criteria(problem_statement)

        research_section = ""
        research_integration_note = ""
        if research_data:
            research_section = f"""
EXTERNAL RESEARCH DATA:
{json.dumps(research_data, indent=2)}
"""
            research_integration_note = """
RESEARCH INTEGRATION REQUIREMENT:
Use specific research insights as evidence to support your analysis. Reference actual data points, market trends, and competitive intelligence in your assessments.
"""

        hypotheses_text = (
            "\n".join([f"- {h}" for h in hypotheses])
            if hypotheses
            else "- No specific hypotheses provided - generate analysis based on problem statement"
        )
        frameworks_text = ", ".join(analysis_frameworks)

        prompt = f"""
You are a McKinsey-grade strategic consultant executing comprehensive multi-criteria analysis using advanced analytical frameworks.

PROBLEM STATEMENT:
{problem_statement}

HYPOTHESES TO ANALYZE:
{hypotheses_text}

{research_section}

ANALYTICAL FRAMEWORKS TO APPLY:
{frameworks_text}

{criteria_guidance}

{research_integration_note}

TASK: Execute rigorous analysis using the specified frameworks and criteria. Respond in JSON format:

{{
  "mental_models_selected": ["{frameworks_text}"],
  "reasoning_description": "Detailed explanation of analytical approach, frameworks applied, and reasoning process specific to this problem context",
  "key_insights": ["specific analytical insight about this problem", "data-driven conclusion", "strategic implication"],
  "confidence_score": 0.85,
  "research_requirements": ["additional analysis needed"],
  "analysis_results": [
    {{
      "criterion": "SPECIFIC criterion relevant to this problem",
      "assessment": "DETAILED assessment with specific reasoning and evidence",
      "score": 0.8,
      "evidence": "SPECIFIC supporting evidence from research, analysis, or logical reasoning",
      "risk_factors": ["specific risks related to this criterion"],
      "confidence_level": "high/medium/low with justification"
    }}
  ],
  "comparative_analysis": {{
    "alternative_approaches": ["approach 1", "approach 2"],
    "trade_offs": "Key trade-offs between different options",
    "recommendation_logic": "Why one approach may be preferred over others"
  }}
}}

CRITICAL REQUIREMENTS:
1. Apply analysis frameworks specifically relevant to this problem type
2. Use concrete evidence and data to support assessments
3. Address the specific hypotheses or problem uncertainties
4. Provide actionable insights, not generic observations
5. Reference research data where available to strengthen analysis
"""

        messages = [
            {
                "role": "system",
                "content": "You are a strategic consultant specializing in multi-criteria analysis and risk assessment.",
            },
            {"role": "user", "content": prompt},
        ]

        response = await self.llm_client.call_best_available_provider(messages)

        return self._parse_structured_response(
            response, "multi_criteria_analysis", research_data is not None
        )

    async def synthesize_deliverable(
        self,
        comprehensive_context: Dict,
        research_data: Optional[Dict] = None,
        **kwargs,
    ) -> CognitiveAnalysisResult:
        """Synthesize final deliverable using pyramid principle with dynamic content generation"""

        # Extract problem statement from comprehensive context
        problem_statement = comprehensive_context.get("problem_statement", "")

        # Identify deliverable type and create dynamic content
        deliverable_config = self._configure_deliverable(problem_statement)

        research_section = ""
        research_attribution = ""
        if research_data:
            total_sources = research_data.get(
                "total_research_sources",
                research_data.get("external_sources_accessed", 0),
            )
            research_section = f"""
EXTERNAL RESEARCH DATA:
{json.dumps(research_data, indent=2)}

RESEARCH SOURCES: {total_sources} external sources accessed
"""
            research_attribution = (
                f"Based on analysis of {total_sources} external research sources"
            )

        # Extract analysis results from comprehensive context
        analysis_results = comprehensive_context.get("analysis_results", [])
        analysis_text = json.dumps(analysis_results, indent=2)

        prompt = f"""
You are a McKinsey-grade strategic consultant creating a final deliverable using the Pyramid Principle for this SPECIFIC strategic challenge.

PROBLEM STATEMENT:
{problem_statement}

ANALYSIS RESULTS:
{analysis_text}

{research_section}

DELIVERABLE FOCUS:
This deliverable addresses: {deliverable_config['focus']}

{deliverable_config['framework']}

TASK: Create a comprehensive, problem-specific deliverable in JSON format:

{{
  "type": "{deliverable_config['type']}",
  "title": "{deliverable_config['title']}",
  "executive_summary": "2-3 sentence summary addressing the SPECIFIC problem and key recommendations",
  "content": "Comprehensive analysis following Pyramid Principle - MUST address the specific problem context and reference actual problem elements",
  "key_findings": [
    "specific finding 1 about this exact problem",
    "specific finding 2 based on analysis",
    "specific finding 3 supported by research"
  ],
  "recommendations": [
    {{
      "recommendation": "SPECIFIC actionable recommendation for this exact problem",
      "rationale": "SPECIFIC reasoning based on this problem's analysis and research",
      "impact": "SPECIFIC expected impact for this situation",
      "timeline": "SPECIFIC implementation timeline for this context",
      "success_metrics": ["specific measurable outcomes"],
      "confidence": 0.85
    }}
  ],
  "implementation_roadmap": {{
    "phase_1": "Immediate actions (0-3 months)",
    "phase_2": "Medium-term execution (3-12 months)", 
    "phase_3": "Long-term optimization (12+ months)"
  }},
  "risk_assessment": {{
    "critical_risks": ["specific risks for this problem"],
    "mitigation_strategies": ["specific mitigation approaches"]
  }},
  "confidence_level": "high",
  "research_attribution": "{research_attribution}"
}}

CRITICAL REQUIREMENTS:
1. Every element MUST be specific to the actual problem statement
2. Reference specific details from the problem context
3. Avoid any generic or template language
4. Ensure recommendations are actionable and contextual
5. Integrate research insights where available
6. Use the Pyramid Principle: conclusion first, then supporting arguments
7. Address the specific stakeholders and success criteria mentioned in the problem
"""

        messages = [
            {
                "role": "system",
                "content": "You are a strategic consultant specializing in deliverable synthesis using the Pyramid Principle.",
            },
            {"role": "user", "content": prompt},
        ]

        response = await self.llm_client.call_best_available_provider(messages)

        return self._parse_structured_response(
            response, "pyramid_principle", research_data is not None
        )

    def _get_context_specific_guidance(
        self, industry: str, company_size: str, decision_type: str
    ) -> str:
        """Generate context-specific analysis guidance"""

        if "enterprise" in industry.lower() or "saas" in industry.lower():
            return """
For enterprise SaaS analysis, focus on:
- Customer acquisition and retention dynamics
- Technology architecture scalability
- Competitive positioning against established players
- Revenue model sustainability and growth patterns
"""
        elif "startup" in company_size.lower():
            return """
For startup analysis, emphasize:
- Market validation and product-market fit
- Resource constraints and prioritization
- Growth trajectory and scaling challenges
- Risk mitigation strategies
"""
        elif "pivot" in decision_type.lower():
            return """
For strategic pivot analysis, consider:
- Current state assessment and change drivers
- Transition risks and mitigation strategies
- Stakeholder impact and change management
- Success metrics and validation points
"""
        else:
            return """
For strategic business analysis, consider:
- Competitive positioning and market dynamics
- Financial implications and resource requirements
- Operational feasibility and execution risks
- Stakeholder alignment and change management
"""

    def _analyze_problem_focus(self, problem_statement: str) -> str:
        """Analyze problem characteristics to determine focus areas"""
        problem_keywords = problem_statement.lower()

        focus_areas = []
        if any(
            word in problem_keywords
            for word in [
                "ai",
                "tech",
                "digital",
                "platform",
                "software",
                "architecture",
            ]
        ):
            focus_areas.append("technical feasibility and implementation")
        if any(
            word in problem_keywords
            for word in ["market", "customer", "competition", "positioning", "segment"]
        ):
            focus_areas.append("market dynamics and competitive response")
        if any(
            word in problem_keywords
            for word in ["revenue", "cost", "roi", "investment", "funding", "pricing"]
        ):
            focus_areas.append("financial impact and business model viability")
        if any(
            word in problem_keywords
            for word in ["operation", "process", "efficiency", "scaling", "workflow"]
        ):
            focus_areas.append("operational execution and scaling")

        # Default to strategic if no specific focus detected
        if not focus_areas:
            focus_areas.append("strategic positioning and execution")

        return ", ".join(focus_areas)

    def _select_analysis_frameworks(self, problem_statement: str) -> List[str]:
        """Select analysis frameworks based on problem type"""
        problem_lower = problem_statement.lower()
        analysis_frameworks = ["multi_criteria_analysis"]

        # Financial analysis for investment/revenue decisions
        if any(
            word in problem_lower
            for word in ["investment", "roi", "revenue", "cost", "financial", "pricing"]
        ):
            analysis_frameworks.extend(
                ["financial_modeling", "investment_analysis", "risk_return_assessment"]
            )

        # Market analysis for competitive/positioning decisions
        if any(
            word in problem_lower
            for word in ["market", "competitor", "positioning", "customer", "segment"]
        ):
            analysis_frameworks.extend(
                [
                    "competitive_analysis",
                    "market_dynamics",
                    "customer_impact_assessment",
                ]
            )

        # Technology analysis for tech decisions
        if any(
            word in problem_lower
            for word in [
                "technology",
                "ai",
                "digital",
                "platform",
                "architecture",
                "technical",
            ]
        ):
            analysis_frameworks.extend(
                [
                    "technology_assessment",
                    "implementation_feasibility",
                    "scalability_analysis",
                ]
            )

        # Risk analysis for strategic pivots
        if any(
            word in problem_lower
            for word in ["pivot", "transformation", "change", "transition"]
        ):
            analysis_frameworks.extend(
                [
                    "change_management",
                    "transition_risk_assessment",
                    "stakeholder_impact",
                ]
            )

        return analysis_frameworks

    def _get_analysis_criteria(self, problem_statement: str) -> str:
        """Get analysis criteria based on problem context"""
        problem_lower = problem_statement.lower()

        if "enterprise" in problem_lower and "saas" in problem_lower:
            return """
ANALYSIS CRITERIA FOR ENTERPRISE SAAS:
- Customer Impact: Effect on existing customer base and retention
- Technical Feasibility: Implementation complexity and resource requirements  
- Competitive Advantage: Differentiation and market positioning impact
- Financial Viability: Revenue impact, cost implications, and ROI potential
- Operational Risk: Implementation risks and business continuity
- Strategic Alignment: Fit with long-term business strategy
"""
        elif "startup" in problem_lower:
            return """
ANALYSIS CRITERIA FOR STARTUP CONTEXT:
- Market Opportunity: Size and accessibility of target market
- Resource Efficiency: Optimal use of limited resources
- Growth Potential: Scalability and expansion opportunities
- Risk Management: Mitigation of existential risks
- Competitive Positioning: Differentiation in crowded market
- Execution Feasibility: Capability to deliver with current team
"""
        else:
            return """
ANALYSIS CRITERIA (GENERAL STRATEGIC):
- Strategic Impact: Alignment with core business objectives
- Implementation Feasibility: Practical execution considerations
- Risk Assessment: Potential downside and mitigation strategies
- Financial Impact: Cost-benefit analysis and resource requirements
- Stakeholder Effects: Impact on key stakeholders and relationships
- Competitive Implications: Effect on competitive position
"""

    def _configure_deliverable(self, problem_statement: str) -> Dict[str, str]:
        """Configure deliverable based on problem characteristics"""
        problem_lower = problem_statement.lower()

        if "pivot" in problem_lower:
            return {
                "type": "strategic_transformation_framework",
                "focus": "transformation strategy and implementation roadmap",
                "title": f"Strategic Pivot Framework: {self._extract_core_challenge(problem_statement)}",
                "framework": """
TRANSFORMATION FRAMEWORK:
- Current State Assessment and Change Drivers
- Strategic Pivot Roadmap with Phases
- Risk Mitigation and Change Management
- Success Metrics and Validation Points
""",
            }
        elif any(word in problem_lower for word in ["investment", "roi", "financial"]):
            return {
                "type": "investment_analysis",
                "focus": "financial analysis and investment recommendations",
                "title": f"Investment Analysis: {self._extract_core_challenge(problem_statement)}",
                "framework": """
INVESTMENT ANALYSIS FRAMEWORK:
- Financial Impact Assessment
- Risk-Return Analysis
- Implementation Cost Analysis
- ROI Projections and Scenarios
""",
            }
        elif "enterprise" in problem_lower and "saas" in problem_lower:
            return {
                "type": "enterprise_strategy",
                "focus": "enterprise SaaS strategic positioning",
                "title": f"Enterprise SaaS Strategy: {self._extract_core_challenge(problem_statement)}",
                "framework": """
ENTERPRISE STRATEGY FRAMEWORK:
- Market Position and Competitive Advantage
- Technology Implementation Strategy
- Customer Impact and Retention Plan
- Financial Impact and Investment Strategy
- Risk Management and Compliance
""",
            }
        else:
            return {
                "type": "strategic_recommendation",
                "focus": "strategic analysis and recommendations",
                "title": f"Strategic Analysis: {self._extract_core_challenge(problem_statement)}",
                "framework": """
STRATEGIC RECOMMENDATION FRAMEWORK:
- Strategic Options Analysis
- Implementation Roadmap
- Risk Assessment and Mitigation
- Success Metrics and Monitoring
""",
            }

    def _extract_core_challenge(self, problem_statement: str) -> str:
        """Extract core challenge from problem statement for title"""
        problem_lower = problem_statement.lower()

        if "ai-native architecture" in problem_lower:
            return "AI-Native Architecture Pivot"
        elif "launch" in problem_lower and "timing" in problem_lower:
            return "Launch Timing Strategy"
        elif "market positioning" in problem_lower:
            return "Market Positioning Strategy"
        else:
            # Extract first significant phrase as core challenge
            words = problem_statement.split()[:8]
            return " ".join(words).replace("\n", " ").strip()

    def _parse_structured_response(
        self, response, default_mental_model: str, research_enhanced: bool
    ) -> CognitiveAnalysisResult:
        """Parse structured JSON response with fallback handling"""
        try:
            # Clean the response content to handle common formatting issues
            content = response.content.strip()

            # Remove markdown formatting if present
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()

            # Try to extract JSON if there's text before/after
            if not content.startswith("{"):
                start_idx = content.find("{")
                if start_idx != -1:
                    content = content[start_idx:]

            if not content.endswith("}"):
                end_idx = content.rfind("}")
                if end_idx != -1:
                    content = content[: end_idx + 1]

            parsed = json.loads(content)

            return CognitiveAnalysisResult(
                mental_models_selected=parsed.get(
                    "mental_models_selected", [default_mental_model]
                ),
                reasoning_description=parsed.get(
                    "reasoning_description", f"Applied {default_mental_model}"
                ),
                key_insights=parsed.get("key_insights", []),
                confidence_score=parsed.get("confidence_score", 0.8),
                research_requirements=parsed.get("research_requirements", []),
                raw_response=response.content,
                tokens_used=response.tokens_used,
                cost_usd=response.cost_usd,
                response_time_ms=response.response_time_ms,
            )

        except json.JSONDecodeError as e:
            # Enhanced fallback with detailed logging
            self.logger.error(f"JSON parsing failed for cognitive analysis: {e}")
            self.logger.error(f"Response length: {len(response.content)} chars")
            self.logger.error(f"Response preview: {response.content[:200]}...")

            return CognitiveAnalysisResult(
                mental_models_selected=[default_mental_model],
                reasoning_description=f"Applied {default_mental_model} with fallback parsing"
                + (" (research enhanced)" if research_enhanced else ""),
                key_insights=["Analysis completed with fallback parsing"],
                confidence_score=0.7,
                research_requirements=[],
                raw_response=response.content,
                tokens_used=response.tokens_used,
                cost_usd=response.cost_usd,
                response_time_ms=response.response_time_ms,
            )
