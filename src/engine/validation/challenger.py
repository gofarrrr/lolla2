#!/usr/bin/env python3
"""
BaselineChallenger - Operation Crucible Implementation
Creates the strongest possible single-prompt competitor for METIS validation

The BaselineChallenger generates sophisticated mega-prompts that attempt to replicate
the entire METIS consulting workflow in a single LLM call. This serves as our "control group"
for measuring the architectural superiority of our multi-phase cognitive system.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import uuid

# V5.3 Manager Pattern - Use LLMManager instead of direct providers
from src.engine.core.llm_manager import get_llm_manager


@dataclass
class MegaPromptMetadata:
    """Metadata about the generated mega-prompt"""

    prompt_length: int
    estimated_tokens: int
    complexity_score: float
    techniques_applied: List[str]
    generation_timestamp: datetime
    prompt_version: str


class BaselineChallenger:
    """
    The BaselineChallenger creates sophisticated single-shot prompts designed to compete
    with METIS's multi-phase architecture. It represents the "strongest possible baseline"
    for validation benchmarking.

    Key Features:
    - DeepSeek V3 integration for state-of-the-art reasoning
    - Anthropic best practices implementation
    - Context and Motivation Framework
    - Advanced reasoning patterns (<thinking> blocks)
    - Comprehensive consulting methodology integration
    """

    def __init__(
        self,
        preferred_model: str = "deepseek",
        enable_reasoning: bool = True,
        context_stream=None,
    ):
        self.logger = logging.getLogger(__name__)
        self.preferred_model = preferred_model
        self.enable_reasoning = enable_reasoning

        # V5.3 Manager Pattern - Use LLMManager for resilient LLM access
        self.llm_manager = get_llm_manager(context_stream=context_stream)
        self.logger.info("🎯 BaselineChallenger initialized with V5.3 LLMManager")

        # Mega-prompt engineering configuration
        self.prompt_techniques = {
            "context_and_motivation": True,
            "thinking_blocks": enable_reasoning,
            "chain_of_thought": True,
            "structured_output": True,
            "mece_framework": True,
            "hypothesis_driven": True,
            "strategic_frameworks": True,
            "executive_synthesis": True,
        }

    def generate_mega_prompt(self, user_query: str, context: Dict[str, Any]) -> str:
        """
        Generate a sophisticated mega-prompt that attempts to replicate the entire
        METIS consulting workflow in a single, comprehensive LLM call.

        This is the core method that creates our "strongest possible baseline" competitor.
        """

        self.logger.info(
            "🔬 Generating mega-prompt using advanced prompt engineering techniques"
        )

        # Extract context information
        industry = context.get("industry", "General Business")
        company = context.get("company", "Target Organization")
        core_competency = context.get("core_competency", "Core business strengths")
        market_threat = context.get("market_threat", "Market challenges")

        # Build the sophisticated mega-prompt using Anthropic best practices
        mega_prompt = f"""# ADVANCED STRATEGIC CONSULTING ANALYSIS

## Context and Motivation Framework

You are an elite strategic consultant with deep expertise in management consulting methodologies, combining the analytical rigor of McKinsey, the strategic insight of BCG, and the implementation focus of Bain. Your task is to provide a comprehensive, board-ready strategic analysis that rivals the best consulting deliverables.

**Business Context:**
- Industry: {industry}
- Organization: {company}
- Core Competencies: {core_competency}
- Strategic Challenge: {market_threat}

**Your Mission:** Deliver a complete strategic analysis that demonstrates exceptional consulting quality, logical rigor, and actionable insights.

## Advanced Reasoning Instructions

<thinking>
Before proceeding with the analysis, I need to approach this systematically using proven consulting methodologies. Let me structure my thinking:

1. **Problem Definition & Scoping**: I'll start by clearly defining the core strategic challenge and its dimensions
2. **MECE Decomposition**: I'll break down the problem into Mutually Exclusive, Collectively Exhaustive components
3. **Hypothesis Development**: I'll generate data-driven hypotheses about potential solutions
4. **Framework Analysis**: I'll apply multiple strategic frameworks (Porter's 5 Forces, Value Chain, etc.)
5. **Synthesis & Recommendations**: I'll synthesize insights into clear, actionable recommendations

This systematic approach will ensure comprehensive coverage while maintaining analytical rigor.
</thinking>

## Strategic Analysis Framework

**PRIMARY QUESTION:** {user_query}

### PHASE 1: MECE PROBLEM DECONSTRUCTION

Using the MECE (Mutually Exclusive, Collectively Exhaustive) principle, systematically decompose this strategic challenge into its core components:

1. **Market Dynamics Analysis**
   - Industry evolution and disruption forces
   - Competitive landscape shifts
   - Customer behavior changes
   - Technology impact assessment

2. **Internal Capability Assessment** 
   - Core competency evaluation
   - Asset and resource analysis
   - Operational efficiency review
   - Cultural and organizational factors

3. **Strategic Response Options**
   - Growth and expansion opportunities  
   - Defensive positioning strategies
   - Transformation and innovation paths
   - Partnership and acquisition possibilities

**DELIVERABLE:** Create a comprehensive issue tree that captures all relevant problem dimensions.

### PHASE 2: HYPOTHESIS-DRIVEN ANALYSIS

Generate 3-5 strategic hypotheses that could address the core challenge:

For each hypothesis:
- **Hypothesis Statement**: "If [specific action], then [expected outcome] because [logical rationale]"
- **Priority Ranking**: Rate 1-10 based on impact potential and feasibility
- **Key Assumptions**: Critical assumptions that must hold true
- **Evidence Requirements**: What data/analysis would prove/disprove this hypothesis
- **Risk Assessment**: Major risks and mitigation strategies

**DELIVERABLE:** Ranked hypothesis set with supporting analysis.

### PHASE 3: MULTI-FRAMEWORK STRATEGIC ANALYSIS

Apply these strategic frameworks systematically:

**3.1 Porter's Five Forces Analysis:**
- Competitive rivalry intensity and key players
- Buyer power and negotiation dynamics
- Supplier power and dependency risks  
- Threat of new entrants and barriers
- Substitute products/services threats

**3.2 Value Chain Analysis:**
- Primary activities (operations, marketing, service)
- Support activities (technology, HR, infrastructure)
- Cost structure and margin analysis
- Differentiation opportunities identification

**3.3 SWOT Matrix Integration:**
- Strengths: Internal competitive advantages
- Weaknesses: Internal limitations and gaps
- Opportunities: External favorable conditions
- Threats: External risks and challenges

**3.4 Financial Impact Modeling:**
- Conservative case scenario (revenue, costs, ROI)
- Base case scenario with realistic assumptions
- Optimistic case with favorable conditions
- Sensitivity analysis on key variables

**DELIVERABLE:** Integrated framework analysis with quantitative impact assessment.

### PHASE 4: EXECUTIVE SYNTHESIS & RECOMMENDATIONS

Synthesize your analysis into a board-ready executive deliverable:

**4.1 Governing Thought (Single Sentence)**
- One clear, compelling statement that captures your core recommendation

**4.2 SCQA Structure:**
- **Situation**: Current state and market position
- **Complication**: Key challenges and threats  
- **Question**: Critical decision to be made
- **Answer**: Your recommended strategic direction

**4.3 Supporting Argument Pillars (3-4 Key Points):**
For each pillar:
- Clear action title
- Supporting narrative with evidence
- Expected business impact
- Implementation considerations

**4.4 Implementation Roadmap:**
- **Phase 1** (0-6 months): Immediate actions and quick wins
- **Phase 2** (6-18 months): Core transformation initiatives  
- **Phase 3** (18+ months): Long-term strategic positioning
- Success metrics and milestone tracking

## Quality Standards & Validation

Your analysis must meet these professional consulting standards:

- **Logical Coherence**: Arguments flow logically with clear cause-and-effect relationships
- **Evidence Grounding**: Claims supported by market data, industry benchmarks, or logical reasoning
- **MECE Compliance**: Problem breakdown is mutually exclusive and collectively exhaustive
- **Actionability**: Recommendations are specific, measurable, and implementable
- **Executive Readiness**: Structured for C-suite presentation with clear business impact
- **Intellectual Rigor**: Demonstrates deep analytical thinking and strategic insight

## Output Format Requirements

Structure your response as a comprehensive JSON object with these sections:

```json
{{
  "executiveSummary": {{
    "governingThought": "Single sentence strategic recommendation",
    "keyRationale": "Core business logic supporting the recommendation",
    "expectedImpact": "Quantified business outcomes",
    "investmentRequired": "Resource requirements and timeline",
    "riskProfile": "Risk assessment and mitigation approach"
  }},
  "problemDeconstruction": {{
    "problemClassification": "Type and nature of strategic challenge",
    "issueTree": {{
      "root": "Primary problem statement", 
      "branches": [
        {{
          "branch": "Major problem category",
          "subIssues": ["Specific sub-problem 1", "Sub-problem 2", "Sub-problem 3"]
        }}
      ]
    }},
    "meceComplianceScore": "0.0-1.0 assessment of MECE quality"
  }},
  "strategicHypotheses": [
    {{
      "hypothesisId": "H1",
      "statement": "If-then-because hypothesis formulation", 
      "priorityScore": "1-10 ranking",
      "killerAnalysis": "Key analysis that would validate/invalidate",
      "evidenceRequired": ["Evidence type 1", "Evidence type 2"],
      "riskFactors": ["Risk 1", "Risk 2"]
    }}
  ],
  "frameworkAnalysis": {{
    "portersAnalysis": {{
      "competitiveRivalry": "Assessment and key insights",
      "buyerPower": "Power dynamics and implications",
      "supplierPower": "Dependency analysis", 
      "entryBarriers": "Barriers to entry assessment",
      "substituteThreats": "Substitution risk evaluation"
    }},
    "swotMatrix": {{
      "strengths": ["Internal strength 1", "Strength 2"],
      "weaknesses": ["Internal weakness 1", "Weakness 2"],
      "opportunities": ["External opportunity 1", "Opportunity 2"], 
      "threats": ["External threat 1", "Threat 2"]
    }},
    "financialProjections": {{
      "conservativeCase": {{"revenue": "$XM", "roi": "X.Xx", "payback": "XX months"}},
      "baseCase": {{"revenue": "$XM", "roi": "X.Xx", "payback": "XX months"}},
      "optimisticCase": {{"revenue": "$XM", "roi": "X.Xx", "payback": "XX months"}}
    }}
  }},
  "finalRecommendations": {{
    "scqaStructure": {{
      "situation": "Current state description",
      "complication": "Key challenges and tensions",
      "question": "Critical decision point",
      "answer": "Strategic recommendation"
    }},
    "supportingPillars": [
      {{
        "actionTitle": "Specific initiative name",
        "narrative": "Detailed explanation and rationale",
        "evidence": ["Supporting fact 1", "Supporting fact 2"],
        "businessImpact": "Expected outcomes and metrics"
      }}
    ],
    "implementationRoadmap": {{
      "phase1": {{
        "timeline": "Time period",
        "actions": ["Action 1", "Action 2"],
        "successMetrics": ["Metric 1", "Metric 2"],
        "resourceRequirements": "Required resources"
      }},
      "phase2": {{
        "timeline": "Time period", 
        "actions": ["Action 1", "Action 2"],
        "successMetrics": ["Metric 1", "Metric 2"],
        "resourceRequirements": "Required resources"
      }},
      "phase3": {{
        "timeline": "Time period",
        "actions": ["Action 1", "Action 2"], 
        "successMetrics": ["Metric 1", "Metric 2"],
        "resourceRequirements": "Required resources"
      }}
    }}
  }},
  "qualityMetrics": {{
    "methodologyCompliance": "Assessment of consulting methodology adherence",
    "evidenceGrounding": "Quality of evidence and data integration",
    "actionability": "Practicality and implementability of recommendations",
    "boardReadiness": "Suitability for executive presentation"
  }}
}}
```

## Final Instructions

1. **Think Before You Act**: Use <thinking> blocks to reason through complex analytical steps
2. **Be Comprehensive**: Address all four phases with equal rigor and depth
3. **Stay Grounded**: Base recommendations on sound business logic and market realities  
4. **Quantify Impact**: Provide specific, measurable business outcomes where possible
5. **Maintain Quality**: Ensure every section meets professional consulting standards

Execute this analysis with the precision and insight of a top-tier strategy consultant. Your goal is to produce an analysis so comprehensive and insightful that it rivals or exceeds the output of sophisticated multi-agent systems.

BEGIN ANALYSIS:"""

        # Calculate metadata
        metadata = MegaPromptMetadata(
            prompt_length=len(mega_prompt),
            estimated_tokens=len(mega_prompt.split()) * 1.3,  # Rough token estimation
            complexity_score=self._calculate_complexity_score(mega_prompt),
            techniques_applied=list(self.prompt_techniques.keys()),
            generation_timestamp=datetime.utcnow(),
            prompt_version="1.0.0",
        )

        self.logger.info(
            f"✅ Generated mega-prompt: {metadata.prompt_length} chars, "
            f"~{metadata.estimated_tokens:.0f} tokens, "
            f"complexity: {metadata.complexity_score:.2f}"
        )

        return mega_prompt

    def _calculate_complexity_score(self, prompt: str) -> float:
        """Calculate a complexity score for the generated prompt"""

        # Base complexity factors
        length_score = min(1.0, len(prompt) / 10000)  # Normalize to ~10k chars

        # Count sophisticated techniques
        technique_indicators = [
            "<thinking>",
            "MECE",
            "hypothesis",
            "framework",
            "SCQA",
            "Porter",
            "Value Chain",
            "SWOT",
            "JSON",
            "quantified",
        ]

        technique_score = sum(
            1
            for indicator in technique_indicators
            if indicator.lower() in prompt.lower()
        ) / len(technique_indicators)

        # Count structural elements
        structure_indicators = ["##", "###", "**", "```", "PHASE", "DELIVERABLE"]
        structure_score = (
            sum(prompt.count(indicator) for indicator in structure_indicators) / 50
        )

        # Overall complexity (weighted average)
        complexity = length_score * 0.3 + technique_score * 0.5 + structure_score * 0.2

        return min(1.0, complexity)

    async def execute_mega_prompt(
        self, mega_prompt: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute the mega-prompt against the LLM and return the structured result.
        This method handles the actual LLM interaction for validation runs.
        """

        if not self.llm_manager:
            raise RuntimeError("No LLM manager available for mega-prompt execution")

        self.logger.info("🚀 Executing mega-prompt against LLM")
        start_time = datetime.utcnow()

        try:
            # V5.3 Manager Pattern - Use LLMManager for execution
            response = await self.llm_manager.execute_completion(
                prompt=mega_prompt,
                system_prompt="",
                temperature=0.1,  # Low temperature for consistency
                max_tokens=4000,
                timeout=120,  # 2 minute timeout for complex analysis
            )

            execution_time = (datetime.utcnow() - start_time).total_seconds()

            # Extract content from LLMManager response
            content = response.raw_text

            # Try to extract JSON from the response
            parsed_result = self._parse_llm_response(content)

            # Package the complete result
            result = {
                "executionId": str(uuid.uuid4()),
                "timestamp": start_time.isoformat(),
                "executionTimeSeconds": execution_time,
                "model": response.model_name,
                "megaPrompt": mega_prompt,
                "rawResponse": content,
                "parsedResult": parsed_result,
                "metadata": {
                    "promptLength": len(mega_prompt),
                    "responseLength": len(content),
                    "parsingSuccess": parsed_result is not None,
                    "reasoningEnabled": self.enable_reasoning,
                },
            }

            self.logger.info(
                f"✅ Mega-prompt execution completed in {execution_time:.2f}s"
            )
            return result

        except Exception as e:
            self.logger.error(f"❌ Mega-prompt execution failed: {str(e)}")
            raise

    def _parse_llm_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse the LLM response to extract the structured JSON output.
        Handles various response formats and extraction scenarios.
        """

        try:
            # First, try to find JSON blocks in markdown
            import re

            json_pattern = r"```json\s*(\{.*?\})\s*```"
            json_matches = re.findall(json_pattern, response, re.DOTALL)

            if json_matches:
                # Use the first (and hopefully only) JSON block
                json_str = json_matches[0]
                return json.loads(json_str)

            # Second, try to find JSON without markdown
            # Look for the first { and last } that could contain a complete JSON object
            start_idx = response.find("{")
            end_idx = response.rfind("}")

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx : end_idx + 1]
                return json.loads(json_str)

            # Third, try parsing the entire response as JSON
            return json.loads(response)

        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse JSON from response: {e}")
            return None
        except Exception as e:
            self.logger.warning(f"Unexpected error parsing response: {e}")
            return None

    def get_prompt_metadata(self) -> Dict[str, Any]:
        """Get metadata about the challenger's capabilities and configuration"""

        return {
            "challengerVersion": "1.0.0",
            "preferredModel": self.preferred_model,
            "reasoningEnabled": self.enable_reasoning,
            "llmManagerAvailable": self.llm_manager is not None,
            "activeProviders": (
                self.llm_manager.get_performance_stats()["providers"]
                if self.llm_manager
                else None
            ),
            "promptTechniques": self.prompt_techniques,
            "designPrinciples": [
                "Anthropic best practices integration",
                "Context and Motivation Framework",
                "Advanced reasoning with <thinking> blocks",
                "Comprehensive consulting methodology",
                "MECE problem decomposition",
                "Multi-framework strategic analysis",
                "Structured JSON output format",
                "Executive-grade synthesis",
            ],
            "competitiveFeatures": [
                "Single-shot comprehensive analysis",
                "Integrated strategic frameworks",
                "Quantified business impact",
                "Board-ready deliverable format",
                "Advanced prompt engineering",
                "DeepSeek V3 reasoning capabilities",
            ],
        }


# Global challenger instance for easy access
_global_challenger: Optional[BaselineChallenger] = None


def get_baseline_challenger(
    preferred_model: str = "deepseek", enable_reasoning: bool = True
) -> BaselineChallenger:
    """Get or create the global BaselineChallenger instance"""
    global _global_challenger

    if _global_challenger is None:
        _global_challenger = BaselineChallenger(preferred_model, enable_reasoning)

    return _global_challenger


# Convenience functions
async def generate_baseline_mega_prompt(
    user_query: str, context: Dict[str, Any]
) -> str:
    """Generate a mega-prompt for baseline validation"""
    challenger = get_baseline_challenger()
    return challenger.generate_mega_prompt(user_query, context)


async def execute_baseline_challenge(
    user_query: str, context: Dict[str, Any]
) -> Dict[str, Any]:
    """Execute a complete baseline challenge run"""
    challenger = get_baseline_challenger()
    mega_prompt = challenger.generate_mega_prompt(user_query, context)
    return await challenger.execute_mega_prompt(mega_prompt, context)
