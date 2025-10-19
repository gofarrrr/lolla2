"""
Critical Thinking Cognitive Model
Focused implementation of critical thinking analysis for systematic evaluation
"""

import time
import re
from typing import List, Optional
from .base_cognitive_model import (
    BaseCognitiveModel,
    CognitiveModelType,
    ModelApplicationContext,
    ModelApplicationResult,
)


class CriticalThinkingModel(BaseCognitiveModel):
    """
    Implements critical thinking cognitive model for systematic evaluation
    Focuses on assumption analysis, evidence evaluation, and logical reasoning
    """

    def __init__(self, llm_orchestrator: Optional["LLMOrchestrator"] = None):
        super().__init__("critical_thinking", llm_orchestrator)
        self.model_specific_config = {
            "focus_areas": [
                "assumption_identification",
                "evidence_evaluation",
                "logical_reasoning",
                "bias_detection",
                "alternative_hypotheses",
                "argument_validation",
            ],
            "quality_indicators": [
                "assumption_clarity",
                "evidence_rigor",
                "logical_coherence",
                "bias_awareness",
                "hypothesis_completeness",
            ],
        }

    def _get_model_type(self) -> CognitiveModelType:
        return CognitiveModelType.CRITICAL_THINKING

    async def apply_model(
        self, context: ModelApplicationContext
    ) -> ModelApplicationResult:
        """Apply critical thinking analysis"""
        start_time = time.time()

        try:
            # Build critical thinking specific prompt
            prompt = self._build_prompt(context)

            # Get LLM response with critical thinking optimization
            if self.llm_orchestrator:
                llm_response = await self.llm_orchestrator.generate_response(
                    prompt=prompt,
                    model_type="critical_thinking",
                    temperature=0.2,  # Very low temperature for rigorous analysis
                    max_tokens=1200,
                    require_high_quality=True,
                )
                response_text = llm_response.content
                base_confidence = llm_response.confidence_score
            else:
                # Fallback to template-based analysis
                response_text = self._generate_template_analysis(context)
                base_confidence = 0.6

            # Parse and structure the response
            result = self._parse_critical_thinking_response(
                response_text, context, base_confidence
            )
            result.processing_time_ms = (time.time() - start_time) * 1000

            self.logger.info(
                f"✅ Critical thinking analysis completed: confidence={result.confidence_score:.3f}"
            )
            return result

        except Exception as e:
            self.logger.error(f"❌ Critical thinking analysis failed: {e}")
            # Return fallback result
            return self._create_fallback_analysis(context, str(e))

    def _build_prompt(self, context: ModelApplicationContext) -> str:
        """Build critical thinking specific prompt"""

        # Extract context for analysis depth
        problem_complexity = self._assess_problem_complexity(context.problem_statement)
        business_domain = context.business_context.get("industry", "general business")

        prompt = f"""
You are applying CRITICAL THINKING methodology to systematically evaluate this problem:

PROBLEM STATEMENT:
{context.problem_statement}

BUSINESS CONTEXT:
{context.business_context}

CRITICAL THINKING FRAMEWORK - Apply systematically:

1. ASSUMPTION IDENTIFICATION
   - Identify explicit and implicit assumptions in the problem statement
   - Evaluate which assumptions are well-founded vs questionable
   - Assess the impact if key assumptions prove incorrect

2. EVIDENCE EVALUATION  
   - What evidence is available to support different positions?
   - What is the quality and credibility of available evidence?
   - What evidence is missing but critical for decision-making?

3. LOGICAL REASONING ANALYSIS
   - Evaluate the logical structure of any arguments presented
   - Identify logical fallacies or reasoning gaps
   - Assess cause-and-effect relationships

4. BIAS DETECTION
   - What cognitive biases might affect stakeholder perspectives?
   - How might confirmation bias or anchoring affect decision-making?
   - What perspectives or viewpoints might be overlooked?

5. ALTERNATIVE HYPOTHESES
   - What alternative explanations exist for the situation?
   - What different approaches could address the core problem?
   - How would we test between competing hypotheses?

6. ARGUMENT VALIDATION
   - What are the strongest arguments for different courses of action?
   - Where are the weakest points in each argument?
   - What questions remain unanswered?

REQUIREMENTS:
- Be specific to this exact problem context
- Identify concrete assumptions and evidence gaps
- Provide actionable critical analysis
- Focus on {business_domain} domain considerations
- Address the {problem_complexity} complexity level appropriately

Deliver a systematic critical evaluation that strengthens decision-making rigor.
"""

        return prompt

    def _parse_critical_thinking_response(
        self,
        response_text: str,
        context: ModelApplicationContext,
        base_confidence: float,
    ) -> ModelApplicationResult:
        """Parse and structure critical thinking response"""

        # Extract key insights from structured sections
        insights = []
        evidence = []
        assumptions = []

        # Look for assumption analysis
        assumption_matches = re.findall(
            r"assumption[s]?[:\-\s]*([^\.]+\.)", response_text, re.IGNORECASE
        )
        assumptions.extend([match.strip() for match in assumption_matches[:5]])

        # Look for evidence evaluation
        evidence_matches = re.findall(
            r"evidence[:\-\s]*([^\.]+\.)", response_text, re.IGNORECASE
        )
        evidence.extend([match.strip() for match in evidence_matches[:3]])

        # Look for bias detection
        bias_matches = re.findall(
            r"bias[es]*[:\-\s]*([^\.]+\.)", response_text, re.IGNORECASE
        )
        if bias_matches:
            insights.append(f"Bias Analysis: {bias_matches[0].strip()}")

        # Look for logical reasoning insights
        logic_matches = re.findall(
            r"logical?[:\-\s]*([^\.]+\.)", response_text, re.IGNORECASE
        )
        if logic_matches:
            insights.append(f"Logical Analysis: {logic_matches[0].strip()}")

        # Look for alternative hypotheses
        alternative_matches = re.findall(
            r"alternative[s]?[:\-\s]*([^\.]+\.)", response_text, re.IGNORECASE
        )
        if alternative_matches:
            insights.append(
                f"Alternative Perspectives: {alternative_matches[0].strip()}"
            )

        # Fallback insight extraction
        if not insights:
            sentences = response_text.split(".")
            insights = [
                sent.strip() + "." for sent in sentences if len(sent.strip()) > 20
            ][:3]

        # Calculate critical thinking specific confidence
        ct_confidence = self._calculate_critical_thinking_confidence(
            response_text, assumptions, evidence, insights
        )

        # Combine with base confidence
        final_confidence = (base_confidence + ct_confidence) / 2

        return ModelApplicationResult(
            reasoning_text=response_text,
            confidence_score=final_confidence,
            key_insights=insights[:5],
            supporting_evidence=evidence[:3],
            assumptions_made=assumptions[:5],
            quality_metrics={
                "critical_thinking_rigor": ct_confidence,
                "assumption_count": len(assumptions),
                "evidence_quality": len(evidence) / 5.0,  # Normalize to 0-1
                "bias_awareness": (
                    1.0 if any("bias" in text.lower() for text in insights) else 0.5
                ),
                "logical_structure": self._assess_logical_structure(response_text),
            },
            processing_time_ms=0.0,
            model_id="critical_thinking",
        )

    def _calculate_critical_thinking_confidence(
        self,
        response_text: str,
        assumptions: List[str],
        evidence: List[str],
        insights: List[str],
    ) -> float:
        """Calculate confidence based on critical thinking quality indicators"""

        confidence = 0.5  # Base confidence

        # Boost for systematic analysis
        if len(assumptions) >= 2:
            confidence += 0.1
        if len(evidence) >= 1:
            confidence += 0.1
        if len(insights) >= 3:
            confidence += 0.1

        # Boost for critical thinking keywords
        critical_keywords = [
            "assumption",
            "evidence",
            "bias",
            "logical",
            "reasoning",
            "alternative",
            "hypothesis",
            "evaluate",
            "assess",
            "rigor",
        ]

        keyword_density = sum(
            1 for keyword in critical_keywords if keyword in response_text.lower()
        )
        confidence += min(0.2, keyword_density * 0.03)

        # Boost for structured analysis
        if "1." in response_text and "2." in response_text:
            confidence += 0.05

        return min(0.95, confidence)

    def _assess_logical_structure(self, response_text: str) -> float:
        """Assess the logical structure quality of the response"""

        structure_score = 0.0

        # Check for logical flow indicators
        if any(
            word in response_text.lower()
            for word in ["therefore", "because", "since", "thus", "consequently"]
        ):
            structure_score += 0.3

        # Check for structured reasoning
        if any(
            phrase in response_text.lower()
            for phrase in ["first", "second", "additionally", "furthermore", "however"]
        ):
            structure_score += 0.2

        # Check for conclusion indicators
        if any(
            phrase in response_text.lower()
            for phrase in ["in conclusion", "overall", "in summary", "this suggests"]
        ):
            structure_score += 0.3

        # Check for balanced analysis
        if any(
            word in response_text.lower()
            for word in ["both", "however", "although", "while"]
        ):
            structure_score += 0.2

        return min(1.0, structure_score)

    def _assess_problem_complexity(self, problem_statement: str) -> str:
        """Assess problem complexity for prompt customization"""

        complexity_indicators = {
            "high": [
                "strategic",
                "enterprise",
                "transformation",
                "multiple stakeholders",
                "long-term",
            ],
            "medium": [
                "operational",
                "departmental",
                "process",
                "workflow",
                "improvement",
            ],
            "low": ["tactical", "specific", "short-term", "single", "simple"],
        }

        problem_lower = problem_statement.lower()

        for level, indicators in complexity_indicators.items():
            if sum(1 for indicator in indicators if indicator in problem_lower) >= 2:
                return level

        return "medium"  # Default

    def _generate_template_analysis(self, context: ModelApplicationContext) -> str:
        """Generate template-based analysis when LLM unavailable"""

        return f"""
CRITICAL THINKING ANALYSIS

Problem: {context.problem_statement[:100]}...

KEY ASSUMPTIONS IDENTIFIED:
1. Current approach is suboptimal
2. Alternative solutions exist
3. Stakeholders are rational actors
4. Resources are available for change

EVIDENCE EVALUATION:
- Limited evidence provided in problem statement
- Need for additional data gathering
- Require stakeholder perspective validation

LOGICAL REASONING:
- Problem structure suggests multiple variables
- Cause-effect relationships need clarification
- Decision criteria require explicit definition

BIAS DETECTION:
- Potential confirmation bias in problem framing
- Status quo bias may affect solution evaluation
- Need for diverse perspective inclusion

ALTERNATIVE HYPOTHESES:
1. Problem may be symptom of deeper issue
2. Current metrics may be misleading
3. Solution may require systemic change

CRITICAL GAPS:
- Missing quantitative evidence
- Unclear success criteria
- Stakeholder impact assessment needed
"""

    def _create_fallback_analysis(
        self, context: ModelApplicationContext, error_message: str
    ) -> ModelApplicationResult:
        """Create fallback analysis for critical thinking"""

        fallback_insights = [
            "Critical thinking analysis requires systematic assumption evaluation",
            "Evidence quality assessment needed for reliable conclusions",
            "Bias detection essential for objective decision-making",
        ]

        return ModelApplicationResult(
            reasoning_text=f"Critical Thinking Analysis: {context.problem_statement[:200]}... [Analysis limited due to technical constraints]",
            confidence_score=0.4,
            key_insights=fallback_insights,
            supporting_evidence=[],
            assumptions_made=[f"Analysis constrained by: {error_message}"],
            quality_metrics={
                "fallback_result": True,
                "critical_thinking_rigor": 0.3,
                "assumption_count": 0,
                "evidence_quality": 0.0,
                "bias_awareness": 0.5,
            },
            processing_time_ms=0.0,
            model_id="critical_thinking",
        )

    def _validate_output_quality(self, result: ModelApplicationResult) -> bool:
        """Validate critical thinking output quality"""

        # Check minimum requirements
        min_requirements = [
            len(result.reasoning_text) > 100,
            len(result.key_insights) >= 2,
            result.confidence_score > 0.3,
            result.quality_metrics.get("critical_thinking_rigor", 0) > 0.4,
        ]

        # Check for critical thinking indicators
        reasoning_lower = result.reasoning_text.lower()
        critical_indicators = [
            "assumption" in reasoning_lower,
            "evidence" in reasoning_lower or "bias" in reasoning_lower,
            "logical" in reasoning_lower or "reasoning" in reasoning_lower,
            len(result.assumptions_made) > 0 or "alternative" in reasoning_lower,
        ]

        # Quality validation
        basic_quality = sum(min_requirements) >= 3
        critical_thinking_quality = sum(critical_indicators) >= 2

        return basic_quality and critical_thinking_quality
