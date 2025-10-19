#!/usr/bin/env python3
"""
BlindJudge - Operation Crucible Implementation
Provides objective, unbiased evaluation of competing analytical outputs

The BlindJudge implements sophisticated blinding methodology and multi-criteria evaluation
to deliver quantitative scorecards comparing METIS multi-phase architecture against
baseline single-prompt approaches. This ensures intellectual honesty in our validation process.
"""

import json
import logging
import asyncio
import random
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import hashlib

# V5.3 Manager Pattern - Use LLMManager instead of direct providers
from src.engine.core.llm_manager import get_llm_manager


@dataclass
class EvaluationCriterion:
    """Definition of a single evaluation criterion"""

    name: str
    weight: float
    description: str
    scale_definition: Dict[int, str]
    max_score: int = 10


@dataclass
class BlindedOutput:
    """A blinded output ready for evaluation"""

    label: str  # "Analysis A" or "Analysis B"
    content: Dict[str, Any]
    content_hash: str
    original_source: str  # Hidden from judge


@dataclass
class EvaluationResult:
    """Complete evaluation result with detailed breakdown"""

    evaluation_id: str
    timestamp: datetime
    judge_model: str
    execution_time: float

    # Blinding information (for audit trail)
    analysis_a_source: str
    analysis_b_source: str

    # Scores
    analysis_a_scores: Dict[str, float]
    analysis_b_scores: Dict[str, float]

    # Overall results
    analysis_a_total: float
    analysis_b_total: float
    winner: str
    margin: float

    # Detailed analysis
    criterion_breakdown: Dict[str, Dict[str, Any]]
    qualitative_analysis: str
    judge_confidence: float


class BlindJudge:
    """
    The BlindJudge provides objective, unbiased evaluation of competing analytical outputs.

    Key Features:
    - DeepSeek V3 integration for sophisticated reasoning
    - Rigorous blinding methodology to eliminate bias
    - Multi-criteria evaluation with weighted scoring
    - Structured JSON response parsing
    - Comprehensive audit trail and transparency
    - Statistical confidence measurement
    """

    def __init__(
        self,
        judge_model: str = "deepseek",
        enable_reasoning: bool = True,
        context_stream=None,
    ):
        self.logger = logging.getLogger(__name__)
        self.judge_model = judge_model
        self.enable_reasoning = enable_reasoning

        # V5.3 Manager Pattern - Use LLMManager for resilient LLM access
        self.llm_manager = get_llm_manager(context_stream=context_stream)

        # Define evaluation criteria based on audit findings
        self.evaluation_criteria = {
            "logical_structure": EvaluationCriterion(
                name="Logical Structure",
                weight=0.20,
                description="MECE compliance, argument flow, analytical coherence",
                scale_definition={
                    10: "Perfect MECE structure, flawless logical flow, crystal clear argumentation",
                    8: "Strong logical structure with minor gaps, good MECE compliance",
                    6: "Adequate structure but some logical inconsistencies or gaps",
                    4: "Weak structure, significant logical flaws, poor organization",
                    2: "Severely flawed logic, incoherent structure, major gaps",
                },
            ),
            "evidence_grounding": EvaluationCriterion(
                name="Evidence Grounding",
                weight=0.25,  # Highest weight per audit findings
                description="Research integration, factual accuracy, external validation",
                scale_definition={
                    10: "Exceptional evidence integration, thorough research backing, high factual accuracy",
                    8: "Strong evidence base with good research integration, mostly accurate",
                    6: "Adequate evidence but some claims lack proper backing",
                    4: "Weak evidence base, limited research integration, questionable claims",
                    2: "Poor evidence, minimal research, factual errors or unsupported assertions",
                },
            ),
            "depth_of_analysis": EvaluationCriterion(
                name="Depth of Analysis",
                weight=0.18,
                description="Framework sophistication, insight quality, multi-dimensional thinking",
                scale_definition={
                    10: "Exceptional analytical depth, sophisticated frameworks, breakthrough insights",
                    8: "Strong analytical depth with good framework application",
                    6: "Adequate analysis but lacks sophistication or deep insights",
                    4: "Shallow analysis, limited framework use, superficial insights",
                    2: "Very shallow, minimal analytical thinking, obvious conclusions only",
                },
            ),
            "actionability": EvaluationCriterion(
                name="Actionability",
                weight=0.20,
                description="Implementation clarity, executive readiness, practical recommendations",
                scale_definition={
                    10: "Crystal clear implementation path, executive-ready, highly practical",
                    8: "Clear recommendations with good implementation guidance",
                    6: "Somewhat actionable but lacks specificity or implementation details",
                    4: "Vague recommendations, unclear implementation path",
                    2: "Not actionable, theoretical only, no practical guidance",
                },
            ),
            "consulting_quality": EvaluationCriterion(
                name="Consulting Quality",
                weight=0.12,
                description="Professional standards, client presentation readiness",
                scale_definition={
                    10: "Exceptional consulting quality, top-tier firm standard, board-ready",
                    8: "High consulting quality with professional presentation",
                    6: "Good quality but not quite top-tier consulting standard",
                    4: "Below professional consulting standards, needs significant improvement",
                    2: "Poor quality, not suitable for professional presentation",
                },
            ),
            "factual_accuracy": EvaluationCriterion(
                name="Factual Accuracy",
                weight=0.05,
                description="Truth verification, hallucination detection",
                scale_definition={
                    10: "Completely accurate, no factual errors or hallucinations detected",
                    8: "Mostly accurate with minor factual issues",
                    6: "Generally accurate but some questionable claims",
                    4: "Several factual errors or unsupported claims",
                    2: "Significant factual errors, clear hallucinations or false information",
                },
            ),
        }

        # Validate criteria weights sum to 1.0
        total_weight = sum(
            criterion.weight for criterion in self.evaluation_criteria.values()
        )
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(
                f"Evaluation criteria weights must sum to 1.0, got {total_weight}"
            )

        self.logger.info(
            f"🏛️ BlindJudge initialized with V5.3 LLMManager and {len(self.evaluation_criteria)} criteria"
        )

    def evaluate_outputs(
        self,
        metis_output: Dict[str, Any],
        baseline_output: Dict[str, Any],
        original_query: str = "",
        context: Dict[str, Any] = None,
    ) -> EvaluationResult:
        """
        Main evaluation method - blinds the outputs and executes comprehensive evaluation.

        This is the core method that implements our objective validation methodology.
        """

        evaluation_id = f"blind-judge-{uuid.uuid4().hex[:8]}"
        start_time = datetime.utcnow()

        self.logger.info(f"🔍 Starting blind evaluation {evaluation_id}")

        # Step 1: Apply blinding methodology
        blinded_outputs = self._apply_blinding(metis_output, baseline_output)

        self.logger.info(
            f"🎭 Blinding applied: METIS → {blinded_outputs['metis'].label}, "
            f"Baseline → {blinded_outputs['baseline'].label}"
        )

        # Step 2: Generate comprehensive evaluation prompt
        evaluation_prompt = self._generate_evaluation_prompt(
            blinded_outputs, original_query, context
        )

        # Step 3: Execute evaluation
        judge_response = asyncio.run(self._execute_evaluation(evaluation_prompt))

        # Step 4: Parse evaluation results
        evaluation_data = self._parse_evaluation_response(judge_response)

        # Step 5: Map blinded results back to original sources
        evaluation_result = self._compile_evaluation_result(
            evaluation_id, start_time, blinded_outputs, evaluation_data, judge_response
        )

        execution_time = (datetime.utcnow() - start_time).total_seconds()
        evaluation_result.execution_time = execution_time

        self.logger.info(
            f"✅ Blind evaluation completed: Winner = {evaluation_result.winner}, "
            f"Margin = {evaluation_result.margin:.2f}, Time = {execution_time:.2f}s"
        )

        return evaluation_result

    def _apply_blinding(
        self, metis_output: Dict[str, Any], baseline_output: Dict[str, Any]
    ) -> Dict[str, BlindedOutput]:
        """
        Apply rigorous blinding methodology to eliminate evaluation bias.
        Randomly assigns outputs to "Analysis A" and "Analysis B".
        """

        # Generate content hashes for audit trail
        metis_hash = hashlib.sha256(
            json.dumps(metis_output, sort_keys=True).encode()
        ).hexdigest()[:16]
        baseline_hash = hashlib.sha256(
            json.dumps(baseline_output, sort_keys=True).encode()
        ).hexdigest()[:16]

        # Random blinding assignment
        if random.random() < 0.5:
            # METIS → A, Baseline → B
            analysis_a = BlindedOutput("Analysis A", metis_output, metis_hash, "metis")
            analysis_b = BlindedOutput(
                "Analysis B", baseline_output, baseline_hash, "baseline"
            )
        else:
            # Baseline → A, METIS → B
            analysis_a = BlindedOutput(
                "Analysis A", baseline_output, baseline_hash, "baseline"
            )
            analysis_b = BlindedOutput("Analysis B", metis_output, metis_hash, "metis")

        return {
            "metis": (
                analysis_a if analysis_a.original_source == "metis" else analysis_b
            ),
            "baseline": (
                analysis_a if analysis_a.original_source == "baseline" else analysis_b
            ),
            "analysis_a": analysis_a,
            "analysis_b": analysis_b,
        }

    def _generate_evaluation_prompt(
        self,
        blinded_outputs: Dict[str, BlindedOutput],
        original_query: str = "",
        context: Dict[str, Any] = None,
    ) -> str:
        """
        Generate a sophisticated evaluation prompt using Anthropic best practices.
        This prompt instructs the judge to evaluate blindly with detailed criteria.
        """

        analysis_a = blinded_outputs["analysis_a"]
        analysis_b = blinded_outputs["analysis_b"]

        # Extract key content for evaluation (handle various content structures)
        a_content = self._extract_evaluable_content(analysis_a.content)
        b_content = self._extract_evaluable_content(analysis_b.content)

        context_info = context or {}

        evaluation_prompt = f"""# BLIND STRATEGIC ANALYSIS EVALUATION

## Mission & Context

You are a senior partner at a top-tier management consulting firm (McKinsey/BCG/Bain level) conducting a blind evaluation of two strategic analyses. Your task is to provide an objective, quantitative assessment of analytical quality without knowing the source or methodology of either analysis.

**Original Business Question:** {original_query}

**Evaluation Context:**
- Industry: {context_info.get('industry', 'General Business')}
- Organization: {context_info.get('company', 'Target Organization')}
- Strategic Challenge: {context_info.get('market_threat', 'Market challenges')}

## Evaluation Methodology

You will evaluate both analyses against six critical dimensions using a rigorous 10-point scale. Your evaluation must be completely objective, focusing solely on analytical quality, logical rigor, and professional consulting standards.

### Evaluation Criteria & Weights

1. **Logical Structure** (20% weight): MECE compliance, argument flow, analytical coherence
2. **Evidence Grounding** (25% weight): Research integration, factual accuracy, external validation  
3. **Depth of Analysis** (18% weight): Framework sophistication, insight quality, multi-dimensional thinking
4. **Actionability** (20% weight): Implementation clarity, executive readiness, practical recommendations
5. **Consulting Quality** (12% weight): Professional standards, client presentation readiness
6. **Factual Accuracy** (5% weight): Truth verification, hallucination detection

### Scoring Scale (1-10)
- **10**: Exceptional, top 5% of consulting work
- **8**: Strong, solid professional quality
- **6**: Adequate, meets basic standards
- **4**: Below standard, needs significant improvement  
- **2**: Poor, major deficiencies

## Analysis A - Strategic Analysis

{a_content}

## Analysis B - Strategic Analysis

{b_content}

## Evaluation Instructions

<thinking>
I need to evaluate both analyses objectively across the six criteria. Let me approach this systematically:

1. First, I'll read through both analyses completely to understand their approaches
2. Then I'll evaluate each criterion independently for both analyses
3. I'll provide specific reasoning for each score
4. Finally, I'll calculate weighted totals and determine the winner

I must remain completely objective and focus only on analytical quality, not methodology or style preferences.
</thinking>

### Detailed Evaluation Process

For each analysis and each criterion, provide:
- A specific score (1-10)
- Detailed justification for the score
- Specific examples from the content
- Comparison insights where relevant

### Required Output Format

Provide your evaluation in the following JSON structure:

```json
{{
  "evaluation_metadata": {{
    "judge_model": "Your model identifier",
    "evaluation_timestamp": "{datetime.utcnow().isoformat()}",
    "confidence_level": "High/Medium/Low confidence in evaluation"
  }},
  "detailed_scores": {{
    "analysis_a": {{
      "logical_structure": {{
        "score": 0,
        "justification": "Specific reasoning with examples"
      }},
      "evidence_grounding": {{
        "score": 0,
        "justification": "Specific reasoning with examples"
      }},
      "depth_of_analysis": {{
        "score": 0,
        "justification": "Specific reasoning with examples"
      }},
      "actionability": {{
        "score": 0,
        "justification": "Specific reasoning with examples"
      }},
      "consulting_quality": {{
        "score": 0,
        "justification": "Specific reasoning with examples"
      }},
      "factual_accuracy": {{
        "score": 0,
        "justification": "Specific reasoning with examples"
      }}
    }},
    "analysis_b": {{
      "logical_structure": {{
        "score": 0,
        "justification": "Specific reasoning with examples"
      }},
      "evidence_grounding": {{
        "score": 0,
        "justification": "Specific reasoning with examples"
      }},
      "depth_of_analysis": {{
        "score": 0,
        "justification": "Specific reasoning with examples"
      }},
      "actionability": {{
        "score": 0,
        "justification": "Specific reasoning with examples"
      }},
      "consulting_quality": {{
        "score": 0,
        "justification": "Specific reasoning with examples"
      }},
      "factual_accuracy": {{
        "score": 0,
        "justification": "Specific reasoning with examples"
      }}
    }}
  }},
  "weighted_totals": {{
    "analysis_a_total": 0.0,
    "analysis_b_total": 0.0,
    "calculation_method": "sum(score * weight) for each criterion"
  }},
  "comparative_analysis": {{
    "winner": "Analysis A" or "Analysis B",
    "margin": 0.0,
    "key_differentiators": [
      "Primary factor that distinguished the winner",
      "Secondary differentiating factor",
      "Additional distinguishing elements"
    ],
    "relative_strengths": {{
      "analysis_a_strengths": ["Key strength 1", "Key strength 2"],
      "analysis_b_strengths": ["Key strength 1", "Key strength 2"]
    }},
    "improvement_areas": {{
      "analysis_a_improvements": ["Area 1", "Area 2"],
      "analysis_b_improvements": ["Area 1", "Area 2"]
    }}
  }},
  "qualitative_assessment": {{
    "overall_summary": "2-3 sentence summary of the evaluation outcome",
    "methodology_observations": "Comments on analytical approaches without revealing bias",
    "quality_differential": "Explanation of the quality gap between analyses",
    "professional_verdict": "Final professional judgment on comparative quality"
  }},
  "confidence_factors": {{
    "evaluation_confidence": "High/Medium/Low",
    "confidence_rationale": "Factors affecting confidence in this evaluation",
    "potential_biases_considered": ["Bias type 1", "Bias type 2"],
    "evaluation_limitations": ["Limitation 1", "Limitation 2"]
  }}
}}
```

## Critical Evaluation Guidelines

1. **Maintain Complete Objectivity**: Evaluate only the content quality, not the methodology
2. **Use Specific Examples**: Ground all scores in concrete evidence from the analyses  
3. **Apply Professional Standards**: Use top-tier consulting firm quality benchmarks
4. **Consider Client Value**: Assess practical business value and implementability
5. **Detect Quality Differentials**: Identify subtle but meaningful quality differences
6. **Provide Clear Reasoning**: Every score must be justified with specific examples

Execute this evaluation with the rigor and objectivity of a senior consulting partner. Your goal is to provide a definitive, quantitative assessment of analytical quality that would stand up to peer review.

BEGIN EVALUATION:"""

        return evaluation_prompt

    def _extract_evaluable_content(self, content: Dict[str, Any]) -> str:
        """
        Extract and format content for evaluation, handling various content structures.
        """

        if isinstance(content, str):
            return content

        # Handle common content structures
        evaluable_sections = []

        # Executive summary
        if "executiveSummary" in content or "executive_summary" in content:
            exec_summary = content.get("executiveSummary") or content.get(
                "executive_summary"
            )
            evaluable_sections.append(
                f"**Executive Summary:**\n{json.dumps(exec_summary, indent=2)}"
            )

        # Governing thought
        if "governingThought" in content:
            evaluable_sections.append(
                f"**Governing Thought:** {content['governingThought']}"
            )

        # Problem deconstruction
        if "problemDeconstruction" in content:
            evaluable_sections.append(
                f"**Problem Analysis:**\n{json.dumps(content['problemDeconstruction'], indent=2)}"
            )

        # Strategic hypotheses
        if "strategicHypotheses" in content:
            evaluable_sections.append(
                f"**Strategic Hypotheses:**\n{json.dumps(content['strategicHypotheses'], indent=2)}"
            )

        # Framework analysis
        if "frameworkAnalysis" in content:
            evaluable_sections.append(
                f"**Framework Analysis:**\n{json.dumps(content['frameworkAnalysis'], indent=2)}"
            )

        # Final recommendations
        if "finalRecommendations" in content:
            evaluable_sections.append(
                f"**Recommendations:**\n{json.dumps(content['finalRecommendations'], indent=2)}"
            )

        # SCQA structure
        if "scqaIntroduction" in content:
            evaluable_sections.append(
                f"**SCQA Structure:**\n{json.dumps(content['scqaIntroduction'], indent=2)}"
            )

        # Implementation roadmap
        if "implementationRoadmap" in content:
            evaluable_sections.append(
                f"**Implementation:**\n{json.dumps(content['implementationRoadmap'], indent=2)}"
            )

        # If no structured content found, convert entire content to string
        if not evaluable_sections:
            evaluable_sections.append(json.dumps(content, indent=2))

        return "\n\n".join(evaluable_sections)

    async def _execute_evaluation(self, evaluation_prompt: str) -> str:
        """Execute the evaluation prompt against the judge LLM"""

        if not self.llm_manager:
            raise RuntimeError("No LLM manager available for evaluation")

        self.logger.info("⚖️ Executing evaluation with judge LLM")

        try:
            # V5.3 Manager Pattern - Use LLMManager for execution
            response = await self.llm_manager.execute_completion(
                prompt=evaluation_prompt,
                system_prompt="",
                temperature=0.1,  # Low temperature for consistency
                max_tokens=4000,
                timeout=120,  # 2 minute timeout for complex evaluation
            )

            # Extract content from LLMManager response
            return response.raw_text

        except Exception as e:
            self.logger.error(f"❌ Evaluation execution failed: {str(e)}")
            raise

    def _parse_evaluation_response(self, judge_response: str) -> Dict[str, Any]:
        """Parse the judge's structured JSON response"""

        try:
            # Extract JSON from response (similar to challenger parsing)
            import re

            # First, try to find JSON blocks in markdown
            json_pattern = r"```json\s*(\{.*?\})\s*```"
            json_matches = re.findall(json_pattern, judge_response, re.DOTALL)

            if json_matches:
                json_str = json_matches[0]
                return json.loads(json_str)

            # Second, try to find JSON without markdown
            start_idx = judge_response.find("{")
            end_idx = judge_response.rfind("}")

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = judge_response[start_idx : end_idx + 1]
                return json.loads(json_str)

            # Third, try parsing entire response
            return json.loads(judge_response)

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse judge response as JSON: {e}")
            # Return mock structure for error cases
            return {
                "evaluation_metadata": {"confidence_level": "Low"},
                "detailed_scores": {
                    "analysis_a": {
                        criterion: {"score": 5, "justification": "Parse error"}
                        for criterion in self.evaluation_criteria.keys()
                    },
                    "analysis_b": {
                        criterion: {"score": 5, "justification": "Parse error"}
                        for criterion in self.evaluation_criteria.keys()
                    },
                },
                "weighted_totals": {"analysis_a_total": 5.0, "analysis_b_total": 5.0},
                "comparative_analysis": {
                    "winner": "Analysis A",
                    "margin": 0.0,
                    "key_differentiators": ["Parse error"],
                },
                "qualitative_assessment": {
                    "overall_summary": "Evaluation failed due to response parsing error"
                },
                "parse_error": str(e),
                "raw_response": (
                    judge_response[:1000] + "..."
                    if len(judge_response) > 1000
                    else judge_response
                ),
            }

    def _compile_evaluation_result(
        self,
        evaluation_id: str,
        start_time: datetime,
        blinded_outputs: Dict[str, BlindedOutput],
        evaluation_data: Dict[str, Any],
        raw_response: str,
    ) -> EvaluationResult:
        """Compile the final evaluation result with proper source mapping"""

        # Map scores back to original sources
        analysis_a_scores = {}
        analysis_b_scores = {}

        # Extract individual criterion scores
        detailed_scores = evaluation_data.get("detailed_scores", {})

        for criterion_name in self.evaluation_criteria.keys():
            a_score = (
                detailed_scores.get("analysis_a", {})
                .get(criterion_name, {})
                .get("score", 5.0)
            )
            b_score = (
                detailed_scores.get("analysis_b", {})
                .get(criterion_name, {})
                .get("score", 5.0)
            )

            analysis_a_scores[criterion_name] = float(a_score)
            analysis_b_scores[criterion_name] = float(b_score)

        # Calculate weighted totals
        weighted_totals = evaluation_data.get("weighted_totals", {})
        analysis_a_total = float(weighted_totals.get("analysis_a_total", 5.0))
        analysis_b_total = float(weighted_totals.get("analysis_b_total", 5.0))

        # Determine winner and map back to source
        comparative = evaluation_data.get("comparative_analysis", {})
        judge_winner = comparative.get("winner", "Analysis A")

        # Map blinded winner back to actual source
        if judge_winner == "Analysis A":
            actual_winner = blinded_outputs["analysis_a"].original_source
            winner_score = analysis_a_total
            loser_score = analysis_b_total
        else:
            actual_winner = blinded_outputs["analysis_b"].original_source
            winner_score = analysis_b_total
            loser_score = analysis_a_total

        margin = abs(winner_score - loser_score)

        # Map scores to METIS/Baseline format
        if blinded_outputs["metis"].label == "Analysis A":
            metis_scores = analysis_a_scores
            metis_total = analysis_a_total
            baseline_scores = analysis_b_scores
            baseline_total = analysis_b_total
        else:
            metis_scores = analysis_b_scores
            metis_total = analysis_b_total
            baseline_scores = analysis_a_scores
            baseline_total = analysis_a_total

        # Extract qualitative analysis
        qualitative = evaluation_data.get("qualitative_assessment", {})
        qualitative_text = qualitative.get(
            "overall_summary", "No qualitative analysis available"
        )

        # Extract confidence
        confidence_factors = evaluation_data.get("confidence_factors", {})
        judge_confidence = 0.8  # Default
        if confidence_factors.get("evaluation_confidence") == "High":
            judge_confidence = 0.9
        elif confidence_factors.get("evaluation_confidence") == "Low":
            judge_confidence = 0.6

        # Build criterion breakdown
        criterion_breakdown = {}
        for criterion_name, criterion_def in self.evaluation_criteria.items():
            criterion_breakdown[criterion_name] = {
                "weight": criterion_def.weight,
                "metis_score": metis_scores.get(criterion_name, 5.0),
                "baseline_score": baseline_scores.get(criterion_name, 5.0),
                "winner": (
                    "METIS"
                    if metis_scores.get(criterion_name, 5.0)
                    > baseline_scores.get(criterion_name, 5.0)
                    else "Baseline"
                ),
                "difference": metis_scores.get(criterion_name, 5.0)
                - baseline_scores.get(criterion_name, 5.0),
            }

        return EvaluationResult(
            evaluation_id=evaluation_id,
            timestamp=start_time,
            judge_model=response.model_name if "response" in locals() else "unknown",
            execution_time=0.0,  # Will be set by caller
            # Audit trail
            analysis_a_source=blinded_outputs["analysis_a"].original_source,
            analysis_b_source=blinded_outputs["analysis_b"].original_source,
            # Mapped scores
            analysis_a_scores=metis_scores,
            analysis_b_scores=baseline_scores,
            # Results
            analysis_a_total=metis_total,
            analysis_b_total=baseline_total,
            winner=actual_winner.upper(),
            margin=margin,
            # Detailed analysis
            criterion_breakdown=criterion_breakdown,
            qualitative_analysis=qualitative_text,
            judge_confidence=judge_confidence,
        )

    def get_evaluation_criteria(self) -> Dict[str, EvaluationCriterion]:
        """Get the evaluation criteria for transparency"""
        return self.evaluation_criteria.copy()

    def get_judge_metadata(self) -> Dict[str, Any]:
        """Get metadata about the judge's capabilities and configuration"""

        return {
            "judgeVersion": "1.0.0",
            "judgeModel": self.judge_model,
            "reasoningEnabled": self.enable_reasoning,
            "llmManagerAvailable": self.llm_manager is not None,
            "activeProviders": (
                self.llm_manager.get_performance_stats()["providers"]
                if self.llm_manager
                else None
            ),
            "evaluationCriteria": {
                name: {
                    "weight": criterion.weight,
                    "description": criterion.description,
                    "maxScore": criterion.max_score,
                }
                for name, criterion in self.evaluation_criteria.items()
            },
            "methodologyFeatures": [
                "Rigorous blinding to eliminate bias",
                "Multi-criteria weighted evaluation",
                "DeepSeek V3 reasoning integration",
                "Structured JSON response parsing",
                "Professional consulting standards",
                "Statistical confidence measurement",
                "Complete audit trail maintenance",
            ],
        }


# Global judge instance for easy access
_global_judge: Optional[BlindJudge] = None


def get_blind_judge(
    judge_model: str = "deepseek", enable_reasoning: bool = True
) -> BlindJudge:
    """Get or create the global BlindJudge instance"""
    global _global_judge

    if _global_judge is None:
        _global_judge = BlindJudge(judge_model, enable_reasoning)

    return _global_judge


# Convenience functions
def evaluate_metis_vs_baseline(
    metis_output: Dict[str, Any],
    baseline_output: Dict[str, Any],
    original_query: str = "",
    context: Dict[str, Any] = None,
) -> EvaluationResult:
    """Execute a complete METIS vs Baseline evaluation"""
    judge = get_blind_judge()
    return judge.evaluate_outputs(
        metis_output, baseline_output, original_query, context
    )
