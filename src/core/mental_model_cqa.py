"""
Mental Model Cognitive Quality Assurance (CQA) Extension
========================================================

Specialized quality assessment system for mental models that extends
the METIS V5.3 CQA framework with mental model-specific evaluation
capabilities, rubrics, and validation processes.
"""

import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Core METIS imports
from src.core.contracts.mental_model_rubrics import (
    MentalModelType,
    MentalModelQualityDimension,
    MentalModelRubric,
    get_mental_model_rubric_registry,
)
from src.core.unified_context_stream import get_unified_context_stream, ContextEventType
from src.engine.core.llm_manager import get_llm_manager

# Import parser models
from src.ingestion.mental_model_parser import MentalModelData


@dataclass
class MentalModelQualityScore:
    """Quality score for a specific mental model dimension."""

    dimension: MentalModelQualityDimension
    score: float  # 1.0-10.0 scale
    rationale: str
    evidence: List[str]  # Supporting evidence from the mental model text
    confidence: float  # 0.0-1.0 scale


@dataclass
class MentalModelCQAResult:
    """Complete CQA evaluation result for a mental model."""

    mental_model_id: str
    mental_model_name: str
    model_type: MentalModelType
    rubric_used: str
    evaluation_timestamp: str

    # Individual dimension scores
    dimension_scores: Dict[MentalModelQualityDimension, MentalModelQualityScore]

    # Aggregate metrics
    overall_score: float
    weighted_score: float
    confidence_level: float

    # Quality assessment
    quality_tier: str  # "excellent", "good", "average", "poor"
    validation_status: str  # "passed", "failed", "review_needed"

    # Audit information
    evaluator_version: str
    execution_time_ms: int
    context_stream_id: Optional[str] = None


class MentalModelCQAEvaluator:
    """
    Specialized quality evaluator for mental models.

    Extends the METIS CQA framework with mental model-specific evaluation
    logic, rubrics, and scoring mechanisms.
    """

    VERSION = "1.0"

    def __init__(self, context_stream_id: Optional[str] = None):
        """
        Initialize the mental model CQA evaluator.

        Args:
            context_stream_id: Optional context stream ID for audit trail
        """
        self.rubric_registry = get_mental_model_rubric_registry()

        self.context_stream = get_unified_context_stream()
        self.llm_manager = None  # Initialized on first use

        # Quality tier thresholds
        self.quality_thresholds = {
            "excellent": 8.5,
            "good": 7.0,
            "average": 5.5,
            "poor": 0.0,
        }

        # Validation pass/fail thresholds
        self.validation_thresholds = {
            "auto_pass": 7.5,  # Automatically passes validation
            "auto_fail": 4.0,  # Automatically fails validation
            "review_needed": 4.0,  # Requires human review
        }

    async def _get_llm_manager(self):
        """Get LLM manager instance (lazy initialization)."""
        if self.llm_manager is None:
            self.llm_manager = get_llm_manager(context_stream=self.context_stream)
        return self.llm_manager

    def _determine_model_type(self, mental_model: MentalModelData) -> MentalModelType:
        """
        Determine the mental model type based on content analysis.

        Args:
            mental_model: The parsed mental model data

        Returns:
            Detected mental model type
        """
        name_lower = mental_model.name.lower()
        description_lower = mental_model.description.lower()

        # Type detection keywords
        type_keywords = {
            MentalModelType.COGNITIVE_BIASES: [
                "bias",
                "cognitive",
                "heuristic",
                "fallacy",
                "error",
                "psychology",
            ],
            MentalModelType.DECISION_FRAMEWORKS: [
                "decision",
                "framework",
                "choose",
                "evaluate",
                "criteria",
                "options",
            ],
            MentalModelType.SYSTEMS_THINKING: [
                "system",
                "systems",
                "holistic",
                "interconnect",
                "feedback",
                "loop",
            ],
            MentalModelType.PROBLEM_SOLVING: [
                "problem",
                "solve",
                "solution",
                "approach",
                "method",
                "technique",
            ],
            MentalModelType.STRATEGIC_FRAMEWORKS: [
                "strategic",
                "strategy",
                "competitive",
                "advantage",
                "positioning",
            ],
            MentalModelType.ANALYTICAL_TOOLS: [
                "analysis",
                "analytical",
                "tool",
                "model",
                "framework",
                "methodology",
            ],
        }

        # Score each type based on keyword matches
        type_scores = {}
        combined_text = f"{name_lower} {description_lower}"

        for model_type, keywords in type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            type_scores[model_type] = score

        # Return type with highest score, or default to analytical tools
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            if type_scores[best_type] > 0:
                return best_type

        return MentalModelType.ANALYTICAL_TOOLS

    async def _evaluate_dimension(
        self,
        mental_model: MentalModelData,
        dimension: MentalModelQualityDimension,
        rubric: MentalModelRubric,
    ) -> MentalModelQualityScore:
        """
        Evaluate a single quality dimension for a mental model.

        Args:
            mental_model: The mental model to evaluate
            dimension: The quality dimension to assess
            rubric: The evaluation rubric to use

        Returns:
            Quality score for the specified dimension
        """
        # Find criteria for this dimension
        criteria = None
        for crit in rubric.criteria:
            if crit.dimension == dimension:
                criteria = crit
                break

        if not criteria:
            # Fallback generic evaluation
            return MentalModelQualityScore(
                dimension=dimension,
                score=5.0,
                rationale="No specific criteria found for this dimension",
                evidence=[],
                confidence=0.5,
            )

        # Prepare evaluation prompt
        evaluation_prompt = self._build_evaluation_prompt(
            mental_model, dimension, criteria
        )

        llm_manager = await self._get_llm_manager()

        # Get LLM evaluation
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a specialized mental model quality evaluator. Provide detailed, objective assessment based on the given criteria.",
                },
                {"role": "user", "content": evaluation_prompt},
            ]

            response = await llm_manager.call_llm(
                messages=messages,
                temperature=0.3,  # Lower temperature for consistent evaluation
                max_tokens=1000,
            )

            # Parse the evaluation response
            return self._parse_evaluation_response(response, dimension)

        except Exception as e:
            self.context_stream.add_event(
                ContextEventType.ERROR,
                {"error": f"LLM evaluation failed for dimension {dimension}: {str(e)}"},
            )

            # Fallback to basic heuristic evaluation
            return self._heuristic_evaluation(mental_model, dimension)

    def _build_evaluation_prompt(
        self,
        mental_model: MentalModelData,
        dimension: MentalModelQualityDimension,
        criteria: Any,
    ) -> str:
        """Build evaluation prompt for a specific dimension."""

        prompt = f"""
MENTAL MODEL QUALITY EVALUATION

Evaluate the following mental model on the dimension of {dimension.value.upper()}.

MENTAL MODEL:
Name: {mental_model.name}
Description: {mental_model.description}
Applications: {', '.join(mental_model.applications)}
Examples: {', '.join(mental_model.examples)}
Related Concepts: {', '.join(mental_model.related_concepts)}

EVALUATION CRITERIA FOR {dimension.value.upper()}:
- Poor (1-3): {criteria.criteria_1_3}
- Average (4-6): {criteria.criteria_4_6}  
- Good (7-8): {criteria.criteria_7_8}
- Excellent (9-10): {criteria.criteria_9_10}

INSTRUCTIONS:
1. Carefully analyze the mental model content against the criteria
2. Provide a score from 1-10 
3. Give detailed rationale for your score
4. List specific evidence from the mental model that supports your assessment
5. Rate your confidence in this evaluation (0-100%)

RESPONSE FORMAT (JSON):
{{
    "score": <numeric_score_1_to_10>,
    "rationale": "<detailed_explanation_of_score>",
    "evidence": ["<evidence_point_1>", "<evidence_point_2>", "..."],
    "confidence": <confidence_0_to_100>
}}
"""
        return prompt.strip()

    def _parse_evaluation_response(
        self, response: str, dimension: MentalModelQualityDimension
    ) -> MentalModelQualityScore:
        """Parse LLM evaluation response into structured score."""
        try:
            # Try to parse as JSON
            if "```json" in response:
                json_part = response.split("```json")[1].split("```")[0]
            elif "{" in response and "}" in response:
                start_idx = response.find("{")
                end_idx = response.rfind("}") + 1
                json_part = response[start_idx:end_idx]
            else:
                json_part = response

            data = json.loads(json_part.strip())

            return MentalModelQualityScore(
                dimension=dimension,
                score=float(data.get("score", 5.0)),
                rationale=data.get("rationale", "No rationale provided"),
                evidence=data.get("evidence", []),
                confidence=float(data.get("confidence", 50)) / 100.0,
            )

        except Exception:
            # Fallback parsing for non-JSON responses
            score = 5.0
            rationale = response[:500] + "..." if len(response) > 500 else response

            # Try to extract numeric score
            import re

            score_matches = re.findall(r"\b([1-9]|10)\b", response)
            if score_matches:
                score = float(score_matches[0])

            return MentalModelQualityScore(
                dimension=dimension,
                score=score,
                rationale=rationale,
                evidence=[],
                confidence=0.6,
            )

    def _heuristic_evaluation(
        self, mental_model: MentalModelData, dimension: MentalModelQualityDimension
    ) -> MentalModelQualityScore:
        """Fallback heuristic evaluation when LLM is unavailable."""

        # Simple heuristic scoring based on content length and completeness
        description_length = len(mental_model.description)
        applications_count = len(mental_model.applications)
        examples_count = len(mental_model.examples)

        # Base score calculation
        base_score = 5.0

        if dimension == MentalModelQualityDimension.CLARITY:
            # Clarity based on description length and structure
            if description_length > 500:
                base_score += 1.5
            if examples_count > 2:
                base_score += 1.0

        elif dimension == MentalModelQualityDimension.APPLICABILITY:
            # Applicability based on number of applications
            base_score += min(applications_count * 0.5, 3.0)

        elif dimension == MentalModelQualityDimension.COMPLETENESS:
            # Completeness based on filled fields
            completeness_score = 0
            if description_length > 200:
                completeness_score += 1
            if applications_count > 1:
                completeness_score += 1
            if examples_count > 1:
                completeness_score += 1
            if len(mental_model.related_concepts) > 1:
                completeness_score += 1
            base_score += completeness_score

        # Cap at 10.0
        final_score = min(base_score, 10.0)

        return MentalModelQualityScore(
            dimension=dimension,
            score=final_score,
            rationale=f"Heuristic evaluation based on content analysis (dimension: {dimension.value})",
            evidence=[
                f"Description length: {description_length} chars",
                f"Applications: {applications_count}",
                f"Examples: {examples_count}",
            ],
            confidence=0.7,
        )

    def _determine_quality_tier(self, overall_score: float) -> str:
        """Determine quality tier based on overall score."""
        for tier, threshold in self.quality_thresholds.items():
            if overall_score >= threshold:
                return tier
        return "poor"

    def _determine_validation_status(
        self, overall_score: float, confidence: float
    ) -> str:
        """Determine validation status based on score and confidence."""
        if (
            overall_score >= self.validation_thresholds["auto_pass"]
            and confidence >= 0.8
        ):
            return "passed"
        elif (
            overall_score <= self.validation_thresholds["auto_fail"] or confidence < 0.5
        ):
            return "failed"
        else:
            return "review_needed"

    async def evaluate_mental_model(
        self,
        mental_model: MentalModelData,
        model_type: Optional[MentalModelType] = None,
    ) -> MentalModelCQAResult:
        """
        Perform complete CQA evaluation of a mental model.

        Args:
            mental_model: The mental model to evaluate
            model_type: Optional override for model type detection

        Returns:
            Complete CQA evaluation result
        """
        start_time = time.time()
        evaluation_id = str(uuid.uuid4())

        # Log evaluation start
        self.context_stream.add_event(
            ContextEventType.CONSULTANT_ANALYSIS_START,
            {
                "evaluation_id": evaluation_id,
                "mental_model_name": mental_model.name,
                "evaluator_version": self.VERSION,
            },
        )

        try:
            # Determine model type
            detected_type = model_type or self._determine_model_type(mental_model)
            rubric = self.rubric_registry.get_rubric_for_model_type(detected_type)

            # Evaluate each dimension
            dimension_scores = {}
            confidence_scores = []

            for criterion in rubric.criteria:
                dimension_score = await self._evaluate_dimension(
                    mental_model, criterion.dimension, rubric
                )
                dimension_scores[criterion.dimension] = dimension_score
                confidence_scores.append(dimension_score.confidence)

            # Calculate aggregate metrics
            raw_scores = [score.score for score in dimension_scores.values()]
            overall_score = sum(raw_scores) / len(raw_scores)

            # Calculate weighted score using rubric weights
            dimension_score_dict = {
                dim: score.score for dim, score in dimension_scores.items()
            }
            weighted_score = self.rubric_registry.calculate_weighted_score(
                rubric.rubric_id, dimension_score_dict
            )

            overall_confidence = sum(confidence_scores) / len(confidence_scores)

            # Determine quality assessments
            quality_tier = self._determine_quality_tier(weighted_score)
            validation_status = self._determine_validation_status(
                weighted_score, overall_confidence
            )

            execution_time_ms = int((time.time() - start_time) * 1000)

            # Create result
            result = MentalModelCQAResult(
                mental_model_id=mental_model.id,
                mental_model_name=mental_model.name,
                model_type=detected_type,
                rubric_used=rubric.rubric_id,
                evaluation_timestamp=datetime.now().isoformat(),
                dimension_scores=dimension_scores,
                overall_score=overall_score,
                weighted_score=weighted_score,
                confidence_level=overall_confidence,
                quality_tier=quality_tier,
                validation_status=validation_status,
                evaluator_version=self.VERSION,
                execution_time_ms=execution_time_ms,
                context_stream_id=getattr(self.context_stream, "trace_id", None),
            )

            # Log evaluation completion
            self.context_stream.add_event(
                ContextEventType.CONSULTANT_ANALYSIS_COMPLETE,
                {
                    "evaluation_id": evaluation_id,
                    "overall_score": overall_score,
                    "weighted_score": weighted_score,
                    "quality_tier": quality_tier,
                    "validation_status": validation_status,
                    "execution_time_ms": execution_time_ms,
                },
            )

            return result

        except Exception as e:
            self.context_stream.add_event(
                ContextEventType.ERROR_OCCURRED,
                {
                    "evaluation_id": evaluation_id,
                    "error": str(e),
                    "execution_time_ms": int((time.time() - start_time) * 1000),
                },
            )
            raise

    async def batch_evaluate_mental_models(
        self, mental_models: List[MentalModelData]
    ) -> List[MentalModelCQAResult]:
        """
        Evaluate multiple mental models in batch.

        Args:
            mental_models: List of mental models to evaluate

        Returns:
            List of CQA evaluation results
        """
        self.context_stream.add_event(
            ContextEventType.ENGAGEMENT_STARTED,
            {"operation": "batch_mental_model_cqa", "model_count": len(mental_models)},
        )

        results = []
        for mental_model in mental_models:
            try:
                result = await self.evaluate_mental_model(mental_model)
                results.append(result)
            except Exception as e:
                self.context_stream.add_event(
                    ContextEventType.ERROR_OCCURRED,
                    {
                        "mental_model_id": mental_model.id,
                        "error": f"Batch evaluation failed: {str(e)}",
                    },
                )
                continue

        self.context_stream.add_event(
            ContextEventType.ENGAGEMENT_COMPLETED,
            {
                "operation": "batch_mental_model_cqa",
                "results_count": len(results),
                "success_rate": len(results) / len(mental_models),
            },
        )

        return results
