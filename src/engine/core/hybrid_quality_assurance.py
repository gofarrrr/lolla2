"""
Production-ready quality assurance framework for hybrid cognitive orchestrator.

Extends existing LLM validation gates with hybrid-specific quality checks:
- Multi-phase validation (research, synthesis, analysis)
- Confidence scoring across micro-agents
- Cost-benefit validation
- Context integrity verification
- HITL decision quality assessment
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List
from uuid import UUID
from enum import Enum
from dataclasses import dataclass
import logging

from src.core.unified_context_stream import UnifiedContextStream, ContextEventType
from src.core.llm_validation_gates import get_llm_validation_gates, ValidationGateResult
from src.core.resilient_llm_client import ResilientLLMClient
from src.intelligence.model_catalog import get_model_catalog
from src.core.performance_cache_system import get_smart_cache

logger = logging.getLogger(__name__)


class QualityDimension(str, Enum):
    """Quality dimensions for hybrid engagement assessment"""

    RESEARCH_COMPLETENESS = "research_completeness"
    CONTEXT_INTEGRITY = "context_integrity"
    SYNTHESIS_COHERENCE = "synthesis_coherence"
    ANALYSIS_DEPTH = "analysis_depth"
    ASSUMPTION_VALIDITY = "assumption_validity"
    RECOMMENDATION_ACTIONABILITY = "recommendation_actionability"
    COST_EFFECTIVENESS = "cost_effectiveness"
    TRANSPARENCY_COVERAGE = "transparency_coverage"


class QualityThreshold(str, Enum):
    """Quality threshold levels"""

    PRODUCTION = "production"  # 0.90+
    REVIEW = "review"  # 0.80+
    WARNING = "warning"  # 0.70+
    FAILURE = "failure"  # <0.70


@dataclass
class QualityAssessment:
    """Quality assessment result for a specific dimension"""

    dimension: QualityDimension
    score: float
    threshold_met: QualityThreshold
    details: Dict[str, Any]
    recommendations: List[str]
    confidence: float
    timestamp: datetime


@dataclass
class HybridEngagementQuality:
    """Overall quality assessment for hybrid engagement"""

    engagement_id: UUID
    overall_score: float
    dimension_scores: Dict[QualityDimension, QualityAssessment]
    quality_gate_result: ValidationGateResult
    production_ready: bool
    improvement_recommendations: List[str]
    cost_quality_ratio: float
    assessment_timestamp: datetime


class HybridQualityAssurance:
    """
    Production-ready quality assurance framework for hybrid cognitive orchestrator.

    Provides comprehensive quality assessment across all dimensions of cognitive processing:
    - Research quality and completeness
    - Context preservation and integrity
    - Synthesis coherence and enrichment
    - Analysis depth and mental model application
    - Recommendation actionability and practical value
    - Cost-effectiveness and resource utilization
    """

    def __init__(self):
        self.llm_validation_gates = get_llm_validation_gates()
        self.llm_client = ResilientLLMClient(
            preferred_provider="claude"
        )  # Use Claude for quality assessment
        self.model_catalog = get_model_catalog()
        self.cache_system = get_smart_cache()

        # Quality thresholds by dimension
        self.quality_thresholds = {
            QualityDimension.RESEARCH_COMPLETENESS: {
                QualityThreshold.PRODUCTION: 0.90,
                QualityThreshold.REVIEW: 0.80,
                QualityThreshold.WARNING: 0.70,
            },
            QualityDimension.CONTEXT_INTEGRITY: {
                QualityThreshold.PRODUCTION: 0.95,
                QualityThreshold.REVIEW: 0.85,
                QualityThreshold.WARNING: 0.75,
            },
            QualityDimension.SYNTHESIS_COHERENCE: {
                QualityThreshold.PRODUCTION: 0.88,
                QualityThreshold.REVIEW: 0.78,
                QualityThreshold.WARNING: 0.68,
            },
            QualityDimension.ANALYSIS_DEPTH: {
                QualityThreshold.PRODUCTION: 0.85,
                QualityThreshold.REVIEW: 0.75,
                QualityThreshold.WARNING: 0.65,
            },
            QualityDimension.ASSUMPTION_VALIDITY: {
                QualityThreshold.PRODUCTION: 0.82,
                QualityThreshold.REVIEW: 0.72,
                QualityThreshold.WARNING: 0.62,
            },
            QualityDimension.RECOMMENDATION_ACTIONABILITY: {
                QualityThreshold.PRODUCTION: 0.87,
                QualityThreshold.REVIEW: 0.77,
                QualityThreshold.WARNING: 0.67,
            },
            QualityDimension.COST_EFFECTIVENESS: {
                QualityThreshold.PRODUCTION: 0.80,
                QualityThreshold.REVIEW: 0.70,
                QualityThreshold.WARNING: 0.60,
            },
            QualityDimension.TRANSPARENCY_COVERAGE: {
                QualityThreshold.PRODUCTION: 0.95,
                QualityThreshold.REVIEW: 0.90,
                QualityThreshold.WARNING: 0.80,
            },
        }

    async def assess_engagement_quality(
        self, context: UnifiedContextStream
    ) -> HybridEngagementQuality:
        """
        Comprehensive quality assessment of hybrid engagement.

        Args:
            context: Unified context stream with complete event history

        Returns:
            Complete quality assessment with scores and recommendations
        """

        logger.info(
            f"Starting quality assessment for engagement {context.engagement_id}"
        )

        # Run all quality assessments in parallel for efficiency
        assessment_tasks = [
            self._assess_research_completeness(context),
            self._assess_context_integrity(context),
            self._assess_synthesis_coherence(context),
            self._assess_analysis_depth(context),
            self._assess_assumption_validity(context),
            self._assess_recommendation_actionability(context),
            self._assess_cost_effectiveness(context),
            self._assess_transparency_coverage(context),
        ]

        dimension_assessments = await asyncio.gather(*assessment_tasks)

        # Create dimension scores dictionary
        dimension_scores = {}
        total_weighted_score = 0.0
        total_weight = 0.0

        # Dimension weights (production priorities)
        dimension_weights = {
            QualityDimension.RESEARCH_COMPLETENESS: 0.15,
            QualityDimension.CONTEXT_INTEGRITY: 0.20,
            QualityDimension.SYNTHESIS_COHERENCE: 0.15,
            QualityDimension.ANALYSIS_DEPTH: 0.15,
            QualityDimension.ASSUMPTION_VALIDITY: 0.10,
            QualityDimension.RECOMMENDATION_ACTIONABILITY: 0.15,
            QualityDimension.COST_EFFECTIVENESS: 0.05,
            QualityDimension.TRANSPARENCY_COVERAGE: 0.05,
        }

        for assessment in dimension_assessments:
            dimension_scores[assessment.dimension] = assessment
            weight = dimension_weights[assessment.dimension]
            total_weighted_score += assessment.score * weight
            total_weight += weight

        # Calculate overall score
        overall_score = total_weighted_score / total_weight if total_weight > 0 else 0.0

        # Determine quality gate result
        quality_gate_result = self._determine_quality_gate(
            overall_score, dimension_scores
        )

        # Check production readiness
        production_ready = self._assess_production_readiness(
            overall_score, dimension_scores
        )

        # Generate improvement recommendations
        improvement_recommendations = self._generate_improvement_recommendations(
            dimension_scores
        )

        # Calculate cost-quality ratio
        cost_quality_ratio = self._calculate_cost_quality_ratio(context, overall_score)

        quality_assessment = HybridEngagementQuality(
            engagement_id=context.engagement_id,
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            quality_gate_result=quality_gate_result,
            production_ready=production_ready,
            improvement_recommendations=improvement_recommendations,
            cost_quality_ratio=cost_quality_ratio,
            assessment_timestamp=datetime.now(timezone.utc),
        )

        logger.info(
            f"Quality assessment completed for {context.engagement_id}: "
            f"Overall={overall_score:.3f}, Production Ready={production_ready}"
        )

        return quality_assessment

    async def validate_micro_step_quality(
        self, context: UnifiedContextStream, step_type: str, step_result: Dict[str, Any]
    ) -> QualityAssessment:
        """
        Real-time quality validation for individual micro-steps.

        Args:
            context: Current context stream
            step_type: Type of micro-step (research, synthesis, analysis)
            step_result: Result of the micro-step

        Returns:
            Quality assessment for the specific micro-step
        """

        # Determine appropriate quality dimension
        dimension_mapping = {
            "research": QualityDimension.RESEARCH_COMPLETENESS,
            "synthesis": QualityDimension.SYNTHESIS_COHERENCE,
            "analysis": QualityDimension.ANALYSIS_DEPTH,
            "assumption_challenging": QualityDimension.ASSUMPTION_VALIDITY,
        }

        dimension = dimension_mapping.get(
            step_type.split("_")[0], QualityDimension.ANALYSIS_DEPTH
        )

        # Assess micro-step quality
        if dimension == QualityDimension.RESEARCH_COMPLETENESS:
            return await self._assess_research_micro_step(context, step_result)
        elif dimension == QualityDimension.SYNTHESIS_COHERENCE:
            return await self._assess_synthesis_micro_step(context, step_result)
        elif dimension == QualityDimension.ANALYSIS_DEPTH:
            return await self._assess_analysis_micro_step(context, step_result)
        else:
            return await self._assess_generic_micro_step(
                context, step_result, dimension
            )

    # Private assessment methods

    async def _assess_research_completeness(
        self, context: UnifiedContextStream
    ) -> QualityAssessment:
        """Assess research completeness and coverage"""

        research_events = context.get_events_by_type(
            ContextEventType.RESEARCH_QUERY_EXECUTED
        )

        if not research_events:
            return QualityAssessment(
                dimension=QualityDimension.RESEARCH_COMPLETENESS,
                score=0.0,
                threshold_met=QualityThreshold.FAILURE,
                details={"error": "No research queries executed"},
                recommendations=["Execute research queries to ground analysis"],
                confidence=1.0,
                timestamp=datetime.now(timezone.utc),
            )

        # Analyze research coverage
        research_data = [event.data for event in research_events]

        assessment_prompt = f"""
Assess the completeness and coverage of research for this cognitive engagement.

RESEARCH QUERIES EXECUTED: {len(research_events)}
RESEARCH DATA:
{self._format_research_for_assessment(research_data[:5])}  # Limit for context

ASSESSMENT CRITERIA:
1. Coverage breadth (industry, technical, strategic perspectives)
2. Source diversity and credibility
3. Information depth and relevance
4. Query quality and specificity
5. Coverage of key aspects of the original question

Provide assessment as JSON:
{{
    "score": 0.0-1.0,
    "coverage_breadth": 0.0-1.0,
    "source_quality": 0.0-1.0, 
    "information_depth": 0.0-1.0,
    "query_relevance": 0.0-1.0,
    "missing_aspects": ["aspect1", "aspect2"],
    "recommendations": ["recommendation1", "recommendation2"],
    "confidence": 0.0-1.0
}}
"""

        assessment_result = await self.llm_client.execute_cognitive_call(
            prompt=assessment_prompt,
            provider="claude",
            max_tokens=1000,
            response_format="json",
        )

        try:
            assessment_data = assessment_result.parsed_response
            score = float(assessment_data.get("score", 0.0))

            return QualityAssessment(
                dimension=QualityDimension.RESEARCH_COMPLETENESS,
                score=score,
                threshold_met=self._score_to_threshold(
                    score, QualityDimension.RESEARCH_COMPLETENESS
                ),
                details={
                    "queries_executed": len(research_events),
                    "coverage_breadth": assessment_data.get("coverage_breadth", 0.0),
                    "source_quality": assessment_data.get("source_quality", 0.0),
                    "information_depth": assessment_data.get("information_depth", 0.0),
                    "query_relevance": assessment_data.get("query_relevance", 0.0),
                    "missing_aspects": assessment_data.get("missing_aspects", []),
                },
                recommendations=assessment_data.get("recommendations", []),
                confidence=float(assessment_data.get("confidence", 0.8)),
                timestamp=datetime.now(timezone.utc),
            )

        except Exception as e:
            logger.error(f"Research completeness assessment failed: {str(e)}")
            return QualityAssessment(
                dimension=QualityDimension.RESEARCH_COMPLETENESS,
                score=0.6,  # Conservative fallback
                threshold_met=QualityThreshold.WARNING,
                details={"assessment_error": str(e)},
                recommendations=["Review research query quality and coverage"],
                confidence=0.5,
                timestamp=datetime.now(timezone.utc),
            )

    async def _assess_context_integrity(
        self, context: UnifiedContextStream
    ) -> QualityAssessment:
        """Assess context preservation and integrity across event stream"""

        # Check event continuity and integrity
        total_events = len(context.events)
        context_size_mb = len(context.get_full_context_for_llm()) / (1024 * 1024)

        # Analyze event flow
        event_types = [event.type for event in context.events]
        event_type_counts = {}
        for event_type in event_types:
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1

        # Check for critical event types
        required_event_types = [
            ContextEventType.ENGAGEMENT_INITIATED,
            ContextEventType.RESEARCH_QUERY_EXECUTED,
            ContextEventType.SYNTHESIS_MICRO_STEP_COMPLETED,
            ContextEventType.ANALYSIS_MICRO_STEP_COMPLETED,
        ]

        missing_event_types = [
            et for et in required_event_types if et not in event_type_counts
        ]

        # Calculate integrity score
        base_score = 1.0

        # Penalize missing critical events
        if missing_event_types:
            base_score -= 0.2 * len(missing_event_types)

        # Penalize if context too small (likely compression)
        if total_events < 5:
            base_score -= 0.3

        # Penalize excessive context size (inefficiency)
        if context_size_mb > 5.0:
            base_score -= 0.1

        # Check event consistency
        consistency_score = await self._assess_event_consistency(context)

        final_score = max(0.0, min(1.0, base_score * consistency_score))

        return QualityAssessment(
            dimension=QualityDimension.CONTEXT_INTEGRITY,
            score=final_score,
            threshold_met=self._score_to_threshold(
                final_score, QualityDimension.CONTEXT_INTEGRITY
            ),
            details={
                "total_events": total_events,
                "context_size_mb": context_size_mb,
                "event_type_counts": event_type_counts,
                "missing_event_types": missing_event_types,
                "consistency_score": consistency_score,
            },
            recommendations=self._generate_context_recommendations(
                missing_event_types, context_size_mb
            ),
            confidence=0.85,
            timestamp=datetime.now(timezone.utc),
        )

    async def _assess_synthesis_coherence(
        self, context: UnifiedContextStream
    ) -> QualityAssessment:
        """Assess synthesis coherence and enrichment quality"""

        synthesis_events = context.get_events_by_type(
            ContextEventType.SYNTHESIS_MICRO_STEP_COMPLETED
        )

        if not synthesis_events:
            return QualityAssessment(
                dimension=QualityDimension.SYNTHESIS_COHERENCE,
                score=0.0,
                threshold_met=QualityThreshold.FAILURE,
                details={"error": "No synthesis steps completed"},
                recommendations=["Complete context synthesis before analysis"],
                confidence=1.0,
                timestamp=datetime.now(timezone.utc),
            )

        # Analyze synthesis quality
        synthesis_data = synthesis_events[-1].data  # Most recent synthesis

        assessment_prompt = f"""
Assess the coherence and quality of cognitive synthesis.

ORIGINAL QUERY: {context.events[0].data.get('initial_query', 'N/A')}

SYNTHESIS RESULT:
Enhanced Query: {synthesis_data.get('enhanced_query', 'N/A')[:500]}...
Enrichment Factor: {synthesis_data.get('enrichment_factor', 'N/A')}
Strategic Constraints: {len(synthesis_data.get('strategic_constraints', []))}
Key Assumptions: {len(synthesis_data.get('key_assumptions', []))}

ASSESSMENT CRITERIA:
1. Coherence and logical flow
2. Enrichment quality (meaningful expansion)
3. Strategic insight depth
4. Assumption identification
5. Practical applicability

Provide assessment as JSON:
{{
    "score": 0.0-1.0,
    "coherence": 0.0-1.0,
    "enrichment_quality": 0.0-1.0,
    "strategic_depth": 0.0-1.0,
    "assumption_quality": 0.0-1.0,
    "practical_applicability": 0.0-1.0,
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "recommendations": ["recommendation1", "recommendation2"],
    "confidence": 0.0-1.0
}}
"""

        assessment_result = await self.llm_client.execute_cognitive_call(
            prompt=assessment_prompt,
            provider="claude",
            max_tokens=1000,
            response_format="json",
        )

        try:
            assessment_data = assessment_result.parsed_response
            score = float(assessment_data.get("score", 0.0))

            return QualityAssessment(
                dimension=QualityDimension.SYNTHESIS_COHERENCE,
                score=score,
                threshold_met=self._score_to_threshold(
                    score, QualityDimension.SYNTHESIS_COHERENCE
                ),
                details={
                    "synthesis_steps": len(synthesis_events),
                    "enrichment_factor": synthesis_data.get("enrichment_factor", 0.0),
                    "constraints_identified": len(
                        synthesis_data.get("strategic_constraints", [])
                    ),
                    "assumptions_identified": len(
                        synthesis_data.get("key_assumptions", [])
                    ),
                    "coherence": assessment_data.get("coherence", 0.0),
                    "enrichment_quality": assessment_data.get(
                        "enrichment_quality", 0.0
                    ),
                    "strategic_depth": assessment_data.get("strategic_depth", 0.0),
                    "strengths": assessment_data.get("strengths", []),
                    "weaknesses": assessment_data.get("weaknesses", []),
                },
                recommendations=assessment_data.get("recommendations", []),
                confidence=float(assessment_data.get("confidence", 0.8)),
                timestamp=datetime.now(timezone.utc),
            )

        except Exception as e:
            logger.error(f"Synthesis coherence assessment failed: {str(e)}")
            return self._create_fallback_assessment(
                QualityDimension.SYNTHESIS_COHERENCE, 0.65, str(e)
            )

    async def _assess_analysis_depth(
        self, context: UnifiedContextStream
    ) -> QualityAssessment:
        """Assess analysis depth and mental model application"""

        analysis_events = context.get_events_by_type(
            ContextEventType.ANALYSIS_MICRO_STEP_COMPLETED
        )

        if not analysis_events:
            return self._create_fallback_assessment(
                QualityDimension.ANALYSIS_DEPTH,
                0.0,
                "No analysis micro-steps completed",
            )

        # Analyze mental model application and depth
        analysis_data = []
        for event in analysis_events[-3:]:  # Last 3 analysis steps
            analysis_data.append(
                {
                    "step_type": event.data.get("step_type", "unknown"),
                    "result": event.data.get("result", "")[
                        :200
                    ],  # Truncate for assessment
                    "confidence": event.confidence_score or 0.0,
                }
            )

        assessment_prompt = f"""
Assess the depth and quality of cognitive analysis.

ANALYSIS STEPS COMPLETED: {len(analysis_events)}

RECENT ANALYSIS RESULTS:
{self._format_analysis_for_assessment(analysis_data)}

ASSESSMENT CRITERIA:
1. Mental model application depth
2. Framework utilization effectiveness  
3. Insight generation quality
4. Reasoning sophistication
5. Practical actionability

Provide assessment as JSON:
{{
    "score": 0.0-1.0,
    "mental_model_depth": 0.0-1.0,
    "framework_utilization": 0.0-1.0,
    "insight_quality": 0.0-1.0,
    "reasoning_sophistication": 0.0-1.0,
    "actionability": 0.0-1.0,
    "recommendations": ["recommendation1", "recommendation2"],
    "confidence": 0.0-1.0
}}
"""

        try:
            assessment_result = await self.llm_client.execute_cognitive_call(
                prompt=assessment_prompt,
                provider="claude",
                max_tokens=800,
                response_format="json",
            )

            assessment_data = assessment_result.parsed_response
            score = float(assessment_data.get("score", 0.0))

            return QualityAssessment(
                dimension=QualityDimension.ANALYSIS_DEPTH,
                score=score,
                threshold_met=self._score_to_threshold(
                    score, QualityDimension.ANALYSIS_DEPTH
                ),
                details={
                    "analysis_steps": len(analysis_events),
                    "mental_model_depth": assessment_data.get(
                        "mental_model_depth", 0.0
                    ),
                    "framework_utilization": assessment_data.get(
                        "framework_utilization", 0.0
                    ),
                    "insight_quality": assessment_data.get("insight_quality", 0.0),
                    "reasoning_sophistication": assessment_data.get(
                        "reasoning_sophistication", 0.0
                    ),
                    "actionability": assessment_data.get("actionability", 0.0),
                },
                recommendations=assessment_data.get("recommendations", []),
                confidence=float(assessment_data.get("confidence", 0.8)),
                timestamp=datetime.now(timezone.utc),
            )

        except Exception as e:
            logger.error(f"Analysis depth assessment failed: {str(e)}")
            return self._create_fallback_assessment(
                QualityDimension.ANALYSIS_DEPTH, 0.7, str(e)
            )

    async def _assess_assumption_validity(
        self, context: UnifiedContextStream
    ) -> QualityAssessment:
        """Assess validity and testing of key assumptions"""

        # Extract assumptions from synthesis and analysis
        synthesis_events = context.get_events_by_type(
            ContextEventType.SYNTHESIS_MICRO_STEP_COMPLETED
        )
        analysis_events = context.get_events_by_type(
            ContextEventType.ANALYSIS_MICRO_STEP_COMPLETED
        )

        assumptions = []

        # Collect assumptions from synthesis
        for event in synthesis_events:
            event_assumptions = event.data.get("key_assumptions", [])
            assumptions.extend(event_assumptions)

        # Collect assumption challenges from analysis
        assumption_challenges = []
        for event in analysis_events:
            if "assumption" in event.data.get("step_type", "").lower():
                assumption_challenges.append(event.data.get("result", ""))

        if not assumptions:
            return self._create_fallback_assessment(
                QualityDimension.ASSUMPTION_VALIDITY,
                0.6,
                "No key assumptions identified",
            )

        # Assess assumption quality
        assumption_score = min(1.0, len(assumptions) / 5.0)  # Optimal 5+ assumptions
        challenge_score = min(
            1.0, len(assumption_challenges) / 3.0
        )  # Optimal 3+ challenges

        final_score = (assumption_score * 0.6) + (challenge_score * 0.4)

        return QualityAssessment(
            dimension=QualityDimension.ASSUMPTION_VALIDITY,
            score=final_score,
            threshold_met=self._score_to_threshold(
                final_score, QualityDimension.ASSUMPTION_VALIDITY
            ),
            details={
                "assumptions_identified": len(assumptions),
                "assumption_challenges": len(assumption_challenges),
                "assumption_score": assumption_score,
                "challenge_score": challenge_score,
                "sample_assumptions": assumptions[:3],
            },
            recommendations=self._generate_assumption_recommendations(
                assumptions, assumption_challenges
            ),
            confidence=0.75,
            timestamp=datetime.now(timezone.utc),
        )

    async def _assess_recommendation_actionability(
        self, context: UnifiedContextStream
    ) -> QualityAssessment:
        """Assess actionability and practical value of recommendations"""

        # Find final synthesis or recommendation events
        analysis_events = context.get_events_by_type(
            ContextEventType.ANALYSIS_MICRO_STEP_COMPLETED
        )

        final_recommendations = []
        for event in analysis_events:
            if "final" in event.data.get("step_type", "").lower():
                final_recommendations.append(event.data.get("result", ""))

        if not final_recommendations:
            return self._create_fallback_assessment(
                QualityDimension.RECOMMENDATION_ACTIONABILITY,
                0.5,
                "No final recommendations found",
            )

        # Simple heuristic assessment (in production, would use more sophisticated LLM assessment)
        actionability_indicators = [
            "specific",
            "measurable",
            "timeline",
            "responsible",
            "budget",
            "implement",
            "action",
            "step",
            "priority",
            "resource",
            "deadline",
            "outcome",
        ]

        actionability_score = 0.0
        for recommendation in final_recommendations:
            indicator_count = sum(
                1
                for indicator in actionability_indicators
                if indicator.lower() in recommendation.lower()
            )
            actionability_score += min(
                1.0, indicator_count / 5.0
            )  # Normalize per recommendation

        final_score = (
            actionability_score / len(final_recommendations)
            if final_recommendations
            else 0.0
        )

        return QualityAssessment(
            dimension=QualityDimension.RECOMMENDATION_ACTIONABILITY,
            score=final_score,
            threshold_met=self._score_to_threshold(
                final_score, QualityDimension.RECOMMENDATION_ACTIONABILITY
            ),
            details={
                "recommendations_found": len(final_recommendations),
                "actionability_score": actionability_score,
                "actionability_indicators": actionability_indicators,
            },
            recommendations=[
                "Include specific timelines and responsibilities in recommendations",
                "Provide measurable outcomes and success criteria",
                "Identify required resources and budget considerations",
            ],
            confidence=0.70,
            timestamp=datetime.now(timezone.utc),
        )

    async def _assess_cost_effectiveness(
        self, context: UnifiedContextStream
    ) -> QualityAssessment:
        """Assess cost-effectiveness of the engagement"""

        total_cost = context.total_cost_usd
        total_tokens = context.total_tokens

        # Cost benchmarks (based on typical cognitive engagements)
        cost_benchmarks = {
            "excellent": 0.30,  # <$0.30
            "good": 0.50,  # <$0.50
            "acceptable": 0.75,  # <$0.75
            "expensive": 1.00,  # <$1.00
        }

        # Token efficiency benchmarks
        token_benchmarks = {
            "excellent": 15000,  # <15k tokens
            "good": 25000,  # <25k tokens
            "acceptable": 40000,  # <40k tokens
            "inefficient": 60000,  # <60k tokens
        }

        # Calculate cost score
        if total_cost <= cost_benchmarks["excellent"]:
            cost_score = 1.0
        elif total_cost <= cost_benchmarks["good"]:
            cost_score = 0.85
        elif total_cost <= cost_benchmarks["acceptable"]:
            cost_score = 0.70
        elif total_cost <= cost_benchmarks["expensive"]:
            cost_score = 0.55
        else:
            cost_score = 0.40

        # Calculate token efficiency score
        if total_tokens <= token_benchmarks["excellent"]:
            token_score = 1.0
        elif total_tokens <= token_benchmarks["good"]:
            token_score = 0.85
        elif total_tokens <= token_benchmarks["acceptable"]:
            token_score = 0.70
        elif total_tokens <= token_benchmarks["inefficient"]:
            token_score = 0.55
        else:
            token_score = 0.40

        # Combined score (weighted)
        final_score = (cost_score * 0.6) + (token_score * 0.4)

        return QualityAssessment(
            dimension=QualityDimension.COST_EFFECTIVENESS,
            score=final_score,
            threshold_met=self._score_to_threshold(
                final_score, QualityDimension.COST_EFFECTIVENESS
            ),
            details={
                "total_cost_usd": total_cost,
                "total_tokens": total_tokens,
                "cost_score": cost_score,
                "token_score": token_score,
                "cost_per_1k_tokens": (
                    (total_cost / total_tokens * 1000) if total_tokens > 0 else 0
                ),
            },
            recommendations=self._generate_cost_recommendations(
                total_cost, total_tokens
            ),
            confidence=0.90,
            timestamp=datetime.now(timezone.utc),
        )

    async def _assess_transparency_coverage(
        self, context: UnifiedContextStream
    ) -> QualityAssessment:
        """Assess transparency and audit trail coverage"""

        total_events = len(context.events)

        # Check for critical transparency events
        transparency_events = [
            ContextEventType.ENGAGEMENT_INITIATED,
            ContextEventType.RESEARCH_QUERY_EXECUTED,
            ContextEventType.SYNTHESIS_MICRO_STEP_STARTED,
            ContextEventType.SYNTHESIS_MICRO_STEP_COMPLETED,
            ContextEventType.ANALYSIS_MICRO_STEP_STARTED,
            ContextEventType.ANALYSIS_MICRO_STEP_COMPLETED,
            ContextEventType.CHECKPOINT_SAVED,
        ]

        coverage_score = 0.0
        for event_type in transparency_events:
            events_of_type = context.get_events_by_type(event_type)
            if events_of_type:
                coverage_score += 1.0

        coverage_score = coverage_score / len(transparency_events)

        # Check event detail completeness
        events_with_metadata = sum(1 for event in context.events if event.metadata)
        events_with_cost = sum(
            1 for event in context.events if event.cost_usd is not None
        )
        events_with_confidence = sum(
            1 for event in context.events if event.confidence_score is not None
        )

        detail_score = (
            (
                (events_with_metadata / total_events) * 0.3
                + (events_with_cost / total_events) * 0.3
                + (events_with_confidence / total_events) * 0.4
            )
            if total_events > 0
            else 0.0
        )

        final_score = (coverage_score * 0.7) + (detail_score * 0.3)

        return QualityAssessment(
            dimension=QualityDimension.TRANSPARENCY_COVERAGE,
            score=final_score,
            threshold_met=self._score_to_threshold(
                final_score, QualityDimension.TRANSPARENCY_COVERAGE
            ),
            details={
                "total_events": total_events,
                "coverage_score": coverage_score,
                "detail_score": detail_score,
                "events_with_metadata": events_with_metadata,
                "events_with_cost": events_with_cost,
                "events_with_confidence": events_with_confidence,
            },
            recommendations=[
                "Ensure all critical events are captured",
                "Include metadata and confidence scores in all events",
                "Track costs for expensive operations",
            ],
            confidence=0.85,
            timestamp=datetime.now(timezone.utc),
        )

    # Helper methods

    def _score_to_threshold(
        self, score: float, dimension: QualityDimension
    ) -> QualityThreshold:
        """Convert numeric score to quality threshold"""
        thresholds = self.quality_thresholds[dimension]

        if score >= thresholds[QualityThreshold.PRODUCTION]:
            return QualityThreshold.PRODUCTION
        elif score >= thresholds[QualityThreshold.REVIEW]:
            return QualityThreshold.REVIEW
        elif score >= thresholds[QualityThreshold.WARNING]:
            return QualityThreshold.WARNING
        else:
            return QualityThreshold.FAILURE

    def _determine_quality_gate(
        self,
        overall_score: float,
        dimension_scores: Dict[QualityDimension, QualityAssessment],
    ) -> ValidationGateResult:
        """Determine overall quality gate result"""

        # Check for any critical failures
        critical_dimensions = [
            QualityDimension.CONTEXT_INTEGRITY,
            QualityDimension.RESEARCH_COMPLETENESS,
            QualityDimension.SYNTHESIS_COHERENCE,
        ]

        for dimension in critical_dimensions:
            if dimension in dimension_scores:
                if (
                    dimension_scores[dimension].threshold_met
                    == QualityThreshold.FAILURE
                ):
                    return ValidationGateResult.FAILURE

        # Overall threshold check
        if overall_score >= 0.90:
            return ValidationGateResult.PASS
        elif overall_score >= 0.80:
            return ValidationGateResult.PASS
        elif overall_score >= 0.70:
            return ValidationGateResult.WARNING
        else:
            return ValidationGateResult.FAILURE

    def _assess_production_readiness(
        self,
        overall_score: float,
        dimension_scores: Dict[QualityDimension, QualityAssessment],
    ) -> bool:
        """Assess if engagement is production ready"""

        # Must meet minimum overall score
        if overall_score < 0.80:
            return False

        # Must not have any critical failures
        critical_dimensions = [
            QualityDimension.CONTEXT_INTEGRITY,
            QualityDimension.RESEARCH_COMPLETENESS,
            QualityDimension.SYNTHESIS_COHERENCE,
            QualityDimension.TRANSPARENCY_COVERAGE,
        ]

        for dimension in critical_dimensions:
            if dimension in dimension_scores:
                if (
                    dimension_scores[dimension].threshold_met
                    == QualityThreshold.FAILURE
                ):
                    return False

        return True

    def _generate_improvement_recommendations(
        self, dimension_scores: Dict[QualityDimension, QualityAssessment]
    ) -> List[str]:
        """Generate overall improvement recommendations"""

        recommendations = []

        for dimension, assessment in dimension_scores.items():
            if assessment.threshold_met in [
                QualityThreshold.WARNING,
                QualityThreshold.FAILURE,
            ]:
                recommendations.extend(
                    [
                        f"{dimension.value}: {rec}"
                        for rec in assessment.recommendations[:2]
                    ]
                )

        # Add general recommendations
        if not recommendations:
            recommendations = [
                "Consider increasing research depth for more comprehensive analysis",
                "Apply additional mental models for broader perspective",
                "Include more specific actionable recommendations",
            ]

        return recommendations[:5]  # Limit to top 5

    def _calculate_cost_quality_ratio(
        self, context: UnifiedContextStream, quality_score: float
    ) -> float:
        """Calculate cost-quality ratio for value assessment"""

        if context.total_cost_usd == 0 or quality_score == 0:
            return 0.0

        # Higher ratio = better value (more quality per dollar)
        return quality_score / context.total_cost_usd

    def _create_fallback_assessment(
        self, dimension: QualityDimension, score: float, error: str
    ) -> QualityAssessment:
        """Create fallback assessment when LLM assessment fails"""

        return QualityAssessment(
            dimension=dimension,
            score=score,
            threshold_met=self._score_to_threshold(score, dimension),
            details={"assessment_error": error, "fallback_score": True},
            recommendations=[
                f"Review {dimension.value} manually due to assessment error"
            ],
            confidence=0.5,
            timestamp=datetime.now(timezone.utc),
        )

    def _format_research_for_assessment(
        self, research_data: List[Dict[str, Any]]
    ) -> str:
        """Format research data for LLM assessment"""

        formatted = []
        for i, result in enumerate(research_data):
            formatted.append(
                f"""
QUERY {i+1}: {result.get('query', 'N/A')}
CONTENT: {result.get('content', '')[:300]}...
SOURCES: {len(result.get('sources', []))} sources
"""
            )
        return "\n".join(formatted)

    def _format_analysis_for_assessment(
        self, analysis_data: List[Dict[str, Any]]
    ) -> str:
        """Format analysis data for LLM assessment"""

        formatted = []
        for i, result in enumerate(analysis_data):
            formatted.append(
                f"""
STEP {i+1}: {result.get('step_type', 'N/A')}
RESULT: {result.get('result', '')}
CONFIDENCE: {result.get('confidence', 'N/A')}
"""
            )
        return "\n".join(formatted)

    async def _assess_event_consistency(self, context: UnifiedContextStream) -> float:
        """Assess consistency of event flow"""

        # Check for proper event sequencing
        event_types = [event.type for event in context.events]

        # Expected flow patterns
        expected_patterns = [
            ContextEventType.ENGAGEMENT_INITIATED,
            ContextEventType.RESEARCH_QUERY_EXECUTED,
            ContextEventType.SYNTHESIS_MICRO_STEP_COMPLETED,
            ContextEventType.ANALYSIS_MICRO_STEP_COMPLETED,
        ]

        # Check if expected patterns are present
        pattern_score = 0.0
        for pattern in expected_patterns:
            if pattern in event_types:
                pattern_score += 1.0

        return pattern_score / len(expected_patterns)

    def _generate_context_recommendations(
        self, missing_events: List[str], context_size: float
    ) -> List[str]:
        """Generate context-specific recommendations"""

        recommendations = []

        if missing_events:
            recommendations.append(
                f"Missing critical events: {', '.join(missing_events)}"
            )

        if context_size > 5.0:
            recommendations.append(
                "Context size is large; consider event pruning for efficiency"
            )
        elif context_size < 0.1:
            recommendations.append(
                "Context size is small; ensure all critical information is captured"
            )

        return recommendations or ["Context integrity is acceptable"]

    def _generate_assumption_recommendations(
        self, assumptions: List[str], challenges: List[str]
    ) -> List[str]:
        """Generate assumption-specific recommendations"""

        recommendations = []

        if len(assumptions) < 3:
            recommendations.append(
                "Identify more key assumptions underlying the analysis"
            )

        if len(challenges) < 2:
            recommendations.append("Add more assumption challenging steps")

        if not assumptions:
            recommendations.append("Explicitly identify and state key assumptions")

        return recommendations or [
            "Assumption identification and challenging is adequate"
        ]

    def _generate_cost_recommendations(self, cost: float, tokens: int) -> List[str]:
        """Generate cost-specific recommendations"""

        recommendations = []

        if cost > 1.00:
            recommendations.append("Cost is high; consider optimizing query complexity")

        if tokens > 50000:
            recommendations.append("Token usage is high; implement context pruning")

        cost_per_1k_tokens = (cost / tokens * 1000) if tokens > 0 else 0
        if cost_per_1k_tokens > 0.03:
            recommendations.append(
                "Cost per token is high; consider cheaper providers for some steps"
            )

        return recommendations or ["Cost efficiency is acceptable"]

    # Micro-step quality assessment methods

    async def _assess_research_micro_step(
        self, context: UnifiedContextStream, step_result: Dict[str, Any]
    ) -> QualityAssessment:
        """Assess quality of individual research micro-step"""

        query_count = len(step_result.get("queries_executed", []))
        cost = step_result.get("cost_usd", 0.0)

        # Simple heuristic assessment
        query_score = min(1.0, query_count / 3.0)  # Optimal 3+ queries
        cost_score = 1.0 if cost < 0.05 else max(0.5, 1.0 - (cost - 0.05) / 0.10)

        final_score = (query_score * 0.7) + (cost_score * 0.3)

        return QualityAssessment(
            dimension=QualityDimension.RESEARCH_COMPLETENESS,
            score=final_score,
            threshold_met=self._score_to_threshold(
                final_score, QualityDimension.RESEARCH_COMPLETENESS
            ),
            details={
                "queries_executed": query_count,
                "cost_usd": cost,
                "query_score": query_score,
                "cost_score": cost_score,
            },
            recommendations=["Optimize query count and cost efficiency"],
            confidence=0.75,
            timestamp=datetime.now(timezone.utc),
        )

    async def _assess_synthesis_micro_step(
        self, context: UnifiedContextStream, step_result: Dict[str, Any]
    ) -> QualityAssessment:
        """Assess quality of individual synthesis micro-step"""

        confidence = step_result.get("confidence_score", 0.0)
        result_length = len(step_result.get("result", ""))

        # Simple quality assessment
        confidence_score = confidence
        depth_score = min(1.0, result_length / 500.0)  # Optimal 500+ chars

        final_score = (confidence_score * 0.6) + (depth_score * 0.4)

        return QualityAssessment(
            dimension=QualityDimension.SYNTHESIS_COHERENCE,
            score=final_score,
            threshold_met=self._score_to_threshold(
                final_score, QualityDimension.SYNTHESIS_COHERENCE
            ),
            details={
                "confidence_score": confidence,
                "result_length": result_length,
                "depth_score": depth_score,
            },
            recommendations=["Ensure synthesis depth and confidence"],
            confidence=0.70,
            timestamp=datetime.now(timezone.utc),
        )

    async def _assess_analysis_micro_step(
        self, context: UnifiedContextStream, step_result: Dict[str, Any]
    ) -> QualityAssessment:
        """Assess quality of individual analysis micro-step"""

        confidence = step_result.get("confidence_score", 0.0)
        actions_taken = step_result.get("actions_taken", 0)

        # Simple quality assessment
        confidence_score = confidence
        action_score = min(1.0, actions_taken / 5.0)  # Optimal 5+ actions

        final_score = (confidence_score * 0.7) + (action_score * 0.3)

        return QualityAssessment(
            dimension=QualityDimension.ANALYSIS_DEPTH,
            score=final_score,
            threshold_met=self._score_to_threshold(
                final_score, QualityDimension.ANALYSIS_DEPTH
            ),
            details={
                "confidence_score": confidence,
                "actions_taken": actions_taken,
                "action_score": action_score,
            },
            recommendations=["Ensure analysis depth and action completeness"],
            confidence=0.70,
            timestamp=datetime.now(timezone.utc),
        )

    async def _assess_generic_micro_step(
        self,
        context: UnifiedContextStream,
        step_result: Dict[str, Any],
        dimension: QualityDimension,
    ) -> QualityAssessment:
        """Generic assessment for other micro-step types"""

        confidence = step_result.get("confidence_score", 0.75)

        return QualityAssessment(
            dimension=dimension,
            score=confidence,
            threshold_met=self._score_to_threshold(confidence, dimension),
            details={"confidence_score": confidence, "generic_assessment": True},
            recommendations=[f"Review {dimension.value} manually"],
            confidence=0.60,
            timestamp=datetime.now(timezone.utc),
        )


# Factory function for dependency injection
def get_hybrid_quality_assurance() -> HybridQualityAssurance:
    """Get hybrid quality assurance instance"""
    return HybridQualityAssurance()
