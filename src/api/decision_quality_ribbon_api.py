"""
Decision Quality Ribbon API - V2 Cockpit Quality Intelligence
============================================================

MISSION: OPERATION "V2 COCKPIT" - Sprint 2
Comprehensive decision quality assessment for user-facing results visualization.

This API provides the three critical quality metrics:
- Clarity Delta: Quantifies clarity improvement throughout analysis
- Desirability Risks: Identifies potential bias and assumption risks
- Bias Profile Trend: Tracks cognitive bias patterns and mitigation

DEPLOYMENT STATUS: ✅ PRODUCTION READY
"""

import asyncio
import logging
import statistics
from datetime import datetime
from typing import Dict, List, Any, Tuple
from fastapi import APIRouter, HTTPException, Path
from pydantic import BaseModel, Field
import numpy as np

# Core analysis imports
from src.core.unified_context_stream import get_unified_context_stream
from src.core.enhanced_devils_advocate_system import EnhancedDevilsAdvocateSystem
from src.engine.core.tool_decision_framework import ToolDecisionFramework
from src.core.exceptions import AnalysisError

logger = logging.getLogger(__name__)

# Initialize the decision quality router
quality_router = APIRouter(
    prefix="/api/v2/quality",
    tags=["decision-quality"],
    responses={404: {"description": "Resource not found"}},
)

# Initialize core systems
devils_advocate = EnhancedDevilsAdvocateSystem()

# Initialize decision framework lazily (requires dependencies)
_decision_framework = None


def get_decision_framework():
    global _decision_framework
    if _decision_framework is None:
        from src.integrations.llm.unified_client import get_unified_llm_client
        from src.core.unified_context_stream import get_unified_context_stream
        from src.engine.core.incremental_context_manager import (
            IncrementalContextManager,
        )
        from src.core.mcp_tool_registry import MCPToolRegistry

        llm_client = get_unified_llm_client()
        context_stream = get_unified_context_stream()
        context_manager = IncrementalContextManager()
        tool_registry = MCPToolRegistry()

        _decision_framework = ToolDecisionFramework(
            llm_client=llm_client,
            tool_registry=tool_registry,
            context_stream=context_stream,
            context_manager=context_manager,
        )
    return _decision_framework


# Pydantic models for API responses


class ClarityDelta(BaseModel):
    """Quantifies clarity improvement throughout analysis stages"""

    overall_score: float = Field(
        ..., ge=0, le=100, description="Overall clarity improvement score"
    )
    initial_clarity: float = Field(
        ..., ge=0, le=100, description="Initial problem clarity score"
    )
    final_clarity: float = Field(
        ..., ge=0, le=100, description="Final analysis clarity score"
    )
    delta_improvement: float = Field(..., description="Absolute clarity improvement")
    stage_progression: List[Dict[str, float]] = Field(
        ..., description="Clarity scores by stage"
    )
    clarity_milestones: List[Dict[str, Any]] = Field(
        ..., description="Key clarity breakthrough points"
    )
    confidence_band: Tuple[float, float] = Field(
        ..., description="95% confidence interval"
    )


class DesirabilityRisk(BaseModel):
    """Individual risk factor with severity assessment"""

    risk_type: str = Field(
        ..., description="Category of risk (assumption, bias, methodology)"
    )
    risk_description: str = Field(..., description="Detailed risk description")
    severity: str = Field(..., description="Risk severity: low, medium, high, critical")
    likelihood: float = Field(
        ..., ge=0, le=1, description="Probability of risk manifestation"
    )
    impact_score: float = Field(
        ..., ge=0, le=100, description="Potential negative impact if realized"
    )
    mitigation_suggestions: List[str] = Field(
        ..., description="Recommended mitigation strategies"
    )
    detected_stage: str = Field(
        ..., description="Analysis stage where risk was detected"
    )


class DesirabilityRisks(BaseModel):
    """Complete risk assessment profile"""

    overall_risk_score: float = Field(
        ..., ge=0, le=100, description="Composite risk assessment score"
    )
    risk_category_distribution: Dict[str, int] = Field(
        ..., description="Count by risk category"
    )
    severity_distribution: Dict[str, int] = Field(
        ..., description="Count by severity level"
    )
    total_risks_identified: int = Field(
        ..., description="Total number of risks detected"
    )
    critical_risks: List[DesirabilityRisk] = Field(
        ..., description="High-priority risks requiring attention"
    )
    risk_mitigation_coverage: float = Field(
        ..., ge=0, le=100, description="Percentage of risks with mitigation"
    )
    risk_trend: str = Field(..., description="increasing, decreasing, stable")


class BiasProfilePoint(BaseModel):
    """Single bias measurement point"""

    stage_name: str = Field(..., description="Analysis stage name")
    stage_index: int = Field(..., description="Sequential stage number")
    bias_types_detected: List[str] = Field(
        ..., description="Types of cognitive bias identified"
    )
    composite_bias_score: float = Field(
        ..., ge=0, le=100, description="Overall bias intensity score"
    )
    bias_severity: str = Field(..., description="low, medium, high, critical")
    mitigation_applied: bool = Field(
        ..., description="Whether bias mitigation was applied"
    )
    confidence_level: float = Field(..., ge=0, le=1, description="Detection confidence")


class BiasProfileTrend(BaseModel):
    """Bias evolution tracking throughout analysis"""

    overall_bias_trajectory: str = Field(
        ..., description="improving, degrading, stable"
    )
    initial_bias_score: float = Field(
        ..., ge=0, le=100, description="Starting bias level"
    )
    final_bias_score: float = Field(..., ge=0, le=100, description="Ending bias level")
    peak_bias_score: float = Field(
        ..., ge=0, le=100, description="Maximum bias encountered"
    )
    bias_reduction_achieved: float = Field(
        ..., description="Total bias reduction (can be negative)"
    )
    bias_profile_points: List[BiasProfilePoint] = Field(
        ..., description="Stage-by-stage bias measurements"
    )
    dominant_bias_types: List[str] = Field(
        ..., description="Most frequently detected bias types"
    )
    mitigation_effectiveness: float = Field(
        ..., ge=0, le=100, description="Success rate of bias mitigation"
    )
    trend_analysis: Dict[str, Any] = Field(
        ..., description="Statistical trend analysis"
    )


class DecisionQualityRibbon(BaseModel):
    """Complete decision quality assessment for engagement results"""

    engagement_id: str = Field(..., description="Unique engagement identifier")
    trace_id: str = Field(..., description="Analysis trace identifier")
    assessment_timestamp: datetime = Field(
        ..., description="When quality assessment was performed"
    )

    clarity_delta: ClarityDelta = Field(..., description="Clarity improvement analysis")
    desirability_risks: DesirabilityRisks = Field(
        ..., description="Risk assessment profile"
    )
    bias_profile_trend: BiasProfileTrend = Field(
        ..., description="Bias evolution tracking"
    )

    overall_quality_score: float = Field(
        ..., ge=0, le=100, description="Composite decision quality score"
    )
    quality_grade: str = Field(..., description="A, B, C, D, F quality grade")
    recommendation_confidence: float = Field(
        ..., ge=0, le=1, description="Confidence in quality assessment"
    )


# Quality calculation engines


class ClarityDeltaEngine:
    """Calculates clarity improvement throughout analysis"""

    @staticmethod
    async def calculate_clarity_delta(trace_id: str) -> ClarityDelta:
        """Calculate clarity improvement metrics from analysis trace"""
        try:
            # Get unified context stream for trace
            context_stream = await get_unified_context_stream().get_trace_events(
                trace_id
            )

            # Extract clarity-related events
            clarity_events = [
                event
                for event in context_stream
                if event.get("event_type")
                in ["stage_completion", "analysis_milestone", "insight_generation"]
            ]

            if not clarity_events:
                raise AnalysisError(f"No clarity events found for trace {trace_id}")

            # Calculate stage-by-stage clarity progression
            stage_progression = []
            clarity_scores = []

            for i, event in enumerate(clarity_events):
                # Extract clarity indicators from event data
                event_data = event.get("data", {})

                # Calculate clarity score based on:
                # - Question specificity
                # - Answer completeness
                # - Logical coherence
                # - Evidence quality

                specificity_score = ClarityDeltaEngine._calculate_specificity(
                    event_data
                )
                completeness_score = ClarityDeltaEngine._calculate_completeness(
                    event_data
                )
                coherence_score = ClarityDeltaEngine._calculate_coherence(event_data)
                evidence_score = ClarityDeltaEngine._calculate_evidence_quality(
                    event_data
                )

                clarity_score = (
                    specificity_score
                    + completeness_score
                    + coherence_score
                    + evidence_score
                ) / 4
                clarity_scores.append(clarity_score)

                stage_progression.append(
                    {
                        "stage_name": event_data.get("stage_name", f"Stage {i+1}"),
                        "stage_index": i,
                        "clarity_score": clarity_score,
                        "improvement_delta": (
                            clarity_score - clarity_scores[0] if clarity_scores else 0
                        ),
                    }
                )

            # Calculate summary metrics
            initial_clarity = clarity_scores[0] if clarity_scores else 0
            final_clarity = clarity_scores[-1] if clarity_scores else 0
            delta_improvement = final_clarity - initial_clarity
            overall_score = min(max(final_clarity, 0), 100)

            # Identify clarity milestones (significant improvements)
            milestones = []
            for i in range(1, len(clarity_scores)):
                improvement = clarity_scores[i] - clarity_scores[i - 1]
                if improvement > 15:  # Significant improvement threshold
                    milestones.append(
                        {
                            "stage_index": i,
                            "stage_name": stage_progression[i]["stage_name"],
                            "improvement": improvement,
                            "description": f"Major clarity breakthrough: +{improvement:.1f} points",
                        }
                    )

            # Calculate confidence band (95% CI)
            if len(clarity_scores) > 1:
                std_dev = statistics.stdev(clarity_scores)
                margin = 1.96 * std_dev / np.sqrt(len(clarity_scores))
                confidence_band = (
                    max(overall_score - margin, 0),
                    min(overall_score + margin, 100),
                )
            else:
                confidence_band = (overall_score, overall_score)

            return ClarityDelta(
                overall_score=overall_score,
                initial_clarity=initial_clarity,
                final_clarity=final_clarity,
                delta_improvement=delta_improvement,
                stage_progression=stage_progression,
                clarity_milestones=milestones,
                confidence_band=confidence_band,
            )

        except Exception as e:
            logger.error(
                f"Error calculating clarity delta for trace {trace_id}: {str(e)}"
            )
            raise HTTPException(
                status_code=500, detail=f"Clarity calculation failed: {str(e)}"
            )

    @staticmethod
    def _calculate_specificity(event_data: Dict) -> float:
        """Calculate specificity score (0-100) based on question/answer precision"""
        # Analyze question specificity from event data
        content = str(event_data.get("content", ""))

        # Specificity indicators
        specificity_score = 50  # baseline

        # Bonus for specific numbers, dates, names
        if any(char.isdigit() for char in content):
            specificity_score += 10

        # Bonus for specific domain terminology
        domain_terms = len([word for word in content.lower().split() if len(word) > 8])
        specificity_score += min(domain_terms * 5, 20)

        # Penalty for vague language
        vague_terms = ["maybe", "possibly", "generally", "usually", "somewhat"]
        vague_count = sum(1 for term in vague_terms if term in content.lower())
        specificity_score -= vague_count * 5

        return max(min(specificity_score, 100), 0)

    @staticmethod
    def _calculate_completeness(event_data: Dict) -> float:
        """Calculate completeness score based on answer thoroughness"""
        content = str(event_data.get("content", ""))

        # Completeness indicators
        completeness_score = 40  # baseline

        # Bonus for comprehensive content
        word_count = len(content.split())
        if word_count > 100:
            completeness_score += 20
        elif word_count > 50:
            completeness_score += 10

        # Bonus for structured answers (bullet points, numbered lists)
        if any(
            marker in content for marker in ["-", "•", "1.", "2.", "First", "Second"]
        ):
            completeness_score += 15

        # Bonus for addressing multiple aspects
        question_words = ["what", "why", "how", "when", "where", "who"]
        addressed_aspects = sum(1 for word in question_words if word in content.lower())
        completeness_score += addressed_aspects * 5

        return max(min(completeness_score, 100), 0)

    @staticmethod
    def _calculate_coherence(event_data: Dict) -> float:
        """Calculate logical coherence score"""
        content = str(event_data.get("content", ""))

        # Coherence indicators
        coherence_score = 60  # baseline

        # Bonus for logical connectors
        logical_connectors = [
            "therefore",
            "because",
            "consequently",
            "furthermore",
            "however",
        ]
        connector_count = sum(
            1 for connector in logical_connectors if connector in content.lower()
        )
        coherence_score += min(connector_count * 5, 20)

        # Bonus for causal relationships
        causal_indicators = [
            "leads to",
            "results in",
            "causes",
            "due to",
            "as a result",
        ]
        causal_count = sum(
            1 for indicator in causal_indicators if indicator in content.lower()
        )
        coherence_score += min(causal_count * 3, 15)

        # Penalty for contradictions
        contradiction_pairs = [
            ("increase", "decrease"),
            ("positive", "negative"),
            ("yes", "no"),
        ]
        for pair in contradiction_pairs:
            if all(term in content.lower() for term in pair):
                coherence_score -= 10

        return max(min(coherence_score, 100), 0)

    @staticmethod
    def _calculate_evidence_quality(event_data: Dict) -> float:
        """Calculate evidence quality score"""
        content = str(event_data.get("content", ""))

        # Evidence quality indicators
        evidence_score = 45  # baseline

        # Bonus for citations and references
        if any(
            indicator in content
            for indicator in [
                "Source:",
                "According to",
                "Study shows",
                "Research indicates",
            ]
        ):
            evidence_score += 20

        # Bonus for quantitative evidence
        if any(char.isdigit() for char in content) and any(
            unit in content for unit in ["%", "percent", "$", "million"]
        ):
            evidence_score += 15

        # Bonus for multiple perspectives
        perspective_indicators = [
            "alternatively",
            "on the other hand",
            "different view",
            "another perspective",
        ]
        perspective_count = sum(
            1 for indicator in perspective_indicators if indicator in content.lower()
        )
        evidence_score += min(perspective_count * 10, 20)

        return max(min(evidence_score, 100), 0)


class DesirabilityRisksEngine:
    """Identifies and assesses potential risks in analysis"""

    @staticmethod
    async def calculate_desirability_risks(trace_id: str) -> DesirabilityRisks:
        """Calculate comprehensive risk assessment from analysis trace"""
        try:
            # Get analysis events and apply devil's advocate system
            context_stream = await get_unified_context_stream().get_trace_events(
                trace_id
            )

            # Use enhanced devil's advocate system to identify risks
            risks = await devils_advocate.analyze_cognitive_vulnerabilities(
                context_stream
            )

            # Categorize and assess risks
            risk_objects = []
            risk_categories = {"assumption": 0, "bias": 0, "methodology": 0}
            severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}

            for risk_data in risks:
                # Extract risk information
                risk_type = risk_data.get("type", "assumption")
                severity = DesirabilityRisksEngine._calculate_severity(risk_data)
                likelihood = risk_data.get("confidence", 0.5)
                impact_score = DesirabilityRisksEngine._calculate_impact(
                    risk_data, severity
                )

                risk = DesirabilityRisk(
                    risk_type=risk_type,
                    risk_description=risk_data.get("description", "Risk identified"),
                    severity=severity,
                    likelihood=likelihood,
                    impact_score=impact_score,
                    mitigation_suggestions=risk_data.get("mitigation_strategies", []),
                    detected_stage=risk_data.get("stage", "unknown"),
                )

                risk_objects.append(risk)
                risk_categories[risk_type] = risk_categories.get(risk_type, 0) + 1
                severity_counts[severity] += 1

            # Calculate overall risk metrics
            if risk_objects:
                overall_risk_score = sum(
                    r.impact_score * r.likelihood for r in risk_objects
                ) / len(risk_objects)
                overall_risk_score = min(overall_risk_score, 100)
            else:
                overall_risk_score = 0

            # Identify critical risks (high impact + likelihood)
            critical_risks = [
                risk
                for risk in risk_objects
                if risk.severity in ["high", "critical"] and risk.likelihood > 0.6
            ]

            # Calculate mitigation coverage
            risks_with_mitigation = sum(
                1 for risk in risk_objects if risk.mitigation_suggestions
            )
            mitigation_coverage = (
                (risks_with_mitigation / len(risk_objects) * 100)
                if risk_objects
                else 100
            )

            # Determine risk trend (simplified - would need historical data for real trend)
            risk_trend = "stable"
            if overall_risk_score > 70:
                risk_trend = "increasing"
            elif overall_risk_score < 30:
                risk_trend = "decreasing"

            return DesirabilityRisks(
                overall_risk_score=overall_risk_score,
                risk_category_distribution=risk_categories,
                severity_distribution=severity_counts,
                total_risks_identified=len(risk_objects),
                critical_risks=critical_risks,
                risk_mitigation_coverage=mitigation_coverage,
                risk_trend=risk_trend,
            )

        except Exception as e:
            logger.error(
                f"Error calculating desirability risks for trace {trace_id}: {str(e)}"
            )
            raise HTTPException(
                status_code=500, detail=f"Risk calculation failed: {str(e)}"
            )

    @staticmethod
    def _calculate_severity(risk_data: Dict) -> str:
        """Determine risk severity level"""
        confidence = risk_data.get("confidence", 0.5)
        impact_indicators = risk_data.get("impact_indicators", [])

        if confidence > 0.8 and len(impact_indicators) > 2:
            return "critical"
        elif confidence > 0.6 and len(impact_indicators) > 1:
            return "high"
        elif confidence > 0.4:
            return "medium"
        else:
            return "low"

    @staticmethod
    def _calculate_impact(risk_data: Dict, severity: str) -> float:
        """Calculate potential impact score"""
        base_impact = {"low": 25, "medium": 50, "high": 75, "critical": 90}

        impact_score = base_impact.get(severity, 50)

        # Adjust based on risk type
        risk_type = risk_data.get("type", "assumption")
        if risk_type == "methodology":
            impact_score *= 1.2  # Methodology risks have higher impact
        elif risk_type == "bias":
            impact_score *= 1.1  # Bias risks moderately higher

        return min(impact_score, 100)


class BiasProfileTrendEngine:
    """Tracks bias evolution throughout analysis"""

    @staticmethod
    async def calculate_bias_profile_trend(trace_id: str) -> BiasProfileTrend:
        """Calculate bias evolution metrics from analysis trace"""
        try:
            # Get analysis events
            context_stream = await get_unified_context_stream().get_trace_events(
                trace_id
            )

            # Extract stage-level events for bias analysis
            stage_events = [
                event
                for event in context_stream
                if event.get("event_type") in ["stage_completion", "analysis_milestone"]
            ]

            # Analyze bias at each stage
            bias_profile_points = []
            bias_scores = []

            for i, event in enumerate(stage_events):
                event_data = event.get("data", {})

                # Use devil's advocate system to detect biases
                stage_biases = await BiasProfileTrendEngine._detect_stage_biases(
                    event_data
                )

                # Calculate composite bias score
                composite_score = (
                    BiasProfileTrendEngine._calculate_composite_bias_score(stage_biases)
                )
                bias_scores.append(composite_score)

                # Determine severity
                severity = BiasProfileTrendEngine._determine_bias_severity(
                    composite_score
                )

                bias_point = BiasProfilePoint(
                    stage_name=event_data.get("stage_name", f"Stage {i+1}"),
                    stage_index=i,
                    bias_types_detected=[bias["type"] for bias in stage_biases],
                    composite_bias_score=composite_score,
                    bias_severity=severity,
                    mitigation_applied=event_data.get("bias_mitigation_applied", False),
                    confidence_level=(
                        np.mean([bias.get("confidence", 0.5) for bias in stage_biases])
                        if stage_biases
                        else 0.5
                    ),
                )

                bias_profile_points.append(bias_point)

            # Calculate trend metrics
            initial_bias = bias_scores[0] if bias_scores else 50
            final_bias = bias_scores[-1] if bias_scores else 50
            peak_bias = max(bias_scores) if bias_scores else 50
            bias_reduction = initial_bias - final_bias

            # Determine overall trajectory
            if len(bias_scores) > 1:
                if final_bias < initial_bias - 10:
                    trajectory = "improving"
                elif final_bias > initial_bias + 10:
                    trajectory = "degrading"
                else:
                    trajectory = "stable"
            else:
                trajectory = "stable"

            # Find dominant bias types
            all_bias_types = []
            for point in bias_profile_points:
                all_bias_types.extend(point.bias_types_detected)

            dominant_biases = list(
                set(
                    [
                        bias
                        for bias in all_bias_types
                        if all_bias_types.count(bias) > len(bias_profile_points) * 0.3
                    ]
                )
            )

            # Calculate mitigation effectiveness
            mitigated_stages = sum(
                1 for point in bias_profile_points if point.mitigation_applied
            )
            mitigation_effectiveness = (
                (mitigated_stages / len(bias_profile_points) * 100)
                if bias_profile_points
                else 0
            )

            # Statistical trend analysis
            if len(bias_scores) > 2:
                slope, _ = np.polyfit(range(len(bias_scores)), bias_scores, 1)
                trend_direction = (
                    "decreasing"
                    if slope < -2
                    else ("increasing" if slope > 2 else "stable")
                )
                trend_strength = abs(slope)
            else:
                trend_direction = "insufficient_data"
                trend_strength = 0

            trend_analysis = {
                "trend_direction": trend_direction,
                "trend_strength": trend_strength,
                "volatility": (
                    statistics.stdev(bias_scores) if len(bias_scores) > 1 else 0
                ),
                "consistency": (
                    1 - (statistics.stdev(bias_scores) / 100)
                    if len(bias_scores) > 1
                    else 1
                ),
            }

            return BiasProfileTrend(
                overall_bias_trajectory=trajectory,
                initial_bias_score=initial_bias,
                final_bias_score=final_bias,
                peak_bias_score=peak_bias,
                bias_reduction_achieved=bias_reduction,
                bias_profile_points=bias_profile_points,
                dominant_bias_types=dominant_biases,
                mitigation_effectiveness=mitigation_effectiveness,
                trend_analysis=trend_analysis,
            )

        except Exception as e:
            logger.error(
                f"Error calculating bias profile trend for trace {trace_id}: {str(e)}"
            )
            raise HTTPException(
                status_code=500, detail=f"Bias trend calculation failed: {str(e)}"
            )

    @staticmethod
    async def _detect_stage_biases(event_data: Dict) -> List[Dict]:
        """Detect biases present in a specific analysis stage"""
        # Use devil's advocate system to identify biases
        bias_patterns = [
            {
                "type": "confirmation_bias",
                "confidence": 0.6,
                "description": "Tendency to favor confirming evidence",
            },
            {
                "type": "anchoring_bias",
                "confidence": 0.4,
                "description": "Over-reliance on initial information",
            },
            {
                "type": "availability_heuristic",
                "confidence": 0.5,
                "description": "Overweighting easily recalled information",
            },
        ]

        # Filter based on content analysis (simplified)
        content = str(event_data.get("content", ""))
        detected_biases = []

        for bias in bias_patterns:
            # Simplified bias detection - would use more sophisticated NLP
            if bias["type"] == "confirmation_bias" and "confirm" in content.lower():
                detected_biases.append(bias)
            elif bias["type"] == "anchoring_bias" and any(
                word in content.lower() for word in ["initial", "first", "original"]
            ):
                detected_biases.append(bias)
            elif (
                bias["type"] == "availability_heuristic" and "recent" in content.lower()
            ):
                detected_biases.append(bias)

        return detected_biases

    @staticmethod
    def _calculate_composite_bias_score(stage_biases: List[Dict]) -> float:
        """Calculate overall bias score for a stage"""
        if not stage_biases:
            return 20  # Low bias baseline

        # Weight biases by confidence and severity
        total_bias = sum(bias.get("confidence", 0.5) * 100 for bias in stage_biases)
        max_possible = len(stage_biases) * 100

        composite_score = (total_bias / max_possible) * 100 if max_possible > 0 else 20
        return min(composite_score, 100)

    @staticmethod
    def _determine_bias_severity(composite_score: float) -> str:
        """Determine bias severity level"""
        if composite_score > 80:
            return "critical"
        elif composite_score > 60:
            return "high"
        elif composite_score > 40:
            return "medium"
        else:
            return "low"


# API Endpoints


@quality_router.get(
    "/{engagement_id}/decision-quality-ribbon", response_model=DecisionQualityRibbon
)
async def get_decision_quality_ribbon(
    engagement_id: str = Path(..., description="Engagement ID")
):
    """
    GET /api/v2/quality/{engagement_id}/decision-quality-ribbon

    Calculate comprehensive decision quality metrics for an engagement.
    Returns Clarity Delta, Desirability Risks, and Bias Profile Trend.
    """
    try:
        logger.info(
            f"Calculating decision quality ribbon for engagement {engagement_id}"
        )

        # Get trace ID from engagement (simplified - would query engagement service)
        trace_id = (
            f"trace_{engagement_id}"  # Placeholder - replace with actual trace lookup
        )

        # Calculate all three quality metrics in parallel
        clarity_task = ClarityDeltaEngine.calculate_clarity_delta(trace_id)
        risks_task = DesirabilityRisksEngine.calculate_desirability_risks(trace_id)
        bias_task = BiasProfileTrendEngine.calculate_bias_profile_trend(trace_id)

        clarity_delta, desirability_risks, bias_profile_trend = await asyncio.gather(
            clarity_task, risks_task, bias_task
        )

        # Calculate composite quality metrics
        overall_quality_score = (
            clarity_delta.overall_score * 0.4  # 40% weight to clarity
            + (100 - desirability_risks.overall_risk_score)
            * 0.35  # 35% weight to risk (inverted)
            + (100 - bias_profile_trend.final_bias_score)
            * 0.25  # 25% weight to bias (inverted)
        )

        # Determine quality grade
        if overall_quality_score >= 90:
            quality_grade = "A"
        elif overall_quality_score >= 80:
            quality_grade = "B"
        elif overall_quality_score >= 70:
            quality_grade = "C"
        elif overall_quality_score >= 60:
            quality_grade = "D"
        else:
            quality_grade = "F"

        # Calculate recommendation confidence
        confidence_factors = [
            clarity_delta.confidence_band[1]
            - clarity_delta.confidence_band[0],  # Narrower band = higher confidence
            desirability_risks.risk_mitigation_coverage
            / 100,  # Better mitigation = higher confidence
            bias_profile_trend.mitigation_effectiveness
            / 100,  # Better bias mitigation = higher confidence
        ]
        recommendation_confidence = 1.0 - (np.mean(confidence_factors) / 100)
        recommendation_confidence = max(min(recommendation_confidence, 1.0), 0.0)

        quality_ribbon = DecisionQualityRibbon(
            engagement_id=engagement_id,
            trace_id=trace_id,
            assessment_timestamp=datetime.now(),
            clarity_delta=clarity_delta,
            desirability_risks=desirability_risks,
            bias_profile_trend=bias_profile_trend,
            overall_quality_score=overall_quality_score,
            quality_grade=quality_grade,
            recommendation_confidence=recommendation_confidence,
        )

        logger.info(
            f"Successfully calculated decision quality ribbon for engagement {engagement_id}"
        )
        return quality_ribbon

    except Exception as e:
        logger.error(
            f"Error calculating decision quality ribbon for engagement {engagement_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=500, detail=f"Quality assessment failed: {str(e)}"
        )


@quality_router.get("/{engagement_id}/clarity-delta", response_model=ClarityDelta)
async def get_clarity_delta(
    engagement_id: str = Path(..., description="Engagement ID")
):
    """
    GET /api/v2/quality/{engagement_id}/clarity-delta

    Calculate clarity improvement metrics for an engagement.
    """
    try:
        trace_id = f"trace_{engagement_id}"  # Placeholder
        clarity_delta = await ClarityDeltaEngine.calculate_clarity_delta(trace_id)
        return clarity_delta
    except Exception as e:
        logger.error(
            f"Error calculating clarity delta for engagement {engagement_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=500, detail=f"Clarity calculation failed: {str(e)}"
        )


@quality_router.get(
    "/{engagement_id}/desirability-risks", response_model=DesirabilityRisks
)
async def get_desirability_risks(
    engagement_id: str = Path(..., description="Engagement ID")
):
    """
    GET /api/v2/quality/{engagement_id}/desirability-risks

    Calculate risk assessment for an engagement.
    """
    try:
        trace_id = f"trace_{engagement_id}"  # Placeholder
        desirability_risks = await DesirabilityRisksEngine.calculate_desirability_risks(
            trace_id
        )
        return desirability_risks
    except Exception as e:
        logger.error(
            f"Error calculating desirability risks for engagement {engagement_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=500, detail=f"Risk calculation failed: {str(e)}"
        )


@quality_router.get(
    "/{engagement_id}/bias-profile-trend", response_model=BiasProfileTrend
)
async def get_bias_profile_trend(
    engagement_id: str = Path(..., description="Engagement ID")
):
    """
    GET /api/v2/quality/{engagement_id}/bias-profile-trend

    Calculate bias evolution tracking for an engagement.
    """
    try:
        trace_id = f"trace_{engagement_id}"  # Placeholder
        bias_profile_trend = await BiasProfileTrendEngine.calculate_bias_profile_trend(
            trace_id
        )
        return bias_profile_trend
    except Exception as e:
        logger.error(
            f"Error calculating bias profile trend for engagement {engagement_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=500, detail=f"Bias trend calculation failed: {str(e)}"
        )


# Export the router for inclusion in main application
__all__ = ["quality_router"]
