"""
Chunking Quality Monitor - Real-Time Assessment & Adaptive Re-chunking
=====================================================================

Real-time quality assessment based on research-validated metrics:
1. Coverage (collectively exhaustive test)
2. Overlap (mutually exclusive test)
3. Causal completeness (can we trace all cause-effect chains?)
4. Information density (efficiency - information value per token)
5. Time-to-insight (speed vs accuracy trade-off)
6. Constraint consistency (hard constraints preserved?)

Implements adaptive re-chunking based on decision theory principles.
"""

import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging

from src.core.strategic_query_decomposer import (
    MECEDecomposition,
)
from src.core.boundary_detection_engine import (
    get_boundary_detection_engine,
)
from src.integrations.llm.unified_client import UnifiedLLMClient

logger = logging.getLogger(__name__)


class QualityMetric(Enum):
    """Types of quality metrics for chunking assessment"""

    COVERAGE = "coverage"  # Collectively exhaustive
    OVERLAP = "overlap"  # Mutually exclusive
    CAUSAL_COMPLETENESS = "causal_completeness"  # Causal chain traceability
    INFORMATION_DENSITY = "information_density"  # Information value per token
    TIME_TO_INSIGHT = "time_to_insight"  # Speed vs accuracy
    CONSTRAINT_CONSISTENCY = "constraint_consistency"  # Hard constraints preserved
    BOUNDARY_QUALITY = "boundary_quality"  # Natural boundary detection


class ReChunkingTrigger(Enum):
    """Triggers that indicate re-chunking may be beneficial"""

    LOW_COVERAGE = "low_coverage"  # Missing important components
    HIGH_OVERLAP = "high_overlap"  # Components not mutually exclusive
    POOR_BOUNDARIES = "poor_boundaries"  # Weak or missing boundaries
    LOW_EFFICIENCY = "low_efficiency"  # Poor information density
    COMPLETENESS_GAPS = "completeness_gaps"  # Causal chains incomplete
    CONSTRAINT_VIOLATIONS = "constraint_violations"  # Hard constraints violated


@dataclass
class QualityAssessment:
    """Complete quality assessment of a decomposition"""

    decomposition_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Individual metric scores (0.0-1.0)
    coverage_score: float = 0.0
    overlap_score: float = 0.0  # Higher is better (less overlap)
    causal_completeness_score: float = 0.0
    information_density_score: float = 0.0
    time_to_insight_score: float = 0.0
    constraint_consistency_score: float = 0.0
    boundary_quality_score: float = 0.0

    # Overall quality score
    overall_quality_score: float = 0.0

    # Detailed analysis
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    improvement_recommendations: List[str] = field(default_factory=list)

    # Re-chunking analysis
    should_rechunk: bool = False
    rechunk_triggers: List[ReChunkingTrigger] = field(default_factory=list)
    rechunk_cost_estimate: float = 0.0
    improvement_value_estimate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decomposition_id": self.decomposition_id,
            "timestamp": self.timestamp.isoformat(),
            "coverage_score": self.coverage_score,
            "overlap_score": self.overlap_score,
            "causal_completeness_score": self.causal_completeness_score,
            "information_density_score": self.information_density_score,
            "time_to_insight_score": self.time_to_insight_score,
            "constraint_consistency_score": self.constraint_consistency_score,
            "boundary_quality_score": self.boundary_quality_score,
            "overall_quality_score": self.overall_quality_score,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "improvement_recommendations": self.improvement_recommendations,
            "should_rechunk": self.should_rechunk,
            "rechunk_triggers": [t.value for t in self.rechunk_triggers],
            "rechunk_cost_estimate": self.rechunk_cost_estimate,
            "improvement_value_estimate": self.improvement_value_estimate,
        }


@dataclass
class ProcessingContext:
    """Context information for processing decisions"""

    user_timeline_pressure: float = 0.5  # 0.0 = no rush, 1.0 = urgent
    available_resources: float = 1.0  # 0.0 = constrained, 1.0 = abundant
    decision_stakes: float = 0.5  # 0.0 = low impact, 1.0 = critical
    current_confidence: float = 0.5  # 0.0 = low confidence, 1.0 = high confidence
    processing_time_budget_ms: int = 30000  # Time available for processing

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_timeline_pressure": self.user_timeline_pressure,
            "available_resources": self.available_resources,
            "decision_stakes": self.decision_stakes,
            "current_confidence": self.current_confidence,
            "processing_time_budget_ms": self.processing_time_budget_ms,
        }


class ChunkingQualityMonitor:
    """
    Real-time quality assessment and adaptive re-chunking system.

    Implements research-validated metrics for assessing decomposition quality
    and uses decision theory to determine when re-chunking is beneficial.
    """

    def __init__(self):
        self.llm_client = UnifiedLLMClient()
        self.boundary_engine = get_boundary_detection_engine()

        # Metric weights (can be adjusted based on context)
        self.metric_weights = {
            QualityMetric.COVERAGE: 0.20,
            QualityMetric.OVERLAP: 0.15,
            QualityMetric.CAUSAL_COMPLETENESS: 0.20,
            QualityMetric.INFORMATION_DENSITY: 0.15,
            QualityMetric.TIME_TO_INSIGHT: 0.10,
            QualityMetric.CONSTRAINT_CONSISTENCY: 0.15,
            QualityMetric.BOUNDARY_QUALITY: 0.05,
        }

        logger.info("📊 Chunking Quality Monitor initialized")

    async def assess_quality(
        self,
        decomposition: MECEDecomposition,
        context: Optional[ProcessingContext] = None,
    ) -> QualityAssessment:
        """
        Perform comprehensive quality assessment of a decomposition.

        Args:
            decomposition: The MECE decomposition to assess
            context: Processing context for decision-making

        Returns:
            Complete quality assessment with recommendations
        """

        if context is None:
            context = ProcessingContext()

        assessment = QualityAssessment(decomposition_id=decomposition.query_id)

        # Assess individual metrics
        assessment.coverage_score = await self._assess_coverage(decomposition)
        assessment.overlap_score = await self._assess_overlap(decomposition)
        assessment.causal_completeness_score = await self._assess_causal_completeness(
            decomposition
        )
        assessment.information_density_score = self._assess_information_density(
            decomposition
        )
        assessment.time_to_insight_score = self._assess_time_to_insight(
            decomposition, context
        )
        assessment.constraint_consistency_score = (
            await self._assess_constraint_consistency(decomposition)
        )
        assessment.boundary_quality_score = self._assess_boundary_quality(decomposition)

        # Calculate overall quality score
        assessment.overall_quality_score = self._calculate_overall_score(assessment)

        # Generate analysis and recommendations
        assessment.strengths, assessment.weaknesses = (
            self._analyze_strengths_weaknesses(assessment)
        )
        assessment.improvement_recommendations = await self._generate_recommendations(
            assessment, decomposition
        )

        # Determine if re-chunking is beneficial
        assessment.should_rechunk, assessment.rechunk_triggers = self._should_rechunk(
            assessment, context
        )
        assessment.rechunk_cost_estimate = self._estimate_rechunk_cost(context)
        assessment.improvement_value_estimate = self._estimate_improvement_value(
            assessment, context
        )

        logger.info(
            f"📊 Quality assessment complete: {assessment.overall_quality_score:.2f} overall score"
        )

        return assessment

    async def _assess_coverage(self, decomposition: MECEDecomposition) -> float:
        """
        Assess collective exhaustiveness - do we have all necessary components?

        Based on MECE principle that decomposition should be collectively exhaustive.
        """

        # Check presence of essential MECE components
        component_scores = []

        # Constraints (should almost always have some)
        if decomposition.constraints:
            component_scores.append(1.0)
        else:
            component_scores.append(0.3)  # Missing constraints is concerning

        # Decisions (critical for most queries)
        if decomposition.decisions:
            component_scores.append(1.0)
        else:
            component_scores.append(0.2)  # Missing decisions is very concerning

        # Unknowns (should usually have some)
        if decomposition.unknowns:
            component_scores.append(1.0)
        else:
            component_scores.append(0.6)  # Missing unknowns is moderately concerning

        # Success metrics (important for goal clarity)
        if decomposition.success_metrics:
            component_scores.append(1.0)
        else:
            component_scores.append(0.4)  # Missing success metrics is concerning

        # Conventions (may or may not be present)
        if decomposition.conventions:
            component_scores.append(1.0)
        else:
            component_scores.append(0.8)  # Missing conventions is less concerning

        base_coverage = sum(component_scores) / len(component_scores)

        # Use LLM to assess if anything major is missing
        llm_coverage = await self._llm_assess_coverage(decomposition)

        # Combine scores
        coverage_score = (base_coverage * 0.7) + (llm_coverage * 0.3)

        return min(1.0, coverage_score)

    async def _llm_assess_coverage(self, decomposition: MECEDecomposition) -> float:
        """Use LLM to assess if the decomposition covers all important aspects"""

        prompt = f"""
        Assess if this query decomposition is collectively exhaustive (covers all important aspects).
        
        ORIGINAL QUERY: {decomposition.original_query}
        
        DECOMPOSITION:
        - Constraints: {len(decomposition.constraints)} items
        - Conventions: {len(decomposition.conventions)} items  
        - Decisions: {len(decomposition.decisions)} items
        - Unknowns: {len(decomposition.unknowns)} items
        - Success Metrics: {len(decomposition.success_metrics)} items
        
        What major aspects of this query are NOT covered by the decomposition?
        Rate completeness from 0.0 (major gaps) to 1.0 (comprehensive coverage).
        
        Return JSON with "coverage_score" and "missing_aspects".
        """

        try:
            response = await self.llm_client.call_llm(
                [
                    {
                        "role": "system",
                        "content": "You are an expert in strategic analysis and completeness assessment.",
                    },
                    {"role": "user", "content": prompt},
                ]
            )
            response = response.content

            result = json.loads(response)
            return result.get("coverage_score", 0.7)

        except Exception as e:
            logger.warning(f"⚠️ LLM coverage assessment failed: {e}")
            return 0.7

    async def _assess_overlap(self, decomposition: MECEDecomposition) -> float:
        """
        Assess mutual exclusivity - are components properly separated?

        Higher score means less overlap (better).
        """

        # Collect all component descriptions
        all_descriptions = []
        all_descriptions.extend([c.description for c in decomposition.constraints])
        all_descriptions.extend([c.description for c in decomposition.conventions])
        all_descriptions.extend([d.description for d in decomposition.decisions])
        all_descriptions.extend([u.description for u in decomposition.unknowns])
        all_descriptions.extend([s.description for s in decomposition.success_metrics])

        if len(all_descriptions) <= 1:
            return 1.0

        # Calculate semantic overlap using simple keyword analysis
        overlap_score = self._calculate_description_overlap(all_descriptions)

        # Use LLM for more sophisticated overlap assessment
        llm_overlap = await self._llm_assess_overlap(all_descriptions)

        # Combine scores (higher is better - less overlap)
        final_score = (overlap_score * 0.6) + (llm_overlap * 0.4)

        return final_score

    def _calculate_description_overlap(self, descriptions: List[str]) -> float:
        """Calculate overlap between component descriptions"""

        if len(descriptions) <= 1:
            return 1.0

        total_pairs = 0
        overlap_count = 0

        for i in range(len(descriptions)):
            for j in range(i + 1, len(descriptions)):
                total_pairs += 1

                # Calculate word overlap
                words1 = set(descriptions[i].lower().split())
                words2 = set(descriptions[j].lower().split())

                if len(words1) == 0 or len(words2) == 0:
                    continue

                overlap = len(words1.intersection(words2))
                min_words = min(len(words1), len(words2))

                # If more than 40% word overlap, consider it overlapping
                if overlap > 0.4 * min_words:
                    overlap_count += 1

        if total_pairs == 0:
            return 1.0

        # Return non-overlap score (higher is better)
        return 1.0 - (overlap_count / total_pairs)

    async def _llm_assess_overlap(self, descriptions: List[str]) -> float:
        """Use LLM to assess semantic overlap between components"""

        prompt = f"""
        Assess if these component descriptions are mutually exclusive (no significant overlap).
        
        COMPONENT DESCRIPTIONS:
        {json.dumps(descriptions, indent=2)}
        
        Rate mutual exclusivity from 0.0 (high overlap) to 1.0 (clearly separated).
        Identify any components that significantly overlap in scope or responsibility.
        
        Return JSON with "mutual_exclusivity_score" and "overlapping_pairs".
        """

        try:
            response = await self.llm_client.call_llm(
                [
                    {
                        "role": "system",
                        "content": "You are an expert in logical analysis and categorization.",
                    },
                    {"role": "user", "content": prompt},
                ]
            )
            response = response.content

            result = json.loads(response)
            return result.get("mutual_exclusivity_score", 0.7)

        except Exception as e:
            logger.warning(f"⚠️ LLM overlap assessment failed: {e}")
            return 0.7

    async def _assess_causal_completeness(
        self, decomposition: MECEDecomposition
    ) -> float:
        """
        Assess if we can trace all cause-effect relationships.

        Good decomposition should preserve causal chains.
        """

        prompt = f"""
        Assess causal completeness of this decomposition - can we trace cause-effect relationships?
        
        ORIGINAL QUERY: {decomposition.original_query}
        
        COMPONENTS:
        Constraints: {[c.description for c in decomposition.constraints]}
        Decisions: {[d.description for d in decomposition.decisions]}
        Unknowns: {[u.description for u in decomposition.unknowns]}
        
        Questions:
        1. Can we trace what causes what in this decomposition?
        2. Are there broken causal chains or missing links?
        3. Do the components connect logically?
        
        Rate causal completeness from 0.0 (broken chains) to 1.0 (complete traceability).
        
        Return JSON with "causal_completeness_score" and "missing_links".
        """

        try:
            response = await self.llm_client.call_llm(
                [
                    {
                        "role": "system",
                        "content": "You are an expert in systems thinking and causal analysis.",
                    },
                    {"role": "user", "content": prompt},
                ]
            )

            result = json.loads(response.content)
            return result.get("causal_completeness_score", 0.7)

        except Exception as e:
            logger.warning(f"⚠️ Causal completeness assessment failed: {e}")
            return 0.7

    def _assess_information_density(self, decomposition: MECEDecomposition) -> float:
        """
        Assess information value per token - efficiency metric.

        Higher density means more valuable information per unit of text.
        """

        # Calculate total token count (rough estimate)
        total_text = decomposition.original_query
        total_text += " ".join([c.description for c in decomposition.constraints])
        total_text += " ".join([c.description for c in decomposition.conventions])
        total_text += " ".join([d.description for d in decomposition.decisions])
        total_text += " ".join([u.description for u in decomposition.unknowns])
        total_text += " ".join([s.description for s in decomposition.success_metrics])

        # Rough token estimate (1 token ≈ 0.75 words)
        word_count = len(total_text.split())
        estimated_tokens = int(word_count / 0.75)

        # Calculate component count (proxy for information value)
        component_count = (
            len(decomposition.constraints)
            + len(decomposition.conventions)
            + len(decomposition.decisions)
            + len(decomposition.unknowns)
            + len(decomposition.success_metrics)
        )

        if estimated_tokens == 0:
            return 0.0

        # Information density = components per 100 tokens
        density = (component_count / estimated_tokens) * 100

        # Normalize to 0-1 scale (assume 2 components per 100 tokens is good)
        normalized_density = min(1.0, density / 2.0)

        return normalized_density

    def _assess_time_to_insight(
        self, decomposition: MECEDecomposition, context: ProcessingContext
    ) -> float:
        """
        Assess speed vs accuracy trade-off.

        Considers processing time and timeline pressure.
        """

        processing_time_ms = decomposition.processing_time_ms
        available_time_ms = context.processing_time_budget_ms

        # If processing took longer than available time, that's bad
        if processing_time_ms > available_time_ms:
            time_efficiency = available_time_ms / processing_time_ms
        else:
            time_efficiency = 1.0

        # Factor in timeline pressure
        pressure_factor = 1.0 - context.user_timeline_pressure

        # If user is under pressure, faster is better (even if slightly less accurate)
        if context.user_timeline_pressure > 0.7:
            # High pressure: prioritize speed
            insight_score = time_efficiency
        else:
            # Low pressure: balance speed and quality
            quality_proxy = decomposition.confidence_score
            insight_score = (time_efficiency * 0.4) + (quality_proxy * 0.6)

        return min(1.0, insight_score)

    async def _assess_constraint_consistency(
        self, decomposition: MECEDecomposition
    ) -> float:
        """
        Assess if hard constraints are properly preserved and not violated.

        Hard constraints should be internally consistent and not contradictory.
        """

        if not decomposition.constraints:
            return 1.0  # No constraints to violate

        constraint_descriptions = [c.description for c in decomposition.constraints]

        prompt = f"""
        Check if these hard constraints are internally consistent and non-contradictory.
        
        CONSTRAINTS: {json.dumps(constraint_descriptions, indent=2)}
        
        Questions:
        1. Do any constraints contradict each other?
        2. Are all constraints actually "hard" (unfalsifiable, unchangeable)?
        3. Are constraints properly classified?
        
        Rate consistency from 0.0 (major contradictions) to 1.0 (perfectly consistent).
        
        Return JSON with "consistency_score" and "issues".
        """

        try:
            response = await self.llm_client.call_llm(
                [
                    {
                        "role": "system",
                        "content": "You are an expert in logical consistency and constraint analysis.",
                    },
                    {"role": "user", "content": prompt},
                ]
            )

            result = json.loads(response.content)
            return result.get("consistency_score", 0.8)

        except Exception as e:
            logger.warning(f"⚠️ Constraint consistency assessment failed: {e}")
            return 0.8

    def _assess_boundary_quality(self, decomposition: MECEDecomposition) -> float:
        """
        Assess quality of detected boundaries.

        Uses boundary detection engine analysis.
        """

        if not decomposition.boundaries:
            return 0.5  # No boundaries detected

        boundary_analysis = self.boundary_engine.analyze_boundary_quality(
            decomposition.boundaries
        )
        return boundary_analysis.get("quality_score", 0.5)

    def _calculate_overall_score(self, assessment: QualityAssessment) -> float:
        """Calculate weighted overall quality score"""

        scores = {
            QualityMetric.COVERAGE: assessment.coverage_score,
            QualityMetric.OVERLAP: assessment.overlap_score,
            QualityMetric.CAUSAL_COMPLETENESS: assessment.causal_completeness_score,
            QualityMetric.INFORMATION_DENSITY: assessment.information_density_score,
            QualityMetric.TIME_TO_INSIGHT: assessment.time_to_insight_score,
            QualityMetric.CONSTRAINT_CONSISTENCY: assessment.constraint_consistency_score,
            QualityMetric.BOUNDARY_QUALITY: assessment.boundary_quality_score,
        }

        weighted_sum = sum(
            score * self.metric_weights[metric] for metric, score in scores.items()
        )
        return weighted_sum

    def _analyze_strengths_weaknesses(
        self, assessment: QualityAssessment
    ) -> Tuple[List[str], List[str]]:
        """Analyze strengths and weaknesses based on metric scores"""

        strengths = []
        weaknesses = []

        # Define thresholds
        strong_threshold = 0.8
        weak_threshold = 0.5

        metrics = [
            ("Coverage", assessment.coverage_score),
            ("Overlap Control", assessment.overlap_score),
            ("Causal Completeness", assessment.causal_completeness_score),
            ("Information Density", assessment.information_density_score),
            ("Time to Insight", assessment.time_to_insight_score),
            ("Constraint Consistency", assessment.constraint_consistency_score),
            ("Boundary Quality", assessment.boundary_quality_score),
        ]

        for metric_name, score in metrics:
            if score >= strong_threshold:
                strengths.append(f"Strong {metric_name.lower()} (score: {score:.2f})")
            elif score <= weak_threshold:
                weaknesses.append(f"Weak {metric_name.lower()} (score: {score:.2f})")

        return strengths, weaknesses

    async def _generate_recommendations(
        self, assessment: QualityAssessment, decomposition: MECEDecomposition
    ) -> List[str]:
        """Generate specific improvement recommendations"""

        recommendations = []

        # Coverage recommendations
        if assessment.coverage_score < 0.6:
            recommendations.append(
                "Expand decomposition to cover missing aspects - check for uncaptured requirements or stakeholders"
            )

        # Overlap recommendations
        if assessment.overlap_score < 0.6:
            recommendations.append(
                "Reduce component overlap - clarify boundaries and responsibilities between components"
            )

        # Causal completeness recommendations
        if assessment.causal_completeness_score < 0.6:
            recommendations.append(
                "Strengthen causal relationships - ensure cause-effect chains are complete and traceable"
            )

        # Information density recommendations
        if assessment.information_density_score < 0.4:
            recommendations.append(
                "Improve information density - combine similar components or add more specific details"
            )

        # Constraint consistency recommendations
        if assessment.constraint_consistency_score < 0.7:
            recommendations.append(
                "Review constraint consistency - check for contradictions or misclassified constraints"
            )

        # Boundary quality recommendations
        if assessment.boundary_quality_score < 0.5:
            recommendations.append(
                "Strengthen chunk boundaries - look for clearer separation signals and transition points"
            )

        return recommendations

    def _should_rechunk(
        self, assessment: QualityAssessment, context: ProcessingContext
    ) -> Tuple[bool, List[ReChunkingTrigger]]:
        """
        Determine if re-chunking would be beneficial using decision theory.

        Based on cost-benefit analysis of re-chunking vs current quality.
        """

        triggers = []

        # Check for specific trigger conditions
        if assessment.coverage_score < 0.5:
            triggers.append(ReChunkingTrigger.LOW_COVERAGE)

        if assessment.overlap_score < 0.5:
            triggers.append(ReChunkingTrigger.HIGH_OVERLAP)

        if assessment.boundary_quality_score < 0.4:
            triggers.append(ReChunkingTrigger.POOR_BOUNDARIES)

        if assessment.information_density_score < 0.3:
            triggers.append(ReChunkingTrigger.LOW_EFFICIENCY)

        if assessment.causal_completeness_score < 0.5:
            triggers.append(ReChunkingTrigger.COMPLETENESS_GAPS)

        if assessment.constraint_consistency_score < 0.6:
            triggers.append(ReChunkingTrigger.CONSTRAINT_VIOLATIONS)

        # Decision logic based on context
        should_rechunk = False

        if triggers and assessment.overall_quality_score < 0.6:
            # Poor quality with specific issues
            if context.decision_stakes > 0.7:
                # High stakes - quality is critical
                should_rechunk = True
            elif (
                context.user_timeline_pressure < 0.3
                and context.available_resources > 0.6
            ):
                # Low pressure, good resources - worth improving
                should_rechunk = True
            elif len(triggers) >= 3:
                # Multiple issues - likely worth fixing
                should_rechunk = True

        return should_rechunk, triggers

    def _estimate_rechunk_cost(self, context: ProcessingContext) -> float:
        """Estimate the cost of re-chunking (time, resources)"""

        # Base cost in relative units
        base_cost = 1.0

        # Adjust for available resources
        resource_factor = (
            2.0 - context.available_resources
        )  # Less resources = higher cost

        # Adjust for timeline pressure
        pressure_factor = (
            1.0 + context.user_timeline_pressure
        )  # More pressure = higher cost

        estimated_cost = base_cost * resource_factor * pressure_factor

        return estimated_cost

    def _estimate_improvement_value(
        self, assessment: QualityAssessment, context: ProcessingContext
    ) -> float:
        """Estimate the value of quality improvement"""

        # Potential improvement = 1.0 - current_quality
        potential_improvement = 1.0 - assessment.overall_quality_score

        # Value depends on decision stakes
        stakes_multiplier = 1.0 + context.decision_stakes

        # Value depends on confidence gap
        confidence_gap = 1.0 - context.current_confidence
        confidence_multiplier = 1.0 + confidence_gap

        estimated_value = (
            potential_improvement * stakes_multiplier * confidence_multiplier
        )

        return estimated_value


# Global instance
_quality_monitor: Optional[ChunkingQualityMonitor] = None


def get_chunking_quality_monitor() -> ChunkingQualityMonitor:
    """Get or create the global chunking quality monitor instance"""
    global _quality_monitor
    if _quality_monitor is None:
        _quality_monitor = ChunkingQualityMonitor()
    return _quality_monitor
