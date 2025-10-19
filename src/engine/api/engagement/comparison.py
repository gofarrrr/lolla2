"""
Engagement Comparison System
Operation Crystal Day 1 - Compare findings and recommendations between engagements
"""

import logging
from typing import Dict, Any, List
from uuid import UUID
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ComparisonResult:
    """Result of engagement comparison analysis"""

    engagement1_id: str
    engagement2_id: str
    comparison_summary: Dict[str, Any]
    detailed_differences: Dict[str, Any]
    confidence_analysis: Dict[str, Any]
    generated_at: str


class EngagementComparator:
    """
    Compare findings and recommendations between two engagements.

    Focuses on high-level differences: Governing Thought, Key Recommendations,
    and final confidence scores as specified in user requirements.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("📊 EngagementComparator initialized")

    def compare_engagements(
        self,
        engagement1_data: Dict[str, Any],
        engagement2_data: Dict[str, Any],
        engagement1_id: UUID,
        engagement2_id: UUID,
    ) -> ComparisonResult:
        """
        Compare two engagements and generate structured diff.

        Args:
            engagement1_data: First engagement contract data
            engagement2_data: Second engagement contract data
            engagement1_id: UUID of first engagement
            engagement2_id: UUID of second engagement

        Returns:
            ComparisonResult with structured differences
        """

        self.logger.info(
            f"📊 Comparing engagements {engagement1_id} vs {engagement2_id}"
        )

        # Extract key findings from each engagement
        findings1 = self._extract_key_findings(engagement1_data, "Engagement 1")
        findings2 = self._extract_key_findings(engagement2_data, "Engagement 2")

        # Generate comparison summary
        comparison_summary = self._generate_comparison_summary(findings1, findings2)

        # Generate detailed differences
        detailed_differences = self._generate_detailed_differences(findings1, findings2)

        # Analyze confidence differences
        confidence_analysis = self._analyze_confidence_differences(findings1, findings2)

        result = ComparisonResult(
            engagement1_id=str(engagement1_id),
            engagement2_id=str(engagement2_id),
            comparison_summary=comparison_summary,
            detailed_differences=detailed_differences,
            confidence_analysis=confidence_analysis,
            generated_at=datetime.utcnow().isoformat(),
        )

        self.logger.info(
            f"✅ Comparison completed - {len(detailed_differences)} difference categories found"
        )

        return result

    def _extract_key_findings(
        self, engagement_data: Dict[str, Any], label: str
    ) -> Dict[str, Any]:
        """Extract key findings from engagement contract data"""

        try:
            # Navigate contract structure to find key findings
            workflow_state = engagement_data.get("workflow_state", {})
            phase_results = workflow_state.get("phase_results", {})

            # Extract governing thought (usually in synthesis phase)
            synthesis_result = phase_results.get("synthesis_delivery", {})
            synthesis_data = synthesis_result.get("result", {})

            governing_thought = (
                synthesis_data.get("governing_thought")
                or synthesis_data.get("key_insight")
                or synthesis_data.get("executive_summary", "Not available")
            )

            # Extract key recommendations
            recommendations = (
                synthesis_data.get("recommendations", [])
                or synthesis_data.get("key_recommendations", [])
                or []
            )

            # Extract confidence score
            confidence_score = (
                synthesis_data.get("confidence_score")
                or engagement_data.get("overall_confidence")
                or 0.0
            )

            # Extract problem statement for context
            engagement_context = engagement_data.get("engagement_context", {})
            problem_statement = engagement_context.get(
                "problem_statement", "Not available"
            )

            findings = {
                "label": label,
                "governing_thought": governing_thought,
                "recommendations": (
                    recommendations
                    if isinstance(recommendations, list)
                    else [recommendations]
                ),
                "confidence_score": float(confidence_score),
                "problem_statement": problem_statement,
                "has_synthesis": bool(synthesis_result),
                "recommendations_count": (
                    len(recommendations)
                    if isinstance(recommendations, list)
                    else (1 if recommendations else 0)
                ),
            }

            self.logger.info(
                f"📄 Extracted findings for {label}: {findings['recommendations_count']} recommendations, {findings['confidence_score']:.2f} confidence"
            )

            return findings

        except Exception as e:
            self.logger.error(f"❌ Failed to extract findings for {label}: {e}")

            return {
                "label": label,
                "governing_thought": "Error extracting data",
                "recommendations": [],
                "confidence_score": 0.0,
                "problem_statement": "Error extracting data",
                "has_synthesis": False,
                "recommendations_count": 0,
            }

    def _generate_comparison_summary(
        self, findings1: Dict[str, Any], findings2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate high-level comparison summary"""

        confidence_diff = abs(
            findings1["confidence_score"] - findings2["confidence_score"]
        )

        # Determine which engagement has higher confidence
        higher_confidence = (
            "Engagement 1"
            if findings1["confidence_score"] > findings2["confidence_score"]
            else "Engagement 2"
        )
        if confidence_diff < 0.05:  # Very small difference
            higher_confidence = "Similar"

        # Count recommendation differences
        rec_count_diff = abs(
            findings1["recommendations_count"] - findings2["recommendations_count"]
        )

        return {
            "confidence_comparison": {
                "engagement1_confidence": findings1["confidence_score"],
                "engagement2_confidence": findings2["confidence_score"],
                "difference": round(confidence_diff, 3),
                "higher_confidence": higher_confidence,
            },
            "recommendations_comparison": {
                "engagement1_count": findings1["recommendations_count"],
                "engagement2_count": findings2["recommendations_count"],
                "count_difference": rec_count_diff,
            },
            "data_quality": {
                "engagement1_has_synthesis": findings1["has_synthesis"],
                "engagement2_has_synthesis": findings2["has_synthesis"],
                "both_complete": findings1["has_synthesis"]
                and findings2["has_synthesis"],
            },
        }

    def _generate_detailed_differences(
        self, findings1: Dict[str, Any], findings2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate detailed differences between engagements"""

        differences = {}

        # Governing thought comparison
        differences["governing_thought"] = {
            "engagement1": (
                findings1["governing_thought"][:200] + "..."
                if len(findings1["governing_thought"]) > 200
                else findings1["governing_thought"]
            ),
            "engagement2": (
                findings2["governing_thought"][:200] + "..."
                if len(findings2["governing_thought"]) > 200
                else findings2["governing_thought"]
            ),
            "are_similar": self._are_texts_similar(
                findings1["governing_thought"], findings2["governing_thought"]
            ),
        }

        # Recommendations comparison
        differences["recommendations"] = {
            "engagement1": [
                rec[:100] + "..." if len(str(rec)) > 100 else str(rec)
                for rec in findings1["recommendations"][:3]  # Limit to first 3
            ],
            "engagement2": [
                rec[:100] + "..." if len(str(rec)) > 100 else str(rec)
                for rec in findings2["recommendations"][:3]  # Limit to first 3
            ],
            "count_difference": abs(
                findings1["recommendations_count"] - findings2["recommendations_count"]
            ),
            "overlap_detected": self._detect_recommendation_overlap(
                findings1["recommendations"], findings2["recommendations"]
            ),
        }

        # Problem statement comparison (for context)
        differences["problem_context"] = {
            "engagement1_problem": (
                findings1["problem_statement"][:150] + "..."
                if len(findings1["problem_statement"]) > 150
                else findings1["problem_statement"]
            ),
            "engagement2_problem": (
                findings2["problem_statement"][:150] + "..."
                if len(findings2["problem_statement"]) > 150
                else findings2["problem_statement"]
            ),
            "are_same_problem": self._are_texts_similar(
                findings1["problem_statement"], findings2["problem_statement"]
            ),
        }

        return differences

    def _analyze_confidence_differences(
        self, findings1: Dict[str, Any], findings2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze confidence score differences between engagements"""

        conf1 = findings1["confidence_score"]
        conf2 = findings2["confidence_score"]
        diff = abs(conf1 - conf2)

        # Categorize difference magnitude
        if diff < 0.1:
            magnitude = "minimal"
        elif diff < 0.3:
            magnitude = "moderate"
        else:
            magnitude = "significant"

        return {
            "confidence_scores": {"engagement1": conf1, "engagement2": conf2},
            "absolute_difference": round(diff, 3),
            "relative_difference_percent": round(
                (diff / max(conf1, conf2, 0.01)) * 100, 1
            ),
            "difference_magnitude": magnitude,
            "interpretation": self._interpret_confidence_difference(
                conf1, conf2, magnitude
            ),
        }

    def _are_texts_similar(self, text1: str, text2: str) -> bool:
        """Simple text similarity check (could be enhanced with embeddings later)"""
        if not text1 or not text2:
            return False

        # Basic keyword overlap check
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return False

        overlap = words1.intersection(words2)
        overlap_ratio = len(overlap) / max(len(words1), len(words2))

        return overlap_ratio > 0.3  # 30% word overlap threshold

    def _detect_recommendation_overlap(self, recs1: List, recs2: List) -> bool:
        """Detect if recommendations have significant overlap"""
        if not recs1 or not recs2:
            return False

        # Check for similar recommendations
        for rec1 in recs1[:3]:  # Limit comparison scope
            for rec2 in recs2[:3]:
                if self._are_texts_similar(str(rec1), str(rec2)):
                    return True

        return False

    def _interpret_confidence_difference(
        self, conf1: float, conf2: float, magnitude: str
    ) -> str:
        """Generate human-readable interpretation of confidence differences"""

        higher_conf = max(conf1, conf2)
        lower_conf = min(conf1, conf2)

        if magnitude == "minimal":
            return (
                f"Both engagements show similar confidence levels (~{higher_conf:.1f})"
            )
        elif magnitude == "moderate":
            return (
                f"Notable confidence difference: {higher_conf:.1f} vs {lower_conf:.1f}"
            )
        else:
            return f"Significant confidence gap: {higher_conf:.1f} vs {lower_conf:.1f} - may indicate different problem complexity or data quality"

    def to_api_response(self, result: ComparisonResult) -> Dict[str, Any]:
        """Convert comparison result to API response format"""

        return {
            "comparison_id": f"{result.engagement1_id}_vs_{result.engagement2_id}",
            "engagements_compared": [result.engagement1_id, result.engagement2_id],
            "generated_at": result.generated_at,
            "summary": result.comparison_summary,
            "differences": result.detailed_differences,
            "confidence_analysis": result.confidence_analysis,
            "metadata": {
                "comparison_type": "high_level_findings",
                "focus_areas": [
                    "governing_thought",
                    "recommendations",
                    "confidence_scores",
                ],
                "note": "Comparison focuses on strategic differences rather than implementation details",
            },
        }


# Factory function for easy instantiation
def create_engagement_comparator() -> EngagementComparator:
    """Create and configure an EngagementComparator instance."""
    return EngagementComparator()
