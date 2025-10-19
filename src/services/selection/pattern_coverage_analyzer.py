# src/services/selection/pattern_coverage_analyzer.py
from __future__ import annotations

from typing import Any, Dict, List
import logging

from src.services.selection.pattern_contracts import ICoverageAnalyzer, CoverageAnalysis

logger = logging.getLogger(__name__)


class V1CoverageAnalyzer(ICoverageAnalyzer):
    """
    V1 implementation for coverage analysis and recommendations.

    K-04: analyze(...) now performs full coverage computation and calls
    _recommend(...) internally to produce a complete CoverageAnalysis model.
    """

    def analyze(
        self,
        selected_patterns: List[str],
        framework_type: str,
        task_classification: Dict[str, Any],
        target_coverage: float = 0.90,
    ) -> CoverageAnalysis:
        # Define critical analytical dimensions
        critical_dimensions = {
            "systems_thinking": {
                "description": "Holistic view of interconnected components and relationships",
                "patterns": [
                    "NWAY_SYNTHESIS_001",
                    "NWAY_COMPLEXITY_002",
                    "NWAY_REASONING_001",
                ],
                "weight": 0.12,
            },
            "critical_analysis": {
                "description": "Systematic evaluation of evidence and assumptions",
                "patterns": [
                    "NWAY_REASONING_001",
                    "NWAY_DECISION_002",
                    "NWAY_PERCEPTION_001",
                ],
                "weight": 0.11,
            },
            "causal_reasoning": {
                "description": "Understanding cause-effect relationships and dependencies",
                "patterns": [
                    "NWAY_REASONING_001",
                    "NWAY_COMPLEXITY_002",
                    "NWAY_DECOMPOSITION_002",
                ],
                "weight": 0.10,
            },
            "pattern_recognition": {
                "description": "Identifying recurring themes and structural similarities",
                "patterns": [
                    "NWAY_PERCEPTION_001",
                    "NWAY_SYNTHESIS_001",
                    "NWAY_REASONING_001",
                ],
                "weight": 0.10,
            },
            "scenario_analysis": {
                "description": "Exploring multiple possible futures and outcomes",
                "patterns": [
                    "NWAY_DECISION_002",
                    "NWAY_SYNTHESIS_001",
                    "NWAY_COMPLEXITY_002",
                ],
                "weight": 0.10,
            },
            "first_principles": {
                "description": "Breaking down to fundamental truths and building up",
                "patterns": [
                    "NWAY_DECOMPOSITION_002",
                    "NWAY_REASONING_001",
                    "NWAY_COMPLIANCE_001",
                ],
                "weight": 0.09,
            },
            "risk_assessment": {
                "description": "Identifying and evaluating potential negative outcomes",
                "patterns": [
                    "NWAY_DECISION_002",
                    "NWAY_FINANCIAL_001",
                    "NWAY_COMPLEXITY_002",
                ],
                "weight": 0.09,
            },
            "trade_off_analysis": {
                "description": "Evaluating competing priorities and opportunity costs",
                "patterns": [
                    "NWAY_DECISION_002",
                    "NWAY_FINANCIAL_001",
                    "NWAY_SYNTHESIS_001",
                ],
                "weight": 0.09,
            },
            "outside_view": {
                "description": "External perspective and reference class forecasting",
                "patterns": [
                    "NWAY_PERCEPTION_001",
                    "NWAY_BEHAVIORAL_001",
                    "NWAY_METACOGNITION_001",
                ],
                "weight": 0.10,
            },
            "verification": {
                "description": "Validating conclusions through multiple independent methods",
                "patterns": [
                    "NWAY_COMPLIANCE_001",
                    "NWAY_EXECUTION_001",
                    "NWAY_REASONING_001",
                ],
                "weight": 0.10,
            },
        }

        # Calculate coverage
        dimension_coverage: Dict[str, Any] = {}
        total_weighted_coverage = 0.0
        coverage_gaps: List[Dict[str, Any]] = []

        for dimension, info in critical_dimensions.items():
            patterns_for_dimension = info["patterns"]
            weight = info["weight"]
            covered_patterns = set(selected_patterns) & set(patterns_for_dimension)
            coverage_ratio = (
                len(covered_patterns) / len(patterns_for_dimension)
                if patterns_for_dimension
                else 0.0
            )
            dimension_coverage[dimension] = {
                "coverage_ratio": coverage_ratio,
                "covered_patterns": list(covered_patterns),
                "missing_patterns": list(set(patterns_for_dimension) - covered_patterns),
                "weight": weight,
                "weighted_coverage": coverage_ratio * weight,
            }
            total_weighted_coverage += coverage_ratio * weight
            if coverage_ratio < 0.5 and weight >= 0.09:
                coverage_gaps.append(
                    {
                        "dimension": dimension,
                        "coverage": coverage_ratio,
                        "missing_patterns": list(set(patterns_for_dimension) - covered_patterns),
                        "impact": "HIGH" if weight >= 0.10 else "MEDIUM",
                    }
                )

        meets_target = total_weighted_coverage >= target_coverage
        coverage_grade = self._calculate_coverage_grade(total_weighted_coverage)
        recommendations = self._recommend(
            coverage_gaps, dimension_coverage, selected_patterns, framework_type
        )

        analysis = CoverageAnalysis(
            meets_target=meets_target,
            total_coverage=total_weighted_coverage,
            target_coverage=target_coverage,
            coverage_grade=coverage_grade,
            coverage_gaps=coverage_gaps,
            recommendations=recommendations,
            dimension_coverage=dimension_coverage,
            gap_count=len(coverage_gaps),
            validation_summary=(
                f"Coverage: {total_weighted_coverage:.1%} ({'PASS' if meets_target else 'FAIL'}) - {coverage_grade}"
            ),
        )

        if meets_target:
            logger.info(
                f"✅ DIMENSION COVERAGE: {total_weighted_coverage:.1%} coverage PASSES target {target_coverage:.0%}"
            )
        else:
            logger.warning(
                f"⚠️ DIMENSION COVERAGE: {total_weighted_coverage:.1%} coverage FAILS target {target_coverage:.0%}"
            )
            logger.warning(
                f"🔍 Coverage gaps found in {len(coverage_gaps)} critical dimensions"
            )

        return analysis

    def _recommend(
        self,
        coverage_gaps: List[Dict[str, Any]],
        dimension_coverage: Dict[str, Any],
        selected_patterns: List[str],
        framework_type: str,
    ) -> List[str]:
        """Generate specific recommendations to improve dimension coverage.
        Ported verbatim from the legacy _generate_coverage_recommendations method to
        ensure golden-master parity.
        """
        recommendations: List[str] = []

        if not coverage_gaps:
            recommendations.append("✅ Excellent dimension coverage - no gaps identified")
            return recommendations

        # Prioritize high-impact gaps
        high_impact_gaps = [gap for gap in coverage_gaps if gap.get("impact") == "HIGH"]
        medium_impact_gaps = [gap for gap in coverage_gaps if gap.get("impact") == "MEDIUM"]

        if high_impact_gaps:
            recommendations.append(
                f"🚨 HIGH PRIORITY: Address {len(high_impact_gaps)} critical dimension gaps"
            )

            for gap in high_impact_gaps[:3]:  # Top 3 high impact gaps
                missing_patterns = gap.get("missing_patterns") or []
                if missing_patterns:
                    best_pattern = missing_patterns[0]  # First missing pattern as recommendation
                    recommendations.append(
                        f"  → Add {best_pattern} for {gap.get('dimension')} dimension"
                    )

        if medium_impact_gaps:
            recommendations.append(
                f"⚠️ MEDIUM PRIORITY: Consider addressing {len(medium_impact_gaps)} additional gaps"
            )

            for gap in medium_impact_gaps[:2]:  # Top 2 medium impact gaps
                missing_patterns = gap.get("missing_patterns") or []
                if missing_patterns:
                    best_pattern = missing_patterns[0]
                    recommendations.append(
                        f"  → Consider {best_pattern} for {gap.get('dimension')} dimension"
                    )

        # Suggest pattern optimization
        if len(selected_patterns) < 3:
            recommendations.append(
                "💡 Consider using multi-pattern selection for better dimension coverage"
            )

        # Framework-specific recommendations
        framework_lower = (framework_type or "").lower()
        if "strategic" in framework_lower and any(
            "systems_thinking" in (gap.get("dimension") or "") for gap in coverage_gaps
        ):
            recommendations.append(
                "📊 Strategic frameworks benefit from stronger systems thinking coverage"
            )
        elif "financial" in framework_lower and any(
            "risk_assessment" in (gap.get("dimension") or "") for gap in coverage_gaps
        ):
            recommendations.append(
                "💰 Financial frameworks should prioritize risk assessment dimension"
            )

        return recommendations

    def _calculate_coverage_grade(self, coverage: float) -> str:
        if coverage >= 0.95:
            return "A+ (Excellent)"
        elif coverage >= 0.90:
            return "A (Very Good)"
        elif coverage >= 0.80:
            return "B (Good)"
        elif coverage >= 0.70:
            return "C (Acceptable)"
        elif coverage >= 0.60:
            return "D (Poor)"
        else:
            return "F (Failing)"
