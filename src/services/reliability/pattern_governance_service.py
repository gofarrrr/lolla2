"""
METIS Pattern Governance Service
Part of Reliability Services Cluster - Focused on emergent pattern discovery and governance

Extracted from vulnerability_solutions.py PatternGovernanceManager during Phase 5 decomposition.
Single Responsibility: Discover, validate, and govern emergent patterns through systematic workflows.
"""

import logging
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict

from src.services.contracts.reliability_contracts import (
    EmergentPatternContract,
    IPatternGovernanceService,
    PatternStatus,
    EmergentPattern,
)


class PatternGovernanceService(IPatternGovernanceService):
    """
    Focused service for emergent pattern discovery and governance workflow management
    Clean extraction from vulnerability_solutions.py PatternGovernanceManager
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Pattern discovery thresholds
        self.discovery_thresholds = {
            "min_similar_engagements": 3,
            "min_success_rate": 0.75,
            "min_confidence_improvement": 0.15,
            "min_statistical_significance": 0.05,
        }

        # Governance workflow stages
        self.governance_stages = {
            PatternStatus.DISCOVERED: {
                "duration_days": 7,
                "requirements": ["initial_validation", "pattern_definition"],
                "next_stage": PatternStatus.SANDBOX_TESTING,
            },
            PatternStatus.SANDBOX_TESTING: {
                "duration_days": 14,
                "requirements": ["sandbox_validation", "performance_metrics"],
                "next_stage": PatternStatus.PEER_REVIEW,
            },
            PatternStatus.PEER_REVIEW: {
                "duration_days": 10,
                "requirements": ["peer_validation", "expert_review"],
                "next_stage": PatternStatus.PRODUCTION,
            },
            PatternStatus.PRODUCTION: {
                "duration_days": None,  # Indefinite
                "requirements": ["production_monitoring"],
                "next_stage": None,
            },
        }

        # Pattern tracking storage (in production would be database)
        self.discovered_patterns = {}
        self.pattern_performance_history = defaultdict(list)

        self.logger.info("🔍 PatternGovernanceService initialized")

    async def evaluate_for_pattern_discovery(
        self, engagement_data: Dict[str, Any], similar_engagements: List[Dict[str, Any]]
    ) -> Optional[EmergentPatternContract]:
        """
        Core service method: Evaluate engagement for emergent pattern discovery
        Clean, focused implementation with single responsibility
        """
        try:
            if (
                len(similar_engagements)
                < self.discovery_thresholds["min_similar_engagements"]
            ):
                # Not enough similar engagements for pattern discovery
                return None

            # Analyze engagement patterns
            pattern_analysis = await self._analyze_engagement_patterns(
                engagement_data, similar_engagements
            )

            if not pattern_analysis["pattern_detected"]:
                return None

            # Create new pattern discovery
            pattern = await self._create_emergent_pattern(
                engagement_data, pattern_analysis, similar_engagements
            )

            # Store pattern for governance workflow
            self.discovered_patterns[pattern.pattern_id] = pattern

            # Convert to service contract
            return EmergentPatternContract(
                pattern_id=pattern.pattern_id,
                pattern_name=pattern.name,
                discovery_engagement_id=engagement_data.get("engagement_id", "unknown"),
                pattern_status=pattern.status.value,
                confidence_score=pattern.confidence_score,
                supporting_cases=pattern.supporting_cases,
                validation_metrics=pattern.validation_metrics,
                governance_stage=pattern.status.value,
                discovery_timestamp=pattern.created_at,
                service_version="v5_modular",
            )

        except Exception as e:
            self.logger.error(f"❌ Pattern discovery evaluation failed: {e}")
            return None

    async def _analyze_engagement_patterns(
        self, engagement_data: Dict[str, Any], similar_engagements: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze engagement patterns for emergent behavior discovery"""
        analysis = {
            "pattern_detected": False,
            "pattern_type": None,
            "confidence_score": 0.0,
            "supporting_evidence": [],
            "validation_metrics": {},
        }

        # Pattern Type 1: Consistent High Performance Pattern
        high_perf_pattern = self._detect_high_performance_pattern(
            engagement_data, similar_engagements
        )
        if high_perf_pattern["detected"]:
            analysis.update(
                {
                    "pattern_detected": True,
                    "pattern_type": "high_performance_methodology",
                    "confidence_score": high_perf_pattern["confidence"],
                    "supporting_evidence": high_perf_pattern["evidence"],
                    "validation_metrics": high_perf_pattern["metrics"],
                }
            )
            return analysis

        # Pattern Type 2: Novel Problem-Solution Mapping
        novel_mapping_pattern = self._detect_novel_mapping_pattern(
            engagement_data, similar_engagements
        )
        if novel_mapping_pattern["detected"]:
            analysis.update(
                {
                    "pattern_detected": True,
                    "pattern_type": "novel_problem_solution_mapping",
                    "confidence_score": novel_mapping_pattern["confidence"],
                    "supporting_evidence": novel_mapping_pattern["evidence"],
                    "validation_metrics": novel_mapping_pattern["metrics"],
                }
            )
            return analysis

        # Pattern Type 3: Cross-Industry Transfer Success
        transfer_pattern = self._detect_transfer_success_pattern(
            engagement_data, similar_engagements
        )
        if transfer_pattern["detected"]:
            analysis.update(
                {
                    "pattern_detected": True,
                    "pattern_type": "cross_industry_transfer_success",
                    "confidence_score": transfer_pattern["confidence"],
                    "supporting_evidence": transfer_pattern["evidence"],
                    "validation_metrics": transfer_pattern["metrics"],
                }
            )

        return analysis

    def _detect_high_performance_pattern(
        self, engagement_data: Dict[str, Any], similar_engagements: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Detect consistently high-performing methodology patterns"""

        # Calculate success metrics across similar engagements
        success_scores = []
        methodology_consistency = []

        current_methodology = engagement_data.get("methodology_used", "unknown")
        current_score = engagement_data.get("quality_score", 0.0)

        for engagement in similar_engagements:
            score = engagement.get("quality_score", 0.0)
            methodology = engagement.get("methodology_used", "unknown")

            success_scores.append(score)
            if methodology == current_methodology:
                methodology_consistency.append(True)
            else:
                methodology_consistency.append(False)

        # Calculate pattern metrics
        avg_success_rate = (
            sum(success_scores) / len(success_scores) if success_scores else 0.0
        )
        methodology_consistency_rate = (
            sum(methodology_consistency) / len(methodology_consistency)
            if methodology_consistency
            else 0.0
        )

        # Check thresholds
        if (
            avg_success_rate >= self.discovery_thresholds["min_success_rate"]
            and methodology_consistency_rate >= 0.8
            and current_score >= 0.85
        ):

            return {
                "detected": True,
                "confidence": min(
                    0.95, avg_success_rate * methodology_consistency_rate
                ),
                "evidence": [
                    f"Average success rate: {avg_success_rate:.2f}",
                    f"Methodology consistency: {methodology_consistency_rate:.2f}",
                    f"Current engagement score: {current_score:.2f}",
                ],
                "metrics": {
                    "success_rate": avg_success_rate,
                    "consistency_rate": methodology_consistency_rate,
                    "sample_size": len(similar_engagements),
                },
            }

        return {"detected": False}

    def _detect_novel_mapping_pattern(
        self, engagement_data: Dict[str, Any], similar_engagements: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Detect novel problem-solution mapping patterns"""

        problem_type = engagement_data.get("problem_type", "unknown")
        solution_approach = engagement_data.get("solution_approach", "unknown")
        current_score = engagement_data.get("quality_score", 0.0)

        # Look for novel combinations that work well
        similar_combinations = []
        for engagement in similar_engagements:
            if (
                engagement.get("problem_type") == problem_type
                and engagement.get("solution_approach") == solution_approach
            ):
                similar_combinations.append(engagement.get("quality_score", 0.0))

        if len(similar_combinations) >= 2:
            avg_combination_score = sum(similar_combinations) / len(
                similar_combinations
            )

            # Check if this combination consistently outperforms
            if avg_combination_score >= 0.8 and current_score >= 0.8:
                return {
                    "detected": True,
                    "confidence": min(0.9, avg_combination_score),
                    "evidence": [
                        f"Novel combination: {problem_type} + {solution_approach}",
                        f"Average performance: {avg_combination_score:.2f}",
                        f"Sample cases: {len(similar_combinations)}",
                    ],
                    "metrics": {
                        "combination_performance": avg_combination_score,
                        "case_count": len(similar_combinations),
                        "novelty_score": 0.7,  # Simplified novelty calculation
                    },
                }

        return {"detected": False}

    def _detect_transfer_success_pattern(
        self, engagement_data: Dict[str, Any], similar_engagements: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Detect successful cross-industry transfer patterns"""

        source_industry = engagement_data.get("business_context", {}).get(
            "industry", "unknown"
        )
        frameworks_used = engagement_data.get("frameworks_used", [])
        current_score = engagement_data.get("quality_score", 0.0)

        # Look for cross-industry framework applications
        cross_industry_successes = []
        for engagement in similar_engagements:
            eng_industry = engagement.get("business_context", {}).get(
                "industry", "unknown"
            )
            if eng_industry != source_industry:
                # Different industry using similar frameworks
                eng_frameworks = engagement.get("frameworks_used", [])
                framework_overlap = len(set(frameworks_used) & set(eng_frameworks))

                if framework_overlap >= 2:
                    cross_industry_successes.append(
                        {
                            "score": engagement.get("quality_score", 0.0),
                            "industry": eng_industry,
                            "overlap": framework_overlap,
                        }
                    )

        if len(cross_industry_successes) >= 2:
            avg_transfer_score = sum(
                s["score"] for s in cross_industry_successes
            ) / len(cross_industry_successes)

            if avg_transfer_score >= 0.75 and current_score >= 0.75:
                return {
                    "detected": True,
                    "confidence": min(0.85, avg_transfer_score),
                    "evidence": [
                        f"Cross-industry framework transfer from {source_industry}",
                        f"Average transfer performance: {avg_transfer_score:.2f}",
                        f"Transfer cases: {len(cross_industry_successes)}",
                    ],
                    "metrics": {
                        "transfer_performance": avg_transfer_score,
                        "transfer_count": len(cross_industry_successes),
                        "industries_reached": len(
                            set(s["industry"] for s in cross_industry_successes)
                        ),
                    },
                }

        return {"detected": False}

    async def _create_emergent_pattern(
        self,
        engagement_data: Dict[str, Any],
        pattern_analysis: Dict[str, Any],
        similar_engagements: List[Dict[str, Any]],
    ) -> EmergentPattern:
        """Create emergent pattern object from analysis"""

        # Generate unique pattern ID
        pattern_content = f"{pattern_analysis['pattern_type']}_{engagement_data.get('engagement_id', 'unknown')}_{datetime.utcnow().isoformat()}"
        pattern_id = hashlib.md5(pattern_content.encode()).hexdigest()[:12]

        # Generate descriptive pattern name
        pattern_name = self._generate_pattern_name(pattern_analysis)

        # Generate pattern description
        description = self._generate_pattern_description(
            pattern_analysis, engagement_data
        )

        # Collect supporting case IDs
        supporting_cases = [
            eng.get("engagement_id", "unknown") for eng in similar_engagements
        ]
        supporting_cases.append(engagement_data.get("engagement_id", "unknown"))

        return EmergentPattern(
            pattern_id=pattern_id,
            name=pattern_name,
            description=description,
            discovery_engagement_id=engagement_data.get("engagement_id", "unknown"),
            discovery_context=engagement_data.get("business_context", {}),
            supporting_cases=supporting_cases,
            confidence_score=pattern_analysis["confidence_score"],
            validation_metrics=pattern_analysis["validation_metrics"],
            status=PatternStatus.DISCOVERED,
            created_at=datetime.utcnow(),
        )

    def _generate_pattern_name(self, pattern_analysis: Dict[str, Any]) -> str:
        """Generate human-readable pattern name"""
        pattern_type = pattern_analysis["pattern_type"]

        if pattern_type == "high_performance_methodology":
            return "High-Performance Strategic Methodology Pattern"
        elif pattern_type == "novel_problem_solution_mapping":
            return "Novel Problem-Solution Mapping Pattern"
        elif pattern_type == "cross_industry_transfer_success":
            return "Cross-Industry Framework Transfer Pattern"
        else:
            return f"Emergent Pattern: {pattern_type.replace('_', ' ').title()}"

    def _generate_pattern_description(
        self, pattern_analysis: Dict[str, Any], engagement_data: Dict[str, Any]
    ) -> str:
        """Generate detailed pattern description"""
        pattern_type = pattern_analysis["pattern_type"]
        evidence = pattern_analysis["supporting_evidence"]

        base_desc = (
            "Emergent pattern discovered through analysis of similar engagements. "
        )

        if pattern_type == "high_performance_methodology":
            return (
                base_desc
                + f"This pattern represents a consistently high-performing strategic methodology "
                f"with demonstrated success across multiple similar business contexts. "
                f"Evidence: {'; '.join(evidence)}"
            )

        elif pattern_type == "novel_problem_solution_mapping":
            return (
                base_desc
                + f"This pattern represents a novel and effective mapping between specific problem types "
                f"and solution approaches that has shown consistent positive results. "
                f"Evidence: {'; '.join(evidence)}"
            )

        elif pattern_type == "cross_industry_transfer_success":
            return (
                base_desc
                + f"This pattern represents successful transfer of strategic frameworks across different "
                f"industries, demonstrating broader applicability than originally anticipated. "
                f"Evidence: {'; '.join(evidence)}"
            )

        else:
            return base_desc + f"Evidence: {'; '.join(evidence)}"

    async def advance_pattern_through_governance(self, pattern_id: str) -> bool:
        """Advance pattern through governance workflow"""
        try:
            if pattern_id not in self.discovered_patterns:
                self.logger.warning(
                    f"⚠️ Pattern {pattern_id} not found for governance advancement"
                )
                return False

            pattern = self.discovered_patterns[pattern_id]
            current_stage = pattern.status

            # Check if pattern can advance
            if current_stage == PatternStatus.PRODUCTION:
                self.logger.info(f"ℹ️ Pattern {pattern_id} already in production")
                return True

            # Get governance stage requirements
            stage_config = self.governance_stages[current_stage]
            next_stage = stage_config["next_stage"]

            # In full implementation, would check requirements
            # For now, auto-advance after time threshold
            time_since_creation = datetime.utcnow() - pattern.created_at
            required_duration = timedelta(days=stage_config["duration_days"])

            if time_since_creation >= required_duration:
                # Advance to next stage
                pattern.status = next_stage

                self.logger.info(
                    f"✅ Pattern {pattern_id} advanced from {current_stage.value} to {next_stage.value}"
                )
                return True
            else:
                self.logger.info(
                    f"ℹ️ Pattern {pattern_id} not ready for advancement (needs {required_duration - time_since_creation})"
                )
                return False

        except Exception as e:
            self.logger.error(f"❌ Pattern governance advancement failed: {e}")
            return False

    async def get_governance_council_status(self) -> Dict[str, Any]:
        """Get status of pattern governance council"""
        patterns_by_stage = defaultdict(int)

        for pattern in self.discovered_patterns.values():
            patterns_by_stage[pattern.status.value] += 1

        return {
            "total_patterns": len(self.discovered_patterns),
            "patterns_by_stage": dict(patterns_by_stage),
            "governance_stages": {
                stage.value: config["duration_days"]
                for stage, config in self.governance_stages.items()
            },
        }

    async def get_service_health(self) -> Dict[str, Any]:
        """Return service health status"""
        return {
            "service_name": "PatternGovernanceService",
            "status": "healthy",
            "version": "v5_modular",
            "capabilities": [
                "emergent_pattern_discovery",
                "governance_workflow_management",
                "pattern_validation",
                "cross_engagement_analysis",
            ],
            "discovery_thresholds": self.discovery_thresholds,
            "governance_stages_count": len(self.governance_stages),
            "patterns_tracked": len(self.discovered_patterns),
            "last_health_check": datetime.utcnow().isoformat(),
        }


# Global service instance for dependency injection
_pattern_governance_service: Optional[PatternGovernanceService] = None


def get_pattern_governance_service() -> PatternGovernanceService:
    """Get or create global pattern governance service instance"""
    global _pattern_governance_service

    if _pattern_governance_service is None:
        _pattern_governance_service = PatternGovernanceService()

    return _pattern_governance_service
