"""
METIS Reliability Coordinator Service
Part of Reliability Services Cluster - Orchestrates all reliability services

Clean extraction from vulnerability_solutions.py during Phase 5 decomposition.
Single Responsibility: Coordinate all reliability services and provide unified reliability assessment.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

from src.services.contracts.reliability_contracts import (
    ReliabilityAssessmentContract,
    IReliabilityCoordinatorService,
)

# Import all reliability services
from src.services.reliability.failure_detection_service import (
    get_failure_detection_service,
)
from src.services.reliability.exploration_strategy_service import (
    get_exploration_strategy_service,
)
from src.services.reliability.feedback_orchestration_service import (
    get_feedback_orchestration_service,
)
from src.services.reliability.validation_engine_service import (
    get_validation_engine_service,
)
from src.services.reliability.pattern_governance_service import (
    get_pattern_governance_service,
)


class ReliabilityCoordinatorService(IReliabilityCoordinatorService):
    """
    Coordinator service that orchestrates all reliability services
    Provides unified reliability assessment and cluster health monitoring
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize all reliability services
        self.failure_detection = get_failure_detection_service()
        self.exploration_strategy = get_exploration_strategy_service()
        self.feedback_orchestration = get_feedback_orchestration_service()
        self.validation_engine = get_validation_engine_service()
        self.pattern_governance = get_pattern_governance_service()

        # Service coordination weights
        self.assessment_weights = {
            "failure_analysis": 0.30,
            "validation_results": 0.25,
            "exploration_decision": 0.15,
            "feedback_quality": 0.15,
            "pattern_discovery": 0.15,
        }

        self.logger.info(
            "🎯 ReliabilityCoordinatorService initialized - all reliability services coordinated"
        )

    async def assess_engagement_reliability(
        self, engagement_data: Dict[str, Any]
    ) -> ReliabilityAssessmentContract:
        """
        Core coordination method: Orchestrate all reliability services for comprehensive assessment
        Clean, focused implementation that coordinates without duplicating functionality
        """
        try:
            engagement_id = engagement_data.get("engagement_id", "unknown")

            # Execute all reliability services in parallel for efficiency
            self.logger.info(
                f"🔄 Starting comprehensive reliability assessment for engagement {engagement_id}"
            )

            # Parallel service execution
            reliability_tasks = await asyncio.gather(
                self._execute_failure_analysis(engagement_data),
                self._execute_exploration_decision(engagement_data),
                self._execute_feedback_orchestration(engagement_data),
                self._execute_validation_analysis(engagement_data),
                self._execute_pattern_discovery(engagement_data),
                return_exceptions=True,
            )

            # Unpack results (handle exceptions)
            (
                failure_analysis,
                exploration_decision,
                feedback_requests,
                validation_results,
                pattern_discovery,
            ) = reliability_tasks

            # Handle any service failures gracefully
            if isinstance(failure_analysis, Exception):
                self.logger.error(f"❌ Failure analysis failed: {failure_analysis}")
                failure_analysis = None

            if isinstance(exploration_decision, Exception):
                self.logger.error(
                    f"❌ Exploration decision failed: {exploration_decision}"
                )
                exploration_decision = None

            if isinstance(feedback_requests, Exception):
                self.logger.error(
                    f"❌ Feedback orchestration failed: {feedback_requests}"
                )
                feedback_requests = []

            if isinstance(validation_results, Exception):
                self.logger.error(
                    f"❌ Validation analysis failed: {validation_results}"
                )
                validation_results = []

            if isinstance(pattern_discovery, Exception):
                self.logger.error(f"❌ Pattern discovery failed: {pattern_discovery}")
                pattern_discovery = None

            # Calculate overall reliability score
            reliability_score = self._calculate_overall_reliability_score(
                failure_analysis,
                exploration_decision,
                feedback_requests,
                validation_results,
                pattern_discovery,
            )

            # Generate coordinated recommendations
            recommendations = self._generate_coordinated_recommendations(
                failure_analysis,
                exploration_decision,
                feedback_requests,
                validation_results,
                pattern_discovery,
            )

            # Create comprehensive assessment
            assessment = ReliabilityAssessmentContract(
                engagement_id=engagement_id,
                failure_analysis=failure_analysis,
                exploration_decision=exploration_decision,
                feedback_requests=feedback_requests or [],
                validation_results=validation_results or [],
                pattern_discovery=pattern_discovery,
                overall_reliability_score=reliability_score,
                recommendations=recommendations,
                assessment_timestamp=datetime.utcnow(),
                service_version="v5_modular",
            )

            self.logger.info(
                f"✅ Reliability assessment completed for engagement {engagement_id} (score: {reliability_score:.2f})"
            )

            return assessment

        except Exception as e:
            self.logger.error(f"❌ Reliability coordination failed: {e}")
            return self._create_fallback_assessment(engagement_data, str(e))

    async def _execute_failure_analysis(self, engagement_data: Dict[str, Any]):
        """Execute failure detection analysis"""
        cognitive_state = {
            "confidence_scores": engagement_data.get("confidence_scores", {}),
            "validation_results": engagement_data.get("validation_results", {}),
            "research_intelligence": engagement_data.get("research_intelligence", {}),
            "reasoning_steps": engagement_data.get("reasoning_steps", []),
            "engagement_id": engagement_data.get("engagement_id", "unknown"),
        }

        return await self.failure_detection.analyze_engagement_health(cognitive_state)

    async def _execute_exploration_decision(self, engagement_data: Dict[str, Any]):
        """Execute exploration strategy decision"""
        problem_analysis = {
            "engagement_id": engagement_data.get("engagement_id", "unknown"),
            "model_performance": engagement_data.get("model_performance", {}),
            "complexity_level": engagement_data.get("complexity_level", "medium"),
        }

        business_context = engagement_data.get("business_context", {})

        return await self.exploration_strategy.determine_exploration_strategy(
            problem_analysis, business_context
        )

    async def _execute_feedback_orchestration(self, engagement_data: Dict[str, Any]):
        """Execute feedback request generation"""
        user_context = {
            "feedback_history": engagement_data.get("user_feedback_history", {}),
            "partnership_tier": engagement_data.get("partnership_tier", "bronze"),
        }

        engagement_id = engagement_data.get("engagement_id", "unknown")

        return await self.feedback_orchestration.generate_feedback_requests(
            engagement_id, user_context
        )

    async def _execute_validation_analysis(self, engagement_data: Dict[str, Any]):
        """Execute LLM output validation"""
        llm_responses = engagement_data.get("llm_responses", [])
        research_base = engagement_data.get("research_base", [])
        context = engagement_data.get("business_context", {})
        context["engagement_id"] = engagement_data.get("engagement_id", "unknown")

        validation_results = []

        # Validate each LLM response
        for response in llm_responses:
            if isinstance(response, str):
                validation_result = await self.validation_engine.validate_llm_output(
                    response, context, research_base
                )
                validation_results.append(validation_result)

        return validation_results

    async def _execute_pattern_discovery(self, engagement_data: Dict[str, Any]):
        """Execute pattern discovery evaluation"""
        similar_engagements = engagement_data.get("similar_engagements", [])

        if len(similar_engagements) < 3:
            # Not enough data for pattern discovery
            return None

        return await self.pattern_governance.evaluate_for_pattern_discovery(
            engagement_data, similar_engagements
        )

    def _calculate_overall_reliability_score(
        self,
        failure_analysis,
        exploration_decision,
        feedback_requests,
        validation_results,
        pattern_discovery,
    ) -> float:
        """Calculate weighted overall reliability score from all service results"""
        score_components = {}

        # Failure Analysis Component
        if failure_analysis:
            failure_score = 1.0 - (len(failure_analysis.failure_modes) * 0.2)
            score_components["failure_analysis"] = max(0.0, failure_score)
        else:
            score_components["failure_analysis"] = 0.5  # Neutral when unavailable

        # Validation Results Component
        if validation_results:
            validation_scores = [
                1.0 if result.overall_passed else 0.3 for result in validation_results
            ]
            avg_validation_score = sum(validation_scores) / len(validation_scores)
            score_components["validation_results"] = avg_validation_score
        else:
            score_components["validation_results"] = (
                0.7  # Assume decent when unavailable
            )

        # Exploration Decision Component
        if exploration_decision:
            # Higher score for well-reasoned exploration decisions
            exploration_score = 0.8 if exploration_decision.should_explore else 0.9
            score_components["exploration_decision"] = exploration_score
        else:
            score_components["exploration_decision"] = 0.8

        # Feedback Quality Component
        if feedback_requests:
            # Score based on feedback sophistication
            feedback_score = min(1.0, len(feedback_requests) * 0.3 + 0.4)
            score_components["feedback_quality"] = feedback_score
        else:
            score_components["feedback_quality"] = 0.6

        # Pattern Discovery Component
        if pattern_discovery:
            pattern_score = min(
                1.0, pattern_discovery.confidence_score + 0.1
            )  # Bonus for discovery
            score_components["pattern_discovery"] = pattern_score
        else:
            score_components["pattern_discovery"] = 0.7  # Neutral when no patterns

        # Calculate weighted average
        total_score = 0.0
        for component, score in score_components.items():
            weight = self.assessment_weights[component]
            total_score += score * weight

        return min(1.0, total_score)

    def _generate_coordinated_recommendations(
        self,
        failure_analysis,
        exploration_decision,
        feedback_requests,
        validation_results,
        pattern_discovery,
    ) -> List[str]:
        """Generate coordinated recommendations from all service results"""
        recommendations = []

        # Failure Analysis Recommendations
        if failure_analysis and failure_analysis.recovery_steps:
            recommendations.extend(
                failure_analysis.recovery_steps[:2]
            )  # Top 2 recovery steps

        # Exploration Recommendations
        if exploration_decision and exploration_decision.should_explore:
            recommendations.append(
                f"Execute exploration strategy: {exploration_decision.exploration_strategy}"
            )

        # Validation Recommendations
        if validation_results:
            for result in validation_results:
                if not result.overall_passed and result.issues_detected:
                    recommendations.append(
                        f"Address validation issues: {result.issues_detected[0]}"
                    )
                    break  # Only add one validation recommendation to avoid overwhelming

        # Pattern Discovery Recommendations
        if pattern_discovery:
            recommendations.append(
                f"Monitor emergent pattern: {pattern_discovery.pattern_name}"
            )

        # Feedback Recommendations
        if feedback_requests:
            tier_counts = {}
            for request in feedback_requests:
                tier = request.feedback_tier
                tier_counts[tier] = tier_counts.get(tier, 0) + 1

            if tier_counts:
                most_common_tier = max(tier_counts.keys(), key=lambda k: tier_counts[k])
                recommendations.append(
                    f"Prioritize {most_common_tier} feedback collection"
                )

        # Add coordination-level recommendations
        if len(recommendations) > 3:
            recommendations.append(
                "Implement recommendations in priority order to avoid overwhelming users"
            )

        return recommendations[:5]  # Limit to top 5 recommendations

    def _create_fallback_assessment(
        self, engagement_data: Dict[str, Any], error_msg: str
    ) -> ReliabilityAssessmentContract:
        """Create fallback assessment when coordination fails"""
        return ReliabilityAssessmentContract(
            engagement_id=engagement_data.get("engagement_id", "unknown"),
            failure_analysis=None,
            exploration_decision=None,
            feedback_requests=[],
            validation_results=[],
            pattern_discovery=None,
            overall_reliability_score=0.3,  # Low score for coordination failure
            recommendations=[
                f"Reliability coordination failed: {error_msg}",
                "Retry reliability assessment",
                "Contact technical support if issue persists",
            ],
            assessment_timestamp=datetime.utcnow(),
            service_version="v5_modular_fallback",
        )

    async def get_cluster_health(self) -> Dict[str, Any]:
        """Get health status of entire reliability services cluster"""
        try:
            # Get health from all services in parallel
            health_tasks = await asyncio.gather(
                self.failure_detection.get_service_health(),
                self.exploration_strategy.get_service_health(),
                self.feedback_orchestration.get_service_health(),
                self.validation_engine.get_service_health(),
                self.pattern_governance.get_service_health(),
                return_exceptions=True,
            )

            service_healths = []
            failed_services = []

            service_names = [
                "FailureDetectionService",
                "ExplorationStrategyService",
                "FeedbackOrchestrationService",
                "ValidationEngineService",
                "PatternGovernanceService",
            ]

            for i, health_result in enumerate(health_tasks):
                if isinstance(health_result, Exception):
                    failed_services.append(service_names[i])
                else:
                    service_healths.append(health_result)

            # Calculate cluster health
            healthy_services = len(service_healths)
            total_services = len(service_names)
            cluster_health_score = healthy_services / total_services

            cluster_status = (
                "healthy"
                if cluster_health_score >= 0.8
                else "degraded" if cluster_health_score >= 0.6 else "unhealthy"
            )

            return {
                "cluster_name": "ReliabilityServicesCluster",
                "coordinator_status": "healthy",
                "cluster_status": cluster_status,
                "cluster_health_score": cluster_health_score,
                "total_services": total_services,
                "healthy_services": healthy_services,
                "failed_services": failed_services,
                "service_healths": service_healths,
                "assessment_weights": self.assessment_weights,
                "last_health_check": datetime.utcnow().isoformat(),
                "coordinator_version": "v5_modular",
            }

        except Exception as e:
            self.logger.error(f"❌ Cluster health check failed: {e}")
            return {
                "cluster_name": "ReliabilityServicesCluster",
                "coordinator_status": "error",
                "cluster_status": "unknown",
                "error": str(e),
                "last_health_check": datetime.utcnow().isoformat(),
            }

    async def get_service_health(self) -> Dict[str, Any]:
        """Get coordinator service health"""
        return {
            "service_name": "ReliabilityCoordinatorService",
            "status": "healthy",
            "version": "v5_modular",
            "capabilities": [
                "multi_service_coordination",
                "comprehensive_reliability_assessment",
                "weighted_scoring",
                "coordinated_recommendations",
                "cluster_health_monitoring",
            ],
            "coordinated_services": 5,
            "assessment_weights": self.assessment_weights,
            "last_health_check": datetime.utcnow().isoformat(),
        }


# Global service instance for dependency injection
_reliability_coordinator_service: Optional[ReliabilityCoordinatorService] = None


def get_reliability_coordinator_service() -> ReliabilityCoordinatorService:
    """Get or create global reliability coordinator service instance"""
    global _reliability_coordinator_service

    if _reliability_coordinator_service is None:
        _reliability_coordinator_service = ReliabilityCoordinatorService()

    return _reliability_coordinator_service
