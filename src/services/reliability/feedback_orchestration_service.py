"""
METIS Feedback Orchestration Service
Part of Reliability Services Cluster - Focused on multi-tier feedback collection with incentives

Extracted from vulnerability_solutions.py FeedbackManager during Phase 5 decomposition.
Single Responsibility: Orchestrate feedback collection with partnership tiers and incentive management.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from src.services.contracts.reliability_contracts import (
    FeedbackRequestContract,
    IFeedbackOrchestrationService,
    FeedbackTier,
    PartnershipTier,
    FeedbackIncentive,
)


class FeedbackOrchestrationService(IFeedbackOrchestrationService):
    """
    Focused service for managing multi-tier feedback collection with smart incentives
    Clean extraction from vulnerability_solutions.py FeedbackManager
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Partnership tier requirements and benefits
        self.tier_requirements = {
            PartnershipTier.BRONZE: {"feedback_count": 5, "quality_score": 0.6},
            PartnershipTier.SILVER: {"feedback_count": 15, "quality_score": 0.7},
            PartnershipTier.GOLD: {"feedback_count": 40, "quality_score": 0.8},
            PartnershipTier.PLATINUM: {"feedback_count": 100, "quality_score": 0.85},
        }

        # Incentive structures by tier
        self.incentive_structures = {
            PartnershipTier.BRONZE: FeedbackIncentive(
                tier=PartnershipTier.BRONZE,
                credits_earned=10,
                discount_percentage=5.0,
                premium_access=False,
                co_innovation_access=False,
            ),
            PartnershipTier.SILVER: FeedbackIncentive(
                tier=PartnershipTier.SILVER,
                credits_earned=25,
                discount_percentage=10.0,
                premium_access=True,
                co_innovation_access=False,
            ),
            PartnershipTier.GOLD: FeedbackIncentive(
                tier=PartnershipTier.GOLD,
                credits_earned=50,
                discount_percentage=15.0,
                premium_access=True,
                co_innovation_access=True,
            ),
            PartnershipTier.PLATINUM: FeedbackIncentive(
                tier=PartnershipTier.PLATINUM,
                credits_earned=100,
                discount_percentage=25.0,
                premium_access=True,
                co_innovation_access=True,
                revenue_sharing=True,
            ),
        }

        self.logger.info("📊 FeedbackOrchestrationService initialized")

    async def generate_feedback_requests(
        self, engagement_id: str, user_context: Dict[str, Any]
    ) -> List[FeedbackRequestContract]:
        """
        Core service method: Generate multi-tier feedback requests with appropriate incentives
        Clean, focused implementation with single responsibility
        """
        try:
            # Determine user's current partnership tier
            partnership_tier = await self.determine_partnership_tier(
                user_context.get("feedback_history", {})
            )

            # Generate feedback requests for different time horizons
            feedback_requests = []

            # Immediate feedback (within hours)
            immediate_request = self._create_immediate_feedback_request(
                engagement_id, partnership_tier, user_context
            )
            feedback_requests.append(immediate_request)

            # Short-term feedback (1-2 weeks)
            short_term_request = self._create_short_term_feedback_request(
                engagement_id, partnership_tier, user_context
            )
            feedback_requests.append(short_term_request)

            # Long-term feedback (6-12 months) for higher tier users
            if partnership_tier in [PartnershipTier.GOLD, PartnershipTier.PLATINUM]:
                long_term_request = self._create_long_term_feedback_request(
                    engagement_id, partnership_tier, user_context
                )
                feedback_requests.append(long_term_request)

            return feedback_requests

        except Exception as e:
            self.logger.error(f"❌ Feedback request generation failed: {e}")
            # Return basic fallback request
            return [self._create_fallback_feedback_request(engagement_id, str(e))]

    async def determine_partnership_tier(
        self, user_feedback_history: Dict[str, Any]
    ) -> str:
        """
        Determine user's partnership tier based on feedback history and quality
        """
        try:
            feedback_count = user_feedback_history.get("total_feedback", 0)
            average_quality = user_feedback_history.get("average_quality_score", 0.0)

            # Check tier requirements in descending order
            for tier in [
                PartnershipTier.PLATINUM,
                PartnershipTier.GOLD,
                PartnershipTier.SILVER,
                PartnershipTier.BRONZE,
            ]:
                requirements = self.tier_requirements[tier]
                if (
                    feedback_count >= requirements["feedback_count"]
                    and average_quality >= requirements["quality_score"]
                ):
                    return tier.value

            # Default to Bronze for new users
            return PartnershipTier.BRONZE.value

        except Exception as e:
            self.logger.error(f"❌ Partnership tier determination failed: {e}")
            return PartnershipTier.BRONZE.value

    def _create_immediate_feedback_request(
        self, engagement_id: str, partnership_tier: str, user_context: Dict[str, Any]
    ) -> FeedbackRequestContract:
        """Create immediate feedback request (within hours)"""

        tier_enum = PartnershipTier(partnership_tier)
        incentive = self.incentive_structures[tier_enum]

        questions = [
            "How relevant was this strategic analysis to your current business challenges?",
            "What key insights stood out as most actionable for your organization?",
            "Which aspects of the analysis would you like explored in more depth?",
        ]

        # Add tier-specific questions
        if tier_enum in [PartnershipTier.GOLD, PartnershipTier.PLATINUM]:
            questions.append(
                "What alternative strategic approaches would you consider based on this analysis?"
            )

        return FeedbackRequestContract(
            engagement_id=engagement_id,
            feedback_tier=FeedbackTier.IMMEDIATE.value,
            questions=questions,
            incentive_offered={
                "credits": incentive.credits_earned,
                "discount_percentage": incentive.discount_percentage,
                "premium_access": incentive.premium_access,
                "description": f"{partnership_tier} tier benefits applied",
            },
            deadline=datetime.utcnow() + timedelta(hours=24),
            estimated_completion_minutes=5,
            request_timestamp=datetime.utcnow(),
            service_version="v5_modular",
        )

    def _create_short_term_feedback_request(
        self, engagement_id: str, partnership_tier: str, user_context: Dict[str, Any]
    ) -> FeedbackRequestContract:
        """Create short-term feedback request (1-2 weeks)"""

        tier_enum = PartnershipTier(partnership_tier)
        incentive = self.incentive_structures[tier_enum]

        questions = [
            "How have you applied the strategic recommendations from this analysis?",
            "What results have you observed from implementing the suggested approaches?",
            "Were there any unexpected challenges or opportunities that emerged?",
            "How would you adjust the implementation strategy based on early results?",
        ]

        return FeedbackRequestContract(
            engagement_id=engagement_id,
            feedback_tier=FeedbackTier.SHORT_TERM.value,
            questions=questions,
            incentive_offered={
                "credits": incentive.credits_earned
                * 2,  # Double credits for implementation feedback
                "discount_percentage": incentive.discount_percentage,
                "premium_access": incentive.premium_access,
                "description": f"Implementation feedback bonus - {partnership_tier} tier",
            },
            deadline=datetime.utcnow() + timedelta(weeks=3),
            estimated_completion_minutes=15,
            request_timestamp=datetime.utcnow(),
            service_version="v5_modular",
        )

    def _create_long_term_feedback_request(
        self, engagement_id: str, partnership_tier: str, user_context: Dict[str, Any]
    ) -> FeedbackRequestContract:
        """Create long-term feedback request (6-12 months) for Gold/Platinum tiers"""

        tier_enum = PartnershipTier(partnership_tier)
        incentive = self.incentive_structures[tier_enum]

        questions = [
            "What were the long-term business outcomes from the strategic recommendations?",
            "How did market conditions affect the implementation of the strategy?",
            "What would you change about the original strategic approach given what you know now?",
            "How has your perspective on this strategic challenge evolved over time?",
            "What patterns do you see across multiple strategic analyses we've provided?",
        ]

        # Add revenue sharing opportunity for Platinum tier
        incentive_details = {
            "credits": incentive.credits_earned
            * 5,  # 5x credits for long-term insights
            "discount_percentage": incentive.discount_percentage,
            "premium_access": incentive.premium_access,
            "co_innovation_access": incentive.co_innovation_access,
            "description": f"Long-term strategic insights - {partnership_tier} tier",
        }

        if tier_enum == PartnershipTier.PLATINUM:
            incentive_details["revenue_sharing"] = True
            incentive_details["description"] += " with revenue sharing opportunity"

        return FeedbackRequestContract(
            engagement_id=engagement_id,
            feedback_tier=FeedbackTier.LONG_TERM.value,
            questions=questions,
            incentive_offered=incentive_details,
            deadline=datetime.utcnow() + timedelta(days=365),
            estimated_completion_minutes=30,
            request_timestamp=datetime.utcnow(),
            service_version="v5_modular",
        )

    def _create_fallback_feedback_request(
        self, engagement_id: str, error_msg: str
    ) -> FeedbackRequestContract:
        """Create basic fallback feedback request when service fails"""
        return FeedbackRequestContract(
            engagement_id=engagement_id,
            feedback_tier=FeedbackTier.IMMEDIATE.value,
            questions=[
                "How would you rate the overall quality of this strategic analysis?"
            ],
            incentive_offered={
                "credits": 10,
                "discount_percentage": 5.0,
                "description": f"Basic feedback request (service error: {error_msg[:100]})",
            },
            deadline=datetime.utcnow() + timedelta(days=7),
            estimated_completion_minutes=2,
            request_timestamp=datetime.utcnow(),
            service_version="v5_modular_fallback",
        )

    async def calculate_proxy_metrics(
        self, engagement_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate proxy metrics that correlate with long-term business value
        Used to estimate feedback before actual user feedback is available
        """
        try:
            metrics = {}

            # Analysis depth proxy
            reasoning_steps = engagement_data.get("reasoning_steps", [])
            metrics["analysis_depth"] = min(
                1.0, len(reasoning_steps) / 8.0
            )  # Normalize to 8 expected steps

            # Research grounding proxy
            research_sources = engagement_data.get("research_sources", [])
            metrics["research_grounding"] = min(
                1.0, len(research_sources) / 5.0
            )  # Normalize to 5 expected sources

            # Confidence calibration proxy
            overall_confidence = engagement_data.get("confidence_scores", {}).get(
                "overall", 0.5
            )
            metrics["confidence_calibration"] = overall_confidence

            # Stakeholder consideration proxy
            stakeholder_analysis = engagement_data.get("stakeholder_analysis", [])
            metrics["stakeholder_consideration"] = min(
                1.0, len(stakeholder_analysis) / 4.0
            )  # Normalize to 4 stakeholders

            # Implementation feasibility proxy
            implementation_steps = engagement_data.get("implementation_steps", [])
            metrics["implementation_feasibility"] = min(
                1.0, len(implementation_steps) / 6.0
            )  # Normalize to 6 steps

            return metrics

        except Exception as e:
            self.logger.error(f"❌ Proxy metrics calculation failed: {e}")
            return {"error": 1.0}  # Signal that proxy calculation failed

    async def get_service_health(self) -> Dict[str, Any]:
        """Return service health status"""
        return {
            "service_name": "FeedbackOrchestrationService",
            "status": "healthy",
            "version": "v5_modular",
            "capabilities": [
                "multi_tier_feedback_generation",
                "partnership_tier_management",
                "incentive_structure_management",
                "proxy_metrics_calculation",
            ],
            "tier_structures": {
                "bronze": self.tier_requirements[PartnershipTier.BRONZE],
                "silver": self.tier_requirements[PartnershipTier.SILVER],
                "gold": self.tier_requirements[PartnershipTier.GOLD],
                "platinum": self.tier_requirements[PartnershipTier.PLATINUM],
            },
            "last_health_check": datetime.utcnow().isoformat(),
        }


# Global service instance for dependency injection
_feedback_orchestration_service: Optional[FeedbackOrchestrationService] = None


def get_feedback_orchestration_service() -> FeedbackOrchestrationService:
    """Get or create global feedback orchestration service instance"""
    global _feedback_orchestration_service

    if _feedback_orchestration_service is None:
        _feedback_orchestration_service = FeedbackOrchestrationService()

    return _feedback_orchestration_service
