"""
METIS V5 Transparency Engine Refactoring - Target #3
UserExpertiseService

Extracted from transparency_engine.py (1,902 lines → modular architecture)
Handles user expertise level assessment and adaptation

Single Responsibility: User expertise analysis and interaction pattern matching
"""

import os
import logging
from typing import Dict, Any

from src.models.transparency_models import UserExpertiseLevel, UserProfile


class UserExpertiseService:
    """Service for assessing and adapting to user expertise level"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        self.interaction_patterns = {
            UserExpertiseLevel.EXECUTIVE: {
                "typical_session_duration": 300,  # 5 minutes
                "prefers_summaries": True,
                "drill_down_frequency": "low",
                "focuses_on_outcomes": True,
            },
            UserExpertiseLevel.STRATEGIC: {
                "typical_session_duration": 900,  # 15 minutes
                "prefers_summaries": False,
                "drill_down_frequency": "medium",
                "focuses_on_methodology": True,
            },
            UserExpertiseLevel.ANALYTICAL: {
                "typical_session_duration": 1800,  # 30 minutes
                "prefers_summaries": False,
                "drill_down_frequency": "high",
                "focuses_on_evidence": True,
            },
            UserExpertiseLevel.TECHNICAL: {
                "typical_session_duration": 2700,  # 45 minutes
                "prefers_summaries": False,
                "drill_down_frequency": "very_high",
                "focuses_on_implementation": True,
            },
        }

    async def assess_expertise(self, user_profile: UserProfile) -> UserExpertiseLevel:
        """Assess user expertise based on interaction history"""

        # Cold start problem fix - no interaction history available
        if not user_profile.interaction_history:
            # No history - set default complexity threshold based on role
            default_thresholds = {
                "executive": float(
                    os.getenv("METIS_DEFAULT_THRESHOLD_EXECUTIVE", "0.4")
                ),
                "strategic": float(
                    os.getenv("METIS_DEFAULT_THRESHOLD_STRATEGIC", "0.6")
                ),
                "analytical": float(
                    os.getenv("METIS_DEFAULT_THRESHOLD_ANALYTICAL", "0.8")
                ),
                "technical": float(
                    os.getenv("METIS_DEFAULT_THRESHOLD_TECHNICAL", "1.0")
                ),
            }

            user_role = getattr(user_profile, "role", "analytical").lower()
            threshold = default_thresholds.get(
                user_role, default_thresholds["executive"]
            )  # Default to most conservative

            # Store the threshold in the user profile for future use
            if not hasattr(user_profile, "drop_off_complexity_threshold"):
                user_profile.drop_off_complexity_threshold = threshold
                self.logger.info(
                    f"🎯 Cold start: Set complexity threshold {threshold} for {user_role} role"
                )

            return user_profile.expertise_level  # Use declared level

        # Analyze interaction patterns from history
        recent_interactions = user_profile.interaction_history[
            -10:
        ]  # Last 10 interactions

        avg_session_duration = sum(
            interaction.get("duration_seconds", 0)
            for interaction in recent_interactions
        ) / len(recent_interactions)

        drill_down_frequency = sum(
            interaction.get("drill_down_count", 0)
            for interaction in recent_interactions
        ) / len(recent_interactions)

        # Match patterns to expertise levels
        best_match = user_profile.expertise_level
        best_score = 0

        for level, patterns in self.interaction_patterns.items():
            score = 0

            # Session duration match
            duration_diff = abs(
                avg_session_duration - patterns["typical_session_duration"]
            )
            score += max(0, 600 - duration_diff) / 600  # Normalize to 0-1

            # Drill-down frequency match
            if patterns["drill_down_frequency"] == "low" and drill_down_frequency < 2:
                score += 1
            elif (
                patterns["drill_down_frequency"] == "medium"
                and 2 <= drill_down_frequency < 5
            ):
                score += 1
            elif (
                patterns["drill_down_frequency"] == "high"
                and 5 <= drill_down_frequency < 10
            ):
                score += 1
            elif (
                patterns["drill_down_frequency"] == "very_high"
                and drill_down_frequency >= 10
            ):
                score += 1

            if score > best_score:
                best_score = score
                best_match = level

        return best_match

    def get_interaction_pattern(
        self, expertise_level: UserExpertiseLevel
    ) -> Dict[str, Any]:
        """Get interaction pattern for specific expertise level"""
        return self.interaction_patterns.get(
            expertise_level, self.interaction_patterns[UserExpertiseLevel.ANALYTICAL]
        )

    def calculate_expertise_score(self, user_profile: UserProfile) -> float:
        """Calculate expertise score based on interaction patterns"""
        if not user_profile.interaction_history:
            return 0.5  # Neutral score for new users

        recent_interactions = user_profile.interaction_history[
            -5:
        ]  # Last 5 interactions

        # Calculate complexity of content accessed
        complexity_scores = [
            interaction.get("content_complexity", 0.5)
            for interaction in recent_interactions
        ]

        avg_complexity = (
            sum(complexity_scores) / len(complexity_scores)
            if complexity_scores
            else 0.5
        )

        # Calculate engagement depth
        avg_drill_downs = sum(
            interaction.get("drill_down_count", 0)
            for interaction in recent_interactions
        ) / len(recent_interactions)

        # Normalize drill-down count to 0-1 scale
        engagement_score = min(1.0, avg_drill_downs / 10)

        # Combined expertise score
        expertise_score = (avg_complexity * 0.7) + (engagement_score * 0.3)

        return max(0.0, min(1.0, expertise_score))

    def recommend_transparency_level(self, user_profile: UserProfile) -> str:
        """Recommend transparency level based on expertise assessment"""
        expertise_score = self.calculate_expertise_score(user_profile)

        if expertise_score < 0.3:
            return "executive_summary"
        elif expertise_score < 0.6:
            return "strategic_overview"
        elif expertise_score < 0.8:
            return "detailed_analysis"
        else:
            return "technical_deep_dive"


# Factory function for service creation
def get_user_expertise_service() -> UserExpertiseService:
    """Factory function to create UserExpertiseService instance"""
    return UserExpertiseService()
