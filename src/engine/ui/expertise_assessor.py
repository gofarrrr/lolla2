"""
METIS User Expertise Assessor
Progressive transparency module for user expertise assessment

Assesses and adapts to user expertise level through interaction pattern
analysis and behavioral adaptation.
"""

from src.models.transparency_models import UserExpertiseLevel, UserProfile


class UserExpertiseAssessor:
    """Assesses and adapts to user expertise level"""

    def __init__(self):
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

        if not user_profile.interaction_history:
            return user_profile.expertise_level  # Use declared level

        # Analyze interaction patterns
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
