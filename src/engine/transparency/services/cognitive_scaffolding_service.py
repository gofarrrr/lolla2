"""
METIS V5 Transparency Engine Refactoring - Target #3
CognitiveScaffoldingService

Extracted from transparency_engine.py (1,902 lines → modular architecture)
Handles cognitive scaffolding and load management

Single Responsibility: Cognitive load assessment and scaffolding strategies
"""

from typing import Dict, List, Any

from src.engine.models.data_contracts import ReasoningStep, MentalModelDefinition
from src.models.transparency_models import (
    TransparencyLayer,
    UserExpertiseLevel,
    CognitiveLoadLevel,
    UserProfile,
)


class CognitiveScaffoldingService:
    """Service for cognitive scaffolding and load management"""

    def __init__(self):
        self.complexity_thresholds = {
            CognitiveLoadLevel.LOW: 500,  # characters
            CognitiveLoadLevel.MEDIUM: 1500,
            CognitiveLoadLevel.HIGH: 3000,
            CognitiveLoadLevel.OVERWHELMING: 5000,
        }

        self.expertise_adaptations = {
            UserExpertiseLevel.EXECUTIVE: {
                "default_layer": TransparencyLayer.EXECUTIVE_SUMMARY,
                "max_reasoning_steps": 3,
                "show_technical_details": False,
                "emphasize_business_impact": True,
            },
            UserExpertiseLevel.STRATEGIC: {
                "default_layer": TransparencyLayer.REASONING_OVERVIEW,
                "max_reasoning_steps": 7,
                "show_technical_details": False,
                "emphasize_methodology": True,
            },
            UserExpertiseLevel.ANALYTICAL: {
                "default_layer": TransparencyLayer.DETAILED_AUDIT_TRAIL,
                "max_reasoning_steps": 15,
                "show_technical_details": True,
                "emphasize_evidence": True,
            },
            UserExpertiseLevel.TECHNICAL: {
                "default_layer": TransparencyLayer.TECHNICAL_EXECUTION,
                "max_reasoning_steps": 25,
                "show_technical_details": True,
                "emphasize_implementation": True,
            },
        }

    async def assess_cognitive_load(
        self,
        content: str,
        reasoning_steps: List[ReasoningStep],
        mental_models: List[MentalModelDefinition],
    ) -> CognitiveLoadLevel:
        """Assess cognitive load of content"""

        # Content length assessment
        content_length = len(content)
        step_count = len(reasoning_steps)
        model_count = len(mental_models)

        # Calculate complexity score
        complexity_score = (
            content_length
            + (step_count * 200)  # Each reasoning step adds complexity
            + (model_count * 150)  # Each mental model adds complexity
        )

        # Determine cognitive load level
        if complexity_score < self.complexity_thresholds[CognitiveLoadLevel.LOW]:
            return CognitiveLoadLevel.LOW
        elif complexity_score < self.complexity_thresholds[CognitiveLoadLevel.MEDIUM]:
            return CognitiveLoadLevel.MEDIUM
        elif complexity_score < self.complexity_thresholds[CognitiveLoadLevel.HIGH]:
            return CognitiveLoadLevel.HIGH
        else:
            return CognitiveLoadLevel.OVERWHELMING

    async def apply_scaffolding(
        self,
        content: str,
        user_profile: UserProfile,
        cognitive_load: CognitiveLoadLevel,
    ) -> Dict[str, Any]:
        """Apply cognitive scaffolding based on user profile and load"""

        scaffolding = {
            "chunking_strategy": "linear",
            "progressive_hints": [],
            "contextual_assistance": [],
            "navigation_aids": [],
            "complexity_reduction": [],
        }

        expertise_config = self.expertise_adaptations[user_profile.expertise_level]

        # Chunking strategy based on cognitive load
        if cognitive_load in [CognitiveLoadLevel.HIGH, CognitiveLoadLevel.OVERWHELMING]:
            scaffolding["chunking_strategy"] = "hierarchical"
            scaffolding["progressive_hints"] = [
                "Content is complex - consider starting with key insights",
                "Use drill-down navigation to explore details gradually",
                "Focus on one reasoning step at a time",
            ]

        # Contextual assistance based on expertise
        if not expertise_config["show_technical_details"]:
            scaffolding["contextual_assistance"].append(
                {
                    "type": "tooltip",
                    "content": "Technical details hidden - click to reveal",
                    "trigger": "technical_section",
                }
            )

        # Navigation aids
        scaffolding["navigation_aids"] = [
            {
                "type": "progress_indicator",
                "description": "Shows your position in the reasoning process",
            },
            {
                "type": "complexity_toggle",
                "description": "Adjust detail level to match your needs",
            },
        ]

        # Complexity reduction strategies
        if cognitive_load == CognitiveLoadLevel.OVERWHELMING:
            scaffolding["complexity_reduction"] = [
                "auto_summarize_long_sections",
                "hide_supporting_evidence_by_default",
                "group_similar_reasoning_steps",
                "show_only_high_confidence_insights",
            ]

        return scaffolding

    def get_expertise_config(
        self, expertise_level: UserExpertiseLevel
    ) -> Dict[str, Any]:
        """Get configuration for specific expertise level"""
        return self.expertise_adaptations.get(
            expertise_level, self.expertise_adaptations[UserExpertiseLevel.ANALYTICAL]
        )

    def get_complexity_threshold(self, load_level: CognitiveLoadLevel) -> int:
        """Get complexity threshold for specific load level"""
        return self.complexity_thresholds.get(
            load_level, self.complexity_thresholds[CognitiveLoadLevel.MEDIUM]
        )


# Factory function for service creation
def get_cognitive_scaffolding_service() -> CognitiveScaffoldingService:
    """Factory function to create CognitiveScaffoldingService instance"""
    return CognitiveScaffoldingService()
