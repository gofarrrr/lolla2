#!/usr/bin/env python3
"""
Sequential Query Enhancement Bridge Module

This module provides a bridge to maintain backward compatibility with existing imports.
It re-exports classes from their actual locations throughout the system.

This file was created to fix the broken import chain in the METIS system where
multiple modules import from 'src.sequential_query_enhancement' but the module
didn't exist. Rather than updating all imports, this bridge maintains compatibility.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List

# Import classes from their actual locations
from src.simple_query_enhancer import (
    EngagementBrief,
    SimpleQueryEnhancementResult as SequentialQueryEnhancementResult,
    SimpleQueryEnhancer as SequentialQueryEnhancer,
)

from src.engine.models.data_contracts import ClarificationQuestion

from src.engine.api.engagement.models import QuestionTier

# Create missing classes that are expected by the importing modules


class EngagementType(str, Enum):
    """Engagement type classification for query enhancement"""

    STRATEGIC_ANALYSIS = "strategic_analysis"
    PROBLEM_SOLVING = "problem_solving"
    IMPLEMENTATION_PLANNING = "implementation_planning"
    OPTIMIZATION = "optimization"
    INNOVATION = "innovation"
    GENERAL_CONSULTING = "general_consulting"


@dataclass
class TieredQuestions:
    """Tiered questions structure for progressive clarification"""

    essential_questions: List[ClarificationQuestion] = field(default_factory=list)
    expert_questions: List[ClarificationQuestion] = field(default_factory=list)
    estimated_time_minutes: int = 5
    total_questions: int = 0

    def __post_init__(self):
        """Calculate total questions after initialization"""
        self.total_questions = len(self.essential_questions) + len(
            self.expert_questions
        )

    def get_all_questions(self) -> List[ClarificationQuestion]:
        """Get all questions in priority order (essential first)"""
        return self.essential_questions + self.expert_questions

    def get_questions_by_tier(self, tier: QuestionTier) -> List[ClarificationQuestion]:
        """Get questions filtered by tier"""
        if tier == QuestionTier.ESSENTIAL:
            return self.essential_questions
        elif tier == QuestionTier.EXPERT:
            return self.expert_questions
        else:
            return []


# Export all classes for backward compatibility
__all__ = [
    "EngagementBrief",
    "ClarificationQuestion",
    "TieredQuestions",
    "SequentialQueryEnhancementResult",
    "SequentialQueryEnhancer",
    "QuestionTier",
    "EngagementType",
]
