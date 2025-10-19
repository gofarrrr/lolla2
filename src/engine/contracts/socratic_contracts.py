"""
Socratic Engine Contracts - METIS V5 API Contract Registry
Data contracts for Socratic Cognitive Forge component

This contract addresses the Dossier X failure:
- Missing 'generate_progressive_questions' method
- Inconsistent question data structures
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from .common_contracts import EngagementContext, ProcessingMetrics


@dataclass
class Question:
    """Individual question structure"""

    question_id: str
    text: str
    tier: str  # "essential", "strategic", "expert"
    reasoning: Optional[str] = None
    expected_improvement: Optional[str] = None
    category: Optional[str] = None
    is_required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "question_id": self.question_id,
            "text": self.text,
            "tier": self.tier,
            "reasoning": self.reasoning,
            "expected_improvement": self.expected_improvement,
            "category": self.category,
            "is_required": self.is_required,
        }


@dataclass
class QuestionSet:
    """Set of questions for a specific tier"""

    tier: str  # "essential", "strategic", "expert"
    title: str
    description: str
    quality_target: int  # Target quality percentage (60, 85, 95)
    questions: List[Question] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "tier": self.tier,
            "title": self.title,
            "description": self.description,
            "quality_target": self.quality_target,
            "questions": [q.to_dict() for q in self.questions],
        }


@dataclass
class SocraticRequest:
    """Request contract for Socratic Cognitive Forge"""

    engagement_context: EngagementContext
    force_real_llm_call: bool = False
    max_questions_per_tier: int = 3
    tiers_to_generate: List[str] = field(
        default_factory=lambda: ["essential", "strategic", "expert"]
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "engagement_context": self.engagement_context.to_dict(),
            "force_real_llm_call": self.force_real_llm_call,
            "max_questions_per_tier": self.max_questions_per_tier,
            "tiers_to_generate": self.tiers_to_generate,
        }


@dataclass
class SocraticResponse:
    """Response contract for Socratic Cognitive Forge"""

    success: bool
    engagement_id: str
    problem_statement: str
    question_sets: List[QuestionSet] = field(default_factory=list)
    processing_metrics: Optional[ProcessingMetrics] = None
    is_real_llm_call: bool = False
    total_questions_generated: int = 0
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "success": self.success,
            "engagement_id": self.engagement_id,
            "problem_statement": self.problem_statement,
            "question_sets": [qs.to_dict() for qs in self.question_sets],
            "processing_metrics": (
                self.processing_metrics.to_dict() if self.processing_metrics else None
            ),
            "is_real_llm_call": self.is_real_llm_call,
            "total_questions_generated": self.total_questions_generated,
            "error_message": self.error_message,
        }


# Standard method signatures that all Socratic components must implement
class SocraticEngineInterface:
    """Abstract interface that all Socratic engines must implement"""

    async def generate_progressive_questions(
        self, request: SocraticRequest
    ) -> SocraticResponse:
        """
        PRIMARY CONTRACT METHOD - This is what failed in Dossier X

        All Socratic engines MUST implement this exact signature.
        No exceptions. No variations.
        """
        raise NotImplementedError(
            "Socratic engine must implement generate_progressive_questions"
        )

    async def health_check(self) -> Dict[str, Any]:
        """Standard health check method"""
        raise NotImplementedError("Socratic engine must implement health_check")
