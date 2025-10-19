"""
Enhanced Query Contracts - Standardized Definition for METIS V5
=================================================================

This provides a single, authoritative EnhancedQuery definition that all components must use.
Fixes the import chain issues by providing a consistent contract.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum


class QueryComplexity(Enum):
    """Query complexity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXPERT = "expert"


class QueryIntent(Enum):
    """Query intent categories"""

    STRATEGIC = "strategic"
    OPERATIONAL = "operational"
    ANALYTICAL = "analytical"
    INNOVATION = "innovation"
    DIAGNOSTIC = "diagnostic"


@dataclass
class UserResponse:
    """User response to a clarifying question"""

    question_id: str
    question_text: str
    user_answer: str
    confidence_level: Optional[str] = None
    follow_up_needed: bool = False


@dataclass
class EnhancedQuery:
    """
    Enhanced query built from original statement + Socratic responses.

    This is the SINGLE AUTHORITATIVE definition for EnhancedQuery in METIS V5.
    All components must use this exact structure.
    """

    # Core fields (required)
    original_statement: str
    enhanced_statement: str
    confidence_score: float

    # Context fields
    context_enrichment: Optional[Dict[str, Any]] = None
    engagement_id: Optional[str] = None

    # Quality tracking
    quality_level: str = "medium"  # For PSA compatibility (was int, now string)

    # Socratic engine fields
    user_responses: List[UserResponse] = field(default_factory=list)

    # Classification fields (for advanced routing)
    complexity_level: Optional[QueryComplexity] = None
    query_intent: Optional[QueryIntent] = None

    # Metadata
    processing_timestamp: datetime = field(default_factory=datetime.now)
    source_engine: str = "socratic_cognitive_forge"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "original_statement": self.original_statement,
            "enhanced_statement": self.enhanced_statement,
            "confidence_score": self.confidence_score,
            "context_enrichment": self.context_enrichment,
            "engagement_id": self.engagement_id,
            "quality_level": self.quality_level,
            "user_responses": [
                {
                    "question_id": ur.question_id,
                    "question_text": ur.question_text,
                    "user_answer": ur.user_answer,
                    "confidence_level": ur.confidence_level,
                    "follow_up_needed": ur.follow_up_needed,
                }
                for ur in self.user_responses
            ],
            "complexity_level": (
                self.complexity_level.value if self.complexity_level else None
            ),
            "query_intent": self.query_intent.value if self.query_intent else None,
            "processing_timestamp": self.processing_timestamp.isoformat(),
            "source_engine": self.source_engine,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnhancedQuery":
        """Create from dictionary"""
        user_responses = []
        for ur_data in data.get("user_responses", []):
            user_responses.append(
                UserResponse(
                    question_id=ur_data["question_id"],
                    question_text=ur_data["question_text"],
                    user_answer=ur_data["user_answer"],
                    confidence_level=ur_data.get("confidence_level"),
                    follow_up_needed=ur_data.get("follow_up_needed", False),
                )
            )

        return cls(
            original_statement=data["original_statement"],
            enhanced_statement=data["enhanced_statement"],
            confidence_score=data["confidence_score"],
            context_enrichment=data.get("context_enrichment"),
            engagement_id=data.get("engagement_id"),
            quality_level=data.get("quality_level", "medium"),
            user_responses=user_responses,
            complexity_level=(
                QueryComplexity(data["complexity_level"])
                if data.get("complexity_level")
                else None
            ),
            query_intent=(
                QueryIntent(data["query_intent"]) if data.get("query_intent") else None
            ),
            processing_timestamp=datetime.fromisoformat(
                data.get("processing_timestamp", datetime.now().isoformat())
            ),
            source_engine=data.get("source_engine", "socratic_cognitive_forge"),
        )

    # Legacy compatibility methods
    @property
    def original_query(self) -> str:
        """Alias for original_statement (legacy compatibility)"""
        return self.original_statement

    @property
    def enhanced_query(self) -> str:
        """Alias for enhanced_statement (legacy compatibility)"""
        return self.enhanced_statement


# Export all necessary types
__all__ = ["EnhancedQuery", "UserResponse", "QueryComplexity", "QueryIntent"]
