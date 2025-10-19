"""
Data models for Pyramid Principle Engine
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from src.engine.models.data_contracts import ConfidenceLevel
from .enums import PyramidLevel, ArgumentType, DeliverableType


@dataclass
class PyramidNode:
    """Node in the pyramid structure"""

    node_id: UUID = field(default_factory=uuid4)
    level: PyramidLevel = PyramidLevel.SUPPORTING_POINTS
    content: str = ""
    children: List["PyramidNode"] = field(default_factory=list)
    parent_id: Optional[UUID] = None

    # Metadata
    argument_type: ArgumentType = ArgumentType.INDUCTIVE
    evidence_strength: float = 0.0  # 0-1 scale
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    supporting_data: List[Dict] = field(default_factory=list)

    # Quality metrics
    mece_score: float = 0.0  # MECE compliance
    clarity_score: float = 0.0  # Communication clarity
    persuasion_score: float = 0.0  # Persuasive strength

    # Sprint 2.1: Context Intelligence metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_child(self, child: "PyramidNode") -> None:
        """Add child node and set parent relationship"""
        child.parent_id = self.node_id
        self.children.append(child)

    def get_text_content(self) -> str:
        """Get text content with proper formatting"""
        if self.level == PyramidLevel.GOVERNING_THOUGHT:
            return f"**{self.content}**"
        elif self.level == PyramidLevel.KEY_LINES:
            return f"• {self.content}"
        else:
            return f"  - {self.content}"


@dataclass
class ExecutiveDeliverable:
    """Executive-ready deliverable structure"""

    deliverable_id: UUID = field(default_factory=uuid4)
    type: DeliverableType = DeliverableType.EXECUTIVE_SUMMARY
    title: str = ""

    # Pyramid structure
    pyramid_structure: Optional[PyramidNode] = None

    # Content sections
    executive_summary: str = ""
    key_recommendations: List[str] = field(default_factory=list)
    supporting_analysis: Dict[str, Any] = field(default_factory=dict)
    implementation_roadmap: Dict[str, Any] = field(default_factory=dict)
    appendices: List[Dict] = field(default_factory=list)

    # Quality metrics
    partner_ready_score: float = 0.0  # 0-1 scale for partner readiness
    structure_quality: float = 0.0  # Pyramid structure quality
    content_quality: float = 0.0  # Content quality and clarity
    persuasiveness: float = 0.0  # Persuasive impact

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    engagement_id: Optional[UUID] = None
    author: str = "pyramid_engine"
    review_status: str = "draft"
