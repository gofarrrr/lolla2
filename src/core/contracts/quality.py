"""
Cognitive Quality Assurance (CQA) Framework Models
==================================================

Defines the data structures for standardized quality scoring of cognitive outputs.
Part of Operation "Quality Assurance Framework" - METIS V5.3
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class QualityDimension(str, Enum):
    """RIVA quality dimensions for cognitive assessment."""

    RIGOR = "rigor"
    INSIGHT = "insight"
    VALUE = "value"
    ALIGNMENT = "alignment"


class RIVAScore(BaseModel):
    """
    Individual dimension score within the RIVA framework.

    Attributes:
        dimension: The quality dimension being scored
        score: Numerical score from 1 (poor) to 10 (excellent)
        rationale: Evidence-based justification for the score
        evidence: Specific examples supporting the score
    """

    dimension: QualityDimension
    score: int = Field(..., ge=1, le=10, description="Score from 1-10")
    rationale: str = Field(..., min_length=10, description="Justification for score")
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence")

    @field_validator("rationale")
    @classmethod
    def validate_rationale(cls, v: str) -> str:
        """Ensure rationale is substantive."""
        if len(v.split()) < 5:
            raise ValueError("Rationale must contain at least 5 words")
        return v


class CQA_Result(BaseModel):
    """
    Complete quality assessment result using the RIVA framework.

    Attributes:
        rigor: Score for analytical depth and logical consistency
        insight: Score for novel perspectives and creative thinking
        value: Score for practical utility and actionability
        alignment: Score for query responsiveness and goal achievement
        average_score: Computed average across all dimensions
        confidence: Rater's confidence in the assessment (0-1)
        metadata: Additional context about the assessment
    """

    rigor: RIVAScore
    insight: RIVAScore
    value: RIVAScore
    alignment: RIVAScore
    average_score: float = Field(default=0.0)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def model_post_init(self, __context):
        """Calculate average score after initialization."""
        if self.average_score == 0.0:
            scores = [
                self.rigor.score,
                self.insight.score,
                self.value.score,
                self.alignment.score,
            ]
            self.average_score = sum(scores) / len(scores)


class QualityAuditRequest(BaseModel):
    """
    Request structure for quality assessment.

    Attributes:
        system_prompt: The system instructions/context
        user_prompt: The user's query or request
        llm_response: The generated response to evaluate
        agent_name: Name of the agent that produced the response
        context: Additional context for evaluation
    """

    system_prompt: str
    user_prompt: str
    llm_response: str
    agent_name: Optional[str] = None
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    evaluation_focus: Optional[List[QualityDimension]] = None


class QualityBenchmark(BaseModel):
    """
    Aggregated quality metrics for benchmarking.

    Attributes:
        test_suite: Name of the test suite
        case_id: Unique identifier for the test case
        agent_scores: Per-agent quality scores
        pipeline_average: Average score across all agents
        timestamp: When the benchmark was run
    """

    test_suite: str
    case_id: str
    agent_scores: Dict[str, CQA_Result]
    pipeline_average: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    git_commit: Optional[str] = None

    def model_post_init(self, __context):
        """Calculate pipeline average if not provided."""
        if self.pipeline_average == 0.0 and self.agent_scores:
            averages = [result.average_score for result in self.agent_scores.values()]
            self.pipeline_average = sum(averages) / len(averages) if averages else 0.0
