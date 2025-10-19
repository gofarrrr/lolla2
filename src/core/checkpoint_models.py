"""
Iteration Engine - Checkpoint Models
====================================

ARCHITECTURAL PRINCIPLE: Separate "State" from "Log" + Immutable Analysis Branches

This module implements the refined checkpoint models that store only essential state
needed for resuming pipeline execution, not the entire context history.

The UnifiedContextStream remains the immutable log of what happened.
Checkpoints are lightweight snapshots of state at specific pipeline stages.
"""

from pydantic import BaseModel, Field, validator, ConfigDict
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4


class PipelineStage(str, Enum):
    """Pipeline stages for checkpoint management - Operation Genesis: 10-Stage Golden Master"""

    SOCRATIC_QUESTIONS = "socratic_questions"
    # Legacy alias for test compatibility
    QUERY_ENHANCEMENT = "socratic_questions"
    PROBLEM_STRUCTURING = "problem_structuring"
    INTERACTION_SWEEP = "interaction_sweep"
    HYBRID_DATA_RESEARCH = "hybrid_data_research"  # Operation Oracle - Stage 4
    CONSULTANT_SELECTION = "consultant_selection"
    SYNERGY_PROMPTING = "synergy_prompting"
    PARALLEL_ANALYSIS = "parallel_analysis"
    DEVILS_ADVOCATE = "devils_advocate"
    SENIOR_ADVISOR = "senior_advisor"
    ARBITRATION_CAPTURE = "arbitration_capture"  # Final persistence - Stage 10
    # Legacy alias for test compatibility
    ARBITRATION = "arbitration_capture"
    # Legacy stage supported for compatibility only
    CALIBRATION = "calibration"
    COMPLETED = "completed"

    def get_next_stage(self) -> Optional["PipelineStage"]:
        """Get the next stage in the pipeline - Operation Genesis: 10-Stage Order"""
        stage_order = [
            self.SOCRATIC_QUESTIONS,
            self.PROBLEM_STRUCTURING,
            self.INTERACTION_SWEEP,
            self.HYBRID_DATA_RESEARCH,      # Oracle Stage
            self.CONSULTANT_SELECTION,
            self.SYNERGY_PROMPTING,
            self.PARALLEL_ANALYSIS,
            self.DEVILS_ADVOCATE,
            self.SENIOR_ADVISOR,
            self.ARBITRATION_CAPTURE,       # Final Persistence
            self.COMPLETED,
        ]

        try:
            current_index = stage_order.index(self)
            if current_index < len(stage_order) - 1:
                return stage_order[current_index + 1]
        except ValueError:
            pass

        return None

    def get_previous_stage(self) -> Optional["PipelineStage"]:
        """Get the previous stage in the pipeline - Operation Genesis: 10-Stage Order"""
        stage_order = [
            self.SOCRATIC_QUESTIONS,
            self.PROBLEM_STRUCTURING,
            self.INTERACTION_SWEEP,
            self.HYBRID_DATA_RESEARCH,      # Oracle Stage
            self.CONSULTANT_SELECTION,
            self.SYNERGY_PROMPTING,
            self.PARALLEL_ANALYSIS,
            self.DEVILS_ADVOCATE,
            self.SENIOR_ADVISOR,
            self.ARBITRATION_CAPTURE,       # Final Persistence
            self.COMPLETED,
        ]

        try:
            current_index = stage_order.index(self)
            if current_index > 0:
                return stage_order[current_index - 1]
        except ValueError:
            pass

        return None

    @property
    def is_revisable(self) -> bool:
        """Whether this stage can be used as a revision point"""
        # All stages except COMPLETED can be revision points
        return self != self.COMPLETED

    @property
    def display_name(self) -> str:
        """Human-readable display name for the stage - Operation Genesis"""
        display_names = {
            self.SOCRATIC_QUESTIONS: "Socratic Questions",
            self.PROBLEM_STRUCTURING: "Problem Structuring",
            self.INTERACTION_SWEEP: "Interaction Sweep",
            self.HYBRID_DATA_RESEARCH: "Hybrid Data Research (Oracle)",
            self.CONSULTANT_SELECTION: "Consultant Selection",
            self.SYNERGY_PROMPTING: "Synergy Prompting",
            self.PARALLEL_ANALYSIS: "Parallel Analysis",
            self.DEVILS_ADVOCATE: "Devils Advocate",
            self.SENIOR_ADVISOR: "Senior Advisor",
            self.ARBITRATION_CAPTURE: "Arbitration & Capture",
            self.COMPLETED: "Analysis Complete",
        }
        return display_names.get(self, self.value.replace("_", " ").title())


class RevisionStatus(str, Enum):
    """Status of checkpoint revision requests"""

    NO_REVISION = "no_revision"
    REVISION_REQUESTED = "revision_requested"
    REVISION_PROCESSING = "revision_processing"
    REVISION_COMPLETED = "revision_completed"
    REVISION_FAILED = "revision_failed"


class StateCheckpoint(BaseModel):
    """
    Lightweight checkpoint storing only essential state for pipeline resumption.

    ARCHITECTURAL PRINCIPLE: This does NOT store the complete context history.
    That remains in the UnifiedContextStream. This only stores the stage output
    and essential metadata needed to resume from this point.
    """

    # Primary identification
    checkpoint_id: Optional[UUID] = None
    trace_id: UUID

    # Pipeline position
    stage_completed: PipelineStage = PipelineStage.SOCRATIC_QUESTIONS
    next_stage: PipelineStage = PipelineStage.PROBLEM_STRUCTURING

    # Legacy compatibility fields (optional)
    current_stage: Optional[PipelineStage] = None
    stage_number: Optional[int] = None
    stage_data: Optional[Dict[str, Any]] = None

    # Core state data (LIGHTWEIGHT - only this stage's output)
    stage_output: Dict[str, Any] = Field(
        default_factory=dict,
        description="Output from the completed stage - NOT the entire context"
    )

    # Checkpoint metadata (First-class product features)
    checkpoint_name: Optional[str] = None
    checkpoint_description: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # State control
    is_revisable: bool = Field(default=True)
    is_shareable: bool = Field(default=False)
    revision_status: RevisionStatus = Field(default=RevisionStatus.NO_REVISION)

    # Performance metrics (for this stage only)
    stage_processing_time_ms: Optional[int] = None
    stage_tokens_consumed: Optional[int] = None
    stage_cost_incurred: Optional[float] = None
    stage_confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)

    # User context
    user_id: Optional[UUID] = None
    session_id: Optional[UUID] = None

    # Audit fields
    indexed_at: Optional[datetime] = None
    last_accessed_at: Optional[datetime] = None

    @validator("trace_id", pre=True, always=True)
    def normalize_trace_id(cls, v):
        """Accept strings and auto-generate UUID for invalid inputs (test compatibility)."""
        try:
            return UUID(str(v))
        except Exception:
            return uuid4()

    @validator("stage_completed", pre=True, always=True)
    def set_stage_completed_from_legacy(cls, v, values):
        """Set stage_completed from legacy current_stage if needed."""
        if v is None:
            legacy = values.get("current_stage")
            if isinstance(legacy, PipelineStage):
                return legacy
        return v or PipelineStage.SOCRATIC_QUESTIONS

    @validator("stage_output", pre=True, always=True)
    def set_stage_output_from_legacy(cls, v, values):
        if v is None and values.get("stage_data") is not None:
            return values.get("stage_data")
        return v or {}

    @validator("next_stage", pre=True, always=True)
    def set_next_stage(cls, v, values):
        """Automatically set next_stage based on stage_completed"""
        if v is None and "stage_completed" in values:
            stage_completed = values["stage_completed"]
            if isinstance(stage_completed, PipelineStage):
                return stage_completed.get_next_stage() or PipelineStage.COMPLETED
        return v or PipelineStage.PROBLEM_STRUCTURING

    @validator("is_revisable", pre=True, always=True)
    def set_is_revisable(cls, v, values):
        """Set revisability based on stage"""
        if v is None and "stage_completed" in values:
            stage_completed = values["stage_completed"]
            if isinstance(stage_completed, PipelineStage):
                return stage_completed.is_revisable
        return v if v is not None else True

    def can_resume_from(self) -> bool:
        """Whether this checkpoint can be used to resume pipeline execution"""
        return (
            self.is_revisable
            and self.revision_status != RevisionStatus.REVISION_PROCESSING
            and self.next_stage != PipelineStage.COMPLETED
        )

    def get_stage_summary(self) -> str:
        """Generate a brief summary of what was accomplished in this stage"""
        stage_summaries = {
            PipelineStage.SOCRATIC_QUESTIONS: lambda output: f"Generated {len(output.get('questions', []))} clarifying questions",
            PipelineStage.PROBLEM_STRUCTURING: lambda output: f"Structured problem with {output.get('framework_type', 'unknown')} framework",
            PipelineStage.INTERACTION_SWEEP: lambda output: f"Identified {len(output.get('cross_term_risks', []))} interaction risks",
            PipelineStage.HYBRID_DATA_RESEARCH: lambda output: f"Completed Oracle research with {output.get('citation_count', 0)} citations",
            PipelineStage.CONSULTANT_SELECTION: lambda output: f"Selected {len(output.get('selected_consultants', []))} consultants",
            PipelineStage.SYNERGY_PROMPTING: lambda output: "Enhanced prompts with synergy insights",
            PipelineStage.PARALLEL_ANALYSIS: lambda output: f"Analyzed via {len(output.get('consultant_results', []))} consultant perspectives",
            PipelineStage.DEVILS_ADVOCATE: lambda output: f"Generated {len(output.get('challenges', []))} critical challenges",
            PipelineStage.SENIOR_ADVISOR: lambda output: "Synthesized final recommendations",
            PipelineStage.ARBITRATION_CAPTURE: lambda output: "Persisted final analysis artifacts",
        }

        summary_func = stage_summaries.get(self.stage_completed)
        if summary_func:
            try:
                return summary_func(self.stage_output)
            except (KeyError, TypeError):
                pass

        # Safe fallback when stage_summaries has no handler
        try:
            stage_enum = (
                self.stage_completed
                if isinstance(self.stage_completed, PipelineStage)
                else PipelineStage(self.stage_completed)
            )
            display = stage_enum.display_name
        except Exception:
            # Final safeguard if value isn't a valid PipelineStage
            display = str(self.stage_completed).replace("_", " ").title()

        return f"Completed {display}"

    def to_database_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for database storage"""
        return {
            "checkpoint_id": self.checkpoint_id,
            "trace_id": self.trace_id,
            "stage_completed": self.stage_completed.value,
            "stage_output": self.stage_output,
            "next_stage": self.next_stage.value,
            "checkpoint_name": self.checkpoint_name,
            "checkpoint_description": self.checkpoint_description,
            "created_at": self.created_at,
            "is_revisable": self.is_revisable,
            "is_shareable": self.is_shareable,
            "revision_status": self.revision_status.value,
            "stage_processing_time_ms": self.stage_processing_time_ms,
            "stage_tokens_consumed": self.stage_tokens_consumed,
            "stage_cost_incurred": self.stage_cost_incurred,
            "stage_confidence_score": self.stage_confidence_score,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "indexed_at": self.indexed_at,
            "last_accessed_at": self.last_accessed_at,
        }

    @classmethod
    def from_database_dict(cls, data: Dict[str, Any]) -> "StateCheckpoint":
        """Create StateCheckpoint from database record"""
        # Convert string enum values back to enums
        if isinstance(data.get("stage_completed"), str):
            data["stage_completed"] = PipelineStage(data["stage_completed"])
        if isinstance(data.get("next_stage"), str):
            data["next_stage"] = PipelineStage(data["next_stage"])
        if isinstance(data.get("revision_status"), str):
            data["revision_status"] = RevisionStatus(data["revision_status"])

        return cls(**data)

    model_config = ConfigDict(use_enum_values=True)


class AnalysisRevision(BaseModel):
    """
    Model for tracking analysis revision requests and forking operations.

    This represents a request to create a new analysis branch from an existing checkpoint.
    """

    # Identification
    revision_id: Optional[UUID] = None

    # Parent-child relationship (Analysis Tree)
    parent_trace_id: UUID = Field(description="Original analysis trace_id")
    child_trace_id: Optional[UUID] = Field(
        None, description="New analysis trace_id after forking"
    )
    source_checkpoint_id: UUID = Field(description="Checkpoint to restart from")

    # Revision details
    restart_from_stage: PipelineStage = Field(description="Which stage to restart from")
    revision_data: Dict[str, Any] = Field(description="New user inputs for the stage")
    revision_rationale: Optional[str] = Field(
        None, description="User's explanation for revision"
    )

    # Revision metadata
    requested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    processing_started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Status tracking
    status: str = Field(
        default="pending"
    )  # 'pending', 'processing', 'completed', 'failed'
    error_message: Optional[str] = None

    # User context
    user_id: Optional[UUID] = None
    session_id: Optional[UUID] = None

    def mark_processing_started(self):
        """Mark revision as started processing"""
        self.status = "processing"
        self.processing_started_at = datetime.now(timezone.utc)

    def mark_completed(self, child_trace_id: UUID):
        """Mark revision as completed successfully"""
        self.status = "completed"
        self.child_trace_id = child_trace_id
        self.completed_at = datetime.now(timezone.utc)

    def mark_failed(self, error_message: str):
        """Mark revision as failed with error"""
        self.status = "failed"
        self.error_message = error_message
        self.completed_at = datetime.now(timezone.utc)

    @property
    def is_complete(self) -> bool:
        """Whether the revision has been completed (successfully or failed)"""
        return self.status in ["completed", "failed"]

    @property
    def processing_time_ms(self) -> Optional[int]:
        """Calculate processing time in milliseconds"""
        if self.processing_started_at and self.completed_at:
            delta = self.completed_at - self.processing_started_at
            return int(delta.total_seconds() * 1000)
        return None

    model_config = ConfigDict(use_enum_values=True)


class CheckpointShare(BaseModel):
    """
    Model for sharing checkpoints between users (Collaborative checkpoints).

    This enables the "Checkpoints as a Product" feature where users can share
    their analytical frameworks and thought processes.
    """

    # Identification
    share_id: Optional[UUID] = None
    checkpoint_id: UUID

    # Sharing details
    shared_by_user_id: UUID
    shared_with_user_id: Optional[UUID] = Field(
        None, description="NULL for public shares"
    )
    share_token: Optional[str] = Field(None, description="Secure sharing token")

    # Share metadata
    share_name: str = Field(description="Public name for the shared checkpoint")
    share_description: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None

    # Access control
    is_public: bool = Field(default=False)
    access_count: int = Field(default=0)
    last_accessed_at: Optional[datetime] = None

    def is_accessible_by(self, user_id: UUID) -> bool:
        """Check if a user can access this shared checkpoint"""
        return (
            self.is_public
            or self.shared_by_user_id == user_id
            or self.shared_with_user_id == user_id
        )

    def is_expired(self) -> bool:
        """Check if the share has expired"""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def record_access(self):
        """Record an access to this shared checkpoint"""
        self.access_count += 1
        self.last_accessed_at = datetime.now(timezone.utc)

    model_config = ConfigDict(use_enum_values=True)


class AnalysisTreeNode(BaseModel):
    """
    Represents a node in the Analysis Tree for visualization purposes.

    This is used for API responses showing the complete tree structure
    of an analysis and all its revision branches.
    """

    # Node identification
    trace_id: UUID
    parent_trace_id: Optional[UUID] = None

    # Tree structure
    depth: int = 0
    children: List["AnalysisTreeNode"] = Field(default_factory=list)

    # Analysis metadata
    engagement_type: Optional[str] = None
    case_id: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    final_status: Optional[str] = None

    # Checkpoints in this analysis
    checkpoints: List[StateCheckpoint] = Field(default_factory=list)

    # Revision metadata (if this is a revision branch)
    revision_info: Optional[Dict[str, Any]] = None

    def add_child(self, child_node: "AnalysisTreeNode"):
        """Add a child node to this tree node"""
        child_node.depth = self.depth + 1
        self.children.append(child_node)

    def get_total_nodes(self) -> int:
        """Get total number of nodes in this subtree"""
        return 1 + sum(child.get_total_nodes() for child in self.children)

    def get_max_depth(self) -> int:
        """Get maximum depth of this subtree"""
        if not self.children:
            return self.depth
        return max(child.get_max_depth() for child in self.children)

    @property
    def is_leaf(self) -> bool:
        """Whether this node is a leaf (has no children)"""
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        """Whether this node is the root (has no parent)"""
        return self.parent_trace_id is None

    model_config = ConfigDict(use_enum_values=True)


# Forward reference resolution for recursive model
AnalysisTreeNode.model_rebuild()


# Utility functions for checkpoint management


def create_socratic_checkpoint(
    trace_id: UUID,
    questions: List[Dict[str, Any]],
    user_answers: List[Dict[str, Any]],
    processing_time_ms: int,
    tokens_consumed: int,
    cost_incurred: float,
    user_id: Optional[UUID] = None,
) -> StateCheckpoint:
    """Create a checkpoint after Socratic Questions stage"""
    return StateCheckpoint(
        trace_id=trace_id,
        stage_completed=PipelineStage.SOCRATIC_QUESTIONS,
        stage_output={
            "questions": questions,
            "user_answers": user_answers,
            "questions_count": len(questions),
            "answered_count": len([a for a in user_answers if a.get("answer")]),
        },
        checkpoint_name="Socratic Dialogue Complete",
        checkpoint_description="User has answered clarifying questions about their challenge",
        stage_processing_time_ms=processing_time_ms,
        stage_tokens_consumed=tokens_consumed,
        stage_cost_incurred=cost_incurred,
        stage_confidence_score=len([a for a in user_answers if a.get("answer")])
        / max(1, len(questions)),
        user_id=user_id,
    )


def create_problem_structuring_checkpoint(
    trace_id: UUID,
    framework: Dict[str, Any],
    processing_time_ms: int,
    tokens_consumed: int,
    cost_incurred: float,
    user_id: Optional[UUID] = None,
) -> StateCheckpoint:
    """Create a checkpoint after Problem Structuring stage"""
    return StateCheckpoint(
        trace_id=trace_id,
        stage_completed=PipelineStage.PROBLEM_STRUCTURING,
        stage_output={
            "framework": framework,
            "framework_type": framework.get("framework_type"),
            "dimensions_count": len(framework.get("primary_dimensions", [])),
        },
        checkpoint_name="Problem Framework Defined",
        checkpoint_description="Analytical framework has been structured for the problem",
        stage_processing_time_ms=processing_time_ms,
        stage_tokens_consumed=tokens_consumed,
        stage_cost_incurred=cost_incurred,
        stage_confidence_score=framework.get("confidence_score", 0.8),
        user_id=user_id,
    )
