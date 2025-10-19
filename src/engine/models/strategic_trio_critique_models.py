#!/usr/bin/env python3
"""
Strategic Trio + Devil's Advocate Integration Models
Data contracts for Multi-Single-Agent critique system following Optional, Post-Human, Per-Consultant pattern
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from src.cognitive_architecture.mental_models_system import ConsultantRole
from src.core.enhanced_devils_advocate_system import (
    ComprehensiveChallengeResult,
    DevilsAdvocateChallenge,
)


class CritiqueRequestStatus(Enum):
    """Status of critique request"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ConsultantCritiqueRequest:
    """Individual consultant critique request"""

    consultant_role: ConsultantRole
    analysis_text: str
    business_context: Dict[str, Any] = field(default_factory=dict)
    critique_engines: List[str] = field(
        default_factory=lambda: ["munger", "ackoff", "cognitive_audit"]
    )
    priority: str = "normal"  # normal, high, urgent


@dataclass
class ConsultantCritiqueResult:
    """Independent critique result for single consultant - NO synthesis with others"""

    consultant_role: ConsultantRole
    original_analysis: str
    comprehensive_challenge_result: ComprehensiveChallengeResult
    critique_request_id: str
    processing_time_seconds: float
    critique_timestamp: datetime = field(default_factory=datetime.utcnow)

    # Quality metrics for this specific consultant's critique
    challenges_by_engine: Dict[str, int] = field(default_factory=dict)
    highest_risk_challenge: Optional[DevilsAdvocateChallenge] = None
    consultant_specific_insights: List[str] = field(default_factory=list)


@dataclass
class MultiConsultantCritiqueRequest:
    """Request for critiquing multiple consultants independently"""

    original_execution_id: str
    consultant_requests: List[ConsultantCritiqueRequest]
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    async_processing: bool = True  # Stream results as they complete

    # Human-in-the-loop controls
    human_triggered: bool = True  # Must be explicitly requested by human
    post_analysis_timing: bool = True  # After Strategic Trio results seen by human

    request_id: str = field(
        default_factory=lambda: f"critique_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MultiConsultantCritiqueResult:
    """
    Complete Multi-Single-Agent critique result preserving all consultant independence

    Key principle: Each consultant's critique is independent - NO synthesis between them
    """

    original_execution_id: str
    consultant_critiques: Dict[
        ConsultantRole, ConsultantCritiqueResult
    ]  # Independent critiques

    # Aggregated metrics (informational only - not for synthesis)
    total_processing_time: float
    critiques_completed: int
    critiques_failed: int

    # Human choice facilitation
    critique_summary: str  # Overview to help human navigate critiques
    recommended_next_actions: List[str]  # Suggestions, not decisions

    # Multi-Single-Agent validation
    independence_preserved: bool = True  # Each critique is independent
    synthesis_avoided: bool = True  # No merging of critique insights
    human_choice_enabled: bool = True  # Human can choose which critiques to act on

    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CritiqueStreamingUpdate:
    """Real-time streaming update for async critique processing"""

    update_type: str  # consultant_started, consultant_completed, all_completed, error
    consultant_role: Optional[ConsultantRole] = None
    progress_percent: float = 0.0
    current_engine: Optional[str] = None
    partial_result: Optional[ConsultantCritiqueResult] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class StrategicTrioCritiqueOrchestrationResult:
    """
    Complete result combining Strategic Trio execution with optional Devil's Advocate critique
    Implements Optional, Post-Human, Per-Consultant pattern
    """

    # Phase 1: Strategic Trio Results (always provided)
    strategic_trio_result: (
        Any  # CognitiveExecutionResult from our previous implementation
    )

    # Phase 2: Optional Critique Results (only if requested by human)
    critique_result: Optional[MultiConsultantCritiqueResult] = None
    critique_requested: bool = False
    critique_in_progress: bool = False

    # Human orchestration metadata
    human_seen_original: bool = False  # Has human reviewed Strategic Trio results?
    human_requested_critique: bool = False  # Did human explicitly request critique?
    critique_streaming_enabled: bool = True  # Stream critique results as they complete

    # Integration metadata
    orchestration_id: str = field(
        default_factory=lambda: f"orchestration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    total_consultants_analyzed: int = 0
    total_consultants_critiqued: int = 0

    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


# Factory functions for creating critique requests


def create_consultant_critique_request(
    consultant_role: ConsultantRole,
    analysis_text: str,
    business_context: Optional[Dict[str, Any]] = None,
    engines: Optional[List[str]] = None,
) -> ConsultantCritiqueRequest:
    """Create critique request for single consultant"""
    return ConsultantCritiqueRequest(
        consultant_role=consultant_role,
        analysis_text=analysis_text,
        business_context=business_context or {},
        critique_engines=engines or ["munger", "ackoff", "cognitive_audit"],
    )


def create_multi_consultant_critique_request(
    original_execution_id: str,
    consultant_analyses: Dict[ConsultantRole, str],
    business_context: Optional[Dict[str, Any]] = None,
) -> MultiConsultantCritiqueRequest:
    """
    Create multi-consultant critique request from Strategic Trio results
    Each consultant gets independent critique - no coordination between them
    """
    consultant_requests = []

    for consultant_role, analysis_text in consultant_analyses.items():
        # Create context specific to this consultant's role and analysis
        consultant_context = {
            **(business_context or {}),
            "consultant_role": consultant_role.value,
            "analysis_perspective": f"Independent {consultant_role.value} analysis",
            "critique_independence": True,  # Flag for devil's advocate to maintain independence
        }

        request = create_consultant_critique_request(
            consultant_role=consultant_role,
            analysis_text=analysis_text,
            business_context=consultant_context,
        )
        consultant_requests.append(request)

    return MultiConsultantCritiqueRequest(
        original_execution_id=original_execution_id,
        consultant_requests=consultant_requests,
        human_triggered=True,
        post_analysis_timing=True,
    )
