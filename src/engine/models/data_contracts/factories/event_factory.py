"""
Event Factory Functions

Functions for creating CloudEvents-compliant event structures.
"""

from typing import Dict, Any, List
from datetime import datetime, timezone
from uuid import UUID, uuid4
from ..models.engagement_models import (
    EngagementContext,
    ExplorationContext,
    WorkflowState,
)
from ..models.consultant_models import *
from ..models.analysis_models import (
    MentalModelDefinition,
    MetisDataContract,
    CognitiveState,
    VulnerabilityContext,
    HallucinationCheck,
)
from ..models.enums import EngagementPhase


def create_engagement_initiated_event(
    problem_statement: str,
    business_context: Dict[str, Any] = None,
    client_name: str = "Unknown Client",
) -> MetisDataContract:
    """Factory for engagement initiation events"""
    return MetisDataContract(
        type="metis.engagement_initiated",
        source="/metis/cognitive_engine",
        engagement_context=EngagementContext(
            problem_statement=problem_statement,
            client_name=client_name,
            business_context=business_context or {},
        ),
        cognitive_state=CognitiveState(),
        workflow_state=WorkflowState(current_phase=EngagementPhase.PROBLEM_STRUCTURING),
    )


def create_model_selection_event(
    engagement_id: UUID,
    selected_models: List[MentalModelDefinition],
    client_name: str = "Unknown Client",
) -> MetisDataContract:
    """Factory for mental model selection events"""
    return MetisDataContract(
        type="metis.cognitive_model_selected",
        source="/metis/model_selector",
        engagement_context=EngagementContext(
            engagement_id=engagement_id, problem_statement="", client_name=client_name
        ),
        cognitive_state=CognitiveState(selected_mental_models=selected_models),
        workflow_state=WorkflowState(
            current_phase=EngagementPhase.HYPOTHESIS_GENERATION
        ),
    )


def create_vulnerability_assessment_event(
    engagement_id: UUID,
    vulnerability_context: VulnerabilityContext,
    client_name: str = "Unknown Client",
) -> MetisDataContract:
    """Factory for vulnerability assessment events"""
    return MetisDataContract(
        type="metis.vulnerability_assessment_completed",
        source="/metis/vulnerability_coordinator",
        engagement_context=EngagementContext(
            engagement_id=engagement_id,
            problem_statement="Test vulnerability assessment problem statement",
            client_name=client_name,
        ),
        cognitive_state=CognitiveState(),
        workflow_state=WorkflowState(current_phase=EngagementPhase.ANALYSIS_EXECUTION),
        vulnerability_context=vulnerability_context,
    )


def create_exploration_strategy_event(
    engagement_id: UUID,
    exploration_context: ExplorationContext,
    client_name: str = "Unknown Client",
) -> MetisDataContract:
    """Factory for exploration strategy events"""
    return MetisDataContract(
        type="metis.exploration_strategy_applied",
        source="/metis/exploration_engine",
        engagement_context=EngagementContext(
            engagement_id=engagement_id,
            problem_statement="Test vulnerability assessment problem statement",
            client_name=client_name,
        ),
        cognitive_state=CognitiveState(),
        workflow_state=WorkflowState(
            current_phase=EngagementPhase.HYPOTHESIS_GENERATION
        ),
        vulnerability_context=VulnerabilityContext(
            session_id=f"exploration_{engagement_id}",
            exploration_context=exploration_context,
        ),
    )


def create_hallucination_detection_event(
    engagement_id: UUID,
    hallucination_checks: List[HallucinationCheck],
    client_name: str = "Unknown Client",
) -> MetisDataContract:
    """Factory for hallucination detection events"""
    return MetisDataContract(
        type="metis.hallucination_detected",
        source="/metis/hallucination_detector",
        engagement_context=EngagementContext(
            engagement_id=engagement_id,
            problem_statement="Test vulnerability assessment problem statement",
            client_name=client_name,
        ),
        cognitive_state=CognitiveState(),
        workflow_state=WorkflowState(current_phase=EngagementPhase.ANALYSIS_EXECUTION),
        vulnerability_context=VulnerabilityContext(
            session_id=f"hallucination_{engagement_id}",
            hallucination_checks=hallucination_checks,
        ),
    )
