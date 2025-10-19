"""
METIS Engagement Factory
Factory functions for creating engagement-related events and objects
"""

from typing import Dict, List, Any
from uuid import UUID

from ..models.enums import EngagementPhase
from ..models.engagement_models import EngagementContext, MentalModelDefinition
from ..models.analysis_models import CognitiveState, WorkflowState, MetisDataContract


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
