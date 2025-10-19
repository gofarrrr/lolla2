"""
Contract to API Response Mapping Functions for METIS Engagement API
"""

from datetime import datetime

from .models import (
    EngagementResponse,
    EngagementPhase,
    EngagementStatus,
    PhaseResult,
    ProblemStatement,
)

try:
    from src.engine.models.data_contracts import MetisDataContract, DeliverableArtifact

    CONTRACTS_AVAILABLE = True
except ImportError:
    CONTRACTS_AVAILABLE = False
    MetisDataContract = None
    DeliverableArtifact = None


def map_contract_to_engagement_response(
    contract: MetisDataContract, client_name: str
) -> EngagementResponse:
    """Convert MetisDataContract to EngagementResponse for backward compatibility"""

    if not CONTRACTS_AVAILABLE or not contract:
        raise ImportError("Data contracts not available or contract is None")

    # Calculate progress based on completed phases
    phase_weights = {
        EngagementPhase.PROBLEM_STRUCTURING: 25,
        EngagementPhase.HYPOTHESIS_GENERATION: 25,
        EngagementPhase.ANALYSIS_EXECUTION: 30,
        EngagementPhase.SYNTHESIS_DELIVERY: 20,
    }

    completed_weight = 0
    phases_dict = {}

    # Map contract phases to API phases
    for phase_name, phase_result in contract.workflow_state.phase_results.items():
        # Convert phase names between contract and API formats
        api_phase_name = (
            phase_name  # They should match, but this mapping can be customized
        )

        # Map status
        status = "completed" if phase_result.get("status") == "completed" else "error"

        # Create PhaseResult from contract data
        phase_result_obj = PhaseResult(
            phase=(
                EngagementPhase(api_phase_name)
                if api_phase_name in [p.value for p in EngagementPhase]
                else EngagementPhase.PROBLEM_STRUCTURING
            ),
            status=status,
            confidence=0.85,  # Default confidence - could be extracted from contract
            insights=[f"Phase {api_phase_name} completed successfully"],
            data=phase_result.get("result", {}),
            execution_time=phase_result.get("execution_time", 0.0),
            timestamp=datetime.fromisoformat(
                phase_result.get("timestamp", datetime.utcnow().isoformat())
            ),
        )

        phases_dict[api_phase_name] = phase_result_obj

        # Calculate progress
        if status == "completed" and api_phase_name in [
            p.value for p in EngagementPhase
        ]:
            completed_weight += phase_weights.get(EngagementPhase(api_phase_name), 0)

    # Calculate overall confidence from deliverable artifacts
    confidences = []
    for artifact in contract.deliverable_artifacts:
        # Handle both dict and object formats for artifacts
        if isinstance(artifact, dict):
            confidence_level = artifact.get("confidence_level", "medium")
            if isinstance(confidence_level, str):
                conf_map = {"high": 0.9, "medium": 0.8, "low": 0.6, "uncertain": 0.4}
                confidences.append(conf_map.get(confidence_level.lower(), 0.8))
        elif hasattr(artifact, "confidence_level") and hasattr(
            artifact.confidence_level, "value"
        ):
            conf_map = {"high": 0.9, "medium": 0.8, "low": 0.6, "uncertain": 0.4}
            confidences.append(
                conf_map.get(artifact.confidence_level.value.lower(), 0.8)
            )

    overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    # Determine current status
    if len(phases_dict) == 4:
        status = EngagementStatus.COMPLETED
    elif len(phases_dict) > 0:
        status = EngagementStatus.IN_PROGRESS
    else:
        status = EngagementStatus.CREATED

    # Create problem statement from contract
    problem_statement = ProblemStatement(
        problem_description=contract.engagement_context.problem_statement,
        business_context=contract.engagement_context.business_context,
        stakeholders=[],  # Could be extracted from business_context
        success_criteria=[],  # Could be extracted from business_context
    )

    return EngagementResponse(
        engagement_id=contract.engagement_context.engagement_id,
        client_name=client_name,
        problem_statement=problem_statement,
        status=status,
        current_phase=EngagementPhase(contract.workflow_state.current_phase.value),
        progress_percentage=min(completed_weight, 100),
        phases=phases_dict,
        overall_confidence=overall_confidence,
        estimated_cost=0.72,  # Default cost - could be calculated from contract metadata
        created_at=contract.engagement_context.created_at,
        updated_at=contract.time,
        deliverable_ready=(len(phases_dict) == 4),
    )


def map_contract_phase_to_phase_result(
    contract: MetisDataContract, phase_name: str
) -> PhaseResult:
    """Extract specific phase result from contract"""

    if not CONTRACTS_AVAILABLE or not contract:
        raise ImportError("Data contracts not available or contract is None")

    phase_data = contract.workflow_state.phase_results.get(phase_name, {})

    if not phase_data:
        raise ValueError(f"Phase {phase_name} not found in contract")

    # Find related deliverable artifacts for this phase
    phase_artifacts = []
    for artifact in contract.deliverable_artifacts:
        # Handle both dict and object formats
        if isinstance(artifact, dict):
            artifact_type = artifact.get("type", "")
            if artifact_type == phase_name or phase_name in artifact_type:
                phase_artifacts.append(artifact)
        elif hasattr(artifact, "artifact_type"):
            if (
                artifact.artifact_type == phase_name
                or phase_name in artifact.artifact_type
            ):
                phase_artifacts.append(artifact)

    insights = []
    if phase_artifacts:
        artifact = phase_artifacts[-1]  # Get latest artifact
        # Handle both dict and object formats
        if isinstance(artifact, dict):
            artifact_type = artifact.get("type", "deliverable")
            insights = [f"Deliverable created: {artifact_type}"]
            title = artifact.get("title", "Strategic deliverable")
            insights.append(f"Title: {title}")
        elif hasattr(artifact, "artifact_type"):
            insights = [f"Deliverable created: {artifact.artifact_type}"]
            if hasattr(artifact, "methodology_used"):
                insights.extend(
                    [
                        f"Methodology: {method}"
                        for method in artifact.methodology_used[:2]
                    ]
                )

    return PhaseResult(
        phase=(
            EngagementPhase(phase_name)
            if phase_name in [p.value for p in EngagementPhase]
            else EngagementPhase.PROBLEM_STRUCTURING
        ),
        status=phase_data.get("status", "completed"),
        confidence=0.85,  # Could be derived from artifact confidence
        insights=insights or [f"Phase {phase_name} completed"],
        data=phase_data.get("result", {}),
        execution_time=phase_data.get("execution_time", 0.0),
        timestamp=datetime.fromisoformat(
            phase_data.get("timestamp", datetime.utcnow().isoformat())
        ),
    )
