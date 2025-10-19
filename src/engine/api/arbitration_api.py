"""
Arbitration API for Red Team Council User-in-the-Loop Feature
Purpose: Handle user arbitration of critiques and trigger synthesis

This module implements the arbitration endpoint that:
1. Accepts user-selected critique IDs
2. Updates the MetisDataContract with selections
3. Triggers transition to SYNTHESIS_DELIVERY phase
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.config import get_settings
from src.engine.adapters.core.structured_logging import get_logger
from src.engine.engines.core.consultant_orchestrator import get_consultant_orchestrator
from src.engine.models.data_contracts import MetisDataContract, EngagementPhase

settings = get_settings()
logger = get_logger(__name__, component="arbitration_api")

# Create API router
router = APIRouter(
    prefix="/api/v2/engagements",
    tags=["Arbitration"],
    responses={404: {"description": "Not found"}},
)


class ArbitrationRequest(BaseModel):
    """Request model for critique arbitration"""

    selected_critiques: List[str] = Field(
        default_factory=list,
        description="List of critique IDs selected by the user for synthesis integration",
    )
    skip_arbitration: bool = Field(
        default=False,
        description="If true, proceed with AI's autonomous synthesis without user priorities",
    )
    user_comments: Optional[str] = Field(
        default=None, description="Optional user comments about their selections"
    )


class ArbitrationResponse(BaseModel):
    """Response model for critique arbitration"""

    status: str
    engagement_id: str
    selected_count: int
    total_critiques: int
    synthesis_triggered: bool
    timestamp: str
    message: str


class ArbitrationService:
    """Service for handling critique arbitration"""

    def __init__(self):
        self.logger = logger.with_component("arbitration_service")
        self.orchestrator = get_consultant_orchestrator()
        # Store for idempotency
        self.arbitration_cache: Dict[str, ArbitrationResponse] = {}

    async def process_arbitration(
        self,
        engagement_id: UUID,
        request: ArbitrationRequest,
        contract: MetisDataContract,
    ) -> ArbitrationResponse:
        """
        Process user arbitration of critiques

        This method:
        1. Validates the critique IDs
        2. Updates the contract with user selections
        3. Triggers synthesis phase
        4. Returns arbitration confirmation
        """

        # Check idempotency - if we've seen this exact request before
        cache_key = f"{engagement_id}:{','.join(sorted(request.selected_critiques))}"
        if cache_key in self.arbitration_cache:
            self.logger.info(
                "arbitration_idempotent_response",
                engagement_id=str(engagement_id),
                cache_hit=True,
            )
            return self.arbitration_cache[cache_key]

        try:
            # Validate current phase
            current_phase = contract.workflow_state.current_phase
            if current_phase != EngagementPhase.VALIDATION_DEBATE:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Arbitration only allowed in VALIDATION_DEBATE phase. Current: {current_phase}",
                )

            # Get all available critiques from validation results
            available_critiques = self._extract_all_critique_ids(contract)

            # Validate selected critique IDs
            invalid_ids = [
                cid
                for cid in request.selected_critiques
                if cid not in available_critiques
            ]

            if invalid_ids and not request.skip_arbitration:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid critique IDs: {invalid_ids}",
                )

            # Update contract with user arbitration
            if not hasattr(contract, "validation_results"):
                contract.validation_results = {}

            contract.validation_results["user_arbitration"] = {
                "prioritized_critiques": request.selected_critiques,
                "skip_arbitration": request.skip_arbitration,
                "user_comments": request.user_comments,
                "arbitration_timestamp": datetime.utcnow().isoformat(),
                "total_available_critiques": len(available_critiques),
                "selected_count": len(request.selected_critiques),
            }

            self.logger.info(
                "user_arbitration_recorded",
                engagement_id=str(engagement_id),
                selected_count=len(request.selected_critiques),
                total_available=len(available_critiques),
                skip_arbitration=request.skip_arbitration,
            )

            # Trigger state transition to SYNTHESIS_DELIVERY
            # This would normally be done through the state machine
            contract.workflow_state.current_phase = EngagementPhase.SYNTHESIS_DELIVERY

            # Create response
            response = ArbitrationResponse(
                status="success",
                engagement_id=str(engagement_id),
                selected_count=len(request.selected_critiques),
                total_critiques=len(available_critiques),
                synthesis_triggered=True,
                timestamp=datetime.utcnow().isoformat(),
                message=(
                    "AI autonomous synthesis triggered"
                    if request.skip_arbitration
                    else f"User priorities recorded: {len(request.selected_critiques)} critiques selected for synthesis"
                ),
            )

            # Cache for idempotency
            self.arbitration_cache[cache_key] = response

            # Log analytics event
            self._log_arbitration_analytics(engagement_id, request, available_critiques)

            return response

        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(
                "arbitration_processing_failed",
                engagement_id=str(engagement_id),
                error=str(e),
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Arbitration processing failed: {str(e)}",
            )

    def _extract_all_critique_ids(self, contract: MetisDataContract) -> Set[str]:
        """Extract all available critique IDs from validation results"""
        critique_ids = set()

        if not hasattr(contract, "validation_results"):
            return critique_ids

        for challenger_name in ["munger_critique", "ackoff_critique", "bias_audit"]:
            if challenger_name in contract.validation_results:
                result = contract.validation_results[challenger_name]
                if isinstance(result, dict) and "critiques" in result:
                    for critique in result["critiques"]:
                        if isinstance(critique, dict) and "id" in critique:
                            critique_ids.add(critique["id"])

        return critique_ids

    def _log_arbitration_analytics(
        self,
        engagement_id: UUID,
        request: ArbitrationRequest,
        available_critiques: Set[str],
    ):
        """Log analytics for arbitration decisions"""

        # Calculate arbitration rate
        arbitration_rate = (
            len(request.selected_critiques) / len(available_critiques) * 100
            if available_critiques
            else 0
        )

        # Analyze which types of critiques were selected
        selected_types = {"munger": 0, "ackoff": 0, "bias": 0}

        for critique_id in request.selected_critiques:
            if "munger" in critique_id:
                selected_types["munger"] += 1
            elif "ackoff" in critique_id:
                selected_types["ackoff"] += 1
            elif "bias" in critique_id:
                selected_types["bias"] += 1

        self.logger.info(
            "arbitration_analytics",
            engagement_id=str(engagement_id),
            arbitration_rate=arbitration_rate,
            skip_arbitration=request.skip_arbitration,
            total_available=len(available_critiques),
            total_selected=len(request.selected_critiques),
            selected_types=selected_types,
            has_user_comments=bool(request.user_comments),
        )


# Initialize service
arbitration_service = ArbitrationService()


@router.post("/{engagement_id}/arbitrate_critiques", response_model=ArbitrationResponse)
async def arbitrate_critiques(
    engagement_id: UUID, request: ArbitrationRequest
) -> ArbitrationResponse:
    """
    User arbitration endpoint for Red Team Council critiques

    This endpoint allows users to:
    1. Select which critiques should be prioritized in the final synthesis
    2. Skip arbitration and proceed with AI's autonomous synthesis
    3. Add optional comments about their selections

    The endpoint is idempotent - submitting the same selection multiple times
    will return the same result without re-processing.
    """

    logger.info(
        "arbitration_request_received",
        engagement_id=str(engagement_id),
        selected_count=len(request.selected_critiques),
        skip_arbitration=request.skip_arbitration,
    )

    # TODO: In production, retrieve contract from database
    # For now, create a mock contract for demonstration
    contract = MetisDataContract(
        engagement_id=engagement_id,
        workflow_state={
            "current_phase": EngagementPhase.VALIDATION_DEBATE,
            "completed_phases": [
                EngagementPhase.PROBLEM_STRUCTURING,
                EngagementPhase.HYPOTHESIS_GENERATION,
                EngagementPhase.ANALYSIS_EXECUTION,
                EngagementPhase.RESEARCH_GROUNDING,
                EngagementPhase.VALIDATION_DEBATE,
            ],
        },
        validation_results={
            "munger_critique": {
                "status": "success",
                "critiques": [
                    {"id": "munger.failure_mode.001", "type": "failure_mode"},
                    {"id": "munger.failure_mode.002", "type": "failure_mode"},
                ],
            },
            "ackoff_critique": {
                "status": "success",
                "critiques": [
                    {"id": "ackoff.assumption.001", "type": "assumption"},
                    {"id": "ackoff.assumption.002", "type": "assumption"},
                ],
            },
            "bias_audit": {
                "status": "success",
                "critiques": [
                    {"id": "bias.confirmation.001", "type": "bias"},
                    {"id": "bias.planning_fallacy.001", "type": "bias"},
                ],
            },
        },
    )

    # Process arbitration
    response = await arbitration_service.process_arbitration(
        engagement_id, request, contract
    )

    return response


@router.get("/{engagement_id}/arbitration_status")
async def get_arbitration_status(engagement_id: UUID) -> Dict[str, Any]:
    """
    Get the current arbitration status for an engagement

    Returns information about:
    - Whether arbitration has been completed
    - Number of critiques selected
    - Whether synthesis has been triggered
    """

    # TODO: In production, retrieve from database
    # For now, return mock status
    return {
        "engagement_id": str(engagement_id),
        "arbitration_complete": False,
        "awaiting_user_input": True,
        "total_critiques_available": 6,
        "critiques_selected": 0,
        "synthesis_triggered": False,
        "current_phase": "VALIDATION_DEBATE",
    }


# Export router for inclusion in main API
__all__ = ["router"]