"""
Enhanced Arbitration API for Red Team Council
Purpose: Handle rich user feedback including dispositions, rationales, and custom critiques

This enhanced module supports:
1. Three-state disposition (NEUTRAL/PRIORITIZE/DISAGREE)
2. Disagreement rationales
3. User-generated critiques (Write-In Candidates)
4. Complete audit trail for the Flywheel
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from uuid import UUID
from enum import Enum

from fastapi import APIRouter, HTTPException, status, Request
from pydantic import BaseModel, Field

from src.config import get_settings
from src.core.structured_logging import get_logger
from src.engine.engines.core.consultant_orchestrator import get_consultant_orchestrator
from src.engine.models.data_contracts import MetisDataContract, EngagementPhase
from src.engine.api.feature_flag_decorators import (
    enhanced_arbitration_ab_test,
    feature_flag_required,
    FeatureFlag,
)

settings = get_settings()
logger = get_logger(__name__, component="enhanced_arbitration_api")

# Create API router
router = APIRouter(
    prefix="/api/v2/engagements",
    tags=["Enhanced Arbitration"],
    responses={404: {"description": "Not found"}},
)


class DispositionType(str, Enum):
    """Three-state disposition for critiques"""

    NEUTRAL = "NEUTRAL"
    PRIORITIZE = "PRIORITIZE"
    DISAGREE = "DISAGREE"


class CritiqueDisposition(BaseModel):
    """Enhanced critique disposition with optional rationale"""

    critique_id: str = Field(..., description="ID of the critique")
    disposition: DispositionType = Field(
        ..., description="User's disposition toward this critique"
    )
    rationale: Optional[str] = Field(
        None, description="Optional rationale for disagreement", max_length=500
    )


class EnhancedArbitrationRequest(BaseModel):
    """Enhanced request model with dispositions and user-generated critique"""

    critique_dispositions: List[CritiqueDisposition] = Field(
        default_factory=list, description="User's disposition for each critique"
    )
    user_generated_critique: str = Field(
        default="",
        description="User's custom critique (Write-In Candidate)",
        max_length=2000,
    )


class EnhancedArbitrationResponse(BaseModel):
    """Response model for enhanced arbitration"""

    status: str
    engagement_id: str
    prioritized_count: int
    disagreed_count: int
    neutral_count: int
    has_user_critique: bool
    total_critiques: int
    synthesis_triggered: bool
    timestamp: str
    message: str
    flywheel_data_captured: bool


class EnhancedArbitrationService:
    """Service for handling enhanced critique arbitration"""

    def __init__(self):
        self.logger = logger.with_component("enhanced_arbitration_service")
        self.orchestrator = get_consultant_orchestrator()
        # Store for idempotency
        self.arbitration_cache: Dict[str, EnhancedArbitrationResponse] = {}

    async def process_enhanced_arbitration(
        self,
        engagement_id: UUID,
        request: EnhancedArbitrationRequest,
        contract: MetisDataContract,
    ) -> EnhancedArbitrationResponse:
        """
        Process enhanced user arbitration with dispositions and custom critiques

        This method:
        1. Validates and categorizes dispositions
        2. Processes user-generated critique as highest priority
        3. Captures rich feedback for the Flywheel
        4. Triggers synthesis with full context
        """

        # Create cache key for idempotency
        disposition_keys = sorted(
            [
                f"{d.critique_id}:{d.disposition}:{d.rationale or ''}"
                for d in request.critique_dispositions
            ]
        )
        cache_key = f"{engagement_id}:{','.join(disposition_keys)}:{hash(request.user_generated_critique)}"

        if cache_key in self.arbitration_cache:
            self.logger.info(
                "enhanced_arbitration_idempotent_response",
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

            # Get all available critiques
            available_critiques = self._extract_all_critique_ids(contract)

            # Categorize dispositions
            prioritized_critiques = []
            disagreed_critiques = []
            neutral_critiques = []

            for disposition in request.critique_dispositions:
                # Validate critique ID exists
                if disposition.critique_id not in available_critiques:
                    self.logger.warning(
                        "invalid_critique_id_in_disposition",
                        critique_id=disposition.critique_id,
                        engagement_id=str(engagement_id),
                    )
                    continue

                # Get full critique details
                critique_details = self._get_critique_details(
                    contract, disposition.critique_id
                )

                # Categorize based on disposition
                disposition_data = {
                    "critique_id": disposition.critique_id,
                    "disposition": disposition.disposition,
                    "rationale": disposition.rationale,
                    "critique_details": critique_details,
                }

                if disposition.disposition == DispositionType.PRIORITIZE:
                    prioritized_critiques.append(disposition_data)
                elif disposition.disposition == DispositionType.DISAGREE:
                    disagreed_critiques.append(disposition_data)
                else:  # NEUTRAL
                    neutral_critiques.append(disposition_data)

            # Process user-generated critique (highest priority)
            user_critique_data = None
            if request.user_generated_critique:
                user_critique_data = {
                    "critique_id": f"user_generated_{engagement_id.hex[:8]}",
                    "type": "user_generated",
                    "title": "User-Generated Critique",
                    "description": request.user_generated_critique,
                    "significance": 1.0,  # Highest significance
                    "source": "user",
                    "timestamp": datetime.utcnow().isoformat(),
                }

            # Update contract with enhanced arbitration data
            if not hasattr(contract, "validation_results"):
                contract.validation_results = {}

            contract.validation_results["enhanced_user_arbitration"] = {
                "prioritized_critiques": prioritized_critiques,
                "disagreed_critiques": disagreed_critiques,
                "neutral_critiques": neutral_critiques,
                "user_generated_critique": user_critique_data,
                "arbitration_timestamp": datetime.utcnow().isoformat(),
                "total_available_critiques": len(available_critiques),
                "disposition_summary": {
                    "prioritized_count": len(prioritized_critiques),
                    "disagreed_count": len(disagreed_critiques),
                    "neutral_count": len(neutral_critiques),
                    "has_user_critique": bool(user_critique_data),
                },
                "flywheel_metadata": {
                    "captured_at": datetime.utcnow().isoformat(),
                    "interaction_type": "enhanced_arbitration",
                    "user_engagement_level": self._calculate_engagement_level(
                        len(prioritized_critiques),
                        len(disagreed_critiques),
                        bool(user_critique_data),
                        len([d for d in disagreed_critiques if d.get("rationale")]),
                    ),
                },
            }

            self.logger.info(
                "enhanced_arbitration_recorded",
                engagement_id=str(engagement_id),
                prioritized_count=len(prioritized_critiques),
                disagreed_count=len(disagreed_critiques),
                neutral_count=len(neutral_critiques),
                has_user_critique=bool(user_critique_data),
                total_available=len(available_critiques),
            )

            # Trigger state transition to SYNTHESIS_DELIVERY
            contract.workflow_state.current_phase = EngagementPhase.SYNTHESIS_DELIVERY

            # Create response
            response = EnhancedArbitrationResponse(
                status="success",
                engagement_id=str(engagement_id),
                prioritized_count=len(prioritized_critiques),
                disagreed_count=len(disagreed_critiques),
                neutral_count=len(neutral_critiques),
                has_user_critique=bool(user_critique_data),
                total_critiques=len(available_critiques),
                synthesis_triggered=True,
                timestamp=datetime.utcnow().isoformat(),
                message=self._generate_response_message(
                    len(prioritized_critiques),
                    len(disagreed_critiques),
                    bool(user_critique_data),
                ),
                flywheel_data_captured=True,
            )

            # Cache for idempotency
            self.arbitration_cache[cache_key] = response

            # Log comprehensive analytics
            self._log_enhanced_analytics(
                engagement_id,
                request,
                prioritized_critiques,
                disagreed_critiques,
                neutral_critiques,
                user_critique_data,
                available_critiques,
            )

            return response

        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(
                "enhanced_arbitration_processing_failed",
                engagement_id=str(engagement_id),
                error=str(e),
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Enhanced arbitration processing failed: {str(e)}",
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

    def _get_critique_details(
        self, contract: MetisDataContract, critique_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get full details for a specific critique"""
        if not hasattr(contract, "validation_results"):
            return None

        for challenger_name in ["munger_critique", "ackoff_critique", "bias_audit"]:
            if challenger_name in contract.validation_results:
                result = contract.validation_results[challenger_name]
                if isinstance(result, dict) and "critiques" in result:
                    for critique in result["critiques"]:
                        if (
                            isinstance(critique, dict)
                            and critique.get("id") == critique_id
                        ):
                            return {**critique, "source_challenger": challenger_name}

        return None

    def _calculate_engagement_level(
        self,
        prioritized_count: int,
        disagreed_count: int,
        has_user_critique: bool,
        rationale_count: int,
    ) -> str:
        """Calculate user engagement level for Flywheel analytics"""
        score = 0
        score += prioritized_count * 2  # Prioritizing shows active engagement
        score += disagreed_count * 3  # Disagreeing shows critical thinking
        score += 5 if has_user_critique else 0  # Custom critique shows deep engagement
        score += rationale_count * 2  # Providing rationales shows thoughtfulness

        if score >= 15:
            return "very_high"
        elif score >= 10:
            return "high"
        elif score >= 5:
            return "medium"
        elif score > 0:
            return "low"
        else:
            return "minimal"

    def _generate_response_message(
        self, prioritized_count: int, disagreed_count: int, has_user_critique: bool
    ) -> str:
        """Generate descriptive response message"""
        parts = []

        if has_user_critique:
            parts.append("User critique will be given highest priority")

        if prioritized_count > 0:
            parts.append(f"{prioritized_count} critiques prioritized for synthesis")

        if disagreed_count > 0:
            parts.append(f"{disagreed_count} critiques marked as not applicable")

        if not parts:
            return "Proceeding with AI's autonomous synthesis"

        return ". ".join(parts) + "."

    def _log_enhanced_analytics(
        self,
        engagement_id: UUID,
        request: EnhancedArbitrationRequest,
        prioritized_critiques: List[Dict],
        disagreed_critiques: List[Dict],
        neutral_critiques: List[Dict],
        user_critique_data: Optional[Dict],
        available_critiques: Set[str],
    ):
        """Log comprehensive analytics for the enhanced arbitration"""

        # Calculate metrics
        total_dispositions = len(request.critique_dispositions)
        rationale_count = len([d for d in request.critique_dispositions if d.rationale])

        # Analyze critique type distributions
        type_distribution = {
            "failure_mode": {"prioritized": 0, "disagreed": 0, "neutral": 0},
            "assumption": {"prioritized": 0, "disagreed": 0, "neutral": 0},
            "bias": {"prioritized": 0, "disagreed": 0, "neutral": 0},
        }

        for critique in prioritized_critiques:
            critique_type = self._get_critique_type(critique["critique_id"])
            if critique_type in type_distribution:
                type_distribution[critique_type]["prioritized"] += 1

        for critique in disagreed_critiques:
            critique_type = self._get_critique_type(critique["critique_id"])
            if critique_type in type_distribution:
                type_distribution[critique_type]["disagreed"] += 1

        for critique in neutral_critiques:
            critique_type = self._get_critique_type(critique["critique_id"])
            if critique_type in type_distribution:
                type_distribution[critique_type]["neutral"] += 1

        self.logger.info(
            "enhanced_arbitration_analytics",
            engagement_id=str(engagement_id),
            total_available=len(available_critiques),
            total_dispositions=total_dispositions,
            prioritized_count=len(prioritized_critiques),
            disagreed_count=len(disagreed_critiques),
            neutral_count=len(neutral_critiques),
            has_user_critique=bool(user_critique_data),
            user_critique_length=(
                len(request.user_generated_critique)
                if request.user_generated_critique
                else 0
            ),
            rationale_count=rationale_count,
            rationale_rate=(
                (rationale_count / len(disagreed_critiques) * 100)
                if disagreed_critiques
                else 0
            ),
            type_distribution=type_distribution,
            engagement_level=self._calculate_engagement_level(
                len(prioritized_critiques),
                len(disagreed_critiques),
                bool(user_critique_data),
                rationale_count,
            ),
        )

    def _get_critique_type(self, critique_id: str) -> str:
        """Determine critique type from ID"""
        if "munger" in critique_id:
            return "failure_mode"
        elif "ackoff" in critique_id:
            return "assumption"
        elif "bias" in critique_id:
            return "bias"
        else:
            return "unknown"


# Initialize service
enhanced_arbitration_service = EnhancedArbitrationService()


@router.post(
    "/{engagement_id}/arbitrate_critiques", response_model=EnhancedArbitrationResponse
)
@feature_flag_required(
    FeatureFlag.ENABLE_ENHANCED_ARBITRATION,
    fallback_response={
        "error": "Enhanced arbitration not available",
        "code": "FEATURE_DISABLED",
    },
    fallback_status_code=404,
)
@enhanced_arbitration_ab_test
async def arbitrate_critiques_enhanced(
    engagement_id: UUID,
    request: EnhancedArbitrationRequest,
    http_request: Request,
    ab_test_group=None,
) -> EnhancedArbitrationResponse:
    """
    Enhanced user arbitration endpoint for Red Team Council critiques

    This endpoint supports:
    1. Three-state disposition (NEUTRAL/PRIORITIZE/DISAGREE)
    2. Optional rationales for disagreements
    3. User-generated critiques (Write-In Candidates)
    4. Complete audit trail for the Flywheel

    The endpoint is idempotent - identical requests return cached results.
    """

    logger.info(
        "enhanced_arbitration_request_received",
        engagement_id=str(engagement_id),
        disposition_count=len(request.critique_dispositions),
        has_user_critique=bool(request.user_generated_critique),
    )

    # TODO: In production, retrieve contract from database
    # For demonstration, create a mock contract
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
                    {
                        "id": "munger.failure_mode.001",
                        "type": "failure_mode",
                        "title": "Resource Constraint Cascade",
                        "description": "Linear scaling assumption fails at 70% utilization",
                        "significance": 0.85,
                    },
                    {
                        "id": "munger.failure_mode.002",
                        "type": "failure_mode",
                        "title": "Single Point of Failure",
                        "description": "Critical dependency on single vendor",
                        "significance": 0.75,
                    },
                ],
            },
            "ackoff_critique": {
                "status": "success",
                "critiques": [
                    {
                        "id": "ackoff.assumption.001",
                        "type": "assumption",
                        "title": "Problem Boundary Too Narrow",
                        "description": "Treats as technical problem, ignoring organizational aspects",
                        "significance": 0.90,
                    },
                    {
                        "id": "ackoff.assumption.002",
                        "type": "assumption",
                        "title": "Stakeholder Misalignment",
                        "description": "Assumes all stakeholders want efficiency over flexibility",
                        "significance": 0.80,
                    },
                ],
            },
            "bias_audit": {
                "status": "success",
                "critiques": [
                    {
                        "id": "bias.confirmation.001",
                        "type": "bias",
                        "title": "Confirmation Bias",
                        "description": "Only cites successful case studies",
                        "significance": 0.85,
                    },
                    {
                        "id": "bias.planning_fallacy.001",
                        "type": "bias",
                        "title": "Planning Fallacy",
                        "description": "Timeline estimates show optimism bias",
                        "significance": 0.75,
                    },
                ],
            },
        },
    )

    # Process enhanced arbitration
    response = await enhanced_arbitration_service.process_enhanced_arbitration(
        engagement_id, request, contract
    )

    return response


@router.get("/{engagement_id}/arbitration_status/enhanced")
async def get_enhanced_arbitration_status(engagement_id: UUID) -> Dict[str, Any]:
    """
    Get enhanced arbitration status including disposition breakdown

    Returns detailed information about:
    - Disposition counts by type
    - User-generated critique status
    - Rationale provision rate
    - Synthesis readiness
    """

    # TODO: In production, retrieve from database
    return {
        "engagement_id": str(engagement_id),
        "arbitration_complete": False,
        "awaiting_user_input": True,
        "total_critiques_available": 6,
        "disposition_breakdown": {"prioritized": 0, "disagreed": 0, "neutral": 6},
        "has_user_critique": False,
        "rationales_provided": 0,
        "synthesis_ready": False,
        "current_phase": "VALIDATION_DEBATE",
        "flywheel_tracking": True,
    }


# Export router for inclusion in main API
__all__ = ["router"]
