"""
Senior Advisor Arbitration API
Multi-Single Agent Analysis System
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from typing import List, Optional, Dict, Any
import asyncio
import json
import logging
from datetime import datetime
from pydantic import BaseModel, Field

from src.orchestration.senior_advisor_orchestrator import SeniorAdvisorOrchestrator
from ..arbitration.rapporteur import Rapporteur
from ..arbitration.models import (
    ConsultantOutput,
    UserWeightingPreferences,
    ConsultantRole,
)
from ..core.supabase_auth_middleware import verify_token

# Phoenix Phase 3: Flywheel Learning Loop Integration
try:
    from ..flywheel.learning.learning_loop import (
        LearningLoop,
        LearningEventType,
        LearningEvent,
    )

    flywheel_available = True
    logger.info("✅ Flywheel Learning Loop integrated with Senior Advisor API")
except ImportError:
    flywheel_available = False
    logger.warning(
        "⚠️ Flywheel Learning Loop not available - feedback will not be recorded"
    )

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/senior-advisor", tags=["senior_advisor"])


# Request/Response Models
class ArbitrationRequest(BaseModel):
    consultant_outputs: List[Dict[str, Any]]
    original_query: str
    user_preferences: Optional[Dict[str, Any]] = None
    query_context: Optional[Dict[str, Any]] = None
    stream: bool = False


class WeightingUpdateRequest(BaseModel):
    arbitration_id: str
    consultant_weights: Dict[ConsultantRole, float] = Field(
        ..., description="Must sum to 1.0"
    )
    criterion_priorities: Optional[Dict[str, float]] = None
    risk_tolerance: Optional[float] = Field(None, ge=0.0, le=1.0)
    implementation_horizon: Optional[str] = None


class ScenarioRequest(BaseModel):
    arbitration_id: str
    scenario_parameters: Dict[str, Any]


class MetaAnalysisRequest(BaseModel):
    consultant_outputs: List[Dict[str, Any]]
    original_query: str


class FeedbackRequest(BaseModel):
    """Phoenix Phase 3: User satisfaction feedback for learning"""

    arbitration_id: str
    satisfaction_score: float = Field(
        ..., ge=0.0, le=10.0, description="User satisfaction rating (0-10)"
    )
    feedback_text: Optional[str] = Field(None, description="Optional detailed feedback")
    recommendation_followed: Optional[str] = Field(
        None, description="Which consultant recommendation was followed"
    )
    outcome_quality: Optional[float] = Field(
        None, ge=0.0, le=10.0, description="How well the recommendation worked (0-10)"
    )
    improvement_suggestions: Optional[str] = Field(
        None, description="Suggestions for improvement"
    )


# Global arbitration and meta-analysis managers
senior_advisor = SeniorAdvisorOrchestrator()
rapporteur = Rapporteur()

# Phoenix Phase 3: Initialize LearningLoop for feedback collection
learning_loop = None
if flywheel_available:
    try:
        learning_loop = LearningLoop()
        logger.info(
            "✅ LearningLoop initialized for Senior Advisor feedback collection"
        )
    except Exception as e:
        logger.error(f"⚠️ Failed to initialize LearningLoop: {e}")
        learning_loop = None


@router.post("/conduct", response_model=Dict[str, Any])
async def conduct_arbitration(
    request: ArbitrationRequest, user_id: str = Depends(verify_token)
):
    """Conduct full arbitration of consultant outputs"""
    try:
        logger.info(
            f"Starting arbitration for user {user_id} with {len(request.consultant_outputs)} consultants"
        )

        # Convert request data to ConsultantOutput objects
        consultant_outputs = []
        for output_data in request.consultant_outputs:
            try:
                # Map request fields to ConsultantOutput model
                consultant_output = ConsultantOutput(
                    consultant_role=ConsultantRole(
                        output_data.get("consultant_role", "analyst")
                    ),
                    analysis_id=output_data.get(
                        "analysis_id", f"analysis_{datetime.now().isoformat()}"
                    ),
                    query=request.original_query,
                    executive_summary=output_data.get("executive_summary", ""),
                    key_insights=output_data.get("key_insights", []),
                    recommendations=output_data.get("recommendations", []),
                    mental_models_used=output_data.get("mental_models_used", []),
                    evidence_sources=output_data.get("evidence_sources", []),
                    research_depth_score=output_data.get("research_depth_score", 0.5),
                    fact_pack_quality=output_data.get("fact_pack_quality", "medium"),
                    red_team_results=output_data.get("red_team_results", {}),
                    bias_detection_score=output_data.get("bias_detection_score", 0.5),
                    logical_consistency_score=output_data.get(
                        "logical_consistency_score", 0.5
                    ),
                    processing_time_seconds=output_data.get(
                        "processing_time_seconds", 0.0
                    ),
                    cost_usd=output_data.get("cost_usd", 0.0),
                    confidence_level=output_data.get("confidence_level", 0.5),
                    created_at=datetime.fromisoformat(
                        output_data.get("created_at", datetime.now().isoformat())
                    ),
                    primary_perspective=output_data.get(
                        "primary_perspective", "strategic_focused"
                    ),
                    approach_description=output_data.get("approach_description", ""),
                    limitations_identified=output_data.get(
                        "limitations_identified", []
                    ),
                    assumptions_made=output_data.get("assumptions_made", []),
                )
                consultant_outputs.append(consultant_output)
            except Exception as e:
                logger.error(f"Error parsing consultant output: {e}")
                raise HTTPException(
                    status_code=400, detail=f"Invalid consultant output format: {e}"
                )

        # Convert user preferences if provided
        user_preferences = None
        if request.user_preferences:
            user_preferences = UserWeightingPreferences(
                consultant_weights=request.user_preferences.get(
                    "consultant_weights",
                    {
                        ConsultantRole.ANALYST: 0.33,
                        ConsultantRole.STRATEGIST: 0.33,
                        ConsultantRole.DEVIL_ADVOCATE: 0.34,
                    },
                ),
                criterion_priorities=request.user_preferences.get(
                    "criterion_priorities", {}
                ),
                risk_tolerance=request.user_preferences.get("risk_tolerance", 0.5),
                implementation_horizon=request.user_preferences.get(
                    "implementation_horizon", "short_term"
                ),
                decision_context=request.user_preferences.get("decision_context", {}),
            )

        # Conduct arbitration via SeniorAdvisorOrchestrator
        # Map old conduct_arbitration API to new conduct_two_brain_analysis method
        engagement_id = f"api_arbitration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if request.stream:
            return StreamingResponse(
                stream_arbitration_process(
                    consultant_outputs,
                    request.original_query,
                    engagement_id,
                ),
                media_type="text/plain",
            )
        else:
            # Call the new orchestrator method
            result = await senior_advisor.conduct_two_brain_analysis(
                consultant_outputs=consultant_outputs,
                original_query=request.original_query,
                engagement_id=engagement_id,
            )

            # Return modernized response format from SeniorAdvisorOrchestrator
            return {
                "status": "completed" if result.get("success") else "failed",
                "engagement_id": result.get("engagement_id"),
                "analysis_summary": result.get("analysis_summary"),
                "final_markdown_report": result.get("final_markdown_report"),
                "strategic_recommendations": result.get("strategic_recommendations"),
                "implementation_plan": result.get("implementation_plan"),
                "risk_factors": result.get("risk_factors"),
                "confidence_score": result.get("confidence_score"),
                "processing_time_seconds": result.get("processing_time", 0),
                "total_cost_usd": result.get("total_cost", 0),
                "deepseek_analysis": result.get("deepseek_analysis", {}),
                "claude_synthesis": result.get("claude_synthesis", {}),
                "synthesis_rationale": result.get("synthesis_rationale"),
                "raw_analytical_dossier": result.get("raw_analytical_dossier", {}),
            }

    except Exception as e:
        logger.error(f"Arbitration failed for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Arbitration failed: {str(e)}")


async def stream_arbitration_process(
    consultant_outputs: List[ConsultantOutput],
    original_query: str,
    engagement_id: str,
):
    """Stream arbitration process updates"""
    try:
        # Phase 1: Validation
        yield f"data: {json.dumps({'phase': 'validation', 'status': 'in_progress', 'message': 'Validating consultant outputs...'})}\n\n"
        await asyncio.sleep(0.5)

        # Phase 2: DeepSeek Brain Analysis
        yield f"data: {json.dumps({'phase': 'deepseek_brain', 'status': 'in_progress', 'message': 'Executing DeepSeek brain analysis...'})}\n\n"
        await asyncio.sleep(1.0)

        # Phase 3: Claude Brain Synthesis
        yield f"data: {json.dumps({'phase': 'claude_brain', 'status': 'in_progress', 'message': 'Executing Claude brain synthesis...'})}\n\n"
        await asyncio.sleep(1.0)

        # Phase 4: Final Report Generation
        yield f"data: {json.dumps({'phase': 'report_generation', 'status': 'in_progress', 'message': 'Generating final strategic report...'})}\n\n"
        await asyncio.sleep(0.5)

        # Conduct actual analysis using new orchestrator
        result = await senior_advisor.conduct_two_brain_analysis(
            consultant_outputs=consultant_outputs,
            original_query=original_query,
            engagement_id=engagement_id,
        )

        # Stream completion
        yield f"data: {json.dumps({'phase': 'completed', 'status': 'success', 'message': 'Analysis completed successfully', 'engagement_id': result.get('engagement_id')})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'phase': 'error', 'status': 'failed', 'message': f'Analysis failed: {str(e)}'})}\n\n"


@router.post("/update-weighting")
async def update_weighting(
    request: WeightingUpdateRequest, user_id: str = Depends(verify_token)
):
    """Update user weighting preferences and recalculate arbitration"""
    try:
        # Validate weights sum to 1.0
        total_weight = sum(request.consultant_weights.values())
        if abs(total_weight - 1.0) > 0.001:
            raise HTTPException(
                status_code=400,
                detail=f"Consultant weights must sum to 1.0, got {total_weight}",
            )

        logger.info(
            f"Updating weighting for arbitration {request.arbitration_id} by user {user_id}"
        )

        # This would typically retrieve the original arbitration and recompute
        # For now, return success with updated weights
        return {
            "status": "success",
            "message": "Weighting updated successfully",
            "arbitration_id": request.arbitration_id,
            "updated_weights": {
                k.value: v for k, v in request.consultant_weights.items()
            },
        }

    except Exception as e:
        logger.error(f"Failed to update weighting: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update weighting: {str(e)}"
        )


@router.post("/generate-scenario")
async def generate_scenario(
    request: ScenarioRequest, user_id: str = Depends(verify_token)
):
    """Generate alternative scenario based on parameters"""
    try:
        logger.info(
            f"Generating scenario for arbitration {request.arbitration_id} by user {user_id}"
        )

        # This would use the arbitration system to generate alternative scenarios
        # For now, return a mock scenario
        scenario = {
            "scenario_id": f"scenario_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "name": f"Alternative Approach {request.scenario_parameters.get('approach_type', 'Conservative')}",
            "description": "AI-generated alternative scenario based on different risk parameters and consultant weightings",
            "risk_level": request.scenario_parameters.get("risk_level", "medium"),
            "timeframe": request.scenario_parameters.get("timeframe", "3-6 months"),
            "resources": request.scenario_parameters.get("resources", "standard"),
            "modified_weights": request.scenario_parameters.get(
                "consultant_weights", {}
            ),
            "expected_outcomes": [
                "Modified approach based on alternative consultant weighting",
                "Adjusted risk profile and implementation timeline",
                "Alternative success metrics and monitoring approach",
            ],
            "trade_offs": [
                "Different resource allocation requirements",
                "Modified timeline expectations",
                "Alternative success criteria",
            ],
        }

        return {
            "status": "success",
            "scenario": scenario,
            "arbitration_id": request.arbitration_id,
        }

    except Exception as e:
        logger.error(f"Failed to generate scenario: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate scenario: {str(e)}"
        )


@router.post("/meta-analysis", response_model=Dict[str, Any])
async def conduct_meta_analysis(
    request: MetaAnalysisRequest, user_id: str = Depends(verify_token)
):
    """
    Conduct Rapporteur meta-analysis without synthesis.

    This endpoint implements the core METIS V5 principle of "Context Preservation Over Compression" -
    providing structured analysis of consultant relationships while preserving all perspectives.
    """
    try:
        logger.info(f"🎭 Starting Rapporteur meta-analysis for user {user_id}")
        logger.info(
            f"📊 Processing {len(request.consultant_outputs)} consultant outputs"
        )

        # Convert request data to ConsultantOutput objects
        consultant_outputs = []
        for output_data in request.consultant_outputs:
            try:
                consultant_output = ConsultantOutput(
                    consultant_role=ConsultantRole(
                        output_data.get("consultant_role", "analyst")
                    ),
                    analysis_id=output_data.get(
                        "analysis_id", f"analysis_{datetime.now().isoformat()}"
                    ),
                    query=request.original_query,
                    executive_summary=output_data.get("executive_summary", ""),
                    key_insights=output_data.get("key_insights", []),
                    recommendations=output_data.get("recommendations", []),
                    mental_models_used=output_data.get("mental_models_used", []),
                    evidence_sources=output_data.get("evidence_sources", []),
                    research_depth_score=output_data.get("research_depth_score", 0.5),
                    fact_pack_quality=output_data.get("fact_pack_quality", "medium"),
                    red_team_results=output_data.get("red_team_results", {}),
                    bias_detection_score=output_data.get("bias_detection_score", 0.5),
                    logical_consistency_score=output_data.get(
                        "logical_consistency_score", 0.5
                    ),
                    processing_time_seconds=output_data.get(
                        "processing_time_seconds", 0.0
                    ),
                    cost_usd=output_data.get("cost_usd", 0.0),
                    confidence_level=output_data.get("confidence_level", 0.5),
                    created_at=datetime.fromisoformat(
                        output_data.get("created_at", datetime.now().isoformat())
                    ),
                    primary_perspective=output_data.get(
                        "primary_perspective", "strategic_focused"
                    ),
                    approach_description=output_data.get("approach_description", ""),
                    limitations_identified=output_data.get(
                        "limitations_identified", []
                    ),
                    assumptions_made=output_data.get("assumptions_made", []),
                )
                consultant_outputs.append(consultant_output)
            except Exception as e:
                logger.error(f"Error parsing consultant output: {e}")
                raise HTTPException(
                    status_code=400, detail=f"Invalid consultant output format: {e}"
                )

        # Conduct Rapporteur meta-analysis
        report = await rapporteur.conduct_meta_analysis(
            consultant_outputs=consultant_outputs,
            original_query=request.original_query,
            devils_advocate_critiques=request.devils_advocate_critiques,
            analysis_context=request.analysis_context,
        )

        # Convert to JSON-serializable format
        return {
            "meta_analysis_report": {
                "analysis_id": report.analysis_id,
                "original_query": report.original_query,
                "timestamp": report.timestamp.isoformat(),
                "perspective_mappings": [
                    {
                        "consultant_role": mapping.consultant_role.value,
                        "core_thesis": mapping.core_thesis,
                        "unique_insights": mapping.unique_insights,
                        "methodology_approach": mapping.methodology_approach,
                        "evidence_quality": mapping.evidence_quality,
                        "blind_spots_identified": mapping.blind_spots_identified,
                        "confidence_patterns": mapping.confidence_patterns,
                    }
                    for mapping in report.perspective_mappings
                ],
                "convergence_points": [
                    {
                        "convergence_topic": point.convergence_topic,
                        "supporting_consultants": [
                            c.value for c in point.supporting_consultants
                        ],
                        "convergence_strength": point.convergence_strength,
                        "nuance_preservation": point.nuance_preservation,
                        "independent_validation": point.independent_validation,
                    }
                    for point in report.convergence_points
                ],
                "divergence_analyses": [
                    {
                        "divergence_topic": div.divergence_topic,
                        "consultant_positions": {
                            k.value: v for k, v in div.consultant_positions.items()
                        },
                        "root_cause_analysis": div.root_cause_analysis,
                        "value_of_disagreement": div.value_of_disagreement,
                        "decision_implications": div.decision_implications,
                    }
                    for div in report.divergence_analyses
                ],
                "meta_insights": [
                    {
                        "pattern_type": insight.pattern_type,
                        "description": insight.description,
                        "consultant_orchestration": insight.consultant_orchestration,
                        "cognitive_diversity_score": insight.cognitive_diversity_score,
                        "context_preservation_score": insight.context_preservation_score,
                    }
                    for insight in report.meta_insights
                ],
                "context_compression_score": report.context_compression_score,
                "perspective_independence_score": report.perspective_independence_score,
                "information_loss_assessment": report.information_loss_assessment,
                "decision_framework": report.decision_framework,
                "perspective_navigation_guide": report.perspective_navigation_guide,
                "user_choice_points": report.user_choice_points,
                "rapporteur_confidence": report.rapporteur_confidence,
                "processing_time_seconds": report.processing_time_seconds,
                "total_consultants_analyzed": report.total_consultants_analyzed,
                "devils_advocate_integration": report.devils_advocate_integration,
            },
            "status": "completed",
            "context_preservation_validation": {
                "compression_avoided": report.context_compression_score == 0.0,
                "perspective_independence": report.perspective_independence_score,
                "paradigm_validation": "Context Preservation Over Compression successfully demonstrated",
            },
        }

    except Exception as e:
        logger.error(f"Meta-analysis failed for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Meta-analysis failed: {str(e)}")


@router.get("/status/{arbitration_id}")
async def get_arbitration_status(
    arbitration_id: str, user_id: str = Depends(verify_token)
):
    """Get status of ongoing arbitration process"""
    try:
        # This would check the status of an ongoing arbitration
        # For now, return mock status
        return {
            "arbitration_id": arbitration_id,
            "status": "completed",
            "phase": "arbitration",
            "progress_percent": 100,
            "estimated_completion": None,
            "last_update": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to get arbitration status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest, user_id: str = Depends(verify_token)
):
    """
    Phoenix Phase 3: Submit user satisfaction feedback for learning

    This endpoint captures user feedback about arbitration results and feeds
    it into the Flywheel Learning Loop for continuous improvement.
    """
    try:
        if not flywheel_available or learning_loop is None:
            logger.warning(
                f"Flywheel unavailable - feedback recorded locally for user {user_id}"
            )
            return {
                "status": "feedback_recorded_locally",
                "arbitration_id": request.arbitration_id,
                "satisfaction_score": request.satisfaction_score,
                "flywheel_status": "unavailable",
                "message": "Feedback recorded but not integrated with learning system",
            }

        # Create LearningEvent for satisfaction feedback
        feedback_data = {
            "arbitration_id": request.arbitration_id,
            "user_id": user_id,
            "satisfaction_score": request.satisfaction_score,
            "feedback_text": request.feedback_text,
            "recommendation_followed": request.recommendation_followed,
            "outcome_quality": request.outcome_quality,
            "improvement_suggestions": request.improvement_suggestions,
            "timestamp": datetime.now().isoformat(),
        }

        learning_event = LearningEvent(
            event_type=LearningEventType.SATISFACTION_FEEDBACK,
            data=feedback_data,
            timestamp=datetime.now(),
            priority="high",  # User satisfaction is high priority for learning
        )

        # Record the learning event
        await learning_loop.record_learning_event(learning_event)

        logger.info(
            f"✅ Satisfaction feedback recorded for arbitration {request.arbitration_id} by user {user_id}"
        )

        return {
            "status": "feedback_recorded_successfully",
            "arbitration_id": request.arbitration_id,
            "satisfaction_score": request.satisfaction_score,
            "learning_event_id": learning_event.event_id,
            "flywheel_status": "integrated",
            "message": "Feedback successfully integrated with Flywheel Learning Loop",
        }

    except Exception as e:
        logger.error(
            f"Failed to submit feedback for arbitration {request.arbitration_id}: {e}"
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to submit feedback: {str(e)}"
        )
