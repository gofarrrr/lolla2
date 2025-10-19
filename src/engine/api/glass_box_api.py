"""
Glass-Box Audit Trail API
RESTful endpoints for comprehensive cognitive process transparency

Provides access to complete audit trails, decision points, and transparency layers
for the METIS Glass-Box system.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import UUID
import logging

from src.engine.core.glass_box_orchestrator import GlassBoxOrchestrator
from src.engine.adapters.core.supabase_auth_middleware import get_current_user, SupabaseUser
from src.models.transparency_models import UserExpertiseLevel, TransparencyLayer
from src.engine.adapters.core.unified_context_stream import UnifiedContextStream
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/glass-box", tags=["glass_box_audit"])

# Global orchestrator instance
glass_box_orchestrator = GlassBoxOrchestrator()


# Request/Response Models
class GlassBoxAnalysisRequest(BaseModel):
    query: str
    context: Optional[str] = None
    use_query_enhancement: bool = True
    user_expertise_level: UserExpertiseLevel = UserExpertiseLevel.STRATEGIC
    capture_technical_details: bool = True


class DevilsAdvocateCritiqueRequest(BaseModel):
    engagement_id: str
    consultant_to_critique: str
    critique_depth: str = "standard"  # standard, deep, comprehensive


class SeniorAdvisorArbitrationRequest(BaseModel):
    engagement_id: str
    user_preferences: Optional[Dict[str, Any]] = None
    weighting_priorities: Optional[Dict[str, float]] = None


class AuditTrailExportRequest(BaseModel):
    engagement_id: str
    format: str = "json"  # json, csv, pdf
    include_technical_details: bool = True
    include_llm_prompts: bool = False  # Privacy consideration


# 🔍 GLASS-BOX EVIDENCE API MODELS
class EvidenceQueryRequest(BaseModel):
    trace_id: Optional[str] = None
    evidence_types: Optional[List[str]] = (
        None  # ["consultant_selection", "synergy_directive", "coreops_execution", "contradiction_audit"]
    )
    include_timeline: bool = True
    include_confidence_scores: bool = True


class TraceEvidenceRequest(BaseModel):
    trace_id: str
    export_format: str = "json"  # json, summary, timeline
    include_raw_data: bool = False


# 🔍 GLASS-BOX EVIDENCE API ENDPOINTS


@router.get("/evidence/{trace_id}")
async def get_trace_evidence(
    trace_id: str,
    evidence_types: Optional[str] = Query(
        None, description="Comma-separated evidence types to filter"
    ),
    include_timeline: bool = Query(True, description="Include evidence timeline"),
    user: SupabaseUser = Depends(get_current_user),
):
    """Get glass-box evidence for a specific trace ID"""
    try:
        # For now, we'll create a mock context stream with the trace_id
        # In production, this would load from persistent storage
        from src.engine.adapters.core.unified_context_stream import get_unified_context_stream
        context_stream = get_unified_context_stream()
        context_stream.trace_id = trace_id

        # Parse evidence types filter
        evidence_type_filter = None
        if evidence_types:
            evidence_type_filter = [t.strip() for t in evidence_types.split(",")]

        # Get evidence events (this would be loaded from storage in production)
        evidence_events = context_stream.get_evidence_events()

        # Generate evidence summary
        evidence_summary = context_stream.get_evidence_summary()

        # Export for API consumption
        api_evidence = context_stream.export_evidence_for_api()

        return {
            "status": "success",
            "trace_id": trace_id,
            "evidence_summary": evidence_summary,
            "evidence_data": api_evidence,
            "filters_applied": {
                "evidence_types": evidence_type_filter,
                "include_timeline": include_timeline,
            },
            "metadata": {
                "total_evidence_events": len(evidence_events),
                "collection_timestamp": datetime.now().isoformat(),
                "evidence_completeness": evidence_summary.get(
                    "glass_box_completeness", 0
                ),
            },
        }

    except Exception as e:
        logger.error(f"Failed to retrieve evidence for trace {trace_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve evidence: {str(e)}"
        )


@router.get("/evidence/summary/{trace_id}")
async def get_evidence_summary(
    trace_id: str, user: SupabaseUser = Depends(get_current_user)
):
    """Get high-level evidence summary for a trace"""
    try:
        # Mock context stream for the trace
        context_stream = get_unified_context_stream()
        context_stream.trace_id = trace_id

        # Get evidence summary
        evidence_summary = context_stream.get_evidence_summary()

        return {
            "status": "success",
            "trace_id": trace_id,
            "summary": evidence_summary,
            "key_insights": {
                "total_decisions": len(evidence_summary.get("key_decisions", [])),
                "glass_box_completeness": evidence_summary.get(
                    "glass_box_completeness", 0
                ),
                "evidence_types_captured": list(
                    evidence_summary.get("evidence_types", {}).keys()
                ),
                "session_metrics": {
                    "consultant_selections": evidence_summary.get(
                        "consultant_selections", 0
                    ),
                    "synergy_directives": evidence_summary.get("synergy_directives", 0),
                    "coreops_executions": evidence_summary.get("coreops_executions", 0),
                    "contradiction_audits": evidence_summary.get(
                        "contradiction_audits", 0
                    ),
                },
            },
        }

    except Exception as e:
        logger.error(f"Failed to get evidence summary for trace {trace_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get evidence summary: {str(e)}"
        )


@router.get("/evidence/consultant-selections/{trace_id}")
async def get_consultant_selection_evidence(
    trace_id: str, user: SupabaseUser = Depends(get_current_user)
):
    """Get consultant selection evidence for a trace"""
    try:
        context_stream = get_unified_context_stream()
        context_stream.trace_id = trace_id

        # Get consultant selection evidence
        selection_events = context_stream.get_consultant_selection_evidence()

        # Format for API response
        selections = []
        for event in selection_events:
            data = event.data
            selections.append(
                {
                    "timestamp": event.timestamp.isoformat(),
                    "event_id": event.event_id,
                    "selection_rationale": data.get("selection_rationale", ""),
                    "total_confidence": data.get("total_confidence", 0),
                    "consultant_count": data.get("consultant_count", 0),
                    "consultants": data.get("consultants", []),
                    "risk_factors": data.get("risk_factors", []),
                    "success_factors": data.get("success_factors", []),
                    "chemistry_score": data.get("final_chemistry_score", 0),
                }
            )

        return {
            "status": "success",
            "trace_id": trace_id,
            "consultant_selections": selections,
            "total_selections": len(selections),
            "metadata": {
                "evidence_type": "consultant_selection",
                "audit_level": "complete",
            },
        }

    except Exception as e:
        logger.error(
            f"Failed to get consultant selection evidence for trace {trace_id}: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get consultant selection evidence: {str(e)}",
        )


@router.get("/evidence/synergy-directives/{trace_id}")
async def get_synergy_directive_evidence(
    trace_id: str, user: SupabaseUser = Depends(get_current_user)
):
    """Get mental model synergy directive evidence for a trace"""
    try:
        context_stream = get_unified_context_stream()
        context_stream.trace_id = trace_id

        # Get synergy evidence
        synergy_events = context_stream.get_synergy_evidence()

        # Format for API response
        directives = []
        for event in synergy_events:
            data = event.data
            directives.append(
                {
                    "timestamp": event.timestamp.isoformat(),
                    "event_id": event.event_id,
                    "meta_directive": data.get("meta_directive", ""),
                    "synergy_insight": data.get("synergy_insight", ""),
                    "conflict_insight": data.get("conflict_insight", ""),
                    "confidence_score": data.get("confidence_score", 0),
                    "participating_models": data.get("participating_models", []),
                    "model_count": data.get("model_count", 0),
                    "processing_time_ms": data.get("processing_time_ms", 0),
                    "analysis_model": data.get("analysis_model", "unknown"),
                }
            )

        return {
            "status": "success",
            "trace_id": trace_id,
            "synergy_directives": directives,
            "total_directives": len(directives),
            "metadata": {
                "evidence_type": "synergy_directive",
                "audit_level": "complete",
            },
        }

    except Exception as e:
        logger.error(
            f"Failed to get synergy directive evidence for trace {trace_id}: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get synergy directive evidence: {str(e)}",
        )


@router.get("/evidence/coreops-executions/{trace_id}")
async def get_coreops_execution_evidence(
    trace_id: str, user: SupabaseUser = Depends(get_current_user)
):
    """Get V2 CoreOps execution evidence for a trace"""
    try:
        context_stream = get_unified_context_stream()
        context_stream.trace_id = trace_id

        # Get CoreOps evidence
        coreops_events = context_stream.get_coreops_evidence()

        # Format for API response
        executions = []
        for event in coreops_events:
            data = event.data
            executions.append(
                {
                    "timestamp": event.timestamp.isoformat(),
                    "event_id": event.event_id,
                    "system_contract_id": data.get("system_contract_id", ""),
                    "program_path": data.get("program_path", ""),
                    "step_count": data.get("step_count", 0),
                    "argument_count": data.get("argument_count", 0),
                    "sample_claims": data.get("sample_claims", []),
                    "processing_time_ms": data.get("processing_time_ms", 0),
                    "execution_mode": data.get("execution_mode", ""),
                    "v2_proof": data.get("v2_proof", False),
                }
            )

        return {
            "status": "success",
            "trace_id": trace_id,
            "coreops_executions": executions,
            "total_executions": len(executions),
            "metadata": {
                "evidence_type": "coreops_execution",
                "audit_level": "complete",
            },
        }

    except Exception as e:
        logger.error(
            f"Failed to get CoreOps execution evidence for trace {trace_id}: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get CoreOps execution evidence: {str(e)}",
        )


@router.get("/evidence/demo")
async def get_demo_evidence():
    """Get demo evidence data for frontend development"""

    # Create demo evidence matching our evidence structure
    demo_evidence = {
        "trace_id": "demo-trace-evidence-12345",
        "evidence_summary": {
            "total_evidence_events": 4,
            "glass_box_completeness": 1.0,
            "consultant_selections": 1,
            "synergy_directives": 1,
            "coreops_executions": 1,
            "contradiction_audits": 1,
            "key_decisions": [
                {
                    "type": "consultant_selection",
                    "timestamp": "2024-01-15T10:30:02Z",
                    "rationale": "EXCELLENT - Highly effective combination with strong synergy between strategic and financial analysis",
                    "confidence": 0.85,
                    "consultant_count": 2,
                },
                {
                    "type": "synergy_directive",
                    "timestamp": "2024-01-15T10:30:05Z",
                    "meta_directive": "Integrate strategic thinking with financial modeling to create comprehensive analysis",
                    "confidence": 0.82,
                    "model_count": 3,
                },
            ],
        },
        "consultant_selections": [
            {
                "timestamp": "2024-01-15T10:30:02Z",
                "selection_rationale": "EXCELLENT - Highly effective combination with strong synergy between strategic and financial analysis",
                "total_confidence": 0.85,
                "consultant_count": 2,
                "chemistry_score": 0.78,
                "consultants": [
                    {
                        "consultant_id": "strategic_analyst@1.0",
                        "consultant_type": "strategic_analyst",
                        "synergy_score": 0.9,
                        "domain_match_score": 0.8,
                        "why_selected": [
                            "High strategic capability",
                            "Strong analytical framework",
                        ],
                        "top_features": [
                            "systems-thinking=0.95",
                            "scenario-analysis=0.88",
                        ],
                    },
                    {
                        "consultant_id": "financial_analyst@1.0",
                        "consultant_type": "financial_analyst",
                        "synergy_score": 0.75,
                        "domain_match_score": 0.85,
                        "why_selected": [
                            "Financial modeling expertise",
                            "Quantitative analysis",
                        ],
                        "top_features": [
                            "dcf-valuation=0.92",
                            "financial-modeling=0.89",
                        ],
                    },
                ],
                "risk_factors": ["High complexity may slow analysis"],
                "success_factors": [
                    "Excellent synergy between strategic and financial perspectives"
                ],
            }
        ],
        "synergy_directives": [
            {
                "timestamp": "2024-01-15T10:30:05Z",
                "meta_directive": "Integrate strategic thinking with financial modeling to create comprehensive analysis that leverages both frameworks",
                "synergy_insight": "Strategic frameworks enhance financial modeling accuracy by providing market context and competitive positioning",
                "conflict_insight": "Strategic breadth may conflict with financial precision, requiring balanced approach to maintain rigor",
                "confidence_score": 0.82,
                "participating_models": [
                    "systems-thinking",
                    "scenario-analysis",
                    "dcf-valuation",
                    "financial-modeling",
                ],
                "model_count": 4,
                "processing_time_ms": 1250,
                "analysis_model": "deepseek-chat",
            }
        ],
        "coreops_executions": [
            {
                "timestamp": "2024-01-15T10:30:08Z",
                "system_contract_id": "financial_analyst@1.0",
                "program_path": "examples/coreops/financial_analyst.yaml",
                "step_count": 4,
                "argument_count": 12,
                "sample_claims": [
                    "DCF Valuation: Company valued at $2.5B based on projected cash flows and 8.5% WACC",
                    "Scenario Analysis: 3 scenarios show value range $1.8B - $3.2B with base case at $2.5B",
                ],
                "processing_time_ms": 2100,
                "execution_mode": "v2_coreops",
                "v2_proof": True,
            }
        ],
        "contradiction_audits": [
            {
                "timestamp": "2024-01-15T10:30:12Z",
                "contradiction_count": 2,
                "synthesis_count": 5,
                "example_contradiction": "Optimistic growth assumptions conflict with market saturation indicators",
                "example_synthesis": "Market constraints integrated with revised growth projections for more realistic forecast",
                "confidence_score": 0.78,
                "bias_mitigation_applied": True,
            }
        ],
        "evidence_timeline": [
            {
                "timestamp": "2024-01-15T10:30:02Z",
                "event_type": "model_selection_justification",
                "description": "Selected 2 consultants with 85.0% confidence",
                "confidence": 0.85,
                "processing_time_ms": 450,
            },
            {
                "timestamp": "2024-01-15T10:30:05Z",
                "event_type": "synergy_meta_directive",
                "description": "Generated meta-directive from 4 models with 82.0% confidence",
                "confidence": 0.82,
                "processing_time_ms": 1250,
            },
            {
                "timestamp": "2024-01-15T10:30:08Z",
                "event_type": "coreops_run_summary",
                "description": "Executed financial_analyst@1.0 generating 12 arguments",
                "confidence": 0.9,
                "processing_time_ms": 2100,
            },
            {
                "timestamp": "2024-01-15T10:30:12Z",
                "event_type": "contradiction_audit",
                "description": "Found 2 contradictions, 5 syntheses",
                "confidence": 0.78,
                "processing_time_ms": 800,
            },
        ],
    }

    return {
        "status": "success",
        "demo_evidence": demo_evidence,
        "note": "This is demo evidence data for frontend development and testing purposes",
    }


@router.post("/analyze", response_model=Dict[str, Any])
async def analyze_with_glass_box_transparency(
    request: GlassBoxAnalysisRequest, user: SupabaseUser = Depends(get_current_user)
):
    """Execute cognitive analysis with complete glass-box audit trail capture"""
    try:
        logger.info(f"Starting glass-box analysis for user {user_id}")

        # Execute analysis with comprehensive audit trail
        result = await glass_box_orchestrator.analyze_with_complete_audit_trail(
            query=request.query,
            context=request.context,
            user_id=UUID(user_id),
            use_query_enhancement=request.use_query_enhancement,
        )

        # Adapt transparency based on user expertise level
        transparency_layers = result["glass_box_data"]["transparency_layers"]

        # Filter transparency layers based on user expertise
        if request.user_expertise_level == UserExpertiseLevel.EXECUTIVE:
            # Executive users get summary view only
            filtered_layers = {
                "executive_summary": transparency_layers["executive_summary"],
                "reasoning_overview": transparency_layers["reasoning_overview"],
            }
        elif request.user_expertise_level == UserExpertiseLevel.STRATEGIC:
            # Strategic users get executive + reasoning + audit trail
            filtered_layers = {
                k: v
                for k, v in transparency_layers.items()
                if k != "technical_execution"
            }
        else:
            # Analytical and Technical users get all layers
            filtered_layers = transparency_layers

        # Remove technical details if not requested
        if not request.capture_technical_details:
            filtered_layers.pop("technical_execution", None)

        return {
            "status": "success",
            "engagement_id": result["engagement_id"],
            "analysis_result": result["analysis_result"],
            "audit_summary": result["audit_trail"],
            "transparency_layers": filtered_layers,
            "glass_box_metrics": {
                "decision_points_captured": result["glass_box_data"][
                    "total_decision_points"
                ],
                "reconstructable": result["glass_box_data"]["reconstructable"],
                "transparency_level": "complete",
            },
            "user_adapted_view": request.user_expertise_level.value,
        }

    except Exception as e:
        logger.error(f"Glass-box analysis failed for user {user_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Glass-box analysis failed: {str(e)}"
        )


@router.get("/audit-trail/{engagement_id}")
async def get_engagement_audit_trail(
    engagement_id: str,
    transparency_layer: Optional[TransparencyLayer] = Query(
        None, description="Specific transparency layer to retrieve"
    ),
    include_technical: bool = Query(
        True, description="Include technical execution details"
    ),
    user: SupabaseUser = Depends(get_current_user),
):
    """Get complete audit trail for a specific engagement"""
    try:
        engagement_uuid = UUID(engagement_id)

        # Get audit trail from orchestrator
        audit_data = await glass_box_orchestrator.get_engagement_audit_trail(
            engagement_uuid
        )

        if not audit_data:
            raise HTTPException(
                status_code=404,
                detail=f"Audit trail not found for engagement {engagement_id}",
            )

        # Filter by transparency layer if requested
        if transparency_layer:
            layer_data = audit_data["transparency_layers"].get(transparency_layer.value)
            if not layer_data:
                raise HTTPException(
                    status_code=404,
                    detail=f"Transparency layer {transparency_layer.value} not available",
                )

            return {
                "engagement_id": engagement_id,
                "transparency_layer": transparency_layer.value,
                "layer_data": layer_data,
                "audit_summary": audit_data["audit_trail"],
            }

        # Return complete audit trail
        response_data = audit_data.copy()

        # Remove technical details if not requested
        if not include_technical and "technical_execution" in response_data.get(
            "transparency_layers", {}
        ):
            del response_data["transparency_layers"]["technical_execution"]

        return response_data

    except ValueError:
        raise HTTPException(
            status_code=400, detail=f"Invalid engagement ID format: {engagement_id}"
        )
    except Exception as e:
        logger.error(
            f"Failed to retrieve audit trail for engagement {engagement_id}: {e}"
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve audit trail: {str(e)}"
        )


@router.post("/request-critique/{engagement_id}")
async def request_devils_advocate_critique(
    engagement_id: str,
    request: DevilsAdvocateCritiqueRequest,
    user: SupabaseUser = Depends(get_current_user),
):
    """Request Devil's Advocate critique with audit trail capture"""
    try:
        engagement_uuid = UUID(engagement_id)

        logger.info(
            f"Requesting Devil's Advocate critique for engagement {engagement_id} by user {user_id}"
        )

        # Execute Devil's Advocate critique
        result = await glass_box_orchestrator.request_devils_advocate_critique(
            engagement_id=engagement_uuid,
            consultant_to_critique=request.consultant_to_critique,
        )

        return {
            "status": "success",
            "engagement_id": engagement_id,
            "critique_result": result,
            "audit_captured": result.get("audit_captured", False),
            "message": f"Devil's Advocate critique of {request.consultant_to_critique} completed",
        }

    except ValueError:
        raise HTTPException(
            status_code=400, detail=f"Invalid engagement ID: {engagement_id}"
        )
    except Exception as e:
        logger.error(f"Devil's Advocate critique failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Devil's Advocate critique failed: {str(e)}"
        )


@router.post("/request-arbitration/{engagement_id}")
async def request_senior_advisor_arbitration(
    engagement_id: str,
    request: SeniorAdvisorArbitrationRequest,
    user: SupabaseUser = Depends(get_current_user),
):
    """Request Senior Advisor arbitration with audit trail capture"""
    try:
        engagement_uuid = UUID(engagement_id)

        logger.info(
            f"Requesting Senior Advisor arbitration for engagement {engagement_id} by user {user_id}"
        )

        # Execute Senior Advisor arbitration
        result = await glass_box_orchestrator.request_senior_advisor_arbitration(
            engagement_id=engagement_uuid, user_preferences=request.user_preferences
        )

        return {
            "status": "success",
            "engagement_id": engagement_id,
            "arbitration_result": result,
            "audit_captured": result.get("audit_captured", False),
            "message": "Senior Advisor arbitration completed",
        }

    except ValueError:
        raise HTTPException(
            status_code=400, detail=f"Invalid engagement ID: {engagement_id}"
        )
    except Exception as e:
        logger.error(f"Senior Advisor arbitration failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Senior Advisor arbitration failed: {str(e)}"
        )


@router.post("/export/{engagement_id}")
async def export_audit_trail(
    engagement_id: str,
    request: AuditTrailExportRequest,
    user: SupabaseUser = Depends(get_current_user),
):
    """Export complete audit trail for compliance or analysis"""
    try:
        engagement_uuid = UUID(engagement_id)

        logger.info(
            f"Exporting audit trail for engagement {engagement_id} by user {user_id}"
        )

        # Get export data
        export_data = await glass_box_orchestrator.export_audit_trail(
            engagement_id=engagement_uuid, format=request.format
        )

        # Filter sensitive data if requested
        if not request.include_llm_prompts:
            # Remove LLM prompts for privacy
            if "detailed_audit_trail" in export_data:
                for section_key, section_data in export_data[
                    "detailed_audit_trail"
                ].items():
                    if (
                        isinstance(section_data, dict)
                        and "execution_steps" in section_data
                    ):
                        for consultant_key, steps in section_data[
                            "execution_steps"
                        ].items():
                            for step in steps:
                                if "llm_interaction" in step:
                                    step["llm_interaction"]["prompt"] = {
                                        "model_used": step["llm_interaction"]["prompt"][
                                            "model_used"
                                        ],
                                        "prompt_removed": "Removed for privacy compliance",
                                    }

        if not request.include_technical_details:
            # Remove technical details
            export_data.pop("technical_execution_details", None)

        # Return as JSON response with appropriate headers
        response = JSONResponse(
            content=export_data,
            headers={
                "Content-Disposition": f"attachment; filename=audit_trail_{engagement_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "Content-Type": "application/json",
            },
        )

        return response

    except ValueError:
        raise HTTPException(
            status_code=400, detail=f"Invalid engagement ID: {engagement_id}"
        )
    except Exception as e:
        logger.error(f"Audit trail export failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Audit trail export failed: {str(e)}"
        )


@router.get("/transparency-layers")
async def get_transparency_layer_definitions():
    """Get definitions of all transparency layers for frontend configuration"""
    return {
        "transparency_layers": {
            "executive_summary": {
                "name": "Executive Summary",
                "description": "Strategic conclusions and key insights",
                "cognitive_load": "low",
                "target_audience": ["executive", "strategic"],
                "estimated_reading_time": 2,
                "includes": [
                    "final_recommendations",
                    "key_metrics",
                    "high_level_reasoning",
                ],
            },
            "reasoning_overview": {
                "name": "Reasoning Overview",
                "description": "Methodology visibility and decision rationale",
                "cognitive_load": "medium",
                "target_audience": ["strategic", "analytical"],
                "estimated_reading_time": 5,
                "includes": [
                    "consultant_selection_rationale",
                    "mental_models_used",
                    "confidence_scores",
                ],
            },
            "detailed_audit_trail": {
                "name": "Detailed Audit Trail",
                "description": "Complete step-by-step reasoning process",
                "cognitive_load": "high",
                "target_audience": ["analytical", "technical"],
                "estimated_reading_time": 15,
                "includes": [
                    "every_decision_point",
                    "evidence_sources",
                    "assumptions_made",
                ],
            },
            "technical_execution": {
                "name": "Technical Execution",
                "description": "LLM prompts, responses, and system implementation details",
                "cognitive_load": "high",
                "target_audience": ["technical"],
                "estimated_reading_time": 20,
                "includes": [
                    "llm_prompts",
                    "token_consumption",
                    "performance_metrics",
                    "error_logs",
                ],
            },
        },
        "user_expertise_levels": {
            "executive": {
                "name": "Executive",
                "description": "C-suite and board members",
                "default_layers": ["executive_summary"],
                "max_cognitive_load": "low",
            },
            "strategic": {
                "name": "Strategic",
                "description": "Strategy professionals and consultants",
                "default_layers": ["executive_summary", "reasoning_overview"],
                "max_cognitive_load": "medium",
            },
            "analytical": {
                "name": "Analytical",
                "description": "Analysts and researchers",
                "default_layers": [
                    "executive_summary",
                    "reasoning_overview",
                    "detailed_audit_trail",
                ],
                "max_cognitive_load": "high",
            },
            "technical": {
                "name": "Technical",
                "description": "Data scientists and engineers",
                "default_layers": [
                    "executive_summary",
                    "reasoning_overview",
                    "detailed_audit_trail",
                    "technical_execution",
                ],
                "max_cognitive_load": "high",
            },
        },
    }


@router.get("/health")
async def glass_box_health_check():
    """Health check endpoint for glass-box system"""
    try:
        # Basic health checks
        orchestrator_healthy = glass_box_orchestrator is not None

        return {
            "status": "healthy" if orchestrator_healthy else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "glass_box_orchestrator": (
                    "healthy" if orchestrator_healthy else "failed"
                ),
                "audit_trail_capture": "healthy",
                "transparency_layers": "healthy",
            },
            "capabilities": {
                "comprehensive_audit_capture": True,
                "decision_point_tracking": True,
                "llm_prompt_response_logging": True,
                "progressive_transparency": True,
                "export_functionality": True,
            },
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
        }


@router.get("/demo-audit-trail")
async def get_demo_audit_trail():
    """Get demo audit trail data for frontend development and testing"""

    # Create comprehensive demo data that matches our data contracts
    demo_audit_trail = {
        "engagement_id": "demo-engagement-12345",
        "user_id": "demo-user-67890",
        "session_id": "demo-session-abcde",
        "timestamp_start": "2024-01-15T10:30:00Z",
        "timestamp_end": "2024-01-15T10:35:30Z",
        "raw_query": "How can we develop a comprehensive 5-year strategic plan to enter the Asian market while maintaining our competitive advantage in North America?",
        "classification_decision": {
            "raw_query": "How can we develop a comprehensive 5-year strategic plan to enter the Asian market while maintaining our competitive advantage in North America?",
            "detected_intent": "strategic_planning",
            "intent_confidence": 0.92,
            "complexity_level": "HIGHLY_COMPLEX",
            "complexity_score": 4,
            "urgency_level": "medium",
            "scope_assessment": "enterprise",
            "keyword_extraction_results": [
                "strategic",
                "plan",
                "Asian",
                "market",
                "competitive",
                "advantage",
                "North America",
            ],
            "pattern_matching_scores": {
                "strategic_analysis": 0.91,
                "market_expansion": 0.87,
                "competitive_positioning": 0.79,
            },
            "historical_query_similarities": [],
            "routing_suggestions": [
                "strategic_analysis_cluster",
                "market_expansion_cluster",
            ],
            "classification_timestamp": "2024-01-15T10:30:01Z",
            "processing_time_seconds": 0.234,
            "classifier_model_version": "enhanced_v1.2",
        },
        "selection_decision": {
            "nway_cluster_scores": {
                "strategic_analysis_cluster": 0.91,
                "operational_optimization_cluster": 0.34,
                "innovation_discovery_cluster": 0.67,
                "market_expansion_cluster": 0.89,
            },
            "selected_nway_cluster_id": "strategic_analysis_cluster",
            "cluster_selection_reasoning": "High strategic complexity with market expansion elements",
            "consultant_prediction_scores": {
                "strategic_analyst": 0.89,
                "market_research_specialist": 0.92,
                "implementation_strategist": 0.76,
                "competitive_intelligence_analyst": 0.84,
            },
            "selected_consultants": [
                "strategic_analyst",
                "market_research_specialist",
                "implementation_strategist",
            ],
            "prediction_confidence": 0.86,
            "alternative_consultant_combinations": [
                {
                    "consultants": [
                        "strategic_analyst",
                        "competitive_intelligence_analyst",
                        "implementation_strategist",
                    ],
                    "confidence": 0.82,
                }
            ],
            "historical_effectiveness_data": {
                "strategic_analyst": 0.87,
                "market_research_specialist": 0.91,
            },
            "pattern_matching_results": {},
            "learning_model_version": "predictive_v1.1",
            "selection_timestamp": "2024-01-15T10:30:02Z",
            "selection_processing_time": 0.445,
        },
        "execution_steps": {
            "strategic_analyst": [
                {
                    "step_id": "strategic_step_1",
                    "step_index": 0,
                    "consultant_role": "strategic_analyst",
                    "step_description": "Analyze strategic market positioning for Asian expansion",
                    "input_context": "Original query and classification results",
                    "context_length_tokens": 350,
                    "llm_prompt": {
                        "system_prompt": "You are a strategic analyst specializing in international market expansion...",
                        "user_prompt": "Analyze the strategic implications of expanding into Asian markets...",
                        "model_used": "claude-3-sonnet",
                        "temperature": 0.7,
                        "max_tokens": 2000,
                        "prompt_length_tokens": 450,
                        "estimated_cost_usd": 0.002,
                    },
                    "llm_response": {
                        "raw_response": "Based on comprehensive market analysis, entering Asian markets presents significant opportunities but requires careful strategic positioning. Key considerations include: 1) Market segmentation analysis showing highest potential in Southeast Asian markets with 23% growth rates...",
                        "completion_tokens": 1247,
                        "prompt_tokens": 450,
                        "total_tokens": 1697,
                        "actual_cost_usd": 0.0034,
                        "processing_time_seconds": 4.2,
                        "finish_reason": "completed",
                        "model_version": "claude-3-sonnet-20240229",
                        "response_timestamp": "2024-01-15T10:30:08Z",
                        "confidence_indicators": {
                            "factual_accuracy": 0.89,
                            "logical_consistency": 0.92,
                        },
                    },
                    "extracted_reasoning": "Strategic analysis reveals three critical success factors for Asian market entry: market timing alignment with regional economic growth, localization of value propositions for diverse cultural contexts, and phased expansion approach to manage risk while capturing opportunities.",
                    "extracted_context_for_next_step": "Strategic framework established - market research specialist should focus on specific market dynamics and competitive landscape analysis",
                    "mental_models_applied": [
                        "MECE Framework",
                        "Five Forces Analysis",
                        "Market Entry Strategy",
                    ],
                    "assumptions_made": [
                        "Regional economic growth rates will continue at current pace",
                        "Regulatory environment remains stable across target markets",
                        "Company has sufficient capital for 5-year investment horizon",
                    ],
                    "evidence_sources": [
                        "McKinsey Asia Growth Report 2024",
                        "Regional Economic Indicators",
                        "Competitor Financial Filings",
                    ],
                    "confidence_score": 0.87,
                    "logical_consistency_score": 0.92,
                    "factual_accuracy_score": 0.89,
                    "validation_flags": [],
                    "status": "completed",
                    "execution_start_time": "2024-01-15T10:30:03Z",
                    "execution_end_time": "2024-01-15T10:30:08Z",
                    "error_message": null,
                }
            ],
            "market_research_specialist": [
                {
                    "step_id": "market_research_step_1",
                    "step_index": 1,
                    "consultant_role": "market_research_specialist",
                    "step_description": "Conduct detailed market research and competitive landscape analysis",
                    "input_context": "Strategic framework from strategic analyst + original query context",
                    "context_length_tokens": 680,
                    "llm_prompt": {
                        "system_prompt": "You are a market research specialist with expertise in Asian markets...",
                        "user_prompt": "Conduct comprehensive market research for Asian expansion based on strategic framework...",
                        "model_used": "claude-3-sonnet",
                        "temperature": 0.5,
                        "max_tokens": 2000,
                        "prompt_length_tokens": 780,
                        "estimated_cost_usd": 0.0035,
                    },
                    "llm_response": {
                        "raw_response": "Market research reveals compelling opportunities across three primary segments: Singapore and Hong Kong for premium positioning (market size $2.3B, growth 18%), Southeast Asian emerging markets for volume expansion (Vietnam, Thailand, Malaysia - combined TAM $8.7B), and China tier-2 cities for long-term strategic positioning...",
                        "completion_tokens": 1456,
                        "prompt_tokens": 780,
                        "total_tokens": 2236,
                        "actual_cost_usd": 0.0045,
                        "processing_time_seconds": 5.8,
                        "finish_reason": "completed",
                        "model_version": "claude-3-sonnet-20240229",
                        "response_timestamp": "2024-01-15T10:30:15Z",
                    },
                    "extracted_reasoning": "Market research identifies three-tier expansion approach: premium markets (Singapore/Hong Kong) for immediate revenue and brand establishment, emerging markets (Vietnam/Thailand/Malaysia) for volume growth, and strategic markets (China tier-2 cities) for long-term positioning. Competitive analysis shows fragmented landscape with opportunities for differentiated positioning.",
                    "extracted_context_for_next_step": "Market segments and competitive landscape defined - implementation strategist should develop execution roadmap",
                    "mental_models_applied": [
                        "TAM-SAM-SOM Analysis",
                        "Competitive Positioning Matrix",
                        "Market Segmentation Framework",
                    ],
                    "assumptions_made": [
                        "Currency fluctuations remain within historical ranges",
                        "Political stability maintained across target regions",
                        "Digital infrastructure continues rapid development",
                    ],
                    "evidence_sources": [
                        "Euromonitor Market Reports",
                        "Competitive Intelligence Database",
                        "Government Trade Statistics",
                    ],
                    "confidence_score": 0.91,
                    "logical_consistency_score": 0.88,
                    "factual_accuracy_score": 0.94,
                    "validation_flags": ["currency_risk_noted"],
                    "status": "completed",
                    "execution_start_time": "2024-01-15T10:30:08Z",
                    "execution_end_time": "2024-01-15T10:30:15Z",
                }
            ],
            "implementation_strategist": [
                {
                    "step_id": "implementation_step_1",
                    "step_index": 2,
                    "consultant_role": "implementation_strategist",
                    "step_description": "Develop detailed implementation roadmap and execution strategy",
                    "input_context": "Strategic framework + market research findings + implementation requirements",
                    "context_length_tokens": 1020,
                    "llm_prompt": {
                        "system_prompt": "You are an implementation strategist specializing in international expansion execution...",
                        "user_prompt": "Develop comprehensive 5-year implementation roadmap based on strategic framework and market research...",
                        "model_used": "claude-3-sonnet",
                        "temperature": 0.3,
                        "max_tokens": 2500,
                        "prompt_length_tokens": 1150,
                        "estimated_cost_usd": 0.0055,
                    },
                    "llm_response": {
                        "raw_response": "Implementation roadmap structured in three phases: Phase 1 (Years 1-2) - Premium market entry via Singapore/Hong Kong with local partnerships and regulatory compliance framework, estimated investment $12M, projected revenue $25M by Year 2. Phase 2 (Years 2-4) - Emerging market expansion with localized product variations and distribution partnerships, additional investment $28M, revenue target $75M by Year 4. Phase 3 (Years 4-5) - China strategic positioning with JV partnerships and technology transfer agreements...",
                        "completion_tokens": 1789,
                        "prompt_tokens": 1150,
                        "total_tokens": 2939,
                        "actual_cost_usd": 0.0071,
                        "processing_time_seconds": 7.3,
                        "finish_reason": "completed",
                        "model_version": "claude-3-sonnet-20240229",
                        "response_timestamp": "2024-01-15T10:30:25Z",
                    },
                    "extracted_reasoning": "Five-year implementation roadmap with phased approach minimizes risk while maximizing market capture opportunities. Phase 1 focuses on premium markets for immediate cash flow and brand establishment, Phase 2 scales to emerging markets for volume growth, Phase 3 establishes strategic foothold in China for long-term competitive advantage. Total investment requirement: $52M over 5 years, projected cumulative revenue: $180M by Year 5.",
                    "extracted_context_for_next_step": "Complete implementation roadmap defined with timelines, investments, and success metrics",
                    "mental_models_applied": [
                        "Phased Implementation Strategy",
                        "Risk Management Framework",
                        "Resource Allocation Optimization",
                    ],
                    "assumptions_made": [
                        "Partnership negotiations conclude within projected timelines",
                        "Regulatory approvals proceed as scheduled",
                        "Market conditions remain favorable for international expansion",
                        "Internal capabilities can be scaled to support multi-region operations",
                    ],
                    "evidence_sources": [
                        "Implementation Best Practices Database",
                        "Partnership Case Studies",
                        "Financial Modeling Templates",
                    ],
                    "confidence_score": 0.84,
                    "logical_consistency_score": 0.90,
                    "factual_accuracy_score": 0.82,
                    "validation_flags": [
                        "partnership_risk_highlighted",
                        "regulatory_complexity_noted",
                    ],
                    "status": "completed",
                    "execution_start_time": "2024-01-15T10:30:15Z",
                    "execution_end_time": "2024-01-15T10:30:25Z",
                }
            ],
        },
        "devils_advocate_results": {},
        "senior_advisor_result": null,
        "current_phase": "completed",
        "final_status": "completed",
        "total_cost_usd": 0.015,
        "total_tokens_consumed": 6872,
        "total_processing_time_seconds": 330.0,
        "user_interactions": [
            {
                "timestamp": "2024-01-15T10:30:00Z",
                "interaction_type": "query_submitted",
                "metadata": {"query_length": 123},
            },
            {
                "timestamp": "2024-01-15T10:35:30Z",
                "interaction_type": "results_viewed",
                "metadata": {"layer": "executive_summary"},
            },
        ],
        "transparency_layer_access": {
            "executive_summary": 1,
            "reasoning_overview": 0,
            "detailed_audit_trail": 0,
        },
        "user_feedback": null,
        "error_events": [],
        "performance_warnings": [],
        "system_health_snapshots": [],
    }

    return {
        "status": "success",
        "demo_data": demo_audit_trail,
        "note": "This is demo data for frontend development and testing purposes",
    }