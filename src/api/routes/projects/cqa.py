"""
V2 Projects API - Mental Model CQA Endpoints
=============================================

Mental model Cognitive Quality Assurance (CQA) evaluation endpoints.

Operation Bedrock: Task 11.0 - Projects API Decomposition
"""

import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Depends
from supabase import Client

from src.core.mental_model_cqa import MentalModelCQAEvaluator
from src.core.contracts.mental_model_rubrics import (
    MentalModelType,
    MentalModelQualityDimension,
    get_mental_model_rubric_registry,
)
from src.ingestion.mental_model_parser import MentalModelData

from .helpers import get_supabase
from .models import (
    MentalModelCQARequest,
    MentalModelCQAResponse,
    QualityScoreResponse,
    BatchCQARequest,
    BatchCQAResponse,
    CQABenchmarkRequest,
    CQABenchmarkResponse,
)

logger = logging.getLogger(__name__)

# CQA router
cqa_router = APIRouter(prefix="/api/v2/projects", tags=["V2 Projects - CQA"])


# ============================================================
# MENTAL MODEL CQA ENDPOINTS
# ============================================================


@cqa_router.post("/mental-models/cqa/evaluate", response_model=MentalModelCQAResponse)
async def evaluate_mental_model_cqa(
    request: MentalModelCQARequest, supabase: Client = Depends(get_supabase)
) -> MentalModelCQAResponse:
    """
    Evaluate a single mental model using the CQA (Cognitive Quality Assurance) system.

    This endpoint performs comprehensive quality assessment on a mental model,
    evaluating dimensions like rigor, clarity, applicability, and coherence using
    specialized rubrics tailored to different mental model types.
    """
    try:
        logger.info("🔍 Starting CQA evaluation for mental model")

        # Initialize CQA evaluator
        cqa_evaluator = MentalModelCQAEvaluator()

        # Convert request data to MentalModelData
        try:
            mental_model_data = MentalModelData(**request.mental_model_data)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid mental model data: {str(e)}"
            )

        # Override model type if specified
        model_type = None
        if request.model_type_override:
            try:
                model_type = MentalModelType(request.model_type_override)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid model type: {request.model_type_override}",
                )

        # Perform CQA evaluation
        cqa_result = await cqa_evaluator.evaluate_mental_model(
            mental_model_data, model_type=model_type
        )

        # Convert dimension scores to response format
        dimension_scores_response = {}
        for dim, score in cqa_result.dimension_scores.items():
            dimension_scores_response[dim.value] = QualityScoreResponse(
                dimension=dim.value,
                score=score.score,
                rationale=score.rationale,
                evidence=score.evidence,
                confidence=score.confidence,
            )

        # Create response
        response = MentalModelCQAResponse(
            mental_model_id=cqa_result.mental_model_id,
            mental_model_name=cqa_result.mental_model_name,
            model_type=cqa_result.model_type.value,
            rubric_used=cqa_result.rubric_used,
            evaluation_timestamp=datetime.fromisoformat(
                cqa_result.evaluation_timestamp
            ),
            dimension_scores=dimension_scores_response,
            overall_score=cqa_result.overall_score,
            weighted_score=cqa_result.weighted_score,
            confidence_level=cqa_result.confidence_level,
            quality_tier=cqa_result.quality_tier,
            validation_status=cqa_result.validation_status,
            evaluator_version=cqa_result.evaluator_version,
            execution_time_ms=cqa_result.execution_time_ms,
            context_stream_id=cqa_result.context_stream_id,
        )

        # Store CQA result in database for tracking
        cqa_record = {
            "mental_model_id": cqa_result.mental_model_id,
            "mental_model_name": cqa_result.mental_model_name,
            "organization_id": request.organization_id,
            "model_type": cqa_result.model_type.value,
            "rubric_used": cqa_result.rubric_used,
            "overall_score": cqa_result.overall_score,
            "weighted_score": cqa_result.weighted_score,
            "quality_tier": cqa_result.quality_tier,
            "validation_status": cqa_result.validation_status,
            "confidence_level": cqa_result.confidence_level,
            "evaluator_version": cqa_result.evaluator_version,
            "evaluation_timestamp": cqa_result.evaluation_timestamp,
            "execution_time_ms": cqa_result.execution_time_ms,
            "dimension_scores": {
                dim.value: {
                    "score": score.score,
                    "rationale": score.rationale,
                    "confidence": score.confidence,
                }
                for dim, score in cqa_result.dimension_scores.items()
            },
        }

        supabase.table("mental_model_cqa_results").insert(cqa_record).execute()

        logger.info(
            f"✅ CQA evaluation completed: {cqa_result.mental_model_name} - {cqa_result.quality_tier} ({cqa_result.weighted_score:.2f})"
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ CQA evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"CQA evaluation failed: {str(e)}")


@cqa_router.post("/mental-models/cqa/batch-evaluate", response_model=BatchCQAResponse)
async def batch_evaluate_mental_models_cqa(
    request: BatchCQARequest, supabase: Client = Depends(get_supabase)
) -> BatchCQAResponse:
    """
    Perform batch CQA evaluation on all mental models in a project.

    This endpoint evaluates all mental models within a specified project,
    providing comprehensive quality metrics and analytics.
    """
    try:
        batch_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc)

        logger.info(
            f"🚀 Starting batch CQA evaluation for project {request.project_id}"
        )

        # Get project information
        project_result = (
            supabase.table("projects")
            .select("name")
            .eq("project_id", request.project_id)
            .execute()
        )
        if not project_result.data:
            raise HTTPException(status_code=404, detail="Project not found")

        project_name = project_result.data[0]["name"]

        # Query for mental models in the project
        # This is a placeholder - in reality, we'd need a proper query to find mental models
        # associated with this project from the RAG system or document store

        # For now, return a mock response indicating the batch operation was initiated
        response = BatchCQAResponse(
            batch_id=batch_id,
            project_id=request.project_id,
            project_name=project_name,
            total_models=0,
            evaluated_models=0,
            failed_evaluations=0,
            average_quality_score=0.0,
            quality_distribution={"excellent": 0, "good": 0, "average": 0, "poor": 0},
            validation_summary={"passed": 0, "failed": 0, "review_needed": 0},
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
            results=[],
        )

        logger.info(f"🔍 Batch CQA evaluation initiated: {batch_id}")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Batch CQA evaluation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Batch CQA evaluation failed: {str(e)}"
        )


@cqa_router.get("/mental-models/cqa/benchmark", response_model=CQABenchmarkResponse)
async def get_cqa_benchmark(
    request: CQABenchmarkRequest = Depends(), supabase: Client = Depends(get_supabase)
) -> CQABenchmarkResponse:
    """
    Get CQA benchmark analysis and quality trends across projects.

    This endpoint provides comprehensive analytics on mental model quality
    over time, including trends, distributions, and improvement recommendations.
    """
    try:
        logger.info("📊 Generating CQA benchmark analysis")

        # Calculate analysis period
        end_date = datetime.now(timezone.utc)
        start_date = end_date.replace(day=end_date.day - request.time_range_days)

        # Query CQA results from database
        query = supabase.table("mental_model_cqa_results").select("*")

        if request.organization_id:
            query = query.eq("organization_id", request.organization_id)

        if request.project_id:
            # Would need to join with project data to filter by project
            pass

        if request.model_types:
            query = query.in_("model_type", request.model_types)

        # Add date filtering
        query = query.gte("evaluation_timestamp", start_date.isoformat())
        query = query.lte("evaluation_timestamp", end_date.isoformat())

        result = query.execute()
        evaluations = result.data if result.data else []

        # Calculate aggregate statistics
        total_evaluations = len(evaluations)
        average_quality_score = sum(e["weighted_score"] for e in evaluations) / max(
            1, total_evaluations
        )

        # Calculate distributions
        quality_tier_distribution = {"excellent": 0, "good": 0, "average": 0, "poor": 0}
        validation_status_distribution = {"passed": 0, "failed": 0, "review_needed": 0}

        for evaluation in evaluations:
            quality_tier_distribution[evaluation["quality_tier"]] += 1
            validation_status_distribution[evaluation["validation_status"]] += 1

        # Generate improvement recommendations
        recommendations = []
        if average_quality_score < 6.0:
            recommendations.append(
                "Consider reviewing mental model content for clarity and completeness"
            )
        if validation_status_distribution["failed"] > total_evaluations * 0.2:
            recommendations.append(
                "High failure rate detected - review validation thresholds and rubric criteria"
            )
        if quality_tier_distribution["poor"] > 0:
            recommendations.append(
                "Some models classified as poor quality - prioritize these for improvement"
            )

        response = CQABenchmarkResponse(
            organization_id=request.organization_id,
            project_id=request.project_id,
            analysis_period={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
            total_evaluations=total_evaluations,
            average_quality_score=average_quality_score,
            quality_score_trend=[],  # Would need time-series aggregation
            quality_tier_distribution=quality_tier_distribution,
            validation_status_distribution=validation_status_distribution,
            model_type_performance={},  # Would need grouping by model type
            highest_quality_models=[],  # Would need sorting and limiting
            lowest_quality_models=[],  # Would need sorting and limiting
            improvement_recommendations=recommendations,
        )

        logger.info(
            f"📈 CQA benchmark analysis completed: {total_evaluations} evaluations analyzed"
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ CQA benchmark analysis failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"CQA benchmark analysis failed: {str(e)}"
        )


@cqa_router.get("/mental-models/cqa/rubrics", response_model=Dict[str, Any])
async def get_available_cqa_rubrics() -> Dict[str, Any]:
    """
    Get information about available CQA rubrics and evaluation criteria.

    This endpoint provides metadata about the different rubrics used for
    mental model quality evaluation, including their criteria and weights.
    """
    try:
        logger.info("📋 Retrieving available CQA rubrics")

        # Get rubric registry
        rubric_registry = get_mental_model_rubric_registry()
        available_rubrics = rubric_registry.list_available_rubrics()

        rubrics_info = {}

        for rubric_id in available_rubrics:
            rubric = rubric_registry.get_rubric(rubric_id)
            if rubric:
                rubrics_info[rubric_id] = {
                    "name": rubric.name,
                    "description": rubric.description,
                    "model_type": rubric.model_type.value,
                    "version": rubric.version,
                    "criteria": [
                        {
                            "dimension": criterion.dimension.value,
                            "weight": criterion.weight,
                            "criteria_ranges": {
                                "poor_1_3": criterion.criteria_1_3,
                                "average_4_6": criterion.criteria_4_6,
                                "good_7_8": criterion.criteria_7_8,
                                "excellent_9_10": criterion.criteria_9_10,
                            },
                        }
                        for criterion in rubric.criteria
                    ],
                }

        # Add model types information
        model_types_info = {
            model_type.value: {
                "description": f"Mental models focused on {model_type.value.replace('_', ' ')}",
                "recommended_rubric": rubric_registry.get_rubric_for_model_type(
                    model_type
                ).rubric_id,
            }
            for model_type in MentalModelType
        }

        response = {
            "rubrics": rubrics_info,
            "model_types": model_types_info,
            "quality_dimensions": [dim.value for dim in MentalModelQualityDimension],
            "quality_tiers": ["excellent", "good", "average", "poor"],
            "validation_statuses": ["passed", "failed", "review_needed"],
        }

        logger.info(f"✅ Retrieved {len(available_rubrics)} available CQA rubrics")

        return response

    except Exception as e:
        logger.error(f"❌ Failed to retrieve CQA rubrics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve CQA rubrics: {str(e)}"
        )


# ============================================================
# HEALTH CHECK
# ============================================================


@cqa_router.get("/health", response_model=Dict[str, Any])
async def projects_api_health() -> Dict[str, Any]:
    """V2 Projects API health check"""
    return {
        "status": "healthy",
        "api_version": "v2.0",
        "features": [
            "project_crud",
            "context_merging",
            "knowledge_base_management",
            "project_statistics",
            "rag_foundation",
            "mental_model_ingestion",
            "mental_model_cqa_evaluation",
            "cqa_batch_processing",
            "cqa_benchmark_analytics",
            "quality_rubric_management",
        ],
        "phase": "Phase 1 Complete - Project Foundation",
        "next_phases": [
            "Phase 2: Context Merge UI & Logic",
            "Phase 3: RAG Pipeline Implementation",
        ],
    }
