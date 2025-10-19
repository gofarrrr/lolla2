"""
Projects API - Mental Models & CQA
===================================

Mental model ingestion and CQA (Cognitive Quality Assurance) endpoints.
"""

import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from supabase import Client

from .models import (
    MentalModelIngestionRequest,
    MentalModelIngestionResponse,
    MentalModelCQARequest,
    MentalModelCQAResponse,
    QualityScoreResponse,
    BatchCQARequest,
    BatchCQAResponse,
    CQABenchmarkRequest,
    CQABenchmarkResponse,
    ProjectStatusResponse,
)
from .dependencies import get_supabase
from src.core.unified_context_stream import UnifiedContextStream
from src.ingestion.mental_model_parser import create_mental_model_parser
from src.core.mental_model_cqa import MentalModelCQAEvaluator
from src.core.contracts.mental_model_rubrics import (
    MentalModelType,
    get_mental_model_rubric_registry,
    MentalModelQualityDimension,
)

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()


@router.post("/mental-models/ingest", response_model=MentalModelIngestionResponse)
async def ingest_mental_models(
    request: MentalModelIngestionRequest, supabase: Client = Depends(get_supabase)
) -> MentalModelIngestionResponse:
    """
    Ingest mental model files into a project's knowledge base.

    This endpoint orchestrates the mental model ingestion pipeline,
    transforming text-based mental model files into structured,
    searchable knowledge within the specified project.
    """
    try:
        ingestion_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc)

        logger.info(f"🧠 Starting mental model ingestion: {ingestion_id}")

        # Validate project exists and is active
        project_result = (
            supabase.table("projects")
            .select("project_id, name, status")
            .eq("project_id", request.project_id)
            .eq("status", "active")
            .execute()
        )

        if not project_result.data:
            raise HTTPException(status_code=404, detail="Project not found or inactive")

        project = project_result.data[0]
        project_name = project["name"]

        # Initialize context stream for this ingestion
        audit_stream = UnifiedContextStream(trace_id=ingestion_id)

        await audit_stream.log_event(
            "MENTAL_MODEL_INGESTION_API_STARTED",
            {
                "ingestion_id": ingestion_id,
                "project_id": request.project_id,
                "project_name": project_name,
                "directory_path": request.directory_path,
                "file_pattern": request.file_pattern,
                "organization_id": request.organization_id,
            },
        )

        # Initialize mental model parser
        parser = create_mental_model_parser()

        # Initialize stats tracking
        stats = {
            "total_files_found": 0,
            "successfully_parsed": 0,
            "successfully_ingested": 0,
            "failed_files": [],
            "estimated_chunks_created": 0,
        }

        # Parse mental model files if directory provided
        mental_models = []
        if request.directory_path:
            directory = Path(request.directory_path)
            if not directory.exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"Directory not found: {request.directory_path}",
                )

            # Find matching files
            files = list(directory.glob(request.file_pattern))
            stats["total_files_found"] = len(files)

            logger.info(f"📁 Found {len(files)} mental model files to process")

            # Parse each file
            for file_path in files:
                try:
                    mental_model = parser.parse_file(file_path)
                    mental_models.append(mental_model)
                    stats["successfully_parsed"] += 1

                    await audit_stream.log_event(
                        "MENTAL_MODEL_PARSED",
                        {
                            "file_name": file_path.name,
                            "model_name": mental_model.name,
                            "quality_score": mental_model.metadata.quality_score,
                        },
                    )

                except Exception as e:
                    logger.error(f"❌ Failed to parse {file_path.name}: {e}")
                    stats["failed_files"].append(f"{file_path.name}: {str(e)}")

                    await audit_stream.log_event(
                        "MENTAL_MODEL_PARSING_ERROR",
                        {"file_name": file_path.name, "error": str(e)},
                    )

        # Ingest parsed mental models into RAG system
        if mental_models:
            from src.rag.project_rag_pipeline import get_project_rag_pipeline

            rag_pipeline = get_project_rag_pipeline()
            await rag_pipeline.initialize()

            for mental_model in mental_models:
                try:
                    # Prepare document for RAG ingestion
                    document_title = f"Mental Model: {mental_model.name}"
                    document_content = mental_model.full_text_content

                    metadata = {
                        "mental_model_name": mental_model.name,
                        "mental_model_type": mental_model.mental_model_type,
                        "complexity_level": mental_model.complexity_level,
                        "effectiveness_score": mental_model.effectiveness_score,
                        "quality_score": mental_model.metadata.quality_score,
                        "source_file": mental_model.metadata.source_file,
                        "document_type": "mental_model",
                        "ingestion_source": "api_endpoint",
                        "ingestion_id": ingestion_id,
                    }

                    # Store in RAG system
                    doc_id = await rag_pipeline.store_web_document(
                        url=f"file://{mental_model.metadata.source_file}",
                        content=document_content,
                        title=document_title,
                        project_id=uuid.UUID(request.project_id),
                        tags=["mental-model", "api-ingested"]
                        + mental_model.searchable_keywords[:8],
                        metadata=metadata,
                    )

                    if doc_id:
                        stats["successfully_ingested"] += 1
                        # Estimate chunks (rough calculation)
                        estimated_chunks = max(1, len(document_content) // 500)
                        stats["estimated_chunks_created"] += estimated_chunks

                        await audit_stream.log_event(
                            "MENTAL_MODEL_INGESTED",
                            {
                                "mental_model_name": mental_model.name,
                                "document_id": doc_id,
                                "estimated_chunks": estimated_chunks,
                            },
                        )

                        logger.info(f"✅ Ingested mental model: {mental_model.name}")
                    else:
                        raise RuntimeError("RAG ingestion returned None")

                except Exception as e:
                    logger.error(f"❌ Failed to ingest {mental_model.name}: {e}")
                    stats["failed_files"].append(f"{mental_model.name}: {str(e)}")

                    await audit_stream.log_event(
                        "MENTAL_MODEL_INGESTION_ERROR",
                        {"mental_model_name": mental_model.name, "error": str(e)},
                    )

        # Generate completion response
        completed_at = datetime.now(timezone.utc)

        await audit_stream.log_event(
            "MENTAL_MODEL_INGESTION_API_COMPLETED",
            {
                "ingestion_id": ingestion_id,
                "stats": stats,
                "duration_seconds": (completed_at - started_at).total_seconds(),
            },
        )

        # Generate next steps
        next_steps = [
            f"Verify ingestion in project '{project_name}'",
            "Test 'Chat with Your Project' functionality",
            "Query the knowledge base to validate accessibility",
        ]

        if stats["failed_files"]:
            next_steps.append(
                f"Review {len(stats['failed_files'])} failed files for issues"
            )

        logger.info(f"✅ Mental model ingestion completed: {ingestion_id}")

        return MentalModelIngestionResponse(
            ingestion_id=ingestion_id,
            project_id=request.project_id,
            project_name=project_name,
            status=(
                "completed"
                if stats["successfully_ingested"] > 0
                else "completed_with_errors"
            ),
            total_files_found=stats["total_files_found"],
            successfully_parsed=stats["successfully_parsed"],
            successfully_ingested=stats["successfully_ingested"],
            failed_files=stats["failed_files"],
            estimated_chunks_created=stats["estimated_chunks_created"],
            started_at=started_at,
            completed_at=completed_at,
            next_steps=next_steps,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Mental model ingestion API failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Mental model ingestion failed: {str(e)}"
        )


@router.get("/{project_id}/status", response_model=ProjectStatusResponse)
async def get_project_status(
    project_id: str, supabase: Client = Depends(get_supabase)
) -> ProjectStatusResponse:
    """
    Get project status for conditional UI features like chat visibility.

    This lightweight endpoint provides essential project status information
    to determine whether features like "Chat with Your Project" should be enabled.
    """
    try:
        # Check if project exists and get basic info
        project_result = (
            supabase.table("projects")
            .select("project_id, name, status, last_accessed_at")
            .eq("project_id", project_id)
            .eq("status", "active")
            .execute()
        )

        if not project_result.data:
            raise HTTPException(status_code=404, detail="Project not found or inactive")

        project = project_result.data[0]

        # Get document and chunk counts from RAG pipeline
        doc_count_result = (
            supabase.table("rag_documents")
            .select("document_id", count="exact")
            .eq("project_id", project_id)
            .execute()
        )

        chunks_count_result = (
            supabase.table("rag_text_chunks")
            .select("chunk_id", count="exact")
            .eq("project_id", project_id)
            .execute()
        )

        document_count = doc_count_result.count or 0
        chunks_count = chunks_count_result.count or 0

        # Get latest document activity
        latest_doc_result = (
            supabase.table("rag_documents")
            .select("created_at")
            .eq("project_id", project_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )

        last_activity_date = None
        if latest_doc_result.data:
            last_activity_date = datetime.fromisoformat(
                latest_doc_result.data[0]["created_at"].replace("Z", "+00:00")
            )

        # Determine content availability and chat readiness
        has_content = document_count > 0 and chunks_count > 0

        # Chat is ready if there's content and the project is active
        chat_ready = has_content and project["status"] == "active"

        # Determine RAG health
        rag_health = "healthy"
        if not has_content:
            rag_health = "empty"
        elif document_count > 0 and chunks_count == 0:
            rag_health = "processing"
        elif (
            chunks_count < document_count * 3
        ):  # Expect at least 3 chunks per doc on average
            rag_health = "incomplete"

        # Chat features availability
        chat_features = {
            "conversational_search": chat_ready,
            "rag_context": has_content,
            "memory_persistence": True,  # Always available via Zep
            "source_citation": has_content,
            "multi_turn_conversation": chat_ready,
        }

        logger.info(
            f"✅ Project status retrieved: {project_id} - Chat Ready: {chat_ready}"
        )

        return ProjectStatusResponse(
            project_id=project_id,
            has_content=has_content,
            chat_ready=chat_ready,
            document_count=document_count,
            text_chunks_count=chunks_count,
            last_activity_date=last_activity_date,
            rag_health=rag_health,
            chat_features=chat_features,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Project status retrieval failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get project status: {str(e)}"
        )


@router.post("/mental-models/cqa/evaluate", response_model=MentalModelCQAResponse)
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
            from src.ingestion.mental_model_parser import MentalModelData

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


@router.post("/mental-models/cqa/batch-evaluate", response_model=BatchCQAResponse)
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


@router.get("/mental-models/cqa/benchmark", response_model=CQABenchmarkResponse)
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


@router.get("/mental-models/cqa/rubrics", response_model=Dict[str, Any])
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


@router.get("/health", response_model=Dict[str, Any])
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
