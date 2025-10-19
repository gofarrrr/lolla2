"""
Projects API - Knowledge Base & Analysis
=========================================

Knowledge base management and analysis creation endpoints with context merging.
"""

import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Body, Form, File, UploadFile, Depends
from supabase import Client

from .models import (
    ContextMergePreview,
    ProjectKnowledgeSearchRequest,
)
from .dependencies import get_supabase
from src.core.unified_context_stream import UnifiedContextStream
from src.services.document_ingestion_service import (
    get_document_ingestion_service,
    DocumentProcessingError,
)

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()


@router.post("/{project_id}/context-merge-preview", response_model=ContextMergePreview)
async def preview_context_merge(
    project_id: str,
    query: str = Body(
        ..., embed=True, description="Analysis query for context matching"
    ),
    supabase: Client = Depends(get_supabase),
) -> ContextMergePreview:
    """
    Preview what context would be merged for a new analysis.
    This helps users understand the value of context merging.
    """
    try:
        # Get project info
        project_result = (
            supabase.table("projects")
            .select("name")
            .eq("project_id", project_id)
            .execute()
        )
        if not project_result.data:
            raise HTTPException(status_code=404, detail="Project not found")

        project_name = project_result.data[0]["name"]

        # Get knowledge base statistics
        kb_stats = (
            supabase.table("v2_project_knowledge_base")
            .select("*")
            .eq("project_id", project_id)
            .execute()
        )

        total_docs = 0
        total_chunks = 0
        if kb_stats.data:
            total_docs = kb_stats.data[0].get("total_documents", 0) or 0
            total_chunks = kb_stats.data[0].get("total_chunks", 0) or 0

        # TODO: Implement semantic search preview when RAG is fully implemented
        # For now, return recent chunks as preview
        preview_chunks = []
        estimated_tokens = 0

        if total_chunks > 0:
            # Get sample chunks (most recent for now)
            chunks_result = (
                supabase.table("rag_text_chunks")
                .select("content, content_type, created_at, document_id")
                .eq("project_id", project_id)
                .order("created_at", desc=True)
                .limit(3)
                .execute()
            )

            if chunks_result.data:
                for chunk in chunks_result.data:
                    preview_chunks.append(
                        {
                            "content": (
                                chunk["content"][:200] + "..."
                                if len(chunk["content"]) > 200
                                else chunk["content"]
                            ),
                            "content_type": chunk["content_type"],
                            "created_at": chunk["created_at"],
                            "document_id": chunk["document_id"],
                        }
                    )
                    # Rough token estimation (4 chars ≈ 1 token)
                    estimated_tokens += len(chunk["content"]) // 4

        # Estimate additional cost (rough calculation)
        estimated_cost = estimated_tokens * 0.0001  # Approximate cost per token

        return ContextMergePreview(
            project_name=project_name,
            total_available_documents=total_docs,
            total_available_chunks=total_chunks,
            preview_chunks=preview_chunks,
            estimated_context_tokens=estimated_tokens,
            estimated_additional_cost=estimated_cost,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Context merge preview failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to preview context merge: {str(e)}"
        )


@router.post("/analyses", response_model=Dict[str, Any])
async def create_analysis(
    project_id: str = Form(...),
    user_query: str = Form(...),
    engagement_type: str = Form(default="deep_dive"),
    merge_project_context: bool = Form(default=False),
    organization_id: str = Form(...),
    case_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    supabase: Client = Depends(get_supabase),
) -> Dict[str, Any]:
    """
    Create a new analysis within a project with optional context merging and file upload.
    This endpoint supports both JSON requests and multipart form data (with files).
    """
    try:
        # Validate project exists and is active
        project_result = (
            supabase.table("projects")
            .select("*")
            .eq("project_id", project_id)
            .eq("status", "active")
            .execute()
        )

        if not project_result.data:
            raise HTTPException(status_code=404, detail="Project not found or inactive")

        project = project_result.data[0]

        # Handle file upload if provided
        uploaded_file_info = None
        if file and file.filename:
            # Validate file type and size
            allowed_types = [
                "application/pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ]
            if file.content_type not in allowed_types:
                raise HTTPException(
                    status_code=400, detail="Only PDF and DOCX files are supported"
                )

            if file.size and file.size > 10 * 1024 * 1024:  # 10MB limit
                raise HTTPException(
                    status_code=400, detail="File size must be less than 10MB"
                )

            # Store file info for processing (Phase 3 will handle actual processing)
            uploaded_file_info = {
                "filename": file.filename,
                "content_type": file.content_type,
                "size": file.size,
                "status": "pending_processing",
            }

        # Initialize context stream for this analysis
        trace_id = str(uuid.uuid4())
        audit_stream = UnifiedContextStream(trace_id=trace_id)

        # Log analysis initiation
        await audit_stream.log_event(
            "ANALYSIS_INITIATED",
            {
                "project_id": project_id,
                "project_name": project["name"],
                "user_query": user_query,
                "engagement_type": engagement_type,
                "merge_project_context": merge_project_context,
                "has_uploaded_file": uploaded_file_info is not None,
                "uploaded_file": uploaded_file_info,
            },
            metadata={
                "api_version": "v2",
                "project_based": True,
                "context_merge_requested": merge_project_context,
                "file_upload_enabled": True,
            },
        )

        # Prepare initial context based on merge decision
        initial_context = None
        if merge_project_context:
            await audit_stream.log_event(
                "CONTEXT_MERGE_STARTED",
                {"project_id": project_id, "search_query": user_query},
            )

            # TODO: Implement RAG-based context retrieval
            # For now, log that merge was requested but not yet implemented
            await audit_stream.log_event(
                "CONTEXT_MERGE_PLACEHOLDER",
                {
                    "message": "RAG-based context retrieval will be implemented in Phase 3",
                    "fallback_action": "proceeding_without_context",
                },
            )

            initial_context = {
                "project_context_requested": True,
                "project_name": project["name"],
                "note": "Full RAG context merging will be available in Phase 3",
            }

        # Process uploaded document if provided
        document_ingestion_result = None
        if uploaded_file_info:
            await audit_stream.log_event(
                "FILE_UPLOAD_RECEIVED",
                {
                    "filename": uploaded_file_info["filename"],
                    "content_type": uploaded_file_info["content_type"],
                    "size_bytes": uploaded_file_info["size"],
                    "processing_status": "starting_ingestion",
                },
            )

            try:
                # Initialize document ingestion service
                doc_service = get_document_ingestion_service(
                    context_stream=audit_stream
                )

                # Process the document
                document_ingestion_result = await doc_service.ingest_document(
                    file=file,
                    project_id=project_id,
                    analysis_id=trace_id,
                    metadata={
                        "analysis_query": user_query,
                        "engagement_type": engagement_type,
                    },
                )

                # Update uploaded file info with processing results
                uploaded_file_info.update(
                    {
                        "status": "processed",
                        "chunks_created": document_ingestion_result["total_chunks"],
                        "text_preview": document_ingestion_result["text_preview"],
                        "processing_completed_at": datetime.now(
                            timezone.utc
                        ).isoformat(),
                    }
                )

                await audit_stream.log_event(
                    "DOCUMENT_PROCESSED",
                    {
                        "filename": uploaded_file_info["filename"],
                        "status": "success",
                        "chunks_created": document_ingestion_result["total_chunks"],
                        "text_length": len(
                            document_ingestion_result["document"]["text_content"]
                        ),
                    },
                )

            except DocumentProcessingError as e:
                # Log processing error but don't fail the analysis
                uploaded_file_info["status"] = "processing_failed"
                uploaded_file_info["error"] = str(e)

                await audit_stream.log_event(
                    "DOCUMENT_PROCESSING_ERROR",
                    {
                        "filename": uploaded_file_info["filename"],
                        "error": str(e),
                        "status": "failed",
                    },
                )

                logger.warning(
                    f"⚠️ Document processing failed for {uploaded_file_info['filename']}: {e}"
                )
                # Continue with analysis even if document processing fails
        else:
            await audit_stream.log_event(
                "CONTEXT_MERGE_SKIPPED", {"reason": "user_requested_fresh_start"}
            )

        # Create context stream record
        context_stream_data = {
            "trace_id": trace_id,
            "session_id": str(uuid.uuid4()),  # Generate session ID
            "organization_id": organization_id,
            "project_id": project_id,
            "engagement_type": engagement_type,
            "case_id": case_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "context_stream": await audit_stream.get_complete_log(),
            "final_status": "initiated",
            "rag_ingestion_status": "pending",
            "uploaded_file_info": uploaded_file_info,
        }

        # Insert context stream record
        stream_result = (
            supabase.table("context_streams").insert(context_stream_data).execute()
        )

        if not stream_result.data:
            raise HTTPException(
                status_code=500, detail="Failed to create analysis record"
            )

        analysis_record = stream_result.data[0]

        # RESOLVED: Trigger the actual analysis pipeline
        try:
            # Import the analysis components for pipeline execution
            from src.orchestration.dispatch_orchestrator import (
                StructuredAnalyticalFramework,
            )

            # Create framework from user's problem statement
            framework = StructuredAnalyticalFramework.from_prompt(
                user_prompt=user_query,
                additional_context="" if not initial_context else str(initial_context),
                trace_id=trace_id,
            )

            # Trigger async analysis pipeline (fire-and-forget for API responsiveness)
            import asyncio

            asyncio.create_task(
                _execute_analysis_pipeline(
                    framework=framework,
                    trace_id=trace_id,
                    project_id=project_id,
                    organization_id=organization_id,
                )
            )

            logger.info(f"🚀 Analysis pipeline triggered for {trace_id}")

        except Exception as e:
            logger.warning(f"⚠️ Analysis pipeline trigger failed for {trace_id}: {e}")
            # Continue without pipeline - analysis record still created for manual processing

        logger.info(
            f"✅ Analysis created for project {project_id}: {trace_id}"
            + (
                f" with file: {uploaded_file_info['filename']}"
                if uploaded_file_info
                else ""
            )
        )

        # Prepare response
        response_data = {
            "analysis_id": trace_id,
            "project_id": project_id,
            "project_name": project["name"],
            "status": "initiated",
            "context_merge_applied": merge_project_context,
            "initial_context": initial_context,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "uploaded_file": uploaded_file_info,
            "document_processing": document_ingestion_result,
            "message": "Analysis initiated successfully.",
        }

        # Add specific message based on file processing status
        if uploaded_file_info:
            if uploaded_file_info.get("status") == "processed":
                response_data[
                    "message"
                ] += f" Document '{uploaded_file_info['filename']}' processed successfully with {uploaded_file_info.get('chunks_created', 0)} chunks."
            elif uploaded_file_info.get("status") == "processing_failed":
                response_data[
                    "message"
                ] += f" Document '{uploaded_file_info['filename']}' processing failed, but analysis will continue."

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Analysis creation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create analysis: {str(e)}"
        )


@router.get("/{project_id}/knowledge-base", response_model=Dict[str, Any])
async def get_project_knowledge_base(
    project_id: str, supabase: Client = Depends(get_supabase)
) -> Dict[str, Any]:
    """
    Get comprehensive overview of project's knowledge base including
    document statistics, content types, and health metrics.
    """
    try:
        # Get knowledge base overview
        kb_result = (
            supabase.table("v2_project_knowledge_base")
            .select("*")
            .eq("project_id", project_id)
            .execute()
        )

        if not kb_result.data:
            return {
                "project_id": project_id,
                "total_documents": 0,
                "total_chunks": 0,
                "avg_content_quality": 0,
                "avg_semantic_density": 0,
                "latest_document_date": None,
                "document_type_distribution": {},
                "health_status": "empty",
            }

        kb_data = kb_result.data[0]

        # Determine health status
        health_status = "healthy"
        if kb_data["total_documents"] == 0:
            health_status = "empty"
        elif kb_data["avg_content_quality"] and kb_data["avg_content_quality"] < 0.6:
            health_status = "needs_improvement"
        elif not kb_data["latest_document_date"]:
            health_status = "stale"

        return {**kb_data, "health_status": health_status}

    except Exception as e:
        logger.error(f"❌ Knowledge base retrieval failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get knowledge base: {str(e)}"
        )


@router.post("/{project_id}/search", response_model=Dict[str, Any])
async def search_project_knowledge(
    project_id: str,
    request: ProjectKnowledgeSearchRequest,
    supabase: Client = Depends(get_supabase),
) -> Dict[str, Any]:
    """
    Search project knowledge base using semantic and keyword search.
    This will be fully implemented with vector search in Phase 3.
    """
    try:
        # Validate project exists
        project_result = (
            supabase.table("projects")
            .select("name")
            .eq("project_id", project_id)
            .execute()
        )
        if not project_result.data:
            raise HTTPException(status_code=404, detail="Project not found")

        project_name = project_result.data[0]["name"]

        # TODO: Implement full semantic search with embeddings in Phase 3
        # For now, implement basic text search on available chunks

        query = (
            supabase.table("rag_text_chunks")
            .select("chunk_id, content, content_type, created_at, document_id")
            .eq("project_id", project_id)
        )

        # Apply content type filters if specified
        if request.content_types:
            query = query.in_("content_type", request.content_types)

        # Basic text search (placeholder for vector search)
        # Note: This is a simple implementation - full semantic search comes in Phase 3
        query = query.ilike("content", f"%{request.query}%")

        result = (
            query.order("created_at", desc=True).limit(request.max_results).execute()
        )

        search_results = []
        if result.data:
            for chunk in result.data:
                search_results.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "content": chunk["content"],
                        "content_type": chunk["content_type"],
                        "document_id": chunk["document_id"],
                        "created_at": chunk["created_at"],
                        "similarity_score": 0.8,  # Placeholder - will be real similarity in Phase 3
                        "relevance_explanation": "Text match found",  # Placeholder
                    }
                )

        logger.info(
            f"✅ Knowledge search completed for project {project_id}: {len(search_results)} results"
        )

        return {
            "project_id": project_id,
            "project_name": project_name,
            "query": request.query,
            "results": search_results,
            "total_results": len(search_results),
            "search_type": "text_match",  # Will be "semantic_vector" in Phase 3
            "note": "Full semantic vector search will be available in Phase 3",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Knowledge search failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Knowledge search failed: {str(e)}"
        )


@router.get("/{project_id}/statistics", response_model=Dict[str, Any])
async def get_project_statistics(
    project_id: str, supabase: Client = Depends(get_supabase)
) -> Dict[str, Any]:
    """
    Get comprehensive project statistics including usage analytics,
    cost tracking, and performance metrics.
    """
    try:
        # Get project statistics using the helper function
        stats_result = supabase.rpc(
            "get_project_stats", {"project_id_param": project_id}
        ).execute()

        if not stats_result.data:
            raise HTTPException(status_code=404, detail="Project not found")

        stats = stats_result.data[0]

        # Get recent activity (last 30 days)
        recent_activity = (
            supabase.table("context_streams")
            .select("completed_at, final_status, total_tokens, total_cost")
            .eq("project_id", project_id)
            .gte("started_at", (datetime.now(timezone.utc).replace(day=1)).isoformat())
            .execute()
        )

        # Calculate additional metrics
        recent_analyses = len(recent_activity.data) if recent_activity.data else 0
        avg_tokens_per_analysis = (
            (stats["total_tokens_used"] // stats["total_analyses"])
            if stats["total_analyses"] > 0
            else 0
        )
        avg_cost_per_analysis = (
            (stats["total_cost"] / stats["total_analyses"])
            if stats["total_analyses"] > 0
            else 0
        )

        # Calculate ROI metrics (placeholder)
        roi_score = min(
            100, (stats["total_analyses"] * 10) + (stats["total_rag_documents"] * 5)
        )

        return {
            **stats,
            "recent_analyses_30d": recent_analyses,
            "avg_tokens_per_analysis": avg_tokens_per_analysis,
            "avg_cost_per_analysis": round(avg_cost_per_analysis, 6),
            "roi_score": roi_score,
            "efficiency_metrics": {
                "knowledge_base_utilization": min(
                    100,
                    (stats["total_text_chunks"] / max(1, stats["total_analyses"])) * 10,
                ),
                "context_merge_potential": stats["total_rag_documents"] > 0,
                "cost_efficiency_rating": (
                    "excellent"
                    if avg_cost_per_analysis < 0.10
                    else (
                        "good" if avg_cost_per_analysis < 0.50 else "needs_optimization"
                    )
                ),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Project statistics retrieval failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get project statistics: {str(e)}"
        )


# ============================================================
# HELPER FUNCTIONS
# ============================================================


async def _execute_analysis_pipeline(
    framework,
    trace_id: str,
    project_id: str,
    organization_id: str,
) -> None:
    """Execute the analysis pipeline asynchronously"""
    try:
        from src.orchestration.dispatch_orchestrator import DispatchOrchestrator
        from src.services.selection.nway_pattern_service import NWayPatternService
        from src.engine.agents.quality_rater_agent_v2 import get_quality_rater
        from src.integrations.llm.unified_client import UnifiedLLMClient

        logger.info(f"🚀 Starting analysis pipeline for {trace_id}")

        # Initialize components
        orchestrator = DispatchOrchestrator()
        nway_service = NWayPatternService()
        quality_rater = get_quality_rater()
        llm_client = UnifiedLLMClient()

        # Execute dispatch orchestration
        dispatch_package = await orchestrator.run_dispatch(framework)

        # Get mental models from NWay service
        models = await nway_service.get_models_for_consultant(
            dispatch_package.selected_consultants[0].consultant_id
            if dispatch_package.selected_consultants
            else "strategic_analyst"
        )

        # Generate analysis using unified LLM client
        analysis_result = await llm_client.call_llm_unified(
            prompt=framework.user_prompt,
            task_name="strategic_analysis",
            system_prompt=f"You are a {dispatch_package.selected_consultants[0].consultant_id if dispatch_package.selected_consultants else 'strategic analyst'} consultant.",
            models=models[:3] if models else [],  # Use top 3 models
        )

        # Rate quality of analysis
        quality_score = await quality_rater.rate_quality(
            analysis_content=analysis_result,
            context={
                "user_prompt": framework.user_prompt,
                "trace_id": trace_id,
                "project_id": project_id,
            },
        )

        logger.info(
            f"✅ Analysis pipeline completed for {trace_id} with quality score: {quality_score.get('total', 0):.2f}"
        )

    except Exception as e:
        logger.error(f"❌ Analysis pipeline failed for {trace_id}: {e}")
        # Pipeline failure doesn't prevent API response - analysis record was already created
