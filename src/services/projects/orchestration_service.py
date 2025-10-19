"""
Project Orchestration Service
=============================

Complex workflow coordination service for project operations. Extracts all
orchestration logic from the monolithic project service, handling multi-step
workflows that require coordination between multiple services and systems.

Responsibilities:
- Analysis creation workflow orchestration
- Context merging coordination  
- Mental model ingestion workflow
- Document processing coordination
- Multi-service transaction management
- Event emission and audit trail management

This service coordinates between repository, validation, analytics, and
external services while maintaining consistency and observability.
"""

import logging
import uuid
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from fastapi import UploadFile

from .specialized_contracts import (
    IProjectOrchestrator,
    WorkflowContext,
    WorkflowResult,
    ProjectOrchestrationError,
    IProjectRepository,
    IProjectValidator,
    IProjectAnalytics,
)
from .contracts import (
    AnalysisCreateRequest,
    ContextMergePreview,
    MentalModelIngestionRequest,
    MentalModelIngestionResponse,
    BatchCQARequest,
    BatchCQAResponse,
    ProjectKnowledgeSearchRequest,
)
from src.core.unified_context_stream import get_unified_context_stream, UnifiedContextStream
from src.services.document_ingestion_service import (
    get_document_ingestion_service,
    DocumentProcessingError,
)


class ProjectOrchestrationService(IProjectOrchestrator):
    """
    Complex workflow coordination service for projects
    
    Handles multi-step workflows that require coordination between multiple
    services while maintaining consistency, observability, and error recovery.
    """
    
    def __init__(
        self,
        repository: IProjectRepository,
        validator: IProjectValidator,
        analytics: IProjectAnalytics,
        context_stream: Optional[UnifiedContextStream] = None
    ):
        """Initialize orchestration service with dependencies"""
        self.logger = logging.getLogger(__name__)
        self.repository = repository
        self.validator = validator
        self.analytics = analytics
        self.context_stream = context_stream or get_unified_context_stream()
        
        self.logger.debug("🏗️ ProjectOrchestrationService initialized")
    
    async def orchestrate_analysis_creation(
        self, 
        request: AnalysisCreateRequest,
        context: WorkflowContext
    ) -> WorkflowResult:
        """Orchestrate complete analysis creation workflow"""
        workflow_id = context.workflow_id
        stages_completed = []
        
        try:
            self.logger.info(f"🚀 Starting analysis creation workflow: {workflow_id}")
            
            await self.context_stream.log_event(
                "ANALYSIS_WORKFLOW_STARTED",
                {
                    "workflow_id": workflow_id,
                    "project_id": request.project_id,
                    "user_query": request.user_query,
                    "engagement_type": request.engagement_type,
                },
                metadata={"workflow": "analysis_creation", "stage": "initiation"}
            )
            
            # Stage 1: Validation
            await self.context_stream.log_event(
                "ANALYSIS_VALIDATION_STARTED",
                {"workflow_id": workflow_id},
                metadata={"stage": "validation"}
            )
            
            validation_result = await self.validator.validate_analysis_request(request)
            if not validation_result.is_valid:
                raise ProjectOrchestrationError(
                    "analysis_creation",
                    "validation",
                    {"validation_errors": validation_result.errors}
                )
            
            stages_completed.append("validation")
            
            await self.context_stream.log_event(
                "ANALYSIS_VALIDATION_COMPLETED",
                {
                    "workflow_id": workflow_id,
                    "warnings": validation_result.warnings,
                },
                metadata={"stage": "validation"}
            )
            
            # Stage 2: Project verification
            await self.context_stream.log_event(
                "PROJECT_VERIFICATION_STARTED",
                {"workflow_id": workflow_id, "project_id": request.project_id},
                metadata={"stage": "project_verification"}
            )
            
            project = await self.repository.get_project_by_id(request.project_id)
            if not project or project.status != "active":
                raise ProjectOrchestrationError(
                    "analysis_creation",
                    "project_verification",
                    {"project_id": request.project_id, "status": project.status if project else "not_found"}
                )
            
            stages_completed.append("project_verification")
            
            await self.context_stream.log_event(
                "PROJECT_VERIFICATION_COMPLETED",
                {
                    "workflow_id": workflow_id,
                    "project_id": request.project_id,
                    "project_name": project.name,
                },
                metadata={"stage": "project_verification"}
            )
            
            # Stage 3: Context preparation
            if request.merge_project_context:
                await self._prepare_project_context(request, context, project)
                stages_completed.append("context_preparation")
            
            # Stage 4: File processing (if provided)
            file_processing_result = None
            if hasattr(context, 'uploaded_file') and context.parameters.get('uploaded_file'):
                file_processing_result = await self._process_uploaded_file(
                    request, context, context.parameters['uploaded_file']
                )
                stages_completed.append("file_processing")
            
            # Stage 5: Analysis record creation
            analysis_record = await self._create_analysis_record(
                request, context, project, file_processing_result
            )
            stages_completed.append("analysis_creation")
            
            # Stage 6: Analysis pipeline trigger (placeholder)
            await self._trigger_analysis_pipeline(analysis_record, context)
            stages_completed.append("pipeline_trigger")
            
            # Stage 7: Workflow completion
            final_result = {
                "analysis_id": analysis_record["trace_id"],
                "project_id": request.project_id,
                "project_name": project.name,
                "status": "initiated",
                "context_merge_applied": request.merge_project_context,
                "created_at": analysis_record["started_at"],
                "file_processing_result": file_processing_result,
                "message": "Analysis initiated successfully",
            }
            
            await self.context_stream.log_event(
                "ANALYSIS_WORKFLOW_COMPLETED",
                {
                    "workflow_id": workflow_id,
                    "analysis_id": analysis_record["trace_id"],
                    "stages_completed": stages_completed,
                },
                metadata={"workflow": "analysis_creation", "stage": "completion"}
            )
            
            self.logger.info(f"✅ Analysis creation workflow completed: {workflow_id}")
            
            return WorkflowResult(
                workflow_id=workflow_id,
                status="completed",
                stages_completed=stages_completed,
                final_result=final_result,
                execution_time_ms=int((datetime.now(timezone.utc) - context.started_at).total_seconds() * 1000),
                errors=[]
            )
            
        except Exception as e:
            self.logger.error(f"❌ Analysis creation workflow failed: {e}")
            
            await self.context_stream.log_event(
                "ANALYSIS_WORKFLOW_FAILED",
                {
                    "workflow_id": workflow_id,
                    "error": str(e),
                    "stages_completed": stages_completed,
                },
                metadata={"workflow": "analysis_creation", "stage": "error"}
            )
            
            if isinstance(e, ProjectOrchestrationError):
                raise
            
            raise ProjectOrchestrationError(
                "analysis_creation",
                stages_completed[-1] if stages_completed else "unknown",
                {"original_error": str(e), "stages_completed": stages_completed}
            ) from e
    
    async def orchestrate_context_merge(
        self, 
        project_id: str,
        query: str,
        context: WorkflowContext
    ) -> ContextMergePreview:
        """Orchestrate context merging workflow"""
        try:
            self.logger.info(f"🔄 Starting context merge workflow for project: {project_id}")
            
            await self.context_stream.log_event(
                "CONTEXT_MERGE_WORKFLOW_STARTED",
                {
                    "workflow_id": context.workflow_id,
                    "project_id": project_id,
                    "query": query,
                },
                metadata={"workflow": "context_merge"}
            )
            
            # Get project information
            project = await self.repository.get_project_by_id(project_id)
            if not project:
                raise ProjectOrchestrationError(
                    "context_merge",
                    "project_lookup",
                    {"project_id": project_id}
                )
            
            # Get knowledge base statistics
            kb_stats = await self.repository.get_knowledge_base_stats(project_id)
            
            # Search for relevant context
            search_results = await self.repository.search_project_knowledge(
                project_id, query, limit=5
            )
            
            # Calculate estimated tokens and cost
            estimated_tokens = sum(len(result["content"]) // 4 for result in search_results)
            estimated_cost = estimated_tokens * 0.0001  # Rough calculation
            
            # Create preview
            preview = ContextMergePreview(
                project_name=project.name,
                total_available_documents=kb_stats.get("total_documents", 0),
                total_available_chunks=kb_stats.get("total_chunks", 0),
                preview_chunks=[
                    {
                        "content": result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"],
                        "content_type": result["content_type"],
                        "created_at": result["created_at"],
                        "document_id": result["document_id"],
                    }
                    for result in search_results
                ],
                estimated_context_tokens=estimated_tokens,
                estimated_additional_cost=estimated_cost,
            )
            
            await self.context_stream.log_event(
                "CONTEXT_MERGE_PREVIEW_GENERATED",
                {
                    "workflow_id": context.workflow_id,
                    "project_id": project_id,
                    "preview_chunks_count": len(search_results),
                    "estimated_tokens": estimated_tokens,
                },
                metadata={"workflow": "context_merge"}
            )
            
            self.logger.info(f"✅ Context merge preview generated for project: {project_id}")
            
            return preview
            
        except Exception as e:
            self.logger.error(f"❌ Context merge workflow failed: {e}")
            
            if isinstance(e, ProjectOrchestrationError):
                raise
            
            raise ProjectOrchestrationError(
                "context_merge",
                "unknown",
                {"project_id": project_id, "query": query, "original_error": str(e)}
            ) from e
    
    async def orchestrate_mental_model_ingestion(
        self, 
        request: MentalModelIngestionRequest,
        context: WorkflowContext
    ) -> MentalModelIngestionResponse:
        """Orchestrate mental model ingestion workflow"""
        workflow_id = context.workflow_id
        started_at = datetime.now(timezone.utc)
        
        try:
            self.logger.info(f"🧠 Starting mental model ingestion workflow: {workflow_id}")
            
            await self.context_stream.log_event(
                "MENTAL_MODEL_WORKFLOW_STARTED",
                {
                    "workflow_id": workflow_id,
                    "request": request.dict() if hasattr(request, 'dict') else str(request),
                },
                metadata={"workflow": "mental_model_ingestion"}
            )
            
            # Validation
            validation_result = await self.validator.validate_mental_model_request(request)
            if not validation_result.is_valid:
                raise ProjectOrchestrationError(
                    "mental_model_ingestion",
                    "validation",
                    {"validation_errors": validation_result.errors}
                )
            
            # Project verification (if project_id provided)
            project = None
            if hasattr(request, 'project_id') and request.project_id:
                project = await self.repository.get_project_by_id(request.project_id)
                if not project:
                    raise ProjectOrchestrationError(
                        "mental_model_ingestion",
                        "project_verification",
                        {"project_id": request.project_id}
                    )
            
            # Mental model processing (placeholder implementation)
            # In a real implementation, this would:
            # 1. Parse mental model files
            # 2. Extract structured data
            # 3. Validate model quality
            # 4. Ingest into RAG system
            
            completed_at = datetime.now(timezone.utc)
            processing_duration = int((completed_at - started_at).total_seconds())
            
            # Generate response
            response = MentalModelIngestionResponse(
                ingestion_id=workflow_id,
                project_id=getattr(request, 'project_id', None),
                project_name=project.name if project else None,
                status="completed",
                files_processed=0,  # Placeholder
                files_successful=0,  # Placeholder
                files_failed=0,  # Placeholder
                estimated_chunks_created=0,  # Placeholder
                processing_duration_seconds=processing_duration,
                failed_files=[],
                success_percentage=100.0,
                next_steps=[
                    "Verify ingestion in project knowledge base",
                    "Test context merging functionality",
                    "Run CQA evaluation on ingested models",
                ],
                completed_at=completed_at,
            )
            
            await self.context_stream.log_event(
                "MENTAL_MODEL_WORKFLOW_COMPLETED",
                {
                    "workflow_id": workflow_id,
                    "ingestion_id": response.ingestion_id,
                    "processing_duration": processing_duration,
                },
                metadata={"workflow": "mental_model_ingestion"}
            )
            
            self.logger.info(f"✅ Mental model ingestion workflow completed: {workflow_id}")
            
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Mental model ingestion workflow failed: {e}")
            
            await self.context_stream.log_event(
                "MENTAL_MODEL_WORKFLOW_FAILED",
                {
                    "workflow_id": workflow_id,
                    "error": str(e),
                },
                metadata={"workflow": "mental_model_ingestion"}
            )
            
            if isinstance(e, ProjectOrchestrationError):
                raise
            
            raise ProjectOrchestrationError(
                "mental_model_ingestion",
                "unknown",
                {"request": str(request), "original_error": str(e)}
            ) from e
    
    async def orchestrate_batch_cqa_evaluation(
        self, 
        request: BatchCQARequest,
        context: WorkflowContext
    ) -> BatchCQAResponse:
        """Orchestrate batch CQA evaluation workflow"""
        workflow_id = context.workflow_id
        started_at = datetime.now(timezone.utc)
        
        try:
            self.logger.info(f"🔍 Starting batch CQA evaluation workflow: {workflow_id}")
            
            await self.context_stream.log_event(
                "BATCH_CQA_WORKFLOW_STARTED",
                {
                    "workflow_id": workflow_id,
                    "project_id": getattr(request, 'project_id', None),
                },
                metadata={"workflow": "batch_cqa_evaluation"}
            )
            
            # Project verification (if project_id provided)
            project = None
            if hasattr(request, 'project_id') and request.project_id:
                project = await self.repository.get_project_by_id(request.project_id)
                if not project:
                    raise ProjectOrchestrationError(
                        "batch_cqa_evaluation",
                        "project_verification",
                        {"project_id": request.project_id}
                    )
            
            # Placeholder implementation for batch CQA
            # Real implementation would:
            # 1. Discover mental models in project
            # 2. Run CQA evaluation on each
            # 3. Aggregate results
            # 4. Generate summary statistics
            
            completed_at = datetime.now(timezone.utc)
            
            response = BatchCQAResponse(
                batch_id=workflow_id,
                project_id=getattr(request, 'project_id', None),
                project_name=project.name if project else None,
                total_models=0,
                evaluated_models=0,
                failed_evaluations=0,
                average_quality_score=0.0,
                quality_distribution={"excellent": 0, "good": 0, "average": 0, "poor": 0},
                validation_summary={"passed": 0, "failed": 0, "review_needed": 0},
                started_at=started_at,
                completed_at=completed_at,
                results=[],
            )
            
            await self.context_stream.log_event(
                "BATCH_CQA_WORKFLOW_COMPLETED",
                {
                    "workflow_id": workflow_id,
                    "batch_id": response.batch_id,
                    "models_evaluated": response.evaluated_models,
                },
                metadata={"workflow": "batch_cqa_evaluation"}
            )
            
            self.logger.info(f"✅ Batch CQA evaluation workflow completed: {workflow_id}")
            
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Batch CQA evaluation workflow failed: {e}")
            
            if isinstance(e, ProjectOrchestrationError):
                raise
            
            raise ProjectOrchestrationError(
                "batch_cqa_evaluation",
                "unknown",
                {"request": str(request), "original_error": str(e)}
            ) from e
    
    async def orchestrate_knowledge_search(
        self, 
        project_id: str,
        request: ProjectKnowledgeSearchRequest,
        context: WorkflowContext
    ) -> Dict[str, Any]:
        """Orchestrate knowledge base search workflow"""
        try:
            self.logger.info(f"🔍 Starting knowledge search workflow for project: {project_id}")
            
            await self.context_stream.log_event(
                "KNOWLEDGE_SEARCH_WORKFLOW_STARTED",
                {
                    "workflow_id": context.workflow_id,
                    "project_id": project_id,
                    "query": request.query,
                    "search_type": getattr(request, 'search_type', 'semantic'),
                },
                metadata={"workflow": "knowledge_search"}
            )
            
            # Project verification
            project = await self.repository.get_project_by_id(project_id)
            if not project:
                raise ProjectOrchestrationError(
                    "knowledge_search",
                    "project_verification",
                    {"project_id": project_id}
                )
            
            # Perform search
            search_filters = {
                "content_types": getattr(request, 'content_types', None),
            }
            
            search_results = await self.repository.search_project_knowledge(
                project_id,
                request.query,
                filters=search_filters,
                limit=getattr(request, 'max_results', 10)
            )
            
            # Prepare response
            response = {
                "project_id": project_id,
                "project_name": project.name,
                "query": request.query,
                "results": search_results,
                "total_results": len(search_results),
                "search_type": "text_match",  # Placeholder for semantic search
                "note": "Full semantic vector search will be available in future updates",
            }
            
            await self.context_stream.log_event(
                "KNOWLEDGE_SEARCH_WORKFLOW_COMPLETED",
                {
                    "workflow_id": context.workflow_id,
                    "project_id": project_id,
                    "results_count": len(search_results),
                },
                metadata={"workflow": "knowledge_search"}
            )
            
            self.logger.info(f"✅ Knowledge search workflow completed for project: {project_id}")
            
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Knowledge search workflow failed: {e}")
            
            if isinstance(e, ProjectOrchestrationError):
                raise
            
            raise ProjectOrchestrationError(
                "knowledge_search",
                "unknown",
                {
                    "project_id": project_id,
                    "request": request.dict() if hasattr(request, 'dict') else str(request),
                    "original_error": str(e)
                }
            ) from e
    
    # ============================================================
    # Private Helper Methods
    # ============================================================
    
    async def _prepare_project_context(
        self, 
        request: AnalysisCreateRequest,
        context: WorkflowContext,
        project
    ) -> None:
        """Prepare project context for analysis"""
        await self.context_stream.log_event(
            "CONTEXT_PREPARATION_STARTED",
            {
                "workflow_id": context.workflow_id,
                "project_id": request.project_id,
                "search_query": request.user_query,
            },
            metadata={"stage": "context_preparation"}
        )
        
        # TODO: Implement actual context preparation
        # This would involve:
        # 1. Searching relevant knowledge base content
        # 2. Extracting relevant context chunks
        # 3. Preparing context for analysis pipeline
        
        await self.context_stream.log_event(
            "CONTEXT_PREPARATION_COMPLETED",
            {
                "workflow_id": context.workflow_id,
                "message": "Context preparation placeholder completed",
            },
            metadata={"stage": "context_preparation"}
        )
    
    async def _process_uploaded_file(
        self, 
        request: AnalysisCreateRequest,
        context: WorkflowContext,
        uploaded_file: UploadFile
    ) -> Dict[str, Any]:
        """Process uploaded file for analysis"""
        await self.context_stream.log_event(
            "FILE_PROCESSING_STARTED",
            {
                "workflow_id": context.workflow_id,
                "filename": uploaded_file.filename,
                "content_type": uploaded_file.content_type,
                "size": uploaded_file.size,
            },
            metadata={"stage": "file_processing"}
        )
        
        try:
            # Initialize document ingestion service
            doc_service = get_document_ingestion_service(
                context_stream=self.context_stream
            )
            
            # Process the document
            processing_result = await doc_service.ingest_document(
                file=uploaded_file,
                project_id=request.project_id,
                analysis_id=context.trace_id,
                metadata={
                    "analysis_query": request.user_query,
                    "engagement_type": request.engagement_type,
                    "workflow_id": context.workflow_id,
                },
            )
            
            await self.context_stream.log_event(
                "FILE_PROCESSING_COMPLETED",
                {
                    "workflow_id": context.workflow_id,
                    "filename": uploaded_file.filename,
                    "chunks_created": processing_result["total_chunks"],
                },
                metadata={"stage": "file_processing"}
            )
            
            return {
                "status": "processed",
                "filename": uploaded_file.filename,
                "chunks_created": processing_result["total_chunks"],
                "text_preview": processing_result["text_preview"],
                "processing_completed_at": datetime.now(timezone.utc).isoformat(),
            }
            
        except DocumentProcessingError as e:
            await self.context_stream.log_event(
                "FILE_PROCESSING_FAILED",
                {
                    "workflow_id": context.workflow_id,
                    "filename": uploaded_file.filename,
                    "error": str(e),
                },
                metadata={"stage": "file_processing"}
            )
            
            return {
                "status": "processing_failed",
                "filename": uploaded_file.filename,
                "error": str(e),
            }
    
    async def _create_analysis_record(
        self, 
        request: AnalysisCreateRequest,
        context: WorkflowContext,
        project,
        file_processing_result: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create analysis record in database"""
        await self.context_stream.log_event(
            "ANALYSIS_RECORD_CREATION_STARTED",
            {
                "workflow_id": context.workflow_id,
                "project_id": request.project_id,
            },
            metadata={"stage": "analysis_creation"}
        )
        
        # Create analysis record data
        analysis_record = {
            "trace_id": context.trace_id,
            "session_id": str(uuid.uuid4()),
            "organization_id": request.organization_id,
            "project_id": request.project_id,
            "engagement_type": request.engagement_type,
            "case_id": request.case_id,
            "started_at": context.started_at.isoformat(),
            "context_stream": await self.context_stream.get_complete_log(),
            "final_status": "initiated",
            "rag_ingestion_status": "pending",
            "uploaded_file_info": file_processing_result,
            "workflow_id": context.workflow_id,
        }
        
        # Note: In a real implementation, this would use repository service
        # to persist the analysis record. For now, it's a placeholder.
        
        await self.context_stream.log_event(
            "ANALYSIS_RECORD_CREATED",
            {
                "workflow_id": context.workflow_id,
                "analysis_id": context.trace_id,
                "project_name": project.name,
            },
            metadata={"stage": "analysis_creation"}
        )
        
        return analysis_record
    
    async def _trigger_analysis_pipeline(
        self, 
        analysis_record: Dict[str, Any],
        context: WorkflowContext
    ) -> None:
        """Trigger analysis pipeline execution"""
        await self.context_stream.log_event(
            "ANALYSIS_PIPELINE_TRIGGER_STARTED",
            {
                "workflow_id": context.workflow_id,
                "analysis_id": analysis_record["trace_id"],
            },
            metadata={"stage": "pipeline_trigger"}
        )
        
        # TODO: Implement actual pipeline trigger
        # This would involve:
        # 1. Preparing pipeline configuration
        # 2. Triggering analysis execution
        # 3. Setting up monitoring and callbacks
        
        await self.context_stream.log_event(
            "ANALYSIS_PIPELINE_TRIGGERED",
            {
                "workflow_id": context.workflow_id,
                "analysis_id": analysis_record["trace_id"],
                "message": "Pipeline trigger placeholder completed",
            },
            metadata={"stage": "pipeline_trigger"}
        )


# ============================================================
# Factory Function
# ============================================================

def get_project_orchestrator(
    repository: IProjectRepository,
    validator: IProjectValidator,
    analytics: IProjectAnalytics,
    context_stream: Optional[UnifiedContextStream] = None
) -> IProjectOrchestrator:
    """Factory function for dependency injection"""
    return ProjectOrchestrationService(
        repository=repository,
        validator=validator,
        analytics=analytics,
        context_stream=context_stream
    )
