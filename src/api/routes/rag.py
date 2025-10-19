"""
RAG API Router for METIS 2.0
============================

API endpoints for Retrieval-Augmented Generation (RAG) operations.
Provides multi-tenant document ingestion and semantic search capabilities.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from uuid import UUID
import os
import time
import logging
from datetime import datetime

from src.rag.rag_pipeline import EnhancedRAGPipeline, RAGPipelineFactory
from src.rag.project_rag_pipeline import get_project_rag_pipeline
from src.core.unified_context_stream import UnifiedContextStream
from src.web_intelligence.web_intelligence_manager import WebIntelligenceManager
from src.engine.core.llm_manager import LLMManager
from src.storage.zep_memory import ZepMemoryManager, ZepMessage

# Router setup
router = APIRouter(prefix="/api/v2/rag", tags=["RAG"])

# Global instances
_rag_pipeline: Optional[EnhancedRAGPipeline] = None
_llm_manager: Optional[LLMManager] = None
_zep_memory: Optional[ZepMemoryManager] = None


# Request/Response Models
class IngestTextRequest(BaseModel):
    """Request model for text ingestion"""

    content: str = Field(..., min_length=10, description="Text content to ingest")
    title: str = Field(..., min_length=1, description="Document title")
    source_type: str = Field(default="api", description="Source type identifier")
    url: Optional[str] = Field(None, description="Source URL if applicable")
    author: Optional[str] = Field(None, description="Document author")
    tags: Optional[List[str]] = Field(None, description="Document tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    organization_id: str = Field(..., description="Organization ID for multi-tenancy")


class IngestTextResponse(BaseModel):
    """Response model for text ingestion"""

    success: bool
    document_id: str
    message: str
    timestamp: datetime
    organization_id: str


class SearchRequest(BaseModel):
    """Request model for semantic search"""

    query: str = Field(..., min_length=1, description="Search query")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of results")
    organization_id: str = Field(..., description="Organization ID for multi-tenancy")
    min_similarity: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Minimum similarity threshold"
    )


class SearchResult(BaseModel):
    """Individual search result"""

    document_id: str
    title: str
    content: str
    score: float
    source: str
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    """Response model for search"""

    success: bool
    results: List[SearchResult]
    query: str
    total_results: int
    processing_time_ms: int
    timestamp: datetime
    organization_id: str


class IngestUrlRequest(BaseModel):
    """Request model for URL ingestion"""

    url: str = Field(..., min_length=1, description="URL to extract content from")
    project_id: UUID = Field(..., description="Project ID for project-scoped storage")
    extraction_type: str = Field(
        default="comprehensive",
        description="Extraction type: comprehensive, quick, or specific",
    )
    title: Optional[str] = Field(
        None, description="Optional custom title for the document"
    )
    tags: Optional[List[str]] = Field(None, description="Document tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    organization_id: str = Field(..., description="Organization ID for multi-tenancy")


class IngestUrlResponse(BaseModel):
    """Response model for URL ingestion"""

    success: bool
    document_ids: List[str]
    url: str
    extraction_metadata: Dict[str, Any]
    processing_time_ms: int
    timestamp: datetime
    project_id: UUID
    organization_id: str


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    components: Dict[str, str]
    uptime_seconds: float
    document_count: int
    search_count: int


class ChatRequest(BaseModel):
    """Request model for project chat"""

    project_id: UUID = Field(..., description="Project ID for context scoping")
    user_question: str = Field(
        ..., min_length=1, description="User's question about the project"
    )
    session_id: Optional[str] = Field(
        None, description="Optional conversation session ID"
    )
    user_id: str = Field(..., description="User ID for conversation tracking")
    organization_id: str = Field(..., description="Organization ID for multi-tenancy")
    max_context_chunks: int = Field(
        default=5, ge=1, le=20, description="Maximum RAG context chunks to retrieve"
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="LLM temperature for response generation",
    )


class ChatResponse(BaseModel):
    """Response model for project chat"""

    success: bool
    answer: str
    sources: List[Dict[str, Any]]
    session_id: str
    conversation_turn: int
    processing_time_ms: int
    rag_context_used: bool
    context_chunks_retrieved: int
    project_id: UUID
    timestamp: datetime


# Dependencies
async def get_llm_manager() -> LLMManager:
    """Get or initialize the LLM manager"""
    global _llm_manager

    if _llm_manager is None:
        from src.core.unified_context_stream import get_unified_context_stream
        context_stream = get_unified_context_stream()
        _llm_manager = LLMManager(context_stream=context_stream)

    return _llm_manager


async def get_zep_memory() -> ZepMemoryManager:
    """Get or initialize Zep memory manager"""
    global _zep_memory

    if _zep_memory is None:
        from src.core.unified_context_stream import get_unified_context_stream
        context_stream = get_unified_context_stream()
        _zep_memory = ZepMemoryManager(context_stream=context_stream)

    return _zep_memory


async def get_rag_pipeline() -> EnhancedRAGPipeline:
    """Get or initialize the RAG pipeline"""
    global _rag_pipeline

    if _rag_pipeline is None:
        voyage_api_key = os.getenv("VOYAGE_API_KEY")
        if not voyage_api_key:
            raise HTTPException(status_code=500, detail="VOYAGE_API_KEY not configured")

        _rag_pipeline = RAGPipelineFactory.create_default_pipeline(
            voyage_api_key=voyage_api_key
        )

        # Initialize pipeline
        success = await _rag_pipeline.initialize()
        if not success:
            raise HTTPException(
                status_code=500, detail="Failed to initialize RAG pipeline"
            )

    return _rag_pipeline


# API Endpoints


@router.post("/ingest/text", response_model=IngestTextResponse)
async def ingest_text(
    request: IngestTextRequest,
    rag_pipeline: EnhancedRAGPipeline = Depends(get_rag_pipeline),
):
    """
    Ingest text document into the knowledge base with multi-tenant isolation

    This endpoint allows organizations to add text documents to their private
    knowledge base partition. Each organization's data is isolated using the
    organization_id parameter.
    """
    start_time = datetime.now()

    try:
        # Add organization_id to metadata for multi-tenant isolation
        metadata = request.metadata or {}
        metadata["organization_id"] = request.organization_id
        metadata["ingested_at"] = start_time.isoformat()
        metadata["source_api"] = "v2_rag_ingest_text"

        # Ingest document
        document_id = await rag_pipeline.add_document(
            content=request.content,
            title=request.title,
            source_type=request.source_type,
            url=request.url,
            author=request.author,
            tags=request.tags,
            metadata=metadata,
        )

        if not document_id:
            raise HTTPException(status_code=500, detail="Failed to ingest document")

        return IngestTextResponse(
            success=True,
            document_id=document_id,
            message=f"Document '{request.title}' successfully ingested",
            timestamp=datetime.now(),
            organization_id=request.organization_id,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error ingesting document: {str(e)}"
        )


@router.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    rag_pipeline: EnhancedRAGPipeline = Depends(get_rag_pipeline),
):
    """
    Perform semantic search within organization's knowledge base

    Searches are automatically scoped to the specified organization using
    multi-tenant data isolation. Only documents belonging to the organization
    will be returned in results.
    """
    start_time = datetime.now()

    try:
        # Add organization filter for multi-tenant isolation
        filters = {"organization_id": request.organization_id}

        # Perform search
        results = await rag_pipeline.intelligent_search(
            query=request.query,
            limit=request.limit,
            filters=filters,
            min_similarity=request.min_similarity,
        )

        # Convert results to response format
        search_results = []
        for result in results:
            search_results.append(
                SearchResult(
                    document_id=result.get("id", "unknown"),
                    title=result.get("title", ""),
                    content=result.get("content", ""),
                    score=result.get("score", 0.0),
                    source=result.get("source", ""),
                    metadata=result.get("metadata", {}),
                )
            )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return SearchResponse(
            success=True,
            results=search_results,
            query=request.query,
            total_results=len(search_results),
            processing_time_ms=int(processing_time),
            timestamp=datetime.now(),
            organization_id=request.organization_id,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error performing search: {str(e)}"
        )


@router.get("/health", response_model=HealthResponse)
async def health_check(rag_pipeline: EnhancedRAGPipeline = Depends(get_rag_pipeline)):
    """
    Get RAG system health status and statistics
    """
    try:
        # Get health status
        health = await rag_pipeline.health_check()

        # Get statistics
        stats = await rag_pipeline.get_collection_stats()
        pipeline_stats = stats.get("pipeline_stats", {})

        return HealthResponse(
            status=health.get("overall_health", "unknown"),
            components=health.get("components", {}),
            uptime_seconds=pipeline_stats.get("uptime_seconds", 0),
            document_count=pipeline_stats.get("documents_indexed", 0),
            search_count=pipeline_stats.get("searches_performed", 0),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting health status: {str(e)}"
        )


@router.get("/stats")
async def get_statistics(rag_pipeline: EnhancedRAGPipeline = Depends(get_rag_pipeline)):
    """
    Get detailed RAG pipeline statistics
    """
    try:
        stats = await rag_pipeline.get_collection_stats()
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "statistics": stats,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting statistics: {str(e)}"
        )


# Optional: Batch operations for efficiency
@router.post("/ingest/batch")
async def ingest_batch(
    documents: List[IngestTextRequest],
    background_tasks: BackgroundTasks,
    rag_pipeline: EnhancedRAGPipeline = Depends(get_rag_pipeline),
):
    """
    Batch ingest multiple documents (async processing)
    """
    if len(documents) > 100:
        raise HTTPException(
            status_code=400, detail="Batch size cannot exceed 100 documents"
        )

    # Process in background
    background_tasks.add_task(_process_batch_ingest, documents, rag_pipeline)

    return {
        "success": True,
        "message": f"Batch processing {len(documents)} documents",
        "batch_size": len(documents),
        "status": "processing",
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/ingest/url", response_model=IngestUrlResponse)
async def ingest_url(request: IngestUrlRequest):
    """
    Ingest content from URL into project-scoped RAG pipeline

    This endpoint extracts content from the provided URL using WebIntelligenceManager
    and stores it in the project-scoped RAG pipeline for the specified project.
    """
    start_time = datetime.now()

    try:
        # Initialize WebIntelligenceManager
        web_intelligence = WebIntelligenceManager()

        # Get project RAG pipeline
        project_rag = get_project_rag_pipeline()

        # Extract content based on extraction type
        if request.extraction_type == "quick":
            extraction_result = await web_intelligence.quick_scrape(request.url)
        else:
            # Use comprehensive extraction for both "comprehensive" and "specific"
            extraction_result = await web_intelligence.intelligent_web_extraction(
                url=request.url, extraction_type=request.extraction_type
            )

        # Prepare content for ingestion
        extracted_content = extraction_result.get("content", "")
        extracted_title = request.title or extraction_result.get(
            "title", "Extracted from URL"
        )
        extracted_metadata = request.metadata or {}

        # Add extraction metadata
        extracted_metadata.update(
            {
                "organization_id": request.organization_id,
                "project_id": str(request.project_id),
                "source_url": request.url,
                "extraction_type": request.extraction_type,
                "extraction_timestamp": start_time.isoformat(),
                "source_api": "v2_rag_ingest_url",
                **extraction_result.get("metadata", {}),
            }
        )

        # Store in project RAG pipeline
        document_id = await project_rag.store_web_document(
            url=request.url,
            content=extracted_content,
            title=extracted_title,
            project_id=request.project_id,
            tags=request.tags,
            metadata=extracted_metadata,
        )

        if not document_id:
            raise HTTPException(
                status_code=500,
                detail="Failed to store web document in project RAG pipeline",
            )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return IngestUrlResponse(
            success=True,
            document_ids=[document_id],
            url=request.url,
            extraction_metadata=extraction_result.get("metadata", {}),
            processing_time_ms=int(processing_time),
            timestamp=datetime.now(),
            project_id=request.project_id,
            organization_id=request.organization_id,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error ingesting URL {request.url}: {str(e)}"
        )


async def _process_batch_ingest(
    documents: List[IngestTextRequest], rag_pipeline: EnhancedRAGPipeline
):
    """Background task for batch processing"""
    for doc in documents:
        try:
            metadata = doc.metadata or {}
            metadata["organization_id"] = doc.organization_id
            metadata["ingested_at"] = datetime.now().isoformat()
            metadata["source_api"] = "v2_rag_batch_ingest"

            await rag_pipeline.add_document(
                content=doc.content,
                title=doc.title,
                source_type=doc.source_type,
                url=doc.url,
                author=doc.author,
                tags=doc.tags,
                metadata=metadata,
            )
        except Exception as e:
            # Log error but continue processing
            print(f"Error processing document {doc.title}: {e}")


@router.post("/chat", response_model=ChatResponse)
async def chat_with_project(
    request: ChatRequest,
    project_rag: Any = Depends(get_project_rag_pipeline),
    llm_manager: LLMManager = Depends(get_llm_manager),
    zep_memory: ZepMemoryManager = Depends(get_zep_memory),
):
    """
    Chat with project knowledge base using RAG + LLM synthesis

    This endpoint provides conversational access to project-specific knowledge:
    1. Retrieves relevant context from project RAG pipeline
    2. Manages conversation history via Zep Memory
    3. Synthesizes responses using LLM with project context
    4. Returns grounded answers with source attribution
    """
    start_time = datetime.now()

    try:
        # Generate or use existing session ID
        session_id = (
            request.session_id
            or f"project_{request.project_id}_{request.user_id}_{int(time.time())}"
        )

        # Get relevant project context via RAG
        rag_context = await project_rag.get_initial_context_for_query(
            project_id=request.project_id, problem_statement=request.user_question
        )

        context_chunks_retrieved = len(rag_context.get("documents", []))
        rag_context_used = context_chunks_retrieved > 0

        # Get conversation history from Zep Memory
        try:
            # Create session if it doesn't exist
            await zep_memory.create_session(
                user_id=request.user_id,
                session_id=session_id,
                metadata={
                    "project_id": str(request.project_id),
                    "organization_id": request.organization_id,
                    "session_type": "project_chat",
                },
            )

            # Get recent conversation history
            conversation_history = await zep_memory.get_session_messages(
                session_id=session_id, limit=10  # Last 10 messages for context
            )
        except Exception as memory_error:
            # Graceful degradation if Zep is unavailable
            logging.warning(f"Zep memory unavailable: {memory_error}")
            conversation_history = []

        # Build context-aware prompt
        context_section = ""
        sources = []

        if rag_context_used:
            context_docs = rag_context.get("documents", [])[
                : request.max_context_chunks
            ]
            context_section = "\n\nRELEVANT PROJECT CONTEXT:\n"

            for i, doc in enumerate(context_docs, 1):
                content = doc.get("content", "")[:1000]  # Limit content length
                title = doc.get("title", "Unknown Document")
                source_url = doc.get("metadata", {}).get("source_url", "")

                context_section += f"\n[{i}] {title}\n{content}\n"

                sources.append(
                    {
                        "title": title,
                        "content_preview": (
                            content[:200] + "..." if len(content) > 200 else content
                        ),
                        "source_url": source_url,
                        "document_id": doc.get("id", ""),
                        "relevance_score": doc.get("score", 0.0),
                    }
                )

        # Build conversation context
        conversation_context = ""
        if conversation_history:
            conversation_context = "\n\nRECENT CONVERSATION:\n"
            for msg in conversation_history[-5:]:  # Last 5 messages
                role = msg.get("role", "user")
                content = msg.get("content", "")[:300]  # Limit length
                conversation_context += f"{role.upper()}: {content}\n"

        # Create system prompt for project chat
        system_prompt = f"""You are an AI assistant helping users explore and understand their project knowledge base.

Project ID: {request.project_id}
User: {request.user_id}

Your role:
1. Answer questions using the provided project context when available
2. Maintain conversation continuity using recent chat history
3. Cite sources when using project information
4. Be helpful and concise
5. If no relevant context is found, say so clearly

When referencing project context, use format: "According to [Source Title]..."
Always prioritize accuracy over completeness."""

        # Build complete prompt
        complete_prompt = f"""Question: {request.user_question}
{context_section}
{conversation_context}

Please provide a helpful answer based on the above context and conversation history."""

        # Generate LLM response
        llm_response = await llm_manager.execute_completion(
            prompt=complete_prompt,
            system_prompt=system_prompt,
            temperature=request.temperature,
            max_tokens=1500,
            timeout=60,
        )

        if not llm_response.success:
            raise HTTPException(
                status_code=500, detail=f"LLM generation failed: {llm_response.error}"
            )

        answer = llm_response.content

        # Store conversation in Zep Memory
        try:
            # Store user message
            await zep_memory.add_message(
                session_id=session_id,
                message=ZepMessage(
                    content=request.user_question,
                    role="user",
                    metadata={
                        "project_id": str(request.project_id),
                        "rag_context_used": rag_context_used,
                        "context_chunks_retrieved": context_chunks_retrieved,
                    },
                ),
            )

            # Store assistant response
            await zep_memory.add_message(
                session_id=session_id,
                message=ZepMessage(
                    content=answer,
                    role="assistant",
                    metadata={
                        "project_id": str(request.project_id),
                        "sources_count": len(sources),
                        "llm_provider": llm_response.provider,
                        "processing_time_ms": (
                            datetime.now() - start_time
                        ).total_seconds()
                        * 1000,
                    },
                ),
            )

            # Get conversation turn count
            session_stats = await zep_memory.get_session_stats(session_id)
            conversation_turn = (
                session_stats.get("message_count", 0) // 2
            )  # Pairs of user/assistant

        except Exception as memory_error:
            logging.warning(f"Failed to store conversation in Zep: {memory_error}")
            conversation_turn = 1  # Default fallback

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return ChatResponse(
            success=True,
            answer=answer,
            sources=sources,
            session_id=session_id,
            conversation_turn=conversation_turn,
            processing_time_ms=int(processing_time),
            rag_context_used=rag_context_used,
            context_chunks_retrieved=context_chunks_retrieved,
            project_id=request.project_id,
            timestamp=datetime.now(),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing chat request: {str(e)}"
        )
