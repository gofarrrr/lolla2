"""
Document Upload API Router for METIS Lolla Platform
===================================================

API endpoints for document (PDF/DOCX) upload and ingestion.
Provides context enhancement for strategic analyses.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import uuid

from src.services.document_ingestion_service import (
    DocumentIngestionService,
    DocumentProcessingError,
)
from src.core.unified_context_stream import UnifiedContextStream

logger = logging.getLogger(__name__)

# Router setup
router = APIRouter(prefix="/api/documents", tags=["documents"])


# Response Models
class DocumentUploadResponse(BaseModel):
    """Response model for successful document upload"""

    document_id: str
    filename: str
    content_type: str
    size: int
    text_content_length: int
    chunk_count: int
    metadata: Dict[str, Any]
    processing_status: str
    uploaded_at: str


class DocumentListResponse(BaseModel):
    """Response model for listing uploaded documents"""

    documents: List[DocumentUploadResponse]
    total: int
    trace_id: str


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    trace_id: str = Query(..., description="Trace ID for the analysis"),
    analysis_id: Optional[str] = Query(
        None, description="Optional analysis ID (defaults to trace_id)"
    ),
):
    """
    Upload a PDF or DOCX document for analysis context enhancement.

    Supports:
    - PDF files (application/pdf)
    - DOCX files (application/vnd.openxmlformats-officedocument.wordprocessingml.document)

    Max file size: 10MB
    Max files per analysis: 5

    The document will be:
    1. Text extracted
    2. Chunked for processing
    3. Associated with the specified trace_id
    4. Available for RAG integration (Phase 4)
    """
    try:
        logger.info(f"📤 Document upload requested - File: {file.filename}, Trace: {trace_id}")

        # Validate file size (10MB limit)
        if file.size and file.size > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400, detail="File size exceeds 10MB limit"
            )

        # Initialize document ingestion service
        context_stream = UnifiedContextStream(trace_id=trace_id)
        service = DocumentIngestionService(
            project_rag_pipeline=None,  # Phase 4 will integrate RAG
            context_stream=context_stream,
        )

        # Ingest the document
        result = await service.ingest_document(
            file=file,
            project_id=trace_id,  # Using trace_id as project_id for now
            analysis_id=analysis_id or trace_id,
            metadata={
                "uploaded_via": "api",
                "upload_timestamp": datetime.utcnow().isoformat(),
            },
        )

        logger.info(
            f"✅ Document uploaded successfully - ID: {result['document_id']}, Chunks: {result['chunk_count']}"
        )

        return DocumentUploadResponse(
            document_id=result["document_id"],
            filename=result["filename"],
            content_type=result["content_type"],
            size=result["size"],
            text_content_length=len(result["text_content"]),
            chunk_count=result["chunk_count"],
            metadata=result["metadata"],
            processing_status=result["processing_status"],
            uploaded_at=result["metadata"]["ingested_at"],
        )

    except DocumentProcessingError as e:
        logger.error(f"❌ Document processing error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Unexpected error during document upload: {e}")
        raise HTTPException(
            status_code=500, detail=f"Document upload failed: {str(e)}"
        )


@router.get("/{trace_id}/list", response_model=DocumentListResponse)
async def list_documents(trace_id: str):
    """
    List all documents uploaded for a specific trace_id/analysis.

    Note: In the current implementation (Phase 3), documents are stored in-memory.
    Phase 4 will integrate with persistent storage and RAG pipeline.
    """
    try:
        # Placeholder: Phase 4 will retrieve from database/storage
        # For now, return empty list with trace_id confirmation
        logger.info(f"📋 Listing documents for trace: {trace_id}")

        return DocumentListResponse(documents=[], total=0, trace_id=trace_id)

    except Exception as e:
        logger.error(f"❌ Error listing documents: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list documents: {str(e)}"
        )


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a specific document.

    Note: Phase 4 implementation will handle persistent storage deletion.
    """
    try:
        logger.info(f"🗑️ Document deletion requested - ID: {document_id}")

        # Placeholder: Phase 4 will implement actual deletion
        return {
            "message": f"Document {document_id} deletion queued",
            "status": "pending",
        }

    except Exception as e:
        logger.error(f"❌ Error deleting document: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete document: {str(e)}"
        )


@router.get("/health")
async def documents_health():
    """
    Health check endpoint for document upload service
    """
    return {
        "status": "healthy",
        "service": "Document Upload & Ingestion",
        "supported_formats": ["PDF", "DOCX"],
        "max_file_size_mb": 10,
        "max_files_per_analysis": 5,
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
    }
