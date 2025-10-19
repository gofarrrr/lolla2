"""
Document Ingestion Service for METIS V2 - Operation "PDF Ingestion"
==================================================================

Handles PDF and DOCX document processing for project-based knowledge bootstrap.
Integrates with ProjectRAGPipeline to provide seamless document ingestion
capabilities for the V2 project workspace architecture.

Key Features:
- PDF and DOCX text extraction
- Document chunking and preprocessing
- Integration with project-scoped RAG pipeline
- Metadata preservation and tracking
- Error handling and validation
"""

import logging
import io
import tempfile
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path
from fastapi import UploadFile

# Document processing imports (with optional dependencies)
try:
    import PyPDF2

    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    PyPDF2 = None

try:
    from docx import Document as DocxDocument

    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    DocxDocument = None

# Core imports
from src.core.unified_context_stream import UnifiedContextStream, ContextEventType
from src.rag.project_rag_pipeline import ProjectRAGPipeline

logger = logging.getLogger(__name__)


class DocumentProcessingError(Exception):
    """Custom exception for document processing errors"""

    pass


class DocumentIngestionService:
    """
    Document ingestion service for PDF and DOCX files.
    Handles extraction, chunking, and integration with project RAG pipeline.
    """

    def __init__(
        self,
        project_rag_pipeline: Optional[ProjectRAGPipeline] = None,
        context_stream: Optional[UnifiedContextStream] = None,
    ):
        """
        Initialize document ingestion service

        Args:
            project_rag_pipeline: Project-specific RAG pipeline for storage
            context_stream: Context stream for logging
        """
        self.project_rag_pipeline = project_rag_pipeline
        self.context_stream = context_stream
        self.supported_types = [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ]

    async def ingest_document(
        self,
        file: UploadFile,
        project_id: str,
        analysis_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a document file into the project knowledge base

        Args:
            file: Uploaded file (PDF or DOCX)
            project_id: Project UUID
            analysis_id: Analysis ID for tracking
            metadata: Additional metadata

        Returns:
            Dictionary containing ingestion results
        """
        try:
            if self.context_stream:
                await self.context_stream.log_event(
                    ContextEventType.PROCESSING_STARTED,
                    {
                        "operation": "document_ingestion",
                        "filename": file.filename,
                        "content_type": file.content_type,
                        "size": file.size,
                        "project_id": project_id,
                        "analysis_id": analysis_id,
                    },
                )

            # Validate file type
            if file.content_type not in self.supported_types:
                raise DocumentProcessingError(
                    f"Unsupported file type: {file.content_type}"
                )

            # Extract text content
            text_content, document_metadata = await self._extract_text(file)

            if not text_content.strip():
                raise DocumentProcessingError("No text content found in document")

            # Chunk the document
            chunks = await self._chunk_document(text_content, file.filename)

            # Prepare document record for storage
            document_record = {
                "document_id": f"doc_{analysis_id}_{file.filename}",
                "project_id": project_id,
                "analysis_id": analysis_id,
                "filename": file.filename,
                "content_type": file.content_type,
                "size": file.size,
                "text_content": text_content,
                "chunk_count": len(chunks),
                "metadata": {
                    **(metadata or {}),
                    **document_metadata,
                    "ingested_at": datetime.utcnow().isoformat(),
                },
                "processing_status": "completed",
            }

            # Store in RAG pipeline if available (placeholder for Phase 4)
            if self.project_rag_pipeline:
                # Phase 4 will implement the actual storage integration
                logger.info(
                    f"📋 Document ready for RAG storage: {file.filename} ({len(chunks)} chunks)"
                )

            if self.context_stream:
                await self.context_stream.log_event(
                    ContextEventType.PROCESSING_COMPLETE,
                    {
                        "operation": "document_ingestion",
                        "filename": file.filename,
                        "chunks_created": len(chunks),
                        "text_length": len(text_content),
                        "document_id": document_record["document_id"],
                        "status": "success",
                    },
                )

            logger.info(
                f"✅ Document ingested successfully: {file.filename} -> {len(chunks)} chunks"
            )

            return {
                "status": "success",
                "document": document_record,
                "chunks": chunks[:5],  # Return first 5 chunks for preview
                "total_chunks": len(chunks),
                "text_preview": text_content[:500]
                + ("..." if len(text_content) > 500 else ""),
            }

        except Exception as e:
            error_msg = f"Document ingestion failed: {str(e)}"
            logger.error(f"❌ {error_msg}")

            if self.context_stream:
                await self.context_stream.log_event(
                    ContextEventType.ERROR_OCCURRED,
                    {
                        "operation": "document_ingestion",
                        "filename": file.filename if file else "unknown",
                        "error": error_msg,
                        "status": "failed",
                    },
                )

            raise DocumentProcessingError(error_msg) from e

    async def _extract_text(self, file: UploadFile) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text content from PDF or DOCX file

        Args:
            file: Uploaded file

        Returns:
            Tuple of (text_content, metadata)
        """
        try:
            # Read file content into memory
            content = await file.read()

            if file.content_type == "application/pdf":
                return await self._extract_pdf_text(content, file.filename)
            elif (
                file.content_type
                == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ):
                return await self._extract_docx_text(content, file.filename)
            else:
                raise DocumentProcessingError(
                    f"Unsupported file type: {file.content_type}"
                )

        except Exception as e:
            raise DocumentProcessingError(f"Text extraction failed: {str(e)}") from e

    async def _extract_pdf_text(
        self, content: bytes, filename: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Extract text from PDF content"""
        if not PYPDF2_AVAILABLE:
            raise DocumentProcessingError(
                "PyPDF2 is not installed - cannot process PDF files"
            )

        try:
            pdf_stream = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_stream)

            text_content = ""
            page_count = len(pdf_reader.pages)

            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                except Exception as e:
                    logger.warning(
                        f"⚠️ Failed to extract text from page {page_num + 1}: {e}"
                    )
                    continue

            # Extract metadata
            metadata = {
                "document_type": "pdf",
                "page_count": page_count,
                "extraction_method": "PyPDF2",
            }

            # Try to get PDF metadata
            try:
                if pdf_reader.metadata:
                    metadata.update(
                        {
                            "title": pdf_reader.metadata.get("/Title", ""),
                            "author": pdf_reader.metadata.get("/Author", ""),
                            "creator": pdf_reader.metadata.get("/Creator", ""),
                            "producer": pdf_reader.metadata.get("/Producer", ""),
                            "creation_date": str(
                                pdf_reader.metadata.get("/CreationDate", "")
                            ),
                            "modification_date": str(
                                pdf_reader.metadata.get("/ModDate", "")
                            ),
                        }
                    )
            except Exception as e:
                logger.debug(f"Could not extract PDF metadata: {e}")

            return text_content.strip(), metadata

        except Exception as e:
            raise DocumentProcessingError(f"PDF extraction failed: {str(e)}") from e

    async def _extract_docx_text(
        self, content: bytes, filename: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Extract text from DOCX content"""
        try:
            # Save content to temporary file for docx processing
            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name

            try:
                doc = DocxDocument(temp_file_path)

                text_content = ""
                paragraph_count = 0

                for paragraph in doc.paragraphs:
                    if paragraph.text.strip():
                        text_content += paragraph.text + "\n"
                        paragraph_count += 1

                # Extract metadata
                metadata = {
                    "document_type": "docx",
                    "paragraph_count": paragraph_count,
                    "extraction_method": "python-docx",
                }

                # Try to get document core properties
                try:
                    core_props = doc.core_properties
                    metadata.update(
                        {
                            "title": core_props.title or "",
                            "author": core_props.author or "",
                            "subject": core_props.subject or "",
                            "keywords": core_props.keywords or "",
                            "created": (
                                str(core_props.created) if core_props.created else ""
                            ),
                            "modified": (
                                str(core_props.modified) if core_props.modified else ""
                            ),
                        }
                    )
                except Exception as e:
                    logger.debug(f"Could not extract DOCX metadata: {e}")

                return text_content.strip(), metadata

            finally:
                # Clean up temporary file
                try:
                    Path(temp_file_path).unlink()
                except Exception:
                    pass

        except Exception as e:
            raise DocumentProcessingError(f"DOCX extraction failed: {str(e)}") from e

    async def _chunk_document(
        self, text: str, filename: str, chunk_size: int = 1000, overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Chunk document text for RAG processing

        Args:
            text: Full text content
            filename: Original filename
            chunk_size: Maximum characters per chunk
            overlap: Overlap between chunks

        Returns:
            List of chunk dictionaries
        """
        try:
            chunks = []

            # Simple sentence-aware chunking
            sentences = self._split_into_sentences(text)
            current_chunk = ""
            chunk_index = 0

            for sentence in sentences:
                # Check if adding this sentence would exceed chunk size
                if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                    # Save current chunk
                    chunks.append(
                        {
                            "chunk_id": f"{filename}_{chunk_index}",
                            "chunk_index": chunk_index,
                            "text": current_chunk.strip(),
                            "character_count": len(current_chunk),
                            "source_document": filename,
                            "chunk_type": "text",
                        }
                    )

                    # Start new chunk with overlap
                    overlap_text = (
                        current_chunk[-overlap:]
                        if len(current_chunk) > overlap
                        else current_chunk
                    )
                    current_chunk = overlap_text + " " + sentence
                    chunk_index += 1
                else:
                    current_chunk += " " + sentence if current_chunk else sentence

            # Add final chunk
            if current_chunk.strip():
                chunks.append(
                    {
                        "chunk_id": f"{filename}_{chunk_index}",
                        "chunk_index": chunk_index,
                        "text": current_chunk.strip(),
                        "character_count": len(current_chunk),
                        "source_document": filename,
                        "chunk_type": "text",
                    }
                )

            logger.info(
                f"📄 Chunked {filename}: {len(text)} chars -> {len(chunks)} chunks"
            )
            return chunks

        except Exception as e:
            raise DocumentProcessingError(f"Document chunking failed: {str(e)}") from e

    def _split_into_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting"""
        # Basic sentence splitting on periods, exclamation marks, question marks
        import re

        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    async def validate_file(self, file: UploadFile) -> Dict[str, Any]:
        """
        Validate uploaded file before processing

        Args:
            file: Uploaded file

        Returns:
            Validation result dictionary
        """
        validation_result = {"valid": True, "errors": [], "warnings": []}

        try:
            # Check file type
            if file.content_type not in self.supported_types:
                validation_result["valid"] = False
                validation_result["errors"].append(
                    f"Unsupported file type: {file.content_type}"
                )

            # Check file size (10MB limit)
            if file.size and file.size > 10 * 1024 * 1024:
                validation_result["valid"] = False
                validation_result["errors"].append("File size exceeds 10MB limit")

            # Check filename
            if not file.filename or not file.filename.strip():
                validation_result["valid"] = False
                validation_result["errors"].append("Invalid filename")

            # Add size warning for large files
            if file.size and file.size > 5 * 1024 * 1024:
                validation_result["warnings"].append(
                    "Large file detected - processing may take longer"
                )

            return validation_result

        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Validation error: {str(e)}")
            return validation_result


# Global instance for dependency injection
_document_ingestion_service: Optional[DocumentIngestionService] = None


def get_document_ingestion_service(
    project_rag_pipeline: Optional[ProjectRAGPipeline] = None,
    context_stream: Optional[UnifiedContextStream] = None,
) -> DocumentIngestionService:
    """Get or create global DocumentIngestionService instance"""
    global _document_ingestion_service

    if _document_ingestion_service is None:
        _document_ingestion_service = DocumentIngestionService(
            project_rag_pipeline=project_rag_pipeline, context_stream=context_stream
        )

    return _document_ingestion_service


def reset_document_ingestion_service() -> None:
    """Reset global DocumentIngestionService instance (primarily for testing)"""
    global _document_ingestion_service
    _document_ingestion_service = None
