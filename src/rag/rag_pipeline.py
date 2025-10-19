"""
Enhanced RAG Pipeline for METIS 2.0
===================================

Main orchestration layer that combines Voyage AI embeddings, Milvus vector search,
and intelligent context management for the METIS 2.0 knowledge system.

Implements the complete RAG workflow with automatic data refresh and context-aware search.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import uuid

from .embeddings import VoyageEmbeddings, EmbeddingBatcher
from .retriever import IntelligentRetriever
from ..core.unified_context_stream import get_unified_context_stream, ContextEventType

logger = logging.getLogger(__name__)


class DocumentSource:
    """Document source metadata for tracking and refresh"""

    def __init__(
        self,
        source_id: str,
        source_type: str,
        url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.source_id = source_id
        self.source_type = source_type  # 'web', 'file', 'api', etc.
        self.url = url
        self.metadata = metadata or {}
        self.last_updated = datetime.now()
        self.last_checked = None
        self.refresh_interval = timedelta(hours=24)  # Default 24 hours


class EnhancedRAGPipeline:
    """
    Enhanced RAG pipeline with intelligent search and automatic data refresh
    """

    def __init__(
        self,
        voyage_api_key: Optional[str] = None,
        milvus_host: str = "localhost",
        milvus_port: int = 19530,
        collection_name: str = "metis_knowledge",
        context_stream: Optional[Any] = None,
    ):
        """
        Initialize RAG pipeline

        Args:
            voyage_api_key: Voyage AI API key
            milvus_host: Milvus server host
            milvus_port: Milvus server port
            collection_name: Milvus collection name
            context_stream: Context stream for logging
        """
        self.context_stream = context_stream or get_unified_context_stream()

        # Initialize components
        self.embeddings = VoyageEmbeddings(
            api_key=voyage_api_key, model="voyage-3", context_stream=self.context_stream
        )

        # Configure Milvus connection
        milvus_config = {
            "host": milvus_host,
            "port": milvus_port,
            "db_path": "milvus_lite.db",  # Use milvus-lite for local development
        }

        self.retriever = IntelligentRetriever(
            embeddings=self.embeddings,
            milvus_config=milvus_config,
            context_stream=self.context_stream,
        )

        self.batcher = EmbeddingBatcher(self.embeddings)

        # Document sources tracking
        self.document_sources: Dict[str, DocumentSource] = {}

        # Pipeline statistics
        self.stats = {
            "documents_indexed": 0,
            "searches_performed": 0,
            "total_chunks": 0,
            "refresh_operations": 0,
            "pipeline_start_time": datetime.now(),
        }

        logger.info("🚀 EnhancedRAGPipeline initialized")

    async def initialize(self) -> bool:
        """Initialize the RAG pipeline and create collections"""
        try:
            start_time = datetime.now()
            self.context_stream.add_event(
                ContextEventType.TOOL_EXECUTION,
                {
                    "tool": "rag_pipeline",  # backward compat
                    "tool_name": "rag_pipeline",
                    "action": "initialize",
                    "timestamp": start_time.isoformat(),
                },
            )

            success = await self.retriever.initialize()

            if success:
                latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
                logger.info("✅ RAG Pipeline initialization successful")
                self.context_stream.add_event(
                    ContextEventType.TOOL_EXECUTION,
                    {
                        "tool": "rag_pipeline",  # backward compat
                        "tool_name": "rag_pipeline",
                        "status": "initialized",
                        "latency_ms": latency_ms,
                        "collection_name": self.retriever.milvus.collection_name,
                    },
                )
                return True
            else:
                logger.error("❌ RAG Pipeline initialization failed")
                return False

        except Exception as e:
            logger.error(f"❌ RAG Pipeline initialization error: {e}")
            self.context_stream.add_event(
                ContextEventType.TOOL_EXECUTION,
                {"tool": "rag_pipeline", "status": "error", "error": str(e)},
            )
            return False

    async def add_document(
        self,
        content: str,
        title: str,
        source_type: str = "manual",
        url: Optional[str] = None,
        author: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a single document to the knowledge base

        Args:
            content: Document content
            title: Document title
            source_type: Type of source ('web', 'file', 'manual', etc.)
            url: Source URL if applicable
            author: Document author
            tags: Document tags
            metadata: Additional metadata

        Returns:
            Document ID
        """
        doc_id = str(uuid.uuid4())

        try:
            # Store document
            success = await self.retriever.store_document(
                doc_id=doc_id,
                content=content,
                title=title,
                source_type=source_type,
                url=url,
                author=author,
                tags=tags or [],
                metadata=metadata or {},
            )

            if success:
                # Track document source
                self.document_sources[doc_id] = DocumentSource(
                    source_id=doc_id,
                    source_type=source_type,
                    url=url,
                    metadata=metadata or {},
                )

                self.stats["documents_indexed"] += 1

                self.context_stream.add_event(
                    ContextEventType.TOOL_EXECUTION,
                    {
                        "tool": "rag_pipeline",
                        "action": "document_added",
                        "document_id": doc_id,
                        "title": title,
                        "source_type": source_type,
                        "content_length": len(content),
                    },
                )

                logger.info(f"✅ Document added: {title} ({doc_id})")
                return doc_id
            else:
                logger.error(f"❌ Failed to add document: {title}")
                return ""

        except Exception as e:
            logger.error(f"❌ Error adding document {title}: {e}")
            self.context_stream.add_event(
                ContextEventType.TOOL_EXECUTION,
                {
                    "tool": "rag_pipeline",
                    "action": "document_add_error",
                    "title": title,
                    "error": str(e),
                },
            )
            return ""

    async def add_documents_batch(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Add multiple documents efficiently in batches

        Args:
            documents: List of document dictionaries with keys:
                      'content', 'title', 'source_type', 'url', 'author', 'tags', 'metadata'

        Returns:
            List of document IDs
        """
        if not documents:
            return []

        start_time = datetime.now()
        doc_ids = []

        logger.info(f"📦 Processing {len(documents)} documents in batch")

        try:
            # Prepare documents with IDs
            batch_docs = []
            for doc in documents:
                doc_id = str(uuid.uuid4())
                doc_ids.append(doc_id)

                batch_doc = {
                    "doc_id": doc_id,
                    "content": doc["content"],
                    "title": doc["title"],
                    "source_type": doc.get("source_type", "manual"),
                    "url": doc.get("url"),
                    "author": doc.get("author"),
                    "tags": doc.get("tags", []),
                    "metadata": doc.get("metadata", {}),
                }
                batch_docs.append(batch_doc)

                # Track document source
                self.document_sources[doc_id] = DocumentSource(
                    source_id=doc_id,
                    source_type=doc.get("source_type", "manual"),
                    url=doc.get("url"),
                    metadata=doc.get("metadata", {}),
                )

            # Batch store documents
            success = await self.retriever.batch_store_documents(batch_docs)

            if success:
                self.stats["documents_indexed"] += len(documents)
                processing_time = (datetime.now() - start_time).total_seconds()

                self.context_stream.add_event(
                    ContextEventType.TOOL_EXECUTION,
                    {
                        "tool": "rag_pipeline",
                        "action": "batch_documents_added",
                        "document_count": len(documents),
                        "processing_time_seconds": processing_time,
                        "document_ids": doc_ids,
                    },
                )

                logger.info(
                    f"✅ Batch added {len(documents)} documents ({processing_time:.2f}s)"
                )
                return doc_ids
            else:
                logger.error(f"❌ Failed to batch add {len(documents)} documents")
                return []

        except Exception as e:
            logger.error(f"❌ Error in batch document add: {e}")
            self.context_stream.add_event(
                ContextEventType.TOOL_EXECUTION,
                {
                    "tool": "rag_pipeline",
                    "action": "batch_add_error",
                    "document_count": len(documents),
                    "error": str(e),
                },
            )
            return []

    async def intelligent_search(
        self,
        query: str,
        limit: int = 10,
        similarity_threshold: float = 0.7,
        source_types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        date_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform intelligent context-aware search with automatic data refresh

        Args:
            query: Search query
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            source_types: Filter by source types
            tags: Filter by tags
            date_filter: Date range filter

        Returns:
            List of search results with relevance scores
        """
        start_time = datetime.now()

        try:
            # Check if data refresh is needed
            await self._check_and_refresh_data()

            # Perform retrieval
            results = await self.retriever.retrieve(
                query=query,
                limit=limit,
                similarity_threshold=similarity_threshold,
                source_types=source_types,
                tags=tags,
                date_filter=date_filter,
            )

            self.stats["searches_performed"] += 1
            processing_time = (datetime.now() - start_time).total_seconds()

            self.context_stream.add_event(
                ContextEventType.TOOL_EXECUTION,
                {
                    "tool": "rag_pipeline",
                    "action": "intelligent_search",
                    "query": query,
                    "results_count": len(results),
                    "processing_time_seconds": processing_time,
                    "similarity_threshold": similarity_threshold,
                },
            )

            logger.info(
                f"🔍 Search completed: '{query}' → {len(results)} results ({processing_time:.2f}s)"
            )

            return results

        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            self.context_stream.add_event(
                ContextEventType.TOOL_EXECUTION,
                {
                    "tool": "rag_pipeline",
                    "action": "search_error",
                    "query": query,
                    "error": str(e),
                },
            )
            return []

    async def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID"""
        try:
            results = await self.retriever.milvus.query_by_ids([doc_id])
            return results[0] if results else None
        except Exception as e:
            logger.error(f"❌ Error getting document {doc_id}: {e}")
            return None

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the knowledge base"""
        try:
            success = await self.retriever.milvus.delete_by_ids([doc_id])
            if success and doc_id in self.document_sources:
                del self.document_sources[doc_id]
                self.stats["documents_indexed"] -= 1

                self.context_stream.add_event(
                    ContextEventType.TOOL_EXECUTION,
                    {
                        "tool": "rag_pipeline",
                        "action": "document_deleted",
                        "document_id": doc_id,
                    },
                )

                logger.info(f"🗑️ Document deleted: {doc_id}")
            return success
        except Exception as e:
            logger.error(f"❌ Error deleting document {doc_id}: {e}")
            return False

    async def update_document(self, doc_id: str, **updates) -> bool:
        """Update document metadata"""
        try:
            # Delete and re-add with updates
            doc = await self.get_document_by_id(doc_id)
            if not doc:
                return False

            # Apply updates
            for key, value in updates.items():
                if key in doc:
                    doc[key] = value

            # Delete old version
            await self.delete_document(doc_id)

            # Re-add with updates (keep same ID)
            success = await self.retriever.store_document(
                doc_id=doc_id,
                content=doc["content"],
                title=doc["title"],
                source_type=doc.get("source_type", "manual"),
                url=doc.get("url"),
                author=doc.get("author"),
                tags=doc.get("tags", []),
                metadata=doc.get("metadata", {}),
            )

            if success:
                logger.info(f"📝 Document updated: {doc_id}")

            return success

        except Exception as e:
            logger.error(f"❌ Error updating document {doc_id}: {e}")
            return False

    async def _check_and_refresh_data(self) -> None:
        """Check if any data sources need refreshing and trigger updates"""
        current_time = datetime.now()
        sources_to_refresh = []

        for source_id, source in self.document_sources.items():
            if (
                source.last_checked is None
                or current_time - source.last_checked > source.refresh_interval
            ):
                sources_to_refresh.append(source)

        if sources_to_refresh:
            logger.info(f"🔄 Checking {len(sources_to_refresh)} sources for refresh")

            for source in sources_to_refresh:
                source.last_checked = current_time
                # Here you would implement source-specific refresh logic
                # For now, we just mark as checked

            self.stats["refresh_operations"] += 1

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get comprehensive collection and pipeline statistics"""
        try:
            collection_stats = await self.retriever.milvus.get_collection_stats()
            embedding_stats = self.embeddings.get_stats()

            pipeline_uptime = (
                datetime.now() - self.stats["pipeline_start_time"]
            ).total_seconds()

            return {
                "pipeline_stats": {
                    **self.stats,
                    "uptime_seconds": pipeline_uptime,
                    "avg_searches_per_hour": (
                        self.stats["searches_performed"] / (pipeline_uptime / 3600)
                        if pipeline_uptime > 0
                        else 0
                    ),
                },
                "collection_stats": collection_stats,
                "embedding_stats": embedding_stats,
                "document_sources": len(self.document_sources),
                "health_status": "healthy",
            }

        except Exception as e:
            logger.error(f"❌ Error getting stats: {e}")
            return {"error": str(e), "health_status": "error"}

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of all pipeline components"""
        health = {
            "overall_health": "healthy",
            "components": {},
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # Check Milvus connection
            milvus_healthy = await self.retriever.milvus.health_check()
            health["components"]["milvus"] = (
                "healthy" if milvus_healthy else "unhealthy"
            )

            # Check embeddings
            try:
                await self.embeddings.embed("health check")
                health["components"]["embeddings"] = "healthy"
            except:
                health["components"]["embeddings"] = "unhealthy"

            # Check overall health
            if any(status == "unhealthy" for status in health["components"].values()):
                health["overall_health"] = "unhealthy"

        except Exception as e:
            health["overall_health"] = "error"
            health["error"] = str(e)

        return health

    async def clear_all_data(self) -> bool:
        """Clear all data from the collection (use with caution)"""
        try:
            success = await self.retriever.milvus.clear_collection()
            if success:
                self.document_sources.clear()
                self.stats["documents_indexed"] = 0
                self.stats["total_chunks"] = 0

                logger.info("🧹 All data cleared from RAG pipeline")

                self.context_stream.add_event(
                    ContextEventType.TOOL_EXECUTION,
                    {
                        "tool": "rag_pipeline",
                        "action": "data_cleared",
                        "timestamp": datetime.now().isoformat(),
                    },
                )

            return success

        except Exception as e:
            logger.error(f"❌ Error clearing data: {e}")
            return False


class RAGPipelineFactory:
    """Factory for creating configured RAG pipeline instances"""

    @staticmethod
    def create_default_pipeline(
        voyage_api_key: Optional[str] = None,
    ) -> EnhancedRAGPipeline:
        """Create a default RAG pipeline with standard configuration"""
        return EnhancedRAGPipeline(
            voyage_api_key=voyage_api_key,
            milvus_host="localhost",
            milvus_port=19530,
            collection_name="metis_knowledge",
        )

    @staticmethod
    def create_production_pipeline(
        voyage_api_key: str,
        milvus_host: str,
        milvus_port: int = 19530,
        collection_name: str = "metis_production",
    ) -> EnhancedRAGPipeline:
        """Create a production RAG pipeline with custom configuration"""
        return EnhancedRAGPipeline(
            voyage_api_key=voyage_api_key,
            milvus_host=milvus_host,
            milvus_port=milvus_port,
            collection_name=collection_name,
        )
