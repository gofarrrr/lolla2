"""
Intelligent Retriever
====================

Advanced retrieval system with Milvus vector database integration.
Implements context-aware search with relevance filtering and reranking.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import uuid
import os

from ..core.unified_context_stream import get_unified_context_stream, ContextEventType
from .embeddings import VoyageEmbeddings

logger = logging.getLogger(__name__)


class RetrievalResult:
    """Container for retrieval results with metadata"""

    def __init__(
        self,
        content: str,
        source: str,
        score: float,
        metadata: Dict[str, Any] = None,
        doc_id: str = None,
    ):
        self.content = content
        self.source = source
        self.score = score
        self.metadata = metadata or {}
        self.doc_id = doc_id or str(uuid.uuid4())
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "source": self.source,
            "score": self.score,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class MilvusConnector:
    """
    Milvus vector database connector with simplified interface
    """

    def __init__(self, host: str = "localhost", port: int = 19530, db_path: str = None):
        """
        Initialize Milvus connector

        Args:
            host: Milvus server host
            port: Milvus server port
            db_path: Path for Milvus Lite (local file-based)
        """
        self.host = host
        self.port = port
        self.db_path = db_path or os.getenv("MILVUS_DB_PATH", "milvus_lite.db")

        self.client = None
        self.collection_name = "metis_knowledge"
        self.is_connected = False

        # Schema configuration
        self.schema_config = {
            "dimension": 1024,  # Voyage-3 embedding dimension
            "fields": [
                {"name": "id", "type": "VARCHAR", "max_length": 36, "is_primary": True},
                {"name": "vector", "type": "FLOAT_VECTOR", "dim": 1024},
                {"name": "content", "type": "VARCHAR", "max_length": 65535},
                {"name": "source", "type": "VARCHAR", "max_length": 255},
                {"name": "url", "type": "VARCHAR", "max_length": 1024},
                {"name": "title", "type": "VARCHAR", "max_length": 512},
                {"name": "scrape_date", "type": "INT64"},
                {"name": "provider", "type": "VARCHAR", "max_length": 50},
                {"name": "user_id", "type": "VARCHAR", "max_length": 36},
                {"name": "analysis_id", "type": "VARCHAR", "max_length": 36},
                {"name": "content_type", "type": "VARCHAR", "max_length": 50},
                {"name": "confidence_score", "type": "FLOAT"},
                {"name": "metadata", "type": "JSON"},
            ],
        }

    async def connect(self) -> bool:
        """Connect to Milvus database"""
        try:
            # Try to import pymilvus
            try:
                from pymilvus import (
                    MilvusClient,
                    DataType,
                    CollectionSchema,
                    FieldSchema,
                    IndexParams,
                )

                self.MilvusClient = MilvusClient
                self.DataType = DataType
                self.CollectionSchema = CollectionSchema
                self.FieldSchema = FieldSchema
                self.IndexParams = IndexParams
            except ImportError:
                logger.warning(
                    "⚠️ pymilvus not installed. Using mock connector for development"
                )
                self.client = MockMilvusClient()
                self.is_connected = True
                return True

            # Initialize client
            if self.db_path and self.host == "localhost":
                # Use Milvus Lite for local development
                self.client = self.MilvusClient(uri=self.db_path)
                logger.info(f"📁 Connected to Milvus Lite: {self.db_path}")
            else:
                # Use Milvus server
                self.client = self.MilvusClient(host=self.host, port=self.port)
                logger.info(f"🌐 Connected to Milvus server: {self.host}:{self.port}")

            # Initialize collection
            await self._ensure_collection_exists()

            self.is_connected = True
            return True

        except Exception as e:
            logger.error(f"❌ Failed to connect to Milvus: {e}")
            # Fall back to mock client for development
            self.client = MockMilvusClient()
            self.is_connected = True
            return False

    async def _ensure_collection_exists(self):
        """Ensure the knowledge collection exists"""
        try:
            # Check if collection exists
            if not self.client.has_collection(self.collection_name):
                logger.info(f"🏗️ Creating collection: {self.collection_name}")

                # Create fields
                fields = []
                for field_config in self.schema_config["fields"]:
                    if field_config["type"] == "VARCHAR":
                        field = self.FieldSchema(
                            name=field_config["name"],
                            dtype=self.DataType.VARCHAR,
                            max_length=field_config["max_length"],
                            is_primary=field_config.get("is_primary", False),
                        )
                    elif field_config["type"] == "FLOAT_VECTOR":
                        field = self.FieldSchema(
                            name=field_config["name"],
                            dtype=self.DataType.FLOAT_VECTOR,
                            dim=field_config["dim"],
                        )
                    elif field_config["type"] == "INT64":
                        field = self.FieldSchema(
                            name=field_config["name"], dtype=self.DataType.INT64
                        )
                    elif field_config["type"] == "FLOAT":
                        field = self.FieldSchema(
                            name=field_config["name"], dtype=self.DataType.FLOAT
                        )
                    elif field_config["type"] == "JSON":
                        field = self.FieldSchema(
                            name=field_config["name"], dtype=self.DataType.JSON
                        )

                    fields.append(field)

                # Create schema
                schema = self.CollectionSchema(
                    fields=fields, description="METIS 2.0 Knowledge Base"
                )

                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name, schema=schema
                )

                # Create index on vector field
                index_params = self.IndexParams()
                index_params.add_index(
                    field_name="vector",
                    index_type="IVF_FLAT",
                    metric_type="COSINE",
                    params={"nlist": 128},
                )

                self.client.create_index(
                    collection_name=self.collection_name, index_params=index_params
                )

                logger.info("✅ Collection created successfully")
            else:
                logger.info(f"📊 Collection {self.collection_name} already exists")

        except Exception as e:
            logger.error(f"❌ Failed to create collection: {e}")
            raise

    async def insert(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Insert documents into Milvus"""
        try:
            if not self.is_connected:
                await self.connect()

            # Insert documents
            result = self.client.insert(
                collection_name=self.collection_name, data=documents
            )

            logger.info(f"✅ Inserted {len(documents)} documents into Milvus")
            return [doc["id"] for doc in documents]

        except Exception as e:
            logger.error(f"❌ Failed to insert documents: {e}")
            raise

    async def search(
        self, query_vector: List[float], limit: int = 20, filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in Milvus"""
        try:
            if not self.is_connected:
                await self.connect()

            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

            # Build filter expression
            filter_expr = None
            if filters:
                filter_conditions = []
                for key, value in filters.items():
                    if isinstance(value, str):
                        filter_conditions.append(f"{key} == '{value}'")
                    else:
                        filter_conditions.append(f"{key} == {value}")

                if filter_conditions:
                    filter_expr = " and ".join(filter_conditions)

            # Perform search
                results = self.client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                anns_field="vector",
                search_params=search_params,
                limit=limit,
                expr=filter_expr,
                output_fields=[
                    "id",
                    "content",
                    "source",
                    "url",
                    "title",
                    "provider",
                    "content_type",
                    "confidence_score",
                    "metadata",
                    "scrape_date",
                ],
            )

            # Format results
            formatted_results = []
            if results:
                for hit in results[0]:
                    formatted_results.append(
                        {
                            "id": hit.get("id"),
                            "score": float(hit.get("distance", 0)),
                            "content": hit.get("content", ""),
                            "source": hit.get("source", ""),
                            "url": hit.get("url", ""),
                            "title": hit.get("title", ""),
                            "provider": hit.get("provider", ""),
                            "content_type": hit.get("content_type", ""),
                            "confidence_score": hit.get("confidence_score", 0.0),
                            "metadata": hit.get("metadata", {}),
                        }
                    )

            return formatted_results

        except Exception as e:
            logger.error(f"❌ Failed to search Milvus: {e}")
            return []


class MockMilvusClient:
    """Mock Milvus client for development"""

    def __init__(self):
        self._collections = {}
        self._data = {}
        logger.info("🎭 Using MockMilvusClient for development")

    def has_collection(self, collection_name: str) -> bool:
        return collection_name in self._collections

    def create_collection(self, collection_name: str, schema=None):
        self._collections[collection_name] = schema
        self._data[collection_name] = []

    def create_index(self, collection_name: str, field_name: str, index_params: Dict):
        pass  # Mock implementation

    def insert(self, collection_name: str, data: List[Dict[str, Any]]):
        if collection_name not in self._data:
            self._data[collection_name] = []
        self._data[collection_name].extend(data)
        return {"insert_count": len(data)}

    def search(self, collection_name: str, data: List[List[float]], **kwargs):
        # Mock search - return empty results
        return [[]]


class IntelligentRetriever:
    """
    Advanced retrieval system with context awareness and relevance filtering
    """

    def __init__(
        self,
        embeddings: VoyageEmbeddings,
        milvus_config: Dict[str, Any] = None,
        context_stream: Optional[Any] = None,
    ):
        """
        Initialize intelligent retriever

        Args:
            embeddings: Voyage embeddings client
            milvus_config: Milvus configuration
            context_stream: Context stream for logging
        """
        self.embeddings = embeddings
        self.context_stream = context_stream or get_unified_context_stream()

        # Initialize Milvus connector
        milvus_config = milvus_config or {}
        self.milvus = MilvusConnector(
            host=milvus_config.get("host", "localhost"),
            port=milvus_config.get("port", 19530),
            db_path=milvus_config.get("db_path"),
        )

        # Retrieval settings
        self.default_top_k = 20
        self.similarity_threshold = 0.7
        self.max_age_days = 30

        # Statistics
        self.stats = {
            "total_searches": 0,
            "total_documents_retrieved": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        logger.info("🧠 IntelligentRetriever initialized")

    async def initialize(self) -> bool:
        """Initialize retriever and connect to Milvus"""
        success = await self.milvus.connect()
        if success:
            logger.info("✅ IntelligentRetriever ready")
        else:
            logger.warning("⚠️ IntelligentRetriever using mock backend")
        return success

    async def store_document(
        self,
        content: str,
        source: str,
        metadata: Dict[str, Any] = None,
        user_id: str = None,
        analysis_id: str = None,
    ) -> str:
        """
        Store a document in the knowledge base

        Args:
            content: Document content
            source: Source identifier
            metadata: Additional metadata
            user_id: User who added the document
            analysis_id: Associated analysis ID

        Returns:
            Document ID
        """
        doc_id = str(uuid.uuid4())
        metadata = metadata or {}

        # Log storage attempt
        self.context_stream.add_event(
            ContextEventType.RAG_DOCUMENT_STORED,
            {
                "doc_id": doc_id,
                "source": source,
                "content_length": len(content),
                "user_id": user_id,
                "analysis_id": analysis_id,
            },
        )

        try:
            # Generate embedding
            embedding = await self.embeddings.embed_documents([content])

            # Prepare document for storage
            document = {
                "id": doc_id,
                "vector": embedding[0],
                "content": content[:65535],  # Truncate if needed
                "source": source,
                "url": metadata.get("url", ""),
                "title": metadata.get("title", ""),
                "scrape_date": int(datetime.now().timestamp()),
                "provider": metadata.get("provider", "manual"),
                "user_id": user_id or "",
                "analysis_id": analysis_id or "",
                "content_type": metadata.get("content_type", "text"),
                "confidence_score": metadata.get("confidence", 1.0),
                "metadata": metadata,
            }

            # Store in Milvus
            await self.milvus.insert([document])

            logger.info(f"📝 Stored document: {doc_id} ({len(content)} chars)")
            return doc_id

        except Exception as e:
            logger.error(f"❌ Failed to store document: {e}")
            raise

    async def retrieve(
        self,
        query: str,
        top_k: int = None,
        user_id: str = None,
        filters: Dict[str, Any] = None,
        min_similarity: float = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query

        Args:
            query: Search query
            top_k: Number of results to return
            user_id: User ID for personalization
            filters: Additional filters
            min_similarity: Minimum similarity threshold

        Returns:
            List of retrieval results
        """
        top_k = top_k or self.default_top_k
        min_similarity = min_similarity or self.similarity_threshold
        filters = filters or {}

        self.stats["total_searches"] += 1

        # Log search request
        self.context_stream.add_event(
            ContextEventType.RAG_SEARCH_REQUEST,
            {
                "query": query,
                "top_k": top_k,
                "user_id": user_id,
                "filters": filters,
                "min_similarity": min_similarity,
            },
        )

        start_time = datetime.now()

        try:
            # Generate query embedding
            query_embedding = await self.embeddings.embed_query(query)

            # Add user filter if provided
            if user_id:
                filters["user_id"] = user_id

            # Search in Milvus
            results = await self.milvus.search(
                query_vector=query_embedding,
                limit=top_k * 2,  # Get extra results for filtering
                filters=filters,
            )

            # Convert to RetrievalResult objects and filter by similarity
            retrieval_results = []
            import os
            from math import pow
            half_life_days = float(os.getenv("RETRIEVAL_DECAY_HALF_LIFE_DAYS", "30"))
            now_ts = datetime.now().timestamp()
            for result in results:
                sim = float(result["score"]) if isinstance(result.get("score"), (int, float)) else 0.0
                if sim >= min_similarity:
                    # Time decay weight
                    scrape_ts = float(result.get("scrape_date") or 0)
                    age_days = max(0.0, (now_ts - scrape_ts) / 86400.0) if scrape_ts else 0.0
                    decay = pow(0.5, age_days / half_life_days) if half_life_days > 0 and age_days > 0 else 1.0
                    combined = sim * decay
                    retrieval_result = RetrievalResult(
                        content=result.get("content", ""),
                        source=result.get("source", ""),
                        score=combined,
                        metadata={
                            "url": result.get("url", ""),
                            "title": result.get("title", ""),
                            "provider": result.get("provider", ""),
                            "content_type": result.get("content_type", ""),
                            "similarity": sim,
                            "time_decay": decay,
                            "confidence_score": result.get("confidence_score", 0.0),
                            **result.get("metadata", {}),
                        },
                        doc_id=result.get("id"),
                    )
                    retrieval_results.append(retrieval_result)

            # Sort by combined score descending
            retrieval_results.sort(key=lambda r: r.score, reverse=True)

            # Limit to requested number of results
            retrieval_results = retrieval_results[:top_k]

            # Update statistics
            self.stats["total_documents_retrieved"] += len(retrieval_results)

            processing_time = (datetime.now() - start_time).total_seconds()

            # Log search response
            self.context_stream.add_event(
                ContextEventType.RAG_SEARCH_RESPONSE,
                {
                    "query": query,
                    "results_count": len(retrieval_results),
                    "processing_time_seconds": processing_time,
                    "top_score": (
                        retrieval_results[0].score if retrieval_results else 0.0
                    ),
                    "avg_score": (
                        sum(r.score for r in retrieval_results) / len(retrieval_results)
                        if retrieval_results
                        else 0.0
                    ),
                },
            )

            logger.info(
                f"🔍 Retrieved {len(retrieval_results)} documents for query "
                f"(top score: {retrieval_results[0].score:.3f})"
                if retrieval_results
                else "🔍 No documents retrieved"
            )

            return retrieval_results

        except Exception as e:
            logger.error(f"❌ Retrieval failed: {e}")
            self.context_stream.add_event(
                ContextEventType.RAG_SEARCH_RESPONSE,
                {"query": query, "status": "error", "error": str(e)},
            )
            return []

    async def batch_store_documents(
        self,
        documents: List[Dict[str, Any]],
        user_id: str = None,
        analysis_id: str = None,
    ) -> List[str]:
        """
        Store multiple documents efficiently

        Args:
            documents: List of documents with 'content', 'source', 'metadata'
            user_id: User who added the documents
            analysis_id: Associated analysis ID

        Returns:
            List of document IDs
        """
        if not documents:
            return []

        logger.info(f"📚 Batch storing {len(documents)} documents")

        # Generate embeddings in batches
        contents = [doc["content"] for doc in documents]
        from .embeddings import EmbeddingBatcher

        batcher = EmbeddingBatcher(self.embeddings, batch_size=50)
        embeddings = await batcher.embed_in_batches(contents, input_type="document")

        # Prepare documents for storage
        doc_ids = []
        milvus_documents = []

        for i, doc in enumerate(documents):
            doc_id = str(uuid.uuid4())
            doc_ids.append(doc_id)

            metadata = doc.get("metadata", {})

            milvus_doc = {
                "id": doc_id,
                "vector": embeddings[i],
                "content": doc["content"][:65535],
                "source": doc["source"],
                "url": metadata.get("url", ""),
                "title": metadata.get("title", ""),
                "scrape_date": int(datetime.now().timestamp()),
                "provider": metadata.get("provider", "batch"),
                "user_id": user_id or "",
                "analysis_id": analysis_id or "",
                "content_type": metadata.get("content_type", "text"),
                "confidence_score": metadata.get("confidence", 1.0),
                "metadata": metadata,
            }
            milvus_documents.append(milvus_doc)

        # Store in Milvus
        await self.milvus.insert(milvus_documents)

        logger.info(f"✅ Batch stored {len(doc_ids)} documents")
        return doc_ids

    async def cleanup_old_documents(self, max_age_days: int = None) -> int:
        """
        Clean up old documents (placeholder - would require delete operations)

        Args:
            max_age_days: Maximum age in days

        Returns:
            Number of documents cleaned up
        """
        max_age_days = max_age_days or self.max_age_days
        cutoff_timestamp = int(
            (datetime.now() - timedelta(days=max_age_days)).timestamp()
        )

        logger.info(
            f"🧹 Cleanup placeholder: would remove documents older than {max_age_days} days"
        )
        # Note: Would implement actual cleanup when Milvus delete operations are needed
        return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics"""
        return {
            **self.stats,
            "avg_documents_per_search": (
                self.stats["total_documents_retrieved"] / self.stats["total_searches"]
                if self.stats["total_searches"] > 0
                else 0.0
            ),
            "milvus_connected": self.milvus.is_connected,
            "collection_name": self.milvus.collection_name,
        }
