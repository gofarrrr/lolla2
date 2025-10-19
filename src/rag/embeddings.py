"""
Voyage AI Embeddings Integration
================================

Advanced embedding generation using Voyage AI's context-aware models.
Provides high-quality embeddings for METIS 2.0 RAG pipeline.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import hashlib
import os

import aiohttp
from ..core.unified_context_stream import get_unified_context_stream, ContextEventType, UnifiedContextStream

logger = logging.getLogger(__name__)


class VoyageEmbeddings:
    """
    Voyage AI embeddings client with caching and optimization
    """

    API_BASE = "https://api.voyageai.com/v1"

    # Model configurations
    MODELS = {
        "voyage-3": {
            "dimension": 1024,
            "context_length": 32000,
            "cost_per_token": 0.00002,
            "best_for": ["general", "multilingual", "code"],
        },
        "voyage-3-lite": {
            "dimension": 512,
            "context_length": 32000,
            "cost_per_token": 0.00001,
            "best_for": ["speed", "cost_optimization"],
        },
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "voyage-3",
        context_stream: Optional[UnifiedContextStream] = None,
    ):
        """
        Initialize Voyage embeddings client

        Args:
            api_key: Voyage API key (defaults to VOYAGE_API_KEY env var)
            model: Model to use for embeddings
            context_stream: Context stream for logging
        """
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Voyage API key required (VOYAGE_API_KEY env var or api_key param)"
            )

        self.model = model
        if model not in self.MODELS:
            raise ValueError(
                f"Unsupported model: {model}. Available: {list(self.MODELS.keys())}"
            )

        self.context_stream = context_stream or get_unified_context_stream()

        # Caching
        self._cache = {}
        self._cache_max_size = 10000

        # Statistics
        self.stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        logger.info(f"🚀 VoyageEmbeddings initialized with model {model}")

    @property
    def dimension(self) -> int:
        """Get embedding dimension for current model"""
        return self.MODELS[self.model]["dimension"]

    async def embed(
        self,
        texts: Union[str, List[str]],
        input_type: str = "document",
        truncate: bool = True,
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text(s)

        Args:
            texts: Single text or list of texts
            input_type: Type of input ('document' or 'query')
            truncate: Whether to truncate long texts

        Returns:
            Single embedding or list of embeddings
        """
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]

        # Check cache first
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text, input_type)
            if cache_key in self._cache:
                embeddings.append(self._cache[cache_key])
                self.stats["cache_hits"] += 1
            else:
                embeddings.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)
                self.stats["cache_misses"] += 1

        # Generate embeddings for uncached texts
        if uncached_texts:
            new_embeddings = await self._generate_embeddings(
                uncached_texts, input_type, truncate
            )

            # Store in cache and fill results
            for i, embedding in enumerate(new_embeddings):
                original_idx = uncached_indices[i]
                text = uncached_texts[i]
                cache_key = self._get_cache_key(text, input_type)

                self._cache[cache_key] = embedding
                embeddings[original_idx] = embedding

                # Manage cache size
                if len(self._cache) > self._cache_max_size:
                    # Remove oldest entry (simple FIFO)
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]

        return embeddings[0] if is_single else embeddings

    async def _generate_embeddings(
        self, texts: List[str], input_type: str, truncate: bool
    ) -> List[List[float]]:
        """Generate embeddings via Voyage API"""

        # Log request
        await self.context_stream.add_event(
            ContextEventType.TOOL_EXECUTION,
            {
                "tool": "voyage_embeddings",
                "action": "generate_embeddings",
                "model": self.model,
                "input_type": input_type,
                "text_count": len(texts),
                "total_chars": sum(len(t) for t in texts),
            },
        )

        start_time = datetime.now()

        try:
            payload = {"input": texts, "model": self.model, "input_type": input_type}

            if truncate:
                payload["truncation"] = True

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "METIS-2.0-RAG-Pipeline",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.API_BASE}/embeddings", json=payload, headers=headers
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(
                            f"Voyage API error: {response.status} - {error_text}"
                        )

                    result = await response.json()

            embeddings = [item["embedding"] for item in result["data"]]
            tokens_used = result.get("usage", {}).get("total_tokens", 0)

            # Update stats
            self.stats["total_requests"] += 1
            self.stats["total_tokens"] += tokens_used
            cost = tokens_used * self.MODELS[self.model]["cost_per_token"]
            self.stats["total_cost"] += cost

            processing_time = (datetime.now() - start_time).total_seconds()

            # Log success
            await self.context_stream.add_event(
                ContextEventType.TOOL_EXECUTION,
                {
                    "tool": "voyage_embeddings",
                    "status": "success",
                    "tokens_used": tokens_used,
                    "cost_usd": cost,
                    "processing_time_seconds": processing_time,
                    "embeddings_generated": len(embeddings),
                },
            )

            logger.info(
                f"✅ Generated {len(embeddings)} embeddings "
                f"({tokens_used} tokens, ${cost:.6f}, {processing_time:.2f}s)"
            )

            return embeddings

        except Exception as e:
            # Log failure
            await self.context_stream.add_event(
                ContextEventType.TOOL_EXECUTION,
                {"tool": "voyage_embeddings", "status": "error", "error": str(e)},
            )

            logger.error(f"❌ Voyage embeddings failed: {e}")
            raise

    async def embed_query(self, query: str) -> List[float]:
        """Embed a search query"""
        return await self.embed(query, input_type="query")

    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        return await self.embed(documents, input_type="document")

    async def compute_similarity(
        self, query_embedding: List[float], document_embeddings: List[List[float]]
    ) -> List[float]:
        """
        Compute cosine similarity between query and documents

        Args:
            query_embedding: Query embedding vector
            document_embeddings: List of document embedding vectors

        Returns:
            List of similarity scores
        """
        import numpy as np

        query_vec = np.array(query_embedding)
        doc_vecs = np.array(document_embeddings)

        # Compute cosine similarity
        query_norm = np.linalg.norm(query_vec)
        doc_norms = np.linalg.norm(doc_vecs, axis=1)

        similarities = np.dot(doc_vecs, query_vec) / (doc_norms * query_norm)

        return similarities.tolist()

    def _get_cache_key(self, text: str, input_type: str) -> str:
        """Generate cache key for text"""
        content = f"{self.model}:{input_type}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get_stats(self) -> Dict[str, Any]:
        """Get embedding statistics"""
        cache_hit_rate = (
            self.stats["cache_hits"]
            / (self.stats["cache_hits"] + self.stats["cache_misses"])
            if (self.stats["cache_hits"] + self.stats["cache_misses"]) > 0
            else 0.0
        )

        return {
            **self.stats,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self._cache),
            "model": self.model,
            "dimension": self.dimension,
            "average_cost_per_request": (
                self.stats["total_cost"] / self.stats["total_requests"]
                if self.stats["total_requests"] > 0
                else 0.0
            ),
        }

    def clear_cache(self) -> None:
        """Clear embedding cache"""
        self._cache.clear()
        logger.info("🧹 Embedding cache cleared")


class EmbeddingBatcher:
    """
    Utility for efficient batch processing of embeddings
    """

    def __init__(
        self,
        voyage_client: VoyageEmbeddings,
        batch_size: int = 100,
        max_chars_per_batch: int = 100000,
    ):
        """
        Initialize embedding batcher

        Args:
            voyage_client: Voyage embeddings client
            batch_size: Maximum texts per batch
            max_chars_per_batch: Maximum characters per batch
        """
        self.voyage = voyage_client
        self.batch_size = batch_size
        self.max_chars_per_batch = max_chars_per_batch

    async def embed_in_batches(
        self, texts: List[str], input_type: str = "document"
    ) -> List[List[float]]:
        """
        Embed large list of texts in optimized batches

        Args:
            texts: List of texts to embed
            input_type: Type of input ('document' or 'query')

        Returns:
            List of embeddings corresponding to input texts
        """
        if not texts:
            return []

        batches = self._create_batches(texts)
        all_embeddings = []

        logger.info(f"📦 Processing {len(texts)} texts in {len(batches)} batches")

        for i, batch in enumerate(batches):
            logger.info(
                f"🔄 Processing batch {i+1}/{len(batches)} ({len(batch)} texts)"
            )

            batch_embeddings = await self.voyage.embed(batch, input_type=input_type)
            all_embeddings.extend(batch_embeddings)

            # Small delay between batches to be respectful
            if i < len(batches) - 1:
                await asyncio.sleep(0.1)

        return all_embeddings

    def _create_batches(self, texts: List[str]) -> List[List[str]]:
        """Create optimized batches based on size and character limits"""
        batches = []
        current_batch = []
        current_chars = 0

        for text in texts:
            text_chars = len(text)

            # Check if adding this text would exceed limits
            if (
                len(current_batch) >= self.batch_size
                or current_chars + text_chars > self.max_chars_per_batch
            ) and current_batch:

                # Start new batch
                batches.append(current_batch)
                current_batch = [text]
                current_chars = text_chars
            else:
                # Add to current batch
                current_batch.append(text)
                current_chars += text_chars

        # Add final batch if not empty
        if current_batch:
            batches.append(current_batch)

        return batches
