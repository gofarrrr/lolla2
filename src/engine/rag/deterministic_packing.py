"""
Deterministic RAG Packing - Reproducible Document Assembly

Ensures RAG operations produce consistent, reproducible results:
- Deterministic document ordering (hash-based sorting)
- Consistent chunk packing (stable ordering)
- Content fingerprinting (SHA-256 hashing)
- Token budget management (smart truncation)
- Reproducibility guarantees (same input → same output)

Architecture:
- Content-based ordering (not time-based or random)
- Hash-based document fingerprinting
- Deterministic chunk selection
- Token-aware packing strategies

Reproducibility Guarantees:
- FULLY_DETERMINISTIC: Exact same output every time
- MOSTLY_DETERMINISTIC: Minor variations (< 5% difference)
- NON_DETERMINISTIC: Non-reproducible (legacy mode)

ROI:
- Enables debugging and testing
- Reproducible analysis results
- Audit trail compliance
- A/B testing capability

Implementation:
- 2-day implementation with hash-based ordering
- Content fingerprinting for de-duplication
- Token budget management
"""

import hashlib
import logging
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
from dataclasses import dataclass
from collections import OrderedDict

logger = logging.getLogger(__name__)


class ReproducibilityLevel(str, Enum):
    """Reproducibility guarantee levels"""
    FULLY_DETERMINISTIC = "fully_deterministic"      # Exact reproduction
    MOSTLY_DETERMINISTIC = "mostly_deterministic"    # < 5% variation
    NON_DETERMINISTIC = "non_deterministic"          # Legacy mode


class PackingStrategy(str, Enum):
    """Document packing strategies"""
    RELEVANCE_ORDERED = "relevance_ordered"          # By relevance score
    CONTENT_HASH_ORDERED = "content_hash_ordered"    # By content hash
    HYBRID = "hybrid"                                 # Relevance + hash tiebreaker


@dataclass
class DocumentChunk:
    """Document chunk with metadata"""
    content: str
    source_id: str
    chunk_index: int
    relevance_score: float
    token_count: int
    content_hash: str
    metadata: Dict[str, Any]


@dataclass
class PackedDocument:
    """Packed document bundle"""
    chunks: List[DocumentChunk]
    total_tokens: int
    total_chunks: int
    content_fingerprint: str
    truncated: bool
    packing_strategy: PackingStrategy
    reproducibility_level: ReproducibilityLevel


@dataclass
class PackingResult:
    """Result of document packing"""
    packed_content: str
    packed_document: PackedDocument
    packing_metadata: Dict[str, Any]
    warnings: List[str]


# ============================================================================
# CONTENT FINGERPRINTING
# ============================================================================


def compute_content_hash(content: str) -> str:
    """
    Compute SHA-256 hash of content for fingerprinting.

    Args:
        content: Text content to hash

    Returns:
        Hex digest of SHA-256 hash
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_chunk_hash(chunk: DocumentChunk) -> str:
    """
    Compute deterministic hash for a chunk.

    Includes content, source_id, and chunk_index for uniqueness.

    Args:
        chunk: Document chunk

    Returns:
        Hex digest of combined hash
    """
    combined = f"{chunk.content}|{chunk.source_id}|{chunk.chunk_index}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def compute_document_fingerprint(chunks: List[DocumentChunk]) -> str:
    """
    Compute fingerprint of entire packed document.

    Args:
        chunks: List of document chunks in order

    Returns:
        Hex digest of combined content hash
    """
    combined_content = "".join([c.content for c in chunks])
    return compute_content_hash(combined_content)


# ============================================================================
# TOKEN ESTIMATION
# ============================================================================


def estimate_token_count(text: str) -> int:
    """
    Estimate token count for text.

    Simple heuristic: ~4 chars per token (conservative estimate).

    Args:
        text: Text to estimate

    Returns:
        Estimated token count
    """
    # Conservative estimate: 1 token ≈ 4 characters
    return len(text) // 4 + 1


# ============================================================================
# DETERMINISTIC RAG PACKER
# ============================================================================


class DeterministicRAGPacker:
    """
    Deterministic RAG document packer for reproducible results.

    Features:
    - Content-based deterministic ordering
    - Hash-based chunk fingerprinting
    - Token budget management
    - Reproducibility guarantees
    - De-duplication

    Usage:
        packer = DeterministicRAGPacker(
            max_tokens=4000,
            strategy=PackingStrategy.HYBRID
        )

        result = packer.pack_documents(
            documents=retrieved_docs,
            query="What are market trends?"
        )

        print(f"Packed: {result.packed_document.total_chunks} chunks")
        print(f"Fingerprint: {result.packed_document.content_fingerprint}")
    """

    def __init__(
        self,
        enabled: bool = True,
        max_tokens: int = 4000,
        strategy: PackingStrategy = PackingStrategy.HYBRID,
        allow_truncation: bool = True,
        deduplication_enabled: bool = True
    ):
        """
        Initialize deterministic RAG packer.

        Args:
            enabled: Whether packing is active
            max_tokens: Maximum token budget for packed content
            strategy: Packing strategy to use
            allow_truncation: Allow truncation if over budget
            deduplication_enabled: Remove duplicate chunks
        """
        self.enabled = enabled
        self.max_tokens = max_tokens
        self.strategy = strategy
        self.allow_truncation = allow_truncation
        self.deduplication_enabled = deduplication_enabled
        self.logger = logging.getLogger(__name__)

        if enabled:
            self.logger.info(
                f"✅ Deterministic RAG Packer enabled: "
                f"max_tokens={max_tokens}, strategy={strategy.value}"
            )
        else:
            self.logger.warning("⚠️ Deterministic RAG Packer DISABLED")

    def pack_documents(
        self,
        documents: List[Dict[str, Any]],
        query: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> PackingResult:
        """
        Pack documents into deterministic bundle.

        Args:
            documents: List of document dicts with content and metadata
            query: Original query (optional, for relevance scoring)
            context: Additional context (optional)

        Returns:
            PackingResult with packed content and metadata
        """
        if not self.enabled:
            # Non-deterministic mode: just concatenate
            packed_content = "\n\n".join([
                d.get("content", "") for d in documents
            ])
            return PackingResult(
                packed_content=packed_content,
                packed_document=PackedDocument(
                    chunks=[],
                    total_tokens=estimate_token_count(packed_content),
                    total_chunks=len(documents),
                    content_fingerprint="",
                    truncated=False,
                    packing_strategy=PackingStrategy.RELEVANCE_ORDERED,
                    reproducibility_level=ReproducibilityLevel.NON_DETERMINISTIC
                ),
                packing_metadata={},
                warnings=["Deterministic packing disabled"]
            )

        warnings = []

        # 1. Convert documents to chunks
        chunks = self._create_chunks_from_documents(documents)

        # 2. De-duplicate chunks
        if self.deduplication_enabled:
            original_count = len(chunks)
            chunks = self._deduplicate_chunks(chunks)
            if len(chunks) < original_count:
                warnings.append(
                    f"Removed {original_count - len(chunks)} duplicate chunks"
                )

        # 3. Sort chunks deterministically
        sorted_chunks = self._sort_chunks_deterministically(chunks, query)

        # 4. Pack chunks within token budget
        packed_chunks, truncated = self._pack_within_budget(sorted_chunks)

        if truncated:
            warnings.append(
                f"Content truncated to fit {self.max_tokens} token budget"
            )

        # 5. Compute fingerprint
        fingerprint = compute_document_fingerprint(packed_chunks)

        # 6. Determine reproducibility level
        reproducibility = self._assess_reproducibility(
            sorted_chunks, packed_chunks, truncated
        )

        # 7. Create packed document
        packed_document = PackedDocument(
            chunks=packed_chunks,
            total_tokens=sum(c.token_count for c in packed_chunks),
            total_chunks=len(packed_chunks),
            content_fingerprint=fingerprint,
            truncated=truncated,
            packing_strategy=self.strategy,
            reproducibility_level=reproducibility
        )

        # 8. Generate packed content string
        packed_content = self._generate_packed_content(packed_chunks)

        # 9. Log result
        self._log_packing(packed_document, warnings)

        return PackingResult(
            packed_content=packed_content,
            packed_document=packed_document,
            packing_metadata={
                "original_chunk_count": len(chunks),
                "deduplication_enabled": self.deduplication_enabled,
                "strategy": self.strategy.value,
                "token_budget": self.max_tokens,
            },
            warnings=warnings
        )

    def _create_chunks_from_documents(
        self, documents: List[Dict[str, Any]]
    ) -> List[DocumentChunk]:
        """Convert documents to chunks"""
        chunks = []

        for doc_idx, doc in enumerate(documents):
            content = doc.get("content", "")
            if not content:
                continue

            # Create chunk
            chunk = DocumentChunk(
                content=content,
                source_id=doc.get("source_id", f"doc_{doc_idx}"),
                chunk_index=doc.get("chunk_index", 0),
                relevance_score=doc.get("relevance_score", 0.5),
                token_count=estimate_token_count(content),
                content_hash=compute_content_hash(content),
                metadata=doc.get("metadata", {})
            )
            chunks.append(chunk)

        return chunks

    def _deduplicate_chunks(
        self, chunks: List[DocumentChunk]
    ) -> List[DocumentChunk]:
        """Remove duplicate chunks based on content hash"""
        seen_hashes = set()
        unique_chunks = []

        for chunk in chunks:
            if chunk.content_hash not in seen_hashes:
                seen_hashes.add(chunk.content_hash)
                unique_chunks.append(chunk)

        return unique_chunks

    def _sort_chunks_deterministically(
        self,
        chunks: List[DocumentChunk],
        query: Optional[str]
    ) -> List[DocumentChunk]:
        """
        Sort chunks deterministically based on strategy.

        Args:
            chunks: List of chunks to sort
            query: Optional query for relevance ordering

        Returns:
            Sorted list of chunks (deterministic order)
        """
        if self.strategy == PackingStrategy.CONTENT_HASH_ORDERED:
            # Pure hash-based ordering (fully deterministic)
            return sorted(chunks, key=lambda c: c.content_hash)

        elif self.strategy == PackingStrategy.RELEVANCE_ORDERED:
            # Relevance-based with hash tiebreaker
            return sorted(
                chunks,
                key=lambda c: (-c.relevance_score, c.content_hash)
            )

        elif self.strategy == PackingStrategy.HYBRID:
            # Hybrid: relevance primary, hash tiebreaker
            # Group by relevance bins, then sort by hash within bins
            return sorted(
                chunks,
                key=lambda c: (
                    -round(c.relevance_score, 1),  # Bin by 0.1 increments
                    c.content_hash                  # Deterministic tiebreaker
                )
            )

        else:
            # Fallback: hash-based
            return sorted(chunks, key=lambda c: c.content_hash)

    def _pack_within_budget(
        self, chunks: List[DocumentChunk]
    ) -> Tuple[List[DocumentChunk], bool]:
        """
        Pack chunks within token budget.

        Args:
            chunks: Sorted list of chunks

        Returns:
            (packed_chunks, truncated_flag)
        """
        packed = []
        total_tokens = 0
        truncated = False

        for chunk in chunks:
            if total_tokens + chunk.token_count <= self.max_tokens:
                packed.append(chunk)
                total_tokens += chunk.token_count
            else:
                if self.allow_truncation:
                    truncated = True
                    break
                else:
                    # Try to fit remaining chunks
                    continue

        return packed, truncated

    def _assess_reproducibility(
        self,
        sorted_chunks: List[DocumentChunk],
        packed_chunks: List[DocumentChunk],
        truncated: bool
    ) -> ReproducibilityLevel:
        """Assess reproducibility level of packing"""
        if not truncated and self.strategy == PackingStrategy.CONTENT_HASH_ORDERED:
            # Perfect reproducibility
            return ReproducibilityLevel.FULLY_DETERMINISTIC

        elif self.strategy == PackingStrategy.HYBRID and not truncated:
            # Minor variations possible due to relevance score rounding
            return ReproducibilityLevel.MOSTLY_DETERMINISTIC

        else:
            # Truncation or pure relevance ordering
            return ReproducibilityLevel.MOSTLY_DETERMINISTIC

    def _generate_packed_content(
        self, chunks: List[DocumentChunk]
    ) -> str:
        """Generate packed content string"""
        sections = []

        for idx, chunk in enumerate(chunks):
            # Include source attribution
            header = f"[Source {idx + 1}: {chunk.source_id}]"
            sections.append(f"{header}\n{chunk.content}")

        return "\n\n".join(sections)

    def _log_packing(
        self, packed_document: PackedDocument, warnings: List[str]
    ):
        """Log packing result"""
        self.logger.info(
            f"📦 RAG Packing: {packed_document.total_chunks} chunks, "
            f"{packed_document.total_tokens} tokens, "
            f"{packed_document.reproducibility_level.value}"
        )

        if warnings:
            for warning in warnings:
                self.logger.warning(f"⚠️ {warning}")

        # Log to glass-box
        try:
            from src.engine.adapters.context_stream import (  # Migrated
                get_unified_context_stream,
                ContextEventType,
            )

            cs = get_unified_context_stream()
            cs.add_event(
                ContextEventType.RAG_RETRIEVAL_COMPLETE,
                {
                    "event_type": "deterministic_rag_packing",
                    "total_chunks": packed_document.total_chunks,
                    "total_tokens": packed_document.total_tokens,
                    "content_fingerprint": packed_document.content_fingerprint,
                    "truncated": packed_document.truncated,
                    "strategy": packed_document.packing_strategy.value,
                    "reproducibility": packed_document.reproducibility_level.value,
                    "warnings_count": len(warnings),
                },
            )
        except Exception as e:
            self.logger.warning(f"Failed to log to glass-box: {e}")


# ============================================================================
# GLOBAL PACKER INSTANCE
# ============================================================================

_rag_packer: Optional[DeterministicRAGPacker] = None


def get_rag_packer(
    enabled: bool = True,
    max_tokens: int = 4000,
    strategy: PackingStrategy = PackingStrategy.HYBRID
) -> DeterministicRAGPacker:
    """Get or create global RAG packer"""
    global _rag_packer

    if _rag_packer is None:
        _rag_packer = DeterministicRAGPacker(
            enabled=enabled,
            max_tokens=max_tokens,
            strategy=strategy
        )

    return _rag_packer
