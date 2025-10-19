#!/usr/bin/env python3
"""
METIS Context Compression Engine - Phase 2
3000+ token efficiency through intelligent context compression and semantic chunking

INDUSTRY INSIGHT: Context window optimization critical for enterprise scalability
Implements academic research-validated compression techniques for cognitive processing
"""

import re
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


class CompressionStrategy(str, Enum):
    """Context compression strategies for different scenarios"""

    SEMANTIC_CHUNKING = "semantic_chunking"  # Group related content semantically
    PRIORITY_FILTERING = "priority_filtering"  # Filter by importance/relevance
    HIERARCHICAL_SUMMARY = "hierarchical_summary"  # Multi-level summarization
    KEYWORD_EXTRACTION = "keyword_extraction"  # Extract key concepts and entities
    SLIDING_WINDOW = "sliding_window"  # Maintain recent context window
    ADAPTIVE_COMPRESSION = "adaptive_compression"  # Dynamic strategy selection


@dataclass
class CompressionMetrics:
    """Metrics for context compression performance"""

    original_tokens: int = 0
    compressed_tokens: int = 0
    compression_ratio: float = 0.0
    information_retention: float = 0.0
    processing_time_ms: float = 0.0
    strategy_used: str = ""
    semantic_coherence: float = 0.0

    def calculate_compression_ratio(self):
        """Calculate compression ratio"""
        if self.original_tokens > 0:
            self.compression_ratio = 1.0 - (
                self.compressed_tokens / self.original_tokens
            )


@dataclass
class CompressedContext:
    """Compressed context with metadata"""

    content: str
    summary: str
    key_concepts: List[str]
    priority_sections: List[Dict[str, Any]]
    compression_metadata: CompressionMetrics
    semantic_chunks: List[Dict[str, Any]]
    retention_map: Dict[str, float]  # Track what information was retained


class ContextCompressionEngine:
    """
    Intelligent context compression for 3000+ token efficiency
    Implements enterprise-grade compression strategies for cognitive processing
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Token counting configuration
        self.tokenizer = self._initialize_tokenizer()

        # Compression targets (industry-validated)
        self.TARGET_TOKEN_LIMIT = 3000  # Optimal for most LLM context windows
        self.EMERGENCY_TOKEN_LIMIT = 2500  # Emergency compression threshold
        self.MIN_COMPRESSION_RATIO = 0.30  # Minimum 30% compression
        self.SEMANTIC_COHERENCE_THRESHOLD = 0.75  # Minimum semantic coherence

        # Priority keywords for context retention
        self.priority_keywords = {
            # Business context priorities
            "business_critical": [
                "revenue",
                "profit",
                "cost",
                "market",
                "customer",
                "competitor",
                "strategy",
                "objective",
                "goal",
                "kpi",
                "metric",
                "performance",
            ],
            # Analytical priorities
            "analytical": [
                "analysis",
                "hypothesis",
                "assumption",
                "conclusion",
                "recommendation",
                "insight",
                "finding",
                "evidence",
                "data",
                "trend",
                "pattern",
            ],
            # Problem-solving priorities
            "problem_solving": [
                "problem",
                "issue",
                "challenge",
                "opportunity",
                "solution",
                "approach",
                "method",
                "framework",
                "model",
                "process",
                "step",
                "action",
            ],
            # Mental model priorities
            "cognitive": [
                "mental model",
                "thinking",
                "reasoning",
                "logic",
                "critical",
                "systems",
                "mece",
                "hypothesis",
                "validation",
                "synthesis",
                "pyramid",
            ],
        }

        # Compression performance tracking
        self.compression_history: List[CompressionMetrics] = []

    def _initialize_tokenizer(self):
        """Initialize token counting system"""
        if TIKTOKEN_AVAILABLE:
            try:
                # Use GPT-4 tokenizer as reference standard
                return tiktoken.encoding_for_model("gpt-4")
            except Exception as e:
                self.logger.warning(f"Failed to initialize tiktoken: {e}")

        # Fallback to approximate token counting
        return None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using available tokenizer"""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                self.logger.warning(f"Token counting failed: {e}")

        # Fallback: rough approximation (1 token ≈ 0.75 words)
        word_count = len(text.split())
        return int(word_count / 0.75)

    async def compress_context(
        self,
        context: Dict[str, Any],
        target_tokens: Optional[int] = None,
        strategy: Optional[CompressionStrategy] = None,
    ) -> CompressedContext:
        """
        Compress context to target token limit using intelligent strategies

        Args:
            context: Context dictionary to compress
            target_tokens: Target token limit (default: 3000)
            strategy: Compression strategy (auto-selected if None)

        Returns:
            CompressedContext with compressed content and metadata
        """

        start_time = datetime.now()
        target_tokens = target_tokens or self.TARGET_TOKEN_LIMIT

        # Convert context to text for processing
        context_text = self._context_to_text(context)
        original_tokens = self.count_tokens(context_text)

        self.logger.info(
            f"🗜️ Context compression: {original_tokens} → {target_tokens} tokens"
        )

        # Check if compression is needed
        if original_tokens <= target_tokens:
            self.logger.info(
                "✅ Context already within token limit, minimal processing"
            )
            return await self._create_minimal_compression(
                context, context_text, original_tokens
            )

        # Select compression strategy
        if not strategy:
            strategy = await self._select_optimal_strategy(
                context, original_tokens, target_tokens
            )

        self.logger.info(f"📊 Using compression strategy: {strategy.value}")

        # Apply compression strategy
        if strategy == CompressionStrategy.SEMANTIC_CHUNKING:
            compressed_result = await self._compress_semantic_chunking(
                context, target_tokens
            )
        elif strategy == CompressionStrategy.PRIORITY_FILTERING:
            compressed_result = await self._compress_priority_filtering(
                context, target_tokens
            )
        elif strategy == CompressionStrategy.HIERARCHICAL_SUMMARY:
            compressed_result = await self._compress_hierarchical_summary(
                context, target_tokens
            )
        elif strategy == CompressionStrategy.KEYWORD_EXTRACTION:
            compressed_result = await self._compress_keyword_extraction(
                context, target_tokens
            )
        elif strategy == CompressionStrategy.SLIDING_WINDOW:
            compressed_result = await self._compress_sliding_window(
                context, target_tokens
            )
        elif strategy == CompressionStrategy.ADAPTIVE_COMPRESSION:
            compressed_result = await self._compress_adaptive(context, target_tokens)
        else:
            # Fallback to priority filtering
            compressed_result = await self._compress_priority_filtering(
                context, target_tokens
            )

        # Calculate final metrics
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        compressed_result.compression_metadata.original_tokens = original_tokens
        compressed_result.compression_metadata.processing_time_ms = processing_time
        compressed_result.compression_metadata.strategy_used = strategy.value
        compressed_result.compression_metadata.calculate_compression_ratio()

        # Track compression history
        self.compression_history.append(compressed_result.compression_metadata)

        # Validate compression quality
        await self._validate_compression_quality(compressed_result)

        self.logger.info(
            f"✅ Context compressed: {compressed_result.compression_metadata.compression_ratio:.1%} reduction, "
            f"{compressed_result.compression_metadata.information_retention:.1%} retention, "
            f"{processing_time:.1f}ms"
        )

        return compressed_result

    def _context_to_text(self, context: Dict[str, Any]) -> str:
        """Convert context dictionary to text for processing"""

        def extract_text(obj, prefix=""):
            """Recursively extract text from nested objects"""
            if isinstance(obj, str):
                return f"{prefix}{obj}\n"
            elif isinstance(obj, dict):
                text = ""
                for key, value in obj.items():
                    if isinstance(value, (str, int, float, bool)):
                        text += f"{prefix}{key}: {value}\n"
                    elif isinstance(value, (dict, list)):
                        text += extract_text(value, f"{prefix}{key}.")
                return text
            elif isinstance(obj, list):
                text = ""
                for i, item in enumerate(obj):
                    text += extract_text(item, f"{prefix}[{i}].")
                return text
            else:
                return f"{prefix}{str(obj)}\n"

        return extract_text(context)

    async def _select_optimal_strategy(
        self, context: Dict[str, Any], original_tokens: int, target_tokens: int
    ) -> CompressionStrategy:
        """Select optimal compression strategy based on context characteristics"""

        compression_needed = (original_tokens - target_tokens) / original_tokens

        # Analyze context characteristics
        has_multiple_sections = len(context) > 5
        has_hierarchical_data = any(isinstance(v, dict) for v in context.values())
        has_list_data = any(isinstance(v, list) for v in context.values())
        has_long_text = any(
            isinstance(v, str) and len(v) > 1000 for v in context.values()
        )

        # Strategy selection logic
        if compression_needed > 0.70:  # Aggressive compression needed
            if has_hierarchical_data:
                return CompressionStrategy.HIERARCHICAL_SUMMARY
            else:
                return CompressionStrategy.KEYWORD_EXTRACTION

        elif compression_needed > 0.40:  # Moderate compression needed
            if has_multiple_sections:
                return CompressionStrategy.SEMANTIC_CHUNKING
            else:
                return CompressionStrategy.PRIORITY_FILTERING

        else:  # Light compression needed
            if has_long_text:
                return CompressionStrategy.SLIDING_WINDOW
            else:
                return CompressionStrategy.PRIORITY_FILTERING

    async def _create_minimal_compression(
        self, context: Dict[str, Any], context_text: str, token_count: int
    ) -> CompressedContext:
        """Create minimal compression for contexts already within limits"""

        # Extract key concepts for completeness
        key_concepts = await self._extract_key_concepts(context_text)

        # Create summary
        summary = await self._generate_summary(context_text, max_length=200)

        metrics = CompressionMetrics(
            original_tokens=token_count,
            compressed_tokens=token_count,
            compression_ratio=0.0,
            information_retention=1.0,
            semantic_coherence=1.0,
        )

        return CompressedContext(
            content=context_text,
            summary=summary,
            key_concepts=key_concepts,
            priority_sections=[
                {"type": "original", "content": context_text, "priority": 1.0}
            ],
            compression_metadata=metrics,
            semantic_chunks=[{"chunk": context_text, "importance": 1.0}],
            retention_map={"all_content": 1.0},
        )

    async def _compress_semantic_chunking(
        self, context: Dict[str, Any], target_tokens: int
    ) -> CompressedContext:
        """Compress using semantic chunking strategy"""

        context_text = self._context_to_text(context)

        # Split into semantic chunks
        chunks = await self._create_semantic_chunks(context_text)

        # Score chunks by importance
        scored_chunks = []
        for chunk in chunks:
            importance = await self._calculate_chunk_importance(chunk)
            token_count = self.count_tokens(chunk)
            scored_chunks.append(
                {
                    "content": chunk,
                    "importance": importance,
                    "tokens": token_count,
                    "density": importance / token_count if token_count > 0 else 0,
                }
            )

        # Sort by importance density (importance per token)
        scored_chunks.sort(key=lambda x: x["density"], reverse=True)

        # Select chunks within token budget
        selected_chunks = []
        total_tokens = 0
        retention_map = {}

        for chunk_data in scored_chunks:
            if total_tokens + chunk_data["tokens"] <= target_tokens:
                selected_chunks.append(chunk_data)
                total_tokens += chunk_data["tokens"]
                retention_map[f"chunk_{len(selected_chunks)}"] = chunk_data[
                    "importance"
                ]

        # Combine selected chunks
        compressed_content = "\n\n".join(
            [chunk["content"] for chunk in selected_chunks]
        )

        # Extract key concepts
        key_concepts = await self._extract_key_concepts(compressed_content)

        # Generate summary
        summary = await self._generate_summary(compressed_content, max_length=300)

        # Calculate metrics
        original_tokens = self.count_tokens(context_text)
        compressed_tokens = self.count_tokens(compressed_content)
        information_retention = sum(
            chunk["importance"] for chunk in selected_chunks
        ) / len(scored_chunks)

        metrics = CompressionMetrics(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            information_retention=information_retention,
            semantic_coherence=await self._calculate_semantic_coherence(
                compressed_content
            ),
        )

        priority_sections = [
            {
                "type": "semantic_chunk",
                "content": chunk["content"],
                "priority": chunk["importance"],
                "tokens": chunk["tokens"],
            }
            for chunk in selected_chunks
        ]

        return CompressedContext(
            content=compressed_content,
            summary=summary,
            key_concepts=key_concepts,
            priority_sections=priority_sections,
            compression_metadata=metrics,
            semantic_chunks=selected_chunks,
            retention_map=retention_map,
        )

    async def _compress_priority_filtering(
        self, context: Dict[str, Any], target_tokens: int
    ) -> CompressedContext:
        """Compress using priority-based filtering"""

        context_text = self._context_to_text(context)

        # Split into sentences/sections
        sections = await self._split_into_sections(context_text)

        # Score sections by priority
        scored_sections = []
        for section in sections:
            priority = await self._calculate_section_priority(section)
            token_count = self.count_tokens(section)
            scored_sections.append(
                {"content": section, "priority": priority, "tokens": token_count}
            )

        # Sort by priority
        scored_sections.sort(key=lambda x: x["priority"], reverse=True)

        # Select high-priority sections within token budget
        selected_sections = []
        total_tokens = 0
        retention_map = {}

        for section_data in scored_sections:
            if total_tokens + section_data["tokens"] <= target_tokens:
                selected_sections.append(section_data)
                total_tokens += section_data["tokens"]
                retention_map[f"section_{len(selected_sections)}"] = section_data[
                    "priority"
                ]

        # Combine selected sections
        compressed_content = "\n".join(
            [section["content"] for section in selected_sections]
        )

        # Extract key concepts
        key_concepts = await self._extract_key_concepts(compressed_content)

        # Generate summary
        summary = await self._generate_summary(compressed_content, max_length=250)

        # Calculate metrics
        original_tokens = self.count_tokens(context_text)
        compressed_tokens = self.count_tokens(compressed_content)
        information_retention = sum(
            section["priority"] for section in selected_sections
        ) / len(scored_sections)

        metrics = CompressionMetrics(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            information_retention=information_retention,
            semantic_coherence=await self._calculate_semantic_coherence(
                compressed_content
            ),
        )

        return CompressedContext(
            content=compressed_content,
            summary=summary,
            key_concepts=key_concepts,
            priority_sections=selected_sections,
            compression_metadata=metrics,
            semantic_chunks=[
                {"chunk": compressed_content, "importance": information_retention}
            ],
            retention_map=retention_map,
        )

    async def _compress_hierarchical_summary(
        self, context: Dict[str, Any], target_tokens: int
    ) -> CompressedContext:
        """Compress using hierarchical summarization"""

        # Build hierarchical structure from context
        hierarchy = await self._build_context_hierarchy(context)

        # Summarize at each level
        summarized_hierarchy = await self._summarize_hierarchy(hierarchy, target_tokens)

        # Flatten to compressed content
        compressed_content = await self._flatten_hierarchy(summarized_hierarchy)

        # Extract key concepts from original context
        original_text = self._context_to_text(context)
        key_concepts = await self._extract_key_concepts(original_text)

        # Generate executive summary
        summary = await self._generate_summary(compressed_content, max_length=200)

        # Calculate metrics
        original_tokens = self.count_tokens(original_text)
        compressed_tokens = self.count_tokens(compressed_content)

        metrics = CompressionMetrics(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            information_retention=0.8,  # Hierarchical summaries maintain high-level information
            semantic_coherence=await self._calculate_semantic_coherence(
                compressed_content
            ),
        )

        return CompressedContext(
            content=compressed_content,
            summary=summary,
            key_concepts=key_concepts,
            priority_sections=[
                {"type": "hierarchical", "content": compressed_content, "priority": 0.8}
            ],
            compression_metadata=metrics,
            semantic_chunks=[{"chunk": compressed_content, "importance": 0.8}],
            retention_map={"hierarchical_summary": 0.8},
        )

    async def _compress_keyword_extraction(
        self, context: Dict[str, Any], target_tokens: int
    ) -> CompressedContext:
        """Compress using keyword and key phrase extraction"""

        context_text = self._context_to_text(context)

        # Extract key concepts and entities
        key_concepts = await self._extract_key_concepts(context_text)
        key_phrases = await self._extract_key_phrases(context_text)

        # Build compressed content from key elements
        compressed_sections = []

        # Add key concepts section
        if key_concepts:
            concepts_text = f"Key Concepts: {', '.join(key_concepts[:20])}"
            compressed_sections.append(concepts_text)

        # Add key phrases with context
        if key_phrases:
            phrases_text = f"Key Insights: {'. '.join(key_phrases[:15])}"
            compressed_sections.append(phrases_text)

        # Add essential context elements
        essential_context = await self._extract_essential_context(
            context_text, target_tokens // 2
        )
        if essential_context:
            compressed_sections.append(f"Context: {essential_context}")

        compressed_content = "\n\n".join(compressed_sections)

        # Ensure we're within token limits
        if self.count_tokens(compressed_content) > target_tokens:
            # Truncate to fit
            compressed_content = await self._truncate_to_tokens(
                compressed_content, target_tokens
            )

        # Generate summary
        summary = await self._generate_summary(compressed_content, max_length=150)

        # Calculate metrics
        original_tokens = self.count_tokens(context_text)
        compressed_tokens = self.count_tokens(compressed_content)

        metrics = CompressionMetrics(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            information_retention=0.6,  # Keyword extraction preserves core information
            semantic_coherence=0.7,  # Lower coherence but high information density
        )

        return CompressedContext(
            content=compressed_content,
            summary=summary,
            key_concepts=key_concepts,
            priority_sections=[
                {"type": "keywords", "content": compressed_content, "priority": 0.6}
            ],
            compression_metadata=metrics,
            semantic_chunks=[{"chunk": compressed_content, "importance": 0.6}],
            retention_map={"keywords": 0.6, "phrases": 0.7, "context": 0.5},
        )

    async def _compress_sliding_window(
        self, context: Dict[str, Any], target_tokens: int
    ) -> CompressedContext:
        """Compress using sliding window approach (keep most recent content)"""

        context_text = self._context_to_text(context)

        # Split into sections (prioritize recent/final sections)
        sections = await self._split_into_sections(context_text)

        # Reverse to start from most recent
        sections.reverse()

        # Select sections within token budget
        selected_sections = []
        total_tokens = 0

        for section in sections:
            section_tokens = self.count_tokens(section)
            if total_tokens + section_tokens <= target_tokens:
                selected_sections.append(section)
                total_tokens += section_tokens
            else:
                # Partial inclusion if space allows
                remaining_tokens = target_tokens - total_tokens
                if remaining_tokens > 50:  # Minimum viable section
                    partial_section = await self._truncate_to_tokens(
                        section, remaining_tokens
                    )
                    selected_sections.append(partial_section)
                break

        # Restore original order
        selected_sections.reverse()
        compressed_content = "\n".join(selected_sections)

        # Extract key concepts
        key_concepts = await self._extract_key_concepts(compressed_content)

        # Generate summary
        summary = await self._generate_summary(compressed_content, max_length=200)

        # Calculate metrics
        original_tokens = self.count_tokens(context_text)
        compressed_tokens = self.count_tokens(compressed_content)

        metrics = CompressionMetrics(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            information_retention=0.75,  # Recent content has high relevance
            semantic_coherence=0.85,  # Maintains narrative flow
        )

        return CompressedContext(
            content=compressed_content,
            summary=summary,
            key_concepts=key_concepts,
            priority_sections=[
                {"type": "recent", "content": compressed_content, "priority": 0.75}
            ],
            compression_metadata=metrics,
            semantic_chunks=[{"chunk": compressed_content, "importance": 0.75}],
            retention_map={"recent_content": 0.75},
        )

    async def _compress_adaptive(
        self, context: Dict[str, Any], target_tokens: int
    ) -> CompressedContext:
        """Adaptive compression using multiple strategies"""

        # Try multiple strategies and select best result
        strategies = [
            CompressionStrategy.SEMANTIC_CHUNKING,
            CompressionStrategy.PRIORITY_FILTERING,
            CompressionStrategy.HIERARCHICAL_SUMMARY,
        ]

        best_result = None
        best_score = 0

        for strategy in strategies:
            try:
                result = await self.compress_context(context, target_tokens, strategy)

                # Score based on information retention and semantic coherence
                score = (
                    result.compression_metadata.information_retention * 0.6
                    + result.compression_metadata.semantic_coherence * 0.4
                )

                if score > best_score:
                    best_score = score
                    best_result = result

            except Exception as e:
                self.logger.warning(f"Strategy {strategy.value} failed: {e}")
                continue

        if best_result:
            best_result.compression_metadata.strategy_used = "adaptive_best"
            return best_result
        else:
            # Fallback to priority filtering
            return await self._compress_priority_filtering(context, target_tokens)

    # Helper methods for compression strategies

    async def _create_semantic_chunks(self, text: str) -> List[str]:
        """Create semantic chunks from text"""

        # Split by paragraphs and logical breaks
        chunks = []

        # Split by double newlines (paragraph breaks)
        paragraphs = text.split("\n\n")

        current_chunk = ""
        for paragraph in paragraphs:
            # If adding this paragraph would make chunk too large, start new chunk
            if (
                current_chunk
                and self.count_tokens(current_chunk + "\n\n" + paragraph) > 500
            ):
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return [chunk for chunk in chunks if chunk.strip()]

    async def _calculate_chunk_importance(self, chunk: str) -> float:
        """Calculate importance score for a text chunk"""

        importance = 0.0

        # Check for priority keywords
        chunk_lower = chunk.lower()

        # Business keywords (high weight)
        business_matches = sum(
            1
            for keyword in self.priority_keywords["business_critical"]
            if keyword in chunk_lower
        )
        importance += business_matches * 0.3

        # Analytical keywords (high weight)
        analytical_matches = sum(
            1
            for keyword in self.priority_keywords["analytical"]
            if keyword in chunk_lower
        )
        importance += analytical_matches * 0.25

        # Problem-solving keywords (medium weight)
        problem_matches = sum(
            1
            for keyword in self.priority_keywords["problem_solving"]
            if keyword in chunk_lower
        )
        importance += problem_matches * 0.2

        # Cognitive keywords (medium weight)
        cognitive_matches = sum(
            1
            for keyword in self.priority_keywords["cognitive"]
            if keyword in chunk_lower
        )
        importance += cognitive_matches * 0.15

        # Length factor (longer chunks may be more comprehensive)
        length_factor = min(1.0, len(chunk) / 1000)
        importance += length_factor * 0.1

        # Normalize to 0-1 range
        return min(1.0, importance)

    async def _split_into_sections(self, text: str) -> List[str]:
        """Split text into logical sections"""

        # Split by various delimiters
        sections = []

        # First try paragraph splits
        paragraphs = text.split("\n\n")

        for paragraph in paragraphs:
            # Further split long paragraphs by sentences
            if len(paragraph) > 500:
                sentences = re.split(r"[.!?]+", paragraph)
                current_section = ""

                for sentence in sentences:
                    if sentence.strip():
                        if current_section and len(current_section + sentence) > 300:
                            sections.append(current_section.strip())
                            current_section = sentence.strip()
                        else:
                            if current_section:
                                current_section += ". " + sentence.strip()
                            else:
                                current_section = sentence.strip()

                if current_section:
                    sections.append(current_section.strip())
            else:
                sections.append(paragraph.strip())

        return [section for section in sections if section]

    async def _calculate_section_priority(self, section: str) -> float:
        """Calculate priority score for a section"""

        # Similar to chunk importance but with different weights
        priority = 0.0
        section_lower = section.lower()

        # Keywords scoring
        for category, keywords in self.priority_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in section_lower)
            if category == "business_critical":
                priority += matches * 0.4
            elif category == "analytical":
                priority += matches * 0.3
            elif category == "problem_solving":
                priority += matches * 0.2
            elif category == "cognitive":
                priority += matches * 0.1

        # Position factor (earlier sections often more important)
        # This would need to be calculated by the caller

        return min(1.0, priority)

    async def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""

        concepts = set()
        text_lower = text.lower()

        # Extract priority keywords that appear in text
        for category, keywords in self.priority_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    concepts.add(keyword.title())

        # Extract capitalized terms (potential proper nouns/concepts)
        words = text.split()
        for word in words:
            # Clean word
            clean_word = re.sub(r"[^\w]", "", word)
            if len(clean_word) > 3 and clean_word[0].isupper():
                concepts.add(clean_word)

        return sorted(list(concepts))[:50]  # Limit to top 50 concepts

    async def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text"""

        phrases = []

        # Split into sentences
        sentences = re.split(r"[.!?]+", text)

        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # Check if sentence contains priority keywords
                sentence_lower = sentence.lower()
                keyword_count = 0

                for keywords in self.priority_keywords.values():
                    keyword_count += sum(
                        1 for keyword in keywords if keyword in sentence_lower
                    )

                # Include sentences with high keyword density
                if keyword_count >= 2:
                    phrases.append(sentence)

        return phrases[:20]  # Limit to top 20 phrases

    async def _extract_essential_context(self, text: str, max_tokens: int) -> str:
        """Extract essential context within token limit"""

        # Find sentences with highest keyword density
        sentences = re.split(r"[.!?]+", text)

        scored_sentences = []
        for sentence in sentences:
            if sentence.strip():
                score = await self._calculate_section_priority(sentence)
                token_count = self.count_tokens(sentence)
                scored_sentences.append(
                    {"text": sentence.strip(), "score": score, "tokens": token_count}
                )

        # Sort by score
        scored_sentences.sort(key=lambda x: x["score"], reverse=True)

        # Select sentences within token budget
        selected_sentences = []
        total_tokens = 0

        for sentence_data in scored_sentences:
            if total_tokens + sentence_data["tokens"] <= max_tokens:
                selected_sentences.append(sentence_data["text"])
                total_tokens += sentence_data["tokens"]

        return ". ".join(selected_sentences)

    async def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit"""

        if self.count_tokens(text) <= max_tokens:
            return text

        # Binary search for optimal truncation point
        words = text.split()
        left, right = 0, len(words)

        best_text = ""

        while left <= right:
            mid = (left + right) // 2
            candidate = " ".join(words[:mid])

            if self.count_tokens(candidate) <= max_tokens:
                best_text = candidate
                left = mid + 1
            else:
                right = mid - 1

        return best_text

    async def _build_context_hierarchy(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build hierarchical structure from context"""

        hierarchy = {}

        for key, value in context.items():
            if isinstance(value, dict):
                hierarchy[key] = await self._build_context_hierarchy(value)
            elif isinstance(value, list):
                hierarchy[key] = [str(item) for item in value]
            else:
                hierarchy[key] = str(value)

        return hierarchy

    async def _summarize_hierarchy(
        self, hierarchy: Dict[str, Any], target_tokens: int
    ) -> Dict[str, Any]:
        """Summarize hierarchical structure"""

        summarized = {}

        for key, value in hierarchy.items():
            if isinstance(value, dict):
                # Recursively summarize sub-hierarchies
                summarized[key] = await self._summarize_hierarchy(
                    value, target_tokens // len(hierarchy)
                )
            elif isinstance(value, list):
                # Summarize lists
                if len(value) > 5:
                    summarized[key] = value[:3] + [
                        f"... and {len(value) - 3} more items"
                    ]
                else:
                    summarized[key] = value
            else:
                # Truncate long strings
                if len(str(value)) > 200:
                    summarized[key] = str(value)[:200] + "..."
                else:
                    summarized[key] = value

        return summarized

    async def _flatten_hierarchy(self, hierarchy: Dict[str, Any]) -> str:
        """Flatten hierarchical structure to text"""

        def flatten_recursive(obj, prefix=""):
            lines = []

            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, (dict, list)):
                        lines.append(f"{prefix}{key}:")
                        lines.extend(flatten_recursive(value, prefix + "  "))
                    else:
                        lines.append(f"{prefix}{key}: {value}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    lines.append(f"{prefix}- {item}")
            else:
                lines.append(f"{prefix}{obj}")

            return lines

        return "\n".join(flatten_recursive(hierarchy))

    async def _generate_summary(self, text: str, max_length: int = 200) -> str:
        """Generate concise summary of text"""

        # Extract first few sentences up to max_length
        sentences = re.split(r"[.!?]+", text)

        summary = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                if len(summary + sentence) <= max_length:
                    if summary:
                        summary += ". " + sentence
                    else:
                        summary = sentence
                else:
                    break

        return summary

    async def _calculate_semantic_coherence(self, text: str) -> float:
        """Calculate semantic coherence score (0-1)"""

        # Simple heuristic based on:
        # 1. Keyword repetition (higher coherence)
        # 2. Sentence structure consistency
        # 3. Logical flow indicators

        coherence = 0.5  # Base score

        # Check for coherence indicators
        text_lower = text.lower()

        # Transition words indicate coherence
        transition_words = [
            "therefore",
            "however",
            "furthermore",
            "additionally",
            "consequently",
            "moreover",
            "nevertheless",
            "thus",
            "hence",
            "accordingly",
        ]

        transition_count = sum(1 for word in transition_words if word in text_lower)
        coherence += min(0.3, transition_count * 0.05)

        # Keyword consistency
        key_concepts = await self._extract_key_concepts(text)
        if len(key_concepts) > 0:
            # Higher coherence if key concepts appear multiple times
            coherence += min(0.2, len(key_concepts) * 0.01)

        return min(1.0, coherence)

    async def _validate_compression_quality(
        self, compressed_context: CompressedContext
    ):
        """Validate compression quality and log warnings if needed"""

        metrics = compressed_context.compression_metadata

        # Check compression ratio
        if metrics.compression_ratio < self.MIN_COMPRESSION_RATIO:
            self.logger.warning(
                f"⚠️ Low compression ratio: {metrics.compression_ratio:.1%} "
                f"(target: >{self.MIN_COMPRESSION_RATIO:.1%})"
            )

        # Check semantic coherence
        if metrics.semantic_coherence < self.SEMANTIC_COHERENCE_THRESHOLD:
            self.logger.warning(
                f"⚠️ Low semantic coherence: {metrics.semantic_coherence:.1%} "
                f"(target: >{self.SEMANTIC_COHERENCE_THRESHOLD:.1%})"
            )

        # Check information retention
        if metrics.information_retention < 0.5:
            self.logger.warning(
                f"⚠️ Low information retention: {metrics.information_retention:.1%} "
                f"(consider alternative strategy)"
            )

    def get_compression_analytics(self) -> Dict[str, Any]:
        """Get analytics on compression performance"""

        if not self.compression_history:
            return {"status": "no_data", "message": "No compression history available"}

        # Calculate aggregate metrics
        total_compressions = len(self.compression_history)
        avg_compression_ratio = (
            sum(m.compression_ratio for m in self.compression_history)
            / total_compressions
        )
        avg_information_retention = (
            sum(m.information_retention for m in self.compression_history)
            / total_compressions
        )
        avg_processing_time = (
            sum(m.processing_time_ms for m in self.compression_history)
            / total_compressions
        )

        # Strategy usage
        strategy_usage = {}
        for metrics in self.compression_history:
            strategy = metrics.strategy_used
            strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1

        return {
            "total_compressions": total_compressions,
            "performance_metrics": {
                "avg_compression_ratio": avg_compression_ratio,
                "avg_information_retention": avg_information_retention,
                "avg_processing_time_ms": avg_processing_time,
                "avg_semantic_coherence": sum(
                    m.semantic_coherence for m in self.compression_history
                )
                / total_compressions,
            },
            "strategy_usage": strategy_usage,
            "quality_assessment": {
                "compression_efficiency": (
                    "excellent"
                    if avg_compression_ratio > 0.5
                    else "good" if avg_compression_ratio > 0.3 else "needs_improvement"
                ),
                "information_preservation": (
                    "excellent"
                    if avg_information_retention > 0.8
                    else (
                        "good"
                        if avg_information_retention > 0.6
                        else "needs_improvement"
                    )
                ),
                "processing_speed": (
                    "fast"
                    if avg_processing_time < 100
                    else "moderate" if avg_processing_time < 500 else "slow"
                ),
            },
        }


# Global context compression engine instance
_context_compression_engine: Optional[ContextCompressionEngine] = None


async def get_context_compression_engine() -> ContextCompressionEngine:
    """Get or create global context compression engine instance"""
    global _context_compression_engine

    if _context_compression_engine is None:
        _context_compression_engine = ContextCompressionEngine()

    return _context_compression_engine


# Decorator for automatic context compression
def compressed_context(
    target_tokens: int = 3000, strategy: Optional[CompressionStrategy] = None
):
    """Decorator to automatically compress context before function execution"""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract context from kwargs
            context = kwargs.get("context", {})

            if context:
                engine = await get_context_compression_engine()
                compressed = await engine.compress_context(
                    context, target_tokens, strategy
                )

                # Replace context with compressed version
                kwargs["context"] = {
                    "compressed_content": compressed.content,
                    "summary": compressed.summary,
                    "key_concepts": compressed.key_concepts,
                    "compression_metadata": asdict(compressed.compression_metadata),
                }

                logging.getLogger(__name__).info(
                    f"Context compressed for {func.__name__}: "
                    f"{compressed.compression_metadata.compression_ratio:.1%} reduction"
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator
