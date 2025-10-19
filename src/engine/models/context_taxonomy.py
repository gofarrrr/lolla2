"""
Operation Synapse Sprint 1.4: Manus Context Taxonomy Implementation
F002: Context classification, scoring, and management utilities based on Manus Labs methodology

This module implements the Manus Context Engineering Framework with enhancements
from industry insights synthesis, providing structured context management for
the Context Intelligence Revolution.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import hashlib
from dataclasses import dataclass, field
from enum import Enum

from .data_contracts import (
    ContextType,
    ContextRelevanceLevel,
    ContextElement,
    ContextRelevanceScore,
    ContextIntelligenceResult,
)


class ContextCompressionStrategy(str, Enum):
    """Manus-inspired context compression strategies"""

    NONE = "none"  # No compression
    SUMMARIZATION = "summarization"  # Intelligent summarization
    KEYPOINTS = "keypoints"  # Extract key points only
    EMBEDDING = "embedding"  # Semantic embedding compression
    HIERARCHICAL = "hierarchical"  # Hierarchical importance-based compression


class ContextValidationLevel(str, Enum):
    """Context validation levels from LangChain patterns"""

    BASIC = "basic"  # Basic structure validation
    SEMANTIC = "semantic"  # Semantic coherence validation
    FACTUAL = "factual"  # Factual accuracy validation
    COMPLETE = "complete"  # Complete validation pipeline


@dataclass
class ContextTaxonomyClassifier:
    """
    Manus Taxonomy Context Classifier

    Classifies context elements according to Manus framework dimensions:
    - Immediate: Current request and user intent
    - Session: Conversation history and established context
    - Domain: Relevant knowledge and expertise area
    - Procedural: How-to knowledge and methodologies
    - Temporal: Time-sensitive information and trends
    - Relational: Connections and dependencies
    """

    # Classification keywords for each context type
    type_keywords: Dict[ContextType, List[str]] = field(
        default_factory=lambda: {
            ContextType.IMMEDIATE: [
                "current",
                "now",
                "present",
                "today",
                "this",
                "analyze",
                "solve",
                "help",
                "question",
            ],
            ContextType.SESSION: [
                "previous",
                "earlier",
                "before",
                "we discussed",
                "conversation",
                "history",
                "remember",
            ],
            ContextType.DOMAIN: [
                "knowledge",
                "expertise",
                "theory",
                "concept",
                "principle",
                "model",
                "framework",
            ],
            ContextType.PROCEDURAL: [
                "how to",
                "step by step",
                "process",
                "method",
                "approach",
                "technique",
                "procedure",
            ],
            ContextType.TEMPORAL: [
                "trend",
                "recent",
                "forecast",
                "prediction",
                "timeline",
                "schedule",
                "deadline",
            ],
            ContextType.RELATIONAL: [
                "relationship",
                "connection",
                "dependency",
                "link",
                "correlation",
                "association",
            ],
        }
    )

    def classify_context(self, content: str) -> Tuple[ContextType, float]:
        """
        Classify context content into Manus taxonomy with confidence score

        Returns:
            Tuple of (ContextType, confidence_score)
        """
        content_lower = content.lower()
        type_scores = {}

        # Score each context type based on keyword matching
        for context_type, keywords in self.type_keywords.items():
            score = 0.0
            for keyword in keywords:
                if keyword in content_lower:
                    # Weight by keyword specificity (longer keywords get higher weight)
                    weight = len(keyword.split()) * 0.1 + 0.1
                    score += weight

            # Normalize by number of keywords to get average relevance
            if keywords:
                type_scores[context_type] = score / len(keywords)
            else:
                type_scores[context_type] = 0.0

        # Find highest scoring type
        best_type = max(type_scores.keys(), key=lambda k: type_scores[k])
        confidence = min(1.0, type_scores[best_type])

        # If no strong classification, default to IMMEDIATE with low confidence
        if confidence < 0.1:
            return ContextType.IMMEDIATE, 0.3

        return best_type, confidence


@dataclass
class ContextRelevanceScorer:
    """
    Manus-inspired Context Relevance Scorer

    Implements the enhanced Manus relevance scoring methodology with
    cognitive coherence integration from Operation Mindforge.
    """

    # Scoring weights (configurable)
    semantic_weight: float = 0.4
    temporal_weight: float = 0.2
    frequency_weight: float = 0.1
    cognitive_weight: float = 0.3  # Revolutionary cognitive coherence

    def score_context_relevance(
        self,
        context_element: ContextElement,
        current_query: str,
        cognitive_coherence_score: float = 0.0,
    ) -> ContextRelevanceScore:
        """
        Score context element relevance using Manus enhanced methodology

        Args:
            context_element: Context element to score
            current_query: Current user query/goal
            cognitive_coherence_score: Cognitive coherence from Operation Mindforge

        Returns:
            ContextRelevanceScore with detailed breakdown
        """
        # 1. Semantic similarity (simple implementation for Sprint 1.4)
        semantic_score = self._calculate_semantic_similarity(
            context_element.content, current_query
        )

        # 2. Temporal recency
        temporal_score = self._calculate_temporal_recency(context_element.created_at)

        # 3. Usage frequency
        frequency_score = self._calculate_usage_frequency(context_element.access_count)

        # 4. Cognitive coherence (revolutionary feature)
        cognitive_score = cognitive_coherence_score

        # Weighted combination
        overall_score = (
            semantic_score * self.semantic_weight
            + temporal_score * self.temporal_weight
            + frequency_score * self.frequency_weight
            + cognitive_score * self.cognitive_weight
        )

        # Generate explanation
        explanation = (
            f"Context relevance: {overall_score:.3f} "
            f"(semantic={semantic_score:.2f}, temporal={temporal_score:.2f}, "
            f"frequency={frequency_score:.2f}, cognitive={cognitive_score:.2f})"
        )

        return ContextRelevanceScore(
            element_id=context_element.element_id,
            overall_score=overall_score,
            semantic_similarity=semantic_score,
            temporal_recency=temporal_score,
            usage_frequency=frequency_score,
            cognitive_coherence=cognitive_score,
            explanation=explanation,
        )

    def _calculate_semantic_similarity(self, content: str, query: str) -> float:
        """Simple semantic similarity using word overlap (foundational implementation)"""
        content_words = set(content.lower().split())
        query_words = set(query.lower().split())

        if not content_words or not query_words:
            return 0.0

        intersection = len(content_words.intersection(query_words))
        union = len(content_words.union(query_words))

        return intersection / union if union > 0 else 0.0

    def _calculate_temporal_recency(self, created_at: datetime) -> float:
        """Calculate temporal recency with exponential decay"""
        hours_ago = (datetime.utcnow() - created_at).total_seconds() / 3600

        # Exponential decay: 1.0 for current, 0.5 for 24h ago, ~0.1 for 1 week ago
        recency_score = 2 ** (-hours_ago / 24)

        return min(1.0, recency_score)

    def _calculate_usage_frequency(self, access_count: int) -> float:
        """Calculate usage frequency with logarithmic scaling"""
        if access_count <= 0:
            return 0.0

        # Logarithmic scaling: 1 use = 0.0, 10 uses = ~0.5, 100 uses = 1.0
        frequency_score = min(1.0, (access_count - 1) / 20)

        return frequency_score

    def categorize_relevance_level(
        self, relevance_score: float
    ) -> ContextRelevanceLevel:
        """Categorize relevance score into Manus levels"""
        if relevance_score > 0.9:
            return ContextRelevanceLevel.CRITICAL
        elif relevance_score > 0.7:
            return ContextRelevanceLevel.HIGH
        elif relevance_score > 0.5:
            return ContextRelevanceLevel.MEDIUM
        elif relevance_score > 0.3:
            return ContextRelevanceLevel.LOW
        else:
            return ContextRelevanceLevel.IRRELEVANT


@dataclass
class ContextCompressor:
    """
    Manus-inspired Context Compression Engine

    Implements intelligent context compression strategies to optimize
    context window usage while preserving semantic meaning.
    """

    def compress_context(
        self,
        context_elements: List[ContextElement],
        target_compression_ratio: float,
        strategy: ContextCompressionStrategy = ContextCompressionStrategy.SUMMARIZATION,
    ) -> Tuple[List[ContextElement], float]:
        """
        Compress context elements using specified strategy

        Args:
            context_elements: Elements to compress
            target_compression_ratio: Target ratio (0.5 = 50% compression)
            strategy: Compression strategy to use

        Returns:
            Tuple of (compressed_elements, actual_compression_ratio)
        """
        if strategy == ContextCompressionStrategy.NONE:
            return context_elements, 1.0

        # Calculate current content size
        original_size = sum(len(element.content) for element in context_elements)
        target_size = int(original_size * (1.0 - target_compression_ratio))

        if strategy == ContextCompressionStrategy.KEYPOINTS:
            return self._compress_to_keypoints(context_elements, target_size)
        elif strategy == ContextCompressionStrategy.HIERARCHICAL:
            return self._compress_hierarchical(context_elements, target_size)
        else:
            # Default to summarization
            return self._compress_summarization(context_elements, target_size)

    def _compress_to_keypoints(
        self, elements: List[ContextElement], target_size: int
    ) -> Tuple[List[ContextElement], float]:
        """Extract key points from context elements"""
        compressed_elements = []
        current_size = 0

        # Sort by relevance score (highest first)
        sorted_elements = sorted(
            elements, key=lambda x: x.relevance_score, reverse=True
        )

        for element in sorted_elements:
            # Simple key point extraction: take first sentence or first 100 chars
            sentences = element.content.split(". ")
            keypoint = sentences[0] if sentences else element.content[:100]

            if current_size + len(keypoint) <= target_size:
                compressed_element = ContextElement(
                    element_id=f"{element.element_id}_compressed",
                    content=f"[KEYPOINT] {keypoint}",
                    context_type=element.context_type,
                    relevance_score=element.relevance_score,
                    relevance_level=element.relevance_level,
                    source_engagement_id=element.source_engagement_id,
                    compression_ratio=len(keypoint) / len(element.content),
                )
                compressed_elements.append(compressed_element)
                current_size += len(keypoint)

        original_size = sum(len(e.content) for e in elements)
        actual_ratio = current_size / original_size if original_size > 0 else 0.0

        return compressed_elements, actual_ratio

    def _compress_hierarchical(
        self, elements: List[ContextElement], target_size: int
    ) -> Tuple[List[ContextElement], float]:
        """Hierarchical compression based on relevance levels"""
        # Group by relevance level
        by_level = {}
        for element in elements:
            level = element.relevance_level
            if level not in by_level:
                by_level[level] = []
            by_level[level].append(element)

        compressed_elements = []
        current_size = 0

        # Priority order: CRITICAL > HIGH > MEDIUM > LOW > IRRELEVANT
        priority_order = [
            ContextRelevanceLevel.CRITICAL,
            ContextRelevanceLevel.HIGH,
            ContextRelevanceLevel.MEDIUM,
            ContextRelevanceLevel.LOW,
            ContextRelevanceLevel.IRRELEVANT,
        ]

        for level in priority_order:
            if level not in by_level:
                continue

            for element in by_level[level]:
                if current_size + len(element.content) <= target_size:
                    compressed_elements.append(element)
                    current_size += len(element.content)
                else:
                    # Partial content inclusion for this element
                    remaining_space = target_size - current_size
                    if remaining_space > 50:  # Only if meaningful space left
                        partial_content = (
                            element.content[: remaining_space - 10] + "..."
                        )
                        partial_element = ContextElement(
                            element_id=f"{element.element_id}_partial",
                            content=partial_content,
                            context_type=element.context_type,
                            relevance_score=element.relevance_score,
                            relevance_level=element.relevance_level,
                            source_engagement_id=element.source_engagement_id,
                            compression_ratio=len(partial_content)
                            / len(element.content),
                        )
                        compressed_elements.append(partial_element)
                        current_size = target_size
                    break

            if current_size >= target_size:
                break

        original_size = sum(len(e.content) for e in elements)
        actual_ratio = current_size / original_size if original_size > 0 else 0.0

        return compressed_elements, actual_ratio

    def _compress_summarization(
        self, elements: List[ContextElement], target_size: int
    ) -> Tuple[List[ContextElement], float]:
        """Basic summarization compression (placeholder for LLM-based summarization)"""
        # For Sprint 1.4: Simple truncation-based summarization
        # TODO: Replace with actual LLM summarization in future sprints

        compressed_elements = []
        current_size = 0

        for element in sorted(elements, key=lambda x: x.relevance_score, reverse=True):
            summary_length = min(len(element.content) // 2, target_size - current_size)
            if summary_length > 20:  # Only include if meaningful
                summary = element.content[:summary_length] + "..."

                summary_element = ContextElement(
                    element_id=f"{element.element_id}_summary",
                    content=f"[SUMMARY] {summary}",
                    context_type=element.context_type,
                    relevance_score=element.relevance_score
                    * 0.9,  # Slight penalty for compression
                    relevance_level=element.relevance_level,
                    source_engagement_id=element.source_engagement_id,
                    compression_ratio=summary_length / len(element.content),
                )
                compressed_elements.append(summary_element)
                current_size += summary_length

            if current_size >= target_size:
                break

        original_size = sum(len(e.content) for e in elements)
        actual_ratio = current_size / original_size if original_size > 0 else 0.0

        return compressed_elements, actual_ratio


class ContextTaxonomyManager:
    """
    Master Context Taxonomy Manager

    Integrates all Manus taxonomy components for comprehensive context intelligence
    """

    def __init__(self):
        self.classifier = ContextTaxonomyClassifier()
        self.scorer = ContextRelevanceScorer()
        self.compressor = ContextCompressor()

    def analyze_contexts(
        self,
        context_contents: List[str],
        current_query: str,
        engagement_id: str,
        cognitive_coherence_scores: Optional[List[float]] = None,
    ) -> ContextIntelligenceResult:
        """
        Comprehensive context analysis using full Manus methodology

        Args:
            context_contents: Raw context content strings
            current_query: Current user query/goal
            engagement_id: Engagement identifier
            cognitive_coherence_scores: Optional cognitive coherence scores

        Returns:
            ContextIntelligenceResult with complete analysis
        """
        start_time = datetime.utcnow()

        # Initialize cognitive coherence scores if not provided
        if cognitive_coherence_scores is None:
            cognitive_coherence_scores = [0.0] * len(context_contents)

        # Process each context content
        context_elements = []
        context_scores = []

        for i, content in enumerate(context_contents):
            # Generate unique element ID
            element_id = hashlib.md5(
                f"{engagement_id}:{i}:{content[:50]}".encode()
            ).hexdigest()[:16]

            # Classify context type
            context_type, type_confidence = self.classifier.classify_context(content)

            # Create context element
            element = ContextElement(
                element_id=element_id,
                content=content,
                context_type=context_type,
                relevance_score=0.0,  # Will be set after scoring
                relevance_level=ContextRelevanceLevel.MEDIUM,  # Default, will be updated
                source_engagement_id=engagement_id,
                cognitive_coherence_score=cognitive_coherence_scores[i],
            )

            # Score relevance
            relevance_score = self.scorer.score_context_relevance(
                element, current_query, cognitive_coherence_scores[i]
            )

            # Update element with scoring results
            element.relevance_score = relevance_score.overall_score
            element.relevance_level = self.scorer.categorize_relevance_level(
                relevance_score.overall_score
            )

            context_elements.append(element)
            context_scores.append(relevance_score)

        # Calculate analysis statistics
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000  # ms

        contexts_by_type = {}
        contexts_by_relevance = {}

        for element in context_elements:
            # Count by type
            if element.context_type in contexts_by_type:
                contexts_by_type[element.context_type] += 1
            else:
                contexts_by_type[element.context_type] = 1

            # Count by relevance
            if element.relevance_level in contexts_by_relevance:
                contexts_by_relevance[element.relevance_level] += 1
            else:
                contexts_by_relevance[element.relevance_level] = 1

        # Find dominant context type
        dominant_type = max(contexts_by_type.keys(), key=lambda k: contexts_by_type[k])

        # Calculate averages
        avg_relevance = (
            sum(e.relevance_score for e in context_elements) / len(context_elements)
            if context_elements
            else 0.0
        )
        avg_coherence = (
            sum(e.cognitive_coherence_score for e in context_elements)
            / len(context_elements)
            if context_elements
            else 0.0
        )

        # Get top scoring contexts
        top_contexts = sorted(
            context_elements, key=lambda x: x.relevance_score, reverse=True
        )[:5]

        # Generate recommendations
        compression_candidates = [
            e.element_id
            for e in context_elements
            if e.relevance_level
            in [ContextRelevanceLevel.LOW, ContextRelevanceLevel.IRRELEVANT]
        ]

        archival_candidates = [
            e.element_id
            for e in context_elements
            if e.relevance_level == ContextRelevanceLevel.IRRELEVANT
        ]

        prefetch_recommendations = [
            e.element_id
            for e in context_elements
            if e.relevance_level
            in [ContextRelevanceLevel.CRITICAL, ContextRelevanceLevel.HIGH]
        ]

        return ContextIntelligenceResult(
            engagement_id=engagement_id,
            total_contexts_analyzed=len(context_elements),
            contexts_by_type=contexts_by_type,
            contexts_by_relevance=contexts_by_relevance,
            average_relevance_score=avg_relevance,
            highest_scoring_contexts=top_contexts,
            dominant_context_type=dominant_type,
            cognitive_coherence_average=avg_coherence,
            cache_distribution={},  # Will be populated by cache layers
            processing_time_ms=int(processing_time),
            compression_candidates=compression_candidates,
            archival_candidates=archival_candidates,
            prefetch_recommendations=prefetch_recommendations,
        )
