"""
Next-Generation Query Chunker - Strategic Decomposition Engine
=============================================================

Research-validated strategic query decomposition system that transforms queries
from "keyword sorting" to "strategic understanding" using:

1. MECE Decomposition (Mutually Exclusive, Collectively Exhaustive)
2. First-Principles Constraint Separation
3. Natural Boundary Detection
4. Unknowns as First-Class Citizens
5. Real-Time Quality Monitoring
6. Adaptive Re-chunking

Integrates with existing enhanced query classifier as an optional upgrade.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

# Core components
from src.core.strategic_query_decomposer import (
    MECEDecomposition,
    get_strategic_query_decomposer,
)
from src.core.boundary_detection_engine import (
    get_boundary_detection_engine,
)
from src.core.chunking_quality_monitor import (
    QualityAssessment,
    ProcessingContext,
    get_chunking_quality_monitor,
)
from src.core.unified_context_stream import UnifiedContextStream, ContextEventType

# Existing system integration
try:
    from src.engine.engines.selection.enhanced_query_classifier import (
        EnhancedQueryClassifier,
        QueryAnalysis,
        get_enhanced_classifier,
    )

    ENHANCED_CLASSIFIER_AVAILABLE = True
except ImportError:
    ENHANCED_CLASSIFIER_AVAILABLE = False
    logger.warning("⚠️ Enhanced query classifier not available")

logger = logging.getLogger(__name__)


class ChunkingMode(Enum):
    """Different modes for query chunking"""

    LEGACY = "legacy"  # Use existing enhanced classifier only
    STRATEGIC = "strategic"  # Use new strategic decomposition
    HYBRID = "hybrid"  # Combine both approaches
    ADAPTIVE = "adaptive"  # Automatically choose best approach


@dataclass
class ChunkingResult:
    """Complete result from next-generation query chunking"""

    query_id: str
    original_query: str
    mode_used: ChunkingMode

    # Strategic decomposition (if used)
    strategic_decomposition: Optional[MECEDecomposition] = None

    # Legacy analysis (if used)
    legacy_analysis: Optional[QueryAnalysis] = None

    # Quality assessment
    quality_assessment: Optional[QualityAssessment] = None

    # Processing metadata
    processing_time_ms: int = 0
    confidence_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)

    # Integration data for downstream systems
    integration_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "original_query": self.original_query,
            "mode_used": self.mode_used.value,
            "strategic_decomposition": (
                self.strategic_decomposition.to_dict()
                if self.strategic_decomposition
                else None
            ),
            "legacy_analysis": (
                self.legacy_analysis.__dict__ if self.legacy_analysis else None
            ),
            "quality_assessment": (
                self.quality_assessment.to_dict() if self.quality_assessment else None
            ),
            "processing_time_ms": self.processing_time_ms,
            "confidence_score": self.confidence_score,
            "recommendations": self.recommendations,
            "integration_data": self.integration_data,
        }




# Facade with architectural seams
from src.core.chunking.contracts import IChunkingStrategy, IChunkingEvaluator, IChunkingFinalizer


class NextGenQueryChunker:
    """
    Facade for the Next-Generation Query Chunker with architectural seams.

    The facade orchestrates injected strategy and evaluator services directly.
    """

    def __init__(
        self,
        strategy: Optional[IChunkingStrategy] = None,
        evaluator: Optional[IChunkingEvaluator] = None,
        finalizer: Optional[IChunkingFinalizer] = None,
    ) -> None:
        # Support direct construction with no args for backward compatibility
        if strategy is None or evaluator is None:
            # Lazy import to avoid import cycles
            from src.services.chunking.facade_implementations import (
                V1ChunkingStrategy,
                V1ChunkingEvaluator,
                V1ChunkingFinalizer,
            )
            strategy = strategy or V1ChunkingStrategy()
            evaluator = evaluator or V1ChunkingEvaluator()
            finalizer = finalizer or V1ChunkingFinalizer()
        else:
            if finalizer is None:
                from src.services.chunking.facade_implementations import V1ChunkingFinalizer
                finalizer = V1ChunkingFinalizer()

        self._strategy = strategy
        self._evaluator = evaluator
        self._finalizer = finalizer

        # Facade state
        self.context_stream: Optional[UnifiedContextStream] = None
        self.default_context = ProcessingContext()

    async def chunk_query(
        self,
        query: str,
        mode: ChunkingMode = ChunkingMode.ADAPTIVE,
        context: Optional[ProcessingContext] = None,
        user_context: Optional[Dict[str, Any]] = None,
    ) -> ChunkingResult:
        start_time = time.time()
        if context is None:
            context = self.default_context

        try:
            # 1) Select processing mode
            actual_mode = await self._select_processing_mode(query, mode, context)

            # 2) Execute decomposition/analysis according to mode
            result = ChunkingResult(
                query_id=str(int(time.time() * 1000)),
                original_query=query,
                mode_used=actual_mode,
            )

            strategic_task = None
            legacy_task = None

            if actual_mode == ChunkingMode.STRATEGIC:
                strategic_task = asyncio.create_task(self._strategy.decompose(query, user_context))
            elif actual_mode == ChunkingMode.LEGACY:
                if ENHANCED_CLASSIFIER_AVAILABLE:
                    try:
                        classifier = get_enhanced_classifier()
                        legacy_task = asyncio.create_task(classifier.analyze_query(query, user_context))
                    except Exception:
                        # Fall back to strategic
                        strategic_task = asyncio.create_task(self._strategy.decompose(query, user_context))
                        result.mode_used = ChunkingMode.STRATEGIC
                else:
                    strategic_task = asyncio.create_task(self._strategy.decompose(query, user_context))
                    result.mode_used = ChunkingMode.STRATEGIC
            elif actual_mode == ChunkingMode.HYBRID:
                strategic_task = asyncio.create_task(self._strategy.decompose(query, user_context))
                if ENHANCED_CLASSIFIER_AVAILABLE:
                    try:
                        classifier = get_enhanced_classifier()
                        legacy_task = asyncio.create_task(classifier.analyze_query(query, user_context))
                    except Exception:
                        legacy_task = None
            else:
                strategic_task = asyncio.create_task(self._strategy.decompose(query, user_context))
                result.mode_used = ChunkingMode.STRATEGIC

            # Gather whichever tasks are running
            tasks = [t for t in [strategic_task, legacy_task] if t is not None]
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                idx = 0
                if strategic_task is not None:
                    v = results[idx]
                    idx += 1
                    if not isinstance(v, Exception):
                        result.strategic_decomposition = v
                if legacy_task is not None and idx < len(results):
                    v = results[idx]
                    if not isinstance(v, Exception):
                        result.legacy_analysis = v

            # 3) Quality assessment and potential re-chunking (strategy-only)
            if result.strategic_decomposition:
                qa = await self._evaluator.assess(result.strategic_decomposition, context)
                result.quality_assessment = qa

                if qa.should_rechunk and context.available_resources > 0.5:
                    improved = await self._execute_rechunking(query, result, context)
                    if improved:
                        result = improved

            # 4) Finalization: integration data, scoring, recommendations
            self._finalizer.finalize(result)
            processing_time = int((time.time() - start_time) * 1000)
            result.processing_time_ms = processing_time

            # 5) Context logging
            if self.context_stream:
                self._log_chunking_result(result)

            logger.info(
                f"🎯 Query chunked: mode={result.mode_used.value}, "
                f"confidence={result.confidence_score:.2f}, time={result.processing_time_ms}ms"
            )
            return result

        except Exception as e:
            logger.error(f"❌ Query chunking failed: {e}")
            return ChunkingResult(
                query_id=str(time.time()),
                original_query=query,
                mode_used=ChunkingMode.LEGACY,
                processing_time_ms=int((time.time() - start_time) * 1000),
                confidence_score=0.3,
            )

    async def _select_processing_mode(
        self, query: str, requested_mode: ChunkingMode, context: ProcessingContext
    ) -> ChunkingMode:
        """
        Select the optimal processing mode based on query characteristics and context.

        Adaptive mode selection considers:
        1. Query complexity and stakes
        2. Available processing time and resources
        3. User timeline pressure
        4. Historical performance of different modes
        """

        if requested_mode != ChunkingMode.ADAPTIVE:
            return requested_mode

        # Analyze query characteristics for mode selection
        query_length = len(query.split())
        has_complex_concepts = any(
            word in query.lower()
            for word in [
                "strategy",
                "system",
                "optimize",
                "transform",
                "analyze",
                "framework",
            ]
        )

        # Decision logic for adaptive mode selection
        if context.decision_stakes > 0.7 or has_complex_concepts:
            # High stakes or complex query - use strategic mode
            if context.available_resources > 0.5:
                return ChunkingMode.STRATEGIC
            else:
                return (
                    ChunkingMode.HYBRID
                )  # Resource-constrained but still need quality

        elif context.user_timeline_pressure > 0.7:
            # High time pressure - use fastest available mode
            if ENHANCED_CLASSIFIER_AVAILABLE:
                return ChunkingMode.LEGACY
            else:
                return ChunkingMode.STRATEGIC

        elif query_length > 20:
            # Long query - strategic decomposition likely beneficial
            return ChunkingMode.STRATEGIC

        else:
            # Default to hybrid approach
            return (
                ChunkingMode.HYBRID
                if ENHANCED_CLASSIFIER_AVAILABLE
                else ChunkingMode.STRATEGIC
            )

    async def _execute_rechunking(
        self, query: str, original_result: ChunkingResult, context: ProcessingContext
    ) -> Optional[ChunkingResult]:
        """
        Execute re-chunking with improved parameters based on quality assessment.

        Uses lessons from the quality assessment to adjust the decomposition approach.
        """

        try:
            logger.info("🔄 Executing adaptive re-chunking")

            # Analyze what went wrong in the original chunking
            quality_issues = original_result.quality_assessment.rechunk_triggers

            # Adjust processing based on identified issues
            enhanced_context: Dict[str, Any] = {}

            if "low_coverage" in [t.value for t in quality_issues]:
                enhanced_context["focus_on_coverage"] = True

            if "high_overlap" in [t.value for t in quality_issues]:
                enhanced_context["emphasize_separation"] = True

            if "poor_boundaries" in [t.value for t in quality_issues]:
                enhanced_context["strengthen_boundaries"] = True

            # Re-execute with enhanced context via strategy
            improved_decomposition = await self._strategy.decompose(query, enhanced_context)

            # Create improved result
            improved_result = ChunkingResult(
                query_id=original_result.query_id + "_rechunked",
                original_query=query,
                mode_used=original_result.mode_used,
                strategic_decomposition=improved_decomposition,
                legacy_analysis=original_result.legacy_analysis,
            )

            # Assess if the re-chunking actually improved quality
            improved_quality = await self._evaluator.assess(improved_decomposition, context)

            if (
                improved_quality.overall_quality_score
                > original_result.quality_assessment.overall_quality_score
            ):
                logger.info(
                    f"✅ Re-chunking improved quality: "
                    f"{original_result.quality_assessment.overall_quality_score:.2f} → "
                    f"{improved_quality.overall_quality_score:.2f}"
                )
                improved_result.quality_assessment = improved_quality
                return improved_result
            else:
                logger.info("⚠️ Re-chunking did not improve quality, keeping original")
                return None

        except Exception as e:
            logger.error(f"❌ Re-chunking failed: {e}")
            return None











    def _log_chunking_result(self, result: ChunkingResult):
        """Log chunking result to context stream"""

        if not self.context_stream:
            return

        event_data = {
            "query_id": result.query_id,
            "mode_used": result.mode_used.value,
            "processing_time_ms": result.processing_time_ms,
            "confidence_score": result.confidence_score,
            "components_identified": {},
        }

        if result.strategic_decomposition:
            event_data["components_identified"] = {
                "constraints": len(result.strategic_decomposition.constraints),
                "conventions": len(result.strategic_decomposition.conventions),
                "decisions": len(result.strategic_decomposition.decisions),
                "unknowns": len(result.strategic_decomposition.unknowns),
                "success_metrics": len(result.strategic_decomposition.success_metrics),
            }

        self.context_stream.add_event(
            ContextEventType.QUERY_RECEIVED,
            event_data,
            {
                "chunking_approach": "next_generation_strategic",
                "quality_score": (
                    result.quality_assessment.overall_quality_score
                    if result.quality_assessment
                    else None
                ),
            },
        )

    def set_context_stream(self, context_stream: UnifiedContextStream):
        # Propagate to injected services where supported
        if hasattr(self._strategy, "set_context_stream"):
            try:
                self._strategy.set_context_stream(context_stream)  # type: ignore[attr-defined]
            except Exception:
                pass
        if hasattr(self._evaluator, "set_context_stream"):
            try:
                self._evaluator.set_context_stream(context_stream)  # type: ignore[attr-defined]
            except Exception:
                pass
        self.context_stream = context_stream

    def set_default_context(self, context: ProcessingContext):
        self.default_context = context


# Global instance
_next_gen_chunker: Optional[NextGenQueryChunker] = None


def get_next_gen_query_chunker() -> NextGenQueryChunker:
    """Get or create the global next-generation query chunker instance"""
    global _next_gen_chunker
    if _next_gen_chunker is None:
        # Lazy import to avoid cycles at import time
        from src.services.chunking.facade_implementations import (
            V1ChunkingStrategy,
            V1ChunkingEvaluator,
        )

        _next_gen_chunker = NextGenQueryChunker(
            strategy=V1ChunkingStrategy(), evaluator=V1ChunkingEvaluator()
        )
    return _next_gen_chunker
