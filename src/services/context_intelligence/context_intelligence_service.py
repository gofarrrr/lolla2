"""
Context Intelligence Service - Business Logic Extraction
A2 - Contracts & Service Extraction (Red Team Amendment Applied)
"""
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from .contracts import (
    IContextIntelligenceEngine, ContextPacket, IntelligenceReport,
    ICognitionCache, IContextProviderClient
)
from .error_taxonomy import ContextIntelligenceErrorMapper, CircuitBreaker
from src.core.async_helpers import timeout, bounded, monitor_slow_calls
from src.core.unified_context_stream import UnifiedContextStream, ContextEventType

logger = logging.getLogger(__name__)

class ContextIntelligenceService:
    """Core business logic for context intelligence (extracted from engine)"""
    
    def __init__(
        self,
        l1_cache: ICognitionCache,
        l2_cache: Optional[ICognitionCache] = None,
        l3_cache: Optional[ICognitionCache] = None,
        context_stream: Optional[UnifiedContextStream] = None,
        max_concurrent: int = 10
    ):
        self.l1_cache = l1_cache
        self.l2_cache = l2_cache
        self.l3_cache = l3_cache
        self.context_stream = context_stream
        self.error_mapper = ContextIntelligenceErrorMapper()
        
        # Red Team Amendment: Circuit breakers per provider
        self.circuit_breakers = {
            'l1': CircuitBreaker(failure_threshold=3, recovery_timeout=30),
            'l2': CircuitBreaker(failure_threshold=5, recovery_timeout=60),
            'l3': CircuitBreaker(failure_threshold=5, recovery_timeout=120),
        }
        
        # Structured async
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.logger = logger
    
    async def analyze_context(self, packet: ContextPacket) -> IntelligenceReport:
        """Main context analysis with resiliency and partial results"""
        operation_start = datetime.now()
        
        try:
            # Start observability
            if self.context_stream:
                await self.context_stream.emit_event(
                    event_type=ContextEventType.ANALYSIS_STARTED,
                    details={
                        "operation": "analyze_context",
                        "session_id": packet.session_id,
                        "has_metadata": bool(packet.metadata)
                    },
                    correlation_id=packet.session_id,
                    trace_id=packet.trace_id
                )
            
            # Extract contexts from all available cache levels with circuit breaker protection
            contexts_l1, contexts_l2, contexts_l3 = await self._fetch_contexts_with_resilience(
                packet.content, packet.session_id, packet.trace_id
            )
            
            # Combine all contexts
            all_contexts = contexts_l1 + contexts_l2 + contexts_l3
            
            # Calculate relevance scores (extracted from original complex method)
            scored_contexts = await self._calculate_relevance_scores(
                packet.content, all_contexts, packet.trace_id
            )
            
            # Build intelligence report with partial result support
            report = await self._build_intelligence_report(
                scored_contexts, 
                operation_start,
                {
                    'l1_available': len(contexts_l1) > 0,
                    'l2_available': len(contexts_l2) > 0, 
                    'l3_available': len(contexts_l3) > 0,
                }
            )
            
            # Success observability
            if self.context_stream:
                await self.context_stream.emit_event(
                    event_type=ContextEventType.ANALYSIS_COMPLETE,
                    details={
                        "operation": "analyze_context",
                        "contexts_found": len(all_contexts),
                        "processing_time": (datetime.now() - operation_start).total_seconds(),
                        "is_partial": report.is_partial
                    },
                    correlation_id=packet.session_id,
                    trace_id=packet.trace_id
                )
            
            return report
            
        except Exception as exc:
            # Map and handle error
            mapped_error = self.error_mapper.map_exception(exc)
            
            if self.context_stream:
                await self.error_mapper.emit_error_event(
                    self.context_stream,
                    mapped_error,
                    "analyze_context",
                    packet.session_id,
                    packet.trace_id,
                    {"processing_time": (datetime.now() - operation_start).total_seconds()}
                )
            
            raise mapped_error
    
    async def _fetch_contexts_with_resilience(
        self, query: str, session_id: Optional[str], trace_id: Optional[str]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Fetch contexts from all cache levels with circuit breaker protection"""
        
        async def fetch_l1() -> List[Dict[str, Any]]:
            if not self.circuit_breakers['l1'].can_execute():
                return []
            try:
                result = await timeout(
                    self.l1_cache.get_cognitive_exhaust_contexts(query), 
                    seconds=2.0
                )
                self.circuit_breakers['l1'].record_success()
                return result or []
            except Exception as e:
                self.circuit_breakers['l1'].record_failure()
                logger.warning(f"L1 cache failed: {e}")
                return []
        
        async def fetch_l2() -> List[Dict[str, Any]]:
            if not self.l2_cache or not self.circuit_breakers['l2'].can_execute():
                return []
            try:
                result = await timeout(
                    self.l2_cache.get_cognitive_exhaust_contexts(query),
                    seconds=5.0
                )
                self.circuit_breakers['l2'].record_success()
                return result or []
            except Exception as e:
                self.circuit_breakers['l2'].record_failure()
                logger.warning(f"L2 cache failed: {e}")
                return []
        
        async def fetch_l3() -> List[Dict[str, Any]]:
            if not self.l3_cache or not self.circuit_breakers['l3'].can_execute():
                return []
            try:
                result = await timeout(
                    self.l3_cache.get_cognitive_exhaust_contexts(query),
                    seconds=10.0
                )
                self.circuit_breakers['l3'].record_success()
                return result or []
            except Exception as e:
                self.circuit_breakers['l3'].record_failure()
                logger.warning(f"L3 cache failed: {e}")
                return []
        
        # Execute all fetches concurrently with bounded semaphore
        contexts_l1, contexts_l2, contexts_l3 = await asyncio.gather(
            bounded(self.semaphore, fetch_l1()),
            bounded(self.semaphore, fetch_l2()),
            bounded(self.semaphore, fetch_l3()),
            return_exceptions=False  # Let individual handlers manage exceptions
        )
        
        return contexts_l1, contexts_l2, contexts_l3
    
    async def _calculate_relevance_scores(
        self, query: str, contexts: List[Dict[str, Any]], trace_id: Optional[str]
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Calculate relevance scores for contexts (extracted complex logic)"""
        
        async def score_context(context: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
            try:
                # Simple relevance scoring (placeholder for complex algorithm)
                content = context.get('content', '')
                
                # Basic semantic similarity (would use embeddings in production)
                query_words = set(query.lower().split())
                content_words = set(content.lower().split())
                similarity = len(query_words & content_words) / max(len(query_words | content_words), 1)
                
                # Weight by usage frequency if available
                frequency_weight = context.get('usage_frequency', 0.5)
                
                # Final score
                score = similarity * 0.7 + frequency_weight * 0.3
                
                return context, min(score, 1.0)
                
            except Exception as e:
                logger.warning(f"Scoring failed for context: {e}")
                return context, 0.0
        
        # Score all contexts concurrently
        scoring_tasks = [
            bounded(self.semaphore, score_context(ctx)) for ctx in contexts
        ]
        
        if scoring_tasks:
            scored_results = await monitor_slow_calls(
                asyncio.gather(*scoring_tasks, return_exceptions=True),
                "context_scoring_batch",
                p95_threshold=1.0
            )
            
            # Filter out exceptions
            return [result for result in scored_results if isinstance(result, tuple)]
        
        return []
    
    async def _build_intelligence_report(
        self, 
        scored_contexts: List[Tuple[Dict[str, Any], float]], 
        operation_start: datetime,
        cache_availability: Dict[str, bool]
    ) -> IntelligenceReport:
        """Build intelligence report with partial result semantics"""
        
        # Sort by relevance score
        scored_contexts.sort(key=lambda x: x[1], reverse=True)
        
        # Extract top results
        top_contexts = scored_contexts[:10]  # Limit results
        
        relevant_contexts = [ctx for ctx, score in top_contexts]
        relevance_scores = [score for ctx, score in top_contexts]
        
        # Check if this is a partial result
        failed_caches = [cache for cache, available in cache_availability.items() if not available]
        is_partial = len(failed_caches) > 0
        
        return IntelligenceReport(
            relevant_contexts=relevant_contexts,
            relevance_scores=relevance_scores,
            semantic_similarities=relevance_scores,  # Simplified for now
            usage_frequencies=[ctx.get('usage_frequency', 0.0) for ctx in relevant_contexts],
            temporal_recencies=[ctx.get('temporal_recency', 0.0) for ctx in relevant_contexts],
            cognitive_exhaust_contexts=relevant_contexts,
            engine_stats={
                "processing_time": (datetime.now() - operation_start).total_seconds(),
                "contexts_processed": len(scored_contexts),
                "cache_levels_used": len([c for c in cache_availability.values() if c])
            },
            is_partial=is_partial,
            missing_providers=failed_caches if is_partial else [],
            error_summary=f"Failed cache levels: {failed_caches}" if is_partial else None
        )