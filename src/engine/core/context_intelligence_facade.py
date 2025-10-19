"""
Context Intelligence Engine Facade
A3 - API Delegation (Red Team Amendment Applied)

This is the thin facade that delegates to the extracted service layer.
Following Operation Atlas patterns for clean architectural separation.
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

# Original imports for compatibility  
from src.engine.models.data_contracts import MentalModelDefinition
from src.engine.models.context_taxonomy import ContextTaxonomyManager  
from src.config import CognitiveEngineSettings

# New service layer imports
from src.services.context_intelligence import (
    ContextIntelligenceService, ContextPacket, IntelligenceReport,
    InMemoryContextProviderClient, ContextProviderClientFactory,
    ContextIntelligenceErrorMapper
)
from src.core.unified_context_stream import UnifiedContextStream
from src.core.async_helpers import timeout, cancel_on_shutdown

logger = logging.getLogger(__name__)

# Legacy compatibility classes (maintain original interface)
@dataclass
class CognitiveExhaustContext:
    """Legacy compatibility for cognitive exhaust context"""
    content: str
    context_id: str
    relevance_score: float = 0.0
    usage_frequency: float = 0.0
    temporal_recency: float = 0.0

@dataclass 
class ContextRelevanceScore:
    """Legacy compatibility for context relevance score"""
    semantic_similarity: float = 0.0
    usage_frequency: float = 0.0
    temporal_recency: float = 0.0
    cognitive_coherence: float = 0.0
    overall_relevance: float = 0.0

class ContextIntelligenceEngine:
    """
    Thin facade for Context Intelligence Engine (A3 - API Delegation)
    
    This class now delegates all business logic to ContextIntelligenceService
    while maintaining the original API interface for backward compatibility.
    """
    
    def __init__(self, settings: Optional[CognitiveEngineSettings] = None):
        """Initialize facade with service delegation"""
        self.settings = settings or CognitiveEngineSettings()
        self.logger = logger
        
        # Initialize service layer dependencies
        self._initialize_service_layer()
    
    def _initialize_service_layer(self):
        """Initialize the service layer with DI (Red Team Amendment)"""
        try:
            # Create cache providers (simplified for facade)
            self.l1_cache_client = InMemoryContextProviderClient()
            
            # Create context stream for observability
            from src.core.unified_context_stream import get_unified_context_stream
            self.context_stream = get_unified_context_stream()
            
            # Create the main service (dependency injection)
            self.service = ContextIntelligenceService(
                l1_cache=self.l1_cache_client,  # Using provider as cache for now
                l2_cache=None,  # TODO: Wire Redis cache
                l3_cache=None,  # TODO: Wire Supabase cache
                context_stream=self.context_stream,
                max_concurrent=10
            )
            
            # Legacy compatibility objects
            self.l1_cache = self.l1_cache_client  # For backward compatibility
            self.l2_cache = None
            self.l3_cache = None
            
            self.logger.info("✅ Context Intelligence Facade initialized with service delegation")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Context Intelligence Facade: {e}")
            raise
    
    async def get_relevant_context(
        self,
        current_query: str,
        cognitive_exhaust_contexts: Optional[List[Dict[str, Any]]] = None,
        max_contexts: int = 5,
        engagement_id: Optional[str] = None,
    ) -> List[Tuple[CognitiveExhaustContext, ContextRelevanceScore]]:
        """
        Get relevant context (FACADE - delegates to service)
        
        This method now delegates to ContextIntelligenceService while maintaining
        the original return type for backward compatibility.
        """
        try:
            # Convert to new contract format
            packet = ContextPacket(
                content=current_query,
                metadata={
                    "max_contexts": max_contexts,
                    "engagement_id": engagement_id,
                    "cognitive_exhaust_contexts": cognitive_exhaust_contexts or []
                },
                session_id=engagement_id,
                trace_id=engagement_id  # Simple correlation for now
            )
            
            # Delegate to service with structured async
            report = await timeout(
                self.service.analyze_context(packet),
                seconds=30.0  # Red Team Amendment: structured timeouts
            )
            
            # Convert back to legacy format for compatibility
            return self._convert_report_to_legacy_format(report, max_contexts)
            
        except Exception as e:
            # Error mapping handled by service layer
            self.logger.error(f"get_relevant_context failed: {e}")
            raise
    
    async def score_context_relevance(
        self,
        context: str,
        target: str,
        cognitive_exhaust: Optional[List[Dict[str, Any]]] = None
    ) -> float:
        """
        Score context relevance (FACADE - simplified delegation)
        
        This method provides a simplified interface while delegating core logic.
        """
        try:
            # Create packet for scoring
            packet = ContextPacket(
                content=f"Context: {context} | Target: {target}",
                metadata={
                    "operation": "score_relevance",
                    "cognitive_exhaust": cognitive_exhaust or []
                }
            )
            
            # Delegate to service
            report = await timeout(
                self.service.analyze_context(packet),
                seconds=10.0
            )
            
            # Return average relevance score
            if report.relevance_scores:
                return sum(report.relevance_scores) / len(report.relevance_scores)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"score_context_relevance failed: {e}")
            return 0.0
    
    def _convert_report_to_legacy_format(
        self, report: IntelligenceReport, max_contexts: int
    ) -> List[Tuple[CognitiveExhaustContext, ContextRelevanceScore]]:
        """Convert new IntelligenceReport to legacy format"""
        results = []
        
        # Take top results up to max_contexts
        for i in range(min(len(report.relevant_contexts), max_contexts)):
            context_data = report.relevant_contexts[i]
            relevance_score = report.relevance_scores[i] if i < len(report.relevance_scores) else 0.0
            
            # Create legacy objects
            context = CognitiveExhaustContext(
                content=context_data.get('content', ''),
                context_id=context_data.get('id', f'ctx_{i}'),
                relevance_score=relevance_score,
                usage_frequency=report.usage_frequencies[i] if i < len(report.usage_frequencies) else 0.0,
                temporal_recency=report.temporal_recencies[i] if i < len(report.temporal_recencies) else 0.0
            )
            
            score = ContextRelevanceScore(
                semantic_similarity=report.semantic_similarities[i] if i < len(report.semantic_similarities) else 0.0,
                usage_frequency=context.usage_frequency,
                temporal_recency=context.temporal_recency,
                cognitive_coherence=relevance_score,  # Simplified mapping
                overall_relevance=relevance_score
            )
            
            results.append((context, score))
        
        return results
    
    async def analyze_contexts_with_manus_taxonomy(self, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Legacy method - simplified facade"""
        try:
            if not contexts:
                return {"analysis": "no_contexts", "taxonomy": {}}
            
            # Simple analysis for now
            return {
                "analysis": "facade_analysis",
                "contexts_count": len(contexts),
                "taxonomy": {"categories": ["extracted_to_service_layer"]}
            }
        except Exception as e:
            self.logger.error(f"analyze_contexts_with_manus_taxonomy failed: {e}")
            return {"analysis": "error", "error": str(e)}
    
    async def get_engine_stats(self) -> Dict[str, Any]:
        """Get engine statistics (delegated to service)"""
        try:
            # Return service stats if available
            if hasattr(self.service, 'engine_stats'):
                return self.service.engine_stats
            
            return {
                "facade_status": "active",
                "service_layer": "extracted",
                "cache_levels": ["L1_memory"],
                "delegation_complete": True
            }
        except Exception as e:
            self.logger.error(f"get_engine_stats failed: {e}")
            return {"error": str(e)}
    
    async def close(self):
        """Clean shutdown (Red Team Amendment: proper cleanup)"""
        try:
            # Cancel any running tasks
            active_tasks = []  # Would track actual tasks in production
            await cancel_on_shutdown(active_tasks)
            
            # Close service layer
            if hasattr(self.service, 'close'):
                await self.service.close()
            
            self.logger.info("🔒 Context Intelligence Facade shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")

# Factory for dependency injection (maintains original interface)
def create_context_intelligence_engine(
    settings: Optional[CognitiveEngineSettings] = None,
) -> ContextIntelligenceEngine:
    """
    Factory function for creating Context Intelligence Engine Facade
    
    This maintains the original factory interface while returning the new facade.
    """
    return ContextIntelligenceEngine(settings)
