"""
Context Intelligence Service Contracts
A1 - Discovery & Taxonomy (Red Team Amendment Applied)
"""
from typing import Protocol, Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

# Red Team Amendment: Error taxonomy with concrete mapping
class ContextIntelligenceError(Exception):
    """Base exception for context intelligence operations"""
    pass

class ProviderError(ContextIntelligenceError):
    """External service failures (Redis, Supabase)"""
    pass

class ParseError(ContextIntelligenceError):
    """Data format issues in context parsing"""
    pass

class ValidationError(ContextIntelligenceError):
    """Business rule violations in context validation"""
    pass

class TimeoutError(ContextIntelligenceError):
    """Operation timeouts in context processing"""
    pass

class CancellationError(ContextIntelligenceError):
    """Client disconnection during context operations"""
    pass

@dataclass
class ContextPacket:
    """Input context packet for intelligence analysis"""
    content: str
    metadata: Dict[str, Any]
    session_id: Optional[str] = None
    trace_id: Optional[str] = None

@dataclass
class IntelligenceReport:
    """Output intelligence report with partial result support"""
    relevant_contexts: List[Dict[str, Any]]
    relevance_scores: List[float]
    semantic_similarities: List[float]
    usage_frequencies: List[float]
    temporal_recencies: List[float]
    cognitive_exhaust_contexts: List[Dict[str, Any]]
    engine_stats: Dict[str, Any]
    
    # Red Team Amendment: Partial result semantics
    is_partial: bool = False
    missing_providers: List[str] = None
    error_summary: Optional[str] = None

class IContextIntelligenceEngine(Protocol):
    """Main context intelligence engine interface"""
    
    async def analyze_context(self, packet: ContextPacket) -> IntelligenceReport:
        """Analyze context and extract intelligence with resiliency"""
        ...
    
    async def get_relevant_context(
        self, 
        query: str, 
        cognitive_exhaust_contexts: List[Dict[str, Any]], 
        top_k: int = 5
    ) -> IntelligenceReport:
        """Get relevant context with cognitive exhaust integration"""
        ...
    
    async def score_context_relevance(
        self, 
        context: str, 
        target: str, 
        cognitive_exhaust: Optional[List[Dict[str, Any]]] = None
    ) -> float:
        """Score context relevance with cognitive enhancement"""
        ...

class IContextProviderClient(Protocol):
    """External context provider interface"""
    
    async def fetch_contexts(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Fetch contexts from external provider with circuit breaker"""
        ...
    
    async def health_check(self) -> bool:
        """Check provider health status"""
        ...

class ICognitionCache(Protocol):
    """Cognition cache interface across L1/L2/L3"""
    
    async def get_mental_model(self, key: str) -> Optional[Dict[str, Any]]:
        """Get mental model from cache"""
        ...
    
    async def set_mental_model(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """Set mental model in cache"""
        ...
    
    async def get_cognitive_exhaust_contexts(self, query: str) -> List[Dict[str, Any]]:
        """Get cognitive exhaust contexts"""
        ...
    
    async def store_cognitive_exhaust(self, context: Dict[str, Any]) -> None:
        """Store cognitive exhaust"""
        ...
    
    async def close(self) -> None:
        """Close cache connections"""
        ...