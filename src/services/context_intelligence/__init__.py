"""
Context Intelligence Service Package
A2 - Contracts & Service Extraction Complete
"""
from .contracts import (
    IContextIntelligenceEngine, ContextPacket, IntelligenceReport,
    ICognitionCache, IContextProviderClient,
    ProviderError, ParseError, ValidationError, TimeoutError, CancellationError
)
from .context_intelligence_service import ContextIntelligenceService
from .error_taxonomy import ContextIntelligenceErrorMapper, CircuitBreaker
from .providers import (
    RedisContextProviderClient, SupabaseContextProviderClient,
    InMemoryContextProviderClient, ContextProviderClientFactory
)

__all__ = [
    # Contracts
    'IContextIntelligenceEngine', 'ContextPacket', 'IntelligenceReport',
    'ICognitionCache', 'IContextProviderClient',
    # Errors
    'ProviderError', 'ParseError', 'ValidationError', 'TimeoutError', 'CancellationError',
    # Service
    'ContextIntelligenceService',
    # Error handling
    'ContextIntelligenceErrorMapper', 'CircuitBreaker',
    # Providers
    'RedisContextProviderClient', 'SupabaseContextProviderClient',
    'InMemoryContextProviderClient', 'ContextProviderClientFactory'
]