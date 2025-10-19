"""
Context Provider Client - External Integration Abstraction
A2 - Contracts & Service Extraction (Provider Abstraction)
"""
import logging
import asyncio
from typing import Dict, List, Optional, Any
from ..contracts import IContextProviderClient, ProviderError, TimeoutError
from ..error_taxonomy import CircuitBreaker
from src.core.async_helpers import timeout, monitor_slow_calls

logger = logging.getLogger(__name__)

class RedisContextProviderClient:
    """Redis-backed context provider with circuit breaker"""
    
    def __init__(self, redis_client, circuit_breaker: Optional[CircuitBreaker] = None):
        self.redis_client = redis_client
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.logger = logger
    
    async def fetch_contexts(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Fetch contexts from Redis with resilience"""
        if not self.circuit_breaker.can_execute():
            raise ProviderError("Redis circuit breaker is OPEN")
        
        try:
            # Build Redis query
            query_key = f"contexts:{query.lower().replace(' ', '_')}"
            
            result = await timeout(
                self.redis_client.get(query_key),
                seconds=kwargs.get('timeout', 5.0)
            )
            
            if result:
                import json
                contexts = json.loads(result)
                self.circuit_breaker.record_success()
                return contexts
            
            self.circuit_breaker.record_success()
            return []
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            if isinstance(e, asyncio.TimeoutError):
                raise TimeoutError(f"Redis fetch timeout for query: {query}")
            raise ProviderError(f"Redis fetch failed: {e}")
    
    async def health_check(self) -> bool:
        """Check Redis health"""
        try:
            await timeout(self.redis_client.ping(), seconds=2.0)
            return True
        except Exception:
            return False

class SupabaseContextProviderClient:
    """Supabase-backed context provider with circuit breaker"""
    
    def __init__(self, supabase_client, circuit_breaker: Optional[CircuitBreaker] = None):
        self.supabase_client = supabase_client
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.logger = logger
    
    async def fetch_contexts(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Fetch contexts from Supabase with resilience"""
        if not self.circuit_breaker.can_execute():
            raise ProviderError("Supabase circuit breaker is OPEN")
        
        try:
            # Build Supabase query
            limit = kwargs.get('limit', 10)
            
            async def supabase_query():
                response = self.supabase_client.table('cognitive_contexts') \
                    .select('*') \
                    .ilike('content', f'%{query}%') \
                    .limit(limit) \
                    .execute()
                return response.data or []
            
            contexts = await monitor_slow_calls(
                timeout(supabase_query(), seconds=kwargs.get('timeout', 10.0)),
                "supabase_context_fetch",
                p95_threshold=5.0
            )
            
            self.circuit_breaker.record_success()
            return contexts
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            if isinstance(e, asyncio.TimeoutError):
                raise TimeoutError(f"Supabase fetch timeout for query: {query}")
            raise ProviderError(f"Supabase fetch failed: {e}")
    
    async def health_check(self) -> bool:
        """Check Supabase health"""
        try:
            await timeout(
                self.supabase_client.table('cognitive_contexts').select('id').limit(1).execute(),
                seconds=3.0
            )
            return True
        except Exception:
            return False

class InMemoryContextProviderClient:
    """In-memory context provider (L1 cache simulation)"""
    
    def __init__(self, contexts: Optional[Dict[str, List[Dict[str, Any]]]] = None):
        self.contexts = contexts or {}
        self.logger = logger
    
    async def fetch_contexts(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Fetch contexts from in-memory store"""
        try:
            # Simple keyword matching
            query_lower = query.lower()
            matching_contexts = []
            
            for key, context_list in self.contexts.items():
                if any(word in key.lower() for word in query_lower.split()):
                    matching_contexts.extend(context_list)
            
            # Limit results
            limit = kwargs.get('limit', 5)
            return matching_contexts[:limit]
            
        except Exception as e:
            raise ProviderError(f"In-memory fetch failed: {e}")
    
    async def health_check(self) -> bool:
        """Always healthy for in-memory"""
        return True
    
    def add_contexts(self, key: str, contexts: List[Dict[str, Any]]):
        """Add contexts to in-memory store"""
        self.contexts[key] = contexts

class ContextProviderClientFactory:
    """Factory for creating appropriate context provider clients"""
    
    @staticmethod
    def create_redis_client(redis_client) -> RedisContextProviderClient:
        """Create Redis provider client"""
        return RedisContextProviderClient(
            redis_client,
            CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        )
    
    @staticmethod
    def create_supabase_client(supabase_client) -> SupabaseContextProviderClient:
        """Create Supabase provider client"""
        return SupabaseContextProviderClient(
            supabase_client,
            CircuitBreaker(failure_threshold=3, recovery_timeout=120)
        )
    
    @staticmethod
    def create_inmemory_client(initial_contexts: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> InMemoryContextProviderClient:
        """Create in-memory provider client"""
        client = InMemoryContextProviderClient(initial_contexts)
        
        # Add some default contexts for testing
        if not initial_contexts:
            client.add_contexts("strategic_planning", [
                {"content": "McKinsey 7S Framework for organizational analysis", "relevance": 0.9},
                {"content": "Porter's Five Forces competitive analysis", "relevance": 0.8}
            ])
            client.add_contexts("data_analysis", [
                {"content": "Statistical methods for data analysis", "relevance": 0.7},
                {"content": "Data visualization best practices", "relevance": 0.6}
            ])
        
        return client