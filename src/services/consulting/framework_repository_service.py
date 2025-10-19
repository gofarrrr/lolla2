"""
Framework Repository Service - Production Implementation
C2 - Service Extraction & Batching (Red Team Amendment Applied)
"""
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum

from .contracts import (
    IFrameworkRepository, ICacheProvider, FrameworkMetadata,
    PerformanceConfig, FrameworkNotFoundError, EvaluationTimeoutError
)
from src.core.async_helpers import timeout, bounded
from src.core.unified_context_stream import UnifiedContextStream, ContextEventType

logger = logging.getLogger(__name__)

class FrameworkCategory(str, Enum):
    """Categories of consulting frameworks"""
    STRATEGIC = "strategic"
    ANALYTICAL = "analytical" 
    OPERATIONAL = "operational"
    FINANCIAL = "financial"
    MARKET = "market"
    ORGANIZATIONAL = "organizational"
    DIAGNOSTIC = "diagnostic"

class FrameworkComplexity(str, Enum):
    """Framework complexity levels"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class ProductionFrameworkRepositoryService:
    """
    Production Framework Repository with Performance Optimizations
    Red Team Amendment: Batching, caching, and bounded concurrency
    """
    
    def __init__(
        self, 
        config: PerformanceConfig,
        cache_provider: Optional[ICacheProvider] = None,
        context_stream: Optional[UnifiedContextStream] = None
    ):
        self.config = config
        self.cache_provider = cache_provider
        self.context_stream = context_stream
        self.logger = logger
        
        # Performance controls (Red Team Amendment)
        self.concurrency_semaphore = asyncio.Semaphore(config.max_concurrent_operations)
        
        # Framework storage
        self.frameworks: Dict[str, Dict[str, Any]] = {}
        self.framework_rankings: Dict[str, float] = {}
        
        # Performance metrics
        self.metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "batch_operations": 0,
            "framework_loads": 0,
            "average_load_time": 0.0
        }
        
        # Initialize frameworks
        asyncio.create_task(self._initialize_framework_library())
    
    async def get_metadata(self, framework_id: str) -> Optional[Dict[str, Any]]:
        """Get framework metadata with caching (Red Team Amendment)"""
        operation_start = time.time()
        
        try:
            # Check cache first
            if self.cache_provider:
                cache_key = f"framework_meta:{framework_id}"
                cached_result = await timeout(
                    self.cache_provider.get(cache_key),
                    seconds=0.5  # Fast cache timeout
                )
                
                if cached_result:
                    self.metrics["cache_hits"] += 1
                    await self._emit_performance_event(
                        "FRAMEWORK_METADATA_CACHE_HIT", 
                        {"framework_id": framework_id, "time_ms": (time.time() - operation_start) * 1000}
                    )
                    return cached_result
            
            self.metrics["cache_misses"] += 1
            
            # Load from storage
            async with self.concurrency_semaphore:
                framework_data = await self._load_framework_metadata(framework_id)
                
                if framework_data and self.cache_provider:
                    # Cache for future use
                    await timeout(
                        self.cache_provider.set(
                            cache_key, 
                            framework_data, 
                            ttl_seconds=self.config.cache_ttl_minutes * 60
                        ),
                        seconds=1.0
                    )
            
            processing_time = (time.time() - operation_start) * 1000
            
            # Performance validation (Red Team Amendment)
            if processing_time > self.config.p95_threshold_ms:
                self.logger.warning(f"Framework metadata load exceeded p95 threshold: {processing_time:.2f}ms")
            
            await self._emit_performance_event(
                "FRAMEWORK_METADATA_LOADED",
                {"framework_id": framework_id, "time_ms": processing_time, "cache_hit": False}
            )
            
            return framework_data
            
        except asyncio.TimeoutError:
            raise EvaluationTimeoutError(f"Framework metadata load timeout for {framework_id}")
        except Exception as e:
            self.logger.error(f"Framework metadata load failed for {framework_id}: {e}")
            return None
    
    async def batch_get(self, framework_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Batch get frameworks to reduce N×M calls (Red Team Amendment)"""
        operation_start = time.time()
        self.metrics["batch_operations"] += 1
        
        try:
            # Bounded concurrency for batch operations
            semaphore = asyncio.Semaphore(min(len(framework_ids), self.config.max_concurrent_operations))
            
            async def get_single_framework(framework_id: str) -> Tuple[str, Optional[Dict[str, Any]]]:
                async with semaphore:
                    metadata = await self.get_metadata(framework_id)
                    return framework_id, metadata
            
            # Execute batch with timeout
            tasks = [get_single_framework(fid) for fid in framework_ids]
            results = await timeout(
                asyncio.gather(*tasks, return_exceptions=True),
                seconds=self.config.p95_threshold_ms / 1000.0
            )
            
            # Process results
            batch_result = {}
            for result in results:
                if isinstance(result, Exception):
                    self.logger.warning(f"Batch framework load failed: {result}")
                    continue
                    
                framework_id, metadata = result
                if metadata:
                    batch_result[framework_id] = metadata
            
            processing_time = (time.time() - operation_start) * 1000
            
            await self._emit_performance_event(
                "FRAMEWORK_BATCH_LOADED",
                {
                    "requested_count": len(framework_ids),
                    "loaded_count": len(batch_result),
                    "time_ms": processing_time
                }
            )
            
            return batch_result
            
        except asyncio.TimeoutError:
            raise EvaluationTimeoutError(f"Batch framework load timeout for {len(framework_ids)} frameworks")
        except Exception as e:
            self.logger.error(f"Batch framework load failed: {e}")
            return {}
    
    async def cache_warmup(self, framework_list: List[str]) -> None:
        """Pre-warm cache with common frameworks (Red Team Amendment)"""
        if not self.config.warmup_enabled:
            return
        
        operation_start = time.time()
        self.logger.info(f"Warming up framework cache with {len(framework_list)} frameworks")
        
        try:
            # Load warmup configuration
            warmup_frameworks = await self._load_warmup_configuration(framework_list)
            
            # Batch load for cache warmup
            warmed_frameworks = await self.batch_get(warmup_frameworks)
            
            processing_time = (time.time() - operation_start) * 1000
            
            await self._emit_performance_event(
                "FRAMEWORK_CACHE_WARMUP_COMPLETE",
                {
                    "warmed_count": len(warmed_frameworks),
                    "time_ms": processing_time
                }
            )
            
            self.logger.info(f"Cache warmup complete: {len(warmed_frameworks)} frameworks warmed in {processing_time:.2f}ms")
            
        except Exception as e:
            self.logger.error(f"Framework cache warmup failed: {e}")
    
    async def search_frameworks(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search frameworks with performance budgets (Red Team Amendment)"""
        operation_start = time.time()
        
        try:
            # Use cache for common queries
            cache_key = f"framework_search:{hash(query)}:{limit}"
            
            if self.cache_provider:
                cached_results = await timeout(
                    self.cache_provider.get(cache_key),
                    seconds=0.5
                )
                
                if cached_results:
                    self.metrics["cache_hits"] += 1
                    return cached_results
            
            # Perform search with bounded concurrency
            async with self.concurrency_semaphore:
                search_results = await self._perform_framework_search(query, limit)
            
            # Cache results
            if self.cache_provider and search_results:
                await timeout(
                    self.cache_provider.set(
                        cache_key,
                        search_results,
                        ttl_seconds=self.config.cache_ttl_minutes * 60
                    ),
                    seconds=1.0
                )
            
            processing_time = (time.time() - operation_start) * 1000
            
            # Performance validation
            if processing_time > self.config.p50_threshold_ms:
                self.logger.warning(f"Framework search exceeded p50 threshold: {processing_time:.2f}ms")
            
            await self._emit_performance_event(
                "FRAMEWORK_SEARCH_COMPLETE",
                {"query_length": len(query), "results_count": len(search_results), "time_ms": processing_time}
            )
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Framework search failed for query '{query}': {e}")
            return []
    
    async def _initialize_framework_library(self):
        """Initialize consulting frameworks library with performance tracking"""
        operation_start = time.time()
        
        try:
            # Core consulting frameworks
            frameworks_config = {
                "mece_framework": {
                    "name": "MECE Framework",
                    "category": FrameworkCategory.ANALYTICAL,
                    "complexity": FrameworkComplexity.INTERMEDIATE,
                    "description": "Mutually Exclusive, Collectively Exhaustive problem structuring",
                    "applicability_score": 8.5,
                    "usage_count": 0
                },
                "issue_tree_framework": {
                    "name": "Issue Tree Framework", 
                    "category": FrameworkCategory.ANALYTICAL,
                    "complexity": FrameworkComplexity.ADVANCED,
                    "description": "Hierarchical problem breakdown and hypothesis structuring",
                    "applicability_score": 9.0,
                    "usage_count": 0
                },
                "bcg_growth_share_matrix": {
                    "name": "BCG Growth-Share Matrix",
                    "category": FrameworkCategory.STRATEGIC,
                    "complexity": FrameworkComplexity.INTERMEDIATE,
                    "description": "Portfolio analysis and strategic positioning",
                    "applicability_score": 7.5,
                    "usage_count": 0
                },
                "porters_five_forces": {
                    "name": "Porter's Five Forces",
                    "category": FrameworkCategory.MARKET,
                    "complexity": FrameworkComplexity.INTERMEDIATE,
                    "description": "Industry structure and competitive analysis",
                    "applicability_score": 8.0,
                    "usage_count": 0
                },
                "value_chain_analysis": {
                    "name": "Value Chain Analysis",
                    "category": FrameworkCategory.OPERATIONAL,
                    "complexity": FrameworkComplexity.ADVANCED,
                    "description": "Business process and value creation analysis",
                    "applicability_score": 7.8,
                    "usage_count": 0
                }
            }
            
            # Initialize frameworks with metadata
            for framework_id, config in frameworks_config.items():
                self.frameworks[framework_id] = {
                    **config,
                    "framework_id": framework_id,
                    "last_updated": datetime.now(),
                    "performance_metrics": {
                        "average_execution_time": 0.0,
                        "success_rate": 1.0,
                        "complexity_score": self._map_complexity_to_score(config["complexity"])
                    }
                }
                
                # Initial ranking
                self.framework_rankings[framework_id] = config["applicability_score"]
            
            processing_time = (time.time() - operation_start) * 1000
            
            await self._emit_performance_event(
                "FRAMEWORK_LIBRARY_INITIALIZED",
                {"framework_count": len(self.frameworks), "time_ms": processing_time}
            )
            
            self.logger.info(f"Initialized {len(self.frameworks)} consulting frameworks in {processing_time:.2f}ms")
            
        except Exception as e:
            self.logger.error(f"Framework library initialization failed: {e}")
    
    async def _load_framework_metadata(self, framework_id: str) -> Optional[Dict[str, Any]]:
        """Load framework metadata from storage"""
        if framework_id in self.frameworks:
            self.metrics["framework_loads"] += 1
            return self.frameworks[framework_id]
        
        raise FrameworkNotFoundError(f"Framework not found: {framework_id}")
    
    async def _load_warmup_configuration(self, framework_list: List[str]) -> List[str]:
        """Load framework warmup configuration"""
        # For production, this would load from config file
        # For now, return the provided list + common frameworks
        common_frameworks = ["mece_framework", "issue_tree_framework", "porters_five_forces"]
        return list(set(framework_list + common_frameworks))
    
    async def _perform_framework_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Perform framework search with relevance scoring"""
        query_lower = query.lower()
        scored_frameworks = []
        
        for framework_id, framework_data in self.frameworks.items():
            # Simple relevance scoring
            score = 0.0
            
            # Name match
            if query_lower in framework_data["name"].lower():
                score += 10.0
            
            # Description match
            if query_lower in framework_data["description"].lower():
                score += 5.0
            
            # Category match
            if query_lower in framework_data["category"].value.lower():
                score += 3.0
            
            # Add applicability boost
            score += framework_data["applicability_score"] * 0.1
            
            if score > 0:
                scored_frameworks.append((framework_data, score))
        
        # Sort by score and return top results
        scored_frameworks.sort(key=lambda x: x[1], reverse=True)
        return [fw for fw, score in scored_frameworks[:limit]]
    
    def _map_complexity_to_score(self, complexity: FrameworkComplexity) -> float:
        """Map complexity enum to numeric score"""
        mapping = {
            FrameworkComplexity.BASIC: 1.0,
            FrameworkComplexity.INTERMEDIATE: 2.0,
            FrameworkComplexity.ADVANCED: 3.0,
            FrameworkComplexity.EXPERT: 4.0
        }
        return mapping.get(complexity, 2.0)
    
    async def _emit_performance_event(self, event_type: str, details: Dict[str, Any]):
        """Emit performance events for observability"""
        if self.context_stream:
            try:
                await self.context_stream.emit_event(
                    event_type=ContextEventType.PERFORMANCE_METRIC,
                    details={
                        "operation": event_type,
                        "service": "framework_repository",
                        **details
                    }
                )
            except Exception as e:
                self.logger.warning(f"Failed to emit performance event: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get repository performance metrics"""
        return {
            **self.metrics,
            "framework_count": len(self.frameworks),
            "cache_hit_rate": self.metrics["cache_hits"] / max(1, self.metrics["cache_hits"] + self.metrics["cache_misses"]),
            "config": {
                "p50_threshold_ms": self.config.p50_threshold_ms,
                "p95_threshold_ms": self.config.p95_threshold_ms,
                "max_concurrent_operations": self.config.max_concurrent_operations
            }
        }

class InMemoryCacheProvider:
    """Simple in-memory cache provider for development"""
    
    def __init__(self, max_entries: int = 1000):
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.max_entries = max_entries
    
    async def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            value, expires_at = self.cache[key]
            if time.time() < expires_at:
                return value
            else:
                del self.cache[key]
        return None
    
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        expires_at = time.time() + (ttl_seconds or 3600)
        
        # Simple eviction if cache is full
        if len(self.cache) >= self.max_entries:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[key] = (value, expires_at)
    
    async def batch_get(self, keys: List[str]) -> Dict[str, Any]:
        result = {}
        for key in keys:
            value = await self.get(key)
            if value is not None:
                result[key] = value
        return result
    
    async def invalidate(self, pattern: str) -> None:
        # Simple pattern matching
        keys_to_remove = [k for k in self.cache.keys() if pattern in k]
        for key in keys_to_remove:
            del self.cache[key]
    
    async def get_stats(self) -> Dict[str, Any]:
        return {
            "entries": len(self.cache),
            "max_entries": self.max_entries
        }

class FrameworkRepositoryServiceFactory:
    """Factory for creating framework repository services"""
    
    @staticmethod
    def create_production_service(
        config: PerformanceConfig,
        cache_provider: Optional[ICacheProvider] = None,
        context_stream: Optional[UnifiedContextStream] = None
    ) -> IFrameworkRepository:
        """Create production framework repository service"""
        
        if cache_provider is None:
            cache_provider = InMemoryCacheProvider(max_entries=config.cache_ttl_minutes * 10)
        
        return ProductionFrameworkRepositoryService(config, cache_provider, context_stream)
    
    @staticmethod
    def create_from_env() -> IFrameworkRepository:
        """Create framework repository from environment variables"""
        import os
        
        config = PerformanceConfig(
            p50_threshold_ms=float(os.getenv("FRAMEWORK_P50_THRESHOLD_MS", "200.0")),
            p95_threshold_ms=float(os.getenv("FRAMEWORK_P95_THRESHOLD_MS", "400.0")),
            max_concurrent_operations=int(os.getenv("FRAMEWORK_MAX_CONCURRENT", "10")),
            cache_ttl_minutes=int(os.getenv("FRAMEWORK_CACHE_TTL_MIN", "10")),
            warmup_enabled=os.getenv("FRAMEWORK_WARMUP_ENABLED", "true").lower() == "true"
        )
        
        return FrameworkRepositoryServiceFactory.create_production_service(config)