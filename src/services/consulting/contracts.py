"""
Consulting Frameworks Service Contracts
C1 - Discovery & Performance Analysis (Red Team Amendment Applied)
"""
from typing import Protocol, Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

# Performance-oriented contracts (Red Team Amendment)
class IFrameworkRepository(Protocol):
    """Framework metadata repository with performance focus"""
    
    async def get_metadata(self, framework_id: str) -> Optional[Dict[str, Any]]:
        """Get framework metadata with caching"""
        ...
    
    async def batch_get(self, framework_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Batch get frameworks to reduce N×M calls"""
        ...
    
    async def cache_warmup(self, framework_list: List[str]) -> None:
        """Pre-warm cache with common frameworks"""
        ...
    
    async def search_frameworks(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search frameworks with performance budgets"""
        ...

class IFrameworkEvaluator(Protocol):
    """Framework evaluation with performance constraints"""
    
    async def score_frameworks(
        self, 
        context: str, 
        frameworks: List[Dict[str, Any]],
        **kwargs
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Score frameworks with vectorization support"""
        ...
    
    async def batch_evaluate(
        self, 
        contexts: List[str], 
        frameworks: List[Dict[str, Any]]
    ) -> Dict[str, List[Tuple[Dict[str, Any], float]]]:
        """Batch evaluation for performance"""
        ...
    
    async def filter_by_criteria(
        self,
        frameworks: List[Dict[str, Any]],
        criteria: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Filter frameworks by business criteria"""
        ...

# Performance configuration (Red Team Amendment)
@dataclass
class PerformanceConfig:
    """Performance budgets and constraints"""
    p50_threshold_ms: float = 200.0
    p95_threshold_ms: float = 400.0
    max_concurrent_operations: int = 10
    cache_ttl_minutes: int = 10
    warmup_enabled: bool = True
    benchmark_enabled: bool = True

@dataclass
class FrameworkMetadata:
    """Framework metadata structure"""
    id: str
    name: str
    description: str
    category: str
    complexity: str
    applicability_score: float
    last_updated: datetime
    usage_count: int = 0
    performance_metrics: Dict[str, float] = None

@dataclass
class EvaluationResult:
    """Framework evaluation result with performance tracking"""
    framework: Dict[str, Any]
    score: float
    confidence: float
    processing_time_ms: float
    cache_hit: bool = False
    
# Consulting framework errors
class ConsultingFrameworkError(Exception):
    """Base consulting framework error"""
    pass

class PerformanceThresholdExceeded(ConsultingFrameworkError):
    """Performance threshold exceeded"""
    pass

class FrameworkNotFoundError(ConsultingFrameworkError):
    """Framework not found"""
    pass

class EvaluationTimeoutError(ConsultingFrameworkError):
    """Framework evaluation timeout"""
    pass

# Caching strategy (Red Team Amendment)
@dataclass
class CacheConfig:
    """Caching configuration with TTL policy"""
    ttl_minutes: int = 10
    max_entries: int = 1000
    warmup_config_path: str = "configs/framework_warmup.yaml"
    memory_bounds_mb: int = 100
    
class ICacheProvider(Protocol):
    """Cache provider interface"""
    
    async def get(self, key: str) -> Optional[Any]:
        """Get from cache"""
        ...
    
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set in cache with TTL"""
        ...
    
    async def batch_get(self, keys: List[str]) -> Dict[str, Any]:
        """Batch get from cache"""
        ...
    
    async def invalidate(self, pattern: str) -> None:
        """Invalidate cache entries by pattern"""
        ...
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        ...