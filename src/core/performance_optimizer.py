#!/usr/bin/env python3
"""
METIS Performance Optimizer

Implements aggressive performance optimizations to meet <15 second response time requirements.
Key Features:
- Circuit breakers for slow operations
- Progressive response with background processing
- Parallel operation execution
- Aggressive timeout management
- Response caching
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps


class PerformanceLevel(str, Enum):
    """Performance optimization levels"""

    FAST_RESPONSE = "fast_response"  # <15s - user-facing
    STANDARD = "standard"  # <30s - background
    COMPREHENSIVE = "comprehensive"  # <60s - deep analysis
    UNLIMITED = "unlimited"  # No timeout


@dataclass
class PerformanceConfig:
    """Performance configuration settings"""

    max_response_time: float  # seconds
    enable_parallel: bool = True
    enable_caching: bool = True
    enable_progressive: bool = True
    circuit_breaker_threshold: int = 3
    max_concurrent_operations: int = 10


class TimeoutError(Exception):
    """Timeout exceeded error"""

    pass


class PerformanceOptimizer:
    """
    Central performance optimization manager for METIS system.
    Ensures all operations meet strict performance requirements.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Performance level configurations
        self.configs = {
            PerformanceLevel.FAST_RESPONSE: PerformanceConfig(
                max_response_time=15.0,  # 15 second hard limit
                enable_parallel=True,
                enable_caching=True,
                enable_progressive=True,
                max_concurrent_operations=5,
            ),
            PerformanceLevel.STANDARD: PerformanceConfig(
                max_response_time=30.0,
                enable_parallel=True,
                enable_caching=True,
                enable_progressive=False,
                max_concurrent_operations=8,
            ),
            PerformanceLevel.COMPREHENSIVE: PerformanceConfig(
                max_response_time=60.0,
                enable_parallel=True,
                enable_caching=False,  # Fresh analysis required
                enable_progressive=False,
                max_concurrent_operations=10,
            ),
            PerformanceLevel.UNLIMITED: PerformanceConfig(
                max_response_time=300.0,  # 5 minute max
                enable_parallel=True,
                enable_caching=False,
                enable_progressive=False,
                max_concurrent_operations=15,
            ),
        }

        # Response cache (simple in-memory for now)
        self.response_cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        self.cache_ttl = timedelta(minutes=10)  # 10 minute cache

        # Circuit breaker state
        self.circuit_failures: Dict[str, int] = {}
        self.circuit_last_attempt: Dict[str, datetime] = {}

        # Performance metrics
        self.operation_times: Dict[str, List[float]] = {}
        self.timeout_count = 0
        self.cache_hit_count = 0
        self.cache_miss_count = 0

        self.logger.info("⚡ PerformanceOptimizer initialized with aggressive timeouts")

    def with_performance_optimization(
        self,
        performance_level: PerformanceLevel = PerformanceLevel.FAST_RESPONSE,
        operation_name: str = "unknown",
        cache_key: Optional[str] = None,
        enable_fallback: bool = True,
    ):
        """
        Decorator for applying performance optimization to async functions

        Args:
            performance_level: Performance optimization level
            operation_name: Name for metrics tracking
            cache_key: Optional cache key for response caching
            enable_fallback: Enable graceful fallback on timeout
        """

        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                config = self.configs[performance_level]
                start_time = time.time()

                # Check circuit breaker
                if self._is_circuit_open(operation_name, config):
                    if enable_fallback:
                        self.logger.warning(
                            f"🔌 Circuit breaker open for {operation_name}, returning fallback"
                        )
                        return await self._get_fallback_response(
                            operation_name, *args, **kwargs
                        )
                    else:
                        raise TimeoutError(f"Circuit breaker open for {operation_name}")

                # Check cache if enabled
                if config.enable_caching and cache_key:
                    cached_result = self._get_cached_response(cache_key)
                    if cached_result is not None:
                        self.cache_hit_count += 1
                        self.logger.info(f"🎯 Cache hit for {operation_name}")
                        return cached_result
                    self.cache_miss_count += 1

                try:
                    # Apply timeout
                    result = await asyncio.wait_for(
                        func(*args, **kwargs), timeout=config.max_response_time
                    )

                    # Record success
                    elapsed = time.time() - start_time
                    self._record_operation_time(operation_name, elapsed)
                    self._reset_circuit_breaker(operation_name)

                    # Cache result if enabled
                    if config.enable_caching and cache_key:
                        self._cache_response(cache_key, result)

                    self.logger.info(f"✅ {operation_name} completed in {elapsed:.2f}s")
                    return result

                except asyncio.TimeoutError:
                    elapsed = time.time() - start_time
                    self.timeout_count += 1
                    self._record_circuit_failure(operation_name)

                    self.logger.error(
                        f"⏱️ {operation_name} timeout after {elapsed:.2f}s (limit: {config.max_response_time}s)"
                    )

                    if enable_fallback:
                        return await self._get_fallback_response(
                            operation_name, *args, **kwargs
                        )
                    else:
                        raise TimeoutError(
                            f"{operation_name} exceeded {config.max_response_time}s timeout"
                        )

                except Exception as e:
                    elapsed = time.time() - start_time
                    self._record_circuit_failure(operation_name)
                    self.logger.error(
                        f"❌ {operation_name} failed after {elapsed:.2f}s: {str(e)}"
                    )
                    raise

            return wrapper

        return decorator

    async def execute_parallel_operations(
        self,
        operations: List[Callable],
        performance_level: PerformanceLevel = PerformanceLevel.FAST_RESPONSE,
        operation_name: str = "parallel_ops",
    ) -> List[Any]:
        """
        Execute multiple operations in parallel with performance optimization

        Args:
            operations: List of async functions to execute
            performance_level: Performance level to apply
            operation_name: Name for tracking

        Returns:
            List of results from operations
        """
        config = self.configs[performance_level]

        self.logger.info(
            f"⚡ Executing {len(operations)} parallel operations: {operation_name}"
        )

        start_time = time.time()

        try:
            # Use semaphore to limit concurrent operations
            semaphore = asyncio.Semaphore(config.max_concurrent_operations)

            async def execute_with_semaphore(op):
                async with semaphore:
                    return await op()

            # Execute all operations with timeout
            results = await asyncio.wait_for(
                asyncio.gather(*[execute_with_semaphore(op) for op in operations]),
                timeout=config.max_response_time,
            )

            elapsed = time.time() - start_time
            self.logger.info(
                f"✅ {operation_name} parallel execution completed in {elapsed:.2f}s"
            )

            return results

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            self.timeout_count += 1
            self.logger.error(
                f"⏱️ {operation_name} parallel execution timeout after {elapsed:.2f}s"
            )
            raise TimeoutError(
                f"Parallel operations exceeded {config.max_response_time}s timeout"
            )

    def _is_circuit_open(self, operation_name: str, config: PerformanceConfig) -> bool:
        """Check if circuit breaker is open for operation"""
        failures = self.circuit_failures.get(operation_name, 0)

        if failures >= config.circuit_breaker_threshold:
            last_attempt = self.circuit_last_attempt.get(operation_name)
            if last_attempt and datetime.now() - last_attempt < timedelta(minutes=5):
                return True

        return False

    def _record_circuit_failure(self, operation_name: str):
        """Record a circuit breaker failure"""
        self.circuit_failures[operation_name] = (
            self.circuit_failures.get(operation_name, 0) + 1
        )
        self.circuit_last_attempt[operation_name] = datetime.now()

    def _reset_circuit_breaker(self, operation_name: str):
        """Reset circuit breaker on successful operation"""
        if operation_name in self.circuit_failures:
            del self.circuit_failures[operation_name]
        if operation_name in self.circuit_last_attempt:
            del self.circuit_last_attempt[operation_name]

    def _get_cached_response(self, cache_key: str) -> Optional[Any]:
        """Get response from cache if valid"""
        if cache_key in self.response_cache:
            timestamp = self.cache_timestamps.get(cache_key)
            if timestamp and datetime.now() - timestamp < self.cache_ttl:
                return self.response_cache[cache_key]
            else:
                # Remove expired cache entry
                del self.response_cache[cache_key]
                del self.cache_timestamps[cache_key]

        return None

    def _cache_response(self, cache_key: str, response: Any):
        """Cache response with timestamp"""
        self.response_cache[cache_key] = response
        self.cache_timestamps[cache_key] = datetime.now()

        # Simple cache size management (keep last 1000 entries)
        if len(self.response_cache) > 1000:
            # Remove oldest 100 entries
            oldest_keys = sorted(
                self.cache_timestamps.keys(), key=lambda k: self.cache_timestamps[k]
            )[:100]
            for key in oldest_keys:
                if key in self.response_cache:
                    del self.response_cache[key]
                if key in self.cache_timestamps:
                    del self.cache_timestamps[key]

    def _record_operation_time(self, operation_name: str, elapsed_time: float):
        """Record operation time for metrics"""
        if operation_name not in self.operation_times:
            self.operation_times[operation_name] = []

        self.operation_times[operation_name].append(elapsed_time)

        # Keep only last 100 measurements
        if len(self.operation_times[operation_name]) > 100:
            self.operation_times[operation_name] = self.operation_times[operation_name][
                -100:
            ]

    async def _get_fallback_response(
        self, operation_name: str, *args, **kwargs
    ) -> Dict[str, Any]:
        """Generate fallback response for timed out operations"""
        return {
            "status": "timeout_fallback",
            "operation": operation_name,
            "message": f"Operation {operation_name} exceeded time limit. Partial results or default response provided.",
            "timestamp": datetime.now().isoformat(),
            "fallback": True,
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        avg_times = {}
        for op_name, times in self.operation_times.items():
            if times:
                avg_times[op_name] = {
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "operations_count": len(times),
                }

        cache_hit_rate = 0.0
        total_cache_requests = self.cache_hit_count + self.cache_miss_count
        if total_cache_requests > 0:
            cache_hit_rate = self.cache_hit_count / total_cache_requests

        return {
            "operation_averages": avg_times,
            "timeout_count": self.timeout_count,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.response_cache),
            "circuit_breaker_failures": self.circuit_failures.copy(),
            "active_circuits_open": sum(
                1 for failures in self.circuit_failures.values() if failures >= 3
            ),
        }

    def clear_cache(self):
        """Clear response cache"""
        self.response_cache.clear()
        self.cache_timestamps.clear()
        self.logger.info("🗑️ Performance cache cleared")

    def reset_circuit_breakers(self):
        """Reset all circuit breakers"""
        self.circuit_failures.clear()
        self.circuit_last_attempt.clear()
        self.logger.info("🔌 Circuit breakers reset")


# Global instance
_performance_optimizer = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance"""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer


# Convenience decorators for common performance levels
def fast_response(operation_name: str = "operation", cache_key: Optional[str] = None):
    """Decorator for operations that must complete in <15 seconds"""
    optimizer = get_performance_optimizer()
    return optimizer.with_performance_optimization(
        PerformanceLevel.FAST_RESPONSE, operation_name, cache_key, enable_fallback=True
    )


def standard_performance(
    operation_name: str = "operation", cache_key: Optional[str] = None
):
    """Decorator for operations with <30 second timeout"""
    optimizer = get_performance_optimizer()
    return optimizer.with_performance_optimization(
        PerformanceLevel.STANDARD, operation_name, cache_key, enable_fallback=False
    )


def comprehensive_analysis(operation_name: str = "operation"):
    """Decorator for deep analysis operations with <60 second timeout"""
    optimizer = get_performance_optimizer()
    return optimizer.with_performance_optimization(
        PerformanceLevel.COMPREHENSIVE,
        operation_name,
        cache_key=None,
        enable_fallback=False,
    )
