"""
Rate Limiter Service - Production Redis Implementation
B2 - Infrastructure Service Extraction (Red Team Amendment Applied)
"""
import logging
import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

from .contracts import (
    IRateLimiter, RateLimitConfig, RateLimitResult,
    RateLimitError, BackendConnectionError
)
from src.core.async_helpers import timeout
from src.services.context_intelligence.error_taxonomy import CircuitBreaker

logger = logging.getLogger(__name__)

class RedisRateLimiterService:
    """
    Redis-backed Token Bucket Rate Limiter (Production Implementation)
    Red Team Amendment: Production plan with backout strategy
    """
    
    def __init__(self, config: RateLimitConfig, redis_client=None):
        self.config = config
        self.redis_client = redis_client
        self.logger = logger
        
        # Circuit breaker for Redis health (Red Team Amendment)
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60
        )
        
        # In-memory fallback for resilience
        self.memory_fallback = InMemoryRateLimiterService(config)
        
        # Metrics tracking
        self.metrics = {
            "redis_hits": 0,
            "redis_failures": 0,
            "fallback_hits": 0,
            "last_health_check": None
        }
    
    async def check_rate_limit(
        self, 
        identifier: str, 
        limit: int, 
        window: int, 
        **kwargs
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check rate limit using Redis Token Bucket algorithm
        
        Red Team Amendment: Circuit breaker with automatic fallback
        """
        operation_start = time.time()
        
        # Check circuit breaker state
        if not self.circuit_breaker.can_execute():
            self.logger.warning("Redis circuit breaker OPEN, using memory fallback")
            return await self._use_memory_fallback(identifier, limit, window, **kwargs)
        
        try:
            # Redis Token Bucket implementation
            result = await self._redis_token_bucket_check(identifier, limit, window)
            
            # Record success
            self.circuit_breaker.record_success()
            self.metrics["redis_hits"] += 1
            
            return result
            
        except Exception as e:
            # Record failure and use fallback
            self.circuit_breaker.record_failure()
            self.metrics["redis_failures"] += 1
            
            self.logger.warning(f"Redis rate limit check failed: {e}, using fallback")
            
            if self.config.shadow_mode:
                # In shadow mode, allow all requests but log
                self.logger.info(f"Shadow mode: would have failed rate limit for {identifier}")
                return True, {"shadow_mode": True, "would_block": False}
            
            return await self._use_memory_fallback(identifier, limit, window, **kwargs)
    
    async def _redis_token_bucket_check(
        self, identifier: str, limit: int, window: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """Redis-based token bucket algorithm"""
        
        # Redis Lua script for atomic token bucket operation
        lua_script = """
        local key = KEYS[1]
        local limit = tonumber(ARGV[1])
        local window = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        
        local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
        local tokens = tonumber(bucket[1]) or limit
        local last_refill = tonumber(bucket[2]) or now
        
        -- Calculate tokens to add based on time elapsed
        local elapsed = now - last_refill
        local tokens_to_add = math.floor(elapsed * limit / window)
        tokens = math.min(limit, tokens + tokens_to_add)
        
        if tokens >= 1 then
            tokens = tokens - 1
            redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
            redis.call('EXPIRE', key, window)
            return {1, tokens, now + window}
        else
            redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
            redis.call('EXPIRE', key, window)
            return {0, tokens, now + window}
        end
        """
        
        # Execute atomic operation
        key = f"rate_limit:{identifier}"
        now = int(time.time())
        
        result = await timeout(
            self.redis_client.eval(lua_script, 1, key, limit, window, now),
            seconds=2.0
        )
        
        allowed, remaining, reset_time = result
        
        return bool(allowed), {
            "remaining": int(remaining),
            "reset_time": int(reset_time),
            "backend": "redis",
            "identifier": identifier
        }
    
    async def _use_memory_fallback(
        self, identifier: str, limit: int, window: int, **kwargs
    ) -> Tuple[bool, Dict[str, Any]]:
        """Use in-memory fallback when Redis is unavailable"""
        self.metrics["fallback_hits"] += 1
        
        allowed, info = await self.memory_fallback.check_rate_limit(
            identifier, limit, window, **kwargs
        )
        
        # Mark as fallback in response
        info["backend"] = "memory_fallback"
        info["redis_unavailable"] = True
        
        return allowed, info
    
    async def penalize(self, identifier: str, penalty_seconds: int) -> None:
        """Apply penalty using Redis"""
        try:
            if self.circuit_breaker.can_execute():
                key = f"rate_limit_penalty:{identifier}"
                await timeout(
                    self.redis_client.setex(key, penalty_seconds, "penalized"),
                    seconds=1.0
                )
                self.circuit_breaker.record_success()
            else:
                # Use memory fallback
                await self.memory_fallback.penalize(identifier, penalty_seconds)
                
        except Exception as e:
            self.circuit_breaker.record_failure()
            self.logger.warning(f"Redis penalty failed: {e}")
            await self.memory_fallback.penalize(identifier, penalty_seconds)
    
    async def get_quota(self, identifier: str) -> Dict[str, Any]:
        """Get current quota status"""
        try:
            if self.circuit_breaker.can_execute():
                key = f"rate_limit:{identifier}"
                bucket = await timeout(
                    self.redis_client.hmget(key, "tokens", "last_refill"),
                    seconds=1.0
                )
                
                return {
                    "tokens": int(bucket[0]) if bucket[0] else None,
                    "last_refill": int(bucket[1]) if bucket[1] else None,
                    "backend": "redis"
                }
            else:
                return await self.memory_fallback.get_quota(identifier)
                
        except Exception as e:
            self.logger.warning(f"Redis quota check failed: {e}")
            return await self.memory_fallback.get_quota(identifier)
    
    async def reset_quota(self, identifier: str) -> None:
        """Reset quota for identifier"""
        try:
            if self.circuit_breaker.can_execute():
                key = f"rate_limit:{identifier}"
                await timeout(
                    self.redis_client.delete(key),
                    seconds=1.0
                )
        except Exception as e:
            self.logger.warning(f"Redis quota reset failed: {e}")
            await self.memory_fallback.reset_quota(identifier)
    
    async def health_check(self) -> bool:
        """Check Redis health with circuit breaker update"""
        try:
            await timeout(self.redis_client.ping(), seconds=2.0)
            self.metrics["last_health_check"] = datetime.now()
            
            # Reset circuit breaker on successful health check
            if self.circuit_breaker.state != "CLOSED":
                self.circuit_breaker.record_success()
                self.logger.info("Redis health check successful - circuit breaker reset")
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Redis health check failed: {e}")
            self.circuit_breaker.record_failure()
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get rate limiter metrics"""
        return {
            **self.metrics,
            "circuit_breaker_state": self.circuit_breaker.state,
            "config": {
                "backend": self.config.backend,
                "shadow_mode": self.config.shadow_mode,
                "circuit_breaker_enabled": self.config.circuit_breaker_enabled
            }
        }

class InMemoryRateLimiterService:
    """
    In-memory rate limiter (development/fallback)
    Red Team Amendment: Clean fallback implementation
    """
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.buckets: Dict[str, Dict[str, Any]] = {}
        self.penalties: Dict[str, float] = {}
        self.logger = logger
    
    async def check_rate_limit(
        self, identifier: str, limit: int, window: int, **kwargs
    ) -> Tuple[bool, Dict[str, Any]]:
        """Simple token bucket in memory"""
        now = time.time()
        
        # Check for active penalties
        if identifier in self.penalties:
            if now < self.penalties[identifier]:
                return False, {
                    "remaining": 0,
                    "reset_time": int(self.penalties[identifier]),
                    "backend": "memory",
                    "penalized": True
                }
            else:
                del self.penalties[identifier]
        
        # Get or create bucket
        if identifier not in self.buckets:
            self.buckets[identifier] = {
                "tokens": limit,
                "last_refill": now
            }
        
        bucket = self.buckets[identifier]
        
        # Refill tokens based on elapsed time
        elapsed = now - bucket["last_refill"]
        tokens_to_add = elapsed * limit / window
        bucket["tokens"] = min(limit, bucket["tokens"] + tokens_to_add)
        bucket["last_refill"] = now
        
        # Check if request can proceed
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True, {
                "remaining": int(bucket["tokens"]),
                "reset_time": int(now + window),
                "backend": "memory"
            }
        else:
            return False, {
                "remaining": 0,
                "reset_time": int(now + window),
                "backend": "memory"
            }
    
    async def penalize(self, identifier: str, penalty_seconds: int) -> None:
        """Apply penalty in memory"""
        self.penalties[identifier] = time.time() + penalty_seconds
    
    async def get_quota(self, identifier: str) -> Dict[str, Any]:
        """Get quota from memory"""
        if identifier in self.buckets:
            return {
                "tokens": self.buckets[identifier]["tokens"],
                "last_refill": self.buckets[identifier]["last_refill"],
                "backend": "memory"
            }
        return {"backend": "memory", "no_quota": True}
    
    async def reset_quota(self, identifier: str) -> None:
        """Reset quota in memory"""
        if identifier in self.buckets:
            del self.buckets[identifier]
        if identifier in self.penalties:
            del self.penalties[identifier]
    
    async def health_check(self) -> bool:
        """Always healthy for in-memory"""
        return True

class RateLimiterServiceFactory:
    """
    Factory for creating rate limiter services
    Red Team Amendment: Environment-based configuration
    """
    
    @staticmethod
    def create_from_config(config: RateLimitConfig, redis_client=None):
        """Create rate limiter based on configuration"""
        if config.backend == "redis" and redis_client:
            return RedisRateLimiterService(config, redis_client)
        else:
            logger.info("Using in-memory rate limiter (development mode)")
            return InMemoryRateLimiterService(config)
    
    @staticmethod
    def create_from_env() -> IRateLimiter:
        """Create rate limiter from environment variables"""
        import os
        
        config = RateLimitConfig(
            backend=os.getenv("RATE_LIMIT_BACKEND", "memory"),
            redis_url=os.getenv("RATE_LIMIT_REDIS_URL"),
            shadow_mode=os.getenv("RATE_LIMIT_SHADOW_MODE", "false").lower() == "true",
            circuit_breaker_enabled=os.getenv("RATE_LIMIT_CIRCUIT_BREAKER", "true").lower() == "true"
        )
        
        # Parse endpoint:rate pairs
        buckets_str = os.getenv("RATE_LIMIT_BUCKETS", "")
        if buckets_str:
            config.buckets = {}
            for pair in buckets_str.split(","):
                if ":" in pair:
                    endpoint, rate = pair.split(":", 1)
                    config.buckets[endpoint.strip()] = rate.strip()
        
        # Create Redis client if needed
        redis_client = None
        if config.backend == "redis" and config.redis_url:
            try:
                import redis.asyncio as redis
                redis_client = redis.from_url(config.redis_url)
            except ImportError:
                logger.warning("Redis not available, falling back to memory")
                config.backend = "memory"
        
        return RateLimiterServiceFactory.create_from_config(config, redis_client)