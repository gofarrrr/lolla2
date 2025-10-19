"""
API Rate Limiting Service - Request Rate Control
===============================================

REFACTORING TARGET: Extract RateLimiter from foundation.py
PATTERN: Service Extraction with Rate Limiting Strategy
GOAL: Create focused, testable rate limiting service

Responsibility:
- API request rate limiting
- User-based and endpoint-based limits
- Rate limit tracking and enforcement
- Rate limit statistics and monitoring

Benefits:
- Single Responsibility Principle for rate limiting
- Easily testable rate limiting logic
- Clear rate limiting interfaces
- Pluggable rate limiting strategies
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from functools import wraps

try:
    from fastapi import HTTPException, Request, status

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

logger = logging.getLogger(__name__)


class RateLimitStrategy:
    """
    Base rate limiting strategy interface

    Responsibility: Define rate limiting behavior
    Complexity Target: Grade B (≤10 per method)
    """

    def __init__(self, requests_per_hour: int = 100, burst_size: int = 10):
        self.requests_per_hour = requests_per_hour
        self.burst_size = burst_size
        self.window_size_seconds = 3600  # 1 hour

    def is_allowed(self, client_id: str, request_timestamps: List[datetime]) -> bool:
        """
        Check if request is allowed based on strategy

        Complexity: Target B (≤10)
        """
        now = datetime.now()

        # Remove old requests outside window
        cutoff = now - timedelta(seconds=self.window_size_seconds)
        recent_requests = [ts for ts in request_timestamps if ts > cutoff]

        # Check hourly limit
        if len(recent_requests) >= self.requests_per_hour:
            return False

        # Check burst limit (requests in last 60 seconds)
        burst_cutoff = now - timedelta(seconds=60)
        burst_requests = [ts for ts in recent_requests if ts > burst_cutoff]

        if len(burst_requests) >= self.burst_size:
            return False

        return True

    def get_retry_after_seconds(self, request_timestamps: List[datetime]) -> int:
        """
        Get retry-after seconds when rate limited

        Complexity: Target B (≤10)
        """
        if not request_timestamps:
            return 60

        now = datetime.now()
        oldest_request = min(request_timestamps)

        # Time until oldest request is outside window
        window_reset = oldest_request + timedelta(seconds=self.window_size_seconds)

        if window_reset > now:
            return int((window_reset - now).total_seconds())

        return 60  # Default retry after 60 seconds


class APIRateLimitingService:
    """
    API rate limiting service with multiple strategies

    Responsibility: Centralized API rate limiting management
    Complexity Target: Grade B (≤10 per method)
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Rate limit tracking by client ID
        self.client_requests: Dict[str, deque] = defaultdict(lambda: deque())

        # Rate limit strategies by endpoint pattern
        self.rate_strategies = {
            "default": RateLimitStrategy(requests_per_hour=100, burst_size=10),
            "analysis": RateLimitStrategy(requests_per_hour=50, burst_size=5),
            "admin": RateLimitStrategy(requests_per_hour=500, burst_size=50),
            "public": RateLimitStrategy(requests_per_hour=20, burst_size=5),
        }

        # Statistics tracking
        self.rate_limit_stats = {
            "total_requests": 0,
            "rate_limited_requests": 0,
            "unique_clients": 0,
            "last_reset": datetime.now(),
        }

    def get_client_id(
        self,
        request: Optional[Any] = None,
        security_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Extract client ID from request or security context

        Complexity: Target B (≤10)
        """
        # Try security context first
        if security_context:
            user_id = security_context.get("user_id")
            if user_id:
                return f"user:{user_id}"

        # Fall back to IP address from request
        if request and hasattr(request, "client"):
            client_ip = request.client.host if request.client else "unknown"
            return f"ip:{client_ip}"

        return "anonymous"

    def determine_rate_strategy(
        self, endpoint_path: str, security_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Determine rate limiting strategy based on endpoint and user

        Complexity: Target B (≤10)
        """
        # Admin users get higher limits
        if security_context and security_context.get("user_role") == "admin":
            return "admin"

        # Endpoint-based strategy
        if "/analysis" in endpoint_path or "/cognitive" in endpoint_path:
            return "analysis"
        elif "/public" in endpoint_path or "/health" in endpoint_path:
            return "public"

        return "default"

    async def check_rate_limit(
        self,
        request: Optional[Any] = None,
        endpoint_path: str = "",
        security_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Check if request should be rate limited

        Complexity: Target B (≤10)
        """
        client_id = self.get_client_id(request, security_context)
        strategy_name = self.determine_rate_strategy(endpoint_path, security_context)
        strategy = self.rate_strategies[strategy_name]

        # Get client request history
        client_history = self.client_requests[client_id]
        request_timestamps = list(client_history)

        # Update statistics
        self.rate_limit_stats["total_requests"] += 1
        if client_id not in self.client_requests or not self.client_requests[client_id]:
            self.rate_limit_stats["unique_clients"] += 1

        # Check if request is allowed
        is_allowed = strategy.is_allowed(client_id, request_timestamps)

        if is_allowed:
            # Add current request to history
            now = datetime.now()
            client_history.append(now)

            # Limit history size (keep only last 1000 requests)
            while len(client_history) > 1000:
                client_history.popleft()

            return {
                "allowed": True,
                "client_id": client_id,
                "strategy": strategy_name,
                "requests_remaining": strategy.requests_per_hour
                - len(request_timestamps),
                "reset_time": (
                    datetime.now() + timedelta(seconds=strategy.window_size_seconds)
                ).isoformat(),
            }
        else:
            # Rate limited
            self.rate_limit_stats["rate_limited_requests"] += 1
            retry_after = strategy.get_retry_after_seconds(request_timestamps)

            return {
                "allowed": False,
                "client_id": client_id,
                "strategy": strategy_name,
                "retry_after_seconds": retry_after,
                "current_requests": len(request_timestamps),
            }

    def cleanup_old_requests(self):
        """
        Clean up old request tracking data

        Complexity: Target B (≤10)
        """
        now = datetime.now()
        cutoff = now - timedelta(hours=2)  # Keep 2 hours of history

        clients_to_remove = []

        for client_id, request_history in self.client_requests.items():
            # Remove old requests
            while request_history and request_history[0] < cutoff:
                request_history.popleft()

            # Remove clients with no recent requests
            if not request_history:
                clients_to_remove.append(client_id)

        # Remove empty client histories
        for client_id in clients_to_remove:
            del self.client_requests[client_id]

        self.logger.info(
            f"Cleaned up rate limit data. Removed {len(clients_to_remove)} inactive clients."
        )

    def get_rate_limit_stats(self) -> Dict[str, Any]:
        """
        Get rate limiting service statistics

        Complexity: Target B (≤10)
        """
        # Clean up old data first
        self.cleanup_old_requests()

        # Calculate current active clients
        now = datetime.now()
        recent_cutoff = now - timedelta(minutes=30)

        active_clients = 0
        for request_history in self.client_requests.values():
            if request_history and request_history[-1] > recent_cutoff:
                active_clients += 1

        stats = self.rate_limit_stats.copy()
        stats.update(
            {
                "active_clients": active_clients,
                "total_tracked_clients": len(self.client_requests),
                "available_strategies": list(self.rate_strategies.keys()),
                "rate_limit_percentage": (
                    (stats["rate_limited_requests"] / max(stats["total_requests"], 1))
                    * 100
                ),
            }
        )

        return stats


class RateLimitDecorator:
    """
    Rate limiting decorator for API endpoints

    Responsibility: Decorator-based rate limiting
    Complexity Target: Grade B (≤10 per method)
    """

    def __init__(self, rate_service: APIRateLimitingService):
        self.rate_service = rate_service

    def rate_limit(self, requests_per_hour: int = 100):
        """
        Decorator for rate limiting API endpoints

        Complexity: Target B (≤10)
        """

        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract request and security context
                request = kwargs.get("request")
                security_context = kwargs.get("security_context")

                # Get endpoint path
                endpoint_path = request.url.path if request else func.__name__

                # Check rate limit
                rate_result = await self.rate_service.check_rate_limit(
                    request=request,
                    endpoint_path=endpoint_path,
                    security_context=security_context,
                )

                if not rate_result["allowed"]:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Rate limit exceeded",
                        headers={
                            "Retry-After": str(rate_result["retry_after_seconds"])
                        },
                    )

                # Add rate limit info to response headers (if possible)
                response = await func(*args, **kwargs)

                # Add rate limit headers if response supports it
                if hasattr(response, "headers"):
                    response.headers["X-RateLimit-Limit"] = str(requests_per_hour)
                    response.headers["X-RateLimit-Remaining"] = str(
                        rate_result.get("requests_remaining", 0)
                    )
                    response.headers["X-RateLimit-Reset"] = rate_result.get(
                        "reset_time", ""
                    )

                return response

            return wrapper

        return decorator


# Singleton instances for injection
_rate_limiting_service_instance = None
_rate_limit_decorator_instance = None


def get_api_rate_limiting_service() -> APIRateLimitingService:
    """Factory function for API rate limiting service"""
    global _rate_limiting_service_instance
    if _rate_limiting_service_instance is None:
        _rate_limiting_service_instance = APIRateLimitingService()
    return _rate_limiting_service_instance


def get_rate_limit_decorator() -> RateLimitDecorator:
    """Factory function for rate limit decorator"""
    global _rate_limit_decorator_instance
    if _rate_limit_decorator_instance is None:
        rate_service = get_api_rate_limiting_service()
        _rate_limit_decorator_instance = RateLimitDecorator(rate_service)
    return _rate_limit_decorator_instance
