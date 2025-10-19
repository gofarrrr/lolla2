"""
METIS Enterprise API Gateway
Production-grade API gateway with authentication, rate limiting, and enterprise features

Implements comprehensive API gateway with multi-tenant routing, security, monitoring,
and enterprise-grade features for the METIS cognitive intelligence platform.
"""

import logging
import time
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass, field
from uuid import uuid4
import os
import re
from collections import defaultdict, deque

try:
    from fastapi import FastAPI, Request, Response, HTTPException, Depends, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.responses import JSONResponse
    from starlette.middleware.base import BaseHTTPMiddleware

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

try:
    import jwt

    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    jwt = None


class RateLimitStrategy(str, Enum):
    """Rate limiting strategies"""

    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


class ApiKeyType(str, Enum):
    """API key types"""

    TENANT = "tenant"
    USER = "user"
    SERVICE = "service"
    WEBHOOK = "webhook"
    READONLY = "readonly"


class RequestPriority(str, Enum):
    """Request priority levels"""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration"""

    rule_id: str = ""
    name: str = ""
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW

    # Limits
    requests_per_second: Optional[int] = None
    requests_per_minute: Optional[int] = None
    requests_per_hour: Optional[int] = None
    requests_per_day: Optional[int] = None

    # Scope
    applies_to: str = "tenant"  # tenant, user, endpoint, global
    endpoint_patterns: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)

    # Burst handling
    burst_multiplier: float = 1.5
    burst_duration_seconds: int = 60

    # Actions
    block_on_exceed: bool = True
    delay_on_exceed: bool = False
    delay_seconds: float = 1.0

    # Exceptions
    whitelist_ips: List[str] = field(default_factory=list)
    whitelist_user_agents: List[str] = field(default_factory=list)


@dataclass
class ApiKey:
    """API key configuration and metadata"""

    key_id: str = field(default_factory=lambda: str(uuid4()))
    key_hash: str = ""
    name: str = ""

    # Classification
    key_type: ApiKeyType = ApiKeyType.TENANT
    tenant_id: str = ""
    user_id: str = ""

    # Permissions
    scopes: List[str] = field(default_factory=list)
    allowed_endpoints: List[str] = field(default_factory=list)
    allowed_methods: List[str] = field(default_factory=list)

    # Restrictions
    ip_whitelist: List[str] = field(default_factory=list)
    rate_limit_rules: List[str] = field(default_factory=list)

    # Status
    active: bool = True
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    usage_count: int = 0

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""
    description: str = ""

    def is_valid(self) -> bool:
        """Check if API key is valid"""
        if not self.active:
            return False

        if self.expires_at and self.expires_at < datetime.utcnow():
            return False

        return True

    def has_scope(self, required_scope: str) -> bool:
        """Check if API key has required scope"""
        return required_scope in self.scopes or "admin" in self.scopes

    def can_access_endpoint(self, endpoint: str, method: str) -> bool:
        """Check if API key can access endpoint"""
        if not self.allowed_endpoints:
            return True  # No restrictions

        if not self.allowed_methods:
            method_allowed = True
        else:
            method_allowed = method.upper() in [m.upper() for m in self.allowed_methods]

        endpoint_allowed = any(
            re.match(pattern, endpoint) for pattern in self.allowed_endpoints
        )

        return endpoint_allowed and method_allowed


@dataclass
class RequestMetrics:
    """Request metrics and monitoring data"""

    request_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Request details
    method: str = ""
    endpoint: str = ""
    user_agent: str = ""
    ip_address: str = ""

    # Authentication
    tenant_id: str = ""
    user_id: str = ""
    api_key_id: str = ""

    # Performance
    response_time_ms: float = 0.0
    request_size_bytes: int = 0
    response_size_bytes: int = 0

    # Status
    status_code: int = 200
    error_message: str = ""

    # Features used
    rate_limited: bool = False
    cached_response: bool = False
    priority: RequestPriority = RequestPriority.NORMAL


class RateLimiter:
    """Advanced rate limiting implementation"""

    def __init__(self, redis_client: Optional[Any] = None):
        self.redis_client = redis_client
        self.rules: Dict[str, RateLimitRule] = {}
        self.in_memory_counters: Dict[str, deque] = defaultdict(deque)

        self.logger = logging.getLogger(__name__)

    def add_rule(self, rule: RateLimitRule):
        """Add rate limiting rule"""
        self.rules[rule.rule_id] = rule
        self.logger.info(f"Added rate limit rule: {rule.name}")

    async def check_rate_limit(
        self,
        identifier: str,
        rule_id: str,
        endpoint: str = "",
        method: str = "",
        ip_address: str = "",
        user_agent: str = "",
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if request should be rate limited"""

        if rule_id not in self.rules:
            return True, {}  # No rule, allow request

        rule = self.rules[rule_id]

        # Check whitelist
        if ip_address in rule.whitelist_ips:
            return True, {}

        if any(pattern in user_agent for pattern in rule.whitelist_user_agents):
            return True, {}

        # Check endpoint patterns
        if rule.endpoint_patterns:
            if not any(
                re.match(pattern, endpoint) for pattern in rule.endpoint_patterns
            ):
                return True, {}  # Endpoint not covered by rule

        # Check methods
        if rule.methods:
            if method.upper() not in [m.upper() for m in rule.methods]:
                return True, {}  # Method not covered by rule

        # Apply rate limiting based on strategy
        if rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return await self._check_sliding_window(identifier, rule)
        elif rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return await self._check_token_bucket(identifier, rule)
        else:
            return await self._check_fixed_window(identifier, rule)

    async def _check_sliding_window(
        self, identifier: str, rule: RateLimitRule
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limit using sliding window"""

        now = time.time()

        # Use Redis if available
        if self.redis_client:
            return await self._check_sliding_window_redis(identifier, rule, now)
        else:
            return await self._check_sliding_window_memory(identifier, rule, now)

    async def _check_sliding_window_redis(
        self, identifier: str, rule: RateLimitRule, now: float
    ) -> Tuple[bool, Dict[str, Any]]:
        """Redis-based sliding window rate limiting"""

        key = f"rate_limit:{rule.rule_id}:{identifier}"

        # Determine window and limit
        if rule.requests_per_second:
            window = 1
            limit = rule.requests_per_second
        elif rule.requests_per_minute:
            window = 60
            limit = rule.requests_per_minute
        elif rule.requests_per_hour:
            window = 3600
            limit = rule.requests_per_hour
        else:
            window = 86400
            limit = rule.requests_per_day or 1000

        window_start = now - window

        # Clean old entries and count current
        pipe = self.redis_client.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zcard(key)
        pipe.zadd(key, {str(now): now})
        pipe.expire(key, int(window) + 1)

        results = await pipe.execute()
        current_count = results[1]

        allowed = current_count < limit

        info = {
            "limit": limit,
            "remaining": max(0, limit - current_count - 1),
            "reset_time": int(now + window),
            "window_seconds": window,
        }

        return allowed, info

    async def _check_sliding_window_memory(
        self, identifier: str, rule: RateLimitRule, now: float
    ) -> Tuple[bool, Dict[str, Any]]:
        """In-memory sliding window rate limiting"""

        key = f"{rule.rule_id}:{identifier}"

        # Determine window and limit
        if rule.requests_per_second:
            window = 1
            limit = rule.requests_per_second
        elif rule.requests_per_minute:
            window = 60
            limit = rule.requests_per_minute
        elif rule.requests_per_hour:
            window = 3600
            limit = rule.requests_per_hour
        else:
            window = 86400
            limit = rule.requests_per_day or 1000

        window_start = now - window

        # Clean old entries
        counter = self.in_memory_counters[key]
        while counter and counter[0] < window_start:
            counter.popleft()

        # Check limit
        allowed = len(counter) < limit

        if allowed:
            counter.append(now)

        info = {
            "limit": limit,
            "remaining": max(0, limit - len(counter)),
            "reset_time": int(now + window),
            "window_seconds": window,
        }

        return allowed, info

    async def _check_token_bucket(
        self, identifier: str, rule: RateLimitRule
    ) -> Tuple[bool, Dict[str, Any]]:
        """Token bucket rate limiting"""

        # Simplified token bucket implementation
        # In production, would use more sophisticated algorithm
        return await self._check_sliding_window(identifier, rule)

    async def _check_fixed_window(
        self, identifier: str, rule: RateLimitRule
    ) -> Tuple[bool, Dict[str, Any]]:
        """Fixed window rate limiting"""

        now = time.time()

        # Determine window and limit
        if rule.requests_per_minute:
            window = 60
            limit = rule.requests_per_minute
            window_start = int(now // window) * window
        elif rule.requests_per_hour:
            window = 3600
            limit = rule.requests_per_hour
            window_start = int(now // window) * window
        else:
            window = 86400
            limit = rule.requests_per_day or 1000
            window_start = int(now // window) * window

        key = f"rate_limit_fixed:{rule.rule_id}:{identifier}:{window_start}"

        if self.redis_client:
            # Redis-based counting
            current_count = await self.redis_client.incr(key)
            if current_count == 1:
                await self.redis_client.expire(key, int(window))
        else:
            # In-memory counting
            memory_key = f"{rule.rule_id}:{identifier}:{window_start}"
            if memory_key not in self.in_memory_counters:
                self.in_memory_counters[memory_key] = deque([0])

            counter = self.in_memory_counters[memory_key]
            counter[0] += 1
            current_count = counter[0]

        allowed = current_count <= limit

        info = {
            "limit": limit,
            "remaining": max(0, limit - current_count),
            "reset_time": int(window_start + window),
            "window_seconds": window,
        }

        return allowed, info


class ApiKeyManager:
    """API key management system"""

    def __init__(self):
        self.api_keys: Dict[str, ApiKey] = {}
        self.key_hash_to_id: Dict[str, str] = {}

        self.logger = logging.getLogger(__name__)

    def generate_api_key(
        self,
        name: str,
        key_type: ApiKeyType,
        tenant_id: str = "",
        user_id: str = "",
        scopes: List[str] = None,
        expires_in_days: Optional[int] = None,
    ) -> Tuple[str, str]:
        """Generate new API key"""

        # Generate key
        raw_key = f"metis_{uuid4().hex}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        # Create API key object
        api_key = ApiKey(
            key_hash=key_hash,
            name=name,
            key_type=key_type,
            tenant_id=tenant_id,
            user_id=user_id,
            scopes=scopes or ["read"],
            expires_at=(
                datetime.utcnow() + timedelta(days=expires_in_days)
                if expires_in_days
                else None
            ),
        )

        # Store
        self.api_keys[api_key.key_id] = api_key
        self.key_hash_to_id[key_hash] = api_key.key_id

        self.logger.info(f"Generated API key {api_key.key_id} for {name}")

        return raw_key, api_key.key_id

    def validate_api_key(self, raw_key: str) -> Optional[ApiKey]:
        """Validate API key and return key object"""

        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        key_id = self.key_hash_to_id.get(key_hash)

        if not key_id:
            return None

        api_key = self.api_keys.get(key_id)
        if not api_key or not api_key.is_valid():
            return None

        # Update usage
        api_key.last_used = datetime.utcnow()
        api_key.usage_count += 1

        return api_key

    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke API key"""

        if key_id not in self.api_keys:
            return False

        api_key = self.api_keys[key_id]
        api_key.active = False

        self.logger.info(f"Revoked API key {key_id}")
        return True

    def list_api_keys(
        self, tenant_id: str = "", user_id: str = "", active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """List API keys with filters"""

        keys = []

        for api_key in self.api_keys.values():
            # Apply filters
            if tenant_id and api_key.tenant_id != tenant_id:
                continue

            if user_id and api_key.user_id != user_id:
                continue

            if active_only and not api_key.active:
                continue

            keys.append(
                {
                    "key_id": api_key.key_id,
                    "name": api_key.name,
                    "key_type": api_key.key_type.value,
                    "scopes": api_key.scopes,
                    "created_at": api_key.created_at.isoformat(),
                    "last_used": (
                        api_key.last_used.isoformat() if api_key.last_used else None
                    ),
                    "usage_count": api_key.usage_count,
                    "expires_at": (
                        api_key.expires_at.isoformat() if api_key.expires_at else None
                    ),
                }
            )

        return keys


class RequestLogger:
    """Request logging and metrics collection"""

    def __init__(self, max_entries: int = 10000):
        self.metrics: deque = deque(maxlen=max_entries)
        self.metrics_by_tenant: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )

        self.logger = logging.getLogger(__name__)

    def log_request(self, metrics: RequestMetrics):
        """Log request metrics"""

        self.metrics.append(metrics)

        if metrics.tenant_id:
            self.metrics_by_tenant[metrics.tenant_id].append(metrics)

        # Log to system logger
        self.logger.info(
            f"API_REQUEST: {metrics.method} {metrics.endpoint} | "
            f"Status: {metrics.status_code} | "
            f"Time: {metrics.response_time_ms:.1f}ms | "
            f"Tenant: {metrics.tenant_id} | "
            f"IP: {metrics.ip_address}"
        )

    def get_analytics(
        self, tenant_id: str = "", hours_back: int = 24
    ) -> Dict[str, Any]:
        """Get request analytics"""

        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)

        # Select metrics source
        if tenant_id:
            metrics_source = self.metrics_by_tenant.get(tenant_id, deque())
        else:
            metrics_source = self.metrics

        # Filter by time
        recent_metrics = [m for m in metrics_source if m.timestamp > cutoff_time]

        if not recent_metrics:
            return {"message": "No metrics available"}

        # Calculate analytics
        total_requests = len(recent_metrics)

        # Response time analytics
        response_times = [m.response_time_ms for m in recent_metrics]
        avg_response_time = sum(response_times) / len(response_times)

        # Status code distribution
        status_codes = defaultdict(int)
        for m in recent_metrics:
            status_codes[str(m.status_code)] += 1

        # Endpoint popularity
        endpoints = defaultdict(int)
        for m in recent_metrics:
            endpoints[f"{m.method} {m.endpoint}"] += 1

        # Error rate
        error_count = len([m for m in recent_metrics if m.status_code >= 400])
        error_rate = error_count / total_requests * 100

        # Rate limiting stats
        rate_limited_count = len([m for m in recent_metrics if m.rate_limited])

        return {
            "period": {
                "hours": hours_back,
                "start": cutoff_time.isoformat(),
                "end": datetime.now(timezone.utc).isoformat(),
            },
            "summary": {
                "total_requests": total_requests,
                "avg_response_time_ms": avg_response_time,
                "error_rate_percent": error_rate,
                "rate_limited_requests": rate_limited_count,
            },
            "status_codes": dict(status_codes),
            "top_endpoints": dict(
                sorted(endpoints.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
            "performance": {
                "p50_response_time": sorted(response_times)[len(response_times) // 2],
                "p95_response_time": sorted(response_times)[
                    int(len(response_times) * 0.95)
                ],
                "p99_response_time": sorted(response_times)[
                    int(len(response_times) * 0.99)
                ],
            },
        }


class ApiGatewayMiddleware(BaseHTTPMiddleware):
    """Main API gateway middleware"""

    def __init__(
        self,
        app,
        rate_limiter: RateLimiter,
        api_key_manager: ApiKeyManager,
        request_logger: RequestLogger,
    ):
        super().__init__(app)
        self.rate_limiter = rate_limiter
        self.api_key_manager = api_key_manager
        self.request_logger = request_logger

        self.logger = logging.getLogger(__name__)

    async def dispatch(self, request: Request, call_next):
        """Process API request through gateway"""

        start_time = time.time()
        request_id = str(uuid4())

        # Initialize metrics
        metrics = RequestMetrics(
            request_id=request_id,
            method=request.method,
            endpoint=str(request.url.path),
            user_agent=request.headers.get("user-agent", ""),
            ip_address=self._get_client_ip(request),
        )

        try:
            # Authentication
            api_key = await self._authenticate_request(request)
            if api_key:
                metrics.tenant_id = api_key.tenant_id
                metrics.user_id = api_key.user_id
                metrics.api_key_id = api_key.key_id

            # Authorization
            if not await self._authorize_request(request, api_key):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions",
                )

            # Rate limiting
            rate_limit_check = await self._check_rate_limits(request, api_key, metrics)
            if not rate_limit_check["allowed"]:
                metrics.rate_limited = True
                response = JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "detail": "Rate limit exceeded",
                        "retry_after": rate_limit_check.get("retry_after"),
                    },
                    headers=rate_limit_check.get("headers", {}),
                )
                await self._finalize_metrics(metrics, response, start_time)
                return response

            # Add headers
            request.state.api_key = api_key
            request.state.request_id = request_id
            request.state.tenant_id = metrics.tenant_id

            # Process request
            response = await call_next(request)

            # Add response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Tenant-ID"] = metrics.tenant_id
            if rate_limit_check.get("headers"):
                for key, value in rate_limit_check["headers"].items():
                    response.headers[key] = str(value)

            await self._finalize_metrics(metrics, response, start_time)

            return response

        except HTTPException as e:
            response = JSONResponse(
                status_code=e.status_code, content={"detail": e.detail}
            )
            await self._finalize_metrics(metrics, response, start_time, str(e.detail))
            return response

        except Exception as e:
            self.logger.error(f"API Gateway error: {str(e)}")
            response = JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal server error"},
            )
            await self._finalize_metrics(metrics, response, start_time, str(e))
            return response

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

    async def _authenticate_request(self, request: Request) -> Optional[ApiKey]:
        """Authenticate API request"""

        # Check for API key in header
        api_key_header = request.headers.get("X-API-Key") or request.headers.get(
            "Authorization"
        )

        if not api_key_header:
            # Allow unauthenticated access to public endpoints
            if self._is_public_endpoint(request.url.path):
                return None
            else:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="API key required"
                )

        # Extract key from Authorization header
        if api_key_header.startswith("Bearer "):
            raw_key = api_key_header[7:]
        else:
            raw_key = api_key_header

        # Validate API key
        api_key = self.api_key_manager.validate_api_key(raw_key)
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
            )

        return api_key

    def _is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public"""
        public_patterns = [
            r"^/health$",
            r"^/docs.*",
            r"^/openapi\.json$",
            r"^/status$",
            r"^/auth/login$",
            r"^/auth/register$",
        ]

        return any(re.match(pattern, path) for pattern in public_patterns)

    async def _authorize_request(
        self, request: Request, api_key: Optional[ApiKey]
    ) -> bool:
        """Authorize API request"""

        if not api_key:
            return self._is_public_endpoint(request.url.path)

        # Check endpoint access
        if not api_key.can_access_endpoint(request.url.path, request.method):
            return False

        # Check IP whitelist
        if api_key.ip_whitelist:
            client_ip = self._get_client_ip(request)
            if client_ip not in api_key.ip_whitelist:
                return False

        return True

    async def _check_rate_limits(
        self, request: Request, api_key: Optional[ApiKey], metrics: RequestMetrics
    ) -> Dict[str, Any]:
        """Check rate limits for request"""

        if not api_key:
            # Apply anonymous rate limiting
            identifier = self._get_client_ip(request)
            rule_id = "anonymous"
        else:
            # Apply API key rate limiting
            identifier = api_key.key_id
            rule_id = (
                api_key.rate_limit_rules[0] if api_key.rate_limit_rules else "default"
            )

        allowed, info = await self.rate_limiter.check_rate_limit(
            identifier=identifier,
            rule_id=rule_id,
            endpoint=request.url.path,
            method=request.method,
            ip_address=self._get_client_ip(request),
            user_agent=request.headers.get("user-agent", ""),
        )

        headers = {}
        if info:
            headers.update(
                {
                    "X-RateLimit-Limit": info.get("limit"),
                    "X-RateLimit-Remaining": info.get("remaining"),
                    "X-RateLimit-Reset": info.get("reset_time"),
                }
            )

        return {
            "allowed": allowed,
            "headers": headers,
            "retry_after": info.get("window_seconds", 60) if not allowed else None,
        }

    async def _finalize_metrics(
        self,
        metrics: RequestMetrics,
        response: Response,
        start_time: float,
        error_message: str = "",
    ):
        """Finalize and log request metrics"""

        metrics.response_time_ms = (time.time() - start_time) * 1000
        metrics.status_code = response.status_code
        metrics.error_message = error_message

        # Estimate response size
        if hasattr(response, "body"):
            metrics.response_size_bytes = len(response.body)

        self.request_logger.log_request(metrics)


class EnterpriseApiGateway:
    """Enterprise API Gateway for METIS platform"""

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_client = None
        if REDIS_AVAILABLE and redis_url:
            self.redis_client = redis.from_url(redis_url)

        # Initialize components
        self.rate_limiter = RateLimiter(self.redis_client)
        self.api_key_manager = ApiKeyManager()
        self.request_logger = RequestLogger()

        # FastAPI app
        self.app = None
        if FASTAPI_AVAILABLE:
            self.app = self._create_fastapi_app()

        self.logger = logging.getLogger(__name__)

        # Initialize default configurations
        self._setup_default_rate_limits()
        self._setup_default_api_keys()

    def _create_fastapi_app(self) -> FastAPI:
        """Create FastAPI application with middleware"""

        app = FastAPI(
            title="METIS Enterprise API Gateway",
            description="Enterprise-grade API gateway for METIS cognitive intelligence platform",
            version="1.0.0",
        )

        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["https://*.metis.ai", "http://localhost:3000"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Add trusted host middleware
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*.metis.ai", "localhost", "127.0.0.1"],
        )

        # Add gateway middleware
        app.add_middleware(
            ApiGatewayMiddleware,
            rate_limiter=self.rate_limiter,
            api_key_manager=self.api_key_manager,
            request_logger=self.request_logger,
        )

        # Add routes
        self._add_management_routes(app)

        return app

    def _setup_default_rate_limits(self):
        """Setup default rate limiting rules"""

        # Anonymous user limits
        self.rate_limiter.add_rule(
            RateLimitRule(
                rule_id="anonymous",
                name="Anonymous Rate Limit",
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                requests_per_minute=60,
                requests_per_hour=1000,
                applies_to="ip",
            )
        )

        # Default authenticated user limits
        self.rate_limiter.add_rule(
            RateLimitRule(
                rule_id="default",
                name="Default User Rate Limit",
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                requests_per_minute=300,
                requests_per_hour=5000,
                applies_to="user",
            )
        )

        # Enterprise user limits
        self.rate_limiter.add_rule(
            RateLimitRule(
                rule_id="enterprise",
                name="Enterprise Rate Limit",
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                requests_per_minute=1000,
                requests_per_hour=20000,
                applies_to="tenant",
            )
        )

        # Cognitive engine endpoints (more expensive)
        self.rate_limiter.add_rule(
            RateLimitRule(
                rule_id="cognitive_engine",
                name="Cognitive Engine Rate Limit",
                strategy=RateLimitStrategy.TOKEN_BUCKET,
                requests_per_minute=30,
                requests_per_hour=500,
                endpoint_patterns=[r"^/api/v1/cognitive/.*"],
                applies_to="user",
            )
        )

    def _setup_default_api_keys(self):
        """Setup default API keys for testing"""

        # System admin key
        admin_key, admin_key_id = self.api_key_manager.generate_api_key(
            name="System Admin",
            key_type=ApiKeyType.SERVICE,
            scopes=["admin", "read", "write"],
            expires_in_days=365,
        )

        self.logger.info(f"Generated admin API key: {admin_key}")

    def _add_management_routes(self, app: FastAPI):
        """Add management and monitoring routes"""

        # Removed non-canonical /health endpoint; use /api/v53/health in main app

        @app.get("/status")
        async def status():
            """Detailed status endpoint"""
            return {
                "status": "operational",
                "components": {
                    "rate_limiter": "healthy",
                    "api_key_manager": "healthy",
                    "request_logger": "healthy",
                    "redis": "healthy" if self.redis_client else "not_available",
                },
                "timestamp": datetime.utcnow().isoformat(),
            }

        @app.get("/metrics")
        async def get_metrics(tenant_id: str = "", hours: int = 24):
            """Get API metrics"""
            return self.request_logger.get_analytics(tenant_id, hours)

        @app.post("/api-keys")
        async def create_api_key(
            name: str,
            key_type: str = "user",
            tenant_id: str = "",
            scopes: List[str] = ["read"],
        ):
            """Create new API key"""
            raw_key, key_id = self.api_key_manager.generate_api_key(
                name=name,
                key_type=ApiKeyType(key_type),
                tenant_id=tenant_id,
                scopes=scopes,
            )

            return {
                "key_id": key_id,
                "api_key": raw_key,
                "warning": "Store this key securely. It will not be shown again.",
            }

        @app.get("/api-keys")
        async def list_api_keys(tenant_id: str = "", active_only: bool = True):
            """List API keys"""
            return self.api_key_manager.list_api_keys(
                tenant_id=tenant_id, active_only=active_only
            )

        @app.delete("/api-keys/{key_id}")
        async def revoke_api_key(key_id: str):
            """Revoke API key"""
            success = self.api_key_manager.revoke_api_key(key_id)
            if not success:
                raise HTTPException(status_code=404, detail="API key not found")
            return {"message": "API key revoked successfully"}

    def add_rate_limit_rule(self, rule: RateLimitRule):
        """Add custom rate limiting rule"""
        self.rate_limiter.add_rule(rule)

    def get_app(self) -> FastAPI:
        """Get FastAPI application"""
        if not self.app:
            raise RuntimeError("FastAPI not available")
        return self.app

    async def start(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the API gateway"""
        if not FASTAPI_AVAILABLE:
            raise RuntimeError("FastAPI not available")

        import uvicorn

        self.logger.info(f"Starting METIS Enterprise API Gateway on {host}:{port}")

        config = uvicorn.Config(
            app=self.app, host=host, port=port, log_level="info", access_log=True
        )

        server = uvicorn.Server(config)
        await server.serve()


# Global gateway instance
_global_gateway: Optional[EnterpriseApiGateway] = None


def get_api_gateway() -> EnterpriseApiGateway:
    """Get global API gateway instance"""
    global _global_gateway

    if _global_gateway is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        _global_gateway = EnterpriseApiGateway(redis_url)

    return _global_gateway


# Convenience functions
async def start_api_gateway(host: str = "0.0.0.0", port: int = 8000):
    """Start the API gateway server"""
    gateway = get_api_gateway()
    await gateway.start(host, port)


def create_api_key(
    name: str,
    key_type: ApiKeyType = ApiKeyType.USER,
    tenant_id: str = "",
    scopes: List[str] = None,
) -> Tuple[str, str]:
    """Create new API key"""
    gateway = get_api_gateway()
    return gateway.api_key_manager.generate_api_key(
        name=name, key_type=key_type, tenant_id=tenant_id, scopes=scopes or ["read"]
    )


def add_custom_rate_limit(
    rule_id: str,
    name: str,
    requests_per_minute: int,
    endpoint_patterns: List[str] = None,
):
    """Add custom rate limiting rule"""
    gateway = get_api_gateway()

    rule = RateLimitRule(
        rule_id=rule_id,
        name=name,
        requests_per_minute=requests_per_minute,
        endpoint_patterns=endpoint_patterns or [],
    )

    gateway.add_rate_limit_rule(rule)
