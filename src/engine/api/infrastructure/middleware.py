"""
API Middleware Components - Extracted from foundation.py
CORS, logging, security, and request processing middleware
"""

import time
import logging
from typing import Callable, Dict, Any, Optional
from datetime import datetime

try:
    from fastapi import FastAPI, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from starlette.middleware.base import BaseHTTPMiddleware

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    BaseHTTPMiddleware = object


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security headers and request validation middleware"""

    def __init__(self, app, enable_security_headers: bool = True):
        super().__init__(app)
        self.enable_security_headers = enable_security_headers
        self.logger = logging.getLogger(__name__)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Add security headers to response
        response = await call_next(request)

        if self.enable_security_headers:
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
            response.headers["Content-Security-Policy"] = "default-src 'self'"

        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Request/response logging middleware"""

    def __init__(self, app, log_requests: bool = True, log_responses: bool = False):
        super().__init__(app)
        self.log_requests = log_requests
        self.log_responses = log_responses
        self.logger = logging.getLogger(__name__)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        # Log incoming request
        if self.log_requests:
            self.logger.info(
                f"📥 {request.method} {request.url.path} - {request.client.host if request.client else 'unknown'}"
            )

        # Process request
        response = await call_next(request)

        # Calculate processing time
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)

        # Log response
        if self.log_responses or response.status_code >= 400:
            status_emoji = (
                "✅"
                if response.status_code < 400
                else "❌" if response.status_code >= 500 else "⚠️"
            )
            self.logger.info(
                f"📤 {status_emoji} {request.method} {request.url.path} - {response.status_code} - {process_time:.4f}s"
            )

        return response


class PerformanceMiddleware(BaseHTTPMiddleware):
    """Performance monitoring middleware"""

    def __init__(self, app, slow_request_threshold: float = 2.0):
        super().__init__(app)
        self.slow_request_threshold = slow_request_threshold
        self.logger = logging.getLogger(__name__)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        response = await call_next(request)

        process_time = time.time() - start_time

        # Log slow requests
        if process_time > self.slow_request_threshold:
            self.logger.warning(
                f"🐌 Slow request: {request.method} {request.url.path} - {process_time:.4f}s"
            )

        # Add performance headers
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        response.headers["X-Timestamp"] = datetime.utcnow().isoformat()

        return response


def configure_middleware(
    app: FastAPI,
    enable_cors: bool = True,
    enable_security: bool = True,
    enable_logging: bool = True,
    enable_performance: bool = True,
    cors_config: Optional[Dict[str, Any]] = None,
    trusted_hosts: Optional[list] = None,
) -> None:
    """
    Configure all middleware components for the FastAPI application
    """
    logger = logging.getLogger(__name__)

    # Performance monitoring (first to catch total request time)
    if enable_performance:
        app.add_middleware(PerformanceMiddleware, slow_request_threshold=2.0)
        logger.info("✅ Performance monitoring middleware enabled")

    # Security headers
    if enable_security:
        app.add_middleware(SecurityMiddleware, enable_security_headers=True)
        logger.info("✅ Security middleware enabled")

    # Request/response logging
    if enable_logging:
        app.add_middleware(LoggingMiddleware, log_requests=True, log_responses=False)
        logger.info("✅ Logging middleware enabled")

    # Trusted host middleware
    if trusted_hosts:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)
        logger.info(f"✅ Trusted host middleware enabled: {trusted_hosts}")

    # CORS middleware (last to process OPTIONS requests properly)
    if enable_cors:
        cors_settings = cors_config or {
            "allow_origins": ["*"],  # Configure properly for production
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }

        app.add_middleware(CORSMiddleware, **cors_settings)
        logger.info("✅ CORS middleware enabled")


def configure_production_middleware(app: FastAPI) -> None:
    """
    Configure middleware for production environment with security focus
    """
    configure_middleware(
        app=app,
        enable_cors=True,
        enable_security=True,
        enable_logging=True,
        enable_performance=True,
        cors_config={
            "allow_origins": ["https://your-domain.com"],  # Restrict origins
            "allow_credentials": True,
            "allow_methods": ["GET", "POST", "PUT", "DELETE"],
            "allow_headers": ["Authorization", "Content-Type"],
        },
        trusted_hosts=["your-domain.com", "api.your-domain.com"],
    )


def configure_development_middleware(app: FastAPI) -> None:
    """
    Configure middleware for development environment with permissive settings
    """
    configure_middleware(
        app=app,
        enable_cors=True,
        enable_security=False,  # Disable for easier development
        enable_logging=True,
        enable_performance=True,
        cors_config={
            "allow_origins": ["*"],
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        },
    )
