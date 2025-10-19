"""
API Foundation - Core FastAPI application setup
Extracted from foundation.py for modular API architecture
"""

import logging
from typing import Optional
from dataclasses import dataclass

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


@dataclass
class APIFoundationConfig:
    """Configuration for API foundation setup"""

    title: str = "METIS Cognitive Platform API"
    version: str = "1.0.0"
    description: str = "Enterprise cognitive intelligence platform"
    docs_url: str = "/api/docs"
    redoc_url: str = "/api/redoc"
    openapi_url: str = "/api/openapi.json"
    enable_cors: bool = True
    enable_security: bool = True
    enable_rate_limiting: bool = True
    enable_audit_logging: bool = True


class APIFoundationCore:
    """
    Core API foundation providing basic FastAPI application setup
    """

    def __init__(self, config: APIFoundationConfig):
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required for API foundation")

        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize FastAPI app
        self.app = FastAPI(
            title=config.title,
            version=config.version,
            description=config.description,
            docs_url=config.docs_url,
            redoc_url=config.redoc_url,
            openapi_url=config.openapi_url,
        )

        # Configure global exception handlers
        self._configure_exception_handlers()

        self.logger.info(
            f"🚀 API Foundation initialized: {config.title} v{config.version}"
        )

    def _configure_exception_handlers(self):
        """Configure global exception handlers"""

        @self.app.exception_handler(404)
        async def not_found_handler(request, exc):
            return JSONResponse(
                status_code=404,
                content={
                    "error": "Not Found",
                    "message": f"The requested resource was not found: {request.url.path}",
                    "timestamp": "2025-08-28T12:00:00Z",
                },
            )

        @self.app.exception_handler(500)
        async def internal_error_handler(request, exc):
            self.logger.error(f"Internal server error: {exc}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": "An unexpected error occurred. Please try again later.",
                    "timestamp": "2025-08-28T12:00:00Z",
                },
            )

    def add_health_endpoint(self):
        """Add basic health check endpoint"""

        @self.app.get("/api/health")
        async def health_check():
            return {
                "status": "healthy",
                "service": self.config.title,
                "version": self.config.version,
                "timestamp": "2025-08-28T12:00:00Z",
            }

    def get_app(self) -> FastAPI:
        """Get the configured FastAPI application"""
        return self.app


def create_api_foundation(
    config: Optional[APIFoundationConfig] = None, add_health: bool = True
) -> APIFoundationCore:
    """
    Factory function to create API foundation with configuration
    """
    if config is None:
        config = APIFoundationConfig()

    foundation = APIFoundationCore(config)

    if add_health:
        foundation.add_health_endpoint()

    return foundation


def create_minimal_api(title: str = "METIS API", version: str = "1.0.0") -> FastAPI:
    """
    Create minimal FastAPI app for testing or simple usage
    """
    config = APIFoundationConfig(
        title=title,
        version=version,
        enable_cors=False,
        enable_security=False,
        enable_rate_limiting=False,
        enable_audit_logging=False,
    )

    foundation = create_api_foundation(config, add_health=True)
    return foundation.get_app()
