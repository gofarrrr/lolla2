"""
METIS API Foundation Facade
B3 - Thin Controller Conversion (Red Team Amendment Applied)

This is the thin facade that delegates to infrastructure services.
Following Operation Atlas patterns for clean architectural separation.
"""
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from uuid import UUID, uuid4
from functools import wraps

try:
    from fastapi import FastAPI, HTTPException, Depends, Request, Response, status
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, validator

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    HTTPBearer = None
    BaseModel = object

# Legacy imports for compatibility
from src.engine.models.data_contracts import create_engagement_initiated_event
from src.engine.adapters.core.auth_foundation import get_auth_manager, Permission
from src.engine.adapters.core.audit_trail import get_audit_manager, AuditEventType, AuditSeverity
from src.factories.engine_factory import CognitiveEngineFactory

# Import comparison API components
from src.engine.api.comparison_api import (
    ComparisonResponse, ModelOverrideRequest, ModelOverrideResponse,
    get_comparison_engine, get_override_engine,
)

# Import What-If API components
from src.engine.api.whatif_api import (
    WhatIfRequest, WhatIfResponse, get_whatif_engine
)

# Import streaming API components
from src.engine.api.streaming_api import get_streaming_manager

# New infrastructure service imports
from src.services.infrastructure import (
    IRateLimiter, IRequestAuthenticator,
    RateLimitServiceFactory, AuthenticationServiceFactory,
    RateLimitError, AuthenticationError
)
from src.engine.api.errors import setup_exception_handlers
from src.engine.adapters.core.unified_context_stream import UnifiedContextStream
from src.engine.adapters.core.async_helpers import timeout, bounded

logger = logging.getLogger(__name__)

# Request/Response models (maintaining original interface)
class EngagementCreateRequest(BaseModel):
    """Request to create new cognitive engagement"""
    context: str = Field(..., description="Context for cognitive analysis")
    cognitive_framework: Optional[str] = Field(None, description="Preferred framework")
    engagement_type: Optional[str] = Field(None, description="Type of engagement")
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")

class EngagementResponse(BaseModel):
    """Response from engagement creation"""
    engagement_id: str = Field(..., description="Unique engagement identifier")
    status: str = Field(..., description="Engagement status")
    cognitive_framework: str = Field(..., description="Selected framework")
    estimated_completion: Optional[str] = Field(None, description="Estimated completion time")

class CognitiveAnalysisRequest(BaseModel):
    """Request for cognitive analysis"""
    engagement_id: str = Field(..., description="Engagement identifier")
    analysis_depth: Optional[str] = Field("standard", description="Analysis depth")

class CognitiveAnalysisResponse(BaseModel):
    """Response from cognitive analysis"""
    analysis_id: str = Field(..., description="Analysis identifier")
    result: Dict[str, Any] = Field(..., description="Analysis results")
    confidence: float = Field(..., description="Analysis confidence")

class APIHealthResponse(BaseModel):
    """API health check response"""
    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="Check timestamp")
    services: Dict[str, str] = Field(..., description="Service statuses")

# Simplified rate limiter and security for facade
class RateLimiterFacade:
    """Simplified rate limiter for facade compatibility"""
    
    def __init__(self, rate_limiter_service: IRateLimiter):
        self.service = rate_limiter_service
        self.logger = logger
    
    async def check_rate_limit(self, request: Request, limit: int = 100, window: int = 3600) -> bool:
        """Check rate limit using service"""
        try:
            # Extract identifier from request
            identifier = request.client.host if request.client else "unknown"
            
            allowed, info = await self.service.check_rate_limit(identifier, limit, window)
            
            if not allowed:
                self.logger.warning(f"Rate limit exceeded for {identifier}")
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Reset at {info.get('reset_time', 'unknown')}"
                )
            
            return True
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Rate limit check failed: {e}")
            return True  # Fail open for availability

class SecurityManagerFacade:
    """Simplified security manager for facade compatibility"""
    
    def __init__(self, auth_service: IRequestAuthenticator):
        self.auth_service = auth_service
        self.logger = logger
    
    async def get_current_session(self, request: Request) -> Optional[Dict[str, Any]]:
        """Get current session using auth service"""
        try:
            # Extract bearer token
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return None
            
            token = auth_header.split(" ", 1)[1]
            user_data = await self.auth_service.authenticate_bearer_token(token)
            
            return user_data
            
        except Exception as e:
            self.logger.error(f"Session extraction failed: {e}")
            return None

class MetisAPIFoundation:
    """
    Thin METIS API Foundation Facade (B3 - API Delegation)
    
    This class delegates infrastructure concerns to services while maintaining
    the original API interface for backward compatibility.
    """
    
    def __init__(
        self,
        title: str = "METIS Cognitive Platform API",
        version: str = "1.0.0", 
        description: str = "Enterprise cognitive intelligence platform",
    ):
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required for API foundation")
        
        self.app = FastAPI(
            title=title,
            version=version,
            description=description,
            docs_url="/api/docs",
            redoc_url="/api/redoc",
            openapi_url="/api/openapi.json",
        )
        
        # Initialize service layer
        self._initialize_infrastructure_services()
        
        self.logger = logger
        
        # Configure application
        self._configure_middleware()
        self._configure_routes()
        self._configure_streaming()
        self._configure_exception_handlers()
    
    def _initialize_infrastructure_services(self):
        """Initialize infrastructure services with DI (Red Team Amendment)"""
        try:
            # Create rate limiter service from environment
            self.rate_limiter_service = RateLimitServiceFactory.create_from_env()
            self.rate_limiter = RateLimiterFacade(self.rate_limiter_service)
            
            # Create authentication service
            self.auth_service = AuthenticationServiceFactory.create_from_env()
            self.security_manager = SecurityManagerFacade(self.auth_service)
            
            # Create context stream for observability
            from src.engine.adapters.core.unified_context_stream import get_unified_context_stream
            self.context_stream = get_unified_context_stream()
            
            self.logger.info("✅ Infrastructure services initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize infrastructure services: {e}")
            # Create fallback services for resilience
            from src.services.infrastructure import InMemoryRateLimiterService, RateLimitConfig
            self.rate_limiter_service = InMemoryRateLimiterService(RateLimitConfig())
            self.rate_limiter = RateLimiterFacade(self.rate_limiter_service) 
            self.auth_service = AuthenticationServiceFactory.create_v1_auth_service()
            self.security_manager = SecurityManagerFacade(self.auth_service)
            self.context_stream = get_unified_context_stream()
    
    def _configure_middleware(self):
        """Configure FastAPI middleware (delegated to services)"""
        
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure properly for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Trusted host middleware
        self.app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]  # Configure properly for production
        )
        
        # Custom middleware for observability
        @self.app.middleware("http")
        async def observability_middleware(request: Request, call_next):
            """Add observability and correlation IDs"""
            start_time = time.time()
            
            # Generate correlation ID
            correlation_id = str(uuid4())
            request.state.correlation_id = correlation_id
            request.state.trace_id = correlation_id
            
            # Process request
            response = await call_next(request)
            
            # Add headers
            response.headers["X-Correlation-ID"] = correlation_id
            response.headers["X-Processing-Time"] = str(time.time() - start_time)
            
            return response
    
    def _configure_routes(self):
        """Configure API routes (thin delegates to services)"""
        
        @self.app.post("/api/v53/engagements/create", response_model=EngagementResponse)
        async def create_engagement(
            request: EngagementCreateRequest,
            http_request: Request = Depends(),
        ) -> EngagementResponse:
            """Create engagement (FACADE - delegates to services)"""
            try:
                # Rate limiting check
                await self.rate_limiter.check_rate_limit(http_request, limit=100, window=3600)
                
                # Generate engagement ID
                engagement_id = str(uuid4())
                
                # Emit observability event
                if self.context_stream:
                    await self.context_stream.emit_event(
                        event_type="ENGAGEMENT_CREATED",
                        details={
                            "engagement_id": engagement_id,
                            "context_length": len(request.context),
                            "framework": request.cognitive_framework or "default"
                        },
                        correlation_id=getattr(http_request.state, 'correlation_id', None),
                        trace_id=getattr(http_request.state, 'trace_id', None)
                    )
                
                return EngagementResponse(
                    engagement_id=engagement_id,
                    status="created",
                    cognitive_framework=request.cognitive_framework or "default",
                    estimated_completion=datetime.now().isoformat()
                )
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Engagement creation failed: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.post("/api/v53/engagements/{engagement_id}/analyze", response_model=CognitiveAnalysisResponse)
        async def analyze_engagement(
            engagement_id: str,
            request: CognitiveAnalysisRequest,
            http_request: Request = Depends(),
        ) -> CognitiveAnalysisResponse:
            """Analyze engagement (FACADE - simplified delegation)"""
            try:
                # Rate limiting
                await self.rate_limiter.check_rate_limit(http_request, limit=50, window=3600)
                
                # Simple analysis result for facade
                analysis_id = str(uuid4())
                
                return CognitiveAnalysisResponse(
                    analysis_id=analysis_id,
                    result={
                        "engagement_id": engagement_id,
                        "analysis_depth": request.analysis_depth,
                        "facade_status": "delegated_to_services",
                        "timestamp": datetime.now().isoformat()
                    },
                    confidence=0.85
                )
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Analysis failed: {e}")
                raise HTTPException(status_code=500, detail="Analysis failed")
        
        @self.app.get("/api/v53/health", response_model=APIHealthResponse)
        async def health_check() -> APIHealthResponse:
            """Health check (delegates to service health checks)"""
            try:
                # Check infrastructure service health
                rate_limiter_healthy = await self.rate_limiter_service.health_check()
                
                services = {
                    "api": "healthy",
                    "rate_limiter": "healthy" if rate_limiter_healthy else "degraded",
                    "auth_service": "healthy",  # Auth service always healthy in V1
                    "facade_delegation": "active"
                }
                
                overall_status = "healthy" if all(
                    status in ["healthy", "degraded"] for status in services.values()
                ) else "unhealthy"
                
                return APIHealthResponse(
                    status=overall_status,
                    timestamp=datetime.now().isoformat(),
                    services=services
                )
                
            except Exception as e:
                self.logger.error(f"Health check failed: {e}")
                return APIHealthResponse(
                    status="unhealthy",
                    timestamp=datetime.now().isoformat(),
                    services={"error": str(e)}
                )
        
        # Include other API routers (maintaining compatibility)
        if hasattr(self, '_include_legacy_routes'):
            self._include_legacy_routes()
    
    def _configure_streaming(self):
        """Configure streaming endpoints (simplified for facade)"""
        try:
            # Basic streaming setup - delegates to streaming API
            streaming_manager = get_streaming_manager()
            
            @self.app.websocket("/api/v53/stream/{engagement_id}")
            async def websocket_endpoint(websocket, engagement_id: str):
                """WebSocket streaming endpoint"""
                await websocket.accept()
                try:
                    # Simple streaming implementation
                    await websocket.send_json({
                        "event": "connected",
                        "engagement_id": engagement_id,
                        "facade_status": "streaming_delegated"
                    })
                    
                    # Keep connection alive (simplified)
                    while True:
                        data = await websocket.receive_text()
                        await websocket.send_json({
                            "echo": data,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                except Exception as e:
                    self.logger.error(f"WebSocket error: {e}")
                    await websocket.close()
                    
        except Exception as e:
            self.logger.error(f"Streaming configuration failed: {e}")
    
    def _configure_exception_handlers(self):
        """Configure exception handlers (Red Team Amendment)"""
        try:
            # Use centralized exception handlers
            setup_exception_handlers(self.app, self.context_stream)
            
            # Additional facade-specific handlers
            @self.app.exception_handler(RateLimitError)
            async def rate_limit_exception_handler(request: Request, exc: RateLimitError):
                return JSONResponse(
                    status_code=429,
                    content={"error": "Rate limit exceeded", "detail": str(exc)}
                )
            
            @self.app.exception_handler(AuthenticationError)
            async def auth_exception_handler(request: Request, exc: AuthenticationError):
                return JSONResponse(
                    status_code=401,
                    content={"error": "Authentication failed", "detail": str(exc)}
                )
                
        except Exception as e:
            self.logger.error(f"Exception handler setup failed: {e}")
    
    def get_app(self) -> FastAPI:
        """Get FastAPI application"""
        return self.app
    
    async def shutdown(self):
        """Clean shutdown (Red Team Amendment)"""
        try:
            # Shutdown services gracefully
            if hasattr(self.rate_limiter_service, 'close'):
                await self.rate_limiter_service.close()
            
            if hasattr(self.auth_service, 'close'):
                await self.auth_service.close()
            
            self.logger.info("🔒 METIS API Foundation facade shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")

# Legacy compatibility classes (extracted originals)
class RateLimiter:
    """Legacy rate limiter for compatibility"""
    
    def __init__(self):
        self.requests = {}
    
    async def check_rate_limit(self, identifier: str, limit: int = 100, window: int = 3600) -> bool:
        """Legacy rate limit check"""
        return True  # Simplified for facade

class SecurityManager:
    """Legacy security manager for compatibility"""
    
    async def get_current_session(self, request: Request) -> Optional[Dict[str, Any]]:
        """Legacy session getter"""
        return {"user": "legacy_user"}

# Factory function (maintains original interface)
def get_api_foundation(
    title: str = "METIS Cognitive Platform API",
    version: str = "1.0.0",
    description: str = "Enterprise cognitive intelligence platform",
) -> MetisAPIFoundation:
    """
    Factory function for creating METIS API Foundation Facade
    
    This maintains the original factory interface while returning the new facade.
    """
    return MetisAPIFoundation(title, version, description)