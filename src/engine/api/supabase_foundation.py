"""
METIS API Foundation with Supabase Authentication
Simplified API layer that validates Supabase JWT tokens from frontend
"""

import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import uuid4

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

from src.engine.models.data_contracts import (
    EngagementContext,
    CognitiveState,
    WorkflowState,
)
from src.core.supabase_auth_middleware import (
    get_current_user,
    get_optional_user,
    SupabaseUser,
    has_admin_role,
    can_access_resource,
)
from src.core.audit_trail import get_audit_manager, AuditEventType, AuditSeverity
from src.factories.engine_factory import CognitiveEngineFactory


# API Models
class EngagementCreateRequest(BaseModel):
    """Request model for creating new engagement"""

    problem_statement: str = Field(..., min_length=10, max_length=5000)
    business_context: Dict[str, Any] = Field(default_factory=dict)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)

    @validator("problem_statement")
    def validate_problem_statement(cls, v):
        if not v.strip():
            raise ValueError("Problem statement cannot be empty")
        return v.strip()


class EngagementResponse(BaseModel):
    """Response model for engagement operations"""

    engagement_id: str
    status: str
    created_at: str
    problem_statement: str
    business_context: Dict[str, Any]
    cognitive_state: Dict[str, Any]
    workflow_state: Dict[str, Any]
    user_id: str  # Added for Supabase user tracking


class CognitiveAnalysisRequest(BaseModel):
    """Request model for cognitive analysis"""

    engagement_id: str
    force_model_selection: Optional[List[str]] = None
    analysis_preferences: Dict[str, Any] = Field(default_factory=dict)


class CognitiveAnalysisResponse(BaseModel):
    """Response model for cognitive analysis results"""

    engagement_id: str
    analysis_id: str
    status: str
    cognitive_results: Dict[str, Any]
    transparency_data: Dict[str, Any]
    processing_time_ms: float
    model_selections: List[str]
    confidence_scores: Dict[str, float]


class MetisSupabaseAPIFoundation:
    """
    Simplified METIS API Foundation using Supabase Authentication
    Validates Supabase JWT tokens instead of maintaining custom auth system
    """

    def __init__(self):
        """Initialize API foundation"""
        self.logger = logging.getLogger(__name__)
        self.app: Optional[FastAPI] = None
        self.cognitive_engine = None
        self.audit_manager = None
        self.is_initialized = False

        # Performance tracking
        self.metrics = {
            "total_requests": 0,
            "authenticated_requests": 0,
            "failed_auth": 0,
            "avg_response_time_ms": 0.0,
            "active_engagements": 0,
        }

    async def initialize(self) -> bool:
        """Initialize API foundation with Supabase auth"""
        try:
            if not FASTAPI_AVAILABLE:
                self.logger.error("FastAPI not available")
                return False

            # Create FastAPI app
            self.app = FastAPI(
                title="METIS Cognitive Platform API",
                description="Enterprise cognitive intelligence platform with Supabase authentication",
                version="2.0.0",
                docs_url="/api/docs",
                redoc_url="/api/redoc",
            )

            # Setup CORS
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["http://localhost:3001", "http://localhost:3000"],
                allow_credentials=True,
                allow_methods=["GET", "POST", "PUT", "DELETE"],
                allow_headers=["*"],
            )

            # Initialize cognitive engine
            self.cognitive_engine = CognitiveEngineFactory.create_engine(
                {
                    "enable_hmw_generation": True,
                    "enable_assumption_challenging": True,
                    "enable_research_augmentation": True,
                }
            )

            # Initialize audit manager
            self.audit_manager = await get_audit_manager()

            # Setup routes
            self._setup_routes()

            # Add request tracking middleware
            @self.app.middleware("http")
            async def track_requests(request: Request, call_next):
                start_time = time.time()
                self.metrics["total_requests"] += 1

                # Check if request is authenticated
                user = await get_optional_user(request)
                if user:
                    self.metrics["authenticated_requests"] += 1

                response = await call_next(request)

                # Track response time
                process_time = (time.time() - start_time) * 1000
                self.metrics["avg_response_time_ms"] = (
                    self.metrics["avg_response_time_ms"]
                    * (self.metrics["total_requests"] - 1)
                    + process_time
                ) / self.metrics["total_requests"]

                response.headers["X-Process-Time"] = str(process_time)
                return response

            self.is_initialized = True
            self.logger.info("✅ METIS Supabase API Foundation initialized")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize API foundation: {e}")
            return False

    def _setup_routes(self):
        """Setup API routes with Supabase authentication"""

        # Health check (no auth required)
        @self.app.get("/api/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "auth_type": "supabase",
                "metrics": self.metrics,
            }

        # User info (requires auth)
        @self.app.get("/api/user/profile")
        async def get_user_profile(
            current_user: SupabaseUser = Depends(get_current_user),
        ):
            """Get current user profile from Supabase token"""
            await self._audit_request("get_user_profile", current_user.user_id)

            return {
                "user_id": current_user.user_id,
                "email": current_user.email,
                "role": current_user.role,
                "metadata": current_user.metadata,
                "app_metadata": current_user.app_metadata,
            }

        # Create engagement (requires auth)
        @self.app.post("/api/engagement/create", response_model=EngagementResponse)
        async def create_engagement(
            request: EngagementCreateRequest,
            current_user: SupabaseUser = Depends(get_current_user),
        ):
            """Create new cognitive engagement"""
            try:
                await self._audit_request(
                    "create_engagement",
                    current_user.user_id,
                    {"problem_length": len(request.problem_statement)},
                )

                # Create engagement with user context
                engagement_context = EngagementContext(
                    engagement_id=str(uuid4()),
                    user_id=current_user.user_id,
                    problem_statement=request.problem_statement,
                    business_context=request.business_context,
                    user_preferences=request.user_preferences,
                    created_at=datetime.utcnow(),
                )

                # Initialize cognitive and workflow states
                cognitive_state = CognitiveState(
                    engagement_id=engagement_context.engagement_id,
                    current_phase="initiated",
                    selected_models=[],
                    processing_status="ready",
                )

                workflow_state = WorkflowState(
                    engagement_id=engagement_context.engagement_id,
                    current_stage="preparation",
                    completed_stages=[],
                    pending_approvals=[],
                )

                self.metrics["active_engagements"] += 1

                return EngagementResponse(
                    engagement_id=engagement_context.engagement_id,
                    status="created",
                    created_at=engagement_context.created_at.isoformat(),
                    problem_statement=engagement_context.problem_statement,
                    business_context=engagement_context.business_context,
                    cognitive_state=cognitive_state.__dict__,
                    workflow_state=workflow_state.__dict__,
                    user_id=current_user.user_id,
                )

            except Exception as e:
                await self._audit_error(
                    "create_engagement", current_user.user_id, str(e)
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to create engagement: {str(e)}",
                )

        # Get engagement (requires auth + ownership)
        @self.app.get(
            "/api/engagement/{engagement_id}", response_model=EngagementResponse
        )
        async def get_engagement(
            engagement_id: str, current_user: SupabaseUser = Depends(get_current_user)
        ):
            """Get engagement details"""
            try:
                # TODO: Implement actual engagement retrieval from database
                # For now, return mock data with ownership check

                # Mock ownership check - in real implementation, get from database
                engagement_owner_id = (
                    current_user.user_id
                )  # Mock: user owns their own engagements

                if not can_access_resource(current_user, engagement_owner_id):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Access denied to this engagement",
                    )

                await self._audit_request(
                    "get_engagement",
                    current_user.user_id,
                    {"engagement_id": engagement_id},
                )

                # Return mock engagement data
                return EngagementResponse(
                    engagement_id=engagement_id,
                    status="active",
                    created_at=datetime.utcnow().isoformat(),
                    problem_statement="Sample problem statement",
                    business_context={},
                    cognitive_state={"current_phase": "analysis"},
                    workflow_state={"current_stage": "processing"},
                    user_id=current_user.user_id,
                )

            except HTTPException:
                raise
            except Exception as e:
                await self._audit_error("get_engagement", current_user.user_id, str(e))
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to retrieve engagement: {str(e)}",
                )

        # Start cognitive analysis (requires auth + ownership)
        @self.app.post(
            "/api/engagement/{engagement_id}/analyze",
            response_model=CognitiveAnalysisResponse,
        )
        async def start_cognitive_analysis(
            engagement_id: str,
            request: CognitiveAnalysisRequest,
            current_user: SupabaseUser = Depends(get_current_user),
        ):
            """Start cognitive analysis for engagement"""
            try:
                # Check ownership (mock implementation)
                engagement_owner_id = current_user.user_id

                if not can_access_resource(current_user, engagement_owner_id):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Access denied to this engagement",
                    )

                await self._audit_request(
                    "start_analysis",
                    current_user.user_id,
                    {
                        "engagement_id": engagement_id,
                        "forced_models": request.force_model_selection,
                    },
                )

                # Start analysis with cognitive engine
                start_time = time.time()

                # Mock analysis result
                analysis_result = {
                    "primary_insights": ["Insight 1", "Insight 2"],
                    "recommendations": ["Recommendation 1", "Recommendation 2"],
                    "confidence_level": 0.85,
                }

                processing_time = (time.time() - start_time) * 1000

                return CognitiveAnalysisResponse(
                    engagement_id=engagement_id,
                    analysis_id=str(uuid4()),
                    status="completed",
                    cognitive_results=analysis_result,
                    transparency_data={"reasoning_chain": ["Step 1", "Step 2"]},
                    processing_time_ms=processing_time,
                    model_selections=request.force_model_selection or ["default_model"],
                    confidence_scores={"overall": 0.85},
                )

            except HTTPException:
                raise
            except Exception as e:
                await self._audit_error("start_analysis", current_user.user_id, str(e))
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to start analysis: {str(e)}",
                )

        # List user engagements (requires auth)
        @self.app.get("/api/user/engagements")
        async def list_user_engagements(
            current_user: SupabaseUser = Depends(get_current_user),
            limit: int = 10,
            offset: int = 0,
        ):
            """List engagements for current user"""
            try:
                await self._audit_request(
                    "list_engagements",
                    current_user.user_id,
                    {"limit": limit, "offset": offset},
                )

                # TODO: Implement actual database query
                # Mock response
                return {
                    "engagements": [],
                    "total": 0,
                    "limit": limit,
                    "offset": offset,
                    "user_id": current_user.user_id,
                }

            except Exception as e:
                await self._audit_error(
                    "list_engagements", current_user.user_id, str(e)
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to list engagements: {str(e)}",
                )

        # Admin endpoints (requires admin role)
        @self.app.get("/api/admin/metrics")
        async def get_admin_metrics(
            current_user: SupabaseUser = Depends(get_current_user),
        ):
            """Get admin metrics"""
            if not has_admin_role(current_user):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Admin access required",
                )

            await self._audit_request("get_admin_metrics", current_user.user_id)

            return {
                "api_metrics": self.metrics,
                "timestamp": datetime.utcnow().isoformat(),
                "admin_user": current_user.email,
            }

    async def _audit_request(
        self, action: str, user_id: str, metadata: Optional[Dict] = None
    ):
        """Audit successful request"""
        if self.audit_manager:
            await self.audit_manager.log_event(
                event_type=AuditEventType.API_REQUEST,
                severity=AuditSeverity.INFO,
                user_id=user_id,
                resource_id=None,
                action=action,
                metadata=metadata or {},
                outcome="success",
            )

    async def _audit_error(self, action: str, user_id: str, error: str):
        """Audit failed request"""
        self.metrics["failed_auth"] += 1
        if self.audit_manager:
            await self.audit_manager.log_event(
                event_type=AuditEventType.API_REQUEST,
                severity=AuditSeverity.ERROR,
                user_id=user_id,
                resource_id=None,
                action=action,
                metadata={"error": error},
                outcome="failure",
            )


# Global API foundation instance
_api_foundation: Optional[MetisSupabaseAPIFoundation] = None


async def get_supabase_api_foundation() -> MetisSupabaseAPIFoundation:
    """Get or create API foundation instance"""
    global _api_foundation
    if _api_foundation is None:
        _api_foundation = MetisSupabaseAPIFoundation()
        await _api_foundation.initialize()
    return _api_foundation


def get_fastapi_app() -> FastAPI:
    """Get FastAPI app instance (for uvicorn)"""
    global _api_foundation

    # Create app directly for uvicorn (which handles its own event loop)
    if _api_foundation is None:
        _api_foundation = MetisSupabaseAPIFoundation()
        # Initialize synchronously to avoid event loop conflicts
        if not FASTAPI_AVAILABLE:
            raise RuntimeError("FastAPI not available")

        # Create FastAPI app directly
        _api_foundation.app = FastAPI(
            title="METIS Cognitive Platform API",
            description="Enterprise cognitive intelligence platform with Supabase authentication",
            version="2.0.0",
            docs_url="/api/docs",
            redoc_url="/api/redoc",
        )

        # Setup CORS
        _api_foundation.app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:3001", "http://localhost:3000"],
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["*"],
        )

        # Setup routes immediately
        _api_foundation._setup_routes()

        # Include progressive questions router
        try:
            from src.engine.api.progressive_questions import router as pq_router

            _api_foundation.app.include_router(pq_router)
        except ImportError as e:
            print(f"Warning: Could not load progressive questions router: {e}")

        # Include research history router
        try:
            from src.engine.api.research_history_api import router as research_router

            _api_foundation.app.include_router(research_router)
        except ImportError as e:
            print(f"Warning: Could not load research history router: {e}")

        # Include Socratic Forge API router (Step 2 Integration)
        try:
            from src.engine.api.socratic_forge_api import (
                router as socratic_forge_router,
            )

            _api_foundation.app.include_router(socratic_forge_router)
            print("✅ Socratic Forge API router registered")
        except ImportError as e:
            print(f"Warning: Could not load Socratic Forge API router: {e}")

        # Include Markdown Output API router (Level 3 Quality - Markdown Verification)
        try:
            from src.engine.api.markdown_output_api import router as markdown_router

            _api_foundation.app.include_router(markdown_router)
            print("✅ Markdown Output API router registered")
        except ImportError as e:
            print(f"Warning: Could not load Markdown Output API router: {e}")

        # Include Flywheel Management API router (Operation Phoenix Integration)
        try:
            from src.engine.api.flywheel_management_api import (
                flywheel_management_router,
            )

            _api_foundation.app.include_router(flywheel_management_router)
            print("✅ Operation Phoenix: Flywheel Management API router registered")
        except ImportError as e:
            print(f"Warning: Could not load Flywheel Management API router: {e}")

        # Include Monte Carlo Benchmarking API router (Operation Phoenix Phase 4)
        try:
            from src.engine.api.benchmarking_api import router as benchmarking_router

            _api_foundation.app.include_router(benchmarking_router)
            print(
                "✅ Operation Phoenix Phase 4: Monte Carlo Benchmarking API router registered"
            )
        except ImportError as e:
            print(f"Warning: Could not load Monte Carlo Benchmarking API router: {e}")

        # Include C2 Command Center API router (Operation C2 Integration)
        try:
            from src.engine.api.c2_command_center_api import c2_command_center_router

            _api_foundation.app.include_router(c2_command_center_router)
            print("✅ Operation C2: Command Center API router registered")
        except ImportError as e:
            print(f"Warning: Could not load C2 Command Center API router: {e}")

        _api_foundation.is_initialized = True

    if not _api_foundation.app:
        raise RuntimeError("API foundation app not initialized")
    return _api_foundation.app
