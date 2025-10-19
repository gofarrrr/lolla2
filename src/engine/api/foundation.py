"""
METIS API Foundation Framework
F006: FastAPI-based REST API with enterprise patterns and MCP compliance

Implements comprehensive API layer with authentication, rate limiting,
validation, and integration capabilities.
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

from src.engine.models.data_contracts import (
    create_engagement_initiated_event,
)
from src.core.auth_foundation import (
    get_auth_manager,
    Permission,
)
from src.core.audit_trail import get_audit_manager, AuditEventType, AuditSeverity
from src.factories.engine_factory import CognitiveEngineFactory

# Import comparison API components
from src.engine.api.comparison_api import (
    ComparisonResponse,
    ModelOverrideRequest,
    ModelOverrideResponse,
    get_comparison_engine,
    get_override_engine,
)

# Import What-If API components (Day 3 Sprint Implementation)
from src.engine.api.whatif_api import (
    WhatIfRequest,
    WhatIfResponse,
    WhatIfBatchRequest,
    WhatIfBatchResponse,
    get_whatif_engine,
)


# API Models
class EngagementCreateRequest(BaseModel):
    """Request model for creating new engagement"""

    problem_statement: str = Field(..., min_length=10, max_length=5000)
    business_context: Dict[str, Any] = Field(default_factory=dict)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    compliance_requirements: Dict[str, Any] = Field(default_factory=dict)

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


class CognitiveAnalysisRequest(BaseModel):
    """Request model for cognitive analysis"""

    engagement_id: str
    force_model_selection: Optional[List[str]] = None
    analysis_preferences: Dict[str, Any] = Field(default_factory=dict)


class CognitiveAnalysisResponse(BaseModel):
    """Response model for cognitive analysis"""

    engagement_id: str
    analysis_id: str
    cognitive_state: Dict[str, Any]
    reasoning_steps: List[Dict[str, Any]]
    confidence_scores: Dict[str, float]
    processing_time_ms: float


class ModelListResponse(BaseModel):
    """Response model for available mental models"""

    models: List[Dict[str, Any]]
    total_count: int
    categories: List[str]


class AuditTrailResponse(BaseModel):
    """Response model for audit trail queries"""

    engagement_id: str
    total_events: int
    events: List[Dict[str, Any]]
    summary: Dict[str, Any]


class APIHealthResponse(BaseModel):
    """Response model for API health check"""

    status: str
    timestamp: str
    version: str
    components: Dict[str, Any]


# Rate limiting
class RateLimiter:
    """Simple in-memory rate limiter"""

    def __init__(self):
        self.requests: Dict[str, List[float]] = {}
        self.limits = {
            "default": (100, 3600),  # 100 requests per hour
            "authenticated": (
                1000,
                3600,
            ),  # 1000 requests per hour for authenticated users
            "enterprise": (10000, 3600),  # 10000 requests per hour for enterprise
        }

    async def check_rate_limit(
        self, identifier: str, limit_type: str = "default"
    ) -> bool:
        """Check if request is within rate limits"""

        now = time.time()
        limit, window = self.limits.get(limit_type, self.limits["default"])

        if identifier not in self.requests:
            self.requests[identifier] = []

        # Clean old requests
        self.requests[identifier] = [
            req_time
            for req_time in self.requests[identifier]
            if now - req_time < window
        ]

        # Check limit
        if len(self.requests[identifier]) >= limit:
            return False

        # Record this request
        self.requests[identifier].append(now)
        return True


# Security dependencies
class SecurityManager:
    """Security manager for API authentication and authorization"""

    def __init__(self):
        self.security = HTTPBearer() if FASTAPI_AVAILABLE else None
        self.rate_limiter = RateLimiter()

    async def get_current_session(
        self, request: Request, credentials: HTTPAuthorizationCredentials = None
    ):
        """Get current authenticated session"""

        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
            )

        # Extract session ID from bearer token (simplified)
        # In production, this would validate JWT tokens
        try:
            session_id = UUID(credentials.credentials)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token",
            )

        # Validate session
        auth_manager = await get_auth_manager()
        session = await auth_manager.validate_session(session_id)

        if not session:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired session",
            )

        return session

    def require_permission_dependency(self, permission: Permission):
        """Create dependency for permission checking"""

        async def permission_checker(
            request: Request, session=Depends(self.get_current_session)
        ):
            # Check rate limiting
            rate_limit_key = f"user_{session.user_id}"
            if not await self.rate_limiter.check_rate_limit(
                rate_limit_key, "authenticated"
            ):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                )

            # Check permission
            auth_manager = await get_auth_manager()
            has_permission = await auth_manager.check_permission(
                session.session_id, permission
            )

            if not has_permission:
                # Log permission denial
                audit_manager = await get_audit_manager()
                await audit_manager.log_event(
                    event_type=AuditEventType.PERMISSION_DENIED,
                    severity=AuditSeverity.HIGH,
                    user_id=session.user_id,
                    session_id=session.session_id,
                    action_performed=f"access_{permission.value}",
                    event_description=f"Permission denied for {permission.value}",
                    ip_address=request.client.host,
                    user_agent=request.headers.get("user-agent"),
                )

                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions: {permission.value} required",
                )

            return session

        return permission_checker


class MetisAPIFoundation:
    """
    METIS API Foundation implementing enterprise patterns
    FastAPI-based REST API with authentication, validation, and integration
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

        self.security_manager = SecurityManager()
        self.logger = logging.getLogger(__name__)

        # Configure middleware
        self._configure_middleware()

        # Configure routes
        self._configure_routes()

        # Configure streaming endpoints
        self._configure_streaming()

        # Global exception handler
        self._configure_exception_handlers()

    def _configure_middleware(self):
        """Configure FastAPI middleware"""

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
            allowed_hosts=["*"],  # Configure properly for production
        )

        # Request/Response logging middleware
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            start_time = time.time()

            # Log request
            self.logger.info(
                f"{request.method} {request.url.path} - {request.client.host}"
            )

            # Process request
            response = await call_next(request)

            # Log response
            process_time = time.time() - start_time
            self.logger.info(
                f"{request.method} {request.url.path} - "
                f"{response.status_code} - {process_time:.3f}s"
            )

            return response

    def _configure_routes(self):
        """Configure API routes"""

        # Health check
        @self.app.get("/api/health", response_model=APIHealthResponse)
        async def health_check():
            """API health check endpoint"""

            # Check component health
            components = {}

            try:
                auth_manager = await get_auth_manager()
                auth_health = await auth_manager.get_auth_health_status()
                components["authentication"] = auth_health
            except Exception as e:
                components["authentication"] = {"status": "error", "error": str(e)}

            try:
                audit_manager = await get_audit_manager()
                audit_health = await audit_manager.get_audit_health_status()
                components["audit_trail"] = audit_health
            except Exception as e:
                components["audit_trail"] = {"status": "error", "error": str(e)}

            try:
                cognitive_engine = CognitiveEngineFactory.create_engine()
                components["cognitive_engine"] = {"status": "healthy"}
            except Exception as e:
                components["cognitive_engine"] = {"status": "error", "error": str(e)}

            overall_status = (
                "healthy"
                if all(
                    comp.get("status") == "healthy" or "total_" in str(comp)
                    for comp in components.values()
                )
                else "degraded"
            )

            return APIHealthResponse(
                status=overall_status,
                timestamp=datetime.utcnow().isoformat(),
                version="1.0.0",
                components=components,
            )

        # Authentication routes
        @self.app.post("/api/auth/login")
        async def login(request: Request, email: str, password: str):
            """User authentication endpoint"""

            auth_manager = await get_auth_manager()

            session = await auth_manager.authenticate_user(
                email=email,
                password=password,
                ip_address=request.client.host,
                user_agent=request.headers.get("user-agent"),
            )

            if not session:
                # Log failed authentication
                audit_manager = await get_audit_manager()
                await audit_manager.log_authentication_event(
                    event_type=AuditEventType.LOGIN_FAILED,
                    ip_address=request.client.host,
                    user_agent=request.headers.get("user-agent"),
                    authentication_method="password",
                    success=False,
                    failure_reason="Invalid credentials",
                )

                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials",
                )

            # Log successful authentication
            audit_manager = await get_audit_manager()
            await audit_manager.log_authentication_event(
                event_type=AuditEventType.USER_LOGIN,
                user_id=session.user_id,
                session_id=session.session_id,
                ip_address=request.client.host,
                user_agent=request.headers.get("user-agent"),
                authentication_method="password",
                success=True,
            )

            return {
                "access_token": session.access_token,
                "refresh_token": session.refresh_token,
                "session_id": str(session.session_id),
                "expires_at": session.expires_at.isoformat(),
            }

        # Engagement management routes
        @self.app.post("/api/engagements", response_model=EngagementResponse)
        async def create_engagement(
            request: EngagementCreateRequest,
            session=Depends(
                self.security_manager.require_permission_dependency(
                    Permission.CREATE_ENGAGEMENT
                )
            ),
        ):
            """Create new strategic analysis engagement"""

            # Create engagement event
            engagement_event = create_engagement_initiated_event(
                problem_statement=request.problem_statement,
                business_context=request.business_context,
            )

            # Update engagement context with user preferences
            engagement_event.engagement_context.user_preferences = (
                request.user_preferences
            )
            engagement_event.engagement_context.compliance_requirements = (
                request.compliance_requirements
            )

            # Log engagement creation
            audit_manager = await get_audit_manager()
            await audit_manager.log_event(
                event_type=AuditEventType.ENGAGEMENT_CREATED,
                severity=AuditSeverity.MEDIUM,
                user_id=session.user_id,
                session_id=session.session_id,
                engagement_id=engagement_event.engagement_context.engagement_id,
                resource_type="engagement",
                resource_id=engagement_event.engagement_context.engagement_id,
                action_performed="create_engagement",
                event_description=f"Created engagement: {request.problem_statement[:100]}...",
            )

            return EngagementResponse(
                engagement_id=str(engagement_event.engagement_context.engagement_id),
                status="created",
                created_at=engagement_event.engagement_context.created_at.isoformat(),
                problem_statement=engagement_event.engagement_context.problem_statement,
                business_context=engagement_event.engagement_context.business_context,
                cognitive_state=engagement_event.cognitive_state.dict(),
                workflow_state=engagement_event.workflow_state.dict(),
            )

        @self.app.get(
            "/api/engagements/{engagement_id}", response_model=EngagementResponse
        )
        async def get_engagement(
            engagement_id: str,
            session=Depends(
                self.security_manager.require_permission_dependency(
                    Permission.READ_ENGAGEMENT
                )
            ),
        ):
            """Get engagement details"""

            try:
                engagement_uuid = UUID(engagement_id)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid engagement ID format",
                )

            # Log data access
            audit_manager = await get_audit_manager()
            await audit_manager.log_data_access(
                resource_type="engagement",
                resource_id=engagement_uuid,
                action="read",
                user_id=session.user_id,
                session_id=session.session_id,
                organization_id=session.organization_id,
            )

            # For demonstration, return a mock engagement
            # In production, this would retrieve from database
            return EngagementResponse(
                engagement_id=engagement_id,
                status="active",
                created_at=datetime.utcnow().isoformat(),
                problem_statement="Sample engagement",
                business_context={},
                cognitive_state={},
                workflow_state={},
            )

        # Cognitive analysis routes
        @self.app.post(
            "/api/engagements/{engagement_id}/analyze",
            response_model=CognitiveAnalysisResponse,
        )
        async def execute_cognitive_analysis(
            engagement_id: str,
            request: CognitiveAnalysisRequest,
            session=Depends(
                self.security_manager.require_permission_dependency(
                    Permission.EXECUTE_ANALYSIS
                )
            ),
        ):
            """Execute cognitive analysis on engagement"""

            try:
                engagement_uuid = UUID(engagement_id)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid engagement ID format",
                )

            start_time = time.time()

            # Create mock engagement contract for processing
            engagement_contract = create_engagement_initiated_event(
                problem_statement="Analysis request", business_context={}
            )
            engagement_contract.engagement_context.engagement_id = engagement_uuid

            # Process through cognitive engine
            cognitive_engine = CognitiveEngineFactory.create_engine()
            processed_contract = await cognitive_engine.process_engagement(
                engagement_contract
            )

            processing_time = (time.time() - start_time) * 1000

            # Log analysis execution
            audit_manager = await get_audit_manager()
            await audit_manager.log_reasoning_trace(
                engagement_id=engagement_uuid,
                reasoning_steps=processed_contract.cognitive_state.reasoning_steps,
                user_id=session.user_id,
                session_id=session.session_id,
            )

            return CognitiveAnalysisResponse(
                engagement_id=engagement_id,
                analysis_id=str(uuid4()),
                cognitive_state=processed_contract.cognitive_state.dict(),
                reasoning_steps=[
                    step.dict()
                    for step in processed_contract.cognitive_state.reasoning_steps
                ],
                confidence_scores=processed_contract.cognitive_state.confidence_scores,
                processing_time_ms=processing_time,
            )

        # Mental models routes
        @self.app.get("/api/models", response_model=ModelListResponse)
        async def list_mental_models(
            session=Depends(
                self.security_manager.require_permission_dependency(
                    Permission.SELECT_MODELS
                )
            ),
        ):
            """List available mental models"""

            cognitive_engine = CognitiveEngineFactory.create_engine()
            models = await cognitive_engine.get_available_models()

            # Log data access
            audit_manager = await get_audit_manager()
            await audit_manager.log_data_access(
                resource_type="mental_models",
                resource_id=uuid4(),
                action="list",
                user_id=session.user_id,
                session_id=session.session_id,
            )

            return ModelListResponse(
                models=[model.dict() for model in models],
                total_count=len(models),
                categories=list(set(model.category.value for model in models)),
            )

        # Audit trail routes
        @self.app.get(
            "/api/engagements/{engagement_id}/audit", response_model=AuditTrailResponse
        )
        async def get_engagement_audit_trail(
            engagement_id: str,
            include_reasoning: bool = True,
            session=Depends(
                self.security_manager.require_permission_dependency(
                    Permission.VIEW_AUDIT_LOGS
                )
            ),
        ):
            """Get complete audit trail for engagement"""

            try:
                engagement_uuid = UUID(engagement_id)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid engagement ID format",
                )

            audit_manager = await get_audit_manager()

            # Log audit access
            await audit_manager.log_event(
                event_type=AuditEventType.AUDIT_LOG_ACCESSED,
                severity=AuditSeverity.HIGH,
                user_id=session.user_id,
                session_id=session.session_id,
                resource_type="audit_trail",
                resource_id=engagement_uuid,
                action_performed="view_audit_trail",
                event_description=f"Accessed audit trail for engagement {engagement_id}",
            )

            trail = await audit_manager.get_engagement_audit_trail(
                engagement_id=engagement_uuid, include_reasoning=include_reasoning
            )

            return AuditTrailResponse(
                engagement_id=engagement_id,
                total_events=trail["total_events"],
                events=trail["event_timeline"],
                summary=trail["summary"],
            )

        # What-If Sandbox routes
        @self.app.get("/api/v1/engagements/{engagement_id}/assumptions")
        async def get_editable_assumptions(
            engagement_id: str,
            session=Depends(
                self.security_manager.require_permission_dependency(
                    Permission.READ_ENGAGEMENT
                )
            ),
        ):
            """Get editable assumptions for what-if analysis"""
            try:
                engagement_uuid = UUID(engagement_id)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid engagement ID format",
                )

            # Mock assumptions for now - in a real implementation, these would be extracted
            # from the analysis results
            editable_assumptions = [
                {
                    "assumption_id": "market_growth_rate",
                    "display_name": "Market Growth Rate",
                    "current_value": 15,
                    "category": "Market Dynamics",
                    "category_icon": "📈",
                    "data_type": "float",
                    "suggested_range": {"min": 5, "max": 30},
                    "impact_level": "high",
                    "description": "Annual market growth rate assumption",
                },
                {
                    "assumption_id": "customer_acquisition_cost",
                    "display_name": "Customer Acquisition Cost",
                    "current_value": 250,
                    "category": "Financial",
                    "category_icon": "💰",
                    "data_type": "int",
                    "suggested_range": {"min": 100, "max": 500},
                    "impact_level": "medium",
                    "description": "Cost to acquire each new customer",
                },
                {
                    "assumption_id": "competitive_response_time",
                    "display_name": "Competitive Response Time",
                    "current_value": 6,
                    "category": "Competition",
                    "category_icon": "⚔️",
                    "data_type": "int",
                    "suggested_range": {"min": 3, "max": 12},
                    "impact_level": "medium",
                    "description": "Months for competitors to respond",
                },
            ]

            return {"editable_assumptions": editable_assumptions}

        @self.app.post("/api/v1/engagements/{engagement_id}/reevaluate")
        async def reevaluate_with_new_assumption(
            engagement_id: str,
            request: dict,  # Contains assumption_id, new_value, assumption_context
            session=Depends(
                self.security_manager.require_permission_dependency(
                    Permission.UPDATE_ENGAGEMENT
                )
            ),
        ):
            """Re-evaluate engagement with modified assumptions"""
            try:
                engagement_uuid = UUID(engagement_id)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid engagement ID format",
                )

            assumption_id = request.get("assumption_id")
            new_value = request.get("new_value")
            context = request.get("assumption_context", "")

            if not assumption_id or new_value is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="assumption_id and new_value are required",
                )

            # For now, return a mock updated engagement
            # In a real implementation, this would trigger a re-analysis
            updated_engagement = {
                "engagement_id": engagement_id,
                "status": "completed",
                "deliverable_ready": True,
                "phases": {
                    "problem_structuring": {
                        "status": "completed",
                        "execution_time": 25,
                    },
                    "hypothesis_generation": {
                        "status": "completed",
                        "execution_time": 35,
                    },
                    "analysis_execution": {"status": "completed", "execution_time": 65},
                    "synthesis_delivery": {"status": "completed", "execution_time": 15},
                },
                "what_if_modified": True,
                "modified_assumptions": [
                    {
                        "assumption_id": assumption_id,
                        "new_value": new_value,
                        "context": context,
                        "modified_at": datetime.utcnow().isoformat(),
                    }
                ],
            }

            return updated_engagement

        @self.app.get("/api/v1/engagements/{engagement_id}/comparison")
        async def get_engagement_comparison(
            engagement_id: str,
            session=Depends(
                self.security_manager.require_permission_dependency(
                    Permission.READ_ENGAGEMENT
                )
            ),
        ):
            """Get comparison data between base and modified scenarios"""
            try:
                engagement_uuid = UUID(engagement_id)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid engagement ID format",
                )

            # Mock comparison data - in real implementation, this would compare
            # original vs modified analysis results
            comparison_data = {
                "base_scenario": {
                    "name": "Original Analysis",
                    "key_metrics": {
                        "projected_revenue": 1500000,
                        "implementation_time": 12,
                        "risk_score": 6.5,
                    },
                },
                "modified_scenario": {
                    "name": "What-If Scenario",
                    "key_metrics": {
                        "projected_revenue": 1750000,
                        "implementation_time": 10,
                        "risk_score": 7.2,
                    },
                },
                "impact_analysis": {
                    "revenue_change": "+16.7%",
                    "time_savings": "-16.7%",
                    "risk_increase": "+10.8%",
                },
                "recommendation_changes": [
                    "Higher revenue potential identified with adjusted assumptions",
                    "Faster implementation timeline possible",
                    "Increased risk profile requires additional mitigation",
                ],
            }

            return comparison_data

        # WebSocket endpoint for real-time collaboration
        @self.app.websocket("/ws/engagement/{engagement_id}")
        async def websocket_endpoint(
            websocket, engagement_id: str, user_id: str = "demo-user"
        ):
            """WebSocket endpoint for real-time collaboration"""
            try:
                from src.engine.api.websocket_server import get_websocket_manager
                import json

                manager = get_websocket_manager()
                await manager.start_background_tasks()

                # Connect client
                connection_id = await manager.connect_client(
                    websocket=websocket,
                    user_id=user_id,
                    user_name=f"User-{user_id}",
                    engagement_id=engagement_id,
                )

                try:
                    # Message handling loop
                    while True:
                        data = await websocket.receive_text()
                        try:
                            message = json.loads(data)
                            await manager.handle_client_message(connection_id, message)
                        except json.JSONDecodeError:
                            self.logger.warning(
                                f"Invalid JSON from client {connection_id}: {data}"
                            )

                except Exception as e:
                    self.logger.error(f"WebSocket error for {connection_id}: {e}")
                finally:
                    await manager.disconnect_client(connection_id)

            except Exception as e:
                self.logger.error(f"WebSocket connection failed: {e}")
                try:
                    await websocket.close()
                except:
                    pass

        # WebSocket statistics endpoint
        @self.app.get("/api/v1/websocket/stats")
        async def websocket_stats():
            """Get WebSocket server statistics"""
            try:
                from src.engine.api.websocket_server import get_websocket_manager

                manager = get_websocket_manager()
                return manager.get_statistics()
            except ImportError:
                return {"error": "WebSocket server not available"}

        # Comparison and What-If routes (Day 2 Sprint Implementation)
        @self.app.get("/api/engagements/compare", response_model=ComparisonResponse)
        async def compare_engagements(
            ids: str,  # Comma-separated engagement IDs
            dimensions: Optional[
                str
            ] = "strategic_impact,implementation_complexity,risk_profile,roi_potential",
            analysis_depth: str = "standard",
            include_recommendations: bool = True,
            session=Depends(
                self.security_manager.require_permission_dependency(
                    Permission.READ_ENGAGEMENT
                )
            ),
        ):
            """Compare multiple engagement scenarios for What-If analysis"""

            try:
                # Parse engagement IDs
                engagement_ids = [UUID(id.strip()) for id in ids.split(",")]

                if len(engagement_ids) < 2:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="At least 2 engagement IDs required for comparison",
                    )

                if len(engagement_ids) > 5:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Maximum 5 engagements can be compared at once",
                    )

            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid engagement ID format",
                )

            # Parse comparison dimensions
            comparison_dimensions = [d.strip() for d in dimensions.split(",")]

            # Log comparison request
            audit_manager = await get_audit_manager()
            await audit_manager.log_event(
                event_type=AuditEventType.DATA_ACCESSED,
                severity=AuditSeverity.MEDIUM,
                user_id=session.user_id,
                session_id=session.session_id,
                action_performed="compare_scenarios",
                event_description=f"Compared {len(engagement_ids)} scenarios",
                metadata={
                    "engagement_ids": [str(id) for id in engagement_ids],
                    "dimensions": comparison_dimensions,
                },
            )

            # Execute comparison
            comparison_engine = get_comparison_engine()
            comparison_result = await comparison_engine.compare_engagements(
                engagement_ids=engagement_ids,
                comparison_dimensions=comparison_dimensions,
                analysis_depth=analysis_depth,
            )

            return comparison_result

        @self.app.post(
            "/api/engagements/{engagement_id}/override-models",
            response_model=ModelOverrideResponse,
        )
        async def override_models(
            engagement_id: str,
            request: ModelOverrideRequest,
            session=Depends(
                self.security_manager.require_permission_dependency(
                    Permission.EXECUTE_ANALYSIS
                )
            ),
        ):
            """Apply expert model override for power users"""

            try:
                engagement_uuid = UUID(engagement_id)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid engagement ID format",
                )

            # Log override attempt
            audit_manager = await get_audit_manager()
            await audit_manager.log_event(
                event_type=AuditEventType.CONFIGURATION_CHANGED,
                severity=AuditSeverity.HIGH,
                user_id=session.user_id,
                session_id=session.session_id,
                engagement_id=engagement_uuid,
                action_performed="override_models",
                event_description=f"Applied model override: {', '.join(request.forced_models)}",
                metadata={
                    "forced_models": request.forced_models,
                    "rationale": request.rationale,
                    "expert_confidence": request.expert_confidence,
                },
            )

            # Execute override
            override_engine = get_override_engine()
            override_result = await override_engine.apply_model_override(
                engagement_id=engagement_uuid,
                forced_models=request.forced_models,
                rationale=request.rationale,
                override_scope=request.override_scope,
                expert_confidence=request.expert_confidence,
                expected_improvement=request.expected_improvement,
                success_criteria=request.success_criteria,
            )

            return override_result

        # What-If Scenario routes (Day 3 Sprint Implementation)
        @self.app.post("/api/what-if", response_model=WhatIfResponse)
        async def create_whatif_scenario(
            request: WhatIfRequest,
            session=Depends(
                self.security_manager.require_permission_dependency(
                    Permission.CREATE_ENGAGEMENT
                )
            ),
        ):
            """Create a new What-If scenario by branching from base engagement"""

            try:
                base_engagement_uuid = UUID(request.base_engagement_id)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid base engagement ID format",
                )

            # Log What-If scenario creation
            audit_manager = await get_audit_manager()
            await audit_manager.log_event(
                event_type=AuditEventType.ENGAGEMENT_CREATED,
                severity=AuditSeverity.MEDIUM,
                user_id=session.user_id,
                session_id=session.session_id,
                engagement_id=base_engagement_uuid,
                action_performed="create_whatif_scenario",
                event_description=f"Created What-If scenario: {request.scenario_name}",
                metadata={
                    "base_engagement_id": request.base_engagement_id,
                    "scenario_name": request.scenario_name,
                    "parameter_changes_count": len(request.parameter_changes),
                    "execution_mode": request.execution_mode,
                },
            )

            # Execute What-If creation
            whatif_engine = get_whatif_engine()
            whatif_result = await whatif_engine.create_whatif_scenario(
                base_engagement_id=base_engagement_uuid,
                scenario_name=request.scenario_name,
                scenario_description=request.scenario_description,
                parameter_changes=request.parameter_changes,
                execution_mode=request.execution_mode,
                preserve_mental_models=request.preserve_mental_models,
                include_base_comparison=request.include_base_comparison,
                created_by_rationale=request.created_by_rationale,
                stakeholder_focus=request.stakeholder_focus,
                user_id=session.user_id,
            )

            return whatif_result

        @self.app.post("/api/what-if/batch", response_model=WhatIfBatchResponse)
        async def create_whatif_batch(
            request: WhatIfBatchRequest,
            session=Depends(
                self.security_manager.require_permission_dependency(
                    Permission.CREATE_ENGAGEMENT
                )
            ),
        ):
            """Create multiple What-If scenarios in batch for comprehensive analysis"""

            try:
                base_engagement_uuid = UUID(request.base_engagement_id)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid base engagement ID format",
                )

            # Validate batch size
            if len(request.scenarios) > 10:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Maximum 10 scenarios can be created in a single batch",
                )

            # Log batch What-If creation
            audit_manager = await get_audit_manager()
            await audit_manager.log_event(
                event_type=AuditEventType.ENGAGEMENT_CREATED,
                severity=AuditSeverity.HIGH,
                user_id=session.user_id,
                session_id=session.session_id,
                engagement_id=base_engagement_uuid,
                action_performed="create_whatif_batch",
                event_description=f"Created What-If batch: {request.scenario_batch_name} ({len(request.scenarios)} scenarios)",
                metadata={
                    "base_engagement_id": request.base_engagement_id,
                    "batch_name": request.scenario_batch_name,
                    "scenarios_count": len(request.scenarios),
                    "parallel_execution": request.parallel_execution,
                },
            )

            # Execute batch What-If creation
            whatif_engine = get_whatif_engine()
            batch_result = await whatif_engine.create_whatif_batch(
                base_engagement_id=base_engagement_uuid,
                batch_name=request.scenario_batch_name,
                scenarios=request.scenarios,
                parallel_execution=request.parallel_execution,
                compare_all=request.compare_all,
                user_id=session.user_id,
            )

            return batch_result

        # Export routes
        @self.app.post("/api/engagements/{engagement_id}/export")
        async def export_engagement_results(
            engagement_id: str,
            format: str = "json",
            session=Depends(
                self.security_manager.require_permission_dependency(
                    Permission.EXPORT_RESULTS
                )
            ),
        ):
            """Export engagement results in specified format"""

            try:
                engagement_uuid = UUID(engagement_id)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid engagement ID format",
                )

            # Log export request
            audit_manager = await get_audit_manager()
            await audit_manager.log_event(
                event_type=AuditEventType.DATA_EXPORT_REQUESTED,
                severity=AuditSeverity.MEDIUM,
                user_id=session.user_id,
                session_id=session.session_id,
                engagement_id=engagement_uuid,
                resource_type="engagement_export",
                resource_id=engagement_uuid,
                action_performed="export_results",
                event_description=f"Exported engagement results in {format} format",
                metadata={"export_format": format},
            )

            # Generate export data
            export_data = {
                "engagement_id": engagement_id,
                "export_timestamp": datetime.utcnow().isoformat(),
                "format": format,
                "exported_by": str(session.user_id),
                "data": {
                    "problem_statement": "Sample engagement export",
                    "analysis_results": {},
                    "reasoning_trace": [],
                },
            }

            return export_data

        # V2 Vulnerability Solution Endpoints
        @self.app.get("/api/v2/vulnerability/status")
        async def get_vulnerability_system_status(
            session=Depends(
                self.security_manager.require_permission_dependency(
                    Permission.READ_ENGAGEMENT
                )
            ),
        ):
            """Get overall vulnerability solution system status"""

            try:
                cognitive_engine = CognitiveEngineFactory.create_engine()
                status = cognitive_engine.get_status()

                vulnerability_status = {
                    "system_health": "operational",
                    "solutions_enabled": status.get("settings", {}).get(
                        "vulnerability_solutions_enabled", False
                    ),
                    "solution_components": status.get("vulnerability_solutions", {}),
                    "timestamp": datetime.utcnow().isoformat(),
                }

                return vulnerability_status

            except Exception as e:
                self.logger.error(f"Error getting vulnerability status: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to get vulnerability status: {str(e)}",
                )

        @self.app.get("/api/v2/engagements/{engagement_id}/vulnerability-context")
        async def get_engagement_vulnerability_context(
            engagement_id: str,
            session=Depends(
                self.security_manager.require_permission_dependency(
                    Permission.READ_ENGAGEMENT
                )
            ),
        ):
            """Get vulnerability solution context for specific engagement"""

            try:
                engagement_uuid = UUID(engagement_id)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid engagement ID format",
                )

            # This would typically fetch from database
            # For now, return structured response showing what vulnerability context includes
            vulnerability_context = {
                "engagement_id": engagement_id,
                "session_id": f"vuln_{engagement_id}",
                "exploration_decisions": {
                    "strategy_applied": "balanced",
                    "exploration_rate": 0.15,
                    "diversity_maintained": 0.87,
                    "strategic_mutations_count": 2,
                },
                "hallucination_detection": {
                    "checks_performed": 5,
                    "issues_detected": 0,
                    "confidence_levels": [0.89, 0.92, 0.85, 0.91, 0.88],
                    "validation_methods": [
                        "cross_model_consistency",
                        "logical_coherence",
                        "evidence_grounding",
                    ],
                },
                "failure_mode_responses": [],
                "pattern_governance": {
                    "tier_applied": "L2_peer_review",
                    "patterns_validated": True,
                    "governance_confidence": 0.91,
                },
                "feedback_tier": "silver",
                "overall_risk_level": "low",
                "solution_effectiveness": 0.94,
                "transparency_level": "high",
                "created_at": datetime.utcnow().isoformat(),
            }

            return vulnerability_context

        @self.app.post("/api/v2/vulnerability/exploration-override")
        async def override_exploration_strategy(
            request: Dict[str, Any],
            session=Depends(
                self.security_manager.require_permission_dependency(
                    Permission.MANAGE_SYSTEM
                )
            ),
        ):
            """Override exploration vs exploitation strategy for power users"""

            engagement_id = request.get("engagement_id")
            strategy_override = request.get(
                "strategy", "balanced"
            )  # explore, exploit, balanced
            rationale = request.get("rationale", "User override")

            if not engagement_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="engagement_id is required",
                )

            # Log the override for audit trail
            audit_manager = await get_audit_manager()
            await audit_manager.log_event(
                event_type=AuditEventType.SYSTEM_CONFIG_CHANGED,
                severity=AuditSeverity.HIGH,
                user_id=session.user_id,
                session_id=session.session_id,
                engagement_id=UUID(engagement_id),
                resource_type="exploration_strategy",
                resource_id=UUID(engagement_id),
                action_performed="strategy_override",
                event_description=f"User overrode exploration strategy to {strategy_override}",
                metadata={
                    "new_strategy": strategy_override,
                    "rationale": rationale,
                    "override_timestamp": datetime.utcnow().isoformat(),
                },
            )

            override_response = {
                "engagement_id": engagement_id,
                "strategy_applied": strategy_override,
                "override_rationale": rationale,
                "applied_by": str(session.user_id),
                "applied_at": datetime.utcnow().isoformat(),
                "expected_impact": {
                    "explore": "Higher discovery potential, increased processing time",
                    "exploit": "Faster results, lower discovery potential",
                    "balanced": "Optimized exploration-exploitation balance",
                }.get(strategy_override, "Unknown impact"),
            }

            return override_response

        @self.app.post("/api/v2/vulnerability/feedback-tier")
        async def assign_feedback_tier(
            request: Dict[str, Any],
            session=Depends(
                self.security_manager.require_permission_dependency(
                    Permission.MANAGE_USERS
                )
            ),
        ):
            """Assign or update feedback tier for enhanced partnership"""

            user_id = request.get("user_id", str(session.user_id))
            new_tier = request.get("tier", "bronze")  # bronze, silver, gold, platinum
            justification = request.get("justification", "Tier assignment")

            if new_tier not in ["bronze", "silver", "gold", "platinum"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid tier. Must be one of: bronze, silver, gold, platinum",
                )

            # Log tier assignment
            audit_manager = await get_audit_manager()
            await audit_manager.log_event(
                event_type=AuditEventType.USER_ROLE_CHANGED,
                severity=AuditSeverity.MEDIUM,
                user_id=session.user_id,
                session_id=session.session_id,
                resource_type="feedback_tier",
                resource_id=UUID(user_id),
                action_performed="tier_assignment",
                event_description=f"Assigned feedback tier {new_tier} to user {user_id}",
                metadata={
                    "new_tier": new_tier,
                    "previous_tier": "unknown",  # Would fetch from DB
                    "justification": justification,
                    "tier_benefits": {
                        "bronze": ["Basic analytics", "Standard support"],
                        "silver": [
                            "Enhanced analytics",
                            "Priority support",
                            "Advanced insights",
                        ],
                        "gold": [
                            "Strategic consulting",
                            "Custom analysis",
                            "Executive briefings",
                        ],
                        "platinum": [
                            "Full partnership",
                            "Co-development",
                            "Revenue sharing",
                        ],
                    }.get(new_tier, []),
                },
            )

            tier_response = {
                "user_id": user_id,
                "new_tier": new_tier,
                "assigned_by": str(session.user_id),
                "assigned_at": datetime.utcnow().isoformat(),
                "tier_benefits": {
                    "bronze": ["Basic feedback collection", "Standard reporting"],
                    "silver": [
                        "Enhanced analytics",
                        "Trend analysis",
                        "Priority support",
                    ],
                    "gold": [
                        "Strategic insights",
                        "Custom dashboards",
                        "Executive summaries",
                    ],
                    "platinum": [
                        "Full partnership access",
                        "Co-development opportunities",
                        "Revenue sharing",
                    ],
                }.get(new_tier, []),
                "value_exchange": {
                    "bronze": "Basic system improvement in exchange for usage data",
                    "silver": "Enhanced system capabilities in exchange for detailed feedback",
                    "gold": "Strategic consulting value in exchange for business insights",
                    "platinum": "Full partnership benefits in exchange for strategic collaboration",
                }.get(new_tier, "Unknown value exchange"),
            }

            return tier_response

        @self.app.get("/api/v2/vulnerability/hallucination-detection/{engagement_id}")
        async def get_hallucination_detection_report(
            engagement_id: str,
            session=Depends(
                self.security_manager.require_permission_dependency(
                    Permission.READ_ENGAGEMENT
                )
            ),
        ):
            """Get detailed hallucination detection report for engagement"""

            try:
                engagement_uuid = UUID(engagement_id)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid engagement ID format",
                )

            # This would fetch real detection results from database
            detection_report = {
                "engagement_id": engagement_id,
                "detection_summary": {
                    "total_checks": 5,
                    "checks_passed": 5,
                    "checks_failed": 0,
                    "overall_confidence": 0.89,
                    "risk_level": "low",
                },
                "layer_results": [
                    {
                        "layer": "L1_cross_model_consistency",
                        "status": "passed",
                        "confidence": 0.91,
                        "details": "All 3 models produced consistent reasoning patterns",
                        "evidence_sources": [
                            "claude_sonnet",
                            "reasoning_validator",
                            "synthesis_engine",
                        ],
                    },
                    {
                        "layer": "L2_logical_coherence",
                        "status": "passed",
                        "confidence": 0.88,
                        "details": "Logic flow validated across reasoning steps",
                        "validation_method": "formal_logic_checking",
                    },
                    {
                        "layer": "L3_evidence_grounding",
                        "status": "passed",
                        "confidence": 0.87,
                        "details": "All claims supported by research sources",
                        "evidence_count": 12,
                        "source_diversity": 8,
                    },
                ],
                "risk_assessment": {
                    "hallucination_probability": 0.11,
                    "confidence_calibration": "well_calibrated",
                    "recommendation": "Results approved for delivery",
                },
                "transparency_data": {
                    "validation_methods_used": [
                        "Multi-model consensus checking",
                        "Logical coherence validation",
                        "Evidence source triangulation",
                        "Confidence calibration assessment",
                        "Bias detection screening",
                    ],
                    "detection_timestamp": datetime.utcnow().isoformat(),
                },
            }

            return detection_report

    def _configure_exception_handlers(self):
        """Configure global exception handlers"""

        @self.app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            """Handle HTTP exceptions with logging"""

            self.logger.warning(
                f"HTTP {exc.status_code}: {exc.detail} - "
                f"{request.method} {request.url.path} - {request.client.host}"
            )

            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": exc.detail,
                    "status_code": exc.status_code,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

        @self.app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            """Handle general exceptions with logging"""

            self.logger.error(
                f"Unhandled exception: {str(exc)} - "
                f"{request.method} {request.url.path} - {request.client.host}",
                exc_info=True,
            )

            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Internal server error",
                    "status_code": 500,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

    def _configure_streaming(self):
        """Configure streaming WebSocket endpoints"""
        try:
            from src.engine.api.streaming_api import setup_streaming_api

            # Set up streaming API endpoints
            streaming_api = setup_streaming_api(self.app)
            self.logger.info("✅ Streaming API configured successfully")

        except ImportError as e:
            self.logger.warning(f"⚠️ Streaming API not available: {e}")
        except Exception as e:
            self.logger.error(f"❌ Failed to configure streaming API: {e}")

    def get_app(self) -> FastAPI:
        """Get FastAPI application instance"""
        return self.app


# Global API foundation instance
_api_foundation_instance: Optional[MetisAPIFoundation] = None


def get_api_foundation() -> MetisAPIFoundation:
    """Get or create global API foundation instance"""
    global _api_foundation_instance

    if _api_foundation_instance is None:
        _api_foundation_instance = MetisAPIFoundation()

    return _api_foundation_instance


# Utility decorators for API endpoints
def audit_endpoint(action: str, resource_type: str = "api_endpoint"):
    """Decorator to automatically audit API endpoint access"""

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract session from kwargs (if present)
            session = kwargs.get("session")

            if session:
                # Log API access
                audit_manager = await get_audit_manager()
                await audit_manager.log_event(
                    event_type=AuditEventType.DATA_ACCESSED,
                    user_id=session.user_id,
                    session_id=session.session_id,
                    resource_type=resource_type,
                    action_performed=action,
                    event_description=f"API endpoint accessed: {func.__name__}",
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def rate_limit(requests_per_hour: int = 100):
    """Decorator to add rate limiting to API endpoints"""

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Rate limiting logic would be implemented here
            return await func(*args, **kwargs)

        return wrapper

    return decorator


# MCP (Model Context Protocol) integration utilities
class MCPIntegration:
    """Model Context Protocol integration for tool compatibility"""

    @staticmethod
    async def register_mcp_tools():
        """Register METIS capabilities as MCP tools"""

        tools = [
            {
                "name": "metis_cognitive_analysis",
                "description": "Execute cognitive analysis using mental models",
                "parameters": {
                    "problem_statement": "string",
                    "business_context": "object",
                },
            },
            {
                "name": "metis_model_selection",
                "description": "Select optimal mental models for analysis",
                "parameters": {"problem_type": "string", "complexity_level": "string"},
            },
            {
                "name": "metis_audit_trail",
                "description": "Access engagement audit trail",
                "parameters": {
                    "engagement_id": "string",
                    "include_reasoning": "boolean",
                },
            },
        ]

        return tools

    @staticmethod
    async def execute_mcp_tool(
        tool_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute MCP tool request"""

        if tool_name == "metis_cognitive_analysis":
            # Execute cognitive analysis
            cognitive_engine = CognitiveEngineFactory.create_engine()

            # Create mock engagement for analysis
            engagement_event = create_engagement_initiated_event(
                problem_statement=parameters["problem_statement"],
                business_context=parameters.get("business_context", {}),
            )

            result = await cognitive_engine.process_engagement(engagement_event)

            return {
                "success": True,
                "result": {
                    "cognitive_state": result.cognitive_state.dict(),
                    "reasoning_steps": [
                        step.dict() for step in result.cognitive_state.reasoning_steps
                    ],
                },
            }

        return {"success": False, "error": f"Unknown tool: {tool_name}"}
