"""
Optimal Consultant Orchestrator - Slim Coordination Layer
Refactored from OptimalConsultantEngine monolith using service-oriented architecture

This orchestrator coordinates the extracted services:
- NWaySelectionService: Semantic cluster selection
- BlueprintRegistry: Consultant blueprint management
- Core orchestration logic with Glass-Box transparency

ARCHITECTURAL MANDATE COMPLIANCE:
✅ Glass-Box Transparency: All services inject into UnifiedContextStream
✅ Dependency Injection: Clean service boundaries with proper DI
✅ Single Responsibility: Each service has one clear purpose
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

# Glass-Box Integration - CRITICAL for transparency
from src.core.unified_context_stream import UnifiedContextStream, ContextEventType
from src.engine.core.tool_decision_framework import ToolDecisionFramework

# Extracted Services - New Architecture
from src.engine.engines.services.nway_selection_service import (
    NWaySelectionService,
)
from src.engine.engines.services.blueprint_registry import (
    BlueprintRegistry,
    ConsultantBlueprint,
)

# Contract interfaces for backward compatibility
try:
    from src.contracts.consultant_contracts import (
        ConsultantSelectionRequest,
        ConsultantSelectionResponse,
        ConsultantCandidate,
        ConsultantSelectionCriteria,
        ConsultantSelectionInterface,
    )
    from src.contracts.common_contracts import (
        EngagementContext,
        ProcessingMetrics,
        ProcessingStatus,
    )

    CONTRACTS_AVAILABLE = True
except ImportError:
    # Minimal contracts for standalone operation
    from dataclasses import dataclass
    from typing import Any

    @dataclass
    class ConsultantSelectionRequest:
        engagement_context: Any
        selection_criteria: Any
        enhanced_query: str = ""

    @dataclass
    class ConsultantSelectionResponse:
        success: bool
        selected_consultants: List[Any] = None

    CONTRACTS_AVAILABLE = False

# Database integration
try:
    from supabase import Client

    SUPABASE_AVAILABLE = True
except ImportError:
    Client = None
    SUPABASE_AVAILABLE = False


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator"""

    max_clusters: int = 5
    default_top_k: int = 3
    enable_caching: bool = True
    semantic_threshold: float = 0.7
    performance_tracking: bool = True


@dataclass
class OptimalEngagementResult:
    """Result of engagement processing - backward compatibility"""

    engagement_id: str
    selected_clusters: List[str]
    consultant_blueprints: List[ConsultantBlueprint]
    selection_metadata: Dict[str, Any]
    processing_time_ms: int
    confidence_score: float
    glass_box_events: int


class OptimalConsultantOrchestrator(
    ConsultantSelectionInterface if CONTRACTS_AVAILABLE else object
):
    """
    Optimal Consultant Orchestrator - Slim Coordination Layer

    Refactored from 26K-token monolith into clean service-oriented architecture.
    Coordinates extracted services while maintaining complete Glass-Box transparency.

    Service Dependencies:
    - NWaySelectionService: Handles cluster selection logic
    - BlueprintRegistry: Manages consultant blueprints
    - UnifiedContextStream: Glass-Box transparency
    - ToolDecisionFramework: Decision auditing
    """

    def __init__(
        self,
        context_stream: UnifiedContextStream,
        nway_service: NWaySelectionService,
        blueprint_registry: BlueprintRegistry,
        tool_framework: Optional[ToolDecisionFramework] = None,
        config: Optional[OrchestratorConfig] = None,
    ):
        """
        Initialize Orchestrator with Service Dependencies

        Args:
            context_stream: UnifiedContextStream for Glass-Box transparency
            nway_service: NWaySelectionService for cluster selection
            blueprint_registry: BlueprintRegistry for consultant management
            tool_framework: Optional ToolDecisionFramework for decisions
            config: Optional OrchestratorConfig
        """
        self.context_stream = context_stream
        self.nway_service = nway_service
        self.blueprint_registry = blueprint_registry
        self.tool_framework = tool_framework
        self.config = config or OrchestratorConfig()
        self.logger = logging.getLogger(__name__)

        # Performance tracking
        self.performance_metrics = {
            "total_engagements": 0,
            "successful_selections": 0,
            "failed_selections": 0,
            "average_processing_time_ms": 0,
            "cache_hit_rate": 0.0,
        }

        # Glass-Box: Log orchestrator initialization
        self.context_stream.add_event(
            event_type=ContextEventType.SYSTEM_STATE,
            data={
                "orchestrator_initialized": True,
                "services_connected": {
                    "nway_service": bool(self.nway_service),
                    "blueprint_registry": bool(self.blueprint_registry),
                    "tool_framework": bool(self.tool_framework),
                },
                "config": {
                    "max_clusters": self.config.max_clusters,
                    "default_top_k": self.config.default_top_k,
                    "semantic_threshold": self.config.semantic_threshold,
                },
            },
            metadata={
                "service": "OptimalConsultantOrchestrator",
                "method": "__init__",
                "architecture": "service_oriented",
            },
        )

    async def select_consultants(
        self, request: ConsultantSelectionRequest
    ) -> ConsultantSelectionResponse:
        """
        CONTRACT COMPLIANCE METHOD - Standardized consultant selection

        Orchestrates the extracted services to provide the same interface as the monolith
        while maintaining complete Glass-Box transparency.

        Service Flow:
        1. NWaySelectionService: Select relevant clusters
        2. BlueprintRegistry: Get consultant blueprints for clusters
        3. Build standardized response with metrics
        """
        start_time = datetime.now()
        engagement_id = request.engagement_context.engagement_id

        # Glass-Box: Log selection request start
        self.context_stream.add_event(
            event_type=ContextEventType.ENGAGEMENT_STARTED,
            data={
                "engagement_id": str(engagement_id),
                "enhanced_query": (
                    request.enhanced_query[:200] + "..."
                    if len(request.enhanced_query) > 200
                    else request.enhanced_query
                ),
                "top_k": request.selection_criteria.top_k,
                "orchestrator_mode": "service_oriented",
            },
            metadata={
                "service": "OptimalConsultantOrchestrator",
                "method": "select_consultants",
                "architecture": "refactored",
            },
        )

        try:
            # Step 1: Select N-Way clusters using extracted service
            selection_result = await self.nway_service.select_relevant_nway_clusters(
                enhanced_query=request.enhanced_query,
                top_k=request.selection_criteria.top_k,
                engagement_id=str(engagement_id),
            )

            # Step 2: Get consultant blueprints for selected clusters
            consultant_candidates = []
            all_candidates = []

            for i, cluster_id in enumerate(selection_result.selected_clusters):
                # Try to get blueprint for this cluster
                blueprint = self.blueprint_registry.get_blueprint(cluster_id)

                if blueprint:
                    # Create consultant candidate from blueprint
                    candidate = (
                        ConsultantCandidate(
                            consultant_id=blueprint.consultant_id,
                            consultant_name=blueprint.name,
                            consultant_role=blueprint.specialization,
                            similarity_score=selection_result.confidence_score
                            * (1.0 - i * 0.1),
                            selection_reasoning=f"Selected via {selection_result.selection_method}",
                            nway_cluster_id=cluster_id,
                        )
                        if CONTRACTS_AVAILABLE
                        else {
                            "consultant_id": blueprint.consultant_id,
                            "consultant_name": blueprint.name,
                            "consultant_role": blueprint.specialization,
                            "similarity_score": selection_result.confidence_score
                            * (1.0 - i * 0.1),
                            "selection_reasoning": f"Selected via {selection_result.selection_method}",
                            "nway_cluster_id": cluster_id,
                        }
                    )

                    consultant_candidates.append(candidate)
                    all_candidates.append(candidate)
                else:
                    # Log missing blueprint
                    self.context_stream.add_event(
                        event_type=ContextEventType.ERROR_OCCURRED,
                        data={"cluster_id": cluster_id, "error": "Blueprint not found"},
                        metadata={
                            "service": "OptimalConsultantOrchestrator",
                            "issue": "missing_blueprint",
                        },
                    )

            # Step 3: Calculate processing metrics
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            # Update performance metrics
            self.performance_metrics["total_engagements"] += 1
            if consultant_candidates:
                self.performance_metrics["successful_selections"] += 1
            else:
                self.performance_metrics["failed_selections"] += 1

            # Glass-Box: Log successful orchestration
            self.context_stream.add_event(
                event_type=ContextEventType.PROCESSING_COMPLETE,
                data={
                    "engagement_id": str(engagement_id),
                    "selected_consultants": len(consultant_candidates),
                    "selection_method": selection_result.selection_method,
                    "confidence_score": selection_result.confidence_score,
                    "processing_time_seconds": processing_time,
                    "service_coordination": "successful",
                },
                metadata={
                    "service": "OptimalConsultantOrchestrator",
                    "method": "select_consultants",
                    "result": "success",
                },
            )

            # Step 4: Build contract-compliant response
            if CONTRACTS_AVAILABLE:
                metrics = ProcessingMetrics(
                    component_name="OptimalConsultantOrchestrator",
                    processing_time_seconds=processing_time,
                    start_time=start_time,
                    end_time=end_time,
                    status=ProcessingStatus.COMPLETED,
                )

                response = ConsultantSelectionResponse(
                    success=True,
                    engagement_id=engagement_id,
                    selected_consultants=consultant_candidates,
                    selection_method_used=selection_result.selection_method,
                    all_candidates=all_candidates,
                    processing_metrics=metrics,
                    error_message=None,
                )
            else:
                response = ConsultantSelectionResponse(
                    success=True, selected_consultants=consultant_candidates
                )

            return response

        except Exception as e:
            self.performance_metrics["failed_selections"] += 1

            # Glass-Box: Log orchestration error
            self.context_stream.add_event(
                event_type=ContextEventType.ERROR_OCCURRED,
                data={
                    "engagement_id": str(engagement_id),
                    "error": str(e),
                    "service_coordination": "failed",
                },
                metadata={
                    "service": "OptimalConsultantOrchestrator",
                    "method": "select_consultants",
                    "result": "error",
                },
            )

            self.logger.error(
                f"Consultant selection failed for engagement {engagement_id}: {e}"
            )

            # Return error response
            if CONTRACTS_AVAILABLE:
                return ConsultantSelectionResponse(
                    success=False,
                    engagement_id=engagement_id,
                    selected_consultants=[],
                    selection_method_used="error",
                    all_candidates=[],
                    processing_metrics=None,
                    error_message=str(e),
                )
            else:
                return ConsultantSelectionResponse(
                    success=False, selected_consultants=[]
                )

    async def select_relevant_nway_clusters(
        self, enhanced_query: str, top_k: int = 3
    ) -> List[str]:
        """
        BACKWARD COMPATIBILITY METHOD

        Maintains the original interface while delegating to the extracted service.
        This ensures existing code continues to work during the transition.
        """

        # Glass-Box: Log backward compatibility call
        self.context_stream.add_event(
            event_type=ContextEventType.TOOL_EXECUTION,
            data={
                "method": "select_relevant_nway_clusters",
                "compatibility": "backward",
                "enhanced_query": (
                    enhanced_query[:100] + "..."
                    if len(enhanced_query) > 100
                    else enhanced_query
                ),
            },
            metadata={
                "service": "OptimalConsultantOrchestrator",
                "interface": "legacy",
            },
        )

        # Delegate to extracted service
        result = await self.nway_service.select_relevant_nway_clusters(
            enhanced_query, top_k
        )

        return result.selected_clusters

    def get_blueprint_for_role(self, role_id: str) -> Optional[ConsultantBlueprint]:
        """
        BACKWARD COMPATIBILITY METHOD

        Delegates to BlueprintRegistry while maintaining original interface.
        """

        # Glass-Box: Log backward compatibility call
        self.context_stream.add_event(
            event_type=ContextEventType.TOOL_EXECUTION,
            data={
                "method": "get_blueprint_for_role",
                "compatibility": "backward",
                "role_id": role_id,
            },
            metadata={
                "service": "OptimalConsultantOrchestrator",
                "interface": "legacy",
            },
        )

        return self.blueprint_registry.get_blueprint_for_role(role_id)

    def get_all_blueprints(self) -> Dict[str, ConsultantBlueprint]:
        """Get all available consultant blueprints"""
        return self.blueprint_registry.get_all_blueprints()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get orchestrator performance metrics"""

        # Combine orchestrator metrics with service metrics
        nway_metrics = self.nway_service.get_performance_metrics()
        registry_status = self.blueprint_registry.get_registry_status()

        return {
            "orchestrator": self.performance_metrics,
            "nway_service": nway_metrics,
            "blueprint_registry": registry_status,
            "services_healthy": all(
                [
                    bool(self.nway_service),
                    bool(self.blueprint_registry),
                    bool(self.context_stream),
                ]
            ),
        }

    def get_context_stream_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent context stream events for transparency"""

        # Get recent events from the context stream
        recent_events = []
        for event in self.context_stream.events[-limit:]:
            recent_events.append(event.to_dict())

        return recent_events

    def get_orchestration_status(self) -> Dict[str, Any]:
        """Get complete orchestration health status"""

        return {
            "orchestrator_healthy": True,
            "services_status": {
                "nway_selection": {
                    "available": bool(self.nway_service),
                    "methods": (
                        self.nway_service._get_available_methods()
                        if self.nway_service
                        else []
                    ),
                },
                "blueprint_registry": {
                    "available": bool(self.blueprint_registry),
                    "total_blueprints": (
                        len(self.blueprint_registry.consultant_blueprints)
                        if self.blueprint_registry
                        else 0
                    ),
                },
                "context_stream": {
                    "available": bool(self.context_stream),
                    "total_events": (
                        len(self.context_stream.events) if self.context_stream else 0
                    ),
                },
            },
            "performance": self.get_performance_metrics(),
            "architecture": "service_oriented_refactored",
            "glass_box_enabled": True,
        }


# Factory function for complete orchestrator setup
def create_optimal_consultant_orchestrator(
    supabase_client: Optional[Client] = None,
    config: Optional[OrchestratorConfig] = None,
) -> OptimalConsultantOrchestrator:
    """
    Factory function to create complete OptimalConsultantOrchestrator

    This function creates all necessary dependencies and wires them together
    with proper Glass-Box transparency integration.

    Args:
        supabase_client: Optional Supabase client for database access
        config: Optional OrchestratorConfig for customization

    Returns:
        Fully configured OptimalConsultantOrchestrator
    """

    # Create core Glass-Box infrastructure
    from src.core.unified_context_stream import get_unified_context_stream
    context_stream = get_unified_context_stream()

    # Create extracted services with dependency injection
    nway_service = NWaySelectionService(
        context_stream=context_stream,
        tool_framework=None,  # TODO: Fix tool framework integration
        supabase_client=supabase_client,
    )

    blueprint_registry = BlueprintRegistry(
        context_stream=context_stream, supabase_client=supabase_client
    )

    # Create orchestrator with all dependencies
    orchestrator = OptimalConsultantOrchestrator(
        context_stream=context_stream,
        nway_service=nway_service,
        blueprint_registry=blueprint_registry,
        tool_framework=None,  # TODO: Fix tool framework integration
        config=config,
    )

    return orchestrator
