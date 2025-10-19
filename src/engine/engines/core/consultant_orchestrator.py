"""
METIS V5 Lean Consultant Orchestrator
====================================

Clean, focused orchestrator that coordinates all modular services.
Replaces the monolithic optimal_consultant_engine.py with clean service composition.

Part of the Great Refactoring: Lean orchestration using dependency injection.
"""

import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import uuid4

# Import our service layer
from ..contracts import (
    EngagementRequest,
    OptimalEngagementResult,
    HealthStatus,
)

from ..selection.query_classifier import (
    get_query_classification_service,
)
from ..selection.consultant_selector import (
    ConsultantSelectionService,
    get_consultant_selection_service,
)
from ..services.semantic_cluster_matcher import (
    get_semantic_cluster_matching_service,
)
from ..services.database_adapter import (
    get_database_adapter_service,
)
from ..services.state_manager import (
    get_state_management_service,
    EngagementPhase,
)
from ..monitoring.performance_monitor import (
    get_performance_monitoring_service,
)

# V4 Enhancement: UnifiedContextStream and ToolDecisionFramework integration
from src.core.unified_context_stream import UnifiedContextStream, ContextEventType
from src.core.incremental_context_manager import IncrementalContextManager

# Supabase for database operations
try:
    from supabase import create_client, Client

    SUPABASE_AVAILABLE = True
except ImportError:
    print(
        "⚠️ Supabase not available - orchestrator will operate with limited functionality"
    )
    SUPABASE_AVAILABLE = False
    Client = Any

# Legacy compatibility imports (for existing API endpoints)
from dataclasses import dataclass, field


@dataclass
class ConsultantBlueprint:
    consultant_id: str
    name: str
    specialization: str
    expertise: str
    persona_prompt: str
    stable_frameworks: List[str]
    adaptive_triggers: List[str] = field(default_factory=list)
    effectiveness_score: float = 0.8


@dataclass
class QueryClassification:
    keywords: List[str]
    complexity_score: int
    query_type: str
    matched_triggers: List[str]
    routing_pattern: Optional[str] = None


@dataclass
class ConsultantSelection:
    consultant_id: str
    blueprint: ConsultantBlueprint
    selection_reason: str
    frameworks_used: List[str]
    confidence_score: float


class ConsultantOrchestrator:
    """
    Lean orchestrator that coordinates all modular services.

    This replaces the monolithic OptimalConsultantEngine with clean service composition,
    dependency injection, and clear separation of concerns.

    The orchestrator is stateless - all state management is handled by StateManagementService.
    """

    def __init__(self, supabase_client: Optional[Client] = None):
        """Initialize the orchestrator with all required services"""

        # V4 Enhancement: Initialize UnifiedContextStream for complete audit trail
        self.context_stream = UnifiedContextStream(max_events=50000)

        # V4 Enhancement: Initialize IncrementalContextManager
        self.context_manager = IncrementalContextManager(self.context_stream)

        # Initialize database client
        if supabase_client:
            self.supabase = supabase_client
        elif SUPABASE_AVAILABLE:
            url = os.getenv("SUPABASE_URL", "https://soztmkgednwjhgzvlzch.supabase.co")
            key = os.getenv(
                "SUPABASE_SERVICE_ROLE_KEY",
                "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNvenRta2dlZG53amhnenZsemNoIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NDk4MzYxNywiZXhwIjoyMDcwNTU5NjE3fQ.fe-1KftmBOE_sl4uuMrc0P88LWbKqZvCTEa9vimLARQ",
            )
            try:
                self.supabase = create_client(url, key)
            except Exception as e:
                print(f"⚠️ Supabase initialization failed: {e}")
                self.supabase = None
        else:
            self.supabase = None

        # Initialize all modular services with shared dependencies
        self._initialize_services()

        # Legacy compatibility state
        self.consultant_blueprints: Dict[str, ConsultantBlueprint] = {}
        self.routing_patterns: Dict[str, Dict] = {}
        self.loaded = False

        print(
            "✅ ConsultantOrchestrator: Initialized with modular service architecture"
        )

    def _initialize_services(self):
        """Initialize all modular services with dependency injection"""

        # Core services
        self.query_classifier = get_query_classification_service()
        self.consultant_selector = get_consultant_selection_service()
        self.semantic_cluster_matcher = get_semantic_cluster_matching_service(
            supabase_client=self.supabase, context_stream=self.context_stream
        )
        self.database_adapter = get_database_adapter_service(
            supabase_client=self.supabase
        )
        self.state_manager = get_state_management_service(
            supabase_client=self.supabase, context_stream=self.context_stream
        )
        self.performance_monitor = get_performance_monitoring_service(
            context_stream=self.context_stream
        )

        print("✅ ConsultantOrchestrator: All modular services initialized")

    # === MAIN PROCESSING METHODS ===

    async def process_query(
        self,
        query: str,
        engagement_id: Optional[str] = None,
        context: Optional[Dict] = None,
    ) -> OptimalEngagementResult:
        """
        Main entry point for query processing using modular services.

        This method replaces the monolithic process_query method with clean orchestration.
        """
        # Generate engagement ID if not provided
        if not engagement_id:
            engagement_id = f"eng_{uuid4().hex[:12]}"

        start_time = datetime.now()
        timer_context = self.performance_monitor.start_operation_timing(
            engagement_id, "process_query"
        )

        # Log orchestration start
        self.context_stream.add_event(
            ContextEventType.REASONING_STEP,
            {
                "message": "Lean orchestration started",
                "engagement_id": engagement_id,
                "query": query,
                "context": context or {},
                "orchestrator_version": "v5_modular",
                "services_initialized": True,
            },
            metadata={"source": "ConsultantOrchestrator.process_query"},
        )

        try:
            # Phase 1: Detect schema version and handle stateful processing
            schema_version = await self.state_manager.detect_engagement_schema_version(
                engagement_id
            )

            if schema_version == 1:
                # V1 Legacy processing (if needed for backward compatibility)
                return await self._process_query_v1_legacy(
                    engagement_id, query, context, start_time
                )
            else:
                # V2.1 Stateful processing with modular services
                return await self._process_query_v2_stateful(
                    engagement_id, query, context, start_time
                )

        except Exception as e:
            # Mark engagement as failed
            await self.state_manager.mark_engagement_failed(
                engagement_id,
                {
                    "error_type": "orchestration_error",
                    "error_message": str(e),
                    "phase": "process_query",
                    "timestamp": datetime.now().isoformat(),
                },
            )

            # Record performance metrics
            self.performance_monitor.end_operation_timing(
                timer_context, success=False, metadata={"error": str(e)}
            )

            raise

    async def _process_query_v2_stateful(
        self,
        engagement_id: str,
        query: str,
        context: Optional[Dict],
        start_time: datetime,
    ) -> OptimalEngagementResult:
        """
        V2.1 Stateful processing with modular services and checkpoint recovery.
        """
        # Check for recovery state first
        recovery_state = await self.state_manager.load_recovery_state(engagement_id)

        if recovery_state and self.state_manager.can_recover_from_checkpoint(
            recovery_state
        ):
            print(
                f"🔄 Resuming engagement {engagement_id} from checkpoint: {recovery_state.get('last_checkpoint')}"
            )
            return await self._resume_from_checkpoint(
                engagement_id, recovery_state, query, context, start_time
            )

        # Initialize new V2.1 engagement state
        await self.state_manager.initialize_v21_engagement_state(engagement_id)

        # Load system configuration if needed
        if not self.loaded:
            await self._load_system_configuration()

        # Execute V2.1 processing pipeline with modular services
        return await self._execute_v21_processing_pipeline(
            engagement_id, query, context, start_time
        )

    async def _execute_v21_processing_pipeline(
        self,
        engagement_id: str,
        query: str,
        context: Optional[Dict],
        start_time: datetime,
    ) -> OptimalEngagementResult:
        """
        Execute the V2.1 processing pipeline using modular services.
        """
        processing_results = {}

        try:
            # Create engagement request
            engagement_request = EngagementRequest(
                query=query, context=context, engagement_id=engagement_id
            )

            # Phase 1: Query Classification
            await self.state_manager.start_phase(
                engagement_id, EngagementPhase.SOCRATIC_INTAKE
            )
            classification_result = await self.query_classifier.classify_query(
                engagement_request
            )
            await self.state_manager.create_checkpoint(
                engagement_id,
                EngagementPhase.SOCRATIC_INTAKE,
                {"classification_result": classification_result.dict()},
            )
            processing_results["classification"] = classification_result

            # Phase 2: Consultant Selection
            await self.state_manager.start_phase(
                engagement_id, EngagementPhase.CONSULTANT_SELECTION
            )
            consultant_selection_result = (
                await self.consultant_selector.select_consultants(
                    classification_result, engagement_request
                )
            )
            await self.state_manager.create_checkpoint(
                engagement_id,
                EngagementPhase.CONSULTANT_SELECTION,
                {"consultant_selection_result": consultant_selection_result.dict()},
            )
            processing_results["consultant_selection"] = consultant_selection_result

            # Phase 3: Semantic Cluster Matching
            selected_nway_clusters = (
                await self.semantic_cluster_matcher.select_relevant_nway_clusters(
                    enhanced_query=query, top_k=3
                )
            )
            processing_results["nway_clusters"] = selected_nway_clusters

            # Phase 4: Create final result
            processing_time = (datetime.now() - start_time).total_seconds()

            result = OptimalEngagementResult(
                engagement_id=engagement_id,
                query=query,
                selected_consultants=consultant_selection_result.selected_consultants,
                processing_summary={
                    "total_duration_ms": processing_time * 1000,
                    "consultants_evaluated": consultant_selection_result.total_candidates_evaluated,
                    "nway_clusters_used": len(selected_nway_clusters),
                    "classification_confidence": classification_result.confidence_score,
                    "selection_strategy": consultant_selection_result.selection_strategy,
                },
                performance_metrics={
                    "processing_time_seconds": processing_time,
                    "context_stream_events": len(self.context_stream.events),
                },
                confidence_score=classification_result.confidence_score,
            )

            # Mark engagement as completed
            await self.state_manager.mark_engagement_completed(
                engagement_id,
                {
                    "final_result": result.dict(),
                    "processing_time_seconds": processing_time,
                },
            )

            return result

        except Exception as e:
            await self.state_manager.mark_engagement_failed(
                engagement_id,
                {
                    "error_type": "pipeline_error",
                    "error_message": str(e),
                    "phase": "v21_processing_pipeline",
                },
            )
            raise

    async def _resume_from_checkpoint(
        self,
        engagement_id: str,
        recovery_state: Dict,
        query: str,
        context: Optional[Dict],
        start_time: datetime,
    ) -> OptimalEngagementResult:
        """
        Resume processing from a checkpoint.

        This is a placeholder implementation - full checkpoint recovery would require
        more sophisticated state reconstruction logic.
        """
        print("🔄 Checkpoint recovery not fully implemented - restarting pipeline")

        # For now, restart the pipeline (in production, this would intelligently resume)
        return await self._execute_v21_processing_pipeline(
            engagement_id, query, context, start_time
        )

    async def _process_query_v1_legacy(
        self,
        engagement_id: str,
        query: str,
        context: Optional[Dict],
        start_time: datetime,
    ) -> OptimalEngagementResult:
        """
        V1 Legacy processing for backward compatibility.
        """
        # V1 Legacy pipeline decommissioned - redirect to V2.1
        print("⚠️ V1 Legacy processing deprecated - using V2.1 stateful instead")
        return await self._process_query_v2_stateful(
            engagement_id, query, context, start_time
        )

    # === SYSTEM CONFIGURATION ===

    async def _load_system_configuration(self):
        """Load system configuration using database adapter"""
        try:
            # Load consultant blueprints
            blueprints_dict = await self.database_adapter.load_consultant_blueprints()
            self.consultant_blueprints = blueprints_dict

            # Configure consultant selector with blueprints
            self.consultant_selector = ConsultantSelectionService(
                self.consultant_blueprints
            )

            # Load routing patterns
            self.routing_patterns = await self.database_adapter.load_routing_patterns()

            # Configure routing patterns in consultant selector
            self.consultant_selector.configure_routing_patterns(self.routing_patterns)

            self.loaded = True
            print(
                f"✅ System configuration loaded: {len(blueprints_dict)} blueprints, {len(self.routing_patterns)} patterns"
            )

        except Exception as e:
            print(f"⚠️ Error loading system configuration: {e}")
            self.loaded = True  # Continue with defaults

    # === LEGACY COMPATIBILITY METHODS ===

    async def select_relevant_nway_clusters(
        self, enhanced_query: str, top_k: int = 3
    ) -> List[str]:
        """Legacy compatibility method for N-Way cluster selection"""
        return await self.semantic_cluster_matcher.select_relevant_nway_clusters(
            enhanced_query, top_k
        )

    def get_capabilities(self) -> Dict[str, Any]:
        """Get orchestrator capabilities and service status"""
        return {
            "orchestrator_version": "v5_modular",
            "services_available": {
                "query_classifier": bool(self.query_classifier),
                "consultant_selector": bool(self.consultant_selector),
                "semantic_cluster_matcher": bool(self.semantic_cluster_matcher),
                "database_adapter": bool(self.database_adapter),
                "state_manager": bool(self.state_manager),
                "performance_monitor": bool(self.performance_monitor),
            },
            "database_connected": bool(self.supabase),
            "context_stream_active": bool(self.context_stream),
            "system_loaded": self.loaded,
            "consultant_blueprints_count": len(self.consultant_blueprints),
            "routing_patterns_count": len(self.routing_patterns),
        }

    async def health_check(self) -> Dict[str, HealthStatus]:
        """Comprehensive health check of all services"""
        health_results = {}

        # Orchestrator self-check
        health_results["ConsultantOrchestrator"] = HealthStatus(
            component="ConsultantOrchestrator",
            healthy=True,
            response_time_ms=0.0,
            details="Modular orchestrator operational",
        )

        # Service health checks
        if self.database_adapter:
            health_results.update(
                {"DatabaseAdapter": await self.database_adapter.health_check()}
            )

        if self.performance_monitor:
            performance_health = (
                await self.performance_monitor.comprehensive_health_check()
            )
            health_results.update(performance_health)

        # Service capability checks
        health_results["QueryClassificationService"] = HealthStatus(
            component="QueryClassificationService",
            healthy=bool(self.query_classifier),
            response_time_ms=None,
            details=(
                "Query classification service available"
                if self.query_classifier
                else "Service not available"
            ),
        )

        health_results["ConsultantSelectionService"] = HealthStatus(
            component="ConsultantSelectionService",
            healthy=bool(self.consultant_selector),
            response_time_ms=None,
            details=(
                "Consultant selection service available"
                if self.consultant_selector
                else "Service not available"
            ),
        )

        health_results["SemanticClusterMatcher"] = HealthStatus(
            component="SemanticClusterMatcher",
            healthy=bool(self.semantic_cluster_matcher),
            response_time_ms=None,
            details=(
                "Semantic cluster matching available"
                if self.semantic_cluster_matcher
                else "Service not available"
            ),
        )

        health_results["StateManagementService"] = HealthStatus(
            component="StateManagementService",
            healthy=bool(self.state_manager),
            response_time_ms=None,
            details=(
                "State management service available"
                if self.state_manager
                else "Service not available"
            ),
        )

        return health_results

    # === UTILITY METHODS ===

    def get_context_stream(self) -> UnifiedContextStream:
        """Get the unified context stream for audit trails"""
        return self.context_stream

    def get_performance_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get performance summary from performance monitor"""
        if self.performance_monitor:
            return self.performance_monitor.get_performance_summary(hours_back)
        return {}


# Factory function for orchestrator creation
def get_consultant_orchestrator(
    supabase_client: Optional[Client] = None,
) -> ConsultantOrchestrator:
    """Factory function to create ConsultantOrchestrator instance"""
    return ConsultantOrchestrator(supabase_client)
