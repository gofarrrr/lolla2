"""
Main Engagement Orchestrator for METIS API
"""

from datetime import datetime
from typing import Dict, Any, Optional
from uuid import UUID, uuid4

from fastapi import HTTPException

# Structured logging (Clarity & Consolidation Sprint)
from src.core.structured_logging import (
    get_logger,
    LoggingContext,
    set_engagement_context,
)

from .models import EngagementRequest, EngagementResponse, EngagementPhase, PhaseResult
from .mappers import (
    map_contract_to_engagement_response,
    map_contract_phase_to_phase_result,
)
from .websocket import ConnectionManager

try:
    from src.core.consolidated_neural_lace_orchestrator import (
        get_consolidated_neural_lace_orchestrator,
    )
    from src.engine.models.data_contracts import (
        MetisDataContract,
        create_engagement_initiated_event,
    )
    from src.core.enhanced_event_bus import MetisEventBus
    from src.core.comprehensive_data_capture import ComprehensiveDataCapture
    from src.engine.core.query_clarification_engine import QueryClarificationEngine
    from src.core.hitl_interaction_manager import HITLInteractionManager
    from src.core.state_management import DistributedStateManager, StateType
    from src.optimization.cost_tracker import (
        CostOptimizationEngine,
        UsageMetricType,
        CostCategory,
    )
    from src.monitoring.performance_validator import PerformanceValidator

    # Operation Crystal Day 1: Import contradiction detector
    from src.core.contradiction_detector import ContradictionDetector

    ENGINES_AVAILABLE = True
except ImportError:
    ENGINES_AVAILABLE = False
    MetisDataContract = None
    create_engagement_initiated_event = None


class EngagementOrchestrator:
    """
    Unified orchestrator using SurgicalWorkflowOrchestrator and contract-based state management
    API layer now derives all responses from MetisDataContract (single source of truth)
    """

    def __init__(self):
        # Optional legacy components for backward compatibility
        self.state_manager = DistributedStateManager() if ENGINES_AVAILABLE else None
        self.cost_engine = (
            CostOptimizationEngine(self.state_manager) if ENGINES_AVAILABLE else None
        )
        self.performance_validator = (
            PerformanceValidator(self.state_manager) if ENGINES_AVAILABLE else None
        )

        # P4.5: Use integrated system with event routing
        self.metis_system = None
        self.event_bus = None
        self.workflow_orchestrator = None

        # Operation Synapse: Context Intelligence Engine
        self.context_intelligence = None

        # Comprehensive data capture system
        self.data_capture = ComprehensiveDataCapture() if ENGINES_AVAILABLE else None

        # HITL Clarification system
        self.clarification_engine = (
            QueryClarificationEngine() if ENGINES_AVAILABLE else None
        )
        self.hitl_manager = (
            HITLInteractionManager(self.clarification_engine)
            if ENGINES_AVAILABLE
            else None
        )

        # Operation Crystal Day 1: Contradiction detection pre-flight check
        self.contradiction_detector = (
            ContradictionDetector() if ENGINES_AVAILABLE else None
        )

        # Contract storage (single source of truth)
        self.contracts: Dict[UUID, MetisDataContract] = {}
        self.client_names: Dict[UUID, str] = (
            {}
        )  # Map engagement_id to client_name for API compatibility

        # Captured data storage for transparency
        self.captured_data: Dict[UUID, Dict[str, Any]] = {}

        # WebSocket manager
        self.connection_manager = ConnectionManager()

        # Use structured logger with component context
        self.logger = get_logger(__name__, component="orchestrator")

    async def initialize_engines(self):
        """Initialize unified workflow orchestrator with P4.5 system integration"""
        if not ENGINES_AVAILABLE:
            self.logger.warning(
                "Backend engines not available - running in simulation mode"
            )
            return

        try:
            # P4.5: Initialize complete METIS system with event routing
            from src.core.metis_system_integration import (
                create_metis_system,
                SystemMode,
            )

            # Determine system mode based on environment or configuration
            # This could be read from environment variables in production
            system_mode = SystemMode.DEVELOPMENT  # Default for now

            self.metis_system = await create_metis_system(
                mode=system_mode,
                enable_kafka=False,  # Use in-memory for now
                enable_monitoring=True,
                enable_event_routing=True,
            )

            # Extract components for compatibility
            self.event_bus = self.metis_system.event_bus
            self.workflow_orchestrator = self.metis_system.workflow_orchestrator

            self.logger.info(
                "system_initialized",
                status="success",
                mode=str(system_mode),
                features={"kafka": False, "monitoring": True, "event_routing": True},
            )

            # Operation Synapse: Initialize Context Intelligence Engine
            try:
                from src.core.context_intelligence_engine import (
                    create_context_intelligence_engine,
                )

                self.context_intelligence = create_context_intelligence_engine()
                self.logger.info(
                    "context_intelligence_initialized",
                    status="success",
                    component="context_intelligence_engine",
                )
            except Exception as ctx_error:
                self.logger.error(
                    f"❌ Failed to initialize Context Intelligence Engine: {ctx_error}"
                )
                self.context_intelligence = None
        except Exception as e:
            self.logger.error(f"Failed to initialize METIS system: {str(e)}")
            # Fallback to basic initialization
            try:
                from src.core.enhanced_event_bus import MetisEventBus

                self.event_bus = MetisEventBus()
                await self.event_bus.initialize()
                # NEURAL LACE ORCHESTRATION - Use stateful orchestrator with full data capture
                self.workflow_orchestrator = (
                    await get_consolidated_neural_lace_orchestrator()
                )
                self.logger.warning("Using fallback basic event bus initialization")

                # Operation Synapse: Initialize Context Intelligence Engine (fallback path)
                try:
                    from src.core.context_intelligence_engine import (
                        create_context_intelligence_engine,
                    )

                    self.context_intelligence = create_context_intelligence_engine()
                    self.logger.info(
                        "🧠 Context Intelligence Engine initialized (fallback)"
                    )
                except Exception as ctx_error:
                    self.logger.error(
                        f"❌ Failed to initialize Context Intelligence Engine (fallback): {ctx_error}"
                    )
                    self.context_intelligence = None
            except Exception as fallback_error:
                self.logger.error(
                    f"Fallback initialization also failed: {str(fallback_error)}"
                )

    async def create_engagement(self, request: EngagementRequest) -> EngagementResponse:
        """Create new consulting engagement using unified contract approach"""

        # Operation Crystal Day 1: Pre-flight contradiction detection
        if self.contradiction_detector and ENGINES_AVAILABLE:
            # Set engagement context for all subsequent logs
            engagement_id = uuid4()
            set_engagement_context(engagement_id)

            self.logger.info(
                "contradiction_detection_started",
                phase="pre_flight",
                engagement_id=str(engagement_id),
            )

            # Build business context for contradiction analysis
            business_context = {
                **request.problem_statement.business_context,
                "stakeholders": request.problem_statement.stakeholders,
                "success_criteria": request.problem_statement.success_criteria,
                "engagement_type": request.engagement_type,
                "priority": request.priority,
            }

            # Detect contradictions
            contradiction_result = (
                await self.contradiction_detector.detect_contradictions(
                    query=request.problem_statement.problem_description,
                    business_context=business_context,
                    clarifications=None,  # Could be passed from HITL flow in future
                )
            )

            # Block engagement creation if contradictions found
            if contradiction_result.blocked:
                self.logger.error(
                    "engagement_blocked",
                    reason="contradictions_detected",
                    contradiction_count=len(contradiction_result.contradictions),
                    contradictions=contradiction_result.contradictions,
                )
                contradiction_summary = (
                    self.contradiction_detector.get_contradiction_summary(
                        contradiction_result
                    )
                )

                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Logical contradictions detected",
                        "contradictions": contradiction_summary["contradictions"],
                        "recommendation": contradiction_summary["recommendation"],
                        "analysis_time_ms": contradiction_summary["analysis_time_ms"],
                    },
                )
            elif contradiction_result.has_contradictions:
                # Log warnings for non-blocking contradictions
                self.logger.warning(
                    "potential_contradictions",
                    contradiction_count=len(contradiction_result.contradictions),
                    blocking=False,
                    contradictions=contradiction_result.contradictions,
                )

        # Track cost for engagement creation
        if "engagement_id" not in locals():
            engagement_id = uuid4()
            set_engagement_context(engagement_id)
        if self.cost_engine:
            await self.cost_engine.track_usage(
                metric_type=UsageMetricType.ENGAGEMENTS_PROCESSED,
                metric_value=1,
                cost_category=CostCategory.COMPUTE_RESOURCES,
                engagement_id=engagement_id,
            )

        # Create unified contract (single source of truth)
        if not ENGINES_AVAILABLE or not create_engagement_initiated_event:
            # Simulation mode
            return await self._simulate_engagement_creation(request, engagement_id)

        contract = create_engagement_initiated_event(
            problem_statement=request.problem_statement.problem_description,
            business_context={
                **request.problem_statement.business_context,
                "stakeholders": request.problem_statement.stakeholders,
                "success_criteria": request.problem_statement.success_criteria,
                "engagement_type": request.engagement_type,
                "priority": request.priority,
            },
        )

        # Override engagement_id if needed
        contract.engagement_context.engagement_id = engagement_id

        # Store contract and client name
        self.contracts[engagement_id] = contract
        self.client_names[engagement_id] = request.client_name

        # Store in state manager for backward compatibility
        if self.state_manager:
            await self.state_manager.set_state(
                f"engagement_{engagement_id}",
                contract.to_cloudevents_dict(),
                StateType.ENGAGEMENT,
            )

        self.logger.info(
            "engagement_created_starting_autonomous_execution",
            engagement_id=str(engagement_id),
            client_name=request.client_name,
            engagement_type=request.engagement_type,
            priority=request.priority,
        )

        # AUTONOMOUS ORCHESTRATOR INTEGRATION:
        # Execute complete 6-phase workflow automatically using state machine
        try:
            from src.engine.engines.core.consultant_orchestrator import (
                get_consultant_orchestrator,
            )

            # Create V5 autonomous orchestrator (replaced legacy StateMachineOrchestrator)
            autonomous_orchestrator = get_consultant_orchestrator()

            self.logger.info(
                "autonomous_orchestrator_executing",
                engagement_id=str(engagement_id),
                phases_to_execute=6,
            )

            # Send WebSocket update about autonomous execution start
            await self.connection_manager.send_update(
                str(engagement_id),
                {
                    "type": "autonomous_execution_started",
                    "message": "Full 6-phase cognitive workflow initiated",
                    "phases": [
                        "problem_structuring",
                        "hypothesis_generation",
                        "analysis_execution",
                        "research_grounding",
                        "validation_debate",
                        "synthesis_delivery",
                    ],
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

            # Execute full autonomous workflow
            final_contract = await autonomous_orchestrator.run_full_engagement(contract)

            # Update stored contract with results
            self.contracts[engagement_id] = final_contract

            # Update state manager
            if self.state_manager:
                await self.state_manager.set_state(
                    f"engagement_{engagement_id}",
                    final_contract.to_cloudevents_dict(),
                    StateType.ENGAGEMENT,
                )

            # Send final WebSocket update
            final_status = (
                "completed"
                if final_contract.workflow_state.status.value == "COMPLETED"
                else "failed"
            )
            await self.connection_manager.send_update(
                str(engagement_id),
                {
                    "type": "autonomous_execution_completed",
                    "status": final_status,
                    "phases_completed": len(
                        final_contract.workflow_state.completed_phases
                    ),
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

            self.logger.info(
                "autonomous_orchestrator_completed",
                engagement_id=str(engagement_id),
                final_status=final_status,
                phases_completed=len(final_contract.workflow_state.completed_phases),
            )

        except Exception as e:
            self.logger.error(
                "autonomous_orchestrator_failed",
                engagement_id=str(engagement_id),
                error=str(e),
            )

            # Send error WebSocket update
            await self.connection_manager.send_update(
                str(engagement_id),
                {
                    "type": "autonomous_execution_failed",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

            # Mark contract as failed but don't raise - return the failed engagement
            if engagement_id in self.contracts:
                failed_contract = self.contracts[engagement_id]
                failed_contract.workflow_state.status = "FAILED"
                failed_contract.workflow_state.error_details = {
                    "error_message": str(e),
                    "failed_at": datetime.utcnow().isoformat(),
                }

        # Convert final contract to API response format
        final_contract = self.contracts[engagement_id]
        engagement_response = map_contract_to_engagement_response(
            final_contract, request.client_name
        )

        return engagement_response

    async def execute_phase(
        self, engagement_id: UUID, phase: EngagementPhase
    ) -> PhaseResult:
        """Execute engagement phase using unified workflow orchestrator with comprehensive data capture"""
        if engagement_id not in self.contracts:
            raise HTTPException(status_code=404, detail="Engagement not found")

        # Use logging context for this phase execution
        with LoggingContext(
            engagement_id=engagement_id,
            component="orchestrator",
            span_name="execute_phase",
            span_attributes={"phase": str(phase)},
        ) as ctx:
            contract = self.contracts[engagement_id]
            start_time = datetime.utcnow()

        # Send WebSocket update - handle both string and enum
        phase_name_for_ws = phase.value if hasattr(phase, "value") else str(phase)
        await self.connection_manager.send_update(
            str(engagement_id),
            {
                "type": "phase_started",
                "phase": phase_name_for_ws,
                "timestamp": start_time.isoformat(),
            },
        )

        try:
            # Check if workflow has already been executed
            if not contract.workflow_state.phase_results:
                # Execute complete workflow with comprehensive data capture
                if (
                    self.workflow_orchestrator
                    and ENGINES_AVAILABLE
                    and self.data_capture
                ):
                    self.logger.info(
                        f"Executing complete workflow with data capture for engagement {engagement_id}"
                    )

                    # Operation Synapse: Context Intelligence Phase
                    user_query = contract.engagement_context.problem_statement
                    await self._execute_context_intelligence_phase(
                        engagement_id, user_query
                    )

                    # Use comprehensive data capture system
                    async with self.data_capture.capture_system_flow(
                        user_query, str(engagement_id)
                    ) as capture_context:
                        updated_contract = (
                            await self.workflow_orchestrator.run_full_engagement(
                                contract
                            )
                        )

                        # Store captured data for transparency
                        captured_data = capture_context.get_captured_data()
                        self.captured_data[engagement_id] = captured_data

                        # Send transparency data via WebSocket
                        await self.connection_manager.send_update(
                            str(engagement_id),
                            {
                                "type": "transparency_data",
                                "captured_data_summary": {
                                    "llm_calls_count": len(
                                        captured_data.get("llm_interactions", [])
                                    ),
                                    "database_operations": len(
                                        captured_data.get("database_operations", [])
                                    ),
                                    "research_queries": len(
                                        captured_data.get("research_data", [])
                                    ),
                                    "total_cost": captured_data.get(
                                        "cost_tracking", {}
                                    ).get("total_cost", 0.0),
                                },
                            },
                        )

                    self.contracts[engagement_id] = updated_contract
                elif self.workflow_orchestrator and ENGINES_AVAILABLE:
                    # Fallback without data capture
                    self.logger.info(
                        f"Executing complete workflow (no data capture) for engagement {engagement_id}"
                    )
                    updated_contract = (
                        await self.workflow_orchestrator.run_full_engagement(contract)
                    )
                    self.contracts[engagement_id] = updated_contract
                else:
                    # Simulation mode - populate contract with simulated results
                    await self._simulate_complete_workflow(contract)

            # Extract the requested phase result from the completed workflow
            # Handle both string and enum phase parameters safely
            if hasattr(phase, "value"):
                phase_name = phase.value
            else:
                phase_name = str(phase)
            if phase_name not in contract.workflow_state.phase_results:
                raise HTTPException(
                    status_code=400,
                    detail=f"Phase {phase_name} not available. Available phases: {list(contract.workflow_state.phase_results.keys())}",
                )

            # Map contract phase result to API format
            result = map_contract_phase_to_phase_result(contract, phase_name)
            execution_time = (datetime.utcnow() - start_time).total_seconds()

            # Track performance
            if self.performance_validator and hasattr(
                self.performance_validator, "track_metric"
            ):
                try:
                    await self.performance_validator.track_metric(
                        "phase_execution_time",
                        execution_time,
                        {
                            "phase": phase_name_for_ws,
                            "engagement_id": str(engagement_id),
                        },
                    )
                except Exception as perf_error:
                    self.logger.warning(f"Performance tracking failed: {perf_error}")

            # Send completion update
            await self.connection_manager.send_update(
                str(engagement_id),
                {
                    "type": "phase_completed",
                    "phase": phase_name_for_ws,
                    "result": result.dict(),
                    "overall_progress": len(contract.workflow_state.completed_phases)
                    * 25,  # Simple progress calc
                },
            )

            return result

        except Exception as e:
            self.logger.error(f"Phase execution failed for {engagement_id}: {str(e)}")
            # Send error update
            await self.connection_manager.send_update(
                str(engagement_id),
                {"type": "phase_error", "phase": phase_name_for_ws, "error": str(e)},
            )
            raise HTTPException(
                status_code=500, detail=f"Phase execution failed: {str(e)}"
            )

    async def get_engagement(self, engagement_id: UUID) -> Optional[EngagementResponse]:
        """Get engagement details from contract storage"""
        if engagement_id not in self.contracts:
            return None

        contract = self.contracts[engagement_id]
        client_name = self.client_names.get(engagement_id, "Unknown Client")

        return map_contract_to_engagement_response(contract, client_name)

    async def get_captured_data(self, engagement_id: UUID) -> Dict[str, Any]:
        """Get captured transparency data for an engagement"""
        return self.captured_data.get(engagement_id, {})

    async def _simulate_engagement_creation(
        self, request: EngagementRequest, engagement_id: UUID
    ) -> EngagementResponse:
        """Simulate engagement creation when engines are not available"""
        from .models import EngagementStatus

        # Create simulated response
        return EngagementResponse(
            engagement_id=engagement_id,
            client_name=request.client_name,
            problem_statement=request.problem_statement,
            status=EngagementStatus.CREATED,
            current_phase=EngagementPhase.PROBLEM_STRUCTURING,
            progress_percentage=0,
            phases={},
            overall_confidence=0.0,
            estimated_cost=0.5,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            deliverable_ready=False,
        )

    async def _execute_context_intelligence_phase(
        self, engagement_id: UUID, user_query: str
    ):
        """
        Operation Synapse: Execute Context Intelligence Phase

        This revolutionary phase demonstrates our Context Intelligence capabilities by:
        1. Curating relevant context using cognitive exhaust
        2. Streaming granular progress updates for perceived performance
        3. Building the foundation for our category-defining context intelligence
        """
        try:
            # Stream Event 1: Context curation started
            await self.connection_manager.send_update(
                str(engagement_id),
                {
                    "type": "context_curation_started",
                    "message": "🧠 Analyzing context intelligence patterns...",
                    "timestamp": datetime.utcnow().isoformat(),
                    "phase": "context_intelligence",
                },
            )

            if not self.context_intelligence:
                self.logger.warning("⚠️ Context Intelligence Engine not available")
                return

            # Stream Event 2: L1 cache check
            await self.connection_manager.send_update(
                str(engagement_id),
                {
                    "type": "l1_cache_checked",
                    "message": "⚡ Checking in-memory cache for relevant patterns...",
                    "timestamp": datetime.utcnow().isoformat(),
                    "phase": "context_intelligence",
                },
            )

            # Sprint 1.2: Get L1+L2 cache statistics
            engine_stats = await self.context_intelligence.get_engine_stats()

            # Stream Event 3: Cognitive exhaust analyzed
            await self.connection_manager.send_update(
                str(engagement_id),
                {
                    "type": "cognitive_exhaust_analyzed",
                    "message": f"🎯 Analyzed patterns across L1+L2+L3 cache layers ({engine_stats.get('sprint_version', 'unknown')})",
                    "timestamp": datetime.utcnow().isoformat(),
                    "phase": "context_intelligence",
                    "l1_cache_performance": engine_stats.get(
                        "l1_cache_performance", {}
                    ),
                    "l2_cache_performance": engine_stats.get(
                        "l2_cache_performance", {}
                    ),
                    "l3_cache_performance": engine_stats.get(
                        "l3_cache_performance", {}
                    ),
                    "multi_layer_status": engine_stats.get(
                        "multi_layer_caching", "l1_only"
                    ),
                },
            )

            # Sprint 1.2: Get relevant contexts using our revolutionary Context Intelligence with engagement_id
            relevant_contexts = await self.context_intelligence.get_relevant_context(
                current_query=user_query,
                max_contexts=5,
                engagement_id=str(
                    engagement_id
                ),  # Pass engagement_id for L2 cache access
            )

            # Stream Event 4: Context intelligence complete
            context_intelligence_summary = []
            for context, score in relevant_contexts:
                context_intelligence_summary.append(
                    {
                        "mental_model": context.mental_model,
                        "phase": context.phase,
                        "relevance_score": f"{score.overall_score:.3f}",
                        "confidence": f"{context.confidence:.2f}",
                        "reasoning": score.explanation,
                    }
                )

            await self.connection_manager.send_update(
                str(engagement_id),
                {
                    "type": "context_intelligence_complete",
                    "message": f"✅ Context intelligence analysis complete - {len(relevant_contexts)} relevant patterns identified",
                    "timestamp": datetime.utcnow().isoformat(),
                    "phase": "context_intelligence",
                    "relevant_contexts": context_intelligence_summary,
                    "engine_stats": await self.context_intelligence.get_engine_stats(),  # Sprint 1.2: async engine stats
                },
            )

            self.logger.info(
                f"🚀 Context Intelligence Phase completed for engagement {engagement_id}"
            )

        except Exception as e:
            self.logger.error(
                f"❌ Context Intelligence Phase failed for engagement {engagement_id}: {e}"
            )

            # Send error event
            await self.connection_manager.send_update(
                str(engagement_id),
                {
                    "type": "context_intelligence_error",
                    "message": f"⚠️ Context intelligence encountered an issue: {str(e)[:100]}...",
                    "timestamp": datetime.utcnow().isoformat(),
                    "phase": "context_intelligence",
                    "error": str(e),
                },
            )

    async def _simulate_complete_workflow(self, contract):
        """Simulate workflow execution when engines are not available"""
        # Add simulation logic here if needed
        pass
