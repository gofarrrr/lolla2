"""
METIS System Integration - P4.5
Complete CloudEvents integration with workflow and API systems

Provides unified system initialization with event routing, monitoring,
and enterprise-grade event management.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from uuid import uuid4

from src.core.event_router import (
    EventRouter,
    EventRoute,
    EventSubscription,
    RouteAction,
    create_simple_filter,
    create_pattern_filter,
)
from src.core.enhanced_event_bus import (
    create_metis_event_bus_with_router,
    MetisEventBus,
)
from src.core.consolidated_neural_lace_orchestrator import (
    get_consolidated_neural_lace_orchestrator,
)


class SystemMode(str, Enum):
    """METIS system operational modes"""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class SystemConfiguration:
    """System-wide configuration for METIS"""

    mode: SystemMode = SystemMode.DEVELOPMENT
    enable_kafka: bool = True
    enable_monitoring: bool = True
    enable_audit: bool = True
    enable_performance_tracking: bool = True

    # Event routing configuration
    enable_event_routing: bool = True
    route_critical_events: bool = True
    archive_all_events: bool = False

    # Performance settings
    max_concurrent_workflows: int = 50
    event_batch_size: int = 10
    monitoring_interval_seconds: float = 30.0

    # Kafka settings
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_prefix: str = "metis"


class MetisSystemIntegrator:
    """
    Complete METIS system integrator with CloudEvents v1.0 integration
    Manages event routing, workflow orchestration, and API coordination
    """

    def __init__(self, config: SystemConfiguration):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Core components
        self.event_router: Optional[EventRouter] = None
        self.event_bus: Optional[MetisEventBus] = None
        self.workflow_orchestrator = None  # SurgicalWorkflowOrchestrator instance

        # System state
        self._initialized = False
        self._running = False
        self.start_time: Optional[datetime] = None

        # Event handlers registry
        self.event_handlers: Dict[str, Any] = {}
        self.system_subscriptions: List[str] = []

        # Performance tracking
        self.metrics = {
            "workflows_executed": 0,
            "events_processed": 0,
            "system_errors": 0,
            "uptime_seconds": 0,
            "last_health_check": None,
        }

    async def initialize(self) -> bool:
        """Initialize complete METIS system with event integration"""
        if self._initialized:
            self.logger.warning("System already initialized")
            return True

        try:
            self.logger.info(
                f"Initializing METIS system in {self.config.mode.value} mode"
            )

            # Step 1: Initialize event router if enabled
            if self.config.enable_event_routing:
                self.event_router = EventRouter()
                await self.event_router.start()
                self.logger.info("Event router initialized")

            # Step 2: Initialize event bus with router integration
            if self.event_router:
                self.event_bus = create_metis_event_bus_with_router(
                    event_router=self.event_router,
                    bootstrap_servers=self.config.kafka_bootstrap_servers,
                    topic_prefix=self.config.kafka_topic_prefix,
                )
            else:
                # Fallback to basic event bus
                from src.core.enhanced_event_bus import EnhancedKafkaEventBus

                self.event_bus = EnhancedKafkaEventBus(
                    bootstrap_servers=self.config.kafka_bootstrap_servers,
                    topic_prefix=self.config.kafka_topic_prefix,
                )

            await self.event_bus.initialize()
            self.logger.info("Event bus initialized")

            # Step 3: Initialize surgical orchestrator (phantom-proof)
            self.workflow_orchestrator = (
                await get_consolidated_neural_lace_orchestrator()
            )
            self.logger.info("Surgical orchestrator initialized")

            # Step 4: Setup system-wide event routes and subscriptions
            await self._setup_system_event_routing()

            # Step 5: Setup monitoring and audit trails
            if self.config.enable_monitoring:
                await self._setup_monitoring()

            self._initialized = True
            self._running = True
            self.start_time = datetime.utcnow()

            self.logger.info(
                "🚀 METIS system fully initialized with CloudEvents integration"
            )
            return True

        except Exception as e:
            self.logger.error(f"System initialization failed: {str(e)}")
            await self._cleanup()
            return False

    async def _setup_system_event_routing(self):
        """Setup system-wide event routing patterns"""
        if not self.event_router:
            return

        # Route 1: Critical event escalation
        if self.config.route_critical_events:
            critical_route = EventRoute(
                route_id="system_critical_escalation",
                name="Critical Event Escalation",
                filter_criteria=create_simple_filter(priorities=["critical"]),
                action=RouteAction.DELIVER,
                target_handler=self._handle_critical_events,
                priority=1,  # Highest priority
            )
            self.event_router.add_route(critical_route)

        # Route 2: Workflow monitoring
        workflow_route = EventRoute(
            route_id="workflow_monitoring",
            name="Workflow Event Monitor",
            filter_criteria=create_simple_filter(categories=["workflow", "engagement"]),
            action=RouteAction.DELIVER,
            target_handler=self._handle_workflow_events,
            priority=10,
        )
        self.event_router.add_route(workflow_route)

        # Route 3: Performance monitoring
        if self.config.enable_performance_tracking:
            performance_route = EventRoute(
                route_id="performance_monitoring",
                name="Performance Event Monitor",
                filter_criteria=create_simple_filter(categories=["performance"]),
                action=RouteAction.DELIVER,
                target_handler=self._handle_performance_events,
                priority=50,
            )
            self.event_router.add_route(performance_route)

        # Route 4: Event archival (if enabled)
        if self.config.archive_all_events:
            archive_route = EventRoute(
                route_id="event_archival",
                name="Universal Event Archive",
                filter_criteria=create_simple_filter(),  # All events
                action=RouteAction.ARCHIVE,
                priority=100,  # Lowest priority
            )
            self.event_router.add_route(archive_route)

        # Subscription 1: Cognitive event analysis
        cognitive_subscription = EventSubscription(
            subscription_id="cognitive_analysis",
            subscriber_id="system_cognitive_monitor",
            name="Cognitive Event Analysis",
            filter_criteria=create_simple_filter(categories=["cognitive"]),
            delivery_handler=self._handle_cognitive_events,
            batch_size=self.config.event_batch_size,
            batch_timeout_seconds=5.0,
        )
        self.event_router.add_subscription(cognitive_subscription)
        self.system_subscriptions.append(cognitive_subscription.subscription_id)

        # Subscription 2: System health monitoring
        if self.config.enable_monitoring:
            health_subscription = EventSubscription(
                subscription_id="system_health_monitor",
                subscriber_id="health_service",
                name="System Health Monitor",
                filter_criteria=create_pattern_filter(
                    subject_patterns=["system/health/.*", "system/error/.*"],
                    categories=["system"],
                ),
                delivery_handler=self._handle_health_events,
                batch_size=1,  # Immediate delivery for health events
                max_retry_attempts=5,
            )
            self.event_router.add_subscription(health_subscription)
            self.system_subscriptions.append(health_subscription.subscription_id)

        self.logger.info(
            f"System event routing configured with {len(self.event_router.routes)} routes and {len(self.system_subscriptions)} subscriptions"
        )

    async def _setup_monitoring(self):
        """Setup system monitoring and health checks"""
        # Start background health monitoring
        asyncio.create_task(self._health_monitor_loop())
        self.logger.info("System monitoring enabled")

    async def _health_monitor_loop(self):
        """Background health monitoring loop"""
        while self._running:
            try:
                await asyncio.sleep(self.config.monitoring_interval_seconds)
                await self._perform_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitor error: {str(e)}")
                self.metrics["system_errors"] += 1

    async def _perform_health_check(self):
        """Perform comprehensive system health check"""
        health_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": (
                (datetime.utcnow() - self.start_time).total_seconds()
                if self.start_time
                else 0
            ),
            "event_bus_status": (
                self.event_bus.status.value if self.event_bus else "unavailable"
            ),
            "router_stats": (
                self.event_router.get_routing_stats() if self.event_router else None
            ),
            "workflows_executed": self.metrics["workflows_executed"],
            "events_processed": self.metrics["events_processed"],
        }

        self.metrics["last_health_check"] = datetime.utcnow()
        self.metrics["uptime_seconds"] = health_data["uptime_seconds"]

        # Emit health event
        if self.event_bus:
            from src.schemas.event_factories import create_system_component_ready_event

            health_event = create_system_component_ready_event(
                component_name="metis_system",
                status="healthy",
                health_check=health_data,
            )
            await self.event_bus.publish_event(health_event)

    # Event handlers
    async def _handle_critical_events(self, events):
        """Handle critical system events"""
        for event in events:
            self.logger.critical(f"CRITICAL EVENT: {event.type} - {event.id}")
            # Could integrate with alerting systems here
            self.metrics["system_errors"] += 1

    async def _handle_workflow_events(self, events):
        """Handle workflow and engagement events"""
        for event in events:
            if event.type.startswith("workflow."):
                self.logger.debug(f"Workflow event: {event.type}")
            elif event.type.startswith("engagement."):
                self.logger.info(f"Engagement event: {event.type}")
                if event.type == "engagement.workflow.completed":
                    self.metrics["workflows_executed"] += 1

            self.metrics["events_processed"] += 1

    async def _handle_performance_events(self, events):
        """Handle performance monitoring events"""
        for event in events:
            if event.type == "performance.alert.triggered":
                self.logger.warning(f"Performance alert: {event.data}")
            elif event.type == "performance.measurement.recorded":
                self.logger.debug(f"Performance measurement: {event.data}")

            self.metrics["events_processed"] += 1

    async def _handle_cognitive_events(self, events):
        """Handle cognitive processing events"""
        self.logger.debug(f"Processing {len(events)} cognitive events")

        for event in events:
            if event.type == "cognitive.model.selected":
                model_id = event.data.get("model_id") if event.data else "unknown"
                self.logger.debug(f"Mental model selected: {model_id}")

            self.metrics["events_processed"] += 1

    async def _handle_health_events(self, events):
        """Handle system health events"""
        for event in events:
            if "error" in event.subject or "":
                self.logger.warning(f"System health issue: {event.subject}")
                self.metrics["system_errors"] += 1
            else:
                self.logger.debug(f"Health event: {event.subject}")

    # Public API
    async def create_engagement(
        self, problem_statement: str, client_name: str = None, **kwargs
    ) -> str:
        """Create new engagement using integrated workflow orchestrator"""
        if not self._running or not self.workflow_orchestrator:
            raise RuntimeError("System not initialized or not running")

        from src.engine.models.data_contracts import (
            MetisDataContract,
            EngagementContext,
        )

        # Create engagement context
        engagement_context = EngagementContext(
            engagement_id=str(uuid4()),
            problem_statement=problem_statement,
            client_name=client_name or "Unknown Client",
            business_context={
                **kwargs.get("business_context", {}),
                "engagement_type": kwargs.get("engagement_type", "strategic_analysis"),
                "priority_level": kwargs.get("priority_level", "medium"),
            },
        )

        # Create data contract with all required fields
        from src.engine.models.data_contracts import (
            CognitiveState,
            WorkflowState,
            EngagementPhase,
        )

        contract = MetisDataContract(
            type="metis.engagement_request",
            source="/metis/system_integration",
            engagement_context=engagement_context,
            cognitive_state=CognitiveState(),  # Initialize with defaults
            workflow_state=WorkflowState(
                current_phase=EngagementPhase.PROBLEM_STRUCTURING
            ),  # Initialize with required field
            processing_metadata={"system_mode": self.config.mode.value},
        )

        try:
            # Execute engagement workflow
            updated_contract = await self.workflow_orchestrator.execute_engagement(
                contract
            )

            self.logger.info(
                f"Engagement {engagement_context.engagement_id} completed successfully"
            )
            return engagement_context.engagement_id

        except Exception as e:
            self.logger.error(
                f"Engagement {engagement_context.engagement_id} failed: {str(e)}"
            )
            raise

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "initialized": self._initialized,
            "running": self._running,
            "mode": self.config.mode.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "metrics": self.metrics.copy(),
        }

        if self.event_bus:
            status["event_bus"] = {
                "status": self.event_bus.status.value,
                "metrics": self.event_bus.get_metrics(),
            }

        if self.event_router:
            status["event_router"] = self.event_router.get_routing_stats()

        return status

    async def shutdown(self):
        """Gracefully shutdown the complete system"""
        self.logger.info("Shutting down METIS system...")
        self._running = False

        # Remove system subscriptions
        if self.event_router:
            for subscription_id in self.system_subscriptions:
                self.event_router.remove_subscription(subscription_id)

        await self._cleanup()

        self.logger.info("METIS system shutdown complete")

    async def _cleanup(self):
        """Internal cleanup of system resources"""
        if self.event_router:
            await self.event_router.stop()

        if self.event_bus:
            await self.event_bus.shutdown()

        self._initialized = False


# Convenience functions
async def create_metis_system(
    mode: SystemMode = SystemMode.DEVELOPMENT, **config_kwargs
) -> MetisSystemIntegrator:
    """
    Create and initialize complete METIS system

    Usage:
        system = await create_metis_system(SystemMode.PRODUCTION, enable_kafka=True)
        engagement_id = await system.create_engagement("Improve operational efficiency")
        status = system.get_system_status()
        await system.shutdown()
    """
    config = SystemConfiguration(mode=mode, **config_kwargs)
    system = MetisSystemIntegrator(config)

    success = await system.initialize()
    if not success:
        raise RuntimeError("Failed to initialize METIS system")

    return system


def create_development_system_config(**overrides) -> SystemConfiguration:
    """Create development-optimized system configuration"""
    return SystemConfiguration(
        mode=SystemMode.DEVELOPMENT,
        enable_kafka=False,  # Use in-memory for development
        enable_monitoring=True,
        enable_audit=False,  # Simplified for development
        enable_performance_tracking=True,
        enable_event_routing=True,
        route_critical_events=True,
        archive_all_events=False,
        max_concurrent_workflows=10,
        event_batch_size=5,
        monitoring_interval_seconds=10.0,
        **overrides,
    )


def create_production_system_config(**overrides) -> SystemConfiguration:
    """Create production-optimized system configuration"""
    return SystemConfiguration(
        mode=SystemMode.PRODUCTION,
        enable_kafka=True,
        enable_monitoring=True,
        enable_audit=True,
        enable_performance_tracking=True,
        enable_event_routing=True,
        route_critical_events=True,
        archive_all_events=True,
        max_concurrent_workflows=100,
        event_batch_size=20,
        monitoring_interval_seconds=30.0,
        **overrides,
    )


# Export main classes
__all__ = [
    "MetisSystemIntegrator",
    "SystemConfiguration",
    "SystemMode",
    "create_metis_system",
    "create_development_system_config",
    "create_production_system_config",
]
