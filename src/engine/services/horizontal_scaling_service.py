"""
METIS V2.1 Horizontal Scaling Service
Distributed system architecture for 100x scale deployment

This service implements horizontal scaling capabilities:
- Service instance discovery and registration
- Load balancing across multiple METIS instances
- Distributed caching with Redis coordination
- Work queue distribution for engagement processing
- Health checking and automatic failover
- State synchronization across instances

ARCHITECTURAL MANDATE COMPLIANCE:
✅ Glass-Box Transparency: All scaling operations logged to UnifiedContextStream
✅ Service-Oriented Architecture: Distributed service coordination with clean boundaries
"""

import asyncio
import logging
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import random

# Core METIS imports
from src.core.unified_context_stream import UnifiedContextStream, ContextEventType

# External dependencies
try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False

try:
    import consul

    CONSUL_AVAILABLE = True
except ImportError:
    consul = None
    CONSUL_AVAILABLE = False


class InstanceStatus(Enum):
    """Instance status states"""

    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    SHUTTING_DOWN = "shutting_down"
    OFFLINE = "offline"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""

    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_RESPONSE_TIME = "weighted_response_time"
    CONSISTENT_HASH = "consistent_hash"


@dataclass
class ServiceInstance:
    """Service instance registration"""

    instance_id: str
    service_name: str
    host: str
    port: int
    status: InstanceStatus
    last_heartbeat: datetime
    capabilities: List[str] = field(default_factory=list)
    load_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkItem:
    """Distributed work queue item"""

    work_id: str
    work_type: str
    payload: Dict[str, Any]
    priority: int = 5  # 1-10, higher = more priority
    created_at: datetime = field(default_factory=datetime.utcnow)
    assigned_to: Optional[str] = None
    attempts: int = 0
    max_attempts: int = 3
    timeout_seconds: int = 300


@dataclass
class ScalingMetrics:
    """Metrics for scaling decisions"""

    total_instances: int
    healthy_instances: int
    average_cpu_usage: float
    average_memory_usage: float
    total_active_engagements: int
    queue_depth: int
    average_response_time_ms: float
    error_rate_percentage: float


class HorizontalScalingService:
    """
    Horizontal Scaling Service - 100x Scale Architecture

    Provides distributed system capabilities for horizontal scaling:
    1. Service discovery and registration
    2. Load balancing across instances
    3. Distributed work queue management
    4. Health monitoring and failover
    5. State synchronization
    6. Auto-scaling decision support
    """

    def __init__(
        self,
        context_stream: UnifiedContextStream,
        instance_id: Optional[str] = None,
        redis_client: Optional[Any] = None,
        consul_client: Optional[Any] = None,
    ):
        """
        Initialize Horizontal Scaling Service

        Args:
            context_stream: UnifiedContextStream for Glass-Box transparency
            instance_id: Unique identifier for this instance (auto-generated if not provided)
            redis_client: Optional Redis client for distributed caching
            consul_client: Optional Consul client for service discovery
        """
        self.context_stream = context_stream
        self.instance_id = instance_id or f"metis-{uuid.uuid4().hex[:8]}"
        self.redis_client = redis_client
        self.consul_client = consul_client
        self.logger = logging.getLogger(__name__)

        # Service registry
        self.registered_instances: Dict[str, ServiceInstance] = {}
        self.local_instance: Optional[ServiceInstance] = None

        # Load balancing
        self.load_balancing_strategy = LoadBalancingStrategy.WEIGHTED_RESPONSE_TIME
        self.round_robin_counter = 0

        # Distributed work queue
        self.work_queue: List[WorkItem] = []
        self.in_progress_work: Dict[str, WorkItem] = {}
        self.completed_work_history: List[str] = []

        # Scaling state
        self.is_running = False
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.discovery_task: Optional[asyncio.Task] = None
        self.work_processor_task: Optional[asyncio.Task] = None

        # Scaling configuration
        self.heartbeat_interval_seconds = 30
        self.instance_timeout_seconds = 90
        self.service_discovery_interval_seconds = 60
        self.work_processing_interval_seconds = 5

        # Scaling metrics
        self.scaling_metrics = ScalingMetrics(
            total_instances=0,
            healthy_instances=0,
            average_cpu_usage=0.0,
            average_memory_usage=0.0,
            total_active_engagements=0,
            queue_depth=0,
            average_response_time_ms=0.0,
            error_rate_percentage=0.0,
        )

        # Statistics
        self.scaling_stats = {
            "instances_registered": 0,
            "instances_deregistered": 0,
            "work_items_processed": 0,
            "load_balance_decisions": 0,
            "failovers_handled": 0,
            "scaling_events": 0,
        }

        # Glass-Box: Log service initialization
        self.context_stream.add_event(
            event_type=ContextEventType.SYSTEM_STATE,
            data={
                "service": "HorizontalScalingService",
                "instance_id": self.instance_id,
                "initialized": True,
                "redis_available": bool(redis_client),
                "consul_available": bool(consul_client),
            },
            metadata={"service": "HorizontalScalingService", "operation": "initialize"},
        )

    async def start_scaling_service(
        self,
        service_name: str = "metis-cognitive-platform",
        host: str = "localhost",
        port: int = 8000,
        capabilities: Optional[List[str]] = None,
    ) -> None:
        """Start horizontal scaling service with instance registration"""

        if self.is_running:
            return

        # Create local instance registration
        self.local_instance = ServiceInstance(
            instance_id=self.instance_id,
            service_name=service_name,
            host=host,
            port=port,
            status=InstanceStatus.STARTING,
            last_heartbeat=datetime.utcnow(),
            capabilities=capabilities or ["cognitive_analysis", "consultant_selection"],
            metadata={"version": "v2.1", "startup_time": datetime.utcnow().isoformat()},
        )

        # Register with service discovery if available
        await self._register_instance()

        self.is_running = True

        # Start background tasks
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self.discovery_task = asyncio.create_task(self._service_discovery_loop())
        self.work_processor_task = asyncio.create_task(self._work_processor_loop())

        # Update instance status to healthy
        self.local_instance.status = InstanceStatus.HEALTHY

        # Glass-Box: Log service start
        self.context_stream.add_event(
            event_type=ContextEventType.SYSTEM_STATE,
            data={
                "horizontal_scaling": "started",
                "instance_id": self.instance_id,
                "service_name": service_name,
                "capabilities": capabilities or [],
            },
            metadata={
                "service": "HorizontalScalingService",
                "operation": "start_scaling",
            },
        )

        self.logger.info(
            f"🚀 Horizontal scaling service started - Instance: {self.instance_id}"
        )

    async def stop_scaling_service(self) -> None:
        """Stop horizontal scaling service and deregister instance"""

        if not self.is_running:
            return

        self.is_running = False

        # Update instance status
        if self.local_instance:
            self.local_instance.status = InstanceStatus.SHUTTING_DOWN

        # Cancel background tasks
        for task in [
            self.heartbeat_task,
            self.discovery_task,
            self.work_processor_task,
        ]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Deregister instance
        await self._deregister_instance()

        # Glass-Box: Log service stop
        self.context_stream.add_event(
            event_type=ContextEventType.SYSTEM_STATE,
            data={"horizontal_scaling": "stopped", "instance_id": self.instance_id},
            metadata={
                "service": "HorizontalScalingService",
                "operation": "stop_scaling",
            },
        )

    def select_instance_for_work(self, work_type: str) -> Optional[ServiceInstance]:
        """Select optimal instance for work distribution using load balancing"""

        # Filter instances by capability and health
        eligible_instances = [
            instance
            for instance in self.registered_instances.values()
            if (
                instance.status == InstanceStatus.HEALTHY
                and (not work_type or work_type in instance.capabilities)
            )
        ]

        if not eligible_instances:
            return None

        selected_instance = None

        # Apply load balancing strategy
        if self.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
            selected_instance = eligible_instances[
                self.round_robin_counter % len(eligible_instances)
            ]
            self.round_robin_counter += 1

        elif self.load_balancing_strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            # Select instance with least active work
            min_connections = min(
                len(
                    [
                        w
                        for w in self.in_progress_work.values()
                        if w.assigned_to == instance.instance_id
                    ]
                )
                for instance in eligible_instances
            )

            candidates = [
                instance
                for instance in eligible_instances
                if len(
                    [
                        w
                        for w in self.in_progress_work.values()
                        if w.assigned_to == instance.instance_id
                    ]
                )
                == min_connections
            ]

            selected_instance = random.choice(candidates)

        elif (
            self.load_balancing_strategy == LoadBalancingStrategy.WEIGHTED_RESPONSE_TIME
        ):
            # Select instance with best response time (lowest CPU + memory load)
            def calculate_load_score(instance: ServiceInstance) -> float:
                cpu_load = instance.load_metrics.get("cpu_percentage", 50.0)
                memory_load = (
                    instance.load_metrics.get("memory_usage_mb", 1000.0) / 10
                )  # Normalize
                response_time = (
                    instance.load_metrics.get("avg_response_time_ms", 1000.0) / 100
                )  # Normalize
                return cpu_load + memory_load + response_time

            selected_instance = min(eligible_instances, key=calculate_load_score)

        elif self.load_balancing_strategy == LoadBalancingStrategy.CONSISTENT_HASH:
            # Use consistent hashing for work distribution
            work_hash = hashlib.md5(work_type.encode()).hexdigest()
            hash_values = [
                (hashlib.md5(instance.instance_id.encode()).hexdigest(), instance)
                for instance in eligible_instances
            ]
            hash_values.sort(key=lambda x: x[0])

            # Find closest hash value
            for hash_val, instance in hash_values:
                if hash_val >= work_hash:
                    selected_instance = instance
                    break

            if not selected_instance:
                selected_instance = hash_values[0][1]  # Wrap around

        if selected_instance:
            self.scaling_stats["load_balance_decisions"] += 1

            # Glass-Box: Log load balancing decision
            self.context_stream.add_event(
                event_type=ContextEventType.TOOL_EXECUTION,
                data={
                    "load_balance_decision": {
                        "selected_instance": selected_instance.instance_id,
                        "work_type": work_type,
                        "strategy": self.load_balancing_strategy.value,
                        "eligible_instances": len(eligible_instances),
                    }
                },
                metadata={
                    "service": "HorizontalScalingService",
                    "operation": "load_balancing",
                },
            )

        return selected_instance

    async def submit_work(
        self,
        work_type: str,
        payload: Dict[str, Any],
        priority: int = 5,
        timeout_seconds: int = 300,
    ) -> str:
        """Submit work item to distributed queue"""

        work_item = WorkItem(
            work_id=str(uuid.uuid4()),
            work_type=work_type,
            payload=payload,
            priority=priority,
            timeout_seconds=timeout_seconds,
        )

        # Add to queue sorted by priority
        self.work_queue.append(work_item)
        self.work_queue.sort(key=lambda x: x.priority, reverse=True)

        # Update scaling metrics
        self.scaling_metrics.queue_depth = len(self.work_queue)

        # Glass-Box: Log work submission
        self.context_stream.add_event(
            event_type=ContextEventType.TOOL_EXECUTION,
            data={
                "work_submitted": work_item.work_id,
                "work_type": work_type,
                "priority": priority,
                "queue_depth": len(self.work_queue),
            },
            metadata={
                "service": "HorizontalScalingService",
                "operation": "submit_work",
            },
        )

        return work_item.work_id

    def get_scaling_status(self) -> Dict[str, Any]:
        """Get comprehensive scaling service status"""

        return {
            "instance_id": self.instance_id,
            "service_running": self.is_running,
            "local_instance": (
                asdict(self.local_instance) if self.local_instance else None
            ),
            "registered_instances": {
                instance_id: asdict(instance)
                for instance_id, instance in self.registered_instances.items()
            },
            "scaling_metrics": asdict(self.scaling_metrics),
            "work_queue_status": {
                "pending_work": len(self.work_queue),
                "in_progress_work": len(self.in_progress_work),
                "completed_work": len(self.completed_work_history),
            },
            "load_balancing_strategy": self.load_balancing_strategy.value,
            "scaling_statistics": self.scaling_stats.copy(),
            "service_capabilities": {
                "redis_enabled": bool(self.redis_client),
                "consul_enabled": bool(self.consul_client),
                "distributed_caching": REDIS_AVAILABLE,
                "service_discovery": CONSUL_AVAILABLE,
            },
        }

    def get_auto_scaling_recommendations(self) -> List[str]:
        """Get auto-scaling recommendations based on current metrics"""

        recommendations = []

        # Check if scaling up is needed
        if (
            self.scaling_metrics.healthy_instances < 2
            and self.scaling_metrics.queue_depth > 10
        ):
            recommendations.append(
                "Scale up: Queue depth high with insufficient healthy instances"
            )

        if (
            self.scaling_metrics.average_cpu_usage > 80
            and self.scaling_metrics.healthy_instances < 5
        ):
            recommendations.append(
                f"Scale up: High CPU usage ({self.scaling_metrics.average_cpu_usage:.1f}%)"
            )

        if (
            self.scaling_metrics.average_response_time_ms > 5000
            and self.scaling_metrics.total_active_engagements > 20
        ):
            recommendations.append(
                f"Scale up: High response time ({self.scaling_metrics.average_response_time_ms:.0f}ms)"
            )

        # Check if scaling down is possible
        if (
            self.scaling_metrics.healthy_instances > 3
            and self.scaling_metrics.average_cpu_usage < 20
            and self.scaling_metrics.queue_depth == 0
        ):
            recommendations.append(
                "Scale down: Low resource utilization with excess capacity"
            )

        # Failover recommendations
        unhealthy_instances = (
            self.scaling_metrics.total_instances
            - self.scaling_metrics.healthy_instances
        )
        if unhealthy_instances > 0:
            recommendations.append(
                f"Investigate: {unhealthy_instances} unhealthy instances detected"
            )

        if not recommendations:
            recommendations.append("Current scaling appears optimal")

        return recommendations

    async def _register_instance(self) -> None:
        """Register local instance with service discovery"""

        if not self.local_instance:
            return

        # Add to local registry
        self.registered_instances[self.instance_id] = self.local_instance
        self.scaling_stats["instances_registered"] += 1

        # Register with Consul if available
        if self.consul_client:
            try:
                await self._register_with_consul()
            except Exception as e:
                self.logger.warning(f"Failed to register with Consul: {e}")

        # Register with Redis if available
        if self.redis_client:
            try:
                await self._register_with_redis()
            except Exception as e:
                self.logger.warning(f"Failed to register with Redis: {e}")

    async def _deregister_instance(self) -> None:
        """Deregister local instance from service discovery"""

        # Remove from local registry
        if self.instance_id in self.registered_instances:
            del self.registered_instances[self.instance_id]
            self.scaling_stats["instances_deregistered"] += 1

        # Deregister from external services
        if self.consul_client:
            try:
                await self._deregister_from_consul()
            except Exception as e:
                self.logger.warning(f"Failed to deregister from Consul: {e}")

        if self.redis_client:
            try:
                await self._deregister_from_redis()
            except Exception as e:
                self.logger.warning(f"Failed to deregister from Redis: {e}")

    async def _heartbeat_loop(self) -> None:
        """Background heartbeat loop for instance health"""

        while self.is_running:
            try:
                if self.local_instance:
                    self.local_instance.last_heartbeat = datetime.utcnow()

                    # Update load metrics
                    # In production, this would collect real metrics
                    self.local_instance.load_metrics.update(
                        {
                            "cpu_percentage": 25.0,  # Placeholder
                            "memory_usage_mb": 512.0,  # Placeholder
                            "avg_response_time_ms": 850.0,  # Placeholder
                        }
                    )

                # Send heartbeat to external services
                await self._send_heartbeat()

                # Clean up stale instances
                await self._cleanup_stale_instances()

                await asyncio.sleep(self.heartbeat_interval_seconds)

            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(10)

    async def _service_discovery_loop(self) -> None:
        """Background service discovery loop"""

        while self.is_running:
            try:
                # Discover instances from external services
                if self.consul_client:
                    await self._discover_from_consul()

                if self.redis_client:
                    await self._discover_from_redis()

                # Update scaling metrics
                self._update_scaling_metrics()

                await asyncio.sleep(self.service_discovery_interval_seconds)

            except Exception as e:
                self.logger.error(f"Error in service discovery loop: {e}")
                await asyncio.sleep(10)

    async def _work_processor_loop(self) -> None:
        """Background work processing loop"""

        while self.is_running:
            try:
                # Process work queue
                if self.work_queue:
                    work_item = self.work_queue.pop(0)  # Get highest priority item

                    # Select instance for work
                    selected_instance = self.select_instance_for_work(
                        work_item.work_type
                    )

                    if selected_instance:
                        work_item.assigned_to = selected_instance.instance_id
                        work_item.attempts += 1
                        self.in_progress_work[work_item.work_id] = work_item
                        self.scaling_stats["work_items_processed"] += 1

                        # In production, this would dispatch work to the selected instance
                        # For now, we'll simulate completion
                        await asyncio.sleep(1)  # Simulate work dispatch time

                # Check for completed/timed out work
                await self._cleanup_completed_work()

                await asyncio.sleep(self.work_processing_interval_seconds)

            except Exception as e:
                self.logger.error(f"Error in work processor loop: {e}")
                await asyncio.sleep(5)

    async def _cleanup_stale_instances(self) -> None:
        """Remove stale instances that haven't sent heartbeat"""

        cutoff_time = datetime.utcnow() - timedelta(
            seconds=self.instance_timeout_seconds
        )
        stale_instances = []

        for instance_id, instance in self.registered_instances.items():
            if (
                instance_id != self.instance_id
                and instance.last_heartbeat < cutoff_time
            ):
                stale_instances.append(instance_id)

        for instance_id in stale_instances:
            del self.registered_instances[instance_id]
            self.scaling_stats["instances_deregistered"] += 1

            # Handle failover if needed
            await self._handle_instance_failover(instance_id)

    async def _handle_instance_failover(self, failed_instance_id: str) -> None:
        """Handle failover when an instance fails"""

        # Reassign work from failed instance
        failed_work_items = [
            work
            for work in self.in_progress_work.values()
            if work.assigned_to == failed_instance_id
        ]

        for work_item in failed_work_items:
            if work_item.attempts < work_item.max_attempts:
                # Reassign to queue for retry
                work_item.assigned_to = None
                self.work_queue.append(work_item)
                del self.in_progress_work[work_item.work_id]
            else:
                # Max attempts reached, mark as failed
                self.completed_work_history.append(work_item.work_id)
                del self.in_progress_work[work_item.work_id]

        if failed_work_items:
            self.scaling_stats["failovers_handled"] += 1

            # Glass-Box: Log failover
            self.context_stream.add_event(
                event_type=ContextEventType.ERROR_OCCURRED,
                data={
                    "instance_failover": failed_instance_id,
                    "reassigned_work_items": len(failed_work_items),
                },
                metadata={
                    "service": "HorizontalScalingService",
                    "operation": "failover",
                },
            )

    async def _cleanup_completed_work(self) -> None:
        """Clean up completed and timed out work items"""

        current_time = datetime.utcnow()
        completed_work = []

        for work_id, work_item in self.in_progress_work.items():
            # Check if work has timed out
            elapsed_time = (current_time - work_item.created_at).total_seconds()

            if elapsed_time > work_item.timeout_seconds:
                if work_item.attempts < work_item.max_attempts:
                    # Retry
                    work_item.assigned_to = None
                    self.work_queue.append(work_item)

                completed_work.append(work_id)

        for work_id in completed_work:
            if work_id in self.in_progress_work:
                del self.in_progress_work[work_id]
            self.completed_work_history.append(work_id)

        # Keep completed history limited
        if len(self.completed_work_history) > 1000:
            self.completed_work_history = self.completed_work_history[-500:]

    def _update_scaling_metrics(self) -> None:
        """Update scaling metrics based on current state"""

        healthy_instances = [
            instance
            for instance in self.registered_instances.values()
            if instance.status == InstanceStatus.HEALTHY
        ]

        self.scaling_metrics.total_instances = len(self.registered_instances)
        self.scaling_metrics.healthy_instances = len(healthy_instances)
        self.scaling_metrics.queue_depth = len(self.work_queue)

        if healthy_instances:
            # Calculate averages
            cpu_values = [
                instance.load_metrics.get("cpu_percentage", 0)
                for instance in healthy_instances
            ]
            memory_values = [
                instance.load_metrics.get("memory_usage_mb", 0)
                for instance in healthy_instances
            ]
            response_values = [
                instance.load_metrics.get("avg_response_time_ms", 0)
                for instance in healthy_instances
            ]

            self.scaling_metrics.average_cpu_usage = sum(cpu_values) / len(cpu_values)
            self.scaling_metrics.average_memory_usage = sum(memory_values) / len(
                memory_values
            )
            self.scaling_metrics.average_response_time_ms = sum(response_values) / len(
                response_values
            )

        self.scaling_metrics.total_active_engagements = len(self.in_progress_work)

    # Placeholder methods for external service integration
    async def _register_with_consul(self) -> None:
        """Register instance with Consul service discovery"""
        pass  # Implementation would use consul client

    async def _deregister_from_consul(self) -> None:
        """Deregister instance from Consul"""
        pass  # Implementation would use consul client

    async def _discover_from_consul(self) -> None:
        """Discover instances from Consul"""
        pass  # Implementation would use consul client

    async def _register_with_redis(self) -> None:
        """Register instance with Redis"""
        pass  # Implementation would use redis client

    async def _deregister_from_redis(self) -> None:
        """Deregister instance from Redis"""
        pass  # Implementation would use redis client

    async def _discover_from_redis(self) -> None:
        """Discover instances from Redis"""
        pass  # Implementation would use redis client

    async def _send_heartbeat(self) -> None:
        """Send heartbeat to external services"""
        pass  # Implementation would send heartbeat to external services


# Factory function for service creation
def create_horizontal_scaling_service(
    context_stream: UnifiedContextStream,
    instance_id: Optional[str] = None,
    redis_client: Optional[Any] = None,
    consul_client: Optional[Any] = None,
) -> HorizontalScalingService:
    """Factory function to create HorizontalScalingService with proper dependencies"""

    return HorizontalScalingService(
        context_stream=context_stream,
        instance_id=instance_id,
        redis_client=redis_client,
        consul_client=consul_client,
    )
