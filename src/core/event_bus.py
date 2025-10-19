"""
METIS Event Bus Foundation - Complete Kafka Integration
F002: Production-ready Kafka event coordination system

Implements enterprise-grade event-driven architecture with CloudEvents compliance,
distributed system coordination, circuit breakers, and high availability.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum

try:
    from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
    from aiokafka.errors import KafkaError, KafkaConnectionError, KafkaTimeoutError
    from aiokafka.admin import AIOKafkaAdminClient, NewTopic
    from aiokafka.structs import TopicPartition

    KAFKA_AVAILABLE = True
except ImportError:
    # Fallback for development without Kafka
    KAFKA_AVAILABLE = False
    AIOKafkaProducer = None
    AIOKafkaConsumer = None

from src.engine.models.data_contracts import MetisDataContract

# Import CloudEvent from enhanced event bus
from .enhanced_event_bus import EnhancedKafkaEventBus as _MetisEventBusImpl
from src.engine.models.data_contracts import validate_data_contract_compliance


class EventBusStatus(str, Enum):
    """Event bus operational status"""

    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    DISCONNECTED = "disconnected"


@dataclass
class EventHandler:
    """Event handler registration"""

    event_type: str
    handler_func: Callable
    priority: int = 50
    async_handler: bool = True


class CircuitBreakerState(str, Enum):
    """Circuit breaker states for failure resilience"""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""

    failure_threshold: int = 5
    timeout_seconds: int = 60
    success_threshold: int = 3


class MetisEventBus:
    """
    Event-driven coordination system for METIS components
    Implements Kafka-based messaging with circuit breaker patterns
    """

    def __init__(
        self,
        kafka_bootstrap_servers: str = "localhost:9092",
        topic_prefix: str = "metis",
        consumer_group: str = "metis-cognitive-platform",
    ):
        self.kafka_servers = kafka_bootstrap_servers
        self.topic_prefix = topic_prefix
        self.consumer_group = consumer_group

        # Event handling
        self.event_handlers: Dict[str, List[EventHandler]] = {}
        self.status = EventBusStatus.INITIALIZING

        # Kafka components
        self.producer: Optional[AIOKafkaProducer] = None
        self.consumers: Dict[str, AIOKafkaConsumer] = {}

        # Circuit breaker for resilience
        self.circuit_breakers: Dict[str, Dict] = {}
        self.circuit_config = CircuitBreakerConfig()

        # Performance monitoring
        self.event_metrics: Dict[str, Dict] = {
            "events_published": 0,
            "events_consumed": 0,
            "processing_times": [],
            "errors": [],
        }

        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> bool:
        """Initialize Kafka connections and topics"""
        if not KAFKA_AVAILABLE:
            self.logger.warning("Kafka not available, using in-memory fallback")
            self.status = EventBusStatus.READY
            return True

        try:
            # Initialize producer
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.kafka_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                compression_type="gzip",
                max_batch_size=16384,
                linger_ms=10,
            )
            await self.producer.start()

            # Create standard topics
            await self._create_standard_topics()

            self.status = EventBusStatus.READY
            self.logger.info("METIS Event Bus initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize event bus: {e}")
            self.status = EventBusStatus.ERROR
            return False

    async def _create_standard_topics(self):
        """Create standard METIS event topics"""
        standard_topics = [
            "engagement-lifecycle",
            "cognitive-processing",
            "workflow-coordination",
            "component-integration",
            "error-handling",
            "audit-trail",
        ]

        # Topic creation would be handled by Kafka admin client
        # For now, topics are created automatically on first message
        pass

    async def publish_event(
        self, event: MetisDataContract, topic_suffix: str = "general"
    ) -> bool:
        """
        Publish CloudEvents-compliant event to Kafka
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Apply circuit breaker
            if not await self._check_circuit_breaker(f"publish_{topic_suffix}"):
                self.logger.warning(f"Circuit breaker open for topic {topic_suffix}")
                return False

            # Validate event compliance
            event_dict = event.to_cloudevents_dict()
            if not validate_data_contract_compliance(event_dict):
                raise ValueError("Event does not comply with data contract")

            topic_name = f"{self.topic_prefix}-{topic_suffix}"

            if self.producer:
                # Kafka publishing
                await self.producer.send_and_wait(
                    topic_name,
                    value=event_dict,
                    key=str(event.engagement_context.engagement_id).encode("utf-8"),
                )
            else:
                # In-memory fallback for development
                await self._handle_event_in_memory(event_dict)

            # Update metrics
            self.event_metrics["events_published"] += 1
            processing_time = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000
            self.event_metrics["processing_times"].append(processing_time)

            # Record success for circuit breaker
            await self._record_circuit_breaker_success(f"publish_{topic_suffix}")

            self.logger.debug(f"Published event {event.type} to {topic_name}")
            return True

        except Exception as e:
            # Record failure for circuit breaker
            await self._record_circuit_breaker_failure(f"publish_{topic_suffix}")

            self.event_metrics["errors"].append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error": str(e),
                    "event_type": event.type,
                }
            )

            self.logger.error(f"Failed to publish event {event.type}: {e}")
            return False

    async def subscribe_to_events(
        self, event_types: List[str], handler: Callable, topic_suffix: str = "general"
    ) -> bool:
        """
        Subscribe to specific event types with async handler
        """
        try:
            # Register event handlers
            for event_type in event_types:
                if event_type not in self.event_handlers:
                    self.event_handlers[event_type] = []

                self.event_handlers[event_type].append(
                    EventHandler(event_type=event_type, handler_func=handler)
                )

            topic_name = f"{self.topic_prefix}-{topic_suffix}"

            if KAFKA_AVAILABLE and self.status == EventBusStatus.READY:
                # Start Kafka consumer
                await self._start_kafka_consumer(topic_name)

            self.logger.info(
                f"Subscribed to events {event_types} on topic {topic_name}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to subscribe to events: {e}")
            return False

    async def _start_kafka_consumer(self, topic_name: str):
        """Start Kafka consumer for specific topic"""
        if topic_name in self.consumers:
            return  # Already consuming

        consumer = AIOKafkaConsumer(
            topic_name,
            bootstrap_servers=self.kafka_servers,
            group_id=self.consumer_group,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="latest",
        )

        await consumer.start()
        self.consumers[topic_name] = consumer

        # Start consuming in background
        asyncio.create_task(self._consume_messages(consumer, topic_name))

    async def _consume_messages(self, consumer: AIOKafkaConsumer, topic_name: str):
        """Consume and process messages from Kafka topic"""
        try:
            async for message in consumer:
                start_time = datetime.now(timezone.utc)

                try:
                    event_data = message.value
                    await self._process_consumed_event(event_data)

                    # Update metrics
                    self.event_metrics["events_consumed"] += 1
                    processing_time = (
                        datetime.now(timezone.utc) - start_time
                    ).total_seconds() * 1000
                    self.event_metrics["processing_times"].append(processing_time)

                except Exception as e:
                    self.logger.error(
                        f"Error processing message from {topic_name}: {e}"
                    )

        except Exception as e:
            self.logger.error(f"Consumer error for {topic_name}: {e}")
            # Attempt to restart consumer
            await self._restart_consumer(topic_name)

    async def _process_consumed_event(self, event_data: Dict[str, Any]):
        """Process consumed event by calling registered handlers"""
        event_type = event_data.get("type")
        if not event_type:
            return

        handlers = self.event_handlers.get(event_type, [])
        if not handlers:
            self.logger.debug(f"No handlers registered for event type: {event_type}")
            return

        # Sort handlers by priority
        handlers.sort(key=lambda h: h.priority)

        # Execute handlers
        for handler in handlers:
            try:
                if handler.async_handler:
                    await handler.handler_func(event_data)
                else:
                    handler.handler_func(event_data)
            except Exception as e:
                self.logger.error(f"Handler error for {event_type}: {e}")

    async def _handle_event_in_memory(self, event_data: Dict[str, Any]):
        """Fallback in-memory event handling for development"""
        await self._process_consumed_event(event_data)

    async def _check_circuit_breaker(self, operation: str) -> bool:
        """Check if circuit breaker allows operation"""
        if operation not in self.circuit_breakers:
            self.circuit_breakers[operation] = {
                "state": CircuitBreakerState.CLOSED,
                "failure_count": 0,
                "last_failure_time": None,
                "success_count": 0,
            }

        breaker = self.circuit_breakers[operation]

        if breaker["state"] == CircuitBreakerState.OPEN:
            # Check if timeout period has passed
            if breaker["last_failure_time"]:
                time_since_failure = (
                    datetime.now(timezone.utc) - breaker["last_failure_time"]
                ).total_seconds()
                if time_since_failure >= self.circuit_config.timeout_seconds:
                    breaker["state"] = CircuitBreakerState.HALF_OPEN
                    breaker["success_count"] = 0
                    return True
            return False

        return True

    async def _record_circuit_breaker_failure(self, operation: str):
        """Record failure for circuit breaker"""
        if operation not in self.circuit_breakers:
            await self._check_circuit_breaker(operation)  # Initialize

        breaker = self.circuit_breakers[operation]
        breaker["failure_count"] += 1
        breaker["last_failure_time"] = datetime.now(timezone.utc)

        if breaker["failure_count"] >= self.circuit_config.failure_threshold:
            breaker["state"] = CircuitBreakerState.OPEN
            self.logger.warning(f"Circuit breaker opened for operation: {operation}")

    async def _record_circuit_breaker_success(self, operation: str):
        """Record success for circuit breaker"""
        if operation not in self.circuit_breakers:
            return

        breaker = self.circuit_breakers[operation]

        if breaker["state"] == CircuitBreakerState.HALF_OPEN:
            breaker["success_count"] += 1
            if breaker["success_count"] >= self.circuit_config.success_threshold:
                breaker["state"] = CircuitBreakerState.CLOSED
                breaker["failure_count"] = 0
                self.logger.info(f"Circuit breaker closed for operation: {operation}")
        elif breaker["state"] == CircuitBreakerState.CLOSED:
            breaker["failure_count"] = max(0, breaker["failure_count"] - 1)

    async def _restart_consumer(self, topic_name: str):
        """Restart failed consumer"""
        if topic_name in self.consumers:
            try:
                await self.consumers[topic_name].stop()
            except:
                pass
            del self.consumers[topic_name]

        # Wait before restart
        await asyncio.sleep(5)

        try:
            await self._start_kafka_consumer(topic_name)
            self.logger.info(f"Restarted consumer for topic: {topic_name}")
        except Exception as e:
            self.logger.error(f"Failed to restart consumer for {topic_name}: {e}")

    async def get_health_status(self) -> Dict[str, Any]:
        """Get event bus health and metrics"""
        return {
            "status": self.status.value,
            "kafka_available": KAFKA_AVAILABLE,
            "active_consumers": len(self.consumers),
            "registered_handlers": len(self.event_handlers),
            "circuit_breakers": {
                op: {
                    "state": (
                        breaker["state"].value
                        if hasattr(breaker["state"], "value")
                        else breaker["state"]
                    ),
                    "failure_count": breaker["failure_count"],
                }
                for op, breaker in self.circuit_breakers.items()
            },
            "metrics": {
                "events_published": self.event_metrics["events_published"],
                "events_consumed": self.event_metrics["events_consumed"],
                "avg_processing_time_ms": (
                    sum(self.event_metrics["processing_times"])
                    / len(self.event_metrics["processing_times"])
                    if self.event_metrics["processing_times"]
                    else 0
                ),
                "error_count": len(self.event_metrics["errors"]),
            },
        }

    async def shutdown(self):
        """Graceful shutdown of event bus"""
        self.logger.info("Shutting down METIS Event Bus")

        # Stop all consumers
        for consumer in self.consumers.values():
            try:
                await consumer.stop()
            except:
                pass

        # Stop producer
        if self.producer:
            try:
                await self.producer.stop()
            except:
                pass

        self.status = EventBusStatus.DISCONNECTED


# Global event bus instance
_event_bus_instance: Optional[MetisEventBus] = None


async def get_event_bus() -> MetisEventBus:
    """Get or create global event bus instance"""
    global _event_bus_instance

    if _event_bus_instance is None:
        _event_bus_instance = MetisEventBus()
        await _event_bus_instance.initialize()

    return _event_bus_instance


# Utility functions for common event patterns
async def publish_engagement_event(event: MetisDataContract) -> bool:
    """Publish engagement lifecycle event"""
    bus = await get_event_bus()
    return await bus.publish_event(event, "engagement-lifecycle")


async def publish_cognitive_event(event: MetisDataContract) -> bool:
    """Publish cognitive processing event"""
    bus = await get_event_bus()
    return await bus.publish_event(event, "cognitive-processing")


async def publish_workflow_event(event: MetisDataContract) -> bool:
    """Publish workflow coordination event"""
    bus = await get_event_bus()
    return await bus.publish_event(event, "workflow-coordination")


async def subscribe_to_engagement_events(handler: Callable) -> bool:
    """Subscribe to engagement lifecycle events"""
    bus = await get_event_bus()
    return await bus.subscribe_to_events(
        ["metis.engagement_initiated", "metis.engagement_completed"],
        handler,
        "engagement-lifecycle",
    )


async def subscribe_to_cognitive_events(handler: Callable) -> bool:
    """Subscribe to cognitive processing events"""
    bus = await get_event_bus()
    return await bus.subscribe_to_events(
        ["metis.cognitive_model_selected", "metis.analysis_framework_applied"],
        handler,
        "cognitive-processing",
    )


# Backward-compatibility alias expected by some tests
EventBus = MetisEventBus
