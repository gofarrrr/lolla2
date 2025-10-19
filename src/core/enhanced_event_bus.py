"""
METIS Enhanced Event Bus - Complete Kafka Integration
F002: Production-ready Kafka event coordination system

Implements enterprise-grade event-driven architecture with CloudEvents compliance,
distributed system coordination, circuit breakers, high availability, and monitoring.
"""

import asyncio
import json
import logging
import ssl
import os
from typing import Dict, List, Optional, Callable, Any, Set
from datetime import datetime, timezone
from uuid import uuid4
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import defaultdict, deque

try:
    from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
    from aiokafka.errors import KafkaError, KafkaConnectionError, KafkaTimeoutError
    from aiokafka.admin import AIOKafkaAdminClient, NewTopic
    from aiokafka.structs import TopicPartition, OffsetAndMetadata
    from aiokafka.consumer.subscription_state import SubscriptionState

    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    AIOKafkaProducer = None
    AIOKafkaConsumer = None


class EventBusStatus(str, Enum):
    """Event bus operational status"""

    INITIALIZING = "initializing"
    READY = "ready"
    DEGRADED = "degraded"
    ERROR = "error"
    DISCONNECTED = "disconnected"
    MAINTENANCE = "maintenance"


class EventPriority(str, Enum):
    """Event priority levels"""

    CRITICAL = "critical"  # System critical events
    HIGH = "high"  # Important business events
    MEDIUM = "medium"  # Standard events
    LOW = "low"  # Background/audit events


class EventCategory(str, Enum):
    """Event categories for routing"""

    COGNITIVE = "cognitive"  # Cognitive processing events
    WORKFLOW = "workflow"  # Workflow execution events
    ENGAGEMENT = "engagement"  # Engagement lifecycle events
    SYSTEM = "system"  # System health and monitoring
    AUDIT = "audit"  # Audit and compliance events
    ANALYTICS = "analytics"  # Analytics and metrics events


@dataclass
class CloudEvent:
    """CloudEvents specification compliant event structure"""

    # Required CloudEvents fields
    specversion: str = "1.0"
    type: str = ""
    source: str = ""
    id: str = field(default_factory=lambda: str(uuid4()))
    time: str = field(
        default_factory=lambda: datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )

    # Optional CloudEvents fields
    datacontenttype: str = "application/json"
    dataschema: Optional[str] = None
    subject: Optional[str] = None

    # METIS extensions (CloudEvents v1.0 compliant naming: lowercase + digits only)
    metispriority: str = "medium"  # high, medium, low, critical
    metiscategory: str = (
        "system"  # cognitive, workflow, engagement, system, audit, analytics
    )
    metistenantid: Optional[str] = None
    metisengagementid: Optional[str] = None
    metiscorrelationid: Optional[str] = None

    # Event data (CloudEvents v1.0: supports both data and data_base64)
    data: Optional[Dict[str, Any]] = None
    data_base64: Optional[str] = None  # For binary data, mutually exclusive with data

    # Metadata
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to CloudEvents v1.0 compliant dictionary"""
        # Required CloudEvents fields
        result = {
            "specversion": self.specversion,
            "type": self.type,
            "source": self.source,
            "id": self.id,
            "time": self.time,
            "datacontenttype": self.datacontenttype,
        }

        # Add optional CloudEvents fields only if present
        if self.dataschema:
            result["dataschema"] = self.dataschema
        if self.subject:
            result["subject"] = self.subject

        # Add METIS extensions (CloudEvents v1.0 compliant naming)
        result.update(
            {
                "metispriority": self.metispriority,
                "metiscategory": self.metiscategory,
                "metistenantid": self.metistenantid,
                "metisengagementid": self.metisengagementid,
                "metiscorrelationid": self.metiscorrelationid,
                "metisretrycount": self.retry_count,
            }
        )

        # Add data field (CloudEvents v1.0: data and data_base64 are mutually exclusive)
        if self.data_base64 is not None:
            result["data_base64"] = self.data_base64
        elif self.data is not None:
            result["data"] = self.data

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CloudEvent":
        """Create CloudEvent from CloudEvents v1.0 compliant dictionary"""
        # Handle both data and data_base64 (mutually exclusive)
        event_data = None
        event_data_base64 = None

        if "data_base64" in data:
            event_data_base64 = data["data_base64"]
        elif "data" in data:
            event_data = data["data"]

        event = cls(
            specversion=data.get("specversion", "1.0"),
            type=data["type"],
            source=data["source"],
            id=data["id"],
            time=data["time"],
            datacontenttype=data.get("datacontenttype", "application/json"),
            dataschema=data.get("dataschema"),
            subject=data.get("subject"),
            # CloudEvents v1.0 compliant METIS extensions (with backward compatibility)
            metispriority=data.get(
                "metispriority", data.get("metis_priority", "medium")
            ),
            metiscategory=data.get(
                "metiscategory", data.get("metis_category", "system")
            ),
            metistenantid=data.get("metistenantid", data.get("metis_tenant_id")),
            metisengagementid=data.get(
                "metisengagementid", data.get("metis_engagement_id")
            ),
            metiscorrelationid=data.get(
                "metiscorrelationid", data.get("metis_correlation_id")
            ),
            retry_count=data.get("metisretrycount", data.get("metis_retry_count", 0)),
            data=event_data,
            data_base64=event_data_base64,
        )
        return event

    def is_cloudevents_v1_compliant(self) -> bool:
        """Validate CloudEvents v1.0 compliance"""
        # Check required fields
        if not all([self.specversion, self.type, self.source, self.id]):
            return False

        # Check specversion is 1.0
        if self.specversion != "1.0":
            return False

        # Check mutual exclusivity of data and data_base64
        # Both fields should not be set simultaneously (CloudEvents v1.0 requirement)
        has_data = self.data is not None and (
            self.data != {} if isinstance(self.data, dict) else True
        )
        has_data_base64 = self.data_base64 is not None and self.data_base64 != ""

        if has_data and has_data_base64:
            return False

        # Check attribute naming (METIS extensions should be lowercase + digits)
        metis_attrs = [self.metispriority, self.metiscategory]
        for attr in metis_attrs:
            if attr and not all(
                c.islower() or c.isdigit() for c in attr if c.isalpha()
            ):
                return False

        return True

    def set_cloudevents_json_content_type(self):
        """Set CloudEvents v1.0 JSON content type"""
        self.datacontenttype = "application/cloudevents+json"


def create_metis_cloud_event(
    event_type: str,
    source: str,
    data: Optional[Dict[str, Any]] = None,
    category: str = "system",
    priority: str = "medium",
    tenant_id: Optional[str] = None,
    engagement_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    subject: Optional[str] = None,
) -> CloudEvent:
    """Factory function for METIS CloudEvents v1.0 compliant events"""
    # Only set data if it's provided and not empty
    event_data = data if data else None

    return CloudEvent(
        type=event_type,
        source=source,
        data=event_data,
        metiscategory=category,
        metispriority=priority,
        metistenantid=tenant_id,
        metisengagementid=engagement_id,
        metiscorrelationid=correlation_id,
        subject=subject,
    )


@dataclass
class EventHandler:
    """Event handler registration with enhanced metadata"""

    handler_id: str = field(default_factory=lambda: str(uuid4()))
    event_type: str = ""
    handler_func: Callable = None
    priority: int = 50
    async_handler: bool = True

    # Filtering
    source_filter: Optional[str] = None
    category_filter: Optional[EventCategory] = None
    tenant_filter: Optional[str] = None

    # Performance
    timeout_seconds: int = 30
    max_concurrent: int = 10

    # Reliability
    retry_on_failure: bool = True
    dead_letter_queue: bool = True

    # Monitoring
    execution_count: int = 0
    error_count: int = 0
    avg_execution_time: float = 0.0
    last_execution: Optional[datetime] = None

    def matches_event(self, event: CloudEvent) -> bool:
        """Check if handler matches event (CloudEvents v1.0 compliant)"""
        if self.event_type != "*" and self.event_type != event.type:
            return False

        if self.source_filter and self.source_filter not in event.source:
            return False

        if self.category_filter and self.category_filter.value != event.metiscategory:
            return False

        if self.tenant_filter and self.tenant_filter != event.metistenantid:
            return False

        return True


class CircuitBreakerState(str, Enum):
    """Circuit breaker states for failure resilience"""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""

    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: int = 60
    reset_timeout_seconds: int = 300


@dataclass
class TopicConfiguration:
    """Kafka topic configuration"""

    name: str = ""
    partitions: int = 3
    replication_factor: int = 2
    retention_hours: int = 168  # 1 week
    cleanup_policy: str = "delete"
    compression_type: str = "snappy"

    # Performance tuning
    batch_size: int = 16384
    linger_ms: int = 10
    max_request_size: int = 1048576  # 1MB

    # Custom configuration
    config: Dict[str, str] = field(default_factory=dict)


class EnhancedKafkaEventBus:
    """
    Production-ready Kafka event bus with enterprise features
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        client_id: str = "metis-event-bus",
        security_protocol: str = "PLAINTEXT",
        sasl_mechanism: Optional[str] = None,
        sasl_username: Optional[str] = None,
        sasl_password: Optional[str] = None,
        ssl_context: Optional[ssl.SSLContext] = None,
        topic_prefix: str = "metis",
    ):
        self.bootstrap_servers = bootstrap_servers
        self.client_id = client_id
        self.security_protocol = security_protocol
        self.sasl_mechanism = sasl_mechanism
        self.sasl_username = sasl_username
        self.sasl_password = sasl_password
        self.ssl_context = ssl_context
        self.topic_prefix = topic_prefix

        # Status and health
        self.status = EventBusStatus.INITIALIZING
        self.last_health_check = datetime.now(timezone.utc)
        self.health_check_interval = 30  # seconds

        # Kafka components
        self.producer: Optional[AIOKafkaProducer] = None
        self.consumers: Dict[str, AIOKafkaConsumer] = {}
        self.admin_client: Optional[AIOKafkaAdminClient] = None

        # Event handling
        self.event_handlers: Dict[str, List[EventHandler]] = defaultdict(list)
        self.global_handlers: List[EventHandler] = []

        # Circuit breakers per topic
        self.circuit_breakers: Dict[str, Dict] = {}
        self.circuit_config = CircuitBreakerConfig()

        # Dead letter queue
        self.dlq_enabled = True
        self.dlq_topic = f"{topic_prefix}_dead_letters"

        # Metrics and monitoring
        self.metrics = {
            "events_published": 0,
            "events_consumed": 0,
            "events_failed": 0,
            "handler_executions": 0,
            "handler_failures": 0,
            "avg_publish_time": 0.0,
            "avg_processing_time": 0.0,
            "circuit_breaker_trips": 0,
        }

        self.event_history: deque = deque(maxlen=1000)
        self.processing_times: deque = deque(maxlen=100)

        # Topic management
        self.topic_configs: Dict[str, TopicConfiguration] = {}
        self.auto_create_topics = True

        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        self.health_monitor_task: Optional[asyncio.Task] = None

        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> None:
        """Initialize Kafka event bus"""
        if not KAFKA_AVAILABLE:
            self.logger.warning("Kafka not available, using in-memory event bus")
            self.status = EventBusStatus.DEGRADED
            return

        try:
            self.logger.info("Initializing Kafka event bus...")

            # Create Kafka configuration
            kafka_config = self._create_kafka_config()

            # Initialize admin client
            self.admin_client = AIOKafkaAdminClient(**kafka_config)
            await self.admin_client.start()

            # Initialize producer
            producer_config = kafka_config.copy()
            producer_config.update(
                {
                    "compression_type": "snappy",
                    "batch_size": 16384,
                    "linger_ms": 10,
                    "acks": "all",
                    "retries": 3,
                    "enable_idempotence": True,
                    "max_in_flight_requests_per_connection": 5,
                }
            )

            self.producer = AIOKafkaProducer(**producer_config)
            await self.producer.start()

            # Create default topics
            await self._create_default_topics()

            # Start health monitoring
            self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())

            self.status = EventBusStatus.READY
            self.logger.info("Kafka event bus initialized successfully")

        except Exception as e:
            self.status = EventBusStatus.ERROR
            self.logger.error(f"Failed to initialize Kafka event bus: {str(e)}")
            raise

    def _create_kafka_config(self) -> Dict[str, Any]:
        """Create Kafka client configuration"""
        config = {
            "bootstrap_servers": self.bootstrap_servers,
            "client_id": self.client_id,
            "security_protocol": self.security_protocol,
        }

        # Add SASL configuration
        if self.sasl_mechanism:
            config.update(
                {
                    "sasl_mechanism": self.sasl_mechanism,
                    "sasl_plain_username": self.sasl_username,
                    "sasl_plain_password": self.sasl_password,
                }
            )

        # Add SSL configuration
        if self.ssl_context:
            config["ssl_context"] = self.ssl_context

        return config

    async def _create_default_topics(self) -> None:
        """Create default METIS topics"""
        default_topics = [
            TopicConfiguration(name=f"{self.topic_prefix}_cognitive", partitions=6),
            TopicConfiguration(name=f"{self.topic_prefix}_workflow", partitions=4),
            TopicConfiguration(name=f"{self.topic_prefix}_engagement", partitions=8),
            TopicConfiguration(name=f"{self.topic_prefix}_system", partitions=2),
            TopicConfiguration(name=f"{self.topic_prefix}_audit", partitions=3),
            TopicConfiguration(name=f"{self.topic_prefix}_analytics", partitions=4),
            TopicConfiguration(name=self.dlq_topic, partitions=2),
        ]

        for topic_config in default_topics:
            await self.create_topic(topic_config)

    async def create_topic(self, topic_config: TopicConfiguration) -> bool:
        """Create Kafka topic with configuration"""
        try:
            # Check if topic exists
            metadata = await self.admin_client.describe_topics([topic_config.name])
            if topic_config.name in metadata:
                self.logger.debug(f"Topic {topic_config.name} already exists")
                return True

            # Create topic
            new_topic = NewTopic(
                name=topic_config.name,
                num_partitions=topic_config.partitions,
                replication_factor=topic_config.replication_factor,
                config=topic_config.config,
            )

            await self.admin_client.create_topics([new_topic])
            self.topic_configs[topic_config.name] = topic_config

            self.logger.info(f"Created topic: {topic_config.name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create topic {topic_config.name}: {str(e)}")
            return False

    async def publish_event(
        self,
        event: CloudEvent,
        topic: Optional[str] = None,
        partition: Optional[int] = None,
        timeout: float = 30.0,
    ) -> bool:
        """Publish event to Kafka topic"""

        if self.status not in [EventBusStatus.READY, EventBusStatus.DEGRADED]:
            self.logger.warning("Event bus not ready, dropping event")
            return False

        # Determine topic
        if topic is None:
            topic = self._get_topic_for_event(event)

        # Check circuit breaker
        if not self._check_circuit_breaker(topic):
            self.logger.warning(f"Circuit breaker open for topic {topic}")
            await self._send_to_dlq(event, f"Circuit breaker open for {topic}")
            return False

        start_time = time.time()

        try:
            # Serialize event
            event_data = json.dumps(event.to_dict(), default=str).encode("utf-8")

            # Create message key for partitioning
            message_key = self._create_message_key(event)

            # Publish to Kafka
            if self.producer:
                future = await self.producer.send_and_wait(
                    topic=topic,
                    value=event_data,
                    key=message_key,
                    partition=partition,
                    timeout=timeout,
                )

                # Update metrics
                execution_time = time.time() - start_time
                self.processing_times.append(execution_time)
                self.metrics["events_published"] += 1
                self.metrics["avg_publish_time"] = sum(self.processing_times) / len(
                    self.processing_times
                )

                # Record successful circuit breaker operation
                self._record_circuit_breaker_success(topic)

                # Add to event history
                self.event_history.append(
                    {
                        "event_id": event.id,
                        "type": event.type,
                        "topic": topic,
                        "timestamp": datetime.utcnow(),
                        "status": "published",
                        "execution_time": execution_time,
                    }
                )

                self.logger.debug(f"Published event {event.id} to topic {topic}")
                return True
            else:
                # Fallback for in-memory mode
                await self._process_event_in_memory(event)
                return True

        except Exception as e:
            # Record circuit breaker failure
            self._record_circuit_breaker_failure(topic)

            # Update metrics
            self.metrics["events_failed"] += 1

            # Log error
            self.logger.error(f"Failed to publish event {event.id}: {str(e)}")

            # Send to DLQ if retries exhausted
            if event.retry_count >= event.max_retries:
                await self._send_to_dlq(event, str(e))
            else:
                # Retry event
                event.retry_count += 1
                await asyncio.sleep(
                    min(2**event.retry_count, 30)
                )  # Exponential backoff
                return await self.publish_event(event, topic, partition, timeout)

            return False

    def _get_topic_for_event(self, event: CloudEvent) -> str:
        """Determine topic based on event category (CloudEvents v1.0 compliant)"""
        category_topic_map = {
            "cognitive": f"{self.topic_prefix}_cognitive",
            "workflow": f"{self.topic_prefix}_workflow",
            "engagement": f"{self.topic_prefix}_engagement",
            "system": f"{self.topic_prefix}_system",
            "audit": f"{self.topic_prefix}_audit",
            "analytics": f"{self.topic_prefix}_analytics",
        }

        return category_topic_map.get(
            event.metiscategory, f"{self.topic_prefix}_system"
        )

    def _create_message_key(self, event: CloudEvent) -> bytes:
        """Create message key for partitioning (CloudEvents v1.0 compliant)"""
        # Use engagement_id for related events to go to same partition
        if event.metisengagementid:
            return event.metisengagementid.encode("utf-8")

        # Use tenant_id for tenant isolation
        if event.metistenantid:
            return event.metistenantid.encode("utf-8")

        # Use correlation_id for correlated events
        if event.metiscorrelationid:
            return event.metiscorrelationid.encode("utf-8")

        # Default to event source
        return event.source.encode("utf-8")

    async def subscribe_to_events(
        self,
        handler: EventHandler,
        topics: Optional[List[str]] = None,
        consumer_group: str = "metis_default",
    ) -> str:
        """Subscribe to events with handler"""

        handler_id = handler.handler_id

        # Register handler
        if handler.event_type == "*":
            self.global_handlers.append(handler)
        else:
            self.event_handlers[handler.event_type].append(handler)

        # Create consumer if not exists
        if consumer_group not in self.consumers and self.status == EventBusStatus.READY:
            await self._create_consumer(consumer_group, topics or [])

        self.logger.info(
            f"Registered event handler {handler_id} for type {handler.event_type}"
        )
        return handler_id

    async def _create_consumer(self, consumer_group: str, topics: List[str]) -> None:
        """Create Kafka consumer"""

        if not KAFKA_AVAILABLE:
            return

        try:
            # Create consumer configuration
            consumer_config = self._create_kafka_config()
            consumer_config.update(
                {
                    "group_id": consumer_group,
                    "auto_offset_reset": "latest",
                    "enable_auto_commit": False,
                    "max_poll_records": 500,
                    "session_timeout_ms": 30000,
                    "heartbeat_interval_ms": 10000,
                }
            )

            # Create consumer
            consumer = AIOKafkaConsumer(**consumer_config)

            # Subscribe to topics
            if not topics:
                topics = [
                    f"{self.topic_prefix}_cognitive",
                    f"{self.topic_prefix}_workflow",
                    f"{self.topic_prefix}_engagement",
                    f"{self.topic_prefix}_system",
                    f"{self.topic_prefix}_audit",
                    f"{self.topic_prefix}_analytics",
                ]

            consumer.subscribe(topics)
            await consumer.start()

            self.consumers[consumer_group] = consumer

            # Start consumer task
            task = asyncio.create_task(self._consumer_loop(consumer, consumer_group))
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)

            self.logger.info(f"Created consumer for group {consumer_group}")

        except Exception as e:
            self.logger.error(f"Failed to create consumer {consumer_group}: {str(e)}")

    async def _consumer_loop(
        self, consumer: AIOKafkaConsumer, consumer_group: str
    ) -> None:
        """Main consumer loop for processing events"""

        try:
            async for message in consumer:
                try:
                    # Parse event
                    event_data = json.loads(message.value.decode("utf-8"))
                    event = CloudEvent.from_dict(event_data)

                    # Process event
                    await self._process_event(event, message)

                    # Commit offset
                    await consumer.commit()

                    # Update metrics
                    self.metrics["events_consumed"] += 1

                except Exception as e:
                    self.logger.error(f"Error processing message: {str(e)}")
                    # Could implement retry logic here

        except Exception as e:
            self.logger.error(f"Consumer loop error for {consumer_group}: {str(e)}")
        finally:
            await consumer.stop()

    async def _process_event(self, event: CloudEvent, message=None) -> None:
        """Process event with registered handlers"""

        start_time = time.time()
        handlers_executed = 0

        try:
            # Find matching handlers
            matching_handlers = []

            # Check specific event type handlers
            for handler in self.event_handlers.get(event.type, []):
                if handler.matches_event(event):
                    matching_handlers.append(handler)

            # Check global handlers
            for handler in self.global_handlers:
                if handler.matches_event(event):
                    matching_handlers.append(handler)

            # Execute handlers
            for handler in matching_handlers:
                try:
                    await self._execute_handler(handler, event)
                    handlers_executed += 1

                except Exception as e:
                    handler.error_count += 1
                    self.metrics["handler_failures"] += 1
                    self.logger.error(
                        f"Handler {handler.handler_id} failed for event {event.id}: {str(e)}"
                    )

                    # Send to DLQ if handler consistently fails
                    if handler.dead_letter_queue and handler.error_count > 5:
                        await self._send_to_dlq(
                            event, f"Handler {handler.handler_id} consistently failing"
                        )

            # Update processing metrics
            execution_time = time.time() - start_time
            self.processing_times.append(execution_time)
            self.metrics["avg_processing_time"] = sum(self.processing_times) / len(
                self.processing_times
            )

            # Add to event history
            self.event_history.append(
                {
                    "event_id": event.id,
                    "type": event.type,
                    "timestamp": datetime.utcnow(),
                    "status": "processed",
                    "handlers_executed": handlers_executed,
                    "execution_time": execution_time,
                }
            )

        except Exception as e:
            self.logger.error(f"Failed to process event {event.id}: {str(e)}")

    async def _execute_handler(self, handler: EventHandler, event: CloudEvent) -> None:
        """Execute individual event handler"""

        start_time = time.time()

        try:
            # Apply timeout
            if handler.async_handler:
                await asyncio.wait_for(
                    handler.handler_func(event), timeout=handler.timeout_seconds
                )
            else:
                # Run sync handler in thread pool
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, handler.handler_func, event)

            # Update handler metrics
            execution_time = time.time() - start_time
            handler.execution_count += 1
            handler.avg_execution_time = (
                handler.avg_execution_time * (handler.execution_count - 1)
                + execution_time
            ) / handler.execution_count
            handler.last_execution = datetime.utcnow()

            self.metrics["handler_executions"] += 1

        except asyncio.TimeoutError:
            self.logger.warning(f"Handler {handler.handler_id} timed out")
            raise
        except Exception as e:
            self.logger.error(
                f"Handler {handler.handler_id} execution failed: {str(e)}"
            )
            raise

    async def _process_event_in_memory(self, event: CloudEvent) -> None:
        """Process event in memory mode (fallback when Kafka unavailable)"""
        self.logger.debug(f"Processing event {event.id} in memory")
        await self._process_event(event)

    def _check_circuit_breaker(self, topic: str) -> bool:
        """Check circuit breaker state for topic"""
        if topic not in self.circuit_breakers:
            self.circuit_breakers[topic] = {
                "state": CircuitBreakerState.CLOSED,
                "failure_count": 0,
                "success_count": 0,
                "last_failure": None,
                "last_state_change": datetime.utcnow(),
            }

        breaker = self.circuit_breakers[topic]
        now = datetime.utcnow()

        if breaker["state"] == CircuitBreakerState.OPEN:
            # Check if timeout has passed
            if (
                now - breaker["last_state_change"]
            ).total_seconds() > self.circuit_config.timeout_seconds:
                breaker["state"] = CircuitBreakerState.HALF_OPEN
                breaker["success_count"] = 0
                breaker["last_state_change"] = now
                self.logger.info(f"Circuit breaker for {topic} moved to HALF_OPEN")
            else:
                return False

        return True

    def _record_circuit_breaker_success(self, topic: str) -> None:
        """Record successful operation for circuit breaker"""
        if topic not in self.circuit_breakers:
            return

        breaker = self.circuit_breakers[topic]

        if breaker["state"] == CircuitBreakerState.HALF_OPEN:
            breaker["success_count"] += 1
            if breaker["success_count"] >= self.circuit_config.success_threshold:
                breaker["state"] = CircuitBreakerState.CLOSED
                breaker["failure_count"] = 0
                breaker["last_state_change"] = datetime.utcnow()
                self.logger.info(f"Circuit breaker for {topic} closed")
        elif breaker["state"] == CircuitBreakerState.CLOSED:
            breaker["failure_count"] = max(0, breaker["failure_count"] - 1)

    def _record_circuit_breaker_failure(self, topic: str) -> None:
        """Record failed operation for circuit breaker"""
        if topic not in self.circuit_breakers:
            self._check_circuit_breaker(topic)  # Initialize

        breaker = self.circuit_breakers[topic]
        breaker["failure_count"] += 1
        breaker["last_failure"] = datetime.utcnow()

        if breaker["failure_count"] >= self.circuit_config.failure_threshold:
            if breaker["state"] != CircuitBreakerState.OPEN:
                breaker["state"] = CircuitBreakerState.OPEN
                breaker["last_state_change"] = datetime.utcnow()
                self.metrics["circuit_breaker_trips"] += 1
                self.logger.warning(f"Circuit breaker for {topic} opened")

    async def _send_to_dlq(self, event: CloudEvent, error_message: str) -> None:
        """Send event to dead letter queue"""
        if not self.dlq_enabled:
            return

        try:
            dlq_event = CloudEvent(
                type="deadletter.event",
                source="event_bus/dlq",
                category=EventCategory.SYSTEM,
                priority=EventPriority.LOW,
                data={
                    "original_event": event.to_dict(),
                    "error_message": error_message,
                    "failed_at": datetime.utcnow().isoformat(),
                },
            )

            if self.producer:
                dlq_data = json.dumps(dlq_event.to_dict(), default=str).encode("utf-8")
                await self.producer.send_and_wait(
                    topic=self.dlq_topic, value=dlq_data, key=event.id.encode("utf-8")
                )

            self.logger.info(f"Sent event {event.id} to dead letter queue")

        except Exception as e:
            self.logger.error(f"Failed to send event to DLQ: {str(e)}")

    async def _health_monitor_loop(self) -> None:
        """Background health monitoring loop"""
        while self.status != EventBusStatus.DISCONNECTED:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitor error: {str(e)}")

    async def _perform_health_check(self) -> None:
        """Perform health check on Kafka connections"""
        try:
            if self.producer and KAFKA_AVAILABLE:
                # Simple health check - get cluster metadata
                metadata = await self.producer.client.check_version()
                if metadata:
                    self.last_health_check = datetime.utcnow()
                    if self.status == EventBusStatus.ERROR:
                        self.status = EventBusStatus.READY
                else:
                    if self.status == EventBusStatus.READY:
                        self.status = EventBusStatus.DEGRADED

        except Exception as e:
            self.logger.warning(f"Health check failed: {str(e)}")
            if self.status in [EventBusStatus.READY, EventBusStatus.DEGRADED]:
                self.status = EventBusStatus.ERROR

    async def shutdown(self) -> None:
        """Gracefully shutdown event bus"""
        self.logger.info("Shutting down event bus...")
        self.status = EventBusStatus.DISCONNECTED

        # Cancel background tasks
        if self.health_monitor_task:
            self.health_monitor_task.cancel()

        for task in self.background_tasks:
            task.cancel()

        # Stop consumers
        for consumer in self.consumers.values():
            await consumer.stop()

        # Stop producer
        if self.producer:
            await self.producer.stop()

        # Stop admin client
        if self.admin_client:
            await self.admin_client.close()

        self.logger.info("Event bus shutdown complete")

    def get_metrics(self) -> Dict[str, Any]:
        """Get event bus metrics and statistics"""
        return {
            "status": self.status.value,
            "last_health_check": self.last_health_check.isoformat(),
            "metrics": self.metrics.copy(),
            "circuit_breakers": {
                topic: {
                    "state": (
                        cb["state"].value
                        if isinstance(cb["state"], CircuitBreakerState)
                        else cb["state"]
                    ),
                    "failure_count": cb["failure_count"],
                    "last_failure": (
                        cb["last_failure"].isoformat() if cb["last_failure"] else None
                    ),
                }
                for topic, cb in self.circuit_breakers.items()
            },
            "handlers": {
                "total_handlers": len(self.global_handlers)
                + sum(len(handlers) for handlers in self.event_handlers.values()),
                "global_handlers": len(self.global_handlers),
                "type_specific_handlers": len(self.event_handlers),
            },
            "consumers": list(self.consumers.keys()),
            "recent_events": list(self.event_history)[-10:],  # Last 10 events
        }


# Global event bus instance
_global_event_bus: Optional[EnhancedKafkaEventBus] = None


async def get_event_bus() -> EnhancedKafkaEventBus:
    """Get global event bus instance"""
    global _global_event_bus

    if _global_event_bus is None:
        # Create with configuration from environment
        bootstrap_servers = os.getenv("KAFKA_BROKERS", "localhost:9092")

        _global_event_bus = EnhancedKafkaEventBus(
            bootstrap_servers=bootstrap_servers,
            client_id=os.getenv("KAFKA_CLIENT_ID", "metis-event-bus"),
            topic_prefix=os.getenv("KAFKA_TOPIC_PREFIX", "metis"),
        )

        await _global_event_bus.initialize()

    return _global_event_bus


async def publish_event(event: CloudEvent, topic: Optional[str] = None) -> bool:
    """Convenience function to publish event"""
    event_bus = await get_event_bus()
    return await event_bus.publish_event(event, topic)


async def subscribe_to_events(
    event_type: str, handler_func: Callable, priority: int = 50, **kwargs
) -> str:
    """Convenience function to subscribe to events"""
    event_bus = await get_event_bus()

    handler = EventHandler(
        event_type=event_type, handler_func=handler_func, priority=priority, **kwargs
    )

    return await event_bus.subscribe_to_events(handler)


# Event router integration factory (P4.4)
def create_metis_event_bus_with_router(event_router=None, **kwargs):
    """
    Create METIS Event Bus with optional Event Router integration
    This provides the P4.4 routing capabilities while maintaining compatibility
    """
    bus = EnhancedKafkaEventBus(**kwargs)

    # Store router reference for integration
    bus._event_router = event_router

    # Override publish_event method to include routing
    original_publish_event = bus.publish_event

    async def publish_event_with_routing(event: CloudEvent) -> bool:
        """Enhanced publish_event with router integration"""
        try:
            # Route event through router first (P4.4 integration)
            if hasattr(bus, "_event_router") and bus._event_router:
                routing_result = await bus._event_router.route_event(event)
                bus.logger.debug(
                    f"Event {event.id} routed: {len(routing_result['routes_matched'])} routes, {len(routing_result['subscriptions_matched'])} subscriptions"
                )

            # Continue with original publish
            return await original_publish_event(event)

        except Exception as e:
            bus.logger.error(f"Router integration error: {str(e)}")
            # Fallback to original publish on router error
            return await original_publish_event(event)

    # Replace the method
    bus.publish_event = publish_event_with_routing

    return bus


# Export alias for compatibility
MetisEventBus = EnhancedKafkaEventBus

# Export all public classes
__all__ = [
    "CloudEvent",
    "EnhancedKafkaEventBus",
    "MetisEventBus",
    "EventBusStatus",
    "create_metis_event_bus_with_router",
    "create_metis_cloud_event",
]
