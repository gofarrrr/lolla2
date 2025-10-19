"""
METIS Event Router - P4.4
Advanced event routing and subscription management for CloudEvents v1.0

Provides pattern-based routing, subscription management, and event filtering
for the METIS cognitive platform.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any, Set, Pattern
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import re
from uuid import uuid4
from collections import defaultdict

from src.core.enhanced_event_bus import CloudEvent, MetisEventBus
from src.schemas.event_schemas import MetisEventCategory, MetisEventPriority


class SubscriptionStatus(str, Enum):
    """Subscription status enumeration"""

    ACTIVE = "active"
    PAUSED = "paused"
    EXPIRED = "expired"
    FAILED = "failed"


class RouteAction(str, Enum):
    """Route action types"""

    DELIVER = "deliver"  # Deliver to subscriber
    TRANSFORM = "transform"  # Transform and deliver
    FILTER = "filter"  # Apply filtering
    DUPLICATE = "duplicate"  # Duplicate to multiple targets
    ARCHIVE = "archive"  # Archive event
    DISCARD = "discard"  # Discard event


@dataclass
class EventFilter:
    """Event filtering criteria"""

    event_types: Optional[List[str]] = None
    categories: Optional[List[MetisEventCategory]] = None
    priorities: Optional[List[MetisEventPriority]] = None
    sources: Optional[List[str]] = None
    engagement_ids: Optional[List[str]] = None
    tenant_ids: Optional[List[str]] = None

    # Pattern-based filters
    subject_patterns: Optional[List[Pattern]] = None
    data_filters: Optional[Dict[str, Any]] = None

    # Time-based filters
    time_window: Optional[timedelta] = None
    max_events_per_minute: Optional[int] = None

    def matches(self, event: CloudEvent) -> bool:
        """Check if event matches filter criteria"""
        # Type filtering
        if self.event_types and event.type not in self.event_types:
            return False

        # Category filtering
        if self.categories and hasattr(event, "metiscategory"):
            if event.metiscategory not in [c.value for c in self.categories]:
                return False

        # Priority filtering
        if self.priorities and hasattr(event, "metispriority"):
            if event.metispriority not in [p.value for p in self.priorities]:
                return False

        # Source filtering
        if self.sources and event.source not in self.sources:
            return False

        # Engagement ID filtering
        if self.engagement_ids and hasattr(event, "metisengagementid"):
            if event.metisengagementid not in self.engagement_ids:
                return False

        # Tenant ID filtering
        if self.tenant_ids and hasattr(event, "metistenantid"):
            if event.metistenantid not in self.tenant_ids:
                return False

        # Subject pattern filtering
        if self.subject_patterns and event.subject:
            pattern_match = any(
                pattern.match(event.subject) for pattern in self.subject_patterns
            )
            if not pattern_match:
                return False

        # Data filtering (basic key-value matching)
        if self.data_filters and event.data:
            for key, expected_value in self.data_filters.items():
                if key not in event.data or event.data[key] != expected_value:
                    return False

        return True


@dataclass
class RouteTransformation:
    """Event transformation specification"""

    transform_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    async def apply(self, event: CloudEvent) -> CloudEvent:
        """Apply transformation to event"""
        if self.transform_type == "add_metadata":
            # Add metadata to event
            if hasattr(event, "data") and event.data:
                event.data.update(self.parameters.get("metadata", {}))

        elif self.transform_type == "change_priority":
            # Change event priority
            if hasattr(event, "metispriority"):
                event.metispriority = self.parameters.get("new_priority", "medium")

        elif self.transform_type == "add_correlation":
            # Add correlation ID
            if hasattr(event, "metiscorrelationid"):
                event.metiscorrelationid = str(
                    self.parameters.get("correlation_id", uuid4())
                )

        elif self.transform_type == "custom":
            # Custom transformation function
            transform_func = self.parameters.get("function")
            if transform_func and callable(transform_func):
                event = await transform_func(event)

        return event


@dataclass
class EventRoute:
    """Event routing specification"""

    route_id: str
    name: str
    filter_criteria: EventFilter
    action: RouteAction
    target_handler: Optional[Callable] = None
    transformation: Optional[RouteTransformation] = None
    priority: int = 100  # Lower number = higher priority
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Performance tracking
    events_processed: int = 0
    last_processed: Optional[datetime] = None
    processing_errors: int = 0


@dataclass
class EventSubscription:
    """Event subscription with delivery guarantees"""

    subscription_id: str
    subscriber_id: str
    name: str
    filter_criteria: EventFilter
    delivery_handler: Callable

    # Delivery options
    max_retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    delivery_timeout_seconds: float = 30.0
    batch_size: int = 1  # Events per batch
    batch_timeout_seconds: float = 5.0

    # Status and tracking
    status: SubscriptionStatus = SubscriptionStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: Optional[datetime] = None

    # Performance metrics
    events_delivered: int = 0
    delivery_failures: int = 0
    total_retry_attempts: int = 0
    avg_delivery_time: float = 0.0

    # Expiration
    expires_at: Optional[datetime] = None

    def is_expired(self) -> bool:
        """Check if subscription has expired"""
        return self.expires_at and datetime.utcnow() > self.expires_at

    def is_active(self) -> bool:
        """Check if subscription is active"""
        return self.status == SubscriptionStatus.ACTIVE and not self.is_expired()


class EventRouter:
    """
    Advanced event router for METIS CloudEvents
    Provides pattern-based routing, subscription management, and delivery guarantees
    """

    def __init__(self, event_bus: Optional[MetisEventBus] = None):
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)

        # Routing and subscription storage
        self.routes: Dict[str, EventRoute] = {}
        self.subscriptions: Dict[str, EventSubscription] = {}

        # Performance tracking
        self.events_routed: int = 0
        self.routing_errors: int = 0
        self.start_time = datetime.utcnow()

        # Batching and delivery
        self.pending_deliveries: Dict[str, List[CloudEvent]] = defaultdict(list)
        self.delivery_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

        # Background tasks
        self._running = False
        self._background_tasks: Set[asyncio.Task] = set()

    async def start(self):
        """Start the event router background processes"""
        if self._running:
            return

        self._running = True

        # Start background tasks
        batch_task = asyncio.create_task(self._batch_delivery_processor())
        cleanup_task = asyncio.create_task(self._subscription_cleanup())

        self._background_tasks.add(batch_task)
        self._background_tasks.add(cleanup_task)

        # Clean up completed tasks
        for task in self._background_tasks.copy():
            if task.done():
                self._background_tasks.remove(task)

        self.logger.info("Event router started")

    async def stop(self):
        """Stop the event router and cleanup"""
        self._running = False

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        self._background_tasks.clear()
        self.logger.info("Event router stopped")

    def add_route(self, route: EventRoute) -> str:
        """Add event route"""
        self.routes[route.route_id] = route
        self.logger.info(f"Added route: {route.name} ({route.route_id})")
        return route.route_id

    def remove_route(self, route_id: str) -> bool:
        """Remove event route"""
        if route_id in self.routes:
            route = self.routes.pop(route_id)
            self.logger.info(f"Removed route: {route.name} ({route_id})")
            return True
        return False

    def add_subscription(self, subscription: EventSubscription) -> str:
        """Add event subscription"""
        self.subscriptions[subscription.subscription_id] = subscription
        self.logger.info(
            f"Added subscription: {subscription.name} ({subscription.subscription_id})"
        )
        return subscription.subscription_id

    def remove_subscription(self, subscription_id: str) -> bool:
        """Remove event subscription"""
        if subscription_id in self.subscriptions:
            subscription = self.subscriptions.pop(subscription_id)
            self.logger.info(
                f"Removed subscription: {subscription.name} ({subscription_id})"
            )
            return True
        return False

    def pause_subscription(self, subscription_id: str) -> bool:
        """Pause event subscription"""
        if subscription_id in self.subscriptions:
            self.subscriptions[subscription_id].status = SubscriptionStatus.PAUSED
            self.logger.info(f"Paused subscription: {subscription_id}")
            return True
        return False

    def resume_subscription(self, subscription_id: str) -> bool:
        """Resume event subscription"""
        if subscription_id in self.subscriptions:
            subscription = self.subscriptions[subscription_id]
            if not subscription.is_expired():
                subscription.status = SubscriptionStatus.ACTIVE
                self.logger.info(f"Resumed subscription: {subscription_id}")
                return True
            else:
                subscription.status = SubscriptionStatus.EXPIRED
                self.logger.warning(
                    f"Cannot resume expired subscription: {subscription_id}"
                )
        return False

    async def route_event(self, event: CloudEvent) -> Dict[str, Any]:
        """Route event through all matching routes and subscriptions"""
        routing_results = {
            "event_id": event.id,
            "routes_matched": [],
            "subscriptions_matched": [],
            "actions_taken": [],
            "errors": [],
        }

        try:
            self.events_routed += 1

            # Process routes (priority order)
            sorted_routes = sorted(
                [
                    (route_id, route)
                    for route_id, route in self.routes.items()
                    if route.enabled
                ],
                key=lambda x: x[1].priority,
            )

            for route_id, route in sorted_routes:
                if route.filter_criteria.matches(event):
                    routing_results["routes_matched"].append(route_id)

                    try:
                        await self._process_route(event, route)
                        routing_results["actions_taken"].append(
                            {
                                "route_id": route_id,
                                "action": route.action.value,
                                "status": "success",
                            }
                        )

                        # Update route metrics
                        route.events_processed += 1
                        route.last_processed = datetime.utcnow()

                    except Exception as e:
                        route.processing_errors += 1
                        error_msg = f"Route {route_id} processing failed: {str(e)}"
                        routing_results["errors"].append(error_msg)
                        self.logger.error(error_msg)

            # Process subscriptions
            for subscription_id, subscription in self.subscriptions.items():
                if subscription.is_active() and subscription.filter_criteria.matches(
                    event
                ):
                    routing_results["subscriptions_matched"].append(subscription_id)

                    try:
                        await self._process_subscription(event, subscription)
                        routing_results["actions_taken"].append(
                            {
                                "subscription_id": subscription_id,
                                "action": "deliver",
                                "status": "queued",
                            }
                        )

                    except Exception as e:
                        subscription.delivery_failures += 1
                        error_msg = f"Subscription {subscription_id} processing failed: {str(e)}"
                        routing_results["errors"].append(error_msg)
                        self.logger.error(error_msg)

        except Exception as e:
            self.routing_errors += 1
            error_msg = f"Event routing failed: {str(e)}"
            routing_results["errors"].append(error_msg)
            self.logger.error(error_msg)

        return routing_results

    async def _process_route(self, event: CloudEvent, route: EventRoute):
        """Process event through route"""
        processed_event = event

        # Apply transformation if specified
        if route.transformation:
            processed_event = await route.transformation.apply(event)

        # Execute route action
        if route.action == RouteAction.DELIVER and route.target_handler:
            # Route handlers receive single events, subscription handlers receive lists
            await route.target_handler([processed_event])

        elif route.action == RouteAction.TRANSFORM:
            # Transformation already applied, could re-publish to event bus
            if self.event_bus:
                await self.event_bus.publish_event(processed_event)

        elif route.action == RouteAction.DUPLICATE and route.target_handler:
            # Create duplicate events for multiple targets
            targets = (
                route.transformation.parameters.get("targets", [])
                if route.transformation
                else []
            )
            duplicated_events = []
            for target in targets:
                duplicate_event = CloudEvent(
                    type=processed_event.type,
                    source=processed_event.source,
                    data=processed_event.data.copy() if processed_event.data else None,
                    subject=f"duplicate-{target}",
                )
                duplicated_events.append(duplicate_event)

            if duplicated_events:
                await route.target_handler(duplicated_events)

        elif route.action == RouteAction.ARCHIVE:
            # Archive event (could save to database, file, etc.)
            self.logger.info(f"Archived event: {event.id}")

        elif route.action == RouteAction.DISCARD:
            # Simply discard the event
            self.logger.debug(f"Discarded event: {event.id}")

    async def _process_subscription(
        self, event: CloudEvent, subscription: EventSubscription
    ):
        """Process event for subscription (batching support)"""
        subscription.last_activity = datetime.utcnow()

        # Add to pending deliveries
        self.pending_deliveries[subscription.subscription_id].append(event)

        # If batch is full or single event delivery, trigger immediate delivery
        if (
            len(self.pending_deliveries[subscription.subscription_id])
            >= subscription.batch_size
            or subscription.batch_size == 1
        ):
            await self._deliver_batch(subscription.subscription_id)

    async def _deliver_batch(self, subscription_id: str):
        """Deliver batched events to subscription"""
        if subscription_id not in self.subscriptions:
            return

        subscription = self.subscriptions[subscription_id]
        pending_events = self.pending_deliveries[subscription_id]

        if not pending_events:
            return

        # Get delivery lock for this subscription
        async with self.delivery_locks[subscription_id]:
            # Re-check events (might have been processed by another task)
            events_to_deliver = self.pending_deliveries[subscription_id][
                : subscription.batch_size
            ]

            if not events_to_deliver:
                return

            # Remove events from pending
            self.pending_deliveries[subscription_id] = self.pending_deliveries[
                subscription_id
            ][len(events_to_deliver) :]

            # Attempt delivery with retry
            delivery_start = datetime.utcnow()

            for attempt in range(subscription.max_retry_attempts + 1):
                try:
                    # Delivery with timeout
                    await asyncio.wait_for(
                        subscription.delivery_handler(events_to_deliver),
                        timeout=subscription.delivery_timeout_seconds,
                    )

                    # Success - update metrics
                    delivery_time = (datetime.utcnow() - delivery_start).total_seconds()
                    subscription.events_delivered += len(events_to_deliver)
                    subscription.avg_delivery_time = (
                        subscription.avg_delivery_time
                        * (subscription.events_delivered - len(events_to_deliver))
                        + delivery_time * len(events_to_deliver)
                    ) / subscription.events_delivered

                    break

                except Exception as e:
                    subscription.total_retry_attempts += 1

                    if attempt < subscription.max_retry_attempts:
                        # Retry with delay
                        await asyncio.sleep(
                            subscription.retry_delay_seconds * (2**attempt)
                        )
                        continue
                    else:
                        # Final failure
                        subscription.delivery_failures += len(events_to_deliver)
                        self.logger.error(
                            f"Final delivery failure for subscription {subscription_id}: {str(e)}"
                        )

                        # Mark subscription as failed if too many errors
                        error_rate = subscription.delivery_failures / max(
                            subscription.events_delivered
                            + subscription.delivery_failures,
                            1,
                        )
                        if (
                            error_rate > 0.5
                            and subscription.events_delivered
                            + subscription.delivery_failures
                            > 10
                        ):
                            subscription.status = SubscriptionStatus.FAILED
                            self.logger.warning(
                                f"Marking subscription {subscription_id} as failed due to high error rate"
                            )

    async def _batch_delivery_processor(self):
        """Background task to process batched deliveries"""
        while self._running:
            try:
                for subscription_id, subscription in self.subscriptions.items():
                    if (
                        subscription.is_active()
                        and self.pending_deliveries[subscription_id]
                        and subscription.batch_size > 1
                    ):

                        # Check if batch timeout has elapsed
                        if subscription.last_activity:
                            time_since_last = (
                                datetime.utcnow() - subscription.last_activity
                            )
                            if (
                                time_since_last.total_seconds()
                                >= subscription.batch_timeout_seconds
                            ):
                                await self._deliver_batch(subscription_id)

                await asyncio.sleep(1.0)  # Check every second

            except Exception as e:
                self.logger.error(f"Batch delivery processor error: {str(e)}")
                await asyncio.sleep(5.0)

    async def _subscription_cleanup(self):
        """Background task to clean up expired subscriptions"""
        while self._running:
            try:
                expired_subscriptions = []

                for subscription_id, subscription in self.subscriptions.items():
                    if subscription.is_expired():
                        expired_subscriptions.append(subscription_id)

                for subscription_id in expired_subscriptions:
                    subscription = self.subscriptions[subscription_id]
                    subscription.status = SubscriptionStatus.EXPIRED
                    self.logger.info(
                        f"Subscription expired: {subscription.name} ({subscription_id})"
                    )

                await asyncio.sleep(60.0)  # Check every minute

            except Exception as e:
                self.logger.error(f"Subscription cleanup error: {str(e)}")
                await asyncio.sleep(300.0)

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing performance statistics"""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()

        route_stats = {}
        for route_id, route in self.routes.items():
            route_stats[route_id] = {
                "name": route.name,
                "events_processed": route.events_processed,
                "processing_errors": route.processing_errors,
                "error_rate": route.processing_errors / max(route.events_processed, 1),
                "enabled": route.enabled,
                "last_processed": (
                    route.last_processed.isoformat() if route.last_processed else None
                ),
            }

        subscription_stats = {}
        for sub_id, subscription in self.subscriptions.items():
            subscription_stats[sub_id] = {
                "name": subscription.name,
                "status": subscription.status.value,
                "events_delivered": subscription.events_delivered,
                "delivery_failures": subscription.delivery_failures,
                "total_retry_attempts": subscription.total_retry_attempts,
                "avg_delivery_time": subscription.avg_delivery_time,
                "pending_events": len(self.pending_deliveries[sub_id]),
                "is_expired": subscription.is_expired(),
            }

        return {
            "uptime_seconds": uptime,
            "events_routed": self.events_routed,
            "routing_errors": self.routing_errors,
            "error_rate": self.routing_errors / max(self.events_routed, 1),
            "active_routes": len([r for r in self.routes.values() if r.enabled]),
            "active_subscriptions": len(
                [s for s in self.subscriptions.values() if s.is_active()]
            ),
            "routes": route_stats,
            "subscriptions": subscription_stats,
        }


# Convenience functions for common patterns
def create_simple_filter(
    event_types: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    priorities: Optional[List[str]] = None,
    engagement_id: Optional[str] = None,
) -> EventFilter:
    """Create simple event filter"""
    category_enums = [MetisEventCategory(c) for c in categories] if categories else None
    priority_enums = [MetisEventPriority(p) for p in priorities] if priorities else None
    engagement_ids = [engagement_id] if engagement_id else None

    return EventFilter(
        event_types=event_types,
        categories=category_enums,
        priorities=priority_enums,
        engagement_ids=engagement_ids,
    )


def create_pattern_filter(subject_patterns: List[str], **kwargs) -> EventFilter:
    """Create pattern-based event filter"""
    compiled_patterns = [re.compile(pattern) for pattern in subject_patterns]
    filter_obj = create_simple_filter(**kwargs)
    filter_obj.subject_patterns = compiled_patterns
    return filter_obj


async def simple_event_handler(events: List[CloudEvent]):
    """Simple event handler for testing"""
    print(f"Received {len(events)} events:")
    for event in events:
        print(f"  - {event.type}: {event.id}")


# Export main classes
__all__ = [
    "EventRouter",
    "EventRoute",
    "EventSubscription",
    "EventFilter",
    "RouteTransformation",
    "RouteAction",
    "SubscriptionStatus",
    "create_simple_filter",
    "create_pattern_filter",
    "simple_event_handler",
]
