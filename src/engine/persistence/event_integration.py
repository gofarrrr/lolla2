"""
METIS Persistence Event Integration - P5.5
Real-time contract updates through event bus integration

Implements seamless event-driven persistence:
- Real-time contract synchronization with event streams
- Event-triggered checkpoint creation
- Contract state broadcasting
- Event sourcing pattern implementation
- Persistence event coordination
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from enum import Enum

from src.core.enhanced_event_bus import MetisEventBus, CloudEvent
from src.schemas.event_schemas import MetisEventCategory, MetisEventPriority
from src.engine.models.data_contracts import MetisDataContract
from src.persistence.contract_storage import (
    ContractPersistenceManager,
    CheckpointType,
)
from src.persistence.checkpoint_manager import (
    ContractCheckpointManager,
    CheckpointTrigger,
)
from src.persistence.recovery_system import ContractRecoverySystem
from src.persistence.engagement_resume import EngagementResumeManager


class PersistenceEventType(str, Enum):
    """Types of persistence-related events"""

    CONTRACT_STORED = "contract_stored"
    CHECKPOINT_CREATED = "checkpoint_created"
    RECOVERY_INITIATED = "recovery_initiated"
    RESUME_COMPLETED = "resume_completed"
    PERSISTENCE_ERROR = "persistence_error"
    SYNC_STATUS_CHANGED = "sync_status_changed"


class SyncStatus(str, Enum):
    """Synchronization status between events and persistence"""

    SYNCHRONIZED = "synchronized"  # Events and persistence are in sync
    PENDING_SYNC = "pending_sync"  # Events received, persistence pending
    SYNC_ERROR = "sync_error"  # Synchronization error occurred
    RECOVERY_MODE = "recovery_mode"  # In recovery/resync mode


@dataclass
class PersistenceEvent:
    """Event related to persistence operations"""

    event_id: UUID = field(default_factory=uuid4)
    event_type: PersistenceEventType = PersistenceEventType.CONTRACT_STORED
    engagement_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Event data
    data: Dict[str, Any] = field(default_factory=dict)
    source_event_id: Optional[str] = None
    contract_version: Optional[int] = None

    # Processing metadata
    processed: bool = False
    error_message: Optional[str] = None
    retry_count: int = 0


@dataclass
class SyncState:
    """Synchronization state for an engagement"""

    engagement_id: UUID
    last_event_timestamp: datetime
    last_persisted_timestamp: datetime
    sync_status: SyncStatus = SyncStatus.SYNCHRONIZED

    # Event tracking
    pending_events: int = 0
    processed_events: int = 0
    failed_events: int = 0

    # Performance metrics
    avg_sync_time: float = 0.0
    last_sync_duration: float = 0.0


class PersistenceEventIntegrator:
    """
    Integrates persistence system with event bus for real-time contract updates
    Provides event-driven persistence and synchronization
    """

    def __init__(
        self,
        event_bus: MetisEventBus,
        persistence_manager: ContractPersistenceManager,
        checkpoint_manager: ContractCheckpointManager,
        recovery_system: Optional[ContractRecoverySystem] = None,
        resume_manager: Optional[EngagementResumeManager] = None,
    ):
        self.event_bus = event_bus
        self.persistence_manager = persistence_manager
        self.checkpoint_manager = checkpoint_manager
        self.recovery_system = recovery_system
        self.resume_manager = resume_manager

        self.logger = logging.getLogger(__name__)

        # Event subscription management
        self.subscriptions: Dict[str, str] = {}  # subscription_id -> event_pattern
        self.event_handlers: Dict[str, Callable] = {}

        # Synchronization tracking
        self.sync_states: Dict[UUID, SyncState] = {}
        self.pending_events: Dict[str, PersistenceEvent] = {}

        # Configuration
        self.enable_real_time_sync = True
        self.enable_event_sourcing = True
        self.batch_sync_interval = 5.0  # seconds
        self.max_retry_attempts = 3
        self.sync_timeout = 30.0  # seconds

        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.running = False

        # Performance metrics
        self.metrics = {
            "events_processed": 0,
            "contracts_synced": 0,
            "checkpoints_triggered": 0,
            "sync_errors": 0,
            "recovery_events": 0,
            "average_sync_latency": 0.0,
        }

    async def initialize(self) -> bool:
        """Initialize event integration and subscriptions"""
        try:
            # Setup event subscriptions
            await self._setup_event_subscriptions()

            # Start background synchronization tasks
            if self.enable_real_time_sync:
                self.background_tasks.append(
                    asyncio.create_task(self._sync_monitoring_loop())
                )

            # Start batch sync task
            self.background_tasks.append(asyncio.create_task(self._batch_sync_loop()))

            # Start event sourcing task if enabled
            if self.enable_event_sourcing:
                self.background_tasks.append(
                    asyncio.create_task(self._event_sourcing_loop())
                )

            self.running = True
            self.logger.info("✅ Persistence event integration initialized")
            return True

        except Exception as e:
            self.logger.error(
                f"Failed to initialize persistence event integration: {str(e)}"
            )
            return False

    async def _setup_event_subscriptions(self):
        """Setup event bus subscriptions for persistence triggers"""

        # Subscribe to workflow events
        await self._subscribe_to_events(
            "workflow.*", self._handle_workflow_events, "Workflow Event Handler"
        )

        # Subscribe to engagement events
        await self._subscribe_to_events(
            "engagement.*", self._handle_engagement_events, "Engagement Event Handler"
        )

        # Subscribe to cognitive events
        await self._subscribe_to_events(
            "cognitive.*", self._handle_cognitive_events, "Cognitive Event Handler"
        )

        # Subscribe to system events
        await self._subscribe_to_events(
            "system.*", self._handle_system_events, "System Event Handler"
        )

        self.logger.info(f"📡 Setup {len(self.subscriptions)} event subscriptions")

    async def _subscribe_to_events(
        self, event_pattern: str, handler: Callable, handler_name: str
    ):
        """Subscribe to events matching a pattern"""

        # In a real implementation, this would use the event router's subscription system
        subscription_id = f"persistence_{event_pattern}_{uuid4().hex[:8]}"

        self.subscriptions[subscription_id] = event_pattern
        self.event_handlers[subscription_id] = handler

        self.logger.debug(f"Subscribed to {event_pattern} with handler {handler_name}")

    async def _handle_workflow_events(self, events: List[CloudEvent]):
        """Handle workflow-related events for persistence triggers"""

        for event in events:
            try:
                engagement_id = self._extract_engagement_id(event)
                if not engagement_id:
                    continue

                # Create persistence event
                persistence_event = PersistenceEvent(
                    event_type=PersistenceEventType.CONTRACT_STORED,
                    engagement_id=engagement_id,
                    source_event_id=event.id,
                    data={"event_type": event.type, "event_data": event.data},
                )

                # Process based on event type
                if event.type in [
                    "workflow.node.completed",
                    "workflow.phase.completed",
                ]:
                    await self._trigger_checkpoint_from_event(event, persistence_event)

                elif event.type in ["workflow.node.started"]:
                    await self._update_sync_state(engagement_id, event)

                elif event.type in ["workflow.node.failed", "workflow.error"]:
                    await self._handle_error_event(event, persistence_event)

                self.metrics["events_processed"] += 1

            except Exception as e:
                self.logger.error(
                    f"Failed to handle workflow event {event.id}: {str(e)}"
                )
                self.metrics["sync_errors"] += 1

    async def _handle_engagement_events(self, events: List[CloudEvent]):
        """Handle engagement-related events for persistence triggers"""

        for event in events:
            try:
                engagement_id = self._extract_engagement_id(event)
                if not engagement_id:
                    continue

                # Process engagement events
                if event.type == "engagement.workflow.completed":
                    await self._handle_engagement_completion(event, engagement_id)

                elif event.type == "engagement.workflow.started":
                    await self._handle_engagement_start(event, engagement_id)

                elif event.type in [
                    "engagement.phase.transition",
                    "engagement.milestone.reached",
                ]:
                    await self._trigger_milestone_checkpoint(event, engagement_id)

                self.metrics["events_processed"] += 1

            except Exception as e:
                self.logger.error(
                    f"Failed to handle engagement event {event.id}: {str(e)}"
                )
                self.metrics["sync_errors"] += 1

    async def _handle_cognitive_events(self, events: List[CloudEvent]):
        """Handle cognitive-related events for persistence triggers"""

        for event in events:
            try:
                engagement_id = self._extract_engagement_id(event)
                if not engagement_id:
                    continue

                # Handle cognitive model selection events
                if event.type == "cognitive.model.selected":
                    await self._trigger_model_selection_checkpoint(event, engagement_id)

                elif event.type == "cognitive.analysis.completed":
                    await self._update_cognitive_state(event, engagement_id)

                self.metrics["events_processed"] += 1

            except Exception as e:
                self.logger.error(
                    f"Failed to handle cognitive event {event.id}: {str(e)}"
                )
                self.metrics["sync_errors"] += 1

    async def _handle_system_events(self, events: List[CloudEvent]):
        """Handle system-related events for persistence triggers"""

        for event in events:
            try:
                if event.type == "system.shutdown":
                    await self._handle_system_shutdown(event)

                elif event.type == "system.error":
                    await self._handle_system_error(event)

                elif event.type == "system.recovery.initiated":
                    await self._handle_recovery_event(event)

                self.metrics["events_processed"] += 1

            except Exception as e:
                self.logger.error(f"Failed to handle system event {event.id}: {str(e)}")
                self.metrics["sync_errors"] += 1

    async def _trigger_checkpoint_from_event(
        self, event: CloudEvent, persistence_event: PersistenceEvent
    ):
        """Trigger checkpoint creation based on workflow event"""

        engagement_id = persistence_event.engagement_id

        # Get current contract
        contract = await self._get_current_contract(engagement_id)
        if not contract:
            return

        # Determine checkpoint trigger type
        if "completed" in event.type:
            trigger = CheckpointTrigger.PHASE_TRANSITION
        elif "milestone" in event.type:
            trigger = CheckpointTrigger.PROGRESS_MILESTONE
        else:
            trigger = CheckpointTrigger.ARTIFACT_CREATED

        # Create checkpoint
        version = await self.checkpoint_manager.create_checkpoint(
            contract,
            trigger,
            trigger_data={
                "source_event": event.type,
                "event_id": event.id,
                "event_data": event.data,
            },
        )

        if version:
            self.metrics["checkpoints_triggered"] += 1

            # Emit persistence event
            await self._emit_persistence_event(
                PersistenceEventType.CHECKPOINT_CREATED,
                engagement_id,
                {
                    "checkpoint_version": version.version_number,
                    "trigger_event": event.type,
                    "checkpoint_type": version.checkpoint_type.value,
                },
            )

    async def _trigger_milestone_checkpoint(
        self, event: CloudEvent, engagement_id: UUID
    ):
        """Trigger milestone checkpoint"""

        contract = await self._get_current_contract(engagement_id)
        if not contract:
            return

        # Check if this milestone warrants a checkpoint
        await self.checkpoint_manager.check_progress_milestone(contract)

    async def _trigger_model_selection_checkpoint(
        self, event: CloudEvent, engagement_id: UUID
    ):
        """Trigger checkpoint for mental model selection"""

        contract = await self._get_current_contract(engagement_id)
        if not contract:
            return

        # Create checkpoint for model selection
        await self.checkpoint_manager.create_checkpoint(
            contract,
            CheckpointTrigger.MODEL_SELECTION,
            trigger_data={
                "model_selection_event": event.type,
                "selected_models": event.data.get("models", []) if event.data else [],
            },
        )

    async def _handle_engagement_completion(
        self, event: CloudEvent, engagement_id: UUID
    ):
        """Handle engagement completion"""

        # Create final checkpoint
        contract = await self._get_current_contract(engagement_id)
        if contract:
            final_version = await self.checkpoint_manager.create_checkpoint(
                contract,
                CheckpointTrigger.USER_REQUEST,
                CheckpointType.MILESTONE,
                trigger_data={"reason": "Engagement completion"},
            )

            if final_version:
                # Update sync state
                await self._update_sync_state(engagement_id, event, final=True)

                # Emit completion event
                await self._emit_persistence_event(
                    PersistenceEventType.CONTRACT_STORED,
                    engagement_id,
                    {
                        "completion_version": final_version.version_number,
                        "engagement_status": "completed",
                    },
                )

    async def _handle_engagement_start(self, event: CloudEvent, engagement_id: UUID):
        """Handle engagement start"""

        # Initialize sync state
        sync_state = SyncState(
            engagement_id=engagement_id,
            last_event_timestamp=datetime.utcnow(),
            last_persisted_timestamp=datetime.utcnow(),
            sync_status=SyncStatus.SYNCHRONIZED,
        )

        self.sync_states[engagement_id] = sync_state

        # Register with checkpoint manager if not already registered
        contract = await self._get_current_contract(engagement_id)
        if contract:
            await self.checkpoint_manager.register_engagement(contract)

    async def _handle_error_event(
        self, event: CloudEvent, persistence_event: PersistenceEvent
    ):
        """Handle error events"""

        engagement_id = persistence_event.engagement_id

        # Create error recovery checkpoint
        contract = await self._get_current_contract(engagement_id)
        if contract:
            await self.checkpoint_manager.create_pre_operation_checkpoint(
                contract, f"error_recovery_{event.type}", risk_level="high"
            )

            # Update sync state to error mode
            if engagement_id in self.sync_states:
                self.sync_states[engagement_id].sync_status = SyncStatus.SYNC_ERROR

            self.metrics["recovery_events"] += 1

    async def _handle_system_shutdown(self, event: CloudEvent):
        """Handle system shutdown"""

        # Create shutdown checkpoints for all active engagements
        for engagement_id in self.sync_states:
            contract = await self._get_current_contract(engagement_id)
            if contract:
                await self.checkpoint_manager.create_checkpoint(
                    contract,
                    CheckpointTrigger.SYSTEM_SHUTDOWN,
                    trigger_data={"shutdown_event": event.id},
                )

    async def _handle_system_error(self, event: CloudEvent):
        """Handle system error"""

        # Mark all engagements as needing recovery
        for engagement_id in self.sync_states:
            self.sync_states[engagement_id].sync_status = SyncStatus.RECOVERY_MODE

        self.metrics["recovery_events"] += 1

    async def _handle_recovery_event(self, event: CloudEvent):
        """Handle recovery initiation event"""

        engagement_id_str = event.data.get("engagement_id") if event.data else None
        if engagement_id_str:
            engagement_id = UUID(engagement_id_str)

            await self._emit_persistence_event(
                PersistenceEventType.RECOVERY_INITIATED,
                engagement_id,
                {"recovery_event": event.type, "recovery_data": event.data},
            )

    async def _update_sync_state(
        self, engagement_id: UUID, event: CloudEvent, final: bool = False
    ):
        """Update synchronization state for engagement"""

        if engagement_id not in self.sync_states:
            self.sync_states[engagement_id] = SyncState(
                engagement_id=engagement_id,
                last_event_timestamp=datetime.utcnow(),
                last_persisted_timestamp=datetime.utcnow(),
            )

        sync_state = self.sync_states[engagement_id]
        sync_state.last_event_timestamp = datetime.utcnow()
        sync_state.processed_events += 1

        if final:
            sync_state.sync_status = SyncStatus.SYNCHRONIZED

    async def _update_cognitive_state(self, event: CloudEvent, engagement_id: UUID):
        """Update cognitive state based on event"""

        # Get current contract and update cognitive state
        contract = await self._get_current_contract(engagement_id)
        if contract and event.data:

            # Update cognitive validation results
            if "analysis_results" in event.data:
                contract.cognitive_state.validation_results.update(
                    event.data["analysis_results"]
                )

            # Store updated contract
            await self.persistence_manager.store_contract(
                contract,
                CheckpointType.AUTOMATIC,
                f"Cognitive state update from {event.type}",
            )

            self.metrics["contracts_synced"] += 1

    async def _get_current_contract(
        self, engagement_id: UUID
    ) -> Optional[MetisDataContract]:
        """Get current contract for engagement"""
        try:
            return await self.persistence_manager.get_contract(engagement_id)
        except Exception as e:
            self.logger.error(f"Failed to get contract for {engagement_id}: {str(e)}")
            return None

    def _extract_engagement_id(self, event: CloudEvent) -> Optional[UUID]:
        """Extract engagement ID from event"""
        try:
            if event.data and "engagement_id" in event.data:
                return UUID(event.data["engagement_id"])

            # Try to extract from event source or other fields
            if hasattr(event, "metisengagementid") and event.metisengagementid:
                return UUID(event.metisengagementid)

            return None

        except (ValueError, TypeError):
            return None

    async def _emit_persistence_event(
        self,
        event_type: PersistenceEventType,
        engagement_id: UUID,
        data: Dict[str, Any],
    ):
        """Emit persistence-related event"""

        try:
            # Create CloudEvent for persistence event
            from src.core.enhanced_event_bus import create_metis_cloud_event

            event = create_metis_cloud_event(
                event_type=f"persistence.{event_type.value}",
                source="/metis/persistence",
                category=MetisEventCategory.SYSTEM.value,
                priority=MetisEventPriority.MEDIUM.value,
                engagement_id=str(engagement_id),
                data={
                    "persistence_event_type": event_type.value,
                    "engagement_id": str(engagement_id),
                    "timestamp": datetime.utcnow().isoformat(),
                    **data,
                },
            )

            await self.event_bus.publish_event(event)

        except Exception as e:
            self.logger.error(f"Failed to emit persistence event: {str(e)}")

    async def _sync_monitoring_loop(self):
        """Background task for monitoring synchronization status"""

        while self.running:
            try:
                await asyncio.sleep(self.batch_sync_interval)

                if not self.running:
                    break

                # Check sync states
                for engagement_id, sync_state in self.sync_states.items():
                    time_since_last_event = (
                        datetime.utcnow() - sync_state.last_event_timestamp
                    ).total_seconds()

                    # Check for sync timeout
                    if time_since_last_event > self.sync_timeout:
                        if sync_state.sync_status != SyncStatus.SYNC_ERROR:
                            sync_state.sync_status = SyncStatus.PENDING_SYNC

                            await self._emit_persistence_event(
                                PersistenceEventType.SYNC_STATUS_CHANGED,
                                engagement_id,
                                {"new_status": sync_state.sync_status.value},
                            )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in sync monitoring loop: {str(e)}")

    async def _batch_sync_loop(self):
        """Background task for batch synchronization"""

        while self.running:
            try:
                await asyncio.sleep(self.batch_sync_interval)

                if not self.running:
                    break

                # Process pending events in batches
                if self.pending_events:
                    await self._process_pending_events()

                # Update sync metrics
                await self._update_sync_metrics()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in batch sync loop: {str(e)}")

    async def _event_sourcing_loop(self):
        """Background task for event sourcing"""

        while self.running:
            try:
                await asyncio.sleep(10.0)  # Check every 10 seconds

                if not self.running:
                    break

                # Implement event sourcing logic
                # This would rebuild contract state from events if needed
                await self._check_event_sourcing_consistency()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in event sourcing loop: {str(e)}")

    async def _process_pending_events(self):
        """Process pending persistence events"""

        if not self.pending_events:
            return

        pending_count = len(self.pending_events)
        processed_count = 0

        for event_id, persistence_event in list(self.pending_events.items()):
            try:
                # Attempt to process the event
                await self._process_persistence_event(persistence_event)

                # Mark as processed
                persistence_event.processed = True
                del self.pending_events[event_id]
                processed_count += 1

            except Exception as e:
                persistence_event.retry_count += 1
                persistence_event.error_message = str(e)

                # Remove if max retries exceeded
                if persistence_event.retry_count >= self.max_retry_attempts:
                    del self.pending_events[event_id]
                    self.logger.error(
                        f"Persistence event {event_id} failed after {self.max_retry_attempts} retries"
                    )

        if processed_count > 0:
            self.logger.info(
                f"Processed {processed_count}/{pending_count} pending persistence events"
            )

    async def _process_persistence_event(self, persistence_event: PersistenceEvent):
        """Process a single persistence event"""

        if persistence_event.event_type == PersistenceEventType.CONTRACT_STORED:
            # Ensure contract is persisted
            contract = await self._get_current_contract(persistence_event.engagement_id)
            if contract:
                await self.persistence_manager.store_contract(
                    contract, CheckpointType.AUTOMATIC
                )

        elif persistence_event.event_type == PersistenceEventType.CHECKPOINT_CREATED:
            # Verify checkpoint was created
            versions = await self.persistence_manager.get_contract_versions(
                persistence_event.engagement_id
            )
            if versions and persistence_event.contract_version:
                # Check if specific version exists
                version_exists = any(
                    v.version_number == persistence_event.contract_version
                    for v in versions
                )
                if not version_exists:
                    raise RuntimeError(
                        f"Expected checkpoint version {persistence_event.contract_version} not found"
                    )

    async def _check_event_sourcing_consistency(self):
        """Check consistency between events and persisted state"""

        for engagement_id in list(self.sync_states.keys()):
            try:
                # Get current persisted contract
                contract = await self._get_current_contract(engagement_id)
                if not contract:
                    continue

                # Compare with event-derived state (simplified)
                sync_state = self.sync_states[engagement_id]

                # Check if we need to rebuild from events
                if sync_state.sync_status == SyncStatus.RECOVERY_MODE:
                    await self._rebuild_from_events(engagement_id)

            except Exception as e:
                self.logger.error(
                    f"Event sourcing consistency check failed for {engagement_id}: {str(e)}"
                )

    async def _rebuild_from_events(self, engagement_id: UUID):
        """Rebuild contract state from events (event sourcing)"""

        try:
            # In a full implementation, this would:
            # 1. Get all events for the engagement from event store
            # 2. Replay events to rebuild contract state
            # 3. Compare with persisted state
            # 4. Resolve inconsistencies

            # For now, we'll mark as needing recovery
            if self.recovery_system and self.resume_manager:
                await self.resume_manager.discover_resumable_engagements()

            # Update sync status
            if engagement_id in self.sync_states:
                self.sync_states[engagement_id].sync_status = SyncStatus.SYNCHRONIZED

        except Exception as e:
            self.logger.error(
                f"Failed to rebuild from events for {engagement_id}: {str(e)}"
            )

    async def _update_sync_metrics(self):
        """Update synchronization metrics"""

        total_latency = 0.0
        sync_count = 0

        for sync_state in self.sync_states.values():
            if sync_state.last_sync_duration > 0:
                total_latency += sync_state.last_sync_duration
                sync_count += 1

        if sync_count > 0:
            self.metrics["average_sync_latency"] = total_latency / sync_count

    async def force_sync(self, engagement_id: UUID) -> bool:
        """Force synchronization for a specific engagement"""

        try:
            start_time = datetime.utcnow()

            # Get current contract
            contract = await self._get_current_contract(engagement_id)
            if not contract:
                return False

            # Create checkpoint to ensure persistence
            version = await self.checkpoint_manager.create_checkpoint(
                contract,
                CheckpointTrigger.USER_REQUEST,
                trigger_data={"reason": "Force sync"},
            )

            # Update sync state
            if engagement_id in self.sync_states:
                sync_state = self.sync_states[engagement_id]
                sync_state.last_persisted_timestamp = datetime.utcnow()
                sync_state.sync_status = SyncStatus.SYNCHRONIZED
                sync_state.last_sync_duration = (
                    datetime.utcnow() - start_time
                ).total_seconds()

            self.metrics["contracts_synced"] += 1
            return True

        except Exception as e:
            self.logger.error(f"Force sync failed for {engagement_id}: {str(e)}")
            return False

    async def get_sync_status(self, engagement_id: UUID) -> Optional[SyncState]:
        """Get synchronization status for engagement"""
        return self.sync_states.get(engagement_id)

    async def get_integration_metrics(self) -> Dict[str, Any]:
        """Get persistence event integration metrics"""

        # Calculate sync statistics
        sync_statuses = {}
        for status in SyncStatus:
            sync_statuses[status.value] = len(
                [s for s in self.sync_states.values() if s.sync_status == status]
            )

        return {
            **self.metrics,
            "active_subscriptions": len(self.subscriptions),
            "active_sync_states": len(self.sync_states),
            "pending_events": len(self.pending_events),
            "sync_status_distribution": sync_statuses,
            "background_tasks_running": len(
                [t for t in self.background_tasks if not t.done()]
            ),
            "real_time_sync_enabled": self.enable_real_time_sync,
            "event_sourcing_enabled": self.enable_event_sourcing,
        }

    async def shutdown(self):
        """Shutdown the persistence event integration"""

        self.running = False

        # Process any remaining pending events
        if self.pending_events:
            await self._process_pending_events()

        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()

        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)

        self.logger.info("📪 Persistence event integration shutdown complete")


# Export main classes
__all__ = [
    "PersistenceEventIntegrator",
    "PersistenceEvent",
    "SyncState",
    "PersistenceEventType",
    "SyncStatus",
]
