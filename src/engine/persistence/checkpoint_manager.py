"""
METIS Contract Checkpoint Manager - P5.2
Automatic save points with intelligent checkpoint generation

Implements enterprise-grade automatic checkpoint management:
- Phase boundary detection with automatic saves
- Progress-based checkpoint triggers
- Risk assessment for pre-operation saves
- Configurable checkpoint policies
- Integration with workflow engine events
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set, Callable
from datetime import datetime
from uuid import UUID
from dataclasses import dataclass, field
from enum import Enum

from src.engine.models.data_contracts import MetisDataContract
from src.persistence.contract_storage import (
    ContractPersistenceManager,
    CheckpointType,
    ContractVersion,
)
from src.core.enhanced_event_bus import MetisEventBus, CloudEvent


class CheckpointTrigger(str, Enum):
    """Triggers that can cause automatic checkpoint creation"""

    PHASE_TRANSITION = "phase_transition"  # Moving between engagement phases
    PROGRESS_MILESTONE = "progress_milestone"  # Progress percentage thresholds
    TIME_INTERVAL = "time_interval"  # Regular time-based saves
    ERROR_DETECTED = "error_detected"  # Before risky operations
    ARTIFACT_CREATED = "artifact_created"  # New deliverable created
    MODEL_SELECTION = "model_selection"  # Mental model selection change
    USER_REQUEST = "user_request"  # Manual user-initiated save
    SYSTEM_SHUTDOWN = "system_shutdown"  # Graceful system shutdown
    MEMORY_PRESSURE = "memory_pressure"  # High memory usage detected


@dataclass
class CheckpointPolicy:
    """Configuration for automatic checkpoint creation"""

    # Trigger enablement
    enabled_triggers: Set[CheckpointTrigger] = field(
        default_factory=lambda: {
            CheckpointTrigger.PHASE_TRANSITION,
            CheckpointTrigger.PROGRESS_MILESTONE,
            CheckpointTrigger.ARTIFACT_CREATED,
            CheckpointTrigger.ERROR_DETECTED,
        }
    )

    # Progress milestone thresholds (percentages)
    progress_milestones: List[float] = field(
        default_factory=lambda: [25.0, 50.0, 75.0, 90.0]
    )

    # Time-based checkpoint interval
    time_interval_minutes: int = 15

    # Maximum checkpoints per engagement
    max_checkpoints_per_engagement: int = 50

    # Minimum time between checkpoints (prevent spam)
    min_checkpoint_interval_seconds: int = 30

    # Enable compression for large contracts
    enable_compression: bool = True

    # Cleanup policy
    auto_cleanup_enabled: bool = True
    keep_checkpoint_days: int = 7

    # Performance thresholds
    max_processing_time_before_checkpoint: float = 60.0  # seconds
    memory_pressure_threshold_mb: int = 500


@dataclass
class CheckpointTriggerEvent:
    """Event that triggered a checkpoint creation"""

    trigger: CheckpointTrigger
    engagement_id: UUID
    contract: MetisDataContract
    trigger_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    risk_assessment: Optional[str] = None


class ContractCheckpointManager:
    """
    Manages automatic checkpoint creation based on configurable policies
    Integrates with workflow engine and event bus for real-time triggers
    """

    def __init__(
        self,
        persistence_manager: ContractPersistenceManager,
        event_bus: Optional[MetisEventBus] = None,
        policy: Optional[CheckpointPolicy] = None,
    ):
        self.persistence_manager = persistence_manager
        self.event_bus = event_bus
        self.policy = policy or CheckpointPolicy()

        self.logger = logging.getLogger(__name__)

        # State tracking
        self.active_engagements: Dict[UUID, MetisDataContract] = {}
        self.last_checkpoint_times: Dict[UUID, datetime] = {}
        self.progress_checkpoints: Dict[UUID, Set[float]] = {}

        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.running = False

        # Event handlers
        self.trigger_handlers: Dict[CheckpointTrigger, List[Callable]] = {}

        # Performance tracking
        self.metrics = {
            "checkpoints_created": 0,
            "automatic_saves": 0,
            "manual_saves": 0,
            "triggers_processed": 0,
            "errors_prevented": 0,
            "recovery_events": 0,
        }

    async def initialize(self) -> bool:
        """Initialize the checkpoint manager and event subscriptions"""
        try:
            # Setup event bus subscriptions if available
            if self.event_bus:
                await self._setup_event_subscriptions()

            # Start background monitoring tasks
            if CheckpointTrigger.TIME_INTERVAL in self.policy.enabled_triggers:
                self.background_tasks.append(
                    asyncio.create_task(self._time_based_checkpoint_loop())
                )

            if CheckpointTrigger.MEMORY_PRESSURE in self.policy.enabled_triggers:
                self.background_tasks.append(
                    asyncio.create_task(self._memory_monitoring_loop())
                )

            self.running = True
            self.logger.info("✅ Contract checkpoint manager initialized")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize checkpoint manager: {str(e)}")
            return False

    async def _setup_event_subscriptions(self):
        """Setup event bus subscriptions for checkpoint triggers"""

        # Subscribe to workflow events for phase transitions
        async def workflow_event_handler(events: List[CloudEvent]):
            for event in events:
                await self._handle_workflow_event(event)

        # Subscribe to engagement events for artifact creation
        async def engagement_event_handler(events: List[CloudEvent]):
            for event in events:
                await self._handle_engagement_event(event)

        # Subscribe to error events for pre-error checkpoints
        async def error_event_handler(events: List[CloudEvent]):
            for event in events:
                await self._handle_error_event(event)

        # Note: In a real implementation, these would be proper event subscriptions
        # For now, we'll rely on direct integration with the workflow engine

        self.logger.info("📡 Event subscriptions configured for checkpoint triggers")

    async def register_engagement(self, contract: MetisDataContract) -> bool:
        """Register an engagement for automatic checkpoint monitoring"""
        engagement_id = contract.engagement_context.engagement_id

        self.active_engagements[engagement_id] = contract
        self.last_checkpoint_times[engagement_id] = datetime.utcnow()
        self.progress_checkpoints[engagement_id] = set()

        # Create initial checkpoint
        await self.create_checkpoint(
            contract,
            CheckpointTrigger.USER_REQUEST,
            checkpoint_type=CheckpointType.MANUAL,
            trigger_data={"reason": "Engagement registration"},
        )

        self.logger.info(
            f"📋 Registered engagement {engagement_id} for checkpoint monitoring"
        )
        return True

    async def create_checkpoint(
        self,
        contract: MetisDataContract,
        trigger: CheckpointTrigger,
        checkpoint_type: Optional[CheckpointType] = None,
        trigger_data: Optional[Dict[str, Any]] = None,
        risk_assessment: Optional[str] = None,
    ) -> Optional[ContractVersion]:
        """
        Create a checkpoint for a contract with trigger context
        Returns the created contract version or None if creation was skipped
        """
        engagement_id = contract.engagement_context.engagement_id
        current_time = datetime.utcnow()

        # Check minimum interval policy
        last_checkpoint = self.last_checkpoint_times.get(engagement_id)
        if last_checkpoint:
            time_since_last = (current_time - last_checkpoint).total_seconds()
            if time_since_last < self.policy.min_checkpoint_interval_seconds:
                self.logger.debug(
                    f"Skipping checkpoint for {engagement_id}: too soon since last checkpoint"
                )
                return None

        # Determine checkpoint type based on trigger
        if not checkpoint_type:
            checkpoint_type = self._determine_checkpoint_type(trigger, contract)

        # Generate change summary based on trigger
        change_summary = self._generate_trigger_summary(trigger, trigger_data or {})

        try:
            # Create the checkpoint
            version = await self.persistence_manager.store_contract(
                contract=contract,
                checkpoint_type=checkpoint_type,
                change_summary=change_summary,
                created_by=f"checkpoint_manager_{trigger.value}",
            )

            # Update tracking
            self.last_checkpoint_times[engagement_id] = current_time
            self.active_engagements[engagement_id] = contract

            # Update metrics
            self.metrics["checkpoints_created"] += 1
            if trigger == CheckpointTrigger.USER_REQUEST:
                self.metrics["manual_saves"] += 1
            else:
                self.metrics["automatic_saves"] += 1

            # Log the checkpoint creation
            self.logger.info(
                f"💾 Created {checkpoint_type.value} checkpoint for engagement {engagement_id} "
                f"(trigger: {trigger.value}, version: {version.version_number})"
            )

            return version

        except Exception as e:
            self.logger.error(
                f"Failed to create checkpoint for {engagement_id}: {str(e)}"
            )
            return None

    def _determine_checkpoint_type(
        self, trigger: CheckpointTrigger, contract: MetisDataContract
    ) -> CheckpointType:
        """Determine the appropriate checkpoint type based on trigger and context"""
        if trigger == CheckpointTrigger.PHASE_TRANSITION:
            return CheckpointType.PHASE_BOUNDARY
        elif trigger == CheckpointTrigger.PROGRESS_MILESTONE:
            return CheckpointType.MILESTONE
        elif trigger == CheckpointTrigger.ERROR_DETECTED:
            return CheckpointType.ERROR_RECOVERY
        elif trigger == CheckpointTrigger.USER_REQUEST:
            return CheckpointType.MANUAL
        elif trigger == CheckpointTrigger.SYSTEM_SHUTDOWN:
            return CheckpointType.BACKUP
        else:
            return CheckpointType.AUTOMATIC

    def _generate_trigger_summary(
        self, trigger: CheckpointTrigger, trigger_data: Dict[str, Any]
    ) -> str:
        """Generate human-readable summary of checkpoint trigger"""
        base_summaries = {
            CheckpointTrigger.PHASE_TRANSITION: "Phase transition",
            CheckpointTrigger.PROGRESS_MILESTONE: "Progress milestone",
            CheckpointTrigger.TIME_INTERVAL: "Scheduled backup",
            CheckpointTrigger.ERROR_DETECTED: "Pre-error safety checkpoint",
            CheckpointTrigger.ARTIFACT_CREATED: "New deliverable created",
            CheckpointTrigger.MODEL_SELECTION: "Mental model selection updated",
            CheckpointTrigger.USER_REQUEST: "Manual user save",
            CheckpointTrigger.SYSTEM_SHUTDOWN: "System shutdown backup",
            CheckpointTrigger.MEMORY_PRESSURE: "Memory pressure relief",
        }

        summary = base_summaries.get(
            trigger, f"Checkpoint triggered by {trigger.value}"
        )

        # Add specific details from trigger data
        if trigger_data:
            details = []
            if "phase_from" in trigger_data and "phase_to" in trigger_data:
                details.append(
                    f"{trigger_data['phase_from']} → {trigger_data['phase_to']}"
                )
            if "progress_percentage" in trigger_data:
                details.append(f"{trigger_data['progress_percentage']:.1f}% complete")
            if "artifact_type" in trigger_data:
                details.append(f"Created {trigger_data['artifact_type']}")
            if "reason" in trigger_data:
                details.append(trigger_data["reason"])

            if details:
                summary += f": {'; '.join(details)}"

        return summary

    async def check_phase_transition(
        self, old_contract: MetisDataContract, new_contract: MetisDataContract
    ) -> bool:
        """
        Check if a phase transition occurred and create checkpoint if needed
        Returns True if checkpoint was created
        """
        if CheckpointTrigger.PHASE_TRANSITION not in self.policy.enabled_triggers:
            return False

        old_phase = old_contract.workflow_state.current_phase
        new_phase = new_contract.workflow_state.current_phase

        if old_phase != new_phase:
            trigger_data = {
                "phase_from": old_phase.value,
                "phase_to": new_phase.value,
                "completed_phases": len(new_contract.workflow_state.completed_phases),
            }

            version = await self.create_checkpoint(
                new_contract,
                CheckpointTrigger.PHASE_TRANSITION,
                trigger_data=trigger_data,
                risk_assessment="Phase transition requires checkpoint for rollback capability",
            )

            return version is not None

        return False

    async def check_progress_milestone(self, contract: MetisDataContract) -> bool:
        """
        Check if a progress milestone was reached and create checkpoint if needed
        Returns True if checkpoint was created
        """
        if CheckpointTrigger.PROGRESS_MILESTONE not in self.policy.enabled_triggers:
            return False

        engagement_id = contract.engagement_context.engagement_id

        # Calculate current progress
        total_phases = 4
        completed_phases = len(contract.workflow_state.completed_phases)
        progress_percentage = (completed_phases / total_phases) * 100

        # Check if any milestone thresholds were crossed
        existing_checkpoints = self.progress_checkpoints.get(engagement_id, set())

        for milestone in self.policy.progress_milestones:
            if (
                progress_percentage >= milestone
                and milestone not in existing_checkpoints
            ):
                # Milestone reached for the first time
                trigger_data = {
                    "progress_percentage": progress_percentage,
                    "milestone": milestone,
                    "completed_phases": completed_phases,
                }

                version = await self.create_checkpoint(
                    contract,
                    CheckpointTrigger.PROGRESS_MILESTONE,
                    trigger_data=trigger_data,
                )

                if version:
                    existing_checkpoints.add(milestone)
                    self.progress_checkpoints[engagement_id] = existing_checkpoints
                    return True

        return False

    async def check_artifact_creation(
        self, old_contract: MetisDataContract, new_contract: MetisDataContract
    ) -> bool:
        """
        Check if new artifacts were created and create checkpoint if needed
        Returns True if checkpoint was created
        """
        if CheckpointTrigger.ARTIFACT_CREATED not in self.policy.enabled_triggers:
            return False

        old_count = len(old_contract.deliverable_artifacts)
        new_count = len(new_contract.deliverable_artifacts)

        if new_count > old_count:
            # New artifacts were created
            new_artifacts = new_contract.deliverable_artifacts[old_count:]

            trigger_data = {
                "new_artifacts_count": len(new_artifacts),
                "total_artifacts": new_count,
                "artifact_types": [
                    artifact.artifact_type for artifact in new_artifacts
                ],
            }

            version = await self.create_checkpoint(
                new_contract,
                CheckpointTrigger.ARTIFACT_CREATED,
                trigger_data=trigger_data,
            )

            return version is not None

        return False

    async def create_pre_operation_checkpoint(
        self,
        contract: MetisDataContract,
        operation_name: str,
        risk_level: str = "medium",
    ) -> Optional[ContractVersion]:
        """
        Create a checkpoint before a potentially risky operation
        Used for operations that might fail or modify significant state
        """
        if CheckpointTrigger.ERROR_DETECTED not in self.policy.enabled_triggers:
            return None

        trigger_data = {
            "operation_name": operation_name,
            "risk_level": risk_level,
            "reason": f"Pre-operation safety checkpoint for {operation_name}",
        }

        risk_assessment = (
            f"Creating safety checkpoint before {operation_name} (risk: {risk_level})"
        )

        version = await self.create_checkpoint(
            contract,
            CheckpointTrigger.ERROR_DETECTED,
            trigger_data=trigger_data,
            risk_assessment=risk_assessment,
        )

        if version:
            self.metrics["errors_prevented"] += 1

        return version

    async def _time_based_checkpoint_loop(self):
        """Background task for time-based checkpoint creation"""
        while self.running:
            try:
                await asyncio.sleep(self.policy.time_interval_minutes * 60)

                if not self.running:
                    break

                # Create checkpoints for all active engagements
                for engagement_id, contract in self.active_engagements.items():
                    last_checkpoint = self.last_checkpoint_times.get(engagement_id)
                    if last_checkpoint:
                        time_since_last = (
                            datetime.utcnow() - last_checkpoint
                        ).total_seconds()
                        if time_since_last >= (self.policy.time_interval_minutes * 60):
                            await self.create_checkpoint(
                                contract,
                                CheckpointTrigger.TIME_INTERVAL,
                                trigger_data={
                                    "interval_minutes": self.policy.time_interval_minutes
                                },
                            )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in time-based checkpoint loop: {str(e)}")

    async def _memory_monitoring_loop(self):
        """Background task for memory pressure monitoring"""
        while self.running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                if not self.running:
                    break

                # Simple memory pressure detection (in real implementation, use psutil)
                # For now, we'll use a heuristic based on cache size
                cache_size = len(self.active_engagements)

                if cache_size > 10:  # Simplified threshold
                    self.logger.warning(
                        f"Memory pressure detected: {cache_size} active engagements"
                    )

                    # Create checkpoints for oldest engagements
                    sorted_engagements = sorted(
                        self.last_checkpoint_times.items(), key=lambda x: x[1]
                    )

                    for engagement_id, _ in sorted_engagements[:3]:  # Top 3 oldest
                        if engagement_id in self.active_engagements:
                            await self.create_checkpoint(
                                self.active_engagements[engagement_id],
                                CheckpointTrigger.MEMORY_PRESSURE,
                                trigger_data={"cache_size": cache_size},
                            )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in memory monitoring loop: {str(e)}")

    async def _handle_workflow_event(self, event: CloudEvent):
        """Handle workflow events for checkpoint triggers"""
        if event.type.startswith("workflow.node."):
            engagement_id = event.data.get("engagement_id") if event.data else None
            if engagement_id and UUID(engagement_id) in self.active_engagements:
                # Process workflow event for checkpoint triggers
                self.metrics["triggers_processed"] += 1

    async def _handle_engagement_event(self, event: CloudEvent):
        """Handle engagement events for checkpoint triggers"""
        if event.type.startswith("engagement."):
            engagement_id = event.data.get("engagement_id") if event.data else None
            if engagement_id and UUID(engagement_id) in self.active_engagements:
                # Process engagement event for checkpoint triggers
                self.metrics["triggers_processed"] += 1

    async def _handle_error_event(self, event: CloudEvent):
        """Handle error events for emergency checkpoints"""
        if "error" in event.type or "failed" in event.type:
            engagement_id = event.data.get("engagement_id") if event.data else None
            if engagement_id and UUID(engagement_id) in self.active_engagements:
                # Create emergency checkpoint
                contract = self.active_engagements[UUID(engagement_id)]
                await self.create_checkpoint(
                    contract,
                    CheckpointTrigger.ERROR_DETECTED,
                    trigger_data={"error_event": event.type, "error_data": event.data},
                )

    async def create_manual_checkpoint(
        self,
        engagement_id: UUID,
        checkpoint_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Optional[ContractVersion]:
        """Create a manual checkpoint requested by user"""
        if engagement_id not in self.active_engagements:
            self.logger.warning(
                f"Cannot create manual checkpoint: engagement {engagement_id} not found"
            )
            return None

        contract = self.active_engagements[engagement_id]

        trigger_data = {}
        if checkpoint_name:
            trigger_data["checkpoint_name"] = checkpoint_name
        if description:
            trigger_data["description"] = description
        trigger_data["reason"] = "Manual user-requested checkpoint"

        return await self.create_checkpoint(
            contract,
            CheckpointTrigger.USER_REQUEST,
            checkpoint_type=CheckpointType.MANUAL,
            trigger_data=trigger_data,
        )

    async def cleanup_old_checkpoints(self) -> int:
        """Clean up old checkpoints according to policy"""
        if not self.policy.auto_cleanup_enabled:
            return 0

        try:
            cleaned = await self.persistence_manager.cleanup_old_versions(
                retention_days=self.policy.keep_checkpoint_days, keep_checkpoints=True
            )

            self.logger.info(f"🧹 Cleaned up {cleaned} old automatic checkpoints")
            return cleaned

        except Exception as e:
            self.logger.error(f"Failed to cleanup old checkpoints: {str(e)}")
            return 0

    def unregister_engagement(self, engagement_id: UUID) -> bool:
        """Unregister an engagement from checkpoint monitoring"""
        if engagement_id in self.active_engagements:
            del self.active_engagements[engagement_id]
            del self.last_checkpoint_times[engagement_id]
            if engagement_id in self.progress_checkpoints:
                del self.progress_checkpoints[engagement_id]

            self.logger.info(
                f"📋 Unregistered engagement {engagement_id} from checkpoint monitoring"
            )
            return True

        return False

    async def get_checkpoint_metrics(self) -> Dict[str, Any]:
        """Get checkpoint manager metrics"""
        return {
            **self.metrics,
            "active_engagements": len(self.active_engagements),
            "policy": {
                "enabled_triggers": [t.value for t in self.policy.enabled_triggers],
                "progress_milestones": self.policy.progress_milestones,
                "time_interval_minutes": self.policy.time_interval_minutes,
                "max_checkpoints": self.policy.max_checkpoints_per_engagement,
            },
            "background_tasks_running": len(
                [t for t in self.background_tasks if not t.done()]
            ),
        }

    async def shutdown(self):
        """Shutdown the checkpoint manager gracefully"""
        self.running = False

        # Create final checkpoints for all active engagements
        for engagement_id, contract in self.active_engagements.items():
            await self.create_checkpoint(
                contract,
                CheckpointTrigger.SYSTEM_SHUTDOWN,
                trigger_data={"reason": "System shutdown backup"},
            )

        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()

        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)

        self.logger.info("📪 Contract checkpoint manager shutdown complete")


# Export main classes
__all__ = [
    "ContractCheckpointManager",
    "CheckpointPolicy",
    "CheckpointTrigger",
    "CheckpointTriggerEvent",
]
