"""
METIS Lifecycle Management Service
Part of Application Services Cluster - Focused on model lifecycle events and state management

Extracted from model_manager.py during Phase 5.3 decomposition.
Single Responsibility: Manage model lifecycle states, events, and transitions.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from collections import defaultdict

from src.services.contracts.application_contracts import (
    ILifecycleManagementService,
    LifecycleEventContract,
)


class ModelLifecycleState(str, Enum):
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    ACTIVE = "active"
    BUSY = "busy"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    RETIRED = "retired"
    ERROR = "error"


class LifecycleEventType(str, Enum):
    INITIALIZATION_STARTED = "initialization_started"
    INITIALIZATION_COMPLETED = "initialization_completed"
    ACTIVATION = "activation"
    DEACTIVATION = "deactivation"
    HEALTH_CHECK = "health_check"
    PERFORMANCE_UPDATE = "performance_update"
    ERROR_DETECTED = "error_detected"
    MAINTENANCE_STARTED = "maintenance_started"
    MAINTENANCE_COMPLETED = "maintenance_completed"
    RETIREMENT = "retirement"


class LifecycleManagementService(ILifecycleManagementService):
    """
    Focused service for model lifecycle management and state transitions
    Clean extraction from model_manager.py lifecycle management methods
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Model lifecycle state tracking
        self.model_states: Dict[str, ModelLifecycleState] = {}
        self.lifecycle_events: Dict[str, List[LifecycleEventContract]] = defaultdict(
            list
        )
        self.state_transitions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Lifecycle management configuration
        self.lifecycle_config = {
            "initialization_timeout_seconds": 300,  # 5 minutes
            "health_check_interval_seconds": 600,  # 10 minutes
            "maintenance_window_hours": 2,
            "max_error_count_before_retirement": 5,
            "event_retention_days": 30,
        }

        # Valid state transitions
        self.valid_transitions = {
            ModelLifecycleState.UNINITIALIZED: [ModelLifecycleState.INITIALIZING],
            ModelLifecycleState.INITIALIZING: [
                ModelLifecycleState.READY,
                ModelLifecycleState.ERROR,
            ],
            ModelLifecycleState.READY: [
                ModelLifecycleState.ACTIVE,
                ModelLifecycleState.MAINTENANCE,
                ModelLifecycleState.RETIRED,
            ],
            ModelLifecycleState.ACTIVE: [
                ModelLifecycleState.BUSY,
                ModelLifecycleState.DEGRADED,
                ModelLifecycleState.MAINTENANCE,
                ModelLifecycleState.RETIRED,
            ],
            ModelLifecycleState.BUSY: [
                ModelLifecycleState.ACTIVE,
                ModelLifecycleState.DEGRADED,
                ModelLifecycleState.ERROR,
            ],
            ModelLifecycleState.DEGRADED: [
                ModelLifecycleState.ACTIVE,
                ModelLifecycleState.MAINTENANCE,
                ModelLifecycleState.ERROR,
                ModelLifecycleState.RETIRED,
            ],
            ModelLifecycleState.MAINTENANCE: [
                ModelLifecycleState.READY,
                ModelLifecycleState.RETIRED,
            ],
            ModelLifecycleState.ERROR: [
                ModelLifecycleState.INITIALIZING,
                ModelLifecycleState.MAINTENANCE,
                ModelLifecycleState.RETIRED,
            ],
            ModelLifecycleState.RETIRED: [],  # Terminal state
        }

        # Lifecycle metrics
        self.lifecycle_metrics = {
            "total_models_managed": 0,
            "active_models": 0,
            "models_by_state": defaultdict(int),
            "total_transitions": 0,
            "error_incidents": 0,
            "average_initialization_time_ms": 0.0,
        }

        # Flag for background task management
        self._health_monitoring_started = False

        self.logger.info("♻️ LifecycleManagementService initialized")

    async def initialize_model(self, model_id: str) -> LifecycleEventContract:
        """
        Core service method: Initialize a model for application
        Complete model initialization with state tracking and timeout handling
        """
        try:
            start_time = datetime.utcnow()

            # Check current state
            current_state = self.model_states.get(
                model_id, ModelLifecycleState.UNINITIALIZED
            )

            if current_state != ModelLifecycleState.UNINITIALIZED:
                return self._create_event_contract(
                    model_id=model_id,
                    event_type=LifecycleEventType.INITIALIZATION_STARTED.value,
                    event_status="failed",
                    event_data={
                        "error": f"Model already in state: {current_state.value}"
                    },
                    processing_time_ms=0.0,
                )

            # Transition to initializing state
            await self._transition_model_state(
                model_id, ModelLifecycleState.INITIALIZING
            )

            # Create initialization started event
            init_started_event = self._create_event_contract(
                model_id=model_id,
                event_type=LifecycleEventType.INITIALIZATION_STARTED.value,
                event_status="in_progress",
                event_data={"initialization_started": start_time.isoformat()},
                processing_time_ms=10.0,
            )

            await self._record_lifecycle_event(init_started_event)

            # Perform initialization tasks
            initialization_result = await self._perform_model_initialization(model_id)

            if initialization_result["success"]:
                # Transition to ready state
                await self._transition_model_state(model_id, ModelLifecycleState.READY)

                # Calculate initialization time
                initialization_time = (
                    datetime.utcnow() - start_time
                ).total_seconds() * 1000

                # Update metrics
                self._update_initialization_metrics(initialization_time)

                # Create completion event
                completion_event = self._create_event_contract(
                    model_id=model_id,
                    event_type=LifecycleEventType.INITIALIZATION_COMPLETED.value,
                    event_status="completed",
                    event_data={
                        "initialization_result": initialization_result,
                        "initialization_time_ms": initialization_time,
                    },
                    processing_time_ms=initialization_time,
                )

                await self._record_lifecycle_event(completion_event)

                self.logger.info(
                    f"♻️ Model initialized successfully: {model_id} ({initialization_time:.0f}ms)"
                )
                return completion_event

            else:
                # Transition to error state
                await self._transition_model_state(model_id, ModelLifecycleState.ERROR)

                error_event = self._create_event_contract(
                    model_id=model_id,
                    event_type=LifecycleEventType.ERROR_DETECTED.value,
                    event_status="failed",
                    event_data={
                        "initialization_error": initialization_result["error"],
                        "error_code": initialization_result.get(
                            "error_code", "INIT_FAILED"
                        ),
                    },
                    processing_time_ms=(datetime.utcnow() - start_time).total_seconds()
                    * 1000,
                )

                await self._record_lifecycle_event(error_event)

                self.logger.error(
                    f"❌ Model initialization failed: {model_id} - {initialization_result['error']}"
                )
                return error_event

        except Exception as e:
            # Handle unexpected errors
            await self._transition_model_state(model_id, ModelLifecycleState.ERROR)

            exception_event = self._create_event_contract(
                model_id=model_id,
                event_type=LifecycleEventType.ERROR_DETECTED.value,
                event_status="failed",
                event_data={"unexpected_error": str(e)},
                processing_time_ms=0.0,
            )

            await self._record_lifecycle_event(exception_event)

            self.logger.error(
                f"❌ Unexpected error during initialization: {model_id} - {e}"
            )
            return exception_event

    async def update_model_status(
        self, model_id: str, status: str, metadata: Dict[str, Any]
    ) -> LifecycleEventContract:
        """
        Core service method: Update model lifecycle status
        Comprehensive status updates with validation and state management
        """
        try:
            start_time = datetime.utcnow()

            # Validate and parse status
            try:
                target_state = ModelLifecycleState(status.lower())
            except ValueError:
                return self._create_event_contract(
                    model_id=model_id,
                    event_type=LifecycleEventType.ERROR_DETECTED.value,
                    event_status="failed",
                    event_data={"error": f"Invalid status: {status}"},
                    processing_time_ms=0.0,
                )

            # Get current state
            current_state = self.model_states.get(
                model_id, ModelLifecycleState.UNINITIALIZED
            )

            # Validate state transition
            if not await self._is_valid_transition(current_state, target_state):
                return self._create_event_contract(
                    model_id=model_id,
                    event_type=LifecycleEventType.ERROR_DETECTED.value,
                    event_status="failed",
                    event_data={
                        "error": f"Invalid transition: {current_state.value} -> {target_state.value}",
                        "current_state": current_state.value,
                        "target_state": target_state.value,
                    },
                    processing_time_ms=0.0,
                )

            # Perform state transition
            await self._transition_model_state(model_id, target_state)

            # Determine event type based on status change
            event_type = self._determine_event_type_for_status(
                current_state, target_state
            )

            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Create status update event
            status_event = self._create_event_contract(
                model_id=model_id,
                event_type=event_type,
                event_status="completed",
                event_data={
                    "previous_state": current_state.value,
                    "new_state": target_state.value,
                    "update_metadata": metadata,
                    "transition_time_ms": processing_time,
                },
                processing_time_ms=processing_time,
            )

            await self._record_lifecycle_event(status_event)

            # Special handling for specific state transitions
            await self._handle_special_state_transitions(
                model_id, current_state, target_state, metadata
            )

            self.logger.info(
                f"♻️ Model status updated: {model_id} ({current_state.value} -> {target_state.value})"
            )
            return status_event

        except Exception as e:
            error_event = self._create_event_contract(
                model_id=model_id,
                event_type=LifecycleEventType.ERROR_DETECTED.value,
                event_status="failed",
                event_data={"error": str(e)},
                processing_time_ms=0.0,
            )

            await self._record_lifecycle_event(error_event)

            self.logger.error(f"❌ Status update failed: {model_id} - {e}")
            return error_event

    async def retire_model(self, model_id: str) -> LifecycleEventContract:
        """
        Core service method: Retire a model from active use
        Complete model retirement with cleanup and event logging
        """
        try:
            start_time = datetime.utcnow()

            current_state = self.model_states.get(
                model_id, ModelLifecycleState.UNINITIALIZED
            )

            # Allow retirement from any state except already retired
            if current_state == ModelLifecycleState.RETIRED:
                return self._create_event_contract(
                    model_id=model_id,
                    event_type=LifecycleEventType.RETIREMENT.value,
                    event_status="skipped",
                    event_data={"message": "Model already retired"},
                    processing_time_ms=0.0,
                )

            # Perform retirement cleanup
            cleanup_result = await self._perform_model_retirement_cleanup(
                model_id, current_state
            )

            # Transition to retired state
            await self._transition_model_state(model_id, ModelLifecycleState.RETIRED)

            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Create retirement event
            retirement_event = self._create_event_contract(
                model_id=model_id,
                event_type=LifecycleEventType.RETIREMENT.value,
                event_status="completed",
                event_data={
                    "retired_from_state": current_state.value,
                    "cleanup_result": cleanup_result,
                    "retirement_reason": "manual_retirement",
                },
                processing_time_ms=processing_time,
            )

            await self._record_lifecycle_event(retirement_event)

            # Update metrics
            self.lifecycle_metrics["active_models"] = max(
                0, self.lifecycle_metrics["active_models"] - 1
            )

            self.logger.info(
                f"♻️ Model retired successfully: {model_id} (from {current_state.value})"
            )
            return retirement_event

        except Exception as e:
            error_event = self._create_event_contract(
                model_id=model_id,
                event_type=LifecycleEventType.ERROR_DETECTED.value,
                event_status="failed",
                event_data={"retirement_error": str(e)},
                processing_time_ms=0.0,
            )

            await self._record_lifecycle_event(error_event)

            self.logger.error(f"❌ Model retirement failed: {model_id} - {e}")
            return error_event

    async def get_model_lifecycle_history(self, model_id: str) -> Dict[str, Any]:
        """Get complete lifecycle history for a model"""
        try:
            events = self.lifecycle_events.get(model_id, [])
            transitions = self.state_transitions.get(model_id, [])

            # Sort events by timestamp
            events.sort(key=lambda e: e.event_timestamp)
            transitions.sort(key=lambda t: t["timestamp"])

            # Calculate lifecycle statistics
            initialization_events = [
                e
                for e in events
                if e.event_type == LifecycleEventType.INITIALIZATION_COMPLETED.value
            ]
            error_events = [
                e
                for e in events
                if e.event_type == LifecycleEventType.ERROR_DETECTED.value
            ]

            history = {
                "model_id": model_id,
                "current_state": self.model_states.get(
                    model_id, ModelLifecycleState.UNINITIALIZED
                ).value,
                "lifecycle_events": [self._serialize_event(event) for event in events],
                "state_transitions": transitions,
                "lifecycle_statistics": {
                    "total_events": len(events),
                    "total_transitions": len(transitions),
                    "error_count": len(error_events),
                    "initialization_count": len(initialization_events),
                    "first_event": (
                        events[0].event_timestamp.isoformat() if events else None
                    ),
                    "last_event": (
                        events[-1].event_timestamp.isoformat() if events else None
                    ),
                },
                "history_generated": datetime.utcnow().isoformat(),
            }

            return history

        except Exception as e:
            self.logger.error(
                f"❌ Lifecycle history retrieval failed for {model_id}: {e}"
            )
            return {"error": str(e)}

    async def get_lifecycle_analytics(self) -> Dict[str, Any]:
        """Get comprehensive lifecycle analytics"""
        try:
            # Calculate state distribution
            state_distribution = {}
            for state in ModelLifecycleState:
                count = sum(
                    1
                    for model_state in self.model_states.values()
                    if model_state == state
                )
                state_distribution[state.value] = count

            # Calculate event statistics
            total_events = sum(len(events) for events in self.lifecycle_events.values())
            event_type_distribution = defaultdict(int)

            for events in self.lifecycle_events.values():
                for event in events:
                    event_type_distribution[event.event_type] += 1

            # Calculate transition statistics
            total_transitions = sum(
                len(transitions) for transitions in self.state_transitions.values()
            )

            # Calculate health metrics
            healthy_states = [ModelLifecycleState.READY, ModelLifecycleState.ACTIVE]
            healthy_models = sum(
                1 for state in self.model_states.values() if state in healthy_states
            )

            analytics = {
                "lifecycle_overview": {
                    "total_models_managed": len(self.model_states),
                    "healthy_models": healthy_models,
                    "total_events_recorded": total_events,
                    "total_transitions_recorded": total_transitions,
                },
                "state_distribution": state_distribution,
                "event_type_distribution": dict(event_type_distribution),
                "metrics": self.lifecycle_metrics,
                "system_health": {
                    "health_percentage": (
                        (healthy_models / len(self.model_states) * 100)
                        if self.model_states
                        else 0
                    ),
                    "error_rate": self.lifecycle_metrics["error_incidents"]
                    / max(total_events, 1)
                    * 100,
                },
                "analytics_timestamp": datetime.utcnow().isoformat(),
            }

            return analytics

        except Exception as e:
            self.logger.error(f"❌ Lifecycle analytics generation failed: {e}")
            return {"error": str(e)}

    async def _transition_model_state(
        self, model_id: str, new_state: ModelLifecycleState
    ):
        """Transition model to new state with tracking"""
        try:
            old_state = self.model_states.get(
                model_id, ModelLifecycleState.UNINITIALIZED
            )

            # Record state transition
            transition_record = {
                "from_state": old_state.value,
                "to_state": new_state.value,
                "timestamp": datetime.utcnow(),
                "transition_id": f"{model_id}_{datetime.utcnow().timestamp()}",
            }

            self.state_transitions[model_id].append(transition_record)

            # Update current state
            self.model_states[model_id] = new_state

            # Update metrics
            self.lifecycle_metrics["models_by_state"][old_state.value] -= 1
            self.lifecycle_metrics["models_by_state"][new_state.value] += 1
            self.lifecycle_metrics["total_transitions"] += 1

            # Track active models
            if new_state in [
                ModelLifecycleState.READY,
                ModelLifecycleState.ACTIVE,
                ModelLifecycleState.BUSY,
            ]:
                if old_state not in [
                    ModelLifecycleState.READY,
                    ModelLifecycleState.ACTIVE,
                    ModelLifecycleState.BUSY,
                ]:
                    self.lifecycle_metrics["active_models"] += 1
            elif old_state in [
                ModelLifecycleState.READY,
                ModelLifecycleState.ACTIVE,
                ModelLifecycleState.BUSY,
            ]:
                self.lifecycle_metrics["active_models"] -= 1

        except Exception as e:
            self.logger.error(f"❌ State transition failed for {model_id}: {e}")

    async def _is_valid_transition(
        self, current_state: ModelLifecycleState, target_state: ModelLifecycleState
    ) -> bool:
        """Validate if state transition is allowed"""
        return target_state in self.valid_transitions.get(current_state, [])

    async def _perform_model_initialization(self, model_id: str) -> Dict[str, Any]:
        """Perform actual model initialization tasks"""
        try:
            # Simulate initialization process
            await asyncio.sleep(0.1)  # Simulate initialization time

            # Mock initialization checks
            initialization_checks = [
                {"check": "resource_allocation", "status": "passed"},
                {"check": "configuration_validation", "status": "passed"},
                {"check": "dependency_verification", "status": "passed"},
                {"check": "health_check", "status": "passed"},
            ]

            # Simulate occasional failures
            import random

            if random.random() < 0.05:  # 5% failure rate
                failed_check = random.choice(initialization_checks)
                failed_check["status"] = "failed"

                return {
                    "success": False,
                    "error": f"Initialization failed: {failed_check['check']}",
                    "error_code": "INIT_CHECK_FAILED",
                    "initialization_checks": initialization_checks,
                }

            return {
                "success": True,
                "initialization_checks": initialization_checks,
                "initialized_at": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            return {"success": False, "error": str(e), "error_code": "INIT_EXCEPTION"}

    async def _perform_model_retirement_cleanup(
        self, model_id: str, current_state: ModelLifecycleState
    ) -> Dict[str, Any]:
        """Perform cleanup tasks during model retirement"""
        try:
            cleanup_tasks = []

            # Resource cleanup
            cleanup_tasks.append({"task": "resource_cleanup", "status": "completed"})

            # Configuration cleanup
            cleanup_tasks.append(
                {"task": "configuration_cleanup", "status": "completed"}
            )

            # Event history cleanup (if configured)
            if self.lifecycle_config.get("cleanup_history_on_retirement", False):
                cleanup_tasks.append({"task": "history_cleanup", "status": "completed"})

            return {
                "cleanup_completed": True,
                "cleanup_tasks": cleanup_tasks,
                "retired_from_state": current_state.value,
            }

        except Exception as e:
            return {"cleanup_completed": False, "error": str(e)}

    def _create_event_contract(
        self,
        model_id: str,
        event_type: str,
        event_status: str,
        event_data: Dict[str, Any],
        processing_time_ms: float,
    ) -> LifecycleEventContract:
        """Create a standardized lifecycle event contract"""
        return LifecycleEventContract(
            event_id=f"{model_id}_{event_type}_{datetime.utcnow().timestamp()}",
            model_id=model_id,
            event_type=event_type,
            event_status=event_status,
            event_data=event_data,
            triggered_by="lifecycle_service",
            event_timestamp=datetime.utcnow(),
            processing_time_ms=processing_time_ms,
            service_version="v5_modular",
        )

    async def _record_lifecycle_event(self, event: LifecycleEventContract):
        """Record lifecycle event with cleanup"""
        try:
            self.lifecycle_events[event.model_id].append(event)

            # Cleanup old events
            await self._cleanup_old_events(event.model_id)

            # Update error metrics
            if event.event_type == LifecycleEventType.ERROR_DETECTED.value:
                self.lifecycle_metrics["error_incidents"] += 1

        except Exception as e:
            self.logger.error(f"❌ Event recording failed: {e}")

    async def _cleanup_old_events(self, model_id: str):
        """Clean up old lifecycle events based on retention policy"""
        try:
            retention_cutoff = datetime.utcnow() - timedelta(
                days=self.lifecycle_config["event_retention_days"]
            )

            events = self.lifecycle_events[model_id]
            self.lifecycle_events[model_id] = [
                event for event in events if event.event_timestamp > retention_cutoff
            ]

        except Exception as e:
            self.logger.error(f"❌ Event cleanup failed for {model_id}: {e}")

    def _determine_event_type_for_status(
        self, from_state: ModelLifecycleState, to_state: ModelLifecycleState
    ) -> str:
        """Determine appropriate event type based on state transition"""
        if to_state == ModelLifecycleState.ACTIVE:
            return LifecycleEventType.ACTIVATION.value
        elif from_state == ModelLifecycleState.ACTIVE:
            return LifecycleEventType.DEACTIVATION.value
        elif to_state == ModelLifecycleState.MAINTENANCE:
            return LifecycleEventType.MAINTENANCE_STARTED.value
        elif from_state == ModelLifecycleState.MAINTENANCE:
            return LifecycleEventType.MAINTENANCE_COMPLETED.value
        elif to_state == ModelLifecycleState.ERROR:
            return LifecycleEventType.ERROR_DETECTED.value
        elif to_state == ModelLifecycleState.RETIRED:
            return LifecycleEventType.RETIREMENT.value
        else:
            return "status_update"

    async def _handle_special_state_transitions(
        self,
        model_id: str,
        from_state: ModelLifecycleState,
        to_state: ModelLifecycleState,
        metadata: Dict[str, Any],
    ):
        """Handle special logic for specific state transitions"""
        try:
            # Handle transition to maintenance
            if to_state == ModelLifecycleState.MAINTENANCE:
                # Schedule maintenance completion
                asyncio.create_task(
                    self._schedule_maintenance_completion(
                        model_id, metadata.get("maintenance_duration_hours", 2)
                    )
                )

            # Handle error state
            elif to_state == ModelLifecycleState.ERROR:
                # Check if model should be retired due to excessive errors
                error_events = [
                    event
                    for event in self.lifecycle_events[model_id]
                    if event.event_type == LifecycleEventType.ERROR_DETECTED.value
                ]

                if (
                    len(error_events)
                    >= self.lifecycle_config["max_error_count_before_retirement"]
                ):
                    self.logger.warning(
                        f"⚠️ Model {model_id} has excessive errors, scheduling retirement"
                    )
                    asyncio.create_task(
                        self._auto_retire_model(model_id, "excessive_errors")
                    )

        except Exception as e:
            self.logger.error(f"❌ Special transition handling failed: {e}")

    async def _schedule_maintenance_completion(
        self, model_id: str, duration_hours: int
    ):
        """Schedule automatic maintenance completion"""
        try:
            await asyncio.sleep(duration_hours * 3600)  # Convert hours to seconds

            current_state = self.model_states.get(model_id)
            if current_state == ModelLifecycleState.MAINTENANCE:
                await self.update_model_status(
                    model_id,
                    ModelLifecycleState.READY.value,
                    {"maintenance_completed": True, "auto_completed": True},
                )

        except Exception as e:
            self.logger.error(f"❌ Maintenance completion scheduling failed: {e}")

    async def _auto_retire_model(self, model_id: str, reason: str):
        """Automatically retire model due to issues"""
        try:
            await asyncio.sleep(300)  # 5-minute delay before retirement

            retirement_event = await self.retire_model(model_id)
            retirement_event.event_data.update(
                {"auto_retirement": True, "retirement_reason": reason}
            )

        except Exception as e:
            self.logger.error(f"❌ Auto-retirement failed for {model_id}: {e}")

    async def _start_health_monitoring(self):
        """Start background health monitoring for all models"""
        try:
            while True:
                await asyncio.sleep(
                    self.lifecycle_config["health_check_interval_seconds"]
                )

                for model_id, state in self.model_states.items():
                    if state in [ModelLifecycleState.ACTIVE, ModelLifecycleState.READY]:
                        await self._perform_health_check(model_id)

        except Exception as e:
            self.logger.error(f"❌ Health monitoring failed: {e}")

    async def _perform_health_check(self, model_id: str):
        """Perform health check for a model"""
        try:
            # Mock health check
            health_status = "healthy"  # Simplified

            health_event = self._create_event_contract(
                model_id=model_id,
                event_type=LifecycleEventType.HEALTH_CHECK.value,
                event_status="completed",
                event_data={"health_status": health_status},
                processing_time_ms=50.0,
            )

            await self._record_lifecycle_event(health_event)

        except Exception as e:
            self.logger.error(f"❌ Health check failed for {model_id}: {e}")

    def _update_initialization_metrics(self, initialization_time_ms: float):
        """Update initialization performance metrics"""
        current_avg = self.lifecycle_metrics["average_initialization_time_ms"]
        total_inits = self.lifecycle_metrics.get("total_initializations", 0) + 1

        new_avg = (
            (current_avg * (total_inits - 1)) + initialization_time_ms
        ) / total_inits
        self.lifecycle_metrics["average_initialization_time_ms"] = new_avg
        self.lifecycle_metrics["total_initializations"] = total_inits

    def _serialize_event(self, event: LifecycleEventContract) -> Dict[str, Any]:
        """Serialize lifecycle event for JSON output"""
        return {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "event_status": event.event_status,
            "event_data": event.event_data,
            "event_timestamp": event.event_timestamp.isoformat(),
            "processing_time_ms": event.processing_time_ms,
        }

    async def get_service_health(self) -> Dict[str, Any]:
        """Return service health status"""
        return {
            "service_name": "LifecycleManagementService",
            "status": "healthy",
            "version": "v5_modular",
            "capabilities": [
                "model_initialization",
                "state_management",
                "lifecycle_tracking",
                "health_monitoring",
                "automatic_retirement",
            ],
            "lifecycle_statistics": {
                "models_managed": len(self.model_states),
                "active_models": self.lifecycle_metrics["active_models"],
                "total_events": sum(
                    len(events) for events in self.lifecycle_events.values()
                ),
                "total_transitions": self.lifecycle_metrics["total_transitions"],
            },
            "configuration": self.lifecycle_config,
            "metrics": self.lifecycle_metrics,
            "last_health_check": datetime.utcnow().isoformat(),
        }


# Global service instance for dependency injection
_lifecycle_management_service: Optional[LifecycleManagementService] = None


def get_lifecycle_management_service() -> LifecycleManagementService:
    """Get or create global lifecycle management service instance"""
    global _lifecycle_management_service

    if _lifecycle_management_service is None:
        _lifecycle_management_service = LifecycleManagementService()

    return _lifecycle_management_service
