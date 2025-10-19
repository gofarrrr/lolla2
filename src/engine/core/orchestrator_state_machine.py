"""
Operation Synapse: State Machine Orchestrator with Redis/Supabase Hybrid Persistence

This module implements the architectural upgrade from linear pipeline to resilient
state machine orchestration, providing intelligent failure handling and full audit trail.

Key Features:
- Redis for low-latency state transitions
- Supabase for durable audit and recovery
- Conservative failure handling with controlled retries
- Full state capture for debugging and recovery
- Urgency-aware retry behavior
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from uuid import uuid4

try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from src.engine.models.data_contracts import MetisDataContract


class OrchestrationState(Enum):
    """States for orchestration state machine"""

    INITIALIZING = "initializing"
    ANALYZING_PROBLEM = "analyzing_problem"
    EXECUTING_PHASE = "executing_phase"
    PHASE_RETRY = "phase_retry"
    HANDLING_ERROR = "handling_error"
    TERMINAL_FAILURE = "terminal_failure"
    COMPLETED = "completed"


@dataclass
class StateTransition:
    """Represents a state transition for audit purposes"""

    engagement_id: str
    from_state: Optional[str]
    to_state: str
    phase_name: Optional[str]
    error_details: Optional[Dict[str, Any]]
    retry_count: int
    timestamp: float
    metadata: Dict[str, Any]


@dataclass
class OrchestrationMetrics:
    """Performance metrics for orchestration execution"""

    total_phases: int
    completed_phases: int
    failed_phases: int
    total_retries: int
    execution_time_ms: float
    state_transitions: int


class OrchestrationStateMachine:
    """
    State machine orchestrator implementing hybrid Redis/Supabase persistence.

    Architecture:
    - Redis: Fast state transitions and current state tracking
    - Supabase: Durable audit trail and recovery data
    - Conservative: Fail fast with limited intelligent retries
    - Observable: Complete transition logging for debugging
    """

    def __init__(
        self, redis_client: Optional[redis.Redis] = None, supabase_client=None
    ):
        self.logger = logging.getLogger(__name__)
        self.redis = redis_client
        self.supabase = supabase_client

        # Configuration per architectural decisions
        self.max_retries = 2  # Default, can be adjusted per urgency
        self.redis_ttl = 3600  # 1 hour TTL for Redis state
        self.retry_backoff_base = 2  # Exponential backoff

        # Performance tracking
        self.metrics: Dict[str, OrchestrationMetrics] = {}

        if not REDIS_AVAILABLE and redis_client:
            self.logger.warning("Redis client provided but redis.asyncio not available")

        self.logger.info("🎯 Operation Synapse: State Machine Orchestrator initialized")

    async def initialize_engagement(
        self, engagement_id: str, urgency: str = "normal"
    ) -> None:
        """Initialize orchestration state for new engagement"""

        # Adjust retry behavior based on urgency
        if urgency == "critical":
            self.max_retries = 1
        elif urgency == "low":
            self.max_retries = 3
        else:
            self.max_retries = 2

        # Initialize metrics
        self.metrics[engagement_id] = OrchestrationMetrics(
            total_phases=4,  # Standard 4-phase workflow
            completed_phases=0,
            failed_phases=0,
            total_retries=0,
            execution_time_ms=0,
            state_transitions=0,
        )

        # Initialize state
        await self.transition_to_state(
            engagement_id=engagement_id,
            new_state=OrchestrationState.INITIALIZING,
            metadata={
                "urgency": urgency,
                "max_retries": self.max_retries,
                "initialized_at": datetime.utcnow().isoformat(),
            },
        )

    async def transition_to_state(
        self,
        engagement_id: str,
        new_state: OrchestrationState,
        metadata: Dict[str, Any],
    ) -> None:
        """Execute state transition with Redis for speed, Supabase for durability"""

        start_time = time.time()

        # Get current state for transition logging
        current_state = await self.get_current_state(engagement_id)

        # Create state data for Redis
        state_data = {
            "state": new_state.value,
            "metadata": json.dumps(metadata),
            "timestamp": time.time(),
            "retry_count": metadata.get("retry_count", 0),
            "phase": metadata.get("phase", ""),
            "urgency": metadata.get("urgency", "normal"),
        }

        try:
            # 1. Update Redis immediately for speed
            if self.redis:
                state_key = f"orchestration:{engagement_id}"
                await self.redis.hset(state_key, mapping=state_data)
                await self.redis.expire(state_key, self.redis_ttl)

            # 2. Update metrics
            if engagement_id in self.metrics:
                self.metrics[engagement_id].state_transitions += 1

            # 3. Log transition
            transition = StateTransition(
                engagement_id=engagement_id,
                from_state=current_state.get("state") if current_state else None,
                to_state=new_state.value,
                phase_name=metadata.get("phase"),
                error_details=metadata.get("error_details"),
                retry_count=metadata.get("retry_count", 0),
                timestamp=time.time(),
                metadata=metadata,
            )

            self.logger.info(
                f"🎯 State Transition: {engagement_id} "
                f"{transition.from_state} → {new_state.value} "
                f"(Phase: {transition.phase_name}, Retry: {transition.retry_count})"
            )

            # 4. For terminal states, persist to Supabase asynchronously
            if new_state in [
                OrchestrationState.COMPLETED,
                OrchestrationState.TERMINAL_FAILURE,
            ]:
                asyncio.create_task(
                    self._persist_final_state(engagement_id, transition)
                )

        except Exception as e:
            self.logger.error(f"❌ State transition failed for {engagement_id}: {e}")
            raise

        # Track transition performance
        transition_time = (time.time() - start_time) * 1000
        self.logger.debug(f"State transition completed in {transition_time:.2f}ms")

    async def get_current_state(self, engagement_id: str) -> Dict[str, Any]:
        """Retrieve current orchestration state"""

        if not self.redis:
            return {}

        try:
            state_key = f"orchestration:{engagement_id}"
            state_data = await self.redis.hgetall(state_key)

            if not state_data:
                return {}

            # Decode Redis bytes to strings
            decoded_state = {}
            for key, value in state_data.items():
                if isinstance(key, bytes):
                    key = key.decode("utf-8")
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                decoded_state[key] = value

            # Parse metadata JSON
            if "metadata" in decoded_state:
                try:
                    decoded_state["metadata"] = json.loads(decoded_state["metadata"])
                except json.JSONDecodeError:
                    decoded_state["metadata"] = {}

            return decoded_state

        except Exception as e:
            self.logger.error(
                f"❌ Failed to get current state for {engagement_id}: {e}"
            )
            return {}

    async def handle_phase_failure(
        self, engagement_id: str, phase: str, error: Exception, retry_count: int
    ) -> str:
        """Conservative failure handling: retry or terminal failure"""

        # Update metrics
        if engagement_id in self.metrics:
            self.metrics[engagement_id].total_retries += 1

        error_details = {
            "error_message": str(error),
            "error_type": type(error).__name__,
            "phase": phase,
            "retry_count": retry_count,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if retry_count < self.max_retries:
            # Attempt retry with exponential backoff
            await self.transition_to_state(
                engagement_id=engagement_id,
                new_state=OrchestrationState.PHASE_RETRY,
                metadata={
                    "phase": phase,
                    "retry_count": retry_count + 1,
                    "error_details": error_details,
                    "next_action": "retry",
                },
            )

            self.logger.warning(
                f"⚠️ Phase {phase} failed, retrying ({retry_count + 1}/{self.max_retries}): {error}"
            )
            return "retry"

        else:
            # Terminal failure - capture full state
            current_state = await self.get_current_state(engagement_id)

            # Update metrics
            if engagement_id in self.metrics:
                self.metrics[engagement_id].failed_phases += 1

            await self.transition_to_state(
                engagement_id=engagement_id,
                new_state=OrchestrationState.TERMINAL_FAILURE,
                metadata={
                    "phase": phase,
                    "retry_count": retry_count,
                    "error_details": error_details,
                    "state_at_failure": current_state,
                    "final_action": "terminal_failure",
                },
            )

            self.logger.error(
                f"❌ Terminal failure in phase {phase} after {retry_count} retries: {error}"
            )
            return "terminal_failure"

    async def complete_phase(self, engagement_id: str, phase: str) -> None:
        """Mark phase as successfully completed"""

        # Update metrics
        if engagement_id in self.metrics:
            self.metrics[engagement_id].completed_phases += 1

        await self.transition_to_state(
            engagement_id=engagement_id,
            new_state=OrchestrationState.EXECUTING_PHASE,
            metadata={
                "phase": phase,
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat(),
            },
        )

        self.logger.info(f"✅ Phase {phase} completed successfully for {engagement_id}")

    async def complete_engagement(self, engagement_id: str) -> OrchestrationMetrics:
        """Mark entire engagement as completed and return metrics"""

        # Final metrics update
        if engagement_id in self.metrics:
            metrics = self.metrics[engagement_id]
            metrics.execution_time_ms = time.time() * 1000  # Convert to ms
        else:
            metrics = OrchestrationMetrics(0, 0, 0, 0, 0, 0)

        await self.transition_to_state(
            engagement_id=engagement_id,
            new_state=OrchestrationState.COMPLETED,
            metadata={
                "completion_status": "success",
                "completed_at": datetime.utcnow().isoformat(),
                "metrics": asdict(metrics),
            },
        )

        self.logger.info(
            f"🎉 Engagement {engagement_id} completed successfully "
            f"({metrics.completed_phases}/{metrics.total_phases} phases, "
            f"{metrics.total_retries} retries)"
        )

        return metrics

    async def capture_state_snapshot(
        self, contract: MetisDataContract
    ) -> Dict[str, Any]:
        """Capture comprehensive state snapshot for debugging"""

        return {
            "engagement_id": str(contract.engagement_context.engagement_id),
            "current_phase": contract.processing_metadata.get("current_phase"),
            "reasoning_steps_count": len(contract.cognitive_state.reasoning_steps),
            "processing_metadata": contract.processing_metadata,
            "timestamp": datetime.utcnow().isoformat(),
            "state_capture_id": str(uuid4()),
        }

    async def _persist_final_state(
        self, engagement_id: str, transition: StateTransition
    ) -> None:
        """Persist final state to Supabase for durable audit trail"""

        if not self.supabase:
            return

        try:
            # Create audit record
            audit_record = {
                "engagement_id": engagement_id,
                "final_state": transition.to_state,
                "transition_history": asdict(transition),
                "metrics": asdict(
                    self.metrics.get(
                        engagement_id, OrchestrationMetrics(0, 0, 0, 0, 0, 0)
                    )
                ),
                "created_at": datetime.utcnow().isoformat(),
            }

            # Insert into orchestration_audit table (to be created in migrations)
            result = (
                await self.supabase.table("orchestration_audit")
                .insert(audit_record)
                .execute()
            )

            if result.data:
                self.logger.info(
                    f"📝 Final state persisted to Supabase for {engagement_id}"
                )
            else:
                self.logger.warning(
                    f"⚠️ Failed to persist final state for {engagement_id}"
                )

        except Exception as e:
            self.logger.error(
                f"❌ Error persisting final state for {engagement_id}: {e}"
            )

    async def cleanup_engagement(self, engagement_id: str) -> None:
        """Clean up resources for completed engagement"""

        # Remove from metrics tracking
        if engagement_id in self.metrics:
            del self.metrics[engagement_id]

        # Redis cleanup happens automatically via TTL
        self.logger.debug(f"🧹 Cleaned up resources for engagement {engagement_id}")

    def get_engagement_metrics(
        self, engagement_id: str
    ) -> Optional[OrchestrationMetrics]:
        """Get current metrics for engagement"""
        return self.metrics.get(engagement_id)


# Factory function for dependency injection
async def create_orchestrator_state_machine(
    redis_url: Optional[str] = None, supabase_client=None
):
    """Factory function to create configured state machine"""

    redis_client = None
    if redis_url and REDIS_AVAILABLE:
        try:
            redis_client = redis.from_url(redis_url)
            # Test connection
            await redis_client.ping()
        except Exception as e:
            logging.getLogger(__name__).warning(f"Redis connection failed: {e}")
            redis_client = None

    return OrchestrationStateMachine(redis_client, supabase_client)
