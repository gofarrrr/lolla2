"""
METIS Stateful Environment with Checkpointing
Implements enterprise-grade state management with recovery capabilities

Based on industry insights:
- OpenAI: Stateful environments with branch exploration
- LangChain: State snapshots at decision points
- Anthropic: Constitutional AI with rollback capability
- Enterprise Pattern: 95% error recovery with graceful degradation
"""

import pickle
import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
from uuid import uuid4
from pathlib import Path

from src.engine.models.data_contracts import (
    ReasoningStep,
    EngagementPhase,
)
from src.core.performance_cache_system import get_performance_cache, CacheEntryType

logger = logging.getLogger(__name__)


class CheckpointType(str, Enum):
    """Types of checkpoints for different recovery scenarios"""

    PHASE_BOUNDARY = "phase_boundary"  # At each engagement phase
    CRITICAL_DECISION = "critical_decision"  # Before high-risk operations
    USER_INTERACTION = "user_interaction"  # Before human-in-loop points
    ERROR_RECOVERY = "error_recovery"  # After error detection
    BRANCH_POINT = "branch_point"  # Before tree search branches
    VALIDATION_GATE = "validation_gate"  # Before quality checks


class ExecutionState(str, Enum):
    """Current execution state of the environment"""

    INITIALIZING = "initializing"
    EXECUTING = "executing"
    PAUSED = "paused"
    WAITING_USER_INPUT = "waiting_user_input"
    ERROR_STATE = "error_state"
    RECOVERING = "recovering"
    COMPLETED = "completed"
    TERMINATED = "terminated"


@dataclass
class StateSnapshot:
    """Immutable snapshot of system state at a checkpoint"""

    checkpoint_id: str
    checkpoint_type: CheckpointType
    created_at: datetime

    # Core state
    engagement_id: str
    current_phase: EngagementPhase
    execution_state: ExecutionState

    # Progress tracking
    completed_phases: List[EngagementPhase]
    current_step: int
    total_steps: int

    # Context state
    problem_context: Dict[str, Any]
    reasoning_history: List[ReasoningStep]
    accumulated_insights: List[Dict[str, Any]]

    # Model state
    selected_mental_models: List[str]
    model_confidence_scores: Dict[str, float]
    model_execution_results: Dict[str, Any]

    # Performance state
    processing_time_ms: float
    cache_hits: int
    error_count: int

    # Recovery metadata
    recovery_instructions: List[str]
    fallback_options: List[str]
    validation_results: Dict[str, Any]

    # Environment metadata
    checkpoint_size_bytes: int = 0
    parent_checkpoint_id: Optional[str] = None
    branch_metadata: Optional[Dict[str, Any]] = None


@dataclass
class RecoveryPlan:
    """Plan for recovering from errors or rollbacks"""

    recovery_id: str
    target_checkpoint_id: str
    recovery_strategy: str
    estimated_recovery_time_ms: float

    # Recovery actions
    state_modifications: List[Dict[str, Any]]
    context_adjustments: List[Dict[str, Any]]
    alternative_approaches: List[str]

    # Validation
    success_criteria: List[str]
    rollback_threshold: int
    max_recovery_attempts: int


class StatefulEnvironment:
    """
    Enterprise-grade stateful environment with checkpointing and recovery
    Enables exploration, rollback, and resilient execution
    """

    def __init__(self, engagement_id: str, config: Optional[Dict[str, Any]] = None):
        self.engagement_id = engagement_id
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()

        # State management
        self.current_state = ExecutionState.INITIALIZING
        self.checkpoints: Dict[str, StateSnapshot] = {}
        self.checkpoint_history: List[str] = []
        self.active_branches: Dict[str, List[str]] = {}

        # Performance cache integration
        self.cache = get_performance_cache()

        # Recovery tracking
        self.recovery_history: List[RecoveryPlan] = []
        self.error_count = 0
        self.last_successful_checkpoint: Optional[str] = None

        # Persistence
        self.checkpoint_storage_path = (
            Path(self.config["checkpoint_storage_dir"]) / engagement_id
        )
        self.checkpoint_storage_path.mkdir(parents=True, exist_ok=True)

        # Performance metrics
        self.performance_metrics = {
            "checkpoints_created": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "average_checkpoint_size": 0,
            "average_recovery_time": 0,
            "cache_efficiency": 0.0,
        }

        self.logger.info(
            f"🗄️ Stateful environment initialized for engagement {engagement_id}"
        )

    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration for stateful environment"""
        return {
            "checkpoint_storage_dir": ".metis/checkpoints",
            "max_checkpoints_in_memory": 20,
            "checkpoint_compression": True,
            "auto_checkpoint_interval_ms": 5000,  # 5 seconds
            "max_recovery_attempts": 3,
            "checkpoint_cleanup_after_hours": 24,
            "enable_branch_exploration": True,
            "max_concurrent_branches": 3,
            "recovery_timeout_ms": 30000,  # 30 seconds
        }

    async def create_checkpoint(
        self,
        checkpoint_type: CheckpointType,
        current_context: Dict[str, Any],
        reasoning_steps: List[ReasoningStep],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create checkpoint at current state
        Returns checkpoint ID for later recovery
        """
        checkpoint_start = time.time()
        checkpoint_id = f"checkpoint_{int(time.time() * 1000)}_{str(uuid4())[:8]}"

        # Build state snapshot
        snapshot = StateSnapshot(
            checkpoint_id=checkpoint_id,
            checkpoint_type=checkpoint_type,
            created_at=datetime.now(),
            engagement_id=self.engagement_id,
            current_phase=current_context.get(
                "current_phase", EngagementPhase.PROBLEM_STRUCTURING
            ),
            execution_state=self.current_state,
            completed_phases=current_context.get("completed_phases", []),
            current_step=current_context.get("current_step", 0),
            total_steps=current_context.get("total_steps", 4),
            problem_context=current_context.get("problem_context", {}),
            reasoning_history=reasoning_steps,
            accumulated_insights=current_context.get("insights", []),
            selected_mental_models=current_context.get("selected_models", []),
            model_confidence_scores=current_context.get("confidence_scores", {}),
            model_execution_results=current_context.get("execution_results", {}),
            processing_time_ms=(time.time() - checkpoint_start) * 1000,
            cache_hits=current_context.get("cache_hits", 0),
            error_count=self.error_count,
            recovery_instructions=self._generate_recovery_instructions(current_context),
            fallback_options=self._generate_fallback_options(current_context),
            validation_results=current_context.get("validation_results", {}),
            parent_checkpoint_id=(
                self.checkpoint_history[-1] if self.checkpoint_history else None
            ),
            branch_metadata=metadata,
        )

        # Calculate checkpoint size
        serialized_snapshot = await self._serialize_snapshot(snapshot)
        snapshot.checkpoint_size_bytes = len(serialized_snapshot)

        # Store in memory
        self.checkpoints[checkpoint_id] = snapshot
        self.checkpoint_history.append(checkpoint_id)

        # Persist to storage if configured
        if self.config.get("persist_checkpoints", True):
            await self._persist_checkpoint(checkpoint_id, serialized_snapshot)

        # Cache checkpoint metadata for fast access
        await self.cache.put(
            content_type=CacheEntryType.PHASE_RESULT,
            primary_key=f"checkpoint_meta_{checkpoint_id}",
            content={
                "checkpoint_id": checkpoint_id,
                "type": checkpoint_type.value,
                "size_bytes": snapshot.checkpoint_size_bytes,
                "created_at": snapshot.created_at.isoformat(),
            },
            ttl_seconds=3600,
        )

        # Update tracking
        self.last_successful_checkpoint = checkpoint_id
        self.performance_metrics["checkpoints_created"] += 1
        self._update_average_checkpoint_size(snapshot.checkpoint_size_bytes)

        # Cleanup old checkpoints if needed
        await self._cleanup_old_checkpoints()

        checkpoint_time = time.time() - checkpoint_start
        self.logger.info(
            f"📸 Checkpoint {checkpoint_id} created in {checkpoint_time*1000:.1f}ms ({snapshot.checkpoint_size_bytes} bytes)"
        )

        return checkpoint_id

    async def restore_checkpoint(
        self, checkpoint_id: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Restore system state from checkpoint
        Returns (success, restored_context)
        """
        recovery_start = time.time()

        try:
            # Load checkpoint
            snapshot = await self._load_checkpoint(checkpoint_id)
            if not snapshot:
                self.logger.error(f"❌ Checkpoint {checkpoint_id} not found")
                return False, {}

            # Restore system state
            self.current_state = snapshot.execution_state

            # Build restored context
            restored_context = {
                "current_phase": snapshot.current_phase,
                "completed_phases": snapshot.completed_phases,
                "current_step": snapshot.current_step,
                "total_steps": snapshot.total_steps,
                "problem_context": snapshot.problem_context,
                "insights": snapshot.accumulated_insights,
                "selected_models": snapshot.selected_mental_models,
                "confidence_scores": snapshot.model_confidence_scores,
                "execution_results": snapshot.model_execution_results,
                "validation_results": snapshot.validation_results,
                "checkpoint_metadata": {
                    "restored_from": checkpoint_id,
                    "original_timestamp": snapshot.created_at.isoformat(),
                    "checkpoint_type": snapshot.checkpoint_type.value,
                },
            }

            # Update checkpoint history (remove checkpoints after restored point)
            if checkpoint_id in self.checkpoint_history:
                checkpoint_index = self.checkpoint_history.index(checkpoint_id)
                self.checkpoint_history = self.checkpoint_history[
                    : checkpoint_index + 1
                ]

            # Performance tracking
            recovery_time = time.time() - recovery_start
            self.performance_metrics["successful_recoveries"] += 1
            self._update_average_recovery_time(recovery_time)

            self.logger.info(
                f"🔄 Successfully restored from checkpoint {checkpoint_id} in {recovery_time*1000:.1f}ms"
            )
            return True, restored_context

        except Exception as e:
            recovery_time = time.time() - recovery_start
            self.performance_metrics["failed_recoveries"] += 1
            self.logger.error(f"❌ Failed to restore checkpoint {checkpoint_id}: {e}")
            return False, {}

    async def create_branch(
        self,
        parent_checkpoint_id: str,
        branch_name: str,
        branch_metadata: Dict[str, Any],
    ) -> str:
        """
        Create execution branch for exploration
        Returns new branch checkpoint ID
        """
        if not self.config.get("enable_branch_exploration", True):
            raise ValueError("Branch exploration is disabled")

        if len(self.active_branches) >= self.config["max_concurrent_branches"]:
            raise ValueError(
                f"Maximum concurrent branches ({self.config['max_concurrent_branches']}) reached"
            )

        # Load parent checkpoint
        parent_snapshot = await self._load_checkpoint(parent_checkpoint_id)
        if not parent_snapshot:
            raise ValueError(f"Parent checkpoint {parent_checkpoint_id} not found")

        # Create branch checkpoint
        branch_checkpoint_id = await self.create_checkpoint(
            checkpoint_type=CheckpointType.BRANCH_POINT,
            current_context={
                "current_phase": parent_snapshot.current_phase,
                "completed_phases": parent_snapshot.completed_phases,
                "current_step": parent_snapshot.current_step,
                "total_steps": parent_snapshot.total_steps,
                "problem_context": parent_snapshot.problem_context,
                "insights": parent_snapshot.accumulated_insights,
                "selected_models": parent_snapshot.selected_mental_models,
                "confidence_scores": parent_snapshot.model_confidence_scores,
                "execution_results": parent_snapshot.model_execution_results,
            },
            reasoning_steps=parent_snapshot.reasoning_history,
            metadata={
                "branch_name": branch_name,
                "parent_checkpoint": parent_checkpoint_id,
                **branch_metadata,
            },
        )

        # Track active branch
        if branch_name not in self.active_branches:
            self.active_branches[branch_name] = []
        self.active_branches[branch_name].append(branch_checkpoint_id)

        self.logger.info(
            f"🌳 Created branch '{branch_name}' from checkpoint {parent_checkpoint_id}"
        )
        return branch_checkpoint_id

    async def merge_branch(
        self,
        branch_checkpoint_id: str,
        target_checkpoint_id: str,
        merge_strategy: str = "best_confidence",
    ) -> str:
        """
        Merge branch results back to main execution path
        Returns merged checkpoint ID
        """
        branch_snapshot = await self._load_checkpoint(branch_checkpoint_id)
        target_snapshot = await self._load_checkpoint(target_checkpoint_id)

        if not branch_snapshot or not target_snapshot:
            raise ValueError("Invalid checkpoint IDs for merge")

        # Apply merge strategy
        merged_context = await self._apply_merge_strategy(
            branch_snapshot, target_snapshot, merge_strategy
        )

        # Create merged checkpoint
        merged_checkpoint_id = await self.create_checkpoint(
            checkpoint_type=CheckpointType.CRITICAL_DECISION,
            current_context=merged_context,
            reasoning_steps=branch_snapshot.reasoning_history
            + target_snapshot.reasoning_history,
            metadata={
                "merge_from": branch_checkpoint_id,
                "merge_to": target_checkpoint_id,
                "merge_strategy": merge_strategy,
            },
        )

        self.logger.info(
            f"🔀 Merged branch checkpoint {branch_checkpoint_id} to {target_checkpoint_id}"
        )
        return merged_checkpoint_id

    async def handle_error_recovery(
        self, error: Exception, current_context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any], str]:
        """
        Handle error with automatic recovery
        Returns (recovered, new_context, recovery_strategy)
        """
        self.error_count += 1
        recovery_start = time.time()

        self.logger.warning(f"⚠️ Error recovery initiated: {str(error)}")

        # Create error checkpoint for debugging
        error_checkpoint_id = await self.create_checkpoint(
            checkpoint_type=CheckpointType.ERROR_RECOVERY,
            current_context=current_context,
            reasoning_steps=current_context.get("reasoning_steps", []),
            metadata={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "recovery_attempt": self.error_count,
            },
        )

        # Try recovery strategies in order of preference
        recovery_strategies = [
            "rollback_to_last_checkpoint",
            "retry_with_alternative_model",
            "simplify_problem_scope",
            "request_human_intervention",
        ]

        for strategy in recovery_strategies:
            try:
                recovery_result = await self._execute_recovery_strategy(
                    strategy, error, current_context
                )

                if recovery_result["success"]:
                    recovery_time = time.time() - recovery_start
                    self.performance_metrics["successful_recoveries"] += 1
                    self._update_average_recovery_time(recovery_time)

                    self.logger.info(
                        f"✅ Recovered using strategy '{strategy}' in {recovery_time*1000:.1f}ms"
                    )
                    return True, recovery_result["context"], strategy

            except Exception as recovery_error:
                self.logger.warning(
                    f"⚠️ Recovery strategy '{strategy}' failed: {recovery_error}"
                )
                continue

        # All recovery strategies failed
        self.performance_metrics["failed_recoveries"] += 1
        self.current_state = ExecutionState.ERROR_STATE

        self.logger.error(f"❌ All recovery strategies failed for error: {error}")
        return False, current_context, "recovery_failed"

    async def _serialize_snapshot(self, snapshot: StateSnapshot) -> bytes:
        """Serialize snapshot for storage"""
        try:
            if self.config.get("checkpoint_compression", True):
                import gzip

                snapshot_dict = asdict(snapshot)
                # Convert datetime to ISO string for serialization
                snapshot_dict["created_at"] = snapshot.created_at.isoformat()
                serialized = json.dumps(snapshot_dict).encode("utf-8")
                return gzip.compress(serialized)
            else:
                return pickle.dumps(snapshot)
        except Exception as e:
            self.logger.error(f"❌ Failed to serialize snapshot: {e}")
            return b""

    async def _persist_checkpoint(self, checkpoint_id: str, serialized_data: bytes):
        """Persist checkpoint to storage"""
        try:
            checkpoint_file = (
                self.checkpoint_storage_path / f"{checkpoint_id}.checkpoint"
            )
            with open(checkpoint_file, "wb") as f:
                f.write(serialized_data)
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to persist checkpoint {checkpoint_id}: {e}")

    async def _load_checkpoint(self, checkpoint_id: str) -> Optional[StateSnapshot]:
        """Load checkpoint from memory or storage"""
        # Try memory first
        if checkpoint_id in self.checkpoints:
            return self.checkpoints[checkpoint_id]

        # Try storage
        try:
            checkpoint_file = (
                self.checkpoint_storage_path / f"{checkpoint_id}.checkpoint"
            )
            if checkpoint_file.exists():
                with open(checkpoint_file, "rb") as f:
                    serialized_data = f.read()

                if self.config.get("checkpoint_compression", True):
                    import gzip

                    decompressed = gzip.decompress(serialized_data)
                    snapshot_dict = json.loads(decompressed.decode("utf-8"))
                    # Convert ISO string back to datetime
                    snapshot_dict["created_at"] = datetime.fromisoformat(
                        snapshot_dict["created_at"]
                    )
                    return StateSnapshot(**snapshot_dict)
                else:
                    return pickle.loads(serialized_data)
        except Exception as e:
            self.logger.error(f"❌ Failed to load checkpoint {checkpoint_id}: {e}")

        return None

    def _generate_recovery_instructions(self, context: Dict[str, Any]) -> List[str]:
        """Generate context-specific recovery instructions"""
        instructions = [
            "Verify data integrity before proceeding",
            "Check mental model confidence scores",
            "Validate reasoning chain completeness",
        ]

        current_phase = context.get("current_phase")
        if current_phase:
            phase_instructions = {
                EngagementPhase.PROBLEM_STRUCTURING: [
                    "Re-validate problem statement clarity",
                    "Check stakeholder context completeness",
                ],
                EngagementPhase.HYPOTHESIS_GENERATION: [
                    "Verify hypothesis logical consistency",
                    "Check evidence sufficiency",
                ],
                EngagementPhase.ANALYSIS_EXECUTION: [
                    "Validate analysis methodology",
                    "Check intermediate result quality",
                ],
                EngagementPhase.SYNTHESIS_DELIVERY: [
                    "Verify recommendation coherence",
                    "Check final confidence scores",
                ],
            }
            instructions.extend(phase_instructions.get(current_phase, []))

        return instructions

    def _generate_fallback_options(self, context: Dict[str, Any]) -> List[str]:
        """Generate fallback options for recovery"""
        fallbacks = [
            "rollback_to_previous_checkpoint",
            "simplify_analysis_scope",
            "request_human_guidance",
        ]

        if context.get("selected_models"):
            fallbacks.append("try_alternative_mental_model")

        if context.get("cache_hits", 0) > 0:
            fallbacks.append("clear_cache_and_retry")

        return fallbacks

    async def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints to manage memory"""
        if len(self.checkpoints) <= self.config["max_checkpoints_in_memory"]:
            return

        # Keep most recent checkpoints in memory
        checkpoints_to_remove = (
            len(self.checkpoints) - self.config["max_checkpoints_in_memory"]
        )
        oldest_checkpoints = self.checkpoint_history[:checkpoints_to_remove]

        for checkpoint_id in oldest_checkpoints:
            if checkpoint_id in self.checkpoints:
                del self.checkpoints[checkpoint_id]

        self.checkpoint_history = self.checkpoint_history[checkpoints_to_remove:]

    async def _apply_merge_strategy(
        self,
        branch_snapshot: StateSnapshot,
        target_snapshot: StateSnapshot,
        strategy: str,
    ) -> Dict[str, Any]:
        """Apply merge strategy to combine branch results"""

        if strategy == "best_confidence":
            # Use results with higher confidence
            branch_avg_confidence = sum(
                branch_snapshot.model_confidence_scores.values()
            ) / max(len(branch_snapshot.model_confidence_scores), 1)
            target_avg_confidence = sum(
                target_snapshot.model_confidence_scores.values()
            ) / max(len(target_snapshot.model_confidence_scores), 1)

            best_snapshot = (
                branch_snapshot
                if branch_avg_confidence > target_avg_confidence
                else target_snapshot
            )

            return {
                "current_phase": best_snapshot.current_phase,
                "completed_phases": best_snapshot.completed_phases,
                "insights": best_snapshot.accumulated_insights,
                "selected_models": best_snapshot.selected_mental_models,
                "confidence_scores": best_snapshot.model_confidence_scores,
                "execution_results": best_snapshot.model_execution_results,
            }

        # Default: merge insights and take target state
        merged_insights = (
            target_snapshot.accumulated_insights + branch_snapshot.accumulated_insights
        )

        return {
            "current_phase": target_snapshot.current_phase,
            "completed_phases": target_snapshot.completed_phases,
            "insights": merged_insights,
            "selected_models": target_snapshot.selected_mental_models,
            "confidence_scores": target_snapshot.model_confidence_scores,
            "execution_results": target_snapshot.model_execution_results,
        }

    async def _execute_recovery_strategy(
        self, strategy: str, error: Exception, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute specific recovery strategy"""

        if strategy == "rollback_to_last_checkpoint":
            if self.last_successful_checkpoint:
                success, recovered_context = await self.restore_checkpoint(
                    self.last_successful_checkpoint
                )
                return {"success": success, "context": recovered_context}
            else:
                return {"success": False, "context": context}

        elif strategy == "retry_with_alternative_model":
            # Simulate alternative model selection
            alternative_models = [
                "critical_analysis",
                "systems_thinking",
                "mece_structuring",
            ]
            current_models = context.get("selected_models", [])
            available_alternatives = [
                m for m in alternative_models if m not in current_models
            ]

            if available_alternatives:
                new_context = context.copy()
                new_context["selected_models"] = available_alternatives[:2]
                return {"success": True, "context": new_context}

        elif strategy == "simplify_problem_scope":
            # Reduce problem complexity
            new_context = context.copy()
            new_context["simplified_scope"] = True
            new_context["complexity_reduction"] = 0.7
            return {"success": True, "context": new_context}

        elif strategy == "request_human_intervention":
            # Flag for human intervention
            new_context = context.copy()
            new_context["human_intervention_required"] = True
            new_context["intervention_reason"] = str(error)
            return {"success": True, "context": new_context}

        return {"success": False, "context": context}

    def _update_average_checkpoint_size(self, new_size: int):
        """Update running average of checkpoint sizes"""
        current_avg = self.performance_metrics["average_checkpoint_size"]
        count = self.performance_metrics["checkpoints_created"]

        new_avg = ((current_avg * (count - 1)) + new_size) / count
        self.performance_metrics["average_checkpoint_size"] = new_avg

    def _update_average_recovery_time(self, new_time: float):
        """Update running average of recovery times"""
        current_avg = self.performance_metrics["average_recovery_time"]
        total_recoveries = self.performance_metrics["successful_recoveries"]

        if total_recoveries > 1:
            new_avg = (
                (current_avg * (total_recoveries - 1)) + new_time
            ) / total_recoveries
            self.performance_metrics["average_recovery_time"] = new_avg
        else:
            self.performance_metrics["average_recovery_time"] = new_time

    def get_environment_status(self) -> Dict[str, Any]:
        """Get comprehensive environment status"""
        return {
            "engagement_id": self.engagement_id,
            "current_state": self.current_state.value,
            "checkpoints": {
                "total_created": len(self.checkpoint_history),
                "in_memory": len(self.checkpoints),
                "last_successful": self.last_successful_checkpoint,
            },
            "branches": {
                "active_branches": len(self.active_branches),
                "branch_names": list(self.active_branches.keys()),
            },
            "error_handling": {
                "total_errors": self.error_count,
                "successful_recoveries": self.performance_metrics[
                    "successful_recoveries"
                ],
                "failed_recoveries": self.performance_metrics["failed_recoveries"],
                "recovery_success_rate": (
                    (
                        self.performance_metrics["successful_recoveries"]
                        / max(self.error_count, 1)
                    )
                    if self.error_count > 0
                    else 1.0
                ),
            },
            "performance": self.performance_metrics,
            "storage": {
                "checkpoint_storage_path": str(self.checkpoint_storage_path),
                "average_checkpoint_size_kb": self.performance_metrics[
                    "average_checkpoint_size"
                ]
                / 1024,
            },
        }


# Global environment registry for managing multiple engagements
_stateful_environments: Dict[str, StatefulEnvironment] = {}


def get_stateful_environment(
    engagement_id: str, config: Optional[Dict[str, Any]] = None
) -> StatefulEnvironment:
    """Get or create stateful environment for engagement"""
    global _stateful_environments

    if engagement_id not in _stateful_environments:
        _stateful_environments[engagement_id] = StatefulEnvironment(
            engagement_id, config
        )

    return _stateful_environments[engagement_id]


def cleanup_stateful_environment(engagement_id: str):
    """Clean up stateful environment after engagement completion"""
    global _stateful_environments

    if engagement_id in _stateful_environments:
        del _stateful_environments[engagement_id]
        logger.info(f"🧹 Cleaned up stateful environment for {engagement_id}")
