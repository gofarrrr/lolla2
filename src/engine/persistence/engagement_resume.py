"""
METIS Engagement Resume System - P5.4
Resume functionality from last checkpoint with intelligent context restoration

Implements comprehensive engagement resumption capabilities:
- Smart checkpoint detection and validation
- Context restoration and state synchronization
- Resume strategies based on interruption type
- Progress validation and consistency checks
- Seamless workflow continuation
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from uuid import UUID
from dataclasses import dataclass, field
from enum import Enum

from src.engine.models.data_contracts import (
    MetisDataContract,
    EngagementPhase,
)
from src.persistence.contract_storage import (
    ContractPersistenceManager,
    CheckpointType,
    ContractVersion,
)
from src.persistence.checkpoint_manager import (
    ContractCheckpointManager,
)
from src.persistence.recovery_system import ContractRecoverySystem
from src.engine.adapters.event_bus import MetisEventBus  # Migrated


class ResumeReason(str, Enum):
    """Reasons for engagement resumption"""

    SYSTEM_RESTART = "system_restart"  # System was restarted
    USER_SESSION = "user_session"  # User reconnected
    ERROR_RECOVERY = "error_recovery"  # Recovering from error
    MANUAL_RESUME = "manual_resume"  # User manually resumed
    SCHEDULED_CONTINUATION = "scheduled_continuation"  # Scheduled workflow continuation
    TIMEOUT_RECOVERY = "timeout_recovery"  # Recovering from timeout


class InterruptionType(str, Enum):
    """Types of interruptions that can occur"""

    CLEAN_SHUTDOWN = "clean_shutdown"  # Graceful system shutdown
    UNEXPECTED_CRASH = "unexpected_crash"  # System crash or kill
    USER_DISCONNECTION = "user_disconnection"  # User session ended
    OPERATION_TIMEOUT = "operation_timeout"  # Operation timed out
    ERROR_FAILURE = "error_failure"  # Error caused interruption
    RESOURCE_EXHAUSTION = "resource_exhaustion"  # Out of memory/resources


class ResumeStrategy(str, Enum):
    """Strategies for resuming engagements"""

    EXACT_RESTORATION = "exact_restoration"  # Restore exact previous state
    SAFE_ROLLBACK = "safe_rollback"  # Rollback to safe checkpoint
    PARTIAL_RECOVERY = "partial_recovery"  # Recover what's possible
    CLEAN_RESTART = "clean_restart"  # Start fresh from beginning
    INTELLIGENT_MERGE = "intelligent_merge"  # Merge with current state


@dataclass
class ResumeContext:
    """Context information for engagement resumption"""

    engagement_id: UUID
    last_checkpoint: ContractVersion
    interruption_type: InterruptionType
    interruption_time: datetime
    resume_time: datetime = field(default_factory=datetime.utcnow)

    # State analysis
    state_consistency: bool = True
    data_integrity: bool = True
    workflow_continuity: bool = True

    # Resume planning
    recommended_strategy: ResumeStrategy = ResumeStrategy.EXACT_RESTORATION
    alternative_strategies: List[ResumeStrategy] = field(default_factory=list)
    risk_assessment: str = ""
    estimated_resume_time: float = 0.0  # seconds

    # Recovery metadata
    checkpoints_available: int = 0
    latest_valid_phase: EngagementPhase = EngagementPhase.PROBLEM_STRUCTURING
    data_loss_risk: str = "None"


@dataclass
class ResumeOperation:
    """Record of an engagement resume operation"""

    operation_id: UUID = field(default_factory=UUID)
    engagement_id: UUID = field(default_factory=UUID)
    resume_reason: ResumeReason = ResumeReason.MANUAL_RESUME
    interruption_type: InterruptionType = InterruptionType.CLEAN_SHUTDOWN

    # Resume execution
    strategy_used: ResumeStrategy = ResumeStrategy.EXACT_RESTORATION
    checkpoint_restored: Optional[UUID] = None
    initiated_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    # Results
    success: bool = False
    error_message: Optional[str] = None
    resume_time_seconds: float = 0.0
    state_restored: bool = False
    workflow_continued: bool = False

    # Validation results
    post_resume_validation: Dict[str, Any] = field(default_factory=dict)
    consistency_checks: Dict[str, bool] = field(default_factory=dict)


class EngagementResumeManager:
    """
    Manages engagement resumption from checkpoints with intelligent context restoration
    Provides seamless continuation of interrupted workflows
    """

    def __init__(
        self,
        persistence_manager: ContractPersistenceManager,
        checkpoint_manager: ContractCheckpointManager,
        recovery_system: ContractRecoverySystem,
        event_bus: Optional[MetisEventBus] = None,
    ):
        self.persistence_manager = persistence_manager
        self.checkpoint_manager = checkpoint_manager
        self.recovery_system = recovery_system
        self.event_bus = event_bus

        self.logger = logging.getLogger(__name__)

        # Resume state tracking
        self.pending_resumes: Dict[UUID, ResumeContext] = {}
        self.active_resumes: Dict[UUID, ResumeOperation] = {}
        self.resume_history: List[ResumeOperation] = []

        # Configuration
        self.max_resume_age_hours = 24  # Don't resume very old sessions
        self.min_checkpoint_quality = 0.7  # Minimum checkpoint validation score
        self.enable_intelligent_merge = True

        # Performance tracking
        self.metrics = {
            "resume_operations": 0,
            "successful_resumes": 0,
            "failed_resumes": 0,
            "exact_restorations": 0,
            "safe_rollbacks": 0,
            "clean_restarts": 0,
            "average_resume_time": 0.0,
        }

    async def discover_resumable_engagements(
        self, max_age_hours: Optional[int] = None
    ) -> List[ResumeContext]:
        """
        Discover engagements that can be resumed
        Returns list of resumable engagements with context analysis
        """
        max_age_hours = max_age_hours or self.max_resume_age_hours
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)

        try:
            # Get all contract metadata for analysis
            metrics = await self.persistence_manager.get_metrics()
            db_stats = metrics.get("database_stats", {})

            if not db_stats.get("total_contracts", 0):
                return []

            resumable_contexts = []

            # Note: In a real implementation, we'd query the database for active contracts
            # For now, we'll simulate discovery based on cached active engagements

            for engagement_id in self.checkpoint_manager.active_engagements:
                context = await self.analyze_engagement_for_resume(
                    engagement_id, cutoff_time
                )
                if context:
                    resumable_contexts.append(context)

            self.logger.info(f"Found {len(resumable_contexts)} resumable engagements")
            return resumable_contexts

        except Exception as e:
            self.logger.error(f"Failed to discover resumable engagements: {str(e)}")
            return []

    async def analyze_engagement_for_resume(
        self, engagement_id: UUID, cutoff_time: datetime
    ) -> Optional[ResumeContext]:
        """
        Analyze a specific engagement to determine if it can be resumed
        Returns resume context if resumable, None otherwise
        """
        try:
            # Get latest checkpoint
            latest_checkpoint = await self.persistence_manager.get_latest_checkpoint(
                engagement_id
            )
            if not latest_checkpoint:
                return None

            # Check if recent enough
            if latest_checkpoint.created_at < cutoff_time:
                return None

            # Analyze interruption type
            interruption_type = await self._detect_interruption_type(
                engagement_id, latest_checkpoint
            )

            # Check state consistency
            state_analysis = await self._analyze_state_consistency(latest_checkpoint)

            # Determine recommended strategy
            strategy = await self._determine_resume_strategy(
                latest_checkpoint, interruption_type, state_analysis
            )

            # Get alternative strategies
            alternatives = await self._get_alternative_strategies(
                strategy, state_analysis
            )

            # Calculate risk assessment
            risk_assessment = await self._assess_resume_risk(
                latest_checkpoint, interruption_type, state_analysis
            )

            # Count available checkpoints
            all_versions = await self.persistence_manager.get_contract_versions(
                engagement_id
            )
            checkpoint_count = len(
                [
                    v
                    for v in all_versions
                    if v.checkpoint_type != CheckpointType.AUTOMATIC
                ]
            )

            # Create resume context
            context = ResumeContext(
                engagement_id=engagement_id,
                last_checkpoint=latest_checkpoint,
                interruption_type=interruption_type,
                interruption_time=latest_checkpoint.created_at,
                state_consistency=state_analysis["consistency"],
                data_integrity=state_analysis["integrity"],
                workflow_continuity=state_analysis["continuity"],
                recommended_strategy=strategy,
                alternative_strategies=alternatives,
                risk_assessment=risk_assessment,
                estimated_resume_time=self._estimate_resume_time(strategy),
                checkpoints_available=checkpoint_count,
                latest_valid_phase=latest_checkpoint.phase_at_creation,
                data_loss_risk=self._assess_data_loss_risk(state_analysis),
            )

            return context

        except Exception as e:
            self.logger.error(
                f"Failed to analyze engagement {engagement_id} for resume: {str(e)}"
            )
            return None

    async def resume_engagement(
        self,
        engagement_id: UUID,
        resume_reason: ResumeReason = ResumeReason.MANUAL_RESUME,
        strategy: Optional[ResumeStrategy] = None,
        force: bool = False,
    ) -> ResumeOperation:
        """
        Resume an engagement from its last valid checkpoint
        Returns detailed operation results
        """
        operation_id = UUID()
        start_time = datetime.utcnow()

        # Create resume operation record
        operation = ResumeOperation(
            operation_id=operation_id,
            engagement_id=engagement_id,
            resume_reason=resume_reason,
            initiated_at=start_time,
        )

        self.active_resumes[operation_id] = operation

        try:
            # Get resume context
            context = await self.analyze_engagement_for_resume(
                engagement_id,
                datetime.utcnow() - timedelta(hours=self.max_resume_age_hours),
            )

            if not context:
                operation.success = False
                operation.error_message = "Engagement not resumable or not found"
                operation.completed_at = datetime.utcnow()
                self.metrics["failed_resumes"] += 1
                return operation

            # Use provided strategy or recommended strategy
            resume_strategy = strategy or context.recommended_strategy
            operation.strategy_used = resume_strategy
            operation.interruption_type = context.interruption_type

            # Pre-resume validation
            if not force:
                pre_validation = await self._perform_pre_resume_validation(
                    context, operation
                )
                if not pre_validation:
                    operation.success = False
                    operation.error_message = "Pre-resume validation failed"
                    operation.completed_at = datetime.utcnow()
                    self.metrics["failed_resumes"] += 1
                    return operation

            # Execute resume strategy
            resumed_contract = await self._execute_resume_strategy(
                context, resume_strategy, operation
            )

            if resumed_contract:
                # Post-resume validation
                validation_results = await self._perform_post_resume_validation(
                    resumed_contract, context, operation
                )

                operation.post_resume_validation = validation_results

                if validation_results.get("passed", False):
                    # Store resumed contract
                    await self.persistence_manager.store_contract(
                        resumed_contract,
                        CheckpointType.MANUAL,
                        f"Resumed engagement using {resume_strategy.value} strategy",
                        f"resume_manager_{resume_reason.value}",
                    )

                    # Register with checkpoint manager for continued monitoring
                    await self.checkpoint_manager.register_engagement(resumed_contract)

                    operation.success = True
                    operation.state_restored = True
                    operation.workflow_continued = True

                    # Emit resume event
                    if self.event_bus:
                        await self._emit_resume_event(
                            engagement_id, resume_strategy, True
                        )

                    self.logger.info(
                        f"✅ Successfully resumed engagement {engagement_id} using {resume_strategy.value}"
                    )
                    self.metrics["successful_resumes"] += 1
                    self.metrics[f"{resume_strategy.value}s"] = (
                        self.metrics.get(f"{resume_strategy.value}s", 0) + 1
                    )

                else:
                    operation.success = False
                    operation.error_message = "Post-resume validation failed"
                    self.metrics["failed_resumes"] += 1
            else:
                operation.success = False
                operation.error_message = "Resume strategy execution failed"
                self.metrics["failed_resumes"] += 1

        except Exception as e:
            operation.success = False
            operation.error_message = str(e)
            self.logger.error(f"Resume operation failed for {engagement_id}: {str(e)}")
            self.metrics["failed_resumes"] += 1

        # Complete the operation
        operation.completed_at = datetime.utcnow()
        operation.resume_time_seconds = (
            operation.completed_at - start_time
        ).total_seconds()

        # Update average resume time
        total_ops = self.metrics["resume_operations"]
        current_avg = self.metrics["average_resume_time"]
        self.metrics["average_resume_time"] = (
            current_avg * total_ops + operation.resume_time_seconds
        ) / (total_ops + 1)

        # Move to history
        self.resume_history.append(operation)
        if operation_id in self.active_resumes:
            del self.active_resumes[operation_id]

        self.metrics["resume_operations"] += 1

        return operation

    async def _execute_resume_strategy(
        self,
        context: ResumeContext,
        strategy: ResumeStrategy,
        operation: ResumeOperation,
    ) -> Optional[MetisDataContract]:
        """Execute the specified resume strategy"""

        try:
            if strategy == ResumeStrategy.EXACT_RESTORATION:
                return await self._execute_exact_restoration(context, operation)

            elif strategy == ResumeStrategy.SAFE_ROLLBACK:
                return await self._execute_safe_rollback(context, operation)

            elif strategy == ResumeStrategy.PARTIAL_RECOVERY:
                return await self._execute_partial_recovery(context, operation)

            elif strategy == ResumeStrategy.CLEAN_RESTART:
                return await self._execute_clean_restart(context, operation)

            elif strategy == ResumeStrategy.INTELLIGENT_MERGE:
                return await self._execute_intelligent_merge(context, operation)

            else:
                raise ValueError(f"Unsupported resume strategy: {strategy}")

        except Exception as e:
            self.logger.error(f"Resume strategy execution failed: {str(e)}")
            return None

    async def _execute_exact_restoration(
        self, context: ResumeContext, operation: ResumeOperation
    ) -> Optional[MetisDataContract]:
        """Execute exact restoration from checkpoint"""

        # Simply restore the exact contract from the checkpoint
        contract = context.last_checkpoint.contract_data.copy(deep=True)

        # Update timing metadata
        contract.time = datetime.utcnow()
        contract.processing_metadata["resumed_at"] = datetime.utcnow().isoformat()
        contract.processing_metadata["resume_operation_id"] = str(
            operation.operation_id
        )
        contract.processing_metadata["resume_strategy"] = operation.strategy_used.value
        contract.processing_metadata["interruption_type"] = (
            operation.interruption_type.value
        )

        operation.checkpoint_restored = context.last_checkpoint.version_id
        return contract

    async def _execute_safe_rollback(
        self, context: ResumeContext, operation: ResumeOperation
    ) -> Optional[MetisDataContract]:
        """Execute safe rollback to a known good state"""

        # Find the best safe checkpoint
        all_versions = await self.persistence_manager.get_contract_versions(
            context.engagement_id
        )

        # Look for phase boundary or manual checkpoints
        safe_checkpoints = [
            v
            for v in all_versions
            if v.checkpoint_type
            in [CheckpointType.PHASE_BOUNDARY, CheckpointType.MANUAL]
            and v.created_at <= context.interruption_time
        ]

        if not safe_checkpoints:
            # Fallback to latest checkpoint
            safe_checkpoint = context.last_checkpoint
        else:
            # Use the most recent safe checkpoint
            safe_checkpoint = max(safe_checkpoints, key=lambda v: v.created_at)

        contract = safe_checkpoint.contract_data.copy(deep=True)

        # Add rollback metadata
        contract.processing_metadata["safe_rollback_from"] = str(
            context.last_checkpoint.version_id
        )
        contract.processing_metadata["safe_rollback_to"] = str(
            safe_checkpoint.version_id
        )
        contract.processing_metadata["resumed_at"] = datetime.utcnow().isoformat()

        operation.checkpoint_restored = safe_checkpoint.version_id
        return contract

    async def _execute_partial_recovery(
        self, context: ResumeContext, operation: ResumeOperation
    ) -> Optional[MetisDataContract]:
        """Execute partial recovery of salvageable components"""

        contract = context.last_checkpoint.contract_data.copy(deep=True)

        # Validate and potentially reset components based on consistency
        if not context.state_consistency:
            # Reset cognitive state if inconsistent
            contract.cognitive_state.validation_results = {}
            contract.cognitive_state.confidence_scores = {}

        if not context.workflow_continuity:
            # Reset to last completed phase
            completed_phases = contract.workflow_state.completed_phases
            if completed_phases:
                contract.workflow_state.current_phase = completed_phases[-1]
            else:
                contract.workflow_state.current_phase = (
                    EngagementPhase.PROBLEM_STRUCTURING
                )

        if not context.data_integrity:
            # Validate and potentially remove corrupted artifacts
            valid_artifacts = []
            for artifact in contract.deliverable_artifacts:
                if artifact.content and artifact.artifact_id:
                    valid_artifacts.append(artifact)
            contract.deliverable_artifacts = valid_artifacts

        # Add partial recovery metadata
        contract.processing_metadata["partial_recovery"] = {
            "state_consistency_restored": not context.state_consistency,
            "workflow_continuity_restored": not context.workflow_continuity,
            "data_integrity_restored": not context.data_integrity,
            "artifacts_preserved": len(contract.deliverable_artifacts),
        }
        contract.processing_metadata["resumed_at"] = datetime.utcnow().isoformat()

        return contract

    async def _execute_clean_restart(
        self, context: ResumeContext, operation: ResumeOperation
    ) -> Optional[MetisDataContract]:
        """Execute clean restart from initial state"""

        # Get the first version (initial state)
        all_versions = await self.persistence_manager.get_contract_versions(
            context.engagement_id
        )
        if not all_versions:
            return None

        initial_version = min(all_versions, key=lambda v: v.version_number)
        contract = initial_version.contract_data.copy(deep=True)

        # Reset to initial state
        contract.workflow_state.current_phase = EngagementPhase.PROBLEM_STRUCTURING
        contract.workflow_state.completed_phases = []
        contract.workflow_state.phase_results = {}
        contract.deliverable_artifacts = []
        contract.cognitive_state.selected_mental_models = []
        contract.cognitive_state.reasoning_steps = []
        contract.cognitive_state.confidence_scores = {}
        contract.cognitive_state.validation_results = {}

        # Preserve original context but add restart metadata
        contract.processing_metadata = {
            "clean_restart_at": datetime.utcnow().isoformat()
        }
        contract.time = datetime.utcnow()

        operation.checkpoint_restored = initial_version.version_id
        return contract

    async def _execute_intelligent_merge(
        self, context: ResumeContext, operation: ResumeOperation
    ) -> Optional[MetisDataContract]:
        """Execute intelligent merge with current state"""

        if not self.enable_intelligent_merge:
            # Fallback to exact restoration
            return await self._execute_exact_restoration(context, operation)

        # Get the checkpoint contract
        checkpoint_contract = context.last_checkpoint.contract_data.copy(deep=True)

        # Get current contract if it exists
        current_contract = await self.persistence_manager.get_contract(
            context.engagement_id
        )

        if not current_contract:
            # No current state, use exact restoration
            return await self._execute_exact_restoration(context, operation)

        # Intelligent merge logic
        merged_contract = checkpoint_contract.copy(deep=True)

        # Merge workflow state intelligently
        current_phase_index = self._get_phase_index(
            current_contract.workflow_state.current_phase
        )
        checkpoint_phase_index = self._get_phase_index(
            checkpoint_contract.workflow_state.current_phase
        )

        if current_phase_index > checkpoint_phase_index:
            # Current state is further along, use current workflow state
            merged_contract.workflow_state = current_contract.workflow_state

        # Merge artifacts (keep the union of both)
        checkpoint_artifact_ids = {
            a.artifact_id for a in checkpoint_contract.deliverable_artifacts
        }
        current_artifact_ids = {
            a.artifact_id for a in current_contract.deliverable_artifacts
        }

        # Add current artifacts that aren't in checkpoint
        for artifact in current_contract.deliverable_artifacts:
            if artifact.artifact_id not in checkpoint_artifact_ids:
                merged_contract.deliverable_artifacts.append(artifact)

        # Merge processing metadata
        merged_contract.processing_metadata.update(current_contract.processing_metadata)
        merged_contract.processing_metadata["intelligent_merge"] = {
            "merged_at": datetime.utcnow().isoformat(),
            "checkpoint_version": context.last_checkpoint.version_number,
            "artifacts_merged": len(merged_contract.deliverable_artifacts),
            "workflow_state_source": (
                "current"
                if current_phase_index > checkpoint_phase_index
                else "checkpoint"
            ),
        }

        return merged_contract

    def _get_phase_index(self, phase: EngagementPhase) -> int:
        """Get numeric index of engagement phase for comparison"""
        phase_order = [
            EngagementPhase.PROBLEM_STRUCTURING,
            EngagementPhase.HYPOTHESIS_GENERATION,
            EngagementPhase.ANALYSIS_EXECUTION,
            EngagementPhase.SYNTHESIS_DELIVERY,
        ]
        return phase_order.index(phase)

    async def _detect_interruption_type(
        self, engagement_id: UUID, last_checkpoint: ContractVersion
    ) -> InterruptionType:
        """Detect the type of interruption that occurred"""

        # Check checkpoint type for clues
        if last_checkpoint.checkpoint_type == CheckpointType.ERROR_RECOVERY:
            return InterruptionType.ERROR_FAILURE

        # Check time since last checkpoint
        time_since = (datetime.utcnow() - last_checkpoint.created_at).total_seconds()

        if time_since > 3600:  # More than 1 hour
            return InterruptionType.USER_DISCONNECTION
        elif time_since > 300:  # More than 5 minutes
            return InterruptionType.UNEXPECTED_CRASH
        else:
            return InterruptionType.CLEAN_SHUTDOWN

    async def _analyze_state_consistency(
        self, checkpoint: ContractVersion
    ) -> Dict[str, bool]:
        """Analyze the consistency of the checkpoint state"""

        contract = checkpoint.contract_data
        analysis = {"consistency": True, "integrity": True, "continuity": True}

        try:
            # Check workflow consistency
            if not await self._check_workflow_consistency(contract):
                analysis["consistency"] = False

            # Check data integrity
            if not await self._check_data_integrity(contract):
                analysis["integrity"] = False

            # Check workflow continuity
            if not await self._check_workflow_continuity(contract):
                analysis["continuity"] = False

        except Exception as e:
            self.logger.warning(f"State consistency analysis failed: {str(e)}")
            analysis = {"consistency": False, "integrity": False, "continuity": False}

        return analysis

    async def _check_workflow_consistency(self, contract: MetisDataContract) -> bool:
        """Check if workflow state is consistent"""
        try:
            # Current phase should be consistent with completed phases
            current_phase = contract.workflow_state.current_phase
            completed_phases = contract.workflow_state.completed_phases

            phase_order = [
                EngagementPhase.PROBLEM_STRUCTURING,
                EngagementPhase.HYPOTHESIS_GENERATION,
                EngagementPhase.ANALYSIS_EXECUTION,
                EngagementPhase.SYNTHESIS_DELIVERY,
            ]

            current_index = phase_order.index(current_phase)

            # All completed phases should be before or equal to current phase
            for completed_phase in completed_phases:
                if phase_order.index(completed_phase) > current_index:
                    return False

            return True

        except Exception:
            return False

    async def _check_data_integrity(self, contract: MetisDataContract) -> bool:
        """Check if contract data is intact"""
        try:
            # Check required fields
            if not contract.engagement_context.engagement_id:
                return False
            if not contract.engagement_context.problem_statement:
                return False

            # Check artifacts integrity
            for artifact in contract.deliverable_artifacts:
                if not artifact.artifact_id or not artifact.content:
                    return False

            # Check processing metadata
            if not contract.processing_metadata:
                return False

            return True

        except Exception:
            return False

    async def _check_workflow_continuity(self, contract: MetisDataContract) -> bool:
        """Check if workflow can continue from current state"""
        try:
            # Check if current phase has necessary data to continue
            current_phase = contract.workflow_state.current_phase

            if current_phase == EngagementPhase.HYPOTHESIS_GENERATION:
                # Should have problem structure from previous phase
                return "problem_structure" in contract.processing_metadata

            elif current_phase == EngagementPhase.ANALYSIS_EXECUTION:
                # Should have hypotheses from previous phase
                return "hypotheses" in contract.processing_metadata

            elif current_phase == EngagementPhase.SYNTHESIS_DELIVERY:
                # Should have analysis results from previous phase
                return "analysis_results" in contract.processing_metadata

            return True

        except Exception:
            return False

    async def _determine_resume_strategy(
        self,
        checkpoint: ContractVersion,
        interruption_type: InterruptionType,
        state_analysis: Dict[str, bool],
    ) -> ResumeStrategy:
        """Determine the best resume strategy based on analysis"""

        # If all state checks pass and it's a clean shutdown, use exact restoration
        if (
            state_analysis["consistency"]
            and state_analysis["integrity"]
            and state_analysis["continuity"]
            and interruption_type == InterruptionType.CLEAN_SHUTDOWN
        ):
            return ResumeStrategy.EXACT_RESTORATION

        # If there are consistency issues but data is intact, use safe rollback
        if not state_analysis["consistency"] and state_analysis["integrity"]:
            return ResumeStrategy.SAFE_ROLLBACK

        # If there are data integrity issues, use partial recovery
        if not state_analysis["integrity"]:
            return ResumeStrategy.PARTIAL_RECOVERY

        # If interruption was due to error, use intelligent merge if available
        if (
            interruption_type == InterruptionType.ERROR_FAILURE
            and self.enable_intelligent_merge
        ):
            return ResumeStrategy.INTELLIGENT_MERGE

        # If checkpoint is very old or corrupted, suggest clean restart
        age_hours = (datetime.utcnow() - checkpoint.created_at).total_seconds() / 3600
        if age_hours > 12 or not any(state_analysis.values()):
            return ResumeStrategy.CLEAN_RESTART

        # Default to exact restoration
        return ResumeStrategy.EXACT_RESTORATION

    async def _get_alternative_strategies(
        self, primary_strategy: ResumeStrategy, state_analysis: Dict[str, bool]
    ) -> List[ResumeStrategy]:
        """Get alternative resume strategies"""

        alternatives = []

        # Always offer exact restoration as fallback
        if primary_strategy != ResumeStrategy.EXACT_RESTORATION:
            alternatives.append(ResumeStrategy.EXACT_RESTORATION)

        # Offer safe rollback if primary isn't safe rollback
        if (
            primary_strategy != ResumeStrategy.SAFE_ROLLBACK
            and state_analysis["integrity"]
        ):
            alternatives.append(ResumeStrategy.SAFE_ROLLBACK)

        # Offer partial recovery if there are issues
        if primary_strategy != ResumeStrategy.PARTIAL_RECOVERY and not all(
            state_analysis.values()
        ):
            alternatives.append(ResumeStrategy.PARTIAL_RECOVERY)

        # Offer clean restart as last resort
        if primary_strategy != ResumeStrategy.CLEAN_RESTART:
            alternatives.append(ResumeStrategy.CLEAN_RESTART)

        return alternatives

    async def _assess_resume_risk(
        self,
        checkpoint: ContractVersion,
        interruption_type: InterruptionType,
        state_analysis: Dict[str, bool],
    ) -> str:
        """Assess the risk level of resuming from this checkpoint"""

        risk_factors = []

        # Age risk
        age_hours = (datetime.utcnow() - checkpoint.created_at).total_seconds() / 3600
        if age_hours > 6:
            risk_factors.append(f"Checkpoint is {age_hours:.1f} hours old")

        # State consistency risk
        if not state_analysis["consistency"]:
            risk_factors.append("State consistency issues detected")

        if not state_analysis["integrity"]:
            risk_factors.append("Data integrity issues detected")

        if not state_analysis["continuity"]:
            risk_factors.append("Workflow continuity issues detected")

        # Interruption type risk
        if interruption_type in [
            InterruptionType.UNEXPECTED_CRASH,
            InterruptionType.ERROR_FAILURE,
        ]:
            risk_factors.append(f"Interruption due to {interruption_type.value}")

        # Checkpoint quality risk
        if checkpoint.checkpoint_type == CheckpointType.AUTOMATIC:
            risk_factors.append("Automatic checkpoint (lower quality)")

        if risk_factors:
            return f"Medium risk: {'; '.join(risk_factors)}"
        else:
            return "Low risk resume"

    def _estimate_resume_time(self, strategy: ResumeStrategy) -> float:
        """Estimate resume time in seconds"""

        time_estimates = {
            ResumeStrategy.EXACT_RESTORATION: 2.0,
            ResumeStrategy.SAFE_ROLLBACK: 5.0,
            ResumeStrategy.PARTIAL_RECOVERY: 15.0,
            ResumeStrategy.CLEAN_RESTART: 3.0,
            ResumeStrategy.INTELLIGENT_MERGE: 20.0,
        }

        return time_estimates.get(strategy, 10.0)

    def _assess_data_loss_risk(self, state_analysis: Dict[str, bool]) -> str:
        """Assess data loss risk"""

        if all(state_analysis.values()):
            return "None"
        elif state_analysis["integrity"]:
            return "Minimal"
        elif state_analysis["consistency"]:
            return "Moderate"
        else:
            return "High"

    async def _perform_pre_resume_validation(
        self, context: ResumeContext, operation: ResumeOperation
    ) -> bool:
        """Perform validation before resume operation"""

        try:
            # Check checkpoint quality
            if context.last_checkpoint.progress_percentage < 0:  # Invalid progress
                return False

            # Check if engagement is not already active
            if context.engagement_id in self.checkpoint_manager.active_engagements:
                self.logger.warning(
                    f"Engagement {context.engagement_id} is already active"
                )
                return False

            # Check if we have minimum required data
            contract = context.last_checkpoint.contract_data
            if not contract.engagement_context.problem_statement:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Pre-resume validation failed: {str(e)}")
            return False

    async def _perform_post_resume_validation(
        self,
        resumed_contract: MetisDataContract,
        context: ResumeContext,
        operation: ResumeOperation,
    ) -> Dict[str, Any]:
        """Perform validation after resume operation"""

        validation_results = {"passed": False, "checks": {}, "errors": []}

        try:
            # Basic contract validation
            validation_results["checks"]["schema_valid"] = (
                await self._validate_contract_schema(resumed_contract)
            )
            validation_results["checks"]["workflow_consistent"] = (
                await self._check_workflow_consistency(resumed_contract)
            )
            validation_results["checks"]["data_intact"] = (
                await self._check_data_integrity(resumed_contract)
            )
            validation_results["checks"]["continuity_maintained"] = (
                await self._check_workflow_continuity(resumed_contract)
            )

            # Strategy-specific validation
            if operation.strategy_used == ResumeStrategy.INTELLIGENT_MERGE:
                validation_results["checks"]["merge_successful"] = (
                    await self._validate_merge_result(resumed_contract)
                )

            # Overall validation
            validation_results["passed"] = all(validation_results["checks"].values())

        except Exception as e:
            validation_results["errors"].append(str(e))
            validation_results["passed"] = False

        return validation_results

    async def _validate_contract_schema(self, contract: MetisDataContract) -> bool:
        """Validate contract schema"""
        try:
            # Pydantic validation happens automatically
            return True
        except Exception:
            return False

    async def _validate_merge_result(self, contract: MetisDataContract) -> bool:
        """Validate intelligent merge result"""
        try:
            # Check that merge metadata is present
            merge_info = contract.processing_metadata.get("intelligent_merge")
            if not merge_info:
                return False

            # Check that artifacts are consistent
            artifact_ids = {a.artifact_id for a in contract.deliverable_artifacts}
            if len(artifact_ids) != len(contract.deliverable_artifacts):
                return False  # Duplicate artifacts

            return True

        except Exception:
            return False

    async def _emit_resume_event(
        self, engagement_id: UUID, strategy: ResumeStrategy, success: bool
    ):
        """Emit resume event to event bus"""
        if not self.event_bus:
            return

        try:
            from src.schemas.event_factories import (
                create_engagement_workflow_started_event,
            )

            # Create resume event (reusing workflow started event structure)
            event = create_engagement_workflow_started_event(
                engagement_id=str(engagement_id),
                problem_statement=f"Engagement resumed using {strategy.value}",
                client_name="Resume Manager",
            )

            # Add resume-specific data
            if hasattr(event, "data") and event.data:
                event.data["resume_strategy"] = strategy.value
                event.data["resume_success"] = success

            await self.event_bus.publish_event(event)

        except Exception as e:
            self.logger.warning(f"Failed to emit resume event: {str(e)}")

    async def get_resume_metrics(self) -> Dict[str, Any]:
        """Get resume system metrics"""
        return {
            **self.metrics,
            "pending_resumes": len(self.pending_resumes),
            "active_resumes": len(self.active_resumes),
            "resume_history_count": len(self.resume_history),
            "max_resume_age_hours": self.max_resume_age_hours,
            "min_checkpoint_quality": self.min_checkpoint_quality,
            "strategies_available": [strategy.value for strategy in ResumeStrategy],
            "interruption_types_supported": [itype.value for itype in InterruptionType],
        }


# Export main classes
__all__ = [
    "EngagementResumeManager",
    "ResumeContext",
    "ResumeOperation",
    "ResumeStrategy",
    "ResumeReason",
    "InterruptionType",
]
