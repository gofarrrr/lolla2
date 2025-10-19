"""
Immutable State Manager for Neural Lace Orchestrator
LangGraph-inspired immutable state handling for MetisDataContract

Based on LangGraph best practices:
- Never mutate existing state objects
- Always return new state objects with updates
- Validate state transitions for consistency
- Provide debugging and introspection capabilities
"""

import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from copy import deepcopy
from datetime import datetime
from uuid import uuid4

from src.engine.models.data_contracts import (
    MetisDataContract,
    EngagementPhase,
)


@dataclass
class StateTransition:
    """Record of a state transition for debugging and validation"""

    transition_id: str
    from_phase: Optional[EngagementPhase]
    to_phase: Optional[EngagementPhase]
    timestamp: str
    updates_applied: Dict[str, Any]
    transition_type: str  # "phase_completion", "data_update", "error_recovery"
    validation_passed: bool
    execution_time_ms: float


@dataclass
class StateValidationResult:
    """Result of state transition validation"""

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validation_details: Dict[str, Any] = field(default_factory=dict)


class ImmutableStateManager:
    """
    LangGraph-inspired immutable state management for MetisDataContract.

    Core Principles:
    1. Never mutate existing state objects - always return new objects
    2. Validate all state transitions maintain data integrity
    3. Provide comprehensive debugging and introspection
    4. Maintain rich audit trail of state evolution
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.state_transitions: List[StateTransition] = []
        self.validation_enabled = True

    def create_new_state(
        self,
        current_state: MetisDataContract,
        updates: Dict[str, Any],
        transition_type: str = "data_update",
    ) -> MetisDataContract:
        """
        Create new state object with updates - never mutate existing state.

        Args:
            current_state: Current MetisDataContract to update
            updates: Dictionary of updates to apply
            transition_type: Type of transition for auditing

        Returns:
            New MetisDataContract with updates applied
        """
        start_time = time.time()
        transition_id = str(uuid4())[:8]

        self.logger.debug(
            f"🔄 Creating new state (ID: {transition_id}): {transition_type}"
        )

        # Deep copy current state to ensure immutability
        new_state = deepcopy(current_state)

        # Record original phase for transition tracking
        original_phase = getattr(current_state.workflow_state, "current_phase", None)

        # Apply updates systematically
        self._apply_updates_to_state(new_state, updates)

        # Record state transition
        new_phase = getattr(new_state.workflow_state, "current_phase", None)
        execution_time = (time.time() - start_time) * 1000

        # Validate transition if enabled
        validation_passed = True
        if self.validation_enabled:
            validation_result = self.validate_state_transition(current_state, new_state)
            validation_passed = validation_result.valid

            if not validation_passed:
                self.logger.warning(
                    f"⚠️ State transition validation failed: {validation_result.errors}"
                )

        # Record transition for audit trail
        transition = StateTransition(
            transition_id=transition_id,
            from_phase=original_phase,
            to_phase=new_phase,
            timestamp=datetime.now().isoformat(),
            updates_applied=updates,
            transition_type=transition_type,
            validation_passed=validation_passed,
            execution_time_ms=execution_time,
        )

        self.state_transitions.append(transition)

        # Add transition metadata to new state
        if not hasattr(new_state, "processing_metadata"):
            new_state.processing_metadata = {}

        if "state_transitions" not in new_state.processing_metadata:
            new_state.processing_metadata["state_transitions"] = []

        new_state.processing_metadata["state_transitions"].append(
            {
                "transition_id": transition_id,
                "transition_type": transition_type,
                "timestamp": transition.timestamp,
                "validation_passed": validation_passed,
                "execution_time_ms": execution_time,
            }
        )

        self.logger.debug(
            f"✅ New state created (ID: {transition_id}) in {execution_time:.2f}ms"
        )

        return new_state

    def _apply_updates_to_state(
        self, state: MetisDataContract, updates: Dict[str, Any]
    ) -> None:
        """Apply updates to state object using safe attribute setting."""

        for key, value in updates.items():
            try:
                if "." in key:
                    # Handle nested attribute updates (e.g., "workflow_state.current_phase")
                    parts = key.split(".")
                    obj = state

                    # Navigate to parent object
                    for part in parts[:-1]:
                        obj = getattr(obj, part)

                    final_key = parts[-1]

                    # Check if target is a dict (e.g., phase_results, processing_metadata)
                    if isinstance(obj, dict):
                        obj[final_key] = value
                    else:
                        # Try to set as attribute
                        try:
                            setattr(obj, final_key, value)
                        except AttributeError:
                            # If attribute doesn't exist but object has __dict__, create it
                            if hasattr(obj, "__dict__"):
                                setattr(obj, final_key, value)
                            else:
                                # Last resort: treat as dict if possible
                                if hasattr(obj, "__setitem__"):
                                    obj[final_key] = value
                                else:
                                    raise

                else:
                    # Handle direct attribute updates
                    setattr(state, key, value)

            except AttributeError as e:
                self.logger.warning(f"⚠️ Failed to apply update {key}={value}: {e}")

    def validate_state_transition(
        self, from_state: MetisDataContract, to_state: MetisDataContract
    ) -> StateValidationResult:
        """
        Validate that state transition maintains data integrity and follows business rules.

        Args:
            from_state: Original state before transition
            to_state: New state after transition

        Returns:
            StateValidationResult with validation outcome
        """

        errors = []
        warnings = []
        validation_details = {}

        try:
            # 1. Validate engagement context immutability
            if (
                from_state.engagement_context.engagement_id
                != to_state.engagement_context.engagement_id
            ):
                errors.append("Engagement ID must not change during state transitions")

            if (
                from_state.engagement_context.problem_statement
                != to_state.engagement_context.problem_statement
            ):
                warnings.append("Problem statement changed during processing")

            # 2. Validate workflow state progression
            from_phase = getattr(from_state.workflow_state, "current_phase", None)
            to_phase = getattr(to_state.workflow_state, "current_phase", None)

            if from_phase and to_phase:
                valid_transitions = self._get_valid_phase_transitions()
                if from_phase != to_phase and to_phase not in valid_transitions.get(
                    from_phase, []
                ):
                    warnings.append(
                        f"Unusual phase transition: {from_phase} -> {to_phase}"
                    )

            # 3. Validate cognitive state growth (should only grow, never shrink)
            from_steps = len(getattr(from_state.cognitive_state, "reasoning_steps", []))
            to_steps = len(getattr(to_state.cognitive_state, "reasoning_steps", []))

            if to_steps < from_steps:
                errors.append(
                    f"Reasoning steps decreased from {from_steps} to {to_steps}"
                )

            # 4. Validate required data structures exist
            required_attrs = ["workflow_state", "cognitive_state", "engagement_context"]
            for attr in required_attrs:
                if not hasattr(to_state, attr):
                    errors.append(f"Required attribute missing: {attr}")

            # 5. Validate data capture growth (Neural Lace principle)
            if hasattr(from_state, "raw_outputs") and hasattr(to_state, "raw_outputs"):
                if len(to_state.raw_outputs) < len(from_state.raw_outputs):
                    errors.append(
                        "Raw outputs decreased - violates Neural Lace data capture principle"
                    )

            validation_details = {
                "phase_transition": f"{from_phase} -> {to_phase}",
                "reasoning_steps_growth": f"{from_steps} -> {to_steps}",
                "raw_outputs_count": len(getattr(to_state, "raw_outputs", {})),
                "integration_calls_count": len(
                    getattr(to_state, "integration_calls", [])
                ),
                "validation_timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            errors.append(f"State validation error: {str(e)}")
            self.logger.error(f"State validation exception: {e}")

        is_valid = len(errors) == 0

        return StateValidationResult(
            valid=is_valid,
            errors=errors,
            warnings=warnings,
            validation_details=validation_details,
        )

    def _get_valid_phase_transitions(
        self,
    ) -> Dict[EngagementPhase, List[EngagementPhase]]:
        """Define valid phase transitions for workflow validation."""
        return {
            EngagementPhase.PROBLEM_STRUCTURING: [
                EngagementPhase.HYPOTHESIS_GENERATION,
                EngagementPhase.PROBLEM_STRUCTURING,  # Allow staying in same phase
            ],
            EngagementPhase.HYPOTHESIS_GENERATION: [
                EngagementPhase.ANALYSIS_EXECUTION,
                EngagementPhase.HYPOTHESIS_GENERATION,
            ],
            EngagementPhase.ANALYSIS_EXECUTION: [
                EngagementPhase.SYNTHESIS_DELIVERY,
                EngagementPhase.ANALYSIS_EXECUTION,
            ],
            EngagementPhase.SYNTHESIS_DELIVERY: [
                EngagementPhase.SYNTHESIS_DELIVERY  # Final phase
            ],
        }

    def get_state_history(self, limit: int = 10) -> List[StateTransition]:
        """Get recent state transitions for debugging."""
        return self.state_transitions[-limit:]

    def generate_state_report(self, current_state: MetisDataContract) -> Dict[str, Any]:
        """Generate comprehensive state report for debugging."""

        return {
            "current_phase": getattr(
                current_state.workflow_state, "current_phase", None
            ),
            "reasoning_steps_count": len(
                getattr(current_state.cognitive_state, "reasoning_steps", [])
            ),
            "raw_outputs_count": len(getattr(current_state, "raw_outputs", {})),
            "integration_calls_count": len(
                getattr(current_state, "integration_calls", [])
            ),
            "completed_phases": getattr(
                current_state.workflow_state, "completed_phases", []
            ),
            "confidence_scores": getattr(
                current_state.cognitive_state, "confidence_scores", {}
            ),
            "selected_mental_models": getattr(
                current_state.cognitive_state, "selected_mental_models", []
            ),
            "total_state_transitions": len(self.state_transitions),
            "recent_transitions": [
                {
                    "transition_id": t.transition_id,
                    "type": t.transition_type,
                    "timestamp": t.timestamp,
                    "validation_passed": t.validation_passed,
                }
                for t in self.state_transitions[-5:]
            ],
            "state_size_estimate": len(str(current_state)),
            "report_timestamp": datetime.now().isoformat(),
        }

    def create_phase_completion_state(
        self,
        current_state: MetisDataContract,
        phase_name: str,
        phase_results: Dict[str, Any],
    ) -> MetisDataContract:
        """
        Specialized method for phase completion state transitions.

        Args:
            current_state: Current state
            phase_name: Name of completed phase
            phase_results: Results from phase execution

        Returns:
            New state with phase completion updates
        """

        phase_enum = EngagementPhase(phase_name)

        updates = {
            "workflow_state.current_phase": phase_enum,
            f"workflow_state.phase_results.{phase_name}": phase_results,
        }

        # Add to completed phases if not already there
        new_state = self.create_new_state(
            current_state=current_state,
            updates=updates,
            transition_type="phase_completion",
        )

        # Ensure completed phases list is updated
        if phase_enum not in new_state.workflow_state.completed_phases:
            new_state.workflow_state.completed_phases.append(phase_enum)

        return new_state

    def create_error_recovery_state(
        self,
        current_state: MetisDataContract,
        error: Exception,
        recovery_action: str,
        fallback_data: Dict[str, Any],
    ) -> MetisDataContract:
        """
        Specialized method for error recovery state transitions.

        Args:
            current_state: Current state before error
            error: Exception that occurred
            recovery_action: Description of recovery action taken
            fallback_data: Fallback data to maintain system operation

        Returns:
            New state with error recovery applied
        """

        error_metadata = {
            "error_occurred": True,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "recovery_action": recovery_action,
            "recovery_timestamp": datetime.now().isoformat(),
            "fallback_applied": True,
        }

        updates = {"processing_metadata.error_recovery": error_metadata}

        # Handle fallback data specially - don't set reasoning_step_fallback directly on contract
        for key, value in fallback_data.items():
            if key != "reasoning_step_fallback":  # Skip this problematic field
                updates[key] = value

        # Store reasoning_step_fallback in processing_metadata instead
        if "reasoning_step_fallback" in fallback_data:
            updates["processing_metadata.reasoning_step_fallback"] = fallback_data[
                "reasoning_step_fallback"
            ]

        return self.create_new_state(
            current_state=current_state,
            updates=updates,
            transition_type="error_recovery",
        )

    def disable_validation(self):
        """Disable state validation for performance-critical operations."""
        self.validation_enabled = False
        self.logger.debug("🔧 State validation disabled")

    def enable_validation(self):
        """Re-enable state validation."""
        self.validation_enabled = True
        self.logger.debug("🔧 State validation enabled")


# Global state manager instance
_state_manager_instance: Optional[ImmutableStateManager] = None


def get_immutable_state_manager() -> ImmutableStateManager:
    """Get global immutable state manager instance."""
    global _state_manager_instance
    if _state_manager_instance is None:
        _state_manager_instance = ImmutableStateManager()
    return _state_manager_instance
