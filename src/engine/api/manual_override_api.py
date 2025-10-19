#!/usr/bin/env python3
"""
Week 2 Day 5: Manual Override API
Provides human experts with the ability to intervene in the learning system

This API allows:
1. Forcing model selection overrides
2. Manually approving/rejecting value outcomes
3. Triggering immediate learning cycles
4. Adjusting effectiveness scores manually
5. Emergency circuit breaker controls
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import uuid4
import logging

try:
    from src.intelligence.model_selector import get_model_selector, ModelSelector
    from src.intelligence.bayesian_effectiveness_updater import (
        get_bayesian_updater,
        EffectivenessUpdate,
    )
    from src.core.value_database_integration import get_value_database_integrator
    from src.core.continuous_learning_orchestrator import (
        get_continuous_learning_orchestrator,
        LearningTrigger,
    )
    from src.core.circuit_breaker import get_circuit_breaker

    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    print(f"Warning: Manual Override dependencies not available: {e}")


# Pydantic models for API requests/responses
class ModelOverrideRequest(BaseModel):
    """Request to force specific model selection for an engagement"""

    engagement_id: str = Field(..., description="Engagement to apply override to")
    model_ids: List[str] = Field(..., description="Models to force select")
    reason: str = Field(..., description="Human readable reason for override")
    override_duration_hours: Optional[int] = Field(
        default=24, description="How long override should last"
    )
    expert_id: str = Field(..., description="ID of expert making override")


class ValueOutcomeOverrideRequest(BaseModel):
    """Request to manually approve/reject value outcome"""

    engagement_id: str = Field(..., description="Engagement ID")
    action: str = Field(
        ..., description="'approve' or 'reject'", pattern="^(approve|reject)$"
    )
    reason: str = Field(..., description="Reason for manual decision")
    expert_id: str = Field(..., description="ID of expert making decision")


class EffectivenessOverrideRequest(BaseModel):
    """Request to manually adjust model effectiveness"""

    model_id: str = Field(..., description="Model to adjust")
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Context for effectiveness"
    )
    new_effectiveness: float = Field(
        ..., ge=0.0, le=1.0, description="New effectiveness score"
    )
    reason: str = Field(..., description="Reason for manual adjustment")
    expert_id: str = Field(..., description="ID of expert making adjustment")


class LearningCycleRequest(BaseModel):
    """Request to trigger learning cycle"""

    trigger_reason: str = Field(..., description="Reason for manual trigger")
    expert_id: str = Field(..., description="ID of expert triggering cycle")


class CircuitBreakerRequest(BaseModel):
    """Request to control circuit breaker"""

    action: str = Field(..., description="'open', 'close', or 'half_open'")
    reason: str = Field(..., description="Reason for circuit breaker action")
    expert_id: str = Field(..., description="ID of expert controlling breaker")


class OverrideAuditEntry(BaseModel):
    """Audit entry for manual overrides"""

    override_id: str
    override_type: str
    expert_id: str
    target_id: str  # engagement_id, model_id, etc.
    action: str
    reason: str
    timestamp: datetime
    status: str
    impact: Optional[Dict[str, Any]] = None


# Router for manual override endpoints
router = APIRouter(prefix="/api/v1/manual-override", tags=["manual-override"])


class ManualOverrideManager:
    """Manager for all manual override operations"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_overrides: Dict[str, Dict[str, Any]] = {}
        self.override_audit: List[OverrideAuditEntry] = []

        # Initialize components individually with error handling
        self.model_selector = None
        self.bayesian_updater = None
        self.value_integrator = None
        self.learning_orchestrator = None
        self.circuit_breaker = None

        if DEPENDENCIES_AVAILABLE:
            try:
                self.model_selector = get_model_selector()
            except Exception as e:
                self.logger.debug(f"Model selector not available: {e}")

            try:
                self.bayesian_updater = get_bayesian_updater()
            except Exception as e:
                self.logger.debug(f"Bayesian updater not available: {e}")

            try:
                self.value_integrator = get_value_database_integrator()
            except Exception as e:
                self.logger.debug(f"Value integrator not available: {e}")

            try:
                self.learning_orchestrator = get_continuous_learning_orchestrator()
            except Exception as e:
                self.logger.debug(f"Learning orchestrator not available: {e}")

            try:
                self.circuit_breaker = get_circuit_breaker()
            except Exception as e:
                self.logger.debug(f"Circuit breaker not available: {e}")

    async def create_model_override(
        self, request: ModelOverrideRequest
    ) -> Dict[str, Any]:
        """Create model selection override for engagement"""

        override_id = str(uuid4())
        expiry = datetime.utcnow().timestamp() + (
            request.override_duration_hours * 3600
        )

        override_entry = {
            "override_id": override_id,
            "type": "model_selection",
            "engagement_id": request.engagement_id,
            "forced_models": request.model_ids,
            "expert_id": request.expert_id,
            "reason": request.reason,
            "created_at": datetime.utcnow(),
            "expires_at": expiry,
            "status": "active",
        }

        # Store override
        self.active_overrides[f"model_{request.engagement_id}"] = override_entry

        # Add audit entry
        audit_entry = OverrideAuditEntry(
            override_id=override_id,
            override_type="model_selection",
            expert_id=request.expert_id,
            target_id=request.engagement_id,
            action="force_models",
            reason=request.reason,
            timestamp=datetime.utcnow(),
            status="active",
            impact={"forced_models": request.model_ids},
        )
        self.override_audit.append(audit_entry)

        self.logger.info(
            f"🎯 Model override created: {request.engagement_id} -> {request.model_ids}"
        )

        return {
            "override_id": override_id,
            "status": "active",
            "expires_in_hours": request.override_duration_hours,
            "message": f"Model selection override active for engagement {request.engagement_id}",
        }

    async def override_value_outcome(
        self, request: ValueOutcomeOverrideRequest
    ) -> Dict[str, Any]:
        """Manually approve or reject a flagged value outcome"""

        if not self.value_integrator:
            raise HTTPException(
                status_code=503, detail="Value database integrator not available"
            )

        override_id = str(uuid4())

        try:
            # Get the value outcome
            if request.engagement_id not in self.value_integrator.value_outcomes:
                raise HTTPException(
                    status_code=404,
                    detail=f"Value outcome for {request.engagement_id} not found",
                )

            outcome = self.value_integrator.value_outcomes[request.engagement_id]

            if request.action == "approve":
                # Manually approve the outcome
                success = self.value_integrator.approve_flagged_outcome(
                    request.engagement_id,
                    f"Manual override by {request.expert_id}: {request.reason}",
                )
                action_taken = "approved"
            else:
                # Reject the outcome (exclude from learning)
                outcome.approved_for_learning = False
                outcome.needs_review = True
                outcome.reviewer_notes = (
                    f"Manual rejection by {request.expert_id}: {request.reason}"
                )
                success = True
                action_taken = "rejected"

            # Audit entry
            audit_entry = OverrideAuditEntry(
                override_id=override_id,
                override_type="value_outcome",
                expert_id=request.expert_id,
                target_id=request.engagement_id,
                action=action_taken,
                reason=request.reason,
                timestamp=datetime.utcnow(),
                status="completed",
                impact={"approved_for_learning": outcome.approved_for_learning},
            )
            self.override_audit.append(audit_entry)

            self.logger.info(
                f"✅ Value outcome {action_taken}: {request.engagement_id}"
            )

            return {
                "override_id": override_id,
                "status": "completed",
                "action": action_taken,
                "message": f"Value outcome {action_taken} for {request.engagement_id}",
            }

        except Exception as e:
            self.logger.error(f"❌ Value outcome override failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def override_model_effectiveness(
        self, request: EffectivenessOverrideRequest
    ) -> Dict[str, Any]:
        """Manually adjust model effectiveness score"""

        if not self.bayesian_updater:
            raise HTTPException(
                status_code=503, detail="Bayesian updater not available"
            )

        override_id = str(uuid4())

        try:
            # Create effectiveness update with manual override
            effectiveness_update = EffectivenessUpdate(
                model_id=request.model_id,
                effectiveness_score=request.new_effectiveness,
                context=request.context,
                timestamp=datetime.utcnow(),
            )

            # Apply the manual effectiveness update
            state = await self.bayesian_updater.update_model_effectiveness(
                effectiveness_update
            )

            # Audit entry
            audit_entry = OverrideAuditEntry(
                override_id=override_id,
                override_type="model_effectiveness",
                expert_id=request.expert_id,
                target_id=request.model_id,
                action="adjust_effectiveness",
                reason=request.reason,
                timestamp=datetime.utcnow(),
                status="completed",
                impact={
                    "new_effectiveness": state.get_current_effectiveness(),
                    "context": request.context,
                    "manual_override": True,
                },
            )
            self.override_audit.append(audit_entry)

            self.logger.info(
                f"⚡ Model effectiveness override: {request.model_id} -> {request.new_effectiveness}"
            )

            return {
                "override_id": override_id,
                "status": "completed",
                "model_id": request.model_id,
                "old_effectiveness": "unknown",
                "new_effectiveness": state.get_current_effectiveness(),
                "message": f"Model effectiveness manually adjusted for {request.model_id}",
            }

        except Exception as e:
            self.logger.error(f"❌ Effectiveness override failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def trigger_learning_cycle(
        self, request: LearningCycleRequest
    ) -> Dict[str, Any]:
        """Manually trigger immediate learning cycle"""

        if not self.learning_orchestrator:
            raise HTTPException(
                status_code=503, detail="Learning orchestrator not available"
            )

        override_id = str(uuid4())

        try:
            # Trigger manual learning cycle
            cycle_metrics = await self.learning_orchestrator.execute_learning_cycle(
                LearningTrigger.MANUAL
            )

            # Audit entry
            audit_entry = OverrideAuditEntry(
                override_id=override_id,
                override_type="learning_cycle",
                expert_id=request.expert_id,
                target_id="system",
                action="trigger_cycle",
                reason=request.trigger_reason,
                timestamp=datetime.utcnow(),
                status="completed" if cycle_metrics.success else "failed",
                impact={
                    "cycle_id": str(cycle_metrics.cycle_id),
                    "tests_analyzed": cycle_metrics.tests_analyzed,
                    "models_updated": cycle_metrics.models_updated,
                    "success": cycle_metrics.success,
                },
            )
            self.override_audit.append(audit_entry)

            self.logger.info(
                f"🔄 Manual learning cycle triggered: {cycle_metrics.cycle_id}"
            )

            return {
                "override_id": override_id,
                "cycle_id": str(cycle_metrics.cycle_id),
                "status": "completed" if cycle_metrics.success else "failed",
                "tests_analyzed": cycle_metrics.tests_analyzed,
                "models_updated": cycle_metrics.models_updated,
                "message": "Learning cycle manually triggered",
            }

        except Exception as e:
            self.logger.error(f"❌ Manual learning cycle failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def control_circuit_breaker(
        self, request: CircuitBreakerRequest
    ) -> Dict[str, Any]:
        """Control emergency circuit breaker"""

        if not self.circuit_breaker:
            raise HTTPException(status_code=503, detail="Circuit breaker not available")

        override_id = str(uuid4())

        try:
            # Apply circuit breaker action
            if request.action == "open":
                self.circuit_breaker.open()
                action_taken = "Circuit breaker opened - system disabled"
            elif request.action == "close":
                self.circuit_breaker.close()
                action_taken = "Circuit breaker closed - system enabled"
            elif request.action == "half_open":
                self.circuit_breaker.half_open()
                action_taken = "Circuit breaker half-open - limited operation"
            else:
                raise ValueError(f"Invalid circuit breaker action: {request.action}")

            # Audit entry
            audit_entry = OverrideAuditEntry(
                override_id=override_id,
                override_type="circuit_breaker",
                expert_id=request.expert_id,
                target_id="system",
                action=request.action,
                reason=request.reason,
                timestamp=datetime.utcnow(),
                status="completed",
                impact={"circuit_state": request.action},
            )
            self.override_audit.append(audit_entry)

            self.logger.warning(
                f"🚨 Circuit breaker {request.action}: {request.reason}"
            )

            return {
                "override_id": override_id,
                "status": "completed",
                "circuit_state": request.action,
                "message": action_taken,
            }

        except Exception as e:
            self.logger.error(f"❌ Circuit breaker control failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def get_active_overrides(self) -> List[Dict[str, Any]]:
        """Get list of all active overrides"""

        current_time = datetime.utcnow().timestamp()
        active = []

        # Clean up expired overrides
        expired_keys = []
        for key, override in self.active_overrides.items():
            if override.get("expires_at", float("inf")) < current_time:
                expired_keys.append(key)
            else:
                active.append(override)

        # Remove expired overrides
        for key in expired_keys:
            del self.active_overrides[key]

        return active

    def get_override_audit(self, limit: int = 100) -> List[OverrideAuditEntry]:
        """Get recent override audit entries"""
        return sorted(self.override_audit, key=lambda x: x.timestamp, reverse=True)[
            :limit
        ]


# Global manager instance
_override_manager: Optional[ManualOverrideManager] = None


def get_override_manager() -> ManualOverrideManager:
    """Get or create global manual override manager"""
    global _override_manager
    if _override_manager is None:
        _override_manager = ManualOverrideManager()
    return _override_manager


# API Endpoints


@router.post("/model-selection", response_model=Dict[str, Any])
async def create_model_override(request: ModelOverrideRequest):
    """Force specific model selection for an engagement"""
    manager = get_override_manager()
    return await manager.create_model_override(request)


@router.post("/value-outcome", response_model=Dict[str, Any])
async def override_value_outcome(request: ValueOutcomeOverrideRequest):
    """Manually approve or reject a value outcome"""
    manager = get_override_manager()
    return await manager.override_value_outcome(request)


@router.post("/model-effectiveness", response_model=Dict[str, Any])
async def override_model_effectiveness(request: EffectivenessOverrideRequest):
    """Manually adjust model effectiveness score"""
    manager = get_override_manager()
    return await manager.override_model_effectiveness(request)


@router.post("/trigger-learning", response_model=Dict[str, Any])
async def trigger_learning_cycle(request: LearningCycleRequest):
    """Manually trigger immediate learning cycle"""
    manager = get_override_manager()
    return await manager.trigger_learning_cycle(request)


@router.post("/circuit-breaker", response_model=Dict[str, Any])
async def control_circuit_breaker(request: CircuitBreakerRequest):
    """Control emergency circuit breaker"""
    manager = get_override_manager()
    return await manager.control_circuit_breaker(request)


@router.get("/active", response_model=List[Dict[str, Any]])
async def get_active_overrides():
    """Get all currently active overrides"""
    manager = get_override_manager()
    return manager.get_active_overrides()


@router.get("/audit", response_model=List[OverrideAuditEntry])
async def get_override_audit(limit: int = 100):
    """Get recent override audit entries"""
    manager = get_override_manager()
    return manager.get_override_audit(limit)


@router.get("/status", response_model=Dict[str, Any])
async def get_override_system_status():
    """Get status of override system components"""
    manager = get_override_manager()

    return {
        "system_status": "operational",
        "components": {
            "model_selector": manager.model_selector is not None,
            "bayesian_updater": manager.bayesian_updater is not None,
            "value_integrator": manager.value_integrator is not None,
            "learning_orchestrator": manager.learning_orchestrator is not None,
            "circuit_breaker": manager.circuit_breaker is not None,
        },
        "active_overrides": len(manager.get_active_overrides()),
        "audit_entries": len(manager.override_audit),
        "dependencies_available": DEPENDENCIES_AVAILABLE,
    }
