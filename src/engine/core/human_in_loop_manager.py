"""
METIS Human-in-Loop Manager
Enterprise-grade human approval workflows for cognitive analysis

Based on industry insights:
- Anthropic: Constitutional AI with human oversight
- OpenAI: Approval workflows for sensitive decisions
- Enterprise Pattern: Trust-building through transparency
- Financial Services: Multi-tier approval for high-stakes analysis
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from uuid import uuid4

from src.core.stateful_environment import get_stateful_environment, CheckpointType
from src.core.performance_cache_system import get_performance_cache, CacheEntryType

logger = logging.getLogger(__name__)


class ApprovalLevel(str, Enum):
    """Hierarchy of approval requirements"""

    AUTOMATIC = "automatic"  # No human approval needed
    REVIEWER = "reviewer"  # Single reviewer approval
    MANAGER = "manager"  # Manager-level approval
    EXECUTIVE = "executive"  # Executive approval required
    BOARD = "board"  # Board-level approval


class InterventionType(str, Enum):
    """Types of human interventions"""

    APPROVAL_REQUEST = "approval_request"  # Request approval to proceed
    CLARIFICATION = "clarification"  # Request additional information
    GUIDANCE = "guidance"  # Request strategic guidance
    VALIDATION = "validation"  # Request result validation
    ESCALATION = "escalation"  # Escalate critical decision
    QUALITY_REVIEW = "quality_review"  # Review analysis quality


class InterventionStatus(str, Enum):
    """Status of intervention requests"""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    ESCALATED = "escalated"
    TIMEOUT = "timeout"


@dataclass
class InterventionRequest:
    """Request for human intervention"""

    request_id: str
    engagement_id: str
    intervention_type: InterventionType
    approval_level: ApprovalLevel

    # Request content
    title: str
    description: str
    context: Dict[str, Any]

    # Decision support
    current_analysis: Dict[str, Any]
    recommendations: List[str]
    risk_assessment: Dict[str, Any]
    confidence_scores: Dict[str, float]

    # Approval metadata
    requested_at: datetime
    timeout_at: Optional[datetime]
    priority: str = "medium"

    # State tracking
    status: InterventionStatus = InterventionStatus.PENDING
    assigned_to: Optional[str] = None
    escalation_path: List[str] = field(default_factory=list)

    # Response
    response: Optional[Dict[str, Any]] = None
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    feedback: Optional[str] = None


@dataclass
class ApprovalResponse:
    """Response to intervention request"""

    request_id: str
    decision: InterventionStatus

    # Decision details
    approved_by: str
    approved_at: datetime
    feedback: Optional[str] = None

    # Modifications (if any)
    context_modifications: Optional[Dict[str, Any]] = None
    approach_adjustments: Optional[List[str]] = None

    # Follow-up actions
    additional_requirements: List[str] = field(default_factory=list)
    monitoring_requirements: List[str] = field(default_factory=list)


class HumanInLoopManager:
    """
    Enterprise-grade human-in-loop manager
    Orchestrates approval workflows for cognitive analysis
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cache = get_performance_cache()

        # Active intervention requests
        self.active_requests: Dict[str, InterventionRequest] = {}
        self.request_history: List[InterventionRequest] = []

        # Approval workflow configuration
        self.approval_workflows = self._initialize_approval_workflows()
        self.escalation_rules = self._initialize_escalation_rules()

        # Performance metrics
        self.performance_metrics = {
            "total_requests": 0,
            "automatic_approvals": 0,
            "human_approvals": 0,
            "rejections": 0,
            "timeouts": 0,
            "average_response_time": 0.0,
            "approval_rate": 0.0,
            "escalation_rate": 0.0,
        }

        # Trust scoring
        self.trust_metrics = {
            "system_confidence_accuracy": 0.85,  # How often system confidence matches human assessment
            "recommendation_acceptance_rate": 0.78,  # How often humans accept AI recommendations
            "intervention_necessity_score": 0.72,  # How often interventions actually change outcomes
        }

        self.logger.info("👥 Human-in-Loop Manager initialized")

    async def request_intervention(
        self,
        engagement_id: str,
        intervention_type: InterventionType,
        title: str,
        description: str,
        context: Dict[str, Any],
        timeout_minutes: Optional[int] = None,
        priority: str = "medium",
    ) -> str:
        """
        Request human intervention
        Returns request ID for tracking
        """
        # Determine required approval level
        approval_level = await self._determine_approval_level(
            engagement_id, intervention_type, context
        )

        # Check for automatic approval conditions
        if approval_level == ApprovalLevel.AUTOMATIC:
            return await self._process_automatic_approval(
                engagement_id, intervention_type, context
            )

        # Create intervention request
        request_id = f"intervention_{int(time.time() * 1000)}_{str(uuid4())[:8]}"

        timeout_at = None
        if timeout_minutes:
            timeout_at = datetime.now() + timedelta(minutes=timeout_minutes)

        # Build risk assessment
        risk_assessment = await self._assess_intervention_risk(
            context, intervention_type
        )

        intervention_request = InterventionRequest(
            request_id=request_id,
            engagement_id=engagement_id,
            intervention_type=intervention_type,
            approval_level=approval_level,
            title=title,
            description=description,
            context=context,
            current_analysis=context.get("analysis_results", {}),
            recommendations=context.get("recommendations", []),
            risk_assessment=risk_assessment,
            confidence_scores=context.get("confidence_scores", {}),
            requested_at=datetime.now(),
            timeout_at=timeout_at,
            priority=priority,
        )

        # Store active request
        self.active_requests[request_id] = intervention_request
        self.performance_metrics["total_requests"] += 1

        # Create checkpoint before intervention
        stateful_env = get_stateful_environment(engagement_id)
        await stateful_env.create_checkpoint(
            checkpoint_type=CheckpointType.USER_INTERACTION,
            current_context=context,
            reasoning_steps=context.get("reasoning_steps", []),
            metadata={
                "intervention_request_id": request_id,
                "intervention_type": intervention_type.value,
                "approval_level": approval_level.value,
            },
        )

        # Cache request for fast access
        await self.cache.put(
            content_type=CacheEntryType.CONTEXT_COMBINATION,
            primary_key=f"intervention_request_{request_id}",
            content={
                "request_id": request_id,
                "type": intervention_type.value,
                "level": approval_level.value,
                "status": InterventionStatus.PENDING.value,
            },
            ttl_seconds=3600,
        )

        self.logger.info(
            f"👥 Intervention requested: {request_id} ({intervention_type.value}, {approval_level.value})"
        )

        # Notify appropriate approvers
        await self._notify_approvers(intervention_request)

        return request_id

    async def respond_to_intervention(
        self, request_id: str, response: ApprovalResponse
    ) -> bool:
        """
        Process response to intervention request
        Returns success status
        """
        if request_id not in self.active_requests:
            self.logger.warning(f"⚠️ Intervention request {request_id} not found")
            return False

        request = self.active_requests[request_id]
        response_time = datetime.now() - request.requested_at

        # Update request with response
        request.status = response.decision
        request.response = {
            "decision": response.decision.value,
            "approved_by": response.approved_by,
            "feedback": response.feedback,
            "context_modifications": response.context_modifications,
            "approach_adjustments": response.approach_adjustments,
        }
        request.approved_by = response.approved_by
        request.approved_at = response.approved_at
        request.feedback = response.feedback

        # Update performance metrics
        self._update_response_metrics(response.decision, response_time.total_seconds())

        # Move to history
        self.request_history.append(request)
        del self.active_requests[request_id]

        # Update cache
        await self.cache.put(
            content_type=CacheEntryType.CONTEXT_COMBINATION,
            primary_key=f"intervention_response_{request_id}",
            content={
                "request_id": request_id,
                "decision": response.decision.value,
                "response_time_seconds": response_time.total_seconds(),
                "approved_by": response.approved_by,
            },
            ttl_seconds=3600,
        )

        self.logger.info(
            f"✅ Intervention {request_id} resolved: {response.decision.value} by {response.approved_by}"
        )

        return True

    async def wait_for_intervention_response(
        self, request_id: str, check_interval_seconds: float = 5.0
    ) -> Optional[ApprovalResponse]:
        """
        Wait for human response to intervention request
        Returns response when available or None on timeout
        """
        if request_id not in self.active_requests:
            return None

        request = self.active_requests[request_id]

        while request.status == InterventionStatus.PENDING:
            await asyncio.sleep(check_interval_seconds)

            # Check timeout
            if request.timeout_at and datetime.now() > request.timeout_at:
                await self._handle_intervention_timeout(request_id)
                break

            # Check if request was resolved
            if request_id not in self.active_requests:
                # Request was resolved while waiting
                if (
                    self.request_history
                    and self.request_history[-1].request_id == request_id
                ):
                    completed_request = self.request_history[-1]
                    if completed_request.response:
                        return ApprovalResponse(
                            request_id=request_id,
                            decision=completed_request.status,
                            approved_by=completed_request.approved_by or "system",
                            approved_at=completed_request.approved_at or datetime.now(),
                            feedback=completed_request.feedback,
                            context_modifications=completed_request.response.get(
                                "context_modifications"
                            ),
                            approach_adjustments=completed_request.response.get(
                                "approach_adjustments"
                            ),
                        )
                break

        return None

    async def get_pending_interventions(
        self, approver_role: Optional[str] = None
    ) -> List[InterventionRequest]:
        """Get list of pending interventions for approver"""
        pending = []

        for request in self.active_requests.values():
            if request.status == InterventionStatus.PENDING:
                if approver_role is None or self._can_approve(
                    approver_role, request.approval_level
                ):
                    pending.append(request)

        # Sort by priority and age
        pending.sort(
            key=lambda r: (
                {"high": 0, "medium": 1, "low": 2}.get(r.priority, 1),
                r.requested_at,
            )
        )

        return pending

    async def simulate_human_response(
        self,
        request_id: str,
        auto_approve_probability: float = 0.8,
        response_delay_seconds: float = 2.0,
    ) -> bool:
        """
        Simulate human response for testing/demo purposes
        Returns success status
        """
        if request_id not in self.active_requests:
            return False

        request = self.active_requests[request_id]

        # Simulate human thinking time
        await asyncio.sleep(response_delay_seconds)

        # Determine decision based on context
        import random

        decision = (
            InterventionStatus.APPROVED
            if random.random() < auto_approve_probability
            else InterventionStatus.REJECTED
        )

        # Create simulated response
        simulated_response = ApprovalResponse(
            request_id=request_id,
            decision=decision,
            approved_by="demo_approver",
            approved_at=datetime.now(),
            feedback=(
                f"Simulated {decision.value} decision"
                if decision == InterventionStatus.REJECTED
                else None
            ),
            context_modifications=(
                None
                if decision == InterventionStatus.APPROVED
                else {"reduce_scope": True}
            ),
        )

        return await self.respond_to_intervention(request_id, simulated_response)

    def _initialize_approval_workflows(self) -> Dict[str, Dict[str, Any]]:
        """Initialize approval workflow configurations"""
        return {
            "low_risk": {
                "approval_level": ApprovalLevel.AUTOMATIC,
                "timeout_minutes": 0,
                "escalation_required": False,
            },
            "medium_risk": {
                "approval_level": ApprovalLevel.REVIEWER,
                "timeout_minutes": 30,
                "escalation_required": False,
            },
            "high_risk": {
                "approval_level": ApprovalLevel.MANAGER,
                "timeout_minutes": 60,
                "escalation_required": True,
            },
            "critical": {
                "approval_level": ApprovalLevel.EXECUTIVE,
                "timeout_minutes": 120,
                "escalation_required": True,
            },
        }

    def _initialize_escalation_rules(self) -> Dict[str, List[str]]:
        """Initialize escalation rules"""
        return {
            ApprovalLevel.REVIEWER.value: [ApprovalLevel.MANAGER.value],
            ApprovalLevel.MANAGER.value: [ApprovalLevel.EXECUTIVE.value],
            ApprovalLevel.EXECUTIVE.value: [ApprovalLevel.BOARD.value],
            ApprovalLevel.BOARD.value: [],  # No further escalation
        }

    async def _determine_approval_level(
        self,
        engagement_id: str,
        intervention_type: InterventionType,
        context: Dict[str, Any],
    ) -> ApprovalLevel:
        """Determine required approval level based on context"""

        # Analyze risk factors
        risk_score = 0.0

        # Confidence score factor
        confidence_scores = context.get("confidence_scores", {})
        if confidence_scores:
            avg_confidence = sum(confidence_scores.values()) / len(confidence_scores)
            if avg_confidence < 0.6:
                risk_score += 0.3

        # Business impact factor
        business_context = context.get("business_context", {})
        if business_context.get("urgency") == "high":
            risk_score += 0.2
        if len(business_context.get("stakeholders", [])) > 5:
            risk_score += 0.1

        # Intervention type factor
        intervention_risks = {
            InterventionType.APPROVAL_REQUEST: 0.1,
            InterventionType.CLARIFICATION: 0.0,
            InterventionType.GUIDANCE: 0.2,
            InterventionType.VALIDATION: 0.1,
            InterventionType.ESCALATION: 0.4,
            InterventionType.QUALITY_REVIEW: 0.2,
        }
        risk_score += intervention_risks.get(intervention_type, 0.1)

        # Determine approval level
        if risk_score < 0.2:
            return ApprovalLevel.AUTOMATIC
        elif risk_score < 0.4:
            return ApprovalLevel.REVIEWER
        elif risk_score < 0.6:
            return ApprovalLevel.MANAGER
        else:
            return ApprovalLevel.EXECUTIVE

    async def _assess_intervention_risk(
        self, context: Dict[str, Any], intervention_type: InterventionType
    ) -> Dict[str, Any]:
        """Assess risk factors for intervention"""

        risk_factors = []
        risk_score = 0.0

        # Confidence analysis
        confidence_scores = context.get("confidence_scores", {})
        if confidence_scores:
            avg_confidence = sum(confidence_scores.values()) / len(confidence_scores)
            if avg_confidence < 0.7:
                risk_factors.append(f"Low average confidence: {avg_confidence:.2f}")
                risk_score += 0.3

        # Complexity analysis
        if context.get("problem_complexity") == "high":
            risk_factors.append("High problem complexity")
            risk_score += 0.2

        # Time pressure
        if context.get("business_context", {}).get("urgency") == "high":
            risk_factors.append("High urgency requirement")
            risk_score += 0.2

        # Stakeholder count
        stakeholder_count = len(
            context.get("business_context", {}).get("stakeholders", [])
        )
        if stakeholder_count > 5:
            risk_factors.append(f"High stakeholder count: {stakeholder_count}")
            risk_score += 0.1

        return {
            "risk_score": risk_score,
            "risk_level": (
                "high" if risk_score > 0.6 else "medium" if risk_score > 0.3 else "low"
            ),
            "risk_factors": risk_factors,
            "mitigation_strategies": self._generate_mitigation_strategies(risk_factors),
        }

    def _generate_mitigation_strategies(self, risk_factors: List[str]) -> List[str]:
        """Generate mitigation strategies for identified risks"""
        strategies = []

        for factor in risk_factors:
            if "confidence" in factor.lower():
                strategies.append("Request additional validation steps")
                strategies.append("Consider alternative mental models")
            elif "complexity" in factor.lower():
                strategies.append("Break problem into smaller components")
                strategies.append("Implement staged approval process")
            elif "urgency" in factor.lower():
                strategies.append("Allocate dedicated resources")
                strategies.append("Establish clear decision timeline")
            elif "stakeholder" in factor.lower():
                strategies.append("Implement stakeholder communication plan")
                strategies.append("Identify key decision makers")

        return strategies

    async def _process_automatic_approval(
        self,
        engagement_id: str,
        intervention_type: InterventionType,
        context: Dict[str, Any],
    ) -> str:
        """Process automatic approval for low-risk interventions"""

        request_id = f"auto_approved_{int(time.time() * 1000)}"

        self.performance_metrics["automatic_approvals"] += 1
        self.performance_metrics["total_requests"] += 1

        # Log automatic approval
        self.logger.info(
            f"✅ Automatic approval granted: {request_id} ({intervention_type.value})"
        )

        return request_id

    async def _notify_approvers(self, request: InterventionRequest):
        """Notify appropriate approvers of intervention request"""
        # In a real implementation, this would send notifications
        # via email, Slack, or other communication channels

        self.logger.info(f"📧 Notification sent for intervention {request.request_id}")
        self.logger.info(f"   Title: {request.title}")
        self.logger.info(f"   Approval Level: {request.approval_level.value}")
        self.logger.info(f"   Priority: {request.priority}")

        # For demo purposes, automatically simulate response
        if hasattr(self, "_auto_simulate") and self._auto_simulate:
            asyncio.create_task(self.simulate_human_response(request.request_id))

    async def _handle_intervention_timeout(self, request_id: str):
        """Handle intervention request timeout"""
        if request_id not in self.active_requests:
            return

        request = self.active_requests[request_id]
        request.status = InterventionStatus.TIMEOUT

        self.performance_metrics["timeouts"] += 1

        # Apply default action based on intervention type
        default_actions = {
            InterventionType.APPROVAL_REQUEST: "proceed_with_caution",
            InterventionType.CLARIFICATION: "use_default_assumptions",
            InterventionType.GUIDANCE: "apply_conservative_approach",
            InterventionType.VALIDATION: "flag_for_review",
            InterventionType.ESCALATION: "halt_execution",
            InterventionType.QUALITY_REVIEW: "accept_with_disclaimer",
        }

        default_action = default_actions.get(
            request.intervention_type, "halt_execution"
        )

        self.logger.warning(
            f"⏰ Intervention {request_id} timed out, applying default action: {default_action}"
        )

        # Move to history
        self.request_history.append(request)
        del self.active_requests[request_id]

    def _can_approve(self, approver_role: str, required_level: ApprovalLevel) -> bool:
        """Check if approver role can approve required level"""

        role_levels = {
            "reviewer": [ApprovalLevel.AUTOMATIC, ApprovalLevel.REVIEWER],
            "manager": [
                ApprovalLevel.AUTOMATIC,
                ApprovalLevel.REVIEWER,
                ApprovalLevel.MANAGER,
            ],
            "executive": [
                ApprovalLevel.AUTOMATIC,
                ApprovalLevel.REVIEWER,
                ApprovalLevel.MANAGER,
                ApprovalLevel.EXECUTIVE,
            ],
            "board": [
                ApprovalLevel.AUTOMATIC,
                ApprovalLevel.REVIEWER,
                ApprovalLevel.MANAGER,
                ApprovalLevel.EXECUTIVE,
                ApprovalLevel.BOARD,
            ],
        }

        return required_level in role_levels.get(approver_role.lower(), [])

    def _update_response_metrics(
        self, decision: InterventionStatus, response_time: float
    ):
        """Update performance metrics with response"""

        if decision == InterventionStatus.APPROVED:
            self.performance_metrics["human_approvals"] += 1
        elif decision == InterventionStatus.REJECTED:
            self.performance_metrics["rejections"] += 1

        # Update average response time
        current_avg = self.performance_metrics["average_response_time"]
        total_responses = (
            self.performance_metrics["human_approvals"]
            + self.performance_metrics["rejections"]
        )

        if total_responses > 1:
            new_avg = (
                (current_avg * (total_responses - 1)) + response_time
            ) / total_responses
            self.performance_metrics["average_response_time"] = new_avg
        else:
            self.performance_metrics["average_response_time"] = response_time

        # Update approval rate
        total_decisions = total_responses
        if total_decisions > 0:
            self.performance_metrics["approval_rate"] = (
                self.performance_metrics["human_approvals"] / total_decisions
            )

    def get_hitl_performance_metrics(self) -> Dict[str, Any]:
        """Get human-in-loop performance metrics"""
        return {
            "intervention_metrics": self.performance_metrics,
            "trust_metrics": self.trust_metrics,
            "active_requests": len(self.active_requests),
            "request_history_count": len(self.request_history),
            "approval_workflows": list(self.approval_workflows.keys()),
            "performance_targets": {
                "approval_rate": 0.80,
                "average_response_time_minutes": 30,
                "timeout_rate": 0.05,
                "escalation_rate": 0.10,
            },
        }


# Singleton instance for global access
_hitl_manager: Optional[HumanInLoopManager] = None


def get_hitl_manager() -> HumanInLoopManager:
    """Get singleton human-in-loop manager"""
    global _hitl_manager
    if _hitl_manager is None:
        _hitl_manager = HumanInLoopManager()
    return _hitl_manager
