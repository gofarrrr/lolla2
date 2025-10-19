"""
METIS Human-in-the-Loop Enterprise Framework
Week 4 Sprint: Enterprise Integration & Human Oversight

Implements enterprise-grade human oversight patterns including:
- Blueprint workflow orchestration with approval gates
- Expert review and validation workflows
- Collaborative analysis with human-AI partnership
- Enterprise audit trails and governance
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid

# Enterprise workflow imports
try:
    from src.engine.models.data_contracts import MetisDataContract, CognitiveState
    from src.engine.adapters.workflow import  # Migrated StreamingEvent

    CONTRACT_AVAILABLE = True
except ImportError:
    CONTRACT_AVAILABLE = False


class ApprovalStatus(Enum):
    """Approval status for human oversight checkpoints"""

    PENDING = "pending"  # Awaiting human review
    APPROVED = "approved"  # Human approved, proceed
    REJECTED = "rejected"  # Human rejected, needs revision
    ESCALATED = "escalated"  # Escalated to higher authority
    TIMEOUT = "timeout"  # Review timeout, auto-approval
    BYPASSED = "bypassed"  # Emergency bypass activated


class HumanRole(Enum):
    """Enterprise roles for human oversight"""

    ANALYST = "analyst"  # Junior analyst review
    SENIOR_ANALYST = "senior_analyst"  # Senior analyst validation
    PARTNER = "partner"  # Partner-level review
    DOMAIN_EXPERT = "domain_expert"  # Subject matter expert
    COMPLIANCE_OFFICER = "compliance_officer"  # Regulatory compliance
    CLIENT = "client"  # Client approval checkpoint


class InterventionType(Enum):
    """Types of human intervention in cognitive processing"""

    VALIDATION = "validation"  # Validate AI analysis quality
    DIRECTION = "direction"  # Provide analysis direction
    EXPERTISE = "expertise"  # Add domain expertise
    APPROVAL = "approval"  # Formal approval checkpoint
    CORRECTION = "correction"  # Correct AI errors
    ENHANCEMENT = "enhancement"  # Enhance AI recommendations


@dataclass
class HumanCheckpoint:
    """Human oversight checkpoint in cognitive workflow"""

    checkpoint_id: str
    checkpoint_name: str
    required_role: HumanRole
    intervention_type: InterventionType
    timeout_minutes: int = 30
    auto_approve_on_timeout: bool = False
    escalation_chain: List[HumanRole] = field(default_factory=list)
    bypass_allowed: bool = False
    required_confidence_threshold: float = 0.8

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    blocking: bool = True  # Whether workflow stops until approval


@dataclass
class HumanIntervention:
    """Record of human intervention in cognitive process"""

    intervention_id: str
    checkpoint_id: str
    human_role: HumanRole
    intervention_type: InterventionType
    status: ApprovalStatus

    # Human input
    human_feedback: str = ""
    confidence_override: Optional[float] = None
    recommendations_modified: bool = False
    analysis_direction: str = ""
    domain_expertise_added: str = ""

    # Timing and audit
    requested_at: datetime = field(default_factory=datetime.now)
    responded_at: Optional[datetime] = None
    response_time_minutes: float = 0.0
    escalated_to: Optional[HumanRole] = None

    # Metadata
    reviewer_id: str = ""
    reviewer_email: str = ""
    decision_rationale: str = ""
    ai_analysis_quality_score: float = 0.0


@dataclass
class BlueprintWorkflowStep:
    """Enterprise blueprint workflow step with human oversight"""

    step_id: str
    step_name: str
    cognitive_operation: str  # Mental model or analysis type
    human_checkpoints: List[HumanCheckpoint] = field(default_factory=list)
    parallel_execution_allowed: bool = True
    dependencies: List[str] = field(default_factory=list)
    success_criteria: Dict[str, Any] = field(default_factory=dict)


class HumanOversightEngine:
    """
    Engine for managing human oversight in cognitive workflows
    Orchestrates human-AI collaboration with enterprise governance
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_interventions: Dict[str, HumanIntervention] = {}
        self.intervention_history: List[HumanIntervention] = []
        self.approval_workflows: Dict[str, List[HumanCheckpoint]] = {}

        # Enterprise configuration
        self.auto_approval_enabled = False
        self.emergency_bypass_enabled = False
        self.audit_trail_enabled = True

        # Human reviewer callbacks
        self.notification_callbacks: List[Callable] = []
        self.escalation_callbacks: List[Callable] = []

    async def create_human_checkpoint(
        self,
        engagement_id: str,
        phase: str,
        checkpoint_config: HumanCheckpoint,
        cognitive_results: Dict[str, Any],
    ) -> str:
        """Create human oversight checkpoint in cognitive workflow"""

        checkpoint_id = f"{engagement_id}_{phase}_{checkpoint_config.checkpoint_id}"

        # Check if intervention is needed based on confidence threshold
        analysis_confidence = cognitive_results.get("confidence", 0.7)
        if analysis_confidence >= checkpoint_config.required_confidence_threshold:
            if self.auto_approval_enabled:
                self.logger.info(
                    f"🤖 Auto-approving checkpoint {checkpoint_id} - confidence {analysis_confidence:.1%} above threshold"
                )
                return "auto_approved"

        # Create intervention record
        intervention = HumanIntervention(
            intervention_id=str(uuid.uuid4()),
            checkpoint_id=checkpoint_id,
            human_role=checkpoint_config.required_role,
            intervention_type=checkpoint_config.intervention_type,
            status=ApprovalStatus.PENDING,
            ai_analysis_quality_score=analysis_confidence,
        )

        self.active_interventions[checkpoint_id] = intervention

        # Notify human reviewers
        await self._notify_human_reviewers(intervention, cognitive_results)

        # Start timeout monitoring
        if checkpoint_config.timeout_minutes > 0:
            asyncio.create_task(self._monitor_timeout(checkpoint_id, checkpoint_config))

        self.logger.info(
            f"👥 Created human checkpoint: {checkpoint_id} for {checkpoint_config.required_role.value}"
        )
        return checkpoint_id

    async def await_human_approval(
        self, checkpoint_id: str, max_wait_minutes: int = 60
    ) -> HumanIntervention:
        """Wait for human approval with timeout handling"""

        start_time = datetime.now()
        max_wait_time = timedelta(minutes=max_wait_minutes)

        while checkpoint_id in self.active_interventions:
            intervention = self.active_interventions[checkpoint_id]

            # Check if approval received
            if intervention.status != ApprovalStatus.PENDING:
                # Move to history and return result
                self.intervention_history.append(intervention)
                del self.active_interventions[checkpoint_id]

                self.logger.info(
                    f"👥 Human approval received: {intervention.status.value} for {checkpoint_id}"
                )
                return intervention

            # Check timeout
            if datetime.now() - start_time > max_wait_time:
                intervention.status = ApprovalStatus.TIMEOUT
                intervention.responded_at = datetime.now()
                intervention.response_time_minutes = (
                    datetime.now() - intervention.requested_at
                ).total_seconds() / 60

                self.logger.warning(f"⏰ Human approval timeout for {checkpoint_id}")
                return intervention

            # Wait and check again
            await asyncio.sleep(1)

        # Checkpoint was removed externally
        raise ValueError(f"Checkpoint {checkpoint_id} no longer exists")

    async def submit_human_decision(
        self,
        checkpoint_id: str,
        decision: ApprovalStatus,
        reviewer_info: Dict[str, str],
        feedback: str = "",
        analysis_modifications: Dict[str, Any] = None,
    ) -> bool:
        """Submit human decision for pending checkpoint"""

        if checkpoint_id not in self.active_interventions:
            self.logger.error(
                f"Checkpoint {checkpoint_id} not found or already resolved"
            )
            return False

        intervention = self.active_interventions[checkpoint_id]

        # Update intervention with human decision
        intervention.status = decision
        intervention.responded_at = datetime.now()
        intervention.response_time_minutes = (
            datetime.now() - intervention.requested_at
        ).total_seconds() / 60
        intervention.human_feedback = feedback
        intervention.reviewer_id = reviewer_info.get("reviewer_id", "")
        intervention.reviewer_email = reviewer_info.get("reviewer_email", "")
        intervention.decision_rationale = reviewer_info.get("rationale", "")

        # Apply analysis modifications if provided
        if analysis_modifications:
            intervention.recommendations_modified = True
            intervention.confidence_override = analysis_modifications.get(
                "confidence_override"
            )
            intervention.analysis_direction = analysis_modifications.get(
                "analysis_direction", ""
            )
            intervention.domain_expertise_added = analysis_modifications.get(
                "domain_expertise", ""
            )

        self.logger.info(
            f"👤 Human decision submitted: {decision.value} for {checkpoint_id} by {reviewer_info.get('reviewer_email', 'unknown')}"
        )

        # Audit trail
        if self.audit_trail_enabled:
            await self._log_audit_event(
                "human_decision_submitted",
                {
                    "checkpoint_id": checkpoint_id,
                    "decision": decision.value,
                    "reviewer": reviewer_info,
                    "response_time_minutes": intervention.response_time_minutes,
                },
            )

        return True

    async def _notify_human_reviewers(
        self, intervention: HumanIntervention, cognitive_results: Dict[str, Any]
    ):
        """Notify human reviewers about pending approval"""

        notification_data = {
            "intervention_id": intervention.intervention_id,
            "checkpoint_id": intervention.checkpoint_id,
            "required_role": intervention.human_role.value,
            "intervention_type": intervention.intervention_type.value,
            "analysis_confidence": intervention.ai_analysis_quality_score,
            "cognitive_results_summary": {
                "recommendations_count": len(
                    cognitive_results.get("recommendations", [])
                ),
                "mental_models_applied": len(
                    cognitive_results.get("mental_models_applied", [])
                ),
                "confidence": cognitive_results.get("confidence", 0.7),
            },
            "requested_at": intervention.requested_at.isoformat(),
        }

        # Send notifications through registered callbacks
        for callback in self.notification_callbacks:
            try:
                await callback(notification_data)
            except Exception as e:
                self.logger.error(f"Notification callback failed: {e}")

    async def _monitor_timeout(
        self, checkpoint_id: str, checkpoint_config: HumanCheckpoint
    ):
        """Monitor checkpoint timeout and handle auto-approval"""

        await asyncio.sleep(checkpoint_config.timeout_minutes * 60)

        # Check if still pending
        if checkpoint_id in self.active_interventions:
            intervention = self.active_interventions[checkpoint_id]

            if intervention.status == ApprovalStatus.PENDING:
                if checkpoint_config.auto_approve_on_timeout:
                    intervention.status = ApprovalStatus.APPROVED
                    intervention.human_feedback = "Auto-approved due to timeout"
                    self.logger.info(
                        f"⏰ Auto-approved checkpoint {checkpoint_id} due to timeout"
                    )
                else:
                    intervention.status = ApprovalStatus.TIMEOUT
                    self.logger.warning(
                        f"⏰ Checkpoint {checkpoint_id} timed out without approval"
                    )

                intervention.responded_at = datetime.now()
                intervention.response_time_minutes = checkpoint_config.timeout_minutes

    async def get_intervention_metrics(self) -> Dict[str, Any]:
        """Get human intervention metrics for enterprise reporting"""

        all_interventions = self.intervention_history + list(
            self.active_interventions.values()
        )

        if not all_interventions:
            return {"total_interventions": 0}

        # Calculate metrics
        approved_count = len(
            [i for i in all_interventions if i.status == ApprovalStatus.APPROVED]
        )
        rejected_count = len(
            [i for i in all_interventions if i.status == ApprovalStatus.REJECTED]
        )
        timeout_count = len(
            [i for i in all_interventions if i.status == ApprovalStatus.TIMEOUT]
        )

        response_times = [
            i.response_time_minutes
            for i in all_interventions
            if i.response_time_minutes > 0
        ]
        avg_response_time = (
            sum(response_times) / len(response_times) if response_times else 0
        )

        return {
            "total_interventions": len(all_interventions),
            "pending_interventions": len(self.active_interventions),
            "approval_rate": approved_count / len(all_interventions),
            "rejection_rate": rejected_count / len(all_interventions),
            "timeout_rate": timeout_count / len(all_interventions),
            "avg_response_time_minutes": avg_response_time,
            "intervention_types": {
                int_type.value: len(
                    [i for i in all_interventions if i.intervention_type == int_type]
                )
                for int_type in InterventionType
            },
            "role_distribution": {
                role.value: len([i for i in all_interventions if i.human_role == role])
                for role in HumanRole
            },
        }

    async def _log_audit_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log audit event for enterprise compliance"""

        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "event_data": event_data,
            "service": "metis_human_oversight",
        }

        # In production, this would go to enterprise audit system
        self.logger.info(f"AUDIT: {json.dumps(audit_entry)}")


class BlueprintWorkflowOrchestrator:
    """
    Enterprise blueprint workflow orchestrator
    Manages complex multi-step workflows with human oversight gates
    """

    def __init__(self, human_oversight_engine: HumanOversightEngine):
        self.oversight_engine = human_oversight_engine
        self.logger = logging.getLogger(__name__)
        self.blueprint_templates: Dict[str, List[BlueprintWorkflowStep]] = {}
        self.active_workflows: Dict[str, Dict[str, Any]] = {}

        # Initialize standard enterprise blueprints
        self._initialize_standard_blueprints()

    def _initialize_standard_blueprints(self):
        """Initialize standard enterprise workflow blueprints"""

        # McKinsey-Grade Strategic Analysis Blueprint
        strategic_analysis_blueprint = [
            BlueprintWorkflowStep(
                step_id="problem_structuring",
                step_name="Problem Structuring & Stakeholder Analysis",
                cognitive_operation="mece_problem_decomposition",
                human_checkpoints=[
                    HumanCheckpoint(
                        checkpoint_id="problem_validation",
                        checkpoint_name="Problem Definition Validation",
                        required_role=HumanRole.SENIOR_ANALYST,
                        intervention_type=InterventionType.VALIDATION,
                        timeout_minutes=15,
                        auto_approve_on_timeout=True,
                        required_confidence_threshold=0.8,
                    )
                ],
            ),
            BlueprintWorkflowStep(
                step_id="hypothesis_generation",
                step_name="Strategic Hypothesis Generation",
                cognitive_operation="hypothesis_development",
                human_checkpoints=[
                    HumanCheckpoint(
                        checkpoint_id="hypothesis_review",
                        checkpoint_name="Hypothesis Quality Review",
                        required_role=HumanRole.DOMAIN_EXPERT,
                        intervention_type=InterventionType.EXPERTISE,
                        timeout_minutes=30,
                        required_confidence_threshold=0.75,
                    )
                ],
                dependencies=["problem_structuring"],
            ),
            BlueprintWorkflowStep(
                step_id="analysis_execution",
                step_name="Deep Analysis Execution",
                cognitive_operation="mental_model_analysis",
                human_checkpoints=[
                    HumanCheckpoint(
                        checkpoint_id="methodology_approval",
                        checkpoint_name="Analysis Methodology Approval",
                        required_role=HumanRole.PARTNER,
                        intervention_type=InterventionType.APPROVAL,
                        timeout_minutes=45,
                        escalation_chain=[HumanRole.PARTNER],
                        required_confidence_threshold=0.85,
                    )
                ],
                dependencies=["hypothesis_generation"],
            ),
            BlueprintWorkflowStep(
                step_id="recommendation_synthesis",
                step_name="Strategic Recommendation Synthesis",
                cognitive_operation="pyramid_principle_synthesis",
                human_checkpoints=[
                    HumanCheckpoint(
                        checkpoint_id="final_review",
                        checkpoint_name="Final Deliverable Review",
                        required_role=HumanRole.PARTNER,
                        intervention_type=InterventionType.APPROVAL,
                        timeout_minutes=60,
                        bypass_allowed=False,
                        required_confidence_threshold=0.9,
                    ),
                    HumanCheckpoint(
                        checkpoint_id="client_approval",
                        checkpoint_name="Client Approval Gate",
                        required_role=HumanRole.CLIENT,
                        intervention_type=InterventionType.APPROVAL,
                        timeout_minutes=1440,  # 24 hours
                        blocking=True,
                    ),
                ],
                dependencies=["analysis_execution"],
            ),
        ]

        self.blueprint_templates["strategic_analysis"] = strategic_analysis_blueprint

        # Due Diligence Blueprint (High-Stakes Analysis)
        due_diligence_blueprint = [
            BlueprintWorkflowStep(
                step_id="risk_assessment",
                step_name="Comprehensive Risk Assessment",
                cognitive_operation="risk_analysis",
                human_checkpoints=[
                    HumanCheckpoint(
                        checkpoint_id="risk_validation",
                        checkpoint_name="Risk Assessment Validation",
                        required_role=HumanRole.COMPLIANCE_OFFICER,
                        intervention_type=InterventionType.VALIDATION,
                        timeout_minutes=20,
                        required_confidence_threshold=0.95,
                    )
                ],
            ),
            BlueprintWorkflowStep(
                step_id="financial_analysis",
                step_name="Financial Impact Analysis",
                cognitive_operation="financial_modeling",
                human_checkpoints=[
                    HumanCheckpoint(
                        checkpoint_id="financial_review",
                        checkpoint_name="Financial Model Review",
                        required_role=HumanRole.DOMAIN_EXPERT,
                        intervention_type=InterventionType.EXPERTISE,
                        timeout_minutes=45,
                        required_confidence_threshold=0.9,
                    )
                ],
                dependencies=["risk_assessment"],
            ),
            BlueprintWorkflowStep(
                step_id="recommendation_approval",
                step_name="Multi-Level Recommendation Approval",
                cognitive_operation="recommendation_synthesis",
                human_checkpoints=[
                    HumanCheckpoint(
                        checkpoint_id="senior_partner_approval",
                        checkpoint_name="Senior Partner Approval",
                        required_role=HumanRole.PARTNER,
                        intervention_type=InterventionType.APPROVAL,
                        timeout_minutes=120,
                        escalation_chain=[HumanRole.PARTNER],
                        bypass_allowed=False,
                        required_confidence_threshold=0.95,
                    )
                ],
                dependencies=["financial_analysis"],
            ),
        ]

        self.blueprint_templates["due_diligence"] = due_diligence_blueprint

        self.logger.info(
            f"🏗️ Initialized {len(self.blueprint_templates)} enterprise workflow blueprints"
        )

    async def execute_blueprint_workflow(
        self,
        workflow_id: str,
        blueprint_name: str,
        engagement_data: Dict[str, Any],
        human_oversight_level: str = "standard",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute enterprise blueprint workflow with human oversight"""

        if blueprint_name not in self.blueprint_templates:
            raise ValueError(f"Unknown blueprint: {blueprint_name}")

        blueprint_steps = self.blueprint_templates[blueprint_name]
        workflow_state = {
            "workflow_id": workflow_id,
            "blueprint_name": blueprint_name,
            "status": "running",
            "started_at": datetime.now(),
            "completed_steps": [],
            "pending_approvals": [],
            "human_interventions": [],
        }

        self.active_workflows[workflow_id] = workflow_state

        self.logger.info(
            f"🏗️ Starting blueprint workflow: {blueprint_name} ({workflow_id})"
        )

        # Yield workflow started event
        yield {
            "type": "blueprint_workflow_started",
            "workflow_id": workflow_id,
            "blueprint_name": blueprint_name,
            "total_steps": len(blueprint_steps),
            "human_oversight_level": human_oversight_level,
            "timestamp": datetime.now().isoformat(),
        }

        # Execute blueprint steps
        for step_index, step in enumerate(blueprint_steps):

            # Check dependencies
            if step.dependencies:
                missing_deps = [
                    dep
                    for dep in step.dependencies
                    if dep not in workflow_state["completed_steps"]
                ]
                if missing_deps:
                    self.logger.error(
                        f"Missing dependencies for {step.step_id}: {missing_deps}"
                    )
                    continue

            # Yield step started event
            yield {
                "type": "blueprint_step_started",
                "workflow_id": workflow_id,
                "step_id": step.step_id,
                "step_name": step.step_name,
                "step_index": step_index + 1,
                "total_steps": len(blueprint_steps),
                "timestamp": datetime.now().isoformat(),
            }

            # Execute cognitive operation (placeholder - would integrate with actual cognitive engine)
            cognitive_results = await self._execute_cognitive_operation(
                step.cognitive_operation, engagement_data
            )

            # Process human checkpoints
            for checkpoint in step.human_checkpoints:

                # Create human oversight checkpoint
                checkpoint_id = await self.oversight_engine.create_human_checkpoint(
                    workflow_id, step.step_id, checkpoint, cognitive_results
                )

                workflow_state["pending_approvals"].append(checkpoint_id)

                # Yield checkpoint created event
                yield {
                    "type": "human_checkpoint_created",
                    "workflow_id": workflow_id,
                    "checkpoint_id": checkpoint_id,
                    "required_role": checkpoint.required_role.value,
                    "intervention_type": checkpoint.intervention_type.value,
                    "blocking": checkpoint.blocking,
                    "timestamp": datetime.now().isoformat(),
                }

                # Wait for human approval if blocking
                if checkpoint.blocking:

                    self.logger.info(
                        f"⏸️ Workflow {workflow_id} paused for human approval: {checkpoint_id}"
                    )

                    # Yield workflow paused event
                    yield {
                        "type": "workflow_paused_for_approval",
                        "workflow_id": workflow_id,
                        "checkpoint_id": checkpoint_id,
                        "awaiting_role": checkpoint.required_role.value,
                        "timeout_minutes": checkpoint.timeout_minutes,
                        "timestamp": datetime.now().isoformat(),
                    }

                    # Wait for human decision
                    intervention = await self.oversight_engine.await_human_approval(
                        checkpoint_id
                    )
                    workflow_state["human_interventions"].append(
                        intervention.intervention_id
                    )

                    # Remove from pending
                    if checkpoint_id in workflow_state["pending_approvals"]:
                        workflow_state["pending_approvals"].remove(checkpoint_id)

                    # Yield approval received event
                    yield {
                        "type": "human_approval_received",
                        "workflow_id": workflow_id,
                        "checkpoint_id": checkpoint_id,
                        "approval_status": intervention.status.value,
                        "response_time_minutes": intervention.response_time_minutes,
                        "human_feedback": intervention.human_feedback,
                        "timestamp": datetime.now().isoformat(),
                    }

                    # Handle rejection
                    if intervention.status == ApprovalStatus.REJECTED:
                        self.logger.warning(
                            f"🚫 Human rejected step {step.step_id} - workflow terminated"
                        )
                        workflow_state["status"] = "rejected"

                        yield {
                            "type": "workflow_terminated",
                            "workflow_id": workflow_id,
                            "reason": "human_rejection",
                            "rejected_step": step.step_id,
                            "rejection_feedback": intervention.human_feedback,
                            "timestamp": datetime.now().isoformat(),
                        }
                        return

            # Mark step completed
            workflow_state["completed_steps"].append(step.step_id)

            # Yield step completed event
            yield {
                "type": "blueprint_step_completed",
                "workflow_id": workflow_id,
                "step_id": step.step_id,
                "step_name": step.step_name,
                "cognitive_results_summary": {
                    "confidence": cognitive_results.get("confidence", 0.7),
                    "recommendations_count": len(
                        cognitive_results.get("recommendations", [])
                    ),
                },
                "human_interventions_count": len(step.human_checkpoints),
                "timestamp": datetime.now().isoformat(),
            }

        # Workflow completed successfully
        workflow_state["status"] = "completed"
        workflow_state["completed_at"] = datetime.now()

        self.logger.info(f"✅ Blueprint workflow completed: {workflow_id}")

        # Get final metrics
        intervention_metrics = await self.oversight_engine.get_intervention_metrics()

        # Yield workflow completed event
        yield {
            "type": "blueprint_workflow_completed",
            "workflow_id": workflow_id,
            "blueprint_name": blueprint_name,
            "total_execution_time_minutes": (
                workflow_state["completed_at"] - workflow_state["started_at"]
            ).total_seconds()
            / 60,
            "steps_completed": len(workflow_state["completed_steps"]),
            "human_interventions": len(workflow_state["human_interventions"]),
            "intervention_metrics": intervention_metrics,
            "timestamp": datetime.now().isoformat(),
        }

    async def _execute_cognitive_operation(
        self, operation: str, engagement_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute cognitive operation (placeholder for actual implementation)"""

        # Simulate cognitive processing
        await asyncio.sleep(0.1)

        # Return mock cognitive results based on operation type
        if "problem" in operation:
            return {
                "confidence": 0.85,
                "problem_structure": "MECE decomposition applied",
                "recommendations": [
                    {"priority": "high", "action": "Address market saturation"}
                ],
            }
        elif "hypothesis" in operation:
            return {
                "confidence": 0.78,
                "hypotheses": [
                    {"hypothesis": "B2B pivot increases growth", "confidence": 0.8}
                ],
                "recommendations": [
                    {"priority": "medium", "action": "Test B2B pilot program"}
                ],
            }
        elif "analysis" in operation:
            return {
                "confidence": 0.92,
                "mental_models_applied": ["Systems Thinking", "Decision Analysis"],
                "recommendations": [
                    {"priority": "high", "action": "Execute strategic pivot"},
                    {"priority": "medium", "action": "Monitor market response"},
                ],
            }
        else:
            return {
                "confidence": 0.80,
                "recommendations": [
                    {"priority": "medium", "action": "Continue monitoring"}
                ],
            }

    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get current status of active workflow"""

        if workflow_id not in self.active_workflows:
            return {"error": "Workflow not found"}

        return self.active_workflows[workflow_id]

    def list_available_blueprints(self) -> Dict[str, Dict[str, Any]]:
        """List all available blueprint templates"""

        return {
            name: {
                "step_count": len(steps),
                "human_checkpoints": sum(len(step.human_checkpoints) for step in steps),
                "estimated_duration_hours": len(steps) * 2,  # Rough estimate
                "description": f"Enterprise {name.replace('_', ' ').title()} workflow",
            }
            for name, steps in self.blueprint_templates.items()
        }


# Global instances
_human_oversight_engine: Optional[HumanOversightEngine] = None
_blueprint_orchestrator: Optional[BlueprintWorkflowOrchestrator] = None


async def get_human_oversight_engine() -> HumanOversightEngine:
    """Get or create global human oversight engine"""
    global _human_oversight_engine

    if _human_oversight_engine is None:
        _human_oversight_engine = HumanOversightEngine()

    return _human_oversight_engine


async def get_blueprint_orchestrator() -> BlueprintWorkflowOrchestrator:
    """Get or create global blueprint workflow orchestrator"""
    global _blueprint_orchestrator

    if _blueprint_orchestrator is None:
        oversight_engine = await get_human_oversight_engine()
        _blueprint_orchestrator = BlueprintWorkflowOrchestrator(oversight_engine)

    return _blueprint_orchestrator
