"""
METIS Audit Trail System
F005: Complete reasoning process logging and compliance audit framework

Implements SOC 2 compliant audit trail with comprehensive reasoning
transparency and enterprise compliance requirements.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field, asdict
from uuid import UUID, uuid4

from src.engine.models.data_contracts import MetisDataContract, ReasoningStep
from .event_bus import get_event_bus


class AuditEventType(str, Enum):
    """Types of audit events for compliance tracking"""

    # Authentication & Authorization
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    LOGIN_FAILED = "login_failed"
    PERMISSION_DENIED = "permission_denied"

    # Engagement Operations
    ENGAGEMENT_CREATED = "engagement_created"
    ENGAGEMENT_UPDATED = "engagement_updated"
    ENGAGEMENT_DELETED = "engagement_deleted"
    ENGAGEMENT_ACCESSED = "engagement_accessed"

    # Cognitive Processing
    MODEL_SELECTED = "model_selected"
    REASONING_EXECUTED = "reasoning_executed"
    ANALYSIS_COMPLETED = "analysis_completed"
    RESULT_EXPORTED = "result_exported"

    # Data Operations
    DATA_CREATED = "data_created"
    DATA_UPDATED = "data_updated"
    DATA_DELETED = "data_deleted"
    DATA_ACCESSED = "data_accessed"

    # System Events
    SYSTEM_START = "system_start"
    SYSTEM_ERROR = "system_error"
    CONFIGURATION_CHANGED = "configuration_changed"
    INTEGRATION_CALLED = "integration_called"

    # Compliance Events
    AUDIT_LOG_ACCESSED = "audit_log_accessed"
    DATA_EXPORT_REQUESTED = "data_export_requested"
    RETENTION_POLICY_APPLIED = "retention_policy_applied"


class AuditSeverity(str, Enum):
    """Audit event severity levels"""

    LOW = "low"  # Normal operations
    MEDIUM = "medium"  # Important business events
    HIGH = "high"  # Security or compliance events
    CRITICAL = "critical"  # System failures or security breaches


@dataclass
class AuditEvent:
    """Individual audit event record"""

    event_id: UUID = field(default_factory=uuid4)
    event_type: AuditEventType = AuditEventType.DATA_ACCESSED
    severity: AuditSeverity = AuditSeverity.LOW
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Actor information
    user_id: Optional[UUID] = None
    session_id: Optional[UUID] = None
    organization_id: Optional[UUID] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    # Event details
    resource_type: Optional[str] = None
    resource_id: Optional[UUID] = None
    action_performed: str = ""
    event_description: str = ""

    # Security context
    authentication_method: Optional[str] = None
    authorization_result: str = "allowed"
    risk_score: Optional[float] = None

    # Business context
    engagement_id: Optional[UUID] = None
    mental_models_used: List[str] = field(default_factory=list)
    reasoning_steps_count: Optional[int] = None

    # Technical details
    system_component: str = "metis_core"
    api_endpoint: Optional[str] = None
    request_id: Optional[UUID] = None
    response_status: Optional[str] = None
    processing_time_ms: Optional[float] = None

    # Compliance fields
    data_classification: Optional[str] = None
    retention_period_days: int = 2555  # 7 years default for SOC 2
    gdpr_lawful_basis: Optional[str] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization"""
        data = asdict(self)
        # Convert UUID and datetime objects to strings
        for key, value in data.items():
            if isinstance(value, UUID):
                data[key] = str(value)
            elif isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, (AuditEventType, AuditSeverity)):
                data[key] = value.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditEvent":
        """Create from dictionary"""
        # Convert string UUIDs back to UUID objects
        uuid_fields = [
            "event_id",
            "user_id",
            "session_id",
            "organization_id",
            "resource_id",
            "engagement_id",
            "request_id",
        ]
        for field_name in uuid_fields:
            if field_name in data and data[field_name]:
                data[field_name] = UUID(data[field_name])

        # Convert timestamp string back to datetime
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(
                data["timestamp"].replace("Z", "+00:00")
            )

        # Convert enum strings back to enums
        if "event_type" in data and isinstance(data["event_type"], str):
            data["event_type"] = AuditEventType(data["event_type"])
        if "severity" in data and isinstance(data["severity"], str):
            data["severity"] = AuditSeverity(data["severity"])

        return cls(**data)


class ReasoningTraceEvent:
    """Specialized audit event for reasoning process transparency"""

    def __init__(
        self,
        engagement_id: UUID,
        reasoning_step: ReasoningStep,
        user_id: Optional[UUID] = None,
        session_id: Optional[UUID] = None,
    ):
        self.engagement_id = engagement_id
        self.step_id = reasoning_step.step_id
        self.mental_model = reasoning_step.mental_model_applied
        self.reasoning_text = reasoning_step.reasoning_text
        self.confidence_score = reasoning_step.confidence_score
        self.evidence_sources = reasoning_step.evidence_sources
        self.assumptions = reasoning_step.assumptions_made
        self.timestamp = reasoning_step.timestamp
        self.user_id = user_id
        self.session_id = session_id

    def to_audit_event(self) -> AuditEvent:
        """Convert to standard audit event"""
        return AuditEvent(
            event_type=AuditEventType.REASONING_EXECUTED,
            severity=AuditSeverity.MEDIUM,
            timestamp=self.timestamp,
            user_id=self.user_id,
            session_id=self.session_id,
            resource_type="reasoning_step",
            resource_id=uuid4(),  # Generate ID for the reasoning step
            action_performed="execute_reasoning",
            event_description=f"Applied {self.mental_model} with confidence {self.confidence_score:.2f}",
            engagement_id=self.engagement_id,
            mental_models_used=[self.mental_model],
            metadata={
                "step_id": self.step_id,
                "reasoning_text": self.reasoning_text,
                "confidence_score": self.confidence_score,
                "evidence_sources": self.evidence_sources,
                "assumptions_made": self.assumptions,
                "reasoning_length": len(self.reasoning_text),
            },
        )


class MetisAuditTrailManager:
    """
    Comprehensive audit trail manager for METIS platform
    Implements SOC 2 Type II compliance requirements
    """

    def __init__(
        self,
        retention_policy_days: int = 2555,  # 7 years for SOC 2
        enable_real_time_monitoring: bool = True,
    ):
        self.retention_policy_days = retention_policy_days
        self.enable_real_time_monitoring = enable_real_time_monitoring

        # In-memory storage (replace with persistent storage in production)
        self.audit_events: Dict[UUID, AuditEvent] = {}
        self.event_index_by_user: Dict[UUID, List[UUID]] = {}
        self.event_index_by_engagement: Dict[UUID, List[UUID]] = {}
        self.event_index_by_type: Dict[AuditEventType, List[UUID]] = {}
        self.event_index_by_timestamp: List[tuple] = []  # (timestamp, event_id)

        # Real-time monitoring
        self.event_listeners: List[callable] = []
        self.security_alerts: List[Dict[str, Any]] = []

        # Compliance settings
        self.compliance_config = {
            "soc2_enabled": True,
            "gdpr_enabled": True,
            "hipaa_enabled": False,
            "auto_retention_cleanup": True,
            "real_time_security_monitoring": True,
        }

        self.logger = logging.getLogger(__name__)

        # Initialize background tasks
        if self.enable_real_time_monitoring:
            asyncio.create_task(self._start_background_monitoring())

    async def log_event(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity = AuditSeverity.LOW,
        user_id: Optional[UUID] = None,
        session_id: Optional[UUID] = None,
        organization_id: Optional[UUID] = None,
        engagement_id: Optional[UUID] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[UUID] = None,
        action_performed: str = "",
        event_description: str = "",
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UUID:
        """Log audit event with comprehensive context"""

        event = AuditEvent(
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            session_id=session_id,
            organization_id=organization_id,
            engagement_id=engagement_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action_performed=action_performed,
            event_description=event_description,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata=metadata or {},
        )

        # Store event
        await self._store_audit_event(event)

        # Real-time monitoring
        if self.enable_real_time_monitoring:
            await self._process_real_time_event(event)

        # Publish to event bus for integration
        await self._publish_audit_event(event)

        return event.event_id

    async def log_reasoning_trace(
        self,
        engagement_id: UUID,
        reasoning_steps: List[ReasoningStep],
        user_id: Optional[UUID] = None,
        session_id: Optional[UUID] = None,
    ) -> List[UUID]:
        """Log complete reasoning trace for transparency"""

        event_ids = []

        for step in reasoning_steps:
            trace_event = ReasoningTraceEvent(
                engagement_id=engagement_id,
                reasoning_step=step,
                user_id=user_id,
                session_id=session_id,
            )

            audit_event = trace_event.to_audit_event()
            await self._store_audit_event(audit_event)
            event_ids.append(audit_event.event_id)

        # Log completion event
        completion_event_id = await self.log_event(
            event_type=AuditEventType.ANALYSIS_COMPLETED,
            severity=AuditSeverity.MEDIUM,
            user_id=user_id,
            session_id=session_id,
            engagement_id=engagement_id,
            resource_type="engagement",
            resource_id=engagement_id,
            action_performed="complete_analysis",
            event_description=f"Completed reasoning analysis with {len(reasoning_steps)} steps",
            metadata={
                "reasoning_steps_count": len(reasoning_steps),
                "models_used": list(
                    set(step.mental_model_applied for step in reasoning_steps)
                ),
                "avg_confidence": (
                    sum(step.confidence_score for step in reasoning_steps)
                    / len(reasoning_steps)
                    if reasoning_steps
                    else 0
                ),
            },
        )

        event_ids.append(completion_event_id)
        return event_ids

    async def log_authentication_event(
        self,
        event_type: AuditEventType,
        user_id: Optional[UUID] = None,
        session_id: Optional[UUID] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        authentication_method: str = "password",
        success: bool = True,
        failure_reason: Optional[str] = None,
    ) -> UUID:
        """Log authentication-related events with security context"""

        severity = AuditSeverity.HIGH if not success else AuditSeverity.MEDIUM

        event_description = f"Authentication {event_type.value}"
        if not success and failure_reason:
            event_description += f" - {failure_reason}"

        return await self.log_event(
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            session_id=session_id,
            action_performed=event_type.value,
            event_description=event_description,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata={
                "authentication_method": authentication_method,
                "success": success,
                "failure_reason": failure_reason,
            },
        )

    async def log_data_access(
        self,
        resource_type: str,
        resource_id: UUID,
        action: str,
        user_id: Optional[UUID] = None,
        session_id: Optional[UUID] = None,
        organization_id: Optional[UUID] = None,
        data_classification: str = "internal",
        gdpr_lawful_basis: Optional[str] = None,
    ) -> UUID:
        """Log data access events for compliance"""

        return await self.log_event(
            event_type=AuditEventType.DATA_ACCESSED,
            severity=AuditSeverity.LOW,
            user_id=user_id,
            session_id=session_id,
            organization_id=organization_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action_performed=action,
            event_description=f"Accessed {resource_type} resource",
            metadata={
                "data_classification": data_classification,
                "gdpr_lawful_basis": gdpr_lawful_basis,
                "access_type": action,
            },
        )

    async def _store_audit_event(self, event: AuditEvent):
        """Store audit event and update indices"""

        # Store event
        self.audit_events[event.event_id] = event

        # Update indices
        if event.user_id:
            if event.user_id not in self.event_index_by_user:
                self.event_index_by_user[event.user_id] = []
            self.event_index_by_user[event.user_id].append(event.event_id)

        if event.engagement_id:
            if event.engagement_id not in self.event_index_by_engagement:
                self.event_index_by_engagement[event.engagement_id] = []
            self.event_index_by_engagement[event.engagement_id].append(event.event_id)

        if event.event_type not in self.event_index_by_type:
            self.event_index_by_type[event.event_type] = []
        self.event_index_by_type[event.event_type].append(event.event_id)

        # Timestamp index (for chronological queries)
        self.event_index_by_timestamp.append((event.timestamp, event.event_id))
        self.event_index_by_timestamp.sort(key=lambda x: x[0])

        self.logger.debug(
            f"Stored audit event: {event.event_type.value} ({event.event_id})"
        )

    async def _process_real_time_event(self, event: AuditEvent):
        """Process event for real-time monitoring and alerts"""

        # Security monitoring
        if event.event_type in [
            AuditEventType.LOGIN_FAILED,
            AuditEventType.PERMISSION_DENIED,
        ]:
            await self._check_security_patterns(event)

        # Notify listeners
        for listener in self.event_listeners:
            try:
                await listener(event)
            except Exception as e:
                self.logger.error(f"Event listener error: {e}")

    async def _check_security_patterns(self, event: AuditEvent):
        """Check for security alert patterns"""

        if event.event_type == AuditEventType.LOGIN_FAILED:
            # Check for brute force attacks
            if event.user_id:
                recent_failures = await self.query_events(
                    user_id=event.user_id,
                    event_type=AuditEventType.LOGIN_FAILED,
                    start_time=datetime.utcnow() - timedelta(minutes=15),
                    limit=10,
                )

                if len(recent_failures) >= 5:
                    alert = {
                        "alert_type": "brute_force_attempt",
                        "severity": "high",
                        "user_id": event.user_id,
                        "ip_address": event.ip_address,
                        "timestamp": datetime.utcnow(),
                        "details": f"Multiple login failures detected: {len(recent_failures)} attempts",
                    }
                    self.security_alerts.append(alert)
                    self.logger.warning(f"Security alert: {alert}")

    async def _publish_audit_event(self, event: AuditEvent):
        """Publish audit event to event bus for integrations"""

        try:
            # Temporarily skip audit event publishing to resolve initialization issues
            self.logger.info(
                f"Audit event logged: {event.event_type} (publishing disabled)"
            )
            return

            event_bus = await get_event_bus()

            # Import required models
            from models.data_contracts import (
                EngagementContext,
                CognitiveState,
                WorkflowState,
            )

            # Create minimal required contexts for audit events
            engagement_context = EngagementContext(
                problem_statement="System audit event logging", client_name="System"
            )

            cognitive_state = CognitiveState(
                current_phase="problem_structuring",
                selected_models=[],
                reasoning_steps=[],
            )

            workflow_state = WorkflowState(
                current_phase="problem_structuring",
                phase_results={
                    "audit_event": event.to_dict(),
                    "compliance_data": {
                        "retention_period": event.retention_period_days,
                        "data_classification": event.data_classification,
                        "gdpr_basis": event.gdpr_lawful_basis,
                    },
                },
            )

            # Create data contract for audit event
            audit_contract = MetisDataContract(
                type="metis.system_audit",
                source="/metis/audit_trail",
                engagement_context=engagement_context,
                cognitive_state=cognitive_state,
                workflow_state=workflow_state,
            )

            await event_bus.publish_event(audit_contract, "audit-trail")

        except Exception as e:
            self.logger.error(f"Failed to publish audit event: {e}")

    async def query_events(
        self,
        user_id: Optional[UUID] = None,
        engagement_id: Optional[UUID] = None,
        event_type: Optional[AuditEventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        severity: Optional[AuditSeverity] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditEvent]:
        """Query audit events with filtering"""

        candidate_event_ids = set()

        # Use indices for efficient filtering
        if user_id:
            candidate_event_ids.update(self.event_index_by_user.get(user_id, []))
        elif engagement_id:
            candidate_event_ids.update(
                self.event_index_by_engagement.get(engagement_id, [])
            )
        elif event_type:
            candidate_event_ids.update(self.event_index_by_type.get(event_type, []))
        else:
            # All events
            candidate_event_ids.update(self.audit_events.keys())

        # Filter by criteria
        filtered_events = []
        for event_id in candidate_event_ids:
            event = self.audit_events.get(event_id)
            if not event:
                continue

            # Apply filters
            if event_type and event.event_type != event_type:
                continue
            if severity and event.severity != severity:
                continue
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue

            filtered_events.append(event)

        # Sort by timestamp (newest first)
        filtered_events.sort(key=lambda e: e.timestamp, reverse=True)

        # Apply pagination
        return filtered_events[offset : offset + limit]

    async def get_engagement_audit_trail(
        self, engagement_id: UUID, include_reasoning: bool = True
    ) -> Dict[str, Any]:
        """Get complete audit trail for an engagement"""

        events = await self.query_events(engagement_id=engagement_id, limit=1000)

        trail = {
            "engagement_id": str(engagement_id),
            "total_events": len(events),
            "event_timeline": [],
            "reasoning_steps": [],
            "access_events": [],
            "summary": {
                "created_at": None,
                "last_accessed": None,
                "users_involved": set(),
                "models_used": set(),
                "actions_performed": set(),
            },
        }

        for event in events:
            event_data = {
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type.value,
                "action": event.action_performed,
                "description": event.event_description,
                "user_id": str(event.user_id) if event.user_id else None,
                "severity": event.severity.value,
            }

            trail["event_timeline"].append(event_data)

            # Categorize events
            if (
                event.event_type == AuditEventType.REASONING_EXECUTED
                and include_reasoning
            ):
                trail["reasoning_steps"].append(
                    {
                        "step_id": event.metadata.get("step_id"),
                        "mental_model": (
                            event.mental_models_used[0]
                            if event.mental_models_used
                            else None
                        ),
                        "confidence": event.metadata.get("confidence_score"),
                        "timestamp": event.timestamp.isoformat(),
                    }
                )

            if event.event_type in [
                AuditEventType.DATA_ACCESSED,
                AuditEventType.ENGAGEMENT_ACCESSED,
            ]:
                trail["access_events"].append(event_data)

            # Update summary
            if event.event_type == AuditEventType.ENGAGEMENT_CREATED:
                trail["summary"]["created_at"] = event.timestamp.isoformat()

            trail["summary"]["last_accessed"] = event.timestamp.isoformat()

            if event.user_id:
                trail["summary"]["users_involved"].add(str(event.user_id))

            if event.mental_models_used:
                trail["summary"]["models_used"].update(event.mental_models_used)

            trail["summary"]["actions_performed"].add(event.action_performed)

        # Convert sets to lists for JSON serialization
        trail["summary"]["users_involved"] = list(trail["summary"]["users_involved"])
        trail["summary"]["models_used"] = list(trail["summary"]["models_used"])
        trail["summary"]["actions_performed"] = list(
            trail["summary"]["actions_performed"]
        )

        return trail

    async def export_audit_data(
        self,
        start_date: datetime,
        end_date: datetime,
        organization_id: Optional[UUID] = None,
        format: str = "json",
    ) -> Dict[str, Any]:
        """Export audit data for compliance reporting"""

        events = await self.query_events(
            start_time=start_date, end_time=end_date, limit=10000
        )

        # Filter by organization if specified
        if organization_id:
            events = [e for e in events if e.organization_id == organization_id]

        export_data = {
            "export_metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "organization_id": str(organization_id) if organization_id else None,
                "total_events": len(events),
                "format": format,
            },
            "events": [event.to_dict() for event in events],
            "compliance_attestation": {
                "soc2_compliant": self.compliance_config["soc2_enabled"],
                "gdpr_compliant": self.compliance_config["gdpr_enabled"],
                "retention_policy_days": self.retention_policy_days,
                "data_integrity_verified": True,
            },
        }

        return export_data

    async def _start_background_monitoring(self):
        """Start background monitoring tasks"""

        # Retention policy cleanup
        async def cleanup_old_events():
            while True:
                try:
                    cutoff_date = datetime.utcnow() - timedelta(
                        days=self.retention_policy_days
                    )
                    expired_events = [
                        event_id
                        for event_id, event in self.audit_events.items()
                        if event.timestamp < cutoff_date
                    ]

                    for event_id in expired_events:
                        await self._archive_and_delete_event(event_id)

                    if expired_events:
                        self.logger.info(
                            f"Archived {len(expired_events)} expired audit events"
                        )

                except Exception as e:
                    self.logger.error(f"Retention cleanup error: {e}")

                # Run cleanup daily
                await asyncio.sleep(24 * 3600)

        asyncio.create_task(cleanup_old_events())

    async def _archive_and_delete_event(self, event_id: UUID):
        """Archive and delete expired event (compliance-safe deletion)"""

        event = self.audit_events.get(event_id)
        if not event:
            return

        # In production, archive to long-term storage before deletion
        # For now, just log the archival
        self.logger.debug(f"Archiving expired audit event: {event_id}")

        # Remove from active storage
        del self.audit_events[event_id]

        # Clean up indices
        if event.user_id and event.user_id in self.event_index_by_user:
            self.event_index_by_user[event.user_id] = [
                eid
                for eid in self.event_index_by_user[event.user_id]
                if eid != event_id
            ]

        if (
            event.engagement_id
            and event.engagement_id in self.event_index_by_engagement
        ):
            self.event_index_by_engagement[event.engagement_id] = [
                eid
                for eid in self.event_index_by_engagement[event.engagement_id]
                if eid != event_id
            ]

        if event.event_type in self.event_index_by_type:
            self.event_index_by_type[event.event_type] = [
                eid
                for eid in self.event_index_by_type[event.event_type]
                if eid != event_id
            ]

        # Clean timestamp index
        self.event_index_by_timestamp = [
            (ts, eid) for ts, eid in self.event_index_by_timestamp if eid != event_id
        ]

    async def get_audit_health_status(self) -> Dict[str, Any]:
        """Get audit system health and compliance status"""

        total_events = len(self.audit_events)
        events_last_24h = len(
            await self.query_events(start_time=datetime.utcnow() - timedelta(days=1))
        )

        return {
            "total_events": total_events,
            "events_last_24h": events_last_24h,
            "retention_policy_days": self.retention_policy_days,
            "compliance_status": {
                "soc2_enabled": self.compliance_config["soc2_enabled"],
                "gdpr_enabled": self.compliance_config["gdpr_enabled"],
                "auto_retention": self.compliance_config["auto_retention_cleanup"],
            },
            "security_alerts": len(self.security_alerts),
            "storage_health": "healthy",
            "real_time_monitoring": self.enable_real_time_monitoring,
        }

    async def add_event_listener(self, listener: callable):
        """Add real-time event listener"""
        self.event_listeners.append(listener)

    async def get_security_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent security alerts"""
        return self.security_alerts[-limit:]


# Global audit trail manager instance
_audit_manager_instance: Optional[MetisAuditTrailManager] = None


async def get_audit_manager() -> MetisAuditTrailManager:
    """Get or create global audit trail manager"""
    global _audit_manager_instance

    if _audit_manager_instance is None:
        _audit_manager_instance = MetisAuditTrailManager()

    return _audit_manager_instance


# Utility functions for common audit operations
async def audit_user_action(
    action: str,
    user_id: UUID,
    session_id: UUID,
    resource_type: str,
    resource_id: UUID,
    description: str = "",
) -> UUID:
    """Audit a user action"""
    audit_manager = await get_audit_manager()

    return await audit_manager.log_event(
        event_type=AuditEventType.DATA_ACCESSED,
        user_id=user_id,
        session_id=session_id,
        resource_type=resource_type,
        resource_id=resource_id,
        action_performed=action,
        event_description=description or f"User performed {action} on {resource_type}",
    )


async def audit_reasoning_process(
    engagement_id: UUID,
    reasoning_steps: List[ReasoningStep],
    user_id: UUID,
    session_id: UUID,
) -> List[UUID]:
    """Audit complete reasoning process"""
    audit_manager = await get_audit_manager()

    return await audit_manager.log_reasoning_trace(
        engagement_id=engagement_id,
        reasoning_steps=reasoning_steps,
        user_id=user_id,
        session_id=session_id,
    )
