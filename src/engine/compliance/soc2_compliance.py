"""
METIS SOC 2 Compliance Framework
E003: Enterprise-grade SOC 2 Type II compliance implementation

Implements comprehensive SOC 2 compliance framework covering all five trust criteria:
Security, Availability, Processing Integrity, Confidentiality, and Privacy.
"""

import asyncio
import logging
import json
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from uuid import uuid4
import threading
from collections import defaultdict

try:
    import cryptography
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


class SOC2Criteria(str, Enum):
    """SOC 2 Trust Service Criteria"""

    SECURITY = "security"
    AVAILABILITY = "availability"
    PROCESSING_INTEGRITY = "processing_integrity"
    CONFIDENTIALITY = "confidentiality"
    PRIVACY = "privacy"


class ComplianceStatus(str, Enum):
    """Compliance status levels"""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNDER_REVIEW = "under_review"
    REMEDIATION_REQUIRED = "remediation_required"


class AuditEventType(str, Enum):
    """Types of auditable events"""

    USER_ACCESS = "user_access"
    DATA_ACCESS = "data_access"
    SYSTEM_CHANGE = "system_change"
    SECURITY_EVENT = "security_event"
    COMPLIANCE_CHECK = "compliance_check"
    INCIDENT = "incident"
    BACKUP_OPERATION = "backup_operation"
    CONFIGURATION_CHANGE = "configuration_change"


class RiskLevel(str, Enum):
    """Risk assessment levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ComplianceControl:
    """Individual SOC 2 compliance control"""

    control_id: str = ""
    criteria: SOC2Criteria = SOC2Criteria.SECURITY
    title: str = ""
    description: str = ""
    implementation_guidance: str = ""

    # Status tracking
    status: ComplianceStatus = ComplianceStatus.UNDER_REVIEW
    last_assessment: Optional[datetime] = None
    next_assessment: Optional[datetime] = None

    # Evidence and documentation
    evidence_required: List[str] = field(default_factory=list)
    evidence_collected: List[str] = field(default_factory=list)
    documentation_links: List[str] = field(default_factory=list)

    # Risk assessment
    risk_level: RiskLevel = RiskLevel.MEDIUM
    risk_description: str = ""
    mitigation_measures: List[str] = field(default_factory=list)

    # Testing
    testing_frequency: str = "quarterly"  # monthly, quarterly, annually
    last_test_date: Optional[datetime] = None
    test_results: List[Dict[str, Any]] = field(default_factory=list)

    # Responsible parties
    control_owner: str = ""
    assessor: str = ""
    reviewer: str = ""


@dataclass
class AuditEvent:
    """SOC 2 audit event record"""

    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_type: AuditEventType = AuditEventType.SYSTEM_CHANGE

    # Event details
    user_id: str = ""
    tenant_id: str = ""
    resource: str = ""
    action: str = ""

    # Context
    ip_address: str = ""
    user_agent: str = ""
    session_id: str = ""

    # Data
    before_state: Optional[Dict[str, Any]] = None
    after_state: Optional[Dict[str, Any]] = None
    sensitive_data_involved: bool = False

    # Classification
    criteria_affected: List[SOC2Criteria] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW

    # Integrity
    checksum: str = ""
    signed: bool = False

    def __post_init__(self):
        """Calculate checksum for integrity"""
        if not self.checksum:
            event_data = f"{self.timestamp}{self.event_type}{self.user_id}{self.action}"
            self.checksum = hashlib.sha256(event_data.encode()).hexdigest()


@dataclass
class SecurityIncident:
    """Security incident tracking for SOC 2"""

    incident_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Classification
    incident_type: str = ""
    severity: str = "medium"  # low, medium, high, critical
    status: str = "open"  # open, investigating, resolved, closed

    # Description
    title: str = ""
    description: str = ""
    affected_systems: List[str] = field(default_factory=list)
    affected_data: List[str] = field(default_factory=list)

    # Response
    response_team: List[str] = field(default_factory=list)
    containment_actions: List[str] = field(default_factory=list)
    recovery_actions: List[str] = field(default_factory=list)

    # Timeline
    detection_time: Optional[datetime] = None
    containment_time: Optional[datetime] = None
    resolution_time: Optional[datetime] = None

    # SOC 2 specific
    criteria_affected: List[SOC2Criteria] = field(default_factory=list)
    compliance_impact: str = ""
    notification_required: bool = False

    # Documentation
    evidence: List[str] = field(default_factory=list)
    lessons_learned: str = ""


class AuditLogger:
    """SOC 2 compliant audit logging system"""

    def __init__(self, encryption_key: Optional[bytes] = None):
        self.encryption_key = encryption_key or self._generate_encryption_key()
        self.audit_events: List[AuditEvent] = []
        self.audit_index: Dict[str, List[str]] = defaultdict(list)
        self.retention_days = 2555  # 7 years for SOC 2

        self.logger = logging.getLogger(__name__)

        # Configure tamper-evident logging
        self._setup_tamper_evident_logging()

    def _generate_encryption_key(self) -> bytes:
        """Generate encryption key for audit logs"""
        if CRYPTO_AVAILABLE:
            return Fernet.generate_key()
        return b"dummy_key_for_development"

    def _setup_tamper_evident_logging(self):
        """Setup tamper-evident audit logging"""
        # Configure write-only audit log storage
        # In production, this would use immutable storage
        pass

    async def log_audit_event(
        self,
        event_type: AuditEventType,
        user_id: str,
        action: str,
        resource: str = "",
        tenant_id: str = "",
        before_state: Optional[Dict[str, Any]] = None,
        after_state: Optional[Dict[str, Any]] = None,
        ip_address: str = "",
        user_agent: str = "",
        session_id: str = "",
        sensitive_data_involved: bool = False,
        criteria_affected: List[SOC2Criteria] = None,
        risk_level: RiskLevel = RiskLevel.LOW,
    ) -> str:
        """Log SOC 2 audit event"""

        event = AuditEvent(
            event_type=event_type,
            user_id=user_id,
            tenant_id=tenant_id,
            resource=resource,
            action=action,
            before_state=before_state,
            after_state=after_state,
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id,
            sensitive_data_involved=sensitive_data_involved,
            criteria_affected=criteria_affected or [],
            risk_level=risk_level,
        )

        # Encrypt sensitive data if present
        if sensitive_data_involved:
            event = await self._encrypt_sensitive_event_data(event)

        # Store event
        self.audit_events.append(event)

        # Update indexes
        self.audit_index["user_id"].append(event.event_id)
        self.audit_index["tenant_id"].append(event.event_id)
        self.audit_index["event_type"].append(event.event_id)
        self.audit_index["timestamp"].append(event.event_id)

        # Log to system logger
        self.logger.info(
            f"SOC2_AUDIT: {event.event_type.value} | "
            f"User: {user_id} | Action: {action} | "
            f"Resource: {resource} | ID: {event.event_id}"
        )

        return event.event_id

    async def _encrypt_sensitive_event_data(self, event: AuditEvent) -> AuditEvent:
        """Encrypt sensitive data in audit event"""
        if not CRYPTO_AVAILABLE:
            return event

        fernet = Fernet(self.encryption_key)

        # Encrypt before/after states if they exist
        if event.before_state:
            encrypted_before = fernet.encrypt(json.dumps(event.before_state).encode())
            event.before_state = {"encrypted": encrypted_before.decode()}

        if event.after_state:
            encrypted_after = fernet.encrypt(json.dumps(event.after_state).encode())
            event.after_state = {"encrypted": encrypted_after.decode()}

        return event

    async def search_audit_events(
        self,
        criteria: Dict[str, Any],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[AuditEvent]:
        """Search audit events with filters"""

        filtered_events = []

        for event in self.audit_events:
            # Time range filter
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue

            # Criteria filters
            matches = True
            for key, value in criteria.items():
                if hasattr(event, key):
                    event_value = getattr(event, key)
                    if isinstance(event_value, list):
                        if value not in event_value:
                            matches = False
                            break
                    elif event_value != value:
                        matches = False
                        break

            if matches:
                filtered_events.append(event)

        return filtered_events

    async def generate_audit_report(
        self, criteria: SOC2Criteria, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """Generate SOC 2 audit report for specific criteria"""

        relevant_events = await self.search_audit_events(
            {"criteria_affected": criteria}, start_date, end_date
        )

        # Analyze events
        event_count = len(relevant_events)
        high_risk_events = [
            e for e in relevant_events if e.risk_level == RiskLevel.HIGH
        ]
        critical_events = [
            e for e in relevant_events if e.risk_level == RiskLevel.CRITICAL
        ]

        # Event distribution
        event_types = defaultdict(int)
        for event in relevant_events:
            event_types[event.event_type.value] += 1

        return {
            "criteria": criteria.value,
            "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "summary": {
                "total_events": event_count,
                "high_risk_events": len(high_risk_events),
                "critical_events": len(critical_events),
                "risk_distribution": {
                    "low": len(
                        [e for e in relevant_events if e.risk_level == RiskLevel.LOW]
                    ),
                    "medium": len(
                        [e for e in relevant_events if e.risk_level == RiskLevel.MEDIUM]
                    ),
                    "high": len(high_risk_events),
                    "critical": len(critical_events),
                },
            },
            "event_types": dict(event_types),
            "compliance_status": self._assess_criteria_compliance(
                criteria, relevant_events
            ),
            "generated_at": datetime.utcnow().isoformat(),
        }

    def _assess_criteria_compliance(
        self, criteria: SOC2Criteria, events: List[AuditEvent]
    ) -> str:
        """Assess compliance status based on events"""

        critical_events = [e for e in events if e.risk_level == RiskLevel.CRITICAL]
        high_risk_events = [e for e in events if e.risk_level == RiskLevel.HIGH]

        if critical_events:
            return ComplianceStatus.NON_COMPLIANT.value
        elif len(high_risk_events) > 5:
            return ComplianceStatus.PARTIALLY_COMPLIANT.value
        else:
            return ComplianceStatus.COMPLIANT.value

    async def cleanup_old_events(self):
        """Clean up events beyond retention period"""
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)

        self.audit_events = [
            event for event in self.audit_events if event.timestamp > cutoff_date
        ]

        self.logger.info(
            f"Cleaned up audit events older than {self.retention_days} days"
        )


class SOC2ComplianceManager:
    """Main SOC 2 compliance management system"""

    def __init__(self):
        self.controls: Dict[str, ComplianceControl] = {}
        self.audit_logger = AuditLogger()
        self.incidents: Dict[str, SecurityIncident] = {}

        # Monitoring
        self.monitoring_active = False
        self.monitoring_thread = None

        self.logger = logging.getLogger(__name__)

        # Initialize standard SOC 2 controls
        self._initialize_soc2_controls()

    def _initialize_soc2_controls(self):
        """Initialize standard SOC 2 Type II controls"""

        # Security Controls
        self.controls["SEC-1.1"] = ComplianceControl(
            control_id="SEC-1.1",
            criteria=SOC2Criteria.SECURITY,
            title="Entity Logical and Physical Access Controls",
            description="The entity implements logical and physical access controls to protect against threats to the achievement of its objectives",
            implementation_guidance="Implement multi-factor authentication, role-based access controls, and physical security measures",
            evidence_required=[
                "Access control policies",
                "User access reviews",
                "MFA implementation evidence",
                "Physical security assessments",
            ],
            risk_level=RiskLevel.HIGH,
            testing_frequency="quarterly",
            control_owner="Security Team",
        )

        self.controls["SEC-1.2"] = ComplianceControl(
            control_id="SEC-1.2",
            criteria=SOC2Criteria.SECURITY,
            title="System Operations - Detection and Monitoring",
            description="The entity implements detection and monitoring procedures to identify security events",
            implementation_guidance="Deploy SIEM, implement continuous monitoring, and establish incident response procedures",
            evidence_required=[
                "Monitoring system configuration",
                "Security event logs",
                "Incident response procedures",
                "Detection capability testing",
            ],
            risk_level=RiskLevel.HIGH,
            testing_frequency="monthly",
            control_owner="Security Operations",
        )

        # Availability Controls
        self.controls["AVL-2.1"] = ComplianceControl(
            control_id="AVL-2.1",
            criteria=SOC2Criteria.AVAILABILITY,
            title="System Availability Monitoring",
            description="The entity monitors system availability and implements procedures to maintain agreed-upon availability levels",
            implementation_guidance="Implement uptime monitoring, SLA tracking, and automated failover procedures",
            evidence_required=[
                "Availability monitoring reports",
                "SLA agreements and performance",
                "Failover test results",
                "Capacity planning documentation",
            ],
            risk_level=RiskLevel.MEDIUM,
            testing_frequency="monthly",
            control_owner="Infrastructure Team",
        )

        # Processing Integrity Controls
        self.controls["PI-3.1"] = ComplianceControl(
            control_id="PI-3.1",
            criteria=SOC2Criteria.PROCESSING_INTEGRITY,
            title="Data Processing Accuracy and Completeness",
            description="The entity implements controls to ensure data processing is accurate, complete, and timely",
            implementation_guidance="Implement data validation, integrity checks, and error handling procedures",
            evidence_required=[
                "Data validation procedures",
                "Integrity check mechanisms",
                "Error logging and handling",
                "Processing accuracy reports",
            ],
            risk_level=RiskLevel.MEDIUM,
            testing_frequency="quarterly",
            control_owner="Data Engineering",
        )

        # Confidentiality Controls
        self.controls["CONF-4.1"] = ComplianceControl(
            control_id="CONF-4.1",
            criteria=SOC2Criteria.CONFIDENTIALITY,
            title="Data Classification and Encryption",
            description="The entity implements controls to protect confidential information through classification and encryption",
            implementation_guidance="Classify data sensitivity levels and implement appropriate encryption measures",
            evidence_required=[
                "Data classification policies",
                "Encryption implementation",
                "Key management procedures",
                "Access control to confidential data",
            ],
            risk_level=RiskLevel.HIGH,
            testing_frequency="quarterly",
            control_owner="Data Protection Officer",
        )

        # Privacy Controls
        self.controls["PRIV-5.1"] = ComplianceControl(
            control_id="PRIV-5.1",
            criteria=SOC2Criteria.PRIVACY,
            title="Personal Information Collection and Use",
            description="The entity implements controls over the collection, use, retention, and disposal of personal information",
            implementation_guidance="Implement privacy policies, consent mechanisms, and data lifecycle management",
            evidence_required=[
                "Privacy policies and notices",
                "Consent management records",
                "Data retention schedules",
                "Personal data deletion procedures",
            ],
            risk_level=RiskLevel.HIGH,
            testing_frequency="quarterly",
            control_owner="Privacy Officer",
        )

    async def assess_control_compliance(self, control_id: str) -> Dict[str, Any]:
        """Assess compliance for specific control"""

        if control_id not in self.controls:
            raise ValueError(f"Control {control_id} not found")

        control = self.controls[control_id]

        # Perform assessment
        assessment_result = {
            "control_id": control_id,
            "control_title": control.title,
            "criteria": control.criteria.value,
            "assessment_date": datetime.utcnow().isoformat(),
            "status": control.status.value,
            "risk_level": control.risk_level.value,
        }

        # Check evidence collection
        evidence_percentage = (
            len(control.evidence_collected) / len(control.evidence_required) * 100
            if control.evidence_required
            else 100
        )
        assessment_result["evidence_completeness"] = evidence_percentage

        # Check testing currency
        if control.last_test_date:
            days_since_test = (datetime.utcnow() - control.last_test_date).days
            testing_frequency_days = {"monthly": 30, "quarterly": 90, "annually": 365}
            expected_days = testing_frequency_days.get(control.testing_frequency, 90)
            assessment_result["testing_current"] = days_since_test <= expected_days
        else:
            assessment_result["testing_current"] = False

        # Overall compliance determination
        if (
            evidence_percentage >= 80
            and assessment_result["testing_current"]
            and control.status == ComplianceStatus.COMPLIANT
        ):
            assessment_result["compliance_rating"] = "COMPLIANT"
        elif evidence_percentage >= 50:
            assessment_result["compliance_rating"] = "PARTIALLY_COMPLIANT"
        else:
            assessment_result["compliance_rating"] = "NON_COMPLIANT"

        # Log assessment
        await self.audit_logger.log_audit_event(
            event_type=AuditEventType.COMPLIANCE_CHECK,
            user_id="system",
            action="control_assessment",
            resource=f"control_{control_id}",
            criteria_affected=[control.criteria],
            risk_level=control.risk_level,
        )

        return assessment_result

    async def generate_soc2_report(
        self,
        criteria: Optional[SOC2Criteria] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Generate comprehensive SOC 2 compliance report"""

        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=90)  # Last quarter
        if not end_date:
            end_date = datetime.utcnow()

        report = {
            "report_id": str(uuid4()),
            "generated_at": datetime.utcnow().isoformat(),
            "reporting_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "scope": "METIS Cognitive Intelligence Platform",
            "criteria_assessed": [],
        }

        # Assess each criteria
        criteria_list = [criteria] if criteria else list(SOC2Criteria)

        for criterion in criteria_list:
            criterion_controls = [
                c for c in self.controls.values() if c.criteria == criterion
            ]

            # Assess each control
            control_assessments = []
            for control in criterion_controls:
                assessment = await self.assess_control_compliance(control.control_id)
                control_assessments.append(assessment)

            # Generate audit events report
            audit_report = await self.audit_logger.generate_audit_report(
                criterion, start_date, end_date
            )

            # Determine overall criteria compliance
            compliant_controls = [
                a for a in control_assessments if a["compliance_rating"] == "COMPLIANT"
            ]

            compliance_percentage = (
                len(compliant_controls) / len(control_assessments) * 100
                if control_assessments
                else 0
            )

            if compliance_percentage >= 90:
                overall_status = "COMPLIANT"
            elif compliance_percentage >= 70:
                overall_status = "PARTIALLY_COMPLIANT"
            else:
                overall_status = "NON_COMPLIANT"

            criteria_report = {
                "criteria": criterion.value,
                "overall_status": overall_status,
                "compliance_percentage": compliance_percentage,
                "total_controls": len(control_assessments),
                "compliant_controls": len(compliant_controls),
                "control_assessments": control_assessments,
                "audit_summary": audit_report,
                "recommendations": self._generate_recommendations(
                    criterion, control_assessments
                ),
            }

            report["criteria_assessed"].append(criteria_report)

        # Overall report summary
        all_assessments = []
        for criteria_report in report["criteria_assessed"]:
            all_assessments.extend(criteria_report["control_assessments"])

        total_compliant = len(
            [a for a in all_assessments if a["compliance_rating"] == "COMPLIANT"]
        )
        overall_compliance = (
            total_compliant / len(all_assessments) * 100 if all_assessments else 0
        )

        report["executive_summary"] = {
            "overall_compliance_percentage": overall_compliance,
            "total_controls_assessed": len(all_assessments),
            "compliant_controls": total_compliant,
            "high_risk_findings": len(
                [a for a in all_assessments if a.get("risk_level") == "high"]
            ),
            "critical_findings": len(
                [a for a in all_assessments if a.get("risk_level") == "critical"]
            ),
            "overall_rating": self._determine_overall_rating(overall_compliance),
        }

        return report

    def _generate_recommendations(
        self, criteria: SOC2Criteria, assessments: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations for improving compliance"""

        recommendations = []

        non_compliant = [
            a for a in assessments if a["compliance_rating"] != "COMPLIANT"
        ]

        for assessment in non_compliant:
            control_id = assessment["control_id"]
            control = self.controls.get(control_id)

            if not control:
                continue

            if assessment["evidence_completeness"] < 80:
                recommendations.append(
                    f"Control {control_id}: Complete evidence collection - "
                    f"currently at {assessment['evidence_completeness']:.1f}%"
                )

            if not assessment["testing_current"]:
                recommendations.append(
                    f"Control {control_id}: Perform testing according to "
                    f"{control.testing_frequency} schedule"
                )

            if control.risk_level == RiskLevel.HIGH:
                recommendations.append(
                    f"Control {control_id}: High-risk control requires immediate attention"
                )

        # Add general recommendations by criteria
        if criteria == SOC2Criteria.SECURITY:
            recommendations.append(
                "Enhance security monitoring and incident response capabilities"
            )
        elif criteria == SOC2Criteria.AVAILABILITY:
            recommendations.append("Implement redundancy and disaster recovery testing")
        elif criteria == SOC2Criteria.CONFIDENTIALITY:
            recommendations.append("Strengthen data encryption and access controls")

        return recommendations

    def _determine_overall_rating(self, compliance_percentage: float) -> str:
        """Determine overall SOC 2 compliance rating"""
        if compliance_percentage >= 95:
            return "EXCELLENT"
        elif compliance_percentage >= 85:
            return "GOOD"
        elif compliance_percentage >= 70:
            return "SATISFACTORY"
        elif compliance_percentage >= 50:
            return "NEEDS_IMPROVEMENT"
        else:
            return "UNSATISFACTORY"

    async def record_security_incident(
        self,
        title: str,
        description: str,
        incident_type: str,
        severity: str = "medium",
        affected_systems: List[str] = None,
        affected_data: List[str] = None,
        criteria_affected: List[SOC2Criteria] = None,
    ) -> str:
        """Record security incident for SOC 2 tracking"""

        incident = SecurityIncident(
            title=title,
            description=description,
            incident_type=incident_type,
            severity=severity,
            affected_systems=affected_systems or [],
            affected_data=affected_data or [],
            criteria_affected=criteria_affected or [],
            detection_time=datetime.utcnow(),
        )

        self.incidents[incident.incident_id] = incident

        # Log incident in audit trail
        await self.audit_logger.log_audit_event(
            event_type=AuditEventType.INCIDENT,
            user_id="system",
            action="incident_created",
            resource=f"incident_{incident.incident_id}",
            criteria_affected=criteria_affected or [],
            risk_level=(
                RiskLevel.HIGH if severity in ["high", "critical"] else RiskLevel.MEDIUM
            ),
        )

        self.logger.warning(
            f"Security incident recorded: {incident.incident_id} - {title}"
        )

        return incident.incident_id

    async def start_continuous_monitoring(self):
        """Start continuous SOC 2 compliance monitoring"""
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

        self.logger.info("SOC 2 continuous monitoring started")

    async def stop_continuous_monitoring(self):
        """Stop continuous compliance monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)

        self.logger.info("SOC 2 continuous monitoring stopped")

    def _monitoring_loop(self):
        """Continuous monitoring loop for compliance"""
        while self.monitoring_active:
            try:
                # Perform periodic compliance checks
                for control_id in self.controls.keys():
                    # Check if control needs assessment
                    control = self.controls[control_id]

                    if (
                        control.next_assessment
                        and control.next_assessment <= datetime.utcnow()
                    ):
                        # Schedule assessment (would trigger actual assessment in production)
                        self.logger.info(f"Control {control_id} due for assessment")

                # Clean up old audit events
                asyncio.run(self.audit_logger.cleanup_old_events())

                # Sleep for monitoring interval
                threading.Event().wait(3600)  # Check every hour

            except Exception as e:
                self.logger.error(f"Monitoring loop error: {str(e)}")
                threading.Event().wait(300)  # Wait 5 minutes on error

    def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Get SOC 2 compliance dashboard data"""

        # Control status summary
        control_statuses = defaultdict(int)
        for control in self.controls.values():
            control_statuses[control.status.value] += 1

        # Criteria compliance summary
        criteria_compliance = {}
        for criteria in SOC2Criteria:
            criteria_controls = [
                c for c in self.controls.values() if c.criteria == criteria
            ]
            compliant_controls = [
                c for c in criteria_controls if c.status == ComplianceStatus.COMPLIANT
            ]
            if criteria_controls:
                compliance_rate = len(compliant_controls) / len(criteria_controls) * 100
            else:
                compliance_rate = 0
            criteria_compliance[criteria.value] = compliance_rate

        # Recent incidents
        recent_incidents = [
            i
            for i in self.incidents.values()
            if i.timestamp > datetime.utcnow() - timedelta(days=30)
        ]

        # Audit activity
        recent_audits = len(
            [
                e
                for e in self.audit_logger.audit_events
                if e.timestamp > datetime.utcnow() - timedelta(days=7)
            ]
        )

        return {
            "overall_status": {
                "total_controls": len(self.controls),
                "compliant_controls": control_statuses.get("compliant", 0),
                "non_compliant_controls": control_statuses.get("non_compliant", 0),
                "compliance_percentage": (
                    control_statuses.get("compliant", 0) / len(self.controls) * 100
                    if self.controls
                    else 0
                ),
            },
            "criteria_compliance": criteria_compliance,
            "control_status_distribution": dict(control_statuses),
            "recent_activity": {
                "incidents_last_30_days": len(recent_incidents),
                "audit_events_last_7_days": recent_audits,
                "monitoring_active": self.monitoring_active,
            },
            "high_priority_items": self._get_high_priority_items(),
            "last_updated": datetime.utcnow().isoformat(),
        }

    def _get_high_priority_items(self) -> List[Dict[str, Any]]:
        """Get high priority compliance items requiring attention"""

        items = []

        # Non-compliant high-risk controls
        for control in self.controls.values():
            if (
                control.status == ComplianceStatus.NON_COMPLIANT
                and control.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
            ):
                items.append(
                    {
                        "type": "non_compliant_control",
                        "control_id": control.control_id,
                        "title": control.title,
                        "risk_level": control.risk_level.value,
                        "criteria": control.criteria.value,
                    }
                )

        # Overdue assessments
        for control in self.controls.values():
            if control.next_assessment and control.next_assessment < datetime.utcnow():
                items.append(
                    {
                        "type": "overdue_assessment",
                        "control_id": control.control_id,
                        "title": control.title,
                        "due_date": control.next_assessment.isoformat(),
                    }
                )

        # Open high-severity incidents
        for incident in self.incidents.values():
            if incident.status == "open" and incident.severity in ["high", "critical"]:
                items.append(
                    {
                        "type": "open_incident",
                        "incident_id": incident.incident_id,
                        "title": incident.title,
                        "severity": incident.severity,
                        "age_days": (datetime.utcnow() - incident.timestamp).days,
                    }
                )

        return sorted(items, key=lambda x: x.get("risk_level", "medium"), reverse=True)[
            :10
        ]


# Global SOC 2 compliance manager instance
_global_compliance_manager: Optional[SOC2ComplianceManager] = None


def get_compliance_manager() -> SOC2ComplianceManager:
    """Get global SOC 2 compliance manager instance"""
    global _global_compliance_manager

    if _global_compliance_manager is None:
        _global_compliance_manager = SOC2ComplianceManager()

    return _global_compliance_manager


# Convenience functions for common compliance operations
async def log_compliance_event(
    event_type: AuditEventType,
    user_id: str,
    action: str,
    resource: str = "",
    criteria_affected: List[SOC2Criteria] = None,
    **kwargs,
) -> str:
    """Convenience function for logging compliance events"""
    manager = get_compliance_manager()
    return await manager.audit_logger.log_audit_event(
        event_type=event_type,
        user_id=user_id,
        action=action,
        resource=resource,
        criteria_affected=criteria_affected or [],
        **kwargs,
    )


async def record_incident(
    title: str, description: str, incident_type: str, severity: str = "medium", **kwargs
) -> str:
    """Convenience function for recording security incidents"""
    manager = get_compliance_manager()
    return await manager.record_security_incident(
        title=title,
        description=description,
        incident_type=incident_type,
        severity=severity,
        **kwargs,
    )


async def generate_compliance_report(
    criteria: Optional[SOC2Criteria] = None, days_back: int = 90
) -> Dict[str, Any]:
    """Convenience function for generating compliance reports"""
    manager = get_compliance_manager()

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)

    return await manager.generate_soc2_report(
        criteria=criteria, start_date=start_date, end_date=end_date
    )
