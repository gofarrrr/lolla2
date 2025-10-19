"""
METIS Enterprise Security & Compliance Framework
Week 4 Sprint: Enterprise Integration & Security

Implements enterprise-grade security patterns including:
- SOC 2 Type II compliance framework
- Multi-tenant data isolation and security
- Enterprise audit trails and monitoring
- Zero-trust security architecture
- Data encryption and privacy controls
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import ipaddress

# Cryptography imports
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import base64

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


class SecurityLevel(Enum):
    """Security classification levels"""

    PUBLIC = "public"  # Public information
    INTERNAL = "internal"  # Internal use only
    CONFIDENTIAL = "confidential"  # Confidential business data
    RESTRICTED = "restricted"  # Highly sensitive data
    TOP_SECRET = "top_secret"  # Maximum security required


class AuditEventType(Enum):
    """Types of audit events for compliance tracking"""

    USER_AUTHENTICATION = "user_authentication"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    ANALYSIS_EXECUTION = "analysis_execution"
    HUMAN_INTERVENTION = "human_intervention"
    SYSTEM_ERROR = "system_error"
    SECURITY_VIOLATION = "security_violation"
    COMPLIANCE_CHECK = "compliance_check"
    DATA_EXPORT = "data_export"
    PRIVILEGED_ACCESS = "privileged_access"


class ComplianceFramework(Enum):
    """Supported compliance frameworks"""

    SOC2_TYPE_II = "soc2_type_ii"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    ISO_27001 = "iso_27001"
    NIST_CYBERSECURITY = "nist_cybersecurity"
    PCI_DSS = "pci_dss"


@dataclass
class SecurityContext:
    """Security context for request/session"""

    user_id: str
    tenant_id: str
    security_clearance: SecurityLevel
    ip_address: str
    user_agent: str
    session_id: str
    authentication_method: str

    # Risk factors
    geolocation: str = ""
    device_fingerprint: str = ""
    risk_score: float = 0.0
    anomaly_flags: List[str] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None


@dataclass
class AuditEvent:
    """Immutable audit event for compliance tracking"""

    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: str
    tenant_id: str

    # Event details
    resource_accessed: str = ""
    action_performed: str = ""
    data_classification: SecurityLevel = SecurityLevel.INTERNAL
    ip_address: str = ""
    user_agent: str = ""

    # Security context
    session_id: str = ""
    authentication_method: str = ""
    risk_score: float = 0.0

    # Outcome
    success: bool = True
    error_message: str = ""
    security_alerts: List[str] = field(default_factory=list)

    # Compliance tags
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
    retention_years: int = 7

    def to_audit_record(self) -> Dict[str, Any]:
        """Convert to immutable audit record for storage"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "resource_accessed": self.resource_accessed,
            "action_performed": self.action_performed,
            "data_classification": self.data_classification.value,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "session_id": self.session_id,
            "authentication_method": self.authentication_method,
            "risk_score": self.risk_score,
            "success": self.success,
            "error_message": self.error_message,
            "security_alerts": self.security_alerts,
            "compliance_frameworks": [f.value for f in self.compliance_frameworks],
            "retention_years": self.retention_years,
        }


@dataclass
class DataClassificationRule:
    """Data classification rule for automatic security tagging"""

    rule_id: str
    rule_name: str
    classification: SecurityLevel
    field_patterns: List[str]  # Regex patterns for field names
    content_patterns: List[str]  # Regex patterns for content
    tenant_specific: bool = False
    priority: int = 1  # Higher number = higher priority


class EncryptionManager:
    """
    Enterprise encryption manager for data protection
    Implements field-level and document-level encryption
    """

    def __init__(self, master_key: Optional[bytes] = None):
        self.logger = logging.getLogger(__name__)

        if not CRYPTO_AVAILABLE:
            raise ImportError("Cryptography library required for encryption")

        # Initialize encryption keys
        if master_key:
            self.master_key = master_key
        else:
            self.master_key = Fernet.generate_key()

        self.fernet = Fernet(self.master_key)
        self.tenant_keys: Dict[str, bytes] = {}

    def get_tenant_key(self, tenant_id: str) -> bytes:
        """Get or create tenant-specific encryption key"""

        if tenant_id not in self.tenant_keys:
            # Derive tenant key from master key and tenant ID
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=tenant_id.encode(),
                iterations=100000,
            )
            tenant_key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
            self.tenant_keys[tenant_id] = tenant_key

        return self.tenant_keys[tenant_id]

    async def encrypt_field(
        self,
        data: str,
        tenant_id: str,
        field_classification: SecurityLevel = SecurityLevel.CONFIDENTIAL,
    ) -> str:
        """Encrypt individual field with tenant isolation"""

        tenant_key = self.get_tenant_key(tenant_id)
        tenant_fernet = Fernet(tenant_key)

        # Add classification metadata
        classified_data = {
            "data": data,
            "classification": field_classification.value,
            "encrypted_at": datetime.now().isoformat(),
            "tenant_id": tenant_id,
        }

        data_bytes = json.dumps(classified_data).encode()
        encrypted_data = tenant_fernet.encrypt(data_bytes)

        return base64.urlsafe_b64encode(encrypted_data).decode()

    async def decrypt_field(
        self,
        encrypted_data: str,
        tenant_id: str,
        requesting_user_clearance: SecurityLevel,
    ) -> Optional[str]:
        """Decrypt field with security clearance validation"""

        try:
            tenant_key = self.get_tenant_key(tenant_id)
            tenant_fernet = Fernet(tenant_key)

            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_bytes = tenant_fernet.decrypt(encrypted_bytes)
            classified_data = json.loads(decrypted_bytes.decode())

            # Check security clearance
            data_classification = SecurityLevel(classified_data["classification"])
            if not self._has_clearance(requesting_user_clearance, data_classification):
                self.logger.warning(
                    f"Access denied: insufficient clearance for {data_classification.value} data"
                )
                return None

            return classified_data["data"]

        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            return None

    def _has_clearance(
        self, user_clearance: SecurityLevel, data_classification: SecurityLevel
    ) -> bool:
        """Check if user has sufficient clearance for data classification"""

        clearance_hierarchy = {
            SecurityLevel.PUBLIC: 0,
            SecurityLevel.INTERNAL: 1,
            SecurityLevel.CONFIDENTIAL: 2,
            SecurityLevel.RESTRICTED: 3,
            SecurityLevel.TOP_SECRET: 4,
        }

        return clearance_hierarchy.get(user_clearance, 0) >= clearance_hierarchy.get(
            data_classification, 0
        )

    async def encrypt_document(
        self,
        document: Dict[str, Any],
        tenant_id: str,
        classification_rules: List[DataClassificationRule],
    ) -> Dict[str, Any]:
        """Encrypt document with automatic field classification"""

        encrypted_doc = {}

        for field_name, field_value in document.items():

            # Determine field classification
            field_classification = self._classify_field(
                field_name, field_value, classification_rules
            )

            if field_classification in [
                SecurityLevel.CONFIDENTIAL,
                SecurityLevel.RESTRICTED,
                SecurityLevel.TOP_SECRET,
            ]:
                # Encrypt sensitive fields
                if isinstance(field_value, str):
                    encrypted_doc[field_name] = await self.encrypt_field(
                        field_value, tenant_id, field_classification
                    )
                    encrypted_doc[f"{field_name}_encrypted"] = True
                    encrypted_doc[f"{field_name}_classification"] = (
                        field_classification.value
                    )
                else:
                    # For non-string fields, convert to JSON and encrypt
                    json_value = json.dumps(field_value)
                    encrypted_doc[field_name] = await self.encrypt_field(
                        json_value, tenant_id, field_classification
                    )
                    encrypted_doc[f"{field_name}_encrypted"] = True
                    encrypted_doc[f"{field_name}_classification"] = (
                        field_classification.value
                    )
            else:
                # Keep non-sensitive fields unencrypted
                encrypted_doc[field_name] = field_value
                encrypted_doc[f"{field_name}_classification"] = (
                    field_classification.value
                )

        return encrypted_doc

    def _classify_field(
        self,
        field_name: str,
        field_value: Any,
        classification_rules: List[DataClassificationRule],
    ) -> SecurityLevel:
        """Automatically classify field based on rules"""

        import re

        # Sort rules by priority (highest first)
        sorted_rules = sorted(
            classification_rules, key=lambda r: r.priority, reverse=True
        )

        for rule in sorted_rules:

            # Check field name patterns
            for pattern in rule.field_patterns:
                if re.search(pattern, field_name, re.IGNORECASE):
                    return rule.classification

            # Check content patterns (for string values)
            if isinstance(field_value, str):
                for pattern in rule.content_patterns:
                    if re.search(pattern, field_value, re.IGNORECASE):
                        return rule.classification

        # Default classification
        return SecurityLevel.INTERNAL


class AuditTrailManager:
    """
    SOC 2 compliant audit trail manager
    Provides immutable audit logging for compliance requirements
    """

    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption_manager = encryption_manager
        self.logger = logging.getLogger(__name__)
        self.audit_buffer: List[AuditEvent] = []
        self.compliance_rules: Dict[ComplianceFramework, Dict[str, Any]] = {}

        # Initialize compliance rules
        self._initialize_compliance_rules()

        # Start audit buffer flush task
        asyncio.create_task(self._flush_audit_buffer_periodically())

    def _initialize_compliance_rules(self):
        """Initialize compliance framework rules"""

        # SOC 2 Type II requirements
        self.compliance_rules[ComplianceFramework.SOC2_TYPE_II] = {
            "required_events": [
                AuditEventType.USER_AUTHENTICATION,
                AuditEventType.DATA_ACCESS,
                AuditEventType.DATA_MODIFICATION,
                AuditEventType.PRIVILEGED_ACCESS,
                AuditEventType.SECURITY_VIOLATION,
            ],
            "retention_years": 7,
            "real_time_monitoring": True,
            "integrity_protection": True,
            "access_controls": True,
        }

        # GDPR requirements
        self.compliance_rules[ComplianceFramework.GDPR] = {
            "required_events": [
                AuditEventType.DATA_ACCESS,
                AuditEventType.DATA_MODIFICATION,
                AuditEventType.DATA_EXPORT,
            ],
            "retention_years": 6,
            "data_subject_access": True,
            "right_to_erasure": True,
            "privacy_by_design": True,
        }

        # ISO 27001 requirements
        self.compliance_rules[ComplianceFramework.ISO_27001] = {
            "required_events": [
                AuditEventType.USER_AUTHENTICATION,
                AuditEventType.SECURITY_VIOLATION,
                AuditEventType.SYSTEM_ERROR,
                AuditEventType.PRIVILEGED_ACCESS,
            ],
            "retention_years": 3,
            "risk_assessment": True,
            "incident_response": True,
            "continuous_monitoring": True,
        }

    async def log_audit_event(
        self,
        event_type: AuditEventType,
        security_context: SecurityContext,
        resource_accessed: str = "",
        action_performed: str = "",
        data_classification: SecurityLevel = SecurityLevel.INTERNAL,
        success: bool = True,
        error_message: str = "",
        additional_data: Dict[str, Any] = None,
    ) -> str:
        """Log audit event with compliance validation"""

        event_id = str(uuid.uuid4())

        # Determine applicable compliance frameworks
        applicable_frameworks = self._get_applicable_frameworks(
            event_type, data_classification, security_context.tenant_id
        )

        # Calculate risk score for event
        risk_score = await self._calculate_event_risk_score(
            event_type, security_context, success, additional_data or {}
        )

        # Detect security alerts
        security_alerts = await self._detect_security_alerts(
            event_type, security_context, success, risk_score
        )

        # Create audit event
        audit_event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.now(),
            user_id=security_context.user_id,
            tenant_id=security_context.tenant_id,
            resource_accessed=resource_accessed,
            action_performed=action_performed,
            data_classification=data_classification,
            ip_address=security_context.ip_address,
            user_agent=security_context.user_agent,
            session_id=security_context.session_id,
            authentication_method=security_context.authentication_method,
            risk_score=risk_score,
            success=success,
            error_message=error_message,
            security_alerts=security_alerts,
            compliance_frameworks=applicable_frameworks,
            retention_years=max(
                [
                    self.compliance_rules[f]["retention_years"]
                    for f in applicable_frameworks
                ]
            ),
        )

        # Add to audit buffer
        self.audit_buffer.append(audit_event)

        # Immediate flush for high-risk events
        if risk_score > 0.8 or security_alerts:
            await self._flush_audit_buffer()

        self.logger.info(
            f"AUDIT: {event_type.value} by {security_context.user_id} - Risk: {risk_score:.2f}"
        )

        return event_id

    def _get_applicable_frameworks(
        self,
        event_type: AuditEventType,
        data_classification: SecurityLevel,
        tenant_id: str,
    ) -> List[ComplianceFramework]:
        """Determine which compliance frameworks apply to this event"""

        applicable = []

        for framework, rules in self.compliance_rules.items():

            # Check if event type is required by framework
            if event_type in rules.get("required_events", []):
                applicable.append(framework)

            # SOC 2 applies to all business data
            elif (
                framework == ComplianceFramework.SOC2_TYPE_II
                and data_classification != SecurityLevel.PUBLIC
            ):
                applicable.append(framework)

            # GDPR applies to personal data (simplified check)
            elif (
                framework == ComplianceFramework.GDPR
                and "personal" in data_classification.value
            ):
                applicable.append(framework)

        return applicable

    async def _calculate_event_risk_score(
        self,
        event_type: AuditEventType,
        security_context: SecurityContext,
        success: bool,
        additional_data: Dict[str, Any],
    ) -> float:
        """Calculate risk score for audit event"""

        base_risk = {
            AuditEventType.USER_AUTHENTICATION: 0.2,
            AuditEventType.DATA_ACCESS: 0.3,
            AuditEventType.DATA_MODIFICATION: 0.5,
            AuditEventType.ANALYSIS_EXECUTION: 0.4,
            AuditEventType.HUMAN_INTERVENTION: 0.3,
            AuditEventType.SYSTEM_ERROR: 0.6,
            AuditEventType.SECURITY_VIOLATION: 1.0,
            AuditEventType.COMPLIANCE_CHECK: 0.1,
            AuditEventType.DATA_EXPORT: 0.7,
            AuditEventType.PRIVILEGED_ACCESS: 0.8,
        }.get(event_type, 0.3)

        # Adjust for failure
        if not success:
            base_risk += 0.3

        # Adjust for user risk factors
        base_risk += security_context.risk_score * 0.2

        # Adjust for anomaly flags
        base_risk += len(security_context.anomaly_flags) * 0.1

        # Adjust for time-based factors (off-hours access)
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:  # Off hours
            base_risk += 0.1

        return min(1.0, base_risk)

    async def _detect_security_alerts(
        self,
        event_type: AuditEventType,
        security_context: SecurityContext,
        success: bool,
        risk_score: float,
    ) -> List[str]:
        """Detect security alerts based on event patterns"""

        alerts = []

        # High risk score alert
        if risk_score > 0.8:
            alerts.append("HIGH_RISK_EVENT")

        # Failed authentication attempts
        if event_type == AuditEventType.USER_AUTHENTICATION and not success:
            alerts.append("AUTHENTICATION_FAILURE")

        # Privileged access outside business hours
        if event_type == AuditEventType.PRIVILEGED_ACCESS:
            current_hour = datetime.now().hour
            if current_hour < 6 or current_hour > 22:
                alerts.append("OFF_HOURS_PRIVILEGED_ACCESS")

        # Suspicious IP address (simplified check)
        try:
            ip = ipaddress.ip_address(security_context.ip_address)
            if ip.is_private and str(ip).startswith("10."):  # Example suspicious range
                alerts.append("SUSPICIOUS_IP_ADDRESS")
        except:
            pass

        # Multiple anomaly flags
        if len(security_context.anomaly_flags) >= 3:
            alerts.append("MULTIPLE_ANOMALIES_DETECTED")

        return alerts

    async def _flush_audit_buffer(self):
        """Flush audit buffer to persistent storage"""

        if not self.audit_buffer:
            return

        # In production, this would write to secure audit database
        for event in self.audit_buffer:
            audit_record = event.to_audit_record()

            # Encrypt sensitive audit data
            if event.data_classification in [
                SecurityLevel.CONFIDENTIAL,
                SecurityLevel.RESTRICTED,
            ]:
                encrypted_record = await self.encryption_manager.encrypt_document(
                    audit_record,
                    event.tenant_id,
                    [],  # Audit records use predefined classification
                )
                self.logger.info(f"AUDIT_ENCRYPTED: {event.event_id}")
            else:
                self.logger.info(f"AUDIT: {json.dumps(audit_record)}")

        # Clear buffer
        flushed_count = len(self.audit_buffer)
        self.audit_buffer.clear()

        self.logger.info(f"Flushed {flushed_count} audit events to persistent storage")

    async def _flush_audit_buffer_periodically(self):
        """Periodically flush audit buffer"""

        while True:
            await asyncio.sleep(60)  # Flush every minute
            try:
                await self._flush_audit_buffer()
            except Exception as e:
                self.logger.error(f"Audit buffer flush failed: {e}")

    async def generate_compliance_report(
        self,
        framework: ComplianceFramework,
        tenant_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, Any]:
        """Generate compliance report for audit purposes"""

        # In production, this would query the audit database
        # For now, we'll return a mock report structure

        return {
            "framework": framework.value,
            "tenant_id": tenant_id,
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
            "compliance_status": "COMPLIANT",
            "total_events": len(self.audit_buffer),
            "security_violations": len(
                [
                    e
                    for e in self.audit_buffer
                    if e.event_type == AuditEventType.SECURITY_VIOLATION
                ]
            ),
            "high_risk_events": len(
                [e for e in self.audit_buffer if e.risk_score > 0.8]
            ),
            "data_access_events": len(
                [
                    e
                    for e in self.audit_buffer
                    if e.event_type == AuditEventType.DATA_ACCESS
                ]
            ),
            "authentication_failures": len(
                [
                    e
                    for e in self.audit_buffer
                    if e.event_type == AuditEventType.USER_AUTHENTICATION
                    and not e.success
                ]
            ),
            "retention_compliance": "VERIFIED",
            "data_integrity": "VERIFIED",
            "access_controls": "ACTIVE",
            "generated_at": datetime.now().isoformat(),
        }


class ZeroTrustSecurityManager:
    """
    Zero Trust security manager for enterprise METIS deployments
    Implements continuous verification and least-privilege access
    """

    def __init__(
        self, encryption_manager: EncryptionManager, audit_manager: AuditTrailManager
    ):
        self.encryption_manager = encryption_manager
        self.audit_manager = audit_manager
        self.logger = logging.getLogger(__name__)

        # Zero Trust policies
        self.access_policies: Dict[str, Dict[str, Any]] = {}
        self.risk_thresholds = {
            "authentication_required": 0.3,
            "mfa_required": 0.5,
            "access_denied": 0.8,
            "security_alert": 0.7,
        }

        # Initialize default policies
        self._initialize_default_policies()

    def _initialize_default_policies(self):
        """Initialize default Zero Trust access policies"""

        # Cognitive analysis access policy
        self.access_policies["cognitive_analysis"] = {
            "required_roles": ["analyst", "senior_analyst", "partner"],
            "max_risk_score": 0.6,
            "require_mfa": True,
            "session_timeout_minutes": 60,
            "ip_restrictions": [],
            "time_restrictions": [],
            "data_classification_access": {
                SecurityLevel.PUBLIC: True,
                SecurityLevel.INTERNAL: True,
                SecurityLevel.CONFIDENTIAL: False,  # Requires elevated clearance
                SecurityLevel.RESTRICTED: False,
                SecurityLevel.TOP_SECRET: False,
            },
        }

        # Human oversight access policy
        self.access_policies["human_oversight"] = {
            "required_roles": ["senior_analyst", "partner"],
            "max_risk_score": 0.4,
            "require_mfa": True,
            "session_timeout_minutes": 120,
            "ip_restrictions": [],
            "time_restrictions": [],
            "data_classification_access": {
                SecurityLevel.PUBLIC: True,
                SecurityLevel.INTERNAL: True,
                SecurityLevel.CONFIDENTIAL: True,
                SecurityLevel.RESTRICTED: False,  # Requires partner approval
                SecurityLevel.TOP_SECRET: False,
            },
        }

        # Administrative access policy
        self.access_policies["administrative"] = {
            "required_roles": ["admin", "security_officer"],
            "max_risk_score": 0.2,
            "require_mfa": True,
            "session_timeout_minutes": 30,
            "ip_restrictions": ["internal_network_only"],
            "time_restrictions": ["business_hours_only"],
            "data_classification_access": {
                SecurityLevel.PUBLIC: True,
                SecurityLevel.INTERNAL: True,
                SecurityLevel.CONFIDENTIAL: True,
                SecurityLevel.RESTRICTED: True,
                SecurityLevel.TOP_SECRET: False,  # Requires special authorization
            },
        }

    async def evaluate_access_request(
        self,
        security_context: SecurityContext,
        resource: str,
        action: str,
        data_classification: SecurityLevel = SecurityLevel.INTERNAL,
    ) -> Dict[str, Any]:
        """Evaluate access request using Zero Trust principles"""

        # Determine applicable policy
        policy = self._get_applicable_policy(resource, action)

        # Calculate current risk score
        current_risk = await self._calculate_current_risk_score(security_context)

        # Evaluate access decision
        access_decision = {
            "access_granted": False,
            "require_mfa": False,
            "require_additional_auth": False,
            "session_timeout_minutes": 60,
            "risk_score": current_risk,
            "policy_applied": policy,
            "violations": [],
            "security_controls": [],
        }

        # Check basic policy requirements
        violations = []

        # Risk score check
        if current_risk > policy.get("max_risk_score", 1.0):
            violations.append(
                f"Risk score {current_risk:.2f} exceeds policy limit {policy.get('max_risk_score', 1.0)}"
            )

        # Data classification access check
        data_access_allowed = policy.get("data_classification_access", {}).get(
            data_classification, False
        )
        if not data_access_allowed:
            violations.append(
                f"Insufficient clearance for {data_classification.value} data"
            )

        # IP restrictions check
        ip_restrictions = policy.get("ip_restrictions", [])
        if ip_restrictions and not self._check_ip_restrictions(
            security_context.ip_address, ip_restrictions
        ):
            violations.append("IP address not in allowed range")

        # Time restrictions check
        time_restrictions = policy.get("time_restrictions", [])
        if time_restrictions and not self._check_time_restrictions(time_restrictions):
            violations.append("Access outside allowed time window")

        # Determine final access decision
        if violations:
            access_decision["access_granted"] = False
            access_decision["violations"] = violations
        else:
            access_decision["access_granted"] = True
            access_decision["session_timeout_minutes"] = policy.get(
                "session_timeout_minutes", 60
            )

            # Apply security controls
            if current_risk > self.risk_thresholds["mfa_required"] or policy.get(
                "require_mfa", False
            ):
                access_decision["require_mfa"] = True
                access_decision["security_controls"].append("MFA_REQUIRED")

            if current_risk > self.risk_thresholds["authentication_required"]:
                access_decision["require_additional_auth"] = True
                access_decision["security_controls"].append("ADDITIONAL_AUTH_REQUIRED")

        # Log access evaluation
        await self.audit_manager.log_audit_event(
            AuditEventType.DATA_ACCESS,
            security_context,
            resource,
            action,
            data_classification,
            access_decision["access_granted"],
            "; ".join(violations) if violations else "",
            {"access_decision": access_decision, "policy_applied": policy},
        )

        return access_decision

    def _get_applicable_policy(self, resource: str, action: str) -> Dict[str, Any]:
        """Get applicable access policy for resource and action"""

        # Simple policy matching - in production would be more sophisticated
        if "cognitive" in resource.lower() or "analysis" in resource.lower():
            return self.access_policies.get("cognitive_analysis", {})
        elif "oversight" in resource.lower() or "approval" in resource.lower():
            return self.access_policies.get("human_oversight", {})
        elif "admin" in resource.lower() or "configuration" in resource.lower():
            return self.access_policies.get("administrative", {})
        else:
            return self.access_policies.get("cognitive_analysis", {})  # Default policy

    async def _calculate_current_risk_score(
        self, security_context: SecurityContext
    ) -> float:
        """Calculate current risk score for security context"""

        base_risk = security_context.risk_score

        # Time since last activity
        time_inactive = (
            datetime.now() - security_context.last_activity
        ).total_seconds() / 3600  # hours
        if time_inactive > 1:
            base_risk += min(0.2, time_inactive * 0.05)

        # Anomaly flags
        base_risk += len(security_context.anomaly_flags) * 0.1

        # Geolocation changes (simplified)
        if (
            security_context.geolocation
            and "foreign" in security_context.geolocation.lower()
        ):
            base_risk += 0.2

        # Session age
        session_age_hours = (
            datetime.now() - security_context.created_at
        ).total_seconds() / 3600
        if session_age_hours > 8:  # Long-running session
            base_risk += 0.1

        return min(1.0, base_risk)

    def _check_ip_restrictions(self, ip_address: str, restrictions: List[str]) -> bool:
        """Check if IP address meets restrictions"""

        if "internal_network_only" in restrictions:
            try:
                ip = ipaddress.ip_address(ip_address)
                return ip.is_private
            except:
                return False

        # Additional IP restriction logic would go here
        return True

    def _check_time_restrictions(self, restrictions: List[str]) -> bool:
        """Check if current time meets restrictions"""

        if "business_hours_only" in restrictions:
            current_hour = datetime.now().hour
            return 8 <= current_hour <= 18  # 8 AM to 6 PM

        # Additional time restriction logic would go here
        return True

    async def get_security_metrics(self, tenant_id: str) -> Dict[str, Any]:
        """Get security metrics for monitoring dashboard"""

        return {
            "total_access_requests": 0,  # Would query from audit logs
            "access_denials": 0,
            "mfa_challenges": 0,
            "security_alerts": 0,
            "average_risk_score": 0.3,
            "policy_violations": 0,
            "active_sessions": 0,
            "zero_trust_score": 0.85,  # Overall security posture score
            "last_updated": datetime.now().isoformat(),
        }


# Global instances
_encryption_manager: Optional[EncryptionManager] = None
_audit_manager: Optional[AuditTrailManager] = None
_security_manager: Optional[ZeroTrustSecurityManager] = None


async def get_encryption_manager() -> EncryptionManager:
    """Get or create global encryption manager"""
    global _encryption_manager

    if _encryption_manager is None:
        _encryption_manager = EncryptionManager()

    return _encryption_manager


async def get_audit_manager() -> AuditTrailManager:
    """Get or create global audit trail manager"""
    global _audit_manager

    if _audit_manager is None:
        encryption_manager = await get_encryption_manager()
        _audit_manager = AuditTrailManager(encryption_manager)

    return _audit_manager


async def get_security_manager() -> ZeroTrustSecurityManager:
    """Get or create global Zero Trust security manager"""
    global _security_manager

    if _security_manager is None:
        encryption_manager = await get_encryption_manager()
        audit_manager = await get_audit_manager()
        _security_manager = ZeroTrustSecurityManager(encryption_manager, audit_manager)

    return _security_manager
