"""Audit trail adapter - bridges src.core.audit_trail to src.engine"""

from src.core.audit_trail import (
    get_audit_manager,
    AuditEventType,
    AuditSeverity,
    MetisAuditTrailManager,
)

__all__ = [
    "get_audit_manager",
    "AuditEventType",
    "AuditSeverity",
    "MetisAuditTrailManager",
]
