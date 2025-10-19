"""
Common Data Contracts - METIS V5 API Contract Registry
Shared data structures used across multiple components
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum


class ProcessingStatus(str, Enum):
    """Standard processing status codes"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ErrorSeverity(str, Enum):
    """Standard error severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class EngagementContext:
    """Standard engagement context passed between components"""

    engagement_id: str
    problem_statement: str
    business_context: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "engagement_id": self.engagement_id,
            "problem_statement": self.problem_statement,
            "business_context": self.business_context,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ProcessingMetrics:
    """Standard processing metrics captured by all components"""

    component_name: str
    processing_time_seconds: float
    start_time: datetime
    end_time: datetime
    status: ProcessingStatus
    tokens_consumed: Optional[int] = None
    api_calls_made: Optional[int] = None
    cache_hits: Optional[int] = None
    cache_misses: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "component_name": self.component_name,
            "processing_time_seconds": self.processing_time_seconds,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "status": self.status.value,
            "tokens_consumed": self.tokens_consumed,
            "api_calls_made": self.api_calls_made,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }


@dataclass
class ErrorResponse:
    """Standard error response structure"""

    error_code: str
    error_message: str
    severity: ErrorSeverity
    component: str
    engagement_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    stack_trace: Optional[str] = None
    recovery_suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "error_code": self.error_code,
            "error_message": self.error_message,
            "severity": self.severity.value,
            "component": self.component,
            "engagement_id": self.engagement_id,
            "timestamp": self.timestamp.isoformat(),
            "stack_trace": self.stack_trace,
            "recovery_suggestions": self.recovery_suggestions,
        }
