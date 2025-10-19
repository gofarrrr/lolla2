"""
METIS Engagement API Package

Refactored from engagement_orchestrator.py for better separation of concerns.
Provides backward compatibility through re-exports.
"""

# Core components
from .orchestrator import EngagementOrchestrator
from .websocket import ConnectionManager
from .clarification import ClarificationHandler
from .sandbox import WhatIfSandbox

# API models
from .models import (
    EngagementPhase,
    EngagementStatus,
    ProblemStatement,
    EngagementRequest,
    PhaseResult,
    EngagementResponse,
    DeliverableRequest,
    ReevaluationRequest,
    ClarificationRequest,
    ClarificationQuestionResponse,
    ClarificationResponseRequest,
    ClarificationSkipRequest,
    ClarificationResult,
    EnhancedQueryResult,
)

# Mapping utilities
from .mappers import (
    map_contract_to_engagement_response,
    map_contract_phase_to_phase_result,
)

# FastAPI app
from .routes import app

# For backward compatibility - re-export everything that was in the original file
__all__ = [
    # Core classes
    "EngagementOrchestrator",
    "ConnectionManager",
    "ClarificationHandler",
    "WhatIfSandbox",
    # Models
    "EngagementPhase",
    "EngagementStatus",
    "ProblemStatement",
    "EngagementRequest",
    "PhaseResult",
    "EngagementResponse",
    "DeliverableRequest",
    "ReevaluationRequest",
    "ClarificationRequest",
    "ClarificationQuestionResponse",
    "ClarificationResponseRequest",
    "ClarificationSkipRequest",
    "ClarificationResult",
    "EnhancedQueryResult",
    # Functions
    "map_contract_to_engagement_response",
    "map_contract_phase_to_phase_result",
    # FastAPI app
    "app",
]
