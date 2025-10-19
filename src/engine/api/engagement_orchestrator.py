#!/usr/bin/env python3
"""
DEPRECATED: Use src.api.engagement package instead
This file maintained for backward compatibility
"""

# Re-export everything from the refactored modules
from .engagement import *
from .engagement.routes import app

# Re-export for backward compatibility
__all__ = [
    "EngagementOrchestrator",
    "ConnectionManager",
    "ClarificationHandler",
    "WhatIfSandbox",
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
    "map_contract_to_engagement_response",
    "map_contract_phase_to_phase_result",
    "app",
]
