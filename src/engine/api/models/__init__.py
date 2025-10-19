"""
API Models and Data Contracts
Request/response models and validation for all API endpoints
"""

from .request_models import (
    EngagementCreateRequest,
    CognitiveAnalysisRequest,
    ComparisonRequest,
    ModelOverrideRequest,
    WhatIfRequest,
    WhatIfBatchRequest,
    ExportRequest,
)

from .response_models import (
    EngagementResponse,
    CognitiveAnalysisResponse,
    ModelListResponse,
    AuditTrailResponse,
    APIHealthResponse,
    ComparisonResponse,
    ModelOverrideResponse,
    WhatIfResponse,
    WhatIfBatchResponse,
)

from .validation import validate_problem_statement, validate_business_context

__all__ = [
    # Request models
    "EngagementCreateRequest",
    "CognitiveAnalysisRequest",
    "ComparisonRequest",
    "ModelOverrideRequest",
    "WhatIfRequest",
    "WhatIfBatchRequest",
    "ExportRequest",
    # Response models
    "EngagementResponse",
    "CognitiveAnalysisResponse",
    "ModelListResponse",
    "AuditTrailResponse",
    "APIHealthResponse",
    "ComparisonResponse",
    "ModelOverrideResponse",
    "WhatIfResponse",
    "WhatIfBatchResponse",
    # Validation
    "validate_problem_statement",
    "validate_business_context",
]
