"""
API Response Models - Extracted from foundation.py
Response data contracts for all API endpoints
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel


class EngagementResponse(BaseModel):
    """Response model for engagement operations"""

    engagement_id: str
    status: str
    created_at: str
    problem_statement: str
    business_context: Dict[str, Any]
    cognitive_state: Dict[str, Any]
    workflow_state: Dict[str, Any]


class CognitiveAnalysisResponse(BaseModel):
    """Response model for cognitive analysis"""

    engagement_id: str
    analysis_id: str
    cognitive_state: Dict[str, Any]
    reasoning_steps: List[Dict[str, Any]]
    confidence_scores: Dict[str, float]
    processing_time_ms: float


class ModelListResponse(BaseModel):
    """Response model for available mental models"""

    models: List[Dict[str, Any]]
    total_count: int
    categories: List[str]


class AuditTrailResponse(BaseModel):
    """Response model for audit trail data"""

    engagement_id: str
    audit_events: List[Dict[str, Any]]
    total_events: int
    time_range: Dict[str, str]


class APIHealthResponse(BaseModel):
    """Response model for API health check"""

    status: str
    timestamp: str
    version: str
    dependencies: Dict[str, str]
    uptime_seconds: float


class ComparisonResponse(BaseModel):
    """Response model for engagement comparison"""

    comparison_id: str
    engagement_ids: List[str]
    comparison_results: Dict[str, Any]
    similarity_scores: Dict[str, float]
    key_differences: List[str]
    created_at: str


class ModelOverrideResponse(BaseModel):
    """Response model for model override operations"""

    engagement_id: str
    override_id: str
    applied_models: List[str]
    override_timestamp: str
    status: str
    results: Optional[Dict[str, Any]] = None


class WhatIfResponse(BaseModel):
    """Response model for what-if scenario creation"""

    scenario_id: str
    base_engagement_id: str
    scenario_description: str
    scenario_results: Dict[str, Any]
    comparison_with_base: Optional[Dict[str, Any]] = None
    created_at: str


class WhatIfBatchResponse(BaseModel):
    """Response model for batch what-if scenario processing"""

    batch_id: str
    base_engagement_id: str
    scenarios: List[WhatIfResponse]
    batch_summary: Dict[str, Any]
    processing_time_ms: float
    created_at: str


class ExportResponse(BaseModel):
    """Response model for engagement export"""

    export_id: str
    engagement_id: str
    export_format: str
    file_url: str
    file_size_bytes: int
    created_at: str
    expires_at: str


class VulnerabilityStatusResponse(BaseModel):
    """Response model for vulnerability system status"""

    status: str
    active_explorations: int
    completed_assessments: int
    system_health: Dict[str, Any]
    timestamp: str


class VulnerabilityContextResponse(BaseModel):
    """Response model for vulnerability context"""

    engagement_id: str
    vulnerability_context: Dict[str, Any]
    risk_assessment: Dict[str, float]
    mitigation_strategies: List[str]
    last_updated: str


class VulnerabilityExplorationResponse(BaseModel):
    """Response model for vulnerability exploration override"""

    engagement_id: str
    override_id: str
    exploration_strategy: str
    status: str
    results: Optional[Dict[str, Any]] = None
    created_at: str


class FeedbackTierResponse(BaseModel):
    """Response model for feedback tier assignment"""

    engagement_id: str
    assignment_id: str
    feedback_tier: str
    status: str
    assigned_at: str
    reviewer_id: Optional[str] = None


class HallucinationDetectionResponse(BaseModel):
    """Response model for hallucination detection"""

    engagement_id: str
    detection_id: str
    hallucination_indicators: List[Dict[str, Any]]
    confidence_scores: Dict[str, float]
    risk_level: str
    recommendations: List[str]
    analyzed_at: str


class AuthLoginResponse(BaseModel):
    """Response model for authentication login"""

    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_info: Dict[str, Any]
    permissions: List[str]
