"""
API Request Models - Extracted from foundation.py
Request data contracts and validation for all API endpoints
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator


class EngagementCreateRequest(BaseModel):
    """Request model for creating new engagement"""

    problem_statement: str = Field(..., min_length=10, max_length=5000)
    business_context: Dict[str, Any] = Field(default_factory=dict)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    compliance_requirements: Dict[str, Any] = Field(default_factory=dict)

    @validator("problem_statement")
    def validate_problem_statement(cls, v):
        if not v.strip():
            raise ValueError("Problem statement cannot be empty")
        return v.strip()


class CognitiveAnalysisRequest(BaseModel):
    """Request model for cognitive analysis"""

    engagement_id: str
    force_model_selection: Optional[List[str]] = None
    analysis_preferences: Dict[str, Any] = Field(default_factory=dict)


class ComparisonRequest(BaseModel):
    """Request model for engagement comparison"""

    engagement_ids: List[str] = Field(..., min_items=2, max_items=5)
    comparison_criteria: Dict[str, Any] = Field(default_factory=dict)
    include_detailed_analysis: bool = False


class ModelOverrideRequest(BaseModel):
    """Request model for model override operations"""

    models_to_use: List[str] = Field(..., min_items=1, max_items=5)
    override_reason: str = Field(..., min_length=10)
    preserve_original: bool = True


class WhatIfRequest(BaseModel):
    """Request model for what-if scenario creation"""

    base_engagement_id: str
    scenario_changes: Dict[str, Any] = Field(..., min_items=1)
    scenario_description: str = Field(..., min_length=10, max_length=500)
    include_comparison: bool = True


class WhatIfBatchRequest(BaseModel):
    """Request model for batch what-if scenario processing"""

    base_engagement_id: str
    scenarios: List[Dict[str, Any]] = Field(..., min_items=1, max_items=10)
    batch_description: str = Field(default="Batch scenario analysis")
    parallel_processing: bool = True


class ExportRequest(BaseModel):
    """Request model for engagement export"""

    engagement_id: str
    export_format: str = Field(..., pattern=r"^(json|pdf|csv|xlsx)$")
    include_audit_trail: bool = False
    include_raw_data: bool = False
    include_visualizations: bool = True


class VulnerabilityExplorationRequest(BaseModel):
    """Request model for vulnerability exploration override"""

    engagement_id: str
    exploration_strategy: str = Field(..., min_length=5)
    override_reason: str = Field(..., min_length=10)
    preserve_existing: bool = True


class FeedbackTierRequest(BaseModel):
    """Request model for feedback tier assignment"""

    engagement_id: str
    feedback_tier: str = Field(..., pattern=r"^(tier_1|tier_2|tier_3)$")
    feedback_notes: str = Field(default="")
    reviewer_id: Optional[str] = None


class AuthLoginRequest(BaseModel):
    """Request model for authentication login"""

    username: str = Field(..., min_length=3)
    password: str = Field(..., min_length=6)
    remember_me: bool = False
