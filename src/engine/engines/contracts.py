"""
METIS V5 Engine Service Contracts
=================================

Pydantic data contracts for communication between modular engine services.
Part of the Great Refactoring: Clean, type-safe interfaces between services.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum


class QueryComplexity(str, Enum):
    """Query complexity levels"""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


class QueryIntent(str, Enum):
    """Query intent classifications"""

    STRATEGIC_ANALYSIS = "strategic_analysis"
    PROBLEM_SOLVING = "problem_solving"
    DECISION_SUPPORT = "decision_support"
    RESEARCH_SYNTHESIS = "research_synthesis"
    CREATIVE_IDEATION = "creative_ideation"
    TECHNICAL_ANALYSIS = "technical_analysis"
    GENERAL_INQUIRY = "general_inquiry"


class ConsultantRole(str, Enum):
    """Available consultant roles"""

    STRATEGIC_CONSULTANT = "strategic_consultant"
    MANAGEMENT_CONSULTANT = "management_consultant"
    TECHNICAL_CONSULTANT = "technical_consultant"
    RESEARCH_ANALYST = "research_analyst"
    CREATIVE_DIRECTOR = "creative_director"
    RISK_ANALYST = "risk_analyst"
    OPERATIONS_SPECIALIST = "operations_specialist"


# === ENGAGEMENT LIFECYCLE CONTRACTS ===


class EngagementRequest(BaseModel):
    """Initial engagement request from API layer"""

    query: str = Field(..., description="The user's query or problem statement")
    context: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional context or metadata"
    )
    engagement_id: str = Field(..., description="Unique engagement identifier")
    user_preferences: Optional[Dict[str, Any]] = Field(
        default=None, description="User-specific preferences"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "How should we approach market expansion in Asia?",
                "context": {"industry": "technology", "company_size": "enterprise"},
                "engagement_id": "eng_123456",
                "user_preferences": {"style": "detailed", "urgency": "high"},
            }
        }


class QueryClassificationResult(BaseModel):
    """Result of query analysis and classification"""

    intent: QueryIntent = Field(..., description="Classified intent of the query")
    complexity: QueryComplexity = Field(..., description="Assessed complexity level")
    domain_tags: List[str] = Field(
        default_factory=list, description="Identified domain/industry tags"
    )
    key_entities: List[str] = Field(
        default_factory=list, description="Extracted key entities"
    )
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Classification confidence"
    )
    processing_hints: Dict[str, Any] = Field(
        default_factory=dict, description="Processing optimization hints"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "intent": "strategic_analysis",
                "complexity": "complex",
                "domain_tags": ["business_strategy", "international_expansion"],
                "key_entities": ["market expansion", "Asia", "strategy"],
                "confidence_score": 0.92,
                "processing_hints": {"requires_research": True, "consultant_count": 3},
            }
        }


class ConsultantCandidate(BaseModel):
    """Individual consultant candidate"""

    consultant_id: str = Field(..., description="Unique consultant identifier")
    role: ConsultantRole = Field(..., description="Primary consultant role")
    name: str = Field(..., description="Consultant display name")
    expertise_domains: List[str] = Field(
        default_factory=list, description="Areas of expertise"
    )
    match_score: float = Field(
        ..., ge=0.0, le=1.0, description="Match score for this query"
    )
    reasoning: str = Field(..., description="Why this consultant was selected")

    class Config:
        json_schema_extra = {
            "example": {
                "consultant_id": "strategic_012",
                "role": "strategic_consultant",
                "name": "Strategic Analysis Expert",
                "expertise_domains": ["market_expansion", "competitive_analysis"],
                "match_score": 0.95,
                "reasoning": "Specializes in international market expansion strategies",
            }
        }


class ConsultantSelectionResult(BaseModel):
    """Result of consultant selection process"""

    selected_consultants: List[ConsultantCandidate] = Field(
        ..., description="Selected consultants"
    )
    nway_clusters: List[str] = Field(
        default_factory=list, description="Selected N-Way methodology clusters"
    )
    selection_strategy: str = Field(..., description="Strategy used for selection")
    total_candidates_evaluated: int = Field(
        ..., description="Total candidates considered"
    )
    selection_reasoning: str = Field(..., description="Overall selection rationale")

    class Config:
        json_schema_extra = {
            "example": {
                "selected_consultants": [],  # Array of ConsultantCandidate objects
                "nway_clusters": ["strategic_planning", "market_analysis"],
                "selection_strategy": "intent_based_matching",
                "total_candidates_evaluated": 15,
                "selection_reasoning": "Selected experts in strategic analysis and international markets",
            }
        }


# === STATE MANAGEMENT CONTRACTS ===


class EngagementState(str, Enum):
    """Engagement processing states"""

    INITIALIZED = "initialized"
    CLASSIFYING = "classifying"
    SELECTING_CONSULTANTS = "selecting_consultants"
    PROCESSING = "processing"
    CRITIQUING = "critiquing"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"
    RECOVERED = "recovered"


class StateCheckpoint(BaseModel):
    """State management checkpoint"""

    engagement_id: str = Field(..., description="Engagement identifier")
    current_state: EngagementState = Field(..., description="Current processing state")
    checkpoint_data: Dict[str, Any] = Field(
        default_factory=dict, description="State-specific data"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Checkpoint timestamp"
    )
    can_recover: bool = Field(
        default=True, description="Whether recovery is possible from this state"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "engagement_id": "eng_123456",
                "current_state": "processing",
                "checkpoint_data": {
                    "selected_consultants": ["strategic_012", "analyst_007"],
                    "progress_percent": 60,
                },
                "can_recover": True,
            }
        }


# === INTEGRATION CONTRACTS ===


class DatabaseQuery(BaseModel):
    """Database operation request"""

    operation: str = Field(..., description="Database operation type")
    table: str = Field(..., description="Target table")
    conditions: Optional[Dict[str, Any]] = Field(
        default=None, description="Query conditions"
    )
    data: Optional[Dict[str, Any]] = Field(
        default=None, description="Data for insert/update operations"
    )


class CacheOperation(BaseModel):
    """Cache operation request"""

    operation: str = Field(..., description="Cache operation: get, set, delete, exists")
    key: str = Field(..., description="Cache key")
    value: Optional[Any] = Field(default=None, description="Value for set operations")
    ttl: Optional[int] = Field(default=None, description="Time to live in seconds")


# === MONITORING CONTRACTS ===


class PerformanceMetrics(BaseModel):
    """Performance monitoring metrics"""

    engagement_id: str = Field(..., description="Engagement identifier")
    operation: str = Field(..., description="Operation being measured")
    duration_ms: float = Field(..., description="Operation duration in milliseconds")
    memory_usage_mb: Optional[float] = Field(
        default=None, description="Memory usage in MB"
    )
    success: bool = Field(..., description="Whether operation succeeded")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metrics"
    )


class HealthStatus(BaseModel):
    """System health check result"""

    component: str = Field(..., description="Component name")
    healthy: bool = Field(..., description="Health status")
    response_time_ms: Optional[float] = Field(default=None, description="Response time")
    last_check: datetime = Field(
        default_factory=datetime.utcnow, description="Last health check"
    )
    details: Optional[str] = Field(
        default=None, description="Health details or error message"
    )


# === FINAL OUTPUT CONTRACT ===


class OptimalEngagementResult(BaseModel):
    """Final engagement result - preserves compatibility with existing API"""

    engagement_id: str = Field(..., description="Engagement identifier")
    query: str = Field(..., description="Original query")
    selected_consultants: List[ConsultantCandidate] = Field(
        ..., description="Selected consultants"
    )
    processing_summary: Dict[str, Any] = Field(
        default_factory=dict, description="Processing metadata"
    )
    performance_metrics: Dict[str, Any] = Field(
        default_factory=dict, description="Performance data"
    )
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Overall confidence"
    )

    # Preserve legacy fields for API compatibility
    consultant_selection_result: Optional[Dict[str, Any]] = Field(
        default=None, description="Legacy consultant selection data"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "engagement_id": "eng_123456",
                "query": "How should we approach market expansion in Asia?",
                "selected_consultants": [],  # Array of ConsultantCandidate objects
                "processing_summary": {
                    "total_duration_ms": 2341,
                    "consultants_evaluated": 15,
                    "nway_clusters_used": 2,
                },
                "confidence_score": 0.89,
            }
        }
