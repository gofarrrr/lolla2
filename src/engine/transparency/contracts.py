"""
METIS V5 Transparency Engine Refactoring - Target #3
Transparency Service Contracts

Following the successful pattern from engines/contracts.py
Type-safe communication between transparency services

Pydantic data contracts for all transparency service communication
"""

from typing import Dict, List, Optional, Any
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field

from src.models.transparency_models import (
    UserExpertiseLevel,
    CognitiveLoadLevel,
    TransparencyLayer,
)


class TransparencyRequest(BaseModel):
    """Request for transparency generation"""

    engagement_id: str
    user_id: UUID
    session_id: Optional[str] = None
    content: str
    reasoning_steps: List[Dict[str, Any]] = []
    mental_models: List[Dict[str, Any]] = []
    user_preferences: Dict[str, Any] = {}


class CognitiveLoadAssessment(BaseModel):
    """Result of cognitive load assessment"""

    load_level: CognitiveLoadLevel
    complexity_score: int
    content_length: int
    reasoning_step_count: int
    mental_model_count: int
    assessment_timestamp: datetime = Field(default_factory=datetime.now)


class ScaffoldingStrategy(BaseModel):
    """Cognitive scaffolding strategy result"""

    chunking_strategy: str
    progressive_hints: List[str]
    contextual_assistance: List[Dict[str, Any]]
    navigation_aids: List[Dict[str, Any]]
    complexity_reduction: List[str]
    expertise_adaptations: Dict[str, Any]


class ExpertiseAssessmentResult(BaseModel):
    """Result of user expertise assessment"""

    assessed_level: UserExpertiseLevel
    confidence_score: float
    interaction_patterns: Dict[str, Any]
    recommendation: str
    assessment_basis: str  # "history" or "cold_start" or "cognitive_profile"


class TransparencyLayerContent(BaseModel):
    """Content for a specific transparency layer"""

    layer_type: TransparencyLayer
    content: str
    metadata: Dict[str, Any]
    complexity_score: float
    estimated_reading_time: int  # seconds
    generated_at: datetime = Field(default_factory=datetime.now)


class ProgressiveDisclosureResult(BaseModel):
    """Complete progressive disclosure package"""

    user_id: UUID
    session_id: Optional[str]
    expertise_level: UserExpertiseLevel
    cognitive_load: CognitiveLoadLevel
    recommended_starting_layer: TransparencyLayer
    layers: Dict[TransparencyLayer, TransparencyLayerContent]
    scaffolding: ScaffoldingStrategy
    performance_metrics: Dict[str, int]
    generated_at: datetime = Field(default_factory=datetime.now)


class TransparencyHealthStatus(BaseModel):
    """Health status for transparency services"""

    service_name: str
    healthy: bool
    response_time_ms: Optional[float]
    details: str
    capabilities: Dict[str, Any] = {}
    last_check: datetime = Field(default_factory=datetime.now)


class TransparencyPerformanceMetrics(BaseModel):
    """Performance metrics for transparency operations"""

    operation_name: str
    duration_ms: int
    cognitive_load_assessment_ms: Optional[int] = None
    expertise_assessment_ms: Optional[int] = None
    scaffolding_generation_ms: Optional[int] = None
    layer_generation_ms: Dict[str, int] = {}
    timestamp: datetime = Field(default_factory=datetime.now)


# Service interfaces (similar to engines/contracts.py pattern)


class CognitiveScaffoldingServiceInterface(BaseModel):
    """Interface contract for CognitiveScaffoldingService"""

    service_name: str = "CognitiveScaffoldingService"
    version: str = "v5_modular"

    class Config:
        arbitrary_types_allowed = True


class UserExpertiseServiceInterface(BaseModel):
    """Interface contract for UserExpertiseService"""

    service_name: str = "UserExpertiseService"
    version: str = "v5_modular"

    class Config:
        arbitrary_types_allowed = True


class TransparencyOrchestratorInterface(BaseModel):
    """Interface contract for TransparencyOrchestrator"""

    orchestrator_name: str = "TransparencyOrchestrator"
    version: str = "v5_modular"
    services_count: int

    class Config:
        arbitrary_types_allowed = True


# Error types for transparency operations


class TransparencyServiceError(Exception):
    """Base exception for transparency service errors"""

    def __init__(
        self, message: str, service_name: str, error_code: Optional[str] = None
    ):
        self.message = message
        self.service_name = service_name
        self.error_code = error_code
        super().__init__(self.message)


class CognitiveLoadAssessmentError(TransparencyServiceError):
    """Error during cognitive load assessment"""

    def __init__(self, message: str, complexity_score: Optional[int] = None):
        super().__init__(message, "CognitiveScaffoldingService", "COGNITIVE_LOAD_ERROR")
        self.complexity_score = complexity_score


class ExpertiseAssessmentError(TransparencyServiceError):
    """Error during expertise assessment"""

    def __init__(self, message: str, user_id: Optional[UUID] = None):
        super().__init__(message, "UserExpertiseService", "EXPERTISE_ASSESSMENT_ERROR")
        self.user_id = user_id


class ProgressiveDisclosureError(TransparencyServiceError):
    """Error during progressive disclosure generation"""

    def __init__(self, message: str, layer_type: Optional[TransparencyLayer] = None):
        super().__init__(
            message, "TransparencyOrchestrator", "PROGRESSIVE_DISCLOSURE_ERROR"
        )
        self.layer_type = layer_type
