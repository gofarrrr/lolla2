"""
Lolla Proving Ground - Registry Models
SQLAlchemy models for the challenger prompt system and proving ground infrastructure
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from uuid import uuid4

from sqlalchemy import Column, String, Text, DateTime, Enum as SQLEnum, JSON
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel, Field, validator

Base = declarative_base()


# Enums for type safety
class TargetStation(str, Enum):
    """Target stations for challenger prompts"""

    FULL_PIPELINE = "FULL_PIPELINE"
    STATION_1 = "STATION_1"  # QUICKTHINK
    STATION_2 = "STATION_2"  # DEEPTHINK
    STATION_3 = "STATION_3"  # BLUETHINK
    STATION_4 = "STATION_4"  # REDTHINK
    STATION_5 = "STATION_5"  # GREYTHINK
    STATION_6 = "STATION_6"  # ULTRATHINK
    STATION_7 = "STATION_7"  # DIVERGENTTHINK
    STATION_8 = "STATION_8"  # CONVERGENTTHINK


class ChallengerStatus(str, Enum):
    """Status options for challenger prompts"""

    ACTIVE = "active"
    ARCHIVED = "archived"
    DRAFT = "draft"


class DuelMethod(str, Enum):
    """Methods used in proving ground duels"""

    LOLLA_PIPELINE = "lolla_pipeline"
    LOLLA_STATION_1 = "lolla_station_1"
    LOLLA_STATION_2 = "lolla_station_2"
    LOLLA_STATION_3 = "lolla_station_3"
    LOLLA_STATION_4 = "lolla_station_4"
    LOLLA_STATION_5 = "lolla_station_5"
    LOLLA_STATION_6 = "lolla_station_6"
    LOLLA_STATION_7 = "lolla_station_7"
    LOLLA_STATION_8 = "lolla_station_8"
    CHALLENGER_MONOLITH = "challenger_monolith"
    CHALLENGER_STATION = "challenger_station"


# SQLAlchemy Models
class ChallengerPrompt(Base):
    """
    SQLAlchemy model for storing challenger prompts in the proving ground system
    """

    __tablename__ = "challenger_prompts"

    prompt_id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    prompt_name = Column(String(255), nullable=False)
    prompt_text = Column(Text, nullable=False)
    version = Column(String(50), nullable=False)
    status = Column(SQLEnum(ChallengerStatus), default=ChallengerStatus.ACTIVE)
    target_station = Column(SQLEnum(TargetStation), nullable=False)
    golden_case_id = Column(String, nullable=True)  # References evaluation datasets
    compilation_metadata = Column(
        JSON, nullable=True
    )  # JSON metadata about compilation
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary representation"""
        return {
            "prompt_id": self.prompt_id,
            "prompt_name": self.prompt_name,
            "prompt_text": self.prompt_text,
            "version": self.version,
            "status": self.status.value if self.status else None,
            "target_station": (
                self.target_station.value if self.target_station else None
            ),
            "golden_case_id": self.golden_case_id,
            "compilation_metadata": self.compilation_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# Pydantic Models for API validation
class ChallengerPromptBase(BaseModel):
    """Base model for challenger prompt validation"""

    prompt_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Descriptive name for the challenger prompt",
    )
    prompt_text: str = Field(..., min_length=10, description="The complete prompt text")
    version: str = Field(
        ..., min_length=1, max_length=50, description="Version identifier"
    )
    target_station: TargetStation = Field(
        ..., description="Target station for comparison"
    )
    golden_case_id: Optional[str] = Field(None, description="Associated golden case ID")
    compilation_metadata: Optional[Dict[str, Any]] = Field(
        None, description="Metadata about compilation process"
    )

    @validator("prompt_text")
    def validate_prompt_text(cls, v):
        if len(v.strip()) < 10:
            raise ValueError("Prompt text must be at least 10 characters long")
        return v.strip()

    @validator("compilation_metadata")
    def validate_metadata(cls, v):
        if v is not None and not isinstance(v, dict):
            raise ValueError("Compilation metadata must be a valid JSON object")
        return v


class ChallengerPromptCreate(ChallengerPromptBase):
    """Model for creating new challenger prompts"""

    status: ChallengerStatus = Field(
        default=ChallengerStatus.DRAFT, description="Initial status"
    )


class ChallengerPromptUpdate(BaseModel):
    """Model for updating existing challenger prompts"""

    prompt_name: Optional[str] = Field(None, min_length=1, max_length=255)
    prompt_text: Optional[str] = Field(None, min_length=10)
    version: Optional[str] = Field(None, min_length=1, max_length=50)
    status: Optional[ChallengerStatus] = None
    target_station: Optional[TargetStation] = None
    golden_case_id: Optional[str] = None
    compilation_metadata: Optional[Dict[str, Any]] = None

    @validator("prompt_text")
    def validate_prompt_text(cls, v):
        if v is not None and len(v.strip()) < 10:
            raise ValueError("Prompt text must be at least 10 characters long")
        return v.strip() if v else v


class ChallengerPromptResponse(ChallengerPromptBase):
    """Model for API responses containing challenger prompts"""

    prompt_id: str
    status: ChallengerStatus
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DuelConfiguration(BaseModel):
    """Configuration for launching a proving ground duel"""

    golden_case_id: str = Field(..., description="Golden case to test against")
    challenger_prompt_id: str = Field(..., description="Challenger prompt to use")
    station_to_test: str = Field(
        default="FULL_PIPELINE", description="Station to test or FULL_PIPELINE"
    )

    @validator("station_to_test")
    def validate_station(cls, v):
        valid_stations = [station.value for station in TargetStation]
        if v not in valid_stations:
            raise ValueError(f"Station must be one of: {valid_stations}")
        return v


class DuelResult(BaseModel):
    """Result of a proving ground duel"""

    duel_id: str
    lolla_result: Dict[str, Any]
    challenger_result: Dict[str, Any]
    comparison: Dict[str, Any]
    execution_metadata: Optional[Dict[str, Any]] = None


class CompilationRequest(BaseModel):
    """Request to compile a monolithic challenger from a golden case"""

    golden_case_id: str = Field(..., description="Golden case to compile into monolith")
    prompt_name: Optional[str] = Field(
        None, description="Optional name for generated prompt"
    )
    version: Optional[str] = Field("1.0", description="Version identifier")


class ProvingGroundStats(BaseModel):
    """Statistics for the proving ground system"""

    total_challengers: int
    active_challengers: int
    total_duels: int
    lolla_wins: int
    challenger_wins: int
    ties: int
    avg_quality_delta: float
    latest_duel: Optional[datetime] = None


# Error response models
class ErrorResponse(BaseModel):
    """Standard error response model"""

    detail: str
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Filter models
class ChallengerPromptFilters(BaseModel):
    """Query filters for challenger prompts"""

    target_station: Optional[TargetStation] = None
    status: Optional[ChallengerStatus] = ChallengerStatus.ACTIVE
    golden_case_id: Optional[str] = None
    skip: int = Field(default=0, ge=0)
    limit: int = Field(default=100, ge=1, le=1000)


# Station metadata for UI
STATION_METADATA = {
    TargetStation.STATION_1: {
        "name": "QUICKTHINK",
        "description": "Initial rapid analysis",
    },
    TargetStation.STATION_2: {
        "name": "DEEPTHINK",
        "description": "Comprehensive exploration",
    },
    TargetStation.STATION_3: {
        "name": "BLUETHINK",
        "description": "Conservative analysis",
    },
    TargetStation.STATION_4: {"name": "REDTHINK", "description": "Bold innovation"},
    TargetStation.STATION_5: {"name": "GREYTHINK", "description": "Reality check"},
    TargetStation.STATION_6: {"name": "ULTRATHINK", "description": "Deep synthesis"},
    TargetStation.STATION_7: {
        "name": "DIVERGENTTHINK",
        "description": "Alternative perspectives",
    },
    TargetStation.STATION_8: {
        "name": "CONVERGENTTHINK",
        "description": "Final integration",
    },
    TargetStation.FULL_PIPELINE: {
        "name": "FULL PIPELINE",
        "description": "Complete 8-station analysis",
    },
}


def get_station_display_name(target_station: TargetStation) -> str:
    """Get user-friendly display name for a station"""
    return STATION_METADATA.get(target_station, {}).get("name", target_station.value)


def get_station_description(target_station: TargetStation) -> str:
    """Get description for a station"""
    return STATION_METADATA.get(target_station, {}).get("description", "")
