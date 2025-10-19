# src/config/models.py
from __future__ import annotations

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, field_validator


# --- Enhancer (NWAY_RESEARCH_QUERY_ENHANCER_001.yaml) ---
class ThinVariablesModel(BaseModel):
    framing_invariance_testing: bool = True
    outcome_blindness_enforcement: bool = True
    retrieval_diversification: bool = True
    minority_signal_seeking: float = 0.6  # treat as float threshold


class NWayEnhancerConfig(BaseModel):
    version: Optional[str] = None
    enhancer_name: Optional[str] = None
    thin_variables: ThinVariablesModel = Field(default_factory=ThinVariablesModel)
    question_lenses: Dict[str, Any] = Field(default_factory=dict)
    conversation_personas: Dict[str, Any] = Field(default_factory=dict)
    # Allow additional fields without breaking
    extra_fields: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"


# --- Cognitive Architecture Master ---
class MasterClusterRef(BaseModel):
    file: str
    description: str
    cognitive_role: str


class CognitiveArchitectureMaster(BaseModel):
    version: Optional[str] = None
    architecture_type: Optional[str] = None
    total_models: Optional[int] = None
    total_nway_files: Optional[int] = None
    cognitive_paradigm: Optional[str] = None
    clusters: Optional[Dict[str, MasterClusterRef]] = None
    cognitive_mastery: Optional[Dict[str, Any]] = None
    consultant_personas: Optional[Dict[str, Any]] = None
    system_configuration: Optional[Dict[str, Any]] = None
    performance: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    documentation: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"


# --- Cluster YAML models (behavioral_cluster.yaml etc.) ---
class CognitiveMasteryModel(BaseModel):
    failure_detection_patterns: Union[List[str], Dict[str, str], List[Dict[str, str]]] = Field(default_factory=dict)
    intervention_protocols: Union[List[str], Dict[str, str], List[Dict[str, str]]] = Field(default_factory=dict)
    consultant_behavioral_scripts: Union[List[str], Dict[str, str], List[Dict[str, str]]] = Field(default_factory=dict)
    debiasing_activation_triggers: Union[List[str], Dict[str, str], List[Dict[str, str]]] = Field(default_factory=dict)
    operational_frameworks: Union[List[str], Dict[str, str], List[Dict[str, str]]] = Field(default_factory=dict)
    success_validation_metrics: Union[List[str], Dict[str, str], List[Dict[str, str]]] = Field(default_factory=dict)
    bias_interaction_detection: Optional[Union[List[str], Dict[str, str], List[Dict[str, str]]]] = Field(default_factory=dict)

    @field_validator('success_validation_metrics', 'failure_detection_patterns', 'intervention_protocols',
                     'consultant_behavioral_scripts', 'debiasing_activation_triggers', 'operational_frameworks',
                     'bias_interaction_detection', mode='before')
    @classmethod
    def normalize_metrics(cls, v):
        """Normalize all format variations to Dict[str, str] for consistent downstream usage.

        Supports three YAML formats:
        1. Dict[str, str]: key: description (most common)
        2. List[str]: simple list of metric names
        3. List[Dict[str, str]]: list of single-key dicts (financial_cluster.yaml format)
        """
        if v is None:
            return {}

        # Already a dict - pass through
        if isinstance(v, dict):
            return v

        # List of dicts - flatten to single dict
        if isinstance(v, list) and v and isinstance(v[0], dict):
            result = {}
            for item in v:
                result.update(item)
            return result

        # List of strings - convert to dict with empty descriptions
        if isinstance(v, list):
            return {str(item): "" for item in v}

        # Fallback - return as-is and let Pydantic handle
        return v

    class Config:
        extra = "allow"


class NWayDefinitionModel(BaseModel):
    id: str
    title: Optional[str] = None
    models: List[str] = Field(default_factory=list)
    interactions: Dict[str, str] = Field(default_factory=dict)
    consultant_priority: Dict[str, float] = Field(default_factory=dict)
    consultant_personas: Dict[str, Any] = Field(default_factory=dict)
    system2_triggers: Dict[str, str] = Field(default_factory=dict)
    metacognitive_prompts: Dict[str, str] = Field(default_factory=dict)
    cognitive_mastery: Optional[CognitiveMasteryModel] = None


class ClusterDataModel(BaseModel):
    name: str
    description: str
    cognitive_role: str
    files: Optional[int] = None
    total_models: Optional[int] = None
    nways: List[NWayDefinitionModel] = Field(default_factory=list)


# --- CoreOps Program YAML ---
class CoreOpModel(BaseModel):
    id: str
    op: str
    args: Dict[str, Any] = Field(default_factory=dict)
    budgets: Dict[str, Any] = Field(default_factory=dict)
    on_error: Dict[str, Any] = Field(default_factory=dict)


class CoreProgramModel(BaseModel):
    version: int = 1
    name: str
    steps: List[CoreOpModel]
