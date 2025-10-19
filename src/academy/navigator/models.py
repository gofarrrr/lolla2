"""Navigator data models and shared structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class NavigatorState(Enum):
    """Navigator conversation states - 11-step flow."""

    INITIAL = "initial"
    CLARIFYING = "clarifying"
    CONTEXT_GATHERING = "context_gathering"
    MODEL_DISCOVERY = "model_discovery"
    MODEL_SELECTION = "model_selection"
    MODEL_EXPLANATION = "model_explanation"
    APPLICATION_DESIGN = "application_design"
    IMPLEMENTATION_GUIDANCE = "implementation_guidance"
    VALIDATION_FRAMEWORK = "validation_framework"
    NEXT_STEPS = "next_steps"
    COMPLETED = "completed"


@dataclass
class StructuredContent:
    """Base class for structured content responses."""

    content_type: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelExplanation(StructuredContent):
    """Structured model explanation payload."""

    title: str = ""
    core_concept: str = ""
    application: str = ""
    examples: List[str] = field(default_factory=list)
    pitfalls: List[str] = field(default_factory=list)
    key_questions: List[str] = field(default_factory=list)
    content_type: str = "model_explanation"


@dataclass
class MentalModelMap(StructuredContent):
    """Mental Model Map structured output."""

    core_models: List[str] = field(default_factory=list)
    relationships: List[Dict[str, str]] = field(default_factory=list)
    synergies: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    application_sequence: List[str] = field(default_factory=list)
    unified_strategy: str = ""
    content_type: str = "mental_model_map"


@dataclass
class ReflectionPrompts(StructuredContent):
    """Reflection prompts structured output."""

    prompts: List[str] = field(default_factory=list)
    model_specific_questions: List[Dict[str, str]] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    content_type: str = "reflection_prompts"


@dataclass
class NavigatorSession:
    """Session state for Navigator conversations."""

    session_id: str
    state: NavigatorState = NavigatorState.INITIAL
    current_step: int = 1
    user_goal: str = ""
    domain_context: str = ""
    selected_models: List[Dict[str, Any]] = field(default_factory=list)
    model_explanations: List[ModelExplanation] = field(default_factory=list)
    mental_model_map: Optional[MentalModelMap] = None
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
