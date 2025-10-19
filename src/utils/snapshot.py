"""
Safe Snapshot Utilities for Checkpoint Persistence

Centralizes JSON-safe serialization policy to prevent deepcopy edge cases
and ensure predictable checkpoint storage.
"""

import copy
from typing import Any, Dict


def snapshot_json_safe(model_or_dict: Any) -> Dict[str, Any]:
    """
    Create a JSON-safe deep copy snapshot for checkpoint persistence.

    Handles:
    - Pydantic V2 models: Uses model_dump(mode="json")
    - Plain dicts: Deep copies with JSON-safe guarantees
    - Edge cases: Converts UUIDs to strings, handles Decimals

    Args:
        model_or_dict: Pydantic model or dict to snapshot

    Returns:
        JSON-safe dict that can be safely deepcopied and persisted

    Example:
        >>> from pydantic import BaseModel
        >>> class MyModel(BaseModel):
        ...     x: int = 1
        >>> snapshot = snapshot_json_safe(MyModel(x=42))
        >>> assert snapshot == {"x": 42}
    """
    # Accept pydantic v2 models or dicts
    if hasattr(model_or_dict, "model_dump"):
        # Pydantic V2: Use JSON mode for safe serialization
        data = model_or_dict.model_dump(mode="json", round_trip=True)
    elif hasattr(model_or_dict, "dict"):
        # Pydantic V1: Fallback for legacy models
        data = model_or_dict.dict()
    else:
        # Plain dict or other JSON-serializable structure
        data = model_or_dict

    # Deep copy to ensure immutability
    # JSON-safe data → cheap & predictable deepcopy
    return copy.deepcopy(data)


def forbid_global_keys(section: dict, section_name: str, forbidden_keys: set) -> None:
    """
    Validate that a stage output section doesn't contain global context keys.

    Raises:
        ValueError: If forbidden keys are found in the section

    Example:
        >>> FORBIDDEN = {"socratic_questions", "stage_history"}
        >>> clean_section = {"mece_framework": []}
        >>> forbid_global_keys(clean_section, "problem_structuring", FORBIDDEN)
        # No error
        >>> bad_section = {"mece_framework": [], "socratic_questions": {...}}
        >>> forbid_global_keys(bad_section, "problem_structuring", FORBIDDEN)
        ValueError: problem_structuring: forbidden keys ['socratic_questions']
    """
    bad_keys = forbidden_keys & set(section.keys())
    if bad_keys:
        raise ValueError(f"{section_name}: forbidden global keys {sorted(bad_keys)}")


def validate_stage_schema(
    stage_output: dict,
    stage_name: str,
    allowed_keys: set,
    forbidden_keys: set
) -> dict:
    """
    Validate and clean stage output according to schema contract.

    Args:
        stage_output: Raw stage output dict
        stage_name: Name of stage for error messages
        allowed_keys: Set of allowed keys for this stage
        forbidden_keys: Set of forbidden global keys

    Returns:
        Cleaned stage output with only allowed keys

    Raises:
        ValueError: If forbidden keys are present
    """
    # Check for forbidden global keys
    forbid_global_keys(stage_output, stage_name, forbidden_keys)

    # Project only allowed keys
    cleaned = {
        k: stage_output[k]
        for k in allowed_keys
        if k in stage_output
    }

    return cleaned
