"""
API Validation Utilities - Extracted from foundation.py
Custom validators and validation logic for API models
"""

import re
from typing import Dict, Any, List
from pydantic import validator


def validate_problem_statement(problem_statement: str) -> str:
    """
    Validate problem statement content and format
    """
    if not problem_statement or not problem_statement.strip():
        raise ValueError("Problem statement cannot be empty")

    cleaned = problem_statement.strip()

    # Length validation
    if len(cleaned) < 10:
        raise ValueError("Problem statement must be at least 10 characters long")

    if len(cleaned) > 2000:
        raise ValueError("Problem statement cannot exceed 2000 characters")

    # Content validation
    if len(cleaned.split()) < 3:
        raise ValueError("Problem statement must contain at least 3 words")

    # Check for potentially harmful content patterns
    suspicious_patterns = [
        r"<script[^>]*>.*?</script>",  # Script tags
        r"javascript:",  # JavaScript URLs
        r"data:text/html",  # Data URLs
        r"on\w+\s*=",  # Event handlers
    ]

    for pattern in suspicious_patterns:
        if re.search(pattern, cleaned, re.IGNORECASE):
            raise ValueError("Problem statement contains potentially unsafe content")

    return cleaned


def validate_business_context(business_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate business context structure and content
    """
    if not isinstance(business_context, dict):
        raise ValueError("Business context must be a dictionary")

    # Size limitations
    if len(str(business_context)) > 10000:  # Rough size check
        raise ValueError("Business context is too large (max ~10KB)")

    # Validate known fields if present
    valid_industries = {
        "technology",
        "healthcare",
        "finance",
        "manufacturing",
        "retail",
        "education",
        "government",
        "consulting",
        "logistics",
        "energy",
        "media",
        "real_estate",
        "automotive",
        "aerospace",
        "pharmaceutical",
    }

    if "industry" in business_context:
        industry = business_context["industry"].lower()
        if industry not in valid_industries and industry != "other":
            raise ValueError(
                f"Invalid industry. Must be one of: {', '.join(valid_industries)}, or 'other'"
            )

    # Validate company size if provided
    valid_company_sizes = {"startup", "small", "medium", "large", "enterprise"}
    if "company_size" in business_context:
        size = business_context["company_size"].lower()
        if size not in valid_company_sizes:
            raise ValueError(
                f"Invalid company size. Must be one of: {', '.join(valid_company_sizes)}"
            )

    # Validate urgency level if provided
    valid_urgency_levels = {"low", "medium", "high", "critical"}
    if "urgency" in business_context:
        urgency = business_context["urgency"].lower()
        if urgency not in valid_urgency_levels:
            raise ValueError(
                f"Invalid urgency level. Must be one of: {', '.join(valid_urgency_levels)}"
            )

    return business_context


def validate_engagement_id(engagement_id: str) -> str:
    """
    Validate engagement ID format
    """
    if not engagement_id or not engagement_id.strip():
        raise ValueError("Engagement ID cannot be empty")

    # UUID format validation (loose)
    uuid_pattern = (
        r"^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$"
    )
    if not re.match(uuid_pattern, engagement_id.strip()):
        raise ValueError("Invalid engagement ID format (must be UUID)")

    return engagement_id.strip()


def validate_model_ids(model_ids: List[str]) -> List[str]:
    """
    Validate list of model IDs
    """
    if not model_ids or not isinstance(model_ids, list):
        raise ValueError("Model IDs must be a non-empty list")

    if len(model_ids) > 10:
        raise ValueError("Cannot specify more than 10 models")

    valid_model_pattern = r"^[a-zA-Z][a-zA-Z0-9_]*$"
    validated_models = []

    for model_id in model_ids:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("Each model ID must be a non-empty string")

        cleaned_id = model_id.strip()

        if not re.match(valid_model_pattern, cleaned_id):
            raise ValueError(f"Invalid model ID format: {cleaned_id}")

        if cleaned_id in validated_models:
            raise ValueError(f"Duplicate model ID: {cleaned_id}")

        validated_models.append(cleaned_id)

    return validated_models


def validate_export_format(export_format: str) -> str:
    """
    Validate export format
    """
    if not export_format or not export_format.strip():
        raise ValueError("Export format cannot be empty")

    valid_formats = {"json", "pdf", "csv", "xlsx", "html"}
    format_lower = export_format.strip().lower()

    if format_lower not in valid_formats:
        raise ValueError(
            f"Invalid export format. Must be one of: {', '.join(valid_formats)}"
        )

    return format_lower


def validate_scenario_changes(scenario_changes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate what-if scenario changes
    """
    if not isinstance(scenario_changes, dict) or not scenario_changes:
        raise ValueError("Scenario changes must be a non-empty dictionary")

    # Size limit
    if len(str(scenario_changes)) > 5000:
        raise ValueError("Scenario changes are too large (max ~5KB)")

    # Validate change types
    valid_change_types = {
        "problem_statement_modification",
        "business_context_update",
        "constraint_addition",
        "constraint_removal",
        "assumption_change",
        "parameter_adjustment",
    }

    for key in scenario_changes.keys():
        if not isinstance(key, str):
            raise ValueError("All scenario change keys must be strings")

        # Check if it's a valid change type or starts with a valid prefix
        if not any(key.startswith(change_type) for change_type in valid_change_types):
            raise ValueError(f"Invalid scenario change type: {key}")

    return scenario_changes


# Pydantic validator decorators for use in models


def problem_statement_validator():
    """Pydantic validator for problem statements"""
    return validator("problem_statement", allow_reuse=True)(validate_problem_statement)


def business_context_validator():
    """Pydantic validator for business context"""
    return validator("business_context", allow_reuse=True)(validate_business_context)


def engagement_id_validator():
    """Pydantic validator for engagement IDs"""
    return validator("engagement_id", allow_reuse=True)(validate_engagement_id)


def model_ids_validator():
    """Pydantic validator for model ID lists"""
    return validator("force_model_selection", "models_to_use", allow_reuse=True)(
        validate_model_ids
    )


def export_format_validator():
    """Pydantic validator for export formats"""
    return validator("export_format", allow_reuse=True)(validate_export_format)


def scenario_changes_validator():
    """Pydantic validator for scenario changes"""
    return validator("scenario_changes", allow_reuse=True)(validate_scenario_changes)
