"""
METIS V5 API Contract Registry
The single source of truth for all inter-component communication contracts

This module enforces "THE API CONTRACT ACCORDS" - a systematic approach to
preventing API contract failures that have plagued the system.

All components MUST import their interfaces from this registry.
"""

from .socratic_contracts import SocraticRequest, SocraticResponse, QuestionSet
from .analysis_contracts import AnalysisRequest, AnalysisResponse, ConsultantOutput
from .consultant_contracts import (
    ConsultantSelectionRequest,
    ConsultantSelectionResponse,
)
from .common_contracts import EngagementContext, ProcessingMetrics, ErrorResponse

# FEATURE 2/6: Output Contracts (Enterprise Robustness)
from .output_contracts import (
    # Core validation
    validate_against_contract,
    ContractValidationResult,
    ContractViolation,
    ContractViolationType,
    # Contracts
    AnalysisOutput,
    StructuredQueryResponse,
    ClassificationOutput,
    RefusalResponse,
    # Registry
    CONTRACT_REGISTRY,
    register_contract,
    get_contract,
    # Decorator
    enforce_contract,
    # System prompts
    generate_contract_system_prompt,
    get_contract_prompt,
    # Global management
    enable_contract_validation,
    disable_contract_validation,
    is_contract_validation_enabled,
)

__all__ = [
    # Socratic Engine Contracts
    "SocraticRequest",
    "SocraticResponse",
    "QuestionSet",
    # Analysis Engine Contracts
    "AnalysisRequest",
    "AnalysisResponse",
    "ConsultantOutput",
    # Consultant Selection Contracts
    "ConsultantSelectionRequest",
    "ConsultantSelectionResponse",
    # Common Contracts
    "EngagementContext",
    "ProcessingMetrics",
    "ErrorResponse",
    # Output Contracts (Feature 2/6)
    "validate_against_contract",
    "ContractValidationResult",
    "ContractViolation",
    "ContractViolationType",
    "AnalysisOutput",
    "StructuredQueryResponse",
    "ClassificationOutput",
    "RefusalResponse",
    "CONTRACT_REGISTRY",
    "register_contract",
    "get_contract",
    "enforce_contract",
    "generate_contract_system_prompt",
    "get_contract_prompt",
    "enable_contract_validation",
    "disable_contract_validation",
    "is_contract_validation_enabled",
]
