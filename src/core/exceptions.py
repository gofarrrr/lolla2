"""
Core Exceptions Compatibility Layer
Provides backward compatibility imports for src.core.exceptions
"""

# Import all exception classes from engine and re-export
from src.engine.core.exceptions import (
    ErrorSeverity,
    MetisBaseException,
    ComponentFailureError,
    LLMProviderError,
    ResearchProviderError,
    ValidationError,
    AuthenticationError,
    CacheError,
    ConfigurationError,
    TimeoutError,
)


# Iteration Engine Exceptions
class CheckpointError(MetisBaseException):
    """Raised when checkpoint operations fail"""

    pass


class RevisionError(MetisBaseException):
    """Raised when revision operations fail"""

    pass


class PipelineError(MetisBaseException):
    """Raised when pipeline execution fails"""

    pass


class IterationEngineError(MetisBaseException):
    """Base exception for all iteration engine errors"""

    pass


class StateReconstructionError(MetisBaseException):
    """Raised when state reconstruction from checkpoints fails"""

    pass


class BranchingError(MetisBaseException):
    """Raised when analysis branching operations fail"""

    pass


class AnalysisError(MetisBaseException):
    """Raised when analysis processing fails"""

    pass


# Legacy compatibility exports
__all__ = [
    "ErrorSeverity",
    "MetisBaseException",
    "ComponentFailureError",
    "LLMProviderError",
    "ResearchProviderError",
    "ValidationError",
    "AuthenticationError",
    "CacheError",
    "ConfigurationError",
    "TimeoutError",
    # Iteration Engine exceptions
    "CheckpointError",
    "RevisionError",
    "PipelineError",
    "IterationEngineError",
    "StateReconstructionError",
    "BranchingError",
    "AnalysisError",
]
