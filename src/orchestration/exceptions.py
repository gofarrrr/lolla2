"""
Honest Orchestrator Exception Classes
=====================================

Custom exceptions for each orchestration stage.
Principle: "Fail Loudly, Succeed Honestly"

Each orchestrator must raise these specific exceptions rather than returning mock data.
"""


class OrchestrationError(Exception):
    """Base exception for all orchestration errors"""

    pass


class SocraticEngineError(OrchestrationError):
    """Raised when Socratic inquiry fails"""

    pass


class PSAError(OrchestrationError):
    """Raised when Problem Structuring Agent fails"""

    pass


class DispatchError(OrchestrationError):
    """Raised when consultant dispatch/selection fails"""

    pass


class ForgeError(OrchestrationError):
    """Raised when parallel forge execution fails"""

    pass


class SeniorAdvisorError(OrchestrationError):
    """Raised when Senior Advisor two-brain process fails"""

    pass


class SymphonyError(OrchestrationError):
    """Raised when the entire symphony execution fails"""

    pass
