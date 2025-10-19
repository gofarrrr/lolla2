"""
METIS Cognitive Platform - Structured Exception Hierarchy
Operation Bedrock - The Hardening: Phase 1 Implementation

This module defines a comprehensive exception hierarchy to eliminate silent failures
and enable proper error propagation throughout the cognitive processing pipeline.

Key Principle: Fail-fast with clear error boundaries and recovery context.
"""

from typing import Optional, Dict, Any
from enum import Enum


class ErrorSeverity(Enum):
    """Classification of error severity levels"""

    CRITICAL = "critical"  # System-breaking, halt all processing
    HIGH = "high"  # Component failure, may be recoverable
    MEDIUM = "medium"  # Degraded functionality, continue with warnings
    LOW = "low"  # Minor issues, logging only


class MetisBaseException(Exception):
    """
    Base exception class for all METIS cognitive platform errors.

    Provides structured error context and recovery guidance.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "METIS_UNKNOWN",
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        engagement_id: Optional[str] = None,
        component: Optional[str] = None,
        recovery_suggestions: Optional[list] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.engagement_id = engagement_id or "unknown"
        self.component = component or "unknown"
        self.recovery_suggestions = recovery_suggestions or []
        self.context = context or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to structured dictionary for logging/API responses"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "severity": self.severity.value,
            "engagement_id": self.engagement_id,
            "component": self.component,
            "recovery_suggestions": self.recovery_suggestions,
            "context": self.context,
        }


class ComponentFailureError(MetisBaseException):
    """
    Critical component failure requiring immediate halt of processing.

    Used when a core cognitive engine component cannot function.
    """

    def __init__(self, component: str, message: str, **kwargs):
        super().__init__(
            message=f"Component '{component}' failed: {message}",
            error_code="COMPONENT_FAILURE",
            severity=ErrorSeverity.CRITICAL,
            component=component,
            **kwargs,
        )


class LLMProviderError(MetisBaseException):
    """
    LLM provider communication or processing failures.

    Covers API errors, timeout, rate limiting, content policy violations.
    """

    def __init__(
        self,
        provider: str,
        message: str,
        api_error_code: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            message=f"LLM Provider '{provider}' error: {message}",
            error_code=f"LLM_{provider.upper()}_ERROR",
            severity=ErrorSeverity.HIGH,
            component=f"llm_{provider}",
            context={"api_error_code": api_error_code},
            **kwargs,
        )
        self.provider = provider
        self.api_error_code = api_error_code


class ResearchProviderError(MetisBaseException):
    """
    External research provider failures (Perplexity, Firecrawl, etc.).

    These are often recoverable by falling back to other providers.
    """

    def __init__(self, provider: str, message: str, **kwargs):
        super().__init__(
            message=f"Research Provider '{provider}' error: {message}",
            error_code=f"RESEARCH_{provider.upper()}_ERROR",
            severity=ErrorSeverity.MEDIUM,
            component=f"research_{provider}",
            recovery_suggestions=[
                "Retry with different research provider",
                "Continue analysis without external research",
                "Use cached research data if available",
            ],
            **kwargs,
        )
        self.provider = provider


class ValidationError(MetisBaseException):
    """
    Data validation and contract violations.

    Used when Pydantic models fail validation or data contracts are violated.
    """

    def __init__(self, field: str, message: str, **kwargs):
        super().__init__(
            message=f"Validation error in '{field}': {message}",
            error_code="VALIDATION_ERROR",
            severity=ErrorSeverity.HIGH,
            component="data_validation",
            recovery_suggestions=[
                "Check input data format and required fields",
                "Verify API request matches expected schema",
                "Review data transformation logic",
            ],
            **kwargs,
        )
        self.field = field


class AuthenticationError(MetisBaseException):
    """
    Authentication and authorization failures.

    Covers missing API keys, expired tokens, permission denied.
    """

    def __init__(self, service: str, message: str, **kwargs):
        super().__init__(
            message=f"Authentication error for '{service}': {message}",
            error_code="AUTH_ERROR",
            severity=ErrorSeverity.CRITICAL,
            component=f"auth_{service}",
            recovery_suggestions=[
                "Check environment variables for API keys",
                "Verify API key permissions and quotas",
                "Review authentication configuration",
            ],
            **kwargs,
        )
        self.service = service


class CacheError(MetisBaseException):
    """
    Caching system failures (Redis, L1/L2/L3 cache issues).

    Usually non-critical, system can continue without cache.
    """

    def __init__(self, cache_type: str, message: str, **kwargs):
        super().__init__(
            message=f"Cache '{cache_type}' error: {message}",
            error_code="CACHE_ERROR",
            severity=ErrorSeverity.LOW,
            component=f"cache_{cache_type}",
            recovery_suggestions=[
                "Continue processing without cache",
                "Check cache service connectivity",
                "Clear cache and retry operation",
            ],
            **kwargs,
        )
        self.cache_type = cache_type


class ConfigurationError(MetisBaseException):
    """
    System configuration and environment setup errors.

    Critical errors that prevent system startup or initialization.
    """

    def __init__(self, config_key: str, message: str, **kwargs):
        super().__init__(
            message=f"Configuration error '{config_key}': {message}",
            error_code="CONFIG_ERROR",
            severity=ErrorSeverity.CRITICAL,
            component="system_config",
            recovery_suggestions=[
                "Check .env file and environment variables",
                "Review configuration schema",
                "Verify all required settings are provided",
            ],
            **kwargs,
        )
        self.config_key = config_key


class TimeoutError(MetisBaseException):
    """
    Operation timeout errors for LLM calls, research, etc.

    May be recoverable by retrying with different parameters.
    """

    def __init__(self, operation: str, timeout_seconds: int, **kwargs):
        super().__init__(
            message=f"Operation '{operation}' timed out after {timeout_seconds} seconds",
            error_code="TIMEOUT_ERROR",
            severity=ErrorSeverity.MEDIUM,
            component="timeout_manager",
            recovery_suggestions=[
                "Retry with increased timeout",
                "Break operation into smaller chunks",
                "Use alternative processing approach",
            ],
            **kwargs,
        )
        self.operation = operation
        self.timeout_seconds = timeout_seconds
