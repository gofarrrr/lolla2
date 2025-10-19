"""
Exception classes for Gamma API integration
"""


class GammaAPIError(Exception):
    """Base exception for Gamma API errors"""

    def __init__(
        self, message: str, status_code: int = None, response_data: dict = None
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data or {}


class RateLimitError(GammaAPIError):
    """Rate limit exceeded"""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = None):
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class GenerationError(GammaAPIError):
    """Presentation generation failed"""

    def __init__(self, message: str, generation_id: str = None):
        super().__init__(message)
        self.generation_id = generation_id


class AuthenticationError(GammaAPIError):
    """API authentication failed"""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, status_code=401)


class ValidationError(GammaAPIError):
    """Request validation failed"""

    def __init__(self, message: str, field: str = None):
        super().__init__(message, status_code=400)
        self.field = field


class StorageError(GammaAPIError):
    """Presentation storage operation failed"""

    pass


class TemplateError(GammaAPIError):
    """Template processing failed"""

    pass
