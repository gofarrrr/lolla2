"""
Gamma API Service Module
Handles all interactions with Gamma's Generation API

This module provides integration with Gamma AI for generating professional
presentations, documents, and social media content from METIS cognitive analysis outputs.
"""

from .client import GammaAPIClient
from .builder import PresentationBuilder
from .templates import TemplateEngine, PresentationType
from .config import GammaConfig
from .service import GammaPresentationService
from .storage import PresentationStorage
from .exceptions import (
    GammaAPIError,
    RateLimitError,
    GenerationError,
    AuthenticationError,
)

__all__ = [
    "GammaAPIClient",
    "PresentationBuilder",
    "TemplateEngine",
    "PresentationType",
    "GammaConfig",
    "GammaPresentationService",
    "PresentationStorage",
    "GammaAPIError",
    "RateLimitError",
    "GenerationError",
    "AuthenticationError",
]

__version__ = "1.0.0"
__author__ = "METIS Cognitive Platform"
