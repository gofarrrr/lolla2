"""
API package exports.

Provides shared middleware and router utilities for the Lean stack.
"""

from .middleware import DeprecationHeaderMiddleware

__all__ = ["DeprecationHeaderMiddleware"]
