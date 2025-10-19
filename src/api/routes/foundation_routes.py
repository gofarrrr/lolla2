"""Lean router exposing foundation API endpoints."""

from src.api.routes.enhanced_foundation_lean import MetisEnhancedAPIFoundation

foundation_app = MetisEnhancedAPIFoundation()
router = foundation_app.app.router
