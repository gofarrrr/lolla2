"""
API Endpoints Module
Domain-driven API endpoint organization
"""

from .engagements import register_engagement_endpoints
from .models import register_model_endpoints
from .scenarios import register_scenario_endpoints
from .vulnerability import register_vulnerability_endpoints


def register_all_endpoints(app):
    """
    Register all endpoint modules with the FastAPI app
    """
    register_engagement_endpoints(app)
    register_model_endpoints(app)
    register_scenario_endpoints(app)
    register_vulnerability_endpoints(app)


__all__ = [
    "register_all_endpoints",
    "register_engagement_endpoints",
    "register_model_endpoints",
    "register_scenario_endpoints",
    "register_vulnerability_endpoints",
]
