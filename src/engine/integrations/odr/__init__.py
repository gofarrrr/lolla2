"""
Open Deep Research (ODR) Integration Package
Alternative research backend using LangChain's Open Deep Research framework
"""

from .config import ODRConfiguration
from .agent import ODRResearchAgent
from .autonomous import AutonomousODRClient
from .client import ODRClient
from .context import ContextGapDetector

# Check ODR availability
try:
    from langchain.agents import AgentExecutor

    ODR_AVAILABLE = True
except ImportError:
    ODR_AVAILABLE = False

__all__ = [
    "ODRConfiguration",
    "ODRResearchAgent",
    "AutonomousODRClient",
    "ODRClient",
    "ContextGapDetector",
    "ODR_AVAILABLE",
]
