"""
Cognitive Engine Components Package
Modular components extracted from cognitive_engine.py following SOLID principles
"""

from .model_manager import ModelManager, ModelApplicationStrategy
from .problem_analyzer import ProblemAnalyzer, get_problem_analyzer

# integration_orchestrator.py is missing - removing import to fix startup

__all__ = [
    "ModelManager",
    "ModelApplicationStrategy",
    "ProblemAnalyzer",
    "get_problem_analyzer",
    # 'IntegrationOrchestrator', # Missing file
    # 'ServiceRegistry', # Missing file
    # 'create_integration_orchestrator' # Missing file
]
