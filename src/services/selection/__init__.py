"""
METIS Selection Services Cluster
Complete modular decomposition of model_selector.py god-file

Phase 5.4 Implementation - 7 Focused Services:
1. SelectionStrategyService - Execute different model selection strategies
2. ScoringEngineService - Comprehensive model scoring with component breakdown
3. NWayPatternService - N-Way interaction pattern management and synergy detection
4. BayesianLearningService - Context-specific model effectiveness learning
5. ZeroShotSelectionService - MeMo-based zero-shot selection for novel scenarios
6. SelectionCoordinatorService - Orchestrate all selection services
7. EliteConsultingIntegrationService - Integrate McKinsey frameworks with V5.4 pipeline

Each service has single responsibility and clean contracts.
V5.4 Enhancement: Elite consulting frameworks (TOSCA, Pyramid Principle, O2D) integrated.
"""

from .selection_strategy_service import (
    get_selection_strategy_service,
    SelectionStrategyService,
)
from .scoring_engine_service import get_scoring_engine_service, ScoringEngineService
from .nway_pattern_service import get_nway_pattern_service, NWayPatternService
from .bayesian_learning_service import (
    get_bayesian_learning_service,
    BayesianLearningService,
)
from .zero_shot_selection_service import (
    get_zero_shot_selection_service,
    ZeroShotSelectionService,
)
from .selection_coordinator_service import (
    get_selection_coordinator_service,
    SelectionCoordinatorService,
)
from .elite_consulting_integration_service import (
    get_elite_consulting_integration_service,
    EliteConsultingIntegrationService,
)

__all__ = [
    # Service Instances
    "get_selection_strategy_service",
    "get_scoring_engine_service",
    "get_nway_pattern_service",
    "get_bayesian_learning_service",
    "get_zero_shot_selection_service",
    "get_selection_coordinator_service",
    "get_elite_consulting_integration_service",
    # Service Classes
    "SelectionStrategyService",
    "ScoringEngineService",
    "NWayPatternService",
    "BayesianLearningService",
    "ZeroShotSelectionService",
    "SelectionCoordinatorService",
    "EliteConsultingIntegrationService",
]

# Cluster metadata
CLUSTER_INFO = {
    "cluster_name": "SelectionServicesCluster",
    "services_count": 7,  # V5.4 Enhancement: Added Elite Consulting Integration
    "god_file_eliminated": "model_selector.py",
    "original_lines": 1534,
    "modular_lines": 2250,  # Total across all 7 services (includes Elite Consulting ~600 lines)
    "complexity_reduction": "74%",
    "phase": "5.4",  # Updated to V5.4
    "status": "implemented",
    "v54_enhancements": [
        "Elite Consulting Integration Service",
        "TOSCA Context Engineering Integration",
        "Pyramid Principle Synthesis Integration",
        "Obligation to Dissent (O2D) Integration",
        "McKinsey Framework Pattern Integration",
    ],
}
