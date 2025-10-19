# Orchestration package exports for walking skeleton
from .coreops_executor import execute_core_program

"""
Honest Orchestrator Package
===========================

Clean V2.2 orchestrator implementations with real LLM calls.
NO MOCK DATA. NO FALLBACKS. REAL EXECUTION ONLY.
"""

# Import main orchestrator functions
from .socratic_engine_orchestrator import run_socratic_inquiry
from .problem_structuring_orchestrator import run_problem_structuring
from .dispatch_orchestrator import run_dispatch
from .parallel_forge_orchestrator import run_parallel_forges
from .senior_advisor_orchestrator import run_senior_advisor

# Import exceptions
from .exceptions import (
    OrchestrationError,
    SocraticEngineError,
    PSAError,
    DispatchError,
    ForgeError,
    SeniorAdvisorError,
    SymphonyError,
)

# Import contracts (legacy orchestrator data contracts)
from .contracts import (
    EnhancedQuery,
    TieredQuestion,
    StructuredAnalyticalFramework,
    DispatchPackage,
    ConsultantBlueprint,
    NWayConfiguration,
    ConsultantAnalysisResult,
    AnalysisCritique,
    ParallelForgeResults,
    SeniorAdvisorReport,
    TwoBrainInsight,
    SymphonyExecutionResult,
)

# Expose submodules for explicit package-level access
from . import contracts as contracts  # legacy
from . import flow_contracts as flow_contracts  # new pipeline/flow contracts

__all__ = [
    # Main orchestrator functions
    "run_socratic_inquiry",
    "run_problem_structuring",
    "run_dispatch",
    "run_parallel_forges",
    "run_senior_advisor",
    # Exceptions
    "OrchestrationError",
    "SocraticEngineError",
    "PSAError",
    "DispatchError",
    "ForgeError",
    "SeniorAdvisorError",
    "SymphonyError",
    # Contracts (legacy)
    "EnhancedQuery",
    "TieredQuestion",
    "StructuredAnalyticalFramework",
    "DispatchPackage",
    "ConsultantBlueprint",
    "NWayConfiguration",
    "ConsultantAnalysisResult",
    "AnalysisCritique",
    "ParallelForgeResults",
    "SeniorAdvisorReport",
    "TwoBrainInsight",
    "SymphonyExecutionResult",
    # Submodules
    "contracts",
    "flow_contracts",
]
