"""
Context Engineering Module
First-class context engineering discipline for the Lolla platform
"""

from .context_compiler import (
    CompiledContext,
    StageContextCompiler,
    SocraticEngineCompiler,
    ProblemStructuringCompiler,
    ConsultantSelectionCompiler,
    ParallelAnalysisCompiler,
    DevilsAdvocateCompiler,
    SeniorAdvisorCompiler,
    get_stage_compiler,
)

__all__ = [
    "CompiledContext",
    "StageContextCompiler",
    "SocraticEngineCompiler",
    "ProblemStructuringCompiler",
    "ConsultantSelectionCompiler",
    "ParallelAnalysisCompiler",
    "DevilsAdvocateCompiler",
    "SeniorAdvisorCompiler",
    "get_stage_compiler",
]
