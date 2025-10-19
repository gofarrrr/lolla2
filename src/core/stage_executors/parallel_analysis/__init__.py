"""
Parallel Analysis Executor - Modular Implementation

Operation Bedrock refactoring: Decomposed parallel analysis executor with clean interfaces.

This package contains the modular components of the parallel analysis stage:
- interfaces: Stable interface definitions
- types: Data contracts and type definitions
- prompt_builder: Consultant prompt construction
- runner: Parallel LLM execution
- aggregator: Result merging and orthogonality computation
- depth_pack: Stage 0 enrichment integration
"""

from .interfaces import PromptBuilder, Runner, Aggregator, DepthPack
from .types import (
    PromptSpec,
    LLMResult,
    AggregatedOutput,
    EnrichedOutput,
    ExecutionPolicy,
    AggregationPolicy,
    DepthContext,
    RetryConfig,
    MergeStrategy,
    EnrichmentLevel,
)
from .prompt_builder import StandardPromptBuilder
from .runner import ParallelRunner
from .aggregator import StandardAggregator
from .depth_pack import StandardDepthPack

__all__ = [
    # Interfaces
    "PromptBuilder",
    "Runner",
    "Aggregator",
    "DepthPack",
    # Implementations
    "StandardPromptBuilder",
    "ParallelRunner",
    "StandardAggregator",
    "StandardDepthPack",
    # Types
    "PromptSpec",
    "LLMResult",
    "AggregatedOutput",
    "EnrichedOutput",
    "ExecutionPolicy",
    "AggregationPolicy",
    "DepthContext",
    "RetryConfig",
    "MergeStrategy",
    "EnrichmentLevel",
]
