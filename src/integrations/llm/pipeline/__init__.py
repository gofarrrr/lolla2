"""
LLM Call Pipeline Package

Production-ready pipeline pattern implementation for unified_client.py refactoring.

This package provides:
- LLMCallContext: Immutable context object passed through pipeline stages
- PipelineStage: Abstract base class for pipeline stages
- LLMCallPipeline: Orchestrator that executes stages sequentially

Example Usage:
    ```python
    from src.integrations.llm.pipeline import (
        LLMCallContext,
        LLMCallPipeline,
        PipelineStage,
    )

    # Create context
    context = LLMCallContext(
        messages=[{"role": "user", "content": "Hello"}],
        model="deepseek-chat",
        provider="deepseek",
        kwargs={}
    )

    # Create pipeline with stages
    pipeline = LLMCallPipeline(stages=[
        InjectionFirewallStage(),
        PIIRedactionStage(),
        ProviderAdapterStage(),
    ])

    # Execute
    result = await pipeline.execute(context)
    ```

Architecture:
    Pipeline Pattern (Chain of Responsibility + Strategy)
    - Stages execute sequentially
    - Context flows immutably through stages
    - Each stage returns new context with updates
    - Errors handled gracefully (fatal vs non-fatal)
    - Telemetry tracked per-stage and overall

Refactoring Context:
    This package refactors unified_client.py::call_llm() from:
    - Complexity: CC=84 (Grade F - Unmaintainable)
    - To: CC<10 (Grade A - Maintainable)

Version: 1.0.0
Date: 2025-10-17
"""

# === Core Classes ===
from .context import LLMCallContext
from .stage import PipelineStage
from .pipeline import LLMCallPipeline
from .factory import create_llm_pipeline

# === Exceptions ===
from .stage import (
    PipelineStageError,
    StageValidationError,
    StageExecutionError,
    StageConfigurationError,
)
from .pipeline import (
    PipelineValidationError,
    PipelineExecutionError,
)

# === Public API ===
__all__ = [
    # Core Classes
    "LLMCallContext",
    "PipelineStage",
    "LLMCallPipeline",
    "create_llm_pipeline",
    # Stage Exceptions
    "PipelineStageError",
    "StageValidationError",
    "StageExecutionError",
    "StageConfigurationError",
    # Pipeline Exceptions
    "PipelineValidationError",
    "PipelineExecutionError",
]

# === Version Info ===
__version__ = "1.0.0"
__author__ = "METIS V5.3 Team"
__date__ = "2025-10-17"

# === Package Metadata ===
__package_name__ = "llm_pipeline"
__description__ = "Pipeline pattern implementation for LLM call orchestration"
__refactoring_task__ = "P0 #0 unified_client.py refactoring"
