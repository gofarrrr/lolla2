"""
Problem Structuring Seam
------------------------

Thin seam wrapper around the ProblemStructuringOrchestrator's core steps.
Purpose: enable safe, incremental extraction/refactor without changing behavior.

Each method delegates to the host's legacy implementation and returns the result.
If delegation fails for any reason, errors are propagated to preserve current
error semantics (callers already handle PSAError at higher levels).
"""

from __future__ import annotations

from typing import Any, List, Tuple


class ProblemStructuringSeam:
    async def determine_framework_type(self, host: Any, enhanced_query: Any) -> Any:
        return await host._determine_framework_type(enhanced_query)  # noqa: SLF001

    async def generate_analytical_dimensions(
        self,
        host: Any,
        enhanced_query: Any,
        framework_type: Any,
    ) -> List[Any]:
        return await host._generate_analytical_dimensions(  # noqa: SLF001
            enhanced_query, framework_type
        )

    async def define_analysis_approach(
        self,
        host: Any,
        enhanced_query: Any,
        framework_type: Any,
        dimensions: List[Any],
    ) -> Tuple[List[str], List[str], str]:
        return await host._define_analysis_approach(  # noqa: SLF001
            enhanced_query, framework_type, dimensions
        )

    async def extract_secondary_considerations(self, host: Any, enhanced_query: Any) -> List[str]:
        return await host._extract_secondary_considerations(enhanced_query)  # noqa: SLF001

