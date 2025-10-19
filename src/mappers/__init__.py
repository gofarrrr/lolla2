"""
Mappers Package
===============

Pure mapping functions for converting between pipeline contracts and orchestrator formats.

Purpose:
- Extract conversion logic from stage executors
- Enable isolated testing of type transformations
- Maintain single responsibility principle

Modules:
- senior_advisor_mapper: PipelineState ↔ SeniorAdvisor orchestrator formats
"""

from src.mappers.senior_advisor_mapper import PipelineStateToSeniorAdvisorMapper

__all__ = [
    "PipelineStateToSeniorAdvisorMapper",
]
