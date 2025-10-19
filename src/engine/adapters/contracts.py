"""Contracts adapter"""
from src.core.contracts.quality import (
    QualityRubric,
    QualityDimension,
)
from src.core.contracts.rubric_variants import rubric_registry
__all__ = ["QualityRubric", "QualityDimension", "rubric_registry"]
