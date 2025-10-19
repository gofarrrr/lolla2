"""
Enumerations for Pyramid Principle Engine
"""

from enum import Enum


class PyramidLevel(str, Enum):
    """Levels in the Pyramid Principle structure"""

    GOVERNING_THOUGHT = "governing_thought"  # Top message
    KEY_LINES = "key_lines"  # Main supporting arguments
    SUPPORTING_POINTS = "supporting_points"  # Evidence and details
    DATA_EXHIBITS = "data_exhibits"  # Supporting data and charts


class ArgumentType(str, Enum):
    """Types of logical arguments"""

    DEDUCTIVE = "deductive"  # Logical sequence: situation → complication → resolution
    INDUCTIVE = "inductive"  # Grouped similar ideas supporting conclusion
    ABDUCTIVE = "abductive"  # Best explanation given available evidence
    COMPARATIVE = "comparative"  # Comparison-based reasoning
    TEMPORAL = "temporal"  # Time-based sequence


class DeliverableType(str, Enum):
    """Types of consulting deliverables"""

    EXECUTIVE_SUMMARY = "executive_summary"
    STRATEGY_DOCUMENT = "strategy_document"
    BUSINESS_CASE = "business_case"
    RECOMMENDATION_MEMO = "recommendation_memo"
    FINAL_PRESENTATION = "final_presentation"
    IMPLEMENTATION_PLAN = "implementation_plan"
