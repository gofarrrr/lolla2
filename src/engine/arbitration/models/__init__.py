"""
Senior Advisor Data Models - Multiple Single Agents Arbitration System
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from uuid import UUID


class ConsultantRole(str, Enum):
    """Three consultant roles in the system"""

    ANALYST = "analyst"
    STRATEGIST = "strategist"
    DEVIL_ADVOCATE = "devil_advocate"
    IMPLEMENTER = "implementer"
    SPECIALIST = "specialist"


class PerspectiveType(str, Enum):
    """Types of analytical perspectives"""

    RISK_FOCUSED = "risk_focused"
    OPPORTUNITY_FOCUSED = "opportunity_focused"
    USER_EXPERIENCE_FOCUSED = "user_experience_focused"
    FINANCIAL_FOCUSED = "financial_focused"
    OPERATIONAL_FOCUSED = "operational_focused"
    STRATEGIC_FOCUSED = "strategic_focused"
    COMPLIANCE_FOCUSED = "compliance_focused"


class SynergyType(str, Enum):
    """Types of synergies between consultant outputs"""

    COMPLEMENTARY_BLIND_SPOTS = "complementary_blind_spots"
    REINFORCING_EVIDENCE = "reinforcing_evidence"
    SEQUENTIAL_APPLICATIONS = "sequential_applications"
    RISK_REWARD_BALANCING = "risk_reward_balancing"
    IMPLEMENTATION_LAYERING = "implementation_layering"


class MeritCriterion(str, Enum):
    """Merit assessment criteria"""

    EVIDENCE_QUALITY = "evidence_quality"
    LOGICAL_CONSISTENCY = "logical_consistency"
    QUERY_ALIGNMENT = "query_alignment"
    IMPLEMENTATION_FEASIBILITY = "implementation_feasibility"
    RISK_THOROUGHNESS = "risk_thoroughness"
    NOVEL_INSIGHTS = "novel_insights"
    BIAS_RESISTANCE = "bias_resistance"


@dataclass
class ConsultantOutput:
    """Output from a single consultant"""

    consultant_role: ConsultantRole
    analysis_id: str
    query: str

    # Core Analysis
    executive_summary: str
    key_insights: List[str]
    recommendations: List[str]
    mental_models_used: List[str]

    # Research & Evidence
    evidence_sources: List[str]
    research_depth_score: float  # 0.0 to 1.0
    fact_pack_quality: str  # "high", "medium", "low"

    # Red Team Council Results
    red_team_results: Dict[str, Any]  # Munger, Ackoff, Bias results
    bias_detection_score: float  # 0.0 to 1.0
    logical_consistency_score: float  # 0.0 to 1.0

    # Processing Metadata
    processing_time_seconds: float
    cost_usd: float
    confidence_level: float  # 0.0 to 1.0
    created_at: datetime

    # Analysis Metadata
    primary_perspective: PerspectiveType
    approach_description: str
    limitations_identified: List[str]
    assumptions_made: List[str]


@dataclass
class UniqueInsight:
    """Insight found by only one consultant"""

    consultant_role: ConsultantRole
    insight: str
    supporting_evidence: List[str]
    confidence: float
    why_others_missed: str  # Explanation of why other consultants didn't find this


@dataclass
class ConvergentFinding:
    """Finding where multiple consultants agree"""

    finding: str
    supporting_consultants: List[ConsultantRole]
    evidence_overlap: float  # 0.0 to 1.0
    consensus_strength: float  # 0.0 to 1.0
    independent_validation: bool  # True if arrived at independently


@dataclass
class PerspectiveDifference:
    """Difference in analytical approaches"""

    dimension: str  # "mental_models", "evidence_focus", "risk_appetite", etc.
    consultant_approaches: Dict[ConsultantRole, str]
    impact_on_conclusions: str
    user_choice_needed: bool


@dataclass
class SynergyOpportunity:
    """Opportunity for consultant outputs to complement each other"""

    synergy_type: SynergyType
    involved_consultants: List[ConsultantRole]
    description: str
    potential_value: str
    implementation_approach: str
    confidence: float


@dataclass
class MeritScore:
    """Merit assessment for a consultant output"""

    criterion: MeritCriterion
    score: float  # 0.0 to 1.0
    explanation: str
    supporting_evidence: List[str]
    relative_ranking: int  # 1, 2, or 3 among the three consultants


@dataclass
class ConsultantMeritAssessment:
    """Complete merit assessment for one consultant"""

    consultant_role: ConsultantRole
    overall_merit_score: float  # 0.0 to 1.0
    criterion_scores: Dict[MeritCriterion, MeritScore]
    strengths: List[str]
    weaknesses: List[str]
    query_fitness_score: float  # How well suited for this specific query
    recommended_weight: float  # 0.0 to 1.0


@dataclass
class DifferentialAnalysis:
    """Complete differential analysis of all consultant outputs"""

    analysis_id: str
    original_query: str
    consultant_outputs: List[ConsultantOutput]

    # Unique vs Convergent Analysis
    unique_insights: List[UniqueInsight]
    convergent_findings: List[ConvergentFinding]

    # Perspective Analysis
    perspective_differences: List[PerspectiveDifference]
    perspective_map: Dict[ConsultantRole, Dict[str, Any]]

    # Synergy Analysis
    synergy_opportunities: List[SynergyOpportunity]
    complementarity_score: float  # How well outputs complement each other

    # Merit Assessment
    merit_assessments: Dict[ConsultantRole, ConsultantMeritAssessment]

    # Summary Insights
    analysis_summary: str
    key_decision_points: List[str]
    recommended_approach: str

    # Metadata
    analysis_timestamp: datetime
    processing_time_seconds: float


@dataclass
class UserWeightingPreferences:
    """User's weighting preferences"""

    consultant_weights: Dict[ConsultantRole, float]  # Must sum to 1.0
    criterion_priorities: Dict[MeritCriterion, float]  # User's priorities
    risk_tolerance: float  # 0.0 (risk-averse) to 1.0 (risk-seeking)
    implementation_horizon: str = "medium_term"
    decision_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArbitrationResult:
    """Final arbitration result with user weighting applied"""

    differential_analysis: DifferentialAnalysis
    user_preferences: UserWeightingPreferences

    # Weighted Results
    weighted_recommendations: List[str]
    weighted_insights: List[str]
    weighted_risk_assessment: str

    # Decision Support
    primary_consultant_recommendation: ConsultantRole
    supporting_consultants_rationale: Dict[ConsultantRole, str]
    alternative_scenarios: List[Dict[str, Any]]  # What-if analysis

    # Implementation Guidance
    implementation_priority_order: List[str]
    success_metrics: List[str]
    monitoring_recommendations: List[str]

    # Metadata
    arbitration_timestamp: datetime
    user_satisfaction_prediction: float  # ML-based prediction


# Export all models
__all__ = [
    "ConsultantRole",
    "PerspectiveType",
    "SynergyType",
    "MeritCriterion",
    "ConsultantOutput",
    "UniqueInsight",
    "ConvergentFinding",
    "PerspectiveDifference",
    "SynergyOpportunity",
    "MeritScore",
    "ConsultantMeritAssessment",
    "DifferentialAnalysis",
    "UserWeightingPreferences",
    "ArbitrationResult",
]
