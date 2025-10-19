"""
Honest Orchestrator Data Contracts
==================================

Clean data contracts for the V2.2 Honest Orchestrator pipeline.
Each orchestrator step has well-defined inputs and outputs.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from enum import Enum

# ============================================================================
# STEP 1: SOCRATIC ENGINE CONTRACTS
# ============================================================================


@dataclass
class TieredQuestion:
    """A single clarification question from the Socratic Engine"""

    tier: int
    question: str
    rationale: str
    simulated_answer: Optional[str] = None


@dataclass
class EnhancedQuery:
    """Output of the Socratic Engine - enhanced query with context"""

    original_query: str
    enhanced_query: str
    clarifying_questions: List[TieredQuestion]
    context_enrichment: Dict[str, Any]
    confidence_score: float
    processing_time_seconds: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# STEP 2: PROBLEM STRUCTURING CONTRACTS
# ============================================================================


class FrameworkType(Enum):
    STRATEGIC_ANALYSIS = "strategic_analysis"
    OPERATIONAL_OPTIMIZATION = "operational_optimization"
    INNOVATION_DISCOVERY = "innovation_discovery"
    CRISIS_MANAGEMENT = "crisis_management"


@dataclass
class AnalyticalDimension:
    """A single dimension of the analytical framework"""

    dimension_name: str
    key_questions: List[str]
    analysis_approach: str
    priority_level: int


@dataclass
class StructuredAnalyticalFramework:
    """Output of the Problem Structuring Agent"""

    framework_type: FrameworkType
    primary_dimensions: List[AnalyticalDimension]
    secondary_considerations: List[str]
    analytical_sequence: List[str]
    complexity_assessment: str
    recommended_consultant_types: List[str]
    processing_time_seconds: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# STEP 3: DISPATCH CONTRACTS
# ============================================================================


@dataclass
class ConsultantBlueprint:
    """Definition of a selected consultant"""

    consultant_id: str
    consultant_type: str
    specialization: str
    predicted_effectiveness: float
    assigned_dimensions: List[str]


@dataclass
class NWayConfiguration:
    """N-Way analysis configuration"""

    pattern_name: str
    consultant_cluster: List[ConsultantBlueprint]
    interaction_strategy: str


@dataclass
class DispatchPackage:
    """Output of the Dispatch Orchestrator"""

    selected_consultants: List[ConsultantBlueprint]
    nway_configuration: NWayConfiguration
    dispatch_rationale: str
    confidence_score: float
    processing_time_seconds: float
    s2_tier: str = "S2_DISABLED"  # SYSTEM-2 KERNEL: Reasoning tier
    s2_rationale: str = "not determined"  # SYSTEM-2 KERNEL: Tier reasoning
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# STEP 4: PARALLEL FORGE CONTRACTS
# ============================================================================


@dataclass
class ConsultantAnalysisResult:
    """Result from a single consultant analysis"""

    consultant_id: str
    analysis_content: str
    mental_models_applied: List[str]
    confidence_level: float
    key_insights: List[str]
    recommendations: List[str]
    research_citations: List[str]
    processing_time_seconds: float
    llm_tokens_used: int
    llm_cost_usd: float


@dataclass
class AnalysisCritique:
    """Result from Devil's Advocate critique"""

    target_consultant: str
    critique_content: str
    identified_weaknesses: List[str]
    alternative_perspectives: List[str]
    risk_assessments: List[str]
    confidence_level: float
    processing_time_seconds: float
    engines_used: List[str]


@dataclass
class ParallelForgeResults:
    """Combined results from parallel forge execution"""

    consultant_analyses: List[ConsultantAnalysisResult]
    critiques: List[AnalysisCritique]
    total_processing_time_seconds: float
    successful_analyses: int
    successful_critiques: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# STEP 5: SENIOR ADVISOR CONTRACTS
# ============================================================================


@dataclass
class TwoBrainInsight:
    """Insight from one brain of the two-brain process"""

    brain_name: str  # "deepseek" or "claude"
    insight_content: str
    confidence_level: float
    key_points: List[str]
    processing_time_seconds: float
    tokens_used: int
    cost_usd: float


@dataclass
class SeniorAdvisorReport:
    """Final output of the Senior Advisor two-brain process"""

    executive_summary: str
    strategic_recommendation: str
    implementation_roadmap: List[str]
    risk_mitigation: List[str]
    success_metrics: List[str]

    deepseek_brain: TwoBrainInsight
    claude_brain: TwoBrainInsight
    synthesis_rationale: str

    final_confidence: float
    total_processing_time_seconds: float
    total_cost_usd: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# MASTER SYMPHONY CONTRACT
# ============================================================================


@dataclass
class SymphonyExecutionResult:
    """Complete result from the honest orchestrator symphony"""

    raw_query: str
    enhanced_query: EnhancedQuery
    analytical_framework: StructuredAnalyticalFramework
    dispatch_package: DispatchPackage
    forge_results: ParallelForgeResults
    final_report: SeniorAdvisorReport

    total_execution_time_seconds: float
    total_cost_usd: float
    success: bool
    failure_stage: Optional[str] = None
    failure_reason: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
