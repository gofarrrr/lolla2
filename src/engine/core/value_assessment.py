"""
Value Assessment System for METIS
Captures and analyzes value metrics including actionability, novelty, ROI, and strategic impact
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import threading
from collections import defaultdict, OrderedDict
import statistics
import os
from uuid import UUID
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()


class ValueDimension(str, Enum):
    """Core value dimensions for assessment"""

    ACTIONABILITY = "actionability"  # How implementable the insights are
    NOVELTY = "novelty"  # How new/unique the insights are
    STRATEGIC_IMPACT = "strategic_impact"  # Long-term strategic value
    OPERATIONAL_IMPACT = "operational_impact"  # Short-term operational value
    FINANCIAL_IMPACT = "financial_impact"  # Quantifiable financial benefit
    RISK_MITIGATION = "risk_mitigation"  # Risk reduction value
    COMPETITIVE_ADVANTAGE = "competitive_advantage"  # Market positioning value
    STAKEHOLDER_VALUE = "stakeholder_value"  # Value to different stakeholders


class ValueScale(str, Enum):
    """Standardized value scales"""

    VERY_LOW = "very_low"  # 0.0-0.2
    LOW = "low"  # 0.2-0.4
    MEDIUM = "medium"  # 0.4-0.6
    HIGH = "high"  # 0.6-0.8
    VERY_HIGH = "very_high"  # 0.8-1.0


class TimeHorizon(str, Enum):
    """Time horizons for value realization"""

    IMMEDIATE = "immediate"  # 0-3 months
    SHORT_TERM = "short_term"  # 3-12 months
    MEDIUM_TERM = "medium_term"  # 1-3 years
    LONG_TERM = "long_term"  # 3+ years


class ValueCategory(str, Enum):
    """Categories of value for business analysis"""

    COST_REDUCTION = "cost_reduction"
    REVENUE_ENHANCEMENT = "revenue_enhancement"
    RISK_REDUCTION = "risk_reduction"
    EFFICIENCY_IMPROVEMENT = "efficiency_improvement"
    INNOVATION_ENABLEMENT = "innovation_enablement"
    CUSTOMER_SATISFACTION = "customer_satisfaction"
    MARKET_POSITIONING = "market_positioning"
    OPERATIONAL_EXCELLENCE = "operational_excellence"


@dataclass
class ValueMetric:
    """Individual value metric with measurement details"""

    dimension: ValueDimension
    raw_score: float  # 0.0-1.0
    normalized_score: float  # 0.0-1.0 after normalization
    confidence: float  # 0.0-1.0

    # Supporting data
    evidence: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    measurement_method: str = ""
    rationale: str = ""

    # Contextual information
    time_horizon: TimeHorizon = TimeHorizon.MEDIUM_TERM
    value_category: ValueCategory = ValueCategory.EFFICIENCY_IMPROVEMENT
    stakeholder_impact: Dict[str, float] = field(default_factory=dict)

    # Metadata
    measured_at: datetime = field(default_factory=datetime.utcnow)
    measured_by: Optional[str] = None

    def get_value_scale(self) -> ValueScale:
        """Convert normalized score to value scale"""
        if self.normalized_score < 0.2:
            return ValueScale.VERY_LOW
        elif self.normalized_score < 0.4:
            return ValueScale.LOW
        elif self.normalized_score < 0.6:
            return ValueScale.MEDIUM
        elif self.normalized_score < 0.8:
            return ValueScale.HIGH
        else:
            return ValueScale.VERY_HIGH


@dataclass
class ROIAnalysis:
    """Return on Investment analysis"""

    estimated_cost: float = 0.0
    estimated_benefit: float = 0.0
    payback_period_months: Optional[float] = None
    net_present_value: Optional[float] = None
    internal_rate_of_return: Optional[float] = None

    # Cost breakdown
    implementation_cost: float = 0.0
    operational_cost_annual: float = 0.0
    maintenance_cost_annual: float = 0.0

    # Benefit breakdown
    cost_savings_annual: float = 0.0
    revenue_increase_annual: float = 0.0
    efficiency_gains_annual: float = 0.0

    # Risk factors
    cost_uncertainty: float = 0.2  # ±20% default
    benefit_uncertainty: float = 0.3  # ±30% default
    implementation_risk: float = 0.1  # 10% default

    # Time factors
    time_to_value_months: float = 6.0
    benefit_duration_years: float = 3.0
    discount_rate: float = 0.1  # 10% default

    def calculate_roi_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive ROI metrics"""
        metrics = {}

        # Simple ROI
        if self.estimated_cost > 0:
            metrics["simple_roi"] = (
                self.estimated_benefit - self.estimated_cost
            ) / self.estimated_cost
        else:
            metrics["simple_roi"] = 0.0

        # Payback period
        annual_net_benefit = (
            self.cost_savings_annual
            + self.revenue_increase_annual
            + self.efficiency_gains_annual
            - self.operational_cost_annual
            - self.maintenance_cost_annual
        )
        if annual_net_benefit > 0:
            metrics["payback_period_years"] = self.estimated_cost / annual_net_benefit
        else:
            metrics["payback_period_years"] = float("inf")

        # NPV calculation (simplified)
        if self.discount_rate > 0:
            annual_benefits = []
            for year in range(int(self.benefit_duration_years)):
                annual_benefit = annual_net_benefit / ((1 + self.discount_rate) ** year)
                annual_benefits.append(annual_benefit)
            metrics["net_present_value"] = sum(annual_benefits) - self.estimated_cost
        else:
            metrics["net_present_value"] = self.estimated_benefit - self.estimated_cost

        # Risk-adjusted metrics
        metrics["risk_adjusted_roi"] = metrics["simple_roi"] * (
            1 - self.implementation_risk
        )
        metrics["cost_range_low"] = self.estimated_cost * (1 - self.cost_uncertainty)
        metrics["cost_range_high"] = self.estimated_cost * (1 + self.cost_uncertainty)
        metrics["benefit_range_low"] = self.estimated_benefit * (
            1 - self.benefit_uncertainty
        )
        metrics["benefit_range_high"] = self.estimated_benefit * (
            1 + self.benefit_uncertainty
        )

        return metrics


@dataclass
class ValueAssessment:
    """Comprehensive value assessment for insights or recommendations"""

    id: str
    name: str
    description: str

    # Value metrics across dimensions
    value_metrics: Dict[ValueDimension, ValueMetric] = field(default_factory=dict)

    # Overall scores
    overall_value_score: float = 0.0
    weighted_value_score: float = 0.0
    confidence_score: float = 0.0

    # ROI analysis
    roi_analysis: Optional[ROIAnalysis] = None

    # Actionability assessment
    implementation_complexity: Literal["low", "medium", "high"] = "medium"
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    prerequisites: List[str] = field(default_factory=list)
    barriers_to_implementation: List[str] = field(default_factory=list)
    success_factors: List[str] = field(default_factory=list)

    # Impact assessment
    primary_stakeholders: List[str] = field(default_factory=list)
    secondary_stakeholders: List[str] = field(default_factory=list)
    impact_timeline: Dict[TimeHorizon, List[str]] = field(default_factory=dict)

    # Risk assessment
    implementation_risks: List[Dict[str, Any]] = field(default_factory=list)
    value_risks: List[Dict[str, Any]] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)

    # Provenance and traceability
    data_sources: List[str] = field(default_factory=list)
    analysis_methods: List[str] = field(default_factory=list)
    expert_inputs: List[Dict[str, Any]] = field(default_factory=list)
    validation_steps: List[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    engagement_id: Optional[str] = None
    phase: Optional[str] = None

    @property
    def confidence_level(self) -> float:
        """Property to provide confidence_level as alias for confidence_score"""
        return self.confidence_score

    def add_value_metric(self, metric: ValueMetric) -> None:
        """Add value metric and recalculate overall scores"""
        self.value_metrics[metric.dimension] = metric
        self._recalculate_scores()
        self.last_updated = datetime.utcnow()

    def _recalculate_scores(self) -> None:
        """Recalculate overall value scores"""
        if not self.value_metrics:
            self.overall_value_score = 0.0
            self.weighted_value_score = 0.0
            self.confidence_score = 0.0
            return

        # Calculate simple average
        scores = [metric.normalized_score for metric in self.value_metrics.values()]
        self.overall_value_score = sum(scores) / len(scores)

        # Calculate weighted average (with strategic weights)
        weights = {
            ValueDimension.STRATEGIC_IMPACT: 0.25,
            ValueDimension.FINANCIAL_IMPACT: 0.20,
            ValueDimension.ACTIONABILITY: 0.15,
            ValueDimension.OPERATIONAL_IMPACT: 0.15,
            ValueDimension.NOVELTY: 0.10,
            ValueDimension.COMPETITIVE_ADVANTAGE: 0.10,
            ValueDimension.RISK_MITIGATION: 0.05,
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for dimension, metric in self.value_metrics.items():
            weight = weights.get(dimension, 0.05)  # Default weight
            weighted_sum += metric.normalized_score * weight
            total_weight += weight

        self.weighted_value_score = (
            weighted_sum / total_weight if total_weight > 0 else 0.0
        )

        # Calculate confidence score
        confidences = [metric.confidence for metric in self.value_metrics.values()]
        self.confidence_score = sum(confidences) / len(confidences)

    def get_value_summary(self) -> Dict[str, Any]:
        """Get summary of value assessment"""
        roi_summary = {}
        if self.roi_analysis:
            roi_metrics = self.roi_analysis.calculate_roi_metrics()
            roi_summary = {
                "simple_roi": roi_metrics.get("simple_roi", 0.0),
                "payback_period_years": roi_metrics.get("payback_period_years", 0.0),
                "net_present_value": roi_metrics.get("net_present_value", 0.0),
            }

        return {
            "assessment_id": self.id,
            "name": self.name,
            "overall_value_score": self.overall_value_score,
            "weighted_value_score": self.weighted_value_score,
            "confidence_score": self.confidence_score,
            "implementation_complexity": self.implementation_complexity,
            "value_metrics_count": len(self.value_metrics),
            "roi_analysis": roi_summary,
            "last_updated": self.last_updated.isoformat(),
        }


class ValueAssessmentEngine:
    """
    Engine for comprehensive value assessment with systematic measurement and analysis
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        enable_persistence: bool = True,
        max_cache_size: int = 500,
        enable_supabase: bool = True,
    ):
        self.logger = logging.getLogger(__name__)
        self.storage_path = (
            Path(storage_path) if storage_path else Path("data/value_assessments")
        )
        self.enable_persistence = enable_persistence
        self.max_cache_size = max_cache_size
        self.enable_supabase = enable_supabase

        # In-memory storage
        self.assessments: Dict[str, ValueAssessment] = OrderedDict()

        # Threading safety
        self._lock = threading.RLock()

        # Analytics and benchmarks
        self.value_benchmarks = defaultdict(list)
        self.performance_metrics = defaultdict(list)

        # Supabase configuration
        self.supabase_client = None
        if self.enable_supabase:
            try:
                url = os.environ.get("SUPABASE_URL")
                key = os.environ.get("SUPABASE_ANON_KEY")
                if url and key:
                    self.supabase_client = create_client(url, key)
                    self.logger.info("Supabase client initialized for value assessment")
                else:
                    self.logger.warning(
                        "Supabase credentials not found, falling back to local storage"
                    )
                    self.enable_supabase = False
            except Exception as e:
                self.logger.warning(f"Failed to initialize Supabase client: {e}")
                self.enable_supabase = False

        # Ensure storage directory exists
        if self.enable_persistence:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(
                f"✅ ValueAssessmentEngine initialized with storage: {self.storage_path}"
            )
        else:
            self.logger.info("✅ ValueAssessmentEngine initialized (memory-only mode)")

    def create_value_assessment(
        self,
        name: str,
        description: str,
        engagement_id: Optional[str] = None,
        phase: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> ValueAssessment:
        """Create new value assessment"""
        assessment_id = str(uuid.uuid4())

        assessment = ValueAssessment(
            id=assessment_id,
            name=name,
            description=description,
            engagement_id=engagement_id,
            phase=phase,
            created_by=created_by,
        )

        with self._lock:
            self.assessments[assessment_id] = assessment
            self._enforce_cache_limits()

        if self.enable_persistence:
            self._persist_assessment(assessment)

        self.logger.info(f"💎 Created value assessment: {name} | ID: {assessment_id}")
        return assessment

    def assess_actionability(
        self, assessment_id: str, insight_text: str, business_context: Dict[str, Any]
    ) -> ValueMetric:
        """Assess actionability of insights"""

        # Actionability factors
        actionability_score = 0.5  # Base score

        # Check for specific actions
        action_indicators = [
            "implement",
            "execute",
            "deploy",
            "launch",
            "establish",
            "create",
            "build",
            "develop",
            "introduce",
            "adopt",
        ]

        text_lower = insight_text.lower()
        action_count = sum(
            1 for indicator in action_indicators if indicator in text_lower
        )
        actionability_score += min(0.3, action_count * 0.05)

        # Check for timelines
        timeline_indicators = [
            "immediately",
            "within",
            "by",
            "during",
            "month",
            "quarter",
            "year",
        ]
        timeline_count = sum(
            1 for indicator in timeline_indicators if indicator in text_lower
        )
        actionability_score += min(0.1, timeline_count * 0.02)

        # Check for resource specificity
        resource_indicators = [
            "budget",
            "team",
            "staff",
            "investment",
            "cost",
            "resource",
        ]
        resource_count = sum(
            1 for indicator in resource_indicators if indicator in text_lower
        )
        actionability_score += min(0.1, resource_count * 0.02)

        # Business context factors
        if business_context.get("implementation_readiness") == "high":
            actionability_score += 0.1
        elif business_context.get("implementation_readiness") == "low":
            actionability_score -= 0.1

        # Cap at 1.0
        actionability_score = min(1.0, actionability_score)

        metric = ValueMetric(
            dimension=ValueDimension.ACTIONABILITY,
            raw_score=actionability_score,
            normalized_score=actionability_score,
            confidence=0.7,
            evidence=[
                f"Action indicators: {action_count}",
                f"Timeline references: {timeline_count}",
            ],
            measurement_method="text_analysis_with_context",
            rationale="Assessed based on action specificity and implementation readiness",
        )

        return metric

    def assess_novelty(
        self,
        assessment_id: str,
        insight_text: str,
        historical_insights: List[str] = None,
    ) -> ValueMetric:
        """Assess novelty/uniqueness of insights"""

        novelty_score = 0.7  # Base score (assume moderate novelty)

        # Check for innovation indicators
        innovation_indicators = [
            "new",
            "novel",
            "innovative",
            "breakthrough",
            "revolutionary",
            "unprecedented",
            "first",
            "unique",
            "cutting-edge",
            "disruptive",
        ]

        text_lower = insight_text.lower()
        innovation_count = sum(
            1 for indicator in innovation_indicators if indicator in text_lower
        )
        novelty_score += min(0.2, innovation_count * 0.04)

        # Check against historical insights (simplified similarity)
        if historical_insights:
            similarity_scores = []
            for historical in historical_insights[-10:]:  # Check last 10
                # Simple word overlap similarity
                insight_words = set(insight_text.lower().split())
                historical_words = set(historical.lower().split())

                if insight_words and historical_words:
                    overlap = len(insight_words & historical_words)
                    total = len(insight_words | historical_words)
                    similarity = overlap / total if total > 0 else 0
                    similarity_scores.append(similarity)

            if similarity_scores:
                max_similarity = max(similarity_scores)
                novelty_score -= (
                    max_similarity * 0.5
                )  # Reduce score for high similarity

        # Check for complexity indicators (complex insights often more novel)
        complexity_indicators = [
            "framework",
            "model",
            "methodology",
            "approach",
            "system",
        ]
        complexity_count = sum(
            1 for indicator in complexity_indicators if indicator in text_lower
        )
        novelty_score += min(0.1, complexity_count * 0.02)

        novelty_score = max(0.0, min(1.0, novelty_score))

        metric = ValueMetric(
            dimension=ValueDimension.NOVELTY,
            raw_score=novelty_score,
            normalized_score=novelty_score,
            confidence=0.6,
            evidence=[
                f"Innovation indicators: {innovation_count}",
                f"Complexity indicators: {complexity_count}",
            ],
            measurement_method="novelty_analysis_with_historical_comparison",
            rationale="Assessed based on innovation language and similarity to historical insights",
        )

        return metric

    def assess_strategic_impact(
        self, assessment_id: str, insight_text: str, business_context: Dict[str, Any]
    ) -> ValueMetric:
        """Assess strategic impact and long-term value"""

        strategic_score = 0.5  # Base score

        # Strategic indicators
        strategic_indicators = [
            "strategy",
            "strategic",
            "competitive",
            "market",
            "position",
            "advantage",
            "transformation",
            "vision",
            "mission",
            "goals",
            "long-term",
            "future",
            "growth",
            "expansion",
            "leadership",
        ]

        text_lower = insight_text.lower()
        strategic_count = sum(
            1 for indicator in strategic_indicators if indicator in text_lower
        )
        strategic_score += min(0.3, strategic_count * 0.03)

        # Business context factors
        industry = business_context.get("industry", "").lower()
        if any(term in industry for term in ["technology", "healthcare", "finance"]):
            strategic_score += 0.1  # High-impact industries

        # Company size factor
        company_size = business_context.get("company_size", "medium").lower()
        if company_size == "enterprise":
            strategic_score += 0.1
        elif company_size == "startup":
            strategic_score += 0.05

        # Time horizon factor
        if any(
            term in text_lower for term in ["3-year", "5-year", "decade", "long-term"]
        ):
            strategic_score += 0.1

        strategic_score = min(1.0, strategic_score)

        metric = ValueMetric(
            dimension=ValueDimension.STRATEGIC_IMPACT,
            raw_score=strategic_score,
            normalized_score=strategic_score,
            confidence=0.7,
            evidence=[
                f"Strategic indicators: {strategic_count}",
                f"Industry context: {industry}",
            ],
            measurement_method="strategic_impact_analysis",
            rationale="Assessed based on strategic language and business context",
            time_horizon=TimeHorizon.LONG_TERM,
        )

        return metric

    def assess_financial_impact(
        self, assessment_id: str, insight_text: str, business_context: Dict[str, Any]
    ) -> ValueMetric:
        """Assess financial impact potential"""

        financial_score = 0.4  # Base score

        # Financial indicators
        financial_indicators = [
            "revenue",
            "profit",
            "cost",
            "savings",
            "roi",
            "investment",
            "budget",
            "financial",
            "economic",
            "value",
            "benefit",
            "return",
            "efficiency",
            "productivity",
            "margin",
            "growth",
        ]

        text_lower = insight_text.lower()
        financial_count = sum(
            1 for indicator in financial_indicators if indicator in text_lower
        )
        financial_score += min(0.4, financial_count * 0.04)

        # Look for quantified benefits
        import re

        # Check for percentage improvements
        percentage_matches = re.findall(r"(\d+(?:\.\d+)?)\s*%", insight_text)
        if percentage_matches:
            percentages = [float(p) for p in percentage_matches]
            max_percentage = max(percentages) if percentages else 0
            financial_score += min(
                0.2, max_percentage / 100.0
            )  # Up to 20% boost for high percentages

        # Check for monetary amounts
        money_patterns = [
            r"\$\d+",
            r"\d+\s*million",
            r"\d+\s*billion",
            r"\d+\s*thousand",
        ]
        money_mentions = sum(
            1 for pattern in money_patterns if re.search(pattern, text_lower)
        )
        financial_score += min(0.1, money_mentions * 0.03)

        # Business context factors
        revenue_size = business_context.get("annual_revenue", 0)
        if revenue_size > 1000000000:  # $1B+
            financial_score += 0.1
        elif revenue_size > 100000000:  # $100M+
            financial_score += 0.05

        financial_score = min(1.0, financial_score)

        metric = ValueMetric(
            dimension=ValueDimension.FINANCIAL_IMPACT,
            raw_score=financial_score,
            normalized_score=financial_score,
            confidence=0.6,
            evidence=[
                f"Financial indicators: {financial_count}",
                f"Quantified benefits: {len(percentage_matches)} percentages",
            ],
            measurement_method="financial_impact_analysis",
            rationale="Assessed based on financial language and quantified benefits",
            value_category=ValueCategory.REVENUE_ENHANCEMENT,
        )

        return metric

    def create_roi_analysis(
        self,
        assessment_id: str,
        estimated_cost: float,
        estimated_benefit: float,
        implementation_details: Dict[str, Any] = None,
    ) -> ROIAnalysis:
        """Create comprehensive ROI analysis"""

        details = implementation_details or {}

        roi_analysis = ROIAnalysis(
            estimated_cost=estimated_cost,
            estimated_benefit=estimated_benefit,
            implementation_cost=details.get(
                "implementation_cost", estimated_cost * 0.6
            ),
            operational_cost_annual=details.get(
                "operational_cost_annual", estimated_cost * 0.1
            ),
            maintenance_cost_annual=details.get(
                "maintenance_cost_annual", estimated_cost * 0.05
            ),
            cost_savings_annual=details.get(
                "cost_savings_annual", estimated_benefit * 0.4
            ),
            revenue_increase_annual=details.get(
                "revenue_increase_annual", estimated_benefit * 0.4
            ),
            efficiency_gains_annual=details.get(
                "efficiency_gains_annual", estimated_benefit * 0.2
            ),
            time_to_value_months=details.get("time_to_value_months", 6.0),
            benefit_duration_years=details.get("benefit_duration_years", 3.0),
            discount_rate=details.get("discount_rate", 0.1),
        )

        return roi_analysis

    def perform_comprehensive_assessment(
        self,
        assessment_id: str,
        insight_text: str,
        business_context: Dict[str, Any],
        include_roi: bool = True,
        historical_insights: List[str] = None,
    ) -> ValueAssessment:
        """Perform comprehensive value assessment across all dimensions"""

        with self._lock:
            if assessment_id not in self.assessments:
                self.logger.error(f"❌ Assessment not found: {assessment_id}")
                return None

            assessment = self.assessments[assessment_id]

        # Assess all key dimensions
        metrics = []

        # Core assessments
        metrics.append(
            self.assess_actionability(assessment_id, insight_text, business_context)
        )
        metrics.append(
            self.assess_novelty(assessment_id, insight_text, historical_insights)
        )
        metrics.append(
            self.assess_strategic_impact(assessment_id, insight_text, business_context)
        )
        metrics.append(
            self.assess_financial_impact(assessment_id, insight_text, business_context)
        )

        # Additional assessments
        metrics.append(self._assess_operational_impact(insight_text, business_context))
        metrics.append(self._assess_risk_mitigation(insight_text, business_context))

        # Add metrics to assessment
        for metric in metrics:
            assessment.add_value_metric(metric)

        # ROI analysis if requested
        if (
            include_roi
            and business_context.get("estimated_cost")
            and business_context.get("estimated_benefit")
        ):
            roi_analysis = self.create_roi_analysis(
                assessment_id,
                business_context["estimated_cost"],
                business_context["estimated_benefit"],
                business_context.get("implementation_details", {}),
            )
            assessment.roi_analysis = roi_analysis

        # Assess implementation complexity
        assessment.implementation_complexity = self._assess_implementation_complexity(
            insight_text, business_context
        )

        # Update benchmarks
        self._update_benchmarks(assessment)

        if self.enable_persistence:
            self._persist_assessment(assessment)

        # Store in Supabase if enabled
        if self.enable_supabase and self.supabase_client:
            self._store_assessment_in_supabase(assessment)

        self.logger.info(
            f"💎 Comprehensive assessment completed: {assessment_id} | "
            f"Overall score: {assessment.overall_value_score:.2f} | "
            f"Weighted score: {assessment.weighted_value_score:.2f}"
        )

        return assessment

    def _assess_operational_impact(
        self, insight_text: str, business_context: Dict[str, Any]
    ) -> ValueMetric:
        """Assess operational impact"""
        operational_score = 0.5

        operational_indicators = [
            "process",
            "workflow",
            "efficiency",
            "automation",
            "streamline",
            "optimize",
            "improve",
            "reduce",
            "eliminate",
            "standardize",
        ]

        text_lower = insight_text.lower()
        operational_count = sum(
            1 for indicator in operational_indicators if indicator in text_lower
        )
        operational_score += min(0.3, operational_count * 0.04)

        return ValueMetric(
            dimension=ValueDimension.OPERATIONAL_IMPACT,
            raw_score=operational_score,
            normalized_score=operational_score,
            confidence=0.7,
            evidence=[f"Operational indicators: {operational_count}"],
            measurement_method="operational_impact_analysis",
            time_horizon=TimeHorizon.SHORT_TERM,
        )

    def _assess_risk_mitigation(
        self, insight_text: str, business_context: Dict[str, Any]
    ) -> ValueMetric:
        """Assess risk mitigation value"""
        risk_score = 0.3

        risk_indicators = [
            "risk",
            "mitigate",
            "reduce",
            "prevent",
            "avoid",
            "protect",
            "secure",
            "compliance",
            "regulation",
            "safety",
            "control",
        ]

        text_lower = insight_text.lower()
        risk_count = sum(1 for indicator in risk_indicators if indicator in text_lower)
        risk_score += min(0.4, risk_count * 0.05)

        return ValueMetric(
            dimension=ValueDimension.RISK_MITIGATION,
            raw_score=risk_score,
            normalized_score=risk_score,
            confidence=0.6,
            evidence=[f"Risk indicators: {risk_count}"],
            measurement_method="risk_mitigation_analysis",
            value_category=ValueCategory.RISK_REDUCTION,
        )

    def _assess_implementation_complexity(
        self, insight_text: str, business_context: Dict[str, Any]
    ) -> str:
        """Assess implementation complexity"""
        complexity_indicators = {
            "high": [
                "transformation",
                "overhaul",
                "strategic",
                "enterprise",
                "organization-wide",
            ],
            "medium": ["process", "system", "department", "team", "workflow"],
            "low": ["individual", "specific", "targeted", "tactical", "simple"],
        }

        text_lower = insight_text.lower()

        for level, indicators in complexity_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                return level

        return "medium"  # Default

    def _update_benchmarks(self, assessment: ValueAssessment) -> None:
        """Update value benchmarks for comparative analysis"""
        for dimension, metric in assessment.value_metrics.items():
            self.value_benchmarks[dimension].append(metric.normalized_score)

            # Keep only recent benchmarks (last 100)
            if len(self.value_benchmarks[dimension]) > 100:
                self.value_benchmarks[dimension] = self.value_benchmarks[dimension][
                    -100:
                ]

    def get_value_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """Get statistical benchmarks for value dimensions"""
        benchmarks = {}

        for dimension, scores in self.value_benchmarks.items():
            if len(scores) >= 5:  # Minimum samples for meaningful stats
                benchmarks[dimension] = {
                    "mean": statistics.mean(scores),
                    "median": statistics.median(scores),
                    "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                    "min": min(scores),
                    "max": max(scores),
                    "sample_count": len(scores),
                }

        return benchmarks

    def get_assessment(self, assessment_id: str) -> Optional[ValueAssessment]:
        """Get value assessment by ID"""
        with self._lock:
            return self.assessments.get(assessment_id)

    def list_assessments(
        self,
        engagement_id: Optional[str] = None,
        min_value_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """List assessments with optional filtering"""
        with self._lock:
            assessments = []

            for assessment in self.assessments.values():
                # Apply filters
                if engagement_id and assessment.engagement_id != engagement_id:
                    continue
                if (
                    min_value_score
                    and assessment.weighted_value_score < min_value_score
                ):
                    continue

                assessments.append(assessment.get_value_summary())

            return sorted(
                assessments, key=lambda a: a["weighted_value_score"], reverse=True
            )

    def _persist_assessment(self, assessment: ValueAssessment) -> None:
        """Persist assessment to storage"""
        if not self.enable_persistence:
            return

        try:
            file_path = self.storage_path / f"assessment_{assessment.id}.json"

            # Convert to serializable format
            data = asdict(assessment)

            # Handle datetime serialization
            data["created_at"] = assessment.created_at.isoformat()
            data["last_updated"] = assessment.last_updated.isoformat()

            # Handle nested datetime fields in metrics
            for dimension_key, metric_data in data["value_metrics"].items():
                metric_data["measured_at"] = (
                    metric_data["measured_at"][:19]
                    if isinstance(metric_data["measured_at"], str)
                    else metric_data["measured_at"].isoformat()
                )

            with open(file_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"❌ Failed to persist assessment {assessment.id}: {e}")

    def _convert_uuids_to_strings(self, obj):
        """Recursively convert UUID objects to strings for JSON serialization"""
        if isinstance(obj, UUID):
            return str(obj)
        elif isinstance(obj, dict):
            return {
                key: self._convert_uuids_to_strings(value) for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._convert_uuids_to_strings(item) for item in obj]
        else:
            return obj

    def _confidence_to_float(self, confidence_value) -> Optional[float]:
        """Convert confidence enum or string to float value for database storage"""
        if confidence_value is None:
            return None

        # If already a float, return as-is
        if isinstance(confidence_value, (int, float)):
            return float(confidence_value)

        # Convert string/enum to float
        confidence_str = str(confidence_value).lower()
        confidence_map = {
            "very_low": 0.1,
            "low": 0.3,
            "medium": 0.5,
            "high": 0.7,
            "very_high": 0.9,
            "uncertain": 0.2,
        }

        return confidence_map.get(confidence_str, 0.5)

    def _store_assessment_in_supabase(self, assessment: ValueAssessment) -> None:
        """Store value assessment in Supabase database"""
        try:
            # Prepare assessment data
            assessment_data = {
                "assessment_id": assessment.id,
                "engagement_id": assessment.engagement_id,
                "insight_text": getattr(assessment, "insight_text", assessment.name),
                "phase": assessment.phase or "unknown",
                "assessment_type": getattr(
                    assessment, "assessment_type", "BUSINESS_VALUE"
                ),
                "business_context": getattr(assessment, "business_context", {}),
                "stakeholder_requirements": getattr(
                    assessment, "stakeholder_requirements", []
                ),
                "market_conditions": getattr(assessment, "market_conditions", {}),
                "competitive_landscape": getattr(
                    assessment, "competitive_landscape", {}
                ),
                "overall_score": assessment.overall_value_score,
                "confidence_level": self._confidence_to_float(
                    assessment.confidence_level
                ),
                "assessment_methodology": "weighted_dimensional_analysis",
                "assessment_metadata": {
                    "created_by": assessment.created_by,
                    "implementation_complexity": assessment.implementation_complexity,
                    "roi_analysis": (
                        asdict(assessment.roi_analysis)
                        if assessment.roi_analysis
                        else None
                    ),
                    "total_metrics": len(assessment.value_metrics),
                },
                "created_at": assessment.created_at.isoformat(),
                "updated_at": assessment.last_updated.isoformat(),
            }

            # Convert UUID objects to strings
            assessment_data = self._convert_uuids_to_strings(assessment_data)

            # Insert value assessment
            result = (
                self.supabase_client.table("value_assessments")
                .insert(assessment_data)
                .execute()
            )

            # Handle UUID/string conversion for assessment_row_id
            if result.data and len(result.data) > 0:
                assessment_row_id = result.data[0]["id"]
            else:
                self.logger.warning("No data returned from value assessment insert")
                return

            # Store individual metrics
            for dimension, metric in assessment.value_metrics.items():
                metric_data = {
                    "value_assessment_id": assessment_row_id,
                    "metric_name": dimension,
                    "metric_type": dimension.upper(),
                    "score": metric.normalized_score,
                    "weight": getattr(
                        metric, "weight", 1.0
                    ),  # Default weight if not present
                    "rationale": metric.rationale,
                    "supporting_evidence": metric.supporting_evidence,
                    "calculation_method": getattr(
                        metric, "calculation_method", "weighted_scoring"
                    ),
                    "confidence_score": metric.confidence,
                    "metric_metadata": {
                        "time_horizon": getattr(metric, "time_horizon", "medium_term"),
                        "value_scale": getattr(metric, "value_scale", "medium"),
                        "benchmarks_used": getattr(metric, "benchmarks", []),
                    },
                    "created_at": metric.measured_at.isoformat(),
                }

                self.supabase_client.table("value_metrics").insert(
                    metric_data
                ).execute()

            self.logger.debug(f"Stored value assessment {assessment.id} in Supabase")

        except Exception as e:
            self.logger.error(
                f"Failed to store value assessment {assessment.id} in Supabase: {e}"
            )
            # Continue without failing - fallback to local storage

    def _enforce_cache_limits(self) -> None:
        """Enforce in-memory cache size limits"""
        if len(self.assessments) > self.max_cache_size:
            # Remove oldest entries
            excess_count = len(self.assessments) - self.max_cache_size
            for _ in range(excess_count):
                oldest_id = next(iter(self.assessments))
                del self.assessments[oldest_id]

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance and usage metrics"""
        with self._lock:
            return {
                "assessments_count": len(self.assessments),
                "dimensions_tracked": len(ValueDimension),
                "benchmarks_available": {
                    dim: len(scores) for dim, scores in self.value_benchmarks.items()
                },
                "cache_utilization": len(self.assessments) / self.max_cache_size,
            }


# Global ValueAssessmentEngine instance
_value_assessment_engine_instance: Optional[ValueAssessmentEngine] = None


def get_value_assessment_engine() -> ValueAssessmentEngine:
    """Get or create global ValueAssessmentEngine instance"""
    global _value_assessment_engine_instance

    if _value_assessment_engine_instance is None:
        _value_assessment_engine_instance = ValueAssessmentEngine()

    return _value_assessment_engine_instance
