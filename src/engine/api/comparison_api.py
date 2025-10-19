"""
METIS Engagement Comparison API
Day 2 Sprint Implementation - Critical What-If Experience Enablement

Implements the missing comparison endpoints identified in the audit:
- GET /api/engagements/compare - Side-by-side scenario comparison
- POST /api/engagements/{id}/override-models - Power user model selection
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import UUID, uuid4

try:
    from fastapi import HTTPException, Depends, status
    from pydantic import BaseModel, Field, validator

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    BaseModel = object


# ============================================================
# COMPARISON API DATA MODELS
# ============================================================


class ScenarioComparison(BaseModel):
    """Detailed comparison between two engagement scenarios"""

    scenario_a: Dict[str, Any] = Field(..., description="First scenario data")
    scenario_b: Dict[str, Any] = Field(..., description="Second scenario data")

    # Key differences analysis
    parameter_differences: Dict[str, Any] = Field(default_factory=dict)
    outcome_differences: Dict[str, float] = Field(default_factory=dict)
    confidence_delta: float = Field(
        default=0.0, description="Confidence difference between scenarios"
    )

    # Strategic insights
    strategic_implications: List[str] = Field(default_factory=list)
    risk_assessment_comparison: Dict[str, Any] = Field(default_factory=dict)
    recommendation_differences: List[str] = Field(default_factory=list)


class DifferenceAnalysis(BaseModel):
    """Deep analysis of differences between scenarios"""

    critical_decision_points: List[Dict[str, Any]] = Field(default_factory=list)
    assumption_variations: List[Dict[str, Any]] = Field(default_factory=list)
    mental_model_differences: List[str] = Field(default_factory=list)

    # Impact analysis
    business_impact_variance: Dict[str, float] = Field(default_factory=dict)
    implementation_complexity_delta: float = Field(default=0.0)
    risk_profile_changes: List[str] = Field(default_factory=list)

    # Recommendations
    preferred_scenario: str = Field(
        default="", description="Which scenario is recommended"
    )
    decision_rationale: str = Field(
        default="", description="Why this scenario is preferred"
    )
    hybrid_opportunities: List[str] = Field(default_factory=list)


class ComparisonRequest(BaseModel):
    """Request model for scenario comparison"""

    engagement_ids: List[str] = Field(
        ..., min_items=2, max_items=5, description="Engagement IDs to compare"
    )
    comparison_dimensions: List[str] = Field(
        default=[
            "strategic_impact",
            "implementation_complexity",
            "risk_profile",
            "roi_potential",
        ],
        description="Dimensions to compare across scenarios",
    )
    analysis_depth: str = Field(
        default="standard", description="Analysis depth: quick, standard, comprehensive"
    )
    include_recommendations: bool = Field(
        default=True, description="Include strategic recommendations"
    )


class ComparisonResponse(BaseModel):
    """Response model for scenario comparison"""

    comparison_id: str = Field(..., description="Unique comparison identifier")
    scenarios: List[ScenarioComparison] = Field(
        ..., description="Detailed scenario comparisons"
    )
    differences: DifferenceAnalysis = Field(..., description="Deep difference analysis")
    recommendations: List[str] = Field(
        default_factory=list, description="Strategic recommendations"
    )

    # Metadata
    compared_at: str = Field(..., description="Comparison timestamp")
    comparison_quality_score: float = Field(
        default=0.0, description="Quality of the comparison analysis"
    )
    processing_time_ms: int = Field(default=0, description="Time taken for comparison")


class ModelOverrideRequest(BaseModel):
    """Request model for manual model override"""

    forced_models: List[str] = Field(
        ..., min_items=1, max_items=5, description="Mental models to force"
    )
    rationale: str = Field(
        ..., min_length=10, description="Expert rationale for override"
    )
    override_scope: str = Field(
        default="full_analysis", description="full_analysis or specific_phase"
    )
    expert_confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Expert confidence in override"
    )

    # Learning capture for flywheel
    expected_improvement: str = Field(
        default="", description="What improvement is expected from this override"
    )
    success_criteria: List[str] = Field(
        default_factory=list, description="How to measure override success"
    )


class ModelOverrideResponse(BaseModel):
    """Response model for model override"""

    override_id: str = Field(..., description="Unique override identifier")
    engagement_id: str = Field(..., description="Target engagement ID")
    status: str = Field(default="applied", description="Override status")

    # Override details
    original_models: List[str] = Field(
        default_factory=list, description="Originally selected models"
    )
    overridden_models: List[str] = Field(
        default_factory=list, description="Expert-forced models"
    )
    override_rationale: str = Field(..., description="Expert rationale")

    # System response
    estimated_impact: Dict[str, Any] = Field(
        default_factory=dict, description="Expected impact of override"
    )
    reanalysis_triggered: bool = Field(
        default=False, description="Whether re-analysis was triggered"
    )
    learning_captured: bool = Field(
        default=False, description="Whether override data was captured for learning"
    )


# ============================================================
# COMPARISON ENGINE
# ============================================================


class EngagementComparisonEngine:
    """Engine for comparing multiple engagement scenarios"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def compare_engagements(
        self,
        engagement_ids: List[UUID],
        comparison_dimensions: List[str],
        analysis_depth: str = "standard",
    ) -> ComparisonResponse:
        """Compare multiple engagement scenarios"""

        start_time = datetime.utcnow()
        comparison_id = str(uuid4())

        self.logger.info(
            f"🔍 Starting comparison {comparison_id} for {len(engagement_ids)} scenarios"
        )

        try:
            # Load engagement data (mock implementation - replace with actual database calls)
            scenarios_data = await self._load_scenarios(engagement_ids)

            # Perform comparison analysis
            scenarios = await self._analyze_scenarios(
                scenarios_data, comparison_dimensions, analysis_depth
            )
            differences = await self._analyze_differences(
                scenarios_data, comparison_dimensions
            )
            recommendations = await self._generate_recommendations(
                scenarios, differences
            )

            processing_time = int(
                (datetime.utcnow() - start_time).total_seconds() * 1000
            )

            return ComparisonResponse(
                comparison_id=comparison_id,
                scenarios=scenarios,
                differences=differences,
                recommendations=recommendations,
                compared_at=start_time.isoformat(),
                comparison_quality_score=0.85,  # Mock score - implement quality metrics
                processing_time_ms=processing_time,
            )

        except Exception as e:
            self.logger.error(f"❌ Comparison failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Scenario comparison failed: {str(e)}",
            )

    async def _load_scenarios(self, engagement_ids: List[UUID]) -> List[Dict[str, Any]]:
        """Load engagement data for comparison (mock implementation)"""

        scenarios = []
        for eng_id in engagement_ids:
            # Mock scenario data - replace with actual database calls
            scenario = {
                "engagement_id": str(eng_id),
                "problem_statement": f"Mock problem for {eng_id}",
                "cognitive_state": {
                    "selected_mental_models": ["porter_5_forces", "swot_analysis"],
                    "confidence_scores": {"overall": 0.85, "strategic": 0.78},
                    "reasoning_steps": [],
                },
                "workflow_state": {
                    "current_phase": "synthesis_delivery",
                    "completed_phases": [
                        "problem_structuring",
                        "hypothesis_generation",
                        "analysis_execution",
                    ],
                },
                "deliverables": {
                    "strategic_recommendations": [
                        "Expand market presence",
                        "Optimize operations",
                    ],
                    "risk_assessment": {
                        "high_risk": [],
                        "medium_risk": ["market_volatility"],
                    },
                    "implementation_complexity": 0.65,
                },
                "metadata": {
                    "processing_time_ms": 125000,
                    "research_sources": 15,
                    "validation_score": 0.88,
                },
            }
            scenarios.append(scenario)

        return scenarios

    async def _analyze_scenarios(
        self, scenarios_data: List[Dict[str, Any]], dimensions: List[str], depth: str
    ) -> List[ScenarioComparison]:
        """Analyze scenarios for comparison"""

        comparisons = []

        # Create pairwise comparisons
        for i, scenario_a in enumerate(scenarios_data):
            for j, scenario_b in enumerate(scenarios_data[i + 1 :], i + 1):

                comparison = ScenarioComparison(
                    scenario_a=scenario_a,
                    scenario_b=scenario_b,
                    parameter_differences=await self._calculate_parameter_differences(
                        scenario_a, scenario_b
                    ),
                    outcome_differences=await self._calculate_outcome_differences(
                        scenario_a, scenario_b
                    ),
                    confidence_delta=self._calculate_confidence_delta(
                        scenario_a, scenario_b
                    ),
                    strategic_implications=await self._analyze_strategic_implications(
                        scenario_a, scenario_b, dimensions
                    ),
                    risk_assessment_comparison=await self._compare_risk_assessments(
                        scenario_a, scenario_b
                    ),
                    recommendation_differences=await self._compare_recommendations(
                        scenario_a, scenario_b
                    ),
                )

                comparisons.append(comparison)

        return comparisons

    async def _analyze_differences(
        self, scenarios_data: List[Dict[str, Any]], dimensions: List[str]
    ) -> DifferenceAnalysis:
        """Perform deep difference analysis"""

        # Analyze critical decision points
        decision_points = []
        for i, scenario in enumerate(scenarios_data):
            points = await self._extract_decision_points(scenario)
            decision_points.extend([{**point, "scenario_index": i} for point in points])

        # Find assumption variations
        assumption_variations = await self._find_assumption_variations(scenarios_data)

        # Identify mental model differences
        mental_model_differences = await self._identify_model_differences(
            scenarios_data
        )

        # Calculate business impact variance
        impact_variance = await self._calculate_impact_variance(scenarios_data)

        # Generate recommendations
        preferred_scenario, rationale = await self._determine_preferred_scenario(
            scenarios_data, dimensions
        )
        hybrid_opportunities = await self._identify_hybrid_opportunities(scenarios_data)

        return DifferenceAnalysis(
            critical_decision_points=decision_points,
            assumption_variations=assumption_variations,
            mental_model_differences=mental_model_differences,
            business_impact_variance=impact_variance,
            implementation_complexity_delta=0.15,  # Mock calculation
            risk_profile_changes=[
                "Market entry timing",
                "Resource allocation strategy",
            ],
            preferred_scenario=preferred_scenario,
            decision_rationale=rationale,
            hybrid_opportunities=hybrid_opportunities,
        )

    async def _generate_recommendations(
        self, scenarios: List[ScenarioComparison], differences: DifferenceAnalysis
    ) -> List[str]:
        """Generate strategic recommendations based on comparison"""

        recommendations = [
            f"Recommend proceeding with {differences.preferred_scenario} based on {differences.decision_rationale}",
            "Consider implementing risk mitigation strategies for identified medium-risk factors",
            "Monitor market conditions closely to validate scenario assumptions",
            "Establish success metrics aligned with the chosen scenario's outcomes",
        ]

        # Add hybrid opportunities if available
        if differences.hybrid_opportunities:
            recommendations.append(
                f"Explore hybrid approach incorporating: {', '.join(differences.hybrid_opportunities)}"
            )

        return recommendations

    # Helper methods with mock implementations
    async def _calculate_parameter_differences(
        self, a: Dict, b: Dict
    ) -> Dict[str, Any]:
        return {
            "processing_time_delta": 15000,
            "model_selection_differences": ["porter_5_forces"],
        }

    async def _calculate_outcome_differences(
        self, a: Dict, b: Dict
    ) -> Dict[str, float]:
        return {"confidence_delta": 0.05, "complexity_delta": 0.10}

    def _calculate_confidence_delta(self, a: Dict, b: Dict) -> float:
        conf_a = (
            a.get("cognitive_state", {})
            .get("confidence_scores", {})
            .get("overall", 0.5)
        )
        conf_b = (
            b.get("cognitive_state", {})
            .get("confidence_scores", {})
            .get("overall", 0.5)
        )
        return abs(conf_a - conf_b)

    async def _analyze_strategic_implications(
        self, a: Dict, b: Dict, dimensions: List[str]
    ) -> List[str]:
        return [
            "Market positioning varies significantly",
            "Implementation timeline differs by 3 months",
        ]

    async def _compare_risk_assessments(self, a: Dict, b: Dict) -> Dict[str, Any]:
        return {
            "risk_level_change": "moderate",
            "new_risks_identified": ["regulatory_compliance"],
        }

    async def _compare_recommendations(self, a: Dict, b: Dict) -> List[str]:
        return [
            "Strategic focus differs: growth vs. optimization",
            "Resource allocation priorities vary",
        ]

    async def _extract_decision_points(self, scenario: Dict) -> List[Dict[str, Any]]:
        return [
            {"decision": "market_entry_strategy", "confidence": 0.85, "impact": "high"}
        ]

    async def _find_assumption_variations(
        self, scenarios: List[Dict]
    ) -> List[Dict[str, Any]]:
        return [
            {
                "assumption": "market_growth_rate",
                "variation_range": "3-7%",
                "impact": "medium",
            }
        ]

    async def _identify_model_differences(self, scenarios: List[Dict]) -> List[str]:
        return ["porter_5_forces vs swot_analysis", "bcg_matrix inclusion variance"]

    async def _calculate_impact_variance(
        self, scenarios: List[Dict]
    ) -> Dict[str, float]:
        return {
            "revenue_impact_range": 0.25,
            "cost_variance": 0.15,
            "risk_variance": 0.10,
        }

    async def _determine_preferred_scenario(
        self, scenarios: List[Dict], dimensions: List[str]
    ) -> tuple[str, str]:
        return ("Scenario 1", "Higher confidence score and lower implementation risk")

    async def _identify_hybrid_opportunities(self, scenarios: List[Dict]) -> List[str]:
        return [
            "Combine market analysis approaches",
            "Merge risk mitigation strategies",
        ]


# ============================================================
# MODEL OVERRIDE ENGINE
# ============================================================


class ModelOverrideEngine:
    """Engine for handling expert model overrides"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def apply_model_override(
        self,
        engagement_id: UUID,
        forced_models: List[str],
        rationale: str,
        override_scope: str,
        expert_confidence: float,
        expected_improvement: str,
        success_criteria: List[str],
    ) -> ModelOverrideResponse:
        """Apply expert model override to engagement"""

        override_id = str(uuid4())

        self.logger.info(
            f"🎯 Applying model override {override_id} to engagement {engagement_id}"
        )
        self.logger.info(f"   Models: {forced_models}")
        self.logger.info(f"   Rationale: {rationale[:100]}...")

        try:
            # Get current engagement state (mock implementation)
            current_state = await self._get_engagement_state(engagement_id)
            original_models = current_state.get("selected_models", [])

            # Validate forced models exist
            await self._validate_models(forced_models)

            # Apply override
            override_result = await self._execute_override(
                engagement_id, forced_models, override_scope, expert_confidence
            )

            # Capture learning data for flywheel
            await self._capture_override_learning(
                override_id,
                engagement_id,
                original_models,
                forced_models,
                rationale,
                expected_improvement,
                success_criteria,
                expert_confidence,
            )

            # Estimate impact
            estimated_impact = await self._estimate_override_impact(
                original_models, forced_models, current_state
            )

            return ModelOverrideResponse(
                override_id=override_id,
                engagement_id=str(engagement_id),
                status="applied",
                original_models=original_models,
                overridden_models=forced_models,
                override_rationale=rationale,
                estimated_impact=estimated_impact,
                reanalysis_triggered=override_result.get("reanalysis_triggered", False),
                learning_captured=True,
            )

        except Exception as e:
            self.logger.error(f"❌ Model override failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Model override failed: {str(e)}",
            )

    async def _get_engagement_state(self, engagement_id: UUID) -> Dict[str, Any]:
        """Get current engagement state (mock implementation)"""
        return {
            "selected_models": ["porter_5_forces", "swot_analysis"],
            "current_phase": "analysis_execution",
            "confidence_scores": {"overall": 0.78},
        }

    async def _validate_models(self, models: List[str]) -> None:
        """Validate that forced models exist in the system"""
        # Mock validation - implement actual model catalog validation
        valid_models = [
            "porter_5_forces",
            "swot_analysis",
            "bcg_matrix",
            "ansoff_matrix",
            "value_chain_analysis",
        ]
        invalid_models = [m for m in models if m not in valid_models]

        if invalid_models:
            raise ValueError(f"Invalid models specified: {invalid_models}")

    async def _execute_override(
        self,
        engagement_id: UUID,
        forced_models: List[str],
        scope: str,
        confidence: float,
    ) -> Dict[str, Any]:
        """Execute the model override (mock implementation)"""

        # Mock override execution
        self.logger.info(f"✅ Override executed for engagement {engagement_id}")

        return {
            "reanalysis_triggered": scope == "full_analysis",
            "models_updated": forced_models,
            "confidence_adjustment": confidence - 0.5,  # Mock adjustment
        }

    async def _capture_override_learning(
        self,
        override_id: str,
        engagement_id: UUID,
        original_models: List[str],
        forced_models: List[str],
        rationale: str,
        expected_improvement: str,
        success_criteria: List[str],
        expert_confidence: float,
    ) -> None:
        """Capture override data for flywheel learning"""

        # This would store data for future learning about override effectiveness
        learning_data = {
            "override_id": override_id,
            "engagement_id": str(engagement_id),
            "original_models": original_models,
            "forced_models": forced_models,
            "expert_rationale": rationale,
            "expected_improvement": expected_improvement,
            "success_criteria": success_criteria,
            "expert_confidence": expert_confidence,
            "override_timestamp": datetime.utcnow().isoformat(),
        }

        # Mock storage - implement actual database storage
        self.logger.info("📊 Override learning data captured for flywheel improvement")

    async def _estimate_override_impact(
        self,
        original_models: List[str],
        forced_models: List[str],
        current_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Estimate the impact of the model override"""

        return {
            "confidence_change_estimate": 0.05,  # Mock estimate
            "processing_time_change_estimate": -2000,  # Mock: faster processing
            "analysis_depth_change": "increased",
            "risk_profile_impact": "moderate_change",
            "strategic_focus_shift": "more_comprehensive",
        }


# ============================================================
# GLOBAL INSTANCES
# ============================================================

_comparison_engine: Optional[EngagementComparisonEngine] = None
_override_engine: Optional[ModelOverrideEngine] = None


def get_comparison_engine() -> EngagementComparisonEngine:
    """Get global comparison engine instance"""
    global _comparison_engine
    if _comparison_engine is None:
        _comparison_engine = EngagementComparisonEngine()
    return _comparison_engine


def get_override_engine() -> ModelOverrideEngine:
    """Get global override engine instance"""
    global _override_engine
    if _override_engine is None:
        _override_engine = ModelOverrideEngine()
    return _override_engine
