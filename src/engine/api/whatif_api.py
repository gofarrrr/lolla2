"""
METIS What-If Scenario API
Day 3 Sprint Implementation - Dedicated Scenario Branching

Implements the dedicated What-If endpoint for cleaner, more intuitive
scenario creation and management experience.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from uuid import UUID, uuid4

try:
    from fastapi import HTTPException, status
    from pydantic import BaseModel, Field, validator

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    BaseModel = object


# ============================================================
# WHAT-IF API DATA MODELS
# ============================================================


class WhatIfParameter(BaseModel):
    """Individual parameter change for What-If scenario"""

    parameter_name: str = Field(..., description="Name of the parameter to change")
    original_value: Any = Field(..., description="Original value from base scenario")
    new_value: Any = Field(..., description="New value for What-If scenario")
    parameter_type: str = Field(
        ..., description="Type: assumption, constraint, goal, context"
    )
    impact_expected: str = Field(
        default="moderate", description="Expected impact: low, moderate, high"
    )
    reasoning: str = Field(
        default="", description="Why this parameter is being changed"
    )


class WhatIfRequest(BaseModel):
    """Request model for creating What-If scenarios"""

    base_engagement_id: str = Field(..., description="Base engagement to branch from")
    scenario_name: str = Field(
        ..., min_length=3, max_length=100, description="Name for this What-If scenario"
    )
    scenario_description: str = Field(
        default="", description="Optional description of the scenario"
    )

    # Parameter changes
    parameter_changes: List[WhatIfParameter] = Field(
        ..., min_items=1, description="Parameters to modify"
    )

    # Execution preferences
    execution_mode: str = Field(
        default="full_reanalysis",
        description="full_reanalysis, delta_analysis, or preview_only",
    )
    preserve_mental_models: bool = Field(
        default=True, description="Keep same mental models as base scenario"
    )
    include_base_comparison: bool = Field(
        default=True, description="Include comparison with base scenario"
    )

    # Context
    created_by_rationale: str = Field(
        default="", description="Why this What-If scenario is being created"
    )
    stakeholder_focus: List[str] = Field(
        default_factory=list, description="Which stakeholders care about this scenario"
    )


class WhatIfResponse(BaseModel):
    """Response model for What-If scenario creation"""

    scenario_id: str = Field(..., description="New scenario engagement ID")
    base_engagement_id: str = Field(..., description="Original base engagement ID")
    scenario_name: str = Field(..., description="Name of the What-If scenario")

    # Lineage tracking
    scenario_lineage: Dict[str, Any] = Field(
        default_factory=dict, description="Relationship to base and other scenarios"
    )
    parameter_changes_applied: List[WhatIfParameter] = Field(
        ..., description="Parameters that were changed"
    )

    # Results preview (if available)
    preview_insights: List[str] = Field(
        default_factory=list, description="Quick preview insights"
    )
    estimated_impact: Dict[str, Any] = Field(
        default_factory=dict, description="Estimated impact of changes"
    )

    # Execution details
    execution_status: str = Field(
        default="processing", description="processing, completed, failed"
    )
    processing_time_estimate_ms: int = Field(
        default=120000, description="Estimated processing time"
    )
    actual_processing_time_ms: Optional[int] = Field(
        None, description="Actual processing time if completed"
    )

    # Comparison readiness
    comparison_ready: bool = Field(
        default=False, description="Whether scenario is ready for comparison"
    )
    base_comparison_url: str = Field(
        default="", description="URL for comparing with base scenario"
    )

    # Metadata
    created_at: str = Field(..., description="Scenario creation timestamp")
    created_by_user: str = Field(
        default="", description="User who created this scenario"
    )


class ScenarioMetadata(BaseModel):
    """Metadata about scenario relationships and history"""

    scenario_tree: Dict[str, Any] = Field(
        default_factory=dict, description="Full scenario branching tree"
    )
    related_scenarios: List[str] = Field(
        default_factory=list, description="Related scenario IDs"
    )
    scenario_generation: int = Field(
        default=1, description="How many branches from original"
    )
    total_variations: int = Field(
        default=1, description="Total scenarios in this family"
    )


class WhatIfBatchRequest(BaseModel):
    """Request model for creating multiple What-If scenarios at once"""

    base_engagement_id: str = Field(..., description="Base engagement to branch from")
    scenario_batch_name: str = Field(
        ..., description="Name for this batch of scenarios"
    )

    scenarios: List[Dict[str, Any]] = Field(
        ..., min_items=1, max_items=10, description="Multiple scenarios to create"
    )
    parallel_execution: bool = Field(
        default=True, description="Execute scenarios in parallel"
    )
    compare_all: bool = Field(
        default=True, description="Create comparison matrix when done"
    )


class WhatIfBatchResponse(BaseModel):
    """Response model for batch What-If creation"""

    batch_id: str = Field(..., description="Batch execution ID")
    base_engagement_id: str = Field(..., description="Original base engagement ID")
    scenario_ids: List[str] = Field(..., description="Created scenario engagement IDs")

    batch_status: str = Field(
        default="processing", description="Batch processing status"
    )
    completed_scenarios: int = Field(
        default=0, description="Number of completed scenarios"
    )
    failed_scenarios: int = Field(default=0, description="Number of failed scenarios")

    comparison_matrix_id: Optional[str] = Field(
        None, description="Comparison matrix ID if requested"
    )
    estimated_completion_time: str = Field(
        ..., description="Estimated completion timestamp"
    )


# ============================================================
# WHAT-IF SCENARIO ENGINE
# ============================================================


class WhatIfScenarioEngine:
    """Engine for creating and managing What-If scenarios"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def create_whatif_scenario(
        self,
        base_engagement_id: UUID,
        scenario_name: str,
        scenario_description: str,
        parameter_changes: List[WhatIfParameter],
        execution_mode: str,
        preserve_mental_models: bool,
        include_base_comparison: bool,
        created_by_rationale: str,
        stakeholder_focus: List[str],
        user_id: Optional[UUID] = None,
    ) -> WhatIfResponse:
        """Create a new What-If scenario by branching from base engagement"""

        scenario_id = uuid4()
        start_time = datetime.utcnow()

        self.logger.info(
            f"🔀 Creating What-If scenario '{scenario_name}' from base {base_engagement_id}"
        )
        self.logger.info(f"   Changing {len(parameter_changes)} parameters")
        self.logger.info(f"   Execution mode: {execution_mode}")

        try:
            # Load base engagement
            base_engagement = await self._load_base_engagement(base_engagement_id)

            # Create new engagement with modified parameters
            new_engagement = await self._create_branched_engagement(
                base_engagement,
                scenario_id,
                scenario_name,
                scenario_description,
                parameter_changes,
                preserve_mental_models,
            )

            # Execute based on mode
            execution_result = await self._execute_scenario(
                new_engagement, execution_mode, parameter_changes
            )

            # Create lineage tracking
            lineage = await self._create_scenario_lineage(
                base_engagement_id, scenario_id, parameter_changes
            )

            # Generate preview insights if requested
            preview_insights = []
            estimated_impact = {}

            if execution_mode == "preview_only":
                preview_insights = await self._generate_preview_insights(
                    parameter_changes, base_engagement
                )
                estimated_impact = await self._estimate_parameter_impact(
                    parameter_changes, base_engagement
                )

            # Prepare comparison URL
            base_comparison_url = (
                f"/api/engagements/compare?ids={base_engagement_id},{scenario_id}"
                if include_base_comparison
                else ""
            )

            processing_time = int(
                (datetime.utcnow() - start_time).total_seconds() * 1000
            )

            return WhatIfResponse(
                scenario_id=str(scenario_id),
                base_engagement_id=str(base_engagement_id),
                scenario_name=scenario_name,
                scenario_lineage=lineage,
                parameter_changes_applied=parameter_changes,
                preview_insights=preview_insights,
                estimated_impact=estimated_impact,
                execution_status=execution_result.get("status", "processing"),
                processing_time_estimate_ms=execution_result.get(
                    "estimated_time_ms", 120000
                ),
                actual_processing_time_ms=(
                    processing_time
                    if execution_result.get("status") == "completed"
                    else None
                ),
                comparison_ready=execution_result.get("status") == "completed",
                base_comparison_url=base_comparison_url,
                created_at=start_time.isoformat(),
                created_by_user=str(user_id) if user_id else "",
            )

        except Exception as e:
            self.logger.error(f"❌ What-If scenario creation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"What-If scenario creation failed: {str(e)}",
            )

    async def create_whatif_batch(
        self,
        base_engagement_id: UUID,
        batch_name: str,
        scenarios: List[Dict[str, Any]],
        parallel_execution: bool,
        compare_all: bool,
        user_id: Optional[UUID] = None,
    ) -> WhatIfBatchResponse:
        """Create multiple What-If scenarios in batch"""

        batch_id = str(uuid4())
        start_time = datetime.utcnow()

        self.logger.info(
            f"📦 Creating What-If batch '{batch_name}' with {len(scenarios)} scenarios"
        )

        try:
            scenario_ids = []
            completed_count = 0
            failed_count = 0

            if parallel_execution:
                # Execute scenarios in parallel
                tasks = []
                for i, scenario_config in enumerate(scenarios):
                    task = self._create_single_scenario_from_config(
                        base_engagement_id,
                        f"{batch_name}_scenario_{i+1}",
                        scenario_config,
                        user_id,
                    )
                    tasks.append(task)

                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, Exception):
                        failed_count += 1
                        self.logger.error(f"Batch scenario failed: {result}")
                    else:
                        scenario_ids.append(result["scenario_id"])
                        completed_count += 1
            else:
                # Execute scenarios sequentially
                for i, scenario_config in enumerate(scenarios):
                    try:
                        result = await self._create_single_scenario_from_config(
                            base_engagement_id,
                            f"{batch_name}_scenario_{i+1}",
                            scenario_config,
                            user_id,
                        )
                        scenario_ids.append(result["scenario_id"])
                        completed_count += 1
                    except Exception as e:
                        failed_count += 1
                        self.logger.error(f"Batch scenario {i+1} failed: {e}")

            # Create comparison matrix if requested
            comparison_matrix_id = None
            if compare_all and len(scenario_ids) >= 2:
                comparison_matrix_id = await self._create_comparison_matrix(
                    base_engagement_id, scenario_ids, batch_name
                )

            # Calculate estimated completion time
            remaining_time_estimate = max(
                0, (len(scenarios) - completed_count) * 120000
            )  # 2 minutes per scenario
            estimated_completion = start_time + timedelta(
                milliseconds=remaining_time_estimate
            )

            return WhatIfBatchResponse(
                batch_id=batch_id,
                base_engagement_id=str(base_engagement_id),
                scenario_ids=scenario_ids,
                batch_status="completed" if failed_count == 0 else "partial_failure",
                completed_scenarios=completed_count,
                failed_scenarios=failed_count,
                comparison_matrix_id=comparison_matrix_id,
                estimated_completion_time=estimated_completion.isoformat(),
            )

        except Exception as e:
            self.logger.error(f"❌ What-If batch creation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"What-If batch creation failed: {str(e)}",
            )

    # Helper methods with mock implementations

    async def _load_base_engagement(self, engagement_id: UUID) -> Dict[str, Any]:
        """Load base engagement data (mock implementation)"""
        return {
            "engagement_id": str(engagement_id),
            "problem_statement": "Mock base engagement",
            "cognitive_state": {
                "selected_mental_models": ["porter_5_forces", "swot_analysis"],
                "confidence_scores": {"overall": 0.85},
            },
            "workflow_state": {"current_phase": "completed"},
            "business_context": {"industry": "technology", "market_size": "large"},
            "assumptions": {
                "market_growth": 0.15,
                "competition_intensity": "high",
                "regulatory_stability": "stable",
            },
        }

    async def _create_branched_engagement(
        self,
        base_engagement: Dict[str, Any],
        scenario_id: UUID,
        scenario_name: str,
        scenario_description: str,
        parameter_changes: List[WhatIfParameter],
        preserve_mental_models: bool,
    ) -> Dict[str, Any]:
        """Create new engagement with modified parameters"""

        new_engagement = base_engagement.copy()
        new_engagement["engagement_id"] = str(scenario_id)
        new_engagement["scenario_name"] = scenario_name
        new_engagement["scenario_description"] = scenario_description
        new_engagement["parent_engagement_id"] = base_engagement["engagement_id"]

        # Apply parameter changes
        for change in parameter_changes:
            # Mock parameter application
            if change.parameter_name in new_engagement.get("assumptions", {}):
                new_engagement["assumptions"][change.parameter_name] = change.new_value
            elif change.parameter_name == "problem_statement":
                new_engagement["problem_statement"] = change.new_value
            else:
                # Add to business context or create new field
                if "whatif_parameters" not in new_engagement:
                    new_engagement["whatif_parameters"] = {}
                new_engagement["whatif_parameters"][
                    change.parameter_name
                ] = change.new_value

        # Handle mental model preservation
        if not preserve_mental_models:
            new_engagement["cognitive_state"]["selected_mental_models"] = []

        return new_engagement

    async def _execute_scenario(
        self,
        engagement: Dict[str, Any],
        execution_mode: str,
        parameter_changes: List[WhatIfParameter],
    ) -> Dict[str, Any]:
        """Execute the What-If scenario based on mode"""

        if execution_mode == "preview_only":
            return {"status": "preview", "estimated_time_ms": 0}
        elif execution_mode == "delta_analysis":
            return {
                "status": "processing",
                "estimated_time_ms": 60000,  # 1 minute for delta
            }
        else:  # full_reanalysis
            return {
                "status": "processing",
                "estimated_time_ms": 120000,  # 2 minutes for full analysis
            }

    async def _create_scenario_lineage(
        self,
        base_engagement_id: UUID,
        scenario_id: UUID,
        parameter_changes: List[WhatIfParameter],
    ) -> Dict[str, Any]:
        """Create lineage tracking for scenario relationships"""

        return {
            "parent_scenario_id": str(base_engagement_id),
            "scenario_type": "whatif_branch",
            "branch_point": datetime.utcnow().isoformat(),
            "parameter_changes_count": len(parameter_changes),
            "change_categories": list(
                set(change.parameter_type for change in parameter_changes)
            ),
            "impact_levels": list(
                set(change.impact_expected for change in parameter_changes)
            ),
        }

    async def _generate_preview_insights(
        self, parameter_changes: List[WhatIfParameter], base_engagement: Dict[str, Any]
    ) -> List[str]:
        """Generate quick preview insights without full analysis"""

        insights = []

        # Analyze parameter changes for quick insights
        for change in parameter_changes:
            if change.impact_expected == "high":
                insights.append(
                    f"High impact expected from changing {change.parameter_name}"
                )

            if change.parameter_type == "assumption":
                insights.append(
                    "Assumption change may affect strategic recommendations"
                )

        insights.append("Full analysis required for detailed insights")

        return insights[:3]  # Limit to top 3 insights

    async def _estimate_parameter_impact(
        self, parameter_changes: List[WhatIfParameter], base_engagement: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Estimate impact of parameter changes"""

        high_impact_changes = sum(
            1 for change in parameter_changes if change.impact_expected == "high"
        )

        return {
            "confidence_change_estimate": high_impact_changes
            * -0.05,  # Each high impact change reduces confidence
            "processing_complexity_increase": len(parameter_changes) * 0.1,
            "strategic_shift_probability": 0.3 if high_impact_changes > 0 else 0.1,
            "recommendation_changes_expected": max(1, high_impact_changes),
        }

    async def _create_single_scenario_from_config(
        self,
        base_engagement_id: UUID,
        scenario_name: str,
        config: Dict[str, Any],
        user_id: Optional[UUID],
    ) -> Dict[str, str]:
        """Create a single scenario from configuration (for batch processing)"""

        # Mock implementation - would parse config and create scenario
        scenario_id = str(uuid4())

        return {"scenario_id": scenario_id, "status": "created"}

    async def _create_comparison_matrix(
        self, base_engagement_id: UUID, scenario_ids: List[str], batch_name: str
    ) -> str:
        """Create comparison matrix for batch scenarios"""

        matrix_id = str(uuid4())

        # Mock implementation - would create comprehensive comparison
        self.logger.info(
            f"📊 Created comparison matrix {matrix_id} for {len(scenario_ids)} scenarios"
        )

        return matrix_id


# ============================================================
# GLOBAL INSTANCE
# ============================================================

_whatif_engine: Optional[WhatIfScenarioEngine] = None


def get_whatif_engine() -> WhatIfScenarioEngine:
    """Get global What-If engine instance"""
    global _whatif_engine
    if _whatif_engine is None:
        _whatif_engine = WhatIfScenarioEngine()
    return _whatif_engine
