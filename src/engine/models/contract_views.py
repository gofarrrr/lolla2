"""
METIS Contract View Patterns
Sprint: Clarity & Consolidation (Week 2, Day 9-10)
Purpose: Break the God Object with focused, role-specific data views

This module provides lightweight, focused views of the MetisDataContract
to reduce coupling and improve component interfaces.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict

from src.engine.models.data_contracts import (
    MetisDataContract,
    EngagementPhase,
    MentalModelDefinition,
    ReasoningStep,
    ClarificationSession,
    ResearchIntelligence,
    ConfidenceLevel,
)


# ============================================================================
# FOCUSED VIEW MODELS
# ============================================================================


class ProblemAnalysisView(BaseModel):
    """Minimal view for problem analysis phase"""

    engagement_id: UUID
    problem_statement: str
    business_context: Dict[str, Any]
    stakeholders: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)
    constraints: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class HypothesisGenerationView(BaseModel):
    """Minimal view for hypothesis generation"""

    engagement_id: UUID
    problem_statement: str
    business_context: Dict[str, Any]
    selected_mental_models: List[str]
    problem_classification: str = "general_analysis"
    industry: str = "general"

    model_config = ConfigDict(arbitrary_types_allowed=True)


class AnalysisExecutionView(BaseModel):
    """Minimal view for analysis execution"""

    engagement_id: UUID
    hypotheses: List[Dict[str, Any]]
    mental_models: List[MentalModelDefinition]
    reasoning_steps: List[ReasoningStep]
    confidence_scores: Dict[str, float]
    research_enabled: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SynthesisInputView(BaseModel):
    """Minimal view for synthesis phase"""

    engagement_id: UUID
    problem_statement: str
    reasoning_steps: List[ReasoningStep]
    confidence_scores: Dict[str, float]
    key_insights: List[str]
    recommendations: List[Dict[str, Any]]
    evidence_base: Dict[str, List[str]]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ResearchContextView(BaseModel):
    """Minimal view for research operations"""

    engagement_id: UUID
    query: str
    business_context: Dict[str, Any]
    research_intelligence: Optional[ResearchIntelligence]
    max_sources: int = 5
    confidence_threshold: float = 0.7

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ValidationContextView(BaseModel):
    """Minimal view for validation operations"""

    engagement_id: UUID
    content_to_validate: Any
    validation_type: str
    confidence_threshold: float
    hallucination_checks: List[Dict[str, Any]] = Field(default_factory=list)
    bias_detection_enabled: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DeliverableGenerationView(BaseModel):
    """Minimal view for deliverable generation"""

    engagement_id: UUID
    client_name: str
    problem_statement: str
    executive_summary: str
    key_findings: List[str]
    recommendations: List[Dict[str, Any]]
    confidence_level: ConfidenceLevel
    supporting_evidence: List[str]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ClarificationContextView(BaseModel):
    """Minimal view for HITL clarification"""

    engagement_id: UUID
    original_query: str
    clarification_session: Optional[ClarificationSession]
    required_dimensions: List[str]
    user_preferences: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration"""

        arbitrary_types_allowed = True


class MonitoringMetricsView(BaseModel):
    """Minimal view for monitoring and metrics"""

    engagement_id: UUID
    current_phase: EngagementPhase
    completed_phases: List[EngagementPhase]
    performance_metrics: Dict[str, float]
    error_count: int = 0
    retry_counts: Dict[str, int] = Field(default_factory=dict)
    execution_time_seconds: float = 0.0

    class Config:
        """Pydantic configuration"""

        arbitrary_types_allowed = True


class CostTrackingView(BaseModel):
    """Minimal view for cost tracking"""

    engagement_id: UUID
    llm_tokens_used: int = 0
    api_calls_made: int = 0
    estimated_cost_usd: float = 0.0
    resource_usage: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration"""

        arbitrary_types_allowed = True


# ============================================================================
# CONTRACT VIEW FACTORY
# ============================================================================


class ContractViewFactory:
    """
    Factory for creating focused views from the MetisDataContract.

    This factory extracts only the necessary data for each component,
    reducing coupling and making interfaces explicit.
    """

    @staticmethod
    def create_problem_analysis_view(
        contract: MetisDataContract,
    ) -> ProblemAnalysisView:
        """Create view for problem analysis phase"""
        business_context = contract.engagement_context.business_context or {}

        return ProblemAnalysisView(
            engagement_id=contract.engagement_context.engagement_id,
            problem_statement=contract.engagement_context.problem_statement,
            business_context=business_context,
            stakeholders=business_context.get("stakeholders", []),
            success_criteria=business_context.get("success_criteria", []),
            constraints=business_context.get("constraints", {}),
        )

    @staticmethod
    def create_hypothesis_generation_view(
        contract: MetisDataContract,
    ) -> HypothesisGenerationView:
        """Create view for hypothesis generation"""
        return HypothesisGenerationView(
            engagement_id=contract.engagement_context.engagement_id,
            problem_statement=contract.engagement_context.problem_statement,
            business_context=contract.engagement_context.business_context,
            selected_mental_models=[
                model.name for model in contract.cognitive_state.selected_mental_models
            ],
            problem_classification=contract.engagement_context.problem_classification,
            industry=contract.engagement_context.industry,
        )

    @staticmethod
    def create_analysis_execution_view(
        contract: MetisDataContract,
    ) -> AnalysisExecutionView:
        """Create view for analysis execution"""
        # Extract hypotheses from reasoning steps
        hypotheses = []
        for step in contract.cognitive_state.reasoning_steps:
            if "hypothesis" in step.description.lower():
                hypotheses.append(
                    {
                        "description": step.description,
                        "confidence": step.confidence,
                        "evidence": step.evidence_sources,
                    }
                )

        return AnalysisExecutionView(
            engagement_id=contract.engagement_context.engagement_id,
            hypotheses=hypotheses,
            mental_models=contract.cognitive_state.selected_mental_models,
            reasoning_steps=contract.cognitive_state.reasoning_steps,
            confidence_scores=contract.cognitive_state.confidence_scores,
            research_enabled=bool(contract.cognitive_state.research_intelligence),
        )

    @staticmethod
    def create_synthesis_input_view(contract: MetisDataContract) -> SynthesisInputView:
        """Create view for synthesis phase"""
        # Extract key insights and recommendations
        key_insights = []
        recommendations = []

        for step in contract.cognitive_state.reasoning_steps:
            if step.key_insights:
                key_insights.extend(step.key_insights)

        # Extract from deliverable artifacts if available
        for artifact in contract.deliverable_artifacts:
            if artifact.artifact_type == "recommendations":
                recommendations.extend(artifact.content.get("items", []))

        research_intel = contract.cognitive_state.research_intelligence
        evidence_base = research_intel.evidence_base if research_intel else {}

        return SynthesisInputView(
            engagement_id=contract.engagement_context.engagement_id,
            problem_statement=contract.engagement_context.problem_statement,
            reasoning_steps=contract.cognitive_state.reasoning_steps,
            confidence_scores=contract.cognitive_state.confidence_scores,
            key_insights=key_insights,
            recommendations=recommendations,
            evidence_base=evidence_base,
        )

    @staticmethod
    def create_research_context_view(
        contract: MetisDataContract,
    ) -> ResearchContextView:
        """Create view for research operations"""
        return ResearchContextView(
            engagement_id=contract.engagement_context.engagement_id,
            query=contract.engagement_context.problem_statement,
            business_context=contract.engagement_context.business_context,
            research_intelligence=contract.cognitive_state.research_intelligence,
            max_sources=5,  # Could be from config
            confidence_threshold=0.7,  # Could be from config
        )

    @staticmethod
    def create_validation_context_view(
        contract: MetisDataContract,
        content_to_validate: Any,
        validation_type: str = "general",
    ) -> ValidationContextView:
        """Create view for validation operations"""
        vulnerability_context = contract.vulnerability_context

        hallucination_checks = []
        if vulnerability_context and vulnerability_context.hallucination_checks:
            hallucination_checks = [
                {
                    "check_type": check.check_type,
                    "is_valid": check.is_valid,
                    "confidence": check.confidence_score,
                }
                for check in vulnerability_context.hallucination_checks
            ]

        return ValidationContextView(
            engagement_id=contract.engagement_context.engagement_id,
            content_to_validate=content_to_validate,
            validation_type=validation_type,
            confidence_threshold=0.7,  # Could be from config
            hallucination_checks=hallucination_checks,
            bias_detection_enabled=True,
        )

    @staticmethod
    def create_deliverable_generation_view(
        contract: MetisDataContract,
    ) -> DeliverableGenerationView:
        """Create view for deliverable generation"""
        # Extract executive summary
        executive_summary = ""
        if contract.cognitive_state.research_intelligence:
            executive_summary = (
                contract.cognitive_state.research_intelligence.executive_summary
            )

        # Extract key findings
        key_findings = []
        for step in contract.cognitive_state.reasoning_steps:
            if step.key_insights:
                key_findings.extend(step.key_insights)

        # Extract recommendations
        recommendations = []
        for artifact in contract.deliverable_artifacts:
            if artifact.artifact_type == "recommendations":
                recommendations.extend(artifact.content.get("items", []))

        # Determine confidence level
        avg_confidence = (
            sum(contract.cognitive_state.confidence_scores.values())
            / len(contract.cognitive_state.confidence_scores)
            if contract.cognitive_state.confidence_scores
            else 0.5
        )

        if avg_confidence > 0.9:
            confidence_level = ConfidenceLevel.HIGH
        elif avg_confidence > 0.7:
            confidence_level = ConfidenceLevel.MEDIUM
        elif avg_confidence > 0.5:
            confidence_level = ConfidenceLevel.LOW
        else:
            confidence_level = ConfidenceLevel.UNCERTAIN

        # Extract supporting evidence
        supporting_evidence = []
        for artifact in contract.deliverable_artifacts:
            supporting_evidence.extend(artifact.supporting_evidence)

        return DeliverableGenerationView(
            engagement_id=contract.engagement_context.engagement_id,
            client_name=contract.engagement_context.client_name,
            problem_statement=contract.engagement_context.problem_statement,
            executive_summary=executive_summary,
            key_findings=key_findings,
            recommendations=recommendations,
            confidence_level=confidence_level,
            supporting_evidence=supporting_evidence,
        )

    @staticmethod
    def create_clarification_context_view(
        contract: MetisDataContract,
    ) -> ClarificationContextView:
        """Create view for HITL clarification"""
        # Extract required dimensions from business context
        required_dimensions = []
        if contract.clarification_session:
            required_dimensions = contract.clarification_session.dimensions_clarified

        return ClarificationContextView(
            engagement_id=contract.engagement_context.engagement_id,
            original_query=contract.engagement_context.problem_statement,
            clarification_session=contract.clarification_session,
            required_dimensions=required_dimensions,
            user_preferences=contract.engagement_context.user_preferences,
        )

    @staticmethod
    def create_monitoring_metrics_view(
        contract: MetisDataContract,
    ) -> MonitoringMetricsView:
        """Create view for monitoring and metrics"""
        # Count errors from processing metadata
        error_count = contract.processing_metadata.get("error_count", 0)

        # Extract retry counts
        retry_counts = contract.processing_metadata.get("retry_counts", {})

        # Calculate execution time
        start_time = contract.engagement_context.created_at
        # Ensure start_time is timezone-aware; if not, assume UTC
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        current_time = datetime.now(timezone.utc)
        execution_time = (current_time - start_time).total_seconds()

        return MonitoringMetricsView(
            engagement_id=contract.engagement_context.engagement_id,
            current_phase=contract.workflow_state.current_phase,
            completed_phases=contract.workflow_state.completed_phases,
            performance_metrics=contract.workflow_state.performance_metrics,
            error_count=error_count,
            retry_counts=retry_counts,
            execution_time_seconds=execution_time,
        )

    @staticmethod
    def create_cost_tracking_view(contract: MetisDataContract) -> CostTrackingView:
        """Create view for cost tracking"""
        # Extract cost information from processing metadata
        processing_metadata = contract.processing_metadata

        llm_tokens = 0
        api_calls = 0
        estimated_cost = 0.0

        # Sum up from integration calls
        for call in contract.integration_calls:
            if "tokens" in call:
                llm_tokens += call["tokens"]
            if "api" in call.get("type", "").lower():
                api_calls += 1
            if "cost" in call:
                estimated_cost += call["cost"]

        # Extract resource usage
        resource_usage = processing_metadata.get("resource_usage", {})

        return CostTrackingView(
            engagement_id=contract.engagement_context.engagement_id,
            llm_tokens_used=llm_tokens,
            api_calls_made=api_calls,
            estimated_cost_usd=estimated_cost,
            resource_usage=resource_usage,
        )

    @staticmethod
    def validate_view_completeness(view: BaseModel, required_fields: Set[str]) -> bool:
        """
        Validate that a view has all required fields populated.

        Args:
            view: The view to validate
            required_fields: Set of field names that must be non-empty

        Returns:
            True if all required fields are populated
        """
        for field in required_fields:
            if not hasattr(view, field):
                return False
            value = getattr(view, field)
            if value is None or (isinstance(value, (list, dict, str)) and not value):
                return False
        return True

    @staticmethod
    def create_custom_view(
        contract: MetisDataContract, field_mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Create a custom view with specific field mappings.

        Args:
            contract: The data contract
            field_mapping: Dict mapping view field names to contract paths
                          e.g., {"query": "engagement_context.problem_statement"}

        Returns:
            Dictionary with extracted fields
        """
        view = {}

        for view_field, contract_path in field_mapping.items():
            # Navigate the contract path
            value = contract
            for part in contract_path.split("."):
                if hasattr(value, part):
                    value = getattr(value, part)
                elif isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    value = None
                    break

            view[view_field] = value

        return view


# ============================================================================
# VIEW VALIDATORS
# ============================================================================


class ViewValidator:
    """Validators for ensuring view integrity"""

    @staticmethod
    def validate_problem_analysis_view(view: ProblemAnalysisView) -> bool:
        """Validate problem analysis view has required data"""
        return bool(
            view.engagement_id and view.problem_statement and view.business_context
        )

    @staticmethod
    def validate_hypothesis_generation_view(view: HypothesisGenerationView) -> bool:
        """Validate hypothesis generation view has required data"""
        return bool(
            view.engagement_id
            and view.problem_statement
            and view.selected_mental_models
        )

    @staticmethod
    def validate_synthesis_input_view(view: SynthesisInputView) -> bool:
        """Validate synthesis input view has required data"""
        return bool(
            view.engagement_id and view.problem_statement and view.reasoning_steps
        )

    @staticmethod
    def validate_deliverable_generation_view(view: DeliverableGenerationView) -> bool:
        """Validate deliverable generation view has required data"""
        return bool(
            view.engagement_id
            and view.client_name
            and view.problem_statement
            and (view.key_findings or view.recommendations)
        )
