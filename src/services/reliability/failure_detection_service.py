"""
METIS Failure Detection Service
Part of Reliability Services Cluster - Focused on detecting and classifying system failure modes

Extracted from vulnerability_solutions.py during Phase 5 decomposition.
Single Responsibility: Detect failure modes and generate recovery strategies.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

# Service interfaces and contracts
from src.services.contracts.reliability_contracts import (
    FailureAnalysisContract,
    IFailureDetectionService,
    ConfidenceClassification,
    FailureMode,
    RecommendationStatus,
)


@dataclass
class FailureAnalysis:
    """Analysis of engagement failure modes and limitations"""

    failure_modes: List[FailureMode]
    confidence_classification: ConfidenceClassification
    recommendation_status: RecommendationStatus
    limitations: List[Dict[str, Any]]
    partial_insights: List[Dict[str, Any]]
    actionable_next_steps: List[str]
    alternative_approaches: List[str]
    user_value_provided: str
    learning_opportunity: bool = True


class FailureDetectionService(IFailureDetectionService):
    """
    Focused service for detecting and classifying system failure modes
    Clean extraction from vulnerability_solutions.py FailureModeManager
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.confidence_thresholds = {
            "high": 0.85,
            "medium": 0.65,
            "low": 0.45,
            "insufficient": 0.45,
        }

        self.logger.info("🔍 FailureDetectionService initialized")

    async def analyze_engagement_health(
        self, cognitive_state: Dict[str, Any]
    ) -> FailureAnalysisContract:
        """
        Core service method: Analyze engagement for failure modes and determine response
        Clean, focused implementation with single responsibility
        """
        try:
            overall_confidence = cognitive_state.get("confidence_scores", {}).get(
                "overall", 0.0
            )
            validation_results = cognitive_state.get("validation_results", {})
            research_intelligence = cognitive_state.get("research_intelligence", {})

            # Detect failure modes using focused algorithms
            failure_modes = []
            limitations = []
            partial_insights = []

            # 1. Low confidence analysis detection
            if overall_confidence < self.confidence_thresholds["low"]:
                failure_modes.append(FailureMode.LOW_CONFIDENCE_ANALYSIS)
                limitations.append(
                    {
                        "limitation": "Analysis confidence below reliability threshold",
                        "impact": f"Overall confidence {overall_confidence:.2f} indicates high uncertainty",
                        "mitigation": "Recommend additional research and validation before decisions",
                        "value_provided": "Prevented potentially unreliable strategic guidance",
                    }
                )

            # 2. Devils advocate rejection detection
            if validation_results.get("validation_confidence", 1.0) < 0.6:
                failure_modes.append(FailureMode.DEVILS_ADVOCATE_REJECTION)
                critical_assumptions = validation_results.get(
                    "critical_assumptions_challenged", []
                )
                for assumption in critical_assumptions:
                    if assumption.get("risk_level") == "high":
                        limitations.append(
                            {
                                "limitation": f"Critical assumption '{assumption.get('assumption')}' flagged as high risk",
                                "impact": "Strategy vulnerable to assumption failure",
                                "mitigation": assumption.get(
                                    "mitigation", "Additional validation required"
                                ),
                                "value_provided": "Identified potential strategic failure point",
                            }
                        )

            # 3. Research gaps detection
            research_confidence = research_intelligence.get("overall_confidence", 1.0)
            if research_confidence < 0.7:
                failure_modes.append(FailureMode.RESEARCH_GAPS)
                limitations.append(
                    {
                        "limitation": "Insufficient research foundation for confident analysis",
                        "impact": "Recommendations may not account for market realities",
                        "mitigation": "Commission targeted research on identified gaps",
                        "value_provided": "Prevented decisions based on incomplete market understanding",
                    }
                )

            # Generate partial insights for what did work
            reasoning_steps = cognitive_state.get("reasoning_steps", [])
            for step in reasoning_steps:
                if step.get("confidence", 0) > 0.75:
                    partial_insights.append(
                        {
                            "insight": step.get("description", ""),
                            "confidence": step.get("confidence"),
                            "reliability": (
                                "HIGH - validated through multiple sources"
                                if step.get("research_enhanced")
                                else "MEDIUM - analytical conclusion"
                            ),
                            "key_insights": step.get("key_insights", []),
                        }
                    )

            # Classify overall confidence and determine recommendation status
            confidence_class, recommendation_status = self._classify_confidence(
                overall_confidence
            )

            # Generate recovery strategies
            next_steps = self._generate_recovery_steps(failure_modes, limitations)
            alternatives = self._suggest_alternative_approaches(
                failure_modes, cognitive_state
            )
            user_value = self._calculate_failure_value(failure_modes, partial_insights)

            # Create analysis result
            analysis = FailureAnalysis(
                failure_modes=failure_modes,
                confidence_classification=confidence_class,
                recommendation_status=recommendation_status,
                limitations=limitations,
                partial_insights=partial_insights,
                actionable_next_steps=next_steps,
                alternative_approaches=alternatives,
                user_value_provided=user_value,
                learning_opportunity=True,
            )

            # Convert to service contract
            return FailureAnalysisContract(
                engagement_id=cognitive_state.get("engagement_id", "unknown"),
                failure_modes=[mode.value for mode in failure_modes],
                confidence_classification=confidence_class.value,
                recommendation_status=recommendation_status.value,
                limitations=limitations,
                partial_insights=partial_insights,
                recovery_steps=next_steps,
                alternative_approaches=alternatives,
                user_value_provided=user_value,
                analysis_timestamp=datetime.utcnow(),
                service_version="v5_modular",
            )

        except Exception as e:
            self.logger.error(f"❌ Failure detection analysis failed: {e}")
            # Return safe fallback analysis
            return self._create_fallback_analysis(cognitive_state, str(e))

    def _classify_confidence(
        self, overall_confidence: float
    ) -> tuple[ConfidenceClassification, RecommendationStatus]:
        """Classify confidence level and determine recommendation status"""
        if overall_confidence >= self.confidence_thresholds["high"]:
            return (
                ConfidenceClassification.HIGH_CONFIDENCE,
                RecommendationStatus.RECOMMENDED,
            )
        elif overall_confidence >= self.confidence_thresholds["medium"]:
            return (
                ConfidenceClassification.MEDIUM_CONFIDENCE,
                RecommendationStatus.CONDITIONAL_WITH_CAVEATS,
            )
        elif overall_confidence >= self.confidence_thresholds["low"]:
            return (
                ConfidenceClassification.LOW_CONFIDENCE,
                RecommendationStatus.CONDITIONAL_WITH_CAVEATS,
            )
        else:
            return (
                ConfidenceClassification.INSUFFICIENT_CONFIDENCE,
                RecommendationStatus.INSUFFICIENT_DATA,
            )

    def _generate_recovery_steps(
        self, failure_modes: List[FailureMode], limitations: List[Dict]
    ) -> List[str]:
        """Generate specific recovery steps based on detected failure modes"""
        steps = []

        if FailureMode.RESEARCH_GAPS in failure_modes:
            steps.append(
                "Commission focused market research on identified information gaps"
            )
            steps.append("Engage industry experts for domain-specific validation")

        if FailureMode.LOW_CONFIDENCE_ANALYSIS in failure_modes:
            steps.append("Gather additional data sources for cross-validation")
            steps.append(
                "Consider pilot approach to test assumptions with limited risk"
            )

        if FailureMode.DEVILS_ADVOCATE_REJECTION in failure_modes:
            steps.append("Address high-risk assumptions through mitigation strategies")
            steps.append("Develop contingency plans for assumption failures")

        steps.append("Re-run analysis after addressing identified limitations")

        return steps

    def _suggest_alternative_approaches(
        self, failure_modes: List[FailureMode], cognitive_state: Dict
    ) -> List[str]:
        """Suggest alternative analytical approaches based on failure patterns"""
        alternatives = []

        if FailureMode.LOW_CONFIDENCE_ANALYSIS in failure_modes:
            alternatives.append("Scenario-based analysis with multiple future states")
            alternatives.append("Stakeholder-driven validation approach")

        if FailureMode.RESEARCH_GAPS in failure_modes:
            alternatives.append("Expert interview-based analysis")
            alternatives.append("Competitive intelligence focused approach")

        alternatives.append("Phased decision-making with learning milestones")

        return alternatives

    def _calculate_failure_value(
        self, failure_modes: List[FailureMode], partial_insights: List[Dict]
    ) -> str:
        """Calculate what value was provided despite failure"""
        value_components = []

        if len(partial_insights) > 0:
            value_components.append(
                f"Generated {len(partial_insights)} high-confidence insights"
            )

        if FailureMode.DEVILS_ADVOCATE_REJECTION in failure_modes:
            value_components.append("Prevented potentially risky strategic decisions")

        if FailureMode.RESEARCH_GAPS in failure_modes:
            value_components.append(
                "Identified critical information gaps requiring attention"
            )

        if not value_components:
            value_components.append(
                "Prevented unreliable analysis from reaching stakeholders"
            )

        return "; ".join(value_components)

    def _create_fallback_analysis(
        self, cognitive_state: Dict, error_msg: str
    ) -> FailureAnalysisContract:
        """Create safe fallback analysis when service fails"""
        return FailureAnalysisContract(
            engagement_id=cognitive_state.get("engagement_id", "unknown"),
            failure_modes=["service_failure"],
            confidence_classification="INSUFFICIENT_CONFIDENCE",
            recommendation_status="INSUFFICIENT_DATA",
            limitations=[
                {
                    "limitation": f"Failure detection service error: {error_msg}",
                    "impact": "Unable to analyze engagement health",
                    "mitigation": "Retry analysis or contact support",
                    "value_provided": "Service failure prevented unreliable analysis",
                }
            ],
            partial_insights=[],
            recovery_steps=[
                "Retry failure detection analysis",
                "Contact technical support",
            ],
            alternative_approaches=[
                "Manual engagement review",
                "Simplified analysis approach",
            ],
            user_value_provided="Service failure detection prevented unreliable analysis delivery",
            analysis_timestamp=datetime.utcnow(),
            service_version="v5_modular_fallback",
        )

    async def handle_component_failure(
        self, component_name: str, error: str, context: Any
    ) -> Dict[str, Any]:
        """Handle component failure and provide graceful degradation"""
        self.logger.warning(f"⚠️ Component failure in {component_name}: {error}")

        return {
            "component": component_name,
            "error": error,
            "fallback_strategy": "graceful_degradation",
            "user_message": f"Component {component_name} encountered an issue but the analysis will continue with reduced functionality",
            "continue_processing": True,
            "fallback_analysis": {
                "confidence": 0.3,
                "approach": "simplified_fallback",
                "context_available": bool(context),
            },
        }

    async def handle_catastrophic_failure(
        self, error: Exception, engagement_contract: Any
    ) -> Dict[str, Any]:
        """Handle catastrophic system failure with maximum value extraction"""
        engagement_id = "unknown"

        try:
            if hasattr(engagement_contract, "engagement_id"):
                engagement_id = str(engagement_contract.engagement_id)
            elif isinstance(engagement_contract, dict):
                engagement_id = engagement_contract.get("engagement_id", "unknown")
        except Exception:
            pass

        self.logger.error(
            f"🔥 Catastrophic failure in engagement {engagement_id}: {str(error)}"
        )

        return {
            "engagement_id": engagement_id,
            "failure_type": "catastrophic",
            "error": str(error),
            "user_value_extracted": "System failure prevented completion, but no unreliable analysis was delivered",
            "recovery_suggestions": [
                "Retry the analysis with simplified parameters",
                "Break down the problem into smaller components",
                "Contact support if the issue persists",
            ],
            "continue_processing": False,
        }

    async def get_service_health(self) -> Dict[str, Any]:
        """Return service health status"""
        return {
            "service_name": "FailureDetectionService",
            "status": "healthy",
            "version": "v5_modular",
            "capabilities": [
                "failure_mode_detection",
                "confidence_classification",
                "recovery_step_generation",
                "alternative_approach_suggestion",
            ],
            "last_health_check": datetime.utcnow().isoformat(),
        }


# Global service instance for dependency injection
_failure_detection_service: Optional[FailureDetectionService] = None


def get_failure_detection_service() -> FailureDetectionService:
    """Get or create global failure detection service instance"""
    global _failure_detection_service

    if _failure_detection_service is None:
        _failure_detection_service = FailureDetectionService()

    return _failure_detection_service
