#!/usr/bin/env python3
"""
METIS Universal Interface Adapter System
Unified adapter for all component interfaces - enables seamless integration
between legacy simple interfaces, unified data contracts, and API models.

This adapter system is the foundation for systematic interface standardization
while maintaining 100% backward compatibility.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from uuid import uuid4
from enum import Enum

from src.engine.models.data_contracts import (
    MetisDataContract,
    EngagementContext,
    CognitiveState,
    WorkflowState,
    EngagementPhase,
    ReasoningStep,
)


class InterfacePattern(str, Enum):
    """Different interface patterns in the METIS system"""

    LEGACY_SIMPLE = "legacy_simple"  # Dict-based simple interfaces
    UNIFIED_CONTRACT = "unified_contract"  # MetisDataContract interfaces
    API_MODELS = "api_models"  # REST API model interfaces


class AdapterError(Exception):
    """Base exception for interface adapter errors"""

    pass


class MetisInterfaceAdapter:
    """
    Universal adapter for all METIS component interfaces

    Enables seamless conversion between:
    - Legacy simple interfaces (str, dict, str) -> dict
    - Unified data contracts MetisDataContract -> MetisDataContract
    - API models EngagementRequest -> EngagementResponse

    This adapter maintains backward compatibility while enabling
    progressive migration to unified interfaces.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.conversion_count = {
            "simple_to_contract": 0,
            "contract_to_simple": 0,
            "api_to_contract": 0,
            "contract_to_api": 0,
        }

    # =====================================
    # SIMPLE INTERFACE TO CONTRACT ADAPTERS
    # =====================================

    async def simple_to_contract(
        self,
        method_result: Dict[str, Any],
        engagement_context: EngagementContext,
        phase: EngagementPhase,
        step_id: Optional[str] = None,
    ) -> MetisDataContract:
        """
        Convert legacy simple method result to unified MetisDataContract

        Args:
            method_result: Result from legacy method (e.g., _apply_systems_thinking)
            engagement_context: Current engagement context
            phase: Current workflow phase
            step_id: Optional step identifier

        Returns:
            Unified MetisDataContract with enriched data
        """
        try:
            self.conversion_count["simple_to_contract"] += 1

            # Extract reasoning step from method result
            reasoning_step = self._create_reasoning_step_from_simple(
                method_result, step_id or f"step_{datetime.utcnow().timestamp()}"
            )

            # Create or update cognitive state
            cognitive_state = CognitiveState(
                reasoning_steps=[reasoning_step],
                confidence_scores={
                    reasoning_step.mental_model_applied: reasoning_step.confidence_score
                },
            )

            # Create workflow state for current phase
            workflow_state = WorkflowState(
                current_phase=phase,
                completed_phases=[],
                phase_results={phase.value: method_result},
            )

            # Create unified contract
            contract = MetisDataContract(
                type="metis.analysis_framework_applied",
                source="/metis/cognitive_engine",
                engagement_context=engagement_context,
                cognitive_state=cognitive_state,
                workflow_state=workflow_state,
                deliverable_artifacts=[],
            )

            self.logger.debug(
                f"Converted simple result to contract for phase {phase.value}"
            )
            return contract

        except Exception as e:
            self.logger.error(f"Failed to convert simple result to contract: {e}")
            raise AdapterError(f"Simple to contract conversion failed: {e}")

    def _create_reasoning_step_from_simple(
        self, method_result: Dict[str, Any], step_id: str
    ) -> ReasoningStep:
        """Create ReasoningStep from legacy method result"""

        return ReasoningStep(
            step_id=step_id,
            mental_model_applied=method_result.get("model_applied", "unknown_model"),
            reasoning_text=method_result.get("reasoning_text", str(method_result)),
            confidence_score=method_result.get("confidence_score", 0.8),
            evidence_sources=method_result.get("evidence_sources", []),
            assumptions_made=method_result.get("assumptions_made", []),
        )

    # =====================================
    # CONTRACT TO SIMPLE INTERFACE ADAPTERS
    # =====================================

    async def contract_to_simple(
        self,
        contract: MetisDataContract,
        extract_phase: Optional[EngagementPhase] = None,
    ) -> Tuple[str, Dict[str, Any], str]:
        """
        Convert MetisDataContract to legacy simple interface parameters

        Args:
            contract: Unified MetisDataContract
            extract_phase: Optional phase to extract data from

        Returns:
            Tuple of (problem_statement, business_context, step_id)
        """
        try:
            self.conversion_count["contract_to_simple"] += 1

            # Extract problem statement
            problem_statement = contract.engagement_context.problem_statement

            # Extract business context
            business_context = contract.engagement_context.business_context.copy()

            # Add enriched context from cognitive state
            if contract.cognitive_state.confidence_scores:
                business_context["confidence_scores"] = (
                    contract.cognitive_state.confidence_scores
                )

            # Add workflow context
            if extract_phase and contract.workflow_state.phase_results:
                phase_data = contract.workflow_state.phase_results.get(
                    extract_phase.value, {}
                )
                business_context.update(phase_data)

            # Generate step ID
            step_id = f"step_{contract.id.hex[:8]}_{datetime.utcnow().timestamp()}"

            self.logger.debug("Converted contract to simple parameters")
            return problem_statement, business_context, step_id

        except Exception as e:
            self.logger.error(f"Failed to convert contract to simple: {e}")
            raise AdapterError(f"Contract to simple conversion failed: {e}")

    # =====================================
    # API MODEL TO CONTRACT ADAPTERS
    # =====================================

    async def api_to_contract(
        self,
        engagement_request: Any,  # EngagementRequest from API models
        client_name: Optional[str] = None,
    ) -> MetisDataContract:
        """
        Convert API EngagementRequest to unified MetisDataContract

        Args:
            engagement_request: API EngagementRequest object
            client_name: Optional client name override

        Returns:
            Unified MetisDataContract ready for processing
        """
        try:
            self.conversion_count["api_to_contract"] += 1

            # Extract problem statement and context
            problem_stmt = engagement_request.problem_statement

            # Create engagement context
            engagement_context = EngagementContext(
                engagement_id=uuid4(),
                problem_statement=problem_stmt.problem_description,
                client_name=client_name
                or getattr(engagement_request, "client_name", "Unknown Client"),
                business_context=problem_stmt.business_context or {},
                user_preferences={
                    "engagement_type": getattr(
                        engagement_request, "engagement_type", "strategy_consulting"
                    ),
                    "priority": getattr(engagement_request, "priority", "medium"),
                },
            )

            # Initialize cognitive and workflow states
            cognitive_state = CognitiveState()
            workflow_state = WorkflowState(
                current_phase=EngagementPhase.PROBLEM_STRUCTURING
            )

            # Create unified contract
            contract = MetisDataContract(
                type="metis.engagement_request",
                source="/metis/api_gateway",
                engagement_context=engagement_context,
                cognitive_state=cognitive_state,
                workflow_state=workflow_state,
            )

            self.logger.info(
                f"Converted API request to contract for engagement {engagement_context.engagement_id}"
            )
            return contract

        except Exception as e:
            self.logger.error(f"Failed to convert API request to contract: {e}")
            raise AdapterError(f"API to contract conversion failed: {e}")

    # =====================================
    # CONTRACT TO API MODEL ADAPTERS
    # =====================================

    async def contract_to_api_response(
        self,
        contract: MetisDataContract,
        response_class: Any,  # EngagementResponse class
    ) -> Any:
        """
        Convert MetisDataContract to API response model

        Args:
            contract: Unified MetisDataContract
            response_class: API response model class

        Returns:
            API response model instance
        """
        try:
            self.conversion_count["contract_to_api"] += 1

            # Calculate progress based on completed phases
            completed_phases = contract.workflow_state.completed_phases
            total_phases = len(EngagementPhase)
            progress_percentage = (len(completed_phases) / total_phases) * 100

            # Calculate overall confidence
            confidence_scores = contract.cognitive_state.confidence_scores
            overall_confidence = (
                sum(confidence_scores.values()) / len(confidence_scores)
                if confidence_scores
                else 0.5
            )

            # Extract deliverable status
            deliverable_ready = len(completed_phases) == total_phases

            # Create API response
            response = response_class(
                engagement_id=contract.engagement_context.engagement_id,
                client_name=contract.engagement_context.client_name,
                problem_statement=self._create_problem_statement_from_contract(
                    contract
                ),
                status=self._determine_engagement_status(contract),
                current_phase=contract.workflow_state.current_phase,
                progress_percentage=progress_percentage,
                phases=self._extract_phase_results(contract),
                overall_confidence=overall_confidence,
                estimated_cost=self._calculate_estimated_cost(contract),
                created_at=contract.engagement_context.created_at,
                updated_at=contract.time,
                deliverable_ready=deliverable_ready,
            )

            self.logger.debug("Converted contract to API response")
            return response

        except Exception as e:
            self.logger.error(f"Failed to convert contract to API response: {e}")
            raise AdapterError(f"Contract to API response conversion failed: {e}")

    def _create_problem_statement_from_contract(
        self, contract: MetisDataContract
    ) -> Any:
        """Create ProblemStatement API model from contract"""
        # This would import and create the actual ProblemStatement API model
        # For now, return a simple object structure
        return {
            "problem_description": contract.engagement_context.problem_statement,
            "business_context": contract.engagement_context.business_context,
            "stakeholders": [],
            "success_criteria": [],
        }

    def _determine_engagement_status(self, contract: MetisDataContract) -> str:
        """Determine engagement status from contract state"""
        completed_phases = contract.workflow_state.completed_phases
        total_phases = len(EngagementPhase)

        if len(completed_phases) == total_phases:
            return "COMPLETED"
        elif len(completed_phases) > 0:
            return "IN_PROGRESS"
        else:
            return "CREATED"

    def _extract_phase_results(self, contract: MetisDataContract) -> Dict[str, Any]:
        """Extract phase results for API response"""
        phase_results = {}

        for phase_name, result_data in contract.workflow_state.phase_results.items():
            phase_results[phase_name] = {
                "status": (
                    "completed"
                    if phase_name
                    in [p.value for p in contract.workflow_state.completed_phases]
                    else "pending"
                ),
                "confidence": self._extract_phase_confidence(contract, phase_name),
                "insights": self._extract_phase_insights(result_data),
                "data": result_data,
            }

        return phase_results

    def _extract_phase_confidence(
        self, contract: MetisDataContract, phase_name: str
    ) -> float:
        """Extract confidence score for specific phase"""
        # Look for phase-specific confidence in cognitive state
        confidence_scores = contract.cognitive_state.confidence_scores

        # Try to find phase-specific confidence
        for model_name, confidence in confidence_scores.items():
            if phase_name.lower() in model_name.lower():
                return confidence

        # Return average confidence if no specific match
        if confidence_scores:
            return sum(confidence_scores.values()) / len(confidence_scores)

        return 0.8  # Default confidence

    def _extract_phase_insights(self, result_data: Dict[str, Any]) -> List[str]:
        """Extract insights from phase result data"""
        insights = []

        # Extract from reasoning text if available
        reasoning_text = result_data.get("reasoning_text", "")
        if reasoning_text and len(reasoning_text) > 100:
            # Simple insight extraction - could be enhanced with NLP
            lines = reasoning_text.split("\n")
            for line in lines:
                line = line.strip()
                if (
                    line.startswith("- ")
                    or line.startswith("• ")
                    or "insight:" in line.lower()
                    or "key finding:" in line.lower()
                ):
                    insights.append(line)

        # Extract from other result fields
        if "key_insights" in result_data:
            insights.extend(result_data["key_insights"])

        return insights[:5]  # Limit to top 5 insights

    def _calculate_estimated_cost(self, contract: MetisDataContract) -> float:
        """Calculate estimated cost based on contract processing"""
        # Simple cost estimation based on phases and complexity
        base_cost = 0.10  # Base cost per engagement

        completed_phases = len(contract.workflow_state.completed_phases)
        phase_cost = completed_phases * 0.15

        # Add cost based on reasoning steps
        reasoning_steps = len(contract.cognitive_state.reasoning_steps)
        reasoning_cost = reasoning_steps * 0.02

        return round(base_cost + phase_cost + reasoning_cost, 4)

    # =====================================
    # BATCH CONVERSION UTILITIES
    # =====================================

    async def batch_simple_to_contract(
        self,
        method_results: List[Dict[str, Any]],
        engagement_context: EngagementContext,
        phase: EngagementPhase,
    ) -> MetisDataContract:
        """
        Convert multiple simple method results to single unified contract

        Useful for combining results from multiple mental models
        """
        try:
            reasoning_steps = []
            confidence_scores = {}

            for i, result in enumerate(method_results):
                step_id = f"batch_step_{i}_{datetime.utcnow().timestamp()}"
                reasoning_step = self._create_reasoning_step_from_simple(
                    result, step_id
                )
                reasoning_steps.append(reasoning_step)

                confidence_scores[reasoning_step.mental_model_applied] = (
                    reasoning_step.confidence_score
                )

            # Create consolidated cognitive state
            cognitive_state = CognitiveState(
                reasoning_steps=reasoning_steps, confidence_scores=confidence_scores
            )

            # Create workflow state
            workflow_state = WorkflowState(
                current_phase=phase,
                phase_results={phase.value: {"batch_results": method_results}},
            )

            # Create unified contract
            contract = MetisDataContract(
                type="metis.analysis_framework_applied",
                source="/metis/cognitive_engine",
                engagement_context=engagement_context,
                cognitive_state=cognitive_state,
                workflow_state=workflow_state,
            )

            self.logger.info(
                f"Converted {len(method_results)} simple results to unified contract"
            )
            return contract

        except Exception as e:
            self.logger.error(f"Failed batch conversion: {e}")
            raise AdapterError(f"Batch simple to contract conversion failed: {e}")

    # =====================================
    # ADAPTER DIAGNOSTICS AND MONITORING
    # =====================================

    def get_conversion_metrics(self) -> Dict[str, Any]:
        """Get adapter conversion metrics for monitoring"""
        total_conversions = sum(self.conversion_count.values())

        return {
            "total_conversions": total_conversions,
            "conversion_breakdown": self.conversion_count.copy(),
            "conversion_patterns": {
                pattern: count / total_conversions if total_conversions > 0 else 0
                for pattern, count in self.conversion_count.items()
            },
        }

    def reset_metrics(self):
        """Reset conversion metrics"""
        self.conversion_count = {key: 0 for key in self.conversion_count}
        self.logger.info("Adapter metrics reset")


# Global adapter instance for system-wide use
_global_adapter: Optional[MetisInterfaceAdapter] = None


def get_interface_adapter() -> MetisInterfaceAdapter:
    """Get global interface adapter instance"""
    global _global_adapter
    if _global_adapter is None:
        _global_adapter = MetisInterfaceAdapter()
    return _global_adapter


# Convenience functions for common conversions
async def simple_to_contract(
    method_result: Dict[str, Any],
    engagement_context: EngagementContext,
    phase: EngagementPhase,
    step_id: Optional[str] = None,
) -> MetisDataContract:
    """Convenience function for simple to contract conversion"""
    adapter = get_interface_adapter()
    return await adapter.simple_to_contract(
        method_result, engagement_context, phase, step_id
    )


async def contract_to_simple(
    contract: MetisDataContract, extract_phase: Optional[EngagementPhase] = None
) -> Tuple[str, Dict[str, Any], str]:
    """Convenience function for contract to simple conversion"""
    adapter = get_interface_adapter()
    return await adapter.contract_to_simple(contract, extract_phase)


async def api_to_contract(
    engagement_request: Any, client_name: Optional[str] = None
) -> MetisDataContract:
    """Convenience function for API to contract conversion"""
    adapter = get_interface_adapter()
    return await adapter.api_to_contract(engagement_request, client_name)


async def contract_to_api_response(
    contract: MetisDataContract, response_class: Any
) -> Any:
    """Convenience function for contract to API response conversion"""
    adapter = get_interface_adapter()
    return await adapter.contract_to_api_response(contract, response_class)
