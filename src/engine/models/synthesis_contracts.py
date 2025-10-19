#!/usr/bin/env python3
"""
Synthesis Input Contracts - Operation Synthesis
Re-architected input structure to ensure Devil's Advocate integration
"""

from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class SynthesisInputContract:
    """
    Complete input contract for final synthesis phase
    Forces integration of initial analysis, Devil's Advocate critique, and research
    """

    # Core Components (The Trinity)
    initial_conclusion: str
    devils_advocate_critique: Dict[str, Any]  # Full devils_advocate_proof_data object
    grounding_research: Dict[str, Any]  # Complete research findings and sources

    # Context and Metadata
    engagement_id: str
    complexity_score: float
    quality_threshold: float
    synthesis_requirements: Dict[str, Any]

    # Glass Box Transparency Data
    thinking_process_trail: Dict[str, Any]
    confidence_evolution: Dict[str, float]  # Track confidence through each phase

    def to_synthesis_payload(self) -> Dict[str, Any]:
        """Convert to structured payload for synthesis prompt"""
        return {
            "trinity_inputs": {
                "initial_conclusion": self.initial_conclusion,
                "devils_advocate_critique": self.devils_advocate_critique,
                "grounding_research": self.grounding_research,
            },
            "context": {
                "engagement_id": self.engagement_id,
                "complexity_score": self.complexity_score,
                "quality_threshold": self.quality_threshold,
            },
            "transparency_trail": {
                "thinking_process": self.thinking_process_trail,
                "confidence_evolution": self.confidence_evolution,
            },
            "synthesis_mandate": {
                "integration_required": True,
                "refinement_required": True,
                "house_view_required": True,
                "glass_box_available": True,
            },
        }

    def extract_key_challenges(self) -> List[Dict[str, Any]]:
        """Extract the most critical challenges from Devil's Advocate"""
        if "devils_advocate_impact_analysis" in self.devils_advocate_critique:
            impact_analysis = self.devils_advocate_critique[
                "devils_advocate_impact_analysis"
            ]
            if "transformation_journey" in impact_analysis:
                return impact_analysis["transformation_journey"].get(
                    "critical_challenges_applied", []
                )
        return []

    def extract_bias_corrections(self) -> List[str]:
        """Extract biases that were identified and corrected"""
        if "execution_phases" in self.devils_advocate_critique:
            phase_2 = self.devils_advocate_critique["execution_phases"][
                "phase_2_devils_advocate_activation"
            ]
            if "challenge_systems_activated" in phase_2:
                munger = phase_2["challenge_systems_activated"].get(
                    "munger_overlay", {}
                )
                return munger.get("biases_identified", [])
        return []

    def get_confidence_transformation(self) -> Dict[str, float]:
        """Get the confidence transformation journey"""
        if "devils_advocate_impact_analysis" in self.devils_advocate_critique:
            impact = self.devils_advocate_critique["devils_advocate_impact_analysis"]
            if "transformation_journey" in impact:
                return impact["transformation_journey"].get("confidence_evolution", {})
        return {}


@dataclass
class SynthesisResult:
    """
    Output contract for synthesis phase
    Ensures integrated, refined final output
    """

    # Final Integrated Analysis
    refined_recommendation: str
    house_view_confidence: float
    integration_quality_score: float

    # Synthesis Evidence
    challenges_addressed: List[Dict[str, str]]  # How each challenge was integrated
    biases_corrected: List[Dict[str, str]]  # How each bias was addressed
    research_integration: Dict[str, Any]  # How research validated/refuted points

    # Glass Box Transparency
    synthesis_reasoning: str  # The thinking process of integration
    refinement_evidence: Dict[str, Any]  # Before/after comparison
    confidence_justification: str  # Why final confidence level was chosen

    # Quality Metrics
    initial_vs_final_delta: Dict[str, Any]  # Quantified improvement
    risk_mitigation_achieved: List[str]  # Specific risks addressed
    stakeholder_impact_improved: bool  # Whether stakeholder consideration improved


class SynthesisOrchestrator:
    """
    Orchestrates the synthesis phase with proper integration
    """

    def __init__(self):
        self.synthesis_history = []

    def prepare_synthesis_input(
        self,
        initial_analysis: str,
        devils_advocate_data: Dict[str, Any],
        research_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> SynthesisInputContract:
        """Prepare the complete input contract for synthesis"""

        return SynthesisInputContract(
            initial_conclusion=initial_analysis,
            devils_advocate_critique=devils_advocate_data,
            grounding_research=research_data,
            engagement_id=context.get("engagement_id", "unknown"),
            complexity_score=context.get("complexity_score", 0.5),
            quality_threshold=context.get("quality_threshold", 0.8),
            synthesis_requirements={
                "integration_depth": "comprehensive",
                "refinement_level": "sophisticated",
                "house_view_clarity": "executive_ready",
            },
            thinking_process_trail=self._extract_thinking_trail(devils_advocate_data),
            confidence_evolution=self._extract_confidence_evolution(
                devils_advocate_data
            ),
        )

    def _extract_thinking_trail(
        self, devils_advocate_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract the thinking process for glass box transparency"""
        return {
            "challenges_generated": devils_advocate_data.get("execution_phases", {})
            .get("phase_2_devils_advocate_activation", {})
            .get("challenge_systems_activated", {}),
            "refinement_process": devils_advocate_data.get("execution_phases", {}).get(
                "phase_3_recommendation_refinement", {}
            ),
        }

    def _extract_confidence_evolution(
        self, devils_advocate_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract confidence evolution for tracking"""
        if "devils_advocate_impact_analysis" in devils_advocate_data:
            impact = devils_advocate_data["devils_advocate_impact_analysis"]
            if "transformation_journey" in impact:
                return impact["transformation_journey"].get("confidence_evolution", {})
        return {"initial": 0.5, "post_challenge": 0.3, "final": 0.8}

    def validate_synthesis_quality(
        self, synthesis_result: SynthesisResult
    ) -> Dict[str, Any]:
        """Validate that synthesis properly integrated all inputs"""

        validation = {
            "challenges_addressed": len(synthesis_result.challenges_addressed) > 0,
            "biases_corrected": len(synthesis_result.biases_corrected) > 0,
            "research_integrated": bool(synthesis_result.research_integration),
            "refinement_evident": synthesis_result.integration_quality_score > 0.7,
            "house_view_clear": len(synthesis_result.refined_recommendation) > 1000,
            "glass_box_available": bool(synthesis_result.synthesis_reasoning),
        }

        validation["overall_quality"] = all(validation.values())
        validation["quality_score"] = sum(validation.values()) / len(
            validation.values()
        )

        return validation
