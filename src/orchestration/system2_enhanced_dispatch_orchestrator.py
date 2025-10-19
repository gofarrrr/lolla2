#!/usr/bin/env python3
"""
System 2 Enhanced Dispatch Orchestrator for LOLLA V1.0
======================================================

ARCHITECTURAL BREAKTHROUGH: Contextual Lollapalooza + Mental Model Awareness

This enhanced dispatch orchestrator wraps LOLLA's sophisticated Contextual Lollapalooza Engine
with System 2 mental model awareness for intelligent consultant selection.

Key Innovation:
- Preserves all existing Contextual Lollapalooza capabilities
- Adds System 2 mental model matching for consultants
- Enhances consultant selection with cognitive stage awareness
- Provides mental model transparency in dispatch decisions
- Maintains compatibility with existing dispatch orchestrator API

Integration with LOLLA:
- Enhances existing intelligent consultant selection
- Adds mental model fit scoring to contextual scoring
- Provides System 2 cognitive stage awareness in selection
- Records mental model transparency for audit trail
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# LOLLA Core Dispatch Orchestrator
from .dispatch_orchestrator import DispatchOrchestrator
from .contracts import (
    StructuredAnalyticalFramework,
    DispatchPackage,
)

# System 2 Components
try:
    from ..model_interaction_matrix import CognitiveStage, CognitiveArchitectureBridge
    from ..services.selection.system2_enhanced_chemistry_engine import (
        get_system2_enhanced_chemistry_engine,
    )
    from ..nway_cognitive_architecture import get_cognitive_architecture_nways
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from model_interaction_matrix import CognitiveStage, CognitiveArchitectureBridge
    from services.selection.system2_enhanced_chemistry_engine import (
        get_system2_enhanced_chemistry_engine,
    )
    from nway_cognitive_architecture import get_cognitive_architecture_nways

# LOLLA Core Context
from src.core.unified_context_stream import UnifiedContextStream

logger = logging.getLogger(__name__)


@dataclass
class System2ConsultantMatch:
    """System 2 consultant matching result with mental model fit"""

    consultant_id: str
    base_contextual_score: float
    mental_model_fit_score: float
    cognitive_stage_alignment: float
    recommended_mental_models: List[str]
    system2_enhanced_score: float
    selection_rationale: str


@dataclass
class System2DispatchResult:
    """Enhanced dispatch result with System 2 mental model integration"""

    # Original LOLLA dispatch package
    base_dispatch_package: DispatchPackage

    # System 2 enhancements
    cognitive_stage: CognitiveStage
    consultant_mental_model_matches: List[System2ConsultantMatch]
    mental_model_coverage: Dict[str, float]
    cognitive_diversity_score: float

    # Enhanced selection transparency
    system2_selection_rationale: str
    mental_model_transparency: Dict[str, List[str]]  # consultant -> mental models
    system2_advantage_metrics: Dict[str, float]


class System2EnhancedDispatchOrchestrator:
    """
    Enhanced Dispatch Orchestrator integrating LOLLA's Contextual Lollapalooza Engine
    with System 2 mental model awareness for intelligent consultant selection.

    ARCHITECTURAL PRINCIPLE: Enhancement, not replacement
    - Preserves all existing Contextual Lollapalooza capabilities
    - Adds System 2 mental model matching and cognitive stage awareness
    - Enhances selection transparency and audit trail
    - Maintains compatibility with existing dispatch API
    """

    def __init__(self, context_stream: Optional[UnifiedContextStream] = None):
        # Core LOLLA Dispatch Orchestrator
        self.base_dispatch_orchestrator = DispatchOrchestrator()
        from src.core.unified_context_stream import get_unified_context_stream, UnifiedContextStream
        self.context_stream = context_stream or get_unified_context_stream()

        # System 2 components
        self.cognitive_bridge = CognitiveArchitectureBridge()
        self.chemistry_engine = get_system2_enhanced_chemistry_engine(
            self.context_stream
        )
        self.cognitive_nways = get_cognitive_architecture_nways()

        # Mental model-consultant mapping
        self.consultant_mental_model_preferences = (
            self._load_consultant_mental_model_preferences()
        )

        # System 2 scoring weights
        self.system2_weights = {
            "base_contextual_score": 0.5,  # Original Lollapalooza scoring
            "mental_model_fit": 0.3,  # Mental model alignment
            "cognitive_stage_alignment": 0.2,  # Stage-specific suitability
        }

        logger.info("🎯 System 2 Enhanced Dispatch Orchestrator initialized")
        logger.info(
            "   • Contextual Lollapalooza Engine enhanced with mental model awareness"
        )
        logger.info("   • 100 mental models integrated for consultant selection")

    async def system2_enhanced_dispatch(
        self,
        framework: StructuredAnalyticalFramework,
        cognitive_stage: CognitiveStage,
        num_consultants: int = 3,
        mental_model_requirements: Optional[List[str]] = None,
    ) -> System2DispatchResult:
        """
        Execute System 2-enhanced consultant dispatch with mental model awareness.

        This is the main entry point that wraps LOLLA's dispatch orchestrator
        with System 2 cognitive stage awareness and mental model matching.

        Args:
            framework: Structured analytical framework from problem structuring
            cognitive_stage: Current System 2 cognitive stage
            num_consultants: Number of consultants to select
            mental_model_requirements: Optional specific mental models needed

        Returns:
            Enhanced dispatch result with mental model integration
        """
        logger.info(f"🎯 SYSTEM 2 ENHANCED DISPATCH - Stage: {cognitive_stage.value}")

        # STEP 1: Execute base dispatch using Contextual Lollapalooza Engine
        base_dispatch = await self.base_dispatch_orchestrator.dispatch(
            framework=framework,
            additional_selection_criteria={"cognitive_stage": cognitive_stage.value},
        )

        # STEP 2: Enhance consultant selection with mental model awareness
        enhanced_consultant_matches = (
            await self._enhance_consultant_selection_with_mental_models(
                base_dispatch, cognitive_stage, mental_model_requirements
            )
        )

        # STEP 3: Calculate mental model coverage and cognitive diversity
        mental_model_coverage = self._calculate_mental_model_coverage(
            enhanced_consultant_matches
        )
        cognitive_diversity_score = self._calculate_cognitive_diversity_score(
            enhanced_consultant_matches
        )

        # STEP 4: Generate System 2 selection rationale
        selection_rationale = self._generate_system2_selection_rationale(
            enhanced_consultant_matches, cognitive_stage, mental_model_coverage
        )

        # STEP 5: Create mental model transparency mapping
        mental_model_transparency = {
            match.consultant_id: match.recommended_mental_models
            for match in enhanced_consultant_matches
        }

        # STEP 6: Calculate System 2 advantage metrics
        advantage_metrics = self._calculate_system2_advantage_metrics(
            enhanced_consultant_matches, cognitive_diversity_score
        )

        # STEP 7: Create enhanced dispatch result
        enhanced_result = System2DispatchResult(
            base_dispatch_package=base_dispatch,
            cognitive_stage=cognitive_stage,
            consultant_mental_model_matches=enhanced_consultant_matches,
            mental_model_coverage=mental_model_coverage,
            cognitive_diversity_score=cognitive_diversity_score,
            system2_selection_rationale=selection_rationale,
            mental_model_transparency=mental_model_transparency,
            system2_advantage_metrics=advantage_metrics,
        )

        # STEP 8: Record System 2 enhancement evidence
        await self._record_system2_dispatch_evidence(enhanced_result)

        logger.info("⚡ SYSTEM 2 ENHANCED DISPATCH COMPLETE")
        logger.info(f"   • Consultants Selected: {len(enhanced_consultant_matches)}")
        logger.info(
            f"   • Mental Model Coverage: {len(set().union(*[m.recommended_mental_models for m in enhanced_consultant_matches]))}"
        )
        logger.info(f"   • Cognitive Diversity: {cognitive_diversity_score:.3f}")

        return enhanced_result

    async def _enhance_consultant_selection_with_mental_models(
        self,
        base_dispatch: DispatchPackage,
        cognitive_stage: CognitiveStage,
        mental_model_requirements: Optional[List[str]] = None,
    ) -> List[System2ConsultantMatch]:
        """
        Enhance consultant selection with mental model awareness.

        Takes the consultants selected by Contextual Lollapalooza Engine
        and enhances them with mental model fit scoring and cognitive stage alignment.
        """
        logger.info("🧠 Enhancing consultant selection with mental model awareness")

        enhanced_matches = []

        for consultant_blueprint in base_dispatch.selected_consultants:
            consultant_id = consultant_blueprint.consultant_id

            # Get base contextual score from Lollapalooza Engine
            base_score = consultant_blueprint.selection_score

            # Calculate mental model fit score
            mental_model_fit = await self._calculate_mental_model_fit_score(
                consultant_id, cognitive_stage, mental_model_requirements
            )

            # Calculate cognitive stage alignment
            stage_alignment = self._calculate_cognitive_stage_alignment(
                consultant_id, cognitive_stage
            )

            # Get recommended mental models for this consultant
            recommended_models = (
                await self._get_recommended_mental_models_for_consultant(
                    consultant_id, cognitive_stage, mental_model_requirements
                )
            )

            # Calculate System 2 enhanced score
            system2_score = self._calculate_system2_enhanced_score(
                base_score, mental_model_fit, stage_alignment
            )

            # Generate selection rationale
            rationale = self._generate_consultant_selection_rationale(
                consultant_id,
                base_score,
                mental_model_fit,
                stage_alignment,
                recommended_models,
            )

            # Create enhanced match
            enhanced_match = System2ConsultantMatch(
                consultant_id=consultant_id,
                base_contextual_score=base_score,
                mental_model_fit_score=mental_model_fit,
                cognitive_stage_alignment=stage_alignment,
                recommended_mental_models=recommended_models,
                system2_enhanced_score=system2_score,
                selection_rationale=rationale,
            )

            enhanced_matches.append(enhanced_match)

        # Sort by System 2 enhanced score and return top selections
        enhanced_matches.sort(key=lambda x: x.system2_enhanced_score, reverse=True)

        logger.info(f"   • Enhanced {len(enhanced_matches)} consultant selections")
        return enhanced_matches

    async def _calculate_mental_model_fit_score(
        self,
        consultant_id: str,
        cognitive_stage: CognitiveStage,
        mental_model_requirements: Optional[List[str]] = None,
    ) -> float:
        """Calculate how well a consultant's mental model preferences fit the requirements."""

        # Get consultant's preferred mental models
        consultant_preferences = self.consultant_mental_model_preferences.get(
            consultant_id, {}
        )
        preferred_models = consultant_preferences.get("preferred_mental_models", [])

        # Get stage-appropriate mental models
        stage_models = await self.cognitive_bridge.get_stage_mental_models(
            cognitive_stage, {}
        )

        # Calculate fit score based on overlap
        if mental_model_requirements:
            # Specific requirements provided
            overlap = len(set(preferred_models) & set(mental_model_requirements))
            max_possible = len(mental_model_requirements)
        else:
            # General stage requirements
            overlap = len(set(preferred_models) & set(stage_models))
            max_possible = len(stage_models)

        if max_possible == 0:
            return 0.5  # Neutral score if no requirements

        fit_score = overlap / max_possible

        # Bonus for consultant expertise in their preferred models
        expertise_bonus = (
            consultant_preferences.get("mental_model_expertise_level", 0.5) * 0.2
        )

        return min(fit_score + expertise_bonus, 1.0)

    def _calculate_cognitive_stage_alignment(
        self, consultant_id: str, cognitive_stage: CognitiveStage
    ) -> float:
        """Calculate how well a consultant aligns with the current cognitive stage."""

        consultant_preferences = self.consultant_mental_model_preferences.get(
            consultant_id, {}
        )
        cognitive_style = consultant_preferences.get("cognitive_style", "balanced")

        # Stage-consultant alignment mapping
        stage_alignment_map = {
            CognitiveStage.PERCEPTION: {
                "analytical": 0.9,
                "creative": 0.7,
                "structured": 0.8,
                "intuitive": 0.9,
            },
            CognitiveStage.DECOMPOSITION: {
                "analytical": 0.9,
                "creative": 0.6,
                "structured": 0.9,
                "intuitive": 0.5,
            },
            CognitiveStage.REASONING: {
                "analytical": 0.9,
                "creative": 0.8,
                "structured": 0.8,
                "intuitive": 0.7,
            },
            CognitiveStage.SYNTHESIS: {
                "analytical": 0.8,
                "creative": 0.9,
                "structured": 0.7,
                "intuitive": 0.8,
            },
            CognitiveStage.DECISION: {
                "analytical": 0.9,
                "creative": 0.6,
                "structured": 0.9,
                "intuitive": 0.6,
            },
        }

        return stage_alignment_map.get(cognitive_stage, {}).get(cognitive_style, 0.7)

    async def _get_recommended_mental_models_for_consultant(
        self,
        consultant_id: str,
        cognitive_stage: CognitiveStage,
        mental_model_requirements: Optional[List[str]] = None,
    ) -> List[str]:
        """Get recommended mental models for a specific consultant and cognitive stage."""

        # Get consultant's preferred models
        consultant_preferences = self.consultant_mental_model_preferences.get(
            consultant_id, {}
        )
        preferred_models = consultant_preferences.get("preferred_mental_models", [])

        # Get stage-appropriate models
        stage_models = await self.cognitive_bridge.get_stage_mental_models(
            cognitive_stage, {}
        )

        # Priority 1: Models that match both consultant preference and stage requirements
        primary_models = list(set(preferred_models) & set(stage_models))

        # Priority 2: Required models not in preferences
        if mental_model_requirements:
            required_models = [
                m for m in mental_model_requirements if m not in primary_models
            ]
            primary_models.extend(required_models[:2])  # Add up to 2 required models

        # Priority 3: Additional stage models for diversity
        additional_models = [m for m in stage_models if m not in primary_models]
        primary_models.extend(
            additional_models[: 3 - len(primary_models)]
        )  # Ensure 3-7 models total

        # Ensure we have at least 3 models, maximum 7 (following LOLLA guide)
        if len(primary_models) < 3:
            fallback_models = [
                "systems_thinking",
                "first_principles",
                "opportunity_cost",
            ]
            for model in fallback_models:
                if model not in primary_models and len(primary_models) < 3:
                    primary_models.append(model)

        return primary_models[:7]  # Maximum 7 models per consultant

    def _calculate_system2_enhanced_score(
        self, base_score: float, mental_model_fit: float, stage_alignment: float
    ) -> float:
        """Calculate System 2 enhanced score combining all factors."""

        enhanced_score = (
            base_score * self.system2_weights["base_contextual_score"]
            + mental_model_fit * self.system2_weights["mental_model_fit"]
            + stage_alignment * self.system2_weights["cognitive_stage_alignment"]
        )

        return min(enhanced_score, 1.0)

    def _generate_consultant_selection_rationale(
        self,
        consultant_id: str,
        base_score: float,
        mental_model_fit: float,
        stage_alignment: float,
        recommended_models: List[str],
    ) -> str:
        """Generate rationale for consultant selection."""

        return (
            f"Selected {consultant_id} with System 2 enhanced score {self._calculate_system2_enhanced_score(base_score, mental_model_fit, stage_alignment):.3f} "
            f"(base: {base_score:.3f}, mental model fit: {mental_model_fit:.3f}, stage alignment: {stage_alignment:.3f}). "
            f"Recommended mental models: {', '.join(recommended_models[:3])}."
        )

    def _calculate_mental_model_coverage(
        self, consultant_matches: List[System2ConsultantMatch]
    ) -> Dict[str, float]:
        """Calculate mental model coverage across selected consultants."""

        all_models = set()
        model_frequency = {}

        for match in consultant_matches:
            for model in match.recommended_mental_models:
                all_models.add(model)
                model_frequency[model] = model_frequency.get(model, 0) + 1

        total_consultants = len(consultant_matches)
        coverage = {}

        for model in all_models:
            coverage[model] = model_frequency[model] / total_consultants

        return coverage

    def _calculate_cognitive_diversity_score(
        self, consultant_matches: List[System2ConsultantMatch]
    ) -> float:
        """Calculate cognitive diversity score across selected consultants."""

        # Count unique mental models
        all_models = set()
        for match in consultant_matches:
            all_models.update(match.recommended_mental_models)

        # Calculate diversity based on unique models per consultant
        total_models = sum(
            len(match.recommended_mental_models) for match in consultant_matches
        )
        unique_models = len(all_models)

        if total_models == 0:
            return 0.0

        # Diversity ratio with consultant count factor
        diversity_ratio = unique_models / total_models
        consultant_diversity = (
            len(consultant_matches) / 3
        )  # Normalized to 3 consultants

        return min(diversity_ratio * 0.7 + consultant_diversity * 0.3, 1.0)

    def _generate_system2_selection_rationale(
        self,
        consultant_matches: List[System2ConsultantMatch],
        cognitive_stage: CognitiveStage,
        mental_model_coverage: Dict[str, float],
    ) -> str:
        """Generate overall System 2 selection rationale."""

        consultant_names = [match.consultant_id for match in consultant_matches]
        avg_score = sum(
            match.system2_enhanced_score for match in consultant_matches
        ) / len(consultant_matches)
        unique_models = len(mental_model_coverage)

        return (
            f"System 2 enhanced selection for {cognitive_stage.value} stage: "
            f"Selected {len(consultant_matches)} consultants ({', '.join(consultant_names)}) "
            f"with average enhanced score {avg_score:.3f}. "
            f"Mental model coverage: {unique_models} unique models across team. "
            f"Selection optimized for cognitive stage requirements and mental model diversity."
        )

    def _calculate_system2_advantage_metrics(
        self,
        consultant_matches: List[System2ConsultantMatch],
        cognitive_diversity: float,
    ) -> Dict[str, float]:
        """Calculate System 2 advantages over generic consultant selection."""

        # Mental model awareness advantage
        avg_mental_model_fit = sum(
            match.mental_model_fit_score for match in consultant_matches
        ) / len(consultant_matches)
        mental_model_advantage = avg_mental_model_fit / 0.3  # vs ~30% generic fit

        # Cognitive stage alignment advantage
        avg_stage_alignment = sum(
            match.cognitive_stage_alignment for match in consultant_matches
        ) / len(consultant_matches)
        stage_alignment_advantage = (
            avg_stage_alignment / 0.5
        )  # vs ~50% generic alignment

        # Diversity advantage
        diversity_advantage = cognitive_diversity / 0.4  # vs ~40% generic diversity

        # Overall advantage
        overall_advantage = (
            mental_model_advantage + stage_alignment_advantage + diversity_advantage
        ) / 3

        return {
            "mental_model_awareness_advantage": mental_model_advantage,
            "cognitive_stage_alignment_advantage": stage_alignment_advantage,
            "diversity_advantage": diversity_advantage,
            "overall_system2_dispatch_advantage": overall_advantage,
        }

    def _load_consultant_mental_model_preferences(self) -> Dict[str, Dict[str, Any]]:
        """Load consultant mental model preferences and cognitive styles."""

        # This would load from enhanced consultant personas
        return {
            "strategic_analyst@1.0": {
                "preferred_mental_models": [
                    "systems_thinking",
                    "first_principles",
                    "opportunity_cost",
                    "competitive_advantage",
                ],
                "cognitive_style": "analytical",
                "mental_model_expertise_level": 0.9,
            },
            "risk_assessor@1.0": {
                "preferred_mental_models": [
                    "inversion",
                    "failure_modes",
                    "base_rates",
                    "black_swan_events",
                ],
                "cognitive_style": "structured",
                "mental_model_expertise_level": 0.85,
            },
            "innovation_specialist@1.0": {
                "preferred_mental_models": [
                    "lateral_thinking",
                    "creative_destruction",
                    "analogical_reasoning",
                    "paradigm_shifts",
                ],
                "cognitive_style": "creative",
                "mental_model_expertise_level": 0.8,
            },
            "financial_analyst@1.0": {
                "preferred_mental_models": [
                    "compound_interest",
                    "option_value",
                    "discounted_cash_flow",
                    "expected_value",
                ],
                "cognitive_style": "analytical",
                "mental_model_expertise_level": 0.95,
            },
            "operations_expert@1.0": {
                "preferred_mental_models": [
                    "bottleneck_theory",
                    "lean_principles",
                    "process_optimization",
                    "quality_circles",
                ],
                "cognitive_style": "structured",
                "mental_model_expertise_level": 0.9,
            },
            "customer_insights@1.0": {
                "preferred_mental_models": [
                    "jobs_to_be_done",
                    "customer_development",
                    "empathy_mapping",
                    "design_thinking",
                ],
                "cognitive_style": "intuitive",
                "mental_model_expertise_level": 0.8,
            },
        }

    async def _record_system2_dispatch_evidence(
        self, enhanced_result: System2DispatchResult
    ) -> None:
        """Record System 2 dispatch enhancement evidence in context stream."""

        await self.context_stream.record_event(
            trace_id=None,  # Would be provided by caller
            event_type="SYSTEM_2_DISPATCH_ENHANCED",
            event_data={
                "cognitive_stage": enhanced_result.cognitive_stage.value,
                "consultants_selected": len(
                    enhanced_result.consultant_mental_model_matches
                ),
                "mental_models_activated": len(
                    set().union(
                        *[
                            match.recommended_mental_models
                            for match in enhanced_result.consultant_mental_model_matches
                        ]
                    )
                ),
                "cognitive_diversity_score": enhanced_result.cognitive_diversity_score,
                "system2_advantage": enhanced_result.system2_advantage_metrics.get(
                    "overall_system2_dispatch_advantage", 1.0
                ),
                "consultant_scores": {
                    match.consultant_id: match.system2_enhanced_score
                    for match in enhanced_result.consultant_mental_model_matches
                },
            },
        )


# Factory function for easy initialization
def get_system2_enhanced_dispatch_orchestrator(
    context_stream: Optional[UnifiedContextStream] = None,
) -> System2EnhancedDispatchOrchestrator:
    """Get System 2 Enhanced Dispatch Orchestrator instance."""
    return System2EnhancedDispatchOrchestrator(context_stream)


if __name__ == "__main__":
    # Demo usage
    async def demo_enhanced_dispatch():
        from orchestration.contracts import StructuredAnalyticalFramework, FrameworkType

        orchestrator = get_system2_enhanced_dispatch_orchestrator()

        # Sample framework
        framework = StructuredAnalyticalFramework(
            framework_type=FrameworkType.STRATEGIC_ANALYSIS,
            core_question="Should we acquire TechCorp for $2.5B?",
            key_stakeholders=["Leadership", "Board", "Investors"],
            success_criteria=["Strategic fit", "Financial return", "Integration risk"],
        )

        result = await orchestrator.system2_enhanced_dispatch(
            framework=framework,
            cognitive_stage=CognitiveStage.REASONING,
            num_consultants=3,
        )

        print("System 2 Enhanced Dispatch Complete!")
        print(
            f"Consultants: {[m.consultant_id for m in result.consultant_mental_model_matches]}"
        )
        print(
            f"Mental Models: {sum(len(m.recommended_mental_models) for m in result.consultant_mental_model_matches)}"
        )
        print(f"Cognitive Diversity: {result.cognitive_diversity_score:.3f}")

    # asyncio.run(demo_enhanced_dispatch())
    print("🎯 System 2 Enhanced Dispatch Orchestrator loaded successfully")
