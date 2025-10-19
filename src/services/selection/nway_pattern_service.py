"""
METIS N-Way Pattern Service
Part of Selection Services Cluster - Focused on N-Way interaction pattern management

Extracted from model_selector.py N-Way pattern logic during Phase 5.2 decomposition.
Single Responsibility: Manage N-Way interaction patterns and enhance selections with synergistic combinations.
"""

import logging
import json
import re
import sys
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from src.services.contracts.selection_contracts import (
    INWayPatternService,
    NWayInteractionContract,
    SelectionResultContract,
    SelectionContextContract,
    NWayInteractionType,
)

# Import cognitive architecture loader
sys.path.append(
    str(Path(__file__).parent.parent.parent.parent / "cognitive_architecture")
)
from loader import CognitiveArchitectureLoader


class NWayPatternService(INWayPatternService):
    """
    Focused service for N-Way interaction pattern management and synergy detection
    Clean extraction from model_selector.py N-Way pattern functionality
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # N-Way interaction patterns storage
        self.interaction_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self.synergistic_combinations: Dict[str, List[str]] = {}
        self.conflicting_combinations: Dict[str, List[str]] = {}

        # Pattern scoring weights
        self.pattern_weights = {
            "synergistic": 1.0,
            "conflicting": -0.5,
            "dependent": 0.7,
            "independent": 0.0,
        }

        # Load existing patterns from database
        self._load_nway_patterns()

        self.logger.info("🔗 NWayPatternService initialized")

    async def get_models_for_consultant(self, consultant_name: str) -> List[str]:
        """
        Get mental models that are most relevant for a specific consultant type.

        Args:
            consultant_name: Name of the consultant (e.g., "strategic_analyst")

        Returns:
            List of model names relevant to this consultant
        """
        try:
            # Map consultant names to priority model types (strategy-based selection)
            consultant_priority_models = {
                "strategic_analyst": [
                    "systems_thinking",
                    "first_principles_thinking",
                    "outside_view",
                    "base_rates",
                    "opportunity_cost",
                    "value_chain_analysis",
                    "scenario_planning",
                    "swot_analysis",
                    "competitive_advantage",
                    "market_dynamics",
                ],
                "risk_assessor": [
                    "base_rates",
                    "outside_view",
                    "black_swan_theory",
                    "monte_carlo_simulation",
                    "probability_theory",
                    "expected_value",
                    "worst_case_scenario",
                    "stress_testing",
                    "correlation_analysis",
                    "tail_risk_assessment",
                ],
                "market_researcher": [
                    "behavioral_economics",
                    "survey_design",
                    "statistical_sampling",
                    "pattern_recognition",
                    "demographic_analysis",
                    "trend_analysis",
                    "consumer_psychology",
                    "market_segmentation",
                    "competitive_intelligence",
                    "data_triangulation",
                ],
                "financial_analyst": [
                    "discounted_cash_flow",
                    "financial_modeling",
                    "ratio_analysis",
                    "sensitivity_analysis",
                    "monte_carlo_simulation",
                    "expected_value",
                    "opportunity_cost",
                    "risk_adjusted_returns",
                    "portfolio_theory",
                    "correlation_analysis",
                ],
                "implementation_specialist": [
                    "project_management",
                    "critical_path_method",
                    "lean_methodology",
                    "agile_principles",
                    "change_management",
                    "stakeholder_mapping",
                    "process_optimization",
                    "resource_allocation",
                    "timeline_analysis",
                    "execution_planning",
                ],
                "technology_advisor": [
                    "systems_thinking",
                    "complexity_theory",
                    "technology_adoption_curve",
                    "scalability_analysis",
                    "architecture_patterns",
                    "technical_debt",
                    "performance_optimization",
                    "security_frameworks",
                    "innovation_frameworks",
                    "digital_transformation",
                ],
                "innovation_consultant": [
                    "design_thinking",
                    "creative_problem_solving",
                    "blue_ocean_strategy",
                    "disruptive_innovation",
                    "customer_journey_mapping",
                    "value_proposition_canvas",
                    "lean_startup_methodology",
                    "rapid_prototyping",
                    "innovation_frameworks",
                    "technology_scouting",
                ],
                "crisis_manager": [
                    "crisis_management_frameworks",
                    "rapid_decision_making",
                    "stakeholder_communication",
                    "business_continuity",
                    "risk_mitigation",
                    "emergency_protocols",
                    "resource_mobilization",
                    "damage_assessment",
                    "recovery_planning",
                    "communication_strategy",
                ],
                "operations_expert": [
                    "process_optimization",
                    "lean_methodology",
                    "six_sigma",
                    "supply_chain_management",
                    "quality_control",
                    "efficiency_analysis",
                    "workflow_design",
                    "capacity_planning",
                    "performance_metrics",
                    "continuous_improvement",
                ],
            }

            # Get priority models for this consultant type
            priority_models = consultant_priority_models.get(
                consultant_name,
                [
                    "systems_thinking",
                    "first_principles_thinking",
                    "base_rates",
                    "outside_view",
                ],
            )

            # Find additional models from loaded patterns that are synergistic with priority models
            additional_models = []
            for priority_model in priority_models[
                :3
            ]:  # Use top 3 priority models as seeds
                if priority_model in self.interaction_patterns:
                    for interaction in self.interaction_patterns[priority_model]:
                        target_model = interaction.get("target_model")
                        if (
                            target_model
                            and target_model not in priority_models
                            and target_model not in additional_models
                            and interaction.get("relationship_type") == "synergistic"
                            and interaction.get("strength_score", 0) > 0.7
                        ):
                            additional_models.append(target_model)

            # Combine priority models with additional synergistic models
            final_models = (
                priority_models[:7] + additional_models[:3]
            )  # 7 priority + 3 synergistic

            self.logger.info(
                f"✅ Selected {len(final_models)} models for {consultant_name}"
            )
            return final_models[:10]  # Limit to top 10

        except Exception as e:
            self.logger.error(
                f"❌ Failed to get models for consultant {consultant_name}: {e}"
            )
            # Return some default models
            return [
                "systems_thinking",
                "first_principles_thinking",
                "base_rates",
                "outside_view",
            ]

    async def detect_nway_interactions(
        self, selected_models: List[str], context: SelectionContextContract
    ) -> List[NWayInteractionContract]:
        """
        Core service method: Detect N-Way interaction patterns among selected models
        Clean, focused implementation with single responsibility
        """
        try:
            if len(selected_models) < 2:
                return []  # No interactions possible with less than 2 models

            detected_interactions = []

            # Check all pairs of selected models for interactions
            for i, model1 in enumerate(selected_models):
                for j, model2 in enumerate(selected_models[i + 1 :], i + 1):
                    # Detect interaction between model1 and model2
                    interaction = await self._detect_model_pair_interaction(
                        model1, model2, context
                    )

                    if interaction:
                        detected_interactions.append(interaction)

            # Check for multi-model patterns (triplets, etc.)
            if len(selected_models) >= 3:
                multi_interactions = await self._detect_multi_model_patterns(
                    selected_models, context
                )
                detected_interactions.extend(multi_interactions)

            self.logger.info(
                f"✅ Detected {len(detected_interactions)} N-Way interactions"
            )
            return detected_interactions

        except Exception as e:
            self.logger.error(f"❌ N-Way interaction detection failed: {e}")
            return []

    async def enhance_with_nway_patterns(
        self,
        selection_result: SelectionResultContract,
        context: SelectionContextContract,
    ) -> SelectionResultContract:
        """
        Enhance selection result with N-Way pattern analysis and scoring adjustments
        """
        try:
            selected_models = selection_result.selected_models

            # Detect interactions
            interactions = await self.detect_nway_interactions(selected_models, context)

            if not interactions:
                # No interactions detected, return original result
                return selection_result

            # Calculate synergy score for the combination
            synergy_score = self._calculate_combination_synergy_score(interactions)

            # Enhance model scores with synergy bonuses
            enhanced_scores = self._apply_synergy_bonuses(
                selection_result.model_scores, interactions, synergy_score
            )

            # Update selection metadata
            enhanced_metadata = {
                **selection_result.selection_metadata,
                "nway_interactions_detected": len(interactions),
                "synergy_score": synergy_score,
                "pattern_enhancement_applied": True,
                "interaction_types": list(
                    set(i.relationship_type for i in interactions)
                ),
            }

            # Create enhanced result
            enhanced_result = SelectionResultContract(
                engagement_id=selection_result.engagement_id,
                selected_models=selection_result.selected_models,
                model_scores=enhanced_scores,
                selection_source=f"{selection_result.selection_source}_nway_enhanced",
                strategy_used=selection_result.strategy_used,
                models_evaluated=selection_result.models_evaluated,
                selection_metadata=enhanced_metadata,
                total_selection_time_ms=selection_result.total_selection_time_ms,
                cognitive_load_assessment=selection_result.cognitive_load_assessment,
                selection_timestamp=selection_result.selection_timestamp,
                service_version="v5_modular_nway",
            )

            self.logger.info(
                f"✅ Enhanced selection with {len(interactions)} N-Way patterns (synergy: {synergy_score:.3f})"
            )
            return enhanced_result

        except Exception as e:
            self.logger.error(f"❌ N-Way pattern enhancement failed: {e}")
            return selection_result  # Return original on failure

    async def _detect_model_pair_interaction(
        self, model1: str, model2: str, context: SelectionContextContract
    ) -> Optional[NWayInteractionContract]:
        """Detect interaction between two specific models"""
        try:
            # Check direct synergistic patterns
            synergy = self._get_model_pair_synergy(model1, model2)
            if synergy > 0:
                return self._create_interaction_contract(
                    model1, model2, NWayInteractionType.SYNERGISTIC, synergy, context
                )

            # Check conflicting patterns
            conflict = self._get_model_pair_conflict(model1, model2)
            if conflict > 0:
                return self._create_interaction_contract(
                    model1, model2, NWayInteractionType.CONFLICTING, conflict, context
                )

            # Check dependency patterns
            dependency = self._get_model_pair_dependency(model1, model2)
            if dependency > 0:
                return self._create_interaction_contract(
                    model1, model2, NWayInteractionType.DEPENDENT, dependency, context
                )

            return None  # No significant interaction detected

        except Exception as e:
            self.logger.warning(
                f"⚠️ Pair interaction detection failed for {model1}-{model2}: {e}"
            )
            return None

    async def _detect_multi_model_patterns(
        self, selected_models: List[str], context: SelectionContextContract
    ) -> List[NWayInteractionContract]:
        """Detect multi-model interaction patterns (3+ models)"""
        try:
            multi_interactions = []

            # Look for known triplet patterns
            if len(selected_models) >= 3:
                triplet_patterns = self._find_triplet_patterns(selected_models)
                for pattern in triplet_patterns:
                    interaction = self._create_multi_interaction_contract(
                        pattern, context
                    )
                    if interaction:
                        multi_interactions.append(interaction)

            # Look for framework cascade patterns
            cascade_patterns = self._find_cascade_patterns(selected_models)
            for pattern in cascade_patterns:
                interaction = self._create_cascade_interaction_contract(
                    pattern, context
                )
                if interaction:
                    multi_interactions.append(interaction)

            return multi_interactions

        except Exception as e:
            self.logger.warning(f"⚠️ Multi-model pattern detection failed: {e}")
            return []

    def _get_model_pair_synergy(self, model1: str, model2: str) -> float:
        """Get synergy score between two specific models"""
        # Check direct synergistic relationship
        if model1 in self.synergistic_combinations:
            if model2 in self.synergistic_combinations[model1]:
                # Find the pattern and return its strength
                for pattern in self.interaction_patterns.get(model1, []):
                    if (
                        pattern.get("target_model") == model2
                        and pattern.get("relationship_type") == "synergistic"
                    ):
                        return pattern.get("strength_score", 0.0) * pattern.get(
                            "confidence_score", 1.0
                        )

        # Check reverse relationship
        if model2 in self.synergistic_combinations:
            if model1 in self.synergistic_combinations[model2]:
                for pattern in self.interaction_patterns.get(model2, []):
                    if (
                        pattern.get("target_model") == model1
                        and pattern.get("relationship_type") == "synergistic"
                    ):
                        return pattern.get("strength_score", 0.0) * pattern.get(
                            "confidence_score", 1.0
                        )

        return 0.0

    def _get_model_pair_conflict(self, model1: str, model2: str) -> float:
        """Get conflict score between two specific models"""
        # Check direct conflicting relationship
        if model1 in self.conflicting_combinations:
            if model2 in self.conflicting_combinations[model1]:
                for pattern in self.interaction_patterns.get(model1, []):
                    if (
                        pattern.get("target_model") == model2
                        and pattern.get("relationship_type") == "conflicting"
                    ):
                        return pattern.get("strength_score", 0.0) * pattern.get(
                            "confidence_score", 1.0
                        )

        # Check reverse relationship
        if model2 in self.conflicting_combinations:
            if model1 in self.conflicting_combinations[model2]:
                for pattern in self.interaction_patterns.get(model2, []):
                    if (
                        pattern.get("target_model") == model1
                        and pattern.get("relationship_type") == "conflicting"
                    ):
                        return pattern.get("strength_score", 0.0) * pattern.get(
                            "confidence_score", 1.0
                        )

        return 0.0

    def _get_model_pair_dependency(self, model1: str, model2: str) -> float:
        """Get dependency score between two specific models"""
        # Simplified dependency detection based on model names/types
        dependent_keywords = [
            ("systems_thinking", "stakeholder_analysis"),
            ("porter_5_forces", "competitive_intelligence"),
            ("lean_startup", "customer_development"),
            ("strategic_framework", "implementation_planning"),
        ]

        for primary, secondary in dependent_keywords:
            if primary in model1.lower() and secondary in model2.lower():
                return 0.7
            if secondary in model1.lower() and primary in model2.lower():
                return 0.7

        return 0.0

    def _find_triplet_patterns(
        self, selected_models: List[str]
    ) -> List[Dict[str, Any]]:
        """Find known triplet synergy patterns"""
        triplet_patterns = []

        # Known high-value triplets
        known_triplets = [
            {
                "models": [
                    "systems_thinking",
                    "stakeholder_analysis",
                    "implementation_planning",
                ],
                "synergy_type": "holistic_strategic_analysis",
                "strength": 0.85,
            },
            {
                "models": [
                    "porter_5_forces",
                    "competitive_intelligence",
                    "market_positioning",
                ],
                "synergy_type": "competitive_strategic_framework",
                "strength": 0.80,
            },
            {
                "models": ["lean_startup", "customer_development", "agile_methodology"],
                "synergy_type": "innovation_execution_framework",
                "strength": 0.75,
            },
        ]

        # Check if any triplet matches selected models
        for triplet in known_triplets:
            matches = 0
            for model in triplet["models"]:
                if any(model in selected.lower() for selected in selected_models):
                    matches += 1

            if matches >= 2:  # Partial match
                triplet_patterns.append(
                    {
                        **triplet,
                        "match_ratio": matches / len(triplet["models"]),
                        "matched_models": [
                            m
                            for m in selected_models
                            if any(t in m.lower() for t in triplet["models"])
                        ],
                    }
                )

        return triplet_patterns

    def _find_cascade_patterns(
        self, selected_models: List[str]
    ) -> List[Dict[str, Any]]:
        """Find framework cascade patterns (where one framework feeds into another)"""
        cascade_patterns = []

        # Known cascade sequences
        known_cascades = [
            {
                "sequence": [
                    "problem_definition",
                    "root_cause_analysis",
                    "solution_design",
                ],
                "cascade_type": "analytical_progression",
                "strength": 0.75,
            },
            {
                "sequence": [
                    "market_analysis",
                    "competitive_positioning",
                    "go_to_market_strategy",
                ],
                "cascade_type": "market_entry_progression",
                "strength": 0.80,
            },
        ]

        for cascade in known_cascades:
            sequence_matches = []
            for step in cascade["sequence"]:
                for model in selected_models:
                    if step.replace("_", " ") in model.lower():
                        sequence_matches.append(model)
                        break

            if len(sequence_matches) >= 2:
                cascade_patterns.append(
                    {
                        **cascade,
                        "matched_sequence": sequence_matches,
                        "sequence_completeness": len(sequence_matches)
                        / len(cascade["sequence"]),
                    }
                )

        return cascade_patterns

    def _create_interaction_contract(
        self,
        model1: str,
        model2: str,
        interaction_type: NWayInteractionType,
        strength: float,
        context: SelectionContextContract,
    ) -> NWayInteractionContract:
        """Create interaction contract for model pair"""
        interaction_id = f"{model1}_{model2}_{interaction_type.value}"

        # Generate explanation based on interaction type
        if interaction_type == NWayInteractionType.SYNERGISTIC:
            explanation = (
                f"{model1} and {model2} create synergistic insights when combined"
            )
            enhancement_type = "cross_framework_reinforcement"
        elif interaction_type == NWayInteractionType.CONFLICTING:
            explanation = (
                f"{model1} and {model2} may produce conflicting recommendations"
            )
            enhancement_type = "conflict_resolution_required"
        elif interaction_type == NWayInteractionType.DEPENDENT:
            explanation = (
                f"{model2} depends on insights from {model1} for optimal effectiveness"
            )
            enhancement_type = "sequential_application"
        else:
            explanation = f"{model1} and {model2} operate independently"
            enhancement_type = "parallel_application"

        return NWayInteractionContract(
            interaction_id=interaction_id,
            source_model=model1,
            target_model=model2,
            relationship_type=interaction_type.value,
            strength_score=strength,
            confidence_score=0.8,  # Default confidence
            explanation=explanation,
            context_conditions=context.business_context,
            instructional_cue=f"Apply {enhancement_type} when using these models together",
            enhancement_type=enhancement_type,
            validation_timestamp=datetime.utcnow(),
            service_version="v5_modular",
        )

    def _create_multi_interaction_contract(
        self, pattern: Dict[str, Any], context: SelectionContextContract
    ) -> Optional[NWayInteractionContract]:
        """Create interaction contract for multi-model pattern"""
        try:
            matched_models = pattern.get("matched_models", [])
            if len(matched_models) < 2:
                return None

            interaction_id = f"triplet_{pattern.get('synergy_type', 'unknown')}"

            return NWayInteractionContract(
                interaction_id=interaction_id,
                source_model=matched_models[0],
                target_model=", ".join(matched_models[1:]),
                relationship_type=NWayInteractionType.SYNERGISTIC.value,
                strength_score=pattern.get("strength", 0.7)
                * pattern.get("match_ratio", 1.0),
                confidence_score=pattern.get("match_ratio", 0.8),
                explanation=f"Multi-model {pattern.get('synergy_type', 'pattern')} creates enhanced analytical depth",
                context_conditions=context.business_context,
                instructional_cue=f"Apply models in complementary fashion for {pattern.get('synergy_type', 'analysis')}",
                enhancement_type="multi_model_synergy",
                validation_timestamp=datetime.utcnow(),
                service_version="v5_modular",
            )

        except Exception as e:
            self.logger.warning(f"⚠️ Multi-interaction contract creation failed: {e}")
            return None

    def _create_cascade_interaction_contract(
        self, pattern: Dict[str, Any], context: SelectionContextContract
    ) -> Optional[NWayInteractionContract]:
        """Create interaction contract for cascade pattern"""
        try:
            matched_sequence = pattern.get("matched_sequence", [])
            if len(matched_sequence) < 2:
                return None

            interaction_id = f"cascade_{pattern.get('cascade_type', 'unknown')}"

            return NWayInteractionContract(
                interaction_id=interaction_id,
                source_model=matched_sequence[0],
                target_model=", ".join(matched_sequence[1:]),
                relationship_type=NWayInteractionType.DEPENDENT.value,
                strength_score=pattern.get("strength", 0.7)
                * pattern.get("sequence_completeness", 1.0),
                confidence_score=pattern.get("sequence_completeness", 0.8),
                explanation=f"Cascade pattern {pattern.get('cascade_type', 'progression')} creates sequential analytical flow",
                context_conditions=context.business_context,
                instructional_cue=f"Apply models in sequence for {pattern.get('cascade_type', 'progressive analysis')}",
                enhancement_type="cascade_progression",
                validation_timestamp=datetime.utcnow(),
                service_version="v5_modular",
            )

        except Exception as e:
            self.logger.warning(f"⚠️ Cascade interaction contract creation failed: {e}")
            return None

    def _calculate_combination_synergy_score(
        self, interactions: List[NWayInteractionContract]
    ) -> float:
        """Calculate overall synergy score for model combination"""
        if not interactions:
            return 0.0

        total_synergy = 0.0
        total_weight = 0.0

        for interaction in interactions:
            relationship_type = interaction.relationship_type
            weight = self.pattern_weights.get(relationship_type, 0.0)
            synergy_contribution = (
                interaction.strength_score * weight * interaction.confidence_score
            )

            total_synergy += synergy_contribution
            total_weight += abs(weight)

        if total_weight > 0:
            return total_synergy / total_weight
        else:
            return 0.0

    def _apply_synergy_bonuses(
        self,
        model_scores: List,
        interactions: List[NWayInteractionContract],
        synergy_score: float,
    ) -> List:
        """Apply synergy bonuses to model scores based on interactions"""
        try:
            enhanced_scores = []

            # Get models involved in interactions
            interacting_models = set()
            for interaction in interactions:
                interacting_models.add(interaction.source_model)
                interacting_models.add(interaction.target_model)

            # Apply bonuses to interacting models
            for score in model_scores:
                if hasattr(score, "model_id") and score.model_id in interacting_models:
                    # Apply synergy bonus
                    bonus = synergy_score * 0.15  # 15% weight for synergy

                    # Create enhanced score (assuming it has the required attributes)
                    enhanced_score = type(score)(
                        model_id=score.model_id,
                        total_score=min(1.0, score.total_score + bonus),
                        component_scores={
                            **getattr(score, "component_scores", {}),
                            "nway_synergy": bonus,
                        },
                        rationale=getattr(score, "rationale", "")
                        + f" | N-Way synergy bonus: +{bonus:.3f}",
                        confidence=getattr(score, "confidence", 0.8),
                        risk_factors=getattr(score, "risk_factors", []),
                        scoring_timestamp=getattr(
                            score, "scoring_timestamp", datetime.utcnow()
                        ),
                        service_version="v5_modular_nway",
                    )
                    enhanced_scores.append(enhanced_score)
                else:
                    enhanced_scores.append(score)

            return enhanced_scores

        except Exception as e:
            self.logger.warning(f"⚠️ Synergy bonus application failed: {e}")
            return model_scores

    def _load_nway_patterns(self):
        """Load N-Way interaction patterns from cognitive architecture"""
        try:
            # Try to load from enhanced cognitive architecture first
            if self._load_cognitive_architecture_patterns():
                return

            # Fallback to legacy database files
            if self._load_legacy_db_patterns():
                return

            # Final fallback to defaults
            self._load_default_patterns()

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load N-Way patterns: {e}")
            self._load_default_patterns()

    def _load_cognitive_architecture_patterns(self) -> bool:
        """Load patterns from enhanced cognitive architecture YAML files"""
        try:
            # Get project root
            project_root = Path(__file__).parent.parent.parent.parent
            cognitive_arch_path = project_root / "cognitive_architecture"

            if not cognitive_arch_path.exists():
                self.logger.info("⚠️ Cognitive architecture path not found")
                return False

            # Load cognitive architecture
            loader = CognitiveArchitectureLoader(str(cognitive_arch_path))
            architecture = loader.load_architecture()

            loaded_patterns = 0
            synergistic_pairs = 0
            conflicting_pairs = 0

            # Extract patterns from all clusters
            clusters_data = architecture.get("clusters_data", {})

            for cluster_name, cluster_data in clusters_data.items():
                if not isinstance(cluster_data, dict):
                    continue

                # Process each NWAY in the cluster
                for nway_key, nway_data in cluster_data.items():
                    if not nway_key.startswith("NWAY_") or not isinstance(
                        nway_data, dict
                    ):
                        continue

                    # Extract models and interactions
                    models = nway_data.get("models", [])
                    interactions = nway_data.get("interactions", {})

                    # Build synergistic patterns from interactions
                    for interaction_type, interaction_desc in interactions.items():
                        if (
                            isinstance(interaction_desc, str)
                            and "+" in interaction_desc
                        ):
                            # Parse synergistic combinations like "agile + lean_startup = adaptive execution"
                            parts = interaction_desc.split("=")[0].strip()
                            if "+" in parts:
                                components = [c.strip() for c in parts.split("+")]
                                if len(components) >= 2:
                                    source_model = self._normalize_model_name(
                                        components[0]
                                    )
                                    target_model = self._normalize_model_name(
                                        components[1]
                                    )

                                    if source_model and target_model:
                                        # Store interaction pattern
                                        if (
                                            source_model
                                            not in self.interaction_patterns
                                        ):
                                            self.interaction_patterns[source_model] = []

                                        self.interaction_patterns[source_model].append(
                                            {
                                                "target_model": target_model,
                                                "relationship_type": "synergistic",
                                                "strength_score": 0.9,  # Enhanced patterns have high confidence
                                                "confidence_score": 0.9,
                                                "explanation": interaction_desc,
                                                "instructional_cue": f"Enhanced cognitive pattern from {nway_data.get('title', 'Unknown')}",
                                            }
                                        )

                                        # Build synergistic combinations
                                        if (
                                            source_model
                                            not in self.synergistic_combinations
                                        ):
                                            self.synergistic_combinations[
                                                source_model
                                            ] = []
                                        self.synergistic_combinations[
                                            source_model
                                        ].append(target_model)

                                        synergistic_pairs += 1
                                        loaded_patterns += 1

                    # Extract model-to-model relationships within NWAY
                    for i, model1 in enumerate(models):
                        for j, model2 in enumerate(models):
                            if i != j:
                                norm_model1 = self._normalize_model_name(model1)
                                norm_model2 = self._normalize_model_name(model2)

                                if norm_model1 and norm_model2:
                                    # Store bidirectional synergistic relationship
                                    if norm_model1 not in self.interaction_patterns:
                                        self.interaction_patterns[norm_model1] = []

                                    self.interaction_patterns[norm_model1].append(
                                        {
                                            "target_model": norm_model2,
                                            "relationship_type": "synergistic",
                                            "strength_score": 0.8,
                                            "confidence_score": 0.8,
                                            "explanation": f"Models in same NWAY: {nway_data.get('title', 'Unknown')}",
                                            "instructional_cue": "Enhanced NWAY cluster synergy",
                                        }
                                    )

                                    if norm_model1 not in self.synergistic_combinations:
                                        self.synergistic_combinations[norm_model1] = []
                                    if (
                                        norm_model2
                                        not in self.synergistic_combinations[
                                            norm_model1
                                        ]
                                    ):
                                        self.synergistic_combinations[
                                            norm_model1
                                        ].append(norm_model2)
                                        synergistic_pairs += 1
                                        loaded_patterns += 1

            if loaded_patterns > 0:
                self.logger.info(
                    f"✅ Enhanced N-Way patterns loaded from cognitive architecture: {loaded_patterns} interactions"
                )
                self.logger.info(f"   - Synergistic pairs: {synergistic_pairs}")
                self.logger.info(f"   - Conflicting pairs: {conflicting_pairs}")
                self.logger.info("   - Source: Enhanced YAML cognitive architecture")
                return True
            else:
                self.logger.info("⚠️ No patterns found in cognitive architecture")
                return False

        except Exception as e:
            self.logger.warning(
                f"⚠️ Failed to load cognitive architecture patterns: {e}"
            )
            return False

    def _load_legacy_db_patterns(self) -> bool:
        """Load patterns from legacy database files (fallback)"""
        try:
            # Get project root
            project_root = Path(__file__).parent.parent.parent.parent
            db_path = project_root / "db"

            if not db_path.exists():
                return False

            loaded_patterns = 0
            synergistic_pairs = 0
            conflicting_pairs = 0

            # Load patterns from database files
            for db_file in sorted(db_path.glob("*.json")):
                try:
                    with open(db_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Extract knowledge relationships
                    relationships = data.get("knowledge_relationships", [])

                    for rel in relationships:
                        source_model = self._normalize_model_name(
                            rel.get("source_ke_name", "")
                        )
                        target_model = self._normalize_model_name(
                            rel.get("target_ke_name", "")
                        )

                        if not source_model or not target_model:
                            continue

                        # Store pattern
                        if source_model not in self.interaction_patterns:
                            self.interaction_patterns[source_model] = []

                        self.interaction_patterns[source_model].append(
                            {
                                "target_model": target_model,
                                "relationship_type": rel.get(
                                    "relationship_type", "synergistic"
                                ),
                                "strength_score": rel.get("strength_score", 0.8),
                                "confidence_score": rel.get("confidence_score", 0.8),
                                "explanation": rel.get("explanation", ""),
                                "instructional_cue": rel.get("instructional_cue", ""),
                            }
                        )

                        # Build combination indexes
                        if rel.get("relationship_type") == "synergistic":
                            if source_model not in self.synergistic_combinations:
                                self.synergistic_combinations[source_model] = []
                            self.synergistic_combinations[source_model].append(
                                target_model
                            )
                            synergistic_pairs += 1
                        elif rel.get("relationship_type") == "conflicting":
                            if source_model not in self.conflicting_combinations:
                                self.conflicting_combinations[source_model] = []
                            self.conflicting_combinations[source_model].append(
                                target_model
                            )
                            conflicting_pairs += 1

                        loaded_patterns += 1

                except Exception:
                    # Skip corrupted files silently
                    continue

            if loaded_patterns > 0:
                self.logger.info(
                    f"✅ Legacy N-Way patterns loaded: {loaded_patterns} interactions"
                )
                self.logger.info(f"   - Synergistic pairs: {synergistic_pairs}")
                self.logger.info(f"   - Conflicting pairs: {conflicting_pairs}")
                return True

            return False

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load legacy database patterns: {e}")
            return False

    def _load_default_patterns(self):
        """Load default N-Way patterns when database is unavailable"""
        # Default synergistic combinations
        self.synergistic_combinations = {
            "systems_thinking": ["stakeholder_analysis", "implementation_planning"],
            "porter_5_forces": ["competitive_intelligence", "market_positioning"],
            "lean_startup": ["customer_development", "agile_methodology"],
            "strategic_framework": ["implementation_planning", "performance_metrics"],
        }

        # Default conflicting combinations
        self.conflicting_combinations = {
            "waterfall_methodology": ["agile_methodology", "lean_startup"],
            "cost_leadership": ["differentiation_strategy"],
            "centralized_control": ["decentralized_autonomy"],
        }

        self.logger.info("✅ Default N-Way patterns loaded")

    def _normalize_model_name(self, name: str) -> str:
        """Normalize model names to match catalog IDs"""
        if not name:
            return ""

        # Convert to lowercase and replace special characters
        normalized = name.lower()
        normalized = re.sub(r"[^a-z0-9]", "_", normalized)
        normalized = re.sub(r"_+", "_", normalized)
        normalized = normalized.strip("_")

        return normalized

    async def get_service_health(self) -> Dict[str, Any]:
        """Return service health status"""
        return {
            "service_name": "NWayPatternService",
            "status": "healthy",
            "version": "v5_modular",
            "capabilities": [
                "nway_interaction_detection",
                "synergy_scoring",
                "multi_model_pattern_recognition",
                "cascade_pattern_detection",
                "selection_enhancement",
            ],
            "patterns_loaded": len(self.interaction_patterns),
            "synergistic_combinations": len(self.synergistic_combinations),
            "conflicting_combinations": len(self.conflicting_combinations),
            "pattern_weights": self.pattern_weights,
            "last_health_check": datetime.utcnow().isoformat(),
        }


# Global service instance for dependency injection
_nway_pattern_service: Optional[NWayPatternService] = None


def get_nway_pattern_service() -> NWayPatternService:
    """Get or create global N-Way pattern service instance"""
    global _nway_pattern_service

    if _nway_pattern_service is None:
        _nway_pattern_service = NWayPatternService()

    return _nway_pattern_service
