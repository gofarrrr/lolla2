"""
METIS V5 Consultant Selection Service
====================================

Extracted from monolithic optimal_consultant_engine.py (lines 1897-2074).
Handles consultant selection logic including multiple selection strategies.

Part of the Great Refactoring: Clean separation of consultant selection concerns.
"""

from typing import Dict, List, Optional, Any

# Import our new contracts
from ..contracts import (
    QueryClassificationResult,
    ConsultantSelectionResult,
    ConsultantCandidate,
    ConsultantRole,
    EngagementRequest,
)

# Preserve compatibility with existing blueprint structure
from dataclasses import dataclass, field


@dataclass
class ConsultantBlueprint:
    """Legacy consultant blueprint structure - preserved for compatibility"""

    consultant_id: str
    name: str
    specialization: str
    expertise: str
    persona_prompt: str
    stable_frameworks: List[str]
    adaptive_triggers: List[str] = field(default_factory=list)
    effectiveness_score: float = 0.8


class ConsultantSelectionService:
    """
    Stateless service for consultant selection and matching.

    Extracted from OptimalConsultantEngine to follow Single Responsibility Principle.
    Handles multiple selection strategies: rules-based, predictive, pattern-based, and fallback.
    """

    def __init__(
        self, consultant_blueprints: Optional[Dict[str, ConsultantBlueprint]] = None
    ):
        """Initialize the consultant selection service"""
        self.consultant_blueprints = consultant_blueprints or {}
        self.routing_patterns = {}

        # Optional advanced selection systems (preserved from original)
        self.rules_engine_enabled = False
        self.predictive_enabled = False
        self.rules_engine = None
        self.predictive_selector = None

        # Initialize CQA Score Service for quality-driven selection (Operation Awakening)
        try:
            from src.services.cqa_score_service import get_cqa_score_service
            self.cqa_service = get_cqa_score_service()
            self.cqa_enabled = True
            print("✅ ConsultantSelectionService: CQA feedback loop enabled")
        except Exception as e:
            self.cqa_service = None
            self.cqa_enabled = False
            print(f"⚠️ ConsultantSelectionService: CQA service unavailable: {e}")

        print("✅ ConsultantSelectionService: Initialized with selection strategies")

    async def select_consultants(
        self,
        classification_result: QueryClassificationResult,
        request: EngagementRequest,
        consultant_blueprints: Optional[Dict[str, ConsultantBlueprint]] = None,
    ) -> ConsultantSelectionResult:
        """
        Main entry point: Select optimal consultants based on classification.

        Args:
            classification_result: Result from query classification
            request: Original engagement request
            consultant_blueprints: Available consultant blueprints (optional override)

        Returns:
            ConsultantSelectionResult with selected consultants and metadata
        """
        # Use provided blueprints or fall back to service instance
        blueprints = consultant_blueprints or self.consultant_blueprints
        if not blueprints:
            raise ValueError("No consultant blueprints available for selection")

        # Extract query for logging/context
        query = request.query

        # Convert our new contracts back to legacy format for compatibility
        legacy_classification = self._convert_to_legacy_classification(
            classification_result
        )

        # Execute selection using the extracted optimal trio logic
        selected_consultants = await self._select_optimal_trio(
            legacy_classification, blueprints, query
        )

        # Convert results back to new contract format
        consultant_candidates = self._convert_to_candidates(selected_consultants)

        # Determine N-Way clusters (placeholder for now - can be enhanced)
        nway_clusters = self._determine_nway_clusters(classification_result)

        return ConsultantSelectionResult(
            selected_consultants=consultant_candidates,
            nway_clusters=nway_clusters,
            selection_strategy=self._get_selection_strategy_used(),
            total_candidates_evaluated=len(blueprints),
            selection_reasoning=self._build_selection_reasoning(
                selected_consultants, classification_result
            ),
        )

    async def _select_optimal_trio(
        self, classification, blueprints: Dict[str, ConsultantBlueprint], query: str
    ) -> List[Dict[str, Any]]:
        """
        Core consultant selection logic - extracted from optimal_consultant_engine.py lines 1897-2035.
        Modified to work as a stateless service.
        """
        selected_ids = []
        selection_reason = ""
        self._current_query = query  # Store for context

        # Strategy 0: Use rules-based selection if available
        if self.rules_engine_enabled and self.rules_engine:
            try:
                selections = await self._try_rules_selection(classification, blueprints)
                if selections:
                    return selections
            except Exception as e:
                print(f"⚠️ Rules engine selection failed: {e}")

        # Strategy 1: Use predictive selection if available
        if self.predictive_enabled and self.predictive_selector:
            try:
                selected_ids, selection_reason = await self._try_predictive_selection(
                    classification, blueprints
                )
                if selected_ids:
                    print(f"🤖 Using predictive selection: {selected_ids}")
            except Exception as e:
                print(f"⚠️ Predictive selection failed: {e}")

        # Strategy 2: Use routing pattern if no predictive selection
        if (
            not selected_ids
            and hasattr(classification, "routing_pattern")
            and classification.routing_pattern
        ):
            selected_ids, selection_reason = self._try_pattern_selection(classification)

        # Strategy 3: Select by query type
        if not selected_ids:
            selected_ids = self._select_by_query_type(
                classification.query_type, blueprints
            )
            selection_reason = (
                f"Selected based on {classification.query_type} query type"
            )

        # Strategy 4: Fallback to best available
        if not selected_ids:
            selected_ids = list(blueprints.keys())[:3]
            selection_reason = "Selected fallback consultants for general query"

        # Build consultant selections
        selections = []
        for consultant_id in selected_ids[:3]:  # Ensure max 3
            if consultant_id in blueprints:
                blueprint = blueprints[consultant_id]

                # Determine frameworks to use
                frameworks_used = blueprint.stable_frameworks.copy()

                # Add frameworks based on adaptive triggers (if available)
                if hasattr(classification, "matched_triggers"):
                    for trigger_match in classification.matched_triggers:
                        if trigger_match.startswith(consultant_id + ":"):
                            if len(frameworks_used) < 6:  # Limit total frameworks
                                frameworks_used.append(
                                    f"Enhanced_{trigger_match.split(':')[1]}"
                                )

                selection = {
                    "consultant_id": consultant_id,
                    "blueprint": blueprint,
                    "selection_reason": selection_reason,
                    "frameworks_used": frameworks_used,
                    "confidence_score": self._calculate_confidence(
                        blueprint, classification
                    ),
                }
                selections.append(selection)

        # Ensure we have exactly 3 consultants
        while len(selections) < 3 and len(blueprints) >= len(selections) + 1:
            remaining_consultants = [
                c
                for c in blueprints.keys()
                if c not in [s["consultant_id"] for s in selections]
            ]
            if remaining_consultants:
                consultant_id = remaining_consultants[0]
                blueprint = blueprints[consultant_id]
                selection = {
                    "consultant_id": consultant_id,
                    "blueprint": blueprint,
                    "selection_reason": "Added for trio completion",
                    "frameworks_used": blueprint.stable_frameworks.copy(),
                    "confidence_score": 0.6,
                }
                selections.append(selection)
            else:
                break

        return selections

    async def _try_rules_selection(
        self, classification, blueprints: Dict[str, ConsultantBlueprint]
    ) -> Optional[List[Dict[str, Any]]]:
        """Try rules-based selection - extracted from lines 1901-1943"""
        try:
            # Convert classification to rules engine format
            context = {
                "domain": classification.query_type,
                "suggested_models": [],
                "complexity_score": getattr(classification, "complexity_score", 5),
                "keywords": getattr(classification, "keywords", []),
            }

            # Get selection from rules engine
            rules_result = self.rules_engine.select_optimal_consultants(
                self._current_query, context
            )

            # Convert back to our format
            selections = []
            for consultant_id in rules_result["selected_consultants"]:
                if consultant_id in blueprints:
                    blueprint = blueprints[consultant_id]
                    selection = {
                        "consultant_id": consultant_id,
                        "blueprint": blueprint,
                        "selection_reason": rules_result["selection_metadata"][
                            "selection_reasoning"
                        ].get(
                            consultant_id,
                            f"Selected by rules engine (score: {rules_result['selection_metadata']['average_score']:.3f})",
                        ),
                        "frameworks_used": blueprint.stable_frameworks,
                        "confidence_score": rules_result["selection_metadata"][
                            "average_score"
                        ],
                    }
                    selections.append(selection)

            if len(selections) == 3:
                print(
                    f"🎯 Rules engine selection: {[s['consultant_id'] for s in selections]}"
                )
                return selections

        except Exception as e:
            print(f"⚠️ Rules engine selection error: {e}")

        return None

    async def _try_predictive_selection(
        self, classification, blueprints: Dict[str, ConsultantBlueprint]
    ) -> tuple[List[str], str]:
        """Try predictive selection - extracted from lines 1949-1972"""
        try:
            prediction = self.predictive_selector.predict_optimal_consultants(
                query=getattr(self, "_current_query", "unknown"),
                keywords=getattr(classification, "keywords", []),
                query_type=classification.query_type,
                complexity_score=getattr(classification, "complexity_score", 5),
                routing_pattern=getattr(classification, "routing_pattern", "unknown"),
            )

            if (
                prediction.predicted_consultants
                and prediction.prediction_confidence > 0.6
            ):
                # Use top predictions
                predicted_ids = [
                    rec.consultant_id for rec in prediction.predicted_consultants[:3]
                ]
                # Filter for available consultants
                available_predicted = [
                    cid for cid in predicted_ids if cid in blueprints
                ]

                if available_predicted:
                    selection_reason = f"AI-predicted selection (confidence: {prediction.prediction_confidence:.1%})"
                    return available_predicted[:3], selection_reason

        except Exception as e:
            print(f"⚠️ Predictive selection error: {e}")

        return [], ""

    def _try_pattern_selection(self, classification) -> tuple[List[str], str]:
        """Try pattern-based selection - extracted from lines 1975-1978"""
        if classification.routing_pattern in self.routing_patterns:
            pattern_consultants = self.routing_patterns[classification.routing_pattern][
                "consultants"
            ]
            selected_ids = pattern_consultants[:3]  # Take first 3
            selection_reason = (
                f"Selected based on {classification.routing_pattern} routing pattern"
            )
            return selected_ids, selection_reason
        return [], ""

    def _select_by_query_type(
        self, query_type: str, blueprints: Dict[str, ConsultantBlueprint]
    ) -> List[str]:
        """
        Select consultants based on query type.
        Extracted from optimal_consultant_engine.py lines 2037-2056.
        """
        type_preferences = {
            "strategic": [
                "market_analyst",
                "strategic_synthesizer",
                "solution_architect",
            ],
            "problem_solving": [
                "problem_solver",
                "solution_architect",
                "execution_specialist",
            ],
            "innovation": [
                "solution_architect",
                "strategic_synthesizer",
                "market_analyst",
            ],
            "operational": [
                "execution_specialist",
                "process_expert",
                "operational_integrator",
            ],
            "execution": [
                "execution_specialist",
                "tactical_builder",
                "solution_architect",
            ],
            # Map our new contract enums to legacy types
            "strategic_analysis": [
                "market_analyst",
                "strategic_synthesizer",
                "solution_architect",
            ],
            "decision_support": [
                "strategic_synthesizer",
                "solution_architect",
                "market_analyst",
            ],
            "research_synthesis": [
                "market_analyst",
                "research_specialist",
                "strategic_synthesizer",
            ],
            "creative_ideation": [
                "solution_architect",
                "creative_specialist",
                "innovation_catalyst",
            ],
            "technical_analysis": [
                "technical_architect",
                "solution_architect",
                "execution_specialist",
            ],
            "general_inquiry": [
                "strategic_synthesizer",
                "solution_architect",
                "market_analyst",
            ],
        }

        preferred = type_preferences.get(query_type, [])
        available_preferred = [c for c in preferred if c in blueprints]

        if len(available_preferred) >= 3:
            return available_preferred[:3]

        # Fill remaining slots with any available consultants
        remaining = [c for c in blueprints.keys() if c not in available_preferred]
        return available_preferred + remaining[: 3 - len(available_preferred)]

    def _calculate_confidence(
        self, blueprint: ConsultantBlueprint, classification
    ) -> float:
        """
        Calculate confidence score for consultant selection.
        Extracted from optimal_consultant_engine.py lines 2058-2073.

        OPERATION AWAKENING: Enhanced with CQA quality feedback loop.
        """
        base_confidence = blueprint.effectiveness_score

        # Boost if adaptive triggers match
        trigger_boost = 0
        if hasattr(classification, "keywords") and hasattr(
            blueprint, "adaptive_triggers"
        ):
            trigger_matches = sum(
                1
                for trigger in blueprint.adaptive_triggers
                if trigger in classification.keywords
            )
            trigger_boost = min(trigger_matches * 0.1, 0.3)  # Max 0.3 boost

        # Boost if query type aligns with specialization
        specialization_boost = 0
        if hasattr(classification, "query_type") and hasattr(
            blueprint, "specialization"
        ):
            if classification.query_type in blueprint.specialization:
                specialization_boost = 0.1

        # ⚡ OPERATION AWAKENING: CQA Quality Boost
        # Apply quality-driven adjustment based on mental model CQA scores
        cqa_boost = self._calculate_cqa_boost(blueprint)

        return min(base_confidence + trigger_boost + specialization_boost + cqa_boost, 1.0)

    def _calculate_cqa_boost(self, blueprint: ConsultantBlueprint) -> float:
        """
        Calculate CQA-based boost for consultant selection.

        This is the FEEDBACK LOOP INTEGRATION that connects quality measurement
        back to operational decision-making.

        Returns:
            Float boost value (can be negative for poor quality models)
        """
        if not self.cqa_enabled or not self.cqa_service:
            return 0.0

        try:
            # Get mental models from consultant's stable_frameworks
            mental_models = blueprint.stable_frameworks

            if not mental_models:
                return 0.0

            # Retrieve CQA scores for all mental models
            cqa_scores = self.cqa_service.get_bulk_model_scores(mental_models)

            # Calculate aggregate boost
            boosts = []
            for model_name, cqa_score in cqa_scores.items():
                if cqa_score:
                    # Use the existing effectiveness boost calculator
                    effectiveness_multiplier = self.cqa_service.calculate_cqa_effectiveness_boost(cqa_score)

                    # Convert multiplier (0.7-1.3 range) to additive boost (-0.15 to +0.15)
                    # This keeps the boost within reasonable bounds
                    boost = (effectiveness_multiplier - 1.0) * 0.5  # Scale down to ±0.15
                    boosts.append(boost)

            if not boosts:
                return 0.0

            # Average boost across all mental models
            avg_boost = sum(boosts) / len(boosts)

            return avg_boost

        except Exception as e:
            # Fail gracefully - don't break selection if CQA lookup fails
            print(f"⚠️ CQA boost calculation failed for {blueprint.consultant_id}: {e}")
            return 0.0

    # === UTILITY AND CONVERSION METHODS ===

    def _convert_to_legacy_classification(
        self, classification_result: QueryClassificationResult
    ):
        """Convert new contract to legacy classification format"""
        from dataclasses import dataclass

        @dataclass
        class LegacyQueryClassification:
            keywords: List[str]
            complexity_score: int
            query_type: str
            matched_triggers: List[str]
            routing_pattern: Optional[str] = None

        # Map complexity enum to score
        complexity_mapping = {"simple": 3, "moderate": 5, "complex": 7, "expert": 9}
        complexity_score = complexity_mapping.get(
            classification_result.complexity.value, 5
        )

        return LegacyQueryClassification(
            keywords=classification_result.processing_hints.get("keywords", []),
            complexity_score=complexity_score,
            query_type=classification_result.intent.value,
            matched_triggers=classification_result.domain_tags,
            routing_pattern=classification_result.processing_hints.get(
                "routing_pattern"
            ),
        )

    def _convert_to_candidates(
        self, selections: List[Dict[str, Any]]
    ) -> List[ConsultantCandidate]:
        """Convert legacy selections to new contract format"""
        candidates = []

        for selection in selections:
            blueprint = selection["blueprint"]

            # Map legacy consultant IDs to roles (basic mapping)
            role_mapping = {
                "strategic_synthesizer": ConsultantRole.STRATEGIC_CONSULTANT,
                "market_analyst": ConsultantRole.STRATEGIC_CONSULTANT,
                "solution_architect": ConsultantRole.TECHNICAL_CONSULTANT,
                "research_specialist": ConsultantRole.RESEARCH_ANALYST,
                "creative_specialist": ConsultantRole.CREATIVE_DIRECTOR,
                "execution_specialist": ConsultantRole.OPERATIONS_SPECIALIST,
                "risk_specialist": ConsultantRole.RISK_ANALYST,
            }

            role = role_mapping.get(
                selection["consultant_id"], ConsultantRole.MANAGEMENT_CONSULTANT
            )

            candidate = ConsultantCandidate(
                consultant_id=selection["consultant_id"],
                role=role,
                name=blueprint.name,
                expertise_domains=selection["frameworks_used"][:5],  # Limit to 5
                match_score=selection["confidence_score"],
                reasoning=selection["selection_reason"],
            )
            candidates.append(candidate)

        return candidates

    def _determine_nway_clusters(
        self, classification_result: QueryClassificationResult
    ) -> List[str]:
        """Determine relevant N-Way clusters - placeholder for now"""
        # This would integrate with semantic cluster matching service
        # For now, return domain tags as clusters
        return classification_result.domain_tags[:3]

    def _get_selection_strategy_used(self) -> str:
        """Return the selection strategy that was used"""
        if self.rules_engine_enabled:
            return "rules_based_selection"
        elif self.predictive_enabled:
            return "predictive_selection"
        else:
            return "pattern_and_type_based_selection"

    def _build_selection_reasoning(
        self,
        selections: List[Dict[str, Any]],
        classification: QueryClassificationResult,
    ) -> str:
        """Build overall selection reasoning"""
        if not selections:
            return "No consultants selected"

        strategy = self._get_selection_strategy_used()
        intent = classification.intent.value
        complexity = classification.complexity.value

        return (
            f"Selected {len(selections)} consultants using {strategy} "
            f"for {intent} query with {complexity} complexity. "
            f"Average confidence: {sum(s['confidence_score'] for s in selections) / len(selections):.2f}"
        )

    # === SERVICE CONFIGURATION ===

    def configure_rules_engine(self, rules_engine):
        """Configure rules-based selection engine"""
        self.rules_engine = rules_engine
        self.rules_engine_enabled = True
        print("✅ ConsultantSelectionService: Rules engine configured")

    def configure_predictive_selector(self, predictive_selector):
        """Configure predictive selection system"""
        self.predictive_selector = predictive_selector
        self.predictive_enabled = True
        print("✅ ConsultantSelectionService: Predictive selector configured")

    def configure_routing_patterns(self, routing_patterns: Dict[str, Dict[str, Any]]):
        """Configure routing patterns for pattern-based selection"""
        self.routing_patterns = routing_patterns
        print(
            f"✅ ConsultantSelectionService: {len(routing_patterns)} routing patterns configured"
        )


# Factory function for service creation
def get_consultant_selection_service(
    consultant_blueprints: Optional[Dict[str, ConsultantBlueprint]] = None,
) -> ConsultantSelectionService:
    """Factory function to create ConsultantSelectionService instance"""
    return ConsultantSelectionService(consultant_blueprints)
