"""
METIS Rules-Based Consultant Selector
Implementation of the complete scoring algorithm from RULES_ENGINE_SCORING_ALGORITHM_SPECIFICATION.md

This is the "soul" of the intelligent consultant selection system.
Implements configuration-driven selection with graceful fallbacks.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

from src.engine.models.data_contracts import (
    ConsultantMatrixConfig,
    ConsultantSpecialization,
    ConsultantSelectionInput,
    ConsultantSelectionResult,
    ExtendedConsultantRole,
    ScoringWeights,
    CognitiveFunction,
)
from src.engine.adapters.validation import  # Migrated ConsultantConfigValidator

logger = logging.getLogger(__name__)


@dataclass
class RulesBasedSelector:
    """
    Configuration-driven consultant selection engine

    Implements intelligent consultant selection based on:
    1. Keyword matching from YAML configuration
    2. Domain bias scores from mental_models_system.py
    3. Mental model alignment scoring
    4. Legacy compatibility fallback

    Based on RULES_ENGINE_SCORING_ALGORITHM_SPECIFICATION.md
    """

    matrix: ConsultantMatrixConfig
    scoring_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "keyword_match": 0.4,
            "mental_model_bias": 0.3,
            "strategic_layer_fit": 0.2,
            "cognitive_function_match": 0.1,
        }
    )

    def __post_init__(self):
        """Initialize with configuration validation"""
        # Update scoring weights from matrix config
        if self.matrix.scoring_weights:
            self.scoring_weights = {
                "keyword_match": self.matrix.scoring_weights.keyword_match,
                "mental_model_bias": self.matrix.scoring_weights.mental_model_bias,
                "strategic_layer_fit": self.matrix.scoring_weights.strategic_layer_fit,
                "cognitive_function_match": self.matrix.scoring_weights.cognitive_function_match,
            }

        logger.info(
            f"RulesBasedSelector initialized with {len(self.matrix.consultants)} consultants"
        )

    @classmethod
    def from_config_file(
        cls, config_file_path: Optional[str] = None
    ) -> "RulesBasedSelector":
        """
        Create RulesBasedSelector from YAML configuration file
        Uses ConfigValidator for bulletproof loading with fallback
        """
        validator = ConsultantConfigValidator(config_file_path)
        matrix_config = validator.load_validated_config()

        return cls(matrix=matrix_config)

    def select_consultants(
        self, selection_input: ConsultantSelectionInput
    ) -> ConsultantSelectionResult:
        """
        Main selection method - the 'soul' of the intelligent system
        Select 3 consultants using configuration-driven scoring with legacy fallback

        Based on RULES_ENGINE_SCORING_ALGORITHM_SPECIFICATION.md
        """
        start_time = time.time()

        # 1. Score all consultants
        scored_consultants = []
        for role, spec in self.matrix.consultants.items():
            score = self._calculate_consultant_score(spec, selection_input)
            scored_consultants.append((role, score, spec))
            logger.debug(f"Consultant {role.value}: score={score:.3f}")

        # 2. Sort by score and get top candidates
        scored_consultants.sort(key=lambda x: x[1], reverse=True)
        top_three = [role for role, score, spec in scored_consultants[:3]]

        # 3. Apply legacy compatibility fallback if configured
        fallback_applied = False
        legacy_compatibility_mode = False

        if self._should_enforce_legacy_compatibility():
            original_selection = top_three.copy()
            top_three = self._apply_legacy_fallback(top_three, scored_consultants)
            fallback_applied = top_three != original_selection
            legacy_compatibility_mode = True

            if fallback_applied:
                logger.info(
                    f"Legacy fallback applied: {original_selection} -> {top_three}"
                )

        # 4. Build detailed result
        selection_time_ms = int((time.time() - start_time) * 1000)

        selection_scores = {role: score for role, score, _ in scored_consultants}
        selection_reasoning = {
            role: self._generate_selection_reasoning(role, score, spec, selection_input)
            for role, score, spec in scored_consultants
            if role in top_three
        }

        # Calculate quality indicators
        selected_scores = [selection_scores[role] for role in top_three]

        return ConsultantSelectionResult(
            selected_consultants=top_three,
            selection_scores=selection_scores,
            selection_reasoning=selection_reasoning,
            algorithm_version="rules_based_v1.0",
            scoring_weights_used=ScoringWeights(
                keyword_match=self.scoring_weights["keyword_match"],
                mental_model_bias=self.scoring_weights["mental_model_bias"],
                strategic_layer_fit=self.scoring_weights["strategic_layer_fit"],
                cognitive_function_match=self.scoring_weights[
                    "cognitive_function_match"
                ],
            ),
            fallback_applied=fallback_applied,
            legacy_compatibility_mode=legacy_compatibility_mode,
            selection_time_ms=selection_time_ms,
            total_consultants_scored=len(scored_consultants),
            minimum_score=min(selected_scores),
            maximum_score=max(selected_scores),
            average_score=sum(selected_scores) / len(selected_scores),
            score_distribution=self._calculate_score_distribution(scored_consultants),
        )

    def _calculate_consultant_score(
        self, spec: ConsultantSpecialization, selection_input: ConsultantSelectionInput
    ) -> float:
        """
        Core scoring algorithm - configuration-driven intelligence

        Calculate comprehensive consultant score based on configuration-driven logic
        Based on RULES_ENGINE_SCORING_ALGORITHM_SPECIFICATION.md

        Args:
            spec: Consultant specialization configuration
            selection_input: User query and context

        Returns:
            Float score (0.0-1.0 range for proper weighting)
        """
        total_score = 0.0

        # 1. Keyword Matching Score (40% weight)
        keyword_score = self._calculate_keyword_score(spec, selection_input.query_text)
        total_score += keyword_score * self.scoring_weights["keyword_match"]
        logger.debug(f"{spec.consultant_id} keyword_score: {keyword_score:.3f}")

        # 2. Domain Bias Score (20% weight - reusing strategic_layer_fit weight)
        domain_score = self._calculate_domain_score(spec, selection_input.domain_hint)
        total_score += domain_score * self.scoring_weights["strategic_layer_fit"]
        logger.debug(f"{spec.consultant_id} domain_score: {domain_score:.3f}")

        # 3. Mental Model Alignment Score (30% weight)
        context = {"suggested_models": getattr(selection_input, "suggested_models", [])}
        if selection_input.enhanced_context:
            context.update(selection_input.enhanced_context)

        model_score = self._calculate_mental_model_score(spec, context)
        total_score += model_score * self.scoring_weights["mental_model_bias"]
        logger.debug(f"{spec.consultant_id} model_score: {model_score:.3f}")

        # 4. Cognitive Function Match Score (10% weight)
        function_score = self._calculate_cognitive_function_score(spec, selection_input)
        total_score += function_score * self.scoring_weights["cognitive_function_match"]
        logger.debug(f"{spec.consultant_id} function_score: {function_score:.3f}")

        final_score = min(total_score, 1.0)  # Cap at 1.0 for consistency
        logger.debug(f"{spec.consultant_id} final_score: {final_score:.3f}")

        return final_score

    def _calculate_keyword_score(
        self, spec: ConsultantSpecialization, query_text: str
    ) -> float:
        """
        Simple, normalized keyword matching
        Based on RULES_ENGINE_SCORING_ALGORITHM_SPECIFICATION.md
        """
        if not spec.trigger_keywords:
            return 0.0

        query_lower = query_text.lower()
        keyword_matches = sum(
            1 for keyword in spec.trigger_keywords if keyword.lower() in query_lower
        )

        # Normalize to prevent dominance by consultants with more keywords
        max_possible_matches = len(spec.trigger_keywords)
        return (
            (keyword_matches / max_possible_matches)
            if max_possible_matches > 0
            else 0.0
        )

    def _calculate_domain_score(
        self, spec: ConsultantSpecialization, domain_hint: Optional[str]
    ) -> float:
        """
        Domain scoring using existing mental model biases
        Based on RULES_ENGINE_SCORING_ALGORITHM_SPECIFICATION.md
        """
        if not domain_hint or not spec.bias_scores:
            return 0.5  # Neutral score when no domain hint

        # Use pre-populated bias scores from mental_models_system.py
        domain_bias = spec.bias_scores.get(domain_hint.lower(), 0.5)
        return domain_bias  # Already normalized (0.0-1.0) by mental models system

    def _calculate_mental_model_score(
        self, spec: ConsultantSpecialization, context: Dict[str, Any]
    ) -> float:
        """
        Mental model alignment scoring
        Based on RULES_ENGINE_SCORING_ALGORITHM_SPECIFICATION.md
        """
        if not context or "suggested_models" not in context:
            return 0.3  # Default when no suggested models

        if not spec.preferred_mental_models:
            return 0.3  # Default when consultant has no model preferences

        suggested_models = set(context["suggested_models"])
        preferred_models = set(spec.preferred_mental_models)

        if not suggested_models:
            return 0.3  # Default when no models suggested

        model_overlap = suggested_models & preferred_models
        overlap_score = len(model_overlap) / len(preferred_models)

        # Boost score slightly for any overlap to encourage model-based selection
        return min(overlap_score + 0.2, 1.0) if model_overlap else 0.1

    def _calculate_cognitive_function_score(
        self, spec: ConsultantSpecialization, selection_input: ConsultantSelectionInput
    ) -> float:
        """
        MVP cognitive function scoring based on query characteristics
        Based on RULES_ENGINE_SCORING_ALGORITHM_SPECIFICATION.md
        """
        query_lower = selection_input.query_text.lower()

        # Simple heuristic scoring based on query patterns
        if spec.cognitive_function == CognitiveFunction.ANALYSIS:
            analysis_indicators = [
                "analyze",
                "assessment",
                "evaluation",
                "research",
                "data",
                "metrics",
            ]
            matches = sum(
                1 for indicator in analysis_indicators if indicator in query_lower
            )
            return min(matches * 0.2, 1.0)  # Cap at 1.0

        elif spec.cognitive_function == CognitiveFunction.SYNTHESIS:
            synthesis_indicators = [
                "design",
                "create",
                "integrate",
                "combine",
                "holistic",
                "stakeholder",
            ]
            matches = sum(
                1 for indicator in synthesis_indicators if indicator in query_lower
            )
            return min(matches * 0.2, 1.0)

        elif spec.cognitive_function == CognitiveFunction.IMPLEMENTATION:
            implementation_indicators = [
                "implement",
                "execute",
                "deploy",
                "rollout",
                "process",
                "workflow",
            ]
            matches = sum(
                1 for indicator in implementation_indicators if indicator in query_lower
            )
            return min(matches * 0.2, 1.0)

        return 0.1  # Small baseline score for non-matching functions

    def _should_enforce_legacy_compatibility(self) -> bool:
        """Check if legacy compatibility should be enforced"""
        if not self.matrix.fallback_behavior:
            return True  # Default to legacy compatibility

        return self.matrix.fallback_behavior.always_include_legacy_three

    def _apply_legacy_fallback(
        self,
        current_selection: List[ExtendedConsultantRole],
        all_scored: List[
            Tuple[ExtendedConsultantRole, float, ConsultantSpecialization]
        ],
    ) -> List[ExtendedConsultantRole]:
        """
        Ensure legacy consultant compatibility when configured
        Based on RULES_ENGINE_SCORING_ALGORITHM_SPECIFICATION.md
        """
        legacy_roles = self.matrix.get_legacy_three()
        current_set = set(current_selection)
        legacy_set = set(legacy_roles)

        # If all legacy roles are already selected, no change needed
        if legacy_set.issubset(current_set):
            return current_selection

        # Find missing legacy roles
        missing_legacy = legacy_set - current_set

        # If only one legacy role missing, replace lowest-scoring current selection
        if len(missing_legacy) == 1:
            missing_role = list(missing_legacy)[0]

            # Find the score of the missing legacy role
            legacy_score = next(
                (score for role, score, spec in all_scored if role == missing_role), 0.0
            )
            current_scores = [
                (role, next(score for r, score, s in all_scored if r == role))
                for role in current_selection
            ]

            # Replace lowest-scoring current selection with missing legacy role
            lowest_current = min(current_scores, key=lambda x: x[1])

            # Get tolerance from config
            tolerance = 0.8  # Default
            if self.matrix.fallback_behavior:
                tolerance = self.matrix.fallback_behavior.legacy_preference_tolerance

            # Only replace if legacy role scores higher than current lowest (with tolerance)
            if legacy_score > lowest_current[1] * tolerance:
                final_selection = [
                    role for role in current_selection if role != lowest_current[0]
                ]
                final_selection.append(missing_role)
                return final_selection

        # For more complex cases, fall back to simple legacy inclusion
        # This is a safety net - should rarely be triggered with good configuration
        logger.warning(
            f"Complex legacy fallback triggered, using legacy three: {legacy_roles}"
        )
        return legacy_roles  # Use legacy three as failsafe

    def _generate_selection_reasoning(
        self,
        role: ExtendedConsultantRole,
        score: float,
        spec: ConsultantSpecialization,
        selection_input: ConsultantSelectionInput,
    ) -> str:
        """Generate human-readable reasoning for consultant selection"""

        reasons = []

        # Keyword matching reasoning
        keyword_score = self._calculate_keyword_score(spec, selection_input.query_text)
        if keyword_score > 0.3:
            matched_keywords = [
                kw
                for kw in spec.trigger_keywords
                if kw.lower() in selection_input.query_text.lower()
            ]
            reasons.append(
                f"Strong keyword alignment ({keyword_score:.2f}): {', '.join(matched_keywords[:3])}"
            )

        # Strategic layer and cognitive function
        reasons.append(
            f"{spec.strategic_layer.value.title()} {spec.cognitive_function.value} specialist"
        )

        # Domain expertise
        if selection_input.domain_hint:
            domain_score = self._calculate_domain_score(
                spec, selection_input.domain_hint
            )
            if domain_score > 0.6:
                reasons.append(
                    f"Domain expertise in {selection_input.domain_hint} ({domain_score:.2f})"
                )

        # Mental model alignment
        if selection_input.suggested_models:
            model_score = self._calculate_mental_model_score(
                spec, {"suggested_models": selection_input.suggested_models}
            )
            if model_score > 0.4:
                overlapping_models = set(spec.preferred_mental_models) & set(
                    selection_input.suggested_models
                )
                if overlapping_models:
                    reasons.append(
                        f"Mental model alignment: {', '.join(list(overlapping_models)[:2])}"
                    )

        reasoning = f"Selected with score {score:.3f}. " + "; ".join(reasons)
        return reasoning

    def _calculate_score_distribution(
        self,
        scored_consultants: List[
            Tuple[ExtendedConsultantRole, float, ConsultantSpecialization]
        ],
    ) -> Dict[str, int]:
        """Calculate score distribution across ranges"""
        distribution = {"high": 0, "medium": 0, "low": 0}

        for _, score, _ in scored_consultants:
            if score >= 0.7:
                distribution["high"] += 1
            elif score >= 0.4:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1

        return distribution


class RulesEngineAdapter:
    """
    Adapter class to integrate RulesBasedSelector with existing OptimalConsultantEngine
    Maintains backward compatibility while enabling new rules-based intelligence
    """

    def __init__(self, config_file_path: Optional[str] = None):
        self.rules_selector = RulesBasedSelector.from_config_file(config_file_path)
        logger.info("RulesEngineAdapter initialized")

    def select_optimal_consultants(
        self, query: str, context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Main integration method - converts between old and new interfaces
        Provides backward compatibility with existing system
        """
        # Convert to new input format
        selection_input = ConsultantSelectionInput(
            query_text=query,
            domain_hint=context.get("domain") if context else None,
            suggested_models=context.get("suggested_models", []) if context else [],
            enhanced_context=context or {},
            urgency_level=context.get("urgency", "medium") if context else "medium",
        )

        # Use new rules-based selection
        result = self.rules_selector.select_consultants(selection_input)

        # Convert back to legacy format for compatibility
        legacy_result = {
            "selected_consultants": [
                role.value for role in result.selected_consultants
            ],
            "consultant_roles": result.selected_consultants,
            "selection_metadata": {
                "algorithm_version": result.algorithm_version,
                "selection_time_ms": result.selection_time_ms,
                "fallback_applied": result.fallback_applied,
                "average_score": result.average_score,
                "selection_reasoning": {
                    role.value: reasoning
                    for role, reasoning in result.selection_reasoning.items()
                },
            },
            "timestamp": datetime.utcnow(),
        }

        logger.info(
            f"Selected consultants via rules engine: {[role.value for role in result.selected_consultants]} "
            f"(avg score: {result.average_score:.3f}, time: {result.selection_time_ms}ms)"
        )

        return legacy_result

    def get_selection_details(
        self, query: str, context: Optional[Dict] = None
    ) -> ConsultantSelectionResult:
        """Get detailed selection result with full metadata"""
        selection_input = ConsultantSelectionInput(
            query_text=query,
            domain_hint=context.get("domain") if context else None,
            suggested_models=context.get("suggested_models", []) if context else [],
            enhanced_context=context or {},
        )

        return self.rules_selector.select_consultants(selection_input)


# Factory function for easy integration
def create_rules_based_selector(
    config_file_path: Optional[str] = None,
) -> RulesBasedSelector:
    """Factory function to create RulesBasedSelector with configuration validation"""
    return RulesBasedSelector.from_config_file(config_file_path)


def create_rules_engine_adapter(
    config_file_path: Optional[str] = None,
) -> RulesEngineAdapter:
    """Factory function to create adapter for existing system integration"""
    return RulesEngineAdapter(config_file_path)
