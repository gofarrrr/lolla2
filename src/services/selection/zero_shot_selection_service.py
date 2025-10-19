"""
METIS Zero-Shot Selection Service
Part of Selection Services Cluster - Focused on zero-shot model selection using MeMo methodology

Extracted from model_selector.py during Phase 5.2 decomposition.
Single Responsibility: Perform zero-shot model selection when historical data is insufficient.
Based on MeMo (Model Selection via Memory) paper implementation.
"""

import logging
import re
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.services.contracts.selection_contracts import (
    IZeroShotSelectionService,
    ZeroShotSelectionContract,
    SelectionResultContract,
    SelectionContextContract,
    MergeStrategy,
    ModelScoreContract,
)
from src.integrations.llm.unified_client import UnifiedLLMClient


class ZeroShotSelectionService(IZeroShotSelectionService):
    """
    Focused service for zero-shot model selection using MeMo methodology
    Clean extraction from model_selector.py zero-shot selection methods
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.llm_client = UnifiedLLMClient()

        # MeMo methodology parameters
        self.selection_prompt_template = self._build_selection_prompt_template()
        self.reasoning_chain_depth = 3
        self.confidence_threshold = 0.6
        self.novelty_detection_keywords = [
            "unprecedented",
            "novel",
            "unique",
            "first-time",
            "experimental",
            "innovative",
            "cutting-edge",
            "breakthrough",
            "emerging",
        ]

        # Model capability profiles (simplified for zero-shot reasoning)
        self.model_capability_profiles = {
            "deepseek_chat": {
                "strengths": [
                    "reasoning",
                    "code_analysis",
                    "technical_problems",
                    "mathematical_thinking",
                ],
                "ideal_for": [
                    "complex_analysis",
                    "strategic_planning",
                    "technical_documentation",
                ],
                "complexity_range": ["medium", "high", "very_high"],
                "business_contexts": ["technology", "finance", "research"],
            },
            "claude_sonnet": {
                "strengths": [
                    "creative_thinking",
                    "communication",
                    "analysis",
                    "writing",
                ],
                "ideal_for": [
                    "content_creation",
                    "strategic_communication",
                    "business_analysis",
                ],
                "complexity_range": ["low", "medium", "high"],
                "business_contexts": [
                    "marketing",
                    "communications",
                    "general_business",
                ],
            },
            "gpt4": {
                "strengths": [
                    "versatile_reasoning",
                    "creative_problem_solving",
                    "general_intelligence",
                ],
                "ideal_for": [
                    "brainstorming",
                    "general_analysis",
                    "creative_solutions",
                ],
                "complexity_range": ["low", "medium", "high"],
                "business_contexts": ["general", "creative", "consulting"],
            },
        }

        self.logger.info(
            "🎯 ZeroShotSelectionService initialized with MeMo methodology"
        )

    async def perform_zero_shot_selection(
        self, context: SelectionContextContract
    ) -> ZeroShotSelectionContract:
        """
        Core service method: Perform zero-shot model selection using MeMo approach
        Uses LLM reasoning to select models when historical data is insufficient
        """
        try:
            start_time = datetime.utcnow()

            # Step 1: Analyze context for novelty and complexity
            novelty_analysis = await self._analyze_context_novelty(context)

            # Step 2: Generate reasoning chain for model selection
            reasoning_chain = await self._generate_reasoning_chain(
                context, novelty_analysis
            )

            # Step 3: Perform model selection using reasoning
            selection_result = await self._perform_memo_selection(
                context, reasoning_chain
            )

            # Step 4: Calculate confidence and validate selection
            confidence_score = await self._calculate_selection_confidence(
                selection_result, context, reasoning_chain
            )

            # Step 5: Create zero-shot selection contract
            zero_shot_contract = ZeroShotSelectionContract(
                engagement_id=context.problem_statement[:50] + "_zeroshot",
                selected_models=selection_result["selected_models"],
                confidence_score=confidence_score,
                reasoning_process=reasoning_chain,
                context_analysis=novelty_analysis,
                novelty_factors=selection_result["novelty_factors"],
                selection_timestamp=datetime.utcnow(),
                service_version="v5_modular_memo",
            )

            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.logger.info(
                f"🎯 Zero-shot selection completed in {execution_time:.0f}ms"
            )

            return zero_shot_contract

        except Exception as e:
            self.logger.error(f"❌ Zero-shot selection failed: {e}")
            return self._create_fallback_zero_shot_result(context, str(e))

    async def merge_with_database_selection(
        self,
        zero_shot_result: ZeroShotSelectionContract,
        database_result: SelectionResultContract,
        merge_strategy: MergeStrategy,
    ) -> SelectionResultContract:
        """
        Core service method: Merge zero-shot selection with database-driven selection
        Implements different merge strategies for hybrid selection
        """
        try:
            if merge_strategy == MergeStrategy.WEIGHTED_CONFIDENCE:
                merged_result = await self._merge_weighted_confidence(
                    zero_shot_result, database_result
                )
            elif merge_strategy == MergeStrategy.CONSENSUS_BOOSTING:
                merged_result = await self._merge_consensus_boosting(
                    zero_shot_result, database_result
                )
            elif merge_strategy == MergeStrategy.HYBRID_RANKING:
                merged_result = await self._merge_hybrid_ranking(
                    zero_shot_result, database_result
                )
            else:
                # Default to weighted confidence
                merged_result = await self._merge_weighted_confidence(
                    zero_shot_result, database_result
                )

            # Update metadata to reflect merge
            merged_result.selection_metadata.update(
                {
                    "merge_strategy_used": merge_strategy.value,
                    "zero_shot_confidence": zero_shot_result.confidence_score,
                    "database_models_evaluated": database_result.models_evaluated,
                    "hybrid_selection": True,
                }
            )

            self.logger.info(
                f"🔄 Merged selections using {merge_strategy.value} strategy"
            )
            return merged_result

        except Exception as e:
            self.logger.error(f"❌ Merge failed, using database result: {e}")
            return database_result

    async def _analyze_context_novelty(
        self, context: SelectionContextContract
    ) -> Dict[str, Any]:
        """Analyze context to detect novelty indicators for zero-shot reasoning"""
        try:
            novelty_analysis_prompt = f"""
            Analyze this business problem for novelty and complexity indicators:
            
            Problem: {context.problem_statement}
            Business Context: {context.business_context}
            Problem Type: {context.problem_type}
            Complexity: {context.complexity_level}
            
            Determine:
            1. Novelty level (0-1): How unprecedented/unique is this problem?
            2. Key novelty indicators: What makes this problem unique?
            3. Complexity drivers: What factors contribute to complexity?
            4. Domain specificity: How domain-specific are the requirements?
            5. Solution approach hints: What type of models might excel?
            
            Respond in JSON format with these fields.
            """

            response = await self.llm_client.generate_response(
                novelty_analysis_prompt, max_tokens=1000, temperature=0.3
            )

            # Parse LLM response (simplified JSON extraction)
            analysis_result = self._extract_json_from_response(response)

            if not analysis_result:
                # Fallback analysis
                analysis_result = {
                    "novelty_level": self._calculate_novelty_heuristic(context),
                    "novelty_indicators": self._detect_novelty_keywords(
                        context.problem_statement
                    ),
                    "complexity_drivers": [
                        context.complexity_level,
                        context.problem_type,
                    ],
                    "domain_specificity": 0.7,
                    "solution_approach_hints": ["general_purpose_models"],
                }

            return analysis_result

        except Exception as e:
            self.logger.error(f"❌ Novelty analysis failed: {e}")
            return {"error": str(e), "novelty_level": 0.5}

    async def _generate_reasoning_chain(
        self, context: SelectionContextContract, novelty_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate step-by-step reasoning chain for model selection (MeMo approach)"""
        try:
            reasoning_prompt = self.selection_prompt_template.format(
                problem_statement=context.problem_statement,
                business_context=context.business_context,
                problem_type=context.problem_type,
                complexity_level=context.complexity_level,
                accuracy_requirement=context.accuracy_requirement,
                max_models=context.max_models,
                novelty_level=novelty_analysis.get("novelty_level", 0.5),
                novelty_indicators=novelty_analysis.get("novelty_indicators", []),
                available_models=list(self.model_capability_profiles.keys()),
            )

            response = await self.llm_client.generate_response(
                reasoning_prompt, max_tokens=2000, temperature=0.4
            )

            # Extract reasoning steps from response
            reasoning_steps = self._extract_reasoning_steps(response)

            return reasoning_steps

        except Exception as e:
            self.logger.error(f"❌ Reasoning chain generation failed: {e}")
            return [f"Fallback reasoning: {str(e)}"]

    async def _perform_memo_selection(
        self, context: SelectionContextContract, reasoning_chain: List[str]
    ) -> Dict[str, Any]:
        """Perform model selection using MeMo (Model Selection via Memory) approach"""
        try:
            # Analyze reasoning chain for model preferences
            model_preferences = self._extract_model_preferences(reasoning_chain)

            # Match model capabilities to context requirements
            capability_matches = self._match_model_capabilities(context)

            # Combine preferences and capability matches
            model_scores = {}
            for model_id in self.model_capability_profiles.keys():
                preference_score = model_preferences.get(model_id, 0.5)
                capability_score = capability_matches.get(model_id, 0.5)

                # Weighted combination (preferences get higher weight in zero-shot)
                final_score = (preference_score * 0.7) + (capability_score * 0.3)
                model_scores[model_id] = final_score

            # Select top models
            sorted_models = sorted(
                model_scores.items(), key=lambda x: x[1], reverse=True
            )
            selected_models = [
                model_id for model_id, _ in sorted_models[: context.max_models]
            ]

            # Extract novelty factors from reasoning
            novelty_factors = self._extract_novelty_factors(reasoning_chain)

            return {
                "selected_models": selected_models,
                "model_scores": model_scores,
                "novelty_factors": novelty_factors,
                "selection_rationale": (
                    reasoning_chain[-1] if reasoning_chain else "Default selection"
                ),
            }

        except Exception as e:
            self.logger.error(f"❌ MeMo selection failed: {e}")
            return {
                "selected_models": ["deepseek_chat"],  # Safe default
                "model_scores": {"deepseek_chat": 0.8},
                "novelty_factors": ["fallback_selection"],
                "error": str(e),
            }

    async def _calculate_selection_confidence(
        self,
        selection_result: Dict[str, Any],
        context: SelectionContextContract,
        reasoning_chain: List[str],
    ) -> float:
        """Calculate confidence in zero-shot selection"""
        try:
            confidence_factors = []

            # Factor 1: Reasoning chain quality
            reasoning_quality = len(reasoning_chain) / self.reasoning_chain_depth
            confidence_factors.append(min(reasoning_quality, 1.0))

            # Factor 2: Model score distribution
            scores = list(selection_result.get("model_scores", {}).values())
            if scores:
                score_variance = sum(
                    (s - sum(scores) / len(scores)) ** 2 for s in scores
                ) / len(scores)
                # Lower variance = higher confidence in selection
                confidence_factors.append(1.0 - min(score_variance, 1.0))

            # Factor 3: Context clarity
            context_clarity = self._assess_context_clarity(context)
            confidence_factors.append(context_clarity)

            # Factor 4: Novelty adjustment (higher novelty = lower confidence)
            novelty_penalty = len(selection_result.get("novelty_factors", [])) * 0.1
            confidence_factors.append(max(0.5, 1.0 - novelty_penalty))

            # Calculate weighted confidence
            final_confidence = sum(confidence_factors) / len(confidence_factors)

            return max(0.1, min(final_confidence, 0.95))  # Bound between 0.1 and 0.95

        except Exception as e:
            self.logger.error(f"❌ Confidence calculation failed: {e}")
            return 0.5  # Neutral confidence

    async def _merge_weighted_confidence(
        self, zero_shot: ZeroShotSelectionContract, database: SelectionResultContract
    ) -> SelectionResultContract:
        """Merge using weighted confidence approach"""
        # Weight models based on respective confidence scores
        zero_shot_weight = zero_shot.confidence_score
        database_weight = 1.0 - zero_shot_weight  # Inverse weighting

        # Combine model selections
        all_models = list(set(zero_shot.selected_models + database.selected_models))

        # Create merged scores
        merged_scores = []
        for model_id in all_models[: database.selection_metadata.get("max_models", 3)]:
            # Create merged score contract
            base_score = 0.7 if model_id in zero_shot.selected_models else 0.3

            merged_score = ModelScoreContract(
                model_id=model_id,
                total_score=base_score,
                component_scores={
                    "zero_shot_preference": (
                        1.0 if model_id in zero_shot.selected_models else 0.0
                    ),
                    "database_score": (
                        1.0 if model_id in database.selected_models else 0.0
                    ),
                    "weighted_combination": base_score,
                },
                rationale=f"Merged selection (zero-shot: {zero_shot_weight:.2f}, database: {database_weight:.2f})",
                confidence=zero_shot.confidence_score,
                risk_factors=["hybrid_selection_uncertainty"],
                scoring_timestamp=datetime.utcnow(),
                service_version="v5_modular_merged",
            )
            merged_scores.append(merged_score)

        # Create merged result
        merged_result = SelectionResultContract(
            engagement_id=database.engagement_id + "_merged",
            selected_models=[score.model_id for score in merged_scores],
            model_scores=merged_scores,
            selection_source="hybrid_weighted",
            strategy_used="zero_shot_database_merge",
            models_evaluated=len(all_models),
            selection_metadata={
                **database.selection_metadata,
                "merge_weights": {
                    "zero_shot": zero_shot_weight,
                    "database": database_weight,
                },
            },
            total_selection_time_ms=database.total_selection_time_ms,
            cognitive_load_assessment=database.cognitive_load_assessment,
            selection_timestamp=datetime.utcnow(),
            service_version="v5_modular_merged",
        )

        return merged_result

    async def _merge_consensus_boosting(
        self, zero_shot: ZeroShotSelectionContract, database: SelectionResultContract
    ) -> SelectionResultContract:
        """Merge using consensus boosting - prefer models selected by both"""
        consensus_models = list(
            set(zero_shot.selected_models) & set(database.selected_models)
        )

        # If no consensus, fall back to weighted approach
        if not consensus_models:
            return await self._merge_weighted_confidence(zero_shot, database)

        # Boost consensus models, then add others
        selected_models = consensus_models.copy()

        # Add non-consensus models up to limit
        remaining_slots = min(3, len(database.selected_models)) - len(consensus_models)
        non_consensus = [
            m for m in database.selected_models if m not in consensus_models
        ]
        selected_models.extend(non_consensus[:remaining_slots])

        # Create boosted scores
        merged_scores = []
        for model_id in selected_models:
            boost_factor = 1.3 if model_id in consensus_models else 1.0
            base_score = 0.8 * boost_factor

            merged_score = ModelScoreContract(
                model_id=model_id,
                total_score=min(base_score, 1.0),
                component_scores={
                    "consensus_boost": boost_factor - 1.0,
                    "consensus_selection": 1.0 if model_id in consensus_models else 0.0,
                },
                rationale=f"Consensus boosted selection (boost: {boost_factor})",
                confidence=(
                    zero_shot.confidence_score * 1.1
                    if model_id in consensus_models
                    else zero_shot.confidence_score
                ),
                risk_factors=[],
                scoring_timestamp=datetime.utcnow(),
                service_version="v5_modular_consensus",
            )
            merged_scores.append(merged_score)

        return SelectionResultContract(
            engagement_id=database.engagement_id + "_consensus",
            selected_models=selected_models,
            model_scores=merged_scores,
            selection_source="hybrid_consensus",
            strategy_used="consensus_boosting",
            models_evaluated=len(
                set(zero_shot.selected_models + database.selected_models)
            ),
            selection_metadata={
                **database.selection_metadata,
                "consensus_models": consensus_models,
                "consensus_boost_applied": True,
            },
            total_selection_time_ms=database.total_selection_time_ms,
            cognitive_load_assessment="low",  # Consensus reduces cognitive load
            selection_timestamp=datetime.utcnow(),
            service_version="v5_modular_consensus",
        )

    async def _merge_hybrid_ranking(
        self, zero_shot: ZeroShotSelectionContract, database: SelectionResultContract
    ) -> SelectionResultContract:
        """Merge using hybrid ranking - sophisticated score combination"""
        # Create comprehensive ranking system
        model_ranking = {}

        # Process zero-shot preferences
        for i, model_id in enumerate(zero_shot.selected_models):
            ranking_score = (len(zero_shot.selected_models) - i) / len(
                zero_shot.selected_models
            )
            model_ranking[model_id] = {
                "zero_shot_rank": ranking_score,
                "database_score": 0.0,
                "combined_score": 0.0,
            }

        # Process database scores
        for score_contract in database.model_scores:
            model_id = score_contract.model_id
            if model_id not in model_ranking:
                model_ranking[model_id] = {
                    "zero_shot_rank": 0.0,
                    "database_score": 0.0,
                    "combined_score": 0.0,
                }
            model_ranking[model_id]["database_score"] = score_contract.total_score

        # Calculate combined scores
        for model_id, scores in model_ranking.items():
            # Hybrid ranking: 60% database score, 40% zero-shot rank
            combined = (scores["database_score"] * 0.6) + (
                scores["zero_shot_rank"] * 0.4
            )
            model_ranking[model_id]["combined_score"] = combined

        # Select top models by combined score
        sorted_models = sorted(
            model_ranking.items(), key=lambda x: x[1]["combined_score"], reverse=True
        )

        selected_models = [
            model_id for model_id, _ in sorted_models[: min(3, len(sorted_models))]
        ]

        # Create hybrid scores
        merged_scores = []
        for model_id in selected_models:
            ranking_data = model_ranking[model_id]

            merged_score = ModelScoreContract(
                model_id=model_id,
                total_score=ranking_data["combined_score"],
                component_scores={
                    "zero_shot_rank": ranking_data["zero_shot_rank"],
                    "database_score": ranking_data["database_score"],
                    "hybrid_combined": ranking_data["combined_score"],
                },
                rationale=f"Hybrid ranking (DB:{ranking_data['database_score']:.2f}, ZS:{ranking_data['zero_shot_rank']:.2f})",
                confidence=(zero_shot.confidence_score + 0.8)
                / 2.0,  # Average with database confidence
                risk_factors=["hybrid_ranking_complexity"],
                scoring_timestamp=datetime.utcnow(),
                service_version="v5_modular_hybrid",
            )
            merged_scores.append(merged_score)

        return SelectionResultContract(
            engagement_id=database.engagement_id + "_hybrid",
            selected_models=selected_models,
            model_scores=merged_scores,
            selection_source="hybrid_ranking",
            strategy_used="hybrid_ranking_merge",
            models_evaluated=len(model_ranking),
            selection_metadata={
                **database.selection_metadata,
                "ranking_weights": {"database": 0.6, "zero_shot": 0.4},
                "hybrid_ranking_applied": True,
            },
            total_selection_time_ms=database.total_selection_time_ms,
            cognitive_load_assessment=database.cognitive_load_assessment,
            selection_timestamp=datetime.utcnow(),
            service_version="v5_modular_hybrid",
        )

    def _build_selection_prompt_template(self) -> str:
        """Build the MeMo-style selection prompt template"""
        return """
        You are an expert AI model selector using the MeMo (Model Selection via Memory) methodology.
        
        TASK: Select the best AI models for this business problem through step-by-step reasoning.
        
        BUSINESS PROBLEM:
        Problem: {problem_statement}
        Business Context: {business_context}
        Problem Type: {problem_type}
        Complexity Level: {complexity_level}
        Accuracy Requirement: {accuracy_requirement}
        Max Models: {max_models}
        
        NOVELTY ANALYSIS:
        Novelty Level: {novelty_level}
        Novelty Indicators: {novelty_indicators}
        
        AVAILABLE MODELS: {available_models}
        
        REASONING PROCESS:
        Please provide step-by-step reasoning for model selection:
        
        Step 1: Problem Analysis
        - What are the key cognitive requirements?
        - What type of reasoning is needed?
        - What are the domain-specific challenges?
        
        Step 2: Model Capability Matching
        - Which models have relevant strengths?
        - How do model capabilities align with requirements?
        - What are the trade-offs between models?
        
        Step 3: Selection Decision
        - Which models should be selected and why?
        - How do selected models complement each other?
        - What is the confidence in this selection?
        
        Provide your reasoning followed by final model selection.
        """

    def _extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response (simplified)"""
        try:
            # Look for JSON-like structures in response
            import json

            # Find content between { and }
            start = response.find("{")
            end = response.rfind("}")

            if start != -1 and end != -1 and end > start:
                json_str = response[start : end + 1]
                return json.loads(json_str)

        except Exception:
            pass

        return None

    def _calculate_novelty_heuristic(self, context: SelectionContextContract) -> float:
        """Calculate novelty using heuristic approach"""
        novelty_score = 0.5  # Base novelty

        # Check for novelty keywords
        novelty_words = self._detect_novelty_keywords(context.problem_statement)
        novelty_score += len(novelty_words) * 0.1

        # High complexity often indicates novelty
        if context.complexity_level in ["high", "very_high"]:
            novelty_score += 0.2

        # High accuracy requirements may indicate novel/critical problems
        if context.accuracy_requirement >= 0.9:
            novelty_score += 0.1

        return min(novelty_score, 1.0)

    def _detect_novelty_keywords(self, text: str) -> List[str]:
        """Detect novelty keywords in text"""
        found_keywords = []
        text_lower = text.lower()

        for keyword in self.novelty_detection_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)

        return found_keywords

    def _extract_reasoning_steps(self, response: str) -> List[str]:
        """Extract reasoning steps from LLM response"""
        try:
            # Look for step patterns
            step_pattern = r"Step \d+:([^:]+(?:\n(?!Step \d+:)[^\n]*)*)"
            steps = re.findall(step_pattern, response, re.MULTILINE | re.IGNORECASE)

            if steps:
                return [step.strip() for step in steps]

            # Fallback: split by paragraphs
            paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
            return paragraphs[: self.reasoning_chain_depth]

        except Exception as e:
            self.logger.error(f"❌ Failed to extract reasoning steps: {e}")
            return [response[:200] + "..." if len(response) > 200 else response]

    def _extract_model_preferences(
        self, reasoning_chain: List[str]
    ) -> Dict[str, float]:
        """Extract model preferences from reasoning chain"""
        preferences = {}

        for model_id in self.model_capability_profiles.keys():
            preference_score = 0.5  # Base score

            # Count mentions in reasoning
            mentions = sum(
                1
                for step in reasoning_chain
                if model_id.replace("_", " ").lower() in step.lower()
            )

            if mentions > 0:
                preference_score = min(0.5 + (mentions * 0.2), 0.9)

            preferences[model_id] = preference_score

        return preferences

    def _match_model_capabilities(
        self, context: SelectionContextContract
    ) -> Dict[str, float]:
        """Match model capabilities to context requirements"""
        capability_matches = {}

        for model_id, profile in self.model_capability_profiles.items():
            match_score = 0.0
            factors = 0

            # Match complexity level
            if context.complexity_level in profile["complexity_range"]:
                match_score += 0.3
            factors += 1

            # Match business context (simplified)
            business_keywords = str(context.business_context).lower()
            for biz_context in profile["business_contexts"]:
                if biz_context in business_keywords:
                    match_score += 0.2
                    break
            factors += 1

            # Match problem type (heuristic)
            problem_type_match = self._match_problem_type(
                context.problem_type, profile["strengths"]
            )
            match_score += problem_type_match * 0.5
            factors += 1

            # Normalize by number of factors
            capability_matches[model_id] = match_score / factors if factors > 0 else 0.3

        return capability_matches

    def _match_problem_type(
        self, problem_type: str, model_strengths: List[str]
    ) -> float:
        """Match problem type to model strengths"""
        problem_keywords = problem_type.lower().split()

        for strength in model_strengths:
            strength_keywords = strength.replace("_", " ").split()
            overlap = set(problem_keywords) & set(strength_keywords)
            if overlap:
                return 0.8

        return 0.3  # Default low match

    def _extract_novelty_factors(self, reasoning_chain: List[str]) -> List[str]:
        """Extract novelty factors from reasoning chain"""
        novelty_factors = []

        for step in reasoning_chain:
            step_lower = step.lower()
            for keyword in self.novelty_detection_keywords:
                if keyword in step_lower:
                    novelty_factors.append(f"reasoning_mentions_{keyword}")

        return list(set(novelty_factors))  # Remove duplicates

    def _assess_context_clarity(self, context: SelectionContextContract) -> float:
        """Assess how clear and specific the context is"""
        clarity_score = 0.5

        # Problem statement clarity
        if len(context.problem_statement) > 50:
            clarity_score += 0.1
        if len(context.problem_statement) > 100:
            clarity_score += 0.1

        # Business context specificity
        if (
            isinstance(context.business_context, dict)
            and len(context.business_context) > 2
        ):
            clarity_score += 0.1

        # Problem type specificity
        if context.problem_type and context.problem_type != "general":
            clarity_score += 0.1

        # Complexity level specified
        if context.complexity_level and context.complexity_level != "medium":
            clarity_score += 0.1

        return min(clarity_score, 1.0)

    def _create_fallback_zero_shot_result(
        self, context: SelectionContextContract, error_msg: str
    ) -> ZeroShotSelectionContract:
        """Create fallback zero-shot result when selection fails"""
        return ZeroShotSelectionContract(
            engagement_id=context.problem_statement[:50] + "_fallback",
            selected_models=["deepseek_chat"],  # Safe default
            confidence_score=0.3,  # Low confidence for fallback
            reasoning_process=[
                f"Zero-shot selection failed: {error_msg}",
                "Defaulted to general-purpose model",
                "Fallback selection with low confidence",
            ],
            context_analysis={"error": error_msg, "fallback_triggered": True},
            novelty_factors=["fallback_selection"],
            selection_timestamp=datetime.utcnow(),
            service_version="v5_modular_fallback",
        )

    async def get_service_health(self) -> Dict[str, Any]:
        """Return service health status"""
        return {
            "service_name": "ZeroShotSelectionService",
            "status": "healthy",
            "version": "v5_modular",
            "methodology": "MeMo_based",
            "capabilities": [
                "zero_shot_model_selection",
                "memo_reasoning_chains",
                "novelty_detection",
                "confidence_assessment",
                "hybrid_merge_strategies",
            ],
            "supported_merge_strategies": [
                strategy.value for strategy in MergeStrategy
            ],
            "model_profiles_loaded": len(self.model_capability_profiles),
            "reasoning_depth": self.reasoning_chain_depth,
            "confidence_threshold": self.confidence_threshold,
            "last_health_check": datetime.utcnow().isoformat(),
        }


# Global service instance for dependency injection
_zero_shot_selection_service: Optional[ZeroShotSelectionService] = None


def get_zero_shot_selection_service() -> ZeroShotSelectionService:
    """Get or create global zero-shot selection service instance"""
    global _zero_shot_selection_service

    if _zero_shot_selection_service is None:
        _zero_shot_selection_service = ZeroShotSelectionService()

    return _zero_shot_selection_service
