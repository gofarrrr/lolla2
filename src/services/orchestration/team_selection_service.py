# src/services/orchestration/team_selection_service.py
import logging
import os
from itertools import combinations
from typing import Any, Dict, List, Tuple

from src.orchestration.contracts import (
    StructuredAnalyticalFramework,
    ConsultantBlueprint,
)

logger = logging.getLogger(__name__)


class TeamSelectionService:
    """
    Smart GM team selection extracted from DispatchOrchestrator.
    Responsibilities:
    - Get baseline consultant scores from the contextual engine (with fallback)
    - Evaluate 3-consultant combinations and compute synergy
    - Return ConsultantBlueprints and synergy metadata
    """

    def __init__(
        self,
        consultant_database: Dict[str, Dict[str, Any]],
        contextual_engine: Any,
    ) -> None:
        self.consultant_database = consultant_database
        self.contextual_engine = contextual_engine
        # Operation Primacy: YAML 5-factor scoring is the primary/default path.
        # Contextual engine can be re-enabled explicitly via env if desired.
        self.use_contextual_engine_primary = (
            os.getenv("USE_CONTEXTUAL_ENGINE_PRIMARY", "false").lower()
            in {"1", "true", "yes", "on"}
        )

    async def select_optimal_team(
        self,
        query_domain: str,
        task_classification: Dict[str, Any],
        framework: StructuredAnalyticalFramework,
    ) -> Tuple[List[ConsultantBlueprint], Dict[str, Any], List[Dict[str, Any]]]:
        logger.info(f"📀 Getting baseline scores for domain: {query_domain}")
        baseline_pool = await self._get_baseline_consultant_scores(query_domain)
        logger.info(
            "🧑‍💼 Smart GM evaluating optimal 3-consultant team combinations..."
        )
        consultant_blueprints, synergy_data = await self._select_optimal_team_of_three(
            baseline_pool, task_classification, framework
        )
        return consultant_blueprints, synergy_data, baseline_pool

    async def _get_baseline_consultant_scores(
        self, query_domain: str
    ) -> List[Dict[str, Any]]:
        """Return ranked baseline scores for consultants.

        Operation Primacy: YAML 5‑factor scoring is the default primary path. The
        legacy Contextual Engine is ONLY consulted when explicitly enabled by env
        (`USE_CONTEXTUAL_ENGINE_PRIMARY=true`). This enforces V6 primacy.
        """
        try:
            # 1) Always compute YAML 5‑factor scores first (primary path)
            yaml_scores: List[Dict[str, Any]] = []
            for consultant_id, consultant_data in self.consultant_database.items():
                yaml_enhanced_score = self._calculate_yaml_enhanced_score(
                    consultant_id, consultant_data, query_domain
                )
                yaml_scores.append(
                    {
                        "consultant_id": consultant_id,
                        "score": yaml_enhanced_score,
                        "baseline_method": "yaml_5_factor_primary",
                        "scoring_factors": consultant_data.get("scoring_factors", {}),
                    }
                )
            yaml_scores.sort(key=lambda x: x["score"], reverse=True)
            logger.info(
                f"✅ YAML 5‑factor (primary) scoring complete for {len(yaml_scores)} consultants"
            )
            if yaml_scores:
                top = yaml_scores[0]
                logger.info(
                    f"🏆 Primary YAML top consultant: {top['consultant_id']} (score: {top['score']:.3f})"
                )

            # 2) Optionally consult Contextual Engine if explicitly enabled
            if self.use_contextual_engine_primary and self.contextual_engine:
                try:
                    ce_ranked = await self.contextual_engine.calculate_synergy_scores_for_domain(
                        query_domain
                    )
                    ce_ranked.sort(key=lambda x: x["score"], reverse=True)
                    logger.info(
                        f"ℹ️ Contextual Engine consulted (opt‑in). Top: {ce_ranked[0]['consultant_id'] if ce_ranked else 'n/a'}"
                    )
                    # Return CE scores only if explicitly requested as primary.
                    return ce_ranked
                except Exception as ce_err:
                    logger.warning(
                        f"⚠️ Contextual Engine consultation failed, staying with YAML primary: {ce_err}"
                    )

            # Default: return YAML scores (primary)
            return yaml_scores
        except Exception as e:
            logger.error(f"❌ Baseline scoring failed: {e}")
            raise RuntimeError(f"Unable to get baseline consultant scores: {e}")

    async def _select_optimal_team_of_three(
        self,
        baseline_consultant_pool: List[Dict[str, Any]],
        task_classification: Dict[str, Any],
        framework: StructuredAnalyticalFramework,
    ) -> Tuple[List[ConsultantBlueprint], Dict[str, Any]]:
        logger.info(
            f"🧑‍💼 Smart GM evaluating team combinations for {task_classification['task_type']} task"
        )
        top_candidates = baseline_consultant_pool[
            : min(8, len(baseline_consultant_pool))
        ]
        if len(top_candidates) < 3:
            logger.warning(
                f"⚠️ Only {len(top_candidates)} candidates available, using all"
            )
            selected_team = top_candidates
        else:
            logger.info(
                f"🔍 Evaluating combinations from top {len(top_candidates)} candidates"
            )
            possible_teams = list(combinations(top_candidates, 3))
            logger.info(
                f"📈 Analyzing {len(possible_teams)} potential team combinations"
            )
            best_team: List[Dict[str, Any]] = []
            best_final_score = -1.0
            best_synergy_data: Dict[str, Any] = {}
            for team_combo in possible_teams:
                team_list = list(team_combo)
                avg_individual_score = sum(c["score"] for c in team_list) / len(
                    team_list
                )
                synergy_bonus = await self._calculate_team_synergy_bonus(
                    team_list, task_classification, framework
                )
                final_team_score = avg_individual_score + synergy_bonus
                if final_team_score > best_final_score:
                    best_final_score = final_team_score
                    best_team = team_list
                    best_synergy_data = {
                        "avg_individual_score": avg_individual_score,
                        "synergy_bonus": synergy_bonus,
                        "final_score": final_team_score,
                        "team_composition_logic": self._get_team_composition_reasoning(
                            team_list, task_classification, synergy_bonus
                        ),
                    }
            selected_team = best_team
        consultant_blueprints = self._convert_contextual_results_to_blueprints(
            selected_team, framework
        )
        final_synergy_data = (
            best_synergy_data
            if "best_synergy_data" in locals()
            else {
                "avg_individual_score": sum(c["score"] for c in selected_team)
                / len(selected_team),
                "synergy_bonus": 0.0,
                "final_score": sum(c["score"] for c in selected_team)
                / len(selected_team),
                "team_composition_logic": f"Optimal team selection for {task_classification['task_type']} task",
            }
        )
        logger.info("🏆 Smart GM selected optimal team:")
        logger.info(f"   Team: {[c['consultant_id'] for c in selected_team]}")
        logger.info(
            f"   Individual avg: {final_synergy_data['avg_individual_score']:.3f}"
        )
        logger.info(f"   Synergy bonus: +{final_synergy_data['synergy_bonus']:.3f}")
        logger.info(f"   Final score: {final_synergy_data['final_score']:.3f}")
        return consultant_blueprints, final_synergy_data

    def _convert_contextual_results_to_blueprints(
        self,
        contextual_results: List[Dict[str, Any]],
        framework: StructuredAnalyticalFramework,
    ) -> List[ConsultantBlueprint]:
        blueprints: List[ConsultantBlueprint] = []
        for result in contextual_results:
            consultant_id = result.get("consultant_id", "")
            if not consultant_id:
                logger.warning(
                    f"⚠️ Contextual result has no consultant_id, skipping: {result}"
                )
                continue
            consultant_data = self.consultant_database.get(consultant_id, {})
            if (
                not consultant_data
                or "expertise_areas" not in consultant_data
                or "type" not in consultant_data
            ):
                logger.warning(
                    f"⚠️ Invalid consultant data for {consultant_id}, skipping"
                )
                continue
            assigned_dimensions = self._assign_dimensions_to_consultant(
                consultant_data, framework.primary_dimensions
            )
            blueprint = ConsultantBlueprint(
                consultant_id=consultant_id,
                consultant_type=consultant_data.get("type", consultant_id),
                specialization=consultant_data.get(
                    "specialization", "General Consulting"
                ),
                predicted_effectiveness=result.get("score", 0.8),
                assigned_dimensions=assigned_dimensions,
            )
            # OPERATION UNIFICATION: Attach rich YAML consultant data for Glass Box explanations
            blueprint.consultant_data = consultant_data
            blueprints.append(blueprint)
        logger.info(
            f"🎯 Converted {len(blueprints)} contextual results to consultant blueprints"
        )
        return blueprints

    async def _calculate_team_synergy_bonus(
        self,
        team: List[Dict[str, Any]],
        task_classification: Dict[str, Any],
        framework: StructuredAnalyticalFramework,
    ) -> float:
        task_type = task_classification["task_type"]
        primary_domain = task_classification["primary_domain"]
        total_synergy_bonus = 0.0
        diversity_score = self._calculate_team_cognitive_diversity(team)
        if task_type == "analytical":
            cohesion_score = 1.0 - diversity_score
            diversity_bonus = cohesion_score * 0.15
            total_synergy_bonus += diversity_bonus
        else:
            diversity_bonus = diversity_score * 0.20
            total_synergy_bonus += diversity_bonus
            insight_potential_bonus = await self._calculate_insight_potential_bonus(
                team
            )
            total_synergy_bonus += insight_potential_bonus
        domain_alignment_bonus = self._calculate_domain_expertise_bonus(
            team, primary_domain, framework
        )
        total_synergy_bonus += domain_alignment_bonus
        chemistry_bonus = self._calculate_team_chemistry_bonus(team, task_type)
        total_synergy_bonus += chemistry_bonus
        max_bonus = 0.4 if task_type == "ideation" else 0.3
        total_synergy_bonus = min(total_synergy_bonus, max_bonus)
        return total_synergy_bonus

    def _calculate_domain_expertise_bonus(
        self,
        team: List[Dict[str, Any]],
        primary_domain: str,
        framework: StructuredAnalyticalFramework,
    ) -> float:
        domain_expertise_count = 0
        total_domain_strength = 0.0
        domain_keywords = {
            "strategy": ["strategy", "strategic", "market", "competitive"],
            "finance": ["financial", "finance", "cost", "budget", "revenue"],
            "operations": ["operational", "operations", "process", "efficiency"],
            "creative": ["creative", "innovation", "design", "ideation"],
            "marketing": ["marketing", "brand", "campaign", "promotion"],
            "technology": ["technology", "tech", "digital", "systems"],
        }
        relevant_keywords = domain_keywords.get(primary_domain, ["strategy"])
        for consultant in team:
            consultant_data = self.consultant_database.get(
                consultant["consultant_id"], {}
            )
            expertise_areas = consultant_data.get("expertise_areas", [])
            expertise_match_strength = 0.0
            for expertise in expertise_areas:
                el = expertise.lower()
                for keyword in relevant_keywords:
                    if keyword in el:
                        expertise_match_strength += 1.0
                        break
            if expertise_match_strength > 0:
                domain_expertise_count += 1
                total_domain_strength += expertise_match_strength
        if domain_expertise_count >= 2:
            domain_bonus = min(0.10, total_domain_strength * 0.05)
        elif domain_expertise_count == 1:
            domain_bonus = min(0.05, total_domain_strength * 0.03)
        else:
            domain_bonus = -0.02
        return domain_bonus

    def _calculate_team_chemistry_bonus(
        self, team: List[Dict[str, Any]], task_type: str
    ) -> float:
        chemistry_bonus = 0.0
        mental_model_overlaps: List[float] = []
        for i, consultant_a in enumerate(team):
            for consultant_b in team[i + 1 :]:
                data_a = self.consultant_database.get(consultant_a["consultant_id"], {})
                data_b = self.consultant_database.get(consultant_b["consultant_id"], {})
                models_a = set(data_a.get("mental_models", []))
                models_b = set(data_b.get("mental_models", []))
                if models_a and models_b:
                    overlap_ratio = len(models_a & models_b) / len(models_a | models_b)
                    mental_model_overlaps.append(overlap_ratio)
        if mental_model_overlaps:
            avg_overlap = sum(mental_model_overlaps) / len(mental_model_overlaps)
            if task_type == "analytical":
                if 0.2 <= avg_overlap <= 0.6:
                    chemistry_bonus = 0.08
                elif avg_overlap > 0.6:
                    chemistry_bonus = 0.04
                else:
                    chemistry_bonus = 0.02
            else:
                if avg_overlap < 0.3:
                    chemistry_bonus = 0.10
                elif avg_overlap < 0.5:
                    chemistry_bonus = 0.06
                else:
                    chemistry_bonus = 0.02
        try:
            categories_present = set()
            for member in team:
                data = self.consultant_database.get(member["consultant_id"], {})
                for model in data.get("mental_models", []):
                    ml = model.lower()
                    if any(k in ml for k in ["system", "feedback", "loops"]):
                        categories_present.add("systems")
                    if any(
                        k in ml
                        for k in [
                            "decision",
                            "expected value",
                            "ev",
                            "cost",
                            "benefit",
                            "trade-off",
                            "pareto",
                        ]
                    ):
                        categories_present.add("decision")
                    if any(
                        k in ml
                        for k in [
                            "design thinking",
                            "creative",
                            "innovation",
                            "divergent",
                        ]
                    ):
                        categories_present.add("creative")
                    if any(
                        k in ml
                        for k in ["risk", "scenario", "mitigation", "uncertainty"]
                    ):
                        categories_present.add("risk")
                    if any(
                        k in ml for k in ["execution", "lean", "kanban", "pmo", "raci"]
                    ):
                        categories_present.add("execution")
                    if any(k in ml for k in ["finance", "npv", "dcf", "valuation"]):
                        categories_present.add("finance")
            coverage = len(categories_present)
            category_bonus = min(0.08, 0.02 * min(coverage, 5))
            chemistry_bonus += category_bonus
        except Exception:
            pass
        return chemistry_bonus

    def _get_team_composition_reasoning(
        self,
        team: List[Dict[str, Any]],
        task_classification: Dict[str, Any],
        synergy_bonus: float,
    ) -> str:
        task_type = task_classification["task_type"]
        primary_domain = task_classification["primary_domain"]
        team_ids = [c["consultant_id"] for c in team]
        diversity_score = self._calculate_team_cognitive_diversity(team)
        if task_type == "analytical":
            reasoning = (
                f"Selected cohesive analytical team for {primary_domain} domain. "
                f"Team composition optimized for analytical rigor with diversity score of {diversity_score:.3f} "
                f"(lower is better for analytical tasks). "
            )
            if synergy_bonus > 0.15:
                reasoning += "High synergy bonus (+{:.3f}) due to excellent team cohesion and domain alignment.".format(
                    synergy_bonus
                )
            elif synergy_bonus > 0.08:
                reasoning += "Good synergy bonus (+{:.3f}) from solid analytical team composition.".format(
                    synergy_bonus
                )
            else:
                reasoning += "Modest synergy bonus (+{:.3f}) - room for improvement in team cohesion.".format(
                    synergy_bonus
                )
        else:
            reasoning = (
                f"Selected diverse ideation team for {primary_domain} domain. "
                f"Team composition optimized for cognitive diversity with diversity score of {diversity_score:.3f} "
                f"(higher is better for ideation tasks). "
            )
            if synergy_bonus > 0.25:
                reasoning += "Exceptional synergy bonus (+{:.3f}) due to outstanding cognitive diversity.".format(
                    synergy_bonus
                )
            elif synergy_bonus > 0.15:
                reasoning += "Strong synergy bonus (+{:.3f}) from excellent diverse team composition.".format(
                    synergy_bonus
                )
            else:
                reasoning += "Standard synergy bonus (+{:.3f}) - good baseline team diversity.".format(
                    synergy_bonus
                )
        reasoning += f" Team: {', '.join(team_ids)}."
        return reasoning

    async def _calculate_insight_potential_bonus(
        self, team: List[Dict[str, Any]]
    ) -> float:
        try:
            from src.core.supabase_platform import SupabasePlatform

            db = SupabasePlatform()
            total_insight_potential = 0.0
            nway_count = 0
            for consultant in team:
                consultant_id = consultant["consultant_id"]
                try:
                    query = (
                        "SELECT insight_potential_score, nway_type, interaction_id "
                        "FROM nway_interactions WHERE insight_potential_score >= 0.7 "
                        "ORDER BY insight_potential_score DESC LIMIT 3"
                    )
                    high_insight_nways = await db.fetch(query)
                    if high_insight_nways:
                        best_nway = high_insight_nways[0]
                        insight_score = best_nway.get("insight_potential_score", 0.0)
                        total_insight_potential += insight_score
                        nway_count += 1
                        logger.debug(
                            f"🧠 Consultant {consultant_id} assigned high-insight NWAY {best_nway.get('interaction_id')} "
                            f"(insight={insight_score:.3f})"
                        )
                except Exception as e:
                    logger.warning(
                        f"⚠️ Could not fetch NWAY insight data for {consultant_id}: {e}"
                    )
                    total_insight_potential += 0.6
                    nway_count += 1
            if nway_count == 0:
                return 0.0
            avg_insight_potential = total_insight_potential / nway_count
            if avg_insight_potential >= 0.9:
                insight_bonus = 0.35
            elif avg_insight_potential >= 0.8:
                insight_bonus = 0.25
            elif avg_insight_potential >= 0.7:
                insight_bonus = 0.15
            else:
                insight_bonus = 0.0
            logger.info(
                f"🎯 N-Way Calibration Insight Bonus: Avg Insight Potential={avg_insight_potential:.3f} → Bonus=+{insight_bonus:.3f}"
            )
            return insight_bonus
        except Exception as e:
            logger.error(f"❌ Error calculating insight potential bonus: {e}")
            return 0.0

    def _calculate_team_cognitive_diversity(self, team: List[Dict[str, Any]]) -> float:
        if len(team) < 2:
            return 0.0
        diversity_factors: List[float] = []
        consultant_types: List[str] = []
        for consultant in team:
            consultant_data = self.consultant_database.get(
                consultant["consultant_id"], {}
            )
            consultant_types.append(consultant_data.get("type", "unknown"))
        type_diversity = len(set(consultant_types)) / len(consultant_types)
        diversity_factors.append(type_diversity)
        all_expertise = set()
        individual_expertise: List[set] = []
        for consultant in team:
            consultant_data = self.consultant_database.get(
                consultant["consultant_id"], {}
            )
            expertise = set(consultant_data.get("expertise_areas", []))
            individual_expertise.append(expertise)
            all_expertise.update(expertise)
        if len(individual_expertise) >= 2:
            jaccard_distances: List[float] = []
            for i in range(len(individual_expertise)):
                for j in range(i + 1, len(individual_expertise)):
                    intersection = len(
                        individual_expertise[i] & individual_expertise[j]
                    )
                    union = len(individual_expertise[i] | individual_expertise[j])
                    jaccard_distance = 1 - (intersection / union if union > 0 else 0)
                    jaccard_distances.append(jaccard_distance)
            expertise_diversity = sum(jaccard_distances) / len(jaccard_distances)
            diversity_factors.append(expertise_diversity)
        all_models = set()
        for consultant in team:
            consultant_data = self.consultant_database.get(
                consultant["consultant_id"], {}
            )
            models = set(consultant_data.get("mental_models", []))
            all_models.update(models)
        total_unique_models = len(all_models)
        expected_models_per_consultant = 4
        expected_total = expected_models_per_consultant * len(team)
        model_diversity = (
            min(1.0, total_unique_models / expected_total) if expected_total else 0.0
        )
        diversity_factors.append(model_diversity)
        overall_diversity = sum(diversity_factors) / len(diversity_factors)
        return overall_diversity

    def _assign_dimensions_to_consultant(
        self, consultant_data: Dict[str, Any], dimensions: List[Any]
    ) -> List[str]:
        assigned: List[str] = []
        consultant_expertise = consultant_data["expertise_areas"]
        consultant_type = consultant_data["type"]
        dimension_scores: List[Tuple[str, float]] = []
        for dimension in dimensions:
            dimension_words = dimension.dimension_name.lower().split()
            score = 0.0
            for expertise in consultant_expertise:
                if any(expertise in word for word in dimension_words):
                    score += 1.0
                elif any(word in expertise for word in dimension_words):
                    score += 0.5
            if consultant_type == "financial_analyst" and any(
                word in dimension.dimension_name.lower()
                for word in ["financial", "cost", "revenue", "economic"]
            ):
                score += 0.8
            elif consultant_type == "strategic_analyst" and any(
                word in dimension.dimension_name.lower()
                for word in ["strategic", "market", "competitive", "business"]
            ):
                score += 0.8
            elif consultant_type == "market_researcher" and any(
                word in dimension.dimension_name.lower()
                for word in ["market", "customer", "competitive"]
            ):
                score += 0.8
            dimension_scores.append((dimension.dimension_name, score))
        dimension_scores.sort(key=lambda x: x[1], reverse=True)
        for dim_name, score in dimension_scores:
            if score > 0.3 or len(assigned) < 1:
                assigned.append(dim_name)
                if len(assigned) >= 3:
                    break
        if not assigned and dimensions:
            highest_priority_dim = min(dimensions, key=lambda d: d.priority_level)
            assigned.append(highest_priority_dim.dimension_name)
        return assigned

    def _calculate_yaml_enhanced_score(
        self, 
        consultant_id: str, 
        consultant_data: Dict[str, Any], 
        query_domain: str
    ) -> float:
        """
        OPERATION UNIFICATION: Enhanced scoring using rich YAML cognitive profiles.
        
        Leverages YAML persona data for intelligent consultant selection:
        - mental_model_affinities for cognitive compatibility
        - identity for domain expertise matching
        - cognitive_signature for problem-solving style alignment
        - source_nway for NWAY pattern alignment
        
        Args:
            consultant_id: ID of the consultant
            consultant_data: Rich YAML persona data
            query_domain: Domain being analyzed
            
        Returns:
            Enhanced score (0.0-1.0) based on YAML cognitive profile
        """
        base_score = 0.5  # Start with neutral baseline
        scoring_factors = {}
        
        # 1. Legacy framework affinity (20% weight)
        framework_score = 0.0
        if "framework_affinity" in consultant_data:
            affinities = consultant_data["framework_affinity"].values()
            framework_score = sum(affinities) / len(affinities) if affinities else 0.0
        scoring_factors["framework_affinity"] = framework_score
        
        # 2. YAML mental model affinity matching (30% weight) 
        mental_model_score = 0.0
        if "mental_model_affinities" in consultant_data:
            mental_models = consultant_data["mental_model_affinities"]
            if isinstance(mental_models, dict):
                # Domain-specific scoring based on mental models
                domain_keywords = query_domain.lower().split()
                relevant_models = []
                
                for model_name, model_desc in mental_models.items():
                    model_text = f"{model_name} {model_desc}".lower()
                    # Check if model is relevant to query domain
                    relevance = sum(1 for keyword in domain_keywords if keyword in model_text)
                    if relevance > 0:
                        relevant_models.append(relevance)
                        
                if relevant_models:
                    mental_model_score = min(1.0, sum(relevant_models) / (len(relevant_models) * 2))
                else:
                    mental_model_score = 0.6  # Neutral if no direct match
            scoring_factors["mental_model_affinity"] = mental_model_score
        
        # 3. Identity-based domain expertise (25% weight)
        identity_score = 0.0
        if "identity" in consultant_data:
            identity = consultant_data["identity"].lower()
            domain_keywords = query_domain.lower().split()
            
            # Check for domain expertise indicators in identity
            expertise_indicators = [
                "expert", "specialist", "consultant", "advisor", "analyst", 
                "manager", "director", "strategist", "architect", "engineer"
            ]
            
            domain_match = sum(1 for keyword in domain_keywords if keyword in identity)
            expertise_match = sum(1 for indicator in expertise_indicators if indicator in identity)
            
            identity_score = min(1.0, (domain_match * 0.3 + expertise_match * 0.1))
            scoring_factors["identity_domain_match"] = identity_score
        
        # 4. Cognitive signature alignment (15% weight)
        cognitive_score = 0.0
        if "cognitive_signature" in consultant_data:
            signature = consultant_data["cognitive_signature"].lower()
            
            # Higher scores for analytical and systematic approaches
            analytical_indicators = [
                "analytical", "systematic", "logical", "structured", "methodical",
                "rigorous", "evidence", "data", "scientific", "quantitative"
            ]
            
            cognitive_matches = sum(1 for indicator in analytical_indicators if indicator in signature)
            cognitive_score = min(1.0, cognitive_matches * 0.15)
            scoring_factors["cognitive_signature"] = cognitive_score
        
        # 5. NWAY source pattern bonus (10% weight)
        nway_bonus = 0.0
        if "source_nway" in consultant_data:
            source_nway = consultant_data["source_nway"]
            # Different NWAY patterns have different strengths
            nway_weights = {
                "NWAY_DECOMPOSITION": 0.9,  # Great for analytical problems
                "NWAY_DECISION": 0.8,       # Good for decision-making
                "NWAY_PERCEPTION": 0.7,     # Good for pattern recognition
                "NWAY_REASONING": 0.8,      # Good for logical analysis
                "NWAY_SYNTHESIS": 0.6,      # Good for creative synthesis
            }
            
            for pattern, weight in nway_weights.items():
                if pattern in source_nway:
                    nway_bonus = weight
                    break
            scoring_factors["nway_pattern_bonus"] = nway_bonus
        
        # Calculate weighted final score
        final_score = (
            base_score * 0.0 +           # No base score weight
            framework_score * 0.20 +     # Legacy framework affinity
            mental_model_score * 0.30 +  # YAML mental models (highest weight)
            identity_score * 0.25 +      # Identity domain expertise
            cognitive_score * 0.15 +     # Cognitive signature
            nway_bonus * 0.10           # NWAY pattern bonus
        )
        
        # Ensure score is in valid range
        final_score = max(0.0, min(1.0, final_score))
        
        # Store scoring factors for transparency
        consultant_data["scoring_factors"] = scoring_factors
        consultant_data["yaml_enhanced_score"] = final_score
        
        logger.debug(
            f"🧮 YAML scoring for {consultant_id}: "
            f"final={final_score:.3f} "
            f"(framework={framework_score:.2f}, "
            f"mental_models={mental_model_score:.2f}, "
            f"identity={identity_score:.2f}, "
            f"cognitive={cognitive_score:.2f}, "
            f"nway={nway_bonus:.2f})"
        )
        
        return final_score
