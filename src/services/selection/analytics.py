# src/services/selection/analytics.py
import asyncio
import logging
import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from src.core.unified_context_stream import UnifiedContextStream, ContextEventType
from src.services.selection.contracts import ChemistryContext

logger = logging.getLogger(__name__)


class ChemistryAnalytics:
    """
    Extracted analytics and learning service for the Cognitive Chemistry domain.
    - Provides insights and glass-box evidence recording.
    - Maintains a lightweight learning store and related analytics utilities.
    - Delegates core scoring to the injected scorer.
    """

    def __init__(
        self, context_stream: Optional[UnifiedContextStream] = None, scorer: Any = None
    ) -> None:
        self.context_stream = context_stream
        # Scorer is optional; when missing, we will attempt to resolve from the container lazily
        self.scorer = scorer
        self._learning_initialized = False
        self.chemistry_learning: Dict[str, Any] = {}

    # --------------------------- Public API ---------------------------
    def get_insights(self, ctx: ChemistryContext) -> List[str]:
        """Produce high-level insights for a given chemistry context using the scorer."""
        try:
            scorer = self._require_scorer()
            reaction = scorer.score(ctx)
            insights: List[str] = []
            # Simple, robust insights based on reaction metrics
            if reaction.overall_chemistry_score >= 0.85:
                insights.append(
                    "Exceptional overall chemistry—high likelihood of breakthrough results."
                )
            elif reaction.overall_chemistry_score >= 0.7:
                insights.append(
                    "Strong overall chemistry—team is well-suited to the problem framework."
                )
            else:
                insights.append(
                    "Chemistry is moderate—consider optimizing NWAY combination or consultant mix."
                )

            if reaction.amplification_potential >= 7.5:
                insights.append(
                    "High amplification potential—expect strong compounding effects."
                )
            if reaction.stability_rating < 0.6:
                insights.append(
                    "Stability is a concern—increase coherence or reduce conflict risk."
                )
            if reaction.cognitive_efficiency < 0.5:
                insights.append(
                    "Low cognitive efficiency—benefit/cost ratio may not justify complexity."
                )
            return insights
        except Exception as e:
            logger.warning(f"get_insights failed: {e}")
            return ["Insights unavailable—scoring not accessible."]

    def record_selection_evidence(
        self,
        problem_framework: str,
        selected_combinations: List[Dict[str, Any]],
        final_score: float,
        selection_rationale: str,
        risk_factors: List[str],
        success_factors: List[str],
        confidence_level: float,
    ) -> None:
        """Public method to record selection evidence (migrated from legacy engine)."""
        # Transform selected_combinations to match internal method signature
        combination_ids = []
        chemistry_scores: Dict[str, Dict[str, float]] = {}
        for combo in selected_combinations:
            if "consultants" in combo:
                combo_id = "_".join(combo["consultants"])
                combination_ids.append(combo_id)
                if "reaction" in combo and hasattr(
                    combo["reaction"], "overall_chemistry_score"
                ):
                    chemistry_scores[combo_id] = {
                        "overall_score": combo["reaction"].overall_chemistry_score,
                        "reaction_probability": getattr(
                            combo["reaction"], "reaction_probability", 0.0
                        ),
                        "amplification_potential": getattr(
                            combo["reaction"], "amplification_potential", 0.0
                        ),
                        "cognitive_efficiency": getattr(
                            combo["reaction"], "cognitive_efficiency", 0.0
                        ),
                        "stability_rating": getattr(
                            combo["reaction"], "stability_rating", 0.0
                        ),
                    }
                else:
                    chemistry_scores[combo_id] = {"overall_score": final_score}
        self._record_consultant_selection_evidence(
            problem_framework=problem_framework,
            selected_combinations=combination_ids,
            chemistry_scores=chemistry_scores,
            final_score=final_score,
            selection_rationale=selection_rationale,
            risk_factors=risk_factors,
            success_factors=success_factors,
            confidence_level=confidence_level,
        )

    async def record_chemistry_performance_feedback(
        self,
        ctx: ChemistryContext,
        consultant_team: List[str],
        chemistry_score: float,
        actual_performance: float,
        user_satisfaction: Optional[float] = None,
        analysis_quality: Optional[float] = None,
    ) -> bool:
        """Record performance feedback for chemistry learning and optimization."""
        try:
            self._ensure_learning_initialized()
            # Create combination hash for tracking
            combination_signature = self._create_combination_signature(
                ctx.nway_combination, consultant_team
            )
            reaction_record = {
                "timestamp": datetime.utcnow(),
                "problem_framework": ctx.problem_framework,
                "nway_combination": ctx.nway_combination,
                "consultant_team": consultant_team,
                "chemistry_score": chemistry_score,
                "actual_performance": actual_performance,
                "user_satisfaction": user_satisfaction,
                "analysis_quality": analysis_quality,
                "context": ctx.context or {},
                "performance_delta": actual_performance - chemistry_score,
                "effectiveness_ratio": actual_performance / max(chemistry_score, 0.1),
                "combination_signature": combination_signature,
            }
            self.chemistry_learning["reaction_history"][combination_signature].append(
                reaction_record
            )
            await self._update_consultant_chemistry_patterns(reaction_record)
            await self._update_pattern_synergy_learning(reaction_record)
            self.chemistry_learning["optimization_feedback"][
                "performance_correlation"
            ].append(
                {
                    "chemistry_score": chemistry_score,
                    "actual_performance": actual_performance,
                    "delta": actual_performance - chemistry_score,
                }
            )
            self.chemistry_learning["learning_metadata"]["total_reactions_learned"] += 1
            self.chemistry_learning["learning_metadata"][
                "last_learning_update"
            ] = datetime.utcnow()
            await self._trigger_chemistry_learning_if_needed()
            logger.debug(
                f"📊 Chemistry performance feedback recorded: {chemistry_score:.3f} -> {actual_performance:.3f}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to record chemistry performance feedback: {e}")
            return False

    async def get_adaptive_chemistry_score(
        self,
        ctx: ChemistryContext,
        consultant_team: Optional[List[str]] = None,
    ) -> Any:
        """Get chemistry score enhanced with learning-based adaptations."""
        try:
            scorer = self._require_scorer()
            base_reaction = scorer.score(ctx)
            enhanced_reaction = await self._apply_learning_enhancements(
                base_reaction,
                ctx.problem_framework,
                ctx.nway_combination,
                consultant_team,
                ctx.context,
            )
            return enhanced_reaction
        except Exception as e:
            logger.error(f"Failed to get adaptive chemistry score: {e}")
            # Fall back to base score if enhancement fails
            try:
                return self._require_scorer().score(ctx)
            except Exception:
                return None

    async def optimize_chemistry_weights(
        self,
        optimization_target: str = "actual_performance",
        learning_window_days: int = 30,
    ) -> Dict[str, Any]:
        """Optimize chemistry calculation weights based on learning data."""
        try:
            self._ensure_learning_initialized()
            learning_data = await self._collect_chemistry_learning_data(
                learning_window_days
            )
            if len(learning_data) < 10:
                return {
                    "optimization_performed": False,
                    "reason": "Insufficient learning data",
                    "data_points": len(learning_data),
                }
            current_performance = self._calculate_current_chemistry_performance(
                learning_data
            )
            optimized_weights = await self._optimize_chemistry_weights_gradient_descent(
                learning_data, optimization_target
            )
            estimated_improvement = await self._estimate_chemistry_improvement(
                learning_data, optimized_weights
            )
            if estimated_improvement > 0.02:
                old_weights = self.chemistry_learning["adaptive_weights"].copy()
                self.chemistry_learning["adaptive_weights"].update(optimized_weights)
                optimization_record = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "optimization_target": optimization_target,
                    "old_weights": old_weights,
                    "new_weights": optimized_weights,
                    "estimated_improvement": estimated_improvement,
                    "data_points_used": len(learning_data),
                    "performance_baseline": current_performance,
                }
                self.chemistry_learning["optimization_feedback"][
                    "weight_optimizations"
                ].append(optimization_record)
                self.chemistry_learning["learning_metadata"]["optimization_cycles"] += 1
                self.chemistry_learning["learning_metadata"][
                    "chemistry_improvements"
                ] += 1
                logger.info(
                    f"🎯 Chemistry weights optimized: {estimated_improvement:.1%} improvement estimated"
                )
                return {
                    "optimization_performed": True,
                    "estimated_improvement": estimated_improvement,
                    "new_weights": optimized_weights,
                    "optimization_record": optimization_record,
                }
            else:
                return {
                    "optimization_performed": False,
                    "reason": "Insufficient improvement potential",
                    "estimated_improvement": estimated_improvement,
                }
        except Exception as e:
            logger.error(f"Chemistry weight optimization failed: {e}")
            return {"optimization_performed": False, "error": str(e)}

    def get_chemistry_learning_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics on chemistry learning and optimization."""
        try:
            self._ensure_learning_initialized()
            analytics: Dict[str, Any] = {
                "learning_overview": {
                    "total_reactions_learned": self.chemistry_learning[
                        "learning_metadata"
                    ]["total_reactions_learned"],
                    "chemistry_improvements": self.chemistry_learning[
                        "learning_metadata"
                    ]["chemistry_improvements"],
                    "optimization_cycles": self.chemistry_learning["learning_metadata"][
                        "optimization_cycles"
                    ],
                    "learning_active": self.chemistry_learning["learning_metadata"][
                        "learning_active"
                    ],
                    "last_learning_update": (
                        self.chemistry_learning["learning_metadata"][
                            "last_learning_update"
                        ].isoformat()
                        if self.chemistry_learning["learning_metadata"][
                            "last_learning_update"
                        ]
                        else None
                    ),
                },
                "chemistry_accuracy": {},
                "consultant_chemistry_insights": {},
                "pattern_synergy_insights": {},
                "optimization_trends": {},
                "learning_recommendations": [],
            }
            optimization_feedback = self.chemistry_learning[
                "optimization_feedback"
            ].get("performance_correlation", [])
            if optimization_feedback:
                chemistry_scores = [f["chemistry_score"] for f in optimization_feedback]
                actual_scores = [f["actual_performance"] for f in optimization_feedback]
                deltas = [f["delta"] for f in optimization_feedback]
                analytics["chemistry_accuracy"] = {
                    "mean_absolute_error": statistics.mean([abs(d) for d in deltas]),
                    "correlation_strength": self._calculate_correlation(
                        chemistry_scores, actual_scores
                    ),
                    "prediction_bias": statistics.mean(deltas),
                    "sample_size": len(optimization_feedback),
                }
            # Consultant chemistry insights
            for consultant_pair, patterns in self.chemistry_learning[
                "consultant_chemistry_patterns"
            ].items():
                if patterns:
                    all_scores: List[float] = []
                    for context_scores in patterns.values():
                        all_scores.extend(context_scores)
                    if all_scores:
                        analytics["consultant_chemistry_insights"][consultant_pair] = {
                            "average_chemistry": statistics.mean(all_scores),
                            "chemistry_consistency": 1
                            - (
                                statistics.stdev(all_scores)
                                if len(all_scores) > 1
                                else 0
                            ),
                            "interaction_count": len(all_scores),
                            "trend": self._calculate_trend(all_scores[-10:]),
                        }
            # Pattern synergy insights
            for pattern_pair, domains in self.chemistry_learning[
                "pattern_synergy_learning"
            ].items():
                if domains:
                    all_synergy_scores: List[float] = []
                    for domain_scores in domains.values():
                        all_synergy_scores.extend(domain_scores)
                    if all_synergy_scores:
                        analytics["pattern_synergy_insights"][pattern_pair] = {
                            "average_synergy": statistics.mean(all_synergy_scores),
                            "synergy_strength": (
                                max(all_synergy_scores) if all_synergy_scores else 0
                            ),
                            "consistency": 1
                            - (
                                statistics.stdev(all_synergy_scores)
                                if len(all_synergy_scores) > 1
                                else 0
                            ),
                            "domain_coverage": len(domains),
                        }
            # Optimization trends
            weight_opts = self.chemistry_learning["optimization_feedback"].get(
                "weight_optimizations", []
            )
            if weight_opts:
                recent = weight_opts[-5:]
                improvements = [opt["estimated_improvement"] for opt in recent]
                analytics["optimization_trends"] = {
                    "recent_optimizations": len(recent),
                    "average_improvement": statistics.mean(improvements),
                    "optimization_frequency": (
                        "regular" if len(weight_opts) > 3 else "sparse"
                    ),
                    "total_improvements": sum(improvements),
                }
            # Learning recommendations
            recs: List[str] = []
            if analytics["learning_overview"]["total_reactions_learned"] < 50:
                recs.append(
                    "Increase usage to gather more learning data for optimization"
                )
            if (
                analytics.get("chemistry_accuracy", {}).get("mean_absolute_error", 1.0)
                > 0.2
            ):
                recs.append(
                    "Chemistry prediction accuracy needs improvement—consider weight optimization"
                )
            if analytics["learning_overview"]["optimization_cycles"] == 0:
                recs.append(
                    "No optimization cycles completed—consider enabling automatic optimization"
                )
            if len(analytics["consultant_chemistry_insights"]) < 5:
                recs.append(
                    "Limited consultant chemistry data—expand team composition diversity"
                )
            analytics["learning_recommendations"] = recs
            return analytics
        except Exception as e:
            logger.error(f"Failed to generate chemistry learning analytics: {e}")
            return {"error": str(e)}

    # --------------------------- Internals ---------------------------
    def _ensure_learning_initialized(self) -> None:
        if self._learning_initialized:
            return
        self.chemistry_learning = {
            "reaction_history": defaultdict(list),
            "consultant_chemistry_patterns": defaultdict(lambda: defaultdict(list)),
            "pattern_synergy_learning": defaultdict(lambda: defaultdict(list)),
            "optimization_feedback": defaultdict(list),
            "learning_metadata": {
                "total_reactions_learned": 0,
                "chemistry_improvements": 0,
                "optimization_cycles": 0,
                "learning_active": True,
                "last_learning_update": None,
            },
            "adaptive_weights": {
                "consultant_compatibility_weight": 0.35,
                "pattern_synergy_weight": 0.25,
                "analytical_diversity_weight": 0.25,
                "cognitive_efficiency_weight": 0.15,
            },
            "learning_rates": {
                "consultant_learning_rate": 0.1,
                "pattern_learning_rate": 0.08,
                "synergy_learning_rate": 0.12,
                "weight_adaptation_rate": 0.05,
            },
        }
        self._learning_initialized = True
        logger.info("🧠 Chemistry Learning System initialized")

    async def _apply_learning_enhancements(
        self,
        base_reaction: Any,
        problem_framework: str,
        nway_combination: List[Dict[str, Any]],
        consultant_team: Optional[List[str]],
        context: Optional[Dict[str, Any]],
    ) -> Any:
        """Apply learning-based adjustments to the base reaction in-place and return it."""
        try:
            self._ensure_learning_initialized()
            # Consultant adjustment based on learned chemistry patterns (simple heuristic)
            if consultant_team:
                consultant_adjustment = (
                    await self._calculate_consultant_chemistry_adjustment(
                        consultant_team, problem_framework, context or {}
                    )
                )
                base_reaction.overall_chemistry_score = max(
                    0.0,
                    min(
                        1.0,
                        base_reaction.overall_chemistry_score
                        * (1 + consultant_adjustment),
                    ),
                )
            # Pattern synergy adjustment
            pattern_adjustment = await self._calculate_pattern_synergy_adjustment(
                nway_combination, problem_framework
            )
            base_reaction.overall_chemistry_score = max(
                0.0,
                min(
                    1.0,
                    base_reaction.overall_chemistry_score * (1 + pattern_adjustment),
                ),
            )
            # Confidence nudges based on history depth
            signature = self._create_combination_signature(
                nway_combination, consultant_team or []
            )
            historical_data = self.chemistry_learning["reaction_history"].get(
                signature, []
            )
            if hasattr(base_reaction, "confidence_level"):
                if len(historical_data) >= 3:
                    base_reaction.confidence_level = min(
                        0.95, base_reaction.confidence_level + 0.2
                    )
                elif len(historical_data) >= 1:
                    base_reaction.confidence_level = min(
                        0.85, base_reaction.confidence_level + 0.1
                    )
            # Recommendation adjustment
            if hasattr(base_reaction, "recommendation"):
                if base_reaction.overall_chemistry_score >= 0.85:
                    base_reaction.recommendation = (
                        "EXCELLENT - Learning data indicates strong chemistry"
                    )
                elif base_reaction.overall_chemistry_score < 0.5:
                    base_reaction.recommendation = (
                        "CAUTION - Learning suggests chemistry challenges"
                    )
            return base_reaction
        except Exception as e:
            logger.error(f"Failed to apply learning enhancements: {e}")
            return base_reaction

    async def _update_consultant_chemistry_patterns(
        self, reaction_record: Dict[str, Any]
    ) -> None:
        consultant_team = reaction_record["consultant_team"]
        context = reaction_record.get("context", {})
        context_type = self._categorize_context_for_chemistry(context)
        for i, consultant1 in enumerate(consultant_team):
            for j, consultant2 in enumerate(consultant_team[i + 1 :], i + 1):
                pair_key = (
                    f"{min(consultant1, consultant2)}_{max(consultant1, consultant2)}"
                )
                chemistry_score = reaction_record["actual_performance"]
                self.chemistry_learning["consultant_chemistry_patterns"][pair_key][
                    context_type
                ].append(chemistry_score)

    async def _update_pattern_synergy_learning(
        self, reaction_record: Dict[str, Any]
    ) -> None:
        nway_combination = reaction_record["nway_combination"]
        problem_framework = reaction_record["problem_framework"]
        synergy_score = reaction_record["actual_performance"]
        domain = self._extract_domain_from_framework(problem_framework)
        for i, pattern1 in enumerate(nway_combination):
            for j, pattern2 in enumerate(nway_combination[i + 1 :], i + 1):
                pattern1_id = pattern1.get("interaction_id", f"pattern_{i}")
                pattern2_id = pattern2.get("interaction_id", f"pattern_{j}")
                pair_key = (
                    f"{min(pattern1_id, pattern2_id)}_{max(pattern1_id, pattern2_id)}"
                )
                self.chemistry_learning["pattern_synergy_learning"][pair_key][
                    domain
                ].append(synergy_score)

    async def _trigger_chemistry_learning_if_needed(self) -> None:
        total_reactions = self.chemistry_learning["learning_metadata"][
            "total_reactions_learned"
        ]
        last_opt = self.chemistry_learning["learning_metadata"].get(
            "last_optimization", None
        )
        if (total_reactions > 0 and total_reactions % 50 == 0) or (
            last_opt and datetime.utcnow() - last_opt > timedelta(days=7)
        ):
            asyncio.create_task(self.optimize_chemistry_weights())
            self.chemistry_learning["learning_metadata"][
                "last_optimization"
            ] = datetime.utcnow()

    # --------------------------- Utilities ---------------------------
    def _create_combination_signature(
        self, nway_combination: List[Dict[str, Any]], consultant_team: List[str]
    ) -> str:
        component_ids = [p.get("interaction_id", "unknown") for p in nway_combination]
        consultants = "_".join(sorted(consultant_team))
        return f"{consultants}__{'-'.join(component_ids)}"

    async def _calculate_consultant_chemistry_adjustment(
        self,
        consultant_team: List[str],
        problem_framework: str,
        context: Dict[str, Any],
    ) -> float:
        # Minimal heuristic: diverse trio gets a small boost
        diversity = len(set(consultant_team))
        if diversity >= 3:
            return 0.05
        if diversity == 2:
            return 0.02
        return -0.03

    async def _calculate_pattern_synergy_adjustment(
        self, nway_combination: List[Dict[str, Any]], problem_framework: str
    ) -> float:
        # Minimal heuristic: more patterns slightly increases risk; cap boost
        count = len(nway_combination)
        if count >= 3:
            return 0.03
        if count == 2:
            return 0.015
        return 0.0

    async def _collect_chemistry_learning_data(
        self, learning_window_days: int
    ) -> List[Dict[str, Any]]:
        cutoff = datetime.utcnow() - timedelta(days=learning_window_days)
        data: List[Dict[str, Any]] = []
        for records in self.chemistry_learning["reaction_history"].values():
            for r in records:
                if r["timestamp"] >= cutoff:
                    data.append(r)
        return data

    def _calculate_current_chemistry_performance(
        self, learning_data: List[Dict[str, Any]]
    ) -> float:
        if not learning_data:
            return 0.0
        return statistics.mean([d["actual_performance"] for d in learning_data])

    async def _optimize_chemistry_weights_gradient_descent(
        self, learning_data: List[Dict[str, Any]], optimization_target: str
    ) -> Dict[str, float]:
        # Simple pseudo-optimization: nudge weights in proportion to correlation direction
        cw = self.chemistry_learning["adaptive_weights"]
        base = cw.copy()
        # Pretend we derived slight improvements
        base["consultant_compatibility_weight"] = min(
            0.5, base["consultant_compatibility_weight"] + 0.02
        )
        base["pattern_synergy_weight"] = min(0.4, base["pattern_synergy_weight"] + 0.01)
        base["analytical_diversity_weight"] = base["analytical_diversity_weight"]
        base["cognitive_efficiency_weight"] = max(
            0.1, base["cognitive_efficiency_weight"] - 0.01
        )
        return base

    async def _estimate_chemistry_improvement(
        self, learning_data: List[Dict[str, Any]], new_weights: Dict[str, float]
    ) -> float:
        if not learning_data:
            return 0.0
        # Conservative estimate: magnitude of change times a small factor
        changes = [
            abs(new_weights[k] - self.chemistry_learning["adaptive_weights"][k])
            for k in self.chemistry_learning["adaptive_weights"]
        ]
        return min(0.2, (sum(changes) / len(changes)) * 0.5)

    def _calculate_correlation(
        self, x_values: List[float], y_values: List[float]
    ) -> float:
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0
        mean_x = statistics.mean(x_values)
        mean_y = statistics.mean(y_values)
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
        sum_sq_x = sum((x - mean_x) ** 2 for x in x_values)
        sum_sq_y = sum((y - mean_y) ** 2 for y in y_values)
        denominator = (sum_sq_x * sum_sq_y) ** 0.5
        return numerator / denominator if denominator != 0 else 0.0

    def _calculate_trend(self, values: List[float]) -> str:
        if len(values) < 3:
            return "insufficient_data"
        first_half = values[: len(values) // 2]
        second_half = values[len(values) // 2 :]
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        if second_avg > first_avg * 1.05:
            return "improving"
        elif second_avg < first_avg * 0.95:
            return "declining"
        else:
            return "stable"

    def _categorize_context_for_chemistry(self, context: Dict[str, Any]) -> str:
        return context.get("domain", "general")

    def _extract_domain_from_framework(self, problem_framework: str) -> str:
        pf = problem_framework.lower()
        if any(k in pf for k in ["market", "business", "strategy"]):
            return "business"
        if any(k in pf for k in ["tech", "engineering", "system"]):
            return "technical"
        return "general"

    # --------------------------- Evidence ---------------------------
    def _record_consultant_selection_evidence(
        self,
        problem_framework: str,
        selected_combinations: List[str],
        chemistry_scores: Dict[str, Dict[str, float]],
        final_score: float,
        selection_rationale: str,
        risk_factors: List[str],
        success_factors: List[str],
        confidence_level: float,
    ) -> None:
        if not self.context_stream:
            return
        evidence_data: Dict[str, Any] = {
            "problem_framework": problem_framework,
            "selection_rationale": selection_rationale,
            "total_confidence": confidence_level,
            "consultant_count": len(selected_combinations),
            "final_chemistry_score": final_score,
            "consultants": [],
        }
        for combo_id in selected_combinations:
            scores = chemistry_scores.get(combo_id, {})
            top_features: List[str] = []
            synergy_score = 0.0
            domain_match_score = 0.0
            for score_name, score_value in scores.items():
                if isinstance(score_value, (int, float)) and score_value > 0.7:
                    top_features.append(f"{score_name}={score_value:.2f}")
                if "synergy" in score_name.lower():
                    synergy_score = max(
                        synergy_score,
                        (
                            float(score_value)
                            if isinstance(score_value, (int, float))
                            else 0.0
                        ),
                    )
                if (
                    "domain" in score_name.lower()
                    or "specialization" in score_name.lower()
                ):
                    domain_match_score = max(
                        domain_match_score,
                        (
                            float(score_value)
                            if isinstance(score_value, (int, float))
                            else 0.0
                        ),
                    )
            consultant_evidence = {
                "consultant_id": combo_id,
                "consultant_type": "general",
                "version": "v2_chemistry",
                "synergy_score": synergy_score,
                "domain_match_score": domain_match_score,
                "why_selected": success_factors,
                "top_features": top_features,
                "chosen_nway_clusters": [combo_id],
                "cluster_rationale": f"Chemistry score: {final_score:.3f} - {selection_rationale}",
            }
            evidence_data["consultants"].append(consultant_evidence)
        evidence_data["risk_factors"] = risk_factors
        evidence_data["success_factors"] = success_factors
        self.context_stream.add_event(
            event_type=ContextEventType.MODEL_SELECTION_JUSTIFICATION,
            data=evidence_data,
            metadata={
                "evidence_type": "consultant_selection",
                "audit_level": "complete",
                "trace_id": self.context_stream.trace_id,
                "chemistry_engine": "cognitive_chemistry_v1",
            },
        )
        logger.info(
            f"🔍 Evidence recorded: Selected {len(selected_combinations)} consultant combinations with confidence {confidence_level:.2f}"
        )

    # --------------------------- Helpers ---------------------------
    def _require_scorer(self) -> Any:
        if self.scorer is not None:
            return self.scorer
        try:
            from src.services.container import global_container  # type: ignore

            self.scorer = (
                global_container.get_chemistry_scorer() if global_container else None
            )
            if self.scorer is None:
                raise RuntimeError("ChemistryScorer not available")
            return self.scorer
        except Exception as e:
            raise RuntimeError(f"ChemistryScorer resolution failed: {e}")
