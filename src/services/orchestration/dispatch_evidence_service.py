# src/services/orchestration/dispatch_evidence_service.py
import logging
import time
from typing import Any, Dict, List

from src.core.unified_context_stream import ContextEventType, UnifiedContextStream

logger = logging.getLogger(__name__)


class DispatchEvidenceService:
    """Encapsulates evidence emission, logging, and S2 tier state updates for dispatch orchestration."""

    def __init__(self, context_stream: UnifiedContextStream) -> None:
        self.context_stream = context_stream

    def log_pattern_selection(
        self,
        *,
        framework_type: str,
        pattern_selection: Any,
        selected_consultant_ids: List[str],
        s2_tier: str,
        s2_rationale: str,
        team_synergy_data: Dict[str, Any],
        team_strategy: str,
        domain: str,
        task_classification: Dict[str, Any],
        start_time: float,
    ) -> None:
        """Emit MODEL_SELECTION_JUSTIFICATION evidence and related logs for NWAY pattern selection."""
        try:
            logger.info(
                f"🎯 NWAY Pattern Selection: {pattern_selection.primary_pattern} (confidence: {pattern_selection.confidence_score:.2f})"
            )
            logger.info(
                f"   Selected patterns: {', '.join(pattern_selection.selected_patterns)}"
            )
            logger.info(f"   Rationale: {pattern_selection.selection_rationale}")

            selection_rationale = (
                f"S2_{s2_tier}: Smart GM selected optimal {len(selected_consultant_ids)}-consultant team "
                f"with {team_synergy_data.get('synergy_bonus', 0.0):.3f} synergy bonus for {task_classification['task_type']} "
                f"in {domain} domain. Dynamic NWAY patterns: {', '.join(pattern_selection.selected_patterns)}."
            )

            # Station 3 evidence event
            self.context_stream.add_event(
                ContextEventType.MODEL_SELECTION_JUSTIFICATION,
                {
                    "selected_consultants": selected_consultant_ids,
                    "chosen_nway_patterns": pattern_selection.selected_patterns,
                    "primary_nway_pattern": pattern_selection.primary_pattern,
                    "nway_selection_confidence": pattern_selection.confidence_score,
                    "nway_selection_rationale": pattern_selection.selection_rationale,
                    "s2_tier": s2_tier,
                    "s2_rationale": s2_rationale,
                    "selection_rationale": selection_rationale,
                    "confidence_score": team_synergy_data.get("synergy_bonus", 0.0)
                    + 0.8,
                    "team_strategy": team_strategy,
                    "domain": domain,
                    "task_type": task_classification["task_type"],
                    "processing_time_seconds": time.time() - start_time,
                    "consultant_count": len(selected_consultant_ids),
                    "framework_type": framework_type,
                },
            )
            logger.info(
                f"📊 STATION 3 EVIDENCE: MODEL_SELECTION_JUSTIFICATION recorded with S2_{s2_tier}"
            )
        except Exception as e:
            logger.warning(f"⚠️ Failed to log pattern selection evidence: {e}")

    def update_s2_after_team_selection(
        self,
        *,
        s2_kernel: Any,
        current_s2: Any,
        team_synergy_data: Dict[str, Any],
        consultant_count: int,
    ) -> Any:
        """Update S2 tier after team selection and emit any relevant logs."""
        try:
            if s2_kernel is None:
                return current_s2
            before = getattr(current_s2, "s2_tier", None)
            updated = s2_kernel.update_after_team_selection(
                current_s2, team_synergy_data, consultant_count
            )
            after = getattr(updated, "s2_tier", None)
            if before != after:
                logger.info(f"🎚️ S2 tier updated: {before} → {after}")
            return updated
        except Exception as e:
            logger.warning(f"⚠️ S2 update after team selection failed: {e}")
            return current_s2

    def finalize_s2_evaluation(
        self,
        *,
        s2_kernel: Any,
        current_s2: Any,
        processing_time: float,
    ) -> Any:
        """Finalize S2 evaluation at end of dispatch and log tier change history."""
        try:
            if s2_kernel is None:
                return current_s2
            updated = s2_kernel.finalize_evaluation(current_s2, processing_time)
            if getattr(updated, "tier_history", None):
                logger.info(
                    f"📊 S2 Tier Changes: {len(updated.tier_history)} adjustments during dispatch"
                )
                for change in updated.tier_history:
                    logger.debug(
                        f"   {change.get('from')} → {change.get('to')}: {change.get('reason')}"
                    )
            return updated
        except Exception as e:
            logger.warning(f"⚠️ Final S2 evaluation failed: {e}")
            return current_s2

    def log_smart_gm_team_selection(
        self,
        *,
        task_classification: Dict[str, Any],
        team_strategy: str,
        selected_consultants: List[Any],
        team_synergy_data: Dict[str, Any],
        baseline_consultant_pool: List[Dict[str, Any]],
        final_diversity_score: float,
    ) -> None:
        """Replicates Smart GM transparency logging as a dedicated evidence method."""
        try:
            strategy_description = team_synergy_data.get(
                "team_composition_logic",
                f"Smart GM selected optimal 3-consultant team for {task_classification['task_type']} task",
            )

            enhanced_metadata = {
                "message": f"SMART GM v2.0: Selected optimal team of {len(selected_consultants)} consultants with synergy analysis",
                "domain": task_classification["primary_domain"],
                "selected_team": [
                    {
                        "consultant_id": c.consultant_id,
                        "consultant_type": c.consultant_type,
                        "specialization": c.specialization,
                        "predicted_effectiveness": c.predicted_effectiveness,
                        "assigned_dimensions": c.assigned_dimensions,
                        # OPERATION UNIFICATION: Add YAML-driven selection rationale
                        "selection_rationale": self._generate_consultant_selection_rationale(c),
                    }
                    for c in selected_consultants
                ],
                "engine": "SmartGMOrchestrationEngine",
                "engine_metadata": {
                    "smart_gm_orchestration": True,
                    "team_composition_version": "2.0",
                    "baseline_candidates_evaluated": len(baseline_consultant_pool),
                    "team_combinations_analyzed": "calculated_combinations",
                },
                "task_classification": task_classification,
                "team_composition_strategy": strategy_description,
                "cognitive_diversity_score": final_diversity_score,
                "smart_gm_intelligence": {
                    "classification_confidence": task_classification["confidence"],
                    "strategy_applied": team_strategy,
                    "avg_individual_score": team_synergy_data.get(
                        "avg_individual_score", 0.0
                    ),
                    "synergy_bonus": team_synergy_data.get("synergy_bonus", 0.0),
                    "final_team_score": team_synergy_data.get("final_score", 0.0),
                    "team_size": len(selected_consultants),
                    "optimization_approach": "multi_team_evaluation_with_synergy_bonuses",
                },
            }

            # Emit events
            self.context_stream.add_event(
                ContextEventType.CONTEXTUAL_CONSULTANT_SELECTION_V1,
                enhanced_metadata,
            )
            
            # OPERATION UNIFICATION: Emit CONSULTANT_SELECTION_COMPLETE with YAML-driven rationales
            self.context_stream.add_event(
                ContextEventType.CONSULTANT_SELECTION_COMPLETE,
                {
                    "message": "OPERATION UNIFICATION: Consultant selection complete with YAML-driven explanations",
                    "selected_consultants": [
                        {
                            "consultant_id": c.consultant_id,
                            "selection_rationale": self._generate_consultant_selection_rationale(c),
                            "predicted_effectiveness": c.predicted_effectiveness,
                            "specialization": c.specialization,
                        }
                        for c in selected_consultants
                    ],
                    "team_size": len(selected_consultants),
                    "yaml_enhanced": True,
                    "selection_methodology": "YAML-enhanced cognitive profile scoring with 5-factor weighting",
                },
            )
            self.context_stream.add_event(
                ContextEventType.ADAPTIVE_TEAM_COMPOSITION,
                {
                    "smart_gm_strategy": team_strategy,
                    "task_classification": task_classification,
                    "team_diversity_score": final_diversity_score,
                    "synergy_analysis": team_synergy_data,
                    "selected_consultant_ids": [
                        c.consultant_id for c in selected_consultants
                    ],
                    "baseline_pool_size": len(baseline_consultant_pool),
                    "optimization_details": {
                        "primary_domain": task_classification["primary_domain"],
                        "task_type": task_classification["task_type"],
                        "requires_creativity": task_classification.get(
                            "requires_creativity", False
                        ),
                        "complexity_level": task_classification.get(
                            "complexity_level", "medium"
                        ),
                        "smart_gm_version": "2.0",
                    },
                },
            )

            logger.info(f"🧑‍💼 SMART GM TRANSPARENCY: {strategy_description}")
            logger.info(
                f"🏆 Team synergy bonus: +{team_synergy_data.get('synergy_bonus', 0.0):.3f}"
            )
            logger.info(f"✨ Final team diversity score: {final_diversity_score:.3f}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to log Smart GM team selection: {e}")

    def log_final_package(
        self,
        *,
        dispatch_package: Any,
        consultant_count: int,
        nway_pattern_name: str,
    ) -> None:
        """Log final dispatch summary information."""
        try:
            logger.info(
                f"👥 Selected {consultant_count} consultants in {nway_pattern_name} pattern"
            )
        except Exception:
            pass

    def _generate_consultant_selection_rationale(self, consultant_blueprint: Any) -> str:
        """
        OPERATION UNIFICATION: Generate human-readable selection rationale from YAML cognitive profiles.
        
        Creates explanations like: "Selected 'strategic_analyst' due to strong affinity with 
        'Systems Thinking' mental model (30% weight) and 'analytical, systematic' cognitive signature (15% weight)."
        
        Args:
            consultant_blueprint: ConsultantBlueprint with consultant details
            
        Returns:
            Human-readable rationale string for Glass Box transparency
        """
        try:
            consultant_id = consultant_blueprint.consultant_id
            
            # Try to get rich YAML data from scoring factors (if available)
            consultant_data = getattr(consultant_blueprint, 'consultant_data', {})
            scoring_factors = consultant_data.get('scoring_factors', {})
            
            # Build rationale from scoring factors
            rationale_parts = []
            
            # Mental model affinity (highest weight factor)
            mental_model_score = scoring_factors.get('mental_model_affinity', 0)
            if mental_model_score > 0.5:
                mental_models = consultant_data.get('mental_model_affinities', {})
                if mental_models:
                    top_model = max(mental_models.keys(), key=len, default="systems thinking")
                    rationale_parts.append(f"strong '{top_model}' mental model affinity ({mental_model_score:.1%} contribution)")
            
            # Identity domain expertise
            identity_score = scoring_factors.get('identity_domain_match', 0)
            if identity_score > 0.3:
                identity = consultant_data.get('identity', '')[:50]
                if identity:
                    rationale_parts.append(f"domain expertise match in identity ({identity_score:.1%})")
            
            # Cognitive signature
            cognitive_score = scoring_factors.get('cognitive_signature', 0)
            if cognitive_score > 0.1:
                signature = consultant_data.get('cognitive_signature', '')
                if signature:
                    # Extract key descriptive terms
                    signature_terms = []
                    analytical_terms = ['analytical', 'systematic', 'logical', 'methodical', 'scientific']
                    for term in analytical_terms:
                        if term in signature.lower():
                            signature_terms.append(term)
                    
                    if signature_terms:
                        terms_str = "', '".join(signature_terms[:2])
                        rationale_parts.append(f"'{terms_str}' cognitive approach ({cognitive_score:.1%})")
            
            # NWAY pattern bonus
            nway_bonus = scoring_factors.get('nway_pattern_bonus', 0)
            if nway_bonus > 0.7:
                source_nway = consultant_data.get('source_nway', '')
                if 'DECOMPOSITION' in source_nway:
                    rationale_parts.append("analytical decomposition pattern strength")
                elif 'DECISION' in source_nway:
                    rationale_parts.append("decision-making pattern expertise")
                elif 'PERCEPTION' in source_nway:
                    rationale_parts.append("pattern recognition capabilities")
            
            # Build final rationale
            if rationale_parts:
                rationale = f"Selected '{consultant_id}' due to " + ", ".join(rationale_parts)
                
                # Add effectiveness score if available
                effectiveness = getattr(consultant_blueprint, 'predicted_effectiveness', 0)
                if effectiveness > 0:
                    rationale += f" (effectiveness: {effectiveness:.1%})"
                    
                return rationale
            else:
                # Fallback rationale
                specialization = getattr(consultant_blueprint, 'specialization', consultant_id)
                effectiveness = getattr(consultant_blueprint, 'predicted_effectiveness', 0)
                return f"Selected '{consultant_id}' for {specialization} expertise (effectiveness: {effectiveness:.1%})"
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to generate selection rationale for {consultant_blueprint}: {e}")
            # Safe fallback
            consultant_id = getattr(consultant_blueprint, 'consultant_id', 'consultant')
            return f"Selected '{consultant_id}' based on algorithmic optimization"
