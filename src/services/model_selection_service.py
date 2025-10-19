from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import re

from src.cognitive_architecture.mental_models_system import (
    MentalModelsLibrary,
    ConsultantRole,
    MentalModel,
)
from src.cognitive_architecture.enhanced_nway_interactions_system import (
    Enhanced21ClusterNWayLibrary,
)
from src.core.unified_context_stream import UnifiedContextStream, ContextEventType

# Learning performance + persistence facade via DI container
from src.services.container import global_container


@dataclass
class SelectedModel:
    name: str
    source: str  # 'nway' | 'general_library'
    score: float


def _normalize_text(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if t]


class ModelSelectionService:
    """
    Select 3–7 mental models per consultant by combining:
    - participating_models from selected NWAY clusters
    - models from the general library filtered by concept similarity

    Reuses existing mental models and enhanced NWAY libraries. Emits
    MODEL_SELECTION_JUSTIFICATION events for glass-box transparency.
    """

    def __init__(self, weight_manager: Optional["LearningWeightManager"] = None, database_service: Optional["DatabaseService"] = None, learning_service: Optional["LearningPerformanceService"] = None):
        self.models_lib = MentalModelsLibrary()  # Fallback for hardcoded models
        self.nway_lib = Enhanced21ClusterNWayLibrary()
        self.weight_manager = weight_manager

        # Initialize services via DI container (allow explicit overrides for tests)
        self.learning_service = learning_service or global_container.get_learning_performance_service()
        self.database_service = database_service or global_container.get_database_service()

    def _get_model_from_database(self, model_name: str) -> Optional[Dict]:
        """Get model data via unified DatabaseService (facade)."""
        if not self.database_service:
            return None
        try:
            return self.database_service.fetch_mental_model_by_id(model_name)
        except Exception as e:
            print(f"Warning: Could not fetch model {model_name} from database: {e}")
            return None

    def _get_evidence_based_role_fit(
        self, model_name: str, role: ConsultantRole
    ) -> float:
        """Get evidence-based role fit using learning performance data instead of arbitrary scores"""
        try:
            # Use learning service to get real effectiveness score
            effectiveness_score = self.learning_service.get_model_effectiveness_score(
                model_id=model_name, consultant_role=role, lookback_days=30
            )

            # Effectiveness score is already 0.0-1.0, perfect for role fit
            return effectiveness_score

        except Exception as e:
            print(f"Warning: Could not get evidence-based score for {model_name}: {e}")
            # Fallback to neutral score when learning data unavailable
            return 0.5

    def _consultant_id_to_role(self, consultant_id: str) -> ConsultantRole:
        cid = consultant_id.lower()
        if any(
            k in cid
            for k in [
                "strategic_analyst",
                "mckinsey_strategist",
                "competitive_analyst",
                "strategist",
            ]
        ):
            return ConsultantRole.STRATEGIC_ANALYST
        if any(
            k in cid
            for k in [
                "operations_expert",
                "implementation_specialist",
                "technology_advisor",
                "implementation",
                "ops",
                "architect",
            ]
        ):
            return ConsultantRole.IMPLEMENTATION_DRIVER
        # Default bucket for others (market_researcher, financial_analyst, risk_assessor, innovation_consultant, crisis_manager, turnaround_specialist)
        return ConsultantRole.SYNTHESIS_ARCHITECT

    def _get_nway_models(self, cluster_ids: List[str]) -> List[str]:
        names: List[str] = []
        for cid in cluster_ids:
            inter = self.nway_lib.interactions.get(cid)
            if inter:
                for m in inter.participating_models:
                    if m not in names:
                        names.append(m)
        return names

    def _general_candidates(
        self, concepts: List[str], similarity_threshold: float = 0.01
    ) -> List[Dict]:
        """Get candidate models from database via facade, with library fallback."""
        if self.database_service:
            db_candidates = self._general_candidates_from_database(
                concepts, similarity_threshold
            )
            if db_candidates:
                return db_candidates
            print("   Database returned 0 models, falling back to library")
        # No database or empty response, use hardcoded models
        return self._general_candidates_from_library(concepts, similarity_threshold)

    def _general_candidates_from_database(
        self, concepts: List[str], similarity_threshold: float = 0.01
    ) -> List[Dict]:
        """Get candidate models from the database via DatabaseService."""
        concept_tokens = set()
        for c in concepts:
            concept_tokens.update(_normalize_text(c))

        try:
            if not self.database_service:
                return []
            active_models = self.database_service.fetch_active_mental_models()
            candidates: List[Dict] = []
            for model_data in active_models:
                name_tokens = set(_normalize_text(model_data.get("name", "")))
                cat_tokens = set(_normalize_text(model_data.get("category", "")))
                overlap = len((name_tokens | cat_tokens) & concept_tokens)
                base = len(name_tokens | cat_tokens | {"model"}) or 1
                sim = overlap / base
                if sim >= similarity_threshold:
                    candidates.append(model_data)
            return candidates
        except Exception as e:
            print(f"Warning: Could not fetch models from database: {e}")
            return []

    def _general_candidates_from_library(
        self, concepts: List[str], similarity_threshold: float = 0.01
    ) -> List[Dict]:
        """Fallback: Get candidate models from hardcoded library"""
        print(f"   🔍 Searching library with concepts: {concepts}")
        concept_tokens = set()
        for c in concepts:
            concept_tokens.update(_normalize_text(c))
        print(
            f"   🔍 Concept tokens: {list(concept_tokens)[:10]}..."
        )  # First 10 tokens
        print(f"   🔍 Models library has {len(self.models_lib.models)} total models")
        if self.models_lib.models:
            sample_names = list(self.models_lib.models.keys())[:3]
            print(f"   🔍 Sample model names: {sample_names}")
        out = []
        for i, m in enumerate(self.models_lib.models.values()):
            name_tokens = set(_normalize_text(m.name))
            cat_tokens = set(_normalize_text(m.category.name))
            overlap = len((name_tokens | cat_tokens) & concept_tokens)
            base = len(name_tokens | cat_tokens | {"model"}) or 1
            sim = overlap / base

            # Debug first few models
            if i < 3:
                print(
                    f"   🔍 Model '{m.name}': tokens={list(name_tokens)[:3]}, category={list(cat_tokens)[:3]}, overlap={overlap}, sim={sim:.3f}"
                )

            if sim >= similarity_threshold:
                # Convert MentalModel to dict format for consistency
                out.append(
                    {
                        "model_id": m.name,
                        "name": m.name,
                        "category": m.category.name,
                        "description": getattr(m, "description", ""),
                    }
                )
        print(f"   🔍 Library found {len(out)} candidates")
        return out

    def _role_fit(self, m: MentalModel, role: ConsultantRole) -> float:
        return float(m.consultant_bias.get(role, 0.5))

    def _concept_coverage(self, model_name: str, concepts: List[str]) -> float:
        mtoks = set(_normalize_text(model_name))
        if not concepts:
            return 0.5
        scores = []
        for c in concepts:
            ctoks = set(_normalize_text(c))
            denom = max(1, len(mtoks | ctoks))
            scores.append(len(mtoks & ctoks) / denom)
        return max(scores) if scores else 0.0

    def _get_learning_enhanced_candidates(
        self, role: ConsultantRole, base_candidates: List[Dict]
    ) -> List[Dict]:
        """Enhance candidate selection with top-performing models from learning data"""
        try:
            # Get top performing models for this role from learning data
            top_performers = self.learning_service.get_top_performing_models(
                consultant_role=role, limit=10, lookback_days=60
            )

            # Create a set of existing candidate names for comparison
            existing_names = {c.get("name", "") for c in base_candidates}

            # Add top performers that aren't already in candidates
            enhanced_candidates = base_candidates.copy()
            for model_name, performance_score in top_performers:
                if model_name not in existing_names:
                    # Try to get model data from database
                    model_data = self._get_model_from_database(model_name)
                    if model_data:
                        enhanced_candidates.append(model_data)
                        print(
                            f"🎯 Added top performer {model_name} (score: {performance_score:.2f}) to candidates"
                        )

            return enhanced_candidates

        except Exception as e:
            print(f"Warning: Could not enhance candidates with learning data: {e}")
            return base_candidates

    def _compute_score(
        self,
        model_name: str,
        role: ConsultantRole,
        in_nway: bool,
        concepts: List[str],
        already_selected: List[str],
        weights: Dict[str, float],
    ) -> float:
        # Use evidence-based role fit from learning performance data
        F = self._get_evidence_based_role_fit(model_name, role)
        C = self._concept_coverage(model_name, concepts)
        N = 1.0 if in_nway else 0.0
        # Simple diversity: penalize high overlap with already-selected
        D = 1.0
        if already_selected:
            mtoks = set(_normalize_text(model_name))
            max_sim = 0.0
            for s in already_selected:
                stoks = set(_normalize_text(s))
                denom = max(1, len(mtoks | stoks))
                sim = len(mtoks & stoks) / denom
                max_sim = max(max_sim, sim)
            D = 1.0 - max_sim  # higher is better
        T = 0.0  # tension penalty placeholder (requires curated conflict map)
        P = 0.5  # practicality neutral default
        score = (
            weights.get("C", 0.2) * C
            + weights.get("F", 0.15) * F
            + weights.get("N", 0.25) * N
            + weights.get("D", 0.10) * D
            - weights.get("T", 0.05) * T
            + weights.get("P", 0.10) * P
        )
        return max(0.0, min(1.0, score))

    def select_models_for_consultant(
        self,
        consultant_id: str,
        selected_nway_clusters: List[str],
        concepts: List[str],
        context_stream: Optional[UnifiedContextStream] = None,
        k_min: int = 3,
        k_max: int = 7,
    ) -> Tuple[List[SelectedModel], Dict[str, List[str]]]:
        role = self._consultant_id_to_role(consultant_id)
        weights = (
            self.weight_manager.get_weights(role)
            if self.weight_manager
            else {"C": 0.2, "F": 0.15, "N": 0.25, "D": 0.10, "T": 0.05, "P": 0.10}
        )

        nway_models = self._get_nway_models(selected_nway_clusters)
        general_models = self._general_candidates(concepts)

        # Enhance candidates with top performers from learning data
        enhanced_models = self._get_learning_enhanced_candidates(role, general_models)

        pool_names: List[str] = []
        sources: Dict[str, str] = {}
        for n in nway_models:
            if n not in pool_names:
                pool_names.append(n)
                sources[n] = "nway"
        for gm in enhanced_models:
            model_name = gm.get("name") if isinstance(gm, dict) else gm.name
            if model_name not in pool_names:
                pool_names.append(model_name)
                # Check if this model was added by learning enhancement
                if gm not in general_models:
                    sources[model_name] = "learning_enhanced"
                else:
                    sources[model_name] = "general_library"

        selected: List[SelectedModel] = []
        MIN_GAIN = 0.001  # Very low for testing - force model selection

        # Debug logging
        print("🔍 Model Selection Debug:")
        print(f"   Pool candidates: {len(pool_names)} models")
        print(f"   NWAY models: {len(nway_models)}")
        print(f"   General models: {len(general_models)}")
        print(f"   Enhanced models: {len(enhanced_models)}")
        if pool_names:
            print(f"   First few candidates: {pool_names[:3]}")
        while len(selected) < k_max and pool_names:
            best_name = None
            best_score = -1.0
            for name in pool_names:
                score = self._compute_score(
                    model_name=name,
                    role=role,
                    in_nway=(name in nway_models),
                    concepts=concepts,
                    already_selected=[s.name for s in selected],
                    weights=weights,
                )
                if score > best_score:
                    best_score, best_name = score, name

            # Debug best candidate in this round
            if best_name:
                print(f"   Best candidate: {best_name} (score: {best_score:.3f})")

            if best_name is None or best_score < MIN_GAIN:
                print(f"   Stopping: best_score {best_score:.3f} < MIN_GAIN {MIN_GAIN}")
                break
            selected.append(
                SelectedModel(
                    name=best_name,
                    source=sources.get(best_name, "general_library"),
                    score=best_score,
                )
            )
            pool_names.remove(best_name)
            if len(selected) >= k_min and best_score < MIN_GAIN * 1.5:
                break

        # Debug final selection
        print(f"   Final selection: {len(selected)} models selected")
        for sel in selected:
            print(f"     - {sel.name} (score: {sel.score:.3f}, source: {sel.source})")

        # Build latticework (spine = top 1–2 by score; supports = remaining)
        spine = [
            s.name
            for s in sorted(selected, key=lambda x: x.score, reverse=True)[
                : 2 if len(selected) >= 5 else 1
            ]
        ]
        supports = [s.name for s in selected if s.name not in spine]
        lattice = {"spine_models": spine, "support_models": supports}

        # Log event with evidence-based scoring information
        if context_stream:
            context_stream.add_event(
                ContextEventType.MODEL_SELECTION_JUSTIFICATION,
                {
                    "consultant_id": consultant_id,
                    "consultant_role": role.name,
                    "selected_models": [
                        {"name": s.name, "source": s.source, "score": round(s.score, 3)}
                        for s in selected
                    ],
                    "latticework": lattice,
                    "selected_nway_clusters": selected_nway_clusters,
                    "concepts": concepts,
                    "scoring_method": "evidence_based_learning",
                    "learning_enhanced_count": len(
                        [
                            s
                            for s in selected
                            if sources.get(s.name) == "learning_enhanced"
                        ]
                    ),
                    "total_candidates_considered": len(pool_names) + len(selected),
                },
            )

        return selected, lattice

    @staticmethod
    def render_model_map_as_markdown(
        selected: List[SelectedModel], lattice: Dict[str, List[str]]
    ) -> str:
        lines = []
        lines.append("\n\n## Mental Model Map")
        lines.append("| Model | Source | Score | Role |")
        lines.append("|---|---|---:|---|")
        for s in selected:
            lines.append(f"| {s.name} | {s.source} | {s.score:.2f} | — |")
        if lattice:
            lines.append("\n**Spine:** " + ", ".join(lattice.get("spine_models", [])))
            lines.append(
                "\n**Supports:** " + ", ".join(lattice.get("support_models", []))
            )
        return "\n".join(lines)
