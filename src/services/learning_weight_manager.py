from __future__ import annotations

from typing import Dict, Optional
from datetime import datetime
from src.cognitive_architecture.mental_models_system import ConsultantRole


class LearningWeightManager:
    """
    Minimal role-specific weight provider with future online recalibration hooks.

    Integrates later with CQA signals (RIVA) and session outcomes to adjust
    weights per role. For now provides stable defaults with guardrails.

    Persistence: uses Supabase table 'model_selection_weights' if available
    with columns: role TEXT PRIMARY KEY, weights JSONB, updated_at TIMESTAMP.
    """

    DEFAULTS: Dict[ConsultantRole, Dict[str, float]] = {
        ConsultantRole.STRATEGIC_ANALYST: {
            "C": 0.20,
            "F": 0.15,
            "N": 0.25,
            "E": 0.10,
            "D": 0.15,
            "T": 0.05,
            "P": 0.10,
        },
        ConsultantRole.SYNTHESIS_ARCHITECT: {
            "C": 0.20,
            "F": 0.15,
            "N": 0.20,
            "E": 0.15,
            "D": 0.15,
            "T": 0.05,
            "P": 0.10,
        },
        ConsultantRole.IMPLEMENTATION_DRIVER: {
            "C": 0.20,
            "F": 0.15,
            "N": 0.15,
            "E": 0.10,
            "D": 0.15,
            "T": 0.05,
            "P": 0.20,
        },
    }

    def __init__(self, learning_service: Optional["LearningPerformanceService"] = None):
        self._weights = {k: v.copy() for k, v in self.DEFAULTS.items()}
        self.learning_service = learning_service
        self._last_learning_update = None

        # Try to load persisted weights (best-effort)
        try:
            from src.core.supabase_platform import MetisSupabasePlatform

            platform = MetisSupabasePlatform()
            # Synchronous context; use direct client if available
            client = platform.supabase
            if client:
                data = (
                    client.table("model_selection_weights").select("*").execute().data
                )
                for row in data or []:
                    role_name = row.get("role")
                    weights = row.get("weights") or {}
                    # Map to ConsultantRole if possible
                    from src.cognitive_architecture.mental_models_system import (
                        ConsultantRole,
                    )

                    for r in ConsultantRole:
                        if r.name.lower() == str(role_name).lower():
                            self._weights[r] = {k: float(v) for k, v in weights.items()}
        except Exception:
            pass

    def get_weights(self, role: ConsultantRole) -> Dict[str, float]:
        """Get weights for a consultant role, enhanced with learning data if available"""
        base_weights = self._weights.get(
            role,
            {
                "C": 0.2,
                "F": 0.15,
                "N": 0.25,
                "E": 0.10,
                "D": 0.15,
                "T": 0.05,
                "P": 0.10,
            },
        )

        # Enhance with learning service if available and data is fresh
        if self.learning_service and self._should_update_from_learning():
            try:
                learning_weights = self.learning_service.update_model_learning_weights()

                # Merge learning insights with current weights
                enhanced_weights = base_weights.copy()

                # Apply learning adjustments (conservative approach)
                for key in learning_weights:
                    if key in enhanced_weights:
                        # Weight the learning data with current weights (80% current, 20% learning)
                        enhanced_weights[key] = (0.8 * enhanced_weights[key]) + (
                            0.2 * learning_weights[key]
                        )

                # Store the enhanced weights for caching
                self._weights[role] = enhanced_weights
                self._last_learning_update = datetime.now()

                return enhanced_weights

            except Exception as e:
                print(f"Warning: Could not apply learning weights for {role}: {e}")

        return base_weights

    def _should_update_from_learning(self) -> bool:
        """Check if we should update weights from learning data (every 30 minutes)"""
        if not self._last_learning_update:
            return True

        time_diff = datetime.now() - self._last_learning_update
        return time_diff.total_seconds() > (30 * 60)  # 30 minutes

    def set_weights(self, role: ConsultantRole, weights: Dict[str, float]) -> None:
        # Basic guardrails
        s = sum(weights.values())
        if s == 0:
            return
        norm = {k: max(0.0, v) / s for k, v in weights.items()}
        # Cap dominance
        if max(norm.values()) > 0.5:
            # scale down the max component
            mkey = max(norm, key=norm.get)
            excess = norm[mkey] - 0.5
            norm[mkey] = 0.5
            # redistribute excess
            others = [k for k in norm.keys() if k != mkey]
            for k in others:
                norm[k] += excess / len(others)
        self._weights[role] = norm
        # Persist (best-effort)
        try:
            from src.core.supabase_platform import MetisSupabasePlatform

            platform = MetisSupabasePlatform()
            client = platform.supabase
            if client:
                client.table("model_selection_weights").upsert(
                    {
                        "role": role.name,
                        "weights": norm,
                    }
                ).execute()
        except Exception:
            pass

    # --- NEW: Online nudge from CQA results ---
    def update_from_cqa(
        self, trace_id: str, role: ConsultantRole, riva_scores: Dict[str, float]
    ) -> None:
        """
        Nudge role weights based on CQA RIVA scores for the given trace.
        Expected riva_scores keys: 'rigor', 'insight', 'value', 'alignment'.
        Strategy: Small adjustments mapped to components: C,F,N,D,P.
        """
        current = self.get_weights(role).copy()
        # Map RIVA dimensions to selection components
        rigor = riva_scores.get("rigor", 0.0) / 10.0
        insight = riva_scores.get("insight", 0.0) / 10.0
        value = riva_scores.get("value", 0.0) / 10.0
        alignment = riva_scores.get("alignment", 0.0) / 10.0
        # Learning rate (very small)
        eta = 0.02
        # Nudge:
        current["C"] = max(0.0, current.get("C", 0.2) + eta * (rigor - 0.5))
        current["F"] = max(0.0, current.get("F", 0.15) + eta * (alignment - 0.5))
        current["N"] = max(0.0, current.get("N", 0.2) + eta * (insight - 0.5))
        current["D"] = max(0.0, current.get("D", 0.1) + eta * (insight - 0.5))
        current["P"] = max(0.0, current.get("P", 0.1) + eta * (value - 0.5))
        # Keep T,E stable for now
        self.set_weights(role, current)
