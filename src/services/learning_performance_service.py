"""
Learning Performance Tracking Service
=====================================

This service implements the evidence-based learning system to replace arbitrary
scores with real performance data. It tracks how mental models and consultants
perform in actual analysis sessions.

Key Features:
- CQA score collection and analysis
- Model effectiveness tracking per consultant/role
- Evidence-based model selection weights
- Performance trend analysis
- Learning calibration over time
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import statistics

from src.cognitive_architecture.mental_models_system import ConsultantRole


@dataclass
class PerformanceSession:
    """A single analysis session with performance data"""

    trace_id: str
    user_query: str
    consultant_id: str
    models_used: List[str]
    nway_patterns_used: List[str]

    # CQA Scores (0-100 each)
    cqa_rigor: int
    cqa_insight: int
    cqa_value: int
    cqa_alignment: int

    # Derived metrics
    total_cqa_score: float
    session_duration_ms: int
    total_tokens: int

    # Metadata
    domain: str
    task_type: str
    complexity_level: int
    timestamp: datetime


@dataclass
class ModelPerformanceData:
    """Performance data for a specific model in a session"""

    model_id: str
    consultant_id: str
    session_trace_id: str

    # Performance metrics
    effectiveness_score: float  # 0.0 - 1.0
    contribution_to_cqa: float  # How much this model contributed to final CQA
    tokens_consumed: int
    response_time_ms: int

    # Context
    usage_context: str
    pipeline_stage: str  # 'analysis', 'synthesis', 'refinement', etc.


class LearningPerformanceService:
    """
    Tracks real performance data to build evidence-based model selection
    """

    def __init__(self, database_service: Optional["DatabaseService"] = None):
        """Initialize the learning performance service using unified DB facade.

        Accepts DatabaseService via DI; if not provided, attempts to create one
        from environment (best-effort) to preserve backward compatibility.
        """
        from src.services.persistence.database_service import (
            DatabaseService,
            DatabaseServiceConfig,
            DatabaseOperationError,
        )

        self._db: Optional[DatabaseService] = None
        if database_service is not None:
            self._db = database_service
        else:
            # Best-effort local construction (non-fatal if unavailable)
            try:
                cfg = DatabaseServiceConfig.from_env()
                self._db = DatabaseService(config=cfg)
            except Exception as e:  # pragma: no cover - environment dependent
                print(f"Warning: DatabaseService not available for learning: {e}")
                self._db = None

    def record_analysis_session(self, session: PerformanceSession) -> bool:
        """Record a complete analysis session with performance data"""
        if not self._db:
            print("Warning: Cannot record session - DatabaseService not available")
            return False

        try:
            # Store session in learning_analysis_sessions table
            session_data = {
                "trace_id": session.trace_id,
                "user_query": session.user_query,
                "domain": session.domain,
                "task_type": session.task_type,
                "complexity_level": session.complexity_level,
                "consultants_selected": [session.consultant_id],
                "models_used": session.models_used,
                "nway_patterns_used": session.nway_patterns_used,
                "final_cqa_score": session.total_cqa_score,
                "total_tokens": session.total_tokens,
                "total_time_ms": session.session_duration_ms,
                "pipeline_stage_times": {"total_duration": session.session_duration_ms},
                "selection_reasoning": {
                    "cqa_breakdown": {
                        "rigor": session.cqa_rigor,
                        "insight": session.cqa_insight,
                        "value": session.cqa_value,
                        "alignment": session.cqa_alignment,
                    }
                },
                "started_at": session.timestamp.isoformat(),
                "completed_at": session.timestamp.isoformat(),
            }

            self._db.store_learning_session(session_data)
            print(f"✅ Recorded analysis session: {session.trace_id}")
            return True

        except Exception as e:
            print(f"Error recording analysis session {session.trace_id}: {e}")
            return False

    def record_model_performance(self, performance: ModelPerformanceData) -> bool:
        """Record performance data for a specific model in a session"""
        if not self._db:
            print("Warning: Cannot record model performance - DatabaseService not available")
            return False

        try:
            # Get session_id from trace_id
            session_id = self._db.get_learning_session_id(performance.session_trace_id)
            if not session_id:
                print(
                    f"Warning: No session found for trace_id {performance.session_trace_id}"
                )
                return False

            # Store model performance
            perf_data = {
                "session_id": session_id,
                "trace_id": performance.session_trace_id,
                "model_id": performance.model_id,
                "consultant_id": performance.consultant_id,
                "effectiveness_score": performance.effectiveness_score,
                "contribution_to_outcome": performance.contribution_to_cqa,
                "tokens_consumed": performance.tokens_consumed,
                "response_time_ms": performance.response_time_ms,
                "usage_context": performance.usage_context,
                "pipeline_stage": performance.pipeline_stage,
            }

            self._db.store_model_performance(perf_data)
            print(
                f"✅ Recorded model performance: {performance.model_id} in {performance.session_trace_id}"
            )
            return True

        except Exception as e:
            print(f"Error recording model performance: {e}")
            return False

    def get_model_effectiveness_score(
        self, model_id: str, consultant_role: ConsultantRole, lookback_days: int = 30
    ) -> float:
        """
        Get evidence-based effectiveness score for a model with a specific consultant role

        This replaces the arbitrary scores with real performance data
        """
        if not self._db:
            return 0.5  # Neutral fallback

        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)

            # Query performance data for this model and role
            records = self._db.fetch_model_performances(
                model_id=model_id, start=start_date, end=end_date
            )

            if not records:
                return 0.5  # No data available yet

            # Calculate average effectiveness
            effectiveness_scores = [r.get("effectiveness_score") for r in records if r.get("effectiveness_score") is not None]
            contribution_scores = [r.get("contribution_to_outcome") for r in records if r.get("contribution_to_outcome") is not None]

            if not effectiveness_scores:
                return 0.5

            # Combine effectiveness and contribution scores
            avg_effectiveness = statistics.mean(effectiveness_scores)
            avg_contribution = (
                statistics.mean(contribution_scores)
                if contribution_scores
                else avg_effectiveness
            )

            # Weight them (60% effectiveness, 40% contribution)
            final_score = (0.6 * avg_effectiveness) + (0.4 * avg_contribution)

            return max(0.0, min(1.0, final_score))

        except Exception as e:
            print(f"Error getting effectiveness score for {model_id}: {e}")
            return 0.5

    def get_top_performing_models(
        self, consultant_role: ConsultantRole, limit: int = 10, lookback_days: int = 30
    ) -> List[Tuple[str, float]]:
        """Get the top performing models for a consultant role based on real data"""
        if not self._db:
            return []

        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)

            # Query all model performances
            records = self._db.fetch_model_performances(start=start_date, end=end_date)

            if not records:
                return []

            # Aggregate by model_id
            model_scores = {}
            for record in records:
                model_id = record["model_id"]
                effectiveness = record.get("effectiveness_score", 0.5)
                contribution = record.get("contribution_to_outcome", effectiveness)

                if model_id not in model_scores:
                    model_scores[model_id] = []

                # Combine effectiveness and contribution
                combined_score = (0.6 * effectiveness) + (0.4 * contribution)
                model_scores[model_id].append(combined_score)

            # Calculate averages and sort
            model_averages = []
            for model_id, scores in model_scores.items():
                avg_score = statistics.mean(scores)
                model_averages.append((model_id, avg_score))

            # Sort by score descending
            model_averages.sort(key=lambda x: x[1], reverse=True)

            return model_averages[:limit]

        except Exception as e:
            print(f"Error getting top performing models: {e}")
            return []

    def get_learning_insights(self, lookback_days: int = 30) -> Dict[str, Any]:
        """Get insights from the learning data for system improvement"""
        if not self._db:
            return {}

        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)

            # Get session data
            sessions = self._db.fetch_learning_sessions(start=start_date, end=end_date)

            if not sessions:
                return {"message": "No learning data available yet"}

            # Calculate insights
            total_sessions = len(sessions)
            avg_cqa_score = statistics.mean(
                [s["final_cqa_score"] for s in sessions if s["final_cqa_score"]]
            )

            # Most used models
            all_models = []
            for session in sessions:
                all_models.extend(session.get("models_used", []))

            model_usage = {}
            for model in all_models:
                model_usage[model] = model_usage.get(model, 0) + 1

            most_used_models = sorted(
                model_usage.items(), key=lambda x: x[1], reverse=True
            )[:5]

            # Performance trends
            recent_sessions = sorted(sessions, key=lambda x: x["created_at"])[-10:]
            recent_cqa_scores = [
                s["final_cqa_score"] for s in recent_sessions if s["final_cqa_score"]
            ]

            insights = {
                "total_sessions": total_sessions,
                "average_cqa_score": round(avg_cqa_score, 2),
                "most_used_models": most_used_models,
                "recent_performance_trend": {
                    "recent_avg_cqa": (
                        round(statistics.mean(recent_cqa_scores), 2)
                        if recent_cqa_scores
                        else 0
                    ),
                    "sample_size": len(recent_cqa_scores),
                },
                "learning_status": "ACTIVE" if total_sessions > 10 else "BUILDING_DATA",
                "recommendation": self._generate_learning_recommendation(
                    total_sessions, avg_cqa_score
                ),
            }

            return insights

        except Exception as e:
            print(f"Error getting learning insights: {e}")
            return {"error": str(e)}

    def _generate_learning_recommendation(
        self, total_sessions: int, avg_cqa: float
    ) -> str:
        """Generate recommendations based on learning data"""
        if total_sessions < 10:
            return "Continue using the system to build learning data. Need 10+ sessions for reliable insights."
        elif avg_cqa < 70:
            return "Consider reviewing model selection strategy. Current performance below target."
        elif avg_cqa > 85:
            return "Excellent performance! The learning system is working well."
        else:
            return "Good performance. System is learning and improving model selection."

    def update_model_learning_weights(self) -> Dict[str, float]:
        """
        Update the model selection weights based on learning data

        This replaces static weights with evidence-based weights
        """
        insights = self.get_learning_insights(lookback_days=60)

        if not insights or "error" in insights:
            # Return default weights if no learning data
            return {
                "C": 0.2,  # Concept coverage
                "F": 0.15,  # Role fit
                "N": 0.25,  # NWAY bonus
                "D": 0.10,  # Diversity
                "T": 0.05,  # Tension penalty
                "P": 0.10,  # Practicality
            }

        # Adjust weights based on performance
        avg_cqa = insights.get("average_cqa_score", 75)

        if avg_cqa > 85:
            # High performance - maintain current strategy
            weights = {"C": 0.2, "F": 0.15, "N": 0.25, "D": 0.10, "T": 0.05, "P": 0.10}
        elif avg_cqa < 70:
            # Low performance - emphasize proven models and diversity
            weights = {
                "C": 0.15,
                "F": 0.25,  # Increase role fit importance
                "N": 0.20,  # Reduce NWAY bonus
                "D": 0.15,  # Increase diversity
                "T": 0.10,  # Increase tension awareness
                "P": 0.15,  # Increase practicality
            }
        else:
            # Moderate performance - balanced approach
            weights = {"C": 0.18, "F": 0.20, "N": 0.22, "D": 0.12, "T": 0.08, "P": 0.12}

        return weights


def create_performance_session_from_analysis(
    trace_id: str,
    user_query: str,
    consultant_id: str,
    models_used: List[str],
    nway_patterns: List[str],
    cqa_scores: Dict[
        str, int
    ],  # {'rigor': 85, 'insight': 78, 'value': 82, 'alignment': 88}
    duration_ms: int,
    total_tokens: int,
    domain: str = "general",
    task_type: str = "analysis",
    complexity: int = 3,
) -> PerformanceSession:
    """Helper function to create a PerformanceSession from analysis results"""

    total_cqa = (
        cqa_scores.get("rigor", 0)
        + cqa_scores.get("insight", 0)
        + cqa_scores.get("value", 0)
        + cqa_scores.get("alignment", 0)
    ) / 4.0

    return PerformanceSession(
        trace_id=trace_id,
        user_query=user_query,
        consultant_id=consultant_id,
        models_used=models_used,
        nway_patterns_used=nway_patterns,
        cqa_rigor=cqa_scores.get("rigor", 0),
        cqa_insight=cqa_scores.get("insight", 0),
        cqa_value=cqa_scores.get("value", 0),
        cqa_alignment=cqa_scores.get("alignment", 0),
        total_cqa_score=total_cqa,
        session_duration_ms=duration_ms,
        total_tokens=total_tokens,
        domain=domain,
        task_type=task_type,
        complexity_level=complexity,
        timestamp=datetime.now(),
    )
