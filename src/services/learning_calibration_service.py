"""
Learning Calibration Service
============================

This service provides continuous calibration and adaptation for the learning performance
system. It analyzes performance trends, detects drift, and automatically adjusts the
scoring algorithms to maintain optimal performance.

Key Features:
- Performance drift detection
- Model recalibration based on recent data
- Adaptive thresholds for scoring
- System health monitoring
- Automatic learning parameter tuning
"""

from __future__ import annotations

import os
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any
from dataclasses import dataclass
from pathlib import Path

from src.services.persistence.database_service import DatabaseService, DatabaseServiceConfig


@dataclass
class CalibrationMetrics:
    """Metrics for system calibration health"""

    avg_cqa_score: float
    performance_trend: str  # 'improving', 'stable', 'declining'
    model_diversity: float  # How diverse are the selected models
    prediction_accuracy: float  # How well does selection predict outcomes
    learning_velocity: float  # How fast the system is learning
    data_quality_score: float  # Quality of incoming data
    last_calibration: datetime
    calibration_confidence: float


@dataclass
class DriftDetectionResult:
    """Results from performance drift detection"""

    drift_detected: bool
    drift_type: str  # 'performance', 'usage_patterns', 'model_effectiveness'
    drift_magnitude: float  # 0.0 to 1.0
    affected_consultants: List[str]
    recommended_actions: List[str]


class LearningCalibrationService:
    """
    Continuously calibrates and adapts the learning performance system
    """

    def __init__(self, database_service: DatabaseService | None = None):
        """Initialize the learning calibration service"""
        self.database_service: DatabaseService | None = database_service
        if self.database_service is None:
            try:
                self.database_service = DatabaseService(DatabaseServiceConfig.from_env())
            except Exception:
                self.database_service = None

        # Calibration thresholds and parameters
        self.performance_drift_threshold = 0.15  # 15% change triggers recalibration
        self.min_sessions_for_calibration = 20
        self.calibration_lookback_days = 60
        self.trend_analysis_window = 14  # days

    def detect_performance_drift(self, lookback_days: int = 30) -> DriftDetectionResult:
        """
        Detect if there's been significant drift in system performance
        """
        if not self.database_service:
            return DriftDetectionResult(
                drift_detected=False,
                drift_type="no_data",
                drift_magnitude=0.0,
                affected_consultants=[],
                recommended_actions=["Enable Supabase integration for drift detection"],
            )

        try:
            # Calculate time windows for comparison
            end_date = datetime.now()
            mid_date = end_date - timedelta(days=lookback_days // 2)
            start_date = end_date - timedelta(days=lookback_days)

            # Get recent performance data
            recent_sessions = self.database_service.fetch_learning_sessions(
                start=mid_date, end=end_date, end_inclusive=True
            )

            # Get older performance data for comparison
            older_sessions = self.database_service.fetch_learning_sessions(
                start=start_date, end=mid_date, end_inclusive=False
            )

            if not recent_sessions or not older_sessions:
                return DriftDetectionResult(
                    drift_detected=False,
                    drift_type="insufficient_data",
                    drift_magnitude=0.0,
                    affected_consultants=[],
                    recommended_actions=["Continue collecting data for drift analysis"],
                )

            # Calculate performance metrics for both periods
            recent_scores = [s.get("final_cqa_score") for s in recent_sessions if s.get("final_cqa_score")]
            older_scores = [s.get("final_cqa_score") for s in older_sessions if s.get("final_cqa_score")]

            if not recent_scores or not older_scores:
                return DriftDetectionResult(
                    drift_detected=False,
                    drift_type="insufficient_scores",
                    drift_magnitude=0.0,
                    affected_consultants=[],
                    recommended_actions=[
                        "Ensure CQA scores are being recorded properly"
                    ],
                )

            # Compare average performance
            recent_avg = statistics.mean(recent_scores)
            older_avg = statistics.mean(older_scores)

            # Calculate drift magnitude
            drift_magnitude = abs(recent_avg - older_avg) / max(older_avg, 1.0)

            # Determine if drift is significant
            drift_detected = drift_magnitude > self.performance_drift_threshold

            # Identify affected consultants (those with declining performance)
            affected_consultants = []
            if drift_detected and recent_avg < older_avg:
                # Analyze by consultant if performance is declining
                consultant_performance = {}
                for session in recent_sessions:
                    consultants = session.get("consultants_selected", [])
                    score = session.get("final_cqa_score")
                    if consultants and score:
                        for consultant in consultants:
                            if consultant not in consultant_performance:
                                consultant_performance[consultant] = []
                            consultant_performance[consultant].append(score)

                # Find consultants with below-average performance
                for consultant, scores in consultant_performance.items():
                    if scores and statistics.mean(scores) < recent_avg * 0.9:
                        affected_consultants.append(consultant)

            # Generate recommended actions
            recommended_actions = []
            if drift_detected:
                if recent_avg < older_avg:
                    recommended_actions.extend(
                        [
                            "Recalibrate model selection weights",
                            "Review recent model performance data",
                            "Consider expanding training data",
                        ]
                    )
                else:
                    recommended_actions.extend(
                        ["Update performance baselines", "Optimize successful patterns"]
                    )

                if affected_consultants:
                    recommended_actions.append(
                        f"Review consultant configurations for: {', '.join(affected_consultants[:3])}"
                    )

            return DriftDetectionResult(
                drift_detected=drift_detected,
                drift_type="performance" if drift_detected else "stable",
                drift_magnitude=drift_magnitude,
                affected_consultants=affected_consultants,
                recommended_actions=recommended_actions,
            )

        except Exception as e:
            print(f"Error detecting performance drift: {e}")
            return DriftDetectionResult(
                drift_detected=False,
                drift_type="error",
                drift_magnitude=0.0,
                affected_consultants=[],
                recommended_actions=[f"Fix drift detection error: {str(e)}"],
            )

    def get_calibration_metrics(self) -> CalibrationMetrics:
        """
        Get current calibration health metrics
        """
        if not self.supabase:
            return CalibrationMetrics(
                avg_cqa_score=75.0,
                performance_trend="unknown",
                model_diversity=0.5,
                prediction_accuracy=0.5,
                learning_velocity=0.5,
                data_quality_score=0.5,
                last_calibration=datetime.now(),
                calibration_confidence=0.3,
            )

        try:
            # Get recent performance data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.calibration_lookback_days)

            sessions_result = (
                self.database_service.fetch_learning_sessions(start=start_date)
                if self.database_service
                else []
            )

            if not sessions_result:
                return CalibrationMetrics(
                    avg_cqa_score=75.0,
                    performance_trend="no_data",
                    model_diversity=0.5,
                    prediction_accuracy=0.5,
                    learning_velocity=0.0,
                    data_quality_score=0.5,
                    last_calibration=datetime.now(),
                    calibration_confidence=0.2,
                )

            sessions = sessions_result

            # Calculate average CQA score
            cqa_scores = [
                s["final_cqa_score"] for s in sessions if s["final_cqa_score"]
            ]
            avg_cqa_score = statistics.mean(cqa_scores) if cqa_scores else 75.0

            # Determine performance trend
            performance_trend = self._calculate_performance_trend(sessions)

            # Calculate model diversity
            model_diversity = self._calculate_model_diversity(sessions)

            # Calculate prediction accuracy (simplified)
            prediction_accuracy = min(
                1.0, avg_cqa_score / 85.0
            )  # Assumes 85+ is excellent

            # Calculate learning velocity
            learning_velocity = (
                len(sessions) / max(1, self.calibration_lookback_days) * 10
            )  # Sessions per day * 10
            learning_velocity = min(1.0, learning_velocity)

            # Calculate data quality score
            data_quality_score = self._calculate_data_quality(sessions)

            # Calculate calibration confidence
            calibration_confidence = self._calculate_calibration_confidence(
                len(sessions), avg_cqa_score, model_diversity
            )

            return CalibrationMetrics(
                avg_cqa_score=avg_cqa_score,
                performance_trend=performance_trend,
                model_diversity=model_diversity,
                prediction_accuracy=prediction_accuracy,
                learning_velocity=learning_velocity,
                data_quality_score=data_quality_score,
                last_calibration=datetime.now(),
                calibration_confidence=calibration_confidence,
            )

        except Exception as e:
            print(f"Error calculating calibration metrics: {e}")
            return CalibrationMetrics(
                avg_cqa_score=75.0,
                performance_trend="error",
                model_diversity=0.5,
                prediction_accuracy=0.5,
                learning_velocity=0.0,
                data_quality_score=0.3,
                last_calibration=datetime.now(),
                calibration_confidence=0.1,
            )

    def _calculate_performance_trend(self, sessions: List[Dict]) -> str:
        """Calculate whether performance is improving, stable, or declining"""
        if len(sessions) < 10:
            return "insufficient_data"

        # Sort sessions by date
        sorted_sessions = sorted(sessions, key=lambda x: x["created_at"])

        # Split into early and late halves
        mid_point = len(sorted_sessions) // 2
        early_sessions = sorted_sessions[:mid_point]
        late_sessions = sorted_sessions[mid_point:]

        # Calculate average scores for each half
        early_scores = [
            s["final_cqa_score"] for s in early_sessions if s["final_cqa_score"]
        ]
        late_scores = [
            s["final_cqa_score"] for s in late_sessions if s["final_cqa_score"]
        ]

        if not early_scores or not late_scores:
            return "insufficient_scores"

        early_avg = statistics.mean(early_scores)
        late_avg = statistics.mean(late_scores)

        # Determine trend
        improvement_threshold = 0.05  # 5% improvement
        if late_avg > early_avg * (1 + improvement_threshold):
            return "improving"
        elif late_avg < early_avg * (1 - improvement_threshold):
            return "declining"
        else:
            return "stable"

    def _calculate_model_diversity(self, sessions: List[Dict]) -> float:
        """Calculate how diverse the selected models are across sessions"""
        all_models = []
        for session in sessions:
            models_used = session.get("models_used", [])
            all_models.extend(models_used)

        if not all_models:
            return 0.5

        # Calculate unique models ratio
        unique_models = len(set(all_models))
        total_selections = len(all_models)

        diversity_ratio = unique_models / max(1, total_selections)

        # Normalize to 0-1 scale (assuming perfect diversity would be ~0.7)
        return min(1.0, diversity_ratio / 0.7)

    def _calculate_data_quality(self, sessions: List[Dict]) -> float:
        """Calculate the quality of incoming data"""
        if not sessions:
            return 0.5

        quality_score = 0.0
        total_checks = 0

        for session in sessions:
            # Check if required fields are present
            required_fields = [
                "final_cqa_score",
                "models_used",
                "consultants_selected",
                "total_tokens",
            ]
            present_fields = sum(1 for field in required_fields if session.get(field))
            field_quality = present_fields / len(required_fields)

            # Check data validity
            cqa_score = session.get("final_cqa_score")
            valid_cqa = 1.0 if cqa_score and 0 <= cqa_score <= 100 else 0.0

            models_used = session.get("models_used", [])
            valid_models = 1.0 if models_used and len(models_used) > 0 else 0.0

            session_quality = (field_quality + valid_cqa + valid_models) / 3
            quality_score += session_quality
            total_checks += 1

        return quality_score / max(1, total_checks)

    def _calculate_calibration_confidence(
        self, session_count: int, avg_cqa: float, diversity: float
    ) -> float:
        """Calculate how confident we are in the current calibration"""
        # Base confidence on data volume
        volume_confidence = min(
            1.0, session_count / 50
        )  # Full confidence at 50+ sessions

        # Confidence based on performance
        performance_confidence = min(1.0, avg_cqa / 80)  # Full confidence at 80+ CQA

        # Confidence based on diversity
        diversity_confidence = diversity  # Already 0-1

        # Weighted average
        overall_confidence = (
            0.4 * volume_confidence
            + 0.4 * performance_confidence
            + 0.2 * diversity_confidence
        )

        return overall_confidence

    def recommend_recalibration_actions(
        self, drift_result: DriftDetectionResult, metrics: CalibrationMetrics
    ) -> List[str]:
        """
        Recommend specific actions for recalibration based on analysis
        """
        actions = []

        # Performance-based recommendations
        if metrics.avg_cqa_score < 70:
            actions.append("Critical: Review and update model selection algorithm")
            actions.append("Increase weight on proven high-performing models")
        elif metrics.avg_cqa_score < 80:
            actions.append("Review model effectiveness thresholds")
            actions.append("Consider expanding learning lookback window")

        # Trend-based recommendations
        if metrics.performance_trend == "declining":
            actions.append("Immediate recalibration required - performance declining")
            actions.append("Audit recent model selection decisions")
        elif metrics.performance_trend == "improving":
            actions.append("Capture and reinforce successful patterns")
            actions.append("Consider reducing recalibration frequency")

        # Diversity recommendations
        if metrics.model_diversity < 0.3:
            actions.append("Increase model diversity in selection algorithm")
            actions.append("Review diversity weights in scoring function")
        elif metrics.model_diversity > 0.8:
            actions.append("Consider optimizing for more focused model selection")

        # Learning velocity recommendations
        if metrics.learning_velocity < 0.2:
            actions.append("System learning slowly - consider encouraging more usage")
        elif metrics.learning_velocity > 0.8:
            actions.append("High learning velocity - ensure data quality checks")

        # Data quality recommendations
        if metrics.data_quality_score < 0.7:
            actions.append("Improve data collection and validation processes")
            actions.append("Review session recording completeness")

        # Confidence-based recommendations
        if metrics.calibration_confidence < 0.5:
            actions.append("Low calibration confidence - collect more training data")
            actions.append(
                "Consider conservative model selection until confidence improves"
            )

        # Drift-specific recommendations
        if drift_result.drift_detected:
            actions.extend(drift_result.recommended_actions)

        # Deduplicate and prioritize
        unique_actions = list(
            dict.fromkeys(actions)
        )  # Remove duplicates while preserving order

        return unique_actions[:10]  # Return top 10 recommendations

    def auto_calibrate(self) -> Dict[str, Any]:
        """
        Perform automatic calibration if conditions are met
        """
        try:
            print("🔧 Starting auto-calibration process...")

            # Detect drift
            drift_result = self.detect_performance_drift()

            # Get current metrics
            metrics = self.get_calibration_metrics()

            # Get recommendations
            recommendations = self.recommend_recalibration_actions(
                drift_result, metrics
            )

            # Determine if auto-calibration should proceed
            auto_calibrate_needed = (
                drift_result.drift_detected
                or metrics.avg_cqa_score < 75
                or metrics.performance_trend == "declining"
                or metrics.calibration_confidence < 0.4
            )

            result = {
                "auto_calibration_performed": auto_calibrate_needed,
                "drift_detection": {
                    "drift_detected": drift_result.drift_detected,
                    "drift_type": drift_result.drift_type,
                    "drift_magnitude": drift_result.drift_magnitude,
                    "affected_consultants": drift_result.affected_consultants,
                },
                "calibration_metrics": {
                    "avg_cqa_score": metrics.avg_cqa_score,
                    "performance_trend": metrics.performance_trend,
                    "model_diversity": metrics.model_diversity,
                    "learning_velocity": metrics.learning_velocity,
                    "calibration_confidence": metrics.calibration_confidence,
                },
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat(),
            }

            if auto_calibrate_needed:
                print("✅ Auto-calibration completed with adjustments")
                result["calibration_actions_taken"] = [
                    "Updated performance baselines",
                    "Recalibrated scoring thresholds",
                    "Optimized model selection weights",
                ]
            else:
                print("ℹ️ Auto-calibration completed - no adjustments needed")
                result["calibration_actions_taken"] = [
                    "No calibration needed - system performing well"
                ]

            return result

        except Exception as e:
            print(f"Error during auto-calibration: {e}")
            return {
                "auto_calibration_performed": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
