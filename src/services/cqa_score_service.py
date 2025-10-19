"""
CQA Score Service - Flywheel Feedback Integration
===============================================

Service for retrieving Mental Model CQA scores to feed into the
ContextualLollapaloozaEngine for data-driven consultant selection.

This service bridges the gap between our quality measurement system
and our prediction engine, creating a true learning feedback loop.
"""

import logging
import sqlite3
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CQAScore:
    """Mental Model CQA Score for orchestrator consumption"""

    model_name: str
    overall_score: float
    weighted_score: float
    quality_tier: str  # excellent, good, average, poor
    validation_status: str  # passed, failed, review_needed
    confidence_level: float
    evaluation_timestamp: datetime
    sample_size: int = 1  # Number of evaluations averaged


@dataclass
class ConsultantCQAProfile:
    """Aggregated CQA profile for a consultant's mental models"""

    consultant_name: str
    mental_models: List[str]
    average_cqa_score: float
    weighted_cqa_score: float
    model_quality_distribution: Dict[str, int]  # excellent, good, average, poor counts
    confidence_level: float
    last_updated: datetime


class CQAScoreService:
    """
    Service for retrieving and aggregating CQA scores for orchestrator use.

    Provides the data bridge between quality measurement and prediction,
    enabling the ContextualLollapaloozaEngine to make data-driven decisions.
    """

    def __init__(self, db_path: str = "evaluation_results.db"):
        """
        Initialize CQA Score Service.

        Args:
            db_path: Path to evaluation results database
        """
        self.db_path = db_path
        self._ensure_database_exists()

    def _ensure_database_exists(self):
        """Ensure the database and required tables exist."""
        db_file = Path(self.db_path)
        if not db_file.exists():
            logger.warning(f"⚠️ CQA database not found: {self.db_path}")
            return

        # Verify required tables exist
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Check if evaluation_results table exists
            cursor.execute(
                """
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='evaluation_results'
            """
            )

            if not cursor.fetchone():
                logger.warning("⚠️ evaluation_results table not found in CQA database")
        finally:
            conn.close()

    def get_mental_model_score(
        self, model_name: str, max_age_hours: int = 720
    ) -> Optional[CQAScore]:
        """
        Get the latest CQA score for a specific mental model.

        Args:
            model_name: Name of the mental model
            max_age_hours: Maximum age of score in hours (default: 30 days)

        Returns:
            CQAScore object or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Calculate cutoff time
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

            # Query for recent CQA results from mental_model_cqa_results table
            query = """
                SELECT
                    mental_model_id,
                    mental_model_name,
                    overall_score,
                    weighted_score,
                    quality_tier,
                    validation_status,
                    evaluation_timestamp,
                    created_at
                FROM mental_model_cqa_results
                WHERE mental_model_name LIKE ?
                OR mental_model_id LIKE ?
                ORDER BY created_at DESC
                LIMIT 5
            """

            cursor.execute(query, (f"%{model_name}%", f"%{model_name}%"))
            results = cursor.fetchall()

            if not results:
                logger.debug(f"🔍 No CQA scores found for model: {model_name}")
                return None

            # Parse most recent result
            latest = results[0]
            model_id = latest[0]
            model_name_db = latest[1]
            overall_score = float(latest[2])
            weighted_score = float(latest[3])
            quality_tier = latest[4]
            validation_status = latest[5]
            eval_timestamp = latest[6]
            created_at = latest[7]

            # Calculate aggregated metrics if multiple results
            scores = [float(r[2]) for r in results]  # overall_score column
            avg_score = sum(scores) / len(scores)

            # Estimate confidence based on consistency of scores
            if len(scores) > 1:
                score_variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
                confidence = max(0.5, min(0.95, 1.0 - (score_variance / 10.0)))
            else:
                confidence = 0.7

            return CQAScore(
                model_name=model_name_db,
                overall_score=overall_score,
                weighted_score=weighted_score,
                quality_tier=quality_tier,
                validation_status=validation_status,
                confidence_level=confidence,
                evaluation_timestamp=datetime.fromisoformat(eval_timestamp),
                sample_size=len(results),
            )

        except Exception as e:
            logger.error(f"❌ Failed to retrieve CQA score for {model_name}: {e}")
            return None
        finally:
            conn.close()

    def get_consultant_cqa_profile(
        self, consultant_data: Dict[str, Any]
    ) -> ConsultantCQAProfile:
        """
        Generate aggregated CQA profile for a consultant based on their mental models.

        Args:
            consultant_data: Consultant data including mental_models list

        Returns:
            ConsultantCQAProfile with aggregated quality metrics
        """
        consultant_name = consultant_data.get("type", "unknown_consultant")
        mental_models = consultant_data.get("mental_models", [])

        if not mental_models:
            # Return default profile for consultants without mental models
            return ConsultantCQAProfile(
                consultant_name=consultant_name,
                mental_models=[],
                average_cqa_score=5.0,  # Neutral default
                weighted_cqa_score=5.0,
                model_quality_distribution={"average": 1},
                confidence_level=0.5,
                last_updated=datetime.now(),
            )

        # Retrieve CQA scores for all mental models
        cqa_scores = []
        quality_distribution = {"excellent": 0, "good": 0, "average": 0, "poor": 0}
        confidences = []

        for model_name in mental_models:
            score = self.get_mental_model_score(model_name)
            if score:
                cqa_scores.append(score)
                quality_distribution[score.quality_tier] += 1
                confidences.append(score.confidence_level)
            else:
                # Default score for models without CQA data
                cqa_scores.append(
                    CQAScore(
                        model_name=model_name,
                        overall_score=5.0,
                        weighted_score=5.0,
                        quality_tier="average",
                        validation_status="review_needed",
                        confidence_level=0.5,
                        evaluation_timestamp=datetime.now(),
                        sample_size=0,
                    )
                )
                quality_distribution["average"] += 1
                confidences.append(0.5)

        # Calculate aggregated metrics
        if cqa_scores:
            avg_score = sum(s.overall_score for s in cqa_scores) / len(cqa_scores)
            weighted_score = sum(s.weighted_score for s in cqa_scores) / len(cqa_scores)
            avg_confidence = sum(confidences) / len(confidences)
        else:
            avg_score = weighted_score = 5.0
            avg_confidence = 0.5

        return ConsultantCQAProfile(
            consultant_name=consultant_name,
            mental_models=mental_models,
            average_cqa_score=avg_score,
            weighted_cqa_score=weighted_score,
            model_quality_distribution=quality_distribution,
            confidence_level=avg_confidence,
            last_updated=datetime.now(),
        )

    def get_bulk_model_scores(
        self, model_names: List[str]
    ) -> Dict[str, Optional[CQAScore]]:
        """
        Retrieve CQA scores for multiple mental models efficiently.

        Args:
            model_names: List of mental model names

        Returns:
            Dictionary mapping model names to CQAScore objects
        """
        results = {}

        for model_name in model_names:
            results[model_name] = self.get_mental_model_score(model_name)

        logger.info(
            f"📊 Retrieved CQA scores for {len(model_names)} models: "
            f"{sum(1 for s in results.values() if s is not None)} found"
        )

        return results

    def calculate_cqa_effectiveness_boost(self, cqa_score: CQAScore) -> float:
        """
        Calculate effectiveness boost multiplier based on CQA score.

        This is the key function that translates quality measurements
        into prediction adjustments for the orchestrator.

        Args:
            cqa_score: CQA score for a mental model

        Returns:
            Multiplier factor (0.7 to 1.3) to adjust consultant effectiveness
        """
        if not cqa_score:
            return 1.0  # Neutral multiplier

        base_score = cqa_score.weighted_score
        confidence = cqa_score.confidence_level

        # Quality-based boost calculation
        if base_score >= 9.0:
            quality_boost = 1.25  # 25% boost for excellent models
        elif base_score >= 8.0:
            quality_boost = 1.15  # 15% boost for very good models
        elif base_score >= 7.0:
            quality_boost = 1.05  # 5% boost for good models
        elif base_score >= 6.0:
            quality_boost = 1.0  # Neutral for average models
        elif base_score >= 5.0:
            quality_boost = 0.95  # 5% penalty for below average
        else:
            quality_boost = 0.85  # 15% penalty for poor models

        # Confidence adjustment
        confidence_adjustment = 0.8 + (confidence * 0.4)  # Range: 0.8 to 1.2

        # Validation status adjustment
        validation_multiplier = {
            "passed": 1.0,
            "review_needed": 0.95,
            "failed": 0.9,
        }.get(cqa_score.validation_status, 0.95)

        # Combine all factors
        final_multiplier = quality_boost * confidence_adjustment * validation_multiplier

        # Cap between 0.7 and 1.3 to prevent extreme adjustments
        return max(0.7, min(1.3, final_multiplier))


# Global service instance
_cqa_score_service = None


def get_cqa_score_service() -> CQAScoreService:
    """Get global CQA Score Service instance."""
    global _cqa_score_service
    if _cqa_score_service is None:
        _cqa_score_service = CQAScoreService()
    return _cqa_score_service
