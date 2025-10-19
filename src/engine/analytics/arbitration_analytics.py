"""
Arbitration Analytics and A/B Testing Framework
Purpose: Track key metrics for Red Team Council feature and measure impact

This module provides:
1. Event tracking for arbitration interactions
2. Metric aggregation for dashboards
3. A/B test analysis
4. Outcome measurement (ValueOutcome scores)
"""

import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import UUID
import asyncpg

from src.engine.adapters.logging import get_logger  # Migrated
from src.engine.adapters.feature_flags import FeatureFlag, get_experiment_group  # Migrated
from src.config import get_settings

logger = get_logger(__name__, component="arbitration_analytics")
settings = get_settings()


class EventType(Enum):
    """Analytics event types"""

    # Arbitration events
    ARBITRATION_STARTED = "arbitration_started"
    CRITIQUE_PRIORITIZED = "critique_prioritized"
    CRITIQUE_DISAGREED = "critique_disagreed"
    CRITIQUE_NEUTRAL = "critique_neutral"
    USER_CRITIQUE_ADDED = "user_critique_added"
    ARBITRATION_COMPLETED = "arbitration_completed"

    # Synthesis events
    SYNTHESIS_STARTED = "synthesis_started"
    SYNTHESIS_COMPLETED = "synthesis_completed"
    SYNTHESIS_FEEDBACK_PROVIDED = "synthesis_feedback_provided"

    # Outcome events
    ENGAGEMENT_COMPLETED = "engagement_completed"
    VALUE_OUTCOME_MEASURED = "value_outcome_measured"
    USER_SATISFACTION_MEASURED = "user_satisfaction_measured"


@dataclass
class ArbitrationEvent:
    """Analytics event for arbitration interactions"""

    event_type: EventType
    engagement_id: UUID
    user_id: Optional[UUID]
    org_id: Optional[UUID]
    session_id: Optional[UUID]
    timestamp: datetime

    # Event-specific data
    critique_id: Optional[str] = None
    critique_type: Optional[str] = None
    disposition: Optional[str] = None
    rationale: Optional[str] = None
    user_critique_text: Optional[str] = None

    # Metrics
    time_spent_seconds: Optional[int] = None
    interaction_sequence: Optional[int] = None

    # A/B test context
    experiment_group: Optional[str] = None
    experiment_id: Optional[str] = None

    # Additional metadata
    metadata: Dict[str, Any] = None


@dataclass
class ArbitrationMetrics:
    """Aggregated metrics for arbitration session"""

    engagement_id: UUID
    session_id: UUID

    # Engagement metrics
    total_critiques_available: int
    critiques_prioritized: int
    critiques_disagreed: int
    critiques_neutral: int
    has_user_critique: bool
    user_critique_length: Optional[int]

    # Behavioral metrics
    time_to_first_action_seconds: Optional[int]
    total_time_spent_seconds: Optional[int]
    interaction_count: int
    review_depth_score: float  # 0-1, based on time and interactions

    # Rationale metrics
    rationales_provided: int
    avg_rationale_length: Optional[float]

    # Engagement level
    engagement_level: str  # minimal, low, medium, high, very_high

    # Outcomes
    synthesis_quality_score: Optional[float]  # 0-1
    user_satisfaction: Optional[int]  # 1-5
    value_outcome_score: Optional[float]  # 0-100
    business_outcome_achieved: Optional[bool]


class ArbitrationAnalyticsService:
    """Service for tracking and analyzing arbitration metrics"""

    def __init__(self, db_pool: Optional[asyncpg.Pool] = None):
        self.logger = logger.with_component("arbitration_analytics_service")
        self.db_pool = db_pool
        self._event_buffer: List[ArbitrationEvent] = []
        self._buffer_size = 100
        self._flush_interval = 30  # seconds
        self._last_flush = datetime.utcnow()

    async def initialize(self, db_pool: asyncpg.Pool):
        """Initialize with database connection pool"""
        self.db_pool = db_pool
        # Start background flush task
        asyncio.create_task(self._periodic_flush())

    async def track_event(
        self,
        event_type: EventType,
        engagement_id: UUID,
        user_id: Optional[UUID] = None,
        org_id: Optional[UUID] = None,
        session_id: Optional[UUID] = None,
        **kwargs,
    ):
        """Track an analytics event"""

        # Get experiment group if applicable
        experiment_group = get_experiment_group(
            FeatureFlag.ENABLE_PARALLEL_VALIDATION, user_id=user_id, org_id=org_id
        )

        event = ArbitrationEvent(
            event_type=event_type,
            engagement_id=engagement_id,
            user_id=user_id,
            org_id=org_id,
            session_id=session_id,
            timestamp=datetime.utcnow(),
            experiment_group=experiment_group,
            experiment_id="red_team_council_v1" if experiment_group else None,
            **kwargs,
        )

        self._event_buffer.append(event)

        # Flush if buffer is full
        if len(self._event_buffer) >= self._buffer_size:
            await self._flush_events()

        self.logger.info(
            "event_tracked",
            event_type=event_type.value,
            engagement_id=str(engagement_id),
            experiment_group=experiment_group,
        )

    async def track_arbitration_start(
        self,
        engagement_id: UUID,
        session_id: UUID,
        user_id: Optional[UUID],
        total_critiques: int,
    ):
        """Track arbitration session start"""
        await self.track_event(
            EventType.ARBITRATION_STARTED,
            engagement_id,
            user_id=user_id,
            session_id=session_id,
            metadata={
                "total_critiques": total_critiques,
                "start_time": datetime.utcnow().isoformat(),
            },
        )

    async def track_critique_disposition(
        self,
        engagement_id: UUID,
        session_id: UUID,
        user_id: Optional[UUID],
        critique_id: str,
        critique_type: str,
        disposition: str,
        rationale: Optional[str] = None,
        interaction_sequence: int = 0,
    ):
        """Track individual critique disposition"""

        event_type_map = {
            "PRIORITIZE": EventType.CRITIQUE_PRIORITIZED,
            "DISAGREE": EventType.CRITIQUE_DISAGREED,
            "NEUTRAL": EventType.CRITIQUE_NEUTRAL,
        }

        await self.track_event(
            event_type_map.get(disposition, EventType.CRITIQUE_NEUTRAL),
            engagement_id,
            user_id=user_id,
            session_id=session_id,
            critique_id=critique_id,
            critique_type=critique_type,
            disposition=disposition,
            rationale=rationale,
            interaction_sequence=interaction_sequence,
        )

    async def track_user_critique(
        self,
        engagement_id: UUID,
        session_id: UUID,
        user_id: Optional[UUID],
        critique_text: str,
    ):
        """Track user-generated critique"""
        await self.track_event(
            EventType.USER_CRITIQUE_ADDED,
            engagement_id,
            user_id=user_id,
            session_id=session_id,
            user_critique_text=critique_text,
            metadata={
                "word_count": len(critique_text.split()),
                "char_count": len(critique_text),
            },
        )

    async def track_arbitration_completion(
        self,
        engagement_id: UUID,
        session_id: UUID,
        user_id: Optional[UUID],
        metrics: ArbitrationMetrics,
    ):
        """Track arbitration session completion with metrics"""
        await self.track_event(
            EventType.ARBITRATION_COMPLETED,
            engagement_id,
            user_id=user_id,
            session_id=session_id,
            time_spent_seconds=metrics.total_time_spent_seconds,
            metadata=asdict(metrics),
        )

        # Store metrics in database
        if self.db_pool:
            await self._store_metrics(metrics)

    async def track_value_outcome(
        self,
        engagement_id: UUID,
        user_id: Optional[UUID],
        value_score: float,
        business_outcome_achieved: bool,
        user_satisfaction: Optional[int] = None,
    ):
        """Track business value outcome"""

        experiment_group = get_experiment_group(
            FeatureFlag.ENABLE_PARALLEL_VALIDATION, user_id=user_id
        )

        await self.track_event(
            EventType.VALUE_OUTCOME_MEASURED,
            engagement_id,
            user_id=user_id,
            metadata={
                "value_score": value_score,
                "business_outcome_achieved": business_outcome_achieved,
                "user_satisfaction": user_satisfaction,
                "experiment_group": experiment_group,
            },
        )

        # Update flywheel metrics
        if self.db_pool:
            await self._update_outcome_metrics(
                engagement_id,
                value_score,
                business_outcome_achieved,
                user_satisfaction,
                experiment_group,
            )

    async def get_ab_test_results(
        self, experiment_id: str = "red_team_council_v1", days_back: int = 30
    ) -> Dict[str, Any]:
        """Get A/B test results comparing control vs treatment groups"""

        if not self.db_pool:
            return {}

        cutoff_date = datetime.utcnow() - timedelta(days=days_back)

        async with self.db_pool.acquire() as conn:
            # Get metrics by experiment group
            query = """
                SELECT 
                    f.experiment_group,
                    COUNT(DISTINCT f.engagement_id) as engagement_count,
                    AVG(f.synthesis_quality_score) as avg_synthesis_quality,
                    AVG(CASE WHEN f.business_outcome_achieved THEN 1 ELSE 0 END) as business_outcome_rate,
                    AVG(f.post_arbitration_confidence - f.pre_arbitration_confidence) as avg_confidence_lift,
                    AVG(a.prioritized_count) as avg_prioritized,
                    AVG(a.disagreed_count) as avg_disagreed,
                    AVG(CASE WHEN a.has_user_critique THEN 1 ELSE 0 END) as user_critique_rate,
                    AVG(CASE 
                        WHEN a.engagement_level = 'very_high' THEN 5
                        WHEN a.engagement_level = 'high' THEN 4
                        WHEN a.engagement_level = 'medium' THEN 3
                        WHEN a.engagement_level = 'low' THEN 2
                        ELSE 1
                    END) as avg_engagement_score
                FROM flywheel_metrics f
                JOIN arbitration_sessions a ON f.arbitration_session_id = a.id
                WHERE f.experiment_id = $1
                    AND f.created_at >= $2
                    AND f.experiment_group IN ('control', 'treatment')
                GROUP BY f.experiment_group
            """

            results = await conn.fetch(query, experiment_id, cutoff_date)

            # Calculate lift metrics
            control_data = None
            treatment_data = None

            for row in results:
                if row["experiment_group"] == "control":
                    control_data = dict(row)
                elif row["experiment_group"] == "treatment":
                    treatment_data = dict(row)

            if control_data and treatment_data:
                lift_metrics = self._calculate_lift_metrics(
                    control_data, treatment_data
                )
            else:
                lift_metrics = {}

            return {
                "experiment_id": experiment_id,
                "period_days": days_back,
                "control": control_data,
                "treatment": treatment_data,
                "lift_metrics": lift_metrics,
                "statistical_significance": (
                    await self._calculate_significance(control_data, treatment_data)
                    if control_data and treatment_data
                    else None
                ),
            }

    async def get_engagement_analytics(self, days_back: int = 7) -> Dict[str, Any]:
        """Get engagement analytics for dashboard"""

        if not self.db_pool:
            return {}

        cutoff_date = datetime.utcnow() - timedelta(days=days_back)

        async with self.db_pool.acquire() as conn:
            # Get daily metrics
            query = """
                SELECT 
                    date,
                    total_sessions,
                    unique_engagements,
                    very_high_engagement,
                    high_engagement,
                    medium_engagement,
                    low_engagement,
                    minimal_engagement,
                    avg_prioritized,
                    avg_disagreed,
                    sessions_with_user_critique,
                    sessions_with_rationales,
                    avg_synthesis_quality,
                    business_outcome_rate,
                    avg_confidence_lift
                FROM arbitration_analytics
                WHERE date >= $1
                ORDER BY date DESC
            """

            daily_metrics = await conn.fetch(query, cutoff_date)

            # Get top critique patterns
            critique_query = """
                SELECT 
                    critique_type,
                    disposition,
                    COUNT(*) as count,
                    AVG(LENGTH(rationale)) as avg_rationale_length
                FROM critique_dispositions c
                JOIN arbitration_sessions a ON c.arbitration_session_id = a.id
                WHERE a.created_at >= $1
                GROUP BY critique_type, disposition
                ORDER BY count DESC
                LIMIT 20
            """

            critique_patterns = await conn.fetch(critique_query, cutoff_date)

            # Get user critique themes (simplified - in production use NLP)
            user_critique_query = """
                SELECT 
                    problem_domain,
                    COUNT(*) as count,
                    AVG(word_count) as avg_length,
                    AVG(user_satisfaction_score) as avg_satisfaction
                FROM user_generated_critiques
                WHERE created_at >= $1
                    AND problem_domain IS NOT NULL
                GROUP BY problem_domain
                ORDER BY count DESC
                LIMIT 10
            """

            user_critique_themes = await conn.fetch(user_critique_query, cutoff_date)

            return {
                "period_days": days_back,
                "daily_metrics": [dict(row) for row in daily_metrics],
                "critique_patterns": [dict(row) for row in critique_patterns],
                "user_critique_themes": [dict(row) for row in user_critique_themes],
                "summary": self._calculate_summary_metrics(daily_metrics),
            }

    def _calculate_lift_metrics(
        self, control: Dict[str, Any], treatment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate lift metrics between control and treatment"""

        def safe_lift(treatment_val, control_val):
            if control_val and control_val != 0:
                return ((treatment_val or 0) - control_val) / control_val * 100
            return 0

        return {
            "synthesis_quality_lift": safe_lift(
                treatment.get("avg_synthesis_quality"),
                control.get("avg_synthesis_quality"),
            ),
            "business_outcome_lift": safe_lift(
                treatment.get("business_outcome_rate"),
                control.get("business_outcome_rate"),
            ),
            "confidence_lift_improvement": safe_lift(
                treatment.get("avg_confidence_lift"), control.get("avg_confidence_lift")
            ),
            "engagement_score_lift": safe_lift(
                treatment.get("avg_engagement_score"),
                control.get("avg_engagement_score"),
            ),
            "user_critique_rate_lift": safe_lift(
                treatment.get("user_critique_rate"), control.get("user_critique_rate")
            ),
        }

    async def _calculate_significance(
        self, control: Dict[str, Any], treatment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate statistical significance (simplified)"""

        # In production, use proper statistical tests (t-test, chi-square, etc.)
        # This is a simplified example

        control_n = control.get("engagement_count", 0)
        treatment_n = treatment.get("engagement_count", 0)

        # Minimum sample size check
        min_sample_size = 30
        sufficient_data = (
            control_n >= min_sample_size and treatment_n >= min_sample_size
        )

        # Simplified significance check for business outcome rate
        control_rate = control.get("business_outcome_rate", 0)
        treatment_rate = treatment.get("business_outcome_rate", 0)

        # Very simplified - in production use proper statistical library
        rate_difference = abs(treatment_rate - control_rate)
        is_significant = rate_difference > 0.05 and sufficient_data

        return {
            "sufficient_data": sufficient_data,
            "control_sample_size": control_n,
            "treatment_sample_size": treatment_n,
            "is_significant": is_significant,
            "confidence_level": 0.95 if is_significant else None,
            "p_value": 0.03 if is_significant else 0.15,  # Placeholder
            "recommendation": self._get_ab_test_recommendation(
                sufficient_data, is_significant, control, treatment
            ),
        }

    def _get_ab_test_recommendation(
        self,
        sufficient_data: bool,
        is_significant: bool,
        control: Dict[str, Any],
        treatment: Dict[str, Any],
    ) -> str:
        """Generate recommendation based on A/B test results"""

        if not sufficient_data:
            return "Continue experiment - insufficient data for conclusion"

        if not is_significant:
            return "No significant difference detected - continue monitoring"

        treatment_better = treatment.get("business_outcome_rate", 0) > control.get(
            "business_outcome_rate", 0
        )

        if treatment_better:
            lift = (
                treatment.get("business_outcome_rate", 0)
                - control.get("business_outcome_rate", 0)
            ) * 100
            return f"Treatment shows {lift:.1f}% improvement - consider gradual rollout"
        else:
            return "Control performs better - investigate treatment issues"

    def _calculate_summary_metrics(
        self, daily_metrics: List[asyncpg.Record]
    ) -> Dict[str, Any]:
        """Calculate summary metrics from daily data"""

        if not daily_metrics:
            return {}

        total_sessions = sum(row["total_sessions"] for row in daily_metrics)
        total_engagements = sum(row["unique_engagements"] for row in daily_metrics)

        # Calculate engagement distribution
        engagement_dist = {
            "very_high": sum(row["very_high_engagement"] for row in daily_metrics),
            "high": sum(row["high_engagement"] for row in daily_metrics),
            "medium": sum(row["medium_engagement"] for row in daily_metrics),
            "low": sum(row["low_engagement"] for row in daily_metrics),
            "minimal": sum(row["minimal_engagement"] for row in daily_metrics),
        }

        # Calculate averages
        avg_metrics = {
            "avg_prioritized": sum(row["avg_prioritized"] for row in daily_metrics)
            / len(daily_metrics),
            "avg_disagreed": sum(row["avg_disagreed"] for row in daily_metrics)
            / len(daily_metrics),
            "user_critique_rate": (
                sum(row["sessions_with_user_critique"] for row in daily_metrics)
                / total_sessions
                if total_sessions > 0
                else 0
            ),
            "rationale_rate": (
                sum(row["sessions_with_rationales"] for row in daily_metrics)
                / total_sessions
                if total_sessions > 0
                else 0
            ),
        }

        return {
            "total_sessions": total_sessions,
            "unique_engagements": total_engagements,
            "engagement_distribution": engagement_dist,
            "average_metrics": avg_metrics,
        }

    async def _store_metrics(self, metrics: ArbitrationMetrics):
        """Store metrics in database"""

        if not self.db_pool:
            return

        async with self.db_pool.acquire() as conn:
            query = """
                INSERT INTO flywheel_metrics (
                    arbitration_session_id,
                    engagement_id,
                    time_spent_seconds,
                    interaction_count,
                    critique_review_depth,
                    pre_arbitration_confidence,
                    post_arbitration_confidence,
                    synthesis_quality_score,
                    business_outcome_achieved,
                    experiment_group,
                    experiment_id
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (arbitration_session_id) 
                DO UPDATE SET
                    time_spent_seconds = EXCLUDED.time_spent_seconds,
                    interaction_count = EXCLUDED.interaction_count,
                    synthesis_quality_score = EXCLUDED.synthesis_quality_score,
                    updated_at = NOW()
            """

            # Determine review depth based on score
            review_depth = (
                "deep_analysis"
                if metrics.review_depth_score > 0.7
                else ("reviewed" if metrics.review_depth_score > 0.3 else "skimmed")
            )

            await conn.execute(
                query,
                metrics.session_id,
                metrics.engagement_id,
                metrics.total_time_spent_seconds,
                metrics.interaction_count,
                review_depth,
                0.5,  # Default pre-confidence
                0.7,  # Default post-confidence
                metrics.synthesis_quality_score,
                metrics.business_outcome_achieved,
                get_experiment_group(FeatureFlag.ENABLE_PARALLEL_VALIDATION),
                "red_team_council_v1",
            )

    async def _update_outcome_metrics(
        self,
        engagement_id: UUID,
        value_score: float,
        business_outcome_achieved: bool,
        user_satisfaction: Optional[int],
        experiment_group: Optional[str],
    ):
        """Update outcome metrics in database"""

        if not self.db_pool:
            return

        async with self.db_pool.acquire() as conn:
            query = """
                UPDATE flywheel_metrics
                SET 
                    synthesis_quality_score = $2,
                    business_outcome_achieved = $3,
                    updated_at = NOW()
                WHERE engagement_id = $1
                    AND experiment_group = $4
            """

            await conn.execute(
                query,
                engagement_id,
                value_score / 100.0,  # Convert to 0-1 scale
                business_outcome_achieved,
                experiment_group,
            )

    async def _flush_events(self):
        """Flush event buffer to storage"""

        if not self._event_buffer:
            return

        events_to_flush = self._event_buffer.copy()
        self._event_buffer.clear()

        # In production, send to analytics pipeline (e.g., Segment, Amplitude)
        # For now, just log
        self.logger.info("flushing_analytics_events", event_count=len(events_to_flush))

        self._last_flush = datetime.utcnow()

    async def _periodic_flush(self):
        """Periodically flush events"""
        while True:
            await asyncio.sleep(self._flush_interval)

            if (
                datetime.utcnow() - self._last_flush
            ).total_seconds() >= self._flush_interval:
                await self._flush_events()


# Singleton instance
_analytics_service = None


async def get_analytics_service(
    db_pool: Optional[asyncpg.Pool] = None,
) -> ArbitrationAnalyticsService:
    """Get the singleton analytics service"""
    global _analytics_service
    if _analytics_service is None:
        _analytics_service = ArbitrationAnalyticsService(db_pool)
        if db_pool:
            await _analytics_service.initialize(db_pool)
    return _analytics_service


# Convenience functions
async def track_arbitration_event(**kwargs):
    """Track an arbitration event"""
    service = await get_analytics_service()
    await service.track_event(**kwargs)


async def track_value_outcome(
    engagement_id: UUID, value_score: float, business_outcome_achieved: bool, **kwargs
):
    """Track a value outcome"""
    service = await get_analytics_service()
    await service.track_value_outcome(
        engagement_id,
        value_score=value_score,
        business_outcome_achieved=business_outcome_achieved,
        **kwargs,
    )
