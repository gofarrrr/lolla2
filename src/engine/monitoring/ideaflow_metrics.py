"""
Ideaflow Metrics Tracking
Simple metrics system for tracking creative ideation performance

Implements key metrics from Ideaflow methodology:
- Ideas per minute (velocity)
- Semantic diversity score
- Novelty assessment
- Quality vs quantity balance

Integrates with existing performance monitoring without adding complexity.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import statistics

logger = logging.getLogger(__name__)


@dataclass
class IdeaflowMetrics:
    """Ideaflow performance metrics for a single session"""

    session_id: str
    problem_type: str
    total_ideas: int
    generation_time_minutes: float
    ideas_per_minute: float
    diversity_score: float
    novelty_score: float
    quality_score: float
    timestamp: str


@dataclass
class IdeaflowTrends:
    """Trending analysis across multiple sessions"""

    sessions_analyzed: int
    avg_ideas_per_minute: float
    avg_diversity_score: float
    avg_novelty_score: float
    improvement_trend: str  # 'improving', 'stable', 'declining'
    peak_performance_session: str
    performance_consistency: float


class IdeaflowMetricsTracker:
    """
    Tracks and analyzes ideaflow performance metrics.

    Focused implementation that:
    1. Measures creative velocity (ideas per minute)
    2. Assesses idea diversity using simple semantic analysis
    3. Tracks quality vs quantity balance
    4. Identifies performance trends over time
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session_history: List[IdeaflowMetrics] = []
        self.performance_cache: Dict[str, float] = {}

    def calculate_session_metrics(
        self,
        session_id: str,
        problem_type: str,
        ideas: List[str],
        duration_seconds: float,
        quality_scores: Optional[List[float]] = None,
    ) -> IdeaflowMetrics:
        """
        Calculate comprehensive ideaflow metrics for a session.

        Args:
            session_id: Unique session identifier
            problem_type: Type of problem (creative_ideation, etc.)
            ideas: List of generated ideas/solutions
            duration_seconds: Time spent generating ideas
            quality_scores: Optional quality scores for each idea

        Returns:
            IdeaflowMetrics with calculated scores
        """
        try:
            duration_minutes = duration_seconds / 60.0

            # Basic velocity metrics
            total_ideas = len(ideas)
            ideas_per_minute = (
                total_ideas / duration_minutes if duration_minutes > 0 else 0
            )

            # Diversity analysis
            diversity_score = self._calculate_diversity_score(ideas)

            # Novelty assessment
            novelty_score = self._calculate_novelty_score(ideas)

            # Quality analysis
            if quality_scores:
                quality_score = statistics.mean(quality_scores)
            else:
                quality_score = self._estimate_quality_score(ideas)

            metrics = IdeaflowMetrics(
                session_id=session_id,
                problem_type=problem_type,
                total_ideas=total_ideas,
                generation_time_minutes=duration_minutes,
                ideas_per_minute=ideas_per_minute,
                diversity_score=diversity_score,
                novelty_score=novelty_score,
                quality_score=quality_score,
                timestamp=datetime.utcnow().isoformat(),
            )

            # Store in history
            self.session_history.append(metrics)

            self.logger.info(
                f"Ideaflow metrics calculated: {total_ideas} ideas, "
                f"{ideas_per_minute:.1f} ideas/min, diversity: {diversity_score:.2f}"
            )

            return metrics

        except Exception as e:
            self.logger.error(f"Failed to calculate ideaflow metrics: {e}")
            return IdeaflowMetrics(
                session_id=session_id,
                problem_type=problem_type,
                total_ideas=0,
                generation_time_minutes=0,
                ideas_per_minute=0,
                diversity_score=0,
                novelty_score=0,
                quality_score=0,
                timestamp=datetime.utcnow().isoformat(),
            )

    def _calculate_diversity_score(self, ideas: List[str]) -> float:
        """
        Calculate semantic diversity of ideas using simple text analysis.

        Returns score 0.0-1.0 where higher means more diverse.
        """
        if len(ideas) <= 1:
            return 0.0

        try:
            # Simple diversity based on unique words and length variation
            all_words = set()
            lengths = []

            for idea in ideas:
                words = set(idea.lower().split())
                all_words.update(words)
                lengths.append(len(idea))

            # Word diversity component (unique words vs total words)
            total_words = sum(len(idea.split()) for idea in ideas)
            word_diversity = len(all_words) / total_words if total_words > 0 else 0

            # Length variation component
            length_variation = (
                statistics.stdev(lengths) / statistics.mean(lengths)
                if len(lengths) > 1
                else 0
            )
            length_variation = min(length_variation, 1.0)  # Cap at 1.0

            # Semantic uniqueness (simple jaccard-like measure)
            unique_pairs = 0
            total_pairs = 0

            for i, idea1 in enumerate(ideas):
                for idea2 in ideas[i + 1 :]:
                    words1 = set(idea1.lower().split())
                    words2 = set(idea2.lower().split())

                    if len(words1) > 0 and len(words2) > 0:
                        jaccard = len(words1.intersection(words2)) / len(
                            words1.union(words2)
                        )
                        if jaccard < 0.5:  # Ideas are sufficiently different
                            unique_pairs += 1
                        total_pairs += 1

            uniqueness_ratio = unique_pairs / total_pairs if total_pairs > 0 else 0

            # Combined diversity score
            diversity_score = (
                word_diversity * 0.4 + length_variation * 0.2 + uniqueness_ratio * 0.4
            )

            return min(max(diversity_score, 0.0), 1.0)

        except Exception as e:
            self.logger.warning(f"Diversity calculation failed: {e}")
            return 0.5  # Default moderate diversity

    def _calculate_novelty_score(self, ideas: List[str]) -> float:
        """
        Assess novelty of ideas compared to common/expected solutions.

        Returns score 0.0-1.0 where higher means more novel.
        """
        if not ideas:
            return 0.0

        try:
            # Simple novelty based on uncommon words and phrases
            common_business_words = {
                "improve",
                "increase",
                "optimize",
                "reduce",
                "cost",
                "efficiency",
                "better",
                "faster",
                "more",
                "less",
                "customer",
                "user",
                "system",
                "process",
                "technology",
                "solution",
                "strategy",
                "plan",
                "implement",
            }

            novelty_indicators = [
                "unusual",
                "unique",
                "innovative",
                "creative",
                "unexpected",
                "surprising",
                "unconventional",
                "breakthrough",
                "revolutionary",
                "disruptive",
                "novel",
                "original",
                "imaginative",
                "inventive",
                "experimental",
            ]

            total_words = 0
            novel_word_count = 0
            common_word_count = 0

            for idea in ideas:
                words = idea.lower().split()
                total_words += len(words)

                for word in words:
                    if word in novelty_indicators:
                        novel_word_count += 1
                    elif word in common_business_words:
                        common_word_count += 1

            # Calculate novelty ratio
            if total_words == 0:
                return 0.0

            novel_ratio = novel_word_count / total_words
            common_ratio = common_word_count / total_words

            # Novelty score: high novel words, low common words
            novelty_score = (novel_ratio * 2.0) - (common_ratio * 0.5)

            # Add bonus for length and complexity (novel ideas often more elaborate)
            avg_length = statistics.mean(len(idea) for idea in ideas)
            length_bonus = min(
                avg_length / 100.0, 0.3
            )  # Up to 0.3 bonus for longer ideas

            total_novelty = novelty_score + length_bonus

            return min(max(total_novelty, 0.0), 1.0)

        except Exception as e:
            self.logger.warning(f"Novelty calculation failed: {e}")
            return 0.5  # Default moderate novelty

    def _estimate_quality_score(self, ideas: List[str]) -> float:
        """
        Estimate quality when explicit quality scores unavailable.

        Simple heuristic based on completeness and specificity.
        """
        if not ideas:
            return 0.0

        try:
            quality_indicators = []

            for idea in ideas:
                score = 0.0

                # Length indicates development level
                if len(idea) > 20:
                    score += 0.2
                if len(idea) > 50:
                    score += 0.1

                # Specific details indicate quality
                if any(
                    word in idea.lower()
                    for word in ["how", "why", "when", "where", "who"]
                ):
                    score += 0.2

                # Action orientation
                if any(
                    word in idea.lower()
                    for word in ["implement", "create", "develop", "build", "design"]
                ):
                    score += 0.2

                # Measurable outcomes
                if any(char in idea for char in ["%", "$", "#"]) or any(
                    word in idea.lower()
                    for word in ["increase", "reduce", "improve", "measure"]
                ):
                    score += 0.2

                # Feasibility indicators
                if any(
                    word in idea.lower()
                    for word in [
                        "team",
                        "resource",
                        "timeline",
                        "budget",
                        "stakeholder",
                    ]
                ):
                    score += 0.1

                quality_indicators.append(min(score, 1.0))

            return statistics.mean(quality_indicators)

        except Exception as e:
            self.logger.warning(f"Quality estimation failed: {e}")
            return 0.6  # Default moderate quality

    def analyze_performance_trends(self, days_back: int = 30) -> IdeaflowTrends:
        """
        Analyze performance trends over recent sessions.

        Args:
            days_back: Number of days to analyze

        Returns:
            IdeaflowTrends with trend analysis
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)

        recent_sessions = [
            session
            for session in self.session_history
            if datetime.fromisoformat(session.timestamp.replace("Z", "+00:00"))
            > cutoff_date
        ]

        if len(recent_sessions) < 2:
            return IdeaflowTrends(
                sessions_analyzed=len(recent_sessions),
                avg_ideas_per_minute=0,
                avg_diversity_score=0,
                avg_novelty_score=0,
                improvement_trend="insufficient_data",
                peak_performance_session="",
                performance_consistency=0,
            )

        # Calculate averages
        avg_ideas_per_minute = statistics.mean(
            s.ideas_per_minute for s in recent_sessions
        )
        avg_diversity_score = statistics.mean(
            s.diversity_score for s in recent_sessions
        )
        avg_novelty_score = statistics.mean(s.novelty_score for s in recent_sessions)

        # Analyze trend direction
        session_dates = [
            (datetime.fromisoformat(s.timestamp.replace("Z", "+00:00")), s)
            for s in recent_sessions
        ]
        session_dates.sort()

        first_half = session_dates[: len(session_dates) // 2]
        second_half = session_dates[len(session_dates) // 2 :]

        first_avg = statistics.mean(s[1].ideas_per_minute for s in first_half)
        second_avg = statistics.mean(s[1].ideas_per_minute for s in second_half)

        if second_avg > first_avg * 1.1:
            trend = "improving"
        elif second_avg < first_avg * 0.9:
            trend = "declining"
        else:
            trend = "stable"

        # Find peak performance session
        peak_session = max(
            recent_sessions, key=lambda s: s.ideas_per_minute * s.diversity_score
        )

        # Calculate consistency
        velocities = [s.ideas_per_minute for s in recent_sessions]
        consistency = (
            1.0 - (statistics.stdev(velocities) / statistics.mean(velocities))
            if len(velocities) > 1
            else 1.0
        )
        consistency = max(0.0, min(1.0, consistency))

        return IdeaflowTrends(
            sessions_analyzed=len(recent_sessions),
            avg_ideas_per_minute=avg_ideas_per_minute,
            avg_diversity_score=avg_diversity_score,
            avg_novelty_score=avg_novelty_score,
            improvement_trend=trend,
            peak_performance_session=peak_session.session_id,
            performance_consistency=consistency,
        )

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary for reporting."""

        if not self.session_history:
            return {"total_sessions": 0, "performance_available": False}

        recent_trends = self.analyze_performance_trends(30)
        all_sessions = self.session_history

        return {
            "total_sessions": len(all_sessions),
            "performance_available": True,
            "current_performance": {
                "avg_ideas_per_minute": recent_trends.avg_ideas_per_minute,
                "avg_diversity_score": recent_trends.avg_diversity_score,
                "avg_novelty_score": recent_trends.avg_novelty_score,
                "performance_consistency": recent_trends.performance_consistency,
            },
            "trends": {
                "improvement_trend": recent_trends.improvement_trend,
                "sessions_analyzed": recent_trends.sessions_analyzed,
                "peak_session": recent_trends.peak_performance_session,
            },
            "benchmarks": {
                "target_ideas_per_minute": 3.0,  # Target from Ideaflow methodology
                "target_diversity_score": 0.7,
                "target_novelty_score": 0.6,
            },
        }


# Singleton instance for global access
_ideaflow_tracker = None


def get_ideaflow_tracker() -> IdeaflowMetricsTracker:
    """Get singleton ideaflow metrics tracker."""
    global _ideaflow_tracker
    if _ideaflow_tracker is None:
        _ideaflow_tracker = IdeaflowMetricsTracker()
    return _ideaflow_tracker
