"""
METIS Learning Loop Architecture
Captures user decisions and continuously improves consultant selection and system performance
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict, deque

from src.engine.flywheel.cache.flywheel_cache_system import get_flywheel_cache
from src.core.audit_trail import get_audit_manager, AuditEventType, AuditSeverity

logger = logging.getLogger(__name__)


class LearningEventType(Enum):
    """Types of learning events"""

    USER_SELECTION = "user_selection"
    SATISFACTION_FEEDBACK = "satisfaction_feedback"
    CRITIQUE_REQUEST = "critique_request"
    ARBITRATION_REQUEST = "arbitration_request"
    ENGAGEMENT_COMPLETION = "engagement_completion"


@dataclass
class LearningEvent:
    """Structured learning event for the feedback loop"""

    event_id: str
    event_type: LearningEventType
    engagement_id: str
    user_id: Optional[str]
    timestamp: datetime
    query: str
    context: Dict[str, Any]

    # User decision data
    chosen_consultant: Optional[str] = None
    available_consultants: List[str] = None
    consultant_rankings: Dict[str, float] = None

    # User feedback
    satisfaction_score: Optional[float] = None  # 0.0 to 1.0
    feedback_text: Optional[str] = None
    time_to_decision_seconds: Optional[float] = None

    # System state
    system_recommendations: List[str] = None
    prediction_confidence: Optional[float] = None
    cache_hit: bool = False
    processing_time_ms: float = 0.0


@dataclass
class ConsultantPerformanceMetrics:
    """Performance metrics for individual consultants"""

    consultant_id: str
    total_selections: int
    avg_satisfaction: float
    satisfaction_trend: float  # positive = improving, negative = declining
    query_type_specialization: Dict[str, float]
    complexity_effectiveness: Dict[str, float]
    recent_feedback_scores: deque
    last_updated: datetime


class LearningLoop:
    """
    Core learning system that captures user decisions and improves consultant selection
    """

    def __init__(self):
        self.learning_events: List[LearningEvent] = []
        self.consultant_metrics: Dict[str, ConsultantPerformanceMetrics] = {}
        self.query_patterns: Dict[str, List[str]] = defaultdict(list)
        self.user_preferences: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # Learning hyperparameters
        self.learning_rate = 0.1
        self.recency_weight_decay = 0.95
        self.min_interactions_threshold = 5
        self.confidence_threshold = 0.7

        # Performance tracking
        self.learning_accuracy_history = deque(maxlen=100)
        self.system_improvement_metrics = {
            "consultant_selection_accuracy": 0.0,
            "user_satisfaction_trend": 0.0,
            "prediction_confidence_trend": 0.0,
            "cache_efficiency": 0.0,
        }

    async def record_learning_event(self, event: LearningEvent):
        """Record a learning event and trigger learning updates with reflection pattern validation"""
        # AGENTIC PATTERN: Apply iterative refinement with producer-critic architecture
        refined_event = await self.iterative_learning_refinement(event)
        self.learning_events.append(refined_event)

        # Log to audit trail
        audit_manager = await get_audit_manager()
        await audit_manager.log_event(
            event_type=AuditEventType.USER_INTERACTION,
            severity=AuditSeverity.LOW,
            user_id=event.user_id,
            engagement_id=event.engagement_id,
            action_performed=f"learning_event_{event.event_type.value}",
            event_description=f"Learning event: {event.event_type.value}",
            metadata={
                "event_id": event.event_id,
                "chosen_consultant": event.chosen_consultant,
                "satisfaction_score": event.satisfaction_score,
                "processing_time_ms": event.processing_time_ms,
            },
        )

        # Update flywheel cache with user decision
        if event.chosen_consultant and event.satisfaction_score is not None:
            flywheel_cache = await get_flywheel_cache()
            await flywheel_cache.record_user_decision(
                query=event.query,
                context=event.context,
                chosen_consultant=event.chosen_consultant,
                user_satisfaction=event.satisfaction_score,
            )

        # Trigger learning updates
        await self._update_consultant_metrics(event)
        await self._update_query_patterns(event)
        await self._update_user_preferences(event)

        # Evaluate learning accuracy
        if len(self.learning_events) % 10 == 0:  # Every 10 events
            await self._evaluate_learning_accuracy()

    async def get_improved_consultant_recommendations(
        self, query: str, context: Dict[str, Any], user_id: Optional[str] = None
    ) -> List[Tuple[str, float, str]]:
        """
        Get consultant recommendations improved by learning loop
        Returns: [(consultant_id, confidence_score, reasoning)]
        """
        recommendations = []

        # Get base predictions from flywheel cache
        flywheel_cache = await get_flywheel_cache()
        base_predictions = flywheel_cache.predict_optimal_consultant(query, context)

        # Enhance with learning loop insights
        query_type = self._classify_query(query)
        complexity = self._estimate_complexity(query, context)

        for consultant_id, base_score in base_predictions:
            if consultant_id in self.consultant_metrics:
                metrics = self.consultant_metrics[consultant_id]

                # Calculate enhanced score
                enhanced_score = self._calculate_enhanced_score(
                    base_score=base_score,
                    metrics=metrics,
                    query_type=query_type,
                    complexity=complexity,
                    user_id=user_id,
                )

                # Generate reasoning
                reasoning = self._generate_recommendation_reasoning(
                    consultant_id, metrics, query_type, complexity, enhanced_score
                )

                recommendations.append((consultant_id, enhanced_score, reasoning))

        # Sort by enhanced score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:3]  # Top 3 recommendations

    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights about the learning system performance"""
        return {
            "total_learning_events": len(self.learning_events),
            "consultant_metrics_available": len(self.consultant_metrics),
            "unique_query_patterns": len(self.query_patterns),
            "user_profiles_learned": len(self.user_preferences),
            "system_improvement_metrics": self.system_improvement_metrics,
            "recent_learning_accuracy": (
                np.mean(self.learning_accuracy_history)
                if self.learning_accuracy_history
                else 0.0
            ),
            "learning_system_health": {
                "data_sufficiency": len(self.learning_events)
                >= self.min_interactions_threshold,
                "prediction_confidence": self._calculate_overall_confidence(),
                "improvement_trend": self._calculate_improvement_trend(),
            },
        }

    async def _update_consultant_metrics(self, event: LearningEvent):
        """Update consultant performance metrics based on learning event"""
        if not event.chosen_consultant or event.satisfaction_score is None:
            return

        consultant_id = event.chosen_consultant

        # Initialize metrics if not exists
        if consultant_id not in self.consultant_metrics:
            self.consultant_metrics[consultant_id] = ConsultantPerformanceMetrics(
                consultant_id=consultant_id,
                total_selections=0,
                avg_satisfaction=0.0,
                satisfaction_trend=0.0,
                query_type_specialization={},
                complexity_effectiveness={},
                recent_feedback_scores=deque(maxlen=20),
                last_updated=datetime.utcnow(),
            )

        metrics = self.consultant_metrics[consultant_id]

        # Update basic metrics
        metrics.total_selections += 1
        metrics.recent_feedback_scores.append(event.satisfaction_score)

        # Update average satisfaction with recency weighting
        current_avg = metrics.avg_satisfaction
        metrics.avg_satisfaction = (
            current_avg * self.recency_weight_decay
            + event.satisfaction_score * (1 - self.recency_weight_decay)
        )

        # Update satisfaction trend
        if len(metrics.recent_feedback_scores) >= 5:
            recent_scores = list(metrics.recent_feedback_scores)
            x = np.arange(len(recent_scores))
            trend = np.polyfit(x, recent_scores, 1)[0]  # Linear trend slope
            metrics.satisfaction_trend = trend

        # Update query type specialization
        query_type = self._classify_query(event.query)
        if query_type not in metrics.query_type_specialization:
            metrics.query_type_specialization[query_type] = 0.0

        metrics.query_type_specialization[
            query_type
        ] = metrics.query_type_specialization[
            query_type
        ] * self.recency_weight_decay + event.satisfaction_score * (
            1 - self.recency_weight_decay
        )

        # Update complexity effectiveness
        complexity_bin = self._get_complexity_bin(event.query, event.context)
        if complexity_bin not in metrics.complexity_effectiveness:
            metrics.complexity_effectiveness[complexity_bin] = 0.0

        metrics.complexity_effectiveness[
            complexity_bin
        ] = metrics.complexity_effectiveness[
            complexity_bin
        ] * self.recency_weight_decay + event.satisfaction_score * (
            1 - self.recency_weight_decay
        )

        metrics.last_updated = datetime.utcnow()

    async def _update_query_patterns(self, event: LearningEvent):
        """Update query pattern recognition"""
        if not event.chosen_consultant:
            return

        query_signature = self._create_query_signature(event.query)
        self.query_patterns[query_signature].append(event.chosen_consultant)

        # Keep only recent patterns (last 100 per signature)
        if len(self.query_patterns[query_signature]) > 100:
            self.query_patterns[query_signature] = self.query_patterns[query_signature][
                -100:
            ]

    async def _update_user_preferences(self, event: LearningEvent):
        """Update individual user preference profiles"""
        if not event.user_id or not event.chosen_consultant:
            return

        user_prefs = self.user_preferences[event.user_id]

        # Track consultant preferences
        if "consultant_preferences" not in user_prefs:
            user_prefs["consultant_preferences"] = defaultdict(float)

        user_prefs["consultant_preferences"][event.chosen_consultant] += 1

        # Track query type preferences
        query_type = self._classify_query(event.query)
        if "query_type_patterns" not in user_prefs:
            user_prefs["query_type_patterns"] = defaultdict(list)

        user_prefs["query_type_patterns"][query_type].append(
            {
                "consultant": event.chosen_consultant,
                "satisfaction": event.satisfaction_score,
                "timestamp": event.timestamp.isoformat(),
            }
        )

    async def _evaluate_learning_accuracy(self):
        """Evaluate the accuracy of learning-based predictions"""
        if len(self.learning_events) < self.min_interactions_threshold:
            return

        # Take last 20 events for evaluation
        recent_events = self.learning_events[-20:]
        correct_predictions = 0

        for event in recent_events:
            if not event.chosen_consultant or not event.satisfaction_score:
                continue

            # Get what the system would have predicted
            try:
                recommendations = await self.get_improved_consultant_recommendations(
                    query=event.query, context=event.context, user_id=event.user_id
                )

                if recommendations and recommendations[0][0] == event.chosen_consultant:
                    # Check if the prediction was good (high satisfaction)
                    if event.satisfaction_score >= 0.7:
                        correct_predictions += 1

            except Exception as e:
                logger.warning(f"Error evaluating learning accuracy: {e}")
                continue

        accuracy = correct_predictions / len(recent_events) if recent_events else 0.0
        self.learning_accuracy_history.append(accuracy)

        # Update system improvement metrics
        self.system_improvement_metrics["consultant_selection_accuracy"] = accuracy

    def _calculate_enhanced_score(
        self,
        base_score: float,
        metrics: ConsultantPerformanceMetrics,
        query_type: str,
        complexity: str,
        user_id: Optional[str],
    ) -> float:
        """Calculate enhanced recommendation score"""
        enhanced_score = base_score

        # Boost based on consultant's specialization in this query type
        if query_type in metrics.query_type_specialization:
            specialization_boost = metrics.query_type_specialization[query_type] * 0.3
            enhanced_score += specialization_boost

        # Boost based on complexity effectiveness
        if complexity in metrics.complexity_effectiveness:
            complexity_boost = metrics.complexity_effectiveness[complexity] * 0.2
            enhanced_score += complexity_boost

        # Boost based on recent performance trend
        trend_boost = metrics.satisfaction_trend * 0.1
        enhanced_score += trend_boost

        # User-specific boost
        if user_id and user_id in self.user_preferences:
            user_prefs = self.user_preferences[user_id]
            if "consultant_preferences" in user_prefs:
                pref_count = user_prefs["consultant_preferences"].get(
                    metrics.consultant_id, 0
                )
                if pref_count > 0:
                    user_boost = min(pref_count / 10.0, 0.2)  # Max 20% boost
                    enhanced_score += user_boost

        return min(enhanced_score, 1.0)  # Cap at 1.0

    def _generate_recommendation_reasoning(
        self,
        consultant_id: str,
        metrics: ConsultantPerformanceMetrics,
        query_type: str,
        complexity: str,
        score: float,
    ) -> str:
        """Generate human-readable reasoning for recommendation"""
        reasons = []

        # Base performance
        if metrics.avg_satisfaction >= 0.8:
            reasons.append("consistently high user satisfaction")
        elif metrics.avg_satisfaction >= 0.6:
            reasons.append("solid track record")

        # Specialization
        if query_type in metrics.query_type_specialization:
            spec_score = metrics.query_type_specialization[query_type]
            if spec_score >= 0.8:
                reasons.append(f"strong specialization in {query_type} queries")
            elif spec_score >= 0.6:
                reasons.append(f"good experience with {query_type} queries")

        # Trending performance
        if metrics.satisfaction_trend > 0.05:
            reasons.append("improving performance trend")
        elif metrics.satisfaction_trend < -0.05:
            reasons.append("recent performance decline")

        # Complexity handling
        if complexity in metrics.complexity_effectiveness:
            complexity_score = metrics.complexity_effectiveness[complexity]
            if complexity_score >= 0.8:
                reasons.append(f"excellent at handling {complexity} complexity")

        if not reasons:
            reasons.append("balanced overall performance")

        return f"Recommended ({score:.2f}): " + ", ".join(reasons)

    def _classify_query(self, query: str) -> str:
        """Classify query for learning purposes"""
        query_lower = query.lower()

        if any(word in query_lower for word in ["strategy", "strategic", "market"]):
            return "strategic"
        elif any(
            word in query_lower for word in ["technical", "implementation", "system"]
        ):
            return "technical"
        elif any(word in query_lower for word in ["financial", "budget", "cost"]):
            return "financial"
        elif any(
            word in query_lower for word in ["process", "operational", "efficiency"]
        ):
            return "operational"
        elif any(word in query_lower for word in ["innovation", "creative", "new"]):
            return "innovation"
        else:
            return "general"

    def _estimate_complexity(self, query: str, context: Dict[str, Any]) -> float:
        """Estimate query complexity (0.0 to 1.0)"""
        complexity = 0.0
        complexity += min(len(query) / 500, 0.4)
        complexity += min(len(context) / 10, 0.3)

        complex_indicators = ["integrate", "optimize", "transform", "synthesize"]
        complexity += min(
            sum(1 for word in complex_indicators if word in query.lower())
            / len(complex_indicators),
            0.3,
        )

        return complexity

    def _get_complexity_bin(self, query: str, context: Dict[str, Any]) -> str:
        """Get complexity bin for metrics tracking"""
        complexity = self._estimate_complexity(query, context)
        if complexity <= 0.3:
            return "low"
        elif complexity <= 0.7:
            return "medium"
        else:
            return "high"

    def _create_query_signature(self, query: str) -> str:
        """Create a signature for similar queries"""
        # Simple approach: use key words
        words = query.lower().split()
        key_words = [
            word
            for word in words
            if len(word) > 3 and word not in ["the", "and", "for", "with"]
        ]
        return "_".join(sorted(key_words[:5]))  # Top 5 key words

    def _calculate_overall_confidence(self) -> float:
        """Calculate overall prediction confidence"""
        if not self.consultant_metrics:
            return 0.0

        confidences = []
        for metrics in self.consultant_metrics.values():
            if metrics.total_selections >= self.min_interactions_threshold:
                confidence = min(metrics.avg_satisfaction, 1.0)
                confidences.append(confidence)

        return np.mean(confidences) if confidences else 0.0

    def _calculate_improvement_trend(self) -> float:
        """Calculate overall system improvement trend"""
        if len(self.learning_accuracy_history) < 5:
            return 0.0

        recent_accuracy = list(self.learning_accuracy_history)
        x = np.arange(len(recent_accuracy))
        trend = np.polyfit(x, recent_accuracy, 1)[0]
        return trend

    # AGENTIC DESIGN PATTERN: Reflection Pattern (Producer-Critic Architecture)
    async def validate_learning_quality_critic(
        self, event: LearningEvent
    ) -> Dict[str, Any]:
        """
        Critic agent that validates learning data quality using reflection patterns.
        Implements separate validation agent for quality assurance.
        """
        quality_assessment = {
            "data_validity": 0.0,
            "pattern_coherence": 0.0,
            "bias_risk": 0.0,
            "confidence": 0.0,
            "validation_passed": False,
            "critic_feedback": [],
        }

        # Data validity checks
        if event.chosen_consultant and event.satisfaction_score is not None:
            quality_assessment["data_validity"] = 1.0
        elif event.chosen_consultant:
            quality_assessment["data_validity"] = 0.7
            quality_assessment["critic_feedback"].append("Missing satisfaction score")
        else:
            quality_assessment["data_validity"] = 0.3
            quality_assessment["critic_feedback"].append("Missing consultant selection")

        # Pattern coherence (check against historical patterns)
        if len(self.learning_events) >= 5:
            query_type = self._classify_query(event.query)
            similar_events = [
                e
                for e in self.learning_events[-20:]
                if self._classify_query(e.query) == query_type
            ]

            if similar_events:
                avg_satisfaction = np.mean(
                    [
                        e.satisfaction_score
                        for e in similar_events
                        if e.satisfaction_score is not None
                    ]
                )
                if event.satisfaction_score is not None:
                    deviation = abs(event.satisfaction_score - avg_satisfaction)
                    quality_assessment["pattern_coherence"] = max(0.0, 1.0 - deviation)

                    if deviation > 0.5:
                        quality_assessment["critic_feedback"].append(
                            f"Satisfaction score deviates significantly from pattern (avg: {avg_satisfaction:.2f})"
                        )

        # Bias risk assessment
        if event.user_id and event.chosen_consultant:
            user_prefs = self.user_preferences.get(event.user_id, {})
            consultant_prefs = user_prefs.get("consultant_preferences", {})
            total_selections = sum(consultant_prefs.values())

            if total_selections > 10:
                consultant_ratio = (
                    consultant_prefs.get(event.chosen_consultant, 0) / total_selections
                )
                if consultant_ratio > 0.7:
                    quality_assessment["bias_risk"] = 1.0
                    quality_assessment["critic_feedback"].append(
                        "High user bias toward single consultant"
                    )
                elif consultant_ratio > 0.5:
                    quality_assessment["bias_risk"] = 0.6
                    quality_assessment["critic_feedback"].append(
                        "Moderate user bias detected"
                    )
                else:
                    quality_assessment["bias_risk"] = 0.0

        # Overall confidence calculation
        quality_assessment["confidence"] = (
            quality_assessment["data_validity"] * 0.4
            + quality_assessment["pattern_coherence"] * 0.3
            + (1.0 - quality_assessment["bias_risk"]) * 0.3
        )

        # Validation decision
        quality_assessment["validation_passed"] = (
            quality_assessment["confidence"] >= 0.7
            and quality_assessment["bias_risk"] < 0.8
        )

        logger.info(
            f"Learning quality critic assessment: {quality_assessment['confidence']:.2f} confidence, "
            f"{'PASSED' if quality_assessment['validation_passed'] else 'FAILED'}"
        )

        return quality_assessment

    async def iterative_learning_refinement(
        self, event: LearningEvent, max_iterations: int = 3
    ) -> LearningEvent:
        """
        Implements iterative refinement cycles for learning data improvement.
        Uses producer-critic feedback loops for self-correction.
        """
        refined_event = event

        for iteration in range(max_iterations):
            # Critic evaluation
            quality_assessment = await self.validate_learning_quality_critic(
                refined_event
            )

            # Break if quality is sufficient
            if quality_assessment["validation_passed"]:
                logger.info(
                    f"Learning refinement converged after {iteration + 1} iterations"
                )
                break

            # Producer improvements based on critic feedback
            if iteration < max_iterations - 1:  # Don't refine on last iteration
                refined_event = await self._apply_critic_feedback(
                    refined_event, quality_assessment
                )

        return refined_event

    async def _apply_critic_feedback(
        self, event: LearningEvent, quality_assessment: Dict[str, Any]
    ) -> LearningEvent:
        """Apply critic feedback to improve learning event quality"""
        improved_event = event

        # Adjust satisfaction score if pattern deviation is high
        if "deviates significantly from pattern" in str(
            quality_assessment.get("critic_feedback", [])
        ):
            if event.satisfaction_score is not None:
                # Conservative adjustment toward historical mean
                query_type = self._classify_query(event.query)
                similar_events = [
                    e
                    for e in self.learning_events[-20:]
                    if self._classify_query(e.query) == query_type
                    and e.satisfaction_score is not None
                ]

                if similar_events:
                    avg_satisfaction = np.mean(
                        [e.satisfaction_score for e in similar_events]
                    )
                    # Weighted average: 70% original, 30% historical mean
                    adjusted_score = (
                        0.7 * event.satisfaction_score + 0.3 * avg_satisfaction
                    )
                    improved_event.satisfaction_score = max(
                        0.0, min(1.0, adjusted_score)
                    )

                    logger.info(
                        f"Applied pattern-based satisfaction adjustment: "
                        f"{event.satisfaction_score:.2f} → {improved_event.satisfaction_score:.2f}"
                    )

        return improved_event


# Global learning loop instance
_learning_loop: Optional[LearningLoop] = None


async def get_learning_loop() -> LearningLoop:
    """Get or create global learning loop instance"""
    global _learning_loop

    if _learning_loop is None:
        _learning_loop = LearningLoop()

    return _learning_loop
