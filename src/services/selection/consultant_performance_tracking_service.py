"""
Consultant Performance Tracking Service - SPRINT 1 Implementation
=================================================================

Part of Selection Services Cluster - Specialized service for tracking consultant selection,
performance, and effectiveness across the LOLLA cognitive platform.

Key Features:
- Consultant selection frequency tracking with diversity monitoring
- Performance analytics across domains and frameworks
- Chemistry score tracking and optimization insights
- Selection bias detection and diversity recommendations
- Consultant effectiveness measurement and trending

Integrates with:
- ContextualLollapalozzaEngine for selection tracking
- CognitiveChemistryEngine for chemistry score monitoring
- NWayPatternSelectionService for pattern-consultant correlation analysis

SPRINT 1 TARGET: Address consultant stagnation (mckinsey_strategist 73% selection rate)
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, Counter
from enum import Enum
import statistics


class ConsultantPerformanceMetric(Enum):
    """Types of consultant performance metrics"""

    SELECTION_FREQUENCY = "selection_frequency"
    CHEMISTRY_SCORE = "chemistry_score"
    DOMAIN_EFFECTIVENESS = "domain_effectiveness"
    PATTERN_CORRELATION = "pattern_correlation"
    DIVERSITY_CONTRIBUTION = "diversity_contribution"
    ANALYSIS_QUALITY = "analysis_quality"
    USER_SATISFACTION = "user_satisfaction"


@dataclass
class ConsultantSelectionRecord:
    """Individual consultant selection record"""

    consultant_id: str
    selection_timestamp: datetime
    framework_type: str
    domain: str
    complexity: str
    chemistry_score: float
    selected_patterns: List[str]
    team_composition: List[str]
    context: Dict[str, Any]
    performance_score: Optional[float] = None
    user_feedback: Optional[Dict[str, Any]] = None


@dataclass
class ConsultantPerformanceAnalysis:
    """Comprehensive consultant performance analysis"""

    consultant_id: str
    analysis_period: str
    total_selections: int
    selection_rate: float
    diversity_score: float
    average_chemistry_score: float
    domain_distribution: Dict[str, int]
    framework_distribution: Dict[str, int]
    performance_trend: str
    effectiveness_rating: str
    recommendations: List[str]
    analysis_timestamp: datetime


class ConsultantPerformanceTrackingService:
    """
    Comprehensive service for tracking and analyzing consultant performance across the platform.

    SPRINT 1 Focus: Reduce consultant selection bias and improve diversity
    Target: Reduce mckinsey_strategist from 73% to <40% selection rate
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Performance tracking storage
        self.selection_history: List[ConsultantSelectionRecord] = []
        self.consultant_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.domain_performance: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.chemistry_tracking: Dict[str, List[float]] = defaultdict(list)

        # Performance analytics cache
        self.analytics_cache: Dict[str, Any] = {}
        self.cache_timestamp: Dict[str, datetime] = {}
        self.cache_ttl_minutes = 15

        # Configuration
        self.config = {
            "diversity_target_threshold": 0.40,  # 40% max selection rate per consultant
            "performance_window_days": 30,
            "chemistry_target_threshold": 0.75,
            "min_selections_for_analysis": 5,
            "alert_cooldown_minutes": 60,
            "trend_analysis_window": 20,  # Last 20 selections
        }

        # Alert tracking
        self.active_alerts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.alert_history: List[Dict[str, Any]] = []

        # Known consultant types and their ideal domains
        self.consultant_domains = {
            "mckinsey_strategist": ["strategic", "business", "management"],
            "financial_analyst": ["financial", "investment", "economics"],
            "technical_architect": ["technical", "system", "engineering"],
            "operations_expert": ["operational", "process", "logistics"],
            "innovation_consultant": ["innovation", "creative", "design"],
            "risk_specialist": ["risk", "compliance", "security"],
            "data_scientist": ["data", "analytics", "quantitative"],
            "behavioral_analyst": ["behavioral", "psychological", "social"],
        }

        self.logger.info(
            "🎯 Consultant Performance Tracking Service initialized - SPRINT 1 Active"
        )

    async def record_consultant_selection(
        self,
        consultant_id: str,
        framework_type: str,
        domain: str,
        complexity: str,
        chemistry_score: float,
        selected_patterns: List[str],
        team_composition: List[str],
        context: Dict[str, Any],
    ) -> ConsultantSelectionRecord:
        """
        Record a consultant selection with full context for performance tracking.

        SPRINT 1: Core method for tracking consultant usage and identifying bias patterns.
        """
        try:
            # Create selection record
            selection_record = ConsultantSelectionRecord(
                consultant_id=consultant_id,
                selection_timestamp=datetime.utcnow(),
                framework_type=framework_type,
                domain=domain,
                complexity=complexity,
                chemistry_score=chemistry_score,
                selected_patterns=selected_patterns,
                team_composition=team_composition,
                context=context,
            )

            # Store the record
            self.selection_history.append(selection_record)

            # Update tracking metrics
            await self._update_consultant_metrics(selection_record)

            # Check for diversity alerts
            await self._check_diversity_alerts(consultant_id)

            # Update chemistry tracking
            self.chemistry_tracking[consultant_id].append(chemistry_score)

            # Clean up old records periodically
            await self._cleanup_old_records()

            # Invalidate relevant cache
            self._invalidate_cache(consultant_id)

            self.logger.debug(
                f"📊 Recorded selection: {consultant_id} for {framework_type} (chemistry: {chemistry_score:.3f})"
            )
            return selection_record

        except Exception as e:
            self.logger.error(f"❌ Failed to record consultant selection: {e}")
            raise

    async def get_consultant_performance_analysis(
        self, consultant_id: str, analysis_period_days: int = 30
    ) -> ConsultantPerformanceAnalysis:
        """
        Get comprehensive performance analysis for a specific consultant.

        SPRINT 1: Critical method for identifying overused consultants and performance patterns.
        """
        try:
            # Check cache first
            cache_key = f"analysis_{consultant_id}_{analysis_period_days}"
            if self._is_cache_valid(cache_key):
                return self.analytics_cache[cache_key]

            # Calculate analysis window
            window_start = datetime.utcnow() - timedelta(days=analysis_period_days)

            # Get consultant selections in window
            consultant_selections = [
                record
                for record in self.selection_history
                if (
                    record.consultant_id == consultant_id
                    and record.selection_timestamp >= window_start
                )
            ]

            if not consultant_selections:
                return self._create_empty_analysis(consultant_id, analysis_period_days)

            # Calculate total selections in period for diversity rate
            total_selections_in_period = len(
                [
                    record
                    for record in self.selection_history
                    if record.selection_timestamp >= window_start
                ]
            )

            # Calculate metrics
            total_selections = len(consultant_selections)
            selection_rate = total_selections / max(total_selections_in_period, 1)

            # Diversity score (inverse of selection rate - higher is better)
            diversity_score = max(
                0.0, 1.0 - (selection_rate / self.config["diversity_target_threshold"])
            )

            # Average chemistry score
            chemistry_scores = [r.chemistry_score for r in consultant_selections]
            average_chemistry_score = (
                statistics.mean(chemistry_scores) if chemistry_scores else 0.0
            )

            # Domain and framework distribution
            domain_distribution = Counter(r.domain for r in consultant_selections)
            framework_distribution = Counter(
                r.framework_type for r in consultant_selections
            )

            # Performance trend analysis
            performance_trend = self._calculate_performance_trend(consultant_selections)

            # Effectiveness rating
            effectiveness_rating = self._calculate_effectiveness_rating(
                selection_rate, average_chemistry_score, diversity_score
            )

            # Generate recommendations
            recommendations = await self._generate_consultant_recommendations(
                consultant_id,
                selection_rate,
                average_chemistry_score,
                diversity_score,
                domain_distribution,
                consultant_selections,
            )

            # Create analysis
            analysis = ConsultantPerformanceAnalysis(
                consultant_id=consultant_id,
                analysis_period=f"{analysis_period_days} days",
                total_selections=total_selections,
                selection_rate=selection_rate,
                diversity_score=diversity_score,
                average_chemistry_score=average_chemistry_score,
                domain_distribution=dict(domain_distribution),
                framework_distribution=dict(framework_distribution),
                performance_trend=performance_trend,
                effectiveness_rating=effectiveness_rating,
                recommendations=recommendations,
                analysis_timestamp=datetime.utcnow(),
            )

            # Cache the result
            self.analytics_cache[cache_key] = analysis
            self.cache_timestamp[cache_key] = datetime.utcnow()

            return analysis

        except Exception as e:
            self.logger.error(
                f"❌ Failed to analyze consultant performance for {consultant_id}: {e}"
            )
            return self._create_error_analysis(consultant_id, str(e))

    async def get_diversity_dashboard(self) -> Dict[str, Any]:
        """
        Get comprehensive diversity dashboard showing consultant selection patterns.

        SPRINT 1: Essential for monitoring progress toward diversity targets.
        """
        try:
            # Calculate analysis window (last 30 days)
            window_start = datetime.utcnow() - timedelta(days=30)
            recent_selections = [
                record
                for record in self.selection_history
                if record.selection_timestamp >= window_start
            ]

            if not recent_selections:
                return {
                    "error": "No recent consultant selections found",
                    "recommendations": ["Increase platform usage"],
                }

            total_selections = len(recent_selections)

            # Calculate selection rates by consultant
            consultant_counts = Counter(r.consultant_id for r in recent_selections)
            consultant_rates = {
                consultant: count / total_selections
                for consultant, count in consultant_counts.items()
            }

            # Identify diversity violations
            diversity_violations = [
                (consultant, rate)
                for consultant, rate in consultant_rates.items()
                if rate > self.config["diversity_target_threshold"]
            ]

            # Calculate diversity metrics
            diversity_index = self._calculate_diversity_index(consultant_rates)

            # Domain distribution analysis
            domain_consultants = defaultdict(list)
            for record in recent_selections:
                domain_consultants[record.domain].append(record.consultant_id)

            domain_diversity = {}
            for domain, consultants in domain_consultants.items():
                unique_consultants = len(set(consultants))
                total_domain_selections = len(consultants)
                domain_diversity[domain] = {
                    "unique_consultants": unique_consultants,
                    "total_selections": total_domain_selections,
                    "diversity_ratio": unique_consultants
                    / max(total_domain_selections, 1),
                }

            # Chemistry score analysis
            chemistry_by_consultant = defaultdict(list)
            for record in recent_selections:
                chemistry_by_consultant[record.consultant_id].append(
                    record.chemistry_score
                )

            chemistry_analysis = {}
            for consultant, scores in chemistry_by_consultant.items():
                chemistry_analysis[consultant] = {
                    "average_chemistry": statistics.mean(scores),
                    "chemistry_trend": self._calculate_simple_trend(
                        scores[-10:]
                    ),  # Last 10 scores
                    "selections_count": len(scores),
                }

            # Generate dashboard recommendations
            dashboard_recommendations = await self._generate_diversity_recommendations(
                consultant_rates,
                diversity_violations,
                domain_diversity,
                chemistry_analysis,
            )

            # Performance alerts summary
            active_diversity_alerts = [
                alert
                for alerts in self.active_alerts.values()
                for alert in alerts
                if alert.get("type") == "diversity_violation"
            ]

            dashboard = {
                "diversity_overview": {
                    "analysis_period": "30 days",
                    "total_selections": total_selections,
                    "unique_consultants": len(consultant_counts),
                    "diversity_index": diversity_index,
                    "diversity_target": self.config["diversity_target_threshold"],
                    "diversity_violations": len(diversity_violations),
                },
                "consultant_selection_rates": consultant_rates,
                "diversity_violations": [
                    {
                        "consultant": c,
                        "rate": r,
                        "target": self.config["diversity_target_threshold"],
                    }
                    for c, r in diversity_violations
                ],
                "domain_diversity_analysis": domain_diversity,
                "chemistry_performance": chemistry_analysis,
                "active_alerts": len(active_diversity_alerts),
                "recommendations": dashboard_recommendations,
                "dashboard_generated": datetime.utcnow().isoformat(),
            }

            return dashboard

        except Exception as e:
            self.logger.error(f"❌ Failed to generate diversity dashboard: {e}")
            return {"error": str(e)}

    async def get_consultant_effectiveness_ranking(
        self, domain: Optional[str] = None, framework_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get ranked list of consultants by effectiveness for specific domain/framework.

        SPRINT 1: Helps identify best alternatives to overused consultants.
        """
        try:
            # Filter selections by criteria
            filtered_selections = self.selection_history

            if domain:
                filtered_selections = [
                    r for r in filtered_selections if r.domain.lower() == domain.lower()
                ]

            if framework_type:
                filtered_selections = [
                    r
                    for r in filtered_selections
                    if r.framework_type.lower() == framework_type.lower()
                ]

            # Group by consultant
            consultant_performance = defaultdict(list)
            for record in filtered_selections:
                consultant_performance[record.consultant_id].append(record)

            # Calculate effectiveness scores
            consultant_rankings = []

            for consultant_id, records in consultant_performance.items():
                if len(records) < self.config["min_selections_for_analysis"]:
                    continue  # Skip consultants with insufficient data

                # Calculate metrics
                chemistry_scores = [r.chemistry_score for r in records]
                avg_chemistry = statistics.mean(chemistry_scores)
                chemistry_consistency = 1 - (
                    statistics.stdev(chemistry_scores)
                    if len(chemistry_scores) > 1
                    else 0
                )

                # Selection frequency (recent 30 days)
                recent_window = datetime.utcnow() - timedelta(days=30)
                recent_selections = [
                    r for r in records if r.selection_timestamp >= recent_window
                ]
                selection_frequency = len(recent_selections)

                # Domain specialization score
                domain_specialization = self._calculate_domain_specialization(
                    consultant_id, records
                )

                # Diversity contribution (lower selection rate = higher diversity contribution)
                total_recent = len(
                    [
                        r
                        for r in self.selection_history
                        if r.selection_timestamp >= recent_window
                    ]
                )
                selection_rate = len(recent_selections) / max(total_recent, 1)
                diversity_contribution = 1 - min(
                    selection_rate / self.config["diversity_target_threshold"], 1.0
                )

                # Combined effectiveness score (weighted average)
                effectiveness_score = (
                    avg_chemistry * 0.35  # 35% chemistry performance
                    + chemistry_consistency * 0.20  # 20% consistency
                    + domain_specialization * 0.25  # 25% domain expertise
                    + diversity_contribution * 0.20  # 20% diversity contribution
                )

                consultant_rankings.append(
                    {
                        "consultant_id": consultant_id,
                        "effectiveness_score": effectiveness_score,
                        "average_chemistry": avg_chemistry,
                        "chemistry_consistency": chemistry_consistency,
                        "domain_specialization": domain_specialization,
                        "diversity_contribution": diversity_contribution,
                        "total_selections": len(records),
                        "recent_selections": selection_frequency,
                        "selection_rate": selection_rate,
                        "specialization_domains": list(
                            self.consultant_domains.get(consultant_id, [])
                        ),
                    }
                )

            # Sort by effectiveness score
            consultant_rankings.sort(
                key=lambda x: x["effectiveness_score"], reverse=True
            )

            # Add ranking positions
            for i, consultant in enumerate(consultant_rankings):
                consultant["rank"] = i + 1
                consultant["tier"] = (
                    "TOP"
                    if i < 3
                    else "HIGH" if i < 6 else "MEDIUM" if i < 10 else "LOW"
                )

            return consultant_rankings

        except Exception as e:
            self.logger.error(
                f"❌ Failed to generate consultant effectiveness ranking: {e}"
            )
            return []

    async def record_consultant_feedback(
        self,
        consultant_id: str,
        selection_timestamp: datetime,
        feedback_score: float,
        feedback_details: Dict[str, Any],
    ) -> bool:
        """
        Record user feedback for a specific consultant selection.

        SPRINT 1: Critical for measuring actual consultant effectiveness.
        """
        try:
            # Find the corresponding selection record
            matching_records = [
                record
                for record in self.selection_history
                if (
                    record.consultant_id == consultant_id
                    and abs(
                        (
                            record.selection_timestamp - selection_timestamp
                        ).total_seconds()
                    )
                    < 300
                )  # 5 minute window
            ]

            if not matching_records:
                self.logger.warning(
                    f"⚠️ No matching selection record found for feedback: {consultant_id}"
                )
                return False

            # Update the most recent matching record
            target_record = max(matching_records, key=lambda r: r.selection_timestamp)
            target_record.performance_score = feedback_score
            target_record.user_feedback = feedback_details

            # Update metrics
            self.consultant_metrics[consultant_id]["user_satisfaction"].append(
                feedback_score
            )

            # Invalidate cache
            self._invalidate_cache(consultant_id)

            self.logger.info(
                f"📊 Recorded feedback for {consultant_id}: {feedback_score:.2f}"
            )
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to record consultant feedback: {e}")
            return False

    async def get_optimization_recommendations(self) -> Dict[str, Any]:
        """
        Get comprehensive optimization recommendations for consultant selection.

        SPRINT 1: Core output for improving diversity and performance.
        """
        try:
            # Get current diversity dashboard
            diversity_data = await self.get_diversity_dashboard()

            # Get effectiveness rankings
            effectiveness_rankings = await self.get_consultant_effectiveness_ranking()

            # Identify optimization opportunities
            optimization_recommendations = {
                "critical_actions": [],
                "performance_improvements": [],
                "diversity_enhancements": [],
                "chemistry_optimizations": [],
                "implementation_steps": [],
            }

            # Critical diversity violations
            violations = diversity_data.get("diversity_violations", [])
            for violation in violations:
                consultant = violation["consultant"]
                rate = violation["rate"]
                optimization_recommendations["critical_actions"].append(
                    {
                        "priority": "HIGH",
                        "action": f"Reduce {consultant} selection rate from {rate:.1%} to <{self.config['diversity_target_threshold']:.0%}",
                        "impact": "diversity_improvement",
                        "implementation": f"Apply stronger diversity penalties for {consultant}",
                    }
                )

            # Performance improvement opportunities
            if effectiveness_rankings:
                underutilized_experts = [
                    r
                    for r in effectiveness_rankings
                    if r["effectiveness_score"] > 0.7 and r["selection_rate"] < 0.2
                ]

                for expert in underutilized_experts[:3]:  # Top 3 underutilized
                    optimization_recommendations["performance_improvements"].append(
                        {
                            "action": f"Increase utilization of {expert['consultant_id']} (effectiveness: {expert['effectiveness_score']:.2f})",
                            "current_rate": f"{expert['selection_rate']:.1%}",
                            "potential_domains": expert["specialization_domains"],
                            "impact": "performance_enhancement",
                        }
                    )

            # Chemistry optimization opportunities
            chemistry_data = diversity_data.get("chemistry_performance", {})
            low_chemistry_consultants = [
                (consultant, data)
                for consultant, data in chemistry_data.items()
                if data["average_chemistry"] < 0.7
            ]

            for consultant, data in low_chemistry_consultants:
                optimization_recommendations["chemistry_optimizations"].append(
                    {
                        "consultant": consultant,
                        "current_chemistry": data["average_chemistry"],
                        "target_chemistry": 0.75,
                        "recommendation": "Review consultant-pattern combinations and optimize team compositions",
                    }
                )

            # Implementation steps
            optimization_recommendations["implementation_steps"] = [
                "1. Apply diversity penalties to overused consultants immediately",
                "2. Boost selection probability for underutilized high-performers",
                "3. Review and optimize chemistry calculation for low-performing consultants",
                "4. Implement pattern-consultant matching improvements",
                "5. Monitor selection rates daily and adjust algorithms dynamically",
            ]

            # Summary metrics
            optimization_recommendations["summary"] = {
                "total_violations": len(violations),
                "underutilized_experts": len(
                    [
                        r
                        for r in effectiveness_rankings
                        if r["effectiveness_score"] > 0.7 and r["selection_rate"] < 0.2
                    ]
                ),
                "low_chemistry_consultants": len(low_chemistry_consultants),
                "optimization_potential": "HIGH" if len(violations) > 0 else "MEDIUM",
                "generated_timestamp": datetime.utcnow().isoformat(),
            }

            return optimization_recommendations

        except Exception as e:
            self.logger.error(
                f"❌ Failed to generate optimization recommendations: {e}"
            )
            return {"error": str(e)}

    async def _update_consultant_metrics(self, record: ConsultantSelectionRecord):
        """Update internal consultant metrics tracking"""
        consultant_id = record.consultant_id

        # Update selection frequency
        self.consultant_metrics[consultant_id]["selection_frequency"].append(1.0)

        # Update domain effectiveness
        domain_key = f"domain_{record.domain}"
        self.consultant_metrics[consultant_id][domain_key].append(
            record.chemistry_score
        )

        # Update pattern correlation
        for pattern in record.selected_patterns:
            pattern_key = f"pattern_{pattern}"
            self.consultant_metrics[consultant_id][pattern_key].append(
                record.chemistry_score
            )

    async def _check_diversity_alerts(self, consultant_id: str):
        """Check for and create diversity violation alerts"""
        try:
            # Calculate recent selection rate (last 30 days)
            window_start = datetime.utcnow() - timedelta(days=30)
            recent_selections = [
                record
                for record in self.selection_history
                if record.selection_timestamp >= window_start
            ]

            if len(recent_selections) < 10:  # Need minimum data for meaningful alerts
                return

            consultant_selections = [
                r for r in recent_selections if r.consultant_id == consultant_id
            ]
            selection_rate = len(consultant_selections) / len(recent_selections)

            # Check for diversity violation
            if selection_rate > self.config["diversity_target_threshold"]:
                # Check if already alerted recently
                if not self._is_alert_in_cooldown(consultant_id, "diversity_violation"):
                    alert = {
                        "alert_id": f"diversity_{consultant_id}_{datetime.utcnow().timestamp()}",
                        "type": "diversity_violation",
                        "consultant_id": consultant_id,
                        "selection_rate": selection_rate,
                        "threshold": self.config["diversity_target_threshold"],
                        "severity": "CRITICAL" if selection_rate > 0.6 else "HIGH",
                        "timestamp": datetime.utcnow().isoformat(),
                        "message": f"{consultant_id} selection rate {selection_rate:.1%} exceeds diversity target {self.config['diversity_target_threshold']:.0%}",
                    }

                    self.active_alerts[consultant_id].append(alert)
                    self.alert_history.append(alert.copy())

                    self.logger.warning(f"🚨 DIVERSITY ALERT: {alert['message']}")

        except Exception as e:
            self.logger.error(f"❌ Diversity alert check failed: {e}")

    def _is_alert_in_cooldown(self, consultant_id: str, alert_type: str) -> bool:
        """Check if alert is in cooldown period"""
        cooldown_threshold = datetime.utcnow() - timedelta(
            minutes=self.config["alert_cooldown_minutes"]
        )

        for alert in self.active_alerts.get(consultant_id, []):
            if (
                alert.get("type") == alert_type
                and datetime.fromisoformat(alert["timestamp"]) > cooldown_threshold
            ):
                return True

        return False

    async def _cleanup_old_records(self):
        """Clean up old selection records to manage memory"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(
                days=90
            )  # Keep 90 days of history

            original_count = len(self.selection_history)
            self.selection_history = [
                record
                for record in self.selection_history
                if record.selection_timestamp > cutoff_date
            ]

            cleaned_count = original_count - len(self.selection_history)
            if cleaned_count > 0:
                self.logger.debug(
                    f"🧹 Cleaned {cleaned_count} old consultant selection records"
                )

        except Exception as e:
            self.logger.error(f"❌ Record cleanup failed: {e}")

    def _invalidate_cache(self, consultant_id: str):
        """Invalidate cache entries for consultant"""
        keys_to_remove = [
            key
            for key in self.analytics_cache.keys()
            if consultant_id in key or key.startswith("dashboard")
        ]

        for key in keys_to_remove:
            self.analytics_cache.pop(key, None)
            self.cache_timestamp.pop(key, None)

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid"""
        if cache_key not in self.analytics_cache:
            return False

        cache_age = (
            datetime.utcnow() - self.cache_timestamp.get(cache_key, datetime.min)
        ).total_seconds() / 60
        return cache_age < self.cache_ttl_minutes

    def _calculate_performance_trend(
        self, records: List[ConsultantSelectionRecord]
    ) -> str:
        """Calculate performance trend from chemistry scores"""
        if len(records) < 3:
            return "insufficient_data"

        # Sort by timestamp and get chemistry scores
        sorted_records = sorted(records, key=lambda r: r.selection_timestamp)
        chemistry_scores = [r.chemistry_score for r in sorted_records]

        # Compare first half vs second half
        mid_point = len(chemistry_scores) // 2
        first_half_avg = statistics.mean(chemistry_scores[:mid_point])
        second_half_avg = statistics.mean(chemistry_scores[mid_point:])

        if second_half_avg > first_half_avg * 1.05:
            return "improving"
        elif second_half_avg < first_half_avg * 0.95:
            return "declining"
        else:
            return "stable"

    def _calculate_effectiveness_rating(
        self, selection_rate: float, chemistry_score: float, diversity_score: float
    ) -> str:
        """Calculate overall effectiveness rating"""
        # Penalize high selection rates, reward high chemistry and diversity
        if selection_rate > self.config["diversity_target_threshold"]:
            return "OVERUSED"
        elif chemistry_score >= 0.8 and diversity_score >= 0.7:
            return "EXCELLENT"
        elif chemistry_score >= 0.7 and diversity_score >= 0.5:
            return "GOOD"
        elif chemistry_score >= 0.6:
            return "AVERAGE"
        else:
            return "POOR"

    async def _generate_consultant_recommendations(
        self,
        consultant_id: str,
        selection_rate: float,
        chemistry_score: float,
        diversity_score: float,
        domain_distribution: Counter,
        records: List[ConsultantSelectionRecord],
    ) -> List[str]:
        """Generate specific recommendations for consultant optimization"""
        recommendations = []

        # Selection rate recommendations
        if selection_rate > self.config["diversity_target_threshold"]:
            recommendations.append(
                f"🚨 CRITICAL: Reduce selection rate from {selection_rate:.1%} to <{self.config['diversity_target_threshold']:.0%}"
            )
            recommendations.append(
                "Apply stronger diversity penalties in selection algorithm"
            )
        elif selection_rate < 0.1:
            if chemistry_score > 0.7:
                recommendations.append(
                    f"💡 OPPORTUNITY: Increase utilization (high chemistry: {chemistry_score:.2f})"
                )

        # Chemistry score recommendations
        if chemistry_score < 0.7:
            recommendations.append(
                f"⚗️ CHEMISTRY: Improve team combinations (current: {chemistry_score:.2f}, target: >0.75)"
            )
            recommendations.append(
                "Review consultant-pattern matching and team composition strategies"
            )

        # Domain specialization recommendations
        if domain_distribution:
            dominant_domain = domain_distribution.most_common(1)[0]
            if dominant_domain[1] / sum(domain_distribution.values()) > 0.7:
                recommendations.append(
                    f"🎯 SPECIALIZATION: High specialization in {dominant_domain[0]} domain"
                )
                recommendations.append(
                    "Consider expanding to related domains for better versatility"
                )

        # Diversity recommendations
        if diversity_score < 0.5:
            recommendations.append("🌈 DIVERSITY: Contribute more to team diversity")
            recommendations.append(
                "Reduce overlap with other frequently selected consultants"
            )

        return recommendations[:5]  # Limit to top 5 recommendations

    def _calculate_diversity_index(self, consultant_rates: Dict[str, float]) -> float:
        """Calculate Simpson's diversity index for consultant selection"""
        try:
            # Simpson's Diversity Index: 1 - Σ(pi^2)
            sum_squares = sum(rate**2 for rate in consultant_rates.values())
            diversity_index = 1 - sum_squares
            return diversity_index
        except:
            return 0.0

    def _calculate_simple_trend(self, values: List[float]) -> str:
        """Calculate simple trend direction"""
        if len(values) < 3:
            return "insufficient_data"

        first_third = values[: len(values) // 3]
        last_third = values[-len(values) // 3 :]

        first_avg = statistics.mean(first_third)
        last_avg = statistics.mean(last_third)

        if last_avg > first_avg * 1.05:
            return "improving"
        elif last_avg < first_avg * 0.95:
            return "declining"
        else:
            return "stable"

    def _calculate_domain_specialization(
        self, consultant_id: str, records: List[ConsultantSelectionRecord]
    ) -> float:
        """Calculate how specialized consultant is in their ideal domains"""
        ideal_domains = set(self.consultant_domains.get(consultant_id, []))
        if not ideal_domains:
            return 0.5  # Neutral score for unknown consultants

        domain_counts = Counter(r.domain.lower() for r in records)
        total_selections = len(records)

        # Calculate percentage of selections in ideal domains
        ideal_selections = sum(
            count
            for domain, count in domain_counts.items()
            if any(ideal.lower() in domain for ideal in ideal_domains)
        )

        specialization_score = ideal_selections / max(total_selections, 1)
        return min(specialization_score, 1.0)

    async def _generate_diversity_recommendations(
        self,
        consultant_rates: Dict[str, float],
        violations: List[Tuple[str, float]],
        domain_diversity: Dict[str, Any],
        chemistry_analysis: Dict[str, Any],
    ) -> List[str]:
        """Generate comprehensive diversity recommendations"""
        recommendations = []

        # Diversity violation recommendations
        if violations:
            recommendations.append(
                f"🚨 IMMEDIATE: {len(violations)} consultant(s) exceed diversity target"
            )
            for consultant, rate in violations[:3]:  # Top 3 violations
                recommendations.append(
                    f"  → Reduce {consultant} from {rate:.1%} to <40%"
                )

        # Underutilized consultant recommendations
        underutilized = [(c, r) for c, r in consultant_rates.items() if r < 0.15]
        if (
            underutilized and len(underutilized) < len(consultant_rates) * 0.7
        ):  # If not everyone is underutilized
            recommendations.append(
                f"💡 OPPORTUNITY: {len(underutilized)} consultant(s) underutilized"
            )
            for consultant, rate in underutilized[:2]:  # Top 2 underutilized
                chemistry_data = chemistry_analysis.get(consultant, {})
                if chemistry_data.get("average_chemistry", 0) > 0.7:
                    recommendations.append(
                        f"  → Increase {consultant} usage (strong chemistry: {chemistry_data['average_chemistry']:.2f})"
                    )

        # Domain diversity recommendations
        low_diversity_domains = [
            domain
            for domain, data in domain_diversity.items()
            if data["diversity_ratio"] < 0.3 and data["total_selections"] >= 5
        ]
        if low_diversity_domains:
            recommendations.append(
                f"🎯 DOMAIN FOCUS: Improve diversity in {', '.join(low_diversity_domains[:2])} domains"
            )

        # Chemistry optimization recommendations
        low_chemistry = [
            consultant
            for consultant, data in chemistry_analysis.items()
            if data["average_chemistry"] < 0.7
        ]
        if low_chemistry:
            recommendations.append(
                f"⚗️ CHEMISTRY: Optimize team composition for {len(low_chemistry)} consultant(s)"
            )

        return recommendations[:8]  # Limit to top 8 recommendations

    def _create_empty_analysis(
        self, consultant_id: str, period: int
    ) -> ConsultantPerformanceAnalysis:
        """Create empty analysis for consultants with no data"""
        return ConsultantPerformanceAnalysis(
            consultant_id=consultant_id,
            analysis_period=f"{period} days",
            total_selections=0,
            selection_rate=0.0,
            diversity_score=1.0,  # Perfect diversity when unused
            average_chemistry_score=0.0,
            domain_distribution={},
            framework_distribution={},
            performance_trend="no_data",
            effectiveness_rating="UNUSED",
            recommendations=["No selection data available for analysis"],
            analysis_timestamp=datetime.utcnow(),
        )

    def _create_error_analysis(
        self, consultant_id: str, error: str
    ) -> ConsultantPerformanceAnalysis:
        """Create error analysis when analysis fails"""
        return ConsultantPerformanceAnalysis(
            consultant_id=consultant_id,
            analysis_period="error",
            total_selections=0,
            selection_rate=0.0,
            diversity_score=0.0,
            average_chemistry_score=0.0,
            domain_distribution={},
            framework_distribution={},
            performance_trend="error",
            effectiveness_rating="ERROR",
            recommendations=[f"Analysis failed: {error}"],
            analysis_timestamp=datetime.utcnow(),
        )

    async def get_service_health(self) -> Dict[str, Any]:
        """Return service health and status"""
        total_selections = len(self.selection_history)
        unique_consultants = len(set(r.consultant_id for r in self.selection_history))
        active_alerts = sum(len(alerts) for alerts in self.active_alerts.values())

        # Calculate recent activity (last 24 hours)
        recent_cutoff = datetime.utcnow() - timedelta(hours=24)
        recent_selections = len(
            [
                r
                for r in self.selection_history
                if r.selection_timestamp >= recent_cutoff
            ]
        )

        return {
            "service_name": "ConsultantPerformanceTrackingService",
            "status": "healthy",
            "version": "sprint_1",
            "capabilities": [
                "consultant_selection_tracking",
                "diversity_monitoring",
                "performance_analytics",
                "chemistry_analysis",
                "effectiveness_ranking",
                "optimization_recommendations",
            ],
            "tracking_statistics": {
                "total_selections_tracked": total_selections,
                "unique_consultants": unique_consultants,
                "recent_activity_24h": recent_selections,
                "active_diversity_alerts": active_alerts,
                "cache_entries": len(self.analytics_cache),
            },
            "configuration": self.config,
            "sprint_1_targets": {
                "diversity_threshold": f"<{self.config['diversity_target_threshold']:.0%}",
                "chemistry_target": f">{self.config['chemistry_target_threshold']:.2f}",
                "primary_goal": "Reduce mckinsey_strategist selection bias",
            },
            "last_health_check": datetime.utcnow().isoformat(),
        }


# Global service instance for dependency injection
_consultant_performance_service: Optional[ConsultantPerformanceTrackingService] = None


def get_consultant_performance_service() -> ConsultantPerformanceTrackingService:
    """Get or create global consultant performance tracking service instance"""
    global _consultant_performance_service

    if _consultant_performance_service is None:
        _consultant_performance_service = ConsultantPerformanceTrackingService()

    return _consultant_performance_service
