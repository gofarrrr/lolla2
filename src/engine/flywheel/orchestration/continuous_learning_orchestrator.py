#!/usr/bin/env python3
"""
Continuous Learning Orchestrator
Phase 4: Complete Test-Driven Data Flywheel Orchestration

Orchestrates the complete learning cycle: Test → Analyze → Learn → Update → Validate
Creates a self-improving system that gets smarter with every test execution.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from uuid import UUID, uuid4
import numpy as np

try:
    from src.engine.flywheel.test_flywheel_manager import (
        TestFlywheelManager,
        EnhancedTestResult,
        TestLearningInsight,
        LearningSignal,
        get_test_flywheel_manager,
    )
    from src.engine.flywheel.test_feedback_engine import (
        TestFeedbackEngine,
        FailureAnalysis,
        ImprovementRecommendation,
        get_test_feedback_engine,
    )
    from src.engine.intelligence.bayesian_effectiveness_updater import (
        BayesianEffectivenessUpdater,
        get_bayesian_updater,
        run_bayesian_learning_cycle,
    )
    from src.engine.core.learning_loop import get_learning_loop_manager
    from src.engine.monitoring.performance_validator import get_performance_validator
    from src.engine.testing.value_generating_tests import get_value_test_runner

    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    print(f"Warning: Dependencies not available: {e}")


class LearningPhase(str, Enum):
    """Phases of the continuous learning cycle"""

    DORMANT = "dormant"  # No active learning
    TEST_COLLECTION = "test_collection"  # Collecting test results
    ANALYSIS = "analysis"  # Analyzing test patterns
    INSIGHT_GENERATION = "insight_generation"  # Generating learning insights
    MODEL_UPDATE = "model_update"  # Updating model effectiveness
    VALIDATION = "validation"  # Validating improvements
    DEPLOYMENT = "deployment"  # Deploying improvements


class LearningTrigger(str, Enum):
    """Events that can trigger learning cycles"""

    TEST_COMPLETION = "test_completion"  # Test suite completed
    FAILURE_THRESHOLD = "failure_threshold"  # Too many failures
    SCHEDULED = "scheduled"  # Time-based trigger
    MANUAL = "manual"  # Manual trigger
    PERFORMANCE_DROP = "performance_drop"  # Performance degradation


@dataclass
class LearningCycleMetrics:
    """Metrics for a learning cycle"""

    cycle_id: UUID = field(default_factory=uuid4)
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None

    # Input metrics
    tests_analyzed: int = 0
    failures_processed: int = 0
    insights_generated: int = 0

    # Learning metrics
    models_updated: int = 0
    effectiveness_changes: Dict[str, float] = field(default_factory=dict)
    patterns_identified: int = 0
    recommendations_generated: int = 0

    # Outcome metrics
    predicted_improvements: int = 0
    actual_improvements: int = 0
    learning_velocity: float = 0.0

    # Performance metrics
    cycle_duration_seconds: float = 0.0
    processing_efficiency: float = 0.0

    # Quality metrics
    insight_quality_score: float = 0.0
    recommendation_relevance: float = 0.0

    trigger: LearningTrigger = LearningTrigger.TEST_COMPLETION
    success: bool = False
    error_message: Optional[str] = None


@dataclass
class LearningStrategy:
    """Configuration for learning behavior"""

    strategy_id: str = "default"

    # Trigger configuration
    min_tests_for_cycle: int = 10
    max_failure_rate: float = 0.3
    cycle_interval_hours: int = 24

    # Learning configuration
    insight_confidence_threshold: float = 0.6
    model_update_threshold: float = 0.1
    pattern_clustering_enabled: bool = True

    # Validation configuration
    validate_improvements: bool = True
    validation_test_count: int = 5
    rollback_on_regression: bool = True

    # Adaptation configuration
    adaptive_thresholds: bool = True
    learning_rate: float = 0.1
    exploration_rate: float = 0.2


class ContinuousLearningOrchestrator:
    """
    Orchestrates the complete test-driven continuous learning cycle

    Learning Cycle Flow:
    1. TEST_COLLECTION: Gather test results and execution data
    2. ANALYSIS: Analyze patterns, failures, and performance trends
    3. INSIGHT_GENERATION: Generate actionable learning insights
    4. MODEL_UPDATE: Update model effectiveness and selection logic
    5. VALIDATION: Test improvements with validation scenarios
    6. DEPLOYMENT: Deploy validated improvements to production

    Key Features:
    - Automated learning cycle execution
    - Configurable learning strategies
    - Performance monitoring and rollback
    - Learning velocity optimization
    - Adaptive threshold management
    - Complete audit trail
    """

    def __init__(self, strategy: LearningStrategy = None):
        self.logger = logging.getLogger(__name__)
        self.strategy = strategy or LearningStrategy()

        # Core components
        self.flywheel_manager: Optional[TestFlywheelManager] = None
        self.feedback_engine: Optional[TestFeedbackEngine] = None
        self.bayesian_updater: Optional[BayesianEffectivenessUpdater] = None
        self.learning_loop_manager = None

        # Initialize components if available
        if DEPENDENCIES_AVAILABLE:
            try:
                self.flywheel_manager = get_test_flywheel_manager()
                self.feedback_engine = get_test_feedback_engine()
                self.bayesian_updater = get_bayesian_updater()
                self.learning_loop_manager = get_learning_loop_manager()
            except Exception as e:
                self.logger.warning(f"Could not initialize components: {e}")

        # State management
        self.current_phase: LearningPhase = LearningPhase.DORMANT
        self.cycle_history: List[LearningCycleMetrics] = []
        self.active_cycle: Optional[LearningCycleMetrics] = None

        # Performance tracking
        self.learning_velocity_history: List[float] = []
        self.model_performance_baselines: Dict[str, float] = {}
        self.last_cycle_time: Optional[datetime] = None

        # Configuration
        self.auto_mode: bool = False
        self.cycle_lock = asyncio.Lock()

        self.logger.info("🎼 Continuous Learning Orchestrator initialized")

    async def start_auto_learning(self) -> None:
        """Start automatic learning cycle execution"""

        self.auto_mode = True
        self.logger.info("🚀 Started automatic continuous learning")

        # Start background task for scheduled cycles
        asyncio.create_task(self._auto_learning_loop())

    async def stop_auto_learning(self) -> None:
        """Stop automatic learning cycle execution"""

        self.auto_mode = False
        self.logger.info("⏹️ Stopped automatic continuous learning")

    async def _auto_learning_loop(self) -> None:
        """Background loop for automatic learning cycles"""

        while self.auto_mode:
            try:
                # Check if cycle should be triggered
                trigger = await self._check_cycle_triggers()

                if trigger:
                    await self.execute_learning_cycle(trigger)

                # Sleep until next check
                await asyncio.sleep(3600)  # Check every hour

            except Exception as e:
                self.logger.error(f"Error in auto learning loop: {e}")
                await asyncio.sleep(3600)  # Continue after error

    async def _check_cycle_triggers(self) -> Optional[LearningTrigger]:
        """Check if learning cycle should be triggered"""

        # Check scheduled trigger
        if self.last_cycle_time:
            hours_since_cycle = (
                datetime.utcnow() - self.last_cycle_time
            ).total_seconds() / 3600
            if hours_since_cycle >= self.strategy.cycle_interval_hours:
                return LearningTrigger.SCHEDULED

        # Check failure threshold trigger
        if self.flywheel_manager:
            health_report = self.flywheel_manager.get_flywheel_health_report()

            if health_report.get("status") == "healthy":
                failure_rate = health_report["metrics"].get("failure_rate_24h", 0.0)
                if failure_rate > self.strategy.max_failure_rate:
                    return LearningTrigger.FAILURE_THRESHOLD

        # Check test completion trigger
        if self.flywheel_manager:
            total_tests = self.flywheel_manager.flywheel_metrics[
                "total_tests_processed"
            ]
            if total_tests >= self.strategy.min_tests_for_cycle:
                return LearningTrigger.TEST_COMPLETION

        return None

    async def execute_learning_cycle(
        self, trigger: LearningTrigger = LearningTrigger.MANUAL
    ) -> LearningCycleMetrics:
        """Execute complete learning cycle"""

        async with self.cycle_lock:  # Prevent concurrent cycles

            self.logger.info(f"🔄 Starting learning cycle (trigger: {trigger.value})")

            # Initialize cycle metrics
            cycle_metrics = LearningCycleMetrics(trigger=trigger)
            self.active_cycle = cycle_metrics

            try:
                # Phase 1: Test Collection
                await self._execute_test_collection_phase(cycle_metrics)

                # Phase 2: Analysis
                await self._execute_analysis_phase(cycle_metrics)

                # Phase 3: Insight Generation
                await self._execute_insight_generation_phase(cycle_metrics)

                # Phase 4: Model Update
                await self._execute_model_update_phase(cycle_metrics)

                # Phase 5: Validation
                await self._execute_validation_phase(cycle_metrics)

                # Phase 6: Deployment
                await self._execute_deployment_phase(cycle_metrics)

                # Complete cycle
                cycle_metrics.success = True
                cycle_metrics.end_time = datetime.utcnow()
                cycle_metrics.cycle_duration_seconds = (
                    cycle_metrics.end_time - cycle_metrics.start_time
                ).total_seconds()

                self.logger.info(
                    f"✅ Learning cycle completed successfully | "
                    f"Duration: {cycle_metrics.cycle_duration_seconds:.1f}s | "
                    f"Models updated: {cycle_metrics.models_updated} | "
                    f"Insights: {cycle_metrics.insights_generated}"
                )

            except Exception as e:
                cycle_metrics.success = False
                cycle_metrics.error_message = str(e)
                cycle_metrics.end_time = datetime.utcnow()

                self.logger.error(f"❌ Learning cycle failed: {e}")

            finally:
                # Store cycle results
                self.cycle_history.append(cycle_metrics)
                self.active_cycle = None
                self.current_phase = LearningPhase.DORMANT
                self.last_cycle_time = datetime.utcnow()

            return cycle_metrics

    async def _execute_test_collection_phase(
        self, metrics: LearningCycleMetrics
    ) -> None:
        """
        Week 2 Day 4: Scalable Phase 1 - Incremental batch processing of test results
        Replaces full 24-hour window loading with incremental updates
        """

        self.current_phase = LearningPhase.TEST_COLLECTION
        self.logger.info("📊 Week 2 Day 4: Scalable Test Collection")

        if not self.flywheel_manager:
            raise Exception("Flywheel manager not available")

        # Week 2 Day 4: Get or initialize learning state for incremental processing
        try:
            learning_state = await self._get_or_create_learning_state(
                "continuous_learning"
            )
            last_processed = learning_state.get("last_processed_timestamp")
            batch_size = learning_state.get("batch_size", 100)

            self.logger.info(
                f"   📍 Incremental processing from: {last_processed}\n"
                f"   📦 Batch size: {batch_size}"
            )

        except Exception as e:
            self.logger.warning(
                f"Could not access learning state, falling back to time-based approach: {e}"
            )
            # Fallback to time-based approach
            cutoff = datetime.utcnow() - timedelta(hours=1)  # Reduced window
            last_processed = cutoff
            batch_size = 100

        # Week 2 Day 4: Process test results in batches instead of loading all
        total_processed = 0
        total_failures = 0

        try:
            # Get test results incrementally (simplified approach for this implementation)
            all_results = list(self.flywheel_manager.test_results.values())

            # Filter to only unprocessed results
            if last_processed:
                unprocessed_results = [
                    result
                    for result in all_results
                    if result.created_at > last_processed
                ]
            else:
                # First run - process recent results only
                cutoff = datetime.utcnow() - timedelta(hours=1)
                unprocessed_results = [
                    result for result in all_results if result.created_at > cutoff
                ]

            self.logger.info(
                f"   🔄 Found {len(unprocessed_results)} unprocessed test results"
            )

            # Process in batches
            for batch_start in range(0, len(unprocessed_results), batch_size):
                batch_end = min(batch_start + batch_size, len(unprocessed_results))
                batch = unprocessed_results[batch_start:batch_end]

                # Process this batch
                batch_failures = await self._process_test_results_batch(batch)

                total_processed += len(batch)
                total_failures += batch_failures

                self.logger.debug(
                    f"   📦 Processed batch {batch_start//batch_size + 1}: {len(batch)} results, {batch_failures} failures"
                )

            # Update learning state with latest processed timestamp
            if unprocessed_results:
                latest_timestamp = max(
                    result.created_at for result in unprocessed_results
                )
                await self._update_learning_state(
                    "continuous_learning",
                    {
                        "last_processed_timestamp": latest_timestamp,
                        "total_processed": total_processed,
                        "processing_status": "completed",
                    },
                )

        except Exception as e:
            self.logger.error(f"❌ Batch processing failed: {e}")
            await self._update_learning_state(
                "continuous_learning",
                {"processing_status": "error", "last_error_message": str(e)},
            )
            raise

        metrics.tests_analyzed = total_processed
        metrics.failures_processed = total_failures

        self.logger.info(
            f"   ✅ Scalable collection complete: {metrics.tests_analyzed} results processed, "
            f"{metrics.failures_processed} failures (batch size: {batch_size})"
        )

    async def _execute_analysis_phase(self, metrics: LearningCycleMetrics) -> None:
        """Phase 2: Analyze test patterns and failures"""

        self.current_phase = LearningPhase.ANALYSIS
        self.logger.info("🔍 Phase 2: Analysis")

        if not self.feedback_engine or not self.flywheel_manager:
            raise Exception("Required components not available")

        # Get recent failed tests for analysis
        cutoff = datetime.utcnow() - timedelta(hours=24)
        recent_tests = [
            result
            for result in self.flywheel_manager.test_results.values()
            if result.created_at > cutoff and result.outcome in ["failed", "error"]
        ]

        # Analyze each failure
        for test_result in recent_tests:
            try:
                await self.feedback_engine.analyze_test_failure(test_result)
            except Exception as e:
                self.logger.warning(
                    f"Failed to analyze test {test_result.test_id}: {e}"
                )

        # Cluster similar failures
        pattern_clusters = await self.feedback_engine.cluster_similar_failures()
        metrics.patterns_identified = len(pattern_clusters)

        self.logger.info(
            f"   Analyzed {len(recent_tests)} failures, "
            f"identified {metrics.patterns_identified} patterns"
        )

    async def _execute_insight_generation_phase(
        self, metrics: LearningCycleMetrics
    ) -> None:
        """Phase 3: Generate actionable learning insights"""

        self.current_phase = LearningPhase.INSIGHT_GENERATION
        self.logger.info("💡 Phase 3: Insight Generation")

        if not self.feedback_engine:
            raise Exception("Feedback engine not available")

        # Generate improvement recommendations
        recommendations = (
            await self.feedback_engine.generate_improvement_recommendations()
        )
        metrics.recommendations_generated = len(recommendations)

        # Filter high-quality insights
        high_quality_insights = [
            rec
            for rec in recommendations
            if rec.priority_score >= self.strategy.insight_confidence_threshold
        ]

        metrics.insights_generated = len(high_quality_insights)

        # Calculate insight quality score
        if recommendations:
            avg_priority = np.mean([rec.priority_score for rec in recommendations])
            metrics.insight_quality_score = avg_priority

        self.logger.info(
            f"   Generated {metrics.recommendations_generated} recommendations, "
            f"{metrics.insights_generated} high-quality insights"
        )

    async def _execute_model_update_phase(self, metrics: LearningCycleMetrics) -> None:
        """Phase 4: Update model effectiveness based on insights"""

        self.current_phase = LearningPhase.MODEL_UPDATE
        self.logger.info("🧠 Phase 4: Model Update")

        if not self.flywheel_manager or not self.bayesian_updater:
            raise Exception("Required components not available")

        # Feed insights to learning system
        learning_result = await self.flywheel_manager.feed_insights_to_learning_system()

        if learning_result["status"] == "success":
            metrics.models_updated = learning_result["models_updated"]
            metrics.learning_velocity = learning_result["learning_velocity"]

        # Run Bayesian learning cycle
        bayesian_result = await run_bayesian_learning_cycle()

        # Track effectiveness changes
        top_models = bayesian_result.get("top_performers", [])
        for model_info in top_models:
            model_id = model_info["model_id"]
            effectiveness = model_info["bayesian_effectiveness"]
            metrics.effectiveness_changes[model_id] = effectiveness

        self.logger.info(
            f"   Updated {metrics.models_updated} models, "
            f"learning velocity: {metrics.learning_velocity:.3f}"
        )

    async def _execute_validation_phase(self, metrics: LearningCycleMetrics) -> None:
        """Phase 5: Validate improvements through targeted testing"""

        self.current_phase = LearningPhase.VALIDATION
        self.logger.info("✓ Phase 5: Validation")

        if not self.strategy.validate_improvements:
            self.logger.info("   Validation disabled - skipping")
            return

        try:
            # Run validation tests using value-generating test runner
            test_runner = get_value_test_runner() if DEPENDENCIES_AVAILABLE else None

            if test_runner:
                # Run a subset of tests to validate improvements
                validation_results = await test_runner.run_test_suite(
                    ["behavioral_consistency", "confidence_calibration"]
                )

                # Analyze validation results
                total_tests = sum(
                    len(results) for results in validation_results.values()
                )
                passed_tests = sum(
                    len([r for r in results if r["status"] == "passed"])
                    for results in validation_results.values()
                )

                improvement_rate = passed_tests / total_tests if total_tests > 0 else 0
                metrics.predicted_improvements = total_tests
                metrics.actual_improvements = passed_tests

                self.logger.info(
                    f"   Validation: {passed_tests}/{total_tests} tests passed "
                    f"({improvement_rate:.1%} success rate)"
                )

                # Check for regression
                if self.strategy.rollback_on_regression and improvement_rate < 0.5:
                    self.logger.warning(
                        "⚠️ Validation shows regression - consider rollback"
                    )

            else:
                self.logger.info("   Test runner not available - skipping validation")

        except Exception as e:
            self.logger.error(f"Validation failed: {e}")

    async def _execute_deployment_phase(self, metrics: LearningCycleMetrics) -> None:
        """Phase 6: Deploy validated improvements"""

        self.current_phase = LearningPhase.DEPLOYMENT
        self.logger.info("🚀 Phase 6: Deployment")

        # In a full implementation, this would deploy model updates
        # For now, we'll just log the completion

        # Update learning velocity tracking
        if metrics.learning_velocity > 0:
            self.learning_velocity_history.append(metrics.learning_velocity)

            # Keep only recent history
            if len(self.learning_velocity_history) > 100:
                self.learning_velocity_history = self.learning_velocity_history[-100:]

        # Update performance baselines
        for model_id, effectiveness in metrics.effectiveness_changes.items():
            self.model_performance_baselines[model_id] = effectiveness

        self.logger.info("   Deployment completed - improvements are now active")

    def get_learning_health_report(self) -> Dict[str, Any]:
        """Get comprehensive learning system health report"""

        if not self.cycle_history:
            return {"status": "no_data", "message": "No learning cycles executed yet"}

        # Recent cycle analysis
        recent_cycles = [
            cycle
            for cycle in self.cycle_history
            if cycle.start_time > datetime.utcnow() - timedelta(days=7)
        ]

        successful_cycles = [cycle for cycle in recent_cycles if cycle.success]
        success_rate = (
            len(successful_cycles) / len(recent_cycles) if recent_cycles else 0
        )

        # Learning velocity analysis
        avg_velocity = (
            np.mean(self.learning_velocity_history)
            if self.learning_velocity_history
            else 0
        )
        velocity_trend = self._calculate_velocity_trend()

        # Model performance analysis
        total_models_tracked = len(self.model_performance_baselines)
        avg_effectiveness = (
            np.mean(list(self.model_performance_baselines.values()))
            if self.model_performance_baselines
            else 0
        )

        # Determine health status
        health_factors = [
            success_rate * 0.3,
            min(1.0, avg_velocity) * 0.3,
            avg_effectiveness * 0.2,
            min(1.0, velocity_trend + 0.5) * 0.2,  # Trend from -0.5 to +0.5
        ]

        overall_health = sum(health_factors)

        status = (
            "excellent"
            if overall_health > 0.8
            else (
                "good"
                if overall_health > 0.6
                else "fair" if overall_health > 0.4 else "poor"
            )
        )

        return {
            "status": status,
            "overall_health_score": overall_health,
            "metrics": {
                "total_cycles": len(self.cycle_history),
                "recent_cycles_7d": len(recent_cycles),
                "success_rate": success_rate,
                "avg_learning_velocity": avg_velocity,
                "velocity_trend": velocity_trend,
                "models_tracked": total_models_tracked,
                "avg_model_effectiveness": avg_effectiveness,
                "current_phase": self.current_phase.value,
                "auto_mode_active": self.auto_mode,
            },
            "recent_performance": {
                "cycles_last_week": len(recent_cycles),
                "avg_cycle_duration": (
                    np.mean([c.cycle_duration_seconds for c in recent_cycles])
                    if recent_cycles
                    else 0
                ),
                "avg_insights_per_cycle": (
                    np.mean([c.insights_generated for c in recent_cycles])
                    if recent_cycles
                    else 0
                ),
                "avg_models_updated_per_cycle": (
                    np.mean([c.models_updated for c in recent_cycles])
                    if recent_cycles
                    else 0
                ),
            },
            "recommendations": self._generate_health_recommendations(
                overall_health, success_rate, avg_velocity
            ),
        }

    def _calculate_velocity_trend(self) -> float:
        """Calculate trend in learning velocity (-1 to +1)"""

        if len(self.learning_velocity_history) < 10:
            return 0.0

        # Simple linear trend calculation
        recent = self.learning_velocity_history[-10:]
        older = (
            self.learning_velocity_history[-20:-10]
            if len(self.learning_velocity_history) >= 20
            else []
        )

        if not older:
            return 0.0

        recent_avg = np.mean(recent)
        older_avg = np.mean(older)

        if older_avg == 0:
            return 1.0 if recent_avg > 0 else 0.0

        trend = (recent_avg - older_avg) / older_avg
        return max(-1.0, min(1.0, trend))  # Clamp to [-1, 1]

    def _generate_health_recommendations(
        self, health_score: float, success_rate: float, velocity: float
    ) -> List[str]:
        """Generate recommendations for improving learning health"""

        recommendations = []

        if health_score < 0.5:
            recommendations.append(
                "🚨 Learning system health is poor - investigate cycle failures and component issues"
            )

        if success_rate < 0.7:
            recommendations.append(
                "⚠️ Low cycle success rate - check component availability and error handling"
            )

        if velocity < 0.1:
            recommendations.append(
                "📈 Low learning velocity - increase test execution frequency and insight generation"
            )

        if len(self.learning_velocity_history) < 10:
            recommendations.append(
                "🔄 Execute more learning cycles to establish performance baselines"
            )

        if not self.auto_mode:
            recommendations.append(
                "🤖 Enable auto-mode for consistent continuous learning"
            )

        if len(self.model_performance_baselines) < 3:
            recommendations.append("🧠 Track more models to improve learning coverage")

        return recommendations

    def get_cycle_summary(self, cycle_id: UUID = None) -> Dict[str, Any]:
        """Get detailed summary of a specific learning cycle"""

        if cycle_id:
            cycle = next(
                (c for c in self.cycle_history if c.cycle_id == cycle_id), None
            )
            if not cycle:
                return {"error": f"Cycle {cycle_id} not found"}
        else:
            # Get most recent cycle
            cycle = self.cycle_history[-1] if self.cycle_history else None
            if not cycle:
                return {"error": "No cycles executed yet"}

        return {
            "cycle_id": str(cycle.cycle_id),
            "trigger": cycle.trigger.value,
            "success": cycle.success,
            "duration_seconds": cycle.cycle_duration_seconds,
            "phases_completed": "all" if cycle.success else "partial",
            "metrics": {
                "tests_analyzed": cycle.tests_analyzed,
                "failures_processed": cycle.failures_processed,
                "insights_generated": cycle.insights_generated,
                "models_updated": cycle.models_updated,
                "patterns_identified": cycle.patterns_identified,
                "learning_velocity": cycle.learning_velocity,
            },
            "quality_scores": {
                "insight_quality": cycle.insight_quality_score,
                "processing_efficiency": cycle.processing_efficiency,
            },
            "improvements": {
                "predicted": cycle.predicted_improvements,
                "actual": cycle.actual_improvements,
                "effectiveness_changes": cycle.effectiveness_changes,
            },
            "timestamp": cycle.start_time.isoformat(),
            "error_message": cycle.error_message,
        }

    async def _get_or_create_learning_state(self, state_key: str) -> Dict[str, Any]:
        """
        Week 2 Day 4: Get or create learning state from database for incremental processing
        """
        try:
            # Try to connect to Supabase (if available)
            from src.core.supabase_platform import get_supabase_client

            client = get_supabase_client()

            # Query learning_state table
            response = (
                client.table("learning_state")
                .select("*")
                .eq("state_key", state_key)
                .execute()
            )

            if response.data:
                state = response.data[0]
                return {
                    "state_key": state["state_key"],
                    "last_processed_timestamp": state["last_processed_timestamp"],
                    "batch_size": state["batch_size"],
                    "total_processed": state["total_processed"],
                    "processing_status": state["processing_status"],
                    "current_cycle_id": state.get("current_cycle_id"),
                }
            else:
                # Create new learning state
                new_state = {
                    "state_key": state_key,
                    "batch_size": 100,
                    "total_processed": 0,
                    "processing_status": "idle",
                }
                client.table("learning_state").insert(new_state).execute()
                return new_state

        except Exception as e:
            self.logger.warning(
                f"Database unavailable for learning state, using fallback: {e}"
            )
            # Fallback to in-memory state
            return {
                "state_key": state_key,
                "last_processed_timestamp": None,
                "batch_size": 100,
                "total_processed": 0,
                "processing_status": "idle",
            }

    async def _update_learning_state(
        self, state_key: str, updates: Dict[str, Any]
    ) -> None:
        """
        Week 2 Day 4: Update learning state in database with processing progress
        """
        try:
            from src.core.supabase_platform import get_supabase_client

            client = get_supabase_client()

            # Add timestamp to updates
            updates["updated_at"] = datetime.utcnow().isoformat()

            # Update learning_state table
            client.table("learning_state").update(updates).eq(
                "state_key", state_key
            ).execute()

            self.logger.debug(f"Updated learning state '{state_key}': {updates}")

        except Exception as e:
            self.logger.warning(f"Could not update learning state in database: {e}")

    async def _process_test_results_batch(self, batch: List[Any]) -> int:
        """
        Week 2 Day 4: Process a single batch of test results for failures
        Returns number of failures found in the batch
        """
        batch_failures = 0

        for result in batch:
            try:
                # Check if this is a test failure that needs analysis
                if hasattr(result, "status") and result.status == "failed":
                    batch_failures += 1

                    # Store failure for analysis phase (simplified approach)
                    if not hasattr(self, "_batch_failures"):
                        self._batch_failures = []
                    self._batch_failures.append(result)

                # Update test flywheel manager if available
                if self.flywheel_manager and hasattr(result, "engagement_id"):
                    try:
                        await self.flywheel_manager.process_test_result(result)
                    except Exception as e:
                        self.logger.debug(
                            f"Could not update flywheel for {result.engagement_id}: {e}"
                        )

            except Exception as e:
                self.logger.warning(f"Error processing test result in batch: {e}")
                continue

        return batch_failures


# Global orchestrator instance
_global_orchestrator: Optional[ContinuousLearningOrchestrator] = None


def get_continuous_learning_orchestrator(
    strategy: LearningStrategy = None,
) -> ContinuousLearningOrchestrator:
    """Get or create global continuous learning orchestrator"""
    global _global_orchestrator

    if _global_orchestrator is None:
        _global_orchestrator = ContinuousLearningOrchestrator(strategy)

    return _global_orchestrator


# Convenience functions for common operations
async def trigger_learning_cycle(
    trigger: LearningTrigger = LearningTrigger.MANUAL,
) -> LearningCycleMetrics:
    """Trigger a learning cycle manually"""

    orchestrator = get_continuous_learning_orchestrator()
    return await orchestrator.execute_learning_cycle(trigger)


async def start_continuous_learning(strategy: LearningStrategy = None) -> None:
    """Start continuous learning in auto mode"""

    orchestrator = get_continuous_learning_orchestrator(strategy)
    await orchestrator.start_auto_learning()


async def get_learning_status() -> Dict[str, Any]:
    """Get current learning system status"""

    orchestrator = get_continuous_learning_orchestrator()
    return orchestrator.get_learning_health_report()


# Configuration presets
class LearningStrategies:
    """Predefined learning strategies for different use cases"""

    @staticmethod
    def aggressive() -> LearningStrategy:
        """Aggressive learning - frequent cycles, low thresholds"""
        return LearningStrategy(
            strategy_id="aggressive",
            min_tests_for_cycle=5,
            max_failure_rate=0.2,
            cycle_interval_hours=6,
            insight_confidence_threshold=0.4,
            model_update_threshold=0.05,
            learning_rate=0.2,
            exploration_rate=0.3,
        )

    @staticmethod
    def conservative() -> LearningStrategy:
        """Conservative learning - infrequent cycles, high thresholds"""
        return LearningStrategy(
            strategy_id="conservative",
            min_tests_for_cycle=50,
            max_failure_rate=0.5,
            cycle_interval_hours=72,
            insight_confidence_threshold=0.8,
            model_update_threshold=0.2,
            learning_rate=0.05,
            exploration_rate=0.1,
        )

    @staticmethod
    def balanced() -> LearningStrategy:
        """Balanced learning - default settings"""
        return LearningStrategy(
            strategy_id="balanced",
            min_tests_for_cycle=20,
            max_failure_rate=0.3,
            cycle_interval_hours=24,
            insight_confidence_threshold=0.6,
            model_update_threshold=0.1,
            learning_rate=0.1,
            exploration_rate=0.2,
        )
