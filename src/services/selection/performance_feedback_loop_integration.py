"""
Performance Feedback Loop Integration - SPRINT 2 Implementation
==============================================================

Orchestrates seamless integration of performance feedback across all LOLLA platform
selection and optimization services. Provides unified feedback collection, processing,
and learning system coordination.

Key Features:
- Unified feedback orchestration across pattern, consultant, and chemistry systems
- Real-time performance correlation and learning trigger management
- Cross-system feedback propagation and optimization coordination
- Performance event streaming with intelligent aggregation
- Learning system synchronization and conflict resolution

Integrates with:
- NWayPatternSelectionService (pattern effectiveness feedback)
- ContextualWeightingSystem (weight optimization feedback)
- ConsultantPerformanceTrackingService (consultant selection feedback)
- CognitiveChemistryEngine (chemistry optimization feedback)
- PerformanceMonitoringService (system-wide performance tracking)

SPRINT 2 TARGET: Complete the learning feedback loop for intelligent platform evolution
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import statistics
import weakref


class FeedbackType(Enum):
    """Types of performance feedback in the system"""

    PATTERN_EFFECTIVENESS = "pattern_effectiveness"
    CONSULTANT_PERFORMANCE = "consultant_performance"
    CHEMISTRY_OPTIMIZATION = "chemistry_optimization"
    WEIGHT_OPTIMIZATION = "weight_optimization"
    USER_SATISFACTION = "user_satisfaction"
    ANALYSIS_QUALITY = "analysis_quality"
    SYSTEM_PERFORMANCE = "system_performance"


class FeedbackPriority(Enum):
    """Priority levels for feedback processing"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKGROUND = "background"


@dataclass
class PerformanceFeedbackEvent:
    """Comprehensive performance feedback event"""

    event_id: str
    feedback_type: FeedbackType
    priority: FeedbackPriority
    timestamp: datetime
    source_service: str
    target_services: List[str]
    performance_metrics: Dict[str, float]
    context: Dict[str, Any]
    session_id: str
    user_id: Optional[str] = None
    analysis_id: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    processing_status: str = "pending"
    propagation_results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackProcessor:
    """Feedback processor configuration"""

    service_name: str
    feedback_types: List[FeedbackType]
    processor_function: Callable
    processing_delay_ms: int = 0
    batch_size: int = 1
    enabled: bool = True


@dataclass
class LearningCoordinationState:
    """State for coordinating learning across services"""

    active_optimizations: Dict[str, datetime]
    optimization_conflicts: List[Dict[str, Any]]
    learning_rates: Dict[str, float]
    convergence_status: Dict[str, bool]
    system_learning_enabled: bool = True
    last_global_optimization: Optional[datetime] = None


class PerformanceFeedbackLoopIntegration:
    """
    Orchestrates comprehensive performance feedback loops across all LOLLA platform services.

    Provides intelligent feedback routing, learning coordination, and optimization synchronization
    to ensure coherent system-wide performance improvement.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Core feedback management
        self.feedback_queue: deque = deque(maxlen=10000)
        self.feedback_processors: Dict[str, FeedbackProcessor] = {}
        self.feedback_history: Dict[FeedbackType, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )

        # Learning coordination
        self.learning_coordination: LearningCoordinationState = (
            LearningCoordinationState(
                active_optimizations={}, optimization_conflicts=[], learning_rates={}
            )
        )

        # Service registry and dependencies
        self.service_registry: Dict[str, weakref.ref] = {}
        self.service_dependencies: Dict[str, List[str]] = {}
        self.feedback_routing: Dict[FeedbackType, List[str]] = {}

        # Performance analytics
        self.feedback_analytics = {
            "total_events_processed": 0,
            "events_by_type": defaultdict(int),
            "processing_latencies": defaultdict(list),
            "optimization_triggers": defaultdict(int),
            "learning_improvements": [],
            "system_health_scores": deque(maxlen=100),
        }

        # Configuration
        self.config = {
            "feedback_processing_interval_ms": 100,
            "batch_processing_size": 50,
            "optimization_coordination_timeout_s": 300,
            "feedback_retention_hours": 24,
            "learning_rate_adaptation": True,
            "cross_service_optimization_enabled": True,
            "conflict_resolution_strategy": "priority_based",
            "performance_threshold_for_learning": 0.1,
            "global_optimization_interval_minutes": 60,
        }

        # Initialize feedback routing and processors
        self._initialize_feedback_routing()
        self._initialize_processors()

        # Start background processing
        self._background_task = asyncio.create_task(self._feedback_processing_loop())

        self.logger.info(
            "🔄 Performance Feedback Loop Integration initialized - SPRINT 2 Active"
        )

    def _initialize_feedback_routing(self):
        """Initialize feedback routing configuration"""
        self.feedback_routing = {
            FeedbackType.PATTERN_EFFECTIVENESS: [
                "nway_pattern_selection_service",
                "contextual_weighting_system",
            ],
            FeedbackType.CONSULTANT_PERFORMANCE: [
                "consultant_performance_tracking_service",
                "contextual_lollapalooza_engine",
                "contextual_weighting_system",
            ],
            FeedbackType.CHEMISTRY_OPTIMIZATION: [
                "cognitive_chemistry_engine",
                "contextual_weighting_system",
            ],
            FeedbackType.WEIGHT_OPTIMIZATION: [
                "contextual_weighting_system",
                "nway_pattern_selection_service",
                "cognitive_chemistry_engine",
            ],
            FeedbackType.USER_SATISFACTION: [
                "consultant_performance_tracking_service",
                "nway_pattern_selection_service",
                "performance_monitoring_service",
            ],
            FeedbackType.ANALYSIS_QUALITY: [
                "nway_pattern_selection_service",
                "cognitive_chemistry_engine",
                "consultant_performance_tracking_service",
            ],
            FeedbackType.SYSTEM_PERFORMANCE: [
                "performance_monitoring_service",
                "contextual_weighting_system",
            ],
        }

        self.logger.info(
            f"✅ Feedback routing initialized for {len(self.feedback_routing)} feedback types"
        )

    def _initialize_processors(self):
        """Initialize feedback processors for different services"""
        # Pattern effectiveness processor
        self.feedback_processors["pattern_effectiveness"] = FeedbackProcessor(
            service_name="pattern_effectiveness",
            feedback_types=[
                FeedbackType.PATTERN_EFFECTIVENESS,
                FeedbackType.ANALYSIS_QUALITY,
            ],
            processor_function=self._process_pattern_feedback,
            batch_size=10,
        )

        # Consultant performance processor
        self.feedback_processors["consultant_performance"] = FeedbackProcessor(
            service_name="consultant_performance",
            feedback_types=[
                FeedbackType.CONSULTANT_PERFORMANCE,
                FeedbackType.USER_SATISFACTION,
            ],
            processor_function=self._process_consultant_feedback,
            batch_size=15,
        )

        # Chemistry optimization processor
        self.feedback_processors["chemistry_optimization"] = FeedbackProcessor(
            service_name="chemistry_optimization",
            feedback_types=[FeedbackType.CHEMISTRY_OPTIMIZATION],
            processor_function=self._process_chemistry_feedback,
            batch_size=5,
        )

        # Weight optimization processor
        self.feedback_processors["weight_optimization"] = FeedbackProcessor(
            service_name="weight_optimization",
            feedback_types=[FeedbackType.WEIGHT_OPTIMIZATION],
            processor_function=self._process_weight_feedback,
            processing_delay_ms=50,  # Slight delay for stability
            batch_size=20,
        )

        # System performance processor
        self.feedback_processors["system_performance"] = FeedbackProcessor(
            service_name="system_performance",
            feedback_types=[FeedbackType.SYSTEM_PERFORMANCE],
            processor_function=self._process_system_feedback,
            batch_size=25,
        )

        self.logger.info(
            f"✅ Initialized {len(self.feedback_processors)} feedback processors"
        )

    async def submit_feedback(
        self,
        feedback_type: FeedbackType,
        performance_metrics: Dict[str, float],
        context: Dict[str, Any],
        source_service: str,
        priority: FeedbackPriority = FeedbackPriority.MEDIUM,
        user_id: Optional[str] = None,
        analysis_id: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Submit performance feedback for processing and learning.

        SPRINT 2: Core method for unified feedback submission across all services.

        Args:
            feedback_type: Type of feedback being submitted
            performance_metrics: Dictionary of performance metrics (0.0-1.0 scale)
            context: Full context dictionary from the operation
            source_service: Name of the service submitting feedback
            priority: Processing priority for the feedback
            user_id: Optional user identifier
            analysis_id: Optional analysis session identifier
            additional_data: Optional additional feedback data

        Returns:
            str: Unique event ID for tracking
        """
        try:
            # Generate unique event ID
            event_id = f"feedback_{feedback_type.value}_{datetime.utcnow().timestamp()}"

            # Determine target services
            target_services = self.feedback_routing.get(feedback_type, [])

            # Create feedback event
            feedback_event = PerformanceFeedbackEvent(
                event_id=event_id,
                feedback_type=feedback_type,
                priority=priority,
                timestamp=datetime.utcnow(),
                source_service=source_service,
                target_services=target_services,
                performance_metrics=performance_metrics,
                context=context,
                session_id=context.get("session_id", "unknown"),
                user_id=user_id,
                analysis_id=analysis_id,
                additional_data=additional_data or {},
            )

            # Add to processing queue
            self.feedback_queue.append(feedback_event)

            # Update analytics
            self.feedback_analytics["total_events_processed"] += 1
            self.feedback_analytics["events_by_type"][feedback_type] += 1

            # Store in history
            self.feedback_history[feedback_type].append(feedback_event)

            self.logger.debug(
                f"📥 Feedback submitted: {feedback_type.value} from {source_service} (priority: {priority.value})"
            )
            return event_id

        except Exception as e:
            self.logger.error(f"❌ Failed to submit feedback: {e}")
            return ""

    async def register_service(
        self,
        service_name: str,
        service_instance: Any,
        dependencies: Optional[List[str]] = None,
    ) -> bool:
        """
        Register a service for feedback integration.

        SPRINT 2: Service registration for coordinated feedback processing.

        Args:
            service_name: Unique name for the service
            service_instance: The service instance (stored as weak reference)
            dependencies: Optional list of services this service depends on

        Returns:
            bool: Success status
        """
        try:
            # Store service as weak reference to avoid circular dependencies
            self.service_registry[service_name] = weakref.ref(service_instance)

            # Store dependencies
            if dependencies:
                self.service_dependencies[service_name] = dependencies

            self.logger.info(
                f"📋 Service registered: {service_name} with {len(dependencies or [])} dependencies"
            )
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to register service {service_name}: {e}")
            return False

    async def coordinate_learning_optimization(
        self,
        service_name: str,
        optimization_type: str,
        estimated_duration_minutes: int = 5,
    ) -> bool:
        """
        Coordinate learning optimization across services to prevent conflicts.

        SPRINT 2: Intelligent optimization coordination to prevent learning conflicts.

        Args:
            service_name: Name of service requesting optimization
            optimization_type: Type of optimization being performed
            estimated_duration_minutes: Estimated duration of optimization

        Returns:
            bool: True if optimization is approved, False if conflict detected
        """
        try:
            current_time = datetime.utcnow()

            # Check for conflicting optimizations
            conflicts = await self._detect_optimization_conflicts(
                service_name, optimization_type
            )

            if conflicts:
                self.learning_coordination.optimization_conflicts.extend(conflicts)
                self.logger.warning(
                    f"⚠️ Optimization conflict detected for {service_name}: {len(conflicts)} conflicts"
                )
                return False

            # Register the optimization
            optimization_key = f"{service_name}_{optimization_type}"
            expected_completion = current_time + timedelta(
                minutes=estimated_duration_minutes
            )
            self.learning_coordination.active_optimizations[optimization_key] = (
                expected_completion
            )

            # Update learning rates if adaptive learning is enabled
            if self.config["learning_rate_adaptation"]:
                await self._adapt_learning_rates(service_name, optimization_type)

            self.logger.info(
                f"✅ Learning optimization approved: {service_name} - {optimization_type}"
            )
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to coordinate learning optimization: {e}")
            return False

    async def get_feedback_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive analytics on feedback processing and learning performance.

        SPRINT 2: Analytics dashboard for feedback loop health and effectiveness.
        """
        try:
            analytics = {
                "feedback_overview": {
                    "total_events_processed": self.feedback_analytics[
                        "total_events_processed"
                    ],
                    "events_by_type": dict(self.feedback_analytics["events_by_type"]),
                    "active_processors": len(
                        [p for p in self.feedback_processors.values() if p.enabled]
                    ),
                    "registered_services": len(self.service_registry),
                    "feedback_queue_size": len(self.feedback_queue),
                },
                "learning_coordination": {
                    "active_optimizations": len(
                        self.learning_coordination.active_optimizations
                    ),
                    "optimization_conflicts": len(
                        self.learning_coordination.optimization_conflicts
                    ),
                    "system_learning_enabled": self.learning_coordination.system_learning_enabled,
                    "last_global_optimization": (
                        self.learning_coordination.last_global_optimization.isoformat()
                        if self.learning_coordination.last_global_optimization
                        else None
                    ),
                },
                "performance_trends": {},
                "processing_efficiency": {},
                "learning_improvements": self.feedback_analytics[
                    "learning_improvements"
                ][
                    -10:
                ],  # Last 10 improvements
                "system_health": self._calculate_system_health(),
            }

            # Calculate performance trends by feedback type
            for feedback_type, events in self.feedback_history.items():
                if events:
                    recent_events = [
                        e
                        for e in events
                        if e.timestamp >= datetime.utcnow() - timedelta(hours=1)
                    ]
                    if recent_events:
                        performance_scores = []
                        for event in recent_events:
                            if "overall_score" in event.performance_metrics:
                                performance_scores.append(
                                    event.performance_metrics["overall_score"]
                                )

                        if performance_scores:
                            analytics["performance_trends"][feedback_type.value] = {
                                "recent_average": statistics.mean(performance_scores),
                                "trend": self._calculate_trend(performance_scores),
                                "event_count": len(recent_events),
                                "sample_size": len(performance_scores),
                            }

            # Calculate processing efficiency
            for processor_name, processor in self.feedback_processors.items():
                processing_times = self.feedback_analytics["processing_latencies"].get(
                    processor_name, []
                )
                if processing_times:
                    analytics["processing_efficiency"][processor_name] = {
                        "average_latency_ms": statistics.mean(
                            processing_times[-50:]
                        ),  # Last 50 processing times
                        "enabled": processor.enabled,
                        "batch_size": processor.batch_size,
                        "processed_count": len(processing_times),
                    }

            return analytics

        except Exception as e:
            self.logger.error(f"❌ Failed to generate feedback analytics: {e}")
            return {"error": str(e)}

    async def trigger_global_optimization(self) -> Dict[str, Any]:
        """
        Trigger coordinated global optimization across all registered services.

        SPRINT 2: System-wide optimization coordination for maximum performance gains.
        """
        try:
            if not self.config["cross_service_optimization_enabled"]:
                return {
                    "status": "disabled",
                    "message": "Cross-service optimization is disabled",
                }

            optimization_results = {
                "optimization_timestamp": datetime.utcnow().isoformat(),
                "services_optimized": [],
                "optimization_conflicts": [],
                "total_improvements": 0,
                "optimization_summary": {},
            }

            # Get optimization order based on dependencies
            optimization_order = self._calculate_optimization_order()

            self.logger.info(
                f"🌐 Starting global optimization across {len(optimization_order)} services"
            )

            for service_name in optimization_order:
                service_ref = self.service_registry.get(service_name)
                if not service_ref:
                    continue

                service_instance = service_ref()
                if not service_instance:
                    continue

                try:
                    # Request coordination approval
                    optimization_approved = await self.coordinate_learning_optimization(
                        service_name, "global_optimization", 10
                    )

                    if not optimization_approved:
                        optimization_results["optimization_conflicts"].append(
                            service_name
                        )
                        continue

                    # Trigger service-specific optimization
                    if hasattr(service_instance, "optimize_weights"):
                        service_result = await service_instance.optimize_weights()
                        optimization_results["services_optimized"].append(service_name)
                        optimization_results["optimization_summary"][
                            service_name
                        ] = service_result

                        # Count improvements
                        if (
                            isinstance(service_result, dict)
                            and service_result.get("improvements_found", 0) > 0
                        ):
                            optimization_results[
                                "total_improvements"
                            ] += service_result["improvements_found"]

                    # Small delay between optimizations
                    await asyncio.sleep(0.5)

                except Exception as e:
                    self.logger.error(
                        f"❌ Global optimization failed for {service_name}: {e}"
                    )
                    optimization_results["optimization_conflicts"].append(service_name)

            # Update coordination state
            self.learning_coordination.last_global_optimization = datetime.utcnow()

            # Record improvement
            if optimization_results["total_improvements"] > 0:
                improvement_record = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "type": "global_optimization",
                    "improvements_count": optimization_results["total_improvements"],
                    "services_involved": len(
                        optimization_results["services_optimized"]
                    ),
                }
                self.feedback_analytics["learning_improvements"].append(
                    improvement_record
                )

            self.logger.info(
                f"🎯 Global optimization complete: {optimization_results['total_improvements']} improvements across {len(optimization_results['services_optimized'])} services"
            )
            return optimization_results

        except Exception as e:
            self.logger.error(f"❌ Global optimization failed: {e}")
            return {"error": str(e)}

    async def _feedback_processing_loop(self):
        """Background loop for processing feedback events"""
        while True:
            try:
                if not self.feedback_queue:
                    await asyncio.sleep(
                        self.config["feedback_processing_interval_ms"] / 1000.0
                    )
                    continue

                # Process feedback events by priority
                priority_events = defaultdict(list)
                events_to_process = []

                # Collect events for this batch
                for _ in range(
                    min(self.config["batch_processing_size"], len(self.feedback_queue))
                ):
                    if self.feedback_queue:
                        event = self.feedback_queue.popleft()
                        priority_events[event.priority].append(event)
                        events_to_process.append(event)

                # Process in priority order
                for priority in [
                    FeedbackPriority.CRITICAL,
                    FeedbackPriority.HIGH,
                    FeedbackPriority.MEDIUM,
                    FeedbackPriority.LOW,
                    FeedbackPriority.BACKGROUND,
                ]:
                    if priority in priority_events:
                        await self._process_feedback_batch(priority_events[priority])

                # Clean up old optimizations
                await self._cleanup_expired_optimizations()

                # Trigger global optimization if needed
                await self._check_global_optimization_trigger()

            except Exception as e:
                self.logger.error(f"❌ Feedback processing loop error: {e}")
                await asyncio.sleep(1.0)  # Error recovery delay

    async def _process_feedback_batch(self, events: List[PerformanceFeedbackEvent]):
        """Process a batch of feedback events"""
        for event in events:
            start_time = datetime.utcnow()

            try:
                # Route to appropriate processors
                for feedback_type, processor in self.feedback_processors.items():
                    if (
                        event.feedback_type in processor.feedback_types
                        and processor.enabled
                    ):
                        await processor.processor_function([event])
                        event.processing_status = "completed"
                        event.propagation_results[processor.service_name] = "success"

                # Calculate processing latency
                processing_latency = (
                    datetime.utcnow() - start_time
                ).total_seconds() * 1000

                # Update analytics
                for processor_name, processor in self.feedback_processors.items():
                    if event.feedback_type in processor.feedback_types:
                        self.feedback_analytics["processing_latencies"][
                            processor_name
                        ].append(processing_latency)

            except Exception as e:
                self.logger.error(
                    f"❌ Failed to process feedback event {event.event_id}: {e}"
                )
                event.processing_status = "failed"

    async def _process_pattern_feedback(self, events: List[PerformanceFeedbackEvent]):
        """Process pattern effectiveness feedback"""
        for event in events:
            # Extract pattern effectiveness metrics
            effectiveness_score = event.performance_metrics.get(
                "effectiveness_score", 0.5
            )
            chemistry_score = event.performance_metrics.get("chemistry_score", 0.5)

            # Get service instance for pattern selection
            pattern_service_ref = self.service_registry.get(
                "nway_pattern_selection_service"
            )
            if pattern_service_ref:
                pattern_service = pattern_service_ref()
                if pattern_service and hasattr(
                    pattern_service, "record_pattern_effectiveness"
                ):
                    # Extract pattern information from context
                    selected_patterns = event.context.get("selected_patterns", [])
                    framework_type = event.context.get("framework_type", "unknown")
                    domain = event.context.get("domain", "general")

                    for pattern_id in selected_patterns:
                        await pattern_service.record_pattern_effectiveness(
                            pattern_id=pattern_id,
                            effectiveness_score=effectiveness_score,
                            context=event.context,
                            framework_type=framework_type,
                            domain=domain,
                            chemistry_score=chemistry_score,
                            user_feedback_score=event.performance_metrics.get(
                                "user_satisfaction", None
                            ),
                            analysis_quality_score=event.performance_metrics.get(
                                "analysis_quality", None
                            ),
                        )

    async def _process_consultant_feedback(
        self, events: List[PerformanceFeedbackEvent]
    ):
        """Process consultant performance feedback"""
        for event in events:
            consultant_service_ref = self.service_registry.get(
                "consultant_performance_tracking_service"
            )
            if consultant_service_ref:
                consultant_service = consultant_service_ref()
                if consultant_service and hasattr(
                    consultant_service, "record_consultant_selection"
                ):
                    # Record consultant performance feedback
                    consultant_types = event.context.get("consultant_types", [])
                    framework_type = event.context.get("framework_type", "unknown")
                    domain = event.context.get("domain", "general")
                    chemistry_score = event.performance_metrics.get(
                        "chemistry_score", 0.5
                    )

                    for consultant_id in consultant_types:
                        await consultant_service.record_consultant_selection(
                            consultant_id=consultant_id,
                            framework_type=framework_type,
                            domain=domain,
                            complexity=event.context.get("complexity", "medium"),
                            chemistry_score=chemistry_score,
                            selected_patterns=event.context.get(
                                "selected_patterns", []
                            ),
                            team_composition=consultant_types,
                            context=event.context,
                        )

    async def _process_chemistry_feedback(self, events: List[PerformanceFeedbackEvent]):
        """Process chemistry optimization feedback"""
        for event in events:
            chemistry_service_ref = self.service_registry.get(
                "cognitive_chemistry_engine"
            )
            if chemistry_service_ref:
                chemistry_service = chemistry_service_ref()
                if chemistry_service and hasattr(
                    chemistry_service, "record_chemistry_feedback"
                ):
                    chemistry_score = event.performance_metrics.get(
                        "chemistry_score", 0.5
                    )
                    effectiveness_score = event.performance_metrics.get(
                        "effectiveness_score", 0.5
                    )

                    # Record chemistry feedback (assuming such method exists)
                    # This would need to be implemented in the chemistry engine
                    pass

    async def _process_weight_feedback(self, events: List[PerformanceFeedbackEvent]):
        """Process weight optimization feedback"""
        for event in events:
            weighting_service_ref = self.service_registry.get(
                "contextual_weighting_system"
            )
            if weighting_service_ref:
                weighting_service = weighting_service_ref()
                if weighting_service and hasattr(
                    weighting_service, "record_performance_feedback"
                ):
                    # Determine dimension from context
                    from src.services.selection.contextual_weighting_system import (
                        WeightingDimension,
                    )

                    dimension = WeightingDimension.PATTERN_SELECTION  # Default
                    if "consultant" in event.source_service:
                        dimension = WeightingDimension.CONSULTANT_SELECTION
                    elif "chemistry" in event.source_service:
                        dimension = WeightingDimension.CHEMISTRY_OPTIMIZATION

                    weights_used = event.additional_data.get("weights_used", {})
                    performance_score = event.performance_metrics.get(
                        "overall_score", 0.5
                    )

                    await weighting_service.record_performance_feedback(
                        dimension=dimension,
                        context=event.context,
                        weights_used=weights_used,
                        performance_score=performance_score,
                        additional_metrics=event.performance_metrics,
                    )

    async def _process_system_feedback(self, events: List[PerformanceFeedbackEvent]):
        """Process system-wide performance feedback"""
        for event in events:
            # Update system health tracking
            overall_score = event.performance_metrics.get("overall_score", 0.5)
            self.feedback_analytics["system_health_scores"].append(overall_score)

            # Trigger optimization if performance drops significantly
            if overall_score < self.config["performance_threshold_for_learning"]:
                self.feedback_analytics["optimization_triggers"]["low_performance"] += 1

                # Consider triggering global optimization
                if len(self.feedback_analytics["system_health_scores"]) >= 10:
                    recent_average = statistics.mean(
                        list(self.feedback_analytics["system_health_scores"])[-10:]
                    )
                    if (
                        recent_average
                        < self.config["performance_threshold_for_learning"]
                    ):
                        asyncio.create_task(self.trigger_global_optimization())

    async def _detect_optimization_conflicts(
        self, service_name: str, optimization_type: str
    ) -> List[Dict[str, Any]]:
        """Detect potential optimization conflicts"""
        conflicts = []
        current_time = datetime.utcnow()

        # Check for active optimizations that might conflict
        for (
            active_key,
            completion_time,
        ) in self.learning_coordination.active_optimizations.items():
            if completion_time < current_time:
                continue  # Expired optimization

            active_service, active_type = active_key.split("_", 1)

            # Check for direct conflicts (same service)
            if active_service == service_name:
                conflicts.append(
                    {
                        "type": "direct_conflict",
                        "conflicting_service": active_service,
                        "conflicting_optimization": active_type,
                        "completion_time": completion_time.isoformat(),
                    }
                )

            # Check for dependency conflicts
            service_deps = self.service_dependencies.get(service_name, [])
            if active_service in service_deps:
                conflicts.append(
                    {
                        "type": "dependency_conflict",
                        "conflicting_service": active_service,
                        "conflicting_optimization": active_type,
                        "completion_time": completion_time.isoformat(),
                    }
                )

        return conflicts

    async def _adapt_learning_rates(self, service_name: str, optimization_type: str):
        """Adapt learning rates based on recent performance"""
        # Simple adaptive learning rate adjustment
        recent_performance = self._get_recent_performance_for_service(service_name)

        if recent_performance:
            if recent_performance > 0.8:  # High performance - reduce learning rate
                self.learning_coordination.learning_rates[service_name] = 0.01
            elif recent_performance < 0.5:  # Low performance - increase learning rate
                self.learning_coordination.learning_rates[service_name] = 0.1
            else:  # Medium performance - standard learning rate
                self.learning_coordination.learning_rates[service_name] = 0.05

    def _get_recent_performance_for_service(self, service_name: str) -> Optional[float]:
        """Get recent performance score for a service"""
        recent_events = []
        cutoff_time = datetime.utcnow() - timedelta(hours=1)

        for feedback_type, events in self.feedback_history.items():
            for event in events:
                if (
                    event.source_service == service_name
                    and event.timestamp >= cutoff_time
                    and "overall_score" in event.performance_metrics
                ):
                    recent_events.append(event.performance_metrics["overall_score"])

        return statistics.mean(recent_events) if recent_events else None

    async def _cleanup_expired_optimizations(self):
        """Clean up expired optimizations from coordination state"""
        current_time = datetime.utcnow()
        expired_keys = [
            key
            for key, completion_time in self.learning_coordination.active_optimizations.items()
            if completion_time < current_time
        ]

        for key in expired_keys:
            del self.learning_coordination.active_optimizations[key]

    async def _check_global_optimization_trigger(self):
        """Check if global optimization should be triggered"""
        if not self.config["cross_service_optimization_enabled"]:
            return

        last_optimization = self.learning_coordination.last_global_optimization
        optimization_interval = timedelta(
            minutes=self.config["global_optimization_interval_minutes"]
        )

        if (
            not last_optimization
            or datetime.utcnow() - last_optimization >= optimization_interval
        ):

            # Check if system performance indicates need for optimization
            if len(self.feedback_analytics["system_health_scores"]) >= 20:
                recent_health = statistics.mean(
                    list(self.feedback_analytics["system_health_scores"])[-20:]
                )
                if recent_health < 0.7:  # Health threshold for global optimization
                    asyncio.create_task(self.trigger_global_optimization())

    def _calculate_optimization_order(self) -> List[str]:
        """Calculate optimal order for service optimization based on dependencies"""
        # Simple topological sort based on dependencies
        visited = set()
        result = []

        def visit(service_name: str):
            if service_name in visited:
                return
            visited.add(service_name)

            # Visit dependencies first
            for dep in self.service_dependencies.get(service_name, []):
                if dep in self.service_registry:
                    visit(dep)

            result.append(service_name)

        # Visit all registered services
        for service_name in self.service_registry.keys():
            visit(service_name)

        return result

    def _calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health metrics"""
        health_scores = list(self.feedback_analytics["system_health_scores"])

        if not health_scores:
            return {"status": "unknown", "score": 0.5}

        recent_scores = (
            health_scores[-20:] if len(health_scores) >= 20 else health_scores
        )
        average_health = statistics.mean(recent_scores)

        if average_health >= 0.8:
            status = "excellent"
        elif average_health >= 0.7:
            status = "good"
        elif average_health >= 0.6:
            status = "fair"
        elif average_health >= 0.5:
            status = "poor"
        else:
            status = "critical"

        return {
            "status": status,
            "score": average_health,
            "trend": self._calculate_trend(recent_scores),
            "sample_size": len(recent_scores),
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from list of values"""
        if len(values) < 3:
            return "insufficient_data"

        # Compare first third vs last third
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

    async def get_service_health(self) -> Dict[str, Any]:
        """Return service health and status"""
        return {
            "service_name": "PerformanceFeedbackLoopIntegration",
            "status": "healthy",
            "version": "sprint_2",
            "capabilities": [
                "unified_feedback_orchestration",
                "cross_service_optimization_coordination",
                "learning_conflict_resolution",
                "performance_event_streaming",
                "global_optimization_triggering",
                "feedback_analytics_generation",
            ],
            "integration_statistics": {
                "registered_services": len(self.service_registry),
                "active_processors": len(
                    [p for p in self.feedback_processors.values() if p.enabled]
                ),
                "feedback_queue_size": len(self.feedback_queue),
                "total_events_processed": self.feedback_analytics[
                    "total_events_processed"
                ],
                "active_optimizations": len(
                    self.learning_coordination.active_optimizations
                ),
                "system_learning_enabled": self.learning_coordination.system_learning_enabled,
            },
            "configuration": self.config,
            "sprint_2_targets": {
                "feedback_loop_integration": "active",
                "cross_service_coordination": "active",
                "learning_optimization": "coordinated",
            },
            "last_health_check": datetime.utcnow().isoformat(),
        }


# Global service instance for dependency injection
_feedback_loop_integration: Optional[PerformanceFeedbackLoopIntegration] = None


def get_feedback_loop_integration() -> PerformanceFeedbackLoopIntegration:
    """Get or create global feedback loop integration service instance"""
    global _feedback_loop_integration

    if _feedback_loop_integration is None:
        _feedback_loop_integration = PerformanceFeedbackLoopIntegration()

    return _feedback_loop_integration
