#!/usr/bin/env python3
"""
Phantom Workflow Detector with Flywheel Integration
Advanced detection system for phantom workflows based on UltraThink principles

Detects and prevents "phantom workflows" where phases appear to complete
successfully but actually perform no real work (0.00s execution times).
Integrates with the flywheel system to learn from detection patterns.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from uuid import UUID, uuid4

try:
    # Flywheel integration
    from src.engine.flywheel.test_flywheel_manager import (
        get_test_flywheel_manager,
        TestOutcome,
        LearningSignal,
    )
    from src.engine.flywheel.orchestration.continuous_learning_orchestrator import (
        get_continuous_learning_orchestrator,
        LearningTrigger,
    )

    # Context engineering
    from src.engine.core.context_engineering_optimizer import (
        get_context_engineering_optimizer,
    )

    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    print(f"Warning: Dependencies not available: {e}")

logger = logging.getLogger(__name__)


class PhantomType(Enum):
    """Types of phantom workflow patterns"""

    ZERO_EXECUTION = "zero_execution"  # 0.00s execution time
    SUSPICIOUSLY_FAST = "suspiciously_fast"  # <100ms for complex operations
    REPEATED_IDENTICAL = "repeated_identical"  # Identical timing patterns
    MISSING_SIDE_EFFECTS = "missing_side_effects"  # Expected side effects missing
    CACHED_STALE_RESULT = "cached_stale_result"  # Using stale cached results


class DetectionSeverity(Enum):
    """Severity levels for phantom detection"""

    LOW = "low"  # Suspicious but not critical
    MEDIUM = "medium"  # Likely phantom workflow
    HIGH = "high"  # Definite phantom workflow
    CRITICAL = "critical"  # System integrity at risk


@dataclass
class PhantomEvidence:
    """Evidence for phantom workflow detection"""

    evidence_type: str = ""
    description: str = ""
    confidence_score: float = 0.0  # 0.0-1.0
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PhantomDetection:
    """Complete phantom workflow detection"""

    detection_id: UUID = field(default_factory=uuid4)
    phantom_type: PhantomType = PhantomType.ZERO_EXECUTION
    severity: DetectionSeverity = DetectionSeverity.MEDIUM
    confidence: float = 0.0

    # Context
    workflow_phase: str = ""
    execution_time_ms: float = 0.0
    expected_min_time_ms: float = 100.0

    # Evidence
    evidence: List[PhantomEvidence] = field(default_factory=list)

    # System state
    timestamp: datetime = field(default_factory=datetime.utcnow)
    system_load: float = 0.0
    memory_usage: float = 0.0

    # Learning integration
    captured_for_learning: bool = False
    flywheel_session_id: Optional[UUID] = None


@dataclass
class PhantomPattern:
    """Pattern for phantom workflow detection"""

    pattern_id: str = ""
    name: str = ""
    description: str = ""

    # Detection criteria
    min_execution_time_ms: float = 100.0
    max_execution_time_ms: float = float("inf")
    expected_side_effects: List[str] = field(default_factory=list)

    # Pattern learning
    detection_count: int = 0
    false_positive_count: int = 0
    accuracy: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


class PhantomWorkflowDetector:
    """
    Advanced phantom workflow detector with flywheel integration.

    Based on UltraThink principles and Operation Synapse validation,
    this system detects when workflow phases complete too quickly to
    have performed actual work, learning from detection patterns.
    """

    def __init__(self):
        self.detection_history: List[PhantomDetection] = []
        self.patterns: Dict[str, PhantomPattern] = {}
        self.false_positive_feedback: List[Dict] = []

        # Flywheel integration
        if DEPENDENCIES_AVAILABLE:
            self.flywheel_manager = get_test_flywheel_manager()
            self.learning_orchestrator = get_continuous_learning_orchestrator()
            self.context_optimizer = get_context_engineering_optimizer()

        # Performance baselines (learned over time)
        self.execution_baselines: Dict[str, Dict[str, float]] = {
            "problem_structuring": {
                "min_time_ms": 3000.0,  # 3 seconds minimum
                "typical_time_ms": 15000.0,
                "max_reasonable_time_ms": 120000.0,
            },
            "hypothesis_generation": {
                "min_time_ms": 2000.0,  # 2 seconds minimum
                "typical_time_ms": 8000.0,
                "max_reasonable_time_ms": 60000.0,
            },
            "analysis_execution": {
                "min_time_ms": 5000.0,  # 5 seconds minimum
                "typical_time_ms": 25000.0,
                "max_reasonable_time_ms": 180000.0,
            },
            "synthesis_delivery": {
                "min_time_ms": 3000.0,  # 3 seconds minimum
                "typical_time_ms": 12000.0,
                "max_reasonable_time_ms": 90000.0,
            },
        }

        # Initialize patterns
        self._initialize_detection_patterns()

        logger.info("Phantom Workflow Detector initialized with flywheel integration")

    def _initialize_detection_patterns(self):
        """Initialize phantom detection patterns"""

        # Pattern 1: Zero execution time
        self.patterns["zero_execution"] = PhantomPattern(
            pattern_id="zero_execution",
            name="Zero Execution Time",
            description="Phase reports 0.00s execution time",
            min_execution_time_ms=0.0,
            max_execution_time_ms=10.0,
            expected_side_effects=["llm_calls", "processing_logs", "result_generation"],
        )

        # Pattern 2: Suspiciously fast complex operations
        self.patterns["fast_complex"] = PhantomPattern(
            pattern_id="fast_complex",
            name="Suspiciously Fast Complex Operation",
            description="Complex operation completes unreasonably quickly",
            min_execution_time_ms=0.0,
            max_execution_time_ms=100.0,
            expected_side_effects=[
                "model_selection",
                "reasoning_process",
                "validation",
            ],
        )

        # Pattern 3: Identical timing patterns
        self.patterns["identical_timing"] = PhantomPattern(
            pattern_id="identical_timing",
            name="Identical Timing Pattern",
            description="Multiple phases show identical execution times",
            expected_side_effects=["unique_processing", "different_complexity"],
        )

        # Pattern 4: Missing expected side effects
        self.patterns["missing_effects"] = PhantomPattern(
            pattern_id="missing_effects",
            name="Missing Expected Side Effects",
            description="Phase completes without expected system interactions",
            expected_side_effects=["api_calls", "file_operations", "state_changes"],
        )

    async def detect_phantom_workflow(
        self,
        workflow_phase: str,
        execution_time_ms: float,
        context: Dict[str, Any] = None,
        side_effects: Dict[str, Any] = None,
    ) -> Optional[PhantomDetection]:
        """
        Detect potential phantom workflow in a phase execution.

        Args:
            workflow_phase: Phase name (e.g., "problem_structuring")
            execution_time_ms: Reported execution time in milliseconds
            context: Additional context about the execution
            side_effects: Observable side effects of the execution
        """

        context = context or {}
        side_effects = side_effects or {}

        detection = None
        evidence = []

        # Get baseline expectations for this phase
        baseline = self.execution_baselines.get(workflow_phase, {})
        min_expected = baseline.get("min_time_ms", 1000.0)

        # Evidence 1: Zero or near-zero execution time
        if execution_time_ms < 10.0:
            evidence.append(
                PhantomEvidence(
                    evidence_type="zero_execution",
                    description=f"Execution time {execution_time_ms:.2f}ms is essentially zero",
                    confidence_score=0.95,
                    supporting_data={
                        "execution_time": execution_time_ms,
                        "threshold": 10.0,
                    },
                )
            )

        # Evidence 2: Suspiciously fast for complexity
        elif execution_time_ms < min_expected:
            complexity_score = context.get("complexity_score", 0.5)
            if complexity_score > 0.3:  # Medium+ complexity
                evidence.append(
                    PhantomEvidence(
                        evidence_type="suspiciously_fast",
                        description=f"Execution time {execution_time_ms:.2f}ms too fast for complexity {complexity_score:.2f}",
                        confidence_score=min(
                            0.9, (min_expected - execution_time_ms) / min_expected
                        ),
                        supporting_data={
                            "execution_time": execution_time_ms,
                            "min_expected": min_expected,
                            "complexity": complexity_score,
                        },
                    )
                )

        # Evidence 3: Missing expected side effects
        expected_effects = ["llm_api_calls", "processing_tokens", "model_selection"]
        missing_effects = [
            effect for effect in expected_effects if not side_effects.get(effect)
        ]

        if len(missing_effects) > 0:
            evidence.append(
                PhantomEvidence(
                    evidence_type="missing_side_effects",
                    description=f"Missing expected side effects: {missing_effects}",
                    confidence_score=len(missing_effects) * 0.3,
                    supporting_data={
                        "missing_effects": missing_effects,
                        "observed_effects": list(side_effects.keys()),
                    },
                )
            )

        # Evidence 4: Repeated identical timings
        recent_timings = [
            d.execution_time_ms
            for d in self.detection_history[-5:]
            if d.workflow_phase == workflow_phase
        ]
        if len(recent_timings) >= 3 and all(
            abs(t - execution_time_ms) < 1.0 for t in recent_timings
        ):
            evidence.append(
                PhantomEvidence(
                    evidence_type="identical_timing",
                    description=f"Identical timing pattern: {execution_time_ms:.2f}ms repeated",
                    confidence_score=0.8,
                    supporting_data={
                        "recent_timings": recent_timings,
                        "current_timing": execution_time_ms,
                    },
                )
            )

        # Evidence 5: Cached stale result indicators
        if context.get("cache_hit") and execution_time_ms < 50.0:
            cache_age_hours = context.get("cache_age_hours", 0)
            if cache_age_hours > 24:  # Stale cache
                evidence.append(
                    PhantomEvidence(
                        evidence_type="stale_cache",
                        description=f"Using stale cache result ({cache_age_hours:.1f} hours old)",
                        confidence_score=min(0.7, cache_age_hours / 48),
                        supporting_data={"cache_age_hours": cache_age_hours},
                    )
                )

        # If we have evidence, create detection
        if evidence:
            detection = PhantomDetection(
                workflow_phase=workflow_phase,
                execution_time_ms=execution_time_ms,
                expected_min_time_ms=min_expected,
                evidence=evidence,
            )

            # Determine phantom type and severity
            detection.phantom_type, detection.severity = self._classify_phantom(
                evidence
            )
            detection.confidence = self._calculate_confidence(evidence)

            # Add system context
            detection.system_load = context.get("system_load", 0.0)
            detection.memory_usage = context.get("memory_usage", 0.0)

            # Store detection
            self.detection_history.append(detection)

            # Capture for flywheel learning
            if DEPENDENCIES_AVAILABLE:
                await self._capture_for_flywheel(detection, context)

            logger.warning(
                f"Phantom workflow detected: {workflow_phase} - {detection.phantom_type.value} "
                f"(confidence: {detection.confidence:.2f})"
            )

        return detection

    def _classify_phantom(
        self, evidence: List[PhantomEvidence]
    ) -> Tuple[PhantomType, DetectionSeverity]:
        """Classify phantom type and severity based on evidence"""

        # Count evidence types
        evidence_counts = {}
        max_confidence = 0.0

        for e in evidence:
            evidence_counts[e.evidence_type] = (
                evidence_counts.get(e.evidence_type, 0) + 1
            )
            max_confidence = max(max_confidence, e.confidence_score)

        # Determine phantom type
        if "zero_execution" in evidence_counts:
            phantom_type = PhantomType.ZERO_EXECUTION
        elif "suspiciously_fast" in evidence_counts:
            phantom_type = PhantomType.SUSPICIOUSLY_FAST
        elif "identical_timing" in evidence_counts:
            phantom_type = PhantomType.REPEATED_IDENTICAL
        elif "missing_side_effects" in evidence_counts:
            phantom_type = PhantomType.MISSING_SIDE_EFFECTS
        elif "stale_cache" in evidence_counts:
            phantom_type = PhantomType.CACHED_STALE_RESULT
        else:
            phantom_type = PhantomType.SUSPICIOUSLY_FAST

        # Determine severity
        if max_confidence > 0.9:
            severity = DetectionSeverity.CRITICAL
        elif max_confidence > 0.7:
            severity = DetectionSeverity.HIGH
        elif max_confidence > 0.5:
            severity = DetectionSeverity.MEDIUM
        else:
            severity = DetectionSeverity.LOW

        return phantom_type, severity

    def _calculate_confidence(self, evidence: List[PhantomEvidence]) -> float:
        """Calculate overall confidence in phantom detection"""
        if not evidence:
            return 0.0

        # Weighted average of evidence confidence
        weights = {
            "zero_execution": 1.0,
            "suspiciously_fast": 0.8,
            "missing_side_effects": 0.6,
            "identical_timing": 0.7,
            "stale_cache": 0.5,
        }

        total_weight = 0.0
        weighted_confidence = 0.0

        for e in evidence:
            weight = weights.get(e.evidence_type, 0.5)
            total_weight += weight
            weighted_confidence += e.confidence_score * weight

        return weighted_confidence / total_weight if total_weight > 0 else 0.0

    async def _capture_for_flywheel(
        self, detection: PhantomDetection, context: Dict[str, Any]
    ):
        """Capture phantom detection for flywheel learning"""
        try:
            # Create test result for phantom detection
            test_data = {
                "test_name": f"phantom_detection_{detection.workflow_phase}",
                "test_category": "phantom_workflow",
                "outcome": "failed",  # Phantom is a failure
                "execution_time_ms": detection.execution_time_ms,
                "learning_signals": [
                    "phantom_workflow",
                    detection.phantom_type.value,
                    f"confidence_{detection.confidence:.2f}",
                ],
                "metadata": {
                    "phantom_type": detection.phantom_type.value,
                    "severity": detection.severity.value,
                    "confidence": detection.confidence,
                    "evidence_count": len(detection.evidence),
                    "expected_min_time": detection.expected_min_time_ms,
                },
            }

            # Capture via flywheel manager
            await self.flywheel_manager.capture_test_result_enhanced(test_data)
            detection.captured_for_learning = True

            logger.info(
                f"Captured phantom detection for flywheel learning: {detection.detection_id}"
            )

        except Exception as e:
            logger.error(f"Failed to capture phantom detection for flywheel: {e}")

    async def validate_phase_execution(
        self,
        workflow_phase: str,
        start_time: float,
        end_time: float,
        context: Dict[str, Any] = None,
        side_effects: Dict[str, Any] = None,
    ) -> Tuple[bool, Optional[PhantomDetection]]:
        """
        Validate phase execution and detect phantoms.

        Returns:
            (is_valid, phantom_detection)
        """

        execution_time_ms = (end_time - start_time) * 1000

        # Record execution in context engineering optimizer
        if DEPENDENCIES_AVAILABLE and self.context_optimizer:
            # This would be integrated with context tracking
            pass

        # Detect phantom workflow
        phantom = await self.detect_phantom_workflow(
            workflow_phase, execution_time_ms, context, side_effects
        )

        # Determine if execution is valid
        is_valid = phantom is None or phantom.severity in [
            DetectionSeverity.LOW,
            DetectionSeverity.MEDIUM,
        ]

        if phantom:
            # Update pattern statistics
            pattern_key = phantom.phantom_type.value
            if pattern_key in self.patterns:
                self.patterns[pattern_key].detection_count += 1

        return is_valid, phantom

    def provide_false_positive_feedback(
        self, detection_id: UUID, is_false_positive: bool, explanation: str = ""
    ):
        """Provide feedback on phantom detection accuracy"""

        # Find the detection
        detection = next(
            (d for d in self.detection_history if d.detection_id == detection_id), None
        )
        if not detection:
            logger.warning(f"Detection {detection_id} not found for feedback")
            return

        # Record feedback
        feedback = {
            "detection_id": str(detection_id),
            "phantom_type": detection.phantom_type.value,
            "is_false_positive": is_false_positive,
            "explanation": explanation,
            "timestamp": datetime.utcnow().isoformat(),
        }

        self.false_positive_feedback.append(feedback)

        # Update pattern accuracy
        pattern_key = detection.phantom_type.value
        if pattern_key in self.patterns:
            pattern = self.patterns[pattern_key]
            if is_false_positive:
                pattern.false_positive_count += 1

            # Recalculate accuracy
            total_detections = pattern.detection_count
            if total_detections > 0:
                pattern.accuracy = 1.0 - (
                    pattern.false_positive_count / total_detections
                )

        logger.info(
            f"Recorded feedback for detection {detection_id}: false_positive={is_false_positive}"
        )

    def get_detection_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get summary of phantom detections"""

        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        recent_detections = [
            d for d in self.detection_history if d.timestamp >= cutoff_time
        ]

        # Count by type and severity
        type_counts = {}
        severity_counts = {}

        for detection in recent_detections:
            phantom_type = detection.phantom_type.value
            severity = detection.severity.value

            type_counts[phantom_type] = type_counts.get(phantom_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        # Calculate average confidence
        avg_confidence = (
            sum(d.confidence for d in recent_detections) / len(recent_detections)
            if recent_detections
            else 0.0
        )

        # Pattern accuracy
        pattern_accuracy = {
            pattern_id: pattern.accuracy
            for pattern_id, pattern in self.patterns.items()
        }

        return {
            "total_detections": len(recent_detections),
            "detections_by_type": type_counts,
            "detections_by_severity": severity_counts,
            "average_confidence": avg_confidence,
            "pattern_accuracy": pattern_accuracy,
            "false_positive_rate": len(self.false_positive_feedback)
            / max(len(self.detection_history), 1),
            "time_window_hours": hours_back,
        }

    async def run_diagnostic_scan(self) -> Dict[str, Any]:
        """Run diagnostic scan of the system for phantom vulnerabilities"""

        diagnostic_results = {
            "scan_timestamp": datetime.utcnow().isoformat(),
            "system_health": "healthy",
            "vulnerabilities": [],
            "recommendations": [],
        }

        # Check for patterns indicating systematic phantom issues
        recent_detections = self.detection_history[-20:]  # Last 20 detections

        if len(recent_detections) > 10:
            # High detection rate
            diagnostic_results["system_health"] = "concerning"
            diagnostic_results["vulnerabilities"].append("High phantom detection rate")
            diagnostic_results["recommendations"].append(
                "Review orchestrator caching logic"
            )

        # Check for repeated phantom types
        type_counts = {}
        for detection in recent_detections:
            phantom_type = detection.phantom_type.value
            type_counts[phantom_type] = type_counts.get(phantom_type, 0) + 1

        for phantom_type, count in type_counts.items():
            if count > 5:
                diagnostic_results["vulnerabilities"].append(
                    f"Repeated {phantom_type} phantoms ({count} occurrences)"
                )

                if phantom_type == "zero_execution":
                    diagnostic_results["recommendations"].append(
                        "Check workflow orchestrator execution paths"
                    )
                elif phantom_type == "cached_stale_result":
                    diagnostic_results["recommendations"].append(
                        "Review cache invalidation policies"
                    )

        # Check pattern accuracy
        low_accuracy_patterns = [
            pattern_id
            for pattern_id, pattern in self.patterns.items()
            if pattern.accuracy < 0.7 and pattern.detection_count > 5
        ]

        if low_accuracy_patterns:
            diagnostic_results["vulnerabilities"].append(
                f"Low accuracy patterns: {low_accuracy_patterns}"
            )
            diagnostic_results["recommendations"].append(
                "Retrain detection patterns with more feedback"
            )

        return diagnostic_results


# Singleton instance
_phantom_detector = None


def get_phantom_workflow_detector() -> PhantomWorkflowDetector:
    """Get singleton phantom workflow detector"""
    global _phantom_detector
    if _phantom_detector is None:
        _phantom_detector = PhantomWorkflowDetector()
    return _phantom_detector


# Context manager for phase validation
class PhantomValidatedPhase:
    """Context manager for validating phase execution"""

    def __init__(self, phase_name: str, context: Dict[str, Any] = None):
        self.phase_name = phase_name
        self.context = context or {}
        self.start_time = 0.0
        self.detector = get_phantom_workflow_detector()

    async def __aenter__(self):
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()

        # Collect side effects
        side_effects = {
            "completion_status": "completed" if exc_type is None else "failed",
            "exception_occurred": exc_type is not None,
        }

        # Validate execution
        is_valid, phantom = await self.detector.validate_phase_execution(
            self.phase_name, self.start_time, end_time, self.context, side_effects
        )

        if not is_valid and phantom:
            logger.warning(
                f"Phase {self.phase_name} failed phantom validation: {phantom.phantom_type.value}"
            )

            # In production, might raise an exception or trigger corrective action
            if phantom.severity == DetectionSeverity.CRITICAL:
                logger.critical(
                    f"Critical phantom workflow detected in {self.phase_name}"
                )


async def main():
    """Demo of phantom workflow detector"""
    print("🚨 Phantom Workflow Detector Demo")
    print("=" * 50)

    detector = get_phantom_workflow_detector()

    # Test 1: Normal execution (should not detect phantom)
    print("Test 1: Normal execution")
    phantom1 = await detector.detect_phantom_workflow(
        "problem_structuring",
        15000.0,  # 15 seconds - normal
        {"complexity_score": 0.7},
        {"llm_api_calls": 3, "processing_tokens": 1500},
    )
    print(f"  Phantom detected: {phantom1 is not None}")

    # Test 2: Zero execution time (should detect phantom)
    print("Test 2: Zero execution time")
    phantom2 = await detector.detect_phantom_workflow(
        "analysis_execution",
        0.0,  # 0 seconds - suspicious!
        {"complexity_score": 0.8},
        {},
    )
    print(f"  Phantom detected: {phantom2 is not None}")
    if phantom2:
        print(f"    Type: {phantom2.phantom_type.value}")
        print(f"    Severity: {phantom2.severity.value}")
        print(f"    Confidence: {phantom2.confidence:.2f}")

    # Test 3: Suspiciously fast complex operation
    print("Test 3: Suspiciously fast complex operation")
    phantom3 = await detector.detect_phantom_workflow(
        "hypothesis_generation",
        50.0,  # 50ms for complex operation
        {"complexity_score": 0.9},
        {"llm_api_calls": 0},  # Missing expected side effects
    )
    print(f"  Phantom detected: {phantom3 is not None}")
    if phantom3:
        print(f"    Evidence count: {len(phantom3.evidence)}")

    # Test phase validation context manager
    print("Test 4: Phase validation context manager")
    try:
        async with PhantomValidatedPhase("synthesis_delivery", {"test": True}):
            # Simulate very fast execution (phantom)
            await asyncio.sleep(0.001)
        print("  Phase completed (may have phantom)")
    except Exception as e:
        print(f"  Phase failed: {e}")

    # Get detection summary
    summary = detector.get_detection_summary()
    print("\nDetection Summary:")
    print(f"  Total detections: {summary['total_detections']}")
    print(f"  Average confidence: {summary['average_confidence']:.2f}")

    # Run diagnostic scan
    diagnostic = await detector.run_diagnostic_scan()
    print("\nDiagnostic Scan:")
    print(f"  System health: {diagnostic['system_health']}")
    print(f"  Vulnerabilities: {len(diagnostic['vulnerabilities'])}")
    print(f"  Recommendations: {len(diagnostic['recommendations'])}")


if __name__ == "__main__":
    asyncio.run(main())
