"""
Test Flywheel Manager - Management Module Stub
==============================================

Provides the same interface as the main test_flywheel_manager for imports
that use the management.test_flywheel_manager path.
"""

from src.engine.flywheel.test_flywheel_manager import (
    get_test_flywheel_manager,
    capture_test_failure,
    capture_test_success,
    get_flywheel_status,
    start_flywheel_monitoring,
    stop_flywheel_monitoring,
    TestFlywheelManagerStub,
    TestFlywheelManager,
    TestOutcome,
    LearningSignal,
    EnhancedTestResult,
)

__all__ = [
    "get_test_flywheel_manager",
    "capture_test_failure",
    "capture_test_success",
    "get_flywheel_status",
    "start_flywheel_monitoring",
    "stop_flywheel_monitoring",
    "TestFlywheelManagerStub",
    "TestFlywheelManager",
    "TestOutcome",
    "LearningSignal",
    "EnhancedTestResult",
]
