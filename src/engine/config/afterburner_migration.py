#!/usr/bin/env python3
"""
Afterburner Migration Configuration
Provides feature flags and gradual rollout control for the DeepSeek-First architecture
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum


class MigrationPhase(Enum):
    """Migration phases for gradual Afterburner rollout"""

    DISABLED = "disabled"  # Legacy mode - all components use direct LLM calls
    TESTING = "testing"  # Afterburner enabled for non-critical paths only
    PARTIAL = "partial"  # Afterburner enabled for 50% of requests
    MAJORITY = "majority"  # Afterburner enabled for 90% of requests
    FULL = "full"  # Afterburner enabled for all requests
    ENFORCED = "enforced"  # Afterburner required - legacy paths disabled


class ComponentMigrationStatus(Enum):
    """Migration status for individual components"""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    VERIFIED = "verified"


class AfterburnerMigrationConfig:
    """
    Central configuration for Afterburner optimization migration
    Controls the gradual rollout of the DeepSeek-First architecture
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Load configuration from environment or defaults
        self.phase = self._get_migration_phase()
        self.component_overrides = self._load_component_overrides()
        self.rollout_percentage = self._get_rollout_percentage()
        self.critical_path_protection = self._get_critical_path_protection()

        # Component migration tracking
        self.component_status = {
            "model_manager": ComponentMigrationStatus.COMPLETED,  # We just migrated this
            "assumption_challenger": ComponentMigrationStatus.COMPLETED,
            "hmw_generator": ComponentMigrationStatus.COMPLETED,
            "reasoning_synthesizer": ComponentMigrationStatus.COMPLETED,
            "contradiction_detector": ComponentMigrationStatus.COMPLETED,
            "query_clarification_engine": ComponentMigrationStatus.COMPLETED,
            "internal_challenger_system": ComponentMigrationStatus.COMPLETED,
            "cognitive_auditor": ComponentMigrationStatus.COMPLETED,
            "transparency_engine": ComponentMigrationStatus.COMPLETED,
            "research_orchestrator": ComponentMigrationStatus.COMPLETED,
        }

        # Performance metrics for migration monitoring
        self.metrics = {
            "afterburner_calls": 0,
            "legacy_calls": 0,
            "cost_savings_usd": 0.0,
            "average_response_time_ms": 0.0,
            "fallback_triggers": 0,
            "errors": 0,
        }

        self.logger.info(
            f"🚀 Afterburner Migration Config initialized: "
            f"Phase={self.phase.value}, Rollout={self.rollout_percentage}%"
        )

    def _get_migration_phase(self) -> MigrationPhase:
        """Get current migration phase from environment or config"""
        phase_str = os.getenv("AFTERBURNER_MIGRATION_PHASE", "testing").lower()

        try:
            return MigrationPhase(phase_str)
        except ValueError:
            self.logger.warning(
                f"Invalid migration phase: {phase_str}, defaulting to TESTING"
            )
            return MigrationPhase.TESTING

    def _get_rollout_percentage(self) -> int:
        """Get rollout percentage based on migration phase"""
        phase_percentages = {
            MigrationPhase.DISABLED: 0,
            MigrationPhase.TESTING: 10,
            MigrationPhase.PARTIAL: 50,
            MigrationPhase.MAJORITY: 90,
            MigrationPhase.FULL: 100,
            MigrationPhase.ENFORCED: 100,
        }

        # Allow environment override
        env_percentage = os.getenv("AFTERBURNER_ROLLOUT_PERCENTAGE")
        if env_percentage:
            try:
                return max(0, min(100, int(env_percentage)))
            except ValueError:
                pass

        return phase_percentages.get(self.phase, 10)

    def _load_component_overrides(self) -> Dict[str, bool]:
        """Load component-specific overrides from environment or config file"""
        overrides = {}

        # Check for environment variable overrides
        env_overrides = os.getenv("AFTERBURNER_COMPONENT_OVERRIDES")
        if env_overrides:
            try:
                overrides = json.loads(env_overrides)
            except json.JSONDecodeError:
                self.logger.warning("Invalid AFTERBURNER_COMPONENT_OVERRIDES JSON")

        # Check for config file
        config_path = os.getenv("AFTERBURNER_CONFIG_PATH", "config/afterburner.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    file_config = json.load(f)
                    overrides.update(file_config.get("component_overrides", {}))
            except Exception as e:
                self.logger.warning(f"Failed to load Afterburner config file: {e}")

        return overrides

    def _get_critical_path_protection(self) -> bool:
        """Determine if critical paths should be protected from migration"""
        # In early phases, protect critical paths
        if self.phase in [MigrationPhase.TESTING, MigrationPhase.PARTIAL]:
            return True

        # Allow environment override
        return os.getenv("AFTERBURNER_PROTECT_CRITICAL", "false").lower() == "true"

    def should_use_afterburner(
        self, component: str, task_type: Optional[str] = None, is_critical: bool = False
    ) -> bool:
        """
        Determine if Afterburner should be used for a specific request

        Args:
            component: Name of the component making the request
            task_type: Type of task being performed
            is_critical: Whether this is a critical path operation

        Returns:
            True if Afterburner should be used, False for legacy path
        """

        # Phase-based checks
        if self.phase == MigrationPhase.DISABLED:
            return False

        if self.phase == MigrationPhase.ENFORCED:
            return True

        # Check component overrides
        if component in self.component_overrides:
            return self.component_overrides[component]

        # Check component migration status
        status = self.component_status.get(
            component, ComponentMigrationStatus.NOT_STARTED
        )
        if status == ComponentMigrationStatus.NOT_STARTED:
            return False

        # Protect critical paths if configured
        if is_critical and self.critical_path_protection:
            self.logger.info(
                f"🛡️ Critical path protection: Using legacy for {component}/{task_type}"
            )
            return False

        # Apply rollout percentage
        import random

        use_afterburner = random.randint(1, 100) <= self.rollout_percentage

        # Track metrics
        if use_afterburner:
            self.metrics["afterburner_calls"] += 1
        else:
            self.metrics["legacy_calls"] += 1

        return use_afterburner

    def mark_component_migrated(
        self,
        component: str,
        status: ComponentMigrationStatus = ComponentMigrationStatus.COMPLETED,
    ):
        """Mark a component as migrated to Afterburner"""
        self.component_status[component] = status
        self.logger.info(f"✅ Component {component} migration status: {status.value}")

    def get_migration_report(self) -> Dict[str, Any]:
        """Generate migration status report"""
        total_components = len(self.component_status)
        migrated_components = sum(
            1
            for status in self.component_status.values()
            if status
            in [ComponentMigrationStatus.COMPLETED, ComponentMigrationStatus.VERIFIED]
        )

        return {
            "migration_phase": self.phase.value,
            "rollout_percentage": self.rollout_percentage,
            "critical_path_protection": self.critical_path_protection,
            "components": {
                "total": total_components,
                "migrated": migrated_components,
                "percentage": (
                    (migrated_components / total_components * 100)
                    if total_components > 0
                    else 0
                ),
                "status": {k: v.value for k, v in self.component_status.items()},
            },
            "metrics": self.metrics,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def update_metrics(
        self,
        afterburner_result: Optional[Dict[str, Any]] = None,
        legacy_result: Optional[Dict[str, Any]] = None,
    ):
        """Update migration metrics based on call results"""
        if afterburner_result:
            if "cost_usd" in afterburner_result:
                # Calculate savings vs Claude baseline
                claude_equivalent_cost = (
                    afterburner_result.get("tokens_used", 0) * 0.000015
                )
                savings = claude_equivalent_cost - afterburner_result["cost_usd"]
                self.metrics["cost_savings_usd"] += savings

            if "response_time_ms" in afterburner_result:
                # Update rolling average
                current_avg = self.metrics["average_response_time_ms"]
                total_calls = self.metrics["afterburner_calls"]
                new_avg = (
                    (current_avg * (total_calls - 1))
                    + afterburner_result["response_time_ms"]
                ) / total_calls
                self.metrics["average_response_time_ms"] = new_avg

            if afterburner_result.get("fallback_triggered"):
                self.metrics["fallback_triggers"] += 1

        if afterburner_result and afterburner_result.get("error"):
            self.metrics["errors"] += 1

    def should_rollback(self) -> bool:
        """Determine if migration should be rolled back based on metrics"""
        # Calculate error rate
        total_calls = self.metrics["afterburner_calls"] + self.metrics["legacy_calls"]
        if total_calls > 100:  # Need sufficient sample size
            error_rate = (
                self.metrics["errors"] / self.metrics["afterburner_calls"]
                if self.metrics["afterburner_calls"] > 0
                else 0
            )
            fallback_rate = (
                self.metrics["fallback_triggers"] / self.metrics["afterburner_calls"]
                if self.metrics["afterburner_calls"] > 0
                else 0
            )

            # Rollback if error rate > 5% or fallback rate > 20%
            if error_rate > 0.05 or fallback_rate > 0.20:
                self.logger.error(
                    f"🚨 Migration rollback triggered: "
                    f"Error rate={error_rate:.2%}, Fallback rate={fallback_rate:.2%}"
                )
                return True

        return False


# Global migration config instance
_migration_config = None


def get_afterburner_migration_config() -> AfterburnerMigrationConfig:
    """Get the global Afterburner migration configuration"""
    global _migration_config
    if _migration_config is None:
        _migration_config = AfterburnerMigrationConfig()
    return _migration_config


# Helper functions for component integration
def should_use_afterburner_for_component(
    component: str, task_type: Optional[str] = None, is_critical: bool = False
) -> bool:
    """
    Quick helper to check if a component should use Afterburner

    Example usage:
        if should_use_afterburner_for_component("assumption_challenger", "challenge_generation"):
            # Use UnifiedLLMAdapter
        else:
            # Use legacy ClaudeClient
    """
    config = get_afterburner_migration_config()
    return config.should_use_afterburner(component, task_type, is_critical)


def report_afterburner_result(
    component: str,
    success: bool,
    cost_usd: Optional[float] = None,
    response_time_ms: Optional[int] = None,
    fallback_triggered: bool = False,
):
    """
    Report Afterburner call result for metrics tracking

    Example usage:
        report_afterburner_result(
            "model_manager",
            success=True,
            cost_usd=0.001,
            response_time_ms=1500,
            fallback_triggered=False
        )
    """
    config = get_afterburner_migration_config()
    result = {"component": component, "success": success, "error": not success}

    if cost_usd is not None:
        result["cost_usd"] = cost_usd
    if response_time_ms is not None:
        result["response_time_ms"] = response_time_ms
    if fallback_triggered:
        result["fallback_triggered"] = fallback_triggered

    config.update_metrics(afterburner_result=result if success else None)


def get_migration_status() -> Dict[str, Any]:
    """Get current migration status report"""
    config = get_afterburner_migration_config()
    return config.get_migration_report()


# Legacy compatibility export - for imports expecting AFTERBURNER_CONFIG
AFTERBURNER_CONFIG = get_afterburner_migration_config()
