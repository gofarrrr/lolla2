#!/usr/bin/env python3
"""
METIS Real-Time Monitoring and Alerting System
E004: Real-time monitoring with alerting and automated responses

Provides real-time system monitoring, intelligent alerting,
and automated response capabilities for production deployments.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from uuid import UUID, uuid4
from collections import deque
import numpy as np

try:
    from src.monitoring.performance_validator import (
        PerformanceValidator,
        PerformanceMetricType,
        AlertSeverity,
        ValidationStatus,
        PerformanceAlert,
        get_performance_validator,
    )
    from src.core.enhanced_event_bus import (
        EnhancedKafkaEventBus as MetisEventBus,
        CloudEvent,
    )
    from src.core.state_management import DistributedStateManager, StateType

    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    print("Warning: Monitoring components not available, using mock interfaces")

    # Mock implementations
    class MockPerformanceValidator:
        async def validate_system_performance(self):
            return {"overall_status": "passing"}

    class MockStateManager:
        def __init__(self):
            self.data = {}

        async def set_state(self, key, value, state_type=None):
            self.data[key] = value

        async def get_state(self, key):
            return self.data.get(key)

    PerformanceValidator = MockPerformanceValidator
    DistributedStateManager = MockStateManager
    StateType = None


class MonitoringMode(str, Enum):
    """Real-time monitoring modes"""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    MAINTENANCE = "maintenance"


class AlertChannel(str, Enum):
    """Alert notification channels"""

    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    PAGERDUTY = "pagerduty"
    LOG = "log"


class AutomatedAction(str, Enum):
    """Automated response actions"""

    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    RESTART_SERVICE = "restart_service"
    ENABLE_CIRCUIT_BREAKER = "enable_circuit_breaker"
    THROTTLE_REQUESTS = "throttle_requests"
    SWITCH_TO_FALLBACK = "switch_to_fallback"
    NOTIFY_ONCALL = "notify_oncall"


@dataclass
class MonitoringRule:
    """Real-time monitoring rule definition"""

    rule_id: UUID = field(default_factory=uuid4)
    name: str = ""
    metric_type: str = ""
    condition: str = ""  # e.g., "value > threshold"
    threshold: float = 0.0
    window_minutes: int = 5
    severity: AlertSeverity = AlertSeverity.WARNING
    alert_channels: List[AlertChannel] = field(default_factory=list)
    automated_actions: List[AutomatedAction] = field(default_factory=list)
    enabled: bool = True
    description: str = ""


@dataclass
class MonitoringEvent:
    """Real-time monitoring event"""

    event_id: UUID = field(default_factory=uuid4)
    event_type: str = ""
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)
    severity: AlertSeverity = AlertSeverity.INFO
    processed: bool = False


@dataclass
class SystemMetrics:
    """Current system metrics snapshot"""

    timestamp: datetime = field(default_factory=datetime.utcnow)
    response_time_p95: float = 0.0
    response_time_avg: float = 0.0
    error_rate: float = 0.0
    throughput_rps: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_users: int = 0
    queue_depth: int = 0
    cache_hit_rate: float = 0.0
    database_connections: int = 0


@dataclass
class AlertingConfiguration:
    """Alerting system configuration"""

    enabled: bool = True
    default_channels: List[AlertChannel] = field(
        default_factory=lambda: [AlertChannel.LOG]
    )
    escalation_enabled: bool = True
    escalation_delay_minutes: int = 15
    notification_cooldown_minutes: int = 5
    max_alerts_per_hour: int = 100
    webhook_urls: Dict[str, str] = field(default_factory=dict)
    email_recipients: List[str] = field(default_factory=list)


class RealTimeMonitor:
    """
    Real-time monitoring engine with intelligent alerting
    Provides continuous monitoring and automated responses
    """

    def __init__(
        self,
        performance_validator: PerformanceValidator,
        event_bus: Optional[MetisEventBus] = None,
        state_manager: Optional[DistributedStateManager] = None,
        mode: MonitoringMode = MonitoringMode.DEVELOPMENT,
    ):
        self.performance_validator = performance_validator
        self.event_bus = event_bus
        self.state_manager = state_manager
        self.mode = mode
        self.logger = logging.getLogger(__name__)

        # Monitoring state
        self.monitoring_enabled = True
        self.monitoring_rules: Dict[UUID, MonitoringRule] = {}
        self.event_queue: deque = deque(maxlen=10000)
        self.metrics_buffer: deque = deque(maxlen=1000)

        # Alerting configuration
        self.alerting_config = AlertingConfiguration()
        self.alert_history: List[PerformanceAlert] = []
        self.notification_cooldowns: Dict[str, datetime] = {}

        # Automated response system
        self.automation_enabled = mode == MonitoringMode.PRODUCTION
        self.automation_handlers: Dict[AutomatedAction, Callable] = {}
        self.circuit_breakers: Dict[str, bool] = {}

        # Performance tracking
        self.monitoring_start_time = datetime.utcnow()
        self.events_processed = 0
        self.alerts_sent = 0
        self.automations_executed = 0

        # Initialize monitoring system
        asyncio.create_task(self._initialize_monitoring())

    async def _initialize_monitoring(self):
        """Initialize monitoring system with default rules"""

        # Load default monitoring rules
        await self._load_default_monitoring_rules()

        # Initialize automation handlers
        self._initialize_automation_handlers()

        # Start monitoring loops
        asyncio.create_task(self._monitoring_loop())
        asyncio.create_task(self._metrics_collection_loop())
        asyncio.create_task(self._alert_processing_loop())

        self.logger.info(f"Real-time monitoring initialized in {self.mode.value} mode")

    async def _load_default_monitoring_rules(self):
        """Load default monitoring rules for the system"""

        default_rules = [
            MonitoringRule(
                name="High Response Time",
                metric_type="response_time",
                condition="p95 > 5.0",
                threshold=5.0,
                window_minutes=2,
                severity=AlertSeverity.WARNING,
                alert_channels=[AlertChannel.LOG, AlertChannel.WEBHOOK],
                automated_actions=(
                    [AutomatedAction.SCALE_UP] if self.automation_enabled else []
                ),
                description="Alert when 95th percentile response time exceeds 5 seconds",
            ),
            MonitoringRule(
                name="Critical Response Time",
                metric_type="response_time",
                condition="p95 > 10.0",
                threshold=10.0,
                window_minutes=1,
                severity=AlertSeverity.CRITICAL,
                alert_channels=[
                    AlertChannel.LOG,
                    AlertChannel.WEBHOOK,
                    AlertChannel.PAGERDUTY,
                ],
                automated_actions=(
                    [AutomatedAction.SCALE_UP, AutomatedAction.NOTIFY_ONCALL]
                    if self.automation_enabled
                    else []
                ),
                description="Critical alert when response time becomes unacceptable",
            ),
            MonitoringRule(
                name="High Error Rate",
                metric_type="error_rate",
                condition="rate > 0.05",
                threshold=0.05,
                window_minutes=5,
                severity=AlertSeverity.WARNING,
                alert_channels=[AlertChannel.LOG, AlertChannel.WEBHOOK],
                automated_actions=(
                    [AutomatedAction.ENABLE_CIRCUIT_BREAKER]
                    if self.automation_enabled
                    else []
                ),
                description="Alert when error rate exceeds 5%",
            ),
            MonitoringRule(
                name="Low Cognitive Accuracy",
                metric_type="cognitive_accuracy",
                condition="accuracy < 0.70",
                threshold=0.70,
                window_minutes=10,
                severity=AlertSeverity.WARNING,
                alert_channels=[AlertChannel.LOG, AlertChannel.WEBHOOK],
                automated_actions=(
                    [AutomatedAction.SWITCH_TO_FALLBACK]
                    if self.automation_enabled
                    else []
                ),
                description="Alert when cognitive model accuracy drops significantly",
            ),
            MonitoringRule(
                name="System Integration Failure",
                metric_type="system_integration",
                condition="success_rate < 0.90",
                threshold=0.90,
                window_minutes=3,
                severity=AlertSeverity.CRITICAL,
                alert_channels=[
                    AlertChannel.LOG,
                    AlertChannel.WEBHOOK,
                    AlertChannel.PAGERDUTY,
                ],
                automated_actions=(
                    [AutomatedAction.RESTART_SERVICE, AutomatedAction.NOTIFY_ONCALL]
                    if self.automation_enabled
                    else []
                ),
                description="Critical alert for system integration failures",
            ),
            MonitoringRule(
                name="High Resource Usage",
                metric_type="resource_usage",
                condition="cpu > 0.85 OR memory > 0.90",
                threshold=0.85,
                window_minutes=5,
                severity=AlertSeverity.WARNING,
                alert_channels=[AlertChannel.LOG],
                automated_actions=(
                    [AutomatedAction.SCALE_UP] if self.automation_enabled else []
                ),
                description="Alert when resource usage is high",
            ),
        ]

        for rule in default_rules:
            self.monitoring_rules[rule.rule_id] = rule

            # Store in distributed state
            if self.state_manager:
                await self.state_manager.set_state(
                    f"monitoring_rule_{rule.rule_id}",
                    {
                        "rule_id": str(rule.rule_id),
                        "name": rule.name,
                        "metric_type": rule.metric_type,
                        "condition": rule.condition,
                        "threshold": rule.threshold,
                        "severity": rule.severity.value,
                        "enabled": rule.enabled,
                    },
                    StateType.MONITORING,
                )

    def _initialize_automation_handlers(self):
        """Initialize automated response handlers"""

        self.automation_handlers = {
            AutomatedAction.SCALE_UP: self._handle_scale_up,
            AutomatedAction.SCALE_DOWN: self._handle_scale_down,
            AutomatedAction.RESTART_SERVICE: self._handle_restart_service,
            AutomatedAction.ENABLE_CIRCUIT_BREAKER: self._handle_enable_circuit_breaker,
            AutomatedAction.THROTTLE_REQUESTS: self._handle_throttle_requests,
            AutomatedAction.SWITCH_TO_FALLBACK: self._handle_switch_to_fallback,
            AutomatedAction.NOTIFY_ONCALL: self._handle_notify_oncall,
        }

    async def _monitoring_loop(self):
        """Main monitoring loop - processes events and evaluates rules"""

        while self.monitoring_enabled:
            try:
                # Process queued events
                while self.event_queue:
                    event = self.event_queue.popleft()
                    await self._process_monitoring_event(event)
                    self.events_processed += 1

                # Evaluate monitoring rules
                await self._evaluate_monitoring_rules()

                # Sleep before next iteration
                await asyncio.sleep(1)  # 1-second monitoring interval

            except Exception as e:
                self.logger.error(f"Monitoring loop error: {str(e)}")
                await asyncio.sleep(5)

    async def _metrics_collection_loop(self):
        """Collect system metrics periodically"""

        while self.monitoring_enabled:
            try:
                # Collect current system metrics
                metrics = await self._collect_system_metrics()

                # Buffer metrics for analysis
                self.metrics_buffer.append(metrics)

                # Store metrics in state
                if self.state_manager:
                    await self.state_manager.set_state(
                        f"system_metrics_{metrics.timestamp.isoformat()}",
                        {
                            "timestamp": metrics.timestamp.isoformat(),
                            "response_time_p95": metrics.response_time_p95,
                            "response_time_avg": metrics.response_time_avg,
                            "error_rate": metrics.error_rate,
                            "throughput_rps": metrics.throughput_rps,
                            "active_users": metrics.active_users,
                        },
                        StateType.METRICS,
                    )

                # Sleep before next collection
                await asyncio.sleep(10)  # 10-second metrics collection

            except Exception as e:
                self.logger.error(f"Metrics collection error: {str(e)}")
                await asyncio.sleep(30)

    async def _alert_processing_loop(self):
        """Process and send alerts"""

        while self.monitoring_enabled:
            try:
                # Check for pending alerts from performance validator
                health_report = (
                    await self.performance_validator.validate_system_performance()
                )

                # Process any new alerts
                for alert in health_report.active_alerts:
                    await self._process_alert(alert)

                # Sleep before next check
                await asyncio.sleep(30)  # 30-second alert processing

            except Exception as e:
                self.logger.error(f"Alert processing error: {str(e)}")
                await asyncio.sleep(60)

    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics snapshot"""

        # Simulate metric collection (would integrate with actual monitoring tools)
        metrics = SystemMetrics(
            response_time_p95=np.random.uniform(0.5, 3.0),
            response_time_avg=np.random.uniform(0.3, 2.0),
            error_rate=np.random.uniform(0.0, 0.02),
            throughput_rps=np.random.uniform(10.0, 100.0),
            cpu_usage=np.random.uniform(0.2, 0.8),
            memory_usage=np.random.uniform(0.3, 0.7),
            active_users=np.random.randint(1, 50),
            queue_depth=np.random.randint(0, 20),
            cache_hit_rate=np.random.uniform(0.85, 0.99),
            database_connections=np.random.randint(5, 50),
        )

        return metrics

    async def _process_monitoring_event(self, event: MonitoringEvent):
        """Process individual monitoring event"""

        self.logger.debug(f"Processing monitoring event: {event.event_type}")

        # Mark event as processed
        event.processed = True

        # Check if event triggers any monitoring rules
        triggered_rules = []
        for rule in self.monitoring_rules.values():
            if rule.enabled and await self._evaluate_rule_for_event(rule, event):
                triggered_rules.append(rule)

        # Process triggered rules
        for rule in triggered_rules:
            await self._handle_triggered_rule(rule, event)

    async def _evaluate_monitoring_rules(self):
        """Evaluate all monitoring rules against current state"""

        # Get recent metrics
        if not self.metrics_buffer:
            return

        recent_metrics = list(self.metrics_buffer)[-10:]  # Last 10 metrics

        for rule in self.monitoring_rules.values():
            if not rule.enabled:
                continue

            try:
                if await self._evaluate_rule_against_metrics(rule, recent_metrics):
                    await self._handle_triggered_rule(rule, None)
            except Exception as e:
                self.logger.error(f"Rule evaluation error for {rule.name}: {str(e)}")

    async def _evaluate_rule_for_event(
        self, rule: MonitoringRule, event: MonitoringEvent
    ) -> bool:
        """Evaluate if monitoring rule is triggered by event"""

        # Simple pattern matching for event-based rules
        if rule.metric_type in event.event_type:
            if "error" in rule.condition.lower() and event.severity in [
                AlertSeverity.WARNING,
                AlertSeverity.CRITICAL,
            ]:
                return True

            # Check numeric thresholds in event data
            if "value" in event.data:
                value = event.data["value"]
                if ">" in rule.condition and value > rule.threshold:
                    return True
                elif "<" in rule.condition and value < rule.threshold:
                    return True

        return False

    async def _evaluate_rule_against_metrics(
        self, rule: MonitoringRule, metrics: List[SystemMetrics]
    ) -> bool:
        """Evaluate monitoring rule against metrics"""

        if not metrics:
            return False

        # Get metrics within rule window
        window_start = datetime.utcnow() - timedelta(minutes=rule.window_minutes)
        windowed_metrics = [m for m in metrics if m.timestamp >= window_start]

        if not windowed_metrics:
            return False

        # Evaluate rule condition
        try:
            if rule.metric_type == "response_time":
                values = [m.response_time_p95 for m in windowed_metrics]
                if "p95" in rule.condition:
                    current_value = max(values) if values else 0
                else:
                    current_value = sum(values) / len(values) if values else 0

                return self._evaluate_threshold_condition(
                    rule.condition, current_value, rule.threshold
                )

            elif rule.metric_type == "error_rate":
                values = [m.error_rate for m in windowed_metrics]
                current_value = sum(values) / len(values) if values else 0
                return self._evaluate_threshold_condition(
                    rule.condition, current_value, rule.threshold
                )

            elif rule.metric_type == "resource_usage":
                cpu_values = [m.cpu_usage for m in windowed_metrics]
                memory_values = [m.memory_usage for m in windowed_metrics]
                avg_cpu = sum(cpu_values) / len(cpu_values) if cpu_values else 0
                avg_memory = (
                    sum(memory_values) / len(memory_values) if memory_values else 0
                )

                # Handle OR condition
                if "OR" in rule.condition:
                    return avg_cpu > rule.threshold or avg_memory > 0.90
                else:
                    return avg_cpu > rule.threshold

        except Exception as e:
            self.logger.error(f"Rule evaluation error: {str(e)}")
            return False

        return False

    def _evaluate_threshold_condition(
        self, condition: str, value: float, threshold: float
    ) -> bool:
        """Evaluate threshold condition"""

        if ">" in condition:
            return value > threshold
        elif "<" in condition:
            return value < threshold
        elif "=" in condition:
            return abs(value - threshold) < 0.01

        return False

    async def _handle_triggered_rule(
        self, rule: MonitoringRule, event: Optional[MonitoringEvent]
    ):
        """Handle triggered monitoring rule"""

        self.logger.warning(f"Monitoring rule triggered: {rule.name}")

        # Check notification cooldown
        cooldown_key = f"{rule.rule_id}_{rule.severity.value}"
        if cooldown_key in self.notification_cooldowns:
            last_notification = self.notification_cooldowns[cooldown_key]
            cooldown_period = timedelta(
                minutes=self.alerting_config.notification_cooldown_minutes
            )
            if datetime.utcnow() - last_notification < cooldown_period:
                self.logger.debug(f"Skipping notification due to cooldown: {rule.name}")
                return

        # Send alerts
        for channel in rule.alert_channels:
            await self._send_alert(channel, rule, event)

        # Execute automated actions
        for action in rule.automated_actions:
            if self.automation_enabled:
                await self._execute_automated_action(action, rule, event)

        # Update cooldown
        self.notification_cooldowns[cooldown_key] = datetime.utcnow()
        self.alerts_sent += 1

        # Emit monitoring event
        if self.event_bus:
            await self.event_bus.publish_event(
                CloudEvent(
                    type="monitoring.rule.triggered",
                    source="monitoring/real_time",
                    data={
                        "rule_id": str(rule.rule_id),
                        "rule_name": rule.name,
                        "severity": rule.severity.value,
                        "metric_type": rule.metric_type,
                        "threshold": rule.threshold,
                    },
                )
            )

    async def _send_alert(
        self,
        channel: AlertChannel,
        rule: MonitoringRule,
        event: Optional[MonitoringEvent],
    ):
        """Send alert through specified channel"""

        try:
            message = f"METIS Alert [{rule.severity.value.upper()}]: {rule.name}\n"
            message += f"Description: {rule.description}\n"
            message += f"Threshold: {rule.threshold}\n"
            message += f"Time: {datetime.utcnow().isoformat()}"

            if channel == AlertChannel.LOG:
                if rule.severity == AlertSeverity.CRITICAL:
                    self.logger.critical(message)
                elif rule.severity == AlertSeverity.WARNING:
                    self.logger.warning(message)
                else:
                    self.logger.info(message)

            elif channel == AlertChannel.WEBHOOK:
                await self._send_webhook_alert(message, rule)

            elif channel == AlertChannel.EMAIL:
                await self._send_email_alert(message, rule)

            elif channel == AlertChannel.SLACK:
                await self._send_slack_alert(message, rule)

            elif channel == AlertChannel.PAGERDUTY:
                await self._send_pagerduty_alert(message, rule)

            self.logger.debug(f"Alert sent via {channel.value}: {rule.name}")

        except Exception as e:
            self.logger.error(f"Failed to send alert via {channel.value}: {str(e)}")

    async def _send_webhook_alert(self, message: str, rule: MonitoringRule):
        """Send webhook alert"""
        # Would integrate with actual webhook service
        self.logger.info(f"Webhook alert: {message}")

    async def _send_email_alert(self, message: str, rule: MonitoringRule):
        """Send email alert"""
        # Would integrate with actual email service
        self.logger.info(f"Email alert: {message}")

    async def _send_slack_alert(self, message: str, rule: MonitoringRule):
        """Send Slack alert"""
        # Would integrate with Slack API
        self.logger.info(f"Slack alert: {message}")

    async def _send_pagerduty_alert(self, message: str, rule: MonitoringRule):
        """Send PagerDuty alert"""
        # Would integrate with PagerDuty API
        self.logger.info(f"PagerDuty alert: {message}")

    async def _execute_automated_action(
        self,
        action: AutomatedAction,
        rule: MonitoringRule,
        event: Optional[MonitoringEvent],
    ):
        """Execute automated response action"""

        if not self.automation_enabled:
            self.logger.info(f"Automation disabled, skipping action: {action.value}")
            return

        try:
            handler = self.automation_handlers.get(action)
            if handler:
                await handler(rule, event)
                self.automations_executed += 1
                self.logger.info(f"Executed automated action: {action.value}")
            else:
                self.logger.warning(f"No handler for automated action: {action.value}")

        except Exception as e:
            self.logger.error(f"Automated action failed {action.value}: {str(e)}")

    # Automated action handlers
    async def _handle_scale_up(
        self, rule: MonitoringRule, event: Optional[MonitoringEvent]
    ):
        """Handle scale up automation"""
        self.logger.info("Executing scale up - would increase resource allocation")

    async def _handle_scale_down(
        self, rule: MonitoringRule, event: Optional[MonitoringEvent]
    ):
        """Handle scale down automation"""
        self.logger.info("Executing scale down - would decrease resource allocation")

    async def _handle_restart_service(
        self, rule: MonitoringRule, event: Optional[MonitoringEvent]
    ):
        """Handle service restart automation"""
        self.logger.info(
            "Executing service restart - would restart affected components"
        )

    async def _handle_enable_circuit_breaker(
        self, rule: MonitoringRule, event: Optional[MonitoringEvent]
    ):
        """Handle circuit breaker activation"""
        self.circuit_breakers[rule.metric_type] = True
        self.logger.info(f"Enabled circuit breaker for {rule.metric_type}")

    async def _handle_throttle_requests(
        self, rule: MonitoringRule, event: Optional[MonitoringEvent]
    ):
        """Handle request throttling"""
        self.logger.info("Executing request throttling - would limit incoming requests")

    async def _handle_switch_to_fallback(
        self, rule: MonitoringRule, event: Optional[MonitoringEvent]
    ):
        """Handle fallback switching"""
        self.logger.info("Switching to fallback mode - would use backup systems")

    async def _handle_notify_oncall(
        self, rule: MonitoringRule, event: Optional[MonitoringEvent]
    ):
        """Handle on-call notification"""
        self.logger.info(
            "Notifying on-call engineer - would send high-priority notification"
        )

    async def _process_alert(self, alert: PerformanceAlert):
        """Process performance alert"""

        # Add to monitoring event queue
        event = MonitoringEvent(
            event_type=f"performance.alert.{alert.severity.value}",
            source="performance/validator",
            data={
                "alert_id": str(alert.alert_id),
                "metric_type": alert.metric_type.value,
                "current_value": alert.current_value,
                "target_value": alert.target_value,
            },
            severity=alert.severity,
        )

        self.event_queue.append(event)

    def add_monitoring_rule(self, rule: MonitoringRule) -> UUID:
        """Add custom monitoring rule"""
        self.monitoring_rules[rule.rule_id] = rule
        self.logger.info(f"Added monitoring rule: {rule.name}")
        return rule.rule_id

    def remove_monitoring_rule(self, rule_id: UUID) -> bool:
        """Remove monitoring rule"""
        if rule_id in self.monitoring_rules:
            rule = self.monitoring_rules[rule_id]
            del self.monitoring_rules[rule_id]
            self.logger.info(f"Removed monitoring rule: {rule.name}")
            return True
        return False

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status"""

        uptime = datetime.utcnow() - self.monitoring_start_time

        return {
            "monitoring_enabled": self.monitoring_enabled,
            "automation_enabled": self.automation_enabled,
            "mode": self.mode.value,
            "uptime_seconds": uptime.total_seconds(),
            "events_processed": self.events_processed,
            "alerts_sent": self.alerts_sent,
            "automations_executed": self.automations_executed,
            "active_rules": len(
                [r for r in self.monitoring_rules.values() if r.enabled]
            ),
            "total_rules": len(self.monitoring_rules),
            "current_metrics_buffer_size": len(self.metrics_buffer),
            "event_queue_size": len(self.event_queue),
            "circuit_breakers_active": len(
                [cb for cb in self.circuit_breakers.values() if cb]
            ),
        }


# Global real-time monitor instance
_global_monitor: Optional[RealTimeMonitor] = None


async def get_real_time_monitor() -> RealTimeMonitor:
    """Get or create global real-time monitor instance"""
    global _global_monitor

    if _global_monitor is None:
        validator = await get_performance_validator()
        _global_monitor = RealTimeMonitor(validator)

    return _global_monitor


# Convenience functions for monitoring
async def monitor_operation(operation_name: str, metric_type: str = "custom"):
    """Context manager for monitoring operations"""
    # Would implement context manager for automatic monitoring
    pass


async def trigger_manual_alert(
    message: str, severity: AlertSeverity = AlertSeverity.INFO
):
    """Trigger manual alert"""
    monitor = await get_real_time_monitor()
    event = MonitoringEvent(
        event_type="manual.alert",
        source="user",
        data={"message": message},
        severity=severity,
    )
    monitor.event_queue.append(event)


async def get_system_health_dashboard() -> Dict[str, Any]:
    """Get system health dashboard data"""
    monitor = await get_real_time_monitor()
    validator = await get_performance_validator()

    health_report = await validator.validate_system_performance()
    monitoring_status = monitor.get_monitoring_status()

    return {
        "overall_status": health_report.overall_status.value,
        "monitoring": monitoring_status,
        "active_alerts": len(health_report.active_alerts),
        "uptime_percentage": health_report.uptime_percentage,
        "last_updated": datetime.utcnow().isoformat(),
    }
