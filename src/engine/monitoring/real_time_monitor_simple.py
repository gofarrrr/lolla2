"""
Simplified Real-time Monitor for METIS
Minimal implementation for system validation
"""

import logging
from typing import Dict, Any
from datetime import datetime


class RealTimeMonitor:
    """
    Simplified real-time monitor for testing
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics = {}
        self.start_time = datetime.now()
        self.logger.info("✅ Simple RealTimeMonitor initialized")

    def record_metric(self, name: str, value: float) -> None:
        """Record a metric"""
        self.metrics[name] = {"value": value, "timestamp": datetime.now().isoformat()}

    def get_health_status(self) -> Dict[str, Any]:
        """Get monitor health status"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        return {
            "status": "operational",
            "uptime_seconds": uptime,
            "metrics_tracked": len(self.metrics),
            "initialized": True,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get all recorded metrics"""
        return self.metrics
