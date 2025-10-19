"""
Simple drift detection utilities for telemetry streams.
Maintains rolling stats and raises alerts when metrics drift beyond thresholds.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


@dataclass
class RollingStats:
    window: int
    values: Deque[float]

    def add(self, v: float) -> None:
        if len(self.values) >= self.window:
            self.values.popleft()
        self.values.append(v)

    def avg(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0

    def p05(self) -> float:
        if not self.values:
            return 0.0
        sorted_vals = sorted(self.values)
        idx = max(0, int(0.05 * (len(sorted_vals) - 1)))
        return sorted_vals[idx]

    def p95(self) -> float:
        if not self.values:
            return 0.0
        sorted_vals = sorted(self.values)
        idx = int(0.95 * (len(sorted_vals) - 1))
        return sorted_vals[idx]


class DriftMonitor:
    """Monitors confidence and style scores for drift."""

    def __init__(self, window: int = 200, conf_floor: float = 0.85, style_floor: float = 0.7):
        import os
        # Allow env overrides for calibration
        try:
            conf_floor = float(os.getenv("DRIFT_CONF_FLOOR", str(conf_floor)))
        except Exception:
            pass
        try:
            style_floor = float(os.getenv("DRIFT_STYLE_FLOOR", str(style_floor)))
        except Exception:
            pass
        self.conf_stats = RollingStats(window, deque())
        self.style_stats = RollingStats(window, deque())
        self.conf_floor = conf_floor
        self.style_floor = style_floor

    def record(self, confidence: Optional[float], style: Optional[float]) -> Optional[Dict[str, float]]:
        alert: Optional[Dict[str, float]] = None
        if confidence is not None:
            self.conf_stats.add(confidence)
        if style is not None:
            self.style_stats.add(style)

        # Trigger alert if p05 drops below floor (early warning)
        conf_p05 = self.conf_stats.p05()
        style_p05 = self.style_stats.p05()
        if self.conf_stats.values and conf_p05 < self.conf_floor:
            alert = alert or {}
            alert['confidence_p05'] = conf_p05
        if self.style_stats.values and style_p05 < self.style_floor:
            alert = alert or {}
            alert['style_p05'] = style_p05

        if alert:
            logger.warning(f"🚨 Drift alert @ {datetime.now(timezone.utc).isoformat()}: {alert}")
        return alert


# Singleton
_drift_monitor: Optional[DriftMonitor] = None


def get_drift_monitor() -> DriftMonitor:
    global _drift_monitor
    if _drift_monitor is None:
        _drift_monitor = DriftMonitor()
    return _drift_monitor
