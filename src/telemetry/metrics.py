"""Shared metrics backend for telemetry components."""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, UTC
from pydantic import BaseModel
from enum import Enum
import threading
from collections import defaultdict

class MetricType(Enum):
    """Types of metrics that can be tracked."""
    COUNTER = "counter"      # Monotonically increasing counter
    GAUGE = "gauge"         # Value that can go up and down
    HISTOGRAM = "histogram"  # Distribution of values
    SUMMARY = "summary"     # Percentile statistics

class MetricMetadata(BaseModel):
    """Metadata for a metric."""
    name: str
    type: MetricType
    description: str
    labels: Dict[str, str] = {}
    created_at: datetime = datetime.now(UTC)
    updated_at: datetime = datetime.now(UTC)

class MetricValue(BaseModel):
    """Value for a metric, with optional histogram/summary data."""
    value: float
    timestamp: datetime = datetime.now(UTC)
    labels: Dict[str, str] = {}
    buckets: Optional[Dict[float, int]] = None  # For histograms
    quantiles: Optional[Dict[float, float]] = None  # For summaries

class MetricsBackend:
    """Thread-safe metrics storage and aggregation."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._metrics: Dict[str, MetricMetadata] = {}
        self._values: Dict[str, List[MetricValue]] = defaultdict(list)
        
    def register_metric(
        self,
        name: str,
        type: MetricType,
        description: str,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Register a new metric."""
        with self._lock:
            if name in self._metrics:
                raise ValueError(f"Metric {name} already registered")
            
            self._metrics[name] = MetricMetadata(
                name=name,
                type=type,
                description=description,
                labels=labels or {}
            )
            
    def record_value(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[Dict[float, int]] = None,
        quantiles: Optional[Dict[float, float]] = None
    ) -> None:
        """Record a value for a metric."""
        with self._lock:
            if name not in self._metrics:
                raise ValueError(f"Metric {name} not registered")
            
            # Validate metric type matches data
            metric = self._metrics[name]
            if metric.type == MetricType.HISTOGRAM and not buckets:
                raise ValueError("Histogram metrics require buckets")
            if metric.type == MetricType.SUMMARY and not quantiles:
                raise ValueError("Summary metrics require quantiles")
            
            # Record value
            self._values[name].append(
                MetricValue(
                    value=value,
                    labels=labels or {},
                    buckets=buckets,
                    quantiles=quantiles
                )
            )
            
            # Update metadata
            self._metrics[name].updated_at = datetime.now(UTC)
            
    def get_metric(self, name: str) -> Optional[MetricMetadata]:
        """Get metadata for a metric."""
        with self._lock:
            return self._metrics.get(name)
            
    def get_values(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> List[MetricValue]:
        """Get values for a metric with optional filtering."""
        with self._lock:
            if name not in self._metrics:
                return []
                
            values = self._values[name]
            
            # Apply time filters
            if start_time:
                values = [v for v in values if v.timestamp >= start_time]
            if end_time:
                values = [v for v in values if v.timestamp <= end_time]
                
            # Apply label filters
            if labels:
                values = [
                    mv for mv in values
                    if all(mv.labels.get(k) == val for k, val in labels.items())
                ]
                
            return values
            
    def get_current_value(self, name: str) -> Optional[float]:
        """Get the most recent value for a metric."""
        with self._lock:
            values = self._values.get(name, [])
            if not values:
                return None
            return sorted(values, key=lambda v: v.timestamp)[-1].value
            
    def list_metrics(self) -> List[MetricMetadata]:
        """List all registered metrics."""
        with self._lock:
            return list(self._metrics.values())

# Global metrics instance
metrics = MetricsBackend()