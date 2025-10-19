"""Resource budget tracking for per-analysis cost/latency/tokens."""
from typing import Optional, Dict
from dataclasses import dataclass
from .metrics import metrics, MetricType


def _safe_register(name: str, mtype: MetricType, desc: str) -> None:
    try:
        metrics.register_metric(name=name, type=mtype, description=desc)
    except ValueError:
        pass


_safe_register("tokens_consumed_total", MetricType.COUNTER, "Total tokens consumed")
_safe_register("cost_usd_total", MetricType.COUNTER, "Total cost in USD")
_safe_register("latency_ms", MetricType.GAUGE, "Latency per analysis (ms)")
_safe_register("budget_violations_total", MetricType.COUNTER, "Count of budget violations")


@dataclass
class BudgetConfig:
    max_tokens: Optional[int] = None
    max_cost_usd: Optional[float] = None
    max_latency_ms: Optional[float] = None


class BudgetTracker:
    def __init__(self, config: Optional[BudgetConfig] = None) -> None:
        self.config = config or BudgetConfig()

    def record(self, tokens: Optional[int] = None, cost_usd: Optional[float] = None, latency_ms: Optional[float] = None) -> None:
        if tokens is not None:
            metrics.record_value("tokens_consumed_total", float(tokens))
        if cost_usd is not None:
            metrics.record_value("cost_usd_total", float(cost_usd))
        if latency_ms is not None:
            metrics.record_value("latency_ms", float(latency_ms))

    def evaluate(self, tokens: Optional[int], cost_usd: Optional[float], latency_ms: Optional[float]) -> Dict[str, Dict[str, float]]:
        violations: Dict[str, Dict[str, float]] = {}
        if self.config.max_tokens is not None and tokens is not None and tokens > self.config.max_tokens:
            violations["tokens"] = {"actual": float(tokens), "limit": float(self.config.max_tokens)}
        if self.config.max_cost_usd is not None and cost_usd is not None and cost_usd > self.config.max_cost_usd:
            violations["cost_usd"] = {"actual": float(cost_usd), "limit": float(self.config.max_cost_usd)}
        if self.config.max_latency_ms is not None and latency_ms is not None and latency_ms > self.config.max_latency_ms:
            violations["latency_ms"] = {"actual": float(latency_ms), "limit": float(self.config.max_latency_ms)}
        return violations

    def check_and_record(self, tokens: Optional[int], cost_usd: Optional[float], latency_ms: Optional[float]) -> Dict[str, Dict[str, float]]:
        violations = self.evaluate(tokens, cost_usd, latency_ms)
        if violations:
            metrics.record_value("budget_violations_total", 1.0)
        return violations

    def configure_from_env(self) -> None:
        import os
        mt = os.getenv("BUDGET_MAX_TOKENS")
        mc = os.getenv("BUDGET_MAX_COST_USD")
        ml = os.getenv("BUDGET_MAX_LATENCY_MS")
        try:
            if mt is not None:
                self.config.max_tokens = int(mt)
            if mc is not None:
                self.config.max_cost_usd = float(mc)
            if ml is not None:
                self.config.max_latency_ms = float(ml)
        except Exception:
            pass


# Global tracker
budget_tracker = BudgetTracker()
