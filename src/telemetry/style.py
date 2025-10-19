"""Style consistency telemetry scoring."""
from .metrics import metrics, MetricType
import re

# Register gauge
try:
    metrics.register_metric("style_consistency", MetricType.GAUGE, "Style consistency score (0-1)")
except ValueError:
    pass


def score_style(text: str) -> float:
    if not text:
        metrics.record_value("style_consistency", 0.0)
        return 0.0
    # Simple heuristics: sentence casing, bullet usage, paragraph balance
    sentences = re.split(r"[.!?]+\s", text.strip())
    sentences = [s for s in sentences if s]
    if not sentences:
        metrics.record_value("style_consistency", 0.0)
        return 0.0
    capitalized = sum(1 for s in sentences if s and s[0].isupper()) / len(sentences)
    avg_len = sum(len(s) for s in sentences) / len(sentences)
    length_penalty = 1.0 if 60 <= avg_len <= 400 else 0.7
    bullet_bonus = 1.0 if ("- " in text or "•" in text or "\n\n" in text) else 0.9
    score = max(0.0, min(1.0, 0.5 * capitalized + 0.3 * length_penalty + 0.2 * bullet_bonus))
    metrics.record_value("style_consistency", score)
    return score