"""Summary metrics aggregator aligned with UI (Overall Confidence, Cognitive Diversity, Evidence Strength, Processing Time)."""

from typing import Dict, List, Optional
from collections import Counter
from .confidence import ConfidenceScorer
from .metrics import metrics, MetricType
from src.services.cqa_score_service import get_cqa_score_service


def _safe_register_metric(name: str, mtype: MetricType, desc: str) -> None:
    try:
        metrics.register_metric(name=name, type=mtype, description=desc)
    except ValueError:
        pass


# Register gauges once
_safe_register_metric(
    "processing_time_seconds", MetricType.GAUGE, "End-to-end processing time (seconds)"
)
_safe_register_metric(
    "evidence_strength", MetricType.GAUGE, "Evidence strength (0-1) from CQA and grounding"
)
_safe_register_metric(
    "cognitive_diversity", MetricType.GAUGE, "Cognitive diversity (0-1) across consultants and models"
)


class SummaryMetricsService:
    def __init__(self) -> None:
        self.scorer = ConfidenceScorer()
        self.cqa = get_cqa_score_service()

    def compute_with_provenance(
        self,
        selected_consultants: List[str],
        selected_models: List[str],
        execution_time_ms: float,
        context: Optional[Dict[str, str]] = None,
        quality_scores: Optional[Dict[str, float]] = None,
    ) -> (Dict[str, float], Dict[str, any]):
        diversity = self._compute_diversity(selected_consultants, selected_models)
        evidence_cqa, evidence_details = self._compute_evidence_strength(selected_models, details=True)

        # Extract real-time signals if provided
        signals = self._extract_signals_from_quality(quality_scores or {})
        # Combine CQA evidence with runtime evidence if present (70/30)
        evidence = 0.7 * evidence_cqa + 0.3 * signals.get("evidence", evidence_cqa)

        # Overall confidence from factors
        factors = {
            "evidence": evidence,
            "coherence": signals.get("coherence", 0.8),
            "relevance": signals.get("relevance", 0.8),
            "impact": signals.get("impact", 0.6),
        }
        conf = self.scorer.evaluate_decision_confidence(
            context=context or {}, factors=factors, component="summary"
        )
        weights = self.scorer._get_factor_weights(list(factors.keys()), context or {})  # internal but stable

        processing_secs = max(0.0, execution_time_ms / 1000.0)

        # Emit gauges
        metrics.record_value("processing_time_seconds", processing_secs)
        metrics.record_value("evidence_strength", evidence)
        metrics.record_value("cognitive_diversity", diversity)

        metrics_obj = {
            "overall_confidence": round(conf.overall_score, 4),
            "cognitive_diversity": round(diversity, 4),
            "evidence_strength": round(evidence, 4),
            "processing_time_seconds": round(processing_secs, 3),
        }
        provenance = {
            "version": "1.0",
            "timestamp": __import__("datetime").datetime.now(__import__("datetime").UTC).isoformat(),
            "factors": factors,
            "signals": signals,
            "weights": weights,
            "diversity": {
                "value": diversity,
                "consultants": selected_consultants,
                "models": selected_models,
            },
            "evidence_details": evidence_details,
        }
        return metrics_obj, provenance

    def compute(
        self,
        selected_consultants: List[str],
        selected_models: List[str],
        execution_time_ms: float,
        context: Optional[Dict[str, str]] = None,
        quality_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        metrics_obj, _ = self.compute_with_provenance(
            selected_consultants,
            selected_models,
            execution_time_ms,
            context,
            quality_scores,
        )
        return metrics_obj

    def _compute_diversity(self, consultants: List[str], models: List[str]) -> float:
        def one_minus_hhi(items: List[str]) -> float:
            if not items:
                return 0.0
            counts = Counter(items)
            total = sum(counts.values())
            hhi = sum((c / total) ** 2 for c in counts.values())
            return max(0.0, 1.0 - hhi)

        c_div = one_minus_hhi(consultants)
        m_div = one_minus_hhi(models)
        return (c_div + m_div) / 2.0

    def _compute_evidence_strength(self, models: List[str], details: bool = False):
        if not models:
            return (0.0, []) if details else 0.0
        scores = []
        details_list = []
        for m in models:
            try:
                s = self.cqa.get_mental_model_score(m)
                if s is not None:
                    norm = max(0.0, min(1.0, s.weighted_score / 10.0))
                    scores.append(norm)
                    details_list.append({"model": m, "weighted_score": s.weighted_score, "normalized": norm})
            except Exception:
                continue
        if not scores:
            return (0.5, details_list) if details else 0.5
        avg = sum(scores) / len(scores)
        return (avg, details_list) if details else avg

    def _extract_signals_from_quality(self, qs: Dict[str, float]) -> Dict[str, float]:
        def norm(x: float) -> float:
            try:
                v = float(x)
                if v > 1.0:
                    if v <= 10.0:
                        v = v / 10.0
                    elif v <= 100.0:
                        v = v / 100.0
                    else:
                        v = 1.0
                return max(0.0, min(1.0, v))
            except Exception:
                return 0.0

        signals = {}
        # Evidence-related
        for k in ["evidence_strength", "groundedness", "attribution"]:
            if k in qs:
                signals["evidence"] = max(signals.get("evidence", 0.0), norm(qs[k]))
        # Coherence/self-verification
        for k in ["self_verification", "consistency", "coherence"]:
            if k in qs:
                signals["coherence"] = max(signals.get("coherence", 0.0), norm(qs[k]))
        # Relevance
        if "relevance" in qs:
            signals["relevance"] = norm(qs["relevance"])
        # Impact (proxy via criticality or business_impact if present)
        for k in ["impact", "business_impact", "criticality"]:
            if k in qs:
                signals["impact"] = max(signals.get("impact", 0.0), norm(qs[k]))
        return signals


# Global instance
summary_metrics_service = SummaryMetricsService()