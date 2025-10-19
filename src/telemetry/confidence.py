"""Confidence scoring system for tracking reliability of decisions and responses."""

from typing import Dict, Optional, List
from pydantic import BaseModel, Field
from .metrics import metrics, MetricType
import numpy as np

class ConfidenceScore(BaseModel):
    """Confidence score with factors and metadata."""
    overall_score: float = Field(ge=0.0, le=1.0)
    factors: Dict[str, float] = {}  # Individual factor scores
    reasoning: Optional[str] = None
    metadata: Dict[str, str] = {}

class ConfidenceScorer:
    """Evaluates confidence of decisions and responses."""
    
    def __init__(self):
        self._register_metrics()
        
    def _register_metrics(self) -> None:
        """Register confidence-related metrics."""
        try:
            metrics.register_metric(
                name="confidence_score",
                type=MetricType.HISTOGRAM,
                description="Distribution of confidence scores",
                labels={"component": "string", "type": "string"}
            )
        except ValueError:
            # Skip if already registered
            pass
            
        try:
            metrics.register_metric(
                name="confidence_factors",
                type=MetricType.GAUGE,
                description="Individual confidence factor scores",
                labels={"factor": "string", "component": "string"}
            )
        except ValueError:
            # Skip if already registered
            pass
        
    def evaluate_response_confidence(
        self,
        response: str,
        factors: Dict[str, float],
        component: str = "default",
        metadata: Optional[Dict[str, str]] = None
    ) -> ConfidenceScore:
        """Evaluate confidence of a response."""
        # Normalize factor scores to [0,1]
        normalized_factors = {
            k: max(0.0, min(1.0, v))
            for k, v in factors.items()
        }
        
        # Calculate overall score as weighted average
        if not normalized_factors:
            overall_score = 0.0
        else:
            weights = self._get_factor_weights(list(normalized_factors.keys()))
            denom = sum(weights.values()) or 0.0
            overall_score = (
                sum(score * weights.get(factor, 1.0) for factor, score in normalized_factors.items()) / denom
            ) if denom > 0 else 0.0
        
        # Record metrics
        metrics.record_value(
            "confidence_score",
            overall_score,
            labels={"component": component, "type": "response"},
            buckets=self._get_histogram_buckets()
        )
        
        for factor, score in normalized_factors.items():
            metrics.record_value(
                "confidence_factors",
                score,
                labels={"factor": factor, "component": component}
            )
            
        return ConfidenceScore(
            overall_score=overall_score,
            factors=normalized_factors,
            metadata=metadata or {}
        )
        
    def evaluate_decision_confidence(
        self,
        context: Dict[str, str],
        factors: Dict[str, float],
        component: str = "default",
        metadata: Optional[Dict[str, str]] = None
    ) -> ConfidenceScore:
        """Evaluate confidence of a decision."""
        # Similar to response confidence but with context-specific factors
        normalized_factors = {
            k: max(0.0, min(1.0, v))
            for k, v in factors.items()
        }
        
        # Calculate overall score with context awareness
        if not normalized_factors:
            overall_score = 0.0
        else:
            weights = self._get_factor_weights(list(normalized_factors.keys()), context)
            denom = sum(weights.values()) or 0.0
            overall_score = (
                sum(score * weights.get(factor, 1.0) for factor, score in normalized_factors.items()) / denom
            ) if denom > 0 else 0.0
        
        # Record metrics
        metrics.record_value(
            "confidence_score",
            overall_score,
            labels={"component": component, "type": "decision"},
            buckets=self._get_histogram_buckets()
        )
        
        for factor, score in normalized_factors.items():
            metrics.record_value(
                "confidence_factors",
                score,
                labels={"factor": factor, "component": component}
            )
            
        return ConfidenceScore(
            overall_score=overall_score,
            factors=normalized_factors,
            metadata=metadata or {}
        )
        
    def _get_factor_weights(
        self,
        factors: List[str],
        context: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """Get weights for different confidence factors."""
        # Base weights
        weights = {
            # Response factors
            "coherence": 1.0,
            "relevance": 1.0,
            "groundedness": 1.2,
            "toxicity": 1.5,  # Higher weight for safety
            
            # Decision factors
            "evidence": 1.2,
            "consistency": 1.0,
            "certainty": 0.8,
            "impact": 1.3
        }
        
        # Context-specific weight adjustments
        if context:
            if context.get("criticality") == "high":
                weights = {k: v * 1.2 for k, v in weights.items()}
            if context.get("safety_critical") == "true":
                weights["toxicity"] *= 1.5
                
        return {factor: weights.get(factor, 1.0) for factor in factors}
        
    def _get_histogram_buckets(self) -> Dict[float, int]:
        """Get buckets for confidence score histogram."""
        return {
            0.0: 0,
            0.2: 0,
            0.4: 0,
            0.6: 0,
            0.8: 0,
            0.9: 0,
            0.95: 0,
            0.99: 0,
            1.0: 0
        }

# Global confidence scorer instance
confidence_scorer = ConfidenceScorer()