"""
Base Cognitive Model Interface
Abstract foundation for all cognitive model implementations
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum
import time
import logging

if TYPE_CHECKING:
    from src.engine.engines.integration.llm_orchestrator import LLMOrchestrator


class CognitiveModelType(Enum):
    """Types of cognitive models available"""

    SYSTEMS_THINKING = "systems_thinking"
    CRITICAL_THINKING = "critical_thinking"
    MECE_STRUCTURING = "mece_structuring"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    DECISION_FRAMEWORKS = "decision_frameworks"


@dataclass
class ModelApplicationContext:
    """Context for applying a cognitive model"""

    problem_statement: str
    business_context: Dict[str, Any]
    cognitive_load_level: str = "medium"
    quality_requirements: Dict[str, float] = None
    available_evidence: List[str] = None
    user_preferences: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.quality_requirements is None:
            self.quality_requirements = {
                "accuracy_requirement": 0.8,
                "depth_requirement": 0.7,
                "coherence_requirement": 0.8,
            }
        if self.available_evidence is None:
            self.available_evidence = []


@dataclass
class ModelApplicationResult:
    """Result of applying a cognitive model"""

    reasoning_text: str
    confidence_score: float
    key_insights: List[str]
    supporting_evidence: List[str]
    assumptions_made: List[str]
    quality_metrics: Dict[str, float]
    processing_time_ms: float
    model_id: str = ""

    def __post_init__(self):
        if not self.quality_metrics:
            self.quality_metrics = {}


class BaseCognitiveModel(ABC):
    """Abstract base class for all cognitive models"""

    def __init__(
        self, model_id: str, llm_orchestrator: Optional["LLMOrchestrator"] = None
    ):
        self.model_id = model_id
        self.model_type = self._get_model_type()
        self.llm_orchestrator = llm_orchestrator
        self.performance_history: List[float] = []
        self.quality_threshold = 0.7
        self.logger = logging.getLogger(f"{__name__}.{model_id}")

        # Track usage statistics
        self.usage_count = 0
        self.total_processing_time = 0.0
        self.average_confidence = 0.0

    @abstractmethod
    def _get_model_type(self) -> CognitiveModelType:
        """Return the type of this cognitive model"""
        pass

    @abstractmethod
    async def apply_model(
        self, context: ModelApplicationContext
    ) -> ModelApplicationResult:
        """Apply this cognitive model to the given context"""
        pass

    @abstractmethod
    def _build_prompt(self, context: ModelApplicationContext) -> str:
        """Build model-specific prompt for LLM"""
        pass

    @abstractmethod
    def _validate_output_quality(self, result: ModelApplicationResult) -> bool:
        """Validate the quality of model output"""
        pass

    async def apply_with_quality_validation(
        self, context: ModelApplicationContext
    ) -> ModelApplicationResult:
        """Apply model with automatic quality validation and retry logic"""
        start_time = time.time()
        max_retries = 3
        best_result = None

        for attempt in range(max_retries):
            try:
                self.logger.info(
                    f"Applying {self.model_id}, attempt {attempt + 1}/{max_retries}"
                )

                result = await self.apply_model(context)
                result.model_id = self.model_id

                # Validate quality
                quality_valid = self._validate_output_quality(result)
                confidence_valid = result.confidence_score >= self.quality_threshold

                if quality_valid and confidence_valid:
                    self._update_performance_history(result.confidence_score)
                    self._update_usage_stats(result)
                    self.logger.info(
                        f"✅ {self.model_id} successful: confidence={result.confidence_score:.3f}"
                    )
                    return result

                # Store best attempt
                if (
                    best_result is None
                    or result.confidence_score > best_result.confidence_score
                ):
                    best_result = result

                # Adjust approach for retry
                if attempt < max_retries - 1:
                    context = self._adjust_context_for_retry(context, result, attempt)
                    self.logger.warning(
                        f"⚠️ {self.model_id} quality insufficient, retrying..."
                    )

            except Exception as e:
                self.logger.error(
                    f"❌ {self.model_id} failed on attempt {attempt + 1}: {e}"
                )
                if attempt == max_retries - 1:
                    # Return fallback result on final failure
                    return self._create_fallback_result(context, str(e))

        # Return best effort with quality warning
        if best_result:
            best_result.quality_metrics["quality_warning"] = True
            best_result.quality_metrics["retry_attempts"] = max_retries
            self.logger.warning(f"⚠️ {self.model_id} returning best effort result")
            return best_result

        # Final fallback
        return self._create_fallback_result(context, "All attempts failed")

    def _adjust_context_for_retry(
        self,
        context: ModelApplicationContext,
        previous_result: ModelApplicationResult,
        attempt: int,
    ) -> ModelApplicationContext:
        """Adjust context for retry attempt"""

        # Create new context with adjusted parameters
        adjusted_context = ModelApplicationContext(
            problem_statement=context.problem_statement,
            business_context=context.business_context.copy(),
            cognitive_load_level=context.cognitive_load_level,
            quality_requirements=context.quality_requirements.copy(),
            available_evidence=context.available_evidence.copy(),
            user_preferences=context.user_preferences,
        )

        # Adjust quality requirements based on previous failure
        if previous_result.confidence_score < self.quality_threshold:
            # Lower threshold slightly for retry
            adjusted_context.quality_requirements["accuracy_requirement"] *= 0.9

        # Add guidance from previous attempt
        if hasattr(adjusted_context, "retry_guidance"):
            adjusted_context.retry_guidance = f"Previous attempt achieved {previous_result.confidence_score:.3f} confidence. Focus on improving clarity and evidence support."

        return adjusted_context

    def _create_fallback_result(
        self, context: ModelApplicationContext, error_message: str
    ) -> ModelApplicationResult:
        """Create fallback result when all attempts fail"""

        return ModelApplicationResult(
            reasoning_text=f"Fallback analysis for {self.model_type.value}: Unable to complete full analysis due to technical issues. Basic assessment: {context.problem_statement[:200]}...",
            confidence_score=0.3,  # Low confidence fallback
            key_insights=[
                f"Technical limitation encountered in {self.model_type.value} analysis"
            ],
            supporting_evidence=[],
            assumptions_made=[f"Fallback analysis due to: {error_message}"],
            quality_metrics={
                "fallback_result": True,
                "error_message": error_message,
                "quality_warning": True,
            },
            processing_time_ms=0.0,
            model_id=self.model_id,
        )

    def _update_performance_history(self, score: float):
        """Track model performance over time"""
        self.performance_history.append(score)

        # Keep only last 50 scores for memory efficiency
        if len(self.performance_history) > 50:
            self.performance_history.pop(0)

        # Update average confidence
        self.average_confidence = sum(self.performance_history) / len(
            self.performance_history
        )

    def _update_usage_stats(self, result: ModelApplicationResult):
        """Update usage statistics"""
        self.usage_count += 1
        self.total_processing_time += result.processing_time_ms

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this model"""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type.value,
            "usage_count": self.usage_count,
            "average_confidence": self.average_confidence,
            "average_processing_time_ms": self.total_processing_time
            / max(1, self.usage_count),
            "recent_performance": (
                self.performance_history[-10:] if self.performance_history else []
            ),
            "performance_trend": self._calculate_performance_trend(),
        }

    def _calculate_performance_trend(self) -> str:
        """Calculate performance trend direction"""
        if len(self.performance_history) < 5:
            return "insufficient_data"

        recent = self.performance_history[-5:]
        older = (
            self.performance_history[-10:-5]
            if len(self.performance_history) >= 10
            else self.performance_history[:-5]
        )

        if not older:
            return "stable"

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)

        if recent_avg > older_avg + 0.05:
            return "improving"
        elif recent_avg < older_avg - 0.05:
            return "degrading"
        else:
            return "stable"

    def get_average_performance(self) -> float:
        """Get average performance score"""
        return self.average_confidence

    def reset_performance_history(self):
        """Reset performance history (useful for testing)"""
        self.performance_history = []
        self.usage_count = 0
        self.total_processing_time = 0.0
        self.average_confidence = 0.0
