#!/usr/bin/env python3
"""
Confidence Calibration System
Phase 1: Foundation Systems - Systematic Intelligence Amplification

Implements confidence calibration to ensure system confidence levels accurately reflect actual performance,
preventing overconfidence and improving reliability in cognitive intelligence decisions.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import statistics


class ConfidenceCalibrationMethod(str, Enum):
    """Methods for confidence calibration"""

    PLATT_SCALING = "platt_scaling"  # Sigmoid-based calibration
    ISOTONIC_REGRESSION = "isotonic_regression"  # Monotonic calibration
    BAYESIAN_CALIBRATION = "bayesian_calibration"  # Bayesian approach
    HISTOGRAM_BINNING = "histogram_binning"  # Binning-based calibration
    TEMPERATURE_SCALING = "temperature_scaling"  # Neural network temperature scaling


@dataclass
class ConfidenceObservation:
    """Single confidence observation with outcome"""

    model_id: str
    predicted_confidence: float  # What the model claimed (0.0-1.0)
    actual_outcome: float  # What actually happened (0.0-1.0)
    context: Dict[str, Any]
    observation_time: datetime = field(default_factory=datetime.utcnow)
    engagement_id: Optional[str] = None
    problem_complexity: Optional[str] = None
    domain: Optional[str] = None


class CalibrationBin(NamedTuple):
    """Calibration bin for reliability diagrams"""

    confidence_lower: float
    confidence_upper: float
    predicted_confidence: float  # Average predicted confidence in bin
    actual_accuracy: float  # Actual accuracy in bin
    sample_count: int
    confidence_error: float  # |predicted - actual|


@dataclass
class CalibrationMetrics:
    """Comprehensive calibration assessment metrics"""

    expected_calibration_error: float  # ECE - primary calibration metric
    maximum_calibration_error: float  # MCE - worst-case calibration error
    average_confidence: float  # Average predicted confidence
    average_accuracy: float  # Average actual accuracy
    brier_score: float  # Probabilistic accuracy measure
    reliability: float  # How well calibrated (0-1, 1=perfect)
    resolution: float  # Ability to discriminate between outcomes
    sharpness: float  # Confidence in predictions (variance)

    # Bin-based analysis
    calibration_bins: List[CalibrationBin] = field(default_factory=list)
    underconfident_rate: float = 0.0  # % of underconfident predictions
    overconfident_rate: float = 0.0  # % of overconfident predictions

    # Temporal analysis
    calibration_trend: str = "stable"  # improving, declining, stable
    recent_ece: float = 0.0  # ECE for recent observations
    historical_ece: float = 0.0  # ECE for historical observations


@dataclass
class CalibrationModel:
    """Confidence calibration model for a specific mental model"""

    model_id: str
    calibration_method: ConfidenceCalibrationMethod

    # Calibration parameters (method-dependent)
    calibration_parameters: Dict[str, Any] = field(default_factory=dict)

    # Training data
    observations: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Performance tracking
    last_calibration_update: datetime = field(default_factory=datetime.utcnow)
    calibration_metrics: Optional[CalibrationMetrics] = None
    calibration_quality_score: float = 0.5

    # Reliability tracking
    total_predictions: int = 0
    correct_predictions: int = 0
    total_confidence_error: float = 0.0


class ConfidenceCalibrationEngine:
    """
    Engine for confidence calibration across mental models
    Ensures predicted confidence levels match actual performance
    """

    def __init__(
        self,
        calibration_method: ConfidenceCalibrationMethod = ConfidenceCalibrationMethod.BAYESIAN_CALIBRATION,
        min_observations_for_calibration: int = 20,
        calibration_update_frequency: timedelta = timedelta(hours=24),
        num_bins: int = 10,
    ):

        self.logger = logging.getLogger(__name__)
        self.calibration_method = calibration_method
        self.min_observations_for_calibration = min_observations_for_calibration
        self.calibration_update_frequency = calibration_update_frequency
        self.num_bins = num_bins

        # Model-specific calibration models
        self.calibration_models: Dict[str, CalibrationModel] = {}

        # Global calibration tracking
        self.global_observations: deque = deque(maxlen=5000)
        self.global_calibration_metrics: Optional[CalibrationMetrics] = None

        # Performance thresholds
        self.excellent_ece_threshold = 0.05  # ECE < 5% = excellent calibration
        self.good_ece_threshold = 0.10  # ECE < 10% = good calibration
        self.acceptable_ece_threshold = 0.20  # ECE < 20% = acceptable calibration

        self.logger.info(
            "✅ ConfidenceCalibrationEngine initialized - Confidence calibration active"
        )

    async def record_confidence_observation(
        self,
        model_id: str,
        predicted_confidence: float,
        actual_outcome: float,
        context: Dict[str, Any] = None,
        engagement_id: str = None,
    ) -> None:
        """Record a confidence prediction and its actual outcome"""

        observation = ConfidenceObservation(
            model_id=model_id,
            predicted_confidence=max(
                0.0, min(1.0, predicted_confidence)
            ),  # Clamp to [0,1]
            actual_outcome=max(0.0, min(1.0, actual_outcome)),
            context=context or {},
            engagement_id=engagement_id,
            problem_complexity=context.get("complexity_level") if context else None,
            domain=context.get("domain") if context else None,
        )

        # Initialize calibration model if needed
        if model_id not in self.calibration_models:
            self.calibration_models[model_id] = CalibrationModel(
                model_id=model_id, calibration_method=self.calibration_method
            )

        calibration_model = self.calibration_models[model_id]

        # Add observation to model and global tracking
        calibration_model.observations.append(observation)
        self.global_observations.append(observation)

        # Update running statistics
        calibration_model.total_predictions += 1
        if (
            abs(actual_outcome - predicted_confidence) < 0.1
        ):  # Within 10% = "correct confidence"
            calibration_model.correct_predictions += 1

        calibration_model.total_confidence_error += abs(
            actual_outcome - predicted_confidence
        )

        # Log observation
        confidence_error = abs(actual_outcome - predicted_confidence)
        self.logger.debug(
            f"📊 Confidence observation: {model_id} | "
            f"Predicted: {predicted_confidence:.3f} | "
            f"Actual: {actual_outcome:.3f} | "
            f"Error: {confidence_error:.3f}"
        )

        # Trigger calibration update if needed
        time_since_update = (
            datetime.utcnow() - calibration_model.last_calibration_update
        )
        if (
            len(calibration_model.observations) >= self.min_observations_for_calibration
            and time_since_update >= self.calibration_update_frequency
        ):

            await self._update_model_calibration(model_id)

    async def _update_model_calibration(self, model_id: str) -> None:
        """Update calibration model for a specific mental model"""

        calibration_model = self.calibration_models[model_id]
        observations = list(calibration_model.observations)

        if len(observations) < self.min_observations_for_calibration:
            return

        # Calculate calibration metrics
        metrics = self._calculate_calibration_metrics(observations)
        calibration_model.calibration_metrics = metrics

        # Update calibration parameters based on method
        if self.calibration_method == ConfidenceCalibrationMethod.BAYESIAN_CALIBRATION:
            calibration_model.calibration_parameters = self._fit_bayesian_calibration(
                observations
            )
        elif self.calibration_method == ConfidenceCalibrationMethod.PLATT_SCALING:
            calibration_model.calibration_parameters = self._fit_platt_scaling(
                observations
            )
        elif self.calibration_method == ConfidenceCalibrationMethod.HISTOGRAM_BINNING:
            calibration_model.calibration_parameters = self._fit_histogram_binning(
                observations
            )
        elif self.calibration_method == ConfidenceCalibrationMethod.TEMPERATURE_SCALING:
            calibration_model.calibration_parameters = self._fit_temperature_scaling(
                observations
            )

        # Update calibration quality score
        calibration_model.calibration_quality_score = (
            self._calculate_calibration_quality(metrics)
        )
        calibration_model.last_calibration_update = datetime.utcnow()

        self.logger.info(
            f"📈 Updated calibration for {model_id}: "
            f"ECE = {metrics.expected_calibration_error:.3f}, "
            f"Quality = {calibration_model.calibration_quality_score:.3f}"
        )

    def _calculate_calibration_metrics(
        self, observations: List[ConfidenceObservation]
    ) -> CalibrationMetrics:
        """Calculate comprehensive calibration metrics"""

        if not observations:
            return CalibrationMetrics(
                expected_calibration_error=1.0,
                maximum_calibration_error=1.0,
                average_confidence=0.5,
                average_accuracy=0.5,
                brier_score=0.5,
                reliability=0.0,
                resolution=0.0,
                sharpness=0.0,
            )

        confidences = [obs.predicted_confidence for obs in observations]
        outcomes = [obs.actual_outcome for obs in observations]

        # Basic statistics
        avg_confidence = statistics.mean(confidences)
        avg_accuracy = statistics.mean(outcomes)

        # Brier score (lower is better)
        brier_score = statistics.mean(
            [(conf - outcome) ** 2 for conf, outcome in zip(confidences, outcomes)]
        )

        # Create calibration bins
        bins = self._create_calibration_bins(observations)

        # Expected Calibration Error (ECE)
        ece = 0.0
        mce = 0.0  # Maximum Calibration Error
        total_samples = len(observations)

        for bin_data in bins:
            if bin_data.sample_count > 0:
                bin_weight = bin_data.sample_count / total_samples
                bin_error = abs(
                    bin_data.predicted_confidence - bin_data.actual_accuracy
                )
                ece += bin_weight * bin_error
                mce = max(mce, bin_error)

        # Reliability decomposition
        reliability = statistics.mean(
            [
                bin_data.confidence_error**2 * (bin_data.sample_count / total_samples)
                for bin_data in bins
                if bin_data.sample_count > 0
            ]
        )

        # Resolution (ability to discriminate)
        accuracy_values = [
            bin_data.actual_accuracy for bin_data in bins if bin_data.sample_count > 0
        ]
        if len(accuracy_values) > 1:
            resolution = statistics.variance(accuracy_values)
        else:
            resolution = 0.0

        # Sharpness (confidence variance)
        sharpness = statistics.variance(confidences) if len(confidences) > 1 else 0.0

        # Over/underconfidence analysis
        overconfident_count = sum(
            1 for conf, outcome in zip(confidences, outcomes) if conf > outcome + 0.1
        )
        underconfident_count = sum(
            1 for conf, outcome in zip(confidences, outcomes) if conf < outcome - 0.1
        )

        overconfident_rate = overconfident_count / total_samples
        underconfident_rate = underconfident_count / total_samples

        # Temporal trend analysis
        if len(observations) >= 10:
            recent_obs = observations[-min(len(observations) // 2, 50) :]
            historical_obs = observations[: len(observations) // 2]

            recent_ece = self._calculate_ece_for_observations(recent_obs)
            historical_ece = self._calculate_ece_for_observations(historical_obs)

            if recent_ece < historical_ece - 0.02:
                trend = "improving"
            elif recent_ece > historical_ece + 0.02:
                trend = "declining"
            else:
                trend = "stable"
        else:
            recent_ece = ece
            historical_ece = ece
            trend = "stable"

        return CalibrationMetrics(
            expected_calibration_error=ece,
            maximum_calibration_error=mce,
            average_confidence=avg_confidence,
            average_accuracy=avg_accuracy,
            brier_score=brier_score,
            reliability=reliability,
            resolution=resolution,
            sharpness=sharpness,
            calibration_bins=bins,
            overconfident_rate=overconfident_rate,
            underconfident_rate=underconfident_rate,
            calibration_trend=trend,
            recent_ece=recent_ece,
            historical_ece=historical_ece,
        )

    def _create_calibration_bins(
        self, observations: List[ConfidenceObservation]
    ) -> List[CalibrationBin]:
        """Create calibration bins for reliability diagram"""

        bins = []
        bin_size = 1.0 / self.num_bins

        for i in range(self.num_bins):
            bin_lower = i * bin_size
            bin_upper = (i + 1) * bin_size

            # Find observations in this bin
            bin_observations = [
                obs
                for obs in observations
                if bin_lower <= obs.predicted_confidence < bin_upper
                or (i == self.num_bins - 1 and obs.predicted_confidence == 1.0)
            ]

            if bin_observations:
                avg_confidence = statistics.mean(
                    [obs.predicted_confidence for obs in bin_observations]
                )
                avg_accuracy = statistics.mean(
                    [obs.actual_outcome for obs in bin_observations]
                )
                confidence_error = abs(avg_confidence - avg_accuracy)
            else:
                avg_confidence = (bin_lower + bin_upper) / 2
                avg_accuracy = 0.0
                confidence_error = 0.0

            bins.append(
                CalibrationBin(
                    confidence_lower=bin_lower,
                    confidence_upper=bin_upper,
                    predicted_confidence=avg_confidence,
                    actual_accuracy=avg_accuracy,
                    sample_count=len(bin_observations),
                    confidence_error=confidence_error,
                )
            )

        return bins

    def _calculate_ece_for_observations(
        self, observations: List[ConfidenceObservation]
    ) -> float:
        """Calculate ECE for a subset of observations"""
        if not observations:
            return 0.0

        bins = self._create_calibration_bins(observations)
        ece = 0.0
        total_samples = len(observations)

        for bin_data in bins:
            if bin_data.sample_count > 0:
                bin_weight = bin_data.sample_count / total_samples
                bin_error = abs(
                    bin_data.predicted_confidence - bin_data.actual_accuracy
                )
                ece += bin_weight * bin_error

        return ece

    def _fit_bayesian_calibration(
        self, observations: List[ConfidenceObservation]
    ) -> Dict[str, Any]:
        """Fit Bayesian calibration model"""

        confidences = np.array([obs.predicted_confidence for obs in observations])
        outcomes = np.array([obs.actual_outcome for obs in observations])

        # Simple Bayesian approach: Beta-Binomial conjugate
        # Prior parameters (weak prior)
        alpha_prior = 1.0
        beta_prior = 1.0

        # Update parameters based on observations
        successes = np.sum(outcomes)
        failures = len(outcomes) - successes

        alpha_posterior = alpha_prior + successes
        beta_posterior = beta_prior + failures

        # Calibration mapping parameters
        calibration_params = {
            "method": "bayesian",
            "alpha_posterior": alpha_posterior,
            "beta_posterior": beta_posterior,
            "mean_calibration": alpha_posterior / (alpha_posterior + beta_posterior),
            "confidence_adjustment": self._calculate_confidence_adjustment(
                confidences, outcomes
            ),
        }

        return calibration_params

    def _fit_platt_scaling(
        self, observations: List[ConfidenceObservation]
    ) -> Dict[str, Any]:
        """Fit Platt scaling (sigmoid) calibration"""

        # Convert confidences to logits (inverse sigmoid)
        confidences = np.array(
            [max(0.001, min(0.999, obs.predicted_confidence)) for obs in observations]
        )
        logits = np.log(confidences / (1 - confidences))
        outcomes = np.array([obs.actual_outcome for obs in observations])

        # Fit sigmoid: P_calibrated = 1 / (1 + exp(-(A * logit + B)))
        # Using simple least squares approximation
        X = np.vstack([logits, np.ones(len(logits))]).T
        try:
            A, B = np.linalg.lstsq(X, outcomes, rcond=None)[0]
        except:
            A, B = 1.0, 0.0  # Fallback to identity mapping

        return {
            "method": "platt_scaling",
            "slope": A,
            "intercept": B,
            "scaling_function": lambda conf: 1.0
            / (
                1.0
                + np.exp(
                    -(
                        A
                        * np.log(
                            max(0.001, min(0.999, conf))
                            / (1 - max(0.001, min(0.999, conf)))
                        )
                        + B
                    )
                )
            ),
        }

    def _fit_histogram_binning(
        self, observations: List[ConfidenceObservation]
    ) -> Dict[str, Any]:
        """Fit histogram binning calibration"""

        bins = self._create_calibration_bins(observations)

        # Create calibration mapping
        calibration_mapping = {}
        for bin_data in bins:
            bin_center = (bin_data.confidence_lower + bin_data.confidence_upper) / 2
            if bin_data.sample_count > 0:
                calibration_mapping[bin_center] = bin_data.actual_accuracy
            else:
                calibration_mapping[bin_center] = (
                    bin_center  # Identity mapping for empty bins
                )

        return {
            "method": "histogram_binning",
            "bin_mapping": calibration_mapping,
            "num_bins": self.num_bins,
        }

    def _fit_temperature_scaling(
        self, observations: List[ConfidenceObservation]
    ) -> Dict[str, Any]:
        """Fit temperature scaling calibration"""

        confidences = np.array([obs.predicted_confidence for obs in observations])
        outcomes = np.array([obs.actual_outcome for obs in observations])

        # Convert to logits for temperature scaling
        logits = np.log(
            np.maximum(0.001, confidences) / np.maximum(0.001, 1 - confidences)
        )

        # Find optimal temperature using simple optimization
        best_temperature = 1.0
        best_loss = float("inf")

        for temp in np.arange(0.1, 3.0, 0.1):
            scaled_probs = 1 / (1 + np.exp(-logits / temp))
            loss = np.mean((scaled_probs - outcomes) ** 2)  # MSE loss

            if loss < best_loss:
                best_loss = loss
                best_temperature = temp

        return {
            "method": "temperature_scaling",
            "temperature": best_temperature,
            "scaling_function": lambda conf: 1.0
            / (
                1.0
                + np.exp(
                    -np.log(
                        max(0.001, min(0.999, conf))
                        / (1 - max(0.001, min(0.999, conf)))
                    )
                    / best_temperature
                )
            ),
        }

    def _calculate_confidence_adjustment(
        self, confidences: np.ndarray, outcomes: np.ndarray
    ) -> float:
        """Calculate confidence adjustment factor"""

        if len(confidences) == 0:
            return 1.0

        avg_confidence = np.mean(confidences)
        avg_outcome = np.mean(outcomes)

        # Adjustment factor to bring predictions closer to actual outcomes
        if avg_confidence > 0:
            adjustment = avg_outcome / avg_confidence
        else:
            adjustment = 1.0

        # Clamp adjustment to reasonable range
        return max(0.5, min(2.0, adjustment))

    def _calculate_calibration_quality(self, metrics: CalibrationMetrics) -> float:
        """Calculate overall calibration quality score (0-1, higher is better)"""

        # Primary factor: ECE (lower is better)
        ece_score = max(
            0.0, 1.0 - metrics.expected_calibration_error / 0.2
        )  # Normalize to 0-1

        # Secondary factors
        resolution_score = min(1.0, metrics.resolution * 5)  # Ability to discriminate
        sharpness_score = min(1.0, metrics.sharpness * 2)  # Confidence variance

        # Bias penalty (overconfidence/underconfidence)
        bias_penalty = (metrics.overconfident_rate + metrics.underconfident_rate) / 2

        # Weighted combination
        quality_score = (
            0.6 * ece_score  # Primary: calibration accuracy
            + 0.2 * resolution_score  # Secondary: discriminative ability
            + 0.1 * sharpness_score  # Secondary: confidence spread
            + 0.1 * (1.0 - bias_penalty)  # Penalty: systematic bias
        )

        return max(0.0, min(1.0, quality_score))

    def calibrate_confidence(
        self, model_id: str, raw_confidence: float
    ) -> Tuple[float, Dict[str, Any]]:
        """Apply calibration to raw confidence score"""

        if model_id not in self.calibration_models:
            # No calibration data available, return raw confidence with warning
            return raw_confidence, {
                "calibration_applied": False,
                "reason": "no_calibration_data",
                "raw_confidence": raw_confidence,
            }

        calibration_model = self.calibration_models[model_id]

        if not calibration_model.calibration_parameters:
            # Calibration not yet computed
            return raw_confidence, {
                "calibration_applied": False,
                "reason": "calibration_not_computed",
                "raw_confidence": raw_confidence,
            }

        # Apply calibration based on method
        method = calibration_model.calibration_parameters.get("method")

        try:
            if method == "bayesian":
                calibrated_confidence = self._apply_bayesian_calibration(
                    raw_confidence, calibration_model.calibration_parameters
                )
            elif method == "platt_scaling":
                calibrated_confidence = self._apply_platt_scaling(
                    raw_confidence, calibration_model.calibration_parameters
                )
            elif method == "histogram_binning":
                calibrated_confidence = self._apply_histogram_binning(
                    raw_confidence, calibration_model.calibration_parameters
                )
            elif method == "temperature_scaling":
                calibrated_confidence = self._apply_temperature_scaling(
                    raw_confidence, calibration_model.calibration_parameters
                )
            else:
                calibrated_confidence = raw_confidence

            # Ensure calibrated confidence is in valid range
            calibrated_confidence = max(0.0, min(1.0, calibrated_confidence))

            return calibrated_confidence, {
                "calibration_applied": True,
                "method": method,
                "raw_confidence": raw_confidence,
                "calibrated_confidence": calibrated_confidence,
                "adjustment": calibrated_confidence - raw_confidence,
                "calibration_quality": calibration_model.calibration_quality_score,
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Calibration failed for {model_id}: {e}")
            return raw_confidence, {
                "calibration_applied": False,
                "reason": "calibration_error",
                "error": str(e),
                "raw_confidence": raw_confidence,
            }

    def _apply_bayesian_calibration(
        self, confidence: float, params: Dict[str, Any]
    ) -> float:
        """Apply Bayesian calibration"""
        adjustment = params.get("confidence_adjustment", 1.0)
        return confidence * adjustment

    def _apply_platt_scaling(self, confidence: float, params: Dict[str, Any]) -> float:
        """Apply Platt scaling calibration"""
        scaling_function = params.get("scaling_function")
        if scaling_function:
            return scaling_function(confidence)
        return confidence

    def _apply_histogram_binning(
        self, confidence: float, params: Dict[str, Any]
    ) -> float:
        """Apply histogram binning calibration"""
        bin_mapping = params.get("bin_mapping", {})
        num_bins = params.get("num_bins", 10)

        # Find appropriate bin
        bin_size = 1.0 / num_bins
        bin_index = min(num_bins - 1, int(confidence // bin_size))
        bin_center = (bin_index + 0.5) * bin_size

        return bin_mapping.get(bin_center, confidence)

    def _apply_temperature_scaling(
        self, confidence: float, params: Dict[str, Any]
    ) -> float:
        """Apply temperature scaling calibration"""
        scaling_function = params.get("scaling_function")
        if scaling_function:
            return scaling_function(confidence)
        return confidence

    def get_model_calibration_status(self, model_id: str) -> Dict[str, Any]:
        """Get calibration status for a specific model"""

        if model_id not in self.calibration_models:
            return {
                "model_id": model_id,
                "calibration_available": False,
                "reason": "no_observations",
            }

        calibration_model = self.calibration_models[model_id]
        metrics = calibration_model.calibration_metrics

        status = {
            "model_id": model_id,
            "calibration_available": metrics is not None,
            "calibration_method": calibration_model.calibration_method.value,
            "observations_count": len(calibration_model.observations),
            "total_predictions": calibration_model.total_predictions,
            "calibration_quality_score": calibration_model.calibration_quality_score,
            "last_update": calibration_model.last_calibration_update.isoformat(),
        }

        if metrics:
            # Determine calibration assessment
            if metrics.expected_calibration_error <= self.excellent_ece_threshold:
                calibration_assessment = "excellent"
            elif metrics.expected_calibration_error <= self.good_ece_threshold:
                calibration_assessment = "good"
            elif metrics.expected_calibration_error <= self.acceptable_ece_threshold:
                calibration_assessment = "acceptable"
            else:
                calibration_assessment = "needs_improvement"

            status.update(
                {
                    "calibration_assessment": calibration_assessment,
                    "expected_calibration_error": metrics.expected_calibration_error,
                    "maximum_calibration_error": metrics.maximum_calibration_error,
                    "average_confidence": metrics.average_confidence,
                    "average_accuracy": metrics.average_accuracy,
                    "brier_score": metrics.brier_score,
                    "overconfident_rate": metrics.overconfident_rate,
                    "underconfident_rate": metrics.underconfident_rate,
                    "calibration_trend": metrics.calibration_trend,
                    "bins_with_data": len(
                        [b for b in metrics.calibration_bins if b.sample_count > 0]
                    ),
                }
            )

        return status

    def get_global_calibration_status(self) -> Dict[str, Any]:
        """Get overall system calibration status"""

        if not self.global_observations:
            return {
                "global_calibration_available": False,
                "total_models": len(self.calibration_models),
                "reason": "no_observations",
            }

        # Calculate global metrics
        global_metrics = self._calculate_calibration_metrics(
            list(self.global_observations)
        )

        # Model-level statistics
        model_calibration_scores = [
            model.calibration_quality_score
            for model in self.calibration_models.values()
            if model.calibration_metrics is not None
        ]

        excellent_models = sum(
            1
            for model in self.calibration_models.values()
            if model.calibration_metrics
            and model.calibration_metrics.expected_calibration_error
            <= self.excellent_ece_threshold
        )

        good_models = sum(
            1
            for model in self.calibration_models.values()
            if model.calibration_metrics
            and self.excellent_ece_threshold
            < model.calibration_metrics.expected_calibration_error
            <= self.good_ece_threshold
        )

        return {
            "global_calibration_available": True,
            "total_observations": len(self.global_observations),
            "total_models": len(self.calibration_models),
            "models_with_calibration": len(
                [m for m in self.calibration_models.values() if m.calibration_metrics]
            ),
            # Global metrics
            "global_expected_calibration_error": global_metrics.expected_calibration_error,
            "global_average_confidence": global_metrics.average_confidence,
            "global_average_accuracy": global_metrics.average_accuracy,
            "global_brier_score": global_metrics.brier_score,
            # System-level assessment
            "excellent_calibration_models": excellent_models,
            "good_calibration_models": good_models,
            "average_model_quality_score": (
                statistics.mean(model_calibration_scores)
                if model_calibration_scores
                else 0.0
            ),
            # Trend analysis
            "calibration_trend": global_metrics.calibration_trend,
            "overconfidence_rate": global_metrics.overconfident_rate,
            "underconfidence_rate": global_metrics.underconfident_rate,
        }

    def get_system_health_metrics(self) -> Dict[str, Any]:
        """Get system health metrics from calibration perspective"""

        global_status = self.get_global_calibration_status()

        if not global_status.get("global_calibration_available"):
            return {
                "status": "no_data",
                "models_tracked": len(self.calibration_models),
                "calibration_health": "unknown",
            }

        # Determine overall health
        global_ece = global_status.get("global_expected_calibration_error", 1.0)
        avg_quality = global_status.get("average_model_quality_score", 0.0)
        excellent_models = global_status.get("excellent_calibration_models", 0)
        total_models = global_status.get("models_with_calibration", 1)

        excellent_ratio = excellent_models / max(1, total_models)

        if (
            global_ece <= self.excellent_ece_threshold
            and avg_quality >= 0.8
            and excellent_ratio >= 0.7
        ):
            health_status = "excellent"
        elif (
            global_ece <= self.good_ece_threshold
            and avg_quality >= 0.6
            and excellent_ratio >= 0.5
        ):
            health_status = "good"
        elif global_ece <= self.acceptable_ece_threshold and avg_quality >= 0.4:
            health_status = "acceptable"
        else:
            health_status = "needs_improvement"

        return {
            "status": health_status,
            "models_tracked": len(self.calibration_models),
            "models_calibrated": global_status.get("models_with_calibration", 0),
            "global_ece": global_ece,
            "average_quality_score": avg_quality,
            "excellent_models_ratio": excellent_ratio,
            "calibration_trend": global_status.get("calibration_trend", "unknown"),
            "total_observations": len(self.global_observations),
        }


# Global ConfidenceCalibrationEngine instance
_confidence_calibration_engine_instance: Optional[ConfidenceCalibrationEngine] = None


def get_confidence_calibration_engine() -> ConfidenceCalibrationEngine:
    """Get or create global ConfidenceCalibrationEngine instance"""
    global _confidence_calibration_engine_instance

    if _confidence_calibration_engine_instance is None:
        _confidence_calibration_engine_instance = ConfidenceCalibrationEngine()

    return _confidence_calibration_engine_instance


async def record_confidence_observation(
    model_id: str,
    predicted_confidence: float,
    actual_outcome: float,
    context: Dict[str, Any] = None,
    engagement_id: str = None,
) -> None:
    """Convenience function to record confidence observation"""
    engine = get_confidence_calibration_engine()
    await engine.record_confidence_observation(
        model_id, predicted_confidence, actual_outcome, context, engagement_id
    )


def calibrate_model_confidence(
    model_id: str, raw_confidence: float
) -> Tuple[float, Dict[str, Any]]:
    """Convenience function to calibrate confidence"""
    engine = get_confidence_calibration_engine()
    return engine.calibrate_confidence(model_id, raw_confidence)
