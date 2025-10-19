"""
Human Calibration Infrastructure for CQA Framework
==================================================

Implements human-in-the-loop validation to ensure AI quality ratings
align with human expert judgment.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
from scipy import stats

from src.core.contracts.quality import QualityDimension


class HumanExpertScore(BaseModel):
    """
    Human expert scoring for calibration.

    Attributes:
        expert_id: Identifier of the human expert
        artifact_id: ID of the artifact being scored
        scores: RIVA scores assigned by the expert
        confidence: Expert's confidence in their assessment
        rationale: Expert's reasoning for scores
        timestamp: When the scoring was performed
    """

    expert_id: str
    artifact_id: str
    scores: Dict[QualityDimension, int]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: Dict[QualityDimension, str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    time_spent_seconds: Optional[int] = None

    @field_validator("scores")
    @classmethod
    def validate_all_dimensions_present(cls, v):
        required = {
            QualityDimension.RIGOR,
            QualityDimension.INSIGHT,
            QualityDimension.VALUE,
            QualityDimension.ALIGNMENT,
        }
        if set(v.keys()) != required:
            raise ValueError("All RIVA dimensions must be scored")
        return v


class GoldenSetArtifact(BaseModel):
    """
    A golden set artifact with human expert scores for calibration.

    Attributes:
        artifact_id: Unique identifier
        system_prompt: The system context
        user_prompt: The user query
        llm_response: The response to evaluate
        agent_name: Which agent produced this
        rubric_variant: Which rubric to use for evaluation
        human_scores: List of human expert evaluations
        consensus_score: Aggregated human consensus (if multiple experts)
    """

    artifact_id: str
    system_prompt: str
    user_prompt: str
    llm_response: str
    agent_name: str
    rubric_variant: str = "riva_standard@1.0"
    human_scores: List[HumanExpertScore] = Field(default_factory=list)
    consensus_score: Optional[Dict[QualityDimension, float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def calculate_consensus(self) -> Dict[QualityDimension, float]:
        """
        Calculate consensus scores from multiple human experts.

        Returns:
            Average scores across all experts for each dimension
        """
        if not self.human_scores:
            return {}

        dimension_scores = {dim: [] for dim in QualityDimension}

        for expert_score in self.human_scores:
            for dim, score in expert_score.scores.items():
                dimension_scores[dim].append(score)

        consensus = {}
        for dim, scores in dimension_scores.items():
            if scores:
                consensus[dim] = np.mean(scores)

        self.consensus_score = consensus
        return consensus


class CalibrationResult(BaseModel):
    """
    Results from calibrating AI rater against human scores.

    Attributes:
        rater_version: Version of the AI rater being calibrated
        golden_set_id: ID of the golden set used
        correlation_scores: Correlation between AI and human scores
        dimension_correlations: Per-dimension correlation analysis
        mean_absolute_error: Average difference from human scores
        calibration_passed: Whether calibration meets threshold
        recommendations: Specific improvements needed
    """

    rater_version: str
    golden_set_id: str
    correlation_scores: Dict[str, float]  # Overall and per-dimension
    mean_absolute_error: Dict[QualityDimension, float]
    calibration_passed: bool
    failure_reasons: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def overall_correlation(self) -> float:
        """Get overall correlation score."""
        return self.correlation_scores.get("overall", 0.0)


class CalibrationThresholds(BaseModel):
    """
    Configurable thresholds for calibration validation.

    Attributes:
        min_correlation: Minimum acceptable correlation with human scores
        max_mae: Maximum acceptable mean absolute error
        min_artifacts: Minimum number of golden set artifacts required
        min_experts_per_artifact: Minimum human experts per artifact
    """

    min_correlation: float = Field(default=0.7, ge=0.0, le=1.0)
    max_mae: float = Field(default=1.5, ge=0.0)
    min_artifacts: int = Field(default=50, ge=10)
    min_experts_per_artifact: int = Field(default=2, ge=1)
    require_per_dimension_pass: bool = True


class RaterCalibrationSystem:
    """
    System for validating AI quality raters against human judgment.

    This system ensures that AI-generated quality scores align with
    human expert evaluation before the rater can be trusted in production.
    """

    def __init__(self, thresholds: Optional[CalibrationThresholds] = None):
        """
        Initialize the calibration system.

        Args:
            thresholds: Calibration thresholds (uses defaults if not provided)
        """
        self.thresholds = thresholds or CalibrationThresholds()
        self.golden_sets: Dict[str, List[GoldenSetArtifact]] = {}

    def add_golden_artifact(self, golden_set_id: str, artifact: GoldenSetArtifact):
        """
        Add a golden artifact to a calibration set.

        Args:
            golden_set_id: ID of the golden set
            artifact: The artifact with human scores
        """
        if golden_set_id not in self.golden_sets:
            self.golden_sets[golden_set_id] = []

        # Calculate consensus if multiple experts
        if len(artifact.human_scores) > 1:
            artifact.calculate_consensus()

        self.golden_sets[golden_set_id].append(artifact)

    async def calibrate_rater(
        self, rater, golden_set_id: str  # The AI quality rater to calibrate
    ) -> CalibrationResult:
        """
        Calibrate an AI rater against a golden set.

        Args:
            rater: The AI quality rater agent
            golden_set_id: ID of the golden set to use

        Returns:
            CalibrationResult with correlation analysis and pass/fail
        """
        if golden_set_id not in self.golden_sets:
            raise ValueError(f"Golden set {golden_set_id} not found")

        artifacts = self.golden_sets[golden_set_id]

        # Validate golden set size
        if len(artifacts) < self.thresholds.min_artifacts:
            return CalibrationResult(
                rater_version=getattr(rater, "version", "unknown"),
                golden_set_id=golden_set_id,
                correlation_scores={"overall": 0.0},
                mean_absolute_error={dim: 999.0 for dim in QualityDimension},
                calibration_passed=False,
                failure_reasons=[
                    f"Insufficient golden set size: {len(artifacts)} < {self.thresholds.min_artifacts}"
                ],
            )

        # Collect AI and human scores
        ai_scores = {dim: [] for dim in QualityDimension}
        human_scores = {dim: [] for dim in QualityDimension}

        for artifact in artifacts:
            # Get AI scores
            from src.core.contracts.quality import QualityAuditRequest

            request = QualityAuditRequest(
                system_prompt=artifact.system_prompt,
                user_prompt=artifact.user_prompt,
                llm_response=artifact.llm_response,
                agent_name=artifact.agent_name,
                context={"rubric_variant": artifact.rubric_variant},
            )

            ai_result = await rater.evaluate(request)

            # Use consensus or first expert score
            human_score_dict = (
                artifact.consensus_score
                if artifact.consensus_score
                else artifact.human_scores[0].scores
            )

            # Collect scores for correlation
            ai_scores[QualityDimension.RIGOR].append(ai_result.rigor.score)
            ai_scores[QualityDimension.INSIGHT].append(ai_result.insight.score)
            ai_scores[QualityDimension.VALUE].append(ai_result.value.score)
            ai_scores[QualityDimension.ALIGNMENT].append(ai_result.alignment.score)

            for dim in QualityDimension:
                human_scores[dim].append(human_score_dict[dim])

        # Calculate correlations
        correlations = {}
        mae_scores = {}

        for dim in QualityDimension:
            if len(ai_scores[dim]) > 1:
                correlation, p_value = stats.pearsonr(ai_scores[dim], human_scores[dim])
                correlations[dim.value] = correlation

                # Calculate mean absolute error
                mae = np.mean(
                    np.abs(np.array(ai_scores[dim]) - np.array(human_scores[dim]))
                )
                mae_scores[dim] = mae
            else:
                correlations[dim.value] = 0.0
                mae_scores[dim] = 999.0

        # Overall correlation (average across dimensions)
        correlations["overall"] = np.mean(list(correlations.values()))

        # Determine pass/fail
        passed = True
        failure_reasons = []
        recommendations = []

        # Check overall correlation
        if correlations["overall"] < self.thresholds.min_correlation:
            passed = False
            failure_reasons.append(
                f"Overall correlation {correlations['overall']:.3f} < {self.thresholds.min_correlation}"
            )
            recommendations.append(
                "Revise rater prompts to better align with human judgment"
            )

        # Check per-dimension if required
        if self.thresholds.require_per_dimension_pass:
            for dim in QualityDimension:
                if correlations[dim.value] < self.thresholds.min_correlation:
                    passed = False
                    failure_reasons.append(
                        f"{dim.value} correlation {correlations[dim.value]:.3f} < threshold"
                    )
                    recommendations.append(
                        f"Improve {dim.value} scoring criteria and examples"
                    )

                if mae_scores[dim] > self.thresholds.max_mae:
                    passed = False
                    failure_reasons.append(
                        f"{dim.value} MAE {mae_scores[dim]:.2f} > {self.thresholds.max_mae}"
                    )
                    recommendations.append(
                        f"Reduce scoring variance for {dim.value} dimension"
                    )

        return CalibrationResult(
            rater_version=getattr(rater, "version", "unknown"),
            golden_set_id=golden_set_id,
            correlation_scores=correlations,
            mean_absolute_error=mae_scores,
            calibration_passed=passed,
            failure_reasons=failure_reasons,
            recommendations=recommendations,
        )

    def export_golden_set(self, golden_set_id: str, filepath: str):
        """
        Export a golden set to JSON for backup/sharing.

        Args:
            golden_set_id: ID of the golden set
            filepath: Path to save the JSON file
        """
        import json

        if golden_set_id not in self.golden_sets:
            raise ValueError(f"Golden set {golden_set_id} not found")

        artifacts = self.golden_sets[golden_set_id]

        export_data = {
            "golden_set_id": golden_set_id,
            "artifact_count": len(artifacts),
            "artifacts": [artifact.dict() for artifact in artifacts],
            "export_timestamp": datetime.utcnow().isoformat(),
        }

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        print(f"✅ Golden set exported to {filepath}")
