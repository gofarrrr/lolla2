"""
METIS Phase 2.1.2: Semantic Context Validation Framework
Research Foundation: LangChain validation patterns + N-WAY coherence analysis

Validates context coherence for N-WAY mental model selection and prevents
context degradation and model conflicts.

Performance Targets:
- Context accuracy: 95% validated correctness
- Error reduction: 40% decrease in context-related failures
- Model compatibility: 100% compatibility matrix validation
- Conflict detection: Real-time N-WAY interaction conflict identification
"""

import logging
from typing import Dict, List, Any, Set, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# Import METIS core components
from src.engine.models.data_contracts import (
    CognitiveState,
)


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ValidationErrorType(str, Enum):
    """Types of validation errors"""

    MODEL_INCOMPATIBILITY = "model_incompatibility"
    NWAY_CONFLICT = "nway_conflict"
    CONTEXT_DEGRADATION = "context_degradation"
    COMPLEXITY_MISMATCH = "complexity_mismatch"
    COHERENCE_FAILURE = "coherence_failure"
    REASONING_INCONSISTENCY = "reasoning_inconsistency"


@dataclass
class ValidationError:
    """Individual validation error with context and remediation"""

    error_type: ValidationErrorType
    severity: ValidationSeverity
    message: str
    affected_models: List[str]
    context_location: str
    remediation_suggestion: str
    confidence_score: float
    detection_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ContextNode:
    """Node in context chain for validation analysis"""

    node_id: str
    content: str
    mental_models: List[str]
    reasoning_step: str
    coherence_score: float
    dependencies: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelCompatibilityScore:
    """Compatibility assessment between mental models"""

    model_pair: Tuple[str, str]
    compatibility_score: float
    interaction_type: str
    potential_conflicts: List[str]
    synergy_potential: float
    context_sensitivity: str


@dataclass
class NWayConflictAnalysis:
    """Analysis of N-WAY interaction conflicts"""

    conflict_id: str
    conflicting_models: List[str]
    conflict_type: str
    severity: ValidationSeverity
    impact_description: str
    resolution_strategies: List[str]
    confidence_score: float


@dataclass
class ValidationResult:
    """Comprehensive validation result"""

    valid: bool
    coherence_score: float
    mental_model_compatibility: float
    nway_conflict_score: float
    complexity_progression_valid: bool
    errors: List[ValidationError]
    warnings: List[str]
    validation_timestamp: datetime = field(default_factory=datetime.now)
    processing_time_ms: float = 0.0


class MentalModelCompatibilityMatrix:
    """
    Matrix-based compatibility assessment for mental models
    Based on research-validated interaction patterns
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.compatibility_matrix = self._initialize_compatibility_matrix()
        self.interaction_patterns = self._load_interaction_patterns()

    def _initialize_compatibility_matrix(
        self,
    ) -> Dict[Tuple[str, str], ModelCompatibilityScore]:
        """Initialize compatibility matrix with research-validated scores"""

        # Core mental models used in METIS
        models = [
            "systems_thinking",
            "critical_thinking",
            "mece_structuring",
            "hypothesis_testing",
            "multi_criteria_analysis",
            "strategic_thinking",
            "evidence_based_reasoning",
            "divergent_thinking",
            "lateral_thinking",
        ]

        matrix = {}

        # Define high-compatibility pairs (research-validated)
        high_compatibility_pairs = [
            ("systems_thinking", "critical_thinking"),
            ("critical_thinking", "evidence_based_reasoning"),
            ("mece_structuring", "systems_thinking"),
            ("hypothesis_testing", "evidence_based_reasoning"),
            ("strategic_thinking", "systems_thinking"),
            ("divergent_thinking", "lateral_thinking"),
        ]

        # Medium compatibility pairs
        medium_compatibility_pairs = [
            ("mece_structuring", "multi_criteria_analysis"),
            ("hypothesis_testing", "critical_thinking"),
            ("strategic_thinking", "multi_criteria_analysis"),
        ]

        # Generate compatibility scores for all pairs
        for i, model_a in enumerate(models):
            for model_b in models[i + 1 :]:
                pair = (model_a, model_b)

                if pair in high_compatibility_pairs:
                    score = ModelCompatibilityScore(
                        model_pair=pair,
                        compatibility_score=0.92,
                        interaction_type="synergistic",
                        potential_conflicts=[],
                        synergy_potential=0.88,
                        context_sensitivity="low",
                    )
                elif pair in medium_compatibility_pairs:
                    score = ModelCompatibilityScore(
                        model_pair=pair,
                        compatibility_score=0.78,
                        interaction_type="complementary",
                        potential_conflicts=["cognitive_load"],
                        synergy_potential=0.65,
                        context_sensitivity="medium",
                    )
                else:
                    # Calculate dynamic compatibility
                    dynamic_score = self._calculate_dynamic_compatibility(
                        model_a, model_b
                    )
                    score = ModelCompatibilityScore(
                        model_pair=pair,
                        compatibility_score=dynamic_score,
                        interaction_type="neutral",
                        potential_conflicts=(
                            ["model_interference"] if dynamic_score < 0.5 else []
                        ),
                        synergy_potential=max(dynamic_score - 0.2, 0.0),
                        context_sensitivity="high",
                    )

                matrix[pair] = score
                # Add reverse pair
                matrix[(model_b, model_a)] = score

        return matrix

    def _calculate_dynamic_compatibility(self, model_a: str, model_b: str) -> float:
        """Calculate dynamic compatibility between models"""

        # Model characteristics for compatibility calculation
        model_characteristics = {
            "systems_thinking": {
                "scope": "holistic",
                "approach": "synthetic",
                "complexity": "high",
            },
            "critical_thinking": {
                "scope": "analytical",
                "approach": "evaluative",
                "complexity": "medium",
            },
            "mece_structuring": {
                "scope": "structured",
                "approach": "logical",
                "complexity": "medium",
            },
            "hypothesis_testing": {
                "scope": "empirical",
                "approach": "scientific",
                "complexity": "high",
            },
            "strategic_thinking": {
                "scope": "planning",
                "approach": "forward_looking",
                "complexity": "high",
            },
            "evidence_based_reasoning": {
                "scope": "factual",
                "approach": "data_driven",
                "complexity": "medium",
            },
        }

        chars_a = model_characteristics.get(model_a, {})
        chars_b = model_characteristics.get(model_b, {})

        if not chars_a or not chars_b:
            return 0.6  # Default neutral compatibility

        # Calculate compatibility based on characteristics alignment
        scope_compat = 1.0 if chars_a.get("scope") == chars_b.get("scope") else 0.3
        approach_compat = (
            1.0 if chars_a.get("approach") == chars_b.get("approach") else 0.5
        )
        complexity_compat = (
            1.0 if chars_a.get("complexity") == chars_b.get("complexity") else 0.7
        )

        # Weighted average
        compatibility = (
            scope_compat * 0.4 + approach_compat * 0.4 + complexity_compat * 0.2
        )

        return min(compatibility, 0.95)  # Cap at 95%

    def _load_interaction_patterns(self) -> Dict[str, Any]:
        """Load N-WAY interaction patterns for conflict detection"""
        return {
            "conflicting_patterns": [
                {
                    "models": ["divergent_thinking", "critical_thinking"],
                    "conflict_type": "approach_mismatch",
                    "severity": "medium",
                    "description": "Divergent thinking may conflict with critical evaluation",
                },
                {
                    "models": ["systems_thinking", "mece_structuring"],
                    "conflict_type": "scope_overlap",
                    "severity": "low",
                    "description": "Potential redundancy in structural analysis",
                },
            ],
            "synergistic_patterns": [
                {
                    "models": [
                        "systems_thinking",
                        "critical_thinking",
                        "evidence_based_reasoning",
                    ],
                    "synergy_type": "analytical_rigor",
                    "amplification_factor": 1.8,
                }
            ],
        }

    async def assess_model_compatibility(
        self, selected_models: List[str]
    ) -> Tuple[float, List[ModelCompatibilityScore]]:
        """
        Assess compatibility of selected mental models
        Target: 100% compatibility matrix validation
        """

        if len(selected_models) < 2:
            return 1.0, []  # Single model always compatible

        compatibility_scores = []
        total_compatibility = 0.0
        pair_count = 0

        # Check all pairs of selected models
        for i, model_a in enumerate(selected_models):
            for model_b in selected_models[i + 1 :]:
                pair = (model_a, model_b)

                compatibility = self.compatibility_matrix.get(pair)
                if compatibility:
                    compatibility_scores.append(compatibility)
                    total_compatibility += compatibility.compatibility_score
                    pair_count += 1

        # Calculate overall compatibility
        overall_compatibility = (
            total_compatibility / pair_count if pair_count > 0 else 1.0
        )

        self.logger.info(
            f"Model compatibility assessment: {overall_compatibility:.3f} for {len(selected_models)} models"
        )

        return overall_compatibility, compatibility_scores


class NWayConflictDetector:
    """
    Detects conflicts in N-WAY mental model interactions
    Identifies potential interference and resolution strategies
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.conflict_patterns = self._load_conflict_patterns()

    def _load_conflict_patterns(self) -> Dict[str, Any]:
        """Load known conflict patterns from research"""
        return {
            "cognitive_overload": {
                "trigger_condition": "too_many_analytical_models",
                "models_threshold": 4,
                "severity": ValidationSeverity.HIGH,
                "description": "Too many analytical models may cause cognitive overload",
            },
            "approach_conflicts": {
                "conflicting_pairs": [
                    ("divergent_thinking", "critical_thinking"),
                    ("lateral_thinking", "logical_reasoning"),
                ],
                "severity": ValidationSeverity.MEDIUM,
                "description": "Conflicting cognitive approaches may reduce effectiveness",
            },
            "scope_redundancy": {
                "redundant_groups": [
                    ["systems_thinking", "holistic_analysis"],
                    ["mece_structuring", "logical_frameworks"],
                ],
                "severity": ValidationSeverity.LOW,
                "description": "Redundant models may not add value",
            },
        }

    async def detect_nway_conflicts(
        self, selected_models: List[str], context_chain: List[ContextNode]
    ) -> NWayConflictAnalysis:
        """
        Detect N-WAY interaction conflicts in selected models
        Target: Real-time conflict detection with resolution strategies
        """

        conflicts = []

        # Check for cognitive overload
        analytical_models = [
            m for m in selected_models if "thinking" in m or "analysis" in m
        ]
        if (
            len(analytical_models)
            > self.conflict_patterns["cognitive_overload"]["models_threshold"]
        ):
            conflicts.append(
                {
                    "type": "cognitive_overload",
                    "severity": ValidationSeverity.HIGH,
                    "models": analytical_models,
                    "description": f"Too many analytical models ({len(analytical_models)}) may cause cognitive overload",
                }
            )

        # Check for approach conflicts
        for pair in self.conflict_patterns["approach_conflicts"]["conflicting_pairs"]:
            if pair[0] in selected_models and pair[1] in selected_models:
                conflicts.append(
                    {
                        "type": "approach_conflict",
                        "severity": ValidationSeverity.MEDIUM,
                        "models": list(pair),
                        "description": f"Conflicting approaches: {pair[0]} vs {pair[1]}",
                    }
                )

        # Generate overall conflict analysis
        if conflicts:
            highest_severity = max(
                conflicts, key=lambda x: self._severity_weight(x["severity"])
            )

            conflict_analysis = NWayConflictAnalysis(
                conflict_id=f"conflict_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                conflicting_models=list(
                    set([m for c in conflicts for m in c["models"]])
                ),
                conflict_type=highest_severity["type"],
                severity=highest_severity["severity"],
                impact_description=highest_severity["description"],
                resolution_strategies=self._generate_resolution_strategies(conflicts),
                confidence_score=0.85,
            )
        else:
            conflict_analysis = NWayConflictAnalysis(
                conflict_id="no_conflicts",
                conflicting_models=[],
                conflict_type="none",
                severity=ValidationSeverity.LOW,
                impact_description="No significant conflicts detected",
                resolution_strategies=[],
                confidence_score=0.95,
            )

        return conflict_analysis

    def _severity_weight(self, severity: ValidationSeverity) -> int:
        """Convert severity to numeric weight for comparison"""
        weights = {
            ValidationSeverity.LOW: 1,
            ValidationSeverity.MEDIUM: 2,
            ValidationSeverity.HIGH: 3,
            ValidationSeverity.CRITICAL: 4,
        }
        return weights.get(severity, 0)

    def _generate_resolution_strategies(self, conflicts: List[Dict]) -> List[str]:
        """Generate resolution strategies for detected conflicts"""
        strategies = []

        for conflict in conflicts:
            if conflict["type"] == "cognitive_overload":
                strategies.append("Reduce number of analytical models to 3 or fewer")
                strategies.append("Group similar models into unified approach")

            elif conflict["type"] == "approach_conflict":
                strategies.append(
                    "Apply models sequentially rather than simultaneously"
                )
                strategies.append(
                    "Use context switching between conflicting approaches"
                )

            else:
                strategies.append("Monitor interaction effects during execution")

        return list(set(strategies))  # Remove duplicates


class ComplexityProgressionValidator:
    """
    Validates that context complexity progresses appropriately
    through reasoning chain without sudden jumps or degradation
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def validate_complexity_progression(
        self, context_chain: List[ContextNode]
    ) -> Dict[str, Any]:
        """
        Validate that complexity progression is smooth and logical
        Target: Detect 90% of complexity-related validation issues
        """

        if len(context_chain) < 2:
            return {"valid": True, "issues": [], "progression_score": 1.0}

        issues = []
        complexity_scores = []

        # Calculate complexity for each node
        for node in context_chain:
            complexity = self._calculate_node_complexity(node)
            complexity_scores.append(complexity)

        # Check for sudden complexity jumps
        for i in range(1, len(complexity_scores)):
            complexity_change = complexity_scores[i] - complexity_scores[i - 1]

            # Flag sudden increases (>0.4 jump)
            if complexity_change > 0.4:
                issues.append(
                    {
                        "type": "sudden_complexity_increase",
                        "location": f"node_{i}",
                        "change": complexity_change,
                        "description": f"Complexity jumped by {complexity_change:.2f} at step {i}",
                    }
                )

            # Flag sudden decreases (>0.3 drop might indicate degradation)
            elif complexity_change < -0.3:
                issues.append(
                    {
                        "type": "complexity_degradation",
                        "location": f"node_{i}",
                        "change": complexity_change,
                        "description": f"Complexity dropped by {abs(complexity_change):.2f} at step {i}",
                    }
                )

        # Calculate overall progression score
        if len(complexity_scores) > 1:
            progression_variance = np.var(complexity_scores)
            progression_score = max(0, 1.0 - progression_variance)
        else:
            progression_score = 1.0

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "progression_score": progression_score,
            "complexity_scores": complexity_scores,
        }

    def _calculate_node_complexity(self, node: ContextNode) -> float:
        """Calculate complexity score for a context node"""

        # Base complexity from mental models count
        model_complexity = len(node.mental_models) * 0.15

        # Content complexity from length and sophistication
        content_length_factor = min(len(node.content) / 500, 1.0) * 0.3

        # Reasoning step complexity
        reasoning_complexity_map = {
            "problem_structuring": 0.3,
            "hypothesis_generation": 0.5,
            "analysis_execution": 0.8,
            "synthesis_delivery": 0.6,
        }
        reasoning_complexity = reasoning_complexity_map.get(node.reasoning_step, 0.4)

        # Dependencies complexity
        dependency_complexity = len(node.dependencies) * 0.1

        total_complexity = min(
            model_complexity
            + content_length_factor
            + reasoning_complexity
            + dependency_complexity,
            1.0,
        )

        return total_complexity


class MetisSemanticValidator:
    """
    Comprehensive semantic validation for N-WAY mental model selection
    Prevents context degradation and model conflicts

    Performance Targets:
    - Context accuracy: 95% validated correctness
    - Error reduction: 40% decrease in context-related failures
    - Model compatibility: 100% compatibility matrix validation
    - Real-time validation: <100ms per validation cycle
    """

    def __init__(self):
        self.compatibility_matrix = MentalModelCompatibilityMatrix()
        self.conflict_detector = NWayConflictDetector()
        self.complexity_validator = ComplexityProgressionValidator()
        self.logger = logging.getLogger(__name__)
        self.validation_cache = {}

    async def validate_nway_context_coherence(
        self, context_chain: List[ContextNode], selected_mental_models: List[str]
    ) -> ValidationResult:
        """
        Comprehensive validation with mental model compatibility checks
        Target: 95% context accuracy, 40% error reduction
        """

        start_time = datetime.now()
        validation_errors = []
        warnings = []

        try:
            # 1. Mental model compatibility matrix validation
            model_compatibility, compatibility_scores = (
                await self.compatibility_matrix.assess_model_compatibility(
                    selected_mental_models
                )
            )

            # Check for incompatible model pairs
            for score in compatibility_scores:
                if score.compatibility_score < 0.6:
                    validation_errors.append(
                        ValidationError(
                            error_type=ValidationErrorType.MODEL_INCOMPATIBILITY,
                            severity=ValidationSeverity.HIGH,
                            message=f"Low compatibility between {score.model_pair[0]} and {score.model_pair[1]}",
                            affected_models=list(score.model_pair),
                            context_location="model_selection",
                            remediation_suggestion="Consider alternative model combinations or sequential application",
                            confidence_score=0.9,
                        )
                    )
                elif score.compatibility_score < 0.75:
                    warnings.append(
                        f"Moderate compatibility concern: {score.model_pair[0]} + {score.model_pair[1]}"
                    )

            # 2. N-WAY interaction conflict detection
            nway_conflicts = await self.conflict_detector.detect_nway_conflicts(
                selected_mental_models, context_chain
            )

            # Process detected conflicts
            if nway_conflicts.severity in [
                ValidationSeverity.HIGH,
                ValidationSeverity.CRITICAL,
            ]:
                validation_errors.append(
                    ValidationError(
                        error_type=ValidationErrorType.NWAY_CONFLICT,
                        severity=nway_conflicts.severity,
                        message=nway_conflicts.impact_description,
                        affected_models=nway_conflicts.conflicting_models,
                        context_location="nway_interactions",
                        remediation_suggestion="; ".join(
                            nway_conflicts.resolution_strategies
                        ),
                        confidence_score=nway_conflicts.confidence_score,
                    )
                )
            elif nway_conflicts.severity == ValidationSeverity.MEDIUM:
                warnings.append(
                    f"N-WAY interaction concern: {nway_conflicts.impact_description}"
                )

            # 3. Progressive complexity validation
            complexity_validation = (
                await self.complexity_validator.validate_complexity_progression(
                    context_chain
                )
            )

            # Process complexity issues
            for issue in complexity_validation["issues"]:
                if issue["type"] == "sudden_complexity_increase":
                    validation_errors.append(
                        ValidationError(
                            error_type=ValidationErrorType.COMPLEXITY_MISMATCH,
                            severity=ValidationSeverity.MEDIUM,
                            message=issue["description"],
                            affected_models=selected_mental_models,
                            context_location=issue["location"],
                            remediation_suggestion="Add intermediate reasoning steps to smooth complexity progression",
                            confidence_score=0.85,
                        )
                    )
                elif issue["type"] == "complexity_degradation":
                    validation_errors.append(
                        ValidationError(
                            error_type=ValidationErrorType.CONTEXT_DEGRADATION,
                            severity=ValidationSeverity.HIGH,
                            message=issue["description"],
                            affected_models=selected_mental_models,
                            context_location=issue["location"],
                            remediation_suggestion="Review context preservation and prevent information loss",
                            confidence_score=0.9,
                        )
                    )

            # 4. Calculate coherence scores
            coherence_scores = []
            for node in context_chain:
                node_coherence = (
                    node.coherence_score if hasattr(node, "coherence_score") else 0.85
                )
                coherence_scores.append(node_coherence)

            overall_coherence = (
                sum(coherence_scores) / len(coherence_scores)
                if coherence_scores
                else 0.85
            )

            # 5. Compile final validation result
            processing_time = (
                datetime.now() - start_time
            ).total_seconds() * 1000  # Convert to ms

            validation_result = ValidationResult(
                valid=len(validation_errors) == 0,
                coherence_score=overall_coherence,
                mental_model_compatibility=model_compatibility,
                nway_conflict_score=1.0
                - (len(nway_conflicts.conflicting_models) * 0.2),
                complexity_progression_valid=complexity_validation["valid"],
                errors=validation_errors,
                warnings=warnings,
                processing_time_ms=processing_time,
            )

            # Log validation results
            if validation_result.valid:
                self.logger.info(
                    f"Context validation PASSED: coherence={overall_coherence:.3f}, "
                    f"compatibility={model_compatibility:.3f}, "
                    f"processing_time={processing_time:.1f}ms"
                )
            else:
                self.logger.warning(
                    f"Context validation FAILED: {len(validation_errors)} errors, "
                    f"coherence={overall_coherence:.3f}, "
                    f"processing_time={processing_time:.1f}ms"
                )

            return validation_result

        except Exception as e:
            self.logger.error(f"Context validation failed with exception: {e}")
            return ValidationResult(
                valid=False,
                coherence_score=0.0,
                mental_model_compatibility=0.0,
                nway_conflict_score=0.0,
                complexity_progression_valid=False,
                errors=[
                    ValidationError(
                        error_type=ValidationErrorType.COHERENCE_FAILURE,
                        severity=ValidationSeverity.CRITICAL,
                        message=f"Validation system failure: {str(e)}",
                        affected_models=selected_mental_models,
                        context_location="validation_system",
                        remediation_suggestion="Review validation system configuration and inputs",
                        confidence_score=0.99,
                    )
                ],
                warnings=[],
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )

    async def validate_reasoning_consistency(
        self, reasoning_steps: List[Dict[str, Any]], mental_models: List[str]
    ) -> ValidationResult:
        """
        Validate consistency across reasoning steps
        Ensures logical progression and model application coherence
        """

        validation_errors = []
        warnings = []

        # Check for reasoning gaps
        for i, step in enumerate(reasoning_steps):
            if i > 0:
                prev_step = reasoning_steps[i - 1]

                # Check for model consistency
                current_models = step.get("mental_models_used", [])
                prev_models = prev_step.get("mental_models_used", [])

                # Flag sudden model changes without explanation
                added_models = set(current_models) - set(prev_models)
                removed_models = set(prev_models) - set(current_models)

                if len(added_models) > 2:
                    warnings.append(
                        f"Sudden addition of {len(added_models)} models at step {i}"
                    )

                if len(removed_models) > 1:
                    warnings.append(
                        f"Removal of {len(removed_models)} models at step {i}"
                    )

        # Calculate consistency score
        consistency_score = max(0, 1.0 - (len(warnings) * 0.1))

        return ValidationResult(
            valid=len(validation_errors) == 0,
            coherence_score=consistency_score,
            mental_model_compatibility=1.0,
            nway_conflict_score=1.0,
            complexity_progression_valid=True,
            errors=validation_errors,
            warnings=warnings,
        )

    def generate_validation_report(self, validation_result: ValidationResult) -> str:
        """Generate human-readable validation report"""

        report = "METIS Context Validation Report\n"
        report += f"Validation Timestamp: {validation_result.validation_timestamp}\n"
        report += f"Processing Time: {validation_result.processing_time_ms:.1f}ms\n\n"

        # Overall status
        status = "VALID" if validation_result.valid else "INVALID"
        report += f"Overall Status: {status}\n\n"

        # Scores
        report += f"Coherence Score: {validation_result.coherence_score:.3f}\n"
        report += (
            f"Model Compatibility: {validation_result.mental_model_compatibility:.3f}\n"
        )
        report += f"N-WAY Conflict Score: {validation_result.nway_conflict_score:.3f}\n"
        report += f"Complexity Progression: {'Valid' if validation_result.complexity_progression_valid else 'Invalid'}\n\n"

        # Errors
        if validation_result.errors:
            report += f"Validation Errors ({len(validation_result.errors)}):\n"
            for i, error in enumerate(validation_result.errors, 1):
                report += f"{i}. {error.severity.upper()}: {error.message}\n"
                report += f"   Affected Models: {', '.join(error.affected_models)}\n"
                report += f"   Remediation: {error.remediation_suggestion}\n\n"

        # Warnings
        if validation_result.warnings:
            report += f"Warnings ({len(validation_result.warnings)}):\n"
            for i, warning in enumerate(validation_result.warnings, 1):
                report += f"{i}. {warning}\n"

        return report


# Factory function for easy instantiation
def create_semantic_validator() -> MetisSemanticValidator:
    """Create and configure semantic validator instance"""
    return MetisSemanticValidator()


# Integration helper for workflow engine
async def validate_workflow_context(
    context_chain: List[ContextNode],
    cognitive_state: CognitiveState,
    validation_level: str = "standard",
) -> ValidationResult:
    """
    Helper function to validate workflow context during execution

    Args:
        context_chain: Chain of context nodes to validate
        cognitive_state: Current cognitive processing state
        validation_level: "basic", "standard", "comprehensive"
    """

    validator = create_semantic_validator()

    # Create context nodes from cognitive state if not provided
    if not context_chain:
        context_chain = []
        for i, step in enumerate(cognitive_state.reasoning_steps):
            node = ContextNode(
                node_id=f"step_{i}",
                content=step.get("description", ""),
                mental_models=step.get("mental_models_used", []),
                reasoning_step=step.get("step", "unknown"),
                coherence_score=step.get("confidence", 0.85),
            )
            context_chain.append(node)

    # Perform validation based on level
    if validation_level == "comprehensive":
        # Full validation with reasoning consistency check
        primary_result = await validator.validate_nway_context_coherence(
            context_chain, cognitive_state.selected_mental_models
        )

        reasoning_result = await validator.validate_reasoning_consistency(
            cognitive_state.reasoning_steps, cognitive_state.selected_mental_models
        )

        # Combine results
        primary_result.coherence_score = (
            primary_result.coherence_score + reasoning_result.coherence_score
        ) / 2
        primary_result.warnings.extend(reasoning_result.warnings)

        return primary_result

    else:
        # Standard or basic validation
        return await validator.validate_nway_context_coherence(
            context_chain, cognitive_state.selected_mental_models
        )
