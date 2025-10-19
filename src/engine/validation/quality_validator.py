"""
Quality Validation Framework
Centralized quality validation system for cognitive model outputs
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import re


class QualityDimension(Enum):
    """Dimensions of quality assessment"""

    SYNTACTIC = "syntactic"  # Structure, format, completeness
    SEMANTIC = "semantic"  # Meaning, coherence, relevance
    PRAGMATIC = "pragmatic"  # Usefulness, actionability
    ETHICAL = "ethical"  # Bias, fairness, appropriateness


class ValidationSeverity(Enum):
    """Severity levels for validation issues"""

    CRITICAL = "critical"  # Must fix, blocks usage
    WARNING = "warning"  # Should fix, may impact quality
    INFO = "info"  # Nice to fix, minor issue


@dataclass
class ValidationIssue:
    """A specific quality validation issue"""

    dimension: QualityDimension
    severity: ValidationSeverity
    description: str
    suggestion: Optional[str] = None
    confidence: float = 1.0

    def __post_init__(self):
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class QualityValidationResult:
    """Result of quality validation"""

    overall_score: float  # 0.0 to 1.0
    dimension_scores: Dict[QualityDimension, float]
    issues: List[ValidationIssue]
    passed: bool
    metadata: Dict[str, Any]

    def get_critical_issues(self) -> List[ValidationIssue]:
        """Get issues with critical severity"""
        return [
            issue
            for issue in self.issues
            if issue.severity == ValidationSeverity.CRITICAL
        ]

    def get_warnings(self) -> List[ValidationIssue]:
        """Get issues with warning severity"""
        return [
            issue
            for issue in self.issues
            if issue.severity == ValidationSeverity.WARNING
        ]


class QualityValidator:
    """
    Comprehensive quality validation framework for cognitive model outputs
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.quality_thresholds = {
            QualityDimension.SYNTACTIC: 0.7,
            QualityDimension.SEMANTIC: 0.6,
            QualityDimension.PRAGMATIC: 0.6,
            QualityDimension.ETHICAL: 0.8,
        }
        self.overall_threshold = 0.7

    def validate_output(
        self,
        content: str,
        context: Dict[str, Any],
        model_type: str = "general",
        requirements: Optional[Dict[str, float]] = None,
    ) -> QualityValidationResult:
        """
        Comprehensive quality validation of cognitive model output
        """

        # Use custom requirements or defaults
        thresholds = requirements or self.quality_thresholds.copy()

        # Validate each dimension
        dimension_scores = {}
        all_issues = []

        # Syntactic validation
        syntactic_score, syntactic_issues = self._validate_syntactic_quality(
            content, context, model_type
        )
        dimension_scores[QualityDimension.SYNTACTIC] = syntactic_score
        all_issues.extend(syntactic_issues)

        # Semantic validation
        semantic_score, semantic_issues = self._validate_semantic_quality(
            content, context, model_type
        )
        dimension_scores[QualityDimension.SEMANTIC] = semantic_score
        all_issues.extend(semantic_issues)

        # Pragmatic validation
        pragmatic_score, pragmatic_issues = self._validate_pragmatic_quality(
            content, context, model_type
        )
        dimension_scores[QualityDimension.PRAGMATIC] = pragmatic_score
        all_issues.extend(pragmatic_issues)

        # Ethical validation
        ethical_score, ethical_issues = self._validate_ethical_quality(
            content, context, model_type
        )
        dimension_scores[QualityDimension.ETHICAL] = ethical_score
        all_issues.extend(ethical_issues)

        # Calculate overall score
        overall_score = self._calculate_overall_score(dimension_scores, all_issues)

        # Determine if validation passed
        passed = self._determine_validation_pass(
            dimension_scores, overall_score, all_issues, thresholds
        )

        # Compile metadata
        metadata = {
            "content_length": len(content),
            "model_type": model_type,
            "thresholds_used": thresholds,
            "critical_issue_count": len(
                [i for i in all_issues if i.severity == ValidationSeverity.CRITICAL]
            ),
            "warning_count": len(
                [i for i in all_issues if i.severity == ValidationSeverity.WARNING]
            ),
        }

        result = QualityValidationResult(
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            issues=all_issues,
            passed=passed,
            metadata=metadata,
        )

        self.logger.info(
            f"Quality validation completed: score={overall_score:.3f}, passed={passed}"
        )
        return result

    def _validate_syntactic_quality(
        self, content: str, context: Dict[str, Any], model_type: str
    ) -> Tuple[float, List[ValidationIssue]]:
        """Validate syntactic quality - structure, format, completeness"""

        issues = []
        factors = []

        # Length appropriateness
        content_length = len(content)
        if content_length < 100:
            issues.append(
                ValidationIssue(
                    dimension=QualityDimension.SYNTACTIC,
                    severity=ValidationSeverity.WARNING,
                    description="Content is very short, may lack depth",
                    suggestion="Provide more detailed analysis",
                    confidence=0.9,
                )
            )
            factors.append(0.3)
        elif content_length > 5000:
            issues.append(
                ValidationIssue(
                    dimension=QualityDimension.SYNTACTIC,
                    severity=ValidationSeverity.WARNING,
                    description="Content is very long, may be verbose",
                    suggestion="Consider summarizing key points",
                    confidence=0.7,
                )
            )
            factors.append(0.8)
        else:
            factors.append(1.0)

        # Structure indicators (for structured analysis)
        if model_type in ["systems_thinking", "critical_thinking", "mece_structuring"]:
            structure_patterns = [
                r"[A-Z][A-Z\s]+:",  # Section headers like "SYSTEM OVERVIEW:"
                r"\d+\.",  # Numbered points
                r"[•\-\*]",  # Bullet points
            ]

            structure_score = 0.0
            for pattern in structure_patterns:
                if re.search(pattern, content):
                    structure_score += 0.33

            if structure_score < 0.5:
                issues.append(
                    ValidationIssue(
                        dimension=QualityDimension.SYNTACTIC,
                        severity=ValidationSeverity.WARNING,
                        description="Content lacks clear structure",
                        suggestion="Use headers, bullet points, or numbered sections",
                        confidence=0.8,
                    )
                )

            factors.append(min(1.0, structure_score + 0.5))
        else:
            factors.append(0.9)  # Less strict for general content

        # Grammar and formatting (basic checks)
        grammar_score = self._assess_basic_grammar(content)
        factors.append(grammar_score)

        if grammar_score < 0.5:  # More lenient threshold
            issues.append(
                ValidationIssue(
                    dimension=QualityDimension.SYNTACTIC,
                    severity=ValidationSeverity.WARNING,
                    description="Potential grammar or formatting issues detected",
                    suggestion="Review text for clarity and correctness",
                    confidence=0.6,
                )
            )

        # Calculate syntactic score
        syntactic_score = sum(factors) / len(factors) if factors else 0.5

        return syntactic_score, issues

    def _validate_semantic_quality(
        self, content: str, context: Dict[str, Any], model_type: str
    ) -> Tuple[float, List[ValidationIssue]]:
        """Validate semantic quality - meaning, coherence, relevance"""

        issues = []
        factors = []

        # Relevance to context
        problem_statement = context.get("problem_statement", "")
        if problem_statement:
            relevance_score = self._calculate_relevance(content, problem_statement)
            factors.append(relevance_score)

            if relevance_score < 0.5:
                issues.append(
                    ValidationIssue(
                        dimension=QualityDimension.SEMANTIC,
                        severity=ValidationSeverity.CRITICAL,
                        description="Content appears irrelevant to the problem statement",
                        suggestion="Ensure analysis directly addresses the stated problem",
                        confidence=0.8,
                    )
                )
        else:
            factors.append(0.8)  # Neutral if no problem statement

        # Coherence and logical flow
        coherence_score = self._assess_coherence(content)
        factors.append(coherence_score)

        if coherence_score < 0.6:
            issues.append(
                ValidationIssue(
                    dimension=QualityDimension.SEMANTIC,
                    severity=ValidationSeverity.WARNING,
                    description="Content may lack logical coherence",
                    suggestion="Ensure ideas flow logically and connect clearly",
                    confidence=0.7,
                )
            )

        # Domain-specific semantic checks
        if model_type == "systems_thinking":
            systems_vocab_score = self._check_systems_vocabulary(content)
            factors.append(systems_vocab_score)

            if systems_vocab_score < 0.4:
                issues.append(
                    ValidationIssue(
                        dimension=QualityDimension.SEMANTIC,
                        severity=ValidationSeverity.WARNING,
                        description="Limited use of systems thinking vocabulary",
                        suggestion="Include terms like 'feedback loops', 'leverage points', 'emergence'",
                        confidence=0.6,
                    )
                )

        # Calculate semantic score
        semantic_score = sum(factors) / len(factors) if factors else 0.5

        return semantic_score, issues

    def _validate_pragmatic_quality(
        self, content: str, context: Dict[str, Any], model_type: str
    ) -> Tuple[float, List[ValidationIssue]]:
        """Validate pragmatic quality - usefulness, actionability"""

        issues = []
        factors = []

        # Actionability - presence of recommendations or next steps
        actionable_indicators = [
            "recommend",
            "suggest",
            "should",
            "next step",
            "action",
            "implement",
            "consider",
            "approach",
            "strategy",
        ]

        actionability_score = 0.0
        for indicator in actionable_indicators:
            if indicator.lower() in content.lower():
                actionability_score += 0.2

        actionability_score = min(1.0, actionability_score)
        factors.append(actionability_score)

        if actionability_score < 0.4:
            issues.append(
                ValidationIssue(
                    dimension=QualityDimension.PRAGMATIC,
                    severity=ValidationSeverity.WARNING,
                    description="Content lacks actionable recommendations",
                    suggestion="Include specific next steps or implementation guidance",
                    confidence=0.7,
                )
            )

        # Insight depth - beyond obvious observations
        insight_indicators = [
            "underlying",
            "root cause",
            "pattern",
            "implication",
            "insight",
            "reveals",
            "suggests",
            "indicates",
        ]

        insight_score = 0.0
        for indicator in insight_indicators:
            if indicator.lower() in content.lower():
                insight_score += 0.25

        insight_score = min(1.0, insight_score)
        factors.append(insight_score)

        if insight_score < 0.4:
            issues.append(
                ValidationIssue(
                    dimension=QualityDimension.PRAGMATIC,
                    severity=ValidationSeverity.INFO,
                    description="Content may lack deep insights",
                    suggestion="Identify patterns, root causes, or non-obvious implications",
                    confidence=0.6,
                )
            )

        # Evidence support
        evidence_indicators = [
            "because",
            "evidence",
            "data",
            "shows",
            "indicates",
            "research",
            "study",
            "analysis",
            "reveals",
        ]

        evidence_score = 0.0
        for indicator in evidence_indicators:
            if indicator.lower() in content.lower():
                evidence_score += 0.2

        evidence_score = min(1.0, evidence_score)
        factors.append(
            max(0.5, evidence_score)
        )  # Give at least 0.5 base score for evidence

        # Calculate pragmatic score
        pragmatic_score = sum(factors) / len(factors) if factors else 0.5

        return pragmatic_score, issues

    def _validate_ethical_quality(
        self, content: str, context: Dict[str, Any], model_type: str
    ) -> Tuple[float, List[ValidationIssue]]:
        """Validate ethical quality - bias, fairness, appropriateness"""

        issues = []
        factors = []

        # Bias detection (basic patterns)
        bias_patterns = [
            r"\b(all|every|always|never)\s+(men|women|people|customers)\b",
            r"\b(obviously|clearly|everyone knows)\b",
            r"\b(should|must|have to)\s+\w+\s+(because|since)",
        ]

        bias_score = 1.0
        for pattern in bias_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                bias_score -= 0.2
                issues.append(
                    ValidationIssue(
                        dimension=QualityDimension.ETHICAL,
                        severity=ValidationSeverity.WARNING,
                        description="Potential bias or overgeneralization detected",
                        suggestion="Consider more nuanced language and avoid absolutes",
                        confidence=0.6,
                    )
                )

        bias_score = max(0.0, bias_score)
        factors.append(bias_score)

        # Appropriateness - avoid inappropriate content
        inappropriate_indicators = [
            "stupid",
            "dumb",
            "idiotic",
            "incompetent",
            "worthless",
        ]

        appropriateness_score = 1.0
        for indicator in inappropriate_indicators:
            if indicator.lower() in content.lower():
                appropriateness_score -= 0.3
                issues.append(
                    ValidationIssue(
                        dimension=QualityDimension.ETHICAL,
                        severity=ValidationSeverity.CRITICAL,
                        description="Inappropriate language detected",
                        suggestion="Use professional, respectful language",
                        confidence=0.9,
                    )
                )

        appropriateness_score = max(0.0, appropriateness_score)
        factors.append(appropriateness_score)

        # Balanced perspective
        balance_indicators = [
            "however",
            "although",
            "on the other hand",
            "alternatively",
            "but",
            "nevertheless",
            "consider",
            "may also",
        ]

        balance_score = 0.0
        for indicator in balance_indicators:
            if indicator.lower() in content.lower():
                balance_score += 0.25

        balance_score = min(1.0, balance_score)
        factors.append(balance_score)

        if balance_score < 0.2:  # More lenient threshold for balanced perspective
            issues.append(
                ValidationIssue(
                    dimension=QualityDimension.ETHICAL,
                    severity=ValidationSeverity.INFO,
                    description="Content may lack balanced perspective",
                    suggestion="Consider alternative viewpoints or limitations",
                    confidence=0.5,
                )
            )

        # Calculate ethical score
        ethical_score = sum(factors) / len(factors) if factors else 0.8

        return ethical_score, issues

    def _assess_basic_grammar(self, content: str) -> float:
        """Basic grammar assessment using simple heuristics"""

        issues = 0
        total_checks = 0

        sentences = re.split(r"[.!?]+", content)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 5:
                continue

            total_checks += 1

            # Check for basic patterns that may indicate issues
            if not sentence[0].isupper():  # Sentence should start with capital
                issues += 1

            # Check for multiple spaces
            if "  " in sentence:
                issues += 0.5

            # Check for repeated words
            words = sentence.lower().split()
            for i in range(len(words) - 1):
                if words[i] == words[i + 1] and len(words[i]) > 2:
                    issues += 0.5

        if total_checks == 0:
            return 0.8

        return max(0.0, 1.0 - (issues / total_checks))

    def _calculate_relevance(self, content: str, problem_statement: str) -> float:
        """Calculate relevance of content to problem statement"""

        # Extract keywords from problem statement
        problem_words = set(re.findall(r"\b\w+\b", problem_statement.lower()))
        problem_words = {w for w in problem_words if len(w) > 3}  # Filter short words

        # Extract keywords from content
        content_words = set(re.findall(r"\b\w+\b", content.lower()))

        if not problem_words:
            return 0.8  # Neutral if no keywords to match

        # Calculate overlap
        overlap = len(problem_words & content_words)
        relevance = overlap / len(problem_words)

        return min(1.0, relevance * 2)  # Scale up to make scoring more generous

    def _assess_coherence(self, content: str) -> float:
        """Assess logical coherence of content"""

        # Simple coherence indicators
        coherence_indicators = [
            "therefore",
            "because",
            "since",
            "as a result",
            "consequently",
            "furthermore",
            "moreover",
            "in addition",
            "similarly",
            "however",
            "first",
            "second",
            "third",
            "finally",
            "next",
            "then",
        ]

        paragraph_count = len(re.split(r"\n\s*\n", content))
        indicator_count = sum(
            1
            for indicator in coherence_indicators
            if indicator.lower() in content.lower()
        )

        if paragraph_count <= 1:
            return 0.8  # Single paragraph gets neutral score

        # Expect some coherence indicators in multi-paragraph content
        coherence_ratio = indicator_count / paragraph_count
        return min(1.0, coherence_ratio + 0.5)

    def _check_systems_vocabulary(self, content: str) -> float:
        """Check for systems thinking vocabulary usage"""

        systems_terms = [
            "system",
            "feedback",
            "loop",
            "leverage",
            "emergent",
            "holistic",
            "interconnected",
            "boundary",
            "stakeholder",
            "pattern",
            "structure",
            "dynamic",
            "complexity",
            "emergence",
            "nonlinear",
            "synergy",
        ]

        found_terms = sum(
            1 for term in systems_terms if term.lower() in content.lower()
        )
        return min(1.0, found_terms / len(systems_terms) * 3)  # Scale up for generosity

    def _calculate_overall_score(
        self,
        dimension_scores: Dict[QualityDimension, float],
        issues: List[ValidationIssue],
    ) -> float:
        """Calculate overall quality score"""

        # Weighted average of dimension scores
        weights = {
            QualityDimension.SYNTACTIC: 0.2,
            QualityDimension.SEMANTIC: 0.3,
            QualityDimension.PRAGMATIC: 0.3,
            QualityDimension.ETHICAL: 0.2,
        }

        weighted_score = sum(
            dimension_scores[dim] * weights[dim] for dim in dimension_scores
        )

        # Apply penalty for critical issues
        critical_issues = [
            i for i in issues if i.severity == ValidationSeverity.CRITICAL
        ]
        critical_penalty = len(critical_issues) * 0.1

        overall_score = max(0.0, weighted_score - critical_penalty)
        return overall_score

    def _determine_validation_pass(
        self,
        dimension_scores: Dict[QualityDimension, float],
        overall_score: float,
        issues: List[ValidationIssue],
        thresholds: Dict[QualityDimension, float],
    ) -> bool:
        """Determine if validation passes based on scores and issues"""

        # Check if any critical issues exist
        critical_issues = [
            i for i in issues if i.severity == ValidationSeverity.CRITICAL
        ]
        if critical_issues:
            return False

        # Check overall score threshold
        if overall_score < self.overall_threshold:
            return False

        # Check individual dimension thresholds
        for dimension, score in dimension_scores.items():
            threshold = thresholds.get(dimension, self.quality_thresholds[dimension])
            if score < threshold:
                return False

        return True

    def set_thresholds(
        self,
        thresholds: Dict[QualityDimension, float],
        overall_threshold: Optional[float] = None,
    ):
        """Update quality thresholds"""
        self.quality_thresholds.update(thresholds)
        if overall_threshold is not None:
            self.overall_threshold = overall_threshold

    def get_validation_summary(self, result: QualityValidationResult) -> Dict[str, Any]:
        """Get a summary of validation results for logging/reporting"""

        return {
            "overall_score": result.overall_score,
            "passed": result.passed,
            "dimension_scores": {
                dim.value: score for dim, score in result.dimension_scores.items()
            },
            "issue_counts": {
                "critical": len(result.get_critical_issues()),
                "warnings": len(result.get_warnings()),
                "info": len(
                    [i for i in result.issues if i.severity == ValidationSeverity.INFO]
                ),
            },
            "content_length": result.metadata.get("content_length", 0),
            "model_type": result.metadata.get("model_type", "unknown"),
        }
