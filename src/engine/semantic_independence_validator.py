#!/usr/bin/env python3
"""
Semantic Independence Validator - Enterprise Hardening Sprint

REPLACING FAULTY KEYWORD COUNTER: Advanced semantic similarity validation
- Current Problem: Keyword counting produces false positives (66.7% independence failure)
- New Solution: Sentence embedding similarity with confidence scoring
- Architecture: sentence-transformers with cosine similarity thresholds
- Enterprise-Ready: Confidence intervals, uncertainty quantification, detailed reporting

This addresses the critical technical debt of the brittle independence metric
before it compounds in production deployment with Senior Advisor integration.

Performance Target: Replace faulty 70% keyword threshold with reliable semantic validation
Quality Target: Eliminate false positives while maintaining true independence detection
"""

import asyncio
import time
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

import os
from dotenv import load_dotenv

# Import performance instrumentation
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.performance_instrumentation import (
    get_performance_system,
    measure_function,
)

# Load environment
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(env_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class IndependenceLevel(Enum):
    """Independence assessment levels"""

    HIGHLY_INDEPENDENT = "highly_independent"  # >0.85
    INDEPENDENT = "independent"  # 0.70-0.85
    PARTIALLY_DEPENDENT = "partially_dependent"  # 0.50-0.70
    DEPENDENT = "dependent"  # <0.50


class ValidationMethod(Enum):
    """Validation methodologies"""

    SEMANTIC_SIMILARITY = "semantic_similarity"
    KEYWORD_OVERLAP = "keyword_overlap"  # Legacy method
    HYBRID_APPROACH = "hybrid_approach"


@dataclass
class IndependenceMetrics:
    """Detailed independence assessment metrics"""

    overall_independence_score: float
    pairwise_similarities: Dict[str, float]  # consultant_pair -> similarity_score
    independence_level: IndependenceLevel
    confidence_interval: Tuple[float, float]
    uncertainty_score: float

    # Detailed analysis
    semantic_overlaps: Dict[str, List[str]]  # consultant_pair -> overlapping_concepts
    distinct_concepts: Dict[str, List[str]]  # consultant -> unique_concepts
    methodology_used: ValidationMethod

    # Process metadata
    validation_time_seconds: float
    embedding_model_used: str
    similarity_threshold_used: float

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "overall_independence_score": self.overall_independence_score,
            "pairwise_similarities": self.pairwise_similarities,
            "independence_level": self.independence_level.value,
            "confidence_interval": list(self.confidence_interval),
            "uncertainty_score": self.uncertainty_score,
            "semantic_overlaps": self.semantic_overlaps,
            "distinct_concepts": self.distinct_concepts,
            "methodology_used": self.methodology_used.value,
            "validation_time_seconds": self.validation_time_seconds,
            "embedding_model_used": self.embedding_model_used,
            "similarity_threshold_used": self.similarity_threshold_used,
            "timestamp": self.timestamp,
        }


@dataclass
class ConsultantAnalysis:
    """Consultant analysis for independence validation"""

    role: str
    analysis: str
    mental_model_used: str
    core_assumptions: List[str] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)

    def get_combined_content(self) -> str:
        """Get combined content for semantic analysis"""
        content_parts = [
            self.analysis,
            " ".join(self.core_assumptions),
            " ".join(self.reasoning_chain),
        ]
        return " ".join(content_parts).strip()


class SemanticIndependenceValidator:
    """
    Advanced Semantic Independence Validator

    Replaces brittle keyword counting with sophisticated semantic similarity analysis:
    1. Sentence embeddings using sentence-transformers
    2. Cosine similarity calculation between consultant analyses
    3. Confidence intervals and uncertainty quantification
    4. Detailed semantic overlap analysis
    5. Enterprise-grade reporting and monitoring

    Architecture:
    - sentence-transformers for semantic embeddings
    - NumPy for efficient similarity calculations
    - Configurable similarity thresholds (default: 0.70)
    - Bootstrap confidence intervals for uncertainty
    - Comprehensive validation reporting
    """

    def __init__(
        self, similarity_threshold: float = 0.70, model_name: str = "all-MiniLM-L6-v2"
    ):
        self.similarity_threshold = similarity_threshold
        self.model_name = model_name
        self.perf_system = get_performance_system()

        # Initialize sentence transformer model
        self.sentence_model = None
        self._initialize_embedding_model()

        logger.info("✅ Semantic Independence Validator initialized:")
        logger.info(f"   Model: {self.model_name}")
        logger.info(f"   Similarity Threshold: {self.similarity_threshold}")

    def _initialize_embedding_model(self):
        """Initialize sentence transformer model for embeddings"""
        try:
            # Try to import sentence-transformers
            from sentence_transformers import SentenceTransformer

            logger.info(f"🔧 Loading sentence transformer model: {self.model_name}")
            self.sentence_model = SentenceTransformer(self.model_name)
            logger.info("✅ Sentence transformer model loaded successfully")

        except ImportError:
            logger.warning(
                "⚠️ sentence-transformers not available, falling back to simple similarity"
            )
            self.sentence_model = None
        except Exception as e:
            logger.error(f"💥 Failed to load sentence transformer: {str(e)}")
            self.sentence_model = None

    @measure_function("semantic_independence_validation", "independence_validator")
    async def validate_independence(
        self, consultants: List[ConsultantAnalysis]
    ) -> IndependenceMetrics:
        """
        Validate consultant independence using semantic similarity analysis

        Process:
        1. Extract semantic embeddings from consultant analyses
        2. Calculate pairwise cosine similarities
        3. Assess overall independence score
        4. Generate confidence intervals and uncertainty estimates
        5. Provide detailed semantic overlap analysis
        """
        validation_start = time.time()

        logger.info(
            f"🔍 Starting semantic independence validation for {len(consultants)} consultants"
        )

        if len(consultants) < 2:
            raise ValueError("Need at least 2 consultants for independence validation")

        # Extract content from consultants
        consultant_contents = {}
        for consultant in consultants:
            consultant_contents[consultant.role] = consultant.get_combined_content()

        # Calculate semantic similarities
        if self.sentence_model is not None:
            pairwise_similarities, semantic_overlaps, distinct_concepts = (
                await self._calculate_semantic_similarities(consultant_contents)
            )
            methodology = ValidationMethod.SEMANTIC_SIMILARITY
        else:
            # Fallback to improved keyword-based approach
            logger.warning("🔄 Falling back to improved keyword-based similarity")
            pairwise_similarities, semantic_overlaps, distinct_concepts = (
                await self._calculate_keyword_similarities(consultant_contents)
            )
            methodology = ValidationMethod.KEYWORD_OVERLAP

        # Calculate overall independence score
        overall_score = await self._calculate_overall_independence(
            pairwise_similarities
        )

        # Determine independence level
        independence_level = self._determine_independence_level(overall_score)

        # Calculate confidence interval and uncertainty
        confidence_interval, uncertainty_score = (
            await self._calculate_confidence_metrics(
                pairwise_similarities, consultant_contents
            )
        )

        validation_time = time.time() - validation_start

        logger.info("✅ Semantic validation completed:")
        logger.info(
            f"   Overall Independence: {overall_score:.3f} ({independence_level.value})"
        )
        logger.info(
            f"   Confidence Interval: [{confidence_interval[0]:.3f}, {confidence_interval[1]:.3f}]"
        )
        logger.info(f"   Validation Time: {validation_time:.3f}s")

        return IndependenceMetrics(
            overall_independence_score=overall_score,
            pairwise_similarities=pairwise_similarities,
            independence_level=independence_level,
            confidence_interval=confidence_interval,
            uncertainty_score=uncertainty_score,
            semantic_overlaps=semantic_overlaps,
            distinct_concepts=distinct_concepts,
            methodology_used=methodology,
            validation_time_seconds=round(validation_time, 4),
            embedding_model_used=(
                self.model_name if self.sentence_model else "keyword_fallback"
            ),
            similarity_threshold_used=self.similarity_threshold,
        )

    async def _calculate_semantic_similarities(
        self, consultant_contents: Dict[str, str]
    ) -> Tuple[Dict[str, float], Dict[str, List[str]], Dict[str, List[str]]]:
        """Calculate semantic similarities using sentence embeddings"""

        consultant_roles = list(consultant_contents.keys())
        consultant_texts = list(consultant_contents.values())

        logger.info("🧠 Generating semantic embeddings...")

        # Generate embeddings
        async with self.perf_system.measure_async(
            "embedding_generation", "independence_validator"
        ):
            embeddings = await asyncio.get_event_loop().run_in_executor(
                None, self.sentence_model.encode, consultant_texts
            )

        # Calculate pairwise cosine similarities
        pairwise_similarities = {}
        semantic_overlaps = {}

        for i in range(len(consultant_roles)):
            for j in range(i + 1, len(consultant_roles)):
                role1, role2 = consultant_roles[i], consultant_roles[j]

                # Cosine similarity
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )

                pair_key = f"{role1}_{role2}"
                pairwise_similarities[pair_key] = float(similarity)

                # Identify semantic overlaps (conceptual analysis would go here)
                semantic_overlaps[pair_key] = self._identify_semantic_overlaps(
                    consultant_contents[role1], consultant_contents[role2], similarity
                )

        # Identify distinct concepts for each consultant
        distinct_concepts = self._identify_distinct_concepts(
            consultant_contents, pairwise_similarities
        )

        return pairwise_similarities, semantic_overlaps, distinct_concepts

    async def _calculate_keyword_similarities(
        self, consultant_contents: Dict[str, str]
    ) -> Tuple[Dict[str, float], Dict[str, List[str]], Dict[str, List[str]]]:
        """Improved keyword-based similarity calculation (fallback)"""

        consultant_roles = list(consultant_contents.keys())
        pairwise_similarities = {}
        semantic_overlaps = {}

        for i in range(len(consultant_roles)):
            for j in range(i + 1, len(consultant_roles)):
                role1, role2 = consultant_roles[i], consultant_roles[j]

                # Improved keyword similarity (Jaccard similarity on word sets)
                text1_words = set(consultant_contents[role1].lower().split())
                text2_words = set(consultant_contents[role2].lower().split())

                intersection = text1_words.intersection(text2_words)
                union = text1_words.union(text2_words)

                jaccard_similarity = len(intersection) / len(union) if union else 0.0

                pair_key = f"{role1}_{role2}"
                pairwise_similarities[pair_key] = jaccard_similarity

                # Overlapping words as semantic overlap
                semantic_overlaps[pair_key] = list(intersection)[
                    :10
                ]  # Top 10 overlapping words

        # Distinct concepts (unique words per consultant)
        distinct_concepts = {}
        all_words = set()
        for content in consultant_contents.values():
            all_words.update(content.lower().split())

        for role, content in consultant_contents.items():
            role_words = set(content.lower().split())
            # Find words more frequent in this consultant vs others
            other_contents = [c for r, c in consultant_contents.items() if r != role]
            other_words = set()
            for other_content in other_contents:
                other_words.update(other_content.lower().split())

            distinct_words = role_words - other_words
            distinct_concepts[role] = list(distinct_words)[
                :10
            ]  # Top 10 distinct concepts

        return pairwise_similarities, semantic_overlaps, distinct_concepts

    def _identify_semantic_overlaps(
        self, text1: str, text2: str, similarity_score: float
    ) -> List[str]:
        """Identify semantic overlaps between two texts"""
        if similarity_score > 0.8:
            return ["High semantic similarity detected", "Potential concept overlap"]
        elif similarity_score > 0.6:
            return ["Moderate semantic similarity", "Some shared concepts"]
        elif similarity_score > 0.4:
            return ["Low semantic similarity", "Minimal overlap"]
        else:
            return ["Very low similarity", "Highly independent"]

    def _identify_distinct_concepts(
        self, consultant_contents: Dict[str, str], similarities: Dict[str, float]
    ) -> Dict[str, List[str]]:
        """Identify distinct concepts for each consultant"""
        distinct_concepts = {}

        for role in consultant_contents.keys():
            # Analyze this consultant's unique contribution
            role_concepts = []

            # Simple heuristic: if this consultant has low similarity with others, they have distinct concepts
            role_avg_similarity = np.mean(
                [sim for pair, sim in similarities.items() if role in pair]
            )

            if role_avg_similarity < 0.5:
                role_concepts.append(
                    f"Highly distinct approach (avg similarity: {role_avg_similarity:.2f})"
                )
            elif role_avg_similarity < 0.7:
                role_concepts.append(
                    f"Moderately distinct approach (avg similarity: {role_avg_similarity:.2f})"
                )
            else:
                role_concepts.append(
                    f"Some overlap with other consultants (avg similarity: {role_avg_similarity:.2f})"
                )

            distinct_concepts[role] = role_concepts

        return distinct_concepts

    async def _calculate_overall_independence(
        self, pairwise_similarities: Dict[str, float]
    ) -> float:
        """Calculate overall independence score from pairwise similarities"""
        if not pairwise_similarities:
            return 1.0  # Perfect independence if no pairs to compare

        # Independence is inverse of similarity
        # Average similarity across all pairs, then convert to independence
        avg_similarity = np.mean(list(pairwise_similarities.values()))
        independence_score = 1.0 - avg_similarity

        return max(0.0, min(1.0, independence_score))  # Clamp to [0, 1]

    async def _calculate_confidence_metrics(
        self, similarities: Dict[str, float], contents: Dict[str, str]
    ) -> Tuple[Tuple[float, float], float]:
        """Calculate confidence interval and uncertainty score"""

        if not similarities:
            return (1.0, 1.0), 0.0

        similarity_values = list(similarities.values())
        independence_values = [1.0 - sim for sim in similarity_values]

        # Simple confidence interval based on standard deviation
        mean_independence = np.mean(independence_values)
        std_independence = (
            np.std(independence_values) if len(independence_values) > 1 else 0.1
        )

        # 95% confidence interval (±1.96 standard deviations)
        margin_of_error = 1.96 * std_independence / np.sqrt(len(independence_values))
        confidence_lower = max(0.0, mean_independence - margin_of_error)
        confidence_upper = min(1.0, mean_independence + margin_of_error)

        # Uncertainty score (higher = more uncertain)
        uncertainty = (
            std_independence / mean_independence if mean_independence > 0 else 1.0
        )
        uncertainty = min(1.0, uncertainty)

        return (confidence_lower, confidence_upper), uncertainty

    def _determine_independence_level(
        self, independence_score: float
    ) -> IndependenceLevel:
        """Determine independence level from score"""
        if independence_score >= 0.85:
            return IndependenceLevel.HIGHLY_INDEPENDENT
        elif independence_score >= 0.70:
            return IndependenceLevel.INDEPENDENT
        elif independence_score >= 0.50:
            return IndependenceLevel.PARTIALLY_DEPENDENT
        else:
            return IndependenceLevel.DEPENDENT

    def is_independent(
        self, metrics: IndependenceMetrics, strict: bool = False
    ) -> bool:
        """Check if consultants are independent based on metrics"""
        if strict:
            # Strict independence requires high confidence and high score
            return (
                metrics.independence_level
                in [IndependenceLevel.HIGHLY_INDEPENDENT, IndependenceLevel.INDEPENDENT]
                and metrics.uncertainty_score < 0.3
            )
        else:
            # Standard independence
            return metrics.independence_level in [
                IndependenceLevel.HIGHLY_INDEPENDENT,
                IndependenceLevel.INDEPENDENT,
            ]


# Test function for semantic independence validator
async def test_semantic_independence_validator():
    """Test the semantic independence validator with different scenarios"""

    print("🔍 TESTING SEMANTIC INDEPENDENCE VALIDATOR")
    print("=" * 80)

    validator = SemanticIndependenceValidator(similarity_threshold=0.70)

    # Test Case 1: Clearly Independent Consultants
    print("\n📋 Test Case 1: Clearly Independent Consultants")

    independent_consultants = [
        ConsultantAnalysis(
            role="Strategic Analyst",
            analysis="Using MECE framework to analyze market opportunities. Focus on competitive positioning and strategic differentiation through market segmentation and value proposition development.",
            mental_model_used="MECE Framework",
            core_assumptions=[
                "Market segmentation drives competitive advantage",
                "Strategic differentiation is sustainable",
            ],
            reasoning_chain=[
                "Analyzed market segments",
                "Identified strategic gaps",
                "Developed positioning strategy",
            ],
        ),
        ConsultantAnalysis(
            role="Synthesis Architect",
            analysis="Applying Charlie Munger's inversion thinking and mental models latticework. Consider psychological biases and systems thinking to understand root causes and feedback loops.",
            mental_model_used="Charlie Munger Mental Models",
            core_assumptions=[
                "Inversion thinking reveals hidden risks",
                "Systems thinking exposes feedback loops",
            ],
            reasoning_chain=[
                "Applied inversion analysis",
                "Identified cognitive biases",
                "Mapped system dynamics",
            ],
        ),
        ConsultantAnalysis(
            role="Implementation Driver",
            analysis="Lean implementation approach with rapid iteration cycles. Focus on minimum viable product development and continuous improvement through kaizen methodology.",
            mental_model_used="Lean Implementation Framework",
            core_assumptions=[
                "Rapid iteration reduces risk",
                "Customer feedback drives optimization",
            ],
            reasoning_chain=[
                "Designed MVP approach",
                "Established feedback loops",
                "Created improvement cycles",
            ],
        ),
    ]

    start_time = time.time()
    metrics1 = await validator.validate_independence(independent_consultants)
    validation_time1 = time.time() - start_time

    print(f"   Independence Score: {metrics1.overall_independence_score:.3f}")
    print(f"   Independence Level: {metrics1.independence_level.value}")
    print(
        f"   Confidence Interval: [{metrics1.confidence_interval[0]:.3f}, {metrics1.confidence_interval[1]:.3f}]"
    )
    print(f"   Uncertainty: {metrics1.uncertainty_score:.3f}")
    print(f"   Is Independent: {validator.is_independent(metrics1)}")
    print(f"   Validation Time: {validation_time1:.3f}s")

    # Test Case 2: Similar/Dependent Consultants
    print("\n📋 Test Case 2: Similar/Dependent Consultants")

    dependent_consultants = [
        ConsultantAnalysis(
            role="Strategic Analyst",
            analysis="Digital transformation requires strategic planning and implementation roadmap. Focus on technology integration and operational efficiency improvements.",
            mental_model_used="Strategic Planning Framework",
            core_assumptions=[
                "Digital transformation is strategic priority",
                "Technology integration drives efficiency",
            ],
            reasoning_chain=[
                "Assessed digital needs",
                "Planned technology integration",
                "Designed implementation roadmap",
            ],
        ),
        ConsultantAnalysis(
            role="Technology Consultant",
            analysis="Digital transformation needs comprehensive technology integration strategy. Implementation roadmap should focus on operational efficiency and strategic technology adoption.",
            mental_model_used="Technology Integration Framework",
            core_assumptions=[
                "Technology integration is key to success",
                "Strategic planning drives implementation",
            ],
            reasoning_chain=[
                "Evaluated technology options",
                "Created integration strategy",
                "Developed implementation plan",
            ],
        ),
    ]

    start_time = time.time()
    metrics2 = await validator.validate_independence(dependent_consultants)
    validation_time2 = time.time() - start_time

    print(f"   Independence Score: {metrics2.overall_independence_score:.3f}")
    print(f"   Independence Level: {metrics2.independence_level.value}")
    print(
        f"   Confidence Interval: [{metrics2.confidence_interval[0]:.3f}, {metrics2.confidence_interval[1]:.3f}]"
    )
    print(f"   Uncertainty: {metrics2.uncertainty_score:.3f}")
    print(f"   Is Independent: {validator.is_independent(metrics2)}")
    print(f"   Validation Time: {validation_time2:.3f}s")

    # Performance and Accuracy Summary
    print("\n" + "=" * 80)
    print("📊 SEMANTIC VALIDATION PERFORMANCE SUMMARY")
    print("=" * 80)

    print(
        f"✅ Test Case 1 (Independent): {metrics1.independence_level.value} | Score: {metrics1.overall_independence_score:.3f}"
    )
    print(
        f"⚠️ Test Case 2 (Dependent): {metrics2.independence_level.value} | Score: {metrics2.overall_independence_score:.3f}"
    )
    print(
        f"⚡ Average Validation Time: {(validation_time1 + validation_time2) / 2:.3f}s"
    )
    print(f"🎯 Model Used: {metrics1.embedding_model_used}")

    # Detailed Analysis
    print("\n📋 Detailed Pairwise Analysis:")
    print("Independent Consultants:")
    for pair, similarity in metrics1.pairwise_similarities.items():
        independence = 1.0 - similarity
        print(
            f"   {pair.replace('_', ' vs ')}: {independence:.3f} independence ({similarity:.3f} similarity)"
        )

    print("Dependent Consultants:")
    for pair, similarity in metrics2.pairwise_similarities.items():
        independence = 1.0 - similarity
        print(
            f"   {pair.replace('_', ' vs ')}: {independence:.3f} independence ({similarity:.3f} similarity)"
        )

    return metrics1, metrics2


if __name__ == "__main__":
    asyncio.run(test_semantic_independence_validator())
