#!/usr/bin/env python3
"""
Vector Similarity Pattern Detection System
Phase 1: Foundation Systems - Systematic Intelligence Amplification

Implements vector similarity detection to identify patterns in problem spaces and mental model
applications, enabling recognition of similar problems and application of proven model combinations.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import statistics
import hashlib
from pathlib import Path


class SimilarityMetric(str, Enum):
    """Vector similarity metrics for pattern detection"""

    COSINE_SIMILARITY = "cosine"  # Cosine similarity (most common)
    EUCLIDEAN_DISTANCE = "euclidean"  # Euclidean distance
    MANHATTAN_DISTANCE = "manhattan"  # Manhattan (L1) distance
    DOT_PRODUCT = "dot_product"  # Simple dot product
    JACCARD_SIMILARITY = "jaccard"  # Jaccard similarity for binary features


class PatternType(str, Enum):
    """Types of patterns that can be detected"""

    PROBLEM_SIMILARITY = "problem_similarity"  # Similar problem statements
    MODEL_COMBINATION = "model_combination"  # Successful model combinations
    CONTEXT_PATTERN = "context_pattern"  # Similar business contexts
    OUTCOME_PATTERN = "outcome_pattern"  # Similar outcome patterns
    PERFORMANCE_PATTERN = "performance_pattern"  # Similar performance characteristics


@dataclass
class PatternSimilarity:
    """Represents a pattern match with similarity score"""

    pattern_id: str
    pattern_data: Dict[str, Any]
    similarity_score: float

    def __post_init__(self):
        """Validate similarity score range"""
        if not 0.0 <= self.similarity_score <= 1.0:
            raise ValueError(
                f"Similarity score must be between 0.0 and 1.0, got {self.similarity_score}"
            )


@dataclass
class PatternDetectionResult:
    """Result of pattern similarity detection"""

    patterns_found: List[PatternSimilarity]
    total_patterns_searched: int
    detection_time_ms: float
    similarity_threshold_used: float
    similarity_metric_used: SimilarityMetric
    detection_successful: bool
    average_similarity: float

    def __getitem__(self, index):
        """Allow indexing for backward compatibility"""
        fields = [
            self.patterns_found,
            self.total_patterns_searched,
            self.detection_time_ms,
            self.similarity_threshold_used,
            self.similarity_metric_used,
            self.detection_successful,
            self.average_similarity,
        ]
        return fields[index]


@dataclass
class VectorizedPattern:
    """Vectorized representation of a pattern"""

    pattern_id: str
    pattern_type: PatternType
    vector: np.ndarray
    metadata: Dict[str, Any]

    # Original data
    raw_data: Dict[str, Any]

    # Pattern characteristics
    dimensionality: int = 0
    sparsity: float = 0.0  # Fraction of zero elements
    norm: float = 0.0  # Vector norm

    # Tracking information
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    similarity_matches: int = 0  # How often this pattern was matched

    def __post_init__(self):
        if isinstance(self.vector, np.ndarray):
            self.dimensionality = len(self.vector)
            self.sparsity = np.count_nonzero(self.vector == 0) / len(self.vector)
            self.norm = np.linalg.norm(self.vector)


class SimilarityMatch(NamedTuple):
    """Represents a similarity match between patterns"""

    pattern_id: str
    similarity_score: float
    similarity_metric: SimilarityMetric
    pattern_type: PatternType
    metadata: Dict[str, Any]


@dataclass
class PatternDetectionResult:
    """Result of pattern detection analysis"""

    query_vector: np.ndarray
    query_metadata: Dict[str, Any]

    # Top matches
    top_matches: List[SimilarityMatch] = field(default_factory=list)

    # Analysis metrics
    detection_time_ms: float = 0.0
    patterns_searched: int = 0
    similarity_threshold_used: float = 0.0

    # Insights
    pattern_insights: Dict[str, Any] = field(default_factory=dict)
    recommended_actions: List[str] = field(default_factory=list)


class VectorSimilarityEngine:
    """
    Engine for vector similarity pattern detection
    Identifies patterns in problem spaces and model applications
    """

    def __init__(
        self,
        default_similarity_metric: SimilarityMetric = SimilarityMetric.COSINE_SIMILARITY,
        similarity_threshold: float = 0.7,
        max_patterns_stored: int = 10000,
        vector_dimension: int = 384,  # Common embedding dimension
        pattern_storage_path: Optional[str] = None,
    ):

        self.logger = logging.getLogger(__name__)
        self.default_similarity_metric = default_similarity_metric
        self.similarity_threshold = similarity_threshold
        self.max_patterns_stored = max_patterns_stored
        self.vector_dimension = vector_dimension

        # Pattern storage
        self.patterns: Dict[str, VectorizedPattern] = {}
        self.pattern_index: Dict[PatternType, List[str]] = defaultdict(list)

        # Performance tracking
        self.detection_stats = {
            "total_detections": 0,
            "successful_matches": 0,
            "average_detection_time_ms": 0.0,
            "pattern_types_detected": defaultdict(int),
        }

        # Storage management
        self.storage_path = (
            Path(pattern_storage_path)
            if pattern_storage_path
            else Path("data/vector_patterns")
        )
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Vector processing cache
        self.vector_cache: Dict[str, np.ndarray] = {}

        self.logger.info(
            "✅ VectorSimilarityEngine initialized - Pattern detection active"
        )

    def _create_text_vector(self, text: str) -> np.ndarray:
        """Create vector representation of text using simple TF-IDF-like approach"""

        # For now, use a simple hash-based vectorization
        # In production, this would use proper embeddings (OpenAI, Sentence-BERT, etc.)

        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self.vector_cache:
            return self.vector_cache[cache_key]

        # Simple word-based vectorization
        words = text.lower().split()

        # Create a simple vocabulary hash
        vector = np.zeros(self.vector_dimension)

        for word in words:
            # Hash word to vector position
            word_hash = hash(word) % self.vector_dimension
            vector[word_hash] += 1

        # Normalize vector
        if np.linalg.norm(vector) > 0:
            vector = vector / np.linalg.norm(vector)

        # Cache result
        self.vector_cache[cache_key] = vector

        return vector

    def _create_context_vector(self, context: Dict[str, Any]) -> np.ndarray:
        """Create vector representation of business context"""

        vector = np.zeros(self.vector_dimension)

        # Extract key context features
        features = []

        # Industry features
        if "industry" in context:
            features.append(f"industry_{context['industry']}")

        # Company size features
        if "company_size" in context:
            features.append(f"size_{context['company_size']}")

        # Urgency features
        if "urgency" in context:
            features.append(f"urgency_{context['urgency']}")

        # Complexity features
        if "complexity_level" in context:
            features.append(f"complexity_{context['complexity_level']}")

        # Budget features
        if "budget_range" in context:
            features.append(f"budget_{context['budget_range']}")

        # Stakeholder features
        if "stakeholders" in context and isinstance(context["stakeholders"], list):
            for stakeholder in context["stakeholders"]:
                features.append(f"stakeholder_{stakeholder}")

        # Hash features to vector positions
        for feature in features:
            feature_hash = hash(feature) % self.vector_dimension
            vector[feature_hash] += 1

        # Normalize
        if np.linalg.norm(vector) > 0:
            vector = vector / np.linalg.norm(vector)

        return vector

    def _create_model_combination_vector(self, model_ids: List[str]) -> np.ndarray:
        """Create vector representation of model combination"""

        vector = np.zeros(self.vector_dimension)

        # Hash each model to vector positions
        for model_id in model_ids:
            model_hash = hash(model_id) % self.vector_dimension
            vector[model_hash] += 1

        # Add combination patterns
        if len(model_ids) > 1:
            for i, model1 in enumerate(model_ids):
                for model2 in model_ids[i + 1 :]:
                    combo_hash = hash(f"{model1}+{model2}") % self.vector_dimension
                    vector[combo_hash] += 0.5  # Weight combinations lower

        # Normalize
        if np.linalg.norm(vector) > 0:
            vector = vector / np.linalg.norm(vector)

        return vector

    def _create_performance_vector(
        self, performance_data: Dict[str, Any]
    ) -> np.ndarray:
        """Create vector representation of performance characteristics"""

        vector = np.zeros(self.vector_dimension)

        # Map performance metrics to vector dimensions
        performance_features = [
            ("confidence_score", performance_data.get("confidence_score", 0.5)),
            ("effectiveness_score", performance_data.get("effectiveness_score", 0.5)),
            (
                "stakeholder_satisfaction",
                performance_data.get("stakeholder_satisfaction", 0.5),
            ),
            (
                "implementation_efficiency",
                performance_data.get("implementation_efficiency", 0.5),
            ),
            ("value_realization", performance_data.get("value_realization", 0.5)),
        ]

        # Discretize continuous values and map to vector
        for feature_name, value in performance_features:
            # Discretize value into bins (0-0.2, 0.2-0.4, etc.)
            bin_idx = min(4, int(value * 5))
            feature_hash = hash(f"{feature_name}_bin_{bin_idx}") % self.vector_dimension
            vector[feature_hash] += value

        # Normalize
        if np.linalg.norm(vector) > 0:
            vector = vector / np.linalg.norm(vector)

        return vector

    def calculate_similarity(
        self, vector1: np.ndarray, vector2: np.ndarray, metric: SimilarityMetric = None
    ) -> float:
        """Calculate similarity between two vectors"""

        if metric is None:
            metric = self.default_similarity_metric

        # Ensure vectors have same dimension
        if len(vector1) != len(vector2):
            min_dim = min(len(vector1), len(vector2))
            vector1 = vector1[:min_dim]
            vector2 = vector2[:min_dim]

        try:
            if metric == SimilarityMetric.COSINE_SIMILARITY:
                dot_product = np.dot(vector1, vector2)
                norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
                if norm_product == 0:
                    return 0.0
                return dot_product / norm_product

            elif metric == SimilarityMetric.EUCLIDEAN_DISTANCE:
                distance = np.linalg.norm(vector1 - vector2)
                # Convert distance to similarity (0-1 range)
                return 1.0 / (1.0 + distance)

            elif metric == SimilarityMetric.MANHATTAN_DISTANCE:
                distance = np.sum(np.abs(vector1 - vector2))
                return 1.0 / (1.0 + distance)

            elif metric == SimilarityMetric.DOT_PRODUCT:
                return np.dot(vector1, vector2)

            elif metric == SimilarityMetric.JACCARD_SIMILARITY:
                # Convert to binary vectors
                bin1 = (vector1 > 0).astype(int)
                bin2 = (vector2 > 0).astype(int)
                intersection = np.sum(bin1 & bin2)
                union = np.sum(bin1 | bin2)
                if union == 0:
                    return 0.0
                return intersection / union

            else:
                return 0.0

        except Exception as e:
            self.logger.warning(f"⚠️ Similarity calculation failed: {e}")
            return 0.0

    async def add_pattern(
        self,
        pattern_type: PatternType,
        raw_data: Dict[str, Any],
        metadata: Dict[str, Any] = None,
    ) -> str:
        """Add a new pattern to the detection system"""

        try:
            # Create vector based on pattern type
            if pattern_type == PatternType.PROBLEM_SIMILARITY:
                vector = self._create_text_vector(raw_data.get("problem_statement", ""))
            elif pattern_type == PatternType.CONTEXT_PATTERN:
                vector = self._create_context_vector(
                    raw_data.get("business_context", {})
                )
            elif pattern_type == PatternType.MODEL_COMBINATION:
                vector = self._create_model_combination_vector(
                    raw_data.get("model_ids", [])
                )
            elif pattern_type == PatternType.PERFORMANCE_PATTERN:
                vector = self._create_performance_vector(
                    raw_data.get("performance_data", {})
                )
            elif pattern_type == PatternType.OUTCOME_PATTERN:
                # Combine multiple vectors for outcome patterns
                problem_vec = self._create_text_vector(
                    raw_data.get("problem_statement", "")
                )
                context_vec = self._create_context_vector(
                    raw_data.get("business_context", {})
                )
                vector = (problem_vec + context_vec) / 2
            else:
                vector = np.zeros(self.vector_dimension)

            # Generate pattern ID
            pattern_id = f"{pattern_type.value}_{datetime.utcnow().timestamp()}_{hash(str(raw_data))}"[
                :64
            ]

            # Create vectorized pattern
            pattern = VectorizedPattern(
                pattern_id=pattern_id,
                pattern_type=pattern_type,
                vector=vector,
                metadata=metadata or {},
                raw_data=raw_data,
            )

            # Store pattern
            self.patterns[pattern_id] = pattern
            self.pattern_index[pattern_type].append(pattern_id)

            # Enforce storage limits
            if len(self.patterns) > self.max_patterns_stored:
                await self._cleanup_old_patterns()

            self.logger.debug(
                f"📊 Added pattern: {pattern_type.value} | ID: {pattern_id[:8]}..."
            )

            return pattern_id

        except Exception as e:
            self.logger.error(f"❌ Failed to add pattern: {e}")
            return None

    async def detect_similar_patterns(
        self,
        pattern_type: PatternType,
        query_data: Dict[str, Any],
        top_k: int = 10,
        similarity_threshold: float = None,
    ) -> PatternDetectionResult:
        """Detect similar patterns for given query data"""

        start_time = datetime.utcnow()

        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold

        # Create query vector
        if pattern_type == PatternType.PROBLEM_SIMILARITY:
            query_vector = self._create_text_vector(
                query_data.get("problem_statement", "")
            )
        elif pattern_type == PatternType.CONTEXT_PATTERN:
            query_vector = self._create_context_vector(
                query_data.get("business_context", {})
            )
        elif pattern_type == PatternType.MODEL_COMBINATION:
            query_vector = self._create_model_combination_vector(
                query_data.get("model_ids", [])
            )
        elif pattern_type == PatternType.PERFORMANCE_PATTERN:
            query_vector = self._create_performance_vector(
                query_data.get("performance_data", {})
            )
        elif pattern_type == PatternType.OUTCOME_PATTERN:
            problem_vec = self._create_text_vector(
                query_data.get("problem_statement", "")
            )
            context_vec = self._create_context_vector(
                query_data.get("business_context", {})
            )
            query_vector = (problem_vec + context_vec) / 2
        else:
            query_vector = np.zeros(self.vector_dimension)

        # Find similar patterns
        matches = []
        patterns_searched = 0

        for pattern_id in self.pattern_index[pattern_type]:
            if pattern_id not in self.patterns:
                continue

            pattern = self.patterns[pattern_id]

            # Calculate similarity
            similarity = self.calculate_similarity(query_vector, pattern.vector)
            patterns_searched += 1

            if similarity >= similarity_threshold:
                match = SimilarityMatch(
                    pattern_id=pattern_id,
                    similarity_score=similarity,
                    similarity_metric=self.default_similarity_metric,
                    pattern_type=pattern_type,
                    metadata=pattern.metadata,
                )
                matches.append(match)

                # Update pattern access statistics
                pattern.last_accessed = datetime.utcnow()
                pattern.access_count += 1
                pattern.similarity_matches += 1

        # Sort by similarity and take top k
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        top_matches = matches[:top_k]

        # Calculate detection time
        detection_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Generate insights and recommendations
        insights, recommendations = self._analyze_detection_results(
            pattern_type, top_matches, query_data
        )

        # Update statistics
        self.detection_stats["total_detections"] += 1
        if top_matches:
            self.detection_stats["successful_matches"] += 1
        self.detection_stats["pattern_types_detected"][pattern_type.value] += 1

        # Update average detection time
        old_avg = self.detection_stats["average_detection_time_ms"]
        new_avg = (
            old_avg * (self.detection_stats["total_detections"] - 1) + detection_time
        ) / self.detection_stats["total_detections"]
        self.detection_stats["average_detection_time_ms"] = new_avg

        result = PatternDetectionResult(
            query_vector=query_vector,
            query_metadata=query_data,
            top_matches=top_matches,
            detection_time_ms=detection_time,
            patterns_searched=patterns_searched,
            similarity_threshold_used=similarity_threshold,
            pattern_insights=insights,
            recommended_actions=recommendations,
        )

        if top_matches:
            self.logger.info(
                f"🔍 Pattern detection: {pattern_type.value} | "
                f"Found {len(top_matches)} matches | "
                f"Best similarity: {top_matches[0].similarity_score:.3f} | "
                f"Time: {detection_time:.1f}ms"
            )
        else:
            self.logger.debug(
                f"🔍 Pattern detection: {pattern_type.value} | No matches found above threshold {similarity_threshold:.3f}"
            )

        return result

    def _analyze_detection_results(
        self,
        pattern_type: PatternType,
        matches: List[SimilarityMatch],
        query_data: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Analyze detection results and generate insights"""

        insights = {}
        recommendations = []

        if not matches:
            insights["match_quality"] = "no_matches"
            recommendations.append(
                "This appears to be a novel problem - consider creating new pattern"
            )
            return insights, recommendations

        # Analyze match quality
        best_similarity = matches[0].similarity_score
        avg_similarity = statistics.mean([m.similarity_score for m in matches])

        if best_similarity >= 0.9:
            insights["match_quality"] = "excellent"
            recommendations.append(
                "Very similar patterns found - leverage existing solutions"
            )
        elif best_similarity >= 0.8:
            insights["match_quality"] = "good"
            recommendations.append(
                "Good pattern matches found - adapt existing approaches"
            )
        elif best_similarity >= 0.7:
            insights["match_quality"] = "moderate"
            recommendations.append(
                "Moderate similarities - use as inspiration but customize approach"
            )
        else:
            insights["match_quality"] = "weak"
            recommendations.append("Weak similarities found - proceed with caution")

        insights["best_similarity"] = best_similarity
        insights["average_similarity"] = avg_similarity
        insights["match_count"] = len(matches)

        # Pattern-specific analysis
        if pattern_type == PatternType.PROBLEM_SIMILARITY:
            recommendations.append(
                "Consider reusing successful mental model combinations from similar problems"
            )
        elif pattern_type == PatternType.MODEL_COMBINATION:
            recommendations.append(
                "This model combination has been used successfully before"
            )
        elif pattern_type == PatternType.CONTEXT_PATTERN:
            recommendations.append(
                "Similar business contexts suggest proven engagement approaches"
            )
        elif pattern_type == PatternType.PERFORMANCE_PATTERN:
            recommendations.append(
                "Similar performance patterns indicate realistic outcome expectations"
            )

        return insights, recommendations

    async def _cleanup_old_patterns(self) -> None:
        """Clean up old patterns to maintain storage limits"""

        if len(self.patterns) <= self.max_patterns_stored:
            return

        # Sort patterns by last access time and access count
        patterns_by_access = []
        for pattern_id, pattern in self.patterns.items():
            score = pattern.access_count + (
                pattern.similarity_matches * 2
            )  # Weight matches higher
            patterns_by_access.append((pattern_id, score, pattern.last_accessed))

        # Sort by access score (ascending) and access time (ascending)
        patterns_by_access.sort(key=lambda x: (x[1], x[2]))

        # Remove oldest/least used patterns
        patterns_to_remove = (
            len(self.patterns) - self.max_patterns_stored + 100
        )  # Remove extra for buffer

        for pattern_id, _, _ in patterns_by_access[:patterns_to_remove]:
            if pattern_id in self.patterns:
                pattern = self.patterns[pattern_id]

                # Remove from index
                if pattern.pattern_type in self.pattern_index:
                    if pattern_id in self.pattern_index[pattern.pattern_type]:
                        self.pattern_index[pattern.pattern_type].remove(pattern_id)

                # Remove from storage
                del self.patterns[pattern_id]

        self.logger.info(
            f"🧹 Cleaned up {patterns_to_remove} old patterns - Storage: {len(self.patterns)}/{self.max_patterns_stored}"
        )

    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pattern detection statistics"""

        # Pattern type distribution
        pattern_distribution = {}
        for pattern_type, pattern_ids in self.pattern_index.items():
            pattern_distribution[pattern_type.value] = len(pattern_ids)

        # Performance statistics
        performance_stats = {}
        if self.patterns:
            all_patterns = list(self.patterns.values())

            # Access statistics
            access_counts = [p.access_count for p in all_patterns]
            similarity_matches = [p.similarity_matches for p in all_patterns]

            performance_stats = {
                "avg_access_count": statistics.mean(access_counts),
                "max_access_count": max(access_counts),
                "avg_similarity_matches": statistics.mean(similarity_matches),
                "max_similarity_matches": max(similarity_matches),
            }

        # Vector statistics
        vector_stats = {}
        if self.patterns:
            sparsities = [p.sparsity for p in all_patterns]
            norms = [p.norm for p in all_patterns]

            vector_stats = {
                "avg_sparsity": statistics.mean(sparsities),
                "avg_norm": statistics.mean(norms),
                "vector_dimension": self.vector_dimension,
            }

        return {
            "total_patterns": len(self.patterns),
            "pattern_distribution": pattern_distribution,
            "detection_statistics": self.detection_stats,
            "performance_statistics": performance_stats,
            "vector_statistics": vector_stats,
            "similarity_threshold": self.similarity_threshold,
            "storage_utilization": len(self.patterns) / self.max_patterns_stored,
        }

    def get_system_health_metrics(self) -> Dict[str, Any]:
        """Get system health metrics from pattern detection perspective"""

        stats = self.get_pattern_statistics()

        # Determine health status
        total_patterns = stats["total_patterns"]
        detection_count = stats["detection_statistics"]["total_detections"]
        success_rate = stats["detection_statistics"]["successful_matches"] / max(
            1, detection_count
        )
        avg_detection_time = stats["detection_statistics"]["average_detection_time_ms"]

        # Health assessment
        if total_patterns >= 1000 and success_rate >= 0.7 and avg_detection_time <= 50:
            health_status = "excellent"
        elif (
            total_patterns >= 500 and success_rate >= 0.5 and avg_detection_time <= 100
        ):
            health_status = "good"
        elif total_patterns >= 100 and success_rate >= 0.3:
            health_status = "acceptable"
        else:
            health_status = "building_patterns"

        return {
            "status": health_status,
            "total_patterns": total_patterns,
            "detection_success_rate": success_rate,
            "average_detection_time_ms": avg_detection_time,
            "storage_utilization": stats["storage_utilization"],
            "pattern_types_active": len(
                [pt for pt, count in stats["pattern_distribution"].items() if count > 0]
            ),
        }

    async def simulate_pattern_learning(
        self, pattern_count: int = 50
    ) -> Dict[str, Any]:
        """Simulate pattern learning with synthetic data for testing"""

        import random

        self.logger.info(
            f"🔄 Simulating pattern learning with {pattern_count} patterns..."
        )

        # Industry and context options for realistic simulation
        industries = [
            "technology",
            "healthcare",
            "finance",
            "manufacturing",
            "retail",
            "consulting",
        ]
        company_sizes = ["startup", "small", "medium", "large", "enterprise"]
        urgencies = ["low", "medium", "high", "critical"]
        complexities = ["low", "medium", "high"]

        # Mental models for combinations
        model_pools = [
            ["critical_thinking_framework", "systems_thinking_analysis"],
            ["multi_criteria_decision_analysis", "hypothesis_testing_methodology"],
            ["scenario_planning_framework", "root_cause_analysis"],
            ["swot_analysis", "porter_five_forces"],
            ["cost_benefit_analysis", "risk_assessment_matrix"],
        ]

        simulation_results = {
            "patterns_created": 0,
            "pattern_types_created": defaultdict(int),
            "creation_time_ms": 0.0,
        }

        start_time = datetime.utcnow()

        for i in range(pattern_count):
            # Randomly select pattern type
            pattern_type = random.choice(list(PatternType))

            # Generate synthetic data based on pattern type
            if pattern_type == PatternType.PROBLEM_SIMILARITY:
                problem_templates = [
                    "Should we acquire {company} to expand our {domain} capabilities?",
                    "How do we improve {metric} in our {department} operations?",
                    "What is the best approach to implement {technology} across {scope}?",
                    "How should we respond to {competitor} entering our {market}?",
                    "What pricing strategy should we use for our new {product}?",
                ]

                template = random.choice(problem_templates)
                problem = template.format(
                    company=f"Company{random.randint(1,100)}",
                    domain=random.choice(["AI", "cloud", "mobile", "data"]),
                    metric=random.choice(["efficiency", "revenue", "satisfaction"]),
                    department=random.choice(["sales", "operations", "marketing"]),
                    technology=random.choice(["AI", "blockchain", "IoT", "5G"]),
                    scope=random.choice(["organization", "division", "team"]),
                    competitor=f"Competitor{random.randint(1,50)}",
                    market=random.choice(["domestic", "international", "regional"]),
                    product=random.choice(["service", "platform", "solution"]),
                )

                raw_data = {"problem_statement": problem}

            elif pattern_type == PatternType.CONTEXT_PATTERN:
                raw_data = {
                    "business_context": {
                        "industry": random.choice(industries),
                        "company_size": random.choice(company_sizes),
                        "urgency": random.choice(urgencies),
                        "complexity_level": random.choice(complexities),
                        "budget_range": random.choice(
                            ["$10K-$50K", "$50K-$200K", "$200K-$1M", "$1M+"]
                        ),
                        "stakeholders": random.sample(
                            ["CEO", "CTO", "CFO", "VP Sales", "Head of Operations"],
                            random.randint(2, 4),
                        ),
                    }
                }

            elif pattern_type == PatternType.MODEL_COMBINATION:
                model_pool = random.choice(model_pools)
                raw_data = {
                    "model_ids": random.sample(
                        model_pool, random.randint(1, len(model_pool))
                    )
                }

            elif pattern_type == PatternType.PERFORMANCE_PATTERN:
                base_performance = random.uniform(0.3, 0.9)
                raw_data = {
                    "performance_data": {
                        "confidence_score": base_performance
                        + random.uniform(-0.1, 0.1),
                        "effectiveness_score": base_performance
                        + random.uniform(-0.15, 0.15),
                        "stakeholder_satisfaction": base_performance
                        + random.uniform(-0.2, 0.2),
                        "implementation_efficiency": base_performance
                        + random.uniform(-0.1, 0.1),
                        "value_realization": base_performance
                        + random.uniform(-0.25, 0.25),
                    }
                }

            else:  # OUTCOME_PATTERN
                raw_data = {
                    "problem_statement": f"Strategic decision for {random.choice(industries)} company",
                    "business_context": {
                        "industry": random.choice(industries),
                        "company_size": random.choice(company_sizes),
                    },
                }

            # Add pattern
            metadata = {
                "simulation": True,
                "iteration": i,
                "created_by": "pattern_simulation",
            }

            pattern_id = await self.add_pattern(pattern_type, raw_data, metadata)

            if pattern_id:
                simulation_results["patterns_created"] += 1
                simulation_results["pattern_types_created"][pattern_type.value] += 1

        end_time = datetime.utcnow()
        simulation_results["creation_time_ms"] = (
            end_time - start_time
        ).total_seconds() * 1000

        self.logger.info(
            f"✅ Pattern learning simulation completed: {simulation_results}"
        )

        return simulation_results

    async def add_pattern(
        self,
        pattern_type: PatternType,
        pattern_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a new pattern to the engine

        Args:
            pattern_type: Type of pattern (problem similarity, model combination, etc.)
            pattern_data: Raw data for the pattern
            metadata: Optional metadata (unused for now, for compatibility)

        Returns:
            Pattern ID that was stored
        """
        pattern_id = (
            f"{pattern_type.value}_{datetime.utcnow().isoformat()}_{len(self.patterns)}"
        )

        # Create vector based on pattern type
        vector = await self._create_pattern_vector(pattern_type, pattern_data)

        # Create vectorized pattern
        vectorized_pattern = VectorizedPattern(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            vector=vector,
            metadata={},
            raw_data=pattern_data,
            dimensionality=len(vector),
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            access_count=0,
            similarity_matches=0,
        )

        # Store in engine
        self.patterns[pattern_id] = vectorized_pattern
        self.pattern_index[pattern_type].append(pattern_id)

        # Clean up if at capacity
        if len(self.patterns) > self.max_patterns_stored:
            await self._cleanup_old_patterns()

        self.logger.debug(f"Added pattern {pattern_id} of type {pattern_type.value}")
        return pattern_id

    async def store_pattern(
        self,
        pattern_type: PatternType,
        pattern_data: Dict[str, Any],
        pattern_id: Optional[str] = None,
    ) -> str:
        """Alias for add_pattern for consistency with cognitive engine usage"""
        return await self.add_pattern(pattern_type, pattern_data, None)

    async def detect_similar_patterns(
        self,
        pattern_type: PatternType,
        query_data: Dict[str, Any],
        similarity_threshold: Optional[float] = None,
        max_results: int = 5,
        similarity_metric: Optional[SimilarityMetric] = None,
    ) -> PatternDetectionResult:
        """
        Detect similar patterns to the given query

        Args:
            pattern_type: Type of patterns to search
            query_data: Data to find similar patterns for
            similarity_threshold: Minimum similarity score (uses engine default if None)
            max_results: Maximum number of results to return
            similarity_metric: Metric to use (uses engine default if None)

        Returns:
            PatternDetectionResult with found patterns and metadata
        """
        start_time = datetime.utcnow()

        # Use defaults if not specified
        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold
        if similarity_metric is None:
            similarity_metric = self.default_similarity_metric

        # Create query vector
        query_vector = await self._create_pattern_vector(pattern_type, query_data)

        # Get patterns of this type
        candidate_pattern_ids = self.pattern_index.get(pattern_type, [])

        # Calculate similarities
        pattern_similarities = []
        for pattern_id in candidate_pattern_ids:
            if pattern_id not in self.patterns:
                continue

            pattern = self.patterns[pattern_id]
            similarity_score = self.calculate_similarity(
                query_vector, pattern.vector, similarity_metric
            )

            if similarity_score >= similarity_threshold:
                pattern_similarities.append(
                    PatternSimilarity(
                        pattern_id=pattern_id,
                        pattern_data=pattern.raw_data,
                        similarity_score=similarity_score,
                    )
                )

                # Update usage tracking
                pattern.access_count += 1
                pattern.last_accessed = datetime.utcnow()
                pattern.similarity_matches += 1

        # Sort by similarity score (descending)
        pattern_similarities.sort(key=lambda x: x.similarity_score, reverse=True)

        # Limit results
        top_patterns = pattern_similarities[:max_results]

        # Calculate detection time
        detection_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Update statistics
        self.detection_stats["total_detections"] += 1
        if top_patterns:
            self.detection_stats["successful_matches"] += 1
        self.detection_stats["pattern_types_detected"][pattern_type] += 1

        # Update average detection time
        current_avg = self.detection_stats["average_detection_time_ms"]
        total_detections = self.detection_stats["total_detections"]
        new_avg = (
            (current_avg * (total_detections - 1)) + detection_time
        ) / total_detections
        self.detection_stats["average_detection_time_ms"] = new_avg

        # Create result
        result = PatternDetectionResult(
            patterns_found=top_patterns,
            total_patterns_searched=len(candidate_pattern_ids),
            detection_time_ms=detection_time,
            similarity_threshold_used=similarity_threshold,
            similarity_metric_used=similarity_metric,
            detection_successful=len(top_patterns) > 0,
            average_similarity=(
                sum(p.similarity_score for p in top_patterns) / len(top_patterns)
                if top_patterns
                else 0.0
            ),
        )

        self.logger.debug(
            f"Pattern detection completed: {len(top_patterns)} matches found in {detection_time:.2f}ms"
        )
        return result

    async def _create_pattern_vector(
        self, pattern_type: PatternType, pattern_data: Dict[str, Any]
    ) -> np.ndarray:
        """Create vector representation based on pattern type and data"""

        if pattern_type == PatternType.PROBLEM_SIMILARITY:
            # For problem similarity, focus on problem statement and context
            problem_text = pattern_data.get("problem_statement", "")
            context_text = str(pattern_data.get("business_context", ""))
            combined_text = f"{problem_text} {context_text}"
            return self._create_text_vector(combined_text)

        elif pattern_type == PatternType.MODEL_COMBINATION:
            # For model combinations, use model IDs and performance data
            model_id = pattern_data.get("model_id", "")
            model_ids = [model_id] if model_id else pattern_data.get("model_ids", [])
            if isinstance(model_ids, str):
                model_ids = [model_ids]
            model_vector = self._create_model_combination_vector(model_ids)

            # Add performance context
            performance_data = {
                "performance_score": pattern_data.get("performance_score", 0.5),
                "reasoning_quality": pattern_data.get("reasoning_quality", ""),
            }
            performance_vector = self._create_performance_vector(performance_data)

            # Combine vectors
            combined_vector = np.concatenate([model_vector, performance_vector])
            # Pad or truncate to target dimension
            if len(combined_vector) < self.vector_dimension:
                combined_vector = np.pad(
                    combined_vector, (0, self.vector_dimension - len(combined_vector))
                )
            else:
                combined_vector = combined_vector[: self.vector_dimension]

            return combined_vector

        elif pattern_type == PatternType.CONTEXT_PATTERN:
            # For context patterns, use business context
            context = pattern_data.get("business_context", {})
            if isinstance(context, str):
                return self._create_text_vector(context)
            else:
                return self._create_context_vector(context)

        elif pattern_type == PatternType.OUTCOME_PATTERN:
            # For outcome patterns, focus on results and success metrics
            outcome_text = f"{pattern_data.get('outcome_description', '')} {pattern_data.get('success_metrics', '')}"
            return self._create_text_vector(outcome_text)

        elif pattern_type == PatternType.PERFORMANCE_PATTERN:
            # For performance patterns, use performance metrics
            return self._create_performance_vector(pattern_data)

        else:
            # Default: convert all data to text and vectorize
            text_data = str(pattern_data)
            return self._create_text_vector(text_data)

    async def _cleanup_old_patterns(self, cleanup_ratio: float = 0.1):
        """Remove oldest or least-used patterns to free space"""

        cleanup_count = int(len(self.patterns) * cleanup_ratio)
        if cleanup_count == 0:
            return

        # Sort patterns by last accessed time (oldest first)
        sorted_patterns = sorted(
            self.patterns.items(), key=lambda item: item[1].last_accessed
        )

        # Remove oldest patterns
        for pattern_id, pattern in sorted_patterns[:cleanup_count]:
            # Remove from main storage
            del self.patterns[pattern_id]

            # Remove from index
            pattern_type = pattern.pattern_type
            if pattern_id in self.pattern_index[pattern_type]:
                self.pattern_index[pattern_type].remove(pattern_id)

        self.logger.debug(f"Cleaned up {cleanup_count} old patterns")

    def get_statistics(self) -> Dict[str, int]:
        """Get basic pattern count statistics"""
        stats = {}
        for pattern_type in PatternType:
            stats[pattern_type.value] = len(self.pattern_index.get(pattern_type, []))
        return stats


# Global VectorSimilarityEngine instance
_vector_similarity_engine_instance: Optional[VectorSimilarityEngine] = None


def get_vector_similarity_engine() -> VectorSimilarityEngine:
    """Get or create global VectorSimilarityEngine instance"""
    global _vector_similarity_engine_instance

    if _vector_similarity_engine_instance is None:
        _vector_similarity_engine_instance = VectorSimilarityEngine()

    return _vector_similarity_engine_instance


async def detect_similar_patterns(
    pattern_type: PatternType,
    query_data: Dict[str, Any],
    top_k: int = 10,
    similarity_threshold: float = None,
) -> PatternDetectionResult:
    """Convenience function to detect similar patterns"""
    engine = get_vector_similarity_engine()
    return await engine.detect_similar_patterns(
        pattern_type, query_data, top_k, similarity_threshold
    )


async def add_pattern(
    pattern_type: PatternType, raw_data: Dict[str, Any], metadata: Dict[str, Any] = None
) -> str:
    """Convenience function to add pattern"""
    engine = get_vector_similarity_engine()
    return await engine.add_pattern(pattern_type, raw_data, metadata)
