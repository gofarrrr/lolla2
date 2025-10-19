"""
METIS Cognitive Signature Matching System
Advanced pattern recognition for cognitive processing identification and optimization
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
from collections import Counter
import logging

# Import for vector similarity computations
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class SignatureType(Enum):
    """Types of cognitive signatures we can identify"""

    ANALYTICAL_PATTERN = "analytical_pattern"
    CREATIVE_SYNTHESIS = "creative_synthesis"
    STRATEGIC_REASONING = "strategic_reasoning"
    SYSTEMATIC_DECOMPOSITION = "systematic_decomposition"
    HYPOTHESIS_DRIVEN = "hypothesis_driven"
    EVIDENCE_BASED = "evidence_based"
    HOLISTIC_SYSTEMS = "holistic_systems"
    CRITICAL_EVALUATION = "critical_evaluation"


class MatchConfidence(Enum):
    """Confidence levels for signature matching"""

    LOW = "low"  # 0.3-0.5
    MEDIUM = "medium"  # 0.5-0.7
    HIGH = "high"  # 0.7-0.9
    EXCEPTIONAL = "exceptional"  # 0.9+


@dataclass
class CognitiveSignature:
    """Represents a cognitive processing signature"""

    signature_id: str
    signature_type: SignatureType
    mental_models_used: List[str]
    processing_patterns: List[str]
    complexity_indicators: Dict[str, float]
    success_metrics: Dict[str, float]
    context_markers: List[str]
    timestamp: datetime
    source_engagement_id: str

    def to_vector(self) -> List[float]:
        """Convert signature to numerical vector for similarity matching"""
        # Create feature vector from signature components
        vector = []

        # Mental model features (0-1 for each known model)
        known_models = [
            "systems_thinking",
            "critical_analysis",
            "mece_structuring",
            "hypothesis_testing",
            "decision_analysis",
            "root_cause_analysis",
            "scenario_planning",
            "value_chain_analysis",
            "swot_analysis",
        ]
        for model in known_models:
            vector.append(1.0 if model in self.mental_models_used else 0.0)

        # Processing pattern features
        known_patterns = [
            "top_down_decomposition",
            "bottom_up_synthesis",
            "lateral_thinking",
            "sequential_analysis",
            "parallel_processing",
            "iterative_refinement",
        ]
        for pattern in known_patterns:
            vector.append(1.0 if pattern in self.processing_patterns else 0.0)

        # Complexity indicators
        vector.extend(
            [
                self.complexity_indicators.get("analytical_depth", 0.0),
                self.complexity_indicators.get("synthesis_complexity", 0.0),
                self.complexity_indicators.get("reasoning_layers", 0.0),
                self.complexity_indicators.get("evidence_integration", 0.0),
            ]
        )

        # Success metrics
        vector.extend(
            [
                self.success_metrics.get("accuracy_score", 0.0),
                self.success_metrics.get("insight_quality", 0.0),
                self.success_metrics.get("actionability", 0.0),
                self.success_metrics.get("completeness", 0.0),
            ]
        )

        return vector


@dataclass
class SignatureMatch:
    """Represents a match between cognitive signatures"""

    matched_signature: CognitiveSignature
    similarity_score: float
    confidence_level: MatchConfidence
    matching_dimensions: List[str]
    optimization_suggestions: List[str]
    predicted_performance: Dict[str, float]


class CognitiveSignatureMatcher:
    """Advanced cognitive signature matching system"""

    def __init__(
        self,
        similarity_threshold: float = 0.65,
        min_confidence: float = 0.5,
        max_matches_per_query: int = 5,
    ):
        self.similarity_threshold = similarity_threshold
        self.min_confidence = min_confidence
        self.max_matches_per_query = max_matches_per_query

        # Storage for cognitive signatures
        self.signature_database: Dict[str, CognitiveSignature] = {}
        self.signature_vectors: Dict[str, List[float]] = {}

        # Performance tracking
        self.match_statistics = {
            "total_matches": 0,
            "successful_predictions": 0,
            "average_similarity": 0.0,
            "signature_types_distribution": Counter(),
            "optimization_impact": [],
        }

        # Text-based similarity matcher (if sklearn available)
        self.text_vectorizer = None
        if SKLEARN_AVAILABLE:
            self.text_vectorizer = TfidfVectorizer(
                max_features=100, stop_words="english"
            )

        self.logger = logging.getLogger(__name__)
        self.logger.info("🧠 Cognitive Signature Matcher initialized")

    async def add_signature(self, signature: CognitiveSignature) -> bool:
        """Add a new cognitive signature to the database"""
        try:
            # Store signature
            self.signature_database[signature.signature_id] = signature

            # Compute and store vector representation
            vector = signature.to_vector()
            self.signature_vectors[signature.signature_id] = vector

            # Update statistics
            self.match_statistics["signature_types_distribution"][
                signature.signature_type.value
            ] += 1

            self.logger.info(f"✅ Added cognitive signature {signature.signature_id}")
            return True

        except Exception as e:
            self.logger.error(
                f"❌ Failed to add signature {signature.signature_id}: {e}"
            )
            return False

    async def find_matching_signatures(
        self, query_signature: CognitiveSignature, problem_context: str = ""
    ) -> List[SignatureMatch]:
        """Find signatures that match the query signature"""
        try:
            if len(self.signature_database) == 0:
                self.logger.warning("No signatures in database for matching")
                return []

            query_vector = query_signature.to_vector()
            matches = []

            # Calculate similarities with all stored signatures
            for sig_id, signature in self.signature_database.items():
                if sig_id == query_signature.signature_id:
                    continue  # Skip self-match

                stored_vector = self.signature_vectors[sig_id]
                similarity = self._calculate_vector_similarity(
                    query_vector, stored_vector
                )

                if similarity >= self.similarity_threshold:
                    # Calculate confidence level
                    confidence = self._calculate_confidence(
                        similarity, signature, query_signature
                    )

                    if confidence.value != MatchConfidence.LOW.value:
                        # Generate optimization suggestions
                        optimizations = await self._generate_optimizations(
                            signature, query_signature
                        )

                        # Predict performance based on historical data
                        predicted_perf = self._predict_performance(
                            signature, query_signature
                        )

                        # Identify matching dimensions
                        matching_dims = self._identify_matching_dimensions(
                            signature, query_signature
                        )

                        match = SignatureMatch(
                            matched_signature=signature,
                            similarity_score=similarity,
                            confidence_level=confidence,
                            matching_dimensions=matching_dims,
                            optimization_suggestions=optimizations,
                            predicted_performance=predicted_perf,
                        )
                        matches.append(match)

            # Sort by similarity and take top matches
            matches.sort(key=lambda x: x.similarity_score, reverse=True)
            matches = matches[: self.max_matches_per_query]

            # Update statistics
            self.match_statistics["total_matches"] += len(matches)
            if matches:
                avg_sim = sum(m.similarity_score for m in matches) / len(matches)
                current_avg = self.match_statistics["average_similarity"]
                total_matches = self.match_statistics["total_matches"]
                self.match_statistics["average_similarity"] = (
                    current_avg * (total_matches - len(matches))
                    + avg_sim * len(matches)
                ) / total_matches

            self.logger.info(
                f"🎯 Found {len(matches)} signature matches (similarity >= {self.similarity_threshold})"
            )
            return matches

        except Exception as e:
            self.logger.error(f"❌ Signature matching failed: {e}")
            return []

    async def learn_from_engagement(
        self,
        engagement_id: str,
        mental_models_used: List[str],
        processing_approach: Dict[str, Any],
        performance_metrics: Dict[str, float],
        problem_context: str,
    ) -> CognitiveSignature:
        """Learn and create a cognitive signature from an engagement"""
        try:
            # Extract signature components from engagement data
            signature_type = self._determine_signature_type(
                mental_models_used, processing_approach
            )

            processing_patterns = self._extract_processing_patterns(processing_approach)

            complexity_indicators = {
                "analytical_depth": processing_approach.get("reasoning_depth", 0.5),
                "synthesis_complexity": processing_approach.get("synthesis_level", 0.5),
                "reasoning_layers": min(len(mental_models_used) / 5.0, 1.0),
                "evidence_integration": processing_approach.get("evidence_weight", 0.5),
            }

            context_markers = self._extract_context_markers(problem_context)

            # Create signature
            signature = CognitiveSignature(
                signature_id=f"sig_{engagement_id}_{int(datetime.now().timestamp())}",
                signature_type=signature_type,
                mental_models_used=mental_models_used,
                processing_patterns=processing_patterns,
                complexity_indicators=complexity_indicators,
                success_metrics=performance_metrics,
                context_markers=context_markers,
                timestamp=datetime.now(),
                source_engagement_id=engagement_id,
            )

            # Add to database
            await self.add_signature(signature)

            self.logger.info(
                f"🧠 Learned cognitive signature from engagement {engagement_id}"
            )
            return signature

        except Exception as e:
            self.logger.error(
                f"❌ Failed to learn from engagement {engagement_id}: {e}"
            )
            raise

    def _calculate_vector_similarity(
        self, vec1: List[float], vec2: List[float]
    ) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            if len(vec1) != len(vec2):
                return 0.0

            # Convert to numpy arrays for efficient computation
            v1 = np.array(vec1)
            v2 = np.array(vec2)

            # Cosine similarity
            dot_product = np.dot(v1, v2)
            norms = np.linalg.norm(v1) * np.linalg.norm(v2)

            if norms == 0:
                return 0.0

            similarity = dot_product / norms
            return max(0.0, similarity)  # Ensure non-negative

        except Exception as e:
            self.logger.error(f"Vector similarity calculation failed: {e}")
            return 0.0

    def _calculate_confidence(
        self,
        similarity: float,
        matched_sig: CognitiveSignature,
        query_sig: CognitiveSignature,
    ) -> MatchConfidence:
        """Calculate confidence level for a signature match"""
        # Base confidence from similarity score
        base_confidence = similarity

        # Boost confidence for signature type alignment
        if matched_sig.signature_type == query_sig.signature_type:
            base_confidence += 0.1

        # Boost for overlapping mental models
        overlap_ratio = len(
            set(matched_sig.mental_models_used) & set(query_sig.mental_models_used)
        ) / max(len(matched_sig.mental_models_used), 1)
        base_confidence += overlap_ratio * 0.1

        # Boost for recent successful signatures
        days_old = (datetime.now() - matched_sig.timestamp).days
        if days_old < 7 and matched_sig.success_metrics.get("accuracy_score", 0) > 0.8:
            base_confidence += 0.05

        # Determine confidence level
        if base_confidence >= 0.9:
            return MatchConfidence.EXCEPTIONAL
        elif base_confidence >= 0.7:
            return MatchConfidence.HIGH
        elif base_confidence >= 0.5:
            return MatchConfidence.MEDIUM
        else:
            return MatchConfidence.LOW

    async def _generate_optimizations(
        self, matched_sig: CognitiveSignature, query_sig: CognitiveSignature
    ) -> List[str]:
        """Generate optimization suggestions based on signature comparison"""
        suggestions = []

        # Model recommendations
        matched_models = set(matched_sig.mental_models_used)
        query_models = set(query_sig.mental_models_used)

        missing_models = matched_models - query_models
        if missing_models:
            suggestions.append(
                f"Consider adding mental models: {', '.join(list(missing_models)[:3])}"
            )

        # Processing pattern recommendations
        matched_patterns = set(matched_sig.processing_patterns)
        query_patterns = set(query_sig.processing_patterns)

        missing_patterns = matched_patterns - query_patterns
        if missing_patterns:
            suggestions.append(
                f"Try processing patterns: {', '.join(list(missing_patterns)[:2])}"
            )

        # Complexity adjustments
        if (
            matched_sig.complexity_indicators.get("analytical_depth", 0)
            > query_sig.complexity_indicators.get("analytical_depth", 0) + 0.2
        ):
            suggestions.append("Increase analytical depth for better performance")

        if (
            matched_sig.complexity_indicators.get("evidence_integration", 0)
            > query_sig.complexity_indicators.get("evidence_integration", 0) + 0.2
        ):
            suggestions.append("Integrate more evidence sources for stronger analysis")

        # Performance-based suggestions
        matched_performance = matched_sig.success_metrics.get("insight_quality", 0)
        if matched_performance > 0.8:
            suggestions.append(
                "This signature pattern has shown high insight quality in similar contexts"
            )

        return suggestions[:4]  # Limit to most important suggestions

    def _predict_performance(
        self, matched_sig: CognitiveSignature, query_sig: CognitiveSignature
    ) -> Dict[str, float]:
        """Predict performance based on signature similarity"""
        # Base prediction from matched signature's historical performance
        base_performance = matched_sig.success_metrics.copy()

        # Adjust based on signature differences
        similarity = self._calculate_vector_similarity(
            matched_sig.to_vector(), query_sig.to_vector()
        )

        # Scale performance predictions by similarity
        predicted = {}
        for metric, value in base_performance.items():
            # Higher similarity = more reliable prediction
            confidence_multiplier = 0.7 + (similarity * 0.3)
            predicted[metric] = value * confidence_multiplier

        return predicted

    def _identify_matching_dimensions(
        self, matched_sig: CognitiveSignature, query_sig: CognitiveSignature
    ) -> List[str]:
        """Identify which dimensions contribute to the signature match"""
        dimensions = []

        # Mental model overlap
        model_overlap = set(matched_sig.mental_models_used) & set(
            query_sig.mental_models_used
        )
        if model_overlap:
            dimensions.append(f"Mental models: {', '.join(list(model_overlap)[:3])}")

        # Processing pattern overlap
        pattern_overlap = set(matched_sig.processing_patterns) & set(
            query_sig.processing_patterns
        )
        if pattern_overlap:
            dimensions.append(f"Processing: {', '.join(list(pattern_overlap)[:2])}")

        # Signature type match
        if matched_sig.signature_type == query_sig.signature_type:
            dimensions.append(f"Cognitive approach: {matched_sig.signature_type.value}")

        # Complexity similarity
        complexity_diff = abs(
            matched_sig.complexity_indicators.get("analytical_depth", 0)
            - query_sig.complexity_indicators.get("analytical_depth", 0)
        )
        if complexity_diff < 0.2:
            dimensions.append("Similar analytical complexity")

        return dimensions

    def _determine_signature_type(
        self, mental_models: List[str], processing_approach: Dict[str, Any]
    ) -> SignatureType:
        """Determine the primary signature type based on models and approach"""
        # Analyze mental models for type indicators
        if "systems_thinking" in mental_models or "holistic" in str(
            processing_approach
        ):
            return SignatureType.HOLISTIC_SYSTEMS
        elif (
            "hypothesis_testing" in mental_models
            or "scientific_method" in mental_models
        ):
            return SignatureType.HYPOTHESIS_DRIVEN
        elif (
            "critical_analysis" in mental_models or "critical_thinking" in mental_models
        ):
            return SignatureType.CRITICAL_EVALUATION
        elif "mece_structuring" in mental_models or "decomposition" in str(
            processing_approach
        ):
            return SignatureType.SYSTEMATIC_DECOMPOSITION
        elif "decision_analysis" in mental_models or "strategic" in str(
            processing_approach
        ):
            return SignatureType.STRATEGIC_REASONING
        elif "creative_thinking" in mental_models or "synthesis" in str(
            processing_approach
        ):
            return SignatureType.CREATIVE_SYNTHESIS
        else:
            return SignatureType.ANALYTICAL_PATTERN

    def _extract_processing_patterns(
        self, processing_approach: Dict[str, Any]
    ) -> List[str]:
        """Extract processing patterns from approach data"""
        patterns = []

        approach_str = str(processing_approach).lower()

        if "top_down" in approach_str or "hierarchical" in approach_str:
            patterns.append("top_down_decomposition")
        if "bottom_up" in approach_str or "synthesis" in approach_str:
            patterns.append("bottom_up_synthesis")
        if "iterative" in approach_str or "refinement" in approach_str:
            patterns.append("iterative_refinement")
        if "parallel" in approach_str or "concurrent" in approach_str:
            patterns.append("parallel_processing")
        if "lateral" in approach_str or "creative" in approach_str:
            patterns.append("lateral_thinking")
        if "sequential" in approach_str or "step_by_step" in approach_str:
            patterns.append("sequential_analysis")

        return patterns if patterns else ["sequential_analysis"]  # Default pattern

    def _extract_context_markers(self, problem_context: str) -> List[str]:
        """Extract context markers from problem description"""
        markers = []
        context_lower = problem_context.lower()

        # Domain markers
        domains = {
            "strategy": ["strategy", "strategic", "business model", "competitive"],
            "operations": ["operations", "process", "efficiency", "optimization"],
            "finance": ["financial", "revenue", "cost", "investment", "roi"],
            "technology": ["technology", "digital", "system", "platform", "tech"],
            "marketing": ["marketing", "brand", "customer", "market"],
            "hr": ["human resources", "talent", "people", "culture", "hr"],
        }

        for domain, keywords in domains.items():
            if any(keyword in context_lower for keyword in keywords):
                markers.append(f"domain_{domain}")

        # Complexity markers
        if len(problem_context.split()) > 100:
            markers.append("high_complexity")
        elif len(problem_context.split()) > 50:
            markers.append("medium_complexity")
        else:
            markers.append("low_complexity")

        # Urgency markers
        urgency_keywords = ["urgent", "immediate", "crisis", "emergency", "asap"]
        if any(keyword in context_lower for keyword in urgency_keywords):
            markers.append("high_urgency")

        return markers

    async def get_signature_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about signature matching"""
        return {
            "total_signatures": len(self.signature_database),
            "signature_types": dict(
                self.match_statistics["signature_types_distribution"]
            ),
            "total_matches_performed": self.match_statistics["total_matches"],
            "average_similarity_score": round(
                self.match_statistics["average_similarity"], 3
            ),
            "successful_predictions": self.match_statistics["successful_predictions"],
            "prediction_accuracy": (
                self.match_statistics["successful_predictions"]
                / max(self.match_statistics["total_matches"], 1)
                * 100
            ),
            "recent_signatures": len(
                [
                    s
                    for s in self.signature_database.values()
                    if (datetime.now() - s.timestamp).days < 7
                ]
            ),
            "high_performance_signatures": len(
                [
                    s
                    for s in self.signature_database.values()
                    if s.success_metrics.get("accuracy_score", 0) > 0.8
                ]
            ),
        }

    async def export_signatures(self, filepath: str) -> bool:
        """Export signature database to JSON file"""
        try:
            export_data = {
                "signatures": {},
                "statistics": self.match_statistics,
                "export_timestamp": datetime.now().isoformat(),
                "total_signatures": len(self.signature_database),
            }

            # Convert signatures to serializable format
            for sig_id, signature in self.signature_database.items():
                sig_dict = asdict(signature)
                sig_dict["timestamp"] = signature.timestamp.isoformat()
                sig_dict["signature_type"] = signature.signature_type.value
                export_data["signatures"][sig_id] = sig_dict

            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2)

            self.logger.info(
                f"✅ Exported {len(self.signature_database)} signatures to {filepath}"
            )
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to export signatures: {e}")
            return False


# Global signature matcher instance
_signature_matcher: Optional[CognitiveSignatureMatcher] = None


async def get_signature_matcher() -> CognitiveSignatureMatcher:
    """Get global cognitive signature matcher instance"""
    global _signature_matcher
    if _signature_matcher is None:
        _signature_matcher = CognitiveSignatureMatcher(
            similarity_threshold=0.65, min_confidence=0.5, max_matches_per_query=5
        )
    return _signature_matcher
