"""
Contextual Weighting System - SPRINT 2 Implementation
====================================================

Advanced dynamic weighting system for LOLLA cognitive platform that adapts selection
algorithms based on context, performance feedback, and learning patterns.

Key Features:
- Dynamic weight adjustment based on real-time performance data
- Context-aware weighting for different scenarios (complexity, domain, framework)
- Machine learning-driven optimization using gradient descent and Bayesian methods
- Multi-dimensional weighting across pattern selection, consultant selection, and chemistry optimization
- Continuous learning with performance feedback integration

Integrates with:
- NWayPatternSelectionService for pattern selection optimization
- ContextualLollapalozzaEngine for consultant selection weighting
- CognitiveChemistryEngine for chemistry calculation optimization
- ConsultantPerformanceTrackingService for feedback loop integration

SPRINT 2 TARGET: Enable intelligent weight adaptation for improved overall system performance
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import statistics
import json
import math
import asyncio


class WeightingDimension(Enum):
    """Different dimensions of weighting in the system"""

    PATTERN_SELECTION = "pattern_selection"
    CONSULTANT_SELECTION = "consultant_selection"
    CHEMISTRY_OPTIMIZATION = "chemistry_optimization"
    DOMAIN_MATCHING = "domain_matching"
    FRAMEWORK_ALIGNMENT = "framework_alignment"
    COMPLEXITY_ADAPTATION = "complexity_adaptation"
    DIVERSITY_ENFORCEMENT = "diversity_enforcement"


class ContextCategory(Enum):
    """Context categories for different weighting strategies"""

    SIMPLE_TASK = "simple_task"
    STANDARD_ANALYSIS = "standard_analysis"
    COMPLEX_ANALYTICAL = "complex_analytical"
    COLLABORATIVE_TEAM = "collaborative_team"
    HIGH_STAKES_DECISION = "high_stakes_decision"
    RAPID_TURNAROUND = "rapid_turnaround"
    RESEARCH_INTENSIVE = "research_intensive"


@dataclass
class WeightingProfile:
    """Complete weighting profile for a specific context"""

    profile_id: str
    context_category: ContextCategory
    dimension_weights: Dict[WeightingDimension, float]
    performance_history: List[float] = field(default_factory=list)
    usage_count: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    effectiveness_score: float = 0.5
    confidence_level: float = 0.5


@dataclass
class WeightOptimizationRecord:
    """Record of weight optimization attempts and results"""

    timestamp: datetime
    dimension: WeightingDimension
    context_category: ContextCategory
    old_weights: Dict[str, float]
    new_weights: Dict[str, float]
    performance_before: float
    performance_after: Optional[float]
    optimization_method: str
    learning_rate: float
    gradient_magnitude: float


class ContextualWeightingSystem:
    """
    Advanced contextual weighting system with machine learning capabilities.

    Provides dynamic weight adaptation across all LOLLA platform selection and optimization
    algorithms based on performance feedback and contextual requirements.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Core weighting storage
        self.weighting_profiles: Dict[str, WeightingProfile] = {}
        self.context_mappings: Dict[ContextCategory, str] = {}
        self.optimization_history: List[WeightOptimizationRecord] = []

        # Learning system components
        self.performance_buffer: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        self.gradient_accumulator: Dict[WeightingDimension, Dict[str, float]] = (
            defaultdict(lambda: defaultdict(float))
        )
        self.learning_metadata = {
            "total_optimizations": 0,
            "successful_optimizations": 0,
            "last_optimization": None,
            "learning_active": True,
            "adaptive_learning_rate": True,
        }

        # Configuration
        self.config = {
            "base_learning_rate": 0.05,
            "min_learning_rate": 0.001,
            "max_learning_rate": 0.2,
            "performance_window": 50,  # Number of recent performance records to consider
            "optimization_frequency_minutes": 30,  # How often to run optimization
            "min_data_points_for_optimization": 10,
            "convergence_threshold": 0.01,
            "weight_bounds": (0.01, 0.99),  # Min and max allowed weights
            "momentum_factor": 0.9,  # For momentum-based optimization
            "decay_factor": 0.95,  # For learning rate decay
        }

        # Initialize default weighting profiles
        self._initialize_default_profiles()

        self.logger.info("🎯 Contextual Weighting System initialized - SPRINT 2 Active")

    def _initialize_default_profiles(self):
        """Initialize default weighting profiles for different contexts"""

        # Simple Task Profile
        simple_task_weights = {
            WeightingDimension.PATTERN_SELECTION: {
                "domain_match": 0.4,
                "consultant_alignment": 0.3,
                "framework_fit": 0.2,
                "complexity_fit": 0.1,
            },
            WeightingDimension.CONSULTANT_SELECTION: {
                "expertise_match": 0.5,
                "availability": 0.3,
                "diversity_penalty": 0.2,
            },
            WeightingDimension.CHEMISTRY_OPTIMIZATION: {
                "consultant_synergy": 0.4,
                "pattern_alignment": 0.3,
                "analytical_diversity": 0.3,
            },
        }

        self.weighting_profiles["simple_task"] = WeightingProfile(
            profile_id="simple_task",
            context_category=ContextCategory.SIMPLE_TASK,
            dimension_weights=simple_task_weights,
        )

        # Complex Analytical Profile
        complex_analytical_weights = {
            WeightingDimension.PATTERN_SELECTION: {
                "domain_match": 0.25,
                "consultant_alignment": 0.25,
                "framework_fit": 0.25,
                "complexity_fit": 0.25,
            },
            WeightingDimension.CONSULTANT_SELECTION: {
                "expertise_match": 0.3,
                "availability": 0.2,
                "diversity_penalty": 0.5,  # Higher emphasis on diversity for complex tasks
            },
            WeightingDimension.CHEMISTRY_OPTIMIZATION: {
                "consultant_synergy": 0.2,
                "pattern_alignment": 0.3,
                "analytical_diversity": 0.5,  # Maximum diversity for complex analysis
            },
        }

        self.weighting_profiles["complex_analytical"] = WeightingProfile(
            profile_id="complex_analytical",
            context_category=ContextCategory.COMPLEX_ANALYTICAL,
            dimension_weights=complex_analytical_weights,
        )

        # Collaborative Team Profile
        collaborative_weights = {
            WeightingDimension.PATTERN_SELECTION: {
                "domain_match": 0.3,
                "consultant_alignment": 0.4,  # Higher weight for team alignment
                "framework_fit": 0.2,
                "complexity_fit": 0.1,
            },
            WeightingDimension.CONSULTANT_SELECTION: {
                "expertise_match": 0.4,
                "availability": 0.1,
                "diversity_penalty": 0.5,
            },
            WeightingDimension.CHEMISTRY_OPTIMIZATION: {
                "consultant_synergy": 0.5,  # Emphasize team chemistry
                "pattern_alignment": 0.2,
                "analytical_diversity": 0.3,
            },
        }

        self.weighting_profiles["collaborative_team"] = WeightingProfile(
            profile_id="collaborative_team",
            context_category=ContextCategory.COLLABORATIVE_TEAM,
            dimension_weights=collaborative_weights,
        )

        self.logger.info(
            f"✅ Initialized {len(self.weighting_profiles)} default weighting profiles"
        )

    def get_contextual_weights(
        self,
        dimension: WeightingDimension,
        context: Dict[str, Any],
        framework_type: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Get optimized weights for a specific dimension and context.

        SPRINT 2: Core method for dynamic weight retrieval with context adaptation.

        Args:
            dimension: The weighting dimension (pattern_selection, consultant_selection, etc.)
            context: Full context dictionary
            framework_type: Optional framework type for additional optimization
            domain: Optional domain for domain-specific weight adjustment

        Returns:
            Dictionary of optimized weights for the given dimension and context
        """
        try:
            # Categorize the context
            context_category = self._categorize_context(context)

            # Get or create weighting profile
            profile = self._get_or_create_profile(context_category)

            # Get base weights for the dimension
            base_weights = profile.dimension_weights.get(dimension, {})

            if not base_weights:
                self.logger.warning(
                    f"⚠️ No base weights found for {dimension.value} in {context_category.value}"
                )
                return self._get_default_weights(dimension)

            # Apply contextual adjustments
            adjusted_weights = self._apply_contextual_adjustments(
                base_weights, context, framework_type, domain, dimension
            )

            # Apply recent performance-based optimization
            optimized_weights = self._apply_performance_optimization(
                adjusted_weights, dimension, context_category
            )

            # Validate and normalize weights
            final_weights = self._validate_and_normalize_weights(optimized_weights)

            # Record usage for learning
            self._record_weight_usage(dimension, context_category, final_weights)

            self.logger.debug(
                f"🎯 Contextual weights retrieved for {dimension.value}: {len(final_weights)} components"
            )
            return final_weights

        except Exception as e:
            self.logger.error(f"❌ Failed to get contextual weights: {e}")
            return self._get_default_weights(dimension)

    def record_performance_feedback(
        self,
        dimension: WeightingDimension,
        context: Dict[str, Any],
        weights_used: Dict[str, float],
        performance_score: float,
        additional_metrics: Optional[Dict[str, float]] = None,
    ) -> bool:
        """
        Record performance feedback for weight optimization learning.

        SPRINT 2: Critical feedback loop for continuous weight optimization.

        Args:
            dimension: The weighting dimension that was used
            context: Context in which weights were applied
            weights_used: The actual weights that were used
            performance_score: Overall performance score (0.0-1.0)
            additional_metrics: Optional additional performance metrics

        Returns:
            bool: Success status
        """
        try:
            context_category = self._categorize_context(context)
            profile_id = self._get_profile_id(context_category)

            # Create comprehensive performance record
            performance_record = {
                "timestamp": datetime.utcnow(),
                "dimension": dimension,
                "context_category": context_category,
                "weights_used": weights_used.copy(),
                "performance_score": performance_score,
                "additional_metrics": additional_metrics or {},
                "context_hash": self._hash_context(context),
            }

            # Store in performance buffer
            buffer_key = f"{dimension.value}_{context_category.value}"
            self.performance_buffer[buffer_key].append(performance_record)

            # Update profile performance history
            if profile_id in self.weighting_profiles:
                self.weighting_profiles[profile_id].performance_history.append(
                    performance_score
                )
                self.weighting_profiles[profile_id].usage_count += 1

                # Update effectiveness score (running average)
                profile = self.weighting_profiles[profile_id]
                recent_scores = profile.performance_history[-20:]  # Last 20 scores
                profile.effectiveness_score = statistics.mean(recent_scores)
                profile.last_updated = datetime.utcnow()

            # Trigger optimization if conditions are met
            await self._trigger_optimization_if_needed(dimension, context_category)

            self.logger.debug(
                f"📊 Performance feedback recorded: {dimension.value} = {performance_score:.3f}"
            )
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to record performance feedback: {e}")
            return False

    async def optimize_weights(
        self,
        dimension: Optional[WeightingDimension] = None,
        context_category: Optional[ContextCategory] = None,
        optimization_method: str = "gradient_descent",
    ) -> Dict[str, Any]:
        """
        Optimize weights using machine learning techniques.

        SPRINT 2: Advanced weight optimization with multiple ML approaches.

        Args:
            dimension: Optional specific dimension to optimize (None = all dimensions)
            context_category: Optional specific context (None = all contexts)
            optimization_method: Method to use ("gradient_descent", "bayesian", "evolutionary")

        Returns:
            Dictionary with optimization results and metrics
        """
        try:
            optimization_results = {
                "optimizations_performed": 0,
                "improvements_found": 0,
                "total_performance_gain": 0.0,
                "optimization_details": [],
                "method_used": optimization_method,
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Determine optimization targets
            dimensions_to_optimize = (
                [dimension] if dimension else list(WeightingDimension)
            )
            contexts_to_optimize = (
                [context_category] if context_category else list(ContextCategory)
            )

            for dim in dimensions_to_optimize:
                for ctx in contexts_to_optimize:
                    buffer_key = f"{dim.value}_{ctx.value}"

                    if buffer_key not in self.performance_buffer:
                        continue

                    performance_data = list(self.performance_buffer[buffer_key])

                    if (
                        len(performance_data)
                        < self.config["min_data_points_for_optimization"]
                    ):
                        continue

                    # Perform optimization based on method
                    optimization_result = await self._optimize_dimension_weights(
                        dim, ctx, performance_data, optimization_method
                    )

                    if optimization_result["improvement_found"]:
                        optimization_results["optimizations_performed"] += 1
                        optimization_results["improvements_found"] += 1
                        optimization_results[
                            "total_performance_gain"
                        ] += optimization_result["performance_gain"]
                        optimization_results["optimization_details"].append(
                            optimization_result
                        )

                        # Update learning metadata
                        self.learning_metadata["successful_optimizations"] += 1

                    self.learning_metadata["total_optimizations"] += 1

            self.learning_metadata["last_optimization"] = datetime.utcnow()

            self.logger.info(
                f"🎯 Weight optimization complete: {optimization_results['improvements_found']} improvements found"
            )
            return optimization_results

        except Exception as e:
            self.logger.error(f"❌ Weight optimization failed: {e}")
            return {"error": str(e)}

    def get_optimization_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive analytics on weight optimization performance.

        SPRINT 2: Analytics dashboard for weight optimization insights.
        """
        try:
            analytics = {
                "learning_overview": {
                    "total_optimizations": self.learning_metadata[
                        "total_optimizations"
                    ],
                    "successful_optimizations": self.learning_metadata[
                        "successful_optimizations"
                    ],
                    "success_rate": self.learning_metadata["successful_optimizations"]
                    / max(self.learning_metadata["total_optimizations"], 1),
                    "learning_active": self.learning_metadata["learning_active"],
                    "last_optimization": (
                        self.learning_metadata["last_optimization"].isoformat()
                        if self.learning_metadata["last_optimization"]
                        else None
                    ),
                },
                "profile_performance": {},
                "optimization_trends": {},
                "weight_stability": {},
                "performance_improvements": [],
            }

            # Profile performance analysis
            for profile_id, profile in self.weighting_profiles.items():
                if profile.performance_history:
                    recent_performance = profile.performance_history[-10:]
                    analytics["profile_performance"][profile_id] = {
                        "effectiveness_score": profile.effectiveness_score,
                        "usage_count": profile.usage_count,
                        "recent_average": statistics.mean(recent_performance),
                        "performance_trend": self._calculate_trend(
                            profile.performance_history
                        ),
                        "last_updated": profile.last_updated.isoformat(),
                    }

            # Optimization trends analysis
            recent_optimizations = [
                opt
                for opt in self.optimization_history
                if opt.timestamp >= datetime.utcnow() - timedelta(days=7)
            ]

            optimization_by_dimension = defaultdict(list)
            for opt in recent_optimizations:
                optimization_by_dimension[opt.dimension.value].append(opt)

            for dimension, optimizations in optimization_by_dimension.items():
                performance_gains = [
                    opt.performance_after - opt.performance_before
                    for opt in optimizations
                    if opt.performance_after is not None
                ]
                if performance_gains:
                    analytics["optimization_trends"][dimension] = {
                        "optimization_count": len(optimizations),
                        "average_gain": statistics.mean(performance_gains),
                        "total_gain": sum(performance_gains),
                        "success_rate": len(performance_gains) / len(optimizations),
                    }

            # Weight stability analysis
            for dimension in WeightingDimension:
                recent_weights = []
                for opt in recent_optimizations:
                    if opt.dimension == dimension:
                        recent_weights.extend(opt.new_weights.values())

                if recent_weights and len(recent_weights) > 1:
                    analytics["weight_stability"][dimension.value] = {
                        "weight_variance": statistics.variance(recent_weights),
                        "stability_score": 1.0
                        - min(
                            statistics.variance(recent_weights), 1.0
                        ),  # Higher = more stable
                        "sample_size": len(recent_weights),
                    }

            # Top performance improvements
            significant_improvements = [
                {
                    "dimension": opt.dimension.value,
                    "context": opt.context_category.value,
                    "performance_gain": opt.performance_after - opt.performance_before,
                    "timestamp": opt.timestamp.isoformat(),
                    "method": opt.optimization_method,
                }
                for opt in recent_optimizations
                if opt.performance_after
                and (opt.performance_after - opt.performance_before) > 0.05
            ]

            significant_improvements.sort(
                key=lambda x: x["performance_gain"], reverse=True
            )
            analytics["performance_improvements"] = significant_improvements[:5]

            return analytics

        except Exception as e:
            self.logger.error(f"❌ Failed to generate optimization analytics: {e}")
            return {"error": str(e)}

    def create_custom_weighting_profile(
        self,
        profile_name: str,
        context_category: ContextCategory,
        dimension_weights: Dict[WeightingDimension, Dict[str, float]],
        description: Optional[str] = None,
    ) -> bool:
        """
        Create a custom weighting profile for specific use cases.

        SPRINT 2: Allow customization of weighting profiles for specific scenarios.
        """
        try:
            custom_profile = WeightingProfile(
                profile_id=profile_name,
                context_category=context_category,
                dimension_weights=dimension_weights,
            )

            # Validate all weights
            for dimension, weights in dimension_weights.items():
                validated_weights = self._validate_and_normalize_weights(weights)
                custom_profile.dimension_weights[dimension] = validated_weights

            self.weighting_profiles[profile_name] = custom_profile

            self.logger.info(f"✅ Custom weighting profile created: {profile_name}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to create custom weighting profile: {e}")
            return False

    def _categorize_context(self, context: Dict[str, Any]) -> ContextCategory:
        """Categorize context for appropriate weighting profile selection"""
        complexity = context.get("complexity", "medium")
        s2_tier = context.get("s2_tier", "S2_DISABLED")
        team_size = len(context.get("consultant_types", []))
        urgency = context.get("urgency", "normal")

        # High-stakes decision detection
        if context.get("high_stakes", False) or s2_tier == "S2_TIER_3":
            return ContextCategory.HIGH_STAKES_DECISION

        # Rapid turnaround detection
        if urgency == "urgent" or context.get("time_constraint", False):
            return ContextCategory.RAPID_TURNAROUND

        # Research intensive detection
        if context.get("research_required", False) or context.get(
            "domain_expertise_critical", False
        ):
            return ContextCategory.RESEARCH_INTENSIVE

        # Complexity-based categorization
        if complexity == "high" and s2_tier != "S2_DISABLED":
            return ContextCategory.COMPLEX_ANALYTICAL
        elif team_size >= 3:
            return ContextCategory.COLLABORATIVE_TEAM
        elif complexity == "low":
            return ContextCategory.SIMPLE_TASK
        else:
            return ContextCategory.STANDARD_ANALYSIS

    def _get_or_create_profile(
        self, context_category: ContextCategory
    ) -> WeightingProfile:
        """Get existing profile or create new one for context category"""
        profile_id = self._get_profile_id(context_category)

        if profile_id not in self.weighting_profiles:
            # Create new profile based on closest existing profile
            base_profile = self._find_closest_profile(context_category)
            new_profile = WeightingProfile(
                profile_id=profile_id,
                context_category=context_category,
                dimension_weights=base_profile.dimension_weights.copy(),
            )
            self.weighting_profiles[profile_id] = new_profile
            self.logger.info(f"📋 Created new weighting profile: {profile_id}")

        return self.weighting_profiles[profile_id]

    def _get_profile_id(self, context_category: ContextCategory) -> str:
        """Generate profile ID from context category"""
        return context_category.value

    def _find_closest_profile(
        self, context_category: ContextCategory
    ) -> WeightingProfile:
        """Find the closest existing profile for a new context category"""
        # Simple mapping to existing profiles
        mapping = {
            ContextCategory.HIGH_STAKES_DECISION: "complex_analytical",
            ContextCategory.RAPID_TURNAROUND: "simple_task",
            ContextCategory.RESEARCH_INTENSIVE: "complex_analytical",
        }

        base_profile_id = mapping.get(context_category, "simple_task")
        return self.weighting_profiles[base_profile_id]

    def _apply_contextual_adjustments(
        self,
        base_weights: Dict[str, float],
        context: Dict[str, Any],
        framework_type: Optional[str],
        domain: Optional[str],
        dimension: WeightingDimension,
    ) -> Dict[str, float]:
        """Apply fine-grained contextual adjustments to base weights"""
        adjusted_weights = base_weights.copy()

        # Framework-specific adjustments
        if framework_type and dimension == WeightingDimension.PATTERN_SELECTION:
            if "strategic" in framework_type.lower():
                adjusted_weights["framework_fit"] = min(
                    adjusted_weights.get("framework_fit", 0.2) + 0.1, 0.6
                )
            elif "financial" in framework_type.lower():
                adjusted_weights["domain_match"] = min(
                    adjusted_weights.get("domain_match", 0.3) + 0.1, 0.6
                )

        # Domain-specific adjustments
        if domain and dimension == WeightingDimension.CONSULTANT_SELECTION:
            if "technical" in domain.lower():
                adjusted_weights["expertise_match"] = min(
                    adjusted_weights.get("expertise_match", 0.4) + 0.1, 0.7
                )

        # Complexity adjustments
        complexity = context.get("complexity", "medium")
        if (
            complexity == "high"
            and dimension == WeightingDimension.CHEMISTRY_OPTIMIZATION
        ):
            adjusted_weights["analytical_diversity"] = min(
                adjusted_weights.get("analytical_diversity", 0.3) + 0.2, 0.7
            )

        return adjusted_weights

    def _apply_performance_optimization(
        self,
        base_weights: Dict[str, float],
        dimension: WeightingDimension,
        context_category: ContextCategory,
    ) -> Dict[str, float]:
        """Apply recent performance-based optimization to weights"""
        buffer_key = f"{dimension.value}_{context_category.value}"

        if buffer_key not in self.performance_buffer:
            return base_weights

        recent_data = list(self.performance_buffer[buffer_key])[-20:]  # Last 20 records

        if len(recent_data) < 5:
            return base_weights

        # Calculate weight performance correlations
        optimized_weights = base_weights.copy()

        for weight_name in base_weights.keys():
            # Simple correlation-based adjustment
            weight_performance_correlation = (
                self._calculate_weight_performance_correlation(recent_data, weight_name)
            )

            if abs(weight_performance_correlation) > 0.3:  # Significant correlation
                adjustment = weight_performance_correlation * 0.1  # Max 10% adjustment
                optimized_weights[weight_name] = max(
                    0.01, min(0.99, base_weights[weight_name] + adjustment)
                )

        return optimized_weights

    def _validate_and_normalize_weights(
        self, weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Validate and normalize weights to ensure they sum to appropriate total"""
        if not weights:
            return {}

        # Clamp weights to valid range
        clamped_weights = {
            name: max(
                self.config["weight_bounds"][0],
                min(self.config["weight_bounds"][1], weight),
            )
            for name, weight in weights.items()
        }

        # Normalize to sum to 1.0
        total_weight = sum(clamped_weights.values())
        if total_weight > 0:
            normalized_weights = {
                name: weight / total_weight for name, weight in clamped_weights.items()
            }
        else:
            # Equal weights fallback
            equal_weight = 1.0 / len(clamped_weights)
            normalized_weights = {name: equal_weight for name in clamped_weights.keys()}

        return normalized_weights

    def _record_weight_usage(
        self,
        dimension: WeightingDimension,
        context_category: ContextCategory,
        weights: Dict[str, float],
    ):
        """Record weight usage for learning and analytics"""
        profile_id = self._get_profile_id(context_category)
        if profile_id in self.weighting_profiles:
            self.weighting_profiles[profile_id].usage_count += 1

    def _get_default_weights(self, dimension: WeightingDimension) -> Dict[str, float]:
        """Get safe default weights for a dimension"""
        defaults = {
            WeightingDimension.PATTERN_SELECTION: {
                "domain_match": 0.3,
                "consultant_alignment": 0.25,
                "framework_fit": 0.25,
                "complexity_fit": 0.2,
            },
            WeightingDimension.CONSULTANT_SELECTION: {
                "expertise_match": 0.4,
                "availability": 0.2,
                "diversity_penalty": 0.4,
            },
            WeightingDimension.CHEMISTRY_OPTIMIZATION: {
                "consultant_synergy": 0.33,
                "pattern_alignment": 0.33,
                "analytical_diversity": 0.34,
            },
        }

        return defaults.get(dimension, {"default": 1.0})

    async def _trigger_optimization_if_needed(
        self, dimension: WeightingDimension, context_category: ContextCategory
    ):
        """Trigger optimization if conditions are met"""
        last_optimization = self.learning_metadata.get("last_optimization")

        if not last_optimization or datetime.utcnow() - last_optimization > timedelta(
            minutes=self.config["optimization_frequency_minutes"]
        ):

            buffer_key = f"{dimension.value}_{context_category.value}"
            if (
                buffer_key in self.performance_buffer
                and len(self.performance_buffer[buffer_key])
                >= self.config["min_data_points_for_optimization"]
            ):

                # Trigger background optimization
                asyncio.create_task(self.optimize_weights(dimension, context_category))

    async def _optimize_dimension_weights(
        self,
        dimension: WeightingDimension,
        context_category: ContextCategory,
        performance_data: List[Dict[str, Any]],
        method: str,
    ) -> Dict[str, Any]:
        """Optimize weights for a specific dimension using specified method"""
        try:
            profile_id = self._get_profile_id(context_category)
            current_weights = self.weighting_profiles[profile_id].dimension_weights.get(
                dimension, {}
            )

            if not current_weights:
                return {
                    "improvement_found": False,
                    "reason": "No current weights to optimize",
                }

            # Calculate current performance baseline
            current_performance = statistics.mean(
                [record["performance_score"] for record in performance_data[-10:]]
            )

            # Apply optimization method
            if method == "gradient_descent":
                optimized_weights, performance_gain = (
                    await self._gradient_descent_optimization(
                        current_weights, performance_data
                    )
                )
            elif method == "bayesian":
                optimized_weights, performance_gain = await self._bayesian_optimization(
                    current_weights, performance_data
                )
            else:
                return {
                    "improvement_found": False,
                    "reason": f"Unknown optimization method: {method}",
                }

            # Check for improvement
            if performance_gain > self.config["convergence_threshold"]:
                # Update weights in profile
                self.weighting_profiles[profile_id].dimension_weights[
                    dimension
                ] = optimized_weights

                # Record optimization
                optimization_record = WeightOptimizationRecord(
                    timestamp=datetime.utcnow(),
                    dimension=dimension,
                    context_category=context_category,
                    old_weights=current_weights,
                    new_weights=optimized_weights,
                    performance_before=current_performance,
                    performance_after=current_performance + performance_gain,
                    optimization_method=method,
                    learning_rate=self.config["base_learning_rate"],
                    gradient_magnitude=performance_gain,
                )

                self.optimization_history.append(optimization_record)

                return {
                    "improvement_found": True,
                    "performance_gain": performance_gain,
                    "old_weights": current_weights,
                    "new_weights": optimized_weights,
                    "method": method,
                }
            else:
                return {
                    "improvement_found": False,
                    "reason": f'Performance gain {performance_gain:.4f} below threshold {self.config["convergence_threshold"]}',
                }

        except Exception as e:
            self.logger.error(f"❌ Dimension weight optimization failed: {e}")
            return {"improvement_found": False, "error": str(e)}

    async def _gradient_descent_optimization(
        self, current_weights: Dict[str, float], performance_data: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, float], float]:
        """Gradient descent optimization of weights"""
        # Calculate gradients based on performance correlation
        gradients = {}

        for weight_name in current_weights.keys():
            correlation = self._calculate_weight_performance_correlation(
                performance_data, weight_name
            )
            gradients[weight_name] = correlation * self.config["base_learning_rate"]

        # Apply gradients
        optimized_weights = {}
        for weight_name, current_value in current_weights.items():
            gradient = gradients.get(weight_name, 0.0)
            new_value = current_value + gradient
            optimized_weights[weight_name] = max(
                self.config["weight_bounds"][0],
                min(self.config["weight_bounds"][1], new_value),
            )

        # Normalize weights
        optimized_weights = self._validate_and_normalize_weights(optimized_weights)

        # Estimate performance gain
        estimated_gain = (
            sum(abs(gradients[name]) for name in gradients) * 0.1
        )  # Simplified estimation

        return optimized_weights, estimated_gain

    async def _bayesian_optimization(
        self, current_weights: Dict[str, float], performance_data: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, float], float]:
        """Bayesian optimization of weights (simplified implementation)"""
        # For this implementation, we'll use a simplified approach
        # In production, this would use proper Bayesian optimization libraries

        best_weights = current_weights.copy()
        best_performance = statistics.mean(
            [record["performance_score"] for record in performance_data[-5:]]
        )

        # Try small perturbations and estimate improvements
        for weight_name in current_weights.keys():
            for delta in [-0.05, 0.05]:  # Try small adjustments
                test_weights = current_weights.copy()
                test_weights[weight_name] = max(
                    self.config["weight_bounds"][0],
                    min(
                        self.config["weight_bounds"][1],
                        current_weights[weight_name] + delta,
                    ),
                )

                # Normalize test weights
                test_weights = self._validate_and_normalize_weights(test_weights)

                # Estimate performance (simplified)
                correlation = self._calculate_weight_performance_correlation(
                    performance_data, weight_name
                )
                estimated_performance = best_performance + (delta * correlation * 0.5)

                if estimated_performance > best_performance:
                    best_weights = test_weights
                    best_performance = estimated_performance

        performance_gain = best_performance - statistics.mean(
            [record["performance_score"] for record in performance_data[-5:]]
        )

        return best_weights, max(0.0, performance_gain)

    def _calculate_weight_performance_correlation(
        self, performance_data: List[Dict[str, Any]], weight_name: str
    ) -> float:
        """Calculate correlation between weight values and performance"""
        try:
            weight_values = []
            performance_scores = []

            for record in performance_data:
                if weight_name in record["weights_used"]:
                    weight_values.append(record["weights_used"][weight_name])
                    performance_scores.append(record["performance_score"])

            if len(weight_values) < 3:
                return 0.0

            # Simple correlation calculation
            mean_weight = statistics.mean(weight_values)
            mean_performance = statistics.mean(performance_scores)

            numerator = sum(
                (w - mean_weight) * (p - mean_performance)
                for w, p in zip(weight_values, performance_scores)
            )
            denominator_w = sum((w - mean_weight) ** 2 for w in weight_values)
            denominator_p = sum((p - mean_performance) ** 2 for p in performance_scores)

            if denominator_w == 0 or denominator_p == 0:
                return 0.0

            correlation = numerator / (math.sqrt(denominator_w * denominator_p))
            return max(-1.0, min(1.0, correlation))

        except Exception:
            return 0.0

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from list of values"""
        if len(values) < 3:
            return "insufficient_data"

        # Compare first third vs last third
        first_third = values[: len(values) // 3]
        last_third = values[-len(values) // 3 :]

        first_avg = statistics.mean(first_third)
        last_avg = statistics.mean(last_third)

        if last_avg > first_avg * 1.05:
            return "improving"
        elif last_avg < first_avg * 0.95:
            return "declining"
        else:
            return "stable"

    def _hash_context(self, context: Dict[str, Any]) -> str:
        """Create a hash for context to detect similar contexts"""
        import hashlib

        # Extract key context features for hashing
        key_features = {
            "complexity": context.get("complexity", "medium"),
            "s2_tier": context.get("s2_tier", "S2_DISABLED"),
            "team_size": len(context.get("consultant_types", [])),
            "urgency": context.get("urgency", "normal"),
        }

        context_string = json.dumps(key_features, sort_keys=True)
        return hashlib.md5(context_string.encode()).hexdigest()[:12]

    async def get_service_health(self) -> Dict[str, Any]:
        """Return service health and status"""
        return {
            "service_name": "ContextualWeightingSystem",
            "status": "healthy",
            "version": "sprint_2",
            "capabilities": [
                "contextual_weight_adaptation",
                "machine_learning_optimization",
                "performance_feedback_loops",
                "multi_dimensional_weighting",
                "gradient_descent_optimization",
                "bayesian_optimization",
                "custom_profile_creation",
            ],
            "system_statistics": {
                "weighting_profiles": len(self.weighting_profiles),
                "optimization_history": len(self.optimization_history),
                "total_optimizations": self.learning_metadata["total_optimizations"],
                "successful_optimizations": self.learning_metadata[
                    "successful_optimizations"
                ],
                "learning_active": self.learning_metadata["learning_active"],
            },
            "configuration": self.config,
            "sprint_2_targets": {
                "dynamic_adaptation": "active",
                "learning_optimization": "active",
                "performance_feedback": "integrated",
            },
            "last_health_check": datetime.utcnow().isoformat(),
        }


# Global service instance for dependency injection
_contextual_weighting_service: Optional[ContextualWeightingSystem] = None


def get_contextual_weighting_service() -> ContextualWeightingSystem:
    """Get or create global contextual weighting service instance"""
    global _contextual_weighting_service

    if _contextual_weighting_service is None:
        _contextual_weighting_service = ContextualWeightingSystem()

    return _contextual_weighting_service
