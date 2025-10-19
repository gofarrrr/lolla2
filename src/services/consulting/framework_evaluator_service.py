"""
Framework Evaluator Service - Production Implementation
C2 - Service Extraction & Batching (Red Team Amendment Applied)
"""
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum

from .contracts import (
    IFrameworkEvaluator, PerformanceConfig, EvaluationResult,
    PerformanceThresholdExceeded, EvaluationTimeoutError
)
from .framework_repository_service import FrameworkCategory, FrameworkComplexity
from src.core.async_helpers import timeout, bounded
from src.core.unified_context_stream import UnifiedContextStream, ContextEventType

logger = logging.getLogger(__name__)

class EngagementPhase(str, Enum):
    """Engagement phases for framework evaluation"""
    PROBLEM_STRUCTURING = "problem_structuring"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    ANALYSIS_EXECUTION = "analysis_execution"
    SYNTHESIS_DELIVERY = "synthesis_delivery"

class ProductionFrameworkEvaluatorService:
    """
    Production Framework Evaluator with Vectorized Scoring
    Red Team Amendment: Batch evaluation and performance optimization
    """
    
    def __init__(
        self,
        config: PerformanceConfig,
        context_stream: Optional[UnifiedContextStream] = None
    ):
        self.config = config
        self.context_stream = context_stream
        self.logger = logger
        
        # Performance controls (Red Team Amendment)
        self.evaluation_semaphore = asyncio.Semaphore(config.max_concurrent_operations)
        
        # Evaluation cache for repeated contexts
        self.evaluation_cache: Dict[str, EvaluationResult] = {}
        
        # Performance metrics
        self.metrics = {
            "evaluations_performed": 0,
            "batch_evaluations": 0,
            "cache_hits": 0,
            "average_evaluation_time": 0.0,
            "threshold_violations": 0
        }
        
        # Phase-based framework preferences (Red Team Amendment: Pre-computed)
        self.phase_preferences = {
            EngagementPhase.PROBLEM_STRUCTURING: {
                FrameworkCategory.ANALYTICAL: 0.8,
                FrameworkCategory.DIAGNOSTIC: 0.7,
                FrameworkCategory.STRATEGIC: 0.4,
            },
            EngagementPhase.HYPOTHESIS_GENERATION: {
                FrameworkCategory.ANALYTICAL: 0.6,
                FrameworkCategory.STRATEGIC: 0.8,
                FrameworkCategory.MARKET: 0.6,
            },
            EngagementPhase.ANALYSIS_EXECUTION: {
                FrameworkCategory.OPERATIONAL: 0.8,
                FrameworkCategory.FINANCIAL: 0.7,
                FrameworkCategory.MARKET: 0.7,
            },
            EngagementPhase.SYNTHESIS_DELIVERY: {
                FrameworkCategory.STRATEGIC: 0.9,
                FrameworkCategory.ORGANIZATIONAL: 0.6,
            },
        }
        
        # Pre-computed relevance patterns for performance
        self.relevance_patterns = {
            FrameworkCategory.STRATEGIC: [
                "strategy", "growth", "market", "competitive", "portfolio",
            ],
            FrameworkCategory.ANALYTICAL: [
                "problem", "analysis", "structure", "breakdown", "diagnostic",
            ],
            FrameworkCategory.OPERATIONAL: [
                "process", "efficiency", "operations", "cost", "productivity",
            ],
            FrameworkCategory.FINANCIAL: [
                "revenue", "profit", "cost", "financial", "pricing",
            ],
            FrameworkCategory.MARKET: [
                "customer", "market", "competition", "brand", "positioning",
            ],
            FrameworkCategory.ORGANIZATIONAL: [
                "organization", "people", "culture", "change", "skills",
            ],
        }
        
        # Complexity matching matrix (Red Team Amendment: Pre-computed)
        self.complexity_match_matrix = {
            (FrameworkComplexity.BASIC, FrameworkComplexity.BASIC): 1.0,
            (FrameworkComplexity.BASIC, FrameworkComplexity.INTERMEDIATE): 0.7,
            (FrameworkComplexity.INTERMEDIATE, FrameworkComplexity.INTERMEDIATE): 1.0,
            (FrameworkComplexity.INTERMEDIATE, FrameworkComplexity.ADVANCED): 0.8,
            (FrameworkComplexity.ADVANCED, FrameworkComplexity.ADVANCED): 1.0,
            (FrameworkComplexity.ADVANCED, FrameworkComplexity.EXPERT): 0.9,
        }
    
    async def score_frameworks(
        self,
        context: str,
        frameworks: List[Dict[str, Any]],
        **kwargs
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Score frameworks with vectorization support (Red Team Amendment)"""
        operation_start = time.time()
        
        try:
            # Extract scoring parameters
            phase = kwargs.get('phase', EngagementPhase.PROBLEM_STRUCTURING)
            engagement_context = kwargs.get('engagement_context', {})
            
            # Check cache for repeated evaluations
            cache_key = self._generate_cache_key(context, [fw.get('framework_id', '') for fw in frameworks], phase)
            
            if cache_key in self.evaluation_cache:
                self.metrics["cache_hits"] += 1
                cached_result = self.evaluation_cache[cache_key]
                return [(fw, cached_result.score) for fw in frameworks]
            
            # Vectorized scoring with bounded concurrency
            async with self.evaluation_semaphore:
                scored_frameworks = await self._vectorized_framework_scoring(
                    context, frameworks, phase, engagement_context
                )
            
            processing_time = (time.time() - operation_start) * 1000
            self.metrics["evaluations_performed"] += 1
            self._update_average_evaluation_time(processing_time)
            
            # Performance validation (Red Team Amendment)
            if processing_time > self.config.p95_threshold_ms:
                self.metrics["threshold_violations"] += 1
                self.logger.warning(f"Framework scoring exceeded p95 threshold: {processing_time:.2f}ms")
                
                if processing_time > self.config.p95_threshold_ms * 1.5:
                    raise PerformanceThresholdExceeded(
                        f"Framework scoring time {processing_time:.2f}ms exceeded 150% of p95 threshold"
                    )
            
            # Cache results for future use
            if len(scored_frameworks) > 0:
                avg_score = sum(score for _, score in scored_frameworks) / len(scored_frameworks)
                avg_confidence = 0.85  # Default confidence for batch operations
                
                self.evaluation_cache[cache_key] = EvaluationResult(
                    framework={"batch_key": cache_key},
                    score=avg_score,
                    confidence=avg_confidence,
                    processing_time_ms=processing_time,
                    cache_hit=False
                )
            
            await self._emit_performance_event(
                "FRAMEWORK_SCORING_COMPLETE",
                {
                    "framework_count": len(frameworks),
                    "processing_time_ms": processing_time,
                    "phase": phase.value if hasattr(phase, 'value') else str(phase)
                }
            )
            
            return scored_frameworks
            
        except asyncio.TimeoutError:
            raise EvaluationTimeoutError("Framework scoring timeout")
        except Exception as e:
            self.logger.error(f"Framework scoring failed: {e}")
            # Return frameworks with default scores for resilience
            return [(fw, 0.5) for fw in frameworks]
    
    async def batch_evaluate(
        self,
        contexts: List[str],
        frameworks: List[Dict[str, Any]]
    ) -> Dict[str, List[Tuple[Dict[str, Any], float]]]:
        """Batch evaluation for performance (Red Team Amendment)"""
        operation_start = time.time()
        self.metrics["batch_evaluations"] += 1
        
        try:
            # Bounded concurrency for batch operations
            semaphore = asyncio.Semaphore(min(len(contexts), self.config.max_concurrent_operations))
            
            async def evaluate_single_context(context: str) -> Tuple[str, List[Tuple[Dict[str, Any], float]]]:
                async with semaphore:
                    scored_frameworks = await self.score_frameworks(context, frameworks)
                    return context, scored_frameworks
            
            # Execute batch evaluation with timeout
            tasks = [evaluate_single_context(ctx) for ctx in contexts]
            results = await timeout(
                asyncio.gather(*tasks, return_exceptions=True),
                seconds=self.config.p95_threshold_ms * len(contexts) / 1000.0
            )
            
            # Process results
            batch_results = {}
            for result in results:
                if isinstance(result, Exception):
                    self.logger.warning(f"Batch evaluation failed for context: {result}")
                    continue
                
                context, scored_frameworks = result
                batch_results[context] = scored_frameworks
            
            processing_time = (time.time() - operation_start) * 1000
            
            await self._emit_performance_event(
                "FRAMEWORK_BATCH_EVALUATION_COMPLETE",
                {
                    "context_count": len(contexts),
                    "framework_count": len(frameworks),
                    "completed_count": len(batch_results),
                    "processing_time_ms": processing_time
                }
            )
            
            return batch_results
            
        except asyncio.TimeoutError:
            raise EvaluationTimeoutError(f"Batch evaluation timeout for {len(contexts)} contexts")
        except Exception as e:
            self.logger.error(f"Batch evaluation failed: {e}")
            return {}
    
    async def filter_by_criteria(
        self,
        frameworks: List[Dict[str, Any]],
        criteria: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Filter frameworks by business criteria (Red Team Amendment)"""
        operation_start = time.time()
        
        try:
            filtered_frameworks = []
            
            for framework in frameworks:
                if await self._meets_criteria(framework, criteria):
                    filtered_frameworks.append(framework)
            
            processing_time = (time.time() - operation_start) * 1000
            
            await self._emit_performance_event(
                "FRAMEWORK_FILTERING_COMPLETE",
                {
                    "input_count": len(frameworks),
                    "filtered_count": len(filtered_frameworks),
                    "processing_time_ms": processing_time
                }
            )
            
            return filtered_frameworks
            
        except Exception as e:
            self.logger.error(f"Framework filtering failed: {e}")
            return frameworks  # Return unfiltered for resilience
    
    async def _vectorized_framework_scoring(
        self,
        context: str,
        frameworks: List[Dict[str, Any]],
        phase: EngagementPhase,
        engagement_context: Dict[str, Any]
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Vectorized framework scoring for performance"""
        
        # Parse context once for all frameworks
        context_lower = context.lower()
        business_context = str(engagement_context.get('business_context', '')).lower()
        problem_statement = engagement_context.get('problem_statement', context)
        
        # Estimate engagement complexity once
        problem_length = len(problem_statement.split())
        context_complexity = len(business_context)
        needed_complexity = self._estimate_needed_complexity(problem_length, context_complexity)
        
        scored_frameworks = []
        
        for framework in frameworks:
            # Extract framework metadata
            framework_id = framework.get('framework_id', '')
            category = self._parse_category(framework.get('category', ''))
            complexity = self._parse_complexity(framework.get('complexity', ''))
            base_score = framework.get('applicability_score', 5.0)
            
            # Calculate component scores
            phase_score = self.phase_preferences.get(phase, {}).get(category, 0.3)
            context_score = self._calculate_context_relevance(category, context_lower, business_context)
            complexity_score = self._calculate_complexity_fit(needed_complexity, complexity)
            
            # Combined score with weights
            total_score = (
                base_score * 0.3 +
                phase_score * 0.3 +
                context_score * 0.2 +
                complexity_score * 0.2
            )
            
            scored_frameworks.append((framework, total_score))
        
        # Sort by score
        scored_frameworks.sort(key=lambda x: x[1], reverse=True)
        
        return scored_frameworks
    
    def _calculate_context_relevance(self, category: FrameworkCategory, context: str, business_context: str) -> float:
        """Calculate context relevance score"""
        patterns = self.relevance_patterns.get(category, [])
        
        if not patterns:
            return 0.5
        
        relevance_score = sum(
            1 for pattern in patterns
            if pattern in context or pattern in business_context
        )
        
        return min(1.0, relevance_score / len(patterns))
    
    def _calculate_complexity_fit(self, needed_complexity: FrameworkComplexity, framework_complexity: FrameworkComplexity) -> float:
        """Calculate complexity fit score"""
        return self.complexity_match_matrix.get((needed_complexity, framework_complexity), 0.5)
    
    def _estimate_needed_complexity(self, problem_length: int, context_complexity: int) -> FrameworkComplexity:
        """Estimate needed complexity based on problem characteristics"""
        if problem_length > 50 or context_complexity > 500:
            return FrameworkComplexity.ADVANCED
        elif problem_length > 20 or context_complexity > 200:
            return FrameworkComplexity.INTERMEDIATE
        else:
            return FrameworkComplexity.BASIC
    
    def _parse_category(self, category_str: str) -> FrameworkCategory:
        """Parse category string to enum"""
        try:
            return FrameworkCategory(category_str.lower())
        except ValueError:
            return FrameworkCategory.ANALYTICAL  # Default fallback
    
    def _parse_complexity(self, complexity_str: str) -> FrameworkComplexity:
        """Parse complexity string to enum"""
        try:
            return FrameworkComplexity(complexity_str.lower())
        except ValueError:
            return FrameworkComplexity.INTERMEDIATE  # Default fallback
    
    async def _meets_criteria(self, framework: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
        """Check if framework meets business criteria"""
        # Category filter
        if 'categories' in criteria:
            framework_category = framework.get('category', '')
            if framework_category not in criteria['categories']:
                return False
        
        # Complexity filter
        if 'max_complexity' in criteria:
            framework_complexity = framework.get('complexity', '')
            complexity_levels = ['basic', 'intermediate', 'advanced', 'expert']
            
            try:
                fw_idx = complexity_levels.index(framework_complexity.lower())
                max_idx = complexity_levels.index(criteria['max_complexity'].lower())
                if fw_idx > max_idx:
                    return False
            except ValueError:
                pass
        
        # Minimum score filter
        if 'min_score' in criteria:
            framework_score = framework.get('applicability_score', 0.0)
            if framework_score < criteria['min_score']:
                return False
        
        # Usage count filter (for popular frameworks)
        if 'min_usage' in criteria:
            usage_count = framework.get('usage_count', 0)
            if usage_count < criteria['min_usage']:
                return False
        
        return True
    
    def _generate_cache_key(self, context: str, framework_ids: List[str], phase: EngagementPhase) -> str:
        """Generate cache key for evaluation results"""
        context_hash = hash(context[:100])  # First 100 chars for cache key
        frameworks_hash = hash(tuple(sorted(framework_ids)))
        phase_hash = hash(phase.value if hasattr(phase, 'value') else str(phase))
        
        return f"eval:{context_hash}:{frameworks_hash}:{phase_hash}"
    
    def _update_average_evaluation_time(self, processing_time: float):
        """Update average evaluation time metric"""
        if self.metrics["evaluations_performed"] == 1:
            self.metrics["average_evaluation_time"] = processing_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics["average_evaluation_time"] = (
                alpha * processing_time + 
                (1 - alpha) * self.metrics["average_evaluation_time"]
            )
    
    async def _emit_performance_event(self, event_type: str, details: Dict[str, Any]):
        """Emit performance events for observability"""
        if self.context_stream:
            try:
                await self.context_stream.emit_event(
                    event_type=ContextEventType.PERFORMANCE_METRIC,
                    details={
                        "operation": event_type,
                        "service": "framework_evaluator",
                        **details
                    }
                )
            except Exception as e:
                self.logger.warning(f"Failed to emit performance event: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get evaluator performance metrics"""
        cache_hit_rate = (
            self.metrics["cache_hits"] / max(1, self.metrics["evaluations_performed"])
        )
        
        return {
            **self.metrics,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.evaluation_cache),
            "performance_threshold_violation_rate": (
                self.metrics["threshold_violations"] / max(1, self.metrics["evaluations_performed"])
            ),
            "config": {
                "p50_threshold_ms": self.config.p50_threshold_ms,
                "p95_threshold_ms": self.config.p95_threshold_ms,
                "max_concurrent_operations": self.config.max_concurrent_operations
            }
        }
    
    def clear_cache(self):
        """Clear evaluation cache"""
        self.evaluation_cache.clear()
        self.logger.info("Framework evaluation cache cleared")

class FrameworkEvaluatorServiceFactory:
    """Factory for creating framework evaluator services"""
    
    @staticmethod
    def create_production_service(
        config: PerformanceConfig,
        context_stream: Optional[UnifiedContextStream] = None
    ) -> IFrameworkEvaluator:
        """Create production framework evaluator service"""
        return ProductionFrameworkEvaluatorService(config, context_stream)
    
    @staticmethod
    def create_from_env() -> IFrameworkEvaluator:
        """Create framework evaluator from environment variables"""
        import os
        
        config = PerformanceConfig(
            p50_threshold_ms=float(os.getenv("FRAMEWORK_P50_THRESHOLD_MS", "200.0")),
            p95_threshold_ms=float(os.getenv("FRAMEWORK_P95_THRESHOLD_MS", "400.0")),
            max_concurrent_operations=int(os.getenv("FRAMEWORK_MAX_CONCURRENT", "10")),
            cache_ttl_minutes=int(os.getenv("FRAMEWORK_CACHE_TTL_MIN", "10")),
            warmup_enabled=os.getenv("FRAMEWORK_WARMUP_ENABLED", "true").lower() == "true"
        )
        
        return FrameworkEvaluatorServiceFactory.create_production_service(config)