"""
Foundation Analytics Service
===========================

Operation Chimera Phase 3 - Foundation Service Extraction

Analytics service implementing metrics, performance tracking, and reporting 
for the Foundation API. Extracted from enhanced_foundation.py to separate 
analytics concerns from routing logic.

Key Responsibilities:
- Processing time and performance metrics
- Model effectiveness scoring and analysis
- Health status determination and monitoring
- Engagement analytics and reporting
- Confidence score calculations
"""

import time
import statistics
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime, timedelta

from .foundation_contracts import (
    IFoundationAnalyticsService,
    DatabaseHealthResponse,
    FoundationServiceError,
)
from src.core.unified_context_stream import UnifiedContextStream


class FoundationAnalyticsService(IFoundationAnalyticsService):
    """
    Foundation Analytics Service Implementation
    
    Provides comprehensive analytics, metrics calculation, and performance 
    tracking for all Foundation API operations.
    """
    
    def __init__(self, context_stream: Optional[UnifiedContextStream] = None):
        """Initialize Foundation Analytics Service"""
        from src.core.unified_context_stream import get_unified_context_stream, UnifiedContextStream
        self.context_stream = context_stream or get_unified_context_stream()
        self._performance_cache = {}
        self._health_history = []
    
    async def calculate_processing_metrics(
        self,
        start_time: float,
        end_time: float,
        operation_type: str
    ) -> Dict[str, Any]:
        """
        Calculate processing time and performance metrics
        
        Analyzes:
        - Processing duration in milliseconds
        - Performance classification (fast/medium/slow)
        - Historical comparison metrics
        - Resource utilization estimates
        """
        await self.context_stream.log_event(
            "FOUNDATION_PROCESSING_METRICS_CALCULATION_STARTED",
            {
                "operation_type": operation_type,
                "start_time": start_time,
                "end_time": end_time,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        try:
            processing_time_ms = (end_time - start_time) * 1000
            
            # Calculate performance classification
            performance_class = self._classify_performance(processing_time_ms, operation_type)
            
            # Get historical comparison
            historical_metrics = await self._get_historical_performance(operation_type)
            
            # Calculate percentile ranking
            percentile_rank = await self._calculate_percentile_rank(
                processing_time_ms, operation_type
            )
            
            # Estimate resource utilization
            resource_estimate = await self._estimate_resource_utilization(
                processing_time_ms, operation_type
            )
            
            metrics = {
                "processing_time_ms": round(processing_time_ms, 2),
                "performance_class": performance_class,
                "percentile_rank": percentile_rank,
                "historical_comparison": historical_metrics,
                "resource_utilization": resource_estimate,
                "operation_type": operation_type,
                "calculated_at": datetime.now().isoformat(),
                "efficiency_score": self._calculate_efficiency_score(
                    processing_time_ms, operation_type
                )
            }
            
            # Cache for future comparisons
            await self._cache_performance_metric(operation_type, processing_time_ms)
            
            await self.context_stream.log_event(
                "FOUNDATION_PROCESSING_METRICS_CALCULATION_COMPLETED",
                {
                    "operation_type": operation_type,
                    "processing_time_ms": processing_time_ms,
                    "performance_class": performance_class,
                    "efficiency_score": metrics["efficiency_score"]
                }
            )
            
            return metrics
        
        except Exception as e:
            await self.context_stream.log_event(
                "FOUNDATION_PROCESSING_METRICS_CALCULATION_ERROR",
                {
                    "error": str(e),
                    "error_type": e.__class__.__name__,
                    "operation_type": operation_type
                }
            )
            raise FoundationServiceError(
                f"Processing metrics calculation failed: {str(e)}",
                code="PROCESSING_METRICS_ERROR",
                details={"operation_type": operation_type}
            )
    
    async def calculate_model_effectiveness_scores(
        self,
        models: List[Dict[str, Any]],
        problem_context: str
    ) -> Dict[str, Any]:
        """
        Calculate effectiveness scores for mental models
        
        Analyzes:
        - Individual model effectiveness scores
        - Overall average effectiveness
        - Model category distribution
        - Enhanced model count and quality
        """
        await self.context_stream.log_event(
            "FOUNDATION_MODEL_EFFECTIVENESS_CALCULATION_STARTED",
            {
                "models_count": len(models),
                "problem_context_length": len(problem_context),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        try:
            effectiveness_scores = []
            enhanced_models_count = 0
            categories = set()
            
            # Calculate individual model scores
            for model in models:
                # Extract or calculate effectiveness score
                if "effectiveness_score" in model:
                    score = model["effectiveness_score"]
                else:
                    score = await self._calculate_model_effectiveness(model, problem_context)
                    model["effectiveness_score"] = score
                
                effectiveness_scores.append(score)
                
                # Count enhanced models (score > 0.7)
                if score > 0.7:
                    enhanced_models_count += 1
                
                # Collect categories
                if "category" in model:
                    categories.add(model["category"])
                elif "type" in model:
                    categories.add(model["type"])
            
            # Calculate aggregate metrics
            avg_effectiveness = sum(effectiveness_scores) / len(effectiveness_scores) if effectiveness_scores else 0
            
            # Calculate distribution metrics
            distribution_metrics = await self._calculate_score_distribution(effectiveness_scores)
            
            # Calculate quality assessment
            quality_assessment = await self._assess_model_quality(
                effectiveness_scores, enhanced_models_count, len(models)
            )
            
            result = {
                "avg_effectiveness_score": round(avg_effectiveness, 3),
                "total_models": len(models),
                "enhanced_models_count": enhanced_models_count,
                "categories": list(categories),
                "distribution_metrics": distribution_metrics,
                "quality_assessment": quality_assessment,
                "individual_scores": effectiveness_scores,
                "score_range": {
                    "min": min(effectiveness_scores) if effectiveness_scores else 0,
                    "max": max(effectiveness_scores) if effectiveness_scores else 0
                },
                "calculated_at": datetime.now().isoformat()
            }
            
            await self.context_stream.log_event(
                "FOUNDATION_MODEL_EFFECTIVENESS_CALCULATION_COMPLETED",
                {
                    "avg_effectiveness_score": avg_effectiveness,
                    "enhanced_models_count": enhanced_models_count,
                    "total_models": len(models),
                    "quality_grade": quality_assessment["grade"]
                }
            )
            
            return result
        
        except Exception as e:
            await self.context_stream.log_event(
                "FOUNDATION_MODEL_EFFECTIVENESS_CALCULATION_ERROR",
                {
                    "error": str(e),
                    "error_type": e.__class__.__name__,
                    "models_count": len(models)
                }
            )
            raise FoundationServiceError(
                f"Model effectiveness calculation failed: {str(e)}",
                code="MODEL_EFFECTIVENESS_ERROR",
                details={"models_count": len(models)}
            )
    
    async def determine_engagement_health(
        self,
        engagement_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Determine health status of engagement
        
        Analyzes:
        - Engagement completeness
        - Data quality indicators
        - Processing success rates
        - User interaction patterns
        """
        await self.context_stream.log_event(
            "FOUNDATION_ENGAGEMENT_HEALTH_DETERMINATION_STARTED",
            {
                "engagement_id": engagement_data.get("id"),
                "status": engagement_data.get("status"),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        try:
            health_indicators = {}
            
            # Data completeness assessment
            completeness_score = await self._assess_data_completeness(engagement_data)
            health_indicators["data_completeness"] = completeness_score
            
            # Quality indicators
            quality_score = await self._assess_data_quality(engagement_data)
            health_indicators["data_quality"] = quality_score
            
            # Activity level assessment
            activity_score = await self._assess_activity_level(engagement_data)
            health_indicators["activity_level"] = activity_score
            
            # Processing success assessment
            processing_score = await self._assess_processing_success(engagement_data)
            health_indicators["processing_success"] = processing_score
            
            # Calculate overall health score
            overall_score = statistics.mean([
                completeness_score,
                quality_score,
                activity_score,
                processing_score
            ])
            
            # Determine health status
            health_status = self._determine_health_status(overall_score)
            
            # Generate health recommendations
            recommendations = await self._generate_health_recommendations(health_indicators)
            
            health_result = {
                "overall_health_score": round(overall_score, 3),
                "health_status": health_status,
                "health_indicators": health_indicators,
                "recommendations": recommendations,
                "assessment_timestamp": datetime.now().isoformat(),
                "risk_factors": await self._identify_risk_factors(health_indicators),
                "improvement_opportunities": await self._identify_improvements(health_indicators)
            }
            
            await self.context_stream.log_event(
                "FOUNDATION_ENGAGEMENT_HEALTH_DETERMINATION_COMPLETED",
                {
                    "engagement_id": engagement_data.get("id"),
                    "overall_health_score": overall_score,
                    "health_status": health_status,
                    "risk_factors_count": len(health_result["risk_factors"])
                }
            )
            
            return health_result
        
        except Exception as e:
            await self.context_stream.log_event(
                "FOUNDATION_ENGAGEMENT_HEALTH_DETERMINATION_ERROR",
                {
                    "error": str(e),
                    "error_type": e.__class__.__name__,
                    "engagement_id": engagement_data.get("id")
                }
            )
            raise FoundationServiceError(
                f"Engagement health determination failed: {str(e)}",
                code="ENGAGEMENT_HEALTH_ERROR",
                details={"engagement_id": engagement_data.get("id")}
            )
    
    async def generate_engagement_analytics(
        self,
        engagement_id: UUID,
        analysis_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive analytics for engagement
        
        Includes:
        - Confidence score analysis
        - Pattern detection metrics
        - Transparency layer analytics
        - Processing performance summary
        """
        await self.context_stream.log_event(
            "FOUNDATION_ENGAGEMENT_ANALYTICS_GENERATION_STARTED",
            {
                "engagement_id": str(engagement_id),
                "analysis_keys": list(analysis_data.keys()),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        try:
            analytics = {}
            
            # Confidence score analytics
            if "confidence_scores" in analysis_data:
                confidence_analytics = await self._analyze_confidence_scores(
                    analysis_data["confidence_scores"]
                )
                analytics["confidence_analytics"] = confidence_analytics
            
            # Pattern detection analytics
            if "nway_patterns" in analysis_data:
                pattern_analytics = await self._analyze_nway_patterns(
                    analysis_data["nway_patterns"]
                )
                analytics["pattern_analytics"] = pattern_analytics
            
            # Model selection analytics
            if "selected_models" in analysis_data:
                model_analytics = await self._analyze_model_selection(
                    analysis_data["selected_models"]
                )
                analytics["model_analytics"] = model_analytics
            
            # Reasoning quality analytics
            if "reasoning_steps" in analysis_data:
                reasoning_analytics = await self._analyze_reasoning_quality(
                    analysis_data["reasoning_steps"]
                )
                analytics["reasoning_analytics"] = reasoning_analytics
            
            # Overall engagement score
            overall_score = await self._calculate_overall_engagement_score(analytics)
            
            # Generate insights
            insights = await self._generate_analytics_insights(analytics)
            
            analytics_result = {
                "engagement_id": str(engagement_id),
                "overall_score": overall_score,
                "detailed_analytics": analytics,
                "insights": insights,
                "generated_at": datetime.now().isoformat(),
                "analytics_version": "v1.0"
            }
            
            await self.context_stream.log_event(
                "FOUNDATION_ENGAGEMENT_ANALYTICS_GENERATION_COMPLETED",
                {
                    "engagement_id": str(engagement_id),
                    "overall_score": overall_score,
                    "analytics_components": len(analytics),
                    "insights_count": len(insights)
                }
            )
            
            return analytics_result
        
        except Exception as e:
            await self.context_stream.log_event(
                "FOUNDATION_ENGAGEMENT_ANALYTICS_GENERATION_ERROR",
                {
                    "error": str(e),
                    "error_type": e.__class__.__name__,
                    "engagement_id": str(engagement_id)
                }
            )
            raise FoundationServiceError(
                f"Engagement analytics generation failed: {str(e)}",
                code="ENGAGEMENT_ANALYTICS_ERROR",
                details={"engagement_id": str(engagement_id)}
            )
    
    async def calculate_system_health_metrics(
        self,
        database_health: Dict[str, Any],
        performance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate overall system health metrics
        
        Combines:
        - Database health indicators
        - Performance metrics
        - Service availability
        - Resource utilization
        """
        await self.context_stream.log_event(
            "FOUNDATION_SYSTEM_HEALTH_CALCULATION_STARTED",
            {
                "database_status": database_health.get("status"),
                "performance_metrics_count": len(performance_data),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        try:
            # Database health scoring
            db_health_score = await self._score_database_health(database_health)
            
            # Performance health scoring
            performance_health_score = await self._score_performance_health(performance_data)
            
            # Service availability scoring
            availability_score = await self._calculate_availability_score()
            
            # Resource utilization scoring
            resource_score = await self._calculate_resource_utilization_score()
            
            # Calculate overall system health
            overall_health = statistics.mean([
                db_health_score,
                performance_health_score,
                availability_score,
                resource_score
            ])
            
            # Determine system status
            system_status = self._determine_system_status(overall_health)
            
            # Generate alerts if needed
            alerts = await self._generate_health_alerts(
                db_health_score, performance_health_score, availability_score, resource_score
            )
            
            # Calculate trends
            health_trends = await self._calculate_health_trends()
            
            health_metrics = {
                "overall_health_score": round(overall_health, 3),
                "system_status": system_status,
                "component_scores": {
                    "database_health": db_health_score,
                    "performance_health": performance_health_score,
                    "availability": availability_score,
                    "resource_utilization": resource_score
                },
                "alerts": alerts,
                "trends": health_trends,
                "calculated_at": datetime.now().isoformat(),
                "next_assessment": (datetime.now() + timedelta(minutes=5)).isoformat()
            }
            
            # Store for trend analysis
            await self._store_health_snapshot(health_metrics)
            
            await self.context_stream.log_event(
                "FOUNDATION_SYSTEM_HEALTH_CALCULATION_COMPLETED",
                {
                    "overall_health_score": overall_health,
                    "system_status": system_status,
                    "alerts_count": len(alerts),
                    "components_assessed": 4
                }
            )
            
            return health_metrics
        
        except Exception as e:
            await self.context_stream.log_event(
                "FOUNDATION_SYSTEM_HEALTH_CALCULATION_ERROR",
                {
                    "error": str(e),
                    "error_type": e.__class__.__name__,
                    "database_status": database_health.get("status")
                }
            )
            raise FoundationServiceError(
                f"System health calculation failed: {str(e)}",
                code="SYSTEM_HEALTH_ERROR",
                details={"database_health": database_health}
            )
    
    # Private analytics helper methods
    
    def _classify_performance(self, processing_time_ms: float, operation_type: str) -> str:
        """Classify performance as fast/medium/slow based on operation type"""
        thresholds = {
            "engagement_create": {"fast": 500, "medium": 2000},
            "cognitive_analysis": {"fast": 3000, "medium": 10000},
            "model_listing": {"fast": 200, "medium": 1000},
            "health_check": {"fast": 100, "medium": 500},
            "default": {"fast": 1000, "medium": 5000}
        }
        
        threshold = thresholds.get(operation_type, thresholds["default"])
        
        if processing_time_ms <= threshold["fast"]:
            return "fast"
        elif processing_time_ms <= threshold["medium"]:
            return "medium"
        else:
            return "slow"
    
    async def _get_historical_performance(self, operation_type: str) -> Dict[str, Any]:
        """Get historical performance comparison for operation type"""
        cache_key = f"historical_{operation_type}"
        
        if cache_key not in self._performance_cache:
            return {"average_ms": 0, "sample_count": 0}
        
        history = self._performance_cache[cache_key]
        return {
            "average_ms": round(statistics.mean(history), 2),
            "median_ms": round(statistics.median(history), 2),
            "sample_count": len(history),
            "trend": "stable"  # Could implement trend calculation
        }
    
    async def _calculate_percentile_rank(self, processing_time_ms: float, operation_type: str) -> int:
        """Calculate percentile rank for processing time"""
        cache_key = f"historical_{operation_type}"
        
        if cache_key not in self._performance_cache or not self._performance_cache[cache_key]:
            return 50  # Default to median
        
        history = self._performance_cache[cache_key]
        sorted_times = sorted(history)
        
        # Find position in sorted list
        position = sum(1 for time in sorted_times if time <= processing_time_ms)
        percentile = int((position / len(sorted_times)) * 100)
        
        return min(max(percentile, 1), 99)  # Clamp between 1-99
    
    async def _estimate_resource_utilization(self, processing_time_ms: float, operation_type: str) -> Dict[str, Any]:
        """Estimate resource utilization based on processing time"""
        # Simple heuristic-based estimation
        cpu_utilization = min(processing_time_ms / 10000, 1.0)  # Normalize to 0-1
        memory_utilization = min(processing_time_ms / 15000, 0.8)  # Memory is typically lower
        
        return {
            "cpu_estimate": round(cpu_utilization, 3),
            "memory_estimate": round(memory_utilization, 3),
            "network_io": "low" if processing_time_ms < 1000 else "medium",
            "disk_io": "minimal"
        }
    
    def _calculate_efficiency_score(self, processing_time_ms: float, operation_type: str) -> float:
        """Calculate efficiency score (0-1) where 1 is most efficient"""
        # Base efficiency calculation - lower time = higher efficiency
        base_score = max(0, 1 - (processing_time_ms / 30000))  # 30s = 0 efficiency
        
        # Apply operation-specific bonuses/penalties
        operation_multipliers = {
            "health_check": 1.2,  # Health checks should be very fast
            "cognitive_analysis": 0.8,  # Analysis can take longer
            "engagement_create": 1.0,
            "model_listing": 1.1
        }
        
        multiplier = operation_multipliers.get(operation_type, 1.0)
        return round(min(base_score * multiplier, 1.0), 3)
    
    async def _cache_performance_metric(self, operation_type: str, processing_time_ms: float):
        """Cache performance metric for future analysis"""
        cache_key = f"historical_{operation_type}"
        
        if cache_key not in self._performance_cache:
            self._performance_cache[cache_key] = []
        
        # Keep only last 100 measurements
        self._performance_cache[cache_key].append(processing_time_ms)
        if len(self._performance_cache[cache_key]) > 100:
            self._performance_cache[cache_key] = self._performance_cache[cache_key][-100:]
    
    async def _calculate_model_effectiveness(self, model: Dict[str, Any], context: str) -> float:
        """Calculate effectiveness score for a single model"""
        # Simple heuristic-based calculation
        base_score = 0.5
        
        # Boost for models with complete metadata
        if "description" in model and len(model["description"]) > 50:
            base_score += 0.2
        
        if "category" in model or "type" in model:
            base_score += 0.1
        
        if "examples" in model or "use_cases" in model:
            base_score += 0.1
        
        # Context relevance boost (simplified)
        if context and any(keyword in context.lower() for keyword in ["business", "strategy", "analysis"]):
            base_score += 0.1
        
        return min(base_score, 1.0)
    
    async def _calculate_score_distribution(self, scores: List[float]) -> Dict[str, Any]:
        """Calculate distribution metrics for scores"""
        if not scores:
            return {"mean": 0, "std_dev": 0, "quartiles": [0, 0, 0]}
        
        return {
            "mean": round(statistics.mean(scores), 3),
            "std_dev": round(statistics.stdev(scores) if len(scores) > 1 else 0, 3),
            "quartiles": [
                round(statistics.quantiles(scores, n=4)[0], 3) if len(scores) > 3 else min(scores),
                round(statistics.median(scores), 3),
                round(statistics.quantiles(scores, n=4)[2], 3) if len(scores) > 3 else max(scores)
            ]
        }
    
    async def _assess_model_quality(self, scores: List[float], enhanced_count: int, total_count: int) -> Dict[str, Any]:
        """Assess overall model quality"""
        if not scores:
            return {"grade": "F", "assessment": "No models available"}
        
        avg_score = statistics.mean(scores)
        enhanced_ratio = enhanced_count / total_count if total_count > 0 else 0
        
        # Calculate grade based on average score and enhanced ratio
        if avg_score >= 0.8 and enhanced_ratio >= 0.7:
            grade = "A"
        elif avg_score >= 0.7 and enhanced_ratio >= 0.5:
            grade = "B"
        elif avg_score >= 0.6 and enhanced_ratio >= 0.3:
            grade = "C"
        elif avg_score >= 0.5:
            grade = "D"
        else:
            grade = "F"
        
        return {
            "grade": grade,
            "avg_score": round(avg_score, 3),
            "enhanced_ratio": round(enhanced_ratio, 3),
            "assessment": f"Quality grade {grade} - {enhanced_count}/{total_count} enhanced models"
        }
    
    async def _assess_data_completeness(self, engagement_data: Dict[str, Any]) -> float:
        """Assess data completeness for engagement"""
        required_fields = ["id", "problem_statement", "status", "created_at"]
        optional_fields = ["client_context", "decision_context", "updated_at"]
        
        required_score = sum(1 for field in required_fields if field in engagement_data and engagement_data[field])
        optional_score = sum(1 for field in optional_fields if field in engagement_data and engagement_data[field])
        
        # Required fields are worth 80%, optional 20%
        completeness = (required_score / len(required_fields)) * 0.8 + (optional_score / len(optional_fields)) * 0.2
        return round(completeness, 3)
    
    async def _assess_data_quality(self, engagement_data: Dict[str, Any]) -> float:
        """Assess data quality indicators"""
        quality_score = 0.5  # Base score
        
        # Check problem statement quality
        problem_statement = engagement_data.get("problem_statement", "")
        if len(problem_statement) > 50:
            quality_score += 0.2
        if len(problem_statement.split()) > 10:
            quality_score += 0.1
        
        # Check context quality
        if engagement_data.get("client_context") and len(engagement_data["client_context"]) > 0:
            quality_score += 0.1
        
        # Check status validity
        valid_statuses = ["active", "completed", "pending", "archived"]
        if engagement_data.get("status") in valid_statuses:
            quality_score += 0.1
        
        return min(quality_score, 1.0)
    
    async def _assess_activity_level(self, engagement_data: Dict[str, Any]) -> float:
        """Assess activity level of engagement"""
        # Simple heuristic based on updated_at vs created_at
        created_at = engagement_data.get("created_at")
        updated_at = engagement_data.get("updated_at", created_at)
        
        if not created_at:
            return 0.5
        
        # If never updated, low activity
        if updated_at == created_at:
            return 0.3
        
        # Recent activity gets higher score
        try:
            update_time = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
            hours_since_update = (datetime.now().replace(tzinfo=None) - update_time.replace(tzinfo=None)).total_seconds() / 3600
            
            if hours_since_update < 1:
                return 1.0
            elif hours_since_update < 24:
                return 0.8
            elif hours_since_update < 168:  # 1 week
                return 0.6
            else:
                return 0.4
        except:
            return 0.5
    
    async def _assess_processing_success(self, engagement_data: Dict[str, Any]) -> float:
        """Assess processing success rate"""
        # Simple assessment based on status
        status = engagement_data.get("status", "")
        
        success_scores = {
            "completed": 1.0,
            "active": 0.8,
            "pending": 0.6,
            "error": 0.2,
            "failed": 0.1
        }
        
        return success_scores.get(status, 0.5)
    
    def _determine_health_status(self, overall_score: float) -> str:
        """Determine health status from overall score"""
        if overall_score >= 0.9:
            return "excellent"
        elif overall_score >= 0.7:
            return "good"
        elif overall_score >= 0.5:
            return "fair"
        elif overall_score >= 0.3:
            return "poor"
        else:
            return "critical"
    
    async def _generate_health_recommendations(self, health_indicators: Dict[str, float]) -> List[str]:
        """Generate health improvement recommendations"""
        recommendations = []
        
        if health_indicators.get("data_completeness", 0) < 0.7:
            recommendations.append("Improve data completeness by filling in missing fields")
        
        if health_indicators.get("data_quality", 0) < 0.6:
            recommendations.append("Enhance data quality with more detailed problem statements")
        
        if health_indicators.get("activity_level", 0) < 0.5:
            recommendations.append("Increase engagement activity with regular updates")
        
        if health_indicators.get("processing_success", 0) < 0.8:
            recommendations.append("Address processing issues to improve success rate")
        
        return recommendations
    
    async def _identify_risk_factors(self, health_indicators: Dict[str, float]) -> List[str]:
        """Identify risk factors from health indicators"""
        risks = []
        
        if health_indicators.get("processing_success", 0) < 0.5:
            risks.append("High processing failure rate")
        
        if health_indicators.get("data_quality", 0) < 0.4:
            risks.append("Poor data quality affecting analysis")
        
        if health_indicators.get("activity_level", 0) < 0.3:
            risks.append("Low engagement activity - potential abandonment")
        
        return risks
    
    async def _identify_improvements(self, health_indicators: Dict[str, float]) -> List[str]:
        """Identify improvement opportunities"""
        improvements = []
        
        if health_indicators.get("data_completeness", 0) < 0.8:
            improvements.append("Complete missing data fields for better analysis")
        
        if health_indicators.get("activity_level", 0) < 0.7:
            improvements.append("Increase interaction frequency for better outcomes")
        
        return improvements
    
    async def _analyze_confidence_scores(self, confidence_scores: Dict[str, float]) -> Dict[str, Any]:
        """Analyze confidence scores"""
        if not confidence_scores:
            return {"analysis": "No confidence scores available"}
        
        scores = list(confidence_scores.values())
        return {
            "average_confidence": round(statistics.mean(scores), 3),
            "confidence_range": {
                "min": round(min(scores), 3),
                "max": round(max(scores), 3)
            },
            "components": confidence_scores,
            "overall_assessment": "high" if statistics.mean(scores) > 0.8 else "medium" if statistics.mean(scores) > 0.6 else "low"
        }
    
    async def _analyze_nway_patterns(self, nway_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze N-way patterns"""
        return {
            "patterns_detected": len(nway_patterns),
            "pattern_types": list(set(p.get("type", "unknown") for p in nway_patterns)),
            "complexity_assessment": "high" if len(nway_patterns) > 5 else "medium" if len(nway_patterns) > 2 else "low"
        }
    
    async def _analyze_model_selection(self, selected_models: List[str]) -> Dict[str, Any]:
        """Analyze model selection"""
        return {
            "models_count": len(selected_models),
            "selection_diversity": "high" if len(selected_models) > 5 else "medium" if len(selected_models) > 2 else "low",
            "models": selected_models
        }
    
    async def _analyze_reasoning_quality(self, reasoning_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze reasoning quality"""
        return {
            "steps_count": len(reasoning_steps),
            "reasoning_depth": "deep" if len(reasoning_steps) > 5 else "moderate" if len(reasoning_steps) > 2 else "shallow",
            "step_types": list(set(step.get("type", "unknown") for step in reasoning_steps))
        }
    
    async def _calculate_overall_engagement_score(self, analytics: Dict[str, Any]) -> float:
        """Calculate overall engagement score"""
        scores = []
        
        # Extract numeric scores from different analytics components
        if "confidence_analytics" in analytics:
            scores.append(analytics["confidence_analytics"].get("average_confidence", 0.5))
        
        # Add more scoring logic based on available analytics
        if not scores:
            return 0.5
        
        return round(statistics.mean(scores), 3)
    
    async def _generate_analytics_insights(self, analytics: Dict[str, Any]) -> List[str]:
        """Generate insights from analytics"""
        insights = []
        
        if "confidence_analytics" in analytics:
            confidence = analytics["confidence_analytics"]
            if confidence.get("overall_assessment") == "high":
                insights.append("High confidence in analysis results")
            elif confidence.get("overall_assessment") == "low":
                insights.append("Analysis confidence could be improved")
        
        if "pattern_analytics" in analytics:
            patterns = analytics["pattern_analytics"]
            if patterns.get("patterns_detected", 0) > 3:
                insights.append("Rich pattern detection indicates complex analysis")
        
        return insights
    
    async def _score_database_health(self, database_health: Dict[str, Any]) -> float:
        """Score database health"""
        status = database_health.get("status", "unknown")
        
        if status == "healthy":
            return 1.0
        elif status == "warning":
            return 0.7
        elif status == "degraded":
            return 0.4
        else:
            return 0.1
    
    async def _score_performance_health(self, performance_data: Dict[str, Any]) -> float:
        """Score performance health"""
        # Simple scoring based on presence of performance data
        if not performance_data:
            return 0.5
        
        # Could implement more sophisticated performance analysis
        return 0.8
    
    async def _calculate_availability_score(self) -> float:
        """Calculate service availability score"""
        # Simplified - in production would track actual uptime
        return 0.95
    
    async def _calculate_resource_utilization_score(self) -> float:
        """Calculate resource utilization score"""
        # Simplified - in production would monitor actual resources
        return 0.8
    
    def _determine_system_status(self, overall_health: float) -> str:
        """Determine system status from health score"""
        if overall_health >= 0.9:
            return "operational"
        elif overall_health >= 0.7:
            return "degraded"
        elif overall_health >= 0.5:
            return "partial_outage"
        else:
            return "major_outage"
    
    async def _generate_health_alerts(self, db_score: float, perf_score: float, avail_score: float, resource_score: float) -> List[Dict[str, Any]]:
        """Generate health alerts based on component scores"""
        alerts = []
        
        if db_score < 0.5:
            alerts.append({
                "type": "database",
                "severity": "high",
                "message": "Database health is critically low"
            })
        
        if perf_score < 0.6:
            alerts.append({
                "type": "performance",
                "severity": "medium",
                "message": "Performance metrics indicate degradation"
            })
        
        if avail_score < 0.95:
            alerts.append({
                "type": "availability",
                "severity": "high",
                "message": "Service availability below target"
            })
        
        return alerts
    
    async def _calculate_health_trends(self) -> Dict[str, str]:
        """Calculate health trends from historical data"""
        # Simplified - in production would analyze historical trends
        return {
            "overall": "stable",
            "database": "improving",
            "performance": "stable",
            "availability": "stable"
        }
    
    async def _store_health_snapshot(self, health_metrics: Dict[str, Any]):
        """Store health snapshot for trend analysis"""
        # Keep last 100 snapshots for trend analysis
        if len(self._health_history) >= 100:
            self._health_history = self._health_history[-99:]
        
        self._health_history.append({
            "timestamp": datetime.now().isoformat(),
            "overall_score": health_metrics["overall_health_score"],
            "component_scores": health_metrics["component_scores"]
        })
