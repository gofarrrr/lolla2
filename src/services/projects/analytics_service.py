"""
Project Analytics Service
=========================

Analytics and reporting service for project operations. Extracts all statistical
calculations, health assessments, and performance metrics from the monolithic
project service into a specialized analytics engine.

Responsibilities:
- Project health scoring and assessment
- Usage analytics and cost tracking
- ROI calculations and efficiency metrics
- Knowledge base quality assessment
- Performance benchmarking
- Trend analysis and forecasting

This service focuses on data analysis and metrics generation without
any direct database operations (those are delegated to repository).
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass

from .specialized_contracts import (
    IProjectAnalytics,
    HealthMetrics,
    UsageAnalytics,
    ProjectAnalyticsError,
    ProjectStatistics,
    ProjectRecord,
)


@dataclass
class AnalyticsConfig:
    """Configuration for analytics calculations"""
    health_score_weights: Dict[str, float]
    efficiency_thresholds: Dict[str, float]
    cost_efficiency_tiers: Dict[str, float]
    roi_base_multipliers: Dict[str, float]


class ProjectAnalyticsService(IProjectAnalytics):
    """
    Analytics and reporting service for projects
    
    Provides comprehensive analytics, health assessments, and performance
    metrics without direct database access. Operates on data provided
    by repository and orchestration services.
    """
    
    def __init__(self, config: Optional[AnalyticsConfig] = None, context_stream=None, 
                 cache_ttl_seconds=300, enable_benchmarking=True):
        """Initialize analytics service with configuration"""
        self.logger = logging.getLogger(__name__)
        self.context_stream = context_stream
        self.cache_ttl_seconds = cache_ttl_seconds
        self.enable_benchmarking = enable_benchmarking
        self.config = config or self._get_default_config()
        self.logger.debug("🏗️ ProjectAnalyticsService initialized")
    
    async def calculate_health_metrics(self, project_id: str) -> HealthMetrics:
        """Calculate comprehensive health metrics for a project"""
        try:
            self.logger.debug(f"Calculating health metrics for project: {project_id}")
            
            # This would be called with data from repository service
            # For now, we'll create a placeholder implementation
            
            # Health scoring algorithm
            overall_health_score = await self._calculate_overall_health(project_id)
            rag_health_status = await self._assess_rag_health(project_id)
            activity_score = await self._calculate_activity_score(project_id)
            quality_score = await self._calculate_quality_score(project_id)
            efficiency_score = await self._calculate_efficiency_score(project_id)
            
            health_metrics = HealthMetrics(
                overall_health_score=overall_health_score,
                rag_health_status=rag_health_status,
                activity_score=activity_score,
                quality_score=quality_score,
                efficiency_score=efficiency_score,
                last_calculated=datetime.now(timezone.utc)
            )
            
            self.logger.info(f"✅ Health metrics calculated for project {project_id}: {overall_health_score:.2f}")
            
            return health_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Health metrics calculation failed: {e}")
            raise ProjectAnalyticsError(
                "health_metrics",
                {"project_id": project_id, "original_error": str(e)}
            ) from e
    
    async def generate_usage_analytics(self, project_id: str) -> UsageAnalytics:
        """Generate usage and cost analytics for a project"""
        try:
            self.logger.debug(f"Generating usage analytics for project: {project_id}")
            
            # This would receive data from repository service
            # Placeholder implementation for now
            
            # Calculate analytics from statistics
            total_analyses = 0  # Would come from repository
            recent_analyses_30d = 0  # Would come from repository
            total_cost = 0.0  # Would come from repository
            total_tokens = 0  # Would come from repository
            
            # Derived calculations
            avg_tokens_per_analysis = (
                total_tokens // max(1, total_analyses)
            )
            avg_cost_per_analysis = (
                total_cost / max(1, total_analyses)
            )
            roi_score = await self._calculate_roi_score(
                total_analyses, total_cost, project_id
            )
            efficiency_metrics = await self._calculate_efficiency_metrics(
                total_analyses, total_tokens, total_cost
            )
            
            usage_analytics = UsageAnalytics(
                total_analyses=total_analyses,
                recent_analyses_30d=recent_analyses_30d,
                avg_tokens_per_analysis=avg_tokens_per_analysis,
                avg_cost_per_analysis=avg_cost_per_analysis,
                total_cost=total_cost,
                roi_score=roi_score,
                efficiency_metrics=efficiency_metrics
            )
            
            self.logger.info(f"✅ Usage analytics generated for project {project_id}")
            
            return usage_analytics
            
        except Exception as e:
            self.logger.error(f"❌ Usage analytics generation failed: {e}")
            raise ProjectAnalyticsError(
                "usage_analytics",
                {"project_id": project_id, "original_error": str(e)}
            ) from e
    
    async def assess_knowledge_base_health(self, project_id: str) -> Dict[str, Any]:
        """Assess knowledge base quality and completeness"""
        try:
            self.logger.debug(f"Assessing knowledge base health for project: {project_id}")
            
            # Knowledge base health assessment algorithm
            # This would receive KB stats from repository service
            
            # Placeholder metrics
            total_documents = 0
            total_chunks = 0
            avg_content_quality = 0.0
            avg_semantic_density = 0.0
            latest_document_date = None
            
            # Health determination logic
            health_status = "healthy"
            if total_documents == 0:
                health_status = "empty"
            elif avg_content_quality < 0.6:
                health_status = "needs_improvement"
            elif not latest_document_date:
                health_status = "stale"
            
            # Content analysis
            content_diversity_score = self._calculate_content_diversity(
                total_documents, total_chunks
            )
            freshness_score = self._calculate_freshness_score(latest_document_date)
            completeness_score = self._calculate_completeness_score(
                total_documents, total_chunks
            )
            
            assessment = {
                "project_id": project_id,
                "health_status": health_status,
                "overall_score": (content_diversity_score + freshness_score + completeness_score) / 3,
                "metrics": {
                    "total_documents": total_documents,
                    "total_chunks": total_chunks,
                    "avg_content_quality": avg_content_quality,
                    "avg_semantic_density": avg_semantic_density,
                    "content_diversity_score": content_diversity_score,
                    "freshness_score": freshness_score,
                    "completeness_score": completeness_score,
                },
                "recommendations": self._generate_kb_recommendations(
                    health_status, total_documents, avg_content_quality
                ),
                "assessed_at": datetime.now(timezone.utc).isoformat(),
            }
            
            self.logger.info(f"✅ Knowledge base health assessed for project {project_id}: {health_status}")
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"❌ Knowledge base health assessment failed: {e}")
            raise ProjectAnalyticsError(
                "knowledge_base_health",
                {"project_id": project_id, "original_error": str(e)}
            ) from e
    
    async def calculate_roi_metrics(self, project_id: str) -> Dict[str, float]:
        """Calculate return on investment metrics"""
        try:
            self.logger.debug(f"Calculating ROI metrics for project: {project_id}")
            
            # ROI calculation based on usage and value generated
            # This would receive data from repository and usage tracking
            
            total_cost = 0.0  # From repository
            total_analyses = 0  # From repository
            avg_analysis_value = 100.0  # Estimated value per analysis (configurable)
            
            # Calculate various ROI metrics
            total_value_generated = total_analyses * avg_analysis_value
            basic_roi = ((total_value_generated - total_cost) / max(0.01, total_cost)) * 100
            
            # Cost efficiency metrics
            cost_per_analysis = total_cost / max(1, total_analyses)
            value_per_dollar = total_value_generated / max(0.01, total_cost)
            
            # Trend-based ROI (would require historical data)
            roi_trend = 0.0  # Placeholder
            
            # ROI scoring (0-100 scale)
            roi_score = min(100, max(0, basic_roi))
            
            roi_metrics = {
                "basic_roi_percentage": basic_roi,
                "roi_score": roi_score,
                "cost_per_analysis": cost_per_analysis,
                "value_per_dollar": value_per_dollar,
                "total_value_generated": total_value_generated,
                "total_cost": total_cost,
                "roi_trend": roi_trend,
                "efficiency_rating": self._get_efficiency_rating(cost_per_analysis),
            }
            
            self.logger.info(f"✅ ROI metrics calculated for project {project_id}: {roi_score:.1f}")
            
            return roi_metrics
            
        except Exception as e:
            self.logger.error(f"❌ ROI metrics calculation failed: {e}")
            raise ProjectAnalyticsError(
                "roi_metrics",
                {"project_id": project_id, "original_error": str(e)}
            ) from e
    
    async def generate_efficiency_report(self, project_id: str) -> Dict[str, Any]:
        """Generate efficiency analysis report"""
        try:
            self.logger.debug(f"Generating efficiency report for project: {project_id}")
            
            # Comprehensive efficiency analysis
            # This would combine data from multiple sources
            
            # Resource utilization metrics
            knowledge_base_utilization = await self._calculate_kb_utilization(project_id)
            context_merge_efficiency = await self._calculate_context_efficiency(project_id)
            analysis_throughput = await self._calculate_analysis_throughput(project_id)
            
            # Cost efficiency metrics
            cost_efficiency = await self._calculate_cost_efficiency(project_id)
            token_efficiency = await self._calculate_token_efficiency(project_id)
            
            # Time efficiency metrics
            avg_analysis_time = await self._calculate_avg_analysis_time(project_id)
            processing_efficiency = await self._calculate_processing_efficiency(project_id)
            
            # Overall efficiency score
            efficiency_components = [
                knowledge_base_utilization,
                context_merge_efficiency,
                cost_efficiency,
                token_efficiency,
                processing_efficiency
            ]
            overall_efficiency = sum(efficiency_components) / len(efficiency_components)
            
            # Generate recommendations
            recommendations = self._generate_efficiency_recommendations(
                knowledge_base_utilization,
                cost_efficiency,
                processing_efficiency
            )
            
            efficiency_report = {
                "project_id": project_id,
                "overall_efficiency_score": overall_efficiency,
                "efficiency_grade": self._get_efficiency_grade(overall_efficiency),
                "metrics": {
                    "knowledge_base_utilization": knowledge_base_utilization,
                    "context_merge_efficiency": context_merge_efficiency,
                    "analysis_throughput": analysis_throughput,
                    "cost_efficiency": cost_efficiency,
                    "token_efficiency": token_efficiency,
                    "avg_analysis_time_minutes": avg_analysis_time,
                    "processing_efficiency": processing_efficiency,
                },
                "recommendations": recommendations,
                "benchmark_comparison": await self._get_efficiency_benchmarks(),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            
            self.logger.info(f"✅ Efficiency report generated for project {project_id}: {overall_efficiency:.1f}")
            
            return efficiency_report
            
        except Exception as e:
            self.logger.error(f"❌ Efficiency report generation failed: {e}")
            raise ProjectAnalyticsError(
                "efficiency_report",
                {"project_id": project_id, "original_error": str(e)}
            ) from e
    
    async def benchmark_project_performance(
        self, 
        project_id: str,
        comparison_projects: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Benchmark project against peers"""
        try:
            self.logger.debug(f"Benchmarking project performance: {project_id}")
            
            # Performance benchmarking analysis
            # This would compare against peer projects or industry standards
            
            project_metrics = await self._get_project_performance_metrics(project_id)
            
            if comparison_projects:
                peer_metrics = []
                for peer_id in comparison_projects:
                    peer_metric = await self._get_project_performance_metrics(peer_id)
                    peer_metrics.append(peer_metric)
                
                benchmark_results = self._compare_against_peers(project_metrics, peer_metrics)
            else:
                # Use industry benchmarks
                benchmark_results = self._compare_against_industry_standards(project_metrics)
            
            # Calculate percentile rankings
            percentile_rankings = self._calculate_percentile_rankings(
                project_metrics, benchmark_results
            )
            
            # Generate improvement opportunities
            improvement_opportunities = self._identify_improvement_opportunities(
                project_metrics, benchmark_results
            )
            
            benchmark_report = {
                "project_id": project_id,
                "benchmark_date": datetime.now(timezone.utc).isoformat(),
                "project_metrics": project_metrics,
                "benchmark_results": benchmark_results,
                "percentile_rankings": percentile_rankings,
                "performance_grade": self._get_performance_grade(percentile_rankings),
                "improvement_opportunities": improvement_opportunities,
                "comparison_type": "peer_projects" if comparison_projects else "industry_standards",
                "comparison_count": len(comparison_projects) if comparison_projects else None,
            }
            
            self.logger.info(f"✅ Performance benchmarking completed for project {project_id}")
            
            return benchmark_report
            
        except Exception as e:
            self.logger.error(f"❌ Performance benchmarking failed: {e}")
            raise ProjectAnalyticsError(
                "benchmark_performance",
                {"project_id": project_id, "comparison_projects": comparison_projects, "original_error": str(e)}
            ) from e
    
    # ============================================================
    # Private Helper Methods
    # ============================================================
    
    async def _calculate_overall_health(self, project_id: str) -> float:
        """Calculate overall health score for project"""
        # Weighted combination of various health factors
        activity_weight = self.config.health_score_weights.get("activity", 0.3)
        quality_weight = self.config.health_score_weights.get("quality", 0.3)
        efficiency_weight = self.config.health_score_weights.get("efficiency", 0.2)
        freshness_weight = self.config.health_score_weights.get("freshness", 0.2)
        
        # Placeholder calculations - would use real data
        activity_score = 75.0
        quality_score = 80.0
        efficiency_score = 70.0
        freshness_score = 85.0
        
        overall_score = (
            activity_score * activity_weight +
            quality_score * quality_weight +
            efficiency_score * efficiency_weight +
            freshness_score * freshness_weight
        )
        
        return min(100.0, max(0.0, overall_score))
    
    async def _assess_rag_health(self, project_id: str) -> str:
        """Assess RAG system health status"""
        # This would check various RAG health indicators
        # Placeholder implementation
        return "healthy"
    
    async def _calculate_activity_score(self, project_id: str) -> float:
        """Calculate activity score based on recent usage"""
        # Placeholder implementation
        return 75.0
    
    async def _calculate_quality_score(self, project_id: str) -> float:
        """Calculate quality score based on content and analysis quality"""
        # Placeholder implementation
        return 80.0
    
    async def _calculate_efficiency_score(self, project_id: str) -> float:
        """Calculate efficiency score based on resource utilization"""
        # Placeholder implementation
        return 70.0
    
    async def _calculate_roi_score(self, total_analyses: int, total_cost: float, project_id: str) -> float:
        """Calculate ROI score for project"""
        base_multiplier = self.config.roi_base_multipliers.get("analysis", 10.0)
        estimated_value = total_analyses * base_multiplier
        
        if total_cost <= 0:
            return 100.0 if total_analyses > 0 else 0.0
        
        roi_ratio = estimated_value / total_cost
        return min(100.0, roi_ratio * 10)  # Scale to 0-100
    
    async def _calculate_efficiency_metrics(
        self, total_analyses: int, total_tokens: int, total_cost: float
    ) -> Dict[str, Any]:
        """Calculate efficiency metrics"""
        return {
            "knowledge_base_utilization": min(100, (total_analyses / max(1, total_tokens)) * 1000),
            "context_merge_potential": total_analyses > 0,
            "cost_efficiency_rating": self._get_efficiency_rating(
                total_cost / max(1, total_analyses)
            ),
            "token_efficiency": total_analyses / max(1, total_tokens) * 1000,
        }
    
    def _calculate_content_diversity(self, total_docs: int, total_chunks: int) -> float:
        """Calculate content diversity score"""
        if total_docs == 0:
            return 0.0
        
        chunks_per_doc = total_chunks / total_docs
        # Optimal range is 5-15 chunks per document
        if 5 <= chunks_per_doc <= 15:
            return 100.0
        elif chunks_per_doc < 5:
            return (chunks_per_doc / 5) * 100
        else:
            return max(50.0, 100 - ((chunks_per_doc - 15) * 3))
    
    def _calculate_freshness_score(self, latest_date: Optional[datetime]) -> float:
        """Calculate content freshness score"""
        if not latest_date:
            return 0.0
        
        days_old = (datetime.now(timezone.utc) - latest_date).days
        
        if days_old <= 7:
            return 100.0
        elif days_old <= 30:
            return 90.0 - ((days_old - 7) * 2)
        elif days_old <= 90:
            return 44.0 - ((days_old - 30) * 0.5)
        else:
            return max(10.0, 14.0 - ((days_old - 90) * 0.1))
    
    def _calculate_completeness_score(self, total_docs: int, total_chunks: int) -> float:
        """Calculate knowledge base completeness score"""
        if total_docs == 0:
            return 0.0
        
        # Basic heuristic - more sophisticated analysis would be needed
        doc_score = min(100, total_docs * 10)  # 10 points per document, max 100
        chunk_score = min(100, total_chunks * 2)  # 2 points per chunk, max 100
        
        return (doc_score + chunk_score) / 2
    
    def _generate_kb_recommendations(
        self, health_status: str, total_docs: int, avg_quality: float
    ) -> List[str]:
        """Generate knowledge base improvement recommendations"""
        recommendations = []
        
        if health_status == "empty":
            recommendations.append("Upload documents to build your knowledge base")
            recommendations.append("Start with key reference materials for your domain")
        elif health_status == "needs_improvement":
            recommendations.append("Review and improve document quality")
            recommendations.append("Consider adding more diverse content types")
        elif total_docs < 5:
            recommendations.append("Add more documents to improve analysis depth")
        
        if avg_quality < 0.7:
            recommendations.append("Focus on higher-quality source materials")
        
        return recommendations
    
    def _get_efficiency_rating(self, cost_per_analysis: float) -> str:
        """Get efficiency rating based on cost per analysis"""
        thresholds = self.config.cost_efficiency_tiers
        
        if cost_per_analysis < thresholds.get("excellent", 0.10):
            return "excellent"
        elif cost_per_analysis < thresholds.get("good", 0.50):
            return "good"
        else:
            return "needs_optimization"
    
    def _get_default_config(self) -> AnalyticsConfig:
        """Get default analytics configuration"""
        return AnalyticsConfig(
            health_score_weights={
                "activity": 0.3,
                "quality": 0.3,
                "efficiency": 0.2,
                "freshness": 0.2,
            },
            efficiency_thresholds={
                "excellent": 90.0,
                "good": 75.0,
                "average": 60.0,
            },
            cost_efficiency_tiers={
                "excellent": 0.10,
                "good": 0.50,
                "needs_optimization": 1.00,
            },
            roi_base_multipliers={
                "analysis": 10.0,
                "document": 5.0,
                "search": 2.0,
            }
        )
    
    # Placeholder methods for comprehensive analytics
    async def _calculate_kb_utilization(self, project_id: str) -> float:
        return 75.0
    
    async def _calculate_context_efficiency(self, project_id: str) -> float:
        return 80.0
    
    async def _calculate_analysis_throughput(self, project_id: str) -> float:
        return 85.0
    
    async def _calculate_cost_efficiency(self, project_id: str) -> float:
        return 70.0
    
    async def _calculate_token_efficiency(self, project_id: str) -> float:
        return 75.0
    
    async def _calculate_avg_analysis_time(self, project_id: str) -> float:
        return 2.5  # minutes
    
    async def _calculate_processing_efficiency(self, project_id: str) -> float:
        return 80.0
    
    def _generate_efficiency_recommendations(
        self, kb_util: float, cost_eff: float, proc_eff: float
    ) -> List[str]:
        recommendations = []
        
        if kb_util < 60:
            recommendations.append("Improve knowledge base utilization by adding more relevant content")
        if cost_eff < 70:
            recommendations.append("Optimize cost efficiency by reviewing analysis settings")
        if proc_eff < 75:
            recommendations.append("Improve processing efficiency by optimizing query complexity")
        
        return recommendations
    
    def _get_efficiency_grade(self, score: float) -> str:
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    async def _get_efficiency_benchmarks(self) -> Dict[str, float]:
        return {
            "industry_average": 75.0,
            "top_quartile": 85.0,
            "best_in_class": 95.0,
        }
    
    async def _get_project_performance_metrics(self, project_id: str) -> Dict[str, float]:
        return {
            "efficiency_score": 75.0,
            "cost_per_analysis": 0.25,
            "quality_score": 80.0,
            "throughput": 85.0,
        }
    
    def _compare_against_peers(
        self, project_metrics: Dict[str, float], peer_metrics: List[Dict[str, float]]
    ) -> Dict[str, Any]:
        return {"comparison_type": "peer", "peer_count": len(peer_metrics)}
    
    def _compare_against_industry_standards(
        self, project_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        return {"comparison_type": "industry", "standards_version": "2024.1"}
    
    def _calculate_percentile_rankings(
        self, project_metrics: Dict[str, float], benchmark_results: Dict[str, Any]
    ) -> Dict[str, int]:
        return {
            "efficiency": 75,
            "cost": 80,
            "quality": 70,
            "throughput": 85,
        }
    
    def _identify_improvement_opportunities(
        self, project_metrics: Dict[str, float], benchmark_results: Dict[str, Any]
    ) -> List[str]:
        return [
            "Improve cost efficiency to reach top quartile",
            "Enhance quality score through better content curation",
        ]
    
    def _get_performance_grade(self, percentile_rankings: Dict[str, int]) -> str:
        avg_percentile = sum(percentile_rankings.values()) / len(percentile_rankings)
        return self._get_efficiency_grade(avg_percentile)


# ============================================================
# Factory Function
# ============================================================

def get_project_analytics(config: Optional[AnalyticsConfig] = None) -> IProjectAnalytics:
    """Factory function for dependency injection"""
    return ProjectAnalyticsService(config)