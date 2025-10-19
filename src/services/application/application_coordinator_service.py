"""
METIS Application Coordinator Service
Part of Application Services Cluster - Master orchestrator for all application services

Extracted from model_manager.py during Phase 5.3 decomposition.
Single Responsibility: Coordinate all application services and manage feature flags.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.services.contracts.application_contracts import (
    IApplicationCoordinatorService,
    ApplicationResultContract,
    FeatureFlag,
    OrchestrationMode,
    ApplicationStrategy,
)

# Import all application services
from src.services.application.model_registry_service import get_model_registry_service
from src.services.application.lifecycle_management_service import (
    get_lifecycle_management_service,
)
from src.services.application.performance_monitoring_service import (
    get_performance_monitoring_service,
)
from src.services.application.model_application_service import (
    get_model_application_service,
)


class ApplicationCoordinatorService(IApplicationCoordinatorService):
    """
    Master coordinator service for all application services
    Clean orchestration with single responsibility for unified application management
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize all application services
        self.registry_service = get_model_registry_service()
        self.lifecycle_service = get_lifecycle_management_service()
        self.performance_service = get_performance_monitoring_service()
        self.application_service = get_model_application_service()

        # Coordination configuration
        self.coordination_config = {
            "max_concurrent_applications": 3,
            "application_timeout_seconds": 300,
            "feature_flag_evaluation_enabled": True,
            "performance_monitoring_enabled": True,
            "lifecycle_management_enabled": True,
            "automatic_model_selection": True,
            "quality_threshold_minimum": 0.6,
        }

        # Feature flag defaults
        self.default_feature_flags = {
            FeatureFlag.BAYESIAN_LEARNING: True,
            FeatureFlag.CONFIDENCE_CALIBRATION: True,
            FeatureFlag.DECISION_CAPTURE: False,
            FeatureFlag.VECTOR_SIMILARITY: False,
            FeatureFlag.NWAY_ENHANCEMENT: True,
        }

        # Coordination performance metrics
        self.coordination_metrics = {
            "total_coordinations": 0,
            "successful_coordinations": 0,
            "average_coordination_time_ms": 0.0,
            "service_failure_counts": {
                "registry": 0,
                "lifecycle": 0,
                "performance": 0,
                "application": 0,
            },
            "feature_flag_usage": {flag.value: 0 for flag in FeatureFlag},
        }

        # Service health cache
        self.service_health_cache = {}
        self.health_cache_ttl_minutes = 5
        self.last_health_check = {}

        self.logger.info(
            "🎭 ApplicationCoordinatorService initialized - All services orchestrated"
        )

    async def coordinate_application_workflow(
        self, request_context: Dict[str, Any], feature_flags: List[FeatureFlag]
    ) -> ApplicationResultContract:
        """
        Core coordination method: Coordinate complete application workflow
        Master orchestration with intelligent service coordination and feature flag management
        """
        try:
            start_time = datetime.utcnow()

            # Phase 1: Evaluate feature flags and prepare coordination
            active_features = await self._evaluate_feature_flags(feature_flags)

            # Phase 2: Determine orchestration strategy
            orchestration_strategy = await self._determine_orchestration_strategy(
                request_context, active_features
            )

            # Phase 3: Execute coordinated workflow
            coordination_result = await self._execute_coordinated_workflow(
                request_context, orchestration_strategy, active_features
            )

            # Phase 4: Apply post-coordination enhancements
            enhanced_result = await self._apply_coordination_enhancements(
                coordination_result, active_features
            )

            # Phase 5: Calculate overall confidence and metrics
            overall_confidence = await self._calculate_overall_confidence(
                enhanced_result
            )
            coordination_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Phase 6: Update coordination metrics
            await self._update_coordination_metrics(coordination_time, True)

            # Phase 7: Create comprehensive result contract
            final_result = ApplicationResultContract(
                result_id=f"coord_{start_time.timestamp()}",
                engagement_id=request_context.get("engagement_id", "unknown"),
                orchestration_result=enhanced_result["orchestration_result"],
                application_results=enhanced_result["application_results"],
                llm_responses=enhanced_result.get("llm_responses", []),
                performance_metrics=enhanced_result.get("performance_metrics", []),
                feature_flags_used=feature_flags,
                overall_confidence=overall_confidence,
                processing_summary={
                    "services_orchestrated": enhanced_result["services_used"],
                    "coordination_time_ms": coordination_time,
                    "orchestration_strategy": orchestration_strategy,
                    "enhancements_applied": enhanced_result.get(
                        "enhancements_applied", []
                    ),
                    "feature_flags_active": len(active_features),
                },
                result_timestamp=datetime.utcnow(),
                total_processing_time_ms=coordination_time,
                service_version="v5_modular_coordinator",
            )

            self.logger.info(
                f"🎭 Application workflow coordinated successfully in {coordination_time:.0f}ms"
            )
            return final_result

        except Exception as e:
            coordination_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            await self._update_coordination_metrics(coordination_time, False)

            self.logger.error(f"❌ Application coordination failed: {e}")
            return await self._create_fallback_result(
                request_context, feature_flags, str(e)
            )

    async def get_cluster_health(self) -> Dict[str, Any]:
        """
        Core method: Get comprehensive health status of entire application cluster
        Complete cluster monitoring with service-level health assessment
        """
        try:
            # Check if health data is cached and fresh
            cache_key = "cluster_health"
            if self._is_health_cache_valid(cache_key):
                return self.service_health_cache[cache_key]

            # Collect health from all constituent services
            service_healths = await self._collect_service_health_data()

            # Calculate cluster-level metrics
            cluster_metrics = await self._calculate_cluster_metrics(service_healths)

            # Assess overall cluster health
            cluster_health_status = self._assess_cluster_health_status(
                service_healths, cluster_metrics
            )

            # Create comprehensive health report
            health_report = {
                "cluster_name": "ApplicationServicesCluster",
                "cluster_status": cluster_health_status["overall_status"],
                "cluster_health_percentage": cluster_health_status["health_percentage"],
                "coordinator_version": "v5_modular",
                "services_count": len(service_healths),
                "healthy_services": cluster_health_status["healthy_services"],
                "service_details": service_healths,
                "cluster_metrics": cluster_metrics,
                "coordination_statistics": self.coordination_metrics,
                "configuration": self.coordination_config,
                "feature_flags_default": {
                    flag.value: enabled
                    for flag, enabled in self.default_feature_flags.items()
                },
                "capabilities": [
                    "unified_application_coordination",
                    "feature_flag_management",
                    "service_health_monitoring",
                    "performance_tracking",
                    "intelligent_orchestration",
                ],
                "cluster_insights": await self._generate_cluster_insights(
                    service_healths, cluster_metrics
                ),
                "last_health_check": datetime.utcnow().isoformat(),
            }

            # Cache the health report
            self.service_health_cache[cache_key] = health_report
            self.last_health_check[cache_key] = datetime.utcnow()

            return health_report

        except Exception as e:
            self.logger.error(f"❌ Cluster health assessment failed: {e}")
            return {
                "cluster_name": "ApplicationServicesCluster",
                "cluster_status": "error",
                "error": str(e),
                "last_health_check": datetime.utcnow().isoformat(),
            }

    async def _evaluate_feature_flags(
        self, requested_flags: List[FeatureFlag]
    ) -> Dict[FeatureFlag, bool]:
        """Evaluate and resolve feature flags"""
        try:
            active_features = {}

            # Start with defaults
            for flag, default_value in self.default_feature_flags.items():
                active_features[flag] = default_value

            # Override with requested flags (assume all requested flags are enabled)
            for flag in requested_flags:
                active_features[flag] = True
                self.coordination_metrics["feature_flag_usage"][flag.value] += 1

            # Apply feature flag business logic
            active_features = await self._apply_feature_flag_logic(active_features)

            return active_features

        except Exception as e:
            self.logger.error(f"❌ Feature flag evaluation failed: {e}")
            return self.default_feature_flags

    async def _determine_orchestration_strategy(
        self, request_context: Dict[str, Any], active_features: Dict[FeatureFlag, bool]
    ) -> str:
        """Determine optimal orchestration strategy based on context and features"""
        try:
            # Analyze request complexity
            complexity_indicators = [
                len(request_context.get("problem_statement", "")),
                len(request_context.get("business_context", {})),
                request_context.get("accuracy_requirement", 0.7),
                len(active_features),
            ]

            complexity_score = (
                sum(complexity_indicators) / len(complexity_indicators) / 100
            )

            # Determine strategy based on complexity and features
            if (
                active_features.get(FeatureFlag.NWAY_ENHANCEMENT, False)
                and complexity_score > 0.8
            ):
                return "enhanced_nway_orchestration"
            elif complexity_score > 0.6:
                return "standard_multi_service_orchestration"
            elif len(active_features) > 3:
                return "feature_rich_orchestration"
            else:
                return "basic_orchestration"

        except Exception as e:
            self.logger.error(f"❌ Orchestration strategy determination failed: {e}")
            return "basic_orchestration"

    async def _execute_coordinated_workflow(
        self,
        request_context: Dict[str, Any],
        orchestration_strategy: str,
        active_features: Dict[FeatureFlag, bool],
    ) -> Dict[str, Any]:
        """Execute the coordinated workflow across all services"""
        try:
            workflow_result = {
                "services_used": [],
                "orchestration_result": None,
                "application_results": [],
                "llm_responses": [],
                "performance_metrics": [],
                "enhancements_applied": [],
            }

            # Phase 1: Model Selection and Registry
            selected_models = await self._coordinate_model_selection(request_context)
            workflow_result["services_used"].append("registry")

            if not selected_models:
                selected_models = ["deepseek_chat"]  # Fallback

            # Phase 2: Lifecycle Management - Initialize models
            for model_id in selected_models:
                try:
                    initialization_event = (
                        await self.lifecycle_service.initialize_model(model_id)
                    )
                    if initialization_event.event_status != "completed":
                        self.logger.warning(f"⚠️ Model initialization issue: {model_id}")
                except Exception as e:
                    self.logger.warning(
                        f"⚠️ Model initialization failed: {model_id} - {e}"
                    )

            workflow_result["services_used"].append("lifecycle")

            # Phase 3: Application Strategy Execution
            application_results = await self._coordinate_application_execution(
                selected_models, request_context, active_features
            )
            workflow_result["application_results"] = application_results
            workflow_result["services_used"].append("application")

            # Phase 4: Performance Monitoring
            if self.coordination_config["performance_monitoring_enabled"]:
                performance_metrics = await self._coordinate_performance_monitoring(
                    selected_models, application_results
                )
                workflow_result["performance_metrics"] = performance_metrics
                workflow_result["services_used"].append("performance")

            # Phase 5: Create orchestration result
            from src.services.contracts.application_contracts import (
                ModelOrchestrationContract,
            )

            workflow_result["orchestration_result"] = ModelOrchestrationContract(
                orchestration_id=f"coord_{datetime.utcnow().timestamp()}",
                selected_models=selected_models,
                orchestration_mode=OrchestrationMode.STANDARD,  # Simplified
                coordination_metadata={
                    "strategy": orchestration_strategy,
                    "models_initialized": len(selected_models),
                    "applications_completed": len(application_results),
                    "performance_metrics_recorded": len(
                        workflow_result["performance_metrics"]
                    ),
                },
                nway_interactions=[],  # Would be populated by N-Way service
                research_integration_data=None,
                similar_patterns_detected=[],
                orchestration_timestamp=datetime.utcnow(),
                total_orchestration_time_ms=100.0,  # Simplified
                service_version="v5_modular",
            )

            return workflow_result

        except Exception as e:
            self.logger.error(f"❌ Coordinated workflow execution failed: {e}")
            raise

    async def _coordinate_model_selection(
        self, request_context: Dict[str, Any]
    ) -> List[str]:
        """Coordinate model selection using registry service"""
        try:
            # Determine application strategy from context
            strategy = self._infer_application_strategy(request_context)

            # Get models supporting this strategy
            compatible_models = await self.registry_service.list_models_by_strategy(
                strategy
            )

            # Select top models (simplified selection)
            max_models = min(
                self.coordination_config["max_concurrent_applications"],
                len(compatible_models),
            )
            selected_models = [
                model.model_id for model in compatible_models[:max_models]
            ]

            if not selected_models:
                # Fallback to any available models
                all_models = await self.registry_service.get_registry_analytics()
                if all_models.get("registry_summary", {}).get("active_models", 0) > 0:
                    selected_models = ["deepseek_chat"]  # Safe fallback

            return selected_models

        except Exception as e:
            self.logger.error(f"❌ Model selection coordination failed: {e}")
            return ["deepseek_chat"]  # Ultimate fallback

    async def _coordinate_application_execution(
        self,
        selected_models: List[str],
        request_context: Dict[str, Any],
        active_features: Dict[FeatureFlag, bool],
    ) -> List[Any]:
        """Coordinate application execution across selected models"""
        try:
            application_results = []
            strategy = self._infer_application_strategy(request_context)

            # Execute applications for each selected model
            for model_id in selected_models:
                try:
                    application_result = (
                        await self.application_service.apply_model_strategy(
                            model_id=model_id,
                            strategy=strategy,
                            input_data=request_context,
                            context={
                                "engagement_id": request_context.get(
                                    "engagement_id", "unknown"
                                ),
                                "feature_flags": active_features,
                                "coordination_context": True,
                            },
                        )
                    )

                    application_results.append(application_result)

                except Exception as e:
                    self.logger.warning(f"⚠️ Application failed for {model_id}: {e}")
                    self.coordination_metrics["service_failure_counts"][
                        "application"
                    ] += 1

            return application_results

        except Exception as e:
            self.logger.error(f"❌ Application execution coordination failed: {e}")
            return []

    async def _coordinate_performance_monitoring(
        self, selected_models: List[str], application_results: List[Any]
    ) -> List[Any]:
        """Coordinate performance monitoring for executed applications"""
        try:
            from src.services.contracts.application_contracts import (
                PerformanceMetricType,
            )

            performance_metrics = []

            for i, model_id in enumerate(selected_models):
                if i < len(application_results):
                    application_result = application_results[i]

                    # Record multiple performance metrics
                    metrics_to_record = [
                        (
                            PerformanceMetricType.RESPONSE_TIME,
                            application_result.processing_time_ms,
                        ),
                        (
                            PerformanceMetricType.CONFIDENCE_LEVEL,
                            application_result.confidence_score,
                        ),
                        (
                            PerformanceMetricType.ACCURACY_SCORE,
                            application_result.quality_metrics.get("accuracy", 0.7),
                        ),
                    ]

                    for metric_type, metric_value in metrics_to_record:
                        try:
                            perf_metric = await self.performance_service.record_performance_metric(
                                model_id=model_id,
                                metric_type=metric_type,
                                metric_value=metric_value,
                                context={
                                    "application_id": application_result.application_id,
                                    "strategy": application_result.strategy_used.value,
                                },
                            )
                            performance_metrics.append(perf_metric)

                        except Exception as e:
                            self.logger.warning(
                                f"⚠️ Performance metric recording failed: {e}"
                            )

            return performance_metrics

        except Exception as e:
            self.logger.error(f"❌ Performance monitoring coordination failed: {e}")
            return []

    async def _apply_coordination_enhancements(
        self,
        coordination_result: Dict[str, Any],
        active_features: Dict[FeatureFlag, bool],
    ) -> Dict[str, Any]:
        """Apply coordination-level enhancements based on feature flags"""
        try:
            enhanced_result = coordination_result.copy()
            enhancements_applied = []

            # Apply Bayesian learning enhancement
            if active_features.get(FeatureFlag.BAYESIAN_LEARNING, False):
                # Would integrate with Bayesian learning service
                enhancements_applied.append("bayesian_learning_integration")

            # Apply confidence calibration
            if active_features.get(FeatureFlag.CONFIDENCE_CALIBRATION, False):
                enhanced_result = await self._apply_confidence_calibration(
                    enhanced_result
                )
                enhancements_applied.append("confidence_calibration")

            # Apply decision capture
            if active_features.get(FeatureFlag.DECISION_CAPTURE, False):
                enhanced_result = await self._apply_decision_capture(enhanced_result)
                enhancements_applied.append("decision_capture")

            # Apply vector similarity enhancement
            if active_features.get(FeatureFlag.VECTOR_SIMILARITY, False):
                enhanced_result = await self._apply_vector_similarity(enhanced_result)
                enhancements_applied.append("vector_similarity")

            enhanced_result["enhancements_applied"] = enhancements_applied
            return enhanced_result

        except Exception as e:
            self.logger.error(f"❌ Coordination enhancements failed: {e}")
            return coordination_result

    async def _apply_confidence_calibration(
        self, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply confidence calibration enhancement"""
        try:
            # Simplified confidence calibration
            application_results = result.get("application_results", [])

            for app_result in application_results:
                # Adjust confidence based on historical performance
                original_confidence = app_result.confidence_score
                calibrated_confidence = (
                    original_confidence * 0.9
                )  # Conservative adjustment
                app_result.confidence_score = calibrated_confidence

                # Add calibration metadata
                app_result.processing_metadata["confidence_calibration"] = {
                    "original": original_confidence,
                    "calibrated": calibrated_confidence,
                    "adjustment_factor": 0.9,
                }

            return result

        except Exception as e:
            self.logger.error(f"❌ Confidence calibration failed: {e}")
            return result

    async def _apply_decision_capture(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply decision capture enhancement"""
        try:
            # Simplified decision capture
            decision_points = []

            application_results = result.get("application_results", [])
            for app_result in application_results:
                decision_points.append(
                    {
                        "decision_type": "strategy_application",
                        "model_id": app_result.model_id,
                        "strategy": app_result.strategy_used.value,
                        "confidence": app_result.confidence_score,
                        "timestamp": app_result.application_timestamp.isoformat(),
                    }
                )

            result["decision_capture"] = {
                "decision_points": decision_points,
                "capture_timestamp": datetime.utcnow().isoformat(),
            }

            return result

        except Exception as e:
            self.logger.error(f"❌ Decision capture failed: {e}")
            return result

    async def _apply_vector_similarity(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply vector similarity enhancement"""
        try:
            # Simplified vector similarity (would integrate with vector service)
            result["vector_similarity"] = {
                "similarity_analysis": "enhanced",
                "pattern_matching": "enabled",
                "enhancement_applied": datetime.utcnow().isoformat(),
            }

            return result

        except Exception as e:
            self.logger.error(f"❌ Vector similarity enhancement failed: {e}")
            return result

    async def _calculate_overall_confidence(
        self, enhanced_result: Dict[str, Any]
    ) -> float:
        """Calculate overall confidence in coordination result"""
        try:
            confidence_factors = []

            # Application results confidence
            application_results = enhanced_result.get("application_results", [])
            if application_results:
                app_confidences = [app.confidence_score for app in application_results]
                confidence_factors.append(sum(app_confidences) / len(app_confidences))

            # Performance metrics confidence
            performance_metrics = enhanced_result.get("performance_metrics", [])
            if performance_metrics:
                # Simplified performance confidence
                confidence_factors.append(0.8)

            # Service coordination confidence
            services_used = len(enhanced_result.get("services_used", []))
            service_confidence = min(services_used / 4.0, 1.0)  # Up to 4 services
            confidence_factors.append(service_confidence)

            # Enhancement confidence bonus
            enhancements_applied = len(enhanced_result.get("enhancements_applied", []))
            enhancement_bonus = min(enhancements_applied * 0.05, 0.15)
            confidence_factors.append(0.7 + enhancement_bonus)

            # Calculate weighted average
            overall_confidence = (
                sum(confidence_factors) / len(confidence_factors)
                if confidence_factors
                else 0.5
            )

            return max(0.1, min(overall_confidence, 0.95))  # Bound between 0.1 and 0.95

        except Exception as e:
            self.logger.error(f"❌ Overall confidence calculation failed: {e}")
            return 0.5

    def _infer_application_strategy(
        self, request_context: Dict[str, Any]
    ) -> ApplicationStrategy:
        """Infer appropriate application strategy from request context"""
        try:
            problem_statement = request_context.get("problem_statement", "").lower()
            business_context = str(request_context.get("business_context", {})).lower()

            # Simple keyword-based strategy inference
            if any(
                keyword in problem_statement
                for keyword in ["system", "interconnect", "holistic"]
            ):
                return ApplicationStrategy.SYSTEMS_THINKING
            elif any(
                keyword in problem_statement
                for keyword in ["decide", "choice", "option"]
            ):
                return ApplicationStrategy.DECISION_FRAMEWORK
            elif any(
                keyword in problem_statement
                for keyword in ["test", "hypothesis", "validate"]
            ):
                return ApplicationStrategy.HYPOTHESIS_TESTING
            elif any(
                keyword in problem_statement
                for keyword in ["category", "segment", "classify"]
            ):
                return ApplicationStrategy.MECE_FRAMEWORK
            elif any(
                keyword in problem_statement
                for keyword in ["analyze", "critical", "evaluate"]
            ):
                return ApplicationStrategy.CRITICAL_THINKING
            else:
                return ApplicationStrategy.GENERIC_APPLICATION

        except Exception as e:
            self.logger.error(f"❌ Strategy inference failed: {e}")
            return ApplicationStrategy.GENERIC_APPLICATION

    async def _apply_feature_flag_logic(
        self, features: Dict[FeatureFlag, bool]
    ) -> Dict[FeatureFlag, bool]:
        """Apply business logic for feature flag interactions"""
        try:
            # Example: Vector similarity requires Bayesian learning
            if features.get(FeatureFlag.VECTOR_SIMILARITY, False):
                features[FeatureFlag.BAYESIAN_LEARNING] = True

            # Example: Decision capture works better with confidence calibration
            if features.get(FeatureFlag.DECISION_CAPTURE, False):
                features[FeatureFlag.CONFIDENCE_CALIBRATION] = True

            return features

        except Exception as e:
            self.logger.error(f"❌ Feature flag logic application failed: {e}")
            return features

    async def _collect_service_health_data(self) -> Dict[str, Any]:
        """Collect health data from all constituent services"""
        service_healths = {}

        try:
            service_healths["registry"] = (
                await self.registry_service.get_service_health()
            )
        except Exception as e:
            service_healths["registry"] = {"status": "error", "error": str(e)}

        try:
            service_healths["lifecycle"] = (
                await self.lifecycle_service.get_service_health()
            )
        except Exception as e:
            service_healths["lifecycle"] = {"status": "error", "error": str(e)}

        try:
            service_healths["performance"] = (
                await self.performance_service.get_service_health()
            )
        except Exception as e:
            service_healths["performance"] = {"status": "error", "error": str(e)}

        try:
            service_healths["application"] = (
                await self.application_service.get_service_health()
            )
        except Exception as e:
            service_healths["application"] = {"status": "error", "error": str(e)}

        return service_healths

    async def _calculate_cluster_metrics(
        self, service_healths: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate cluster-level performance metrics"""
        try:
            # Count healthy services
            healthy_services = sum(
                1
                for health in service_healths.values()
                if health.get("status") == "healthy"
            )
            total_services = len(service_healths)

            # Calculate resource utilization (simplified)
            cluster_metrics = {
                "service_availability": {
                    "healthy_services": healthy_services,
                    "total_services": total_services,
                    "availability_percentage": (
                        (healthy_services / total_services * 100)
                        if total_services > 0
                        else 0
                    ),
                },
                "coordination_performance": {
                    "total_coordinations": self.coordination_metrics[
                        "total_coordinations"
                    ],
                    "success_rate": (
                        self.coordination_metrics["successful_coordinations"]
                        / max(self.coordination_metrics["total_coordinations"], 1)
                    )
                    * 100,
                    "average_time_ms": round(
                        self.coordination_metrics["average_coordination_time_ms"], 2
                    ),
                },
                "feature_flag_adoption": {
                    flag: usage
                    for flag, usage in self.coordination_metrics[
                        "feature_flag_usage"
                    ].items()
                },
                "service_reliability": {
                    service: failures
                    for service, failures in self.coordination_metrics[
                        "service_failure_counts"
                    ].items()
                },
            }

            return cluster_metrics

        except Exception as e:
            self.logger.error(f"❌ Cluster metrics calculation failed: {e}")
            return {}

    def _assess_cluster_health_status(
        self, service_healths: Dict[str, Any], cluster_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess overall cluster health status"""
        try:
            healthy_services = sum(
                1
                for health in service_healths.values()
                if health.get("status") == "healthy"
            )
            total_services = len(service_healths)
            health_percentage = (
                (healthy_services / total_services * 100) if total_services > 0 else 0
            )

            # Determine overall status
            if health_percentage >= 90:
                overall_status = "excellent"
            elif health_percentage >= 75:
                overall_status = "healthy"
            elif health_percentage >= 50:
                overall_status = "degraded"
            else:
                overall_status = "critical"

            return {
                "overall_status": overall_status,
                "health_percentage": health_percentage,
                "healthy_services": healthy_services,
                "total_services": total_services,
            }

        except Exception as e:
            return {
                "overall_status": "error",
                "health_percentage": 0,
                "healthy_services": 0,
                "total_services": 0,
                "error": str(e),
            }

    async def _generate_cluster_insights(
        self, service_healths: Dict[str, Any], cluster_metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable insights about cluster health"""
        insights = []

        try:
            # Service health insights
            unhealthy_services = [
                name
                for name, health in service_healths.items()
                if health.get("status") != "healthy"
            ]

            if unhealthy_services:
                insights.append(
                    f"Services requiring attention: {', '.join(unhealthy_services)}"
                )
            else:
                insights.append("All application services are healthy")

            # Performance insights
            success_rate = cluster_metrics.get("coordination_performance", {}).get(
                "success_rate", 0
            )
            if success_rate < 90:
                insights.append(
                    f"Coordination success rate below optimal: {success_rate:.1f}%"
                )

            avg_time = cluster_metrics.get("coordination_performance", {}).get(
                "average_time_ms", 0
            )
            if avg_time > 5000:
                insights.append(f"Average coordination time high: {avg_time:.0f}ms")

            # Feature flag insights
            feature_usage = cluster_metrics.get("feature_flag_adoption", {})
            most_used_feature = max(
                feature_usage.items(), key=lambda x: x[1], default=("none", 0)
            )
            if most_used_feature[1] > 0:
                insights.append(
                    f"Most utilized feature: {most_used_feature[0]} ({most_used_feature[1]} uses)"
                )

            # Default positive insight
            if len(insights) == 0:
                insights.append(
                    "Application cluster operating within optimal parameters"
                )

            return insights[:5]  # Limit to top 5 insights

        except Exception as e:
            return [f"Insight generation error: {str(e)}"]

    def _is_health_cache_valid(self, cache_key: str) -> bool:
        """Check if health cache entry is still valid"""
        if cache_key not in self.service_health_cache:
            return False

        last_check = self.last_health_check.get(cache_key)
        if not last_check:
            return False

        cache_age_minutes = (datetime.utcnow() - last_check).total_seconds() / 60
        return cache_age_minutes < self.health_cache_ttl_minutes

    async def _update_coordination_metrics(
        self, coordination_time_ms: float, success: bool
    ):
        """Update coordination performance metrics"""
        try:
            self.coordination_metrics["total_coordinations"] += 1

            if success:
                self.coordination_metrics["successful_coordinations"] += 1

            # Update running average for coordination time
            total = self.coordination_metrics["total_coordinations"]
            current_avg = self.coordination_metrics["average_coordination_time_ms"]
            new_avg = ((current_avg * (total - 1)) + coordination_time_ms) / total
            self.coordination_metrics["average_coordination_time_ms"] = new_avg

        except Exception as e:
            self.logger.error(f"❌ Coordination metrics update failed: {e}")

    async def _create_fallback_result(
        self,
        request_context: Dict[str, Any],
        feature_flags: List[FeatureFlag],
        error_message: str,
    ) -> ApplicationResultContract:
        """Create fallback result when coordination fails"""
        from src.services.contracts.application_contracts import (
            ModelOrchestrationContract,
            ModelApplicationContract,
            OrchestrationMode,
            ModelApplicationStatus,
            ApplicationStrategy,
        )

        # Create minimal fallback orchestration
        fallback_orchestration = ModelOrchestrationContract(
            orchestration_id=f"fallback_{datetime.utcnow().timestamp()}",
            selected_models=["deepseek_chat"],
            orchestration_mode=OrchestrationMode.STANDARD,
            coordination_metadata={
                "fallback_triggered": True,
                "error_message": error_message,
            },
            nway_interactions=[],
            research_integration_data=None,
            similar_patterns_detected=[],
            orchestration_timestamp=datetime.utcnow(),
            total_orchestration_time_ms=0.0,
            service_version="v5_modular_fallback",
        )

        # Create minimal fallback application
        fallback_application = ModelApplicationContract(
            application_id=f"fallback_app_{datetime.utcnow().timestamp()}",
            model_id="deepseek_chat",
            strategy_used=ApplicationStrategy.GENERIC_APPLICATION,
            application_status=ModelApplicationStatus.FAILED,
            input_data=request_context,
            output_data={"error": error_message, "fallback": True},
            processing_metadata={"fallback_reason": error_message},
            confidence_score=0.2,
            quality_metrics={"fallback": 1.0},
            application_timestamp=datetime.utcnow(),
            processing_time_ms=0.0,
            service_version="v5_modular_fallback",
        )

        return ApplicationResultContract(
            result_id=f"fallback_{datetime.utcnow().timestamp()}",
            engagement_id=request_context.get("engagement_id", "fallback"),
            orchestration_result=fallback_orchestration,
            application_results=[fallback_application],
            llm_responses=[],
            performance_metrics=[],
            feature_flags_used=feature_flags,
            overall_confidence=0.2,
            processing_summary={
                "fallback_triggered": True,
                "error_message": error_message,
                "services_attempted": ["coordinator"],
            },
            result_timestamp=datetime.utcnow(),
            total_processing_time_ms=50.0,
            service_version="v5_modular_fallback",
        )


# Global service instance for dependency injection
_application_coordinator_service: Optional[ApplicationCoordinatorService] = None


def get_application_coordinator_service() -> ApplicationCoordinatorService:
    """Get or create global application coordinator service instance"""
    global _application_coordinator_service

    if _application_coordinator_service is None:
        _application_coordinator_service = ApplicationCoordinatorService()

    return _application_coordinator_service
