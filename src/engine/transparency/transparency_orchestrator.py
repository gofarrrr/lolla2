"""
METIS V5 Transparency Engine Refactoring - Target #3
TransparencyOrchestrator

Extracted from transparency_engine.py (1,902 lines → modular architecture)
Lean orchestrator coordinating transparency services with dependency injection

Single Responsibility: Transparency service orchestration and coordination
"""

import logging
from typing import Dict, Optional, Any
from uuid import UUID

from src.engine.models.data_contracts import MetisDataContract
from src.models.transparency_models import (
    UserProfile,
    ProgressiveDisclosure,
    TransparencyLayer,
    UserExpertiseLevel,
)

# Import our modular services
from .services.cognitive_scaffolding_service import (
    get_cognitive_scaffolding_service,
)
from .services.user_expertise_service import (
    get_user_expertise_service,
)

# Optional integrations (graceful degradation)
try:
    from src.engine.adapters.monitoring import  # Migrated CognitiveProfiler

    COGNITIVE_PROFILER_AVAILABLE = True
except ImportError:
    COGNITIVE_PROFILER_AVAILABLE = False
    CognitiveProfiler = None

try:
    from src.ui.cognitive_trace_visualizer import (
        CognitiveTraceBuilder,
        CognitiveTraceRenderer,
    )

    TRACE_VISUALIZER_AVAILABLE = True
except ImportError:
    TRACE_VISUALIZER_AVAILABLE = False
    CognitiveTraceBuilder = None
    CognitiveTraceRenderer = None


class TransparencyOrchestrator:
    """
    Lean orchestrator coordinating transparency services with dependency injection.

    Replaces the monolithic AdaptiveTransparencyEngine with clean service composition,
    following the same successful pattern as ConsultantOrchestrator.
    """

    def __init__(self):
        """Initialize orchestrator with all transparency services"""
        self.logger = logging.getLogger(__name__)

        # Initialize core services with dependency injection
        self._initialize_services()

        # User profiles (in production, would be stored in database)
        self.user_profiles: Dict[UUID, UserProfile] = {}

        # Performance metrics
        self._performance_metrics: Dict[str, int] = {}

        self.logger.info(
            "✅ TransparencyOrchestrator: Initialized with modular service architecture"
        )

    def _initialize_services(self):
        """Initialize all modular services with dependency injection"""

        # Core transparency services
        self.cognitive_scaffolding = get_cognitive_scaffolding_service()
        self.user_expertise = get_user_expertise_service()

        # Optional enhanced services (graceful degradation)
        if COGNITIVE_PROFILER_AVAILABLE:
            self.cognitive_profiler = CognitiveProfiler()
            self.adaptive_profiling_enabled = True
            self.logger.info(
                "✅ Cognitive profiler initialized - adaptive intelligence enabled"
            )
        else:
            self.cognitive_profiler = None
            self.adaptive_profiling_enabled = False
            self.logger.warning(
                "⚠️ Cognitive profiler not available - using static expertise assessment"
            )

        if TRACE_VISUALIZER_AVAILABLE:
            self.trace_builder = CognitiveTraceBuilder()
            self.trace_renderer = CognitiveTraceRenderer()
            self.logger.info("✅ Cognitive trace visualization available")
        else:
            self.trace_builder = None
            self.trace_renderer = None
            self.logger.info("ℹ️ Cognitive trace visualization not available")

        self.logger.info("✅ TransparencyOrchestrator: All services initialized")

    async def generate_progressive_disclosure(
        self,
        engagement_contract: MetisDataContract,
        user_id: UUID,
        session_id: Optional[str] = None,
    ) -> ProgressiveDisclosure:
        """Generate complete progressive disclosure package with adaptive intelligence"""

        start_time = self._start_performance_timer("total_transparency_generation")

        try:
            # Get or create user profile
            user_profile = await self._get_user_profile(user_id)

            # Assess user expertise using modular service
            current_expertise = await self.user_expertise.assess_expertise(user_profile)
            if current_expertise != user_profile.expertise_level:
                user_profile.expertise_level = current_expertise
                self.logger.info(
                    f"Updated user {user_id} expertise level to {current_expertise}"
                )

            # Assess cognitive load using scaffolding service
            cognitive_load = await self.cognitive_scaffolding.assess_cognitive_load(
                str(engagement_contract),  # Convert contract to string for assessment
                engagement_contract.cognitive_state.reasoning_steps,
                engagement_contract.cognitive_state.mental_models,
            )

            # Apply scaffolding based on user profile and cognitive load
            scaffolding = await self.cognitive_scaffolding.apply_scaffolding(
                str(engagement_contract), user_profile, cognitive_load
            )

            # Create progressive disclosure result
            progressive_disclosure = ProgressiveDisclosure(
                user_id=user_id,
                session_id=session_id,
                expertise_level=user_profile.expertise_level,
                cognitive_load=cognitive_load,
                scaffolding=scaffolding,
                layers={
                    TransparencyLayer.EXECUTIVE_SUMMARY: await self._generate_executive_layer(
                        engagement_contract
                    ),
                    TransparencyLayer.REASONING_OVERVIEW: await self._generate_reasoning_layer(
                        engagement_contract
                    ),
                    TransparencyLayer.DETAILED_AUDIT_TRAIL: await self._generate_detailed_layer(
                        engagement_contract
                    ),
                    TransparencyLayer.TECHNICAL_EXECUTION: await self._generate_technical_layer(
                        engagement_contract
                    ),
                },
                recommended_starting_layer=self._determine_starting_layer(
                    user_profile.expertise_level
                ),
                performance_metrics=self._performance_metrics.copy(),
            )

            return progressive_disclosure

        finally:
            self._end_performance_timer("total_transparency_generation", start_time)

    async def _get_user_profile(self, user_id: UUID) -> UserProfile:
        """Get or create user profile"""
        if user_id not in self.user_profiles:
            # Create default profile
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                expertise_level=UserExpertiseLevel.ANALYTICAL,  # Default to analytical
                preferred_layer=TransparencyLayer.REASONING_OVERVIEW,
                interaction_history=[],
                cognitive_preferences={},
            )

        return self.user_profiles[user_id]

    def _determine_starting_layer(
        self, expertise_level: UserExpertiseLevel
    ) -> TransparencyLayer:
        """Determine recommended starting layer based on expertise"""
        layer_mapping = {
            UserExpertiseLevel.EXECUTIVE: TransparencyLayer.EXECUTIVE_SUMMARY,
            UserExpertiseLevel.STRATEGIC: TransparencyLayer.REASONING_OVERVIEW,
            UserExpertiseLevel.ANALYTICAL: TransparencyLayer.DETAILED_AUDIT_TRAIL,
            UserExpertiseLevel.TECHNICAL: TransparencyLayer.TECHNICAL_EXECUTION,
        }

        return layer_mapping.get(expertise_level, TransparencyLayer.REASONING_OVERVIEW)

    async def _generate_executive_layer(
        self, contract: MetisDataContract
    ) -> Dict[str, Any]:
        """Generate executive summary layer"""
        return {
            "type": "executive_summary",
            "content": "Executive summary generated from engagement contract",
            "key_insights": [],
            "business_impact": {},
            "recommendations": [],
        }

    async def _generate_reasoning_layer(
        self, contract: MetisDataContract
    ) -> Dict[str, Any]:
        """Generate reasoning overview layer"""
        return {
            "type": "reasoning_overview",
            "content": "Reasoning overview generated from engagement contract",
            "methodology": [],
            "decision_points": [],
            "confidence_levels": {},
        }

    async def _generate_detailed_layer(
        self, contract: MetisDataContract
    ) -> Dict[str, Any]:
        """Generate detailed audit trail layer"""
        return {
            "type": "detailed_audit_trail",
            "content": "Detailed audit trail generated from engagement contract",
            "reasoning_steps": contract.cognitive_state.reasoning_steps,
            "evidence": [],
            "validation_results": {},
        }

    async def _generate_technical_layer(
        self, contract: MetisDataContract
    ) -> Dict[str, Any]:
        """Generate technical execution layer"""
        return {
            "type": "technical_execution",
            "content": "Technical execution details generated from engagement contract",
            "implementation_details": [],
            "system_interactions": [],
            "performance_metrics": self._performance_metrics,
        }

    def _start_performance_timer(self, operation_name: str) -> float:
        """Start performance timer for operation"""
        import time

        return time.perf_counter()

    def _end_performance_timer(self, operation_name: str, start_time: float):
        """End performance timer and record metrics"""
        import time

        duration_ms = int((time.perf_counter() - start_time) * 1000)
        self._performance_metrics[operation_name] = duration_ms
        self.logger.info(
            f"🕒 Performance: {operation_name} completed in {duration_ms}ms"
        )

    def get_capabilities(self) -> Dict[str, Any]:
        """Get orchestrator capabilities and service status"""
        return {
            "orchestrator_version": "v5_modular",
            "services_available": {
                "cognitive_scaffolding": bool(self.cognitive_scaffolding),
                "user_expertise": bool(self.user_expertise),
                "cognitive_profiler": self.adaptive_profiling_enabled,
                "trace_visualizer": bool(self.trace_builder),
            },
            "transparency_layers_supported": [
                layer.value for layer in TransparencyLayer
            ],
            "user_profiles_cached": len(self.user_profiles),
            "performance_metrics": self._performance_metrics,
        }


# Factory function for orchestrator creation
def get_transparency_orchestrator() -> TransparencyOrchestrator:
    """Factory function to create TransparencyOrchestrator instance"""
    return TransparencyOrchestrator()
