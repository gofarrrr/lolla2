"""
METIS Cognitive Platform - Main Application Entry Point
Orchestrates all foundation components into unified cognitive intelligence platform

Based on PRD v7 and architecturally ratified specifications.
"""

import asyncio
import logging
import signal
import sys
from typing import Dict, Any, Optional
from datetime import datetime
from uuid import UUID, uuid4

# Import all foundation components
from models.data_contracts import create_engagement_initiated_event
from src.core.event_bus import get_event_bus
from src.core.audit_trail import get_audit_manager, AuditEventType, AuditSeverity
from src.engine.core.system_health_validator import (
    get_system_health_validator,
)

# UPDATED: Using refactored modular components instead of monolithic engine
from src.factories.engine_factory import CognitiveEngineFactory

# from src.engines.components.integration_orchestrator import IntegrationOrchestrator, ServiceRegistry  # Missing file
# TEMP DISABLED - # TEMP DISABLED - from src.ui import get_transparency_engine, generate_user_transparency
# UPDATED: Using Supabase authentication instead of custom auth
from src.engine.api.supabase_foundation import (
    get_fastapi_app,
)
from src.core.auth_foundation import get_auth_manager
from src.engine.api.foundation import get_api_foundation

# Week 2 Day 5: Manual Override API
try:
    from src.engine.api.manual_override_api import router as manual_override_router

    MANUAL_OVERRIDE_API_AVAILABLE = True
except ImportError:
    MANUAL_OVERRIDE_API_AVAILABLE = False

# Operation Crystal - Prompt 5: Streaming Transparency API
try:
    from src.engine.api.transparency_streaming_api import (
        router as transparency_streaming_router,
    )

    TRANSPARENCY_STREAMING_AVAILABLE = True
except ImportError:
    TRANSPARENCY_STREAMING_AVAILABLE = False

# Progressive Questions API
try:
    from src.engine.api.progressive_questions import (
        router as progressive_questions_router,
    )

    PROGRESSIVE_QUESTIONS_AVAILABLE = True
except ImportError:
    PROGRESSIVE_QUESTIONS_AVAILABLE = False

# User Journey API
try:
    from src.engine.api.user_journey_facade import router as user_journey_router

    USER_JOURNEY_AVAILABLE = True
except ImportError:
    USER_JOURNEY_AVAILABLE = False

# Engagement API
try:
    from src.engine.api.engagement.routes import router as engagement_router

    ENGAGEMENT_API_AVAILABLE = True
except ImportError:
    ENGAGEMENT_API_AVAILABLE = False

# Three-Consultant API
try:
    from src.engine.api.three_consultant_api import router as three_consultant_router

    THREE_CONSULTANT_API_AVAILABLE = True
except ImportError:
    THREE_CONSULTANT_API_AVAILABLE = False

# Strategic Trio Critique API
try:
    from src.engine.api.strategic_trio_critique_api import (
        router as strategic_trio_router,
    )

    STRATEGIC_TRIO_API_AVAILABLE = True
except ImportError:
    STRATEGIC_TRIO_API_AVAILABLE = False

# Senior Advisor API
try:
    from src.engine.api.senior_advisor_api import router as senior_advisor_router

    SENIOR_ADVISOR_API_AVAILABLE = True
except ImportError:
    SENIOR_ADVISOR_API_AVAILABLE = False

# Socratic Forge API (Week 3 - V5 Final Implementation)
try:
    from src.engine.api.socratic_forge_api import router as socratic_forge_router

    SOCRATIC_FORGE_API_AVAILABLE = True
except ImportError:
    SOCRATIC_FORGE_API_AVAILABLE = False

# Analysis Execution API (Phase 1 - Critical Fix)
try:
    from src.engine.api.analysis_execution_api import (
        router as analysis_execution_router,
    )

    ANALYSIS_EXECUTION_API_AVAILABLE = True
except ImportError:
    ANALYSIS_EXECUTION_API_AVAILABLE = False

# Devil's Advocate API (Phase 2 - Quality Control)
try:
    from src.engine.api.devils_advocate_api import router as devils_advocate_router

    DEVILS_ADVOCATE_API_AVAILABLE = True
    print("🔥 Devil's Advocate API imported successfully")
except ImportError as e:
    DEVILS_ADVOCATE_API_AVAILABLE = False
    print(f"❌ Devil's Advocate API import failed: {e}")

# Level 3 Enhancement: Markdown Output API ("Intelligence as an Asset")
try:
    from src.engine.api.markdown_output_api import router as markdown_output_router

    MARKDOWN_OUTPUT_API_AVAILABLE = True
    print("📄 Markdown Output API imported successfully")
except ImportError as e:
    MARKDOWN_OUTPUT_API_AVAILABLE = False
    print(f"❌ Markdown Output API import failed: {e}")

# FLYWHEEL: Internal Management API for Admin Dashboard
try:
    from src.engine.api.flywheel_management_api import flywheel_management_router

    FLYWHEEL_MANAGEMENT_API_AVAILABLE = True
    print("🧠 Flywheel Management API imported successfully")
except ImportError as e:
    FLYWHEEL_MANAGEMENT_API_AVAILABLE = False
    print(f"❌ Flywheel Management API import failed: {e}")

# FINAL ASSEMBLY: Engagement Results API (Complete Final API Payload)
try:
    from src.engine.api.engagement_results_api import (
        router as engagement_results_router,
    )

    ENGAGEMENT_RESULTS_API_AVAILABLE = True
    print("🎯 Final Assembly: Engagement Results API imported successfully")
except ImportError as e:
    ENGAGEMENT_RESULTS_API_AVAILABLE = False
    print(f"❌ Engagement Results API import failed: {e}")

# PROOF OF WORK METRICS: Platform Statistics API
try:
    from src.engine.api.platform_stats_api import router as platform_stats_router

    PLATFORM_STATS_API_AVAILABLE = True
    print("📊 Proof of Work: Platform Statistics API imported successfully")
except ImportError as e:
    PLATFORM_STATS_API_AVAILABLE = False
    print(f"❌ Platform Statistics API import failed: {e}")

# FACULTY SHOWCASE: Public-Facing Consultant Registry
try:
    from src.engine.api.public_showcase_api import router as public_showcase_router

    PUBLIC_SHOWCASE_API_AVAILABLE = True
    print("🎭 Faculty Showcase: Public Consultant Registry API imported successfully")
except ImportError as e:
    PUBLIC_SHOWCASE_API_AVAILABLE = False
    print(f"❌ Public Showcase API import failed: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("metis_platform.log"), logging.StreamHandler()],
)


class MetisCognitivePlatform:
    """
    Main METIS Cognitive Platform orchestrator
    Integrates all foundation components into cohesive system with dependency injection
    """

    def __init__(
        self,
        settings: Optional["MetisSettings"] = None,
        cognitive_engine: Optional["ICognitiveEngine"] = None,
        event_bus: Optional["IEventBus"] = None,
    ):
        """
        Initialize platform with dependency injection

        Args:
            settings: Optional centralized configuration
            cognitive_engine: Optional cognitive engine instance
            event_bus: Optional event bus instance
        """
        # Import dependencies
        from src.config import get_settings

        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self.startup_time: Optional[datetime] = None

        # Inject settings dependency
        if settings is not None:
            self.settings = settings
        else:
            self.settings = get_settings()

        # Store injected components for later initialization
        self._injected_cognitive_engine = cognitive_engine
        self._injected_event_bus = event_bus

        # Component health tracking
        self.component_health: Dict[str, bool] = {
            "event_bus": False,
            "auth_manager": False,
            "audit_manager": False,
            "cognitive_engine": False,
            "transparency_engine": False,
            "api_foundation": False,
        }

        # Performance metrics
        self.metrics = {
            "total_engagements": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "avg_processing_time_ms": 0.0,
            "uptime_seconds": 0,
        }

    async def initialize(self) -> bool:
        """Initialize all platform components"""

        self.logger.info("🚀 METIS Cognitive Platform - Initialization Starting")
        self.startup_time = datetime.utcnow()

        try:
            # Initialize Event Bus
            self.logger.info("📡 Initializing Event Bus...")
            self.event_bus = await get_event_bus()
            health = await self.event_bus.get_health_status()
            self.component_health["event_bus"] = health["status"] == "ready"
            self.logger.info(
                f"Event Bus: {'✅ Ready' if self.component_health['event_bus'] else '❌ Failed'}"
            )

            # Initialize Authentication Manager
            self.logger.info("🔐 Initializing Authentication Manager...")
            self.auth_manager = await get_auth_manager()
            auth_health = await self.auth_manager.get_auth_health_status()
            self.component_health["auth_manager"] = (
                auth_health["system_status"] == "healthy"
            )
            self.logger.info(
                f"Authentication: {'✅ Ready' if self.component_health['auth_manager'] else '❌ Failed'}"
            )

            # Initialize Audit Trail Manager
            self.logger.info("📋 Initializing Audit Trail Manager...")
            self.audit_manager = await get_audit_manager()
            audit_health = await self.audit_manager.get_audit_health_status()
            self.component_health["audit_manager"] = (
                audit_health["storage_health"] == "healthy"
            )
            self.logger.info(
                f"Audit Trail: {'✅ Ready' if self.component_health['audit_manager'] else '❌ Failed'}"
            )

            # Initialize Cognitive Engine (using injected instance if available)
            self.logger.info("🧠 Initializing Cognitive Engine...")
            if self._injected_cognitive_engine is not None:
                cognitive_engine = self._injected_cognitive_engine
                self.logger.info("Using injected cognitive engine instance")
            else:
                # UPDATED: Initialize modular cognitive engine with dependency injection
                cognitive_engine = CognitiveEngineFactory.create_engine(
                    config={
                        "enable_hmw_generation": True,
                        "enable_assumption_challenging": True,
                    }
                )

            # Store cognitive engine instance
            self.cognitive_engine = cognitive_engine

            engine_status = cognitive_engine.get_status()
            self.component_health["cognitive_engine"] = engine_status[
                "engine_type"
            ] in ["modular", "modular_v2_vulnerability_enhanced"]
            active_components = sum(
                1
                for status in engine_status["components"].values()
                if status == "active"
            )
            self.logger.info(
                f"Cognitive Engine: {'✅ Ready' if self.component_health['cognitive_engine'] else '❌ Failed'} ({active_components} components active)"
            )

            # Initialize Transparency Engine
            self.logger.info("🔍 Initializing Transparency Engine...")
            transparency_engine = await get_transparency_engine()
            self.component_health["transparency_engine"] = True  # Always available
            self.logger.info("Transparency Engine: ✅ Ready")

            # Initialize API Foundation
            self.logger.info("🌐 Initializing API Foundation...")
            try:
                api_foundation = get_api_foundation()
                self.component_health["api_foundation"] = True
                self.logger.info("API Foundation: ✅ Ready")
            except ImportError:
                self.component_health["api_foundation"] = False
                self.logger.warning(
                    "API Foundation: ⚠️ FastAPI not available - API disabled"
                )

            # Log system startup
            await self.audit_manager.log_event(
                event_type=AuditEventType.SYSTEM_START,
                severity=AuditSeverity.MEDIUM,
                action_performed="platform_startup",
                event_description="METIS Cognitive Platform initialized successfully",
                metadata={
                    "component_health": self.component_health,
                    "startup_time": self.startup_time.isoformat(),
                    "version": "1.0.0",
                },
            )

            # Verify critical components (event_bus made optional for local development)
            critical_components = ["auth_manager", "audit_manager", "cognitive_engine"]
            all_critical_healthy = all(
                self.component_health[comp] for comp in critical_components
            )

            if all_critical_healthy:
                self.is_running = True
                self.logger.info(
                    "🎉 METIS Cognitive Platform - Initialization Complete!"
                )
                self.logger.info(
                    f"Platform Status: {'🟢 All Systems Operational' if all(self.component_health.values()) else '🟡 Some Components Degraded'}"
                )
                return True
            else:
                self.logger.error("❌ Critical component initialization failed")
                return False

        except Exception as e:
            self.logger.error(f"💥 Platform initialization failed: {e}", exc_info=True)
            return False

    async def process_engagement(
        self,
        problem_statement: str,
        business_context: Dict[str, Any] = None,
        user_id: Optional[UUID] = None,
        session_id: Optional[UUID] = None,
    ) -> Dict[str, Any]:
        """
        Process complete engagement from problem to transparent results
        Main cognitive intelligence workflow
        """

        if not self.is_running:
            raise RuntimeError("Platform not initialized")

        start_time = datetime.utcnow()
        engagement_id = uuid4()

        try:
            self.logger.info(f"🎯 Processing engagement {engagement_id}")

            # Create engagement event
            engagement_event = create_engagement_initiated_event(
                problem_statement=problem_statement,
                business_context=business_context or {},
            )
            engagement_event.engagement_context.engagement_id = engagement_id

            # Log engagement creation
            audit_manager = await get_audit_manager()
            await audit_manager.log_event(
                event_type=AuditEventType.ENGAGEMENT_CREATED,
                severity=AuditSeverity.MEDIUM,
                user_id=user_id,
                session_id=session_id,
                engagement_id=engagement_id,
                action_performed="create_engagement",
                event_description=f"Created engagement: {problem_statement[:100]}...",
                metadata={
                    "problem_length": len(problem_statement),
                    "has_business_context": bool(business_context),
                },
            )

            # METIS V5 Great Refactoring - Target #4 Complete: Use modern ConsultantOrchestrator
            self.logger.info(
                "🧠 CONSULTANT ORCHESTRATOR: V5 modular execution with clean services..."
            )
            from src.engine.engines.core.consultant_orchestrator import (
                get_consultant_orchestrator,
            )

            # Create V5 consultant orchestrator with modular services
            consultant_orchestrator = get_consultant_orchestrator()

            # Extract query from engagement context for V5 processing
            query = engagement_event.engagement_context.get("query", "")
            engagement_id = engagement_event.engagement_context.get("engagement_id")

            self.logger.info(f"📝 Processing query: {query[:100]}...")
            self.logger.info(
                f"🔧 V5 services initialized: {consultant_orchestrator.get_capabilities()}"
            )

            # Use V5 ConsultantOrchestrator instead of legacy StateMachineOrchestrator
            # This provides the same functionality with clean modular architecture

            # Execute through V5 consultant orchestrator (replaces legacy state machine)
            result = await consultant_orchestrator.process_query(
                query=query,
                engagement_id=engagement_id,
                context=engagement_event.engagement_context,
            )

            # Log reasoning process - V5 adaptation (ConsultantOrchestrator result → legacy format)
            reasoning_steps = (
                []
            )  # V5 orchestrator tracks reasoning through UnifiedContextStream
            if hasattr(result, "processing_summary"):
                # Create basic reasoning step from V5 result for audit trail compatibility
                reasoning_steps = [
                    {
                        "step_id": "v5_processing",
                        "description": f"V5 processing completed in {result.processing_summary.get('total_duration_ms', 0)}ms",
                        "consultants": result.selected_consultants,
                    }
                ]

            await audit_manager.log_reasoning_trace(
                engagement_id=engagement_id,
                reasoning_steps=reasoning_steps,
                user_id=user_id,
                session_id=session_id,
            )

            # Generate Progressive Transparency - V5 Adaptation
            self.logger.info("🔍 Generating progressive transparency...")

            # V5 uses modular TransparencyOrchestrator instead of monolithic transparency engine
            from src.engine.transparency.transparency_orchestrator import (
                get_transparency_orchestrator,
            )

            transparency_orchestrator = get_transparency_orchestrator()

            # Create minimal contract adapter for transparency engine compatibility
            from src.engine.models.data_contracts import (
                MetisDataContract,
                CognitiveState,
            )

            minimal_contract = MetisDataContract(
                engagement_context=engagement_event.engagement_context,
                cognitive_state=CognitiveState(
                    reasoning_steps=reasoning_steps, mental_models=[]
                ),
                workflow_state=engagement_event.workflow_state,
                type="v5_consultant_result",
                source="consultant_orchestrator",
            )

            # Use default user if none provided
            transparency_user_id = user_id or uuid4()
            progressive_disclosure = (
                await transparency_orchestrator.generate_progressive_disclosure(
                    minimal_contract, transparency_user_id
                )
            )

            # Calculate processing metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Update platform metrics
            self.metrics["total_engagements"] += 1
            self.metrics["successful_analyses"] += 1
            self.metrics["avg_processing_time_ms"] = (
                self.metrics["avg_processing_time_ms"]
                * (self.metrics["successful_analyses"] - 1)
                + processing_time
            ) / self.metrics["successful_analyses"]

            # Prepare results
            results = {
                "engagement_id": str(engagement_id),
                "status": "completed",
                "processing_time_ms": processing_time,
                "cognitive_analysis": {
                    "selected_models": [
                        model.dict()
                        for model in processed_contract.cognitive_state.selected_mental_models
                    ],
                    "reasoning_steps": [
                        step.dict()
                        for step in processed_contract.cognitive_state.reasoning_steps
                    ],
                    "confidence_scores": processed_contract.cognitive_state.confidence_scores,
                    "validation_results": processed_contract.cognitive_state.validation_results,
                },
                "progressive_disclosure": {
                    "layers": {
                        layer.value: {
                            "title": content.title,
                            "cognitive_load": content.cognitive_load.value,
                            "key_insights": content.key_insights,
                            "content_preview": (
                                content.content[:200] + "..."
                                if len(content.content) > 200
                                else content.content
                            ),
                        }
                        for layer, content in progressive_disclosure.layers.items()
                    },
                    "navigation_guidance": progressive_disclosure.navigation_guidance,
                    "personalization": progressive_disclosure.personalization_metadata,
                },
                "metadata": {
                    "created_at": start_time.isoformat(),
                    "completed_at": datetime.utcnow().isoformat(),
                    "schema_version": processed_contract.schema_version,
                    "platform_version": "1.0.0",
                },
            }

            self.logger.info(
                f"✅ Engagement {engagement_id} completed successfully ({processing_time:.1f}ms)"
            )
            return results

        except Exception as e:
            # Log failure
            self.metrics["failed_analyses"] += 1

            audit_manager = await get_audit_manager()
            await audit_manager.log_event(
                event_type=AuditEventType.SYSTEM_ERROR,
                severity=AuditSeverity.CRITICAL,
                user_id=user_id,
                session_id=session_id,
                engagement_id=engagement_id,
                action_performed="process_engagement",
                event_description=f"Engagement processing failed: {str(e)}",
                metadata={"error": str(e), "error_type": type(e).__name__},
            )

            self.logger.error(
                f"❌ Engagement {engagement_id} failed: {e}", exc_info=True
            )
            raise

    async def get_platform_health(self) -> Dict[str, Any]:
        """Get comprehensive platform health status with honest assessment"""

        uptime = (
            (datetime.utcnow() - self.startup_time).total_seconds()
            if self.startup_time
            else 0
        )
        self.metrics["uptime_seconds"] = uptime

        # Use real health validator instead of fake status
        health_validator = await get_system_health_validator()
        real_health = await health_validator.validate_system_health()

        # Legacy component status (for compatibility)
        component_details = {
            "warning": "Component health below shows framework status only - see real_health for actual implementation status"
        }

        if self.component_health["event_bus"]:
            event_bus = await get_event_bus()
            component_details["event_bus"] = await event_bus.get_health_status()

        # Return honest health assessment
        return {
            "platform_status": real_health["overall_status"],
            "is_running": self.is_running,
            "startup_time": (
                self.startup_time.isoformat() if self.startup_time else None
            ),
            "uptime_seconds": uptime,
            # Real implementation status
            "real_health": real_health,
            "implementation_percentage": real_health["overall_implementation"],
            "production_ready": real_health["production_ready"],
            "missing_integrations": real_health["missing_integrations"],
            "implementation_gaps": real_health["implementation_gaps"],
            "next_steps": real_health["next_steps"],
            # Legacy fields (for compatibility)
            "component_health": self.component_health,
            "component_details": component_details,
            "performance_metrics": self.metrics,
            "version": "7.0.0-FRAMEWORK",
            "compliance_status": {
                "soc2_compliant": False,  # Can't be compliant without real processing
                "gdpr_compliant": False,  # Can't be compliant without real data handling
                "audit_trail_active": self.component_health["audit_manager"],
                "note": "Compliance impossible without real LLM integration",
            },
        }

    async def shutdown(self):
        """Graceful platform shutdown"""

        self.logger.info("🛑 METIS Cognitive Platform - Shutdown Initiated")

        try:
            # Log shutdown
            if self.component_health["audit_manager"]:
                audit_manager = await get_audit_manager()
                await audit_manager.log_event(
                    event_type=AuditEventType.SYSTEM_START,  # No shutdown event type defined
                    severity=AuditSeverity.MEDIUM,
                    action_performed="platform_shutdown",
                    event_description="METIS Cognitive Platform shutdown initiated",
                    metadata={
                        "uptime_seconds": self.metrics["uptime_seconds"],
                        "total_engagements": self.metrics["total_engagements"],
                    },
                )

            # Shutdown event bus
            if self.component_health["event_bus"]:
                event_bus = await get_event_bus()
                await event_bus.shutdown()

            self.is_running = False
            self.logger.info("✅ METIS Cognitive Platform - Shutdown Complete")

        except Exception as e:
            self.logger.error(f"❌ Error during shutdown: {e}", exc_info=True)


# Global platform instance
_platform_instance: Optional[MetisCognitivePlatform] = None


async def get_platform(
    settings: Optional["MetisSettings"] = None,
    cognitive_engine: Optional["MetisCognitiveEngine"] = None,
    event_bus: Optional["IEventBus"] = None,
) -> MetisCognitivePlatform:
    """
    Get or create global platform instance with dependency injection

    Args:
        settings: Optional centralized configuration
        cognitive_engine: Optional cognitive engine instance
        event_bus: Optional event bus instance
    """
    global _platform_instance

    if _platform_instance is None:
        _platform_instance = MetisCognitivePlatform(
            settings=settings, cognitive_engine=cognitive_engine, event_bus=event_bus
        )

    return _platform_instance


async def create_platform(
    settings: "MetisSettings",
    cognitive_engine: "MetisCognitiveEngine",
    event_bus: "IEventBus",
) -> MetisCognitivePlatform:
    """
    Create a new platform instance with explicit dependencies

    This function enables true dependency injection for testing and custom configurations
    """
    return MetisCognitivePlatform(
        settings=settings, cognitive_engine=cognitive_engine, event_bus=event_bus
    )


# CLI Interface for testing and development
async def main():
    """Main entry point for CLI testing"""

    print("🚀 METIS Cognitive Platform v1.0.0")
    print("=" * 50)

    # Initialize platform
    platform = await get_platform()

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\n📡 Received signal {signum}, shutting down...")
        asyncio.create_task(platform.shutdown())
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize
    if not await platform.initialize():
        print("❌ Platform initialization failed")
        sys.exit(1)

    # Interactive mode
    print("\n🎯 Enter engagement problems (or 'quit' to exit):")
    print("💡 Example: 'How can we improve customer retention in our SaaS platform?'")

    try:
        while True:
            print("\n" + "─" * 50)
            problem = input("🤔 Problem Statement: ").strip()

            if problem.lower() in ["quit", "exit", "q"]:
                break

            if not problem:
                continue

            try:
                print("\n⏳ Processing engagement...")

                results = await platform.process_engagement(
                    problem_statement=problem,
                    business_context={"source": "cli_interface"},
                )

                print(f"\n✅ Analysis Complete! (ID: {results['engagement_id']})")
                print(f"⚡ Processing Time: {results['processing_time_ms']:.1f}ms")

                # Show cognitive analysis summary
                cognitive = results["cognitive_analysis"]
                print("\n🧠 Cognitive Analysis:")
                print(f"   📊 Models Applied: {len(cognitive['selected_models'])}")
                print(f"   🔍 Reasoning Steps: {len(cognitive['reasoning_steps'])}")
                print(
                    f"   📈 Avg Confidence: {sum(cognitive['confidence_scores'].values()) / len(cognitive['confidence_scores']) if cognitive['confidence_scores'] else 0:.1%}"
                )

                # Show transparency layers
                disclosure = results["progressive_disclosure"]
                print("\n🔍 Transparency Layers Available:")
                for layer_name, layer_info in disclosure["layers"].items():
                    load_indicator = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(
                        layer_info["cognitive_load"], "⚪"
                    )
                    print(
                        f"   {load_indicator} {layer_info['title']} ({layer_info['cognitive_load']} complexity)"
                    )

                # Show key insights
                exec_summary = disclosure["layers"].get("executive_summary", {})
                if exec_summary.get("key_insights"):
                    print("\n💡 Key Insights:")
                    for insight in exec_summary["key_insights"][:3]:
                        print(f"   • {insight}")

            except Exception as e:
                print(f"❌ Error processing engagement: {e}")

    except KeyboardInterrupt:
        pass

    finally:
        print("\n🛑 Shutting down platform...")
        await platform.shutdown()
        print("👋 Goodbye!")


# Export FastAPI app for deployment with Supabase authentication
try:
    # Use new Supabase-based API foundation
    app = get_fastapi_app()
    print("✅ METIS API with Supabase Authentication initialized")

    # Week 2 Day 5: Register Manual Override API
    if MANUAL_OVERRIDE_API_AVAILABLE:
        app.include_router(manual_override_router)
        print("✅ Week 2 Day 5: Manual Override API registered")
    else:
        print("⚠️ Manual Override API not available")

    # Operation Crystal - Prompt 5: Register Streaming Transparency API
    if TRANSPARENCY_STREAMING_AVAILABLE:
        app.include_router(transparency_streaming_router)
        print("✅ Operation Crystal - Prompt 5: Streaming Transparency API registered")
    else:
        print("⚠️ Streaming Transparency API not available")

    # Register Progressive Questions API
    if PROGRESSIVE_QUESTIONS_AVAILABLE:
        app.include_router(progressive_questions_router)
        print("✅ Progressive Questions API registered")
    else:
        print("⚠️ Progressive Questions API not available")

    # Register User Journey API
    if USER_JOURNEY_AVAILABLE:
        app.include_router(user_journey_router)
        print("✅ User Journey API registered")
    else:
        print("⚠️ User Journey API not available")

    # Register Engagement API
    if ENGAGEMENT_API_AVAILABLE:
        app.include_router(engagement_router)
        print("✅ Engagement API registered")
    else:
        print("⚠️ Engagement API not available")

    # Register Three-Consultant API
    if THREE_CONSULTANT_API_AVAILABLE:
        app.include_router(three_consultant_router)
        print("✅ Three-Consultant API registered")
    else:
        print("⚠️ Three-Consultant API not available")

    # Register Strategic Trio Critique API
    if STRATEGIC_TRIO_API_AVAILABLE:
        app.include_router(strategic_trio_router)
        print("✅ Strategic Trio Critique API registered")
    else:
        print("⚠️ Strategic Trio Critique API not available")

    # Register Senior Advisor API
    if SENIOR_ADVISOR_API_AVAILABLE:
        app.include_router(senior_advisor_router)
        print("✅ Senior Advisor API registered")
    else:
        print("⚠️ Senior Advisor API not available")

    # Register Socratic Forge API (Week 3 - V5 Final Implementation)
    if SOCRATIC_FORGE_API_AVAILABLE:
        app.include_router(socratic_forge_router)
        print("✅ Week 3 - V5: Socratic Cognitive Forge API registered")
    else:
        print("⚠️ Socratic Cognitive Forge API not available")

    # Register Analysis Execution API (Phase 1 - Critical Fix)
    if ANALYSIS_EXECUTION_API_AVAILABLE:
        app.include_router(analysis_execution_router)
        print("✅ Phase 1: Analysis Execution API registered")
    else:
        print("⚠️ Analysis Execution API not available")

    # Register Devil's Advocate API (Phase 2 - Quality Control)
    if DEVILS_ADVOCATE_API_AVAILABLE:
        app.include_router(devils_advocate_router)
        print("✅ Phase 2: Devil's Advocate API registered")
    else:
        print("⚠️ Devil's Advocate API not available")

    # Register Markdown Output API (Level 3 Enhancement - "Intelligence as an Asset")
    if MARKDOWN_OUTPUT_API_AVAILABLE:
        app.include_router(markdown_output_router)
        print("✅ Level 3: Markdown Output API registered")
    else:
        print("⚠️ Markdown Output API not available")

    # Register Flywheel Management API (Internal Admin Dashboard)
    if FLYWHEEL_MANAGEMENT_API_AVAILABLE:
        app.include_router(flywheel_management_router)
        print("✅ Flywheel Management API registered")
    else:
        print("⚠️ Flywheel Management API not available")

    # Register Engagement Results API (Final Assembly - Complete Final API Payload)
    if ENGAGEMENT_RESULTS_API_AVAILABLE:
        app.include_router(engagement_results_router)
        print("✅ Final Assembly: Engagement Results API registered")
    else:
        print("⚠️ Engagement Results API not available")

    # Register Platform Statistics API (Proof of Work Metrics Engine)
    if PLATFORM_STATS_API_AVAILABLE:
        app.include_router(platform_stats_router)
        print("✅ Proof of Work: Platform Statistics API registered")
    else:
        print("⚠️ Platform Statistics API not available")

    # Register Public Showcase API (Faculty Showcase - Public Consultant Registry)
    if PUBLIC_SHOWCASE_API_AVAILABLE:
        app.include_router(public_showcase_router)
        print("✅ Faculty Showcase: Public Consultant Registry API registered")
    else:
        print("⚠️ Public Showcase API not available")

    # Register Unified Analysis API (Integration Mandate - ONE TRUE ENDPOINT)
    try:
        from src.engine.api.unified_analysis_api import (
            router as unified_analysis_router,
        )

        app.include_router(unified_analysis_router)
        print("✅ Integration Mandate: Unified Analysis API registered")
    except ImportError as e:
        print(f"⚠️ Unified Analysis API not available: {e}")

    # Phoenix Phase 4: Register Benchmarking API (Monte Carlo & Performance Testing)
    try:
        from src.engine.api.benchmarking_api import router as benchmarking_router

        app.include_router(benchmarking_router)
        print("✅ Monte Carlo Benchmarking API registered")
    except ImportError as e:
        print(f"⚠️ Monte Carlo Benchmarking API not available: {e}")

    # Removed deprecated Progressive Disclosure API route; canonical V2 endpoints are in /api/v2/engagements

except Exception as e:
    # Fallback for when API is not available
    print(f"⚠️ Supabase API initialization failed: {e}")
    from fastapi import FastAPI

    app = FastAPI(title="METIS Cognitive Platform (Fallback)", version="1.0.0")

    # Try to register routers in fallback mode
    if MANUAL_OVERRIDE_API_AVAILABLE:
        app.include_router(manual_override_router)
    if TRANSPARENCY_STREAMING_AVAILABLE:
        app.include_router(transparency_streaming_router)
    if PROGRESSIVE_QUESTIONS_AVAILABLE:
        app.include_router(progressive_questions_router)
    if USER_JOURNEY_AVAILABLE:
        app.include_router(user_journey_router)
    if ENGAGEMENT_API_AVAILABLE:
        app.include_router(engagement_router)
    if THREE_CONSULTANT_API_AVAILABLE:
        app.include_router(three_consultant_router)
    if STRATEGIC_TRIO_API_AVAILABLE:
        app.include_router(strategic_trio_router)
    if SENIOR_ADVISOR_API_AVAILABLE:
        app.include_router(senior_advisor_router)
    if SOCRATIC_FORGE_API_AVAILABLE:
        app.include_router(socratic_forge_router)
    if ANALYSIS_EXECUTION_API_AVAILABLE:
        app.include_router(analysis_execution_router)
    if DEVILS_ADVOCATE_API_AVAILABLE:
        app.include_router(devils_advocate_router)
    if MARKDOWN_OUTPUT_API_AVAILABLE:
        app.include_router(markdown_output_router)
    if ENGAGEMENT_RESULTS_API_AVAILABLE:
        app.include_router(engagement_results_router)
    if PLATFORM_STATS_API_AVAILABLE:
        app.include_router(platform_stats_router)
    if PUBLIC_SHOWCASE_API_AVAILABLE:
        app.include_router(public_showcase_router)


if __name__ == "__main__":
    asyncio.run(main())
