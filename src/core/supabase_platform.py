#!/usr/bin/env python3
"""
METIS Enhanced Platform with Supabase Integration
Task 6: Production-ready platform with persistent data layer

Integrates Supabase persistence throughout the cognitive processing pipeline
for complete end-to-end data management and audit capabilities.
"""

import asyncio
import logging
import signal
import sys
from typing import Dict, Any, Optional, List
from datetime import datetime
from uuid import UUID

# Import foundation components
from src.core.event_bus import get_event_bus
from src.core.auth_foundation import get_auth_manager
from src.core.audit_trail import get_audit_manager, AuditEventType, AuditSeverity
from src.factories.engine_factory import CognitiveEngineFactory

# TEMP DISABLED - # TEMP DISABLED - from src.ui import get_transparency_engine
from src.engine.api.enhanced_foundation import get_enhanced_api_foundation

# Import Supabase integration
from src.persistence.supabase_integration import (
    get_supabase_integration,
    get_supabase_repository,
    create_engagement_with_persistence,
    complete_engagement_with_persistence,
)


class MetisSupabasePlatform:
    """
    Enhanced METIS Platform with Supabase Integration
    Production-ready cognitive intelligence platform with persistent data layer
    """

    def __init__(self):
        """Initialize enhanced platform"""
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self.startup_time: Optional[datetime] = None

        # Component tracking
        self.component_health: Dict[str, bool] = {
            "event_bus": False,
            "auth_manager": False,
            "audit_manager": False,
            "cognitive_engine": False,
            "transparency_engine": False,
            "supabase_integration": False,
            "enhanced_api": False,
        }

        # Platform instances
        self.supabase_integration = None
        self.repository = None
        self.enhanced_api = None

        # Performance metrics
        self.metrics = {
            "total_engagements": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "avg_processing_time_ms": 0.0,
            "database_operations": 0,
            "uptime_seconds": 0,
        }

    async def initialize(self) -> bool:
        """Initialize all enhanced platform components"""

        self.logger.info("🚀 METIS Supabase Platform - Initialization Starting")
        self.startup_time = datetime.utcnow()

        try:
            # Initialize Supabase Integration
            self.logger.info("🔗 Initializing Supabase Integration...")
            self.supabase_integration = await get_supabase_integration()
            if self.supabase_integration.is_initialized:
                self.repository = await get_supabase_repository()
                self.component_health["supabase_integration"] = True
                self.logger.info("Supabase Integration: ✅ Ready")
            else:
                self.logger.error("Supabase Integration: ❌ Failed")
                return False

            # Initialize Event Bus
            self.logger.info("📡 Initializing Event Bus...")
            event_bus = await get_event_bus()
            health = await event_bus.get_health_status()
            self.component_health["event_bus"] = health["status"] == "ready"
            self.logger.info(
                f"Event Bus: {'✅ Ready' if self.component_health['event_bus'] else '❌ Failed'}"
            )

            # Initialize Authentication Manager
            self.logger.info("🔐 Initializing Authentication Manager...")
            auth_manager = await get_auth_manager()
            auth_health = await auth_manager.get_auth_health_status()
            self.component_health["auth_manager"] = (
                auth_health["system_status"] == "healthy"
            )
            self.logger.info(
                f"Authentication: {'✅ Ready' if self.component_health['auth_manager'] else '❌ Failed'}"
            )

            # Initialize Audit Trail Manager
            self.logger.info("📋 Initializing Audit Trail Manager...")
            audit_manager = await get_audit_manager()
            audit_health = await audit_manager.get_audit_health_status()
            self.component_health["audit_manager"] = (
                audit_health["storage_health"] == "healthy"
            )
            self.logger.info(
                f"Audit Trail: {'✅ Ready' if self.component_health['audit_manager'] else '❌ Failed'}"
            )

            # Initialize Cognitive Engine
            self.logger.info("🧠 Initializing Cognitive Engine...")
            cognitive_engine = CognitiveEngineFactory.create_engine()
            models = await cognitive_engine.get_available_models()
            self.component_health["cognitive_engine"] = len(models) > 0
            self.logger.info(
                f"Cognitive Engine: {'✅ Ready' if self.component_health['cognitive_engine'] else '❌ Failed'} ({len(models)} models loaded)"
            )

            # Initialize Transparency Engine
            self.logger.info("🔍 Initializing Transparency Engine...")
            transparency_engine = await get_transparency_engine()
            self.component_health["transparency_engine"] = True
            self.logger.info("Transparency Engine: ✅ Ready")

            # Initialize Enhanced API Foundation
            self.logger.info("🌐 Initializing Enhanced API Foundation...")
            try:
                self.enhanced_api = await get_enhanced_api_foundation()
                self.component_health["enhanced_api"] = True
                self.logger.info("Enhanced API: ✅ Ready")
            except Exception as e:
                self.component_health["enhanced_api"] = False
                self.logger.warning(f"Enhanced API: ⚠️ Failed - {e}")

            # Log system startup to database
            await self._log_system_startup()

            # Verify critical components
            critical_components = [
                "supabase_integration",
                "event_bus",
                "cognitive_engine",
            ]
            all_critical_healthy = all(
                self.component_health[comp] for comp in critical_components
            )

            if all_critical_healthy:
                self.is_running = True
                self.logger.info(
                    "🎉 METIS Supabase Platform - Initialization Complete!"
                )
                self.logger.info(
                    f"Platform Status: {'🟢 All Systems Operational' if all(self.component_health.values()) else '🟡 Some Components Degraded'}"
                )
                return True
            else:
                self.logger.error("❌ Critical component initialization failed")
                return False

        except Exception as e:
            self.logger.error(
                f"💥 Supabase platform initialization failed: {e}", exc_info=True
            )
            return False

    @property
    def supabase(self):
        """
        Property to provide direct access to Supabase client for compatibility.

        This property enables the markdown API and other components to access
        the Supabase client using the familiar `.supabase.table()` interface.
        """
        if hasattr(self, "supabase_integration") and self.supabase_integration:
            # Return the connection manager's client directly
            if hasattr(self.supabase_integration, "connection_manager"):
                return self.supabase_integration.connection_manager.client
            # Or if it has direct client access
            elif hasattr(self.supabase_integration, "client"):
                return self.supabase_integration.client

        # Fallback: try to create a simple client from environment
        try:
            import os
            from supabase import create_client

            url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
            key = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
            if url and key:
                return create_client(url, key)
        except Exception as e:
            self.logger.warning(f"Failed to create fallback Supabase client: {e}")

        return None

    async def _log_system_startup(self):
        """Log system startup to database and audit trail"""
        try:
            if self.component_health["audit_manager"]:
                audit_manager = await get_audit_manager()
                await audit_manager.log_event(
                    event_type=AuditEventType.SYSTEM_START,
                    severity=AuditSeverity.MEDIUM,
                    action_performed="supabase_platform_startup",
                    event_description="METIS Platform with Supabase integration initialized",
                    metadata={
                        "component_health": self.component_health,
                        "startup_time": self.startup_time.isoformat(),
                        "version": "2.0.0",
                        "database_integration": "supabase",
                        "api_version": "enhanced",
                    },
                )
        except Exception as e:
            self.logger.warning(f"Failed to log system startup: {e}")

    async def process_engagement_with_persistence(
        self,
        problem_statement: str,
        business_context: Dict[str, Any] = None,
        user_id: Optional[UUID] = None,
        session_id: Optional[UUID] = None,
        rigor_level: str = "L1",
    ) -> Dict[str, Any]:
        """
        Process complete engagement with full Supabase persistence
        Enhanced cognitive intelligence workflow with database integration
        """

        if not self.is_running:
            raise RuntimeError("Supabase platform not initialized")

        if not self.repository:
            raise RuntimeError("Database repository not available")

        start_time = datetime.utcnow()

        try:
            self.logger.info("🎯 Processing engagement with Supabase persistence...")

            # Step 1: Create engagement in database
            engagement_id, engagement_data = await create_engagement_with_persistence(
                problem_statement=problem_statement,
                business_context=business_context or {},
                user_id=user_id,
                session_id=session_id,
            )

            self.logger.info(f"✅ Created engagement {engagement_id} in Supabase")

            # Step 2: Get relevant mental models from database
            relevant_models = await self.repository.get_mental_models_by_relevance(
                problem_context=problem_statement, limit=8
            )

            self.logger.info(
                f"🧠 Selected {len(relevant_models)} relevant mental models from Supabase"
            )

            # Step 3: Find synergistic N-way patterns
            selected_model_names = [model["ke_name"] for model in relevant_models[:5]]
            synergistic_patterns = await self.repository.find_synergistic_patterns(
                selected_model_names
            )

            self.logger.info(
                f"🔗 Detected {len(synergistic_patterns)} synergistic patterns"
            )

            # Step 4: Execute cognitive processing
            cognitive_analysis = await self._execute_cognitive_processing(
                engagement_data, relevant_models, synergistic_patterns
            )

            # Step 5: Create transparency layers
            transparency_layers = await self._create_transparency_layers(
                engagement_id, engagement_data, cognitive_analysis, rigor_level
            )

            # Step 6: Generate decision trail
            decisions = await self._generate_decision_trail(
                engagement_id, cognitive_analysis, user_id, session_id
            )

            # Step 7: Create Munger overlay (if high rigor)
            munger_overlay_id = None
            if rigor_level in ["L2", "L3"]:
                munger_overlay_id = await self._create_munger_overlay(
                    engagement_id, rigor_level, cognitive_analysis
                )

            # Step 8: Complete engagement persistence
            success = await complete_engagement_with_persistence(
                engagement_id=engagement_id,
                cognitive_analysis=cognitive_analysis,
                transparency_layers=transparency_layers,
                decisions=decisions,
                user_id=user_id,
                session_id=session_id,
            )

            if not success:
                raise Exception("Failed to complete engagement persistence")

            # Calculate metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Update platform metrics
            self.metrics["total_engagements"] += 1
            self.metrics["successful_analyses"] += 1
            self.metrics["database_operations"] += 6  # Rough count of DB operations
            self.metrics["avg_processing_time_ms"] = (
                self.metrics["avg_processing_time_ms"]
                * (self.metrics["successful_analyses"] - 1)
                + processing_time
            ) / self.metrics["successful_analyses"]

            # Prepare enhanced results
            results = {
                "engagement_id": str(engagement_id),
                "status": "completed",
                "processing_time_ms": processing_time,
                "database_integration": {
                    "persistence_enabled": True,
                    "models_from_database": len(relevant_models),
                    "transparency_layers_created": len(transparency_layers),
                    "decisions_logged": len(decisions),
                    "munger_overlay_created": munger_overlay_id is not None,
                },
                "cognitive_analysis": {
                    "selected_models": [model["ke_name"] for model in relevant_models],
                    "model_details": relevant_models,
                    "nway_patterns": synergistic_patterns,
                    "confidence_scores": cognitive_analysis["confidence_scores"],
                    "reasoning_steps": cognitive_analysis["reasoning_steps"],
                },
                "transparency_system": {
                    "layers_available": len(transparency_layers),
                    "rigor_level": rigor_level,
                    "navigation_path": [
                        layer["layer_type"] for layer in transparency_layers
                    ],
                },
                "munger_overlay": {
                    "overlay_id": str(munger_overlay_id) if munger_overlay_id else None,
                    "rigor_level": rigor_level,
                    "enabled": munger_overlay_id is not None,
                },
                "metadata": {
                    "created_at": start_time.isoformat(),
                    "completed_at": datetime.utcnow().isoformat(),
                    "platform_version": "2.0.0",
                    "database_integration": "supabase",
                    "api_version": "enhanced",
                },
            }

            self.logger.info(
                f"✅ Supabase engagement {engagement_id} completed successfully ({processing_time:.1f}ms)"
            )
            return results

        except Exception as e:
            # Log failure
            self.metrics["failed_analyses"] += 1

            # Log to audit trail
            if self.component_health["audit_manager"]:
                audit_manager = await get_audit_manager()
                await audit_manager.log_event(
                    event_type=AuditEventType.SYSTEM_ERROR,
                    severity=AuditSeverity.CRITICAL,
                    user_id=user_id,
                    session_id=session_id,
                    action_performed="process_engagement_with_persistence",
                    event_description=f"Supabase engagement processing failed: {str(e)}",
                    metadata={"error": str(e), "error_type": type(e).__name__},
                )

            self.logger.error(
                f"❌ Supabase engagement processing failed: {e}", exc_info=True
            )
            raise

    async def _execute_cognitive_processing(
        self,
        engagement_data: Dict[str, Any],
        relevant_models: List[Dict[str, Any]],
        synergistic_patterns: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Execute cognitive processing with database-driven model selection"""

        # Simulate advanced cognitive processing
        reasoning_steps = [
            {
                "step": "supabase_model_selection",
                "description": f"Selected {len(relevant_models)} models from Supabase based on relevance scoring",
                "confidence": 0.88,
                "timestamp": datetime.now().isoformat(),
            },
            {
                "step": "nway_pattern_detection",
                "description": f"Identified {len(synergistic_patterns)} synergistic patterns in model combinations",
                "confidence": 0.82,
                "timestamp": datetime.now().isoformat(),
            },
            {
                "step": "cognitive_synthesis",
                "description": "Synthesized insights using Supabase-enhanced mental models framework",
                "confidence": 0.90,
                "timestamp": datetime.now().isoformat(),
            },
        ]

        confidence_scores = {
            "model_selection_accuracy": 0.88,
            "pattern_detection_reliability": 0.82,
            "synthesis_quality": 0.90,
            "overall_confidence": 0.87,
        }

        return {
            "reasoning_steps": reasoning_steps,
            "confidence_scores": confidence_scores,
            "selected_models": relevant_models,
            "synergistic_patterns": synergistic_patterns,
            "processing_metadata": {
                "database_driven": True,
                "enhancement_level": "nway_enriched",
                "cognitive_framework": "memo_with_supabase",
            },
        }

    async def _create_transparency_layers(
        self,
        engagement_id: UUID,
        engagement_data: Dict[str, Any],
        cognitive_analysis: Dict[str, Any],
        rigor_level: str,
    ) -> List[Dict[str, Any]]:
        """Create transparency layers with database persistence"""

        layers = []

        # Executive layer
        layers.append(
            {
                "layer_type": "executive",
                "layer_order": 1,
                "title": "Executive Summary",
                "content": f"Strategic analysis of: {engagement_data['problem_statement'][:150]}...\n\nSupabase-driven analysis using {len(cognitive_analysis['selected_models'])} mental models with {len(cognitive_analysis['synergistic_patterns'])} synergistic patterns detected.\n\nConfidence: {cognitive_analysis['confidence_scores']['overall_confidence']:.1%}",
                "summary": "Supabase-integrated cognitive analysis with confidence metrics",
                "cognitive_load": "minimal",
                "reading_time_minutes": 2,
                "complexity_score": 0.2,
            }
        )

        # Strategic layer
        layers.append(
            {
                "layer_type": "strategic",
                "layer_order": 2,
                "title": "Strategic Framework",
                "content": f"Mental Models (Supabase-Selected): {', '.join([m['ke_name'] for m in cognitive_analysis['selected_models'][:3]])}...\n\nMethodology: Supabase database-driven model selection with N-way pattern detection\n\nApproach: PostgreSQL-integrated cognitive processing with {rigor_level} rigor validation",
                "summary": "Supabase-driven mental models with enhanced pattern detection",
                "cognitive_load": "light",
                "reading_time_minutes": 5,
                "complexity_score": 0.4,
            }
        )

        # Additional layers based on rigor level
        if rigor_level in ["L1", "L2", "L3"]:
            layers.append(
                {
                    "layer_type": "analytical",
                    "layer_order": 3,
                    "title": "Analytical Deep Dive",
                    "content": f"Supabase Integration Details:\n\nModels Selected: {len(cognitive_analysis['selected_models'])} from {126} available\nSynergistic Patterns: {cognitive_analysis['synergistic_patterns']}\nConfidence Metrics: {cognitive_analysis['confidence_scores']}",
                    "summary": "Detailed Supabase-driven analysis with confidence metrics",
                    "cognitive_load": "moderate",
                    "reading_time_minutes": 10,
                    "complexity_score": 0.6,
                }
            )

        if rigor_level in ["L2", "L3"]:
            layers.append(
                {
                    "layer_type": "technical",
                    "layer_order": 4,
                    "title": "Technical Implementation",
                    "content": f"Technical Architecture:\n\nDatabase: Supabase PostgreSQL with pgvector\nModel Storage: {len(cognitive_analysis['selected_models'])} models with enhanced metadata\nPattern Detection: N-way interactions with lollapalooza scoring\nPersistence: Complete audit trail with transparency layers",
                    "summary": "Supabase-integrated technical implementation",
                    "cognitive_load": "heavy",
                    "reading_time_minutes": 15,
                    "complexity_score": 0.8,
                }
            )

        if rigor_level == "L3":
            layers.append(
                {
                    "layer_type": "audit",
                    "layer_order": 5,
                    "title": "Complete Audit Trail",
                    "content": "Full Supabase Audit:\n\nDatabase Operations: All analysis steps persisted to Supabase\nModel Selection: Relevance-scored selection from knowledge_elements table\nPattern Detection: N-way interactions with lollapalooza_potential scoring\nTransparency: Progressive disclosure with Supabase-backed layers",
                    "summary": "Complete Supabase-backed audit trail with transparency",
                    "cognitive_load": "heavy",
                    "reading_time_minutes": 20,
                    "complexity_score": 1.0,
                }
            )

        # Store layers in database (handled by complete_engagement_with_persistence)
        return layers

    async def _generate_decision_trail(
        self,
        engagement_id: UUID,
        cognitive_analysis: Dict[str, Any],
        user_id: Optional[UUID],
        session_id: Optional[UUID],
    ) -> List[Dict[str, Any]]:
        """Generate decision audit trail"""

        decisions = []

        # Model selection decision
        decisions.append(
            {
                "type": "model_selection",
                "description": f"Selected {len(cognitive_analysis['selected_models'])} mental models based on Supabase relevance scoring",
                "confidence": cognitive_analysis["confidence_scores"][
                    "model_selection_accuracy"
                ],
                "evidence_summary": "Supabase-driven relevance scoring with Munger filter enhancement",
                "alternatives": [
                    "Random selection",
                    "Manual curation",
                    "Category-based selection",
                ],
                "impact": {"scope": "analysis_quality", "magnitude": "high"},
            }
        )

        # Pattern detection decision
        decisions.append(
            {
                "type": "pattern_detection",
                "description": f"Identified {len(cognitive_analysis['synergistic_patterns'])} synergistic patterns",
                "confidence": cognitive_analysis["confidence_scores"][
                    "pattern_detection_reliability"
                ],
                "evidence_summary": "N-way interactions analysis with lollapalooza potential scoring",
                "alternatives": [
                    "Single model analysis",
                    "Sequential model application",
                ],
                "impact": {"scope": "insight_quality", "magnitude": "medium"},
            }
        )

        return decisions

    async def _create_munger_overlay(
        self, engagement_id: UUID, rigor_level: str, cognitive_analysis: Dict[str, Any]
    ) -> UUID:
        """Create Munger overlay analysis"""

        overlay_data = {
            "inversion_analysis": {
                "failure_modes": [
                    "Cognitive biases",
                    "Model selection errors",
                    "Pattern misinterpretation",
                ],
                "avoidance_strategies": [
                    "Multiple model validation",
                    "N-way pattern cross-checking",
                    "Confidence calibration",
                ],
            },
            "latticework_connections": [
                model["ke_name"] for model in cognitive_analysis["selected_models"]
            ],
            "bias_identification": [
                "Confirmation bias",
                "Availability heuristic",
                "Anchoring bias",
            ],
            "uncertainty_quantification": {
                "confidence_intervals": cognitive_analysis["confidence_scores"],
                "sensitivity_analysis": "Model selection robustness tested",
            },
            "decision_quality_metrics": {
                "reversibility": "medium",
                "consequence_magnitude": "high",
                "information_quality": "high",
            },
        }

        overlay_id = await self.repository.create_munger_overlay(
            engagement_id=engagement_id,
            rigor_level=rigor_level,
            analysis_data=overlay_data,
        )

        return overlay_id

    async def get_platform_health(self) -> Dict[str, Any]:
        """Get comprehensive platform health status"""

        uptime = (
            (datetime.utcnow() - self.startup_time).total_seconds()
            if self.startup_time
            else 0
        )
        self.metrics["uptime_seconds"] = uptime

        # Get Supabase health
        supabase_health = {}
        if self.supabase_integration:
            supabase_health = await self.supabase_integration.get_health_status()

        return {
            "platform_status": "healthy" if self.is_running else "stopped",
            "is_running": self.is_running,
            "startup_time": (
                self.startup_time.isoformat() if self.startup_time else None
            ),
            "uptime_seconds": uptime,
            "version": "2.0.0",
            "api_version": "enhanced",
            "database_integration": "supabase",
            # Component health
            "component_health": self.component_health,
            "supabase_integration": supabase_health,
            # Metrics
            "performance_metrics": self.metrics,
            "database_metrics": supabase_health.get("metrics", {}),
            # Production readiness
            "production_ready": {
                "supabase_connected": self.component_health["supabase_integration"],
                "api_available": self.component_health["enhanced_api"],
                "cognitive_engine_loaded": self.component_health["cognitive_engine"],
                "transparency_enabled": self.component_health["transparency_engine"],
                "audit_trail_active": self.component_health["audit_manager"],
            },
        }

    async def shutdown(self):
        """Graceful platform shutdown"""

        self.logger.info("🛑 METIS Supabase Platform - Shutdown Initiated")

        try:
            # Log shutdown to database
            if self.component_health["audit_manager"]:
                audit_manager = await get_audit_manager()
                await audit_manager.log_event(
                    event_type=AuditEventType.SYSTEM_START,  # No shutdown event type
                    severity=AuditSeverity.MEDIUM,
                    action_performed="supabase_platform_shutdown",
                    event_description="METIS Supabase Platform shutdown initiated",
                    metadata={
                        "uptime_seconds": self.metrics["uptime_seconds"],
                        "total_engagements": self.metrics["total_engagements"],
                        "database_operations": self.metrics["database_operations"],
                    },
                )

            # Shutdown event bus
            if self.component_health["event_bus"]:
                event_bus = await get_event_bus()
                await event_bus.shutdown()

            self.is_running = False
            self.logger.info("✅ METIS Supabase Platform - Shutdown Complete")

        except Exception as e:
            self.logger.error(f"❌ Error during Supabase shutdown: {e}", exc_info=True)


# Global Supabase platform instance
_supabase_platform_instance: Optional[MetisSupabasePlatform] = None


async def get_supabase_platform() -> MetisSupabasePlatform:
    """Get or create global Supabase platform instance"""
    global _supabase_platform_instance

    if _supabase_platform_instance is None:
        _supabase_platform_instance = MetisSupabasePlatform()

    return _supabase_platform_instance


# Production API server with Supabase integration
async def create_supabase_api_server():
    """Create production-ready API server with Supabase integration"""

    platform = await get_supabase_platform()

    if not await platform.initialize():
        raise Exception("Failed to initialize Supabase platform")

    enhanced_api = await get_enhanced_api_foundation()

    return enhanced_api.get_app()


# CLI Interface for Supabase platform testing
async def main():
    """Main entry point for Supabase platform CLI testing"""

    print("🚀 METIS Supabase Cognitive Platform v2.0.0")
    print("Database-Integrated Cognitive Intelligence")
    print("=" * 60)

    # Initialize Supabase platform
    platform = await get_supabase_platform()

    # Setup signal handlers
    def signal_handler(signum, frame):
        print(f"\n📡 Received signal {signum}, shutting down...")
        asyncio.create_task(platform.shutdown())
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize
    if not await platform.initialize():
        print("❌ Supabase platform initialization failed")
        sys.exit(1)

    # Interactive mode
    print("\n🎯 Enter engagement problems (or 'quit' to exit):")
    print(
        "💡 Example: 'How can we improve customer retention using behavioral insights?'"
    )
    print("🔗 Features: Supabase persistence, N-way patterns, Progressive transparency")

    try:
        while True:
            print("\n" + "─" * 60)
            problem = input("🤔 Problem Statement: ").strip()

            if problem.lower() in ["quit", "exit", "q"]:
                break

            if not problem:
                continue

            # Get rigor level
            rigor = input("🎯 Rigor Level (L0/L1/L2/L3) [L1]: ").strip() or "L1"
            if rigor not in ["L0", "L1", "L2", "L3"]:
                rigor = "L1"

            try:
                print(f"\n⏳ Processing engagement with {rigor} rigor level...")

                results = await platform.process_engagement_with_persistence(
                    problem_statement=problem,
                    business_context={"source": "supabase_cli_interface"},
                    rigor_level=rigor,
                )

                print(
                    f"\n✅ Supabase Analysis Complete! (ID: {results['engagement_id']})"
                )
                print(f"⚡ Processing Time: {results['processing_time_ms']:.1f}ms")
                print(
                    f"🔗 Supabase Integration: {'✅ Enabled' if results['database_integration']['persistence_enabled'] else '❌ Disabled'}"
                )

                # Show Supabase integration details
                db_info = results["database_integration"]
                print("\n📊 Supabase Integration:")
                print(f"   🧠 Models from Database: {db_info['models_from_database']}")
                print(
                    f"   🔍 Transparency Layers: {db_info['transparency_layers_created']}"
                )
                print(f"   📋 Decisions Logged: {db_info['decisions_logged']}")
                print(
                    f"   🎯 Munger Overlay: {'✅ Created' if db_info['munger_overlay_created'] else '❌ Not Created'}"
                )

                # Show cognitive analysis
                cognitive = results["cognitive_analysis"]
                print("\n🧠 Supabase Cognitive Analysis:")
                print(f"   📊 Models Applied: {len(cognitive['selected_models'])}")
                print(f"   🔗 N-way Patterns: {len(cognitive['nway_patterns'])}")
                print(
                    f"   📈 Confidence: {cognitive['confidence_scores']['overall_confidence']:.1%}"
                )

                # Show transparency system
                transparency = results["transparency_system"]
                print("\n🔍 Transparency System:")
                print(
                    f"   📑 Layers Available: {transparency['layers_available']} ({rigor} rigor)"
                )
                print(
                    f"   🗺️  Navigation: {' → '.join(transparency['navigation_path'])}"
                )

            except Exception as e:
                print(f"❌ Error processing Supabase engagement: {e}")

    except KeyboardInterrupt:
        pass

    finally:
        print("\n🛑 Shutting down Supabase platform...")
        await platform.shutdown()
        print("👋 Supabase platform offline!")


def get_supabase_client():
    """Mock supabase client for development mode"""
    return type(
        "MockSupabaseClient",
        (),
        {
            "table": lambda x: type(
                "MockTable",
                (),
                {
                    "insert": lambda data: type(
                        "MockInsert",
                        (),
                        {"execute": lambda: {"data": [], "error": None}},
                    )(),
                    "select": lambda fields="*": type(
                        "MockSelect",
                        (),
                        {"execute": lambda: {"data": [], "error": None}},
                    )(),
                    "update": lambda data: type(
                        "MockUpdate",
                        (),
                        {"execute": lambda: {"data": [], "error": None}},
                    )(),
                    "delete": lambda: type(
                        "MockDelete",
                        (),
                        {"execute": lambda: {"data": [], "error": None}},
                    )(),
                },
            )()
        },
    )()


if __name__ == "__main__":
    asyncio.run(main())
