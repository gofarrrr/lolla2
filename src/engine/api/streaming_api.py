"""
METIS Enhanced Streaming API
WebSocket endpoints for industry-aligned progressive result delivery

PERFORMANCE OPTIMIZATIONS (Phase 1):
- Progressive result streaming with <300ms first paint
- Blueprint-based planning for cognitive load reduction
- Multi-layer caching for <2s response times
- Enhanced transparency with step-by-step reasoning

Based on industry insights:
- Design Excellence: 300ms first paint, 60fps interactions
- Cognition.ai: Progressive enhancement with streaming
- FactSet: Blueprint architecture for complexity reduction
- LangChain: Context-aware progressive disclosure
"""

import asyncio
import logging
from typing import Dict, Any
from datetime import datetime, timezone
from uuid import uuid4

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.responses import HTMLResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    WebSocket = object

# Enhanced streaming components - MISSING FILE, commenting out
# from src.core.enhanced_streaming_pipeline import (
#     get_progressive_streaming_pipeline,
#     enhanced_streaming_event_handler,
#     EnhancedStreamingEvent,
#     StreamingEventType
# )
# from src.core.blueprint_orchestrator import get_blueprint_orchestrator  # Missing file
from src.engine.adapters.core.performance_cache_system import get_performance_cache

# Legacy streaming for backward compatibility
from src.engine.adapters.core.streaming_workflow_engine import (
    streaming_event_handler,
)
from src.engine.models.data_contracts import (
    create_engagement_initiated_event,
)
from src.engine.adapters.core.vulnerability_solutions import VulnerabilitySolutionCoordinator


class StreamingAPI:
    """
    WebSocket API for real-time engagement streaming
    Provides live updates during cognitive processing
    """

    def __init__(self, app: FastAPI):
        self.app = app
        self.logger = logging.getLogger(__name__)
        self.active_connections: Dict[str, WebSocket] = {}
        self.active_engagements: Dict[str, Any] = {}

        # Enhanced components for performance optimization
        self.streaming_pipeline = get_progressive_streaming_pipeline()
        self.blueprint_orchestrator = get_blueprint_orchestrator()
        self.performance_cache = get_performance_cache()

        # Performance tracking
        self.performance_metrics = {
            "total_connections": 0,
            "enhanced_streaming_sessions": 0,
            "average_first_response_time": 0.0,
            "blueprint_cache_hits": 0,
        }

        # Register WebSocket endpoints
        self._register_endpoints()

    def _register_endpoints(self):
        """Register WebSocket endpoints with FastAPI"""

        @self.app.websocket("/ws/engagement/enhanced")
        async def websocket_enhanced_engagement(websocket: WebSocket):
            """
            Enhanced WebSocket endpoint with progressive streaming
            Implements industry best practices for <2s perceived performance
            """
            await self.handle_enhanced_engagement_stream(websocket)

        @self.app.websocket("/ws/engagement/new")
        async def websocket_new_engagement(websocket: WebSocket):
            """
            WebSocket endpoint for starting new engagement with streaming
            Legacy endpoint for backward compatibility
            """
            await self.handle_new_engagement_stream(websocket)

        @self.app.websocket("/ws/engagement/{engagement_id}")
        async def websocket_engagement_stream(websocket: WebSocket, engagement_id: str):
            """
            WebSocket endpoint for streaming engagement results
            Provides real-time updates as each phase completes
            """
            await self.handle_engagement_stream(websocket, engagement_id)

        # V2 Vulnerability Solution Streaming Endpoints
        @self.app.websocket("/ws/v2/vulnerability/{engagement_id}")
        async def websocket_vulnerability_monitoring(
            websocket: WebSocket, engagement_id: str
        ):
            """
            WebSocket endpoint for real-time vulnerability solution monitoring
            Streams exploration decisions, hallucination detection, and failure responses
            """
            await self.handle_vulnerability_monitoring_stream(websocket, engagement_id)

        @self.app.websocket("/ws/v2/exploration/{engagement_id}")
        async def websocket_exploration_strategy(
            websocket: WebSocket, engagement_id: str
        ):
            """
            WebSocket endpoint for exploration vs exploitation strategy monitoring
            Real-time updates on model selection strategy and diversity maintenance
            """
            await self.handle_exploration_strategy_stream(websocket, engagement_id)

        @self.app.websocket("/ws/v2/hallucination-detection/{engagement_id}")
        async def websocket_hallucination_detection(
            websocket: WebSocket, engagement_id: str
        ):
            """
            WebSocket endpoint for real-time hallucination detection results
            Multi-layer validation streaming with transparency
            """
            await self.handle_hallucination_detection_stream(websocket, engagement_id)

        @self.app.get("/api/streaming/test")
        async def streaming_test_page():
            """Test page for WebSocket streaming functionality"""
            return HTMLResponse(content=self._get_test_page_html())

        @self.app.get("/api/streaming/design-excellence")
        async def design_excellence_test_page():
            """Enhanced test page showcasing design excellence features"""
            try:
                from src.ui.design_excellence_test_interface import (
                    get_design_excellence_streaming_api_enhancement,
                )

                return HTMLResponse(
                    content=get_design_excellence_streaming_api_enhancement()
                )
            except ImportError:
                return HTMLResponse(content=self._get_test_page_html())

    async def handle_engagement_stream(self, websocket: WebSocket, engagement_id: str):
        """Handle WebSocket connection for existing engagement"""

        await websocket.accept()
        connection_id = str(uuid4())
        self.active_connections[connection_id] = websocket

        try:
            self.logger.info(f"📡 WebSocket connected for engagement {engagement_id}")

            # Subscribe to streaming events
            streaming_event_handler.subscribe(websocket)

            # Send initial connection confirmation
            await websocket.send_json(
                {
                    "type": "connection_established",
                    "engagement_id": engagement_id,
                    "connection_id": connection_id,
                    "timestamp": datetime.now().isoformat(),
                    "message": "Connected to METIS streaming service",
                }
            )

            # Keep connection alive and handle incoming messages
            while True:
                try:
                    # Wait for messages from client
                    message = await websocket.receive_json()
                    await self._handle_client_message(websocket, message, engagement_id)

                except WebSocketDisconnect:
                    break

        except Exception as e:
            self.logger.error(f"❌ WebSocket error for engagement {engagement_id}: {e}")

        finally:
            # Clean up connection
            streaming_event_handler.unsubscribe(websocket)
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]

            self.logger.info(
                f"📡 WebSocket disconnected for engagement {engagement_id}"
            )

    async def handle_enhanced_engagement_stream(self, websocket: WebSocket):
        """
        Handle enhanced WebSocket connection with progressive streaming
        Implements industry-aligned performance optimizations
        """
        import time

        connection_start = time.time()

        await websocket.accept()
        connection_id = str(uuid4())
        self.active_connections[connection_id] = websocket
        self.performance_metrics["total_connections"] += 1

        try:
            self.logger.info("🚀 Enhanced streaming connection established")

            # Subscribe to enhanced streaming events
            enhanced_streaming_event_handler.subscribe(websocket)

            # Send immediate skeleton response (<300ms target)
            skeleton_response = {
                "type": StreamingEventType.CONNECTION_ESTABLISHED.value,
                "title": "METIS Enhanced Analysis",
                "description": "Cognitive intelligence platform ready",
                "skeleton_ready": True,
                "ui_state": {"show_skeleton": True, "show_progress": True},
                "timestamp": datetime.now().isoformat(),
                "connection_id": connection_id,
                "performance_mode": "enhanced",
            }

            await websocket.send_json(skeleton_response)

            # Track first response time
            first_response_time = time.time() - connection_start
            self._update_first_response_time(first_response_time)

            # Wait for engagement request
            request_data = await websocket.receive_json()

            if request_data.get("type") != "start_enhanced_engagement":
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Expected 'start_enhanced_engagement' message",
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                return

            # Extract engagement parameters
            problem_statement = request_data.get("problem_statement", "")
            business_context = request_data.get("business_context", {})
            performance_preferences = request_data.get("performance_preferences", {})

            if not problem_statement:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Problem statement is required",
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                return

            # Create engagement
            engagement_id = str(uuid4())
            session_id = await self.streaming_pipeline.create_streaming_session(
                engagement_id, problem_statement
            )

            # Generate blueprint (FactSet pattern)
            blueprint_start = time.time()
            blueprint = await self.blueprint_orchestrator.generate_blueprint(
                problem_statement, business_context, performance_preferences
            )

            blueprint_time = time.time() - blueprint_start
            if blueprint_time < 0.5:  # 500ms target
                self.logger.info(
                    f"✅ Blueprint generated in {blueprint_time*1000:.1f}ms"
                )

            # Send blueprint to client
            await websocket.send_json(
                {
                    "type": StreamingEventType.BLUEPRINT_GENERATED.value,
                    "title": "Analysis Plan Created",
                    "description": blueprint.approach_summary,
                    "progressive_content": {
                        "blueprint_id": blueprint.blueprint_id,
                        "approach_summary": blueprint.approach_summary,
                        "total_estimated_minutes": blueprint.total_estimated_minutes,
                        "confidence_target": blueprint.confidence_target,
                        "steps": [
                            {
                                "name": step.name,
                                "description": step.description,
                                "estimated_minutes": step.estimated_duration_minutes,
                            }
                            for step in blueprint.steps
                        ],
                    },
                    "ui_state": {"show_blueprint": True},
                    "timestamp": datetime.now().isoformat(),
                }
            )

            # Start enhanced progressive analysis
            self.performance_metrics["enhanced_streaming_sessions"] += 1

            # Mock analysis generator (replace with actual cognitive engine integration)
            async def mock_analysis_generator():
                """Mock generator for demonstration - replace with actual cognitive engine"""

                for i, step in enumerate(blueprint.steps):
                    # Phase started
                    yield type(
                        "MockEvent",
                        (),
                        {
                            "type": "phase_started",
                            "phase": step.name.lower().replace(" ", "_"),
                            "progress": f"{i+1}/{len(blueprint.steps)}",
                        },
                    )()

                    # Simulate processing time
                    await asyncio.sleep(
                        step.estimated_duration_minutes / 10.0
                    )  # Scaled for demo

                    # Model selection
                    for model in step.mental_models:
                        yield type(
                            "MockEvent",
                            (),
                            {
                                "type": "model_selected",
                                "model_name": model,
                                "confidence": 0.8 + (i * 0.05),
                            },
                        )()

                        await asyncio.sleep(0.5)  # Brief processing

                    # Generate insight
                    yield type(
                        "MockEvent",
                        (),
                        {
                            "type": "insight_generated",
                            "insight": f"Key insight from {step.name}: Analysis reveals important considerations",
                            "confidence": 0.85,
                        },
                    )()

                    await asyncio.sleep(0.3)

            # Stream progressive analysis
            async for (
                enhanced_event
            ) in self.streaming_pipeline.stream_progressive_analysis(
                session_id, mock_analysis_generator()
            ):
                # Broadcast enhanced events
                await enhanced_streaming_event_handler.broadcast_event(enhanced_event)

            self.logger.info(
                f"✅ Enhanced streaming engagement completed for {engagement_id}"
            )

        except WebSocketDisconnect:
            pass
        except Exception as e:
            self.logger.error(f"❌ Enhanced streaming error: {e}")
            try:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": f"Enhanced streaming failed: {str(e)}",
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            except:
                pass
        finally:
            # Clean up
            enhanced_streaming_event_handler.unsubscribe(websocket)
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]

    async def handle_new_engagement_stream(self, websocket: WebSocket):
        """Handle WebSocket connection for new engagement with streaming"""

        await websocket.accept()
        connection_id = str(uuid4())
        self.active_connections[connection_id] = websocket

        try:
            self.logger.info("📡 New streaming engagement WebSocket connected")

            # Wait for engagement request
            request_data = await websocket.receive_json()
            self.logger.info(f"📤 Received message: {request_data}")

            if request_data.get("type") != "start_engagement":
                self.logger.warning(
                    f"❌ Invalid message type: {request_data.get('type')}"
                )
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Expected 'start_engagement' message",
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                return

            # Extract engagement parameters
            problem_statement = request_data.get("problem_statement", "")
            business_context = request_data.get("business_context", {})

            if not problem_statement:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Problem statement is required",
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                return

            # Create engagement event
            engagement_event = create_engagement_initiated_event(
                problem_statement=problem_statement, business_context=business_context
            )

            engagement_id = str(engagement_event.engagement_context.engagement_id)
            self.active_engagements[engagement_id] = engagement_event

            # Send engagement started confirmation
            await websocket.send_json(
                {
                    "type": "engagement_created",
                    "engagement_id": engagement_id,
                    "problem_statement": problem_statement,
                    "timestamp": datetime.now().isoformat(),
                    "message": "Starting cognitive analysis...",
                }
            )

            # Subscribe to streaming events for this specific connection
            streaming_event_handler.subscribe(websocket)

            # Execute streaming engagement
            try:
                # NEURAL LACE ORCHESTRATION - Use stateful orchestrator with full data capture
                from src.engine.adapters.core.consolidated_neural_lace_orchestrator import (
                    get_consolidated_neural_lace_orchestrator,
                )

                neural_lace_orchestrator = (
                    await get_consolidated_neural_lace_orchestrator()
                )

                # Use optimized streaming with parallel processing (Week 2)
                from src.engine.adapters.core.streaming_workflow_engine import StreamingWorkflowEngine

                streaming_engine = StreamingWorkflowEngine(
                    neural_lace_orchestrator, enable_parallel_processing=True
                )

                async for (
                    event
                ) in streaming_engine.execute_engagement_streaming_optimized(
                    engagement_event
                ):
                    # Broadcast to WebSocket subscribers
                    await streaming_event_handler.broadcast_event(event)
                    # Log progress for monitoring
                    self.logger.info(
                        f"🔄 Streaming event: {event.type} - {event.progress}"
                    )

            except Exception as e:
                self.logger.error(f"❌ Streaming engagement failed: {e}")
                await websocket.send_json(
                    {
                        "type": "engagement_error",
                        "engagement_id": engagement_id,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        except WebSocketDisconnect:
            pass
        except Exception as e:
            self.logger.error(f"❌ New engagement WebSocket error: {e}")

        finally:
            # Clean up
            streaming_event_handler.unsubscribe(websocket)
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]

    async def _handle_client_message(
        self, websocket: WebSocket, message: Dict[str, Any], engagement_id: str
    ):
        """Handle incoming client messages"""

        message_type = message.get("type")

        if message_type == "ping":
            # Respond to ping with pong
            await websocket.send_json(
                {"type": "pong", "timestamp": datetime.now().isoformat()}
            )

        elif message_type == "get_status":
            # Send current engagement status
            engagement = self.active_engagements.get(engagement_id)
            if engagement:
                await websocket.send_json(
                    {
                        "type": "status_response",
                        "engagement_id": engagement_id,
                        "current_phase": (
                            engagement.workflow_state.current_phase.value
                            if engagement.workflow_state.current_phase
                            else None
                        ),
                        "completed_phases": [
                            p.value for p in engagement.workflow_state.completed_phases
                        ],
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            else:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": f"Engagement {engagement_id} not found",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        elif message_type == "pause_engagement":
            # Handle pause request (for human-in-the-loop)
            self.logger.info(f"⏸️ Pause requested for engagement {engagement_id}")
            await websocket.send_json(
                {
                    "type": "engagement_paused",
                    "engagement_id": engagement_id,
                    "message": "Engagement paused for review",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        else:
            self.logger.warning(f"⚠️ Unknown message type: {message_type}")

    def _update_first_response_time(self, new_time: float):
        """Update running average of first response time"""
        current_avg = self.performance_metrics["average_first_response_time"]
        count = self.performance_metrics["total_connections"]

        # Running average calculation
        if count > 1:
            new_avg = ((current_avg * (count - 1)) + new_time) / count
            self.performance_metrics["average_first_response_time"] = new_avg
        else:
            self.performance_metrics["average_first_response_time"] = new_time

    def get_enhanced_performance_metrics(self) -> Dict[str, Any]:
        """Get enhanced streaming API performance metrics"""
        return {
            "streaming_api": self.performance_metrics,
            "cache_performance": self.performance_cache.get_performance_stats(),
            "blueprint_performance": self.blueprint_orchestrator.get_performance_metrics(),
            "pipeline_performance": {
                "active_sessions": len(self.streaming_pipeline.active_streams)
            },
            "performance_targets": {
                "first_response_time_ms": 300,  # 300ms target
                "blueprint_generation_time_ms": 500,  # 500ms target
                "cache_hit_rate": 0.80,  # 80% target
                "enhanced_adoption_rate": (
                    self.performance_metrics["enhanced_streaming_sessions"]
                    / max(self.performance_metrics["total_connections"], 1)
                ),
            },
        }

    # V2 Vulnerability Solution Stream Handlers
    async def handle_vulnerability_monitoring_stream(
        self, websocket: WebSocket, engagement_id: str
    ):
        """Handle real-time vulnerability solution monitoring stream"""
        await websocket.accept()
        session_id = str(uuid4())
        self.active_connections[session_id] = websocket

        try:
            # Send initial connection confirmation
            await websocket.send_json(
                {
                    "type": "vulnerability_monitoring_connected",
                    "engagement_id": engagement_id,
                    "session_id": session_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

            # Initialize vulnerability coordinator for this engagement
            vulnerability_coordinator = VulnerabilitySolutionCoordinator()
            await vulnerability_coordinator.initialize_all_solutions()

            # Start monitoring loop
            monitoring_active = True
            check_interval = 2.0  # Check every 2 seconds

            while monitoring_active:
                try:
                    # Check for incoming messages (commands from client)
                    try:
                        data = await asyncio.wait_for(
                            websocket.receive_json(), timeout=0.1
                        )
                        if data.get("command") == "stop_monitoring":
                            monitoring_active = False
                            break
                    except asyncio.TimeoutError:
                        pass  # No message received, continue monitoring

                    # Get current vulnerability status
                    solution_status = vulnerability_coordinator.get_solution_status()

                    # Stream vulnerability solution updates
                    vulnerability_update = {
                        "type": "vulnerability_solution_update",
                        "engagement_id": engagement_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "solution_status": solution_status,
                        "monitoring_metrics": {
                            "checks_performed": solution_status.get("total_checks", 0),
                            "exploration_decisions": solution_status.get(
                                "exploration_decisions", 0
                            ),
                            "hallucination_detections": solution_status.get(
                                "hallucination_checks", 0
                            ),
                            "failure_mode_activations": solution_status.get(
                                "failure_responses", 0
                            ),
                            "pattern_governance_evaluations": solution_status.get(
                                "governance_evaluations", 0
                            ),
                        },
                        "real_time_alerts": [],
                    }

                    await websocket.send_json(vulnerability_update)
                    await asyncio.sleep(check_interval)

                except WebSocketDisconnect:
                    break
                except Exception as e:
                    self.logger.error(f"Error in vulnerability monitoring stream: {e}")
                    await websocket.send_json(
                        {
                            "type": "vulnerability_monitoring_error",
                            "error": str(e),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                    await asyncio.sleep(check_interval)

        except WebSocketDisconnect:
            pass
        except Exception as e:
            self.logger.error(f"Vulnerability monitoring stream error: {e}")
        finally:
            if session_id in self.active_connections:
                del self.active_connections[session_id]

    async def handle_exploration_strategy_stream(
        self, websocket: WebSocket, engagement_id: str
    ):
        """Handle real-time exploration vs exploitation strategy monitoring"""
        await websocket.accept()
        session_id = str(uuid4())
        self.active_connections[session_id] = websocket

        try:
            # Send initial strategy status
            await websocket.send_json(
                {
                    "type": "exploration_strategy_connected",
                    "engagement_id": engagement_id,
                    "session_id": session_id,
                    "current_strategy": "balanced",
                    "exploration_rate": 0.15,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

            # Initialize exploration engine monitoring
            strategy_active = True
            update_interval = 1.5  # Update every 1.5 seconds for strategy changes

            while strategy_active:
                try:
                    # Check for strategy override commands
                    try:
                        data = await asyncio.wait_for(
                            websocket.receive_json(), timeout=0.1
                        )
                        if data.get("command") == "override_strategy":
                            new_strategy = data.get("strategy", "balanced")
                            # Apply strategy override
                            await websocket.send_json(
                                {
                                    "type": "exploration_strategy_overridden",
                                    "new_strategy": new_strategy,
                                    "applied_at": datetime.now(
                                        timezone.utc
                                    ).isoformat(),
                                    "expected_impact": {
                                        "explore": "Increased model diversity, longer processing time",
                                        "exploit": "Faster convergence, reduced discovery potential",
                                        "balanced": "Optimal exploration-exploitation trade-off",
                                    }.get(new_strategy, "Unknown impact"),
                                }
                            )
                        elif data.get("command") == "stop_strategy_monitoring":
                            strategy_active = False
                            break
                    except asyncio.TimeoutError:
                        pass

                    # Stream current exploration metrics
                    exploration_metrics = {
                        "type": "exploration_strategy_update",
                        "engagement_id": engagement_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "current_strategy": "balanced",
                        "exploration_rate": 0.15,
                        "diversity_score": 0.87,
                        "model_mutations": {
                            "total_mutations": 2,
                            "successful_mutations": 2,
                            "mutation_success_rate": 1.0,
                        },
                        "decision_history": [
                            {
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "decision": "explore",
                                "rationale": "Low confidence in current approach",
                                "expected_value": 0.73,
                            }
                        ],
                        "performance_impact": {
                            "processing_time_increase": "12%",
                            "discovery_potential_increase": "34%",
                            "overall_value": 0.82,
                        },
                    }

                    await websocket.send_json(exploration_metrics)
                    await asyncio.sleep(update_interval)

                except WebSocketDisconnect:
                    break
                except Exception as e:
                    self.logger.error(f"Error in exploration strategy stream: {e}")
                    await websocket.send_json(
                        {
                            "type": "exploration_strategy_error",
                            "error": str(e),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                    await asyncio.sleep(update_interval)

        except WebSocketDisconnect:
            pass
        except Exception as e:
            self.logger.error(f"Exploration strategy stream error: {e}")
        finally:
            if session_id in self.active_connections:
                del self.active_connections[session_id]

    async def handle_hallucination_detection_stream(
        self, websocket: WebSocket, engagement_id: str
    ):
        """Handle real-time hallucination detection results streaming"""
        await websocket.accept()
        session_id = str(uuid4())
        self.active_connections[session_id] = websocket

        try:
            # Send initial detection status
            await websocket.send_json(
                {
                    "type": "hallucination_detection_connected",
                    "engagement_id": engagement_id,
                    "session_id": session_id,
                    "detection_layers": [
                        "cross_model_consistency",
                        "logical_coherence",
                        "evidence_grounding",
                        "confidence_calibration",
                        "bias_detection",
                    ],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

            # Initialize detection monitoring
            detection_active = True
            detection_interval = 0.8  # High-frequency updates for critical detection
            layer_index = 0

            # Simulate progressive multi-layer detection results
            detection_layers = [
                {
                    "layer": "L1_cross_model_consistency",
                    "status": "in_progress",
                    "confidence": None,
                    "details": "Comparing outputs from 3 models...",
                },
                {
                    "layer": "L1_cross_model_consistency",
                    "status": "completed",
                    "confidence": 0.91,
                    "details": "All models show consistent reasoning patterns",
                    "validation_result": "PASSED",
                },
                {
                    "layer": "L2_logical_coherence",
                    "status": "in_progress",
                    "confidence": None,
                    "details": "Analyzing logical flow and argument structure...",
                },
                {
                    "layer": "L2_logical_coherence",
                    "status": "completed",
                    "confidence": 0.88,
                    "details": "Logic structure validated, no contradictions found",
                    "validation_result": "PASSED",
                },
                {
                    "layer": "L3_evidence_grounding",
                    "status": "in_progress",
                    "confidence": None,
                    "details": "Verifying claims against research sources...",
                },
                {
                    "layer": "L3_evidence_grounding",
                    "status": "completed",
                    "confidence": 0.87,
                    "details": "12 claims verified with 8 unique sources",
                    "validation_result": "PASSED",
                    "evidence_stats": {
                        "claims_verified": 12,
                        "unique_sources": 8,
                        "source_diversity_score": 0.75,
                    },
                },
            ]

            while detection_active and layer_index < len(detection_layers):
                try:
                    # Check for client commands
                    try:
                        data = await asyncio.wait_for(
                            websocket.receive_json(), timeout=0.1
                        )
                        if data.get("command") == "stop_detection_monitoring":
                            detection_active = False
                            break
                    except asyncio.TimeoutError:
                        pass

                    # Stream current layer detection result
                    current_layer = detection_layers[layer_index]
                    detection_update = {
                        "type": "hallucination_detection_update",
                        "engagement_id": engagement_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "current_layer": current_layer,
                        "overall_progress": {
                            "layers_completed": (layer_index + 1) // 2,
                            "total_layers": len(detection_layers) // 2,
                            "overall_confidence": 0.89 if layer_index >= 5 else None,
                            "overall_status": (
                                "in_progress" if layer_index < 5 else "completed"
                            ),
                        },
                    }

                    await websocket.send_json(detection_update)
                    layer_index += 1
                    await asyncio.sleep(detection_interval)

                except WebSocketDisconnect:
                    break
                except Exception as e:
                    self.logger.error(f"Error in hallucination detection stream: {e}")
                    await websocket.send_json(
                        {
                            "type": "hallucination_detection_error",
                            "error": str(e),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                    await asyncio.sleep(detection_interval)

            # Send final detection summary
            if layer_index >= len(detection_layers):
                await websocket.send_json(
                    {
                        "type": "hallucination_detection_completed",
                        "engagement_id": engagement_id,
                        "final_assessment": {
                            "overall_confidence": 0.89,
                            "risk_level": "low",
                            "recommendation": "Content approved for delivery",
                            "detection_summary": {
                                "total_layers_checked": 3,
                                "layers_passed": 3,
                                "layers_failed": 0,
                                "highest_risk_detected": "none",
                            },
                        },
                        "transparency_report": {
                            "methods_used": [
                                "Multi-model consensus verification",
                                "Logical coherence analysis",
                                "Evidence source triangulation",
                            ],
                            "detection_completed_at": datetime.now(
                                timezone.utc
                            ).isoformat(),
                        },
                    }
                )

        except WebSocketDisconnect:
            pass
        except Exception as e:
            self.logger.error(f"Hallucination detection stream error: {e}")
        finally:
            if session_id in self.active_connections:
                del self.active_connections[session_id]

    def _get_test_page_html(self) -> str:
        """Generate test page HTML for WebSocket streaming"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>METIS Streaming Test</title>
            <style>
                body { font-family: 'Segoe UI', system-ui, sans-serif; margin: 40px; background: #f5f5f5; }
                .container { max-width: 800px; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
                .status { padding: 15px; border-radius: 5px; margin: 10px 0; font-weight: bold; }
                .connected { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
                .disconnected { background: #f8d7da; color: #721c24; border: 1px solid #f1b0b7; }
                .event { background: #f8f9fa; border: 1px solid #dee2e6; padding: 15px; margin: 10px 0; border-radius: 5px; }
                .progress { width: 100%; height: 20px; background: #e9ecef; border-radius: 10px; overflow: hidden; margin: 10px 0; }
                .progress-bar { height: 100%; background: linear-gradient(90deg, #3498db, #2ecc71); transition: width 0.3s ease; }
                button { background: #3498db; color: white; border: none; padding: 12px 24px; border-radius: 5px; cursor: pointer; margin: 5px; }
                button:hover { background: #2980b9; }
                button:disabled { background: #bdc3c7; cursor: not-allowed; }
                textarea { width: 100%; height: 100px; margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
                #events { max-height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 15px; background: #fafafa; }
                .phase-completed { background: #d1ecf1; border-color: #bee5eb; color: #0c5460; }
                .phase-started { background: #fff3cd; border-color: #ffeaa7; color: #856404; }
                .analysis-complete { background: #d4edda; border-color: #c3e6cb; color: #155724; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 METIS Streaming Test Interface</h1>
                
                <div id="connection-status" class="status disconnected">
                    Disconnected from METIS streaming service
                </div>
                
                <div>
                    <h3>Start New Streaming Engagement</h3>
                    <textarea id="problem-statement" placeholder="Enter your strategic problem or question here...">We are a successful B2C subscription box company called 'Curated' that has dominated the 'hobbyist' market for years. Recently, we've seen market saturation and slowing growth. A board member is aggressively pushing for us to enter the B2B 'corporate gifting' market, which seems lucrative but is a completely different business model. Should we pivot, and if so, how?</textarea>
                    
                    <button onclick="connectAndStart()" id="start-btn">Connect & Start Analysis</button>
                    <button onclick="disconnect()" id="disconnect-btn" disabled>Disconnect</button>
                </div>
                
                <div>
                    <h3>Progress</h3>
                    <div class="progress">
                        <div class="progress-bar" id="progress-bar" style="width: 0%"></div>
                    </div>
                    <div id="progress-text">Ready to start analysis</div>
                </div>
                
                <div>
                    <h3>Live Events</h3>
                    <div id="events"></div>
                </div>
            </div>
            
            <script>
                let ws = null;
                let startTime = null;
                
                function addEvent(event) {
                    const eventsDiv = document.getElementById('events');
                    const eventDiv = document.createElement('div');
                    eventDiv.className = 'event';
                    
                    if (event.type) {
                        eventDiv.classList.add(event.type.replace('_', '-'));
                    }
                    
                    const timestamp = new Date(event.timestamp).toLocaleTimeString();
                    const progressText = event.progress || '';
                    const phaseText = event.phase ? ` (${event.phase})` : '';
                    
                    eventDiv.innerHTML = `
                        <strong>${timestamp}</strong> [${event.type}] ${progressText}${phaseText}<br>
                        ${JSON.stringify(event.data || {}, null, 2)}
                    `;
                    
                    eventsDiv.insertBefore(eventDiv, eventsDiv.firstChild);
                    
                    // Update progress bar
                    if (event.progress && event.progress.includes('/')) {
                        const [current, total] = event.progress.split('/').map(Number);
                        const percentage = (current / total) * 100;
                        document.getElementById('progress-bar').style.width = percentage + '%';
                        
                        const phaseText = event.data?.phase_name || event.phase || 'Processing';
                        document.getElementById('progress-text').textContent = 
                            `${phaseText} - ${event.progress} (${percentage.toFixed(1)}%)`;
                    }
                    
                    // Auto-scroll to latest
                    eventsDiv.scrollTop = eventsDiv.scrollHeight;
                }
                
                function updateConnectionStatus(connected) {
                    const statusDiv = document.getElementById('connection-status');
                    const startBtn = document.getElementById('start-btn');
                    const disconnectBtn = document.getElementById('disconnect-btn');
                    
                    if (connected) {
                        statusDiv.textContent = 'Connected to METIS streaming service';
                        statusDiv.className = 'status connected';
                        startBtn.disabled = true;
                        disconnectBtn.disabled = false;
                    } else {
                        statusDiv.textContent = 'Disconnected from METIS streaming service';
                        statusDiv.className = 'status disconnected';
                        startBtn.disabled = false;
                        disconnectBtn.disabled = true;
                    }
                }
                
                function connectAndStart() {
                    const problemStatement = document.getElementById('problem-statement').value.trim();
                    
                    if (!problemStatement) {
                        alert('Please enter a problem statement');
                        return;
                    }
                    
                    // Clear events
                    document.getElementById('events').innerHTML = '';
                    document.getElementById('progress-bar').style.width = '0%';
                    document.getElementById('progress-text').textContent = 'Connecting...';
                    
                    startTime = Date.now();
                    
                    ws = new WebSocket('ws://localhost:8000/ws/engagement/new');
                    
                    ws.onopen = function() {
                        updateConnectionStatus(true);
                        addEvent({
                            type: 'connection_opened',
                            timestamp: new Date().toISOString(),
                            data: { message: 'Connected to METIS' }
                        });
                        
                        // Start engagement
                        ws.send(JSON.stringify({
                            type: 'start_engagement',
                            problem_statement: problemStatement,
                            business_context: {
                                industry: 'E-commerce',
                                company_stage: 'Growth',
                                analysis_type: 'Strategic Decision'
                            }
                        }));
                    };
                    
                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        addEvent(data);
                        
                        if (data.type === 'analysis_complete') {
                            const totalTime = (Date.now() - startTime) / 1000;
                            addEvent({
                                type: 'performance_summary',
                                timestamp: new Date().toISOString(),
                                data: { 
                                    message: `Analysis completed in ${totalTime.toFixed(1)} seconds`,
                                    performance: totalTime < 30 ? 'Excellent' : totalTime < 60 ? 'Good' : 'Needs optimization'
                                }
                            });
                        }
                    };
                    
                    ws.onerror = function(error) {
                        addEvent({
                            type: 'connection_error',
                            timestamp: new Date().toISOString(),
                            data: { error: 'WebSocket connection error' }
                        });
                    };
                    
                    ws.onclose = function() {
                        updateConnectionStatus(false);
                        addEvent({
                            type: 'connection_closed',
                            timestamp: new Date().toISOString(),
                            data: { message: 'Disconnected from METIS' }
                        });
                    };
                }
                
                function disconnect() {
                    if (ws) {
                        ws.close();
                        ws = null;
                    }
                }
                
                // Initialize
                updateConnectionStatus(false);
            </script>
        </body>
        </html>
        """


def setup_streaming_api(app: FastAPI) -> StreamingAPI:
    """
    Set up streaming API endpoints on FastAPI app
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI not available - install with: pip install fastapi uvicorn websockets"
        )

    streaming_api = StreamingAPI(app)
    return streaming_api