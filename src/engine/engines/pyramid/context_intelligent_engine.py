"""
Context-Intelligent Pyramid Synthesis Engine - Sprint 2.1
F004: Revolutionary pyramid synthesis using AI's cognitive exhaust

This module creates the world's first consulting deliverable system that uses
AI's own thinking process for maximum persuasive impact and logical coherence.

Integration with Context Intelligence Revolution enables:
- Cognitive exhaust-driven argument structuring
- Historical pattern recognition for persuasion optimization
- Context-aware quality assessment and validation
- Multi-layer caching for deliverable templates
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

from src.core.enhanced_event_bus import (
    EnhancedKafkaEventBus as MetisEventBus,
    CloudEvent,
)

# State manager with fallback for development
try:
    from src.core.state_management import DistributedStateManager, StateType

    STATE_MANAGER_AVAILABLE = True
except Exception:
    STATE_MANAGER_AVAILABLE = False

    # Mock state manager for development
    class MockStateManager:
        def __init__(self):
            self.data = {}

        async def set_state(self, key, value, state_type=None):
            self.data[key] = value

        async def get_state(self, key):
            return self.data.get(key)

    # Mock event bus for development
    class MockEventBus:
        def __init__(self):
            pass

        async def publish(self, event_type, data):
            pass

        async def subscribe(self, event_type, callback):
            pass

    DistributedStateManager = MockStateManager
    MetisEventBus = MockEventBus
    StateType = None

# Pyramid components
from .models import ExecutiveDeliverable
from .enums import DeliverableType
from .context_aware_builder import ContextIntelligentPyramidBuilder
from .context_aware_quality import (
    create_context_aware_quality_assessor,
)
from .multi_persona_formatter import (
    create_multi_persona_formatter,
    ExecutivePersona,
)
from .engine import PyramidEngine

# Context Intelligence integration
try:
    from src.interfaces.context_intelligence_interface import IContextIntelligence

    CONTEXT_INTELLIGENCE_AVAILABLE = True
except ImportError:
    CONTEXT_INTELLIGENCE_AVAILABLE = False
    IContextIntelligence = Any


class ContextIntelligentPyramidEngine(PyramidEngine):
    """
    Context-Intelligent Pyramid Synthesis Engine

    Enhances traditional Pyramid Principle with Context Intelligence Revolution:
    - Uses AI's cognitive exhaust for logical flow optimization
    - Applies Manus taxonomy for content classification
    - Leverages historical patterns for maximum persuasive impact
    - Implements multi-layer caching for template optimization
    """

    def __init__(
        self,
        state_manager: Optional[DistributedStateManager] = None,
        event_bus: Optional[MetisEventBus] = None,
        context_intelligence: Optional[IContextIntelligence] = None,
    ):
        """Initialize with Context Intelligence integration"""

        # Initialize base pyramid engine with mocks if needed
        if state_manager is None:
            state_manager = DistributedStateManager()
        if event_bus is None:
            event_bus = MetisEventBus()

        super().__init__(state_manager, event_bus)

        # Context Intelligence integration
        self.context_intelligence = context_intelligence
        self.context_enhanced = context_intelligence is not None

        # Replace traditional builder with context-intelligent version
        self.pyramid_builder = ContextIntelligentPyramidBuilder(context_intelligence)

        # Enhanced components
        self.context_aware_quality_assessor = create_context_aware_quality_assessor(
            context_intelligence
        )  # Sprint 2.2
        self.multi_persona_formatter = create_multi_persona_formatter(
            context_intelligence
        )  # Sprint 2.3

        # Context Intelligence metrics
        self.context_intelligence_stats = {
            "deliverables_enhanced": 0,
            "cognitive_coherence_improvements": 0,
            "template_cache_hits": 0,
            "historical_pattern_matches": 0,
        }

        if self.context_enhanced:
            self.logger.info(
                "🧠 Context-Intelligent Pyramid Engine initialized with cognitive exhaust integration"
            )
        else:
            self.logger.info(
                "📊 Traditional Pyramid Engine initialized (Context Intelligence not available)"
            )

    async def synthesize_context_aware_deliverable(
        self,
        engagement_data: Dict[str, Any],
        deliverable_type: DeliverableType = DeliverableType.EXECUTIVE_SUMMARY,
        engagement_id: Optional[str] = None,
        cognitive_coherence_scores: Optional[List[float]] = None,
        target_persona: Optional[str] = None,
    ) -> ExecutiveDeliverable:
        """
        Create context-intelligent executive deliverable with cognitive enhancement

        Args:
            engagement_data: Analysis data from cognitive engine
            deliverable_type: Type of deliverable to generate
            engagement_id: Engagement identifier for context retrieval
            cognitive_coherence_scores: Cognitive coherence from Operation Mindforge
            target_persona: Target audience persona (CEO, CTO, Board, etc.)

        Returns:
            Enhanced executive deliverable with context intelligence
        """

        engagement_id = engagement_id or str(uuid.uuid4())

        self.logger.info(
            f"🏗️ Synthesizing context-aware {deliverable_type.value} deliverable for engagement {engagement_id}"
        )

        start_time = datetime.now()

        # Phase 1: Context Intelligence Pre-Analysis
        context_preparation = await self._prepare_context_intelligence_analysis(
            engagement_data, engagement_id, cognitive_coherence_scores
        )

        # Phase 2: Extract engagement components with context awareness
        insights = engagement_data.get("insights", [])
        hypotheses = engagement_data.get("hypotheses", [])
        frameworks_results = engagement_data.get("frameworks_results", [])
        analysis_findings = engagement_data.get("analysis_findings", {})

        # Phase 3: Build context-aware pyramid structure
        if self.context_enhanced:
            pyramid = await self.pyramid_builder.build_context_aware_pyramid_structure(
                insights=insights,
                hypotheses=hypotheses,
                frameworks_results=frameworks_results,
                analysis_findings=analysis_findings,
                engagement_id=engagement_id,
                cognitive_coherence_scores=cognitive_coherence_scores,
            )
        else:
            # Fallback to traditional pyramid building
            pyramid = await self.pyramid_builder.build_pyramid_structure(
                insights, hypotheses, frameworks_results, analysis_findings
            )

        # Phase 4: Generate context-enhanced deliverable content with persona adaptation
        enhanced_engagement_data = {
            **engagement_data,
            "context_intelligence_metadata": context_preparation,
            "pyramid_metadata": (
                pyramid.metadata if hasattr(pyramid, "metadata") else {}
            ),
            "target_persona": target_persona,
        }

        # Use multi-persona adaptation if target persona is specified
        if target_persona and target_persona != "general":
            try:
                # Convert string persona to enum if needed
                if isinstance(target_persona, str):
                    persona_enum = ExecutivePersona(target_persona.lower())
                else:
                    persona_enum = target_persona

                deliverable = await self.multi_persona_formatter.generate_persona_adapted_deliverable(
                    pyramid=pyramid,
                    deliverable_type=deliverable_type,
                    engagement_data=enhanced_engagement_data,
                    target_persona=persona_enum,
                    context_metadata=context_preparation,
                    engagement_id=engagement_id,
                )
            except (ValueError, AttributeError) as e:
                self.logger.warning(
                    f"⚠️ Persona adaptation failed, using standard formatting: {e}"
                )
                deliverable = (
                    await self.multi_persona_formatter.generate_deliverable_content(
                        pyramid, deliverable_type, enhanced_engagement_data
                    )
                )
        else:
            deliverable = (
                await self.multi_persona_formatter.generate_deliverable_content(
                    pyramid, deliverable_type, enhanced_engagement_data
                )
            )

        # Phase 5: Context-aware quality assessment
        quality_results = await self._assess_context_aware_quality(
            deliverable, context_preparation, engagement_id
        )

        # Update deliverable with quality results
        deliverable.partner_ready_score = quality_results.get(
            "partner_ready_score", 0.85
        )

        # Phase 6: Store with context intelligence metadata
        await self._store_context_aware_deliverable(
            deliverable, context_preparation, engagement_id
        )

        # Phase 7: Update context intelligence stats
        await self._update_context_intelligence_stats(
            deliverable, context_preparation, quality_results
        )

        # Phase 8: Emit enhanced completion event
        processing_time = (datetime.now() - start_time).total_seconds()

        await self.event_bus.publish_event(
            CloudEvent(
                type="pyramid.context_intelligent_synthesis.completed",
                source="pyramid/context_intelligent_engine",
                data={
                    "deliverable_id": str(deliverable.deliverable_id),
                    "type": deliverable_type.value,
                    "engagement_id": engagement_id,
                    "partner_ready_score": deliverable.partner_ready_score,
                    "word_count": len(deliverable.executive_summary.split()),
                    "context_intelligence_applied": self.context_enhanced,
                    "processing_time_seconds": processing_time,
                    "cognitive_coherence_score": context_preparation.get(
                        "overall_coherence", 0.0
                    ),
                    "historical_patterns_matched": context_preparation.get(
                        "historical_patterns_count", 0
                    ),
                    "context_enhancement_stats": self.context_intelligence_stats,
                },
            )
        )

        self.logger.info(
            f"✅ Context-intelligent deliverable synthesis complete in {processing_time:.2f}s"
        )

        return deliverable

    async def _prepare_context_intelligence_analysis(
        self,
        engagement_data: Dict[str, Any],
        engagement_id: str,
        cognitive_coherence_scores: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Prepare context intelligence analysis for pyramid synthesis
        """

        if not self.context_enhanced:
            return {
                "context_intelligence_available": False,
                "overall_coherence": 0.75,  # Default assumption
                "enhancement_applied": False,
            }

        try:
            # Extract all textual content for analysis
            all_content = []

            # From insights
            insights = engagement_data.get("insights", [])
            all_content.extend(insights)

            # From hypotheses
            hypotheses = engagement_data.get("hypotheses", [])
            all_content.extend([h.get("statement", "") for h in hypotheses])

            # From frameworks results
            frameworks_results = engagement_data.get("frameworks_results", [])
            for result in frameworks_results:
                if isinstance(result.get("output"), dict):
                    all_content.append(str(result["output"])[:500])  # Limit length

            # Get relevant historical contexts
            current_query = (
                " ".join(all_content[:2])
                if all_content
                else "Strategic analysis deliverable"
            )

            relevant_contexts = await self.context_intelligence.get_relevant_context(
                current_query=current_query, max_contexts=5, engagement_id=engagement_id
            )

            # Analyze with Manus Taxonomy for deliverable optimization
            if all_content:
                context_analysis = await self.context_intelligence.analyze_contexts_with_manus_taxonomy(
                    context_contents=all_content[:10],  # Limit to top 10 items
                    current_query=current_query,
                    engagement_id=engagement_id,
                    cognitive_coherence_scores=cognitive_coherence_scores,
                )

                return {
                    "context_intelligence_available": True,
                    "context_analysis": context_analysis,
                    "relevant_contexts": relevant_contexts,
                    "overall_coherence": self._extract_coherence_score(
                        context_analysis
                    ),
                    "primary_context_type": self._extract_primary_context_type(
                        context_analysis
                    ),
                    "historical_patterns_count": len(relevant_contexts),
                    "enhancement_applied": True,
                    "analysis_timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            self.logger.warning(f"⚠️ Context intelligence preparation failed: {e}")

        return {
            "context_intelligence_available": False,
            "overall_coherence": 0.75,
            "enhancement_applied": False,
            "error": "Context intelligence analysis failed",
        }

    async def _assess_context_aware_quality(
        self,
        deliverable: ExecutiveDeliverable,
        context_preparation: Dict[str, Any],
        engagement_id: str,
    ) -> Dict[str, Any]:
        """
        Assess deliverable quality with context intelligence enhancement
        """

        try:
            # Use Context-Aware Quality Assessor for comprehensive assessment
            quality_results = await self.context_aware_quality_assessor.assess_context_aware_deliverable_quality(
                deliverable=deliverable,
                pyramid_structure=getattr(deliverable, "pyramid_structure", None),
                context_metadata=context_preparation,
                engagement_id=engagement_id,
                target_persona=None,  # Will be enhanced in Sprint 2.3
            )

            # Extract quality assessment data
            final_quality = quality_results.get("quality_assessment", {})
            optimization_recommendations = quality_results.get(
                "optimization_recommendations", []
            )

            return {
                "partner_ready_score": final_quality.get("partner_ready_score", 0.8),
                "context_enhanced": True,
                "quality_assessment": final_quality,
                "optimization_recommendations": optimization_recommendations,
                "context_analysis": quality_results.get("context_analysis", {}),
                "quality_dimensions": final_quality.get("quality_dimensions", {}),
                "cognitive_coherence_score": final_quality.get(
                    "cognitive_coherence_score", 0.75
                ),
                "historical_effectiveness_score": final_quality.get(
                    "historical_effectiveness_score", 0.8
                ),
            }

        except Exception as e:
            self.logger.warning(
                f"⚠️ Context-aware quality assessment failed, using fallback: {e}"
            )

            # Fallback to traditional assessment
            await self.context_aware_quality_assessor.assess_deliverable_quality(
                deliverable
            )

            return {
                "partner_ready_score": getattr(deliverable, "partner_ready_score", 0.8),
                "context_enhanced": False,
                "fallback_used": True,
                "error": str(e),
            }

    async def _store_context_aware_deliverable(
        self,
        deliverable: ExecutiveDeliverable,
        context_preparation: Dict[str, Any],
        engagement_id: str,
    ) -> None:
        """
        Store deliverable with context intelligence metadata
        """

        # Enhanced serialization with context metadata
        deliverable_data = self.multi_persona_formatter.serialize_deliverable(
            deliverable
        )

        # Add context intelligence metadata
        deliverable_data["context_intelligence_metadata"] = {
            "enhancement_applied": context_preparation.get(
                "enhancement_applied", False
            ),
            "overall_coherence": context_preparation.get("overall_coherence", 0.0),
            "primary_context_type": context_preparation.get(
                "primary_context_type", "DOMAIN"
            ),
            "historical_patterns_matched": context_preparation.get(
                "historical_patterns_count", 0
            ),
            "engagement_id": engagement_id,
            "synthesis_timestamp": datetime.now().isoformat(),
            "context_intelligence_version": "Sprint_2.1",
        }

        # Store enhanced deliverable
        await self.state_manager.set_state(
            f"context_deliverable_{deliverable.deliverable_id}",
            deliverable_data,
            StateType.DELIVERABLE if STATE_MANAGER_AVAILABLE else None,
        )

        # Store cognitive exhaust if Context Intelligence is available
        if self.context_intelligence and context_preparation.get("enhancement_applied"):
            await self._store_deliverable_cognitive_exhaust(
                deliverable, context_preparation, engagement_id
            )

    async def _store_deliverable_cognitive_exhaust(
        self,
        deliverable: ExecutiveDeliverable,
        context_preparation: Dict[str, Any],
        engagement_id: str,
    ) -> None:
        """
        Store deliverable synthesis cognitive exhaust for future context intelligence
        """

        try:
            thinking_process = f"""
            Context-Intelligent Deliverable Synthesis Process:
            
            Executive Summary: {deliverable.executive_summary[:200]}...
            Key Messages: {len(deliverable.key_messages)} strategic messages
            Recommendations: {len(deliverable.recommendations)} actionable recommendations
            
            Context Intelligence Enhancement:
            - Overall Coherence: {context_preparation.get('overall_coherence', 'N/A')}
            - Primary Context Type: {context_preparation.get('primary_context_type', 'N/A')}
            - Historical Patterns: {context_preparation.get('historical_patterns_count', 0)}
            - Partner Ready Score: {deliverable.partner_ready_score}
            
            The deliverable follows pyramid principle with context-aware optimization for maximum impact.
            """

            cleaned_response = "Context-intelligent executive deliverable created with enhanced logical coherence and persuasive structure."

            await self.context_intelligence.store_cognitive_exhaust_triple_layer(
                engagement_id=engagement_id,
                phase="deliverable_synthesis",
                mental_model="context_intelligent_pyramid_synthesis",
                thinking_process=thinking_process,
                cleaned_response=cleaned_response,
                confidence=context_preparation.get("overall_coherence", 0.85),
            )

            self.logger.info(
                f"💾 Deliverable synthesis cognitive exhaust stored for engagement {engagement_id}"
            )

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to store deliverable cognitive exhaust: {e}")

    async def _update_context_intelligence_stats(
        self,
        deliverable: ExecutiveDeliverable,
        context_preparation: Dict[str, Any],
        quality_results: Dict[str, Any],
    ) -> None:
        """
        Update context intelligence usage statistics
        """

        if context_preparation.get("enhancement_applied"):
            self.context_intelligence_stats["deliverables_enhanced"] += 1

            # Track coherence improvements
            coherence_bonus = quality_results.get("coherence_bonus", 0)
            if coherence_bonus > 0:
                self.context_intelligence_stats["cognitive_coherence_improvements"] += 1

            # Track historical pattern matches
            if context_preparation.get("historical_patterns_count", 0) > 0:
                self.context_intelligence_stats["historical_pattern_matches"] += 1

    def get_context_intelligence_stats(self) -> Dict[str, Any]:
        """
        Get context intelligence enhancement statistics
        """
        return {
            "context_intelligence_enabled": self.context_enhanced,
            "enhancement_stats": self.context_intelligence_stats,
            "version": "Sprint_2.1_Context_Intelligent_Pyramid",
            "capabilities": [
                "cognitive_exhaust_driven_structuring",
                "historical_pattern_recognition",
                "context_aware_quality_assessment",
                "multi_layer_template_caching",
            ],
        }

    # Utility methods for context analysis extraction

    def _extract_coherence_score(self, context_analysis: Any) -> float:
        """Extract coherence score from context analysis"""
        try:
            if hasattr(context_analysis, "overall_score"):
                return float(context_analysis.overall_score)
            return 0.75
        except (ValueError, AttributeError):
            return 0.75

    def _extract_primary_context_type(self, context_analysis: Any) -> str:
        """Extract primary context type from analysis"""
        try:
            if hasattr(context_analysis, "primary_context_type"):
                return context_analysis.primary_context_type
            return "DOMAIN"
        except AttributeError:
            return "DOMAIN"


# Factory function for creating context-intelligent pyramid engine
def create_context_intelligent_pyramid_engine(
    state_manager: DistributedStateManager,
    event_bus: MetisEventBus,
    context_intelligence: Optional[IContextIntelligence] = None,
) -> ContextIntelligentPyramidEngine:
    """
    Factory function for creating Context-Intelligent Pyramid Engine
    """
    return ContextIntelligentPyramidEngine(
        state_manager=state_manager,
        event_bus=event_bus,
        context_intelligence=context_intelligence,
    )
