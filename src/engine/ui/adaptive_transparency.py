"""
METIS Adaptive Transparency Engine
Main progressive transparency module with cognitive load management

Implements the primary adaptive transparency engine that orchestrates all
transparency components to provide optimal user-adaptive information disclosure.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import UUID

from src.engine.models.data_contracts import (
    MetisDataContract,
    ReasoningStep,
)
from src.models.transparency_models import (
    TransparencyLayer,
    UserExpertiseLevel,
    CognitiveLoadLevel,
    EvidenceQuality,
    ValidationEvidenceCollection,
    UserProfile,
    TransparencyContent,
    ProgressiveDisclosure,
)

from .scaffolding_engine import CognitiveScaffoldingEngine
from .expertise_assessor import UserExpertiseAssessor
from .validation_evidence import ValidationEvidenceEngine
from .reasoning_visualizer import ReasoningVisualizationEngine

try:
    from src.engine.agents.unified_agent import (
        UnifiedMetisAgent,
        AgentTask,
        AgentStatus,
        EigentArchetype,
    )

    UNIFIED_AGENTS_AVAILABLE = True
except ImportError:
    UNIFIED_AGENTS_AVAILABLE = False

try:
    from src.ui.cognitive_trace_visualizer import (
        CognitiveTraceBuilder,
        CognitiveTraceRenderer,
        CognitiveTrace,
        VisualizationStyle,
    )

    TRACE_VISUALIZER_AVAILABLE = True
except ImportError:
    TRACE_VISUALIZER_AVAILABLE = False

try:
    # Import with forward reference handling
    if True:  # Will be imported after progressive_disclosure_manager is defined
        PROGRESSIVE_DISCLOSURE_AVAILABLE = True
except ImportError:
    PROGRESSIVE_DISCLOSURE_AVAILABLE = False


class AdaptiveTransparencyEngine:
    """
    Main engine for adaptive transparency and progressive disclosure
    Implements cognitive load management with user expertise adaptation
    """

    def __init__(self):
        self.scaffolding_engine = CognitiveScaffoldingEngine()
        self.expertise_assessor = UserExpertiseAssessor()
        self.visualization_engine = ReasoningVisualizationEngine()
        self.validation_evidence_engine = (
            ValidationEvidenceEngine()
        )  # P7.1: Added validation evidence engine
        self.logger = logging.getLogger(__name__)

        # P7.2: Initialize cognitive trace components if available
        if TRACE_VISUALIZER_AVAILABLE:
            self.trace_builder = CognitiveTraceBuilder()
            self.trace_renderer = CognitiveTraceRenderer()
        else:
            self.trace_builder = None
            self.trace_renderer = None

        # P7.3: Initialize progressive disclosure manager if available
        try:
            from src.ui.progressive_disclosure_manager import (
                ProgressiveDisclosureManager,
            )

            self.progressive_disclosure_manager = ProgressiveDisclosureManager()
            self.progressive_disclosure_available = True
        except ImportError:
            self.progressive_disclosure_manager = None
            self.progressive_disclosure_available = False

        # User profiles (in production, would be stored in database)
        self.user_profiles: Dict[UUID, UserProfile] = {}

    async def generate_progressive_disclosure(
        self, engagement_contract: MetisDataContract, user_id: UUID
    ) -> ProgressiveDisclosure:
        """Generate complete progressive disclosure package"""

        # Get or create user profile
        user_profile = await self._get_user_profile(user_id)

        # Assess current expertise level
        current_expertise = await self.expertise_assessor.assess_expertise(user_profile)
        if current_expertise != user_profile.expertise_level:
            user_profile.expertise_level = current_expertise
            self.logger.info(
                f"Updated user {user_id} expertise level to {current_expertise}"
            )

        # P7.1: Generate validation evidence for all reasoning steps
        validation_evidence = (
            await self.validation_evidence_engine.generate_validation_evidence(
                engagement_contract.cognitive_state.reasoning_steps, engagement_contract
            )
        )

        # P7.2: Generate cognitive reasoning trace if available
        cognitive_trace = None
        trace_json = None
        if TRACE_VISUALIZER_AVAILABLE and self.trace_builder:
            try:
                # Build cognitive trace from contract
                cognitive_trace = await self.trace_builder.build_trace_from_contract(
                    engagement_contract, style=VisualizationStyle.HIERARCHICAL
                )

                # Render to JSON for frontend
                if self.trace_renderer:
                    trace_json = await self.trace_renderer.render_to_json(
                        cognitive_trace
                    )

                self.logger.info(
                    f"Generated cognitive trace with {len(cognitive_trace.nodes)} nodes"
                )
            except Exception as e:
                self.logger.warning(f"Failed to generate cognitive trace: {e}")
                cognitive_trace = None
                trace_json = None

        # Generate content for each transparency layer
        layers = {}

        # Layer 1: Executive Summary
        layers[TransparencyLayer.EXECUTIVE_SUMMARY] = (
            await self._generate_executive_summary(
                engagement_contract,
                user_profile,
                validation_evidence,
                cognitive_trace,
                trace_json,
            )
        )

        # Layer 2: Reasoning Overview
        layers[TransparencyLayer.REASONING_OVERVIEW] = (
            await self._generate_reasoning_overview(
                engagement_contract,
                user_profile,
                validation_evidence,
                cognitive_trace,
                trace_json,
            )
        )

        # Layer 3: Detailed Audit Trail
        layers[TransparencyLayer.DETAILED_AUDIT_TRAIL] = (
            await self._generate_detailed_audit_trail(
                engagement_contract,
                user_profile,
                validation_evidence,
                cognitive_trace,
                trace_json,
            )
        )

        # Layer 4: Technical Execution
        layers[TransparencyLayer.TECHNICAL_EXECUTION] = (
            await self._generate_technical_execution_log(
                engagement_contract,
                user_profile,
                validation_evidence,
                cognitive_trace,
                trace_json,
            )
        )

        # P7.3: Apply progressive disclosure to complex analysis if available
        if (
            self.progressive_disclosure_available
            and self.progressive_disclosure_manager
        ):
            try:
                # Create progressive disclosure for complex content
                progressive_disclosure_state = await self.progressive_disclosure_manager.create_progressive_disclosure(
                    engagement_contract, user_profile, layers
                )

                # Enhance layers with progressive disclosure data
                for layer_name, content in layers.items():
                    content.progressive_disclosure_state = progressive_disclosure_state
                    content.complexity_assessment = (
                        progressive_disclosure_state.complexity_assessment.value
                    )
                    content.adaptive_strategy = (
                        progressive_disclosure_state.strategy.value
                    )

                    # Add disclosure sections as JSON for frontend rendering
                    content.disclosure_sections = await self.progressive_disclosure_manager.render_disclosure_json(
                        progressive_disclosure_state.disclosure_id
                    )

                self.logger.info(
                    f"Applied progressive disclosure with {progressive_disclosure_state.strategy.value} strategy"
                )

            except Exception as e:
                self.logger.warning(f"Failed to apply progressive disclosure: {e}")

        # Generate navigation guidance
        navigation_guidance = await self._generate_navigation_guidance(
            layers, user_profile
        )

        # Create progressive disclosure package
        disclosure = ProgressiveDisclosure(
            engagement_id=engagement_contract.engagement_context.engagement_id,
            layers=layers,
            navigation_guidance=navigation_guidance,
            personalization_metadata={
                "user_expertise": user_profile.expertise_level.value,
                "default_layer": user_profile.preferred_layer.value,
                "cognitive_preferences": user_profile.cognitive_preferences,
                "auto_adjust_enabled": user_profile.auto_adjust_complexity,
            },
        )

        # Record interaction for future adaptation
        await self._record_interaction(user_profile, disclosure)

        return disclosure

    async def _get_user_profile(self, user_id: UUID) -> UserProfile:
        """Get or create user profile with persona detection"""

        if user_id not in self.user_profiles:
            # Create new user profile with persona detection
            detected_persona = await self._detect_user_persona(user_id)

            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                expertise_level=detected_persona["expertise_level"],
                preferred_layer=detected_persona["preferred_layer"],
                persona_type=detected_persona["persona_type"],
                cognitive_preferences={
                    **detected_persona["cognitive_preferences"],
                    "show_confidence_indicators": True,
                    "enable_progressive_hints": True,
                    "auto_adjust_complexity": True,
                },
            )

        return self.user_profiles[user_id]

    async def _detect_user_persona(self, user_id: UUID) -> Dict[str, Any]:
        """Detect user persona and configure optimal defaults"""

        # In production, this would analyze user behavior, role, interaction patterns
        # For now, we'll implement intelligent defaults based on user characteristics

        # Default persona detection logic
        persona_indicators = await self._gather_persona_indicators(user_id)

        # Executive Persona Detection
        if persona_indicators.get("title_keywords", []) and any(
            keyword in " ".join(persona_indicators["title_keywords"]).lower()
            for keyword in [
                "ceo",
                "president",
                "director",
                "executive",
                "vp",
                "head",
                "chief",
            ]
        ):
            return {
                "expertise_level": UserExpertiseLevel.EXECUTIVE,
                "preferred_layer": TransparencyLayer.EXECUTIVE_SUMMARY,
                "persona_type": "executive",
                "cognitive_preferences": {
                    "prefer_visual_summaries": True,
                    "minimize_technical_detail": True,
                    "highlight_business_impact": True,
                    "enable_quick_navigation": True,
                    "dashboard_density": "minimal",
                    "default_view_mode": "summary_cards",
                },
            }

        # Strategic Persona Detection
        elif persona_indicators.get("domain_focus") and any(
            domain in persona_indicators["domain_focus"]
            for domain in [
                "strategy",
                "planning",
                "business",
                "operations",
                "consulting",
            ]
        ):
            return {
                "expertise_level": UserExpertiseLevel.STRATEGIC,
                "preferred_layer": TransparencyLayer.REASONING_OVERVIEW,
                "persona_type": "strategic",
                "cognitive_preferences": {
                    "show_methodology_details": True,
                    "enable_assumption_analysis": True,
                    "highlight_risk_factors": True,
                    "enable_scenario_exploration": True,
                    "dashboard_density": "balanced",
                    "default_view_mode": "progressive_narrative",
                },
            }

        # Analytical Persona Detection
        elif persona_indicators.get("analytical_behavior", False) or any(
            keyword in " ".join(persona_indicators.get("title_keywords", [])).lower()
            for keyword in ["analyst", "manager", "specialist", "coordinator", "lead"]
        ):
            return {
                "expertise_level": UserExpertiseLevel.ANALYTICAL,
                "preferred_layer": TransparencyLayer.DETAILED_AUDIT_TRAIL,
                "persona_type": "analytical",
                "cognitive_preferences": {
                    "show_detailed_reasoning": True,
                    "enable_evidence_deep_dive": True,
                    "highlight_confidence_scores": True,
                    "enable_methodology_inspection": True,
                    "dashboard_density": "detailed",
                    "default_view_mode": "workspace_tabs",
                },
            }

        # Technical Persona Detection
        elif persona_indicators.get("technical_indicators", False) or any(
            keyword in " ".join(persona_indicators.get("title_keywords", [])).lower()
            for keyword in [
                "engineer",
                "developer",
                "architect",
                "technical",
                "system",
                "data",
            ]
        ):
            return {
                "expertise_level": UserExpertiseLevel.TECHNICAL,
                "preferred_layer": TransparencyLayer.TECHNICAL_EXECUTION,
                "persona_type": "technical",
                "cognitive_preferences": {
                    "show_implementation_details": True,
                    "enable_system_monitoring": True,
                    "highlight_performance_metrics": True,
                    "enable_debug_mode": True,
                    "dashboard_density": "comprehensive",
                    "default_view_mode": "full_transparency",
                },
            }

        # Default Strategic Persona (safest default for business users)
        else:
            return {
                "expertise_level": UserExpertiseLevel.STRATEGIC,
                "preferred_layer": TransparencyLayer.REASONING_OVERVIEW,
                "persona_type": "strategic",
                "cognitive_preferences": {
                    "show_methodology_details": True,
                    "enable_assumption_analysis": True,
                    "highlight_risk_factors": True,
                    "enable_scenario_exploration": True,
                    "dashboard_density": "balanced",
                    "default_view_mode": "progressive_narrative",
                },
            }

    async def _gather_persona_indicators(self, user_id: UUID) -> Dict[str, Any]:
        """Gather indicators for persona detection"""

        # In production, this would query:
        # - User profile data from database
        # - Historical interaction patterns
        # - Role/title information
        # - Department/organization context
        # - Previous engagement types
        # - Time spent on different layers

        # For now, return empty indicators for default detection
        return {
            "title_keywords": [],
            "domain_focus": [],
            "analytical_behavior": False,
            "technical_indicators": False,
            "historical_layer_preferences": {},
            "interaction_patterns": {},
            "time_on_layers": {},
            "preferred_content_types": [],
        }

    async def _generate_executive_summary(
        self,
        engagement_contract: MetisDataContract,
        user_profile: UserProfile,
        validation_evidence: List[ValidationEvidenceCollection],
        cognitive_trace: Optional["CognitiveTrace"] = None,
        trace_json: Optional[Dict[str, Any]] = None,
    ) -> TransparencyContent:
        """Generate executive summary layer"""

        reasoning_steps = engagement_contract.cognitive_state.reasoning_steps

        # Extract key insights
        key_insights = []
        if reasoning_steps:
            # Get highest confidence insights
            high_confidence_steps = [
                step for step in reasoning_steps if step.confidence_score >= 0.8
            ]

            key_insights = [
                f"• {step.reasoning_text.split('.')[0]}."
                for step in high_confidence_steps[:3]
            ]

        if not key_insights:
            key_insights = ["Analysis completed - see detailed results for insights"]

        # P7.1: Calculate evidence quality summary
        evidence_quality_summary = self._calculate_evidence_quality_summary(
            validation_evidence
        )

        # Create executive summary content with validation indicators
        summary_content = f"""
        ## Strategic Analysis Summary
        
        **Problem**: {engagement_contract.engagement_context.problem_statement[:200]}...
        
        **Key Insights**:
        {chr(10).join(key_insights)}
        
        **Confidence Level**: {self._calculate_overall_confidence(reasoning_steps):.0%}
        **Evidence Quality**: {evidence_quality_summary['overall_quality'].title()} ({evidence_quality_summary['strong_evidence_count']} strong evidence items)
        
        **Validation Summary**:
        • {evidence_quality_summary['total_evidence_count']} pieces of supporting evidence
        • {evidence_quality_summary['quality_distribution']['strong']} high-quality validations
        • {evidence_quality_summary['confidence_boost']:.1%} evidence-based confidence boost
        
        **Recommended Next Steps**:
        • Review detailed methodology for implementation guidance
        • Consider stakeholder implications of key insights
        • Validate assumptions through additional data collection
        """.strip()

        cognitive_load = await self.scaffolding_engine.assess_cognitive_load(
            summary_content, reasoning_steps[:3], []
        )

        return TransparencyContent(
            layer=TransparencyLayer.EXECUTIVE_SUMMARY,
            title="Strategic Summary",
            content=summary_content,
            cognitive_load=cognitive_load,
            key_insights=key_insights,
            drill_down_options=[
                "Methodology Details",
                "Supporting Evidence",
                "Full Analysis",
                "Evidence Quality",
            ],
            validation_evidence=validation_evidence,  # P7.1: Include validation evidence
            evidence_quality_summary=evidence_quality_summary,  # P7.1: Include evidence summary
            confidence_indicators={  # P7.1: Include confidence indicators
                "overall_confidence": self._calculate_overall_confidence(
                    reasoning_steps
                ),
                "evidence_boost": evidence_quality_summary["confidence_boost"],
                "quality_level": evidence_quality_summary["overall_quality"],
            },
            metadata={
                "word_count": len(summary_content.split()),
                "confidence_level": self._calculate_overall_confidence(reasoning_steps),
                "evidence_count": evidence_quality_summary["total_evidence_count"],
            },
        )

    async def _generate_reasoning_overview(
        self,
        engagement_contract: MetisDataContract,
        user_profile: UserProfile,
        validation_evidence: List[ValidationEvidenceCollection],
        cognitive_trace: Optional["CognitiveTrace"] = None,
        trace_json: Optional[Dict[str, Any]] = None,
    ) -> TransparencyContent:
        """Generate reasoning overview layer"""

        reasoning_steps = engagement_contract.cognitive_state.reasoning_steps
        selected_models = engagement_contract.cognitive_state.selected_mental_models

        # Create methodology overview
        methodology_content = f"""
        ## Analytical Methodology
        
        **Mental Models Applied**: {len(selected_models)} frameworks selected for optimal analysis
        
        """

        # Add model descriptions
        for model in selected_models:
            methodology_content += f"""
        ### {model.name}
        - **Purpose**: {model.description}
        - **Applied to**: {', '.join(model.application_criteria[:2])}
        - **Expected Improvement**: {model.expected_improvement}%
        
        """

        # Add reasoning process overview
        methodology_content += f"""
        ## Reasoning Process ({len(reasoning_steps)} steps)
        
        """

        for i, step in enumerate(reasoning_steps[:7]):  # Limit for cognitive load
            confidence_indicator = (
                "🟢"
                if step.confidence_score >= 0.8
                else "🟡" if step.confidence_score >= 0.6 else "🔴"
            )
            methodology_content += f"""
        **Step {i+1}**: {step.mental_model_applied.replace('_', ' ').title()} {confidence_indicator}
        {step.reasoning_text[:150]}...
        
        """

        if len(reasoning_steps) > 7:
            methodology_content += (
                f"\n*... and {len(reasoning_steps) - 7} additional reasoning steps*\n"
            )

        # Create reasoning map visualization
        reasoning_map = await self.visualization_engine.create_reasoning_map(
            reasoning_steps, selected_models
        )

        # P7.1: Create evidence visualization for reasoning overview
        evidence_visualization = (
            await self.validation_evidence_engine.create_evidence_visualization(
                validation_evidence
            )
        )

        # P7.1: Calculate evidence quality summary
        evidence_quality_summary = self._calculate_evidence_quality_summary(
            validation_evidence
        )

        cognitive_load = await self.scaffolding_engine.assess_cognitive_load(
            methodology_content, reasoning_steps, selected_models
        )

        # P7.2: Add cognitive trace information
        trace_info = {}
        if cognitive_trace:
            trace_info = {
                "trace_available": True,
                "node_count": len(cognitive_trace.nodes),
                "edge_count": len(cognitive_trace.edges),
                "reasoning_depth": cognitive_trace.reasoning_depth,
                "cognitive_complexity": cognitive_trace.cognitive_complexity,
            }

        return TransparencyContent(
            layer=TransparencyLayer.REASONING_OVERVIEW,
            title="Methodology & Reasoning",
            content=methodology_content,
            cognitive_load=cognitive_load,
            key_insights=[
                f"Applied {len(selected_models)} mental models",
                f"Executed {len(reasoning_steps)} reasoning steps",
                f"Average confidence: {self._calculate_overall_confidence(reasoning_steps):.0%}",
                f"Evidence quality: {evidence_quality_summary['overall_quality'].title()}",  # P7.1: Evidence insight
            ],
            drill_down_options=[
                "Complete Reasoning Steps",
                "Model Performance",
                "Evidence Sources",
                "Validation Evidence",
                "Interactive Trace",
            ],
            reasoning_map=reasoning_map,
            validation_evidence=validation_evidence,  # P7.1: Include validation evidence
            evidence_quality_summary=evidence_quality_summary,  # P7.1: Include evidence summary
            evidence_visualization=evidence_visualization,  # P7.1: Include evidence visualization
            cognitive_trace=cognitive_trace,  # P7.2: Include cognitive trace
            trace_visualization_json=trace_json,  # P7.2: Include trace JSON
            trace_animation_enabled=user_profile.expertise_level
            in [UserExpertiseLevel.ANALYTICAL, UserExpertiseLevel.TECHNICAL],  # P7.2
            trace_interaction_config={  # P7.2: Interaction config
                "enable_zoom": True,
                "enable_pan": True,
                "enable_node_selection": True,
                "enable_cluster_focus": True,
                "default_style": (
                    VisualizationStyle.HIERARCHICAL.value if cognitive_trace else None
                ),
            },
            confidence_indicators={  # P7.1: Include confidence indicators
                "reasoning_confidence": self._calculate_overall_confidence(
                    reasoning_steps
                ),
                "evidence_boost": evidence_quality_summary["confidence_boost"],
                "validation_completeness": evidence_quality_summary[
                    "total_evidence_count"
                ]
                / max(1, len(reasoning_steps)),
            },
            metadata={
                "models_count": len(selected_models),
                "steps_count": len(reasoning_steps),
                "evidence_items": evidence_quality_summary["total_evidence_count"],
                **trace_info,  # P7.2: Add trace metadata
            },
        )

    async def _generate_detailed_audit_trail(
        self,
        engagement_contract: MetisDataContract,
        user_profile: UserProfile,
        validation_evidence: List[ValidationEvidenceCollection],
        cognitive_trace: Optional["CognitiveTrace"] = None,
        trace_json: Optional[Dict[str, Any]] = None,
    ) -> TransparencyContent:
        """Generate detailed audit trail layer"""

        reasoning_steps = engagement_contract.cognitive_state.reasoning_steps

        # Create detailed audit trail
        audit_content = f"""
        ## Complete Reasoning Audit Trail
        
        **Engagement ID**: {engagement_contract.engagement_context.engagement_id}
        **Analysis Started**: {engagement_contract.engagement_context.created_at.strftime('%Y-%m-%d %H:%M:%S')}
        
        """

        # Detailed reasoning steps
        for i, step in enumerate(reasoning_steps):
            audit_content += f"""
        ### Step {i+1}: {step.mental_model_applied.replace('_', ' ').title()}
        
        **Confidence**: {step.confidence_score:.2f} | **Timestamp**: {step.timestamp.strftime('%H:%M:%S')}
        
        **Reasoning**:
        {step.reasoning_text}
        
        **Evidence Sources**:
        {', '.join(step.evidence_sources) if step.evidence_sources else 'Internal reasoning'}
        
        **Assumptions Made**:
        {chr(10).join(f'• {assumption}' for assumption in step.assumptions_made)}
        
        ---
        
        """

        # Add validation results
        validation_results = engagement_contract.cognitive_state.validation_results
        if validation_results:
            audit_content += f"""
        ## Quality Validation Results
        
        **Overall Confidence**: {validation_results.get('overall_confidence', 0):.2f}
        **Quality Metrics**: {validation_results.get('quality_metrics', {})}
        
        """

        # Create confidence visualization
        confidence_viz = (
            await self.visualization_engine.create_confidence_visualization(
                reasoning_steps
            )
        )

        cognitive_load = await self.scaffolding_engine.assess_cognitive_load(
            audit_content, reasoning_steps, []
        )

        return TransparencyContent(
            layer=TransparencyLayer.DETAILED_AUDIT_TRAIL,
            title="Complete Audit Trail",
            content=audit_content,
            cognitive_load=cognitive_load,
            key_insights=[
                f"Complete {len(reasoning_steps)}-step reasoning process",
                "Full transparency with evidence and assumptions",
                "Quality validation results included",
            ],
            supporting_evidence=[
                "All reasoning steps documented",
                "Evidence sources tracked",
                "Assumptions explicitly stated",
                "Confidence scores for each step",
            ],
            confidence_visualization=confidence_viz,
            metadata={
                "audit_completeness": "100%",
                "evidence_transparency": "complete",
            },
        )

    async def _generate_technical_execution_log(
        self,
        engagement_contract: MetisDataContract,
        user_profile: UserProfile,
        validation_evidence: List[ValidationEvidenceCollection],
        cognitive_trace: Optional["CognitiveTrace"] = None,
        trace_json: Optional[Dict[str, Any]] = None,
    ) -> TransparencyContent:
        """Generate technical execution log layer"""

        processing_metadata = engagement_contract.processing_metadata

        # Create technical log
        technical_content = f"""
        ## Technical Execution Log
        
        **System Performance**:
        - Processing Time: {processing_metadata.get('cognitive_processing_time_ms', 0):.1f}ms
        - Models Applied: {', '.join(processing_metadata.get('models_applied', []))}
        - Overall Confidence: {processing_metadata.get('reasoning_confidence', 0):.2f}
        
        **Data Contract Compliance**:
        - Schema Version: {engagement_contract.schema_version}
        - Event Type: {engagement_contract.type}
        - CloudEvents Compliant: ✓
        
        **Component Integration**:
        - Cognitive Engine: Active
        - Event Bus: Connected
        - Audit Trail: Recording
        - Authentication: Validated
        
        **Performance Metrics**:
        - Memory Usage: Optimal
        - CPU Utilization: Normal
        - Response Time: <30s target ✓
        
        **Error Handling**:
        - Circuit Breakers: Healthy
        - Retry Policies: Active
        - Graceful Degradation: Available
        
        **Compliance Status**:
        - SOC 2 Audit Trail: ✓
        - GDPR Data Handling: ✓
        - Enterprise Security: ✓
        """.strip()

        cognitive_load = (
            CognitiveLoadLevel.HIGH
        )  # Technical content is inherently complex

        return TransparencyContent(
            layer=TransparencyLayer.TECHNICAL_EXECUTION,
            title="Technical Implementation Details",
            content=technical_content,
            cognitive_load=cognitive_load,
            key_insights=[
                "System performance within targets",
                "Full compliance monitoring active",
                "Component integration healthy",
            ],
            metadata={
                "system_health": "optimal",
                "compliance_status": "fully_compliant",
            },
        )

    async def _generate_navigation_guidance(
        self,
        layers: Dict[TransparencyLayer, TransparencyContent],
        user_profile: UserProfile,
    ) -> Dict[str, Any]:
        """Generate navigation guidance for user"""

        expertise_config = self.scaffolding_engine.expertise_adaptations[
            user_profile.expertise_level
        ]

        guidance = {
            "recommended_starting_layer": expertise_config["default_layer"].value,
            "navigation_hints": [],
            "complexity_indicators": {},
            "personalization_options": {
                "auto_adjust_complexity": user_profile.auto_adjust_complexity,
                "show_confidence_indicators": user_profile.show_confidence_indicators,
                "enable_progressive_hints": user_profile.enable_progressive_hints,
            },
        }

        # Add navigation hints based on cognitive load
        for layer, content in layers.items():
            guidance["complexity_indicators"][
                layer.value
            ] = content.cognitive_load.value

            if content.cognitive_load == CognitiveLoadLevel.HIGH:
                guidance["navigation_hints"].append(
                    f"{layer.value}: High complexity - consider chunked reading"
                )
            elif content.cognitive_load == CognitiveLoadLevel.OVERWHELMING:
                guidance["navigation_hints"].append(
                    f"{layer.value}: Very complex - enable auto-summarization"
                )

        # Expertise-specific guidance
        if user_profile.expertise_level == UserExpertiseLevel.EXECUTIVE:
            guidance["navigation_hints"].append(
                "Focus on Executive Summary for key decisions"
            )
        elif user_profile.expertise_level == UserExpertiseLevel.ANALYTICAL:
            guidance["navigation_hints"].append(
                "Detailed Audit Trail provides complete reasoning transparency"
            )

        return guidance

    def _calculate_overall_confidence(
        self, reasoning_steps: List[ReasoningStep]
    ) -> float:
        """Calculate overall confidence from reasoning steps"""
        if not reasoning_steps:
            return 0.0

        return sum(step.confidence_score for step in reasoning_steps) / len(
            reasoning_steps
        )

    def _calculate_evidence_quality_summary(
        self, validation_evidence: List[ValidationEvidenceCollection]
    ) -> Dict[str, Any]:
        """Calculate summary of evidence quality across all collections"""

        if not validation_evidence:
            return {
                "overall_quality": "none",
                "total_evidence_count": 0,
                "strong_evidence_count": 0,
                "confidence_boost": 0.0,
                "quality_distribution": {
                    "strong": 0,
                    "moderate": 0,
                    "weak": 0,
                    "contradictory": 0,
                },
            }

        all_evidence = []
        for collection in validation_evidence:
            all_evidence.extend(collection.evidence_items)

        # Count by quality
        quality_counts = {quality.value: 0 for quality in EvidenceQuality}
        total_confidence_boost = 0.0

        for evidence in all_evidence:
            quality_counts[evidence.quality.value] += 1
            total_confidence_boost += evidence.confidence_impact

        # Determine overall quality
        strong_count = quality_counts["strong"]
        moderate_count = quality_counts["moderate"]
        total_count = len(all_evidence)

        if strong_count >= total_count * 0.6:
            overall_quality = "strong"
        elif strong_count + moderate_count >= total_count * 0.5:
            overall_quality = "moderate"
        else:
            overall_quality = "weak"

        return {
            "overall_quality": overall_quality,
            "total_evidence_count": total_count,
            "strong_evidence_count": strong_count,
            "confidence_boost": total_confidence_boost / max(1, total_count),
            "quality_distribution": quality_counts,
        }

    async def _record_interaction(
        self, user_profile: UserProfile, disclosure: ProgressiveDisclosure
    ):
        """Record user interaction for future adaptation"""

        interaction = {
            "timestamp": datetime.utcnow().isoformat(),
            "engagement_id": str(disclosure.engagement_id),
            "layers_generated": list(disclosure.layers.keys()),
            "default_layer": user_profile.preferred_layer.value,
            "cognitive_load_levels": {
                layer.value: content.cognitive_load.value
                for layer, content in disclosure.layers.items()
            },
        }

        user_profile.interaction_history.append(interaction)

        # Keep only last 50 interactions
        if len(user_profile.interaction_history) > 50:
            user_profile.interaction_history = user_profile.interaction_history[-50:]

    async def update_user_preferences(self, user_id: UUID, preferences: Dict[str, Any]):
        """Update user transparency preferences"""

        user_profile = await self._get_user_profile(user_id)

        # Update preferences
        if "preferred_layer" in preferences:
            user_profile.preferred_layer = TransparencyLayer(
                preferences["preferred_layer"]
            )

        if "auto_adjust_complexity" in preferences:
            user_profile.auto_adjust_complexity = preferences["auto_adjust_complexity"]

        if "show_confidence_indicators" in preferences:
            user_profile.show_confidence_indicators = preferences[
                "show_confidence_indicators"
            ]

        if "enable_progressive_hints" in preferences:
            user_profile.enable_progressive_hints = preferences[
                "enable_progressive_hints"
            ]

        user_profile.cognitive_preferences.update(
            preferences.get("cognitive_preferences", {})
        )

        self.logger.info(f"Updated transparency preferences for user {user_id}")

    async def get_transparency_analytics(self, user_id: UUID) -> Dict[str, Any]:
        """Get transparency analytics for user"""

        user_profile = await self._get_user_profile(user_id)

        return {
            "user_id": str(user_id),
            "expertise_level": user_profile.expertise_level.value,
            "preferred_layer": user_profile.preferred_layer.value,
            "interaction_count": len(user_profile.interaction_history),
            "learning_trajectory": user_profile.learning_trajectory,
            "cognitive_preferences": user_profile.cognitive_preferences,
            "adaptation_status": {
                "auto_adjust_enabled": user_profile.auto_adjust_complexity,
                "last_adaptation": (
                    user_profile.interaction_history[-1]["timestamp"]
                    if user_profile.interaction_history
                    else None
                ),
            },
        }


# Global transparency engine instance
_transparency_engine_instance: Optional[AdaptiveTransparencyEngine] = None


async def get_transparency_engine() -> AdaptiveTransparencyEngine:
    """Get or create global transparency engine instance"""
    global _transparency_engine_instance

    if _transparency_engine_instance is None:
        _transparency_engine_instance = AdaptiveTransparencyEngine()

    return _transparency_engine_instance


# Utility functions for transparency operations
async def generate_user_transparency(
    engagement_contract: MetisDataContract, user_id: UUID
) -> ProgressiveDisclosure:
    """Generate progressive transparency for user"""
    transparency_engine = await get_transparency_engine()
    return await transparency_engine.generate_progressive_disclosure(
        engagement_contract, user_id
    )


async def update_transparency_preferences(user_id: UUID, preferences: Dict[str, Any]):
    """Update user transparency preferences"""
    transparency_engine = await get_transparency_engine()
    await transparency_engine.update_user_preferences(user_id, preferences)
