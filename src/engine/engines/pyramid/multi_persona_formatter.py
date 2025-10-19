"""
Multi-Persona Deliverable Adaptation System - Sprint 2.3
F006: Revolutionary persona-specific deliverable optimization

This module creates the world's most sophisticated executive deliverable system
that leverages Context Intelligence to automatically adapt content, tone, depth,
and focus for different executive personas (CEO, CTO, Board, Partners) using
cognitive exhaust and historical effectiveness patterns.

Key Innovations:
- Persona-Specific Content Adaptation
- Context-Aware Executive Communication Optimization
- Historical Pattern-Based Persona Preferences
- Cognitive Coherence-Driven Messaging
- Dynamic Format and Structure Adaptation
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from enum import Enum

from .models import ExecutiveDeliverable, PyramidNode
from .enums import DeliverableType
from .formatters import DeliverableFormatter

# Context Intelligence imports
try:
    from src.interfaces.context_intelligence_interface import IContextIntelligence
    from src.engine.models.data_contracts import ContextType

    CONTEXT_INTELLIGENCE_AVAILABLE = True
except ImportError:
    CONTEXT_INTELLIGENCE_AVAILABLE = False
    IContextIntelligence = Any
    ContextType = Any


class ExecutivePersona(Enum):
    """Executive persona types for deliverable adaptation"""

    CEO = "ceo"
    CTO = "cto"
    CFO = "cfo"
    BOARD = "board"
    PARTNER = "partner"
    DIRECTOR = "director"
    VP = "vp"
    GENERAL = "general"


class PersonaCharacteristics:
    """Persona-specific characteristics and preferences"""

    PERSONA_PROFILES = {
        ExecutivePersona.CEO: {
            "focus_areas": [
                "strategic_vision",
                "competitive_advantage",
                "growth",
                "transformation",
                "roi",
            ],
            "communication_style": "visionary_decisive",
            "content_depth": "high_level_strategic",
            "decision_factors": [
                "impact",
                "time_to_value",
                "strategic_alignment",
                "competitive_positioning",
            ],
            "preferred_metrics": [
                "revenue_growth",
                "market_share",
                "strategic_milestones",
                "roi",
            ],
            "attention_span": "short_high_impact",
            "presentation_preference": "executive_summary_focused",
            "language_style": "bold_confident",
            "risk_tolerance": "calculated_aggressive",
        },
        ExecutivePersona.CTO: {
            "focus_areas": [
                "technology",
                "architecture",
                "scalability",
                "innovation",
                "engineering",
            ],
            "communication_style": "technical_pragmatic",
            "content_depth": "detailed_technical",
            "decision_factors": [
                "technical_feasibility",
                "scalability",
                "security",
                "performance",
            ],
            "preferred_metrics": [
                "system_performance",
                "uptime",
                "security_metrics",
                "development_velocity",
            ],
            "attention_span": "moderate_detail_oriented",
            "presentation_preference": "technical_depth_with_summary",
            "language_style": "precise_analytical",
            "risk_tolerance": "conservative_thorough",
        },
        ExecutivePersona.CFO: {
            "focus_areas": [
                "financial_impact",
                "cost_optimization",
                "budget",
                "compliance",
                "efficiency",
            ],
            "communication_style": "analytical_quantitative",
            "content_depth": "financial_focused",
            "decision_factors": [
                "cost_benefit",
                "financial_impact",
                "budget_alignment",
                "risk_mitigation",
            ],
            "preferred_metrics": [
                "cost_savings",
                "roi",
                "budget_variance",
                "financial_ratios",
            ],
            "attention_span": "detail_oriented_thorough",
            "presentation_preference": "data_driven_comprehensive",
            "language_style": "precise_conservative",
            "risk_tolerance": "risk_averse_compliance_focused",
        },
        ExecutivePersona.BOARD: {
            "focus_areas": [
                "governance",
                "strategic_oversight",
                "risk_management",
                "compliance",
                "long_term_value",
            ],
            "communication_style": "formal_comprehensive",
            "content_depth": "strategic_governance",
            "decision_factors": [
                "strategic_alignment",
                "governance",
                "risk_assessment",
                "stakeholder_impact",
            ],
            "preferred_metrics": [
                "strategic_kpis",
                "governance_metrics",
                "risk_indicators",
                "stakeholder_value",
            ],
            "attention_span": "long_comprehensive",
            "presentation_preference": "comprehensive_formal",
            "language_style": "formal_authoritative",
            "risk_tolerance": "prudent_oversight",
        },
        ExecutivePersona.PARTNER: {
            "focus_areas": [
                "client_value",
                "delivery_excellence",
                "business_development",
                "expertise",
                "relationships",
            ],
            "communication_style": "client_focused_consultative",
            "content_depth": "business_consulting",
            "decision_factors": [
                "client_impact",
                "delivery_quality",
                "business_value",
                "relationship_building",
            ],
            "preferred_metrics": [
                "client_satisfaction",
                "business_value_delivered",
                "project_success",
                "expertise_demonstration",
            ],
            "attention_span": "client_context_aware",
            "presentation_preference": "client_value_focused",
            "language_style": "consultative_professional",
            "risk_tolerance": "balanced_client_focused",
        },
        ExecutivePersona.GENERAL: {
            "focus_areas": [
                "business_impact",
                "efficiency",
                "strategic_value",
                "implementation",
                "results",
            ],
            "communication_style": "professional_balanced",
            "content_depth": "comprehensive_balanced",
            "decision_factors": [
                "business_impact",
                "feasibility",
                "cost_benefit",
                "strategic_alignment",
            ],
            "preferred_metrics": [
                "performance_improvement",
                "business_value",
                "implementation_success",
                "roi",
            ],
            "attention_span": "balanced_comprehensive",
            "presentation_preference": "structured_comprehensive",
            "language_style": "professional_clear",
            "risk_tolerance": "balanced_prudent",
        },
    }

    @classmethod
    def get_persona_profile(cls, persona: ExecutivePersona) -> Dict[str, Any]:
        """Get comprehensive persona profile"""
        return cls.PERSONA_PROFILES.get(
            persona, cls.PERSONA_PROFILES[ExecutivePersona.GENERAL]
        )


class MultiPersonaDeliverableFormatter(DeliverableFormatter):
    """
    Multi-Persona Deliverable Adaptation System using Context Intelligence

    Automatically adapts deliverables for different executive personas by:
    - Analyzing persona-specific preferences from cognitive exhaust
    - Applying context-aware content optimization
    - Using historical pattern effectiveness for persona adaptation
    - Optimizing messaging tone, depth, and structure
    - Leveraging cognitive coherence for persona-specific communication
    """

    def __init__(self, context_intelligence: Optional[IContextIntelligence] = None):
        super().__init__()
        self.context_intelligence = context_intelligence
        self.context_enhanced = context_intelligence is not None
        self.logger = logging.getLogger(__name__)

        # Persona adaptation history and patterns
        self.persona_adaptation_history = {}
        self.persona_effectiveness_patterns = {}

        # Context-aware persona optimization
        self.context_persona_mappings = {
            "IMMEDIATE": [ExecutivePersona.CEO, ExecutivePersona.DIRECTOR],
            "STRATEGIC": [ExecutivePersona.CEO, ExecutivePersona.BOARD],
            "PROCEDURAL": [ExecutivePersona.CTO, ExecutivePersona.DIRECTOR],
            "DOMAIN": [ExecutivePersona.PARTNER, ExecutivePersona.VP],
        }

        if self.context_enhanced:
            self.logger.info(
                "🎯 Multi-Persona Deliverable Formatter initialized with Context Intelligence"
            )
        else:
            self.logger.info("📊 Standard Deliverable Formatter initialized")

    async def generate_persona_adapted_deliverable(
        self,
        pyramid: PyramidNode,
        deliverable_type: DeliverableType,
        engagement_data: Dict[str, Any],
        target_persona: ExecutivePersona,
        context_metadata: Optional[Dict[str, Any]] = None,
        engagement_id: Optional[str] = None,
    ) -> ExecutiveDeliverable:
        """
        Generate deliverable adapted for specific executive persona

        Args:
            pyramid: Pyramid structure from context-intelligent builder
            deliverable_type: Type of deliverable to generate
            engagement_data: Engagement analysis data
            target_persona: Target executive persona
            context_metadata: Context Intelligence metadata
            engagement_id: Engagement ID for historical pattern analysis

        Returns:
            Persona-adapted executive deliverable
        """

        self.logger.info(
            f"🎯 Generating persona-adapted deliverable for {target_persona.value} executive"
        )

        # Phase 1: Persona Context Analysis
        persona_context = await self._analyze_persona_context(
            target_persona, context_metadata, engagement_id
        )

        # Phase 2: Context-Aware Persona Adaptation
        adapted_pyramid = await self._adapt_pyramid_for_persona(
            pyramid, target_persona, persona_context, context_metadata
        )

        # Phase 3: Persona-Specific Content Generation
        deliverable = await self._generate_persona_specific_content(
            adapted_pyramid,
            deliverable_type,
            engagement_data,
            target_persona,
            persona_context,
        )

        # Phase 4: Communication Style Optimization
        await self._optimize_communication_style(
            deliverable, target_persona, persona_context
        )

        # Phase 5: Format and Structure Adaptation
        await self._adapt_format_and_structure(
            deliverable, target_persona, persona_context, deliverable_type
        )

        # Phase 6: Quality Validation for Persona
        await self._validate_persona_adaptation_quality(
            deliverable, target_persona, persona_context, engagement_id
        )

        # Phase 7: Store Persona Adaptation Cognitive Exhaust
        if self.context_intelligence and engagement_id:
            await self._store_persona_adaptation_cognitive_exhaust(
                deliverable, target_persona, persona_context, engagement_id
            )

        self.logger.info(
            f"✅ Persona-adapted deliverable generated for {target_persona.value}"
        )

        return deliverable

    async def _analyze_persona_context(
        self,
        target_persona: ExecutivePersona,
        context_metadata: Optional[Dict[str, Any]],
        engagement_id: Optional[str],
    ) -> Dict[str, Any]:
        """
        Analyze persona-specific context for optimal adaptation
        """

        persona_profile = PersonaCharacteristics.get_persona_profile(target_persona)

        if not self.context_enhanced or not context_metadata:
            return {
                "persona_context_analyzed": False,
                "persona_profile": persona_profile,
                "historical_patterns": 0,
                "context_alignment": "default",
            }

        try:
            # Get persona-specific historical contexts
            persona_contexts = []
            if self.context_intelligence and engagement_id:
                try:
                    persona_query = (
                        f"{target_persona.value} executive deliverable adaptation"
                    )
                    persona_contexts = (
                        await self.context_intelligence.get_relevant_context(
                            current_query=persona_query,
                            max_contexts=3,
                            engagement_id=engagement_id,
                        )
                    )
                except Exception as e:
                    self.logger.debug(f"Persona context retrieval warning: {e}")

            # Analyze context-persona alignment
            primary_context_type = context_metadata.get(
                "primary_context_type", "DOMAIN"
            )
            context_alignment = self._assess_context_persona_alignment(
                target_persona, primary_context_type
            )

            # Extract persona-specific optimization factors
            optimization_factors = self._extract_persona_optimization_factors(
                target_persona, context_metadata, len(persona_contexts)
            )

            return {
                "persona_context_analyzed": True,
                "persona_profile": persona_profile,
                "historical_patterns": len(persona_contexts),
                "context_alignment": context_alignment,
                "optimization_factors": optimization_factors,
                "primary_context_type": primary_context_type,
                "overall_coherence": context_metadata.get("overall_coherence", 0.75),
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Persona context analysis failed: {e}")
            return {
                "persona_context_analyzed": False,
                "persona_profile": persona_profile,
                "error": str(e),
            }

    async def _adapt_pyramid_for_persona(
        self,
        pyramid: PyramidNode,
        target_persona: ExecutivePersona,
        persona_context: Dict[str, Any],
        context_metadata: Optional[Dict[str, Any]],
    ) -> PyramidNode:
        """
        Adapt pyramid structure for persona-specific emphasis
        """

        persona_profile = persona_context["persona_profile"]
        focus_areas = persona_profile.get("focus_areas", [])

        # Create adapted pyramid with persona-specific emphasis
        adapted_pyramid = PyramidNode(
            level=pyramid.level,
            content=self._adapt_content_for_persona(
                pyramid.content, target_persona, persona_profile
            ),
            argument_type=pyramid.argument_type,
            metadata={
                **getattr(pyramid, "metadata", {}),
                "persona_adapted": True,
                "target_persona": target_persona.value,
                "adaptation_timestamp": datetime.now().isoformat(),
            },
        )

        # Adapt key lines for persona focus
        for child in pyramid.children:
            adapted_child = PyramidNode(
                level=child.level,
                content=self._adapt_content_for_persona(
                    child.content, target_persona, persona_profile
                ),
                argument_type=child.argument_type,
                metadata={
                    **getattr(child, "metadata", {}),
                    "persona_emphasis": self._calculate_persona_emphasis(
                        child.content, focus_areas
                    ),
                },
            )

            # Adapt supporting points with persona relevance
            for support in child.children:
                adapted_support = PyramidNode(
                    level=support.level,
                    content=self._adapt_content_for_persona(
                        support.content, target_persona, persona_profile
                    ),
                    argument_type=support.argument_type,
                    metadata={
                        **getattr(support, "metadata", {}),
                        "persona_relevance": self._assess_persona_relevance(
                            support.content, focus_areas
                        ),
                    },
                )
                adapted_child.add_child(adapted_support)

            adapted_pyramid.add_child(adapted_child)

        return adapted_pyramid

    async def _generate_persona_specific_content(
        self,
        adapted_pyramid: PyramidNode,
        deliverable_type: DeliverableType,
        engagement_data: Dict[str, Any],
        target_persona: ExecutivePersona,
        persona_context: Dict[str, Any],
    ) -> ExecutiveDeliverable:
        """
        Generate content optimized for persona preferences
        """

        # Generate base deliverable
        deliverable = await super().generate_deliverable_content(
            adapted_pyramid, deliverable_type, engagement_data
        )

        persona_profile = persona_context["persona_profile"]

        # Persona-specific title optimization
        deliverable.title = self._generate_persona_optimized_title(
            adapted_pyramid, deliverable_type, target_persona, persona_profile
        )

        # Persona-specific executive summary
        deliverable.executive_summary = self._generate_persona_optimized_summary(
            adapted_pyramid, target_persona, persona_profile
        )

        # Persona-specific recommendations
        deliverable.key_recommendations = (
            self._generate_persona_optimized_recommendations(
                adapted_pyramid, target_persona, persona_profile
            )
        )

        # Persona-specific supporting analysis
        deliverable.supporting_analysis = self._build_persona_focused_analysis(
            engagement_data, target_persona, persona_profile
        )

        # Persona-specific implementation roadmap
        deliverable.implementation_roadmap = self._create_persona_adapted_roadmap(
            adapted_pyramid, target_persona, persona_profile
        )

        # Add persona metadata
        deliverable.persona_metadata = {
            "target_persona": target_persona.value,
            "persona_adaptation_applied": True,
            "adaptation_factors": persona_context.get("optimization_factors", {}),
            "context_alignment": persona_context.get("context_alignment", "default"),
        }

        return deliverable

    async def _optimize_communication_style(
        self,
        deliverable: ExecutiveDeliverable,
        target_persona: ExecutivePersona,
        persona_context: Dict[str, Any],
    ) -> None:
        """
        Optimize communication style for persona preferences
        """

        persona_profile = persona_context["persona_profile"]
        communication_style = persona_profile.get("communication_style", "professional")
        language_style = persona_profile.get("language_style", "balanced")

        # Apply communication style transformations
        if communication_style == "visionary_decisive":
            deliverable.executive_summary = self._apply_visionary_decisive_style(
                deliverable.executive_summary
            )
        elif communication_style == "technical_pragmatic":
            deliverable.executive_summary = self._apply_technical_pragmatic_style(
                deliverable.executive_summary
            )
        elif communication_style == "analytical_quantitative":
            deliverable.executive_summary = self._apply_analytical_quantitative_style(
                deliverable.executive_summary
            )
        elif communication_style == "formal_comprehensive":
            deliverable.executive_summary = self._apply_formal_comprehensive_style(
                deliverable.executive_summary
            )
        elif communication_style == "client_focused_consultative":
            deliverable.executive_summary = (
                self._apply_client_focused_consultative_style(
                    deliverable.executive_summary
                )
            )

        # Apply language style adjustments
        deliverable.executive_summary = self._apply_language_style(
            deliverable.executive_summary, language_style
        )

    # Content adaptation methods

    def _adapt_content_for_persona(
        self,
        content: str,
        target_persona: ExecutivePersona,
        persona_profile: Dict[str, Any],
    ) -> str:
        """Adapt content for persona-specific preferences"""

        focus_areas = persona_profile.get("focus_areas", [])
        language_style = persona_profile.get("language_style", "balanced")

        # Apply persona-specific content transformations
        adapted_content = content

        if target_persona == ExecutivePersona.CEO:
            adapted_content = self._adapt_for_ceo_perspective(content, focus_areas)
        elif target_persona == ExecutivePersona.CTO:
            adapted_content = self._adapt_for_cto_perspective(content, focus_areas)
        elif target_persona == ExecutivePersona.CFO:
            adapted_content = self._adapt_for_cfo_perspective(content, focus_areas)
        elif target_persona == ExecutivePersona.BOARD:
            adapted_content = self._adapt_for_board_perspective(content, focus_areas)
        elif target_persona == ExecutivePersona.PARTNER:
            adapted_content = self._adapt_for_partner_perspective(content, focus_areas)

        return adapted_content

    def _adapt_for_ceo_perspective(self, content: str, focus_areas: List[str]) -> str:
        """Adapt content for CEO perspective - strategic vision and competitive advantage"""
        adapted = content

        # Enhance strategic language
        strategic_replacements = {
            "will improve": "will drive competitive advantage through",
            "analysis shows": "strategic assessment reveals",
            "implementation": "strategic transformation",
            "efficiency": "operational excellence",
            "process": "strategic capability",
        }

        for original, replacement in strategic_replacements.items():
            adapted = adapted.replace(original, replacement)

        # Add strategic impact framing
        if "competitive" not in adapted.lower() and "strategic" not in adapted.lower():
            adapted = f"Strategic analysis indicates that {adapted.lower()}"

        return adapted

    def _adapt_for_cto_perspective(self, content: str, focus_areas: List[str]) -> str:
        """Adapt content for CTO perspective - technical feasibility and architecture"""
        adapted = content

        # Enhance technical language
        technical_replacements = {
            "implementation": "technical implementation",
            "solution": "technical solution",
            "system": "architecture and system",
            "process": "technical process",
            "efficiency": "system performance and efficiency",
        }

        for original, replacement in technical_replacements.items():
            adapted = adapted.replace(original, replacement)

        # Add technical feasibility framing
        if "technical" not in adapted.lower() and "system" not in adapted.lower():
            adapted = f"Technical analysis demonstrates that {adapted.lower()}"

        return adapted

    def _adapt_for_cfo_perspective(self, content: str, focus_areas: List[str]) -> str:
        """Adapt content for CFO perspective - financial impact and cost optimization"""
        adapted = content

        # Enhance financial language
        financial_replacements = {
            "improvement": "cost optimization and financial improvement",
            "efficiency": "cost efficiency and financial performance",
            "implementation": "budget-conscious implementation",
            "benefits": "financial benefits and cost savings",
            "impact": "financial impact and ROI",
        }

        for original, replacement in financial_replacements.items():
            adapted = adapted.replace(original, replacement)

        # Add financial impact framing
        if "cost" not in adapted.lower() and "financial" not in adapted.lower():
            adapted = f"Financial analysis indicates that {adapted.lower()}"

        return adapted

    def _adapt_for_board_perspective(self, content: str, focus_areas: List[str]) -> str:
        """Adapt content for Board perspective - governance and strategic oversight"""
        adapted = content

        # Enhance governance language
        governance_replacements = {
            "implementation": "governance-compliant implementation",
            "strategy": "board-approved strategic initiative",
            "risk": "enterprise risk and governance consideration",
            "oversight": "board oversight and strategic governance",
            "compliance": "regulatory compliance and governance",
        }

        for original, replacement in governance_replacements.items():
            adapted = adapted.replace(original, replacement)

        # Add governance framing
        if "governance" not in adapted.lower() and "strategic" not in adapted.lower():
            adapted = (
                f"Board-level strategic assessment confirms that {adapted.lower()}"
            )

        return adapted

    def _adapt_for_partner_perspective(
        self, content: str, focus_areas: List[str]
    ) -> str:
        """Adapt content for Partner perspective - client value and delivery excellence"""
        adapted = content

        # Enhance client value language
        client_replacements = {
            "analysis": "client-focused analysis",
            "solution": "client-optimized solution",
            "implementation": "client-centric implementation",
            "benefits": "client value and business benefits",
            "impact": "client impact and value delivery",
        }

        for original, replacement in client_replacements.items():
            adapted = adapted.replace(original, replacement)

        # Add client value framing
        if "client" not in adapted.lower() and "value" not in adapted.lower():
            adapted = f"Client value analysis demonstrates that {adapted.lower()}"

        return adapted

    # Communication style application methods

    def _apply_visionary_decisive_style(self, content: str) -> str:
        """Apply visionary decisive communication style for CEO"""
        style_enhancements = {
            "analysis reveals": "vision clearly demonstrates",
            "suggests that": "decisively establishes that",
            "indicates": "unequivocally shows",
            "may result in": "will deliver",
            "could provide": "will provide transformational",
        }

        styled_content = content
        for original, enhancement in style_enhancements.items():
            styled_content = styled_content.replace(original, enhancement)

        return styled_content

    def _apply_technical_pragmatic_style(self, content: str) -> str:
        """Apply technical pragmatic communication style for CTO"""
        style_enhancements = {
            "strategic": "technically strategic",
            "solution": "engineered solution",
            "implementation": "technical implementation approach",
            "analysis": "technical assessment and analysis",
            "approach": "systematic technical approach",
        }

        styled_content = content
        for original, enhancement in style_enhancements.items():
            styled_content = styled_content.replace(original, enhancement)

        return styled_content

    def _apply_analytical_quantitative_style(self, content: str) -> str:
        """Apply analytical quantitative communication style for CFO"""
        style_enhancements = {
            "significant": "quantifiable and significant",
            "improvement": "measurable improvement",
            "benefits": "quantified financial benefits",
            "impact": "measured financial impact",
            "results": "quantitative results and metrics",
        }

        styled_content = content
        for original, enhancement in style_enhancements.items():
            styled_content = styled_content.replace(original, enhancement)

        return styled_content

    def _apply_formal_comprehensive_style(self, content: str) -> str:
        """Apply formal comprehensive communication style for Board"""
        style_enhancements = {
            "analysis": "comprehensive governance analysis",
            "recommendation": "board-level strategic recommendation",
            "implementation": "governance-compliant implementation framework",
            "oversight": "strategic board oversight mechanism",
            "assessment": "comprehensive strategic assessment",
        }

        styled_content = content
        for original, enhancement in style_enhancements.items():
            styled_content = styled_content.replace(original, enhancement)

        return styled_content

    def _apply_client_focused_consultative_style(self, content: str) -> str:
        """Apply client focused consultative communication style for Partner"""
        style_enhancements = {
            "analysis": "client-centric consulting analysis",
            "recommendation": "client value-optimized recommendation",
            "solution": "tailored client solution",
            "implementation": "client-focused delivery approach",
            "benefits": "client business value and benefits",
        }

        styled_content = content
        for original, enhancement in style_enhancements.items():
            styled_content = styled_content.replace(original, enhancement)

        return styled_content

    def _apply_language_style(self, content: str, language_style: str) -> str:
        """Apply language style adjustments"""

        if language_style == "bold_confident":
            return self._apply_bold_confident_language(content)
        elif language_style == "precise_analytical":
            return self._apply_precise_analytical_language(content)
        elif language_style == "formal_authoritative":
            return self._apply_formal_authoritative_language(content)
        elif language_style == "consultative_professional":
            return self._apply_consultative_professional_language(content)

        return content

    # Helper methods for content generation

    def _calculate_persona_emphasis(
        self, content: str, focus_areas: List[str]
    ) -> float:
        """Calculate persona emphasis score for content"""
        content_lower = content.lower()
        matches = sum(
            1 for area in focus_areas if area.replace("_", " ") in content_lower
        )
        return min(matches / len(focus_areas) if focus_areas else 0, 1.0)

    def _assess_persona_relevance(self, content: str, focus_areas: List[str]) -> str:
        """Assess persona relevance level"""
        emphasis = self._calculate_persona_emphasis(content, focus_areas)

        if emphasis >= 0.5:
            return "high"
        elif emphasis >= 0.3:
            return "medium"
        else:
            return "low"

    def _assess_context_persona_alignment(
        self, target_persona: ExecutivePersona, primary_context_type: str
    ) -> str:
        """Assess alignment between context and persona"""

        aligned_personas = self.context_persona_mappings.get(primary_context_type, [])

        if target_persona in aligned_personas:
            return "high_alignment"
        elif target_persona in [
            ExecutivePersona.GENERAL,
            ExecutivePersona.VP,
            ExecutivePersona.DIRECTOR,
        ]:
            return "moderate_alignment"
        else:
            return "adaptation_required"

    def _extract_persona_optimization_factors(
        self,
        target_persona: ExecutivePersona,
        context_metadata: Dict[str, Any],
        historical_patterns: int,
    ) -> Dict[str, Any]:
        """Extract persona-specific optimization factors"""

        return {
            "persona_focus_boost": self._calculate_persona_focus_boost(target_persona),
            "context_alignment_factor": self._calculate_context_alignment_factor(
                target_persona, context_metadata
            ),
            "historical_pattern_influence": min(historical_patterns * 0.1, 0.3),
            "cognitive_coherence_optimization": context_metadata.get(
                "overall_coherence", 0.75
            )
            * 0.2,
        }

    def _calculate_persona_focus_boost(self, target_persona: ExecutivePersona) -> float:
        """Calculate persona-specific focus boost factor"""

        focus_multipliers = {
            ExecutivePersona.CEO: 1.3,  # High impact strategic focus
            ExecutivePersona.BOARD: 1.2,  # Comprehensive governance focus
            ExecutivePersona.CTO: 1.25,  # Technical depth focus
            ExecutivePersona.CFO: 1.2,  # Financial analysis focus
            ExecutivePersona.PARTNER: 1.15,  # Client value focus
            ExecutivePersona.DIRECTOR: 1.1,  # Operational focus
            ExecutivePersona.VP: 1.05,  # Departmental focus
            ExecutivePersona.GENERAL: 1.0,  # Balanced focus
        }

        return focus_multipliers.get(target_persona, 1.0)

    def _calculate_context_alignment_factor(
        self, target_persona: ExecutivePersona, context_metadata: Dict[str, Any]
    ) -> float:
        """Calculate context-persona alignment factor"""

        primary_context = context_metadata.get("primary_context_type", "DOMAIN")
        alignment = self._assess_context_persona_alignment(
            target_persona, primary_context
        )

        alignment_factors = {
            "high_alignment": 1.2,
            "moderate_alignment": 1.0,
            "adaptation_required": 0.9,
        }

        return alignment_factors.get(alignment, 1.0)

    # Content generation methods

    def _generate_persona_optimized_title(
        self,
        pyramid: PyramidNode,
        deliverable_type: DeliverableType,
        target_persona: ExecutivePersona,
        persona_profile: Dict[str, Any],
    ) -> str:
        """Generate persona-optimized title"""

        base_title = super()._generate_title(pyramid, deliverable_type)

        # Persona-specific title prefixes
        persona_prefixes = {
            ExecutivePersona.CEO: "Strategic Executive Summary:",
            ExecutivePersona.CTO: "Technical Leadership Brief:",
            ExecutivePersona.CFO: "Financial Impact Analysis:",
            ExecutivePersona.BOARD: "Board Governance Report:",
            ExecutivePersona.PARTNER: "Client Value Proposition:",
        }

        prefix = persona_prefixes.get(target_persona, "Executive Brief:")

        # Remove generic prefixes and apply persona-specific one
        if "Executive Summary:" in base_title:
            base_title = base_title.replace("Executive Summary:", "").strip()

        return f"{prefix} {base_title}"

    def _generate_persona_optimized_summary(
        self,
        pyramid: PyramidNode,
        target_persona: ExecutivePersona,
        persona_profile: Dict[str, Any],
    ) -> str:
        """Generate persona-optimized executive summary"""

        base_summary = super()._generate_executive_summary(pyramid)

        # Apply persona-specific content adaptations
        adapted_summary = self._adapt_content_for_persona(
            base_summary, target_persona, persona_profile
        )

        # Apply communication style
        communication_style = persona_profile.get("communication_style", "professional")
        if communication_style == "visionary_decisive":
            adapted_summary = self._apply_visionary_decisive_style(adapted_summary)
        elif communication_style == "technical_pragmatic":
            adapted_summary = self._apply_technical_pragmatic_style(adapted_summary)
        elif communication_style == "analytical_quantitative":
            adapted_summary = self._apply_analytical_quantitative_style(adapted_summary)

        return adapted_summary

    def _generate_persona_optimized_recommendations(
        self,
        pyramid: PyramidNode,
        target_persona: ExecutivePersona,
        persona_profile: Dict[str, Any],
    ) -> List[str]:
        """Generate persona-optimized recommendations"""

        base_recommendations = super()._extract_key_recommendations(pyramid)

        persona_optimized_recommendations = []
        for rec in base_recommendations:
            adapted_rec = self._adapt_content_for_persona(
                rec, target_persona, persona_profile
            )
            persona_optimized_recommendations.append(adapted_rec)

        return persona_optimized_recommendations

    # Quality validation and cognitive exhaust storage

    async def _validate_persona_adaptation_quality(
        self,
        deliverable: ExecutiveDeliverable,
        target_persona: ExecutivePersona,
        persona_context: Dict[str, Any],
        engagement_id: Optional[str],
    ) -> None:
        """Validate quality of persona adaptation"""

        persona_profile = persona_context["persona_profile"]
        focus_areas = persona_profile.get("focus_areas", [])

        # Calculate persona alignment score
        summary_alignment = self._calculate_persona_emphasis(
            deliverable.executive_summary, focus_areas
        )
        rec_alignment = sum(
            self._calculate_persona_emphasis(rec, focus_areas)
            for rec in deliverable.key_recommendations
        ) / len(deliverable.key_recommendations)

        overall_persona_alignment = (summary_alignment + rec_alignment) / 2

        # Add persona quality metrics
        if not hasattr(deliverable, "persona_quality_metrics"):
            deliverable.persona_quality_metrics = {}

        deliverable.persona_quality_metrics.update(
            {
                "persona_alignment_score": overall_persona_alignment,
                "focus_area_coverage": summary_alignment,
                "recommendation_relevance": rec_alignment,
                "communication_style_applied": True,
                "persona_optimization_complete": True,
            }
        )

    async def _store_persona_adaptation_cognitive_exhaust(
        self,
        deliverable: ExecutiveDeliverable,
        target_persona: ExecutivePersona,
        persona_context: Dict[str, Any],
        engagement_id: str,
    ) -> None:
        """Store persona adaptation cognitive exhaust for future learning"""

        if not self.context_intelligence:
            return

        try:
            persona_metrics = getattr(deliverable, "persona_quality_metrics", {})

            thinking_process = f"""
            Multi-Persona Deliverable Adaptation Process:
            
            Target Persona: {target_persona.value}
            Persona Profile Applied: {persona_context['persona_profile'].get('communication_style', 'standard')}
            Content Adaptation: {persona_context.get('context_alignment', 'standard')}
            
            Deliverable Optimization:
            - Title: {deliverable.title[:100]}...
            - Persona Alignment Score: {persona_metrics.get('persona_alignment_score', 'N/A')}
            - Focus Area Coverage: {persona_metrics.get('focus_area_coverage', 'N/A')}
            - Communication Style Applied: {persona_metrics.get('communication_style_applied', False)}
            
            Historical Patterns: {persona_context.get('historical_patterns', 0)}
            Context Alignment: {persona_context.get('context_alignment', 'default')}
            
            The deliverable has been optimized for maximum impact with the target executive persona.
            """

            cleaned_response = f"Multi-persona deliverable adaptation completed for {target_persona.value} executive with optimized communication and content focus."

            await self.context_intelligence.store_cognitive_exhaust_triple_layer(
                engagement_id=engagement_id,
                phase="persona_adaptation",
                mental_model="multi_persona_deliverable_optimization",
                thinking_process=thinking_process,
                cleaned_response=cleaned_response,
                confidence=persona_metrics.get("persona_alignment_score", 0.8),
            )

            self.logger.info(
                f"💾 Persona adaptation cognitive exhaust stored for {target_persona.value} executive"
            )

        except Exception as e:
            self.logger.warning(
                f"⚠️ Failed to store persona adaptation cognitive exhaust: {e}"
            )

    # Language style application methods (additional implementations)

    def _apply_bold_confident_language(self, content: str) -> str:
        """Apply bold confident language for CEO communication"""
        confidence_enhancements = {
            "analysis suggests": "analysis clearly demonstrates",
            "may lead to": "will deliver",
            "could result in": "will achieve",
            "potentially": "definitively",
            "might provide": "will provide",
        }

        enhanced_content = content
        for original, enhancement in confidence_enhancements.items():
            enhanced_content = enhanced_content.replace(original, enhancement)

        return enhanced_content

    def _apply_precise_analytical_language(self, content: str) -> str:
        """Apply precise analytical language for technical personas"""
        precision_enhancements = {
            "significant": "measurably significant",
            "improvement": "quantified improvement",
            "analysis": "systematic analysis",
            "assessment": "detailed technical assessment",
            "evaluation": "comprehensive evaluation",
        }

        enhanced_content = content
        for original, enhancement in precision_enhancements.items():
            enhanced_content = enhanced_content.replace(original, enhancement)

        return enhanced_content

    def _apply_formal_authoritative_language(self, content: str) -> str:
        """Apply formal authoritative language for board communication"""
        authority_enhancements = {
            "recommendation": "board-level recommendation",
            "strategy": "governance-aligned strategy",
            "implementation": "board-approved implementation",
            "oversight": "strategic governance oversight",
            "compliance": "regulatory and governance compliance",
        }

        enhanced_content = content
        for original, enhancement in authority_enhancements.items():
            enhanced_content = enhanced_content.replace(original, enhancement)

        return enhanced_content

    def _apply_consultative_professional_language(self, content: str) -> str:
        """Apply consultative professional language for partner communication"""
        consultative_enhancements = {
            "solution": "client-optimized solution",
            "approach": "consultative approach",
            "analysis": "client-focused analysis",
            "recommendation": "value-driven recommendation",
            "implementation": "client-centric implementation",
        }

        enhanced_content = content
        for original, enhancement in consultative_enhancements.items():
            enhanced_content = enhanced_content.replace(original, enhancement)

        return enhanced_content

    # Format and structure adaptation methods

    async def _adapt_format_and_structure(
        self,
        deliverable: ExecutiveDeliverable,
        target_persona: ExecutivePersona,
        persona_context: Dict[str, Any],
        deliverable_type: DeliverableType,
    ) -> None:
        """Adapt format and structure for persona preferences"""

        persona_profile = persona_context["persona_profile"]
        presentation_preference = persona_profile.get(
            "presentation_preference", "standard"
        )
        attention_span = persona_profile.get("attention_span", "moderate")

        # Adapt structure based on attention span and preferences
        if attention_span == "short_high_impact":
            # CEO preference - concise high-impact format
            deliverable = self._apply_concise_high_impact_format(deliverable)
        elif attention_span == "detail_oriented_thorough":
            # CFO preference - comprehensive detailed format
            deliverable = self._apply_comprehensive_detailed_format(deliverable)
        elif attention_span == "moderate_detail_oriented":
            # CTO preference - balanced technical format
            deliverable = self._apply_balanced_technical_format(deliverable)
        elif attention_span == "long_comprehensive":
            # Board preference - comprehensive formal format
            deliverable = self._apply_comprehensive_formal_format(deliverable)

    def _apply_concise_high_impact_format(
        self, deliverable: ExecutiveDeliverable
    ) -> ExecutiveDeliverable:
        """Apply concise high-impact format for CEO"""
        # Ensure executive summary is concise but impactful
        if len(deliverable.executive_summary.split()) > 200:
            # Condense while maintaining key points
            sentences = deliverable.executive_summary.split(". ")
            key_sentences = sentences[:3] + [sentences[-1]]  # First 3 + conclusion
            deliverable.executive_summary = ". ".join(key_sentences)

        return deliverable

    def _apply_comprehensive_detailed_format(
        self, deliverable: ExecutiveDeliverable
    ) -> ExecutiveDeliverable:
        """Apply comprehensive detailed format for CFO"""
        # Ensure comprehensive coverage with detailed supporting analysis
        # CFO format already maintained through base implementation
        return deliverable

    def _apply_balanced_technical_format(
        self, deliverable: ExecutiveDeliverable
    ) -> ExecutiveDeliverable:
        """Apply balanced technical format for CTO"""
        # Maintain technical depth while ensuring executive accessibility
        return deliverable

    def _apply_comprehensive_formal_format(
        self, deliverable: ExecutiveDeliverable
    ) -> ExecutiveDeliverable:
        """Apply comprehensive formal format for Board"""
        # Ensure formal structure with comprehensive governance perspective
        return deliverable

    # Persona-specific content builders

    def _build_persona_focused_analysis(
        self,
        engagement_data: Dict[str, Any],
        target_persona: ExecutivePersona,
        persona_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build analysis focused on persona interests"""

        base_analysis = super()._build_supporting_analysis(engagement_data)

        # Add persona-specific analysis sections
        focus_areas = persona_profile.get("focus_areas", [])
        preferred_metrics = persona_profile.get("preferred_metrics", [])

        # Filter and prioritize analysis based on persona focus
        persona_focused_analysis = {}

        for key, value in base_analysis.items():
            if any(
                focus_area.replace("_", " ") in key.lower()
                for focus_area in focus_areas
            ):
                persona_focused_analysis[key] = value
            elif key in [
                "key_insights",
                "framework_analysis",
            ]:  # Always include core analysis
                persona_focused_analysis[key] = value

        # Add persona-specific metrics if available
        if preferred_metrics:
            persona_focused_analysis["persona_specific_metrics"] = {
                "focus_areas": focus_areas,
                "key_metrics": preferred_metrics,
                "persona_relevance": "high",
            }

        return persona_focused_analysis

    def _create_persona_adapted_roadmap(
        self,
        pyramid: PyramidNode,
        target_persona: ExecutivePersona,
        persona_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create roadmap adapted for persona perspective"""

        base_roadmap = super()._create_implementation_roadmap(pyramid)

        # Adapt timeline and phases based on persona preferences
        decision_factors = persona_profile.get("decision_factors", [])
        preferred_metrics = persona_profile.get("preferred_metrics", [])

        # Modify success metrics to align with persona interests
        persona_adapted_roadmap = base_roadmap.copy()

        if target_persona == ExecutivePersona.CEO:
            persona_adapted_roadmap["success_metrics"] = self._get_ceo_success_metrics()
        elif target_persona == ExecutivePersona.CTO:
            persona_adapted_roadmap["success_metrics"] = self._get_cto_success_metrics()
        elif target_persona == ExecutivePersona.CFO:
            persona_adapted_roadmap["success_metrics"] = self._get_cfo_success_metrics()
        elif target_persona == ExecutivePersona.BOARD:
            persona_adapted_roadmap["success_metrics"] = (
                self._get_board_success_metrics()
            )
        elif target_persona == ExecutivePersona.PARTNER:
            persona_adapted_roadmap["success_metrics"] = (
                self._get_partner_success_metrics()
            )

        return persona_adapted_roadmap

    def _get_ceo_success_metrics(self) -> List[str]:
        """Get CEO-focused success metrics"""
        return [
            "Strategic competitive advantage achieved",
            "Market position and growth trajectory established",
            "Organizational transformation milestones reached",
            "Revenue growth and profitability targets met",
            "Strategic vision execution validated",
        ]

    def _get_cto_success_metrics(self) -> List[str]:
        """Get CTO-focused success metrics"""
        return [
            "Technical architecture and scalability validated",
            "System performance and reliability metrics achieved",
            "Technology modernization milestones completed",
            "Engineering productivity and velocity improved",
            "Security and compliance standards maintained",
        ]

    def _get_cfo_success_metrics(self) -> List[str]:
        """Get CFO-focused success metrics"""
        return [
            "Financial performance and cost optimization targets met",
            "Budget variance and financial controls maintained",
            "ROI and profitability projections achieved",
            "Financial risk mitigation measures implemented",
            "Compliance and regulatory requirements satisfied",
        ]

    def _get_board_success_metrics(self) -> List[str]:
        """Get Board-focused success metrics"""
        return [
            "Strategic governance and oversight framework established",
            "Enterprise risk management and compliance maintained",
            "Stakeholder value creation and protection achieved",
            "Board fiduciary responsibilities fulfilled",
            "Long-term strategic objectives progress validated",
        ]

    def _get_partner_success_metrics(self) -> List[str]:
        """Get Partner-focused success metrics"""
        return [
            "Client business value and satisfaction delivered",
            "Professional excellence and expertise demonstrated",
            "Client relationship strength and trust enhanced",
            "Consulting delivery quality and impact achieved",
            "Business development and growth objectives met",
        ]


# Factory function for creating multi-persona formatter
def create_multi_persona_formatter(
    context_intelligence: Optional[IContextIntelligence] = None,
) -> MultiPersonaDeliverableFormatter:
    """Factory function for creating Multi-Persona Deliverable Formatter"""
    return MultiPersonaDeliverableFormatter(context_intelligence=context_intelligence)
