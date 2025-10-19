#!/usr/bin/env python3
"""
METIS Progressive Disclosure Manager - P7.3
Advanced progressive disclosure for complex analysis with cognitive load management

Implements sophisticated progressive disclosure patterns for complex consultancy
analyses, ensuring optimal information presentation based on user context and
cognitive load assessment.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from src.engine.models.data_contracts import (
    MetisDataContract,
)

try:
    from src.models.transparency_models import (
        TransparencyContent,
        TransparencyLayer,
        UserProfile,
        UserExpertiseLevel,
        CognitiveLoadLevel,
    )

    TRANSPARENCY_MODELS_AVAILABLE = True
except ImportError:
    TRANSPARENCY_MODELS_AVAILABLE = False


class DisclosureStrategy(str, Enum):
    """Progressive disclosure strategies"""

    LAYERED = "layered"  # Traditional layer-by-layer
    CONTEXTUAL = "contextual"  # Context-aware disclosure
    ADAPTIVE = "adaptive"  # AI-driven adaptive disclosure
    EXPLORATORY = "exploratory"  # User-guided exploration
    NARRATIVE = "narrative"  # Story-driven progression
    MODULAR = "modular"  # Component-based disclosure


class ComplexityLevel(str, Enum):
    """Analysis complexity levels"""

    SIMPLE = "simple"  # Single model, few steps
    MODERATE = "moderate"  # Multiple models, moderate steps
    COMPLEX = "complex"  # Many models, many steps
    HIGHLY_COMPLEX = "highly_complex"  # Intricate interconnections


class DisclosureState(str, Enum):
    """Current disclosure state"""

    COLLAPSED = "collapsed"  # Minimal information shown
    PARTIALLY_EXPANDED = "partially_expanded"  # Some details visible
    EXPANDED = "expanded"  # Full details shown
    DEEP_DIVE = "deep_dive"  # Expert-level detail


@dataclass
class DisclosureChunk:
    """Individual chunk of information in progressive disclosure"""

    chunk_id: str
    title: str
    content: str
    chunk_type: str  # summary, detail, evidence, technical, etc.

    # Disclosure properties
    priority: int = 0  # Higher priority shown first
    cognitive_load: CognitiveLoadLevel = CognitiveLoadLevel.MEDIUM
    prerequisite_chunks: List[str] = field(default_factory=list)

    # Visibility control
    visible: bool = False
    expanded: bool = False
    user_requested: bool = False

    # Content metadata
    estimated_read_time_seconds: int = 30
    complexity_score: float = 0.5
    confidence_level: float = 0.8

    # Interactive elements
    expandable: bool = True
    has_drill_down: bool = False
    drill_down_target: Optional[str] = None

    # Visual formatting
    format_hints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DisclosureSection:
    """Section containing multiple disclosure chunks"""

    section_id: str
    title: str
    description: str
    section_type: str  # overview, analysis, evidence, recommendations

    chunks: List[DisclosureChunk] = field(default_factory=list)
    subsections: List["DisclosureSection"] = field(default_factory=list)

    # Section properties
    priority: int = 0
    collapsed: bool = True
    estimated_completion_time: int = 0  # seconds

    # Progress tracking
    chunks_revealed: int = 0
    total_chunks: int = 0
    completion_percentage: float = 0.0

    # Dependencies
    prerequisite_sections: List[str] = field(default_factory=list)
    unlocks_sections: List[str] = field(default_factory=list)


@dataclass
class ProgressiveDisclosureState:
    """Complete state of progressive disclosure for an engagement"""

    disclosure_id: UUID
    engagement_id: UUID
    user_id: UUID
    strategy: DisclosureStrategy

    # Content organization
    sections: List[DisclosureSection] = field(default_factory=list)
    current_section: Optional[str] = None

    # User interaction state
    user_progress: Dict[str, Any] = field(default_factory=dict)
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)

    # Adaptive behavior
    cognitive_load_history: List[float] = field(default_factory=list)
    complexity_assessment: ComplexityLevel = ComplexityLevel.MODERATE

    # Timing and pacing
    session_start_time: datetime = field(default_factory=datetime.utcnow)
    total_time_spent: int = 0  # seconds
    estimated_remaining_time: int = 0  # seconds

    # Personalization
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    adaptive_adjustments: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DisclosureTemplate:
    """Template for disclosure patterns"""

    template_id: str
    name: str
    description: str
    strategy: DisclosureStrategy

    # Template structure
    section_templates: List[Dict[str, Any]] = field(default_factory=list)
    chunk_templates: List[Dict[str, Any]] = field(default_factory=list)

    # Application rules
    complexity_suitability: List[ComplexityLevel] = field(default_factory=list)
    user_type_suitability: List[UserExpertiseLevel] = field(default_factory=list)

    # Configuration
    default_settings: Dict[str, Any] = field(default_factory=dict)
    customization_options: Dict[str, Any] = field(default_factory=dict)


class ProgressiveDisclosureManager:
    """Manages progressive disclosure for complex analyses"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.templates = self._initialize_templates()
        self.active_disclosures: Dict[UUID, ProgressiveDisclosureState] = {}

    def _initialize_templates(self) -> Dict[str, DisclosureTemplate]:
        """Initialize disclosure templates"""

        templates = {}

        # Executive template - for executive users
        templates["executive"] = DisclosureTemplate(
            template_id="executive",
            name="Executive Summary Focus",
            description="Disclosure optimized for executive decision-making",
            strategy=DisclosureStrategy.LAYERED,
            complexity_suitability=[ComplexityLevel.SIMPLE, ComplexityLevel.MODERATE],
            user_type_suitability=[UserExpertiseLevel.EXECUTIVE],
            section_templates=[
                {"type": "executive_summary", "priority": 100, "auto_expand": True},
                {"type": "key_recommendations", "priority": 90, "auto_expand": True},
                {
                    "type": "implementation_roadmap",
                    "priority": 80,
                    "auto_expand": False,
                },
                {"type": "supporting_analysis", "priority": 50, "auto_expand": False},
            ],
        )

        # Analytical template - for analysts and researchers
        templates["analytical"] = DisclosureTemplate(
            template_id="analytical",
            name="Analytical Deep Dive",
            description="Disclosure for detailed analytical exploration",
            strategy=DisclosureStrategy.CONTEXTUAL,
            complexity_suitability=[ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX],
            user_type_suitability=[UserExpertiseLevel.ANALYTICAL],
            section_templates=[
                {"type": "methodology_overview", "priority": 100, "auto_expand": True},
                {"type": "data_analysis", "priority": 90, "auto_expand": True},
                {"type": "evidence_review", "priority": 85, "auto_expand": False},
                {"type": "validation_results", "priority": 80, "auto_expand": False},
                {"type": "detailed_findings", "priority": 70, "auto_expand": False},
            ],
        )

        # Technical template - for technical stakeholders
        templates["technical"] = DisclosureTemplate(
            template_id="technical",
            name="Technical Implementation",
            description="Disclosure for technical implementation details",
            strategy=DisclosureStrategy.MODULAR,
            complexity_suitability=[
                ComplexityLevel.COMPLEX,
                ComplexityLevel.HIGHLY_COMPLEX,
            ],
            user_type_suitability=[UserExpertiseLevel.TECHNICAL],
            section_templates=[
                {"type": "architecture_overview", "priority": 100, "auto_expand": True},
                {
                    "type": "implementation_details",
                    "priority": 90,
                    "auto_expand": False,
                },
                {
                    "type": "technical_specifications",
                    "priority": 85,
                    "auto_expand": False,
                },
                {"type": "performance_analysis", "priority": 80, "auto_expand": False},
                {"type": "integration_guidance", "priority": 75, "auto_expand": False},
            ],
        )

        # Adaptive template - AI-driven adaptation
        templates["adaptive"] = DisclosureTemplate(
            template_id="adaptive",
            name="Adaptive Disclosure",
            description="AI-driven adaptive disclosure based on user behavior",
            strategy=DisclosureStrategy.ADAPTIVE,
            complexity_suitability=list(ComplexityLevel),
            user_type_suitability=list(UserExpertiseLevel),
            section_templates=[
                {"type": "dynamic_summary", "priority": 100, "auto_expand": True},
                {"type": "contextual_details", "priority": 80, "auto_expand": False},
                {
                    "type": "adaptive_recommendations",
                    "priority": 70,
                    "auto_expand": False,
                },
            ],
        )

        return templates

    async def create_progressive_disclosure(
        self,
        engagement_contract: MetisDataContract,
        user_profile: UserProfile,
        transparency_content: Dict[TransparencyLayer, TransparencyContent],
    ) -> ProgressiveDisclosureState:
        """Create progressive disclosure for complex analysis"""

        # Assess complexity
        complexity = await self._assess_analysis_complexity(engagement_contract)

        # Select appropriate template
        template = await self._select_template(user_profile, complexity)

        # Create disclosure state
        disclosure_state = ProgressiveDisclosureState(
            disclosure_id=uuid4(),
            engagement_id=engagement_contract.engagement_context.engagement_id,
            user_id=user_profile.user_id,
            strategy=template.strategy,
            complexity_assessment=complexity,
        )

        # Build sections from transparency content
        sections = await self._build_sections_from_content(
            transparency_content, template, complexity, user_profile
        )
        disclosure_state.sections = sections

        # Apply initial disclosure rules
        await self._apply_initial_disclosure(disclosure_state, template, user_profile)

        # Register active disclosure
        self.active_disclosures[disclosure_state.disclosure_id] = disclosure_state

        self.logger.info(
            f"Created progressive disclosure for {user_profile.expertise_level} user with {complexity} complexity"
        )

        return disclosure_state

    async def _assess_analysis_complexity(
        self, contract: MetisDataContract
    ) -> ComplexityLevel:
        """Assess the complexity of the analysis"""

        reasoning_steps = len(contract.cognitive_state.reasoning_steps)
        mental_models = len(contract.cognitive_state.selected_mental_models)
        deliverables = len(contract.deliverable_artifacts)

        # Calculate complexity score
        complexity_score = (
            reasoning_steps * 0.4 + mental_models * 0.3 + deliverables * 0.3
        )

        # Assess interconnections
        model_diversity = len(
            set(
                step.mental_model_applied
                for step in contract.cognitive_state.reasoning_steps
            )
        )

        interconnection_factor = model_diversity / max(1, mental_models)
        complexity_score *= 1 + interconnection_factor

        # Determine complexity level
        if complexity_score <= 3:
            return ComplexityLevel.SIMPLE
        elif complexity_score <= 6:
            return ComplexityLevel.MODERATE
        elif complexity_score <= 10:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.HIGHLY_COMPLEX

    async def _select_template(
        self, user_profile: UserProfile, complexity: ComplexityLevel
    ) -> DisclosureTemplate:
        """Select appropriate disclosure template"""

        # Find templates suitable for user and complexity
        suitable_templates = []

        for template in self.templates.values():
            if (
                user_profile.expertise_level in template.user_type_suitability
                and complexity in template.complexity_suitability
            ):
                suitable_templates.append(template)

        # Prefer user-specific templates
        user_specific_templates = [
            t
            for t in suitable_templates
            if user_profile.expertise_level.value in t.template_id
        ]

        if user_specific_templates:
            return user_specific_templates[0]
        elif suitable_templates:
            return suitable_templates[0]
        else:
            # Fallback to adaptive template
            return self.templates["adaptive"]

    async def _build_sections_from_content(
        self,
        transparency_content: Dict[TransparencyLayer, TransparencyContent],
        template: DisclosureTemplate,
        complexity: ComplexityLevel,
        user_profile: UserProfile,
    ) -> List[DisclosureSection]:
        """Build disclosure sections from transparency content"""

        sections = []

        # Create sections based on template
        for section_template in template.section_templates:
            section = await self._create_section_from_template(
                section_template, transparency_content, complexity, user_profile
            )
            if section:
                sections.append(section)

        # Sort sections by priority
        sections.sort(key=lambda s: s.priority, reverse=True)

        return sections

    async def _create_section_from_template(
        self,
        section_template: Dict[str, Any],
        transparency_content: Dict[TransparencyLayer, TransparencyContent],
        complexity: ComplexityLevel,
        user_profile: UserProfile,
    ) -> Optional[DisclosureSection]:
        """Create a section from template and content"""

        section_type = section_template["type"]

        if section_type == "executive_summary":
            return await self._create_executive_summary_section(
                transparency_content, section_template, user_profile
            )
        elif section_type == "methodology_overview":
            return await self._create_methodology_section(
                transparency_content, section_template, user_profile
            )
        elif section_type == "evidence_review":
            return await self._create_evidence_section(
                transparency_content, section_template, user_profile
            )
        elif section_type == "technical_specifications":
            return await self._create_technical_section(
                transparency_content, section_template, user_profile
            )
        elif section_type == "dynamic_summary":
            return await self._create_adaptive_summary_section(
                transparency_content, section_template, user_profile, complexity
            )
        else:
            # Create generic section
            return await self._create_generic_section(
                section_type, transparency_content, section_template, user_profile
            )

    async def _create_executive_summary_section(
        self,
        transparency_content: Dict[TransparencyLayer, TransparencyContent],
        template: Dict[str, Any],
        user_profile: UserProfile,
    ) -> DisclosureSection:
        """Create executive summary section"""

        # Get executive summary content
        exec_content = transparency_content.get(TransparencyLayer.EXECUTIVE_SUMMARY)

        section = DisclosureSection(
            section_id="executive_summary",
            title="Executive Summary",
            description="Strategic overview and key recommendations",
            section_type="executive_summary",
            priority=template.get("priority", 100),
            collapsed=not template.get("auto_expand", False),
        )

        if exec_content:
            # Create chunks for executive content
            chunks = []

            # Key insights chunk
            if exec_content.key_insights:
                chunks.append(
                    DisclosureChunk(
                        chunk_id="key_insights",
                        title="Key Insights",
                        content="\n".join(
                            f"• {insight}" for insight in exec_content.key_insights[:3]
                        ),
                        chunk_type="summary",
                        priority=100,
                        cognitive_load=CognitiveLoadLevel.LOW,
                        visible=True,
                        estimated_read_time_seconds=45,
                    )
                )

            # Main content chunk
            content_preview = (
                exec_content.content[:500] + "..."
                if len(exec_content.content) > 500
                else exec_content.content
            )
            chunks.append(
                DisclosureChunk(
                    chunk_id="exec_main_content",
                    title="Strategic Analysis",
                    content=content_preview,
                    chunk_type="summary",
                    priority=90,
                    cognitive_load=CognitiveLoadLevel.MEDIUM,
                    visible=not template.get("auto_expand", False),
                    expandable=True,
                    estimated_read_time_seconds=120,
                )
            )

            # Confidence indicators chunk
            if exec_content.confidence_indicators:
                confidence_text = self._format_confidence_indicators(
                    exec_content.confidence_indicators
                )
                chunks.append(
                    DisclosureChunk(
                        chunk_id="confidence_summary",
                        title="Analysis Confidence",
                        content=confidence_text,
                        chunk_type="evidence",
                        priority=80,
                        cognitive_load=CognitiveLoadLevel.LOW,
                        visible=False,
                        estimated_read_time_seconds=30,
                    )
                )

            section.chunks = chunks
            section.total_chunks = len(chunks)
            section.chunks_revealed = len([c for c in chunks if c.visible])

        return section

    async def _create_methodology_section(
        self,
        transparency_content: Dict[TransparencyLayer, TransparencyContent],
        template: Dict[str, Any],
        user_profile: UserProfile,
    ) -> DisclosureSection:
        """Create methodology overview section"""

        reasoning_content = transparency_content.get(
            TransparencyLayer.REASONING_OVERVIEW
        )

        section = DisclosureSection(
            section_id="methodology",
            title="Analytical Methodology",
            description="Mental models and reasoning process",
            section_type="methodology_overview",
            priority=template.get("priority", 90),
            collapsed=not template.get("auto_expand", False),
        )

        if reasoning_content:
            chunks = []

            # Mental models chunk
            if reasoning_content.metadata.get("models_count", 0) > 0:
                chunks.append(
                    DisclosureChunk(
                        chunk_id="mental_models_overview",
                        title=f"Mental Models Applied ({reasoning_content.metadata['models_count']})",
                        content="Systematic application of proven analytical frameworks",
                        chunk_type="methodology",
                        priority=100,
                        cognitive_load=CognitiveLoadLevel.MEDIUM,
                        visible=True,
                        has_drill_down=True,
                        estimated_read_time_seconds=60,
                    )
                )

            # Reasoning steps chunk
            if reasoning_content.metadata.get("steps_count", 0) > 0:
                chunks.append(
                    DisclosureChunk(
                        chunk_id="reasoning_process",
                        title=f"Reasoning Process ({reasoning_content.metadata['steps_count']} steps)",
                        content="Systematic step-by-step analytical progression",
                        chunk_type="methodology",
                        priority=90,
                        cognitive_load=CognitiveLoadLevel.MEDIUM,
                        visible=template.get("auto_expand", False),
                        has_drill_down=True,
                        estimated_read_time_seconds=90,
                    )
                )

            # Cognitive trace chunk (if available)
            if reasoning_content.cognitive_trace:
                chunks.append(
                    DisclosureChunk(
                        chunk_id="cognitive_trace",
                        title="Interactive Reasoning Trace",
                        content="Visual representation of cognitive reasoning process",
                        chunk_type="visualization",
                        priority=80,
                        cognitive_load=CognitiveLoadLevel.HIGH,
                        visible=False,
                        expandable=True,
                        estimated_read_time_seconds=180,
                    )
                )

            section.chunks = chunks
            section.total_chunks = len(chunks)
            section.chunks_revealed = len([c for c in chunks if c.visible])

        return section

    async def _create_evidence_section(
        self,
        transparency_content: Dict[TransparencyLayer, TransparencyContent],
        template: Dict[str, Any],
        user_profile: UserProfile,
    ) -> DisclosureSection:
        """Create evidence review section"""

        section = DisclosureSection(
            section_id="evidence_review",
            title="Evidence & Validation",
            description="Supporting evidence and quality assessment",
            section_type="evidence_review",
            priority=template.get("priority", 85),
            collapsed=True,
        )

        chunks = []

        # Collect evidence from all layers
        all_evidence = []
        for content in transparency_content.values():
            if content.validation_evidence:
                all_evidence.extend(content.validation_evidence)

        if all_evidence:
            # Evidence quality summary
            total_evidence = sum(len(ev.evidence_items) for ev in all_evidence)
            strong_evidence = sum(
                len(
                    [
                        item
                        for item in ev.evidence_items
                        if item.quality.value == "strong"
                    ]
                )
                for ev in all_evidence
            )

            chunks.append(
                DisclosureChunk(
                    chunk_id="evidence_quality_summary",
                    title="Evidence Quality Overview",
                    content=f"Total evidence items: {total_evidence}\nHigh-quality evidence: {strong_evidence}",
                    chunk_type="evidence",
                    priority=100,
                    cognitive_load=CognitiveLoadLevel.MEDIUM,
                    visible=True,
                    estimated_read_time_seconds=45,
                )
            )

            # Detailed evidence breakdown
            chunks.append(
                DisclosureChunk(
                    chunk_id="evidence_breakdown",
                    title="Detailed Evidence Analysis",
                    content="Comprehensive review of all supporting evidence",
                    chunk_type="evidence",
                    priority=90,
                    cognitive_load=CognitiveLoadLevel.HIGH,
                    visible=False,
                    expandable=True,
                    has_drill_down=True,
                    estimated_read_time_seconds=240,
                )
            )

        section.chunks = chunks
        section.total_chunks = len(chunks)
        section.chunks_revealed = len([c for c in chunks if c.visible])

        return section

    async def _create_technical_section(
        self,
        transparency_content: Dict[TransparencyLayer, TransparencyContent],
        template: Dict[str, Any],
        user_profile: UserProfile,
    ) -> DisclosureSection:
        """Create technical specifications section"""

        tech_content = transparency_content.get(TransparencyLayer.TECHNICAL_EXECUTION)

        section = DisclosureSection(
            section_id="technical_specs",
            title="Technical Implementation",
            description="System architecture and implementation details",
            section_type="technical_specifications",
            priority=template.get("priority", 75),
            collapsed=True,
        )

        if tech_content:
            chunks = []

            # System architecture chunk
            chunks.append(
                DisclosureChunk(
                    chunk_id="system_architecture",
                    title="System Architecture",
                    content="Core system components and integration patterns",
                    chunk_type="technical",
                    priority=100,
                    cognitive_load=CognitiveLoadLevel.HIGH,
                    visible=True,
                    estimated_read_time_seconds=120,
                )
            )

            # Performance metrics chunk
            chunks.append(
                DisclosureChunk(
                    chunk_id="performance_metrics",
                    title="Performance Analysis",
                    content="System performance characteristics and benchmarks",
                    chunk_type="technical",
                    priority=90,
                    cognitive_load=CognitiveLoadLevel.HIGH,
                    visible=False,
                    expandable=True,
                    estimated_read_time_seconds=180,
                )
            )

            section.chunks = chunks
            section.total_chunks = len(chunks)
            section.chunks_revealed = len([c for c in chunks if c.visible])

        return section

    async def _create_adaptive_summary_section(
        self,
        transparency_content: Dict[TransparencyLayer, TransparencyContent],
        template: Dict[str, Any],
        user_profile: UserProfile,
        complexity: ComplexityLevel,
    ) -> DisclosureSection:
        """Create adaptive summary section"""

        section = DisclosureSection(
            section_id="adaptive_summary",
            title="Adaptive Analysis Summary",
            description="Dynamically adjusted content based on your profile",
            section_type="dynamic_summary",
            priority=template.get("priority", 100),
            collapsed=False,
        )

        chunks = []

        # Create adaptive content based on user expertise and complexity
        if user_profile.expertise_level == UserExpertiseLevel.EXECUTIVE:
            if complexity in [ComplexityLevel.COMPLEX, ComplexityLevel.HIGHLY_COMPLEX]:
                chunks.append(
                    DisclosureChunk(
                        chunk_id="simplified_overview",
                        title="Simplified Strategic Overview",
                        content="Complex analysis simplified for strategic decision-making",
                        chunk_type="adaptive_summary",
                        priority=100,
                        cognitive_load=CognitiveLoadLevel.LOW,
                        visible=True,
                        estimated_read_time_seconds=90,
                    )
                )

        elif user_profile.expertise_level == UserExpertiseLevel.ANALYTICAL:
            chunks.append(
                DisclosureChunk(
                    chunk_id="analytical_insights",
                    title="Key Analytical Insights",
                    content="Data-driven insights tailored for analytical review",
                    chunk_type="adaptive_summary",
                    priority=100,
                    cognitive_load=CognitiveLoadLevel.MEDIUM,
                    visible=True,
                    estimated_read_time_seconds=120,
                )
            )

        section.chunks = chunks
        section.total_chunks = len(chunks)
        section.chunks_revealed = len([c for c in chunks if c.visible])

        return section

    async def _create_generic_section(
        self,
        section_type: str,
        transparency_content: Dict[TransparencyLayer, TransparencyContent],
        template: Dict[str, Any],
        user_profile: UserProfile,
    ) -> DisclosureSection:
        """Create generic section for unspecified types"""

        section = DisclosureSection(
            section_id=section_type,
            title=section_type.replace("_", " ").title(),
            description=f"Content for {section_type}",
            section_type=section_type,
            priority=template.get("priority", 50),
            collapsed=True,
        )

        # Add generic chunk
        section.chunks = [
            DisclosureChunk(
                chunk_id=f"{section_type}_content",
                title="Content",
                content="Generic content placeholder",
                chunk_type="generic",
                priority=100,
                cognitive_load=CognitiveLoadLevel.MEDIUM,
                visible=True,
                estimated_read_time_seconds=60,
            )
        ]

        section.total_chunks = len(section.chunks)
        section.chunks_revealed = 1

        return section

    async def _apply_initial_disclosure(
        self,
        disclosure_state: ProgressiveDisclosureState,
        template: DisclosureTemplate,
        user_profile: UserProfile,
    ):
        """Apply initial disclosure rules"""

        # Set initial section visibility based on template
        for section in disclosure_state.sections:
            # Auto-expand high priority sections for executives
            if (
                user_profile.expertise_level == UserExpertiseLevel.EXECUTIVE
                and section.priority >= 90
            ):
                section.collapsed = False
                for chunk in section.chunks[:2]:  # Show first 2 chunks
                    chunk.visible = True
                    section.chunks_revealed += 1 if not chunk.visible else 0

            # Show methodology for analytical users
            elif (
                user_profile.expertise_level == UserExpertiseLevel.ANALYTICAL
                and section.section_type == "methodology_overview"
            ):
                section.collapsed = False
                for chunk in section.chunks:
                    if chunk.priority >= 90:
                        chunk.visible = True
                        section.chunks_revealed += 1 if not chunk.visible else 0

        # Calculate completion percentages
        for section in disclosure_state.sections:
            if section.total_chunks > 0:
                section.completion_percentage = (
                    section.chunks_revealed / section.total_chunks
                )

    def _format_confidence_indicators(self, indicators: Dict[str, Any]) -> str:
        """Format confidence indicators for display"""

        lines = []
        for key, value in indicators.items():
            if isinstance(value, float):
                lines.append(f"• {key.replace('_', ' ').title()}: {value:.0%}")
            else:
                lines.append(f"• {key.replace('_', ' ').title()}: {value}")

        return "\n".join(lines)

    async def handle_user_interaction(
        self,
        disclosure_id: UUID,
        interaction_type: str,
        target_id: str,
        user_context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Handle user interaction with progressive disclosure"""

        disclosure_state = self.active_disclosures.get(disclosure_id)
        if not disclosure_state:
            return {"error": "Disclosure not found"}

        # Record interaction
        interaction = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": interaction_type,
            "target": target_id,
            "context": user_context or {},
        }
        disclosure_state.interaction_history.append(interaction)

        # Handle different interaction types
        if interaction_type == "expand_chunk":
            result = await self._handle_chunk_expansion(disclosure_state, target_id)
        elif interaction_type == "expand_section":
            result = await self._handle_section_expansion(disclosure_state, target_id)
        elif interaction_type == "request_detail":
            result = await self._handle_detail_request(disclosure_state, target_id)
        elif interaction_type == "navigate_to":
            result = await self._handle_navigation(disclosure_state, target_id)
        else:
            result = {"error": f"Unknown interaction type: {interaction_type}"}

        # Update adaptive behavior based on interaction
        await self._update_adaptive_behavior(disclosure_state, interaction)

        return result

    async def _handle_chunk_expansion(
        self, disclosure_state: ProgressiveDisclosureState, chunk_id: str
    ) -> Dict[str, Any]:
        """Handle chunk expansion request"""

        # Find and expand chunk
        for section in disclosure_state.sections:
            for chunk in section.chunks:
                if chunk.chunk_id == chunk_id:
                    chunk.expanded = True
                    chunk.user_requested = True

                    # Make visible if not already
                    if not chunk.visible:
                        chunk.visible = True
                        section.chunks_revealed += 1
                        section.completion_percentage = (
                            section.chunks_revealed / section.total_chunks
                        )

                    return {
                        "success": True,
                        "chunk_id": chunk_id,
                        "expanded": True,
                        "content": chunk.content,
                    }

        return {"error": "Chunk not found"}

    async def _handle_section_expansion(
        self, disclosure_state: ProgressiveDisclosureState, section_id: str
    ) -> Dict[str, Any]:
        """Handle section expansion request"""

        # Find and expand section
        for section in disclosure_state.sections:
            if section.section_id == section_id:
                section.collapsed = False

                # Make high-priority chunks visible
                newly_revealed = 0
                for chunk in section.chunks:
                    if chunk.priority >= 80 and not chunk.visible:
                        chunk.visible = True
                        newly_revealed += 1

                section.chunks_revealed += newly_revealed
                section.completion_percentage = (
                    section.chunks_revealed / section.total_chunks
                )

                return {
                    "success": True,
                    "section_id": section_id,
                    "expanded": True,
                    "chunks_revealed": newly_revealed,
                }

        return {"error": "Section not found"}

    async def _handle_detail_request(
        self, disclosure_state: ProgressiveDisclosureState, target_id: str
    ) -> Dict[str, Any]:
        """Handle request for additional detail"""

        # This would trigger showing more detailed content
        # Implementation depends on specific content structure

        return {
            "success": True,
            "message": "Additional detail provided",
            "target": target_id,
        }

    async def _handle_navigation(
        self, disclosure_state: ProgressiveDisclosureState, target_section: str
    ) -> Dict[str, Any]:
        """Handle navigation to specific section"""

        disclosure_state.current_section = target_section

        return {"success": True, "current_section": target_section}

    async def _update_adaptive_behavior(
        self, disclosure_state: ProgressiveDisclosureState, interaction: Dict[str, Any]
    ):
        """Update adaptive behavior based on user interaction"""

        # Track cognitive load patterns
        if interaction["type"] in ["expand_chunk", "expand_section"]:
            # User is comfortable with current load, can increase
            disclosure_state.cognitive_load_history.append(0.1)
        elif interaction["type"] == "request_detail":
            # User wants more detail, high engagement
            disclosure_state.cognitive_load_history.append(0.2)

        # Adapt future disclosures based on patterns
        recent_interactions = disclosure_state.interaction_history[-5:]
        expansion_requests = len(
            [i for i in recent_interactions if i["type"].startswith("expand")]
        )

        if expansion_requests >= 3:
            # User is comfortable with complexity, can auto-expand more
            adaptation = {
                "timestamp": datetime.utcnow().isoformat(),
                "type": "increase_auto_expansion",
                "reason": "High expansion rate detected",
            }
            disclosure_state.adaptive_adjustments.append(adaptation)

    async def get_disclosure_state(
        self, disclosure_id: UUID
    ) -> Optional[ProgressiveDisclosureState]:
        """Get current disclosure state"""
        return self.active_disclosures.get(disclosure_id)

    async def render_disclosure_json(self, disclosure_id: UUID) -> Dict[str, Any]:
        """Render disclosure state as JSON for frontend"""

        disclosure_state = self.active_disclosures.get(disclosure_id)
        if not disclosure_state:
            return {"error": "Disclosure not found"}

        return {
            "disclosure_id": str(disclosure_state.disclosure_id),
            "engagement_id": str(disclosure_state.engagement_id),
            "strategy": disclosure_state.strategy.value,
            "complexity": disclosure_state.complexity_assessment.value,
            "sections": [
                {
                    "section_id": section.section_id,
                    "title": section.title,
                    "description": section.description,
                    "type": section.section_type,
                    "priority": section.priority,
                    "collapsed": section.collapsed,
                    "completion_percentage": section.completion_percentage,
                    "chunks": [
                        {
                            "chunk_id": chunk.chunk_id,
                            "title": chunk.title,
                            "content": chunk.content if chunk.visible else chunk.title,
                            "type": chunk.chunk_type,
                            "visible": chunk.visible,
                            "expanded": chunk.expanded,
                            "expandable": chunk.expandable,
                            "cognitive_load": chunk.cognitive_load.value,
                            "estimated_read_time": chunk.estimated_read_time_seconds,
                            "has_drill_down": chunk.has_drill_down,
                        }
                        for chunk in section.chunks
                    ],
                }
                for section in disclosure_state.sections
            ],
            "user_progress": disclosure_state.user_progress,
            "estimated_remaining_time": disclosure_state.estimated_remaining_time,
        }


# Export main classes
__all__ = [
    "ProgressiveDisclosureManager",
    "ProgressiveDisclosureState",
    "DisclosureSection",
    "DisclosureChunk",
    "DisclosureStrategy",
    "ComplexityLevel",
    "DisclosureState",
]
