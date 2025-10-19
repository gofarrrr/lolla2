"""
METIS Transparency Models
Extracted transparency-related data models to resolve circular import dependencies

Contains all transparency and progressive disclosure data structures
used across the METIS system without circular import issues.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from uuid import UUID


class TransparencyLayer(str, Enum):
    """Progressive transparency disclosure layers"""

    EXECUTIVE_SUMMARY = "executive_summary"  # Layer 1: Strategic conclusions
    REASONING_OVERVIEW = "reasoning_overview"  # Layer 2: Methodology visibility
    DETAILED_AUDIT_TRAIL = "detailed_audit_trail"  # Layer 3: Complete reasoning process
    TECHNICAL_EXECUTION = "technical_execution"  # Layer 4: Implementation details


class UserExpertiseLevel(str, Enum):
    """User expertise levels for adaptive interface"""

    EXECUTIVE = "executive"  # C-suite, board members
    STRATEGIC = "strategic"  # Strategy professionals, consultants
    ANALYTICAL = "analytical"  # Analysts, researchers
    TECHNICAL = "technical"  # Data scientists, engineers


class CognitiveLoadLevel(str, Enum):
    """Cognitive load assessment levels"""

    LOW = "low"  # Easy to process
    MEDIUM = "medium"  # Moderate complexity
    HIGH = "high"  # Complex, requires focus
    OVERWHELMING = "overwhelming"  # Too complex, needs simplification


class ValidationEvidenceType(str, Enum):
    """Types of validation evidence"""

    LOGICAL_CONSISTENCY = "logical_consistency"
    EMPIRICAL_SUPPORT = "empirical_support"
    STAKEHOLDER_ALIGNMENT = "stakeholder_alignment"
    HISTORICAL_PRECEDENT = "historical_precedent"
    QUANTITATIVE_ANALYSIS = "quantitative_analysis"
    EXPERT_VALIDATION = "expert_validation"
    PEER_REVIEW = "peer_review"


class EvidenceQuality(str, Enum):
    """Quality levels for evidence"""

    STRONG = "strong"  # High confidence, multiple sources
    MODERATE = "moderate"  # Good confidence, some validation
    WEAK = "weak"  # Limited confidence, needs verification
    INSUFFICIENT = "insufficient"  # Not enough evidence


@dataclass
class ValidationEvidence:
    """Individual piece of validation evidence"""

    evidence_id: str
    evidence_type: ValidationEvidenceType
    quality: EvidenceQuality
    confidence_score: float
    description: str
    source: str
    reasoning_step_id: Optional[str] = None
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    validation_notes: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_display_format(self) -> Dict[str, Any]:
        """Convert to format suitable for UI display"""
        return {
            "id": self.evidence_id,
            "type": self.evidence_type.value,
            "quality": self.quality.value,
            "confidence": self.confidence_score,
            "description": self.description,
            "source": self.source,
        }


@dataclass
class ValidationEvidenceCollection:
    """Collection of validation evidence for reasoning steps"""

    reasoning_step_id: str
    evidence_items: List[ValidationEvidence] = field(default_factory=list)
    overall_confidence: float = 0.0
    validation_summary: str = ""

    def add_evidence(self, evidence: ValidationEvidence):
        """Add evidence and recalculate confidence"""
        self.evidence_items.append(evidence)
        self._recalculate_confidence()

    def _recalculate_confidence(self):
        """Recalculate overall confidence based on evidence"""
        if not self.evidence_items:
            self.overall_confidence = 0.0
            return

        # Weighted average based on evidence quality
        total_weight = 0.0
        weighted_confidence = 0.0

        quality_weights = {
            EvidenceQuality.STRONG: 1.0,
            EvidenceQuality.MODERATE: 0.7,
            EvidenceQuality.WEAK: 0.4,
            EvidenceQuality.INSUFFICIENT: 0.1,
        }

        for evidence in self.evidence_items:
            weight = quality_weights.get(evidence.quality, 0.5)
            weighted_confidence += evidence.confidence_score * weight
            total_weight += weight

        if total_weight > 0:
            self.overall_confidence = weighted_confidence / total_weight
        else:
            self.overall_confidence = 0.0


@dataclass
class ValidationEvidenceVisualization:
    """Visualization data for validation evidence"""

    visualization_type: str
    evidence_map: Dict[str, Any] = field(default_factory=dict)
    confidence_indicators: List[Dict[str, Any]] = field(default_factory=list)
    quality_distribution: Dict[str, int] = field(default_factory=dict)

    def generate_confidence_chart(self) -> Dict[str, Any]:
        """Generate confidence visualization data"""
        return {
            "type": "confidence_radar",
            "data": self.confidence_indicators,
            "layout": {
                "title": "Evidence Confidence Analysis",
                "polar": {"radialaxis": {"range": [0, 1]}},
            },
        }


@dataclass
class UserProfile:
    """User profile for transparency personalization"""

    user_id: UUID
    expertise_level: UserExpertiseLevel = UserExpertiseLevel.STRATEGIC
    preferred_cognitive_load: CognitiveLoadLevel = CognitiveLoadLevel.MEDIUM
    preferred_layer: TransparencyLayer = TransparencyLayer.REASONING_OVERVIEW
    persona_type: str = "strategic"  # executive, strategic, analytical, technical
    cognitive_preferences: Dict[str, Any] = field(default_factory=dict)
    auto_adjust_complexity: bool = True
    show_confidence_indicators: bool = True
    enable_progressive_hints: bool = True
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    learning_trajectory: List[str] = field(default_factory=list)
    personalization_preferences: Dict[str, Any] = field(default_factory=dict)
    # Legacy compatibility fields
    role: str = "strategic"  # Maps to persona_type
    drop_off_complexity_threshold: Optional[float] = None

    # Adaptive learning
    session_engagement_patterns: List[Dict[str, Any]] = field(default_factory=list)
    expertise_confidence: float = 0.7

    def update_interaction_pattern(
        self,
        layer: TransparencyLayer,
        engagement_time: float,
        comprehension_indicators: Dict[str, Any],
    ):
        """Update user interaction patterns for adaptive learning"""
        pattern = {
            "timestamp": datetime.utcnow().isoformat(),
            "layer": layer.value,
            "engagement_time": engagement_time,
            "comprehension": comprehension_indicators,
        }
        self.session_engagement_patterns.append(pattern)

        # Limit history to last 50 interactions
        if len(self.session_engagement_patterns) > 50:
            self.session_engagement_patterns = self.session_engagement_patterns[-50:]


@dataclass
class TransparencyContent:
    """Content for specific transparency layer"""

    layer: TransparencyLayer
    title: str
    content: str
    cognitive_load: CognitiveLoadLevel
    key_insights: List[str] = field(default_factory=list)
    supporting_evidence: List[ValidationEvidence] = field(default_factory=list)

    # Interactive elements
    expandable_sections: List[Dict[str, Any]] = field(default_factory=list)
    interactive_visualizations: List[Dict[str, Any]] = field(default_factory=list)

    # Navigation aids
    reading_time_estimate: int = 5  # minutes
    complexity_indicators: Dict[str, Any] = field(default_factory=dict)
    prerequisite_knowledge: List[str] = field(default_factory=list)

    # Additional metadata and visualization support
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_display_format(self, user_profile: UserProfile) -> Dict[str, Any]:
        """Convert to format optimized for specific user"""
        base_format = {
            "layer": self.layer.value,
            "title": self.title,
            "content": self.content,
            "cognitive_load": self.cognitive_load.value,
            "key_insights": self.key_insights,
            "reading_time": self.reading_time_estimate,
        }

        # Adapt based on user expertise
        if user_profile.expertise_level in [
            UserExpertiseLevel.EXECUTIVE,
            UserExpertiseLevel.STRATEGIC,
        ]:
            # Show summary view with key insights prominent
            base_format["content_type"] = "executive_summary"
            base_format["expanded"] = False
        else:
            # Show detailed view for analytical users
            base_format["content_type"] = "detailed_analysis"
            base_format["expanded"] = True
            base_format["supporting_evidence"] = [
                ev.to_display_format() for ev in self.supporting_evidence
            ]

        return base_format


@dataclass
class ProgressiveDisclosure:
    """Complete progressive disclosure package"""

    engagement_id: UUID
    layers: Dict[TransparencyLayer, TransparencyContent] = field(default_factory=dict)
    navigation_guidance: Dict[str, Any] = field(default_factory=dict)
    personalization_metadata: Dict[str, Any] = field(default_factory=dict)

    # User experience optimization
    recommended_path: List[TransparencyLayer] = field(default_factory=list)
    adaptive_suggestions: List[str] = field(default_factory=list)

    def get_layer_content(
        self, layer: TransparencyLayer, user_profile: UserProfile
    ) -> Optional[TransparencyContent]:
        """Get content for specific layer, adapted for user"""
        content = self.layers.get(layer)
        if content:
            # Apply user-specific adaptations
            return content
        return None

    def get_recommended_navigation(
        self, user_profile: UserProfile
    ) -> List[TransparencyLayer]:
        """Get recommended navigation path based on user profile"""
        if user_profile.expertise_level == UserExpertiseLevel.EXECUTIVE:
            return [
                TransparencyLayer.EXECUTIVE_SUMMARY,
                TransparencyLayer.REASONING_OVERVIEW,
            ]
        elif user_profile.expertise_level == UserExpertiseLevel.STRATEGIC:
            return [
                TransparencyLayer.EXECUTIVE_SUMMARY,
                TransparencyLayer.REASONING_OVERVIEW,
                TransparencyLayer.DETAILED_AUDIT_TRAIL,
            ]
        else:
            return [layer for layer in TransparencyLayer]

    def to_api_response(self, user_profile: UserProfile) -> Dict[str, Any]:
        """Convert to API response format"""
        return {
            "engagement_id": str(self.engagement_id),
            "layers": {
                layer.value: content.to_display_format(user_profile)
                for layer, content in self.layers.items()
            },
            "navigation": {
                "recommended_path": [
                    layer.value
                    for layer in self.get_recommended_navigation(user_profile)
                ],
                "guidance": self.navigation_guidance,
                "suggestions": self.adaptive_suggestions,
            },
            "personalization": self.personalization_metadata,
        }
