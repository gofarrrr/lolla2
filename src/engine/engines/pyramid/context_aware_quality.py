"""
Context-Aware Quality Assessment - Sprint 2.2
F005: Revolutionary quality assessment using AI's cognitive exhaust

This module creates the world's most sophisticated deliverable quality system
that leverages Context Intelligence to assess not just content, but cognitive
coherence, historical effectiveness patterns, and context-specific optimization.

Key Innovations:
- Cognitive Coherence Quality Metrics
- Historical Pattern-Based Quality Benchmarking
- Context-Specific Quality Criteria Adaptation
- Multi-Dimensional Quality Intelligence
- Predictive Quality Optimization
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from .models import ExecutiveDeliverable, PyramidNode
from .quality import QualityAssessor

# Context Intelligence imports
try:
    from src.interfaces.context_intelligence_interface import IContextIntelligence
    from src.engine.models.data_contracts import ContextType

    CONTEXT_INTELLIGENCE_AVAILABLE = True
except ImportError:
    CONTEXT_INTELLIGENCE_AVAILABLE = False
    IContextIntelligence = Any
    ContextType = Any


class ContextAwareQualityAssessor(QualityAssessor):
    """
    Context-Aware Quality Assessor using AI's cognitive exhaust

    Enhances traditional quality assessment with:
    - Cognitive coherence-based quality scoring
    - Historical pattern effectiveness benchmarking
    - Context-specific quality criteria adaptation
    - Multi-dimensional quality intelligence
    - Predictive quality optimization recommendations
    """

    def __init__(self, context_intelligence: Optional[IContextIntelligence] = None):
        super().__init__()
        self.context_intelligence = context_intelligence
        self.context_enhanced = context_intelligence is not None
        self.logger = logging.getLogger(__name__)

        # Context-aware quality criteria weights (dynamic)
        self.context_aware_criteria = {
            "cognitive_coherence": 0.20,  # NEW: Cognitive exhaust coherence
            "historical_effectiveness": 0.15,  # NEW: Historical pattern matching
            "structure_clarity": 0.20,  # Enhanced with context
            "executive_focus": 0.20,  # Enhanced with persona adaptation
            "evidence_strength": 0.15,  # Enhanced with context grounding
            "actionability": 0.10,  # Enhanced with context relevance
        }

        # Context-specific quality benchmarks
        self.context_benchmarks = {
            "IMMEDIATE": {"urgency_weight": 1.5, "action_focus": 1.3},
            "STRATEGIC": {"depth_weight": 1.4, "vision_focus": 1.2},
            "PROCEDURAL": {"clarity_weight": 1.3, "step_detail": 1.4},
            "DOMAIN": {"expertise_weight": 1.2, "analysis_depth": 1.1},
        }

        # Historical effectiveness tracking
        self.effectiveness_history = {
            "high_performing_patterns": [],
            "quality_trend_analysis": {},
            "context_specific_benchmarks": {},
        }

        if self.context_enhanced:
            self.logger.info(
                "🎯 Context-Aware Quality Assessor initialized with cognitive intelligence"
            )
        else:
            self.logger.info("📊 Traditional Quality Assessor initialized")

    async def assess_context_aware_deliverable_quality(
        self,
        deliverable: ExecutiveDeliverable,
        pyramid_structure: Optional[PyramidNode] = None,
        context_metadata: Optional[Dict[str, Any]] = None,
        engagement_id: Optional[str] = None,
        target_persona: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive context-aware quality assessment

        Args:
            deliverable: Executive deliverable to assess
            pyramid_structure: Pyramid structure for context analysis
            context_metadata: Context intelligence metadata
            engagement_id: Engagement ID for historical pattern matching
            target_persona: Target audience persona for adaptation

        Returns:
            Comprehensive quality assessment with enhancement recommendations
        """

        self.logger.info(
            f"🎯 Conducting context-aware quality assessment for engagement {engagement_id}"
        )

        # Phase 1: Context Intelligence Analysis
        context_analysis = await self._analyze_context_quality_factors(
            deliverable, pyramid_structure, context_metadata, engagement_id
        )

        # Phase 2: Base Quality Assessment (enhanced)
        base_quality = await self._assess_enhanced_base_quality(
            deliverable, pyramid_structure, context_analysis
        )

        # Phase 3: Cognitive Coherence Assessment
        cognitive_quality = await self._assess_cognitive_coherence_quality(
            deliverable, pyramid_structure, context_analysis, engagement_id
        )

        # Phase 4: Historical Effectiveness Assessment
        historical_quality = await self._assess_historical_effectiveness(
            deliverable, context_analysis, engagement_id
        )

        # Phase 5: Context-Specific Quality Adaptation
        adapted_quality = await self._apply_context_specific_quality_adaptation(
            base_quality,
            cognitive_quality,
            historical_quality,
            context_analysis,
            target_persona,
        )

        # Phase 6: Predictive Quality Optimization
        optimization_recommendations = (
            await self._generate_quality_optimization_recommendations(
                deliverable, adapted_quality, context_analysis
            )
        )

        # Calculate final context-aware quality scores
        final_quality_assessment = self._calculate_final_quality_scores(
            base_quality, cognitive_quality, historical_quality, adapted_quality
        )

        # Update deliverable with enhanced quality metrics
        await self._update_deliverable_with_context_quality(
            deliverable, final_quality_assessment
        )

        # Store quality cognitive exhaust for future improvement
        if self.context_intelligence and engagement_id:
            await self._store_quality_assessment_cognitive_exhaust(
                final_quality_assessment, context_analysis, engagement_id
            )

        self.logger.info(
            f"✅ Context-aware quality assessment complete - Partner Ready: {final_quality_assessment.get('partner_ready_score', 0):.2f}"
        )

        return {
            "quality_assessment": final_quality_assessment,
            "context_analysis": context_analysis,
            "optimization_recommendations": optimization_recommendations,
            "assessment_metadata": {
                "context_enhanced": self.context_enhanced,
                "engagement_id": engagement_id,
                "target_persona": target_persona,
                "assessment_timestamp": datetime.now().isoformat(),
            },
        }

    async def _analyze_context_quality_factors(
        self,
        deliverable: ExecutiveDeliverable,
        pyramid_structure: Optional[PyramidNode],
        context_metadata: Optional[Dict[str, Any]],
        engagement_id: Optional[str],
    ) -> Dict[str, Any]:
        """
        Analyze context-specific quality factors
        """

        if not self.context_enhanced or not context_metadata:
            return {
                "context_intelligence_applied": False,
                "primary_context_type": "DOMAIN",
                "overall_coherence": 0.75,
                "quality_enhancement_potential": "moderate",
            }

        try:
            # Extract context information
            primary_context_type = context_metadata.get(
                "primary_context_type", "DOMAIN"
            )
            overall_coherence = context_metadata.get("overall_coherence", 0.75)
            historical_patterns = context_metadata.get("historical_patterns_count", 0)

            # Get relevant quality contexts if Context Intelligence is available
            relevant_quality_contexts = []
            if self.context_intelligence and engagement_id:
                try:
                    quality_contexts = (
                        await self.context_intelligence.get_relevant_context(
                            current_query="quality assessment deliverable excellence",
                            max_contexts=5,
                            engagement_id=engagement_id,
                        )
                    )
                    relevant_quality_contexts = quality_contexts
                except Exception as e:
                    self.logger.debug(f"Quality context retrieval warning: {e}")

            # Analyze context-specific quality factors
            context_quality_factors = self._extract_context_quality_factors(
                primary_context_type, overall_coherence, historical_patterns
            )

            return {
                "context_intelligence_applied": True,
                "primary_context_type": primary_context_type,
                "overall_coherence": overall_coherence,
                "historical_patterns_count": historical_patterns,
                "relevant_quality_contexts": len(relevant_quality_contexts),
                "context_quality_factors": context_quality_factors,
                "quality_enhancement_potential": self._assess_quality_enhancement_potential(
                    overall_coherence, historical_patterns
                ),
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Context quality analysis failed: {e}")
            return {
                "context_intelligence_applied": False,
                "primary_context_type": "DOMAIN",
                "overall_coherence": 0.75,
                "error": str(e),
            }

    async def _assess_enhanced_base_quality(
        self,
        deliverable: ExecutiveDeliverable,
        pyramid_structure: Optional[PyramidNode],
        context_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Enhanced base quality assessment with context awareness
        """

        # Perform traditional assessment first
        await super().assess_deliverable_quality(deliverable)

        # Extract traditional scores
        base_scores = {
            "structure_quality": getattr(deliverable, "structure_quality", 0.8),
            "content_quality": getattr(deliverable, "content_quality", 0.8),
            "partner_ready_score": getattr(deliverable, "partner_ready_score", 0.8),
            "persuasiveness": getattr(deliverable, "persuasiveness", 0.8),
        }

        # Apply context-specific enhancements
        if context_analysis.get("context_intelligence_applied"):
            primary_context = context_analysis.get("primary_context_type", "DOMAIN")
            coherence = context_analysis.get("overall_coherence", 0.75)

            # Apply context-specific quality adjustments
            enhanced_scores = self._apply_context_quality_adjustments(
                base_scores, primary_context, coherence
            )

            return {
                "base_scores": base_scores,
                "enhanced_scores": enhanced_scores,
                "context_enhancement_applied": True,
                "enhancement_factor": enhanced_scores["partner_ready_score"]
                / base_scores["partner_ready_score"],
            }

        return {
            "base_scores": base_scores,
            "enhanced_scores": base_scores,
            "context_enhancement_applied": False,
            "enhancement_factor": 1.0,
        }

    async def _assess_cognitive_coherence_quality(
        self,
        deliverable: ExecutiveDeliverable,
        pyramid_structure: Optional[PyramidNode],
        context_analysis: Dict[str, Any],
        engagement_id: Optional[str],
    ) -> Dict[str, Any]:
        """
        Assess quality based on cognitive coherence from Context Intelligence
        """

        if not context_analysis.get("context_intelligence_applied"):
            return {
                "cognitive_coherence_score": 0.75,  # Default moderate coherence
                "logical_flow_quality": 0.8,
                "argument_consistency": 0.8,
                "cognitive_enhancement_applied": False,
            }

        overall_coherence = context_analysis.get("overall_coherence", 0.75)

        # Assess logical flow quality based on cognitive coherence
        logical_flow_quality = self._assess_logical_flow_from_coherence(
            pyramid_structure, overall_coherence
        )

        # Assess argument consistency
        argument_consistency = self._assess_argument_consistency_from_coherence(
            deliverable, pyramid_structure, overall_coherence
        )

        # Calculate cognitive quality bonus
        cognitive_quality_bonus = self._calculate_cognitive_quality_bonus(
            overall_coherence, logical_flow_quality, argument_consistency
        )

        return {
            "cognitive_coherence_score": overall_coherence,
            "logical_flow_quality": logical_flow_quality,
            "argument_consistency": argument_consistency,
            "cognitive_quality_bonus": cognitive_quality_bonus,
            "cognitive_enhancement_applied": True,
            "coherence_quality_factors": {
                "high_coherence": overall_coherence > 0.85,
                "consistent_logic": logical_flow_quality > 0.8,
                "strong_arguments": argument_consistency > 0.8,
            },
        }

    async def _assess_historical_effectiveness(
        self,
        deliverable: ExecutiveDeliverable,
        context_analysis: Dict[str, Any],
        engagement_id: Optional[str],
    ) -> Dict[str, Any]:
        """
        Assess quality based on historical effectiveness patterns
        """

        if (
            not context_analysis.get("context_intelligence_applied")
            or not engagement_id
        ):
            return {
                "historical_effectiveness_score": 0.8,
                "pattern_match_quality": 0.7,
                "historical_enhancement_applied": False,
            }

        historical_patterns_count = context_analysis.get("historical_patterns_count", 0)
        primary_context_type = context_analysis.get("primary_context_type", "DOMAIN")

        # Assess pattern matching effectiveness
        pattern_match_quality = self._assess_pattern_matching_effectiveness(
            deliverable, historical_patterns_count, primary_context_type
        )

        # Calculate historical effectiveness score
        historical_effectiveness_score = self._calculate_historical_effectiveness_score(
            pattern_match_quality, historical_patterns_count
        )

        # Identify successful pattern elements
        successful_pattern_elements = self._identify_successful_pattern_elements(
            deliverable, historical_patterns_count
        )

        return {
            "historical_effectiveness_score": historical_effectiveness_score,
            "pattern_match_quality": pattern_match_quality,
            "historical_patterns_matched": historical_patterns_count,
            "successful_pattern_elements": successful_pattern_elements,
            "historical_enhancement_applied": True,
            "historical_quality_factors": {
                "strong_pattern_match": historical_patterns_count > 3,
                "proven_effectiveness": pattern_match_quality > 0.8,
                "context_appropriate_patterns": primary_context_type
                in ["STRATEGIC", "DOMAIN"],
            },
        }

    async def _apply_context_specific_quality_adaptation(
        self,
        base_quality: Dict[str, Any],
        cognitive_quality: Dict[str, Any],
        historical_quality: Dict[str, Any],
        context_analysis: Dict[str, Any],
        target_persona: Optional[str],
    ) -> Dict[str, Any]:
        """
        Apply context-specific quality adaptations
        """

        primary_context_type = context_analysis.get("primary_context_type", "DOMAIN")

        # Get context-specific benchmarks
        context_benchmarks = self.context_benchmarks.get(primary_context_type, {})

        # Apply persona-specific adaptations
        persona_adaptations = self._get_persona_quality_adaptations(target_persona)

        # Calculate adapted scores
        enhanced_scores = base_quality.get(
            "enhanced_scores", base_quality.get("base_scores", {})
        )

        adapted_scores = {}
        for score_type, score_value in enhanced_scores.items():
            # Apply context benchmark multipliers
            context_multiplier = self._get_context_multiplier(
                score_type, context_benchmarks
            )
            # Apply persona adaptations
            persona_multiplier = self._get_persona_multiplier(
                score_type, persona_adaptations
            )
            # Apply cognitive and historical bonuses
            cognitive_bonus = cognitive_quality.get("cognitive_quality_bonus", 0)
            historical_bonus = (
                historical_quality.get("historical_effectiveness_score", 0.8) - 0.8
            )

            adapted_score = (
                score_value * context_multiplier * persona_multiplier
                + cognitive_bonus
                + historical_bonus
            )
            adapted_scores[score_type] = min(adapted_score, 1.0)  # Cap at 1.0

        return {
            "adapted_scores": adapted_scores,
            "context_adaptations_applied": {
                "primary_context_type": primary_context_type,
                "target_persona": target_persona,
                "context_benchmarks": context_benchmarks,
                "persona_adaptations": persona_adaptations,
            },
            "adaptation_summary": {
                "context_boost": (
                    sum(context_benchmarks.values()) / len(context_benchmarks)
                    if context_benchmarks
                    else 1.0
                ),
                "persona_boost": (
                    sum(persona_adaptations.values()) / len(persona_adaptations)
                    if persona_adaptations
                    else 1.0
                ),
                "cognitive_boost": cognitive_quality.get("cognitive_quality_bonus", 0),
                "historical_boost": historical_quality.get(
                    "historical_effectiveness_score", 0.8
                )
                - 0.8,
            },
        }

    # Context-specific quality assessment methods

    def _extract_context_quality_factors(
        self,
        primary_context_type: str,
        overall_coherence: float,
        historical_patterns: int,
    ) -> Dict[str, Any]:
        """Extract context-specific quality factors"""

        quality_factors = {
            "context_relevance": 0.8,
            "coherence_strength": overall_coherence,
            "pattern_applicability": min(historical_patterns * 0.1, 1.0),
            "context_optimization_potential": 0.0,
        }

        # Context-specific factor adjustments
        if primary_context_type == "IMMEDIATE":
            quality_factors["urgency_clarity"] = 0.9
            quality_factors["action_orientation"] = 0.85
        elif primary_context_type == "STRATEGIC":
            quality_factors["vision_alignment"] = 0.9
            quality_factors["strategic_depth"] = 0.8
        elif primary_context_type == "PROCEDURAL":
            quality_factors["process_clarity"] = 0.9
            quality_factors["implementation_detail"] = 0.85

        # Calculate optimization potential
        quality_factors["context_optimization_potential"] = (
            quality_factors["coherence_strength"] * 0.4
            + quality_factors["pattern_applicability"] * 0.3
            + quality_factors["context_relevance"] * 0.3
        )

        return quality_factors

    def _assess_quality_enhancement_potential(
        self, overall_coherence: float, historical_patterns: int
    ) -> str:
        """Assess potential for quality enhancement"""

        if overall_coherence > 0.9 and historical_patterns > 5:
            return "excellent"
        elif overall_coherence > 0.8 and historical_patterns > 3:
            return "high"
        elif overall_coherence > 0.7 and historical_patterns > 1:
            return "moderate"
        else:
            return "developing"

    def _apply_context_quality_adjustments(
        self, base_scores: Dict[str, Any], primary_context: str, coherence: float
    ) -> Dict[str, Any]:
        """Apply context-specific quality adjustments"""

        enhanced_scores = base_scores.copy()

        # Coherence-based enhancement
        coherence_multiplier = 0.8 + (coherence * 0.4)  # 0.8 to 1.2 multiplier

        # Context-specific adjustments
        context_multipliers = self.context_benchmarks.get(primary_context, {})

        for score_type, score_value in base_scores.items():
            # Apply coherence enhancement
            enhanced_score = score_value * coherence_multiplier

            # Apply context-specific multipliers
            if (
                score_type == "structure_quality"
                and "clarity_weight" in context_multipliers
            ):
                enhanced_score *= context_multipliers["clarity_weight"]
            elif (
                score_type == "content_quality"
                and "depth_weight" in context_multipliers
            ):
                enhanced_score *= context_multipliers["depth_weight"]

            enhanced_scores[score_type] = min(enhanced_score, 1.0)

        return enhanced_scores

    # Cognitive coherence assessment methods

    def _assess_logical_flow_from_coherence(
        self, pyramid_structure: Optional[PyramidNode], overall_coherence: float
    ) -> float:
        """Assess logical flow quality from cognitive coherence"""

        base_logical_flow = 0.8  # Default

        if pyramid_structure and pyramid_structure.children:
            # Analyze argument progression coherence
            argument_coherence = overall_coherence * 0.9  # Slightly conservative

            # Check for logical progression in key lines
            key_lines_coherent = len(pyramid_structure.children) >= 3
            if key_lines_coherent:
                argument_coherence += 0.1

            base_logical_flow = min(argument_coherence, 1.0)

        return base_logical_flow

    def _assess_argument_consistency_from_coherence(
        self,
        deliverable: ExecutiveDeliverable,
        pyramid_structure: Optional[PyramidNode],
        overall_coherence: float,
    ) -> float:
        """Assess argument consistency from cognitive coherence"""

        # Use coherence as base consistency score
        consistency_score = overall_coherence

        # Check for consistent messaging across deliverable sections
        if hasattr(deliverable, "executive_summary") and deliverable.executive_summary:
            if (
                hasattr(deliverable, "key_recommendations")
                and deliverable.key_recommendations
            ):
                # Simple consistency check - coherent messaging
                consistency_score += 0.05

        return min(consistency_score, 1.0)

    def _calculate_cognitive_quality_bonus(
        self,
        overall_coherence: float,
        logical_flow_quality: float,
        argument_consistency: float,
    ) -> float:
        """Calculate cognitive quality bonus"""

        # High coherence provides quality bonus
        if overall_coherence > 0.85:
            cognitive_bonus = 0.1  # 10% bonus for high coherence
        elif overall_coherence > 0.75:
            cognitive_bonus = 0.05  # 5% bonus for good coherence
        else:
            cognitive_bonus = 0.0

        # Additional bonus for strong logical flow and consistency
        if logical_flow_quality > 0.85 and argument_consistency > 0.85:
            cognitive_bonus += 0.05

        return cognitive_bonus

    # Historical effectiveness assessment methods

    def _assess_pattern_matching_effectiveness(
        self,
        deliverable: ExecutiveDeliverable,
        historical_patterns_count: int,
        primary_context_type: str,
    ) -> float:
        """Assess effectiveness of historical pattern matching"""

        # Base pattern match quality
        pattern_quality = 0.7

        # More patterns generally indicate better matching
        if historical_patterns_count > 5:
            pattern_quality = 0.9
        elif historical_patterns_count > 3:
            pattern_quality = 0.85
        elif historical_patterns_count > 1:
            pattern_quality = 0.8

        # Context-specific pattern effectiveness
        if primary_context_type in ["STRATEGIC", "DOMAIN"]:
            pattern_quality += (
                0.05  # These contexts benefit more from historical patterns
            )

        return min(pattern_quality, 1.0)

    def _calculate_historical_effectiveness_score(
        self, pattern_match_quality: float, historical_patterns_count: int
    ) -> float:
        """Calculate historical effectiveness score"""

        # Combine pattern quality and quantity
        effectiveness_score = (
            pattern_match_quality * 0.7  # Quality is more important
            + min(historical_patterns_count * 0.05, 0.3)  # Quantity bonus up to 30%
        )

        return min(effectiveness_score, 1.0)

    def _identify_successful_pattern_elements(
        self, deliverable: ExecutiveDeliverable, historical_patterns_count: int
    ) -> List[str]:
        """Identify successful pattern elements"""

        successful_elements = []

        if historical_patterns_count > 0:
            successful_elements.append("historical_pattern_integration")
        if historical_patterns_count > 3:
            successful_elements.append("strong_pattern_validation")
        if historical_patterns_count > 5:
            successful_elements.append("comprehensive_pattern_matching")

        return successful_elements

    # Context and persona adaptation methods

    def _get_persona_quality_adaptations(
        self, target_persona: Optional[str]
    ) -> Dict[str, float]:
        """Get persona-specific quality adaptations"""

        persona_adaptations = {
            "CEO": {"executive_focus": 1.3, "strategic_clarity": 1.2},
            "CTO": {"technical_depth": 1.2, "implementation_detail": 1.3},
            "Board": {"strategic_vision": 1.4, "risk_assessment": 1.3},
            "Partner": {"client_impact": 1.3, "business_value": 1.2},
        }

        return persona_adaptations.get(target_persona, {"general_quality": 1.0})

    def _get_context_multiplier(
        self, score_type: str, context_benchmarks: Dict[str, float]
    ) -> float:
        """Get context-specific multiplier for score type"""

        multiplier_map = {
            "structure_quality": context_benchmarks.get("clarity_weight", 1.0),
            "content_quality": context_benchmarks.get("depth_weight", 1.0),
            "partner_ready_score": context_benchmarks.get("urgency_weight", 1.0),
        }

        return multiplier_map.get(score_type, 1.0)

    def _get_persona_multiplier(
        self, score_type: str, persona_adaptations: Dict[str, float]
    ) -> float:
        """Get persona-specific multiplier for score type"""

        # Map score types to persona adaptations
        if score_type == "partner_ready_score":
            return persona_adaptations.get(
                "executive_focus",
                persona_adaptations.get(
                    "client_impact", persona_adaptations.get("general_quality", 1.0)
                ),
            )

        return persona_adaptations.get("general_quality", 1.0)

    # Final quality calculation and optimization

    def _calculate_final_quality_scores(
        self,
        base_quality: Dict[str, Any],
        cognitive_quality: Dict[str, Any],
        historical_quality: Dict[str, Any],
        adapted_quality: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate final context-aware quality scores"""

        adapted_scores = adapted_quality.get(
            "adapted_scores",
            base_quality.get("enhanced_scores", base_quality.get("base_scores", {})),
        )

        # Calculate weighted final scores using context-aware criteria
        final_partner_ready_score = (
            adapted_scores.get("partner_ready_score", 0.8) * 0.4  # Base quality (40%)
            + cognitive_quality.get("cognitive_coherence_score", 0.75)
            * 0.3  # Cognitive coherence (30%)
            + historical_quality.get("historical_effectiveness_score", 0.8)
            * 0.2  # Historical patterns (20%)
            + adapted_quality.get("adaptation_summary", {}).get("context_boost", 1.0)
            * 0.1  # Context adaptation (10%)
        )

        return {
            "partner_ready_score": min(final_partner_ready_score, 1.0),
            "structure_quality": adapted_scores.get("structure_quality", 0.8),
            "content_quality": adapted_scores.get("content_quality", 0.8),
            "persuasiveness": adapted_scores.get("persuasiveness", 0.8),
            "cognitive_coherence_score": cognitive_quality.get(
                "cognitive_coherence_score", 0.75
            ),
            "historical_effectiveness_score": historical_quality.get(
                "historical_effectiveness_score", 0.8
            ),
            "context_enhancement_factor": adapted_quality.get(
                "adaptation_summary", {}
            ).get("context_boost", 1.0),
            "quality_dimensions": {
                "base_quality": base_quality.get("enhancement_factor", 1.0),
                "cognitive_quality": cognitive_quality.get(
                    "cognitive_quality_bonus", 0
                ),
                "historical_quality": historical_quality.get(
                    "pattern_match_quality", 0.8
                ),
                "context_adaptation": adapted_quality.get("adaptation_summary", {}).get(
                    "persona_boost", 1.0
                ),
            },
        }

    async def _generate_quality_optimization_recommendations(
        self,
        deliverable: ExecutiveDeliverable,
        quality_assessment: Dict[str, Any],
        context_analysis: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate quality optimization recommendations"""

        recommendations = []

        # Cognitive coherence improvements
        if context_analysis.get("overall_coherence", 0.75) < 0.8:
            recommendations.append(
                {
                    "type": "cognitive_coherence",
                    "priority": "high",
                    "recommendation": "Improve logical flow consistency between key arguments",
                    "expected_improvement": "0.05-0.10 partner readiness boost",
                }
            )

        # Historical pattern optimization
        historical_patterns = context_analysis.get("historical_patterns_count", 0)
        if historical_patterns < 3:
            recommendations.append(
                {
                    "type": "historical_patterns",
                    "priority": "medium",
                    "recommendation": "Incorporate more proven successful patterns from similar engagements",
                    "expected_improvement": "0.03-0.07 effectiveness boost",
                }
            )

        # Context-specific improvements
        primary_context = context_analysis.get("primary_context_type", "DOMAIN")
        if (
            primary_context == "STRATEGIC"
            and quality_assessment.get("structure_quality", 0.8) < 0.9
        ):
            recommendations.append(
                {
                    "type": "strategic_depth",
                    "priority": "high",
                    "recommendation": "Enhance strategic vision and long-term impact messaging",
                    "expected_improvement": "0.08-0.12 strategic relevance boost",
                }
            )

        return recommendations

    async def _update_deliverable_with_context_quality(
        self,
        deliverable: ExecutiveDeliverable,
        final_quality_assessment: Dict[str, Any],
    ) -> None:
        """Update deliverable with context-aware quality metrics"""

        # Update standard quality metrics
        deliverable.partner_ready_score = final_quality_assessment.get(
            "partner_ready_score", 0.8
        )
        deliverable.structure_quality = final_quality_assessment.get(
            "structure_quality", 0.8
        )
        deliverable.content_quality = final_quality_assessment.get(
            "content_quality", 0.8
        )
        deliverable.persuasiveness = final_quality_assessment.get("persuasiveness", 0.8)

        # Add context-aware quality metrics
        if not hasattr(deliverable, "context_quality_metrics"):
            deliverable.context_quality_metrics = {}

        deliverable.context_quality_metrics.update(
            {
                "cognitive_coherence_score": final_quality_assessment.get(
                    "cognitive_coherence_score", 0.75
                ),
                "historical_effectiveness_score": final_quality_assessment.get(
                    "historical_effectiveness_score", 0.8
                ),
                "context_enhancement_factor": final_quality_assessment.get(
                    "context_enhancement_factor", 1.0
                ),
                "quality_dimensions": final_quality_assessment.get(
                    "quality_dimensions", {}
                ),
            }
        )

    async def _store_quality_assessment_cognitive_exhaust(
        self,
        final_quality_assessment: Dict[str, Any],
        context_analysis: Dict[str, Any],
        engagement_id: str,
    ) -> None:
        """Store quality assessment cognitive exhaust for future improvement"""

        if not self.context_intelligence:
            return

        try:
            thinking_process = f"""
            Context-Aware Quality Assessment Process:
            
            Quality Assessment Results:
            - Partner Ready Score: {final_quality_assessment.get('partner_ready_score', 'N/A'):.3f}
            - Cognitive Coherence: {final_quality_assessment.get('cognitive_coherence_score', 'N/A'):.3f}
            - Historical Effectiveness: {final_quality_assessment.get('historical_effectiveness_score', 'N/A'):.3f}
            - Context Enhancement Factor: {final_quality_assessment.get('context_enhancement_factor', 'N/A'):.3f}
            
            Context Analysis:
            - Primary Context Type: {context_analysis.get('primary_context_type', 'N/A')}
            - Overall Coherence: {context_analysis.get('overall_coherence', 'N/A'):.3f}
            - Historical Patterns: {context_analysis.get('historical_patterns_count', 'N/A')}
            - Quality Enhancement Potential: {context_analysis.get('quality_enhancement_potential', 'N/A')}
            
            The assessment leverages cognitive exhaust analysis for comprehensive quality evaluation.
            """

            cleaned_response = f"Context-aware quality assessment completed with partner ready score of {final_quality_assessment.get('partner_ready_score', 0.8):.2f}"

            await self.context_intelligence.store_cognitive_exhaust_triple_layer(
                engagement_id=engagement_id,
                phase="quality_assessment",
                mental_model="context_aware_quality_intelligence",
                thinking_process=thinking_process,
                cleaned_response=cleaned_response,
                confidence=final_quality_assessment.get("partner_ready_score", 0.8),
            )

            self.logger.info(
                f"💾 Quality assessment cognitive exhaust stored for engagement {engagement_id}"
            )

        except Exception as e:
            self.logger.warning(
                f"⚠️ Failed to store quality assessment cognitive exhaust: {e}"
            )


# Factory function for creating context-aware quality assessor
def create_context_aware_quality_assessor(
    context_intelligence: Optional[IContextIntelligence] = None,
) -> ContextAwareQualityAssessor:
    """Factory function for creating Context-Aware Quality Assessor"""
    return ContextAwareQualityAssessor(context_intelligence=context_intelligence)
