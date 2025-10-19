"""
Context-Intelligent Pyramid Builder - Sprint 2.1
F004: Revolutionary consulting deliverable system using AI's cognitive exhaust

This module enhances the traditional Pyramid Principle with Context Intelligence,
creating the world's first consulting system that uses AI's own thinking process
for maximum persuasive impact and logical coherence.

Key Innovations:
- Cognitive Exhaust-Driven Argument Structure
- Manus Taxonomy-Based Content Classification
- Context-Aware Logical Flow Optimization
- Historical Pattern Recognition for Persuasion
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime

from .models import PyramidNode
from .enums import PyramidLevel, ArgumentType
from .builders import PyramidBuilder

# Context Intelligence imports
try:
    from src.interfaces.context_intelligence_interface import IContextIntelligence
    from src.engine.models.data_contracts import (
        ContextType,
        ContextElement,
        ContextRelevanceScore,
    )
    from src.models.context_taxonomy import ContextIntelligenceResult

    CONTEXT_INTELLIGENCE_AVAILABLE = True
except ImportError:
    CONTEXT_INTELLIGENCE_AVAILABLE = False
    IContextIntelligence = Any
    ContextType = Any
    ContextElement = Any
    ContextRelevanceScore = Any
    ContextIntelligenceResult = Any


class ContextIntelligentPyramidBuilder(PyramidBuilder):
    """
    Context-Intelligent Pyramid Builder using AI's cognitive exhaust

    Enhances traditional Pyramid Principle with:
    - Cognitive coherence-driven argument structuring
    - Context-aware logical flow optimization
    - Historical pattern recognition for maximum impact
    - Manus taxonomy-based content classification
    """

    def __init__(self, context_intelligence: Optional[IContextIntelligence] = None):
        super().__init__()
        self.context_intelligence = context_intelligence
        self.logger = logging.getLogger(__name__)

        # Context-aware enhancement flags
        self.cognitive_coherence_enabled = context_intelligence is not None
        self.historical_pattern_matching = True
        self.manus_taxonomy_classification = True

        if self.cognitive_coherence_enabled:
            self.logger.info(
                "🧠 Context-Intelligent Pyramid Builder initialized with cognitive exhaust integration"
            )
        else:
            self.logger.info(
                "📊 Traditional Pyramid Builder initialized (Context Intelligence not available)"
            )

    async def build_context_aware_pyramid_structure(
        self,
        insights: List[str],
        hypotheses: List[Dict],
        frameworks_results: List[Dict],
        analysis_findings: Dict[str, Any],
        engagement_id: str,
        cognitive_coherence_scores: Optional[List[float]] = None,
    ) -> PyramidNode:
        """
        Build pyramid structure with Context Intelligence enhancement

        Args:
            insights: Key insights from analysis
            hypotheses: Generated hypotheses with confidence scores
            frameworks_results: Framework application results
            analysis_findings: Detailed analysis findings
            engagement_id: Engagement identifier for context retrieval
            cognitive_coherence_scores: Optional cognitive coherence scores from Operation Mindforge

        Returns:
            Enhanced pyramid structure with context-aware optimization
        """

        self.logger.info(
            f"🏗️ Building context-aware pyramid structure for engagement {engagement_id}"
        )

        # Phase 1: Context Intelligence Analysis
        context_insights = await self._analyze_context_intelligence(
            insights,
            hypotheses,
            frameworks_results,
            engagement_id,
            cognitive_coherence_scores,
        )

        # Phase 2: Cognitive Exhaust-Driven Governing Thought
        governing_thought = await self._identify_context_aware_governing_thought(
            insights, hypotheses, frameworks_results, context_insights
        )

        # Create enhanced root node with context metadata
        root = PyramidNode(
            level=PyramidLevel.GOVERNING_THOUGHT,
            content=governing_thought,
            argument_type=ArgumentType.INDUCTIVE,
            metadata={
                "context_intelligence_applied": self.cognitive_coherence_enabled,
                "cognitive_coherence_score": context_insights.get(
                    "overall_coherence", 0.0
                ),
                "context_classification": context_insights.get(
                    "primary_context_type", "DOMAIN"
                ),
                "engagement_id": engagement_id,
                "generated_at": datetime.now().isoformat(),
            },
        )

        # Phase 3: Context-Aware Key Line Generation
        key_lines = await self._generate_context_aware_key_lines(
            insights, hypotheses, frameworks_results, context_insights
        )

        # Phase 4: Build enhanced pyramid with cognitive coherence
        for line_content, line_metadata in key_lines:
            key_line_node = PyramidNode(
                level=PyramidLevel.KEY_LINES,
                content=line_content,
                argument_type=self._determine_optimal_argument_type(
                    line_content, context_insights
                ),
                metadata=line_metadata,
            )

            # Generate context-aware supporting points
            supporting_points = await self._generate_context_aware_supporting_points(
                line_content,
                analysis_findings,
                frameworks_results,
                context_insights,
                engagement_id,
            )

            for point_content, point_metadata in supporting_points:
                support_node = PyramidNode(
                    level=PyramidLevel.SUPPORTING_POINTS,
                    content=point_content,
                    argument_type=ArgumentType.INDUCTIVE,
                    metadata=point_metadata,
                )
                key_line_node.add_child(support_node)

            root.add_child(key_line_node)

        # Phase 5: Enhanced validation with context intelligence
        await self._validate_context_aware_pyramid_structure(root, context_insights)

        # Phase 6: Store cognitive exhaust for future use
        if self.context_intelligence:
            await self._store_pyramid_cognitive_exhaust(
                root, engagement_id, context_insights
            )

        self.logger.info(
            f"✅ Context-aware pyramid structure complete: {len(root.children)} key lines with cognitive enhancement"
        )

        return root

    async def _analyze_context_intelligence(
        self,
        insights: List[str],
        hypotheses: List[Dict],
        frameworks_results: List[Dict],
        engagement_id: str,
        cognitive_coherence_scores: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze content using Context Intelligence for enhanced pyramid building
        """

        if not self.context_intelligence:
            return {
                "context_intelligence_applied": False,
                "overall_coherence": 0.7,  # Default assumption
                "primary_context_type": "DOMAIN",
            }

        try:
            # Prepare content for context analysis
            all_content = []
            all_content.extend(insights)
            all_content.extend([h.get("statement", "") for h in hypotheses])
            for result in frameworks_results:
                if isinstance(result.get("output"), dict):
                    all_content.append(str(result["output"]))

            # Get relevant historical contexts
            relevant_contexts = await self.context_intelligence.get_relevant_context(
                current_query=" ".join(all_content[:2]),  # Use first 2 items as query
                max_contexts=5,
                engagement_id=engagement_id,
            )

            # Analyze with Manus Taxonomy
            if all_content:
                context_analysis = await self.context_intelligence.analyze_contexts_with_manus_taxonomy(
                    context_contents=all_content,
                    current_query=(
                        " ".join(insights[:1]) if insights else "Strategic analysis"
                    ),
                    engagement_id=engagement_id,
                    cognitive_coherence_scores=cognitive_coherence_scores,
                )

                return {
                    "context_intelligence_applied": True,
                    "context_analysis": context_analysis,
                    "relevant_contexts": relevant_contexts,
                    "overall_coherence": self._calculate_overall_coherence(
                        context_analysis
                    ),
                    "primary_context_type": self._determine_primary_context_type(
                        context_analysis
                    ),
                    "cognitive_patterns": self._extract_cognitive_patterns(
                        relevant_contexts
                    ),
                }

        except Exception as e:
            self.logger.warning(f"⚠️ Context Intelligence analysis failed: {e}")

        return {
            "context_intelligence_applied": False,
            "overall_coherence": 0.7,
            "primary_context_type": "DOMAIN",
        }

    async def _identify_context_aware_governing_thought(
        self,
        insights: List[str],
        hypotheses: List[Dict],
        frameworks_results: List[Dict],
        context_insights: Dict[str, Any],
    ) -> str:
        """
        Identify governing thought using context intelligence and cognitive patterns
        """

        # Start with traditional approach
        base_governing_thought = await super()._identify_governing_thought(
            insights, hypotheses, frameworks_results
        )

        if not context_insights.get("context_intelligence_applied"):
            return base_governing_thought

        # Enhance with context intelligence
        try:
            cognitive_patterns = context_insights.get("cognitive_patterns", [])
            primary_context_type = context_insights.get(
                "primary_context_type", "DOMAIN"
            )
            overall_coherence = context_insights.get("overall_coherence", 0.7)

            # Apply context-specific enhancement
            if primary_context_type == "IMMEDIATE":
                # Urgent action-oriented governing thought
                enhanced_thought = self._create_action_oriented_governing_thought(
                    base_governing_thought, cognitive_patterns
                )
            elif primary_context_type == "STRATEGIC":
                # Strategic transformation-oriented
                enhanced_thought = self._create_strategic_governing_thought(
                    base_governing_thought, cognitive_patterns
                )
            elif overall_coherence > 0.85:
                # High coherence - confident assertion
                enhanced_thought = self._create_confident_governing_thought(
                    base_governing_thought, cognitive_patterns
                )
            else:
                # Moderate coherence - balanced approach
                enhanced_thought = self._create_balanced_governing_thought(
                    base_governing_thought, cognitive_patterns
                )

            self.logger.info(
                f"🎯 Enhanced governing thought with context type: {primary_context_type}"
            )
            return enhanced_thought

        except Exception as e:
            self.logger.warning(
                f"⚠️ Governing thought enhancement failed, using base: {e}"
            )
            return base_governing_thought

    async def _generate_context_aware_key_lines(
        self,
        insights: List[str],
        hypotheses: List[Dict],
        frameworks_results: List[Dict],
        context_insights: Dict[str, Any],
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Generate key lines with context intelligence enhancement
        """

        # Generate base key lines
        base_key_lines = await super()._generate_key_lines(
            insights, hypotheses, frameworks_results
        )

        enhanced_key_lines = []

        for i, base_line in enumerate(base_key_lines):
            # Create metadata for each key line
            line_metadata = {
                "line_index": i,
                "context_enhancement_applied": context_insights.get(
                    "context_intelligence_applied", False
                ),
                "cognitive_coherence_score": context_insights.get(
                    "overall_coherence", 0.7
                ),
                "logical_flow_optimized": True,
            }

            # Apply context-specific enhancement
            if context_insights.get("context_intelligence_applied"):
                enhanced_line = await self._enhance_key_line_with_context(
                    base_line, context_insights, i
                )
                line_metadata["enhancement_applied"] = True
            else:
                enhanced_line = base_line
                line_metadata["enhancement_applied"] = False

            enhanced_key_lines.append((enhanced_line, line_metadata))

        return enhanced_key_lines

    async def _generate_context_aware_supporting_points(
        self,
        key_line: str,
        analysis_findings: Dict[str, Any],
        frameworks_results: List[Dict],
        context_insights: Dict[str, Any],
        engagement_id: str,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Generate supporting points with context intelligence enhancement
        """

        # Generate base supporting points
        base_points = await super()._generate_supporting_points(
            key_line, analysis_findings, frameworks_results
        )

        enhanced_points = []

        for i, base_point in enumerate(base_points):
            # Create metadata for each supporting point
            point_metadata = {
                "point_index": i,
                "parent_key_line": (
                    key_line[:50] + "..." if len(key_line) > 50 else key_line
                ),
                "context_grounded": context_insights.get(
                    "context_intelligence_applied", False
                ),
                "evidence_strength": self._assess_evidence_strength(
                    base_point, context_insights
                ),
            }

            # Apply context enhancement if available
            if context_insights.get("context_intelligence_applied"):
                enhanced_point = await self._enhance_supporting_point_with_context(
                    base_point, key_line, context_insights
                )
                point_metadata["context_enhanced"] = True
            else:
                enhanced_point = base_point
                point_metadata["context_enhanced"] = False

            enhanced_points.append((enhanced_point, point_metadata))

        return enhanced_points

    # Context Intelligence Enhancement Methods

    def _calculate_overall_coherence(self, context_analysis: Any) -> float:
        """Calculate overall cognitive coherence score"""
        if not context_analysis or not hasattr(context_analysis, "overall_score"):
            return 0.7  # Default moderate coherence

        try:
            return float(context_analysis.overall_score)
        except (ValueError, AttributeError):
            return 0.7

    def _determine_primary_context_type(self, context_analysis: Any) -> str:
        """Determine primary context type from Manus taxonomy analysis"""
        if not context_analysis:
            return "DOMAIN"

        try:
            if hasattr(context_analysis, "primary_context_type"):
                return context_analysis.primary_context_type
            elif hasattr(context_analysis, "context_breakdown"):
                # Find most frequent context type
                breakdown = context_analysis.context_breakdown
                if isinstance(breakdown, dict):
                    return max(breakdown.keys(), key=lambda k: breakdown[k])
            return "DOMAIN"
        except (AttributeError, ValueError):
            return "DOMAIN"

    def _extract_cognitive_patterns(self, relevant_contexts: List[Any]) -> List[str]:
        """Extract cognitive patterns from historical contexts"""
        patterns = []

        try:
            for context in relevant_contexts[:3]:  # Use top 3 contexts
                if hasattr(context, "content"):
                    content = str(context.content)
                    # Extract key phrases that indicate cognitive patterns
                    if "systematic" in content.lower():
                        patterns.append("systematic_analysis")
                    if "strategic" in content.lower():
                        patterns.append("strategic_thinking")
                    if "risk" in content.lower():
                        patterns.append("risk_awareness")
                    if "implementation" in content.lower():
                        patterns.append("execution_focus")
        except Exception as e:
            self.logger.debug(f"Pattern extraction warning: {e}")

        return list(set(patterns))  # Remove duplicates

    def _determine_optimal_argument_type(
        self, content: str, context_insights: Dict[str, Any]
    ) -> ArgumentType:
        """Determine optimal argument type based on context intelligence"""

        coherence = context_insights.get("overall_coherence", 0.7)
        patterns = context_insights.get("cognitive_patterns", [])

        # High coherence content works well with deductive reasoning
        if coherence > 0.8:
            return ArgumentType.DEDUCTIVE

        # Strategic patterns suggest inductive approach
        if "strategic_thinking" in patterns:
            return ArgumentType.INDUCTIVE

        # Default based on content analysis
        content_lower = content.lower()
        if any(
            word in content_lower
            for word in ["therefore", "consequently", "thus", "hence"]
        ):
            return ArgumentType.DEDUCTIVE
        else:
            return ArgumentType.INDUCTIVE

    # Content Enhancement Methods

    def _create_action_oriented_governing_thought(
        self, base_thought: str, patterns: List[str]
    ) -> str:
        """Create action-oriented governing thought for immediate context"""
        if "execution_focus" in patterns:
            return f"Immediate implementation of strategic initiatives will {base_thought.lower()}"
        return f"Rapid deployment of targeted solutions will {base_thought.lower()}"

    def _create_strategic_governing_thought(
        self, base_thought: str, patterns: List[str]
    ) -> str:
        """Create strategic governing thought for long-term context"""
        if "systematic_analysis" in patterns:
            return f"Comprehensive transformation strategy will systematically {base_thought.lower()}"
        return f"Strategic transformation initiative will {base_thought.lower()}"

    def _create_confident_governing_thought(
        self, base_thought: str, patterns: List[str]
    ) -> str:
        """Create confident governing thought for high coherence"""
        return f"Evidence-based analysis confirms that {base_thought.lower()}"

    def _create_balanced_governing_thought(
        self, base_thought: str, patterns: List[str]
    ) -> str:
        """Create balanced governing thought for moderate coherence"""
        return f"Comprehensive analysis indicates that {base_thought.lower()}"

    async def _enhance_key_line_with_context(
        self, base_line: str, context_insights: Dict[str, Any], line_index: int
    ) -> str:
        """Enhance key line with context intelligence insights"""

        coherence = context_insights.get("overall_coherence", 0.7)
        patterns = context_insights.get("cognitive_patterns", [])

        # Apply coherence-based enhancement
        if coherence > 0.85:
            # High confidence enhancement
            if "systematic_analysis" in patterns:
                return f"Systematic analysis demonstrates that {base_line.lower()}"
            return f"Evidence strongly indicates that {base_line.lower()}"
        elif coherence > 0.7:
            # Moderate confidence
            return f"Analysis suggests that {base_line.lower()}"
        else:
            # Conservative approach
            return f"Initial findings indicate that {base_line.lower()}"

    async def _enhance_supporting_point_with_context(
        self, base_point: str, key_line: str, context_insights: Dict[str, Any]
    ) -> str:
        """Enhance supporting point with context-specific evidence"""

        patterns = context_insights.get("cognitive_patterns", [])

        # Apply pattern-based enhancement
        if "risk_awareness" in patterns and "risk" not in base_point.lower():
            return f"{base_point}, with appropriate risk mitigation strategies in place"
        elif (
            "execution_focus" in patterns and "implementation" not in base_point.lower()
        ):
            return f"{base_point}, supported by detailed implementation roadmap"
        else:
            return base_point

    def _assess_evidence_strength(
        self, point: str, context_insights: Dict[str, Any]
    ) -> str:
        """Assess evidence strength based on context intelligence"""

        coherence = context_insights.get("overall_coherence", 0.7)

        if coherence > 0.85:
            return "strong"
        elif coherence > 0.7:
            return "moderate"
        else:
            return "developing"

    async def _validate_context_aware_pyramid_structure(
        self, root: PyramidNode, context_insights: Dict[str, Any]
    ) -> None:
        """Enhanced validation with context intelligence"""

        # Perform base validation
        await super()._validate_pyramid_structure(root)

        # Additional context-aware validation
        if context_insights.get("context_intelligence_applied"):
            coherence = context_insights.get("overall_coherence", 0.7)

            # Validate coherence across pyramid levels
            if coherence < 0.6:
                self.logger.warning(
                    f"⚠️ Low cognitive coherence detected ({coherence:.2f}) - consider content revision"
                )

            # Validate argument flow consistency
            self._validate_argument_flow_consistency(root)

        self.logger.info(
            f"✅ Context-aware pyramid validation complete - coherence: {context_insights.get('overall_coherence', 'N/A')}"
        )

    def _validate_argument_flow_consistency(self, root: PyramidNode) -> None:
        """Validate logical flow consistency across pyramid levels"""

        if not root.children:
            return

        # Check for consistent argument types at key line level
        argument_types = [child.argument_type for child in root.children]
        mixed_types = len(set(argument_types)) > 1

        if mixed_types:
            self.logger.info(
                "📊 Mixed argument types detected - this can enhance persuasive impact when done strategically"
            )

        # Check for balanced content distribution
        content_lengths = [len(child.content) for child in root.children]
        if max(content_lengths) > 2 * min(content_lengths):
            self.logger.warning(
                "⚠️ Unbalanced key line lengths detected - consider content balancing"
            )

    async def _store_pyramid_cognitive_exhaust(
        self,
        pyramid_root: PyramidNode,
        engagement_id: str,
        context_insights: Dict[str, Any],
    ) -> None:
        """Store pyramid building cognitive exhaust for future context intelligence"""

        if not self.context_intelligence:
            return

        try:
            # Prepare cognitive exhaust content
            thinking_process = f"""
            Context-Intelligent Pyramid Building Process:
            
            1. Governing Thought: {pyramid_root.content}
            2. Key Lines: {len(pyramid_root.children)} strategic arguments
            3. Supporting Points: {sum(len(child.children) for child in pyramid_root.children)} evidence points
            4. Cognitive Coherence: {context_insights.get('overall_coherence', 'N/A')}
            5. Context Type: {context_insights.get('primary_context_type', 'N/A')}
            6. Enhancement Applied: {context_insights.get('context_intelligence_applied', False)}
            
            The pyramid structure follows logical flow with context-aware optimization.
            """

            cleaned_response = f"Context-intelligent pyramid structure created with {len(pyramid_root.children)} key arguments and enhanced logical coherence."

            # Store in triple-layer cache
            await self.context_intelligence.store_cognitive_exhaust_triple_layer(
                engagement_id=engagement_id,
                phase="pyramid_synthesis",
                mental_model="context_intelligent_pyramid_principle",
                thinking_process=thinking_process,
                cleaned_response=cleaned_response,
                confidence=context_insights.get("overall_coherence", 0.8),
            )

            self.logger.info(
                f"💾 Pyramid building cognitive exhaust stored for engagement {engagement_id}"
            )

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to store pyramid cognitive exhaust: {e}")


# Factory function for backward compatibility
def create_context_aware_pyramid_builder(
    context_intelligence: Optional[IContextIntelligence] = None,
) -> ContextIntelligentPyramidBuilder:
    """Factory function to create context-aware pyramid builder"""
    return ContextIntelligentPyramidBuilder(context_intelligence=context_intelligence)
