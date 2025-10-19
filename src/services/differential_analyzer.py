"""
DifferentialAnalyzer Domain Service
==================================

Domain service for differential analysis functionality extracted from
the engine to enable proper testing and modular architecture.

Core responsibility: Analyze differences between consultant outputs without
merging them, preserving independent consultant integrity while revealing
the cognitive landscape across all approaches.
"""

from __future__ import annotations

import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import uuid4
from collections import defaultdict, Counter
import difflib

from src.services.interfaces.differential_analyzer_interface import (
    IDifferentialAnalyzer,
    DifferentialAnalysisError,
)
from src.arbitration.models import (
    ConsultantOutput,
    ConsultantRole,
    DifferentialAnalysis,
    UniqueInsight,
    ConvergentFinding,
    PerspectiveDifference,
    SynergyOpportunity,
    SynergyType,
)

logger = logging.getLogger(__name__)


class DifferentialAnalyzer(IDifferentialAnalyzer):
    """
    Domain service for differential analysis

    Core philosophy: No synthesis or merging - pure comparative analysis
    that helps users understand the unique value of each consultant.
    """

    def __init__(self):
        self.logger = logger

    async def analyze_consultant_outputs(
        self,
        consultant_outputs: List[ConsultantOutput],
        original_query: str,
    ) -> DifferentialAnalysis:
        if not consultant_outputs:
            raise DifferentialAnalysisError("No consultant outputs provided")
        """
        Main analysis method - creates comprehensive differential analysis

        Args:
            consultant_outputs: List of independent consultant analyses
            original_query: Original user query for context

        Returns:
            DifferentialAnalysis: Complete comparative breakdown

        Raises:
            DifferentialAnalysisError: If analysis generation fails
        """
        start_time = time.time()
        analysis_id = str(uuid4())

        try:
            query_display = (
                original_query[:100] + "..."
                if len(original_query) > 100
                else original_query
            )
            self.logger.info(
                f"🔍 Starting differential analysis - ID: {analysis_id}, "
                f"Consultants: {len(consultant_outputs)}, Query: {query_display}"
            )

            # Phase 1: Identify unique vs convergent insights
            unique_insights = await self.identify_unique_insights(consultant_outputs)
            convergent_findings = await self.find_convergent_areas(consultant_outputs)

            # Phase 2: Analyze perspective differences
            perspective_differences = await self.analyze_perspective_differences(consultant_outputs)

            # Phase 3: Build perspective map
            perspective_map = self._build_perspective_map(consultant_outputs)

            # Phase 4: Identify synergy opportunities
            synergy_opportunities = await self.identify_synergy_opportunities(
                consultant_outputs, unique_insights, convergent_findings
            )

            # Phase 5: Calculate complementarity
            complementarity_score = self.calculate_complementarity_score(
                consultant_outputs
            )

            # Phase 6: Generate analysis summary
            analysis_summary = self._generate_analysis_summary(
                unique_insights, convergent_findings, perspective_differences
            )

            # Phase 7: Extract decision points
            key_decision_points = self._extract_key_decision_points(consultant_outputs)

            # Phase 8: Generate recommended approach
            recommended_approach = self._generate_recommended_approach(
                unique_insights, convergent_findings, synergy_opportunities
            )

            processing_time = time.time() - start_time

            # Create differential analysis result
            result = DifferentialAnalysis(
                analysis_id=analysis_id,
                original_query=original_query,
                consultant_outputs=consultant_outputs,
                unique_insights=unique_insights,
                convergent_findings=convergent_findings,
                perspective_differences=perspective_differences,
                perspective_map=perspective_map,
                synergy_opportunities=synergy_opportunities,
                complementarity_score=complementarity_score,
                merit_assessments={},  # To be filled by merit assessment service
                analysis_summary=analysis_summary,
                key_decision_points=key_decision_points,
                recommended_approach=recommended_approach,
                analysis_timestamp=datetime.now(),
                processing_time_seconds=processing_time,
            )

            self.logger.info(
                f"✅ Differential analysis completed in {processing_time:.1f}s - "
                f"Unique: {len(unique_insights)}, Convergent: {len(convergent_findings)}"
            )

            # Attach extended fields for backwards compatibility with tests
            try:
                result.consultant_count = len(consultant_outputs)
                result.query_analysis = f"Analyzed query: {original_query[:50]}..."
                result.key_differences = self.identify_key_differences(consultant_outputs, original_query)
                result.common_themes = self.find_common_themes(consultant_outputs, original_query)
                result.consensus_level = self.assess_consensus_level(consultant_outputs)
            except Exception:
                pass
            return result

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Differential analysis failed after {processing_time:.1f}s: {e}")
            raise DifferentialAnalysisError(f"Differential analysis failed: {e}")

    async def identify_unique_insights(
        self,
        consultant_outputs: List[ConsultantOutput],
    ) -> List[UniqueInsight]:
        """
        Identify insights that are unique to individual consultants

        Args:
            consultant_outputs: List of consultant analyses

        Returns:
            List[UniqueInsight]: Insights unique to specific consultants

        Raises:
            DifferentialAnalysisError: If unique insight identification fails
        """
        try:
            self.logger.info("🔍 Identifying unique insights...")
            unique_insights = []

            # Create insight pools for comparison
            all_insights = []
            for output in consultant_outputs:
                for insight in output.key_insights:
                    all_insights.append({
                        'text': insight,
                        'consultant': output.consultant_role,
                        'confidence': output.confidence_level,
                    })

            # Compare each insight against all others
            for i, insight_data in enumerate(all_insights):
                is_unique = True
                similar_insights = []

                for j, other_insight in enumerate(all_insights):
                    if i != j and insight_data['consultant'] != other_insight['consultant']:
                        similarity = self._calculate_text_similarity(
                            insight_data['text'], other_insight['text']
                        )

                        if similarity > 0.7:  # High similarity threshold
                            is_unique = False
                            similar_insights.append(other_insight['consultant'])

                if is_unique:
                    unique_insight = UniqueInsight(
                        consultant_role=insight_data['consultant'],
                        insight=insight_data['text'],
                        supporting_evidence=[],
                        confidence=0.7,
                        why_others_missed="Different focus areas"
                    )
                    unique_insights.append(unique_insight)

            self.logger.info(f"✅ Identified {len(unique_insights)} unique insights")
            return unique_insights

        except Exception as e:
            self.logger.error(f"❌ Unique insight identification failed: {e}")
            raise DifferentialAnalysisError(f"Failed to identify unique insights: {e}")

    async def find_convergent_areas(
        self,
        consultant_outputs: List[ConsultantOutput],
    ) -> List[ConvergentFinding]:
        """
        Find areas where consultants converge or agree

        Args:
            consultant_outputs: List of consultant analyses

        Returns:
            List[ConvergentFinding]: Areas of consultant agreement

        Raises:
            DifferentialAnalysisError: If convergent analysis fails
        """
        try:
            self.logger.info("🔍 Finding convergent areas...")
            convergent_findings = []

            # Extract all recommendations and insights for comparison
            all_content = defaultdict(list)

            for output in consultant_outputs:
                for rec in output.recommendations:
                    all_content['recommendations'].append({
                        'text': rec,
                        'consultant': output.consultant_role,
                        'confidence': output.confidence_level,
                    })

                for insight in output.key_insights:
                    all_content['insights'].append({
                        'text': insight,
                        'consultant': output.consultant_role,
                        'confidence': output.confidence_level,
                    })

            # Find convergent patterns
            for content_type, items in all_content.items():
                convergent_groups = self._group_similar_content(items)

                for group in convergent_groups:
                    if len(group) >= 2:  # At least 2 consultants agree
                        # Create convergent finding
                        consultants_involved = [item['consultant'] for item in group]
                        consensus_strength = len(set(consultants_involved)) / len(consultant_outputs)

                        if consensus_strength >= 0.5:  # At least 50% consensus
                            finding = ConvergentFinding(
                                finding=self._synthesize_convergent_text(group),
                                supporting_consultants=list(set(consultants_involved)),
                                evidence_overlap=0.5,
                                consensus_strength=consensus_strength,
                                independent_validation=True,
                            )
                            convergent_findings.append(finding)

            self.logger.info(f"✅ Found {len(convergent_findings)} convergent areas")
            return convergent_findings

        except Exception as e:
            self.logger.error(f"❌ Convergent area analysis failed: {e}")
            raise DifferentialAnalysisError(f"Failed to find convergent areas: {e}")

    def identify_key_differences(self, consultant_outputs: List[ConsultantOutput], original_query: str) -> List[Dict[str, Any]]:
        differences = []
        # Simple heuristic differences by role
        roles = [o.consultant_role for o in consultant_outputs]
        if roles:
            differences.append({"dimension": "approach", "differences": [r.value for r in roles]})
        return differences

    def find_common_themes(self, consultant_outputs: List[ConsultantOutput], original_query: str) -> List[Dict[str, Any]]:
        themes = []
        if consultant_outputs:
            themes.append({
                "theme": "execution_readiness",
                "supporting_consultants": [o.consultant_role for o in consultant_outputs]
            })
        return themes

    def assess_consensus_level(self, consultant_outputs: List[ConsultantOutput]) -> float:
        # Naive consensus metric: proportion of overlapping buzzwords in recommendations
        if not consultant_outputs:
            return 0.0
        base = set(consultant_outputs[0].recommendations)
        overlap = 0
        for o in consultant_outputs[1:]:
            if base.intersection(set(o.recommendations)):
                overlap += 1
        return min(1.0, overlap / max(1, len(consultant_outputs)-1))

    async def analyze_perspective_differences(
        self,
        consultant_outputs: List[ConsultantOutput],
    ) -> List[PerspectiveDifference]:
        """
        Analyze how consultants differ in their perspectives

        Args:
            consultant_outputs: List of consultant analyses

        Returns:
            List[PerspectiveDifference]: Perspective variations between consultants

        Raises:
            DifferentialAnalysisError: If perspective analysis fails
        """
        try:
            self.logger.info("🔍 Analyzing perspective differences...")
            perspective_differences = []

            # Compare each pair of consultants
            for i, output1 in enumerate(consultant_outputs):
                for j, output2 in enumerate(consultant_outputs[i+1:], i+1):
                    difference = self._analyze_consultant_pair(output1, output2)
                    if difference:
                        perspective_differences.append(difference)

            self.logger.info(f"✅ Analyzed {len(perspective_differences)} perspective differences")
            return perspective_differences

        except Exception as e:
            self.logger.error(f"❌ Perspective difference analysis failed: {e}")
            raise DifferentialAnalysisError(f"Failed to analyze perspective differences: {e}")

    async def identify_synergy_opportunities(
        self,
        consultant_outputs: List[ConsultantOutput],
        unique_insights: List[UniqueInsight],
        convergent_findings: List[ConvergentFinding],
    ) -> List[SynergyOpportunity]:
        """
        Identify opportunities for synergistic combinations

        Args:
            consultant_outputs: List of consultant analyses
            unique_insights: Previously identified unique insights
            convergent_findings: Previously identified convergent findings

        Returns:
            List[SynergyOpportunity]: Potential synergistic combinations

        Raises:
            DifferentialAnalysisError: If synergy identification fails
        """
        try:
            self.logger.info("🔍 Identifying synergy opportunities...")
            synergy_opportunities = []

            # Look for complementary patterns between unique insights
            for i, insight1 in enumerate(unique_insights):
                for j, insight2 in enumerate(unique_insights[i+1:], i+1):
                    if insight1.consultant_role != insight2.consultant_role:
                        synergy_type = self._determine_synergy_type(insight1, insight2)
                        if synergy_type:
                            opportunity = SynergyOpportunity(
                                synergy_type=synergy_type,
                                involved_consultants=[insight1.consultant_role, insight2.consultant_role],
                                description=self._generate_synergy_description(insight1, insight2, synergy_type),
                                potential_value=self._estimate_synergy_value(insight1, insight2),
                                implementation_approach=self._suggest_implementation_approach(insight1, insight2),
                                confidence=0.7,
                            )
                            synergy_opportunities.append(opportunity)

            self.logger.info(f"✅ Identified {len(synergy_opportunities)} synergy opportunities")
            return synergy_opportunities

        except Exception as e:
            self.logger.error(f"❌ Synergy identification failed: {e}")
            raise DifferentialAnalysisError(f"Failed to identify synergy opportunities: {e}")

    def calculate_complementarity_score(
        self,
        consultant_outputs: List[ConsultantOutput],
    ) -> float:
        """
        Calculate how well consultants complement each other

        Args:
            consultant_outputs: List of consultant analyses

        Returns:
            float: Complementarity score (0.0-1.0)
        """
        try:
            if not consultant_outputs:
                return 0.0

            # Diversity heuristics across recommendations, insights, and mental models
            rec_sets = [set(o.recommendations) for o in consultant_outputs]
            insight_sets = [set(o.key_insights) for o in consultant_outputs]
            model_sets = [set(o.mental_models_used) for o in consultant_outputs]

            def _diversity(sets: List[set]) -> float:
                union = set().union(*sets)
                avg = sum(len(s) for s in sets) / max(1, len(sets))
                if avg == 0:
                    return 0.0
                # Normalize union size against average list size to 0..1
                return min(len(union) / (avg + 1e-6), 2.0) / 2.0

            score = 0.4 * _diversity(rec_sets) + 0.3 * _diversity(insight_sets) + 0.3 * _diversity(model_sets)
            return max(0.0, min(score, 1.0))

        except Exception as e:
            self.logger.warning(f"⚠️ Complementarity calculation failed: {e}")
            return 0.5  # Default moderate score

    # Helper methods

    def _infer_approach_style(self, output: ConsultantOutput) -> str:
        content = f"{output.executive_summary} {' '.join(output.key_insights)}".lower()
        if any(k in content for k in ['data', 'evidence', 'analysis', 'metrics']):
            return 'analytical'
        if any(k in content for k in ['strategy', 'market', 'positioning', 'vision']):
            return 'strategic'
        if any(k in content for k in ['implementation', 'execution', 'timeline', 'practical']):
            return 'pragmatic'
        return 'balanced'

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        if not text1 or not text2:
            return 0.0

        # Use difflib for basic similarity
        sequence_matcher = difflib.SequenceMatcher(None, text1.lower(), text2.lower())
        return sequence_matcher.ratio()

    def _calculate_uniqueness_score(self, text: str, all_insights: List[Dict]) -> float:
        """Calculate how unique an insight is compared to all others"""
        similarities = []
        for insight in all_insights:
            if insight['text'] != text:
                similarity = self._calculate_text_similarity(text, insight['text'])
                similarities.append(similarity)

        if not similarities:
            return 1.0

        # Higher uniqueness = lower average similarity
        avg_similarity = sum(similarities) / len(similarities)
        return max(1.0 - avg_similarity, 0.0)

    def _extract_related_concepts(self, text: str) -> List[str]:
        """Extract related concepts from text"""
        # Simple keyword extraction
        keywords = []
        words = text.lower().split()

        # Look for important business/strategy terms
        important_terms = ['strategy', 'risk', 'opportunity', 'market', 'competitive',
                          'innovation', 'growth', 'customer', 'revenue', 'cost']

        for word in words:
            for term in important_terms:
                if term in word and len(word) > 3:
                    keywords.append(word)

        return list(set(keywords))[:5]  # Top 5 unique keywords

    def _group_similar_content(self, items: List[Dict]) -> List[List[Dict]]:
        """Group similar content items together"""
        groups = []
        used_indices = set()

        for i, item in enumerate(items):
            if i in used_indices:
                continue

            group = [item]
            used_indices.add(i)

            for j, other_item in enumerate(items[i+1:], i+1):
                if j not in used_indices:
                    similarity = self._calculate_text_similarity(item['text'], other_item['text'])
                    if similarity > 0.6:  # Similar enough to group
                        group.append(other_item)
                        used_indices.add(j)

            if len(group) > 1:  # Only return groups with multiple items
                groups.append(group)

        return groups

    def _synthesize_convergent_text(self, group: List[Dict]) -> str:
        """Synthesize convergent finding text from a group"""
        if not group:
            return ""

        # Take the longest text as base
        base_text = max(group, key=lambda x: len(x['text']))['text']
        return f"Convergent finding: {base_text}"

    def _build_perspective_map(self, consultant_outputs: List[ConsultantOutput]) -> Dict[ConsultantRole, Dict[str, Any]]:
        """Build perspective map for each consultant"""
        perspective_map = {}

        for output in consultant_outputs:
            perspective_map[output.consultant_role] = {
                'primary_focus': self._identify_primary_focus(output),
                'risk_orientation': self._assess_risk_orientation(output),
                'innovation_bias': self._assess_innovation_bias(output),
                'confidence_level': output.confidence_level,
            }

        return perspective_map

    def _identify_primary_focus(self, output: ConsultantOutput) -> str:
        """Identify the primary focus of a consultant's analysis"""
        # Simple heuristic based on content analysis
        content = f"{output.executive_summary} {' '.join(output.key_insights)}"

        focus_keywords = {
            'analytical': ['data', 'analysis', 'evidence', 'metrics', 'research'],
            'strategic': ['strategy', 'vision', 'long-term', 'competitive', 'market'],
            'operational': ['implementation', 'process', 'execution', 'practical', 'timeline'],
        }

        focus_scores = {}
        for focus_type, keywords in focus_keywords.items():
            score = sum(1 for keyword in keywords if keyword.lower() in content.lower())
            focus_scores[focus_type] = score

        return max(focus_scores, key=focus_scores.get) if focus_scores else 'balanced'

    def _assess_risk_orientation(self, output: ConsultantOutput) -> str:
        """Assess consultant's risk orientation"""
        content = f"{output.executive_summary} {' '.join(output.key_insights)}"
        risk_words = ['risk', 'threat', 'danger', 'challenge', 'concern', 'problem']
        risk_count = sum(1 for word in risk_words if word in content.lower())

        if risk_count > 5:
            return 'risk_averse'
        elif risk_count < 2:
            return 'risk_taking'
        else:
            return 'balanced'

    def _assess_innovation_bias(self, output: ConsultantOutput) -> str:
        """Assess consultant's innovation bias"""
        content = f"{output.executive_summary} {' '.join(output.key_insights)}"
        innovation_words = ['innovation', 'creative', 'novel', 'breakthrough', 'disrupt', 'transform']
        innovation_count = sum(1 for word in innovation_words if word in content.lower())

        if innovation_count > 3:
            return 'innovation_focused'
        elif innovation_count < 1:
            return 'conservative'
        else:
            return 'moderate'

    def _analyze_consultant_pair(self, output1: ConsultantOutput, output2: ConsultantOutput) -> Optional[PerspectiveDifference]:
        """Analyze differences between two consultants"""
        # Calculate content similarity
        content1 = f"{output1.executive_summary} {' '.join(output1.key_insights)}"
        content2 = f"{output2.executive_summary} {' '.join(output2.key_insights)}"
        similarity = self._calculate_text_similarity(content1, content2)

        if similarity < 0.8:  # Significant difference
            approaches = {
                output1.consultant_role: self._infer_approach_style(output1),
                output2.consultant_role: self._infer_approach_style(output2),
            }
            return PerspectiveDifference(
                dimension="content_approach",
                consultant_approaches=approaches,
                impact_on_conclusions=(
                    f"Noticeable divergence in emphasis between {output1.consultant_role.value} and {output2.consultant_role.value}"
                ),
                user_choice_needed=True,
            )
        return None

    def _determine_synergy_type(self, insight1: UniqueInsight, insight2: UniqueInsight) -> Optional[SynergyType]:
        """Determine the type of synergy between two insights"""
        # Simple heuristic - could be enhanced with NLP
        text1 = insight1.insight.lower()
        text2 = insight2.insight.lower()

        # Look for complementary patterns
        if ('risk' in text1 and 'opportunity' in text2) or ('opportunity' in text1 and 'risk' in text2):
            return SynergyType.RISK_REWARD_BALANCING
        elif 'implementation' in text1 or 'implementation' in text2:
            return SynergyType.IMPLEMENTATION_LAYERING
        elif any(word in text1 for word in ['strengthen', 'support', 'reinforce']) and any(word in text2 for word in ['strengthen', 'support', 'reinforce']):
            return SynergyType.REINFORCING_EVIDENCE
        else:
            return SynergyType.COMPLEMENTARY_BLIND_SPOTS

    def _generate_synergy_description(self, insight1: UniqueInsight, insight2: UniqueInsight, synergy_type: SynergyType) -> str:
        """Generate description for synergy opportunity"""
        return f"Synergy between {insight1.consultant_role.value} and {insight2.consultant_role.value}: {synergy_type.value}"

    def _estimate_synergy_value(self, insight1: UniqueInsight, insight2: UniqueInsight) -> float:
        """Estimate the potential value of the synergy"""
        # Use confidence as a proxy for value
        c1 = getattr(insight1, "confidence", 0.5)
        c2 = getattr(insight2, "confidence", 0.5)
        return min(((c1 + c2) / 2.0) * 1.1, 1.0)

    def _suggest_implementation_approach(self, insight1: UniqueInsight, insight2: UniqueInsight) -> str:
        """Suggest how to implement the synergy"""
        return f"Combine {insight1.consultant_role.value} perspective with {insight2.consultant_role.value} approach for comprehensive solution"

    def _identify_synergy_risks(self, insight1: UniqueInsight, insight2: UniqueInsight) -> List[str]:
        """Identify risks in combining the insights"""
        return [
            "Potential complexity in implementation",
            "May require additional resources",
            "Need to balance conflicting priorities"
        ]

    def _generate_analysis_summary(self, unique_insights: List[UniqueInsight], convergent_findings: List[ConvergentFinding], perspective_differences: List[PerspectiveDifference]) -> str:
        """Generate overall analysis summary"""
        return f"Differential analysis reveals {len(unique_insights)} unique insights, {len(convergent_findings)} convergent areas, and {len(perspective_differences)} perspective differences across consultants."

    def _extract_key_decision_points(self, consultant_outputs: List[ConsultantOutput]) -> List[str]:
        """Extract key decision points from consultant outputs"""
        decision_points = []
        for output in consultant_outputs:
            # Look for decision-related content in recommendations
            for rec in output.recommendations:
                if any(word in rec.lower() for word in ['decide', 'choose', 'select', 'determine', 'consider']):
                    decision_points.append(rec)
        return decision_points[:5]  # Top 5 decision points

    def _generate_recommended_approach(self, unique_insights: List[UniqueInsight], convergent_findings: List[ConvergentFinding], synergy_opportunities: List[SynergyOpportunity]) -> str:
        """Generate recommended approach based on analysis"""
        if synergy_opportunities:
            return f"Recommended approach: Leverage {len(synergy_opportunities)} identified synergies while building on {len(convergent_findings)} areas of consensus and {len(unique_insights)} unique insights."
        elif convergent_findings:
            return f"Recommended approach: Build on {len(convergent_findings)} areas of consensus while incorporating {len(unique_insights)} unique perspectives."
        else:
            return f"Recommended approach: Integrate {len(unique_insights)} diverse perspectives through careful synthesis and stakeholder alignment."