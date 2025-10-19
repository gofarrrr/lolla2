"""
PerspectiveMapperService Domain Service
======================================

Domain service for perspective mapping functionality extracted from
the engine to enable proper testing and modular architecture.

Core responsibility: Map cognitive approaches and mental models used by consultants,
revealing the diverse analytical perspectives and thinking patterns.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter

from src.services.interfaces.perspective_mapper_interface import (
    IPerspectiveMapperService,
    PerspectiveMappingError,
)
from src.arbitration.models import (
    ConsultantOutput,
    ConsultantRole,
    PerspectiveType,
)

logger = logging.getLogger(__name__)


class PerspectiveMapperService(IPerspectiveMapperService):
    """
    Domain service for perspective mapping

    Core philosophy: Understand and map the cognitive diversity of consultants
    to reveal how different thinking patterns contribute to comprehensive analysis.
    """

    def __init__(self):
        self.logger = logger

        # Perspective indicators for analysis
        self.perspective_indicators = {
            PerspectiveType.RISK_FOCUSED: ['risk', 'threat', 'danger', 'challenge', 'concern', 'downside', 'vulnerability'],
            PerspectiveType.OPPORTUNITY_FOCUSED: ['opportunity', 'potential', 'advantage', 'benefit', 'upside', 'growth', 'innovation'],
            PerspectiveType.USER_EXPERIENCE_FOCUSED: ['user', 'customer', 'experience', 'usability', 'satisfaction', 'journey', 'interface'],
            PerspectiveType.FINANCIAL_FOCUSED: ['revenue', 'cost', 'profit', 'budget', 'roi', 'financial', 'investment', 'pricing'],
            PerspectiveType.OPERATIONAL_FOCUSED: ['process', 'operation', 'efficiency', 'workflow', 'execution', 'implementation'],
            PerspectiveType.STRATEGIC_FOCUSED: ['strategy', 'vision', 'long-term', 'competitive', 'market', 'positioning'],
            PerspectiveType.COMPLIANCE_FOCUSED: ['compliance', 'regulation', 'legal', 'governance', 'policy', 'standards'],
        }

        # Cognitive approach patterns
        self.cognitive_patterns = {
            'analytical': ['analysis', 'data', 'evidence', 'research', 'metrics', 'quantitative'],
            'systems_thinking': ['system', 'interconnected', 'holistic', 'ecosystem', 'network', 'integration'],
            'design_thinking': ['design', 'user-centered', 'iterative', 'prototype', 'empathy', 'creative'],
            'lean_thinking': ['lean', 'waste', 'efficiency', 'continuous', 'improvement', 'agile'],
            'strategic_thinking': ['strategic', 'competitive', 'positioning', 'differentiation', 'advantage'],
        }

    def map_consultant_perspectives(
        self,
        consultant_outputs: List[ConsultantOutput],
    ) -> Dict[str, Any]:
        """
        Map cognitive approaches and mental models used by consultants

        Args:
            consultant_outputs: List of consultant analyses

        Returns:
            Dict[str, Any]: Comprehensive perspective mapping

        Raises:
            PerspectiveMappingError: If perspective mapping fails
        """
        try:
            if not consultant_outputs:
                raise PerspectiveMappingError("No consultant outputs provided")
            self.logger.info(f"🗺️ Mapping perspectives for {len(consultant_outputs)} consultants")

            # Core perspective mapping components
            cognitive_approaches = self.identify_cognitive_approaches(consultant_outputs)
            mental_models = self.extract_mental_models(consultant_outputs)
            perspective_patterns = self.analyze_perspective_patterns(consultant_outputs)
            cognitive_diversity = self.map_cognitive_diversity(consultant_outputs)
            perspective_gaps = self.identify_perspective_gaps(consultant_outputs)

            # Build comprehensive perspective map
            perspective_map = {
                'cognitive_approaches': cognitive_approaches,
                'mental_models': mental_models,
                'perspective_patterns': perspective_patterns,
                'cognitive_diversity': cognitive_diversity,
                'perspective_gaps': perspective_gaps,
                'consultant_profiles': self._build_consultant_profiles(consultant_outputs),
                'perspective_matrix': self._build_perspective_matrix(consultant_outputs),
                'thinking_style_distribution': self._analyze_thinking_styles(consultant_outputs),
                'coverage_analysis': self._analyze_perspective_coverage(consultant_outputs),
            }

            self.logger.info("✅ Perspective mapping completed successfully")
            return perspective_map

        except Exception as e:
            self.logger.error(f"❌ Perspective mapping failed: {e}")
            raise PerspectiveMappingError(f"Failed to map consultant perspectives: {e}")

    def identify_cognitive_approaches(
        self,
        consultant_outputs: List[ConsultantOutput],
    ) -> Dict[ConsultantRole, List[str]]:
        """
        Identify cognitive approaches used by each consultant

        Args:
            consultant_outputs: List of consultant analyses

        Returns:
            Dict[ConsultantRole, List[str]]: Cognitive approaches per consultant

        Raises:
            PerspectiveMappingError: If cognitive approach identification fails
        """
        try:
            self.logger.info("🧠 Identifying cognitive approaches")
            approaches = {}

            for output in consultant_outputs:
                content = self._extract_full_content(output)
                consultant_approaches = []

                # Analyze content for cognitive patterns
                for approach, keywords in self.cognitive_patterns.items():
                    score = self._calculate_pattern_strength(content, keywords)
                    if score > 0.3:  # Threshold for significant presence
                        consultant_approaches.append(f"{approach} (strength: {score:.2f})")

                # Add specific approach indicators
                if self._detect_framework_usage(content):
                    consultant_approaches.append("framework-based")

                if self._detect_evidence_based_reasoning(content):
                    consultant_approaches.append("evidence-based")

                approaches[output.consultant_role] = consultant_approaches

            self.logger.info(f"✅ Identified cognitive approaches for {len(approaches)} consultants")
            return approaches

        except Exception as e:
            self.logger.error(f"❌ Cognitive approach identification failed: {e}")
            raise PerspectiveMappingError(f"Failed to identify cognitive approaches: {e}")

    def extract_mental_models(
        self,
        consultant_outputs: List[ConsultantOutput],
    ) -> Dict[ConsultantRole, List[str]]:
        """
        Extract mental models and frameworks used by consultants

        Args:
            consultant_outputs: List of consultant analyses

        Returns:
            Dict[ConsultantRole, List[str]]: Mental models per consultant

        Raises:
            PerspectiveMappingError: If mental model extraction fails
        """
        try:
            self.logger.info("🧩 Extracting mental models")
            mental_models = {}

            # Common business mental models and frameworks
            framework_indicators = {
                'SWOT Analysis': ['strength', 'weakness', 'opportunity', 'threat', 'swot'],
                'Porter Five Forces': ['competitive', 'supplier', 'buyer', 'substitute', 'barrier'],
                'Value Chain': ['value chain', 'primary activities', 'support activities'],
                'Business Model Canvas': ['value proposition', 'customer segment', 'revenue stream'],
                'Lean Startup': ['mvp', 'pivot', 'validated learning', 'build-measure-learn'],
                'OKR Framework': ['objective', 'key result', 'okr', 'measurable'],
                'Design Thinking': ['empathize', 'define', 'ideate', 'prototype', 'test'],
                'Agile Methodology': ['sprint', 'scrum', 'iterative', 'backlog'],
                'Risk Management': ['risk matrix', 'mitigation', 'contingency', 'probability'],
                'Financial Modeling': ['dcf', 'valuation', 'cash flow', 'irr', 'npv'],
            }

            for output in consultant_outputs:
                content = self._extract_full_content(output)
                consultant_models = []

                # Check for explicit mental models
                for framework, indicators in framework_indicators.items():
                    if self._detect_framework_presence(content, indicators):
                        consultant_models.append(framework)

                # Infer implicit mental models from content patterns
                implicit_models = self._infer_implicit_models(content)
                consultant_models.extend(implicit_models)

                mental_models[output.consultant_role] = consultant_models

            self.logger.info(f"✅ Extracted mental models for {len(mental_models)} consultants")
            return mental_models

        except Exception as e:
            self.logger.error(f"❌ Mental model extraction failed: {e}")
            raise PerspectiveMappingError(f"Failed to extract mental models: {e}")

    def analyze_perspective_patterns(
        self,
        consultant_outputs: List[ConsultantOutput],
    ) -> Dict[str, Any]:
        """
        Analyze patterns in how consultants approach problems

        Args:
            consultant_outputs: List of consultant analyses

        Returns:
            Dict[str, Any]: Perspective pattern analysis

        Raises:
            PerspectiveMappingError: If pattern analysis fails
        """
        try:
            self.logger.info("📊 Analyzing perspective patterns")

            approach_styles = self._analyze_approach_styles(consultant_outputs)
            focus_areas = self._analyze_focus_areas(consultant_outputs)
            reasoning_patterns = self._analyze_reasoning_patterns(consultant_outputs)
            solution_orientations = self._analyze_solution_orientations(consultant_outputs)
            communication_styles = self._analyze_communication_styles(consultant_outputs)

            # Derive summary metrics expected by tests
            # pattern_count: total distinct pattern categories across sections
            pattern_count = (
                len(set(approach_styles.values()))
                + sum(1 for _ in reasoning_patterns.items())
            )
            # dominant_patterns: top occurrences among approach styles
            dominant_counter = Counter(approach_styles.values())
            dominant_patterns = [p for p, _ in dominant_counter.most_common(2)]

            patterns = {
                'approach_styles': approach_styles,
                'focus_areas': focus_areas,
                'reasoning_patterns': reasoning_patterns,
                'solution_orientations': solution_orientations,
                'communication_styles': communication_styles,
                'pattern_count': pattern_count,
                'dominant_patterns': dominant_patterns,
            }

            self.logger.info("✅ Perspective pattern analysis completed")
            return patterns

        except Exception as e:
            self.logger.error(f"❌ Perspective pattern analysis failed: {e}")
            raise PerspectiveMappingError(f"Failed to analyze perspective patterns: {e}")

    def map_cognitive_diversity(
        self,
        consultant_outputs: List[ConsultantOutput],
    ) -> Dict[str, float]:
        """
        Map cognitive diversity across the consultant team

        Args:
            consultant_outputs: List of consultant analyses

        Returns:
            Dict[str, float]: Cognitive diversity metrics

        Raises:
            PerspectiveMappingError: If diversity mapping fails
        """
        try:
            self.logger.info("🌈 Mapping cognitive diversity")

            diversity_metrics = {
                'perspective_diversity': self._calculate_perspective_diversity(consultant_outputs),
                'approach_diversity': self._calculate_approach_diversity(consultant_outputs),
                'cognitive_distance': self._calculate_cognitive_distance(consultant_outputs),
                'thinking_style_variance': self._calculate_thinking_style_variance(consultant_outputs),
                'coverage_completeness': self._calculate_coverage_completeness(consultant_outputs),
            }

            self.logger.info("✅ Cognitive diversity mapping completed")
            return diversity_metrics

        except Exception as e:
            self.logger.error(f"❌ Cognitive diversity mapping failed: {e}")
            raise PerspectiveMappingError(f"Failed to map cognitive diversity: {e}")

    def identify_perspective_gaps(
        self,
        consultant_outputs: List[ConsultantOutput],
        target_perspectives: Optional[List[PerspectiveType]] = None,
    ) -> List[PerspectiveType]:
        """
        Identify gaps in perspective coverage

        Args:
            consultant_outputs: List of consultant analyses
            target_perspectives: Optional list of target perspectives to check

        Returns:
            List[PerspectiveType]: Missing or underrepresented perspectives

        Raises:
            PerspectiveMappingError: If gap identification fails
        """
        try:
            self.logger.info("🔍 Identifying perspective gaps")

            if target_perspectives is None:
                target_perspectives = list(PerspectiveType)

            gaps = []
            perspective_coverage = self._calculate_perspective_coverage(consultant_outputs)

            for perspective in target_perspectives:
                coverage_score = perspective_coverage.get(perspective, 0.0)
                if coverage_score < 0.3:  # Low coverage threshold
                    gaps.append(perspective)

            self.logger.info(f"✅ Identified {len(gaps)} perspective gaps")
            return gaps

        except Exception as e:
            self.logger.error(f"❌ Perspective gap identification failed: {e}")
            raise PerspectiveMappingError(f"Failed to identify perspective gaps: {e}")

    # Helper methods

    def _extract_full_content(self, output: ConsultantOutput) -> str:
        """Extract all textual content from consultant output"""
        content_parts = [
            output.executive_summary,
            ' '.join(output.key_insights),
            ' '.join(output.recommendations),
            ' '.join(output.mental_models_used),
        ]
        return ' '.join(content_parts).lower()

    def _calculate_pattern_strength(self, content: str, keywords: List[str]) -> float:
        """Calculate strength of a pattern in content"""
        word_count = len(content.split())
        if word_count == 0:
            return 0.0

        keyword_matches = sum(1 for keyword in keywords if keyword in content)
        return min(keyword_matches / len(keywords), 1.0)

    def _detect_framework_usage(self, content: str) -> bool:
        """Detect if content shows framework-based thinking"""
        framework_indicators = ['framework', 'model', 'methodology', 'approach', 'systematic']
        return any(indicator in content for indicator in framework_indicators)

    def _detect_evidence_based_reasoning(self, content: str) -> bool:
        """Detect evidence-based reasoning in content"""
        evidence_indicators = ['data', 'evidence', 'research', 'study', 'analysis', 'metrics']
        return sum(1 for indicator in evidence_indicators if indicator in content) >= 2

    def _detect_framework_presence(self, content: str, indicators: List[str]) -> bool:
        """Detect presence of specific framework"""
        matches = sum(1 for indicator in indicators if indicator in content)
        return matches >= 2  # At least 2 indicators present

    def _infer_implicit_models(self, content: str) -> List[str]:
        """Infer implicit mental models from content"""
        implicit_models = []

        # Pattern-based inference
        if 'cause' in content and 'effect' in content:
            implicit_models.append('Causal Reasoning')

        if 'trade-off' in content or 'balance' in content:
            implicit_models.append('Trade-off Analysis')

        if 'stakeholder' in content:
            implicit_models.append('Stakeholder Analysis')

        if 'timeline' in content or 'phase' in content:
            implicit_models.append('Phased Implementation')

        return implicit_models

    def _analyze_approach_styles(self, consultant_outputs: List[ConsultantOutput]) -> Dict[ConsultantRole, str]:
        """Analyze approach styles of consultants"""
        styles = {}
        for output in consultant_outputs:
            content = self._extract_full_content(output)

            # Determine dominant style
            if self._calculate_pattern_strength(content, ['data', 'analysis', 'evidence']) > 0.4:
                style = 'analytical'
            elif self._calculate_pattern_strength(content, ['creative', 'innovative', 'design']) > 0.4:
                style = 'creative'
            elif self._calculate_pattern_strength(content, ['practical', 'implementation', 'execution']) > 0.4:
                style = 'pragmatic'
            else:
                style = 'balanced'

            styles[output.consultant_role] = style

        return styles

    def _analyze_focus_areas(self, consultant_outputs: List[ConsultantOutput]) -> Dict[ConsultantRole, List[str]]:
        """Analyze primary focus areas of consultants"""
        focus_areas = {}

        for output in consultant_outputs:
            content = self._extract_full_content(output)
            areas = []

            for perspective, keywords in self.perspective_indicators.items():
                strength = self._calculate_pattern_strength(content, keywords)
                if strength > 0.2:
                    areas.append(f"{perspective.value} ({strength:.2f})")

            focus_areas[output.consultant_role] = areas

        return focus_areas

    def _analyze_reasoning_patterns(self, consultant_outputs: List[ConsultantOutput]) -> Dict[str, Any]:
        """Analyze reasoning patterns across consultants"""
        patterns = {
            'deductive_reasoning': 0,
            'inductive_reasoning': 0,
            'abductive_reasoning': 0,
            'analogical_reasoning': 0,
        }

        for output in consultant_outputs:
            content = self._extract_full_content(output)

            # Simple heuristics for reasoning pattern detection
            if 'therefore' in content or 'thus' in content:
                patterns['deductive_reasoning'] += 1
            if 'pattern' in content or 'trend' in content:
                patterns['inductive_reasoning'] += 1
            if 'possible' in content or 'likely' in content:
                patterns['abductive_reasoning'] += 1
            if 'similar' in content or 'like' in content:
                patterns['analogical_reasoning'] += 1

        return patterns

    def _analyze_solution_orientations(self, consultant_outputs: List[ConsultantOutput]) -> Dict[ConsultantRole, str]:
        """Analyze solution orientations of consultants"""
        orientations = {}

        for output in consultant_outputs:
            content = self._extract_full_content(output)

            if self._calculate_pattern_strength(content, ['prevent', 'avoid', 'mitigate']) > 0.3:
                orientation = 'problem_prevention'
            elif self._calculate_pattern_strength(content, ['optimize', 'improve', 'enhance']) > 0.3:
                orientation = 'optimization'
            elif self._calculate_pattern_strength(content, ['transform', 'disrupt', 'revolutionize']) > 0.3:
                orientation = 'transformation'
            else:
                orientation = 'balanced'

            orientations[output.consultant_role] = orientation

        return orientations

    def _analyze_communication_styles(self, consultant_outputs: List[ConsultantOutput]) -> Dict[ConsultantRole, str]:
        """Analyze communication styles of consultants"""
        styles = {}

        for output in consultant_outputs:
            summary_length = len(output.executive_summary.split())
            insight_count = len(output.key_insights)

            if summary_length > 100 and insight_count > 5:
                style = 'detailed'
            elif summary_length < 50 and insight_count < 3:
                style = 'concise'
            else:
                style = 'balanced'

            styles[output.consultant_role] = style

        return styles

    def _calculate_perspective_diversity(self, consultant_outputs: List[ConsultantOutput]) -> float:
        """Calculate diversity of perspectives"""
        if not consultant_outputs:
            return 0.0

        perspective_scores = defaultdict(float)

        for output in consultant_outputs:
            content = self._extract_full_content(output)
            for perspective, keywords in self.perspective_indicators.items():
                score = self._calculate_pattern_strength(content, keywords)
                perspective_scores[perspective] += score

        # Calculate entropy-like measure
        total_score = sum(perspective_scores.values())
        if total_score == 0:
            return 0.0

        diversity = 0.0
        for score in perspective_scores.values():
            if score > 0:
                p = score / total_score
                diversity -= p * (p ** 0.5)  # Modified entropy

        return min(diversity, 1.0)

    def _calculate_approach_diversity(self, consultant_outputs: List[ConsultantOutput]) -> float:
        """Calculate diversity of cognitive approaches"""
        approaches = self.identify_cognitive_approaches(consultant_outputs)

        all_approaches = set()
        for consultant_approaches in approaches.values():
            all_approaches.update(consultant_approaches)

        unique_approaches = len(all_approaches)
        max_possible = len(consultant_outputs) * len(self.cognitive_patterns)

        return min(unique_approaches / max_possible if max_possible > 0 else 0.0, 1.0)

    def _calculate_cognitive_distance(self, consultant_outputs: List[ConsultantOutput]) -> float:
        """Calculate cognitive distance between consultants"""
        if len(consultant_outputs) < 2:
            return 0.0

        distances = []
        for i, output1 in enumerate(consultant_outputs):
            for output2 in consultant_outputs[i+1:]:
                content1 = self._extract_full_content(output1)
                content2 = self._extract_full_content(output2)

                # Simple distance measure based on word overlap
                words1 = set(content1.split())
                words2 = set(content2.split())

                intersection = len(words1 & words2)
                union = len(words1 | words2)

                distance = 1.0 - (intersection / union if union > 0 else 0.0)
                distances.append(distance)

        return sum(distances) / len(distances) if distances else 0.0

    def _calculate_thinking_style_variance(self, consultant_outputs: List[ConsultantOutput]) -> float:
        """Calculate variance in thinking styles"""
        styles = self._analyze_approach_styles(consultant_outputs)
        style_counts = Counter(styles.values())

        # Calculate variance in style distribution
        total = len(consultant_outputs)
        if total <= 1:
            return 0.0

        style_proportions = [count / total for count in style_counts.values()]
        mean_proportion = sum(style_proportions) / len(style_proportions)

        variance = sum((p - mean_proportion) ** 2 for p in style_proportions) / len(style_proportions)
        return min(variance * 4, 1.0)  # Scale to 0-1 range

    def _calculate_coverage_completeness(self, consultant_outputs: List[ConsultantOutput]) -> float:
        """Calculate completeness of perspective coverage"""
        coverage = self._calculate_perspective_coverage(consultant_outputs)

        covered_perspectives = sum(1 for score in coverage.values() if score > 0.2)
        total_perspectives = len(PerspectiveType)

        return covered_perspectives / total_perspectives if total_perspectives > 0 else 0.0

    def _calculate_perspective_coverage(self, consultant_outputs: List[ConsultantOutput]) -> Dict[PerspectiveType, float]:
        """Calculate coverage score for each perspective type"""
        coverage = {}

        for perspective, keywords in self.perspective_indicators.items():
            total_strength = 0.0
            for output in consultant_outputs:
                content = self._extract_full_content(output)
                strength = self._calculate_pattern_strength(content, keywords)
                total_strength += strength

            # Average strength across consultants
            coverage[perspective] = total_strength / len(consultant_outputs) if consultant_outputs else 0.0

        return coverage

    def _build_consultant_profiles(self, consultant_outputs: List[ConsultantOutput]) -> Dict[ConsultantRole, Dict[str, Any]]:
        """Build detailed profiles for each consultant"""
        profiles = {}

        for output in consultant_outputs:
            content = self._extract_full_content(output)

            profile = {
                'primary_perspective': self._identify_primary_perspective(content),
                'thinking_style': self._identify_thinking_style(content),
                'communication_pattern': self._analyze_communication_pattern(output),
                'strength_areas': self._identify_strength_areas(content),
                'unique_contributions': self._identify_unique_contributions(output),
            }

            profiles[output.consultant_role] = profile

        return profiles

    def _identify_primary_perspective(self, content: str) -> str:
        """Identify primary perspective of consultant"""
        max_strength = 0.0
        primary_perspective = 'balanced'

        for perspective, keywords in self.perspective_indicators.items():
            strength = self._calculate_pattern_strength(content, keywords)
            if strength > max_strength:
                max_strength = strength
                primary_perspective = perspective.value

        return primary_perspective

    def _identify_thinking_style(self, content: str) -> str:
        """Identify dominant thinking style"""
        max_strength = 0.0
        thinking_style = 'balanced'

        for style, keywords in self.cognitive_patterns.items():
            strength = self._calculate_pattern_strength(content, keywords)
            if strength > max_strength:
                max_strength = strength
                thinking_style = style

        return thinking_style

    def _analyze_communication_pattern(self, output: ConsultantOutput) -> Dict[str, Any]:
        """Analyze communication patterns"""
        return {
            'summary_length': len(output.executive_summary.split()),
            'insight_count': len(output.key_insights),
            'recommendation_count': len(output.recommendations),
            'avg_insight_length': sum(len(insight.split()) for insight in output.key_insights) / max(len(output.key_insights), 1),
            'confidence_level': output.confidence_level,
        }

    def _identify_strength_areas(self, content: str) -> List[str]:
        """Identify areas of strength for consultant"""
        strengths = []

        for perspective, keywords in self.perspective_indicators.items():
            strength = self._calculate_pattern_strength(content, keywords)
            if strength > 0.4:  # High strength threshold
                strengths.append(perspective.value)

        return strengths

    def _identify_unique_contributions(self, output: ConsultantOutput) -> List[str]:
        """Identify unique contributions of consultant"""
        # This would be enhanced with comparison to other consultants
        # For now, return key insights as unique contributions
        return output.key_insights[:3]  # Top 3 insights

    def _build_perspective_matrix(self, consultant_outputs: List[ConsultantOutput]) -> Dict[str, Dict[str, float]]:
        """Build matrix of perspective coverage by consultant"""
        matrix = {}

        for output in consultant_outputs:
            content = self._extract_full_content(output)
            consultant_row = {}

            for perspective, keywords in self.perspective_indicators.items():
                strength = self._calculate_pattern_strength(content, keywords)
                consultant_row[perspective.value] = strength

            matrix[output.consultant_role.value] = consultant_row

        return matrix

    def _analyze_thinking_styles(self, consultant_outputs: List[ConsultantOutput]) -> Dict[str, int]:
        """Analyze distribution of thinking styles"""
        style_distribution = defaultdict(int)

        for output in consultant_outputs:
            content = self._extract_full_content(output)
            style = self._identify_thinking_style(content)
            style_distribution[style] += 1

        return dict(style_distribution)

    def _analyze_perspective_coverage(self, consultant_outputs: List[ConsultantOutput]) -> Dict[str, Any]:
        """Analyze overall perspective coverage"""
        coverage = self._calculate_perspective_coverage(consultant_outputs)

        return {
            'total_perspectives_covered': sum(1 for score in coverage.values() if score > 0.1),
            'strong_coverage_count': sum(1 for score in coverage.values() if score > 0.4),
            'coverage_scores': {p.value: score for p, score in coverage.items()},
            'coverage_completeness': self._calculate_coverage_completeness(consultant_outputs),
        }