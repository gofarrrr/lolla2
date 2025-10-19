"""
Aggregator - Merges consultant results and computes orthogonality.

Responsibilities:
- Merge individual LLM results into unified output
- Normalize consultant analyses from raw LLM responses
- Compute orthogonality index (cognitive diversity score)
- Identify convergent insights (agreed upon by multiple consultants)
- Identify divergent perspectives (areas of disagreement)
- Deduplicate recommendations
"""

import logging
import re
import json
from typing import List, Dict, Any, Set
from collections import defaultdict
from .interfaces import Aggregator
from .types import LLMResult, AggregatedOutput, AggregationPolicy, MergeStrategy


logger = logging.getLogger(__name__)


class StandardAggregator(Aggregator):
    """
    Standard implementation of Aggregator interface.

    Merges consultant results with:
    - Configurable merge strategies (union, majority, weighted, by_role)
    - Orthogonality computation for cognitive diversity
    - Convergence/divergence detection
    - Smart deduplication
    """

    def merge(
        self,
        results: List[LLMResult],
        policy: AggregationPolicy,
    ) -> AggregatedOutput:
        """
        Merge consultant results.

        Args:
            results: List of LLM results from consultants
            policy: Aggregation policy

        Returns:
            AggregatedOutput with merged insights and metrics
        """
        # Filter out failed results
        successful_results = [r for r in results if r.success]

        if not successful_results:
            # No successful results - return empty output
            return AggregatedOutput(
                consultant_analyses=[],
                convergent_insights=[],
                divergent_perspectives=[],
                orthogonality_index=0.0,
                merge_strategy_used=policy.merge_strategy,
                total_insights=0,
            )

        # Normalize LLM results to consultant analyses
        consultant_analyses = [
            self._normalize_llm_result(result) for result in successful_results
        ]

        # Compute orthogonality if enabled
        orthogonality_index = 0.0
        minority_report = None
        if policy.compute_orthogonality:
            orthogonality_index = self.compute_orthogonality(consultant_analyses)

            # Generate minority report only when comparing multiple consultants
            if len(consultant_analyses) >= 2 and orthogonality_index < policy.orthogonality_threshold:
                minority_report = self._generate_minority_report(consultant_analyses)

        # Identify convergence and divergence
        convergent_insights = self.identify_convergence(consultant_analyses)
        divergent_perspectives = self.identify_divergence(consultant_analyses)

        # Deduplicate if enabled
        deduplication_count = 0
        if policy.deduplicate_insights:
            consultant_analyses, dedup_count = self._deduplicate_insights(
                consultant_analyses, policy.similarity_threshold
            )
            deduplication_count = dedup_count

        # Calculate total insights
        total_insights = sum(
            len(ca.get("key_insights", [])) for ca in consultant_analyses
        )

        return AggregatedOutput(
            consultant_analyses=consultant_analyses,
            convergent_insights=convergent_insights,
            divergent_perspectives=divergent_perspectives,
            orthogonality_index=orthogonality_index,
            minority_report=minority_report,
            merge_strategy_used=policy.merge_strategy,
            deduplication_count=deduplication_count,
            total_insights=total_insights,
        )

    def _normalize_llm_result(self, result: LLMResult) -> Dict[str, Any]:
        """
        Normalize a single LLM result into a consultant analysis dict.

        Prefer JSON if present; otherwise, fallback to simple markdown parsing.
        """
        content = (result.content or "").strip()

        # Try JSON normalization first (lenient schema with safe defaults)
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                return {
                    "consultant_id": data.get("consultant_id", result.consultant_id),
                    "key_insights": data.get("key_insights") or data.get("insights") or [],
                    "risk_factors": data.get("risk_factors", []),
                    "opportunities": data.get("opportunities", []),
                    "recommendations": data.get("recommendations", []),
                    # Provide minimal defaults required by downstream code
                    "confidence_level": data.get("confidence_level", "medium"),
                    "analysis_quality": data.get("analysis_quality", "adequate"),
                }
        except json.JSONDecodeError:
            # Not JSON -> fall through to text parsing
            pass
        except Exception as e:
            logger.error(f"Failed to normalize JSON for {result.consultant_id}: {e}")
            # Fall back to text parsing

        # Fallback to text parsing (legacy support)
        return self._fallback_text_parsing(content, result.consultant_id)

    def _fallback_text_parsing(self, content: str, consultant_id: str) -> Dict[str, Any]:
        """Legacy fallback if JSON parsing fails"""
        return {
            "consultant_id": consultant_id,
            "key_insights": self._extract_list_items(content, "key insights?|insights?"),
            "risk_factors": self._extract_list_items(content, "risks?|risk factors?|challenges?"),
            "opportunities": self._extract_list_items(content, "opportunities?"),
            "recommendations": self._extract_list_items(content, "recommendations?|actions?"),
            "confidence_level": self._extract_confidence(content),
            "analysis_quality": self._assess_quality(content),
        }

    def _try_parse_json(self, content: str) -> Dict[str, Any]:
        """Try to parse content as JSON"""
        try:
            # Look for JSON block in content
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except:
            pass
        return None

    def _extract_list_items(self, content: str, section_pattern: str) -> List[str]:
        """
        Extract list items from a section.

        OPERATION GEARBOX: Handles McKinsey-structured multi-paragraph insights.
        Recognizes insight blocks that include:
        - Lead statement (bold)
        - Strategic Analysis section
        - What/So What/Now What paragraphs

        Keeps entire insight blocks together instead of splitting them.
        """
        # Find section header (case-insensitive, flexible formatting)
        # Format can be: **Text:**** or **Text:** or Text:
        pattern = rf"(?i)\*\*({section_pattern})\s*:?\*\*\s*\n"  # **Text:****
        match = re.search(pattern, content)

        if not match:
            # Try **Text**:
            pattern = rf"(?i)\*\*({section_pattern})\*\*\s*:?\s*\n"
            match = re.search(pattern, content)

        if not match:
            # Try without bold
            pattern = rf"(?i)({section_pattern})\s*:?\s*\n"
            match = re.search(pattern, content)

        if not match:
            return []

        # Extract text after header until next section or end
        start = match.end()
        # Look for next major section header (must be bold and end with colon)
        # This prevents matching subsection headers like **Strategic Analysis**
        next_section = re.search(r'\n\s*\*\*\s*(?:Risk|Opportunit|Recommendation|Concern)[^:\n]*\s*:?\*\*\s*\n', content[start:], re.IGNORECASE)
        end = start + next_section.start() if next_section else len(content)

        section_text = content[start:end].strip()

        if not section_text:
            return []

        # OPERATION GEARBOX: Extract McKinsey-structured insight blocks
        items = self._extract_mckinsey_insights(section_text)

        # Fallback: if no structured insights found, try traditional list parsing
        if not items:
            items = self._extract_traditional_list_items(section_text)

        return items[:5]  # Limit to 5 items per section

    def _extract_mckinsey_insights(self, section_text: str) -> List[str]:
        """
        Extract McKinsey-structured insight blocks.

        Structure:
        **Insight 1: Lead statement...**
        [**Strategic Analysis** - optional section header]
        *WHAT:* Paragraph explaining the finding...
        *SO WHAT:* Paragraph on implications...
        *NOW WHAT:* Paragraph on actions...

        Each complete block is kept as ONE insight.
        """
        insights = []

        # Split by double newlines to identify paragraph blocks
        # But we need to be smart about keeping related paragraphs together

        # Strategy: Find insight markers (numbered bold headings or lead statements)
        # Then capture everything until the next insight marker

        # Pattern 1: **Insight N: ...** or **Risk N: ...** etc.
        # Pattern 2: **Bold lead statement** (standalone paragraph)

        # Find all insight block boundaries
        insight_markers = []

        # Look for numbered insights: **Insight 1:**, **Risk 1:**, etc.
        for match in re.finditer(r'\n\s*\*\*\s*(?:Insight|Risk|Opportunity|Recommendation)\s+\d+\s*:', section_text, re.IGNORECASE):
            insight_markers.append(match.start())

        # If no numbered insights, look for bold lead statements at paragraph boundaries
        if not insight_markers:
            # Split by double newlines, look for paragraphs starting with **
            paragraphs = re.split(r'\n\s*\n', section_text)
            current_pos = 0

            for para in paragraphs:
                para = para.strip()
                if para.startswith('**') and para.endswith('**'):
                    # This is a bold lead statement
                    insight_markers.append(current_pos)
                current_pos += len(para) + 2  # +2 for the newlines we split on

        # Extract blocks between markers
        if insight_markers:
            for i, start_pos in enumerate(insight_markers):
                # Find end position (next marker or end of text)
                end_pos = insight_markers[i + 1] if i + 1 < len(insight_markers) else len(section_text)

                # Extract the complete insight block
                block = section_text[start_pos:end_pos].strip()

                if block and len(block) > 50:  # Minimum length for a real insight
                    insights.append(block)

        # If no structured insights found with markers, try to detect
        # multi-paragraph blocks that follow What/So What/Now What pattern
        if not insights:
            # Look for blocks containing WHAT/SO WHAT/NOW WHAT structure
            # This might be one continuous block
            if re.search(r'\*WHAT:\*.*\*SO WHAT:\*.*\*NOW WHAT:\*', section_text, re.DOTALL | re.IGNORECASE):
                # This whole section is likely one insight with full analysis
                # Check if there's a lead statement at the beginning
                paragraphs = section_text.split('\n\n')
                if paragraphs and paragraphs[0].strip().startswith('**'):
                    # First paragraph is lead statement, keep entire block together
                    insights.append(section_text.strip())

        return insights

    def _extract_traditional_list_items(self, section_text: str) -> List[str]:
        """
        Fallback: Extract traditional list items.

        Only matches proper list markers with space after them:
        - Item (hyphen + space)
        * Item (asterisk + space)
        1. Item (number + period + space)

        Does NOT match markdown formatting like **bold** or *italic*
        """
        items = []

        for line in section_text.split('\n'):
            line = line.strip()

            # Match ONLY proper list markers (marker + space)
            # This excludes markdown formatting like ** or *text*
            if re.match(r'^[-•]\s+', line):
                item = line[2:].strip()
            elif re.match(r'^\*\s+', line):
                item = line[2:].strip()
            elif re.match(r'^\d+[\.)]\s+', line):
                item = re.sub(r'^\d+[\.)]\s+', '', line).strip()
            else:
                item = None

            if item is not None and len(item) >= 1:
                items.append(item)

        return items

    def _extract_confidence(self, content: str) -> str:
        """Extract confidence level from content"""
        content_lower = content.lower()
        if "high confidence" in content_lower or "very confident" in content_lower:
            return "high"
        elif "low confidence" in content_lower or "uncertain" in content_lower:
            return "low"
        return "medium"

    def _assess_quality(self, content: str) -> str:
        """Assess analysis quality based on content"""
        if len(content) > 1500:
            return "excellent"
        elif len(content) > 800:
            return "good"
        elif len(content) > 300:
            return "adequate"
        return "poor"

    def compute_orthogonality(
        self,
        consultant_analyses: List[Dict[str, Any]],
    ) -> float:
        """
        Compute orthogonality index (cognitive diversity score).

        Algorithm:
        1. Extract all insights from all consultants
        2. Compute pairwise semantic similarity between consultant insight sets
        3. Orthogonality = 1 - average_similarity

        Returns:
            0.0 = perfect groupthink (all consultants agree)
            1.0 = maximum diversity (completely different perspectives)
        """
        if len(consultant_analyses) < 2:
            return 0.0

        # Extract insight sets per consultant
        insight_sets = []
        for analysis in consultant_analyses:
            insights = analysis.get("key_insights", [])
            if insights:
                # Combine all insights into one text blob
                combined = " ".join(insights).lower()
                insight_sets.append(combined)

        if len(insight_sets) < 2:
            return 0.0

        # Compute pairwise similarities
        similarities = []
        for i in range(len(insight_sets)):
            for j in range(i + 1, len(insight_sets)):
                sim = self._compute_similarity(insight_sets[i], insight_sets[j])
                similarities.append(sim)

        # Orthogonality = 1 - average similarity
        avg_similarity = sum(similarities) / len(similarities)
        orthogonality = 1.0 - avg_similarity

        return round(orthogonality, 2)

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute simple word-overlap similarity between two texts.

        Uses Jaccard similarity: |intersection| / |union|
        """
        if not text1 or not text2:
            return 0.0

        # Tokenize (simple word splitting)
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))

        # Remove common stopwords
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from", "as", "is", "was", "are", "were", "be", "been", "being"}
        words1 = words1 - stopwords
        words2 = words2 - stopwords

        if not words1 or not words2:
            return 0.0

        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def identify_convergence(
        self,
        consultant_analyses: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Identify convergent insights (agreed upon by multiple consultants).

        Returns insights that appear in 2+ consultant analyses (with fuzzy matching).
        """
        if len(consultant_analyses) < 2:
            return []

        # Collect all insights
        all_insights = []
        for analysis in consultant_analyses:
            insights = analysis.get("key_insights", [])
            all_insights.extend(insights)

        if not all_insights:
            return []

        # Find insights mentioned by multiple consultants (fuzzy matching)
        convergent = []
        seen = set()

        for i, insight1 in enumerate(all_insights):
            if i in seen:
                continue

            matches = [i]
            for j, insight2 in enumerate(all_insights):
                if i != j and j not in seen:
                    # Fuzzy match (high similarity threshold)
                    if self._compute_similarity(insight1, insight2) > 0.6:
                        matches.append(j)
                        seen.add(j)

            # If insight appears in 2+ analyses, it's convergent
            if len(matches) >= 2:
                convergent.append(insight1)
                seen.update(matches)

        return convergent[:5]  # Limit to top 5

    def identify_divergence(
        self,
        consultant_analyses: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Identify divergent perspectives (areas of disagreement).

        Looks for recommendations that contradict each other.
        """
        if len(consultant_analyses) < 2:
            return []

        # Collect all recommendations
        all_recommendations = []
        for analysis in consultant_analyses:
            recs = analysis.get("recommendations", [])
            all_recommendations.extend(recs)

        if len(all_recommendations) < 2:
            return []

        # Find contradicting recommendations
        divergent = []
        contradiction_words = [
            ("increase", "decrease"),
            ("expand", "contract"),
            ("invest", "divest"),
            ("hire", "downsize"),
            ("accelerate", "slow"),
            ("add", "remove"),
            ("build", "buy"),
        ]

        for rec1 in all_recommendations:
            rec1_lower = rec1.lower()
            for rec2 in all_recommendations:
                if rec1 == rec2:
                    continue

                rec2_lower = rec2.lower()

                # Check for contradicting word pairs
                for word1, word2 in contradiction_words:
                    if word1 in rec1_lower and word2 in rec2_lower:
                        divergent.append(f"Disagreement: '{rec1[:80]}...' vs '{rec2[:80]}...'")
                        break
                    elif word2 in rec1_lower and word1 in rec2_lower:
                        divergent.append(f"Disagreement: '{rec1[:80]}...' vs '{rec2[:80]}...'")
                        break

        # Remove duplicates
        return list(dict.fromkeys(divergent))[:5]

    def _deduplicate_insights(
        self,
        consultant_analyses: List[Dict[str, Any]],
        similarity_threshold: float,
    ) -> tuple[List[Dict[str, Any]], int]:
        """
        Deduplicate similar insights within each consultant's analysis.

        Returns (deduplicated_analyses, dedup_count)
        """
        dedup_count = 0

        for analysis in consultant_analyses:
            insights = analysis.get("key_insights", [])
            if len(insights) < 2:
                continue

            # Deduplicate insights
            unique_insights = []
            seen_indices = set()

            for i, insight1 in enumerate(insights):
                if i in seen_indices:
                    continue

                unique_insights.append(insight1)

                # Mark similar insights as duplicates
                for j, insight2 in enumerate(insights):
                    if i != j and j not in seen_indices:
                        if self._compute_similarity(insight1, insight2) > similarity_threshold:
                            seen_indices.add(j)
                            dedup_count += 1

            analysis["key_insights"] = unique_insights

        return consultant_analyses, dedup_count

    def _generate_minority_report(
        self,
        consultant_analyses: List[Dict[str, Any]],
    ) -> str:
        """
        Generate a minority report when groupthink is detected.

        Provides a contrarian perspective to challenge consensus.
        """
        # Extract common themes
        all_insights = []
        for analysis in consultant_analyses:
            all_insights.extend(analysis.get("key_insights", []))

        if not all_insights:
            return "Groupthink detected: All consultants agree. Consider alternative perspectives."

        # Generate contrarian view
        report = (
            f"⚠️ Low cognitive diversity detected (groupthink risk). "
            f"All {len(consultant_analyses)} consultants converged on similar conclusions. "
            f"Consider: What if the opposite is true? What are we missing? "
            f"What assumptions are we making that could be wrong?"
        )

        return report
