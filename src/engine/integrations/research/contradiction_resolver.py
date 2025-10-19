"""
Operation Synapse: Contradiction Resolution Protocol with Tie-Breaker Searches

Implements formal protocol for handling contradictory evidence in research,
transforming METIS from detection-only to comprehensive resolution system.

Key Features:
- Automatic tie-breaker searches for contradictory claims
- Source credibility weighting and synthesis generation
- Fixed confidence penalty application (0.15 per architectural decision)
- Comprehensive audit trail of contradiction resolution
- Integration with existing Perplexity research infrastructure
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from src.integrations.research.models import ResearchMode
from src.integrations.research.validators import ConsistencyValidator, SourceValidator


@dataclass
class ContradictionResolution:
    """Complete resolution of a detected contradiction"""

    topic: str
    claim_a: str
    claim_b: str
    source_a_info: Dict[str, Any]
    source_b_info: Dict[str, Any]
    tie_breaker_query: str
    tie_breaker_result: str
    synthesis_note: str
    confidence_penalty: float
    credibility_assessment: Dict[str, float]
    resolution_timestamp: str
    contradiction_detected: bool = True


@dataclass
class ContradictionSummary:
    """Summary of all contradictions detected and resolved"""

    total_contradictions: int
    resolved_contradictions: int
    unresolved_contradictions: int
    total_confidence_penalty: float
    resolution_success_rate: float
    most_significant_contradiction: Optional[ContradictionResolution]


class ContradictionResolver:
    """
    Formal protocol for contradiction resolution implementing architectural decisions:

    - One tie-breaker search per contradiction (budget control)
    - Fixed confidence penalty of 0.15 per contradiction
    - Synthesis note generation for both sides of argument
    - Complete audit trail for transparency
    """

    def __init__(self, perplexity_client=None):
        self.logger = logging.getLogger(__name__)
        self.perplexity = perplexity_client

        # Architectural decisions
        self.CONTRADICTION_PENALTY = 0.15  # Fixed penalty per contradiction
        self.TIE_BREAKER_BUDGET = 1  # One search per contradiction
        self.MIN_CONFIDENCE_FLOOR = 0.1  # Minimum confidence after penalties

        # Source credibility validator
        self.source_validator = SourceValidator()
        self.consistency_validator = ConsistencyValidator()

        # Tie-breaker query template (architectural decision)
        self.TIE_BREAKER_TEMPLATE = (
            "Analyze the discrepancy between the following two claims from "
            "different sources regarding {topic}. "
            "Claim A states: {claim_a}. "
            "Claim B states: {claim_b}. "
            "Provide a synthesis, evaluate the relative credibility of the "
            "likely sources, and determine the most probable conclusion."
        )

        self.logger.info("🎯 Contradiction Resolver initialized with formal protocol")

    async def resolve_contradictions(
        self,
        contradictions: List[str],
        sources: List[Dict[str, Any]],
        original_confidence: float = 1.0,
    ) -> Tuple[List[ContradictionResolution], ContradictionSummary]:
        """
        Execute formal contradiction resolution protocol.

        Returns:
            Tuple of (resolved_contradictions, summary)
        """

        if not contradictions:
            return [], ContradictionSummary(0, 0, 0, 0.0, 1.0, None)

        self.logger.info(
            f"🔍 Starting contradiction resolution: {len(contradictions)} contradictions detected"
        )

        resolved_contradictions = []

        for i, contradiction in enumerate(contradictions[: self.TIE_BREAKER_BUDGET]):
            self.logger.info(
                f"🔄 Resolving contradiction {i+1}/{len(contradictions)}: {contradiction[:100]}..."
            )

            try:
                resolution = await self._resolve_single_contradiction(
                    contradiction, sources, i
                )
                resolved_contradictions.append(resolution)

                self.logger.info(
                    f"✅ Contradiction resolved: {resolution.topic} "
                    f"(confidence penalty: {resolution.confidence_penalty})"
                )

            except Exception as e:
                self.logger.error(f"❌ Failed to resolve contradiction {i+1}: {e}")
                # Create unresolved contradiction record
                unresolved = self._create_unresolved_contradiction(
                    contradiction, str(e)
                )
                resolved_contradictions.append(unresolved)

        # Generate summary
        summary = self._generate_contradiction_summary(
            resolved_contradictions, original_confidence
        )

        self.logger.info(
            f"📊 Contradiction resolution complete: {summary.resolved_contradictions}/"
            f"{summary.total_contradictions} resolved "
            f"(success rate: {summary.resolution_success_rate:.1%})"
        )

        return resolved_contradictions, summary

    async def _resolve_single_contradiction(
        self, contradiction: str, sources: List[Dict[str, Any]], index: int
    ) -> ContradictionResolution:
        """Resolve a single contradiction using formal protocol"""

        # Parse contradiction string to extract claims
        topic, claim_a, claim_b, source_a, source_b = self._parse_contradiction(
            contradiction, sources
        )

        # Assess source credibility
        credibility_a = self.source_validator.calculate_source_credibility(source_a)
        credibility_b = self.source_validator.calculate_source_credibility(source_b)

        # Generate tie-breaker query
        tie_breaker_query = self.TIE_BREAKER_TEMPLATE.format(
            topic=topic, claim_a=claim_a, claim_b=claim_b
        )

        # Execute tie-breaker search
        tie_breaker_result = await self._execute_tie_breaker_search(tie_breaker_query)

        # Generate synthesis note
        synthesis_note = self._generate_synthesis_note(
            topic, claim_a, claim_b, credibility_a, credibility_b, tie_breaker_result
        )

        return ContradictionResolution(
            topic=topic,
            claim_a=claim_a,
            claim_b=claim_b,
            source_a_info=self._extract_source_info(source_a),
            source_b_info=self._extract_source_info(source_b),
            tie_breaker_query=tie_breaker_query,
            tie_breaker_result=tie_breaker_result,
            synthesis_note=synthesis_note,
            confidence_penalty=self.CONTRADICTION_PENALTY,
            credibility_assessment={
                "source_a_credibility": credibility_a,
                "source_b_credibility": credibility_b,
                "credibility_gap": abs(credibility_a - credibility_b),
            },
            resolution_timestamp=datetime.utcnow().isoformat(),
            contradiction_detected=True,
        )

    def _parse_contradiction(
        self, contradiction: str, sources: List[Dict[str, Any]]
    ) -> Tuple[str, str, str, Dict[str, Any], Dict[str, Any]]:
        """Parse contradiction string to extract structured information"""

        # Example contradiction: "Contradictory data for B2B SaaS CAC: 15% vs 25%"
        match = re.search(
            r"Contradictory data for (.+?):\s*(.+?)\s*vs\s*(.+?)$", contradiction
        )

        if match:
            topic = match.group(1).strip()
            claim_a = match.group(2).strip()
            claim_b = match.group(3).strip()
        else:
            # Fallback parsing
            parts = contradiction.split(":")
            topic = (
                parts[0].replace("Contradictory data for", "").strip()
                if len(parts) > 1
                else "Unknown topic"
            )
            claims = (
                parts[1].split(" vs ")
                if len(parts) > 1
                else ["Unknown claim A", "Unknown claim B"]
            )
            claim_a = claims[0].strip() if len(claims) > 0 else "Unknown claim A"
            claim_b = claims[1].strip() if len(claims) > 1 else "Unknown claim B"

        # Find sources containing these claims (simplified heuristic)
        source_a = (
            self._find_source_for_claim(claim_a, sources) or sources[0]
            if sources
            else {}
        )
        source_b = (
            self._find_source_for_claim(claim_b, sources) or sources[-1]
            if sources
            else {}
        )

        return topic, claim_a, claim_b, source_a, source_b

    def _find_source_for_claim(
        self, claim: str, sources: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Find source that most likely contains the given claim"""

        # Simple heuristic: find source with highest word overlap
        best_source = None
        best_overlap = 0

        claim_words = set(claim.lower().split())

        for source in sources:
            content = f"{source.get('title', '')} {source.get('content', '')}".lower()
            content_words = set(content.split())

            overlap = len(claim_words.intersection(content_words))
            if overlap > best_overlap:
                best_overlap = overlap
                best_source = source

        return best_source

    async def _execute_tie_breaker_search(self, query: str) -> str:
        """Execute tie-breaker search with specified template"""

        if not self.perplexity:
            # Mock response for demonstration
            return (
                "Based on available sources, both claims reflect different measurement "
                "methodologies and time periods. The discrepancy likely stems from "
                "variations in data collection approaches rather than fundamental "
                "disagreement on the underlying trend."
            )

        try:
            # Use moderate research mode for balance between speed and depth
            result = await self.perplexity.search(query, mode=ResearchMode.MODERATE)
            return result.text if hasattr(result, "text") else str(result)

        except Exception as e:
            self.logger.error(f"❌ Tie-breaker search failed: {e}")
            return f"Tie-breaker search unavailable: {str(e)}"

    def _generate_synthesis_note(
        self,
        topic: str,
        claim_a: str,
        claim_b: str,
        credibility_a: float,
        credibility_b: float,
        tie_breaker_result: str,
    ) -> str:
        """Generate comprehensive synthesis note explaining the contradiction"""

        # Determine which source appears more credible
        if abs(credibility_a - credibility_b) > 0.2:
            credibility_analysis = (
                f"Source credibility analysis suggests {'Source A' if credibility_a > credibility_b else 'Source B'} "
                f"is more reliable (credibility scores: {credibility_a:.2f} vs {credibility_b:.2f})."
            )
        else:
            credibility_analysis = (
                f"Both sources show similar credibility levels "
                f"({credibility_a:.2f} vs {credibility_b:.2f}), requiring additional analysis."
            )

        synthesis = f"""
**Contradiction Analysis: {topic}**

**Conflicting Claims:**
- Source A: {claim_a}
- Source B: {claim_b}

**Credibility Assessment:**
{credibility_analysis}

**Resolution Analysis:**
{tie_breaker_result}

**Synthesis:**
This discrepancy requires careful interpretation. Users should consider the methodology, 
time period, and context of each claim when making decisions based on this data.
        """.strip()

        return synthesis

    def _extract_source_info(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key information from source for audit trail"""

        return {
            "url": source.get("url", "Unknown URL"),
            "title": source.get("title", "Unknown Title"),
            "domain": self.source_validator.extract_domain(source.get("url", "")),
            "credibility_score": self.source_validator.calculate_source_credibility(
                source
            ),
            "content_length": len(source.get("content", "")),
        }

    def _create_unresolved_contradiction(
        self, contradiction: str, error_message: str
    ) -> ContradictionResolution:
        """Create record for unresolved contradiction"""

        return ContradictionResolution(
            topic="Unresolved",
            claim_a=contradiction[:100],
            claim_b="Resolution failed",
            source_a_info={"error": error_message},
            source_b_info={"error": error_message},
            tie_breaker_query="N/A",
            tie_breaker_result=f"Resolution failed: {error_message}",
            synthesis_note=f"Unable to resolve contradiction: {error_message}",
            confidence_penalty=self.CONTRADICTION_PENALTY,
            credibility_assessment={"error": error_message},
            resolution_timestamp=datetime.utcnow().isoformat(),
            contradiction_detected=True,
        )

    def _generate_contradiction_summary(
        self, resolutions: List[ContradictionResolution], original_confidence: float
    ) -> ContradictionSummary:
        """Generate comprehensive summary of contradiction resolution process"""

        total = len(resolutions)
        resolved = sum(
            1 for r in resolutions if "Resolution failed" not in r.synthesis_note
        )
        unresolved = total - resolved

        total_penalty = sum(r.confidence_penalty for r in resolutions)
        success_rate = resolved / total if total > 0 else 1.0

        # Find most significant contradiction (highest credibility gap)
        most_significant = None
        if resolutions:
            most_significant = max(
                resolutions,
                key=lambda r: r.credibility_assessment.get("credibility_gap", 0),
            )

        return ContradictionSummary(
            total_contradictions=total,
            resolved_contradictions=resolved,
            unresolved_contradictions=unresolved,
            total_confidence_penalty=total_penalty,
            resolution_success_rate=success_rate,
            most_significant_contradiction=most_significant,
        )

    def update_fact_pack(
        self,
        fact_pack: Dict[str, Any],
        resolutions: List[ContradictionResolution],
        summary: ContradictionSummary,
    ) -> Dict[str, Any]:
        """
        Update FactPack with contradiction resolution information.

        Architectural decisions applied:
        - Present both conflicting sources in methodological layer
        - Apply fixed confidence penalty
        - Include synthesis conclusions
        """

        # Add contradiction detection flag
        fact_pack["contradiction_detected"] = len(resolutions) > 0

        # Add detailed resolutions
        if "contradictions" not in fact_pack:
            fact_pack["contradictions"] = []

        for resolution in resolutions:
            fact_pack["contradictions"].append(
                {
                    "topic": resolution.topic,
                    "conflicting_claims": [resolution.claim_a, resolution.claim_b],
                    "source_information": {
                        "source_a": resolution.source_a_info,
                        "source_b": resolution.source_b_info,
                    },
                    "credibility_assessment": resolution.credibility_assessment,
                    "synthesis": resolution.synthesis_note,
                    "tie_breaker_evidence": resolution.tie_breaker_result,
                    "resolution_timestamp": resolution.resolution_timestamp,
                }
            )

        # Apply confidence penalty (architectural decision: max 0, original - penalty)
        original_confidence = fact_pack.get("confidence_score", 1.0)
        new_confidence = max(
            self.MIN_CONFIDENCE_FLOOR,
            original_confidence - summary.total_confidence_penalty,
        )

        fact_pack["confidence_score"] = new_confidence
        fact_pack["confidence_penalty_applied"] = summary.total_confidence_penalty
        fact_pack["confidence_penalty_reason"] = (
            f"{summary.total_contradictions} contradictions detected"
        )

        # Add resolution metadata
        fact_pack["contradiction_resolution_summary"] = {
            "total_contradictions": summary.total_contradictions,
            "resolution_success_rate": summary.resolution_success_rate,
            "most_significant_topic": (
                summary.most_significant_contradiction.topic
                if summary.most_significant_contradiction
                else None
            ),
        }

        self.logger.info(
            f"📊 FactPack updated: {summary.total_contradictions} contradictions, "
            f"confidence: {original_confidence:.2f} → {new_confidence:.2f} "
            f"(penalty: {summary.total_confidence_penalty:.2f})"
        )

        return fact_pack


# Factory function for dependency injection
def create_contradiction_resolver(perplexity_client=None) -> ContradictionResolver:
    """Factory function to create configured contradiction resolver"""
    return ContradictionResolver(perplexity_client)
