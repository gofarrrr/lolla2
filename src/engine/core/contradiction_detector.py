"""
Contradiction Detection Pre-flight System
Operation Crystal Day 1 - Detect logical contradictions in user requests before engagement creation
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from src.integrations.claude_client import ClaudeClient


@dataclass
class Contradiction:
    """Represents a detected contradiction"""

    type: str  # "logical", "temporal", "resource", "constraint"
    description: str
    severity: str  # "high", "medium", "low"
    suggested_resolution: Optional[str] = None


@dataclass
class ContradictionResult:
    """Result of contradiction detection analysis"""

    has_contradictions: bool
    contradictions: List[Contradiction]
    analysis_time_ms: int
    confidence_level: str  # "high", "medium", "low"
    blocked: bool  # Whether to block engagement creation


class ContradictionDetector:
    """
    Pre-flight contradiction detection using Claude Sonnet 3.5

    Analyzes user queries for logical inconsistencies, impossible constraints,
    and contradictory requirements before expensive cognitive processing begins.
    """

    def __init__(self, claude_client: Optional[ClaudeClient] = None):
        self.claude_client = claude_client or ClaudeClient()
        self.logger = logging.getLogger(__name__)

        # Configuration per user decisions
        self.max_latency_ms = 2000  # <2 second requirement
        self.timeout_seconds = 2.5  # Slightly higher timeout for safety

        self.logger.info("🚨 ContradictionDetector initialized (max 2s latency)")

    async def detect_contradictions(
        self,
        query: str,
        business_context: Optional[Dict[str, Any]] = None,
        clarifications: Optional[List[Dict[str, str]]] = None,
    ) -> ContradictionResult:
        """
        Analyze query for contradictions before engagement creation.

        Args:
            query: The user's problem statement/request
            business_context: Optional business context from previous interactions
            clarifications: Optional clarification responses from HITL flow

        Returns:
            ContradictionResult indicating if contradictions were found
        """
        start_time = time.perf_counter()

        self.logger.info(f"🔍 Analyzing query for contradictions: '{query[:100]}...'")

        try:
            # Create comprehensive analysis prompt
            analysis_prompt = self._build_contradiction_analysis_prompt(
                query, business_context, clarifications
            )

            # Call Claude Sonnet 3.5 with timeout
            response = await asyncio.wait_for(
                self.claude_client.generate_response(
                    prompt=analysis_prompt,
                    temperature=0.1,  # Very low for consistent logical analysis
                    max_tokens=1500,
                ),
                timeout=self.timeout_seconds,
            )

            # Parse structured response
            result = self._parse_contradiction_response(response, start_time)

            # Log results
            analysis_time = result.analysis_time_ms
            if result.has_contradictions:
                self.logger.warning(
                    f"❌ {len(result.contradictions)} contradictions detected in {analysis_time}ms"
                )
                for contradiction in result.contradictions:
                    self.logger.warning(
                        f"  - {contradiction.type.upper()}: {contradiction.description}"
                    )
            else:
                self.logger.info(f"✅ No contradictions detected in {analysis_time}ms")

            return result

        except asyncio.TimeoutError:
            analysis_time = int((time.perf_counter() - start_time) * 1000)
            self.logger.error(
                f"⏰ Contradiction detection timeout after {analysis_time}ms"
            )

            return ContradictionResult(
                has_contradictions=False,  # Don't block on timeout
                contradictions=[],
                analysis_time_ms=analysis_time,
                confidence_level="low",
                blocked=False,
            )

        except Exception as e:
            analysis_time = int((time.perf_counter() - start_time) * 1000)
            self.logger.error(f"❌ Contradiction detection failed: {e}")

            return ContradictionResult(
                has_contradictions=False,  # Don't block on error
                contradictions=[],
                analysis_time_ms=analysis_time,
                confidence_level="low",
                blocked=False,
            )

    def _build_contradiction_analysis_prompt(
        self,
        query: str,
        business_context: Optional[Dict[str, Any]],
        clarifications: Optional[List[Dict[str, str]]],
    ) -> str:
        """Build the contradiction analysis prompt for Claude"""

        context_str = ""
        if business_context:
            context_str = (
                f"\nBUSINESS CONTEXT:\n{json.dumps(business_context, indent=2)}"
            )

        clarification_str = ""
        if clarifications:
            clarification_str = "\nCLARIFICATIONS PROVIDED:\n"
            for clarification in clarifications:
                clarification_str += f"• Q: {clarification.get('question', 'N/A')}\n"
                clarification_str += f"  A: {clarification.get('response', 'N/A')}\n"

        return f"""
You are a logical consistency analyzer for business strategy requests. Your job is to detect logical contradictions, impossible constraints, and inconsistent requirements that would make a request impossible to fulfill.

USER REQUEST TO ANALYZE:
{query}
{context_str}
{clarification_str}

ANALYZE FOR THESE TYPES OF CONTRADICTIONS:

1. **LOGICAL CONTRADICTIONS**: Statements that cannot both be true
   - "Cut costs by 50% AND increase headcount by 30%"
   - "Launch in Q1 AND take 12 months to develop"

2. **TEMPORAL CONTRADICTIONS**: Impossible timelines or scheduling conflicts
   - "Complete in 3 months with 12-month dependencies"
   - "Launch before development starts"

3. **RESOURCE CONTRADICTIONS**: Impossible resource allocations
   - "Zero budget but hire 10 consultants" 
   - "No team but complete complex project"

4. **CONSTRAINT CONTRADICTIONS**: Mutually exclusive constraints
   - "Must be cheap, fast, AND perfect quality"
   - "Cannot change anything but need different results"

INSTRUCTIONS:
- Focus ONLY on clear, logical impossibilities
- Ignore challenging but achievable goals
- Do not flag ambitious or difficult requests unless they are logically impossible
- Be precise about WHY something is contradictory

RESPOND IN EXACT JSON FORMAT:
{{
  "has_contradictions": boolean,
  "contradictions": [
    {{
      "type": "logical|temporal|resource|constraint",
      "description": "Clear explanation of the contradiction",
      "severity": "high|medium|low"
    }}
  ],
  "confidence_level": "high|medium|low",
  "analysis_summary": "Brief summary of analysis"
}}

Focus on contradictions that would make the request impossible to fulfill, not just difficult.
"""

    def _parse_contradiction_response(
        self, response: str, start_time: float
    ) -> ContradictionResult:
        """Parse Claude's response into ContradictionResult"""
        analysis_time = int((time.perf_counter() - start_time) * 1000)

        try:
            # Extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")

            json_str = response[json_start:json_end]
            data = json.loads(json_str)

            # Parse contradictions
            contradictions = []
            for contradiction_data in data.get("contradictions", []):
                contradiction = Contradiction(
                    type=contradiction_data["type"],
                    description=contradiction_data["description"],
                    severity=contradiction_data["severity"],
                )
                contradictions.append(contradiction)

            has_contradictions = data.get("has_contradictions", False)
            confidence_level = data.get("confidence_level", "medium")

            # Determine if we should block engagement creation
            # Block if we have high-severity contradictions with high confidence
            should_block = (
                has_contradictions
                and confidence_level == "high"
                and any(c.severity == "high" for c in contradictions)
            )

            return ContradictionResult(
                has_contradictions=has_contradictions,
                contradictions=contradictions,
                analysis_time_ms=analysis_time,
                confidence_level=confidence_level,
                blocked=should_block,
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.error(f"Failed to parse contradiction response: {e}")

            # Fallback: Check for obvious contradiction keywords
            response_lower = response.lower()
            contradiction_keywords = [
                "contradiction",
                "impossible",
                "mutually exclusive",
                "cannot both",
                "inconsistent",
                "conflicting",
            ]

            has_keyword_contradiction = any(
                keyword in response_lower for keyword in contradiction_keywords
            )

            if has_keyword_contradiction:
                fallback_contradiction = Contradiction(
                    type="logical",
                    description="Potential contradiction detected (parsing failed)",
                    severity="medium",
                )

                return ContradictionResult(
                    has_contradictions=True,
                    contradictions=[fallback_contradiction],
                    analysis_time_ms=analysis_time,
                    confidence_level="low",
                    blocked=False,  # Don't block on parsing failure
                )

            # Safe fallback - no contradictions
            return ContradictionResult(
                has_contradictions=False,
                contradictions=[],
                analysis_time_ms=analysis_time,
                confidence_level="low",
                blocked=False,
            )

    def get_contradiction_summary(self, result: ContradictionResult) -> Dict[str, Any]:
        """Generate a user-friendly contradiction summary"""
        return {
            "blocked": result.blocked,
            "contradiction_count": len(result.contradictions),
            "analysis_time_ms": result.analysis_time_ms,
            "confidence": result.confidence_level,
            "contradictions": [
                {"type": c.type, "description": c.description, "severity": c.severity}
                for c in result.contradictions
            ],
            "recommendation": (
                "Please resolve the contradictions before proceeding"
                if result.blocked
                else "No blocking contradictions detected"
            ),
        }


# Factory function for easy instantiation
def create_contradiction_detector(
    claude_client: Optional[ClaudeClient] = None,
) -> ContradictionDetector:
    """Create and configure a ContradictionDetector instance."""
    return ContradictionDetector(claude_client)
