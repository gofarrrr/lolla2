"""
Autonomous ODR Client with self-triggering research capabilities
"""

import logging
from typing import Dict, List, Any

from .agent import ODRResearchAgent
from .config import ODRConfiguration

logger = logging.getLogger(__name__)


class AutonomousODRClient(ODRResearchAgent):
    """Enhanced ODR Research Agent with autonomous capabilities"""

    def __init__(self, config: ODRConfiguration):
        super().__init__(config)
        # Autonomous capabilities
        self.confidence_threshold = 0.7  # Trigger research when confidence < threshold
        self.context_gap_detector = None  # Will be initialized after ContextGapDetector
        self.recursive_depth_limit = 3
        self.research_cache = {}  # Simple cache to avoid duplicate research

        # Performance tracking
        self.autonomous_triggers = 0
        self.recursive_calls = 0
        self.context_gaps_detected = 0

    async def analyze_context_gaps(self, context: Dict[str, Any]) -> List[str]:
        """
        Identify missing information that could improve analysis quality.

        Returns list of gap descriptions that need research.
        """
        gaps = []

        try:
            # Check for missing key context elements
            required_context = [
                "industry",
                "problem_type",
                "stakeholders",
                "constraints",
            ]
            missing_basic = [k for k in required_context if not context.get(k)]

            if missing_basic:
                gaps.extend([f"Missing {field} context" for field in missing_basic])

            # For now, return basic gaps - full LLM implementation will be added later
            self.context_gaps_detected = len(gaps)
            logger.info(f"🔍 Detected {len(gaps)} context gaps: {gaps[:3]}...")

            return gaps[:5]  # Limit to top 5 gaps

        except Exception as e:
            logger.error(f"❌ Context gap analysis failed: {e}")
            return []

    async def should_research(
        self,
        confidence: float,
        context: Dict[str, Any],
        problem_complexity: str = "moderate",
    ) -> bool:
        """
        Determine if autonomous research should be triggered.

        Args:
            confidence: Current confidence level (0.0-1.0)
            context: Current analysis context
            problem_complexity: Problem complexity level

        Returns:
            True if research should be triggered
        """
        triggers = []

        # Confidence threshold trigger
        if confidence < self.confidence_threshold:
            triggers.append(
                f"Low confidence: {confidence:.2f} < {self.confidence_threshold}"
            )

        # Context gap trigger
        gaps = await self.analyze_context_gaps(context)
        if len(gaps) > 2:
            triggers.append(f"Context gaps detected: {len(gaps)}")

        # Problem complexity trigger
        if problem_complexity in ["complex", "strategic"] and confidence < 0.8:
            triggers.append("Complex problem requires higher confidence")

        # Novel problem trigger (no recent similar research)
        problem_key = self._create_problem_key(context)
        if problem_key not in self.research_cache:
            triggers.append("Novel problem type")

        should_trigger = len(triggers) > 0
        if should_trigger:
            self.autonomous_triggers += 1
            logger.info(f"🤖 Autonomous research triggered: {', '.join(triggers)}")

        return should_trigger

    async def recursive_deepen(
        self,
        topic: str,
        context: Dict[str, Any],
        current_depth: int = 0,
        max_depth: int = None,
    ) -> Dict[str, Any]:
        """
        Recursively deepen research on a topic until confidence threshold is met.

        Args:
            topic: Research topic
            context: Analysis context
            current_depth: Current recursion depth
            max_depth: Maximum recursion depth (defaults to self.recursive_depth_limit)

        Returns:
            Enhanced research results
        """
        if max_depth is None:
            max_depth = self.recursive_depth_limit

        if current_depth >= max_depth:
            logger.warning(
                f"⚠️ Max recursive depth {max_depth} reached for topic: {topic[:50]}"
            )
            return {"findings": [], "confidence": 0.5, "depth_limited": True}

        logger.info(
            f"🔄 Recursive research depth {current_depth + 1}/{max_depth} for: {topic[:50]}"
        )
        self.recursive_calls += 1

        # For now, return mock results - will be implemented once conduct_research is available
        return {
            "findings": [f"Mock finding for {topic[:30]}"],
            "overall_confidence": 0.7,
            "depth": current_depth + 1,
        }

    async def validate_and_crossref(self, findings: List[Dict[str, Any]]) -> float:
        """
        Self-validate findings through cross-referencing and consistency checks.

        Returns confidence score (0.0-1.0)
        """
        if not findings:
            return 0.0

        try:
            validation_scores = []

            # Source diversity validation
            sources = set()
            for finding in findings:
                finding_sources = finding.get("sources", [])
                for source in finding_sources:
                    sources.add(source.get("domain", "unknown"))

            source_diversity = min(len(sources) / 5.0, 1.0)  # Max score at 5+ sources
            validation_scores.append(source_diversity)

            # Default validation scores for now
            validation_scores.append(0.7)  # Default claim consistency

            # Quantitative data validation
            has_quantitative = any(
                finding.get("quantitative_data") for finding in findings
            )
            quantitative_score = 0.8 if has_quantitative else 0.5
            validation_scores.append(quantitative_score)

            overall_confidence = sum(validation_scores) / len(validation_scores)
            logger.info(f"🔍 Validation complete. Confidence: {overall_confidence:.2f}")

            return overall_confidence

        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            return 0.5

    def _create_problem_key(self, context: Dict[str, Any]) -> str:
        """Create a key for caching similar problems"""
        key_components = [
            context.get("industry", "general"),
            context.get("problem_type", "analysis"),
            str(len(context.get("constraints", []))),
        ]
        return "|".join(key_components)

    def get_autonomous_stats(self) -> Dict[str, int]:
        """Get statistics about autonomous behavior"""
        return {
            "autonomous_triggers": self.autonomous_triggers,
            "recursive_calls": self.recursive_calls,
            "context_gaps_detected": self.context_gaps_detected,
            "cache_size": len(self.research_cache),
        }
