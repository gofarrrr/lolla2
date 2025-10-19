"""
Task Classification Service - OPERATION ADAPTIVE ORCHESTRATION
============================================================

MISSION: Intelligent task classification to enable domain-aware team composition.

This service analyzes user queries to determine:
1. Primary domain (strategy, finance, creative, operations, etc.)
2. Task type (analytical vs ideation)

This enables the system to adapt team composition strategies based on the
cognitive diversity research findings from Operation Cognitive Diversity.

PRINCIPLE: "Classification drives optimization"
"""

import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime

# LLM Manager for classification calls
from src.engine.core.llm_manager import LLMManager
from src.core.unified_context_stream import UnifiedContextStream, ContextEventType

logger = logging.getLogger(__name__)


class TaskClassificationService:
    """
    Service for analyzing user queries to determine optimal team composition strategy.

    Uses lightweight LLM calls to classify tasks along two dimensions:
    - Primary domain (strategy, finance, creative, operations, technology, etc.)
    - Task type (analytical vs ideation)
    """

    def __init__(self, context_stream: Optional[UnifiedContextStream] = None):
        """Initialize task classification service."""
        self.context_stream = context_stream or UnifiedContextStream(max_events=1000)
        self._classification_cache = {}  # Simple in-memory cache

        logger.info(
            "🎯 Task Classification Service initialized for adaptive orchestration"
        )

    async def classify_task(self, user_query: str) -> Dict[str, Any]:
        """
        Analyze user query and return classification for adaptive team composition.

        Args:
            user_query: The original user query to classify

        Returns:
            Dictionary with classification results:
            {
                "primary_domain": str,     # e.g., "strategy", "finance", "creative"
                "task_type": str,          # "analytical" or "ideation"
                "confidence": float,       # 0-1 confidence score
                "reasoning": str,          # Brief explanation of classification
                "complexity_level": str,   # "low", "medium", "high"
                "requires_creativity": bool, # True if creative thinking needed
                "classification_metadata": dict  # Additional metadata
            }
        """
        start_time = datetime.now()

        # Check cache first (simple hash-based caching)
        query_hash = str(hash(user_query.strip().lower()))
        if query_hash in self._classification_cache:
            cached_result = self._classification_cache[query_hash]
            logger.info(f"🎯 Using cached classification for query hash: {query_hash}")
            return cached_result

        try:
            self.context_stream.add_event(
                ContextEventType.TASK_CLASSIFICATION_STARTED,
                {
                    "query_preview": (
                        user_query[:200] + "..."
                        if len(user_query) > 200
                        else user_query
                    ),
                    "timestamp": start_time.isoformat(),
                    "service": "TaskClassificationService",
                },
            )

            # Construct classification prompt
            classification_prompt = self._build_classification_prompt(user_query)

            # Initialize LLM Manager for classification
            llm_manager = LLMManager(context_stream=self.context_stream)

            # Execute lightweight LLM call for classification
            response = await llm_manager.execute_completion(
                prompt=classification_prompt,
                system_prompt=self._get_classification_system_prompt(),
                temperature=0.1,  # Low temperature for consistent classification
                max_tokens=300,  # Short response for efficiency
                timeout=30,  # Fast timeout for classification
            )

            if not response or not response.raw_text:
                raise RuntimeError("LLM classification call failed - no response")

            # Parse JSON response
            classification_result = self._parse_classification_response(
                response.raw_text
            )

            # Validate and enrich classification result
            classification_result = self._validate_and_enrich_classification(
                classification_result, user_query
            )

            # Cache the result
            self._classification_cache[query_hash] = classification_result

            # Log successful classification
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.context_stream.add_event(
                ContextEventType.TASK_CLASSIFICATION_COMPLETE,
                {
                    "classification_result": classification_result,
                    "duration_ms": duration_ms,
                    "provider": getattr(response, "provider", "unknown"),
                    "tokens_used": getattr(response, "total_tokens", 0),
                    "timestamp": datetime.now().isoformat(),
                },
            )

            logger.info(
                f"🎯 Task classified: {classification_result['primary_domain']} | {classification_result['task_type']} (confidence: {classification_result['confidence']:.2f})"
            )
            return classification_result

        except Exception as e:
            # Log error and return conservative fallback
            error_msg = f"Task classification failed: {e}"
            logger.error(f"❌ {error_msg}")

            self.context_stream.add_event(
                ContextEventType.ERROR_OCCURRED,
                {
                    "error": error_msg,
                    "fallback_classification": "strategy_analytical",
                    "timestamp": datetime.now().isoformat(),
                },
            )

            # Return conservative fallback classification
            return self._get_fallback_classification(user_query, error_msg)

    def _build_classification_prompt(self, user_query: str) -> str:
        """Build the classification prompt for the LLM."""
        return f"""Analyze the following business query and classify it along two key dimensions for optimal AI team composition.

USER QUERY:
{user_query}

Return ONLY a JSON object with these exact keys:
{{
    "primary_domain": "[strategy|finance|operations|creative|technology|marketing|hr|legal]",
    "task_type": "[analytical|ideation]", 
    "confidence": [0.0-1.0],
    "reasoning": "[brief 1-sentence explanation]",
    "complexity_level": "[low|medium|high]",
    "requires_creativity": [true|false]
}}

CLASSIFICATION GUIDELINES:
- primary_domain: The main business area this query focuses on
- task_type: "analytical" for data-driven analysis/evaluation, "ideation" for creative/brainstorming tasks
- confidence: Your confidence in this classification (0.0-1.0)
- reasoning: Brief explanation of why you classified it this way
- complexity_level: Based on scope, stakeholders, and decision complexity
- requires_creativity: Whether this needs creative thinking vs pure analysis

EXAMPLES:
- "Analyze our Q3 financial performance" → strategy, analytical
- "Brainstorm viral marketing campaign ideas" → creative, ideation  
- "Evaluate acquisition targets in fintech" → finance, analytical
- "Design innovative customer onboarding experience" → operations, ideation
"""

    def _get_classification_system_prompt(self) -> str:
        """Get the system prompt for task classification."""
        return """You are an expert business analyst and task classification specialist. Your role is to analyze business queries and classify them to enable optimal AI consultant team composition.

You must return valid JSON only. No additional text or explanation outside the JSON structure.

Focus on the fundamental nature of what the user is asking for:
- Analytical tasks require systematic analysis, data evaluation, and logical reasoning
- Ideation tasks require creative thinking, brainstorming, and innovative solutions

Be decisive but honest about your confidence level."""

    def _parse_classification_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the LLM response and extract classification data."""
        try:
            # Clean up response text
            response_text = response_text.strip()

            # Find JSON block
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif "{" in response_text:
                # Extract JSON from first { to last }
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                response_text = response_text[start:end]

            # Parse JSON
            parsed_result = json.loads(response_text)

            # Validate required keys
            required_keys = ["primary_domain", "task_type", "confidence", "reasoning"]
            for key in required_keys:
                if key not in parsed_result:
                    raise ValueError(f"Missing required key: {key}")

            return parsed_result

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"⚠️ Failed to parse classification response: {e}")
            logger.warning(f"Raw response: {response_text}")

            # Attempt simple regex extraction as fallback
            return self._extract_classification_with_regex(response_text)

    def _extract_classification_with_regex(self, response_text: str) -> Dict[str, Any]:
        """Fallback regex extraction for classification data."""
        import re

        # Extract primary domain
        domain_match = re.search(
            r'(?:primary_domain|domain)["\']?:\s*["\']?(\w+)',
            response_text,
            re.IGNORECASE,
        )
        primary_domain = domain_match.group(1) if domain_match else "strategy"

        # Extract task type
        task_match = re.search(
            r'(?:task_type|type)["\']?:\s*["\']?(\w+)', response_text, re.IGNORECASE
        )
        task_type = task_match.group(1) if task_match else "analytical"

        # Extract confidence
        conf_match = re.search(
            r'confidence["\']?:\s*([0-9.]+)', response_text, re.IGNORECASE
        )
        confidence = float(conf_match.group(1)) if conf_match else 0.7

        logger.info(
            f"🔧 Regex fallback extraction: {primary_domain}, {task_type}, {confidence}"
        )

        return {
            "primary_domain": primary_domain,
            "task_type": task_type,
            "confidence": confidence,
            "reasoning": "Extracted via regex fallback due to parsing issues",
            "complexity_level": "medium",
            "requires_creativity": task_type == "ideation",
        }

    def _validate_and_enrich_classification(
        self, classification: Dict[str, Any], user_query: str
    ) -> Dict[str, Any]:
        """Validate classification result and add enrichments."""

        # Validate primary_domain
        valid_domains = [
            "strategy",
            "finance",
            "operations",
            "creative",
            "technology",
            "marketing",
            "hr",
            "legal",
        ]
        if classification.get("primary_domain") not in valid_domains:
            classification["primary_domain"] = "strategy"  # Default fallback

        # Validate task_type
        if classification.get("task_type") not in ["analytical", "ideation"]:
            classification["task_type"] = "analytical"  # Default fallback

        # Ensure confidence is in valid range
        confidence = classification.get("confidence", 0.7)
        classification["confidence"] = max(0.0, min(1.0, float(confidence)))

        # Add enrichment metadata
        classification["classification_metadata"] = {
            "query_length": len(user_query),
            "query_word_count": len(user_query.split()),
            "classification_timestamp": datetime.now().isoformat(),
            "service_version": "1.0",
            "adaptive_orchestration": True,
        }

        # Set defaults for missing optional fields
        if "complexity_level" not in classification:
            classification["complexity_level"] = "medium"

        if "requires_creativity" not in classification:
            classification["requires_creativity"] = (
                classification["task_type"] == "ideation"
            )

        # Add reasoning if missing
        if not classification.get("reasoning"):
            classification["reasoning"] = (
                f"Classified as {classification['primary_domain']} {classification['task_type']} task based on query content analysis"
            )

        return classification

    def _get_fallback_classification(
        self, user_query: str, error_msg: str
    ) -> Dict[str, Any]:
        """Return conservative fallback classification when LLM classification fails."""

        # Simple keyword-based fallback logic
        query_lower = user_query.lower()

        # Detect creative/ideation keywords
        creative_keywords = [
            "brainstorm",
            "creative",
            "innovative",
            "design",
            "ideate",
            "concept",
            "campaign",
            "brand",
        ]
        is_creative = any(keyword in query_lower for keyword in creative_keywords)

        # Detect domain keywords
        domain = "strategy"  # Default
        if any(
            word in query_lower
            for word in ["financial", "finance", "budget", "cost", "revenue"]
        ):
            domain = "finance"
        elif any(
            word in query_lower
            for word in ["marketing", "campaign", "brand", "promotion"]
        ):
            domain = "marketing"
        elif any(
            word in query_lower
            for word in ["operation", "process", "workflow", "efficiency"]
        ):
            domain = "operations"
        elif any(
            word in query_lower
            for word in ["creative", "design", "innovation", "ideation"]
        ):
            domain = "creative"

        return {
            "primary_domain": domain,
            "task_type": "ideation" if is_creative else "analytical",
            "confidence": 0.3,  # Low confidence for fallback
            "reasoning": f"Fallback classification due to LLM error: {error_msg}",
            "complexity_level": "medium",
            "requires_creativity": is_creative,
            "classification_metadata": {
                "fallback": True,
                "error": error_msg,
                "method": "keyword_based_fallback",
                "timestamp": datetime.now().isoformat(),
            },
        }

    def get_classification_stats(self) -> Dict[str, Any]:
        """Get statistics about classification performance."""
        return {
            "total_classifications": len(self._classification_cache),
            "cache_hit_rate": "N/A",  # Could implement if needed
            "service_uptime": "Active",
            "last_classification": (
                datetime.now().isoformat() if self._classification_cache else None
            ),
        }


# Singleton instance for module-level access
_task_classifier_instance = None


def get_task_classifier(
    context_stream: Optional[UnifiedContextStream] = None,
) -> TaskClassificationService:
    """Get or create the singleton task classifier instance."""
    global _task_classifier_instance
    if _task_classifier_instance is None:
        _task_classifier_instance = TaskClassificationService(context_stream)
    return _task_classifier_instance
