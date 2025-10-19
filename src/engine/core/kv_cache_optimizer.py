"""
Operation Synapse Sprint 1.5: KV-Cache Optimizer
F004: Implementation of Manus.im KV-cache optimization insights for 10x cost reduction

This module implements the KV-cache optimization strategies discovered by Manus.im,
achieving dramatic cost reductions through stable prompt prefixes, cache hit optimization,
and intelligent context management.

Key Insights from Manus.im:
- Cached tokens can be 10x cheaper (0.30 vs 3 USD/MTok with Claude Sonnet)
- KV-cache hit rate is the most critical metric for agent performance
- Stable, append-only, deterministic context design is essential
- "Mask, don't remove" pattern for tool availability
"""

import hashlib
import logging
from typing import Dict, List, Any
from datetime import datetime
from dataclasses import dataclass, field
import re

from src.config import CognitiveEngineSettings
from src.interfaces.context_intelligence_interface import IKVCacheOptimizer


@dataclass
class CacheHitMetrics:
    """Metrics for tracking KV-cache performance"""

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_tokens_saved: int = 0
    cost_savings_usd: float = 0.0
    average_hit_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def calculate_hit_rate(self) -> float:
        """Calculate current cache hit rate"""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    def add_hit(self, tokens_saved: int, cost_saved_usd: float):
        """Record a cache hit"""
        self.cache_hits += 1
        self.total_requests += 1
        self.total_tokens_saved += tokens_saved
        self.cost_savings_usd += cost_saved_usd
        self.average_hit_rate = self.calculate_hit_rate()
        self.last_updated = datetime.utcnow()

    def add_miss(self):
        """Record a cache miss"""
        self.cache_misses += 1
        self.total_requests += 1
        self.average_hit_rate = self.calculate_hit_rate()
        self.last_updated = datetime.utcnow()


@dataclass
class StablePromptTemplate:
    """Template for creating stable, cacheable prompts"""

    prefix: str
    dynamic_sections: Dict[str, str] = field(default_factory=dict)
    suffix: str = ""
    cache_key_hash: str = ""

    def __post_init__(self):
        """Generate cache key hash"""
        stable_parts = f"{self.prefix}|{self.suffix}"
        self.cache_key_hash = hashlib.md5(stable_parts.encode()).hexdigest()[:16]

    def render(self, **dynamic_values) -> str:
        """Render template with dynamic values"""
        rendered_sections = {}
        for key, template in self.dynamic_sections.items():
            if key in dynamic_values:
                rendered_sections[key] = template.format(**{key: dynamic_values[key]})
            else:
                rendered_sections[key] = ""

        # Combine prefix + dynamic sections + suffix
        full_prompt = self.prefix
        for section in rendered_sections.values():
            if section:
                full_prompt += f"\n{section}"
        if self.suffix:
            full_prompt += f"\n{self.suffix}"

        return full_prompt


class KVCacheOptimizer(IKVCacheOptimizer):
    """
    KV-Cache Optimizer implementing Manus.im insights for 10x cost reduction

    Key Strategies:
    1. Stable prompt prefixes for consistent caching
    2. Deterministic serialization to avoid cache breaks
    3. Append-only context updates
    4. Intelligent prompt structure optimization
    5. Cache hit prediction and optimization
    """

    def __init__(self, settings: CognitiveEngineSettings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)

        # Cache hit tracking
        self.metrics = CacheHitMetrics()

        # Prompt templates for stable caching
        self.stable_templates: Dict[str, StablePromptTemplate] = {}

        # Cache history for hit prediction
        self.cache_history: List[str] = []
        self.max_history_size = 1000

        # Cost calculation constants (based on Claude Sonnet pricing)
        self.CACHED_TOKEN_COST_PER_MTOK = 0.30  # USD per million tokens (cached)
        self.UNCACHED_TOKEN_COST_PER_MTOK = 3.0  # USD per million tokens (uncached)

        self.logger.info("🚀 KVCacheOptimizer initialized with Manus.im insights")

        # Initialize default templates
        self._initialize_default_templates()

    def _initialize_default_templates(self):
        """Initialize default stable prompt templates"""

        # Template for cognitive analysis
        self.stable_templates["cognitive_analysis"] = StablePromptTemplate(
            prefix="""<analysis_framework>
You are an expert cognitive analyst applying systematic mental models to complex problems.
Your task is to provide rigorous analysis using the specified mental models.

Guidelines:
- Apply each mental model systematically
- Provide evidence for your reasoning
- Maintain logical consistency
- Show your thinking process in <thinking> tags
</analysis_framework>""",
            dynamic_sections={
                "problem": "\nPROBLEM STATEMENT:\n{problem}",
                "context": "\nCONTEXT:\n{context}",
                "models": "\nMENTAL MODELS TO APPLY:\n{models}",
                "research": "\nRELEVANT RESEARCH:\n{research}",
            },
            suffix="\nProvide your systematic analysis with reasoning for each mental model applied.",
        )

        # Template for context intelligence
        self.stable_templates["context_intelligence"] = StablePromptTemplate(
            prefix="""<context_analysis_framework>
You are analyzing context elements for relevance and classification.
Use systematic evaluation to determine context utility.

Classification Types:
- IMMEDIATE: Current request and user intent
- SESSION: Conversation history and established context
- DOMAIN: Relevant knowledge and expertise area  
- PROCEDURAL: How-to knowledge and methodologies
- TEMPORAL: Time-sensitive information and trends
- RELATIONAL: Connections and dependencies
</context_analysis_framework>""",
            dynamic_sections={
                "contexts": "\nCONTEXT ELEMENTS TO ANALYZE:\n{contexts}",
                "query": "\nCURRENT QUERY:\n{query}",
                "criteria": "\nEVALUATION CRITERIA:\n{criteria}",
            },
            suffix="\nProvide classification and relevance scoring for each context element.",
        )

    def create_stable_prompt_prefix(self, base_context: str) -> str:
        """
        Create stable prompt prefix for KV-cache optimization

        Implements Manus.im insight: stable, deterministic prefixes maximize cache hits
        """
        if not self.settings.STABLE_PROMPT_PREFIX:
            return base_context

        try:
            # Remove timestamps and dynamic elements that break caching
            cleaned_context = self._remove_cache_breakers(base_context)

            # Create deterministic structure
            stable_prefix = f"""<stable_context>
{cleaned_context}
</stable_context>

<processing_instructions>
- Maintain consistency in reasoning approach
- Use systematic mental model application
- Provide evidence-based analysis
- Show thinking process in <thinking> tags
</processing_instructions>

"""

            # Track for cache optimization
            self._update_cache_history(stable_prefix)

            return stable_prefix

        except Exception as e:
            self.logger.error(f"❌ Failed to create stable prompt prefix: {e}")
            return base_context

    def _remove_cache_breakers(self, content: str) -> str:
        """Remove dynamic elements that break KV-cache consistency"""

        # Remove timestamps
        timestamp_patterns = [
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",  # ISO timestamps
            r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}",  # Standard timestamps
            r"at \d{4}-\d{2}-\d{2}",  # Date references
            r"on \d{1,2}/\d{1,2}/\d{4}",  # US dates
            r"timestamp: [^\n]+",  # Explicit timestamps
            r"created_at: [^\n]+",  # Database timestamps
            r"last_updated: [^\n]+",  # Update timestamps
        ]

        cleaned_content = content
        for pattern in timestamp_patterns:
            cleaned_content = re.sub(
                pattern, "[TIMESTAMP_REMOVED]", cleaned_content, flags=re.IGNORECASE
            )

        # Remove session IDs and unique identifiers
        id_patterns = [
            r"session[_-]id: [^\n]+",
            r"engagement[_-]id: [^\n]+",
            r"request[_-]id: [^\n]+",
            r"uuid: [a-f0-9-]+",
        ]

        for pattern in id_patterns:
            cleaned_content = re.sub(
                pattern, "[ID_REMOVED]", cleaned_content, flags=re.IGNORECASE
            )

        # Remove performance metrics that change frequently
        metrics_patterns = [
            r"processing_time: [^\n]+",
            r"latency: [^\n]+",
            r"tokens_used: [^\n]+",
            r"cost: [^\n]+",
        ]

        for pattern in metrics_patterns:
            cleaned_content = re.sub(
                pattern, "[METRIC_REMOVED]", cleaned_content, flags=re.IGNORECASE
            )

        return cleaned_content

    def calculate_cache_hit_probability(
        self, current_prompt: str, cache_history: List[str]
    ) -> float:
        """
        Calculate probability of cache hit for given prompt

        Uses similarity analysis with historical prompts to predict cache hits
        """
        if not cache_history:
            return 0.0

        try:
            # Generate hash for current prompt structure
            current_hash = hashlib.md5(current_prompt.encode()).hexdigest()

            # Check exact matches first
            for historical_prompt in cache_history:
                historical_hash = hashlib.md5(historical_prompt.encode()).hexdigest()
                if current_hash == historical_hash:
                    return 1.0  # Exact match = guaranteed hit

            # Calculate structural similarity
            current_structure = self._extract_prompt_structure(current_prompt)

            similarity_scores = []
            for historical_prompt in cache_history[
                -50:
            ]:  # Check last 50 for performance
                historical_structure = self._extract_prompt_structure(historical_prompt)
                similarity = self._calculate_structure_similarity(
                    current_structure, historical_structure
                )
                similarity_scores.append(similarity)

            if not similarity_scores:
                return 0.0

            # High similarity suggests potential cache hit
            max_similarity = max(similarity_scores)
            avg_similarity = sum(similarity_scores) / len(similarity_scores)

            # Probability based on similarity (heuristic)
            probability = (max_similarity * 0.7) + (avg_similarity * 0.3)

            return min(1.0, probability)

        except Exception as e:
            self.logger.error(f"❌ Cache hit probability calculation failed: {e}")
            return 0.0

    def _extract_prompt_structure(self, prompt: str) -> Dict[str, Any]:
        """Extract structural features from prompt for similarity analysis"""
        structure = {
            "length": len(prompt),
            "lines": len(prompt.split("\n")),
            "tags": len(re.findall(r"<[^>]+>", prompt)),
            "sections": len(re.findall(r"\n[A-Z][A-Z\s]+:", prompt)),
            "bullets": len(re.findall(r"^\s*[-*]\s", prompt, re.MULTILINE)),
            "numbers": len(re.findall(r"^\s*\d+\.", prompt, re.MULTILINE)),
        }

        # Extract key phrases that indicate prompt type
        key_phrases = []
        phrase_patterns = [
            r"mental model",
            r"analysis",
            r"context",
            r"reasoning",
            r"framework",
            r"systematic",
            r"evaluate",
        ]

        for pattern in phrase_patterns:
            matches = len(re.findall(pattern, prompt, re.IGNORECASE))
            if matches > 0:
                key_phrases.append((pattern, matches))

        structure["key_phrases"] = key_phrases
        return structure

    def _calculate_structure_similarity(
        self, struct1: Dict[str, Any], struct2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two prompt structures"""
        try:
            # Numeric feature similarity
            numeric_features = [
                "length",
                "lines",
                "tags",
                "sections",
                "bullets",
                "numbers",
            ]
            numeric_similarities = []

            for feature in numeric_features:
                val1 = struct1.get(feature, 0)
                val2 = struct2.get(feature, 0)

                if val1 == 0 and val2 == 0:
                    similarity = 1.0
                elif val1 == 0 or val2 == 0:
                    similarity = 0.0
                else:
                    # Normalized difference
                    diff = abs(val1 - val2)
                    max_val = max(val1, val2)
                    similarity = 1.0 - (diff / max_val)

                numeric_similarities.append(similarity)

            numeric_score = (
                sum(numeric_similarities) / len(numeric_similarities)
                if numeric_similarities
                else 0.0
            )

            # Key phrase similarity
            phrases1 = {
                phrase: count for phrase, count in struct1.get("key_phrases", [])
            }
            phrases2 = {
                phrase: count for phrase, count in struct2.get("key_phrases", [])
            }

            common_phrases = set(phrases1.keys()).intersection(set(phrases2.keys()))
            total_phrases = set(phrases1.keys()).union(set(phrases2.keys()))

            if total_phrases:
                phrase_score = len(common_phrases) / len(total_phrases)
            else:
                phrase_score = 1.0

            # Combined similarity (weighted)
            combined_similarity = (numeric_score * 0.6) + (phrase_score * 0.4)

            return combined_similarity

        except Exception as e:
            self.logger.error(f"❌ Structure similarity calculation failed: {e}")
            return 0.0

    def optimize_for_caching(self, prompt_components: Dict[str, str]) -> Dict[str, str]:
        """
        Optimize prompt components for maximum cache efficiency

        Implements Manus.im strategies for stable, cacheable prompts
        """
        if not self.settings.ENABLE_KV_CACHE_OPTIMIZATION:
            return prompt_components

        try:
            optimized = prompt_components.copy()

            # 1. Stabilize system prompt
            if "system_prompt" in optimized:
                optimized["system_prompt"] = self._stabilize_system_prompt(
                    optimized["system_prompt"]
                )

            # 2. Structure user input for consistency
            if "user_input" in optimized:
                optimized["user_input"] = self._structure_user_input(
                    optimized["user_input"]
                )

            # 3. Create stable context section
            if "context" in optimized:
                optimized["context"] = self._optimize_context_section(
                    optimized["context"]
                )

            # 4. Apply template if available
            template_type = prompt_components.get("template_type", "cognitive_analysis")
            if template_type in self.stable_templates:
                template = self.stable_templates[template_type]
                optimized = self._apply_stable_template(template, optimized)

            # 5. Predict and log cache hit probability
            combined_prompt = "\n".join(optimized.values())
            hit_probability = self.calculate_cache_hit_probability(
                combined_prompt, self.cache_history
            )

            self.logger.debug(f"🎯 Cache hit probability: {hit_probability:.3f}")

            # Track optimization attempt
            self.metrics.total_requests += 1

            return optimized

        except Exception as e:
            self.logger.error(f"❌ Cache optimization failed: {e}")
            return prompt_components

    def _stabilize_system_prompt(self, system_prompt: str) -> str:
        """Stabilize system prompt for consistent caching"""
        # Remove dynamic elements
        stabilized = self._remove_cache_breakers(system_prompt)

        # Add stable framework structure
        if "<stable_framework>" not in stabilized:
            stabilized = f"""<stable_framework>
{stabilized}
</stable_framework>

<consistency_rules>
- Maintain systematic approach to analysis
- Use evidence-based reasoning
- Provide clear explanations
- Show thinking process
</consistency_rules>"""

        return stabilized

    def _structure_user_input(self, user_input: str) -> str:
        """Structure user input for better caching"""
        # Add consistent formatting
        structured = f"""<user_request>
{user_input.strip()}
</user_request>"""

        return structured

    def _optimize_context_section(self, context: str) -> str:
        """Optimize context section for caching"""
        # Remove cache breakers
        optimized = self._remove_cache_breakers(context)

        # Add stable structure
        optimized = f"""<context_data>
{optimized}
</context_data>"""

        return optimized

    def _apply_stable_template(
        self, template: StablePromptTemplate, components: Dict[str, str]
    ) -> Dict[str, str]:
        """Apply stable template to prompt components"""
        try:
            # Extract dynamic values from components
            dynamic_values = {}
            for key in template.dynamic_sections.keys():
                if key in components:
                    dynamic_values[key] = components[key]

            # Render template
            rendered_prompt = template.render(**dynamic_values)

            # Return as single optimized prompt
            return {
                "optimized_prompt": rendered_prompt,
                "template_type": "stable_template",
                "cache_key": template.cache_key_hash,
            }

        except Exception as e:
            self.logger.error(f"❌ Template application failed: {e}")
            return components

    def _update_cache_history(self, prompt: str):
        """Update cache history for hit prediction"""
        self.cache_history.append(prompt)

        # Keep history size manageable
        if len(self.cache_history) > self.max_history_size:
            self.cache_history = self.cache_history[-self.max_history_size // 2 :]

    def record_cache_hit(self, tokens_saved: int):
        """Record successful cache hit with cost savings"""
        cost_saved = (tokens_saved / 1_000_000) * (
            self.UNCACHED_TOKEN_COST_PER_MTOK - self.CACHED_TOKEN_COST_PER_MTOK
        )
        self.metrics.add_hit(tokens_saved, cost_saved)

        self.logger.info(
            f"💰 Cache hit! Saved {tokens_saved} tokens (${cost_saved:.4f})"
        )

    def record_cache_miss(self):
        """Record cache miss"""
        self.metrics.add_miss()
        self.logger.debug("❌ Cache miss recorded")

    def get_cache_metrics(self) -> Dict[str, Any]:
        """Get comprehensive KV-cache performance metrics"""
        return {
            "kv_cache_optimizer": "operational",
            "total_requests": self.metrics.total_requests,
            "cache_hits": self.metrics.cache_hits,
            "cache_misses": self.metrics.cache_misses,
            "hit_rate": self.metrics.calculate_hit_rate(),
            "target_hit_rate": self.settings.KV_CACHE_HIT_TARGET,
            "hit_rate_vs_target": self.metrics.calculate_hit_rate()
            - self.settings.KV_CACHE_HIT_TARGET,
            "total_tokens_saved": self.metrics.total_tokens_saved,
            "total_cost_savings_usd": self.metrics.cost_savings_usd,
            "cost_reduction_factor": self.UNCACHED_TOKEN_COST_PER_MTOK
            / self.CACHED_TOKEN_COST_PER_MTOK,
            "stable_templates_loaded": len(self.stable_templates),
            "cache_history_size": len(self.cache_history),
            "optimization_enabled": self.settings.ENABLE_KV_CACHE_OPTIMIZATION,
            "stable_prefix_enabled": self.settings.STABLE_PROMPT_PREFIX,
            "last_updated": self.metrics.last_updated.isoformat(),
        }


# Factory function for dependency injection
def create_kv_cache_optimizer(settings: CognitiveEngineSettings) -> KVCacheOptimizer:
    """Factory function for creating KV-Cache Optimizer"""
    return KVCacheOptimizer(settings)
