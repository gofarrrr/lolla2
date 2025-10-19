"""
METIS V5 Query Classification Service
====================================

Extracted from monolithic optimal_consultant_engine.py (lines 1669-1896).
Handles query analysis, intent detection, complexity assessment, and classification.

Part of the Great Refactoring: Clean separation of query classification concerns.
"""

import re
from typing import Dict, List, Optional, Any

# Import our new contracts
from ..contracts import (
    EngagementRequest,
    QueryClassificationResult,
    QueryIntent,
    QueryComplexity,
)

# Try to import enhanced classifier (preserved from original)
try:
    from ...core.enhanced_query_classifier import (
        EnhancedQueryClassifier,
        QueryIntent as EnhancedQueryIntent,
        QueryComplexity as EnhancedQueryComplexity,
    )

    ENHANCED_CLASSIFIER_AVAILABLE = True
except ImportError:
    print("⚠️ Enhanced query classifier not available - using basic classification")
    ENHANCED_CLASSIFIER_AVAILABLE = False


class QueryClassificationService:
    """
    Stateless service for query classification and analysis.

    Extracted from OptimalConsultantEngine to follow Single Responsibility Principle.
    Handles both enhanced and basic classification methods.
    """

    def __init__(self):
        """Initialize the classification service"""
        self.enhanced_classifier = None
        self.use_enhanced_classifier = ENHANCED_CLASSIFIER_AVAILABLE

        # Initialize enhanced classifier if available
        if ENHANCED_CLASSIFIER_AVAILABLE:
            try:
                self.enhanced_classifier = EnhancedQueryClassifier()
                print("✅ QueryClassificationService: Enhanced classifier initialized")
            except Exception as e:
                print(f"⚠️ Enhanced classifier initialization failed: {e}")
                self.use_enhanced_classifier = False

    async def classify_query(
        self, request: EngagementRequest
    ) -> QueryClassificationResult:
        """
        Main entry point: Classify a query and return structured results.

        Args:
            request: EngagementRequest containing query and context

        Returns:
            QueryClassificationResult with classification details
        """
        query = request.query
        context = request.context or {}

        if self.use_enhanced_classifier and self.enhanced_classifier:
            return await self._enhanced_classify_query(query, context)
        else:
            return await self._basic_classify_query(query, context)

    async def _enhanced_classify_query(
        self, query: str, context: Dict[str, Any]
    ) -> QueryClassificationResult:
        """
        Use enhanced query classifier for superior accuracy.
        Extracted from optimal_consultant_engine.py lines 1679-1730.
        """
        # Extract keywords using enhanced method
        keywords = self.enhanced_classifier.extract_keywords(query)

        # Detect intent and complexity
        intent, intent_confidence = self.enhanced_classifier.detect_intent(
            query, keywords
        )
        complexity = self.enhanced_classifier.assess_complexity(query, keywords)

        # Map enhanced classifier results to our contract enums
        mapped_intent = self._map_enhanced_intent(intent)
        mapped_complexity = self._map_enhanced_complexity(complexity)

        # Get mental model suggestions
        mental_model_suggestions = self.enhanced_classifier.suggest_mental_models(
            query, keywords, intent
        )

        # Get routing pattern suggestions
        suggested_patterns = self.enhanced_classifier.suggest_routing_patterns(
            query, keywords, intent, complexity
        )

        # Determine best routing pattern (converted from complexity enum to score for compatibility)
        complexity_score = self._complexity_to_score(mapped_complexity)
        routing_pattern = self._select_best_routing_pattern(
            suggested_patterns, keywords, complexity_score
        )

        # Extract key entities from keywords and query
        key_entities = self._extract_key_entities(query, keywords)

        return QueryClassificationResult(
            intent=mapped_intent,
            complexity=mapped_complexity,
            domain_tags=mental_model_suggestions[:5],  # Top 5 domain suggestions
            key_entities=key_entities,
            confidence_score=intent_confidence,
            processing_hints={
                "routing_pattern": routing_pattern,
                "mental_models": mental_model_suggestions,
                "keywords": keywords,
                "requires_research": complexity_score >= 6,
                "consultant_count": min(3, max(1, complexity_score // 2)),
            },
        )

    async def _basic_classify_query(
        self, query: str, context: Dict[str, Any]
    ) -> QueryClassificationResult:
        """
        Basic classification fallback.
        Extracted from optimal_consultant_engine.py lines 1732-1756.
        """
        # Extract meaningful keywords
        keywords = self._extract_keywords(query)

        # Assess complexity
        complexity_score = self._assess_complexity(query, keywords)
        complexity = self._score_to_complexity(complexity_score)

        # Determine query type/intent
        intent = self._determine_query_intent(keywords, query)

        # Find matching triggers
        matched_triggers = self._find_matched_triggers(keywords)

        # Find best routing pattern
        routing_pattern = self._find_routing_pattern(keywords, complexity_score)

        # Extract key entities
        key_entities = self._extract_key_entities(query, keywords)

        # Calculate basic confidence
        confidence_score = self._calculate_basic_confidence(
            keywords, complexity_score, matched_triggers
        )

        return QueryClassificationResult(
            intent=intent,
            complexity=complexity,
            domain_tags=matched_triggers[:5],  # Use triggers as domain tags
            key_entities=key_entities,
            confidence_score=confidence_score,
            processing_hints={
                "routing_pattern": routing_pattern,
                "keywords": keywords,
                "matched_triggers": matched_triggers,
                "requires_research": complexity_score >= 6,
                "consultant_count": min(3, max(1, complexity_score // 2)),
            },
        )

    # === ENHANCED CLASSIFIER MAPPING METHODS ===

    def _map_enhanced_intent(self, enhanced_intent) -> QueryIntent:
        """Map enhanced classifier intent to our contract enum"""
        intent_mapping = {
            "strategic_analysis": QueryIntent.STRATEGIC_ANALYSIS,
            "problem_solving": QueryIntent.PROBLEM_SOLVING,
            "decision_support": QueryIntent.DECISION_SUPPORT,
            "research": QueryIntent.RESEARCH_SYNTHESIS,
            "creative": QueryIntent.CREATIVE_IDEATION,
            "technical": QueryIntent.TECHNICAL_ANALYSIS,
        }

        enhanced_str = str(enhanced_intent).lower() if enhanced_intent else ""
        return intent_mapping.get(enhanced_str, QueryIntent.GENERAL_INQUIRY)

    def _map_enhanced_complexity(self, enhanced_complexity) -> QueryComplexity:
        """Map enhanced classifier complexity to our contract enum"""
        complexity_mapping = {
            "simple": QueryComplexity.SIMPLE,
            "moderate": QueryComplexity.MODERATE,
            "complex": QueryComplexity.COMPLEX,
            "expert": QueryComplexity.EXPERT,
        }

        enhanced_str = str(enhanced_complexity).lower() if enhanced_complexity else ""
        return complexity_mapping.get(enhanced_str, QueryComplexity.MODERATE)

    # === BASIC CLASSIFICATION METHODS ===

    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract meaningful keywords from query.
        Preserved from original implementation.
        """
        # Remove common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "this",
            "that",
            "these",
            "those",
        }

        # Extract words (preserve numbers, handle hyphenated words)
        words = re.findall(r"\b[\w\-]+\b", query.lower())

        # Filter out stop words and short words
        keywords = [word for word in words if word not in stop_words and len(word) > 2]

        return keywords[:10]  # Limit to top 10 keywords

    def _assess_complexity(self, query: str, keywords: List[str]) -> int:
        """
        Assess query complexity on a scale of 1-10.
        Preserved from original implementation.
        """
        complexity = 1

        # Length-based complexity
        if len(query) > 100:
            complexity += 2
        elif len(query) > 50:
            complexity += 1

        # Keyword-based complexity
        complexity_indicators = [
            "strategy",
            "analyze",
            "implement",
            "optimize",
            "framework",
            "methodology",
            "integration",
            "architecture",
            "scalability",
            "performance",
            "security",
            "compliance",
            "governance",
            "transformation",
            "innovation",
            "disruption",
        ]

        matches = sum(1 for keyword in keywords if keyword in complexity_indicators)
        complexity += min(matches, 4)

        # Question complexity
        question_words = ["how", "why", "what", "when", "where", "which"]
        question_count = sum(1 for word in question_words if word in query.lower())
        if question_count > 1:
            complexity += 1

        return min(complexity, 10)

    def _determine_query_intent(self, keywords: List[str], query: str) -> QueryIntent:
        """
        Determine query intent from keywords and content.
        Basic implementation for fallback.
        """
        query_lower = query.lower()

        # Strategic patterns
        strategic_patterns = [
            "strategy",
            "strategic",
            "planning",
            "roadmap",
            "vision",
            "market",
        ]
        if any(pattern in query_lower for pattern in strategic_patterns):
            return QueryIntent.STRATEGIC_ANALYSIS

        # Problem solving patterns
        problem_patterns = ["problem", "issue", "challenge", "solve", "fix", "resolve"]
        if any(pattern in query_lower for pattern in problem_patterns):
            return QueryIntent.PROBLEM_SOLVING

        # Decision support patterns
        decision_patterns = [
            "should",
            "choose",
            "decide",
            "option",
            "alternative",
            "recommend",
        ]
        if any(pattern in query_lower for pattern in decision_patterns):
            return QueryIntent.DECISION_SUPPORT

        # Research patterns
        research_patterns = ["research", "analyze", "study", "investigate", "examine"]
        if any(pattern in query_lower for pattern in research_patterns):
            return QueryIntent.RESEARCH_SYNTHESIS

        # Creative patterns
        creative_patterns = [
            "create",
            "design",
            "innovate",
            "brainstorm",
            "generate",
            "idea",
        ]
        if any(pattern in query_lower for pattern in creative_patterns):
            return QueryIntent.CREATIVE_IDEATION

        # Technical patterns
        technical_patterns = [
            "implement",
            "technical",
            "architecture",
            "system",
            "integration",
        ]
        if any(pattern in query_lower for pattern in technical_patterns):
            return QueryIntent.TECHNICAL_ANALYSIS

        return QueryIntent.GENERAL_INQUIRY

    def _find_matched_triggers(self, keywords: List[str]) -> List[str]:
        """
        Find domain triggers that match keywords.
        Basic implementation for domain identification.
        """
        domain_triggers = {
            "business_strategy": [
                "strategy",
                "business",
                "market",
                "competitive",
                "planning",
            ],
            "technology": ["technology", "software", "system", "technical", "digital"],
            "finance": ["financial", "budget", "cost", "revenue", "investment", "roi"],
            "operations": [
                "operations",
                "process",
                "workflow",
                "efficiency",
                "optimization",
            ],
            "marketing": ["marketing", "brand", "customer", "campaign", "audience"],
            "leadership": [
                "leadership",
                "management",
                "team",
                "culture",
                "organization",
            ],
            "innovation": ["innovation", "creative", "design", "prototype", "research"],
            "risk_management": [
                "risk",
                "compliance",
                "security",
                "governance",
                "audit",
            ],
        }

        matched = []
        for domain, triggers in domain_triggers.items():
            if any(trigger in keywords for trigger in triggers):
                matched.append(domain)

        return matched

    def _find_routing_pattern(
        self, keywords: List[str], complexity_score: int
    ) -> Optional[str]:
        """
        Find best routing pattern based on keywords and complexity.
        Basic implementation for routing decisions.
        """
        # High complexity queries get comprehensive routing
        if complexity_score >= 8:
            return "comprehensive_analysis"
        elif complexity_score >= 6:
            return "detailed_analysis"
        elif complexity_score >= 4:
            return "standard_analysis"
        else:
            return "quick_analysis"

    # === UTILITY METHODS ===

    def _extract_key_entities(self, query: str, keywords: List[str]) -> List[str]:
        """Extract key entities from query and keywords"""
        # Simple entity extraction - can be enhanced with NLP libraries
        entities = []

        # Extract quoted strings
        quoted = re.findall(r'"([^"]*)"', query)
        entities.extend(quoted)

        # Extract capitalized words (potential proper nouns)
        capitalized = re.findall(r"\b[A-Z][a-z]+\b", query)
        entities.extend(capitalized)

        # Add important keywords
        important_keywords = [kw for kw in keywords if len(kw) > 4]
        entities.extend(important_keywords[:5])

        # Remove duplicates and return unique entities
        return list(dict.fromkeys(entities))[:10]

    def _calculate_basic_confidence(
        self, keywords: List[str], complexity_score: int, matched_triggers: List[str]
    ) -> float:
        """Calculate confidence score for basic classification"""
        confidence = 0.6  # Base confidence

        # Boost for keyword clarity
        if len(keywords) >= 3:
            confidence += 0.1

        # Boost for domain matches
        confidence += min(len(matched_triggers) * 0.1, 0.2)

        # Adjust for complexity (moderate complexity gets highest confidence)
        if 4 <= complexity_score <= 6:
            confidence += 0.1

        return min(confidence, 0.95)  # Cap at 95%

    def _complexity_to_score(self, complexity: QueryComplexity) -> int:
        """Convert complexity enum to numeric score for compatibility"""
        mapping = {
            QueryComplexity.SIMPLE: 3,
            QueryComplexity.MODERATE: 5,
            QueryComplexity.COMPLEX: 7,
            QueryComplexity.EXPERT: 9,
        }
        return mapping.get(complexity, 5)

    def _score_to_complexity(self, score: int) -> QueryComplexity:
        """Convert numeric score to complexity enum"""
        if score <= 3:
            return QueryComplexity.SIMPLE
        elif score <= 5:
            return QueryComplexity.MODERATE
        elif score <= 7:
            return QueryComplexity.COMPLEX
        else:
            return QueryComplexity.EXPERT

    def _select_best_routing_pattern(
        self, suggested_patterns: List[str], keywords: List[str], complexity_score: int
    ) -> Optional[str]:
        """
        Select best routing pattern from enhanced classifier suggestions.
        Preserved from original implementation (lines 1758-1896).
        """
        if not suggested_patterns:
            return self._find_routing_pattern(keywords, complexity_score)

        # Score each suggested pattern
        pattern_scores = {}
        for pattern in suggested_patterns:
            score = self._score_routing_pattern(pattern, keywords, complexity_score)
            pattern_scores[pattern] = score

        # Return pattern with highest score
        if pattern_scores:
            best_pattern = max(pattern_scores, key=pattern_scores.get)
            return best_pattern if pattern_scores[best_pattern] > 0 else None

        return None

    def _score_routing_pattern(
        self, pattern: str, keywords: List[str], complexity_score: int
    ) -> float:
        """Score a routing pattern for selection"""
        score = 0.5  # Base score

        # Pattern-specific scoring logic
        pattern_keywords = {
            "strategic_analysis": ["strategy", "strategic", "analysis", "market"],
            "technical_deep_dive": [
                "technical",
                "system",
                "architecture",
                "implementation",
            ],
            "problem_solving": ["problem", "solve", "issue", "challenge"],
            "creative_exploration": ["creative", "innovation", "design", "brainstorm"],
            "research_synthesis": ["research", "analyze", "study", "investigate"],
        }

        # Boost score for keyword matches
        if pattern in pattern_keywords:
            matches = sum(1 for kw in keywords if kw in pattern_keywords[pattern])
            score += matches * 0.2

        # Adjust for complexity appropriateness
        complexity_fit = {
            "quick_analysis": (1, 4),
            "standard_analysis": (3, 6),
            "detailed_analysis": (5, 8),
            "comprehensive_analysis": (7, 10),
        }

        if pattern in complexity_fit:
            min_complexity, max_complexity = complexity_fit[pattern]
            if min_complexity <= complexity_score <= max_complexity:
                score += 0.3

        return score


# Factory function for service creation
def get_query_classification_service() -> QueryClassificationService:
    """Factory function to create QueryClassificationService instance"""
    return QueryClassificationService()
