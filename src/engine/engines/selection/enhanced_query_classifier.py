"""
Enhanced Query Classifier
Advanced pattern matching and query classification for optimal consultant routing
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import asyncio


class QueryIntent(Enum):
    """Query intent categories"""

    STRATEGIC_PLANNING = "strategic_planning"
    PROBLEM_SOLVING = "problem_solving"
    INNOVATION = "innovation"
    OPERATIONAL_OPTIMIZATION = "operational_optimization"
    CRISIS_MANAGEMENT = "crisis_management"
    TRANSFORMATION = "transformation"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    IMPLEMENTATION = "implementation"
    GENERAL = "general"


class QueryComplexity(Enum):
    """Query complexity levels"""

    SIMPLE = 1
    MODERATE = 2
    COMPLEX = 3
    HIGHLY_COMPLEX = 4
    ULTRA_COMPLEX = 5


@dataclass
class QueryAnalysis:
    """Comprehensive query analysis result"""

    original_query: str
    intent: QueryIntent
    complexity: QueryComplexity
    keywords: List[str]
    entities: List[str]
    context_clues: List[str]
    urgency_level: int  # 1-5 scale
    scope_level: int  # 1-5 scale (team, department, organization, industry, global)
    confidence_score: float
    routing_suggestions: List[str]
    metadata: Dict[str, Any]


class EnhancedQueryClassifier:
    """
    Advanced query classifier using multiple analysis techniques:
    - Keyword extraction and weighting
    - Pattern matching with fuzzy logic
    - Context awareness
    - Intent detection
    - Complexity assessment
    - Entity recognition
    """

    def __init__(self):
        self._initialize_patterns()
        self._initialize_weights()

    def _initialize_patterns(self):
        """Initialize comprehensive pattern library"""

        # Intent patterns with weighted keywords
        self.intent_patterns = {
            QueryIntent.STRATEGIC_PLANNING: {
                "primary": [
                    "strategic",
                    "strategy",
                    "plan",
                    "vision",
                    "mission",
                    "roadmap",
                    "future",
                    "long-term",
                ],
                "secondary": [
                    "market",
                    "competitive",
                    "positioning",
                    "growth",
                    "expansion",
                    "direction",
                ],
                "context": [
                    "5-year",
                    "3-year",
                    "quarterly",
                    "annual",
                    "next year",
                    "future state",
                ],
            },
            QueryIntent.PROBLEM_SOLVING: {
                "primary": [
                    "problem",
                    "issue",
                    "challenge",
                    "difficulty",
                    "trouble",
                    "bottleneck",
                    "obstacle",
                ],
                "secondary": [
                    "solve",
                    "fix",
                    "resolve",
                    "address",
                    "overcome",
                    "tackle",
                ],
                "context": ["root cause", "diagnosis", "symptoms", "impact", "urgency"],
            },
            QueryIntent.INNOVATION: {
                "primary": [
                    "innovation",
                    "creative",
                    "breakthrough",
                    "novel",
                    "disruptive",
                    "revolutionary",
                ],
                "secondary": [
                    "design",
                    "invent",
                    "develop",
                    "create",
                    "pioneer",
                    "transform",
                ],
                "context": [
                    "new approach",
                    "out of the box",
                    "cutting edge",
                    "first-of-kind",
                ],
            },
            QueryIntent.OPERATIONAL_OPTIMIZATION: {
                "primary": [
                    "optimize",
                    "improve",
                    "efficiency",
                    "streamline",
                    "enhance",
                    "process",
                ],
                "secondary": [
                    "operational",
                    "workflow",
                    "performance",
                    "productivity",
                    "quality",
                ],
                "context": [
                    "reduce cost",
                    "save time",
                    "eliminate waste",
                    "lean",
                    "six sigma",
                ],
            },
            QueryIntent.CRISIS_MANAGEMENT: {
                "primary": [
                    "crisis",
                    "emergency",
                    "urgent",
                    "critical",
                    "immediate",
                    "disaster",
                ],
                "secondary": [
                    "response",
                    "recovery",
                    "damage",
                    "containment",
                    "mitigation",
                ],
                "context": [
                    "ASAP",
                    "today",
                    "now",
                    "emergency",
                    "code red",
                    "all hands",
                ],
            },
            QueryIntent.TRANSFORMATION: {
                "primary": [
                    "transformation",
                    "change",
                    "restructure",
                    "reorganize",
                    "modernize",
                ],
                "secondary": [
                    "digital",
                    "cultural",
                    "organizational",
                    "business model",
                    "paradigm",
                ],
                "context": [
                    "enterprise-wide",
                    "company-wide",
                    "across departments",
                    "cultural shift",
                ],
            },
        }

        # Complexity indicators
        self.complexity_patterns = {
            QueryComplexity.ULTRA_COMPLEX: [
                "enterprise-wide",
                "organization-wide",
                "transformation",
                "paradigm shift",
                "multiple stakeholders",
                "cross-functional",
                "regulatory compliance",
                "strategic overhaul",
                "industry disruption",
                "global implementation",
            ],
            QueryComplexity.HIGHLY_COMPLEX: [
                "comprehensive",
                "integrated",
                "multi-phase",
                "cross-departmental",
                "long-term",
                "strategic",
                "complex",
                "sophisticated",
                "advanced",
            ],
            QueryComplexity.COMPLEX: [
                "detailed",
                "thorough",
                "analytical",
                "systematic",
                "structured",
                "framework",
                "methodology",
                "process",
                "evaluation",
            ],
            QueryComplexity.MODERATE: [
                "improve",
                "optimize",
                "enhance",
                "develop",
                "implement",
                "assess",
                "review",
                "analyze",
                "design",
            ],
            QueryComplexity.SIMPLE: [
                "quick",
                "simple",
                "basic",
                "straightforward",
                "easy",
                "brief",
                "summary",
                "overview",
                "list",
            ],
        }

        # Scope indicators
        self.scope_patterns = {
            5: ["global", "worldwide", "international", "industry-wide", "market-wide"],
            4: ["enterprise", "organization-wide", "company-wide", "corporate"],
            3: ["department", "division", "business unit", "multi-team"],
            2: ["team", "group", "project", "local"],
            1: ["individual", "personal", "single", "isolated"],
        }

        # Urgency indicators
        self.urgency_patterns = {
            5: ["emergency", "critical", "urgent", "ASAP", "immediate", "crisis"],
            4: ["high priority", "important", "soon", "quickly", "fast"],
            3: ["moderate", "reasonable", "timely", "planned"],
            2: ["low priority", "when possible", "eventually"],
            1: ["no rush", "whenever", "long-term", "someday"],
        }

        # Entity patterns for business context
        self.entity_patterns = {
            "time_horizon": [
                r"\d+\s*(?:year|month|quarter|week|day)s?",
                r"(?:short|medium|long).?term",
                r"(?:Q\d|quarterly|annual|yearly)",
            ],
            "metrics": [
                r"\d+%",
                r"(?:increase|decrease|reduce|improve).*?(?:by|to)\s*\d+",
                r"(?:ROI|revenue|cost|profit|margin|efficiency)",
            ],
            "departments": [
                r"(?:marketing|sales|finance|HR|IT|operations|customer|support|engineering)",
                r"(?:team|department|division|unit|group)",
            ],
            "business_functions": [
                r"(?:strategy|planning|execution|implementation|analysis|design)",
                r"(?:development|management|leadership|governance)",
            ],
        }

    def _initialize_weights(self):
        """Initialize scoring weights for different aspects"""
        self.weights = {
            "primary_keywords": 3.0,
            "secondary_keywords": 2.0,
            "context_keywords": 1.5,
            "complexity_match": 2.5,
            "scope_indicators": 2.0,
            "urgency_indicators": 1.8,
            "entity_matches": 1.2,
            "phrase_context": 2.2,
        }

    def extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords using advanced techniques"""
        # Convert to lowercase and remove punctuation
        clean_query = re.sub(r"[^\w\s]", " ", query.lower())
        words = clean_query.split()

        # Remove stop words
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
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "among",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "her",
            "its",
            "our",
            "their",
            "am",
            "is",
            "are",
            "was",
            "were",
            "being",
            "been",
            "be",
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
            "can",
            "must",
            "shall",
        }

        # Filter meaningful words
        keywords = []
        for word in words:
            if len(word) > 2 and word not in stop_words:
                keywords.append(word)

        # Extract compound terms and phrases
        phrases = self._extract_phrases(query)
        keywords.extend(phrases)

        return list(set(keywords))  # Remove duplicates

    def _extract_phrases(self, query: str) -> List[str]:
        """Extract meaningful phrases and compound terms"""
        phrases = []

        # Business phrase patterns
        business_phrases = [
            r"digital transformation",
            r"market entry",
            r"competitive advantage",
            r"customer experience",
            r"business model",
            r"value proposition",
            r"go-to-market",
            r"supply chain",
            r"product development",
            r"cost reduction",
            r"revenue growth",
            r"operational excellence",
            r"strategic planning",
            r"change management",
            r"risk management",
            r"performance improvement",
            r"process optimization",
            r"innovation strategy",
        ]

        query_lower = query.lower()
        for pattern in business_phrases:
            matches = re.findall(pattern, query_lower)
            phrases.extend(matches)

        return phrases

    def detect_intent(
        self, query: str, keywords: List[str]
    ) -> Tuple[QueryIntent, float]:
        """Detect query intent using weighted pattern matching"""
        intent_scores = {}

        query_lower = query.lower()
        keyword_set = set(word.lower() for word in keywords)

        for intent, patterns in self.intent_patterns.items():
            score = 0.0

            # Primary keyword matches
            primary_matches = sum(
                1 for word in patterns["primary"] if word in keyword_set
            )
            score += primary_matches * self.weights["primary_keywords"]

            # Secondary keyword matches
            secondary_matches = sum(
                1 for word in patterns["secondary"] if word in keyword_set
            )
            score += secondary_matches * self.weights["secondary_keywords"]

            # Context phrase matches
            context_matches = sum(
                1 for phrase in patterns["context"] if phrase in query_lower
            )
            score += context_matches * self.weights["context_keywords"]

            # Phrase context analysis
            for phrase in patterns.get("context", []):
                if phrase in query_lower:
                    score += self.weights["phrase_context"]

            intent_scores[intent] = score

        if not intent_scores or max(intent_scores.values()) == 0:
            return QueryIntent.GENERAL, 0.5

        best_intent = max(intent_scores, key=intent_scores.get)
        max_score = intent_scores[best_intent]

        # Normalize confidence score
        total_possible = (
            len(self.intent_patterns[best_intent]["primary"])
            * self.weights["primary_keywords"]
            + len(self.intent_patterns[best_intent]["secondary"])
            * self.weights["secondary_keywords"]
            + len(self.intent_patterns[best_intent]["context"])
            * self.weights["context_keywords"]
        )

        confidence = min(max_score / total_possible, 1.0) if total_possible > 0 else 0.5

        return best_intent, confidence

    def assess_complexity(self, query: str, keywords: List[str]) -> QueryComplexity:
        """Assess query complexity using multiple indicators"""
        query_lower = query.lower()
        keyword_set = set(word.lower() for word in keywords)

        complexity_scores = {level: 0 for level in QueryComplexity}

        # Pattern-based complexity assessment
        for complexity, patterns in self.complexity_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern in query_lower or pattern in keyword_set:
                    score += 1
            complexity_scores[complexity] = score

        # Length-based complexity (word count)
        word_count = len(query.split())
        if word_count > 50:
            complexity_scores[QueryComplexity.ULTRA_COMPLEX] += 2
        elif word_count > 30:
            complexity_scores[QueryComplexity.HIGHLY_COMPLEX] += 2
        elif word_count > 20:
            complexity_scores[QueryComplexity.COMPLEX] += 1
        elif word_count < 10:
            complexity_scores[QueryComplexity.SIMPLE] += 1

        # Question complexity (multiple questions)
        question_count = query.count("?")
        if question_count > 3:
            complexity_scores[QueryComplexity.ULTRA_COMPLEX] += 1
        elif question_count > 1:
            complexity_scores[QueryComplexity.COMPLEX] += 1

        # Return the complexity level with the highest score
        if max(complexity_scores.values()) == 0:
            return QueryComplexity.MODERATE  # Default

        return max(complexity_scores, key=complexity_scores.get)

    def assess_scope(self, query: str) -> int:
        """Assess organizational scope (1-5 scale)"""
        query_lower = query.lower()

        for scope_level, patterns in self.scope_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    return scope_level

        # Default scope based on complexity indicators
        if any(word in query_lower for word in ["team", "group", "project"]):
            return 2
        elif any(word in query_lower for word in ["department", "division"]):
            return 3
        elif any(
            word in query_lower for word in ["organization", "company", "enterprise"]
        ):
            return 4

        return 3  # Default to department level

    def assess_urgency(self, query: str) -> int:
        """Assess urgency level (1-5 scale)"""
        query_lower = query.lower()

        for urgency_level, patterns in self.urgency_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    return urgency_level

        return 3  # Default to moderate urgency

    def extract_entities(self, query: str) -> List[str]:
        """Extract business entities from query"""
        entities = []

        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                entities.extend([f"{entity_type}:{match}" for match in matches])

        return entities

    def suggest_routing_patterns(
        self, intent: QueryIntent, complexity: QueryComplexity, scope: int, urgency: int
    ) -> List[str]:
        """Suggest optimal routing patterns based on analysis"""
        suggestions = []

        # Intent-based routing
        intent_routing = {
            QueryIntent.STRATEGIC_PLANNING: [
                "strategic_comprehensive",
                "strategic_analysis",
            ],
            QueryIntent.PROBLEM_SOLVING: ["tactical_execution", "problem_solving"],
            QueryIntent.INNOVATION: [
                "innovation_breakthrough",
                "synthesis_integration",
            ],
            QueryIntent.OPERATIONAL_OPTIMIZATION: [
                "operational_optimization",
                "implementation_delivery",
            ],
            QueryIntent.CRISIS_MANAGEMENT: ["crisis_management", "tactical_execution"],
            QueryIntent.TRANSFORMATION: [
                "transformation_change",
                "strategic_comprehensive",
            ],
            QueryIntent.ANALYSIS: ["analysis_focused", "strategic_analysis"],
            QueryIntent.SYNTHESIS: ["synthesis_integration", "strategic_comprehensive"],
            QueryIntent.IMPLEMENTATION: [
                "implementation_delivery",
                "tactical_execution",
            ],
        }

        base_suggestions = intent_routing.get(intent, ["strategic_analysis"])
        suggestions.extend(base_suggestions)

        # Complexity-based adjustments
        if complexity in [
            QueryComplexity.ULTRA_COMPLEX,
            QueryComplexity.HIGHLY_COMPLEX,
        ]:
            if "strategic_comprehensive" not in suggestions:
                suggestions.insert(0, "strategic_comprehensive")

        # Urgency-based adjustments
        if urgency >= 4:  # High urgency
            if "crisis_management" not in suggestions:
                suggestions.insert(0, "crisis_management")
            if "tactical_execution" not in suggestions:
                suggestions.append("tactical_execution")

        # Scope-based adjustments
        if scope >= 4:  # Enterprise-wide
            if "transformation_change" not in suggestions:
                suggestions.append("transformation_change")

        return suggestions[:3]  # Return top 3 suggestions

    async def analyze_query(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> QueryAnalysis:
        """Perform comprehensive query analysis"""

        # Step 1: Extract keywords
        keywords = self.extract_keywords(query)

        # Step 2: Detect intent
        intent, intent_confidence = self.detect_intent(query, keywords)

        # Step 3: Assess complexity
        complexity = self.assess_complexity(query, keywords)

        # Step 4: Assess scope and urgency
        scope = self.assess_scope(query)
        urgency = self.assess_urgency(query)

        # Step 5: Extract entities
        entities = self.extract_entities(query)

        # Step 6: Extract context clues
        context_clues = self._extract_context_clues(query, context or {})

        # Step 7: Generate routing suggestions
        routing_suggestions = self.suggest_routing_patterns(
            intent, complexity, scope, urgency
        )

        # Step 8: Calculate overall confidence
        confidence_factors = [
            intent_confidence,
            0.8 if len(keywords) >= 3 else 0.6,  # Keyword richness
            (
                0.9 if complexity != QueryComplexity.MODERATE else 0.7
            ),  # Complexity certainty
            0.8 if len(entities) > 0 else 0.6,  # Entity presence
        ]
        overall_confidence = sum(confidence_factors) / len(confidence_factors)

        return QueryAnalysis(
            original_query=query,
            intent=intent,
            complexity=complexity,
            keywords=keywords,
            entities=entities,
            context_clues=context_clues,
            urgency_level=urgency,
            scope_level=scope,
            confidence_score=overall_confidence,
            routing_suggestions=routing_suggestions,
            metadata={
                "word_count": len(query.split()),
                "question_count": query.count("?"),
                "intent_confidence": intent_confidence,
                "extracted_phrases": self._extract_phrases(query),
            },
        )

    def _extract_context_clues(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Extract contextual clues from query and provided context"""
        clues = []

        # Temporal context
        temporal_patterns = [
            r"(?:this|next|last)\s+(?:week|month|quarter|year)",
            r"(?:by|before|after)\s+\w+",
            r"(?:Q\d|quarterly|annual)",
            r"\d{4}",  # Years
        ]

        for pattern in temporal_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            clues.extend([f"temporal:{match}" for match in matches])

        # Quantitative context
        quantity_patterns = [
            r"\d+%",
            r"\$[\d,]+",
            r"\d+\s*(?:million|billion|thousand)",
            r"(?:increase|decrease|reduce|improve).*?\d+",
        ]

        for pattern in quantity_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            clues.extend([f"quantitative:{match}" for match in matches])

        # Add context from provided context dict
        if context:
            for key, value in context.items():
                if isinstance(value, str):
                    clues.append(f"context:{key}:{value}")

        return clues


# Global classifier instance
_global_classifier: Optional[EnhancedQueryClassifier] = None


def get_enhanced_classifier() -> EnhancedQueryClassifier:
    """Get global enhanced query classifier instance"""
    global _global_classifier
    if _global_classifier is None:
        _global_classifier = EnhancedQueryClassifier()
    return _global_classifier


# Testing and demonstration
async def test_enhanced_classifier():
    """Test the enhanced query classifier"""

    print("🧪 Testing Enhanced Query Classifier")
    print("=" * 60)

    classifier = get_enhanced_classifier()

    test_queries = [
        "How can we develop a comprehensive 5-year strategic plan to enter the Asian market while maintaining our competitive advantage in North America?",
        "Our customer support system is failing and we need to diagnose the problem and fix it ASAP!",
        "I need breakthrough innovative approaches to completely redesign our customer experience.",
        "What operational processes should we optimize to reduce costs by 20% this quarter?",
        "We have a critical product recall crisis that requires immediate response and damage control.",
        "How do we execute our digital transformation initiative across all business units?",
        "Can you help me analyze our market position and identify key growth opportunities?",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: {query}")
        print("-" * 50)

        analysis = await classifier.analyze_query(query)

        print("📊 Analysis Results:")
        print(f"   Intent: {analysis.intent.value}")
        print(
            f"   Complexity: {analysis.complexity.name} ({analysis.complexity.value})"
        )
        print(f"   Keywords: {', '.join(analysis.keywords[:5])}...")
        print(f"   Urgency: {analysis.urgency_level}/5")
        print(f"   Scope: {analysis.scope_level}/5")
        print(f"   Confidence: {analysis.confidence_score:.2f}")
        print(f"   Routing Suggestions: {', '.join(analysis.routing_suggestions)}")

        if analysis.entities:
            print(f"   Entities: {', '.join(analysis.entities[:3])}...")

        if analysis.context_clues:
            print(f"   Context Clues: {', '.join(analysis.context_clues[:3])}...")

    print("\n✅ Enhanced query classifier testing completed!")


if __name__ == "__main__":
    asyncio.run(test_enhanced_classifier())
