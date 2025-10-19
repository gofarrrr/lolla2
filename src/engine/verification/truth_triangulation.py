"""
METIS Truth Triangulation System
Three-layer verification for all insights: Database, Logic, Empirical
Only insights passing all three layers become 'verified intelligence'
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime
from uuid import uuid4


class VerificationStatus(Enum):
    VERIFIED = "verified"  # Passed all three layers
    PARTIAL = "partial"  # Passed 2 of 3 layers
    UNVERIFIED = "unverified"  # Passed 1 or fewer layers
    CONTRADICTED = "contradicted"  # Contradicts known patterns


@dataclass
class VerificationResult:
    """Result of truth triangulation verification"""

    insight: str
    status: VerificationStatus
    confidence: float
    verification_details: Dict[str, bool]
    verification_scores: Dict[str, float]
    supporting_evidence: List[str]
    contradictions: List[str]
    timestamp: str
    verification_id: Optional[str] = None


@dataclass
class VerificationLayer:
    """Individual verification layer result"""

    layer_name: str
    passed: bool
    score: float
    evidence: List[str]
    issues: List[str]


class TruthTriangulation:
    """
    Three-layer verification system for all insights.
    Ensures only verified intelligence reaches users.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.verification_cache = {}
        self._initialize_knowledge_base()

    def _initialize_knowledge_base(self):
        """Initialize connection to knowledge base"""
        try:
            # Connect to N-way database
            from src.database.nway_manager import NwayDatabaseManager

            self.nway_manager = NwayDatabaseManager()
            # Note: initialize() is async but called in sync context - using safe pattern
            try:
                import asyncio

                if (
                    hasattr(asyncio, "_get_running_loop")
                    and asyncio._get_running_loop()
                ):
                    # We're in an async context, defer initialization
                    self.nway_manager._deferred_init = True
                else:
                    # We're in sync context, safe to create new event loop
                    asyncio.run(self.nway_manager.initialize())
            except Exception:
                # Fallback: mark for deferred initialization
                self.nway_manager._deferred_init = True

            # Load mental models catalog
            from src.intelligence.model_catalog import get_model_catalog

            self.model_catalog = get_model_catalog()

            self.logger.info("✅ Knowledge base initialized for verification")
        except Exception as e:
            self.logger.warning(f"Knowledge base initialization failed: {e}")
            self.nway_manager = None
            self.model_catalog = None

    async def verify_insight(
        self,
        insight: str,
        supporting_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        """
        Triple verification: Database, Logic, Empirical

        Args:
            insight: The insight to verify
            supporting_data: Data supporting the insight (mental models, reasoning, etc.)
            context: Optional business context

        Returns:
            VerificationResult with status and confidence
        """

        # Check cache first
        cache_key = self._generate_cache_key(insight, supporting_data)
        if cache_key in self.verification_cache:
            cached_result = self.verification_cache[cache_key]
            self.logger.info(
                f"📋 Using cached verification for insight: {insight[:50]}..."
            )
            return cached_result

        self.logger.info(f"🔍 Starting truth triangulation for: {insight[:100]}...")

        # Layer 1: Database Verification
        database_layer = await self._verify_against_knowledge_base(
            insight, supporting_data, context
        )

        # Layer 2: Logic Verification (MECE structure)
        logic_layer = await self._validate_logical_structure(insight, supporting_data)

        # Layer 3: Empirical Verification (Research evidence)
        empirical_layer = await self._check_research_evidence(
            insight, supporting_data, context
        )

        # Aggregate results
        layers_passed = sum(
            [database_layer.passed, logic_layer.passed, empirical_layer.passed]
        )

        # Determine status
        if layers_passed == 3:
            status = VerificationStatus.VERIFIED
            confidence = 0.95
        elif layers_passed == 2:
            status = VerificationStatus.PARTIAL
            confidence = 0.70
        elif any([database_layer.score < 0.3, logic_layer.score < 0.3]):
            status = VerificationStatus.CONTRADICTED
            confidence = 0.20
        else:
            status = VerificationStatus.UNVERIFIED
            confidence = 0.40

        # Compile supporting evidence and contradictions
        supporting_evidence = []
        contradictions = []

        for layer in [database_layer, logic_layer, empirical_layer]:
            supporting_evidence.extend(layer.evidence)
            contradictions.extend(layer.issues)

        result = VerificationResult(
            insight=insight,
            status=status,
            confidence=confidence,
            verification_details={
                "database": database_layer.passed,
                "logic": logic_layer.passed,
                "empirical": empirical_layer.passed,
            },
            verification_scores={
                "database": database_layer.score,
                "logic": logic_layer.score,
                "empirical": empirical_layer.score,
            },
            supporting_evidence=supporting_evidence[:5],  # Top 5 evidence
            contradictions=contradictions[:3],  # Top 3 issues
            timestamp=datetime.now().isoformat(),
        )

        # Cache the result
        self.verification_cache[cache_key] = result

        # Log verification summary
        self._log_verification_result(result)

        return result

    async def _verify_against_knowledge_base(
        self,
        insight: str,
        supporting_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> VerificationLayer:
        """
        Layer 1: Verify insight against N-way interactions and mental models
        """

        layer_evidence = []
        layer_issues = []
        score = 0.0

        try:
            # Extract mental models from supporting data
            mental_models = supporting_data.get("mental_models_selected", [])

            if not mental_models:
                layer_issues.append("No mental models specified for verification")
                return VerificationLayer(
                    layer_name="database",
                    passed=False,
                    score=0.0,
                    evidence=[],
                    issues=layer_issues,
                )

            # Check N-way interactions
            if self.nway_manager:
                try:
                    # Find relevant N-way patterns
                    relevant_interactions = (
                        await self.nway_manager.find_optimal_interaction(
                            problem_keywords=self._extract_keywords(insight),
                            domain=(
                                context.get("industry", "general")
                                if context
                                else "general"
                            ),
                        )
                    )

                    if relevant_interactions:
                        # Check if insight aligns with known patterns
                        interaction = relevant_interactions
                        if any(
                            model in mental_models
                            for model in interaction.get("models_involved", [])
                        ):
                            score += 0.5
                            layer_evidence.append(
                                f"Aligns with N-way pattern: {interaction.get('interaction_id')}"
                            )

                            # Check if emergent effect matches
                            emergent_effect = interaction.get(
                                "emergent_effect_summary", ""
                            )
                            if (
                                self._semantic_similarity(insight, emergent_effect)
                                > 0.6
                            ):
                                score += 0.3
                                layer_evidence.append(
                                    f"Matches emergent effect: {emergent_effect[:100]}..."
                                )
                        else:
                            layer_issues.append(
                                "Mental models don't match known N-way pattern"
                            )
                except Exception as e:
                    self.logger.warning(f"N-way interaction check failed: {e}")
                    layer_issues.append("N-way interaction check failed")

            # Verify against mental model catalog
            if self.model_catalog:
                try:
                    for model in mental_models:
                        if self.model_catalog.is_valid_model(model):
                            score += 0.1
                            layer_evidence.append(f"Valid mental model: {model}")
                        else:
                            layer_issues.append(f"Unknown mental model: {model}")
                            score -= 0.1
                except Exception as e:
                    self.logger.warning(f"Model catalog check failed: {e}")
                    layer_issues.append("Model catalog check failed")

            # Fallback validation - basic model recognition
            if not layer_evidence:
                known_models = [
                    "systems_thinking",
                    "critical_thinking",
                    "mece_structuring",
                    "hypothesis_testing",
                    "multi_criteria_decision",
                    "cognitive_auditor",
                    "ideological_turing_test",
                ]
                valid_models = [m for m in mental_models if m in known_models]
                if valid_models:
                    score += len(valid_models) * 0.15
                    layer_evidence.append(f"Recognized mental models: {valid_models}")

            # Normalize score
            score = max(0.0, min(1.0, score))

        except Exception as e:
            self.logger.error(f"Database verification failed: {e}")
            layer_issues.append(f"Database verification error: {str(e)}")
            score = 0.0

        return VerificationLayer(
            layer_name="database",
            passed=score > 0.6,
            score=score,
            evidence=layer_evidence,
            issues=layer_issues,
        )

    async def _validate_logical_structure(
        self, insight: str, supporting_data: Dict[str, Any]
    ) -> VerificationLayer:
        """
        Layer 2: Validate MECE structure and logical consistency
        """

        layer_evidence = []
        layer_issues = []
        score = 0.0

        try:
            # Check for MECE compliance
            problem_breakdown = supporting_data.get("problem_breakdown", {})

            if problem_breakdown:
                # Check mutual exclusivity
                components = problem_breakdown.get("main_components", [])
                if components:
                    if self._check_mutual_exclusivity(components):
                        score += 0.4
                        layer_evidence.append(
                            f"MECE: {len(components)} mutually exclusive components"
                        )
                    else:
                        layer_issues.append("Components have potential overlap")
                        score -= 0.2

                    # Check collective exhaustiveness
                    if self._check_collective_exhaustiveness(
                        components,
                        insight,
                        supporting_data.get("problem_statement", ""),
                    ):
                        score += 0.4
                        layer_evidence.append("MECE: Collectively exhaustive coverage")
                    else:
                        layer_issues.append("Potential gaps in problem coverage")
                        score -= 0.1
            else:
                # Fallback: check for basic logical structure in insight
                if self._has_logical_structure(insight):
                    score += 0.3
                    layer_evidence.append("Basic logical structure detected")

            # Check logical flow
            reasoning = supporting_data.get("reasoning_description", "")
            if reasoning:
                if self._check_logical_consistency(insight, reasoning):
                    score += 0.2
                    layer_evidence.append("Logical flow from reasoning to insight")
                else:
                    layer_issues.append("Logical inconsistency detected")
                    score -= 0.2

            # Check for logical fallacies
            fallacies = self._detect_logical_fallacies(insight, supporting_data)
            if fallacies:
                for fallacy in fallacies:
                    layer_issues.append(f"Potential logical fallacy: {fallacy}")
                    score -= 0.1
            else:
                layer_evidence.append("No logical fallacies detected")
                score += 0.1

            # Normalize score
            score = max(0.0, min(1.0, score))

        except Exception as e:
            self.logger.error(f"Logic verification failed: {e}")
            layer_issues.append(f"Logic verification error: {str(e)}")
            score = 0.0

        return VerificationLayer(
            layer_name="logic",
            passed=score > 0.5,
            score=score,
            evidence=layer_evidence,
            issues=layer_issues,
        )

    async def _check_research_evidence(
        self,
        insight: str,
        supporting_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> VerificationLayer:
        """
        Layer 3: Verify against external research and empirical data
        """

        layer_evidence = []
        layer_issues = []
        score = 0.0

        try:
            # Check for research backing
            research_data = supporting_data.get("research_data")
            research_requirements = supporting_data.get("research_requirements", [])

            if research_data:
                # Research was conducted
                sources_count = research_data.get("sources_count", 0)
                if sources_count > 0:
                    score += min(0.5, sources_count * 0.1)  # Up to 0.5 for 5+ sources
                    layer_evidence.append(f"Backed by {sources_count} research sources")

                # Check research quality
                research_summary = research_data.get("research_summary", "")
                if (
                    research_summary
                    and self._semantic_similarity(insight, research_summary) > 0.5
                ):
                    score += 0.3
                    layer_evidence.append("Research directly supports insight")
                elif research_summary:
                    score += 0.1
                    layer_evidence.append("Research provides context")

            # Check if research requirements were identified
            if research_requirements:
                if len(research_requirements) > 0:
                    layer_evidence.append(
                        f"Identified {len(research_requirements)} research needs"
                    )
                    score += 0.1

                # Check if critical gaps exist
                critical_gaps = [
                    req for req in research_requirements if "critical" in req.lower()
                ]
                if critical_gaps:
                    layer_issues.append(f"Critical research gaps: {len(critical_gaps)}")
                    score -= 0.2

            # Check confidence score alignment
            confidence = supporting_data.get("confidence_score", 0.5)
            if confidence > 0.8 and not research_data:
                layer_issues.append("High confidence without research validation")
                score -= 0.2
            elif confidence > 0.8 and research_data:
                layer_evidence.append(
                    f"High confidence ({confidence:.2f}) with research"
                )
                score += 0.2

            # Basic empirical check - look for quantitative evidence
            if self._has_quantitative_evidence(insight):
                score += 0.2
                layer_evidence.append("Contains quantitative evidence")

            # Normalize score
            score = max(0.0, min(1.0, score))

        except Exception as e:
            self.logger.error(f"Empirical verification failed: {e}")
            layer_issues.append(f"Empirical verification error: {str(e)}")
            score = 0.0

        return VerificationLayer(
            layer_name="empirical",
            passed=score > 0.4,  # Lower threshold for empirical
            score=score,
            evidence=layer_evidence,
            issues=layer_issues,
        )

    def _has_logical_structure(self, insight: str) -> bool:
        """Check if insight has basic logical structure"""
        logical_indicators = [
            "because",
            "therefore",
            "thus",
            "hence",
            "consequently",
            "as a result",
            "due to",
            "leads to",
            "causes",
            "enables",
        ]
        return any(indicator in insight.lower() for indicator in logical_indicators)

    def _has_quantitative_evidence(self, insight: str) -> bool:
        """Check if insight contains quantitative evidence"""
        import re

        # Look for numbers, percentages, monetary amounts
        patterns = [
            r"\d+%",  # percentages
            r"\$\d+",  # dollar amounts
            r"\d+\.\d+",  # decimals
            r"\d+x",  # multipliers
            r"\d+\s*times",  # times
            r"increase.*\d+",  # increases with numbers
            r"decrease.*\d+",  # decreases with numbers
        ]

        for pattern in patterns:
            if re.search(pattern, insight, re.IGNORECASE):
                return True
        return False

    def _check_mutual_exclusivity(self, components: List[str]) -> bool:
        """Check if components are mutually exclusive"""

        # Simple overlap detection
        for i, comp1 in enumerate(components):
            for comp2 in components[i + 1 :]:
                # Check for word overlap (simple heuristic)
                words1 = set(comp1.lower().split())
                words2 = set(comp2.lower().split())

                # Remove common words
                common_words = {
                    "and",
                    "or",
                    "the",
                    "a",
                    "an",
                    "of",
                    "in",
                    "vs",
                    "versus",
                }
                words1 = words1 - common_words
                words2 = words2 - common_words

                overlap = words1.intersection(words2)
                if len(overlap) > len(words1) * 0.3:  # More than 30% overlap
                    return False

        return True

    def _check_collective_exhaustiveness(
        self, components: List[str], insight: str, problem_statement: str
    ) -> bool:
        """Check if components collectively cover the problem space"""

        # Extract key concepts from problem
        problem_concepts = self._extract_keywords(problem_statement)

        # Check if components cover key concepts
        component_text = " ".join(components).lower()
        coverage_count = sum(
            1 for concept in problem_concepts if concept.lower() in component_text
        )

        coverage_ratio = (
            coverage_count / len(problem_concepts) if problem_concepts else 0
        )

        return coverage_ratio > 0.6  # At least 60% coverage

    def _check_logical_consistency(self, insight: str, reasoning: str) -> bool:
        """Check if insight logically follows from reasoning"""

        # Simple consistency check based on keyword overlap
        insight_keywords = set(self._extract_keywords(insight))
        reasoning_keywords = set(self._extract_keywords(reasoning))

        if not insight_keywords or not reasoning_keywords:
            return False

        # Check if key insight concepts appear in reasoning
        overlap = insight_keywords.intersection(reasoning_keywords)
        consistency_score = len(overlap) / len(insight_keywords)

        return consistency_score > 0.4  # At least 40% concept overlap

    def _detect_logical_fallacies(
        self, insight: str, supporting_data: Dict[str, Any]
    ) -> List[str]:
        """Detect common logical fallacies"""

        fallacies = []
        insight_lower = insight.lower()

        # Check for absolute statements (potential false dichotomy)
        absolute_words = ["always", "never", "all", "none", "every", "no one"]
        if any(word in insight_lower for word in absolute_words):
            fallacies.append("Absolute statement - potential false dichotomy")

        # Check for correlation/causation confusion
        correlation_words = ["correlated", "associated", "linked"]
        causation_words = ["causes", "results in", "leads to", "creates"]

        has_correlation = any(word in insight_lower for word in correlation_words)
        has_causation = any(word in insight_lower for word in causation_words)

        if has_correlation and has_causation:
            fallacies.append("Potential correlation/causation confusion")

        # Check for appeal to authority without evidence
        authority_words = ["expert", "research shows", "studies prove"]
        if any(word in insight_lower for word in authority_words):
            if not supporting_data.get("research_data"):
                fallacies.append("Appeal to authority without cited evidence")

        return fallacies

    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""

        # Simple keyword-based similarity
        keywords1 = set(self._extract_keywords(text1))
        keywords2 = set(self._extract_keywords(text2))

        if not keywords1 or not keywords2:
            return 0.0

        intersection = keywords1.intersection(keywords2)
        union = keywords1.union(keywords2)

        return len(intersection) / len(union) if union else 0.0

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""

        # Simple keyword extraction
        import re

        # Remove punctuation and split
        words = re.findall(r"\b\w+\b", text.lower())

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
            "as",
            "is",
            "was",
            "are",
            "were",
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
            "must",
            "can",
            "this",
            "that",
            "these",
            "those",
        }

        keywords = [w for w in words if w not in stop_words and len(w) > 3]

        return keywords[:20]  # Top 20 keywords

    def _generate_cache_key(self, insight: str, supporting_data: Dict[str, Any]) -> str:
        """Generate cache key for verification result"""

        # Create a stable hash from insight and key supporting data
        import hashlib

        key_components = [
            insight[:200],  # First 200 chars of insight
            str(sorted(supporting_data.get("mental_models_selected", []))),
            str(supporting_data.get("confidence_score", 0)),
        ]

        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _log_verification_result(self, result: VerificationResult):
        """Log verification result for monitoring"""

        status_emoji = {
            VerificationStatus.VERIFIED: "✅",
            VerificationStatus.PARTIAL: "⚠️",
            VerificationStatus.UNVERIFIED: "❌",
            VerificationStatus.CONTRADICTED: "🚫",
        }

        emoji = status_emoji.get(result.status, "❓")

        self.logger.info(
            f"{emoji} Verification Result: {result.status.value} "
            f"(Confidence: {result.confidence:.2f}) "
            f"[D:{result.verification_scores['database']:.2f} "
            f"L:{result.verification_scores['logic']:.2f} "
            f"E:{result.verification_scores['empirical']:.2f}]"
        )

        if result.contradictions:
            self.logger.warning(f"   Issues: {'; '.join(result.contradictions[:2])}")

        if result.supporting_evidence:
            self.logger.info(f"   Evidence: {result.supporting_evidence[0]}")

    async def verify_batch(
        self,
        insights: List[str],
        supporting_data_list: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[VerificationResult]:
        """Verify multiple insights in batch"""

        results = []

        for insight, supporting_data in zip(insights, supporting_data_list):
            result = await self.verify_insight(insight, supporting_data, context)
            results.append(result)

        # Log batch summary
        verified_count = sum(
            1 for r in results if r.status == VerificationStatus.VERIFIED
        )
        partial_count = sum(
            1 for r in results if r.status == VerificationStatus.PARTIAL
        )

        self.logger.info(
            f"📊 Batch Verification Complete: "
            f"{verified_count}/{len(results)} verified, "
            f"{partial_count}/{len(results)} partial"
        )

        return results

    def get_verification_summary(self) -> Dict[str, Any]:
        """Get summary of all verifications performed"""

        if not self.verification_cache:
            return {"total_verifications": 0}

        cache_results = list(self.verification_cache.values())

        return {
            "total_verifications": len(cache_results),
            "verified": sum(
                1 for r in cache_results if r.status == VerificationStatus.VERIFIED
            ),
            "partial": sum(
                1 for r in cache_results if r.status == VerificationStatus.PARTIAL
            ),
            "unverified": sum(
                1 for r in cache_results if r.status == VerificationStatus.UNVERIFIED
            ),
            "contradicted": sum(
                1 for r in cache_results if r.status == VerificationStatus.CONTRADICTED
            ),
            "avg_confidence": (
                sum(r.confidence for r in cache_results) / len(cache_results)
                if cache_results
                else 0
            ),
            "cache_size": len(self.verification_cache),
        }

    async def verify_comprehensive(
        self, claim: str, context: Dict[str, Any]
    ) -> VerificationResult:
        """
        Comprehensive verification method with verification_id for neural lace integration.

        Args:
            claim: The claim/insight to verify
            context: Context including phase, engagement_id, source_type

        Returns:
            VerificationResult with verification_id
        """
        # Generate verification ID
        verification_id = str(uuid4())[:8]

        self.logger.info(
            f"🔍 Comprehensive verification started: {verification_id} for claim: {claim[:50]}..."
        )

        try:
            # Call existing verify_insight with proper parameters
            result = await self.verify_insight(
                insight=claim,
                supporting_data={
                    "source_type": context.get("source_type", "analysis"),
                    "phase": context.get("phase", "unknown"),
                    "engagement_id": context.get("engagement_id"),
                },
                context=context,
            )

            # Add verification ID to result
            result.verification_id = verification_id

            self.logger.info(
                f"✅ Comprehensive verification completed: {verification_id} | "
                f"Status: {result.status.value} | Confidence: {result.confidence:.2f}"
            )

            return result

        except Exception as e:
            self.logger.error(
                f"Comprehensive verification failed for {verification_id}: {e}"
            )

            # Return a failed verification result
            return VerificationResult(
                insight=claim,
                status=VerificationStatus.UNVERIFIED,
                confidence=0.0,
                verification_details={
                    "database": False,
                    "logic": False,
                    "empirical": False,
                },
                verification_scores={"database": 0.0, "logic": 0.0, "empirical": 0.0},
                supporting_evidence=[],
                contradictions=[f"Verification failed: {str(e)}"],
                timestamp=datetime.now().isoformat(),
                verification_id=verification_id,
            )


# Global instance getter
_truth_triangulator = None


def get_truth_triangulator() -> TruthTriangulation:
    """Get the global truth triangulation instance"""
    global _truth_triangulator
    if _truth_triangulator is None:
        _truth_triangulator = TruthTriangulation()
    return _truth_triangulator
