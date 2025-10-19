#!/usr/bin/env python3
"""
COGNITIVE CHEMISTRY SCORING ENGINES
Phase 2 of Operation: Cognitive Particle Accelerator

Revolutionary four-tier scoring system that transforms NWAY selection from
simple affinity matching to sophisticated cognitive chemistry reactions.

This implements the core breakthrough: scoring REACTIONS, not just components.
"""

from typing import Dict, List, Any, Set
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# ======================================================================
# TIER 1: LOLLAPALOOZA TRIGGER SCORING ENGINE ⚡
# ======================================================================


@dataclass
class LollapaloozaTriggers:
    """Environmental triggers that can ignite a Lollapalooza compound"""

    auction_indicators: List[str] = None
    social_selling_indicators: List[str] = None
    brand_marketing_indicators: List[str] = None
    scarcity_indicators: List[str] = None
    authority_indicators: List[str] = None
    commitment_indicators: List[str] = None

    def __post_init__(self):
        self.auction_indicators = self.auction_indicators or [
            "auction",
            "bidding",
            "bid",
            "outbid",
            "highest bidder",
            "reserve price",
            "going once",
            "sold",
            "competitive bidding",
            "public sale",
        ]
        self.social_selling_indicators = self.social_selling_indicators or [
            "home party",
            "tupperware",
            "social selling",
            "friend",
            "hostess",
            "recommendation",
            "peer influence",
            "social gathering",
            "MLM",
            "network marketing",
        ]
        self.brand_marketing_indicators = self.brand_marketing_indicators or [
            "brand loyalty",
            "coca cola",
            "brand association",
            "habit",
            "ubiquitous",
            "market dominance",
            "conditioning",
            "brand power",
            "consumer behavior",
        ]
        self.scarcity_indicators = self.scarcity_indicators or [
            "limited time",
            "scarcity",
            "exclusive",
            "only",
            "last chance",
            "running out",
            "limited supply",
            "rare",
            "unique opportunity",
        ]
        self.authority_indicators = self.authority_indicators or [
            "expert",
            "authority",
            "credential",
            "certification",
            "endorsed by",
            "recommended by",
            "official",
            "professional",
            "specialist",
        ]
        self.commitment_indicators = self.commitment_indicators or [
            "commitment",
            "public",
            "announce",
            "pledge",
            "promise",
            "declare",
            "consistent",
            "follow through",
            "accountability",
            "social pressure",
        ]


class LollapaloozaTriggerScoring:
    """
    TIER 1: Analyzes environmental triggers for Lollapalooza compounds

    This is the highest priority scoring because Lollapalooza effects
    create exponential amplification that overrides rational thought.
    """

    def __init__(self):
        self.triggers = LollapaloozaTriggers()
        self.amplification_patterns = {
            "auction": {
                "core_triggers": ["auction", "bidding", "competition"],
                "psychological_forces": [
                    "social-proof",
                    "scarcity",
                    "commitment",
                    "loss-aversion",
                ],
                "amplification_mechanism": "Public visibility enables social proof validation while scarcity creates near-ownership loss aversion",
                "system_override": True,
                "multiplicative_effect": 8.5,
            },
            "social_selling": {
                "core_triggers": ["friend", "party", "social", "recommendation"],
                "psychological_forces": [
                    "liking",
                    "reciprocity",
                    "social-proof",
                    "commitment",
                ],
                "amplification_mechanism": "Pre-existing relationships amplify all influence mechanisms simultaneously",
                "system_override": True,
                "multiplicative_effect": 7.8,
            },
            "brand_dominance": {
                "core_triggers": ["brand", "habit", "conditioning", "ubiquitous"],
                "psychological_forces": [
                    "classical-conditioning",
                    "habit-formation",
                    "social-proof",
                    "availability",
                ],
                "amplification_mechanism": "Classical conditioning creates positive associations while ubiquity provides constant cues",
                "system_override": True,
                "multiplicative_effect": 9.2,
            },
        }

    def score_environmental_triggers(
        self, problem_framework: str, nway_compound: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Analyze environmental triggers for Lollapalooza activation

        Returns scores for:
        - environmental_match_score: How well context matches triggers
        - amplification_intensity: How explosive the effect will be
        - rational_override_probability: Will it bypass System 2?
        - multiplicative_effect: Exponential vs additive gain
        - predictability: Can we predict when it triggers?
        """

        # Extract compound information
        compound_id = nway_compound.get("interaction_id", "")
        environmental_triggers = nway_compound.get("environmental_triggers", [])
        psychological_principles = nway_compound.get("psychological_principles", [])

        logger.info(f"🔍 Analyzing triggers for {compound_id}")

        # 1. Environmental Match Scoring
        environmental_match = self._calculate_environmental_match(
            problem_framework, environmental_triggers
        )

        # 2. Amplification Intensity (based on psychological principles)
        amplification_intensity = self._calculate_amplification_intensity(
            psychological_principles, compound_id
        )

        # 3. Rational Override Probability
        rational_override = self._calculate_rational_override_probability(
            environmental_match, amplification_intensity, psychological_principles
        )

        # 4. Multiplicative Effect Potential
        multiplicative_effect = self._calculate_multiplicative_effect(
            compound_id, psychological_principles
        )

        # 5. Predictability Assessment
        predictability = self._calculate_predictability(
            environmental_triggers, compound_id
        )

        scores = {
            "environmental_match_score": environmental_match,
            "amplification_intensity": amplification_intensity,
            "rational_override_probability": rational_override,
            "multiplicative_effect": multiplicative_effect,
            "predictability": predictability,
        }

        logger.info(f"⚡ Lollapalooza scores for {compound_id}: {scores}")
        return scores

    def _calculate_environmental_match(
        self, problem_framework: str, triggers: List[str]
    ) -> float:
        """Calculate how well the problem context matches environmental triggers"""
        if not triggers:
            return 0.0

        text_lower = problem_framework.lower()
        matches = 0
        total_possible = len(triggers)

        for trigger in triggers:
            if trigger.lower() in text_lower:
                matches += 1

        # Bonus for multiple trigger types present
        trigger_categories = self._identify_trigger_categories(text_lower)
        category_bonus = len(trigger_categories) * 0.15

        base_score = matches / total_possible if total_possible > 0 else 0
        final_score = min(1.0, base_score + category_bonus)

        return final_score

    def _identify_trigger_categories(self, text: str) -> Set[str]:
        """Identify which categories of triggers are present"""
        categories = set()

        for auction_term in self.triggers.auction_indicators:
            if auction_term in text:
                categories.add("auction")
                break

        for social_term in self.triggers.social_selling_indicators:
            if social_term in text:
                categories.add("social_selling")
                break

        for brand_term in self.triggers.brand_marketing_indicators:
            if brand_term in text:
                categories.add("brand_marketing")
                break

        return categories

    def _calculate_amplification_intensity(
        self, psychological_principles: List[str], compound_id: str
    ) -> float:
        """Calculate the intensity of psychological amplification"""
        if not psychological_principles:
            return 0.3  # Low intensity without clear principles

        # High-amplification principles (Munger's key triggers)
        high_amplification = {
            "social-proof",
            "scarcity",
            "commitment",
            "reciprocity",
            "authority",
            "liking",
        }

        # Medium-amplification principles
        medium_amplification = {
            "loss-aversion",
            "classical-conditioning",
            "habit-formation",
            "availability",
        }

        high_count = sum(1 for p in psychological_principles if p in high_amplification)
        medium_count = sum(
            1 for p in psychological_principles if p in medium_amplification
        )

        # Base intensity from principle count
        base_intensity = (high_count * 0.25) + (medium_count * 0.15)

        # Compound-specific amplification patterns
        compound_bonus = 0.0
        for pattern_name, pattern in self.amplification_patterns.items():
            if pattern_name.upper() in compound_id.upper():
                compound_bonus = 0.3
                break

        # Synergy bonus for multiple high-amplification principles
        synergy_bonus = 0.0
        if high_count >= 3:
            synergy_bonus = 0.25  # Multiple principles create synergy

        final_intensity = min(1.0, base_intensity + compound_bonus + synergy_bonus)
        return final_intensity

    def _calculate_rational_override_probability(
        self, env_match: float, amplification: float, principles: List[str]
    ) -> float:
        """Calculate probability that this will override rational System 2 thinking"""

        # Base probability from environmental conditions and amplification
        base_probability = (env_match * 0.6) + (amplification * 0.4)

        # Bonus for specific System 1 dominance patterns
        system1_dominance_principles = {
            "social-proof",
            "scarcity",
            "liking",
            "classical-conditioning",
            "habit-formation",
        }

        system1_count = sum(1 for p in principles if p in system1_dominance_principles)
        system1_bonus = min(0.3, system1_count * 0.1)

        # Penalty for principles that engage System 2
        system2_engagement_principles = {
            "critical-thinking",
            "evidence-based-reasoning",
            "logic",
        }
        system2_count = sum(1 for p in principles if p in system2_engagement_principles)
        system2_penalty = system2_count * 0.15

        final_probability = max(
            0.0, min(1.0, base_probability + system1_bonus - system2_penalty)
        )
        return final_probability

    def _calculate_multiplicative_effect(
        self, compound_id: str, principles: List[str]
    ) -> float:
        """Calculate multiplicative effect potential (1.0 = additive, 10.0 = maximum multiplier)"""

        # Check for known high-multiplier patterns
        for pattern_name, pattern in self.amplification_patterns.items():
            if pattern_name.upper() in compound_id.upper():
                return pattern["multiplicative_effect"]

        # Calculate based on psychological principle interactions
        principle_count = len(principles)

        if principle_count >= 4:
            return 7.5  # High multiplier for complex interactions
        elif principle_count >= 3:
            return 5.2  # Medium-high multiplier
        elif principle_count >= 2:
            return 3.1  # Medium multiplier
        else:
            return 1.8  # Low multiplier

    def _calculate_predictability(self, triggers: List[str], compound_id: str) -> float:
        """Calculate how predictable the trigger activation is"""

        # Well-defined triggers are more predictable
        if not triggers:
            return 0.2

        # Specific, observable triggers are highly predictable
        observable_triggers = {"auction", "bidding", "party", "friend", "brand"}
        observable_count = sum(
            1 for t in triggers if any(obs in t.lower() for obs in observable_triggers)
        )

        base_predictability = min(1.0, observable_count / len(triggers))

        # Known patterns have high predictability
        known_pattern_bonus = 0.0
        for pattern_name in self.amplification_patterns.keys():
            if pattern_name.upper() in compound_id.upper():
                known_pattern_bonus = 0.25
                break

        final_predictability = min(1.0, base_predictability + known_pattern_bonus)
        return final_predictability


# ======================================================================
# TIER 2: META-FRAMEWORK UNIVERSALITY SCORING ENGINE 🤔
# ======================================================================


class MetaFrameworkUniversalityScoring:
    """
    TIER 2: Scores the universality and foundational value of meta-cognitive frameworks

    These are the "laws of thought" - universal thinking patterns that apply
    across domains and provide foundational cognitive capabilities.
    """

    def __init__(self):
        self.universality_indicators = {
            "decision_making": [
                "decision",
                "choose",
                "choice",
                "option",
                "alternative",
            ],
            "uncertainty": ["uncertain", "unknown", "ambiguous", "unclear", "risk"],
            "bias_mitigation": ["bias", "error", "mistake", "assumption", "prejudice"],
            "analytical_thinking": [
                "analysis",
                "analyze",
                "examine",
                "investigate",
                "study",
            ],
            "strategic_planning": ["strategy", "plan", "future", "long-term", "vision"],
            "problem_solving": ["problem", "solution", "solve", "challenge", "issue"],
        }

        self.meta_framework_patterns = {
            "UNCERTAINTY_DECISION": {
                "universal_domains": [
                    "business",
                    "investment",
                    "strategy",
                    "policy",
                    "medical",
                ],
                "foundational_depth": 0.95,
                "complexity_scaling": 0.9,
                "transferability": 0.98,
                "bias_resistance": 0.85,
            },
            "BIAS_MITIGATION": {
                "universal_domains": [
                    "decision-making",
                    "analysis",
                    "research",
                    "evaluation",
                ],
                "foundational_depth": 0.92,
                "complexity_scaling": 0.85,
                "transferability": 0.95,
                "bias_resistance": 0.98,  # Obviously high for bias mitigation
            },
            "DECISION_TRILEMMA": {
                "universal_domains": ["strategy", "operations", "crisis", "investment"],
                "foundational_depth": 0.88,
                "complexity_scaling": 0.92,
                "transferability": 0.90,
                "bias_resistance": 0.75,
            },
        }

    def score_meta_framework_universality(
        self, problem_framework: str, nway_pattern: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Score meta-cognitive framework universality

        Returns scores for:
        - problem_type_coverage: How many problem types it covers
        - transferability: Works across domains?
        - foundational_depth: How fundamental?
        - complexity_handling: Scales with problem complexity?
        - bias_resistance: Reduces cognitive errors?
        """

        pattern_id = nway_pattern.get("pattern_id", "")
        pattern_type = nway_pattern.get("pattern_type", "")
        application_domains = nway_pattern.get("application_domain", [])

        logger.info(f"🤔 Scoring meta-framework universality for {pattern_id}")

        # Only score if this is actually a meta-framework
        if pattern_type != "meta_framework":
            return self._zero_scores()

        # 1. Problem Type Coverage
        problem_coverage = self._calculate_problem_type_coverage(
            problem_framework, application_domains
        )

        # 2. Transferability Assessment
        transferability = self._calculate_transferability(
            pattern_id, application_domains
        )

        # 3. Foundational Depth
        foundational_depth = self._calculate_foundational_depth(
            pattern_id, nway_pattern
        )

        # 4. Complexity Handling Capability
        complexity_handling = self._calculate_complexity_handling(
            pattern_id, problem_framework
        )

        # 5. Bias Resistance
        bias_resistance = self._calculate_bias_resistance(pattern_id, nway_pattern)

        scores = {
            "problem_type_coverage": problem_coverage,
            "transferability": transferability,
            "foundational_depth": foundational_depth,
            "complexity_handling": complexity_handling,
            "bias_resistance": bias_resistance,
        }

        logger.info(f"🧠 Meta-framework scores for {pattern_id}: {scores}")
        return scores

    def _zero_scores(self) -> Dict[str, float]:
        """Return zero scores for non-meta-frameworks"""
        return {
            "problem_type_coverage": 0.0,
            "transferability": 0.0,
            "foundational_depth": 0.0,
            "complexity_handling": 0.0,
            "bias_resistance": 0.0,
        }

    def _calculate_problem_type_coverage(
        self, problem_framework: str, domains: List[str]
    ) -> float:
        """Calculate how many types of problems this framework covers"""
        if not domains:
            return 0.3  # Low coverage if no domains specified

        text_lower = problem_framework.lower()

        # Check coverage across different problem categories
        covered_categories = 0
        total_categories = len(self.universality_indicators)

        for category, indicators in self.universality_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                covered_categories += 1

        # Base coverage from problem type detection
        base_coverage = covered_categories / total_categories

        # Bonus for breadth of specified domains
        domain_bonus = min(0.3, len(domains) * 0.05)

        final_coverage = min(1.0, base_coverage + domain_bonus)
        return final_coverage

    def _calculate_transferability(self, pattern_id: str, domains: List[str]) -> float:
        """How well does this framework transfer across domains?"""

        # Check known patterns
        for pattern_name, pattern_data in self.meta_framework_patterns.items():
            if pattern_name in pattern_id.upper():
                return pattern_data["transferability"]

        # Calculate based on domain breadth
        if len(domains) >= 5:
            return 0.9  # High transferability
        elif len(domains) >= 3:
            return 0.75  # Medium transferability
        elif len(domains) >= 2:
            return 0.6  # Some transferability
        else:
            return 0.4  # Limited transferability

    def _calculate_foundational_depth(
        self, pattern_id: str, nway_pattern: Dict[str, Any]
    ) -> float:
        """How fundamental/foundational is this framework?"""

        # Check known patterns
        for pattern_name, pattern_data in self.meta_framework_patterns.items():
            if pattern_name in pattern_id.upper():
                return pattern_data["foundational_depth"]

        # Analyze mental models involved for foundational concepts
        models_involved = nway_pattern.get("models_involved", [])
        foundational_models = {
            "critical-thinking",
            "systems-thinking",
            "first-principles-thinking",
            "probability-theory",
            "logic-models",
            "intellectual-humility",
        }

        foundational_count = sum(
            1 for model in models_involved if model in foundational_models
        )
        foundational_ratio = (
            foundational_count / len(models_involved) if models_involved else 0
        )

        return min(1.0, foundational_ratio + 0.3)  # Base bonus for being a framework

    def _calculate_complexity_handling(
        self, pattern_id: str, problem_framework: str
    ) -> float:
        """How well does this scale with problem complexity?"""

        # Check known patterns
        for pattern_name, pattern_data in self.meta_framework_patterns.items():
            if pattern_name in pattern_id.upper():
                return pattern_data["complexity_scaling"]

        # Analyze problem complexity indicators
        complexity_indicators = [
            "complex",
            "complicated",
            "multiple",
            "interdependent",
            "dynamic",
            "uncertain",
            "ambiguous",
            "multifaceted",
            "systemic",
            "strategic",
        ]

        text_lower = problem_framework.lower()
        complexity_signals = sum(
            1 for indicator in complexity_indicators if indicator in text_lower
        )

        # Frameworks that handle complexity well score higher on complex problems
        if complexity_signals >= 3:
            return 0.85  # High complexity handling
        elif complexity_signals >= 2:
            return 0.7  # Medium complexity handling
        else:
            return 0.55  # Basic complexity handling

    def _calculate_bias_resistance(
        self, pattern_id: str, nway_pattern: Dict[str, Any]
    ) -> float:
        """How well does this framework resist cognitive biases?"""

        # Check known patterns
        for pattern_name, pattern_data in self.meta_framework_patterns.items():
            if pattern_name in pattern_id.upper():
                return pattern_data["bias_resistance"]

        # Analyze for bias-resistant mental models
        models_involved = nway_pattern.get("models_involved", [])
        bias_resistant_models = {
            "outside-view",
            "critical-thinking",
            "intellectual-humility",
            "evidence-based-reasoning",
            "scientific-method",
            "logic-models",
        }

        resistant_count = sum(
            1 for model in models_involved if model in bias_resistant_models
        )
        resistance_ratio = (
            resistant_count / len(models_involved) if models_involved else 0
        )

        return min(1.0, resistance_ratio + 0.2)  # Base bonus for systematic thinking


# ======================================================================
# TIER 3: COGNITIVE CLUSTER EXPERTISE SCORING ENGINE 🧠
# ======================================================================


class CognitiveClusterExpertiseScoring:
    """
    TIER 3: Scores professional cognitive clusters and their domain expertise

    These are complete "professional thinking toolkits" - the mental equipment
    that defines how experts in different fields approach problems.
    """

    def __init__(self):
        self.professional_domains = {
            "analytical": ["analysis", "data", "research", "investigation", "evidence"],
            "strategic": ["strategy", "planning", "future", "competitive", "advantage"],
            "creative": ["creative", "innovation", "design", "brainstorm", "ideation"],
            "research": ["research", "study", "scientific", "hypothesis", "experiment"],
            "diagnostic": ["problem", "diagnosis", "troubleshoot", "debug", "solve"],
            "educational": ["learn", "teach", "education", "training", "development"],
        }

        self.cluster_expertise_patterns = {
            "ANALYST_CLUSTER": {
                "domain_specialization": 0.95,
                "toolkit_completeness": 0.92,
                "synergy_coherence": 0.88,
                "professional_authenticity": 0.90,
            },
            "STRATEGIST_CLUSTER": {
                "domain_specialization": 0.90,
                "toolkit_completeness": 0.85,
                "synergy_coherence": 0.92,
                "professional_authenticity": 0.88,
            },
            "RESEARCHER_CLUSTER": {
                "domain_specialization": 0.93,
                "toolkit_completeness": 0.95,
                "synergy_coherence": 0.85,
                "professional_authenticity": 0.92,
            },
        }

    def score_cognitive_cluster_expertise(
        self, problem_framework: str, nway_cluster: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Score cognitive cluster expertise and fit

        Returns scores for:
        - domain_specialization: How specialized for specific domains?
        - toolkit_completeness: Complete professional toolkit?
        - synergy_coherence: Do models work together well?
        - instructional_clarity: Clear activation cues?
        - professional_authenticity: Real expert behavior?
        """

        cluster_id = nway_cluster.get("cluster_id", "")
        cognitive_domain = nway_cluster.get("cognitive_domain", "")
        professional_archetype = nway_cluster.get("professional_archetype", "")
        models_involved = nway_cluster.get("models_involved", [])

        logger.info(f"🧠 Scoring cognitive cluster expertise for {cluster_id}")

        # 1. Domain Specialization
        domain_specialization = self._calculate_domain_specialization(
            problem_framework, cognitive_domain, professional_archetype
        )

        # 2. Toolkit Completeness
        toolkit_completeness = self._calculate_toolkit_completeness(
            cluster_id, models_involved, cognitive_domain
        )

        # 3. Synergy Coherence
        synergy_coherence = self._calculate_synergy_coherence(
            cluster_id, models_involved
        )

        # 4. Instructional Clarity
        instructional_clarity = self._calculate_instructional_clarity(nway_cluster)

        # 5. Professional Authenticity
        professional_authenticity = self._calculate_professional_authenticity(
            cluster_id, professional_archetype, models_involved
        )

        scores = {
            "domain_specialization": domain_specialization,
            "toolkit_completeness": toolkit_completeness,
            "synergy_coherence": synergy_coherence,
            "instructional_clarity": instructional_clarity,
            "professional_authenticity": professional_authenticity,
        }

        logger.info(f"🎯 Cluster expertise scores for {cluster_id}: {scores}")
        return scores

    def _calculate_domain_specialization(
        self, problem_framework: str, domain: str, archetype: str
    ) -> float:
        """How well does this cluster specialize for the problem domain?"""

        text_lower = problem_framework.lower()

        # Check direct domain match
        domain_match = 0.0
        if domain and domain.lower() in text_lower:
            domain_match = 0.4

        # Check archetype match
        archetype_match = 0.0
        if archetype and archetype.lower() in text_lower:
            archetype_match = 0.3

        # Check for domain-specific indicators
        indicator_match = 0.0
        if domain in self.professional_domains:
            indicators = self.professional_domains[domain]
            matched_indicators = sum(
                1 for indicator in indicators if indicator in text_lower
            )
            indicator_match = min(0.4, matched_indicators * 0.1)

        total_specialization = domain_match + archetype_match + indicator_match
        return min(1.0, total_specialization)

    def _calculate_toolkit_completeness(
        self, cluster_id: str, models_involved: List[str], domain: str
    ) -> float:
        """How complete is this professional toolkit?"""

        # Check known cluster patterns
        for pattern_name, pattern_data in self.cluster_expertise_patterns.items():
            if pattern_name in cluster_id.upper():
                return pattern_data["toolkit_completeness"]

        # Calculate based on model count and diversity
        model_count = len(models_involved)

        # Different domains require different toolkit sizes
        expected_toolkit_sizes = {
            "analytical": 7,
            "strategic": 6,
            "research": 8,
            "creative": 5,
            "diagnostic": 6,
            "educational": 7,
        }

        expected_size = expected_toolkit_sizes.get(domain, 6)
        completeness_ratio = min(1.0, model_count / expected_size)

        # Bonus for having foundational models
        foundational_models = {
            "critical-thinking",
            "systems-thinking",
            "evidence-based-reasoning",
        }
        foundational_count = sum(
            1 for model in models_involved if model in foundational_models
        )
        foundational_bonus = min(0.2, foundational_count * 0.1)

        total_completeness = completeness_ratio + foundational_bonus
        return min(1.0, total_completeness)

    def _calculate_synergy_coherence(
        self, cluster_id: str, models_involved: List[str]
    ) -> float:
        """How well do the models in this cluster work together?"""

        # Check known cluster patterns
        for pattern_name, pattern_data in self.cluster_expertise_patterns.items():
            if pattern_name in cluster_id.upper():
                return pattern_data["synergy_coherence"]

        # Analyze model relationships
        # Models that commonly appear together have high coherence
        analytical_models = {
            "critical-thinking",
            "root-cause-analysis",
            "evidence-based-reasoning",
            "correlation-vs-causation",
            "logic-models",
            "first-principles-thinking",
        }

        strategic_models = {
            "systems-thinking",
            "second-order-thinking",
            "outside-view",
            "scenario-analysis",
            "identifying-what-is-important",
        }

        research_models = {
            "scientific-method",
            "evidence-based-reasoning",
            "experimentation",
            "probability-theory",
            "intellectual-humility",
            "peer-review",
        }

        # Calculate coherence based on model clustering
        analytical_count = sum(
            1 for model in models_involved if model in analytical_models
        )
        strategic_count = sum(
            1 for model in models_involved if model in strategic_models
        )
        research_count = sum(1 for model in models_involved if model in research_models)

        # High coherence if models cluster in same domain
        max_cluster = max(analytical_count, strategic_count, research_count)
        total_models = len(models_involved)

        if total_models == 0:
            return 0.5

        coherence_ratio = max_cluster / total_models

        # Bonus for having some cross-domain models (but not too many)
        cross_domain_models = total_models - max_cluster
        if 1 <= cross_domain_models <= 2:
            coherence_bonus = 0.1
        else:
            coherence_bonus = 0.0

        total_coherence = coherence_ratio + coherence_bonus
        return min(1.0, total_coherence)

    def _calculate_instructional_clarity(self, nway_cluster: Dict[str, Any]) -> float:
        """How clear are the instructions for activating this cluster?"""

        instructional_cues = nway_cluster.get("instructional_cues", "")

        if not instructional_cues:
            return 0.3  # Low clarity without instructions

        # Count instructional elements
        instruction_indicators = [
            "apply",
            "use",
            "employ",
            "activate",
            "engage",
            "implement",
            "step",
            "first",
            "then",
            "next",
            "finally",
            "when",
            "if",
            "during",
            "while",
        ]

        cues_lower = instructional_cues.lower()
        instruction_count = sum(
            1 for indicator in instruction_indicators if indicator in cues_lower
        )

        # Length and structure matter for clarity
        word_count = len(cues_lower.split())
        structure_score = min(1.0, instruction_count * 0.15)
        length_score = min(0.4, word_count * 0.01)  # Cap at reasonable length

        total_clarity = structure_score + length_score
        return min(1.0, total_clarity)

    def _calculate_professional_authenticity(
        self, cluster_id: str, archetype: str, models_involved: List[str]
    ) -> float:
        """How authentically does this represent real professional expertise?"""

        # Check known cluster patterns
        for pattern_name, pattern_data in self.cluster_expertise_patterns.items():
            if pattern_name in cluster_id.upper():
                return pattern_data["professional_authenticity"]

        # Analyze model alignment with professional archetype
        archetype_model_mapping = {
            "analyst": {
                "critical-thinking",
                "root-cause-analysis",
                "evidence-based-reasoning",
                "correlation-vs-causation",
                "logic-models",
                "statistics",
            },
            "strategist": {
                "systems-thinking",
                "second-order-thinking",
                "outside-view",
                "scenario-analysis",
                "competitive-analysis",
                "game-theory",
            },
            "researcher": {
                "scientific-method",
                "evidence-based-reasoning",
                "experimentation",
                "intellectual-humility",
                "peer-review",
                "hypothesis-testing",
            },
        }

        if archetype in archetype_model_mapping:
            expected_models = archetype_model_mapping[archetype]
            matched_models = sum(
                1 for model in models_involved if model in expected_models
            )
            authenticity_ratio = matched_models / len(expected_models)
        else:
            authenticity_ratio = 0.5  # Neutral for unknown archetypes

        # Bonus for avoiding contradictory models
        contradictory_pairs = [
            ("action-bias", "analysis-paralysis"),
            ("intuition", "evidence-based-reasoning"),
            ("optimism-bias", "outside-view"),
        ]

        contradiction_penalty = 0.0
        for model1, model2 in contradictory_pairs:
            if model1 in models_involved and model2 in models_involved:
                contradiction_penalty += 0.1

        total_authenticity = authenticity_ratio - contradiction_penalty
        return max(0.0, min(1.0, total_authenticity))


# ======================================================================
# TIER 4: CONTEXTUAL AFFINITY SCORING ENGINE 🔧
# ======================================================================


class ContextualAffinityScoring:
    """
    TIER 4: Scores task-specific fit and contextual appropriateness

    This handles domain-specific toolkits and specialized applications
    that are perfect for specific contexts but may not generalize.
    """

    def __init__(self):
        self.task_contexts = {
            "creativity": [
                "creative",
                "innovation",
                "brainstorm",
                "ideation",
                "design",
            ],
            "diagnosis": ["problem", "troubleshoot", "debug", "diagnose", "fix"],
            "learning": ["learn", "teach", "education", "training", "skill"],
            "persuasion": ["persuade", "influence", "convince", "negotiate", "sell"],
            "decision": ["decide", "choice", "option", "alternative", "select"],
        }

        self.resource_requirements = {
            "creativity": {"time": "medium", "expertise": "low", "tools": "minimal"},
            "diagnosis": {"time": "high", "expertise": "high", "tools": "medium"},
            "learning": {"time": "high", "expertise": "medium", "tools": "low"},
            "persuasion": {"time": "medium", "expertise": "medium", "tools": "low"},
            "decision": {"time": "medium", "expertise": "medium", "tools": "medium"},
        }

    def score_contextual_affinity(
        self, problem_framework: str, nway_pattern: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Score contextual fit for domain-specific applications

        Returns scores for:
        - task_specificity: Perfect for this specific task?
        - prerequisite_match: Right conditions present?
        - outcome_predictability: Reliable results?
        - resource_efficiency: Time/effort vs benefit?
        - complementary_fit: Works with other selected NWAYs?
        """

        pattern_id = nway_pattern.get("pattern_id", "")
        application_domains = nway_pattern.get("application_domain", [])
        pattern_type = nway_pattern.get("pattern_type", "")

        logger.info(f"🔧 Scoring contextual affinity for {pattern_id}")

        # 1. Task Specificity
        task_specificity = self._calculate_task_specificity(
            problem_framework, application_domains, pattern_type
        )

        # 2. Prerequisite Match
        prerequisite_match = self._calculate_prerequisite_match(
            problem_framework, nway_pattern
        )

        # 3. Outcome Predictability
        outcome_predictability = self._calculate_outcome_predictability(
            pattern_id, pattern_type, application_domains
        )

        # 4. Resource Efficiency
        resource_efficiency = self._calculate_resource_efficiency(
            application_domains, problem_framework
        )

        # 5. Complementary Fit (placeholder - needs other selected NWAYs)
        complementary_fit = self._calculate_complementary_fit(pattern_id, nway_pattern)

        scores = {
            "task_specificity": task_specificity,
            "prerequisite_match": prerequisite_match,
            "outcome_predictability": outcome_predictability,
            "resource_efficiency": resource_efficiency,
            "complementary_fit": complementary_fit,
        }

        logger.info(f"🎯 Contextual affinity scores for {pattern_id}: {scores}")
        return scores

    def _calculate_task_specificity(
        self, problem_framework: str, domains: List[str], pattern_type: str
    ) -> float:
        """How specific is this tool for the identified task?"""

        text_lower = problem_framework.lower()

        # Domain toolkits should have high specificity
        if pattern_type == "domain_toolkit":
            specificity_bonus = 0.3
        else:
            specificity_bonus = 0.0

        # Check for domain-specific matches
        domain_matches = 0
        for domain in domains:
            if domain in self.task_contexts:
                context_indicators = self.task_contexts[domain]
                if any(indicator in text_lower for indicator in context_indicators):
                    domain_matches += 1

        domain_specificity = min(0.7, domain_matches * 0.35)

        total_specificity = domain_specificity + specificity_bonus
        return min(1.0, total_specificity)

    def _calculate_prerequisite_match(
        self, problem_framework: str, nway_pattern: Dict[str, Any]
    ) -> float:
        """Are the required conditions/prerequisites present?"""

        prerequisite_knowledge = nway_pattern.get("prerequisite_knowledge", [])

        if not prerequisite_knowledge:
            return 0.8  # No specific prerequisites = good match

        text_lower = problem_framework.lower()

        # Check for prerequisite indicators in problem
        matched_prerequisites = 0
        for prereq in prerequisite_knowledge:
            if prereq.lower() in text_lower:
                matched_prerequisites += 1

        if len(prerequisite_knowledge) == 0:
            return 0.8

        prerequisite_ratio = matched_prerequisites / len(prerequisite_knowledge)

        # High penalty for missing critical prerequisites
        if prerequisite_ratio < 0.5:
            return prerequisite_ratio * 0.6  # Penalty for missing prereqs
        else:
            return prerequisite_ratio

    def _calculate_outcome_predictability(
        self, pattern_id: str, pattern_type: str, domains: List[str]
    ) -> float:
        """How predictable are the outcomes from using this pattern?"""

        # Domain toolkits tend to have more predictable outcomes
        if pattern_type == "domain_toolkit":
            base_predictability = 0.7
        elif pattern_type == "meta_framework":
            base_predictability = 0.6  # More abstract, less predictable
        else:
            base_predictability = 0.5

        # Well-established domains have higher predictability
        established_domains = {"creativity", "diagnosis", "learning", "decision"}
        established_count = sum(
            1 for domain in domains if domain in established_domains
        )

        domain_bonus = min(0.25, established_count * 0.125)

        total_predictability = base_predictability + domain_bonus
        return min(1.0, total_predictability)

    def _calculate_resource_efficiency(
        self, domains: List[str], problem_framework: str
    ) -> float:
        """What's the resource efficiency (benefit vs cost)?"""

        # Estimate resource requirements
        total_time_cost = 0
        total_expertise_cost = 0

        for domain in domains:
            if domain in self.resource_requirements:
                reqs = self.resource_requirements[domain]

                # Time cost mapping
                time_costs = {"low": 1, "medium": 2, "high": 3}
                total_time_cost += time_costs.get(reqs["time"], 2)

                # Expertise cost mapping
                expertise_costs = {"low": 1, "medium": 2, "high": 3}
                total_expertise_cost += expertise_costs.get(reqs["expertise"], 2)

        # Calculate efficiency (lower cost = higher efficiency)
        if not domains:
            return 0.5

        avg_time_cost = total_time_cost / len(domains)
        avg_expertise_cost = total_expertise_cost / len(domains)

        # Efficiency is inverse of cost
        time_efficiency = 1.0 - ((avg_time_cost - 1) / 2)  # Normalize to 0-1
        expertise_efficiency = 1.0 - ((avg_expertise_cost - 1) / 2)  # Normalize to 0-1

        # Weight time and expertise equally
        total_efficiency = (time_efficiency + expertise_efficiency) / 2
        return max(0.0, min(1.0, total_efficiency))

    def _calculate_complementary_fit(
        self, pattern_id: str, nway_pattern: Dict[str, Any]
    ) -> float:
        """How well does this fit with other potential selections?"""

        # This is a placeholder - in the full implementation, this would
        # analyze compatibility with other NWAYs being considered

        models_involved = nway_pattern.get("models_involved", [])

        # Penalize for having too many mental models (complexity)
        model_count = len(models_involved)
        if model_count > 8:
            complexity_penalty = (model_count - 8) * 0.05
        else:
            complexity_penalty = 0.0

        # Base complementary fit
        base_fit = 0.7

        # Bonus for having connector models that work well with others
        connector_models = {
            "systems-thinking",
            "critical-thinking",
            "second-order-thinking",
        }
        connector_count = sum(
            1 for model in models_involved if model in connector_models
        )
        connector_bonus = min(0.2, connector_count * 0.1)

        total_fit = base_fit + connector_bonus - complexity_penalty
        return max(0.0, min(1.0, total_fit))


# ======================================================================
# COGNITIVE CHEMISTRY SCORING ORCHESTRATOR
# ======================================================================


class CognitiveChemistryScoring:
    """
    Master orchestrator for all four tiers of cognitive chemistry scoring

    This is the revolutionary breakthrough - scoring cognitive REACTIONS
    rather than just individual components.
    """

    def __init__(self):
        self.lollapalooza_scorer = LollapaloozaTriggerScoring()
        self.meta_framework_scorer = MetaFrameworkUniversalityScoring()
        self.cluster_scorer = CognitiveClusterExpertiseScoring()
        self.contextual_scorer = ContextualAffinityScoring()

        logger.info("🚀 Cognitive Chemistry Scoring Engine initialized")

    def score_nway_combination(
        self, problem_framework: str, nway_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Score an NWAY interaction using all four tiers

        This is the core breakthrough - multi-dimensional scoring that
        captures the cognitive chemistry potential of the interaction.
        """

        interaction_id = nway_data.get("interaction_id", "unknown")
        logger.info(f"⚡ Scoring cognitive chemistry for {interaction_id}")

        all_scores = {}

        # TIER 1: Lollapalooza Trigger Scoring (if applicable)
        if self._is_lollapalooza_compound(nway_data):
            lolla_scores = self.lollapalooza_scorer.score_environmental_triggers(
                problem_framework, nway_data
            )
            all_scores.update(lolla_scores)
        else:
            # Zero out lollapalooza scores for non-compounds
            all_scores.update(
                {
                    "environmental_match_score": 0.0,
                    "amplification_intensity": 0.0,
                    "rational_override_probability": 0.0,
                    "multiplicative_effect": 1.0,  # Additive by default
                    "predictability": 0.0,
                }
            )

        # TIER 2: Meta-Framework Universality Scoring (if applicable)
        if self._is_meta_framework(nway_data):
            meta_scores = self.meta_framework_scorer.score_meta_framework_universality(
                problem_framework, nway_data
            )
            all_scores.update(meta_scores)
        else:
            all_scores.update(
                {
                    "problem_type_coverage": 0.0,
                    "transferability": 0.0,
                    "foundational_depth": 0.0,
                    "complexity_handling": 0.0,
                    "bias_resistance": 0.0,
                }
            )

        # TIER 3: Cognitive Cluster Expertise Scoring (if applicable)
        if self._is_cognitive_cluster(nway_data):
            cluster_scores = self.cluster_scorer.score_cognitive_cluster_expertise(
                problem_framework, nway_data
            )
            all_scores.update(cluster_scores)
        else:
            all_scores.update(
                {
                    "domain_specialization": 0.0,
                    "toolkit_completeness": 0.0,
                    "synergy_coherence": 0.0,
                    "instructional_clarity": 0.0,
                    "professional_authenticity": 0.0,
                }
            )

        # TIER 4: Contextual Affinity Scoring (always applicable)
        contextual_scores = self.contextual_scorer.score_contextual_affinity(
            problem_framework, nway_data
        )
        all_scores.update(contextual_scores)

        logger.info(
            f"✅ Complete scoring for {interaction_id}: {len(all_scores)} dimensions"
        )
        return all_scores

    def _is_lollapalooza_compound(self, nway_data: Dict[str, Any]) -> bool:
        """Check if this is a true Lollapalooza compound"""
        # Could be stored in database or inferred from structure
        return (
            "environmental_triggers" in nway_data
            or "psychological_principles" in nway_data
            or "AUCTION" in nway_data.get("interaction_id", "").upper()
            or "COCACOLA" in nway_data.get("interaction_id", "").upper()
            or "TUPPERWARE" in nway_data.get("interaction_id", "").upper()
        )

    def _is_meta_framework(self, nway_data: Dict[str, Any]) -> bool:
        """Check if this is a meta-cognitive framework"""
        pattern_type = nway_data.get("pattern_type", "")
        return pattern_type == "meta_framework" or any(
            keyword in nway_data.get("interaction_id", "").upper()
            for keyword in ["DECISION", "BIAS", "UNCERTAINTY", "TRILEMMA"]
        )

    def _is_cognitive_cluster(self, nway_data: Dict[str, Any]) -> bool:
        """Check if this is a cognitive cluster"""
        return (
            "cognitive_domain" in nway_data
            or "professional_archetype" in nway_data
            or "CLUSTER" in nway_data.get("interaction_id", "").upper()
        )


# ======================================================================
# FACTORY FUNCTION
# ======================================================================


def get_cognitive_chemistry_scoring() -> CognitiveChemistryScoring:
    """Get the Cognitive Chemistry Scoring Engine instance"""
    return CognitiveChemistryScoring()


if __name__ == "__main__":
    print("🧠 COGNITIVE CHEMISTRY SCORING ENGINES - Phase 2")
    print("   Revolutionary four-tier scoring system")
    print("   From scoring components to scoring REACTIONS")

    # Test the scoring engine
    scorer = get_cognitive_chemistry_scoring()

    test_problem = "We need to analyze our auction strategy for the upcoming competitive bidding process"
    test_nway = {
        "interaction_id": "NWAY_AUCTION_001",
        "environmental_triggers": ["auction", "bidding", "competitive"],
        "psychological_principles": [
            "social-proof",
            "scarcity",
            "commitment",
            "loss-aversion",
        ],
    }

    scores = scorer.score_nway_combination(test_problem, test_nway)
    print(f"\n🎯 Test Scores: {scores}")
