#!/usr/bin/env python3
"""
SYNERGISTIC COMPATIBILITY MATRIX SERVICE
Phase 3 of Operation: Cognitive Particle Accelerator

This is the revolutionary breakthrough - predicting what happens when you combine
different cognitive "reagents" (NWAY interactions). We're moving from scoring
individual components to scoring CHEMICAL REACTIONS between mental patterns.

This captures emergent effects, conflicts, and synergies that arise when
different thinking frameworks interact with each other.
"""

from typing import Dict, List, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# ======================================================================
# REACTION TYPES AND COMPATIBILITY PATTERNS
# ======================================================================


class ReactionType(Enum):
    """Types of reactions that can occur when combining NWAYs"""

    SYNERGISTIC = "synergistic"  # 1 + 1 = 3 (amplification)
    ADDITIVE = "additive"  # 1 + 1 = 2 (simple combination)
    NEUTRAL = "neutral"  # 1 + 1 = 1.5 (mild benefit)
    CONFLICTING = "conflicting"  # 1 + 1 = 0.5 (interference)
    DESTRUCTIVE = "destructive"  # 1 + 1 = 0 (cancellation)


class CompatibilityCategory(Enum):
    """Categories of compatibility patterns"""

    PERFECT_SYNERGY = "perfect_synergy"
    HIGH_COMPATIBILITY = "high_compatibility"
    MILD_COMPATIBILITY = "mild_compatibility"
    NEUTRAL_COEXISTENCE = "neutral_coexistence"
    MINOR_CONFLICT = "minor_conflict"
    MAJOR_CONFLICT = "major_conflict"
    DESTRUCTIVE_INTERFERENCE = "destructive_interference"


@dataclass
class CompatibilityResult:
    """Result of compatibility analysis between two NWAYs"""

    nway_a: str
    nway_b: str
    reaction_type: ReactionType
    compatibility_category: CompatibilityCategory
    reinforcement_effect: float  # -1.0 to +2.0
    conflict_risk: float  # 0.0 to 1.0
    emergence_potential: float  # 0.0 to 1.0
    stability: float  # 0.0 to 1.0
    cognitive_load_increase: float  # 0.0 to 2.0
    integration_difficulty: float  # 0.0 to 1.0
    context_dependency: float  # 0.0 to 1.0
    confidence_level: float  # 0.0 to 1.0
    explanation: str
    discovered_via: str  # "theoretical", "experimental", "observed"


# ======================================================================
# SYNERGISTIC COMPATIBILITY MATRIX ENGINE
# ======================================================================


class SynergisticCompatibilityMatrix:
    """
    The revolutionary engine that predicts cognitive chemistry reactions

    This is the core breakthrough - understanding how different thinking
    patterns interact, conflict, or amplify each other when combined.
    """

    def __init__(self):
        self.compatibility_cache = {}
        self.known_synergies = self._initialize_known_synergies()
        self.known_conflicts = self._initialize_known_conflicts()
        self.emergence_patterns = self._initialize_emergence_patterns()

        logger.info("🧪 Synergistic Compatibility Matrix initialized")

    def calculate_compatibility(
        self, nway_a: Dict[str, Any], nway_b: Dict[str, Any]
    ) -> CompatibilityResult:
        """
        Calculate the compatibility between two NWAY interactions

        This is the revolutionary breakthrough - predicting cognitive chemistry
        """
        id_a = nway_a.get("interaction_id", "unknown_a")
        id_b = nway_b.get("interaction_id", "unknown_b")

        # Check cache first
        cache_key = self._get_cache_key(id_a, id_b)
        if cache_key in self.compatibility_cache:
            logger.info(f"🔄 Using cached compatibility for {id_a} + {id_b}")
            return self.compatibility_cache[cache_key]

        logger.info(f"🧪 Calculating compatibility: {id_a} + {id_b}")

        # 1. Analyze Mental Model Overlaps and Interactions
        model_analysis = self._analyze_mental_model_interactions(nway_a, nway_b)

        # 2. Check for Known Synergy Patterns
        synergy_analysis = self._check_known_synergies(nway_a, nway_b)

        # 3. Detect Potential Conflicts
        conflict_analysis = self._detect_conflicts(nway_a, nway_b)

        # 4. Assess Emergence Potential
        emergence_analysis = self._assess_emergence_potential(nway_a, nway_b)

        # 5. Calculate Cognitive Load Impact
        load_analysis = self._calculate_cognitive_load_impact(nway_a, nway_b)

        # 6. Determine Integration Difficulty
        integration_analysis = self._assess_integration_difficulty(nway_a, nway_b)

        # 7. Synthesize Final Compatibility Result
        result = self._synthesize_compatibility_result(
            id_a,
            id_b,
            model_analysis,
            synergy_analysis,
            conflict_analysis,
            emergence_analysis,
            load_analysis,
            integration_analysis,
        )

        # Cache the result
        self.compatibility_cache[cache_key] = result

        logger.info(
            f"⚡ Compatibility result: {result.reaction_type.value} ({result.reinforcement_effect:.2f})"
        )
        return result

    def _get_cache_key(self, id_a: str, id_b: str) -> str:
        """Generate cache key for compatibility pair (order-independent)"""
        return f"{min(id_a, id_b)}|{max(id_a, id_b)}"

    def _initialize_known_synergies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize database of known synergistic combinations"""
        return {
            # Perfect analytical synergy
            "ANALYST_CLUSTER + BIAS_MITIGATION": {
                "reinforcement_effect": 1.8,
                "emergence_potential": 0.9,
                "explanation": "Analytical rigor perfectly complements bias mitigation for superior objectivity",
                "stability": 0.95,
                "discovered_via": "theoretical",
            },
            # Strategic thinking amplification
            "STRATEGIST_CLUSTER + UNCERTAINTY_DECISION": {
                "reinforcement_effect": 1.6,
                "emergence_potential": 0.85,
                "explanation": "Strategic frameworks enhanced by uncertainty management create robust decision-making",
                "stability": 0.9,
                "discovered_via": "theoretical",
            },
            # Research methodology synergy
            "RESEARCHER_CLUSTER + DIAGNOSTIC_SOLVING": {
                "reinforcement_effect": 1.7,
                "emergence_potential": 0.8,
                "explanation": "Research methodology amplifies diagnostic capabilities through rigorous validation",
                "stability": 0.88,
                "discovered_via": "theoretical",
            },
            # Creative problem-solving synergy
            "CREATIVITY + OUTLIER_ANALYSIS": {
                "reinforcement_effect": 1.5,
                "emergence_potential": 0.95,
                "explanation": "Creative thinking enhanced by outlier detection generates breakthrough innovations",
                "stability": 0.75,
                "discovered_via": "theoretical",
            },
            # Learning optimization synergy
            "LEARNING_TEACHING + BIAS_MITIGATION": {
                "reinforcement_effect": 1.4,
                "emergence_potential": 0.8,
                "explanation": "Educational frameworks with bias awareness create superior learning systems",
                "stability": 0.85,
                "discovered_via": "theoretical",
            },
        }

    def _initialize_known_conflicts(self) -> Dict[str, Dict[str, Any]]:
        """Initialize database of known conflicting combinations"""
        return {
            # Speed vs depth conflict
            "DECISION_TRILEMMA + LOLLAPALOOZA": {
                "conflict_risk": 0.9,
                "reinforcement_effect": -0.5,
                "explanation": "Rational decision analysis conflicts with irrational amplification effects",
                "stability": 0.3,
                "context_dependency": 0.95,
            },
            # Analysis paralysis vs action bias
            "ANALYST_CLUSTER + ENTREPRENEUR_AGENCY": {
                "conflict_risk": 0.7,
                "reinforcement_effect": 0.2,
                "explanation": "Deep analysis tendency conflicts with rapid action orientation",
                "stability": 0.6,
                "context_dependency": 0.8,
            },
            # Systematic vs creative tension
            "RESEARCHER_CLUSTER + CREATIVITY": {
                "conflict_risk": 0.6,
                "reinforcement_effect": 0.3,
                "explanation": "Rigorous methodology can inhibit free-flowing creative processes",
                "stability": 0.65,
                "context_dependency": 0.7,
            },
            # Multiple Lollapalooza interference
            "AUCTION + TUPPERWARE": {
                "conflict_risk": 0.8,
                "reinforcement_effect": -0.3,
                "explanation": "Multiple explosive psychological effects can interfere with each other",
                "stability": 0.4,
                "context_dependency": 0.9,
            },
        }

    def _initialize_emergence_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize patterns that create emergent capabilities"""
        return {
            # Strategic foresight emergence
            "STRATEGIST + OUTLIER_ANALYSIS": {
                "emergent_capability": "Strategic Early Warning System",
                "emergence_potential": 0.9,
                "description": "Combines strategic thinking with anomaly detection for superior foresight",
            },
            # Meta-learning emergence
            "LEARNING_TEACHING + BIAS_MITIGATION": {
                "emergent_capability": "Self-Correcting Learning System",
                "emergence_potential": 0.85,
                "description": "Creates learning systems that automatically identify and correct their own biases",
            },
            # Innovation intelligence emergence
            "CREATIVITY + ANALYST_CLUSTER": {
                "emergent_capability": "Systematic Innovation Engine",
                "emergence_potential": 0.8,
                "description": "Structured creativity that generates and validates breakthrough ideas systematically",
            },
            # Persuasion mastery emergence
            "STORYTELLER_MARKETER + BIAS_MITIGATION": {
                "emergent_capability": "Ethical Influence Mastery",
                "emergence_potential": 0.75,
                "description": "Powerful persuasion capabilities balanced by bias awareness for ethical influence",
            },
        }

    def _analyze_mental_model_interactions(
        self, nway_a: Dict[str, Any], nway_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze how the mental models in each NWAY interact"""

        models_a = set(nway_a.get("models_involved", []))
        models_b = set(nway_b.get("models_involved", []))

        # Calculate overlaps and interactions
        overlapping_models = models_a & models_b
        complementary_models = models_a ^ models_b  # Symmetric difference
        total_models = models_a | models_b

        overlap_ratio = (
            len(overlapping_models) / len(total_models) if total_models else 0
        )

        # Analyze model relationships
        synergistic_pairs = self._identify_synergistic_model_pairs(models_a, models_b)
        conflicting_pairs = self._identify_conflicting_model_pairs(models_a, models_b)

        return {
            "overlapping_models": overlapping_models,
            "complementary_models": complementary_models,
            "overlap_ratio": overlap_ratio,
            "synergistic_pairs": synergistic_pairs,
            "conflicting_pairs": conflicting_pairs,
            "total_model_count": len(total_models),
        }

    def _identify_synergistic_model_pairs(
        self, models_a: Set[str], models_b: Set[str]
    ) -> List[Tuple[str, str]]:
        """Identify mental model pairs that work synergistically together"""

        synergistic_pairs = [
            # Analysis amplifiers
            ("critical-thinking", "evidence-based-reasoning"),
            ("systems-thinking", "second-order-thinking"),
            ("root-cause-analysis", "first-principles-thinking"),
            # Strategic enhancers
            ("outside-view", "scenario-analysis"),
            ("game-theory", "competitive-analysis"),
            ("systems-thinking", "identifying-what-is-important"),
            # Bias reduction combinations
            ("intellectual-humility", "critical-thinking"),
            ("outside-view", "inside-view"),  # Balancing perspectives
            ("evidence-based-reasoning", "correlation-vs-causation"),
            # Creative amplifiers
            ("divergent-thinking", "lateral-thinking"),
            ("reframing-perspective", "first-principles-thinking"),
            ("scamper", "analogical-reasoning"),
            # Decision enhancement
            ("probability-theory", "outside-view"),
            ("risk-assessment", "margin-of-safety"),
            ("second-order-thinking", "inversion"),
        ]

        found_pairs = []
        for model_a in models_a:
            for model_b in models_b:
                for pair in synergistic_pairs:
                    if (model_a in pair and model_b in pair) or (
                        model_b in pair and model_a in pair
                    ):
                        found_pairs.append((model_a, model_b))

        return found_pairs

    def _identify_conflicting_model_pairs(
        self, models_a: Set[str], models_b: Set[str]
    ) -> List[Tuple[str, str]]:
        """Identify mental model pairs that conflict with each other"""

        conflicting_pairs = [
            # Speed vs depth conflicts
            ("action-bias", "analysis-paralysis"),
            ("making-decisions-quickly", "critical-thinking"),
            ("intuition", "evidence-based-reasoning"),
            # Optimism vs realism conflicts
            ("optimism-bias", "outside-view"),
            ("confidence", "intellectual-humility"),
            ("positive-thinking", "inversion"),
            # Structure vs flexibility conflicts
            ("checklists", "lateral-thinking"),
            ("systematic-approach", "divergent-thinking"),
            ("planning", "adaptation"),
            # Individual vs group conflicts
            ("self-reliance", "social-proof"),
            ("contrarian-thinking", "consensus-building"),
            ("independence", "collaboration"),
        ]

        found_conflicts = []
        for model_a in models_a:
            for model_b in models_b:
                for pair in conflicting_pairs:
                    if (model_a in pair and model_b in pair) or (
                        model_b in pair and model_a in pair
                    ):
                        found_conflicts.append((model_a, model_b))

        return found_conflicts

    def _check_known_synergies(
        self, nway_a: Dict[str, Any], nway_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check against database of known synergistic combinations"""

        id_a = nway_a.get("interaction_id", "").upper()
        id_b = nway_b.get("interaction_id", "").upper()

        # Extract key terms for matching
        key_terms_a = self._extract_key_terms(id_a)
        key_terms_b = self._extract_key_terms(id_b)

        # Check all known synergies
        best_synergy = None
        best_score = 0

        for synergy_key, synergy_data in self.known_synergies.items():
            match_score = self._calculate_synergy_match_score(
                key_terms_a, key_terms_b, synergy_key
            )

            if match_score > best_score:
                best_score = match_score
                best_synergy = synergy_data

        return {
            "synergy_found": best_synergy is not None,
            "synergy_data": best_synergy,
            "match_confidence": best_score,
        }

    def _extract_key_terms(self, interaction_id: str) -> Set[str]:
        """Extract key terms from interaction ID for matching"""
        key_term_mapping = {
            "ANALYST": "ANALYST_CLUSTER",
            "STRATEGIST": "STRATEGIST_CLUSTER",
            "RESEARCHER": "RESEARCHER_CLUSTER",
            "BIAS": "BIAS_MITIGATION",
            "UNCERTAINTY": "UNCERTAINTY_DECISION",
            "DECISION": "DECISION_TRILEMMA",
            "CREATIVITY": "CREATIVITY",
            "DIAGNOSTIC": "DIAGNOSTIC_SOLVING",
            "OUTLIER": "OUTLIER_ANALYSIS",
            "LEARNING": "LEARNING_TEACHING",
            "STORYTELLER": "STORYTELLER_MARKETER",
            "AUCTION": "AUCTION",
            "TUPPERWARE": "TUPPERWARE",
            "COCACOLA": "COCACOLA",
        }

        terms = set()
        for term, mapped_term in key_term_mapping.items():
            if term in interaction_id:
                terms.add(mapped_term)

        return terms

    def _calculate_synergy_match_score(
        self, terms_a: Set[str], terms_b: Set[str], synergy_key: str
    ) -> float:
        """Calculate how well the NWAY pair matches a known synergy pattern"""

        synergy_terms = set(synergy_key.split(" + "))
        nway_terms = terms_a | terms_b

        # Check for exact match
        if synergy_terms == nway_terms:
            return 1.0

        # Check for partial match
        matched_terms = synergy_terms & nway_terms
        if len(matched_terms) == len(synergy_terms):
            return 0.8  # All synergy terms present, but may have extras
        elif len(matched_terms) > 0:
            return len(matched_terms) / len(synergy_terms) * 0.6

        return 0.0

    def _detect_conflicts(
        self, nway_a: Dict[str, Any], nway_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect potential conflicts between NWAY interactions"""

        id_a = nway_a.get("interaction_id", "").upper()
        id_b = nway_b.get("interaction_id", "").upper()

        key_terms_a = self._extract_key_terms(id_a)
        key_terms_b = self._extract_key_terms(id_b)

        # Check known conflicts
        best_conflict = None
        best_score = 0

        for conflict_key, conflict_data in self.known_conflicts.items():
            match_score = self._calculate_synergy_match_score(
                key_terms_a, key_terms_b, conflict_key
            )

            if match_score > best_score:
                best_score = match_score
                best_conflict = conflict_data

        # Analyze type conflicts
        type_conflicts = self._analyze_type_conflicts(nway_a, nway_b)

        return {
            "conflict_found": best_conflict is not None
            or type_conflicts["has_conflicts"],
            "conflict_data": best_conflict,
            "type_conflicts": type_conflicts,
            "match_confidence": best_score,
        }

    def _analyze_type_conflicts(
        self, nway_a: Dict[str, Any], nway_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze conflicts based on NWAY types and characteristics"""

        # Multiple Lollapalooza compounds can interfere
        is_lolla_a = self._is_lollapalooza_compound(nway_a)
        is_lolla_b = self._is_lollapalooza_compound(nway_b)

        if is_lolla_a and is_lolla_b:
            return {
                "has_conflicts": True,
                "conflict_type": "multiple_lollapalooza",
                "severity": 0.8,
                "explanation": "Multiple Lollapalooza effects can interfere with each other",
            }

        # Meta-framework vs domain-specific conflicts
        is_meta_a = self._is_meta_framework(nway_a)
        is_meta_b = self._is_meta_framework(nway_b)
        is_domain_a = self._is_domain_specific(nway_a)
        is_domain_b = self._is_domain_specific(nway_b)

        if (is_meta_a and is_domain_b) or (is_meta_b and is_domain_a):
            return {
                "has_conflicts": True,
                "conflict_type": "abstraction_mismatch",
                "severity": 0.4,
                "explanation": "Abstract frameworks may conflict with specific domain tools",
            }

        return {"has_conflicts": False}

    def _assess_emergence_potential(
        self, nway_a: Dict[str, Any], nway_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess potential for emergent capabilities when combining NWAYs"""

        id_a = nway_a.get("interaction_id", "").upper()
        id_b = nway_b.get("interaction_id", "").upper()

        key_terms_a = self._extract_key_terms(id_a)
        key_terms_b = self._extract_key_terms(id_b)

        # Check known emergence patterns
        best_emergence = None
        best_score = 0

        for pattern_key, pattern_data in self.emergence_patterns.items():
            match_score = self._calculate_synergy_match_score(
                key_terms_a, key_terms_b, pattern_key
            )

            if match_score > best_score:
                best_score = match_score
                best_emergence = pattern_data

        # Calculate general emergence potential
        models_a = set(nway_a.get("models_involved", []))
        models_b = set(nway_b.get("models_involved", []))

        # Emergence is higher when combining different cognitive domains
        cognitive_diversity = self._calculate_cognitive_diversity(models_a, models_b)
        complementarity = self._calculate_complementarity(nway_a, nway_b)

        general_emergence = (cognitive_diversity + complementarity) / 2

        return {
            "known_emergence": best_emergence,
            "emergence_score": best_score,
            "general_emergence_potential": general_emergence,
            "emergent_capability": (
                best_emergence.get("emergent_capability") if best_emergence else None
            ),
        }

    def _calculate_cognitive_diversity(
        self, models_a: Set[str], models_b: Set[str]
    ) -> float:
        """Calculate cognitive diversity between two model sets"""

        cognitive_categories = {
            "analytical": {
                "critical-thinking",
                "root-cause-analysis",
                "evidence-based-reasoning",
            },
            "creative": {"divergent-thinking", "lateral-thinking", "brainstorming"},
            "strategic": {"systems-thinking", "second-order-thinking", "game-theory"},
            "social": {"empathy", "social-proof", "understanding-motivations"},
            "systematic": {"checklists", "scientific-method", "debugging-strategies"},
        }

        categories_a = set()
        categories_b = set()

        for category, category_models in cognitive_categories.items():
            if models_a & category_models:
                categories_a.add(category)
            if models_b & category_models:
                categories_b.add(category)

        total_categories = categories_a | categories_b
        different_categories = categories_a ^ categories_b  # Symmetric difference

        if not total_categories:
            return 0.5

        diversity_ratio = len(different_categories) / len(total_categories)
        return diversity_ratio

    def _calculate_complementarity(
        self, nway_a: Dict[str, Any], nway_b: Dict[str, Any]
    ) -> float:
        """Calculate how well the NWAYs complement each other"""

        # Different application domains increase complementarity
        domains_a = set(nway_a.get("application_domain", []))
        domains_b = set(nway_b.get("application_domain", []))

        domain_overlap = domains_a & domains_b
        total_domains = domains_a | domains_b

        if total_domains:
            domain_complementarity = 1.0 - (len(domain_overlap) / len(total_domains))
        else:
            domain_complementarity = 0.5

        # Different strengths/focuses increase complementarity
        focus_a = self._determine_focus_area(nway_a)
        focus_b = self._determine_focus_area(nway_b)

        focus_complementarity = 1.0 if focus_a != focus_b else 0.3

        return (domain_complementarity + focus_complementarity) / 2

    def _determine_focus_area(self, nway: Dict[str, Any]) -> str:
        """Determine the primary focus area of an NWAY"""

        interaction_id = nway.get("interaction_id", "").upper()

        if any(term in interaction_id for term in ["ANALYST", "ANALYSIS", "RESEARCH"]):
            return "analytical"
        elif any(
            term in interaction_id for term in ["STRATEGIST", "STRATEGIC", "PLANNING"]
        ):
            return "strategic"
        elif any(
            term in interaction_id for term in ["CREATIVE", "INNOVATION", "DESIGN"]
        ):
            return "creative"
        elif any(
            term in interaction_id for term in ["BIAS", "DECISION", "UNCERTAINTY"]
        ):
            return "meta_cognitive"
        elif any(
            term in interaction_id for term in ["LEARNING", "TEACHING", "EDUCATION"]
        ):
            return "educational"
        else:
            return "domain_specific"

    def _calculate_cognitive_load_impact(
        self, nway_a: Dict[str, Any], nway_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate the cognitive load impact of combining these NWAYs"""

        models_a = nway_a.get("models_involved", [])
        models_b = nway_b.get("models_involved", [])

        total_models = len(set(models_a) | set(models_b))
        overlapping_models = len(set(models_a) & set(models_b))

        # Base load from total unique models
        base_load = min(2.0, total_models / 10.0)  # Scale to 0-2.0

        # Reduction from overlapping models (shared mental overhead)
        overlap_reduction = overlapping_models * 0.05

        # Penalty for complexity interactions
        complexity_penalty = 0.0
        if total_models > 12:
            complexity_penalty = (total_models - 12) * 0.1

        cognitive_load_increase = max(
            0.0, base_load - overlap_reduction + complexity_penalty
        )

        return {
            "cognitive_load_increase": cognitive_load_increase,
            "total_unique_models": total_models,
            "overlapping_models": overlapping_models,
            "complexity_level": (
                "low"
                if cognitive_load_increase < 0.5
                else "medium" if cognitive_load_increase < 1.0 else "high"
            ),
        }

    def _assess_integration_difficulty(
        self, nway_a: Dict[str, Any], nway_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess how difficult it is to integrate these NWAYs"""

        # Different integration patterns
        integration_patterns = {
            "sequential": 0.2,  # Use one after the other
            "parallel": 0.4,  # Use simultaneously but independently
            "integrated": 0.7,  # Deep integration required
            "layered": 0.5,  # One provides foundation for the other
        }

        # Determine integration pattern needed
        pattern = self._determine_integration_pattern(nway_a, nway_b)
        base_difficulty = integration_patterns.get(pattern, 0.5)

        # Adjust based on compatibility
        models_a = set(nway_a.get("models_involved", []))
        models_b = set(nway_b.get("models_involved", []))

        overlap_ratio = (
            len(models_a & models_b) / len(models_a | models_b)
            if (models_a | models_b)
            else 0
        )

        # Higher overlap = easier integration
        overlap_bonus = overlap_ratio * 0.3

        final_difficulty = max(0.0, min(1.0, base_difficulty - overlap_bonus))

        return {
            "integration_difficulty": final_difficulty,
            "integration_pattern": pattern,
            "overlap_ratio": overlap_ratio,
            "difficulty_level": (
                "easy"
                if final_difficulty < 0.3
                else "medium" if final_difficulty < 0.7 else "hard"
            ),
        }

    def _determine_integration_pattern(
        self, nway_a: Dict[str, Any], nway_b: Dict[str, Any]
    ) -> str:
        """Determine how the NWAYs should be integrated"""

        # Lollapalooza compounds typically need sequential application
        if self._is_lollapalooza_compound(nway_a) or self._is_lollapalooza_compound(
            nway_b
        ):
            return "sequential"

        # Meta-frameworks can often be integrated deeply
        if self._is_meta_framework(nway_a) and self._is_meta_framework(nway_b):
            return "integrated"

        # Domain-specific tools often work in parallel
        if self._is_domain_specific(nway_a) and self._is_domain_specific(nway_b):
            return "parallel"

        # Meta + domain often creates layered approach
        if (self._is_meta_framework(nway_a) and self._is_domain_specific(nway_b)) or (
            self._is_meta_framework(nway_b) and self._is_domain_specific(nway_a)
        ):
            return "layered"

        return "parallel"  # Default

    def _synthesize_compatibility_result(
        self,
        id_a: str,
        id_b: str,
        model_analysis: Dict,
        synergy_analysis: Dict,
        conflict_analysis: Dict,
        emergence_analysis: Dict,
        load_analysis: Dict,
        integration_analysis: Dict,
    ) -> CompatibilityResult:
        """Synthesize all analyses into final compatibility result"""

        # Calculate reinforcement effect
        base_reinforcement = 0.0

        if synergy_analysis["synergy_found"]:
            base_reinforcement = synergy_analysis["synergy_data"][
                "reinforcement_effect"
            ]
        elif conflict_analysis["conflict_found"]:
            base_reinforcement = conflict_analysis["conflict_data"].get(
                "reinforcement_effect", -0.3
            )
        else:
            # Calculate from model interactions
            synergistic_pairs = len(model_analysis["synergistic_pairs"])
            conflicting_pairs = len(model_analysis["conflicting_pairs"])
            base_reinforcement = (synergistic_pairs * 0.3) - (conflicting_pairs * 0.2)

        # Adjust reinforcement based on emergence potential
        emergence_bonus = emergence_analysis["general_emergence_potential"] * 0.5
        final_reinforcement = min(2.0, max(-1.0, base_reinforcement + emergence_bonus))

        # Calculate conflict risk
        conflict_risk = 0.0
        if conflict_analysis["conflict_found"]:
            if conflict_analysis["conflict_data"]:
                conflict_risk = conflict_analysis["conflict_data"].get(
                    "conflict_risk", 0.5
                )
            elif conflict_analysis["type_conflicts"]["has_conflicts"]:
                conflict_risk = conflict_analysis["type_conflicts"]["severity"]

        conflict_risk += len(model_analysis["conflicting_pairs"]) * 0.1
        conflict_risk = min(1.0, conflict_risk)

        # Determine reaction type and compatibility category
        reaction_type = self._determine_reaction_type(
            final_reinforcement, conflict_risk
        )
        compatibility_category = self._determine_compatibility_category(
            final_reinforcement, conflict_risk
        )

        # Calculate stability (inverse of context dependency and conflict risk)
        stability = max(
            0.0,
            1.0
            - conflict_risk
            - (integration_analysis["integration_difficulty"] * 0.3),
        )

        # Context dependency
        conflict_data = conflict_analysis.get("conflict_data") or {}
        context_dependency = max(
            conflict_data.get("context_dependency", 0.3),
            integration_analysis["integration_difficulty"],
        )

        # Confidence level
        confidence_level = min(
            synergy_analysis["match_confidence"],
            conflict_analysis["match_confidence"],
            0.8,  # Base confidence for theoretical analysis
        )

        # Generate explanation
        explanation = self._generate_explanation(
            synergy_analysis,
            conflict_analysis,
            emergence_analysis,
            final_reinforcement,
            reaction_type,
        )

        return CompatibilityResult(
            nway_a=id_a,
            nway_b=id_b,
            reaction_type=reaction_type,
            compatibility_category=compatibility_category,
            reinforcement_effect=final_reinforcement,
            conflict_risk=conflict_risk,
            emergence_potential=emergence_analysis["general_emergence_potential"],
            stability=stability,
            cognitive_load_increase=load_analysis["cognitive_load_increase"],
            integration_difficulty=integration_analysis["integration_difficulty"],
            context_dependency=context_dependency,
            confidence_level=confidence_level,
            explanation=explanation,
            discovered_via="theoretical",
        )

    def _determine_reaction_type(
        self, reinforcement: float, conflict_risk: float
    ) -> ReactionType:
        """Determine the type of chemical reaction between NWAYs"""

        if reinforcement >= 1.5 and conflict_risk < 0.3:
            return ReactionType.SYNERGISTIC
        elif reinforcement >= 0.5 and conflict_risk < 0.5:
            return ReactionType.ADDITIVE
        elif reinforcement >= 0.0 and conflict_risk < 0.7:
            return ReactionType.NEUTRAL
        elif reinforcement >= -0.5:
            return ReactionType.CONFLICTING
        else:
            return ReactionType.DESTRUCTIVE

    def _determine_compatibility_category(
        self, reinforcement: float, conflict_risk: float
    ) -> CompatibilityCategory:
        """Determine the compatibility category"""

        if reinforcement >= 1.8 and conflict_risk < 0.2:
            return CompatibilityCategory.PERFECT_SYNERGY
        elif reinforcement >= 1.2 and conflict_risk < 0.4:
            return CompatibilityCategory.HIGH_COMPATIBILITY
        elif reinforcement >= 0.5 and conflict_risk < 0.6:
            return CompatibilityCategory.MILD_COMPATIBILITY
        elif reinforcement >= 0.0 and conflict_risk < 0.7:
            return CompatibilityCategory.NEUTRAL_COEXISTENCE
        elif reinforcement >= -0.3:
            return CompatibilityCategory.MINOR_CONFLICT
        elif reinforcement >= -0.7:
            return CompatibilityCategory.MAJOR_CONFLICT
        else:
            return CompatibilityCategory.DESTRUCTIVE_INTERFERENCE

    def _generate_explanation(
        self,
        synergy_analysis: Dict,
        conflict_analysis: Dict,
        emergence_analysis: Dict,
        reinforcement: float,
        reaction_type: ReactionType,
    ) -> str:
        """Generate human-readable explanation of the compatibility"""

        if synergy_analysis["synergy_found"]:
            return synergy_analysis["synergy_data"]["explanation"]
        elif conflict_analysis["conflict_found"] and conflict_analysis["conflict_data"]:
            return conflict_analysis["conflict_data"]["explanation"]
        elif emergence_analysis["known_emergence"]:
            return emergence_analysis["known_emergence"]["description"]
        else:
            if reaction_type == ReactionType.SYNERGISTIC:
                return "These frameworks create synergistic amplification through complementary mental models"
            elif reaction_type == ReactionType.ADDITIVE:
                return (
                    "These frameworks work well together with modest mutual enhancement"
                )
            elif reaction_type == ReactionType.NEUTRAL:
                return "These frameworks can coexist without significant interaction"
            elif reaction_type == ReactionType.CONFLICTING:
                return (
                    "These frameworks have some conflicting elements that may interfere"
                )
            else:
                return "These frameworks have significant conflicts that may cancel each other out"

    # Helper methods for type checking
    def _is_lollapalooza_compound(self, nway: Dict[str, Any]) -> bool:
        """Check if NWAY is a Lollapalooza compound"""
        return any(
            term in nway.get("interaction_id", "").upper()
            for term in ["AUCTION", "TUPPERWARE", "COCACOLA"]
        )

    def _is_meta_framework(self, nway: Dict[str, Any]) -> bool:
        """Check if NWAY is a meta-framework"""
        return any(
            term in nway.get("interaction_id", "").upper()
            for term in ["DECISION", "BIAS", "UNCERTAINTY"]
        )

    def _is_domain_specific(self, nway: Dict[str, Any]) -> bool:
        """Check if NWAY is domain-specific"""
        return any(
            term in nway.get("interaction_id", "").upper()
            for term in ["CREATIVITY", "DIAGNOSTIC", "LEARNING", "OUTLIER"]
        )


# ======================================================================
# BATCH COMPATIBILITY ANALYSIS
# ======================================================================


def analyze_nway_combination_compatibility(
    nway_list: List[Dict[str, Any]],
) -> Dict[str, CompatibilityResult]:
    """
    Analyze compatibility for all pairs in a list of NWAYs

    This is used when evaluating a complete cognitive chemistry combination
    """

    matrix = SynergisticCompatibilityMatrix()
    results = {}

    logger.info(
        f"🧪 Analyzing compatibility for {len(nway_list)} NWAYs ({len(nway_list)*(len(nway_list)-1)//2} pairs)"
    )

    for i, nway_a in enumerate(nway_list):
        for j, nway_b in enumerate(nway_list[i + 1 :], i + 1):
            result = matrix.calculate_compatibility(nway_a, nway_b)

            pair_key = f"{nway_a.get('interaction_id', f'nway_{i}')} + {nway_b.get('interaction_id', f'nway_{j}')}"
            results[pair_key] = result

    return results


# ======================================================================
# FACTORY FUNCTION
# ======================================================================


def get_synergistic_compatibility_matrix() -> SynergisticCompatibilityMatrix:
    """Get the Synergistic Compatibility Matrix instance"""
    return SynergisticCompatibilityMatrix()


if __name__ == "__main__":
    print("🧪 SYNERGISTIC COMPATIBILITY MATRIX - Phase 3")
    print("   Revolutionary cognitive chemistry reaction prediction")
    print("   From scoring components to scoring REACTIONS")

    # Test the compatibility matrix
    matrix = get_synergistic_compatibility_matrix()

    # Test perfect synergy
    analyst_cluster = {
        "interaction_id": "NWAY_ANALYST_CLUSTER_007",
        "models_involved": [
            "critical-thinking",
            "root-cause-analysis",
            "evidence-based-reasoning",
            "logic-models",
        ],
    }

    bias_mitigation = {
        "interaction_id": "NWAY_BIAS_MITIGATION_019",
        "models_involved": [
            "cognitive-biases",
            "intellectual-humility",
            "critical-thinking",
            "outside-view",
        ],
    }

    result = matrix.calculate_compatibility(analyst_cluster, bias_mitigation)

    print("\n🎯 Test Compatibility Result:")
    print(f"   Reaction Type: {result.reaction_type.value}")
    print(f"   Compatibility: {result.compatibility_category.value}")
    print(f"   Reinforcement: {result.reinforcement_effect:.2f}")
    print(f"   Conflict Risk: {result.conflict_risk:.2f}")
    print(f"   Emergence: {result.emergence_potential:.2f}")
    print(f"   Explanation: {result.explanation}")
