#!/usr/bin/env python3
"""
REACTION PROBABILITY ENGINE
Phase 5 of Operation: Cognitive Particle Accelerator

REVOLUTIONARY REPLACEMENT FOR STATION 3 SELECTION LOGIC

This completely transforms METIS from a simple consultant mapper to a true
GENERATIVE INTELLIGENCE ENGINE that assembles bespoke cognitive capabilities
by predicting and orchestrating cognitive chemistry reactions.

The old system: Pick consultants based on keyword matching
The new system: Engineer cognitive reactions based on chemistry analysis

This is the final step in Operation: Cognitive Particle Accelerator
"""

import asyncio
import os
from supabase import create_client
from dotenv import load_dotenv
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

# Import our revolutionary engines
try:
    from ...services.selection.cognitive_chemistry_engine import (
        CognitiveChemistryEngine,
        CognitiveChemistryReaction,
        ReactionQuality,
        get_cognitive_chemistry_engine,
    )
except ImportError:
    # Fallback for direct execution
    import sys

    sys.path.append("../../services/selection")
    from cognitive_chemistry_engine import (
        CognitiveChemistryReaction,
        ReactionQuality,
        get_cognitive_chemistry_engine,
    )

load_dotenv()
logger = logging.getLogger(__name__)

# ======================================================================
# COGNITIVE RECIPE CARDS
# ======================================================================


@dataclass
class CognitiveRecipeCard:
    """
    A complete recipe for assembling bespoke cognitive capabilities

    This replaces simple consultant selection with sophisticated
    cognitive chemistry orchestration.
    """

    recipe_id: str
    problem_framework: str

    # The cognitive chemistry reaction
    primary_reaction: CognitiveChemistryReaction
    alternative_reactions: List[CognitiveChemistryReaction]

    # Orchestration instructions
    integration_pattern: str  # "sequential", "parallel", "layered", "integrated"
    execution_sequence: List[str]  # Order of NWAY activation
    cognitive_load_management: str

    # Consultant assignments
    selected_consultants: List[str]
    consultant_roles: Dict[str, str]  # consultant_id -> role

    # Quality assurance
    overall_confidence: float
    risk_mitigation_strategies: List[str]
    success_metrics: List[str]

    # Metadata
    recipe_quality: ReactionQuality
    created_at: datetime
    estimated_duration: str


@dataclass
class ConsultantAssignment:
    """Enhanced consultant assignment with chemistry-driven roles"""

    consultant_id: str
    consultant_name: str
    specialization: str

    # Chemistry-driven assignments
    assigned_nway_interactions: List[str]
    cognitive_role: str  # "primary_catalyst", "synergy_amplifier", "stability_anchor"
    integration_responsibility: str

    # Performance predictions
    effectiveness_prediction: float
    confidence_level: float


# ======================================================================
# THE REACTION PROBABILITY ENGINE
# ======================================================================


class ReactionProbabilityEngine:
    """
    REVOLUTIONARY REPLACEMENT FOR STATION 3

    This is the culmination of Operation: Cognitive Particle Accelerator.
    Instead of simple consultant mapping, we now engineer bespoke cognitive
    capabilities through chemistry analysis.

    The transformation:
    OLD: "Which consultants know about this topic?"
    NEW: "Which cognitive chemistry reactions will solve this problem most effectively?"
    """

    def __init__(self, context_stream=None):
        """Initialize the revolutionary selection engine"""

        # Initialize Supabase connection
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_ANON_KEY")
        self.supabase = create_client(url, key)

        # Initialize the cognitive chemistry engine with context stream for glass-box evidence
        self.chemistry_engine = get_cognitive_chemistry_engine(
            context_stream=context_stream
        )

        # Cache for NWAY interactions and consultants
        self._nway_interactions = None
        self._consultant_profiles = None

        # Selection strategy parameters
        self.min_reactions_to_evaluate = 3
        self.max_reactions_to_evaluate = 8
        self.chemistry_score_threshold = 0.6
        self.stability_threshold = 0.6

        logger.info(
            "🚀 Reaction Probability Engine initialized - COGNITIVE PARTICLE ACCELERATOR READY"
        )

    async def generate_cognitive_recipe(
        self, problem_framework: str, context: Optional[Dict[str, Any]] = None
    ) -> CognitiveRecipeCard:
        """
        THE REVOLUTIONARY TRANSFORMATION

        Generate a complete cognitive recipe for solving the problem.
        This replaces simple consultant selection with cognitive chemistry engineering.

        Args:
            problem_framework: The problem description and context
            context: Additional context (user preferences, constraints, etc.)

        Returns:
            Complete cognitive recipe with chemistry-optimized solution
        """

        recipe_id = f"cognitive_recipe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"🧬 GENERATING COGNITIVE RECIPE: {recipe_id}")
        logger.info(f"   Problem: {problem_framework[:100]}...")

        # STEP 1: Load available NWAY interactions
        await self._load_nway_interactions()
        await self._load_consultant_profiles()

        # STEP 2: Generate candidate cognitive reactions
        candidate_reactions = await self._generate_candidate_reactions(
            problem_framework, context
        )

        # STEP 3: Evaluate all candidate reactions using chemistry engine
        evaluated_reactions = await self._evaluate_cognitive_reactions(
            problem_framework, candidate_reactions
        )

        # STEP 4: Select optimal primary reaction and alternatives
        primary_reaction, alternative_reactions = self._select_optimal_reactions(
            evaluated_reactions
        )

        # STEP 5: Determine integration pattern
        integration_pattern = self._determine_integration_pattern(primary_reaction)

        # STEP 6: Generate execution sequence
        execution_sequence = self._generate_execution_sequence(
            primary_reaction, integration_pattern
        )

        # STEP 7: Assign consultants based on chemistry analysis
        consultant_assignments = await self._assign_consultants_by_chemistry(
            primary_reaction
        )

        # STEP 8: Generate cognitive load management strategy
        load_management = self._generate_load_management_strategy(primary_reaction)

        # STEP 9: Create quality assurance plan
        qa_plan = self._create_quality_assurance_plan(
            primary_reaction, alternative_reactions
        )

        # STEP 10: Assemble final cognitive recipe
        recipe = CognitiveRecipeCard(
            recipe_id=recipe_id,
            problem_framework=problem_framework,
            primary_reaction=primary_reaction,
            alternative_reactions=alternative_reactions,
            integration_pattern=integration_pattern,
            execution_sequence=execution_sequence,
            cognitive_load_management=load_management,
            selected_consultants=[
                assignment.consultant_id for assignment in consultant_assignments
            ],
            consultant_roles={
                assignment.consultant_id: assignment.cognitive_role
                for assignment in consultant_assignments
            },
            overall_confidence=primary_reaction.confidence_level,
            risk_mitigation_strategies=qa_plan["risk_mitigation"],
            success_metrics=qa_plan["success_metrics"],
            recipe_quality=self._determine_recipe_quality(primary_reaction),
            created_at=datetime.now(),
            estimated_duration=primary_reaction.predicted_execution_time,
        )

        logger.info("✅ COGNITIVE RECIPE GENERATED")
        logger.info(f"   Quality: {recipe.recipe_quality.value}")
        logger.info(
            f"   Primary Chemistry Score: {primary_reaction.overall_chemistry_score:.3f}"
        )
        logger.info(f"   Consultants: {len(consultant_assignments)}")
        logger.info(f"   Duration: {recipe.estimated_duration}")

        return recipe

    async def _load_nway_interactions(self):
        """Load all available NWAY interactions from database"""
        if self._nway_interactions is not None:
            return

        logger.info("📚 Loading NWAY interactions from database")

        result = self.supabase.table("nway_interactions").select("*").execute()
        self._nway_interactions = result.data or []

        logger.info(f"   Loaded {len(self._nway_interactions)} NWAY interactions")

    async def _load_consultant_profiles(self):
        """Load consultant profiles from database"""
        if self._consultant_profiles is not None:
            return

        logger.info("👥 Loading consultant profiles from database")

        # For now, use the static profiles from the old system
        # In the future, this could be loaded from a consultants table
        self._consultant_profiles = {
            "NWAY_STRATEGIST_CLUSTER_009": {
                "consultant_id": "strategist_001",
                "name": "Thomas Anderson - Strategy Director",
                "specialization": "Corporate Strategy & Competitive Intelligence",
                "expertise": [
                    "Corporate strategy",
                    "Competitive analysis",
                    "Market positioning",
                    "Strategic planning",
                ],
                "context": "Corporate Strategy",
            },
            "NWAY_ANALYST_CLUSTER_007": {
                "consultant_id": "analyst_001",
                "name": "Dr. Sarah Kim - Chief Analyst",
                "specialization": "Deep Analysis & Critical Thinking",
                "expertise": [
                    "Root cause analysis",
                    "Evidence-based reasoning",
                    "Critical thinking",
                    "Logic models",
                ],
                "context": "Strategic Analysis",
            },
            "NWAY_BIAS_MITIGATION_019": {
                "consultant_id": "bias_expert_001",
                "name": "Dr. Carlos Mendez - Decision Quality Expert",
                "specialization": "Decision Governance & Bias Mitigation",
                "expertise": [
                    "Decision governance",
                    "Cognitive bias detection",
                    "Process improvement",
                    "Quality assurance",
                ],
                "context": "Decision Excellence & Governance",
            },
            "NWAY_RESEARCHER_CLUSTER_016": {
                "consultant_id": "researcher_001",
                "name": "Dr. Samantha Lee - Research Director",
                "specialization": "Market Research & Consumer Insights",
                "expertise": [
                    "Market research",
                    "Consumer behavior",
                    "Insights generation",
                    "Research methodology",
                ],
                "context": "Market Research & Insights",
            },
            "NWAY_CREATIVITY_003": {
                "consultant_id": "creative_001",
                "name": "Alex Rivera - Innovation Catalyst",
                "specialization": "Creative Problem Solving & Innovation",
                "expertise": [
                    "Creative thinking",
                    "Innovation processes",
                    "Design thinking",
                    "Ideation facilitation",
                ],
                "context": "Innovation & Creativity",
            },
        }

        logger.info(f"   Loaded {len(self._consultant_profiles)} consultant profiles")

    async def _generate_candidate_reactions(
        self, problem_framework: str, context: Optional[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Generate candidate cognitive reaction combinations

        This is where the magic happens - we create different combinations
        of NWAY interactions that could solve the problem.
        """

        logger.info("🧪 Generating candidate cognitive reactions")

        # Filter relevant NWAY interactions
        relevant_nways = self._filter_relevant_nways(problem_framework)

        # Generate different combination strategies
        candidates = []

        # Strategy 1: Single powerful NWAY (Lollapalooza compounds)
        lollapalooza_candidates = self._generate_lollapalooza_candidates(relevant_nways)
        candidates.extend(lollapalooza_candidates)

        # Strategy 2: Meta-framework + supporting tools
        meta_framework_candidates = self._generate_meta_framework_candidates(
            relevant_nways
        )
        candidates.extend(meta_framework_candidates)

        # Strategy 3: Cognitive cluster combinations
        cluster_candidates = self._generate_cluster_candidates(relevant_nways)
        candidates.extend(cluster_candidates)

        # Strategy 4: Domain-specific toolkit combinations
        toolkit_candidates = self._generate_toolkit_candidates(relevant_nways)
        candidates.extend(toolkit_candidates)

        # Strategy 5: Balanced multi-tier combinations
        balanced_candidates = self._generate_balanced_candidates(relevant_nways)
        candidates.extend(balanced_candidates)

        # Limit to manageable number
        final_candidates = candidates[: self.max_reactions_to_evaluate]

        logger.info(f"   Generated {len(final_candidates)} candidate reactions")
        return final_candidates

    def _filter_relevant_nways(self, problem_framework: str) -> List[Dict[str, Any]]:
        """Filter NWAY interactions that are relevant to the problem"""

        # Simple relevance filtering based on keywords and descriptions
        # In a more sophisticated version, this could use semantic similarity

        text_lower = problem_framework.lower()
        keywords = set(text_lower.split())

        relevant_nways = []

        for nway in self._nway_interactions:
            relevance_score = 0.0

            # Check interaction ID for relevance
            interaction_id = nway.get("interaction_id", "").lower()
            if any(
                keyword in interaction_id
                for keyword in ["strategy", "analysis", "decision", "creative", "bias"]
            ):
                relevance_score += 0.3

            # Check models involved
            models_involved = nway.get("models_involved", [])
            for model in models_involved:
                if any(keyword in model.lower() for keyword in keywords):
                    relevance_score += 0.1

            # Check description/summary
            summary = nway.get("emergent_effect_summary", "").lower()
            if any(keyword in summary for keyword in keywords):
                relevance_score += 0.2

            if relevance_score > 0.1:  # Threshold for relevance
                nway["relevance_score"] = relevance_score
                relevant_nways.append(nway)

        # Sort by relevance and return top candidates
        relevant_nways.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return relevant_nways[:15]  # Top 15 most relevant

    def _generate_lollapalooza_candidates(
        self, relevant_nways: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Generate candidates featuring Lollapalooza compounds"""
        candidates = []

        lollapalooza_ids = ["AUCTION_001", "TUPPERWARE_002", "COCACOLA_006"]

        for nway in relevant_nways:
            interaction_id = nway.get("interaction_id", "")
            if any(lolla_id in interaction_id for lolla_id in lollapalooza_ids):
                # Single Lollapalooza compound
                candidates.append([nway])

        return candidates

    def _generate_meta_framework_candidates(
        self, relevant_nways: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Generate candidates featuring meta-frameworks with supporting tools"""
        candidates = []

        meta_framework_ids = [
            "DECISION_TRILEMMA",
            "BIAS_MITIGATION",
            "UNCERTAINTY_DECISION",
        ]

        meta_frameworks = [
            nway
            for nway in relevant_nways
            if any(
                meta_id in nway.get("interaction_id", "")
                for meta_id in meta_framework_ids
            )
        ]

        supporting_tools = [
            nway
            for nway in relevant_nways
            if not any(
                meta_id in nway.get("interaction_id", "")
                for meta_id in meta_framework_ids
            )
        ]

        for meta_framework in meta_frameworks:
            # Meta-framework alone
            candidates.append([meta_framework])

            # Meta-framework + 1-2 supporting tools
            for tool in supporting_tools[:3]:
                candidates.append([meta_framework, tool])

        return candidates

    def _generate_cluster_candidates(
        self, relevant_nways: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Generate candidates featuring cognitive clusters"""
        candidates = []

        cluster_ids = ["CLUSTER"]

        clusters = [
            nway
            for nway in relevant_nways
            if any(
                cluster_id in nway.get("interaction_id", "")
                for cluster_id in cluster_ids
            )
        ]

        for cluster in clusters:
            # Single cluster
            candidates.append([cluster])

        # Combinations of 2 clusters
        for i, cluster1 in enumerate(clusters):
            for cluster2 in clusters[i + 1 :]:
                candidates.append([cluster1, cluster2])

        return candidates

    def _generate_toolkit_candidates(
        self, relevant_nways: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Generate candidates featuring domain-specific toolkits"""
        candidates = []

        toolkit_ids = ["CREATIVITY", "DIAGNOSTIC", "LEARNING", "OUTLIER"]

        toolkits = [
            nway
            for nway in relevant_nways
            if any(
                toolkit_id in nway.get("interaction_id", "")
                for toolkit_id in toolkit_ids
            )
        ]

        # Single toolkits
        for toolkit in toolkits:
            candidates.append([toolkit])

        # Combinations of 2-3 toolkits
        for i, toolkit1 in enumerate(toolkits):
            for toolkit2 in toolkits[i + 1 :]:
                candidates.append([toolkit1, toolkit2])

        return candidates

    def _generate_balanced_candidates(
        self, relevant_nways: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Generate balanced combinations across different tiers"""
        candidates = []

        # Try to create combinations that span multiple tiers
        # This is a simplified version - could be much more sophisticated

        if len(relevant_nways) >= 3:
            # Top 3 most relevant
            candidates.append(relevant_nways[:3])

        if len(relevant_nways) >= 4:
            # Mix of high and medium relevance
            candidates.append([relevant_nways[0], relevant_nways[2], relevant_nways[3]])

        return candidates

    async def _evaluate_cognitive_reactions(
        self, problem_framework: str, candidate_reactions: List[List[Dict[str, Any]]]
    ) -> List[CognitiveChemistryReaction]:
        """
        Evaluate all candidate reactions using the chemistry engine

        This is where the revolutionary analysis happens
        """

        logger.info(f"⚡ Evaluating {len(candidate_reactions)} cognitive reactions")

        evaluated_reactions = []

        for i, candidate in enumerate(candidate_reactions):
            logger.info(
                f"   Evaluating reaction {i+1}/{len(candidate_reactions)}: {[nway.get('interaction_id') for nway in candidate]}"
            )

            try:
                reaction = self.chemistry_engine.calculate_cognitive_chemistry_score(
                    problem_framework, candidate
                )
                evaluated_reactions.append(reaction)

            except Exception as e:
                logger.error(f"   Failed to evaluate reaction {i+1}: {e}")
                continue

        # Sort by overall chemistry score
        evaluated_reactions.sort(key=lambda r: r.overall_chemistry_score, reverse=True)

        logger.info(f"   Successfully evaluated {len(evaluated_reactions)} reactions")
        if evaluated_reactions:
            logger.info(
                f"   Best score: {evaluated_reactions[0].overall_chemistry_score:.3f}"
            )

        return evaluated_reactions

    def _select_optimal_reactions(
        self, evaluated_reactions: List[CognitiveChemistryReaction]
    ) -> Tuple[CognitiveChemistryReaction, List[CognitiveChemistryReaction]]:
        """Select the best primary reaction and good alternatives"""

        if not evaluated_reactions:
            raise Exception("No viable cognitive reactions found")

        # Primary reaction is the highest scoring
        primary_reaction = evaluated_reactions[0]

        # Alternatives are other high-quality reactions
        alternatives = []
        for reaction in evaluated_reactions[1:]:
            if (
                reaction.overall_chemistry_score >= self.chemistry_score_threshold
                and reaction.stability_rating >= self.stability_threshold
                and len(alternatives) < 3
            ):
                alternatives.append(reaction)

        logger.info(
            f"🎯 Selected primary reaction: {primary_reaction.nway_combination}"
        )
        logger.info(f"   Score: {primary_reaction.overall_chemistry_score:.3f}")
        logger.info(f"   Alternatives: {len(alternatives)}")

        return primary_reaction, alternatives

    def _determine_integration_pattern(
        self, reaction: CognitiveChemistryReaction
    ) -> str:
        """Determine how the NWAYs should be integrated"""

        # Analyze the compatibility results to determine integration pattern
        if not reaction.compatibility_results:
            return "sequential"

        # Look at dominant reaction types in compatibility
        reaction_types = []
        for comp_result in reaction.compatibility_results.values():
            reaction_types.append(comp_result.reaction_type.value)

        # Simple heuristics for integration pattern
        if "synergistic" in reaction_types:
            return "integrated"  # Deep integration for synergistic reactions
        elif "additive" in reaction_types:
            return "parallel"  # Parallel for additive reactions
        elif "conflicting" in reaction_types:
            return "sequential"  # Sequential to manage conflicts
        else:
            return "layered"  # Layered as default

    def _generate_execution_sequence(
        self, reaction: CognitiveChemistryReaction, pattern: str
    ) -> List[str]:
        """Generate the optimal execution sequence for NWAYs"""

        nway_ids = reaction.nway_combination

        if pattern == "sequential":
            # Execute in order of stability (most stable first)
            return nway_ids  # Simplified - could analyze individual stability scores
        elif pattern == "parallel":
            # All can execute simultaneously
            return nway_ids
        elif pattern == "layered":
            # Foundation first, then building layers
            return nway_ids  # Simplified - could analyze foundational depth
        else:  # integrated
            # Careful orchestration needed
            return nway_ids

    async def _assign_consultants_by_chemistry(
        self, reaction: CognitiveChemistryReaction
    ) -> List[ConsultantAssignment]:
        """Assign consultants based on chemistry analysis rather than simple mapping"""

        assignments = []

        for nway_id in reaction.nway_combination:
            if nway_id in self._consultant_profiles:
                profile = self._consultant_profiles[nway_id]

                # Determine cognitive role based on chemistry analysis
                cognitive_role = self._determine_cognitive_role(nway_id, reaction)

                assignment = ConsultantAssignment(
                    consultant_id=profile["consultant_id"],
                    consultant_name=profile["name"],
                    specialization=profile["specialization"],
                    assigned_nway_interactions=[nway_id],
                    cognitive_role=cognitive_role,
                    integration_responsibility=self._determine_integration_responsibility(
                        nway_id, reaction
                    ),
                    effectiveness_prediction=reaction.predicted_effectiveness,
                    confidence_level=reaction.confidence_level,
                )

                assignments.append(assignment)

        return assignments

    def _determine_cognitive_role(
        self, nway_id: str, reaction: CognitiveChemistryReaction
    ) -> str:
        """Determine the cognitive role for this NWAY in the reaction"""

        if reaction.primary_nway_type == "lollapalooza" and any(
            lolla in nway_id for lolla in ["AUCTION", "TUPPERWARE", "COCACOLA"]
        ):
            return "primary_catalyst"
        elif "CLUSTER" in nway_id:
            return "expertise_anchor"
        elif "BIAS" in nway_id or "DECISION" in nway_id:
            return "quality_guardian"
        else:
            return "synergy_amplifier"

    def _determine_integration_responsibility(
        self, nway_id: str, reaction: CognitiveChemistryReaction
    ) -> str:
        """Determine integration responsibilities"""

        if "STRATEGIST" in nway_id:
            return "overall_orchestration"
        elif "ANALYST" in nway_id:
            return "analysis_coordination"
        elif "BIAS" in nway_id:
            return "quality_assurance"
        else:
            return "specialized_execution"

    def _generate_load_management_strategy(
        self, reaction: CognitiveChemistryReaction
    ) -> str:
        """Generate cognitive load management strategy"""

        if reaction.cognitive_load_assessment.startswith("Very High"):
            return "Phased execution with recovery periods between NWAYs"
        elif reaction.cognitive_load_assessment.startswith("High"):
            return "Structured breaks and cognitive offloading techniques"
        elif reaction.cognitive_load_assessment.startswith("Medium"):
            return "Standard execution with attention to mental fatigue"
        else:
            return "Standard execution - minimal load management needed"

    def _create_quality_assurance_plan(
        self,
        primary_reaction: CognitiveChemistryReaction,
        alternatives: List[CognitiveChemistryReaction],
    ) -> Dict[str, List[str]]:
        """Create quality assurance and risk mitigation plan"""

        risk_mitigation = []
        success_metrics = []

        # Risk mitigation based on reaction analysis
        for risk_factor in primary_reaction.risk_factors:
            if "stability" in risk_factor.lower():
                risk_mitigation.append("Implement checkpoints to monitor consistency")
            elif "efficiency" in risk_factor.lower():
                risk_mitigation.append("Monitor cognitive load and adjust pacing")
            elif "conflict" in risk_factor.lower():
                risk_mitigation.append("Use sequential execution to manage conflicts")
            elif "probability" in risk_factor.lower():
                risk_mitigation.append("Have backup alternatives ready")

        # Success metrics based on predictions
        success_metrics.append(
            f"Achieve predicted effectiveness of {primary_reaction.predicted_effectiveness:.1%}"
        )
        success_metrics.append(
            f"Complete within estimated {primary_reaction.predicted_execution_time}"
        )
        success_metrics.append(
            f"Maintain stability rating above {primary_reaction.stability_rating:.2f}"
        )

        if alternatives:
            risk_mitigation.append(
                f"Activate alternative reaction if primary falls below {self.chemistry_score_threshold}"
            )

        return {"risk_mitigation": risk_mitigation, "success_metrics": success_metrics}

    def _determine_recipe_quality(
        self, reaction: CognitiveChemistryReaction
    ) -> ReactionQuality:
        """Determine overall recipe quality"""

        score = reaction.overall_chemistry_score

        if score >= 0.9:
            return ReactionQuality.REVOLUTIONARY
        elif score >= 0.8:
            return ReactionQuality.EXCELLENT
        elif score >= 0.7:
            return ReactionQuality.GOOD
        elif score >= 0.6:
            return ReactionQuality.ACCEPTABLE
        elif score >= 0.4:
            return ReactionQuality.POOR
        else:
            return ReactionQuality.HARMFUL


# ======================================================================
# COMPATIBILITY LAYER FOR EXISTING SYSTEM
# ======================================================================


class ReactiveNWayConsultantSelector:
    """
    Compatibility layer that provides the same interface as the old system
    but uses the revolutionary Reaction Probability Engine under the hood.

    This allows seamless integration without breaking existing code.
    """

    def __init__(self, context_stream=None):
        self.reaction_engine = ReactionProbabilityEngine(context_stream=context_stream)
        logger.info(
            "🔄 Reactive NWAY Selector - Compatibility layer for Reaction Probability Engine"
        )

    async def select_consultants(
        self, query: str, max_consultants: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Legacy interface that now uses cognitive chemistry under the hood

        This maintains backward compatibility while providing revolutionary capabilities
        """

        # Generate cognitive recipe using the new engine
        recipe = await self.reaction_engine.generate_cognitive_recipe(query)

        # Convert to legacy format
        legacy_consultants = []

        for consultant_id in recipe.selected_consultants:
            # Find consultant profile
            consultant_profile = None
            for nway_id, profile in self.reaction_engine._consultant_profiles.items():
                if profile["consultant_id"] == consultant_id:
                    consultant_profile = profile
                    break

            if consultant_profile:
                legacy_consultant = {
                    "consultant_id": consultant_id,
                    "consultant_name": consultant_profile["name"],
                    "specialization": consultant_profile["specialization"],
                    "expertise_areas": consultant_profile["expertise"],
                    "business_context": consultant_profile["context"],
                    "relevance_score": recipe.primary_reaction.overall_chemistry_score,
                    "chemistry_analysis": {
                        "overall_score": recipe.primary_reaction.overall_chemistry_score,
                        "amplification_potential": recipe.primary_reaction.amplification_potential,
                        "cognitive_efficiency": recipe.primary_reaction.cognitive_efficiency,
                        "stability_rating": recipe.primary_reaction.stability_rating,
                        "recommendation": recipe.primary_reaction.recommendation,
                    },
                }
                legacy_consultants.append(legacy_consultant)

        # Limit to requested number
        return legacy_consultants[:max_consultants]


# ======================================================================
# FACTORY FUNCTIONS
# ======================================================================


def get_reaction_probability_engine(context_stream=None) -> ReactionProbabilityEngine:
    """Get the Reaction Probability Engine instance"""
    return ReactionProbabilityEngine(context_stream=context_stream)


def get_reactive_nway_selector(context_stream=None) -> ReactiveNWayConsultantSelector:
    """Get the compatibility layer for existing systems"""
    return ReactiveNWayConsultantSelector(context_stream=context_stream)


if __name__ == "__main__":
    print("🚀 REACTION PROBABILITY ENGINE - Phase 5")
    print("   REVOLUTIONARY REPLACEMENT FOR STATION 3")
    print("   From consultant mapping to cognitive chemistry engineering")

    async def test_reaction_engine():
        engine = get_reaction_probability_engine()

        test_problem = "We need to develop a comprehensive market entry strategy for a new product while ensuring we avoid common strategic biases and maintain rigorous analytical standards"

        recipe = await engine.generate_cognitive_recipe(test_problem)

        print("\n🎯 COGNITIVE RECIPE GENERATED:")
        print(f"   Recipe ID: {recipe.recipe_id}")
        print(f"   Quality: {recipe.recipe_quality.value}")
        print(
            f"   Primary Score: {recipe.primary_reaction.overall_chemistry_score:.3f}"
        )
        print(f"   Integration: {recipe.integration_pattern}")
        print(f"   Consultants: {len(recipe.selected_consultants)}")
        print(f"   Duration: {recipe.estimated_duration}")
        print(f"   Confidence: {recipe.overall_confidence:.3f}")
        print(f"   Load Management: {recipe.cognitive_load_management}")

        print("\n✅ OPERATION: COGNITIVE PARTICLE ACCELERATOR - COMPLETE!")
        print("   METIS has been transformed into a GENERATIVE INTELLIGENCE ENGINE")

    # Run the test
    asyncio.run(test_reaction_engine())
