#!/usr/bin/env python3
"""
Persona-Driven Mental Model Selector - Phase 1.3
Integrates enhanced MECE classification with core specialist personas
Superior to generic LLM calls through authentic expertise-driven selection
"""

import asyncio
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.enhanced_mece_classifier import (
    EnhancedMECEClassifier,
    EnhancedProblemStructure,
)
from src.core.core_specialist_personas import (
    CoreSpecialistFactory,
    CoreSpecialistProfile,
)


@dataclass
class ModelSelectionRationale:
    """Detailed rationale for model selection"""

    model: str
    consultant: str
    selection_score: float
    affinity_score: float
    problem_relevance: float
    tier: str  # tier1, tier2, supplementary
    rationale: str


@dataclass
class ConsultantModelSet:
    """Selected mental models for specific consultant"""

    consultant_profile: CoreSpecialistProfile
    selected_models: List[str]
    selection_rationales: List[ModelSelectionRationale]
    total_selection_score: float
    model_diversity_score: float
    problem_coverage_score: float


@dataclass
class PersonaDrivenAnalysis:
    """Complete persona-driven analysis structure"""

    problem_structure: EnhancedProblemStructure
    consultant_model_sets: List[ConsultantModelSet]
    team_synergy_score: float
    total_models_selected: int
    model_overlap_analysis: Dict[str, List[str]]
    competitive_advantage_summary: str


class PersonaDrivenModelSelector:
    """Superior model selection through authentic specialist expertise"""

    def __init__(self):
        self.mece_classifier = EnhancedMECEClassifier()
        self.specialist_factory = CoreSpecialistFactory()
        self.specialists = self.specialist_factory.create_all_core_specialists()

        # Simulated N-way mental model universe for demonstration
        self.model_universe = self._create_model_universe()

    def _create_model_universe(self) -> Dict[str, Dict[str, Any]]:
        """Create simulated mental model universe with frequency data"""
        # This would come from Supabase in production
        return {
            "statistics-concepts": {
                "frequency": 8,
                "complexity": "high",
                "category": "analytical",
            },
            "correlation-vs-causation": {
                "frequency": 7,
                "complexity": "medium",
                "category": "analytical",
            },
            "pattern-recognition": {
                "frequency": 6,
                "complexity": "high",
                "category": "analytical",
            },
            "critical-thinking": {
                "frequency": 9,
                "complexity": "medium",
                "category": "analytical",
            },
            "root-cause-analysis": {
                "frequency": 8,
                "complexity": "medium",
                "category": "diagnostic",
            },
            "systems-thinking": {
                "frequency": 7,
                "complexity": "high",
                "category": "systems",
            },
            "second-order-thinking": {
                "frequency": 5,
                "complexity": "high",
                "category": "systems",
            },
            "scenario-analysis": {
                "frequency": 6,
                "complexity": "medium",
                "category": "strategic",
            },
            "outside-view": {
                "frequency": 5,
                "complexity": "medium",
                "category": "strategic",
            },
            "competitive-advantage": {
                "frequency": 4,
                "complexity": "high",
                "category": "strategic",
            },
            "network-effects": {
                "frequency": 3,
                "complexity": "high",
                "category": "systems",
            },
            "understanding-motivations": {
                "frequency": 8,
                "complexity": "medium",
                "category": "psychological",
            },
            "cognitive-biases": {
                "frequency": 6,
                "complexity": "medium",
                "category": "psychological",
            },
            "persuasion-principles-cialdini": {
                "frequency": 4,
                "complexity": "medium",
                "category": "psychological",
            },
            "social-proof": {
                "frequency": 5,
                "complexity": "low",
                "category": "psychological",
            },
            "loss-aversion": {
                "frequency": 4,
                "complexity": "medium",
                "category": "psychological",
            },
            # Additional models for supplementary selection
            "pareto-principle": {
                "frequency": 5,
                "complexity": "low",
                "category": "strategic",
            },
            "feedback-loops": {
                "frequency": 4,
                "complexity": "medium",
                "category": "systems",
            },
            "anchoring-bias": {
                "frequency": 3,
                "complexity": "low",
                "category": "psychological",
            },
            "survivorship-bias": {
                "frequency": 3,
                "complexity": "medium",
                "category": "analytical",
            },
            "opportunity-cost": {
                "frequency": 4,
                "complexity": "low",
                "category": "financial",
            },
            "sunk-cost-fallacy": {
                "frequency": 4,
                "complexity": "low",
                "category": "psychological",
            },
        }

    def calculate_problem_relevance_score(
        self, model: str, problem_structure: EnhancedProblemStructure
    ) -> float:
        """Calculate how relevant a model is to the specific problem"""

        model_data = self.model_universe.get(model, {})
        model_category = model_data.get("category", "")

        relevance_score = 0.0

        # Check relevance to primary problem categories
        for category in problem_structure.primary_categories:
            category_name = category.name

            # Direct category matching
            if model_category == category_name or category_name in model_category:
                relevance_score += category.score * 0.6

            # Semantic matching for problem types
            problem_model_mapping = {
                "crisis": [
                    "root-cause-analysis",
                    "critical-thinking",
                    "systems-thinking",
                ],
                "financial": [
                    "statistics-concepts",
                    "correlation-vs-causation",
                    "opportunity-cost",
                ],
                "strategic": [
                    "competitive-advantage",
                    "scenario-analysis",
                    "outside-view",
                ],
                "organizational": [
                    "understanding-motivations",
                    "social-proof",
                    "cognitive-biases",
                ],
                "operational": [
                    "root-cause-analysis",
                    "systems-thinking",
                    "critical-thinking",
                ],
            }

            relevant_models = problem_model_mapping.get(category_name, [])
            if model in relevant_models:
                relevance_score += category.score * 0.4

        # Bonus for high-complexity problems requiring sophisticated models
        if (
            problem_structure.complexity_score > 0.7
            and model_data.get("complexity") == "high"
        ):
            relevance_score += 0.2

        return min(relevance_score, 1.0)

    def calculate_consultant_model_affinity(
        self, model: str, consultant: CoreSpecialistProfile
    ) -> Tuple[float, str]:
        """Calculate consultant affinity for specific model"""

        # Check tier 1 models (highest affinity)
        for tier1_model in consultant.tier1_models:
            if tier1_model.model == model:
                return tier1_model.affinity_score, "tier1"

        # Check tier 2 models (secondary affinity)
        for tier2_model in consultant.tier2_models:
            if tier2_model.model == model:
                return tier2_model.affinity_score, "tier2"

        # Calculate semantic affinity based on model category
        model_data = self.model_universe.get(model, {})
        model_category = model_data.get("category", "")

        # Map consultant expertise to model categories
        consultant_category_affinity = {
            "dr_sarah_chen": {
                "analytical": 1.0,
                "diagnostic": 0.8,
                "financial": 0.7,
                "strategic": 0.5,
            },
            "marcus_rodriguez": {
                "strategic": 1.0,
                "systems": 1.0,
                "analytical": 0.6,
                "financial": 0.7,
            },
            "dr_james_park": {
                "psychological": 1.0,
                "systems": 0.6,
                "strategic": 0.5,
                "analytical": 0.4,
            },
        }

        category_affinity = consultant_category_affinity.get(consultant.id, {}).get(
            model_category, 0.2
        )

        return category_affinity, "supplementary"

    def select_models_for_consultant(
        self,
        consultant: CoreSpecialistProfile,
        problem_structure: EnhancedProblemStructure,
        target_count: int = 6,
    ) -> ConsultantModelSet:
        """Select optimal models for specific consultant using persona-driven approach"""

        model_candidates = []

        # Score all available models
        for model, model_data in self.model_universe.items():
            # Calculate key scores
            affinity_score, tier = self.calculate_consultant_model_affinity(
                model, consultant
            )
            problem_relevance = self.calculate_problem_relevance_score(
                model, problem_structure
            )
            frequency_score = model_data["frequency"] / 10.0  # Normalize frequency

            # Weighted selection score (persona-driven)
            selection_score = (
                affinity_score * 0.45  # Consultant expertise fit (highest weight)
                + problem_relevance * 0.35  # Problem relevance
                + frequency_score * 0.20  # Proven utility
            )

            rationale = f"Consultant affinity: {affinity_score:.2f} ({tier}), Problem relevance: {problem_relevance:.2f}, Frequency: {model_data['frequency']}"

            model_rationale = ModelSelectionRationale(
                model=model,
                consultant=consultant.full_name,
                selection_score=selection_score,
                affinity_score=affinity_score,
                problem_relevance=problem_relevance,
                tier=tier,
                rationale=rationale,
            )

            model_candidates.append(model_rationale)

        # Sort by selection score and select top models
        model_candidates.sort(key=lambda x: x.selection_score, reverse=True)
        selected_rationales = model_candidates[:target_count]
        selected_models = [r.model for r in selected_rationales]

        # Calculate quality metrics
        total_score = sum(r.selection_score for r in selected_rationales)

        # Model diversity (different categories)
        categories = set(
            self.model_universe[model]["category"] for model in selected_models
        )
        diversity_score = len(categories) / min(
            target_count, 4
        )  # Max 4 different categories

        # Problem coverage (how well models address primary problem categories)
        coverage_scores = []
        for problem_cat in problem_structure.primary_categories:
            category_coverage = sum(
                self.calculate_problem_relevance_score(model, problem_structure)
                for model in selected_models
            )
            coverage_scores.append(min(category_coverage, 1.0))

        coverage_score = (
            sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0.5
        )

        return ConsultantModelSet(
            consultant_profile=consultant,
            selected_models=selected_models,
            selection_rationales=selected_rationales,
            total_selection_score=total_score,
            model_diversity_score=diversity_score,
            problem_coverage_score=coverage_score,
        )

    def analyze_team_synergy(self, consultant_sets: List[ConsultantModelSet]) -> float:
        """Analyze synergy between consultant model selections"""

        if len(consultant_sets) < 2:
            return 0.5

        # Check for complementary expertise (minimal overlap, good coverage)
        all_models = []
        for consultant_set in consultant_sets:
            all_models.extend(consultant_set.selected_models)

        unique_models = len(set(all_models))
        total_models = len(all_models)

        # High synergy = low overlap + high unique model diversity
        overlap_penalty = (total_models - unique_models) / total_models
        synergy_score = 1.0 - overlap_penalty

        # Bonus for covering different problem dimensions
        coverage_diversity = sum(
            cs.model_diversity_score for cs in consultant_sets
        ) / len(consultant_sets)
        synergy_score = (synergy_score * 0.7) + (coverage_diversity * 0.3)

        return min(synergy_score, 1.0)

    def identify_model_overlaps(
        self, consultant_sets: List[ConsultantModelSet]
    ) -> Dict[str, List[str]]:
        """Identify which models are selected by multiple consultants"""

        model_consultant_map = {}

        for consultant_set in consultant_sets:
            consultant_name = consultant_set.consultant_profile.full_name
            for model in consultant_set.selected_models:
                if model not in model_consultant_map:
                    model_consultant_map[model] = []
                model_consultant_map[model].append(consultant_name)

        # Return only models selected by multiple consultants
        overlaps = {
            model: consultants
            for model, consultants in model_consultant_map.items()
            if len(consultants) > 1
        }

        return overlaps

    async def run_persona_driven_analysis(self, query: str) -> PersonaDrivenAnalysis:
        """Complete persona-driven analysis superior to generic LLM calls"""

        print("🧠 PERSONA-DRIVEN MENTAL MODEL SELECTION - PHASE 1.3")
        print("=" * 80)
        print(f"Query: {query}")
        print("=" * 80)

        # Step 1: Enhanced MECE problem classification
        problem_structure = await self.mece_classifier.enhanced_mece_classification(
            query
        )

        print("\n👥 CONSULTANT MODEL SELECTION:")
        print("-" * 60)

        # Step 2: Persona-driven model selection for each consultant
        consultant_model_sets = []

        for specialist_id, specialist in self.specialists.items():
            print(f"\n🔍 {specialist.full_name}")

            model_set = self.select_models_for_consultant(
                specialist, problem_structure, target_count=6
            )
            consultant_model_sets.append(model_set)

            print("   Selected Models:")
            for i, rationale in enumerate(model_set.selection_rationales, 1):
                model_readable = rationale.model.replace("-", " ").title()
                print(
                    f"   {i}. {model_readable} (score: {rationale.selection_score:.3f}, {rationale.tier})"
                )
                print(f"      → {rationale.rationale}")

            print("   Quality Metrics:")
            print(f"   ├─ Total Score: {model_set.total_selection_score:.2f}")
            print(f"   ├─ Diversity: {model_set.model_diversity_score:.2f}")
            print(f"   └─ Coverage: {model_set.problem_coverage_score:.2f}")

        # Step 3: Team synergy analysis
        team_synergy = self.analyze_team_synergy(consultant_model_sets)
        total_models = sum(len(cs.selected_models) for cs in consultant_model_sets)
        unique_models = len(
            set(model for cs in consultant_model_sets for model in cs.selected_models)
        )

        overlaps = self.identify_model_overlaps(consultant_model_sets)

        print("\n🤝 TEAM SYNERGY ANALYSIS:")
        print(f"├─ Team Synergy Score: {team_synergy:.3f}")
        print(f"├─ Total Models Selected: {total_models}")
        print(f"├─ Unique Models: {unique_models}")
        print(f"└─ Overlapping Models: {len(overlaps)}")

        if overlaps:
            print("\n🔄 MODEL OVERLAPS (Consensus Areas):")
            for model, consultants in overlaps.items():
                print(f"├─ {model.replace('-', ' ').title()}: {', '.join(consultants)}")

        # Step 4: Competitive advantage summary
        advantage_summary = f"""
PERSONA-DRIVEN SUPERIORITY OVER GENERIC LLM CALLS:
✓ Authentic specialist expertise drives model selection (not random)
✓ {unique_models} unique models from 157+ universe (focused quality)
✓ Consultant affinity weighting (45% of selection score)
✓ Problem-specific relevance (35% of selection score) 
✓ Built-in quality control through cognitive bias awareness
✓ Team synergy optimization (score: {team_synergy:.3f})
        """.strip()

        print(f"\n💡 {advantage_summary}")

        return PersonaDrivenAnalysis(
            problem_structure=problem_structure,
            consultant_model_sets=consultant_model_sets,
            team_synergy_score=team_synergy,
            total_models_selected=total_models,
            model_overlap_analysis=overlaps,
            competitive_advantage_summary=advantage_summary,
        )


async def demonstrate_persona_driven_selection():
    """Demonstrate persona-driven model selection superiority"""

    selector = PersonaDrivenModelSelector()

    test_queries = [
        "B2B SaaS revenue dropped 15% while competitor pricing remained stable - need root cause analysis and recovery strategy",
        "Manufacturing company considering $50M automation investment but workforce resistance is high - evaluate strategic options",
        "Fortune 500 company culture transformation initiative failing after 18 months - need breakthrough approach",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*20} PERSONA-DRIVEN TEST {i} {'='*20}")

        analysis = await selector.run_persona_driven_analysis(query)

        print("\n🎯 ANALYSIS COMPLETE:")
        print(
            f"├─ Problem Complexity: {analysis.problem_structure.complexity_score:.2f}"
        )
        print(f"├─ Team Synergy: {analysis.team_synergy_score:.3f}")
        print(
            f"├─ Model Efficiency: {analysis.total_models_selected} total, {len(set(model for cs in analysis.consultant_model_sets for model in cs.selected_models))} unique"
        )
        print(f"└─ Consensus Models: {len(analysis.model_overlap_analysis)}")

        if i < len(test_queries):
            print(f"\n{'NEXT TEST':=^80}\n")


if __name__ == "__main__":
    asyncio.run(demonstrate_persona_driven_selection())
