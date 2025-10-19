"""
CognitiveConsultantRouter - Advanced Consultant Selection System
Implements sophisticated routing logic from IMPLEMENTATION_HANDOVER.md
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from collections import defaultdict

from src.core.unified_context_stream import UnifiedContextStream, ContextEventType
from src.core.twelve_factor_compliance import TwelveFactorAgent
from src.engine.core.feature_flags import FeatureFlagService as FeatureFlagManager
from src.engine.core.llm_manager import get_llm_manager
from src.engine.core.structured_logging import get_logger

logger = get_logger(__name__)


@dataclass
class ConsultantProfile:
    """Profile defining a consultant's capabilities"""

    id: str
    name: str
    expertise: List[str]
    problem_types: List[str]
    strengths: List[str]
    weaknesses: List[str]
    typical_approach: str
    thinking_style: str  # e.g., "analytical", "creative", "systematic"
    confidence_threshold: float = 0.7
    max_concurrent_problems: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsultantSelection:
    """Result of consultant selection process"""

    selected_consultants: List[ConsultantProfile]
    selection_rationale: Dict[str, str]
    diversity_score: float
    expertise_coverage: float
    redundancy_factor: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class CognitiveConsultantRouter(TwelveFactorAgent):
    """
    Advanced consultant router that selects optimal consultant mix based on:
    - Problem complexity and type
    - Required expertise domains
    - Diversity of perspectives
    - Complementary skills
    """

    def __init__(
        self,
        context_stream: UnifiedContextStream,
        feature_flags: Optional[FeatureFlagManager] = None,
        consultant_pool: Optional[List[ConsultantProfile]] = None,
    ):
        """Initialize the consultant router"""
        from src.core.twelve_factor_compliance import TwelveFactorConfig

        twelve_factor_config = TwelveFactorConfig(
            service_name="cognitive_consultant_router"
        )
        super().__init__(twelve_factor_config)

        self.context_stream = context_stream
        self.feature_flags = feature_flags or FeatureFlagManager()

        # Initialize default consultant pool if none provided
        self.consultant_pool = consultant_pool or self._initialize_default_consultants()

        # Performance tracking
        self._routing_history: List[ConsultantSelection] = []
        self._expertise_usage: Dict[str, int] = defaultdict(int)

        logger.info(
            f"CognitiveConsultantRouter initialized with {len(self.consultant_pool)} consultants"
        )

    def _initialize_default_consultants(self) -> List[ConsultantProfile]:
        """Initialize the default consultant pool"""
        return [
            ConsultantProfile(
                id="strategic_analyst",
                name="Strategic Analyst",
                expertise=["strategy", "business_analysis", "market_dynamics"],
                problem_types=["strategic_planning", "competitive_analysis", "growth"],
                strengths=["big_picture_thinking", "trend_analysis", "synthesis"],
                weaknesses=["implementation_details", "technical_depth"],
                typical_approach="Top-down strategic framework analysis",
                thinking_style="analytical",
            ),
            ConsultantProfile(
                id="risk_assessor",
                name="Risk Assessor",
                expertise=["risk_management", "compliance", "security"],
                problem_types=["risk_assessment", "threat_analysis", "mitigation"],
                strengths=["threat_identification", "scenario_planning", "prevention"],
                weaknesses=["innovation", "growth_opportunities"],
                typical_approach="Systematic risk identification and mitigation",
                thinking_style="systematic",
            ),
            ConsultantProfile(
                id="innovation_catalyst",
                name="Innovation Catalyst",
                expertise=["innovation", "creativity", "disruption"],
                problem_types=["innovation", "transformation", "new_opportunities"],
                strengths=["creative_thinking", "paradigm_shifts", "ideation"],
                weaknesses=["risk_awareness", "operational_constraints"],
                typical_approach="Creative exploration and boundary pushing",
                thinking_style="creative",
            ),
            ConsultantProfile(
                id="operations_optimizer",
                name="Operations Optimizer",
                expertise=["operations", "efficiency", "process_improvement"],
                problem_types=["optimization", "efficiency", "cost_reduction"],
                strengths=["process_analysis", "bottleneck_identification", "metrics"],
                weaknesses=["strategic_vision", "innovation"],
                typical_approach="Data-driven operational analysis",
                thinking_style="analytical",
            ),
            ConsultantProfile(
                id="customer_advocate",
                name="Customer Advocate",
                expertise=["customer_experience", "user_research", "satisfaction"],
                problem_types=["customer_issues", "experience_design", "retention"],
                strengths=["empathy", "user_perspective", "feedback_analysis"],
                weaknesses=["technical_implementation", "financial_constraints"],
                typical_approach="User-centered design thinking",
                thinking_style="empathetic",
            ),
            ConsultantProfile(
                id="technical_architect",
                name="Technical Architect",
                expertise=["technology", "architecture", "systems_design"],
                problem_types=["technical_design", "scalability", "integration"],
                strengths=["technical_depth", "system_thinking", "feasibility"],
                weaknesses=["business_context", "user_experience"],
                typical_approach="Systematic technical decomposition",
                thinking_style="systematic",
            ),
            ConsultantProfile(
                id="financial_analyst",
                name="Financial Analyst",
                expertise=["finance", "economics", "valuation"],
                problem_types=["financial_analysis", "investment", "budgeting"],
                strengths=["quantitative_analysis", "modeling", "roi_calculation"],
                weaknesses=["qualitative_factors", "innovation"],
                typical_approach="Financial modeling and valuation",
                thinking_style="analytical",
            ),
            ConsultantProfile(
                id="change_facilitator",
                name="Change Management Facilitator",
                expertise=["change_management", "culture", "transformation"],
                problem_types=["organizational_change", "culture_shift", "adoption"],
                strengths=["stakeholder_management", "communication", "adoption"],
                weaknesses=["technical_details", "financial_modeling"],
                typical_approach="People-centered change methodology",
                thinking_style="empathetic",
            ),
        ]

    async def select_consultants(
        self,
        problem_context: Dict[str, Any],
        max_consultants: int = 5,
        min_diversity_score: float = 0.6,
    ) -> ConsultantSelection:
        """
        Select optimal mix of consultants for the given problem.

        Args:
            problem_context: Problem description and requirements
            max_consultants: Maximum number of consultants to select
            min_diversity_score: Minimum required diversity score

        Returns:
            ConsultantSelection with selected consultants and metrics
        """
        logger.info(f"Selecting consultants for problem with max={max_consultants}")

        # Analyze problem to determine requirements
        requirements = await self._analyze_problem_requirements(problem_context)

        # Score all consultants
        consultant_scores = await self._score_consultants(requirements)

        # Select optimal combination
        selected = await self._optimize_selection(
            consultant_scores, requirements, max_consultants, min_diversity_score
        )

        # Calculate selection metrics
        diversity_score = self._calculate_diversity_score(selected)
        expertise_coverage = self._calculate_expertise_coverage(selected, requirements)
        redundancy_factor = self._calculate_redundancy_factor(selected)

        # Build selection rationale
        rationale = self._build_selection_rationale(selected, requirements)

        # Create selection result
        selection = ConsultantSelection(
            selected_consultants=selected,
            selection_rationale=rationale,
            diversity_score=diversity_score,
            expertise_coverage=expertise_coverage,
            redundancy_factor=redundancy_factor,
        )

        # Track selection
        self._routing_history.append(selection)
        for consultant in selected:
            for expertise in consultant.expertise:
                self._expertise_usage[expertise] += 1

        # Log selection to context stream
        self.context_stream.add_event(
            ContextEventType.PHASE_COMPLETED,
            {
                "stage": "consultant_selection",
                "selected_count": len(selected),
                "diversity_score": diversity_score,
                "expertise_coverage": expertise_coverage,
                "consultants": [c.id for c in selected],
            },
        )

        logger.info(
            f"Selected {len(selected)} consultants with diversity={diversity_score:.2f}, "
            f"coverage={expertise_coverage:.2f}"
        )

        return selection

    async def _analyze_problem_requirements(
        self, problem_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze the problem to determine consultant requirements.

        Args:
            problem_context: Problem description and context

        Returns:
            Requirements dictionary
        """
        problem_statement = problem_context.get("problem_statement", "")
        structured_problem = problem_context.get("structured_problem", {})

        # Use LLM to analyze problem characteristics
        llm_manager = get_llm_manager(context_stream=self.context_stream)

        analysis_prompt = f"""
        Analyze this problem and identify required expertise areas:
        
        Problem: {problem_statement}
        
        Structured Details: {json.dumps(structured_problem, indent=2)}
        
        Please provide analysis in JSON format:
        {{
            "problem_type": "primary problem type",
            "complexity_level": "low/medium/high",
            "required_expertise": ["list", "of", "expertise", "areas"],
            "key_challenges": ["main", "challenges"],
            "thinking_styles_needed": ["analytical", "creative", "systematic", "empathetic"]
        }}
        """

        response = await llm_manager.call_llm(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at analyzing problems and determining required expertise.",
                },
                {"role": "user", "content": analysis_prompt},
            ],
            temperature=0.3,
        )

        try:
            requirements = json.loads(response.get("content", "{}"))
        except:
            # Fallback to basic analysis
            requirements = {
                "problem_type": "general_analysis",
                "complexity_level": "medium",
                "required_expertise": ["strategy", "analysis"],
                "key_challenges": [],
                "thinking_styles_needed": ["analytical"],
            }

        requirements["problem_context"] = problem_context
        return requirements

    async def _score_consultants(
        self, requirements: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Score each consultant based on problem requirements.

        Args:
            requirements: Problem requirements

        Returns:
            Dictionary mapping consultant IDs to scores
        """
        scores = {}
        required_expertise = set(requirements.get("required_expertise", []))
        required_styles = set(requirements.get("thinking_styles_needed", []))
        problem_type = requirements.get("problem_type", "")

        for consultant in self.consultant_pool:
            score = 0.0

            # Score based on expertise match
            expertise_match = len(set(consultant.expertise) & required_expertise)
            score += expertise_match * 0.3

            # Score based on problem type match
            if problem_type in consultant.problem_types:
                score += 0.25

            # Score based on thinking style match
            if consultant.thinking_style in required_styles:
                score += 0.2

            # Bonus for complementary strengths
            for strength in consultant.strengths:
                if strength in str(requirements.get("key_challenges", [])):
                    score += 0.1

            # Penalty for weaknesses in critical areas
            for weakness in consultant.weaknesses:
                if weakness in required_expertise:
                    score -= 0.15

            # Normalize score
            scores[consultant.id] = max(0, min(1, score))

        return scores

    async def _optimize_selection(
        self,
        consultant_scores: Dict[str, float],
        requirements: Dict[str, Any],
        max_consultants: int,
        min_diversity_score: float,
    ) -> List[ConsultantProfile]:
        """
        Optimize consultant selection for best combination.

        Args:
            consultant_scores: Individual consultant scores
            requirements: Problem requirements
            max_consultants: Maximum consultants to select
            min_diversity_score: Minimum diversity requirement

        Returns:
            List of selected consultants
        """
        # Sort consultants by score
        sorted_consultants = sorted(
            self.consultant_pool,
            key=lambda c: consultant_scores.get(c.id, 0),
            reverse=True,
        )

        selected = []
        selected_expertise = set()
        selected_styles = set()

        for consultant in sorted_consultants:
            if len(selected) >= max_consultants:
                break

            # Check if consultant adds value
            new_expertise = set(consultant.expertise) - selected_expertise
            adds_diversity = consultant.thinking_style not in selected_styles

            # Include if score is good and adds value
            if consultant_scores.get(consultant.id, 0) > 0.3 and (
                len(new_expertise) > 0 or adds_diversity or len(selected) < 2
            ):

                selected.append(consultant)
                selected_expertise.update(consultant.expertise)
                selected_styles.add(consultant.thinking_style)

        # Ensure minimum diversity
        current_diversity = self._calculate_diversity_score(selected)
        if current_diversity < min_diversity_score and len(selected) < max_consultants:
            # Add consultants to increase diversity
            for consultant in sorted_consultants:
                if (
                    consultant not in selected
                    and consultant.thinking_style not in selected_styles
                ):
                    selected.append(consultant)
                    selected_styles.add(consultant.thinking_style)
                    if self._calculate_diversity_score(selected) >= min_diversity_score:
                        break
                    if len(selected) >= max_consultants:
                        break

        return selected

    def _calculate_diversity_score(self, consultants: List[ConsultantProfile]) -> float:
        """Calculate diversity score for selected consultants"""
        if len(consultants) <= 1:
            return 0.0

        # Diversity based on thinking styles
        unique_styles = len(set(c.thinking_style for c in consultants))
        style_diversity = unique_styles / len(consultants)

        # Diversity based on expertise areas
        all_expertise = []
        for c in consultants:
            all_expertise.extend(c.expertise)
        unique_expertise = len(set(all_expertise))
        expertise_diversity = unique_expertise / max(len(all_expertise), 1)

        # Combined diversity score
        return style_diversity * 0.6 + expertise_diversity * 0.4

    def _calculate_expertise_coverage(
        self, consultants: List[ConsultantProfile], requirements: Dict[str, Any]
    ) -> float:
        """Calculate how well selected consultants cover required expertise"""
        required_expertise = set(requirements.get("required_expertise", []))
        if not required_expertise:
            return 1.0

        covered_expertise = set()
        for consultant in consultants:
            covered_expertise.update(consultant.expertise)

        coverage = len(covered_expertise & required_expertise) / len(required_expertise)
        return coverage

    def _calculate_redundancy_factor(
        self, consultants: List[ConsultantProfile]
    ) -> float:
        """Calculate redundancy in selected consultants"""
        if len(consultants) <= 1:
            return 0.0

        # Count overlapping expertise
        expertise_counts = defaultdict(int)
        for consultant in consultants:
            for expertise in consultant.expertise:
                expertise_counts[expertise] += 1

        # Calculate redundancy
        total_expertise = sum(expertise_counts.values())
        redundant_expertise = sum(
            count - 1 for count in expertise_counts.values() if count > 1
        )

        redundancy = redundant_expertise / max(total_expertise, 1)
        return redundancy

    def _build_selection_rationale(
        self, consultants: List[ConsultantProfile], requirements: Dict[str, Any]
    ) -> Dict[str, str]:
        """Build rationale explaining consultant selection"""
        rationale = {}

        for consultant in consultants:
            reasons = []

            # Check expertise match
            required_expertise = set(requirements.get("required_expertise", []))
            matched_expertise = set(consultant.expertise) & required_expertise
            if matched_expertise:
                reasons.append(f"Expertise in {', '.join(matched_expertise)}")

            # Check thinking style
            if consultant.thinking_style in requirements.get(
                "thinking_styles_needed", []
            ):
                reasons.append(
                    f"{consultant.thinking_style.capitalize()} thinking style"
                )

            # Check problem type match
            problem_type = requirements.get("problem_type", "")
            if problem_type in consultant.problem_types:
                reasons.append(f"Specializes in {problem_type}")

            rationale[consultant.id] = (
                " | ".join(reasons) if reasons else "Complementary perspective"
            )

        return rationale

    def get_routing_metrics(self) -> Dict[str, Any]:
        """Get routing performance metrics"""
        if not self._routing_history:
            return {}

        return {
            "total_routings": len(self._routing_history),
            "average_diversity_score": np.mean(
                [s.diversity_score for s in self._routing_history]
            ),
            "average_expertise_coverage": np.mean(
                [s.expertise_coverage for s in self._routing_history]
            ),
            "average_redundancy": np.mean(
                [s.redundancy_factor for s in self._routing_history]
            ),
            "most_used_expertise": dict(
                sorted(self._expertise_usage.items(), key=lambda x: x[1], reverse=True)[
                    :5
                ]
            ),
        }
