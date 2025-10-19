"""
TOSCA Context Engineering Framework
==================================

Elite consulting framework for systematic context gathering before analysis.
Based on McKinsey methodologies for problem definition and boundary setting.

TOSCA Framework:
- Trouble: Gap between situation and aspiration
- Owner: Decision maker who judges success
- Success criteria: Metrics, accuracy level, timeframe
- Constraints: Solution boundaries and limitations
- Actors: Stakeholder empathy mapping
"""

import asyncio
import yaml
from typing import Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class TroubleAnalysis:
    """Analysis of the gap between situation and aspiration"""

    symptom_definition: str
    urgency_trigger: str
    magnitude_measurement: str
    why_now_factor: str
    confidence_score: float = 0.0


@dataclass
class OwnerProfile:
    """Decision maker profile and authority mapping"""

    primary_decision_maker: str
    authority_scope: str
    success_judgment_criteria: str
    perspective_framing: str
    implementation_capacity: float = 0.0


@dataclass
class SuccessCriteria:
    """Concrete success definition with measurable outcomes"""

    success_definition: str
    required_accuracy_level: str
    decision_timeframe: str
    measurable_outcomes: List[str] = field(default_factory=list)
    acceptance_threshold: float = 0.0


@dataclass
class ConstraintProfile:
    """Solution boundaries and limitation mapping"""

    resource_constraints: List[str] = field(default_factory=list)
    political_limitations: List[str] = field(default_factory=list)
    scope_boundaries: List[str] = field(default_factory=list)
    relaxable_constraints: List[str] = field(default_factory=list)
    constraint_severity: float = 0.0


@dataclass
class ActorMapping:
    """Comprehensive stakeholder empathy and influence mapping"""

    primary_stakeholders: List[str] = field(default_factory=list)
    stakeholder_objectives: Dict[str, str] = field(default_factory=dict)
    influence_levels: Dict[str, float] = field(default_factory=dict)
    concern_areas: Dict[str, List[str]] = field(default_factory=dict)
    empathy_insights: Dict[str, str] = field(default_factory=dict)


@dataclass
class TOSCAContextMap:
    """Complete TOSCA context analysis"""

    trouble: TroubleAnalysis
    owner: OwnerProfile
    success_criteria: SuccessCriteria
    constraints: ConstraintProfile
    actors: ActorMapping

    # Meta-analysis
    context_completeness_score: float = 0.0
    complexity_level: str = "unknown"
    s1_vs_s2_recommendation: str = "s1"
    porpoising_iterations: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


class TOSCAContextEngineer:
    """Elite consulting context engineering using TOSCA framework"""

    def __init__(self):
        self.framework_config = self._load_framework_config()
        self.context_questions = self._initialize_context_questions()
        self.empathy_protocols = self._initialize_empathy_protocols()

    def _load_framework_config(self) -> Dict[str, Any]:
        """Load TOSCA framework configuration"""
        try:
            config_path = (
                Path(__file__).parent.parent.parent
                / "cognitive_architecture"
                / "NWAY_ELITE_CONSULTING_FRAMEWORKS_001.yaml"
            )
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load framework config: {e}")
            return {}

    def _initialize_context_questions(self) -> Dict[str, List[str]]:
        """Initialize systematic context gathering questions"""
        return {
            "trouble": [
                "What specific symptoms define the gap between current situation and desired aspiration?",
                "Why now? What triggered the urgency of addressing this problem?",
                "How do we measure the magnitude and impact of this trouble?",
                "What would happen if this problem remains unsolved?",
                "What are the underlying root causes versus surface symptoms?",
            ],
            "owner": [
                "Who has the ultimate authority to implement any solution we recommend?",
                "Whose perspective and priorities will frame how we define success?",
                "Who bears the primary responsibility for outcomes and consequences?",
                "What is the decision-making process and hierarchy?",
                "How does the owner's background influence their view of the problem?",
            ],
            "success_criteria": [
                "What does success look like in concrete, measurable terms?",
                "What level of accuracy and precision is required for the solution?",
                "What is the timeframe for making and implementing decisions?",
                "What are the minimum acceptable outcomes versus stretch goals?",
                "How will success be measured and validated?",
            ],
            "constraints": [
                "What resources (budget, time, people) are available or unavailable?",
                "What political, organizational, or regulatory limitations exist?",
                "What scope boundaries have been set, and why?",
                "Which constraints are truly fixed versus potentially negotiable?",
                "What creative solutions could relax key constraints?",
            ],
            "actors": [
                "Who are all the stakeholders affected by this problem and potential solutions?",
                "What are each stakeholder's primary objectives and concerns?",
                "How much influence does each actor have over the outcome?",
                "What are the potential areas of resistance or support?",
                "How do different stakeholders view this problem differently?",
            ],
        }

    def _initialize_empathy_protocols(self) -> Dict[str, str]:
        """Initialize stakeholder empathy protocols"""
        return {
            "perspective_taking": "Model stakeholder's view as compellingly as they would",
            "concern_mapping": "Identify underlying fears, motivations, and objectives",
            "influence_assessment": "Evaluate decision-making power and implementation capacity",
            "resistance_anticipation": "Predict objections and areas of pushback",
            "value_alignment": "Understand what each stakeholder values most",
        }

    async def conduct_tosca_analysis(
        self,
        initial_problem_statement: str,
        available_information: Dict[str, Any] = None,
    ) -> TOSCAContextMap:
        """
        Conduct comprehensive TOSCA context analysis

        Args:
            initial_problem_statement: Raw problem description
            available_information: Any existing context or constraints

        Returns:
            Complete TOSCA context map with systematic analysis
        """
        logger.info("🔍 Starting TOSCA Context Engineering Analysis")

        # Initialize with available information
        if available_information is None:
            available_information = {}

        # Conduct systematic analysis for each TOSCA component
        trouble_analysis = await self._analyze_trouble(
            initial_problem_statement, available_information
        )
        owner_profile = await self._analyze_owner(
            initial_problem_statement, available_information
        )
        success_criteria = await self._analyze_success_criteria(
            initial_problem_statement, available_information
        )
        constraints = await self._analyze_constraints(
            initial_problem_statement, available_information
        )
        actors = await self._analyze_actors(
            initial_problem_statement, available_information
        )

        # Create comprehensive context map
        context_map = TOSCAContextMap(
            trouble=trouble_analysis,
            owner=owner_profile,
            success_criteria=success_criteria,
            constraints=constraints,
            actors=actors,
        )

        # Perform meta-analysis
        context_map = await self._perform_meta_analysis(context_map)

        logger.info(
            f"✅ TOSCA Analysis Complete - Complexity: {context_map.complexity_level}, "
            f"Completeness: {context_map.context_completeness_score:.2f}"
        )

        return context_map

    async def _analyze_trouble(
        self, problem_statement: str, available_info: Dict[str, Any]
    ) -> TroubleAnalysis:
        """Analyze the trouble component: gap between situation and aspiration"""

        # Extract symptom definition
        symptom_definition = self._extract_symptoms(problem_statement)

        # Identify urgency trigger
        urgency_trigger = self._identify_urgency_trigger(
            problem_statement, available_info
        )

        # Determine magnitude measurement approach
        magnitude_measurement = self._determine_magnitude_measurement(problem_statement)

        # Identify "why now" factor
        why_now_factor = self._identify_why_now_factor(
            problem_statement, available_info
        )

        # Calculate confidence score
        confidence_score = self._calculate_trouble_confidence(
            symptom_definition, urgency_trigger, magnitude_measurement, why_now_factor
        )

        return TroubleAnalysis(
            symptom_definition=symptom_definition,
            urgency_trigger=urgency_trigger,
            magnitude_measurement=magnitude_measurement,
            why_now_factor=why_now_factor,
            confidence_score=confidence_score,
        )

    async def _analyze_owner(
        self, problem_statement: str, available_info: Dict[str, Any]
    ) -> OwnerProfile:
        """Analyze the owner component: decision maker identification"""

        # Identify primary decision maker
        primary_decision_maker = self._identify_decision_maker(
            problem_statement, available_info
        )

        # Map authority scope
        authority_scope = self._map_authority_scope(
            primary_decision_maker, available_info
        )

        # Define success judgment criteria
        success_judgment_criteria = self._define_success_judgment(
            primary_decision_maker, problem_statement
        )

        # Analyze perspective framing
        perspective_framing = self._analyze_perspective_framing(
            primary_decision_maker, problem_statement
        )

        # Assess implementation capacity
        implementation_capacity = self._assess_implementation_capacity(
            primary_decision_maker, available_info
        )

        return OwnerProfile(
            primary_decision_maker=primary_decision_maker,
            authority_scope=authority_scope,
            success_judgment_criteria=success_judgment_criteria,
            perspective_framing=perspective_framing,
            implementation_capacity=implementation_capacity,
        )

    async def _analyze_success_criteria(
        self, problem_statement: str, available_info: Dict[str, Any]
    ) -> SuccessCriteria:
        """Analyze success criteria: measurable outcomes and thresholds"""

        # Define concrete success
        success_definition = self._define_concrete_success(problem_statement)

        # Determine required accuracy
        required_accuracy_level = self._determine_accuracy_requirements(
            problem_statement, available_info
        )

        # Establish decision timeframe
        decision_timeframe = self._establish_decision_timeframe(
            problem_statement, available_info
        )

        # Identify measurable outcomes
        measurable_outcomes = self._identify_measurable_outcomes(problem_statement)

        # Set acceptance threshold
        acceptance_threshold = self._set_acceptance_threshold(
            success_definition, available_info
        )

        return SuccessCriteria(
            success_definition=success_definition,
            required_accuracy_level=required_accuracy_level,
            decision_timeframe=decision_timeframe,
            measurable_outcomes=measurable_outcomes,
            acceptance_threshold=acceptance_threshold,
        )

    async def _analyze_constraints(
        self, problem_statement: str, available_info: Dict[str, Any]
    ) -> ConstraintProfile:
        """Analyze constraints: solution boundaries and limitations"""

        # Identify resource constraints
        resource_constraints = self._identify_resource_constraints(
            problem_statement, available_info
        )

        # Map political limitations
        political_limitations = self._map_political_limitations(
            problem_statement, available_info
        )

        # Define scope boundaries
        scope_boundaries = self._define_scope_boundaries(
            problem_statement, available_info
        )

        # Identify relaxable constraints
        relaxable_constraints = self._identify_relaxable_constraints(
            resource_constraints, political_limitations, scope_boundaries
        )

        # Assess constraint severity
        constraint_severity = self._assess_constraint_severity(
            resource_constraints, political_limitations, scope_boundaries
        )

        return ConstraintProfile(
            resource_constraints=resource_constraints,
            political_limitations=political_limitations,
            scope_boundaries=scope_boundaries,
            relaxable_constraints=relaxable_constraints,
            constraint_severity=constraint_severity,
        )

    async def _analyze_actors(
        self, problem_statement: str, available_info: Dict[str, Any]
    ) -> ActorMapping:
        """Analyze actors: comprehensive stakeholder empathy mapping"""

        # Identify primary stakeholders
        primary_stakeholders = self._identify_primary_stakeholders(
            problem_statement, available_info
        )

        # Map stakeholder objectives
        stakeholder_objectives = {}
        influence_levels = {}
        concern_areas = {}
        empathy_insights = {}

        for stakeholder in primary_stakeholders:
            # Analyze each stakeholder's objectives
            stakeholder_objectives[stakeholder] = self._analyze_stakeholder_objectives(
                stakeholder, problem_statement
            )

            # Assess influence level
            influence_levels[stakeholder] = self._assess_stakeholder_influence(
                stakeholder, available_info
            )

            # Map concern areas
            concern_areas[stakeholder] = self._map_stakeholder_concerns(
                stakeholder, problem_statement
            )

            # Generate empathy insights
            empathy_insights[stakeholder] = self._generate_empathy_insights(
                stakeholder, problem_statement, stakeholder_objectives[stakeholder]
            )

        return ActorMapping(
            primary_stakeholders=primary_stakeholders,
            stakeholder_objectives=stakeholder_objectives,
            influence_levels=influence_levels,
            concern_areas=concern_areas,
            empathy_insights=empathy_insights,
        )

    async def _perform_meta_analysis(
        self, context_map: TOSCAContextMap
    ) -> TOSCAContextMap:
        """Perform meta-analysis on complete TOSCA context map"""

        # Calculate context completeness score
        completeness_score = self._calculate_completeness_score(context_map)
        context_map.context_completeness_score = completeness_score

        # Determine complexity level
        complexity_level = self._determine_complexity_level(context_map)
        context_map.complexity_level = complexity_level

        # Generate S1 vs S2 recommendation
        s1_vs_s2_recommendation = self._generate_tier_recommendation(context_map)
        context_map.s1_vs_s2_recommendation = s1_vs_s2_recommendation

        return context_map

    def _extract_symptoms(self, problem_statement: str) -> str:
        """Extract specific symptoms from problem statement"""
        # Simple extraction - in production would use NLP
        symptoms = []

        # Look for symptom keywords
        symptom_keywords = [
            "declining",
            "increasing",
            "problem",
            "issue",
            "challenge",
            "failure",
            "loss",
            "reduced",
            "poor",
            "low",
            "high",
        ]

        words = problem_statement.lower().split()
        for i, word in enumerate(words):
            if word in symptom_keywords and i + 1 < len(words):
                symptoms.append(f"{word} {words[i+1]}")

        return (
            "; ".join(symptoms)
            if symptoms
            else "Symptoms not clearly defined in problem statement"
        )

    def _identify_urgency_trigger(
        self, problem_statement: str, available_info: Dict[str, Any]
    ) -> str:
        """Identify what triggered the urgency of this problem"""
        urgency_keywords = [
            "urgent",
            "immediate",
            "crisis",
            "emergency",
            "deadline",
            "quickly",
            "soon",
            "asap",
            "critical",
        ]

        words = problem_statement.lower().split()
        for word in words:
            if word in urgency_keywords:
                return f"Urgency indicated by: {word}"

        return "Urgency trigger not specified - requires clarification"

    def _determine_magnitude_measurement(self, problem_statement: str) -> str:
        """Determine how to measure the magnitude of the problem"""
        # Look for quantitative indicators
        measurement_keywords = [
            "percent",
            "%",
            "million",
            "thousand",
            "dollars",
            "$",
            "revenue",
            "profit",
            "cost",
            "growth",
            "decline",
        ]

        words = problem_statement.lower().split()
        measurements = [
            word for word in words if any(kw in word for kw in measurement_keywords)
        ]

        if measurements:
            return f"Quantitative indicators present: {', '.join(measurements)}"
        else:
            return "Magnitude measurement approach needs definition"

    def _identify_why_now_factor(
        self, problem_statement: str, available_info: Dict[str, Any]
    ) -> str:
        """Identify the 'why now' factor driving current attention"""
        timing_keywords = [
            "recent",
            "recently",
            "new",
            "suddenly",
            "since",
            "after",
            "market",
            "competition",
            "change",
            "shift",
        ]

        words = problem_statement.lower().split()
        timing_factors = [word for word in words if word in timing_keywords]

        if timing_factors:
            return f"Timing factors: {', '.join(timing_factors)}"
        else:
            return "Why now factor requires investigation"

    def _calculate_trouble_confidence(
        self, symptom_def: str, urgency: str, magnitude: str, why_now: str
    ) -> float:
        """Calculate confidence score for trouble analysis"""
        score = 0.0

        # Check if each component is well-defined
        if (
            "not clearly defined" not in symptom_def
            and "not specified" not in symptom_def
        ):
            score += 0.25

        if "not specified" not in urgency and "requires clarification" not in urgency:
            score += 0.25

        if "needs definition" not in magnitude:
            score += 0.25

        if "requires investigation" not in why_now:
            score += 0.25

        return score

    def _identify_decision_maker(
        self, problem_statement: str, available_info: Dict[str, Any]
    ) -> str:
        """Identify the primary decision maker"""
        decision_keywords = [
            "ceo",
            "cto",
            "manager",
            "director",
            "president",
            "owner",
            "leader",
            "executive",
            "board",
            "committee",
        ]

        words = problem_statement.lower().split()
        decision_makers = [word for word in words if word in decision_keywords]

        if decision_makers:
            return decision_makers[0].upper()
        else:
            return "Decision maker not specified - requires identification"

    def _map_authority_scope(
        self, decision_maker: str, available_info: Dict[str, Any]
    ) -> str:
        """Map the scope of the decision maker's authority"""
        if "not specified" in decision_maker:
            return "Authority scope unclear without decision maker identification"

        # Default mapping based on role
        authority_mapping = {
            "CEO": "Full organizational authority",
            "CTO": "Technology and engineering decisions",
            "CFO": "Financial and budget decisions",
            "MANAGER": "Departmental authority",
            "DIRECTOR": "Division-level authority",
        }

        return authority_mapping.get(
            decision_maker, "Authority scope requires clarification"
        )

    def _define_success_judgment(
        self, decision_maker: str, problem_statement: str
    ) -> str:
        """Define how the decision maker will judge success"""
        if "not specified" in decision_maker:
            return "Success judgment criteria unclear without decision maker"

        # Extract success indicators from problem statement
        success_keywords = [
            "increase",
            "improve",
            "reduce",
            "eliminate",
            "achieve",
            "revenue",
            "profit",
            "efficiency",
            "quality",
            "satisfaction",
        ]

        words = problem_statement.lower().split()
        success_indicators = [word for word in words if word in success_keywords]

        if success_indicators:
            return f"Success likely judged by: {', '.join(success_indicators)}"
        else:
            return "Success judgment criteria require explicit definition"

    def _analyze_perspective_framing(
        self, decision_maker: str, problem_statement: str
    ) -> str:
        """Analyze how the decision maker's perspective frames the problem"""
        if "not specified" in decision_maker:
            return "Perspective framing unclear without decision maker identification"

        # Role-based perspective framing
        perspective_mapping = {
            "CEO": "Strategic and organizational performance perspective",
            "CTO": "Technical feasibility and innovation perspective",
            "CFO": "Financial impact and ROI perspective",
            "MANAGER": "Operational efficiency perspective",
            "DIRECTOR": "Division performance and resource perspective",
        }

        return perspective_mapping.get(
            decision_maker, "Perspective framing requires analysis"
        )

    def _assess_implementation_capacity(
        self, decision_maker: str, available_info: Dict[str, Any]
    ) -> float:
        """Assess the decision maker's implementation capacity"""
        if "not specified" in decision_maker:
            return 0.0

        # Simple capacity mapping
        capacity_mapping = {
            "CEO": 0.9,
            "CTO": 0.7,
            "CFO": 0.6,
            "MANAGER": 0.5,
            "DIRECTOR": 0.7,
        }

        return capacity_mapping.get(decision_maker, 0.3)

    def _define_concrete_success(self, problem_statement: str) -> str:
        """Define concrete, measurable success criteria"""
        success_keywords = ["increase", "improve", "reduce", "eliminate", "achieve"]

        words = problem_statement.lower().split()
        success_elements = []

        for i, word in enumerate(words):
            if word in success_keywords and i + 1 < len(words):
                success_elements.append(f"{word} {words[i+1]}")

        if success_elements:
            return f"Success defined as: {'; '.join(success_elements)}"
        else:
            return "Concrete success definition requires explicit articulation"

    def _determine_accuracy_requirements(
        self, problem_statement: str, available_info: Dict[str, Any]
    ) -> str:
        """Determine required accuracy level for the solution"""
        accuracy_keywords = [
            "precise",
            "exact",
            "approximately",
            "rough",
            "ballpark",
            "detailed",
            "high-level",
            "estimate",
        ]

        words = problem_statement.lower().split()
        accuracy_indicators = [word for word in words if word in accuracy_keywords]

        if accuracy_indicators:
            return f"Accuracy level indicated: {', '.join(accuracy_indicators)}"
        else:
            return "Required accuracy level needs specification"

    def _establish_decision_timeframe(
        self, problem_statement: str, available_info: Dict[str, Any]
    ) -> str:
        """Establish the decision and implementation timeframe"""
        timing_keywords = [
            "immediately",
            "urgent",
            "asap",
            "month",
            "year",
            "quarter",
            "deadline",
            "by",
            "before",
            "soon",
            "quickly",
        ]

        words = problem_statement.lower().split()
        timing_indicators = [word for word in words if word in timing_keywords]

        if timing_indicators:
            return f"Timeframe indicated: {', '.join(timing_indicators)}"
        else:
            return "Decision timeframe requires explicit definition"

    def _identify_measurable_outcomes(self, problem_statement: str) -> List[str]:
        """Identify specific measurable outcomes"""
        outcome_keywords = [
            "revenue",
            "profit",
            "cost",
            "efficiency",
            "quality",
            "satisfaction",
            "performance",
            "growth",
            "market share",
        ]

        words = problem_statement.lower().split()
        outcomes = [word for word in words if word in outcome_keywords]

        return outcomes if outcomes else ["Measurable outcomes require definition"]

    def _set_acceptance_threshold(
        self, success_definition: str, available_info: Dict[str, Any]
    ) -> float:
        """Set acceptance threshold for success"""
        if "requires explicit" in success_definition:
            return 0.0

        # Default threshold based on success clarity
        return 0.8 if "Success defined as" in success_definition else 0.3

    def _identify_resource_constraints(
        self, problem_statement: str, available_info: Dict[str, Any]
    ) -> List[str]:
        """Identify resource constraints"""
        resource_keywords = [
            "budget",
            "money",
            "funds",
            "cost",
            "time",
            "people",
            "staff",
            "team",
            "resources",
            "limited",
            "constraint",
        ]

        words = problem_statement.lower().split()
        constraints = [word for word in words if word in resource_keywords]

        return constraints if constraints else ["Resource constraints not specified"]

    def _map_political_limitations(
        self, problem_statement: str, available_info: Dict[str, Any]
    ) -> List[str]:
        """Map political and organizational limitations"""
        political_keywords = [
            "approval",
            "policy",
            "regulation",
            "compliance",
            "stakeholder",
            "resistance",
            "politics",
            "culture",
        ]

        words = problem_statement.lower().split()
        limitations = [word for word in words if word in political_keywords]

        return (
            limitations if limitations else ["Political limitations require assessment"]
        )

    def _define_scope_boundaries(
        self, problem_statement: str, available_info: Dict[str, Any]
    ) -> List[str]:
        """Define scope boundaries"""
        scope_keywords = [
            "scope",
            "focus",
            "limit",
            "boundary",
            "include",
            "exclude",
            "department",
            "division",
            "region",
            "product",
        ]

        words = problem_statement.lower().split()
        boundaries = [word for word in words if word in scope_keywords]

        return boundaries if boundaries else ["Scope boundaries need definition"]

    def _identify_relaxable_constraints(
        self,
        resource_constraints: List[str],
        political_limitations: List[str],
        scope_boundaries: List[str],
    ) -> List[str]:
        """Identify which constraints might be relaxable"""
        all_constraints = (
            resource_constraints + political_limitations + scope_boundaries
        )

        # Simple heuristic: time and budget constraints are often more flexible
        relaxable = []
        for constraint in all_constraints:
            if any(word in constraint for word in ["time", "budget", "scope"]):
                relaxable.append(f"Potentially relaxable: {constraint}")

        return relaxable if relaxable else ["Constraint flexibility requires analysis"]

    def _assess_constraint_severity(
        self,
        resource_constraints: List[str],
        political_limitations: List[str],
        scope_boundaries: List[str],
    ) -> float:
        """Assess overall constraint severity"""
        total_constraints = len(
            resource_constraints + political_limitations + scope_boundaries
        )

        # Simple severity mapping
        if total_constraints == 0:
            return 0.0
        elif total_constraints <= 3:
            return 0.3
        elif total_constraints <= 6:
            return 0.6
        else:
            return 0.9

    def _identify_primary_stakeholders(
        self, problem_statement: str, available_info: Dict[str, Any]
    ) -> List[str]:
        """Identify primary stakeholders"""
        stakeholder_keywords = [
            "customer",
            "client",
            "user",
            "employee",
            "team",
            "manager",
            "executive",
            "board",
            "investor",
            "partner",
        ]

        words = problem_statement.lower().split()
        stakeholders = [word for word in words if word in stakeholder_keywords]

        # Remove duplicates and ensure we have at least basic stakeholders
        unique_stakeholders = list(set(stakeholders))
        if not unique_stakeholders:
            unique_stakeholders = ["Primary stakeholders require identification"]

        return unique_stakeholders

    def _analyze_stakeholder_objectives(
        self, stakeholder: str, problem_statement: str
    ) -> str:
        """Analyze individual stakeholder objectives"""
        if "require identification" in stakeholder:
            return "Objectives unclear without stakeholder identification"

        # Role-based objective mapping
        objective_mapping = {
            "customer": "Value, quality, service, satisfaction",
            "employee": "Job security, career growth, work conditions",
            "manager": "Team performance, efficiency, results",
            "executive": "Strategic outcomes, organizational performance",
            "investor": "ROI, growth, risk management",
            "board": "Governance, oversight, strategic direction",
        }

        return objective_mapping.get(
            stakeholder, f"Objectives for {stakeholder} require analysis"
        )

    def _assess_stakeholder_influence(
        self, stakeholder: str, available_info: Dict[str, Any]
    ) -> float:
        """Assess stakeholder influence level"""
        if "require identification" in stakeholder:
            return 0.0

        # Role-based influence mapping
        influence_mapping = {
            "customer": 0.8,
            "employee": 0.4,
            "manager": 0.6,
            "executive": 0.9,
            "investor": 0.7,
            "board": 0.9,
        }

        return influence_mapping.get(stakeholder, 0.3)

    def _map_stakeholder_concerns(
        self, stakeholder: str, problem_statement: str
    ) -> List[str]:
        """Map stakeholder concern areas"""
        if "require identification" in stakeholder:
            return ["Concerns unclear without stakeholder identification"]

        # Role-based concern mapping
        concern_mapping = {
            "customer": [
                "Quality degradation",
                "Price increases",
                "Service disruption",
            ],
            "employee": ["Job security", "Workload changes", "Skill requirements"],
            "manager": ["Team disruption", "Performance impact", "Resource allocation"],
            "executive": ["Strategic risk", "Financial impact", "Reputation"],
            "investor": ["ROI impact", "Market position", "Risk exposure"],
            "board": [
                "Governance implications",
                "Strategic alignment",
                "Risk management",
            ],
        }

        return concern_mapping.get(
            stakeholder, [f"Concerns for {stakeholder} require analysis"]
        )

    def _generate_empathy_insights(
        self, stakeholder: str, problem_statement: str, objectives: str
    ) -> str:
        """Generate empathy insights for stakeholder"""
        if "require identification" in stakeholder or "require analysis" in objectives:
            return "Empathy insights require stakeholder and objective identification"

        return (
            f"From {stakeholder} perspective: {objectives} drive their view of this problem. "
            f"Success for them means achieving these objectives without negative side effects."
        )

    def _calculate_completeness_score(self, context_map: TOSCAContextMap) -> float:
        """Calculate overall context completeness score"""
        scores = []

        # Trouble completeness
        trouble_score = context_map.trouble.confidence_score
        scores.append(trouble_score)

        # Owner completeness
        owner_score = context_map.owner.implementation_capacity
        scores.append(owner_score)

        # Success criteria completeness
        success_score = context_map.success_criteria.acceptance_threshold
        scores.append(success_score)

        # Constraint completeness (inverse of severity, since high severity = good mapping)
        constraint_score = min(context_map.constraints.constraint_severity, 0.8)
        scores.append(constraint_score)

        # Actor completeness (based on number of identified stakeholders)
        actor_score = min(len(context_map.actors.primary_stakeholders) * 0.2, 1.0)
        scores.append(actor_score)

        return sum(scores) / len(scores)

    def _determine_complexity_level(self, context_map: TOSCAContextMap) -> str:
        """Determine problem complexity level"""
        complexity_score = 0

        # Add complexity factors
        if context_map.constraints.constraint_severity > 0.6:
            complexity_score += 1

        if len(context_map.actors.primary_stakeholders) > 5:
            complexity_score += 1

        if context_map.trouble.confidence_score < 0.5:
            complexity_score += 1

        if context_map.success_criteria.acceptance_threshold < 0.5:
            complexity_score += 1

        # Map to complexity levels
        if complexity_score == 0:
            return "simple"
        elif complexity_score <= 2:
            return "moderate"
        else:
            return "complex"

    def _generate_tier_recommendation(self, context_map: TOSCAContextMap) -> str:
        """Generate S1 vs S2 tier processing recommendation"""
        s2_triggers = 0

        # Check S2 escalation criteria from McKinsey framework
        if context_map.complexity_level == "complex":
            s2_triggers += 1

        if context_map.constraints.constraint_severity > 0.7:
            s2_triggers += 1

        if len(context_map.actors.primary_stakeholders) > 6:
            s2_triggers += 1

        if context_map.context_completeness_score < 0.6:
            s2_triggers += 1

        return "s2" if s2_triggers >= 2 else "s1"

    async def porpoising_refinement(
        self, context_map: TOSCAContextMap, new_information: Dict[str, Any]
    ) -> TOSCAContextMap:
        """
        Perform 'porpoising' iterative refinement with new information

        Args:
            context_map: Current TOSCA context map
            new_information: New facts or insights to incorporate

        Returns:
            Refined TOSCA context map
        """
        logger.info("🔄 Performing TOSCA Porpoising Refinement")

        # Increment porpoising iterations
        context_map.porpoising_iterations += 1

        # Re-analyze components with new information
        # This would integrate new insights and refine existing analysis
        # For now, we'll update the timestamp and recalculate meta-analysis

        context_map.timestamp = datetime.now()
        context_map = await self._perform_meta_analysis(context_map)

        logger.info(
            f"✅ Porpoising Iteration {context_map.porpoising_iterations} Complete"
        )

        return context_map

    def export_context_map(self, context_map: TOSCAContextMap) -> Dict[str, Any]:
        """Export TOSCA context map for integration with V5.4 pipeline"""
        return {
            "tosca_analysis": {
                "trouble": {
                    "symptom_definition": context_map.trouble.symptom_definition,
                    "urgency_trigger": context_map.trouble.urgency_trigger,
                    "magnitude_measurement": context_map.trouble.magnitude_measurement,
                    "why_now_factor": context_map.trouble.why_now_factor,
                    "confidence_score": context_map.trouble.confidence_score,
                },
                "owner": {
                    "primary_decision_maker": context_map.owner.primary_decision_maker,
                    "authority_scope": context_map.owner.authority_scope,
                    "success_judgment_criteria": context_map.owner.success_judgment_criteria,
                    "perspective_framing": context_map.owner.perspective_framing,
                    "implementation_capacity": context_map.owner.implementation_capacity,
                },
                "success_criteria": {
                    "success_definition": context_map.success_criteria.success_definition,
                    "required_accuracy_level": context_map.success_criteria.required_accuracy_level,
                    "decision_timeframe": context_map.success_criteria.decision_timeframe,
                    "measurable_outcomes": context_map.success_criteria.measurable_outcomes,
                    "acceptance_threshold": context_map.success_criteria.acceptance_threshold,
                },
                "constraints": {
                    "resource_constraints": context_map.constraints.resource_constraints,
                    "political_limitations": context_map.constraints.political_limitations,
                    "scope_boundaries": context_map.constraints.scope_boundaries,
                    "relaxable_constraints": context_map.constraints.relaxable_constraints,
                    "constraint_severity": context_map.constraints.constraint_severity,
                },
                "actors": {
                    "primary_stakeholders": context_map.actors.primary_stakeholders,
                    "stakeholder_objectives": context_map.actors.stakeholder_objectives,
                    "influence_levels": context_map.actors.influence_levels,
                    "concern_areas": context_map.actors.concern_areas,
                    "empathy_insights": context_map.actors.empathy_insights,
                },
                "meta_analysis": {
                    "context_completeness_score": context_map.context_completeness_score,
                    "complexity_level": context_map.complexity_level,
                    "s1_vs_s2_recommendation": context_map.s1_vs_s2_recommendation,
                    "porpoising_iterations": context_map.porpoising_iterations,
                    "timestamp": context_map.timestamp.isoformat(),
                },
            }
        }


# Usage Example
async def main():
    """Example usage of TOSCA Context Engineering"""

    # Initialize TOSCA engineer
    tosca_engineer = TOSCAContextEngineer()

    # Example problem statement
    problem_statement = """
    Our company's quarterly revenue has declined by 15% over the past two quarters, 
    and the CEO is concerned about our competitive position in the market. 
    We need to identify the root causes and develop a strategic response plan 
    within the next 30 days to present to the board.
    """

    # Conduct TOSCA analysis
    context_map = await tosca_engineer.conduct_tosca_analysis(problem_statement)

    # Export for integration
    exported_context = tosca_engineer.export_context_map(context_map)

    print("TOSCA Context Engineering Complete:")
    print(f"Complexity Level: {context_map.complexity_level}")
    print(f"Completeness Score: {context_map.context_completeness_score:.2f}")
    print(f"Recommended Tier: {context_map.s1_vs_s2_recommendation}")


if __name__ == "__main__":
    asyncio.run(main())
