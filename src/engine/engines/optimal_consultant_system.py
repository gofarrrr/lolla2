"""
METIS Optimal Consultant System - Implementation Example
Date: September 4, 2025
Purpose: Reference implementation for 9-consultant specialization matrix
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import re
import hashlib
from datetime import datetime
import asyncio

# ===============================================================
# DATA MODELS
# ===============================================================


class SpecializationRow(Enum):
    STRATEGIC = "strategic"
    TACTICAL = "tactical"
    OPERATIONAL = "operational"


class SpecializationColumn(Enum):
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    IMPLEMENTATION = "implementation"


class FrameworkCategory(Enum):
    CONSULTING_FRAMEWORK = "consulting_framework"
    NWAY_PATTERN = "nway_pattern"
    MENTAL_MODEL = "mental_model"


class AssignmentType(Enum):
    STABLE_CORE = "stable_core"
    ADAPTIVE_LAYER = "adaptive_layer"


@dataclass
class Framework:
    id: str
    name: str
    category: FrameworkCategory
    effectiveness_score: float
    usage_context: List[str]
    consultant_affinity: List[str]


@dataclass
class ConsultantBlueprint:
    consultant_id: str
    name: str
    specialization_row: SpecializationRow
    specialization_column: SpecializationColumn
    expertise_description: str
    persona_prompt: str
    stable_core_weight: float = 0.70
    adaptive_layer_weight: float = 0.30
    active: bool = True


@dataclass
class FrameworkAssignment:
    framework: Framework
    assignment_type: AssignmentType
    priority_rank: int
    effectiveness_score: float
    usage_frequency: int = 0


@dataclass
class AdaptiveTrigger:
    trigger_name: str
    keywords: List[str]
    framework: Framework
    consultant_affinity: List[str]
    boost_score: float
    activation_threshold: int = 1


@dataclass
class QueryClassification:
    keywords: List[str]
    complexity_score: int
    specialization_focus: Tuple[SpecializationRow, SpecializationColumn]
    adaptive_triggers: List[str]
    routing_pattern: Optional[str] = None


@dataclass
class ConsultantSelection:
    consultant_id: str
    blueprint: ConsultantBlueprint
    stable_frameworks: List[Framework]
    adaptive_frameworks: List[Framework]
    total_cognitive_load: float
    selection_reason: str


@dataclass
class EngagementResult:
    query: str
    selected_consultants: List[ConsultantSelection]
    processing_metadata: Dict[str, Any]
    timestamp: datetime


# ===============================================================
# CONSULTANT SYSTEM IMPLEMENTATION
# ===============================================================


class OptimalConsultantSystem:
    """
    Main system for intelligent consultant selection and framework routing.
    Implements the 9-consultant specialization matrix with stable/adaptive layers.
    """

    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self.consultant_blueprints: Dict[str, ConsultantBlueprint] = {}
        self.framework_library: Dict[str, Framework] = {}
        self.framework_assignments: Dict[str, List[FrameworkAssignment]] = {}
        self.adaptive_triggers: List[AdaptiveTrigger] = []
        self.routing_patterns: Dict[str, Dict] = {}

        # Initialize system
        asyncio.create_task(self._load_system_configuration())

    async def _load_system_configuration(self):
        """Load all configuration from Supabase database"""

        # Load consultant blueprints
        blueprints_result = (
            await self.supabase.table("consultant_blueprints")
            .select("*")
            .eq("active", True)
            .execute()
        )
        for blueprint_data in blueprints_result.data:
            blueprint = ConsultantBlueprint(
                consultant_id=blueprint_data["consultant_id"],
                name=blueprint_data["consultant_name"],
                specialization_row=SpecializationRow(
                    blueprint_data["specialization_row"]
                ),
                specialization_column=SpecializationColumn(
                    blueprint_data["specialization_column"]
                ),
                expertise_description=blueprint_data["expertise_description"],
                persona_prompt=blueprint_data["persona_prompt"],
                stable_core_weight=blueprint_data["stable_core_weight"],
                adaptive_layer_weight=blueprint_data["adaptive_layer_weight"],
            )
            self.consultant_blueprints[blueprint["consultant_id"]] = blueprint

        # Load frameworks from knowledge_elements and nway_interactions
        await self._load_frameworks()

        # Load framework assignments
        await self._load_framework_assignments()

        # Load adaptive triggers
        await self._load_adaptive_triggers()

        # Load routing patterns
        await self._load_routing_patterns()

    async def _load_frameworks(self):
        """Load frameworks from knowledge_elements and nway_interactions"""

        # Load from knowledge_elements
        ke_result = (
            await self.supabase.table("knowledge_elements").select("*").execute()
        )
        for ke in ke_result.data:
            if ke.get("framework_category"):
                framework = Framework(
                    id=ke["id"],
                    name=ke["ke_name"],
                    category=FrameworkCategory(ke["framework_category"]),
                    effectiveness_score=ke.get("effectiveness_score", 0.5),
                    usage_context=ke.get("usage_context", []),
                    consultant_affinity=ke.get("consultant_affinity", []),
                )
                self.framework_library[framework.id] = framework

        # Load from nway_interactions
        ni_result = await self.supabase.table("nway_interactions").select("*").execute()
        for ni in ni_result.data:
            framework = Framework(
                id=ni["id"],
                name=ni["interaction_id"],
                category=FrameworkCategory.NWAY_PATTERN,
                effectiveness_score=ni.get("lollapalooza_potential", 0.8),
                usage_context=ni.get("trigger_keywords", []),
                consultant_affinity=ni.get("consultant_affinity", []),
            )
            self.framework_library[framework.id] = framework

    async def _load_framework_assignments(self):
        """Load consultant framework assignments"""

        assignments_result = (
            await self.supabase.table("consultant_framework_assignments")
            .select("*")
            .execute()
        )

        for assignment_data in assignments_result.data:
            consultant_id = assignment_data["consultant_id"]
            framework_id = assignment_data["framework_id"]

            if consultant_id not in self.framework_assignments:
                self.framework_assignments[consultant_id] = []

            if framework_id in self.framework_library:
                assignment = FrameworkAssignment(
                    framework=self.framework_library[framework_id],
                    assignment_type=AssignmentType(assignment_data["assignment_type"]),
                    priority_rank=assignment_data["priority_rank"],
                    effectiveness_score=assignment_data["effectiveness_score"],
                    usage_frequency=assignment_data["usage_frequency"],
                )
                self.framework_assignments[consultant_id].append(assignment)

    async def _load_adaptive_triggers(self):
        """Load adaptive triggers"""

        triggers_result = (
            await self.supabase.table("adaptive_triggers")
            .select("*")
            .eq("active", True)
            .execute()
        )

        for trigger_data in triggers_result.data:
            framework_id = trigger_data["framework_id"]
            if framework_id in self.framework_library:
                trigger = AdaptiveTrigger(
                    trigger_name=trigger_data["trigger_name"],
                    keywords=trigger_data["trigger_keywords"],
                    framework=self.framework_library[framework_id],
                    consultant_affinity=trigger_data["consultant_affinity"],
                    boost_score=trigger_data["boost_score"],
                    activation_threshold=trigger_data["activation_threshold"],
                )
                self.adaptive_triggers.append(trigger)

    async def _load_routing_patterns(self):
        """Load query routing patterns"""

        patterns_result = (
            await self.supabase.table("query_routing_patterns")
            .select("*")
            .eq("active", True)
            .execute()
        )

        for pattern_data in patterns_result.data:
            self.routing_patterns[pattern_data["pattern_name"]] = {
                "keywords": pattern_data["query_keywords"],
                "selection_rule": pattern_data["consultant_selection_rule"],
                "complexity_threshold": pattern_data["complexity_threshold"],
                "success_rate": pattern_data["success_rate"],
            }

    async def process_query(
        self, query: str, context: Optional[Dict] = None
    ) -> EngagementResult:
        """
        Main entry point: process query and return consultant selections.

        This is the core method that preserves Multi-Single-Agent principles:
        1. Classify query
        2. Select 3 independent consultants
        3. Assign frameworks per consultant independently
        4. Return all perspectives (NO SYNTHESIS)
        """

        start_time = datetime.now()

        # Step 1: Classify the query
        classification = await self._classify_query(query, context)

        # Step 2: Select 3 consultants based on classification
        selected_consultant_ids = await self._select_consultant_trio(classification)

        # Step 3: For each consultant, independently select their frameworks
        consultant_selections = []
        for consultant_id in selected_consultant_ids:
            selection = await self._prepare_consultant_selection(
                consultant_id, classification
            )
            consultant_selections.append(selection)

        # Step 4: Log engagement for benchmarking
        await self._log_engagement(query, classification, consultant_selections)

        processing_time = (datetime.now() - start_time).total_seconds()

        return EngagementResult(
            query=query,
            selected_consultants=consultant_selections,
            processing_metadata={
                "classification": classification,
                "processing_time_seconds": processing_time,
                "selected_consultant_ids": selected_consultant_ids,
                "total_frameworks_used": sum(
                    len(cs.stable_frameworks) + len(cs.adaptive_frameworks)
                    for cs in consultant_selections
                ),
            },
            timestamp=datetime.now(),
        )

    async def _classify_query(
        self, query: str, context: Optional[Dict] = None
    ) -> QueryClassification:
        """Classify query to determine optimal consultant selection"""

        # Extract keywords
        keywords = self._extract_keywords(query)

        # Assess complexity (1-5 scale)
        complexity_score = self._assess_complexity(query, keywords)

        # Determine specialization focus
        specialization_focus = self._determine_specialization_focus(
            keywords, complexity_score
        )

        # Find adaptive triggers
        adaptive_triggers = self._identify_adaptive_triggers(keywords)

        # Match routing pattern
        routing_pattern = self._find_best_routing_pattern(keywords, complexity_score)

        return QueryClassification(
            keywords=keywords,
            complexity_score=complexity_score,
            specialization_focus=specialization_focus,
            adaptive_triggers=adaptive_triggers,
            routing_pattern=routing_pattern,
        )

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract relevant keywords from query"""

        # Simple keyword extraction (can be enhanced with NLP)
        query_lower = query.lower()

        # Remove common words and extract meaningful terms
        stopwords = {
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
        }

        words = re.findall(r"\b\w+\b", query_lower)
        keywords = [word for word in words if word not in stopwords and len(word) > 2]

        return keywords

    def _assess_complexity(self, query: str, keywords: List[str]) -> int:
        """Assess query complexity on 1-5 scale"""

        complexity_indicators = {
            "ultra_complex": [
                "comprehensive",
                "holistic",
                "integrated",
                "multi-faceted",
                "complex",
                "transform",
                "strategic",
            ],
            "high_complex": [
                "strategic",
                "innovation",
                "breakthrough",
                "competitive",
                "market",
                "analysis",
            ],
            "medium_complex": [
                "problem",
                "solution",
                "process",
                "improve",
                "optimize",
                "design",
            ],
            "simple": ["quick", "simple", "basic", "straightforward"],
        }

        # Count matches for each complexity level
        ultra_matches = sum(
            1
            for keyword in keywords
            if keyword in complexity_indicators["ultra_complex"]
        )
        high_matches = sum(
            1
            for keyword in keywords
            if keyword in complexity_indicators["high_complex"]
        )
        medium_matches = sum(
            1
            for keyword in keywords
            if keyword in complexity_indicators["medium_complex"]
        )
        simple_matches = sum(
            1 for keyword in keywords if keyword in complexity_indicators["simple"]
        )

        # Query length as complexity factor
        word_count = len(query.split())

        # Determine complexity score
        if ultra_matches >= 2 or word_count > 50:
            return 5
        elif high_matches >= 2 or word_count > 30:
            return 4
        elif medium_matches >= 1 or word_count > 15:
            return 3
        elif simple_matches >= 1:
            return 1
        else:
            return 2  # Default medium-low complexity

    def _determine_specialization_focus(
        self, keywords: List[str], complexity_score: int
    ) -> Tuple[SpecializationRow, SpecializationColumn]:
        """Determine primary specialization focus based on keywords"""

        # Row determination (Strategic/Tactical/Operational)
        strategic_keywords = {
            "strategic",
            "strategy",
            "market",
            "competitive",
            "industry",
            "vision",
            "future",
        }
        tactical_keywords = {
            "solution",
            "approach",
            "method",
            "technique",
            "problem",
            "solve",
            "design",
        }
        operational_keywords = {
            "process",
            "execution",
            "implement",
            "deliver",
            "performance",
            "efficiency",
        }

        strategic_score = sum(1 for kw in keywords if kw in strategic_keywords)
        tactical_score = sum(1 for kw in keywords if kw in tactical_keywords)
        operational_score = sum(1 for kw in keywords if kw in operational_keywords)

        if strategic_score >= max(tactical_score, operational_score):
            row = SpecializationRow.STRATEGIC
        elif tactical_score >= operational_score:
            row = SpecializationRow.TACTICAL
        else:
            row = SpecializationRow.OPERATIONAL

        # Column determination (Analysis/Synthesis/Implementation)
        analysis_keywords = {
            "analysis",
            "analyze",
            "research",
            "investigate",
            "data",
            "insights",
        }
        synthesis_keywords = {
            "integrate",
            "combine",
            "synthesize",
            "connect",
            "framework",
            "design",
        }
        implementation_keywords = {
            "implement",
            "execute",
            "deliver",
            "build",
            "deploy",
            "action",
        }

        analysis_score = sum(1 for kw in keywords if kw in analysis_keywords)
        synthesis_score = sum(1 for kw in keywords if kw in synthesis_keywords)
        implementation_score = sum(
            1 for kw in keywords if kw in implementation_keywords
        )

        if analysis_score >= max(synthesis_score, implementation_score):
            column = SpecializationColumn.ANALYSIS
        elif synthesis_score >= implementation_score:
            column = SpecializationColumn.SYNTHESIS
        else:
            column = SpecializationColumn.IMPLEMENTATION

        return (row, column)

    def _identify_adaptive_triggers(self, keywords: List[str]) -> List[str]:
        """Identify which adaptive triggers should be activated"""

        activated_triggers = []

        for trigger in self.adaptive_triggers:
            # Check if enough keywords match to activate trigger
            matching_keywords = set(keywords) & set(trigger.keywords)
            if len(matching_keywords) >= trigger.activation_threshold:
                activated_triggers.append(trigger.trigger_name)

        return activated_triggers

    def _find_best_routing_pattern(
        self, keywords: List[str], complexity_score: int
    ) -> Optional[str]:
        """Find the best routing pattern for the query"""

        best_pattern = None
        best_score = 0

        for pattern_name, pattern_data in self.routing_patterns.items():
            # Check keyword overlap
            pattern_keywords = set(pattern_data["keywords"])
            query_keywords = set(keywords)
            overlap = len(pattern_keywords & query_keywords)

            # Check complexity match
            complexity_match = (
                abs(pattern_data["complexity_threshold"] - complexity_score) <= 1
            )

            # Calculate score
            score = overlap * pattern_data["success_rate"]
            if complexity_match:
                score *= 1.2  # Boost for complexity match

            if score > best_score:
                best_score = score
                best_pattern = pattern_name

        return best_pattern if best_score > 0 else None

    async def _select_consultant_trio(
        self, classification: QueryClassification
    ) -> List[str]:
        """Select 3 consultants based on classification"""

        # If we have a routing pattern, use it
        if classification.routing_pattern:
            pattern_rule = self.routing_patterns[classification.routing_pattern][
                "selection_rule"
            ]
            return self._apply_routing_rule(pattern_rule, classification)

        # Otherwise, use specialization-based selection
        if classification.complexity_score >= 4:
            # High complexity: select diverse trio (one from each column)
            return self._select_diverse_trio()
        else:
            # Focused selection based on specialization
            return self._select_specialized_trio(classification.specialization_focus)

    def _apply_routing_rule(
        self, rule: Dict, classification: QueryClassification
    ) -> List[str]:
        """Apply specific routing rule"""

        selection_type = rule.get("selection_type", "default")

        if selection_type == "max_diversity":
            return self._select_diverse_trio()
        elif selection_type == "strategic_focused":
            return [rule["primary"], rule["secondary"], rule["tertiary"]]
        elif selection_type == "innovation_focused":
            return [rule["primary"], rule["secondary"], rule["tertiary"]]
        elif selection_type == "operational_focused":
            return [rule["primary"], rule["secondary"], rule["tertiary"]]
        elif selection_type == "diagnostic_focused":
            return [rule["primary"], rule["secondary"], rule["tertiary"]]
        else:
            # Default to specialization-based
            return self._select_specialized_trio(classification.specialization_focus)

    def _select_diverse_trio(self) -> List[str]:
        """Select one consultant from each column for maximum diversity"""

        analysis_consultants = [
            cid
            for cid, cb in self.consultant_blueprints.items()
            if cb.specialization_column == SpecializationColumn.ANALYSIS
        ]
        synthesis_consultants = [
            cid
            for cid, cb in self.consultant_blueprints.items()
            if cb.specialization_column == SpecializationColumn.SYNTHESIS
        ]
        implementation_consultants = [
            cid
            for cid, cb in self.consultant_blueprints.items()
            if cb.specialization_column == SpecializationColumn.IMPLEMENTATION
        ]

        # Select best from each column (can be enhanced with performance metrics)
        return [
            analysis_consultants[0] if analysis_consultants else "problem_solver",
            synthesis_consultants[0] if synthesis_consultants else "solution_architect",
            (
                implementation_consultants[0]
                if implementation_consultants
                else "execution_specialist"
            ),
        ]

    def _select_specialized_trio(
        self, specialization_focus: Tuple[SpecializationRow, SpecializationColumn]
    ) -> List[str]:
        """Select focused trio based on specialization"""

        row, column = specialization_focus

        # Start with the most relevant consultant
        primary = self._get_consultant_by_specialization(row, column)

        # Select complementary consultants
        if column == SpecializationColumn.ANALYSIS:
            # Analysis -> add synthesis and implementation
            secondary = self._get_consultant_by_specialization(
                row, SpecializationColumn.SYNTHESIS
            )
            tertiary = self._get_consultant_by_specialization(
                row, SpecializationColumn.IMPLEMENTATION
            )
        elif column == SpecializationColumn.SYNTHESIS:
            # Synthesis -> add analysis and implementation
            secondary = self._get_consultant_by_specialization(
                row, SpecializationColumn.ANALYSIS
            )
            tertiary = self._get_consultant_by_specialization(
                row, SpecializationColumn.IMPLEMENTATION
            )
        else:
            # Implementation -> add analysis and synthesis
            secondary = self._get_consultant_by_specialization(
                row, SpecializationColumn.ANALYSIS
            )
            tertiary = self._get_consultant_by_specialization(
                row, SpecializationColumn.SYNTHESIS
            )

        return [primary, secondary, tertiary]

    def _get_consultant_by_specialization(
        self, row: SpecializationRow, column: SpecializationColumn
    ) -> str:
        """Get consultant ID by specialization"""

        for consultant_id, blueprint in self.consultant_blueprints.items():
            if (
                blueprint.specialization_row == row
                and blueprint.specialization_column == column
            ):
                return consultant_id

        # Fallback mapping
        fallback_map = {
            (
                SpecializationRow.STRATEGIC,
                SpecializationColumn.ANALYSIS,
            ): "market_analyst",
            (
                SpecializationRow.STRATEGIC,
                SpecializationColumn.SYNTHESIS,
            ): "strategic_synthesizer",
            (
                SpecializationRow.STRATEGIC,
                SpecializationColumn.IMPLEMENTATION,
            ): "strategic_implementer",
            (
                SpecializationRow.TACTICAL,
                SpecializationColumn.ANALYSIS,
            ): "problem_solver",
            (
                SpecializationRow.TACTICAL,
                SpecializationColumn.SYNTHESIS,
            ): "solution_architect",
            (
                SpecializationRow.TACTICAL,
                SpecializationColumn.IMPLEMENTATION,
            ): "tactical_builder",
            (
                SpecializationRow.OPERATIONAL,
                SpecializationColumn.ANALYSIS,
            ): "process_expert",
            (
                SpecializationRow.OPERATIONAL,
                SpecializationColumn.SYNTHESIS,
            ): "operational_integrator",
            (
                SpecializationRow.OPERATIONAL,
                SpecializationColumn.IMPLEMENTATION,
            ): "execution_specialist",
        }

        return fallback_map.get((row, column), "strategic_synthesizer")

    async def _prepare_consultant_selection(
        self, consultant_id: str, classification: QueryClassification
    ) -> ConsultantSelection:
        """Prepare complete consultant selection with frameworks"""

        blueprint = self.consultant_blueprints[consultant_id]

        # Get stable core frameworks (70%)
        stable_frameworks = self._get_stable_core_frameworks(consultant_id)

        # Get adaptive layer frameworks (30%) based on triggers
        adaptive_frameworks = self._get_adaptive_layer_frameworks(
            consultant_id, classification.adaptive_triggers
        )

        # Calculate cognitive load
        total_frameworks = len(stable_frameworks) + len(adaptive_frameworks)
        cognitive_load = min(
            total_frameworks / 8.0, 1.0
        )  # Normalize to 0-1, optimal is ~8 frameworks

        # Generate NWAY-enhanced selection reasoning
        selection_reason = self._generate_nway_enhanced_selection_reasoning(
            blueprint, classification, stable_frameworks, adaptive_frameworks
        )

        return ConsultantSelection(
            consultant_id=consultant_id,
            blueprint=blueprint,
            stable_frameworks=stable_frameworks,
            adaptive_frameworks=adaptive_frameworks,
            total_cognitive_load=cognitive_load,
            selection_reason=selection_reason,
        )

    def _get_stable_core_frameworks(self, consultant_id: str) -> List[Framework]:
        """Get stable core frameworks for consultant (always used)"""

        if consultant_id not in self.framework_assignments:
            return []

        stable_assignments = [
            assignment
            for assignment in self.framework_assignments[consultant_id]
            if assignment.assignment_type == AssignmentType.STABLE_CORE
        ]

        # Sort by priority rank
        stable_assignments.sort(key=lambda x: x.priority_rank)

        return [assignment.framework for assignment in stable_assignments]

    def _get_adaptive_layer_frameworks(
        self, consultant_id: str, adaptive_triggers: List[str]
    ) -> List[Framework]:
        """Get adaptive layer frameworks based on triggers"""

        adaptive_frameworks = []

        for trigger in self.adaptive_triggers:
            if (
                trigger.trigger_name in adaptive_triggers
                and consultant_id in trigger.consultant_affinity
            ):

                adaptive_frameworks.append(trigger.framework)

        # Limit to prevent cognitive overload (max 3-4 adaptive frameworks)
        adaptive_frameworks.sort(key=lambda f: f.effectiveness_score, reverse=True)
        return adaptive_frameworks[:4]

    def _generate_selection_reasoning(
        self,
        blueprint: ConsultantBlueprint,
        classification: QueryClassification,
        stable_frameworks: List[Framework],
        adaptive_frameworks: List[Framework],
    ) -> str:
        """Generate human-readable reasoning for consultant selection"""

        reasons = [
            f"Selected {blueprint.name} for {blueprint.specialization_row.value} {blueprint.specialization_column.value} expertise"
        ]

        if stable_frameworks:
            framework_names = [f.name for f in stable_frameworks[:3]]  # Show top 3
            reasons.append(f"Core frameworks: {', '.join(framework_names)}")

        if adaptive_frameworks:
            framework_names = [f.name for f in adaptive_frameworks[:2]]  # Show top 2
            reasons.append(f"Adaptive enhancements: {', '.join(framework_names)}")

        if classification.complexity_score >= 4:
            reasons.append(
                f"High complexity ({classification.complexity_score}/5) requires specialized expertise"
            )

        return ". ".join(reasons)

    def _generate_nway_enhanced_selection_reasoning(
        self,
        blueprint: ConsultantBlueprint,
        classification: QueryClassification,
        stable_frameworks: List[Framework],
        adaptive_frameworks: List[Framework],
    ) -> str:
        """Generate NWAY-enhanced reasoning for consultant selection with 70/30 integration"""

        reasons = [
            f"Selected {blueprint.name} for {blueprint.specialization_row.value} {blueprint.specialization_column.value} expertise"
        ]

        # 70/30 Split Explanation
        reasons.append(
            f"70/30 Architecture: {blueprint.stable_core_weight*100:.0f}% stable core + {blueprint.adaptive_layer_weight*100:.0f}% adaptive NWAY layer"
        )

        # Stable Core (70%)
        if stable_frameworks:
            framework_names = [f.name for f in stable_frameworks[:3]]  # Show top 3
            reasons.append(
                f"Stable Core Frameworks (70%): {', '.join(framework_names)}"
            )

        # Adaptive Layer (30%) with NWAY integration
        if adaptive_frameworks:
            framework_names = [f.name for f in adaptive_frameworks[:2]]  # Show top 2
            reasons.append(f"NWAY Adaptive Layer (30%): {', '.join(framework_names)}")

        # NWAY Cluster Integration
        nway_triggers = (
            len(classification.adaptive_triggers)
            if hasattr(classification, "adaptive_triggers")
            else 0
        )
        if nway_triggers > 0:
            reasons.append(
                f"NWAY Clusters Activated: {nway_triggers} sophisticated interaction patterns"
            )

        # Complexity and UltraThink considerations
        if classification.complexity_score >= 4:
            reasons.append(
                f"UltraThink Mode: High complexity ({classification.complexity_score}/5) activates extended reasoning time"
            )

        # Cognitive synergies
        total_frameworks = len(stable_frameworks) + len(adaptive_frameworks)
        if total_frameworks >= 5:
            reasons.append(
                f"Cognitive Synergies: {total_frameworks} frameworks enable cross-disciplinary insights"
            )

        return ". ".join(reasons)

    async def _log_engagement(
        self,
        query: str,
        classification: QueryClassification,
        selections: List[ConsultantSelection],
    ):
        """Log engagement for benchmarking and optimization"""

        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]

        frameworks_applied = {}
        for selection in selections:
            frameworks_applied[selection.consultant_id] = {
                "stable_core": [f.name for f in selection.stable_frameworks],
                "adaptive_layer": [f.name for f in selection.adaptive_frameworks],
                "total_count": len(selection.stable_frameworks)
                + len(selection.adaptive_frameworks),
            }

        engagement_data = {
            "query_text": query,
            "query_hash": query_hash,
            "query_classification": {
                "keywords": classification.keywords,
                "complexity_score": classification.complexity_score,
                "specialization_focus": [
                    classification.specialization_focus[0].value,
                    classification.specialization_focus[1].value,
                ],
                "adaptive_triggers": classification.adaptive_triggers,
                "routing_pattern": classification.routing_pattern,
            },
            "selected_consultants": [s.consultant_id for s in selections],
            "routing_pattern_used": classification.routing_pattern,
            "frameworks_applied": frameworks_applied,
            "selection_reasoning": " | ".join([s.selection_reason for s in selections]),
        }

        await self.supabase.table("engagement_selections").insert(
            engagement_data
        ).execute()


# ===============================================================
# USAGE EXAMPLE
# ===============================================================


async def main():
    """Example usage of the OptimalConsultantSystem"""

    # Initialize system (would use real Supabase client)
    from supabase import create_client

    supabase_client = create_client("your_url", "your_key")

    system = OptimalConsultantSystem(supabase_client)

    # Wait for system to load
    await asyncio.sleep(2)

    # Process a strategic query
    strategic_query = "How can we develop a competitive market entry strategy for the European fintech sector?"
    result = await system.process_query(strategic_query)

    print("=== STRATEGIC QUERY RESULT ===")
    print(f"Query: {result.query}")
    print(
        f"Selected Consultants: {[s.consultant_id for s in result.selected_consultants]}"
    )

    for selection in result.selected_consultants:
        print(f"\n{selection.blueprint.name}:")
        print(
            f"  Specialization: {selection.blueprint.specialization_row.value} × {selection.blueprint.specialization_column.value}"
        )
        print(f"  Stable Frameworks: {[f.name for f in selection.stable_frameworks]}")
        print(
            f"  Adaptive Frameworks: {[f.name for f in selection.adaptive_frameworks]}"
        )
        print(f"  Selection Reason: {selection.selection_reason}")

    # Process an operational query
    operational_query = (
        "How can we optimize our customer service process to reduce response times?"
    )
    result2 = await system.process_query(operational_query)

    print("\n=== OPERATIONAL QUERY RESULT ===")
    print(f"Query: {result2.query}")
    print(
        f"Selected Consultants: {[s.consultant_id for s in result2.selected_consultants]}"
    )


if __name__ == "__main__":
    asyncio.run(main())
