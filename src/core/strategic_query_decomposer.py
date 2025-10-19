"""
Strategic Query Decomposer - Next-Generation Query Chunking Engine
================================================================

Research-validated strategic decomposition based on MECE principles, first-principles reasoning,
and decision theory. Transforms queries from "keyword sorting" to "strategic understanding".

Key Innovations:
1. MECE Decomposition (Mutually Exclusive, Collectively Exhaustive)
2. First-Principles Constraint Separation (Hard constraints vs conventions)
3. Natural Boundary Detection (Causal seams, interfaces, timescales)
4. Unknowns as First-Class Citizens (Hypotheses, experiments, premortems)
5. Optionality Preservation (Reversible vs irreversible decisions)

Based on principles from decision theory, cognitive science, and strategic thinking research.
"""

import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4
import logging

# Core integrations
from src.core.unified_context_stream import UnifiedContextStream, ContextEventType
from src.integrations.llm.unified_client import UnifiedLLMClient

logger = logging.getLogger(__name__)


def _safe_json_parse(response_text: str, fallback_value: dict = None) -> dict:
    """Safely parse JSON with multiple fallback strategies"""
    if fallback_value is None:
        fallback_value = {}

    if not response_text or not response_text.strip():
        return fallback_value

    # Try direct parsing first
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown blocks

    json_blocks = re.findall(
        r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL
    )
    for block in json_blocks:
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue

    # Try to find JSON-like structures
    json_matches = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response_text)
    for match in json_matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    # Last resort: return fallback
    logger.warning(f"Failed to parse JSON from response: {response_text[:200]}...")
    return fallback_value


def _map_constraint_type(text: str) -> "ConstraintType":
    """Map descriptive text to ConstraintType enum"""
    text_lower = text.lower()

    if any(
        word in text_lower
        for word in ["legal", "regulatory", "compliance", "law", "regulation"]
    ):
        return ConstraintType.LEGAL
    elif any(
        word in text_lower
        for word in ["physical", "physics", "thermodynamic", "natural"]
    ):
        return ConstraintType.PHYSICAL
    elif any(
        word in text_lower
        for word in ["economic", "financial", "cost", "budget", "resource"]
    ):
        return ConstraintType.ECONOMIC
    elif any(
        word in text_lower
        for word in ["logical", "mathematical", "logical consistency"]
    ):
        return ConstraintType.LOGICAL
    elif any(
        word in text_lower for word in ["temporal", "time", "timeline", "deadline"]
    ):
        return ConstraintType.TEMPORAL
    else:
        return ConstraintType.LOGICAL  # Default fallback


def _map_convention_type(text: str) -> "ConventionType":
    """Map descriptive text to ConventionType enum"""
    text_lower = text.lower()

    if any(word in text_lower for word in ["policy", "policies"]):
        return ConventionType.POLICY
    elif any(word in text_lower for word in ["cultural", "culture", "norm", "habit"]):
        return ConventionType.CULTURAL
    elif any(
        word in text_lower
        for word in ["procedural", "procedure", "process", "operating"]
    ):
        return ConventionType.PROCEDURAL
    elif any(word in text_lower for word in ["strategic", "strategy", "positioning"]):
        return ConventionType.STRATEGIC
    elif any(word in text_lower for word in ["operational", "operation", "preference"]):
        return ConventionType.OPERATIONAL
    else:
        return ConventionType.PROCEDURAL  # Default fallback


class ConstraintType(Enum):
    """Types of hard constraints (unfalsifiable facts)"""

    PHYSICAL = "physical"  # Laws of physics, thermodynamics
    ECONOMIC = "economic"  # Supply/demand, resource limitations
    LOGICAL = "logical"  # Mathematical, logical consistency
    LEGAL = "legal"  # Regulatory requirements, compliance
    TEMPORAL = "temporal"  # Time-based constraints


class ConventionType(Enum):
    """Types of conventions (changeable assumptions)"""

    POLICY = "policy"  # Organizational policies
    CULTURAL = "cultural"  # Cultural norms, habits
    PROCEDURAL = "procedural"  # Standard operating procedures
    STRATEGIC = "strategic"  # Strategic choices, positioning
    OPERATIONAL = "operational"  # Operational preferences


class DecisionReversibility(Enum):
    """Decision reversibility classification"""

    REVERSIBLE = "reversible"  # Two-way door, low cost to undo
    IRREVERSIBLE = "irreversible"  # One-way door, high cost to undo
    PARTIALLY_REVERSIBLE = "partially_reversible"  # Some aspects reversible


class BoundaryType(Enum):
    """Types of natural chunk boundaries"""

    CAUSAL = "causal"  # Cause-effect relationship changes
    INTERFACE = "interface"  # System component interactions
    TEMPORAL = "temporal"  # Different timescales
    STAKEHOLDER = "stakeholder"  # Responsibility/ownership changes
    DATA_SCHEMA = "data_schema"  # Information structure changes
    DECISION = "decision"  # Decision point boundaries


@dataclass
class HardConstraint:
    """Represents an unfalsifiable, unchangeable constraint"""

    description: str
    constraint_type: ConstraintType
    falsifiability_test: str
    evidence_basis: str
    confidence: float = 0.9

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "type": self.constraint_type.value,
            "falsifiability_test": self.falsifiability_test,
            "evidence_basis": self.evidence_basis,
            "confidence": self.confidence,
        }


@dataclass
class Convention:
    """Represents a changeable assumption or convention"""

    description: str
    convention_type: ConventionType
    changeability_assessment: str
    change_cost: str  # Low, Medium, High
    change_timeline: str
    stakeholders_affected: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "type": self.convention_type.value,
            "changeability_assessment": self.changeability_assessment,
            "change_cost": self.change_cost,
            "change_timeline": self.change_timeline,
            "stakeholders_affected": self.stakeholders_affected,
        }


@dataclass
class DecisionPoint:
    """Represents a decision that needs to be made"""

    description: str
    reversibility: DecisionReversibility
    decision_timeline: str
    information_requirements: List[str]
    stakeholders: List[str]
    consequences_if_delayed: str
    options: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "reversibility": self.reversibility.value,
            "timeline": self.decision_timeline,
            "information_requirements": self.information_requirements,
            "stakeholders": self.stakeholders,
            "consequences_if_delayed": self.consequences_if_delayed,
            "options": self.options,
        }


@dataclass
class Unknown:
    """Represents something we don't know (known unknown or unknown unknown)"""

    description: str
    unknown_type: str  # "known_unknown", "unknown_unknown", "assumption"
    impact_if_wrong: str
    learning_approach: str
    experiment_design: Optional[str] = None
    kill_criteria: Optional[str] = None
    learning_timeline: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "type": self.unknown_type,
            "impact_if_wrong": self.impact_if_wrong,
            "learning_approach": self.learning_approach,
            "experiment_design": self.experiment_design,
            "kill_criteria": self.kill_criteria,
            "learning_timeline": self.learning_timeline,
        }


@dataclass
class SuccessMetric:
    """Represents how success will be measured"""

    description: str
    measurement_approach: str
    target_value: Optional[str] = None
    timeline: Optional[str] = None
    leading_indicators: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "measurement_approach": self.measurement_approach,
            "target_value": self.target_value,
            "timeline": self.timeline,
            "leading_indicators": self.leading_indicators,
        }


@dataclass
class ChunkBoundary:
    """Represents a natural boundary between chunks"""

    boundary_type: BoundaryType
    description: str
    strength: float  # 0.0-1.0, how strong the boundary signal is
    rationale: str
    before_chunk: str
    after_chunk: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": boundary_type.value,
            "description": self.description,
            "strength": self.strength,
            "rationale": self.rationale,
            "before_chunk": self.before_chunk,
            "after_chunk": self.after_chunk,
        }


@dataclass
class MECEDecomposition:
    """Complete MECE decomposition of a query"""

    query_id: str = field(default_factory=lambda: str(uuid4()))
    original_query: str = ""

    # MECE components
    constraints: List[HardConstraint] = field(default_factory=list)
    conventions: List[Convention] = field(default_factory=list)
    decisions: List[DecisionPoint] = field(default_factory=list)
    unknowns: List[Unknown] = field(default_factory=list)
    success_metrics: List[SuccessMetric] = field(default_factory=list)

    # Quality metrics
    completeness_score: float = 0.0
    separability_score: float = 0.0
    boundaries: List[ChunkBoundary] = field(default_factory=list)

    # Metadata
    processing_time_ms: int = 0
    confidence_score: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "original_query": self.original_query,
            "constraints": [c.to_dict() for c in self.constraints],
            "conventions": [c.to_dict() for c in self.conventions],
            "decisions": [d.to_dict() for d in self.decisions],
            "unknowns": [u.to_dict() for u in self.unknowns],
            "success_metrics": [s.to_dict() for s in self.success_metrics],
            "completeness_score": self.completeness_score,
            "separability_score": self.separability_score,
            "boundaries": [b.to_dict() for b in self.boundaries],
            "processing_time_ms": self.processing_time_ms,
            "confidence_score": self.confidence_score,
            "timestamp": self.timestamp.isoformat(),
        }


class StrategicQueryDecomposer:
    """
    Next-generation query decomposer using research-validated strategic principles.

    Transforms queries from keyword matching to strategic understanding through:
    1. MECE decomposition ensuring mutual exclusivity and collective exhaustiveness
    2. First-principles separation of hard constraints from changeable conventions
    3. Natural boundary detection using systematic signals
    4. Treatment of unknowns as first-class citizens requiring targeted learning
    """

    def __init__(self):
        self.llm_client = UnifiedLLMClient()
        self.context_stream: Optional[UnifiedContextStream] = None

        # Initialize first-principles patterns
        self._initialize_constraint_patterns()
        self._initialize_boundary_patterns()

        logger.info(
            "🧠 Strategic Query Decomposer initialized with research-validated principles"
        )

    def _initialize_constraint_patterns(self):
        """Initialize patterns for identifying hard constraints vs conventions"""

        # Physical constraint indicators
        self.physical_indicators = [
            "thermodynamics",
            "conservation",
            "entropy",
            "energy",
            "mass",
            "speed of light",
            "gravity",
            "friction",
            "temperature",
            "pressure",
            "physical laws",
            "physics",
        ]

        # Economic constraint indicators
        self.economic_indicators = [
            "supply and demand",
            "scarcity",
            "opportunity cost",
            "marginal utility",
            "diminishing returns",
            "economic laws",
            "market forces",
            "resource limitations",
        ]

        # Logical constraint indicators
        self.logical_indicators = [
            "mathematical",
            "logical consistency",
            "contradiction",
            "proof",
            "theorem",
            "axiom",
            "deduction",
            "inference",
            "logical impossibility",
        ]

        # Convention indicators (changeable)
        self.convention_indicators = [
            "policy",
            "best practice",
            "standard procedure",
            "traditional approach",
            "cultural norm",
            "habit",
            "preference",
            "strategic choice",
            "convention",
        ]

    def _initialize_boundary_patterns(self):
        """Initialize patterns for detecting natural chunk boundaries"""

        # Causal boundary indicators
        self.causal_indicators = [
            "because",
            "therefore",
            "results in",
            "leads to",
            "causes",
            "due to",
            "impact",
            "effect",
            "consequence",
            "influences",
            "drives",
            "triggers",
        ]

        # Temporal boundary indicators
        self.temporal_indicators = [
            "before",
            "after",
            "during",
            "while",
            "then",
            "next",
            "later",
            "short-term",
            "long-term",
            "immediate",
            "future",
            "timeline",
            "phase",
        ]

        # Stakeholder boundary indicators
        self.stakeholder_indicators = [
            "responsible",
            "owns",
            "manages",
            "decides",
            "approves",
            "executes",
            "team",
            "department",
            "organization",
            "customer",
            "vendor",
            "partner",
        ]

    async def decompose_query(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> MECEDecomposition:
        """
        Main entry point: Decompose a query using MECE principles and first-principles reasoning.

        Args:
            query: The query to decompose
            context: Optional context information

        Returns:
            Complete MECE decomposition with quality metrics
        """
        start_time = datetime.now()

        try:
            # Step 1: Extract hard constraints vs conventions
            constraints, conventions = await self._separate_constraints_and_conventions(
                query
            )

            # Step 2: Identify decision points with reversibility classification
            decisions = await self._extract_decision_points(query)

            # Step 3: Surface unknowns and assumptions
            unknowns = await self._extract_unknowns(query)

            # Step 4: Define success metrics
            success_metrics = await self._extract_success_metrics(query)

            # Step 5: Detect natural boundaries
            boundaries = await self._detect_natural_boundaries(
                query,
                {
                    "constraints": constraints,
                    "conventions": conventions,
                    "decisions": decisions,
                    "unknowns": unknowns,
                },
            )

            # Step 6: Assess MECE quality
            completeness_score = self._assess_completeness(
                constraints, conventions, decisions, unknowns, success_metrics
            )
            separability_score = self._assess_separability(
                constraints, conventions, decisions, unknowns
            )

            # Step 7: Calculate overall confidence
            confidence_score = self._calculate_confidence(
                completeness_score, separability_score, len(boundaries)
            )

            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)

            decomposition = MECEDecomposition(
                original_query=query,
                constraints=constraints,
                conventions=conventions,
                decisions=decisions,
                unknowns=unknowns,
                success_metrics=success_metrics,
                completeness_score=completeness_score,
                separability_score=separability_score,
                boundaries=boundaries,
                processing_time_ms=processing_time,
                confidence_score=confidence_score,
            )

            # Log to context stream if available
            if self.context_stream:
                self.context_stream.add_event(
                    ContextEventType.QUERY_RECEIVED,
                    {
                        "query": query,
                        "decomposition_summary": {
                            "constraints_count": len(constraints),
                            "conventions_count": len(conventions),
                            "decisions_count": len(decisions),
                            "unknowns_count": len(unknowns),
                            "completeness_score": completeness_score,
                            "separability_score": separability_score,
                            "confidence_score": confidence_score,
                        },
                    },
                    {
                        "decomposition_approach": "strategic_mece",
                        "processing_time_ms": processing_time,
                    },
                )

            logger.info(
                f"🎯 Query decomposed: {len(constraints)} constraints, {len(conventions)} conventions, "
                f"{len(decisions)} decisions, {len(unknowns)} unknowns (confidence: {confidence_score:.2f})"
            )

            return decomposition

        except Exception as e:
            logger.error(f"❌ Query decomposition failed: {e}")
            raise

    async def _separate_constraints_and_conventions(
        self, query: str
    ) -> Tuple[List[HardConstraint], List[Convention]]:
        """
        Use first-principles reasoning to separate hard constraints from conventions.

        Hard constraints: Physics, economics, logic - unfalsifiable facts
        Conventions: Policies, habits, preferences - changeable assumptions
        """

        prompt = f"""
        Apply first-principles reasoning to separate hard constraints from conventions in this query:
        
        QUERY: {query}
        
        HARD CONSTRAINTS are unfalsifiable facts based on:
        - Physical laws (thermodynamics, conservation, gravity)
        - Economic laws (supply/demand, scarcity, opportunity cost)  
        - Logical requirements (mathematical consistency, logical impossibility)
        - Legal requirements (regulatory mandates, compliance)
        
        CONVENTIONS are changeable assumptions based on:
        - Organizational policies
        - Cultural norms and habits
        - Strategic choices and preferences
        - Standard procedures
        
        For each constraint/convention identified, provide:
        1. Description
        2. Type classification
        3. Reasoning for classification
        4. Evidence basis or changeability assessment
        
        Return as structured JSON with "hard_constraints" and "conventions" arrays.
        """

        try:
            response = await self.llm_client.call_llm(
                [
                    {
                        "role": "system",
                        "content": "You are an expert in first-principles reasoning and strategic analysis.",
                    },
                    {"role": "user", "content": prompt},
                ]
            )
            response = response.content

            result = _safe_json_parse(
                response, {"hard_constraints": [], "conventions": []}
            )

            # Parse hard constraints
            constraints = []
            for item in result.get("hard_constraints", []):
                constraint_type = _map_constraint_type(item.get("type", "logical"))
                constraint = HardConstraint(
                    description=item.get("description", ""),
                    constraint_type=constraint_type,
                    falsifiability_test=item.get("falsifiability_test", ""),
                    evidence_basis=item.get("evidence_basis", ""),
                    confidence=item.get("confidence", 0.8),
                )
                constraints.append(constraint)

            # Parse conventions
            conventions = []
            for item in result.get("conventions", []):
                convention_type = _map_convention_type(item.get("type", "procedural"))
                convention = Convention(
                    description=item.get("description", ""),
                    convention_type=convention_type,
                    changeability_assessment=item.get("changeability_assessment", ""),
                    change_cost=item.get("change_cost", "Medium"),
                    change_timeline=item.get("change_timeline", ""),
                    stakeholders_affected=item.get("stakeholders_affected", []),
                )
                conventions.append(convention)

            return constraints, conventions

        except Exception as e:
            logger.error(f"❌ Constraint separation failed: {e}")
            return [], []

    async def _extract_decision_points(self, query: str) -> List[DecisionPoint]:
        """
        Extract decision points and classify by reversibility (ASAP vs ALAP principle).

        Reversible decisions (two-way doors): Make ASAP to gather information through action
        Irreversible decisions (one-way doors): Make ALAP after gathering maximum information
        """

        prompt = f"""
        Identify all decision points in this query and classify by reversibility:
        
        QUERY: {query}
        
        REVERSIBLE DECISIONS (Two-way doors):
        - Low cost to undo or change
        - Can be reversed with minimal consequences
        - Action provides valuable learning
        - Should be made ASAP (As Soon As Possible)
        
        IRREVERSIBLE DECISIONS (One-way doors):
        - High cost to undo or change  
        - Significant consequences if wrong
        - Require maximum information gathering
        - Should be made ALAP (As Late As Possible)
        
        For each decision point, provide:
        1. Description of the decision
        2. Reversibility classification
        3. Timeline for decision
        4. Information requirements
        5. Key stakeholders
        6. Consequences if delayed
        7. Available options
        
        Return as structured JSON with "decisions" array.
        """

        try:
            response = await self.llm_client.call_llm(
                [
                    {
                        "role": "system",
                        "content": "You are an expert in decision theory and strategic planning.",
                    },
                    {"role": "user", "content": prompt},
                ]
            )
            response = response.content

            result = _safe_json_parse(response, {"decisions": []})

            decisions = []
            for item in result.get("decisions", []):
                reversibility = DecisionReversibility(
                    item.get("reversibility", "partially_reversible")
                )
                decision = DecisionPoint(
                    description=item.get("description", ""),
                    reversibility=reversibility,
                    decision_timeline=item.get("timeline", ""),
                    information_requirements=item.get("information_requirements", []),
                    stakeholders=item.get("stakeholders", []),
                    consequences_if_delayed=item.get("consequences_if_delayed", ""),
                    options=item.get("options", []),
                )
                decisions.append(decision)

            return decisions

        except Exception as e:
            logger.error(f"❌ Decision point extraction failed: {e}")
            return []

    async def _extract_unknowns(self, query: str) -> List[Unknown]:
        """
        Surface unknowns using the Rumsfeld Matrix and convert them to testable hypotheses.

        Known Unknowns: Things we know we don't know
        Unknown Unknowns: Things we don't know we don't know
        Assumptions: Things we're assuming but haven't validated
        """

        prompt = f"""
        Apply the Rumsfeld Matrix to identify unknowns and assumptions in this query:
        
        QUERY: {query}
        
        KNOWN UNKNOWNS: Things we know we don't know
        - Explicit uncertainties and gaps
        - Questions that need answers
        - Missing data or information
        
        UNKNOWN UNKNOWNS: Things we don't know we don't know
        - Hidden risks and blind spots
        - Systemic vulnerabilities  
        - Unquestioned assumptions
        
        ASSUMPTIONS: Things we're assuming but haven't validated
        - Implicit beliefs and premises
        - Unverified hypotheses
        - Taken-for-granted conditions
        
        For each unknown/assumption, provide:
        1. Description of what we don't know
        2. Type (known_unknown, unknown_unknown, assumption)
        3. Impact if our assumption is wrong
        4. Approach for learning/testing
        5. Experiment design (if applicable)
        6. Kill criteria (when to abandon)
        7. Learning timeline
        
        Return as structured JSON with "unknowns" array.
        """

        try:
            response = await self.llm_client.call_llm(
                [
                    {
                        "role": "system",
                        "content": "You are an expert in risk analysis and systematic thinking.",
                    },
                    {"role": "user", "content": prompt},
                ]
            )
            response = response.content

            result = _safe_json_parse(response, {"unknowns": []})

            unknowns = []
            for item in result.get("unknowns", []):
                unknown = Unknown(
                    description=item.get("description", ""),
                    unknown_type=item.get("type", "assumption"),
                    impact_if_wrong=item.get("impact_if_wrong", ""),
                    learning_approach=item.get("learning_approach", ""),
                    experiment_design=item.get("experiment_design"),
                    kill_criteria=item.get("kill_criteria"),
                    learning_timeline=item.get("learning_timeline"),
                )
                unknowns.append(unknown)

            return unknowns

        except Exception as e:
            logger.error(f"❌ Unknown extraction failed: {e}")
            return []

    async def _extract_success_metrics(self, query: str) -> List[SuccessMetric]:
        """Extract how success will be measured and defined."""

        prompt = f"""
        Define success metrics for this query using SMART criteria:
        
        QUERY: {query}
        
        Success metrics should be:
        - Specific: Clear and well-defined
        - Measurable: Quantifiable or verifiable  
        - Achievable: Realistic given constraints
        - Relevant: Aligned with the goal
        - Time-bound: Has a deadline or timeline
        
        For each success metric, provide:
        1. Description of what success looks like
        2. How it will be measured
        3. Target value (if quantifiable)
        4. Timeline for achievement
        5. Leading indicators (early signals)
        
        Return as structured JSON with "success_metrics" array.
        """

        try:
            response = await self.llm_client.call_llm(
                [
                    {
                        "role": "system",
                        "content": "You are an expert in performance measurement and goal setting.",
                    },
                    {"role": "user", "content": prompt},
                ]
            )
            response = response.content

            result = _safe_json_parse(response, {"success_metrics": []})

            metrics = []
            for item in result.get("success_metrics", []):
                metric = SuccessMetric(
                    description=item.get("description", ""),
                    measurement_approach=item.get("measurement_approach", ""),
                    target_value=item.get("target_value"),
                    timeline=item.get("timeline"),
                    leading_indicators=item.get("leading_indicators", []),
                )
                metrics.append(metric)

            return metrics

        except Exception as e:
            logger.error(f"❌ Success metrics extraction failed: {e}")
            return []

    async def _detect_natural_boundaries(
        self, query: str, components: Dict[str, Any]
    ) -> List[ChunkBoundary]:
        """
        Detect natural boundaries using systematic signals:
        - Causal seams (cause-effect relationship changes)
        - Interface boundaries (system component interactions)
        - Temporal boundaries (different timescales)
        - Stakeholder boundaries (responsibility changes)
        - Data schema boundaries (information structure changes)
        """

        # Use text analysis to detect boundary signals
        boundaries = []

        # Detect causal boundaries
        causal_boundaries = self._detect_causal_boundaries(query)
        boundaries.extend(causal_boundaries)

        # Detect temporal boundaries
        temporal_boundaries = self._detect_temporal_boundaries(query)
        boundaries.extend(temporal_boundaries)

        # Detect stakeholder boundaries
        stakeholder_boundaries = self._detect_stakeholder_boundaries(query)
        boundaries.extend(stakeholder_boundaries)

        return boundaries

    def _detect_causal_boundaries(self, query: str) -> List[ChunkBoundary]:
        """Detect boundaries where cause-effect relationships change"""
        boundaries = []

        # Find causal indicators
        for indicator in self.causal_indicators:
            if indicator.lower() in query.lower():
                # Find position and context
                position = query.lower().find(indicator.lower())
                context_before = query[max(0, position - 50) : position].strip()
                context_after = query[
                    position + len(indicator) : position + len(indicator) + 50
                ].strip()

                boundary = ChunkBoundary(
                    boundary_type=BoundaryType.CAUSAL,
                    description=f"Causal relationship indicated by '{indicator}'",
                    strength=0.7,  # Could be made more sophisticated
                    rationale=f"Causal indicator '{indicator}' suggests cause-effect boundary",
                    before_chunk=context_before,
                    after_chunk=context_after,
                )
                boundaries.append(boundary)

        return boundaries

    def _detect_temporal_boundaries(self, query: str) -> List[ChunkBoundary]:
        """Detect boundaries based on different timescales"""
        boundaries = []

        for indicator in self.temporal_indicators:
            if indicator.lower() in query.lower():
                position = query.lower().find(indicator.lower())
                context_before = query[max(0, position - 50) : position].strip()
                context_after = query[
                    position + len(indicator) : position + len(indicator) + 50
                ].strip()

                boundary = ChunkBoundary(
                    boundary_type=BoundaryType.TEMPORAL,
                    description=f"Temporal boundary indicated by '{indicator}'",
                    strength=0.6,
                    rationale=f"Temporal indicator '{indicator}' suggests time-based boundary",
                    before_chunk=context_before,
                    after_chunk=context_after,
                )
                boundaries.append(boundary)

        return boundaries

    def _detect_stakeholder_boundaries(self, query: str) -> List[ChunkBoundary]:
        """Detect boundaries based on stakeholder responsibility changes"""
        boundaries = []

        for indicator in self.stakeholder_indicators:
            if indicator.lower() in query.lower():
                position = query.lower().find(indicator.lower())
                context_before = query[max(0, position - 50) : position].strip()
                context_after = query[
                    position + len(indicator) : position + len(indicator) + 50
                ].strip()

                boundary = ChunkBoundary(
                    boundary_type=BoundaryType.STAKEHOLDER,
                    description=f"Stakeholder boundary indicated by '{indicator}'",
                    strength=0.5,
                    rationale=f"Stakeholder indicator '{indicator}' suggests responsibility boundary",
                    before_chunk=context_before,
                    after_chunk=context_after,
                )
                boundaries.append(boundary)

        return boundaries

    def _assess_completeness(
        self,
        constraints: List[HardConstraint],
        conventions: List[Convention],
        decisions: List[DecisionPoint],
        unknowns: List[Unknown],
        success_metrics: List[SuccessMetric],
    ) -> float:
        """
        Assess collective exhaustiveness - do we have all necessary components?

        Based on the principle that good decomposition should be collectively exhaustive.
        """
        component_scores = []

        # Score based on presence of each MECE component
        component_scores.append(
            1.0 if constraints else 0.5
        )  # Always expect some constraints
        component_scores.append(
            1.0 if conventions else 0.8
        )  # Usually expect conventions
        component_scores.append(1.0 if decisions else 0.3)  # Decisions are critical
        component_scores.append(1.0 if unknowns else 0.6)  # Usually expect unknowns
        component_scores.append(
            1.0 if success_metrics else 0.4
        )  # Success metrics are important

        return sum(component_scores) / len(component_scores)

    def _assess_separability(
        self,
        constraints: List[HardConstraint],
        conventions: List[Convention],
        decisions: List[DecisionPoint],
        unknowns: List[Unknown],
    ) -> float:
        """
        Assess mutual exclusivity - are components properly separated?

        Based on the principle that good decomposition should have mutually exclusive parts.
        """
        # Check for overlap in descriptions (simple text similarity)
        all_descriptions = []
        all_descriptions.extend([c.description for c in constraints])
        all_descriptions.extend([c.description for c in conventions])
        all_descriptions.extend([d.description for d in decisions])
        all_descriptions.extend([u.description for u in unknowns])

        if len(all_descriptions) <= 1:
            return 1.0

        # Simple overlap detection based on shared keywords
        overlap_count = 0
        total_pairs = 0

        for i in range(len(all_descriptions)):
            for j in range(i + 1, len(all_descriptions)):
                total_pairs += 1
                desc1_words = set(all_descriptions[i].lower().split())
                desc2_words = set(all_descriptions[j].lower().split())

                # If more than 30% word overlap, consider it overlapping
                overlap = len(desc1_words.intersection(desc2_words))
                if overlap > 0.3 * min(len(desc1_words), len(desc2_words)):
                    overlap_count += 1

        if total_pairs == 0:
            return 1.0

        return 1.0 - (overlap_count / total_pairs)

    def _calculate_confidence(
        self, completeness: float, separability: float, boundary_count: int
    ) -> float:
        """Calculate overall confidence in the decomposition quality"""

        # Base confidence from MECE scores
        base_confidence = (completeness + separability) / 2

        # Adjust for boundary detection quality
        boundary_factor = min(
            1.0, boundary_count / 3.0
        )  # Expect at least 3 boundaries for high confidence

        # Combined confidence
        confidence = (base_confidence * 0.8) + (boundary_factor * 0.2)

        return min(1.0, confidence)

    def set_context_stream(self, context_stream: UnifiedContextStream):
        """Set the context stream for logging decomposition events"""
        self.context_stream = context_stream


# Global instance for the application
_strategic_decomposer: Optional[StrategicQueryDecomposer] = None


def get_strategic_query_decomposer() -> StrategicQueryDecomposer:
    """Get or create the global strategic query decomposer instance"""
    global _strategic_decomposer
    if _strategic_decomposer is None:
        _strategic_decomposer = StrategicQueryDecomposer()
    return _strategic_decomposer
