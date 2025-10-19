"""
Pyramid Principle Synthesis Engine
=================================

Implementation of Minto's Pyramid Principle for elite consulting communication.
Transforms analysis and findings into McKinsey-style structured synthesis.

Core Principles:
- Vertical Logic: Question/Answer dialogue (deduction/induction)
- Horizontal Logic: MECE at every level
- Top-Down: Start with Governing Thought
- Bottom-Up: Cluster findings into implications

Integration Points:
- Senior Advisor synthesis enhancement
- Glass-Box transparency with logical structure
- Method Actor narrative templates
"""

import asyncio
import yaml
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LogicalStructure(str, Enum):
    """Types of logical structures in Pyramid Principle"""

    DEDUCTIVE = "deductive"  # General rules to specific conclusions
    INDUCTIVE = "inductive"  # Specific observations to general conclusion
    GROUPING = "grouping"  # Independent MECE groups supporting conclusion
    ARGUMENT = "argument"  # Sequential chain of reasoning


class NarrativePattern(str, Enum):
    """Narrative patterns for different contexts"""

    SCQA = "scqa"  # Situation-Complication-Question-Answer
    DIRECT = "direct"  # Answer-Situation-Complication
    STANDARD = "standard"  # Situation-Complication-Answer
    HOW_TO = "how_to"  # Process/implementation focus
    ARGUMENT = "argument"  # Skeptical audience, step-by-step revelation


@dataclass
class PyramidElement:
    """Individual element in the pyramid structure"""

    content: str
    level: int
    element_type: str  # "governing_thought", "key_line", "supporting_fact"
    logical_relationship: str  # How it relates to parent element
    mece_group: Optional[str] = None
    evidence_strength: float = 0.0
    children: List["PyramidElement"] = field(default_factory=list)


@dataclass
class GoverningThought:
    """Core message/conclusion of the pyramid"""

    message: str
    answer_type: str  # "what_to_do", "why_it_matters", "how_to_proceed"
    confidence_level: float
    action_orientation: bool
    measurable_outcome: Optional[str] = None


@dataclass
class KeyLine:
    """Major supporting argument (3-5 MECE groups)"""

    argument: str
    supporting_evidence: List[str] = field(default_factory=list)
    logical_type: LogicalStructure = LogicalStructure.INDUCTIVE
    evidence_strength: float = 0.0
    actionability_score: float = 0.0


@dataclass
class PyramidStructure:
    """Complete pyramid structure following Minto principles"""

    governing_thought: GoverningThought
    key_lines: List[KeyLine] = field(default_factory=list)
    narrative_pattern: NarrativePattern = NarrativePattern.SCQA
    logical_consistency: float = 0.0
    mece_compliance: float = 0.0
    audience_appropriateness: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class PyramidPrincipleSynthesizer:
    """Elite consulting synthesis engine using Pyramid Principle"""

    def __init__(self):
        self.framework_config = self._load_framework_config()
        self.narrative_templates = self._initialize_narrative_templates()
        self.mece_validators = self._initialize_mece_validators()
        self.logical_checkers = self._initialize_logical_checkers()

    def _load_framework_config(self) -> Dict[str, Any]:
        """Load Pyramid Principle configuration from NWAY framework"""
        try:
            config_path = (
                Path(__file__).parent.parent.parent
                / "cognitive_architecture"
                / "NWAY_ELITE_CONSULTING_FRAMEWORKS_001.yaml"
            )
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                return config.get("pyramid_principle", {})
        except Exception as e:
            logger.warning(f"Could not load pyramid config: {e}")
            return {}

    def _initialize_narrative_templates(self) -> Dict[NarrativePattern, Dict[str, str]]:
        """Initialize narrative templates for different contexts"""
        return {
            NarrativePattern.SCQA: {
                "structure": "Situation → Complication → Question → Answer",
                "use_case": "General purpose, neutral audience",
                "opening": "In situation X, complication Y creates question Z, answered by...",
                "flow": "context_problem_solution",
            },
            NarrativePattern.DIRECT: {
                "structure": "Answer → Situation → Complication",
                "use_case": "Action-oriented, 'What should I do?' focus",
                "opening": "We recommend X because situation Y has complication Z...",
                "flow": "solution_context_justification",
            },
            NarrativePattern.STANDARD: {
                "structure": "Situation → Complication → Answer",
                "use_case": "Problem resolution focus",
                "opening": "Given situation X with complication Y, the solution is...",
                "flow": "context_problem_solution",
            },
            NarrativePattern.HOW_TO: {
                "structure": "Process steps in MECE sequence",
                "use_case": "Implementation and transformation",
                "opening": "To achieve X, follow these steps...",
                "flow": "step_by_step_process",
            },
            NarrativePattern.ARGUMENT: {
                "structure": "Deductive chain revealing conclusion",
                "use_case": "Skeptical or hostile audience",
                "opening": "If we accept premise A, and observe B, then...",
                "flow": "premise_evidence_conclusion",
            },
        }

    def _initialize_mece_validators(self) -> Dict[str, callable]:
        """Initialize MECE validation functions"""
        return {
            "mutually_exclusive": self._check_mutual_exclusivity,
            "collectively_exhaustive": self._check_collective_exhaustiveness,
            "logical_completeness": self._check_logical_completeness,
            "actionable_depth": self._check_actionable_depth,
        }

    def _initialize_logical_checkers(self) -> Dict[str, callable]:
        """Initialize logical consistency checkers"""
        return {
            "deductive_validity": self._check_deductive_validity,
            "inductive_strength": self._check_inductive_strength,
            "question_answer_alignment": self._check_question_answer_alignment,
            "evidence_support": self._check_evidence_support,
        }

    async def synthesize_pyramid(
        self,
        analysis_findings: Dict[str, Any],
        target_audience: str = "executive",
        synthesis_goal: str = "recommendation",
    ) -> PyramidStructure:
        """
        Synthesize analysis findings into Pyramid Principle structure

        Args:
            analysis_findings: Raw analysis results and data
            target_audience: "executive", "technical", "board", "operational"
            synthesis_goal: "recommendation", "explanation", "decision_support"

        Returns:
            Complete pyramid structure ready for communication
        """
        logger.info(f"🏗️ Starting Pyramid Principle Synthesis for {target_audience}")

        # Step 1: Extract and cluster findings (Bottom-Up)
        clustered_findings = await self._cluster_findings_bottom_up(analysis_findings)

        # Step 2: Generate implications (So What?)
        implications = await self._generate_implications(clustered_findings)

        # Step 3: Synthesize governing thought (Now What?)
        governing_thought = await self._synthesize_governing_thought(
            implications, synthesis_goal, target_audience
        )

        # Step 4: Structure key lines (MECE breakdown)
        key_lines = await self._structure_key_lines(governing_thought, implications)

        # Step 5: Select narrative pattern
        narrative_pattern = self._select_narrative_pattern(
            target_audience, synthesis_goal
        )

        # Step 6: Validate logical structure
        pyramid = PyramidStructure(
            governing_thought=governing_thought,
            key_lines=key_lines,
            narrative_pattern=narrative_pattern,
        )

        # Step 7: Quality assurance
        pyramid = await self._validate_pyramid_quality(pyramid)

        logger.info(
            f"✅ Pyramid Synthesis Complete - Logic: {pyramid.logical_consistency:.2f}, "
            f"MECE: {pyramid.mece_compliance:.2f}"
        )

        return pyramid

    async def _cluster_findings_bottom_up(
        self, analysis_findings: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Cluster analysis findings into logical groups (MECE)"""

        # Extract raw findings
        raw_findings = []

        # Process different types of findings
        if "analysis_results" in analysis_findings:
            raw_findings.extend(
                self._extract_analysis_results(analysis_findings["analysis_results"])
            )

        if "research_data" in analysis_findings:
            raw_findings.extend(
                self._extract_research_data(analysis_findings["research_data"])
            )

        if "evidence_events" in analysis_findings:
            raw_findings.extend(
                self._extract_evidence_events(analysis_findings["evidence_events"])
            )

        # Cluster findings using McKinsey frameworks
        clustered_findings = {
            "quantitative_evidence": [],
            "qualitative_insights": [],
            "market_dynamics": [],
            "operational_factors": [],
            "strategic_implications": [],
        }

        for finding in raw_findings:
            cluster = self._classify_finding(finding)
            if cluster in clustered_findings:
                clustered_findings[cluster].append(finding)

        # Remove empty clusters
        clustered_findings = {k: v for k, v in clustered_findings.items() if v}

        return clustered_findings

    async def _generate_implications(
        self, clustered_findings: Dict[str, List[str]]
    ) -> Dict[str, str]:
        """Generate 'So What?' implications from clustered findings"""

        implications = {}

        for cluster_name, findings in clustered_findings.items():
            # Generate implication for each cluster
            if findings:
                implication = self._synthesize_cluster_implication(
                    cluster_name, findings
                )
                implications[cluster_name] = implication

        return implications

    async def _synthesize_governing_thought(
        self, implications: Dict[str, str], synthesis_goal: str, target_audience: str
    ) -> GoverningThought:
        """Synthesize the governing thought (core message)"""

        # Determine answer type based on synthesis goal
        answer_type_mapping = {
            "recommendation": "what_to_do",
            "explanation": "why_it_matters",
            "decision_support": "how_to_proceed",
        }
        answer_type = answer_type_mapping.get(synthesis_goal, "what_to_do")

        # Synthesize core message from implications
        core_message = self._create_core_message(
            implications, answer_type, target_audience
        )

        # Assess confidence level
        confidence_level = self._assess_message_confidence(implications, core_message)

        # Determine action orientation
        action_orientation = (
            answer_type == "what_to_do" or "recommend" in core_message.lower()
        )

        # Extract measurable outcome if present
        measurable_outcome = self._extract_measurable_outcome(core_message)

        return GoverningThought(
            message=core_message,
            answer_type=answer_type,
            confidence_level=confidence_level,
            action_orientation=action_orientation,
            measurable_outcome=measurable_outcome,
        )

    async def _structure_key_lines(
        self, governing_thought: GoverningThought, implications: Dict[str, str]
    ) -> List[KeyLine]:
        """Structure 3-5 MECE key lines supporting the governing thought"""

        key_lines = []

        # Convert implications to key lines
        for cluster_name, implication in implications.items():

            # Create supporting evidence list
            supporting_evidence = self._generate_supporting_evidence(
                cluster_name, implication
            )

            # Determine logical type
            logical_type = self._determine_logical_type(
                implication, governing_thought.message
            )

            # Assess evidence strength
            evidence_strength = self._assess_evidence_strength(supporting_evidence)

            # Calculate actionability score
            actionability_score = self._calculate_actionability_score(implication)

            key_line = KeyLine(
                argument=implication,
                supporting_evidence=supporting_evidence,
                logical_type=logical_type,
                evidence_strength=evidence_strength,
                actionability_score=actionability_score,
            )

            key_lines.append(key_line)

        # Ensure MECE compliance (limit to 3-5 key lines)
        if len(key_lines) > 5:
            key_lines = self._consolidate_key_lines(key_lines)

        # Sort by evidence strength and actionability
        key_lines.sort(
            key=lambda x: (x.evidence_strength + x.actionability_score) / 2,
            reverse=True,
        )

        return key_lines[:5]  # Maximum 5 key lines

    def _select_narrative_pattern(
        self, target_audience: str, synthesis_goal: str
    ) -> NarrativePattern:
        """Select appropriate narrative pattern based on context"""

        # Audience-based pattern selection
        if target_audience == "executive" and synthesis_goal == "recommendation":
            return NarrativePattern.DIRECT  # Lead with answer
        elif target_audience == "board" and synthesis_goal == "decision_support":
            return NarrativePattern.SCQA  # Full context
        elif synthesis_goal == "explanation":
            return NarrativePattern.STANDARD  # Problem-solution
        elif "implementation" in synthesis_goal.lower():
            return NarrativePattern.HOW_TO  # Process focus
        else:
            return NarrativePattern.SCQA  # Default general purpose

    async def _validate_pyramid_quality(
        self, pyramid: PyramidStructure
    ) -> PyramidStructure:
        """Validate and score pyramid quality using McKinsey standards"""

        # Check logical consistency
        logical_consistency = await self._check_logical_consistency(pyramid)
        pyramid.logical_consistency = logical_consistency

        # Check MECE compliance
        mece_compliance = await self._check_mece_compliance(pyramid)
        pyramid.mece_compliance = mece_compliance

        # Check audience appropriateness
        audience_appropriateness = self._check_audience_appropriateness(pyramid)
        pyramid.audience_appropriateness = audience_appropriateness

        return pyramid

    def _extract_analysis_results(self, analysis_results: Any) -> List[str]:
        """Extract findings from analysis results"""
        findings = []

        if isinstance(analysis_results, dict):
            for key, value in analysis_results.items():
                if isinstance(value, str) and len(value) > 10:
                    findings.append(f"{key}: {value}")
                elif isinstance(value, (int, float)):
                    findings.append(f"{key}: {value}")
        elif isinstance(analysis_results, str):
            findings.append(analysis_results)

        return findings

    def _extract_research_data(self, research_data: Any) -> List[str]:
        """Extract findings from research data"""
        findings = []

        if isinstance(research_data, dict):
            if "content" in research_data:
                findings.append(research_data["content"])
            if "sources" in research_data:
                findings.append(f"Sources: {len(research_data['sources'])} references")

        return findings

    def _extract_evidence_events(
        self, evidence_events: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract findings from evidence events"""
        findings = []

        for event in evidence_events:
            if "content" in event:
                findings.append(event["content"])
            elif "description" in event:
                findings.append(event["description"])

        return findings

    def _classify_finding(self, finding: str) -> str:
        """Classify finding into appropriate cluster"""
        finding_lower = finding.lower()

        # Quantitative evidence
        if any(
            indicator in finding_lower
            for indicator in [
                "%",
                "percent",
                "million",
                "billion",
                "increase",
                "decrease",
                "revenue",
                "cost",
            ]
        ):
            return "quantitative_evidence"

        # Market dynamics
        elif any(
            indicator in finding_lower
            for indicator in [
                "market",
                "competition",
                "customer",
                "demand",
                "supply",
                "trend",
            ]
        ):
            return "market_dynamics"

        # Operational factors
        elif any(
            indicator in finding_lower
            for indicator in [
                "operational",
                "process",
                "efficiency",
                "capacity",
                "resource",
            ]
        ):
            return "operational_factors"

        # Strategic implications
        elif any(
            indicator in finding_lower
            for indicator in [
                "strategy",
                "strategic",
                "long-term",
                "vision",
                "positioning",
            ]
        ):
            return "strategic_implications"

        # Default to qualitative insights
        else:
            return "qualitative_insights"

    def _synthesize_cluster_implication(
        self, cluster_name: str, findings: List[str]
    ) -> str:
        """Synthesize implication from cluster of findings"""

        # Simple synthesis based on cluster type
        synthesis_templates = {
            "quantitative_evidence": "The data indicates that {summary}",
            "qualitative_insights": "Stakeholder analysis reveals that {summary}",
            "market_dynamics": "Market conditions suggest that {summary}",
            "operational_factors": "Operational analysis shows that {summary}",
            "strategic_implications": "Strategic assessment indicates that {summary}",
        }

        template = synthesis_templates.get(
            cluster_name, "Analysis shows that {summary}"
        )

        # Create summary of findings
        summary = self._summarize_findings(findings)

        return template.format(summary=summary)

    def _summarize_findings(self, findings: List[str]) -> str:
        """Create concise summary of findings"""
        if not findings:
            return "no significant patterns identified"

        # Simple summarization - in production would use advanced NLP
        if len(findings) == 1:
            return findings[0]
        elif len(findings) <= 3:
            return " and ".join(findings)
        else:
            return f"{findings[0]} along with {len(findings)-1} supporting factors"

    def _create_core_message(
        self, implications: Dict[str, str], answer_type: str, target_audience: str
    ) -> str:
        """Create core message from implications"""

        # Extract key themes from implications
        themes = list(implications.values())

        if not themes:
            return "Analysis complete - specific recommendations require additional context"

        # Message templates by answer type
        if answer_type == "what_to_do":
            if target_audience == "executive":
                return (
                    f"Recommend immediate action on {self._extract_primary_theme(themes)} "
                    f"to address {self._extract_secondary_theme(themes)}"
                )
            else:
                return f"Primary recommendation: {self._extract_primary_theme(themes)}"

        elif answer_type == "why_it_matters":
            return (
                f"Critical significance: {self._extract_primary_theme(themes)} "
                f"with implications for {self._extract_secondary_theme(themes)}"
            )

        else:  # how_to_proceed
            return (
                f"Recommended approach: {self._extract_primary_theme(themes)} "
                f"through {self._extract_secondary_theme(themes)}"
            )

    def _extract_primary_theme(self, themes: List[str]) -> str:
        """Extract primary theme from implications"""
        if not themes:
            return "comprehensive analysis"

        # Simple extraction - in production would use semantic analysis
        return themes[0].split(".")[0] if themes else "analysis findings"

    def _extract_secondary_theme(self, themes: List[str]) -> str:
        """Extract secondary theme from implications"""
        if len(themes) > 1:
            return themes[1].split(".")[0]
        else:
            return "supporting factors"

    def _assess_message_confidence(
        self, implications: Dict[str, str], core_message: str
    ) -> float:
        """Assess confidence level in core message"""
        confidence_score = 0.5  # Base confidence

        # Increase confidence based on number of supporting implications
        confidence_score += min(len(implications) * 0.1, 0.3)

        # Increase confidence if specific metrics mentioned
        if any(
            indicator in core_message.lower()
            for indicator in ["%", "percent", "million", "significant", "major"]
        ):
            confidence_score += 0.2

        return min(confidence_score, 1.0)

    def _extract_measurable_outcome(self, core_message: str) -> Optional[str]:
        """Extract measurable outcome from core message"""
        # Look for quantitative indicators
        if any(
            indicator in core_message.lower()
            for indicator in [
                "%",
                "percent",
                "million",
                "billion",
                "increase",
                "reduce",
            ]
        ):
            return "Quantitative outcome specified"
        else:
            return None

    def _generate_supporting_evidence(
        self, cluster_name: str, implication: str
    ) -> List[str]:
        """Generate supporting evidence for key line"""

        # Evidence templates by cluster type
        evidence_templates = {
            "quantitative_evidence": [
                "Data analysis results",
                "Statistical significance testing",
                "Comparative benchmarking",
            ],
            "qualitative_insights": [
                "Stakeholder interviews",
                "Expert opinion synthesis",
                "Observational analysis",
            ],
            "market_dynamics": [
                "Market research data",
                "Competitive analysis",
                "Industry trend analysis",
            ],
            "operational_factors": [
                "Process efficiency metrics",
                "Resource utilization analysis",
                "Operational benchmarking",
            ],
            "strategic_implications": [
                "Strategic framework analysis",
                "Long-term impact assessment",
                "Risk-benefit evaluation",
            ],
        }

        return evidence_templates.get(
            cluster_name, ["Supporting analysis", "Empirical evidence"]
        )

    def _determine_logical_type(
        self, implication: str, governing_message: str
    ) -> LogicalStructure:
        """Determine logical relationship type"""

        # Simple heuristic for logical type determination
        if "because" in implication.lower() or "since" in implication.lower():
            return LogicalStructure.DEDUCTIVE
        elif "indicates" in implication.lower() or "suggests" in implication.lower():
            return LogicalStructure.INDUCTIVE
        else:
            return LogicalStructure.GROUPING

    def _assess_evidence_strength(self, supporting_evidence: List[str]) -> float:
        """Assess strength of supporting evidence"""

        # Base strength on evidence type and quantity
        strength = 0.5  # Base strength

        # Increase strength based on evidence types
        for evidence in supporting_evidence:
            if any(
                strong_indicator in evidence.lower()
                for strong_indicator in [
                    "data",
                    "statistical",
                    "quantitative",
                    "empirical",
                ]
            ):
                strength += 0.15
            elif any(
                moderate_indicator in evidence.lower()
                for moderate_indicator in ["analysis", "research", "benchmarking"]
            ):
                strength += 0.1

        return min(strength, 1.0)

    def _calculate_actionability_score(self, implication: str) -> float:
        """Calculate how actionable the implication is"""

        actionability = 0.3  # Base score

        # Increase for action-oriented language
        if any(
            action_word in implication.lower()
            for action_word in [
                "recommend",
                "should",
                "must",
                "need",
                "require",
                "action",
            ]
        ):
            actionability += 0.3

        # Increase for specific outcomes
        if any(
            outcome_word in implication.lower()
            for outcome_word in [
                "increase",
                "reduce",
                "improve",
                "eliminate",
                "achieve",
            ]
        ):
            actionability += 0.2

        # Increase for time-bound language
        if any(
            time_word in implication.lower()
            for time_word in ["immediate", "urgent", "month", "quarter", "year"]
        ):
            actionability += 0.2

        return min(actionability, 1.0)

    def _consolidate_key_lines(self, key_lines: List[KeyLine]) -> List[KeyLine]:
        """Consolidate key lines to maintain MECE with maximum 5"""

        # Sort by combined score
        key_lines.sort(
            key=lambda x: (x.evidence_strength + x.actionability_score) / 2,
            reverse=True,
        )

        # Keep top 5
        return key_lines[:5]

    async def _check_logical_consistency(self, pyramid: PyramidStructure) -> float:
        """Check logical consistency of pyramid structure"""

        consistency_score = 0.5  # Base score

        # Check governing thought clarity
        if len(pyramid.governing_thought.message) > 10:
            consistency_score += 0.2

        # Check key line support
        if len(pyramid.key_lines) >= 3:
            consistency_score += 0.2

        # Check evidence alignment
        total_evidence_strength = sum(kl.evidence_strength for kl in pyramid.key_lines)
        if pyramid.key_lines:
            avg_evidence_strength = total_evidence_strength / len(pyramid.key_lines)
            consistency_score += avg_evidence_strength * 0.1

        return min(consistency_score, 1.0)

    async def _check_mece_compliance(self, pyramid: PyramidStructure) -> float:
        """Check MECE compliance of key lines"""

        mece_score = 0.5  # Base score

        # Check number of key lines (3-5 is optimal)
        num_key_lines = len(pyramid.key_lines)
        if 3 <= num_key_lines <= 5:
            mece_score += 0.3
        elif num_key_lines == 2:
            mece_score += 0.1

        # Check for mutual exclusivity (simple heuristic)
        key_line_texts = [kl.argument.lower() for kl in pyramid.key_lines]
        overlap_count = 0
        for i, text1 in enumerate(key_line_texts):
            for j, text2 in enumerate(key_line_texts[i + 1 :], i + 1):
                common_words = set(text1.split()) & set(text2.split())
                if len(common_words) > 3:  # Significant overlap
                    overlap_count += 1

        if overlap_count == 0:
            mece_score += 0.2

        return min(mece_score, 1.0)

    def _check_audience_appropriateness(self, pyramid: PyramidStructure) -> float:
        """Check appropriateness for target audience"""

        appropriateness = 0.7  # Base score for structured approach

        # Check message clarity
        message_length = len(pyramid.governing_thought.message.split())
        if 10 <= message_length <= 25:  # Optimal length
            appropriateness += 0.2

        # Check action orientation for executive audience
        if pyramid.governing_thought.action_orientation:
            appropriateness += 0.1

        return min(appropriateness, 1.0)

    # MECE Validation Methods
    def _check_mutual_exclusivity(self, key_lines: List[KeyLine]) -> float:
        """Check if key lines are mutually exclusive"""
        if len(key_lines) <= 1:
            return 1.0

        overlap_score = 0.0
        comparisons = 0

        for i, kl1 in enumerate(key_lines):
            for kl2 in key_lines[i + 1 :]:
                comparisons += 1
                # Simple word overlap check
                words1 = set(kl1.argument.lower().split())
                words2 = set(kl2.argument.lower().split())
                overlap = len(words1 & words2)
                total_words = len(words1 | words2)

                if total_words > 0:
                    overlap_ratio = overlap / total_words
                    overlap_score += 1.0 - overlap_ratio

        return overlap_score / comparisons if comparisons > 0 else 1.0

    def _check_collective_exhaustiveness(self, key_lines: List[KeyLine]) -> float:
        """Check if key lines collectively cover the governing thought"""
        # Simplified check based on coverage breadth
        if len(key_lines) >= 3:
            return 0.8
        elif len(key_lines) == 2:
            return 0.6
        else:
            return 0.4

    def _check_logical_completeness(self, pyramid: PyramidStructure) -> float:
        """Check logical completeness of argument structure"""
        completeness = 0.0

        # Check governing thought presence
        if pyramid.governing_thought.message:
            completeness += 0.3

        # Check key line support
        if pyramid.key_lines:
            completeness += 0.4

        # Check evidence backing
        evidence_count = sum(len(kl.supporting_evidence) for kl in pyramid.key_lines)
        if evidence_count > 0:
            completeness += 0.3

        return completeness

    def _check_actionable_depth(self, pyramid: PyramidStructure) -> float:
        """Check if pyramid reaches actionable depth"""
        actionability_scores = [kl.actionability_score for kl in pyramid.key_lines]
        if actionability_scores:
            return sum(actionability_scores) / len(actionability_scores)
        return 0.0

    # Logical Consistency Checkers
    def _check_deductive_validity(self, pyramid: PyramidStructure) -> float:
        """Check validity of deductive reasoning"""
        deductive_lines = [
            kl
            for kl in pyramid.key_lines
            if kl.logical_type == LogicalStructure.DEDUCTIVE
        ]

        if not deductive_lines:
            return 1.0  # No deductive claims to validate

        # Simple validity check based on evidence strength
        avg_evidence = sum(kl.evidence_strength for kl in deductive_lines) / len(
            deductive_lines
        )
        return avg_evidence

    def _check_inductive_strength(self, pyramid: PyramidStructure) -> float:
        """Check strength of inductive reasoning"""
        inductive_lines = [
            kl
            for kl in pyramid.key_lines
            if kl.logical_type == LogicalStructure.INDUCTIVE
        ]

        if not inductive_lines:
            return 1.0  # No inductive claims to validate

        # Strength based on evidence quantity and quality
        avg_evidence = sum(kl.evidence_strength for kl in inductive_lines) / len(
            inductive_lines
        )
        return avg_evidence

    def _check_question_answer_alignment(self, pyramid: PyramidStructure) -> float:
        """Check if governing thought answers implied question"""
        # Check if governing thought has appropriate answer orientation
        if pyramid.governing_thought.answer_type in [
            "what_to_do",
            "why_it_matters",
            "how_to_proceed",
        ]:
            return 0.8
        return 0.5

    def _check_evidence_support(self, pyramid: PyramidStructure) -> float:
        """Check if evidence adequately supports conclusions"""
        if not pyramid.key_lines:
            return 0.0

        evidence_strengths = [kl.evidence_strength for kl in pyramid.key_lines]
        return sum(evidence_strengths) / len(evidence_strengths)

    def generate_narrative_output(self, pyramid: PyramidStructure) -> Dict[str, Any]:
        """Generate narrative output using selected pattern"""

        narrative_template = self.narrative_templates[pyramid.narrative_pattern]

        output = {
            "narrative_pattern": pyramid.narrative_pattern.value,
            "structure": narrative_template["structure"],
            "governing_thought": pyramid.governing_thought.message,
            "key_arguments": [kl.argument for kl in pyramid.key_lines],
            "logical_flow": narrative_template["flow"],
            "quality_scores": {
                "logical_consistency": pyramid.logical_consistency,
                "mece_compliance": pyramid.mece_compliance,
                "audience_appropriateness": pyramid.audience_appropriateness,
            },
            "evidence_strength": {
                kl.argument: kl.evidence_strength for kl in pyramid.key_lines
            },
            "actionability": {
                kl.argument: kl.actionability_score for kl in pyramid.key_lines
            },
        }

        return output

    def export_for_senior_advisor(self, pyramid: PyramidStructure) -> Dict[str, Any]:
        """Export pyramid structure for Senior Advisor integration"""

        return {
            "pyramid_synthesis": {
                "governing_thought": {
                    "message": pyramid.governing_thought.message,
                    "answer_type": pyramid.governing_thought.answer_type,
                    "confidence_level": pyramid.governing_thought.confidence_level,
                    "action_orientation": pyramid.governing_thought.action_orientation,
                    "measurable_outcome": pyramid.governing_thought.measurable_outcome,
                },
                "key_lines": [
                    {
                        "argument": kl.argument,
                        "supporting_evidence": kl.supporting_evidence,
                        "logical_type": kl.logical_type.value,
                        "evidence_strength": kl.evidence_strength,
                        "actionability_score": kl.actionability_score,
                    }
                    for kl in pyramid.key_lines
                ],
                "narrative_pattern": pyramid.narrative_pattern.value,
                "quality_metrics": {
                    "logical_consistency": pyramid.logical_consistency,
                    "mece_compliance": pyramid.mece_compliance,
                    "audience_appropriateness": pyramid.audience_appropriateness,
                },
                "pyramid_metadata": {
                    "timestamp": pyramid.timestamp.isoformat(),
                    "synthesis_method": "pyramid_principle",
                    "framework_source": "mckinsey_minto_methodology",
                },
            }
        }


# Usage Example
async def main():
    """Example usage of Pyramid Principle Synthesizer"""

    # Initialize synthesizer
    synthesizer = PyramidPrincipleSynthesizer()

    # Example analysis findings
    analysis_findings = {
        "analysis_results": {
            "revenue_decline": "Quarterly revenue decreased by 15% due to market competition",
            "customer_satisfaction": "Customer satisfaction scores dropped 12 points",
            "operational_efficiency": "Process efficiency improved by 8% through automation",
        },
        "research_data": {
            "content": "Industry analysis shows 23% market growth with new entrants",
            "sources": [
                "McKinsey Global Institute",
                "BCG Research",
                "Internal Analysis",
            ],
        },
        "evidence_events": [
            {"content": "Competitive pricing pressure identified in Q3"},
            {"content": "New product launches by 3 major competitors"},
            {"content": "Customer acquisition cost increased 28%"},
        ],
    }

    # Synthesize pyramid
    pyramid = await synthesizer.synthesize_pyramid(
        analysis_findings=analysis_findings,
        target_audience="executive",
        synthesis_goal="recommendation",
    )

    # Generate narrative output
    narrative = synthesizer.generate_narrative_output(pyramid)

    print("Pyramid Principle Synthesis Complete:")
    print(f"Governing Thought: {pyramid.governing_thought.message}")
    print(f"Key Lines: {len(pyramid.key_lines)}")
    print(f"Logical Consistency: {pyramid.logical_consistency:.2f}")
    print(f"MECE Compliance: {pyramid.mece_compliance:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
