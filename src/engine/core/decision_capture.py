"""
Decision Capture System for METIS
Captures decision trees, alternative analysis, and decision provenance with full traceability
"""

import json
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Literal, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import threading
from collections import defaultdict, OrderedDict
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()


class DecisionType(str, Enum):
    """Types of decisions captured in the system"""

    STRATEGIC = "strategic"
    OPERATIONAL = "operational"
    ANALYTICAL = "analytical"
    TECHNICAL = "technical"
    INVESTMENT = "investment"
    RESOURCE_ALLOCATION = "resource_allocation"
    RISK_MANAGEMENT = "risk_management"
    PROCESS_IMPROVEMENT = "process_improvement"


class ConfidenceLevel(str, Enum):
    """Confidence levels for decisions and alternatives"""

    VERY_LOW = "very_low"  # 0.0-0.2
    LOW = "low"  # 0.2-0.4
    MEDIUM = "medium"  # 0.4-0.6
    HIGH = "high"  # 0.6-0.8
    VERY_HIGH = "very_high"  # 0.8-1.0


class DecisionStatus(str, Enum):
    """Status of decision analysis"""

    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    VALIDATED = "validated"
    IMPLEMENTED = "implemented"
    ARCHIVED = "archived"


@dataclass
class DecisionCriteria:
    """Criteria used for decision evaluation"""

    name: str
    description: str
    weight: float  # 0.0-1.0
    measurement_type: Literal["quantitative", "qualitative", "binary"]
    preferred_direction: Literal["maximize", "minimize", "target"]  # For optimization
    target_value: Optional[float] = None
    min_threshold: Optional[float] = None
    max_threshold: Optional[float] = None

    def __post_init__(self):
        """Validate criteria configuration"""
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"Weight must be between 0.0 and 1.0, got {self.weight}")


@dataclass
class AlternativeScore:
    """Score for a single criterion on an alternative"""

    criterion_name: str
    raw_value: Union[float, str, bool]
    normalized_score: float  # 0.0-1.0
    confidence: float  # 0.0-1.0
    evidence: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    rationale: str = ""


@dataclass
class DecisionAlternative:
    """Single alternative in a decision analysis"""

    id: str
    name: str
    description: str

    # Scoring and evaluation
    scores: Dict[str, AlternativeScore] = field(default_factory=dict)
    total_score: float = 0.0
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM

    # Implementation details
    implementation_complexity: Literal["low", "medium", "high"] = "medium"
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    timeline_estimate: Optional[str] = None
    risk_factors: List[str] = field(default_factory=list)

    # Provenance
    data_sources: List[str] = field(default_factory=list)
    analysis_methods: List[str] = field(default_factory=list)
    expert_opinions: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None

    def calculate_total_score(self, criteria: List[DecisionCriteria]) -> float:
        """Calculate weighted total score across all criteria"""
        if not self.scores or not criteria:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for criterion in criteria:
            if criterion.name in self.scores:
                score = self.scores[criterion.name]
                weighted_sum += score.normalized_score * criterion.weight
                total_weight += criterion.weight

        self.total_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        self.last_updated = datetime.utcnow()
        return self.total_score


@dataclass
class DecisionTree:
    """Hierarchical decision structure with alternatives and sub-decisions"""

    id: str
    name: str
    description: str
    decision_type: DecisionType

    # Decision structure
    criteria: List[DecisionCriteria] = field(default_factory=list)
    alternatives: List[DecisionAlternative] = field(default_factory=list)
    sub_decisions: List["DecisionTree"] = field(default_factory=list)

    # Analysis context
    business_context: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)

    # Decision outcome
    recommended_alternative: Optional[str] = None
    decision_rationale: str = ""
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    status: DecisionStatus = DecisionStatus.DRAFT

    # Provenance and traceability
    data_sources: List[str] = field(default_factory=list)
    analysis_lineage: List[Dict[str, Any]] = field(default_factory=list)
    review_comments: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    engagement_id: Optional[str] = None
    phase: Optional[str] = None

    def add_alternative(self, alternative: DecisionAlternative) -> None:
        """Add alternative and recalculate scores"""
        self.alternatives.append(alternative)
        self._recalculate_all_scores()
        self.last_updated = datetime.utcnow()

    def add_criterion(self, criterion: DecisionCriteria) -> None:
        """Add decision criterion"""
        self.criteria.append(criterion)
        self.last_updated = datetime.utcnow()

    def _recalculate_all_scores(self) -> None:
        """Recalculate total scores for all alternatives"""
        for alternative in self.alternatives:
            alternative.calculate_total_score(self.criteria)

    def get_ranked_alternatives(self) -> List[DecisionAlternative]:
        """Get alternatives ranked by total score (descending)"""
        self._recalculate_all_scores()
        return sorted(self.alternatives, key=lambda a: a.total_score, reverse=True)

    def get_decision_summary(self) -> Dict[str, Any]:
        """Get summary of decision analysis"""
        ranked_alts = self.get_ranked_alternatives()

        return {
            "decision_id": self.id,
            "name": self.name,
            "type": self.decision_type,
            "status": self.status,
            "criteria_count": len(self.criteria),
            "alternatives_count": len(self.alternatives),
            "recommended_alternative": self.recommended_alternative,
            "top_alternative": ranked_alts[0].name if ranked_alts else None,
            "confidence_level": self.confidence_level,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class DecisionAnalysisResult:
    """Result of decision analysis with full traceability"""

    decision_tree: DecisionTree
    analysis_metadata: Dict[str, Any]

    # Analysis metrics
    criteria_coverage: float  # 0.0-1.0
    alternative_completeness: float  # 0.0-1.0
    confidence_consistency: float  # 0.0-1.0
    data_quality_score: float  # 0.0-1.0

    # Traceability
    analysis_steps: List[Dict[str, Any]] = field(default_factory=list)
    data_lineage: Dict[str, List[str]] = field(default_factory=dict)
    assumptions_validated: List[str] = field(default_factory=list)

    # Performance metrics
    analysis_duration_ms: int = 0
    computation_complexity: Literal["low", "medium", "high"] = "medium"

    def get_quality_assessment(self) -> Dict[str, Any]:
        """Assess overall quality of decision analysis"""
        overall_quality = (
            0.3 * self.criteria_coverage
            + 0.25 * self.alternative_completeness
            + 0.25 * self.confidence_consistency
            + 0.2 * self.data_quality_score
        )

        return {
            "overall_quality": overall_quality,
            "criteria_coverage": self.criteria_coverage,
            "alternative_completeness": self.alternative_completeness,
            "confidence_consistency": self.confidence_consistency,
            "data_quality_score": self.data_quality_score,
            "quality_level": (
                "excellent"
                if overall_quality >= 0.8
                else (
                    "good"
                    if overall_quality >= 0.6
                    else "fair" if overall_quality >= 0.4 else "poor"
                )
            ),
        }


class DecisionCapture:
    """
    Main decision capture system with complete traceability and analysis capabilities
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        enable_persistence: bool = True,
        max_cache_size: int = 1000,
        enable_supabase: bool = True,
    ):
        self.logger = logging.getLogger(__name__)
        self.storage_path = (
            Path(storage_path) if storage_path else Path("data/decisions")
        )
        self.enable_persistence = enable_persistence
        self.max_cache_size = max_cache_size
        self.enable_supabase = enable_supabase

        # In-memory storage
        self.decision_trees: Dict[str, DecisionTree] = OrderedDict()
        self.analysis_results: Dict[str, DecisionAnalysisResult] = OrderedDict()

        # Threading safety
        self._lock = threading.RLock()

        # Analytics
        self.decision_metrics = defaultdict(int)
        self.performance_metrics = defaultdict(list)

        # Supabase configuration
        self.supabase_client = None
        if self.enable_supabase:
            try:
                url = os.environ.get("SUPABASE_URL")
                key = os.environ.get("SUPABASE_ANON_KEY")
                if url and key:
                    self.supabase_client = create_client(url, key)
                    self.logger.info("Supabase client initialized for decision capture")
                else:
                    self.logger.warning(
                        "Supabase credentials not found, falling back to local storage"
                    )
                    self.enable_supabase = False
            except Exception as e:
                self.logger.warning(f"Failed to initialize Supabase client: {e}")
                self.enable_supabase = False

        # Ensure storage directory exists
        if self.enable_persistence:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(
                f"✅ DecisionCapture initialized with storage: {self.storage_path}"
            )
        else:
            self.logger.info("✅ DecisionCapture initialized (memory-only mode)")

    def create_decision_tree(
        self,
        name: str,
        description: str,
        decision_type: DecisionType,
        engagement_id: Optional[str] = None,
        phase: Optional[str] = None,
        business_context: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None,
    ) -> DecisionTree:
        """Create new decision tree"""
        decision_id = str(uuid.uuid4())

        decision_tree = DecisionTree(
            id=decision_id,
            name=name,
            description=description,
            decision_type=decision_type,
            engagement_id=engagement_id,
            phase=phase,
            business_context=business_context or {},
            created_by=created_by,
        )

        with self._lock:
            self.decision_trees[decision_id] = decision_tree
            self.decision_metrics["decisions_created"] += 1
            self._enforce_cache_limits()

        if self.enable_persistence:
            self._persist_decision_tree(decision_tree)

        # Store in Supabase if enabled
        if self.enable_supabase and self.supabase_client:
            self._store_decision_tree_in_supabase(decision_tree)

        self.logger.info(
            f"📊 Created decision tree: {name} ({decision_type}) | ID: {decision_id}"
        )
        return decision_tree

    def add_decision_criteria(
        self, decision_id: str, criteria: List[DecisionCriteria]
    ) -> bool:
        """Add criteria to decision tree"""
        with self._lock:
            if decision_id not in self.decision_trees:
                self.logger.error(f"❌ Decision tree not found: {decision_id}")
                return False

            decision_tree = self.decision_trees[decision_id]

            # Validate criteria weights sum
            existing_weight = sum(c.weight for c in decision_tree.criteria)
            new_weight = sum(c.weight for c in criteria)

            if existing_weight + new_weight > 1.0:
                self.logger.warning(
                    f"⚠️ Criteria weights exceed 1.0: {existing_weight + new_weight}"
                )

            # Add criteria
            for criterion in criteria:
                decision_tree.add_criterion(criterion)

            self.decision_metrics["criteria_added"] += len(criteria)

        if self.enable_persistence:
            self._persist_decision_tree(decision_tree)

        self.logger.info(f"📋 Added {len(criteria)} criteria to decision {decision_id}")
        return True

    def add_decision_alternatives(
        self, decision_id: str, alternatives: List[DecisionAlternative]
    ) -> bool:
        """Add alternatives to decision tree"""
        with self._lock:
            if decision_id not in self.decision_trees:
                self.logger.error(f"❌ Decision tree not found: {decision_id}")
                return False

            decision_tree = self.decision_trees[decision_id]

            # Add alternatives
            for alternative in alternatives:
                decision_tree.add_alternative(alternative)

            self.decision_metrics["alternatives_added"] += len(alternatives)

        if self.enable_persistence:
            self._persist_decision_tree(decision_tree)

        self.logger.info(
            f"🔀 Added {len(alternatives)} alternatives to decision {decision_id}"
        )
        return True

    def score_alternative(
        self,
        decision_id: str,
        alternative_id: str,
        criterion_name: str,
        raw_value: Union[float, str, bool],
        confidence: float,
        evidence: Optional[List[str]] = None,
        assumptions: Optional[List[str]] = None,
        rationale: str = "",
    ) -> bool:
        """Score an alternative on a specific criterion"""
        with self._lock:
            if decision_id not in self.decision_trees:
                self.logger.error(f"❌ Decision tree not found: {decision_id}")
                return False

            decision_tree = self.decision_trees[decision_id]

            # Find alternative
            alternative = None
            for alt in decision_tree.alternatives:
                if alt.id == alternative_id:
                    alternative = alt
                    break

            if not alternative:
                self.logger.error(f"❌ Alternative not found: {alternative_id}")
                return False

            # Find criterion
            criterion = None
            for crit in decision_tree.criteria:
                if crit.name == criterion_name:
                    criterion = crit
                    break

            if not criterion:
                self.logger.error(f"❌ Criterion not found: {criterion_name}")
                return False

            # Normalize score based on criterion type
            normalized_score = self._normalize_score(raw_value, criterion)

            # Create score
            score = AlternativeScore(
                criterion_name=criterion_name,
                raw_value=raw_value,
                normalized_score=normalized_score,
                confidence=confidence,
                evidence=evidence or [],
                assumptions=assumptions or [],
                rationale=rationale,
            )

            alternative.scores[criterion_name] = score
            alternative.calculate_total_score(decision_tree.criteria)

            self.decision_metrics["scores_recorded"] += 1

        if self.enable_persistence:
            self._persist_decision_tree(decision_tree)

        self.logger.debug(
            f"📊 Scored {alternative_id} on {criterion_name}: {normalized_score:.2f}"
        )
        return True

    def _normalize_score(
        self, raw_value: Union[float, str, bool], criterion: DecisionCriteria
    ) -> float:
        """Normalize raw score to 0.0-1.0 range"""
        if criterion.measurement_type == "binary":
            return 1.0 if raw_value else 0.0

        elif criterion.measurement_type == "qualitative":
            # Simple qualitative mapping
            if isinstance(raw_value, str):
                qualitative_map = {
                    "excellent": 1.0,
                    "very high": 1.0,
                    "good": 0.8,
                    "high": 0.8,
                    "average": 0.5,
                    "medium": 0.5,
                    "poor": 0.2,
                    "low": 0.2,
                    "very poor": 0.0,
                    "very low": 0.0,
                }
                return qualitative_map.get(raw_value.lower(), 0.5)
            return 0.5

        elif criterion.measurement_type == "quantitative":
            if not isinstance(raw_value, (int, float)):
                return 0.0

            # Use thresholds if available
            if (
                criterion.min_threshold is not None
                and criterion.max_threshold is not None
            ):
                min_val = criterion.min_threshold
                max_val = criterion.max_threshold

                if criterion.preferred_direction == "minimize":
                    # For minimize: lower values get higher scores
                    if raw_value <= min_val:
                        return 1.0
                    elif raw_value >= max_val:
                        return 0.0
                    else:
                        return 1.0 - ((raw_value - min_val) / (max_val - min_val))
                else:
                    # For maximize: higher values get higher scores
                    if raw_value <= min_val:
                        return 0.0
                    elif raw_value >= max_val:
                        return 1.0
                    else:
                        return (raw_value - min_val) / (max_val - min_val)

            # Default normalization (assume 0-100 scale)
            return min(1.0, max(0.0, float(raw_value) / 100.0))

        return 0.5  # Default fallback

    def analyze_decision(self, decision_id: str) -> Optional[DecisionAnalysisResult]:
        """Perform comprehensive decision analysis"""
        start_time = time.time()

        with self._lock:
            if decision_id not in self.decision_trees:
                self.logger.error(f"❌ Decision tree not found: {decision_id}")
                return None

            decision_tree = self.decision_trees[decision_id]

        # Calculate analysis metrics
        criteria_coverage = self._calculate_criteria_coverage(decision_tree)
        alternative_completeness = self._calculate_alternative_completeness(
            decision_tree
        )
        confidence_consistency = self._calculate_confidence_consistency(decision_tree)
        data_quality_score = self._calculate_data_quality_score(decision_tree)

        # Create analysis steps
        analysis_steps = [
            {
                "step": "criteria_validation",
                "timestamp": datetime.utcnow().isoformat(),
                "result": f"Validated {len(decision_tree.criteria)} criteria",
            },
            {
                "step": "alternative_scoring",
                "timestamp": datetime.utcnow().isoformat(),
                "result": f"Scored {len(decision_tree.alternatives)} alternatives",
            },
            {
                "step": "ranking_calculation",
                "timestamp": datetime.utcnow().isoformat(),
                "result": "Calculated weighted rankings",
            },
        ]

        # Determine recommended alternative
        ranked_alternatives = decision_tree.get_ranked_alternatives()
        if ranked_alternatives:
            decision_tree.recommended_alternative = ranked_alternatives[0].id
            decision_tree.decision_rationale = f"Highest scoring alternative with total score: {ranked_alternatives[0].total_score:.3f}"

        analysis_duration_ms = int((time.time() - start_time) * 1000)

        result = DecisionAnalysisResult(
            decision_tree=decision_tree,
            analysis_metadata={
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "analysis_version": "1.0",
                "total_alternatives": len(decision_tree.alternatives),
                "total_criteria": len(decision_tree.criteria),
            },
            criteria_coverage=criteria_coverage,
            alternative_completeness=alternative_completeness,
            confidence_consistency=confidence_consistency,
            data_quality_score=data_quality_score,
            analysis_steps=analysis_steps,
            analysis_duration_ms=analysis_duration_ms,
            computation_complexity=(
                "medium" if len(decision_tree.alternatives) <= 10 else "high"
            ),
        )

        with self._lock:
            self.analysis_results[decision_id] = result
            self.decision_metrics["analyses_completed"] += 1
            self.performance_metrics["analysis_duration_ms"].append(
                analysis_duration_ms
            )

        self.logger.info(
            f"🔍 Decision analysis completed: {decision_id} | "
            f"Quality: {result.get_quality_assessment()['quality_level']} | "
            f"{analysis_duration_ms}ms"
        )

        return result

    def _calculate_criteria_coverage(self, decision_tree: DecisionTree) -> float:
        """Calculate how well criteria are covered by alternatives"""
        if not decision_tree.criteria or not decision_tree.alternatives:
            return 0.0

        total_scores = 0
        possible_scores = len(decision_tree.criteria) * len(decision_tree.alternatives)

        for alternative in decision_tree.alternatives:
            total_scores += len(alternative.scores)

        return total_scores / possible_scores if possible_scores > 0 else 0.0

    def _calculate_alternative_completeness(self, decision_tree: DecisionTree) -> float:
        """Calculate completeness of alternative descriptions and scoring"""
        if not decision_tree.alternatives:
            return 0.0

        completeness_scores = []

        for alternative in decision_tree.alternatives:
            score = 0.0

            # Check required fields
            if alternative.description.strip():
                score += 0.2
            if alternative.implementation_complexity:
                score += 0.1
            if alternative.timeline_estimate:
                score += 0.1
            if alternative.resource_requirements:
                score += 0.1
            if alternative.data_sources:
                score += 0.1

            # Check scoring coverage
            if decision_tree.criteria:
                scoring_coverage = len(alternative.scores) / len(decision_tree.criteria)
                score += 0.4 * scoring_coverage

            completeness_scores.append(score)

        return sum(completeness_scores) / len(completeness_scores)

    def _calculate_confidence_consistency(self, decision_tree: DecisionTree) -> float:
        """Calculate consistency of confidence levels across alternatives"""
        if not decision_tree.alternatives:
            return 0.0

        all_confidences = []

        for alternative in decision_tree.alternatives:
            for score in alternative.scores.values():
                all_confidences.append(score.confidence)

        if not all_confidences:
            return 0.0

        # Calculate coefficient of variation (lower = more consistent)
        mean_confidence = sum(all_confidences) / len(all_confidences)
        if mean_confidence == 0:
            return 0.0

        variance = sum((c - mean_confidence) ** 2 for c in all_confidences) / len(
            all_confidences
        )
        std_dev = variance**0.5
        cv = std_dev / mean_confidence

        # Convert to consistency score (1.0 = perfectly consistent, 0.0 = highly variable)
        return max(0.0, 1.0 - cv)

    def _calculate_data_quality_score(self, decision_tree: DecisionTree) -> float:
        """Calculate overall data quality score"""
        if not decision_tree.alternatives:
            return 0.0

        quality_factors = []

        # Check data source coverage
        total_sources = 0
        for alternative in decision_tree.alternatives:
            total_sources += len(alternative.data_sources)

        source_coverage = min(
            1.0, total_sources / (len(decision_tree.alternatives) * 3)
        )  # Expect ~3 sources per alternative
        quality_factors.append(source_coverage)

        # Check evidence coverage
        total_evidence = 0
        for alternative in decision_tree.alternatives:
            for score in alternative.scores.values():
                total_evidence += len(score.evidence)

        evidence_coverage = min(
            1.0,
            total_evidence
            / (len(decision_tree.alternatives) * len(decision_tree.criteria)),
        )
        quality_factors.append(evidence_coverage)

        # Check assumption documentation
        total_assumptions = 0
        for alternative in decision_tree.alternatives:
            for score in alternative.scores.values():
                total_assumptions += len(score.assumptions)

        assumption_coverage = min(
            1.0,
            total_assumptions
            / (len(decision_tree.alternatives) * len(decision_tree.criteria) * 0.5),
        )
        quality_factors.append(assumption_coverage)

        return sum(quality_factors) / len(quality_factors) if quality_factors else 0.0

    def get_decision_tree(self, decision_id: str) -> Optional[DecisionTree]:
        """Get decision tree by ID"""
        with self._lock:
            return self.decision_trees.get(decision_id)

    def get_analysis_result(self, decision_id: str) -> Optional[DecisionAnalysisResult]:
        """Get analysis result by decision ID"""
        with self._lock:
            return self.analysis_results.get(decision_id)

    def list_decisions(
        self,
        engagement_id: Optional[str] = None,
        decision_type: Optional[DecisionType] = None,
        status: Optional[DecisionStatus] = None,
    ) -> List[Dict[str, Any]]:
        """List decisions with optional filtering"""
        with self._lock:
            decisions = []

            for decision_tree in self.decision_trees.values():
                # Apply filters
                if engagement_id and decision_tree.engagement_id != engagement_id:
                    continue
                if decision_type and decision_tree.decision_type != decision_type:
                    continue
                if status and decision_tree.status != status:
                    continue

                decisions.append(decision_tree.get_decision_summary())

            return sorted(decisions, key=lambda d: d["last_updated"], reverse=True)

    def export_decision_analysis(
        self, decision_id: str, format: Literal["json", "csv"] = "json"
    ) -> Optional[Dict[str, Any]]:
        """Export complete decision analysis"""
        result = self.get_analysis_result(decision_id)
        if not result:
            return None

        export_data = {
            "decision_analysis": {
                "metadata": result.analysis_metadata,
                "quality_assessment": result.get_quality_assessment(),
                "decision_tree": {
                    "id": result.decision_tree.id,
                    "name": result.decision_tree.name,
                    "description": result.decision_tree.description,
                    "type": result.decision_tree.decision_type,
                    "status": result.decision_tree.status,
                    "recommended_alternative": result.decision_tree.recommended_alternative,
                    "rationale": result.decision_tree.decision_rationale,
                    "confidence_level": result.decision_tree.confidence_level,
                },
                "criteria": [asdict(c) for c in result.decision_tree.criteria],
                "alternatives": [],
                "analysis_steps": result.analysis_steps,
                "performance": {
                    "duration_ms": result.analysis_duration_ms,
                    "complexity": result.computation_complexity,
                },
            }
        }

        # Export alternatives with scores
        for alternative in result.decision_tree.alternatives:
            alt_data = {
                "id": alternative.id,
                "name": alternative.name,
                "description": alternative.description,
                "total_score": alternative.total_score,
                "confidence_level": alternative.confidence_level,
                "implementation_complexity": alternative.implementation_complexity,
                "scores": {},
            }

            for criterion_name, score in alternative.scores.items():
                alt_data["scores"][criterion_name] = {
                    "raw_value": score.raw_value,
                    "normalized_score": score.normalized_score,
                    "confidence": score.confidence,
                    "evidence_count": len(score.evidence),
                    "assumptions_count": len(score.assumptions),
                    "rationale": score.rationale,
                }

            export_data["decision_analysis"]["alternatives"].append(alt_data)

        return export_data

    def _persist_decision_tree(self, decision_tree: DecisionTree) -> None:
        """Persist decision tree to storage"""
        if not self.enable_persistence:
            return

        try:
            file_path = self.storage_path / f"decision_{decision_tree.id}.json"

            # Convert to serializable format
            data = asdict(decision_tree)

            # Handle datetime serialization
            data["created_at"] = decision_tree.created_at.isoformat()
            data["last_updated"] = decision_tree.last_updated.isoformat()

            # Handle nested datetime fields
            for alt in data["alternatives"]:
                alt["created_at"] = (
                    alt["created_at"][:19]
                    if isinstance(alt["created_at"], str)
                    else alt["created_at"].isoformat()
                )
                alt["last_updated"] = (
                    alt["last_updated"][:19]
                    if isinstance(alt["last_updated"], str)
                    else alt["last_updated"].isoformat()
                )

            with open(file_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(
                f"❌ Failed to persist decision tree {decision_tree.id}: {e}"
            )

    def _confidence_to_float(self, confidence_value) -> Optional[float]:
        """Convert confidence enum or string to float value for database storage"""
        if confidence_value is None:
            return None

        # If already a float, return as-is
        if isinstance(confidence_value, (int, float)):
            return float(confidence_value)

        # Convert string/enum to float
        confidence_str = str(confidence_value).lower()
        confidence_map = {
            "very_low": 0.1,
            "low": 0.3,
            "medium": 0.5,
            "high": 0.7,
            "very_high": 0.9,
            "uncertain": 0.2,
        }

        return confidence_map.get(confidence_str, 0.5)

    def _store_decision_tree_in_supabase(self, decision_tree: DecisionTree) -> None:
        """Store decision tree in Supabase database"""
        try:
            # Prepare decision tree data
            decision_data = {
                "decision_id": decision_tree.id,
                "engagement_id": decision_tree.engagement_id,
                "name": decision_tree.name,
                "description": decision_tree.description,
                "decision_type": (
                    decision_tree.decision_type
                    if isinstance(decision_tree.decision_type, str)
                    else decision_tree.decision_type.value
                ),
                "phase": decision_tree.phase or "unknown",
                "context_data": decision_tree.business_context,
                "criteria": [asdict(criterion) for criterion in decision_tree.criteria],
                "weights": {
                    criterion.name: criterion.weight
                    for criterion in decision_tree.criteria
                },
                "scoring_methodology": getattr(
                    decision_tree, "scoring_method", "weighted_sum"
                ),
                "confidence_level": self._confidence_to_float(
                    decision_tree.confidence_level
                ),
                "decision_rationale": decision_tree.decision_rationale,
                "sensitivity_analysis": getattr(
                    decision_tree, "sensitivity_analysis", {}
                ),
                "stakeholder_input": getattr(decision_tree, "stakeholder_input", []),
                "decision_metadata": {
                    "created_by": decision_tree.created_by,
                    "status": (
                        decision_tree.status
                        if isinstance(decision_tree.status, str)
                        else decision_tree.status.value
                    ),
                    "total_alternatives": len(decision_tree.alternatives),
                    "implementation_notes": getattr(
                        decision_tree, "implementation_notes", ""
                    ),
                },
                "created_at": decision_tree.created_at.isoformat(),
                "updated_at": decision_tree.last_updated.isoformat(),
            }

            # Insert decision tree
            result = (
                self.supabase_client.table("decision_trees")
                .insert(decision_data)
                .execute()
            )

            # Store alternatives separately
            for alternative in decision_tree.alternatives:
                alt_data = {
                    "decision_tree_id": result.data[0]["id"],
                    "alternative_id": alternative.id,
                    "name": alternative.name,
                    "description": alternative.description,
                    "scores": {
                        criterion.name: alternative.scores.get(criterion.name, 0.0)
                        for criterion in decision_tree.criteria
                    },
                    "total_score": alternative.total_score,
                    "rank": alternative.rank,
                    "pros": alternative.pros,
                    "cons": alternative.cons,
                    "risks": alternative.risks,
                    "implementation_complexity": alternative.implementation_complexity,
                    "resource_requirements": alternative.resource_requirements,
                    "timeline_estimate": alternative.timeline_estimate,
                    "alternative_metadata": {
                        "notes": alternative.notes,
                        "created_by": decision_tree.created_by,
                    },
                    "created_at": alternative.created_at.isoformat(),
                }

                self.supabase_client.table("decision_alternatives").insert(
                    alt_data
                ).execute()

            self.logger.debug(f"Stored decision tree {decision_tree.id} in Supabase")

        except Exception as e:
            self.logger.error(
                f"Failed to store decision tree {decision_tree.id} in Supabase: {e}"
            )
            # Continue without failing - fallback to local storage

    def _enforce_cache_limits(self) -> None:
        """Enforce in-memory cache size limits"""
        if len(self.decision_trees) > self.max_cache_size:
            # Remove oldest entries
            excess_count = len(self.decision_trees) - self.max_cache_size
            for _ in range(excess_count):
                oldest_id = next(iter(self.decision_trees))
                del self.decision_trees[oldest_id]
                if oldest_id in self.analysis_results:
                    del self.analysis_results[oldest_id]

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance and usage metrics"""
        with self._lock:
            avg_analysis_time = (
                sum(self.performance_metrics["analysis_duration_ms"])
                / len(self.performance_metrics["analysis_duration_ms"])
                if self.performance_metrics["analysis_duration_ms"]
                else 0
            )

            return {
                "decision_metrics": dict(self.decision_metrics),
                "performance_metrics": {
                    "average_analysis_time_ms": avg_analysis_time,
                    "total_analyses": len(
                        self.performance_metrics["analysis_duration_ms"]
                    ),
                },
                "cache_metrics": {
                    "decision_trees_cached": len(self.decision_trees),
                    "analysis_results_cached": len(self.analysis_results),
                    "cache_utilization": len(self.decision_trees) / self.max_cache_size,
                },
            }


# Global DecisionCapture instance
_decision_capture_instance: Optional[DecisionCapture] = None


def get_decision_capture() -> DecisionCapture:
    """Get or create global DecisionCapture instance"""
    global _decision_capture_instance

    if _decision_capture_instance is None:
        _decision_capture_instance = DecisionCapture()

    return _decision_capture_instance
