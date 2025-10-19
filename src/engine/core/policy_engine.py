#!/usr/bin/env python3
"""
METIS Policy Engine - Complete Policy Integration System
Orchestrates dynamic policy management for the cognitive intelligence platform.

Integrates with:
- Munger overlay system (rigor level determination)
- N-way interaction policies (interaction selection criteria)
- Cognitive engagement policies (workflow orchestration rules)
- Enterprise compliance policies (SOC 2, security, audit requirements)

This provides the "Systematic Intelligence Governance" layer that ensures
all cognitive operations follow appropriate organizational policies and standards.
"""

import os
import uuid
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from dotenv import load_dotenv

# Import database client
try:
    from supabase import create_client, Client
except ImportError:
    print("⚠️  Supabase client not available - operating in offline mode")
    Client = None

# Import related systems
try:
    from .munger_overlay import DecisionContext, RigorLevel
    from ..intelligence.nway_munger_bridge import CognitiveIntelligenceRequest
except ImportError:
    try:
        from src.core.munger_overlay import DecisionContext, RigorLevel
        from src.intelligence.nway_munger_bridge import CognitiveIntelligenceRequest
    except ImportError:
        print("⚠️  Core systems not available - using standalone mode")
        DecisionContext = None
        RigorLevel = None
        CognitiveIntelligenceRequest = None


class PolicyType(Enum):
    """Types of policies in the system"""

    MUNGER_RIGOR = "munger_rigor"  # Munger overlay rigor level policies
    NWAY_SELECTION = "nway_selection"  # N-way interaction selection policies
    COGNITIVE_WORKFLOW = "cognitive_workflow"  # Workflow orchestration policies
    ENTERPRISE_COMPLIANCE = "enterprise_compliance"  # SOC 2, security, audit policies
    PERFORMANCE_OPTIMIZATION = (
        "performance_optimization"  # Performance and cost optimization
    )
    QUALITY_ASSURANCE = "quality_assurance"  # Quality gates and validation policies


class PolicyScope(Enum):
    """Policy application scope"""

    GLOBAL = "global"  # Apply to all operations
    DOMAIN = "domain"  # Apply to specific domain (security, product, etc.)
    USER = "user"  # Apply to specific users
    ENGAGEMENT = "engagement"  # Apply to specific engagement types
    ORGANIZATION = "organization"  # Apply to specific organizations (multi-tenant)


class PolicyPriority(Enum):
    """Policy priority levels"""

    CRITICAL = "critical"  # Must be enforced, system fails if violated
    HIGH = "high"  # Should be enforced, warnings if violated
    MEDIUM = "medium"  # Default enforcement level
    LOW = "low"  # Optional guidance, informational only


@dataclass
class PolicyCondition:
    """Individual policy condition"""

    field: str  # Field to evaluate (e.g., "impact", "domain", "user_role")
    operator: str  # Operator (eq, neq, in, not_in, gt, lt, contains)
    value: Any  # Value to compare against
    description: str  # Human-readable description


@dataclass
class PolicyRule:
    """Individual policy rule with conditions and actions"""

    rule_id: str
    name: str
    conditions: List[PolicyCondition]
    actions: Dict[str, Any]  # Actions to take when conditions match
    priority: PolicyPriority
    enabled: bool = True
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Policy:
    """Complete policy definition"""

    policy_id: str
    name: str
    policy_type: PolicyType
    scope: PolicyScope
    rules: List[PolicyRule]
    default_action: Dict[str, Any]
    max_tokens_overhead: Optional[int] = None
    max_latency_ms: Optional[int] = None

    # Metadata
    version: str = "1.0"
    description: Optional[str] = None
    created_by: Optional[str] = None
    organization_id: Optional[str] = None
    active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = datetime.now(timezone.utc)


@dataclass
class PolicyEvaluation:
    """Result of policy evaluation"""

    policy_id: str
    rule_matches: List[str]  # IDs of rules that matched
    actions_applied: List[Dict[str, Any]]
    enforcement_result: str  # "enforced", "warning", "failed", "skipped"
    evaluation_time_ms: int
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PolicyDecision:
    """Final policy decision across all applicable policies"""

    decision_id: str
    context: Dict[str, Any]  # Input context that was evaluated
    policy_evaluations: List[PolicyEvaluation]
    final_actions: Dict[str, Any]  # Consolidated actions to apply
    compliance_status: str  # "compliant", "warning", "violation", "error"

    # Performance metrics
    total_policies_evaluated: int
    total_evaluation_time_ms: int
    created_at: datetime


class PolicyEngine:
    """
    Core Policy Engine - Systematic Intelligence Governance

    Orchestrates all policy evaluation and enforcement across the cognitive
    intelligence platform. Provides dynamic policy management with database
    integration and real-time policy evaluation.

    Key responsibilities:
    1. Load and manage policies from database
    2. Evaluate policies against decision contexts
    3. Enforce policy actions and compliance requirements
    4. Provide policy audit trail and governance reporting
    5. Integrate with all cognitive intelligence systems
    """

    def __init__(self, supabase_client: Optional[Client] = None):
        """Initialize policy engine with database connection"""
        load_dotenv()

        # Database connection
        if supabase_client:
            self.supabase = supabase_client
        elif Client:
            self.supabase = create_client(
                os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_ANON_KEY")
            )
        else:
            self.supabase = None
            print("⚠️  Policy engine operating without database connection")

        # Policy cache
        self.policies: Dict[str, Policy] = {}
        self.policies_by_type: Dict[PolicyType, List[Policy]] = {}
        self.policies_by_scope: Dict[PolicyScope, List[Policy]] = {}

        # Performance metrics
        self.total_evaluations = 0
        self.total_enforcement_actions = 0
        self.compliance_violations = 0

        # Initialize with default policies
        self._initialize_default_policies()

        # Load policies from database
        asyncio.create_task(self._load_policies_from_database())

    def _initialize_default_policies(self) -> None:
        """Initialize default policies for core system functionality"""

        # Default Munger rigor policy
        munger_policy = Policy(
            policy_id="default_munger_rigor",
            name="Default Munger Rigor Policy",
            policy_type=PolicyType.MUNGER_RIGOR,
            scope=PolicyScope.GLOBAL,
            rules=[
                PolicyRule(
                    rule_id="high_impact_l3",
                    name="High Impact Requires L3 Rigor",
                    conditions=[
                        PolicyCondition(
                            "impact", "eq", "high", "High impact decisions"
                        ),
                        PolicyCondition(
                            "reversibility",
                            "eq",
                            "irreversible",
                            "Irreversible decisions",
                        ),
                    ],
                    actions={"rigor_level": "L3", "require_approval": True},
                    priority=PolicyPriority.CRITICAL,
                ),
                PolicyRule(
                    rule_id="security_domain_l3",
                    name="Security Domain Requires L3 Rigor",
                    conditions=[
                        PolicyCondition(
                            "domain", "eq", "security", "Security domain decisions"
                        )
                    ],
                    actions={"rigor_level": "L3", "require_security_review": True},
                    priority=PolicyPriority.HIGH,
                ),
                PolicyRule(
                    rule_id="medium_impact_l2",
                    name="Medium Impact Requires L2 Rigor",
                    conditions=[
                        PolicyCondition(
                            "impact", "eq", "medium", "Medium impact decisions"
                        )
                    ],
                    actions={"rigor_level": "L2"},
                    priority=PolicyPriority.MEDIUM,
                ),
            ],
            default_action={"rigor_level": "L1"},
            max_tokens_overhead=2000,
            max_latency_ms=10000,
        )

        # Default N-way selection policy
        nway_policy = Policy(
            policy_id="default_nway_selection",
            name="Default N-way Interaction Selection Policy",
            policy_type=PolicyType.NWAY_SELECTION,
            scope=PolicyScope.GLOBAL,
            rules=[
                PolicyRule(
                    rule_id="high_lollapalooza_required",
                    name="High-Stakes Decisions Require High Lollapalooza",
                    conditions=[
                        PolicyCondition(
                            "impact", "in", ["high", "critical"], "High/critical impact"
                        ),
                        PolicyCondition(
                            "blast_radius", "eq", "wide", "Wide blast radius"
                        ),
                    ],
                    actions={
                        "min_lollapalooza_threshold": 0.8,
                        "require_multiple_interactions": True,
                    },
                    priority=PolicyPriority.HIGH,
                ),
                PolicyRule(
                    rule_id="standard_lollapalooza",
                    name="Standard Lollapalooza Threshold",
                    conditions=[
                        PolicyCondition(
                            "impact", "eq", "medium", "Medium impact decisions"
                        )
                    ],
                    actions={"min_lollapalooza_threshold": 0.6},
                    priority=PolicyPriority.MEDIUM,
                ),
            ],
            default_action={"min_lollapalooza_threshold": 0.5},
            max_tokens_overhead=1500,
            max_latency_ms=5000,
        )

        # Enterprise compliance policy
        compliance_policy = Policy(
            policy_id="default_enterprise_compliance",
            name="Default Enterprise Compliance Policy",
            policy_type=PolicyType.ENTERPRISE_COMPLIANCE,
            scope=PolicyScope.GLOBAL,
            rules=[
                PolicyRule(
                    rule_id="audit_trail_required",
                    name="All Operations Require Audit Trail",
                    conditions=[
                        PolicyCondition(
                            "operation_type",
                            "in",
                            ["cognitive_analysis", "decision_support"],
                            "Core operations",
                        )
                    ],
                    actions={"require_audit_trail": True, "log_level": "detailed"},
                    priority=PolicyPriority.CRITICAL,
                ),
                PolicyRule(
                    rule_id="sensitive_data_protection",
                    name="Sensitive Data Protection Requirements",
                    conditions=[
                        PolicyCondition(
                            "data_classification",
                            "in",
                            ["sensitive", "confidential"],
                            "Sensitive data",
                        )
                    ],
                    actions={
                        "require_encryption": True,
                        "restrict_access": True,
                        "enhanced_logging": True,
                    },
                    priority=PolicyPriority.CRITICAL,
                ),
            ],
            default_action={"basic_compliance": True},
        )

        # Add to policy cache
        for policy in [munger_policy, nway_policy, compliance_policy]:
            self._add_policy_to_cache(policy)

    def _add_policy_to_cache(self, policy: Policy) -> None:
        """Add policy to internal cache with indexing"""
        self.policies[policy.policy_id] = policy

        # Index by type
        if policy.policy_type not in self.policies_by_type:
            self.policies_by_type[policy.policy_type] = []
        self.policies_by_type[policy.policy_type].append(policy)

        # Index by scope
        if policy.scope not in self.policies_by_scope:
            self.policies_by_scope[policy.scope] = []
        self.policies_by_scope[policy.scope].append(policy)

    async def _load_policies_from_database(self) -> None:
        """Load policies from database"""
        if not self.supabase:
            return

        try:
            # Load munger policies
            munger_result = (
                self.supabase.table("munger_policies")
                .select("*")
                .eq("active", True)
                .execute()
            )
            for row in munger_result.data or []:
                policy = self._convert_db_munger_policy(row)
                if policy:
                    self._add_policy_to_cache(policy)

            print(
                f"✅ Loaded {len(munger_result.data or [])} Munger policies from database"
            )

        except Exception as e:
            print(f"⚠️  Failed to load policies from database: {e}")

    def _convert_db_munger_policy(self, row: Dict[str, Any]) -> Optional[Policy]:
        """Convert database munger policy to Policy object"""
        try:
            # Convert database row to policy rules
            rules = []
            triggers = row.get("triggers", {})

            if isinstance(triggers, dict) and "conditions" in triggers:
                for i, condition in enumerate(triggers["conditions"]):
                    rule = PolicyRule(
                        rule_id=f"db_rule_{row['policy_id']}_{i}",
                        name=f"Rule for {condition}",
                        conditions=[
                            PolicyCondition(
                                field="context_match",
                                operator="contains",
                                value=condition,
                                description=f"Match condition: {condition}",
                            )
                        ],
                        actions={"rigor_level": row["default_rigor_level"]},
                        priority=PolicyPriority.MEDIUM,
                    )
                    rules.append(rule)

            return Policy(
                policy_id=row["policy_id"],
                name=row.get("policy_name", row["policy_id"]),
                policy_type=PolicyType.MUNGER_RIGOR,
                scope=PolicyScope.DOMAIN,
                rules=rules,
                default_action={"rigor_level": row["default_rigor_level"]},
                max_tokens_overhead=row.get("max_tokens_overhead"),
                max_latency_ms=row.get("max_latency_ms"),
                active=row.get("active", True),
            )

        except Exception as e:
            print(f"⚠️  Failed to convert database policy {row.get('policy_id')}: {e}")
            return None

    def _evaluate_condition(
        self, condition: PolicyCondition, context: Dict[str, Any]
    ) -> bool:
        """Evaluate a single policy condition against context"""
        try:
            field_value = context.get(condition.field)

            if condition.operator == "eq":
                return field_value == condition.value
            elif condition.operator == "neq":
                return field_value != condition.value
            elif condition.operator == "in":
                return field_value in condition.value
            elif condition.operator == "not_in":
                return field_value not in condition.value
            elif condition.operator == "gt":
                return field_value > condition.value
            elif condition.operator == "lt":
                return field_value < condition.value
            elif condition.operator == "contains":
                return condition.value in str(field_value)
            elif condition.operator == "context_match":
                # Special operator for database-loaded policies
                return condition.value in str(context)
            else:
                print(f"⚠️  Unknown operator: {condition.operator}")
                return False

        except Exception as e:
            print(f"⚠️  Error evaluating condition {condition.field}: {e}")
            return False

    def _evaluate_rule(
        self, rule: PolicyRule, context: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Evaluate a policy rule against context"""
        if not rule.enabled:
            return False, ["Rule disabled"]

        # All conditions must be true for rule to match
        for condition in rule.conditions:
            if not self._evaluate_condition(condition, context):
                return False, [f"Condition failed: {condition.description}"]

        return True, []

    async def evaluate_policy(
        self, policy: Policy, context: Dict[str, Any]
    ) -> PolicyEvaluation:
        """Evaluate a single policy against context"""
        start_time = datetime.now()

        rule_matches = []
        actions_applied = []
        enforcement_result = "skipped"

        try:
            # Evaluate all rules
            for rule in policy.rules:
                matches, reasons = self._evaluate_rule(rule, context)
                if matches:
                    rule_matches.append(rule.rule_id)
                    actions_applied.append(rule.actions)
                    enforcement_result = "enforced"

            # Apply default action if no rules matched
            if not rule_matches:
                actions_applied.append(policy.default_action)
                enforcement_result = "default"

        except Exception as e:
            print(f"⚠️  Error evaluating policy {policy.policy_id}: {e}")
            enforcement_result = "error"

        end_time = datetime.now()
        evaluation_time_ms = int((end_time - start_time).total_seconds() * 1000)

        return PolicyEvaluation(
            policy_id=policy.policy_id,
            rule_matches=rule_matches,
            actions_applied=actions_applied,
            enforcement_result=enforcement_result,
            evaluation_time_ms=evaluation_time_ms,
        )

    async def evaluate_context(
        self,
        context: Dict[str, Any],
        policy_types: Optional[List[PolicyType]] = None,
        organization_id: Optional[str] = None,
    ) -> PolicyDecision:
        """Evaluate context against all applicable policies"""
        start_time = datetime.now()
        decision_id = f"POLICY-{uuid.uuid4().hex[:8].upper()}"

        # Determine which policies to evaluate
        policies_to_evaluate = []

        if policy_types:
            for policy_type in policy_types:
                policies_to_evaluate.extend(self.policies_by_type.get(policy_type, []))
        else:
            policies_to_evaluate = list(self.policies.values())

        # Filter by organization if specified
        if organization_id:
            policies_to_evaluate = [
                p
                for p in policies_to_evaluate
                if p.organization_id is None or p.organization_id == organization_id
            ]

        # Filter active policies
        policies_to_evaluate = [p for p in policies_to_evaluate if p.active]

        # Evaluate each policy
        policy_evaluations = []
        for policy in policies_to_evaluate:
            evaluation = await self.evaluate_policy(policy, context)
            policy_evaluations.append(evaluation)

        # Consolidate actions from all evaluations
        final_actions = self._consolidate_actions(policy_evaluations)

        # Determine compliance status
        compliance_status = self._determine_compliance_status(policy_evaluations)

        end_time = datetime.now()
        total_evaluation_time_ms = int((end_time - start_time).total_seconds() * 1000)

        # Update metrics
        self.total_evaluations += 1
        if compliance_status == "violation":
            self.compliance_violations += 1
        if final_actions:
            self.total_enforcement_actions += len(final_actions)

        return PolicyDecision(
            decision_id=decision_id,
            context=context,
            policy_evaluations=policy_evaluations,
            final_actions=final_actions,
            compliance_status=compliance_status,
            total_policies_evaluated=len(policies_to_evaluate),
            total_evaluation_time_ms=total_evaluation_time_ms,
            created_at=end_time,
        )

    def _consolidate_actions(
        self, evaluations: List[PolicyEvaluation]
    ) -> Dict[str, Any]:
        """Consolidate actions from multiple policy evaluations"""
        consolidated = {}

        # Priority-based action consolidation
        for evaluation in evaluations:
            for action_set in evaluation.actions_applied:
                for key, value in action_set.items():
                    if key not in consolidated:
                        consolidated[key] = value
                    else:
                        # Handle conflicts based on key type
                        if key == "rigor_level":
                            # Use highest rigor level
                            current = consolidated[key]
                            if self._compare_rigor_levels(value, current) > 0:
                                consolidated[key] = value
                        elif key.startswith("min_"):
                            # Use higher minimum values
                            consolidated[key] = max(consolidated[key], value)
                        elif key.startswith("max_"):
                            # Use lower maximum values
                            consolidated[key] = min(consolidated[key], value)
                        elif key.startswith("require_"):
                            # OR boolean requirements
                            consolidated[key] = consolidated[key] or value
                        else:
                            # Default: last value wins
                            consolidated[key] = value

        return consolidated

    def _compare_rigor_levels(self, level1: str, level2: str) -> int:
        """Compare rigor levels (returns -1, 0, 1)"""
        order = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}
        return order.get(level1, 0) - order.get(level2, 0)

    def _determine_compliance_status(self, evaluations: List[PolicyEvaluation]) -> str:
        """Determine overall compliance status"""
        has_error = any(e.enforcement_result == "error" for e in evaluations)
        has_violation = any(e.enforcement_result == "failed" for e in evaluations)
        has_warning = any(e.enforcement_result == "warning" for e in evaluations)

        if has_error:
            return "error"
        elif has_violation:
            return "violation"
        elif has_warning:
            return "warning"
        else:
            return "compliant"

    async def apply_munger_policy(self, context: DecisionContext) -> Dict[str, Any]:
        """Apply Munger rigor policies to decision context"""
        policy_context = {
            "reversibility": context.reversibility,
            "impact": context.impact,
            "blast_radius": context.blast_radius,
            "domain": context.domain,
            "time_pressure": context.time_pressure,
            "operation_type": "munger_rigor_determination",
        }

        decision = await self.evaluate_context(
            context=policy_context, policy_types=[PolicyType.MUNGER_RIGOR]
        )

        return decision.final_actions

    async def apply_nway_policy(self, request: Any) -> Dict[str, Any]:
        """Apply N-way interaction selection policies"""
        if hasattr(request, "context"):
            policy_context = {
                "impact": (
                    request.context.impact
                    if hasattr(request.context, "impact")
                    else "medium"
                ),
                "domain": (
                    request.context.domain
                    if hasattr(request.context, "domain")
                    else "general"
                ),
                "blast_radius": (
                    request.context.blast_radius
                    if hasattr(request.context, "blast_radius")
                    else "narrow"
                ),
                "operation_type": "nway_interaction_selection",
                "applied_mental_models": getattr(request, "applied_mental_models", []),
            }
        else:
            policy_context = {"operation_type": "nway_interaction_selection"}

        decision = await self.evaluate_context(
            context=policy_context, policy_types=[PolicyType.NWAY_SELECTION]
        )

        return decision.final_actions

    async def save_policy_decision(self, decision: PolicyDecision) -> None:
        """Save policy decision to database for audit trail"""
        if not self.supabase:
            return

        try:
            db_record = {
                "decision_id": decision.decision_id,
                "context": decision.context,
                "compliance_status": decision.compliance_status,
                "total_policies_evaluated": decision.total_policies_evaluated,
                "total_evaluation_time_ms": decision.total_evaluation_time_ms,
                "final_actions": decision.final_actions,
                "policy_evaluations": [asdict(e) for e in decision.policy_evaluations],
            }

            result = self.supabase.table("policy_decisions").insert(db_record).execute()

        except Exception as e:
            print(f"⚠️  Failed to save policy decision: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get policy engine performance metrics"""
        return {
            "total_evaluations": self.total_evaluations,
            "total_enforcement_actions": self.total_enforcement_actions,
            "compliance_violations": self.compliance_violations,
            "violation_rate": self.compliance_violations
            / max(self.total_evaluations, 1),
            "total_policies": len(self.policies),
            "policies_by_type": {
                pt.value: len(self.policies_by_type.get(pt, [])) for pt in PolicyType
            },
            "status": "operational" if self.supabase else "offline_mode",
        }


# Example usage and testing
async def demonstrate_policy_engine():
    """Demonstrate the Policy Engine capabilities"""
    print("⚖️  POLICY ENGINE DEMONSTRATION")
    print("=" * 60)

    # Initialize policy engine
    engine = PolicyEngine()

    # Test Munger rigor policy
    if DecisionContext:
        print("\n🧠 Testing Munger Rigor Policy...")
        context = DecisionContext(
            reversibility="irreversible",
            impact="high",
            blast_radius="wide",
            domain="security",
            time_pressure="medium",
        )

        actions = await engine.apply_munger_policy(context)
        print(f"Munger Policy Actions: {actions}")

    # Test N-way selection policy
    print("\n🔗 Testing N-way Selection Policy...")
    nway_context = {
        "impact": "high",
        "domain": "product",
        "blast_radius": "wide",
        "operation_type": "nway_interaction_selection",
    }

    decision = await engine.evaluate_context(
        context=nway_context, policy_types=[PolicyType.NWAY_SELECTION]
    )

    print(f"N-way Policy Decision: {decision.compliance_status}")
    print(f"Final Actions: {decision.final_actions}")

    # Test enterprise compliance
    print("\n🏢 Testing Enterprise Compliance Policy...")
    compliance_context = {
        "operation_type": "cognitive_analysis",
        "data_classification": "sensitive",
    }

    compliance_decision = await engine.evaluate_context(
        context=compliance_context, policy_types=[PolicyType.ENTERPRISE_COMPLIANCE]
    )

    print(f"Compliance Status: {compliance_decision.compliance_status}")
    print(f"Compliance Actions: {compliance_decision.final_actions}")

    # Show performance metrics
    metrics = engine.get_performance_metrics()
    print("\n📊 POLICY ENGINE METRICS")
    print(f"Total Evaluations: {metrics['total_evaluations']}")
    print(f"Enforcement Actions: {metrics['total_enforcement_actions']}")
    print(f"Violation Rate: {metrics['violation_rate']:.1%}")
    print(f"Total Policies: {metrics['total_policies']}")

    print("\n⚖️  POLICY ENGINE OPERATIONAL!")
    return engine


if __name__ == "__main__":
    # Run demonstration
    asyncio.run(demonstrate_policy_engine())
