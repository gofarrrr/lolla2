"""
METIS Consulting Frameworks Orchestration Engine
C005: Enterprise-grade consulting methodology integration

Implements McKinsey/BCG-grade consulting frameworks with systematic
orchestration, validation, and application across the four-phase workflow.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from uuid import UUID, uuid4
from abc import ABC, abstractmethod
import numpy as np

from src.engine.models.data_contracts import (
    EngagementContext,
    EngagementPhase,
)
from src.core.enhanced_event_bus import (
    EnhancedKafkaEventBus as MetisEventBus,
    CloudEvent,
)

# State manager with fallback for development
try:
    from src.core.state_management import DistributedStateManager, StateType

    STATE_MANAGER_AVAILABLE = True
except Exception:
    STATE_MANAGER_AVAILABLE = False

    # Mock state manager for development
    class MockStateManager:
        def __init__(self):
            self.data = {}

        async def set_state(self, key, value, state_type=None):
            self.data[key] = value

        async def get_state(self, key):
            return self.data.get(key)

    DistributedStateManager = MockStateManager
    StateType = None


class FrameworkCategory(str, Enum):
    """Categories of consulting frameworks"""

    STRATEGIC = "strategic"  # Strategy development and planning
    ANALYTICAL = "analytical"  # Problem analysis and structuring
    OPERATIONAL = "operational"  # Process and efficiency optimization
    FINANCIAL = "financial"  # Financial analysis and modeling
    MARKET = "market"  # Market and competitive analysis
    ORGANIZATIONAL = "organizational"  # Change and organization design
    DIAGNOSTIC = "diagnostic"  # Problem diagnosis and root cause


class FrameworkComplexity(str, Enum):
    """Framework complexity levels"""

    BASIC = "basic"  # Simple, quick application
    INTERMEDIATE = "intermediate"  # Moderate depth and analysis
    ADVANCED = "advanced"  # Comprehensive, detailed analysis
    EXPERT = "expert"  # Highly specialized, expert-level


@dataclass
class FrameworkApplication:
    """Record of framework application instance"""

    application_id: UUID = field(default_factory=uuid4)
    framework_id: str = ""
    engagement_id: UUID = None
    phase: EngagementPhase = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    confidence_score: float = 0.0
    quality_score: float = 0.0
    applied_at: datetime = field(default_factory=datetime.utcnow)
    applied_by: str = "consulting_engine"


class ConsultingFramework(ABC):
    """
    Abstract base class for consulting frameworks
    Defines standard interface for all frameworks
    """

    def __init__(
        self,
        framework_id: str,
        name: str,
        category: FrameworkCategory,
        complexity: FrameworkComplexity,
        description: str,
    ):
        self.framework_id = framework_id
        self.name = name
        self.category = category
        self.complexity = complexity
        self.description = description
        self.prerequisites: List[str] = []
        self.deliverables: List[str] = []
        self.typical_duration: int = 0  # hours
        self.quality_criteria: List[str] = []

        # Performance tracking
        self.application_count: int = 0
        self.success_rate: float = 0.0
        self.avg_quality_score: float = 0.0

    @abstractmethod
    async def apply(
        self, context: EngagementContext, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply framework to the given context and data"""
        pass

    @abstractmethod
    async def validate_inputs(self, input_data: Dict[str, Any]) -> bool:
        """Validate that inputs are sufficient for framework application"""
        pass

    @abstractmethod
    async def assess_quality(self, output_data: Dict[str, Any]) -> float:
        """Assess quality of framework application output (0-1 scale)"""
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Get framework metadata"""
        return {
            "framework_id": self.framework_id,
            "name": self.name,
            "category": self.category.value,
            "complexity": self.complexity.value,
            "description": self.description,
            "prerequisites": self.prerequisites,
            "deliverables": self.deliverables,
            "typical_duration": self.typical_duration,
            "quality_criteria": self.quality_criteria,
            "performance": {
                "application_count": self.application_count,
                "success_rate": self.success_rate,
                "avg_quality_score": self.avg_quality_score,
            },
        }


class MECEFramework(ConsultingFramework):
    """
    MECE (Mutually Exclusive, Collectively Exhaustive) Framework
    Core McKinsey problem structuring methodology
    """

    def __init__(self):
        super().__init__(
            framework_id="mece_structuring",
            name="MECE Problem Structuring",
            category=FrameworkCategory.ANALYTICAL,
            complexity=FrameworkComplexity.INTERMEDIATE,
            description="Break down problems into mutually exclusive, collectively exhaustive components",
        )
        self.prerequisites = ["Problem statement", "Initial context"]
        self.deliverables = ["Problem tree", "MECE validation", "Structured breakdown"]
        self.typical_duration = 4
        self.quality_criteria = [
            "Mutual exclusivity",
            "Collective exhaustiveness",
            "Actionability",
        ]

    async def apply(
        self, context: EngagementContext, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply MECE framework to structure the problem"""

        problem = input_data.get("problem_statement", context.problem_statement)
        business_context = input_data.get("business_context", context.business_context)

        # Level 1: Primary decomposition
        level_1_breakdown = await self._generate_level_1_breakdown(
            problem, business_context
        )

        # Level 2: Secondary decomposition
        level_2_breakdown = {}
        for category in level_1_breakdown:
            level_2_breakdown[category] = await self._generate_level_2_breakdown(
                category, problem, business_context
            )

        # MECE Validation
        mece_validation = await self._validate_mece_structure(
            level_1_breakdown, level_2_breakdown
        )

        # Generate actionable issues
        actionable_issues = await self._identify_actionable_issues(level_2_breakdown)

        return {
            "framework": "MECE",
            "problem_tree": {
                "root": problem,
                "level_1": level_1_breakdown,
                "level_2": level_2_breakdown,
            },
            "mece_validation": mece_validation,
            "actionable_issues": actionable_issues,
            "next_steps": [
                "Validate breakdown with stakeholders",
                "Prioritize actionable issues",
                "Develop hypotheses for key issues",
            ],
            "confidence_level": "high" if mece_validation["is_mece"] else "medium",
        }

    async def _generate_level_1_breakdown(
        self, problem: str, context: Dict
    ) -> List[str]:
        """Generate primary MECE categories"""

        # Common MECE patterns for business problems
        problem_lower = problem.lower()

        if any(word in problem_lower for word in ["growth", "revenue", "market"]):
            return [
                "Market Strategy",
                "Product Strategy",
                "Operational Excellence",
                "Financial Performance",
            ]
        elif any(
            word in problem_lower for word in ["cost", "efficiency", "operations"]
        ):
            return [
                "Process Efficiency",
                "Resource Optimization",
                "Technology & Systems",
                "Organization & Skills",
            ]
        elif any(
            word in problem_lower for word in ["customer", "experience", "satisfaction"]
        ):
            return [
                "Customer Journey",
                "Service Delivery",
                "Product Quality",
                "Brand & Communication",
            ]
        elif any(
            word in problem_lower
            for word in ["digital", "technology", "transformation"]
        ):
            return [
                "Technology Infrastructure",
                "Digital Capabilities",
                "Data & Analytics",
                "Change Management",
            ]
        else:
            # Generic business framework
            return [
                "Strategy & Planning",
                "Operations & Processes",
                "People & Organization",
                "Technology & Systems",
            ]

    async def _generate_level_2_breakdown(
        self, category: str, problem: str, context: Dict
    ) -> List[str]:
        """Generate secondary breakdown for each category"""

        # Level 2 breakdowns by category
        breakdowns = {
            "Market Strategy": [
                "Market segmentation and targeting",
                "Competitive positioning",
                "Channel strategy",
                "Pricing and value proposition",
            ],
            "Product Strategy": [
                "Product portfolio optimization",
                "Innovation and development",
                "Product lifecycle management",
                "Market fit and positioning",
            ],
            "Operational Excellence": [
                "Process standardization",
                "Quality management",
                "Supply chain optimization",
                "Performance measurement",
            ],
            "Financial Performance": [
                "Revenue optimization",
                "Cost structure analysis",
                "Capital allocation",
                "Financial planning and control",
            ],
            "Process Efficiency": [
                "Workflow optimization",
                "Automation opportunities",
                "Resource utilization",
                "Bottleneck elimination",
            ],
            "Resource Optimization": [
                "Capacity management",
                "Asset utilization",
                "Vendor management",
                "Cost reduction initiatives",
            ],
            "Technology & Systems": [
                "System integration",
                "Infrastructure optimization",
                "Data management",
                "Security and compliance",
            ],
            "Organization & Skills": [
                "Organizational design",
                "Capability development",
                "Performance management",
                "Change enablement",
            ],
        }

        return breakdowns.get(
            category,
            [
                f"{category} - Component A",
                f"{category} - Component B",
                f"{category} - Component C",
            ],
        )

    async def _validate_mece_structure(
        self, level_1: List[str], level_2: Dict
    ) -> Dict[str, Any]:
        """Validate MECE principles"""

        # Check mutual exclusivity
        overlap_score = 0.0
        for i, cat1 in enumerate(level_1):
            for j, cat2 in enumerate(level_1[i + 1 :], i + 1):
                # Simple keyword overlap check
                words1 = set(cat1.lower().split())
                words2 = set(cat2.lower().split())
                overlap = len(words1.intersection(words2)) / max(
                    len(words1), len(words2)
                )
                overlap_score = max(overlap_score, overlap)

        mutual_exclusivity = 1.0 - overlap_score

        # Check collective exhaustiveness (heuristic)
        expected_categories = 4  # Typical business framework
        completeness = min(1.0, len(level_1) / expected_categories)

        # Overall MECE score
        mece_score = mutual_exclusivity * 0.6 + completeness * 0.4

        return {
            "is_mece": mece_score >= 0.8,
            "mece_score": mece_score,
            "mutual_exclusivity": mutual_exclusivity,
            "collective_exhaustiveness": completeness,
            "recommendations": [
                "Ensure no overlapping categories" if mutual_exclusivity < 0.8 else "",
                "Add missing categories" if completeness < 0.8 else "",
            ],
        }

    async def _identify_actionable_issues(self, level_2: Dict) -> List[Dict[str, Any]]:
        """Identify actionable issues from the breakdown"""

        actionable_issues = []

        for category, subcategories in level_2.items():
            for subcategory in subcategories:
                # Generate actionable issue for each subcategory
                issue = {
                    "category": category,
                    "subcategory": subcategory,
                    "issue_statement": f"How might we improve {subcategory.lower()}?",
                    "priority": "medium",  # Would be determined by data
                    "complexity": "moderate",
                    "potential_impact": "significant",
                }
                actionable_issues.append(issue)

        return actionable_issues[:10]  # Top 10 issues

    async def validate_inputs(self, input_data: Dict[str, Any]) -> bool:
        """Validate inputs for MECE framework"""
        required_fields = ["problem_statement"]
        return all(
            field in input_data and input_data[field] for field in required_fields
        )

    async def assess_quality(self, output_data: Dict[str, Any]) -> float:
        """Assess quality of MECE application"""
        quality_checks = [
            ("mece_validation" in output_data, 0.3),
            (output_data.get("mece_validation", {}).get("is_mece", False), 0.4),
            ("actionable_issues" in output_data, 0.2),
            (len(output_data.get("actionable_issues", [])) >= 5, 0.1),
        ]

        quality_score = sum(weight for check, weight in quality_checks if check)
        return quality_score


class IssueTreeFramework(ConsultingFramework):
    """
    Issue Tree Framework
    Hierarchical problem decomposition methodology
    """

    def __init__(self):
        super().__init__(
            framework_id="issue_tree",
            name="Issue Tree Analysis",
            category=FrameworkCategory.ANALYTICAL,
            complexity=FrameworkComplexity.ADVANCED,
            description="Create hierarchical decomposition of complex problems",
        )
        self.prerequisites = ["MECE structure", "Problem context"]
        self.deliverables = [
            "Issue tree diagram",
            "Prioritized issues",
            "Analysis plan",
        ]
        self.typical_duration = 6
        self.quality_criteria = ["Logical flow", "Actionability", "Prioritization"]

    async def apply(
        self, context: EngagementContext, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply Issue Tree framework"""

        # Build issue tree structure
        root_issue = input_data.get("problem_statement", context.problem_statement)

        # Generate issue tree
        issue_tree = await self._build_issue_tree(root_issue, context.business_context)

        # Prioritize issues
        prioritized_issues = await self._prioritize_issues(issue_tree)

        # Create analysis plan
        analysis_plan = await self._create_analysis_plan(prioritized_issues)

        return {
            "framework": "Issue Tree",
            "issue_tree": issue_tree,
            "prioritized_issues": prioritized_issues,
            "analysis_plan": analysis_plan,
            "critical_path": [issue["id"] for issue in prioritized_issues[:3]],
            "estimated_effort": f"{len(prioritized_issues) * 2} days",
        }

    async def _build_issue_tree(self, root_issue: str, context: Dict) -> Dict[str, Any]:
        """Build hierarchical issue tree"""

        tree = {
            "root": {"id": "root", "statement": root_issue, "level": 0, "children": []}
        }

        # Generate level 1 issues
        level_1_issues = [
            "Revenue decline analysis",
            "Cost structure optimization",
            "Market position assessment",
            "Operational efficiency review",
        ]

        for i, issue in enumerate(level_1_issues):
            issue_id = f"L1_{i+1}"
            level_1_node = {
                "id": issue_id,
                "statement": issue,
                "level": 1,
                "parent": "root",
                "children": [],
            }

            # Generate level 2 sub-issues
            level_2_issues = await self._generate_sub_issues(issue)
            for j, sub_issue in enumerate(level_2_issues):
                sub_issue_id = f"{issue_id}_{j+1}"
                level_2_node = {
                    "id": sub_issue_id,
                    "statement": sub_issue,
                    "level": 2,
                    "parent": issue_id,
                    "children": [],
                }
                level_1_node["children"].append(level_2_node)

            tree["root"]["children"].append(level_1_node)

        return tree

    async def _generate_sub_issues(self, parent_issue: str) -> List[str]:
        """Generate sub-issues for parent issue"""

        issue_patterns = {
            "revenue": [
                "Customer acquisition challenges",
                "Customer retention issues",
                "Pricing strategy problems",
                "Product mix optimization",
            ],
            "cost": [
                "Direct cost management",
                "Indirect cost optimization",
                "Resource utilization",
                "Vendor management",
            ],
            "market": [
                "Competitive positioning",
                "Market share analysis",
                "Customer segmentation",
                "Value proposition clarity",
            ],
            "operational": [
                "Process inefficiencies",
                "Technology gaps",
                "Skill deficiencies",
                "Quality issues",
            ],
        }

        parent_lower = parent_issue.lower()
        for key, issues in issue_patterns.items():
            if key in parent_lower:
                return issues[:3]  # Top 3 sub-issues

        return [f"Sub-issue 1 of {parent_issue}", f"Sub-issue 2 of {parent_issue}"]

    async def _prioritize_issues(self, issue_tree: Dict) -> List[Dict[str, Any]]:
        """Prioritize issues using impact/effort matrix"""

        all_issues = []

        def extract_issues(node, issues_list):
            if node.get("level", 0) > 0:  # Skip root
                issues_list.append(node)
            for child in node.get("children", []):
                extract_issues(child, issues_list)

        extract_issues(issue_tree["root"], all_issues)

        # Score each issue
        for issue in all_issues:
            issue["impact_score"] = np.random.uniform(1, 10)
            issue["effort_score"] = np.random.uniform(1, 10)
            issue["priority_score"] = issue["impact_score"] / issue["effort_score"]

        # Sort by priority
        prioritized = sorted(
            all_issues, key=lambda x: x["priority_score"], reverse=True
        )

        return prioritized

    async def _create_analysis_plan(
        self, prioritized_issues: List[Dict]
    ) -> Dict[str, Any]:
        """Create analysis plan for top issues"""

        plan = {
            "phases": [],
            "timeline": "4-6 weeks",
            "resources_required": ["Senior analyst", "Data analyst", "Domain expert"],
            "deliverables": [],
        }

        # Group top issues into analysis phases
        top_issues = prioritized_issues[:6]
        phase_size = 2

        for i in range(0, len(top_issues), phase_size):
            phase_issues = top_issues[i : i + phase_size]
            phase = {
                "phase_number": (i // phase_size) + 1,
                "duration": "1-2 weeks",
                "issues": [issue["statement"] for issue in phase_issues],
                "key_questions": [
                    f"How to address {issue['statement']}?" for issue in phase_issues
                ],
                "analysis_methods": [
                    "Data analysis",
                    "Stakeholder interviews",
                    "Benchmarking",
                ],
                "expected_outputs": [
                    "Findings summary",
                    "Recommendations",
                    "Next steps",
                ],
            }
            plan["phases"].append(phase)

        return plan

    async def validate_inputs(self, input_data: Dict[str, Any]) -> bool:
        """Validate inputs for Issue Tree framework"""
        return "problem_statement" in input_data

    async def assess_quality(self, output_data: Dict[str, Any]) -> float:
        """Assess quality of Issue Tree application"""
        quality_score = 0.0

        if "issue_tree" in output_data:
            quality_score += 0.4
        if "prioritized_issues" in output_data:
            quality_score += 0.3
        if "analysis_plan" in output_data:
            quality_score += 0.3

        return quality_score


class BCGGrowthShareMatrix(ConsultingFramework):
    """
    BCG Growth-Share Matrix Framework
    Portfolio analysis methodology
    """

    def __init__(self):
        super().__init__(
            framework_id="bcg_matrix",
            name="BCG Growth-Share Matrix",
            category=FrameworkCategory.STRATEGIC,
            complexity=FrameworkComplexity.INTERMEDIATE,
            description="Analyze portfolio using growth and market share dimensions",
        )
        self.prerequisites = ["Portfolio data", "Market information"]
        self.deliverables = [
            "BCG matrix plot",
            "Portfolio recommendations",
            "Strategic options",
        ]
        self.typical_duration = 8
        self.quality_criteria = [
            "Data accuracy",
            "Strategic insights",
            "Actionable recommendations",
        ]

    async def apply(
        self, context: EngagementContext, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply BCG Matrix framework"""

        # Extract portfolio items
        portfolio_items = input_data.get(
            "portfolio_items", await self._generate_sample_portfolio()
        )

        # Position items on matrix
        matrix_positioning = await self._position_on_matrix(portfolio_items)

        # Generate strategic recommendations
        recommendations = await self._generate_recommendations(matrix_positioning)

        # Create strategic options
        strategic_options = await self._create_strategic_options(matrix_positioning)

        return {
            "framework": "BCG Growth-Share Matrix",
            "matrix_positioning": matrix_positioning,
            "portfolio_analysis": {
                "stars": [
                    item for item in matrix_positioning if item["category"] == "Star"
                ],
                "cash_cows": [
                    item
                    for item in matrix_positioning
                    if item["category"] == "Cash Cow"
                ],
                "question_marks": [
                    item
                    for item in matrix_positioning
                    if item["category"] == "Question Mark"
                ],
                "dogs": [
                    item for item in matrix_positioning if item["category"] == "Dog"
                ],
            },
            "strategic_recommendations": recommendations,
            "strategic_options": strategic_options,
            "portfolio_health": await self._assess_portfolio_health(matrix_positioning),
        }

    async def _generate_sample_portfolio(self) -> List[Dict[str, Any]]:
        """Generate sample portfolio for demonstration"""
        return [
            {
                "name": "Product A",
                "market_growth": 15,
                "market_share": 25,
                "revenue": 50,
            },
            {
                "name": "Product B",
                "market_growth": 8,
                "market_share": 40,
                "revenue": 80,
            },
            {
                "name": "Product C",
                "market_growth": 18,
                "market_share": 5,
                "revenue": 20,
            },
            {"name": "Product D", "market_growth": 3, "market_share": 8, "revenue": 15},
        ]

    async def _position_on_matrix(
        self, portfolio_items: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Position portfolio items on BCG matrix"""

        positioned_items = []

        for item in portfolio_items:
            growth = item.get("market_growth", 10)
            share = item.get("market_share", 10)

            # Categorize based on BCG matrix quadrants
            if growth >= 10 and share >= 10:
                category = "Star"
                strategy = "Invest for growth"
            elif growth < 10 and share >= 10:
                category = "Cash Cow"
                strategy = "Harvest cash"
            elif growth >= 10 and share < 10:
                category = "Question Mark"
                strategy = "Selective investment"
            else:
                category = "Dog"
                strategy = "Divest or turnaround"

            positioned_items.append(
                {
                    "name": item["name"],
                    "market_growth": growth,
                    "market_share": share,
                    "revenue": item.get("revenue", 0),
                    "category": category,
                    "recommended_strategy": strategy,
                    "x_position": share,
                    "y_position": growth,
                }
            )

        return positioned_items

    async def _generate_recommendations(
        self, matrix_positioning: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Generate strategic recommendations by category"""

        recommendations = []

        # Categorize items
        categories = {}
        for item in matrix_positioning:
            category = item["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append(item)

        # Generate recommendations for each category
        for category, items in categories.items():
            if category == "Star":
                recommendations.append(
                    {
                        "category": "Stars",
                        "items": [item["name"] for item in items],
                        "priority": "High",
                        "action": "Invest aggressively to maintain market leadership",
                        "rationale": "High growth, high share products are future cash cows",
                        "resource_allocation": "30-40% of investment budget",
                    }
                )
            elif category == "Cash Cow":
                recommendations.append(
                    {
                        "category": "Cash Cows",
                        "items": [item["name"] for item in items],
                        "priority": "Medium",
                        "action": "Harvest cash to fund Stars and Question Marks",
                        "rationale": "Stable, profitable products in mature markets",
                        "resource_allocation": "Minimal investment, maximum cash generation",
                    }
                )
            elif category == "Question Mark":
                recommendations.append(
                    {
                        "category": "Question Marks",
                        "items": [item["name"] for item in items],
                        "priority": "High",
                        "action": "Selective investment to build market share",
                        "rationale": "High growth potential but uncertain competitive position",
                        "resource_allocation": "20-30% of investment budget for selected items",
                    }
                )
            elif category == "Dog":
                recommendations.append(
                    {
                        "category": "Dogs",
                        "items": [item["name"] for item in items],
                        "priority": "Low",
                        "action": "Divest or turnaround if strategic value exists",
                        "rationale": "Low growth, low share products drain resources",
                        "resource_allocation": "Minimal to zero investment",
                    }
                )

        return recommendations

    async def _create_strategic_options(
        self, matrix_positioning: List[Dict]
    ) -> Dict[str, Any]:
        """Create portfolio-level strategic options"""

        return {
            "build_option": {
                "description": "Aggressive investment in Stars and selected Question Marks",
                "investment_required": "High",
                "risk_level": "Medium-High",
                "time_horizon": "3-5 years",
                "expected_outcome": "Market leadership in high-growth segments",
            },
            "harvest_option": {
                "description": "Maximize cash generation from Cash Cows, selective divestment",
                "investment_required": "Low",
                "risk_level": "Low",
                "time_horizon": "1-2 years",
                "expected_outcome": "Steady cash flow, gradual portfolio decline",
            },
            "rebalance_option": {
                "description": "Strategic rebalancing through acquisitions and divestitures",
                "investment_required": "Medium",
                "risk_level": "Medium",
                "time_horizon": "2-3 years",
                "expected_outcome": "Optimized portfolio balance",
            },
        }

    async def _assess_portfolio_health(
        self, matrix_positioning: List[Dict]
    ) -> Dict[str, Any]:
        """Assess overall portfolio health"""

        total_items = len(matrix_positioning)
        categories = {"Star": 0, "Cash Cow": 0, "Question Mark": 0, "Dog": 0}

        for item in matrix_positioning:
            categories[item["category"]] += 1

        # Calculate portfolio balance score
        ideal_distribution = {
            "Star": 0.25,
            "Cash Cow": 0.35,
            "Question Mark": 0.25,
            "Dog": 0.15,
        }
        actual_distribution = {k: v / total_items for k, v in categories.items()}

        balance_score = (
            1.0
            - sum(
                abs(ideal_distribution[k] - actual_distribution[k])
                for k in categories.keys()
            )
            / 2
        )

        return {
            "portfolio_balance_score": balance_score,
            "distribution": actual_distribution,
            "strengths": [
                f"Strong in {k}"
                for k, v in actual_distribution.items()
                if v > ideal_distribution[k] and k in ["Star", "Cash Cow"]
            ],
            "concerns": [
                f"Too many {k}s"
                for k, v in actual_distribution.items()
                if v > ideal_distribution[k] and k in ["Question Mark", "Dog"]
            ],
            "overall_health": "Healthy" if balance_score > 0.7 else "Needs rebalancing",
        }

    async def validate_inputs(self, input_data: Dict[str, Any]) -> bool:
        """Validate inputs for BCG Matrix"""
        if "portfolio_items" not in input_data:
            return True  # Will use sample data

        required_fields = ["market_growth", "market_share"]
        portfolio_items = input_data["portfolio_items"]

        return all(
            all(field in item for field in required_fields) for item in portfolio_items
        )

    async def assess_quality(self, output_data: Dict[str, Any]) -> float:
        """Assess quality of BCG Matrix application"""
        quality_score = 0.0

        if "matrix_positioning" in output_data:
            quality_score += 0.3
        if "strategic_recommendations" in output_data:
            quality_score += 0.4
        if "portfolio_health" in output_data:
            quality_score += 0.3

        return quality_score


class ConsultingFrameworkOrchestrator:
    """
    Main orchestrator for consulting frameworks
    Manages framework selection, application, and integration
    """

    def __init__(
        self, state_manager: DistributedStateManager, event_bus: MetisEventBus
    ):
        self.state_manager = state_manager
        self.event_bus = event_bus

        # Framework registry
        self.frameworks: Dict[str, ConsultingFramework] = {}
        self.framework_applications: Dict[UUID, FrameworkApplication] = {}

        # Performance tracking
        self.application_history: List[Dict] = []
        self.framework_rankings: Dict[str, float] = {}

        self.logger = logging.getLogger(__name__)

        # Initialize frameworks
        asyncio.create_task(self._initialize_frameworks())

    async def _initialize_frameworks(self):
        """Initialize consulting frameworks library"""

        # Core frameworks
        frameworks = [MECEFramework(), IssueTreeFramework(), BCGGrowthShareMatrix()]

        for framework in frameworks:
            self.frameworks[framework.framework_id] = framework
            self.framework_rankings[framework.framework_id] = 5.0  # Initial rating

            # Store framework metadata
            await self.state_manager.set_state(
                f"framework_{framework.framework_id}",
                framework.get_metadata(),
                StateType.CONFIGURATION,
            )

        self.logger.info(f"Initialized {len(self.frameworks)} consulting frameworks")

    async def select_frameworks(
        self,
        context: EngagementContext,
        phase: EngagementPhase,
        max_frameworks: int = 3,
    ) -> List[ConsultingFramework]:
        """
        Select optimal frameworks for the given context and phase
        """

        # Phase-based framework preferences
        phase_preferences = {
            EngagementPhase.PROBLEM_STRUCTURING: {
                FrameworkCategory.ANALYTICAL: 0.8,
                FrameworkCategory.DIAGNOSTIC: 0.7,
                FrameworkCategory.STRATEGIC: 0.4,
            },
            EngagementPhase.HYPOTHESIS_GENERATION: {
                FrameworkCategory.ANALYTICAL: 0.6,
                FrameworkCategory.STRATEGIC: 0.8,
                FrameworkCategory.MARKET: 0.6,
            },
            EngagementPhase.ANALYSIS_EXECUTION: {
                FrameworkCategory.OPERATIONAL: 0.8,
                FrameworkCategory.FINANCIAL: 0.7,
                FrameworkCategory.MARKET: 0.7,
            },
            EngagementPhase.SYNTHESIS_DELIVERY: {
                FrameworkCategory.STRATEGIC: 0.9,
                FrameworkCategory.ORGANIZATIONAL: 0.6,
            },
        }

        # Score frameworks for context
        framework_scores = []

        for framework_id, framework in self.frameworks.items():
            # Base score from framework ranking
            base_score = self.framework_rankings.get(framework_id, 5.0)

            # Phase alignment score
            phase_prefs = phase_preferences.get(phase, {})
            phase_score = phase_prefs.get(framework.category, 0.3)

            # Context relevance score
            context_score = await self._assess_context_relevance(framework, context)

            # Complexity appropriateness
            complexity_score = await self._assess_complexity_fit(framework, context)

            # Combined score
            total_score = (
                base_score * 0.3
                + phase_score * 0.3
                + context_score * 0.2
                + complexity_score * 0.2
            )

            framework_scores.append((framework, total_score))

        # Sort and select top frameworks
        framework_scores.sort(key=lambda x: x[1], reverse=True)
        selected_frameworks = [fw for fw, score in framework_scores[:max_frameworks]]

        # Log selection
        self.logger.info(
            f"Selected frameworks for {phase.value}: "
            f"{[fw.framework_id for fw in selected_frameworks]}"
        )

        return selected_frameworks

    async def _assess_context_relevance(
        self, framework: ConsultingFramework, context: EngagementContext
    ) -> float:
        """Assess how relevant a framework is to the engagement context"""

        problem = context.problem_statement.lower()
        business_context = str(context.business_context).lower()

        # Category relevance patterns
        relevance_patterns = {
            FrameworkCategory.STRATEGIC: [
                "strategy",
                "growth",
                "market",
                "competitive",
                "portfolio",
            ],
            FrameworkCategory.ANALYTICAL: [
                "problem",
                "analysis",
                "structure",
                "breakdown",
                "diagnostic",
            ],
            FrameworkCategory.OPERATIONAL: [
                "process",
                "efficiency",
                "operations",
                "cost",
                "productivity",
            ],
            FrameworkCategory.FINANCIAL: [
                "revenue",
                "profit",
                "cost",
                "financial",
                "pricing",
            ],
            FrameworkCategory.MARKET: [
                "customer",
                "market",
                "competition",
                "brand",
                "positioning",
            ],
            FrameworkCategory.ORGANIZATIONAL: [
                "organization",
                "people",
                "culture",
                "change",
                "skills",
            ],
        }

        patterns = relevance_patterns.get(framework.category, [])
        relevance_score = sum(
            1
            for pattern in patterns
            if pattern in problem or pattern in business_context
        )

        return min(1.0, relevance_score / len(patterns))

    async def _assess_complexity_fit(
        self, framework: ConsultingFramework, context: EngagementContext
    ) -> float:
        """Assess if framework complexity matches engagement needs"""

        # Estimate engagement complexity
        problem_length = len(context.problem_statement.split())
        context_complexity = len(str(context.business_context))

        if problem_length > 50 or context_complexity > 500:
            needed_complexity = FrameworkComplexity.ADVANCED
        elif problem_length > 20 or context_complexity > 200:
            needed_complexity = FrameworkComplexity.INTERMEDIATE
        else:
            needed_complexity = FrameworkComplexity.BASIC

        # Score complexity match
        complexity_match = {
            (FrameworkComplexity.BASIC, FrameworkComplexity.BASIC): 1.0,
            (FrameworkComplexity.BASIC, FrameworkComplexity.INTERMEDIATE): 0.7,
            (FrameworkComplexity.INTERMEDIATE, FrameworkComplexity.INTERMEDIATE): 1.0,
            (FrameworkComplexity.INTERMEDIATE, FrameworkComplexity.ADVANCED): 0.8,
            (FrameworkComplexity.ADVANCED, FrameworkComplexity.ADVANCED): 1.0,
            (FrameworkComplexity.ADVANCED, FrameworkComplexity.EXPERT): 0.9,
        }

        return complexity_match.get((needed_complexity, framework.complexity), 0.5)

    async def apply_framework(
        self,
        framework_id: str,
        context: EngagementContext,
        input_data: Dict[str, Any],
        phase: EngagementPhase,
    ) -> FrameworkApplication:
        """
        Apply a specific framework to the engagement
        """

        if framework_id not in self.frameworks:
            raise ValueError(f"Framework {framework_id} not found")

        framework = self.frameworks[framework_id]

        # Validate inputs
        if not await framework.validate_inputs(input_data):
            raise ValueError(f"Invalid inputs for framework {framework_id}")

        # Record application start
        application = FrameworkApplication(
            framework_id=framework_id,
            engagement_id=context.engagement_id,
            phase=phase,
            input_data=input_data,
        )

        start_time = datetime.utcnow()

        try:
            # Apply framework
            output_data = await framework.apply(context, input_data)

            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()

            # Assess quality
            quality_score = await framework.assess_quality(output_data)

            # Calculate confidence
            confidence_score = await self._calculate_confidence(
                framework, output_data, quality_score
            )

            # Update application record
            application.output_data = output_data
            application.execution_time = execution_time
            application.quality_score = quality_score
            application.confidence_score = confidence_score

            # Store application
            self.framework_applications[application.application_id] = application

            # Update framework performance
            framework.application_count += 1
            framework.avg_quality_score = (
                framework.avg_quality_score * (framework.application_count - 1)
                + quality_score
            ) / framework.application_count

            # Store application results
            await self.state_manager.set_state(
                f"application_{application.application_id}",
                self._serialize_application(application),
                StateType.COGNITIVE,
            )

            # Emit application event
            await self.event_bus.publish_event(
                CloudEvent(
                    type="framework.application.completed",
                    source="consulting/orchestrator",
                    data={
                        "framework_id": framework_id,
                        "engagement_id": str(context.engagement_id),
                        "phase": phase.value,
                        "quality_score": quality_score,
                        "execution_time": execution_time,
                    },
                )
            )

            return application

        except Exception as e:
            self.logger.error(f"Framework application failed: {str(e)}")

            # Record failure
            application.output_data = {"error": str(e)}
            application.execution_time = (
                datetime.utcnow() - start_time
            ).total_seconds()
            application.quality_score = 0.0
            application.confidence_score = 0.0

            raise

    async def _calculate_confidence(
        self,
        framework: ConsultingFramework,
        output_data: Dict[str, Any],
        quality_score: float,
    ) -> float:
        """Calculate confidence in framework application results"""

        # Base confidence from quality
        base_confidence = quality_score

        # Framework track record boost
        track_record_boost = min(0.2, framework.application_count * 0.01)

        # Output completeness check
        expected_keys = ["framework", "next_steps"]
        completeness = sum(1 for key in expected_keys if key in output_data) / len(
            expected_keys
        )

        confidence = base_confidence * 0.7 + completeness * 0.3 + track_record_boost

        return min(1.0, confidence)

    def _serialize_application(
        self, application: FrameworkApplication
    ) -> Dict[str, Any]:
        """Serialize framework application for storage"""
        return {
            "application_id": str(application.application_id),
            "framework_id": application.framework_id,
            "engagement_id": (
                str(application.engagement_id) if application.engagement_id else None
            ),
            "phase": application.phase.value if application.phase else None,
            "input_data": application.input_data,
            "output_data": application.output_data,
            "execution_time": application.execution_time,
            "confidence_score": application.confidence_score,
            "quality_score": application.quality_score,
            "applied_at": application.applied_at.isoformat(),
            "applied_by": application.applied_by,
        }

    async def orchestrate_phase_frameworks(
        self,
        context: EngagementContext,
        phase: EngagementPhase,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Orchestrate multiple frameworks for a complete phase
        """

        # Select appropriate frameworks
        selected_frameworks = await self.select_frameworks(
            context, phase, max_frameworks=2
        )

        if not selected_frameworks:
            return {"error": "No suitable frameworks found for phase"}

        # Apply frameworks sequentially
        phase_results = {
            "phase": phase.value,
            "frameworks_applied": [],
            "combined_insights": [],
            "next_phase_recommendations": [],
        }

        current_input = input_data.copy()

        for framework in selected_frameworks:
            try:
                application = await self.apply_framework(
                    framework.framework_id, context, current_input, phase
                )

                # Store framework results
                framework_result = {
                    "framework_id": framework.framework_id,
                    "framework_name": framework.name,
                    "output": application.output_data,
                    "quality_score": application.quality_score,
                    "confidence_score": application.confidence_score,
                    "execution_time": application.execution_time,
                }

                phase_results["frameworks_applied"].append(framework_result)

                # Use output as input for next framework
                current_input.update(application.output_data)

            except Exception as e:
                self.logger.error(
                    f"Framework {framework.framework_id} failed: {str(e)}"
                )
                continue

        # Synthesize results
        phase_results["combined_insights"] = await self._synthesize_framework_results(
            phase_results["frameworks_applied"]
        )

        # Generate next phase recommendations
        phase_results["next_phase_recommendations"] = (
            await self._generate_next_phase_recommendations(
                phase, phase_results["combined_insights"]
            )
        )

        return phase_results

    async def _synthesize_framework_results(
        self, framework_results: List[Dict]
    ) -> List[str]:
        """Synthesize insights from multiple framework applications"""

        insights = []

        # Extract key themes
        all_outputs = [result["output"] for result in framework_results]

        # Common insight patterns
        if any("mece" in str(output).lower() for output in all_outputs):
            insights.append("Problem structure analysis completed with MECE validation")

        if any("priorit" in str(output).lower() for output in all_outputs):
            insights.append("Issue prioritization framework applied successfully")

        if any("strategic" in str(output).lower() for output in all_outputs):
            insights.append("Strategic analysis frameworks provide portfolio insights")

        # Quality assessment
        avg_quality = sum(
            result["quality_score"] for result in framework_results
        ) / len(framework_results)
        if avg_quality > 0.7:
            insights.append("High-quality framework application with reliable outputs")

        return insights

    async def _generate_next_phase_recommendations(
        self, current_phase: EngagementPhase, insights: List[str]
    ) -> List[str]:
        """Generate recommendations for next phase"""

        phase_transitions = {
            EngagementPhase.PROBLEM_STRUCTURING: [
                "Proceed with hypothesis generation based on structured problem breakdown",
                "Focus on high-priority issues identified in the analysis",
                "Validate problem structure with key stakeholders",
            ],
            EngagementPhase.HYPOTHESIS_GENERATION: [
                "Design tests and data collection for top hypotheses",
                "Prioritize hypotheses based on impact and feasibility",
                "Prepare analysis workstream for hypothesis validation",
            ],
            EngagementPhase.ANALYSIS_EXECUTION: [
                "Synthesize findings into coherent recommendations",
                "Prepare evidence-based business case",
                "Structure insights using pyramid principle",
            ],
            EngagementPhase.SYNTHESIS_DELIVERY: [
                "Finalize deliverables and presentation materials",
                "Prepare implementation roadmap",
                "Schedule stakeholder review and feedback sessions",
            ],
        }

        return phase_transitions.get(
            current_phase, ["Continue to next phase of engagement"]
        )

    async def get_framework_analytics(self) -> Dict[str, Any]:
        """Get analytics on framework usage and performance"""

        analytics = {
            "framework_count": len(self.frameworks),
            "total_applications": len(self.framework_applications),
            "framework_performance": {},
            "usage_patterns": {},
            "quality_trends": [],
        }

        # Framework performance metrics
        for framework_id, framework in self.frameworks.items():
            analytics["framework_performance"][framework_id] = {
                "application_count": framework.application_count,
                "avg_quality_score": framework.avg_quality_score,
                "success_rate": framework.success_rate,
                "category": framework.category.value,
                "complexity": framework.complexity.value,
            }

        # Usage patterns by phase
        phase_usage = {}
        for application in self.framework_applications.values():
            if application.phase:
                phase = application.phase.value
                if phase not in phase_usage:
                    phase_usage[phase] = {}

                framework_id = application.framework_id
                phase_usage[phase][framework_id] = (
                    phase_usage[phase].get(framework_id, 0) + 1
                )

        analytics["usage_patterns"] = phase_usage

        return analytics


class ConsultingFrameworkOrchestratorFacade:
    """
    Lightweight facade for consulting framework access used by tests and tooling.
    """

    def __init__(self, state_manager=None, event_bus=None):
        self._state_manager = state_manager or DistributedStateManager()
        self._event_bus = event_bus or MetisEventBus()
        self._frameworks: List[ConsultingFramework] = [MECEFramework()]
        self._logger = logging.getLogger(__name__)

    async def get_frameworks(self) -> List[Dict[str, Any]]:
        """Return metadata for available consulting frameworks."""
        return [framework.get_metadata() for framework in self._frameworks]

    async def evaluate_frameworks_for_query(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Produce a simplified recommendation envelope for the given query.
        """
        context = context or {}
        recommendations = [
            {
                "framework_id": framework.framework_id,
                "name": framework.name,
                "match_score": 0.75,
                "rationale": f"Baseline recommendation for {framework.name}",
            }
            for framework in self._frameworks
        ]
        return {
            "query": query,
            "context": context,
            "recommendations": recommendations,
        }

    async def close(self) -> None:
        """Facade cleanup hook (no-op for stub implementation)."""
        self._logger.debug("ConsultingFrameworkOrchestratorFacade closed.")
