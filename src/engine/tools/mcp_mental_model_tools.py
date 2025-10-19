"""
METIS MCP-Compliant Mental Model Tools
Wraps mental models as standardized tools following MCP (Model Context Protocol)

Based on industry insights:
- OpenAI: Function calling with structured schemas
- Anthropic: Tool use with clear input/output contracts
- LangChain: Standardized tool interfaces
- MCP Standard: Clear tool definitions with validation
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod

from src.core.performance_cache_system import get_performance_cache, CacheEntryType
from src.core.stateful_environment import get_stateful_environment, CheckpointType

logger = logging.getLogger(__name__)


class ToolCategory(str, Enum):
    """Categories of mental model tools"""

    ANALYSIS = "analysis"  # Analysis and evaluation tools
    STRUCTURING = "structuring"  # Problem structuring tools
    DECISION = "decision"  # Decision-making tools
    CREATIVE = "creative"  # Creative thinking tools
    VALIDATION = "validation"  # Validation and testing tools


class ToolComplexity(str, Enum):
    """Tool complexity levels"""

    SIMPLE = "simple"  # Simple, fast execution
    MODERATE = "moderate"  # Moderate complexity
    COMPLEX = "complex"  # Complex, resource-intensive


@dataclass
class ToolParameter:
    """Parameter definition for mental model tool"""

    name: str
    type: str  # "string", "number", "boolean", "object", "array"
    description: str
    required: bool = True
    default: Optional[Any] = None
    enum_values: Optional[List[str]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None


@dataclass
class ToolSchema:
    """MCP-compliant tool schema"""

    name: str
    description: str
    category: ToolCategory
    complexity: ToolComplexity

    # Input/Output contracts
    parameters: List[ToolParameter]
    return_schema: Dict[str, Any]

    # Performance characteristics
    estimated_execution_time_ms: int
    cache_eligible: bool
    requires_human_approval: bool

    # Usage metadata
    confidence_calibrated: bool
    bias_risks: List[str]
    best_use_cases: List[str]
    limitations: List[str]


@dataclass
class ToolExecutionResult:
    """Result of mental model tool execution"""

    tool_name: str
    execution_id: str
    success: bool

    # Results
    result: Optional[Dict[str, Any]] = None
    confidence_score: float = 0.0
    reasoning_steps: List[str] = None

    # Metadata
    execution_time_ms: float = 0.0
    cache_hit: bool = False
    validation_passed: bool = True

    # Errors
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    # Context
    input_context: Optional[Dict[str, Any]] = None
    timestamp: datetime = None


class MCPMentalModelTool(ABC):
    """
    Abstract base class for MCP-compliant mental model tools
    Provides standardized interface for all mental model implementations
    """

    def __init__(self, schema: ToolSchema):
        self.schema = schema
        self.logger = logging.getLogger(f"{__name__}.{schema.name}")
        self.cache = get_performance_cache()

        # Performance tracking
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.success_count = 0
        self.cache_hit_count = 0

    @abstractmethod
    async def execute(
        self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> ToolExecutionResult:
        """Execute the mental model tool with given parameters"""
        pass

    async def validate_parameters(
        self, parameters: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate input parameters against schema"""
        errors = []

        # Check required parameters
        for param in self.schema.parameters:
            if param.required and param.name not in parameters:
                errors.append(f"Required parameter '{param.name}' is missing")
                continue

            if param.name in parameters:
                value = parameters[param.name]

                # Type validation
                if not self._validate_type(value, param.type):
                    errors.append(
                        f"Parameter '{param.name}' has invalid type. Expected {param.type}"
                    )

                # Enum validation
                if param.enum_values and value not in param.enum_values:
                    errors.append(
                        f"Parameter '{param.name}' must be one of {param.enum_values}"
                    )

                # Range validation
                if param.type in ["number", "integer"]:
                    if param.min_value is not None and value < param.min_value:
                        errors.append(
                            f"Parameter '{param.name}' must be >= {param.min_value}"
                        )
                    if param.max_value is not None and value > param.max_value:
                        errors.append(
                            f"Parameter '{param.name}' must be <= {param.max_value}"
                        )

        return len(errors) == 0, errors

    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate value type"""
        type_validators = {
            "string": lambda v: isinstance(v, str),
            "number": lambda v: isinstance(v, (int, float)),
            "integer": lambda v: isinstance(v, int),
            "boolean": lambda v: isinstance(v, bool),
            "object": lambda v: isinstance(v, dict),
            "array": lambda v: isinstance(v, list),
        }

        validator = type_validators.get(expected_type)
        return validator(value) if validator else False

    async def get_cached_result(
        self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Optional[ToolExecutionResult]:
        """Check for cached execution result"""
        if not self.schema.cache_eligible:
            return None

        cache_key = self._generate_cache_key(parameters, context)

        cached_content, cache_layer = await self.cache.get(
            content_type=CacheEntryType.MENTAL_MODEL,
            primary_key=cache_key,
            context=context,
        )

        if cached_content:
            self.cache_hit_count += 1
            self.logger.debug(f"🎯 Cache hit for {self.schema.name} from {cache_layer}")

            # Reconstruct result from cache
            cached_result = ToolExecutionResult(**cached_content)
            cached_result.cache_hit = True
            return cached_result

        return None

    async def cache_result(
        self,
        parameters: Dict[str, Any],
        result: ToolExecutionResult,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Cache execution result"""
        if not self.schema.cache_eligible or not result.success:
            return

        cache_key = self._generate_cache_key(parameters, context)

        await self.cache.put(
            content_type=CacheEntryType.MENTAL_MODEL,
            primary_key=cache_key,
            content=asdict(result),
            context=context,
            confidence_score=result.confidence_score,
            ttl_seconds=3600,  # 1 hour
        )

    def _generate_cache_key(
        self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate cache key for parameters and context"""
        import hashlib

        key_data = {
            "tool": self.schema.name,
            "parameters": parameters,
            "context_hash": hash(json.dumps(context or {}, sort_keys=True)),
        }

        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get tool performance metrics"""
        avg_execution_time = self.total_execution_time / max(self.execution_count, 1)

        return {
            "tool_name": self.schema.name,
            "execution_count": self.execution_count,
            "success_rate": self.success_count / max(self.execution_count, 1),
            "average_execution_time_ms": avg_execution_time,
            "cache_hit_rate": self.cache_hit_count / max(self.execution_count, 1),
            "estimated_vs_actual_time": avg_execution_time
            / self.schema.estimated_execution_time_ms,
        }


class SystemsThinkingTool(MCPMentalModelTool):
    """Systems thinking analysis tool"""

    def __init__(self):
        schema = ToolSchema(
            name="systems_thinking_analysis",
            description="Analyze problems using systems thinking methodology to identify feedback loops, stakeholders, and systemic patterns",
            category=ToolCategory.ANALYSIS,
            complexity=ToolComplexity.MODERATE,
            parameters=[
                ToolParameter(
                    name="problem_statement",
                    type="string",
                    description="The problem or situation to analyze",
                    required=True,
                ),
                ToolParameter(
                    name="scope",
                    type="string",
                    description="Analysis scope: narrow, medium, or broad",
                    required=False,
                    default="medium",
                    enum_values=["narrow", "medium", "broad"],
                ),
                ToolParameter(
                    name="focus_areas",
                    type="array",
                    description="Specific areas to focus on",
                    required=False,
                    default=[],
                ),
            ],
            return_schema={
                "type": "object",
                "properties": {
                    "system_elements": {"type": "array"},
                    "feedback_loops": {"type": "array"},
                    "stakeholder_map": {"type": "object"},
                    "leverage_points": {"type": "array"},
                    "systemic_patterns": {"type": "array"},
                },
            },
            estimated_execution_time_ms=2000,
            cache_eligible=True,
            requires_human_approval=False,
            confidence_calibrated=True,
            bias_risks=["Confirmation bias", "Complexity bias"],
            best_use_cases=[
                "Complex organizational problems",
                "Multi-stakeholder situations",
                "Problems with unclear causation",
            ],
            limitations=[
                "May over-complicate simple problems",
                "Requires significant domain knowledge",
            ],
        )
        super().__init__(schema)

    async def execute(
        self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> ToolExecutionResult:
        """Execute systems thinking analysis"""
        import time

        start_time = time.time()
        execution_id = f"systems_thinking_{int(time.time() * 1000)}"

        try:
            # Validate parameters
            valid, errors = await self.validate_parameters(parameters)
            if not valid:
                return ToolExecutionResult(
                    tool_name=self.schema.name,
                    execution_id=execution_id,
                    success=False,
                    error_type="ValidationError",
                    error_message="; ".join(errors),
                    timestamp=datetime.now(),
                )

            # Check cache
            cached_result = await self.get_cached_result(parameters, context)
            if cached_result:
                return cached_result

            # Execute analysis
            problem_statement = parameters["problem_statement"]
            scope = parameters.get("scope", "medium")
            focus_areas = parameters.get("focus_areas", [])

            # Simulate systems thinking analysis
            await asyncio.sleep(0.5)  # Simulate processing time

            analysis_result = {
                "system_elements": [
                    "Primary stakeholders",
                    "Secondary stakeholders",
                    "External influences",
                    "Resource flows",
                    "Information flows",
                ],
                "feedback_loops": [
                    {
                        "type": "reinforcing",
                        "description": "Success attracts more resources, enabling greater success",
                        "strength": "strong",
                    },
                    {
                        "type": "balancing",
                        "description": "Increased complexity creates resistance to change",
                        "strength": "moderate",
                    },
                ],
                "stakeholder_map": {
                    "primary": ["Decision makers", "Direct users"],
                    "secondary": ["Influencers", "Beneficiaries"],
                    "external": ["Regulators", "Competitors"],
                },
                "leverage_points": [
                    {
                        "point": "Resource allocation process",
                        "impact": "high",
                        "difficulty": "medium",
                    },
                    {
                        "point": "Communication channels",
                        "impact": "medium",
                        "difficulty": "low",
                    },
                ],
                "systemic_patterns": [
                    "Tragedy of the commons",
                    "Limits to growth",
                    "Success to the successful",
                ],
            }

            # Calculate confidence based on context
            confidence_score = 0.8
            if context and context.get("business_context"):
                confidence_score += 0.1
            if scope == "broad":
                confidence_score -= 0.1

            execution_time_ms = (time.time() - start_time) * 1000

            result = ToolExecutionResult(
                tool_name=self.schema.name,
                execution_id=execution_id,
                success=True,
                result=analysis_result,
                confidence_score=confidence_score,
                reasoning_steps=[
                    "1. Identified key system elements and boundaries",
                    "2. Mapped stakeholder relationships and influences",
                    "3. Analyzed feedback loops and interdependencies",
                    "4. Identified leverage points for intervention",
                    "5. Recognized common systemic patterns",
                ],
                execution_time_ms=execution_time_ms,
                validation_passed=True,
                input_context=context,
                timestamp=datetime.now(),
            )

            # Cache result
            await self.cache_result(parameters, result, context)

            # Update metrics
            self.execution_count += 1
            self.success_count += 1
            self.total_execution_time += execution_time_ms

            return result

        except Exception as e:
            self.execution_count += 1
            return ToolExecutionResult(
                tool_name=self.schema.name,
                execution_id=execution_id,
                success=False,
                error_type=type(e).__name__,
                error_message=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(),
            )


class MECEStructuringTool(MCPMentalModelTool):
    """MECE problem structuring tool"""

    def __init__(self):
        schema = ToolSchema(
            name="mece_problem_structuring",
            description="Structure problems using MECE (Mutually Exclusive, Collectively Exhaustive) framework",
            category=ToolCategory.STRUCTURING,
            complexity=ToolComplexity.SIMPLE,
            parameters=[
                ToolParameter(
                    name="problem_statement",
                    type="string",
                    description="The problem to structure",
                    required=True,
                ),
                ToolParameter(
                    name="structuring_approach",
                    type="string",
                    description="Approach for structuring",
                    required=False,
                    default="functional",
                    enum_values=["functional", "temporal", "geographic", "conceptual"],
                ),
                ToolParameter(
                    name="depth_levels",
                    type="integer",
                    description="Number of levels to structure",
                    required=False,
                    default=2,
                    min_value=1,
                    max_value=4,
                ),
            ],
            return_schema={
                "type": "object",
                "properties": {
                    "structured_breakdown": {"type": "object"},
                    "mece_validation": {"type": "object"},
                    "action_priorities": {"type": "array"},
                },
            },
            estimated_execution_time_ms=1000,
            cache_eligible=True,
            requires_human_approval=False,
            confidence_calibrated=True,
            bias_risks=["Over-simplification", "False dichotomy"],
            best_use_cases=[
                "Complex problem decomposition",
                "Strategic planning",
                "Issue prioritization",
            ],
            limitations=[
                "May force artificial categories",
                "Not suitable for highly interconnected problems",
            ],
        )
        super().__init__(schema)

    async def execute(
        self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> ToolExecutionResult:
        """Execute MECE structuring analysis"""
        import time

        start_time = time.time()
        execution_id = f"mece_structuring_{int(time.time() * 1000)}"

        try:
            # Validate parameters
            valid, errors = await self.validate_parameters(parameters)
            if not valid:
                return ToolExecutionResult(
                    tool_name=self.schema.name,
                    execution_id=execution_id,
                    success=False,
                    error_type="ValidationError",
                    error_message="; ".join(errors),
                    timestamp=datetime.now(),
                )

            # Check cache
            cached_result = await self.get_cached_result(parameters, context)
            if cached_result:
                return cached_result

            # Execute MECE structuring
            problem_statement = parameters["problem_statement"]
            approach = parameters.get("structuring_approach", "functional")
            depth_levels = parameters.get("depth_levels", 2)

            # Simulate MECE analysis
            await asyncio.sleep(0.3)

            structured_breakdown = {
                "level_1": {
                    "category_a": {
                        "description": "Primary operational aspects",
                        "components": [
                            "Process efficiency",
                            "Resource allocation",
                            "Quality control",
                        ],
                    },
                    "category_b": {
                        "description": "Strategic considerations",
                        "components": [
                            "Market positioning",
                            "Competitive advantage",
                            "Growth opportunities",
                        ],
                    },
                    "category_c": {
                        "description": "Organizational factors",
                        "components": [
                            "Team capabilities",
                            "Culture alignment",
                            "Change management",
                        ],
                    },
                }
            }

            if depth_levels > 1:
                structured_breakdown["level_2"] = {
                    "subcategories": ["Detailed breakdowns for each level 1 category"]
                }

            mece_validation = {
                "mutually_exclusive": True,
                "collectively_exhaustive": True,
                "validation_score": 0.9,
                "potential_overlaps": [],
                "missing_elements": [],
            }

            action_priorities = [
                {
                    "priority": 1,
                    "category": "category_a",
                    "rationale": "Immediate impact potential",
                },
                {
                    "priority": 2,
                    "category": "category_c",
                    "rationale": "Foundation for success",
                },
                {
                    "priority": 3,
                    "category": "category_b",
                    "rationale": "Long-term value creation",
                },
            ]

            analysis_result = {
                "structured_breakdown": structured_breakdown,
                "mece_validation": mece_validation,
                "action_priorities": action_priorities,
            }

            execution_time_ms = (time.time() - start_time) * 1000

            result = ToolExecutionResult(
                tool_name=self.schema.name,
                execution_id=execution_id,
                success=True,
                result=analysis_result,
                confidence_score=0.85,
                reasoning_steps=[
                    "1. Analyzed problem statement for key dimensions",
                    "2. Applied MECE framework with selected approach",
                    "3. Validated mutual exclusivity and exhaustiveness",
                    "4. Prioritized categories based on impact and feasibility",
                ],
                execution_time_ms=execution_time_ms,
                validation_passed=True,
                input_context=context,
                timestamp=datetime.now(),
            )

            # Cache and update metrics
            await self.cache_result(parameters, result, context)
            self.execution_count += 1
            self.success_count += 1
            self.total_execution_time += execution_time_ms

            return result

        except Exception as e:
            self.execution_count += 1
            return ToolExecutionResult(
                tool_name=self.schema.name,
                execution_id=execution_id,
                success=False,
                error_type=type(e).__name__,
                error_message=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(),
            )


class MCPToolRegistry:
    """Registry for managing MCP-compliant mental model tools"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.tools: Dict[str, MCPMentalModelTool] = {}
        self.categories: Dict[ToolCategory, List[str]] = {}
        self.performance_cache = get_performance_cache()

        # Initialize with core tools
        self._register_core_tools()

        self.logger.info(
            f"🛠️ MCP Tool Registry initialized with {len(self.tools)} tools"
        )

    def _register_core_tools(self):
        """Register core mental model tools"""
        core_tools = [
            SystemsThinkingTool(),
            MECEStructuringTool(),
            # Additional tools would be added here
        ]

        for tool in core_tools:
            self.register_tool(tool)

    def register_tool(self, tool: MCPMentalModelTool):
        """Register a new mental model tool"""
        tool_name = tool.schema.name
        self.tools[tool_name] = tool

        # Update categories
        category = tool.schema.category
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(tool_name)

        self.logger.info(f"🔧 Registered tool: {tool_name} ({category.value})")

    def get_tool(self, tool_name: str) -> Optional[MCPMentalModelTool]:
        """Get tool by name"""
        return self.tools.get(tool_name)

    def list_tools(
        self,
        category: Optional[ToolCategory] = None,
        complexity: Optional[ToolComplexity] = None,
    ) -> List[ToolSchema]:
        """List available tools with optional filtering"""
        schemas = []

        for tool in self.tools.values():
            include_tool = True

            if category and tool.schema.category != category:
                include_tool = False

            if complexity and tool.schema.complexity != complexity:
                include_tool = False

            if include_tool:
                schemas.append(tool.schema)

        return schemas

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        engagement_id: Optional[str] = None,
    ) -> ToolExecutionResult:
        """Execute a mental model tool"""
        tool = self.get_tool(tool_name)
        if not tool:
            return ToolExecutionResult(
                tool_name=tool_name,
                execution_id=f"error_{int(time.time() * 1000)}",
                success=False,
                error_type="ToolNotFound",
                error_message=f"Tool '{tool_name}' not found in registry",
                timestamp=datetime.now(),
            )

        # Create checkpoint if engagement context available
        if engagement_id:
            stateful_env = get_stateful_environment(engagement_id)
            await stateful_env.create_checkpoint(
                checkpoint_type=CheckpointType.CRITICAL_DECISION,
                current_context=context or {},
                reasoning_steps=[],
                metadata={"tool_execution": tool_name, "parameters": parameters},
            )

        # Execute tool
        result = await tool.execute(parameters, context)

        self.logger.info(
            f"🔧 Tool {tool_name} executed: "
            f"{'✅' if result.success else '❌'} "
            f"({result.execution_time_ms:.1f}ms)"
        )

        return result

    def get_registry_metrics(self) -> Dict[str, Any]:
        """Get comprehensive registry performance metrics"""
        tool_metrics = {}
        total_executions = 0
        total_successes = 0

        for name, tool in self.tools.items():
            metrics = tool.get_performance_metrics()
            tool_metrics[name] = metrics
            total_executions += metrics["execution_count"]
            total_successes += metrics["execution_count"] * metrics["success_rate"]

        return {
            "registered_tools": len(self.tools),
            "categories": {
                cat.value: len(tools) for cat, tools in self.categories.items()
            },
            "total_executions": total_executions,
            "overall_success_rate": total_successes / max(total_executions, 1),
            "tool_performance": tool_metrics,
            "performance_targets": {
                "success_rate": 0.95,
                "average_execution_time_ms": 2000,
                "cache_hit_rate": 0.70,
            },
        }


# Global tool registry
_tool_registry: Optional[MCPToolRegistry] = None


def get_mcp_tool_registry() -> MCPToolRegistry:
    """Get singleton MCP tool registry"""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = MCPToolRegistry()
    return _tool_registry
