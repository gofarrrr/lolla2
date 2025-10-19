#!/usr/bin/env python3
"""
Function Call Executor for DeepSeek V3.1 Native Function Calling
Processes function calls returned by DeepSeek and executes them with actual implementations
"""

import json
import asyncio
import logging
from typing import Dict, List, Any, Optional

# Import Perplexity integration
from ..perplexity_client import PerplexityClient, KnowledgeQueryType

logger = logging.getLogger(__name__)


class FunctionCallExecutor:
    """Executes function calls returned by DeepSeek V3.1 native function calling"""

    def __init__(self):
        self.perplexity_client: Optional[PerplexityClient] = None
        self.logger = logger

    async def _get_perplexity_client(self) -> PerplexityClient:
        """Get or initialize Perplexity client"""
        if not self.perplexity_client:
            from ..perplexity_client import get_perplexity_client

            self.perplexity_client = await get_perplexity_client()
        return self.perplexity_client

    async def execute_function_call(
        self, function_call: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a single function call from DeepSeek V3.1

        Args:
            function_call: Function call dict with 'name' and 'arguments'

        Returns:
            Dict with execution result
        """
        function_name = function_call.get("name")
        arguments = function_call.get("arguments", {})

        # Parse arguments if they're a JSON string
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Failed to parse function arguments: {e}",
                    "function_name": function_name,
                }

        self.logger.info(f"🔧 Executing function call: {function_name}")
        self.logger.info(f"   Arguments: {arguments}")

        try:
            # Enforce tool governance (allowlist + HITL)
            try:
                from src.security.tool_governance import get_tool_governance
                governance = get_tool_governance()
                governance.enforce_tool_policy(
                    function_name,
                    arguments if isinstance(arguments, dict) else {},
                    allow_high_risk=bool(arguments.get("allow_high_risk", False) if isinstance(arguments, dict) else False),
                    user_confirmed=bool(arguments.get("user_confirmed", False) if isinstance(arguments, dict) else False),
                )
            except PermissionError as pe:
                return {
                    "success": False,
                    "error": f"Tool policy violation: {pe}",
                    "function_name": function_name,
                }
            except Exception as ge:
                # Fail safe-open only for governance import issues; log and proceed
                self.logger.warning(f"Tool governance check skipped: {ge}")

            if function_name == "perplexity_query_knowledge":
                return await self._execute_perplexity_query_knowledge(arguments)
            elif function_name == "perplexity_deep_research":
                return await self._execute_perplexity_deep_research(arguments)
            else:
                return {
                    "success": False,
                    "error": f"Unknown function: {function_name}",
                    "function_name": function_name,
                }

        except Exception as e:
            self.logger.error(f"❌ Function execution failed: {e}")
            return {"success": False, "error": str(e), "function_name": function_name}

    async def _execute_perplexity_query_knowledge(
        self, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute perplexity_query_knowledge function"""
        client = await self._get_perplexity_client()

        # Extract arguments with defaults
        query = args.get("query")
        query_type_str = args.get("query_type", "context_grounding")
        model = args.get("model", "sonar-pro")
        max_tokens = args.get("max_tokens", 1000)
        operation_context = args.get("operation_context", "")

        if not query:
            return {
                "success": False,
                "error": "Query parameter is required",
                "function_name": "perplexity_query_knowledge",
            }

        # Convert string to enum
        try:
            query_type = KnowledgeQueryType(query_type_str)
        except ValueError:
            query_type = KnowledgeQueryType.CONTEXT_GROUNDING

        # Execute the Perplexity query
        research_interaction = await client.query_knowledge(
            query=query,
            query_type=query_type,
            model=model,
            max_tokens=max_tokens,
            operation_context=operation_context,
        )

        return {
            "success": True,
            "function_name": "perplexity_query_knowledge",
            "result": {
                "content": research_interaction.raw_response_received,
                "sources": [
                    source.get("url", "")
                    for source in research_interaction.sources_extracted
                ],
                "confidence_score": research_interaction.confidence_score,
                "sources_count": research_interaction.sources_consulted_count,
                "research_id": research_interaction.research_id,
                "search_mode": research_interaction.search_mode,
            },
            "metadata": {
                "timestamp": research_interaction.timestamp.isoformat(),
                "query_sent": research_interaction.query_sent,
                "cost_estimate": f"~${client._calculate_cost(max_tokens, model):.4f}",
            },
        }

    async def _execute_perplexity_deep_research(
        self, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute perplexity_deep_research function"""
        client = await self._get_perplexity_client()

        # Extract arguments
        query = args.get("query")
        context = args.get("context", {})
        focus_areas = args.get(
            "focus_areas",
            ["market analysis", "competitive landscape", "strategic implications"],
        )

        if not query:
            return {
                "success": False,
                "error": "Query parameter is required",
                "function_name": "perplexity_deep_research",
            }

        # Execute the deep research
        research_interaction = await client.conduct_deep_research(
            query=query, context=context, focus_areas=focus_areas
        )

        return {
            "success": True,
            "function_name": "perplexity_deep_research",
            "result": {
                "content": research_interaction.raw_response_received,
                "sources": [
                    source.get("url", "")
                    for source in research_interaction.sources_extracted
                ],
                "confidence_score": research_interaction.confidence_score,
                "sources_count": research_interaction.sources_consulted_count,
                "research_id": research_interaction.research_id,
                "search_mode": research_interaction.search_mode,
            },
            "metadata": {
                "timestamp": research_interaction.timestamp.isoformat(),
                "query_sent": research_interaction.query_sent,
                "focus_areas": focus_areas,
                "context": context,
                "cost_estimate": f"~${client._calculate_cost(2000, 'sonar-deep-research'):.4f}",
            },
        }

    async def execute_multiple_function_calls(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple function calls in parallel

        Args:
            tool_calls: List of function call dicts

        Returns:
            List of execution results
        """
        if not tool_calls:
            return []

        self.logger.info(f"🔧 Executing {len(tool_calls)} function calls in parallel")

        # Execute all function calls in parallel for efficiency
        tasks = [self.execute_function_call(call) for call in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    {
                        "success": False,
                        "error": str(result),
                        "function_name": tool_calls[i].get("name", "unknown"),
                    }
                )
            else:
                processed_results.append(result)

        self.logger.info(f"✅ Completed {len(processed_results)} function executions")
        return processed_results


# Global executor instance
_executor_instance: Optional[FunctionCallExecutor] = None


def get_function_call_executor() -> FunctionCallExecutor:
    """Get the global function call executor instance"""
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = FunctionCallExecutor()
    return _executor_instance
