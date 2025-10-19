#!/usr/bin/env python3
"""
Parallel Devil's Advocate Architecture
Re-architect critique system for parallel execution while maintaining DeepSeek V3.1 quality

ARCHITECTURE CHANGE:
- Old: Sequential critique chain (Ackoff → Munger → Audit)
- New: Parallel critique engines + fast synthesis
"""

import asyncio
import time
import requests
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CritiqueEngine:
    """Individual critique engine configuration"""

    name: str
    focus_area: str
    system_prompt: str
    temperature: float = 0.8


class ParallelDevilsAdvocate:
    """
    Parallel execution of multiple critique engines for maximum performance
    while maintaining analytical depth through DeepSeek V3.1
    """

    def __init__(self):
        self.critique_engines = self._initialize_critique_engines()
        self.api_key = os.getenv("DEEPSEEK_API_KEY")

        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY required for critique execution")

    def _initialize_critique_engines(self) -> List[CritiqueEngine]:
        """Initialize the three parallel critique engines"""
        return [
            CritiqueEngine(
                name="assumption_challenger",
                focus_area="Assumption Analysis & Logic Gaps",
                system_prompt="""You are an Assumption Challenger specializing in identifying flawed premises and logic gaps.

Your focus:
1. Challenge unstated assumptions in strategic recommendations
2. Identify gaps in logical reasoning and causal relationships  
3. Question the validity of evidence and data interpretation
4. Examine whether conclusions follow from premises

Be rigorous but constructive in identifying logical weaknesses.""",
            ),
            CritiqueEngine(
                name="risk_assessor",
                focus_area="Implementation Risks & Market Reality",
                system_prompt="""You are a Risk Assessor specializing in implementation challenges and market realities.

Your focus:
1. Identify execution risks and potential failure modes
2. Challenge market assumptions and competitive dynamics
3. Assess resource constraints and capability gaps  
4. Evaluate external factors that could derail strategies

Focus on practical, real-world obstacles to implementation.""",
            ),
            CritiqueEngine(
                name="contrarian_analyst",
                focus_area="Alternative Perspectives & Counterarguments",
                system_prompt="""You are a Contrarian Analyst specializing in alternative viewpoints and counterarguments.

Your focus:
1. Present compelling cases against the recommendations
2. Identify scenarios where the strategy would fail
3. Highlight overlooked alternatives and opportunity costs
4. Challenge the strategic direction from different stakeholder perspectives

Provide substantive counterarguments that stress-test the recommendations.""",
            ),
        ]

    async def execute_parallel_critique(
        self, consultant_analyses: List[Dict[str, Any]], original_query: str
    ) -> Dict[str, Any]:
        """
        Execute all three critique engines in parallel

        Returns comprehensive critique synthesized from parallel execution
        """

        start_time = time.time()
        logger.info("👿 Starting parallel Devil's Advocate critique")
        logger.info(f"   Engines: {len(self.critique_engines)} running in parallel")
        logger.info("   Target: <15s for all critique engines")

        # Prepare combined analysis for critique
        combined_analysis = self._prepare_analysis_for_critique(
            consultant_analyses, original_query
        )

        # Execute all critique engines in parallel
        critique_results = {}

        try:
            async with asyncio.TaskGroup() as group:
                tasks = []
                for engine in self.critique_engines:
                    task = group.create_task(
                        self._execute_single_critique_engine(engine, combined_analysis)
                    )
                    tasks.append((engine.name, task))

            # Collect results
            for engine_name, task in tasks:
                try:
                    result = task.result()
                    if result and result.get("success"):
                        critique_results[engine_name] = result
                        logger.info(
                            f"   ✅ {engine_name}: {result['latency_ms']}ms, {result['tokens']} tokens"
                        )
                    else:
                        logger.error(f"   ❌ {engine_name}: Failed")
                except Exception as e:
                    logger.error(f"   💥 {engine_name}: {str(e)}")

        except Exception as e:
            logger.error(f"Parallel critique execution failed: {e}")
            return {"success": False, "error": str(e)}

        # Fast synthesis of parallel critiques
        synthesis_start = time.time()
        synthesized_critique = await self._synthesize_parallel_critiques(
            critique_results, combined_analysis
        )
        synthesis_time = (time.time() - synthesis_start) * 1000

        total_time = (time.time() - start_time) * 1000

        logger.info(f"   🔄 Synthesis: {synthesis_time:.0f}ms")
        logger.info(f"✅ Parallel critique completed in {total_time:.0f}ms")

        return {
            "success": True,
            "parallel_critiques": critique_results,
            "synthesized_critique": synthesized_critique,
            "execution_metadata": {
                "total_engines": len(self.critique_engines),
                "successful_engines": len(critique_results),
                "total_time_ms": total_time,
                "synthesis_time_ms": synthesis_time,
                "parallel_execution": True,
            },
        }

    def _prepare_analysis_for_critique(
        self, consultant_analyses: List[Dict[str, Any]], original_query: str
    ) -> str:
        """Prepare consultant analyses for critique"""

        combined = f"ORIGINAL QUERY: {original_query}\n\n"
        combined += "CONSULTANT ANALYSES TO CRITIQUE:\n"
        combined += "=" * 60 + "\n\n"

        for i, analysis in enumerate(consultant_analyses, 1):
            consultant_name = analysis.get("consultant_role", f"Consultant_{i}")
            mental_model = analysis.get("mental_model", "Unknown")
            content = analysis.get("result", {}).get(
                "content", analysis.get("content", "")
            )

            combined += f"=== {consultant_name.upper()} ({mental_model}) ===\n"
            combined += content
            combined += "\n\n" + "-" * 40 + "\n\n"

        return combined

    async def _execute_single_critique_engine(
        self, engine: CritiqueEngine, combined_analysis: str
    ) -> Optional[Dict[str, Any]]:
        """Execute single critique engine with DeepSeek V3.1"""

        user_prompt = f"""Perform {engine.focus_area} on the following strategic analyses:

{combined_analysis}

Focus specifically on your area of expertise: {engine.focus_area}

Provide structured critique with:
1. Key issues identified in your focus area
2. Specific examples and evidence
3. Potential consequences if issues are not addressed
4. Constructive suggestions for improvement

Be thorough but concise."""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": engine.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": engine.temperature,
            "max_tokens": 1500,
        }

        try:
            start_time = time.time()

            response = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )

            latency_ms = int((time.time() - start_time) * 1000)

            if response.status_code == 200:
                data = response.json()
                usage = data["usage"]

                return {
                    "success": True,
                    "engine_name": engine.name,
                    "focus_area": engine.focus_area,
                    "content": data["choices"][0]["message"]["content"],
                    "tokens": usage["total_tokens"],
                    "cost_usd": (usage["total_tokens"] / 1_000_000) * 2.19,
                    "latency_ms": latency_ms,
                    "model": "deepseek-chat",
                }
            else:
                logger.error(f"{engine.name} API error: {response.status_code}")
                return {"success": False, "error": f"HTTP {response.status_code}"}

        except Exception as e:
            logger.error(f"{engine.name} execution error: {e}")
            return {"success": False, "error": str(e)}

    async def _synthesize_parallel_critiques(
        self, critique_results: Dict[str, Dict[str, Any]], original_analysis: str
    ) -> Dict[str, Any]:
        """Fast synthesis of parallel critique results"""

        if not critique_results:
            return {"success": False, "error": "No critique results to synthesize"}

        # Prepare synthesis prompt
        critique_summary = "PARALLEL CRITIQUE RESULTS:\n\n"

        for engine_name, result in critique_results.items():
            focus_area = result.get("focus_area", engine_name)
            content = result.get("content", "")

            critique_summary += f"=== {focus_area.upper()} ===\n"
            critique_summary += content
            critique_summary += "\n\n" + "-" * 30 + "\n\n"

        synthesis_prompt = f"""Synthesize these parallel critique analyses into a unified Devil's Advocate report:

{critique_summary}

Create a comprehensive critique that:
1. Integrates insights from all three critique perspectives
2. Prioritizes the most critical risks and flawed assumptions
3. Provides actionable recommendations for addressing weaknesses
4. Maintains constructive tone while being rigorous

Structure as a unified Devil's Advocate analysis."""

        # Execute synthesis with DeepSeek V3.1
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a Senior Critic synthesizing multiple critique perspectives into a unified analysis.",
                },
                {"role": "user", "content": synthesis_prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 2000,
        }

        try:
            start_time = time.time()

            response = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers=headers,
                json=payload,
                timeout=45,
            )

            latency_ms = int((time.time() - start_time) * 1000)

            if response.status_code == 200:
                data = response.json()
                usage = data["usage"]

                return {
                    "success": True,
                    "content": data["choices"][0]["message"]["content"],
                    "tokens": usage["total_tokens"],
                    "cost_usd": (usage["total_tokens"] / 1_000_000) * 2.19,
                    "latency_ms": latency_ms,
                    "model": "deepseek-chat",
                    "synthesis_method": "parallel_integration",
                }
            else:
                return {
                    "success": False,
                    "error": f"Synthesis HTTP {response.status_code}",
                }

        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return {"success": False, "error": str(e)}


# Test function for parallel critique
async def test_parallel_devils_advocate():
    """Test the parallel Devil's Advocate system"""

    print("🧪 Testing Parallel Devil's Advocate")
    print("=" * 50)

    # Mock consultant analyses for testing
    mock_analyses = [
        {
            "consultant_role": "strategic_analyst",
            "mental_model": "First-Principles Thinking",
            "result": {
                "content": """Strategic Analysis: Aperture Labs should pivot from consumer cameras to B2B optical components. 
                Core competency is precision optics, not camera manufacturing. 
                Recommended markets: medical imaging, autonomous vehicles, AR/VR systems.
                Implementation: Partner with tech companies, license IP, develop custom solutions."""
            },
        },
        {
            "consultant_role": "innovation_catalyst",
            "mental_model": "Disruptive Innovation",
            "result": {
                "content": """Innovation Strategy: Transform into 'Intel Inside' for vision systems.
                Blue ocean opportunities in quantum sensors, holographic displays, space telescopes.
                Create optical API platform for developers. 
                Disrupt the vision stack with computational optics."""
            },
        },
    ]

    query = "Camera company losing to smartphones - how to pivot our optical expertise?"

    # Execute parallel critique
    critic = ParallelDevilsAdvocate()

    start_time = time.time()
    result = await critic.execute_parallel_critique(mock_analyses, query)
    total_time = time.time() - start_time

    print("\n🏁 Parallel Critique Results:")
    print(f"   ⏱️  Total Time: {total_time:.2f}s")
    print(f"   ✅ Success: {result.get('success', False)}")

    if result.get("success"):
        metadata = result.get("execution_metadata", {})
        print(
            f"   🔧 Engines: {metadata.get('successful_engines', 0)}/{metadata.get('total_engines', 0)}"
        )
        print(f"   ⚡ Parallel Execution: {metadata.get('parallel_execution', False)}")

        # Show critique preview
        synthesis = result.get("synthesized_critique", {})
        if synthesis.get("content"):
            preview = synthesis["content"][:200] + "..."
            print(f"   🔍 Critique Preview: {preview}")

    return result


if __name__ == "__main__":
    asyncio.run(test_parallel_devils_advocate())
