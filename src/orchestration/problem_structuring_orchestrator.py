"""
ProblemStructuring Orchestrator - STEP 2 of Honest Orchestra
===========================================================

PRINCIPLE: "Fail Loudly, Succeed Honestly"

This orchestrator executes real problem structuring analysis with authentic LLM calls.
Uses the PSA's Core N-Way model to create analytical frameworks.

Process:
1. Initialize ProblemStructuringAgent
2. Analyze enhanced query via REAL LLM call
3. Structure into analytical framework
4. Return StructuredAnalyticalFramework or raise PSAError
"""

import time
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any

from .exceptions import PSAError
from .contracts import (
    EnhancedQuery,
    StructuredAnalyticalFramework,
    AnalyticalDimension,
    FrameworkType,
)

logger = logging.getLogger(__name__)


class ProblemStructuringOrchestrator:
    """Orchestrator for authentic problem structuring analysis"""

    def __init__(self):
        self.llm_client = None
        # Seam wrapper for safe step-wise extraction
        try:
            from src.orchestration.seams.problem_structuring_seam import (
                ProblemStructuringSeam,
            )

            self._seam = ProblemStructuringSeam()
        except Exception:
            self._seam = None
        # New: Core adapter for V2 integration
        try:
            from src.orchestration.cognitive_core_adapter import CognitiveCoreAdapter

            self.core_adapter = CognitiveCoreAdapter()
        except Exception:
            self.core_adapter = None

    async def _initialize_llm_client(self):
        """Initialize real LLM client - DeepSeek primary, Claude fallback"""
        try:
            import os
            from dotenv import load_dotenv

            load_dotenv()

            deepseek_key = os.getenv("DEEPSEEK_API_KEY")
            claude_key = os.getenv("ANTHROPIC_API_KEY")

            if deepseek_key:
                self.llm_client = DirectPSAClient(deepseek_key, "deepseek")
                logger.info("✅ Initialized DeepSeek PSA client")
            elif claude_key:
                self.llm_client = DirectPSAClient(claude_key, "claude")
                logger.info("✅ Initialized Claude PSA client")
            else:
                raise PSAError("No LLM API keys available - cannot proceed")

        except Exception as e:
            raise PSAError(f"Failed to initialize PSA LLM client: {e}")

    async def run_problem_structuring(
        self, enhanced_query: EnhancedQuery
    ) -> StructuredAnalyticalFramework:
        # V2 default: CoreOps + Cognitive Core
        if getattr(self, "core_adapter", None):
            payload = None
            try:
                payload = await self.core_adapter.execute_analysis(
                    system_contract_id="problem_structuring@1.0",
                    context={
                        "enhanced_query": enhanced_query.enhanced_query,
                        "context_enrichment": enhanced_query.context_enrichment,
                    },
                )
            except Exception as e:
                logger.error(f"CoreOps V2 failed in Problem Structuring: {e}")
            if payload:
                framework = payload.get("framework")
                if framework:
                    return framework
            logger.warning("Falling back to legacy PSA path due to V2 failure")
        """
        Execute complete problem structuring process with real LLM calls
        
        Args:
            enhanced_query: Output from Socratic inquiry step
            
        Returns:
            StructuredAnalyticalFramework: Complete analytical framework
            
        Raises:
            PSAError: If any step fails
        """
        start_time = time.time()

        try:
            logger.info(
                f"🏗️ Starting problem structuring for: {enhanced_query.enhanced_query[:100]}..."
            )

            # Step 1: Initialize real LLM client
            await self._initialize_llm_client()

            # Step 2: Analyze problem and determine framework type
            logger.info("🔍 Analyzing problem complexity and type...")
            if self._seam is not None:
                framework_type = await self._seam.determine_framework_type(self, enhanced_query)
            else:
                framework_type = await self._determine_framework_type(enhanced_query)

            # Step 3: Generate analytical dimensions via REAL LLM call
            logger.info("📊 Generating analytical dimensions...")
            if self._seam is not None:
                analytical_dimensions = await self._seam.generate_analytical_dimensions(
                    self, enhanced_query, framework_type
                )
            else:
                analytical_dimensions = await self._generate_analytical_dimensions(
                    enhanced_query, framework_type
                )

            # Step 4: Define analytical sequence and consultant recommendations
            logger.info("🎯 Defining analysis sequence...")
            if self._seam is not None:
                sequence, consultant_types, complexity = await self._seam.define_analysis_approach(
                    self, enhanced_query, framework_type, analytical_dimensions
                )
            else:
                sequence, consultant_types, complexity = (
                    await self._define_analysis_approach(
                        enhanced_query, framework_type, analytical_dimensions
                    )
                )

            # Step 5: Extract secondary considerations
            if self._seam is not None:
                secondary_considerations = await self._seam.extract_secondary_considerations(
                    self, enhanced_query
                )
            else:
                secondary_considerations = await self._extract_secondary_considerations(
                    enhanced_query
                )

            processing_time = time.time() - start_time

            # Step 6: Construct final framework
            framework = StructuredAnalyticalFramework(
                framework_type=framework_type,
                primary_dimensions=analytical_dimensions,
                secondary_considerations=secondary_considerations,
                analytical_sequence=sequence,
                complexity_assessment=complexity,
                recommended_consultant_types=consultant_types,
                processing_time_seconds=processing_time,
                timestamp=datetime.now(timezone.utc),
            )

            logger.info(f"✅ Problem structuring completed in {processing_time:.1f}s")
            logger.info(
                f"📋 Framework: {framework_type.value}, {len(analytical_dimensions)} dimensions"
            )

            return framework

        except PSAError:
            raise
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                f"❌ Problem structuring failed after {processing_time:.1f}s: {e}"
            )
            raise PSAError(f"Problem structuring failed: {e}")

    async def _determine_framework_type(
        self, enhanced_query: EnhancedQuery
    ) -> FrameworkType:
        """Determine the appropriate analytical framework type"""

        prompt = f"""You are a strategic problem structuring expert analyzing business problems.

ENHANCED PROBLEM: {enhanced_query.enhanced_query}

CONTEXT ENRICHMENT:
{enhanced_query.context_enrichment}

Determine the most appropriate analytical framework type for this problem. 

Choose from these framework types:
1. STRATEGIC_ANALYSIS - Long-term direction, competitive positioning, market analysis
2. OPERATIONAL_OPTIMIZATION - Process improvement, efficiency, resource allocation  
3. INNOVATION_DISCOVERY - New opportunities, product development, market expansion
4. CRISIS_MANAGEMENT - Urgent issues, damage control, rapid response needed

Respond with JSON:
{{
    "framework_type": "STRATEGIC_ANALYSIS",
    "rationale": "Detailed explanation of why this framework is most appropriate..."
}}

Consider problem urgency, scope, complexity, and strategic impact."""

        try:
            response = await self.llm_client.call_llm(prompt)

            parsed = self._extract_json_from_response(response)

            framework_str = parsed.get("framework_type", "STRATEGIC_ANALYSIS")
            rationale = parsed.get("rationale", "")

            # Convert string to enum
            framework_type = FrameworkType(framework_str.lower())

            logger.info(f"🎯 Determined framework type: {framework_type.value}")
            logger.info(f"📝 Rationale: {rationale[:100]}...")

            return framework_type

        except ValueError as e:
            logger.warning(
                f"Invalid framework type, defaulting to STRATEGIC_ANALYSIS: {e}"
            )
            return FrameworkType.STRATEGIC_ANALYSIS
        except Exception as e:
            raise PSAError(f"Failed to determine framework type: {e}")

    async def _generate_analytical_dimensions(
        self, enhanced_query: EnhancedQuery, framework_type: FrameworkType
    ) -> List[AnalyticalDimension]:
        """Generate analytical dimensions via REAL LLM call"""

        prompt = f"""You are a strategic consultant creating an analytical framework.

PROBLEM: {enhanced_query.enhanced_query}

CONTEXT: {enhanced_query.context_enrichment}

FRAMEWORK TYPE: {framework_type.value}

Create 3-5 primary analytical dimensions for this {framework_type.value} framework.

Each dimension should have:
1. Clear dimension name
2. 3-4 key questions to explore
3. Specific analysis approach
4. Priority level (1=highest, 5=lowest)

Format as JSON:
{{
    "dimensions": [
        {{
            "dimension_name": "Market Position Analysis",
            "key_questions": [
                "What is our competitive advantage?",
                "How is market share distributed?",
                "What are key differentiators?"
            ],
            "analysis_approach": "Competitive benchmarking and market research",
            "priority_level": 1
        }},
        ...
    ]
}}

Make dimensions specific, actionable, and complementary."""

        try:
            response = await self.llm_client.call_llm(prompt)
            parsed = self._extract_json_from_response(response)

            dimensions_data = parsed.get("dimensions", [])

            if len(dimensions_data) < 2:
                raise PSAError(
                    f"Insufficient analytical dimensions generated: {len(dimensions_data)}"
                )

            dimensions = []
            for dim_data in dimensions_data:
                dimension = AnalyticalDimension(
                    dimension_name=dim_data.get("dimension_name", ""),
                    key_questions=dim_data.get("key_questions", []),
                    analysis_approach=dim_data.get("analysis_approach", ""),
                    priority_level=dim_data.get("priority_level", 3),
                )
                dimensions.append(dimension)

            logger.info(f"📊 Generated {len(dimensions)} analytical dimensions")
            return dimensions

        except Exception as e:
            raise PSAError(f"Failed to generate analytical dimensions: {e}")

    async def _define_analysis_approach(
        self,
        enhanced_query: EnhancedQuery,
        framework_type: FrameworkType,
        dimensions: List[AnalyticalDimension],
    ) -> tuple[List[str], List[str], str]:
        """Define analysis sequence and consultant recommendations"""

        dimensions_summary = "\n".join(
            [f"- {d.dimension_name}: {d.analysis_approach}" for d in dimensions]
        )

        prompt = f"""You are a strategic project manager defining the analysis approach.

PROBLEM: {enhanced_query.enhanced_query}
FRAMEWORK: {framework_type.value}

ANALYTICAL DIMENSIONS:
{dimensions_summary}

Define the optimal approach:

1. ANALYSIS SEQUENCE - Order of tackling dimensions (consider dependencies)
2. CONSULTANT TYPES - What types of consultants are needed
3. COMPLEXITY ASSESSMENT - Overall complexity level

Format as JSON:
{{
    "analytical_sequence": [
        "Start with market analysis",
        "Then assess internal capabilities", 
        "Followed by strategic option development",
        "Conclude with implementation planning"
    ],
    "recommended_consultant_types": [
        "strategic_analyst",
        "market_researcher", 
        "implementation_specialist"
    ],
    "complexity_assessment": "HIGH - Multi-faceted strategic challenge requiring specialized expertise"
}}

Be specific about consultant expertise needed."""

        try:
            response = await self.llm_client.call_llm(prompt)
            parsed = self._extract_json_from_response(response)

            sequence = parsed.get("analytical_sequence", [])
            consultant_types = parsed.get("recommended_consultant_types", [])
            complexity = parsed.get("complexity_assessment", "MODERATE")

            logger.info(f"📋 Defined {len(sequence)}-step analysis sequence")
            logger.info(f"👥 Recommended {len(consultant_types)} consultant types")

            return sequence, consultant_types, complexity

        except Exception as e:
            raise PSAError(f"Failed to define analysis approach: {e}")

    async def _extract_secondary_considerations(
        self, enhanced_query: EnhancedQuery
    ) -> List[str]:
        """Extract secondary considerations from enhanced query context"""

        considerations = []

        context = enhanced_query.context_enrichment

        # Extract from various context categories
        if "key_constraints" in context:
            considerations.extend(
                [f"Constraint: {c}" for c in context["key_constraints"]]
            )

        if "stakeholder_considerations" in context:
            considerations.extend(
                [f"Stakeholder: {s}" for s in context["stakeholder_considerations"]]
            )

        if "success_criteria" in context:
            considerations.extend(
                [f"Success Factor: {sc}" for sc in context["success_criteria"]]
            )

        # Add some standard secondary considerations if none found
        if not considerations:
            considerations = [
                "Stakeholder alignment and communication",
                "Resource availability and constraints",
                "Risk mitigation and contingency planning",
                "Timeline and milestone management",
            ]

        logger.info(f"📝 Identified {len(considerations)} secondary considerations")
        return considerations[:6]  # Limit to 6 most important

    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response"""
        import json
        import re

        response_text = response.strip()

        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_text = response_text[start:end].strip()
        elif response_text.startswith("{"):
            json_text = response_text
        else:
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group()
            else:
                raise PSAError("Could not extract JSON from LLM response")

        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            raise PSAError(f"Failed to parse JSON from response: {e}")


# ============================================================================
# DIRECT PSA CLIENT IMPLEMENTATIONS
# ============================================================================


class DirectPSAClient:
    """Direct LLM API client for PSA operations"""

    def __init__(self, api_key: str, provider: str):
        self.api_key = api_key
        self.provider = provider

    async def call_llm(self, prompt: str) -> str:
        """Make real LLM API call"""

        if self.provider == "deepseek":
            return await self._call_deepseek(prompt)
        elif self.provider == "claude":
            return await self._call_claude(prompt)
        else:
            raise PSAError(f"Unsupported provider: {self.provider}")

    async def _call_deepseek(self, prompt: str) -> str:
        """DeepSeek API call"""
        import httpx

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2500,
            "temperature": 0.7,
        }

        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )

        if response.status_code != 200:
            raise PSAError(
                f"DeepSeek API error: {response.status_code} - {response.text}"
            )

        result = response.json()
        return result["choices"][0]["message"]["content"]

    async def _call_claude(self, prompt: str) -> str:
        """Claude API call"""
        import httpx

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 2500,
            "messages": [{"role": "user", "content": prompt}],
        }

        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages", headers=headers, json=payload
            )

        if response.status_code != 200:
            raise PSAError(
                f"Claude API error: {response.status_code} - {response.text}"
            )

        result = response.json()
        return result["content"][0]["text"]


# ============================================================================
# MAIN FUNCTION FOR STEP 2
# ============================================================================


async def run_problem_structuring(
    enhanced_query: EnhancedQuery,
) -> StructuredAnalyticalFramework:
    """
    Main function for Step 2: Execute problem structuring with real LLM calls

    Args:
        enhanced_query: Output from Socratic inquiry step

    Returns:
        StructuredAnalyticalFramework: Complete analytical framework

    Raises:
        PSAError: If any step fails
    """
    orchestrator = ProblemStructuringOrchestrator()
    return await orchestrator.run_problem_structuring(enhanced_query)
