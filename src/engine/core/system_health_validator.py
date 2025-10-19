"""
METIS System Health Validator
Honest assessment of actual system capabilities vs architectural framework
"""

import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass


class ComponentStatus(str, Enum):
    """Component readiness status"""

    READY = "ready"
    FRAMEWORK_ONLY = "framework_only"
    NOT_IMPLEMENTED = "not_implemented"
    ERROR = "error"


class IntegrationRequirement(str, Enum):
    """Required integrations for full functionality"""

    CLAUDE_SONNET = "claude_sonnet_3_5"
    NWAY_DATABASE = "nway_interactions_database"
    PERPLEXITY_API = "perplexity_knowledge"
    REAL_PROCESSING = "real_cognitive_processing"


@dataclass
class ComponentHealth:
    """Health status for individual component"""

    name: str
    status: ComponentStatus
    implementation_percentage: int
    missing_requirements: List[IntegrationRequirement]
    description: str
    error_message: Optional[str] = None


class SystemHealthValidator:
    """
    Validates actual system implementation vs architectural framework
    Provides honest assessment of what's working vs what's mock/placeholder
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def validate_system_health(self) -> Dict[str, Any]:
        """
        Comprehensive system health validation
        Returns honest assessment of implementation status
        """

        health_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": ComponentStatus.FRAMEWORK_ONLY,
            "overall_implementation": 25,  # Only architecture exists
            "production_ready": False,
            "component_health": {},
            "missing_integrations": [],
            "implementation_gaps": [],
            "next_steps": [],
        }

        # Validate each component
        components = [
            await self._validate_llm_integration(),
            await self._validate_cognitive_engine(),
            await self._validate_nway_integration(),
            await self._validate_frontend_ui(),
            await self._validate_api_layer(),
            await self._validate_knowledge_grounding(),
            await self._validate_deliverable_generation(),
        ]

        # Aggregate results
        for component in components:
            health_report["component_health"][component.name] = {
                "status": component.status,
                "implementation_percentage": component.implementation_percentage,
                "missing_requirements": [
                    req.value for req in component.missing_requirements
                ],
                "description": component.description,
                "error_message": component.error_message,
            }

            # Collect missing integrations
            for req in component.missing_requirements:
                if req not in health_report["missing_integrations"]:
                    health_report["missing_integrations"].append(req.value)

        # Calculate overall implementation percentage
        total_impl = sum(c.implementation_percentage for c in components)
        health_report["overall_implementation"] = total_impl // len(components)

        # Determine overall status
        if health_report["overall_implementation"] >= 90:
            health_report["overall_status"] = ComponentStatus.READY
            health_report["production_ready"] = True
        elif health_report["overall_implementation"] >= 50:
            health_report["overall_status"] = ComponentStatus.FRAMEWORK_ONLY
        else:
            health_report["overall_status"] = ComponentStatus.NOT_IMPLEMENTED

        # Add implementation gaps and next steps
        health_report["implementation_gaps"] = self._identify_implementation_gaps(
            components
        )
        health_report["next_steps"] = self._generate_next_steps(components)

        return health_report

    async def _validate_llm_integration(self) -> ComponentHealth:
        """Validate actual LLM API integration"""

        # Check for API keys
        has_anthropic_key = bool(os.getenv("ANTHROPIC_API_KEY"))
        has_openai_key = bool(os.getenv("OPENAI_API_KEY"))

        # Check for actual LLM client implementation
        try:
            # Try to import anthropic client
            import anthropic

            has_anthropic_lib = True
        except ImportError:
            has_anthropic_lib = False

        if has_anthropic_key and has_anthropic_lib:
            # Test actual API connection with real call
            try:
                from src.integrations.claude_client import (
                    get_claude_client,
                    LLMCallType,
                )

                client = await get_claude_client()
                test_response = await client.call_claude(
                    prompt="Respond with exactly: 'Health check successful'",
                    call_type=LLMCallType.VALIDATION,
                    system_prompt="You are a health check assistant.",
                    max_tokens=10,
                )

                if test_response and "successful" in test_response.content.lower():
                    status = ComponentStatus.READY
                    impl_pct = 90
                    description = "LLM API connectivity verified - real Claude integration working"
                    missing_reqs = []
                else:
                    status = ComponentStatus.ERROR
                    impl_pct = 30
                    description = "LLM API connected but response validation failed"
                    missing_reqs = [IntegrationRequirement.REAL_PROCESSING]

            except Exception as e:
                status = ComponentStatus.ERROR
                impl_pct = 10
                description = f"LLM API connection failed: {str(e)[:100]}"
                missing_reqs = [
                    IntegrationRequirement.CLAUDE_SONNET,
                    IntegrationRequirement.REAL_PROCESSING,
                ]
        else:
            status = ComponentStatus.NOT_IMPLEMENTED
            impl_pct = 0
            description = "No LLM integration: missing API keys or libraries"
            missing_reqs = [IntegrationRequirement.CLAUDE_SONNET]

        return ComponentHealth(
            name="llm_integration",
            status=status,
            implementation_percentage=impl_pct,
            missing_requirements=missing_reqs,
            description=description,
        )

    async def _validate_cognitive_engine(self) -> ComponentHealth:
        """Validate cognitive processing implementation"""

        try:
            from src.factories.engine_factory import CognitiveEngineFactory

            engine = CognitiveEngineFactory.create_engine()
            models = await engine.get_available_models()

            # Check if models actually work (not just return hardcoded data)
            if len(models) > 0:
                # Test one model to see if it returns real processing
                test_result = await engine._apply_systems_thinking(
                    "test problem", {}, "test"
                )

                if "NOT_IMPLEMENTED" in test_result.get("reasoning_text", ""):
                    status = ComponentStatus.FRAMEWORK_ONLY
                    impl_pct = 30
                    description = "Cognitive engine framework exists but returns placeholder responses"
                    missing_reqs = [
                        IntegrationRequirement.CLAUDE_SONNET,
                        IntegrationRequirement.REAL_PROCESSING,
                    ]
                else:
                    status = ComponentStatus.READY
                    impl_pct = 85
                    description = (
                        "Cognitive engine fully operational with real LLM processing"
                    )
                    missing_reqs = []
            else:
                status = ComponentStatus.NOT_IMPLEMENTED
                impl_pct = 10
                description = "Cognitive engine exists but no mental models loaded"
                missing_reqs = [IntegrationRequirement.CLAUDE_SONNET]

        except Exception:
            status = ComponentStatus.ERROR
            impl_pct = 0
            description = "Cognitive engine failed to initialize"
            missing_reqs = [IntegrationRequirement.CLAUDE_SONNET]

        return ComponentHealth(
            name="cognitive_engine",
            status=status,
            implementation_percentage=impl_pct,
            missing_requirements=missing_reqs,
            description=description,
        )

    async def _validate_nway_integration(self) -> ComponentHealth:
        """Validate N-way interactions database integration with real Supabase connectivity"""

        # Check for Supabase credentials
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")

        if not supabase_url or not supabase_key:
            status = ComponentStatus.NOT_IMPLEMENTED
            impl_pct = 0
            description = "N-WAY database: Missing Supabase credentials"
            missing_reqs = [IntegrationRequirement.NWAY_DATABASE]
        else:
            # Test actual Supabase connectivity with real query
            try:
                # Try to import supabase
                try:
                    from supabase import create_client, Client

                    has_supabase_lib = True
                except ImportError:
                    has_supabase_lib = False

                if not has_supabase_lib:
                    status = ComponentStatus.NOT_IMPLEMENTED
                    impl_pct = 10
                    description = "N-WAY database: Supabase library not installed"
                    missing_reqs = [IntegrationRequirement.NWAY_DATABASE]
                else:
                    # Test actual connection
                    client: Client = create_client(supabase_url, supabase_key)

                    # Try a simple query to verify connectivity
                    try:
                        # Test with a basic query - check if we can access any table
                        response = (
                            client.table("nway_interactions")
                            .select("count", count="exact")
                            .limit(1)
                            .execute()
                        )

                        if response and hasattr(response, "count"):
                            status = ComponentStatus.READY
                            impl_pct = 85
                            description = f"N-WAY database: Live Supabase connection verified with {response.count} patterns"
                            missing_reqs = []
                        else:
                            status = ComponentStatus.FRAMEWORK_ONLY
                            impl_pct = 30
                            description = "N-WAY database: Connected but pattern table not accessible"
                            missing_reqs = [IntegrationRequirement.NWAY_DATABASE]

                    except Exception as query_error:
                        # Connection works but specific table/query failed
                        status = ComponentStatus.FRAMEWORK_ONLY
                        impl_pct = 25
                        description = f"N-WAY database: Supabase connected but query failed: {str(query_error)[:100]}"
                        missing_reqs = [IntegrationRequirement.NWAY_DATABASE]

            except Exception as e:
                status = ComponentStatus.ERROR
                impl_pct = 5
                description = (
                    f"N-WAY database: Supabase connection failed: {str(e)[:100]}"
                )
                missing_reqs = [IntegrationRequirement.NWAY_DATABASE]

        return ComponentHealth(
            name="nway_integration",
            status=status,
            implementation_percentage=impl_pct,
            missing_requirements=missing_reqs,
            description=description,
        )

    async def _validate_frontend_ui(self) -> ComponentHealth:
        """Validate frontend implementation"""

        # Frontend exists but shows mock data
        status = ComponentStatus.FRAMEWORK_ONLY
        impl_pct = 40
        description = "Frontend UI framework complete but displays mock/hardcoded data"
        missing_reqs = [IntegrationRequirement.REAL_PROCESSING]

        return ComponentHealth(
            name="frontend_ui",
            status=status,
            implementation_percentage=impl_pct,
            missing_requirements=missing_reqs,
            description=description,
        )

    async def _validate_api_layer(self) -> ComponentHealth:
        """Validate API implementation"""

        # API endpoints exist but return static data
        status = ComponentStatus.FRAMEWORK_ONLY
        impl_pct = 35
        description = "API endpoints exist but return static/mock responses"
        missing_reqs = [IntegrationRequirement.REAL_PROCESSING]

        return ComponentHealth(
            name="api_layer",
            status=status,
            implementation_percentage=impl_pct,
            missing_requirements=missing_reqs,
            description=description,
        )

    async def _validate_knowledge_grounding(self) -> ComponentHealth:
        """Validate external knowledge integration with real Perplexity API test"""

        # Check for Perplexity API key
        perplexity_key = os.getenv("PERPLEXITY_API_KEY")

        if not perplexity_key:
            status = ComponentStatus.NOT_IMPLEMENTED
            impl_pct = 0
            description = "Knowledge grounding: Missing Perplexity API key"
            missing_reqs = [IntegrationRequirement.PERPLEXITY_API]
        else:
            # Test actual Perplexity API connectivity
            try:
                import httpx

                # Test Perplexity API with real request
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            "https://api.perplexity.ai/chat/completions",
                            headers={
                                "Authorization": f"Bearer {perplexity_key}",
                                "Content-Type": "application/json",
                            },
                            json={
                                "model": "sonar",
                                "messages": [
                                    {
                                        "role": "system",
                                        "content": "You are a health check assistant.",
                                    },
                                    {
                                        "role": "user",
                                        "content": "Respond with exactly: 'Knowledge grounding operational'",
                                    },
                                ],
                                "max_tokens": 20,
                                "temperature": 0.1,
                            },
                            timeout=10.0,
                        )

                    if response.status_code == 200:
                        result = response.json()
                        content = (
                            result.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                        )

                        if "operational" in content.lower():
                            status = ComponentStatus.READY
                            impl_pct = 85
                            description = "Knowledge grounding: Perplexity API connectivity verified - research capabilities ready"
                            missing_reqs = []
                        else:
                            status = ComponentStatus.FRAMEWORK_ONLY
                            impl_pct = 50
                            description = "Knowledge grounding: Perplexity API connected but response validation failed"
                            missing_reqs = [IntegrationRequirement.PERPLEXITY_API]
                    else:
                        status = ComponentStatus.ERROR
                        impl_pct = 20
                        description = f"Knowledge grounding: Perplexity API error - HTTP {response.status_code}"
                        missing_reqs = [IntegrationRequirement.PERPLEXITY_API]

                except Exception as api_error:
                    status = ComponentStatus.ERROR
                    impl_pct = 10
                    description = f"Knowledge grounding: Perplexity API connection failed: {str(api_error)[:100]}"
                    missing_reqs = [IntegrationRequirement.PERPLEXITY_API]

            except ImportError:
                status = ComponentStatus.NOT_IMPLEMENTED
                impl_pct = 5
                description = "Knowledge grounding: httpx library required for Perplexity integration"
                missing_reqs = [IntegrationRequirement.PERPLEXITY_API]

        return ComponentHealth(
            name="knowledge_grounding",
            status=status,
            implementation_percentage=impl_pct,
            missing_requirements=missing_reqs,
            description=description,
        )

    async def _validate_deliverable_generation(self) -> ComponentHealth:
        """Validate deliverable generation capabilities"""

        # Templates exist but no real content generation
        status = ComponentStatus.FRAMEWORK_ONLY
        impl_pct = 25
        description = "Deliverable templates exist but content generation requires real cognitive processing"
        missing_reqs = [IntegrationRequirement.REAL_PROCESSING]

        return ComponentHealth(
            name="deliverable_generation",
            status=status,
            implementation_percentage=impl_pct,
            missing_requirements=missing_reqs,
            description=description,
        )

    def _identify_implementation_gaps(
        self, components: List[ComponentHealth]
    ) -> List[str]:
        """Identify key implementation gaps"""

        gaps = []

        # Check for common patterns
        if any(
            IntegrationRequirement.CLAUDE_SONNET in c.missing_requirements
            for c in components
        ):
            gaps.append(
                "No Claude Sonnet 3.5 API integration - core cognitive processing unavailable"
            )

        if any(
            IntegrationRequirement.NWAY_DATABASE in c.missing_requirements
            for c in components
        ):
            gaps.append(
                "N-way interactions database not integrated with model selection algorithm"
            )

        if any(
            IntegrationRequirement.REAL_PROCESSING in c.missing_requirements
            for c in components
        ):
            gaps.append(
                "All responses are static templates - no real cognitive intelligence"
            )

        if any(
            IntegrationRequirement.PERPLEXITY_API in c.missing_requirements
            for c in components
        ):
            gaps.append("No external knowledge grounding or fact-checking capabilities")

        return gaps

    def _generate_next_steps(self, components: List[ComponentHealth]) -> List[str]:
        """Generate prioritized next steps for implementation"""

        steps = []

        # Priority 1: LLM Integration
        if any(
            IntegrationRequirement.CLAUDE_SONNET in c.missing_requirements
            for c in components
        ):
            steps.append(
                "1. Implement Claude Sonnet 3.5 API integration with proper error handling"
            )

        # Priority 2: Connect LLM to cognitive engine
        if any(
            IntegrationRequirement.REAL_PROCESSING in c.missing_requirements
            for c in components
        ):
            steps.append(
                "2. Replace static templates with real LLM calls in cognitive engine"
            )

        # Priority 3: N-way integration
        if any(
            IntegrationRequirement.NWAY_DATABASE in c.missing_requirements
            for c in components
        ):
            steps.append(
                "3. Integrate N-way interactions database with model selection logic"
            )

        # Priority 4: Frontend connection
        steps.append("4. Connect frontend to real API responses via WebSocket")

        # Priority 5: Knowledge grounding
        if any(
            IntegrationRequirement.PERPLEXITY_API in c.missing_requirements
            for c in components
        ):
            steps.append("5. Add Perplexity API for external knowledge grounding")

        return steps


# Global health validator instance
_health_validator_instance: Optional[SystemHealthValidator] = None


async def get_system_health_validator() -> SystemHealthValidator:
    """Get or create global health validator instance"""
    global _health_validator_instance

    if _health_validator_instance is None:
        _health_validator_instance = SystemHealthValidator()

    return _health_validator_instance
