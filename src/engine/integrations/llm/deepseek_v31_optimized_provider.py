#!/usr/bin/env python3
"""
DeepSeek V3.1 Optimized Provider - Implementing Official Guide Best Practices
Follows the complete DeepSeek V3.1 prompting guide recommendations:
- Official chat template syntax with special tokens
- Task-specific parameter tuning
- Enhanced domain vocabulary for expert activation
- Self-reflection prompts for complex reasoning
- Structured JSON output mode
"""

import asyncio
import httpx
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

from .provider_interface import (
    BaseLLMProvider,
    LLMResponse,
    ProviderAPIError,
    InvalidResponseError,
)


class DeepSeekV31Mode(Enum):
    """Official DeepSeek V3.1 API endpoints"""

    CHAT = "deepseek-chat"  # Non-thinking mode - fast, direct responses
    REASONER = "deepseek-reasoner"  # Thinking mode - chain-of-thought reasoning


@dataclass
class TaskConfig:
    """Task-specific configuration following DeepSeek V3.1 guide"""

    temperature: float
    top_p: float
    description: str
    use_self_reflection: bool = False
    requires_structured_output: bool = False


class DeepSeekV31OptimizedProvider(BaseLLMProvider):
    """
    Optimized DeepSeek V3.1 provider implementing complete guide recommendations

    Key optimizations:
    1. Official chat template with special tokens
    2. Task-specific parameter tuning
    3. Domain-rich vocabulary for expert activation
    4. Self-reflection prompts for complex reasoning
    5. Structured output support
    """

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        super().__init__(api_key, base_url)
        self.provider_name = "deepseek"

        # Task-specific configurations from DeepSeek V3.1 guide
        self._task_configs = {
            # Strategic Analysis - McKinsey/BCG precision
            "strategic_analysis": TaskConfig(
                temperature=0.3,
                top_p=0.95,
                description="Strategic consulting analysis requiring frameworks precision",
                use_self_reflection=True,
            ),
            "competitive_analysis": TaskConfig(
                temperature=0.4,
                top_p=0.95,
                description="Market and competitive intelligence analysis",
            ),
            # Problem Solving - balanced accuracy and creativity
            "problem_solving": TaskConfig(
                temperature=0.5,
                top_p=0.95,
                description="Structured problem-solving with creative solutions",
                use_self_reflection=True,
            ),
            "assumption_challenge": TaskConfig(
                temperature=0.6,
                top_p=0.95,
                description="Critical thinking and assumption testing",
                use_self_reflection=True,
            ),
            # Innovation - high creativity while maintaining coherence
            "innovation": TaskConfig(
                temperature=0.8,
                top_p=0.95,
                description="Creative ideation and breakthrough thinking",
            ),
            "design_thinking": TaskConfig(
                temperature=0.7,
                top_p=0.95,
                description="Human-centered design and creative solutions",
            ),
            # Implementation - high precision for execution
            "implementation": TaskConfig(
                temperature=0.2,
                top_p=0.95,
                description="Tactical implementation and execution planning",
            ),
            "project_management": TaskConfig(
                temperature=0.3,
                top_p=0.95,
                description="Project planning and resource optimization",
            ),
            # Research Synthesis - balance of accuracy and narrative
            "research_synthesis": TaskConfig(
                temperature=0.4,
                top_p=0.95,
                description="Research integration and evidence synthesis",
            ),
            # Data Extraction - maximum precision
            "data_extraction": TaskConfig(
                temperature=0.1,
                top_p=1.0,
                description="Factual data extraction and structured output",
                requires_structured_output=True,
            ),
            # Code Generation - high precision with minor variation
            "code_generation": TaskConfig(
                temperature=0.2,
                top_p=0.95,
                description="Code generation and technical implementation",
            ),
            # General balanced conversation
            "balanced_chat": TaskConfig(
                temperature=0.6,
                top_p=0.95,
                description="General conversational interaction",
            ),
        }

        # Enhanced consultant domain vocabularies for expert activation
        self._consultant_vocabularies = {
            "strategic_analyst": [
                "strategic positioning",
                "competitive advantage",
                "market dynamics",
                "value chain analysis",
                "blue ocean strategy",
                "Porter's Five Forces",
                "BCG matrix",
                "McKinsey 7S framework",
                "core competencies",
                "strategic options",
                "value proposition",
                "market segmentation",
                "differentiation strategy",
                "cost leadership",
                "strategic alliances",
            ],
            "synthesis_architect": [
                "design thinking",
                "systems thinking",
                "stakeholder mapping",
                "journey mapping",
                "service design",
                "human-centered design",
                "design sprint methodology",
                "empathy mapping",
                "ideation",
                "prototype development",
                "iterative design",
                "user experience",
                "holistic solutions",
                "integration patterns",
                "synergy creation",
            ],
            "implementation_driver": [
                "project management",
                "agile methodology",
                "waterfall approach",
                "resource allocation",
                "timeline optimization",
                "risk mitigation",
                "change management",
                "stakeholder engagement",
                "governance structure",
                "performance metrics",
                "KPI tracking",
                "execution excellence",
                "operational efficiency",
                "process optimization",
                "quality assurance",
            ],
        }

        self.logger.info(
            "🚀 DeepSeek V3.1 Optimized Provider initialized with guide best practices"
        )

    def get_available_models(self) -> List[str]:
        """Get available models"""
        return [DeepSeekV31Mode.CHAT.value, DeepSeekV31Mode.REASONER.value]

    async def is_available(self) -> bool:
        """Check API availability"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                headers = {"Authorization": f"Bearer {self.api_key}"}
                response = await client.get(f"{self.base_url}/models", headers=headers)
                return response.status_code == 200
        except Exception as e:
            self.logger.warning(f"⚠️ DeepSeek availability check failed: {e}")
            return False

    async def call_llm(self, messages: List[Dict], model: str, **kwargs) -> LLMResponse:
        """
        Standard LLM provider interface - Required abstract method implementation

        Converts generic message format to optimized call and delegates to call_optimized_llm.
        This method provides compatibility with the BaseLLMProvider interface.

        Args:
            messages: List of message dictionaries (OpenAI format)
            model: Model to use (will be mapped to DeepSeek modes)
            **kwargs: Additional parameters
        """
        try:
            # Extract system and user messages
            system_message = ""
            user_message = ""

            for msg in messages:
                if msg.get("role") == "system":
                    system_message = msg.get("content", "")
                elif msg.get("role") == "user":
                    user_message = msg.get("content", "")

            # Extract task-specific parameters from kwargs
            task_type = kwargs.get("task_type", "balanced_chat")
            consultant_role = kwargs.get("consultant_role")
            complexity_score = kwargs.get("complexity_score", 0.5)
            use_structured_output = kwargs.get("use_structured_output", False)

            # Delegate to optimized implementation
            result = await self.call_optimized_llm(
                prompt=user_message,
                task_type=task_type,
                consultant_role=consultant_role,
                complexity_score=complexity_score,
                use_structured_output=use_structured_output,
                custom_system_prompt=system_message if system_message else None,
            )

            # Record the call for tracking
            self.record_call(result, "call_llm")

            return result

        except Exception as e:
            self.logger.error(f"❌ call_llm failed: {e}")
            # Return error response compatible with LLMResponse
            return LLMResponse(
                content=f"Error: {str(e)}",
                provider="deepseek",
                model=model,
                tokens_used=0,
                cost_usd=0.0,
                response_time_ms=0,
                reasoning_steps=[],
                mental_models=[],
                confidence=0.0,
            )

    async def call_optimized_llm(
        self,
        prompt: str,
        task_type: str = "balanced_chat",
        consultant_role: str = None,
        complexity_score: float = 0.5,
        use_structured_output: bool = False,
        custom_system_prompt: str = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Call DeepSeek V3.1 with complete optimization following the guide

        Args:
            prompt: The user query/prompt
            task_type: Type of task for parameter optimization
            consultant_role: Role for domain vocabulary enhancement
            complexity_score: 0.0-1.0 complexity for mode selection
            use_structured_output: Whether to use JSON output mode
        """

        start_time = datetime.now()

        try:
            # 1. Select optimal mode based on complexity and task
            mode = self._select_optimal_mode(task_type, complexity_score)

            # 2. Get task-specific configuration
            task_config = self._task_configs.get(
                task_type, self._task_configs["balanced_chat"]
            )

            # 3. Build enhanced system prompt with domain vocabulary
            system_prompt = self._build_enhanced_system_prompt(
                consultant_role, task_type, task_config, custom_system_prompt
            )

            # 4. Add self-reflection if required for complex reasoning
            if task_config.use_self_reflection and mode == DeepSeekV31Mode.REASONER:
                prompt = self._add_self_reflection_prompts(prompt, task_type)

            # 5. Format with official DeepSeek V3.1 chat template
            formatted_prompt = self._format_with_official_template(
                system_prompt, prompt, mode
            )

            # 6. Prepare API request with optimized parameters
            request_payload = self._build_optimized_request(
                formatted_prompt, mode, task_config, use_structured_output
            )

            # 7. Execute API call
            response_data = await self._execute_api_call(request_payload, mode)

            # 8. Parse and enhance response
            return self._parse_optimized_response(
                response_data, mode, start_time, task_type
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            self.logger.error(f"DeepSeek V3.1 optimized call failed: {e}")

            return LLMResponse(
                content=f"Optimized analysis failed: {str(e)}",
                provider="deepseek_v31_optimized",
                model=mode.value if "mode" in locals() else "unknown",
                tokens_used=0,
                cost_usd=0.0,
                response_time_ms=int(execution_time),
                reasoning_steps=[],
                mental_models=[],
                confidence=0.0,
            )

    async def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.3,
        max_tokens: int = 2000,
        **kwargs,
    ):
        """
        Backward compatibility method for legacy LLM Manager interface.
        Returns the OLD LLMResponse format expected by LLM Manager.
        """
        # Import the OLD LLMResponse from providers
        from ...providers.llm.base import LLMResponse as OldLLMResponse

        # Convert to messages format for call_llm method
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Get response from new interface
        new_response = await self.call_llm(
            messages=messages,
            model="deepseek-chat",  # Use chat mode for compatibility
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        # Convert to old interface format
        return OldLLMResponse(
            raw_text=new_response.content,
            prompt_tokens=0,  # Not available in new response
            completion_tokens=new_response.tokens_used,
            total_tokens=new_response.tokens_used,
            cost=new_response.cost_usd,
            time_seconds=new_response.response_time_ms / 1000.0,
            model_name=new_response.model,
            provider_name=new_response.provider,
            metadata=new_response.metadata,
            raw_provider_response=new_response.raw_provider_response,
        )

    def _select_optimal_mode(
        self, task_type: str, complexity_score: float
    ) -> DeepSeekV31Mode:
        """Select optimal mode following guide recommendations"""

        # Tasks that definitely need thinking mode
        thinking_required_tasks = {
            "strategic_analysis",
            "problem_solving",
            "assumption_challenge",
            "complex_reasoning",
            "multi_step_analysis",
        }

        # Tasks that can use fast mode
        fast_mode_tasks = {
            "data_extraction",
            "code_generation",
            "research_synthesis",
            "quick_insights",
            "factual_lookup",
        }

        if task_type in thinking_required_tasks or complexity_score > 0.7:
            return DeepSeekV31Mode.REASONER
        elif task_type in fast_mode_tasks and complexity_score < 0.4:
            return DeepSeekV31Mode.CHAT
        else:
            # Default decision based on complexity
            return (
                DeepSeekV31Mode.REASONER
                if complexity_score > 0.5
                else DeepSeekV31Mode.CHAT
            )

    def _build_enhanced_system_prompt(
        self,
        consultant_role: str,
        task_type: str,
        task_config: TaskConfig,
        custom_system_prompt: str = None,
    ) -> str:
        """Build system prompt with enhanced domain vocabulary for expert activation"""

        if custom_system_prompt:
            return custom_system_prompt

        # Base expert identity
        if consultant_role == "strategic_analyst":
            base_identity = """You are a Senior Partner at a top-tier strategy consulting firm (McKinsey, BCG, Bain) with 15+ years of experience in strategic analysis, competitive intelligence, and value creation. Your expertise spans market entry, competitive positioning, strategic transformations, and corporate strategy development."""
        elif consultant_role == "synthesis_architect":
            base_identity = """You are a Principal Design Strategist and Systems Thinker with extensive experience in human-centered design, service design, and stakeholder alignment. You excel at synthesizing diverse perspectives, creating holistic solutions, and designing integrated approaches that address complex, multi-faceted challenges."""
        elif consultant_role == "implementation_driver":
            base_identity = """You are a Senior Implementation Director with deep expertise in project management, change management, and operational excellence. You specialize in translating strategic plans into executable roadmaps, optimizing resource allocation, and delivering measurable results within time and budget constraints."""
        else:
            base_identity = """You are an expert analyst with advanced pattern recognition and systematic reasoning capabilities."""

        # Add domain-specific vocabulary for expert network activation
        domain_vocab = []
        if consultant_role in self._consultant_vocabularies:
            vocab_list = self._consultant_vocabularies[consultant_role]
            domain_vocab = [
                f"Apply frameworks including: {', '.join(vocab_list[:8])}.",
                f"Consider methodologies such as: {', '.join(vocab_list[8:15])}.",
            ]

        # Task-specific guidance
        task_guidance = f"Task Focus: {task_config.description}"

        # Combine into enhanced system prompt
        system_components = [base_identity]
        system_components.extend(domain_vocab)
        system_components.append(task_guidance)
        system_components.append(
            "Provide structured, evidence-based analysis with clear reasoning and actionable insights."
        )

        return "\n\n".join(system_components)

    def _add_self_reflection_prompts(self, prompt: str, task_type: str) -> str:
        """Add self-reflection and critique prompts for complex reasoning"""

        self_reflection_instructions = """

After providing your initial analysis, please also include:

<self_reflection>
1. Identify three potential blind spots or limitations in your analysis
2. Consider what additional information would strengthen your conclusions
3. Suggest one alternative interpretation or approach to this problem
4. Assess the confidence level of your key recommendations (high/medium/low)
</self_reflection>"""

        return prompt + self_reflection_instructions

    def _format_with_official_template(
        self, system_prompt: str, user_prompt: str, mode: DeepSeekV31Mode
    ) -> str:
        """Format using official DeepSeek V3.1 chat template syntax"""

        # Official template with special tokens from the guide
        if mode == DeepSeekV31Mode.REASONER:
            # Thinking mode: ends with <think>
            template = f"<｜begin of sentence｜>{system_prompt}<｜User｜>{user_prompt}<｜Assistant｜><think>"
        else:
            # Non-thinking mode: ends with </think>
            template = f"<｜begin of sentence｜>{system_prompt}<｜User｜>{user_prompt}<｜Assistant｜></think>"

        return template

    def _build_optimized_request(
        self,
        formatted_prompt: str,
        mode: DeepSeekV31Mode,
        task_config: TaskConfig,
        use_structured_output: bool,
    ) -> Dict[str, Any]:
        """Build API request with task-optimized parameters"""

        # Base payload
        payload = {
            "model": mode.value,
            "messages": [{"role": "user", "content": formatted_prompt}],
            "temperature": task_config.temperature,
            "top_p": task_config.top_p,
            "stream": False,
        }

        # Reasoning mode optimizations
        if mode == DeepSeekV31Mode.REASONER:
            payload.update(
                {
                    "max_tokens": 4000,  # More tokens for reasoning
                    "temperature": max(task_config.temperature, 0.1),  # Minimum temp
                }
            )
        else:
            payload.update({"max_tokens": 2000})  # Standard tokens for fast mode

        # Structured output mode (JSON)
        if use_structured_output or task_config.requires_structured_output:
            payload["response_format"] = {"type": "json_object"}

            # Add JSON instruction to the prompt
            json_instruction = (
                "\n\nIMPORTANT: Provide your response as a valid JSON object."
            )
            payload["messages"][0]["content"] += json_instruction

        return payload

    async def _execute_api_call(
        self, payload: Dict[str, Any], mode: DeepSeekV31Mode
    ) -> Dict:
        """Execute optimized API call with proper timeout"""

        # Mode-specific timeouts from the guide
        timeout = 180.0 if mode == DeepSeekV31Mode.REASONER else 60.0

        async with httpx.AsyncClient(timeout=timeout) as client:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "METIS-V31-Optimized/1.0",
            }

            response = await client.post(
                f"{self.base_url}/v1/chat/completions", headers=headers, json=payload
            )

            if response.status_code != 200:
                error_msg = (
                    f"DeepSeek API error {response.status_code}: {response.text}"
                )
                self.logger.error(error_msg)
                raise ProviderAPIError(error_msg)

            return response.json()

    def _parse_optimized_response(
        self,
        response_data: Dict,
        mode: DeepSeekV31Mode,
        start_time: datetime,
        task_type: str,
    ) -> LLMResponse:
        """Parse response with V3.1 optimizations"""

        if "choices" not in response_data or not response_data["choices"]:
            raise InvalidResponseError("No choices in response")

        choice = response_data["choices"][0]
        message = choice.get("message", {})

        # Extract content and reasoning_content (V3.1 feature)
        content = message.get("content", "")
        reasoning_content = message.get("reasoning_content", "")

        # Process thinking mode content
        if mode == DeepSeekV31Mode.REASONER and reasoning_content:
            # Combine reasoning and final answer
            content = f"**Chain of Thought:**\n{reasoning_content}\n\n**Analysis:**\n{content}"

        # Extract usage and calculate cost
        usage = response_data.get("usage", {})
        total_tokens = usage.get("total_tokens", 0)

        # V3.1 pricing (from the guide)
        if mode == DeepSeekV31Mode.REASONER:
            input_cost = (usage.get("prompt_tokens", 0) / 1_000_000) * 0.55
            output_cost = (usage.get("completion_tokens", 0) / 1_000_000) * 2.19
        else:
            input_cost = (usage.get("prompt_tokens", 0) / 1_000_000) * 0.14
            output_cost = (usage.get("completion_tokens", 0) / 1_000_000) * 0.28

        total_cost = input_cost + output_cost

        # Calculate response time
        response_time = int((datetime.now() - start_time).total_seconds() * 1000)

        # Enhanced reasoning steps extraction
        reasoning_steps = self._extract_enhanced_reasoning_steps(content, mode)

        # Calculate confidence with V3.1 enhancements
        confidence = self._calculate_enhanced_confidence(content, mode, task_type)

        return LLMResponse(
            content=content,
            provider="deepseek_v31_optimized",
            model=mode.value,
            tokens_used=total_tokens,
            cost_usd=total_cost,
            response_time_ms=response_time,
            reasoning_steps=reasoning_steps,
            mental_models=[],
            confidence=confidence,
            raw_provider_response=response_data,  # RADICAL TRANSPARENCY: Complete API response
        )

    def _extract_enhanced_reasoning_steps(
        self, content: str, mode: DeepSeekV31Mode
    ) -> List[Dict[str, Any]]:
        """Enhanced reasoning step extraction"""

        reasoning_steps = []

        # Look for chain of thought section
        if "**Chain of Thought:**" in content:
            cot_section = content.split("**Chain of Thought:**")[1].split(
                "**Analysis:**"
            )[0]
            lines = cot_section.strip().split("\n")
        else:
            lines = content.split("\n")

        import re

        current_step = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect reasoning patterns
            if re.match(r"^\d+[\.\)]\s", line):
                if current_step:
                    reasoning_steps.append(current_step)
                current_step = {
                    "step_type": "numbered_reasoning",
                    "description": line,
                    "confidence": 0.9 if mode == DeepSeekV31Mode.REASONER else 0.8,
                }
            elif any(
                keyword in line.lower()
                for keyword in ["therefore", "because", "since", "given"]
            ):
                if current_step:
                    reasoning_steps.append(current_step)
                current_step = {
                    "step_type": "logical_inference",
                    "description": line,
                    "confidence": 0.85,
                }
            elif "<self_reflection>" in line.lower():
                reasoning_steps.append(
                    {
                        "step_type": "self_reflection",
                        "description": "Self-reflection and critique analysis",
                        "confidence": 0.9,
                    }
                )
            elif current_step:
                current_step["description"] += f" {line}"

        if current_step:
            reasoning_steps.append(current_step)

        # Ensure at least one step
        if not reasoning_steps:
            reasoning_steps.append(
                {
                    "step_type": "analysis",
                    "description": f"V3.1 {mode.value} analysis completed",
                    "confidence": 0.85 if mode == DeepSeekV31Mode.REASONER else 0.75,
                }
            )

        return reasoning_steps

    def _calculate_enhanced_confidence(
        self, content: str, mode: DeepSeekV31Mode, task_type: str
    ) -> float:
        """Calculate confidence with V3.1 and task-specific enhancements"""

        # Base confidence by mode
        base_confidence = 0.82 if mode == DeepSeekV31Mode.REASONER else 0.75

        # Task-specific confidence adjustments
        task_bonuses = {
            "strategic_analysis": 0.05,  # High confidence for strategy work
            "data_extraction": 0.08,  # Very high for factual tasks
            "implementation": 0.06,  # High for practical tasks
            "innovation": -0.03,  # Lower due to creative uncertainty
        }

        task_bonus = task_bonuses.get(task_type, 0.0)

        # Content quality indicators
        quality_indicators = [
            ("reasoning process", 0.03),
            ("evidence", 0.02),
            ("framework", 0.02),
            ("analysis", 0.01),
            ("conclusion", 0.01),
            ("self_reflection", 0.05),  # Bonus for self-reflection
        ]

        content_lower = content.lower()
        quality_bonus = sum(
            bonus
            for indicator, bonus in quality_indicators
            if indicator in content_lower
        )
        quality_bonus = min(quality_bonus, 0.12)  # Cap at 0.12

        # Uncertainty penalties
        uncertainty_phrases = ["uncertain", "unclear", "might", "possibly", "perhaps"]
        uncertainty_penalty = min(
            sum(0.02 for phrase in uncertainty_phrases if phrase in content_lower), 0.08
        )

        # Calculate final confidence
        confidence = base_confidence + task_bonus + quality_bonus - uncertainty_penalty

        return max(0.3, min(0.95, confidence))

    def get_task_configs(self) -> Dict[str, TaskConfig]:
        """Get all available task configurations"""
        return self._task_configs.copy()

    def get_consultant_vocabularies(self) -> Dict[str, List[str]]:
        """Get domain vocabularies for expert activation"""
        return self._consultant_vocabularies.copy()


# Convenience function for easy integration
async def execute_optimized_v31_analysis(
    prompt: str,
    task_type: str = "strategic_analysis",
    consultant_role: str = "strategic_analyst",
    complexity_score: float = 0.8,
    api_key: str = None,
) -> LLMResponse:
    """Execute analysis with full DeepSeek V3.1 optimizations"""

    if not api_key:
        import os

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment")

    provider = DeepSeekV31OptimizedProvider(api_key)

    return await provider.call_optimized_llm(
        prompt=prompt,
        task_type=task_type,
        consultant_role=consultant_role,
        complexity_score=complexity_score,
        use_structured_output=False,
    )


if __name__ == "__main__":
    import asyncio

    async def test_optimizations():
        """Test the optimized provider"""

        print("🧠 TESTING DEEPSEEK V3.1 OPTIMIZED PROVIDER")
        print("=" * 60)

        # Test strategic analysis
        test_query = """A B2B SaaS company is experiencing 15% monthly churn rate and needs a comprehensive retention strategy. The company has 500+ enterprise clients, average contract value of $50K annually, and operates in a competitive market with 3 major players."""

        try:
            result = await execute_optimized_v31_analysis(
                prompt=test_query,
                task_type="strategic_analysis",
                consultant_role="strategic_analyst",
                complexity_score=0.9,
            )

            print(f"✅ Mode Used: {result.model}")
            print(f"✅ Tokens Used: {result.tokens_used}")
            print(f"✅ Cost: ${result.cost_usd:.6f}")
            print(f"✅ Response Time: {result.response_time_ms}ms")
            print(f"✅ Confidence: {result.confidence:.2f}")
            print(f"✅ Reasoning Steps: {len(result.reasoning_steps)}")

            print("\nResponse Preview:")
            print(result.content[:300] + "...")

        except Exception as e:
            print(f"❌ Test failed: {e}")

    asyncio.run(test_optimizations())
