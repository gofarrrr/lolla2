"""
Claude Sonnet 3.5 Integration Client
Real LLM integration replacing all mock/hardcoded responses
"""

import os
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import httpx

# Import prompt capture system
try:
    from src.engine.adapters.workflow import  # Migrated get_prompt_capture, PromptPhase, PromptType

    PROMPT_CAPTURE_AVAILABLE = True
except ImportError:
    PROMPT_CAPTURE_AVAILABLE = False
    get_prompt_capture = None
    PromptPhase = None
    PromptType = None

# Load environment variables
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # dotenv not available

try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None


class LLMCallType(str, Enum):
    """Types of LLM calls for cost tracking"""

    MENTAL_MODEL = "mental_model"
    NWAY_ORCHESTRATION = "nway_orchestration"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"


@dataclass
class LLMResponse:
    """Structured LLM response with metadata"""

    content: str
    confidence: float
    reasoning_steps: List[str]
    evidence_sources: List[str]
    assumptions_made: List[str]
    tokens_used: int
    cost_usd: float
    processing_time_ms: float
    model_version: str
    call_type: LLMCallType

    # Prompt tracking fields
    prompt_id: Optional[str] = None
    response_id: Optional[str] = None
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    prompt_captured: bool = False


@dataclass
class LLMUsageMetrics:
    """Track LLM usage and costs"""

    total_calls: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    calls_by_type: Dict[LLMCallType, int] = None
    avg_processing_time_ms: float = 0.0

    def __post_init__(self):
        if self.calls_by_type is None:
            self.calls_by_type = {call_type: 0 for call_type in LLMCallType}


class ClaudeClient:
    """
    Production-ready Claude Sonnet 3.5 client
    Handles authentication, rate limiting, error handling, and cost tracking
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.usage_metrics = LLMUsageMetrics()

        # Rate limiting
        self.max_requests_per_minute = 50
        self.request_timestamps: List[datetime] = []

        # Cost tracking (approximate rates for Claude Sonnet 3.5)
        self.cost_per_input_token = 0.000003  # $3 per 1M tokens
        self.cost_per_output_token = 0.000015  # $15 per 1M tokens

        # Initialize prompt capture
        self.prompt_capture = None
        if PROMPT_CAPTURE_AVAILABLE:
            try:
                self.prompt_capture = get_prompt_capture()
                self.logger.info("✅ Prompt capture system initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize prompt capture: {e}")
        else:
            self.logger.warning("⚠️ Prompt capture system not available")

        # Initialize client
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Anthropic client with API key validation"""

        if not ANTHROPIC_AVAILABLE:
            self.logger.error(
                "Anthropic library not installed. Run: pip install anthropic>=0.34.0"
            )
            self.client = None
            return

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            self.logger.error("ANTHROPIC_API_KEY environment variable not set")
            self.client = None
            return

        try:
            self.client = anthropic.AsyncAnthropic(
                api_key=api_key,
                timeout=httpx.Timeout(30.0, read=60.0, write=30.0, connect=10.0),
            )
            self.logger.info("✅ Claude Sonnet 3.5 client initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Claude client: {e}")
            self.client = None

    async def is_available(self) -> bool:
        """Check if Claude client is available and working"""
        if not self.client:
            return False

        try:
            # Quick test call (minimal cost)
            response = await self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return True
        except Exception as e:
            self.logger.error(f"Claude availability check failed: {e}")
            return False

    async def _check_rate_limit(self):
        """Check and enforce rate limiting"""
        now = datetime.utcnow()

        # Remove timestamps older than 1 minute
        self.request_timestamps = [
            ts for ts in self.request_timestamps if (now - ts).total_seconds() < 60
        ]

        if len(self.request_timestamps) >= self.max_requests_per_minute:
            wait_time = 60 - (now - self.request_timestamps[0]).total_seconds()
            if wait_time > 0:
                self.logger.warning(f"Rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)

        self.request_timestamps.append(now)

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for token usage"""
        input_cost = input_tokens * self.cost_per_input_token
        output_cost = output_tokens * self.cost_per_output_token
        return input_cost + output_cost

    async def call_claude(
        self,
        prompt: str,
        call_type: LLMCallType,
        max_tokens: int = 2000,
        temperature: float = 0.3,
        system_prompt: Optional[str] = None,
        phase: Optional[PromptPhase] = None,
        engagement_id: Optional[str] = None,
        context_data: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """
        Make a call to Claude Sonnet 3.5 with proper error handling and metrics
        """

        if not self.client:
            raise RuntimeError(
                "Claude client not available. Check ANTHROPIC_API_KEY and installation."
            )

        # Capture prompt before API call
        prompt_id = None
        prompt_captured = False

        if self.prompt_capture and PROMPT_CAPTURE_AVAILABLE:
            try:
                # Map LLMCallType to PromptPhase if not provided
                if phase is None:
                    phase_mapping = {
                        LLMCallType.MENTAL_MODEL: PromptPhase.PROBLEM_STRUCTURING,
                        LLMCallType.NWAY_ORCHESTRATION: PromptPhase.ANALYSIS_EXECUTION,
                        LLMCallType.SYNTHESIS: PromptPhase.SYNTHESIS_DELIVERY,
                        LLMCallType.VALIDATION: PromptPhase.VERIFICATION,
                    }
                    phase = phase_mapping.get(call_type, PromptPhase.OTHER)

                prompt_id = self.prompt_capture.capture_prompt(
                    system_prompt=system_prompt or "",
                    user_prompt=prompt,
                    phase=phase,
                    prompt_type=PromptType.USER_PROMPT,
                    engagement_id=engagement_id,
                    provider="anthropic",
                    model="claude-3-5-sonnet-20241022",
                    context_data=context_data or {},
                )
                prompt_captured = True
                self.logger.debug(
                    f"📋 Prompt captured: {prompt_id[:8]} for {call_type.value}"
                )

            except Exception as e:
                self.logger.warning(f"⚠️ Failed to capture prompt: {e}")

        await self._check_rate_limit()

        start_time = datetime.utcnow()

        try:
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]

            # Make the API call
            request_params = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages,
            }

            # Only add system prompt if provided
            if system_prompt:
                request_params["system"] = system_prompt

            response = await self.client.messages.create(**request_params)

            # 🚨 CODE RED: BRUTAL LOGGING - RAW API RESPONSE
            self.logger.error("🚨 CODE RED DIAGNOSTIC - RAW CLAUDE RESPONSE:")
            self.logger.error(f"🚨 Response type: {type(response)}")
            self.logger.error(
                f"🚨 Response content length: {len(response.content) if response.content else 0}"
            )
            self.logger.error(f"🚨 Response usage: {response.usage}")
            self.logger.error(f"🚨 Response model: {response.model}")
            if response.content:
                self.logger.error(f"🚨 First content item: {response.content[0]}")
                self.logger.error(
                    f"🚨 First content text (first 200 chars): {response.content[0].text[:200] if hasattr(response.content[0], 'text') else 'NO TEXT ATTR'}"
                )

            # Extract response data
            content = response.content[0].text if response.content else ""
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            total_tokens = input_tokens + output_tokens

            # Calculate metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            cost = self._calculate_cost(input_tokens, output_tokens)

            # Update usage metrics
            self.usage_metrics.total_calls += 1
            self.usage_metrics.total_tokens += total_tokens
            self.usage_metrics.total_cost_usd += cost
            self.usage_metrics.calls_by_type[call_type] += 1

            # Update average processing time
            self.usage_metrics.avg_processing_time_ms = (
                self.usage_metrics.avg_processing_time_ms
                * (self.usage_metrics.total_calls - 1)
                + processing_time
            ) / self.usage_metrics.total_calls

            # Parse structured response if possible
            confidence, reasoning_steps, evidence_sources, assumptions = (
                self._parse_structured_response(content)
            )

            # 🚨 CODE RED: BRUTAL LOGGING - PARSED RESPONSE DATA
            self.logger.error("🚨 CODE RED DIAGNOSTIC - PARSED CLAUDE DATA:")
            self.logger.error(f"🚨 Content length: {len(content)}")
            self.logger.error(f"🚨 Content preview: {content[:500]}")
            self.logger.error(f"🚨 Parsed confidence: {confidence}")
            self.logger.error(
                f"🚨 Parsed reasoning_steps count: {len(reasoning_steps) if reasoning_steps else 0}"
            )
            self.logger.error(f"🚨 Parsed reasoning_steps: {reasoning_steps}")
            self.logger.error(f"🚨 Parsed evidence_sources: {evidence_sources}")
            self.logger.error(f"🚨 Parsed assumptions: {assumptions}")
            self.logger.error(
                f"🚨 Input tokens: {input_tokens}, Output tokens: {output_tokens}, Cost: {cost}"
            )
            self.logger.error(f"🚨 Processing time: {processing_time}ms")

            # Generate response ID for tracking
            import uuid

            response_id = str(uuid.uuid4())

            # Link response to prompt if capture is enabled
            if self.prompt_capture and prompt_captured and prompt_id:
                try:
                    self.prompt_capture.link_response(
                        prompt_id=prompt_id,
                        response_id=response_id,
                        response_tokens=output_tokens,
                        response_cost_usd=cost,
                        response_time_ms=processing_time,
                        success=True,
                        quality_score=confidence,  # Use extracted confidence as quality score
                    )
                    self.logger.debug(
                        f"🔗 Response linked: {prompt_id[:8]} -> {response_id[:8]}"
                    )
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to link response: {e}")

            self.logger.info(
                f"Claude call successful: {call_type.value} | "
                f"{total_tokens} tokens | ${cost:.4f} | {processing_time:.1f}ms"
                f"{' | prompt captured' if prompt_captured else ''}"
            )

            # 🚨 CODE RED: BRUTAL LOGGING - FINAL LLMResponse OBJECT
            llm_response = LLMResponse(
                content=content,
                confidence=confidence,
                reasoning_steps=reasoning_steps,
                evidence_sources=evidence_sources,
                assumptions_made=assumptions,
                tokens_used=total_tokens,
                cost_usd=cost,
                processing_time_ms=processing_time,
                model_version="claude-3-5-sonnet-20241022",
                call_type=call_type,
                prompt_id=prompt_id,
                response_id=response_id,
                system_prompt=system_prompt,
                user_prompt=prompt,
                prompt_captured=prompt_captured,
            )

            # 🚨 CODE RED: BRUTAL LOGGING - RETURNING LLMResponse
            self.logger.error("🚨 CODE RED DIAGNOSTIC - RETURNING LLMResponse:")
            self.logger.error(
                f"🚨 LLMResponse content length: {len(llm_response.content) if llm_response.content else 0}"
            )
            self.logger.error(f"🚨 LLMResponse confidence: {llm_response.confidence}")
            self.logger.error(
                f"🚨 LLMResponse reasoning_steps: {llm_response.reasoning_steps}"
            )
            self.logger.error(f"🚨 LLMResponse tokens_used: {llm_response.tokens_used}")
            self.logger.error(f"🚨 LLMResponse cost_usd: {llm_response.cost_usd}")

            return llm_response

        except Exception as e:
            # Link failed response to prompt if capture is enabled
            if self.prompt_capture and prompt_captured and prompt_id:
                try:
                    import uuid

                    error_response_id = str(uuid.uuid4())
                    self.prompt_capture.link_response(
                        prompt_id=prompt_id,
                        response_id=error_response_id,
                        response_tokens=0,
                        response_cost_usd=0.0,
                        response_time_ms=(
                            datetime.utcnow() - start_time
                        ).total_seconds()
                        * 1000,
                        success=False,
                        error_message=str(e),
                    )
                    self.logger.debug(
                        f"🔗 Error response linked: {prompt_id[:8]} -> {error_response_id[:8]}"
                    )
                except Exception as link_error:
                    self.logger.warning(
                        f"⚠️ Failed to link error response: {link_error}"
                    )

            self.logger.error(f"Claude API call failed: {e}")
            raise

    def _parse_structured_response(
        self, content: str
    ) -> Tuple[float, List[str], List[str], List[str]]:
        """
        Parse structured elements from Claude response
        Returns: (confidence, reasoning_steps, evidence_sources, assumptions)
        """

        # Default values
        confidence = 0.8  # Default confidence
        reasoning_steps = []
        evidence_sources = []
        assumptions = []

        try:
            # Look for confidence indicators
            if "high confidence" in content.lower():
                confidence = 0.9
            elif "medium confidence" in content.lower():
                confidence = 0.7
            elif "low confidence" in content.lower():
                confidence = 0.5
            elif "uncertain" in content.lower():
                confidence = 0.4

            # Extract reasoning steps (look for numbered lists or bullet points)
            lines = content.split("\n")
            for line in lines:
                line = line.strip()
                if (
                    line.startswith(("1.", "2.", "3.", "•", "-")) and len(line) > 10
                ):  # Avoid short lines
                    reasoning_steps.append(line)

            # Extract evidence sources (look for keywords)
            evidence_keywords = ["evidence", "data", "research", "study", "analysis"]
            for line in lines:
                if any(keyword in line.lower() for keyword in evidence_keywords):
                    if len(line.strip()) > 20:  # Avoid short lines
                        evidence_sources.append(line.strip())

            # Extract assumptions (look for assumption keywords)
            assumption_keywords = [
                "assume",
                "assumption",
                "given that",
                "provided that",
            ]
            for line in lines:
                if any(keyword in line.lower() for keyword in assumption_keywords):
                    if len(line.strip()) > 20:
                        assumptions.append(line.strip())

        except Exception as e:
            self.logger.warning(f"Failed to parse structured response: {e}")

        return confidence, reasoning_steps, evidence_sources, assumptions

    def get_usage_metrics(self) -> Dict[str, Any]:
        """Get current usage metrics"""
        return {
            "total_calls": self.usage_metrics.total_calls,
            "total_tokens": self.usage_metrics.total_tokens,
            "total_cost_usd": self.usage_metrics.total_cost_usd,
            "calls_by_type": {
                k.value: v for k, v in self.usage_metrics.calls_by_type.items()
            },
            "avg_processing_time_ms": self.usage_metrics.avg_processing_time_ms,
            "estimated_monthly_cost": (
                self.usage_metrics.total_cost_usd * 30
                if self.usage_metrics.total_calls > 0
                else 0
            ),
        }

    def get_prompt_analytics(self) -> Optional[Dict[str, Any]]:
        """Get prompt capture analytics if available"""
        if not self.prompt_capture or not PROMPT_CAPTURE_AVAILABLE:
            return None

        try:
            analytics = self.prompt_capture.get_prompt_analytics()
            return {
                "total_prompts": analytics.total_prompts,
                "unique_templates": analytics.unique_templates,
                "avg_prompt_length": analytics.avg_prompt_length,
                "avg_response_time_ms": analytics.avg_response_time_ms,
                "success_rate": analytics.success_rate,
                "cost_per_prompt_usd": analytics.cost_per_prompt_usd,
                "prompts_by_phase": {
                    k.value if hasattr(k, "value") else str(k): v
                    for k, v in analytics.prompts_by_phase.items()
                },
                "avg_quality_score": analytics.avg_quality_score,
                "prompts_per_hour": analytics.prompts_per_hour,
            }
        except Exception as e:
            self.logger.error(f"Failed to get prompt analytics: {e}")
            return None

    def export_prompts(self, output_path: Optional[str] = None) -> Optional[str]:
        """Export captured prompts to file"""
        if not self.prompt_capture or not PROMPT_CAPTURE_AVAILABLE:
            self.logger.warning("Prompt capture not available for export")
            return None

        try:
            return self.prompt_capture.export_prompts(output_path=output_path)
        except Exception as e:
            self.logger.error(f"Failed to export prompts: {e}")
            return None

    async def test_connection(self) -> Dict[str, Any]:
        """Test Claude connection and return diagnostics"""

        diagnostics = {
            "anthropic_library": ANTHROPIC_AVAILABLE,
            "api_key_present": bool(os.getenv("ANTHROPIC_API_KEY")),
            "client_initialized": self.client is not None,
            "connection_test": False,
            "error_message": None,
        }

        if not diagnostics["anthropic_library"]:
            diagnostics["error_message"] = "Anthropic library not installed"
            return diagnostics

        if not diagnostics["api_key_present"]:
            diagnostics["error_message"] = "ANTHROPIC_API_KEY not set"
            return diagnostics

        if not diagnostics["client_initialized"]:
            diagnostics["error_message"] = "Client initialization failed"
            return diagnostics

        try:
            # Test with minimal call
            response = await self.call_claude(
                prompt="Test connection. Respond with 'OK'.",
                call_type=LLMCallType.VALIDATION,
                max_tokens=10,
            )
            diagnostics["connection_test"] = True
        except Exception as e:
            diagnostics["error_message"] = str(e)

        return diagnostics


# Global Claude client instance
_claude_client_instance: Optional[ClaudeClient] = None


async def get_claude_client() -> ClaudeClient:
    """Get or create global Claude client instance"""
    global _claude_client_instance

    if _claude_client_instance is None:
        _claude_client_instance = ClaudeClient()

    return _claude_client_instance
