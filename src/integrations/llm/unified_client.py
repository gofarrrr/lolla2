#!/usr/bin/env python3
"""
Unified LLM Client
Orchestrates multiple LLM providers and provides unified interface
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from .provider_interface import (
    LLMResponse,
    CognitiveAnalysisResult,
    ProviderUnavailableError,
)
from .claude_provider import ClaudeProvider
from .openai_provider import OpenAIProvider
from .openrouter_provider import OpenRouterProvider
from .deepseek_provider import DeepSeekProvider  # Use local bridge to avoid circular import
from .cognitive_analyzer import CognitiveAnalyzer
from src.core.unified_context_stream import get_unified_context_stream, ContextEventType

# Import pipeline refactoring components (P0 #0)
from .pipeline import create_llm_pipeline, LLMCallContext, LLMCallPipeline

# Import intelligent caching system
try:
    from src.engine.core.intelligent_cache import get_cache_manager, CacheStrategy

    INTELLIGENT_CACHE_AVAILABLE = True
except ImportError:
    INTELLIGENT_CACHE_AVAILABLE = False

# Load environment variables with enhanced error handling
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent.parent.parent / ".env"

    if env_path.exists():
        load_dotenv(env_path, override=True)
except ImportError:
    pass  # dotenv not available


class UnifiedLLMClient:
    """Unified interface for all LLM providers with cognitive analysis capabilities"""

    CONFIDENCE_FALLBACK_THRESHOLD = 0.85

    def __init__(
        self,
        pii_redaction_enabled: bool = True,
        sensitivity_routing_enabled: bool = True,
        output_contracts_enabled: bool = True,
        injection_firewall_enabled: bool = True,
        grounding_contract_enabled: bool = True,
        self_verification_enabled: bool = True
    ):
        """
        Initialize UnifiedLLMClient with enterprise security features.

        Args:
            pii_redaction_enabled: Enable PII redaction (default: True)
            sensitivity_routing_enabled: Enable sensitivity-based provider routing (default: True)
            output_contracts_enabled: Enable output contract validation (default: True)
            injection_firewall_enabled: Enable prompt injection detection (default: True)
            grounding_contract_enabled: Enable response grounding validation (default: True)
            self_verification_enabled: Enable response self-verification (default: True)
        """
        self.logger = logging.getLogger(__name__)
        self._providers = {}
        self._call_recorder = []  # For audit trail

        # ENTERPRISE SECURITY (Phase 6): PII Redaction + Sensitivity Routing
        self.pii_redaction_enabled = pii_redaction_enabled
        self.sensitivity_routing_enabled = sensitivity_routing_enabled

        if self.pii_redaction_enabled:
            try:
                from src.engine.security.pii_redaction import get_pii_redaction_engine
                self.pii_engine = get_pii_redaction_engine(enabled=True)
                self.logger.info("✅ PII redaction enabled for LLM calls")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize PII redaction: {e}")
                self.pii_engine = None
        else:
            self.pii_engine = None

        if self.sensitivity_routing_enabled:
            try:
                from src.engine.security.sensitivity_routing import get_sensitivity_router
                self.sensitivity_router = get_sensitivity_router(enabled=True)
                self.logger.info("✅ Sensitivity routing enabled for LLM calls")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize sensitivity routing: {e}")
                self.sensitivity_router = None
        else:
            self.sensitivity_router = None

        # ENTERPRISE SECURITY (Feature 3/6): Injection Firewall
        self.injection_firewall_enabled = injection_firewall_enabled
        if self.injection_firewall_enabled:
            try:
                from src.engine.security.injection_firewall import (
                    get_injection_firewall,
                    FirewallAction,
                    InjectionSeverity,
                )
                self.injection_firewall = get_injection_firewall(
                    enabled=True,
                    action_mode=FirewallAction.SANITIZE,
                    block_threshold=InjectionSeverity.HIGH
                )
                self.logger.info("✅ Injection firewall enabled for LLM calls")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize injection firewall: {e}")
                self.injection_firewall = None
        else:
            self.injection_firewall = None

        # ENTERPRISE ROBUSTNESS (Feature 2/6): Output Contracts
        self.output_contracts_enabled = output_contracts_enabled
        if self.output_contracts_enabled:
            try:
                from src.engine.contracts import (
                    get_contract_prompt,
                    validate_against_contract,
                    get_contract,
                    is_contract_validation_enabled,
                )
                self._get_contract_prompt = get_contract_prompt
                self._validate_contract = validate_against_contract
                self._get_contract = get_contract
                self._is_contract_validation_enabled = is_contract_validation_enabled
                self.logger.info("✅ Output contracts enabled for LLM calls")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize output contracts: {e}")
                self.output_contracts_enabled = False
        else:
            self.logger.warning("⚠️ Output contracts DISABLED")

        # ENTERPRISE QUALITY (Feature 4/6): Grounding Contract
        self.grounding_contract_enabled = grounding_contract_enabled
        if self.grounding_contract_enabled:
            try:
                from src.engine.quality.grounding_contract import get_grounding_contract
                self.grounding_contract = get_grounding_contract(enabled=True)
                self.logger.info("✅ Grounding contract enabled for LLM calls")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize grounding contract: {e}")
                self.grounding_contract = None
        else:
            self.grounding_contract = None

        # ENTERPRISE QUALITY (Feature 5/6): Self-Verification
        self.self_verification_enabled = self_verification_enabled
        if self.self_verification_enabled:
            try:
                from src.engine.quality.self_verification import get_self_verifier
                self.self_verifier = get_self_verifier(enabled=True)
                self.logger.info("✅ Self-verification enabled for LLM calls")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize self-verification: {e}")
                self.self_verifier = None
        else:
            self.self_verifier = None

        # Initialize intelligent caching
        self._init_intelligent_cache()

        # Initialize providers
        self._initialize_providers()

        # Lazily import telemetry helpers
        try:
            from src.telemetry.style import score_style as _score_style  # type: ignore
            self._score_style = _score_style
        except Exception:
            self._score_style = lambda x: None  # type: ignore

        # Initialize cognitive analyzer
        self.cognitive = CognitiveAnalyzer(self)

        # PIPELINE REFACTORING (P0 #0): Initialize LLM call pipeline AFTER providers
        self._init_pipeline()

        # Runtime quality signals accumulator (for summary metrics)
        self._runtime_quality_scores: Dict[str, float] = {}

    def _init_intelligent_cache(self):
        """Initialize intelligent caching system for LLM responses"""
        if not INTELLIGENT_CACHE_AVAILABLE:
            self.logger.warning(
                "⚠️ Intelligent cache not available - responses will not be cached"
            )
            self.cache_enabled = False
            self.llm_cache = None
            return

        try:
            # Get cache manager - it's a singleton, so caches persist across instances
            manager = get_cache_manager()

            # Get existing caches or create new ones if they don't exist
            # This prevents "Cache already exists" warnings on multiple UnifiedLLMClient instances

            self.claude_cache = manager.get_cache("claude_responses")
            if not self.claude_cache:
                self.claude_cache = manager.create_cache(
                    name="claude_responses",
                    strategy=CacheStrategy.TTL,  # Claude has built-in caching
                    max_size_mb=50.0,
                    default_ttl=300,  # 5 minutes - Claude's built-in caching
                )

            self.deepseek_cache = manager.get_cache("deepseek_responses")
            if not self.deepseek_cache:
                self.deepseek_cache = manager.create_cache(
                    name="deepseek_responses",
                    strategy=CacheStrategy.LFU,  # Frequent access pattern for budget optimization
                    max_size_mb=100.0,
                    default_ttl=3600,  # 1 hour - budget optimization
                )

            self.deepseek_reasoning_cache = manager.get_cache("deepseek_reasoning")
            if not self.deepseek_reasoning_cache:
                self.deepseek_reasoning_cache = manager.create_cache(
                    name="deepseek_reasoning",
                    strategy=CacheStrategy.LFU,  # Cache expensive reasoning
                    max_size_mb=200.0,
                    default_ttl=7200,  # 2 hours - expensive to regenerate
                )

            self.cache_enabled = True
            self.logger.info(
                "✅ Intelligent LLM caching initialized (reusing existing caches if available)"
            )

        except Exception as e:
            self.logger.warning(f"⚠️ Cache initialization failed: {e}")
            self.cache_enabled = False
            self.llm_cache = None

    def _init_pipeline(self):
        """
        PIPELINE REFACTORING (P0 #0): Initialize LLM call pipeline with all stages.

        This replaces inline logic in call_llm() with modular pipeline stages.
        """
        try:
            # Import helper functions for pipeline stages
            get_contract_prompt_func = None
            build_rag_context_func = None
            reasoning_selector = None

            # Import contract prompt function if contracts enabled
            if self.output_contracts_enabled:
                try:
                    from src.engine.contracts import get_contract_prompt
                    get_contract_prompt_func = get_contract_prompt
                except Exception:
                    pass

            # Import RAG context builder (always import, stage will check feature flag)
            try:
                from src.engine.retrieval.context_injector import build_context_system_message
                build_rag_context_func = build_context_system_message
            except Exception:
                pass

            # Import reasoning mode selector
            try:
                from src.services.reasoning_mode_selector import get_reasoning_mode_selector
                reasoning_selector = get_reasoning_mode_selector()
            except Exception:
                pass

            # Create pipeline with all dependencies
            self.pipeline = create_llm_pipeline(
                # Stage dependencies
                injection_firewall=self.injection_firewall if self.injection_firewall_enabled else None,
                pii_engine=self.pii_engine if self.pii_redaction_enabled else None,
                sensitivity_router=self.sensitivity_router if self.sensitivity_routing_enabled else None,
                get_contract_prompt_func=get_contract_prompt_func,
                build_rag_context_func=build_rag_context_func,
                reasoning_selector=reasoning_selector,
                style_scorer=self._score_style,
                style_gate=None,  # Will be imported in stage
                escalate_func=self._escalate_on_low_confidence,
                # Configuration
                available_providers=list(self._providers.keys()),
                rag_k=3,
                confidence_threshold=self.CONFIDENCE_FALLBACK_THRESHOLD,
                # Enable/disable flags (match existing feature flags)
                enable_injection_firewall=self.injection_firewall_enabled,
                enable_pii_redaction=self.pii_redaction_enabled,
                enable_sensitivity_routing=self.sensitivity_routing_enabled,
                enable_output_contracts=self.output_contracts_enabled,
                enable_rag_context=True,  # Always enabled, stage checks FF_RAG_DECAY_RETRIEVAL
                enable_provider_adapter=True,
                enable_reasoning_mode=True,
                enable_style_gate=True,
                enable_confidence_escalation=True,
            )

            self.logger.info("✅ Pipeline initialized with 9 stages")

        except Exception as e:
            self.logger.error(f"❌ Pipeline initialization failed: {e}")
            self.logger.error("   Pipeline is REQUIRED - cannot proceed without it")
            raise RuntimeError("LLM pipeline initialization failed - legacy code removed") from e

    def _initialize_providers(self):
        """Initialize available LLM providers with enhanced diagnostics"""

        self.logger.info("🔍 Initializing LLM providers...")

        # Check environment variables
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        openrouter_key = os.getenv("OPENROUTER_API_KEY")

        self.logger.info("🔑 Environment check:")
        self.logger.info(
            f"   ANTHROPIC_API_KEY: {'✅ Present' if anthropic_key else '❌ Missing'}"
        )
        self.logger.info(
            f"   DEEPSEEK_API_KEY: {'✅ Present' if deepseek_key else '❌ Missing'}"
        )
        self.logger.info(
            f"   OPENAI_API_KEY: {'✅ Present' if openai_key else '❌ Missing'}"
        )
        self.logger.info(
            f"   OPENROUTER_API_KEY: {'✅ Present' if openrouter_key else '❌ Missing'}"
        )

        # Initialize OpenRouter/Grok-4-Fast (Priority provider for cost-efficiency)
        if openrouter_key:
            try:
                self._providers["openrouter"] = OpenRouterProvider(openrouter_key)
                self.logger.info("✅ OpenRouter/Grok-4-Fast provider initialized (PRIMARY)")

            except Exception as e:
                self.logger.error(f"❌ Failed to initialize OpenRouter provider: {e}")
        else:
            self.logger.warning("❌ OPENROUTER_API_KEY not found - OpenRouter disabled")

        # Initialize Anthropic/Claude (Fallback provider)
        if anthropic_key:
            try:
                self._providers["anthropic"] = ClaudeProvider(anthropic_key)
                self.logger.info("✅ Claude/Anthropic provider initialized")

                # Test connectivity asynchronously (store for later use)
                self._anthropic_available = None  # Will be set on first use

            except Exception as e:
                self.logger.error(f"❌ Failed to initialize Claude provider: {e}")
        else:
            self.logger.warning("❌ ANTHROPIC_API_KEY not found - Claude disabled")

        # Initialize DeepSeek (Budget-friendly alternative to Anthropic)
        if deepseek_key:
            try:
                self._providers["deepseek"] = DeepSeekProvider(deepseek_key)
                self.logger.info("✅ DeepSeek provider initialized")

            except Exception as e:
                self.logger.error(f"❌ Failed to initialize DeepSeek provider: {e}")
        else:
            self.logger.info("ℹ️ DeepSeek API key not found - DeepSeek disabled")

        # Initialize OpenAI
        if openai_key:
            try:
                self._providers["openai"] = OpenAIProvider(openai_key)
                self.logger.info("✅ OpenAI provider initialized")

            except Exception as e:
                self.logger.error(f"❌ Failed to initialize OpenAI provider: {e}")
        else:
            self.logger.info("ℹ️ OpenAI API key not found - OpenAI disabled")

        # Final status
        if self._providers:
            providers = list(self._providers.keys())
            self.logger.info(f"🎯 LLM providers initialized: {providers}")
        else:
            self.logger.error(
                "❌ CRITICAL: No LLM providers available - system will fail"
            )
            self.logger.error("   Check API keys in .env file")

    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return list(self._providers.keys())

    def _prompt_hash(self, messages: List[Dict]) -> str:
        import json, hashlib
        payload = json.dumps(messages, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _aggregate_confidence(self, response: LLMResponse) -> float:
        # Combine provider confidence and available quality scores
        base = max(0.0, min(1.0, float(getattr(response, "confidence", 0.0))))
        qs = {}
        try:
            qs = (response.metadata or {}).get("quality_scores", {})
        except Exception:
            qs = {}
        signals = [base]
        for k in ("groundedness", "self_verification", "coherence", "consistency"):
            v = qs.get(k)
            if v is not None:
                try:
                    signals.append(float(v))
                except Exception:
                    pass
        if not signals:
            return base
        return sum(signals) / len(signals)

    async def _log_turn(
        self,
        *,
        messages: List[Dict],
        response: LLMResponse,
        phase: Optional[str],
        engagement_id: Optional[str],
        contract_valid: Optional[bool] = None,
        groundedness: Optional[float] = None,
        self_ver: Optional[float] = None,
        style_score: Optional[float] = None,
        context_ids: Optional[List[str]] = None,
    ) -> None:
        try:
            from src.telemetry.turn_logger import log_turn, TurnLogRecord, ValidationVerdicts
        except Exception:
            return
        prompt_hash = self._prompt_hash(messages)
        pt = int(getattr(response, "prompt_tokens", 0) or 0)
        ct = int(getattr(response, "completion_tokens", 0) or 0)
        total = int(getattr(response, "tokens_used", pt + ct) or 0)
        record = TurnLogRecord(
            timestamp=datetime.now().isoformat(),
            prompt_hash=prompt_hash,
            provider=response.provider,
            model=response.model,
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=total,
            cost_usd=float(getattr(response, "cost_usd", 0.0)),
            latency_ms=float(getattr(response, "response_time_ms", 0.0)),
            confidence=self._aggregate_confidence(response),
            context_ids=[str(x) for x in (context_ids or [])],
            engagement_id=engagement_id,
            phase=phase,
            validation=ValidationVerdicts(
                contract_valid=contract_valid,
                groundedness=groundedness,
                self_verification=self_ver,
                style_score=style_score,
            ),
            extra={},
        )
        log_turn(record)
        # Drift monitor
        try:
            from src.telemetry.drift import get_drift_monitor
            dm = get_drift_monitor()
            dm.record(record.confidence, record.validation.style_score)
        except Exception:
            pass

    async def _escalate_on_low_confidence(
        self,
        messages: List[Dict],
        response: LLMResponse,
        *,
        phase: Optional[str] = None,
        engagement_id: Optional[str] = None,
        context_data: Optional[Dict[str, Any]] = None,
        threshold: float = None,
    ) -> LLMResponse:
        """Escalate to a stronger provider if confidence is below threshold."""
        thr = threshold or self.CONFIDENCE_FALLBACK_THRESHOLD
        conf = self._aggregate_confidence(response)
        if conf >= thr:
            return response
        # Try Anthropic escalation first, then OpenAI
        try:
            esc = await self.call_llm(
                messages=messages,
                model="claude-3-5-sonnet-20241022",
                provider="anthropic",
                phase=phase,
                engagement_id=engagement_id,
            )
            return esc
        except Exception:
            try:
                esc2 = await self.call_llm(
                    messages=messages,
                    model="gpt-4o-mini",
                    provider="openai",
                    phase=phase,
                    engagement_id=engagement_id,
                )
                return esc2
            except Exception:
                return response

    async def call_llm(
        self,
        messages: List[Dict],
        model: str = "grok-4-fast",
        provider: str = "openrouter",
        functions: Optional[List[Dict]] = None,
        response_format: Optional[Dict[str, str]] = None,
        sensitivity_override: Optional[str] = None,
        output_contract: Optional[str] = None,
        engagement_id: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        V2.1 MASTER COMMUNICATOR - Multi-Provider LLM Routing
        ENTERPRISE SECURITY (Phase 6): PII redaction + sensitivity-based provider routing
        ENTERPRISE SECURITY (Feature 3/6): Injection firewall
        ENTERPRISE ROBUSTNESS (Feature 2/6): Output contract validation

        Supports routing to different providers for Two-Brain Senior Advisor architecture

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model name (deepseek-chat, claude-3-5-sonnet-20241022, etc.)
            provider: Provider to use ("deepseek", "claude", "anthropic")
            functions: Function schemas for tool calling (optional)
            response_format: Response format specification (optional)
            sensitivity_override: Manual sensitivity level override (low/medium/high/critical)
            output_contract: Contract name for output validation (analysis/structured_query/classification)
            engagement_id: Optional engagement/trace identifier for cost + audit logging
            **kwargs: Additional parameters (max_tokens, temperature, etc.)

        Returns:
            LLMResponse: Response from the specified provider

        Raises:
            ProviderUnavailableError: If the requested provider is not available
            ValueError: If invalid provider specified
            InjectionAttemptError: If injection attempt is blocked
            Exception: For any other errors
        """
        # Pop engagement metadata from kwargs if supplied via legacy call sites
        engagement_id = kwargs.pop("engagement_id", engagement_id)
        phase = kwargs.get("phase")

        # ========================================================================
        # PIPELINE EXECUTION (Pre-LLM Stages 1-7)
        # ========================================================================
        # All pre-LLM processing now happens in pipeline:
        #   1. InjectionFirewallStage
        #   2. PIIRedactionStage
        #   3. SensitivityRoutingStage
        #   4. OutputContractStage
        #   5. RAGContextInjectionStage
        #   6. ProviderAdapterStage
        #   7. ReasoningModeStage
        # ========================================================================

        try:
            # Create pipeline context with all kwargs merged
            merged_kwargs = kwargs.copy()
            merged_kwargs.update({
                "functions": functions,
                "response_format": response_format,
                "sensitivity_override": sensitivity_override,
                "output_contract": output_contract,
                "phase": phase,
            })

            context = LLMCallContext(
                messages=messages,
                model=model,
                provider=provider,
                kwargs=merged_kwargs
            )

            # Execute pipeline (pre-LLM stages only: 1-7)
            self.logger.info("🚀 Executing pipeline (pre-LLM stages)")
            context = await self.pipeline.execute(context)

            # Extract modified values from pipeline
            messages = context.get_effective_messages()
            provider = context.get_effective_provider()
            model = context.get_effective_model()
            call_kwargs = context.get_effective_kwargs()

            # Remove pipeline-specific kwargs that aren't LLM parameters
            call_kwargs.pop("sensitivity_override", None)
            call_kwargs.pop("output_contract", None)

            # Note: functions and response_format have been filtered by ProviderAdapterStage
            # so they're only in call_kwargs if the provider supports them

            self.logger.info(f"✅ Pipeline complete: {len(context.stage_metadata)} stages executed")

        except Exception as e:
            self.logger.error(f"❌ Pipeline execution failed: {e}")
            raise  # Fail fast - no legacy fallback

        # ========================================================================
        # PROVIDER VALIDATION & SETUP (Orchestration logic, not pipeline processing)
        # ========================================================================

        # Normalize provider names
        if provider in ["claude", "anthropic"]:
            provider_key = "anthropic"
        elif provider == "deepseek":
            provider_key = "deepseek"
        elif provider == "openai":
            provider_key = "openai"
        elif provider in ["openrouter", "grok", "grok-4-fast"]:
            provider_key = "openrouter"
        else:
            raise ValueError(
                f"Unsupported provider: {provider}. Use 'openrouter', 'deepseek', 'claude', or 'openai'"
            )

        # Check if provider is available
        if provider_key not in self._providers:
            raise ProviderUnavailableError(f"{provider} provider not initialized")

        provider_instance = self._providers[provider_key]

        # SAFETY: Validate model/provider combination (Phase 5)
        try:
            from src.engine.services.llm.model_registry import get_model_registry

            registry = get_model_registry()
            registry.validate_or_raise(provider_key, model)
        except ValueError as e:
            self.logger.error(f"❌ Model validation failed: {e}")
            raise

        # Check provider availability
        if not await provider_instance.is_available():
            raise ProviderUnavailableError(f"{provider} provider not available")

        try:
            self.logger.info(
                f"🤖 V2.1 Routing: {provider} with {len(messages)} messages, model: {model}"
            )

            # ====================================================================
            # LLM CALL (Provider-specific logic now handled by ProviderAdapterStage)
            # ====================================================================
            response = await provider_instance.call_llm(messages, model, **call_kwargs)

            # Ensure metadata dict
            if response.metadata is None:
                try:
                    response.metadata = {}
                except Exception:
                    pass

            # Style score and gate (Phase 2)
            try:
                style_score = self._score_style(response.content) or None
            except Exception:
                style_score = None
            # Evaluate style gate
            try:
                from src.telemetry.style_gate import evaluate as _eval_style
                action = _eval_style(style_score, phase=phase)
                if isinstance(response.metadata, dict):
                    response.metadata.setdefault("style_gate", {})["action"] = action
                    response.metadata["style_gate"]= {**response.metadata.get("style_gate", {}), "score": style_score}
                if action == "block":
                    # Replace content; lower confidence
                    response.content = "Output blocked by style policy"
                    try:
                        response.confidence = min(float(getattr(response, "confidence", 0.0) or 0.0), 0.2)
                    except Exception:
                        pass
            except Exception:
                pass

            # Optional: prompt/schema version tracking (Phase 1)
            try:
                versions = {"model": model, "provider": provider_key}
                prompt_name = call_kwargs.get("prompt_name") or kwargs.get("prompt_name")
                prompt_tmpl = call_kwargs.get("prompt_template") or kwargs.get("prompt_template")
                if prompt_name and prompt_tmpl:
                    from src.engine.services.prompt_version_registry import get_prompt_version_registry
                    preg = get_prompt_version_registry()
                    pv = preg.register_prompt(prompt_name, prompt_tmpl)
                    preg.bind(prompt_name, provider_key, model)
                    versions.update({"prompt_name": prompt_name, "prompt_version": pv})
                schema_name = call_kwargs.get("schema_name") or kwargs.get("schema_name")
                output_schema = call_kwargs.get("output_schema") or kwargs.get("output_schema")
                if schema_name and isinstance(output_schema, dict):
                    from src.engine.services.schema_registry import get_schema_registry
                    sreg = get_schema_registry()
                    sv = sreg.register_schema(schema_name, output_schema)
                    versions.update({"schema_name": schema_name, "schema_version": sv})
                # Attach to metadata
                if isinstance(response.metadata, dict):
                    response.metadata.setdefault("versions", {}).update(versions)
            except Exception:
                pass

            # Collect quality signals if present
            qs = (response.metadata or {}).get("quality_scores") if response.metadata else {}
            groundedness = float(qs.get("groundedness")) if qs and qs.get("groundedness") is not None else None
            self_ver = float(qs.get("self_verification")) if qs and qs.get("self_verification") is not None else None

            # Optional shadow self-consistency (Phase 1) behind flag
            shadow_consistency = None
            try:
                import os as _os
                if _os.getenv("ENABLE_SHADOW_SELF_CONSISTENCY", "false").lower() == "true":
                    from src.telemetry.self_consistency import run_shadow_check
                    shadow_consistency = await run_shadow_check(
                        provider_instance=provider_instance,
                        messages=messages,
                        model=model,
                        primary_text=response.content,
                    )
            except Exception as _e:
                # Non-fatal
                shadow_consistency = None

            # Confidence scoring (Phase 1)
            try:
                from src.telemetry.confidence import confidence_scorer
                factors = {
                    "groundedness": groundedness or 0.0,
                    "self_verification": self_ver or 0.0,
                    "style": style_score or 0.0,
                }
                if shadow_consistency is not None:
                    factors["consistency"] = shadow_consistency
                cs = confidence_scorer.evaluate_response_confidence(
                    response=response.content,
                    factors=factors,
                    component="llm",
                )
                # Merge quality scores and confidence into metadata
                if isinstance(response.metadata, dict):
                    qsd = response.metadata.setdefault("quality_scores", {})
                    if shadow_consistency is not None:
                        qsd["consistency"] = float(shadow_consistency)
                    response.metadata["computed_confidence"] = float(cs.overall_score)
                # Update response.confidence conservatively
                try:
                    response.confidence = max(float(getattr(response, "confidence", 0.0) or 0.0), float(cs.overall_score))
                except Exception:
                    pass
            except Exception:
                pass

            await self._log_turn(
                messages=messages,
                response=response,
                phase=phase,
                engagement_id=engagement_id,
                contract_valid=None,  # filled below if contract validation enabled
                groundedness=groundedness,
                self_ver=self_ver,
                style_score=style_score,
                context_ids=[src.get("id") for src in (kwargs.get("sources") or []) if isinstance(src, dict)],
            )

            # Track cost event for dashboard (Phase 3)
            try:
                from src.engine.persistence.llm_cost_repository import get_llm_cost_repository

                cost_repo = get_llm_cost_repository()
                cost_repo.insert_event(
                    engagement_id=engagement_id,
                    phase=phase or "unknown",
                    provider=provider_key,
                    model=model,
                    tokens_input=getattr(response, "prompt_tokens", 0),
                    tokens_output=getattr(response, "completion_tokens", 0),
                    cost_usd=getattr(response, "cost_usd", 0.0),
                    latency_ms=getattr(response, "response_time_ms", 0),
                    reasoning_enabled=call_kwargs.get("reasoning_enabled", False),
                    success=True,
                    request_metadata={
                        "task_type": call_kwargs.get("task_type"),
                        "requires_multi_step": call_kwargs.get("requires_multi_step"),
                        "stakeholder_impact": call_kwargs.get("stakeholder_impact")
                    }
                )
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to track cost event: {e}")

            # Telemetry budget tracking (Phase 1)
            try:
                from src.telemetry.budget import budget_tracker
                tokens_total = getattr(response, "tokens_used", 0)
                if tokens_total == 0:
                    tokens_total = (
                        int(getattr(response, "prompt_tokens", 0))
                        + int(getattr(response, "completion_tokens", 0))
                    )
                budget_tracker.record(
                    tokens=tokens_total,
                    cost_usd=float(getattr(response, "cost_usd", 0.0)),
                    latency_ms=float(getattr(response, "response_time_ms", 0.0)),
                )
                _ = budget_tracker.check_and_record(
                    tokens=tokens_total,
                    cost_usd=float(getattr(response, "cost_usd", 0.0)),
                    latency_ms=float(getattr(response, "response_time_ms", 0.0)),
                )
            except Exception as e:
                self.logger.debug(f"Budget telemetry skipped: {e}")

            # Record the call
            self._record_call(response, provider_key)

            # Emit LLM output to UnifiedContextStream for complete capture
            try:
                cs = get_unified_context_stream()
                LLMEventEmitter(cs).response(
                    provider=provider_key,
                    model=response.model,
                    latency_ms=response.response_time_ms,
                    tokens_used=response.tokens_used,
                    response=response.content,
                    cost_usd=response.cost_usd,
                )
            except Exception:
                pass

            self.logger.info(
                f"✅ {provider} response: {len(response.content)} chars, {response.tokens_used} tokens"
            )

            # ENTERPRISE QUALITY: Output validation pipeline
            # 1. Contract validation (if output_contract was specified)
            if output_contract:
                pre_validation_response = response
                response = self._validate_response_contract(response, output_contract)
                # We treat contract validation pass if content unchanged and/or validation passes silently
                contract_valid = True if response is not None else None
            else:
                contract_valid = None

            # 2. Grounding validation
            sources = kwargs.get("sources")  # Sources from kwargs if provided
            response = self._validate_grounding(response, sources=sources)

            # 3. Self-verification
            query = messages[0].get("content") if messages else None
            response = self._verify_response(response, query=query)

            # Optional: persistent memory write-back (profile/work/summaries)
            try:
                from src.config.feature_flags import get_feature_flags
                from src.core.memory.persistent_memory import get_persistent_memory
                from src.core.unified_context_stream import get_unified_context_stream
                flags = get_feature_flags()
                if await flags.enabled("MEMORY_WRITEBACK_ENABLED", True):
                    cs = get_unified_context_stream()
                    pm = get_persistent_memory()
                    # Simple write-back: store brief summary as working set
                    await pm.write_back(
                        user_id=getattr(cs, "user_id", None),
                        session_id=getattr(cs, "session_id", None),
                        profile_facts=None,
                        working_set_updates=[response.content[:300]],
                        summary=None,
                    )
            except Exception:
                pass

            # Confidence-based escalation (fallback ladder)
            response = await self._escalate_on_low_confidence(
                messages, response, phase=phase, engagement_id=engagement_id, context_data=kwargs.get("context")
            )

            return response

        except Exception as e:
            self.logger.error(f"❌ {provider} call failed: {e}")
            raise

    async def call_with_perplexity_research(
        self, messages: List[Dict], model: str = "deepseek-reasoner", **kwargs
    ) -> LLMResponse:
        """
        Call LLM with native Perplexity research function calling enabled

        This method automatically includes Perplexity function schemas and enables
        native function calling for research tasks.
        """
        from .perplexity_function_schemas import get_perplexity_function_schemas

        # Get Perplexity function schemas
        functions = get_perplexity_function_schemas()

        self.logger.info(
            f"🔬 Enabling native Perplexity research with {len(functions)} functions"
        )

        return await self.call_llm(
            messages=messages, model=model, functions=functions, **kwargs
        )

    async def call_with_json_enforcement(
        self, messages: List[Dict], model: str = "deepseek-chat", **kwargs
    ) -> LLMResponse:
        """
        Call LLM with guaranteed JSON output enforcement

        Uses DeepSeek V3.1's native JSON response format to ensure valid JSON output
        """
        response_format = {"type": "json_object"}

        self.logger.info("📋 Enabling guaranteed JSON output enforcement")

        return await self.call_llm(
            messages=messages, model=model, response_format=response_format, **kwargs
        )

    async def execute_function_calls_if_present(
        self, response: LLMResponse
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Execute function calls if present in LLM response

        Args:
            response: LLM response that may contain function calls

        Returns:
            List of function execution results if function calls were present, None otherwise
        """
        if not response.metadata:
            return None

        function_call = response.metadata.get("function_call")
        tool_calls = response.metadata.get("tool_calls")

        if not function_call and not tool_calls:
            return None

        from .function_call_executor import get_function_call_executor

        executor = get_function_call_executor()

        self.logger.info("🔧 Executing function calls from DeepSeek response")

        if function_call:
            # Single function call
            result = await executor.execute_function_call(function_call)
            return [result]
        elif tool_calls:
            # Multiple function calls
            results = await executor.execute_multiple_function_calls(tool_calls)
            return results

        return None

    async def call_best_available_provider(
        self,
        messages: List[Dict],
        phase: Optional[str] = None,
        engagement_id: Optional[str] = None,
        context_data: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> LLMResponse:
        """
        REFACTORED: Grade D (23) → Grade B (≤10) complexity

        Call best available provider using Strategy Pattern orchestrator
        with centralized provider policy (provider_policy.py).

        Pattern: Delegation to ProviderChainOrchestrator service
        Complexity Target: Grade B (≤10)
        """
        from src.engine.services.llm.provider_selection_service import (
            get_provider_chain_orchestrator,
            ProviderCallContext,
        )
        from src.engine.services.llm.provider_policy import get_provider_chain_for_phase

        # Get provider preference from centralized policy
        provider_preference = get_provider_chain_for_phase(phase or "general")

        # ROUTING_AB: optionally alter chain using A/B routing
        try:
            import os as _os
            if _os.getenv("FF_ROUTING_AB", "false").lower() in ("1", "true", "yes", "on"):
                from src.engine.services.llm.routing_ab import get_routing_ab
                ab = get_routing_ab()
                # Use prompt hash of first user message for arm assignment
                key = (messages[0].get("content", "") if messages else "")
                import hashlib as _hashlib
                prompt_hash = _hashlib.sha256(key.encode()).hexdigest()[:16]
                provider_preference = ab.get_chain(
                    phase=phase or "general", prompt_hash=prompt_hash, policy_chain=provider_preference
                )
        except Exception:
            pass

        # Create call context with policy-driven provider order
        context = ProviderCallContext(
            messages=messages,
            phase=phase,
            engagement_id=engagement_id,
            context_data=context_data,
            model=model,
            use_cache=use_cache and self.cache_enabled,
            kwargs={**kwargs, "provider_preference": provider_preference},
        )

        # Get orchestrator and execute call
        orchestrator = get_provider_chain_orchestrator()
        response = await orchestrator.call_best_available_provider(
            self._providers, context
        )

        # Ensure metadata dict
        if response.metadata is None:
            try:
                response.metadata = {}
            except Exception:
                pass

        # Optional: prompt/schema version tracking (Phase 1)
        try:
            versions = {"model": response.model, "provider": response.provider}
            prompt_name = (kwargs or {}).get("prompt_name")
            prompt_tmpl = (kwargs or {}).get("prompt_template")
            if prompt_name and prompt_tmpl:
                from src.engine.services.prompt_version_registry import get_prompt_version_registry
                preg = get_prompt_version_registry()
                pv = preg.register_prompt(prompt_name, prompt_tmpl)
                preg.bind(prompt_name, response.provider, response.model)
                versions.update({"prompt_name": prompt_name, "prompt_version": pv})
            schema_name = (kwargs or {}).get("schema_name")
            output_schema = (kwargs or {}).get("output_schema")
            if schema_name and isinstance(output_schema, dict):
                from src.engine.services.schema_registry import get_schema_registry
                sreg = get_schema_registry()
                sv = sreg.register_schema(schema_name, output_schema)
                versions.update({"schema_name": schema_name, "schema_version": sv})
            # Attach to metadata
            if isinstance(response.metadata, dict):
                response.metadata.setdefault("versions", {}).update(versions)
        except Exception:
            pass

        # Style + gate + log turn
        try:
            style_score = self._score_style(response.content) or None
        except Exception:
            style_score = None
        # Evaluate style gate
        try:
            from src.telemetry.style_gate import evaluate as _eval_style
            action = _eval_style(style_score, phase=phase)
            if isinstance(response.metadata, dict):
                response.metadata.setdefault("style_gate", {})["action"] = action
                response.metadata["style_gate"] = {**response.metadata.get("style_gate", {}), "score": style_score}
            if action == "block":
                response.content = "Output blocked by style policy"
                try:
                    response.confidence = min(float(getattr(response, "confidence", 0.0) or 0.0), 0.2)
                except Exception:
                    pass
        except Exception:
            pass
        await self._log_turn(
            messages=messages,
            response=response,
            phase=phase,
            engagement_id=engagement_id,
            contract_valid=None,
            groundedness=None,
            self_ver=None,
            style_score=style_score,
            context_ids=[src.get("id") for src in (context_data or {}).get("sources", []) if isinstance(src, dict)] if context_data else None,
        )

        # Confidence-based escalation (fallback ladder)
        response = await self._escalate_on_low_confidence(
            messages, response, phase=phase, engagement_id=engagement_id, context_data=context_data
        )

        # Record the successful call (maintain existing behavior)
        self._record_call(response, response.provider)
        if response.provider in self._providers:
            provider = self._providers[response.provider]
            provider.record_call(response, phase or "unknown")

        # Emit LLM output to UnifiedContextStream
        try:
            cs = get_unified_context_stream()
            LLMEventEmitter(cs).response(
                provider=response.provider,
                model=response.model,
                latency_ms=response.response_time_ms,
                tokens_used=response.tokens_used,
                response=response.content,
                cost_usd=response.cost_usd,
                phase=phase or "unknown",
            )
        except Exception:
            pass

        # Telemetry budget tracking (Phase 1) for best-available path
        try:
            from src.telemetry.budget import budget_tracker
            budget_tracker.record(
                tokens=int(getattr(response, "tokens_used", 0)),
                cost_usd=float(getattr(response, "cost_usd", 0.0)),
                latency_ms=float(getattr(response, "response_time_ms", 0.0)),
            )
            _ = budget_tracker.check_and_record(
                tokens=int(getattr(response, "tokens_used", 0)),
                cost_usd=float(getattr(response, "cost_usd", 0.0)),
                latency_ms=float(getattr(response, "response_time_ms", 0.0)),
            )
        except Exception as _:
            pass

        return response

    def _redact_pii_from_messages(self, messages: List[Dict]) -> List[Dict]:
        """
        ENTERPRISE SECURITY (Phase 6): Redact PII from messages before sending to LLM.

        Args:
            messages: List of message dicts

        Returns:
            Messages with PII redacted
        """
        redacted_messages = []

        for msg in messages:
            redacted_msg = msg.copy()

            if "content" in msg and isinstance(msg["content"], str):
                result = self.pii_engine.redact(msg["content"])
                redacted_msg["content"] = result.redacted_text

                if result.redaction_count > 0:
                    self.logger.warning(
                        f"🔒 PII REDACTED from message: {result.redaction_count} instances"
                    )

            redacted_messages.append(redacted_msg)

        return redacted_messages

    def _apply_sensitivity_routing(
        self,
        messages: List[Dict],
        provider: str,
        sensitivity_override: Optional[str],
        context: Dict[str, Any]
    ) -> str:
        """
        ENTERPRISE SECURITY (Phase 6): Apply sensitivity-based provider routing.

        Args:
            messages: Message list
            provider: Requested provider
            sensitivity_override: Manual sensitivity level
            context: Request context

        Returns:
            Allowed provider (may override requested provider)
        """
        # Extract content for sensitivity detection
        content = " ".join(
            msg.get("content", "") for msg in messages if isinstance(msg.get("content"), str)
        )

        # Make routing decision
        decision = self.sensitivity_router.route(
            content=content,
            context=context,
            sensitivity_override=sensitivity_override,
            available_providers=list(self._providers.keys())
        )

        # If requested provider not allowed, use first allowed provider
        if provider not in decision.allowed_providers:
            original_provider = provider
            provider = decision.allowed_providers[0] if decision.allowed_providers else provider

            self.logger.warning(
                f"🔐 SENSITIVITY ROUTING: {original_provider} → {provider} "
                f"(level={decision.sensitivity_level.value}, "
                f"reason={decision.reasons[0] if decision.reasons else 'policy'})"
            )

            # Log glass-box event
            try:
                from src.core.unified_context_stream import get_unified_context_stream, ContextEventType
                cs = get_unified_context_stream()
                cs.add_event(ContextEventType.ERROR, {  # Using ERROR as proxy for security event
                    "event_type": "sensitivity_routing_override",
                    "original_provider": original_provider,
                    "routed_provider": provider,
                    "sensitivity_level": decision.sensitivity_level.value,
                    "reasons": decision.reasons,
                    "restrictions": decision.restrictions
                })
            except Exception:
                pass

        return provider

    def _check_injection_firewall(self, messages: List[Dict]) -> List[Dict]:
        """
        ENTERPRISE SECURITY (Feature 3/6): Check messages for injection attempts.

        Args:
            messages: Original message list

        Returns:
            Sanitized message list (or raises InjectionAttemptError if blocked)

        Raises:
            InjectionAttemptError: If HIGH/CRITICAL injection detected and action is BLOCK
        """
        if not self.injection_firewall_enabled or not self.injection_firewall:
            return messages

        try:
            from src.engine.security.injection_firewall import (
                FirewallAction,
                InjectionAttemptError,
            )

            sanitized_messages = []

            for msg in messages:
                # Only check user messages (not system/assistant)
                if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                    result = self.injection_firewall.check_input(msg["content"])

                    if result.action_taken == FirewallAction.BLOCK:
                        # BLOCK: Raise exception
                        self.logger.error(
                            f"🚫 INJECTION BLOCKED: {len(result.detections)} pattern(s) detected"
                        )
                        raise InjectionAttemptError(
                            f"Injection attempt blocked: {result.detections[0].pattern_name}",
                            result.detections,
                        )

                    elif result.action_taken == FirewallAction.SANITIZE:
                        # SANITIZE: Use cleaned content
                        sanitized_msg = msg.copy()
                        sanitized_msg["content"] = result.sanitized_input
                        sanitized_messages.append(sanitized_msg)

                        self.logger.warning(
                            f"🧹 INPUT SANITIZED: {len(result.detections)} pattern(s) removed"
                        )

                    else:
                        # LOG_ONLY: Pass through
                        sanitized_messages.append(msg)
                        if result.detections:
                            self.logger.info(
                                f"ℹ️ LOW-RISK PATTERNS: {len(result.detections)} detected (allowed)"
                            )

                else:
                    # System/assistant messages pass through
                    sanitized_messages.append(msg)

            return sanitized_messages

        except InjectionAttemptError:
            raise  # Re-raise blocking errors
        except Exception as e:
            self.logger.warning(f"⚠️ Injection firewall error: {e}")
            return messages  # Fail open on errors

    def _append_contract_prompt(
        self, messages: List[Dict], contract_name: str
    ) -> List[Dict]:
        """
        ENTERPRISE ROBUSTNESS (Feature 2/6): Append contract prompt to messages.

        Args:
            messages: Original message list
            contract_name: Contract to use (analysis/structured_query/classification)

        Returns:
            Modified message list with contract prompt appended to system message
        """
        if not self.output_contracts_enabled:
            return messages

        try:
            contract_prompt = self._get_contract_prompt(contract_name)
            if not contract_prompt:
                self.logger.warning(
                    f"⚠️ Contract '{contract_name}' not found, skipping"
                )
                return messages

            # Find system message or create one
            modified_messages = messages.copy()
            system_msg_idx = None

            for idx, msg in enumerate(modified_messages):
                if msg.get("role") == "system":
                    system_msg_idx = idx
                    break

            if system_msg_idx is not None:
                # Append to existing system message
                modified_messages[system_msg_idx]["content"] += (
                    "\n\n" + contract_prompt
                )
            else:
                # Insert new system message at the beginning
                modified_messages.insert(
                    0, {"role": "system", "content": contract_prompt}
                )

            self.logger.info(
                f"✅ Contract prompt appended: {contract_name}"
            )
            return modified_messages

        except Exception as e:
            self.logger.warning(
                f"⚠️ Failed to append contract prompt: {e}"
            )
            return messages

    def _validate_response_contract(
        self, response: LLMResponse, contract_name: str
    ) -> LLMResponse:
        """
        ENTERPRISE ROBUSTNESS (Feature 2/6): Validate LLM response against contract.

        Args:
            response: Raw LLM response
            contract_name: Contract to validate against

        Returns:
            Original response (validation logged but not blocking)
        """
        if not self.output_contracts_enabled or not self._is_contract_validation_enabled():
            return response

        try:
            contract = self._get_contract(contract_name)
            if not contract:
                self.logger.warning(
                    f"⚠️ Contract '{contract_name}' not found for validation"
                )
                return response

            validation = self._validate_contract(
                response.content, contract, contract_name
            )

            if validation.is_valid:
                self.logger.info(
                    f"✅ Response validated against contract: {contract_name}"
                )

                # Log glass-box event
                try:
                    from src.core.unified_context_stream import (
                        get_unified_context_stream,
                        ContextEventType,
                    )

                    cs = get_unified_context_stream()
                    cs.add_event(
                        ContextEventType.LLM_CALL_COMPLETE,
                        {
                            "event_type": "contract_validation_success",
                            "contract_name": contract_name,
                            "provider": response.provider,
                            "model": response.model,
                        },
                    )
                except Exception:
                    pass

            else:
                self.logger.error(
                    f"❌ Contract validation FAILED: {contract_name} "
                    f"({len(validation.violations)} violations)"
                )

                # Log violations
                for violation in validation.violations[:3]:  # First 3
                    self.logger.error(
                        f"   - {violation.violation_type.value}: "
                        f"{violation.field_path} - {violation.error_message}"
                    )

                # Log glass-box event
                try:
                    from src.core.unified_context_stream import (
                        get_unified_context_stream,
                        ContextEventType,
                    )

                    cs = get_unified_context_stream()
                    cs.add_event(
                        ContextEventType.ERROR,
                        {
                            "event_type": "contract_validation_failure",
                            "contract_name": contract_name,
                            "provider": response.provider,
                            "model": response.model,
                            "violation_count": len(validation.violations),
                            "violations": [
                                {
                                    "type": v.violation_type.value,
                                    "field": v.field_path,
                                    "error": v.error_message,
                                }
                                for v in validation.violations[:5]
                            ],
                        },
                    )
                except Exception:
                    pass

                # NON-BLOCKING: Return original response even on validation failure
                # This prevents cascading failures while we're still testing contracts

        except Exception as e:
            self.logger.warning(
                f"⚠️ Contract validation error: {e}"
            )

        return response

    def _merge_quality_scores(self, response: LLMResponse, scores: Dict[str, float]) -> None:
        if response.metadata is None:
            response.metadata = {}
        qs = response.metadata.get("quality_scores") or {}
        qs.update({k: float(v) for k, v in scores.items() if v is not None})
        response.metadata["quality_scores"] = qs
        # Also store on client for later aggregation
        self._runtime_quality_scores.update(qs)

    def get_runtime_quality_scores(self) -> Dict[str, float]:
        return dict(self._runtime_quality_scores)

    def _validate_grounding(
        self,
        response: LLMResponse,
        sources: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """
        ENTERPRISE QUALITY (Feature 4/6): Validate response grounding in sources.

        Args:
            response: LLM response to validate
            sources: Optional list of source documents
            context: Optional context metadata

        Returns:
            Original response (grounding logged but not blocking)
        """
        if not self.grounding_contract_enabled or not self.grounding_contract:
            return response

        try:
            result = self.grounding_contract.validate(
                response=response.content,
                sources=sources,
                context=context
            )

            # Merge into quality scores for summary metrics
            try:
                groundedness = max(0.0, min(1.0, float(result.assessment.grounding_ratio)))
                # Simple attribution proxy: normalize citation count up to 5 citations
                attribution = max(0.0, min(1.0, float(result.assessment.citation_count) / 5.0))
                self._merge_quality_scores(
                    response,
                    {
                        "groundedness": groundedness,
                        "attribution": attribution,
                    },
                )
            except Exception:
                pass

            if result.is_grounded:
                self.logger.info(
                    f"✅ Response well-grounded: {result.assessment.grounding_level.value} "
                    f"({result.assessment.citation_count} citations)"
                )
            else:
                self.logger.warning(
                    f"⚠️ Response grounding issues: {result.assessment.grounding_level.value} "
                    f"(ratio={result.assessment.grounding_ratio:.1%})"
                )

                # Log recommendations
                if result.assessment.recommendations:
                    self.logger.info(
                        f"💡 Recommendations: {result.assessment.recommendations[0]}"
                    )

        except Exception as e:
            self.logger.warning(f"⚠️ Grounding validation error: {e}")

        return response

    def _verify_response(
        self,
        response: LLMResponse,
        query: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """
        ENTERPRISE QUALITY (Feature 5/6): Self-verify response quality.

        Args:
            response: LLM response to verify
            query: Original user query
            context: Optional context metadata

        Returns:
            Original response (verification logged but not blocking)
        """
        if not self.self_verification_enabled or not self.self_verifier:
            return response

        try:
            result = self.self_verifier.verify(
                response=response.content,
                query=query,
                context=context
            )

            # Merge into quality scores for summary metrics
            try:
                self._merge_quality_scores(
                    response,
                    {
                        "self_verification": float(result.overall_quality),
                        "consistency": float(result.consistency_score),
                        # Use consistency as proxy for coherence when available
                        "coherence": float(result.consistency_score),
                    },
                )
            except Exception:
                pass

            if result.passed:
                if result.status.value == "verified":
                    self.logger.info(
                        f"✅ Response verified: quality={result.overall_quality:.1%}"
                    )
                else:
                    self.logger.info(
                        f"✓ Response acceptable: {result.status.value} "
                        f"({len(result.issues)} minor issues, quality={result.overall_quality:.1%})"
                    )
            else:
                self.logger.warning(
                    f"⚠️ Response quality issues: {result.status.value} "
                    f"(quality={result.overall_quality:.1%})"
                )

                # Log recommendations
                if result.recommendations:
                    self.logger.info(
                        f"💡 Recommendation: {result.recommendations[0]}"
                    )

        except Exception as e:
            self.logger.warning(f"⚠️ Self-verification error: {e}")

        return response

    def _record_call(self, response: LLMResponse, provider_name: str):
        """Record a call in the unified audit trail"""
        self._call_recorder.append(
            {
                "timestamp": datetime.now().isoformat(),
                "provider": provider_name,
                "model": response.model,
                "tokens": response.tokens_used,
                "cost": response.cost_usd,
                "response_time_ms": response.response_time_ms,
            }
        )

        # Record in system-wide recorder if available
        try:
            from src.engine.core.system_recorder import record_llm_call

            record_llm_call(
                provider=response.provider,
                model=response.model,
                method="unified_client_call",
                tokens=response.tokens_used,
                cost=response.cost_usd,
                time_ms=response.response_time_ms,
            )
        except ImportError:
            pass  # System recorder not available

    # Cognitive Analysis Methods (delegate to cognitive analyzer)

    async def analyze_problem_structure_with_research(
        self,
        problem_statement: str,
        business_context: Dict[str, Any],
        research_data: Optional[Dict] = None,
        engagement_id: Optional[str] = None,
    ) -> CognitiveAnalysisResult:
        """Analyze problem structure using MECE framework with research enhancement"""
        return await self.cognitive.analyze_problem_structure_with_research(
            problem_statement, business_context, research_data, engagement_id
        )

    async def analyze_problem_structure(
        self,
        problem_statement: str,
        business_context: Dict[str, Any],
        engagement_id: Optional[str] = None,
    ) -> CognitiveAnalysisResult:
        """Analyze problem structure using MECE framework"""
        return await self.cognitive.analyze_problem_structure(
            problem_statement, business_context, engagement_id
        )

    async def generate_hypotheses(
        self,
        problem_statement: str,
        research_data: Optional[Dict] = None,
        previous_analysis: Optional[str] = None,
        **kwargs,
    ) -> CognitiveAnalysisResult:
        """Generate testable hypotheses with research integration"""
        return await self.cognitive.generate_hypotheses(
            problem_statement, research_data, previous_analysis, **kwargs
        )

    async def execute_analysis(
        self,
        problem_statement: str,
        hypotheses: List[str] = None,
        research_data: Optional[Dict] = None,
        accumulated_context: Optional[Dict] = None,
        **kwargs,
    ) -> CognitiveAnalysisResult:
        """Execute multi-criteria analysis with dynamic framework selection"""
        return await self.cognitive.execute_analysis(
            problem_statement, hypotheses, research_data, accumulated_context, **kwargs
        )

    async def generate_analysis(
        self, prompt: str, context: Optional[Dict] = None, **kwargs
    ) -> CognitiveAnalysisResult:
        """Generic analysis method for Enhanced LLM Manager compatibility"""
        self.logger.info(f"🤖 Generic analysis call: {prompt[:100]}...")

        # Create messages from prompt
        messages = [{"role": "user", "content": prompt}]

        try:
            # Use best available provider
            response = await self.call_best_available_provider(messages=messages)

            # Return in CognitiveAnalysisResult format
            return CognitiveAnalysisResult(
                mental_models_selected=[response.model],
                reasoning_description=response.content[:200] + "...",
                key_insights=["Generated analysis using " + response.provider],
                confidence_score=response.confidence,
                research_requirements=[],
                raw_response=response.content,
                tokens_used=response.tokens_used,
                cost_usd=response.cost_usd,
                response_time_ms=response.response_time_ms,
            )

        except Exception as e:
            self.logger.error(f"❌ Generic analysis failed: {e}")
            # Return fallback result
            return CognitiveAnalysisResult(
                mental_models_selected=["fallback"],
                reasoning_description=f"Analysis failed: {str(e)}. Fallback applied.",
                key_insights=["Fallback analysis applied"],
                confidence_score=0.3,
                research_requirements=[],
                raw_response="",
                tokens_used=0,
                cost_usd=0.0,
                response_time_ms=0,
            )

    async def synthesize_deliverable(
        self,
        comprehensive_context: Dict,
        research_data: Optional[Dict] = None,
        **kwargs,
    ) -> CognitiveAnalysisResult:
        """Synthesize final deliverable using pyramid principle"""
        return await self.cognitive.synthesize_deliverable(
            comprehensive_context, research_data, **kwargs
        )

    # Utility Methods

    def get_call_history(self) -> List[Dict]:
        """Get unified call history for audit trail"""
        return self._call_recorder.copy()

    def get_total_cost(self) -> float:
        """Get total cost across all providers"""
        total = sum(call.get("cost", 0) for call in self._call_recorder)

        # Add provider-specific costs
        for provider in self._providers.values():
            total += provider.get_total_cost()

        return total

    def get_provider_stats(self) -> Dict[str, Dict]:
        """Get detailed statistics for each provider"""
        stats = {}

        for name, provider in self._providers.items():
            history = provider.get_call_history()
            stats[name] = {
                "total_calls": len(history),
                "total_cost": provider.get_total_cost(),
                "total_tokens": sum(call.get("tokens", 0) for call in history),
                "average_response_time": (
                    sum(call.get("response_time_ms", 0) for call in history)
                    / len(history)
                    if history
                    else 0
                ),
                "available_models": provider.get_available_models(),
            }

        return stats

    # Intelligent Caching Methods

    def _generate_cache_key(
        self,
        messages: List[Dict],
        phase: Optional[str],
        context_data: Optional[Dict],
        model: Optional[str],
        **kwargs,
    ) -> str:
        """Generate deterministic cache key for LLM requests"""
        import json
        import hashlib

        # Create normalized request for hashing
        cache_data = {
            "messages": messages,
            "phase": phase,
            "model": model,
            # Include key parameters that affect output
            "temperature": kwargs.get("temperature", 0.3),
            "max_tokens": kwargs.get("max_tokens", 2000),
            # Include context data but exclude engagement_id for cross-engagement caching
            "requires_reasoning": (
                context_data.get("requires_reasoning", False) if context_data else False
            ),
        }

        # Generate hash
        json_str = json.dumps(cache_data, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    async def _get_cached_response(
        self,
        messages: List[Dict],
        phase: Optional[str],
        context_data: Optional[Dict],
        model: Optional[str],
        **kwargs,
    ) -> Optional[LLMResponse]:
        """Attempt to get cached response"""
        try:
            cache_key = self._generate_cache_key(
                messages, phase, context_data, model, **kwargs
            )

            # Try different caches based on likely provider
            if model == "deepseek-reasoner":
                cached = await self.deepseek_reasoning_cache.get(cache_key)
                if cached:
                    self.logger.info(
                        f"🎯 Cache HIT: DeepSeek Reasoning ({cache_key[:8]})"
                    )
                    return cached
            elif model and "deepseek" in model:
                cached = await self.deepseek_cache.get(cache_key)
                if cached:
                    self.logger.info(f"🎯 Cache HIT: DeepSeek ({cache_key[:8]})")
                    return cached
            elif model and "claude" in model:
                cached = await self.claude_cache.get(cache_key)
                if cached:
                    self.logger.info(f"🎯 Cache HIT: Claude ({cache_key[:8]})")
                    return cached

            # Try all caches if model is not specified
            if not model:
                for cache_name, cache in [
                    ("Claude", self.claude_cache),
                    ("DeepSeek", self.deepseek_cache),
                    ("DeepSeek Reasoning", self.deepseek_reasoning_cache),
                ]:
                    cached = await cache.get(cache_key)
                    if cached:
                        self.logger.info(
                            f"🎯 Cache HIT: {cache_name} ({cache_key[:8]})"
                        )
                        return cached

            return None

        except Exception as e:
            self.logger.warning(f"⚠️ Cache get error: {e}")
            return None

    async def _cache_response(
        self,
        messages: List[Dict],
        response: LLMResponse,
        provider_name: str,
        model: Optional[str],
        phase: Optional[str],
        context_data: Optional[Dict],
        **kwargs,
    ) -> None:
        """Cache successful response with intelligent routing"""
        try:
            cache_key = self._generate_cache_key(
                messages, phase, context_data, model, **kwargs
            )

            # Select appropriate cache based on provider and model
            if provider_name == "deepseek" and model == "deepseek-reasoner":
                await self.deepseek_reasoning_cache.set(cache_key, response)
                self.logger.debug(
                    f"💾 Cached DeepSeek Reasoning response ({cache_key[:8]})"
                )
            elif provider_name == "deepseek":
                await self.deepseek_cache.set(cache_key, response)
                self.logger.debug(f"💾 Cached DeepSeek response ({cache_key[:8]})")
            elif provider_name == "anthropic":
                await self.claude_cache.set(cache_key, response)
                self.logger.debug(f"💾 Cached Claude response ({cache_key[:8]})")
            else:
                # Default to DeepSeek cache for other providers
                await self.deepseek_cache.set(cache_key, response)
                self.logger.debug(
                    f"💾 Cached {provider_name} response in default cache ({cache_key[:8]})"
                )

        except Exception as e:
            self.logger.warning(f"⚠️ Cache set error: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        if not self.cache_enabled:
            return {"cache_enabled": False}

        try:
            claude_stats = self.claude_cache.get_metrics()
            deepseek_stats = self.deepseek_cache.get_metrics()
            reasoning_stats = self.deepseek_reasoning_cache.get_metrics()

            total_requests = (
                claude_stats["total_requests"]
                + deepseek_stats["total_requests"]
                + reasoning_stats["total_requests"]
            )

            total_hits = (
                claude_stats["cache_hits"]
                + deepseek_stats["cache_hits"]
                + reasoning_stats["cache_hits"]
            )

            overall_hit_rate = total_hits / total_requests if total_requests > 0 else 0

            return {
                "cache_enabled": True,
                "overall_hit_rate": overall_hit_rate,
                "total_requests": total_requests,
                "total_hits": total_hits,
                "claude_cache": claude_stats,
                "deepseek_cache": deepseek_stats,
                "reasoning_cache": reasoning_stats,
                "total_memory_mb": (
                    claude_stats["memory_usage_mb"]
                    + deepseek_stats["memory_usage_mb"]
                    + reasoning_stats["memory_usage_mb"]
                ),
            }

        except Exception as e:
            self.logger.error(f"Cache stats error: {e}")
            return {"cache_enabled": True, "error": str(e)}

    async def clear_cache(self, provider: Optional[str] = None) -> Dict[str, int]:
        """Clear cache entries, optionally filtered by provider"""
        if not self.cache_enabled:
            return {"error": "Cache not enabled"}

        cleared = {}

        try:
            if provider == "claude" or provider is None:
                cleared["claude"] = await self.claude_cache.clear()

            if provider == "deepseek" or provider is None:
                cleared["deepseek"] = await self.deepseek_cache.clear()
                cleared["deepseek_reasoning"] = (
                    await self.deepseek_reasoning_cache.clear()
                )

            self.logger.info(f"🧹 Cleared cache entries: {cleared}")
            return cleared

        except Exception as e:
            self.logger.error(f"Cache clear error: {e}")
            return {"error": str(e)}


# Global instance for backward compatibility
_unified_client_instance: Optional[UnifiedLLMClient] = None


def get_unified_llm_client() -> UnifiedLLMClient:
    """Get or create global unified LLM client instance"""
    global _unified_client_instance

    if _unified_client_instance is None:
        _unified_client_instance = UnifiedLLMClient()

    return _unified_client_instance
